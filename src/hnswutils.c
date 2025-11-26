#include "postgres.h"

#include <math.h>

#include "access/generic_xlog.h"
#include "catalog/pg_type.h"
#include "catalog/pg_type_d.h"
#include "common/hashfn.h"
#include "fmgr.h"
#include "hnsw.h"
#include "lib/pairingheap.h"
#include "sparsevec.h"
#include "storage/bufmgr.h"
#include "storage/buf_internals.h"
#include "storage/smgr.h"
#include "utils/datum.h"
#include "utils/memdebug.h"
#include "utils/rel.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if PG_VERSION_NUM < 170000
static inline uint64
murmurhash64(uint64 data)
{
	uint64		h = data;

	h ^= h >> 33;
	h *= 0xff51afd7ed558ccd;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53;
	h ^= h >> 33;

	return h;
}
#endif

/* TID hash table */
static uint32
hash_tid(ItemPointerData tid)
{
	union
	{
		uint64		i;
		ItemPointerData tid;
	}			x;

	/* Initialize unused bytes */
	x.i = 0;
	x.tid = tid;

	return murmurhash64(x.i);
}

#define SH_PREFIX		tidhash
#define SH_ELEMENT_TYPE	TidHashEntry
#define SH_KEY_TYPE		ItemPointerData
#define	SH_KEY			tid
#define SH_HASH_KEY(tb, key)	hash_tid(key)
#define SH_EQUAL(tb, a, b)		ItemPointerEquals(&a, &b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Pointer hash table */
static uint32
hash_pointer(uintptr_t ptr)
{
#if SIZEOF_VOID_P == 8
	return murmurhash64((uint64) ptr);
#else
	return murmurhash32((uint32) ptr);
#endif
}

#define SH_PREFIX		pointerhash
#define SH_ELEMENT_TYPE	PointerHashEntry
#define SH_KEY_TYPE		uintptr_t
#define	SH_KEY			ptr
#define SH_HASH_KEY(tb, key)	hash_pointer(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Offset hash table */
static uint32
hash_offset(Size offset)
{
#if SIZEOF_SIZE_T == 8
	return murmurhash64((uint64) offset);
#else
	return murmurhash32((uint32) offset);
#endif
}

#define SH_PREFIX		offsethash
#define SH_ELEMENT_TYPE	OffsetHashEntry
#define SH_KEY_TYPE		Size
#define	SH_KEY			offset
#define SH_HASH_KEY(tb, key)	hash_offset(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/*
 * Get the max number of connections in an upper layer for each element in the index
 */
int
HnswGetM(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->m;

	return HNSW_DEFAULT_M;
}

/*
 * Get the size of the dynamic candidate list in the index
 */
int
HnswGetEfConstruction(Relation index)
{
	HnswOptions *opts = (HnswOptions *) index->rd_options;

	if (opts)
		return opts->efConstruction;

	return HNSW_DEFAULT_EF_CONSTRUCTION;
}

/*
 * Get proc
 */
FmgrInfo *
HnswOptionalProcInfo(Relation index, uint16 procnum)
{
	if (!OidIsValid(index_getprocid(index, 1, procnum)))
		return NULL;

	return index_getprocinfo(index, 1, procnum);
}

/*
 * Init support functions
 */
void
HnswInitSupport(HnswSupport * support, Relation index)
{
	support->procinfo = index_getprocinfo(index, 1, HNSW_DISTANCE_PROC);
	support->collation = index->rd_indcollation[0];
	support->normprocinfo = HnswOptionalProcInfo(index, HNSW_NORM_PROC);
}

/*
 * Normalize value
 */
Datum
HnswNormValue(const HnswTypeInfo * typeInfo, Oid collation, Datum value)
{
	return DirectFunctionCall1Coll(typeInfo->normalize, collation, value);
}

/*
 * Check if non-zero norm
 */
bool
HnswCheckNorm(HnswSupport * support, Datum value)
{
	return DatumGetFloat8(FunctionCall1Coll(support->normprocinfo, support->collation, value)) > 0;
}

/*
 * New buffer
 */
Buffer
HnswNewBuffer(Relation index, ForkNumber forkNum)
{
	Buffer		buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	return buf;
}

/*
 * Init page
 */
void
HnswInitPage(Buffer buf, Page page)
{
	PageInit(page, BufferGetPageSize(buf), sizeof(HnswPageOpaqueData));
	HnswPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
	HnswPageGetOpaque(page)->page_id = HNSW_PAGE_ID;
}

/*
 * Allocate a neighbor array
 */
HnswNeighborArray *
HnswInitNeighborArray(int lm, HnswAllocator * allocator)
{
	HnswNeighborArray *a = HnswAlloc(allocator, HNSW_NEIGHBOR_ARRAY_SIZE(lm));

	a->length = 0;
	a->closerSet = false;
	return a;
}

/*
 * Allocate neighbors
 */
void
HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator * allocator)
{
	int			level = element->level;
	HnswNeighborArrayPtr *neighborList = (HnswNeighborArrayPtr *) HnswAlloc(allocator, sizeof(HnswNeighborArrayPtr) * (level + 1));

	HnswPtrStore(base, element->neighbors, neighborList);

	for (int lc = 0; lc <= level; lc++)
		HnswPtrStore(base, neighborList[lc], HnswInitNeighborArray(HnswGetLayerM(m, lc), allocator));
}

/*
 * Allocate memory from the allocator
 */
void *
HnswAlloc(HnswAllocator * allocator, Size size)
{
	if (allocator)
		return (*(allocator)->alloc) (size, (allocator)->state);

	return palloc(size);
}

/*
 * Allocate an element
 */
HnswElement
HnswInitElement(char *base, ItemPointer heaptid, int m, double ml, int maxLevel, HnswAllocator * allocator)
{
	HnswElement element = HnswAlloc(allocator, sizeof(HnswElementData));

	int			level = (int) (-log(RandomDouble()) * ml);

	/* Cap level */
	if (level > maxLevel)
		level = maxLevel;

	element->heaptidsLength = 0;
	HnswAddHeapTid(element, heaptid);

	element->level = level;
	element->deleted = 0;
	/* Start at one to make it easier to find issues */
	element->version = 1;

	HnswInitNeighbors(base, element, m, allocator);

	HnswPtrStore(base, element->value, (Pointer) NULL);

	return element;
}

/*
 * Add a heap TID to an element
 */
void
HnswAddHeapTid(HnswElement element, ItemPointer heaptid)
{
	element->heaptids[element->heaptidsLength++] = *heaptid;
}

/*
 * Allocate an element from block and offset numbers
 */
HnswElement
HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno)
{
	HnswElement element = palloc(sizeof(HnswElementData));
	char	   *base = NULL;

	element->blkno = blkno;
	element->offno = offno;
	HnswPtrStore(base, element->neighbors, (HnswNeighborArrayPtr *) NULL);
	HnswPtrStore(base, element->value, (Pointer) NULL);
	return element;
}

/*
 * Get the metapage info
 */
void
HnswGetMetaPageInfo(Relation index, int *m, HnswElement * entryPoint)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (unlikely(metap->magicNumber != HNSW_MAGIC_NUMBER))
		elog(ERROR, "hnsw index is not valid");

	if (m != NULL)
		*m = metap->m;

	if (entryPoint != NULL)
	{
		if (BlockNumberIsValid(metap->entryBlkno))
		{
			*entryPoint = HnswInitElementFromBlock(metap->entryBlkno, metap->entryOffno);
			(*entryPoint)->level = metap->entryLevel;
		}
		else
			*entryPoint = NULL;
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Get the entry point
 */
HnswElement
HnswGetEntryPoint(Relation index)
{
	HnswElement entryPoint;

	HnswGetMetaPageInfo(index, NULL, &entryPoint);

	return entryPoint;
}

/*
 * Update the metapage info
 */
static void
HnswUpdateMetaPageInfo(Page page, int updateEntry, HnswElement entryPoint, BlockNumber insertPage)
{
	HnswMetaPage metap = HnswPageGetMeta(page);

	if (updateEntry)
	{
		if (entryPoint == NULL)
		{
			metap->entryBlkno = InvalidBlockNumber;
			metap->entryOffno = InvalidOffsetNumber;
			metap->entryLevel = -1;
		}
		else if (entryPoint->level > metap->entryLevel || updateEntry == HNSW_UPDATE_ENTRY_ALWAYS)
		{
			metap->entryBlkno = entryPoint->blkno;
			metap->entryOffno = entryPoint->offno;
			metap->entryLevel = entryPoint->level;
		}
	}

	if (BlockNumberIsValid(insertPage))
		metap->insertPage = insertPage;
}

/*
 * Update the metapage
 */
void
HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;

	buf = ReadBufferExtended(index, forkNum, HNSW_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	HnswUpdateMetaPageInfo(page, updateEntry, entryPoint, insertPage);

	if (building)
		MarkBufferDirty(buf);
	else
		GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
}

/*
 * Form index value
 */
bool
HnswFormIndexValue(Datum *out, Datum *values, bool *isnull, const HnswTypeInfo * typeInfo, HnswSupport * support)
{
	/* Detoast once for all calls */
	Datum		value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));

	/* Check value */
	if (typeInfo->checkValue != NULL)
		typeInfo->checkValue(DatumGetPointer(value));

	/* Normalize if needed */
	if (support->normprocinfo != NULL)
	{
		if (!HnswCheckNorm(support, value))
			return false;

		value = HnswNormValue(typeInfo, support->collation, value);
	}

	*out = value;

	return true;
}

/*
 * Set element tuple, except for neighbor info
 */
void
HnswSetElementTuple(char *base, HnswElementTuple etup, HnswElement element)
{
	Pointer		valuePtr = HnswPtrAccess(base, element->value);

	etup->type = HNSW_ELEMENT_TUPLE_TYPE;
	etup->level = element->level;
	etup->deleted = 0;
	etup->version = element->version;
	for (int i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (i < element->heaptidsLength)
			etup->heaptids[i] = element->heaptids[i];
		else
			ItemPointerSetInvalid(&etup->heaptids[i]);
	}
	memcpy(&etup->data, valuePtr, VARSIZE_ANY(valuePtr));
}

/*
 * Set neighbor tuple
 */
void
HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m)
{
	int			idx = 0;

	ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;

	for (int lc = e->level; lc >= 0; lc--)
	{
		HnswNeighborArray *neighbors = HnswGetNeighbors(base, e, lc);
		int			lm = HnswGetLayerM(m, lc);

		for (int i = 0; i < lm; i++)
		{
			ItemPointer indextid = &ntup->indextids[idx++];

			if (i < neighbors->length)
			{
				HnswCandidate *hc = &neighbors->items[i];
				HnswElement hce = HnswPtrAccess(base, hc->element);

				ItemPointerSet(indextid, hce->blkno, hce->offno);
			}
			else
				ItemPointerSetInvalid(indextid);
		}
	}

	ntup->count = idx;
	ntup->version = e->version;
}

/*
 * Load an element from a tuple
 */
void
HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec)
{
	element->level = etup->level;
	element->deleted = etup->deleted;
	element->version = etup->version;
	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	element->heaptidsLength = 0;

	if (loadHeaptids)
	{
		for (int i = 0; i < HNSW_HEAPTIDS; i++)
		{
			/* Can stop at first invalid */
			if (!ItemPointerIsValid(&etup->heaptids[i]))
				break;

			HnswAddHeapTid(element, &etup->heaptids[i]);
		}
	}

	if (loadVec)
	{
		char	   *base = NULL;
		Datum		value = datumCopy(PointerGetDatum(&etup->data), false, -1);

		HnswPtrStore(base, element->value, DatumGetPointer(value));
	}
}

/*
 * Calculate the distance between values
 */
static inline double
HnswGetDistance(Datum a, Datum b, HnswSupport * support)
{
	return DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation, a, b));
}

static float
VectorL2SquaredDistance(int dim, float *ax, float *bx)
{
	float		distance = 0.0;

	/* Auto-vectorized */
	for (int i = 0; i < dim; i++)
	{
		float		diff = ax[i] - bx[i];

		distance += diff * diff;
	}

	return distance;
}

/*
 * Calculate L2 squared distance optimistically - assumes 'a' is stable but 'b' may be modified
 * Directly computes distance without copying by validating varlena header first
 */
__attribute__((target_clones("default", "fma"))) static inline double
HnswGetDistanceOptimistic(Datum a, Vector *b_ptr, Page page, HnswSupport * support)
{
	Size		varlena_len;
	char	   *page_start;
	char	   *page_end;
	char	   *b_end;
	int16		dim;
	Vector	   *a_vec;
	float		distance;
	Vector      b_header = *b_ptr;
	/* 
	 * Read varlena header with memory barrier to ensure consistency
	 * Note: b_ptr->vl_len_ could be corrupted by concurrent writers
	 */
	pg_read_barrier();
	varlena_len = VARSIZE_ANY(&b_header);
	
	/* Calculate page bounds */
	page_start = (char *) page;
	page_end = page_start + BLCKSZ;
	b_end = ((char *) b_ptr) + varlena_len;
	
	/* 
	 * Validate varlena length is within page bounds
	 * If corrupted, validation will fail - return dummy value
	 */
	if (varlena_len < VARHDRSZ || 
	    varlena_len > BLCKSZ ||
	    ((char *) b_ptr) < page_start ||
	    ((char *) b_ptr) >= page_end ||
	    b_end > page_end)
	{
		/* Invalid varlena header - validation will fail, return dummy value */
		return 0.0;
	}
	
	/* 
	 * After varlena header validation passes, we can safely read the dimension field
	 * The varlena header is valid, so reading dim is safe
	 */
	pg_read_barrier();
	dim = b_header.dim;
	// /* Get query vector */
	// a_vec = DatumGetVector(a);
	
	// /* 
	//  * Validate dimension matches query dimension
	//  * If corrupted, validation will fail - return dummy value
	//  */
	// if (dim != a_vec->dim || dim <= 0 || dim > VECTOR_MAX_DIM)
	// {
	// 	/* Invalid dimension - validation will fail, return dummy value */
	// 	return 0.0;
	// }
	
	/* 
	 * Now safe to directly compute L2 squared distance
	 * Inline the computation to avoid function call overhead
	 */
	return VectorL2SquaredDistance(dim, ((Vector *) DatumGetPointer(a))->x, b_ptr->x);
}

/*
 * Load element data optimistically from tuple
 * This is like HnswLoadElementFromTuple but makes defensive copies for optimistic reads
 */
static inline void
HnswLoadElementFromTupleOptimistic(HnswElement element, HnswElementTuple etup, 
								   bool loadHeaptids, bool loadVec, Page page)
{
	/* Read basic fields - these are small and atomic on most architectures */
	element->level = etup->level;
	element->deleted = etup->deleted;
	element->version = etup->version;
	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	element->heaptidsLength = 0;
	
	if (loadHeaptids)
	{
		for (int i = 0; i < HNSW_HEAPTIDS; i++)
		{
			/* Can stop at first invalid */
			if (!ItemPointerIsValid(&etup->heaptids[i]))
				break;

			HnswAddHeapTid(element, &etup->heaptids[i]);
		}
	}
	
	if (loadVec)
	{
		char	   *base = NULL;
		Vector	   *vec_ptr = (Vector *) &etup->data;
		Size		varlena_len;
		Vector	   *vec_copy;
		char	   *page_start;
		char	   *page_end;
		char	   *vec_end;
		
		/* Read varlena header with memory barrier */
		pg_read_barrier();
		varlena_len = VARSIZE_ANY(vec_ptr);
		
		/* Calculate page bounds */
		page_start = (char *) page;
		page_end = page_start + BLCKSZ;
		vec_end = ((char *) vec_ptr) + varlena_len;
		
		/* Validate varlena length is within page bounds */
		if (varlena_len < VARHDRSZ || 
		    varlena_len > BLCKSZ ||
		    ((char *) vec_ptr) < page_start ||
		    ((char *) vec_ptr) >= page_end ||
		    vec_end > page_end)
		{
			/* Invalid varlena header - will be caught by validation */
			HnswPtrStore(base, element->value, (Pointer) NULL);
			return;
		}
		
		/* Make a proper copy of the entire varlena datum */
		vec_copy = (Vector *) palloc(varlena_len);
		memcpy(vec_copy, vec_ptr, varlena_len);
		
		HnswPtrStore(base, element->value, (Pointer) vec_copy);
	}
}

/*
 * Load an element and optionally get its distance from q
 */
static void
HnswLoadElementImpl(BlockNumber blkno, OffsetNumber offno, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance, HnswElement * element)
{
	Buffer		buf;
	Page		page;
	HnswElementTuple etup;
	SMgrRelation smgr;
	bool		found;
	uint64		version;
	BufferDesc *bufDesc;

	/*
	 * Try optimistic read path if enabled and buffer manager supports it.
	 * This avoids taking buffer locks for read-only access.
	 */
	if (enable_optimistic_buffer_reads && 
	    ActiveBufMgr->LookupAndOptimisticPin != NULL)
	{
		uint32 buf_id = 0;
		void *entry_ptr = NULL;
		bool validation_passed = false;
		bool was_element_null = (*element == NULL);
		double old_distance = (distance != NULL) ? *distance : 0.0;
		smgr = RelationGetSmgr(index);
		
		/* Use standard optimistic pin with TLS cache optimization */
		bufDesc = ActiveBufMgr->LookupAndOptimisticPin(
			smgr,
			MAIN_FORKNUM,
			blkno,
			&found,
			&version,
			&buf_id,
			&entry_ptr
		);
		
		if (found)
		{
			/* 
			 * CRITICAL: All reads must happen BEFORE validation check.
			 * After validation fails, the page may be evicted or modified.
			 */
			
			/* Read shared_buffer page data optimistically */
			page = BufferGetPage(buf_id + 1);
			//ItemIdData item_id_data = *PageGetItemId(page, offno);
			bool should_load = false;
			// validate item_id_data is wihtin bound of page
			// if (ItemIdGetLength(&item_id_data) <= 0 ||
			//     ItemIdGetOffset(&item_id_data) < SizeOfPageHeaderData ||
			//     (ItemIdGetOffset(&item_id_data) + ItemIdGetLength(&item_id_data)) >= BLCKSZ)
			// {
			// 	/* Invalid item_id_data - will be caught by validation */
			// 	validation_passed = false;
			// } else {
				/* Safe to read the tuple */
				etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));
				/* Calculate distance using safe optimistic method */
				if (distance != NULL)
				{
					if (DatumGetPointer(q->value) == NULL)
						*distance = 0;
					else
						/* Use specialized optimistic distance calculation with page bounds check */
						*distance = HnswGetDistanceOptimistic(
							q->value, 
							(Vector *) &etup->data, 
							page,
							support
						);
						// *distance = HnswGetDistance(
						// 	q->value, 
						// 	PointerGetDatum(&etup->data), 
						// 	support
						// );
				}
				
				/* Check if we should load element data */
				should_load = (distance == NULL || maxDistance == NULL || 
									*distance < *maxDistance);
				
				if (should_load)
				{
					// /* Allocate element if needed */
					if (*element == NULL)
						*element = HnswInitElementFromBlock(blkno, offno);
					
					// // /* Load element data using safe optimistic method with page bounds check */
					// // HnswLoadElementFromTupleOptimistic(*element, etup, true, loadVec, page);
					// // //HnswLoadElementFromTuple(*element, etup, true, loadVec);
					if (loadVec) {
						validation_passed = false; // Force validation failure
					} else {
						HnswLoadElementFromTuple(*element, etup, true, false);
						validation_passed = ActiveBufMgr->LookupAndOptimisticValidate(entry_ptr, version);
					}
					
				} else {
					validation_passed = ActiveBufMgr->LookupAndOptimisticValidate(entry_ptr, version);
				}
			//}
			/*
			 * NOW validate the optimistic read.
			 * This MUST be the last operation before returning.
			 */
			// if (validation_passed == false)
			// 	validation_passed = ActiveBufMgr->LookupAndOptimisticValidate(entry_ptr, version);
			
			if (validation_passed)
			{
				/* SUCCESS: All data read is consistent */
				return;
			}
			
			/*
			 * VALIDATION FAILED
			 * 
			 * All data we read may be corrupted. We must:
			 * 1. Free any allocated memory (element->value from loadVec)
			 * 2. Reset element to pre-read state
			 * 3. Fall through to locked path
			 */
			
			if (should_load && *element != NULL)
			{
				// restore *element
				// No need to free memory because memory context will clean up the allocated objects at once at the end
				if (was_element_null)
				{
					*element = NULL;
				}
			}
			if (distance != NULL)
			{
				/* Restore old distance value */
				*distance = old_distance;
			}
			
			/* Fall through to locked path */
		}
	}
	
locked_path:
	/*
	 * Traditional locked path:
	 * Either optimistic read is disabled, not supported, or validation failed.
	 */
	/* Read vector */
	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));

	Assert(HnswIsElementTuple(etup));

	/* Calculate distance */
	if (distance != NULL)
	{
		if (DatumGetPointer(q->value) == NULL)
			*distance = 0;
		else
			*distance = HnswGetDistance(q->value, PointerGetDatum(&etup->data), support);
	}

	/* Load element */
	if (distance == NULL || maxDistance == NULL || *distance < *maxDistance)
	{
		if (*element == NULL)
			*element = HnswInitElementFromBlock(blkno, offno);

		HnswLoadElementFromTuple(*element, etup, true, loadVec);
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Load an element and optionally get its distance from q
 */
void
HnswLoadElement(HnswElement element, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance)
{
	/* No cached relation_id for this public wrapper */
	HnswLoadElementImpl(element->blkno, element->offno, distance, q, index, support, loadVec, maxDistance, &element);
}

/*
 * Get the distance for an element
 */
static double
GetElementDistance(char *base, HnswElement element, HnswQuery * q, HnswSupport * support)
{
	Datum		value = HnswGetValue(base, element);

	return HnswGetDistance(q->value, value, support);
}

/*
 * Allocate a search candidate
 */
static HnswSearchCandidate *
HnswInitSearchCandidate(char *base, HnswElement element, double distance)
{
	HnswSearchCandidate *sc = palloc(sizeof(HnswSearchCandidate));

	HnswPtrStore(base, sc->element, element);
	sc->distance = distance;
	return sc;
}

/*
 * Create a candidate for the entry point
 */
HnswSearchCandidate *
HnswEntryCandidate(char *base, HnswElement entryPoint, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec)
{
	bool		inMemory = index == NULL;
	double		distance;

	if (inMemory)
		distance = GetElementDistance(base, entryPoint, q, support);
	else
		HnswLoadElement(entryPoint, &distance, q, index, support, loadVec, NULL);

	return HnswInitSearchCandidate(base, entryPoint, distance);
}

/*
 * Compare candidate distances
 */
static int
CompareNearestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(c_node, a)->distance < HnswGetSearchCandidateConst(c_node, b)->distance)
		return 1;

	if (HnswGetSearchCandidateConst(c_node, a)->distance > HnswGetSearchCandidateConst(c_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare discarded candidate distances
 */
static int
CompareNearestDiscardedCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance < HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance > HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare candidate distances
 */
static int
CompareFurthestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (HnswGetSearchCandidateConst(w_node, a)->distance < HnswGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	if (HnswGetSearchCandidateConst(w_node, a)->distance > HnswGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	return 0;
}

/*
 * Init visited
 */
static inline void
InitVisited(char *base, visited_hash * v, bool inMemory, int ef, int m)
{
	if (!inMemory)
		v->tids = tidhash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else if (base != NULL)
		v->offsets = offsethash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else
		v->pointers = pointerhash_create(CurrentMemoryContext, ef * m * 2, NULL);
}

/*
 * Add to visited
 */
static inline void
AddToVisited(char *base, visited_hash * v, HnswElementPtr elementPtr, bool inMemory, bool *found)
{
	if (!inMemory)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);
		ItemPointerData indextid;

		ItemPointerSet(&indextid, element->blkno, element->offno);
		tidhash_insert(v->tids, indextid, found);
	}
	else if (base != NULL)
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		offsethash_insert_hash(v->offsets, HnswPtrOffset(elementPtr), element->hash, found);
	}
	else
	{
		HnswElement element = HnswPtrAccess(base, elementPtr);

		pointerhash_insert_hash(v->pointers, (uintptr_t) HnswPtrPointer(elementPtr), element->hash, found);
	}
}

/*
 * Count element towards ef
 */
static inline bool
CountElement(HnswElement skipElement, HnswElement e)
{
	if (skipElement == NULL)
		return true;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	/* Keep scan-build happy on Mac x86-64 */
	Assert(e);

	return e->heaptidsLength != 0;
}

/*
 * Load unvisited neighbors from memory
 */
static void
HnswLoadUnvisitedFromMemory(char *base, HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength, visited_hash * v, int lc, HnswNeighborArray * localNeighborhood, Size neighborhoodSize)
{
	/* Get the neighborhood at layer lc */
	HnswNeighborArray *neighborhood = HnswGetNeighbors(base, element, lc);

	/* Copy neighborhood to local memory */
	LWLockAcquire(&element->lock, LW_SHARED);
	memcpy(localNeighborhood, neighborhood, neighborhoodSize);
	LWLockRelease(&element->lock);

	*unvisitedLength = 0;

	for (int i = 0; i < localNeighborhood->length; i++)
	{
		HnswCandidate *hc = &localNeighborhood->items[i];
		bool		found;

		AddToVisited(base, v, hc->element, true, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].element = HnswPtrAccess(base, hc->element);
	}
}

/*
 * Load neighbor index TIDs
 */
bool
HnswLoadNeighborTids(HnswElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc)
{
	Buffer		buf;
	Page		page;
	HnswNeighborTuple ntup;
	int			start;
	SMgrRelation smgr;
	bool		found;
	uint64		version;
	uint32		buf_id;
	void		*entry_ptr;

	/*
	 * Try optimistic read path if enabled and buffer manager supports it.
	 * This avoids taking buffer locks for read-only access to neighbor data.
	 */
	if (enable_optimistic_buffer_reads && 
	    ActiveBufMgr->LookupAndOptimisticPin != NULL)
	{
		smgr = RelationGetSmgr(index);
		
		/* Try optimistic pin without locking, get frameId and entry_ptr directly */
		if (ActiveBufMgr->LookupAndOptimisticPin(
			smgr,
			MAIN_FORKNUM,
			element->neighborPage,
			&found,
			&version,
			&buf_id,
			&entry_ptr) != NULL && found)
		{
			/* Read neighbor tuple data optimistically */
			page = BufferGetPage(buf_id + 1);
			ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, element->neighborOffno));

			/*
			 * Check version consistency:
			 * Ensure the neighbor tuple has not been deleted or replaced
			 */
			if (ntup->version != element->version || ntup->count != (element->level + 2) * m)
			{
				/* Data validation failed - fall through to locked path */
			}
			else
			{
				/* Copy neighbor TIDs optimistically */
				start = (element->level - lc) * m;
				memcpy(indextids, ntup->indextids + start, lm * sizeof(ItemPointerData));

				/* Validate the optimistic read using cached entry_ptr - fast path */
				if (ActiveBufMgr->LookupAndOptimisticValidate(entry_ptr, version))
				{
					/* Validation successful - data is consistent */
					return true;
				}

				/*
				 * Validation failed - fall through to locked path.
				 * The neighbor page might have been evicted or modified.
				 */
			}
		}
	}

	/*
	 * Traditional locked path:
	 * Either optimistic read is disabled, not supported, or validation failed.
	 */
	buf = ReadBuffer(index, element->neighborPage);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, element->neighborOffno));

	/*
	 * Ensure the neighbor tuple has not been deleted or replaced between
	 * index scan iterations
	 */
	if (ntup->version != element->version || ntup->count != (element->level + 2) * m)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	/* Copy to minimize lock time */
	start = (element->level - lc) * m;
	memcpy(indextids, ntup->indextids + start, lm * sizeof(ItemPointerData));

	UnlockReleaseBuffer(buf);
	return true;
}

/* Structure for caching prefetch info */
typedef struct HnswPrefetchCache
{
	Page		page;
	ItemId		itemid;
} HnswPrefetchCache;

/*
 * Load unvisited neighbors from disk
 * Returns prefetch_cache and prefetch_count as output parameters for caller to issue vector data prefetch
 */
static void
HnswLoadUnvisitedFromDisk(HnswElement element, HnswUnvisited * unvisited, int *unvisitedLength, visited_hash * v, Relation index, int m, int lm, int lc, HnswPrefetchCache *prefetch_cache_out, int *prefetch_count_out)
{
	ItemPointerData indextids[HNSW_MAX_M * 2];
	SMgrRelation smgr = NULL;
	bool		use_prefetch = false;
	
	/* Arrays for batch lookup */
	BlockNumber batch_blocknums[HNSW_MAX_M];
	uint32		batch_frameids[HNSW_MAX_M];
	bool		batch_found[HNSW_MAX_M];
	int			batch_start = 0;  /* Track where current batch starts in unvisited array */

	*unvisitedLength = 0;
	*prefetch_count_out = 0;

	if (!HnswLoadNeighborTids(element, indextids, index, m, lm, lc))
		return;

	/* Check if optimistic prefetch is available */
	if (enable_optimistic_buffer_reads && 
	    hnsw_enable_prefetch &&
	    ActiveBufMgr->LookupAndOptimisticPinBatch != NULL)
	{
		smgr = RelationGetSmgr(index);
		use_prefetch = true;
	}

	/* First pass: identify unvisited neighbors and issue batch prefetches when we reach BATCH_SIZE */
	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &indextids[i];
		bool		found;

		if (!ItemPointerIsValid(indextid))
			break;

		tidhash_insert(v->tids, *indextid, &found);

		if (!found)
		{
			unvisited[*unvisitedLength].indextid = *indextid;
			
			/* Collect block number for batch prefetch */
			if (use_prefetch)
				batch_blocknums[*unvisitedLength] = ItemPointerGetBlockNumber(indextid);
			
			(*unvisitedLength)++;
		}
	}

	/* Handle remaining items (less than BATCH_SIZE) */
	if (use_prefetch && *unvisitedLength > batch_start)
	{
		int batch_count = *unvisitedLength - batch_start;
		BlockNumber missing_blocks[HNSW_MAX_M];
		int			missing_count = 0;
		
		/* Final batch call for remaining items */
		ActiveBufMgr->LookupAndOptimisticPinBatch(
			smgr,
			MAIN_FORKNUM,
			&batch_blocknums[batch_start],
			batch_count,
			&batch_frameids[batch_start],
			&batch_found[batch_start]);

		/* Process results and issue prefetches with loop unrolling for better MLP */
		int j = batch_start;
		
		/* Process 4 items at a time for better ILP/MLP */
		for (; j + 3 < *unvisitedLength; j += 4)
		{
			Page		prefetch_page0, prefetch_page1, prefetch_page2, prefetch_page3;
			OffsetNumber offno0, offno1, offno2, offno3;
			ItemId		itemid0, itemid1, itemid2, itemid3;
			bool		found0, found1, found2, found3;
			
			/* Load all found flags */
			found0 = batch_found[j];
			found1 = batch_found[j + 1];
			found2 = batch_found[j + 2];
			found3 = batch_found[j + 3];
			
			/* Collect missing blocks for async I/O */
			if (!found0)
				missing_blocks[missing_count++] = batch_blocknums[j];
			if (!found1)
				missing_blocks[missing_count++] = batch_blocknums[j + 1];
			if (!found2)
				missing_blocks[missing_count++] = batch_blocknums[j + 2];
			if (!found3)
				missing_blocks[missing_count++] = batch_blocknums[j + 3];
			
			/* Get pages for all 4 items - these loads can happen in parallel */
			if (found0)
			{
				ItemPointer tid0 = &unvisited[j].indextid;
				prefetch_page0 = BufferGetPage(batch_frameids[j] + 1);
				offno0 = ItemPointerGetOffsetNumber(tid0);
				itemid0 = PageGetItemId(prefetch_page0, offno0);
			}
			if (found1)
			{
				ItemPointer tid1 = &unvisited[j + 1].indextid;
				prefetch_page1 = BufferGetPage(batch_frameids[j + 1] + 1);
				offno1 = ItemPointerGetOffsetNumber(tid1);
				itemid1 = PageGetItemId(prefetch_page1, offno1);
			}
			if (found2)
			{
				ItemPointer tid2 = &unvisited[j + 2].indextid;
				prefetch_page2 = BufferGetPage(batch_frameids[j + 2] + 1);
				offno2 = ItemPointerGetOffsetNumber(tid2);
				itemid2 = PageGetItemId(prefetch_page2, offno2);
			}
			if (found3)
			{
				ItemPointer tid3 = &unvisited[j + 3].indextid;
				prefetch_page3 = BufferGetPage(batch_frameids[j + 3] + 1);
				offno3 = ItemPointerGetOffsetNumber(tid3);
				itemid3 = PageGetItemId(prefetch_page3, offno3);
			}
			
			/* Cache and prefetch - issue all prefetches together for MLP */
			if (found0)
			{
				prefetch_cache_out[*prefetch_count_out].page = prefetch_page0;
				prefetch_cache_out[*prefetch_count_out].itemid = itemid0;
				(*prefetch_count_out)++;
				__builtin_prefetch(itemid0, 0, 3);
			}
			if (found1)
			{
				prefetch_cache_out[*prefetch_count_out].page = prefetch_page1;
				prefetch_cache_out[*prefetch_count_out].itemid = itemid1;
				(*prefetch_count_out)++;
				__builtin_prefetch(itemid1, 0, 3);
			}
			if (found2)
			{
				prefetch_cache_out[*prefetch_count_out].page = prefetch_page2;
				prefetch_cache_out[*prefetch_count_out].itemid = itemid2;
				(*prefetch_count_out)++;
				__builtin_prefetch(itemid2, 0, 3);
			}
			if (found3)
			{
				prefetch_cache_out[*prefetch_count_out].page = prefetch_page3;
				prefetch_cache_out[*prefetch_count_out].itemid = itemid3;
				(*prefetch_count_out)++;
				__builtin_prefetch(itemid3, 0, 3);
			}
		}
		
		/* Handle remaining items */
		for (; j < *unvisitedLength; j++)
		{
			if (batch_found[j])
			{
				Page		prefetch_page;
				OffsetNumber offno;
				ItemId		itemid;
				ItemPointer tid = &unvisited[j].indextid;
				
				/* Get the page and ItemId */
				prefetch_page = BufferGetPage(batch_frameids[j] + 1);
				offno = ItemPointerGetOffsetNumber(tid);
				itemid = PageGetItemId(prefetch_page, offno);
				
				/* Cache page and ItemId for second-phase prefetch */
				prefetch_cache_out[*prefetch_count_out].page = prefetch_page;
				prefetch_cache_out[*prefetch_count_out].itemid = itemid;
				(*prefetch_count_out)++;
				
				/* Prefetch ItemId */
				__builtin_prefetch(itemid, 0, 3);
			}
			else
			{
				/* Collect missing block */
				missing_blocks[missing_count++] = batch_blocknums[j];
			}
		}
		
		/*
		 * Issue bulk async I/O for all missing blocks.
		 * ReadBuffersAsync will group consecutive blocks and submit
		 * all I/Os in one batch for maximum efficiency.
		 */
		if (missing_count > 0)
		{
			ReadBuffersAsync(smgr,
							 index->rd_rel->relpersistence,
							 MAIN_FORKNUM,
							 missing_blocks,
							 missing_count);
		}
	}
	
	/* Note: Second pass (vector data prefetch) is now done by caller (HnswSearchLayer) */
}

/*
 * Algorithm 2 from paper
 */
List *
HnswSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples)
{
	List	   *w = NIL;
	pairingheap *C = pairingheap_allocate(CompareNearestCandidates, NULL);
	pairingheap *W = pairingheap_allocate(CompareFurthestCandidates, NULL);
	int			wlen = 0;
	visited_hash vh;
	ListCell   *lc2;
	HnswNeighborArray *localNeighborhood = NULL;
	Size		neighborhoodSize = 0;
	int			lm = HnswGetLayerM(m, lc);
	HnswUnvisited *unvisited = palloc(lm * sizeof(HnswUnvisited));
	int			unvisitedLength;
	bool		inMemory = index == NULL;
	HnswPrefetchCache prefetch_cache[HNSW_MAX_M];
	int			prefetch_count = 0;

	if (v == NULL)
	{
		v = &vh;
		initVisited = true;
	}

	if (initVisited)
	{
		InitVisited(base, v, inMemory, ef, m);

		if (discarded != NULL)
			*discarded = pairingheap_allocate(CompareNearestDiscardedCandidates, NULL);
	}

	/* Create local memory for neighborhood if needed */
	if (inMemory)
	{
		neighborhoodSize = HNSW_NEIGHBOR_ARRAY_SIZE(lm);
		localNeighborhood = palloc(neighborhoodSize);
	}

	/* Add entry points to v, C, and W */
	foreach(lc2, ep)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
		bool		found;

		if (initVisited)
		{
			AddToVisited(base, v, sc->element, inMemory, &found);

			/* OK to count elements instead of tuples */
			if (tuples != NULL)
				(*tuples)++;
		}

		pairingheap_add(C, &sc->c_node);
		pairingheap_add(W, &sc->w_node);

		/*
		 * Do not count elements being deleted towards ef when vacuuming. It
		 * would be ideal to do this for inserts as well, but this could
		 * affect insert performance.
		 */
		if (CountElement(skipElement, HnswPtrAccess(base, sc->element)))
			wlen++;
	}

	while (!pairingheap_is_empty(C))
	{
		HnswSearchCandidate *c = HnswGetSearchCandidate(c_node, pairingheap_remove_first(C));
		HnswSearchCandidate *f = HnswGetSearchCandidate(w_node, pairingheap_first(W));
		HnswElement cElement;

		if (c->distance > f->distance)
			break;

		cElement = HnswPtrAccess(base, c->element);

		if (inMemory)
			HnswLoadUnvisitedFromMemory(base, cElement, unvisited, &unvisitedLength, v, lc, localNeighborhood, neighborhoodSize);
		else
			HnswLoadUnvisitedFromDisk(cElement, unvisited, &unvisitedLength, v, index, m, lm, lc, prefetch_cache, &prefetch_count);

		/* OK to count elements instead of tuples */
		if (tuples != NULL)
			(*tuples) += unvisitedLength;

		for (int i = 0; i < unvisitedLength; i++)
		{
			HnswElement eElement;
			HnswSearchCandidate *e;
			double		eDistance;
			bool		alwaysAdd = wlen < ef;

			/* Prefetch next element's vector data if available and not in memory */
			if (!inMemory && prefetch_count > 0 && i + 1 < prefetch_count)
			{
				HnswElementTuple etup = (HnswElementTuple) PageGetItem(prefetch_cache[i + 1].page, prefetch_cache[i + 1].itemid);
				__builtin_prefetch(&etup->data, 0, 3);
			}

			f = HnswGetSearchCandidate(w_node, pairingheap_first(W));

			if (inMemory)
			{
				eElement = unvisited[i].element;
				eDistance = GetElementDistance(base, eElement, q, support);
			}
			else
			{
				ItemPointer indextid = &unvisited[i].indextid;
				BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
				OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);

				/* Avoid any allocations if not adding */
				eElement = NULL;
				HnswLoadElementImpl(blkno, offno, &eDistance, q, index, support, inserting, alwaysAdd || discarded != NULL ? NULL : &f->distance, &eElement);

				if (eElement == NULL)
					continue;
			}

			if (eElement == NULL || !(eDistance < f->distance || alwaysAdd))
			{
				if (discarded != NULL)
				{
					/* Create a new candidate */
					e = HnswInitSearchCandidate(base, eElement, eDistance);
					pairingheap_add(*discarded, &e->w_node);
				}

				continue;
			}

			/* Make robust to issues */
			if (eElement->level < lc)
				continue;

			/* Create a new candidate */
			e = HnswInitSearchCandidate(base, eElement, eDistance);
			pairingheap_add(C, &e->c_node);
			pairingheap_add(W, &e->w_node);

			/*
			 * Do not count elements being deleted towards ef when vacuuming.
			 * It would be ideal to do this for inserts as well, but this
			 * could affect insert performance.
			 */
			if (CountElement(skipElement, eElement))
			{
				wlen++;

				/* No need to decrement wlen */
				if (wlen > ef)
				{
					HnswSearchCandidate *d = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

					if (discarded != NULL)
						pairingheap_add(*discarded, &d->w_node);
				}
			}
		}
	}

	/* Add each element of W to w */
	while (!pairingheap_is_empty(W))
	{
		HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

		w = lappend(w, sc);
	}

	return w;
}

/*
 * Compare candidate distances with pointer tie-breaker
 */
static int
CompareCandidateDistances(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (HnswPtrPointer(hca->element) < HnswPtrPointer(hcb->element))
		return 1;

	if (HnswPtrPointer(hca->element) > HnswPtrPointer(hcb->element))
		return -1;

	return 0;
}

/*
 * Compare candidate distances with offset tie-breaker
 */
static int
CompareCandidateDistancesOffset(const ListCell *a, const ListCell *b)
{
	HnswCandidate *hca = lfirst(a);
	HnswCandidate *hcb = lfirst(b);

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (HnswPtrOffset(hca->element) < HnswPtrOffset(hcb->element))
		return 1;

	if (HnswPtrOffset(hca->element) > HnswPtrOffset(hcb->element))
		return -1;

	return 0;
}

/*
 * Check if an element is closer to q than any element from R
 */
static bool
CheckElementCloser(char *base, HnswCandidate * e, List *r, HnswSupport * support)
{
	HnswElement eElement = HnswPtrAccess(base, e->element);
	Datum		eValue = HnswGetValue(base, eElement);
	ListCell   *lc2;

	foreach(lc2, r)
	{
		HnswCandidate *ri = lfirst(lc2);
		HnswElement riElement = HnswPtrAccess(base, ri->element);
		Datum		riValue = HnswGetValue(base, riElement);
		float		distance = HnswGetDistance(eValue, riValue, support);

		if (distance <= e->distance)
			return false;
	}

	return true;
}

/*
 * Algorithm 4 from paper
 */
static List *
SelectNeighbors(char *base, List *c, int lm, HnswSupport * support, bool *closerSet, HnswCandidate * newCandidate, HnswCandidate * *pruned, bool sortCandidates)
{
	List	   *r = NIL;
	List	   *w = list_copy(c);
	HnswCandidate **wd;
	int			wdlen = 0;
	int			wdoff = 0;
	bool		mustCalculate = !(*closerSet);
	List	   *added = NIL;
	bool		removedAny = false;

	if (list_length(w) <= lm)
		return w;

	wd = palloc(sizeof(HnswCandidate *) * list_length(w));

	/* Ensure order of candidates is deterministic for closer caching */
	if (sortCandidates)
	{
		if (base == NULL)
			list_sort(w, CompareCandidateDistances);
		else
			list_sort(w, CompareCandidateDistancesOffset);
	}

	while (list_length(w) > 0 && list_length(r) < lm)
	{
		/* Assumes w is already ordered desc */
		HnswCandidate *e = llast(w);

		w = list_delete_last(w);

		/* Use previous state of r and wd to skip work when possible */
		if (mustCalculate)
			e->closer = CheckElementCloser(base, e, r, support);
		else if (list_length(added) > 0)
		{
			/* Keep Valgrind happy for in-memory, parallel builds */
			if (base != NULL)
				VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

			/*
			 * If the current candidate was closer, we only need to compare it
			 * with the other candidates that we have added.
			 */
			if (e->closer)
			{
				e->closer = CheckElementCloser(base, e, added, support);

				if (!e->closer)
					removedAny = true;
			}
			else
			{
				/*
				 * If we have removed any candidates from closer, a candidate
				 * that was not closer earlier might now be.
				 */
				if (removedAny)
				{
					e->closer = CheckElementCloser(base, e, r, support);
					if (e->closer)
						added = lappend(added, e);
				}
			}
		}
		else if (e == newCandidate)
		{
			e->closer = CheckElementCloser(base, e, r, support);
			if (e->closer)
				added = lappend(added, e);
		}

		/* Keep Valgrind happy for in-memory, parallel builds */
		if (base != NULL)
			VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

		if (e->closer)
			r = lappend(r, e);
		else
			wd[wdlen++] = e;
	}

	/* Cached value can only be used in future if sorted deterministically */
	*closerSet = sortCandidates;

	/* Keep pruned connections */
	while (wdoff < wdlen && list_length(r) < lm)
		r = lappend(r, wd[wdoff++]);

	/* Return pruned for update connections */
	if (pruned != NULL)
	{
		if (wdoff < wdlen)
			*pruned = wd[wdoff];
		else
			*pruned = linitial(w);
	}

	return r;
}

/*
 * Add connections
 */
static void
AddConnections(char *base, HnswElement element, List *neighbors, int lc)
{
	ListCell   *lc2;
	HnswNeighborArray *a = HnswGetNeighbors(base, element, lc);

	foreach(lc2, neighbors)
		a->items[a->length++] = *((HnswCandidate *) lfirst(lc2));
}

/*
 * Update connections
 */
void
HnswUpdateConnection(char *base, HnswNeighborArray * neighbors, HnswElement newElement, float distance, int lm, int *updateIdx, Relation index, HnswSupport * support)
{
	HnswCandidate newHc;

	HnswPtrStore(base, newHc.element, newElement);
	newHc.distance = distance;

	if (neighbors->length < lm)
	{
		neighbors->items[neighbors->length++] = newHc;

		/* Track update */
		if (updateIdx != NULL)
			*updateIdx = -2;
	}
	else
	{
		/* Shrink connections */
		List	   *c = NIL;
		HnswCandidate *pruned = NULL;

		/* Add candidates */
		for (int i = 0; i < neighbors->length; i++)
			c = lappend(c, &neighbors->items[i]);
		c = lappend(c, &newHc);

		SelectNeighbors(base, c, lm, support, &neighbors->closerSet, &newHc, &pruned, true);

		/* Should not happen */
		if (pruned == NULL)
			return;

		/* Find and replace the pruned element */
		for (int i = 0; i < neighbors->length; i++)
		{
			if (HnswPtrEqual(base, neighbors->items[i].element, pruned->element))
			{
				neighbors->items[i] = newHc;

				/* Track update */
				if (updateIdx != NULL)
					*updateIdx = i;

				break;
			}
		}
	}
}

/*
 * Remove elements being deleted or skipped
 */
static List *
RemoveElements(char *base, List *w, HnswElement skipElement)
{
	ListCell   *lc2;
	List	   *w2 = NIL;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	foreach(lc2, w)
	{
		HnswCandidate *hc = (HnswCandidate *) lfirst(lc2);
		HnswElement hce = HnswPtrAccess(base, hc->element);

		/* Skip self for vacuuming update */
		if (skipElement != NULL && hce->blkno == skipElement->blkno && hce->offno == skipElement->offno)
			continue;

		if (hce->heaptidsLength != 0)
			w2 = lappend(w2, hc);
	}

	return w2;
}

/*
 * Precompute hash
 */
static void
PrecomputeHash(char *base, HnswElement element)
{
	HnswElementPtr ptr;

	HnswPtrStore(base, ptr, element);

	if (base == NULL)
		element->hash = hash_pointer((uintptr_t) HnswPtrPointer(ptr));
	else
		element->hash = hash_offset(HnswPtrOffset(ptr));
}

/*
 * Algorithm 1 from paper
 */
void
HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, HnswSupport * support, int m, int efConstruction, bool existing)
{
	List	   *ep;
	List	   *w;
	int			level = element->level;
	int			entryLevel;
	HnswQuery	q;
	HnswElement skipElement = existing ? element : NULL;
	bool		inMemory = index == NULL;

	q.value = HnswGetValue(base, element);

	/* Precompute hash */
	if (inMemory)
		PrecomputeHash(base, element);

	/* No neighbors if no entry point */
	if (entryPoint == NULL)
		return;

	/* Get entry point and level */
	ep = list_make1(HnswEntryCandidate(base, entryPoint, &q, index, support, true));
	entryLevel = entryPoint->level;

	/* 1st phase: greedy search to insert level */
	for (int lc = entryLevel; lc >= level + 1; lc--)
	{
		w = HnswSearchLayer(base, &q, ep, 1, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL);
		ep = w;
	}

	if (level > entryLevel)
		level = entryLevel;

	/* Add one for existing element */
	if (existing)
		efConstruction++;

	/* 2nd phase */
	for (int lc = level; lc >= 0; lc--)
	{
		int			lm = HnswGetLayerM(m, lc);
		List	   *neighbors;
		List	   *lw = NIL;
		ListCell   *lc2;

		w = HnswSearchLayer(base, &q, ep, efConstruction, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL);

		/* Convert search candidates to candidates */
		foreach(lc2, w)
		{
			HnswSearchCandidate *sc = lfirst(lc2);
			HnswCandidate *hc = palloc(sizeof(HnswCandidate));

			hc->element = sc->element;
			hc->distance = sc->distance;

			lw = lappend(lw, hc);
		}

		/* Elements being deleted or skipped can help with search */
		/* but should be removed before selecting neighbors */
		if (!inMemory)
			lw = RemoveElements(base, lw, skipElement);

		/*
		 * Candidates are sorted, but not deterministically. Could set
		 * sortCandidates to true for in-memory builds to enable closer
		 * caching, but there does not seem to be a difference in performance.
		 */
		neighbors = SelectNeighbors(base, lw, lm, support, &HnswGetNeighbors(base, element, lc)->closerSet, NULL, NULL, false);

		AddConnections(base, element, neighbors, lc);

		ep = w;
	}
}

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum halfvec_l2_normalize(PG_FUNCTION_ARGS);
PGDLLEXPORT Datum sparsevec_l2_normalize(PG_FUNCTION_ARGS);

static void
SparsevecCheckValue(Pointer v)
{
	SparseVector *vec = (SparseVector *) v;

	if (vec->nnz > HNSW_MAX_NNZ)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("sparsevec cannot have more than %d non-zero elements for hnsw index", HNSW_MAX_NNZ)));
}

/*
 * Get type info
 */
const		HnswTypeInfo *
HnswGetTypeInfo(Relation index)
{
	FmgrInfo   *procinfo = HnswOptionalProcInfo(index, HNSW_TYPE_INFO_PROC);

	if (procinfo == NULL)
	{
		static const HnswTypeInfo typeInfo = {
			.maxDimensions = HNSW_MAX_DIM,
			.normalize = l2_normalize,
			.checkValue = NULL
		};

		return (&typeInfo);
	}
	else
		return (const HnswTypeInfo *) DatumGetPointer(FunctionCall0Coll(procinfo, InvalidOid));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_halfvec_support);
Datum
hnsw_halfvec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 2,
		.normalize = halfvec_l2_normalize,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_bit_support);
Datum
hnsw_bit_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = HNSW_MAX_DIM * 32,
		.normalize = NULL,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_sparsevec_support);
Datum
hnsw_sparsevec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = SPARSEVEC_MAX_DIM,
		.normalize = sparsevec_l2_normalize,
		.checkValue = SparsevecCheckValue
	};

	PG_RETURN_POINTER(&typeInfo);
}
