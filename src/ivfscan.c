#include "postgres.h"

#include <float.h>

#include "access/relscan.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "lib/pairingheap.h"
#include "ivfflat.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"

#define GetScanList(ptr) pairingheap_container(IvfflatScanList, ph_node, ptr)
#define GetScanListConst(ptr) pairingheap_const_container(IvfflatScanList, ph_node, ptr)
#define IVFFLAT_TOPK_PRUNE_MAX_LIMIT 50

/* from vector.c */
extern Datum vector_l2_squared_distance(PG_FUNCTION_ARGS);

typedef struct IvfflatTopKState
{
	int			limit;
	int			count;
	double		threshold;
	pairingheap *heap;
	struct IvfflatTopKNode
	{
		pairingheap_node ph_node;
		double		distance;
	}			nodes[IVFFLAT_TOPK_PRUNE_MAX_LIMIT];
} IvfflatTopKState;

/* Max-heap by distance (worst candidate at root) */
static int
CompareTopKNodes(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	const struct IvfflatTopKNode *na = pairingheap_const_container(struct IvfflatTopKNode, ph_node, a);
	const struct IvfflatTopKNode *nb = pairingheap_const_container(struct IvfflatTopKNode, ph_node, b);

	if (na->distance > nb->distance)
		return 1;

	if (na->distance < nb->distance)
		return -1;

	return 0;
}

static inline void
IvfflatTopKInit(IvfflatTopKState *state, int limit)
{
	state->limit = limit;
	state->count = 0;
	state->threshold = DBL_MAX;
	state->heap = pairingheap_allocate(CompareTopKNodes, NULL);
}

static inline void
IvfflatTopKFree(IvfflatTopKState *state)
{
	pairingheap_free(state->heap);
	state->heap = NULL;
}

static inline void
IvfflatTopKConsider(IvfflatTopKState *state, double distance)
{
	struct IvfflatTopKNode *node;

	if (state->count < state->limit)
	{
		node = &state->nodes[state->count++];
		node->distance = distance;
		pairingheap_add(state->heap, &node->ph_node);

		if (state->count == state->limit)
			state->threshold = pairingheap_container(struct IvfflatTopKNode, ph_node, pairingheap_first(state->heap))->distance;

		return;
	}

	if (distance >= state->threshold)
		return;

	node = pairingheap_container(struct IvfflatTopKNode, ph_node, pairingheap_remove_first(state->heap));
	node->distance = distance;
	pairingheap_add(state->heap, &node->ph_node);
	state->threshold = pairingheap_container(struct IvfflatTopKNode, ph_node, pairingheap_first(state->heap))->distance;
}

/*
 * Compute L2 squared distance with chunked early-stop checks.
 *
 * Returns true if full distance is computed, false if computation is pruned
 * because the partial distance exceeds threshold.
 */
static inline bool
IvfflatL2SquaredDistancePruned(Vector *query, Vector *item, double threshold, double *distance)
{
	float		dist = 0.0f;
	const float *qx = query->x;
	const float *ix = item->x;
	int			dim = query->dim;
	int			i = 0;

	Assert(item->dim == dim);

	/*
	 * Process in 64-dimension chunks and check threshold once per chunk to
	 * reduce branch overhead in the hot loop.
	 */
	for (; i + 64 <= dim; i += 64)
	{
		float		blockDist = 0.0f;

		for (int j = i; j < i + 64; j++)
		{
			float		diff = qx[j] - ix[j];

			blockDist += diff * diff;
		}

		dist += blockDist;

		if ((double) dist > threshold)
		{
			*distance = (double) dist;
			return false;
		}
	}

	/* Tail dimensions (at most 63) */
	for (; i < dim; i++)
	{
		float		diff = qx[i] - ix[i];

		dist += diff * diff;
	}

	*distance = (double) dist;
	return true;
}

static inline Vector *
IvfflatGetVectorForRead(Datum value, bool *needFree)
{
	Pointer		original = DatumGetPointer(value);
	Vector	   *vec = DatumGetVector(value);

	*needFree = PointerGetDatum(vec) != PointerGetDatum(original);
	return vec;
}

/*
 * Compare list distances
 */
static int
CompareLists(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (GetScanListConst(a)->distance > GetScanListConst(b)->distance)
		return 1;

	if (GetScanListConst(a)->distance < GetScanListConst(b)->distance)
		return -1;

	return 0;
}

/*
 * Get lists and sort by distance
 */
static void
GetScanLists(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
	int			listCount = 0;
	double		maxDistance = DBL_MAX;

	/* Search all list pages */
	while (BlockNumberIsValid(nextblkno))
	{
		Buffer		cbuf;
		Page		cpage;
		OffsetNumber maxoffno;

		cbuf = ReadBuffer(scan->indexRelation, nextblkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);

		maxoffno = PageGetMaxOffsetNumber(cpage);

		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
			double		distance;

			/* Use procinfo from the index instead of scan key for performance */
			distance = DatumGetFloat8(so->distfunc(so->procinfo, so->collation, PointerGetDatum(&list->center), value));

			if (listCount < so->maxProbes)
			{
				IvfflatScanList *scanlist;

				scanlist = &so->lists[listCount];
				scanlist->startPage = list->startPage;
				scanlist->distance = distance;
				listCount++;

				/* Add to heap */
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Calculate max distance */
				if (listCount == so->maxProbes)
					maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
			else if (distance < maxDistance)
			{
				IvfflatScanList *scanlist;

				/* Remove */
				scanlist = GetScanList(pairingheap_remove_first(so->listQueue));

				/* Reuse */
				scanlist->startPage = list->startPage;
				scanlist->distance = distance;
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Update max distance */
				maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
		}

		nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;

		UnlockReleaseBuffer(cbuf);
	}

	for (int i = listCount - 1; i >= 0; i--)
		so->listPages[i] = GetScanList(pairingheap_remove_first(so->listQueue))->startPage;

	Assert(pairingheap_is_empty(so->listQueue));
}

/*
 * Get items
 */
static void
GetScanItems(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	TupleDesc	tupdesc = RelationGetDescr(scan->indexRelation);
	TupleTableSlot *slot = so->vslot;
	IvfflatTopKState topk = {0};
	bool		useTopKPrune = so->topKPruneActive && DatumGetPointer(value) != NULL;
	bool		queryNeedsFree = false;
	Vector	   *queryVec = NULL;
	int			batchProbes = 0;

	if (useTopKPrune)
		queryVec = IvfflatGetVectorForRead(value, &queryNeedsFree);

	tuplesort_reset(so->sortstate);
	if (useTopKPrune)
		IvfflatTopKInit(&topk, so->topKPruneLimit);

	/* Search closest probes lists */
	while (so->listIndex < so->maxProbes && (++batchProbes) <= so->probes)
	{
		BlockNumber searchPage = so->listPages[so->listIndex++];

		/* Search all entry pages for list */
		while (BlockNumberIsValid(searchPage))
		{
			Buffer		buf;
			Page		page;
			OffsetNumber maxoffno;

			buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, searchPage, RBM_NORMAL, so->bas);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			maxoffno = PageGetMaxOffsetNumber(page);

				for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
				{
					IndexTuple	itup;
					Datum		datum;
					double		distance;
					bool		isnull;
					ItemId		itemid = PageGetItemId(page, offno);

					itup = (IndexTuple) PageGetItem(page, itemid);
					datum = index_getattr(itup, 1, tupdesc, &isnull);

					if (useTopKPrune)
					{
						bool		itemNeedsFree = false;
						Vector	   *itemVec = IvfflatGetVectorForRead(datum, &itemNeedsFree);
						bool		fullDistance = IvfflatL2SquaredDistancePruned(queryVec, itemVec, topk.threshold, &distance);

						if (itemNeedsFree)
							pfree(itemVec);

						if (!fullDistance)
							continue;

						IvfflatTopKConsider(&topk, distance);
					}
					else
						distance = DatumGetFloat8(so->distfunc(so->procinfo, so->collation, datum, value));

					/*
					 * Add virtual tuple
					 *
					 * Use procinfo from the index instead of scan key for
					 * performance
					 */
					ExecClearTuple(slot);
					slot->tts_values[0] = Float8GetDatum(distance);
					slot->tts_isnull[0] = false;
					slot->tts_values[1] = PointerGetDatum(&itup->t_tid);
					slot->tts_isnull[1] = false;
					ExecStoreVirtualTuple(slot);

				tuplesort_puttupleslot(so->sortstate, slot);
			}

			searchPage = IvfflatPageGetOpaque(page)->nextblkno;

			UnlockReleaseBuffer(buf);
		}
	}

	if (queryNeedsFree)
		pfree(queryVec);
	if (useTopKPrune)
		IvfflatTopKFree(&topk);

	tuplesort_performsort(so->sortstate);

#if defined(IVFFLAT_MEMORY)
	elog(INFO, "memory: %zu MB", MemoryContextMemAllocated(CurrentMemoryContext, true) / (1024 * 1024));
#endif
}

/*
 * Zero distance
 */
static Datum
ZeroDistance(FmgrInfo *flinfo, Oid collation, Datum arg1, Datum arg2)
{
	return Float8GetDatum(0.0);
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
	{
		value = PointerGetDatum(NULL);
		so->distfunc = ZeroDistance;
	}
	else
	{
		value = scan->orderByData->sk_argument;
		so->distfunc = FunctionCall2Coll;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->normprocinfo != NULL)
		{
			MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

			value = IvfflatNormValue(so->typeInfo, so->collation, value);

			MemoryContextSwitchTo(oldCtx);
		}
	}

	return value;
}

/*
 * Initialize scan sort state
 */
static Tuplesortstate *
InitScanSortState(TupleDesc tupdesc)
{
	AttrNumber	attNums[] = {1};
	Oid			sortOperators[] = {Float8LessOperator};
	Oid			sortCollations[] = {InvalidOid};
	bool		nullsFirstFlags[] = {false};

	return tuplesort_begin_heap(tupdesc, 1, attNums, sortOperators, sortCollations, nullsFirstFlags, work_mem, NULL, false);
}

/*
 * Prepare for an index scan
 */
IndexScanDesc
ivfflatbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	IvfflatScanOpaque so;
	int			lists;
	int			dimensions;
	int			probes = ivfflat_probes;
	int			maxProbes;
	MemoryContext oldCtx;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	/* Get lists and dimensions from metapage */
	IvfflatGetMetaPageInfo(index, &lists, &dimensions);

	if (ivfflat_iterative_scan != IVFFLAT_ITERATIVE_SCAN_OFF)
		maxProbes = Max(ivfflat_max_probes, probes);
	else
		maxProbes = probes;

	if (probes > lists)
		probes = lists;

	if (maxProbes > lists)
		maxProbes = lists;

	so = (IvfflatScanOpaque) palloc(sizeof(IvfflatScanOpaqueData));
	so->typeInfo = IvfflatGetTypeInfo(index);
	so->first = true;
	so->probes = probes;
	so->maxProbes = maxProbes;
	so->dimensions = dimensions;

	/* Set support functions */
	so->procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
	so->normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
	so->collation = index->rd_indcollation[0];
	so->topKPruneLimit = ivfflat_topk_prune_limit;
	so->topKPruneActive = ivfflat_topk_prune_limit > 0 && ivfflat_topk_prune_limit <= IVFFLAT_TOPK_PRUNE_MAX_LIMIT &&
		so->procinfo->fn_addr == vector_l2_squared_distance;

	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Ivfflat scan temporary context",
									   ALLOCSET_DEFAULT_SIZES);

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/* Create tuple description for sorting */
	so->tupdesc = CreateTemplateTupleDesc(2);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 1, "distance", FLOAT8OID, -1, 0);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 2, "heaptid", TIDOID, -1, 0);

	/* Prep sort */
	so->sortstate = InitScanSortState(so->tupdesc);

	/* Need separate slots for puttuple and gettuple */
	so->vslot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsVirtual);
	so->mslot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsMinimalTuple);

	/*
	 * Reuse same set of shared buffers for scan
	 *
	 * See postgres/src/backend/storage/buffer/README for description
	 */
	so->bas = GetAccessStrategy(BAS_BULKREAD);

	so->listQueue = pairingheap_allocate(CompareLists, scan);
	so->listPages = palloc(maxProbes * sizeof(BlockNumber));
	so->listIndex = 0;
	so->lists = palloc(maxProbes * sizeof(IvfflatScanList));

	MemoryContextSwitchTo(oldCtx);

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void
ivfflatrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;

	so->first = true;
	pairingheap_reset(so->listQueue);
	so->listIndex = 0;

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
bool
ivfflatgettuple(IndexScanDesc scan, ScanDirection dir)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	ItemPointer heaptid;
	bool		isnull;

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan ivfflat index without order");

		/* Requires MVCC-compliant snapshot as not able to pin during sorting */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with ivfflat");

		value = GetScanValue(scan);
		IvfflatBench("GetScanLists", GetScanLists(scan, value));
		IvfflatBench("GetScanItems", GetScanItems(scan, value));
		so->first = false;
		so->value = value;
	}

	while (!tuplesort_gettupleslot(so->sortstate, true, false, so->mslot, NULL))
	{
		if (so->listIndex == so->maxProbes)
			return false;

		IvfflatBench("GetScanItems", GetScanItems(scan, so->value));
	}

	heaptid = (ItemPointer) DatumGetPointer(slot_getattr(so->mslot, 2, &isnull));

	scan->xs_heaptid = *heaptid;
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;
	return true;
}

/*
 * End a scan and release resources
 */
void
ivfflatendscan(IndexScanDesc scan)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;

	/* Free any temporary files */
	tuplesort_end(so->sortstate);

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}
