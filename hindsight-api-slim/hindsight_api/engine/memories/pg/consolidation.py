"""The consolidation writes and checks: source liveness, observation insert / rewrite, dedup folds.

The SQL is lifted verbatim from ``engine/consolidation/consolidator.py``, which now makes one
store call per site. Each runs on the caller's connection, inside the consolidation batch's
transaction, so every write derived from one LLM response commits or rolls back together
(#3876), and the ``FOR SHARE`` locks hold until that transaction ends.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from ....config import get_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...consolidation.consolidator import _TemporalBounds

#: Change-record event types: TEXT_CHANGE for a consolidation text rewrite,
#: FOLD_IN for a new source id on a read observation, SRC_EDIT for a
#: source-memory edit/refine/supersede/invalidate.
TEXT_CHANGE = "TEXT_CHANGE"
FOLD_IN = "FOLD_IN"
SRC_EDIT = "SRC_EDIT"


def _is_postgres(conn) -> bool:
    """Whether ``conn`` targets Postgres (the only backend with change records)."""
    return getattr(conn, "backend_type", "postgresql") == "postgresql"


def _row_version_token(row) -> str:
    """The row's pre-write version token, for a stable event identity.

    Tolerant of rows that carry no ``updated_at`` (mock connections in unit
    tests): such a token is empty, and the derived identity still
    deduplicates a replayed identical call.
    """
    try:
        value = row["updated_at"]
    except (KeyError, IndexError, TypeError):
        return ""
    return str(value)


def stable_event_id(
    *,
    bank_id: str,
    subject_id: str | uuid.UUID,
    event_type: str,
    version_token: str,
    before_text: str | None = None,
    after_text: str | None = None,
    detail: dict | None = None,
) -> str:
    """A stable identity for one committed transition.

    The version token is the subject row's own pre-write identity (its
    ``updated_at`` at the read that gated the write): a replay of the same
    write carries the same token and deduplicates, while a distinct later
    transition of the same row carries a new token and records even when its
    before/after values repeat an earlier transition's. ``detail`` is part of
    the identity when the caller supplies one: two folds of
    one row in one transaction share the token and the before/after values
    but carry different ``added_source_ids``, and omitting the detail would
    collapse them to one record and lose the second fold's audit detail.
    A retry replays the same detail, so the same transition still dedupes.
    """
    payload = json.dumps(
        {
            "bank_id": bank_id,
            "subject_id": str(subject_id),
            "event_type": event_type,
            "version_token": version_token,
            "before_text": before_text,
            "after_text": after_text,
            "detail": detail,
        },
        sort_keys=True,
        default=str,
    )
    return f"{event_type.lower()}:{hashlib.sha256(payload.encode()).hexdigest()[:32]}"


def _derived_event_id(
    *,
    bank_id: str,
    subject_id: str | uuid.UUID,
    event_type: str,
    before_text: str | None,
    after_text: str | None,
    detail: dict | None,
) -> str:
    """The identity for a call that supplied none: derived from the transition
    itself, so a replayed call with identical arguments deduplicates instead
    of minting a fresh identity per retry. Write sites
    that know the row's version token pass an explicit stable id instead,
    which keeps distinct transitions with identical values apart."""
    payload = json.dumps(
        {
            "bank_id": bank_id,
            "subject_id": str(subject_id),
            "event_type": event_type,
            "before_text": before_text,
            "after_text": after_text,
            "detail": detail,
        },
        sort_keys=True,
        default=str,
    )
    return f"{event_type.lower()}:{hashlib.sha256(payload.encode()).hexdigest()[:32]}"


def _text_changed(before: str, after: str) -> bool:
    """The server-side before/after verdict (owned by the
    consolidation worker). Imported lazily: the consolidator reaches this
    module through the memories store, so a module-level import would cycle.
    """
    from ...consolidation.consolidator import compare_text  # noqa: PLC0415

    return compare_text(before, after)


async def record_change_event(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    subject_id: str | uuid.UUID,
    event_type: str,
    event_id: str | None = None,
    before_text: str | None = None,
    after_text: str | None = None,
    detail: dict | None = None,
) -> None:
    """Write one durable, timestamped change record on the caller's connection.

    Runs inside whatever transaction the caller already holds, so the record
    commits or rolls back with the write it describes. ``event_id`` is the
    stable operation identity: a retry replaying the same identity inserts
    nothing (unique index on ``(bank_id, event_id)``), while distinct
    committed transitions record even when their before/after values repeat —
    identity, never values, deduplicates. A caller that supplies no id gets
    one derived from the transition itself, so a replayed call with identical
    arguments still deduplicates; write sites that know the row's version
    token pass an explicit stable id, which keeps distinct transitions with
    identical values apart.

    The change-record table exists only on Postgres (its migration runs for
    pg), so on Oracle this is a no-op: the consolidation rewrite path runs
    there and must not fail on the missing table.

    Before inserting, the call takes a per-bank transaction advisory lock.
    ``seq`` is an identity allocated at INSERT time, which does not follow
    commit order: a record can allocate seq N and commit long after a later
    seq. The advisory lock serializes record commits per bank, so every seq
    at or below a baseline captured under the same lock (the refresh's
    refresh start capture) has already committed, and every later-committing record
    allocates a higher seq. The lock is held to the
    caller's transaction end, which the caller must hold for atomicity with
    the write in any case.
    """
    from ...schema import _is_oracle  # noqa: PLC0415

    if _is_oracle() or not _is_postgres(conn):
        return
    # The advisory lock below is xact-scoped: outside a transaction each
    # statement autocommits and the ordering guarantee it buys would silently
    # vanish. Fail loud instead; every writer's
    # caller holds the transaction that carries the record. asyncpg names
    # the probe ``is_in_transaction`` and the PostgresConnection wrapper
    # exposes ``in_transaction``; accept either.
    in_transaction = getattr(conn, "in_transaction", None) or getattr(conn, "is_in_transaction", None)
    if callable(in_transaction) and not in_transaction():
        raise RuntimeError(
            "record_change_event requires an open transaction: the per-bank advisory lock "
            "serializing record commits only holds inside one"
        )
    await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1)::bigint)", bank_id)
    await conn.execute(
        f"""
        INSERT INTO {fq_table("memory_change_events")}
            (event_id, bank_id, subject_id, event_type, before_text, after_text, detail)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (bank_id, event_id) DO NOTHING
        """,
        event_id
        or _derived_event_id(
            bank_id=bank_id,
            subject_id=subject_id,
            event_type=event_type,
            before_text=before_text,
            after_text=after_text,
            detail=detail,
        ),
        bank_id,
        uuid.UUID(str(subject_id)),
        event_type,
        before_text,
        after_text,
        json.dumps(detail) if detail is not None else None,
    )


def _search_vector_clause(param: str) -> str:
    """UPDATE-clause fragment that rebuilds ``search_vector`` from the text bound at ``param``,
    or '' when the backend keeps no inline column to rebuild.

    Every observation write that changes the text must rebuild it the way the insert built it,
    or keyword search keeps matching the old wording (#5299 — VectorChord's ``bm25vector`` is a
    plain column nothing refreshes on its own). The per-backend expression comes from
    ``pg_search_vector_expr``, the one the ``memory_units`` and ``mental_models`` writes use.
    On Oracle ``search_vector`` is a CLOB kept by Oracle's own text index, and the PG-only
    ``::regconfig`` cast would reach it as an unbound placeholder (DPY-4010, #3021).
    """
    expr = observation_search_vector_expr(param)
    return f",\n            search_vector = {expr}" if expr else ""


def observation_search_vector_expr(param: str) -> str | None:
    """The ``search_vector`` value for an observation whose text is ``param``; None to leave it out.

    Shared with repair-bank's rebuild (pg/admin.py), so the repair builds exactly what the writes do.
    """
    from ...db.ops_postgresql import pg_search_vector_expr  # noqa: PLC0415
    from ...schema import _is_oracle  # noqa: PLC0415

    if _is_oracle():
        return None
    return pg_search_vector_expr(get_config(), text_col=param, context_col=None, signals_col=None)


async def lock_live_memory_ids(
    *, conn, fq_table: Callable[[str], str], bank_id: str, unit_ids: list[uuid.UUID]
) -> set[str]:
    """Which of ``unit_ids`` still exist, ``FOR SHARE``-locked until the transaction ends.

    (Oracle has no ``FOR SHARE``; the SQL rewriter promotes it to ``FOR UPDATE`` — more
    conservative, still correct.)
    """
    rows = await conn.fetch(
        f"SELECT id FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[]) AND bank_id = $2 FOR SHARE",
        unit_ids,
        bank_id,
    )
    return {str(r["id"]) for r in rows}


async def lock_consolidation_write_set(
    *, conn, fq_table: Callable[[str], str], bank_id: str, fact_ids: list[str], observation_ids: list[str]
) -> None:
    """Lock everything one consolidation response writes, in the order retains lock it (#5433).

    A retain row-locks its document first and holds it to commit, then sweeps observations and
    rewrites the document's facts. Consolidation reached those same facts and observations
    without the document, in the opposite order, and the two deadlocked — on plain row locks,
    and on pgvectorscale through the DiskANN index's transaction-wide insert lock. So: first the
    documents of every memory this transaction can touch (the batch, the observations, and the
    observations' current sources, which the update, merge and dedup paths re-check), sorted;
    then all those memories in one sorted statement. A retain of one of those documents is
    waited for while nothing is held, and once this returns the transaction waits on no retain,
    so it never holds the index lock while waiting. ``FOR SHARE`` on the documents still
    conflicts with a retain's document lock but lets parallel consolidation groups share one.
    """
    from .reads import get_memories  # noqa: PLC0415

    observations = (
        await get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=observation_ids)
        if observation_ids
        else []
    )
    unit_ids = sorted(
        {
            uuid.UUID(str(u))
            for u in [*fact_ids, *observation_ids, *(s for o in observations for s in o.source_memory_ids)]
        }
    )
    if not unit_ids:
        return
    await conn.execute(
        f"SELECT d.id FROM {fq_table('documents')} d WHERE d.bank_id = $1 AND d.id IN "
        f"(SELECT mu.document_id FROM {fq_table('memory_units')} mu WHERE mu.bank_id = $1 AND mu.id = ANY($2::uuid[])) "
        "ORDER BY d.id FOR SHARE",
        bank_id,
        unit_ids,
    )
    await conn.execute(
        f"SELECT id FROM {fq_table('memory_units')} WHERE bank_id = $1 AND id = ANY($2::uuid[]) "
        "ORDER BY id FOR NO KEY UPDATE",
        bank_id,
        unit_ids,
    )


async def lock_observation_tags(
    *, conn, fq_table: Callable[[str], str], bank_id: str, observation_id: str
) -> list[str] | None:
    """The observation's current tags, row-locked until the transaction ends; None if it is gone.

    ``FOR NO KEY UPDATE`` still lets other rows take FK references to it. Callers lock the
    source rows first (``lock_live_memory_ids``), keeping the sources-before-observation order.
    """
    row = await conn.fetchrow(
        f"SELECT tags FROM {fq_table('memory_units')} WHERE id = $1 AND bank_id = $2 FOR NO KEY UPDATE",
        uuid.UUID(observation_id),
        bank_id,
    )
    return None if row is None else list(row["tags"] or [])


async def memories_changed_since(
    *, conn, fq_table: Callable[[str], str], bank_id: str, read_at: dict[str, datetime]
) -> list[str]:
    """Ids in ``read_at`` whose ``updated_at`` moved since it was read, ``FOR SHARE``-locked."""
    rows = await conn.fetch(
        f"SELECT id, updated_at FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[]) AND bank_id = $2 FOR SHARE",
        [uuid.UUID(mid) for mid in read_at],
        bank_id,
    )
    return [str(r["id"]) for r in rows if r["updated_at"] != read_at[str(r["id"])]]


async def count_observations_with_tags(*, conn, fq_table: Callable[[str], str], bank_id: str, tags: list[str]) -> int:
    """Observations whose tags contain every one of ``tags``."""
    return await conn.fetchval(
        f"SELECT COUNT(*) FROM {fq_table('memory_units')} "
        f"WHERE bank_id = $1 AND fact_type = 'observation' AND tags @> $2::varchar[]",
        bank_id,
        tags,
    )


async def fold_sources_into_observation(
    *,
    conn,
    fq_table: Callable[[str], str],
    observation_id: str,
    expected_text: str,
    merged_text: str,
    source_memory_ids: list[uuid.UUID],
    bounds: _TemporalBounds,
) -> bool:
    """Fold ``source_memory_ids`` and ``merged_text`` into the observation, widening its dates.

    Keeps the observation's existing embedding (the merged text is >= threshold similar, so it
    stays representative and avoids a re-embed + a dialect-specific vector UPDATE). False when
    the row vanished or was rewritten since ``expected_text`` was read.

    Records the durable change events transactionally with the fold: a TEXT_CHANGE
    when the text moved, a FOLD_IN when a new source id appeared — a fold-in
    with unchanged text still records the source addition, and one with
    neither records nothing.
    """
    # RETURNING-gate on the twin's probe-time text so a concurrent
    # survivor rewrite during the connection-free LLM window can't be clobbered.
    search_vector_clause = _search_vector_clause("$1")
    # The pre-fold sources, read under the same text gate the UPDATE uses, so
    # the FOLD_IN record names exactly the ids this fold added.
    current = await conn.fetchrow(
        f"SELECT bank_id, source_memory_ids, updated_at FROM {fq_table('memory_units')} WHERE id = $1::uuid AND text = $2",
        uuid.UUID(observation_id),
        expected_text,
    )
    known_source_ids = {str(s) for s in (current["source_memory_ids"] or [])} if current is not None else set()
    added_source_ids = [str(s) for s in dict.fromkeys(source_memory_ids) if str(s) not in known_source_ids]
    folded = await conn.fetchval(
        f"""
            UPDATE {fq_table("memory_units")}
            SET text = $1,
                source_memory_ids = (SELECT array_agg(DISTINCT e) FROM unnest(source_memory_ids || $2::uuid[]) e),
                proof_count = (SELECT count(DISTINCT e) FROM unnest(source_memory_ids || $2::uuid[]) e),
                event_date = LEAST(event_date, COALESCE($5, event_date)),
                occurred_start = LEAST(occurred_start, COALESCE($6, occurred_start)),
                occurred_end = GREATEST(occurred_end, COALESCE($7, occurred_end)),
                mentioned_at = GREATEST(mentioned_at, COALESCE($8, mentioned_at)),
                updated_at = now(){search_vector_clause}
            WHERE id = $3::uuid AND text = $4
            RETURNING id
            """,
        merged_text,
        source_memory_ids,
        uuid.UUID(observation_id),
        expected_text,
        bounds.event_date,
        bounds.occurred_start,
        bounds.occurred_end,
        bounds.mentioned_at,
    )
    if folded is not None and current is not None:
        version_token = _row_version_token(current)
        if _text_changed(expected_text, merged_text):
            await record_change_event(
                conn=conn,
                fq_table=fq_table,
                bank_id=current["bank_id"],
                subject_id=observation_id,
                event_type=TEXT_CHANGE,
                event_id=stable_event_id(
                    bank_id=current["bank_id"],
                    subject_id=observation_id,
                    event_type=TEXT_CHANGE,
                    version_token=version_token,
                    before_text=expected_text,
                    after_text=merged_text,
                ),
                before_text=expected_text,
                after_text=merged_text,
            )
        if added_source_ids:
            await record_change_event(
                conn=conn,
                fq_table=fq_table,
                bank_id=current["bank_id"],
                subject_id=observation_id,
                event_type=FOLD_IN,
                event_id=stable_event_id(
                    bank_id=current["bank_id"],
                    subject_id=observation_id,
                    event_type=FOLD_IN,
                    version_token=version_token,
                    before_text=expected_text,
                    after_text=merged_text,
                    detail={"added_source_ids": added_source_ids},
                ),
                before_text=expected_text,
                after_text=merged_text,
                detail={"added_source_ids": added_source_ids},
            )
    return folded is not None


async def fold_observation_into_twin(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    observation_id: str,
    observation_text: str,
    twin_id: str,
    twin_text: str,
    merged_text: str,
) -> bool:
    """Fold the observation's live sources and dates into its twin, with ``merged_text``.

    False when there is nothing to fold (the observation or all its sources are gone) or either
    row was rewritten since it was read. The caller deletes the folded observation.
    """
    # Snapshot the updated row's sources with a PLAIN read (no FOR UPDATE). Lock order
    # must be sources-before-observation: the liveness check below takes
    # FOR SHARE on the SOURCE rows first, then the fold UPDATE locks the observation
    # rows -- the same order as the create fold and the normal write paths
    # (insert_observation / rewrite_observation). Locking the observation
    # here (FOR UPDATE) would invert that against the invalidation path and deadlock.
    updated_row = await conn.fetchrow(
        f"""
            SELECT source_memory_ids
            FROM {fq_table("memory_units")}
            WHERE id = $1::uuid AND text = $2
            """,
        uuid.UUID(observation_id),
        observation_text,
    )
    if updated_row is None:
        return False
    sources = list(updated_row["source_memory_ids"] or [])
    if not sources:
        return False
    live = await lock_live_memory_ids(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=sources)
    live_u_sources = [mid for mid in sources if str(mid) in live]
    if not live_u_sources:
        return False
    # The twin's pre-fold sources, under the same text gate the UPDATE uses,
    # so the FOLD_IN record names exactly the ids this fold added to it.
    twin = await conn.fetchrow(
        f"SELECT source_memory_ids, updated_at FROM {fq_table('memory_units')} WHERE id = $1::uuid AND text = $2",
        uuid.UUID(twin_id),
        twin_text,
    )
    known_twin_source_ids = {str(s) for s in (twin["source_memory_ids"] or [])} if twin is not None else set()
    added_to_twin = [str(s) for s in dict.fromkeys(live_u_sources) if str(s) not in known_twin_source_ids]
    # RETURNING-gate on both rows' probe-time text so a survivor/updated rewrite during
    # the connection-free LLM window can't be clobbered or fold a stale row.
    search_vector_clause = _search_vector_clause("$1")
    folded = await conn.fetchval(
        f"""
            UPDATE {fq_table("memory_units")} t
            SET text = $1,
                source_memory_ids = (
                    SELECT array_agg(DISTINCT e) FROM unnest(t.source_memory_ids || $6::uuid[]) e
                ),
                proof_count = (
                    SELECT count(DISTINCT e) FROM unnest(t.source_memory_ids || $6::uuid[]) e
                ),
                event_date = LEAST(t.event_date, COALESCE(u.event_date, t.event_date)),
                occurred_start = LEAST(t.occurred_start, COALESCE(u.occurred_start, t.occurred_start)),
                occurred_end = GREATEST(t.occurred_end, COALESCE(u.occurred_end, t.occurred_end)),
                mentioned_at = GREATEST(t.mentioned_at, COALESCE(u.mentioned_at, t.mentioned_at)),
                updated_at = now(){search_vector_clause}
            FROM {fq_table("memory_units")} u
            WHERE t.id = $2::uuid AND u.id = $3::uuid AND t.text = $4 AND u.text = $5
            RETURNING t.id
            """,
        merged_text,
        uuid.UUID(twin_id),
        uuid.UUID(observation_id),
        twin_text,
        observation_text,
        live_u_sources,
    )
    # None: twin or updated row vanished during the LLM window — the caller keeps the
    # updated row as a distinct observation instead of deleting it unfolded.
    if folded is not None and twin is not None:
        version_token = _row_version_token(twin)
        if _text_changed(twin_text, merged_text):
            await record_change_event(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                subject_id=twin_id,
                event_type=TEXT_CHANGE,
                event_id=stable_event_id(
                    bank_id=bank_id,
                    subject_id=twin_id,
                    event_type=TEXT_CHANGE,
                    version_token=version_token,
                    before_text=twin_text,
                    after_text=merged_text,
                ),
                before_text=twin_text,
                after_text=merged_text,
            )
        if added_to_twin:
            await record_change_event(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                subject_id=twin_id,
                event_type=FOLD_IN,
                event_id=stable_event_id(
                    bank_id=bank_id,
                    subject_id=twin_id,
                    event_type=FOLD_IN,
                    version_token=version_token,
                    before_text=twin_text,
                    after_text=merged_text,
                    detail={"added_source_ids": added_to_twin},
                ),
                before_text=twin_text,
                after_text=merged_text,
                detail={"added_source_ids": added_to_twin},
            )
    return folded is not None


async def rewrite_observation(
    *,
    conn,
    fq_table: Callable[[str], str],
    observation_id: str,
    text: str,
    embedding: str | None,
    source_memory_ids: list[uuid.UUID],
    tags: list[str],
    bounds: _TemporalBounds,
) -> bool:
    """Rewrite an observation in place, widening its dates by ``bounds``. False if the row is gone.

    A changed text records a durable TEXT_CHANGE event on the
    caller's transaction; a new source id records a FOLD_IN even when the
    text is unchanged (this UPDATE replaces the
    whole source set, so it is a fold-in site like the fold paths); neither
    records nothing.
    """
    search_vector_clause = _search_vector_clause("$1")
    # The stored text and sources before this write: the "before" of the
    # server-side compare (inventory H1 - before and after are in this scope)
    # and the pre-write source set the FOLD_IN record diffs against. Read
    # first because the UPDATE below overwrites both.
    current = await conn.fetchrow(
        f"SELECT bank_id, text, source_memory_ids, updated_at FROM {fq_table('memory_units')} WHERE id = $1",
        uuid.UUID(observation_id),
    )
    before_text = current["text"] if current is not None else None
    known_source_ids = {str(s) for s in (current["source_memory_ids"] or [])} if current is not None else set()
    added_source_ids = [str(s) for s in dict.fromkeys(source_memory_ids) if str(s) not in known_source_ids]
    # Unlike the dedup folds this statement also runs on Oracle, where LEAST/GREATEST
    # return NULL as soon as ANY argument is NULL (PostgreSQL ignores NULL arguments).
    # The inner COALESCE covers a NULL *parameter*; the outer one covers a NULL
    # *column* — an observation with no occurred interval yet, which is precisely the
    # #3477 case. Without it Oracle would compute LEAST(NULL, <source date>) = NULL and
    # silently drop the date it was told to inherit. Keep the inner
    # ``COALESCE($n, col)`` spelled exactly like this: the Oracle driver shim keys its
    # TIMESTAMP-TZ input-size hint off that pattern (db/oracle.py::_apply_clob_input_sizes),
    # and a NULL parameter binds as VARCHAR2 (ORA-00932) without it.
    updated_rows = await conn.execute_rows_affected(
        f"""
            UPDATE {fq_table("memory_units")}
            SET text = $1,
                embedding = $2::vector,
                source_memory_ids = $3,
                proof_count = $4,
                tags = $10,
                updated_at = now(),
                event_date = COALESCE(LEAST(event_date, COALESCE($6, event_date)), $6),
                occurred_start = COALESCE(LEAST(occurred_start, COALESCE($7, occurred_start)), $7),
                occurred_end = COALESCE(GREATEST(occurred_end, COALESCE($8, occurred_end)), $8),
                mentioned_at = COALESCE(GREATEST(mentioned_at, COALESCE($9, mentioned_at)), $9){search_vector_clause}
            WHERE id = $5
            """,
        text,
        embedding,
        source_memory_ids,
        len(source_memory_ids),
        uuid.UUID(observation_id),
        bounds.event_date,
        bounds.occurred_start,
        bounds.occurred_end,
        bounds.mentioned_at,
        tags,
    )
    if updated_rows != 0 and before_text is not None:
        version_token = _row_version_token(current)
        if _text_changed(before_text, text):
            await record_change_event(
                conn=conn,
                fq_table=fq_table,
                bank_id=current["bank_id"],
                subject_id=observation_id,
                event_type=TEXT_CHANGE,
                event_id=stable_event_id(
                    bank_id=current["bank_id"],
                    subject_id=observation_id,
                    event_type=TEXT_CHANGE,
                    version_token=version_token,
                    before_text=before_text,
                    after_text=text,
                ),
                before_text=before_text,
                after_text=text,
            )
        if added_source_ids:
            await record_change_event(
                conn=conn,
                fq_table=fq_table,
                bank_id=current["bank_id"],
                subject_id=observation_id,
                event_type=FOLD_IN,
                event_id=stable_event_id(
                    bank_id=current["bank_id"],
                    subject_id=observation_id,
                    event_type=FOLD_IN,
                    version_token=version_token,
                    before_text=before_text,
                    after_text=text,
                    detail={"added_source_ids": added_source_ids},
                ),
                before_text=before_text,
                after_text=text,
                detail={"added_source_ids": added_source_ids},
            )
    return updated_rows != 0


async def insert_observation(
    *,
    conn,
    ops,
    fq_table: Callable[[str], str],
    bank_id: str,
    observation_id: uuid.UUID,
    text: str,
    embedding: str | None,
    source_memory_ids: list[uuid.UUID],
    tags: list[str],
    event_date: datetime | None,
    occurred_start: datetime | None,
    occurred_end: datetime | None,
    mentioned_at: datetime | None,
) -> str:
    """Insert a new observation as a `memory_units` row (and its Oracle source postings)."""
    # pg_textsearch, pgroonga, pg_search and Oracle index the text column itself: no search_vector.
    sv_expr = observation_search_vector_expr("$3")
    sv_col, sv_val = (", search_vector", f", {sv_expr}") if sv_expr else ("", "")
    query = f"""
            INSERT INTO {fq_table("memory_units")} (
                id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                tags, event_date, occurred_start, occurred_end, mentioned_at{sv_col}
            )
            VALUES ($1, $2, $3, 'observation', $4::vector, $11, $5, $6, $7, $8, $9, $10{sv_val})
            RETURNING id
        """

    row = await conn.fetchrow(
        query,
        observation_id,
        bank_id,
        text,
        embedding,
        source_memory_ids,
        tags,
        event_date,
        occurred_start,
        occurred_end,
        mentioned_at,
        len(source_memory_ids),  # proof_count: the caller passes each live source once
    )

    # Populate observation_sources junction table (Oracle only — PG uses native array ops).
    if ops.uses_observation_sources_table and source_memory_ids:
        await conn.executemany(
            f"""
                INSERT INTO {fq_table("observation_sources")} (observation_id, source_id)
                VALUES ($1, $2)
                ON CONFLICT (observation_id, source_id) DO NOTHING
                """,
            [(observation_id, sid) for sid in dict.fromkeys(source_memory_ids)],
        )
    return str(row["id"])


async def delete_observation(*, conn, fq_table: Callable[[str], str], bank_id: str, observation_id: str) -> None:
    """Delete one observation row (the caller drops its history)."""
    # Its change records go with it.
    from .engine_curation import purge_change_events_for_subjects  # noqa: PLC0415

    await purge_change_events_for_subjects(conn, fq_table, bank_id, [observation_id])
    await conn.execute(
        f"DELETE FROM {fq_table('memory_units')} WHERE id = $1 AND bank_id = $2 AND fact_type = 'observation'",
        uuid.UUID(observation_id),
        bank_id,
    )
