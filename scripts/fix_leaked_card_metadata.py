#!/usr/bin/env python3
"""Remove LEAKED catalogue-card metadata from readable prose chunks.

Old card round-trips left ``full_card`` (str) and ``is_catalogue_card=true`` on
real prose chunks. ``_select_extraction_text_points`` now ignores those flags
(content wins), but the stale metadata still makes ``count_source_chunk_points``
under-count the source. This maintenance op rewrites the affected chunks in
place (NO re-parse, NO re-embed):

  * readable prose chunks -> ``full_card`` removed, ``is_catalogue_card`` and
    ``is_catalogue_card_segment`` set to ``false``;
  * real card segments (content starts with ``# CATALOGUE CARD``) and binary
    content -> left untouched.

Usage (inside the Cat container):
    python /app/cat/plugins/cat_graphrag/scripts/fix_leaked_card_metadata.py --dry-run
    python /app/cat/plugins/cat_graphrag/scripts/fix_leaked_card_metadata.py --yes
    python /app/cat/plugins/cat_graphrag/scripts/fix_leaked_card_metadata.py \
        --show-source consigli.pdf --show-source 11_Capitolo_1.pdf --dry-run

Exit codes:
    0 = write completed (only with ``--yes``)
    1 = partial failure (one or more chunk rewrites/verification failed)
    2 = aborted (dry-run plan printed, or ``--yes`` missing)

Import safety: this file lives inside a plugin folder, so it has ZERO top-level
side effects (stdlib only; every driver/plugin import happens inside main).
"""

import argparse
import asyncio
import json
import os
import sys

DEFAULT_URI = "bolt://neo4j:7687"
DEFAULT_USER = "neo4j"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove leaked full_card/is_catalogue_card metadata from readable "
            "prose chunks. Dry-run by default; --yes performs the rewrite."
        )
    )
    parser.add_argument("--uri", default=os.environ.get("NEO4J_URI", DEFAULT_URI))
    parser.add_argument("--user", default=os.environ.get("NEO4J_USER", DEFAULT_USER))
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD")
        or os.environ.get("NEO4J_AUTH", "").partition("/")[2]
        or None,
    )
    parser.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    parser.add_argument(
        "--show-source",
        action="append",
        default=[],
        metavar="SOURCE",
        help="Print the extraction filter result for this source (repeatable).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only (default).")
    parser.add_argument("--yes", action="store_true", help="Perform the rewrite.")
    return parser.parse_args()


def _load_filter_helpers():
    """Import the plugin's content checks WITHOUT re-implementing them."""
    try:
        from cat.plugins.cat_graphrag.graphrag_handler import (  # noqa: PLC0415
            GraphRAGHandler,
        )

        return GraphRAGHandler
    except ImportError:
        pass
    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)
    from graphrag_handler import GraphRAGHandler  # noqa: PLC0415,E402

    return GraphRAGHandler


class _Point:
    __slots__ = ("id", "payload")

    def __init__(self, pid, content, metadata):
        self.id = pid
        self.payload = {"page_content": content or "", "metadata": metadata or {}}


def _readable_prose(handler, content: str) -> bool:
    content = content or ""
    if not content.strip():
        return False
    if handler._looks_binary_content(content):
        return False
    if handler._looks_like_catalogue_card_content(content):
        return False
    return any(ch.isalnum() for ch in content)


async def _fetch_leaked_ids(session):
    """Read leaked candidate IDs only, paginated.

    ``full_card`` is a large string duplicated into every chunk's metadata, so
    returning metadata/content for all candidates at once exhausts the Neo4j
    transaction memory pool. IDs are tiny; content and metadata are fetched one
    chunk at a time below.
    """
    rows = []
    skip = 0
    page = 500
    while True:
        query = """
        MATCH (d:Document)
        WHERE d.metadata CONTAINS '"full_card"'
        RETURN d.id AS id, d.tenant_id AS tenant_id
        ORDER BY d.tenant_id, d.id
        SKIP $skip LIMIT $page
        """
        result = await session.run(query, skip=skip, page=page)
        batch = [
            {"id": record["id"], "tenant_id": record["tenant_id"]}
            async for record in result
        ]
        rows.extend(batch)
        if len(batch) < page:
            break
        skip += page
    return rows


async def _fetch_chunk(session, row):
    query = """
    MATCH (d:Document {id: $id})
    WHERE d.tenant_id = $tenant_id
    RETURN d.content AS content, d.metadata AS metadata
    """
    result = await session.run(query, id=row["id"], tenant_id=row["tenant_id"])
    record = await result.single()
    if not record:
        return "", {}
    raw = record["metadata"]
    try:
        metadata = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except (TypeError, ValueError):
        metadata = {}
    return record["content"] or "", metadata


async def _fetch_source(session, source: str):
    query = """
    MATCH (d:Document)
    WHERE d.metadata CONTAINS $source_frag
    RETURN d.id AS id, d.content AS content, d.metadata AS metadata
    ORDER BY d.metadata
    """
    result = await session.run(query, source_frag=f'"source": {json.dumps(source)}')
    points = []
    async for record in result:
        raw = record["metadata"]
        try:
            metadata = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (TypeError, ValueError):
            metadata = {}
        points.append(_Point(record["id"], record["content"], metadata))
    return points


def _show_source(handler, points, source: str) -> None:
    selected = handler._select_extraction_text_points(points)
    chosen = {(p.payload or {}).get("page_content", "") for p in selected}
    print(f"\n--- filter for {source!r}: {len(points)} point(s), "
          f"{len(selected)} selected ---")
    for p in points:
        content = (p.payload or {}).get("page_content", "") or ""
        meta = (p.payload or {}).get("metadata", {}) or {}
        kept = content in chosen
        print(
            f"  {'KEEP' if kept else 'DROP'} id={p.id} len={len(content)} "
            f"full_card={bool(meta.get('full_card'))} "
            f"is_catalogue_card={meta.get('is_catalogue_card')!r} "
            f"head={content[:40]!r}"
        )


async def _rewrite(session, row) -> bool:
    metadata = dict(row["metadata"])
    metadata.pop("full_card", None)
    metadata["is_catalogue_card"] = False
    if "is_catalogue_card_segment" in metadata:
        metadata["is_catalogue_card_segment"] = False
    query = """
    MATCH (d:Document {id: $id})
    WHERE d.tenant_id = $tenant_id
    SET d.metadata = $metadata
    RETURN d.id AS id
    """
    result = await session.run(
        query,
        id=row["id"],
        tenant_id=row["tenant_id"],
        metadata=json.dumps(metadata),
    )
    return await result.single() is not None


async def _run(args) -> int:
    from neo4j import AsyncGraphDatabase  # noqa: PLC0415

    handler = _load_filter_helpers()
    driver = AsyncGraphDatabase.driver(args.uri, auth=(args.user, args.password))
    exit_code = 0
    try:
        async with driver.session(database=args.database) as session:
            if args.show_source:
                for source in args.show_source:
                    points = await _fetch_source(session, source)
                    _show_source(handler, points, source)

            rows = await _fetch_leaked_ids(session)
            for row in rows:
                row["content"], row["metadata"] = await _fetch_chunk(session, row)
            by_source: dict[str, list[dict]] = {}
            for row in rows:
                by_source.setdefault(row["metadata"].get("source") or "<none>", []).append(row)

            prose = [r for r in rows if _readable_prose(handler, r["content"])]
            prose_ids = {id(r) for r in prose}

            print(f"\n=== leaked full_card metadata: {len(rows)} chunk(s) "
                  f"across {len(by_source)} source(s) ===")
            for source, items in sorted(by_source.items()):
                clean = sum(1 for r in items if id(r) in prose_ids)
                print(f"  {source}: {len(items)} leaked, {clean} readable-prose")
            print(f"  readable prose to clean: {len(prose)}")
            print(f"  real card / binary left untouched: {len(rows) - len(prose)}")

            if not args.yes:
                print("\n(dry-run: exiting 2, nothing was written)")
                return 2

            written = 0
            failed = 0
            for row in prose:
                try:
                    ok = await _rewrite(session, row)
                    if ok:
                        written += 1
                    else:
                        failed += 1
                        print(f"  FAILED (not matched) id={row['id']}")
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"  FAILED id={row['id']}: {exc}")
            print(f"\n=== rewrite: {written} chunk(s) cleaned, {failed} failed ===")
            if failed:
                exit_code = 1

            if args.show_source:
                print("\n=== filter AFTER cleanup ===")
                for source in args.show_source:
                    points = await _fetch_source(session, source)
                    _show_source(handler, points, source)
    finally:
        await driver.close()
    return exit_code


def main() -> None:
    args = _parse_args()
    if not args.password:
        print("ERROR: no Neo4j password (pass --password or set NEO4J_PASSWORD)")
        raise SystemExit(2)
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
