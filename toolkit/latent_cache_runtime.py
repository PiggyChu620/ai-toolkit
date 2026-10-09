import atexit
import os
import sqlite3
import threading
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

from safetensors.torch import load as load_safetensors, save as save_safetensors


LATENT_INDEX_DB_NAME = "_latent_shards.sqlite3"


def get_latent_index_db_path(latent_dir: str) -> str:
    return os.path.join(latent_dir, LATENT_INDEX_DB_NAME)


_sqlite_local = threading.local()
_sqlite_init_lock = threading.Lock()


def _get_connection_cache() -> Dict[str, sqlite3.Connection]:
    cache = getattr(_sqlite_local, "connections", None)
    if cache is None:
        cache = {}
        _sqlite_local.connections = cache
    return cache


def _connect_index(latent_dir: str) -> sqlite3.Connection:
    db_path = get_latent_index_db_path(latent_dir)
    cache = _get_connection_cache()
    connection = cache.get(db_path)
    if connection is not None:
        return connection

    os.makedirs(latent_dir, exist_ok=True)
    connection = sqlite3.connect(
        db_path,
        timeout=30.0,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    with _sqlite_init_lock:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                cache_key TEXT PRIMARY KEY,
                shard TEXT NOT NULL,
                key_prefix TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_latents (
                cache_key TEXT PRIMARY KEY,
                bucket_key TEXT NOT NULL,
                state BLOB NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_shard ON entries(shard)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_latents_bucket ON pending_latents(bucket_key)"
        )
        connection.commit()
    cache[db_path] = connection
    return connection


def _get_meta(connection: sqlite3.Connection, key: str, default=None):
    row = connection.execute(
        "SELECT value FROM meta WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return default
    value = row["value"]
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return value


def _serialize_pending_state(state_dict: OrderedDict) -> bytes:
    normalized_state = OrderedDict()
    for key, value in state_dict.items():
        normalized_state[key] = value.detach().to("cpu").contiguous()
    return save_safetensors(normalized_state)


def _deserialize_pending_state(blob: bytes) -> OrderedDict:
    state_dict = load_safetensors(blob)
    ordered = OrderedDict()
    for key in ("latent", "first_frame_latent", "audio_latent", "first_frame_condition"):
        if key in state_dict:
            ordered[key] = state_dict[key]
    for key, value in state_dict.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def load_latent_index_manifest(latent_dir: str) -> Optional[dict]:
    db_path = get_latent_index_db_path(latent_dir)
    if not os.path.exists(db_path):
        return None

    connection = _connect_index(latent_dir)
    rows = connection.execute(
        "SELECT cache_key, shard, key_prefix FROM entries"
    ).fetchall()
    return {
        "version": _get_meta(connection, "version", 1),
        "shard_size": _get_meta(connection, "shard_size", 1),
        "next_shard_index": _get_meta(connection, "next_shard_index", 0),
        "items": {
            row["cache_key"]: {
                "shard": row["shard"],
                "key_prefix": row["key_prefix"],
            }
            for row in rows
        },
    }


def rewrite_latent_index_manifest(latent_dir: str, manifest: dict):
    connection = _connect_index(latent_dir)
    items = manifest.get("items", {})
    with connection:
        connection.execute("DELETE FROM entries")
        connection.executemany(
            """
            INSERT INTO entries (cache_key, shard, key_prefix)
            VALUES (?, ?, ?)
            """,
            [
                (
                    cache_key,
                    entry.get("shard", ""),
                    entry.get("key_prefix", cache_key),
                )
                for cache_key, entry in items.items()
                if entry.get("shard")
            ],
        )
        connection.executemany(
            """
            INSERT INTO meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            [
                ("version", str(manifest.get("version", 1))),
                ("shard_size", str(manifest.get("shard_size", 1))),
                ("next_shard_index", str(manifest.get("next_shard_index", 0))),
            ],
        )


def update_latent_index_entries(
        latent_dir: str,
        entries: List[Tuple[str, str, str]],
        *,
        version: Optional[int] = None,
        shard_size: Optional[int] = None,
        next_shard_index: Optional[int] = None,
):
    if len(entries) == 0 and version is None and shard_size is None and next_shard_index is None:
        return

    connection = _connect_index(latent_dir)
    with connection:
        if len(entries) > 0:
            connection.executemany(
                """
                INSERT INTO entries (cache_key, shard, key_prefix)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    shard = excluded.shard,
                    key_prefix = excluded.key_prefix
                """,
                entries,
            )
        meta_updates = []
        if version is not None:
            meta_updates.append(("version", str(version)))
        if shard_size is not None:
            meta_updates.append(("shard_size", str(shard_size)))
        if next_shard_index is not None:
            meta_updates.append(("next_shard_index", str(next_shard_index)))
        if len(meta_updates) > 0:
            connection.executemany(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                meta_updates,
            )


def lookup_latent_index_entry(latent_dir: str, cache_key: str) -> Optional[dict]:
    db_path = get_latent_index_db_path(latent_dir)
    if not os.path.exists(db_path):
        return None

    connection = _connect_index(latent_dir)
    row = connection.execute(
        """
        SELECT shard, key_prefix
        FROM entries
        WHERE cache_key = ?
        """,
        (cache_key,),
    ).fetchone()
    if row is None:
        return None
    return {
        "shard": row["shard"],
        "key_prefix": row["key_prefix"],
    }


def upsert_pending_latent_states(
        latent_dir: str,
        pending_entries: List[Tuple[str, str, OrderedDict]],
):
    if len(pending_entries) == 0:
        return

    connection = _connect_index(latent_dir)
    serialized_entries = [
        (cache_key, bucket_key, sqlite3.Binary(_serialize_pending_state(state_dict)))
        for cache_key, bucket_key, state_dict in pending_entries
    ]
    with connection:
        connection.executemany(
            """
            INSERT INTO pending_latents (cache_key, bucket_key, state)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                bucket_key = excluded.bucket_key,
                state = excluded.state
            """,
            serialized_entries,
        )


def has_pending_latent_state(latent_dir: str, cache_key: str) -> bool:
    db_path = get_latent_index_db_path(latent_dir)
    if not os.path.exists(db_path):
        return False

    connection = _connect_index(latent_dir)
    row = connection.execute(
        """
        SELECT 1
        FROM pending_latents
        WHERE cache_key = ?
        """,
        (cache_key,),
    ).fetchone()
    return row is not None


def load_pending_latent_state(latent_dir: str, cache_key: str) -> Optional[OrderedDict]:
    db_path = get_latent_index_db_path(latent_dir)
    if not os.path.exists(db_path):
        return None

    connection = _connect_index(latent_dir)
    row = connection.execute(
        """
        SELECT state
        FROM pending_latents
        WHERE cache_key = ?
        """,
        (cache_key,),
    ).fetchone()
    if row is None:
        return None
    return _deserialize_pending_state(row["state"])


def remove_pending_latent_states(latent_dir: str, cache_keys: Iterable[str]):
    cache_key_list = list(cache_keys)
    if len(cache_key_list) == 0:
        return

    connection = _connect_index(latent_dir)
    with connection:
        connection.executemany(
            "DELETE FROM pending_latents WHERE cache_key = ?",
            [(cache_key,) for cache_key in cache_key_list],
        )


def clear_pending_latent_states(latent_dir: str):
    db_path = get_latent_index_db_path(latent_dir)
    if not os.path.exists(db_path):
        return

    connection = _connect_index(latent_dir)
    with connection:
        connection.execute("DELETE FROM pending_latents")


def close_latent_cache_connections():
    cache = getattr(_sqlite_local, "connections", None)
    if cache is None:
        return
    for connection in cache.values():
        try:
            connection.close()
        except Exception:
            pass
    cache.clear()


atexit.register(close_latent_cache_connections)
