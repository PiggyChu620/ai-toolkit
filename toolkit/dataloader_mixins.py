import base64
import gc
import glob
import hashlib
import json
import math
import os
import random
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Dict, Union
import traceback

import cv2
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, SiglipImageProcessor

from toolkit.audio.preserve_pitch import time_stretch_preserve_pitch
from toolkit.basic import flush, value_map
from toolkit.buckets import get_bucket_for_image_size, get_resolution
from toolkit.cache_paths import get_cache_dir_candidates, get_cache_file_candidates
from toolkit.config_modules import ControlTypes
from toolkit.control_generator import ControlGenerator
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.pixtral_vision import PixtralVisionImagePreprocessorCompatible
from toolkit.latent_cache_runtime import (
    has_pending_latent_state,
    load_latent_index_manifest,
    load_pending_latent_state,
    lookup_latent_index_entry,
    remove_pending_latent_states,
    rewrite_latent_index_manifest,
    upsert_pending_latent_states,
    update_latent_index_entries,
)
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.prompt_utils import (
    PromptEmbeds,
    concat_prompt_embeds,
    inject_trigger_into_prompt,
    split_prompt_embeds,
)
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
from PIL.ImageOps import exif_transpose
import albumentations as A
from toolkit.print import print_acc
from toolkit.accelerator import get_accelerator
from torchvision.transforms import functional as TF

from toolkit.train_tools import get_torch_dtype

if TYPE_CHECKING:
    from toolkit.data_loader import AiToolkitDataset
    from toolkit.data_transfer_object.data_loader import FileItemDTO
    from toolkit.stable_diffusion_model import StableDiffusion

accelerator = get_accelerator()

# def get_associated_caption_from_img_path(img_path):
# https://demo.albumentations.ai/
class Augments:
    def __init__(self, **kwargs):
        self.method_name = kwargs.get('method', None)
        self.params = kwargs.get('params', {})

        # convert kwargs enums for cv2
        for key, value in self.params.items():
            if isinstance(value, str):
                # split the string
                split_string = value.split('.')
                if len(split_string) == 2 and split_string[0] == 'cv2':
                    if hasattr(cv2, split_string[1]):
                        self.params[key] = getattr(cv2, split_string[1].upper())
                    else:
                        raise ValueError(f"invalid cv2 enum: {split_string[1]}")


transforms_dict = {
    'ColorJitter': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    'RandomEqualize': transforms.RandomEqualize(p=0.2),
}

img_ext_list = ['.jpg', '.jpeg', '.png', '.webp']

LATENT_CACHE_SHARD_MANIFEST_NAME = "_latent_shards.json"
LATENT_CACHE_SHARD_VERSION = 1
LATENT_CACHE_SHARD_KEY_SEP = "__"
LATENT_AUX_TENSOR_KEYS = (
    "first_frame_latent",
    "audio_latent",
    "first_frame_condition",
)
latent_shard_manifest_cache = {}
_legacy_latent_manifest_migrated = set()


def _maybe_migrate_legacy_latent_manifest(latent_dir: str):
    """One-shot legacy-JSON-manifest probe for a latent dir; None afterwards.

    Callers reach here only when the indexed SQLite point lookup MISSED. The old
    fallback then called load_latent_shard_manifest(), which runs an unbounded
    `SELECT cache_key, shard, key_prefix FROM entries` and materialises every row
    into a dict -- to look up the same key the point lookup just failed to find,
    so it could never succeed. With a 116k-image dataset that is a full-table scan
    per miss, in every worker and again in the training thread, growing linearly
    with cache size.

    The JSON manifest only exists for pre-SQLite caches, and load_latent_shard_manifest
    migrates it into SQLite on first read, so one attempt per latent_dir per process is
    enough. Gating on "does the DB file exist yet" instead would be wrong: another
    writer can create the DB first and permanently strand a legacy JSON cache.
    """
    if latent_dir in _legacy_latent_manifest_migrated:
        return None
    _legacy_latent_manifest_migrated.add(latent_dir)
    return load_latent_shard_manifest(latent_dir)
lazy_latent_shard_buffer_lock = threading.Lock()
lazy_latent_shard_buffers = {}
lazy_latent_shard_sizes = {}
lazy_latent_manifest_buffers = {}

TEXT_EMBED_CACHE_SHARD_MANIFEST_NAME = "_text_embedding_shards.json"
TEXT_EMBED_CACHE_SHARD_VERSION = 1
TEXT_EMBED_CACHE_SHARD_KEY_SEP = "__"
text_embed_shard_manifest_cache = {}

MANIFEST_IO_RETRIES = 12
MANIFEST_IO_RETRY_DELAY = 0.05


def get_latent_shard_manifest_path(latent_dir: str) -> str:
    return os.path.join(latent_dir, LATENT_CACHE_SHARD_MANIFEST_NAME)


def get_latent_shard_bucket_key(file_item: 'FileItemDTO') -> str:
    return f"{file_item.crop_width}x{file_item.crop_height}"


def get_latent_shard_tensor_key(key_prefix: str, tensor_name: str) -> str:
    return f"{key_prefix}{LATENT_CACHE_SHARD_KEY_SEP}{tensor_name}"


def ensure_pending_latent_session_id() -> str:
    return "sqlite-pending-cache"


def _load_json_manifest_with_cache(manifest_path: str, cache_store: dict, expected_version: int):
    cached = cache_store.get(manifest_path)
    last_error = None

    def load_manifest_file(path: str):
        with open(path, "r", encoding="utf-8") as f:
            loaded_manifest = json.load(f)
        if loaded_manifest.get("version") != expected_version:
            return None
        return loaded_manifest

    for attempt in range(MANIFEST_IO_RETRIES):
        try:
            if not os.path.exists(manifest_path):
                cache_store.pop(manifest_path, None)
                return None

            stat = os.stat(manifest_path)
            cached = cache_store.get(manifest_path)
            if cached is not None:
                if cached["mtime_ns"] == stat.st_mtime_ns and cached["size"] == stat.st_size:
                    return cached["manifest"]

            manifest = load_manifest_file(manifest_path)
            if manifest is None:
                return None

            cache_store[manifest_path] = {
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
                "manifest": manifest,
            }
            return manifest
        except FileNotFoundError:
            return None
        except (PermissionError, json.JSONDecodeError, OSError) as e:
            last_error = e
            if attempt < MANIFEST_IO_RETRIES - 1:
                time.sleep(MANIFEST_IO_RETRY_DELAY)
                continue

    if cached is not None:
        return cached["manifest"]
    backup_path = f"{manifest_path}.bak"
    if os.path.exists(backup_path):
        try:
            manifest = load_manifest_file(backup_path)
            if manifest is not None:
                stat = os.stat(manifest_path) if os.path.exists(manifest_path) else os.stat(backup_path)
                cache_store[manifest_path] = {
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                    "manifest": manifest,
                }
                return manifest
        except (PermissionError, json.JSONDecodeError, OSError):
            pass
    if isinstance(last_error, PermissionError):
        return None
    return None


def _save_json_manifest_with_cache(manifest_path: str, manifest: dict, cache_store: dict):
    manifest_dir = os.path.dirname(manifest_path)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    temp_path = (
        f"{manifest_path}."
        f"{os.getpid()}."
        f"{threading.get_ident()}."
        f"{time.time_ns()}.tmp"
    )
    last_error = None
    try:
        for attempt in range(MANIFEST_IO_RETRIES):
            try:
                fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
                try:
                    offset = 0
                    while offset < len(manifest_bytes):
                        offset += os.write(fd, manifest_bytes[offset:])
                    os.fsync(fd)
                finally:
                    os.close(fd)
                temp_stat = os.stat(temp_path)
                if temp_stat.st_size != len(manifest_bytes):
                    raise OSError(
                        f"incomplete manifest write: {temp_stat.st_size} != {len(manifest_bytes)}"
                    )
                with open(temp_path, "rb") as f:
                    first_byte = f.read(1)
                    f.seek(-1, os.SEEK_END)
                    last_byte = f.read(1)
                if first_byte != b"{" or last_byte != b"}":
                    raise OSError("manifest write failed sanity check")
                os.replace(temp_path, manifest_path)
                backup_path = f"{manifest_path}.bak"
                backup_temp_path = (
                    f"{backup_path}."
                    f"{os.getpid()}."
                    f"{threading.get_ident()}."
                    f"{time.time_ns()}.tmp"
                )
                backup_fd = os.open(backup_temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
                try:
                    offset = 0
                    while offset < len(manifest_bytes):
                        offset += os.write(backup_fd, manifest_bytes[offset:])
                    os.fsync(backup_fd)
                finally:
                    os.close(backup_fd)
                os.replace(backup_temp_path, backup_path)
                stat = os.stat(manifest_path)
                cache_store[manifest_path] = {
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                    "manifest": manifest,
                }
                return
            except (PermissionError, json.JSONDecodeError, OSError) as e:
                last_error = e
                for cleanup_path in (temp_path, locals().get("backup_temp_path")):
                    if cleanup_path and os.path.exists(cleanup_path):
                        try:
                            os.remove(cleanup_path)
                        except OSError:
                            pass
                if attempt < MANIFEST_IO_RETRIES - 1:
                    time.sleep(MANIFEST_IO_RETRY_DELAY)
                    continue
                raise
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if last_error is not None:
        raise last_error


def load_latent_shard_manifest(latent_dir: str):
    manifest = load_latent_index_manifest(latent_dir)
    if manifest is not None:
        return manifest

    manifest_path = get_latent_shard_manifest_path(latent_dir)
    manifest = _load_json_manifest_with_cache(
        manifest_path,
        latent_shard_manifest_cache,
        LATENT_CACHE_SHARD_VERSION,
    )
    if manifest is not None:
        rewrite_latent_index_manifest(latent_dir, manifest)
    return manifest


def save_latent_shard_manifest(latent_dir: str, manifest: dict):
    rewrite_latent_index_manifest(latent_dir, manifest)


def save_latent_shard_manifest_meta(latent_dir: str, manifest: dict):
    update_latent_index_entries(
        latent_dir,
        [],
        version=manifest.get("version", LATENT_CACHE_SHARD_VERSION),
        shard_size=manifest.get("shard_size"),
        next_shard_index=manifest.get("next_shard_index"),
    )


def infer_next_latent_shard_index(manifest: dict) -> int:
    max_index = -1
    for entry in manifest.get("items", {}).values():
        shard_name = entry.get("shard", "")
        shard_root, shard_ext = os.path.splitext(shard_name)
        if shard_ext != ".safetensors":
            continue
        try:
            max_index = max(max_index, int(shard_root.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max_index + 1


def get_or_create_latent_shard_manifest(latent_dir: str, shard_size: int) -> dict:
    manifest = load_latent_shard_manifest(latent_dir)
    if manifest is None:
        return {
            "version": LATENT_CACHE_SHARD_VERSION,
            "shard_size": shard_size,
            "next_shard_index": 0,
            "items": {},
        }

    manifest["version"] = LATENT_CACHE_SHARD_VERSION
    manifest["shard_size"] = shard_size
    manifest.setdefault("items", {})
    if "next_shard_index" not in manifest:
        manifest["next_shard_index"] = infer_next_latent_shard_index(manifest)
    return manifest


def build_latent_shard_storage_entry(latent_dir: str, shard_name: str, key_prefix: str):
    return {
        "storage": "shard",
        "path": os.path.join(latent_dir, shard_name),
        "shard": shard_name,
        "key_prefix": key_prefix,
    }


def build_buffered_latent_storage_entry(latent_dir: str, key_prefix: str):
    return {
        "storage": "buffered",
        "latent_dir": latent_dir,
        "key_prefix": key_prefix,
    }


def get_bucket_key_from_latent_shard_name(shard_name: str):
    if not shard_name.endswith(".safetensors") or not shard_name.startswith("bucket_"):
        return None
    shard_root = shard_name[:-len(".safetensors")]
    bucket_key, sep, shard_index = shard_root[len("bucket_"):].rpartition("_")
    if sep == "" or not shard_index.isdigit() or bucket_key == "":
        return None
    return bucket_key


def _atomic_save_file(state_dict, path, metadata=None):
    # Write to a temp file in the same directory, then atomically os.replace() it
    # into place. A crash/BSOD mid-write then leaves only a stray .tmp file rather
    # than a truncated/corrupt cache file that would wedge every item in the shard.
    tmp_path = f"{path}.tmp-{os.getpid()}"
    try:
        if metadata is not None:
            save_file(state_dict, tmp_path, metadata=metadata)
        else:
            save_file(state_dict, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _save_legacy_latent_cache_file(file_item: 'FileItemDTO', state_dict: OrderedDict):
    latent_path = file_item.get_latent_path(recalculate=True)
    meta = get_meta_for_safetensors(file_item.get_latent_info_dict())
    os.makedirs(os.path.dirname(latent_path), exist_ok=True)
    _atomic_save_file(state_dict, latent_path, metadata=meta)
    file_item.set_latent_storage_entry({
        "storage": "legacy",
        "path": latent_path,
    })
    file_item.is_latent_cached = True


def _write_latent_shard_entries(
        latent_dir: str,
        bucket_key: str,
        shard_entries: List[tuple],
        manifest: dict,
):
    if len(shard_entries) == 0:
        return None

    shard_index = manifest.get("next_shard_index", 0)
    shard_name = f"bucket_{bucket_key}_{shard_index:05d}.safetensors"
    shard_path = os.path.join(latent_dir, shard_name)

    shard_state_dict = OrderedDict()
    for key_prefix, state_dict, _ in shard_entries:
        shard_state_dict[
            get_latent_shard_tensor_key(key_prefix, "latent")
        ] = state_dict["latent"]
        for tensor_name in LATENT_AUX_TENSOR_KEYS:
            if tensor_name in state_dict:
                shard_state_dict[
                    get_latent_shard_tensor_key(key_prefix, tensor_name)
                ] = state_dict[tensor_name]

    shard_meta = get_meta_for_safetensors(
        OrderedDict([
            ("latent_cache_shard_version", LATENT_CACHE_SHARD_VERSION),
            ("bucket", bucket_key),
            ("items", len(shard_entries)),
        ])
    )
    os.makedirs(latent_dir, exist_ok=True)
    _atomic_save_file(shard_state_dict, shard_path, metadata=shard_meta)

    index_entries = []
    for key_prefix, _, file_item in shard_entries:
        manifest["items"][key_prefix] = {
            "shard": shard_name,
            "key_prefix": key_prefix,
        }
        index_entries.append((key_prefix, shard_name, key_prefix))
        if file_item is not None:
            file_item.set_latent_storage_entry(
                build_latent_shard_storage_entry(latent_dir, shard_name, key_prefix)
            )
            file_item.is_latent_cached = True

    manifest["next_shard_index"] = shard_index + 1
    update_latent_index_entries(
        latent_dir,
        index_entries,
        version=manifest.get("version", LATENT_CACHE_SHARD_VERSION),
        shard_size=manifest.get("shard_size"),
        next_shard_index=manifest.get("next_shard_index"),
    )
    return shard_name


def _write_latent_shard_pairs(
        latent_dir: str,
        bucket_key: str,
        shard_pairs: List[tuple],
        manifest: dict,
):
    shard_entries = [
        (file_item.get_latent_cache_key(), state_dict, file_item)
        for file_item, state_dict in shard_pairs
    ]
    return _write_latent_shard_entries(latent_dir, bucket_key, shard_entries, manifest)


def flush_buffered_file_item_latents_to_disk(force=False):
    dirty_manifests = {}
    with lazy_latent_shard_buffer_lock:
        for buffer_key in list(lazy_latent_shard_buffers.keys()):
            pending_map = lazy_latent_shard_buffers.get(buffer_key)
            if not pending_map:
                continue

            latent_dir, bucket_key = buffer_key
            shard_size = max(1, lazy_latent_shard_sizes.get(buffer_key, 1))
            manifest = lazy_latent_manifest_buffers.get(latent_dir)
            if manifest is None:
                manifest = get_or_create_latent_shard_manifest(latent_dir, shard_size)
                lazy_latent_manifest_buffers[latent_dir] = manifest

            while len(pending_map) >= shard_size or (force and len(pending_map) > 0):
                item_count = shard_size if len(pending_map) >= shard_size else len(pending_map)
                shard_pairs = []
                removed_key_prefixes = []
                for _ in range(item_count):
                    key_prefix, pair = pending_map.popitem(last=False)
                    removed_key_prefixes.append(key_prefix)
                    shard_pairs.append(pair)
                _write_latent_shard_pairs(latent_dir, bucket_key, shard_pairs, manifest)
                remove_pending_latent_states(latent_dir, removed_key_prefixes)
                dirty_manifests[latent_dir] = manifest

            if len(pending_map) == 0:
                lazy_latent_shard_buffers.pop(buffer_key, None)
                lazy_latent_shard_sizes.pop(buffer_key, None)

        for latent_dir, manifest in dirty_manifests.items():
            save_latent_shard_manifest_meta(latent_dir, manifest)


def get_buffered_file_item_latent_count() -> int:
    with lazy_latent_shard_buffer_lock:
        return sum(len(pending_map) for pending_map in lazy_latent_shard_buffers.values())


def clear_buffered_file_item_latents_to_disk():
    with lazy_latent_shard_buffer_lock:
        lazy_latent_shard_buffers.clear()
        lazy_latent_shard_sizes.clear()
        lazy_latent_manifest_buffers.clear()


def _best_effort_remove_shard(path: str) -> bool:
    # On Windows, safetensors safe_open mmaps the shard and the OS won't release the
    # file handle until the mapping is garbage-collected, so a just-read shard can hit
    # PermissionError (WinError 5) on os.remove. Drop lingering refs via gc and retry;
    # if it's still locked (another process / AV scan), give up and leave a harmless
    # orphan rather than crashing startup (the manifest no longer references it).
    for attempt in range(4):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except (PermissionError, OSError):
            gc.collect()
            if attempt < 3:
                time.sleep(0.1 * (attempt + 1))
    return False


def compact_latent_shards(latent_dir: str, shard_size: int):
    shard_size = max(1, shard_size)
    if shard_size <= 1 or not os.path.isdir(latent_dir):
        return {
            "compacted": False,
            "moved_items": 0,
            "removed_shards": 0,
            "created_shards": 0,
        }

    manifest = load_latent_shard_manifest(latent_dir)
    if manifest is None:
        return {
            "compacted": False,
            "moved_items": 0,
            "removed_shards": 0,
            "created_shards": 0,
        }

    manifest["version"] = LATENT_CACHE_SHARD_VERSION
    manifest["shard_size"] = shard_size
    manifest.setdefault("items", {})
    manifest.setdefault("next_shard_index", infer_next_latent_shard_index(manifest))

    bucket_entries = {}
    shard_counts = {}
    for cache_key, shard_entry in manifest.get("items", {}).items():
        shard_name = shard_entry.get("shard")
        if not shard_name:
            continue
        bucket_key = get_bucket_key_from_latent_shard_name(shard_name)
        if bucket_key is None:
            continue
        key_prefix = shard_entry.get("key_prefix", cache_key)
        bucket_entries.setdefault(bucket_key, {}).setdefault(shard_name, []).append(
            (cache_key, key_prefix)
        )
        shard_counts[shard_name] = shard_counts.get(shard_name, 0) + 1

    moved_items = 0
    removed_shards = 0
    created_shards = 0
    did_compact = False

    bucket_items = bucket_entries.items()
    # A fragmented cache (many underfilled shards left by crash-interrupted caching)
    # can take minutes to rewrite. Show a progress bar so this one-time work doesn't
    # look like a hang (it otherwise runs silently).
    if len(bucket_entries) > 1:
        bucket_items = tqdm(
            bucket_items,
            total=len(bucket_entries),
            desc=f"Compacting latent cache {latent_dir}",
            unit="bucket",
        )
    for bucket_key, entries_by_shard in bucket_items:
        partial_shards = [
            shard_name
            for shard_name, shard_entries in entries_by_shard.items()
            if 0 < len(shard_entries) < shard_size and os.path.exists(os.path.join(latent_dir, shard_name))
        ]
        if len(partial_shards) <= 1:
            continue

        total_partial_items = sum(len(entries_by_shard[shard_name]) for shard_name in partial_shards)
        if len(partial_shards) <= math.ceil(total_partial_items / shard_size):
            continue

        compact_buffer = []
        created_for_bucket = 0

        for shard_name in partial_shards:
            shard_path = os.path.join(latent_dir, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as shard_file:
                available_keys = set(shard_file.keys())
                for cache_key, key_prefix in entries_by_shard[shard_name]:
                    latent_key = get_latent_shard_tensor_key(key_prefix, "latent")
                    if latent_key not in available_keys:
                        continue
                    state_dict = OrderedDict([
                        ("latent", shard_file.get_tensor(latent_key)),
                    ])
                    for tensor_name in LATENT_AUX_TENSOR_KEYS:
                        tensor_key = get_latent_shard_tensor_key(key_prefix, tensor_name)
                        if tensor_key in available_keys:
                            state_dict[tensor_name] = shard_file.get_tensor(tensor_key)
                    compact_buffer.append((key_prefix, state_dict, None))

                    if len(compact_buffer) >= shard_size:
                        _write_latent_shard_entries(latent_dir, bucket_key, compact_buffer, manifest)
                        created_for_bucket += 1
                        compact_buffer = []

        if len(compact_buffer) > 0:
            _write_latent_shard_entries(latent_dir, bucket_key, compact_buffer, manifest)
            created_for_bucket += 1

        save_latent_shard_manifest(latent_dir, manifest)

        # Release the safe_open mmaps held by the read loop above so Windows lets us
        # delete the just-read shards (a still-mapped shard -> os.remove WinError 5).
        compact_buffer = []
        try:
            del shard_file
        except NameError:
            pass
        gc.collect()

        for shard_name in partial_shards:
            shard_path = os.path.join(latent_dir, shard_name)
            # One quick attempt only. A just-read shard can still be briefly locked
            # (antivirus/indexer on the data drive, or mmap-release lag), and retrying
            # every file here would stall startup. Don't warn — the end-of-run orphan
            # sweep below deletes whatever's left once the locks clear, and anything
            # still stuck is removed on the next startup.
            try:
                os.remove(shard_path)
                removed_shards += 1
            except (FileNotFoundError, PermissionError, OSError):
                pass

        moved_items += total_partial_items
        created_shards += created_for_bucket
        did_compact = True

    if did_compact:
        referenced_shards = {
            shard_entry.get("shard")
            for shard_entry in manifest.get("items", {}).values()
            if shard_entry.get("shard")
        }
        stuck_orphans = 0
        for file_name in os.listdir(latent_dir):
            if not file_name.endswith(".safetensors") or not file_name.startswith("bucket_"):
                continue
            if file_name in referenced_shards:
                continue
            if not _best_effort_remove_shard(os.path.join(latent_dir, file_name)):
                stuck_orphans += 1
        if stuck_orphans:
            print_acc(
                f"Note: {stuck_orphans} superseded latent shard(s) in {latent_dir} are "
                f"temporarily locked (antivirus/indexer); they'll be removed on the next startup."
            )

    return {
        "compacted": did_compact,
        "moved_items": moved_items,
        "removed_shards": removed_shards,
        "created_shards": created_shards,
    }


def get_text_embed_shard_manifest_path(te_dir: str) -> str:
    return os.path.join(te_dir, TEXT_EMBED_CACHE_SHARD_MANIFEST_NAME)


def load_text_embed_shard_manifest(te_dir: str):
    manifest_path = get_text_embed_shard_manifest_path(te_dir)
    return _load_json_manifest_with_cache(
        manifest_path,
        text_embed_shard_manifest_cache,
        TEXT_EMBED_CACHE_SHARD_VERSION,
    )


def save_text_embed_shard_manifest(te_dir: str, manifest: dict):
    manifest_path = get_text_embed_shard_manifest_path(te_dir)
    _save_json_manifest_with_cache(
        manifest_path,
        manifest,
        text_embed_shard_manifest_cache,
    )


def infer_next_text_embed_shard_index(manifest: dict) -> int:
    max_index = -1
    for entry in manifest.get("items", {}).values():
        shard_name = entry.get("shard", "")
        shard_root, shard_ext = os.path.splitext(shard_name)
        if shard_ext != ".safetensors":
            continue
        try:
            max_index = max(max_index, int(shard_root.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max_index + 1


def rebuild_text_embed_shard_manifest_from_shards(te_dir: str, shard_size: int):
    if not os.path.isdir(te_dir):
        return None

    manifest = {
        "version": TEXT_EMBED_CACHE_SHARD_VERSION,
        "shard_size": shard_size,
        "next_shard_index": 0,
        "items": {},
    }
    max_index = -1

    for file_name in os.listdir(te_dir):
        if not file_name.endswith(".safetensors") or not file_name.startswith("text_embeddings_"):
            continue

        shard_root, _ = os.path.splitext(file_name)
        try:
            max_index = max(max_index, int(shard_root.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue

        shard_path = os.path.join(te_dir, file_name)
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as shard_file:
                for key in shard_file.keys():
                    key_prefix = key.split(TEXT_EMBED_CACHE_SHARD_KEY_SEP, 1)[0]
                    if key_prefix:
                        manifest["items"][key_prefix] = {
                            "shard": file_name,
                            "key_prefix": key_prefix,
                        }
        except (OSError, ValueError):
            continue

    manifest["next_shard_index"] = max_index + 1
    if max_index < 0 and len(manifest["items"]) == 0:
        return None
    return manifest


def get_or_create_text_embed_shard_manifest(te_dir: str, shard_size: int) -> dict:
    manifest = load_text_embed_shard_manifest(te_dir)
    if manifest is None:
        rebuilt_manifest = rebuild_text_embed_shard_manifest_from_shards(te_dir, shard_size)
        if rebuilt_manifest is not None:
            return rebuilt_manifest
        return {
            "version": TEXT_EMBED_CACHE_SHARD_VERSION,
            "shard_size": shard_size,
            "next_shard_index": 0,
            "items": {},
        }

    manifest["version"] = TEXT_EMBED_CACHE_SHARD_VERSION
    manifest["shard_size"] = shard_size
    manifest.setdefault("items", {})
    if "next_shard_index" not in manifest:
        manifest["next_shard_index"] = infer_next_text_embed_shard_index(manifest)
    return manifest


def build_text_embed_shard_storage_entry(te_dir: str, shard_name: str, key_prefix: str):
    return {
        "storage": "shard",
        "path": os.path.join(te_dir, shard_name),
        "shard": shard_name,
        "key_prefix": key_prefix,
    }


def assign_latent_tensors(
        file_item: 'FileItemDTO',
        latent: torch.Tensor,
        first_frame_latent: Union[torch.Tensor, None] = None,
        audio_latent: Union[torch.Tensor, None] = None,
        first_frame_condition: Union[torch.Tensor, None] = None,
        dtype=None,
):
    target_dtype = dtype if dtype is not None else latent.dtype
    file_item._encoded_latent = latent.to("cpu", dtype=target_dtype)
    file_item._cached_first_frame_latent = None
    file_item._cached_audio_latent = None
    file_item._cached_first_frame_condition = None

    if first_frame_latent is not None:
        first_frame_dtype = dtype if dtype is not None else first_frame_latent.dtype
        file_item._cached_first_frame_latent = first_frame_latent.to(
            "cpu", dtype=first_frame_dtype
        )

    if audio_latent is not None:
        audio_dtype = dtype if dtype is not None else audio_latent.dtype
        file_item._cached_audio_latent = audio_latent.to("cpu", dtype=audio_dtype)

    if first_frame_condition is not None:
        condition_dtype = dtype if dtype is not None else first_frame_condition.dtype
        file_item._cached_first_frame_condition = first_frame_condition.to(
            "cpu", dtype=condition_dtype
        )


def hydrate_file_items_from_cached_latents(
        file_items: List['FileItemDTO'],
        dtype=None,
):
    shard_groups: Dict[str, List[tuple]] = {}

    for file_item in file_items:
        if not file_item.is_latent_cached or file_item._encoded_latent is not None:
            continue

        storage_entry = file_item.get_latent_storage_entry()
        if storage_entry is None:
            continue

        if storage_entry["storage"] == "buffered":
            buffer_key = (
                storage_entry["latent_dir"],
                get_latent_shard_bucket_key(file_item),
            )
            pending_map = lazy_latent_shard_buffers.get(buffer_key)
            buffered_pair = None if pending_map is None else pending_map.get(storage_entry["key_prefix"])
            if buffered_pair is None:
                state_dict = load_pending_latent_state(
                    storage_entry["latent_dir"],
                    storage_entry["key_prefix"],
                )
                if state_dict is None:
                    refreshed_storage_entry = file_item.get_latent_storage_entry(recalculate=True)
                    if refreshed_storage_entry is None:
                        continue
                    if refreshed_storage_entry.get("storage") == "buffered":
                        continue
                    storage_entry = refreshed_storage_entry
                else:
                    assign_latent_tensors(
                        file_item,
                        state_dict["latent"],
                        state_dict.get("first_frame_latent"),
                        state_dict.get("audio_latent"),
                        state_dict.get("first_frame_condition"),
                        dtype=dtype,
                    )
                    continue

            if storage_entry["storage"] == "buffered":
                state_dict = buffered_pair[1]
                assign_latent_tensors(
                    file_item,
                    state_dict["latent"],
                    state_dict.get("first_frame_latent"),
                    state_dict.get("audio_latent"),
                    state_dict.get("first_frame_condition"),
                    dtype=dtype,
                )
                continue

        if storage_entry["storage"] == "legacy":
            state_dict = load_file(storage_entry["path"], device="cpu")
            assign_latent_tensors(
                file_item,
                state_dict["latent"],
                state_dict.get("first_frame_latent"),
                state_dict.get("audio_latent"),
                state_dict.get("first_frame_condition"),
                dtype=dtype,
            )
            continue

        shard_groups.setdefault(storage_entry["path"], []).append(
            (file_item, storage_entry["key_prefix"])
        )

    for shard_path, shard_items in shard_groups.items():
        with safe_open(shard_path, framework="pt", device="cpu") as shard_file:
            available_keys = set(shard_file.keys())
            for file_item, key_prefix in shard_items:
                latent_key = get_latent_shard_tensor_key(key_prefix, "latent")
                if latent_key not in available_keys:
                    raise RuntimeError(
                        f"Latent cache shard '{shard_path}' is missing key "
                        f"'{latent_key}' (likely a truncated/corrupt shard). Delete "
                        f"this shard and its '_latent_shards.sqlite3' index to re-cache."
                    )
                assign_latent_tensors(
                    file_item,
                    shard_file.get_tensor(latent_key),
                    shard_file.get_tensor(
                        get_latent_shard_tensor_key(key_prefix, "first_frame_latent")
                    )
                    if get_latent_shard_tensor_key(key_prefix, "first_frame_latent") in available_keys
                    else None,
                    shard_file.get_tensor(
                        get_latent_shard_tensor_key(key_prefix, "audio_latent")
                    )
                    if get_latent_shard_tensor_key(key_prefix, "audio_latent") in available_keys
                    else None,
                    shard_file.get_tensor(
                        get_latent_shard_tensor_key(key_prefix, "first_frame_condition")
                    )
                    if get_latent_shard_tensor_key(key_prefix, "first_frame_condition") in available_keys
                    else None,
                    dtype=dtype,
                )


def cache_file_item_latents_to_disk(
        file_item_state_dict_pairs: List[tuple],
        shard_size: int,
        buffered: bool = False,
):
    if len(file_item_state_dict_pairs) == 0:
        return

    pending_pairs = []
    for file_item, state_dict in file_item_state_dict_pairs:
        storage_entry = file_item.get_latent_storage_entry(recalculate=True)
        if storage_entry is not None:
            file_item.is_latent_cached = True
            continue
        pending_pairs.append((file_item, state_dict))

    if len(pending_pairs) == 0:
        return

    shard_size = max(1, shard_size)
    if shard_size == 1:
        for file_item, state_dict in pending_pairs:
            _save_legacy_latent_cache_file(file_item, state_dict)
        return

    if buffered:
        pending_entries = []
        with lazy_latent_shard_buffer_lock:
            for file_item, state_dict in pending_pairs:
                latent_dir = file_item.get_latent_cache_dir()
                bucket_key = get_latent_shard_bucket_key(file_item)
                buffer_key = (latent_dir, bucket_key)
                key_prefix = file_item.get_latent_cache_key()
                pending_map = lazy_latent_shard_buffers.setdefault(buffer_key, OrderedDict())
                pending_map[key_prefix] = (file_item, state_dict)
                lazy_latent_shard_sizes[buffer_key] = shard_size
                pending_entries.append((latent_dir, key_prefix, bucket_key, state_dict))
                file_item.set_latent_storage_entry(
                    build_buffered_latent_storage_entry(latent_dir, key_prefix)
                )
                file_item.is_latent_cached = True
        entries_by_dir: Dict[str, List[tuple]] = {}
        for latent_dir, key_prefix, bucket_key, state_dict in pending_entries:
            entries_by_dir.setdefault(latent_dir, []).append((key_prefix, bucket_key, state_dict))
        for latent_dir, entries in entries_by_dir.items():
            upsert_pending_latent_states(latent_dir, entries)
        flush_buffered_file_item_latents_to_disk(force=False)
        return

    grouped_pairs: Dict[tuple, List[tuple]] = {}
    for file_item, state_dict in pending_pairs:
        group_key = (
            file_item.get_latent_cache_dir(),
            get_latent_shard_bucket_key(file_item),
        )
        grouped_pairs.setdefault(group_key, []).append((file_item, state_dict))

    manifests: Dict[str, dict] = {}
    for (latent_dir, bucket_key), group in grouped_pairs.items():
        manifest = manifests.get(latent_dir)
        if manifest is None:
            manifest = get_or_create_latent_shard_manifest(latent_dir, shard_size)
            manifests[latent_dir] = manifest

        for shard_start in range(0, len(group), shard_size):
            shard_pairs = group[shard_start:shard_start + shard_size]
            _write_latent_shard_pairs(latent_dir, bucket_key, shard_pairs, manifest)

    for latent_dir, manifest in manifests.items():
        save_latent_shard_manifest(latent_dir, manifest)


def load_control_images_for_text_embedding(file_item: 'FileItemDTO', sd: 'StableDiffusion'):
    if not file_item.encode_control_in_text_embeddings:
        return None
    if file_item.control_path is None:
        if getattr(sd, "encodes_text_without_control", False):
            # a plain text-to-image item for a model that handles one: the caption alone
            return None
        raise Exception(f"Could not find a control image for {file_item.path} which is needed for this model")

    ctrl_img_list = []
    control_path_list = file_item.control_path
    if not isinstance(control_path_list, list):
        control_path_list = [control_path_list]
    for control_path in control_path_list:
        try:
            img = Image.open(control_path).convert("RGB")
            img = exif_transpose(img)
            img = TF.to_tensor(img).unsqueeze(0).to(sd.device_torch, dtype=sd.torch_dtype)
            ctrl_img_list.append(img)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading control image: {control_path}")

    if len(ctrl_img_list) == 0:
        return None
    if not sd.has_multiple_control_images:
        return ctrl_img_list[0]
    return ctrl_img_list


def encode_prompt_embeds_for_file_items(
        sd: 'StableDiffusion',
        file_items: List['FileItemDTO'],
) -> PromptEmbeds:
    if len(file_items) == 0:
        raise ValueError("file_items cannot be empty")

    for file_item in file_items:
        if file_item.caption is None:
            file_item.load_caption()

    uses_control = any([file_item.encode_control_in_text_embeddings for file_item in file_items])
    if not uses_control:
        prompts = [file_item.caption for file_item in file_items]
        return sd.encode_prompt(prompts)

    prompt_embeds_list = []
    for file_item in file_items:
        control_images = load_control_images_for_text_embedding(file_item, sd)
        prompt_embeds_list.append(
            sd.encode_prompt(file_item.caption, control_images=control_images)
        )

    return concat_prompt_embeds(prompt_embeds_list, padding_side=file_items[0].te_padding_side)


def hydrate_file_items_from_cached_text_embeddings(
        file_items: List['FileItemDTO'],
):
    shard_groups: Dict[str, List[tuple]] = {}

    for file_item in file_items:
        if not file_item.is_text_embedding_cached or file_item.prompt_embeds is not None:
            continue

        storage_entry = file_item.get_text_embedding_storage_entry()
        if storage_entry is None:
            continue

        if storage_entry["storage"] == "legacy":
            # The legacy WRITER (shard_size == 1, below) is polymorphic --
            # prompt_embeds.save(path) -- but this READER was hardcoded to
            # PromptEmbeds, so an AdvancedPromptEmbeds entry (krea2 / minimax_h3)
            # blew up with "No text embeds found in state dict". Reached three ways:
            # text_embedding_cache_shard_size == 1, ANY leftover legacy .safetensors
            # (probed unconditionally by get_text_embedding_storage_entry regardless
            # of shard_size), or the caches_only glob fallback.
            # AdvancedPromptEmbeds.save() stamps class_name in the metadata; dispatch on it.
            _legacy_path = storage_entry["path"]
            _legacy_cls = PromptEmbeds
            try:
                with safe_open(_legacy_path, framework="pt") as _lf:
                    _lmeta = _lf.metadata() or {}
                if _lmeta.get("class_name") == "AdvancedPromptEmbeds":
                    _legacy_cls = AdvancedPromptEmbeds
            except Exception:
                pass
            file_item.prompt_embeds = _legacy_cls.load(_legacy_path)
            continue

        shard_groups.setdefault(storage_entry["path"], []).append(
            (file_item, storage_entry["key_prefix"])
        )

    for shard_path, shard_items in shard_groups.items():
        with safe_open(shard_path, framework="pt", device="cpu") as shard_file:
            available_keys = sorted(list(shard_file.keys()))
            for file_item, key_prefix in shard_items:
                state_dict = OrderedDict()
                tensor_prefix = f"{key_prefix}{TEXT_EMBED_CACHE_SHARD_KEY_SEP}"
                for key in available_keys:
                    if key.startswith(tensor_prefix):
                        state_dict[key[len(tensor_prefix):]] = shard_file.get_tensor(key)
                # Shards can hold entries from either class (krea2 / minimax_h3 use
                # AdvancedPromptEmbeds), and rebuilding the wrong one yields an object
                # the model cannot consume. The writer stamps a marker key; dispatch on it.
                if AdvancedPromptEmbeds.is_advanced_state_dict(state_dict):
                    file_item.prompt_embeds = AdvancedPromptEmbeds.from_state_dict(state_dict)
                else:
                    file_item.prompt_embeds = PromptEmbeds.from_state_dict(state_dict)


def cache_file_item_text_embeddings_to_disk(
        file_item_prompt_embeds_pairs: List[tuple],
        shard_size: int,
):
    if len(file_item_prompt_embeds_pairs) == 0:
        return

    pending_pairs = []
    for file_item, prompt_embeds in file_item_prompt_embeds_pairs:
        storage_entry = file_item.get_text_embedding_storage_entry(recalculate=True)
        if storage_entry is not None:
            file_item.is_text_embedding_cached = True
            continue
        pending_pairs.append((file_item, prompt_embeds))

    if len(pending_pairs) == 0:
        return

    shard_size = max(1, shard_size)
    if shard_size == 1:
        for file_item, prompt_embeds in pending_pairs:
            text_embedding_path = file_item.get_text_embedding_path(recalculate=True)
            os.makedirs(os.path.dirname(text_embedding_path), exist_ok=True)
            prompt_embeds.save(text_embedding_path)
            file_item.set_text_embedding_storage_entry({
                "storage": "legacy",
                "path": text_embedding_path,
            })
            file_item.is_text_embedding_cached = True
        return

    grouped_pairs: Dict[str, List[tuple]] = {}
    for file_item, prompt_embeds in pending_pairs:
        grouped_pairs.setdefault(file_item.get_text_embedding_cache_dir(), []).append((file_item, prompt_embeds))

    manifests: Dict[str, dict] = {}
    for te_dir, group in grouped_pairs.items():
        manifest = manifests.get(te_dir)
        if manifest is None:
            manifest = get_or_create_text_embed_shard_manifest(te_dir, shard_size)
            manifests[te_dir] = manifest

        for shard_start in range(0, len(group), shard_size):
            shard_pairs = group[shard_start:shard_start + shard_size]
            shard_index = manifest.get("next_shard_index", 0)
            shard_name = f"text_embeddings_{shard_index:05d}.safetensors"
            shard_path = os.path.join(te_dir, shard_name)

            shard_state_dict = OrderedDict()
            for file_item, prompt_embeds in shard_pairs:
                key_prefix = file_item.get_text_embedding_cache_key()
                tensor_prefix = f"{key_prefix}{TEXT_EMBED_CACHE_SHARD_KEY_SEP}"
                for key, value in prompt_embeds.to_state_dict(prefix=tensor_prefix).items():
                    shard_state_dict[key] = value

            shard_meta = get_meta_for_safetensors(
                OrderedDict([
                    ("text_embedding_cache_shard_version", TEXT_EMBED_CACHE_SHARD_VERSION),
                    ("items", len(shard_pairs)),
                ])
            )
            os.makedirs(te_dir, exist_ok=True)
            _atomic_save_file(shard_state_dict, shard_path, metadata=shard_meta)

            for file_item, _ in shard_pairs:
                key_prefix = file_item.get_text_embedding_cache_key()
                manifest["items"][key_prefix] = {
                    "shard": shard_name,
                    "key_prefix": key_prefix,
                }
                file_item.set_text_embedding_storage_entry(
                    build_text_embed_shard_storage_entry(te_dir, shard_name, key_prefix)
                )
                file_item.is_text_embedding_cached = True

            manifest["next_shard_index"] = shard_index + 1

    for te_dir, manifest in manifests.items():
        save_text_embed_shard_manifest(te_dir, manifest)


def standardize_images(images):
    """
    Standardize the given batch of images using the specified mean and std.
    Expects values of 0 - 1

    Args:
    images (torch.Tensor): A batch of images in the shape of (N, C, H, W),
                           where N is the number of images, C is the number of channels,
                           H is the height, and W is the width.

    Returns:
    torch.Tensor: Standardized images.
    """
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    # Define the normalization transform
    normalize = transforms.Normalize(mean=mean, std=std)

    # Apply normalization to each image in the batch
    standardized_images = torch.stack([normalize(img) for img in images])

    return standardized_images

def clean_caption(caption):
    # this doesnt make any sense anymore in a world that is not based on comma seperated tokens
    # # remove any newlines
    # caption = caption.replace('\n', ', ')
    # # remove new lines for all operating systems
    # caption = caption.replace('\r', ', ')
    # caption_split = caption.split(',')
    # # remove empty strings
    # caption_split = [p.strip() for p in caption_split if p.strip()]
    # # join back together
    # caption = ', '.join(caption_split)
    return caption


class CaptionMixin:
    def get_caption_item(self: 'AiToolkitDataset', index):
        if not hasattr(self, 'caption_type'):
            raise Exception('caption_type not found on class instance')
        if not hasattr(self, 'file_list'):
            raise Exception('file_list not found on class instance')
        img_path_or_tuple = self.file_list[index]
        ext = self.dataset_config.caption_ext
        if isinstance(img_path_or_tuple, tuple):
            img_path = img_path_or_tuple[0] if isinstance(img_path_or_tuple[0], str) else img_path_or_tuple[0].path
            # check if either has a prompt file
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = None
            prompt_path = path_no_ext + ext
        else:
            img_path = img_path_or_tuple if isinstance(img_path_or_tuple, str) else img_path_or_tuple.path
            # see if prompt file exists
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = path_no_ext + ext
                
        # allow folders to have a default prompt
        default_prompt_path = os.path.join(os.path.dirname(img_path), 'default.txt')
        default_prompt_path_with_ext = os.path.join(os.path.dirname(img_path), 'default' + ext)

        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                # check if is json
                if prompt_path.endswith('.json'):
                    prompt = json.loads(prompt)
                    if 'caption' in prompt:
                        prompt = prompt['caption']

                prompt = clean_caption(prompt)
        elif os.path.exists(default_prompt_path_with_ext):
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                prompt = clean_caption(prompt)
        elif os.path.exists(default_prompt_path):
            with open(default_prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                prompt = clean_caption(prompt)
        else:
            prompt = ''
            # get default_prompt if it exists on the class instance
            if hasattr(self, 'default_prompt'):
                prompt = self.default_prompt
            if hasattr(self, 'default_caption'):
                prompt = self.default_caption

        # handle replacements
        replacement_list = self.dataset_config.replacements if isinstance(self.dataset_config.replacements, list) else []
        for replacement in replacement_list:
            from_string, to_string = replacement.split('|')
            prompt = prompt.replace(from_string, to_string)

        return prompt


if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig
    from toolkit.data_transfer_object.data_loader import FileItemDTO


class Bucket:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.file_list_idx: List[int] = []


class BucketsMixin:
    def __init__(self):
        self.buckets: Dict[str, Bucket] = {}
        self.batch_indices: List[List[int]] = []
        self.active_file_indices: List[int] = []

    def _use_balanced_same_folder_clip_sampling(self: 'AiToolkitDataset') -> bool:
        return bool(
            getattr(self.dataset_config, 'clip_image_from_same_folder', False)
            and getattr(self.dataset_config, 'clip_image_from_same_folder_balanced', False)
        )

    def _get_file_item_parent_folder(self: 'AiToolkitDataset', file_item: 'FileItemDTO') -> str:
        return os.path.dirname(file_item.path)

    def _build_balanced_folder_order(self: 'AiToolkitDataset', indices: List[int]) -> List[int]:
        if not self._use_balanced_same_folder_clip_sampling() or len(indices) <= 1:
            return list(indices)

        folder_queues: Dict[str, List[int]] = defaultdict(list)
        for idx in indices:
            folder_key = self._get_file_item_parent_folder(self.file_list[idx])
            folder_queues[folder_key].append(idx)

        if len(folder_queues) <= 1:
            return list(indices)

        for queue in folder_queues.values():
            random.shuffle(queue)

        active_folders = list(folder_queues.keys())
        random.shuffle(active_folders)

        ordered_indices: List[int] = []
        while active_folders:
            random.shuffle(active_folders)
            next_round: List[str] = []
            for folder_key in active_folders:
                queue = folder_queues[folder_key]
                if len(queue) == 0:
                    continue
                ordered_indices.append(queue.pop())
                if len(queue) > 0:
                    next_round.append(folder_key)
            active_folders = next_round

        return ordered_indices

    def prepare_same_folder_clip_image_pairs(self: 'AiToolkitDataset'):
        if not self._use_balanced_same_folder_clip_sampling():
            for file_item in self.file_list:
                file_item.clip_image_epoch_path = None
            return

        # Only pair among active (bucketed, non-upscale-skipped) images, so a
        # skipped image can't be selected as the clip reference for a trained one.
        if getattr(self.dataset_config, "buckets", False):
            active_indices = list(self.active_file_indices)
        else:
            active_indices = list(range(len(self.file_list)))

        # clear everything first; inactive items simply stay unpaired
        for file_item in self.file_list:
            file_item.clip_image_epoch_path = None

        folder_groups: Dict[str, List[int]] = defaultdict(list)
        for idx in active_indices:
            folder_groups[self._get_file_item_parent_folder(self.file_list[idx])].append(idx)

        for folder_indices in folder_groups.values():
            if len(folder_indices) == 0:
                continue
            if len(folder_indices) == 1:
                only_idx = folder_indices[0]
                self.file_list[only_idx].clip_image_epoch_path = self.file_list[only_idx].path
                continue

            target_indices = list(folder_indices)
            reference_indices = list(folder_indices)
            random.shuffle(target_indices)
            random.shuffle(reference_indices)

            remaining_refs = list(reference_indices)
            assigned_refs: Dict[int, int] = {}

            for target_idx in target_indices:
                target_path = self.file_list[target_idx].path
                valid_refs = [
                    ref_idx for ref_idx in remaining_refs
                    if self.file_list[ref_idx].path != target_path
                ]
                if len(valid_refs) == 0:
                    valid_refs = list(remaining_refs)

                chosen_ref_idx = random.choice(valid_refs)
                remaining_refs.remove(chosen_ref_idx)
                assigned_refs[target_idx] = chosen_ref_idx

            for target_idx, ref_idx in assigned_refs.items():
                self.file_list[target_idx].clip_image_epoch_path = self.file_list[ref_idx].path

    def build_batch_indices(self: 'AiToolkitDataset'):
        self.batch_indices = []
        for key, bucket in self.buckets.items():
            ordered_indices = self._build_balanced_folder_order(bucket.file_list_idx)
            for start_idx in range(0, len(ordered_indices), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(bucket.file_list_idx))
                batch = ordered_indices[start_idx:end_idx]
                self.batch_indices.append(batch)

    def shuffle_buckets(self: 'AiToolkitDataset'):
        for key, bucket in self.buckets.items():
            random.shuffle(bucket.file_list_idx)

    def get_active_bucket_file_items(self: 'AiToolkitDataset') -> List['FileItemDTO']:
        if not getattr(self.dataset_config, "buckets", False):
            return list(self.file_list)
        return [self.file_list[idx] for idx in self.active_file_indices]

    def setup_buckets(self: 'AiToolkitDataset', quiet=False):
        if not hasattr(self, 'file_list'):
            raise Exception(f'file_list not found on class instance {self.__class__.__name__}')
        if not hasattr(self, 'dataset_config'):
            raise Exception(f'dataset_config not found on class instance {self.__class__.__name__}')

        if self.epoch_num > 0 and self.dataset_config.poi is None:
            # no need to rebuild buckets for now
            # todo handle random cropping for buckets
            return
        self.buckets = {}  # clear it
        self.active_file_indices = []

        config: 'DatasetConfig' = self.dataset_config
        resolution = config.resolution
        bucket_tolerance = config.bucket_tolerance
        file_list: List['FileItemDTO'] = self.file_list

        # for file_item in enumerate(file_list):
        for idx, file_item in enumerate(file_list):
            file_item: 'FileItemDTO' = file_item
            width = int(file_item.width * file_item.dataset_config.scale)
            height = int(file_item.height * file_item.dataset_config.scale)

            did_process_poi = False
            if file_item.has_point_of_interest:
                # Attempt to process the poi if we can. It wont process if the image is smaller than the resolution
                did_process_poi = file_item.setup_poi_bucket()
            if self.dataset_config.square_crop:
                # we scale first so smallest size matches resolution
                scale_factor_x = resolution / width
                scale_factor_y = resolution / height
                scale_factor = max(scale_factor_x, scale_factor_y)
                file_item.scale_to_width = math.ceil(width * scale_factor)
                file_item.scale_to_height = math.ceil(height * scale_factor)
                file_item.crop_width = resolution
                file_item.crop_height = resolution
                if width > height:
                    file_item.crop_x = int(file_item.scale_to_width / 2 - resolution / 2)
                    file_item.crop_y = 0
                else:
                    file_item.crop_x = 0
                    file_item.crop_y = int(file_item.scale_to_height / 2 - resolution / 2)
            elif not did_process_poi:
                bucket_resolution = get_bucket_for_image_size(
                    width, height,
                    resolution=resolution,
                    divisibility=bucket_tolerance
                )

                # Calculate scale factors for width and height
                width_scale_factor = bucket_resolution["width"] / width
                height_scale_factor = bucket_resolution["height"] / height

                # Use the maximum of the scale factors to ensure both dimensions are scaled above the bucket resolution
                max_scale_factor = max(width_scale_factor, height_scale_factor)

                # round up
                file_item.scale_to_width = int(math.ceil(width * max_scale_factor))
                file_item.scale_to_height = int(math.ceil(height * max_scale_factor))

                file_item.crop_height = bucket_resolution["height"]
                file_item.crop_width = bucket_resolution["width"]

                new_width = bucket_resolution["width"]
                new_height = bucket_resolution["height"]

                if self.dataset_config.random_crop:
                    # random crop
                    crop_x = random.randint(0, file_item.scale_to_width - new_width)
                    crop_y = random.randint(0, file_item.scale_to_height - new_height)
                    file_item.crop_x = crop_x
                    file_item.crop_y = crop_y
                else:
                    # do central crop
                    file_item.crop_x = int((file_item.scale_to_width - new_width) / 2)
                    file_item.crop_y = int((file_item.scale_to_height - new_height) / 2)

                if file_item.crop_y < 0 or file_item.crop_x < 0:
                    print_acc('debug')

            # check if bucket exists, if not, create it
            bucket_key = f'{file_item.crop_width}x{file_item.crop_height}'
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = Bucket(file_item.crop_width, file_item.crop_height)
            self.buckets[bucket_key].file_list_idx.append(idx)
            self.active_file_indices.append(idx)

        if len(self.active_file_indices) == 0:
            raise Exception(
                f"No images remain in {self.dataset_path} after bucketing"
            )

        # print the buckets
        self.shuffle_buckets()
        self.build_batch_indices()
        if not quiet:
            print_acc(f'Bucket sizes for {self.dataset_path}:')
            for key, bucket in self.buckets.items():
                print_acc(f'{key}: {len(bucket.file_list_idx)} files')
            print_acc(f'{len(self.buckets)} buckets made')


class CaptionProcessingDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
            self.raw_caption: str = None
            self.raw_caption_short: str = None
            self.caption: str = None
            self.caption_short: str = None

            dataset_config: DatasetConfig = kwargs.get('dataset_config', None)
            self.extra_values: List[float] = dataset_config.extra_values
            self.trigger_word = dataset_config.trigger_word

    # todo allow for loading from sd-scripts style dict
    def load_caption(self: 'FileItemDTO', caption_dict: Union[dict, None]=None):
        if self.raw_caption is not None:
            # we already loaded it
            pass
        elif caption_dict is not None and self.path in caption_dict and "caption" in caption_dict[self.path]:
            self.raw_caption = caption_dict[self.path]["caption"]
            if 'caption_short' in caption_dict[self.path]:
                self.raw_caption_short = caption_dict[self.path]["caption_short"]
                if self.dataset_config.use_short_captions:
                    self.raw_caption = caption_dict[self.path]["caption_short"]
        else:
            # see if prompt file exists
            path_no_ext = os.path.splitext(self.path)[0]
            prompt_ext = self.dataset_config.caption_ext
            prompt_path = path_no_ext + prompt_ext
            short_caption = None

            prompt_text = None
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt_text = f.read()
            elif self.dataset_config.caches_only and getattr(self, 'cached_caption_text', None) is not None:
                # caption sidecar is gone (caches_only): use the caption text
                # persisted in the size database so the cached text embedding
                # key still matches
                prompt_text = self.cached_caption_text

            if prompt_text is not None:
                prompt = prompt_text
                short_caption = None
                if prompt_path.endswith('.json'):
                    # replace any line endings with commas for \n \r \r\n
                    prompt = prompt.replace('\r\n', ' ')
                    prompt = prompt.replace('\n', ' ')
                    prompt = prompt.replace('\r', ' ')

                    prompt_json = json.loads(prompt)
                    if 'caption' in prompt_json:
                        prompt = prompt_json['caption']
                    if 'caption_short' in prompt_json:
                        short_caption = prompt_json['caption_short']
                        if self.dataset_config.use_short_captions:
                            prompt = short_caption
                    if 'extra_values' in prompt_json:
                        self.extra_values = prompt_json['extra_values']

                prompt = clean_caption(prompt)
                if short_caption is not None:
                    short_caption = clean_caption(short_caption)

                if prompt.strip() == '' and self.dataset_config.default_caption is not None:
                    prompt = self.dataset_config.default_caption
            else:
                prompt = ''
                if self.dataset_config.default_caption is not None:
                    prompt = self.dataset_config.default_caption

            if short_caption is None:
                short_caption = self.dataset_config.default_caption
            self.raw_caption = prompt
            self.raw_caption_short = short_caption

        self.caption = self.get_caption()
        if self.raw_caption_short is not None:
            self.caption_short = self.get_caption(short_caption=True)

    def get_caption(
            self: 'FileItemDTO',
            trigger=None,
            to_replace_list=None,
            add_if_not_present=False,
            short_caption=False
    ):
        if trigger is None and self.trigger_word is not None:
            trigger = self.trigger_word
        
        if trigger is not None and not self.is_reg:
            # add if not present if not regularization
            add_if_not_present = True
            
        if short_caption:
            raw_caption = self.raw_caption_short
        else:
            raw_caption = self.raw_caption
        if raw_caption is None:
            raw_caption = ''
        # handle dropout
        if self.dataset_config.caption_dropout_rate > 0 and not short_caption and not self.dataset_config.cache_text_embeddings:
            # get a random float form 0 to 1
            rand = random.random()
            if rand < self.dataset_config.caption_dropout_rate:
                # drop the caption
                return ''

        # get tokens
        token_list = raw_caption.split(',')
        # trim whitespace
        token_list = [x.strip() for x in token_list]
        # remove empty strings
        token_list = [x for x in token_list if x]

        # handle token dropout
        if self.dataset_config.token_dropout_rate > 0 and not short_caption and not self.dataset_config.cache_text_embeddings:
            new_token_list = []
            keep_tokens: int = self.dataset_config.keep_tokens
            for idx, token in enumerate(token_list):
                if idx < keep_tokens:
                    new_token_list.append(token)
                elif self.dataset_config.token_dropout_rate >= 1.0:
                    # drop the token
                    pass
                else:
                    # get a random float form 0 to 1
                    rand = random.random()
                    if rand > self.dataset_config.token_dropout_rate:
                        # keep the token
                        new_token_list.append(token)
            token_list = new_token_list

        if self.dataset_config.shuffle_tokens:
            random.shuffle(token_list)

        # join back together
        caption = ', '.join(token_list)
        caption = inject_trigger_into_prompt(caption, trigger, to_replace_list, add_if_not_present)

        if self.dataset_config.random_triggers:
            num_triggers = self.dataset_config.random_triggers_max
            if num_triggers > 1:
                num_triggers = random.randint(0, num_triggers)

            if num_triggers > 0:
                triggers = random.sample(self.dataset_config.random_triggers, num_triggers)
                caption = caption + ', ' + ', '.join(triggers)
                # add random triggers
                # for i in range(num_triggers):
                #     # fastest method
                #     trigger = self.dataset_config.random_triggers[int(random.random() * (len(self.dataset_config.random_triggers)))]
                #     caption = caption + ', ' + trigger

        if self.dataset_config.shuffle_tokens:
            # shuffle again
            token_list = caption.split(',')
            # trim whitespace
            token_list = [x.strip() for x in token_list]
            # remove empty strings
            token_list = [x for x in token_list if x]
            random.shuffle(token_list)
            caption = ', '.join(token_list)
        if caption == '':
            pass
        return caption


class ImageProcessingDTOMixin:
    def load_and_process_video(
        self: 'FileItemDTO',
        transform: Union[None, transforms.Compose],
        only_load_latents=False
    ):
        
        if self.augments is not None and len(self.augments) > 0:
            raise Exception('Augments not supported for videos')
            
        if self.has_augmentations:
            raise Exception('Augmentations not supported for videos')
        
        if not self.dataset_config.buckets:
            raise Exception('Buckets required for video processing')
        
        do_audio = self.dataset_config.do_audio
        
        try:
            # Use OpenCV to capture video frames
            cap = cv2.VideoCapture(self.path)
            
            if not cap.isOpened():
                raise Exception(f"Failed to open video file: {self.path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate the max valid frame index (accounting for zero-indexing)
            max_frame_index = total_frames - 1
            
            # Only log video properties if in debug mode
            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                print_acc(f"Video properties: {self.path}")
                print_acc(f"  Total frames: {total_frames}")
                print_acc(f"  Max valid frame index: {max_frame_index}")
                print_acc(f"  FPS: {video_fps}")
            
            frames_to_extract = []
            
            # Always stretch/shrink to the requested number of frames if needed
            if self.dataset_config.shrink_video_to_frames or total_frames < self.dataset_config.num_frames:
                # Distribute frames evenly across the entire video
                interval = max_frame_index / (self.dataset_config.num_frames - 1) if self.dataset_config.num_frames > 1 else 0
                frames_to_extract = [min(int(round(i * interval)), max_frame_index) for i in range(self.dataset_config.num_frames)]
            else:
                # Calculate frame interval based on FPS ratio
                fps_ratio = video_fps / self.dataset_config.fps
                frame_interval = max(1, int(round(fps_ratio)))
                
                # Calculate max consecutive frames we can extract at desired FPS
                max_consecutive_frames = (total_frames // frame_interval)
                
                if max_consecutive_frames < self.dataset_config.num_frames:
                    # Not enough frames at desired FPS, so stretch instead
                    interval = max_frame_index / (self.dataset_config.num_frames - 1) if self.dataset_config.num_frames > 1 else 0
                    frames_to_extract = [min(int(round(i * interval)), max_frame_index) for i in range(self.dataset_config.num_frames)]
                else:
                    # Calculate max start frame to ensure we can get all num_frames
                    max_start_frame = max_frame_index - ((self.dataset_config.num_frames - 1) * frame_interval)
                    start_frame = random.randint(0, max(0, max_start_frame))
                    
                    # Generate list of frames to extract
                    frames_to_extract = [start_frame + (i * frame_interval) for i in range(self.dataset_config.num_frames)]
                    
            # Final safety check - ensure no frame exceeds max valid index
            frames_to_extract = [min(frame_idx, max_frame_index) for frame_idx in frames_to_extract]
            
            # Only log frames to extract if in debug mode
            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                print_acc(f"  Frames to extract: {frames_to_extract}")
            
            # Extract frames
            frames = []
            for frame_idx in frames_to_extract:
                # Safety check - ensure frame_idx is within bounds (silently fix)
                if frame_idx > max_frame_index:
                    frame_idx = max_frame_index
                
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Silently verify position was set correctly (no warnings unless debug mode)
                if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                    actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if actual_pos != frame_idx:
                        print_acc(f"Warning: Failed to set exact frame position. Requested: {frame_idx}, Actual: {actual_pos}")
                
                ret, frame = cap.read()
                if not ret:
                    # Try to provide more detailed error information
                    actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_pos_info = f"Requested frame: {frame_idx}, Actual frame position: {actual_frame}"
                    
                    # Try to read the next available frame as a fallback
                    fallback_success = False
                    for fallback_offset in [1, -1, 5, -5, 10, -10]:
                        fallback_pos = max(0, min(frame_idx + fallback_offset, max_frame_index))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_pos)
                        fallback_ret, fallback_frame = cap.read()
                        if fallback_ret:
                            # Only log in debug mode
                            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                                print_acc(f"Falling back to nearby frame {fallback_pos} instead of {frame_idx}")
                            frame = fallback_frame
                            fallback_success = True
                            break
                    else:
                        # No fallback worked, raise a more detailed exception
                        video_info = f"Video: {self.path}, Total frames: {total_frames}, FPS: {video_fps}"
                        raise Exception(f"Failed to read frame {frame_idx} from video. {frame_pos_info}. {video_info}")
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                img = Image.fromarray(frame)
                
                # Apply the same processing as for single images
                img = img.convert('RGB')
                
                if self.flip_x:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if self.flip_y:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                # Apply bucketing
                img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
                img = img.crop((
                    self.crop_x,
                    self.crop_y,
                    self.crop_x + self.crop_width,
                    self.crop_y + self.crop_height
                ))
                
                # Apply transform if provided
                if transform:
                    img = transform(img)
                
                frames.append(img)
            
            # Release the video capture
            cap.release()
            
            # Stack frames into tensor [frames, channels, height, width]
            self.tensor = torch.stack(frames)

            # ------------------------------
            # Audio extraction + stretching
            # ------------------------------
            if do_audio:
                # Default to "no audio" unless we successfully extract it
                self.audio_data = None
                self.audio_tensor = None

                try:
                    import torchaudio
                    import torch.nn.functional as F

                    # Compute the time range of the selected frames in the *source* video
                    # Include the last frame by extending to the next frame boundary.
                    if video_fps and video_fps > 0 and len(frames_to_extract) > 0:
                        clip_start_frame = int(frames_to_extract[0])
                        clip_end_frame = int(frames_to_extract[-1])
                        clip_start_time = clip_start_frame / float(video_fps)
                        clip_end_time = (clip_end_frame + 1) / float(video_fps)
                        source_duration = max(0.0, clip_end_time - clip_start_time)
                    else:
                        clip_start_time = 0.0
                        clip_end_time = 0.0
                        source_duration = 0.0

                    # Target duration is how this sampled/stretched clip is interpreted for training
                    # (i.e. num_frames at the configured dataset FPS).
                    if hasattr(self.dataset_config, "fps") and self.dataset_config.fps and self.dataset_config.fps > 0:
                        target_duration = float(self.dataset_config.num_frames) / float(self.dataset_config.fps)
                    else:
                        target_duration = source_duration

                    waveform, sample_rate = torchaudio.load(self.path)  # [channels, samples]
                    
                    if self.dataset_config.audio_normalize:
                        peak = waveform.abs().amax()  # global peak across channels
                        eps = 1e-9
                        target_peak = 0.999  # ~ -0.01 dBFS
                        gain = target_peak / (peak + eps)
                        waveform = waveform * gain

                    # Slice to the selected clip region (when we have a meaningful time range)
                    if source_duration > 0.0:
                        start_sample = int(round(clip_start_time * sample_rate))
                        end_sample = int(round(clip_end_time * sample_rate))
                        start_sample = max(0, min(start_sample, waveform.shape[-1]))
                        end_sample = max(0, min(end_sample, waveform.shape[-1]))
                        if end_sample > start_sample:
                            waveform = waveform[..., start_sample:end_sample]
                        else:
                            # No valid audio segment
                            waveform = None
                    else:
                        # If we can't compute a meaningful time range, treat as no-audio
                        waveform = None

                    if waveform is not None and waveform.numel() > 0:
                        target_samples = int(round(target_duration * sample_rate))
                        if target_samples > 0 and waveform.shape[-1] != target_samples:
                            # Time-stretch/shrink to match the video clip duration implied by dataset FPS.
                            if self.dataset_config.audio_preserve_pitch:
                                waveform = time_stretch_preserve_pitch(waveform, sample_rate, target_samples)  # waveform is [C, L]
                            else:
                                # Use linear interpolation over the time axis.
                                wf = waveform.unsqueeze(0)  # [1, C, L]
                                wf = F.interpolate(wf, size=target_samples, mode="linear", align_corners=False)
                                waveform = wf.squeeze(0)  # [C, L]

                        self.audio_tensor = waveform
                        self.audio_data = {"waveform": waveform, "sample_rate": int(sample_rate)}

                except Exception as e:
                    # Keep behavior identical for non-audio datasets; for audio datasets, just skip if missing/broken.
                    if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                        print_acc(f"Could not extract/stretch audio for {self.path}: {e}")
                    self.audio_data = None
                    self.audio_tensor = None
            
            # Only log success in debug mode
            if hasattr(self.dataset_config, 'debug') and self.dataset_config.debug:
                print_acc(f"Successfully loaded video with {len(frames)} frames: {self.path}")
        
        except Exception as e:
            # Print full traceback
            traceback.print_exc()
            
            # Provide more context about the error
            error_msg = str(e)
            try:
                if 'Failed to read frame' in error_msg and cap is not None:
                    # Try to get more info about the video that failed
                    cap_status = "Opened" if cap.isOpened() else "Closed"
                    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if cap.isOpened() else "Unknown"
                    reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "Unknown"
                    
                    print_acc(f"Video details when error occurred:")
                    print_acc(f"  Cap status: {cap_status}")
                    print_acc(f"  Current position: {current_pos}")
                    print_acc(f"  Reported total frames: {reported_total}")
                    
                    # Try to verify if the video is corrupted
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Go to start
                        start_ret, _ = cap.read()
                        
                        # Try to read the last frame to check if it's accessible
                        if reported_total > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, reported_total - 1)
                            end_ret, _ = cap.read()
                            print_acc(f"  Can read first frame: {start_ret}, Can read last frame: {end_ret}")
                    
                    # Close the cap if it's still open
                    cap.release()
            except Exception as debug_err:
                print_acc(f"Error during error diagnosis: {debug_err}")
            
            print_acc(f"Error: {error_msg}")
            print_acc(f"Error loading video: {self.path}")
            
            # Re-raise with more detailed information
            raise Exception(f"Video loading error ({self.path}): {error_msg}") from e
        
    def load_and_process_image(
            self: 'FileItemDTO',
            transform: Union[None, transforms.Compose],
            only_load_latents=False,
            prefer_cached_latent=True,
    ):
        # if we are caching latents, just do that
        if self.is_latent_cached and prefer_cached_latent:
            # For disk-backed caches, defer latent hydration until collation so a
            # batch can load shard entries together instead of one file per item.
            if self._encoded_latent is None and not self.is_caching_to_disk:
                self.get_latent()
            if self.has_control_image:
                self.load_control_image()
            if self.has_inpaint_image:
                self.load_inpaint_image()
            if self.has_clip_image:
                self.load_clip_image()
            if self.has_mask_image:
                self.load_mask_image()
            if self.has_unconditional:
                self.load_unconditional_image()
            return
        if self.dataset_config.num_frames > 1:
            self.load_and_process_video(transform, only_load_latents)
            return
        try:
            img = Image.open(self.path)
            img = exif_transpose(img)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.path}")

        if self.use_alpha_as_mask:
            # we do this to make sure it does not replace the alpha with another color
            # we want the image just without the alpha channel
            np_img = np.array(img)
            # strip off alpha
            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        img = img.convert('RGB')
        w, h = img.size
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            print_acc(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            print_acc(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if self.dataset_config.buckets:
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
            # crop to x_crop, y_crop, x_crop + crop_width, y_crop + crop_height
            if img.width < self.crop_x + self.crop_width or img.height < self.crop_y + self.crop_height:
                # todo look into this. This still happens sometimes
                print_acc('size mismatch')
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))

            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
        else:
            # Downscale the source image first
            # TODO this is nto right
            img = img.resize(
                (int(img.size[0] * self.dataset_config.scale), int(img.size[1] * self.dataset_config.scale)),
                Image.BICUBIC)
            min_img_size = min(img.size)
            if self.dataset_config.random_crop:
                if self.dataset_config.random_scale and min_img_size > self.dataset_config.resolution:
                    if min_img_size < self.dataset_config.resolution:
                        print_acc(
                            f"Unexpected values: min_img_size={min_img_size}, self.resolution={self.dataset_config.resolution}, image file={self.path}")
                        scale_size = self.dataset_config.resolution
                    else:
                        scale_size = random.randint(self.dataset_config.resolution, int(min_img_size))
                    scaler = scale_size / min_img_size
                    scale_width = int((img.width + 5) * scaler)
                    scale_height = int((img.height + 5) * scaler)
                    img = img.resize((scale_width, scale_height), Image.BICUBIC)
                img = transforms.RandomCrop(self.dataset_config.resolution)(img)
            else:
                img = transforms.CenterCrop(min_img_size)(img)
                img = img.resize((self.dataset_config.resolution, self.dataset_config.resolution), Image.BICUBIC)

        if self.augments is not None and len(self.augments) > 0:
            # do augmentations
            for augment in self.augments:
                if augment in transforms_dict:
                    img = transforms_dict[augment](img)

        if self.has_augmentations:
            # augmentations handles transforms
            img = self.augment_image(img, transform=transform)
        elif transform:
            img = transform(img)

        self.tensor = img
        if not only_load_latents:
            if self.has_control_image:
                self.load_control_image()
            if self.has_inpaint_image:
                self.load_inpaint_image()
            if self.has_clip_image:
                self.load_clip_image()
            if self.has_mask_image:
                self.load_mask_image()
            if self.has_unconditional:
                self.load_unconditional_image()


class InpaintControlFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_inpaint_image = False
        self.inpaint_path: Union[str, None] = None
        self.inpaint_tensor: Union[torch.Tensor, None] = None
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        if dataset_config.inpaint_path is not None:
            # find the control image path
            inpaint_path = dataset_config.inpaint_path
            # we are using control images
            img_path = kwargs.get('path', None)
            img_inpaint_ext_list = ['.png', '.webp']
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]

            for ext in img_inpaint_ext_list:
                p = os.path.join(inpaint_path, file_name_no_ext + ext)
                if os.path.exists(p):
                    self.inpaint_path = p
                    self.has_inpaint_image = True
                    break
                
    def load_inpaint_image(self: 'FileItemDTO'):
        try:
            # image must have alpha channel for inpaint
            img = Image.open(self.inpaint_path)
            # make sure has aplha
            if img.mode != 'RGBA':
                return
            img = exif_transpose(img)
        
            w, h = img.size
            if w > h and self.scale_to_width < self.scale_to_height:
                # throw error, they should match
                raise ValueError(
                    f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            elif h > w and self.scale_to_height < self.scale_to_width:
                # throw error, they should match
                raise ValueError(
                    f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

            if self.flip_x:
                # do a flip
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.flip_y:
                # do a flip
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

            if self.dataset_config.buckets:
                # scale and crop based on file item
                img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
                # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
                # crop
                img = img.crop((
                    self.crop_x,
                    self.crop_y,
                    self.crop_x + self.crop_width,
                    self.crop_y + self.crop_height
                ))
            else:
                raise Exception("Inpaint images not supported for non-bucket datasets")
            
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if self.aug_replay_spatial_transforms:
                tensor = self.augment_spatial_control(img, transform=transform)
            else:
                tensor = transform(img)
            
            # is 0 to 1 with alpha
            self.inpaint_tensor = tensor
        
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.inpaint_path}")

    
    def cleanup_inpaint(self: 'FileItemDTO'):
        self.inpaint_tensor = None
                

class ControlFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_control_image = False
        self.control_path: Union[str, List[str], None] = None
        self.control_tensor: Union[torch.Tensor, None] = None
        self.control_tensor_list: Union[List[torch.Tensor], None] = None
        sd = kwargs.get('sd', None)
        self.use_raw_control_images = sd is not None and sd.use_raw_control_images
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.full_size_control_images = False
        if dataset_config.control_path is not None:
            # find the control image path
            control_path_list = dataset_config.control_path
            if not isinstance(control_path_list, list):
                control_path_list = [control_path_list]
            self.full_size_control_images = dataset_config.full_size_control_images
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            
            found_control_images = []
            for control_path in control_path_list:
                for ext in img_ext_list:
                    if os.path.exists(os.path.join(control_path, file_name_no_ext + ext)):
                        found_control_images.append(os.path.join(control_path, file_name_no_ext + ext))
                        self.has_control_image = True
                        break
            self.control_path = found_control_images
            if len(self.control_path) == 0:
                self.control_path = None
            elif len(self.control_path) == 1:
                # only do one
                self.control_path = self.control_path[0]

    def load_control_image(self: 'FileItemDTO'):
        control_tensors = []
        control_path_list = self.control_path
        if not isinstance(self.control_path, list):
            control_path_list = [self.control_path]
        
        for control_path in control_path_list:
            try:
                img = Image.open(control_path)
                img = exif_transpose(img)

                if img.mode in ("RGBA", "LA"):
                    # Create a background with the specified transparent color
                    transparent_color = tuple(self.dataset_config.control_transparent_color)
                    background = Image.new("RGB", img.size, transparent_color)
                    # Paste the image on top using its alpha channel as mask
                    background.paste(img, mask=img.getchannel("A"))
                    img = background
                else:
                    # Already no alpha channel
                    img = img.convert("RGB")
            except Exception as e:
                print_acc(f"Error: {e}")
                print_acc(f"Error loading image: {control_path}")
            
            if not self.full_size_control_images:
                # we just scale them to 512x512:
                w, h = img.size
                img = img.resize((512, 512), Image.BICUBIC)

            elif not self.use_raw_control_images:
                w, h = img.size
                if self.flip_x:
                    # do a flip
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if self.flip_y:
                    # do a flip
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)

                if self.dataset_config.buckets:
                    # scale and crop based on file item
                    img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
                    # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
                    # crop
                    img = img.crop((
                        self.crop_x,
                        self.crop_y,
                        self.crop_x + self.crop_width,
                        self.crop_y + self.crop_height
                    ))
                else:
                    raise Exception("Control images not supported for non-bucket datasets")
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if self.aug_replay_spatial_transforms:
                tensor = self.augment_spatial_control(img, transform=transform)
            else:
                tensor = transform(img)
            control_tensors.append(tensor)
            
        if len(control_tensors) == 0:
            self.control_tensor = None
        elif len(control_tensors) == 1:
            self.control_tensor = control_tensors[0]
        elif self.use_raw_control_images:
            # just send the list of tensors as their shapes wont match
            self.control_tensor_list = control_tensors
        else:
            self.control_tensor = torch.stack(control_tensors, dim=0)

    def cleanup_control(self: 'FileItemDTO'):
        self.control_tensor = None
        self.control_tensor_list = None


class ClipImageFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_clip_image = False
        self.clip_image_path: Union[str, None] = None
        self.clip_image_epoch_path: Union[str, None] = None
        self.clip_image_tensor: Union[torch.Tensor, None] = None
        self.clip_image_embeds: Union[dict, None] = None
        self.clip_image_embeds_unconditional: Union[dict, None] = None
        self.has_clip_augmentations = False
        self.clip_image_aug_transform: Union[None, A.Compose] = None
        self.clip_image_processor: Union[None, CLIPImageProcessor] = None
        self.clip_image_encoder_path: Union[str, None] = None
        self.is_caching_clip_vision_to_disk = False
        self.is_vision_clip_cached = False
        self.clip_vision_is_quad = False
        self.clip_vision_load_device = 'cpu'
        self.clip_vision_unconditional_paths: Union[List[str], None] = None
        self._clip_vision_embeddings_path: Union[str, None] = None
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        if dataset_config.clip_image_path is not None or dataset_config.clip_image_from_same_folder:
            # copy the clip image processor so the dataloader can do it
            sd = kwargs.get('sd', None)
            if hasattr(sd.adapter, 'clip_image_processor'):
                self.clip_image_processor = sd.adapter.clip_image_processor
        if dataset_config.clip_image_path is not None:
            # find the control image path
            clip_image_path = dataset_config.clip_image_path
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(clip_image_path, file_name_no_ext + ext)):
                    self.clip_image_path = os.path.join(clip_image_path, file_name_no_ext + ext)
                    self.has_clip_image = True
                    break
            self.build_clip_imag_augmentation_transform()
            
        if dataset_config.clip_image_from_same_folder:
            # assume we have one. We will pull it on load.
            self.has_clip_image = True
            self.build_clip_imag_augmentation_transform()

    def build_clip_imag_augmentation_transform(self: 'FileItemDTO'):
        if self.dataset_config.clip_image_augmentations is not None and len(self.dataset_config.clip_image_augmentations) > 0:
            self.has_clip_augmentations = True
            augmentations = [Augments(**aug) for aug in self.dataset_config.clip_image_augmentations]

            if self.dataset_config.clip_image_shuffle_augmentations:
                random.shuffle(augmentations)

            augmentation_list = []
            for aug in augmentations:
                # make sure method name is valid
                assert hasattr(A, aug.method_name), f"invalid augmentation method: {aug.method_name}"
                # get the method
                method = getattr(A, aug.method_name)
                # add the method to the list
                augmentation_list.append(method(**aug.params))

            self.clip_image_aug_transform = A.Compose(augmentation_list)

    def augment_clip_image(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose], ):
        if self.dataset_config.clip_image_shuffle_augmentations:
            self.build_clip_imag_augmentation_transform()

        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        if self.clip_vision_is_quad:
            # image is in a 2x2 gris. split, run augs, and recombine
            # split
            img1, img2 = np.hsplit(open_cv_image, 2)
            img1_1, img1_2 = np.vsplit(img1, 2)
            img2_1, img2_2 = np.vsplit(img2, 2)
            # apply augmentations
            img1_1 = self.clip_image_aug_transform(image=img1_1)["image"]
            img1_2 = self.clip_image_aug_transform(image=img1_2)["image"]
            img2_1 = self.clip_image_aug_transform(image=img2_1)["image"]
            img2_2 = self.clip_image_aug_transform(image=img2_2)["image"]
            # recombine
            augmented = np.vstack((np.hstack((img1_1, img1_2)), np.hstack((img2_1, img2_2))))

        else:
            # apply augmentations
            augmented = self.clip_image_aug_transform(image=open_cv_image)["image"]

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)

        return augmented_tensor

    def get_clip_vision_info_dict(self: 'FileItemDTO'):
        resolved_clip_image_path = self.get_current_clip_image_path()
        item = OrderedDict([
            ("image_encoder_path", self.clip_image_encoder_path),
            ("filename", os.path.basename(resolved_clip_image_path)),
            ("is_quad", self.clip_vision_is_quad)
        ])
        # when adding items, do it after so we dont change old latents
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True
        return item

    def get_clip_vision_cache_dir_candidates(self: 'FileItemDTO'):
        cache_source_path = self.get_current_clip_image_path()
        return get_cache_dir_candidates(
            cache_source_path,
            '_clip_vision_cache',
            override_root=self.dataset_config.cache_dir_override,
            min_free_gb=self.dataset_config.cache_dir_override_min_free_gb,
        )

    def get_clip_vision_embeddings_path_candidates(self: 'FileItemDTO', recalculate=False):
        cache_source_path = self.get_current_clip_image_path()
        hash_dict = self.get_clip_vision_info_dict()
        filename_no_ext = os.path.splitext(os.path.basename(cache_source_path))[0]
        hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
        hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
        hash_str = hash_str.replace('=', '')
        candidates = []
        for latent_dir in self.get_clip_vision_cache_dir_candidates():
            candidates.append(os.path.join(latent_dir, f'{filename_no_ext}_{hash_str}.safetensors'))
        return candidates

    def get_clip_vision_embeddings_path(self: 'FileItemDTO', recalculate=False):
        if self._clip_vision_embeddings_path is not None and not recalculate:
            return self._clip_vision_embeddings_path

        candidate_paths = self.get_clip_vision_embeddings_path_candidates(recalculate=recalculate)
        existing_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        self._clip_vision_embeddings_path = existing_path or candidate_paths[0]

        return self._clip_vision_embeddings_path
    
    def get_current_clip_image_path(self: 'FileItemDTO'):
        if self.clip_image_epoch_path is not None:
            return self.clip_image_epoch_path
        if self.clip_image_path is not None:
            return self.clip_image_path
        return self.path

    def get_new_clip_image_path(self: 'FileItemDTO'):
        if self.clip_image_epoch_path is not None:
            return self.clip_image_epoch_path
        if self.dataset_config.clip_image_from_same_folder:
            # randomly grab an image path from the same folder
            pool_folder = os.path.dirname(self.path)
            # find all images in the folder
            img_files = []
            for ext in img_ext_list:
                img_files += glob.glob(os.path.join(pool_folder, f'*{ext}'))
            # remove the current image if len is greater than 1
            if len(img_files) > 1:
                img_files.remove(self.path)
            # randomly grab one
            return random.choice(img_files)
        else:
            return self.clip_image_path

    def load_clip_image(self: 'FileItemDTO'):
        is_dynamic_size_and_aspect = isinstance(self.clip_image_processor, PixtralVisionImagePreprocessorCompatible) or \
                                    isinstance(self.clip_image_processor, SiglipImageProcessor)
        if self.clip_image_processor is None:
            is_dynamic_size_and_aspect = True # serving it raw
        if self.is_vision_clip_cached:
            self.clip_image_embeds = load_file(self.get_clip_vision_embeddings_path())

            # get a random unconditional image
            if self.clip_vision_unconditional_paths is not None:
                unconditional_path = random.choice(self.clip_vision_unconditional_paths)
                self.clip_image_embeds_unconditional = load_file(unconditional_path)

            return
        clip_image_path = self.get_new_clip_image_path()
        try:
            img = Image.open(clip_image_path).convert('RGB')
            img = exif_transpose(img)
        except Exception as e:
            # make a random noise image
            img = Image.new('RGB', (self.dataset_config.resolution, self.dataset_config.resolution))
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {clip_image_path}")

        img = img.convert('RGB')

        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
        if is_dynamic_size_and_aspect:
            pass  # let the image processor handle it
        elif img.width != img.height:
            min_size = min(img.width, img.height)
            if self.dataset_config.square_crop:
                # center crop to a square
                img = transforms.CenterCrop(min_size)(img)
            else:
                # image must be square. If it is not, we will resize/squish it so it is, that way we don't crop out data
                # resize to the smallest dimension
                img = img.resize((min_size, min_size), Image.BICUBIC)

        if self.has_clip_augmentations:
            self.clip_image_tensor = self.augment_clip_image(img, transform=None)
        else:
            self.clip_image_tensor = transforms.ToTensor()(img)

        # random crop
        # if self.dataset_config.clip_image_random_crop:
        #     # crop up to 20% on all sides. Keep is square
        #     crop_percent = random.randint(0, 20) / 100
        #     crop_width = int(self.clip_image_tensor.shape[2] * crop_percent)
        #     crop_height = int(self.clip_image_tensor.shape[1] * crop_percent)
        #     crop_left = random.randint(0, crop_width)
        #     crop_top = random.randint(0, crop_height)
        #     crop_right = self.clip_image_tensor.shape[2] - crop_width - crop_left
        #     crop_bottom = self.clip_image_tensor.shape[1] - crop_height - crop_top
        #     if len(self.clip_image_tensor.shape) == 3:
        #         self.clip_image_tensor = self.clip_image_tensor[:, crop_top:-crop_bottom, crop_left:-crop_right]
        #     elif len(self.clip_image_tensor.shape) == 4:
        #         self.clip_image_tensor = self.clip_image_tensor[:, :, crop_top:-crop_bottom, crop_left:-crop_right]

        if self.clip_image_processor is not None:
            # run it
            tensors_0_1 = self.clip_image_tensor.to(dtype=torch.float16)
            clip_out = self.clip_image_processor(
                images=tensors_0_1,
                return_tensors="pt",
                do_resize=True,
                do_rescale=False,
            ).pixel_values
            self.clip_image_tensor = clip_out.squeeze(0).clone().detach()

    def cleanup_clip_image(self: 'FileItemDTO'):
        self.clip_image_tensor = None
        self.clip_image_embeds = None




class AugmentationFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_augmentations = False
        self.unaugmented_tensor: Union[torch.Tensor, None] = None
        # self.augmentations: Union[None, List[Augments]] = None
        self.dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.aug_transform: Union[None, A.Compose] = None
        self.aug_replay_spatial_transforms = None
        self.build_augmentation_transform()

    def build_augmentation_transform(self: 'FileItemDTO'):
        if self.dataset_config.augmentations is not None and len(self.dataset_config.augmentations) > 0:
            self.has_augmentations = True
            augmentations = [Augments(**aug) for aug in self.dataset_config.augmentations]

            if self.dataset_config.shuffle_augmentations:
                random.shuffle(augmentations)

            augmentation_list = []
            for aug in augmentations:
                # make sure method name is valid
                assert hasattr(A, aug.method_name), f"invalid augmentation method: {aug.method_name}"
                # get the method
                method = getattr(A, aug.method_name)
                # add the method to the list
                augmentation_list.append(method(**aug.params))

            # add additional targets so we can augment the control image
            self.aug_transform = A.ReplayCompose(augmentation_list, additional_targets={'image2': 'image'})

    def augment_image(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose], ):

        # rebuild each time if shuffle
        if self.dataset_config.shuffle_augmentations:
            self.build_augmentation_transform()

        # save the original tensor
        self.unaugmented_tensor = transforms.ToTensor()(img) if transform is None else transform(img)

        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # apply augmentations
        transformed = self.aug_transform(image=open_cv_image)
        augmented = transformed["image"]

        # save just the spatial transforms for controls and masks
        augmented_params = transformed["replay"]
        spatial_transforms = ['Rotate', 'Flip', 'HorizontalFlip', 'VerticalFlip', 'Resize', 'Crop', 'RandomCrop',
                              'ElasticTransform', 'GridDistortion', 'OpticalDistortion']
        # only store the spatial transforms
        augmented_params['transforms'] = [t for t in augmented_params['transforms'] if t['__class_fullname__'].split('.')[-1] in spatial_transforms]

        if self.dataset_config.replay_transforms:
            self.aug_replay_spatial_transforms = augmented_params

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)

        return augmented_tensor

    # augment control images spatially consistent with transforms done to the main image
    def augment_spatial_control(self: 'FileItemDTO', img: Image, transform: Union[None, transforms.Compose] ):
        if self.aug_replay_spatial_transforms is None:
            # no transforms
            return transform(img)

        # save colorspace to convert back to
        colorspace = img.mode

        # convert to rgb
        img = img.convert('RGB')

        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Replay transforms
        transformed = A.ReplayCompose.replay(self.aug_replay_spatial_transforms, image=open_cv_image)
        augmented = transformed["image"]

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        # convert back to original colorspace
        augmented = augmented.convert(colorspace)

        augmented_tensor = transforms.ToTensor()(augmented) if transform is None else transform(augmented)
        return augmented_tensor

    def cleanup_control(self: 'FileItemDTO'):
        self.unaugmented_tensor = None


class MaskFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_mask_image = False
        self.mask_path: Union[str, None] = None
        self.mask_tensor: Union[torch.Tensor, None] = None
        self.use_alpha_as_mask: bool = False
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)
        self.mask_min_value = dataset_config.mask_min_value
        if dataset_config.alpha_mask:
            self.use_alpha_as_mask = True
            self.mask_path = kwargs.get('path', None)
            self.has_mask_image = True
        elif dataset_config.mask_path is not None:
            # find the control image path
            mask_path = dataset_config.mask_path if dataset_config.mask_path is not None else dataset_config.alpha_mask
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(mask_path, file_name_no_ext + ext)):
                    self.mask_path = os.path.join(mask_path, file_name_no_ext + ext)
                    self.has_mask_image = True
                    break

    def load_mask_image(self: 'FileItemDTO'):
        try:
            img = Image.open(self.mask_path)
            img = exif_transpose(img)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.mask_path}")

        if self.use_alpha_as_mask:
            # pipeline expectws an rgb image so we need to put alpha in all channels
            np_img = np.array(img)
            np_img[:, :, :3] = np_img[:, :, 3:]

            np_img = np_img[:, :, :3]
            img = Image.fromarray(np_img)

        img = img.convert('RGB')
        if self.dataset_config.invert_mask:
            img = ImageOps.invert(img)
        w, h = img.size
        fix_size = False
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            print_acc(f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            fix_size = True
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            print_acc(f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
            fix_size = True

        if fix_size:
            # swap all the sizes
            self.scale_to_width, self.scale_to_height = self.scale_to_height, self.scale_to_width
            self.crop_width, self.crop_height = self.crop_height, self.crop_width
            self.crop_x, self.crop_y = self.crop_y, self.crop_x




        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # randomly apply a blur up to 0.5% of the size of the min (width, height)
        min_size = min(img.width, img.height)
        blur_radius = int(min_size * random.random() * 0.005)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # make grayscale
        img = img.convert('L')

        if self.dataset_config.buckets:
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
            # crop
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))
        else:
            raise Exception("Mask images not supported for non-bucket datasets")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.aug_replay_spatial_transforms:
            self.mask_tensor = self.augment_spatial_control(img, transform=transform)
        else:
            self.mask_tensor = transform(img)
        self.mask_tensor = value_map(self.mask_tensor, 0, 1.0, self.mask_min_value, 1.0)
        # convert to grayscale

    def cleanup_mask(self: 'FileItemDTO'):
        self.mask_tensor = None


class UnconditionalFileItemDTOMixin:
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.has_unconditional = False
        self.unconditional_path: Union[str, None] = None
        self.unconditional_tensor: Union[torch.Tensor, None] = None
        self.unconditional_latent: Union[torch.Tensor, None] = None
        self.unconditional_transforms = self.dataloader_transforms
        dataset_config: 'DatasetConfig' = kwargs.get('dataset_config', None)

        if dataset_config.unconditional_path is not None:
            # we are using control images
            img_path = kwargs.get('path', None)
            file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            for ext in img_ext_list:
                if os.path.exists(os.path.join(dataset_config.unconditional_path, file_name_no_ext + ext)):
                    self.unconditional_path = os.path.join(dataset_config.unconditional_path, file_name_no_ext + ext)
                    self.has_unconditional = True
                    break

    def load_unconditional_image(self: 'FileItemDTO'):
        try:
            img = Image.open(self.unconditional_path)
            img = exif_transpose(img)
        except Exception as e:
            print_acc(f"Error: {e}")
            print_acc(f"Error loading image: {self.mask_path}")

        img = img.convert('RGB')
        w, h = img.size
        if w > h and self.scale_to_width < self.scale_to_height:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")
        elif h > w and self.scale_to_height < self.scale_to_width:
            # throw error, they should match
            raise ValueError(
                f"unexpected values: w={w}, h={h}, file_item.scale_to_width={self.scale_to_width}, file_item.scale_to_height={self.scale_to_height}, file_item.path={self.path}")

        if self.flip_x:
            # do a flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_y:
            # do a flip
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if self.dataset_config.buckets:
            # scale and crop based on file item
            img = img.resize((self.scale_to_width, self.scale_to_height), Image.BICUBIC)
            # img = transforms.CenterCrop((self.crop_height, self.crop_width))(img)
            # crop
            img = img.crop((
                self.crop_x,
                self.crop_y,
                self.crop_x + self.crop_width,
                self.crop_y + self.crop_height
            ))
        else:
            raise Exception("Unconditional images are not supported for non-bucket datasets")

        if self.aug_replay_spatial_transforms:
            self.unconditional_tensor = self.augment_spatial_control(img, transform=self.unconditional_transforms)
        else:
            self.unconditional_tensor = self.unconditional_transforms(img)

    def cleanup_unconditional(self: 'FileItemDTO'):
        self.unconditional_tensor = None
        self.unconditional_latent = None


class PoiFileItemDTOMixin:
    # Point of interest bounding box. Allows for dynamic cropping without cropping out the main subject
    # items in the poi will always be inside the image when random cropping
    def __init__(self: 'FileItemDTO', *args, **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        # poi is a name of the box point of interest in the caption json file
        dataset_config = kwargs.get('dataset_config', None)
        path = kwargs.get('path', None)
        self.poi: Union[str, None] = dataset_config.poi
        self.has_point_of_interest = self.poi is not None
        self.poi_x: Union[int, None] = None
        self.poi_y: Union[int, None] = None
        self.poi_width: Union[int, None] = None
        self.poi_height: Union[int, None] = None

        if self.poi is not None:
            # make sure latent caching is off
            if dataset_config.cache_latents or dataset_config.cache_latents_to_disk:
                raise Exception(
                    f"Error: poi is not supported when caching latents. Please set cache_latents and cache_latents_to_disk to False in the dataset config"
                )
                # make sure we are loading through json
            if dataset_config.caption_ext != 'json':
                raise Exception(
                    f"Error: poi is only supported when using json captions. Please set caption_ext to json in the dataset config"
                )
            self.poi = self.poi.strip()
            # get the caption path
            file_path_no_ext = os.path.splitext(path)[0]
            caption_path = file_path_no_ext + '.json'
            if not os.path.exists(caption_path):
                raise Exception(f"Error: caption file not found for poi: {caption_path}")
            with open(caption_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            if 'poi' not in json_data:
                print_acc(f"Warning: poi not found in caption file: {caption_path}")
            if self.poi not in json_data['poi']:
                print_acc(f"Warning: poi not found in caption file: {caption_path}")
            # poi has, x, y, width, height
            # do full image if no poi
            self.poi_x = 0
            self.poi_y = 0
            self.poi_width = self.width
            self.poi_height = self.height
            try:
                if self.poi in json_data['poi']:
                    poi = json_data['poi'][self.poi]
                    self.poi_x = int(poi['x'])
                    self.poi_y = int(poi['y'])
                    self.poi_width = int(poi['width'])
                    self.poi_height = int(poi['height'])
            except Exception as e:
                pass

            # handle flipping
            if kwargs.get('flip_x', False):
                # flip the poi
                self.poi_x = self.width - self.poi_x - self.poi_width
            if kwargs.get('flip_y', False):
                # flip the poi
                self.poi_y = self.height - self.poi_y - self.poi_height

    def setup_poi_bucket(self: 'FileItemDTO'):
        initial_width = int(self.width * self.dataset_config.scale)
        initial_height = int(self.height * self.dataset_config.scale)
        # we are using poi, so we need to calculate the bucket based on the poi

        # if img resolution is less than dataset resolution, just return and let the normal bucketing happen
        img_resolution = get_resolution(initial_width, initial_height)
        if img_resolution <= self.dataset_config.resolution:
            return False  # will trigger normal bucketing

        bucket_tolerance = self.dataset_config.bucket_tolerance
        poi_x = int(self.poi_x * self.dataset_config.scale)
        poi_y = int(self.poi_y * self.dataset_config.scale)
        poi_width = int(self.poi_width * self.dataset_config.scale)
        poi_height = int(self.poi_height * self.dataset_config.scale)

        # loop to keep expanding until we are at the proper resolution. This is not ideal, we can probably handle it better
        num_loops = 0
        while True:
            # crop left
            if poi_x > 0:
                poi_x = random.randint(0, poi_x)
            else:
                poi_x = 0

            # crop right
            cr_min = poi_x + poi_width
            if cr_min < initial_width:
                crop_right = random.randint(poi_x + poi_width, initial_width)
            else:
                crop_right = initial_width

            poi_width = crop_right - poi_x

            if poi_y > 0:
                poi_y = random.randint(0, poi_y)
            else:
                poi_y = 0

            if poi_y + poi_height < initial_height:
                crop_bottom = random.randint(poi_y + poi_height, initial_height)
            else:
                crop_bottom = initial_height

            poi_height = crop_bottom - poi_y
            try:
                # now we have our random crop, but it may be smaller than resolution. Check and expand if needed
                current_resolution = get_resolution(poi_width, poi_height)
            except Exception as e:
                print_acc(f"Error: {e}")
                print_acc(f"Error getting resolution: {self.path}")
                raise e
                return False
            if current_resolution >= self.dataset_config.resolution:
                # We can break now
                break
            else:
                num_loops += 1
                if num_loops > 100:
                    print_acc(
                        f"Warning: poi bucketing looped too many times. This should not happen. Please report this issue.")
                    return False

        new_width = poi_width
        new_height = poi_height

        bucket_resolution = get_bucket_for_image_size(
            new_width, new_height,
            resolution=self.dataset_config.resolution,
            divisibility=bucket_tolerance
        )

        width_scale_factor = bucket_resolution["width"] / new_width
        height_scale_factor = bucket_resolution["height"] / new_height
        # Use the maximum of the scale factors to ensure both dimensions are scaled above the bucket resolution
        max_scale_factor = max(width_scale_factor, height_scale_factor)

        self.scale_to_width = math.ceil(initial_width * max_scale_factor)
        self.scale_to_height = math.ceil(initial_height * max_scale_factor)
        self.crop_width = bucket_resolution['width']
        self.crop_height = bucket_resolution['height']
        self.crop_x = int(poi_x * max_scale_factor)
        self.crop_y = int(poi_y * max_scale_factor)

        if self.scale_to_width < self.crop_x + self.crop_width or self.scale_to_height < self.crop_y + self.crop_height:
            # todo look into this. This still happens sometimes
            print_acc('size mismatch')

        return True


class ArgBreakMixin:
    # just stops super calls form hitting object
    def __init__(self, *args, **kwargs):
        pass


class LatentCachingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self._encoded_latent: Union[torch.Tensor, None] = None
        self._cached_first_frame_latent: Union[torch.Tensor, None] = None
        self._cached_audio_latent: Union[torch.Tensor, None] = None
        self._cached_first_frame_condition: Union[torch.Tensor, None] = None
        self._latent_path: Union[str, None] = None
        self._latent_cache_key: Union[str, None] = None
        self._latent_storage_entry: Union[dict, None] = None
        self._latent_storage_checked = False
        self.is_latent_cached = False
        self.is_caching_to_disk = False
        self.is_caching_to_memory = False
        self.latent_load_device = 'cpu'
        # todo, increment this if we change the latent format to invalidate cache
        self.latent_version = 1

    def get_latent_info_dict(self: 'FileItemDTO'):
        item = OrderedDict([
            ("filename", os.path.basename(self.path)),
            ("scale_to_width", self.scale_to_width),
            ("scale_to_height", self.scale_to_height),
            ("crop_x", self.crop_x),
            ("crop_y", self.crop_y),
            ("crop_width", self.crop_width),
            ("crop_height", self.crop_height),
            ("latent_space_version", self.latent_space_version),
            ("latent_version", self.latent_version),
        ])
        # when adding items, do it after so we dont change old latents
        # Include the source file's size + mtime so that editing an image in place
        # (same path and bucket dims) invalidates its cached latent instead of
        # silently reusing a stale one. The first run after this is added re-caches
        # once because the key changes; unchanged files stay cached thereafter.
        try:
            _src_stat = os.stat(self.path)
            item["src_size"] = _src_stat.st_size
            item["src_mtime_ns"] = _src_stat.st_mtime_ns
        except OSError:
            # source file is gone (caches_only): use the stat persisted in the
            # size database so the key still matches the existing cache entry
            _cached_stat = getattr(self, 'cached_src_stat', None)
            if _cached_stat is not None:
                item["src_size"], item["src_mtime_ns"] = _cached_stat
        if self.flip_x:
            item["flip_x"] = True
        if self.flip_y:
            item["flip_y"] = True
        if self.dataset_config.num_frames > 1:
            item["num_frames"] = self.dataset_config.num_frames
            if self.dataset_config.do_i2v:
                item["do_i2v"] = True
        if self.dataset_config.do_audio:
            item["do_audio"] = True
            if self.dataset_config.audio_normalize:
                item["audio_normalize"] = True
            if self.dataset_config.audio_preserve_pitch:
                item["audio_preserve_pitch"] = True
        return item

    def get_latent_cache_dir(self: 'FileItemDTO'):
        return self.get_latent_cache_dir_candidates()[0]

    def get_latent_cache_dir_candidates(self: 'FileItemDTO'):
        return get_cache_dir_candidates(
            self.path,
            '_latent_cache',
            override_root=self.dataset_config.cache_dir_override,
            min_free_gb=self.dataset_config.cache_dir_override_min_free_gb,
        )

    def get_latent_cache_key(self: 'FileItemDTO', recalculate=False):
        if self._latent_cache_key is not None and not recalculate:
            return self._latent_cache_key

        hash_dict = self.get_latent_info_dict()
        hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
        hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
        self._latent_cache_key = hash_str.replace('=', '')
        self._latent_path = None
        self._latent_storage_entry = None
        self._latent_storage_checked = False
        return self._latent_cache_key

    def get_latent_path(self: 'FileItemDTO', recalculate=False):
        if self._latent_path is not None and not recalculate:
            return self._latent_path

        latent_dir = self.get_latent_cache_dir()
        filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
        cache_key = self.get_latent_cache_key(recalculate=recalculate)
        self._latent_path = os.path.join(latent_dir, f'{filename_no_ext}_{cache_key}.safetensors')
        return self._latent_path

    def set_latent_storage_entry(self: 'FileItemDTO', storage_entry: Union[dict, None]):
        self._latent_storage_entry = storage_entry
        self._latent_storage_checked = True

    def get_latent_storage_entry(self: 'FileItemDTO', recalculate=False):
        if self._latent_storage_checked and not recalculate:
            return self._latent_storage_entry

        cache_key = self.get_latent_cache_key(recalculate=recalculate)
        storage_entry = None

        for latent_dir in self.get_latent_cache_dir_candidates():
            buffer_key = (latent_dir, get_latent_shard_bucket_key(self))
            pending_map = lazy_latent_shard_buffers.get(buffer_key)
            if pending_map is not None and cache_key in pending_map:
                storage_entry = build_buffered_latent_storage_entry(
                    latent_dir,
                    cache_key,
                )
                break

            if has_pending_latent_state(latent_dir, cache_key):
                storage_entry = build_buffered_latent_storage_entry(
                    latent_dir,
                    cache_key,
                )
                break

            shard_entry = lookup_latent_index_entry(latent_dir, cache_key)
            if shard_entry is None:
                manifest = _maybe_migrate_legacy_latent_manifest(latent_dir)
                if manifest is not None:
                    shard_entry = manifest.get("items", {}).get(cache_key)
            if shard_entry is not None:
                shard_name = shard_entry.get("shard")
                if shard_name is not None:
                    candidate = build_latent_shard_storage_entry(
                        latent_dir,
                        shard_name,
                        shard_entry.get("key_prefix", cache_key),
                    )
                    if os.path.exists(candidate["path"]):
                        storage_entry = candidate
                        break

            latent_path = os.path.join(
                latent_dir,
                f'{os.path.splitext(os.path.basename(self.path))[0]}_{cache_key}.safetensors',
            )
            if os.path.exists(latent_path):
                storage_entry = {
                    "storage": "legacy",
                    "path": latent_path,
                }
                break

        if storage_entry is None and self.dataset_config.caches_only and not os.path.exists(self.path):
            storage_entry = self._find_caches_only_latent_fallback()

        self.set_latent_storage_entry(storage_entry)
        return storage_entry

    def _find_caches_only_latent_fallback(self: 'FileItemDTO'):
        # caches_only fallbacks for a deleted source file whose exact key missed.
        # 1) caches written before the src stat was part of the key: recompute the
        #    key without the stat fields and look it up again.
        hash_dict = self.get_latent_info_dict()
        if "src_size" in hash_dict or "src_mtime_ns" in hash_dict:
            alt_dict = OrderedDict(
                (k, v) for k, v in hash_dict.items()
                if k not in ("src_size", "src_mtime_ns")
            )
            hash_input = json.dumps(alt_dict, sort_keys=True).encode('utf-8')
            alt_key = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii').replace('=', '')
            for latent_dir in self.get_latent_cache_dir_candidates():
                shard_entry = lookup_latent_index_entry(latent_dir, alt_key)
                if shard_entry is None:
                    manifest = _maybe_migrate_legacy_latent_manifest(latent_dir)
                    if manifest is not None:
                        shard_entry = manifest.get("items", {}).get(alt_key)
                if shard_entry is not None and shard_entry.get("shard"):
                    candidate = build_latent_shard_storage_entry(
                        latent_dir,
                        shard_entry["shard"],
                        shard_entry.get("key_prefix", alt_key),
                    )
                    if os.path.exists(candidate["path"]):
                        return candidate
                legacy_path = os.path.join(
                    latent_dir,
                    f'{os.path.splitext(os.path.basename(self.path))[0]}_{alt_key}.safetensors',
                )
                if os.path.exists(legacy_path):
                    return {"storage": "legacy", "path": legacy_path}

        # 2) single unambiguous legacy per-file cache matching this filename. Only
        #    safe for non-flipped items (flip variants share the filename prefix).
        if not self.flip_x and not self.flip_y:
            filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
            for latent_dir in self.get_latent_cache_dir_candidates():
                try:
                    matches = glob.glob(
                        os.path.join(latent_dir, f'{glob.escape(filename_no_ext)}_*.safetensors')
                    )
                except OSError:
                    continue
                matches = [m for m in matches if not os.path.basename(m).startswith('bucket_')]
                if len(matches) == 1:
                    return {"storage": "legacy", "path": matches[0]}
        return None

    def cleanup_latent(self):
        if self._encoded_latent is not None:
            if not self.is_caching_to_memory:
                # we are caching on disk, don't save in memory
                self._encoded_latent = None
                self._cached_first_frame_latent = None
                self._cached_audio_latent = None
                self._cached_first_frame_condition = None
            else:
                # move it back to cpu
                self._encoded_latent = self._encoded_latent.to('cpu')
                if self._cached_first_frame_latent is not None:
                    self._cached_first_frame_latent = self._cached_first_frame_latent.to('cpu')
                if self._cached_audio_latent is not None:
                    self._cached_audio_latent = self._cached_audio_latent.to('cpu')
                if self._cached_first_frame_condition is not None:
                    self._cached_first_frame_condition = self._cached_first_frame_condition.to('cpu')

    def get_latent(self, device=None):
        if not self.is_latent_cached:
            return None
        if self._encoded_latent is None:
            hydrate_file_items_from_cached_latents([self])
        return self._encoded_latent


class LatentCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.latent_cache = {}

    def cache_latents_all_latents(self: 'AiToolkitDataset'):
        with accelerator.main_process_first():
            print_acc(f"Caching latents for {self.dataset_path}")
            # cache all latents to disk
            to_disk = self.is_caching_latents_to_disk
            to_memory = self.is_caching_latents_to_memory
            shard_size = max(1, self.dataset_config.latent_cache_shard_size)
            use_shards = to_disk and shard_size > 1

            if to_disk:
                if use_shards:
                    print_acc(f" - Saving latents to disk in shards of {shard_size}")
                else:
                    print_acc(" - Saving latents to disk")
            if to_memory:
                print_acc(" - Keeping latents in memory")
            # move sd items to cpu except for vae
            self.sd.set_device_state_preset('cache_latents')

            shard_manifests = {}
            shard_buffers = {}
            pending_batches: Dict[tuple, List['FileItemDTO']] = {}

            def infer_next_shard_index(manifest: dict) -> int:
                max_index = -1
                for entry in manifest.get("items", {}).values():
                    shard_name = entry.get("shard", "")
                    shard_root, shard_ext = os.path.splitext(shard_name)
                    if shard_ext != ".safetensors":
                        continue
                    try:
                        max_index = max(max_index, int(shard_root.rsplit("_", 1)[1]))
                    except (IndexError, ValueError):
                        continue
                return max_index + 1

            def get_or_create_manifest(latent_dir: str) -> dict:
                manifest = shard_manifests.get(latent_dir)
                if manifest is not None:
                    return manifest

                manifest = load_latent_shard_manifest(latent_dir)
                if manifest is None:
                    manifest = {
                        "version": LATENT_CACHE_SHARD_VERSION,
                        "shard_size": shard_size,
                        "next_shard_index": 0,
                        "items": {},
                    }
                else:
                    manifest["version"] = LATENT_CACHE_SHARD_VERSION
                    manifest["shard_size"] = shard_size
                    manifest.setdefault("items", {})
                    manifest.setdefault(
                        "next_shard_index",
                        infer_next_shard_index(manifest),
                    )

                shard_manifests[latent_dir] = manifest
                return manifest

            def flush_shard_buffer(buffer_key):
                pending_items = shard_buffers.get(buffer_key, [])
                if len(pending_items) == 0:
                    return

                latent_dir, bucket_key = buffer_key
                manifest = get_or_create_manifest(latent_dir)
                shard_index = manifest.get("next_shard_index", 0)
                shard_name = f"bucket_{bucket_key}_{shard_index:05d}.safetensors"
                shard_path = os.path.join(latent_dir, shard_name)

                shard_state_dict = OrderedDict()
                for pending_file_item, pending_state_dict in pending_items:
                    key_prefix = pending_file_item.get_latent_cache_key()
                    shard_state_dict[
                        get_latent_shard_tensor_key(key_prefix, "latent")
                    ] = pending_state_dict["latent"]
                    for tensor_name in LATENT_AUX_TENSOR_KEYS:
                        if tensor_name in pending_state_dict:
                            shard_state_dict[
                                get_latent_shard_tensor_key(key_prefix, tensor_name)
                            ] = pending_state_dict[tensor_name]

                shard_meta = get_meta_for_safetensors(
                    OrderedDict([
                        ("latent_cache_shard_version", LATENT_CACHE_SHARD_VERSION),
                        ("bucket", bucket_key),
                        ("items", len(pending_items)),
                    ])
                )
                os.makedirs(latent_dir, exist_ok=True)
                _atomic_save_file(shard_state_dict, shard_path, metadata=shard_meta)

                index_entries = []
                for pending_file_item, _ in pending_items:
                    key_prefix = pending_file_item.get_latent_cache_key()
                    manifest["items"][key_prefix] = {
                        "shard": shard_name,
                        "key_prefix": key_prefix,
                    }
                    index_entries.append((key_prefix, shard_name, key_prefix))
                    pending_file_item.set_latent_storage_entry(
                        build_latent_shard_storage_entry(
                            latent_dir,
                            shard_name,
                            key_prefix,
                        )
                    )
                    pending_file_item.is_latent_cached = True

                manifest["next_shard_index"] = shard_index + 1
                # Persist the REAL item->shard index entries (not just meta) so the
                # cache is found on the next run instead of re-encoding everything
                # and orphaning these shards. (Was: save_latent_shard_manifest_meta
                # with an empty entries list.)
                update_latent_index_entries(
                    latent_dir,
                    index_entries,
                    version=manifest.get("version", LATENT_CACHE_SHARD_VERSION),
                    shard_size=manifest.get("shard_size"),
                    next_shard_index=manifest.get("next_shard_index"),
                )
                shard_buffers[buffer_key] = []

            def is_cuda_oom_error(error: Exception) -> bool:
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    return True
                return "out of memory" in str(error).lower()

            def get_target_cache_batch_size() -> int:
                if self.sd.device_torch.type != "cuda":
                    return max(1, min(4, self.batch_size))
                if self.dataset_config.num_frames > 1:
                    if self.batch_size > 1:
                        return max(1, min(4, self.batch_size))
                    return 2
                if self.batch_size > 1:
                    return max(2, min(16, self.batch_size * 4))
                return 8

            def get_target_cache_load_workers() -> int:
                if self.dataset_config.num_frames > 1:
                    return 1
                cpu_count = os.cpu_count() or 1
                return max(1, min(8, cpu_count, target_cache_batch_size))

            def get_latent_cache_batch_key(file_item: 'FileItemDTO'):
                if self.dataset_config.num_frames > 1:
                    return (
                        "video",
                        self.dataset_config.num_frames,
                        file_item.crop_height,
                        file_item.crop_width,
                    )
                return (
                    "image",
                    file_item.crop_height,
                    file_item.crop_width,
                )

            def load_file_item_for_latent_cache(file_item: 'FileItemDTO'):
                file_item.load_and_process_image(self.transform, only_load_latents=True)
                return file_item

            def get_first_frame_tensor(file_item: 'FileItemDTO'):
                frames = file_item.tensor
                if len(frames.shape) == 3:
                    return frames
                if len(frames.shape) == 4:
                    return frames[0]
                raise ValueError(f"Unknown frame shape {frames.shape}")

            def save_latent_state(file_item: 'FileItemDTO', state_dict: OrderedDict):
                if not to_disk:
                    return

                if use_shards:
                    buffer_key = (
                        file_item.get_latent_cache_dir(),
                        get_latent_shard_bucket_key(file_item),
                    )
                    shard_buffers.setdefault(buffer_key, []).append((file_item, state_dict))
                    if len(shard_buffers[buffer_key]) >= shard_size:
                        flush_shard_buffer(buffer_key)
                    return

                latent_path = file_item.get_latent_path(recalculate=True)
                meta = get_meta_for_safetensors(file_item.get_latent_info_dict())
                os.makedirs(os.path.dirname(latent_path), exist_ok=True)
                _atomic_save_file(state_dict, latent_path, metadata=meta)
                file_item.set_latent_storage_entry({
                    "storage": "legacy",
                    "path": latent_path,
                })

            def finalize_uncached_file_item(
                    file_item: 'FileItemDTO',
                    latent: torch.Tensor,
                    first_frame_latent: Union[torch.Tensor, None] = None,
                    audio_latent: Union[torch.Tensor, None] = None,
                    first_frame_condition: Union[torch.Tensor, None] = None,
            ):
                state_dict = OrderedDict()
                if to_disk:
                    state_dict["latent"] = latent.clone().detach().cpu()
                    if first_frame_latent is not None:
                        state_dict["first_frame_latent"] = (
                            first_frame_latent.clone().detach().cpu()
                        )
                    if audio_latent is not None:
                        state_dict["audio_latent"] = audio_latent.clone().detach().cpu()
                    if first_frame_condition is not None:
                        state_dict["first_frame_condition"] = (
                            first_frame_condition.clone().detach().cpu()
                        )
                    save_latent_state(file_item, state_dict)

                if to_memory:
                    assign_latent_tensors(
                        file_item,
                        latent,
                        first_frame_latent,
                        audio_latent,
                        first_frame_condition,
                        dtype=self.sd.torch_dtype,
                    )

                file_item.cleanup()
                file_item.is_latent_cached = True

            # Same rule as the lazy path (BaseSDTrainProcess._encode_images_for_lazy_cache):
            # under low_vram or layer offloading the VAE encodes one image at a time.
            # get_target_cache_batch_size() returns 8 for batch_size 1 regardless of
            # VRAM, so this pre-pass used to push 8 full-resolution images through the
            # VAE at once -- at 1024 that is ~1 GB of activations per image, which
            # OOM'd a 16 GB card on the very first batch while the lazy path (one at
            # a time, *during training*) had been fine. Images still load and group
            # in batches, so disk/decode keeps its parallelism; only the encode is split.
            _latent_model_config = getattr(self.sd, 'model_config', None)
            chunk_latent_encode = any([
                getattr(_latent_model_config, 'low_vram', False),
                getattr(_latent_model_config, 'layer_offloading', False),
            ])

            def encode_latent_batch(tensors):
                if not chunk_latent_encode or len(tensors) <= 1:
                    return self.sd.encode_images(tensors)
                return torch.cat([self.sd.encode_images([t]) for t in tensors], dim=0)

            def process_uncached_batch(
                    file_items: List['FileItemDTO'],
                    max_batch_size: Union[int, None] = None,
            ) -> int:
                if len(file_items) == 0:
                    return 0

                if max_batch_size is None:
                    max_batch_size = target_cache_batch_size

                if len(file_items) > max_batch_size:
                    processed = 0
                    for start in range(0, len(file_items), max_batch_size):
                        processed += process_uncached_batch(
                            file_items[start:start + max_batch_size],
                            max_batch_size=max_batch_size,
                        )
                    return processed

                unloaded_items = [x for x in file_items if x.tensor is None]
                if len(unloaded_items) > 0:
                    try:
                        if target_cache_load_workers > 1 and len(unloaded_items) > 1:
                            with ThreadPoolExecutor(
                                    max_workers=min(target_cache_load_workers, len(unloaded_items))
                            ) as executor:
                                list(executor.map(load_file_item_for_latent_cache, unloaded_items))
                        else:
                            for file_item in unloaded_items:
                                load_file_item_for_latent_cache(file_item)
                    except Exception as e:
                        print_acc(
                            f"Error loading latent cache batch starting with {file_items[0].path}"
                        )
                        print_acc(f"Error: {str(e)}")
                        raise

                latents = None
                first_frame_latents = None
                first_frame_conditions = None
                audio_latents = None
                audio_index_pairs = []
                retry_batch_size = None
                try:
                    latents = encode_latent_batch([x.tensor for x in file_items])

                    if self.dataset_config.num_frames > 1 and self.dataset_config.do_i2v:
                        need_first_frame_latents = self.sd.uses_i2v_first_frame_latents()
                        need_first_frame_conditions = self.sd.uses_i2v_first_frame_conditions()
                        first_frames = [get_first_frame_tensor(x) for x in file_items]
                        if need_first_frame_latents:
                            first_frame_latents = self.sd.encode_images(
                                first_frames
                            )
                        if need_first_frame_conditions:
                            first_frame_conditions = self.sd.build_i2v_first_frame_conditions(
                                latents=latents,
                                first_frames=torch.stack(first_frames),
                                first_frame_latents=first_frame_latents,
                                device=self.sd.device_torch,
                                dtype=self.sd.torch_dtype,
                            )

                    audio_index_pairs = [
                        (idx, x.audio_data)
                        for idx, x in enumerate(file_items)
                        if x.audio_data is not None
                    ]
                    if len(audio_index_pairs) > 0:
                        audio_latents = self.sd.encode_audio(
                            [audio_data for _, audio_data in audio_index_pairs]
                        )
                except Exception as e:
                    if not (len(file_items) > 1 and is_cuda_oom_error(e)):
                        print_acc(f"Error processing latent cache batch starting with {file_items[0].path}")
                        print_acc(f"Error: {str(e)}")
                        raise
                    retry_batch_size = max(1, len(file_items) // 2)

                if retry_batch_size is not None:
                    # Retry only AFTER leaving the except block. Inside it, the OOM's
                    # traceback still references every frame of the failed encode --
                    # including the VAE's activation tensors -- so flush() freed
                    # nothing and each smaller retry started on top of the previous
                    # attempt's memory (a run went 12.28 -> 13.35 -> 13.63 -> 14.41 GiB
                    # allocated while the batch shrank 8 -> 4 -> 2 -> 1, then died).
                    latents = first_frame_latents = first_frame_conditions = audio_latents = None
                    flush()
                    print_acc(
                        f"Latent cache batch of {len(file_items)} hit OOM. "
                        f"Retrying with batch size {retry_batch_size}"
                    )
                    processed = 0
                    for start in range(0, len(file_items), retry_batch_size):
                        processed += process_uncached_batch(
                            file_items[start:start + retry_batch_size],
                            max_batch_size=retry_batch_size,
                        )
                    return processed

                audio_latent_map = {}
                if audio_latents is not None:
                    for (idx, _), audio_latent in zip(audio_index_pairs, audio_latents):
                        audio_latent_map[idx] = audio_latent

                for idx, file_item in enumerate(file_items):
                    finalize_uncached_file_item(
                        file_item,
                        latents[idx],
                        first_frame_latents[idx] if first_frame_latents is not None else None,
                        audio_latent_map.get(idx),
                        first_frame_conditions[idx] if first_frame_conditions is not None else None,
                    )

                del latents
                if first_frame_latents is not None:
                    del first_frame_latents
                if first_frame_conditions is not None:
                    del first_frame_conditions
                if audio_latents is not None:
                    del audio_latents
                return len(file_items)

            target_cache_batch_size = get_target_cache_batch_size()
            target_cache_load_workers = get_target_cache_load_workers()
            cache_iteration_file_list = self.get_active_bucket_file_items()
            uncached_target_count = 0
            for file_item in cache_iteration_file_list:
                storage_entry = file_item.get_latent_storage_entry(recalculate=True)
                should_migrate_legacy_cache = (
                    use_shards
                    and storage_entry is not None
                    and storage_entry["storage"] == "legacy"
                )
                if storage_entry is None or should_migrate_legacy_cache:
                    uncached_target_count += 1
            if target_cache_batch_size > 1:
                cache_iteration_file_list.sort(
                    key=lambda file_item: (
                        get_latent_cache_batch_key(file_item),
                        file_item.path,
                    )
                )
            if target_cache_batch_size > 1:
                print_acc(
                    f" - Encoding uncached latents in batches of up to {target_cache_batch_size}"
                )
                print_acc(" - Grouping latent cache pass by bucket dimensions")
                if chunk_latent_encode:
                    print_acc(" - Low-VRAM mode: the VAE encodes one image at a time within each batch")
            if target_cache_load_workers > 1:
                print_acc(
                    f" - Loading uncached images with up to {target_cache_load_workers} worker threads"
                )

            cache_started_at = time.monotonic()
            cached_items_completed = 0

            def update_cache_progress(processed_delta: int):
                nonlocal cached_items_completed
                if processed_delta <= 0:
                    return
                cached_items_completed += processed_delta
                self.report_cache_progress(
                    "latents",
                    cached_items_completed,
                    uncached_target_count,
                    time.monotonic() - cache_started_at,
                )

            progress = tqdm(
                total=len(cache_iteration_file_list),
                desc=f'Caching latents{" to disk" if to_disk else ""}',
            )
            try:
                for file_item in cache_iteration_file_list:
                    file_item.is_caching_to_disk = to_disk
                    file_item.is_caching_to_memory = to_memory
                    file_item.latent_load_device = self.sd.device

                    storage_entry = file_item.get_latent_storage_entry()
                    should_migrate_legacy_cache = (
                        use_shards
                        and storage_entry is not None
                        and storage_entry["storage"] == "legacy"
                    )

                    if storage_entry is not None and not should_migrate_legacy_cache:
                        if to_memory:
                            hydrate_file_items_from_cached_latents(
                                [file_item],
                                dtype=self.sd.torch_dtype,
                            )
                        file_item.is_latent_cached = True
                        progress.update(1)
                        continue

                    if should_migrate_legacy_cache:
                        state_dict = load_file(storage_entry["path"], device="cpu")
                        if to_memory:
                            assign_latent_tensors(
                                file_item,
                                state_dict["latent"],
                                state_dict.get("first_frame_latent"),
                                state_dict.get("audio_latent"),
                                state_dict.get("first_frame_condition"),
                                dtype=self.sd.torch_dtype,
                            )

                        buffer_key = (
                            file_item.get_latent_cache_dir(),
                            get_latent_shard_bucket_key(file_item),
                        )
                        shard_buffers.setdefault(buffer_key, []).append((
                            file_item,
                            OrderedDict(
                                [
                                    ("latent", state_dict["latent"]),
                                ]
                            ),
                        ))
                        for tensor_name in LATENT_AUX_TENSOR_KEYS:
                            if tensor_name in state_dict:
                                shard_buffers[buffer_key][-1][1][tensor_name] = (
                                    state_dict[tensor_name]
                                )

                        if len(shard_buffers[buffer_key]) >= shard_size:
                            flush_shard_buffer(buffer_key)

                        file_item.is_latent_cached = True
                        progress.update(1)
                        update_cache_progress(1)
                        continue

                    batch_key = get_latent_cache_batch_key(file_item)
                    pending_batches.setdefault(batch_key, []).append(file_item)
                    if len(pending_batches[batch_key]) >= target_cache_batch_size:
                        processed_count = process_uncached_batch(
                            pending_batches.pop(batch_key)
                        )
                        progress.update(processed_count)
                        update_cache_progress(processed_count)

                for batch_key in list(pending_batches.keys()):
                    processed_count = process_uncached_batch(
                        pending_batches.pop(batch_key)
                    )
                    progress.update(processed_count)
                    update_cache_progress(processed_count)

                for buffer_key in list(shard_buffers.keys()):
                    flush_shard_buffer(buffer_key)
            finally:
                if uncached_target_count > 0:
                    self.report_cache_progress(
                        "latents",
                        cached_items_completed,
                        uncached_target_count,
                        time.monotonic() - cache_started_at,
                    )
                progress.close()
                self.sd.restore_device_state()


class TextEmbeddingFileItemDTOMixin:
    def __init__(self, *args, **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(*args, **kwargs)
        self.prompt_embeds: Union[PromptEmbeds, None] = None
        self._text_embedding_path: Union[str, None] = None
        self._text_embedding_cache_key: Union[str, None] = None
        self._text_embedding_storage_entry: Union[dict, None] = None
        self._text_embedding_storage_checked = False
        self.is_text_embedding_cached = False
        self.text_embedding_load_device = 'cpu'
        # Z-Image stores prompt embeddings as a list of per-prompt 2D tensors.
        # Older cache entries split those tensors incorrectly, so keep them out
        # of future lookups without invalidating other model families.
        text_embedding_space_version = getattr(self, "text_embedding_space_version", "")
        self.text_embedding_version = 2 if text_embedding_space_version.startswith("zimage") else 1

    def get_text_embedding_info_dict(self: 'FileItemDTO'):
        # make sure the caption is loaded here
        # TODO: we need a way to cache all the other features like trigger words, DOP, etc. For now, we need to throw an error if not compatible.
        if self.caption is None:
            self.load_caption()
        item = OrderedDict([
            ("caption", self.caption),
            ("text_embedding_space_version", self.text_embedding_space_version),
            ("text_embedding_version", self.text_embedding_version),
        ])
        # if we have a control image, cache the path
        if self.encode_control_in_text_embeddings and self.control_path is not None:
            item["control_path"] = self.control_path
        return item

    def get_text_embedding_cache_dir(self: 'FileItemDTO'):
        return self.get_text_embedding_cache_dir_candidates()[0]

    def get_text_embedding_cache_dir_candidates(self: 'FileItemDTO'):
        return get_cache_dir_candidates(
            self.path,
            '_t_e_cache',
            override_root=self.dataset_config.cache_dir_override,
            min_free_gb=self.dataset_config.cache_dir_override_min_free_gb,
        )

    def get_text_embedding_cache_key(self: 'FileItemDTO', recalculate=False):
        if self._text_embedding_cache_key is not None and not recalculate:
            return self._text_embedding_cache_key

        hash_dict = self.get_text_embedding_info_dict()
        hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
        hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
        self._text_embedding_cache_key = hash_str.replace('=', '')
        self._text_embedding_path = None
        self._text_embedding_storage_entry = None
        self._text_embedding_storage_checked = False
        return self._text_embedding_cache_key

    def get_text_embedding_path(self: 'FileItemDTO', recalculate=False):
        if self._text_embedding_path is not None and not recalculate:
            return self._text_embedding_path
        else:
            te_dir = self.get_text_embedding_cache_dir()
            filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
            cache_key = self.get_text_embedding_cache_key(recalculate=recalculate)
            self._text_embedding_path = os.path.join(te_dir, f'{filename_no_ext}_{cache_key}.safetensors')

        return self._text_embedding_path

    def set_text_embedding_storage_entry(self: 'FileItemDTO', storage_entry: Union[dict, None]):
        self._text_embedding_storage_entry = storage_entry
        self._text_embedding_storage_checked = True

    def get_text_embedding_storage_entry(self: 'FileItemDTO', recalculate=False):
        if self._text_embedding_storage_checked and not recalculate:
            return self._text_embedding_storage_entry

        cache_key = self.get_text_embedding_cache_key(recalculate=recalculate)
        storage_entry = None

        for te_dir in self.get_text_embedding_cache_dir_candidates():
            manifest = load_text_embed_shard_manifest(te_dir)
            if manifest is not None:
                shard_entry = manifest.get("items", {}).get(cache_key)
                if shard_entry is not None:
                    shard_name = shard_entry.get("shard")
                    if shard_name is not None:
                        candidate = build_text_embed_shard_storage_entry(
                            te_dir,
                            shard_name,
                            shard_entry.get("key_prefix", cache_key),
                        )
                        if os.path.exists(candidate["path"]):
                            storage_entry = candidate
                            break

            text_embedding_path = os.path.join(
                te_dir,
                f'{os.path.splitext(os.path.basename(self.path))[0]}_{cache_key}.safetensors',
            )
            if os.path.exists(text_embedding_path):
                storage_entry = {
                    "storage": "legacy",
                    "path": text_embedding_path,
                }
                break

        if storage_entry is None and self.dataset_config.caches_only:
            # caches_only: if the caption sidecar is gone and its text was never
            # persisted, the recomputed key can't match. Fall back to a single
            # unambiguous legacy per-file cache for this filename.
            caption_path = os.path.splitext(self.path)[0] + self.dataset_config.caption_ext
            if not os.path.exists(caption_path) and getattr(self, 'cached_caption_text', None) is None:
                filename_no_ext = os.path.splitext(os.path.basename(self.path))[0]
                for te_dir in self.get_text_embedding_cache_dir_candidates():
                    try:
                        matches = glob.glob(
                            os.path.join(te_dir, f'{glob.escape(filename_no_ext)}_*.safetensors')
                        )
                    except OSError:
                        continue
                    if len(matches) == 1:
                        storage_entry = {"storage": "legacy", "path": matches[0]}
                        break

        self.set_text_embedding_storage_entry(storage_entry)
        return storage_entry

    def cleanup_text_embedding(self):
        if self.prompt_embeds is not None:
            # we are caching on disk, don't save in memory
            self.prompt_embeds = None

    def load_prompt_embedding(self, device=None):
        if not self.is_text_embedding_cached:
            return
        if self.prompt_embeds is None:
            hydrate_file_items_from_cached_text_embeddings([self])

class TextEmbeddingCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.is_caching_text_embeddings = self.dataset_config.cache_text_embeddings

    def cache_text_embeddings(self: 'AiToolkitDataset'):
        with accelerator.main_process_first():
            print_acc(f"Caching text_embeddings for {self.dataset_path}")
            print_acc(" - Saving text embeddings to disk")
            
            did_move = False

            encode_batch_size = max(1, min(32, self.dataset_config.text_embedding_cache_shard_size))
            cache_started_at = time.monotonic()
            processed_items = 0
            active_file_items = self.get_active_bucket_file_items()
            total_items = len(active_file_items)
            pending_file_items = []
            progress = tqdm(total=total_items, desc='Caching text embeddings to disk')

            def encode_and_cache_pending():
                nonlocal did_move
                if len(pending_file_items) == 0:
                    return

                if not did_move:
                    self.sd.set_device_state_preset('cache_text_encoder')
                    did_move = True

                batch_prompt_embeds = encode_prompt_embeds_for_file_items(self.sd, pending_file_items)
                split_embeds = split_prompt_embeds(batch_prompt_embeds, len(pending_file_items))
                cache_file_item_text_embeddings_to_disk(
                    list(zip(pending_file_items, split_embeds)),
                    self.dataset_config.text_embedding_cache_shard_size,
                )
                pending_file_items.clear()
                del batch_prompt_embeds

            try:
                for file_item in active_file_items:
                    file_item.load_caption(self.caption_dict)
                    storage_entry = file_item.get_text_embedding_storage_entry(recalculate=True)
                    if storage_entry is not None:
                        file_item.is_text_embedding_cached = True
                    else:
                        pending_file_items.append(file_item)
                        if len(pending_file_items) >= encode_batch_size:
                            encode_and_cache_pending()

                    processed_items += 1
                    progress.update(1)
                    self.report_cache_progress(
                        "text_embeddings",
                        processed_items,
                        total_items,
                        time.monotonic() - cache_started_at,
                    )
                encode_and_cache_pending()
            finally:
                if total_items > 0:
                    self.report_cache_progress(
                        "text_embeddings",
                        processed_items,
                        total_items,
                        time.monotonic() - cache_started_at,
                    )
                progress.close()
            # restore device state
            # if did_move:
            #     self.sd.restore_device_state()


class CLIPCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        # if we have super, call it
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
        self.clip_vision_num_unconditional_cache = 20
        self.clip_vision_unconditional_cache = []

    def cache_clip_vision_to_disk(self: 'AiToolkitDataset'):
        if not self.is_caching_clip_vision_to_disk:
            return
        with torch.no_grad():
            print_acc(f"Caching clip vision for {self.dataset_path}")

            print_acc(" - Saving clip to disk")
            # move sd items to cpu except for vae
            self.sd.set_device_state_preset('cache_clip')

            # make sure the adapter has attributes
            if self.sd.adapter is None:
                raise Exception("Error: must have an adapter to cache clip vision to disk")

            clip_image_processor: CLIPImageProcessor = None
            if hasattr(self.sd.adapter, 'clip_image_processor'):
                clip_image_processor = self.sd.adapter.clip_image_processor

            if clip_image_processor is None:
                raise Exception("Error: must have a clip image processor to cache clip vision to disk")

            vision_encoder: CLIPVisionModelWithProjection = None
            if hasattr(self.sd.adapter, 'image_encoder'):
                vision_encoder = self.sd.adapter.image_encoder
            if hasattr(self.sd.adapter, 'vision_encoder'):
                vision_encoder = self.sd.adapter.vision_encoder

            if vision_encoder is None:
                raise Exception("Error: must have a vision encoder to cache clip vision to disk")

            # move vision encoder to device
            vision_encoder.to(self.sd.device)

            is_quad = self.sd.adapter.config.quad_image
            image_encoder_path = self.sd.adapter.config.image_encoder_path

            dtype = self.sd.torch_dtype
            device = self.sd.device_torch
            if hasattr(self.sd.adapter, 'clip_noise_zero') and self.sd.adapter.clip_noise_zero:
                # just to do this, we did :)
                # need more samples as it is random noise
                self.clip_vision_num_unconditional_cache = self.clip_vision_num_unconditional_cache
            else:
                # only need one since it doesnt change
                self.clip_vision_num_unconditional_cache = 1

            # cache unconditionals
            print_acc(f" - Caching {self.clip_vision_num_unconditional_cache} unconditional clip vision to disk")
            unconditional_paths = []
            pending_file_items = []
            for file_item in self.get_active_bucket_file_items():
                embedding_path = file_item.get_clip_vision_embeddings_path(recalculate=True)
                if not os.path.exists(embedding_path):
                    pending_file_items.append(file_item)
            cache_started_at = time.monotonic()
            total_pending_items = len(pending_file_items)
            processed_items = 0

            is_noise_zero = hasattr(self.sd.adapter, 'clip_noise_zero') and self.sd.adapter.clip_noise_zero

            for i in range(self.clip_vision_num_unconditional_cache):
                hash_dict = OrderedDict([
                    ("image_encoder_path", image_encoder_path),
                    ("is_quad", is_quad),
                    ("is_noise_zero", is_noise_zero),
                ])
                # get base64 hash of md5 checksum of hash_dict
                hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
                hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
                hash_str = hash_str.replace('=', '')

                uncond_path_candidates = get_cache_file_candidates(
                    self.dataset_config.clip_image_path,
                    f'uncond_{hash_str}_{i}.safetensors',
                    override_root=self.dataset_config.cache_dir_override,
                    min_free_gb=self.dataset_config.cache_dir_override_min_free_gb,
                )
                existing_uncond_path = next(
                    (path for path in uncond_path_candidates if os.path.exists(path)),
                    None,
                )
                uncond_path = existing_uncond_path or uncond_path_candidates[0]
                if existing_uncond_path is not None:
                    # skip it
                    unconditional_paths.append(uncond_path)
                    continue

                # generate a random image
                img_shape = (1, 3, self.sd.adapter.input_size, self.sd.adapter.input_size)
                if is_noise_zero:
                    tensors_0_1 = torch.rand(img_shape).to(device, dtype=torch.float32)
                else:
                    tensors_0_1 = torch.zeros(img_shape).to(device, dtype=torch.float32)
                clip_image = clip_image_processor(
                    images=tensors_0_1,
                    return_tensors="pt",
                    do_resize=True,
                    do_rescale=False,
                ).pixel_values

                if is_quad:
                    # split the 4x4 grid and stack on batch
                    ci1, ci2 = clip_image.chunk(2, dim=2)
                    ci1, ci3 = ci1.chunk(2, dim=3)
                    ci2, ci4 = ci2.chunk(2, dim=3)
                    clip_image = torch.cat([ci1, ci2, ci3, ci4], dim=0).detach()

                clip_output = vision_encoder(
                    clip_image.to(device, dtype=dtype),
                    output_hidden_states=True
                )
                # make state_dict ['last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
                state_dict = OrderedDict([
                    ('image_embeds', clip_output.image_embeds.clone().detach().cpu()),
                    ('last_hidden_state', clip_output.hidden_states[-1].clone().detach().cpu()),
                    ('penultimate_hidden_states', clip_output.hidden_states[-2].clone().detach().cpu()),
                ])

                os.makedirs(os.path.dirname(uncond_path), exist_ok=True)
                _atomic_save_file(state_dict, uncond_path)
                unconditional_paths.append(uncond_path)

            self.clip_vision_unconditional_cache = unconditional_paths

            # use tqdm to show progress
            i = 0
            for file_item in tqdm(self.file_list, desc=f'Caching clip vision to disk'):
                file_item.is_caching_clip_vision_to_disk = True
                file_item.clip_vision_load_device = self.sd.device
                file_item.clip_vision_is_quad = is_quad
                file_item.clip_image_encoder_path = image_encoder_path
                file_item.clip_vision_unconditional_paths = unconditional_paths
                if file_item.has_clip_augmentations:
                    raise Exception("Error: clip vision caching is not supported with clip augmentations")

                embedding_path = file_item.get_clip_vision_embeddings_path(recalculate=True)
                # check if it is saved to disk already
                if not os.path.exists(embedding_path):
                    # load the image first
                    file_item.load_clip_image()
                    # add batch dimension
                    clip_image = file_item.clip_image_tensor.unsqueeze(0).to(device, dtype=dtype)

                    if is_quad:
                        # split the 4x4 grid and stack on batch
                        ci1, ci2 = clip_image.chunk(2, dim=2)
                        ci1, ci3 = ci1.chunk(2, dim=3)
                        ci2, ci4 = ci2.chunk(2, dim=3)
                        clip_image = torch.cat([ci1, ci2, ci3, ci4], dim=0).detach()

                    clip_output = vision_encoder(
                        clip_image.to(device, dtype=dtype),
                        output_hidden_states=True
                    )

                    # make state_dict ['last_hidden_state', 'image_embeds', 'penultimate_hidden_states']
                    state_dict = OrderedDict([
                        ('image_embeds', clip_output.image_embeds.clone().detach().cpu()),
                        ('last_hidden_state', clip_output.hidden_states[-1].clone().detach().cpu()),
                        ('penultimate_hidden_states', clip_output.hidden_states[-2].clone().detach().cpu()),
                    ])
                    # metadata
                    meta = get_meta_for_safetensors(file_item.get_clip_vision_info_dict())
                    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
                    _atomic_save_file(state_dict, embedding_path, metadata=meta)

                    del clip_image
                    del clip_output
                    del file_item.clip_image_tensor

                    # flush(garbage_collect=False)
                    processed_items += 1
                    self.report_cache_progress(
                        "clip_vision",
                        processed_items,
                        total_pending_items,
                        time.monotonic() - cache_started_at,
                    )
                file_item.is_vision_clip_cached = True
                i += 1
            # flush every 100
            # if i % 100 == 0:
            #     flush()

            if total_pending_items > 0:
                self.report_cache_progress(
                    "clip_vision",
                    processed_items,
                    total_pending_items,
                    time.monotonic() - cache_started_at,
                )

        # restore device state
        self.sd.restore_device_state()



class ControlCachingMixin:
    def __init__(self: 'AiToolkitDataset', **kwargs):
        if hasattr(super(), '__init__'):
            super().__init__(**kwargs)
            self.control_generator: ControlGenerator = None
    
    def add_control_path_to_file_item(self: 'AiToolkitDataset', file_item: 'FileItemDTO', control_path: str, control_type: ControlTypes):
        if control_type == 'inpaint':
            file_item.inpaint_path = control_path
            file_item.has_inpaint_image = True
        elif control_type == 'mask':
            file_item.mask_path = control_path
            file_item.has_mask_image = True
        else:
            if file_item.control_path is None:
                file_item.control_path = [control_path]
            elif isinstance(file_item.control_path, str):
                file_item.control_path = [file_item.control_path, control_path]
            elif isinstance(file_item.control_path, list):
                file_item.control_path.append(control_path)
            else:
                raise Exception(f"Error: control_path is not a string or list: {file_item.control_path}")
            file_item.has_control_image = True

    def setup_controls(self: 'AiToolkitDataset'):
        if not self.is_generating_controls:
            return
        with torch.no_grad():
            print_acc(f"Generating controls for {self.dataset_path}")
            device = self.sd.device
            
            self.control_generator = ControlGenerator(
                device=device,
                sd=self.sd,
            )

            # use tqdm to show progress
            for file_item in tqdm(self.file_list, desc=f'Generating Controls'):
                for control_type in self.dataset_config.controls:
                    # generates the control if it is not already there
                    control_path = self.control_generator.get_control_path(file_item.path, control_type)
                    if control_path is not None:
                        self.add_control_path_to_file_item(file_item, control_path, control_type)
                
            # remove models
            self.control_generator.cleanup()
            self.control_generator = None
            
            flush()
