import os
import shutil
from typing import List, Union


def normalize_source_dir(source_path: str) -> str:
    normalized_path = os.path.abspath(source_path)
    if os.path.isdir(normalized_path):
        return os.path.normpath(normalized_path)
    return os.path.normpath(os.path.dirname(normalized_path))


def get_existing_disk_usage_path(path: str) -> str:
    current_path = os.path.abspath(path)
    while not os.path.exists(current_path):
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            drive, _ = os.path.splitdrive(current_path)
            if drive:
                return drive + os.sep
            return os.sep
        current_path = parent_path
    return current_path


def has_sufficient_free_space(path: Union[str, None], min_free_gb: Union[int, float, None]) -> bool:
    if not path:
        return False

    threshold_gb = 0.0 if min_free_gb is None else max(0.0, float(min_free_gb))
    try:
        usage = shutil.disk_usage(get_existing_disk_usage_path(path))
    except Exception:
        return False

    return usage.free >= int(threshold_gb * (1024 ** 3))


def get_mirrored_source_dir(source_path: str, override_root: str) -> str:
    source_dir = normalize_source_dir(source_path)
    override_root = os.path.abspath(override_root)
    drive, tail = os.path.splitdrive(source_dir)
    drive_part = drive.replace(":", "") if drive else "_root"
    tail_parts = [part for part in os.path.normpath(tail).split(os.sep) if part]
    return os.path.join(override_root, drive_part, *tail_parts)


def _dedupe_paths(paths: List[str]) -> List[str]:
    unique_paths = []
    seen_paths = set()
    for path in paths:
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        unique_paths.append(path)
    return unique_paths


def get_cache_root_candidates(
        source_path: str,
        override_root: Union[str, None] = None,
) -> List[str]:
    source_dir = normalize_source_dir(source_path)
    candidates = [source_dir]
    if override_root:
        candidates.append(get_mirrored_source_dir(source_dir, override_root))
    return _dedupe_paths(candidates)


def get_cache_dir_candidates(
        source_path: str,
        cache_dir_name: str,
        override_root: Union[str, None] = None,
        min_free_gb: Union[int, float, None] = None,
) -> List[str]:
    source_dir = normalize_source_dir(source_path)
    default_dir = os.path.join(source_dir, cache_dir_name)

    if not override_root:
        return [default_dir]

    override_dir = os.path.join(
        get_mirrored_source_dir(source_dir, override_root),
        cache_dir_name,
    )
    if has_sufficient_free_space(override_root, min_free_gb):
        return _dedupe_paths([override_dir, default_dir])
    return _dedupe_paths([default_dir, override_dir])


def get_cache_file_candidates(
        source_path: str,
        file_name: str,
        override_root: Union[str, None] = None,
        min_free_gb: Union[int, float, None] = None,
) -> List[str]:
    cache_root_candidates = get_cache_root_candidates(source_path, override_root)
    if len(cache_root_candidates) == 1:
        return [os.path.join(cache_root_candidates[0], file_name)]

    default_path = os.path.join(cache_root_candidates[0], file_name)
    override_path = os.path.join(cache_root_candidates[1], file_name)
    if has_sufficient_free_space(override_root, min_free_gb):
        return _dedupe_paths([override_path, default_path])
    return _dedupe_paths([default_path, override_path])
