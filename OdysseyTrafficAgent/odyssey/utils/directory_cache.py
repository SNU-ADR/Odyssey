"""Bounded cache for validated JSON asset directories, invalidated by file metadata."""
from copy import deepcopy
from functools import lru_cache, wraps
from pathlib import Path


def directory_cache(loader):
    @lru_cache(maxsize=4)
    def cached(directory, signature):
        return loader(directory)

    @wraps(loader)
    def load(directory):
        directory = Path(directory).resolve()
        signature = []
        for path in sorted(directory.rglob('*.json')):
            stat = path.stat()
            signature.append((str(path.relative_to(directory)), stat.st_size,
                              stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
        # Callers cannot mutate the validated value retained for the next request.
        return deepcopy(cached(str(directory), tuple(signature)))
    return load
