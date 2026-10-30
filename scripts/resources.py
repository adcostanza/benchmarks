#!/usr/bin/env python3
"""Resource bookkeeping for one server run (called by serve.sh).

  resources.py cached <dir>                 fraction of <dir>'s bytes in the page cache (0-1)
  resources.py summary <run-dir> <key=value>...
      merges key=value pairs and the peaks from <run-dir>/resources.csv into
      <run-dir>/resources.json
"""

import ctypes
import ctypes.util
import csv
import json
import mmap
import os
import sys
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]


def cached_bytes(path):
    """Bytes of `path` resident in the page cache, via mincore(2)."""
    size = path.stat().st_size
    if size == 0:
        return 0, 0
    pages = (size + PAGE - 1) // PAGE
    vec = (ctypes.c_ubyte * pages)()
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), f"mmap {path}")
        try:
            if libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), f"mincore {path}")
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)
    return min(sum(b & 1 for b in vec) * PAGE, size), size


def cached(directory):
    hit = total = 0
    for p in Path(directory).rglob("*"):
        if p.is_file() and not p.is_symlink():
            h, t = cached_bytes(p)
            hit += h
            total += t
    print(f"{hit / total:.3f}" if total else "0")


def summary(run_dir, pairs):
    run_dir = Path(run_dir)
    out_file = run_dir / "resources.json"
    data = json.loads(out_file.read_text()) if out_file.exists() else {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            data[key] = json.loads(value)
        except json.JSONDecodeError:
            data[key] = value
    samples = run_dir / "resources.csv"
    if samples.exists():
        rows = list(csv.DictReader(samples.open()))
        if rows:
            data["gpu_mem_peak_mib"] = max(int(r["gpu_mem_mib"]) for r in rows)
            data["container_mem_peak_gib"] = round(max(float(r["container_mem_gib"]) for r in rows), 2)
            data["host_mem_used_peak_gib"] = round(max(float(r["host_mem_used_gib"]) for r in rows), 2)
            data["samples"] = len(rows)
    out_file.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "cached":
        cached(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "summary":
        summary(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
