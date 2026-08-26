"""VecCore -- HNSW, Product Quantization and BM25, from scratch in C++17.

A thin Python package over the pybind11 module. The C++ library knows nothing
about Python; this is the seam, and it is one-directional by design
(`PLAN.md` §3).

    import veccore
    index = veccore.HnswIndex(vectors, M=16, ef_construction=200)
    ids, distances = index.search(query, k=10, ef_search=64)

Distances are **squared L2** throughout -- `sqrt` is monotonic so it cannot
change a ranking, and skipping it removes an expensive instruction from the
hottest loop (`CONTEXT.md` D4). If you need true Euclidean distance, take the
square root at the boundary, not inside the index.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The extension is built out-of-tree (CONTEXT.md D2: the build directory lives
# on the Linux filesystem, not next to the source on /mnt/d). Look there before
# giving up, so the package works without an install step.
_BUILD_DIRS = [
    Path.home() / "veccore-build",
    Path(__file__).resolve().parent.parent.parent / "build",
]
for _d in _BUILD_DIRS:
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

try:
    from _veccore import (  # noqa: F401
        Bm25Index,
        FlatIndex,
        HnswIndex,
        reciprocal_rank_fusion,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "the _veccore extension is not built. From the repo root:\n"
        "  cmake -S . -B ~/veccore-build -G Ninja -DCMAKE_BUILD_TYPE=Release \\\n"
        "        -Dpybind11_DIR=$(python -c 'import pybind11;print(pybind11.get_cmake_dir())')\n"
        "  cmake --build ~/veccore-build\n"
        f"(searched: {', '.join(str(d) for d in _BUILD_DIRS)})"
    ) from exc

__all__ = ["HnswIndex", "FlatIndex", "Bm25Index", "reciprocal_rank_fusion", "VecCoreIndex"]

from .edgerag import VecCoreIndex  # noqa: E402
