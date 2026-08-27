"""VecCore -- HNSW, Product Quantization and BM25, from scratch in C++17.

A thin Python package over the pybind11 module. The C++ library knows nothing
about Python; this is the seam, and it is one-directional by design
(`PLAN.md` §3).

    import veccore
    index = veccore.HnswIndex(vectors, M=16, ef_construction=200)
    ids, distances = index.search(query, k=10, ef_search=64)
    new_id = index.add(vector)                    # incremental, thread-safe

Distances are **squared L2** throughout -- `sqrt` is monotonic so it cannot
change a ranking, and skipping it removes an expensive instruction from the
hottest loop (`CONTEXT.md` D4). If you need true Euclidean distance, take the
square root at the boundary, not inside the index.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Two ways the extension can be present, and both have to work.
#
#   1. Installed from a wheel, where it sits inside this package as
#      ``veccore._veccore``.
#   2. Built out-of-tree by CMake -- ``CONTEXT.md`` D2 puts the build directory
#      on the Linux filesystem rather than next to the source on /mnt/d, and
#      that is the flow every benchmark in this repo uses. Nothing is installed.
#
# The installed case is tried first, so a stale build directory left in $HOME
# cannot silently shadow a properly installed package with an older extension.
_BUILD_DIRS = [
    Path.home() / "veccore-build",
    Path(__file__).resolve().parent.parent.parent / "build",
]

_NAMES = ("Bm25Index", "FlatIndex", "HnswIndex", "PqIndex", "reciprocal_rank_fusion")

try:
    from ._veccore import (  # noqa: F401
        Bm25Index,
        FlatIndex,
        HnswIndex,
        PqIndex,
        reciprocal_rank_fusion,
    )
except ImportError:
    for _d in _BUILD_DIRS:
        if _d.is_dir() and str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
    try:
        from _veccore import (  # noqa: F401
            Bm25Index,
            FlatIndex,
            HnswIndex,
            PqIndex,
            reciprocal_rank_fusion,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the _veccore extension is not available. Either install the package:\n"
            "  pip install .\n"
            "or build it out-of-tree, which is what the benchmarks use:\n"
            "  cmake -S . -B ~/veccore-build -G Ninja -DCMAKE_BUILD_TYPE=Release \\\n"
            "        -Dpybind11_DIR=$(python -c 'import pybind11;print(pybind11.get_cmake_dir())')\n"
            "  cmake --build ~/veccore-build\n"
            f"(searched for a built extension in: {', '.join(str(d) for d in _BUILD_DIRS)})"
        ) from exc

__all__ = [
    "HnswIndex",
    "FlatIndex",
    "PqIndex",
    "Bm25Index",
    "reciprocal_rank_fusion",
    "VecCoreIndex",
]

from .edgerag import VecCoreIndex  # noqa: E402
