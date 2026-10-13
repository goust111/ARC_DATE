from __future__ import annotations

import itertools
from typing import Literal

import numpy as np


BACKGROUND = 0
COLOR_POOL = np.arange(1, 10, dtype=int)
MAX_RETRIES = 400
VALID_DIMS = [
    (7, 9),
    (9, 7),
    (9, 11),
    (11, 9),
    (9, 13),
    (13, 9),
    (11, 13),
    (13, 11),
]
CORNER_NAMES = ("TL", "TR", "BL", "BR")


def _corner_positions(height: int, width: int) -> dict[str, tuple[int, int]]:
    return {
        "TL": (0, 0),
        "TR": (0, width - 1),
        "BL": (height - 1, 0),
        "BR": (height - 1, width - 1),
    }


def _find_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    coords = np.argwhere(g != BACKGROUND)
    markers = [(int(r), int(c), int(g[r, c])) for r, c in coords]
    markers.sort(key=lambda item: (item[0], item[1], item[2]))
    return markers


def _sort_marker_indices(
    markers: list[tuple[int, int, int]],
    counts: list[int],
) -> list[int]:
    return sorted(
        range(len(markers)),
        key=lambda idx: (-int(counts[idx]), int(markers[idx][0]), int(markers[idx][1])),
    )


def _build_histogram(
    markers: list[tuple[int, int, int]],
    counts: list[int],
) -> np.ndarray:
    height = max(int(v) for v in counts)
    width = len(markers)
    out = np.zeros((height, width), dtype=int)
    for col, idx in enumerate(_sort_marker_indices(markers, counts)):
        _, _, color = markers[idx]
        out[: int(counts[idx]), int(col)] = int(color)
    return out


def _distance_value(
    r: int,
    c: int,
    r0: int,
    c0: int,
    metric: Literal["chebyshev", "euclidean_sq", "manhattan"],
) -> int:
    if metric == "chebyshev":
        return max(abs(r - r0), abs(c - c0))
    if metric == "euclidean_sq":
        return (r - r0) * (r - r0) + (c - c0) * (c - c0)
    return abs(r - r0) + abs(c - c0)


def _reference_analysis(
    grid: np.ndarray,
    *,
    ownership_metric: Literal["chebyshev", "euclidean_sq", "manhattan"] = "chebyshev",
    parity_metric: Literal["chebyshev", "manhattan"] = "chebyshev",
) -> dict:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    markers = _find_markers(g)
    stats = [
        {
            "marker": marker,
            "owned_total": 0,
            "even_count": 0,
            "odd_count": 0,
        }
        for marker in markers
    ]
    unowned = 0

    for r in range(height):
        for c in range(width):
            distances = [
                _distance_value(r, c, r0, c0, ownership_metric)
                for r0, c0, _color in markers
            ]
            best = min(distances)
            if distances.count(best) != 1:
                unowned += 1
                continue

            owner = distances.index(best)
            r0, c0, _color = markers[owner]
            parity_distance = _distance_value(r, c, r0, c0, parity_metric)
            stats[owner]["owned_total"] += 1
            if parity_distance % 2 == 0:
                stats[owner]["even_count"] += 1
            else:
                stats[owner]["odd_count"] += 1

    return {"markers": markers, "stats": stats, "unowned": unowned}


def _reference_output(grid: np.ndarray, **kwargs) -> np.ndarray:
    analysis = _reference_analysis(grid, **kwargs)
    counts = [int(item["even_count"]) for item in analysis["stats"]]
    return _build_histogram(analysis["markers"], counts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert corner-marker ring-territory counts into a descending top-aligned histogram.

    Input structure:
    - A black background grid contains two or three distinct non-black single-pixel markers.
    - Each marker sits in a corner, and all other cells are empty.

    Rule:
    - For every cell, compute its Chebyshev distance to each corner marker.
    - A cell belongs to the unique nearest marker; tied cells stay unowned.
    - For each marker, count owned cells whose Chebyshev distance from that marker is even, including the marker itself.
    - Build a black histogram whose height is the largest count and whose width is the number of markers, ordering columns from left to right by descending count and filling each column from the top with the marker color.

    Key insight:
    - The bar heights are not read directly from the input; they emerge from Voronoi-like corner ownership under Chebyshev distance, then a second even-ring filter.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    markers = _find_markers(g)
    if not (2 <= len(markers) <= 3):
        raise ValueError("expected 2 or 3 markers")

    height, width = map(int, g.shape)
    rows = np.arange(height, dtype=int)[None, :, None]
    cols = np.arange(width, dtype=int)[None, None, :]
    marker_rows = np.array([r for r, _c, _color in markers], dtype=int)[:, None, None]
    marker_cols = np.array([c for _r, c, _color in markers], dtype=int)[:, None, None]

    distances = np.maximum(np.abs(rows - marker_rows), np.abs(cols - marker_cols))
    best = distances.min(axis=0)
    unique_owner = (distances == best[None, :, :]).sum(axis=0) == 1
    owner = distances.argmin(axis=0)

    counts: list[int] = []
    for idx in range(len(markers)):
        even_ring = distances[idx] % 2 == 0
        owned = unique_owner & (owner == idx)
        counts.append(int(np.count_nonzero(owned & even_ring)))

    return _build_histogram(markers, counts)


def _sample_grid() -> np.ndarray:
    height, width = VALID_DIMS[int(np.random.randint(len(VALID_DIMS)))]
    present_names = list(np.random.choice(np.array(CORNER_NAMES), size=3, replace=False))
    colors = np.random.choice(COLOR_POOL, size=3, replace=False)

    grid = np.zeros((int(height), int(width)), dtype=int)
    for name, color in zip(present_names, colors):
        r, c = _corner_positions(height, width)[str(name)]
        grid[int(r), int(c)] = int(color)
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    analysis = _reference_analysis(grid)
    markers = analysis["markers"]
    counts = [int(item["even_count"]) for item in analysis["stats"]]
    totals = [int(item["owned_total"]) for item in analysis["stats"]]
    odd_counts = [int(item["odd_count"]) for item in analysis["stats"]]

    if len(markers) != 3:
        return False
    if len(set(counts)) != len(counts):
        return False
    if max(counts) > 30:
        return False
    if analysis["unowned"] <= 0:
        return False
    if any(total <= 1 for total in totals):
        return False
    if any(odd <= 0 for odd in odd_counts):
        return False

    expected = _build_histogram(markers, counts)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return False
    if not np.array_equal(main(grid), expected):
        return False

    no_parity = _build_histogram(markers, totals)
    if np.array_equal(expected, no_parity):
        return False

    euclidean = _reference_output(grid, ownership_metric="euclidean_sq")
    if np.array_equal(expected, euclidean):
        return False

    manhattan_owner = _reference_output(grid, ownership_metric="manhattan")
    if np.array_equal(expected, manhattan_owner):
        return False

    manhattan_parity = _reference_output(grid, parity_metric="manhattan")
    if np.array_equal(expected, manhattan_parity):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _sample_grid()
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid ring-territory histogram input")


if __name__ == "__main__":
    for i in range(3):
        sample = generate_input()
        print(f"sample {i}: input={sample.shape}, output={main(sample).shape}")
