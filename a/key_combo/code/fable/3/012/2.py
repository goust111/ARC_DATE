from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    idx: int
    color: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def w(self) -> int:
        return self.c1 - self.c0 + 1

    def contains(self, r: int, c: int) -> bool:
        return self.r0 <= r <= self.r1 and self.c0 <= c <= self.c1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _unique_marker(g: np.ndarray) -> tuple[int, tuple[int, int]]:
    counts = Counter(int(v) for v in g.reshape(-1))
    singletons = [color for color, count in counts.items() if count == 1]
    assert len(singletons) == 1, "expected exactly one single-cell marker color"
    marker = int(singletons[0])
    coords = np.argwhere(g == marker)
    assert coords.shape[0] == 1
    r, c = map(int, coords[0])
    return marker, (r, c)


def _region_map(shape: tuple[int, int], regions: list[Region]) -> np.ndarray:
    region_at = np.full(shape, -1, dtype=int)
    for expected_idx, region in enumerate(regions):
        assert region.idx == expected_idx
        for r in range(region.r0, region.r1 + 1):
            for c in range(region.c0, region.c1 + 1):
                assert region_at[r, c] == -1, "rectangular regions must not overlap"
                region_at[r, c] = region.idx
    assert np.all(region_at >= 0), "rectangular regions must tile the grid"
    return region_at


def _regions_from_grid(
    g: np.ndarray, marker: int, marker_cell: tuple[int, int]
) -> tuple[list[Region], np.ndarray]:
    regions: list[Region] = []
    marker_r, marker_c = marker_cell
    for color in sorted(int(v) for v in np.unique(g) if int(v) != marker):
        for coords in _components4(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            crop = g[r0 : r1 + 1, c0 : c1 + 1]
            marker_count = int(np.count_nonzero(crop == marker))
            assert marker_count <= 1, "a region can contain at most the one input marker"
            assert np.all((crop == color) | (crop == marker)), (
                "each component bbox must be one solid rectangle, possibly with the marker hole"
            )
            assert coords.shape[0] == crop.size - marker_count, (
                "expected a complete rectangular region"
            )
            if marker_count:
                assert r0 <= marker_r <= r1 and c0 <= marker_c <= c1
            regions.append(Region(len(regions), color, r0, c0, r1, c1))

    region_at = np.full(g.shape, -1, dtype=int)
    for region in regions:
        for r in range(region.r0, region.r1 + 1):
            for c in range(region.c0, region.c1 + 1):
                assert region_at[r, c] == -1, "rectangular regions must not overlap"
                value = int(g[r, c])
                assert value == region.color or (value == marker and (r, c) == marker_cell)
                region_at[r, c] = region.idx
    assert np.all(region_at >= 0), "rectangular regions must tile the grid"
    return regions, region_at


def _propagate(
    grid: np.ndarray,
    marker: int,
    start: tuple[int, int],
    regions: list[Region],
    region_at: np.ndarray,
    *,
    collect_stats: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    out = grid.copy()
    q = deque([tuple(map(int, start))])
    queued = {int(region_at[start])}
    reached: set[int] = set()
    stats = {"wide": 0, "tall": 0, "square": 0, "edge": 0, "regions": 0}

    while q:
        r, c = q.popleft()
        rid = int(region_at[r, c])
        assert rid >= 0, "marker must land inside a known rectangle"
        assert rid in queued
        queued.remove(rid)
        assert rid not in reached, "propagation path must not revisit a rectangle"
        reached.add(rid)
        stats["regions"] += 1

        region = regions[rid]
        assert region.contains(r, c)

        if region.h == region.w:
            out[r, c] = marker
            stats["square"] += 1
            continue

        endpoints: list[tuple[int, int, int, int]]
        if region.w > region.h:
            assert region.c0 <= c <= region.c1
            out[region.r0 : region.r1 + 1, c] = marker
            endpoints = [(region.r0, c, -1, 0), (region.r1, c, 1, 0)]
            stats["wide"] += 1
        else:
            assert region.r0 <= r <= region.r1
            out[r, region.c0 : region.c1 + 1] = marker
            endpoints = [(r, region.c0, 0, -1), (r, region.c1, 0, 1)]
            stats["tall"] += 1

        for er, ec, dr, dc in endpoints:
            nr = er + dr
            nc = ec + dc
            if not _in_bounds(grid.shape, nr, nc):
                stats["edge"] += 1
                continue
            nrid = int(region_at[nr, nc])
            assert nrid >= 0 and nrid != rid
            assert nrid not in reached and nrid not in queued, (
                "each rectangle may be reached at most once"
            )
            q.append((nr, nc))
            queued.add(nrid)

    if collect_stats:
        stats["marker_cells"] = int(np.count_nonzero(out == marker))
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trigger a chain of divider lines from the unique marker through adjacent rectangles.

    Input structure:
    - The grid is a row/column-aligned mosaic of solid monochrome rectangular regions.
    - Exactly one rare marker pixel appears inside one non-square rectangle.
    - Rectangle adjacencies form an acyclic propagation path, so no region is reached twice.

    Rule:
    - In each reached wide rectangle, draw the marker-colored full column through the marker.
    - In each reached tall rectangle, draw the marker-colored full row through the marker.
    - The two line endpoints wake the first cell across the boundary in adjacent rectangles;
      a square rectangle only colors that arrived cell, and the grid edge stops that branch.

    Key insight:
    - The divider direction is chosen from each reached rectangle's own aspect ratio, so
      the initial marker launches a chain whose direction can flip until a square or edge
      absorbs the propagation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    marker, start = _unique_marker(g)
    regions, region_at = _regions_from_grid(g, marker, start)
    start_region = regions[int(region_at[start])]
    assert start_region.h != start_region.w, "the input marker must start in a non-square region"
    out, _stats = _propagate(g, marker, start, regions, region_at)
    return out


# --------------- generator ---------------

_GEN_ATTEMPTS = 2000
_LAYOUTS = ((3, 3), (2, 4), (4, 2), (2, 3), (3, 2))
_SIZE_VALUES = np.array([2, 3, 4, 5, 6], dtype=int)
_COLORS = np.arange(10, dtype=int)


def _sample_sizes(count: int) -> list[int]:
    return [int(v) for v in np.random.choice(_SIZE_VALUES, size=count, replace=True)]


def _build_partition() -> tuple[np.ndarray, list[Region], int]:
    rows_n, cols_n = _LAYOUTS[int(np.random.randint(len(_LAYOUTS)))]
    row_heights = _sample_sizes(rows_n)
    col_widths = _sample_sizes(cols_n)

    if np.random.random() < 0.85:
        value = int(np.random.choice(_SIZE_VALUES))
        row_heights[int(np.random.randint(rows_n))] = value
        col_widths[int(np.random.randint(cols_n))] = value

    height = int(sum(row_heights))
    width = int(sum(col_widths))
    assert height <= 30 and width <= 30

    marker = int(np.random.choice(_COLORS))
    region_colors = np.random.choice(_COLORS[_COLORS != marker], size=rows_n * cols_n, replace=False)

    grid = np.empty((height, width), dtype=int)
    regions: list[Region] = []
    r0 = 0
    color_i = 0
    for h in row_heights:
        c0 = 0
        for w in col_widths:
            color = int(region_colors[color_i])
            color_i += 1
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            regions.append(Region(len(regions), color, r0, c0, r1, c1))
            c0 += w
        r0 += h
    return grid, regions, marker


def _candidate_starts(regions: list[Region]) -> list[Region]:
    starts = [region for region in regions if region.h != region.w and region.h >= 3 and region.w >= 3]
    order = np.random.permutation(len(starts)) if starts else []
    return [starts[int(i)] for i in order]


def _place_initial_marker(base: np.ndarray, region: Region, marker: int) -> tuple[np.ndarray, tuple[int, int]]:
    r = int(np.random.randint(region.r0 + 1, region.r1))
    c = int(np.random.randint(region.c0 + 1, region.c1))
    grid = base.copy()
    grid[r, c] = marker
    return grid, (r, c)


def _good_training_case(expected: np.ndarray, marker: int, stats: dict[str, int]) -> bool:
    if stats["regions"] < 3:
        return False
    if stats["wide"] == 0 or stats["tall"] == 0:
        return False
    if stats["square"] == 0 or stats["edge"] == 0:
        return False
    marker_cells = int(np.count_nonzero(expected == marker))
    return 4 <= marker_cells <= max(4, expected.size // 2)


def _gen_attempt() -> np.ndarray | None:
    base, regions, marker = _build_partition()
    region_at = _region_map(base.shape, regions)

    for start_region in _candidate_starts(regions):
        for _ in range(4):
            inp, start = _place_initial_marker(base, start_region, marker)
            try:
                expected, stats = _propagate(
                    inp, marker, start, regions, region_at, collect_stats=True
                )
            except AssertionError:
                continue
            if not _good_training_case(expected, marker, stats):
                continue
            try:
                actual = main(inp)
            except AssertionError:
                continue
            if not np.array_equal(actual, expected):
                continue
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid divider-chain input")
