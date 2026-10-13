from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np


SMALLEST_REGION_COLOR = 8  # teal
LARGEST_REGION_COLOR = 1  # blue

_BG_CHOICES = (0, 5, 7)
_OBJECT_COLOR_POOL = tuple(c for c in range(10) if c not in {SMALLEST_REGION_COLOR, LARGEST_REGION_COLOR})
_GENERATED_SIGNATURES: set[str] = set()
_PROFILE_BAG: list[tuple[str, int, int, bool]] = []


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    excluded = set() if exclude is None else {int(x) for x in exclude}
    return [int(v) for v in np.unique(grid) if int(v) not in excluded]


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    out: list[dict] = []
    for coords in _components_from_mask(grid == int(color)):
        out.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox(coords),
                "area": int(coords.shape[0]),
            }
        )
    return out


def _assert_solid_rectangle(grid: np.ndarray, component: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    color = int(component["color"])
    area = int(component["area"])
    expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert area == expected_area, "each foreground component must be a solid rectangle"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be fully filled"
    return int(r0), int(c0), int(r1), int(c1)


def _facing_corners(
    upper: tuple[int, int, int, int],
    lower: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    ur0, uc0, ur1, uc1 = upper
    lr0, lc0, lr1, lc1 = lower
    assert ur1 < lr0, "paired rectangles must be vertically separated"

    if uc1 < lc0:
        return (ur1, uc1), (lr0, lc0)
    if lc1 < uc0:
        return (ur1, uc0), (lr0, lc1)
    raise AssertionError("paired rectangles must be diagonally offset")


def _draw_connectors(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, list[tuple[int, int]]]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    segments: list[tuple[int, list[tuple[int, int]]]] = []

    for color in _colors_present(g, exclude={bg}):
        components = _components_for_color(g, color)
        assert len(components) == 2, "each foreground color must form exactly two rectangles"

        rectangles = [_assert_solid_rectangle(g, comp) for comp in components]
        rectangles.sort(key=lambda b: (b[0], b[1]))
        start, end = _facing_corners(rectangles[0], rectangles[1])
        row_delta = int(end[0] - start[0])
        col_delta = int(end[1] - start[1])
        assert row_delta > 0 and abs(row_delta) == abs(col_delta), "facing corners must lie on one 45-degree diagonal"

        dc = 1 if col_delta > 0 else -1
        rr = int(start[0] + 1)
        cc = int(start[1] + dc)
        segment: list[tuple[int, int]] = []
        while (rr, cc) != end:
            assert int(g[rr, cc]) == bg, "diagonal connector must occupy background in the input"
            assert int(out[rr, cc]) == bg, "connectors must not overlap"
            out[rr, cc] = int(color)
            segment.append((rr, cc))
            rr += 1
            cc += dc
        segments.append((int(color), segment))

    return out, segments


def _background_regions(grid: np.ndarray, bg: int | None = None) -> list[np.ndarray]:
    bg_color = _get_bg(grid) if bg is None else int(bg)
    return _components_from_mask(np.asarray(grid, dtype=int) == bg_color)


def _recolor_extreme_regions(connected_grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(connected_grid, dtype=int)
    bg = _get_bg(g)
    regions = _background_regions(g, bg)
    assert regions, "expected at least one background region after drawing connectors"

    areas = [int(region.shape[0]) for region in regions]
    min_area = min(areas)
    max_area = max(areas)
    assert min_area < max_area, "valid puzzles must not have all background regions tied"

    out = g.copy()
    for region, area in zip(regions, areas):
        fill = None
        if area == min_area:
            fill = SMALLEST_REGION_COLOR
        elif area == max_area:
            fill = LARGEST_REGION_COLOR
        if fill is not None:
            out[region[:, 0], region[:, 1]] = int(fill)
    return out, areas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each missing diagonal connector, then recolor the smallest and largest background regions created by those new walls.

    Input structure:
    - One dominant background color fills most of the grid.
    - Every foreground color appears as exactly two separated solid rectangles.
    - For each color, the two rectangles face each other diagonally, with their facing corners on one 45-degree line; some rectangles may touch the grid border.

    Rule:
    - Keep every rectangle unchanged.
    - For each same-color pair, draw the open diagonal segment between the two facing corners, excluding the endpoint pixels already belonging to rectangles.
    - After all connectors are added, find the 4-connected background regions: recolor every minimum-area region teal, recolor every maximum-area region blue, and leave all other background cells unchanged.

    Key insight:
    - The regions that must be recolored are defined only after the diagonal walls are completed, so the connector step is causally necessary before any area comparison makes sense.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30

    connected, _segments = _draw_connectors(g)
    out, _areas = _recolor_extreme_regions(connected)
    return out


def _sample_line_parameters(n: int, pair_count: int, rect_size: int, side: str) -> list[int] | None:
    min_gap = rect_size + 2 + random.randint(0, 2)
    if side == "left":
        low = rect_size + 2
        high = n - rect_size - 1
    else:
        low = 0
        high = n - 2 * rect_size - 2

    max_base = high - (min_gap - 1) * (pair_count - 1)
    if max_base < low + pair_count - 1:
        return None

    bases = sorted(random.sample(range(low, max_base + 1), pair_count))
    return [int(base + i * (min_gap - 1)) for i, base in enumerate(bases)]


def _minimum_grid_size(pair_count: int, rect_size: int) -> int:
    return 2 * rect_size + 2 + (rect_size + 2) * (pair_count - 1)


def _choose_profile() -> tuple[str, int, int, bool]:
    global _PROFILE_BAG
    if not _PROFILE_BAG:
        _PROFILE_BAG = []
        for side in ("left", "right"):
            for rect_size in (2, 3):
                for pair_count in (3, 4, 5):
                    _PROFILE_BAG.append((side, rect_size, pair_count, False))
        _PROFILE_BAG.extend(
            [
                ("right", 2, 3, True),
                ("left", 2, 4, True),
                ("right", 3, 3, True),
            ]
        )
        random.shuffle(_PROFILE_BAG)
    return _PROFILE_BAG.pop()


def _build_input_from_profile(
    side: str,
    rect_size: int,
    pair_count: int,
    require_tie: bool,
) -> np.ndarray | None:
    min_n = _minimum_grid_size(pair_count, rect_size)
    if min_n > 30:
        return None

    n = random.randint(min_n, 30)
    bg = random.choice(_BG_CHOICES)
    colors = random.sample([c for c in _OBJECT_COLOR_POOL if c != bg], pair_count)
    params = _sample_line_parameters(n, pair_count, rect_size, side)
    if params is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for color, a in zip(colors, params):
        grid[0:rect_size, a : a + rect_size] = int(color)
        if side == "left":
            grid[a : a + rect_size, 0:rect_size] = int(color)
        else:
            row0 = n - rect_size - a
            grid[row0 : row0 + rect_size, n - rect_size : n] = int(color)

    try:
        connected, segments = _draw_connectors(grid)
        out, areas = _recolor_extreme_regions(connected)
    except AssertionError:
        return None

    if require_tie and areas.count(min(areas)) == 1 and areas.count(max(areas)) == 1:
        return None

    bg_regions_before = _background_regions(grid, bg)
    bg_regions_after = _background_regions(connected, bg)
    if len(bg_regions_after) <= len(bg_regions_before):
        return None
    if len(bg_regions_after) < pair_count + 1:
        return None

    try:
        direct_fill, _ = _recolor_extreme_regions(grid)
    except AssertionError:
        direct_fill = None
    if direct_fill is not None and np.array_equal(direct_fill, out):
        return None

    for skipped_color, _skipped_segment in segments:
        partial = grid.copy()
        for color, segment in segments:
            if color == skipped_color:
                continue
            for r, c in segment:
                partial[r, c] = int(color)
        try:
            partial_out, _ = _recolor_extreme_regions(partial)
        except AssertionError:
            continue
        if np.array_equal(partial_out, out):
            return None

    signature = hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()
    if signature in _GENERATED_SIGNATURES:
        return None
    _GENERATED_SIGNATURES.add(signature)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        side, rect_size, pair_count, require_tie = _choose_profile()
        for _attempt in range(80):
            grid = _build_input_from_profile(side, rect_size, pair_count, require_tie)
            if grid is not None:
                return grid
        if require_tie:
            for _attempt in range(80):
                grid = _build_input_from_profile(side, rect_size, pair_count, False)
                if grid is not None:
                    return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
