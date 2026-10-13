from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]

BACKGROUND = 0
OUTPUT_COLOR = 5
INPUT_COLORS = [1, 2, 3, 4, 5]
GEN_RETRIES = 400
MIN_SEGMENT_CELLS = 5


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            pts: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _diag_segment(p0: Coord, p1: Coord) -> list[Coord]:
    r0, c0 = map(int, p0)
    r1, c1 = map(int, p1)
    dr = r1 - r0
    dc = c1 - c0
    assert dr != 0 and abs(dr) == abs(dc), "same-color endpoints must define a strict diagonal"
    sr = _sign(dr)
    sc = _sign(dc)
    return [(r0 + k * sr, c0 + k * sc) for k in range(abs(dr) + 1)]


def _connect_pairs(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    colors = sorted(int(c) for c in np.unique(out) if int(c) != BACKGROUND)
    for color in colors:
        coords = np.argwhere(out == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        cells = _diag_segment(tuple(coords[0]), tuple(coords[1]))
        for r, c in cells:
            if int(out[r, c]) == BACKGROUND:
                out[r, c] = color
    return out


def _largest_background_region(filled: np.ndarray) -> np.ndarray:
    comps = _components_mask(filled == BACKGROUND)
    assert comps, "expected at least one background region"
    comps.sort(key=lambda coords: (-int(coords.shape[0]), _bbox_coords(coords)[0], _bbox_coords(coords)[1]))
    return comps[0]


def _crop_region(coords: np.ndarray, fill: int = OUTPUT_COLOR) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = int(fill)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored diagonal endpoint pair, then crop the largest background gap.

    Input structure:
    - The grid has a black background.
    - There are 4 or 5 foreground colors, and each foreground color appears exactly twice.
    - Each same-colored pair is aligned on a strict 45-degree diagonal, so the two pixels are the endpoints of one diagonal segment.

    Rule:
    - For every foreground color, fill the full diagonal segment between its two endpoints, preserving any already-painted non-black cell at crossings.
    - In the completed grid, find all 4-connected black background regions.
    - Select the region with the largest area; if multiple regions tie, choose the one whose bounding box starts furthest toward the upper-left (smaller row first, then smaller column).
    - Output the tight crop of that chosen region, recoloring the region to gray and leaving other cells in the crop black.

    Key insight:
    - The colored diagonals are only separators: the answer is the largest black space created after all same-color diagonal links are drawn.
    """
    filled = _connect_pairs(np.asarray(grid, dtype=int))
    region = _largest_background_region(filled)
    return _crop_region(region)


def _full_backslash(h: int, w: int, diff: int) -> list[Coord]:
    return [(r, r + diff) for r in range(h) if 0 <= r + diff < w]


def _full_slash(h: int, w: int, total: int) -> list[Coord]:
    return [(r, total - r) for r in range(h) if 0 <= total - r < w]


def _sample_spaced(values: list[int], count: int, min_gap: int) -> list[int] | None:
    choices = values[:]
    random.shuffle(choices)
    picked: list[int] = []
    for value in choices:
        if all(abs(value - prev) >= min_gap for prev in picked):
            picked.append(int(value))
            if len(picked) == count:
                return sorted(picked)
    return None


def _trim_segment(cells: list[Coord], force_internal: bool = False) -> list[Coord]:
    if len(cells) <= MIN_SEGMENT_CELLS:
        return cells
    max_trim = min(2, len(cells) - MIN_SEGMENT_CELLS)
    if max_trim <= 0:
        return cells

    left = random.randint(0, max_trim)
    right_limit = min(max_trim, len(cells) - MIN_SEGMENT_CELLS - left)
    right = random.randint(0, right_limit)

    if force_internal and left == 0 and right == 0:
        if random.random() < 0.5 and right_limit > 0:
            right = 1
        else:
            left = 1

    if left == 0 and right == 0:
        return cells
    return cells[left : len(cells) - right]


def _chebyshev_gap(a: list[Coord], b: list[Coord]) -> int:
    best = 10**9
    for r0, c0 in a:
        for r1, c1 in b:
            best = min(best, max(abs(r0 - r1), abs(c0 - c1)))
            if best == 0:
                return 0
    return best


def _build_input(h: int, w: int, segments: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for seg in segments:
        color = int(seg["color"])
        cells = seg["cells"]
        grid[cells[0]] = color
        grid[cells[-1]] = color
    return grid


def _expected_from_segments(h: int, w: int, segments: list[dict]) -> np.ndarray:
    filled = np.zeros((h, w), dtype=int)
    for seg in segments:
        for r, c in seg["cells"]:
            filled[r, c] = 1
    return filled


def _segments_are_clear(segments: list[dict]) -> bool:
    endpoints = set()
    cell_sets = [set(seg["cells"]) for seg in segments]
    for i, seg in enumerate(segments):
        seg_endpoints = {seg["cells"][0], seg["cells"][-1]}
        if endpoints & seg_endpoints:
            return False
        endpoints |= seg_endpoints
        for ep in seg_endpoints:
            for j, cells in enumerate(cell_sets):
                if i != j and ep in cells:
                    return False

    for i in range(len(segments)):
        inter_i = {segments[i]["cells"][0], segments[i]["cells"][-1]}
        for j in range(i + 1, len(segments)):
            overlap = cell_sets[i] & cell_sets[j]
            if len(overlap) > 1:
                return False
            if overlap and (overlap & inter_i or overlap & {segments[j]["cells"][0], segments[j]["cells"][-1]}):
                return False
    return True


def _has_cross_or_close_opposites(segments: list[dict]) -> bool:
    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1 :]:
            if seg_a["kind"] == seg_b["kind"]:
                continue
            if _chebyshev_gap(seg_a["cells"], seg_b["cells"]) <= 1:
                return True
    return False


def _check_semantics(grid: np.ndarray, segments: list[dict], expect_tie: bool) -> bool:
    h, w = grid.shape
    if not (grid.ndim == 2 and max(h, w) <= 30):
        return False

    colors = sorted(int(c) for c in np.unique(grid) if int(c) != BACKGROUND)
    if len(colors) != len(segments) or not (4 <= len(colors) <= 5):
        return False
    if any(int(np.count_nonzero(grid == color)) != 2 for color in colors):
        return False

    if not _segments_are_clear(segments) or not _has_cross_or_close_opposites(segments):
        return False

    filled = _connect_pairs(grid)
    expected_mask = _expected_from_segments(h, w, segments)
    if not np.array_equal((filled != BACKGROUND), expected_mask.astype(bool)):
        return False

    comps = _components_mask(expected_mask == 0)
    if len(comps) < 3:
        return False

    ranked = sorted(comps, key=lambda coords: (-int(coords.shape[0]), _bbox_coords(coords)[0], _bbox_coords(coords)[1]))
    if len(ranked) < 2:
        return False
    top_area = int(ranked[0].shape[0])
    second_area = int(ranked[1].shape[0])
    if expect_tie:
        if top_area != second_area:
            return False
    else:
        if top_area == second_area:
            return False

    r0, c0, r1, c1 = _bbox_coords(ranked[0])
    if (r0, c0, r1, c1) == (0, 0, h - 1, w - 1):
        return False
    if top_area < 6:
        return False

    expected = _crop_region(ranked[0])
    actual = main(grid)
    return np.array_equal(actual, expected)


def _gen_tie_segments() -> tuple[int, int, list[dict]] | None:
    size = random.choice([11, 13, 15])
    max_a = size // 2 - 2
    if max_a < 1:
        return None
    a = random.randint(1, max_a)
    b = random.randint(2, size // 2 - 1)
    specs = [
        ("back", _full_backslash(size, size, -a)),
        ("back", _full_backslash(size, size, a)),
        ("slash", _full_slash(size, size, size - 1 - b)),
        ("slash", _full_slash(size, size, size - 1 + b)),
    ]
    colors = random.sample(INPUT_COLORS, len(specs))
    segments = [{"kind": kind, "cells": cells, "color": color} for (kind, cells), color in zip(specs, colors)]
    return size, size, segments


def _gen_unique_segments() -> tuple[int, int, list[dict]] | None:
    h = random.randint(11, 15)
    w = random.randint(11, 15)
    n = random.choice([4, 5])
    back_count, slash_count = random.choice([(2, 2), (3, 2), (2, 3)] if n == 5 else [(2, 2)])

    back_values = list(range(-(h - 4), w - 3))
    slash_values = list(range(3, h + w - 4))
    back = _sample_spaced(back_values, back_count, 3)
    slash = _sample_spaced(slash_values, slash_count, 3)
    if back is None or slash is None:
        return None

    specs: list[tuple[str, list[Coord]]] = []
    for diff in back:
        specs.append(("back", _full_backslash(h, w, diff)))
    for total in slash:
        specs.append(("slash", _full_slash(h, w, total)))
    random.shuffle(specs)

    segments: list[dict] = []
    trimmed_any = False
    for idx, (kind, cells) in enumerate(specs):
        force_internal = idx == 0
        trimmed = _trim_segment(cells, force_internal=force_internal)
        trimmed_any |= trimmed != cells
        segments.append({"kind": kind, "cells": trimmed})

    if not trimmed_any:
        return None

    colors = random.sample(INPUT_COLORS, len(segments))
    for seg, color in zip(segments, colors):
        seg["color"] = color
    return h, w, segments


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        expect_tie = random.random() < 0.3
        generated = _gen_tie_segments() if expect_tie else _gen_unique_segments()
        if generated is None:
            continue
        h, w, segments = generated
        grid = _build_input(h, w, segments)
        if _check_semantics(grid, segments, expect_tie=expect_tie):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
