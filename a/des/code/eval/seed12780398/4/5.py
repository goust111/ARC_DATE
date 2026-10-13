import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = grid != BG
    candidates = []
    for comp in _components(mask):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if c0 != c1:
            continue
        if len(comp) != r1 - r0 + 1:
            continue
        if sorted(rows) != list(range(r0, r1 + 1)):
            continue
        candidates.append((c1, r1, -len(comp), sorted(comp)))
    if not candidates:
        raise AssertionError("legend not found")
    candidates.sort()
    return candidates[-1][3]


def _choose_start(comp: list[tuple[int, int]]) -> tuple[int, int]:
    comp_set = set(comp)
    h = max(r for r, _ in comp) + 1
    w = max(c for _, c in comp) + 1
    endpoints = []
    for r, c in comp:
        deg = sum((nr, nc) in comp_set for nr, nc in _neighbors4(r, c, h, w))
        if deg <= 1:
            endpoints.append((r, c))
    pool = endpoints if endpoints else comp
    return min(pool, key=lambda p: (p[1], p[0]))


def _extract_segments(grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]], list[dict[str, object]]]:
    legend_cells = _find_legend_cells(grid)
    legend_colors = [int(grid[r, c]) for r, c in sorted(legend_cells)]

    mask = grid != BG
    for r, c in legend_cells:
        mask[r, c] = False

    h, w = grid.shape
    intersections: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or int(grid[r, c]) != YELLOW:
                continue
            deg = sum(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w))
            if deg in (3, 4):
                intersections.add((r, c))

    segment_mask = mask.copy()
    for r, c in intersections:
        segment_mask[r, c] = False

    segments = []
    for comp in _components(segment_mask):
        start = _choose_start(comp)
        segments.append(
            {
                "cells": comp,
                "length": len(comp),
                "start": start,
            }
        )
    segments.sort(key=lambda s: (int(s["length"]), int(s["start"][1]), int(s["start"][0])))
    return legend_colors, intersections, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a path network at yellow intersections and recolor the pieces by length.
    """
    g = np.asarray(grid, dtype=int)
    legend_colors, intersections, segments = _extract_segments(g)

    out = g.copy()
    for idx, seg in enumerate(segments):
        color = legend_colors[idx % len(legend_colors)]
        for r, c in seg["cells"]:
            out[r, c] = color
    for r, c in intersections:
        out[r, c] = YELLOW
    return out


def _sample_lengths(keys: list[str]) -> dict[str, int]:
    lengths = {key: random.randint(1, 5) for key in keys}
    if len(keys) >= 2 and random.random() < 0.5:
        common = random.randint(1, 4)
        dup_count = random.randint(2, min(4, len(keys)))
        for key in random.sample(keys, dup_count):
            lengths[key] = common
    return lengths


def _build_vertical_geometry(h: int, w: int):
    m = random.choice([1, 2, 3])
    branch_types = [random.choices(["L", "R", "B"], weights=[1, 1, 2])[0] for _ in range(m)]
    arm_count = sum(1 for t in branch_types if t in ("L", "B")) + sum(1 for t in branch_types if t in ("R", "B"))
    seg_count = (m + 1) + arm_count
    if not (4 <= seg_count <= len(PALETTE)):
        return None

    keys = ["top", "bottom"] + [f"gap{i}" for i in range(m - 1)]
    for i, t in enumerate(branch_types):
        if t in ("L", "B"):
            keys.append(f"L{i}")
        if t in ("R", "B"):
            keys.append(f"R{i}")
    lengths = _sample_lengths(keys)

    total_height = lengths["top"] + lengths["bottom"] + sum(lengths[f"gap{i}"] + 1 for i in range(m - 1))
    left_lens = [lengths.get(f"L{i}", 0) for i in range(m)]
    right_lens = [lengths.get(f"R{i}", 0) for i in range(m)]
    max_left = max(left_lens, default=0)
    max_right = max(right_lens, default=0)
    max_net_col = w - 4
    if total_height > h - 2 or max_net_col < 2:
        return None

    c_min = 1 + max_left
    c_max = max_net_col - max_right
    if c_min > c_max:
        return None

    top_endpoint = random.randint(1, h - total_height - 1)
    rows = [top_endpoint + lengths["top"]]
    for i in range(m - 1):
        rows.append(rows[-1] + lengths[f"gap{i}"] + 1)
    bottom_endpoint = rows[-1] + lengths["bottom"]
    trunk_col = random.randint(c_min, c_max)

    mask = np.zeros((h, w), dtype=bool)
    mask[top_endpoint : bottom_endpoint + 1, trunk_col] = True
    intersections = {(r, trunk_col) for r in rows}
    for i, r in enumerate(rows):
        if left_lens[i] > 0:
            mask[r, trunk_col - left_lens[i] : trunk_col + 1] = True
        if right_lens[i] > 0:
            mask[r, trunk_col : trunk_col + right_lens[i] + 1] = True
    return mask, intersections, seg_count


def _build_horizontal_geometry(h: int, w: int):
    m = random.choice([1, 2, 3])
    branch_types = [random.choices(["U", "D", "B"], weights=[1, 1, 2])[0] for _ in range(m)]
    arm_count = sum(1 for t in branch_types if t in ("U", "B")) + sum(1 for t in branch_types if t in ("D", "B"))
    seg_count = (m + 1) + arm_count
    if not (4 <= seg_count <= len(PALETTE)):
        return None

    keys = ["left", "right"] + [f"gap{i}" for i in range(m - 1)]
    for i, t in enumerate(branch_types):
        if t in ("U", "B"):
            keys.append(f"U{i}")
        if t in ("D", "B"):
            keys.append(f"D{i}")
    lengths = _sample_lengths(keys)

    total_width = lengths["left"] + lengths["right"] + sum(lengths[f"gap{i}"] + 1 for i in range(m - 1))
    up_lens = [lengths.get(f"U{i}", 0) for i in range(m)]
    down_lens = [lengths.get(f"D{i}", 0) for i in range(m)]
    max_up = max(up_lens, default=0)
    max_down = max(down_lens, default=0)
    max_net_col = w - 4
    if total_width > max_net_col - 1 or h - 2 < 1:
        return None

    r_min = 1 + max_up
    r_max = h - 2 - max_down
    if r_min > r_max:
        return None

    left_endpoint = random.randint(1, max_net_col - total_width)
    cols = [left_endpoint + lengths["left"]]
    for i in range(m - 1):
        cols.append(cols[-1] + lengths[f"gap{i}"] + 1)
    right_endpoint = cols[-1] + lengths["right"]
    trunk_row = random.randint(r_min, r_max)

    mask = np.zeros((h, w), dtype=bool)
    mask[trunk_row, left_endpoint : right_endpoint + 1] = True
    intersections = {(trunk_row, c) for c in cols}
    for i, c in enumerate(cols):
        if up_lens[i] > 0:
            mask[trunk_row - up_lens[i] : trunk_row + 1, c] = True
        if down_lens[i] > 0:
            mask[trunk_row : trunk_row + down_lens[i] + 1, c] = True
    return mask, intersections, seg_count


def _degrees_ok(mask: np.ndarray, intersections: set[tuple[int, int]]) -> bool:
    h, w = mask.shape
    for r, c in intersections:
        deg = sum(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w))
        if deg not in (3, 4):
            return False
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or (r, c) in intersections:
                continue
            deg = sum(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w))
            if deg > 2:
                return False
    return True


def _paint_input(mask: np.ndarray, intersections: set[tuple[int, int]], legend_colors: list[int]) -> np.ndarray:
    h, w = mask.shape
    grid = np.zeros((h, w), dtype=int)
    segment_mask = mask.copy()
    for r, c in intersections:
        segment_mask[r, c] = False
    segs = _components(segment_mask)
    seg_colors = random.sample(PALETTE, len(segs))
    random.shuffle(seg_colors)
    for color, comp in zip(seg_colors, segs):
        for r, c in comp:
            grid[r, c] = color
    for r, c in intersections:
        grid[r, c] = YELLOW
    legend_col = w - 1
    legend_start = h - len(legend_colors)
    for i, color in enumerate(legend_colors):
        grid[legend_start + i, legend_col] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 20)
    w = random.randint(14, 24)
    builder = random.choice([_build_vertical_geometry, _build_horizontal_geometry])
    built = builder(h, w)
    if built is None:
        return None
    mask, intersections, seg_count = built
    if not _degrees_ok(mask, intersections):
        return None

    branch_cycle = seg_count > 2 and random.random() < 0.5
    if branch_cycle:
        legend_len = random.randint(2, seg_count - 1)
    else:
        legend_len = random.randint(min(seg_count, len(PALETTE)), min(len(PALETTE), seg_count + 2))
    legend_colors = random.sample(PALETTE, legend_len)

    grid = _paint_input(mask, intersections, legend_colors)
    legend_found, intersections_found, segments_found = _extract_segments(grid)

    if legend_found != legend_colors:
        return None
    if intersections_found != intersections:
        return None
    if len(segments_found) != seg_count:
        return None
    if len({tuple(seg["cells"]) for seg in segments_found}) != seg_count:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
