from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _find_divider(g: np.ndarray) -> tuple[str, int]:
    h, w = g.shape
    row_hits = [r for r in range(h) if np.all(g[r, :] == DIVIDER)]
    col_hits = [c for c in range(w) if np.all(g[:, c] == DIVIDER)]
    assert len(row_hits) + len(col_hits) == 1, "expected exactly one full grey divider"
    if row_hits:
        return "h", int(row_hits[0])
    return "v", int(col_hits[0])


def _components_by_color(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color in {BG, DIVIDER}:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = sorted({rr for rr, _ in coords})
            cols = sorted({cc for _, cc in coords})
            straight = len(rows) == 1 or len(cols) == 1
            assert straight, "every colored object must be a straight 1-pixel line segment"
            if len(rows) == 1:
                assert len(cols) == len(coords) and cols[-1] - cols[0] + 1 == len(coords)
                direction = "h"
            else:
                assert len(rows) == len(coords) and rows[-1] - rows[0] + 1 == len(coords)
                direction = "v"
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "length": len(coords),
                    "dir": direction,
                    "r0": min(rows),
                    "r1": max(rows),
                    "c0": min(cols),
                    "c1": max(cols),
                }
            )
    return comps


def _classify_segments(g: np.ndarray) -> tuple[str, int, list[dict], list[dict]]:
    axis, divider_idx = _find_divider(g)
    side_a: list[dict] = []
    side_b: list[dict] = []
    for comp in _components_by_color(g):
        if axis == "h":
            assert comp["r1"] < divider_idx or comp["r0"] > divider_idx, "segment may not cross divider"
            (side_a if comp["r1"] < divider_idx else side_b).append(comp)
        else:
            assert comp["c1"] < divider_idx or comp["c0"] > divider_idx, "segment may not cross divider"
            (side_a if comp["c1"] < divider_idx else side_b).append(comp)
    return axis, divider_idx, side_a, side_b


def _ring_cells(center_r: int, center_c: int, radius: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    top = center_r - radius
    bottom = center_r + radius
    left = center_c - radius
    right = center_c + radius
    assert 0 <= top <= bottom < h
    assert 0 <= left <= right < w
    cells = set()
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _draw_ring(out: np.ndarray, center_r: int, center_c: int, radius: int, color: int) -> None:
    top = center_r - radius
    bottom = center_r + radius
    left = center_c - radius
    right = center_c + radius
    out[top, left : right + 1] = color
    out[bottom, left : right + 1] = color
    out[top : bottom + 1, left] = color
    out[top : bottom + 1, right] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Pair matching strips across the grey divider, then redraw pairs as concentric rectangular rings."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    axis, divider_idx, side_a, side_b = _classify_segments(g)
    h, w = g.shape

    by_color_a: dict[int, list[dict]] = defaultdict(list)
    by_color_b: dict[int, list[dict]] = defaultdict(list)
    for comp in side_a:
        by_color_a[int(comp["color"])].append(comp)
    for comp in side_b:
        by_color_b[int(comp["color"])].append(comp)

    pairs: list[tuple[int, int, dict, dict]] = []
    for color in sorted(set(by_color_a) & set(by_color_b)):
        assert len(by_color_a[color]) == 1 and len(by_color_b[color]) == 1, "paired colors must be unique on each side"
        comp_a = by_color_a[color][0]
        comp_b = by_color_b[color][0]
        pairs.append((max(int(comp_a["length"]), int(comp_b["length"])), color, comp_a, comp_b))

    center_r = divider_idx if axis == "h" else h // 2
    center_c = w // 2 if axis == "h" else divider_idx
    radius_count = len(pairs)
    if radius_count:
        assert center_r - radius_count >= 0 and center_r + radius_count < h
        assert center_c - radius_count >= 0 and center_c + radius_count < w

    out = g.copy()
    for _param, _color, comp_a, comp_b in pairs:
        for r, c in comp_a["coords"] + comp_b["coords"]:
            out[r, c] = BG

    pairs.sort(key=lambda item: (-item[0], item[1]))
    for idx, (_param, color, _comp_a, _comp_b) in enumerate(pairs):
        radius = radius_count - idx
        _draw_ring(out, center_r, center_c, radius, int(color))

    return out


def _segment_cells(orientation: str, fixed: int, start: int, length: int) -> set[tuple[int, int]]:
    if orientation == "h":
        return {(fixed, c) for c in range(start, start + length)}
    return {(r, fixed) for r in range(start, start + length)}


def _place_segment(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    orientation: str,
    fixed: int,
    length: int,
    color: int,
) -> bool:
    h, w = grid.shape
    if orientation == "h":
        starts = list(range(0, w - length + 1))
    else:
        starts = list(range(0, h - length + 1))
    random.shuffle(starts)
    for start in starts:
        cells = _segment_cells(orientation, fixed, start, length)
        if cells & occupied:
            continue
        for r, c in cells:
            grid[r, c] = color
        occupied |= cells
        return True
    return False


def _choose_lengths(param: int) -> tuple[int, int]:
    mode = random.choice(("equal", "a_shorter", "b_shorter"))
    if param == 2:
        mode = "equal"
    if mode == "equal":
        return param, param
    short_len = random.randint(1, param - 1)
    if mode == "a_shorter":
        return short_len, param
    return param, short_len


def _ring_forbidden(shape: tuple[int, int], axis: str, divider_idx: int, pair_count: int) -> set[tuple[int, int]]:
    h, w = shape
    center_r = divider_idx if axis == "h" else h // 2
    center_c = w // 2 if axis == "h" else divider_idx
    cells: set[tuple[int, int]] = set()
    for radius in range(1, pair_count + 1):
        cells |= _ring_cells(center_r, center_c, radius, shape)
    return cells


def _sample_attempt() -> np.ndarray | None:
    axis = random.choice(("h", "v"))
    pair_count = random.randint(1, 4)
    distractor_count = random.randint(0, min(3, len(PALETTE) - pair_count))
    distractor_sides = [random.choice((0, 1)) for _ in range(distractor_count)]
    side_a_count = pair_count + distractor_sides.count(0)
    side_b_count = pair_count + distractor_sides.count(1)

    ring_margin = pair_count + 2
    half_along_divider = ring_margin + max(side_a_count, side_b_count) + random.randint(0, 2)
    along_divider_size = 2 * half_along_divider + 1
    other_size_min = max(15, 2 * (pair_count + 3) + 1)
    other_size = random.choice([n for n in range(other_size_min, 30, 2)])

    if axis == "h":
        h, w = along_divider_size, other_size
        divider_idx = h // 2
        free_side_a = list(range(0, divider_idx - ring_margin))
        free_side_b = list(range(divider_idx + ring_margin + 1, h))
        seg_orientation = "h"
        max_param = min(w - 1, 8)
    else:
        h, w = other_size, along_divider_size
        divider_idx = w // 2
        free_side_a = list(range(0, divider_idx - ring_margin))
        free_side_b = list(range(divider_idx + ring_margin + 1, w))
        seg_orientation = "v"
        max_param = min(h - 1, 8)

    if len(free_side_a) < side_a_count or len(free_side_b) < side_b_count or max_param < pair_count + 1:
        return None

    pair_params = random.sample(list(range(2, max_param + 1)), pair_count)
    colors = random.sample(PALETTE, pair_count + distractor_count)
    pair_colors = colors[:pair_count]
    distractor_colors = colors[pair_count:]

    grid = np.full((h, w), BG, dtype=int)
    if axis == "h":
        grid[divider_idx, :] = DIVIDER
    else:
        grid[:, divider_idx] = DIVIDER

    occupied = _ring_forbidden((h, w), axis, divider_idx, pair_count)
    if axis == "h":
        occupied |= {(divider_idx, c) for c in range(w)}
    else:
        occupied |= {(r, divider_idx) for r in range(h)}

    fixed_a = random.sample(free_side_a, side_a_count)
    fixed_b = random.sample(free_side_b, side_b_count)
    random.shuffle(fixed_a)
    random.shuffle(fixed_b)

    for color, param in zip(pair_colors, pair_params):
        len_a, len_b = _choose_lengths(param)
        fa = fixed_a.pop()
        fb = fixed_b.pop()
        if not _place_segment(grid, occupied, seg_orientation, fa, len_a, color):
            return None
        if not _place_segment(grid, occupied, seg_orientation, fb, len_b, color):
            return None

    for color, side in zip(distractor_colors, distractor_sides):
        length = random.randint(1, max_param)
        fixed = fixed_a.pop() if side == 0 else fixed_b.pop()
        if not _place_segment(grid, occupied, seg_orientation, fixed, length, color):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if isinstance(out, np.ndarray) and out.shape == grid.shape:
            return grid
    raise RuntimeError("failed to generate valid input")
