from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
CROSS = 5
RULER = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400
QUADRANT_MIN = 8
QUADRANT_MAX = 12


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _shape_bbox(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    arr = _normalize(cells)
    for _ in range(random.randint(0, 3)):
        arr = {(c, -r) for r, c in arr}
        arr = _normalize(arr)
    if random.random() < 0.5:
        max_c = max(c for _r, c in arr)
        arr = {(r, max_c - c) for r, c in arr}
        arr = _normalize(arr)
    if random.random() < 0.5:
        max_r = max(r for r, _c in arr)
        arr = {(max_r - r, c) for r, c in arr}
        arr = _normalize(arr)
    return arr


def _sample_shape(max_h: int, max_w: int) -> set[tuple[int, int]] | None:
    area_cap = min(8, max_h * max_w)
    if max_h < 2 or max_w < 2 or area_cap < 3:
        return None

    for _ in range(120):
        box_h = random.randint(2, min(4, max_h))
        box_w = random.randint(2, min(4, max_w))
        target = random.randint(3, min(8, box_h * box_w))
        seed = (random.randrange(box_h), random.randrange(box_w))
        cells = {seed}

        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target:
            continue

        cells = _transform_shape(cells)
        shape_h, shape_w = _shape_bbox(cells)
        if shape_h <= max_h and shape_w <= max_w:
            return cells
    return None


def _find_cross(g: np.ndarray) -> tuple[int, int, int]:
    h, w = g.shape
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(g):
        rows = np.where(np.all(g == int(color), axis=1))[0]
        cols = np.where(np.all(g == int(color), axis=0))[0]
        if rows.size != 1 or cols.size != 1:
            continue
        expected = w + h - 1
        if int(np.count_nonzero(g == int(color))) != expected:
            continue
        candidates.append((int(color), int(rows[0]), int(cols[0])))
    assert len(candidates) == 1, f"expected one cross color, got {candidates}"
    return candidates[0]


def _quadrants(cross_r: int, cross_c: int, h: int, w: int):
    return (
        ("top", "left", slice(0, cross_r), slice(0, cross_c)),
        ("top", "right", slice(0, cross_r), slice(cross_c + 1, w)),
        ("bottom", "left", slice(cross_r + 1, h), slice(0, cross_c)),
        ("bottom", "right", slice(cross_r + 1, h), slice(cross_c + 1, w)),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each quadrant's colored object away from the central cross by the ruler-bar length and axis.

    Input structure:
    - Black background fills the grid.
    - One grey full row and one grey full column form a central cross, splitting the grid into four rectangular quadrants.
    - In each quadrant, exactly one maroon ruler bar lies on the edge adjacent to the cross, either horizontal or vertical.
    - Each quadrant also contains one small connected monochrome object away from the edges.

    Rule:
    - For each quadrant, count the maroon ruler cells to get N.
    - If the ruler is horizontal, move the object horizontally away from the cross by N cells.
    - If the ruler is vertical, move the object vertically away from the cross by N cells.
    - Clear the object's original cells, and keep the grey cross plus all ruler bars unchanged.

    Key insight:
    - The ruler encodes two independent signals at once: its length gives the translation distance, and its orientation chooses whether the motion is horizontal or vertical.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(np.bincount(g.ravel()).argmax())
    _cross_color, cross_r, cross_c = _find_cross(g)
    out = g.copy()

    for vertical_side, horizontal_side, rs, cs in _quadrants(cross_r, cross_c, h, w):
        region = g[rs, cs]
        assert region.size > 0, "quadrants must be non-empty"
        qh, qw = region.shape

        edge_r = qh - 1 if vertical_side == "top" else 0
        edge_c = qw - 1 if horizontal_side == "left" else 0

        row_mask = region[edge_r, :] != bg
        col_mask = region[:, edge_c] != bg
        row_mask[edge_c] = False
        col_mask[edge_r] = False

        if bool(row_mask.any()):
            assert not bool(col_mask.any()), "ruler orientation must be unique"
            move_dr = 0
            move_dc = -int(row_mask.sum()) if horizontal_side == "left" else int(row_mask.sum())
            ruler_mask = np.zeros_like(region, dtype=bool)
            ruler_mask[edge_r, row_mask] = True
        else:
            assert bool(col_mask.any()), "expected one ruler per quadrant"
            move_dr = -int(col_mask.sum()) if vertical_side == "top" else int(col_mask.sum())
            move_dc = 0
            ruler_mask = np.zeros_like(region, dtype=bool)
            ruler_mask[col_mask, edge_c] = True

        obj_mask = (region != bg) & ~ruler_mask
        comps = _components(obj_mask)
        assert len(comps) == 1, f"expected one object per quadrant, got {len(comps)}"
        comp = comps[0]
        obj_color = int(region[comp[0][0], comp[0][1]])
        assert all(int(region[r, c]) == obj_color for r, c in comp), "object must be monochrome"

        for r, c in comp:
            out[rs.start + r, cs.start + c] = bg
        for r, c in comp:
            rr = rs.start + r + move_dr
            cc = cs.start + c + move_dc
            assert 0 <= rr < h and 0 <= cc < w, "translated object must stay in bounds"
            out[rr, cc] = obj_color

    return out


def _pick_bar_start(length: int, extent: int, cross_on_high_end: bool) -> int:
    if cross_on_high_end:
        starts = list(range(0, extent - length))
    else:
        starts = list(range(1, extent - length + 1))
    if not starts:
        raise ValueError("no valid bar placement")
    return random.choice(starts)


def _candidate_positions(qh: int, qw: int, shape_h: int, shape_w: int, dr: int, dc: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    center_r = (qh - 1) / 2.0
    center_c = (qw - 1) / 2.0

    for r0 in range(1, qh - shape_h):
        for c0 in range(1, qw - shape_w):
            if r0 + dr < 0 or r0 + shape_h - 1 + dr >= qh:
                continue
            if c0 + dc < 0 or c0 + shape_w - 1 + dc >= qw:
                continue
            positions.append((r0, c0))

    positions.sort(
        key=lambda rc: abs((rc[0] + (shape_h - 1) / 2.0) - center_r)
        + abs((rc[1] + (shape_w - 1) / 2.0) - center_c)
    )
    return positions


def _paint_shape(region: np.ndarray, cells: set[tuple[int, int]], top: int, left: int, color: int) -> list[tuple[int, int]]:
    placed: list[tuple[int, int]] = []
    for r, c in cells:
        rr = top + r
        cc = left + c
        region[rr, cc] = color
        placed.append((rr, cc))
    return placed


def _gen_quadrant(
    qh: int,
    qw: int,
    vertical_side: str,
    horizontal_side: str,
    color: int,
) -> tuple[np.ndarray, dict] | None:
    region = np.full((qh, qw), BG, dtype=int)
    orientation = random.choice(("h", "v"))

    shape = _sample_shape(max_h=min(4, qh - 2), max_w=min(4, qw - 2))
    if shape is None:
        return None
    shape_h, shape_w = _shape_bbox(shape)

    if orientation == "h":
        max_n = min(5, qw - shape_w - 1, qw - 1)
        if max_n < 1:
            return None
        n = random.randint(1, max_n)
        edge_r = qh - 1 if vertical_side == "top" else 0
        bar_start = _pick_bar_start(n, qw, cross_on_high_end=(horizontal_side == "left"))
        region[edge_r, bar_start : bar_start + n] = RULER
        dr = 0
        dc = -n if horizontal_side == "left" else n
    else:
        max_n = min(5, qh - shape_h - 1, qh - 1)
        if max_n < 1:
            return None
        n = random.randint(1, max_n)
        edge_c = qw - 1 if horizontal_side == "left" else 0
        bar_start = _pick_bar_start(n, qh, cross_on_high_end=(vertical_side == "top"))
        region[bar_start : bar_start + n, edge_c] = RULER
        dr = -n if vertical_side == "top" else n
        dc = 0

    positions = _candidate_positions(qh, qw, shape_h, shape_w, dr, dc)
    if not positions:
        return None
    best_pool = positions[: min(len(positions), 6)]
    top, left = random.choice(best_pool)
    obj_cells = _paint_shape(region, shape, top, left, color)

    return region, {
        "shape": shape,
        "top": top,
        "left": left,
        "dr": dr,
        "dc": dc,
        "color": color,
        "obj_cells": obj_cells,
    }


def _build_expected(inp: np.ndarray, meta: dict, cross_r: int, cross_c: int) -> np.ndarray:
    out = inp.copy()
    h, w = inp.shape
    for quad_name, (_, _, rs, cs) in zip(meta["order"], _quadrants(cross_r, cross_c, h, w), strict=False):
        info = meta["quads"][quad_name]
        for r, c in info["obj_cells"]:
            out[rs.start + r, cs.start + c] = BG
        for r, c in info["obj_cells"]:
            out[rs.start + r + info["dr"], cs.start + c + info["dc"]] = info["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(QUADRANT_MIN, QUADRANT_MAX)
    bottom_h = random.randint(QUADRANT_MIN, QUADRANT_MAX)
    left_w = random.randint(QUADRANT_MIN, QUADRANT_MAX)
    right_w = random.randint(QUADRANT_MIN, QUADRANT_MAX)
    h = top_h + 1 + bottom_h
    w = left_w + 1 + right_w

    grid = np.full((h, w), BG, dtype=int)
    cross_r = top_h
    cross_c = left_w
    grid[cross_r, :] = CROSS
    grid[:, cross_c] = CROSS

    quad_colors = random.sample(OBJECT_COLORS, 4)
    meta = {"quads": {}, "order": ("tl", "tr", "bl", "br")}

    specs = (
        ("tl", "top", "left", slice(0, cross_r), slice(0, cross_c), quad_colors[0]),
        ("tr", "top", "right", slice(0, cross_r), slice(cross_c + 1, w), quad_colors[1]),
        ("bl", "bottom", "left", slice(cross_r + 1, h), slice(0, cross_c), quad_colors[2]),
        ("br", "bottom", "right", slice(cross_r + 1, h), slice(cross_c + 1, w), quad_colors[3]),
    )

    for name, vertical_side, horizontal_side, rs, cs, color in specs:
        region, info = _gen_quadrant(rs.stop - rs.start, cs.stop - cs.start, vertical_side, horizontal_side, color) or (None, None)
        if region is None or info is None:
            return None
        grid[rs, cs] = region
        meta["quads"][name] = info

    expected = _build_expected(grid, meta, cross_r, cross_c)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
