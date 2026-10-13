from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG_COLOR = 0
PALETTE_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _choice(seq):
    return seq[int(_RNG.integers(len(seq)))]


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate90(cells):
    max_r = max(r for r, _ in cells)
    return tuple((c, max_r - r) for r, c in cells)


def _flip_lr(cells):
    max_c = max(c for _, c in cells)
    return tuple((r, max_c - c) for r, c in cells)


def _variants(cells):
    cur = tuple(cells)
    seen = set()
    out = []
    for _ in range(4):
        for candidate in (cur, _flip_lr(cur)):
            norm = _normalize(candidate)
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
        cur = _rotate90(cur)
    return out


def _shape_perimeter(cells):
    cell_set = set(cells)
    perimeter = 0
    for r, c in cell_set:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) not in cell_set:
                perimeter += 1
    return perimeter


def _shape_dims(cells):
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _build_catalog():
    raw_templates = [
        ("line3", ((0, 0), (0, 1), (0, 2))),
        ("L3", ((0, 0), (1, 0), (1, 1))),
        ("square2", ((0, 0), (0, 1), (1, 0), (1, 1))),
        ("line4", ((0, 0), (0, 1), (0, 2), (0, 3))),
        ("L4", ((0, 0), (1, 0), (2, 0), (2, 1))),
        ("T4", ((0, 0), (0, 1), (0, 2), (1, 1))),
        ("Z4", ((0, 0), (0, 1), (1, 1), (1, 2))),
        ("rect2x3", ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))),
        ("line5", ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))),
        ("plus5", ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))),
        ("U5", ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2))),
        ("T5", ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1))),
        ("corner8", ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1))),
        ("rect2x4", ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3))),
        ("line6", ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5))),
        ("L6", ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3))),
        ("zig6", ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3))),
        ("T6", ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (3, 1))),
        ("line7", ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6))),
        ("snake7", ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3))),
    ]

    catalog = []
    by_name = {}
    by_perimeter = defaultdict(list)
    for name, base_cells in raw_templates:
        norm = _normalize(base_cells)
        item = {
            "name": name,
            "cells": norm,
            "area": len(norm),
            "perimeter": _shape_perimeter(norm),
            "variants": _variants(norm),
        }
        catalog.append(item)
        by_name[name] = item
        by_perimeter[item["perimeter"]].append(item)
    return catalog, by_name, dict(by_perimeter)


_TEMPLATES, _TEMPLATE_BY_NAME, _PERIMETER_GROUPS = _build_catalog()

_TIE_GROUPS = {
    8: (("line3", "L3"), ("square2",)),
    10: (("line4", "L4", "T4", "Z4"), ("rect2x3",)),
    12: (("line5", "plus5", "U5", "T5"), ("corner8", "rect2x4")),
}


def _background_color(grid):
    g = np.asarray(grid, dtype=int)
    counts = np.bincount(g.ravel(), minlength=max(10, int(g.max()) + 1))
    return int(np.argmax(counts))


def _extract_components(grid, bg, forbidden):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or forbidden[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            cells = []

            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or forbidden[nr, nc] or int(g[nr, nc]) == bg:
                        continue
                    assert int(g[nr, nc]) == color, "each object must be monochrome"
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            cell_set = set(cells)
            perimeter = 0
            rows = np.array([rr for rr, _ in cells], dtype=int)
            cols = np.array([cc for _, cc in cells], dtype=int)
            for rr, cc in cell_set:
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        perimeter += 1
                    elif forbidden[nr, nc] or int(g[nr, nc]) == bg:
                        perimeter += 1

            components.append(
                {
                    "rows": rows,
                    "cols": cols,
                    "area": len(cells),
                    "perimeter": perimeter,
                    "top": int(rows.min()),
                    "left": int(cols.min()),
                    "bottom": int(rows.max()),
                    "right": int(cols.max()),
                    "color": color,
                }
            )

    return components


def _extract_scene(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    assert int(g[0, 0]) != bg, "palette must start at the top-left corner"

    palette = []
    forbidden = np.zeros_like(g, dtype=bool)
    c = 0
    while c < g.shape[1] and int(g[0, c]) != bg:
        palette.append(int(g[0, c]))
        forbidden[0, c] = True
        c += 1

    assert palette, "palette must contain at least one color"
    assert bool(np.all(g[0, c:] == bg)), "top row after the palette must be background"
    if g.shape[0] > 1:
        assert bool(np.all(g[1] == bg)), "palette must be separated from shapes by a blank row"

    components = _extract_components(g, bg, forbidden)
    return bg, palette, forbidden, components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each disconnected shape by perimeter rank using the palette strip in the top-left corner.

    Input structure:
    - The background is black.
    - Row 0 starts with a contiguous left-to-right palette of single colored pixels.
    - Below a blank separator row, the main area contains several monochrome 4-connected shapes.

    Rule:
    - Exclude the top-left palette pixels from the shape set.
    - Compute each shape's 4-neighbor contour perimeter: count every side that touches background or the grid boundary.
    - Sort shapes by decreasing perimeter; break perimeter ties by smaller top row, then smaller left column.
    - Repaint the ranked shapes with the palette colors from left to right, and erase the palette itself back to background.

    Key insight:
    - The input object colors are distractions; the controlling signal is the palette row, and the assignment order is determined by perimeter rather than area or position.
    """
    g = np.asarray(grid, dtype=int)
    bg, palette, forbidden, components = _extract_scene(g)
    assert len(components) == len(palette), "palette length must match the number of shapes"

    ranked = sorted(components, key=lambda comp: (-comp["perimeter"], comp["top"], comp["left"]))
    out = g.copy()
    out[forbidden] = bg

    for color, comp in zip(palette, ranked):
        out[comp["rows"], comp["cols"]] = int(color)

    return out


def _can_place(reserved, r, c, h, w):
    rr0 = max(0, r - 1)
    rr1 = min(reserved.shape[0], r + h + 1)
    cc0 = max(0, c - 1)
    cc1 = min(reserved.shape[1], c + w + 1)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _mark_reserved(reserved, r, c, h, w):
    rr0 = max(0, r - 1)
    rr1 = min(reserved.shape[0], r + h + 1)
    cc0 = max(0, c - 1)
    cc1 = min(reserved.shape[1], c + w + 1)
    reserved[rr0:rr1, cc0:cc1] = True


def _touches_edge(r, c, h, w, grid_h, grid_w):
    return c == 0 or c + w == grid_w or r + h == grid_h


def _candidate_positions(reserved, shape, row=None):
    grid_h, grid_w = reserved.shape
    h, w = _shape_dims(shape)
    if row is None:
        row_iter = range(2, grid_h - h + 1)
    else:
        if not (2 <= row <= grid_h - h):
            return []
        row_iter = (row,)

    positions = []
    for r in row_iter:
        for c in range(0, grid_w - w + 1):
            if _can_place(reserved, r, c, h, w):
                positions.append((r, c))
    _RNG.shuffle(positions)
    return positions


def _place_at(grid, shape, color, top, left):
    for dr, dc in shape:
        grid[top + dr, left + dc] = int(color)


def _place_same_row_tie_pair(reserved, small_shape, large_shape):
    small_h, small_w = _shape_dims(small_shape)
    large_h, large_w = _shape_dims(large_shape)
    for small_r, small_c in _candidate_positions(reserved, small_shape):
        temp_reserved = reserved.copy()
        _mark_reserved(temp_reserved, small_r, small_c, small_h, small_w)
        large_candidates = [
            (r, c)
            for r, c in _candidate_positions(temp_reserved, large_shape, row=small_r)
            if c >= small_c + small_w + 2
        ]
        if large_candidates:
            large_r, large_c = _choice(large_candidates)
            return (small_r, small_c), (large_r, large_c)
    return None


def _place_higher_row_tie_pair(reserved, small_shape, large_shape):
    small_h, small_w = _shape_dims(small_shape)
    large_h, large_w = _shape_dims(large_shape)
    for small_r, small_c in _candidate_positions(reserved, small_shape):
        temp_reserved = reserved.copy()
        _mark_reserved(temp_reserved, small_r, small_c, small_h, small_w)
        large_candidates = [(r, c) for r, c in _candidate_positions(temp_reserved, large_shape) if r > small_r]
        if large_candidates:
            large_r, large_c = _choice(large_candidates)
            return (small_r, small_c), (large_r, large_c)
    return None


def _place_random_shape(reserved, shape, require_edge):
    h, w = _shape_dims(shape)
    positions = _candidate_positions(reserved, shape)
    if require_edge:
        positions = [pos for pos in positions if _touches_edge(pos[0], pos[1], h, w, *reserved.shape)]
    if not positions:
        return None
    top, left = _choice(positions)
    _mark_reserved(reserved, top, left, h, w)
    return top, left


def _select_templates(count, tie_mode):
    if tie_mode:
        tie_perimeter = int(_choice(sorted(_TIE_GROUPS)))
        small_names, large_names = _TIE_GROUPS[tie_perimeter]
        names = [_choice(small_names), _choice(large_names)]
        while len(names) < count:
            names.append(_choice(_TEMPLATES)["name"])
        return names

    names = ["rect2x3", _choice(("line5", "plus5", "U5", "T5"))]
    extra_perimeters = list(_RNG.choice(np.array([8, 14, 16], dtype=int), size=count - 2, replace=False))
    for perimeter in extra_perimeters:
        names.append(_choice(_PERIMETER_GROUPS[int(perimeter)])["name"])
    return names


def _rank_indices(components, key_name):
    if key_name == "perimeter":
        return sorted(range(len(components)), key=lambda i: (-components[i]["perimeter"], components[i]["top"], components[i]["left"]))
    if key_name == "area":
        return sorted(range(len(components)), key=lambda i: (-components[i]["area"], components[i]["top"], components[i]["left"]))
    raise ValueError(key_name)


def _validate_generated(grid, expected_count, tie_mode, expect_edge):
    try:
        bg, palette, _forbidden, components = _extract_scene(grid)
        if bg != BG_COLOR or len(palette) != expected_count or len(components) != expected_count:
            return False

        perimeters = [comp["perimeter"] for comp in components]
        if tie_mode and len(set(perimeters)) == len(perimeters):
            return False
        if (not tie_mode) and len(set(perimeters)) != len(perimeters):
            return False

        if expect_edge:
            h, w = grid.shape
            if not any(comp["left"] == 0 or comp["right"] == w - 1 or comp["bottom"] == h - 1 for comp in components):
                return False

        if _rank_indices(components, "perimeter") == _rank_indices(components, "area"):
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False
        if not bool(np.all(out[0, : len(palette)] == BG_COLOR)):
            return False

        ranked = sorted(components, key=lambda comp: (-comp["perimeter"], comp["top"], comp["left"]))
        recolored = False
        for assigned_color, comp in zip(palette, ranked):
            if comp["color"] != assigned_color:
                recolored = True
                break
        return recolored
    except AssertionError:
        return False


def _gen_attempt():
    count = int(_RNG.integers(3, 6))
    tie_mode = bool(_RNG.integers(2))
    same_row_tie = tie_mode and bool(_RNG.integers(2))
    expect_edge = bool(_RNG.integers(2))

    template_names = _select_templates(count, tie_mode)
    shapes = [_choice(_TEMPLATE_BY_NAME[name]["variants"]) for name in template_names]
    shape_dims = [_shape_dims(shape) for shape in shapes]
    max_h = max(h for h, _ in shape_dims)
    max_w = max(w for _, w in shape_dims)

    height_low = max(12, 5 + max_h + max(0, count - 3))
    width_low = max(16, 7 + max_w + 2 * max(0, count - 3))
    if same_row_tie:
        tie_w = shape_dims[0][1] + shape_dims[1][1] + 4
        width_low = max(width_low, tie_w)

    if height_low > 30 or width_low > 30:
        return None

    grid_h = int(_RNG.integers(height_low, min(30, height_low + 8) + 1))
    grid_w = int(_RNG.integers(width_low, 31))
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    palette = _RNG.choice(np.array(PALETTE_COLORS, dtype=int), size=count, replace=False)
    grid[0, :count] = palette

    placements = [None] * count
    edge_used = False

    if tie_mode:
        pair = _place_same_row_tie_pair(reserved, shapes[0], shapes[1]) if same_row_tie else _place_higher_row_tie_pair(reserved, shapes[0], shapes[1])
        if pair is None:
            return None

        for idx, (top, left) in enumerate(pair):
            h, w = shape_dims[idx]
            placements[idx] = (top, left)
            _mark_reserved(reserved, top, left, h, w)
            edge_used = edge_used or _touches_edge(top, left, h, w, grid_h, grid_w)

    remaining = [idx for idx in range(count) if placements[idx] is None]

    if expect_edge and not edge_used:
        edge_candidates = []
        for idx in remaining:
            h, w = shape_dims[idx]
            positions = [pos for pos in _candidate_positions(reserved, shapes[idx]) if _touches_edge(pos[0], pos[1], h, w, grid_h, grid_w)]
            if positions:
                edge_candidates.append((idx, positions))
        if not edge_candidates:
            return None
        edge_idx, positions = _choice(edge_candidates)
        top, left = _choice(positions)
        h, w = shape_dims[edge_idx]
        placements[edge_idx] = (top, left)
        _mark_reserved(reserved, top, left, h, w)
        edge_used = True
        remaining.remove(edge_idx)

    _RNG.shuffle(remaining)
    for idx in remaining:
        pos = _place_random_shape(reserved, shapes[idx], require_edge=False)
        if pos is None:
            return None
        placements[idx] = pos

    original_colors = _RNG.choice(np.array(PALETTE_COLORS, dtype=int), size=count, replace=True)
    for idx, ((top, left), shape, color) in enumerate(zip(placements, shapes, original_colors)):
        assert top is not None and left is not None
        _place_at(grid, shape, color, top, left)

    if not _validate_generated(grid, expected_count=count, tie_mode=tie_mode, expect_edge=expect_edge):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
