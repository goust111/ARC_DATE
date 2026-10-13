from __future__ import annotations

import random
from collections import Counter, deque
from itertools import combinations

import numpy as np


GREY = 5
SHAPE_LIBRARY = [
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
]
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _normalize_shape(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0 = min(rs)
    c0 = min(cs)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _rotate_shape(shape: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    pts = list(shape)
    for _ in range(turns % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize_shape(pts)


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    variants = {_rotate_shape(shape, k) for k in range(4)}
    return tuple(sorted(variants))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _translate_shape(shape: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top + r), int(left + c)) for r, c in shape]


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _infer_skeleton_color(grid: np.ndarray, bg: int, holes: list[list[tuple[int, int]]]) -> int:
    cover = Counter()
    for hole in holes:
        seen_here = set()
        for r, c in hole:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    color = int(grid[nr, nc])
                    if color != bg:
                        seen_here.add(color)
        for color in seen_here:
            cover[color] += 1
    assert cover, "expected colors adjacent to holes"
    candidates = [color for color, cnt in cover.items() if cnt == len(holes)]
    pool = candidates if candidates else list(cover)
    return max(pool, key=lambda color: (int(np.count_nonzero(grid == color)), int(cover[color])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into matching holes in the grey skeleton and recolor them from nearby color sources.

    Input structure:
    - The background is the most common color.
    - A large central grey skeleton contains several enclosed background holes of
      area 2-5, each with a distinct small shape.
    - Around the outer empty area, there are small monochrome fragments whose
      shapes match the holes up to 90-degree rotation.
    - Some holes also touch a tiny non-background, non-grey source marker already
      embedded on the skeleton boundary.

    Rule:
    - Find each enclosed hole inside the skeleton.
    - Match every scattered fragment to the unique hole with the same shape under
      rotation, erase the fragment from its original location, and fill the hole.
    - If a hole is adjacent to a source marker, recolor the inserted fragment to
      that source color; otherwise keep the fragment's original color.
    - Preserve the grey skeleton and the source markers themselves.

    Key insight:
    Shape decides where each fragment goes, but the final color can be overridden
    by the local context around the destination hole rather than by the fragment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    holes: list[list[tuple[int, int]]] = []
    for comp in _components(g == bg):
        if 2 <= len(comp) <= 5 and not _touches_border(comp, g.shape):
            neighbor_colors = {
                int(g[nr, nc])
                for r, c in comp
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) != bg
            }
            if neighbor_colors:
                holes.append(comp)
    assert holes, "expected at least one enclosed hole"

    skeleton = _infer_skeleton_color(g, bg, holes)
    hole_cells = {cell for hole in holes for cell in hole}

    fragments = []
    for comp in _components((g != bg) & (g != skeleton)):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "expected monochrome fragments and source markers"
        is_source = any(
            (nr, nc) in hole_cells
            for r, c in comp
            for nr, nc in _neighbors4(r, c)
        )
        if is_source:
            continue
        fragments.append(
            {
                "cells": comp,
                "color": int(next(iter(colors))),
                "variants": set(_shape_variants(_normalize_shape(comp))),
            }
        )

    assert len(fragments) == len(holes), "fragment count must match hole count"

    out = g.copy()
    for frag in fragments:
        for r, c in frag["cells"]:
            out[r, c] = bg

    used: set[int] = set()
    for hole in holes:
        hole_shape = _normalize_shape(hole)
        matches = [idx for idx, frag in enumerate(fragments) if idx not in used and hole_shape in frag["variants"]]
        assert len(matches) == 1, "every hole must match exactly one fragment"
        frag = fragments[matches[0]]
        used.add(matches[0])

        source_colors = []
        for r, c in hole:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    color = int(g[nr, nc])
                    if color not in (bg, skeleton):
                        source_colors.append(color)
        fill = Counter(source_colors).most_common(1)[0][0] if source_colors else frag["color"]
        for r, c in hole:
            out[r, c] = int(fill)

    return out


def _mark_with_padding(mask: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                mask[rr, cc] = True


def _can_place_hole(
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
    forbidden: np.ndarray,
    rect_top: int,
    rect_left: int,
    rect_bottom: int,
    rect_right: int,
) -> bool:
    cells = _translate_shape(shape, top, left)
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not (rect_top <= rr <= rect_bottom and rect_left <= cc <= rect_right):
                    return False
                if forbidden[rr, cc]:
                    return False
    return True


def _pick_source_cells(
    rng: random.Random,
    hole: list[tuple[int, int]],
    grid: np.ndarray,
) -> list[tuple[int, int]]:
    boundary = sorted(
        {
            (nr, nc)
            for r, c in hole
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == GREY
        }
    )
    assert boundary, "each hole needs at least one adjacent grey cell"
    pairs = [pair for pair in combinations(boundary, 2) if abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1]) == 1]
    if pairs and rng.random() < 0.5:
        choice = rng.choice(pairs)
        return [choice[0], choice[1]]
    return [rng.choice(boundary)]


def _can_place_fragment(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    bg: int,
    pad: int = 1,
) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 < r < h - 1 and 0 < c < w - 1):
            return False
        if int(grid[r, c]) != bg:
            return False
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                if (rr, cc) in cell_set:
                    continue
                if int(grid[rr, cc]) != bg:
                    return False
    return True


def _side_ranges(
    side: str,
    shape_h: int,
    shape_w: int,
    grid_h: int,
    grid_w: int,
    skel_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    sr0, sc0, sr1, sc1 = skel_bbox
    if side == "top":
        row_hi = sr0 - shape_h - 2
        if row_hi < 1:
            return None
        return (1, row_hi), (1, grid_w - shape_w - 2)
    if side == "bottom":
        row_lo = sr1 + 2
        row_hi = grid_h - shape_h - 2
        if row_lo > row_hi:
            return None
        return (row_lo, row_hi), (1, grid_w - shape_w - 2)
    if side == "left":
        col_hi = sc0 - shape_w - 2
        if col_hi < 1:
            return None
        return (1, grid_h - shape_h - 2), (1, col_hi)
    col_lo = sc1 + 2
    col_hi = grid_w - shape_w - 2
    if col_lo > col_hi:
        return None
    return (1, grid_h - shape_h - 2), (col_lo, col_hi)


def _expected_output(
    grid: np.ndarray,
    bg: int,
    holes: list[list[tuple[int, int]]],
    fragments: list[list[tuple[int, int]]],
    fragment_colors: list[int],
    source_fill: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for cells in fragments:
        for r, c in cells:
            out[r, c] = bg
    for idx, hole in enumerate(holes):
        fill = source_fill.get(idx, fragment_colors[idx])
        for r, c in hole:
            out[r, c] = int(fill)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    hole_count = rng.choice([2, 3, 4])

    rect_h = rng.randint(max(10, hole_count * 3 + 1), 15)
    rect_w = rng.randint(max(10, hole_count * 3 + 1), 15)
    grid_h = rng.randint(rect_h + 6, 30)
    grid_w = rng.randint(rect_w + 6, 30)

    bg = rng.choice([color for color in range(10) if color != GREY])
    palette = [color for color in range(10) if color not in (bg, GREY)]
    rng.shuffle(palette)

    fragment_colors = palette[:hole_count]
    source_count = rng.randint(1, hole_count - 1)
    source_indices = set(rng.sample(range(hole_count), source_count))
    source_fill = {hole_idx: palette[hole_count + offset] for offset, hole_idx in enumerate(sorted(source_indices))}

    grid = np.full((grid_h, grid_w), bg, dtype=int)

    r0 = (grid_h - rect_h) // 2 + rng.randint(-1, 1)
    c0 = (grid_w - rect_w) // 2 + rng.randint(-1, 1)
    r0 = max(3, min(r0, grid_h - rect_h - 3))
    c0 = max(3, min(c0, grid_w - rect_w - 3))

    skeleton_mask = np.zeros((grid_h, grid_w), dtype=bool)
    skeleton_mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
    for _ in range(rng.randint(2, 4)):
        side = rng.choice(SIDES)
        depth = rng.randint(1, 2)
        if side in ("top", "bottom"):
            width = rng.randint(3, rect_w - 2)
            start = rng.randint(c0 + 1, c0 + rect_w - width - 1)
            if side == "top" and r0 - depth >= 1:
                skeleton_mask[r0 - depth : r0, start : start + width] = True
            if side == "bottom" and r0 + rect_h + depth <= grid_h - 1:
                skeleton_mask[r0 + rect_h : r0 + rect_h + depth, start : start + width] = True
        else:
            height = rng.randint(3, rect_h - 2)
            start = rng.randint(r0 + 1, r0 + rect_h - height - 1)
            if side == "left" and c0 - depth >= 1:
                skeleton_mask[start : start + height, c0 - depth : c0] = True
            if side == "right" and c0 + rect_w + depth <= grid_w - 1:
                skeleton_mask[start : start + height, c0 + rect_w : c0 + rect_w + depth] = True
    grid[skeleton_mask] = GREY

    chosen_shapes = rng.sample(SHAPE_LIBRARY, hole_count)
    forbidden = np.zeros((grid_h, grid_w), dtype=bool)
    holes: list[list[tuple[int, int]]] = []
    hole_shapes: list[tuple[tuple[int, int], ...]] = []
    rect_bottom = r0 + rect_h - 1
    rect_right = c0 + rect_w - 1
    for base in chosen_shapes:
        shape = rng.choice(_shape_variants(base))
        sh, sw = _shape_dims(shape)
        placed = False
        for _ in range(120):
            top = rng.randint(r0 + 1, r0 + rect_h - sh - 1)
            left = rng.randint(c0 + 1, c0 + rect_w - sw - 1)
            if not _can_place_hole(shape, top, left, forbidden, r0, c0, rect_bottom, rect_right):
                continue
            cells = _translate_shape(shape, top, left)
            holes.append(cells)
            hole_shapes.append(shape)
            _mark_with_padding(forbidden, cells, pad=1)
            for r, c in cells:
                grid[r, c] = bg
            placed = True
            break
        if not placed:
            return None

    for idx in sorted(source_indices):
        source_cells = _pick_source_cells(rng, holes[idx], grid)
        for r, c in source_cells:
            grid[r, c] = int(source_fill[idx])

    skel_cells = [(int(r), int(c)) for r, c in np.argwhere(skeleton_mask)]
    skel_bbox = _bbox(skel_cells)

    fragments: list[list[tuple[int, int]]] = [None] * hole_count  # type: ignore[assignment]
    order = sorted(range(hole_count), key=lambda idx: (-len(holes[idx]), -max(_shape_dims(hole_shapes[idx]))))
    for idx in order:
        shape = rng.choice(_shape_variants(chosen_shapes[idx]))
        sh, sw = _shape_dims(shape)
        placed = False
        side_order = list(SIDES)
        rng.shuffle(side_order)
        for side in side_order:
            ranges = _side_ranges(side, sh, sw, grid_h, grid_w, skel_bbox)
            if ranges is None:
                continue
            (row_lo, row_hi), (col_lo, col_hi) = ranges
            for _ in range(100):
                top = rng.randint(row_lo, row_hi)
                left = rng.randint(col_lo, col_hi)
                cells = _translate_shape(shape, top, left)
                if not _can_place_fragment(grid, cells, bg, pad=1):
                    continue
                for r, c in cells:
                    grid[r, c] = int(fragment_colors[idx])
                fragments[idx] = cells
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if any(cells is None for cells in fragments):
        return None

    colors, counts = np.unique(grid, return_counts=True)
    if int(colors[int(np.argmax(counts))]) != bg:
        return None

    expected = _expected_output(grid, bg, holes, fragments, fragment_colors, source_fill)  # type: ignore[arg-type]
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None
    if np.array_equal(actual, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
