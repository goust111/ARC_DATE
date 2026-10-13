from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
TEMP_CANVAS = 30
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < int(shape[0]) and 0 <= c < int(shape[1])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _border_neighbors(border: set[tuple[int, int]], cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4 if (r + dr, c + dc) in border]


def _cycle_order_from(
    border: set[tuple[int, int]],
    start: tuple[int, int],
    nxt: tuple[int, int],
) -> list[tuple[int, int]]:
    order = [start]
    prev = start
    cur = nxt
    while cur != start:
        order.append(cur)
        options = [cell for cell in _border_neighbors(border, cur) if cell != prev]
        assert len(options) == 1, "border must form a simple 4-connected cycle"
        prev, cur = cur, options[0]
    return order


def _clockwise_score(order: list[tuple[int, int]], region: set[tuple[int, int]]) -> int:
    score = 0
    for a, b in zip(order, order[1:] + order[:1]):
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        assert abs(dr) + abs(dc) == 1, "border walk must move by 4-neighbors"
        right = (dc, -dr)
        if (a[0] + right[0], a[1] + right[1]) in region or (b[0] + right[0], b[1] + right[1]) in region:
            score += 1
    return score


def _clockwise_border_order(border: set[tuple[int, int]], region: set[tuple[int, int]]) -> list[tuple[int, int]]:
    assert border, "expected a non-empty border"
    start = min(border)
    neighbors = _border_neighbors(border, start)
    assert len(neighbors) == 2, "top-left border pixel must have degree 2"
    order_a = _cycle_order_from(border, start, neighbors[0])
    order_b = _cycle_order_from(border, start, neighbors[1])
    score_a = _clockwise_score(order_a, region)
    score_b = _clockwise_score(order_b, region)
    if score_b > score_a:
        return order_b
    if score_a > score_b:
        return order_a
    if (start[0], start[1] + 1) in border:
        return order_a if order_a[1] == (start[0], start[1] + 1) else order_b
    return order_a


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each seeded enclosure, then project its filled area onto the surrounding grey border as evenly spaced markers.

    Input structure:
    - The background is black.
    - One-cell-thick grey outlines form disjoint closed axis-aligned polygons; each outline is a rectangle or an L shape.
    - Inside each closed region there is exactly one non-black, non-grey seed pixel, and all remaining interior cells are black.

    Rule:
    - Treat grey as walls and find every enclosed non-grey-connected interior component that does not touch the outer grid border.
    - For each such region, recolor all interior black cells to the seed color while keeping the seed pixel unchanged.
    - Count how many black cells were recolored, then walk the adjacent grey border clockwise from its top-left pixel and recolor every other border pixel with the seed color until that many placements have been made, wrapping around the cycle if needed.

    Key insight:
    - The same per-region number is used twice: first as the count of cells filled inside the enclosure, then as the count of equally spaced border markers placed around that enclosure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_components = _components(g == GREY)
    grey_lookup: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(grey_components):
        for cell in comp:
            grey_lookup[cell] = idx

    regions = []
    for comp in _components(g != GREY):
        if _touches_border(comp, g.shape):
            continue
        cells = set(comp)
        seed_cells = [(r, c) for r, c in comp if int(g[r, c]) not in {BLACK, GREY}]
        assert len(seed_cells) == 1, "each enclosed region must contain exactly one seed"
        seed_r, seed_c = seed_cells[0]
        seed_color = int(g[seed_r, seed_c])
        black_cells = [(r, c) for r, c in comp if int(g[r, c]) == BLACK]
        border_ids = set()
        for r, c in comp:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == GREY:
                    border_ids.add(grey_lookup[(nr, nc)])
        assert len(border_ids) == 1, "each enclosed region must belong to exactly one grey outline"
        border = set(grey_components[next(iter(border_ids))])
        regions.append(
            {
                "cells": cells,
                "seed_color": seed_color,
                "black_cells": black_cells,
                "border_order": _clockwise_border_order(border, cells),
            }
        )

    out = g.copy()
    for region in regions:
        color = region["seed_color"]
        for r, c in region["black_cells"]:
            out[r, c] = color
        count = len(region["black_cells"])
        border_order = region["border_order"]
        for idx in range(count):
            r, c = border_order[(2 * idx) % len(border_order)]
            out[r, c] = color
    return out


def _draw_segment(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    cells = set()
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.add((r0, c))
    elif c0 == c1:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.add((r, c0))
    else:
        raise AssertionError("polygon edges must be axis-aligned")
    return cells


def _draw_polygon(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        border |= _draw_segment(a, b)
    return border


def _interior_from_border(border: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_r = max(r for r, _ in border)
    max_c = max(c for _, c in border)
    canvas = np.zeros((max_r + 3, max_c + 3), dtype=int)
    for r, c in border:
        canvas[r + 1, c + 1] = 1

    seen = np.zeros_like(canvas, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1] and canvas[nr, nc] == 0 and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    interior = set()
    for r in range(max_r + 1):
        for c in range(max_c + 1):
            if canvas[r + 1, c + 1] == 0 and not seen[r + 1, c + 1]:
                interior.add((r, c))
    return interior


def _is_simple_cycle(border: set[tuple[int, int]]) -> bool:
    if not border:
        return False
    for cell in border:
        if len(_border_neighbors(border, cell)) != 2:
            return False
    return len(_components(_mask_from_cells(border))) == 1


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    return len(_components(_mask_from_cells(cells))) == 1


def _rectangle_candidate(interior_h: int, interior_w: int) -> dict:
    border = set()
    full_h = interior_h + 2
    full_w = interior_w + 2
    for c in range(full_w):
        border.add((0, c))
        border.add((full_h - 1, c))
    for r in range(full_h):
        border.add((r, 0))
        border.add((r, full_w - 1))
    interior = {(r, c) for r in range(1, full_h - 1) for c in range(1, full_w - 1)}
    return {"kind": "rectangle", "h": full_h, "w": full_w, "border": border, "interior": interior}


def _l_vertices(h: int, w: int, notch_h: int, notch_w: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tr":
        return [(0, 0), (0, w - 1 - notch_w), (notch_h, w - 1 - notch_w), (notch_h, w - 1), (h - 1, w - 1), (h - 1, 0)]
    if corner == "tl":
        return [(0, notch_w), (0, w - 1), (h - 1, w - 1), (h - 1, 0), (notch_h, 0), (notch_h, notch_w)]
    if corner == "br":
        return [(0, 0), (0, w - 1), (h - 1 - notch_h, w - 1), (h - 1 - notch_h, w - 1 - notch_w), (h - 1, w - 1 - notch_w), (h - 1, 0)]
    if corner == "bl":
        return [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, notch_w), (h - 1 - notch_h, notch_w), (h - 1 - notch_h, 0)]
    raise AssertionError("unknown L orientation")


def _build_shape_candidates() -> dict[int, dict[str, list[dict]]]:
    candidates = {count: {"rectangle": [], "l": []} for count in range(1, 11)}

    for total_area in range(2, 12):
        black_count = total_area - 1
        for interior_h in range(1, total_area + 1):
            if total_area % interior_h != 0:
                continue
            interior_w = total_area // interior_h
            candidates[black_count]["rectangle"].append(_rectangle_candidate(interior_h, interior_w))

    seen_l: set[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = set()
    for h in range(4, 9):
        for w in range(4, 9):
            for notch_h in range(1, h - 2):
                for notch_w in range(1, w - 2):
                    for corner in ("tr", "tl", "br", "bl"):
                        border = _draw_polygon(_l_vertices(h, w, notch_h, notch_w, corner))
                        if not _is_simple_cycle(border):
                            continue
                        interior = _interior_from_border(border)
                        if not interior or not _is_connected(interior):
                            continue
                        black_count = len(interior) - 1
                        if black_count not in candidates:
                            continue
                        min_r = min(r for r, _ in interior)
                        min_c = min(c for _, c in interior)
                        max_r = max(r for r, _ in interior)
                        max_c = max(c for _, c in interior)
                        if len(interior) == (max_r - min_r + 1) * (max_c - min_c + 1):
                            continue
                        key = (tuple(sorted(border)), tuple(sorted(interior)))
                        if key in seen_l:
                            continue
                        seen_l.add(key)
                        candidates[black_count]["l"].append(
                            {"kind": "l", "h": h, "w": w, "border": border, "interior": interior}
                        )
    return candidates


SHAPE_CANDIDATES = _build_shape_candidates()


def _choose_shapes() -> list[dict] | None:
    count_options = [1, 2, 2, 3, 3, 4]
    num_shapes = random.choice(count_options)
    black_counts = random.sample(list(range(1, 11)), num_shapes)
    colors = random.sample(SEED_COLORS, num_shapes)

    desired_kinds: list[str] = ["" for _ in range(num_shapes)]
    l_slots = [i for i, count in enumerate(black_counts) if SHAPE_CANDIDATES[count]["l"]]
    rect_slots = [i for i, count in enumerate(black_counts) if SHAPE_CANDIDATES[count]["rectangle"]]
    if num_shapes >= 2 and l_slots and rect_slots and random.random() < 0.6:
        l_idx = random.choice(l_slots)
        desired_kinds[l_idx] = "l"
        rect_choices = [idx for idx in rect_slots if idx != l_idx]
        if rect_choices:
            desired_kinds[random.choice(rect_choices)] = "rectangle"

    shapes = []
    for idx, black_count in enumerate(black_counts):
        options = []
        if SHAPE_CANDIDATES[black_count]["rectangle"]:
            options.append("rectangle")
        if SHAPE_CANDIDATES[black_count]["l"]:
            options.append("l")
        if not options:
            return None
        kind = desired_kinds[idx] or random.choice(options)
        if kind not in options:
            kind = random.choice(options)
        base = random.choice(SHAPE_CANDIDATES[black_count][kind])
        shapes.append(
            {
                "kind": kind,
                "h": base["h"],
                "w": base["w"],
                "border": base["border"],
                "interior": base["interior"],
                "seed_color": colors[idx],
                "black_count": black_count,
            }
        )
    return shapes


def _try_place_shapes(shapes: list[dict]) -> np.ndarray | None:
    canvas = np.zeros((TEMP_CANVAS, TEMP_CANVAS), dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)

    for shape in sorted(shapes, key=lambda item: item["h"] * item["w"], reverse=True):
        placed = False
        for _ in range(120):
            r0 = random.randint(1, TEMP_CANVAS - shape["h"] - 1)
            c0 = random.randint(1, TEMP_CANVAS - shape["w"] - 1)
            r1 = r0 + shape["h"] - 1
            c1 = c0 + shape["w"] - 1
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(TEMP_CANVAS, r1 + 2)
            pc1 = min(TEMP_CANVAS, c1 + 2)
            if occupied[pr0:pr1, pc0:pc1].any():
                continue

            seed_pos = random.choice(sorted(shape["interior"]))
            for rr, cc in shape["border"]:
                canvas[r0 + rr, c0 + cc] = GREY
            for rr, cc in shape["interior"]:
                canvas[r0 + rr, c0 + cc] = shape["seed_color"] if (rr, cc) == seed_pos else BLACK
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            placed = True
            break
        if not placed:
            return None
    return canvas


def _crop_canvas(canvas: np.ndarray) -> np.ndarray:
    coords = np.argwhere(canvas != BLACK)
    assert coords.size > 0
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    top = random.randint(0, max(0, int(min_r) - 1))
    left = random.randint(0, max(0, int(min_c) - 1))
    bottom = random.randint(int(max_r) + 2, canvas.shape[0])
    right = random.randint(int(max_c) + 2, canvas.shape[1])
    return canvas[top:bottom, left:right]


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        shapes = _choose_shapes()
        if shapes is None:
            continue
        canvas = _try_place_shapes(shapes)
        if canvas is None:
            continue
        grid = _crop_canvas(canvas)
        if max(grid.shape) > 30:
            continue
        try:
            main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
