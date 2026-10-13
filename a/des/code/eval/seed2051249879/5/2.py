from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SHAPE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GEN_RETRIES = 400


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
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...] | set[tuple[int, int]]):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_cells(cells: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]):
    r0, c0, r1, c1 = _bbox(cells)
    norm = tuple(sorted((r - r0, c - c0) for r, c in cells))
    return norm, r1 - r0 + 1, c1 - c0 + 1


def _make_shape(mask: np.ndarray, color: int) -> dict:
    cells = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
    norm, h, w = _normalize_cells(cells)
    arr = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        arr[r, c] = True
    return {
        "color": int(color),
        "mask": arr,
        "cells": norm,
        "h": int(h),
        "w": int(w),
        "area": int(len(norm)),
    }


def _candidate_placements(shape: dict, hole_mask: np.ndarray) -> list[tuple[int, int, frozenset[tuple[int, int]]]]:
    h, w = hole_mask.shape
    sh, sw = int(shape["h"]), int(shape["w"])
    if sh > h or sw > w:
        return []
    cands: list[tuple[int, int, frozenset[tuple[int, int]]]] = []
    for top in range(h - sh + 1):
        for left in range(w - sw + 1):
            pts: list[tuple[int, int]] = []
            ok = True
            for dr, dc in shape["cells"]:
                rr, cc = top + dr, left + dc
                if not hole_mask[rr, cc]:
                    ok = False
                    break
                pts.append((rr, cc))
            if ok:
                cands.append((top, left, frozenset(pts)))
    return cands


def _solve_placements(
    shapes: list[dict],
    hole_mask: np.ndarray,
    max_solutions: int = 2,
) -> list[list[tuple[int, int]]]:
    hole_cells = frozenset(tuple(map(int, pt)) for pt in np.argwhere(hole_mask))
    candidates = [_candidate_placements(shape, hole_mask) for shape in shapes]
    if any(not cand for cand in candidates):
        return []

    all_cells = [frozenset().union(*(cells for _, _, cells in cand)) for cand in candidates]
    solutions: list[list[tuple[int, int]]] = []
    n = len(shapes)

    def recurse(covered: frozenset[tuple[int, int]], remaining: tuple[int, ...], placed: dict[int, tuple[int, int]]):
        if len(solutions) >= max_solutions:
            return
        uncovered = hole_cells - covered
        if not remaining:
            if not uncovered:
                sol = [(-1, -1)] * n
                for idx, pos in placed.items():
                    sol[idx] = pos
                solutions.append(sol)
            return

        coverable = frozenset().union(*(all_cells[idx] for idx in remaining))
        if not uncovered.issubset(coverable):
            return

        best_idx = min(remaining, key=lambda idx: len(candidates[idx]))
        next_remaining = tuple(idx for idx in remaining if idx != best_idx)
        next_coverable = frozenset().union(*(all_cells[idx] for idx in next_remaining))
        options = sorted(
            candidates[best_idx],
            key=lambda item: (-len(item[2] - covered), item[0], item[1]),
        )
        for top, left, cells in options:
            new_covered = covered | cells
            new_uncovered = hole_cells - new_covered
            if new_uncovered and not new_uncovered.issubset(next_coverable):
                continue
            placed[best_idx] = (int(top), int(left))
            recurse(new_covered, next_remaining, placed)
            del placed[best_idx]
            if len(solutions) >= max_solutions:
                return

    recurse(frozenset(), tuple(range(n)), {})
    return solutions


def _render_canvas(
    canvas: np.ndarray,
    shapes: list[dict],
    placements: list[tuple[int, int]],
    priority: dict[int, int],
) -> np.ndarray:
    out = np.array(canvas, dtype=int, copy=True)
    order = sorted(range(len(shapes)), key=lambda idx: priority[int(shapes[idx]["color"])])
    for idx in order:
        color = int(shapes[idx]["color"])
        top, left = placements[idx]
        for dr, dc in shapes[idx]["cells"]:
            out[1 + top + dr, 1 + left + dc] = color
    return out


def _find_separator_row(grid: np.ndarray) -> int:
    h = grid.shape[0]
    candidates = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    for r in candidates:
        if r == 0 or r == h - 1:
            continue
        if np.any((grid[:r, :] != BLACK) & (grid[:r, :] != GREY)) and np.any(grid[r + 1 :, :] == GREY):
            return int(r)
    raise AssertionError("separator row not found")


def _extract_shapes(top: np.ndarray) -> list[dict]:
    comps = _components((top != BLACK) & (top != GREY))
    shapes: list[dict] = []
    for comp in comps:
        colors = {int(top[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = next(iter(colors))
        norm, h, w = _normalize_cells(comp)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        shapes.append(_make_shape(mask, color))
    assert shapes
    return shapes


def _find_canvas_bbox(lower: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(lower == GREY)
    best = None
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        sub = lower[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(sub[0, :] == GREY)
            and np.all(sub[-1, :] == GREY)
            and np.all(sub[:, 0] == GREY)
            and np.all(sub[:, -1] == GREY)
        ):
            continue
        score = len(comp)
        if best is None or score > best[0]:
            best = (score, (r0, c0, r1, c1))
    assert best is not None
    return best[1]


def _read_priority(lower: np.ndarray, canvas_bbox: tuple[int, int, int, int]) -> dict[int, int]:
    _r0, _c0, r1, _c1 = canvas_bbox
    bar_r = r1 + 1
    assert bar_r < lower.shape[0]
    bar = [int(v) for v in lower[bar_r, :] if int(v) not in (BLACK, GREY)]
    assert bar
    priority: dict[int, int] = {}
    for i, color in enumerate(bar):
        priority[int(color)] = int(i)
    return priority


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the top sprites to the lower grey template, then color the black holes
    using the bottom color-strip stacking order.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_row(g)
    top = g[:sep, :]
    lower = g[sep + 1 :, :]

    shapes = _extract_shapes(top)
    canvas_bbox = _find_canvas_bbox(lower)
    r0, c0, r1, c1 = canvas_bbox
    canvas = lower[r0 : r1 + 1, c0 : c1 + 1].copy()
    hole_mask = canvas[1:-1, 1:-1] == BLACK
    assert hole_mask.any()

    priority = _read_priority(lower, canvas_bbox)
    for shape in shapes:
        assert int(shape["color"]) in priority

    solutions = _solve_placements(shapes, hole_mask, max_solutions=2)
    assert len(solutions) == 1
    return _render_canvas(canvas, shapes, solutions[0], priority)


def _random_shape_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(100):
        target = rng.randint(4, 8)
        max_h = rng.randint(2, 4)
        max_w = rng.randint(3, 5)
        cells: set[tuple[int, int]] = {(0, 0)}
        while len(cells) < target:
            frontier: list[set[tuple[int, int]]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    cand = set(cells)
                    cand.add((nr, nc))
                    norm, h, w = _normalize_cells(cand)
                    if h <= max_h and w <= max_w:
                        frontier.append(set(norm))
            if not frontier:
                break
            cells = rng.choice(frontier)
        norm, h, w = _normalize_cells(cells)
        if len(norm) != target or h < 2 or w < 2:
            continue
        if len(norm) == h * w:
            continue
        if all(r == norm[0][0] for r, _ in norm) or all(c == norm[0][1] for _, c in norm):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        return mask
    return None


def _sample_shapes(rng: random.Random, count: int) -> list[dict] | None:
    colors = rng.sample(SHAPE_COLORS, count)
    shapes: list[dict] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for color in colors:
        shape = None
        for _ in range(100):
            mask = _random_shape_mask(rng)
            if mask is None:
                continue
            cand = _make_shape(mask, color)
            sig = tuple(cand["cells"])
            if sig in seen:
                continue
            seen.add(sig)
            shape = cand
            break
        if shape is None:
            return None
        shapes.append(shape)
    return shapes


def _cells_shifted(shape: dict, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape["cells"]}


def _expanded(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _sample_overlap_pair(shape_a: dict, shape_b: dict, rng: random.Random):
    a_cells = set(shape_a["cells"])
    candidates = []
    for dr in range(-int(shape_b["h"]) + 1, int(shape_a["h"])):
        for dc in range(-int(shape_b["w"]) + 1, int(shape_a["w"])):
            b_cells = {(r + dr, c + dc) for r, c in shape_b["cells"]}
            inter = a_cells & b_cells
            if not inter or len(inter) >= min(int(shape_a["area"]), int(shape_b["area"])):
                continue
            union = a_cells | b_cells
            norm_union, uh, uw = _normalize_cells(union)
            hole_mask = np.zeros((uh, uw), dtype=bool)
            for r, c in norm_union:
                hole_mask[r, c] = True
            sols = _solve_placements([shape_a, shape_b], hole_mask, max_solutions=2)
            if len(sols) != 1:
                continue
            placements = sols[0]
            inter_ratio = len(inter) / min(int(shape_a["area"]), int(shape_b["area"]))
            candidates.append((inter_ratio, hole_mask, placements))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    band = candidates[: max(1, len(candidates) // 2)]
    _score, hole_mask, placements = rng.choice(band)
    return hole_mask, placements


def _place_shape_nonoverlap(
    shape: dict,
    inner_h: int,
    inner_w: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
):
    max_top = inner_h - int(shape["h"]) - 1
    max_left = inner_w - int(shape["w"]) - 1
    if max_top < 1 or max_left < 1:
        return None
    positions = [(r, c) for r in range(1, max_top + 1) for c in range(1, max_left + 1)]
    rng.shuffle(positions)
    for top, left in positions:
        cells = _cells_shifted(shape, top, left)
        if cells & blocked:
            continue
        return top, left, cells
    return None


def _place_top_shapes(shapes: list[dict], h: int, w: int, rng: random.Random):
    blocked: set[tuple[int, int]] = set()
    placements: list[tuple[int, int]] = []
    order = sorted(range(len(shapes)), key=lambda idx: -int(shapes[idx]["area"]))
    pos_by_idx: dict[int, tuple[int, int]] = {}
    for idx in order:
        shape = shapes[idx]
        max_top = h - int(shape["h"])
        max_left = w - int(shape["w"])
        if max_top < 1 or max_left < 1:
            return None
        positions = [(r, c) for r in range(0, max_top + 1) for c in range(0, max_left + 1)]
        rng.shuffle(positions)
        placed = None
        for top, left in positions:
            cells = _cells_shifted(shape, top, left)
            if cells & blocked:
                continue
            placed = (top, left, cells)
            break
        if placed is None:
            return None
        top, left, cells = placed
        pos_by_idx[idx] = (top, left)
        blocked |= _expanded(cells, h, w)
    for idx in range(len(shapes)):
        placements.append(pos_by_idx[idx])
    return placements


def _make_bar(colors: list[int], rng: random.Random) -> list[int]:
    bar = colors[:]
    rng.shuffle(bar)
    if rng.random() < 0.5:
        extra = rng.randint(1, min(2, len(colors)))
        for _ in range(extra):
            color = rng.choice(colors)
            pos = rng.randint(0, len(bar))
            bar.insert(pos, color)
    return bar


def _priority_from_bar(bar: list[int]) -> dict[int, int]:
    priority: dict[int, int] = {}
    for i, color in enumerate(bar):
        priority[int(color)] = int(i)
    return priority


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape_count = rng.choice([2, 3, 4])
    shapes = _sample_shapes(rng, shape_count)
    if shapes is None:
        return None

    overlap_mode = rng.choice([False, True])
    pair_mask = None
    pair_places = None
    overlap_pair = (0, 1)
    if overlap_mode:
        sampled = _sample_overlap_pair(shapes[0], shapes[1], rng)
        if sampled is None:
            return None
        pair_mask, pair_places = sampled

    shape_areas = sum(int(shape["area"]) for shape in shapes)
    max_shape_h = max(int(shape["h"]) for shape in shapes)
    max_shape_w = max(int(shape["w"]) for shape in shapes)
    base_h = max(8, max_shape_h + 4, shape_areas // 5 + 5)
    base_w = max(10, max_shape_w + 6, shape_areas // 4 + 6)
    if pair_mask is not None:
        base_h = max(base_h, pair_mask.shape[0] + 4)
        base_w = max(base_w, pair_mask.shape[1] + 4)
    base_h += max(0, shape_count - 3)
    base_w += max(0, shape_count - 2)
    inner_h = rng.randint(base_h, min(base_h + 4, 15))
    inner_w = rng.randint(base_w, min(base_w + 5, 18))

    placements: list[tuple[int, int] | None] = [None] * shape_count
    hole_cells: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    if pair_mask is not None and pair_places is not None:
        ph, pw = pair_mask.shape
        if ph > inner_h - 2 or pw > inner_w - 2:
            return None
        base_top = rng.randint(1, inner_h - ph - 1)
        base_left = rng.randint(1, inner_w - pw - 1)
        pair_union = {(base_top + r, base_left + c) for r, c in np.argwhere(pair_mask)}
        hole_cells |= pair_union
        blocked |= _expanded(pair_union, inner_h, inner_w)
        for local_idx, shape_idx in enumerate(overlap_pair):
            sr, sc = pair_places[local_idx]
            placements[shape_idx] = (base_top + sr, base_left + sc)

    order = list(range(shape_count))
    rng.shuffle(order)
    for idx in order:
        if placements[idx] is not None:
            continue
        placed = _place_shape_nonoverlap(shapes[idx], inner_h, inner_w, blocked, rng)
        if placed is None:
            return None
        top, left, cells = placed
        placements[idx] = (top, left)
        hole_cells |= cells
        blocked |= _expanded(cells, inner_h, inner_w)

    final_places = [pos for pos in placements if pos is not None]
    if len(final_places) != shape_count:
        return None
    final_places = list(final_places)

    if overlap_mode:
        cells_a = _cells_shifted(shapes[0], final_places[0][0], final_places[0][1])
        cells_b = _cells_shifted(shapes[1], final_places[1][0], final_places[1][1])
        if not (cells_a & cells_b):
            return None
    else:
        for i in range(shape_count):
            for j in range(i + 1, shape_count):
                cells_i = _cells_shifted(shapes[i], final_places[i][0], final_places[i][1])
                cells_j = _cells_shifted(shapes[j], final_places[j][0], final_places[j][1])
                if cells_i & cells_j:
                    return None

    hole_mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r, c in hole_cells:
        hole_mask[r, c] = True
    sols = _solve_placements(shapes, hole_mask, max_solutions=2)
    if len(sols) != 1:
        return None

    bar = _make_bar([int(shape["color"]) for shape in shapes], rng)
    priority = _priority_from_bar(bar)
    canvas = np.full((inner_h + 2, inner_w + 2), GREY, dtype=int)
    canvas[1:-1, 1:-1][hole_mask] = BLACK
    expected = _render_canvas(canvas, shapes, sols[0], priority)

    canvas_h, canvas_w = canvas.shape
    lower_top_margin = rng.randint(1, 2)
    lower_bottom_margin = rng.randint(1, 2)
    lower_h = lower_top_margin + canvas_h + 1 + lower_bottom_margin
    lower_min_w = canvas_w + 4

    top_h = rng.randint(max(6, max_shape_h + 2), max(8, max_shape_h + 6))
    top_min_w = sum(int(shape["w"]) for shape in shapes) + 2 * shape_count + 2
    grid_w = rng.randint(max(lower_min_w, top_min_w), max(lower_min_w, top_min_w) + 6)

    top_places = _place_top_shapes(shapes, top_h, grid_w, rng)
    if top_places is None:
        return None

    lower = np.full((lower_h, grid_w), BLACK, dtype=int)
    canvas_left = rng.randint(1, grid_w - canvas_w - 1)
    canvas_top = lower_top_margin
    lower[canvas_top : canvas_top + canvas_h, canvas_left : canvas_left + canvas_w] = canvas

    if len(bar) > canvas_w - 2:
        return None
    bar_row = canvas_top + canvas_h
    bar_start = rng.randint(canvas_left + 1, canvas_left + canvas_w - 1 - len(bar))
    lower[bar_row, bar_start : bar_start + len(bar)] = np.array(bar, dtype=int)

    top = np.full((top_h, grid_w), BLACK, dtype=int)
    for shape, (top_r, top_c) in zip(shapes, top_places):
        for dr, dc in shape["cells"]:
            top[top_r + dr, top_c + dc] = int(shape["color"])

    separator = np.full((1, grid_w), GREY, dtype=int)
    grid = np.vstack([top, separator, lower])

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
