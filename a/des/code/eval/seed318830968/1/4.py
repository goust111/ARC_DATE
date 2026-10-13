from __future__ import annotations

import numpy as np


BLACK = 0
GREEN = 3
ORANGE = 7

FILL_CHOICES = (1, 2, 4, 5, 6, 8, 9)
GRID_MIN = 18
GRID_MAX = 30
MIN_INNER_BOX = 5
GEN_RETRIES = 400
CHAIN_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c))
        out.add((r1, c))
    for r in range(r0, r1 + 1):
        out.add((r, c0))
        out.add((r, c1))
    return out


def _detect_legend(g: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = g.shape
    candidates: list[tuple[int, int, int, int]] = []
    allowed = {BLACK, GREEN, ORANGE}

    for c in (0, w - 1):
        r = 0
        while r < h:
            if int(g[r, c]) in allowed:
                r += 1
                continue
            start = r
            while r < h and int(g[r, c]) not in allowed:
                r += 1
            end = r
            vals = [int(v) for v in g[start:end, c].tolist()]
            if len(vals) >= 2 and len(set(vals)) == len(vals) and (start == 0 or end == h):
                candidates.append((len(vals), start, end, c))

    assert candidates, "expected one vertical corner legend on the border"
    candidates.sort(reverse=True)
    _length, start, end, c = candidates[0]
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[start:end, c] = True
    colors = [int(v) for v in g[start:end, c].tolist()]
    return colors, legend_mask


def _find_rectangles(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for comp in _components(g == GREEN):
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "green component must be a proper rectangle border"
        expected = _border_cells(box)
        assert set(comp) == expected, "green components must be 1-pixel rectangle borders"
        rects.append(box)
    assert rects, "expected at least one rectangle border"
    return rects


def _region_infos(
    g: np.ndarray,
    legend_mask: np.ndarray,
    rectangles: list[tuple[int, int, int, int]],
) -> list[dict[str, object]]:
    open_mask = (g != GREEN) & (~legend_mask)
    infos: list[dict[str, object]] = []
    for cells in _components(open_mask):
        r, c = cells[0]
        depth = sum(1 for r0, c0, r1, c1 in rectangles if r0 < r < r1 and c0 < c < c1)
        info = {
            "cells": cells,
            "depth": int(depth),
            "top": min(rr for rr, _ in cells),
            "left": min(cc for _, cc in cells),
        }
        infos.append(info)
    infos.sort(key=lambda info: (int(info["depth"]), int(info["top"]), int(info["left"])))
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each enclosed region by nesting depth, and use orange-marker counts to set stripe spacing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_colors, legend_mask = _detect_legend(g)
    rectangles = _find_rectangles(g)
    infos = _region_infos(g, legend_mask, rectangles)
    max_depth = max(int(info["depth"]) for info in infos)
    assert len(legend_colors) >= max_depth, "legend must cover every nesting depth"

    out = g.copy()
    for info in infos:
        cells = info["cells"]
        depth = int(info["depth"])
        top = int(info["top"])
        marker_count = sum(1 for r, c in cells if int(g[r, c]) == ORANGE)

        if depth == 0:
            fill_color = BLACK
            spacing = 0
        else:
            fill_color = int(legend_colors[depth - 1])
            spacing = 1 if marker_count == 0 else marker_count + 1

        for r, c in cells:
            if depth == 0:
                out[r, c] = BLACK
            elif spacing == 1 or (r - top) % spacing == 0:
                out[r, c] = fill_color
            else:
                out[r, c] = BLACK

    return out


def _min_outer_size(depth: int) -> int:
    return MIN_INNER_BOX + 4 * (depth - 1)


def _paint_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_legend(grid: np.ndarray, colors: list[int], corner: str) -> np.ndarray:
    h, w = grid.shape
    n = len(colors)
    mask = np.zeros((h, w), dtype=bool)

    if corner == "tl":
        rows = range(0, n)
        col = 0
    elif corner == "tr":
        rows = range(0, n)
        col = w - 1
    elif corner == "bl":
        rows = range(h - n, h)
        col = 0
    else:
        rows = range(h - n, h)
        col = w - 1

    for row, color in zip(rows, colors, strict=False):
        grid[int(row), int(col)] = int(color)
        mask[int(row), int(col)] = True
    return mask


def _nested_chain(
    outer: tuple[int, int, int, int],
    depth: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]] | None:
    boxes = [outer]
    current = outer
    for level in range(1, depth):
        rem = depth - level - 1
        min_child = _min_outer_size(rem + 1)
        r0, c0, r1, c1 = current
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        max_child_h = height - 4
        max_child_w = width - 4
        if max_child_h < min_child or max_child_w < min_child:
            return None

        child_h = int(rng.integers(min_child, max_child_h + 1))
        child_w = int(rng.integers(min_child, max_child_w + 1))

        row_lo = r0 + 2
        row_hi = r1 - child_h - 1
        col_lo = c0 + 2
        col_hi = c1 - child_w - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None

        child_r0 = int(rng.integers(row_lo, row_hi + 1))
        child_c0 = int(rng.integers(col_lo, col_hi + 1))
        current = (
            child_r0,
            child_c0,
            child_r0 + child_h - 1,
            child_c0 + child_w - 1,
        )
        boxes.append(current)
    return boxes


def _sample_outer_box(
    shape: tuple[int, int],
    depth: int,
    blocked: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    min_size = _min_outer_size(depth)
    slack = 6 if bool(blocked.any()) else 8
    max_h = min(h - 4, min_size + slack)
    max_w = min(w - 4, min_size + slack)
    if max_h < min_size or max_w < min_size:
        return None

    for _ in range(CHAIN_ATTEMPTS):
        box_h = int(rng.integers(min_size, max_h + 1))
        box_w = int(rng.integers(min_size, max_w + 1))
        row_hi = h - box_h - 1
        col_hi = w - box_w - 1
        if row_hi <= 2 or col_hi <= 2:
            continue
        r0 = int(rng.integers(2, row_hi))
        c0 = int(rng.integers(2, col_hi))
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1
        if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            continue
        return r0, c0, r1, c1
    return None


def _spread_points(
    cells: list[tuple[int, int]],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if count == 0:
        return []
    pts = [(int(r), int(c)) for r, c in cells]
    first = pts[int(rng.integers(len(pts)))]
    chosen = [first]
    while len(chosen) < count:
        best = None
        best_score = -1
        for idx in rng.permutation(len(pts)):
            pt = pts[int(idx)]
            if pt in chosen:
                continue
            score = min(abs(pt[0] - rr) + abs(pt[1] - cc) for rr, cc in chosen)
            if score > best_score:
                best = pt
                best_score = score
        assert best is not None
        chosen.append(best)
    return chosen


def _choose_marker_counts(
    infos: list[dict[str, object]],
    rng: np.random.Generator,
) -> list[int]:
    counts = []
    for info in infos:
        max_k = min(4, len(info["cells"]))
        counts.append(int(rng.integers(0, max_k + 1)))

    positive = [i for i, info in enumerate(infos) if int(info["depth"]) > 0]
    if not positive:
        return counts

    can_mark = [i for i in positive if min(4, len(infos[i]["cells"])) >= 1]
    can_dense = [i for i in positive if min(4, len(infos[i]["cells"])) >= 2]

    if can_mark and all(counts[i] == 0 for i in positive):
        idx = can_mark[int(rng.integers(len(can_mark)))]
        counts[idx] = int(rng.integers(1, min(4, len(infos[idx]["cells"])) + 1))

    if all(counts[i] > 0 for i in positive):
        idx = positive[int(rng.integers(len(positive)))]
        counts[idx] = 0

    if len({counts[i] for i in positive}) == 1 and can_mark:
        idx = can_mark[int(rng.integers(len(can_mark)))]
        counts[idx] = 0 if counts[idx] != 0 else 1

    if can_dense and max(counts[i] for i in positive) < 2 and float(rng.random()) < 0.6:
        idx = can_dense[int(rng.integers(len(can_dense)))]
        counts[idx] = int(rng.integers(2, min(4, len(infos[idx]["cells"])) + 1))

    return counts


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    max_depth = int(rng.integers(2, 5))
    min_side = max(GRID_MIN, _min_outer_size(max_depth) + 4)
    if min_side > GRID_MAX:
        return None

    h = int(rng.integers(min_side, GRID_MAX + 1))
    w = int(rng.integers(min_side, GRID_MAX + 1))
    grid = np.full((h, w), BLACK, dtype=int)

    chain_cap = 1
    if min(h, w) >= _min_outer_size(max_depth) + 10:
        chain_cap = 2
    if min(h, w) >= 28 and max_depth == 2:
        chain_cap = 3
    chain_count = int(rng.integers(1, chain_cap + 1))

    chain_depths = [max_depth]
    for _ in range(chain_count - 1):
        chain_depths.append(int(rng.integers(1, max_depth + 1)))
    chain_depths.sort(reverse=True)

    legend_colors = [int(v) for v in rng.choice(np.array(FILL_CHOICES), size=max_depth, replace=False).tolist()]
    legend_mask = _paint_legend(grid, legend_colors, str(rng.choice(["tl", "tr", "bl", "br"])))

    blocked = np.zeros((h, w), dtype=bool)
    rectangles: list[tuple[int, int, int, int]] = []
    for depth in chain_depths:
        outer = _sample_outer_box((h, w), depth, blocked, rng)
        if outer is None:
            return None
        chain = _nested_chain(outer, depth, rng)
        if chain is None:
            return None
        for box in chain:
            _paint_box(grid, box, GREEN)
            rectangles.append(box)
        r0, c0, r1, c1 = outer
        blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True

    infos = _region_infos(grid, legend_mask, rectangles)
    positive_depths = {int(info["depth"]) for info in infos if int(info["depth"]) > 0}
    if positive_depths != set(range(1, max_depth + 1)):
        return None

    counts = _choose_marker_counts(infos, rng)
    for info, count in zip(infos, counts, strict=False):
        for r, c in _spread_points(info["cells"], int(count), rng):
            grid[r, c] = ORANGE

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
