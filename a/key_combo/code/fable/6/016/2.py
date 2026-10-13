from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, get_bg


GEN_MAX_ATTEMPTS = 4000
GEN_RECT_COUNT_RANGE = (2, 4)
GEN_GRID_H_RANGE = (10, 18)
GEN_GRID_W_RANGE = (10, 18)
GEN_RECT_H_RANGE = (3, 8)
GEN_RECT_W_RANGE = (3, 8)


def _rect_area(rect: dict) -> int:
    r0, c0, r1, c1 = rect["bbox"]
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _overlap_bbox(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _paint_rect(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    *,
    row_shift: int = 0,
) -> None:
    h, w = map(int, grid.shape)
    r0, c0, r1, c1 = bbox
    r0 += int(row_shift)
    r1 += int(row_shift)
    if c0 >= w or c1 < 0 or r0 >= h or r1 < 0:
        return
    rr0 = max(0, r0)
    rr1 = min(h - 1, r1)
    cc0 = max(0, c0)
    cc1 = min(w - 1, c1)
    grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)


def _render_rects(
    rects: list[dict],
    order: tuple[int, ...],
    shape: tuple[int, int],
    bg: int,
    *,
    shifts: dict[int, int] | None = None,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    shifts = {} if shifts is None else shifts
    for idx in order:
        rect = rects[idx]
        _paint_rect(
            out,
            rect["bbox"],
            rect["color"],
            row_shift=int(shifts.get(idx, 0)),
        )
    return out


def _recover_rects(grid: np.ndarray, bg: int) -> list[dict]:
    rects: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != int(bg)):
        mask = grid == int(color)
        if not np.any(mask):
            continue
        bbox = _bbox_from_mask(mask)
        rects.append(
            {
                "color": int(color),
                "bbox": bbox,
                "visible": int(mask.sum()),
                "area": int((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)),
            }
        )
    return rects


def _find_consistent_orders(rects: list[dict], grid: np.ndarray, bg: int) -> list[tuple[int, ...]]:
    if not rects:
        return [tuple()]
    matches: list[tuple[int, ...]] = []
    for order in itertools.permutations(range(len(rects))):
        if np.array_equal(_render_rects(rects, order, grid.shape, bg), grid):
            matches.append(order)
    return matches


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recover the full rectangles, count each rectangle's hidden pixels, then drop
    every rectangle by that count and redraw them in the original stacking order.
    """
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = int(get_bg(grid))
    rects = _recover_rects(grid, bg)
    if not rects:
        return grid.copy()

    orders = _find_consistent_orders(rects, grid, bg)
    if not orders:
        raise ValueError("could not reconstruct a valid rectangle stacking order")
    order = orders[0]

    shifts = {
        idx: int(rects[idx]["area"] - rects[idx]["visible"])
        for idx in range(len(rects))
    }
    return _render_rects(rects, order, tuple(grid.shape), bg, shifts=shifts)


def _random_rect(h: int, w: int) -> tuple[int, int, int, int]:
    rect_h = random.randint(GEN_RECT_H_RANGE[0], min(GEN_RECT_H_RANGE[1], h - 1))
    rect_w = random.randint(GEN_RECT_W_RANGE[0], min(GEN_RECT_W_RANGE[1], w - 1))
    r0 = random.randint(0, h - rect_h)
    c0 = random.randint(0, w - rect_w)
    return (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)


def _sample_rects(h: int, w: int, count: int) -> list[dict] | None:
    rects: list[dict] = []
    for _ in range(count):
        bbox = None
        for _ in range(120):
            candidate = _random_rect(h, w)
            if not rects:
                bbox = candidate
                break
            if any(_overlap_bbox(candidate, rect["bbox"]) is not None for rect in rects):
                bbox = candidate
                break
        if bbox is None:
            return None
        rects.append({"bbox": bbox})
    return rects


def _edge_visible(grid: np.ndarray, rect: dict) -> bool:
    color = int(rect["color"])
    r0, c0, r1, c1 = rect["bbox"]
    return bool(
        np.any(grid[r0, c0 : c1 + 1] == color)
        and np.any(grid[r1, c0 : c1 + 1] == color)
        and np.any(grid[r0 : r1 + 1, c0] == color)
        and np.any(grid[r0 : r1 + 1, c1] == color)
    )


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    bg_count = int(np.count_nonzero(grid == int(bg)))
    return all(bg_count > int(np.count_nonzero(grid == color)) for color in np.unique(grid) if int(color) != int(bg))


def _build_case(rects: list[dict], shape: tuple[int, int], bg: int) -> tuple[np.ndarray, np.ndarray]:
    order = tuple(range(len(rects)))
    inp = _render_rects(rects, order, shape, bg)
    shifts = {
        idx: int(_rect_area(rects[idx]) - np.count_nonzero(inp == int(rects[idx]["color"])))
        for idx in range(len(rects))
    }
    out = _render_rects(rects, order, shape, bg, shifts=shifts)
    return inp, out


def _valid_case(rects: list[dict], inp: np.ndarray, bg: int) -> bool:
    if not _bg_is_dominant(inp, bg):
        return False

    hidden_counts = []
    for rect in rects:
        visible = int(np.count_nonzero(inp == int(rect["color"])))
        hidden = int(_rect_area(rect) - visible)
        if visible <= 0:
            return False
        if not _edge_visible(inp, rect):
            return False
        hidden_counts.append(hidden)

    if max(hidden_counts, default=0) <= 0:
        return False

    orders = _find_consistent_orders(rects, inp, bg)
    return len(orders) == 1


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_MAX_ATTEMPTS):
        h = random.randint(*GEN_GRID_H_RANGE)
        w = random.randint(*GEN_GRID_W_RANGE)
        count = random.randint(*GEN_RECT_COUNT_RANGE)

        sampled = _sample_rects(h, w, count)
        if sampled is None:
            continue

        bg = random.choice(Color.ALL_COLORS)
        palette = [color for color in Color.ALL_COLORS if color != bg]
        if len(palette) < count:
            continue
        colors = random.sample(palette, count)

        rects: list[dict] = []
        for idx, rect in enumerate(sampled):
            rects.append({"bbox": rect["bbox"], "color": int(colors[idx])})

        inp, out = _build_case(rects, (h, w), bg)
        if _valid_case(rects, inp, bg):
            return inp, out

    raise RuntimeError("failed to generate a valid ARC rectangle-occlusion case")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
