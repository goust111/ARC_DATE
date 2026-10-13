from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GRID_H_RANGE = (10, 22)
GRID_W_RANGE = (12, 24)
RECT_COUNT_RANGE = (2, 5)
GEN_ATTEMPTS = 500
RECT_ATTEMPTS = 120
SIDE_ALIGN_PROB = 0.65
MIN_VISIBLE_FRAC = 0.5
MIN_EDGE_VIS_FRAC = 0.6

_RNG = random.Random()


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _frame_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _layer_order(grid: np.ndarray, bg: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for value in map(int, grid[-1]):
        if value == bg or value in seen:
            continue
        seen.add(value)
        order.append(value)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each rectangle frame from its visible pixels, then redraw the
    complete frames from back to front using the color order encoded in the
    bottom row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 2 and w >= 3

    bg = get_bg(g)
    body = g[:-1, :]
    order = _layer_order(g, bg)
    assert order, "expected a non-empty bottom-row layer legend"

    body_colors = {int(v) for v in np.unique(body) if int(v) != bg}
    assert body_colors == set(order), "body colors and legend colors must match"

    out = np.full_like(g, bg)
    for color in order:
        coords = np.argwhere(body == color)
        assert coords.size > 0, f"missing visible pixels for color {color}"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        assert int(r1 - r0) >= 2 and int(c1 - c0) >= 2
        _draw_frame(out, (int(r0), int(c0), int(r1), int(c1)), int(color))

    return out


def _sample_bbox(
    h: int,
    w: int,
    anchor_r: int,
    anchor_c: int,
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    for _ in range(RECT_ATTEMPTS):
        r0 = _RNG.randint(1, anchor_r - 1)
        r1 = _RNG.randint(anchor_r + 1, h - 3)
        c0 = _RNG.randint(1, anchor_c - 1)
        c1 = _RNG.randint(anchor_c + 1, w - 2)

        if existing and _RNG.random() < SIDE_ALIGN_PROB:
            pr0, pc0, pr1, pc1 = _RNG.choice(existing)
            side = _RNG.choice(("top", "bottom", "left", "right"))
            if side == "top":
                r0 = pr0
            elif side == "bottom":
                r1 = pr1
            elif side == "left":
                c0 = pc0
            else:
                c1 = pc1

        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        bbox = (int(r0), int(c0), int(r1), int(c1))
        if bbox not in existing:
            return bbox
    return None


def _render_input(
    h: int,
    w: int,
    rects_by_color: dict[int, tuple[int, int, int, int]],
    order: list[int],
    legend_cols: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for color in order:
        _draw_frame(grid, rects_by_color[color], color)
    for col, color in zip(legend_cols, order):
        grid[h - 1, col] = color
    return grid


def _edge_sets(bbox: tuple[int, int, int, int]) -> dict[str, set[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    return {
        "top": {(r0, c) for c in range(c0, c1 + 1)},
        "bottom": {(r1, c) for c in range(c0, c1 + 1)},
        "left": {(r, c0) for r in range(r0, r1 + 1)},
        "right": {(r, c1) for r in range(r0, r1 + 1)},
    }


def _visibility_ok(
    body: np.ndarray,
    rects_by_color: dict[int, tuple[int, int, int, int]],
    order: list[int],
) -> bool:
    hidden_total = 0
    occluded_back_layer = False
    top_color = order[-1]

    for color in order:
        bbox = rects_by_color[color]
        visible_coords = np.argwhere(body == color)
        if visible_coords.size == 0:
            return False

        r0, c0 = visible_coords.min(axis=0)
        r1, c1 = visible_coords.max(axis=0)
        if (int(r0), int(c0), int(r1), int(c1)) != bbox:
            return False

        visible = {tuple(map(int, rc)) for rc in visible_coords}
        full = set(_frame_cells(*bbox))
        hidden = full - visible
        hidden_total += len(hidden)
        if hidden and color != top_color:
            occluded_back_layer = True

        if len(visible) < max(4, int(np.ceil(len(full) * MIN_VISIBLE_FRAC))):
            return False

        edges = _edge_sets(bbox)
        edge_vis = {
            name: len(visible & cells) / len(cells)
            for name, cells in edges.items()
        }
        adjacent_pairs = (
            ("top", "left"),
            ("top", "right"),
            ("bottom", "left"),
            ("bottom", "right"),
        )
        if not any(
            edge_vis[a] >= MIN_EDGE_VIS_FRAC and edge_vis[b] >= MIN_EDGE_VIS_FRAC
            for a, b in adjacent_pairs
        ):
            return False

    return hidden_total >= 2 and occluded_back_layer


def _sample_rectangles(h: int, w: int, n_rects: int) -> list[tuple[int, int, int, int]] | None:
    anchor_r = _RNG.randint(2, h - 4)
    anchor_c = _RNG.randint(2, w - 3)
    rects: list[tuple[int, int, int, int]] = []

    for _ in range(n_rects):
        bbox = _sample_bbox(h, w, anchor_r, anchor_c, rects)
        if bbox is None:
            return None
        rects.append(bbox)

    return rects


def _attempt_input() -> np.ndarray | None:
    h = _RNG.randint(*GRID_H_RANGE)
    w = _RNG.randint(*GRID_W_RANGE)
    n_rects = _RNG.randint(*RECT_COUNT_RANGE)
    if n_rects > min(len(PALETTE), w):
        return None

    rects = _sample_rectangles(h, w, n_rects)
    if rects is None:
        return None

    colors = list(_RNG.sample(PALETTE, n_rects))
    order = colors[:]
    rect_indices = list(range(n_rects))
    _RNG.shuffle(rect_indices)
    rects_by_color = {order[i]: rects[rect_indices[i]] for i in range(n_rects)}

    legend_cols = sorted(_RNG.sample(range(w), n_rects))
    grid = _render_input(h, w, rects_by_color, order, legend_cols)

    if get_bg(grid) != BG:
        return None
    if np.any(grid[h - 2, :] != BG):
        return None
    if not _visibility_ok(grid[:-1, :], rects_by_color, order):
        return None

    expected = np.full((h, w), BG, dtype=int)
    for color in order:
        _draw_frame(expected, rects_by_color[color], color)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangle-layer input")
