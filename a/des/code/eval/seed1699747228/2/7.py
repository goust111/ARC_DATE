from __future__ import annotations

import random

import numpy as np


BG = 0
TRIGGER_COLOR = 9
RECT_SHAPES = ((2, 3), (3, 2))
RECT_AREA = 6
PALETTE = tuple(color for color in range(10) if color not in (BG, TRIGGER_COLOR))

GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 30)
RECT_COUNT_RANGE = (4, 7)
MIN_SEPARATION = 2
PLACEMENT_ATTEMPTS = 250
COLOR_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_from_bbox(bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(
        [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)],
        dtype=int,
    )


def _axis_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return int(b0 - a1)
    if b1 < a0:
        return int(a0 - b1)
    return 0


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = _axis_gap(a[0], a[2], b[0], b[2])
    col_gap = _axis_gap(a[1], a[3], b[1], b[3])
    return int(max(row_gap, col_gap))


def _extract_rectangles(
    grid: np.ndarray,
) -> tuple[int, int, tuple[int, int, int, int], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    trigger_color = int(g[h - 1, 0])
    assert trigger_color != bg, "bottom-left trigger must be non-background"

    trigger_cells = np.argwhere(g == trigger_color)
    assert trigger_cells.shape[0] == 1, "trigger color must appear exactly once"
    assert tuple(map(int, trigger_cells[0])) == (h - 1, 0), "trigger must be at the bottom-left corner"

    mask = (g != bg) & (g != trigger_color)
    rects: list[dict[str, object]] = []
    for coords in _components(mask):
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        assert coords.shape[0] == RECT_AREA, "foreground objects must have area 6"
        assert coords.shape[0] == rect_h * rect_w, "foreground objects must be solid rectangles"
        assert (rect_h, rect_w) in RECT_SHAPES, "foreground rectangles must be 2x3 or 3x2"
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "each rectangle must be monochromatic"
        rects.append(
            {
                "bbox": bbox,
                "coords": coords,
                "color": int(colors[0]),
            }
        )

    assert rects, "expected at least one rectangle"
    return bg, trigger_color, (h - 1, 0, h - 1, 0), rects


def _greedy_chain(
    rects: list[dict[str, object]],
    start_bbox: tuple[int, int, int, int],
) -> list[int] | None:
    unvisited = set(range(len(rects)))
    current = start_bbox
    chain: list[int] = []

    while unvisited:
        ranked = sorted((_bbox_distance(current, rects[i]["bbox"]), i) for i in unvisited)
        if len(ranked) > 1 and ranked[1][0] == ranked[0][0]:
            return None
        nxt = ranked[0][1]
        chain.append(nxt)
        current = rects[nxt]["bbox"]
        unvisited.remove(nxt)

    return chain


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the unique nearest-neighbor chain starting from the bottom-left trigger
    pixel, then shift each rectangle's original color forward by one step along
    that chain while keeping the first rectangle and the trigger unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _bg, trigger_color, trigger_bbox, rects = _extract_rectangles(g)
    chain = _greedy_chain(rects, trigger_bbox)
    assert chain is not None, "nearest rectangle must be unique at every step"

    out = g.copy()
    for src_idx, dst_idx in zip(chain, chain[1:]):
        dst = rects[dst_idx]["coords"]
        out[dst[:, 0], dst[:, 1]] = int(rects[src_idx]["color"])

    out[g == trigger_color] = trigger_color
    return out


def _rects_from_bboxes(bboxes: list[tuple[int, int, int, int]]) -> list[dict[str, object]]:
    return [
        {
            "bbox": bbox,
            "coords": _coords_from_bbox(bbox),
            "color": 1,
        }
        for bbox in bboxes
    ]


def _can_place(
    bbox: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    trigger_bbox: tuple[int, int, int, int],
) -> bool:
    if _bbox_distance(bbox, trigger_bbox) < MIN_SEPARATION:
        return False
    return all(_bbox_distance(bbox, other) >= MIN_SEPARATION for other in placed)


def _sample_layout() -> tuple[int, int, list[tuple[int, int, int, int]], list[int]] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    rect_count = random.randint(*RECT_COUNT_RANGE)
    trigger_bbox = (h - 1, 0, h - 1, 0)

    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(rect_count):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            rect_h, rect_w = random.choice(RECT_SHAPES)
            r0 = random.randint(0, h - rect_h)
            c0 = random.randint(0, w - rect_w)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if _can_place(bbox, bboxes, trigger_bbox):
                bboxes.append(bbox)
                placed = True
                break
        if not placed:
            return None

    rects = _rects_from_bboxes(bboxes)
    chain = _greedy_chain(rects, trigger_bbox)
    if chain is None or len(chain) != rect_count:
        return None

    return h, w, bboxes, chain


def _sample_colors(rect_count: int, chain: list[int]) -> list[int] | None:
    max_distinct = min(5, rect_count, len(PALETTE))
    min_distinct = min(2, max_distinct)
    for _ in range(COLOR_ATTEMPTS):
        distinct = random.randint(min_distinct, max_distinct)
        palette = random.sample(PALETTE, distinct)
        colors = [random.choice(palette) for _ in range(rect_count)]
        if len(set(colors)) < 2:
            continue
        if any(colors[chain[i - 1]] != colors[chain[i]] for i in range(1, rect_count)):
            return colors
    return None


def _build_grid(
    h: int,
    w: int,
    bboxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[h - 1, 0] = TRIGGER_COLOR
    for bbox, color in zip(bboxes, colors):
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _is_valid_instance(grid: np.ndarray) -> bool:
    try:
        _bg, _trigger_color, trigger_bbox, rects = _extract_rectangles(grid)
        chain = _greedy_chain(rects, trigger_bbox)
        if chain is None:
            return False
        colors = [int(rect["color"]) for rect in rects]
        if len(set(colors)) < 2:
            return False
        if not any(colors[chain[i - 1]] != colors[chain[i]] for i in range(1, len(chain))):
            return False
        out = main(grid)
        return isinstance(out, np.ndarray) and out.shape == grid.shape
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None

    h, w, bboxes, chain = layout
    colors = _sample_colors(len(bboxes), chain)
    if colors is None:
        return None

    grid = _build_grid(h, w, bboxes, colors)
    if not _is_valid_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
