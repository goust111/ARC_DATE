import random
from typing import Dict, Iterable, List, Tuple

import numpy as np


BG = 0
GREY = 5
CARD_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_CARDS = 2
MAX_CARDS = 5
SCENE_H_RANGE = (10, 22)
SCENE_W_RANGE = (12, 26)
TOP_CARD_H_RANGE = (2, 6)
TOP_CARD_W_RANGE = (2, 8)
CARD_H_MAX = 10
CARD_W_MAX = 11
GAP_RANGE = (1, 3)
GEN_RETRIES = 500

Rect = Tuple[int, int, int, int]


def _bbox(coords: np.ndarray) -> Rect:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_dims(rect: Rect) -> Tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _intersection_area(a: Rect, b: Rect) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _union_bbox(rects: Iterable[Rect]) -> Rect:
    rects = list(rects)
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def _render_scene(scene_h: int, scene_w: int, rects: Dict[int, Rect], order: List[int]) -> np.ndarray:
    scene = np.full((scene_h, scene_w), BG, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rects[color]
        scene[r0 : r1 + 1, c0 : c1 + 1] = color
    return scene


def _assemble_output(order: List[int], rects: Dict[int, Rect]) -> np.ndarray:
    dims = {color: _rect_dims(rects[color]) for color in order}
    out_h = max(h for h, _ in dims.values())
    out_w = sum(w for _, w in dims.values()) + len(order) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for idx, color in enumerate(order):
        h, w = dims[color]
        top = (out_h - h) // 2
        out[top : top + h, col : col + w] = color
        col += w
        if idx != len(order) - 1:
            out[:, col] = GREY
            col += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate overlapped colored rectangles using the bottom color-order legend.

    Input structure:
    - Black background.
    - Several solid colored rectangles partially occlude one another.
    - The last non-black row is a one-pixel-per-card color legend ordered
      from bottom layer to top layer, with at least one black row above it.

    Rule:
    - Read the legend row left to right to recover layer order.
    - For each legend color, take the visible pixels of that color above the
      gap and recover the full rectangle from their row/column extremes.
    - Output the full rectangles in bottom-to-top order, separated by one
      grey column and vertically centered inside the tallest card height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    nonempty_rows = np.flatnonzero(np.any(g != BG, axis=1))
    assert nonempty_rows.size >= 2, "expected card area plus bottom legend row"

    legend_row = int(nonempty_rows[-1])
    card_rows = nonempty_rows[nonempty_rows < legend_row]
    assert card_rows.size > 0 and int(card_rows[-1]) < legend_row - 1, "expected a black gap above legend"

    order = [int(v) for v in g[legend_row] if int(v) != BG]
    assert order and len(order) == len(set(order)), "legend colors must be unique"

    scene = g[: int(card_rows[-1]) + 1]
    rects: Dict[int, Rect] = {}
    for color in order:
        coords = np.argwhere(scene == color)
        assert coords.size > 0, f"missing visible pixels for color {color}"
        rects[color] = _bbox(coords)

    return _assemble_output(order, rects)


def _choose_scene_size(n_cards: int, rng: random.Random) -> Tuple[int, int]:
    min_h = max(SCENE_H_RANGE[0], 2 * n_cards + 6)
    min_w = max(SCENE_W_RANGE[0], 2 * n_cards + 8)
    scene_h = rng.randint(min_h, SCENE_H_RANGE[1])
    scene_w = rng.randint(min_w, SCENE_W_RANGE[1])
    return scene_h, scene_w


def _sample_top_rect(scene_h: int, scene_w: int, n_cards: int, rng: random.Random) -> Rect | None:
    margin = max(1, n_cards - 1)
    max_h = min(TOP_CARD_H_RANGE[1], scene_h - 2 * margin)
    max_w = min(TOP_CARD_W_RANGE[1], scene_w - 2 * margin)
    if max_h < TOP_CARD_H_RANGE[0] or max_w < TOP_CARD_W_RANGE[0]:
        return None

    h = rng.randint(TOP_CARD_H_RANGE[0], max_h)
    w = rng.randint(TOP_CARD_W_RANGE[0], max_w)
    r0 = rng.randint(margin, scene_h - h - margin)
    c0 = rng.randint(margin, scene_w - w - margin)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _sample_lower_rect(scene_h: int, scene_w: int, upper_rects: List[Rect], rng: random.Random) -> Rect | None:
    ur0, uc0, ur1, uc1 = _union_bbox(upper_rects)
    corners = []
    if ur0 >= 1 and uc0 >= 1:
        corners.append("TL")
    if ur0 >= 1 and uc1 + 1 < scene_w:
        corners.append("TR")
    if ur1 + 1 < scene_h and uc0 >= 1:
        corners.append("BL")
    if ur1 + 1 < scene_h and uc1 + 1 < scene_w:
        corners.append("BR")
    rng.shuffle(corners)

    for corner in corners:
        for _ in range(80):
            if corner == "TL":
                r0 = rng.randint(0, ur0 - 1)
                c0 = rng.randint(0, uc0 - 1)
                min_h = max(2, ur0 - r0 + 1)
                min_w = max(2, uc0 - c0 + 1)
                max_h = min(CARD_H_MAX, scene_h - r0)
                max_w = min(CARD_W_MAX, scene_w - c0)
                if min_h > max_h or min_w > max_w:
                    continue
                h = rng.randint(min_h, max_h)
                w = rng.randint(min_w, max_w)
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
            elif corner == "TR":
                r0 = rng.randint(0, ur0 - 1)
                c1 = rng.randint(uc1 + 1, scene_w - 1)
                min_h = max(2, ur0 - r0 + 1)
                min_w = max(2, c1 - uc1 + 1)
                max_h = min(CARD_H_MAX, scene_h - r0)
                max_w = min(CARD_W_MAX, c1 + 1)
                if min_h > max_h or min_w > max_w:
                    continue
                h = rng.randint(min_h, max_h)
                w = rng.randint(min_w, max_w)
                rect = (r0, c1 - w + 1, r0 + h - 1, c1)
            elif corner == "BL":
                r1 = rng.randint(ur1 + 1, scene_h - 1)
                c0 = rng.randint(0, uc0 - 1)
                min_h = max(2, r1 - ur1 + 1)
                min_w = max(2, uc0 - c0 + 1)
                max_h = min(CARD_H_MAX, r1 + 1)
                max_w = min(CARD_W_MAX, scene_w - c0)
                if min_h > max_h or min_w > max_w:
                    continue
                h = rng.randint(min_h, max_h)
                w = rng.randint(min_w, max_w)
                rect = (r1 - h + 1, c0, r1, c0 + w - 1)
            else:
                r1 = rng.randint(ur1 + 1, scene_h - 1)
                c1 = rng.randint(uc1 + 1, scene_w - 1)
                min_h = max(2, r1 - ur1 + 1)
                min_w = max(2, c1 - uc1 + 1)
                max_h = min(CARD_H_MAX, r1 + 1)
                max_w = min(CARD_W_MAX, c1 + 1)
                if min_h > max_h or min_w > max_w:
                    continue
                h = rng.randint(min_h, max_h)
                w = rng.randint(min_w, max_w)
                rect = (r1 - h + 1, c1 - w + 1, r1, c1)

            if any(_intersection_area(rect, upper) > 0 for upper in upper_rects):
                return rect
    return None


def _is_valid_instance(grid: np.ndarray, rects: Dict[int, Rect], order: List[int]) -> bool:
    scene_rows = np.flatnonzero(np.any(grid[:-1] != BG, axis=1))
    if scene_rows.size == 0:
        return False
    scene = grid[: int(scene_rows[-1]) + 1]

    has_overlap = False
    for i, color in enumerate(order):
        rect = rects[color]
        if i < len(order) - 1 and any(_intersection_area(rect, rects[other]) > 0 for other in order[i + 1 :]):
            has_overlap = True
        coords = np.argwhere(scene == color)
        if coords.size == 0:
            return False
        if _bbox(coords) != rect:
            return False

    if not has_overlap:
        return False

    expected = _assemble_output(order, rects)
    produced = main(grid)
    return produced.shape == expected.shape and np.array_equal(produced, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_cards = rng.randint(MIN_CARDS, MAX_CARDS)
    scene_h, scene_w = _choose_scene_size(n_cards, rng)
    order = rng.sample(CARD_COLORS, n_cards)

    top_rect = _sample_top_rect(scene_h, scene_w, n_cards, rng)
    if top_rect is None:
        return None

    rects: Dict[int, Rect] = {order[-1]: top_rect}
    upper_rects = [top_rect]
    for color in reversed(order[:-1]):
        rect = _sample_lower_rect(scene_h, scene_w, upper_rects, rng)
        if rect is None:
            return None
        rects[color] = rect
        upper_rects.append(rect)

    scene = _render_scene(scene_h, scene_w, rects, order)
    gap = rng.randint(GAP_RANGE[0], GAP_RANGE[1])
    grid = np.full((scene_h + gap + 1, scene_w), BG, dtype=int)
    grid[:scene_h] = scene

    start = rng.randint(0, scene_w - n_cards)
    grid[-1, start : start + n_cards] = np.array(order, dtype=int)

    if _is_valid_instance(grid, rects, order):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid card-overlap instance")
