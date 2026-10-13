from __future__ import annotations

import random

import numpy as np


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, "every frame color must remain visible"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _side_masks(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox
    top = np.zeros(shape, dtype=bool)
    bottom = np.zeros(shape, dtype=bool)
    left = np.zeros(shape, dtype=bool)
    right = np.zeros(shape, dtype=bool)
    top[r0, c0 : c1 + 1] = True
    bottom[r1, c0 : c1 + 1] = True
    left[r0 : r1 + 1, c0] = True
    right[r0 : r1 + 1, c1] = True
    return [top, bottom, left, right]


def _reachable(adjacency: dict[int, set[int]], start: int) -> set[int]:
    seen: set[int] = set()
    stack = list(adjacency[start])
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency[node] - seen)
    return seen


def _infer_order(
    grid: np.ndarray,
    rect_masks: dict[int, np.ndarray],
) -> list[int]:
    colors = list(rect_masks)
    above: dict[int, set[int]] = {color: set() for color in colors}

    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            overlap = rect_masks[color_a] & rect_masks[color_b]
            if not np.any(overlap):
                continue
            a_visible = bool(np.any(grid[overlap] == color_a))
            b_visible = bool(np.any(grid[overlap] == color_b))
            assert not (a_visible and b_visible), "two frames cannot alternate who is above"
            if a_visible:
                above[color_a].add(color_b)
            elif b_visible:
                above[color_b].add(color_a)

    closure = {color: _reachable(above, color) for color in colors}
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            a_over_b = color_b in closure[color_a]
            b_over_a = color_a in closure[color_b]
            assert a_over_b ^ b_over_a, "overlap evidence must determine a total order"

    bottom_to_top = sorted(colors, key=lambda color: len(closure[color]))
    assert [len(closure[color]) for color in bottom_to_top] == list(range(len(colors)))
    return bottom_to_top


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct overlapping hollow rectangles and encode each frame's hidden border count as a vertical bar.

    Input structure:
    - The grid has black background and 3 or 4 differently colored one-pixel-thick hollow rectangles.
    - Rectangle borders partially overlap, and only the topmost color is visible at each shared pixel.
    - Each color keeps enough visible pixels that its full rectangle bbox can be recovered from min/max rows and cols.

    Rule:
    - For each foreground color, take the visible-pixel bbox and reconstruct the full hollow rectangle border.
    - Compare reconstructed borders pairwise: on any shared border pixel, whichever rectangle's color is visible there is above the other one, then take the transitive closure to get the full bottom-to-top order.
    - Count, for each rectangle, how many pixels of its reconstructed border are covered by higher rectangles in the input.
    - Output a compact bar chart whose columns follow the layer order from bottom to top and whose bar heights equal those hidden-pixel counts.

    Key insight:
    - The visible fragments are only clues: their bbox recovers each full frame, and the same overlap pixels simultaneously reveal both the occlusion order and the number that must be encoded as bar height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(color) for color in np.unique(g) if int(color) != 0]
    assert 3 <= len(colors) <= 4

    bboxes = {color: _bbox_of_color(g, color) for color in colors}
    rect_masks = {color: _border_mask(g.shape, bboxes[color]) for color in colors}
    order = _infer_order(g, rect_masks)

    fg_mask = g != 0
    hidden_counts: dict[int, int] = {}
    for color in colors:
        hidden = rect_masks[color] & fg_mask & (g != color)
        hidden_counts[color] = int(np.count_nonzero(hidden))

    out_h = max(max(hidden_counts.values()), 1)
    out = np.zeros((out_h, len(order)), dtype=int)
    for col, color in enumerate(order):
        height = hidden_counts[color]
        if height > 0:
            out[out_h - height :, col] = color
    return out


def _render_scene(shape: tuple[int, int], rects: list[dict[str, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        bbox = (rect["r0"], rect["c0"], rect["r1"], rect["c1"])
        grid[_border_mask(shape, bbox)] = int(rect["color"])
    return grid


def _hidden_counts(grid: np.ndarray, rects: list[dict[str, int]]) -> dict[int, int]:
    fg_mask = grid != 0
    counts: dict[int, int] = {}
    for rect in rects:
        color = int(rect["color"])
        bbox = (rect["r0"], rect["c0"], rect["r1"], rect["c1"])
        hidden = _border_mask(grid.shape, bbox) & fg_mask & (grid != color)
        counts[color] = int(np.count_nonzero(hidden))
    return counts


def _intended_output(grid: np.ndarray, rects: list[dict[str, int]]) -> np.ndarray:
    counts = _hidden_counts(grid, rects)
    out_h = max(max(counts.values()), 1)
    out = np.zeros((out_h, len(rects)), dtype=int)
    for col, rect in enumerate(rects):
        color = int(rect["color"])
        height = counts[color]
        if height > 0:
            out[out_h - height :, col] = color
    return out


def _has_visible_side(grid: np.ndarray, rect: dict[str, int]) -> bool:
    color = int(rect["color"])
    bbox = (rect["r0"], rect["c0"], rect["r1"], rect["c1"])
    return any(np.all(grid[side] == color) for side in _side_masks(grid.shape, bbox))


def _valid_scene(grid: np.ndarray, rects: list[dict[str, int]]) -> bool:
    counts = _hidden_counts(grid, rects)
    values = [counts[int(rect["color"])] for rect in rects]
    if values[-1] != 0 or len(set(values)) != len(values):
        return False
    if sum(value > 0 for value in values) < 2:
        return False

    for rect in rects:
        color = int(rect["color"])
        bbox = (rect["r0"], rect["c0"], rect["r1"], rect["c1"])
        if _bbox_of_color(grid, color) != bbox:
            return False
        if not _has_visible_side(grid, rect):
            return False

    try:
        return np.array_equal(main(grid), _intended_output(grid, rects))
    except AssertionError:
        return False


def _make_three_rects(for_four: bool) -> tuple[tuple[int, int], list[dict[str, int]]] | None:
    for _ in range(200):
        a_h = random.randint(4, 6)
        a_w = random.randint(7, 8)
        a_r0, a_c0 = 2, 2
        a_r1 = a_r0 + a_h - 1
        a_c1 = a_c0 + a_w - 1

        extra_h = 9 - a_h
        if extra_h < 2:
            continue
        top_gap = random.randint(1, min(2, extra_h - 1))
        bottom_gap = random.randint(1, min(2, extra_h - top_gap))

        b_w_max = min(6, a_w - 2)
        if b_w_max < 4:
            continue
        b_w = random.randint(4, b_w_max)
        b_c0_min = a_c0 + (3 if for_four else 2)
        b_c0_max = a_c1 - b_w
        if b_c0_min > b_c0_max:
            continue
        b_c0 = random.randint(b_c0_min, b_c0_max)
        b_c1 = b_c0 + b_w - 1
        b_r0 = a_r0 - top_gap
        b_r1 = a_r1 + bottom_gap

        c_c0_min = a_c0 + 1
        c_c0_max = b_c0 - (2 if for_four else 1)
        if c_c0_min > c_c0_max:
            continue
        c_c0 = random.randint(c_c0_min, c_c0_max)
        c_r0_max = a_r1 - (2 if for_four else 1)
        if a_r0 + 1 > c_r0_max:
            continue
        c_r0 = random.randint(a_r0 + 1, c_r0_max)

        max_right_gap = 9 - (a_c1 - c_c0 + 1)
        min_right_gap = 2 if for_four else 1
        if max_right_gap < min_right_gap:
            continue
        c_right_gap = random.randint(min_right_gap, min(3, max_right_gap))
        c_c1 = a_c1 + c_right_gap

        max_down_gap = 9 - (b_r1 - c_r0 + 1)
        if max_down_gap < 1:
            continue
        c_down_gap = random.randint(1, min(2, max_down_gap))
        c_r1 = b_r1 + c_down_gap

        max_r = c_r1
        max_c = c_c1
        h_min = max_r + 2
        w_min = max_c + 2
        if h_min > 18 or w_min > 18:
            continue
        h = random.randint(max(12, h_min), 18)
        w = random.randint(max(12, w_min), 18)
        shift_r = random.randint(0, h - 1 - max_r)
        shift_c = random.randint(0, w - 1 - max_c)

        rects = [
            {"r0": a_r0 + shift_r, "c0": a_c0 + shift_c, "r1": a_r1 + shift_r, "c1": a_c1 + shift_c},
            {"r0": b_r0 + shift_r, "c0": b_c0 + shift_c, "r1": b_r1 + shift_r, "c1": b_c1 + shift_c},
            {"r0": c_r0 + shift_r, "c0": c_c0 + shift_c, "r1": c_r1 + shift_r, "c1": c_c1 + shift_c},
        ]
        return (h, w), rects
    return None


def _add_fourth_rect(
    shape: tuple[int, int],
    rects: list[dict[str, int]],
) -> list[dict[str, int]] | None:
    a_rect, b_rect, c_rect = rects
    a_r1, a_c1 = a_rect["r1"], a_rect["c1"]
    b_r0, b_c0, b_r1, b_c1 = b_rect["r0"], b_rect["c0"], b_rect["r1"], b_rect["c1"]
    c_r0, c_c0, c_r1, c_c1 = c_rect["r0"], c_rect["c0"], c_rect["r1"], c_rect["c1"]

    d_c0_min = c_c0 + 1
    d_c0_max = b_c0 - 1
    d_c1_min = a_c1 + 1
    d_c1_max = c_c1 - 1
    d_r0_min = c_r0 + 1
    d_r0_max = a_r1 - 1
    if d_c0_min > d_c0_max or d_c1_min > d_c1_max or d_r0_min > d_r0_max:
        return None

    candidates: list[dict[str, int]] = []
    for _ in range(60):
        d_c0 = random.randint(d_c0_min, d_c0_max)
        d_c1 = random.randint(d_c1_min, d_c1_max)
        if d_c1 - d_c0 + 1 < 4 or d_c1 - d_c0 + 1 > 9:
            continue
        d_r0 = random.randint(d_r0_min, d_r0_max)
        d_r1_min = max(c_r1 + 1, d_r0 + 3)
        d_r1_max = min(shape[0] - 1, d_r0 + 8)
        if d_r1_min > d_r1_max:
            continue
        d_r1 = random.randint(d_r1_min, d_r1_max)
        candidates.append({"r0": d_r0, "c0": d_c0, "r1": d_r1, "c1": d_c1})

    if not candidates:
        return None
    return rects + [random.choice(candidates)]


def generate_input() -> np.ndarray:
    for _ in range(300):
        want_four = random.random() < 0.4
        base = _make_three_rects(for_four=want_four)
        if base is None:
            continue
        shape, rects = base

        if want_four:
            rects4 = _add_fourth_rect(shape, rects)
            if rects4 is not None:
                rects = rects4

        palette = random.sample(range(1, 10), len(rects))
        colored_rects = [
            {**rect, "color": color}
            for rect, color in zip(rects, palette)
        ]
        grid = _render_scene(shape, colored_rects)
        if _valid_scene(grid, colored_rects):
            return grid
    raise RuntimeError("failed to generate valid input")
