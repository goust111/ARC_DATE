import random
from collections import Counter

import numpy as np


Rect = tuple[int, int, int, int]

MIN_GRID = 14
MAX_GRID = 30
MIN_RECTS = 2
MAX_RECTS = 5
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_area(bbox: Rect) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _bbox_of_mask(mask: np.ndarray) -> Rect | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_size(rect: Rect) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _intersect(a: Rect, b: Rect) -> Rect | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _draw_filled_rect(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _draw_frame(grid: np.ndarray, bbox: Rect, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, Rect]:
    candidates: list[tuple[int, int, Rect]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        mask = grid == color
        bbox = _bbox_of_mask(mask)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(window[0, :] == color):
            continue
        if not np.all(window[-1, :] == color):
            continue
        if not np.all(window[:, 0] == color):
            continue
        if not np.all(window[:, -1] == color):
            continue
        if np.all(window == color):
            continue
        candidates.append((_bbox_area(bbox), color, bbox))
    if not candidates:
        raise ValueError("frame not found")
    candidates.sort(reverse=True)
    _, frame_color, frame_bbox = candidates[0]
    return int(frame_color), frame_bbox


def _recover_rectangles(grid: np.ndarray) -> tuple[int, int, Rect, dict[int, Rect]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox
    inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    rects: dict[int, Rect] = {}
    for color in map(int, np.unique(inner)):
        if color in {bg, frame_color}:
            continue
        mask = np.zeros_like(g, dtype=bool)
        mask[fr0 + 1 : fr1, fc0 + 1 : fc1] = inner == color
        bbox = _bbox_of_mask(mask)
        if bbox is not None:
            rects[int(color)] = bbox
    if not rects:
        raise ValueError("no colored rectangles found")
    return int(bg), int(frame_color), frame_bbox, rects


def _recover_order(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    _bg, _frame_color, _frame_bbox, rects = _recover_rectangles(g)
    colors = list(rects)

    edges = {color: set() for color in colors}
    incoming = {color: set() for color in colors}

    for i, lower in enumerate(colors):
        for upper in colors[i + 1 :]:
            overlap = _intersect(rects[lower], rects[upper])
            if overlap is None:
                continue
            r0, c0, r1, c1 = overlap
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            sees_lower = bool(np.any(patch == lower))
            sees_upper = bool(np.any(patch == upper))
            if sees_lower and sees_upper:
                raise ValueError("ambiguous overlap relation")
            if sees_lower:
                edges[upper].add(lower)
                incoming[lower].add(upper)
            elif sees_upper:
                edges[lower].add(upper)
                incoming[upper].add(lower)

    ready = sorted(color for color in colors if not incoming[color])
    order: list[int] = []
    while ready:
        if len(ready) != 1:
            raise ValueError("overlap order is not uniquely determined")
        color = ready.pop()
        order.append(int(color))
        for nxt in list(edges[color]):
            incoming[nxt].remove(color)
            if not incoming[nxt]:
                ready.append(int(nxt))
        ready.sort()

    if len(order) != len(colors):
        raise ValueError("cycle in overlap order")
    return order


def _rings_from_order(order: list[int]) -> np.ndarray:
    side = 2 * len(order) - 1
    out = np.zeros((side, side), dtype=int)
    for depth, color in enumerate(order):
        r0 = c0 = int(depth)
        r1 = c1 = int(side - 1 - depth)
        out[r0, c0 : c1 + 1] = int(color)
        out[r1, c0 : c1 + 1] = int(color)
        out[r0 : r1 + 1, c0] = int(color)
        out[r0 : r1 + 1, c1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct layered rectangles from visible color fragments and encode the
    bottom-to-top order as nested one-pixel square rings.
    """
    order = _recover_order(np.asarray(grid, dtype=int))
    return _rings_from_order(order)


def _local_nested_rects(rng: random.Random, count: int) -> list[Rect] | None:
    min_outer_h = 3 + 2 * (count - 1)
    min_outer_w = 3 + 2 * (count - 1)
    outer_h = rng.randint(min_outer_h, min(min_outer_h + 6, 14))
    outer_w = rng.randint(min_outer_w, min(min_outer_w + 6, 14))
    rects = [(0, 0, outer_h - 1, outer_w - 1)]
    for idx in range(1, count):
        prev = rects[-1]
        prev_h, prev_w = _rect_size(prev)
        remaining = count - 1 - idx
        min_h = 3 + 2 * remaining
        min_w = 3 + 2 * remaining
        slack_h = prev_h - min_h
        slack_w = prev_w - min_w
        if slack_h < 2 or slack_w < 2:
            return None
        top_inset = rng.randint(1, slack_h - 1)
        bottom_inset = rng.randint(1, slack_h - top_inset)
        left_inset = rng.randint(1, slack_w - 1)
        right_inset = rng.randint(1, slack_w - left_inset)
        rects.append(
            (
                int(prev[0] + top_inset),
                int(prev[1] + left_inset),
                int(prev[2] - bottom_inset),
                int(prev[3] - right_inset),
            )
        )
    return rects


def _local_chain_rects(rng: random.Random, count: int) -> list[Rect] | None:
    rects: list[Rect] = []
    height = rng.randint(4, 8)
    width = rng.randint(4, 8)
    rects.append((0, 0, height - 1, width - 1))
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for _ in range(1, count):
        prev = rects[-1]
        prev_h, prev_w = _rect_size(prev)
        placed = False
        for _attempt in range(120):
            next_h = max(3, min(8, prev_h + rng.randint(-1, 1)))
            next_w = max(3, min(8, prev_w + rng.randint(-1, 1)))
            dr_sign, dc_sign = directions[rng.randrange(len(directions))]
            max_dr = min(prev_h - 1, next_h - 1)
            max_dc = min(prev_w - 1, next_w - 1)
            if max_dr < 1 or max_dc < 1:
                continue
            dr = dr_sign * rng.randint(1, max_dr)
            dc = dc_sign * rng.randint(1, max_dc)
            candidate = (
                int(prev[0] + dr),
                int(prev[1] + dc),
                int(prev[0] + dr + next_h - 1),
                int(prev[1] + dc + next_w - 1),
            )
            overlap = _intersect(prev, candidate)
            if overlap is None:
                continue
            ov_h, ov_w = _rect_size(overlap)
            if ov_h * ov_w < 2:
                continue
            if any(_intersect(candidate, older) is not None for older in rects[:-1]):
                continue
            rects.append(candidate)
            placed = True
            break
        if not placed:
            return None
    return rects


def _translate_rects(rects: list[Rect], dr: int, dc: int) -> list[Rect]:
    return [
        (int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc))
        for r0, c0, r1, c1 in rects
    ]


def _compose_scene(
    bg: int, frame_color: int, frame_bbox: Rect, rects: list[Rect], rect_colors: list[int]
) -> np.ndarray:
    h = int(frame_bbox[2] + 2)
    w = int(frame_bbox[3] + 2)
    grid = np.full((h, w), int(bg), dtype=int)
    _draw_frame(grid, frame_bbox, frame_color)
    for rect, color in zip(rects, rect_colors):
        _draw_filled_rect(grid, rect, color)
    return grid


def _counts_have_unique_bg(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flat))
    bg_count = counts.pop(int(bg), 0)
    return all(bg_count > count for count in counts.values())


def _rect_bboxes_match_visible(grid: np.ndarray, rects: list[Rect], rect_colors: list[int]) -> bool:
    for rect, color in zip(rects, rect_colors):
        visible_bbox = _bbox_of_mask(grid == color)
        if visible_bbox != rect:
            return False
    return True


def _make_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(MIN_RECTS, MAX_RECTS)
    mode = rng.choice(["nested", "chain"])
    if mode == "nested":
        local_rects = _local_nested_rects(rng, count)
    else:
        local_rects = _local_chain_rects(rng, count)
    if local_rects is None:
        return None

    min_r = min(rect[0] for rect in local_rects)
    min_c = min(rect[1] for rect in local_rects)
    max_r = max(rect[2] for rect in local_rects)
    max_c = max(rect[3] for rect in local_rects)
    span_h = int(max_r - min_r + 1)
    span_w = int(max_c - min_c + 1)

    inner_pad_top = rng.randint(1, 3)
    inner_pad_bottom = rng.randint(1, 3)
    inner_pad_left = rng.randint(1, 3)
    inner_pad_right = rng.randint(1, 3)
    frame_h = span_h + inner_pad_top + inner_pad_bottom + 2
    frame_w = span_w + inner_pad_left + inner_pad_right + 2
    if frame_h + 2 > MAX_GRID or frame_w + 2 > MAX_GRID:
        return None

    h_min = max(MIN_GRID, frame_h + 2)
    w_min = max(MIN_GRID, frame_w + 2)
    h_max = min(MAX_GRID, frame_h + 10)
    w_max = min(MAX_GRID, frame_w + 10)
    if h_min > h_max or w_min > w_max:
        return None
    grid_h = rng.randint(h_min, h_max)
    grid_w = rng.randint(w_min, w_max)

    frame_r0 = rng.randint(1, grid_h - frame_h - 1)
    frame_c0 = rng.randint(1, grid_w - frame_w - 1)
    frame_bbox = (
        int(frame_r0),
        int(frame_c0),
        int(frame_r0 + frame_h - 1),
        int(frame_c0 + frame_w - 1),
    )

    shift_r = frame_r0 + 1 + inner_pad_top - min_r
    shift_c = frame_c0 + 1 + inner_pad_left - min_c
    rects = _translate_rects(local_rects, shift_r, shift_c)

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    rng.shuffle(palette)
    frame_color = palette[0]
    rect_colors = palette[1 : 1 + count]

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    _draw_frame(grid, frame_bbox, frame_color)
    for rect, color in zip(rects, rect_colors):
        _draw_filled_rect(grid, rect, color)

    if not _counts_have_unique_bg(grid, bg):
        return None
    if not _rect_bboxes_match_visible(grid, rects, rect_colors):
        return None

    expected = _rings_from_order(rect_colors)
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
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid scene")
