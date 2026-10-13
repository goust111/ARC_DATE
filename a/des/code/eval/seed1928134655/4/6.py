from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 9
ARC_MAX = 30
GEN_RETRIES = 500
CORNER_MODES = ("TL", "TR", "BL", "BR")
PALETTE = tuple(c for c in range(10) if c not in (BG, SEP))
OBJECT_COUNT_RANGE = (2, 4)
TOP_SIZE_RANGE = (2, 4)
EXTRA_MAX = 3


def _segments(line: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(line)
    if idx.size == 0:
        return []
    segs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for value in idx[1:]:
        cur = int(value)
        if cur == prev + 1:
            prev = cur
            continue
        segs.append((start, prev))
        start = prev = cur
    segs.append((start, prev))
    return segs


def _find_layout(grid: np.ndarray) -> tuple[int, int, list[int], int]:
    height, width = map(int, grid.shape)
    for sep_row in range(1, height - 1):
        sep_color = int(grid[sep_row, 0])
        if sep_color == BG or not np.all(grid[sep_row] == sep_color):
            continue

        lower = grid[sep_row + 1 :]
        if lower.size == 0:
            continue
        sep_cols = [c for c in range(width) if np.all(lower[:, c] == sep_color)]
        if len(sep_cols) < 2:
            continue

        slot_widths = [b - a - 1 for a, b in zip(sep_cols, sep_cols[1:])]
        if min(slot_widths) <= 0 or any(w != slot_widths[0] for w in slot_widths):
            continue

        slot_height = height - sep_row - 1
        if slot_height != sep_row:
            continue
        return sep_row, sep_color, sep_cols, slot_widths[0]

    raise AssertionError("failed to parse separator row and slot layout")


def _infer_bbox(scene: np.ndarray, color: int) -> tuple[int, int, int, int]:
    mask = scene == int(color)
    assert bool(mask.any()), "expected visible pixels for each object color"

    row_candidates: list[tuple[int, int]] = []
    max_width = 0
    for r in np.flatnonzero(mask.any(axis=1)):
        for c0, c1 in _segments(mask[int(r)]):
            width = c1 - c0 + 1
            if width > max_width:
                max_width = width
                row_candidates = [(c0, c1)]
            elif width == max_width:
                row_candidates.append((c0, c1))
    row_bounds = set(row_candidates)
    assert len(row_bounds) == 1, "rectangle width must be uniquely inferable"
    left, right = next(iter(row_bounds))

    col_candidates: list[tuple[int, int]] = []
    max_height = 0
    for c in np.flatnonzero(mask.any(axis=0)):
        for r0, r1 in _segments(mask[:, int(c)]):
            height = r1 - r0 + 1
            if height > max_height:
                max_height = height
                col_candidates = [(r0, r1)]
            elif height == max_height:
                col_candidates.append((r0, r1))
    col_bounds = set(col_candidates)
    assert len(col_bounds) == 1, "rectangle height must be uniquely inferable"
    top, bottom = next(iter(col_bounds))

    assert np.any(mask[top : bottom + 1, left]), "expected a full visible vertical edge"
    assert np.any(mask[top, left : right + 1]), "expected a full visible horizontal edge"
    assert np.all(mask[np.where(mask)]), "mask integrity check failed"
    return int(top), int(left), int(bottom), int(right)


def _intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    top = max(a[0], b[0])
    left = max(a[1], b[1])
    bottom = min(a[2], b[2])
    right = min(a[3], b[3])
    if top > bottom or left > right:
        return None
    return top, left, bottom, right


def _depth_order(
    scene: np.ndarray, colors: list[int], bboxes: dict[int, tuple[int, int, int, int]]
) -> list[int]:
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}

    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            overlap = _intersect(bboxes[color_a], bboxes[color_b])
            if overlap is None:
                continue
            r0, c0, r1, c1 = overlap
            window = scene[r0 : r1 + 1, c0 : c1 + 1]
            sees_a = bool(np.any(window == color_a))
            sees_b = bool(np.any(window == color_b))
            if sees_a and sees_b:
                raise AssertionError("pairwise overlap relation is ambiguous")
            if not sees_a and not sees_b:
                continue
            top = color_a if sees_a else color_b
            bottom = color_b if sees_a else color_a
            if bottom not in above[top]:
                above[top].add(bottom)
                indegree[bottom] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert len(ready) == 1, "depth ordering must be unique"
        color = ready[0]
        remaining.remove(color)
        order.append(color)
        for lower in above[color]:
            indegree[lower] -= 1

    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the occluded rectangles, sort them by depth, and draw one full object per slot.

    Input structure
    - The upper area is a black scene with several overlapping monochrome rectangles.
    - The lower area contains blank black slots separated by a maroon scaffold.
    - Each slot has the same size as the upper scene.

    Rule
    - Reconstruct each rectangle's full bbox from its visible full-width/full-height evidence.
    - Infer pairwise above/below relations from visible colors inside reconstructed overlap areas.
    - Topologically order the objects from shallowest to deepest.
    - Fill the bottom slots from left to right with the recovered full rectangles.

    Key insight
    - The visible fragments are enough because every rectangle keeps at least one full row and
      one full column, while overlap windows reveal which color sits on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    scene_h, sep_color, sep_cols, scene_w = _find_layout(g)
    slot_count = len(sep_cols) - 1
    scene_left = sep_cols[0] + 1
    scene_right = scene_left + scene_w
    scene = g[:scene_h, scene_left:scene_right]

    top_area = g[:scene_h]
    colored = (top_area != BG) & (top_area != sep_color)
    assert not colored[:, :scene_left].any()
    assert not colored[:, scene_right:].any()

    colors = [int(c) for c in np.unique(scene) if int(c) not in (BG, sep_color)]
    assert len(colors) == slot_count, "slot count must match object count"

    bboxes = {color: _infer_bbox(scene, color) for color in colors}
    order = _depth_order(scene, colors, bboxes)
    assert len(order) == slot_count

    out = g.copy()
    slot_top = scene_h + 1
    for idx, color in enumerate(order):
        left_sep = sep_cols[idx]
        right_sep = sep_cols[idx + 1]
        out[slot_top : slot_top + scene_h, left_sep + 1 : right_sep] = BG
        r0, c0, r1, c1 = bboxes[color]
        out[slot_top + r0 : slot_top + r1 + 1, left_sep + 1 + c0 : left_sep + 1 + c1 + 1] = color

    return out


def _sample_side_pair(min_a: int, min_b: int, max_total: int) -> tuple[int, int] | None:
    options = [
        (a, b)
        for a in range(min_a, EXTRA_MAX + 1)
        for b in range(min_b, EXTRA_MAX + 1)
        if a + b <= max_total
    ]
    if not options:
        return None
    return random.choice(options)


def _grow_parent(
    child: tuple[int, int, int, int], mode: str, max_h: int, max_w: int, future_steps: int
) -> tuple[int, int, int, int] | None:
    child_h = child[2] - child[0] + 1
    child_w = child[3] - child[1] + 1
    max_h_now = max_h - future_steps
    max_w_now = max_w - future_steps
    if max_h_now <= child_h or max_w_now <= child_w:
        return None

    top_min = 1 if "T" in mode else 0
    bottom_min = 1 if "B" in mode else 0
    left_min = 1 if "L" in mode else 0
    right_min = 1 if "R" in mode else 0

    vertical = _sample_side_pair(top_min, bottom_min, max_h_now - child_h)
    horizontal = _sample_side_pair(left_min, right_min, max_w_now - child_w)
    if vertical is None or horizontal is None:
        return None
    extra_top, extra_bottom = vertical
    extra_left, extra_right = horizontal

    return (
        child[0] - extra_top,
        child[1] - extra_left,
        child[2] + extra_bottom,
        child[3] + extra_right,
    )


def _sample_rect_chain(count: int, max_h: int, max_w: int) -> tuple[list[tuple[int, int, int, int]], tuple[str, ...]] | None:
    min_top_h = TOP_SIZE_RANGE[0]
    min_top_w = TOP_SIZE_RANGE[0]
    max_top_h = min(TOP_SIZE_RANGE[1], max_h - (count - 1))
    max_top_w = min(TOP_SIZE_RANGE[1], max_w - (count - 1))
    if max_top_h < min_top_h or max_top_w < min_top_w:
        return None

    top_h = random.randint(min_top_h, max_top_h)
    top_w = random.randint(min_top_w, max_top_w)
    rects = [(0, 0, top_h - 1, top_w - 1)]  # shallowest -> deepest
    modes: list[str] = []

    for depth in range(1, count):
        child = rects[-1]
        future_steps = count - depth - 1
        random_modes = random.sample(CORNER_MODES, len(CORNER_MODES))
        parent = None
        picked_mode = None
        for mode in random_modes:
            parent = _grow_parent(child, mode, max_h, max_w, future_steps)
            if parent is not None:
                picked_mode = mode
                break
        if parent is None or picked_mode is None:
            return None
        rects.append(parent)
        modes.append(picked_mode)

    min_row = min(r0 for r0, _c0, _r1, _c1 in rects)
    min_col = min(c0 for _r0, c0, _r1, _c1 in rects)
    rects = [
        (r0 - min_row, c0 - min_col, r1 - min_row, c1 - min_col)
        for r0, c0, r1, c1 in rects
    ]
    return rects, tuple(modes)


def _shift_chain(
    rects: list[tuple[int, int, int, int]], scene_h: int, scene_w: int
) -> list[tuple[int, int, int, int]]:
    outer = rects[-1]
    outer_h = outer[2] - outer[0] + 1
    outer_w = outer[3] - outer[1] + 1
    dr = random.randint(0, scene_h - outer_h)
    dc = random.randint(0, scene_w - outer_w)
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in rects]


def _render_scene(
    scene_h: int, scene_w: int, rects: list[tuple[int, int, int, int]], colors: list[int]
) -> np.ndarray:
    scene = np.full((scene_h, scene_w), BG, dtype=int)
    for (r0, c0, r1, c1), color in reversed(list(zip(rects, colors))):
        scene[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return scene


def _build_input(scene: np.ndarray, count: int) -> np.ndarray:
    scene_h, scene_w = map(int, scene.shape)
    width = count * scene_w + count + 1
    grid = np.full((2 * scene_h + 1, width), BG, dtype=int)
    grid[scene_h, :] = SEP
    for idx in range(count + 1):
        col = idx * (scene_w + 1)
        grid[scene_h + 1 :, col] = SEP
    grid[:scene_h, 1 : 1 + scene_w] = scene
    return grid


def _build_expected_output(
    inp: np.ndarray, rects: list[tuple[int, int, int, int]], colors: list[int], scene_h: int, scene_w: int
) -> np.ndarray:
    out = inp.copy()
    slot_top = scene_h + 1
    for idx, ((r0, c0, r1, c1), color) in enumerate(zip(rects, colors)):
        slot_left = idx * (scene_w + 1)
        out[slot_top : slot_top + scene_h, slot_left + 1 : slot_left + 1 + scene_w] = BG
        out[slot_top + r0 : slot_top + r1 + 1, slot_left + 1 + c0 : slot_left + 1 + c1 + 1] = int(color)
    return out


def _attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    max_scene_h = (ARC_MAX - 1) // 2
    max_scene_w = (ARC_MAX - (count + 1)) // count
    chain = _sample_rect_chain(count, max_scene_h, max_scene_w)
    if chain is None:
        return None
    rects, _modes = chain
    outer = rects[-1]
    outer_h = outer[2] - outer[0] + 1
    outer_w = outer[3] - outer[1] + 1

    scene_h = random.randint(outer_h, min(max_scene_h, outer_h + 3))
    scene_w = random.randint(outer_w, min(max_scene_w, outer_w + 3))
    rects = _shift_chain(rects, scene_h, scene_w)
    colors = random.sample(PALETTE, count)

    scene = _render_scene(scene_h, scene_w, rects, colors)
    inp = _build_input(scene, count)
    expected = _build_expected_output(inp, rects, colors, scene_h, scene_w)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-rectangles puzzle")
