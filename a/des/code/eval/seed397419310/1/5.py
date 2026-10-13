from __future__ import annotations

import random

import numpy as np


BG_COLOR = 7
FRAME_COLOR = 5
PIECE_COLOR_POOL = (1, 2, 3, 4, 6, 8, 9)
GEN_RETRIES = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_shape(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _draw_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _intersection(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_target_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue

        mask = np.zeros(grid.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        expected = np.zeros(grid.shape, dtype=bool)
        expected[r0 : r1 + 1, c0] = True
        expected[r0 : r1 + 1, c1] = True
        expected[r0, c0 : c1 + 1] = True
        expected[r1, c0 : c1 + 1] = True
        if not np.array_equal(mask, expected):
            continue
        if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
            continue
        candidates.append(((r1 - r0 + 1) * (c1 - c0 + 1), color, box))

    assert len(candidates) == 1, "expected a unique empty rectangular frame"
    _area, color, box = candidates[0]
    return int(color), box


def _extract_pieces(
    grid: np.ndarray, bg: int, frame_color: int
) -> dict[int, tuple[int, int, int, int]]:
    pieces: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg), int(frame_color)}):
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        pieces[int(color)] = _bbox(coords)
    assert pieces, "expected at least one colored rectangle fragment"
    return pieces


def _infer_order(
    grid: np.ndarray, pieces: dict[int, tuple[int, int, int, int]]
) -> list[int]:
    colors = list(pieces)
    edges = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            inter = _intersection(pieces[a], pieces[b])
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            window = grid[r0 : r1 + 1, c0 : c1 + 1]
            a_visible = bool(np.any(window == a))
            b_visible = bool(np.any(window == b))
            assert not (a_visible and b_visible), "two rectangles cannot both be visible in one overlap"
            if a_visible == b_visible:
                continue
            lower, upper = (b, a) if a_visible else (a, b)
            if upper not in edges[lower]:
                edges[lower].add(upper)
                indegree[upper] += 1

    order: list[int] = []
    ready = sorted(color for color in colors if indegree[color] == 0)
    while ready:
        assert len(ready) == 1, "occlusion constraints must define a unique stacking order"
        color = ready.pop()
        order.append(color)
        for upper in sorted(edges[color]):
            indegree[upper] -= 1
            if indegree[upper] == 0:
                ready.append(upper)
        ready.sort()

    assert len(order) == len(colors), "stacking graph must be acyclic and complete"
    return order


def _render_output(
    shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    order: list[int],
    pieces: dict[int, tuple[int, int, int, int]],
    bg: int,
    frame_color: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    _draw_frame(out, frame_box, int(frame_color))

    inner_r0 = int(frame_box[0] + 1)
    inner_c0 = int(frame_box[1] + 1)
    inner_r1 = int(frame_box[2] - 1)
    inner_c1 = int(frame_box[3] - 1)

    for idx, color in enumerate(order):
        height, width = _bbox_shape(pieces[color])
        start_r = inner_r0 + idx
        start_c = inner_c0 + idx
        end_r = min(inner_r1, start_r + height - 1)
        end_c = min(inner_c1, start_c + width - 1)
        if start_r > inner_r1 or start_c > inner_c1:
            continue
        out[start_r : end_r + 1, start_c : end_c + 1] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the bottom-to-top order of overlapping colored rectangles, then redraw them
    into the empty framed target area using diagonal cascade offsets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame_color, frame_box = _find_target_frame(g, bg)
    pieces = _extract_pieces(g, bg, frame_color)
    order = _infer_order(g, pieces)
    return _render_output(g.shape, frame_box, order, pieces, bg, frame_color)


def _visibility_ok(top: int, bottom: int, left: int, right: int) -> bool:
    if top == 0 and left + right == 0:
        return False
    if bottom == 0 and left + right == 0:
        return False
    if left == 0 and top + bottom == 0:
        return False
    if right == 0 and top + bottom == 0:
        return False
    return True


def _sample_nested_boxes(count: int) -> list[tuple[int, int, int, int]] | None:
    outer_h = random.randint(max(7, count + 3), 14)
    outer_w = random.randint(max(7, count + 3), 14)
    boxes = [(0, 0, outer_h - 1, outer_w - 1)]

    for _level in range(1, count):
        pr0, pc0, pr1, pc1 = boxes[-1]
        prev_h = pr1 - pr0 + 1
        prev_w = pc1 - pc0 + 1
        found = None
        for _ in range(200):
            top = random.randint(0, min(3, prev_h - 2))
            bottom = random.randint(0, min(3, prev_h - top - 2))
            left = random.randint(0, min(3, prev_w - 2))
            right = random.randint(0, min(3, prev_w - left - 2))
            if top + bottom + left + right == 0:
                continue
            new_h = prev_h - top - bottom
            new_w = prev_w - left - right
            if new_h < 2 or new_w < 2:
                continue
            if not _visibility_ok(top, bottom, left, right):
                continue
            found = (pr0 + top, pc0 + left, pr1 - bottom, pc1 - right)
            break
        if found is None:
            return None
        boxes.append(found)

    return boxes


def _render_source_local(
    boxes: list[tuple[int, int, int, int]], colors: list[int], bg: int
) -> np.ndarray:
    height, width = _bbox_shape(boxes[0])
    local = np.full((height, width), int(bg), dtype=int)
    for box, color in zip(boxes, colors, strict=True):
        _draw_rect(local, box, int(color))
    return local


def _required_inner_size(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    need_h = 0
    need_w = 0
    for idx, box in enumerate(boxes):
        height, width = _bbox_shape(box)
        need_h = max(need_h, height + idx)
        need_w = max(need_w, width + idx)
    return int(need_h), int(need_w)


def _sample_inner_size(need_h: int, need_w: int) -> tuple[int, int]:
    if random.random() < 0.5:
        return need_h + random.randint(0, 4), need_w + random.randint(0, 4)

    shrink_h = random.randint(0, min(3, max(0, need_h - 3)))
    shrink_w = random.randint(0, min(3, max(0, need_w - 3)))
    if shrink_h == 0 and shrink_w == 0:
        if need_h >= need_w and need_h > 3:
            shrink_h = 1
        elif need_w > 3:
            shrink_w = 1
        else:
            return need_h, need_w
    return max(3, need_h - shrink_h), max(3, need_w - shrink_w)


def _place_horizontal(
    source_shape: tuple[int, int], frame_shape: tuple[int, int], source_left: bool
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    source_h, source_w = source_shape
    frame_h, frame_w = frame_shape
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    gap = random.randint(3, 6)
    extra_h = random.randint(2, 8)
    height = max(source_h, frame_h) + extra_h
    width = left_margin + source_w + gap + frame_w + right_margin
    source_r = random.randint(1, height - source_h - 1)
    frame_r = random.randint(1, height - frame_h - 1)
    if source_left:
        source_c = left_margin
        frame_c = left_margin + source_w + gap
    else:
        frame_c = left_margin
        source_c = left_margin + frame_w + gap
    return (height, width), (source_r, source_c), (frame_r, frame_c)


def _place_vertical(
    source_shape: tuple[int, int], frame_shape: tuple[int, int], source_top: bool
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    source_h, source_w = source_shape
    frame_h, frame_w = frame_shape
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    gap = random.randint(3, 6)
    extra_w = random.randint(2, 8)
    width = max(source_w, frame_w) + extra_w
    height = top_margin + source_h + gap + frame_h + bottom_margin
    source_c = random.randint(1, width - source_w - 1)
    frame_c = random.randint(1, width - frame_w - 1)
    if source_top:
        source_r = top_margin
        frame_r = top_margin + source_h + gap
    else:
        frame_r = top_margin
        source_r = top_margin + frame_h + gap
    return (height, width), (source_r, source_c), (frame_r, frame_c)


def _make_case() -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = random.randint(3, 5)
    colors = random.sample(list(PIECE_COLOR_POOL), piece_count)
    boxes = _sample_nested_boxes(piece_count)
    if boxes is None:
        return None

    source_local = _render_source_local(boxes, colors, BG_COLOR)
    source_h, source_w = source_local.shape

    need_h, need_w = _required_inner_size(boxes)
    inner_h, inner_w = _sample_inner_size(need_h, need_w)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    orientation = random.choice(("lr", "rl", "tb", "bt"))
    if orientation == "lr":
        shape, source_pos, frame_pos = _place_horizontal((source_h, source_w), (frame_h, frame_w), True)
    elif orientation == "rl":
        shape, source_pos, frame_pos = _place_horizontal((source_h, source_w), (frame_h, frame_w), False)
    elif orientation == "tb":
        shape, source_pos, frame_pos = _place_vertical((source_h, source_w), (frame_h, frame_w), True)
    else:
        shape, source_pos, frame_pos = _place_vertical((source_h, source_w), (frame_h, frame_w), False)

    grid = np.full(shape, BG_COLOR, dtype=int)
    sr, sc = source_pos
    local_mask = source_local != BG_COLOR
    grid[sr : sr + source_h, sc : sc + source_w][local_mask] = source_local[local_mask]

    fr, fc = frame_pos
    frame_box = (fr, fc, fr + frame_h - 1, fc + frame_w - 1)
    _draw_frame(grid, frame_box, FRAME_COLOR)

    bg_count = int(np.sum(grid == BG_COLOR))
    if bg_count <= (grid.size // 2):
        return None

    expected_pieces = {
        color: (sr + box[0], sc + box[1], sr + box[2], sc + box[3])
        for color, box in zip(colors, boxes, strict=True)
    }
    expected = _render_output(grid.shape, frame_box, colors, expected_pieces, BG_COLOR, FRAME_COLOR)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _make_case()
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
