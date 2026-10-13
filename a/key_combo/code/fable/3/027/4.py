from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BBox = tuple[int, int, int, int]
Direction = tuple[int, int]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _box_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _box_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _box_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _is_full(mask: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _solid_box(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    box = _bbox_coords(coords)
    if int(coords.shape[0]) != _box_area(box):
        return None
    if _box_h(box) < 2 or _box_w(box) < 2:
        return None
    return box


def _shift_box(box: BBox, dr: int, dc: int, amount: int) -> BBox:
    r0, c0, r1, c1 = box
    return (
        int(r0 + dr * amount),
        int(c0 + dc * amount),
        int(r1 + dr * amount),
        int(c1 + dc * amount),
    )


def _inside_box(coords: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return (
        (coords[:, 0] >= r0)
        & (coords[:, 0] <= r1)
        & (coords[:, 1] >= c0)
        & (coords[:, 1] <= c1)
    )


def _tail_line_ok(
    outside: np.ndarray,
    direction: Direction,
    body: BBox,
    length: int,
) -> bool:
    if outside.shape[0] != length or length < 2:
        return False

    r0, c0, r1, c1 = body
    dr, dc = direction
    rows = set(map(int, outside[:, 0]))
    cols = set(map(int, outside[:, 1]))

    if dc > 0:
        return len(rows) == 1 and next(iter(rows)) in range(r0, r1 + 1) and cols == set(range(c1 + 1, c1 + length + 1))
    if dc < 0:
        return len(rows) == 1 and next(iter(rows)) in range(r0, r1 + 1) and cols == set(range(c0 - length, c0))
    if dr > 0:
        return len(cols) == 1 and next(iter(cols)) in range(c0, c1 + 1) and rows == set(range(r1 + 1, r1 + length + 1))
    return len(cols) == 1 and next(iter(cols)) in range(c0, c1 + 1) and rows == set(range(r0 - length, r0))


def _detect_tailed_body(mask: np.ndarray) -> tuple[BBox, Direction, int] | None:
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty color mask"
    r0, c0, r1, c1 = _bbox_coords(coords)
    candidates: list[tuple[BBox, Direction, int]] = []

    def try_body(body: BBox, direction: Direction, length: int) -> None:
        if length < 2 or _box_h(body) < 2 or _box_w(body) < 2:
            return
        if not _is_full(mask, body):
            return
        in_body = _inside_box(coords, body)
        outside = coords[~in_body]
        if _box_area(body) + int(outside.shape[0]) != int(coords.shape[0]):
            return
        if _tail_line_ok(outside, direction, body, length):
            candidates.append((body, direction, int(length)))

    for body_c1 in range(c0, c1):
        try_body((r0, c0, r1, body_c1), (0, 1), c1 - body_c1)
    for body_c0 in range(c0 + 1, c1 + 1):
        try_body((r0, body_c0, r1, c1), (0, -1), body_c0 - c0)
    for body_r1 in range(r0, r1):
        try_body((r0, c0, body_r1, c1), (1, 0), r1 - body_r1)
    for body_r0 in range(r0 + 1, r1 + 1):
        try_body((body_r0, c0, r1, c1), (-1, 0), body_r0 - r0)

    assert len(candidates) <= 1, "body/tail decomposition must be unique"
    return candidates[0] if candidates else None


def _axis_interval(box: BBox, axis: int) -> tuple[int, int]:
    if axis == 0:
        return int(box[0]), int(box[2])
    return int(box[1]), int(box[3])


def _cross_interval(box: BBox, axis: int) -> tuple[int, int]:
    if axis == 0:
        return int(box[1]), int(box[3])
    return int(box[0]), int(box[2])


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _gap_between(cur: BBox, nxt: BBox, axis: int, sign: int) -> int:
    cur_lo, cur_hi = _axis_interval(cur, axis)
    next_lo, next_hi = _axis_interval(nxt, axis)
    if sign > 0:
        return int(next_lo - cur_hi - 1)
    return int(cur_lo - next_hi - 1)


def _order_key(box: BBox, axis: int, sign: int) -> int:
    lo, hi = _axis_interval(box, axis)
    return int(lo if sign > 0 else -hi)


def _simulate_chain(
    body_by_color: dict[int, BBox],
    slider_color: int,
    direction: Direction,
    budget: int,
    shape: tuple[int, int],
) -> dict[int, BBox]:
    dr, dc = direction
    axis = 0 if dr else 1
    sign = dr if axis == 0 else dc
    assert sign in (-1, 1)

    slider_box = body_by_color[slider_color]
    ordered_colors = sorted(body_by_color, key=lambda color: _order_key(body_by_color[color], axis, sign))
    assert ordered_colors[0] == slider_color, "tailed rectangle must be the first object in its travel direction"

    for left, right in zip(ordered_colors, ordered_colors[1:]):
        left_box = body_by_color[left]
        right_box = body_by_color[right]
        assert _gap_between(left_box, right_box, axis, sign) >= 1, "rectangles must be separated by background gaps"
        assert _intervals_overlap(
            _cross_interval(left_box, axis), _cross_interval(right_box, axis)
        ), "consecutive corridor rectangles must overlap across the corridor"

    out_boxes = dict(body_by_color)
    remaining = int(budget)
    for i, color in enumerate(ordered_colors):
        if remaining <= 0:
            break
        box = body_by_color[color]
        if i == len(ordered_colors) - 1:
            move = remaining
            remaining = 0
        else:
            gap = _gap_between(box, body_by_color[ordered_colors[i + 1]], axis, sign)
            if remaining <= gap:
                move = remaining
                remaining = 0
            else:
                move = gap
                remaining -= gap

        out_boxes[color] = _shift_box(box, dr, dc, move)

    height, width = shape
    occupied = np.zeros(shape, dtype=bool)
    for color, box in out_boxes.items():
        r0, c0, r1, c1 = box
        assert 0 <= r0 <= r1 < height and 0 <= c0 <= c1 < width, "moved rectangles must remain in bounds"
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any(region), f"rectangle {color} overlaps another result rectangle"
        region[:, :] = True
    return out_boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay the tail length as one conserved displacement budget through a line of colliding solid rectangles.

    Input structure:
    - The background is black.
    - Several differently colored solid rectangles are arranged along one horizontal or vertical corridor with background gaps between them.
    - Exactly one rectangle has a one-cell-thick straight tail of length at least two extending from one side in the corridor direction, toward the other rectangles.

    Rule:
    - Erase the tail and give its rectangle a nominal displacement equal to the tail length.
    - If a moving rectangle would touch the next rectangle before spending that displacement, it stops adjacent to that rectangle and passes the unspent displacement on unchanged.
    - Continue this collision transfer along the corridor; the final recipient moves all remaining cells, while rectangles never reached by the budget stay fixed.

    Key insight:
    - The tail length is the total injected displacement, not the first rectangle's own travel distance; each collision consumes only the intervening gap and relays the remaining budget.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(g.min()) >= 0 and int(g.max()) <= 9, "ARC colors must be 0..9"
    assert int(Counter(map(int, g.ravel())).most_common(1)[0][0]) == BG, "expected black background"

    body_by_color: dict[int, BBox] = {}
    slider: tuple[int, Direction, int] | None = None

    for color in sorted(set(map(int, g.ravel())) - {BG}):
        mask = g == int(color)
        solid = _solid_box(mask)
        if solid is not None:
            body_by_color[int(color)] = solid
            continue

        tailed = _detect_tailed_body(mask)
        assert tailed is not None, f"color {color} is neither a solid rectangle nor a tailed rectangle"
        assert slider is None, "expected exactly one tailed rectangle"
        body, direction, length = tailed
        body_by_color[int(color)] = body
        slider = (int(color), direction, int(length))

    assert slider is not None, "expected exactly one tailed rectangle"
    assert len(body_by_color) >= 3, "expected a corridor of at least three rectangles"

    slider_color, direction, length = slider
    result_boxes = _simulate_chain(body_by_color, slider_color, direction, length, g.shape)

    out = np.full_like(g, BG)
    for color, (r0, c0, r1, c1) in result_boxes.items():
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 600
COLORS = list(range(1, 10))
_GEN_CALL_COUNT = 0
_GEN_BRANCH_OFFSET = random.randrange(4)


def _choose_budget(gaps: list[int], branch: int) -> tuple[int, str]:
    n_gaps = len(gaps)
    if branch == 0 and n_gaps >= 2:
        consumed = random.randint(2, n_gaps)
        residual = random.randint(1, 4)
        return int(sum(gaps[:consumed]) + residual), "multi"

    if branch == 1 and n_gaps >= 3:
        consumed = random.randint(1, n_gaps - 2)
        residual = random.randint(1, max(1, gaps[consumed]))
        return int(sum(gaps[:consumed]) + residual), "partial"

    if branch == 2 and n_gaps >= 2:
        consumed = random.randint(2, n_gaps)
        return int(sum(gaps[:consumed])), "exact_gap"

    first_gap = max(2, gaps[0])
    return int(random.randint(2, first_gap)), "no_first_collision"


def _axis_boxes(lengths: list[int], gaps: list[int], sign: int) -> list[tuple[int, int]]:
    boxes: list[tuple[int, int]] = []
    if sign > 0:
        cur = 0
        for i, length in enumerate(lengths):
            boxes.append((cur, cur + int(length) - 1))
            if i < len(gaps):
                cur = boxes[-1][1] + int(gaps[i]) + 1
    else:
        lo, hi = 0, int(lengths[0]) - 1
        boxes.append((lo, hi))
        cur_hi = lo - int(gaps[0]) - 1 if gaps else lo - 1
        for i, length in enumerate(lengths[1:], start=1):
            boxes.append((cur_hi - int(length) + 1, cur_hi))
            if i < len(gaps):
                cur_hi = boxes[-1][0] - int(gaps[i]) - 1
    return boxes


def _moves_from_budget(gaps: list[int], budget: int, count: int) -> list[int]:
    moves = [0] * count
    remaining = int(budget)
    for i in range(count):
        if remaining <= 0:
            break
        if i == count - 1:
            moves[i] = remaining
            break
        if remaining <= gaps[i]:
            moves[i] = remaining
            break
        moves[i] = gaps[i]
        remaining -= gaps[i]
    return moves


def _box_from_axis_cross(
    axis_interval: tuple[int, int],
    cross_interval: tuple[int, int],
    axis: int,
) -> BBox:
    a0, a1 = axis_interval
    x0, x1 = cross_interval
    if axis == 1:
        return int(x0), int(a0), int(x1), int(a1)
    return int(a0), int(x0), int(a1), int(x1)


def _paint_box(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    body_boxes: list[BBox],
    colors: list[int],
    slider_color: int,
    direction: Direction,
    budget: int,
    kind: str,
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False

    body_by_color = {int(color): box for color, box in zip(colors, body_boxes)}
    dr, dc = direction
    axis = 0 if dr else 1
    sign = dr if axis == 0 else dc
    ordered = sorted(colors, key=lambda color: _order_key(body_by_color[int(color)], axis, sign))
    if int(ordered[0]) != int(slider_color):
        return False

    gaps = [_gap_between(body_by_color[int(a)], body_by_color[int(b)], axis, sign) for a, b in zip(ordered, ordered[1:])]
    if any(gap < 1 for gap in gaps):
        return False
    if any(
        not _intervals_overlap(_cross_interval(body_by_color[int(a)], axis), _cross_interval(body_by_color[int(b)], axis))
        for a, b in zip(ordered, ordered[1:])
    ):
        return False

    moves = _moves_from_budget(gaps, budget, len(ordered))
    if kind in {"multi", "partial", "exact_gap"} and moves[0] == budget:
        return False
    if kind == "partial" and not any(move == 0 for move in moves[2:]):
        return False
    if kind == "multi" and sum(move > 0 for move in moves) < 3:
        return False
    if kind == "exact_gap" and budget not in {sum(gaps[:i]) for i in range(2, len(gaps) + 1)}:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    return True


def _gen_attempt(branch: int) -> np.ndarray | None:
    orientation = "H" if branch in (0, 2) else "V"
    sign = 1 if branch in (0, 1) else -1
    axis = 1 if orientation == "H" else 0
    direction = (0, sign) if axis == 1 else (sign, 0)

    count = random.randint(4, 6) if branch != 3 else random.randint(3, 5)
    colors = random.sample(COLORS, count)
    lengths = [random.randint(2, 4)] + [random.randint(2, 5) for _ in range(count - 1)]
    gaps = [random.randint(1, 4) for _ in range(count - 1)]
    if branch == 3:
        gaps[0] = random.randint(3, 6)

    budget, kind = _choose_budget(gaps, branch)
    if budget < 2:
        return None

    axis_intervals = _axis_boxes(lengths, gaps, sign)
    moves = _moves_from_budget(gaps, budget, count)
    shifted_axis_intervals = [
        (a0 + sign * move, a1 + sign * move)
        for (a0, a1), move in zip(axis_intervals, moves)
    ]

    source_lo, source_hi = axis_intervals[0]
    if sign > 0:
        tail_interval = (source_hi + 1, source_hi + budget)
    else:
        tail_interval = (source_lo - budget, source_lo - 1)

    axis_min = min(
        [a0 for a0, _ in axis_intervals]
        + [a0 for a0, _ in shifted_axis_intervals]
        + [tail_interval[0]]
    )
    axis_max = max(
        [a1 for _, a1 in axis_intervals]
        + [a1 for _, a1 in shifted_axis_intervals]
        + [tail_interval[1]]
    )
    offset = random.randint(1, 3) - axis_min
    axis_size = axis_max + offset + random.randint(2, 4)
    if axis_size > 30:
        return None
    axis_intervals = [(a0 + offset, a1 + offset) for a0, a1 in axis_intervals]
    shifted_axis_intervals = [(a0 + offset, a1 + offset) for a0, a1 in shifted_axis_intervals]
    tail_interval = (tail_interval[0] + offset, tail_interval[1] + offset)

    band = random.randint(2, 3)
    lane_after = random.choice([False, True])
    if lane_after:
        source_cross = (0, band)
        other_cross = (0, band - 1)
        tail_cross = band
    else:
        source_cross = (0, band)
        other_cross = (1, band)
        tail_cross = 0

    cross_extent = band + 1
    cross_size = random.randint(cross_extent + 2, min(12, cross_extent + 8))
    cross_offset = random.randint(1, cross_size - cross_extent - 1)
    source_cross = (source_cross[0] + cross_offset, source_cross[1] + cross_offset)
    other_cross = (other_cross[0] + cross_offset, other_cross[1] + cross_offset)
    tail_cross += cross_offset

    shape = (cross_size, axis_size) if axis == 1 else (axis_size, cross_size)
    body_boxes: list[BBox] = []
    result_boxes: list[BBox] = []
    for i in range(count):
        cross = source_cross if i == 0 else other_cross
        body_boxes.append(_box_from_axis_cross(axis_intervals[i], cross, axis))
        result_boxes.append(_box_from_axis_cross(shifted_axis_intervals[i], cross, axis))

    inp = np.zeros(shape, dtype=int)
    expected = np.zeros(shape, dtype=int)
    for color, box in zip(colors, body_boxes):
        _paint_box(inp, box, color)
    for color, box in zip(colors, result_boxes):
        _paint_box(expected, box, color)

    source_box = body_boxes[0]
    if axis == 1:
        c0, c1 = sorted(tail_interval)
        inp[tail_cross, c0 : c1 + 1] = int(colors[0])
    else:
        r0, r1 = sorted(tail_interval)
        inp[r0 : r1 + 1, tail_cross] = int(colors[0])

    if not _semantic_check(inp, expected, body_boxes, colors, colors[0], direction, budget, kind):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    start_branch = (_GEN_BRANCH_OFFSET + _GEN_CALL_COUNT) % 4
    _GEN_CALL_COUNT += 1

    branches = [(start_branch + i) % 4 for i in range(4)]
    for branch in branches:
        for _ in range(GEN_ATTEMPTS // 4):
            grid = _gen_attempt(branch)
            if grid is not None:
                return grid
    raise RuntimeError(f"failed to generate a valid tail-relay input after {GEN_ATTEMPTS} attempts")
