from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


BBox = tuple[int, int, int, int]
Direction = tuple[int, int]

BG = 0
CARDINALS: tuple[Direction, ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))
GEN_ATTEMPTS = 400


def _shift_box(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _box_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_solid_rectangle(coords: np.ndarray) -> tuple[bool, BBox]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    box = (int(r0), int(c0), int(r1), int(c1))
    return coords.shape[0] == _box_area(box), box


def _local_mask(coords: np.ndarray) -> tuple[np.ndarray, BBox]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _detect_slider(coords: np.ndarray) -> tuple[BBox, Direction, int] | None:
    mask, (r0, c0, r1, c1) = _local_mask(coords)
    h, w = mask.shape
    hits: list[tuple[BBox, Direction, int]] = []

    for length in range(2, w):
        body = mask[:, length:]
        tail = mask[:, :length]
        if body.size and body.all() and tail.sum() == length:
            tail_rows, tail_cols = np.where(tail)
            if len(set(map(int, tail_rows))) == 1 and set(map(int, tail_cols)) == set(range(length)):
                hits.append(((r0, c0 + length, r1, c1), (0, 1), length))

        body = mask[:, : w - length]
        tail = mask[:, w - length :]
        if body.size and body.all() and tail.sum() == length:
            tail_rows, tail_cols = np.where(tail)
            if len(set(map(int, tail_rows))) == 1 and set(map(int, tail_cols)) == set(range(length)):
                hits.append(((r0, c0, r1, c1 - length), (0, -1), length))

    for length in range(2, h):
        body = mask[length:, :]
        tail = mask[:length, :]
        if body.size and body.all() and tail.sum() == length:
            tail_rows, tail_cols = np.where(tail)
            if len(set(map(int, tail_cols))) == 1 and set(map(int, tail_rows)) == set(range(length)):
                hits.append(((r0 + length, c0, r1, c1), (1, 0), length))

        body = mask[: h - length, :]
        tail = mask[h - length :, :]
        if body.size and body.all() and tail.sum() == length:
            tail_rows, tail_cols = np.where(tail)
            if len(set(map(int, tail_cols))) == 1 and set(map(int, tail_rows)) == set(range(length)):
                hits.append(((r0, c0, r1 - length, c1), (-1, 0), length))

    if len(hits) != 1:
        return None
    return hits[0]


def _sort_key(box: BBox, direction: Direction) -> int:
    dr, dc = direction
    if dc > 0:
        return box[1]
    if dc < 0:
        return -box[3]
    if dr > 0:
        return box[0]
    return -box[2]


def _orthogonally_overlap(a: BBox, b: BBox, direction: Direction) -> bool:
    dr, dc = direction
    if dc != 0:
        return max(a[0], b[0]) <= min(a[2], b[2])
    return max(a[1], b[1]) <= min(a[3], b[3])


def _forward_gap(back: BBox, front: BBox, direction: Direction) -> int:
    dr, dc = direction
    if dc > 0:
        return front[1] - back[3] - 1
    if dc < 0:
        return back[1] - front[3] - 1
    if dr > 0:
        return front[0] - back[2] - 1
    return back[0] - front[2] - 1


def _propagate_boxes(boxes: list[BBox], start_index: int, direction: Direction, budget: int) -> list[BBox]:
    out = list(boxes)
    dr, dc = direction
    remaining = int(budget)
    current = int(start_index)
    while remaining > 0:
        if current == len(out) - 1:
            out[current] = _shift_box(out[current], dr * remaining, dc * remaining)
            break

        gap = _forward_gap(out[current], out[current + 1], direction)
        assert gap >= 0, "chain order must have non-negative gaps"
        step = min(remaining, gap)
        if step:
            out[current] = _shift_box(out[current], dr * step, dc * step)
        remaining -= step
        if remaining == 0:
            break
        assert step == gap, "remaining budget can transfer only after contact"
        current += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the tailed block along the rectangle chain, transferring the remaining displacement budget at each collision.

    Input structure:
    - The background is black.
    - Several disconnected foreground objects lie on one horizontal or vertical chain; each object is a single-color solid rectangle except for one rectangle with a one-cell-thick straight tail attached on its rear side.
    - The tail length is at least two, and all rectangles are spaced apart by empty gaps along the motion axis while overlapping on the orthogonal span.

    Rule:
    - Identify the unique tailed rectangle, erase its tail, and treat the tail length as the total displacement budget in the direction opposite the tail.
    - The moving rectangle slides forward until it first touches the next rectangle, spending exactly that gap distance; any unused budget is transferred to the struck rectangle, which then repeats the same process in the same direction.
    - Redraw every rectangle at its final position after the budget is fully exhausted; colors stay the same and the tail disappears.

    Key insight:
    - The tail encodes one shared motion budget for the whole collision chain, not a permanent translation of one block or a rigid push of the entire row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG, "expected black background"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(components) >= 2, "expected at least two foreground objects"

    objects: list[dict] = []
    slider_count = 0
    direction: Direction | None = None
    budget = None
    slider_component_index = None

    for idx, comp in enumerate(components):
        coords = comp["coords"]
        color = int(comp["color"])
        slider = _detect_slider(coords)
        if slider is not None:
            slider_count += 1
            body_box, direction, budget = slider
            slider_component_index = idx
            objects.append({"color": color, "box": body_box, "is_slider": True})
            continue

        ok, box = _is_solid_rectangle(coords)
        assert ok, "non-slider objects must be solid rectangles"
        objects.append({"color": color, "box": box, "is_slider": False})

    assert slider_count == 1 and direction is not None and budget is not None
    assert slider_component_index is not None

    order = sorted(range(len(objects)), key=lambda i: _sort_key(objects[i]["box"], direction))
    ordered_boxes = [objects[i]["box"] for i in order]
    ordered_colors = [objects[i]["color"] for i in order]
    ordered_slider = [objects[i]["is_slider"] for i in order]
    assert sum(ordered_slider) == 1
    start_index = ordered_slider.index(True)

    for i in range(len(ordered_boxes) - 1):
        assert _orthogonally_overlap(ordered_boxes[i], ordered_boxes[i + 1], direction), "chain rectangles must overlap orthogonally"
        assert _forward_gap(ordered_boxes[i], ordered_boxes[i + 1], direction) >= 0, "chain rectangles must be ordered along one axis"

    final_boxes = _propagate_boxes(ordered_boxes, start_index, direction, budget)
    out = np.full_like(g, bg)
    for color, box in zip(ordered_colors, final_boxes, strict=True):
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _simulate_shifts(gaps: list[int], collision_count: int, residual: int, n_blocks: int) -> list[int]:
    shifts = [0] * n_blocks
    for i in range(collision_count):
        shifts[i] = gaps[i]
    shifts[collision_count] = residual
    return shifts


def _canonical_expected(starts: list[int], lengths: list[int], gaps: list[int], collision_count: int, residual: int) -> list[tuple[int, int]]:
    shifts = _simulate_shifts(gaps, collision_count, residual, len(lengths))
    return [(start + shift, start + shift + length - 1) for start, shift, length in zip(starts, shifts, lengths, strict=True)]


def _apply_orientation(inp: np.ndarray, out: np.ndarray, orientation: str) -> tuple[np.ndarray, np.ndarray]:
    if orientation == "right":
        return inp, out
    if orientation == "left":
        return np.fliplr(inp), np.fliplr(out)
    if orientation == "down":
        return inp.T, out.T
    if orientation == "up":
        return np.flipud(inp.T), np.flipud(out.T)
    raise ValueError(f"unknown orientation {orientation}")


def _paint_rect(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _gen_attempt() -> np.ndarray | None:
    n_blocks = random.randint(2, 4)
    colors = random.sample(range(1, 10), n_blocks)
    cross = random.randint(3, 5)
    lengths = [random.randint(2, 4) for _ in range(n_blocks)]
    gaps = [random.randint(1, 3) for _ in range(n_blocks - 1)]

    collision_count = random.randint(0, n_blocks - 1)
    spent_before = sum(gaps[:collision_count])
    if collision_count < n_blocks - 1:
        min_residual = max(0, 2 - spent_before)
        max_residual = gaps[collision_count] - 1
    else:
        min_residual = max(0, 2 - spent_before)
        max_residual = 3
    if min_residual > max_residual:
        return None

    if collision_count > 0 and min_residual == 0 and random.random() < 0.3:
        residual = 0
    else:
        residual = random.randint(min_residual, max_residual)
    budget = spent_before + residual
    if budget < 2:
        return None

    base_span = budget + sum(lengths) + sum(gaps)
    if base_span > 28:
        return None

    left_margin = random.randint(0, 30 - base_span)
    right_margin = random.randint(0, 30 - base_span - left_margin)
    top_margin = random.randint(0, 30 - cross)
    bottom_margin = random.randint(0, 30 - cross - top_margin)

    width = base_span + left_margin + right_margin
    height = cross + top_margin + bottom_margin
    if width > 30 or height > 30:
        return None

    starts = [left_margin + budget]
    for length, gap in zip(lengths[:-1], gaps, strict=True):
        starts.append(starts[-1] + length + gap)

    slider_row = top_margin + random.randint(0, cross - 1)
    body_top = top_margin
    body_bottom = top_margin + cross - 1

    inp = np.full((height, width), BG, dtype=int)
    out = np.full_like(inp, BG)

    final_intervals = _canonical_expected(starts, lengths, gaps, collision_count, residual)
    for i, (start, length, color) in enumerate(zip(starts, lengths, colors, strict=True)):
        box = (body_top, start, body_bottom, start + length - 1)
        _paint_rect(inp, box, color)
        if i == 0:
            inp[slider_row, start - budget : start] = color

    for color, (c0, c1) in zip(colors, final_intervals, strict=True):
        _paint_rect(out, (body_top, c0, body_bottom, c1), color)

    orientation = random.choice(["right", "left", "down", "up"])
    inp, out = _apply_orientation(inp, out, orientation)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
