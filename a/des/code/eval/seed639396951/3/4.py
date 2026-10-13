from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

RECT_COUNT_RANGE = (3, 4)
TOP_RECT_H_RANGE = (2, 5)
TOP_RECT_W_RANGE = (2, 5)
SAFE_MARGIN_RANGE = (1, 4)
OTHER_MARGIN_RANGE = (0, 3)
STACK_PAD_RANGE = (1, 3)
RIGHT_PAD_RANGE = (0, 4)
GRID_H_MAX = 24
GRID_W_MAX = 30
GEN_RETRIES = 300

_CORNERS = ("TL", "TR", "BL", "BR")


def _bbox_of_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _run_row(grid: np.ndarray, row: int, col: int, step: int, color: int) -> int:
    width = grid.shape[1]
    length = 0
    cc = int(col)
    while 0 <= cc < width and int(grid[row, cc]) == int(color):
        length += 1
        cc += int(step)
    return int(length)


def _run_col(grid: np.ndarray, row: int, col: int, step: int, color: int) -> int:
    height = grid.shape[0]
    length = 0
    rr = int(row)
    while 0 <= rr < height and int(grid[rr, col]) == int(color):
        length += 1
        rr += int(step)
    return int(length)


def _point_in_box(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = box
    return bool(r0 <= r <= r1 and c0 <= c <= c1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return bool(ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1)


def _paint_box(canvas: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    canvas[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w - 1):
        col = grid[:, c]
        if not np.all(col == col[0]):
            continue
        sep_color = int(col[0])
        if sep_color == int(bg):
            continue
        if not np.all(grid[:, c + 1 :] == int(bg)):
            continue
        candidates.append((c, sep_color))
    assert len(candidates) == 1, "expected exactly one gray separator column"
    return candidates[0]


def _candidate_boxes(stack: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    coords = np.argwhere(stack == int(color))
    assert coords.size > 0
    rmin, cmin, rmax, cmax = _bbox_of_coords(coords)
    h, w = stack.shape
    candidates: set[tuple[int, int, int, int]] = set()

    def add_candidate(box: tuple[int, int, int, int]) -> None:
        r0, c0, r1, c1 = box
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
            return
        inside = (
            (coords[:, 0] >= r0)
            & (coords[:, 0] <= r1)
            & (coords[:, 1] >= c0)
            & (coords[:, 1] <= c1)
        )
        if bool(np.all(inside)):
            candidates.add((int(r0), int(c0), int(r1), int(c1)))

    if int(stack[rmin, cmin]) == int(color):
        width = _run_row(stack, rmin, cmin, +1, color)
        height = _run_col(stack, rmin, cmin, +1, color)
        add_candidate((rmin, cmin, rmin + height - 1, cmin + width - 1))

    if int(stack[rmin, cmax]) == int(color):
        width = _run_row(stack, rmin, cmax, -1, color)
        height = _run_col(stack, rmin, cmax, +1, color)
        add_candidate((rmin, cmax - width + 1, rmin + height - 1, cmax))

    if int(stack[rmax, cmin]) == int(color):
        width = _run_row(stack, rmax, cmin, +1, color)
        height = _run_col(stack, rmax, cmin, -1, color)
        add_candidate((rmax - height + 1, cmin, rmax, cmin + width - 1))

    if int(stack[rmax, cmax]) == int(color):
        width = _run_row(stack, rmax, cmax, -1, color)
        height = _run_col(stack, rmax, cmax, -1, color)
        add_candidate((rmax - height + 1, cmax - width + 1, rmax, cmax))

    ordered = sorted(candidates, key=lambda b: ((b[2] - b[0] + 1) * (b[3] - b[1] + 1), b))
    assert ordered, f"failed to infer any bbox for color {color}"
    return ordered


def _render_stack(
    shape: tuple[int, int],
    bg: int,
    order: tuple[int, ...],
    boxes: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color in reversed(order):
        _paint_box(out, boxes[int(color)], int(color))
    return out


def _solve_stack(
    stack: np.ndarray,
    bg: int,
    max_solutions: int = 1,
) -> list[tuple[tuple[int, ...], dict[int, tuple[int, int, int, int]]]]:
    colors = [int(c) for c in np.unique(stack) if int(c) != int(bg)]
    assert colors, "expected at least one rectangle color"

    data = {}
    for color in colors:
        coords = np.argwhere(stack == int(color))
        data[color] = {
            "coords": coords,
            "points": [tuple(map(int, pt)) for pt in coords],
            "cands": _candidate_boxes(stack, color),
        }

    solutions: list[tuple[tuple[int, ...], dict[int, tuple[int, int, int, int]]]] = []
    cand_index_lists = [range(len(data[color]["cands"])) for color in colors]
    for combo in itertools.product(*cand_index_lists):
        boxes = {
            int(color): data[color]["cands"][idx]
            for color, idx in zip(colors, combo)
        }

        above: dict[int, set[int]] = {int(color): set() for color in colors}
        valid = True
        for i, color_a in enumerate(colors):
            box_a = boxes[int(color_a)]
            pts_a = data[color_a]["points"]
            for color_b in colors[i + 1 :]:
                box_b = boxes[int(color_b)]
                if not _boxes_overlap(box_a, box_b):
                    continue
                pts_b = data[color_b]["points"]
                a_over_b = any(_point_in_box(pt, box_b) for pt in pts_a)
                b_over_a = any(_point_in_box(pt, box_a) for pt in pts_b)
                if a_over_b and b_over_a:
                    valid = False
                    break
                if a_over_b:
                    above[int(color_b)].add(int(color_a))
                elif b_over_a:
                    above[int(color_a)].add(int(color_b))
            if not valid:
                break
        if not valid:
            continue

        for order in itertools.permutations(colors):
            top_index = {int(color): i for i, color in enumerate(order)}
            if any(
                top_index[top] >= top_index[under]
                for under, tops in above.items()
                for top in tops
            ):
                continue
            rendered = _render_stack(stack.shape, bg, tuple(map(int, order)), boxes)
            if np.array_equal(rendered, stack):
                solutions.append((tuple(map(int, order)), boxes.copy()))
                if len(solutions) >= max_solutions:
                    return solutions
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden full rectangles in the left stack and redraw them, top to bottom,
    as an uncovered horizontal strip on the right side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    sep_c, sep_color = _find_separator(g, bg)
    assert sep_color != bg

    stack = g[:, :sep_c]
    solutions = _solve_stack(stack, bg, max_solutions=1)
    assert solutions, "failed to recover rectangle stack"
    order, boxes = solutions[0]

    out = g.copy()
    out[:, sep_c + 1 :] = bg
    cursor = sep_c + 1
    for color in order:
        r0, c0, r1, c1 = boxes[int(color)]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert cursor + width <= g.shape[1], "arrangement area is too narrow"
        out[:height, cursor : cursor + width] = int(color)
        cursor += width + 1
    return out


def _sample_nested_boxes() -> tuple[list[int], list[tuple[int, int, int, int]]] | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    colors = random.sample(PALETTE, n_rects)

    top_h = random.randint(*TOP_RECT_H_RANGE)
    top_w = random.randint(*TOP_RECT_W_RANGE)
    boxes = [(0, 0, top_h - 1, top_w - 1)]

    for _ in range(1, n_rects):
        cur_r0, cur_c0, cur_r1, cur_c1 = boxes[-1]
        corner = random.choice(_CORNERS)
        top_pad = random.randint(*OTHER_MARGIN_RANGE)
        bottom_pad = random.randint(*OTHER_MARGIN_RANGE)
        left_pad = random.randint(*OTHER_MARGIN_RANGE)
        right_pad = random.randint(*OTHER_MARGIN_RANGE)

        if corner == "TL":
            top_pad = random.randint(*SAFE_MARGIN_RANGE)
            left_pad = random.randint(*SAFE_MARGIN_RANGE)
        elif corner == "TR":
            top_pad = random.randint(*SAFE_MARGIN_RANGE)
            right_pad = random.randint(*SAFE_MARGIN_RANGE)
        elif corner == "BL":
            bottom_pad = random.randint(*SAFE_MARGIN_RANGE)
            left_pad = random.randint(*SAFE_MARGIN_RANGE)
        else:
            bottom_pad = random.randint(*SAFE_MARGIN_RANGE)
            right_pad = random.randint(*SAFE_MARGIN_RANGE)

        new_box = (
            cur_r0 - top_pad,
            cur_c0 - left_pad,
            cur_r1 + bottom_pad,
            cur_c1 + right_pad,
        )
        if new_box == boxes[-1]:
            return None
        boxes.append(new_box)

    min_r = min(r0 for r0, _c0, _r1, _c1 in boxes)
    min_c = min(c0 for _r0, c0, _r1, _c1 in boxes)
    shifted = []
    for r0, c0, r1, c1 in boxes:
        shifted.append((r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c))
    return colors, shifted


def _render_input(
    order: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray | None:
    heights = [r1 - r0 + 1 for r0, _c0, r1, _c1 in boxes]
    widths = [c1 - c0 + 1 for _r0, c0, _r1, c1 in boxes]

    stack_h = max(r1 for _r0, _c0, r1, _c1 in boxes) + 1
    stack_w = max(c1 for _r0, _c0, _r1, c1 in boxes) + 1

    top_pad = random.randint(*STACK_PAD_RANGE)
    bottom_pad = random.randint(*STACK_PAD_RANGE)
    left_pad = random.randint(*STACK_PAD_RANGE)
    stack_right_pad = random.randint(*STACK_PAD_RANGE)
    right_extra = random.randint(*RIGHT_PAD_RANGE)

    grid_h = stack_h + top_pad + bottom_pad
    arrange_w = sum(widths) + (len(widths) - 1) + right_extra
    grid_w = left_pad + stack_w + stack_right_pad + 1 + arrange_w
    if grid_h > GRID_H_MAX or grid_w > GRID_W_MAX:
        return None

    shifted_boxes = []
    for r0, c0, r1, c1 in boxes:
        shifted_boxes.append((r0 + top_pad, c0 + left_pad, r1 + top_pad, c1 + left_pad))

    sep_c = left_pad + stack_w + stack_right_pad
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:, sep_c] = SEP

    for color, box in reversed(list(zip(order, shifted_boxes))):
        _paint_box(grid, box, color)

    if int(np.argmax(np.bincount(grid.ravel(), minlength=10))) != BG:
        return None

    target = main(grid)
    if np.array_equal(target, grid):
        return None

    stack = grid[:, :sep_c]
    solutions = _solve_stack(stack, BG, max_solutions=2)
    if len(solutions) != 1:
        return None

    solved_order, solved_boxes = solutions[0]
    solved_dims = [(solved_boxes[c][2] - solved_boxes[c][0] + 1, solved_boxes[c][3] - solved_boxes[c][1] + 1) for c in solved_order]
    true_dims = [(box[2] - box[0] + 1, box[3] - box[1] + 1) for color, box in zip(order, shifted_boxes)]
    if list(solved_order) != order or solved_dims != true_dims:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _sample_nested_boxes()
        if sample is None:
            continue
        order, boxes = sample
        grid = _render_input(order, boxes)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacked-rectangle puzzle")
