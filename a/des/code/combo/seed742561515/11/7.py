from __future__ import annotations

from collections import Counter
import random

import numpy as np


MIN_LEFT_H = 8
MAX_LEFT_H = 16
MIN_LEFT_W = 8
MAX_LEFT_W = 16
MIN_RECTS = 3
MAX_RECTS = 6
MIN_RECT_SIDE = 2
MAX_GENERATION_ATTEMPTS = 500
COLOR_VALUES = tuple(range(10))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    best = int(counts.max())
    winners = [int(v) for v, c in zip(values, counts) if int(c) == best]
    assert len(winners) == 1, "background color must be uniquely dominant"
    return winners[0]


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert len(coords) > 0, "color must be visible"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _find_separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    assert h > 0 and w % 2 == 1, "grid must split into two equal halves with one separator column"
    sep_col = w // 2
    assert np.all(grid[:, sep_col] == grid[0, sep_col]), "separator column must be uniform"
    return sep_col


def _rectangles_from_left(
    left: np.ndarray, bg: int, sep_color: int
) -> list[dict[str, int | tuple[int, int, int, int]]]:
    rectangles = []
    for color in map(int, np.unique(left)):
        if color in (bg, sep_color):
            continue
        rectangles.append({"color": color, "bbox": _bbox_of_color(left, color)})
    assert rectangles, "expected at least one rectangle"
    return rectangles


def _infer_bottom_to_top(
    left: np.ndarray,
    rectangles: list[dict[str, int | tuple[int, int, int, int]]],
    bg: int,
    sep_color: int,
) -> list[dict[str, int | tuple[int, int, int, int]]]:
    by_color = {int(rect["color"]): rect for rect in rectangles}
    edges: dict[int, set[int]] = {color: set() for color in by_color}
    indeg = {color: 0 for color in by_color}

    for color, rect in by_color.items():
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[assignment]
        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        above_colors = {
            int(other)
            for other in np.unique(patch)
            if int(other) not in (bg, sep_color, color)
        }
        for other in above_colors:
            edges[color].add(other)

    for color, targets in edges.items():
        for other in targets:
            indeg[other] += 1

    order: list[dict[str, int | tuple[int, int, int, int]]] = []
    ready = [
        color
        for color, degree in indeg.items()
        if degree == 0
    ]
    ready.sort(key=lambda color: by_color[color]["bbox"])  # type: ignore[index]

    while ready:
        color = ready.pop(0)
        order.append(by_color[color])
        for other in sorted(edges[color], key=lambda item: by_color[item]["bbox"]):  # type: ignore[index]
            indeg[other] -= 1
            if indeg[other] == 0:
                ready.append(other)
        ready.sort(key=lambda item: by_color[item]["bbox"])  # type: ignore[index]

    assert len(order) == len(rectangles), "rectangle precedence graph must be acyclic"
    return order


def _render_stack(
    shape: tuple[int, int],
    ordered_rectangles: list[dict[str, int | tuple[int, int, int, int]]],
    bg: int,
) -> np.ndarray:
    canvas = np.full(shape, bg, dtype=int)
    for rect in ordered_rectangles:
        color = int(rect["color"])
        r0, c0, r1, c1 = rect["bbox"]  # type: ignore[assignment]
        canvas[r0 : r1 + 1, c0 : c1 + 1] = color
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the stacking order of visible solid rectangles and redraw the reversed stack on the right half.

    Input structure:
    - A single uniform separator column splits the grid into equal left and right halves.
    - The left half shows several distinct solid rectangles after opaque overpainting.
    - The right half is blank background.

    Rule:
    - Recover each rectangle's full extent as the bounding box of its visible color on the left half.
    - Infer the bottom-to-top order: if color B appears inside color A's recovered box,
      then B must be above A.
    - Redraw the same rectangles on the right half using the reversed stack order.
    - Keep the left half unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)
    sep_col = _find_separator_col(g)
    sep_color = int(g[0, sep_col])
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape, "left and right halves must match"

    rectangles = _rectangles_from_left(left, bg, sep_color)
    bottom_to_top = _infer_bottom_to_top(left, rectangles, bg, sep_color)
    reversed_stack = list(reversed(bottom_to_top))
    rerendered = _render_stack(left.shape, reversed_stack, bg)

    out = g.copy()
    out[:, sep_col + 1 :] = rerendered
    return out


def _sample_rectangles(left_h: int, left_w: int, n_rects: int, strict: bool) -> list[tuple[int, int, int, int]] | None:
    step = 2 if strict else 1
    min_outer_h = MIN_RECT_SIDE + step * (n_rects - 1)
    min_outer_w = MIN_RECT_SIDE + step * (n_rects - 1)
    if min_outer_h > left_h or min_outer_w > left_w:
        return None

    outer_h = random.randint(min_outer_h, left_h)
    outer_w = random.randint(min_outer_w, left_w)
    if outer_h == left_h and outer_w == left_w:
        if outer_h > min_outer_h and (outer_w == min_outer_w or random.random() < 0.5):
            outer_h -= 1
        elif outer_w > min_outer_w:
            outer_w -= 1

    r0 = random.randint(0, left_h - outer_h)
    c0 = random.randint(0, left_w - outer_w)
    cur_r0, cur_c0 = r0, c0
    cur_h, cur_w = outer_h, outer_w
    rects: list[tuple[int, int, int, int]] = []

    for i in range(n_rects):
        rects.append((cur_r0, cur_c0, cur_r0 + cur_h - 1, cur_c0 + cur_w - 1))
        if i == n_rects - 1:
            break

        remaining = n_rects - i - 2
        if strict:
            min_child_h = MIN_RECT_SIDE + 2 * remaining
            min_child_w = MIN_RECT_SIDE + 2 * remaining
            max_child_h = cur_h - 2
            max_child_w = cur_w - 2
            if min_child_h > max_child_h or min_child_w > max_child_w:
                return None
            child_h = random.randint(min_child_h, max_child_h)
            child_w = random.randint(min_child_w, max_child_w)
            slack_h = cur_h - child_h
            slack_w = cur_w - child_w
            top = random.randint(1, slack_h - 1)
            left = random.randint(1, slack_w - 1)
        else:
            min_child_h = MIN_RECT_SIDE + remaining
            min_child_w = MIN_RECT_SIDE + remaining
            max_child_h = cur_h - 1
            max_child_w = cur_w - 1
            if min_child_h > max_child_h or min_child_w > max_child_w:
                return None
            child_h = random.randint(min_child_h, max_child_h)
            child_w = random.randint(min_child_w, max_child_w)
            slack_h = cur_h - child_h
            slack_w = cur_w - child_w
            top = random.randint(0, slack_h)
            left = random.randint(0, slack_w)

        cur_r0 += top
        cur_c0 += left
        cur_h = child_h
        cur_w = child_w

    return rects


def _paint_case(
    left_h: int,
    left_w: int,
    bg: int,
    sep_color: int,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    left = np.full((left_h, left_w), bg, dtype=int)
    right_blank = np.full((left_h, left_w), bg, dtype=int)

    rect_specs: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color, (r0, c0, r1, c1) in zip(colors, rects):
        left[r0 : r1 + 1, c0 : c1 + 1] = color
        rect_specs.append({"color": int(color), "bbox": (r0, c0, r1, c1)})

    reversed_right = _render_stack((left_h, left_w), list(reversed(rect_specs)), bg)
    separator = np.full((left_h, 1), sep_color, dtype=int)
    inp = np.concatenate([left, separator, right_blank], axis=1)
    expected = np.concatenate([left, separator, reversed_right], axis=1)
    return inp, expected


def _visible_bboxes_match(
    left: np.ndarray, rects: list[tuple[int, int, int, int]], colors: list[int]
) -> bool:
    for color, bbox in zip(colors, rects):
        coords = np.argwhere(left == color)
        if len(coords) == 0:
            return False
        if _bbox_of_color(left, color) != bbox:
            return False
    return True


def _background_is_unique_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    return counts[bg] > max(count for color, count in counts.items() if color != bg)


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    strict = random.random() < 0.5
    left_h = random.randint(MIN_LEFT_H, MAX_LEFT_H)
    left_w = random.randint(MIN_LEFT_W, MAX_LEFT_W)
    step = 2 if strict else 1
    max_rects = min(MAX_RECTS, min(left_h, left_w) - MIN_RECT_SIDE + (0 if strict else 1))
    while max_rects >= MIN_RECTS and MIN_RECT_SIDE + step * (max_rects - 1) > min(left_h, left_w):
        max_rects -= 1
    if max_rects < MIN_RECTS:
        return None

    n_rects = random.randint(MIN_RECTS, max_rects)
    rects = _sample_rectangles(left_h, left_w, n_rects, strict)
    if rects is None:
        return None

    chosen = random.sample(COLOR_VALUES, n_rects + 2)
    bg = chosen[0]
    sep_color = chosen[1]
    colors = chosen[2:]
    inp, expected = _paint_case(left_h, left_w, bg, sep_color, rects, colors)

    left = inp[:, :left_w]
    if not _visible_bboxes_match(left, rects, colors):
        return None
    if not _background_is_unique_dominant(inp, bg):
        return None
    if np.array_equal(inp, expected):
        return None
    try:
        if np.array_equal(main(inp), expected):
            return inp, expected
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _sample_case()
        if case is not None:
            inp, _expected = case
            return inp
    raise RuntimeError("failed to generate a valid input")
