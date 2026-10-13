from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
PINK = 6
INPUT_COLORS = (1, 2, 3, 4, 5, 7, 8, 9)
MAX_ATTEMPTS = 400


@dataclass(frozen=True)
class HiddenRect:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    diag_mode: int

    @property
    def original_corners(self) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.diag_mode == 0:
            return ((self.r0, self.c0), (self.r1, self.c1))
        return ((self.r0, self.c1), (self.r1, self.c0))


def _boundary_cells(rect: HiddenRect) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(rect.c0, rect.c1 + 1):
        cells.add((rect.r0, c))
        cells.add((rect.r1, c))
    for r in range(rect.r0, rect.r1 + 1):
        cells.add((r, rect.c0))
        cells.add((r, rect.c1))
    return cells


def _strictly_inside(rect: HiddenRect, r: int, c: int) -> bool:
    return rect.r0 < r < rect.r1 and rect.c0 < c < rect.c1


def _parse_rectangles(grid: np.ndarray) -> list[HiddenRect]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BLACK]
    rects: list[HiddenRect] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        (r_a, c_a), (r_b, c_b) = coords.tolist()
        assert r_a != r_b and c_a != c_b, "corner pair must be diagonal"
        r0, r1 = sorted((int(r_a), int(r_b)))
        c0, c1 = sorted((int(c_a), int(c_b)))
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "rectangle must have a strict interior"
        if (r_a, c_a) == (r0, c0):
            diag_mode = 0 if (r_b, c_b) == (r1, c1) else 1
        elif (r_a, c_a) == (r0, c1):
            diag_mode = 1
        else:
            diag_mode = 0
        rects.append(HiddenRect(color=color, r0=r0, c0=c0, r1=r1, c1=c1, diag_mode=diag_mode))
    return rects


def _render_output(shape: tuple[int, int], rects: list[HiddenRect]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    visible_colors: dict[tuple[int, int], set[int]] = {}

    for rect in rects:
        corners = set(rect.original_corners)
        for r, c in _boundary_cells(rect):
            if (r, c) in corners:
                continue
            if any(_strictly_inside(other, r, c) for other in rects if other.color != rect.color):
                visible_colors.setdefault((r, c), set()).add(rect.color)

    for (r, c), colors in visible_colors.items():
        out[r, c] = PINK if len(colors) >= 2 else next(iter(colors))

    for rect in rects:
        for r, c in rect.original_corners:
            out[r, c] = rect.color
    return out


def _reveal_cells_for_rect(rect: HiddenRect, rects: list[HiddenRect]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    corners = set(rect.original_corners)
    for r, c in _boundary_cells(rect):
        if (r, c) in corners:
            continue
        if any(_strictly_inside(other, r, c) for other in rects if other.color != rect.color):
            cells.add((r, c))
    return cells


def _make_input(shape: tuple[int, int], rects: list[HiddenRect]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for rect in rects:
        for r, c in rect.original_corners:
            assert grid[r, c] == BLACK
            grid[r, c] = rect.color
    return grid


def _sample_disjoint_intervals(length: int, count: int, span_lo: int, span_hi: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    min_needed = count * span_lo + (count - 1)
    if min_needed > length:
        raise RuntimeError("interval packing is impossible")
    for _ in range(120):
        spans: list[int] = []
        remaining = length - (count - 1)
        feasible = True
        for idx in range(count):
            rest = count - idx - 1
            max_span = min(span_hi, remaining - rest * span_lo)
            if max_span < span_lo:
                feasible = False
                break
            span = random.randint(span_lo, max_span)
            spans.append(span)
            remaining -= span
        if not feasible:
            continue
        slack = length - (sum(spans) + (count - 1))
        gap_extra = [0] * (count + 1)
        for _ in range(slack):
            gap_extra[random.randrange(count + 1)] += 1
        cursor = gap_extra[0]
        intervals: list[tuple[int, int]] = []
        for idx, span in enumerate(spans):
            intervals.append((cursor, cursor + span - 1))
            cursor += span
            if idx < count - 1:
                cursor += 1 + gap_extra[idx + 1]
        return intervals
    raise RuntimeError("failed to sample disjoint intervals")


def _build_rectangles(force_pink: bool) -> tuple[tuple[int, int], list[HiddenRect]] | None:
    if force_pink:
        extra = 1 if random.random() < 0.45 else 0
        vertical_count = 1 + (1 if extra and random.random() < 0.5 else 0)
        horizontal_count = 1 + (1 if extra and vertical_count == 1 else 0)
    else:
        axis = random.choice(("v", "h"))
        extra = 1 if random.random() < 0.35 else 0
        vertical_count = 2 + extra if axis == "v" else 0
        horizontal_count = 2 + extra if axis == "h" else 0

    min_host_h = max(9, 2 + horizontal_count * 3 + max(0, horizontal_count - 1))
    min_host_w = max(9, 2 + vertical_count * 3 + max(0, vertical_count - 1))
    if min_host_h > 13 or min_host_w > 13:
        return None

    host_h = random.randint(min_host_h, 13)
    host_w = random.randint(min_host_w, 13)

    if vertical_count > 0:
        max_v_span = max(3, min(5, host_w - 3))
        v_intervals = _sample_disjoint_intervals(host_w - 2, vertical_count, 3, max_v_span)
        v_ext = [(random.randint(2, 5), random.randint(2, 5)) for _ in range(vertical_count)]
    else:
        v_intervals = []
        v_ext = []

    if horizontal_count > 0:
        max_h_span = max(3, min(5, host_h - 3))
        h_intervals = _sample_disjoint_intervals(host_h - 2, horizontal_count, 3, max_h_span)
        h_ext = [(random.randint(2, 5), random.randint(2, 5)) for _ in range(horizontal_count)]
    else:
        h_intervals = []
        h_ext = []

    top_overhang = max((up for up, _ in v_ext), default=0)
    bottom_overhang = max((down for _, down in v_ext), default=0)
    left_overhang = max((left for left, _ in h_ext), default=0)
    right_overhang = max((right for _, right in h_ext), default=0)

    margin_top = random.randint(1, 3)
    margin_bottom = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    margin_right = random.randint(1, 3)

    height = margin_top + top_overhang + host_h + bottom_overhang + margin_bottom
    width = margin_left + left_overhang + host_w + right_overhang + margin_right
    if height > 30 or width > 30:
        return None

    ar0 = margin_top + top_overhang
    ac0 = margin_left + left_overhang
    ar1 = ar0 + host_h - 1
    ac1 = ac0 + host_w - 1

    cross_specs: list[tuple[int, int, int, int]] = []

    for (start, end), (up, down) in zip(v_intervals, v_ext):
        cross_specs.append((ar0 - up, ac0 + 1 + start, ar1 + down, ac0 + 1 + end))

    for (start, end), (left, right) in zip(h_intervals, h_ext):
        cross_specs.append((ar0 + 1 + start, ac0 - left, ar0 + 1 + end, ac1 + right))

    random.shuffle(cross_specs)
    rect_specs = [(ar0, ac0, ar1, ac1), *cross_specs]
    colors = random.sample(INPUT_COLORS, len(rect_specs))
    rects = [
        HiddenRect(color=color, r0=r0, c0=c0, r1=r1, c1=c1, diag_mode=random.randint(0, 1))
        for color, (r0, c0, r1, c1) in zip(colors, rect_specs)
    ]
    return (height, width), rects


def _validate_candidate(shape: tuple[int, int], rects: list[HiddenRect], force_pink: bool) -> bool:
    input_grid = _make_input(shape, rects)
    output_grid = _render_output(shape, rects)

    parsed = sorted(_parse_rectangles(input_grid), key=lambda rect: rect.color)
    expected = sorted(rects, key=lambda rect: rect.color)
    if parsed != expected:
        return False

    corner_cells = {corner for rect in rects for corner in rect.original_corners}
    reveal_counts = {rect.color: len(_reveal_cells_for_rect(rect, rects)) for rect in rects}
    hidden_counts = {
        rect.color: len(_boundary_cells(rect) - set(rect.original_corners)) - reveal_counts[rect.color]
        for rect in rects
    }

    if any(count <= 0 for count in reveal_counts.values()):
        return False
    if not any(count > 0 for count in hidden_counts.values()):
        return False

    visible_noncorner = sum(1 for (r, c), value in np.ndenumerate(output_grid) if value != BLACK and (r, c) not in corner_cells)
    if visible_noncorner < 10:
        return False

    pink_count = int(np.count_nonzero(output_grid == PINK))
    if force_pink and pink_count < 2:
        return False
    if not force_pink and pink_count != 0:
        return False

    nonpink_reveals = visible_noncorner - pink_count
    if nonpink_reveals < 6:
        return False

    if np.count_nonzero(output_grid) <= np.count_nonzero(input_grid):
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal only the boundary segments of hidden rectangles that pass through other rectangles' interiors.

    Input structure:
    - The grid has a black background and several foreground colors.
    - Each foreground color appears exactly twice as a pair of diagonal rectangle corners.
    - The rectangles themselves are not drawn; only their generating corner pixels are visible.

    Rule:
    - Reconstruct one axis-aligned one-cell-thick hollow rectangle from each same-color diagonal pair.
    - Keep the original corner pixels visible.
    - For every other boundary cell of a rectangle, reveal it in that rectangle's color only when the cell lies strictly inside at least one different hidden rectangle.
    - If boundary cells from two or more different rectangles are revealed on the same location, recolor that location pink.

    Key insight:
    - The visible arcs come from intersections between invisible rectangle frames and other rectangles' strict interiors, so the solver must infer every full rectangle before clipping and overlap recoloring.
    """
    rects = _parse_rectangles(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid).shape, rects)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        force_pink = random.random() < 0.7
        built = _build_rectangles(force_pink)
        if built is None:
            continue
        shape, rects = built
        if _validate_candidate(shape, rects, force_pink):
            return _make_input(shape, rects)
    raise RuntimeError("failed to generate a valid hidden-rectangle input")
