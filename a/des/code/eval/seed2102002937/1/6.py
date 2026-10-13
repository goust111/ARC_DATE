import random

import numpy as np


BLACK = 0

# Generator configuration.
GENERATION_ATTEMPTS = 500
GRID_H_RANGE = (12, 19)
GRID_W_RANGE = (14, 22)
SEPARATOR_WIDTH_RANGE = (1, 2)
MAX_OVERLAP_RANGE = (2, 4)
EXTRA_LENS_MAX = 2
INNER_H_RANGE = (2, 4)
INNER_W_RANGE = (2, 5)
INSET_RANGE = (1, 2)
EXTRA_RECT_H_RANGE = (2, 4)
EXTRA_RECT_W_RANGE = (2, 5)
EXTRA_PLACEMENT_ATTEMPTS = 120
LEGEND_GAP_RANGE = (1, 2)
CHAIN_ATTEMPTS = 120


def _find_legend_column(grid: np.ndarray) -> int:
    h, w = grid.shape
    for c in range(w - 1, -1, -1):
        rows = np.flatnonzero(grid[:, c] != BLACK)
        if rows.size == 0:
            continue
        if np.all(np.diff(rows) > 1):
            return int(c)
    raise ValueError("legend column not found")


def _legend_colors(grid: np.ndarray, legend_c: int) -> list[int]:
    rows = np.flatnonzero(grid[:, legend_c] != BLACK)
    return [int(grid[r, legend_c]) for r in rows]


def _recover_rectangles(grid: np.ndarray, legend_c: int) -> list[tuple[int, int, int, int]]:
    work = np.array(grid, copy=True)
    work[:, legend_c] = BLACK

    rectangles: list[tuple[int, int, int, int]] = []
    for color in sorted(int(c) for c in np.unique(work) if int(c) != BLACK):
        coords = np.argwhere(work == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        rectangles.append((int(r0), int(c0), int(r1), int(c1)))
    return rectangles


def _coverage_counts(shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rectangles:
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
    return counts


def _render_from_rectangles(
    grid: np.ndarray,
    legend_c: int,
    legend_colors: list[int],
    rectangles: list[tuple[int, int, int, int]],
) -> np.ndarray:
    counts = _coverage_counts(grid.shape, rectangles)
    out = np.zeros_like(grid)
    if legend_colors:
        palette = np.array(legend_colors, dtype=int)
        mask = counts > 0
        out[mask] = palette[counts[mask] - 1]
    out[:, legend_c] = grid[:, legend_c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every pixel by lens overlap depth using the right-side legend.

    Input structure:
    - black background
    - several colored axis-aligned rectangular lenses, with one lens per color
    - a right-side legend column whose non-black pixels map depth 1, 2, 3, ...

    Rule:
    - recover each lens rectangle
    - count how many rectangles cover each pixel
    - paint depth k with the k-th legend color
    - keep the legend column unchanged
    """
    g = np.asarray(grid, dtype=int)
    legend_c = _find_legend_column(g)
    legend = _legend_colors(g, legend_c)
    rectangles = _recover_rectangles(g, legend_c)
    return _render_from_rectangles(g, legend_c, legend, rectangles)


def _rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _sample_chain(h: int, w: int, depth: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(CHAIN_ATTEMPTS):
        top_steps = [random.randint(*INSET_RANGE) for _ in range(depth - 1)]
        bottom_steps = [random.randint(*INSET_RANGE) for _ in range(depth - 1)]
        left_steps = [random.randint(*INSET_RANGE) for _ in range(depth - 1)]
        right_steps = [random.randint(*INSET_RANGE) for _ in range(depth - 1)]

        inner_h_max = min(INNER_H_RANGE[1], h - sum(top_steps) - sum(bottom_steps))
        inner_w_max = min(INNER_W_RANGE[1], w - sum(left_steps) - sum(right_steps))
        if inner_h_max < INNER_H_RANGE[0] or inner_w_max < INNER_W_RANGE[0]:
            continue

        inner_h = random.randint(INNER_H_RANGE[0], inner_h_max)
        inner_w = random.randint(INNER_W_RANGE[0], inner_w_max)

        outer_h = inner_h + sum(top_steps) + sum(bottom_steps)
        outer_w = inner_w + sum(left_steps) + sum(right_steps)
        if outer_h > h or outer_w > w:
            continue

        r0 = random.randint(0, h - outer_h)
        c0 = random.randint(0, w - outer_w)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        rects: list[tuple[int, int, int, int]] = []
        inset_top = inset_bottom = inset_left = inset_right = 0
        for i in range(depth):
            rects.append((r0 + inset_top, c0 + inset_left, r1 - inset_bottom, c1 - inset_right))
            if i + 1 < depth:
                inset_top += top_steps[i]
                inset_bottom += bottom_steps[i]
                inset_left += left_steps[i]
                inset_right += right_steps[i]
        return rects
    return None


def _mark_blocked(blocked: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_extra_rectangles(
    h: int,
    w: int,
    count: int,
    existing: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]] | None:
    if count == 0:
        return []

    blocked = np.zeros((h, w), dtype=bool)
    for rect in existing:
        _mark_blocked(blocked, rect)

    placed: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        rect = None
        for _ in range(EXTRA_PLACEMENT_ATTEMPTS):
            rh = random.randint(*EXTRA_RECT_H_RANGE)
            rw = random.randint(*EXTRA_RECT_W_RANGE)
            if rh > h or rw > w:
                continue
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw)
            cand = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if blocked[r0 : r0 + rh, c0 : c0 + rw].any():
                continue
            rect = cand
            break
        if rect is None:
            return None
        placed.append(rect)
        _mark_blocked(blocked, rect)
    return placed


def _visible_input(
    h: int,
    w: int,
    chain_rects: list[tuple[int, int, int, int]],
    extra_rects: list[tuple[int, int, int, int]],
    lens_colors: list[int],
    legend_colors: list[int],
    legend_rows: list[int],
    legend_c: int,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)

    for i in range(len(chain_rects) - 1):
        outer = _rect_mask((h, w), chain_rects[i])
        inner = _rect_mask((h, w), chain_rects[i + 1])
        grid[outer & (~inner)] = int(lens_colors[i])
    grid[_rect_mask((h, w), chain_rects[-1])] = int(lens_colors[len(chain_rects) - 1])

    for i, rect in enumerate(extra_rects, start=len(chain_rects)):
        grid[_rect_mask((h, w), rect)] = int(lens_colors[i])

    for row, color in zip(legend_rows, legend_colors):
        grid[row, legend_c] = int(color)
    return grid


def _legend_rows(h: int, depth: int) -> list[int] | None:
    gaps = [random.randint(*LEGEND_GAP_RANGE) for _ in range(depth - 1)]
    total_span = depth + sum(gaps)
    if total_span > h:
        return None
    start = random.randint(0, h - total_span)
    rows = [start]
    for gap in gaps:
        rows.append(rows[-1] + gap + 1)
    return rows


def _rectangles_match(
    recovered: list[tuple[int, int, int, int]],
    expected: list[tuple[int, int, int, int]],
) -> bool:
    return sorted(recovered) == sorted(expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    sep_width = random.randint(*SEPARATOR_WIDTH_RANGE)
    draw_w = w - sep_width - 1
    if draw_w < 6:
        return None

    max_overlap = random.randint(*MAX_OVERLAP_RANGE)
    max_lenses = min(max_overlap + EXTRA_LENS_MAX, 9 - max_overlap)
    if max_lenses < max_overlap:
        return None
    lens_count = random.randint(max_overlap, max_lenses)

    colors = random.sample(range(1, 10), lens_count + max_overlap)
    lens_colors = colors[:lens_count]
    legend_colors = colors[lens_count:]

    chain_rects = _sample_chain(h, draw_w, max_overlap)
    if chain_rects is None:
        return None

    extra_rects = _place_extra_rectangles(h, draw_w, lens_count - max_overlap, chain_rects)
    if extra_rects is None:
        return None

    legend_rows = _legend_rows(h, max_overlap)
    if legend_rows is None:
        return None
    legend_c = w - 1

    grid = _visible_input(
        h,
        w,
        chain_rects,
        extra_rects,
        lens_colors,
        legend_colors,
        legend_rows,
        legend_c,
    )

    rectangles = chain_rects + extra_rects
    counts = _coverage_counts((h, w), rectangles)
    if counts.max() != max_overlap:
        return None
    if any(not np.any(counts == depth) for depth in range(1, max_overlap + 1)):
        return None
    if np.any(counts[:, legend_c] != 0):
        return None

    recovered = _recover_rectangles(grid, legend_c)
    if not _rectangles_match(recovered, rectangles):
        return None

    expected = _render_from_rectangles(grid, legend_c, legend_colors, rectangles)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lens-overlap input")
