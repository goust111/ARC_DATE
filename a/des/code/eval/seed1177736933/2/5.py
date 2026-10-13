import random
from typing import Iterable

import numpy as np


BG = 8
GREY = 5
ORANGE = 7
MAROON = 9
LEGEND_H = 3
MARKER_COLORS = [1, 2, 3, 4, 6]
PATTERN_LIBRARY = [
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
]
GEN_RETRIES = 400


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            boxes.append((min(rs), min(cs), max(rs), max(cs)))
    return boxes


def _find_legend(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    h, w = grid.shape
    rows: list[int] = []
    for r in range(h - 1, -1, -1):
        vals = set(map(int, np.unique(grid[r])))
        if vals <= {BG, GREY, MAROON} and (GREY in vals or MAROON in vals):
            rows.append(r)
        elif rows:
            break
    rows.reverse()
    assert len(rows) == LEGEND_H, "legend must be a 3-row suffix band"

    sep_cols = [c for c in range(w) if np.all(grid[rows, c] == MAROON)]
    assert len(sep_cols) >= 2, "legend must have maroon separators"

    patterns: list[np.ndarray] = []
    for c0, c1 in zip(sep_cols, sep_cols[1:]):
        assert c1 - c0 == 4, "legend cells must be 3 columns wide"
        pat = grid[np.ix_(rows, np.arange(c0 + 1, c1))] == GREY
        assert pat.shape == (3, 3)
        assert np.any(pat), "legend patterns must be non-empty"
        patterns.append(pat)
    assert patterns, "legend must contain at least one pattern"
    return rows[0], patterns


def _tile_pattern(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None] % pattern.shape[0]
    cc = np.arange(width)[None, :] % pattern.shape[1]
    return pattern[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each marker-defined rectangle with the legend pattern indexed by its obstacle count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_r0, patterns = _find_legend(g)
    work = g[:legend_r0, :]

    marker_colors = [
        int(color)
        for color in np.unique(work)
        if color not in {BG, GREY, ORANGE, MAROON}
    ]
    assert marker_colors, "input must contain marker pairs"

    obstacle_boxes = _component_bboxes(work == ORANGE)
    for r0, c0, r1, c1 in obstacle_boxes:
        block = work[r0 : r1 + 1, c0 : c1 + 1]
        assert (r1 - r0, c1 - c0) == (1, 1) and np.all(block == ORANGE), "obstacles must be solid 2x2 blocks"

    out = g.copy()
    for color in marker_colors:
        pts = np.argwhere(work == color)
        assert len(pts) == 2, "each marker color must appear exactly twice"
        (r0, c0), (r1, c1) = map(tuple, pts)
        rr0, rr1 = sorted((int(r0), int(r1)))
        cc0, cc1 = sorted((int(c0), int(c1)))

        count = 0
        for box in obstacle_boxes:
            if _boxes_overlap((rr0, cc0, rr1, cc1), box):
                count += 1
        assert count < len(patterns), "obstacle count must index the legend"

        pattern = patterns[count]
        rh = rr1 - rr0 + 1
        rw = cc1 - cc0 + 1
        tiled = _tile_pattern(pattern, rh, rw)
        region = np.full((rh, rw), BG, dtype=int)
        region[tiled] = int(color)
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = region

    out[g == ORANGE] = ORANGE
    return out


def _count_choices(pattern_count: int, pair_count: int) -> list[int]:
    all_counts = list(range(pattern_count))
    if pair_count >= pattern_count:
        counts = all_counts[:]
    elif pair_count == 2:
        counts = [0, random.choice(all_counts[1:])]
    else:
        middle = all_counts[1:-1]
        need = pair_count - 2
        counts = [0, pattern_count - 1]
        if need > 0:
            counts.extend(random.sample(middle, need))
    random.shuffle(counts)
    return counts


def _size_for_count(count: int) -> tuple[int, int]:
    min_side = 4 + (count >= 2) + (count >= 4)
    max_side = min_side + 4
    return random.randint(min_side, max_side), random.randint(min_side, max_side + 1)


def _place_rectangles(
    main_h: int, width: int, counts: list[int]
) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    rects: list[tuple[int, int, int, int] | None] = [None] * len(counts)
    placed: list[tuple[int, int, int, int]] = []

    for idx in order:
        for _ in range(200):
            rh, rw = _size_for_count(counts[idx])
            if rh > main_h or rw > width:
                continue
            r0 = random.randint(0, main_h - rh)
            c0 = random.randint(0, width - rw)
            rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(_boxes_overlap(rect, other, pad=2) for other in placed):
                continue
            rects[idx] = rect
            placed.append(rect)
            break
        else:
            return None
    return [rect for rect in rects if rect is not None]


def _marker_positions(rect: tuple[int, int, int, int], diag: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if diag == 0:
        return (r0, c0), (r1, c1)
    return (r0, c1), (r1, c0)


def _block_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _fully_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 <= ir0 and ic0 >= oc0 and ir1 <= or1 and ic1 <= oc1


def _candidate_blocks(
    rect: tuple[int, int, int, int],
    rects: Iterable[tuple[int, int, int, int]],
    assigned_idx: int,
    main_h: int,
    width: int,
    occupied: np.ndarray,
    orange_boxes: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    r0, c0, r1, c1 = rect
    inside: list[tuple[int, int, int, int]] = []
    partial: list[tuple[int, int, int, int]] = []
    rect_list = list(rects)
    for br in range(r0 - 1, r1 + 1):
        for bc in range(c0 - 1, c1 + 1):
            box = (br, bc, br + 1, bc + 1)
            if br < 0 or bc < 0 or br + 1 >= main_h or bc + 1 >= width:
                continue
            if np.any(occupied[br : br + 2, bc : bc + 2]):
                continue
            if any(_boxes_overlap(box, other, pad=1) for other in orange_boxes):
                continue
            if not _boxes_overlap(box, rect):
                continue
            if any(_boxes_overlap(box, other) for j, other in enumerate(rect_list) if j != assigned_idx):
                continue
            if _fully_inside(box, rect):
                inside.append(box)
            else:
                partial.append(box)
    return inside, partial


def _place_obstacles(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    counts: list[int],
    markers: list[tuple[tuple[int, int], tuple[int, int]]],
    main_h: int,
    width: int,
) -> bool:
    occupied = np.zeros((main_h, width), dtype=bool)
    for pair in markers:
        for r, c in pair:
            occupied[r, c] = True

    orange_boxes: list[tuple[int, int, int, int]] = []
    for idx, rect in enumerate(rects):
        need = counts[idx]
        for n in range(need):
            inside, partial = _candidate_blocks(rect, rects, idx, main_h, width, occupied, orange_boxes)
            want_partial = (n == 0 and need > 0 and random.random() < 0.4) or (partial and not inside)
            options = partial if want_partial and partial else inside if inside else partial
            if not options:
                return False
            box = random.choice(options)
            orange_boxes.append(box)
            for r, c in _block_cells(box):
                occupied[r, c] = True
                grid[r, c] = ORANGE

    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        options = []
        for br in range(0, main_h - 1):
            for bc in range(0, width - 1):
                box = (br, bc, br + 1, bc + 1)
                if np.any(occupied[br : br + 2, bc : bc + 2]):
                    continue
                if any(_boxes_overlap(box, other, pad=1) for other in orange_boxes):
                    continue
                if any(_boxes_overlap(box, rect) for rect in rects):
                    continue
                options.append(box)
        if not options:
            break
        box = random.choice(options)
        orange_boxes.append(box)
        for r, c in _block_cells(box):
            occupied[r, c] = True
            grid[r, c] = ORANGE
    return True


def _draw_legend(grid: np.ndarray, start_row: int, left: int, patterns: list[np.ndarray]) -> None:
    for i in range(len(patterns) + 1):
        grid[start_row : start_row + LEGEND_H, left + 4 * i] = MAROON
    for i, pattern in enumerate(patterns):
        cols = slice(left + 4 * i + 1, left + 4 * i + 4)
        grid[start_row : start_row + LEGEND_H, cols][pattern] = GREY


def _gen_attempt() -> np.ndarray | None:
    pattern_count = random.randint(4, 5)
    pair_count = random.randint(2, min(4, len(MARKER_COLORS)))
    patterns = [pat.copy() for pat in random.sample(PATTERN_LIBRARY, pattern_count)]
    counts = _count_choices(pattern_count, pair_count)

    legend_w = 4 * pattern_count + 1
    width = random.randint(max(legend_w + 2, 18), 30)
    main_h = random.randint(12 + (pair_count >= 4), 20)
    total_h = main_h + LEGEND_H

    rects = _place_rectangles(main_h, width, counts)
    if rects is None:
        return None

    grid = np.full((total_h, width), BG, dtype=int)
    legend_left = random.randint(0, width - legend_w)
    _draw_legend(grid, main_h, legend_left, patterns)

    colors = random.sample(MARKER_COLORS, pair_count)
    diags = [random.randint(0, 1) for _ in range(pair_count)]
    markers = [_marker_positions(rect, diag) for rect, diag in zip(rects, diags)]
    for color, pair in zip(colors, markers):
        for r, c in pair:
            grid[r, c] = color

    if not _place_obstacles(grid, rects, counts, markers, main_h, width):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    for color in colors:
        assert np.count_nonzero(grid[:main_h] == color) == 2
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
