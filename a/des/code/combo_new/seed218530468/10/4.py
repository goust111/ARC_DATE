from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = list(range(10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LAYOUT_CHOICES = [(2, 3), (3, 2), (2, 4), (3, 3)]
REGION_SIZE_RANGE = (5, 9)
PATTERN_SIZE_MAX = 5
MIN_PATTERN_AREA = 6
PATTERN_FILL_RANGE = (0.4, 0.7)
GENERATION_ATTEMPTS = 400
MASK_ATTEMPTS = 80
NOISE_ASSIGN_ATTEMPTS = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "expected a unique dominant background"
    return int(winners[0])


def _majority_value(values) -> int:
    counts = Counter(int(v) for v in values)
    best = max(counts.values())
    winners = [value for value, count in counts.items() if count == best]
    assert len(winners) == 1, "majority vote must be unambiguous"
    return int(winners[0])


def _positive_intervals(lines: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(int(line) for line in lines), int(length)]
    intervals = []
    for left, right in zip(bounds, bounds[1:]):
        start, end = int(left) + 1, int(right)
        if start < end:
            intervals.append((start, end))
    return intervals


def _find_layout(
    grid: np.ndarray,
) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    candidates = []

    for color in (int(v) for v in np.unique(g) if int(v) != bg):
        row_lines = [r for r in range(h) if bool(np.all(g[r, :] == color))]
        col_lines = [c for c in range(w) if bool(np.all(g[:, c] == color))]
        if not row_lines or not col_lines:
            continue

        row_intervals = _positive_intervals(row_lines, h)
        col_intervals = _positive_intervals(col_lines, w)
        if len(row_intervals) < 2 or len(col_intervals) < 2:
            continue

        row_sizes = {r1 - r0 for r0, r1 in row_intervals}
        col_sizes = {c1 - c0 for c0, c1 in col_intervals}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        expected = np.zeros_like(g, dtype=bool)
        expected[row_lines, :] = True
        expected[:, col_lines] = True
        if not np.array_equal(g == color, expected):
            continue

        candidates.append((color, row_lines, col_lines, row_intervals, col_intervals))

    assert len(candidates) == 1, "expected a unique separator color and regular region grid"
    return candidates[0]


def _content_bbox(region: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    mask = np.asarray(region, dtype=int) != int(bg)
    assert bool(mask.any()), "every region must contain motif pixels"
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise the centered motif shared across separator-delimited regions and write the consensus back into every region.

    Input structure:
    - The background is the most common color.
    - One non-background color forms complete one-pixel separator rows and columns that split the grid into equal-sized rectangular regions.
    - Each region contains a centered motif inside a shared local bounding box; motif pixels may use any non-background, non-separator colors.
    - Every region is a noisy copy of the same motif, differing at only a small number of local cells.

    Rule:
    - Identify the common region layout from the separator rows and columns.
    - Infer the shared motif bounding box by majority vote over the noisy region-local content bounds.
    - For each local cell inside that box, take the most frequent color across aligned regions to reconstruct the consensus motif.
    - Keep separator lines unchanged, clear each region to background, and place the consensus motif back at the same centered location in every region.

    Key insight:
    The correct output is not copied from any one panel; it is the per-cell consensus obtained by aligning all noisy region copies.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    _sep, _rows, _cols, row_intervals, col_intervals = _find_layout(g)

    regions = [g[r0:r1, c0:c1] for r0, r1 in row_intervals for c0, c1 in col_intervals]
    assert regions, "expected at least one region"
    region_h, region_w = regions[0].shape
    assert all(region.shape == (region_h, region_w) for region in regions)

    bboxes = [_content_bbox(region, bg) for region in regions]
    top = _majority_value(box[0] for box in bboxes)
    left = _majority_value(box[1] for box in bboxes)
    bottom = _majority_value(box[2] for box in bboxes)
    right = _majority_value(box[3] for box in bboxes)
    assert 0 <= top <= bottom < region_h
    assert 0 <= left <= right < region_w

    consensus = np.full((bottom - top + 1, right - left + 1), bg, dtype=int)
    for rr in range(consensus.shape[0]):
        for cc in range(consensus.shape[1]):
            values = [int(region[top + rr, left + cc]) for region in regions]
            consensus[rr, cc] = _majority_value(values)

    out = g.copy()
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            out[r0:r1, c0:c1] = bg
            out[r0 + top : r0 + bottom + 1, c0 + left : c0 + right + 1] = consensus
    return out


def _grid_dim(count: int, region_size: int, with_outer_frame: bool) -> int:
    return count * region_size + (count + 1 if with_outer_frame else count - 1)


def _separator_positions(count: int, region_size: int, with_outer_frame: bool) -> list[int]:
    if with_outer_frame:
        step = region_size + 1
        return [idx * step for idx in range(count + 1)]
    return [(idx + 1) * region_size + idx for idx in range(count - 1)]


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _random_connected_mask(h: int, w: int, on_count: int) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        frontier = set(_neighbors4(seed[0], seed[1], h, w))
        while frontier and len(cells) < on_count:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(cell[0], cell[1], h, w):
                if nb not in cells:
                    frontier.add(nb)
        if len(cells) != on_count:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if mask.all():
            continue
        return mask
    return None


def _choose_pattern_mask(region_h: int, region_w: int, region_count: int) -> np.ndarray | None:
    max_cell_flips = (region_count - 1) // 2
    if max_cell_flips <= 0:
        return None

    for _ in range(MASK_ATTEMPTS):
        ph = random.randint(2, min(PATTERN_SIZE_MAX, region_h - 2))
        pw = random.randint(2, min(PATTERN_SIZE_MAX, region_w - 2))
        area = ph * pw
        if area < MIN_PATTERN_AREA:
            continue

        min_on = max(4, int(np.ceil(area * PATTERN_FILL_RANGE[0])))
        max_on = min(area - 1, int(np.floor(area * PATTERN_FILL_RANGE[1])))
        if min_on > max_on:
            continue

        on_count = random.randint(min_on, max_on)
        mask = _random_connected_mask(ph, pw, on_count)
        if mask is None:
            continue

        total_noise = sum(1 + random.randrange(2) for _ in range(region_count))
        if total_noise > area * max_cell_flips:
            continue
        return mask
    return None


def _assign_noise_cells(mask_shape: tuple[int, int], noise_counts: list[int], max_usage: int) -> list[list[tuple[int, int]]] | None:
    h, w = mask_shape
    positions = [(r, c) for r in range(h) for c in range(w)]
    region_order = sorted(range(len(noise_counts)), key=lambda idx: noise_counts[idx], reverse=True)

    for _ in range(NOISE_ASSIGN_ATTEMPTS):
        usage = {pos: 0 for pos in positions}
        assignments: list[list[tuple[int, int]] | None] = [None] * len(noise_counts)
        ok = True
        for idx in region_order:
            need = noise_counts[idx]
            chosen = []
            for _slot in range(need):
                candidates = [pos for pos in positions if usage[pos] < max_usage and pos not in chosen]
                if not candidates:
                    ok = False
                    break
                weights = [max_usage - usage[pos] for pos in candidates]
                pick = random.choices(candidates, weights=weights, k=1)[0]
                chosen.append(pick)
            if not ok:
                break
            for pos in chosen:
                usage[pos] += 1
            assignments[idx] = chosen
        if ok and all(item is not None for item in assignments):
            return [item for item in assignments if item is not None]
    return None


def _layout_records(
    row_count: int,
    col_count: int,
    region_h: int,
    region_w: int,
    with_outer_frame: bool,
) -> tuple[int, int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    height = _grid_dim(row_count, region_h, with_outer_frame)
    width = _grid_dim(col_count, region_w, with_outer_frame)
    row_lines = _separator_positions(row_count, region_h, with_outer_frame)
    col_lines = _separator_positions(col_count, region_w, with_outer_frame)
    row_intervals = _positive_intervals(row_lines, height)
    col_intervals = _positive_intervals(col_lines, width)
    return height, width, row_lines, col_lines, row_intervals, col_intervals


def _make_clean_output(
    bg: int,
    sep: int,
    fg: int,
    row_lines: list[int],
    col_lines: list[int],
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    height = max(max(row_lines, default=-1), max((r1 - 1 for _, r1 in row_intervals), default=-1)) + 1
    width = max(max(col_lines, default=-1), max((c1 - 1 for _, c1 in col_intervals), default=-1)) + 1
    out = np.full((height, width), bg, dtype=int)
    out[row_lines, :] = sep
    out[:, col_lines] = sep

    ph, pw = mask.shape
    region_h = row_intervals[0][1] - row_intervals[0][0]
    region_w = col_intervals[0][1] - col_intervals[0][0]
    top = (region_h - ph) // 2
    left = (region_w - pw) // 2

    patch = np.where(mask, fg, bg).astype(int)
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            out[r0 + top : r0 + top + ph, c0 + left : c0 + left + pw] = patch
    return out, (top, left)


def _gen_attempt() -> np.ndarray | None:
    row_count, col_count = random.choice(LAYOUT_CHOICES)
    with_outer_frame = random.choice([False, True])
    region_h = random.randint(*REGION_SIZE_RANGE)
    region_w = random.randint(*REGION_SIZE_RANGE)
    height, width, row_lines, col_lines, row_intervals, col_intervals = _layout_records(
        row_count, col_count, region_h, region_w, with_outer_frame
    )
    if height > 30 or width > 30:
        return None

    region_count = row_count * col_count
    mask = _choose_pattern_mask(region_h, region_w, region_count)
    if mask is None:
        return None

    colors = random.sample(PALETTE, 3)
    bg, sep, fg = colors
    clean, (top, left) = _make_clean_output(bg, sep, fg, row_lines, col_lines, row_intervals, col_intervals, mask)

    values, counts = np.unique(clean, return_counts=True)
    if int(values[np.argmax(counts)]) != bg or np.count_nonzero(counts == counts.max()) != 1:
        return None

    inp = clean.copy()
    noise_counts = [1 + random.randrange(2) for _ in range(region_count)]
    assignments = _assign_noise_cells(mask.shape, noise_counts, (region_count - 1) // 2)
    if assignments is None:
        return None

    regions = [(r0, c0) for r0, _ in row_intervals for c0, _ in col_intervals]
    for idx, (r0, c0) in enumerate(regions):
        for rr, cc in assignments[idx]:
            ar, ac = r0 + top + rr, c0 + left + cc
            inp[ar, ac] = bg if inp[ar, ac] == fg else fg

    if not np.array_equal(main(inp), clean):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
