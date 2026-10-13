from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MOTIF_PALETTE = [BLUE, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]

GRID_H_RANGE = (15, 20)
GRID_W_RANGE = (18, 26)
MOTIF_H_RANGE = (3, 4)
MOTIF_W_RANGE = (4, 5)
SEED_COUNT_RANGE = (2, 4)
GEN_ATTEMPTS = 500


def _weak_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _pick_motif_counts(area: int, n_colors: int, rare_count: int) -> list[int] | None:
    other_count = n_colors - 1
    min_other = rare_count + 1
    extra = area - rare_count - other_count * min_other
    if extra < 0:
        return None
    extras = _weak_composition(extra, other_count)
    return [rare_count] + [min_other + value for value in extras]


def _unique_least_frequent_color(grid: np.ndarray, exclude: set[int] | None = None) -> int:
    values, counts = np.unique(grid, return_counts=True)
    candidates: list[tuple[int, int]] = []
    blocked = exclude or set()
    for value, count in zip(values, counts):
        value = int(value)
        if value not in blocked:
            candidates.append((value, int(count)))
    assert candidates, "expected at least one eligible color"
    least = min(count for _, count in candidates)
    winners = [value for value, count in candidates if count == least]
    assert len(winners) == 1, "expected a unique least-frequent color"
    return winners[0]


def _extract_motif(grid: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    grey_cells = np.argwhere(grid == GREY)
    assert grey_cells.size > 0, "expected a grey marker column"
    grey_cols = np.unique(grey_cells[:, 1])
    assert grey_cols.size == 1, "expected a single grey marker column"
    marker_col = int(grey_cols[0])

    marker_rows = np.sort(grey_cells[:, 0])
    marker_h = int(marker_rows.size)
    assert np.array_equal(marker_rows, np.arange(marker_h)), "marker column must start at row 0 and be contiguous"

    top_band = grid[:marker_h, marker_col + 1 :]
    motif_mask = (top_band != BLACK) & (top_band != GREY)
    motif_cols = np.flatnonzero(np.any(motif_mask, axis=0))
    assert motif_cols.size > 0, "expected a motif to the right of the marker column"
    expected = np.arange(int(motif_cols[0]), int(motif_cols[-1]) + 1)
    assert np.array_equal(motif_cols, expected), "motif columns must be contiguous"

    motif_c0 = marker_col + 1 + int(motif_cols[0])
    motif_c1 = marker_col + 1 + int(motif_cols[-1])
    motif = grid[:marker_h, motif_c0 : motif_c1 + 1].copy()
    return marker_h, marker_col, motif_c0, motif


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread the grey-bracketed motif downward from rare-color seed pixels until red blockers cut each stream off.

    Input structure:
    - The background is black.
    - A grey marker column at the upper left delimits the height of a rectangular motif immediately to its right.
    - The motif uses 2-3 non-black colors, and exactly one motif color is the unique least frequent color.
    - Outside the motif, isolated single pixels of that same rare color act as seeds, while red pixels act as blockers.

    Rule:
    - Find the motif height from the contiguous grey marker run, erase that marker column, and extract the rectangle to its right as motif M.
    - Within M, count every non-black non-grey color and identify the unique least frequent color C_rare.
    - Erase the whole original motif area to black, then for every external C_rare seed, start one row below it and copy motif rows downward cyclically with the seed column aligned to M's left edge.
    - If a row's write span contains any red blocker pixel from the input, stop that seed before writing the row; earlier painted pixels and blockers remain unchanged.
    - When multiple seed streams would overlap, process seeds by increasing column and never overwrite an already non-black output pixel.

    Key insight:
    - The least frequent motif color has a double role: inside the motif it is ordinary content, but outside the motif it marks the locations where the whole motif must be replayed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_h, marker_col, motif_c0, motif = _extract_motif(g)
    motif_h, motif_w = motif.shape
    rare = _unique_least_frequent_color(motif, exclude={BLACK, GREY})

    motif_area = np.zeros_like(g, dtype=bool)
    motif_area[:marker_h, marker_col : motif_c0 + motif_w] = True
    blocker_mask = (g == RED) & ~motif_area

    out = g.copy()
    out[motif_area] = BLACK

    seed_mask = (g == rare) & ~motif_area
    seeds = [(int(r), int(c)) for r, c in np.argwhere(seed_mask)]
    seeds.sort(key=lambda pos: (pos[1], pos[0]))

    height, width = g.shape
    for seed_r, seed_c in seeds:
        if seed_c + motif_w > width:
            continue
        for row_index, write_r in enumerate(range(seed_r + 1, height)):
            span = slice(seed_c, seed_c + motif_w)
            if np.any(blocker_mask[write_r, span]):
                break
            motif_row = motif[row_index % motif_h]
            for dc, color in enumerate(motif_row):
                if int(color) != BLACK and out[write_r, seed_c + dc] == BLACK:
                    out[write_r, seed_c + dc] = int(color)

    return out


def _generate_motif(height: int, width: int) -> tuple[np.ndarray, int]:
    area = height * width
    for _ in range(200):
        n_colors = random.randint(2, 3)
        rare_count = random.randint(1, 2)
        counts = _pick_motif_counts(area, n_colors, rare_count)
        if counts is None:
            continue
        colors = random.sample(MOTIF_PALETTE, n_colors)
        cells: list[int] = []
        for color, count in zip(colors, counts):
            cells.extend([color] * count)
        random.shuffle(cells)
        motif = np.array(cells, dtype=int).reshape(height, width)
        if _unique_least_frequent_color(motif) != colors[0]:
            continue
        return motif, colors[0]
    raise RuntimeError("failed to build a valid motif")


def _choose_seed_starts(width: int, motif_w: int, seed_count: int) -> list[int] | None:
    min_gap = 1
    required = seed_count * motif_w + (seed_count - 1) * min_gap
    extra = width - required
    if extra < 0:
        return None
    gaps = _weak_composition(extra, seed_count + 1)
    starts: list[int] = []
    pos = gaps[0]
    for i in range(seed_count):
        starts.append(pos)
        pos += motif_w
        if i < seed_count - 1:
            pos += min_gap + gaps[i + 1]
    return starts


def _pick_seed_rows(height: int, motif_h: int, seed_count: int) -> list[int] | None:
    low = motif_h + 2
    high = height - 3
    if high - low + 1 < seed_count:
        return None

    long_rows = [row for row in range(low, high + 1) if height - row - 2 >= motif_h + 1]
    if not long_rows:
        return None

    rows = [random.choice(long_rows)]
    pool = [row for row in range(low, high + 1) if row != rows[0]]
    rows.extend(random.sample(pool, seed_count - 1))
    random.shuffle(rows)
    return rows


def _pick_fill_lengths(seed_rows: list[int], height: int, motif_h: int) -> list[int] | None:
    seed_set = set(seed_rows)
    order = list(range(len(seed_rows)))
    random.shuffle(order)

    lengths = [0] * len(seed_rows)
    long_assigned = False
    short_assigned = False
    for index in order:
        row = seed_rows[index]
        max_fill = height - row - 2
        if max_fill < 1:
            return None

        candidates = [fill for fill in range(1, max_fill + 1) if row + fill + 1 not in seed_set]
        if not candidates:
            return None

        if not long_assigned:
            long_candidates = [fill for fill in candidates if fill >= motif_h + 1]
            if long_candidates:
                lengths[index] = random.choice(long_candidates)
                long_assigned = True
                continue

        if not short_assigned:
            short_candidates = [fill for fill in candidates if fill <= motif_h]
            if short_candidates:
                lengths[index] = random.choice(short_candidates)
                short_assigned = True
                continue

        lengths[index] = random.choice(candidates)

    if not long_assigned:
        return None
    if not short_assigned:
        return None
    return lengths


def _build_input() -> np.ndarray | None:
    motif_h = random.randint(*MOTIF_H_RANGE)
    motif_w = random.randint(*MOTIF_W_RANGE)
    seed_count = random.randint(*SEED_COUNT_RANGE)

    min_width = max(GRID_W_RANGE[0], seed_count * motif_w + (seed_count - 1))
    if min_width > GRID_W_RANGE[1]:
        return None
    width = random.randint(min_width, GRID_W_RANGE[1])
    height = random.randint(*GRID_H_RANGE)

    starts = _choose_seed_starts(width, motif_w, seed_count)
    seed_rows = _pick_seed_rows(height, motif_h, seed_count)
    if starts is None or seed_rows is None:
        return None

    fill_lengths = _pick_fill_lengths(seed_rows, height, motif_h)
    if fill_lengths is None:
        return None

    motif, rare = _generate_motif(motif_h, motif_w)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[:motif_h, 0] = GREY
    grid[:motif_h, 1 : 1 + motif_w] = motif

    intervals = list(zip(starts, [start + motif_w - 1 for start in starts]))
    for row, start in zip(seed_rows, starts):
        grid[row, start] = rare

    seed_row_set = set(seed_rows)
    blocker_rows: list[int] = []
    for row, start, fill_len in zip(seed_rows, starts, fill_lengths):
        blocker_r = row + fill_len + 1
        blocker_rows.append(blocker_r)
        blocker_count = random.randint(1, min(2, motif_w))
        blocker_cols = random.sample(range(start, start + motif_w), blocker_count)
        for col in blocker_cols:
            grid[blocker_r, col] = RED

        extra_rows = [rr for rr in range(blocker_r + 1, height) if rr not in seed_row_set]
        if extra_rows and random.random() < 0.65:
            extra_r = random.choice(extra_rows)
            extra_c = random.randint(start, start + motif_w - 1)
            grid[extra_r, extra_c] = RED

    free_columns = {
        c
        for c in range(width)
        if all(not (left <= c <= right) for left, right in intervals)
    }
    clutter_rows = [r for r in range(motif_h + 1, height) if r not in seed_row_set]
    clutter_budget = random.randint(0, 3)
    for _ in range(clutter_budget):
        if not free_columns or not clutter_rows:
            break
        r = random.choice(clutter_rows)
        c = random.choice(sorted(free_columns))
        grid[r, c] = RED

    if np.count_nonzero(grid == rare) != int(np.count_nonzero(motif == rare)) + seed_count:
        return None
    if np.count_nonzero(grid == GREY) != motif_h:
        return None
    if any(grid[row, start] != rare for row, start in zip(seed_rows, starts)):
        return None
    if any(blocker_r in seed_row_set for blocker_r in blocker_rows):
        return None

    solved = main(grid)
    if solved.shape != grid.shape:
        return None

    added = int(np.count_nonzero((solved != grid) & (solved != BLACK)))
    if added < motif_h * motif_w:
        return None

    long_streams = sum(fill_len >= motif_h + 1 for fill_len in fill_lengths)
    short_streams = sum(fill_len <= motif_h for fill_len in fill_lengths)
    if long_streams == 0 or short_streams == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
