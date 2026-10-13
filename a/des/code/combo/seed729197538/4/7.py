from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BACKGROUND = 0
LINE_COLOR = 5
EVEN_LOCAL_COLORS = (1, 2)
ODD_LOCAL_COLORS = (6, 7)
SPY_COLORS = (3, 4, 8, 9)

# --------------- generator config ---------------

MIN_REGIONS_PER_AXIS = 2
MAX_REGIONS_PER_AXIS = 4
REGION_SIZE_MIN = 5
REGION_SIZE_MAX = 8
GRID_SIDE_MIN = 13
GRID_SIDE_MAX = 32
MAX_SPY_DEGREE = 2
GENERATION_RETRIES = 400


def _contiguous_runs(indices: list[int] | np.ndarray) -> list[tuple[int, int]]:
    vals = [int(x) for x in indices]
    if not vals:
        return []

    runs: list[tuple[int, int]] = []
    start = vals[0]
    prev = vals[0]
    for value in vals[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = value
        prev = value
    runs.append((start, prev))
    return runs


def _line_color(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    best = None
    for color in np.unique(grid):
        full_rows = np.flatnonzero(np.all(grid == int(color), axis=1))
        full_cols = np.flatnonzero(np.all(grid == int(color), axis=0))
        if full_rows.size < 2 or full_cols.size < 2:
            continue

        row_runs = _contiguous_runs(full_rows)
        col_runs = _contiguous_runs(full_cols)
        if len(row_runs) < 2 or len(col_runs) < 2:
            continue

        score = (
            len(row_runs) + len(col_runs),
            int(full_rows.size + full_cols.size),
            int(np.sum(grid == int(color))),
        )
        if best is None or score > best[0]:
            best = (score, int(color), full_rows, full_cols)

    assert best is not None, "expected a full-row/full-column line color"
    return best[1], best[2], best[3]


def _region_spans(line_indices: np.ndarray, limit: int) -> list[tuple[int, int]]:
    runs = _contiguous_runs(line_indices)
    assert runs, "expected boundary lines"
    assert runs[0][0] == 0, "expected an outer boundary at the start"
    assert runs[-1][1] == limit - 1, "expected an outer boundary at the end"

    spans: list[tuple[int, int]] = []
    for (_, prev_end), (next_start, _) in zip(runs, runs[1:], strict=False):
        gap_start = prev_end + 1
        gap_end = next_start - 1
        if gap_start <= gap_end:
            spans.append((gap_start, gap_end))

    assert spans, "expected at least one interior region span"
    return spans


def _neighbors(region_rows: int, region_cols: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < region_rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < region_cols:
        out.append((r, c + 1))
    return out


def _majority_color(counts: Counter[int]) -> int:
    assert counts, "expected at least one surviving vote color"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if int(count) == int(best)]
    assert len(winners) == 1, "expected a unique majority color"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangle using the majority color after removing spy colors
    shared with adjacent rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    line_color, row_lines, col_lines = _line_color(g)
    row_spans = _region_spans(row_lines, g.shape[0])
    col_spans = _region_spans(col_lines, g.shape[1])

    region_rows = len(row_spans)
    region_cols = len(col_spans)

    counts_grid: list[list[Counter[int]]] = [
        [Counter() for _ in range(region_cols)] for _ in range(region_rows)
    ]
    color_sets: list[list[set[int]]] = [
        [set() for _ in range(region_cols)] for _ in range(region_rows)
    ]

    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(patch == int(line_color)), "region interior must exclude line cells"
            votes = patch[(patch != BACKGROUND) & (patch != int(line_color))]
            region_counts = Counter(map(int, votes.tolist()))
            assert region_counts, "expected colored pixels inside every region"
            counts_grid[rr][cc] = region_counts
            color_sets[rr][cc] = set(region_counts)

    out = g.copy()
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            region_counts = counts_grid[rr][cc]
            shared = set()
            for nr, nc in _neighbors(region_rows, region_cols, rr, cc):
                shared |= color_sets[rr][cc] & color_sets[nr][nc]

            filtered = Counter(
                {
                    int(color): int(count)
                    for color, count in region_counts.items()
                    if int(color) not in shared
                }
            )
            fill_color = _majority_color(filtered)
            out[r0 : r1 + 1, c0 : c1 + 1] = fill_color

    return out


def _line_positions(sizes: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
    lines = [0]
    spans: list[tuple[int, int]] = []
    cursor = 1
    for size in sizes:
        start = cursor
        end = cursor + int(size) - 1
        spans.append((start, end))
        cursor = end + 1
        lines.append(cursor)
        cursor += 1
    return lines, spans


def _edge_list(region_rows: int, region_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for rr in range(region_rows):
        for cc in range(region_cols):
            if rr + 1 < region_rows:
                edges.append(((rr, cc), (rr + 1, cc)))
            if cc + 1 < region_cols:
                edges.append(((rr, cc), (rr, cc + 1)))
    return edges


def _choose_spy_edges(
    region_rows: int, region_cols: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = _edge_list(region_rows, region_cols)
    random.shuffle(edges)

    max_target = min(len(edges), max(1, region_rows * region_cols))
    target = random.randint(1, max_target)
    degree: defaultdict[tuple[int, int], int] = defaultdict(int)
    chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for edge in edges:
        a, b = edge
        if degree[a] >= MAX_SPY_DEGREE or degree[b] >= MAX_SPY_DEGREE:
            continue
        chosen.append(edge)
        degree[a] += 1
        degree[b] += 1
        if len(chosen) == target:
            break

    assert chosen, "expected at least one spy edge"
    return chosen


def _candidate_slots(
    r0: int, r1: int, c0: int, c1: int
) -> list[list[tuple[int, int]]]:
    slots: list[list[tuple[int, int]]] = []
    for roff in (0, 1):
        rows = list(range(r0 + roff, r1 + 1, 2))
        if not rows:
            continue
        for coff in (0, 1):
            cols = list(range(c0 + coff, c1 + 1, 2))
            if not cols:
                continue
            slots.append([(r, c) for r in rows for c in cols])
    return slots


def _expected_output(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    winners: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(winners[(rr, cc)])
    return out


def _attempt() -> np.ndarray | None:
    region_rows = random.randint(MIN_REGIONS_PER_AXIS, MAX_REGIONS_PER_AXIS)
    region_cols = random.randint(MIN_REGIONS_PER_AXIS, MAX_REGIONS_PER_AXIS)
    row_sizes = [random.randint(REGION_SIZE_MIN, REGION_SIZE_MAX) for _ in range(region_rows)]
    col_sizes = [random.randint(REGION_SIZE_MIN, REGION_SIZE_MAX) for _ in range(region_cols)]

    height = sum(row_sizes) + region_rows + 1
    width = sum(col_sizes) + region_cols + 1
    if not (GRID_SIDE_MIN <= height <= GRID_SIDE_MAX and GRID_SIDE_MIN <= width <= GRID_SIDE_MAX):
        return None

    row_lines, row_spans = _line_positions(row_sizes)
    col_lines, col_spans = _line_positions(col_sizes)

    winners: dict[tuple[int, int], int] = {}
    region_counts: dict[tuple[int, int], Counter[int]] = {}
    for rr in range(region_rows):
        for cc in range(region_cols):
            local_palette = EVEN_LOCAL_COLORS if (rr + cc) % 2 == 0 else ODD_LOCAL_COLORS
            if random.random() < 0.5:
                winner, decoy = int(local_palette[0]), int(local_palette[1])
            else:
                winner, decoy = int(local_palette[1]), int(local_palette[0])

            winners[(rr, cc)] = winner
            region_counts[(rr, cc)] = Counter({winner: 2, decoy: 1})

    spy_edges = _choose_spy_edges(region_rows, region_cols)
    for edge in spy_edges:
        a, b = edge
        spy_color = random.choice(SPY_COLORS)
        region_counts[a][int(spy_color)] += random.randint(1, 2)
        region_counts[b][int(spy_color)] += random.randint(1, 2)

    regions_with_spies = [
        key
        for key, counts in region_counts.items()
        if any(int(color) in SPY_COLORS for color in counts)
    ]
    assert regions_with_spies, "spy edges must affect at least one region"

    chosen_region = random.choice(regions_with_spies)
    spy_colors_here = [int(color) for color in region_counts[chosen_region] if int(color) in SPY_COLORS]
    boosted = random.choice(spy_colors_here)
    region_counts[chosen_region][boosted] = max(int(region_counts[chosen_region][boosted]), 3)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[row_lines, :] = LINE_COLOR
    grid[:, col_lines] = LINE_COLOR

    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            counts = region_counts[(rr, cc)]
            total_votes = sum(int(v) for v in counts.values())
            candidate_sets = [
                cells for cells in _candidate_slots(r0, r1, c0, c1) if len(cells) >= total_votes
            ]
            if not candidate_sets:
                return None

            cells = random.choice(candidate_sets)
            chosen_cells = random.sample(cells, total_votes)
            random.shuffle(chosen_cells)

            cursor = 0
            for color, count in counts.items():
                for r, c in chosen_cells[cursor : cursor + int(count)]:
                    grid[r, c] = int(color)
                cursor += int(count)

    expected = _expected_output(grid, row_spans, col_spans, winners)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spy-voting grid")
