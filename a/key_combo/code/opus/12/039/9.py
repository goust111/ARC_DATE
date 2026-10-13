from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

MARKER_PALETTE = (BLUE, RED, GREEN, YELLOW, ORANGE)
GENERATION_ATTEMPTS = 400

_GEN_COUNTER = 0


def _find_teal_columns(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == TEAL)]


def _find_gray_row(grid: np.ndarray, teal_cols: list[int]) -> int:
    teal_col_set = set(teal_cols)
    matches = []
    for r in range(grid.shape[0]):
        ok = True
        for c in range(grid.shape[1]):
            target = TEAL if c in teal_col_set else GREY
            if int(grid[r, c]) != target:
                ok = False
                break
        if ok:
            matches.append(r)
    assert len(matches) == 1, "expected exactly one gray horizontal divider"
    return int(matches[0])


def _find_teal_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == TEAL)]


def _usable_band_sizes(height: int, divider_rows: list[int], gray_row: int) -> list[int] | None:
    boundaries = [-1] + sorted(divider_rows) + [height]
    sizes: list[int] = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        usable = sum(1 for r in range(lo + 1, hi) if r != gray_row)
        if usable <= 0:
            return None
        sizes.append(usable)
    return sizes


def _infer_region_rows(grid: np.ndarray, gray_row: int) -> int:
    height = int(grid.shape[0])
    teal_rows = _find_teal_rows(grid)
    candidates: list[tuple[bool, int]] = []
    for include_gray in (False, True):
        divider_rows = teal_rows + ([gray_row] if include_gray else [])
        sizes = _usable_band_sizes(height, divider_rows, gray_row)
        if sizes is not None and len(set(sizes)) == 1:
            candidates.append((include_gray, len(sizes)))
    assert len(candidates) == 1, "expected a unique region-row interpretation"
    return int(candidates[0][1])


def _marker_differences(grid: np.ndarray, gray_row: int) -> list[tuple[int, int]]:
    diffs = []
    for color in MARKER_PALETTE:
        up = int(np.count_nonzero(grid[:gray_row] == color))
        down = int(np.count_nonzero(grid[gray_row + 1 :] == color))
        if up or down:
            diffs.append((down - up, color))
    assert len(diffs) >= 3, "expected at least three marker colors"
    assert len({diff for diff, _color in diffs}) == len(diffs), "differences must be unique"
    return diffs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count partition rows/cols from the teal lattice, then fill the output by the strongest growth and shrink colors.

    Input structure:
    - Black background contains a teal lattice made of full teal columns and several full teal rows,
      dividing the canvas into a matrix of rectangular regions.
    - Exactly one gray horizontal divider crosses the grid; on teal column intersections the teal
      columns stay visible, so the gray line uses the non-column cells of that row.
    - Sparse single-pixel markers use colors from {blue, red, green, yellow, orange} and lie only
      inside the rectangular regions.

    Rule:
    - Count teal divider columns to get the number of region columns `C`.
    - Determine whether the gray row is also a lattice row by checking which interpretation yields
      equal usable band heights, then count region rows `R`.
    - For every marker color, count pixels above the gray row (`U`) and below it (`D`), and compute
      the growth score `G = D - U`.
    - Let the unique maximum-score color be the growth color and the unique minimum-score color be
      the shrink color.
    - Return an `R x C` grid whose top `ceil(R/2)` rows are the growth color and whose remaining
      rows are the shrink color.

    Key insight:
    - The teal lattice encodes output size, while the gray row defines a separate counting boundary;
      solving the task requires combining structural counting with above/below color comparison.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    teal_cols = _find_teal_columns(g)
    assert teal_cols, "expected at least one teal divider column"
    gray_row = _find_gray_row(g, teal_cols)
    region_rows = _infer_region_rows(g, gray_row)
    region_cols = len(teal_cols) + 1

    diffs = _marker_differences(g, gray_row)
    growth_color = max(diffs)[1]
    shrink_color = min(diffs)[1]

    top_rows = (region_rows + 1) // 2
    out = np.full((region_rows, region_cols), shrink_color, dtype=int)
    out[:top_rows, :] = growth_color
    return out


def _uniform_candidate_kind(grid: np.ndarray) -> tuple[str, int]:
    teal_cols = _find_teal_columns(grid)
    gray_row = _find_gray_row(grid, teal_cols)
    teal_rows = _find_teal_rows(grid)

    results = []
    for include_gray, kind in ((False, "independent"), (True, "overlap")):
        divider_rows = teal_rows + ([gray_row] if include_gray else [])
        sizes = _usable_band_sizes(grid.shape[0], divider_rows, gray_row)
        if sizes is not None and len(set(sizes)) == 1:
            results.append((kind, len(sizes)))
    assert len(results) == 1
    return results[0]


def _branch_profile() -> tuple[str, int]:
    global _GEN_COUNTER
    mode = _GEN_COUNTER % 4
    _GEN_COUNTER += 1
    kind = "overlap" if mode < 2 else "independent"
    parity = 1 if mode % 2 == 0 else 0
    return kind, parity


def _choose_region_shape(parity: int) -> tuple[int, int, int, int]:
    if parity == 1:
        region_rows = random.choice((3, 5))
    else:
        region_rows = random.choice((4, 6))
    region_cols = random.randint(3, 6)
    cell_h = random.randint(2, 4)
    cell_w = random.randint(2, 4)
    return region_rows, region_cols, cell_h, cell_w


def _choose_gray_position(kind: str, region_rows: int, cell_h: int) -> tuple[int, int]:
    if kind == "overlap":
        split_after = random.randint(1, region_rows - 1)
        return split_after - 1, -1

    candidates = []
    for band in range(region_rows):
        for offset in range(1, cell_h):
            top_rows = band * cell_h + offset
            bottom_rows = (region_rows - band - 1) * cell_h + (cell_h - offset)
            if top_rows > 0 and bottom_rows > 0:
                candidates.append((abs(top_rows - bottom_rows), band, offset))
    _score, band, offset = random.choice(sorted(candidates)[: min(6, len(candidates))])
    return band, offset


def _build_geometry(
    kind: str,
    region_rows: int,
    region_cols: int,
    cell_h: int,
    cell_w: int,
    gray_band: int,
    gray_offset: int,
) -> tuple[np.ndarray, list[list[int]], list[list[int]], int]:
    row_groups: list[list[int]] = [[] for _ in range(region_rows)]
    row_types: list[str] = []
    overlap_divider = gray_band if kind == "overlap" else None

    for rr in range(region_rows):
        if kind == "independent":
            for local in range(cell_h):
                if rr == gray_band and local == gray_offset:
                    row_types.append("gray")
                row_groups[rr].append(len(row_types))
                row_types.append("region")
        else:
            for _ in range(cell_h):
                row_groups[rr].append(len(row_types))
                row_types.append("region")

        if rr < region_rows - 1:
            row_types.append("gray" if rr == overlap_divider else "teal")

    gray_row = row_types.index("gray")
    height = len(row_types)
    width = region_cols * cell_w + (region_cols - 1)
    grid = np.zeros((height, width), dtype=int)

    col_groups: list[list[int]] = []
    teal_cols = []
    col = 0
    for cc in range(region_cols):
        cols = list(range(col, col + cell_w))
        col_groups.append(cols)
        col += cell_w
        if cc < region_cols - 1:
            teal_cols.append(col)
            col += 1

    if teal_cols:
        grid[:, teal_cols] = TEAL

    for r, kind_name in enumerate(row_types):
        if kind_name == "teal":
            grid[r, :] = TEAL
        elif kind_name == "gray":
            for c in range(width):
                if c not in teal_cols:
                    grid[r, c] = GREY

    return grid, row_groups, col_groups, gray_row


def _region_cell_map(
    row_groups: list[list[int]],
    col_groups: list[list[int]],
    gray_row: int,
) -> tuple[dict[tuple[int, int], list[tuple[int, int]]], list[tuple[int, int, tuple[int, int]]], list[tuple[int, int, tuple[int, int]]]]:
    region_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    top_cells: list[tuple[int, int, tuple[int, int]]] = []
    bottom_cells: list[tuple[int, int, tuple[int, int]]] = []
    for rr, rows in enumerate(row_groups):
        for cc, cols in enumerate(col_groups):
            key = (rr, cc)
            cells: list[tuple[int, int]] = []
            for r in rows:
                for c in cols:
                    cells.append((r, c))
                    if r < gray_row:
                        top_cells.append((r, c, key))
                    elif r > gray_row:
                        bottom_cells.append((r, c, key))
            region_cells[key] = cells
    return region_cells, top_cells, bottom_cells


def _choose_color_counts(top_capacity: int, bottom_capacity: int) -> tuple[list[int], dict[int, tuple[int, int]]] | None:
    n_colors = 3 if min(top_capacity, bottom_capacity) < 8 else random.choice((3, 3, 4))
    colors = random.sample(list(MARKER_PALETTE), n_colors)

    diff_pool = (-2, -1, 0, 1, 2)
    for _ in range(80):
        diffs = random.sample(diff_pool, n_colors)
        if not any(diff > 0 for diff in diffs):
            continue
        if not any(diff < 0 for diff in diffs):
            continue

        counts: dict[int, tuple[int, int]] = {}
        total_top = 0
        total_bottom = 0
        ok = True
        for color, diff in zip(colors, diffs):
            low_u = max(1, 1 - diff)
            high_u = 3
            if low_u > high_u:
                ok = False
                break
            up = random.randint(low_u, high_u)
            down = up + diff
            counts[color] = (up, down)
            total_top += up
            total_bottom += down
        if not ok:
            continue
        if total_top > top_capacity or total_bottom > bottom_capacity:
            continue
        return colors, counts
    return None


def _place_markers(
    grid: np.ndarray,
    top_cells: list[tuple[int, int, tuple[int, int]]],
    bottom_cells: list[tuple[int, int, tuple[int, int]]],
    counts: dict[int, tuple[int, int]],
    force_split_band: tuple[list[tuple[int, int, tuple[int, int]]], list[tuple[int, int, tuple[int, int]]]] | None,
) -> bool:
    used: set[tuple[int, int]] = set()
    region_load: Counter[tuple[int, int]] = Counter()

    def choose_from(pool: list[tuple[int, int, tuple[int, int]]]) -> tuple[int, int, tuple[int, int]] | None:
        eligible = [cell for cell in pool if cell[:2] not in used and region_load[cell[2]] < 3]
        if not eligible:
            return None
        min_load = min(region_load[cell[2]] for cell in eligible)
        best = [cell for cell in eligible if region_load[cell[2]] == min_load]
        return random.choice(best)

    if force_split_band is not None:
        forced_top, forced_bottom = force_split_band
        top_pick = choose_from(forced_top)
        bottom_pick = choose_from(forced_bottom)
        if top_pick is None or bottom_pick is None:
            return False
        for (r, c, _key), color in zip((top_pick, bottom_pick), random.sample(list(counts), 2)):
            grid[r, c] = color
            used.add((r, c))
            region_load[_key] += 1
            up, down = counts[color]
            if r < bottom_pick[0]:
                counts[color] = (up - 1, down)
            else:
                counts[color] = (up, down - 1)

    top_sequence = [color for color, (up, _down) in counts.items() for _ in range(up)]
    bottom_sequence = [color for color, (_up, down) in counts.items() for _ in range(down)]
    random.shuffle(top_sequence)
    random.shuffle(bottom_sequence)

    for sequence, pool in ((top_sequence, top_cells), (bottom_sequence, bottom_cells)):
        for color in sequence:
            pick = choose_from(pool)
            if pick is None:
                return False
            r, c, key = pick
            grid[r, c] = color
            used.add((r, c))
            region_load[key] += 1
    return all(load <= 3 for load in region_load.values())


def _expected_output(region_rows: int, region_cols: int, counts: dict[int, tuple[int, int]]) -> np.ndarray:
    diffs = [(down - up, color) for color, (up, down) in counts.items()]
    growth = max(diffs)[1]
    shrink = min(diffs)[1]
    top_rows = (region_rows + 1) // 2
    out = np.full((region_rows, region_cols), shrink, dtype=int)
    out[:top_rows, :] = growth
    return out


def _count_region_markers(
    grid: np.ndarray,
    region_cells: dict[tuple[int, int], list[tuple[int, int]]],
) -> dict[tuple[int, int], int]:
    counts = {}
    for key, cells in region_cells.items():
        counts[key] = sum(1 for r, c in cells if int(grid[r, c]) in MARKER_PALETTE)
    return counts


def _gen_attempt() -> np.ndarray | None:
    kind, parity = _branch_profile()
    region_rows, region_cols, cell_h, cell_w = _choose_region_shape(parity)
    gray_band, gray_offset = _choose_gray_position(kind, region_rows, cell_h)

    grid, row_groups, col_groups, gray_row = _build_geometry(
        kind, region_rows, region_cols, cell_h, cell_w, gray_band, gray_offset
    )
    region_cells, top_cells, bottom_cells = _region_cell_map(row_groups, col_groups, gray_row)

    top_capacity = min(len(top_cells), 3 * len(region_cells))
    bottom_capacity = min(len(bottom_cells), 3 * len(region_cells))
    color_setup = _choose_color_counts(top_capacity, bottom_capacity)
    if color_setup is None:
        return None
    colors, planned_counts = color_setup
    counts = {color: value for color, value in planned_counts.items()}

    split_force = None
    if kind == "independent":
        split_top = []
        split_bottom = []
        for cc in range(region_cols):
            key = (gray_band, cc)
            for r, c in region_cells[key]:
                if r < gray_row:
                    split_top.append((r, c, key))
                elif r > gray_row:
                    split_bottom.append((r, c, key))
        if split_top and split_bottom and len(colors) >= 2:
            split_force = (split_top, split_bottom)

    if not _place_markers(grid, top_cells, bottom_cells, counts, split_force):
        return None

    actual = {}
    for color in MARKER_PALETTE:
        up = int(np.count_nonzero(grid[:gray_row] == color))
        down = int(np.count_nonzero(grid[gray_row + 1 :] == color))
        if up or down:
            actual[color] = (up, down)
    if set(actual) != set(planned_counts):
        return None
    if actual != planned_counts:
        return None

    per_region = _count_region_markers(grid, region_cells)
    if max(per_region.values()) > 3:
        return None
    if not any(count == 0 for count in per_region.values()):
        return None

    inferred_kind, inferred_rows = _uniform_candidate_kind(grid)
    if inferred_kind != kind or inferred_rows != region_rows:
        return None

    diffs = [(down - up, color) for color, (up, down) in actual.items()]
    if len(diffs) < 3 or len({diff for diff, _ in diffs}) != len(diffs):
        return None
    if min(diff for diff, _ in diffs) >= 0 or max(diff for diff, _ in diffs) <= 0:
        return None

    expected = _expected_output(region_rows, region_cols, actual)
    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid grid for puzzle 9")
