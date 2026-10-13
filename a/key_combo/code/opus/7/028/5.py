from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


SEP = 5
BG = 0
ROW_COLORS = (1, 2, 3)
DISTRACTORS = (4, 7)

TILE_ROWS_RANGE = (3, 5)
TILE_COLS_RANGE = (3, 5)
GROUP_COUNT_RANGE = (2, 4)
MIN_SPAN = 3
MAX_GENERATION_ATTEMPTS = 400
MAX_DISTRACTORS_PER_ROW = 2


def _runs(active: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return runs


def _lattice_runs(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    assert np.all(grid[0, :] == SEP) and np.all(grid[-1, :] == SEP), "missing separator border rows"
    assert np.all(grid[:, 0] == SEP) and np.all(grid[:, -1] == SEP), "missing separator border cols"

    row_runs = _runs([not np.all(grid[r, :] == SEP) for r in range(grid.shape[0])])
    col_runs = _runs([not np.all(grid[:, c] == SEP) for c in range(grid.shape[1])])
    assert row_runs and col_runs, "expected a non-empty tile lattice"
    assert all(end - start == 3 for start, end in row_runs), "expected 3-cell tile interiors"
    assert all(end - start == 3 for start, end in col_runs), "expected 3-cell tile interiors"
    return row_runs, col_runs


def _row_color(row: np.ndarray) -> int:
    counts = Counter(int(v) for v in row if int(v) not in (BG, SEP))
    assert counts, "every non-separator row must contain a visible color"

    max_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == max_count]
    if len(candidates) > 1:
        first_col = {color: int(np.flatnonzero(row == color)[0]) for color in candidates}
        leftmost = min(first_col.values())
        candidates = [color for color in candidates if first_col[color] == leftmost]

    assert len(candidates) == 1, "row main color must be unique after tie-break"
    return candidates[0]


def _invert_rows(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    out = grid.copy()
    row_colors: dict[int, int] = {}
    for r in range(grid.shape[0]):
        if np.all(grid[r, :] == SEP):
            continue
        color = _row_color(grid[r])
        row_colors[r] = color
        row = grid[r]
        out[r] = np.where(row == color, BG, row)
        out[r] = np.where(row == BG, color, out[r])
    return out, row_colors


def _pixel_from_lane(
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
    tile_r: int,
    tile_c: int,
    rel_r: int,
    rel_c: int,
) -> tuple[int, int]:
    return row_runs[tile_r][0] + rel_r, col_runs[tile_c][0] + rel_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal hidden same-color tile markers by row inversion, then connect them across the tile lattice.

    Input structure:
    - Grey separator rows and columns partition the canvas into a regular lattice of 3x3 tile interiors.
    - Every non-separator row contains one dominant non-black, non-grey main color, plus optional distractor colors.
    - Some black cells inside tile interiors are hidden markers that sit on otherwise main-color rows.

    Rule:
    - For each non-separator row independently, find the main color as the most frequent non-black, non-grey color; if tied, choose the tied color whose leftmost run starts first.
    - In that row, swap the main color with black while leaving grey separators and all other colors unchanged.
    - Treat cells that were black before the swap and become colored after it as revealed endpoints. Group them by revealed color and by their relative 3x3 tile position.
    - Whenever a group has endpoints sharing a tile row or a tile column, fill every intervening tile at that same relative position with the group's color, overwriting any existing non-separator content there.

    Key insight:
    - The colored endpoints are encoded as black holes in main-color rows, so the solver must first perform the row-wise inversion and only then read the coarse tile-lattice connections using each endpoint's relative tile position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_runs, col_runs = _lattice_runs(g)
    inverted, _ = _invert_rows(g)

    revealed_mask = (g == BG) & (inverted != BG) & (inverted != SEP)
    groups: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    for tile_r, (r0, r1) in enumerate(row_runs):
        for tile_c, (c0, c1) in enumerate(col_runs):
            for rel_r, r in enumerate(range(r0, r1)):
                for rel_c, c in enumerate(range(c0, c1)):
                    if revealed_mask[r, c]:
                        color = int(inverted[r, c])
                        groups[(color, rel_r, rel_c)].append((tile_r, tile_c))

    out = inverted.copy()
    for (color, rel_r, rel_c), points in groups.items():
        by_row: dict[int, list[int]] = defaultdict(list)
        by_col: dict[int, list[int]] = defaultdict(list)
        for tile_r, tile_c in points:
            by_row[tile_r].append(tile_c)
            by_col[tile_c].append(tile_r)

        for tile_r, tile_cols in by_row.items():
            if len(tile_cols) < 2:
                continue
            for tile_c in range(min(tile_cols), max(tile_cols) + 1):
                r, c = _pixel_from_lane(row_runs, col_runs, tile_r, tile_c, rel_r, rel_c)
                out[r, c] = color

        for tile_c, tile_rows in by_col.items():
            if len(tile_rows) < 2:
                continue
            for tile_r in range(min(tile_rows), max(tile_rows) + 1):
                r, c = _pixel_from_lane(row_runs, col_runs, tile_r, tile_c, rel_r, rel_c)
                out[r, c] = color

    return out


def _lane_color(row_colors: tuple[int, int, int], rel_r: int) -> int:
    return int(row_colors[rel_r])


def _build_candidates(
    tile_rows: int, tile_cols: int, row_colors: tuple[int, int, int]
) -> list[dict]:
    candidates: list[dict] = []
    for rel_r in range(3):
        for rel_c in range(3):
            color = _lane_color(row_colors, rel_r)
            for tile_r in range(tile_rows):
                for span in range(MIN_SPAN, tile_cols + 1):
                    for start_c in range(tile_cols - span + 1):
                        end_c = start_c + span - 1
                        cells = tuple((tile_r, tile_c, rel_r, rel_c) for tile_c in range(start_c, end_c + 1))
                        endpoints = (
                            (tile_r, start_c, rel_r, rel_c),
                            (tile_r, end_c, rel_r, rel_c),
                        )
                        candidates.append(
                            {
                                "orientation": "H",
                                "color": color,
                                "rel_r": rel_r,
                                "rel_c": rel_c,
                                "cells": cells,
                                "endpoints": endpoints,
                            }
                        )
            for tile_c in range(tile_cols):
                for span in range(MIN_SPAN, tile_rows + 1):
                    for start_r in range(tile_rows - span + 1):
                        end_r = start_r + span - 1
                        cells = tuple((tile_r, tile_c, rel_r, rel_c) for tile_r in range(start_r, end_r + 1))
                        endpoints = (
                            (start_r, tile_c, rel_r, rel_c),
                            (end_r, tile_c, rel_r, rel_c),
                        )
                        candidates.append(
                            {
                                "orientation": "V",
                                "color": color,
                                "rel_r": rel_r,
                                "rel_c": rel_c,
                                "cells": cells,
                                "endpoints": endpoints,
                            }
                        )
    return candidates


def _select_groups(
    tile_rows: int, tile_cols: int, row_colors: tuple[int, int, int], group_count: int
) -> list[dict] | None:
    candidates = _build_candidates(tile_rows, tile_cols, row_colors)
    groups: list[dict] = []
    occupied: set[tuple[int, int, int, int]] = set()
    used_keys: set[tuple[int, int]] = set()
    present_orientations: set[str] = set()

    while len(groups) < group_count:
        need_orientations = []
        if group_count >= 2:
            if "H" not in present_orientations:
                need_orientations.append("H")
            if "V" not in present_orientations:
                need_orientations.append("V")
        wanted = random.choice(need_orientations) if need_orientations else random.choice(["H", "V"])

        pool = [
            cand
            for cand in candidates
            if cand["orientation"] == wanted
            and (cand["rel_r"], cand["rel_c"]) not in used_keys
            and not any(cell in occupied for cell in cand["cells"])
        ]
        if not pool:
            pool = [
                cand
                for cand in candidates
                if (cand["rel_r"], cand["rel_c"]) not in used_keys
                and not any(cell in occupied for cell in cand["cells"])
            ]
        if not pool:
            return None

        chosen = random.choice(pool)
        groups.append(chosen)
        present_orientations.add(chosen["orientation"])
        used_keys.add((chosen["rel_r"], chosen["rel_c"]))
        occupied.update(chosen["cells"])

    return groups


def _empty_grid(tile_rows: int, tile_cols: int, row_colors: tuple[int, int, int]) -> np.ndarray:
    grid = np.full((tile_rows * 4 + 1, tile_cols * 4 + 1), SEP, dtype=int)
    for r in range(grid.shape[0]):
        if r % 4 == 0:
            continue
        rel_r = (r - 1) % 4
        color = row_colors[rel_r]
        for c in range(grid.shape[1]):
            if c % 4 == 0:
                continue
            grid[r, c] = color
    return grid


def _planned_pixels(
    tile_rows: int, tile_cols: int, row_runs: list[tuple[int, int]], col_runs: list[tuple[int, int]], groups: list[dict]
) -> tuple[dict[tuple[int, int, int], list[tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
    grouped_cells: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    endpoint_pixels: set[tuple[int, int]] = set()
    line_pixels: set[tuple[int, int]] = set()

    del tile_rows, tile_cols
    for group in groups:
        key = (group["color"], group["rel_r"], group["rel_c"])
        for tile_r, tile_c, rel_r, rel_c in group["cells"]:
            grouped_cells[key].append((tile_r, tile_c))
            line_pixels.add(_pixel_from_lane(row_runs, col_runs, tile_r, tile_c, rel_r, rel_c))
        for tile_r, tile_c, rel_r, rel_c in group["endpoints"]:
            endpoint_pixels.add(_pixel_from_lane(row_runs, col_runs, tile_r, tile_c, rel_r, rel_c))

    return grouped_cells, endpoint_pixels, line_pixels


def _place_distractors(
    grid: np.ndarray,
    groups: list[dict],
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
) -> None:
    row_load = defaultdict(int)

    conflict_groups = [group for group in groups if len(group["cells"]) > 2]
    random.shuffle(conflict_groups)
    target_conflicts = min(len(conflict_groups), random.choice([0, 1, 1, 2]))
    placed_conflicts = 0
    for group in conflict_groups:
        if placed_conflicts >= target_conflicts:
            break
        interiors = [cell for cell in group["cells"] if cell not in set(group["endpoints"])]
        random.shuffle(interiors)
        for tile_r, tile_c, rel_r, rel_c in interiors:
            r, c = _pixel_from_lane(row_runs, col_runs, tile_r, tile_c, rel_r, rel_c)
            if row_load[r] >= MAX_DISTRACTORS_PER_ROW:
                continue
            grid[r, c] = random.choice(DISTRACTORS)
            row_load[r] += 1
            placed_conflicts += 1
            break

    all_group_cells = set()
    for group in groups:
        all_group_cells.update(group["cells"])

    free_positions = []
    for tile_r, (r0, r1) in enumerate(row_runs):
        for tile_c, (c0, c1) in enumerate(col_runs):
            for rel_r, r in enumerate(range(r0, r1)):
                for rel_c, c in enumerate(range(c0, c1)):
                    if (tile_r, tile_c, rel_r, rel_c) in all_group_cells:
                        continue
                    free_positions.append((r, c))
    random.shuffle(free_positions)

    persistent_count = random.randint(1, min(5, len(free_positions)))
    placed = 0
    for r, c in free_positions:
        if placed >= persistent_count:
            break
        if row_load[r] >= MAX_DISTRACTORS_PER_ROW:
            continue
        grid[r, c] = random.choice(DISTRACTORS)
        row_load[r] += 1
        placed += 1


def _expected_output(grid: np.ndarray, groups: list[dict]) -> np.ndarray:
    row_runs, col_runs = _lattice_runs(grid)
    inverted, _ = _invert_rows(grid)
    out = inverted.copy()
    for group in groups:
        color = int(group["color"])
        for tile_r, tile_c, rel_r, rel_c in group["cells"]:
            r, c = _pixel_from_lane(row_runs, col_runs, tile_r, tile_c, rel_r, rel_c)
            out[r, c] = color
    return out


def _validate_semantics(
    grid: np.ndarray,
    groups: list[dict],
    row_colors: tuple[int, int, int],
    expected: np.ndarray,
) -> bool:
    row_runs, col_runs = _lattice_runs(grid)
    inverted, inferred = _invert_rows(grid)

    for r in range(grid.shape[0]):
        if r % 4 == 0:
            continue
        if inferred[r] != row_colors[(r - 1) % 4]:
            return False

    _, endpoint_pixels, line_pixels = _planned_pixels(len(row_runs), len(col_runs), row_runs, col_runs, groups)
    revealed = {tuple(map(int, rc)) for rc in np.argwhere((grid == BG) & (inverted != BG) & (inverted != SEP))}
    if revealed != endpoint_pixels:
        return False

    endpoint_counts = Counter(group["color"] for group in groups for _ in group["endpoints"])
    if any(count % 2 for count in endpoint_counts.values()):
        return False

    if not line_pixels > endpoint_pixels:
        return False

    distractor_mask = np.isin(grid, DISTRACTORS)
    if not np.any(distractor_mask):
        return False

    overwritten_conflict = False
    for r, c in np.argwhere(distractor_mask):
        if (int(r), int(c)) in line_pixels and expected[r, c] != grid[r, c]:
            overwritten_conflict = True
            break
    persistent_distractor = False
    for r, c in np.argwhere(distractor_mask):
        if (int(r), int(c)) not in line_pixels and expected[r, c] == grid[r, c]:
            persistent_distractor = True
            break
    if not overwritten_conflict or not persistent_distractor:
        return False

    return np.array_equal(main(grid), expected)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    tile_rows = random.randint(*TILE_ROWS_RANGE)
    tile_cols = random.randint(*TILE_COLS_RANGE)
    group_count = random.randint(*GROUP_COUNT_RANGE)
    row_colors = tuple(random.sample(list(ROW_COLORS), k=3))

    groups = _select_groups(tile_rows, tile_cols, row_colors, group_count)
    if groups is None:
        return None

    grid = _empty_grid(tile_rows, tile_cols, row_colors)
    row_runs, col_runs = _lattice_runs(grid)
    for group in groups:
        for tile_r, tile_c, rel_r, rel_c in group["endpoints"]:
            r, c = _pixel_from_lane(row_runs, col_runs, tile_r, tile_c, rel_r, rel_c)
            grid[r, c] = BG

    _place_distractors(grid, groups, row_runs, col_runs)
    expected = _expected_output(grid, groups)
    if not _validate_semantics(grid, groups, row_colors, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is None:
            continue
        grid, expected = result
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
