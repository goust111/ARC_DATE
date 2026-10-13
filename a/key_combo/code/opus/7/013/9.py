from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SCATTER_COLORS = (1, 2, 3, 4)
WALL = 5
MARKER = 7
GEN_RETRIES = 400


def _wall_rows_and_doors(grid: np.ndarray) -> tuple[list[int], list[int]]:
    wall_rows: list[int] = []
    door_cols: list[int] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.count_nonzero(row == WALL) + np.count_nonzero(row == BG) != row.size:
            continue
        zero_cols = np.flatnonzero(row == BG)
        if zero_cols.size != 3:
            continue
        if not np.array_equal(zero_cols, np.arange(int(zero_cols[0]), int(zero_cols[0]) + 3)):
            continue
        if np.count_nonzero(row == WALL) != row.size - 3:
            continue
        wall_rows.append(int(r))
        door_cols.append(int(zero_cols[1]))
    assert len(wall_rows) >= 2, "expected at least two horizontal walls"
    return wall_rows, door_cols


def _markers(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = np.argwhere(grid == MARKER)
    assert coords.shape[0] == 2, "expected exactly two orange markers"
    pts = [tuple(map(int, rc)) for rc in coords]
    pts.sort()
    return pts[0], pts[1]


def _corridor_bounds(height: int, wall_rows: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    top = 0
    for wall_r in wall_rows:
        bottom = wall_r - 1
        assert top <= bottom, "corridor must have positive height"
        bounds.append((top, bottom))
        top = wall_r + 1
    assert top <= height - 1, "last corridor must exist"
    bounds.append((top, height - 1))
    return bounds


def _segment_cells(a: tuple[int, int], b: tuple[int, int], include_start: bool) -> set[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    assert r0 == r1 or c0 == c1, "segments must be orthogonal"
    cells: set[tuple[int, int]] = set()
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        start = lo if include_start else lo + 1 if c0 <= c1 else lo
        for c in range(lo, hi + 1):
            if include_start or c != c0:
                cells.add((r0, c))
    else:
        lo, hi = sorted((r0, r1))
        for r in range(lo, hi + 1):
            if include_start or r != r0:
                cells.add((r, c0))
    return cells


def _build_path_plan(
    shape: tuple[int, int],
    wall_rows: list[int],
    door_cols: list[int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[list[set[tuple[int, int]]], list[tuple[int, int]]]:
    height, width = map(int, shape)
    del width
    assert len(wall_rows) == len(door_cols)
    assert start[0] < end[0], "markers must be ordered top to bottom"

    corridors = _corridor_bounds(height, wall_rows)
    assert start[0] == corridors[0][0], "top marker must be on the top edge of the first corridor"
    assert end[0] == corridors[-1][1], "bottom marker must be on the bottom edge of the last corridor"

    corridor_cells: list[set[tuple[int, int]]] = []
    door_cells: list[tuple[int, int]] = []
    cur_r, cur_c = start
    for idx, wall_r in enumerate(wall_rows):
        _, corridor_bottom = corridors[idx]
        cells = set()
        cells |= _segment_cells((cur_r, cur_c), (corridor_bottom, cur_c), include_start=False)
        cells |= _segment_cells((corridor_bottom, cur_c), (corridor_bottom, door_cols[idx]), include_start=True)
        cells.discard(start)
        cells.discard(end)
        corridor_cells.append(cells)
        door_cells.append((wall_r, door_cols[idx]))
        cur_r, cur_c = wall_r, door_cols[idx]

    cells = set()
    cells |= _segment_cells((cur_r, cur_c), (end[0], cur_c), include_start=False)
    cells |= _segment_cells((end[0], cur_c), end, include_start=True)
    cells.discard(start)
    cells.discard(end)
    corridor_cells.append(cells)
    return corridor_cells, door_cells


def _majority_color(
    grid: np.ndarray,
    bounds: tuple[int, int],
    excluded: set[tuple[int, int]],
) -> int:
    r0, r1 = bounds
    counts: Counter[int] = Counter()
    for r in range(r0, r1 + 1):
        for c in range(grid.shape[1]):
            if (r, c) in excluded:
                continue
            color = int(grid[r, c])
            if color in (BG, WALL, MARKER):
                continue
            counts[color] += 1
    assert counts, "each corridor must contain scatter pixels"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "corridor majority color must be unique"
    return int(ranked[0][0])


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    wall_rows, door_cols = _wall_rows_and_doors(g)
    start, end = _markers(g)
    corridors = _corridor_bounds(g.shape[0], wall_rows)
    corridor_cells, door_cells = _build_path_plan(g.shape, wall_rows, door_cols, start, end)

    majorities: list[int] = []
    for bounds, cells in zip(corridors, corridor_cells):
        marker_exclusions = {
            (r, c)
            for r, c in (start, end)
            if bounds[0] <= r <= bounds[1]
        }
        majority = _majority_color(g, bounds, excluded=set(cells) | marker_exclusions)
        majorities.append(majority)

    out = g.copy()
    scatter_mask = ~np.isin(out, (BG, WALL, MARKER))
    out[scatter_mask] = BG

    for cells, color in zip(corridor_cells, majorities):
        for r, c in cells:
            out[r, c] = int(color)
    for i, (r, c) in enumerate(door_cells):
        out[r, c] = int(majorities[i])

    out[start] = MARKER
    out[end] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the door-threaded corridor path and recolor each corridor segment by that corridor's majority scatter color.

    Input structure:
    - The grid has a black background, several full-width horizontal gray walls, and one
      three-cell door gap in each wall.
    - One orange marker sits on the top edge of the first corridor and one on the bottom
      edge of the last corridor.
    - Inside each corridor, non-path scatter pixels use colors 1-4 and have a unique
      majority color for that corridor.

    Rule:
    - Follow the horizontal-wall route pattern from the top marker to the bottom marker:
      in each corridor, move vertically to the row just above the next wall, move
      horizontally to that wall's door center, then cross the door center.
    - Compute each corridor's majority scatter color while ignoring walls, markers, and
      the path cells themselves.
    - Erase the scatter pixels, paint each corridor segment with its corridor majority
      color, paint each door-center crossing with the majority color of the corridor
      above it, and keep walls plus markers unchanged.

    Key insight:
    - The path geometry comes from the ordered door centers, but the path color is local:
      each corridor contributes its own majority-vote color to only its own segment.
    """
    return _solve(grid)


def _pick_dimensions(rng: random.Random) -> tuple[int, int, int, list[int]]:
    corridor_count = rng.choice((3, 4))
    heights = [rng.randint(4, 6) for _ in range(corridor_count)]
    total_h = sum(heights) + (corridor_count - 1)
    if total_h > 30:
        raise ValueError("height overflow")
    width = rng.randint(16, 24)
    return corridor_count, total_h, width, heights


def _wall_rows_from_heights(heights: list[int]) -> list[int]:
    rows: list[int] = []
    cur = heights[0]
    for h in heights[1:]:
        rows.append(cur)
        cur += 1 + h
    return rows


def _sample_route_columns(width: int, wall_count: int, rng: random.Random) -> tuple[int, list[int], int]:
    choices = list(range(1, width - 1))
    for _ in range(200):
        start = rng.choice(choices)
        doors = [rng.randint(2, width - 3) for _ in range(wall_count)]
        end = rng.choice(choices)
        seq = [start] + doors + [end]
        if all(abs(a - b) >= 2 for a, b in zip(seq, seq[1:])):
            return start, doors, end
    raise ValueError("could not sample a non-degenerate route")


def _corridor_for_row(bounds: list[tuple[int, int]], row: int) -> int:
    for i, (r0, r1) in enumerate(bounds):
        if r0 <= row <= r1:
            return i
    raise AssertionError("row is not inside any corridor")


def _place_scatter(
    grid: np.ndarray,
    bounds: list[tuple[int, int]],
    corridor_cells: list[set[tuple[int, int]]],
    start: tuple[int, int],
    end: tuple[int, int],
    majority_colors: list[int],
    rng: random.Random,
) -> None:
    for i, (r0, r1) in enumerate(bounds):
        forbidden = set(corridor_cells[i])
        if r0 <= start[0] <= r1:
            forbidden.add(start)
        if r0 <= end[0] <= r1:
            forbidden.add(end)
        available = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(grid.shape[1])
            if (r, c) not in forbidden
        ]
        rng.shuffle(available)
        assert len(available) >= 8, "corridor needs room for scatter"

        majority = majority_colors[i]
        distractors = [c for c in SCATTER_COLORS if c != majority]
        rng.shuffle(distractors)
        distractors = distractors[: rng.choice((1, 2))]

        majority_count = rng.randint(4, 6)
        counts = {majority: majority_count}
        remaining_slots = len(available) - majority_count
        for color in distractors:
            max_count = min(majority_count - 1, 3, remaining_slots)
            assert max_count >= 1
            counts[color] = rng.randint(1, max_count)
            remaining_slots -= counts[color]

        assert max(counts.values()) == counts[majority]
        assert list(counts.values()).count(counts[majority]) == 1
        needed = sum(counts.values())
        assert needed >= 5
        picked = available[:needed]
        offset = 0
        for color, count in counts.items():
            for r, c in picked[offset : offset + count]:
                grid[r, c] = int(color)
            offset += count


def _oracle_output(
    grid: np.ndarray,
    wall_rows: list[int],
    door_cols: list[int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> np.ndarray:
    bounds = _corridor_bounds(grid.shape[0], wall_rows)
    corridor_cells, door_cells = _build_path_plan(grid.shape, wall_rows, door_cols, start, end)
    majorities: list[int] = []
    for corridor_bounds, cells in zip(bounds, corridor_cells):
        marker_exclusions = {
            (r, c)
            for r, c in (start, end)
            if corridor_bounds[0] <= r <= corridor_bounds[1]
        }
        majorities.append(_majority_color(grid, corridor_bounds, excluded=set(cells) | marker_exclusions))

    out = grid.copy()
    out[~np.isin(out, (BG, WALL, MARKER))] = BG
    for cells, color in zip(corridor_cells, majorities):
        for r, c in cells:
            out[r, c] = color
    for i, (r, c) in enumerate(door_cells):
        out[r, c] = majorities[i]
    out[start] = MARKER
    out[end] = MARKER
    return out


def _validate_generated(
    grid: np.ndarray,
    wall_rows: list[int],
    door_cols: list[int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    try:
        bounds = _corridor_bounds(grid.shape[0], wall_rows)
        corridor_cells, door_cells = _build_path_plan(grid.shape, wall_rows, door_cols, start, end)

        assert len(bounds) in (3, 4)
        assert len(set(door_cols)) >= 2
        assert start[0] == 0 and end[0] == grid.shape[0] - 1
        assert all(abs(a - b) >= 2 for a, b in zip([start[1]] + door_cols, door_cols + [end[1]]))

        path_cells = set().union(*corridor_cells) | set(door_cells)
        scatter_on_path = np.argwhere((~np.isin(grid, (BG, WALL, MARKER))) & np.array(
            [[(r, c) in path_cells for c in range(grid.shape[1])] for r in range(grid.shape[0])],
            dtype=bool,
        ))
        assert scatter_on_path.size == 0

        majorities: list[int] = []
        for cells, corridor_bounds in zip(corridor_cells, bounds):
            marker_exclusions = {
                (r, c)
                for r, c in (start, end)
                if corridor_bounds[0] <= r <= corridor_bounds[1]
            }
            majority = _majority_color(grid, corridor_bounds, excluded=set(cells) | marker_exclusions)
            majorities.append(majority)
            counts = Counter(
                int(grid[r, c])
                for r in range(corridor_bounds[0], corridor_bounds[1] + 1)
                for c in range(grid.shape[1])
                if int(grid[r, c]) not in (BG, WALL, MARKER) and (r, c) not in cells and (r, c) not in marker_exclusions
            )
            assert sum(counts.values()) >= 3
        assert len(set(majorities)) == len(majorities)

        expected = _oracle_output(grid, wall_rows, door_cols, start, end)
        actual = main(grid)
        assert np.array_equal(actual, expected)

        first_color = majorities[0]
        global_counts = Counter(int(x) for x in grid[~np.isin(grid, (BG, WALL, MARKER))])
        global_majority = global_counts.most_common(1)[0][0]
        uniform_first = expected.copy()
        uniform_global = expected.copy()
        for cells in corridor_cells:
            for r, c in cells:
                uniform_first[r, c] = first_color
                uniform_global[r, c] = global_majority
        for r, c in door_cells:
            uniform_first[r, c] = first_color
            uniform_global[r, c] = global_majority
        assert not np.array_equal(uniform_first, expected)
        assert not np.array_equal(uniform_global, expected)
        return True
    except Exception:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    corridor_count, height, width, heights = _pick_dimensions(rng)
    wall_rows = _wall_rows_from_heights(heights)
    wall_count = corridor_count - 1
    start_col, door_cols, end_col = _sample_route_columns(width, wall_count, rng)

    grid = np.zeros((height, width), dtype=int)
    for wall_r, door_c in zip(wall_rows, door_cols):
        grid[wall_r, :] = WALL
        grid[wall_r, door_c - 1 : door_c + 2] = BG

    start = (0, start_col)
    end = (height - 1, end_col)
    grid[start] = MARKER
    grid[end] = MARKER

    bounds = _corridor_bounds(height, wall_rows)
    corridor_cells, _ = _build_path_plan(grid.shape, wall_rows, door_cols, start, end)
    majority_colors = rng.sample(list(SCATTER_COLORS), corridor_count)
    _place_scatter(grid, bounds, corridor_cells, start, end, majority_colors, rng)

    if _validate_generated(grid, wall_rows, door_cols, start, end):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
