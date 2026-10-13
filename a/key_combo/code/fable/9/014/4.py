from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SEED_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


def _singleton_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BLACK, GREY)]
    assert len(colors) == 2, "expected exactly two non-grey seed colors"

    seeds = []
    for color in sorted(colors):
        coords = np.argwhere(grid == color)
        assert coords.shape == (1, 2), "each seed color must appear exactly once"
        r, c = map(int, coords[0])
        seeds.append((r, c, color))
    return seeds


def _orientation_from_seeds(shape: tuple[int, int], seeds: list[tuple[int, int, int]]) -> str:
    _height, width = map(int, shape)
    return "row" if any(c in (0, width - 1) for _r, c, _color in seeds) else "col"


def _period_from_orientation(seeds: list[tuple[int, int, int]], orientation: str) -> int:
    if orientation == "row":
        axis_values = [r for r, _c, _color in seeds]
    else:
        axis_values = [c for _r, c, _color in seeds]
    d = abs(int(axis_values[0]) - int(axis_values[1]))
    assert d > 0, "seed separation along the stripe axis must be positive"
    return d


def _paint_from_orientation(
    grid: np.ndarray,
    seeds: list[tuple[int, int, int]],
    orientation: str,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    d = _period_from_orientation(seeds, orientation)
    period = 2 * d

    (ar, ac, acolor), (br, bc, bcolor) = seeds
    a_axis = ar if orientation == "row" else ac
    b_axis = br if orientation == "row" else bc

    for r_raw, c_raw in np.argwhere(out == GREY):
        r = int(r_raw)
        c = int(c_raw)
        pos = r if orientation == "row" else c
        on_a = (pos - a_axis) % period == 0
        on_b = (pos - b_axis) % period == 0
        assert not (on_a and on_b), "grey point cannot land on both seed phases"

        if on_a:
            out[r, c] = acolor
        elif on_b:
            out[r, c] = bcolor
        else:
            out[r, c] = BLACK

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor or erase scattered grey points using the invisible periodic stripe phases defined by two seeds.

    Input structure:
    - The grid is black except for exactly two singleton seed pixels of different colors and several grey scattered pixels.
    - If either seed touches the left or right border, the hidden stripes are full rows; otherwise the hidden stripes are full columns.
    - The two seeds are separated by distance d along that stripe axis, and their unseen phases repeat every 2d.

    Rule:
    - Keep both seed pixels unchanged and never draw the stripes themselves.
    - For each grey pixel, compare its row or column to the corresponding coordinate of each seed using the chosen stripe axis.
    - If it matches the first seed modulo 2d, recolor it to that seed color; if it matches the second seed modulo 2d, recolor it to the other seed color; otherwise erase it to black.

    Key insight:
    - The seeds are not objects to connect; they are generators of an invisible alternating phase field, and every grey point is judged only by its phase within that hidden periodic coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert np.all(np.isin(g, range(10))), "expected ARC palette ids"

    seeds = _singleton_seeds(g)
    orientation = _orientation_from_seeds(g.shape, seeds)
    return _paint_from_orientation(g, seeds, orientation)


def _choose_stripe_layout(limit: int) -> tuple[list[int], int]:
    for _ in range(200):
        max_d = min(5, (int(limit) - 1) // 3)
        if max_d < 2:
            break
        d = random.randint(2, max_d)
        max_stripes = min(6, (int(limit) - 1) // d + 1)
        if max_stripes < 4:
            continue
        n_stripes = random.randint(4, max_stripes)
        start = random.randint(0, int(limit) - 1 - (n_stripes - 1) * d)
        stripes = [start + i * d for i in range(n_stripes)]
        seed_index = random.randint(1, n_stripes - 3)
        return stripes, seed_index
    raise RuntimeError("failed to choose a non-degenerate stripe layout")


def _row_seed_columns(width: int, d: int, both_edges: bool) -> tuple[int, int]:
    if both_edges:
        candidates = [(0, width - 1), (width - 1, 0)]
        return random.choice(candidates)

    for _ in range(200):
        edge_col = random.choice((0, width - 1))
        inner_col = random.randint(1, width - 2)
        if abs(edge_col - inner_col) in (0, d):
            continue
        if random.choice((True, False)):
            return edge_col, inner_col
        return inner_col, edge_col
    raise RuntimeError("failed to place row-oriented seed columns")


def _column_seed_rows(height: int, d: int, force_border_row: bool) -> tuple[int, int]:
    for _ in range(300):
        if force_border_row:
            edge_row = random.choice((0, height - 1))
            inner_row = random.randint(1, height - 2)
            if random.choice((True, False)):
                rows = (edge_row, inner_row)
            else:
                rows = (inner_row, edge_row)
        else:
            rows = tuple(random.sample(range(1, height - 1), 2))

        if abs(rows[0] - rows[1]) in (0, d):
            continue
        return int(rows[0]), int(rows[1])
    raise RuntimeError("failed to place column-oriented seed rows")


def _place_grey_on_axis(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    orientation: str,
    axis_value: int,
) -> tuple[int, int]:
    height, width = map(int, grid.shape)
    if orientation == "row":
        candidates = [(int(axis_value), c) for c in range(width) if (int(axis_value), c) not in occupied]
    else:
        candidates = [(r, int(axis_value)) for r in range(height) if (r, int(axis_value)) not in occupied]

    if not candidates:
        raise RuntimeError("no free location left for grey point")
    r, c = random.choice(candidates)
    grid[r, c] = GREY
    occupied.add((r, c))
    return r, c


def _sample_off_axis_value(limit: int, forbidden: set[int]) -> int:
    choices = [value for value in range(int(limit)) if value not in forbidden]
    if not choices:
        raise RuntimeError("expected at least one off-stripe axis position")
    return int(random.choice(choices))


def _grey_annotations(
    grid: np.ndarray,
    seeds: list[tuple[int, int, int]],
    orientation: str,
) -> list[dict[str, int | tuple[int, int]]]:
    d = _period_from_orientation(seeds, orientation)
    period = 2 * d
    (ar, ac, acolor), (br, bc, bcolor) = seeds
    a_axis = ar if orientation == "row" else ac
    b_axis = br if orientation == "row" else bc

    notes = []
    for r_raw, c_raw in np.argwhere(grid == GREY):
        r = int(r_raw)
        c = int(c_raw)
        pos = r if orientation == "row" else c
        on_a = (pos - a_axis) % period == 0
        on_b = (pos - b_axis) % period == 0
        assert not (on_a and on_b)
        if on_a:
            out_color = acolor
        elif on_b:
            out_color = bcolor
        else:
            out_color = BLACK
        notes.append(
            {
                "coord": (r, c),
                "axis": pos,
                "out_color": out_color,
                "is_a": int(on_a),
                "is_b": int(on_b),
            }
        )
    return notes


def _validate_semantics(grid: np.ndarray, expect_border_counterexample: bool) -> bool:
    try:
        seeds = _singleton_seeds(grid)
        orientation = _orientation_from_seeds(grid.shape, seeds)
        actual = main(grid)
        expected = _paint_from_orientation(grid, seeds, orientation)
        if not np.array_equal(actual, expected):
            return False

        notes = _grey_annotations(grid, seeds, orientation)
        if len(notes) < 8:
            return False

        grey_outs = [int(note["out_color"]) for note in notes]
        if BLACK not in grey_outs:
            return False

        seed_colors = {color for _r, _c, color in seeds}
        if not seed_colors.issubset(set(grey_outs) | seed_colors):
            return False

        a_axis, b_axis = (
            [r for r, _c, _color in seeds]
            if orientation == "row"
            else [c for _r, c, _color in seeds]
        )
        seed_low = min(a_axis, b_axis)
        seed_high = max(a_axis, b_axis)

        repeated_a = [note for note in notes if note["is_a"] and int(note["axis"]) != a_axis]
        repeated_b = [note for note in notes if note["is_b"] and int(note["axis"]) != b_axis]
        if not repeated_a or not repeated_b:
            return False

        if not any(int(note["out_color"]) == BLACK for note in notes if seed_low < int(note["axis"]) < seed_high):
            return False
        if not any(int(note["out_color"]) in seed_colors and int(note["axis"]) < seed_low for note in notes):
            return False
        if not any(int(note["out_color"]) in seed_colors and int(note["axis"]) > seed_high for note in notes):
            return False

        shortcut = grid.copy()
        for r_raw, c_raw in np.argwhere(shortcut == GREY):
            r = int(r_raw)
            c = int(c_raw)
            if orientation == "row":
                if r == seeds[0][0]:
                    shortcut[r, c] = seeds[0][2]
                elif r == seeds[1][0]:
                    shortcut[r, c] = seeds[1][2]
                else:
                    shortcut[r, c] = BLACK
            else:
                if c == seeds[0][1]:
                    shortcut[r, c] = seeds[0][2]
                elif c == seeds[1][1]:
                    shortcut[r, c] = seeds[1][2]
                else:
                    shortcut[r, c] = BLACK
        if np.array_equal(shortcut, expected):
            return False

        opposite = "col" if orientation == "row" else "row"
        if np.array_equal(_paint_from_orientation(grid, seeds, opposite), expected):
            return False

        if expect_border_counterexample:
            seed_rows = [r for r, _c, _color in seeds]
            seed_cols = [c for _r, c, _color in seeds]
            touches_tb_only = any(r in (0, grid.shape[0] - 1) for r in seed_rows) and all(
                c not in (0, grid.shape[1] - 1) for c in seed_cols
            )
            if not touches_tb_only or orientation != "col":
                return False

        return True
    except Exception:
        return False


def _attempt_input(orientation: str, both_side_edges: bool, force_tb_border: bool) -> np.ndarray | None:
    height = random.randint(10, 24)
    width = random.randint(10, 24)
    if orientation == "row" and width < 5:
        return None
    if orientation == "col" and height < 5:
        return None

    if orientation == "row":
        stripes, seed_index = _choose_stripe_layout(height)
        d = stripes[seed_index + 1] - stripes[seed_index]
        rows = (stripes[seed_index], stripes[seed_index + 1])
        cols = _row_seed_columns(width, d, both_side_edges)
        seed_positions = ((rows[0], cols[0]), (rows[1], cols[1]))
        axis_limit = height
    else:
        stripes, seed_index = _choose_stripe_layout(width)
        d = stripes[seed_index + 1] - stripes[seed_index]
        cols = (stripes[seed_index], stripes[seed_index + 1])
        rows = _column_seed_rows(height, d, force_tb_border)
        seed_positions = ((rows[0], cols[0]), (rows[1], cols[1]))
        axis_limit = width

    color_a, color_b = random.sample(SEED_PALETTE, 2)
    grid = np.zeros((height, width), dtype=int)
    occupied: set[tuple[int, int]] = set()
    for (r, c), color in zip(seed_positions, (color_a, color_b)):
        grid[int(r), int(c)] = int(color)
        occupied.add((int(r), int(c)))

    if orientation == "row":
        a_axis = rows[0]
        b_axis = rows[1]
    else:
        a_axis = cols[0]
        b_axis = cols[1]

    mandatory_axes = [
        stripes[seed_index - 1],
        a_axis,
        b_axis,
        stripes[seed_index + 2],
        random.randint(a_axis + 1, b_axis - 1),
        random.randint(b_axis + 1, stripes[seed_index + 2] - 1),
    ]
    for axis_value in mandatory_axes:
        _place_grey_on_axis(grid, occupied, orientation, axis_value)

    used_axis_values = set(mandatory_axes)
    for axis_value in stripes:
        if axis_value not in used_axis_values:
            _place_grey_on_axis(grid, occupied, orientation, axis_value)
            used_axis_values.add(axis_value)

    off_axis_values = [value for value in range(axis_limit) if value not in set(stripes)]
    if len(off_axis_values) < 2:
        return None
    for axis_value in random.sample(off_axis_values, k=2):
        _place_grey_on_axis(grid, occupied, orientation, axis_value)

    extra_points = random.randint(2, 6)
    for _ in range(extra_points):
        if random.random() < 0.65:
            axis_value = random.choice(stripes)
        else:
            axis_value = _sample_off_axis_value(axis_limit, set(stripes))
        _place_grey_on_axis(grid, occupied, orientation, axis_value)

    if not _validate_semantics(grid, expect_border_counterexample=force_tb_border):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    orientation = "row" if _GEN_CALL_COUNT % 2 == 1 else "col"
    both_side_edges = orientation == "row" and _GEN_CALL_COUNT % 6 == 3
    force_tb_border = orientation == "col" and _GEN_CALL_COUNT % 4 == 0

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input(orientation, both_side_edges, force_tb_border)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-stripe-field input")
