from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 7)

HEIGHT_RANGE = (13, 18)
WIDTH_RANGE = (11, 17)
SEPARATOR_ROWS = (3, 4)
ACTIVE_COL_RANGE = (4, 8)
MAX_GEN_ATTEMPTS = 500
ROW_PLACEMENT_ATTEMPTS = 200


def _separator_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    return int(rows[0])


def _lower_points(grid: np.ndarray, sep: int) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(grid[sep + 1 :, :] != BG):
        rr = int(r + sep + 1)
        cc = int(c)
        color = int(grid[rr, cc])
        assert color != GREY, "grey is reserved for the separator in the input"
        points.append((rr, cc, color))
    return points


def _counts_oracle(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep = _separator_row(g)
    ref_row = sep - 1

    out = g.copy()
    out[sep + 1 :, :] = BG

    for c in range(w):
        ref_color = int(g[ref_row, c])
        col_points = [(r, int(g[r, c])) for r in range(sep + 1, h) if int(g[r, c]) != BG]
        if not col_points:
            continue
        assert ref_color != BG, "every lower-zone point must have a reference pixel in its column"
        down_count = sum(color == ref_color for _r, color in col_points)
        up_count = len(col_points) - down_count
        assert down_count + up_count <= h - sep - 1, "column overflow in lower zone"
        for i in range(up_count):
            out[sep + 1 + i, c] = GREY
        for i in range(down_count):
            out[h - 1 - i, c] = ref_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Gate each lower-zone pixel by its column reference color, then drop matches downward and float mismatches upward in grey.

    Input structure:
    - Black background with one full-width grey horizontal separator line.
    - The row immediately above the separator contains sparse reference pixels, one per used column at most.
    - Single non-grey foreground pixels appear only below the separator, and every such pixel lies in a column whose reference row is non-background.

    Rule:
    - For each lower-zone pixel, compare its color with the reference pixel in the same column.
    - If the colors match, erase the source cell and let that pixel fall downward until the grid bottom or an already settled lower pixel stops it; the color stays unchanged.
    - If the colors differ, erase the source cell and let that pixel rise upward until the separator or an already settled lower pixel stops it; recolor it grey.
    - Process all downward movers first from lower rows to higher rows, then process all upward movers from higher rows to lower rows. Keep the separator and reference row unchanged.

    Key insight:
    - A color does not have one global meaning: the same visible color can sink in one column and float in another because only the column-aligned reference pixel decides the direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    sep = _separator_row(g)
    assert sep >= 1 and sep + 1 < h, "separator must leave room for a reference row and a lower zone"
    ref_row = sep - 1

    out = g.copy()
    out[sep + 1 :, :] = BG

    downward: list[tuple[int, int, int]] = []
    upward: list[tuple[int, int, int]] = []
    for r, c, color in _lower_points(g, sep):
        ref_color = int(g[ref_row, c])
        assert ref_color in PALETTE, "lower-zone columns must point to a colored reference pixel"
        if color == ref_color:
            downward.append((r, c, color))
        else:
            upward.append((r, c, color))

    for r, c, color in sorted(downward, key=lambda item: (-item[0], item[1])):
        rr = int(r)
        while rr + 1 < h and int(out[rr + 1, c]) == BG:
            rr += 1
        out[rr, c] = int(color)

    for r, c, _color in sorted(upward, key=lambda item: (item[0], item[1])):
        rr = int(r)
        while rr - 1 > sep and int(out[rr - 1, c]) == BG:
            rr -= 1
        out[rr, c] = GREY

    return out


def _spaced_combos(rows: list[int], count: int) -> list[tuple[int, ...]]:
    return [
        combo
        for combo in itertools.combinations(rows, count)
        if all(b - a >= 2 for a, b in zip(combo, combo[1:]))
    ]


def _build_references(width: int) -> tuple[dict[int, int], list[int], list[int]]:
    max_active = min(ACTIVE_COL_RANGE[1], width - 1)
    active_count = random.randint(ACTIVE_COL_RANGE[0], max_active)
    active_cols = sorted(random.sample(range(width), active_count))

    ref_color_count = random.randint(3, min(len(PALETTE), active_count))
    ref_colors = random.sample(PALETTE, ref_color_count)
    refs: dict[int, int] = {}

    shuffled = active_cols[:]
    random.shuffle(shuffled)
    for col, color in zip(shuffled, ref_colors):
        refs[int(col)] = int(color)
    for col in shuffled[len(ref_colors) :]:
        refs[int(col)] = int(random.choice(ref_colors))

    return refs, active_cols, ref_colors


def _seed_plans(
    refs: dict[int, int],
    active_cols: list[int],
    max_per_col: int,
) -> dict[int, list[int]]:
    ref_colors = sorted(set(int(color) for color in refs.values()))
    witness = int(random.choice(ref_colors))

    mixed_col = int(random.choice([col for col in active_cols if refs[col] == witness]))
    up_col = int(random.choice([col for col in active_cols if refs[col] != witness]))
    down_col = int(random.choice([col for col in active_cols if col not in {mixed_col, up_col}]))

    mismatch_pool = [color for color in ref_colors if color != witness]
    mixed: list[int]
    if max_per_col >= 4 and random.random() < 0.5:
        mixed = [witness, random.choice(mismatch_pool), witness, random.choice(mismatch_pool)]
    else:
        mixed = [witness, random.choice(mismatch_pool), witness]

    up_pool = [color for color in ref_colors if color != refs[up_col]]
    up_count = random.randint(2, max_per_col)
    upward = [witness] + [int(random.choice(up_pool)) for _ in range(up_count - 1)]
    random.shuffle(upward)

    down_count = random.randint(3, max_per_col)
    downward = [int(refs[down_col])] * down_count

    plans = {
        mixed_col: mixed,
        up_col: upward,
        down_col: downward,
    }

    extra_candidates = [col for col in active_cols if col not in plans]
    max_extra_cols = max(0, min(2, len(extra_candidates) - 1))
    if max_extra_cols > 0:
        extra_count = random.randint(0, max_extra_cols)
        extra_cols = random.sample(extra_candidates, extra_count)
        for col in extra_cols:
            point_count = random.randint(1, min(2, max_per_col))
            colors: list[int] = []
            choices = [color for color in ref_colors if color != refs[col]]
            for _ in range(point_count):
                if random.random() < 0.5:
                    colors.append(int(refs[col]))
                else:
                    colors.append(int(random.choice(choices)))
            random.shuffle(colors)
            plans[int(col)] = colors

    return plans


def _place_rows(height: int, sep: int, plans: dict[int, list[int]]) -> dict[int, tuple[int, ...]] | None:
    interior_rows = list(range(sep + 2, height - 1))
    combos_by_size = {
        size: _spaced_combos(interior_rows, size)
        for size in {len(colors) for colors in plans.values()}
    }
    if any(not combos_by_size[len(colors)] for colors in plans.values()):
        return None

    columns = sorted(plans, key=lambda col: (-len(plans[col]), col))
    placed: dict[int, tuple[int, ...]] = {}

    def backtrack(i: int) -> bool:
        if i == len(columns):
            return True
        col = columns[i]
        candidates = combos_by_size[len(plans[col])][:]
        random.shuffle(candidates)
        for combo in candidates:
            valid = True
            for other_col, other_rows in placed.items():
                if abs(other_col - col) == 1 and set(combo) & set(other_rows):
                    valid = False
                    break
            if not valid:
                continue
            placed[col] = combo
            if backtrack(i + 1):
                return True
            del placed[col]
        return False

    for _ in range(ROW_PLACEMENT_ATTEMPTS):
        placed.clear()
        if backtrack(0):
            return {col: tuple(sorted(rows)) for col, rows in placed.items()}
    return None


def _materialize_input(
    height: int,
    width: int,
    sep: int,
    refs: dict[int, int],
    plans: dict[int, list[int]],
    rows_by_col: dict[int, tuple[int, ...]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[sep, :] = GREY
    ref_row = sep - 1
    for col, color in refs.items():
        grid[ref_row, int(col)] = int(color)
    for col, colors in plans.items():
        rows = rows_by_col[col]
        assert len(rows) == len(colors)
        for row, color in zip(rows, colors):
            grid[int(row), int(col)] = int(color)
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    sep = _separator_row(inp)
    ref_row = sep - 1
    h, _w = inp.shape
    points = _lower_points(inp, sep)
    if not (8 <= len(points) <= 14):
        return False

    by_col: dict[int, list[tuple[int, int]]] = {}
    fates_by_color: dict[int, set[str]] = {}
    moved_down = False
    moved_up = False
    for r, c, color in points:
        by_col.setdefault(int(c), []).append((int(r), int(color)))
        ref_color = int(inp[ref_row, c])
        if ref_color == BG:
            return False
        fate = "down" if color == ref_color else "up"
        fates_by_color.setdefault(int(color), set()).add(fate)
        if fate == "down" and r < h - 1:
            moved_down = True
        if fate == "up" and r > sep + 1:
            moved_up = True

    if not moved_down or not moved_up:
        return False
    if not any(
        any(color == int(inp[ref_row, c]) for _r, color in pts)
        and any(color != int(inp[ref_row, c]) for _r, color in pts)
        for c, pts in by_col.items()
    ):
        return False
    if not any(len(fates) == 2 for fates in fates_by_color.values()):
        return False
    if not any(len(pts) >= 3 for pts in by_col.values()):
        return False
    if len(set(int(inp[ref_row, c]) for c in by_col)) < 2:
        return False
    if not np.any(inp[ref_row] == BG):
        return False

    expected = _counts_oracle(inp)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, inp):
        return False

    for c, pts in by_col.items():
        ref_color = int(inp[ref_row, c])
        up_count = sum(color != ref_color for _r, color in pts)
        down_count = len(pts) - up_count
        lower = actual[sep + 1 :, c]
        if int(np.count_nonzero(lower == GREY)) != up_count:
            return False
        if int(np.count_nonzero(lower == ref_color)) != down_count:
            return False
        if up_count and not np.all(lower[:up_count] == GREY):
            return False
        if down_count and not np.all(lower[-down_count:] == ref_color):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    sep = int(random.choice(SEPARATOR_ROWS))
    if sep + 3 >= height:
        return None

    lower_height = height - sep - 1
    max_per_col = min(4, max(3, (lower_height - 1) // 2))
    refs, active_cols, _ref_colors = _build_references(width)
    plans = _seed_plans(refs, active_cols, max_per_col)
    rows_by_col = _place_rows(height, sep, plans)
    if rows_by_col is None:
        return None

    grid = _materialize_input(height, width, sep, refs, plans, rows_by_col)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color-gated bidirectional gravity input")
