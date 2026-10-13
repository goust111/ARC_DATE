from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
TEAL = 8

ABOVE = -1
BELOW = 1

CATEGORIES = (
    (RED, ABOVE),
    (RED, BELOW),
    (BLUE, ABOVE),
    (BLUE, BELOW),
)

GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(values.argmax())


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _find_divider_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1, "expected exactly one full-width grey divider row"
    return int(rows[0])


def _side_of(r: int, divider_row: int) -> int:
    return ABOVE if int(r) < int(divider_row) else BELOW


def _horizontal_step(col: int, width: int) -> int:
    return -1 if int(col) <= int(width) - 1 - int(col) else 1


def _marker_records(grid: np.ndarray, divider_row: int) -> list[dict]:
    mask = (grid == RED) | (grid == BLUE)
    markers = []
    for r, c in np.argwhere(mask):
        color = int(grid[r, c])
        markers.append(
            {
                "color": color,
                "r": int(r),
                "c": int(c),
                "side": _side_of(int(r), int(divider_row)),
            }
        )
    markers.sort(key=lambda item: item["c"])
    assert len({item["c"] for item in markers}) == len(markers), "marker columns must be unique"
    return markers


def _behavior(marker_color: int, *, swapped: bool = False) -> bool:
    toward_divider = int(marker_color) == RED
    return not toward_divider if swapped else toward_divider


def _path_spec(
    marker: dict,
    shape: tuple[int, int],
    divider_row: int,
    *,
    swapped: bool = False,
    reverse_horizontal: bool = False,
) -> tuple[int, int, int, int]:
    height, width = map(int, shape)
    r = int(marker["r"])
    c = int(marker["c"])
    toward_divider = _behavior(int(marker["color"]), swapped=swapped)

    if toward_divider:
        vertical_step = 1 if r < divider_row else -1
        target_row = divider_row - 1 if r < divider_row else divider_row + 1
        path_color = TEAL
    else:
        vertical_step = -1 if r < divider_row else 1
        target_row = 0 if r < divider_row else height - 1
        path_color = YELLOW

    horizontal_step = _horizontal_step(c, width)
    if reverse_horizontal:
        horizontal_step *= -1
    return int(path_color), int(vertical_step), int(target_row), int(horizontal_step)


def _simulate(
    grid: np.ndarray,
    *,
    ignore_paths: bool = False,
    reverse: bool = False,
    swapped: bool = False,
    reverse_horizontal: bool = False,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    divider_row = _find_divider_row(g)
    markers = _marker_records(g, divider_row)
    if reverse:
        markers = list(reversed(markers))

    out = g.copy()
    original_blocked = g != BG
    paths: list[list[tuple[int, int]]] = []

    for marker in markers:
        path_color, vertical_step, target_row, horizontal_step = _path_spec(
            marker, g.shape, divider_row, swapped=swapped, reverse_horizontal=reverse_horizontal
        )
        r = int(marker["r"])
        c = int(marker["c"])
        cells: list[tuple[int, int]] = []

        reached_corner = False
        rr = r + vertical_step
        while 0 <= rr < g.shape[0]:
            blocked = bool(original_blocked[rr, c]) if ignore_paths else int(out[rr, c]) != BG
            if blocked:
                break
            out[rr, c] = path_color
            cells.append((int(rr), c))
            if rr == target_row:
                reached_corner = True
                break
            rr += vertical_step

        if reached_corner:
            cc = c + horizontal_step
            while 0 <= cc < g.shape[1]:
                blocked = (
                    bool(original_blocked[target_row, cc]) if ignore_paths else int(out[target_row, cc]) != BG
                )
                if blocked:
                    break
                out[target_row, cc] = path_color
                cells.append((target_row, int(cc)))
                cc += horizontal_step

        paths.append(cells)

    return out, paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw color-directed L paths from single-pixel markers relative to a grey divider, with left-to-right blocking.

    Input structure:
    - The grid has a black background and exactly one full-width horizontal grey divider row.
    - Above and below the divider are isolated red and blue single-pixel markers, with no two markers sharing a column.
    - Markers stay off the divider and away from the top and bottom edges.

    Rule:
    - Process markers from left to right by column.
    - A red marker draws a teal L path vertically toward the divider until the adjacent row, then horizontally toward the nearer left/right edge, with ties going left.
    - A blue marker draws a yellow L path vertically away from the divider until the outer edge row, then horizontally toward the nearer left/right edge, again with ties going left.
    - The marker cell itself is not recolored, and any segment stops before the first non-background cell, including the divider, another marker, or an already drawn path.

    Key insight:
    - The grey divider fixes the vertical reference, marker color chooses whether the L heads toward or away from that reference, and the visible result depends on incremental left-to-right drawing because earlier same-row paths can block later corners.
    """
    out, _paths = _simulate(np.asarray(grid, dtype=int))
    return out


def _sample_total_markers(rng: np.random.Generator) -> int:
    return int(rng.choice(np.array([5, 6, 7, 8]), p=np.array([0.15, 0.35, 0.30, 0.20])))


def _allocate_counts(
    rng: np.random.Generator, total_markers: int
) -> tuple[dict[tuple[int, int], int], tuple[int, int], tuple[int, int] | None]:
    counts = {cat: 1 for cat in CATEGORIES}
    extras = int(total_markers) - len(CATEGORIES)

    primary = CATEGORIES[int(rng.integers(len(CATEGORIES)))]
    counts[primary] += 1
    extras -= 1

    secondary: tuple[int, int] | None = None
    if extras > 0 and rng.random() < 0.75:
        candidates = [cat for cat in CATEGORIES if cat[0] != primary[0]]
        secondary = candidates[int(rng.integers(len(candidates)))]
        counts[secondary] += 1
        extras -= 1

    while extras > 0:
        preferred = [primary]
        if secondary is not None:
            preferred.append(secondary)
        candidates = [cat for cat in preferred if counts[cat] < 3]
        if not candidates:
            candidates = [cat for cat in CATEGORIES if counts[cat] < 3]
        chosen = candidates[int(rng.integers(len(candidates)))]
        counts[chosen] += 1
        extras -= 1

    return counts, primary, secondary


def _assign_rows(
    rng: np.random.Generator, height: int, divider_row: int
) -> dict[tuple[int, int], int]:
    above_rows = rng.choice(np.arange(1, divider_row - 1), size=2, replace=False)
    below_rows = rng.choice(np.arange(divider_row + 2, height - 1), size=2, replace=False)
    return {
        (RED, ABOVE): int(above_rows[0]),
        (BLUE, ABOVE): int(above_rows[1]),
        (RED, BELOW): int(below_rows[0]),
        (BLUE, BELOW): int(below_rows[1]),
    }


def _left_columns(width: int) -> list[int]:
    return list(range(1, (int(width) - 1) // 2 + 1))


def _right_columns(width: int) -> list[int]:
    start = (int(width) - 1) // 2 + 1
    return list(range(start, int(width) - 1))


def _take_columns(rng: np.random.Generator, pool: list[int], count: int) -> list[int] | None:
    if len(pool) < int(count):
        return None
    chosen = sorted(int(x) for x in rng.choice(np.array(pool), size=int(count), replace=False))
    chosen_set = set(chosen)
    pool[:] = [value for value in pool if value not in chosen_set]
    return chosen


def _assign_columns(
    rng: np.random.Generator,
    width: int,
    counts: dict[tuple[int, int], int],
    primary: tuple[int, int],
    secondary: tuple[int, int] | None,
) -> dict[tuple[int, int], list[int]] | None:
    left_pool = _left_columns(width)
    right_pool = _right_columns(width)
    by_cat = {cat: [] for cat in CATEGORIES}

    primary_cols = _take_columns(rng, right_pool, counts[primary])
    if primary_cols is None:
        return None
    by_cat[primary] = primary_cols

    if secondary is not None and counts[secondary] > 1:
        preferred_pool = left_pool if rng.random() < 0.7 else right_pool
        backup_pool = right_pool if preferred_pool is left_pool else left_pool
        cols = _take_columns(rng, preferred_pool, counts[secondary])
        if cols is None:
            cols = _take_columns(rng, backup_pool, counts[secondary])
        if cols is None:
            return None
        by_cat[secondary] = cols

    single_cats = [cat for cat in CATEGORIES if not by_cat[cat]]
    has_left_direction = any(any(_horizontal_step(col, width) == -1 for col in cols) for cols in by_cat.values())
    must_left_cat = None
    if not has_left_direction and single_cats:
        must_left_cat = single_cats[int(rng.integers(len(single_cats)))]

    center_col = width // 2 if width % 2 == 1 else None
    tie_cat = None
    if (
        center_col is not None
        and must_left_cat is not None
        and center_col in left_pool
        and rng.random() < 0.35
    ):
        tie_cat = must_left_cat
        left_pool.remove(center_col)

    for cat in single_cats:
        if tie_cat == cat:
            by_cat[cat] = [int(center_col)]
            continue
        if must_left_cat == cat:
            cols = _take_columns(rng, left_pool, 1)
            if cols is None:
                return None
            by_cat[cat] = cols
            continue

        if left_pool and right_pool:
            target_pool = left_pool if rng.random() < 0.5 else right_pool
        else:
            target_pool = left_pool if left_pool else right_pool
        cols = _take_columns(rng, target_pool, 1)
        if cols is None:
            return None
        by_cat[cat] = cols

    all_cols = [col for cols in by_cat.values() for col in cols]
    if len(all_cols) != len(set(all_cols)):
        return None
    if {-1, 1} != {_horizontal_step(col, width) for col in all_cols}:
        return None
    return by_cat


def _build_grid(
    height: int,
    width: int,
    divider_row: int,
    rows_by_cat: dict[tuple[int, int], int],
    cols_by_cat: dict[tuple[int, int], list[int]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[divider_row, :] = GREY
    for (color, _side), cols in cols_by_cat.items():
        row = rows_by_cat[(color, _side)]
        for col in cols:
            grid[row, col] = color
    return grid


def _validate_structure(grid: np.ndarray) -> tuple[int, list[dict]]:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    height, width = map(int, grid.shape)
    assert max(height, width) <= 30
    assert _get_bg(grid) == BG
    divider_row = _find_divider_row(grid)
    markers = _marker_records(grid, divider_row)
    assert 5 <= len(markers) <= 8
    counts = {RED: 0, BLUE: 0}
    category_counts = {cat: 0 for cat in CATEGORIES}
    for marker in markers:
        color = int(marker["color"])
        counts[color] += 1
        category_counts[(color, marker["side"])] += 1
        assert 0 < marker["r"] < height - 1
        assert marker["r"] not in {divider_row - 1, divider_row, divider_row + 1}
    assert counts[RED] >= 2 and counts[BLUE] >= 2
    assert all(category_counts[cat] >= 1 for cat in CATEGORIES)
    return divider_row, markers


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        divider_row, markers = _validate_structure(grid)
    except AssertionError:
        return False

    out_main = main(grid)
    out_sim, actual_paths = _simulate(grid)
    if not np.array_equal(out_main, out_sim):
        return False

    _no_path_blocking, free_paths = _simulate(grid, ignore_paths=True)

    swapped_mapping, _ = _simulate(grid, swapped=True)
    if np.array_equal(out_sim, swapped_mapping):
        return False

    reversed_horizontal, _ = _simulate(grid, reverse_horizontal=True)
    if np.array_equal(out_sim, reversed_horizontal):
        return False

    if TEAL not in set(map(int, np.unique(out_sim))) or YELLOW not in set(map(int, np.unique(out_sim))):
        return False
    if np.array_equal(out_sim, grid):
        return False

    if not any(len(actual) < len(free) for actual, free in zip(actual_paths, free_paths)):
        return False

    directions = {_horizontal_step(marker["c"], grid.shape[1]) for marker in markers}
    if directions != {-1, 1}:
        return False

    right_blocked_groups = 0
    by_category: dict[tuple[int, int], list[dict]] = {cat: [] for cat in CATEGORIES}
    for marker in markers:
        by_category[(int(marker["color"]), int(marker["side"]))].append(marker)
    for cat_markers in by_category.values():
        if len(cat_markers) < 2:
            continue
        rows = {marker["r"] for marker in cat_markers}
        dirs = {_horizontal_step(marker["c"], grid.shape[1]) for marker in cat_markers}
        if len(rows) == 1 and dirs == {1}:
            right_blocked_groups += 1
    if right_blocked_groups == 0:
        return False

    if divider_row <= 3 or divider_row >= grid.shape[0] - 4:
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(13, 20))
    width = int(rng.integers(13, 20))
    divider_row = int(rng.integers(4, height - 4))

    total_markers = _sample_total_markers(rng)
    counts, primary, secondary = _allocate_counts(rng, total_markers)
    rows_by_cat = _assign_rows(rng, height, divider_row)
    cols_by_cat = _assign_columns(rng, width, counts, primary, secondary)
    if cols_by_cat is None:
        return None

    grid = _build_grid(height, width, divider_row, rows_by_cat, cols_by_cat)
    return grid if _validate_semantics(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
