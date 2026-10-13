from __future__ import annotations

import random
from collections import Counter

import numpy as np


ALL_COLORS = tuple(range(10))
GAP_OFFSETS = ((0, -1), (0, 1), (-1, 0), (1, 0))
ARM_BARS = (
    ((0, -3), (0, -2)),  # left horizontal bar
    ((0, 2), (0, 3)),    # right horizontal bar
    ((-3, 0), (-2, 0)),  # upper vertical bar
    ((2, 0), (3, 0)),    # lower vertical bar
)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = shape
    return 0 <= int(r) < int(height) and 0 <= int(c) < int(width)


def _cells_from_offsets(r: int, c: int, offsets: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for dr, dc in offsets]


def _bars_for_center(r: int, c: int) -> list[list[tuple[int, int]]]:
    return [_cells_from_offsets(r, c, bar) for bar in ARM_BARS]


def _gaps_for_center(r: int, c: int) -> list[tuple[int, int]]:
    return _cells_from_offsets(r, c, GAP_OFFSETS)


def _foreground_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(grid != int(bg)):
        sr = int(sr)
        sc = int(sc)
        if bool(seen[sr, sc]):
            continue

        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                if bool(seen[nr, nc]) or int(grid[nr, nc]) == int(bg):
                    continue
                seen[nr, nc] = True
                stack.append((nr, nc))

        comps.append(np.asarray(coords, dtype=int))

    return comps


def _assert_two_cell_bars(grid: np.ndarray, bg: int) -> None:
    comps = _foreground_components(grid, bg)
    assert comps, "expected foreground two-pixel bars"

    for coords in comps:
        assert coords.shape == (2, 2), "each foreground object must be a two-pixel bar"
        (r0, c0), (r1, c1) = coords
        same_row = int(r0) == int(r1) and abs(int(c0) - int(c1)) == 1
        same_col = int(c0) == int(c1) and abs(int(r0) - int(r1)) == 1
        assert same_row or same_col, "bar cells must be adjacent horizontally or vertically"
        assert int(grid[r0, c0]) == int(grid[r1, c1]), "each two-pixel bar must be monochrome"


def _four_bar_colors(grid: np.ndarray, r: int, c: int, bg: int) -> list[int] | None:
    if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != int(bg):
        return None

    for gr, gc in _gaps_for_center(r, c):
        if not _in_bounds(grid.shape, gr, gc) or int(grid[gr, gc]) != int(bg):
            return None

    colors: list[int] = []
    for bar in _bars_for_center(r, c):
        if any(not _in_bounds(grid.shape, br, bc) for br, bc in bar):
            return None
        vals = [int(grid[br, bc]) for br, bc in bar]
        if vals[0] == int(bg) or vals[0] != vals[1]:
            return None
        colors.append(vals[0])
    return colors


def _is_valid_center(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    colors = _four_bar_colors(grid, r, c, bg)
    return colors is not None and len(set(colors)) == 1


def _is_mixed_four_bar_center(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    colors = _four_bar_colors(grid, r, c, bg)
    return colors is not None and len(set(colors)) > 1


def _find_centers(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    height, width = grid.shape
    return [
        (r, c)
        for r in range(3, height - 3)
        for c in range(3, width - 3)
        if _is_valid_center(grid, r, c, bg)
    ]


def _rectangle_from_centers(centers: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    assert len(centers) == 4, "expected exactly four implied centers"
    rows = sorted({r for r, _c in centers})
    cols = sorted({c for _r, c in centers})
    assert len(rows) == 2 and len(cols) == 2, "centers must form two rows and two columns"
    expected = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    assert set(centers) == expected, "the four implied centers must be the rectangle corners"
    return int(rows[0]), int(cols[0]), int(rows[1]), int(cols[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the grid using the four missing centers of same-color implied plus signs as corners.

    Input structure:
    - The dominant color is the background.
    - Every foreground object is an isolated monochrome two-pixel horizontal or vertical bar.
    - Exactly four hidden centers have one background gap and then a two-pixel bar in each cardinal direction, with the four bars around each center sharing one color.
    - Other bars may form mixed-color near crosses or scattered distractors, but they do not satisfy the same-color condition.

    Rule:
    - For each background cell, check the four aligned two-pixel bars at offsets 2-3 left, right, up, and down, with the adjacent gap cells still background.
    - Keep only cells whose four bars are all the same non-background color.
    - The four valid hidden centers must occupy two rows and two columns; crop the input to the inclusive rectangle bounded by those rows and columns.

    Key insight:
    - The centers are never drawn or marked; same-color implied pluses identify four invisible corner points that define the crop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    _assert_two_cell_bars(g, bg)

    centers = _find_centers(g, bg)
    r0, c0, r1, c1 = _rectangle_from_centers(centers)
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


# --------------- generator ---------------

GEN_ATTEMPTS = 400
GRID_MIN = 20
GRID_MAX = 30
CENTER_MARGIN = 4
MIN_CENTER_SEPARATION = 8
TRUE_PLUS_COUNT = 4


def _choose_spaced_pair(rng: random.Random, low: int, high: int) -> tuple[int, int] | None:
    pairs = [(a, b) for a in range(low, high + 1) for b in range(a + MIN_CENTER_SEPARATION, high + 1)]
    if not pairs:
        return None
    return rng.choice(pairs)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS4]


def _can_place_bar_set(
    grid: np.ndarray,
    bars: list[list[tuple[int, int]]],
    bg: int,
    reserved_bg: set[tuple[int, int]],
) -> bool:
    cell_to_bar: dict[tuple[int, int], int] = {}
    for idx, bar in enumerate(bars):
        if len(bar) != 2:
            return False
        for cell in bar:
            if cell in cell_to_bar:
                return False
            cell_to_bar[cell] = idx

    for r, c in cell_to_bar:
        if not _in_bounds(grid.shape, r, c):
            return False
        if (r, c) in reserved_bg or int(grid[r, c]) != int(bg):
            return False

    for cell, bar_idx in cell_to_bar.items():
        for nr, nc in _neighbors4(cell):
            other_idx = cell_to_bar.get((nr, nc))
            if other_idx is not None:
                if other_idx != bar_idx:
                    return False
                continue
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != int(bg):
                return False

    return True


def _paint_bar_set(grid: np.ndarray, bars: list[list[tuple[int, int]]], colors: list[int]) -> None:
    assert len(bars) == len(colors)
    for bar, color in zip(bars, colors):
        for r, c in bar:
            grid[r, c] = int(color)


def _clear_bar_set(grid: np.ndarray, bars: list[list[tuple[int, int]]], bg: int) -> None:
    for bar in bars:
        for r, c in bar:
            grid[r, c] = int(bg)


def _center_reserved_cells(r: int, c: int) -> set[tuple[int, int]]:
    return {(r, c), *_gaps_for_center(r, c)}


def _can_use_hidden_center(
    grid: np.ndarray,
    r: int,
    c: int,
    bg: int,
    reserved_bg: set[tuple[int, int]],
) -> bool:
    hidden = _center_reserved_cells(r, c)
    if any((not _in_bounds(grid.shape, rr, cc)) or int(grid[rr, cc]) != int(bg) for rr, cc in hidden):
        return False
    if bool(hidden & reserved_bg):
        return False
    return _can_place_bar_set(grid, _bars_for_center(r, c), bg, reserved_bg)


def _same_center_set(grid: np.ndarray, bg: int, true_centers: set[tuple[int, int]]) -> bool:
    return set(_find_centers(grid, bg)) == set(true_centers)


def _place_true_pluses(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    colors: list[int],
    bg: int,
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for (r, c), color in zip(centers, colors):
        bars = _bars_for_center(r, c)
        if not _can_place_bar_set(grid, bars, bg, reserved_bg):
            return False
        _paint_bar_set(grid, bars, [int(color)] * len(bars))
    return True


def _place_mixed_cross(
    grid: np.ndarray,
    rng: random.Random,
    bg: int,
    palette: list[int],
    true_centers: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    height, width = grid.shape
    for _ in range(220):
        r = rng.randint(3, height - 4)
        c = rng.randint(3, width - 4)
        if (r, c) in true_centers:
            continue
        if not _can_use_hidden_center(grid, r, c, bg, reserved_bg):
            continue

        base = rng.choice(palette)
        other_choices = [x for x in palette if int(x) != int(base)]
        if not other_choices:
            continue
        colors = [int(base)] * len(ARM_BARS)
        for idx in rng.sample(range(len(colors)), rng.choice([1, 2])):
            colors[idx] = int(rng.choice(other_choices))

        bars = _bars_for_center(r, c)
        _paint_bar_set(grid, bars, colors)
        if _same_center_set(grid, bg, true_centers):
            reserved_bg.update(_center_reserved_cells(r, c))
            return True
        _clear_bar_set(grid, bars, bg)

    return False


def _place_scattered_bar(
    grid: np.ndarray,
    rng: random.Random,
    bg: int,
    palette: list[int],
    true_centers: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> str | None:
    height, width = grid.shape
    for _ in range(160):
        orient = rng.choice(["h", "v"])
        if orient == "h":
            r = rng.randrange(height)
            c = rng.randrange(width - 1)
            bars = [[(r, c), (r, c + 1)]]
        else:
            r = rng.randrange(height - 1)
            c = rng.randrange(width)
            bars = [[(r, c), (r + 1, c)]]

        if not _can_place_bar_set(grid, bars, bg, reserved_bg):
            continue
        color = int(rng.choice(palette))
        _paint_bar_set(grid, bars, [color])
        if _same_center_set(grid, bg, true_centers):
            return orient
        _clear_bar_set(grid, bars, bg)

    return None


def _count_mixed_four_bar_centers(grid: np.ndarray, bg: int) -> int:
    height, width = grid.shape
    return sum(
        1
        for r in range(3, height - 3)
        for c in range(3, width - 3)
        if _is_mixed_four_bar_center(grid, r, c, bg)
    )


def _semantic_oracle(grid: np.ndarray, true_centers: set[tuple[int, int]], bg: int) -> bool:
    try:
        _assert_two_cell_bars(grid, bg)
        centers = _find_centers(grid, bg)
        if set(centers) != set(true_centers):
            return False
        r0, c0, r1, c1 = _rectangle_from_centers(centers)
        expected = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        out = main(grid)
    except Exception:
        return False

    if not np.array_equal(out, expected):
        return False
    if out.shape[0] >= grid.shape[0] or out.shape[1] >= grid.shape[1]:
        return False
    if _count_mixed_four_bar_centers(grid, bg) < 1:
        return False
    if int(np.count_nonzero(out != int(bg))) < TRUE_PLUS_COUNT:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice(ALL_COLORS)
    palette = [c for c in ALL_COLORS if c != int(bg)]

    row_pair = _choose_spaced_pair(rng, CENTER_MARGIN, height - CENTER_MARGIN - 1)
    col_pair = _choose_spaced_pair(rng, CENTER_MARGIN, width - CENTER_MARGIN - 1)
    if row_pair is None or col_pair is None:
        return None

    r0, r1 = row_pair
    c0, c1 = col_pair
    centers = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    true_centers = set(centers)

    grid = np.full((height, width), int(bg), dtype=int)
    reserved_bg: set[tuple[int, int]] = set()
    for r, c in centers:
        reserved_bg.update(_center_reserved_cells(r, c))

    center_colors = [int(rng.choice(palette)) for _ in range(TRUE_PLUS_COUNT)]
    if len(set(center_colors)) == 1:
        center_colors[1] = int(rng.choice([c for c in palette if c != center_colors[0]]))

    if not _place_true_pluses(grid, centers, center_colors, bg, reserved_bg):
        return None
    if not _same_center_set(grid, bg, true_centers):
        return None

    mixed_target = rng.randint(1, 3)
    mixed_placed = 0
    for _ in range(mixed_target):
        if _place_mixed_cross(grid, rng, bg, palette, true_centers, reserved_bg):
            mixed_placed += 1
    if mixed_placed < 1:
        return None

    scatter_target = rng.randint(4, 11)
    scatter_orients: list[str] = []
    for _ in range(scatter_target * 50):
        if len(scatter_orients) >= scatter_target:
            break
        orient = _place_scattered_bar(grid, rng, bg, palette, true_centers, reserved_bg)
        if orient is not None:
            scatter_orients.append(orient)

    if len(scatter_orients) < 4:
        return None
    if "h" not in scatter_orients or "v" not in scatter_orients:
        return None

    if _get_bg(grid) != int(bg):
        return None
    if not _semantic_oracle(grid, true_centers, bg):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-corner crop input")
