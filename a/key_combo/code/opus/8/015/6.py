import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

GEN_ATTEMPTS = 500


def _guide_span(grid: np.ndarray) -> tuple[int, int, int]:
    greens = np.argwhere(grid == GREEN)
    assert greens.size > 0, "expected a green guide bar"
    guide_cols = set(map(int, greens[:, 1]))
    assert len(guide_cols) == 1, "guide bar must occupy exactly one column"
    guide_col = next(iter(guide_cols))

    fg_rows = np.where(np.any(grid != BG, axis=1))[0]
    assert fg_rows.size > 0, "expected non-background structure"
    top = int(fg_rows[0])
    bottom = int(fg_rows[-1])
    assert top < bottom, "guide span must cover multiple rows"
    return guide_col, top, bottom


def _scatter_pixels(grid: np.ndarray) -> list[tuple[int, int, int, str]]:
    guide_col, top, bottom = _guide_span(grid)
    pixels: list[tuple[int, int, int, str]] = []

    for color, side in ((RED, "L"), (BLUE, "R")):
        coords = np.argwhere(grid == color)
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            assert top <= rr <= bottom, "scatter pixel row must lie inside the guide span"
            if side == "L":
                assert cc < guide_col, "red pixels must stay left of the guide"
            else:
                assert cc > guide_col, "blue pixels must stay right of the guide"
            pixels.append((rr, cc, int(color), side))

    rows = [p[0] for p in pixels]
    assert len(rows) == len(set(rows)), "scatter pixels must use distinct rows"
    return pixels


def _ray_cells(start_r: int, start_c: int, dr: int, dc: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    rr = int(start_r) + int(dr)
    cc = int(start_c) + int(dc)
    cells = []
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += int(dr)
        cc += int(dc)
    return cells


def _survivors(grid: np.ndarray) -> list[tuple[int, int, str]]:
    guide_col, _top, _bottom = _guide_span(grid)
    survivors: list[tuple[int, int, str]] = []
    for row, _col, color, side in _scatter_pixels(grid):
        if int(grid[row, guide_col]) == GREEN:
            survivors.append((int(row), int(color), side))
    survivors.sort(key=lambda item: (int(item[0]), 0 if item[2] == "L" else 1))
    return survivors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project scattered pixels onto a transient guide bar, erase those that land in gaps, and emit diagonal rays from the surviving launch points.

    Input structure:
    - The background is black.
    - A single vertical green guide bar occupies one column and may contain black gaps.
    - Red singleton pixels lie strictly to the left of the guide and blue singleton pixels lie strictly to the right, with at most one scattered pixel per row.

    Rule:
    - For each red or blue scattered pixel, look at the guide-bar column on the same row.
    - If that guide position is a green bar cell, move the pixel to the cell immediately adjacent to the guide on its original side; if it is a black gap, erase the pixel.
    - From each surviving adjacent launch point, draw two diagonal rays of the same color until the grid edge: red launches go up-right and down-right, blue launches go up-left and down-left.
    - Process launch points from top to bottom so lower sources overwrite earlier ray pixels where diagonals cross, and erase the guide bar itself from the final output.

    Key insight:
    - The green bar is only a temporary filter and anchor: its gaps decide which pixels survive, and each surviving pixel remembers its original side only through the direction of the emitted diagonals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = map(int, g.shape)
    guide_col, _top, _bottom = _guide_span(g)
    out = np.full((h, w), BG, dtype=int)

    for row, color, side in _survivors(g):
        start_c = guide_col - 1 if side == "L" else guide_col + 1
        out[row, start_c] = int(color)
        directions = [(-1, 1), (1, 1)] if side == "L" else [(-1, -1), (1, -1)]
        for dr, dc in directions:
            for rr, cc in _ray_cells(row, start_c, dr, dc, g.shape):
                out[rr, cc] = int(color)

    return out


def _emission_room(row: int, guide_col: int, shape: tuple[int, int], side: str) -> bool:
    h, w = map(int, shape)
    start_c = guide_col - 1 if side == "L" else guide_col + 1
    horiz = (w - 1 - start_c) if side == "L" else start_c
    up = min(int(row), int(horiz))
    down = min(int(h - 1 - row), int(horiz))
    return up >= 2 and down >= 2


def _spaced_sample(
    rng: np.random.Generator,
    pool: list[int],
    count: int,
    locked: list[int],
) -> list[int] | None:
    locked = sorted(set(map(int, locked)))
    if any(abs(a - b) < 2 for a, b in zip(locked, locked[1:])):
        return None

    for _ in range(200):
        chosen = list(locked)
        available = [int(r) for r in pool if all(abs(int(r) - x) >= 2 for x in chosen)]
        rng.shuffle(available)
        for row in available:
            if len(chosen) == count:
                break
            if all(abs(int(row) - x) >= 2 for x in chosen):
                chosen.append(int(row))
        if len(chosen) == count:
            return sorted(chosen)
    return None


def _left_col(row: int, guide_col: int) -> int:
    span = int(guide_col)
    dist = 1 + ((3 * int(row) + int(guide_col)) % span)
    return int(guide_col - dist)


def _right_col(row: int, guide_col: int, width: int) -> int:
    span = int(width - guide_col - 1)
    dist = 1 + ((2 * int(row) + int(width)) % span)
    return int(guide_col + dist)


def _crossing_exists(entries: list[tuple[int, int, str]], guide_col: int, shape: tuple[int, int]) -> bool:
    red_cells = set()
    blue_cells = set()
    for row, color, side in entries:
        start_c = guide_col - 1 if side == "L" else guide_col + 1
        cells = {(int(row), start_c)}
        directions = [(-1, 1), (1, 1)] if side == "L" else [(-1, -1), (1, -1)]
        for dr, dc in directions:
            cells.update(_ray_cells(row, start_c, dr, dc, shape))
        if color == RED:
            red_cells.update(cells)
        else:
            blue_cells.update(cells)
    return bool(red_cells & blue_cells)


def _validate_instance(grid: np.ndarray, force_cross: bool) -> None:
    g = np.asarray(grid, dtype=int)
    guide_col, top, bottom = _guide_span(g)
    pixels = _scatter_pixels(g)
    assert max(g.shape) <= 30
    assert 1 <= guide_col <= g.shape[1] - 2
    assert top >= 1 and bottom <= g.shape[0] - 2

    red_rows = {row for row, _c, color, _side in pixels if color == RED}
    blue_rows = {row for row, _c, color, _side in pixels if color == BLUE}
    gap_rows = {row for row in range(top, bottom + 1) if int(g[row, guide_col]) != GREEN}
    assert len(gap_rows) == 2
    assert len(red_rows & gap_rows) >= 1
    assert len(blue_rows & gap_rows) >= 1

    survivors = _survivors(g)
    red_survivors = [entry for entry in survivors if entry[1] == RED]
    blue_survivors = [entry for entry in survivors if entry[1] == BLUE]
    assert len(red_survivors) >= 2
    assert len(blue_survivors) >= 2

    left_dists = [guide_col - c for row, c, color, _side in pixels if color == RED]
    right_dists = [c - guide_col for row, c, color, _side in pixels if color == BLUE]
    assert max(left_dists) > 1 and max(right_dists) > 1

    for row, color, side in survivors:
        assert _emission_room(row, guide_col, g.shape, side)

    if force_cross:
        assert _crossing_exists(survivors, guide_col, g.shape)

    out = main(g)
    assert out.shape == g.shape
    assert GREEN not in set(map(int, np.unique(out)))
    assert RED in out and BLUE in out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(14, 20))
    w = int(rng.integers(14, 20))

    guide_min = 4
    guide_max = min(6, w - 5)
    if guide_max < guide_min:
        return None
    guide_col = int(rng.integers(guide_min, guide_max + 1))

    top = int(rng.integers(1, 3))
    bottom = int(rng.integers(h - 4, h - 1))
    if bottom - top < 8:
        return None

    usable_rows = [
        row
        for row in range(top + 1, bottom)
        if _emission_room(row, guide_col, (h, w), "L") and _emission_room(row, guide_col, (h, w), "R")
    ]
    if len(usable_rows) < 6:
        return None

    n_red_survive = int(rng.integers(2, 4))
    n_blue_survive = int(rng.integers(2, 4))
    total_rows = n_red_survive + n_blue_survive + 2
    force_cross = bool(rng.integers(0, 2))

    locked_rows: list[int] = []
    locked_red: list[int] = []
    locked_blue: list[int] = []
    if force_cross:
        cross_pairs = []
        for row in usable_rows:
            if row + 2 in usable_rows:
                cross_pairs.append((int(row), int(row + 2)))
                cross_pairs.append((int(row + 2), int(row)))
        if not cross_pairs:
            return None
        cross_red, cross_blue = cross_pairs[int(rng.integers(0, len(cross_pairs)))]
        locked_rows = [cross_red, cross_blue]
        locked_red = [cross_red]
        locked_blue = [cross_blue]

    rows = _spaced_sample(rng, usable_rows, total_rows, locked_rows)
    if rows is None:
        return None

    remaining = [row for row in rows if row not in locked_rows]
    rng.shuffle(remaining)
    if len(remaining) < total_rows - len(locked_rows):
        return None

    red_gap_row = int(remaining.pop())
    blue_gap_row = int(remaining.pop())

    red_need = n_red_survive - len(locked_red)
    blue_need = n_blue_survive - len(locked_blue)
    if len(remaining) != red_need + blue_need:
        return None

    rng.shuffle(remaining)
    red_rows = locked_red + [int(x) for x in remaining[:red_need]]
    blue_rows = locked_blue + [int(x) for x in remaining[red_need : red_need + blue_need]]
    if len(set(red_rows)) != len(red_rows) or len(set(blue_rows)) != len(blue_rows):
        return None

    grid = np.zeros((h, w), dtype=int)
    gap_rows = {red_gap_row, blue_gap_row}
    for row in range(top, bottom + 1):
        if row not in gap_rows:
            grid[row, guide_col] = GREEN

    red_rows_all = sorted(red_rows + [red_gap_row])
    blue_rows_all = sorted(blue_rows + [blue_gap_row])
    for row in red_rows_all:
        col = _left_col(row, guide_col)
        grid[row, col] = RED
    for row in blue_rows_all:
        col = _right_col(row, guide_col, w)
        grid[row, col] = BLUE

    try:
        _validate_instance(grid, force_cross)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid guide-bar ray instance")
