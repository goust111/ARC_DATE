import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
ORANGE = 7

GRID_MIN = 8
GRID_MAX = 15
MAX_ATTEMPTS = 400


def _find_bar_rows(grid: np.ndarray) -> dict[int, int]:
    bar_rows: dict[int, int] = {}
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == BLUE):
            bar_rows[int(r)] = BLUE
        elif np.all(row == GREEN):
            bar_rows[int(r)] = GREEN
        else:
            assert not np.any((row == BLUE) | (row == GREEN)), "bars must occupy whole rows"
    return bar_rows


def _trace_path(
    start_r: int,
    start_c: int,
    bar_rows: dict[int, int],
    out_shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    height, width = out_shape
    drift = 0
    step = 0
    painted: list[tuple[int, int]] = []
    hits: list[tuple[int, int, int]] = []

    while True:
        rr = int(start_r + step)
        cc = int(start_c + step + drift)
        if rr < 0 or rr >= height or cc < 0 or cc >= width:
            break

        bar_color = bar_rows.get(rr)
        if bar_color == BLUE:
            hits.append((rr, cc, BLUE))
            drift += 1
        elif bar_color == GREEN:
            hits.append((rr, cc, GREEN))
            drift -= 1
        else:
            painted.append((rr, cc))
        step += 1

    return painted, hits


def main(grid: np.ndarray) -> np.ndarray:
    """Copy each orange motif pixel along the down-right diagonal while full-width blue/green rows bend the path sideways.

    Input structure:
    - The input is a black grid containing a sparse orange motif near the upper area.
    - One or two full-width horizontal obstacle rows appear below the motif.
    - A blue obstacle row and a green obstacle row are visually distinct control rows.

    Rule:
    - Create a black output canvas with twice the input height and twice the input width.
    - Copy every blue or green full-width row to the same row index in the output, stretched across the doubled width.
    - For each orange pixel at input position (r, c), follow the diagonal trial positions (r+k, c+k+d) with step counter k and current drift d starting at 0.
    - On a blue obstacle row, place nothing and increase d by 1; on a green obstacle row, place nothing and decrease d by 1; on any other in-bounds row, place orange at the current trial position; stop once the trial position leaves the output.

    Key insight:
    - The orange pixels are not scaled or flood-filled: each one traces its own diagonal ray, and obstacle rows only change the accumulated sideways drift before the ray continues.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0

    allowed = {BLACK, BLUE, GREEN, ORANGE}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors present"

    orange_coords = np.argwhere(g == ORANGE)
    assert len(orange_coords) > 0, "expected at least one orange pixel"

    bar_rows = _find_bar_rows(g)

    out = np.full((2 * height, 2 * width), BLACK, dtype=int)
    for r, color in bar_rows.items():
        out[int(r), :] = int(color)

    for r, c in orange_coords:
        painted, _hits = _trace_path(int(r), int(c), bar_rows, out.shape)
        for rr, cc in painted:
            out[int(rr), int(cc)] = ORANGE

    return out


def _is_irregular_shape(cells: list[tuple[int, int]]) -> bool:
    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    if len(rows) < 2 or len(cols) < 2:
        return False

    if len({r - c for r, c in cells}) == 1:
        return False
    if len({r + c for r, c in cells}) == 1:
        return False

    r0 = min(r for r, _c in cells)
    r1 = max(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    c1 = max(c for _r, c in cells)
    if (r1 - r0 + 1) * (c1 - c0 + 1) == len(cells):
        return False

    return True


def _sample_motif(
    rng: random.Random,
    height: int,
    width: int,
    n_bars: int,
) -> list[tuple[int, int]] | None:
    max_motif_row = height - (2 * n_bars + 2)
    box_h_max = min(5, max_motif_row + 1)
    right_limit = min(width - 3, 2 * width - height - 3)
    box_w_max = min(5, right_limit)
    if box_h_max < 3 or box_w_max < 3:
        return None

    box_h = rng.randint(3, box_h_max)
    box_w = rng.randint(3, box_w_max)
    top_max = max_motif_row - box_h + 1
    left_min = 1
    left_max = right_limit - box_w + 1
    if top_max < 0 or left_max < left_min:
        return None

    top = rng.randint(0, min(3, top_max))
    left = rng.randint(left_min, left_max)
    cell_count = rng.randint(3, min(5, box_h * box_w - 1))
    local_cells = [(r, c) for r in range(box_h) for c in range(box_w)]

    for _ in range(80):
        picked = rng.sample(local_cells, cell_count)
        if not _is_irregular_shape(picked):
            continue
        return [(top + r, left + c) for r, c in picked]
    return None


def _sample_bar_rows(
    rng: random.Random,
    height: int,
    motif_rows: list[int],
    n_bars: int,
) -> list[int] | None:
    rows: list[int] = []
    next_min = max(motif_rows) + 2
    for idx in range(n_bars):
        remaining = n_bars - idx - 1
        latest = height - 2 - 2 * remaining
        earliest = next_min
        latest = min(latest, earliest + 2)
        if earliest > latest:
            return None
        row = rng.randint(earliest, latest)
        rows.append(row)
        next_min = row + 2
    return rows


def _sample_bar_colors(rng: random.Random, n_bars: int) -> list[int]:
    if n_bars == 1:
        return [BLUE if rng.random() < 0.5 else GREEN]
    if rng.random() < 0.5:
        return [BLUE, GREEN]
    return [GREEN, BLUE]


def _validate_grid(grid: np.ndarray) -> bool:
    out = main(grid)
    height, width = map(int, grid.shape)
    if out.shape != (2 * height, 2 * width):
        return False

    bar_rows = _find_bar_rows(grid)
    orange_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == ORANGE)]
    if len(orange_coords) < 3:
        return False

    hit_counts = {row: 0 for row in bar_rows}
    reaches_after_last_bar = False
    last_bar = max(bar_rows)
    for r, c in orange_coords:
        painted, hits = _trace_path(r, c, bar_rows, out.shape)
        for row in {rr for rr, _cc, _color in hits}:
            hit_counts[row] += 1
        if any(rr > last_bar for rr, _cc in painted):
            reaches_after_last_bar = True

    if any(count == 0 for count in hit_counts.values()):
        return False
    if not reaches_after_last_bar:
        return False
    if not np.any(out[height:, :] == ORANGE):
        return False

    for row, color in bar_rows.items():
        if not np.all(out[row, :] == color):
            return False
        if row <= 0 or row + 1 >= out.shape[0]:
            return False
        if not np.any(out[row - 1, :] == ORANGE):
            return False
        if not np.any(out[row + 1, :] == ORANGE):
            return False

    orange_total = int(np.count_nonzero(out == ORANGE))
    if orange_total <= len(orange_coords) + len(bar_rows):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(max(GRID_MIN, height - 2), GRID_MAX)
    n_bars = 1 if rng.random() < 0.5 else 2

    motif = _sample_motif(rng, height, width, n_bars)
    if motif is None:
        return None

    bar_rows = _sample_bar_rows(rng, height, [r for r, _c in motif], n_bars)
    if bar_rows is None:
        return None
    bar_colors = _sample_bar_colors(rng, n_bars)

    grid = np.zeros((height, width), dtype=int)
    for row, color in zip(bar_rows, bar_colors):
        grid[row, :] = color
    for r, c in motif:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = ORANGE

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-detour ARC input")
