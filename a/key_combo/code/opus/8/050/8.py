import random
from math import ceil, floor

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

BASE_COLORS = (TEAL, GREEN, GREY)
CONTROL_COLORS = (GREY, PINK, ORANGE, MAROON)
PAIR_MAP = {
    BLACK: BLACK,
    TEAL: RED,
    GREEN: BLUE,
    GREY: YELLOW,
}

GRID_SIDE_CHOICES = (8, 10, 12)
GRID_SIDE_WEIGHTS = (6, 3, 1)
COUNT_CHOICES = (2, 3, 4, 5)
COUNT_WEIGHTS_Q4 = (1, 2, 2, 4)
COUNT_WEIGHTS_Q5P = (2, 2, 2, 2)
BASE_TWO_COLOR_PROB = 0.72
TEMPLATE_FOREGROUND_FRAC = (0.28, 0.62)
MAX_GENERATION_ATTEMPTS = 500
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _tile_to_shape(tile: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    qh, qw = tile.shape
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            out[r, c] = int(tile[r % qh, c % qw])
    return out


def _split_quadrants(grid: np.ndarray):
    h, w = grid.shape
    qh, qw = h // 2, w // 2
    return (
        grid[:qh, :qw],
        grid[:qh, qw:],
        grid[qh:, :qw],
        grid[qh:, qw:],
    )


def _paired_invert(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for src, dst in PAIR_MAP.items():
        out[values == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the top control pixels to size a tiled output, then invert the tiled base where the tiled template is marked.

    Input structure:
    - The input is an even square split into four equal quadrants.
    - Black is the background.
    - The top-left and top-right quadrants are control regions whose non-black pixel counts encode the output height and width.
    - The bottom-left quadrant is a black-background base tile using teal, green, and/or grey foreground pixels.
    - The bottom-right quadrant is a black-background template tile whose foreground color is red.

    Rule:
    - Let H be the number of non-black pixels in the top-left quadrant and W the number of non-black pixels in the top-right quadrant.
    - Tile the bottom-left quadrant over an HxW canvas by modulo row/column lookup.
    - Tile the bottom-right quadrant over the same HxW canvas.
    - Wherever the tiled template is red, replace the tiled base color by its paired inverted color: teal->red, green->blue, grey->yellow, black->black.
    - Wherever the tiled template is black, keep the tiled base color unchanged.

    Key insight:
    - The top half only sets output size, while the lower-right tile acts as an inversion switch over the lower-left tile instead of a visibility mask or direct replacement pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    upper_left, upper_right, base_tile, template_tile = _split_quadrants(g)
    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    assert out_h > 0 and out_w > 0

    base = _tile_to_shape(base_tile, out_h, out_w)
    template = _tile_to_shape(template_tile, out_h, out_w)
    mask = template == RED

    out = base.copy()
    out[mask] = _paired_invert(base)[mask]
    return out


def _all_positions(q: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(q) for c in range(q)]


def _target_count(q: int, frac_range: tuple[float, float], rng: random.Random) -> int:
    area = q * q
    lo = max(1, ceil(area * frac_range[0]))
    hi = min(area - 1, floor(area * frac_range[1]))
    return rng.randint(lo, hi)


def _grow_positions(q: int, k: int, rng: random.Random) -> set[tuple[int, int]]:
    cells = {rng.choice(_all_positions(q))}
    while len(cells) < k:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < q and 0 <= nc < q and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if frontier and rng.random() < 0.88:
            cells.add(rng.choice(frontier))
        else:
            remaining = [p for p in _all_positions(q) if p not in cells]
            cells.add(rng.choice(remaining))
    return cells


def _score_positions(q: int, rng: random.Random, style: str) -> list[tuple[int, int]]:
    positions = _all_positions(q)
    scored = []
    if style == "bands":
        hot_rows = set(rng.sample(list(range(q)), rng.randint(1, min(2, q))))
        hot_cols = set(rng.sample(list(range(q)), rng.randint(1, min(2, q))))
        parity = rng.randrange(2)
        for r, c in positions:
            score = 3 * (r in hot_rows) + 3 * (c in hot_cols) + ((r + c) % 2 == parity)
            scored.append((score + rng.random() * 0.15, (r, c)))
    elif style == "checker":
        parity = rng.randrange(2)
        row_bias = rng.randrange(q)
        col_bias = rng.randrange(q)
        for r, c in positions:
            score = 3 * (((r + c) % 2) == parity) + (r == row_bias) + (c == col_bias)
            scored.append((score + rng.random() * 0.15, (r, c)))
    else:
        center_r = rng.randrange(q)
        center_c = rng.randrange(q)
        for r, c in positions:
            score = -(abs(r - center_r) + abs(c - center_c))
            scored.append((score + rng.random() * 0.15, (r, c)))
    scored.sort(reverse=True)
    return [pos for _, pos in scored]


def _pick_positions(q: int, k: int, rng: random.Random, styles: tuple[str, ...]) -> set[tuple[int, int]]:
    style = rng.choice(styles)
    if style == "grow":
        return _grow_positions(q, k, rng)
    ranked = _score_positions(q, rng, style)
    return set(ranked[:k])


def _assign_base_colors(
    tile: np.ndarray,
    cells: list[tuple[int, int]],
    colors: tuple[int, ...],
    rng: random.Random,
) -> None:
    if len(colors) == 1:
        for r, c in cells:
            tile[r, c] = colors[0]
        return

    mode = rng.choice(("parity", "rows", "cols", "random"))
    row_cut = rng.randrange(tile.shape[0])
    col_cut = rng.randrange(tile.shape[1])
    for r, c in cells:
        if mode == "parity":
            idx = (r + c + rng.randrange(2)) % 2
        elif mode == "rows":
            idx = 0 if r <= row_cut else 1
        elif mode == "cols":
            idx = 0 if c <= col_cut else 1
        else:
            idx = 0 if rng.random() < 0.5 else 1
        tile[r, c] = colors[idx]

    used = {int(tile[r, c]) for r, c in cells}
    if len(used) == 1:
        r, c = rng.choice(cells)
        tile[r, c] = colors[1 if tile[r, c] == colors[0] else 0]


def _make_control_quadrant(q: int, count: int, rng: random.Random) -> np.ndarray:
    quad = np.zeros((q, q), dtype=int)
    styles = ("grow", "scatter", "bands")
    cells = list(_pick_positions(q, count, rng, styles))
    if rng.random() < 0.82:
        colors = (GREY,)
    else:
        n_colors = 2 if rng.random() < 0.75 else 3
        colors = tuple(rng.sample(CONTROL_COLORS, n_colors))
    for idx, (r, c) in enumerate(cells):
        quad[r, c] = colors[idx % len(colors)]
    return quad


def _make_base_tile(q: int, rng: random.Random) -> np.ndarray:
    tile = np.zeros((q, q), dtype=int)
    k = _target_count(q, (0.40, 0.60), rng)
    cells = list(_pick_positions(q, k, rng, ("grow", "bands", "checker", "scatter")))
    n_colors = 2 if rng.random() < BASE_TWO_COLOR_PROB else 1
    colors = tuple(rng.sample(BASE_COLORS, n_colors))
    _assign_base_colors(tile, cells, colors, rng)
    return tile


def _make_template_tile(q: int, rng: random.Random) -> np.ndarray:
    tile = np.zeros((q, q), dtype=int)
    k = _target_count(q, TEMPLATE_FOREGROUND_FRAC, rng)
    cells = _pick_positions(q, k, rng, ("bands", "checker", "grow", "scatter"))
    for r, c in cells:
        tile[r, c] = RED
    return tile


def _count_for_q(q: int, rng: random.Random) -> int:
    weights = COUNT_WEIGHTS_Q4 if q == 4 else COUNT_WEIGHTS_Q5P
    return rng.choices(COUNT_CHOICES, weights=weights, k=1)[0]


def _semantic_ok(grid: np.ndarray) -> bool:
    upper_left, upper_right, base_tile, template_tile = _split_quadrants(grid)
    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    if not (2 <= out_h <= 5 and 2 <= out_w <= 5):
        return False

    q = base_tile.shape[0]
    if not (0.40 <= np.count_nonzero(base_tile) / float(q * q) <= 0.60):
        return False
    if not np.all(np.isin(base_tile, (BLACK,) + BASE_COLORS)):
        return False
    base_colors = sorted(set(np.unique(base_tile).tolist()) - {BLACK})
    if not (1 <= len(base_colors) <= 2):
        return False
    if not np.all(np.isin(template_tile, (BLACK, RED))):
        return False

    base = _tile_to_shape(base_tile, out_h, out_w)
    template = _tile_to_shape(template_tile, out_h, out_w)
    mask = template == RED
    if not mask.any() or mask.all():
        return False

    changed_nonblack = mask & (base != BLACK)
    unchanged_nonblack = (~mask) & (base != BLACK)
    if not changed_nonblack.any() or not unchanged_nonblack.any():
        return False

    if not any(
        np.any(mask & (base == color)) and np.any((~mask) & (base == color))
        for color in BASE_COLORS
    ):
        return False

    visible_pairs = {PAIR_MAP[int(color)] for color in np.unique(base[changed_nonblack]).tolist()}
    if visible_pairs == {RED}:
        return False

    out = main(grid)
    if np.array_equal(out, base):
        return False
    if np.array_equal(out, _paired_invert(base)):
        return False
    if len(np.unique(out)) < 3:
        return False

    if out_h > q or out_w > q:
        if out_h > q and np.array_equal(base[: q, :out_w], base[out_h - q :, :out_w]):
            return False
        if out_w > q and np.array_equal(base[:out_h, : q], base[:out_h, out_w - q :]):
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choices(GRID_SIDE_CHOICES, weights=GRID_SIDE_WEIGHTS, k=1)[0]
    q = side // 2
    out_h = _count_for_q(q, rng)
    out_w = _count_for_q(q, rng)

    base_tile = _make_base_tile(q, rng)
    template_tile = _make_template_tile(q, rng)
    upper_left = _make_control_quadrant(q, out_h, rng)
    upper_right = _make_control_quadrant(q, out_w, rng)

    grid = np.zeros((side, side), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = base_tile
    grid[q:, q:] = template_tile

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
