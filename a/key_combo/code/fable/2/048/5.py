from __future__ import annotations

from typing import Iterable

import numpy as np


BLOCK_SIZE = 3
BACKGROUND = 0
GREY = 5
COLOR_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

CARDINAL_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_DIRECTIONS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

MINI_SIDE_MIN = 5
MINI_SIDE_MAX = 8
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _direction_from_grey(block: np.ndarray, bg: int) -> tuple[int, tuple[int, int], bool]:
    if np.all(block == bg):
        return bg, (0, 0), False

    grey_coords = np.argwhere(block == GREY)
    assert grey_coords.shape == (1, 2), "each non-background block must contain exactly one grey cell"

    colors = [int(color) for color in np.unique(block) if int(color) not in (bg, GREY)]
    assert len(colors) == 1, "each active block must contain exactly one main color"
    color = colors[0]

    assert int(np.sum(block == color)) == 8, "each active block must have eight cells of the main color"
    assert np.all((block == color) | (block == GREY)), "active blocks may contain only the main color and one grey cell"

    local_r, local_c = map(int, grey_coords[0])
    return color, (local_r - 1, local_c - 1), True


def _decode_tiles(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % BLOCK_SIZE == 0 and w % BLOCK_SIZE == 0, "grid dimensions must be multiples of 3"

    bg = _get_bg(g)
    small_h = h // BLOCK_SIZE
    small_w = w // BLOCK_SIZE
    compressed = np.full((small_h, small_w), bg, dtype=int)
    movers: list[tuple[int, int, int, int, int]] = []

    for br in range(small_h):
        for bc in range(small_w):
            r0 = br * BLOCK_SIZE
            c0 = bc * BLOCK_SIZE
            block = g[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
            color, (dr, dc), is_active = _direction_from_grey(block, bg)
            if not is_active:
                assert np.all(block == bg), "background blocks must be uniformly background"
                continue
            compressed[br, bc] = color
            movers.append((br, bc, color, dr, dc))

    assert movers, "expected at least one active block"
    return compressed, movers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress every 3x3 tile to one pixel, then move that pixel one step in the grey-offset direction.

    Input structure:
    - The grid dimensions are multiples of 3, forming a lattice of 3x3 tiles.
    - Every active tile has one main color in eight cells and exactly one grey cell.
    - Empty tiles are entirely background, while the grey cell's offset from tile center
      encodes one of the eight neighboring directions or the zero shift.

    Rule:
    - First collapse each 3x3 tile to one low-resolution pixel: active tiles become their
      main color and empty tiles become background.
    - Then move each non-background low-resolution pixel by one step equal to its tile's
      grey-cell offset relative to the tile center.
    - The source position becomes background, zero-offset pixels stay in place, and the
      generated inputs guarantee that moved pixels do not collide or leave the grid.

    Key insight:
    - The grey cell is not noise to ignore after compression; it is a per-tile motion
      instruction that must still be applied on the compressed image.
    """
    compressed, movers = _decode_tiles(np.asarray(grid, dtype=int))
    bg = _get_bg(np.asarray(grid, dtype=int))
    out = np.full_like(compressed, bg)
    small_h, small_w = compressed.shape

    for r, c, color, dr, dc in movers:
        rr = r + dr
        cc = c + dc
        assert 0 <= rr < small_h and 0 <= cc < small_w, "movement must stay inside the compressed grid"
        assert out[rr, cc] == bg, "movement destinations must be unique"
        out[rr, cc] = color

    return out


def _expand_scene(
    small_shape: tuple[int, int],
    assignments: Iterable[tuple[int, int, int, tuple[int, int]]],
) -> np.ndarray:
    small_h, small_w = map(int, small_shape)
    grid = np.full((small_h * BLOCK_SIZE, small_w * BLOCK_SIZE), BACKGROUND, dtype=int)

    for r, c, color, (dr, dc) in assignments:
        r0 = int(r) * BLOCK_SIZE
        c0 = int(c) * BLOCK_SIZE
        grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = int(color)
        grid[r0 + 1 + int(dr), c0 + 1 + int(dc)] = GREY

    return grid


def _too_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1


def _pick_positions(
    rng: np.random.Generator, small_h: int, small_w: int, directions: list[tuple[int, int]]
) -> list[tuple[int, int, tuple[int, int]]] | None:
    sources: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()
    destinations: set[tuple[int, int]] = set()
    placed: list[tuple[int, int, tuple[int, int]]] = []
    cells = [(r, c) for r in range(small_h) for c in range(small_w)]

    placement_order = sorted(
        directions,
        key=lambda delta: (delta == (0, 0), abs(delta[0]) + abs(delta[1])),
    )

    for dr, dc in placement_order:
        order = rng.permutation(len(cells))
        candidates: list[tuple[int, int]] = []
        for idx in order:
            r, c = cells[int(idx)]
            if (r, c) in occupied:
                continue
            if any(_too_close((r, c), src) for src in sources):
                continue

            rr = r + dr
            cc = c + dc
            if not (0 <= rr < small_h and 0 <= cc < small_w):
                continue

            dst = (rr, cc)
            if dst in destinations:
                continue
            if dst != (r, c) and dst in occupied:
                continue
            candidates.append((r, c))

        if not candidates:
            return None

        r, c = candidates[int(rng.integers(len(candidates)))]
        dst = (r + dr, c + dc)
        sources.append((r, c))
        occupied.add((r, c))
        if dst != (r, c):
            occupied.add(dst)
        destinations.add(dst)
        placed.append((r, c, (dr, dc)))

    rng.shuffle(placed)
    return placed


def _build_low_res_scene(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, tuple[int, int]]]] | None:
    small_h = int(rng.integers(MINI_SIDE_MIN, MINI_SIDE_MAX + 1))
    small_w = int(rng.integers(MINI_SIDE_MIN, MINI_SIDE_MAX + 1))

    max_spaced = ((small_h + 1) // 2) * ((small_w + 1) // 2)
    if max_spaced < 4:
        return None

    mandatory = [
        (0, 0),
        CARDINAL_DIRECTIONS[int(rng.integers(len(CARDINAL_DIRECTIONS)))],
        DIAGONAL_DIRECTIONS[int(rng.integers(len(DIAGONAL_DIRECTIONS)))],
    ]
    remaining = [delta for delta in ALL_DIRECTIONS if delta not in mandatory]
    extra_limit = min(4, len(remaining), max_spaced - len(mandatory))
    if extra_limit <= 0:
        return None
    extra_count = int(rng.integers(1, extra_limit + 1))
    extra_indices = rng.choice(len(remaining), size=extra_count, replace=False)
    directions = mandatory + [remaining[int(idx)] for idx in np.atleast_1d(extra_indices)]

    placed = _pick_positions(rng, small_h, small_w, directions)
    if placed is None:
        return None

    colors = rng.choice(COLOR_POOL, size=len(placed), replace=False)
    small_in = np.full((small_h, small_w), BACKGROUND, dtype=int)
    small_out = np.full((small_h, small_w), BACKGROUND, dtype=int)
    assignments: list[tuple[int, int, int, tuple[int, int]]] = []

    for color, (r, c, (dr, dc)) in zip(colors, placed):
        color = int(color)
        rr = int(r + dr)
        cc = int(c + dc)
        small_in[int(r), int(c)] = color
        small_out[rr, cc] = color
        assignments.append((int(r), int(c), color, (int(dr), int(dc))))

    assert not np.array_equal(small_in, small_out), "compression-only shortcut must not solve the puzzle"
    return small_in, small_out, assignments


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    built = _build_low_res_scene(rng)
    if built is None:
        return None
    _small_in, small_out, assignments = built
    inp = _expand_scene(small_out.shape, assignments)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, small_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compress-and-drift input")
