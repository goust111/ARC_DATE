from __future__ import annotations

import numpy as np


BLOCK_SIDE = 3
GRID_SIDE = BLOCK_SIDE * BLOCK_SIDE
GEN_ATTEMPTS = 400


def _render_kronecker(outer: np.ndarray, inner: np.ndarray, bg: int, fg: int) -> np.ndarray:
    binary = np.kron(outer.astype(int), inner.astype(int)).astype(bool)
    out = np.full((GRID_SIDE, GRID_SIDE), bg, dtype=int)
    out[binary] = fg
    return out


def _decode_with_colors(
    grid: np.ndarray, bg: int, fg: int
) -> tuple[np.ndarray, np.ndarray] | None:
    layout = np.zeros((BLOCK_SIDE, BLOCK_SIDE), dtype=bool)
    tile: np.ndarray | None = None

    for br in range(BLOCK_SIDE):
        for bc in range(BLOCK_SIDE):
            block = grid[
                br * BLOCK_SIDE : (br + 1) * BLOCK_SIDE,
                bc * BLOCK_SIDE : (bc + 1) * BLOCK_SIDE,
            ]
            if np.all(block == bg):
                continue
            if not np.all((block == bg) | (block == fg)):
                return None

            mask = block == fg
            if not mask.any() or mask.all():
                return None
            if tile is None:
                tile = mask.copy()
            elif not np.array_equal(mask, tile):
                return None
            layout[br, bc] = True

    if tile is None:
        return None
    if not layout.any() or layout.all():
        return None
    if np.array_equal(layout, tile):
        return None
    if not np.array_equal(grid, _render_kronecker(layout, tile, bg, fg)):
        return None
    return layout, tile


def _all_decompositions(grid: np.ndarray) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    colors = [int(color) for color in np.unique(grid)]
    if len(colors) != 2:
        return []

    decompositions: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for bg in colors:
        fg = colors[0] if colors[1] == bg else colors[1]
        decoded = _decode_with_colors(grid, bg, fg)
        if decoded is not None:
            layout, tile = decoded
            decompositions.append((bg, fg, layout, tile))
    return decompositions


def main(grid: np.ndarray) -> np.ndarray:
    """Swap the roles of the recovered outer 3x3 layout and inner 3x3 tile in a 9x9 Kronecker product.

    Input structure:
    - The input is a 9x9 two-color grid, uniquely partitioned into a 3x3 array of 3x3 sub-blocks.
    - There exists a 3x3 outer layout mask L and a different 3x3 inner tile mask T, both nonempty and not full.
    - Each foreground cell of L corresponds to one sub-block equal to T in a single foreground color, while each background cell of L corresponds to a solid background sub-block.

    Rule:
    - Recover L from which 3x3 sub-blocks are occupied and recover T from the repeated occupied sub-block pattern.
    - Exchange their roles.
    - Build the output as the Kronecker product T ⊗ L using the same background and foreground colors, so each foreground cell of T places a full copy of L inside its 3x3 block slot.

    Key insight:
    - The hidden coarse 3x3 occupancy pattern is the outer factor and the visible repeated 3x3 sub-block is the inner factor; the task is to reverse those two levels, not to move or recolor pixels locally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIDE, GRID_SIDE), "expected a 9x9 grid"

    decompositions = _all_decompositions(g)
    assert len(decompositions) == 1, "input must admit exactly one valid factorization"
    bg, fg, layout, tile = decompositions[0]
    return _render_kronecker(tile, layout, bg, fg)


def _touches_every_side(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _component_count(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return 0

    remaining = {tuple(map(int, coord)) for coord in coords}
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    stack.append((nr, nc))
    return components


def _build_mask_library() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1, (1 << (BLOCK_SIDE * BLOCK_SIDE)) - 1):
        mask = np.array(
            [(bits >> idx) & 1 for idx in range(BLOCK_SIDE * BLOCK_SIDE)],
            dtype=bool,
        ).reshape(BLOCK_SIDE, BLOCK_SIDE)
        area = int(mask.sum())
        if not (3 <= area <= 6):
            continue
        if not _touches_every_side(mask):
            continue
        masks.append(mask)
    return masks


MASK_LIBRARY = _build_mask_library()
MASK_METADATA = [
    (mask, int(mask.sum()), _component_count(mask))
    for mask in MASK_LIBRARY
]


def _build_pair_pools() -> tuple[list[tuple[np.ndarray, np.ndarray]], ...]:
    pools: tuple[list[tuple[np.ndarray, np.ndarray]], ...] = ([], [], [], [])
    for layout, layout_area, layout_components in MASK_METADATA:
        for tile, tile_area, tile_components in MASK_METADATA:
            if np.array_equal(layout, tile):
                continue

            inp = _render_kronecker(layout, tile, 0, 1)
            out = _render_kronecker(tile, layout, 0, 1)
            if np.array_equal(inp, out):
                continue

            pair = (layout.copy(), tile.copy())
            if layout_area == tile_area:
                pools[0].append(pair)
            else:
                pools[1].append(pair)
            if (layout_components == 1) != (tile_components == 1):
                pools[2].append(pair)
            if layout_components == 1 and tile_components == 1:
                pools[3].append(pair)
    return pools


PAIR_POOLS = _build_pair_pools()
ALL_VALID_PAIRS = [pair for pool in PAIR_POOLS for pair in pool]


def _pick_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mode = int(rng.integers(4))
    pool = PAIR_POOLS[mode] if PAIR_POOLS[mode] else ALL_VALID_PAIRS
    choice = int(rng.integers(len(pool)))
    return pool[choice][0].copy(), pool[choice][1].copy()


def _semantic_checks(grid: np.ndarray, output: np.ndarray) -> bool:
    decompositions = _all_decompositions(grid)
    if len(decompositions) != 1:
        return False
    bg, fg, layout, tile = decompositions[0]

    expected = _render_kronecker(tile, layout, bg, fg)
    if not np.array_equal(output, expected):
        return False
    if np.array_equal(grid, output):
        return False

    swapped = _all_decompositions(output)
    if len(swapped) != 1:
        return False
    out_bg, out_fg, out_layout, out_tile = swapped[0]
    if out_bg != bg or out_fg != fg:
        return False
    if not np.array_equal(out_layout, tile):
        return False
    if not np.array_equal(out_tile, layout):
        return False

    full = np.ones((BLOCK_SIDE, BLOCK_SIDE), dtype=bool)
    shortcuts = (
        _render_kronecker(layout, layout, bg, fg),
        _render_kronecker(tile, tile, bg, fg),
        _render_kronecker(tile, full, bg, fg),
        _render_kronecker(full, layout, bg, fg),
    )
    return all(not np.array_equal(output, shortcut) for shortcut in shortcuts)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    palette = np.arange(10, dtype=int)

    for _ in range(GEN_ATTEMPTS):
        layout, tile = _pick_pair(rng)
        bg, fg = rng.choice(palette, size=2, replace=False)
        grid = _render_kronecker(layout, tile, int(bg), int(fg))

        try:
            output = main(grid)
        except AssertionError:
            continue
        if _semantic_checks(grid, output):
            return grid

    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
