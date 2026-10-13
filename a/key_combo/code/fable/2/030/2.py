import random

import numpy as np


BLOCK = 3
SIDE = 3
GRID_SIZE = BLOCK * SIDE
BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
MASK_COUNT_OPTIONS = ((2, 2), (2, 3), (3, 2))
MASK_COUNT_WEIGHTS = (0.6, 0.2, 0.2)
NOISE_STYLE_WEIGHTS = {
    "near_black": 0.35,
    "near_pure": 0.35,
    "balanced": 0.30,
}
GEN_ATTEMPTS = 500
_SEEN_INPUT_SIGS: set[bytes] = set()
_SEEN_OUTPUT_SIGS: set[bytes] = set()


def _block_view(grid: np.ndarray, br: int, bc: int) -> np.ndarray:
    return grid[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]


def _mask_from_cells(cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=int)
    for r, c in cells:
        mask[r, c] = 1
    return mask


def _decode_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 9x9 grid"

    outer = np.zeros((SIDE, SIDE), dtype=int)
    inner = np.zeros((SIDE, SIDE), dtype=int)
    pure_color = None

    for br in range(SIDE):
        for bc in range(SIDE):
            tile = _block_view(g, br, bc)
            vals = np.unique(tile)
            if vals.size == 1 and int(vals[0]) == BLACK:
                outer[br, bc] = 1
            elif vals.size == 1:
                color = int(vals[0])
                if pure_color is None:
                    pure_color = color
                else:
                    assert pure_color == color, "all solid non-black tiles must share one color"
                inner[br, bc] = 1
            else:
                assert vals.size >= 2, "noise tiles must be multicolor"

    assert pure_color is not None, "expected at least one solid non-black tile"
    assert outer.any(), "expected at least one solid black tile"
    assert inner.any(), "expected at least one solid non-black tile"
    assert not np.any((outer == 1) & (inner == 1))
    return outer, inner, pure_color


def _render_output(outer: np.ndarray, inner: np.ndarray, color: int) -> np.ndarray:
    stamp = np.where(inner == 1, color, BLACK).astype(int)
    return np.kron(outer.astype(int), stamp).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the black outer mask and the solid-color inner mask, then stamp their Kronecker product.

    Input structure:
    - The input is a 9x9 grid partitioned into a 3x3 array of aligned 3x3 tiles.
    - Some tiles are entirely black, some tiles are entirely filled with one shared non-black color,
      and the remaining tiles are multicolor noise patches.

    Rule:
    - Build the outer 3x3 mask A by marking each fully black tile with 1.
    - Build the inner 3x3 mask B by marking each fully uniform non-black tile with 1.
    - Let the shared color of those uniform non-black tiles be the paint color.
    - Output a 9x9 grid equal to the Kronecker product of A with B: every 1-cell of A becomes a
      3x3 copy of B painted in the shared color, and every 0-cell of A becomes a black 3x3 tile.

    Key insight:
    - The two shapes are encoded indirectly: black void tiles define the outer placement mask,
      while oversaturated solid-color tiles define the inner stamp and its color.
    """
    outer, inner, pure_color = _decode_input(grid)
    return _render_output(outer, inner, pure_color)


def _grid_signature(grid: np.ndarray) -> bytes:
    arr = np.asarray(grid, dtype=np.uint8)
    return arr.shape[0].to_bytes(1, "big") + arr.shape[1].to_bytes(1, "big") + arr.tobytes()


def _sample_disjoint_masks() -> tuple[np.ndarray, np.ndarray]:
    cells = [(r, c) for r in range(SIDE) for c in range(SIDE)]
    for _ in range(300):
        outer_count, inner_count = random.choices(MASK_COUNT_OPTIONS, weights=MASK_COUNT_WEIGHTS, k=1)[0]
        outer_cells = random.sample(cells, outer_count)
        remaining = [cell for cell in cells if cell not in outer_cells]
        inner_cells = random.sample(remaining, inner_count)

        outer = _mask_from_cells(outer_cells)
        inner = _mask_from_cells(inner_cells)
        kron_ab = np.kron(outer, inner)
        kron_ba = np.kron(inner, outer)

        if np.array_equal(kron_ab, kron_ba):
            continue
        if np.array_equal(outer, inner):
            continue
        if outer.sum() == 0 or inner.sum() == 0:
            continue
        if outer.sum() == 9 or inner.sum() == 9:
            continue
        return outer, inner
    raise RuntimeError("failed to sample non-degenerate masks")


def _make_noise_tile(pure_color: int, style: str) -> np.ndarray:
    extras = [c for c in NON_BLACK_COLORS if c != pure_color]
    accent1, accent2 = random.sample(extras, 2)

    while True:
        tile = np.zeros((BLOCK, BLOCK), dtype=int)
        if style == "near_black":
            tile.fill(BLACK)
            changed = random.sample(range(BLOCK * BLOCK), random.choice((2, 2, 3, 4)))
            pool = [pure_color, accent1, accent2]
            for idx in changed:
                tile[idx // BLOCK, idx % BLOCK] = random.choice(pool)
        elif style == "near_pure":
            tile.fill(pure_color)
            changed = random.sample(range(BLOCK * BLOCK), random.choice((1, 2, 2, 3)))
            pool = [BLACK, accent1, accent2]
            for idx in changed:
                tile[idx // BLOCK, idx % BLOCK] = random.choice(pool)
        else:
            palette = random.sample([BLACK, pure_color, accent1, accent2], random.choice((2, 3, 4)))
            tile = np.array(random.choices(palette, k=BLOCK * BLOCK), dtype=int).reshape(BLOCK, BLOCK)

        vals = np.unique(tile)
        if vals.size < 2:
            continue
        return tile


def _render_input(outer: np.ndarray, inner: np.ndarray, pure_color: int) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    noise_cells = []
    for br in range(SIDE):
        for bc in range(SIDE):
            if outer[br, bc] == 1:
                _block_view(grid, br, bc)[:, :] = BLACK
            elif inner[br, bc] == 1:
                _block_view(grid, br, bc)[:, :] = pure_color
            else:
                noise_cells.append((br, bc))

    styles = ["near_black", "near_pure"]
    while len(styles) < len(noise_cells):
        styles.append(random.choices(tuple(NOISE_STYLE_WEIGHTS), weights=tuple(NOISE_STYLE_WEIGHTS.values()), k=1)[0])
    random.shuffle(styles)

    for (br, bc), style in zip(noise_cells, styles):
        _block_view(grid, br, bc)[:, :] = _make_noise_tile(pure_color, style)

    return grid


def _validate_generated_sample(grid: np.ndarray, outer: np.ndarray, inner: np.ndarray, pure_color: int) -> None:
    decoded_outer, decoded_inner, decoded_color = _decode_input(grid)
    assert np.array_equal(decoded_outer, outer)
    assert np.array_equal(decoded_inner, inner)
    assert decoded_color == pure_color

    out = main(grid)
    expected = _render_output(outer, inner, pure_color)
    assert np.array_equal(out, expected)
    assert not np.array_equal(expected, _render_output(inner, outer, pure_color))

    noise_count = 0
    for br in range(SIDE):
        for bc in range(SIDE):
            tile = _block_view(grid, br, bc)
            vals = np.unique(tile)
            if outer[br, bc] == 1:
                assert vals.size == 1 and int(vals[0]) == BLACK
            elif inner[br, bc] == 1:
                assert vals.size == 1 and int(vals[0]) == pure_color
            else:
                noise_count += 1
                assert vals.size >= 2
                assert not np.all(tile == BLACK)
                assert not np.all(tile == pure_color)
    assert noise_count >= 4


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pure_color = random.choice(NON_BLACK_COLORS)
        outer, inner = _sample_disjoint_masks()
        grid = _render_input(outer, inner, pure_color)
        _validate_generated_sample(grid, outer, inner, pure_color)

        input_sig = _grid_signature(grid)
        output_sig = _grid_signature(main(grid))
        if input_sig in _SEEN_INPUT_SIGS or output_sig in _SEEN_OUTPUT_SIGS:
            continue

        _SEEN_INPUT_SIGS.add(input_sig)
        _SEEN_OUTPUT_SIGS.add(output_sig)
        return grid
    raise RuntimeError("failed to generate a unique valid sample")
