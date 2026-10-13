from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
USABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
TILE_SIZES = (3, 4)
PALETTE_LENGTHS = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 300

_GEN_BRANCHES = (
    (3, 3, "anti_only"),
    (4, 4, "rot180_only"),
    (3, 5, "plain"),
    (4, 3, "plain"),
    (3, 4, "rot180_only"),
    (4, 5, "anti_only"),
)
_GEN_COUNTER = 0


def _segments(n: int, dividers: list[int]) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    start = 0
    for divider in dividers:
        assert divider > start, "divider rows/cols must be isolated single lines"
        parts.append((start, divider))
        start = divider + 1
    assert start < n, "separator cannot be the final row/col"
    parts.append((start, n))
    return parts


def _foreground_color(tile: np.ndarray) -> int:
    colors = sorted(int(v) for v in set(tile.ravel()) - {BACKGROUND})
    assert len(colors) == 1, "each tile must use exactly one foreground color"
    assert colors[0] != SEPARATOR, "tile foreground color cannot be the separator"
    return colors[0]


def _mask_code(mask: np.ndarray) -> int:
    code = 0
    for bit in mask.astype(int).ravel():
        code = (code << 1) | int(bit)
    return code + 97 * int(mask.shape[0])


def _anti_diagonal_reflect(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(np.flipud(mask.T))


def _spans_tile(mask: np.ndarray) -> bool:
    return (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _tile_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    tile = np.full(mask.shape, BACKGROUND, dtype=int)
    tile[mask] = int(color)
    return tile


def _split_input_tiles(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width in TILE_SIZES, "tile width must be 3 or 4"

    divider_rows = [r for r in range(height) if np.all(g[r, :] == SEPARATOR)]
    row_segments = _segments(height, divider_rows) if divider_rows else [(0, height)]

    assert 4 <= len(row_segments) <= 6, "expected 4-6 stacked tiles"
    assert all(r1 - r0 == width for r0, r1 in row_segments), "tiles must be square"
    return [g[r0:r1, :].copy() for r0, r1 in row_segments], width


def _split_output_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "output must be square"
    side = int(g.shape[0])
    divider_rows = [r for r in range(side) if np.all(g[r, :] == SEPARATOR)]
    divider_cols = [c for c in range(side) if np.all(g[:, c] == SEPARATOR)]
    row_segments = _segments(side, divider_rows) if divider_rows else [(0, side)]
    col_segments = _segments(side, divider_cols) if divider_cols else [(0, side)]

    assert len(row_segments) == len(col_segments), "expected a square cell layout"
    assert 3 <= len(row_segments) <= 5, "expected a 3-5 by 3-5 output cell grid"
    cell_size = row_segments[0][1] - row_segments[0][0]
    assert all(r1 - r0 == cell_size for r0, r1 in row_segments), "row cell sizes must match"
    assert all(c1 - c0 == cell_size for c0, c1 in col_segments), "col cell sizes must match"
    assert cell_size in TILE_SIZES, "cell size must match the input tile size"
    return row_segments, col_segments, cell_size


def _build_output(template_mask: np.ndarray, palette: list[int]) -> np.ndarray:
    size = int(template_mask.shape[0])
    cells = len(palette)
    side = cells * size + (cells - 1)
    out = np.full((side, side), BACKGROUND, dtype=int)
    for idx in range(1, cells):
        sep = idx * size + (idx - 1)
        out[sep, :] = SEPARATOR
        out[:, sep] = SEPARATOR

    for r in range(cells):
        for c in range(cells):
            r0 = r * (size + 1)
            c0 = c * (size + 1)
            out[r0 : r0 + size, c0 : c0 + size][template_mask] = palette[max(r, c)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the lone non-transpose-symmetric tile, use it as a mask, and tile it with an order-derived diagonal palette.

    Input structure:
    - The grid is a single vertical stack of 4-6 square tiles of size 3x3 or 4x4.
    - Neighboring tiles are separated by full-width single gray rows, and every tile is drawn on black with one foreground color.
    - Exactly one tile is not symmetric under main-diagonal transpose; all other tiles are transpose-symmetric and have distinct colors.

    Rule:
    - Split the stack at the gray separator rows.
    - Find the unique tile whose full pixel matrix differs from its transpose, and keep only its binary foreground mask.
    - Read the foreground colors of the symmetric tiles from top to bottom to form palette c0..cP-1.
    - Build a P x P cell grid with gray separator bars, and stamp the mask into cell (r, c) using color c[max(r, c)] on a black background.

    Key insight:
    - Diagonal symmetry assigns roles instead of choosing the answer directly: the asymmetric tile gives shape, while the symmetric tiles contribute only an ordered color palette.
    """
    tiles, size = _split_input_tiles(grid)
    sym_flags = [np.array_equal(tile, tile.T) for tile in tiles]
    asym_indices = [idx for idx, is_sym in enumerate(sym_flags) if not is_sym]
    assert len(asym_indices) == 1, "expected exactly one asymmetric tile"

    asym_index = asym_indices[0]
    assert 0 < asym_index < len(tiles) - 1, "asymmetric tile must be interior"

    template_mask = tiles[asym_index] != BACKGROUND
    assert template_mask.shape == (size, size)
    assert bool(template_mask.any()), "template mask must be non-empty"

    palette = [_foreground_color(tile) for idx, tile in enumerate(tiles) if idx != asym_index]
    assert len(palette) == len(set(palette)), "symmetric tile colors must be distinct"
    return _build_output(template_mask, palette)


@lru_cache(maxsize=None)
def _symmetric_masks(size: int) -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    cells = size * size
    for bits in range(1, (1 << cells) - 1):
        mask = np.array([(bits >> i) & 1 for i in range(cells)], dtype=bool).reshape(size, size)
        area = int(mask.sum())
        if not (size <= area <= cells - 2):
            continue
        if not _spans_tile(mask):
            continue
        if not np.array_equal(mask, mask.T):
            continue
        masks.append(mask)
    masks.sort(key=_mask_code)
    assert len(masks) >= 6, "need enough symmetric masks"
    return tuple(masks)


@lru_cache(maxsize=None)
def _asymmetric_masks_by_kind(size: int) -> dict[str, tuple[np.ndarray, ...]]:
    groups = {"anti_only": [], "rot180_only": [], "plain": []}
    cells = size * size
    for bits in range(1, (1 << cells) - 1):
        mask = np.array([(bits >> i) & 1 for i in range(cells)], dtype=bool).reshape(size, size)
        area = int(mask.sum())
        if not (size <= area <= cells - size):
            continue
        if not _spans_tile(mask):
            continue
        if np.array_equal(mask, mask.T):
            continue
        if int((mask != mask.T).sum()) < 2:
            continue

        anti = np.array_equal(mask, _anti_diagonal_reflect(mask))
        rot180 = np.array_equal(mask, np.rot90(mask, 2))
        if anti:
            groups["anti_only"].append(mask)
        elif rot180:
            groups["rot180_only"].append(mask)
        else:
            groups["plain"].append(mask)

    for key in groups:
        groups[key].sort(key=_mask_code)
        assert groups[key], f"missing asymmetric masks for {key}"
        groups[key] = tuple(groups[key])
    return groups


def _choose_asymmetric_index(template_mask: np.ndarray, palette_len: int) -> int:
    return 1 + (_mask_code(template_mask) % (palette_len - 1))


def _choose_template_color(palette: list[int]) -> int:
    for color in USABLE_COLORS:
        if color not in palette:
            return color
    raise AssertionError("palette exhausted the usable colors")


def _choose_symmetric_masks(size: int, template_mask: np.ndarray, palette: list[int]) -> list[np.ndarray]:
    candidates = _symmetric_masks(size)
    seed = _mask_code(template_mask) + 17 * size + sum((idx + 1) * color for idx, color in enumerate(palette))
    chosen: list[np.ndarray] = []
    used: set[int] = set()
    for idx, color in enumerate(palette):
        probe = (seed + idx * 11 + color * 7) % len(candidates)
        while probe in used or np.array_equal(candidates[probe], template_mask):
            probe = (probe + 1) % len(candidates)
        used.add(probe)
        chosen.append(candidates[probe])
    return chosen


def _build_canonical_input(size: int, template_mask: np.ndarray, palette: list[int]) -> np.ndarray:
    palette_len = len(palette)
    tile_count = palette_len + 1
    asym_index = _choose_asymmetric_index(template_mask, palette_len)
    asym_color = _choose_template_color(palette)
    symmetric_masks = _choose_symmetric_masks(size, template_mask, palette)

    height = tile_count * size + (tile_count - 1)
    grid = np.full((height, size), BACKGROUND, dtype=int)
    for idx in range(1, tile_count):
        grid[idx * size + (idx - 1), :] = SEPARATOR

    sym_cursor = 0
    for tile_idx in range(tile_count):
        r0 = tile_idx * (size + 1)
        if tile_idx == asym_index:
            tile = _tile_from_mask(template_mask, asym_color)
        else:
            tile = _tile_from_mask(symmetric_masks[sym_cursor], palette[sym_cursor])
            sym_cursor += 1
        grid[r0 : r0 + size, :] = tile
    return grid


def _reconstruct_input_from_output(output_grid: np.ndarray) -> np.ndarray:
    rows, cols, size = _split_output_cells(output_grid)
    template_mask = output_grid[rows[0][0] : rows[0][1], cols[0][0] : cols[0][1]] != BACKGROUND
    palette: list[int] = []
    for idx in range(len(rows)):
        r0, r1 = rows[idx]
        c0, c1 = cols[idx]
        palette.append(_foreground_color(output_grid[r0:r1, c0:c1]))
    return _build_canonical_input(size, template_mask, palette)


def _validate_case(input_grid: np.ndarray) -> None:
    tiles, size = _split_input_tiles(input_grid)
    assert size in TILE_SIZES
    sym_flags = [np.array_equal(tile, tile.T) for tile in tiles]
    asym_indices = [idx for idx, flag in enumerate(sym_flags) if not flag]
    assert len(asym_indices) == 1
    assert 0 < asym_indices[0] < len(tiles) - 1

    template_mask = tiles[asym_indices[0]] != BACKGROUND
    palette = [_foreground_color(tile) for idx, tile in enumerate(tiles) if idx != asym_indices[0]]
    assert len(palette) in PALETTE_LENGTHS
    assert len(set(palette)) == len(palette)
    assert _choose_asymmetric_index(template_mask, len(palette)) == asym_indices[0]

    out = main(input_grid)
    assert np.array_equal(out, _build_output(template_mask, palette))
    assert np.array_equal(input_grid, _reconstruct_input_from_output(out))

    row_segments, col_segments, _ = _split_output_cells(out)
    assert len(row_segments) == len(palette)
    for r, (r0, r1) in enumerate(row_segments):
        for c, (c0, c1) in enumerate(col_segments):
            cell = out[r0:r1, c0:c1]
            assert np.array_equal(cell != BACKGROUND, template_mask)
            assert _foreground_color(cell) == palette[max(r, c)]


def _branch_for_call() -> tuple[int, int, str]:
    global _GEN_COUNTER
    branch = _GEN_BRANCHES[_GEN_COUNTER % len(_GEN_BRANCHES)]
    _GEN_COUNTER += 1
    return branch


def _generate_attempt() -> np.ndarray:
    size, palette_len, mask_kind = _branch_for_call()
    groups = _asymmetric_masks_by_kind(size)
    masks = groups.get(mask_kind) or groups["plain"]
    template_mask = random.choice(masks).copy()
    palette = random.sample(list(USABLE_COLORS), palette_len)
    input_grid = _build_canonical_input(size, template_mask, palette)
    _validate_case(input_grid)
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _generate_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
