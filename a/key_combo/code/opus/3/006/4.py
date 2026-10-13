from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    _ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if _ARC_UTILS_DIR.exists():
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import Color, paste
except Exception:
    class Color:
        BLACK = 0
        MAROON = 9

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        mask: np.ndarray | None = None,
        overwrite: bool = True,
        clip: bool = False,
    ) -> np.ndarray:
        if transparent is not None or mask is not None or not overwrite or clip:
            raise ValueError("fallback paste only supports opaque in-bounds placement")
        h, w = patch.shape
        dst[r0 : r0 + h, c0 : c0 + w] = patch
        return dst


SEPARATOR = Color.MAROON
NON_SEPARATOR_COLORS = [c for c in range(10) if c != SEPARATOR]
LATTICE_CHOICES = (4, 5)
CELL_SIZE_CHOICES = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands = []
    start = 0
    for cut in separators:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _separator_bands(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_colors = {int(g[r, 0]) for r in range(h) if np.all(g[r, :] == g[r, 0])}
    col_colors = {int(g[0, c]) for c in range(w) if np.all(g[:, c] == g[0, c])}
    sep_colors = row_colors & col_colors
    assert len(sep_colors) == 1, "expected exactly one separator color"
    sep = next(iter(sep_colors))
    assert row_colors == {sep}, "all full rows must be separator rows"
    assert col_colors == {sep}, "all full columns must be separator columns"

    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    assert sep_rows and sep_cols, "separator lattice must have both row and column bars"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert row_bands and col_bands, "separator lattice must leave at least one cell"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(row_heights) == 1 and len(col_widths) == 1, "lattice cells must be equal-sized"
    return int(sep), row_bands, col_bands


def _extract_solid_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep, row_bands, col_bands = _separator_bands(g)

    solid_cells: dict[tuple[int, int], int] = {}
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(cell == cell[0, 0]):
                color = int(cell[0, 0])
                assert color != sep, "solid signal cells must not use the separator color"
                solid_cells[(ri, ci)] = color

    assert solid_cells, "expected at least one solid lattice cell"
    solid_rows = [ri for ri, _ in solid_cells]
    solid_cols = [ci for _, ci in solid_cells]
    r0, r1 = min(solid_rows), max(solid_rows)
    c0, c1 = min(solid_cols), max(solid_cols)
    assert len(solid_cells) == (r1 - r0 + 1) * (c1 - c0 + 1), "solid cells must form one rectangle"

    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for ri in range(r0, r1 + 1):
        for ci in range(c0, c1 + 1):
            out[ri - r0, ci - c0] = solid_cells[(ri, ci)]
    return out


def _active_header(header: np.ndarray) -> np.ndarray:
    hdr = np.asarray(header, dtype=int)
    assert hdr.ndim == 1 and hdr.size > 0, "header must be a nonempty row"
    non_black = np.flatnonzero(hdr != Color.BLACK)
    assert non_black.size > 0, "header must contain at least one non-black color"
    active = hdr[: int(non_black[-1]) + 1]
    assert active.size > 0, "header must contain a nonempty active prefix"
    return active


def _diag_tile_from_header(header: np.ndarray) -> np.ndarray:
    active = _active_header(header)
    doubled = np.resize(active, active.size * 2)
    index = np.maximum.outer(np.arange(doubled.size), np.arange(doubled.size))
    return doubled[index]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract a solid-cell table from a separator lattice and expand its first row by diagonal indexing.

    Input structure:
    - The grid contains full-row and full-column separator bars of one color that split it into equal rectangular cells.
    - Exactly one contiguous rectangular block of cells is solid monochrome; the remaining cells are multicolor textured noise.
    - Solid cells may use black, but never the separator color.

    Rule:
    - Split the lattice into cells using the separator bars.
    - Keep only the solid monochrome cells, preserving lattice order, and crop them to their occupied rectangle to form a reduced grid.
    - Read the reduced grid's first row as a header, remove only its trailing black cells, repeat the remaining sequence to double its length, and fill a doubled square where each cell takes the header color at index max(row, column).

    Key insight:
    - The useful information is hidden twice: first in which lattice cells are fully solid, then in the trimmed first-row sequence of the extracted mini-grid.
    """
    extracted = _extract_solid_grid(np.asarray(grid, dtype=int))
    return _diag_tile_from_header(extracted[0])


def _pattern_cell(cell_size: int, palette: list[int], tag: int) -> np.ndarray:
    rr, cc = np.indices((cell_size, cell_size))
    k = random.randint(2, min(4, len(palette)))
    colors = np.array(random.sample(palette, k), dtype=int)
    phase = (tag + random.randrange(k)) % k
    mode = (tag + random.randrange(5)) % 5

    if mode == 0:
        idx = rr + cc + phase
    elif mode == 1:
        idx = cc + phase
    elif mode == 2:
        idx = rr + phase
    elif mode == 3:
        idx = rr + 2 * cc + phase
    else:
        idx = 2 * rr + cc + phase

    cell = colors[idx % k]
    if random.choice((False, True)):
        cell = cell.T
    if random.choice((False, True)):
        cell = np.flipud(cell)
    if random.choice((False, True)):
        cell = np.fliplr(cell)
    assert len(np.unique(cell)) >= 2, "noise cells must be non-solid"
    return np.asarray(cell, dtype=int)


def _header_sequence(active_len: int, trailing_zeros: int, require_internal_black: bool) -> list[int]:
    active = [random.choice([c for c in NON_SEPARATOR_COLORS if c != Color.BLACK]) for _ in range(active_len)]

    if require_internal_black:
        zero_pos = random.randrange(active_len - 1)
        active[zero_pos] = Color.BLACK
        if len(set(active)) == 1:
            alt = [c for c in NON_SEPARATOR_COLORS if c not in {SEPARATOR, active[zero_pos], Color.BLACK}]
            active[-1] = random.choice(alt)
    else:
        active[-1] = random.choice([c for c in NON_SEPARATOR_COLORS if c not in {SEPARATOR, Color.BLACK}])

    if len(set(active)) < 2:
        change_at = 0 if active_len == 1 else random.randrange(active_len - 1 if require_internal_black else active_len)
        choices = [c for c in NON_SEPARATOR_COLORS if c != active[change_at] and c != SEPARATOR]
        if require_internal_black and change_at == active_len - 1:
            choices = [c for c in choices if c != Color.BLACK]
        active[change_at] = random.choice(choices)

    if require_internal_black:
        assert Color.BLACK in active[:-1], "internal black must appear before the tail"
    assert active[-1] != Color.BLACK, "active header must end with a non-black color"
    return active + [Color.BLACK] * trailing_zeros


def _sample_mode() -> dict[str, object]:
    global _GEN_CALL_COUNT
    mode_id = _GEN_CALL_COUNT % 4
    lattice_hint = LATTICE_CHOICES[_GEN_CALL_COUNT % len(LATTICE_CHOICES)]
    _GEN_CALL_COUNT += 1
    return {
        "lattice_hint": lattice_hint,
        "require_trailing_zeros": mode_id in {0, 2},
        "require_internal_black": mode_id in {0, 3},
    }


def _sample_geometry(mode: dict[str, object]) -> tuple[int, int, int, int, int, int, int]:
    lattice_options = [int(mode["lattice_hint"]), random.choice(LATTICE_CHOICES)]
    random.shuffle(lattice_options)
    lattice = lattice_options[0]
    cell_size = random.choice(CELL_SIZE_CHOICES)

    max_block_h = min(lattice, 4)
    block_h = random.randint(2, max_block_h)

    min_block_w = 4 if mode["require_internal_black"] else 3
    block_w = random.randint(min_block_w, lattice)

    if mode["require_trailing_zeros"]:
        min_active = 3 if mode["require_internal_black"] else 2
        active_len = random.randint(min_active, block_w - 1)
    else:
        active_len = block_w

    block_r0 = random.randint(0, lattice - block_h)
    block_c0 = random.randint(0, lattice - block_w)
    return lattice, cell_size, block_h, block_w, active_len, block_r0, block_c0


def _build_block(block_h: int, block_w: int, active_len: int, require_internal_black: bool) -> np.ndarray:
    trailing_zeros = block_w - active_len
    block = np.empty((block_h, block_w), dtype=int)
    block[0] = np.array(
        _header_sequence(active_len, trailing_zeros, require_internal_black=bool(require_internal_black)),
        dtype=int,
    )

    for r in range(1, block_h):
        row = np.array(random.choices(NON_SEPARATOR_COLORS, k=block_w), dtype=int)
        if np.array_equal(row, block[0]):
            diff_at = random.randrange(block_w)
            choices = [c for c in NON_SEPARATOR_COLORS if c != int(row[diff_at])]
            row[diff_at] = random.choice(choices)
        block[r] = row

    assert np.any(block[1:] != block[0]), "lower rows must add information beyond the header"
    return block


def _render_grid(lattice: int, cell_size: int, block_r0: int, block_c0: int, block: np.ndarray) -> np.ndarray:
    size = lattice * cell_size + (lattice - 1)
    stride = cell_size + 1
    grid = np.full((size, size), SEPARATOR, dtype=int)

    for ri in range(lattice):
        for ci in range(lattice):
            r0 = ri * stride
            c0 = ci * stride
            if block_r0 <= ri < block_r0 + block.shape[0] and block_c0 <= ci < block_c0 + block.shape[1]:
                color = int(block[ri - block_r0, ci - block_c0])
                cell = np.full((cell_size, cell_size), color, dtype=int)
            else:
                cell = _pattern_cell(cell_size, NON_SEPARATOR_COLORS, ri * lattice + ci)
            paste(grid, cell, r0, c0)
    return grid


def _semantic_checks(
    grid: np.ndarray,
    *,
    block: np.ndarray,
    active_len: int,
    require_trailing_zeros: bool,
    require_internal_black: bool,
) -> bool:
    try:
        extracted = _extract_solid_grid(grid)
    except AssertionError:
        return False

    if not np.array_equal(extracted, block):
        return False

    header = extracted[0]
    try:
        active = _active_header(header)
    except AssertionError:
        return False

    if active.size != active_len:
        return False
    if require_trailing_zeros and (active_len >= header.size or not np.all(header[active_len:] == Color.BLACK)):
        return False
    if not require_trailing_zeros and header[-1] == Color.BLACK:
        return False
    if require_internal_black and Color.BLACK not in header[:active_len - 1]:
        return False
    if len(np.unique(active)) < 2:
        return False
    if extracted.shape[0] < 2 or not np.any(extracted[1:] != extracted[0]):
        return False

    out = _diag_tile_from_header(header)
    if out.shape != (2 * active_len, 2 * active_len):
        return False
    if len(np.unique(out)) < 2:
        return False
    if out.shape == extracted.shape and np.array_equal(out, extracted):
        return False

    oracle = main(grid)
    return np.array_equal(oracle, out)


def _gen_attempt(mode: dict[str, object]) -> np.ndarray | None:
    lattice, cell_size, block_h, block_w, active_len, block_r0, block_c0 = _sample_geometry(mode)
    block = _build_block(
        block_h,
        block_w,
        active_len,
        require_internal_black=bool(mode["require_internal_black"]),
    )
    grid = _render_grid(lattice, cell_size, block_r0, block_c0, block)
    if max(grid.shape) > 30:
        return None
    if not _semantic_checks(
        grid,
        block=block,
        active_len=active_len,
        require_trailing_zeros=bool(mode["require_trailing_zeros"]),
        require_internal_black=bool(mode["require_internal_black"]),
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    mode = _sample_mode()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-lattice diagonal-tiling input")
