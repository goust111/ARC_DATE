from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

TILE_SIZE = 3
SEPARATOR_SIZE = 1
STRIDE = TILE_SIZE + SEPARATOR_SIZE

ROW_BAND_CHOICES = (3, 4)
COL_BAND_CHOICES = (3, 4)
MIN_TEAL_PER_TILE = 2
MAX_TEAL_PER_TILE = 7
GENERATION_ATTEMPTS = 600

_SEEN_INPUT_SIGNATURES: set[str] = set()
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _runs(indices: Iterable[int]) -> list[tuple[int, int]]:
    ordered = [int(v) for v in indices]
    assert ordered, "expected at least one occupied index"
    runs: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _tile_runs(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray, np.ndarray]:
    sep_rows = np.all(grid == GREY, axis=1)
    sep_cols = np.all(grid == GREY, axis=0)
    assert not bool(sep_rows[0]) and not bool(sep_rows[-1]), "separator rows must be internal"
    assert not bool(sep_cols[0]) and not bool(sep_cols[-1]), "separator cols must be internal"

    row_runs = _runs(np.flatnonzero(~sep_rows))
    col_runs = _runs(np.flatnonzero(~sep_cols))
    assert all(r1 - r0 + 1 == TILE_SIZE for r0, r1 in row_runs), "expected 3-row tile bands"
    assert all(c1 - c0 + 1 == TILE_SIZE for c0, c1 in col_runs), "expected 3-col tile bands"
    assert all(row_runs[i + 1][0] - row_runs[i][1] == 2 for i in range(len(row_runs) - 1)), "expected 1-row separators"
    assert all(col_runs[i + 1][0] - col_runs[i][1] == 2 for i in range(len(col_runs) - 1)), "expected 1-col separators"
    assert len(row_runs) >= 2 and len(col_runs) >= 2, "expected at least a 2x2 tile grid"
    return row_runs, col_runs, sep_rows, sep_cols


def _invert_tiles(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, GREY, TEAL}), "input must use only black, grey, teal"

    row_runs, col_runs, sep_rows, sep_cols = _tile_runs(g)
    out = g.copy()
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert set(map(int, np.unique(tile))).issubset({BLACK, TEAL}), "tiles may contain only black and teal"
            out[r0 : r1 + 1, c0 : c1 + 1] = np.where(tile == TEAL, BLACK, RED)
    return out, row_runs, col_runs, sep_rows, sep_cols


def _propagated_color(source_color: int, row: int) -> int:
    if source_color == BLACK:
        return BLACK
    assert source_color == RED, "top-row source must be black, red, or a grey separator"
    return BLUE if row % 2 == 1 else RED


def _oracle_transform(grid: np.ndarray) -> np.ndarray:
    inverted, _row_runs, _col_runs, sep_rows, _sep_cols = _invert_tiles(grid)
    source_row = inverted[0].copy()
    out = inverted.copy()
    height, width = map(int, inverted.shape)

    for r in range(1, height):
        if sep_rows[r]:
            continue
        for c in range(width):
            if out[r, c] != BLACK:
                continue
            source_c = c - r
            if not 0 <= source_c < width:
                continue
            source_color = int(source_row[source_c])
            if source_color == GREY:
                continue
            fill_color = _propagated_color(source_color, r)
            if fill_color != BLACK:
                out[r, c] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert every 3x3 teal tile, then use the inverted top row as a diagonal source constrained by the new black cells.

    Input structure:
    - The grid is partitioned into 3x3 tile regions by full grey separator rows and columns of width 1.
    - Every tile cell is either black or teal; separators are the only grey cells.
    - The first row belongs to the top tile band, so the inverted row 0 is the global source stripe.

    Rule:
    - Independently invert occupancy inside each 3x3 tile: teal becomes black and black becomes red; grey separators stay unchanged.
    - For each non-separator row `r >= 1`, each non-separator cell `(r, c)` reads its source from inverted row 0 at column `c - r`.
    - A red source alternates by row parity as it propagates diagonally downward (`row 1 -> blue`, `row 2 -> red`, then repeating); a black source stays black.
    - Propagation may write only onto cells that are black after inversion; red tile cells and grey separators are never overwritten.

    Key insight:
    - The same inversion does two jobs at once: it creates the top-row source pattern and it also carves the only cells that propagation is allowed to enter.
    """
    return _oracle_transform(grid)


def _mask_from_bits(bits: int) -> np.ndarray:
    flat = np.array([(bits >> shift) & 1 for shift in range(TILE_SIZE * TILE_SIZE)], dtype=int)
    return flat.reshape(TILE_SIZE, TILE_SIZE).astype(bool)


TOP_TILE_MASKS: list[np.ndarray] = []
GENERAL_TILE_MASKS: list[np.ndarray] = []
for _bits in range(1, 2 ** (TILE_SIZE * TILE_SIZE) - 1):
    _mask = _mask_from_bits(_bits)
    _count = int(_mask.sum())
    if not MIN_TEAL_PER_TILE <= _count <= MAX_TEAL_PER_TILE:
        continue
    GENERAL_TILE_MASKS.append(_mask)
    if 1 <= int(_mask[0].sum()) <= 2:
        TOP_TILE_MASKS.append(_mask)


def _sample_masks(rng: np.random.Generator, row_bands: int, col_bands: int) -> list[list[np.ndarray]] | None:
    total_tiles = row_bands * col_bands
    if total_tiles > len(GENERAL_TILE_MASKS):
        return None

    used: set[tuple[int, ...]] = set()
    masks: list[list[np.ndarray]] = []
    for band_r in range(row_bands):
        row_masks: list[np.ndarray] = []
        for _band_c in range(col_bands):
            pool = TOP_TILE_MASKS if band_r == 0 else GENERAL_TILE_MASKS
            candidates = [mask for mask in pool if tuple(mask.astype(int).ravel()) not in used]
            if not candidates:
                return None
            choice = candidates[int(rng.integers(len(candidates)))]
            used.add(tuple(choice.astype(int).ravel()))
            row_masks.append(choice.copy())
        masks.append(row_masks)

    top_patterns = {tuple(mask[0].astype(int)) for mask in masks[0]}
    if len(top_patterns) < 2:
        return None
    return masks


def _render_grid(row_bands: int, col_bands: int, masks: list[list[np.ndarray]]) -> np.ndarray:
    height = row_bands * TILE_SIZE + (row_bands - 1) * SEPARATOR_SIZE
    width = col_bands * TILE_SIZE + (col_bands - 1) * SEPARATOR_SIZE
    grid = np.zeros((height, width), dtype=int)

    for band_r in range(1, row_bands):
        grid[band_r * STRIDE - 1, :] = GREY
    for band_c in range(1, col_bands):
        grid[:, band_c * STRIDE - 1] = GREY

    for band_r in range(row_bands):
        for band_c in range(col_bands):
            r0 = band_r * STRIDE
            c0 = band_c * STRIDE
            grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = np.where(masks[band_r][band_c], TEAL, BLACK)
    return grid


def _semantic_counts(grid: np.ndarray) -> dict[str, int]:
    inverted, _row_runs, _col_runs, sep_rows, sep_cols = _invert_tiles(grid)
    out = _oracle_transform(grid)
    height, width = map(int, grid.shape)
    source_row = inverted[0]

    counts = {
        "source_red": 0,
        "source_black": 0,
        "blue_fill": 0,
        "red_fill": 0,
        "deep_fill": 0,
        "untouched_black": 0,
        "blocked_red": 0,
        "grey_source_skip": 0,
        "source_black_skip": 0,
        "oob_skip": 0,
    }
    counts["source_red"] = int(np.count_nonzero(source_row == RED))
    counts["source_black"] = int(np.count_nonzero(source_row == BLACK))

    for r in range(1, height):
        if sep_rows[r]:
            continue
        for c in range(width):
            if sep_cols[c]:
                continue
            if inverted[r, c] != BLACK:
                continue
            source_c = c - r
            if not 0 <= source_c < width:
                counts["oob_skip"] += 1
                if out[r, c] == BLACK:
                    counts["untouched_black"] += 1
                continue
            source_color = int(source_row[source_c])
            if source_color == GREY:
                counts["grey_source_skip"] += 1
                if out[r, c] == BLACK:
                    counts["untouched_black"] += 1
                continue
            if source_color == BLACK:
                counts["source_black_skip"] += 1
                if out[r, c] == BLACK:
                    counts["untouched_black"] += 1
                continue
            if out[r, c] == BLUE:
                counts["blue_fill"] += 1
                if r >= STRIDE:
                    counts["deep_fill"] += 1
            elif out[r, c] == RED:
                counts["red_fill"] += 1
                if r >= STRIDE:
                    counts["deep_fill"] += 1
            else:
                raise AssertionError("red source over a black cell must paint blue or red")

    red_source_coords = np.argwhere(inverted == RED)
    for r, c in red_source_coords:
        r = int(r)
        c = int(c)
        if r == 0 or sep_rows[r] or sep_cols[c]:
            continue
        source_c = c - r
        if 0 <= source_c < width and int(source_row[source_c]) == RED:
            counts["blocked_red"] += 1

    return counts


def _valid_case(grid: np.ndarray) -> bool:
    masks_seen: set[tuple[int, ...]] = set()
    row_runs, col_runs, sep_rows, sep_cols = _tile_runs(grid)
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            tile = grid[r0 : r1 + 1, c0 : c1 + 1]
            sig = tuple(tile.ravel())
            if sig in masks_seen:
                return False
            masks_seen.add(sig)

    counts = _semantic_counts(grid)
    if counts["source_red"] == 0 or counts["source_black"] == 0:
        return False
    if counts["blue_fill"] < 2:
        return False
    if counts["red_fill"] < 1:
        return False
    if counts["deep_fill"] < 2:
        return False
    if counts["blocked_red"] < 1:
        return False
    if counts["grey_source_skip"] < 1:
        return False
    if counts["untouched_black"] < 2:
        return False
    if counts["source_black_skip"] + counts["oob_skip"] < 1:
        return False

    out = _oracle_transform(grid)
    if not np.array_equal(out, main(grid)):
        return False
    if np.count_nonzero(out == BLUE) < 2:
        return False
    if np.count_nonzero(sep_rows) == 0 or np.count_nonzero(sep_cols) == 0:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_bands = int(rng.choice(ROW_BAND_CHOICES))
    col_bands = int(rng.choice(COL_BAND_CHOICES))
    masks = _sample_masks(rng, row_bands, col_bands)
    if masks is None:
        return None
    grid = _render_grid(row_bands, col_bands, masks)
    if not _valid_case(grid):
        return None

    input_sig = _signature(grid)
    output_sig = _signature(main(grid))
    if input_sig in _SEEN_INPUT_SIGNATURES or output_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_INPUT_SIGNATURES.add(input_sig)
    _SEEN_OUTPUT_SIGNATURES.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
