from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_PALETTE = np.arange(10, dtype=int)
_MAX_GRID_SIDE = 30
_MAX_GENERATION_ATTEMPTS = 500
_CORE_SIZE_CHOICES = np.array([2, 3, 4, 5, 6], dtype=int)


def _axis_run_starts_and_lengths(grid: np.ndarray, axis: int) -> tuple[list[int], list[int]]:
    lines = grid if axis == 0 else grid.T
    starts = [0]
    lengths: list[int] = []
    for i in range(1, lines.shape[0]):
        if not np.array_equal(lines[i], lines[i - 1]):
            lengths.append(i - starts[-1])
            starts.append(i)
    lengths.append(lines.shape[0] - starts[-1])
    return starts, lengths


def _expand_core(core: np.ndarray, row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    expanded_rows = np.repeat(core, np.asarray(row_heights, dtype=int), axis=0)
    return np.repeat(expanded_rows, np.asarray(col_widths, dtype=int), axis=1)


def _decompose(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    row_starts, row_heights = _axis_run_starts_and_lengths(grid, axis=0)
    col_starts, col_widths = _axis_run_starts_and_lengths(grid, axis=1)
    assert len(row_heights) == len(col_widths), "deduplicated core must be square"
    core = grid[np.asarray(row_starts, dtype=int)][:, np.asarray(col_starts, dtype=int)]
    assert np.array_equal(_expand_core(core, row_heights, col_widths), grid), (
        "input must be obtainable by independently repeating core rows and columns"
    )
    return core, row_heights, col_widths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the row-band and column-band repeat vectors while keeping the core pattern unchanged.

    Input structure:
    - The input is a grid produced from a square core pattern by independently
      repeating each core row and each core column.
    - Adjacent identical full rows form row bands; adjacent identical full
      columns form column bands.
    - Removing consecutive duplicate rows and columns yields the square core.

    Rule:
    - Record the row-band heights H from top to bottom and the column-band
      widths W from left to right.
    - Recover the deduplicated square core pattern without rotating,
      transposing, mirroring, or recoloring it.
    - Re-expand that same core using W as the new row-band heights and H as the
      new column-band widths.

    Key insight:
    - The visible picture stays in the same orientation; only the hidden
      run-length vectors attached to rows and columns are exchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    core, row_heights, col_widths = _decompose(g)
    return _expand_core(core, col_widths, row_heights)


def _adjacent_lines_distinct(core: np.ndarray) -> bool:
    for r in range(1, core.shape[0]):
        if np.array_equal(core[r], core[r - 1]):
            return False
    for c in range(1, core.shape[1]):
        if np.array_equal(core[:, c], core[:, c - 1]):
            return False
    return True


def _core_is_instructive(core: np.ndarray) -> bool:
    if len(np.unique(core)) < 2:
        return False
    if not _adjacent_lines_distinct(core):
        return False
    if all(np.unique(row).size == 1 for row in core):
        return False
    if all(np.unique(col).size == 1 for col in core.T):
        return False
    return True


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = np.sort(_RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    values = np.diff(np.concatenate(([0], cuts, [total])))
    return [int(v) for v in values.tolist()]


def _sample_run_pair(n: int) -> tuple[list[int], list[int]]:
    max_total = min(_MAX_GRID_SIDE, max(n + 3, 5 * n))
    for _ in range(200):
        mode = int(_RNG.integers(0, 3))
        if mode == 0:
            row_total = int(_RNG.integers(n + 2, max_total + 1))
            col_total = int(_RNG.integers(n + 2, max_total + 1))
            if row_total == col_total:
                continue
        elif mode == 1:
            row_total = col_total = int(_RNG.integers(n + 2, max_total + 1))
        else:
            small_total = int(_RNG.integers(n + 1, min(max_total, 2 * n + 2) + 1))
            large_low = max(n + 3, small_total + 2)
            if large_low > max_total:
                continue
            large_total = int(_RNG.integers(large_low, max_total + 1))
            if bool(_RNG.integers(0, 2)):
                row_total, col_total = large_total, small_total
            else:
                row_total, col_total = small_total, large_total

        row_heights = _random_composition(row_total, n)
        col_widths = _random_composition(col_total, n)
        if row_heights == col_widths:
            continue
        if max(row_heights) == 1 or max(col_widths) == 1:
            continue
        if len(set(row_heights)) == 1 and len(set(col_widths)) == 1:
            continue
        return row_heights, col_widths
    raise RuntimeError("failed to sample distinct run vectors")


def _make_affine_core(n: int, palette: np.ndarray) -> np.ndarray:
    rr, cc = np.indices((n, n))
    k = len(palette)
    row_step = int(_RNG.integers(1, k))
    col_step = int(_RNG.integers(1, k))
    offset = int(_RNG.integers(0, k))
    return palette[(row_step * rr + col_step * cc + offset) % k]


def _make_rowcol_core(n: int, palette: np.ndarray) -> np.ndarray:
    k = len(palette)
    row_codes = _RNG.integers(0, k, size=n)
    col_codes = _RNG.integers(0, k, size=n)
    row_scale = int(_RNG.integers(1, k))
    col_scale = int(_RNG.integers(1, k))
    offset = int(_RNG.integers(0, k))
    return palette[(row_scale * row_codes[:, None] + col_scale * col_codes[None, :] + offset) % k]


def _make_random_core(n: int, palette: np.ndarray) -> np.ndarray:
    return _RNG.choice(palette, size=(n, n), replace=True)


def _sample_core(n: int) -> np.ndarray:
    for _ in range(300):
        palette_size = int(_RNG.integers(2, min(6, len(_PALETTE)) + 1))
        palette = _RNG.choice(_PALETTE, size=palette_size, replace=False)
        family = int(_RNG.integers(0, 3))
        if family == 0:
            core = _make_affine_core(n, palette)
        elif family == 1:
            core = _make_rowcol_core(n, palette)
        else:
            core = _make_random_core(n, palette)
        core = np.asarray(core, dtype=int)
        if _core_is_instructive(core):
            return core
    raise RuntimeError("failed to sample a valid core pattern")


def _instance_is_valid(
    inp: np.ndarray,
    out: np.ndarray,
    core: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > _MAX_GRID_SIDE or max(out.shape) > _MAX_GRID_SIDE:
        return False
    if np.array_equal(inp, out):
        return False
    if np.array_equal(out, inp.T):
        return False

    recovered_core, recovered_h, recovered_w = _decompose(inp)
    if recovered_h != row_heights or recovered_w != col_widths:
        return False
    if not np.array_equal(recovered_core, core):
        return False
    if not np.array_equal(_expand_core(core, row_heights, col_widths), inp):
        return False
    if not np.array_equal(_expand_core(core, col_widths, row_heights), out):
        return False
    if not np.array_equal(main(inp), out):
        return False
    if not np.array_equal(main(out), inp):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = int(_RNG.choice(_CORE_SIZE_CHOICES))
    core = _sample_core(n)
    row_heights, col_widths = _sample_run_pair(n)
    inp = _expand_core(core, row_heights, col_widths)
    out = _expand_core(core, col_widths, row_heights)

    if _instance_is_valid(inp, out, core, row_heights, col_widths):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 036/2")
