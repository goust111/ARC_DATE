# Combo variant of b99e7126 (index 48)
# combo_spec: D3/1+D4/1
# Name: 铺满整个网格 + 反向分形
#
# Design analysis for combo D3/1 + D4/1
# 1. Concrete behavior trace
#    - Original task: a few minority cells reveal a 3x3 mask; the solver enlarges that minority
#      cell pattern to a 3x3 block of cells and completes the missing minority cells there.
#    - Fused task here: there is exactly one abnormal cell. Its 3x3 interior is not the source to
#      enlarge; it is a compressed, damaged copy of the normal cell pattern. The output repairs the
#      grid by repainting every cell with the repeated majority pattern, so the odd cell disappears.
#    - Example at the cell level:
#      majority mask      compressed anomaly
#      1 1 1              0 1 0
#      0 1 0      ->      0 1 0
#      1 1 1              0 1 0
#      The original would treat the odd cell as a blueprint for a larger placement. The fused
#      version instead reads it as a shrunken defect and restores the full-width majority motif.
# 2. Complexity calibration
#    - A pure "replace the odd cell with the common cell" rule would be too weak and visually bland.
#      To keep the fusion visible, the abnormal cell is always derived from the majority pattern by
#      removing the four corners and optionally restoring the center if that would otherwise become
#      too sparse. This gives a stable reverse-fractal family instead of arbitrary noise.
#    - The palette is limited to three colors: separator, cell background, and cell foreground.
#      That keeps the rule readable from a few examples and avoids hidden color-order logic.
# 3. Hidden assumptions broken from the source solver
#    - The source solver assumed the minority cell encodes a mask whose active color can be inferred
#      and whose placement among cells must be solved. None of that remains true after D3/1+D4/1.
#    - The new solver instead assumes exactly two cell patterns exist and the majority occupies
#      48 of the 49 cells. The generator enforces this explicitly.
# 4. Placement legality
#    - The grid geometry is unchanged: 7x7 cells, each with a 3x3 interior separated by 1-pixel
#      divider lines. The single anomaly is placed at one valid cell coordinate, so no overlap or
#      boundary issues can arise.
# 5. Visual observability and cleanliness
#    - Input/output differences are confined to one 3x3 cell interior, but that cell is a clear
#      compressed version of the majority motif rather than an unrelated pattern. The output is
#      visually cleaner: a fully uniform repeated tiling.
#    - Because the anomaly is structurally tied to the majority shape, a human can infer "repair the
#      squeezed odd tile by restoring the normal tile everywhere" from a few examples.

from functools import reduce
from math import gcd
from itertools import product

import numpy as np
import random

from arc_utils import paste


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
COLOR_POOL = 10
MAX_RETRIES = 200


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    """Infer the regular divider-lined cell grid."""
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1) if row_divs[i + 1] > row_divs[i]]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1) if col_divs[i + 1] > col_divs[i]]
    step = reduce(gcd, drs + dcs)
    assert step > 1
    assert row_divs[0] == 0 and col_divs[0] == 0
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1
    assert (h - 1) % step == 0 and (w - 1) % step == 0
    return int(step), row_divs, col_divs


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _extract_cells(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Return unique cell arrays and the per-cell pattern keys."""
    n = (g.shape[0] - 1) // step
    inner = step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            key = tuple(arr.ravel().tolist())
            patterns.setdefault(key, arr)
            row.append(key)
        cell_pat.append(row)
    return patterns, cell_pat


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a single compressed anomaly by re-tiling the whole cell grid with the normal cell.

    Input structure:
    - The picture is a 7x7 grid of 3x3 cell interiors separated by 1-pixel divider lines.
    - All but one cell share the same interior pattern; the lone outlier is a shrunken-looking
      version of that normal pattern.

    Rule:
    - Identify the repeated majority cell pattern.
    - Ignore the abnormal cell as a source for enlargement; instead use the majority pattern as the
      reconstruction template.
    - Repaint every cell interior in the grid with that majority pattern, preserving divider lines.

    Key insight:
    - The odd cell is not a blueprint to expand. It is a damaged miniature of the surrounding motif,
      so the correct transformation is global repair: cover the entire cell grid with the normal tile.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    patterns, cell_pat = _extract_cells(g, step)

    counts: dict[tuple[int, ...], int] = {}
    for i, j in product(range(n), range(n)):
        key = cell_pat[i][j]
        counts[key] = counts.get(key, 0) + 1

    assert len(counts) == 2, "expected one majority pattern and one unique anomaly pattern"
    majority_pat, majority_count = max(counts.items(), key=lambda kv: kv[1])
    assert majority_count == n * n - 1, "fused variant expects exactly one anomalous cell"
    majority_arr = patterns[majority_pat]

    out = g.copy()
    for i, j in product(range(n), range(n)):
        r0 = i * step + 1
        c0 = j * step + 1
        out[r0 : r0 + inner, c0 : c0 + inner] = majority_arr
    return out


def _bool_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _compress_majority_mask(mask: np.ndarray) -> np.ndarray:
    """
    Reverse-fractal shrink used by the generator:
    remove the four corners, then restore the center if the result would be too sparse.
    """
    shrunk = mask.copy()
    shrunk[0, 0] = False
    shrunk[0, 2] = False
    shrunk[2, 0] = False
    shrunk[2, 2] = False
    if int(shrunk.sum()) < 3:
        shrunk[1, 1] = True
    return shrunk


def _precompute_pattern_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    """Enumerate majority/anomaly mask pairs with a visible shrink relation."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    optional_groups = [
        _bool_mask(["000", "101", "000"]),  # middle left/right pair
        _bool_mask(["010", "000", "000"]),  # top center
        _bool_mask(["000", "010", "000"]),  # center
        _bool_mask(["000", "000", "010"]),  # bottom center
    ]
    base = _bool_mask(["101", "000", "101"])  # corners are always present

    for bits in range(1 << len(optional_groups)):
        mask = base.copy()
        for idx, addon in enumerate(optional_groups):
            if bits & (1 << idx):
                mask |= addon
        active = int(mask.sum())
        if active < 6 or active > 8:
            continue
        shrunk = _compress_majority_mask(mask)
        shrunk_active = int(shrunk.sum())
        if shrunk_active < 3 or shrunk_active > 5:
            continue
        if np.array_equal(mask, shrunk):
            continue
        pairs.append((mask.astype(bool), shrunk.astype(bool)))

    assert pairs, "need at least one valid majority/anomaly pair"
    return pairs


_PATTERN_PAIRS = _precompute_pattern_pairs()


def _pattern_from_mask(mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    return np.where(mask, fg, bg).astype(int)


def _build_grid(separator: int, majority: np.ndarray, anomaly: np.ndarray, anomaly_pos: tuple[int, int] | None) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i, j in product(range(CELL_GRID_N), range(CELL_GRID_N)):
        paste(grid, majority, *_cell_origin(i, j))
    if anomaly_pos is not None:
        paste(grid, anomaly, *_cell_origin(*anomaly_pos))
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    separator, bg, fg = random.sample(range(COLOR_POOL), 3)
    majority_mask, anomaly_mask = random.choice(_PATTERN_PAIRS)
    majority = _pattern_from_mask(majority_mask, bg, fg)
    anomaly = _pattern_from_mask(anomaly_mask, bg, fg)
    if np.array_equal(majority, anomaly):
        return None

    anomaly_pos = (random.randrange(CELL_GRID_N), random.randrange(CELL_GRID_N))
    inp = _build_grid(separator, majority, anomaly, anomaly_pos)
    out = _build_grid(separator, majority, anomaly, None)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid b99e7126 combo D3/1+D4/1 input")
