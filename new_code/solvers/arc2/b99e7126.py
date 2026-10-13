import numpy as np

from arc_utils import gcd_list as _gcd_list
from arc_utils import most_common as _most_common

def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]] | None:
    """
    Infer a regular cell grid separated by full rows/cols of the same line color.
    Returns (step, row_divs, col_divs) where:
      - row_divs/col_divs are divider indices
      - step is the spacing between dividers (cell size including divider).
    """
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    if len(row_divs) < 2 or len(col_divs) < 2:
        return None
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1) if row_divs[i + 1] > row_divs[i]]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1) if col_divs[i + 1] > col_divs[i]]
    step = _gcd_list(drs + dcs)
    if step <= 1:
        return None
    # Prefer exact regular grids: dividers should occur at multiples of step from 0.
    if row_divs[0] != 0 or col_divs[0] != 0:
        return None
    if (h - 1) % step != 0 or (w - 1) % step != 0:
        return None
    # Ensure last divider at boundary.
    if row_divs[-1] != h - 1 or col_divs[-1] != w - 1:
        return None
    return int(step), row_divs, col_divs


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Scale a minority cell's inner pattern up to the cell-grid level, completing a fractal overlay.

    Input structure:
    - Grid divided into NxN cells by separator lines of a single color.
    - Most cells share an identical interior pattern (majority); a few have a distinct pattern (minority).

    Rule:
    - Extract the inner pixel pattern from the minority cell.
    - Treat each pixel as a cell-sized stamp: active pixels → minority pattern, inactive → majority pattern.
    - Position this scaled-up pattern on the cell grid so existing minority cells align with active pixels.
    - Fill in all remaining active positions with the minority pattern.

    Key insight: The minority cell's interior is a miniature blueprint of where minority cells should appear at the grid scale — fractal self-similarity.
    """
    g = np.asarray(grid, dtype=int)
    params = _infer_cell_grid_params(g)
    if params is None:
        return g.copy()
    step, _, _ = params
    n = (g.shape[0] - 1) // step
    inner = step - 1

    # Extract interior patterns per cell.
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)

    # Find minority/majority cell patterns.
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            p = cell_pat[i][j]
            counts[p] = counts.get(p, 0) + 1
    if len(counts) < 2:
        return g.copy()

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]

    # Mask derived from the minority cell's inner pattern:
    # active color is the most frequent color within the minority pattern.
    active_color = _most_common(minority_arr)
    mask = (minority_arr == active_color)
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

    # Find the unique placement of the m×m mask in the cell grid such that:
    # - all existing minority cells lie within the block
    # - within the block, existing minority cells only occur where mask is True
    best = None
    for top in range(0, n - mh + 1):
        for left in range(0, n - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            best = (top, left)
            break
        if best is not None:
            break
    if best is None:
        return g.copy()
    top, left = best

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = minority_arr if mask[i, j] else patterns[majority_pat]
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out
