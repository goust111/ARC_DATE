# Variant of b99e7126
# Dimension: D8 — 单元格尺寸：铺贴单元的大小 (meta)
# Change: 固定的3x3单元格大小... → 单元格尺寸可变（需自行推断）

from itertools import combinations
import random

import numpy as np

from arc_utils import paste


# Design analysis for the D8 variant
# ----------------------------------
# The base solver already tolerates different square cell sizes, so this variant has
# to make the size inference genuinely matter. The generated boards therefore use
# rectangular cell interiors whose height and width both vary across samples. The
# same fractal rule still applies: the minority interior is the blueprint, but now
# the solver must infer separate row/column cell steps and handle a rectangular mask.

SIZE_CONFIGS = (
    {"inner_h": 2, "inner_w": 3, "n_rows": 7, "n_cols": 7, "active_counts": (4, 5), "given_sizes": (3, 4)},
    {"inner_h": 3, "inner_w": 2, "n_rows": 7, "n_cols": 7, "active_counts": (4, 5), "given_sizes": (3, 4)},
    {"inner_h": 4, "inner_w": 3, "n_rows": 5, "n_cols": 6, "active_counts": (7, 8), "given_sizes": (4, 5, 6)},
)
COLOR_POOL = 10
MAX_RETRIES = 400


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    """
    Infer a regular rectangular cell grid separated by full rows/cols of one line color.
    Returns (row_step, col_step, row_divs, col_divs).
    """
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"
    row_divs.sort()
    col_divs.sort()
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must be at index 0"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must be at boundary"

    row_steps = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1)]
    col_steps = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1)]
    assert len(set(row_steps)) == 1 and len(set(col_steps)) == 1, "dividers must be regularly spaced"
    row_step = int(row_steps[0])
    col_step = int(col_steps[0])
    assert row_step > 1 and col_step > 1, "cell steps must exceed 1"
    assert (h - 1) % row_step == 0 and (w - 1) % col_step == 0, "grid dimensions must be k*step+1"
    return row_step, col_step, row_divs, col_divs


def _extract_patterns(g: np.ndarray, row_step: int, col_step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Extract every cell interior as both a flat key and a 2D array."""
    n_rows = (g.shape[0] - 1) // row_step
    n_cols = (g.shape[1] - 1) // col_step
    inner_h = row_step - 1
    inner_w = col_step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n_rows):
        row: list[tuple[int, ...]] = []
        for j in range(n_cols):
            r0 = i * row_step + 1
            c0 = j * col_step + 1
            pat_arr = g[r0 : r0 + inner_h, c0 : c0 + inner_w].copy()
            pat = tuple(int(v) for v in pat_arr.ravel())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return patterns, cell_pat


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a repeated cell mosaic after inferring a variable rectangular cell size.

    Input structure:
    - The board is split by one-color divider rows and columns into equal rectangular cells.
    - Different samples may use 2x3, 3x2, or 4x3 cell interiors instead of a fixed 3x3 size.
    - Most cells share one interior motif, while a smaller subset shows one minority motif.

    Rule:
    - Infer the row and column divider spacing separately to recover the cell size.
    - Read the minority cell's full interior as a rectangular boolean mask, using its most frequent color
      as the active value.
    - Slide that mask across the coarse cell grid until the visible minority cells fit in exactly one place,
      then repaint the entire covered rectangle with minority cells on active mask pixels and majority cells
      on inactive mask pixels.

    Key insight: The unusual cell is still a miniature blueprint, but now its resolution is a rectangular
    size that must be inferred from the divider lattice instead of assumed to be 3x3.
    """
    g = np.asarray(grid, dtype=int)
    row_step, col_step, _, _ = _infer_cell_grid_params(g)
    inner_h = row_step - 1
    inner_w = col_step - 1
    n_rows = (g.shape[0] - 1) // row_step
    n_cols = (g.shape[1] - 1) // col_step

    patterns, cell_pat = _extract_patterns(g, row_step, col_step)
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        for j in range(n_cols):
            existing[i, j] = cell_pat[i][j] == minority_pat

    candidates: list[tuple[int, int]] = []
    for top in range(0, n_rows - mh + 1):
        for left in range(0, n_cols - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique mask placement aligning minority cells"
    top, left = candidates[0]

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target = minority_arr if mask[i, j] else majority_arr
            r0 = (top + i) * row_step + 1
            c0 = (left + j) * col_step + 1
            out[r0 : r0 + inner_h, c0 : c0 + inner_w] = target
    return out


def _precompute_masks() -> dict[tuple[int, int, int], list[np.ndarray]]:
    """Enumerate valid rectangular masks keyed by (inner_h, inner_w, active_count)."""
    masks: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for cfg in SIZE_CONFIGS:
        inner_h = int(cfg["inner_h"])
        inner_w = int(cfg["inner_w"])
        total = inner_h * inner_w
        coords = list(range(total))
        for active_count in cfg["active_counts"]:
            key = (inner_h, inner_w, int(active_count))
            bucket: list[np.ndarray] = []
            for combo in combinations(coords, int(active_count)):
                mask = np.zeros((inner_h, inner_w), dtype=bool)
                for idx in combo:
                    r, c = divmod(idx, inner_w)
                    mask[r, c] = True
                if not mask[0, :].any():
                    continue
                if not mask[-1, :].any():
                    continue
                if not mask[:, 0].any():
                    continue
                if not mask[:, -1].any():
                    continue
                bucket.append(mask)
            masks[key] = bucket
    return masks


MASKS_BY_KEY = _precompute_masks()


def _random_pattern(inner_h: int, inner_w: int, color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform two-color rectangular pattern."""
    arr = np.array(
        [[random.choice((color_a, color_b)) for _ in range(inner_w)] for _ in range(inner_h)],
        dtype=int,
    )
    if np.all(arr == arr[0, 0]):
        arr[0, 0] = color_b if arr[0, 0] == color_a else color_a
    return arr


def _choose_mask(inner_h: int, inner_w: int, active_counts: tuple[int, ...]) -> np.ndarray:
    """Choose a size-appropriate minority mask."""
    active_count = int(random.choice(active_counts))
    return random.choice(MASKS_BY_KEY[(inner_h, inner_w, active_count)]).copy()


def _cell_origin(i: int, j: int, row_step: int, col_step: int) -> tuple[int, int]:
    return i * row_step + 1, j * col_step + 1


def _grid_shape(inner_h: int, inner_w: int, n_rows: int, n_cols: int) -> tuple[int, int]:
    return n_rows * (inner_h + 1) + 1, n_cols * (inner_w + 1) + 1


def _build_grid(
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    active_cells: list[tuple[int, int]],
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Fill the whole cell grid with majority cells, then overwrite active positions."""
    inner_h, inner_w = majority.shape
    row_step = inner_h + 1
    col_step = inner_w + 1
    h, w = _grid_shape(inner_h, inner_w, n_rows, n_cols)
    grid = np.full((h, w), separator, dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            paste(grid, majority, *_cell_origin(i, j, row_step, col_step))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j, row_step, col_step))
    return grid


def _placement_candidates(mask: np.ndarray, given_cells: tuple[tuple[int, int], ...], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    """Return all placements that can explain the visible minority cells."""
    existing = np.zeros((n_rows, n_cols), dtype=bool)
    for r, c in given_cells:
        existing[r, c] = True

    mh, mw = mask.shape
    candidates: list[tuple[int, int]] = []
    for top in range(n_rows - mh + 1):
        for left in range(n_cols - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[top : top + mh, left : left + mw] & ~mask):
                continue
            candidates.append((top, left))
    return candidates


def _choose_given_cells(
    mask: np.ndarray,
    top: int,
    left: int,
    n_rows: int,
    n_cols: int,
    given_sizes: tuple[int, ...],
) -> list[tuple[int, int]] | None:
    """Choose a strict subset of active cells that fixes the rectangular placement uniquely."""
    active_abs = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]
    sizes = list(given_sizes)
    random.shuffle(sizes)
    for size in sizes:
        if size >= len(active_abs):
            continue
        valid = [
            list(subset)
            for subset in combinations(active_abs, size)
            if _placement_candidates(mask, subset, n_rows, n_cols) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt for the variable-rectangular-cell variant."""
    cfg = random.choice(SIZE_CONFIGS)
    inner_h = int(cfg["inner_h"])
    inner_w = int(cfg["inner_w"])
    n_rows = int(cfg["n_rows"])
    n_cols = int(cfg["n_cols"])
    active_counts = tuple(int(v) for v in cfg["active_counts"])
    given_sizes = tuple(int(v) for v in cfg["given_sizes"])

    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_pattern(inner_h, inner_w, shared, maj_exclusive)
    mask = _choose_mask(inner_h, inner_w, active_counts)
    minority = np.where(mask, min_exclusive, shared).astype(int)
    if np.array_equal(majority, minority):
        return None

    top = random.randrange(n_rows - inner_h + 1)
    left = random.randrange(n_cols - inner_w + 1)
    active_cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]
    out = _build_grid(separator, majority, minority, active_cells, n_rows, n_cols)

    given_cells = _choose_given_cells(mask, top, left, n_rows, n_cols, given_sizes)
    if given_cells is None:
        return None
    inp = _build_grid(separator, majority, minority, given_cells, n_rows, n_cols)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 D8 input")
