# Combo variant of b99e7126 (index 30)
# combo_spec: D6/3+D8/0
# Name: 带边框单元格 + 可变单元格尺寸

from itertools import combinations
import random

import numpy as np

from arc_utils import paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    - The board is a square cell lattice with 1-pixel separator rows/cols. If separators are at
#      rows/cols 0,5,10,15,20,25 then the cell step is 5 and each usable interior is 4x4.
#    - Suppose the minority interior is
#        9 9 8 8
#        9 8 8 8
#        9 9 9 8
#        8 9 9 9
#      The dominant color 9 becomes a 4x4 boolean mask. At the coarse cell-grid level, only a strict
#      subset of the mask-active cells is visible in the input. The output completes the whole 4x4
#      placement by repainting active coarse cells with the minority pattern and inactive covered cells
#      with the majority pattern.
#    - Relative to the fixed-size seed, two linked things now vary together: the separator cadence and
#      the minority blueprint resolution. The solver must read the lattice first before it even knows
#      the size of the miniature blueprint.
# b) Complexity calibration:
#    - Only two interior sizes are used: 2x2 and 4x4. This makes the size change visually obvious
#      without introducing too many regimes.
#    - For 2x2, the minority mask uses exactly 3 active pixels so the dominant color is unambiguous.
#      For 4x4, active counts are 9-11, which keeps both colors visible while preserving a clear majority.
# c) Hidden-assumption audit against the original main():
#    - Fixed 3x3 extraction is invalid now, so the solver infers a regular separator lattice first and
#      derives `inner = step - 1`.
#    - Assuming the separator color is simply `grid[0,0]` is unnecessarily brittle. The solver instead
#      searches all colors for the unique one that forms full rows/cols at equal spacing.
#    - The placement search still assumes the minority cell's dominant color defines the active mask.
#      The generator guarantees that by making the active count strictly greater than half the interior.
# d) Placement legality:
#    - Every config satisfies `n_cells >= inner`, so the inner blueprint fits inside the coarse grid.
#    - Candidate top/left offsets stay in `0 .. n_cells - inner`, and all pastes use derived cell origins,
#      so cells never overlap incorrectly and never leave the canvas.
# e) Visual observability:
#    - The separator color is reserved and never reused inside cell interiors.
#    - The majority motif is a clean two-color texture; the minority motif is a connected edge-spanning
#      mask. Across 2-4 examples a human can see both the border lattice and the fact that the unusual
#      cell is a resized placement blueprint.
# f) Visual cleanliness:
#    - There are only four colors total. The divider lattice, majority cells, and minority cells each have
#      a distinct visual role, so the completed output reads as a single structured overlay rather than as
#      layered clutter.
# g) Inferrable variation:
#    - The only changing meta-parameter is the cell interior size. Because the grid dimensions and divider
#      cadence both change with that size, the variation is visible from the pixels alone rather than from
#      any hidden dependence on color ids or ordering.

SIZE_CONFIGS = (
    {"inner": 2, "n_cells": 7, "active_counts": (3,), "given_sizes": (2, 3)},
    {"inner": 4, "n_cells": 5, "active_counts": (9, 10, 11), "given_sizes": (4, 5, 6)},
)
COLOR_POOL = 10
MAX_RETRIES = 500


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in arr.ravel())


def _full_line_positions(g: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == color)]
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == color)]
    return rows, cols


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    """
    Infer a square cell lattice from full separator rows/cols of one color.
    Returns (separator_color, step, row_divs, col_divs).
    """
    h, w = g.shape
    candidates: list[tuple[int, int, list[int], list[int]]] = []
    for color in np.unique(g):
        line = int(color)
        row_divs, col_divs = _full_line_positions(g, line)
        if len(row_divs) < 2 or len(col_divs) < 2:
            continue
        row_divs.sort()
        col_divs.sort()
        if row_divs[0] != 0 or col_divs[0] != 0:
            continue
        if row_divs[-1] != h - 1 or col_divs[-1] != w - 1:
            continue
        row_steps = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1)]
        col_steps = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1)]
        if not row_steps or not col_steps:
            continue
        if len(set(row_steps)) != 1 or len(set(col_steps)) != 1:
            continue
        step = int(row_steps[0])
        if step <= 1 or int(col_steps[0]) != step:
            continue
        if (h - 1) % step != 0 or (w - 1) % step != 0:
            continue
        candidates.append((line, step, row_divs, col_divs))
    assert len(candidates) == 1, "must have exactly one regular separator lattice"
    return candidates[0]


def _extract_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Extract each cell interior as a flattened key plus its 2D array."""
    n_cells = (g.shape[0] - 1) // step
    inner = step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_patterns: list[list[tuple[int, ...]]] = []
    for i in range(n_cells):
        row: list[tuple[int, ...]] = []
        for j in range(n_cells):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            key = _pattern_key(arr)
            if key not in patterns:
                patterns[key] = arr
            row.append(key)
        cell_patterns.append(row)
    return patterns, cell_patterns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal cell mosaic after inferring a bordered lattice with variable cell size.

    Input structure:
    - The board is divided by 1-pixel-wide separator rows and columns into equal square cells.
    - Different samples use different cell steps, so the usable interior may be 2x2 or 4x4 rather
      than a fixed 3x3.
    - Nearly every cell shares one repeated interior motif, while a minority motif appears in only a
      few coarse-grid positions.

    Rule:
    - First detect the separator lattice and derive the interior size from its spacing.
    - Read the minority cell's interior as a square boolean mask, using its most frequent color as
      the active value.
    - Slide that mask over the coarse cell grid and find the unique placement whose active positions
      are consistent with the already visible minority cells.
    - Repaint the whole covered window: minority motif on active mask cells, majority motif on inactive
      mask cells, while leaving the separator lattice unchanged.

    Key insight: The odd cell is both the anomaly and the legend. Its interior resolution changes from
    sample to sample, and the border lattice is the clue that tells you how large that legend really is.
    """
    g = np.asarray(grid, dtype=int)
    _separator, step, _row_divs, _col_divs = _infer_cell_grid_params(g)
    inner = step - 1
    n_cells = (g.shape[0] - 1) // step

    patterns, cell_patterns = _extract_patterns(g, step)
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_patterns:
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
    assert mask.shape == (inner, inner), "minority mask must match inferred cell interior size"

    existing = np.zeros((n_cells, n_cells), dtype=bool)
    for i in range(n_cells):
        for j in range(n_cells):
            existing[i, j] = cell_patterns[i][j] == minority_pat

    candidates: list[tuple[int, int]] = []
    for top in range(n_cells - inner + 1):
        for left in range(n_cells - inner + 1):
            block = np.zeros_like(existing)
            block[top : top + inner, left : left + inner] = True
            if np.any(existing & ~block):
                continue
            visible = existing[top : top + inner, left : left + inner]
            if np.any(visible & ~mask):
                continue
            candidates.append((top, left))
    assert len(candidates) == 1, "must have a unique placement for the minority blueprint"
    top, left = candidates[0]

    out = g.copy()
    for i in range(inner):
        for j in range(inner):
            target = minority_arr if mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if not mask[nr, nc] or nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return len(seen) == len(coords)


def _precompute_masks() -> dict[tuple[int, int], list[np.ndarray]]:
    """Enumerate clean square masks keyed by (inner, active_count)."""
    masks: dict[tuple[int, int], list[np.ndarray]] = {}
    for cfg in SIZE_CONFIGS:
        inner = int(cfg["inner"])
        total = inner * inner
        coords = list(range(total))
        for active_count in cfg["active_counts"]:
            bucket: list[np.ndarray] = []
            for combo in combinations(coords, int(active_count)):
                mask = np.zeros((inner, inner), dtype=bool)
                for idx in combo:
                    r, c = divmod(idx, inner)
                    mask[r, c] = True
                if not mask[0, :].any() or not mask[-1, :].any():
                    continue
                if not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                if not _is_connected(mask):
                    continue
                if not _is_connected(~mask):
                    continue
                bucket.append(mask)
            assert bucket, f"no masks for inner={inner}, active_count={active_count}"
            masks[(inner, int(active_count))] = bucket
    return masks


MASKS_BY_KEY = _precompute_masks()


def _majority_pattern(inner: int, shared: int, maj_exclusive: int) -> np.ndarray:
    """Generate a neat two-color repeating interior motif for majority cells."""
    styles = ["checker", "rows", "cols"]
    if inner >= 4:
        styles.append("frame")
    style = random.choice(styles)
    arr = np.empty((inner, inner), dtype=int)
    if style == "checker":
        for r in range(inner):
            for c in range(inner):
                arr[r, c] = shared if (r + c) % 2 == 0 else maj_exclusive
    elif style == "rows":
        for r in range(inner):
            arr[r, :] = shared if r % 2 == 0 else maj_exclusive
    elif style == "cols":
        for c in range(inner):
            arr[:, c] = shared if c % 2 == 0 else maj_exclusive
    else:
        arr.fill(shared)
        arr[0, :] = maj_exclusive
        arr[-1, :] = maj_exclusive
        arr[:, 0] = maj_exclusive
        arr[:, -1] = maj_exclusive
    return arr


def _choose_mask(inner: int, active_counts: tuple[int, ...]) -> np.ndarray:
    active_count = int(random.choice(active_counts))
    return random.choice(MASKS_BY_KEY[(inner, active_count)]).copy()


def _grid_shape(n_cells: int, step: int) -> int:
    return n_cells * step + 1


def _cell_origin(i: int, j: int, step: int) -> tuple[int, int]:
    return i * step + 1, j * step + 1


def _build_grid(
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    active_cells: list[tuple[int, int]],
    n_cells: int,
) -> np.ndarray:
    """Construct the bordered grid from cell motifs plus a subset of active coarse cells."""
    inner = int(majority.shape[0])
    step = inner + 1
    size = _grid_shape(n_cells, step)
    grid = np.full((size, size), separator, dtype=int)
    for i in range(n_cells):
        for j in range(n_cells):
            paste(grid, majority, *_cell_origin(i, j, step))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j, step))
    return grid


def _placement_candidates(mask: np.ndarray, given_cells: tuple[tuple[int, int], ...], n_cells: int) -> list[tuple[int, int]]:
    """Return all placements that can explain the visible minority coarse cells."""
    existing = np.zeros((n_cells, n_cells), dtype=bool)
    for r, c in given_cells:
        existing[int(r), int(c)] = True

    inner = int(mask.shape[0])
    candidates: list[tuple[int, int]] = []
    for top in range(n_cells - inner + 1):
        for left in range(n_cells - inner + 1):
            block = np.zeros_like(existing)
            block[top : top + inner, left : left + inner] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[top : top + inner, left : left + inner] & ~mask):
                continue
            candidates.append((top, left))
    return candidates


def _choose_given_cells(
    mask: np.ndarray,
    top: int,
    left: int,
    n_cells: int,
    given_sizes: tuple[int, ...],
) -> list[tuple[int, int]] | None:
    """Choose a strict active subset that uniquely pins down the placement."""
    active_abs = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]
    sizes = list(given_sizes)
    random.shuffle(sizes)
    for size in sizes:
        if size >= len(active_abs):
            continue
        valid = [
            list(subset)
            for subset in combinations(active_abs, size)
            if _placement_candidates(mask, subset, n_cells) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output first, then remove some active coarse cells."""
    cfg = random.choice(SIZE_CONFIGS)
    inner = int(cfg["inner"])
    n_cells = int(cfg["n_cells"])
    active_counts = tuple(int(v) for v in cfg["active_counts"])
    given_sizes = tuple(int(v) for v in cfg["given_sizes"])

    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _majority_pattern(inner, shared, maj_exclusive)
    mask = _choose_mask(inner, active_counts)
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(n_cells - inner + 1)
    left = random.randrange(n_cells - inner + 1)
    active_cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]
    out = _build_grid(separator, majority, minority, active_cells, n_cells)

    given_cells = _choose_given_cells(mask, top, left, n_cells, given_sizes)
    if given_cells is None:
        return None
    inp = _build_grid(separator, majority, minority, given_cells, n_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
