# Combo variant of b99e7126 (index 18)
# combo_spec: D8/0+D15/2
# Name: 可变放大倍数 + 可变单元格尺寸

from itertools import combinations
import random

import numpy as np

from arc_utils import paste


# Design analysis for the combo variant
# -------------------------------------
# Concrete behavior trace:
# - Base task: a 3x3 minority cell is read as a 3x3 blueprint, and each blueprint pixel
#   maps to exactly one cell on the coarse grid.
# - This combo keeps the "odd cell is a blueprint" idea but changes two coupled stages.
#   First, the cell interior size is not fixed: a sample may use 3x3, 4x4, or 5x5 cells.
#   Second, the blueprint is no longer copied at 1-cell-per-pixel on the coarse grid.
#   If a 4x4 minority cell has 3 shared-color holes, the active 4x4 mask is expanded to a
#   12x12 coarse-cell mask because every source pixel becomes a 3x3 block of cells.
# - Example: with inner size 4 and hole count 3, the original rule would fill a 4x4 cell
#   block; the combo rule fills a 12x12 block with the same hole layout, just blown up in
#   block steps. The visible minority cells in the input are only a sparse subset of those
#   12x12 active positions.
#
# Complexity calibration:
# - The scale factor must be visually inferable and small enough to fit inside the board, so
#   it is limited to {2, 3}.
# - The factor is encoded by the number of holes inside the minority cell. That count is easy
#   for a human to see and is independent from the divider spacing that reveals the cell size.
# - Cell sizes use a short menu {3, 4, 5}; this is enough to visibly differ from the base task
#   without creating too many regimes for the solver.
#
# Implicit-assumption audit against the base solver:
# - "mask height equals placement height" no longer holds. The source mask is size inner x inner,
#   but the placed mask is kron(mask, ones(scale, scale)).
# - "scale factor can be read off from divider spacing" no longer holds. Divider spacing only
#   reveals cell size; scale is recovered separately from the minority cell content.
# - "active color is simply the most frequent color" is fragile once scale is encoded by holes.
#   The solver instead identifies the minority-exclusive color by comparing majority/minority
#   palettes, then counts the shared-color holes to recover the scale factor.
#
# Placement legality and cleanliness:
# - The generator chooses n_cells so the expanded mask always fits with margin, so there is no
#   overlap or out-of-bounds risk when the output fills the whole scaled mask.
# - The board still contains only divider lines plus two repeated cell motifs, so the enlarged
#   completion remains visually clean instead of becoming a cluttered multi-object scene.
#
# Observability:
# - A human can infer the cell size from divider spacing.
# - A human can infer the scale factor from the number of holes inside the odd cell.
# - The output makes the decoupling visible because the completed footprint spans inner*scale
#   cells per side, which differs from the original "inner cells per side" behavior whenever
#   scale != 1 and, in this generator, always differs from the cell size as well.

SIZE_CONFIGS = (
    {"inner": 3, "n_cells": 11, "scales": (2, 3), "given_sizes": (4, 5, 6)},
    {"inner": 4, "n_cells": 14, "scales": (2, 3), "given_sizes": (4, 5, 6, 7)},
    {"inner": 5, "n_cells": 17, "scales": (2, 3), "given_sizes": (5, 6, 7, 8)},
)
COLOR_POOL = 10
MAX_RETRIES = 400


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    """
    Infer a regular square cell grid separated by full rows/cols of one line color.
    Returns (step, n_cells, row_divs, col_divs).
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
    assert len(set(row_steps)) == 1 and len(set(col_steps)) == 1, "dividers must be regular"
    step_r = int(row_steps[0])
    step_c = int(col_steps[0])
    assert step_r == step_c, "combo generator uses square cells"
    step = step_r
    assert step > 1, "cell step must exceed 1"
    assert (h - 1) % step == 0 and (w - 1) % step == 0, "grid dimensions must be k*step+1"
    n_cells = (h - 1) // step
    assert n_cells == (w - 1) // step, "combo generator uses square cell grids"
    return step, int(n_cells), row_divs, col_divs


def _extract_patterns(g: np.ndarray, step: int, n_cells: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Extract cell interiors as both flat keys and 2D arrays."""
    inner = step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n_cells):
        row: list[tuple[int, ...]] = []
        for j in range(n_cells):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(int(v) for v in pat_arr.ravel())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return patterns, cell_pat


def _find_majority_minority(cell_pat: list[list[tuple[int, ...]]]) -> tuple[tuple[int, ...], tuple[int, ...], dict[tuple[int, ...], int]]:
    """Return (minority_key, majority_key, counts)."""
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) == 2, "generator guarantees exactly two cell patterns"
    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    return minority_pat, majority_pat, counts


def _infer_active_color(minority_arr: np.ndarray, majority_arr: np.ndarray) -> int:
    """The active color is the one that appears in the minority pattern but not the majority."""
    minority_colors = {int(v) for v in np.unique(minority_arr)}
    majority_colors = {int(v) for v in np.unique(majority_arr)}
    extra = sorted(minority_colors - majority_colors)
    assert len(extra) == 1, "minority pattern must contain exactly one exclusive color"
    return extra[0]


def _scaled_mask_from_patterns(minority_arr: np.ndarray, majority_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Recover the source mask, the expanded mask, and the independent scale factor."""
    active_color = _infer_active_color(minority_arr, majority_arr)
    source_mask = minority_arr == active_color
    scale = int(source_mask.size - np.count_nonzero(source_mask))
    assert scale > 1, "scale factor must be encoded by at least two holes"
    scaled_mask = np.kron(source_mask.astype(bool), np.ones((scale, scale), dtype=bool))
    return source_mask, scaled_mask, scale


def _find_unique_placement(existing: np.ndarray, scaled_mask: np.ndarray) -> tuple[int, int]:
    """Find the only placement where all observed minority cells land on active expanded pixels."""
    n_rows, n_cols = existing.shape
    sh, sw = scaled_mask.shape
    obs = [tuple(int(v) for v in coord) for coord in np.argwhere(existing)]
    assert obs, "input must expose at least one minority cell"

    candidates: list[tuple[int, int]] = []
    for top in range(n_rows - sh + 1):
        for left in range(n_cols - sw + 1):
            ok = True
            for r, c in obs:
                rr = r - top
                cc = c - left
                if rr < 0 or cc < 0 or rr >= sh or cc >= sw or not scaled_mask[rr, cc]:
                    ok = False
                    break
            if ok:
                candidates.append((top, left))
    assert len(candidates) == 1, "expanded mask placement must be unique"
    return candidates[0]


def _infer_instance_signature(grid: np.ndarray) -> tuple[int, int, tuple[int, int]]:
    """Helper for behavior audits: return (cell_inner_size, scale_factor, expanded_mask_shape)."""
    g = np.asarray(grid, dtype=int)
    step, n_cells, _, _ = _infer_cell_grid_params(g)
    patterns, cell_pat = _extract_patterns(g, step, n_cells)
    minority_pat, majority_pat, _ = _find_majority_minority(cell_pat)
    _, scaled_mask, scale = _scaled_mask_from_patterns(patterns[minority_pat], patterns[majority_pat])
    return step - 1, scale, tuple(int(v) for v in scaled_mask.shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal cell mosaic with independently inferred cell size and scale factor.

    Input structure:
    - One-color divider rows and columns split the board into equal square cells, but the cell
      interior size changes across samples instead of staying fixed at 3x3.
    - Most cells share one repeating interior pattern. A smaller set of cells shows a second
      pattern whose interior is mostly filled, with a few holes.

    Rule:
    - Infer the cell size from the divider spacing, then extract the majority and minority cell
      interiors.
    - Read the minority-exclusive color as the active blueprint pixels. The number of holes in
      that blueprint gives a separate scale factor k.
    - Expand the blueprint by replacing every source pixel with a kxk block of coarse cells,
      slide that expanded mask across the cell grid, and find the unique placement consistent
      with the visible minority cells.
    - Fill every active coarse cell in that placement with the minority pattern and every inactive
      coarse cell inside the bounding rectangle with the majority pattern.

    Key insight: The odd cell is still a miniature blueprint, but now two quantities must be
    disentangled: divider spacing tells you the cell resolution, while the blueprint's hole count
    tells you how many coarse cells each source pixel should expand into.
    """
    g = np.asarray(grid, dtype=int)
    step, n_cells, _, _ = _infer_cell_grid_params(g)
    inner = step - 1

    patterns, cell_pat = _extract_patterns(g, step, n_cells)
    minority_pat, majority_pat, _ = _find_majority_minority(cell_pat)
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]
    _, scaled_mask, _ = _scaled_mask_from_patterns(minority_arr, majority_arr)

    existing = np.zeros((n_cells, n_cells), dtype=bool)
    for i in range(n_cells):
        for j in range(n_cells):
            existing[i, j] = cell_pat[i][j] == minority_pat

    top, left = _find_unique_placement(existing, scaled_mask)

    out = g.copy()
    sh, sw = scaled_mask.shape
    for i in range(sh):
        for j in range(sw):
            target = minority_arr if scaled_mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target
    return out


def _precompute_masks() -> dict[tuple[int, int], list[np.ndarray]]:
    """Enumerate source masks keyed by (inner, scale) where scale equals the hole count."""
    by_key: dict[tuple[int, int], list[np.ndarray]] = {}
    for cfg in SIZE_CONFIGS:
        inner = int(cfg["inner"])
        coords = [(r, c) for r in range(inner) for c in range(inner)]
        for scale in cfg["scales"]:
            bucket: list[np.ndarray] = []
            for holes in combinations(coords, int(scale)):
                mask = np.ones((inner, inner), dtype=bool)
                for r, c in holes:
                    mask[r, c] = False
                if not mask[0, :].any():
                    continue
                if not mask[-1, :].any():
                    continue
                if not mask[:, 0].any():
                    continue
                if not mask[:, -1].any():
                    continue
                if np.array_equal(mask, np.fliplr(mask)):
                    continue
                if np.array_equal(mask, np.flipud(mask)):
                    continue
                row_codes = {tuple(int(v) for v in row) for row in mask}
                col_codes = {tuple(int(v) for v in col) for col in mask.T}
                if len(row_codes) < 2 or len(col_codes) < 2:
                    continue
                bucket.append(mask)
            assert bucket, f"no masks available for inner={inner}, scale={scale}"
            by_key[(inner, int(scale))] = bucket
    return by_key


MASKS_BY_KEY = _precompute_masks()


def _random_pattern(inner: int, color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform two-color square pattern."""
    arr = np.array(
        [[random.choice((color_a, color_b)) for _ in range(inner)] for _ in range(inner)],
        dtype=int,
    )
    if np.all(arr == arr[0, 0]):
        arr[0, 0] = color_b if arr[0, 0] == color_a else color_a
    return arr


def _choose_mask(inner: int, scale_choices: tuple[int, ...]) -> tuple[np.ndarray, int]:
    """Pick a source mask and return it with its independent scale factor."""
    scale = int(random.choice(scale_choices))
    return random.choice(MASKS_BY_KEY[(inner, scale)]).copy(), scale


def _signature_source_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    """Prioritize active source pixels that sit on the boundary or next to a hole."""
    inner = mask.shape[0]
    holes = [tuple(int(v) for v in coord) for coord in np.argwhere(~mask)]
    active = [tuple(int(v) for v in coord) for coord in np.argwhere(mask)]
    chosen: list[tuple[int, int]] = []
    for r, c in active:
        on_boundary = r in (0, inner - 1) or c in (0, inner - 1)
        near_hole = any(abs(r - hr) <= 1 and abs(c - hc) <= 1 for hr, hc in holes)
        if on_boundary or near_hole:
            chosen.append((r, c))
    seen = set(chosen)
    for cell in active:
        if len(chosen) >= min(len(active), 16):
            break
        if cell not in seen:
            chosen.append(cell)
            seen.add(cell)
    return chosen


def _representative_abs_cell(src_r: int, src_c: int, scale: int, top: int, left: int) -> tuple[int, int]:
    """
    Pick one deterministic visible cell inside an active kxk block.
    The offset varies with the source coordinate so different blocks do not all reveal
    the same internal phase.
    """
    dr = (2 * src_r + src_c) % scale
    dc = (src_r + 2 * src_c) % scale
    return top + src_r * scale + dr, left + src_c * scale + dc


def _placement_candidates(n_cells: int, scaled_mask: np.ndarray, given_cells: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    """Return all placements compatible with the visible minority cells."""
    sh, sw = scaled_mask.shape
    out: list[tuple[int, int]] = []
    for top in range(n_cells - sh + 1):
        for left in range(n_cells - sw + 1):
            ok = True
            for r, c in given_cells:
                rr = r - top
                cc = c - left
                if rr < 0 or cc < 0 or rr >= sh or cc >= sw or not scaled_mask[rr, cc]:
                    ok = False
                    break
            if ok:
                out.append((top, left))
    return out


def _choose_given_cells(
    mask: np.ndarray,
    scale: int,
    top: int,
    left: int,
    n_cells: int,
    given_sizes: tuple[int, ...],
) -> list[tuple[int, int]] | None:
    """
    Choose a strict subset of visible minority cells that uniquely fixes the expanded placement.
    Search happens over informative source blocks, not over every active cell in the enlarged mask.
    """
    source_pool = _signature_source_cells(mask)
    scaled_mask = np.kron(mask.astype(bool), np.ones((scale, scale), dtype=bool))
    reps = [_representative_abs_cell(r, c, scale, top, left) for r, c in source_pool]
    total_active_blocks = int(np.count_nonzero(mask))

    sizes = list(given_sizes)
    random.shuffle(sizes)
    for size in sizes:
        if size >= total_active_blocks or size > len(reps):
            continue
        combos = list(combinations(reps, size))
        random.shuffle(combos)
        for subset in combos:
            if _placement_candidates(n_cells, scaled_mask, subset) == [(top, left)]:
                return [tuple(int(v) for v in cell) for cell in subset]
    return None


def _cell_origin(i: int, j: int, step: int) -> tuple[int, int]:
    return i * step + 1, j * step + 1


def _build_grid(
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    active_cells: list[tuple[int, int]],
    n_cells: int,
) -> np.ndarray:
    """Fill the whole coarse grid with majority cells, then overwrite active positions."""
    inner = majority.shape[0]
    step = inner + 1
    size = n_cells * step + 1
    grid = np.full((size, size), separator, dtype=int)
    for i in range(n_cells):
        for j in range(n_cells):
            paste(grid, majority, *_cell_origin(i, j, step))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j, step))
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, int | tuple[int, int]]] | None:
    """Single output-first generation attempt for the combo variant."""
    cfg = random.choice(SIZE_CONFIGS)
    inner = int(cfg["inner"])
    n_cells = int(cfg["n_cells"])
    given_sizes = tuple(int(v) for v in cfg["given_sizes"])

    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_pattern(inner, shared, maj_exclusive)
    mask, scale = _choose_mask(inner, tuple(int(v) for v in cfg["scales"]))
    minority = np.where(mask, min_exclusive, shared).astype(int)

    scaled_mask = np.kron(mask.astype(bool), np.ones((scale, scale), dtype=bool))
    sh, sw = scaled_mask.shape
    if sh > n_cells or sw > n_cells:
        return None

    top = random.randrange(n_cells - sh + 1)
    left = random.randrange(n_cells - sw + 1)
    active_cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(scaled_mask)]
    out = _build_grid(separator, majority, minority, active_cells, n_cells)

    given_cells = _choose_given_cells(mask, scale, top, left, n_cells, given_sizes)
    if given_cells is None:
        return None
    inp = _build_grid(separator, majority, minority, given_cells, n_cells)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    meta = {
        "inner": inner,
        "scale": scale,
        "expanded_h": int(sh),
        "expanded_w": int(sw),
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        item = _gen_attempt()
        if item is not None:
            inp, _, _ = item
            return inp
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
