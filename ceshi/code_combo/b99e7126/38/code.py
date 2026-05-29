# Combo variant of b99e7126 (index 38)
# combo_spec: D13/2+D14/1
# Name: 镜像反射边界 + 镜像翻转叠加
#
# Design analysis for the fused variant:
# a) Concrete behavior: treat the minority 3x3 interior as a cell-scale mask as before,
#    but first mirror that mask along one axis, then place the conceptual 3x3 source block
#    on the 7x7 cell lattice even if part of it crosses a border. Example: with source
#    origin (-1, 5) and a horizontal mirror, an active local pixel at (0, 2) becomes
#    absolute (-1, 5); after reflective folding it lands at (1, 5). The original task
#    would instead keep the mask unflipped and require all target cells to stay inside 0..6.
# b) Complexity calibration: the axis choice has only two states (horizontal/vertical),
#    determined by which source-block center offset dominates relative to grid center.
#    This keeps the rule visible while avoiding too many unrelated transform cases.
# c) Broken assumptions from the original solver:
#    - LR-symmetric masks would make horizontal mirroring invisible, so the generator now
#      enforces asymmetry under both left-right and up-down flips.
#    - The placement is no longer restricted to a fully in-bounds 3x3 window.
#    - Candidate checking can no longer use a simple rectangular block test; it must
#      evaluate mirrored and reflect-folded active coordinates explicitly.
# d) Placement legality: the conceptual source origin is limited to [-2, 6] per axis,
#    which is enough to express single-border overflow for a 3x3 block without introducing
#    periodic aliases from repeated reflections. Generator verification still requires a
#    unique valid source candidate.
# e) Visual observability: every generated sample has real overflow and a non-symmetric
#    mask, so output cells differ from both the original unclipped overlay and a
#    reflection-only overlay. The border fold is visible as edge-adjacent symmetry.
# f) Visual cleanliness: only two cell interiors remain in play (majority/minority), so
#    even with folded overlaps the output stays readable; duplicate hits simply re-stamp
#    the same minority cell pattern.
# g) Inferability: there is no hidden color remapping. Colors keep the original role split
#    (separator, majority-only, shared, minority-only), and the new information is carried
#    by geometric transformation alone.

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = (4, 3)
COLOR_POOL = 10
MAX_RETRIES = 500


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
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
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1) if row_divs[i + 1] > row_divs[i]]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1) if col_divs[i + 1] > col_divs[i]]
    step = reduce(gcd, drs + dcs)
    assert step > 1, "divider step must be > 1"
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must be at index 0"
    assert (h - 1) % step == 0 and (w - 1) % step == 0, "grid dimensions must be (k*step + 1)"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must be at boundary"
    return int(step), row_divs, col_divs


def _reflect_index(idx: int, n: int) -> int:
    """Mirror-fold an integer index into [0, n-1] with repeated boundary reflection."""
    if n == 1:
        return 0
    period = 2 * n - 2
    folded = idx % period
    if folded < n:
        return int(folded)
    return int(period - folded)


def _source_ranges(mask: np.ndarray, n: int) -> tuple[range, range]:
    """Smallest origin ranges that allow a 3x3 source block to overflow by up to its full extent."""
    mh, mw = mask.shape
    return range(-(mh - 1), n), range(-(mw - 1), n)


def _dominant_axis(top: int, left: int, mh: int, mw: int, n: int) -> str | None:
    """Choose mirror axis from source-block center vs grid center; ties are intentionally invalid."""
    center = (n - 1) / 2.0
    dr = top + (mh - 1) / 2.0 - center
    dc = left + (mw - 1) / 2.0 - center
    if abs(dc) > abs(dr):
        return "h"
    if abs(dr) > abs(dc):
        return "v"
    return None


def _project_positions(mask: np.ndarray, top: int, left: int, n: int, *, apply_flip: bool) -> tuple[set[tuple[int, int]], str | None, int]:
    """
    Project active mask pixels onto the n x n cell lattice.
    Returns (final_positions, axis, n_overflowing_pixels).
    """
    mh, mw = mask.shape
    axis = _dominant_axis(top, left, mh, mw, n) if apply_flip else None
    if apply_flip and axis is None:
        return set(), None, -1

    final_positions: set[tuple[int, int]] = set()
    overflow = 0
    for rr, cc in np.argwhere(mask):
        r = int(rr)
        c = int(cc)
        if axis == "h":
            c = mw - 1 - c
        elif axis == "v":
            r = mh - 1 - r
        raw_r = top + r
        raw_c = left + c
        if not (0 <= raw_r < n) or not (0 <= raw_c < n):
            overflow += 1
        final_positions.add((_reflect_index(raw_r, n), _reflect_index(raw_c, n)))
    return final_positions, axis, overflow


def _extract_state(grid: np.ndarray) -> dict[str, object]:
    """Parse the grid into cell-level patterns and the visible minority-cell map."""
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            counts[pat] = counts.get(pat, 0) + 1
            row.append(pat)
        cell_pat.append(row)

    assert len(counts) >= 2, "must have at least 2 distinct cell patterns"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    order = np.argsort(-cnts)
    assert len(order) >= 2 and cnts[order[0]] > cnts[order[1]], "minority pattern needs a unique dominant color"
    active_color = int(vals[order[0]])
    mask = minority_arr == active_color

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

    return {
        "grid": g,
        "step": step,
        "inner": inner,
        "n": n,
        "mask": mask,
        "minority_arr": minority_arr,
        "majority_arr": majority_arr,
        "existing": existing,
    }


def _candidate_sources(mask: np.ndarray, existing: np.ndarray) -> list[dict[str, object]]:
    """Enumerate source origins whose mirrored+reflected projection contains all visible minority cells."""
    n = existing.shape[0]
    existing_pts = {tuple(pt) for pt in np.argwhere(existing)}
    candidates: list[dict[str, object]] = []
    row_range, col_range = _source_ranges(mask, n)
    for top in row_range:
        for left in col_range:
            final_positions, axis, overflow = _project_positions(mask, top, left, n, apply_flip=True)
            if axis is None:
                continue
            if existing_pts <= final_positions:
                candidates.append(
                    {
                        "top": int(top),
                        "left": int(left),
                        "axis": axis,
                        "overflow": int(overflow),
                        "positions": final_positions,
                    }
                )
    return candidates


def _build_output(state: dict[str, object], candidate: dict[str, object]) -> np.ndarray:
    """Render the full output from a solved candidate source."""
    g = np.asarray(state["grid"], dtype=int)
    out = g.copy()
    step = int(state["step"])
    inner = int(state["inner"])
    n = int(state["n"])
    majority_arr = np.asarray(state["majority_arr"], dtype=int)
    minority_arr = np.asarray(state["minority_arr"], dtype=int)
    active_positions = set(candidate["positions"])

    for i in range(n):
        for j in range(n):
            pat = minority_arr if (i, j) in active_positions else majority_arr
            r0 = i * step + 1
            c0 = j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = pat
    return out


def _solve_candidate(grid: np.ndarray) -> tuple[dict[str, object], dict[str, object]]:
    state = _extract_state(grid)
    candidates = _candidate_sources(np.asarray(state["mask"], dtype=bool), np.asarray(state["existing"], dtype=bool))
    assert len(candidates) == 1, "must have a unique mirrored source placement"
    return state, candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal cell overlay after mirroring the blueprint and reflecting any overflow at the borders.

    Input structure:
    - The canvas is a 7x7 grid of 3x3 cell interiors separated by 1-pixel divider lines.
    - Most cells share one interior pattern; a minority pattern appears in only a few visible cells.
    - The minority cell's dominant color defines a 3x3 binary blueprint, but the shown minority cells are only a subset of the final target positions.

    Rule:
    - Read the minority cell's 3x3 interior as a cell-scale mask.
    - Choose the mirror axis from the conceptual source block's offset relative to the grid center: larger horizontal offset means a left-right flip, larger vertical offset means an up-down flip.
    - Overlay that mirrored 3x3 source block on the 7x7 cell lattice, allowing it to cross borders; any out-of-range cell coordinate folds back by mirror reflection at the touched boundary.
    - Fill every reflected active position with the minority cell pattern and restore all other cells to the majority pattern.

    Key insight:
    The minority cell still acts as a miniature blueprint, but now the correct large-scale footprint is obtained only after combining two geometric steps: mirror the blueprint first, then reflect any overflowing target cells back into the grid.
    """
    state, candidate = _solve_candidate(grid)
    return _build_output(state, candidate)


def _iter_masks() -> list[np.ndarray]:
    """Enumerate visually informative 3x3 masks for the fused variant."""
    masks: list[np.ndarray] = []
    cells = list(range(MASK_SIZE * MASK_SIZE))
    for active_count in ACTIVE_PIXEL_COUNTS:
        for combo in combinations(cells, active_count):
            mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
            for idx in combo:
                mask[idx // MASK_SIZE, idx % MASK_SIZE] = True
            if not all(mask[r, :].any() for r in range(MASK_SIZE)):
                continue
            if not all(mask[:, c].any() for c in range(MASK_SIZE)):
                continue
            if np.array_equal(mask, np.fliplr(mask)):
                continue
            if np.array_equal(mask, np.flipud(mask)):
                continue
            masks.append(mask)
    return masks


_VALID_MASKS = _iter_masks()


def _random_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform 3x3 pattern using exactly two colors."""
    arr = np.array([random.choice((color_a, color_b)) for _ in range(MASK_SIZE * MASK_SIZE)], dtype=int).reshape(MASK_SIZE, MASK_SIZE)
    if len(set(arr.ravel().tolist())) < 2:
        rr = random.randrange(MASK_SIZE)
        cc = random.randrange(MASK_SIZE)
        arr[rr, cc] = color_b if arr[rr, cc] == color_a else color_a
    return arr


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, active_cells: set[tuple[int, int]]) -> np.ndarray:
    """Construct the full pixel grid by stamping cell interiors on a divider-colored canvas."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _all_source_specs(mask: np.ndarray) -> list[dict[str, object]]:
    """Precompute all legal source placements worth sampling for a given mask."""
    n = CELL_GRID_N
    row_range, col_range = _source_ranges(mask, n)
    specs: list[dict[str, object]] = []
    for top in row_range:
        for left in col_range:
            positions, axis, overflow = _project_positions(mask, top, left, n, apply_flip=True)
            if axis is None or overflow <= 0:
                continue
            base_positions, _, _ = _project_positions(mask, top, left, n, apply_flip=False)
            if positions == base_positions:
                continue
            if len(positions) < 5:
                continue
            if axis == "h" and not (left < 0 or left + mask.shape[1] - 1 >= n):
                continue
            if axis == "v" and not (top < 0 or top + mask.shape[0] - 1 >= n):
                continue
            specs.append(
                {
                    "top": int(top),
                    "left": int(left),
                    "axis": axis,
                    "overflow": int(overflow),
                    "positions": positions,
                    "legacy_positions": base_positions,
                }
            )
    return specs


def _subset_candidates(mask: np.ndarray, subset: tuple[tuple[int, int], ...]) -> list[dict[str, object]]:
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[r, c] = True
    return _candidate_sources(mask, existing)


def _choose_given_subset(mask: np.ndarray, target_spec: dict[str, object]) -> tuple[tuple[int, int], ...] | None:
    """Choose a visible subset that uniquely pins down the hidden mirrored source."""
    active = sorted(target_spec["positions"])
    for size in GIVEN_CELL_SIZES:
        if size >= len(active):
            continue
        valid = [
            subset
            for subset in combinations(active, size)
            if len(cands := _subset_candidates(mask, subset)) == 1
            and cands[0]["top"] == target_spec["top"]
            and cands[0]["left"] == target_spec["left"]
        ]
        if valid:
            return random.choice(valid)
    return None


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt; reject unless solver reconstruction is exact and non-degenerate."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_pattern(shared, maj_exclusive)
    mask = random.choice(_VALID_MASKS)
    minority = np.where(mask, min_exclusive, shared).astype(int)

    specs = _all_source_specs(mask)
    if not specs:
        return None
    target_spec = random.choice(specs)
    final_positions = set(target_spec["positions"])

    subset = _choose_given_subset(mask, target_spec)
    if subset is None:
        return None

    out = _build_grid(separator, majority, minority, final_positions)
    inp = _build_grid(separator, majority, minority, set(subset))

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
    raise RuntimeError("failed to generate a valid fused b99e7126 input")
