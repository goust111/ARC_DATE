# Combo variant of b99e7126 (index 44)
# combo_spec: D12/2+D16/1
# Name: 仅覆盖背景 + 内部对称图案

from __future__ import annotations

from functools import reduce
from itertools import combinations
from math import gcd
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.insert(0, str(ROOT / "solvers"))

from arc_utils import paste


# Design analysis for combo D12/2 + D16/1
# ----------------------------------------
# spec grounding:
#   - Keep the original 29x29 geometry exactly: a 7x7 lattice of 3x3 cell interiors,
#     divided by 1-pixel separator rows/cols.
#   - Keep the original output-first generator shape: choose a 3x3 cell-level mask,
#     place that mask once on the 7x7 cell lattice, reveal only a strict subset of its
#     active locations in the input, and require unique placement.
#   - Change only two linked ideas:
#       1. normal cells are recognized by an internal symmetry-bearing scaffold
#       2. the fractal replay paints only scaffold-background pixels instead of copying
#          the anomalous cell verbatim
#
# a) Concretization and behavior trace
#   Example trace with a LR-symmetric scaffold:
#   - Normal cell: three scaffold pixels use color F, the other six pixels use color B.
#     The pattern is mirror-symmetric.
#   - Abnormal blueprint cell: the same 3x3 footprint now uses only B and accent color A.
#     Its A-mask is asymmetric and lives only on positions that are background in the
#     normal scaffold.
#   - Original task would copy that abnormal cell into every mask-active lattice slot.
#   - This combo instead:
#       1. reads the symmetric scaffold from the repeated normal cells
#       2. reads the asymmetric A-mask from the blueprint cell
#       3. locates the unique 3x3 lattice placement consistent with the revealed
#          blueprint cells
#       4. reconstructs each active output cell as "normal scaffold + A painted only on
#          scaffold background pixels selected by the blueprint mask"
#   Difference: the input anomaly is now a raw stencil, while the output active cells are
#   hybrids that restore the symmetric scaffold and add the accent only in empty slots.
#
# b) Complexity calibration
#   - Use exactly two repeated cell patterns in the input: one symmetric scaffold and one
#     asymmetric blueprint. This keeps the task readable from a few ARC examples.
#   - Limit scaffold families to three clear symmetries (LR, UD, 180-degree rotation).
#   - Limit blueprint active counts to 4 or 5 pixels. That keeps the lattice placement
#     constrained without making the mask visually noisy.
#
# c) Hidden assumption audit against the original main()
#   - The old solver assumed the rare cell pattern was also the final cell to stamp into
#     the output. That is now false: the rare cell is only a blueprint.
#   - The old solver treated the minority cell's dominant color as both "active pixels"
#     and the final cell content. That breaks when the final active cell must preserve the
#     normal scaffold foreground.
#   - The old solver never needed to know which color inside the majority cell was
#     background. The combo does: only those pixels are paintable.
#   - The old solver relied only on frequency. The combo still uses repetition, but now
#     also asserts that the repeated normal cell has an internal symmetry and the blueprint
#     breaks it.
#
# d) Placement legality
#   - The 7x7 lattice and 3x3 mask footprint are unchanged, so boundary safety is still
#     `top,left in 0..4`.
#   - Blueprint masks are generated only from scaffold-background coordinates, so the
#     "paint only background" rule is always legal.
#   - The generator still reveals only a strict subset of active lattice cells and rejects
#     ambiguous subsets, preserving unique placement.
#
# e) Visual observability
#   - Humans can see that most cells share a small symmetric motif, while the rare cells
#     are asymmetric stencils in a different color pair.
#   - In the output, those rare-looking cells do not merely spread; they turn into
#     scaffold-preserving hybrid cells, making the background-only repaint visible.
#
# f) Visual tidiness
#   - There is only one overlay layer: accent A printed onto the scaffold background.
#   - The scaffold remains readable because its foreground color is never overwritten.
#   - No two objects overlap beyond the intended per-cell hybridization.
#
# g) Change inferability
#   - Colors are tied to structure: B is the repeated scaffold background, F is the
#     repeated scaffold foreground, and A is the blueprint/overlay accent.
#   - The output rule is supported by repeated evidence: every active output cell keeps F
#     in the same scaffold positions while A appears only where the blueprint mask says so.
#
# h) New solver assumptions and generator contract
#   - Exactly two cell patterns appear in the input.
#   - The majority pattern has a non-empty symmetry set among {LR, UD, ROT180}.
#   - The minority pattern uses the majority background color plus one accent color.
#   - Minority active pixels are a strict subset of majority-background positions and they
#     break every symmetry shared by the majority scaffold.
#   The generator enforces all of these assumptions explicitly.


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (4, 5)
GIVEN_CELL_SIZES = (3, 4)
COLOR_POOL = 10
MAX_RETRIES = 500

SYMMETRY_KINDS = ("lr", "ud", "rot180")

SCAFFOLD_FG = {
    "lr": ((0, 0), (0, 2), (1, 1)),
    "ud": ((0, 0), (2, 0), (1, 1)),
    "rot180": ((0, 0), (1, 1), (2, 2)),
}


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
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


def _has_symmetry(arr: np.ndarray, kind: str) -> bool:
    if kind == "lr":
        return np.array_equal(arr, np.fliplr(arr))
    if kind == "ud":
        return np.array_equal(arr, np.flipud(arr))
    if kind == "rot180":
        return np.array_equal(arr, np.rot90(arr, 2))
    raise ValueError(kind)


def _symmetry_kinds(arr: np.ndarray) -> tuple[str, ...]:
    return tuple(kind for kind in SYMMETRY_KINDS if _has_symmetry(arr, kind))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal repaint by reading an asymmetric blueprint against a symmetric cell scaffold.

    Input structure:
    - The board is split into equal 3x3 cell interiors by solid divider rows and columns.
    - Most cells share one repeated scaffold pattern whose 3x3 colors satisfy at least one
      internal symmetry.
    - A small subset of cells instead show the same asymmetric blueprint made from the
      scaffold background color plus one accent color.

    Rule:
    - Recover the repeated scaffold and identify its background color.
    - Read the blueprint's accent mask, then place that 3x3 mask uniquely on the 7x7 cell
      lattice so the revealed blueprint cells land on active mask positions.
    - For every active lattice position, rebuild the scaffold cell and recolor only those
      scaffold-background pixels selected by the blueprint mask with the accent color.

    Key insight:
    - The rare cell is no longer the final tile to stamp directly. It is a stencil: its
      accent pixels determine both the large-scale active cell positions and the small-scale
      pixels to repaint, while the repeated symmetric scaffold supplies the preserved
      foreground that survives the overlay.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    assert inner == MASK_SIZE, "this combo expects 3x3 cell interiors"

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(int(x) for x in pat_arr.ravel())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)

    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) == 2, "combo generator produces exactly scaffold + blueprint"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]
    minority_arr = patterns[minority_pat]

    maj_syms = _symmetry_kinds(majority_arr)
    assert maj_syms, "normal cell must carry an internal symmetry"
    assert not all(_has_symmetry(minority_arr, kind) for kind in maj_syms), "blueprint must break scaffold symmetry"

    maj_vals, maj_cnts = np.unique(majority_arr, return_counts=True)
    assert len(maj_vals) == 2, "scaffold should use 2 colors"
    background = int(maj_vals[maj_cnts.argmax()])
    majority_bg = majority_arr == background

    min_colors = [int(v) for v in np.unique(minority_arr)]
    assert background in min_colors and len(min_colors) == 2, "blueprint should share only the scaffold background"
    accent = next(color for color in min_colors if color != background)
    mask = minority_arr == accent
    assert np.any(mask), "blueprint needs active pixels"
    assert np.all(mask <= majority_bg), "overlay may only target scaffold background"
    assert any(not _has_symmetry(mask, kind) for kind in maj_syms), "active mask must visibly break scaffold symmetry"

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

    candidates: list[tuple[int, int]] = []
    for top in range(0, n - inner + 1):
        for left in range(0, n - inner + 1):
            block = np.zeros_like(existing)
            block[top : top + inner, left : left + inner] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + inner, left : left + inner]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "blueprint placement must be unique"
    top, left = candidates[0]

    active_cell = majority_arr.copy()
    active_cell[mask] = accent

    out = g.copy()
    for i in range(inner):
        for j in range(inner):
            target = active_cell if mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target
    return out


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _make_pattern(fill: int, coords: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    arr = np.full((MASK_SIZE, MASK_SIZE), fill, dtype=int)
    for r, c in coords:
        arr[r, c] = color
    return arr


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _precompute_scaffolds():
    scaffolds = []
    for kind, fg_coords in SCAFFOLD_FG.items():
        fg_mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
        for r, c in fg_coords:
            fg_mask[r, c] = True
        bg_mask = ~fg_mask
        bg_coords = [tuple(int(x) for x in rc) for rc in np.argwhere(bg_mask)]

        signal_masks: list[np.ndarray] = []
        for active_count in ACTIVE_PIXEL_COUNTS:
            for combo in combinations(bg_coords, active_count):
                mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
                for r, c in combo:
                    mask[r, c] = True
                if not _touches_all_edges(mask):
                    continue
                if _has_symmetry(mask, kind):
                    continue
                signal_masks.append(mask)

        scaffolds.append(
            {
                "kind": kind,
                "fg_coords": fg_coords,
                "fg_mask": fg_mask,
                "bg_mask": bg_mask,
                "signal_masks": signal_masks,
            }
        )
    return scaffolds


SCAFFOLDS = _precompute_scaffolds()


def _placement_candidates(mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True

    candidates: list[tuple[int, int]] = []
    for t in range(PLACEMENT_RANGE):
        for l in range(PLACEMENT_RANGE):
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[t : t + MASK_SIZE, l : l + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[t : t + MASK_SIZE, l : l + MASK_SIZE] & ~mask):
                continue
            candidates.append((t, l))
    return candidates


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...] | None:
    active = [tuple(int(x) for x in rc) for rc in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        if size >= len(active):
            continue
        valid = [
            subset
            for subset in combinations(active, size)
            if _placement_candidates(mask, subset, top, left) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _build_grid(separator: int, majority: np.ndarray, blueprint: np.ndarray, given_cells: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in given_cells:
        paste(grid, blueprint, *_cell_origin(i, j))
    return grid


def _build_output_grid(separator: int, majority: np.ndarray, active_cell: np.ndarray, active_cells: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, active_cell, *_cell_origin(i, j))
    return grid


def _gen_attempt() -> np.ndarray | None:
    separator, background, scaffold_fg, accent = random.sample(range(COLOR_POOL), 4)
    scaffold = random.choice(SCAFFOLDS)
    majority = _make_pattern(background, scaffold["fg_coords"], scaffold_fg)
    mask = random.choice(scaffold["signal_masks"]).copy()

    blueprint = np.full((MASK_SIZE, MASK_SIZE), background, dtype=int)
    blueprint[mask] = accent

    active_cell = majority.copy()
    active_cell[mask] = accent

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]

    inp = _build_grid(separator, majority, blueprint, given_cells)
    out = _build_output_grid(separator, majority, active_cell, active_cells)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    assert not np.array_equal(blueprint, active_cell)
    assert any(_has_symmetry(majority, kind) for kind in SYMMETRY_KINDS)
    assert not any(_has_symmetry(blueprint, kind) for kind in _symmetry_kinds(majority))
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo input")
