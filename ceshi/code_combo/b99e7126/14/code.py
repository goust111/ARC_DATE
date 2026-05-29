# Combo variant of b99e7126 (index 14)
# combo_spec: D5/3+D11/1
# Name: 颜色交换叠加 + 颜色继承效果

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# Design analysis for the D5/3 + D11/1 fused variant
# a) Concrete behavior and trace:
#    The 29x29 board is still a 7x7 lattice of 3x3 interiors split by 1-pixel divider lines.
#    The rare 3x3 cell still encodes a boolean mask: its dominant color marks active positions.
#    The target window is still a 3x3 block placed somewhere in the 7x7 lattice, and the input
#    still reveals only a strict subset of the active positions as minority anchor cells.
#    The fusion changes what happens after placement:
#    - Original task: every active enlarged slot is overwritten by one shared minority template.
#    - D11 alone: every active slot reveals the position-specific template that belongs there.
#    - This combo: every active slot reveals that position-specific template after locally swapping
#      its two colors. Example with shared=s and accent=a:
#        inherited template at absolute cell (i,j): [[s,s,s],[s,a,s],[s,s,s]]
#        fused output at the same active cell:      [[a,a,a],[a,s,a],[a,a,a]]
#      So the stencil comes from the odd cell, while the detailed colors come from the cell's own
#      inherited template field and are then inverted in place.
# b) Complexity calibration:
#    A pure inheritance rule would be weak if all normal cells were identical, and a pure color-swap
#    rule would be weak if the hidden target block contained only blank cells. I therefore keep the
#    D11-style repeated 3x3 position field outside the target block and keep the target block blank in
#    the input except for 3-4 minority anchors. This gives repeated evidence for the inherited source
#    templates and makes the swap visually obvious on multiple completed cells.
# c) Hidden-assumption audit against the original main():
#    - Original `majority_pat = max(...)` is invalid because normal cells are no longer a single
#      repeated template; there are 9 residue-class templates plus a blank filler and the minority.
#    - Original final write loop assumes every active slot receives the same minority pattern; this is
#      broken by D11 and remains broken in the combo.
#    - The grid parsing, minority extraction, mask derivation, and placement search still work because
#      they only rely on the minority anchors, which are preserved.
#    - The new final step assumes each inherited position template contains exactly two colors so the
#      local swap is well-defined; generator guarantees all non-blank non-minority templates use exactly
#      `{shared, accent}` and the blank template uses only `shared`.
# d) Placement legality:
#    The enlarged mask still occupies one 3x3 block inside the 7x7 cell grid, so `top,left` remain in
#    0..4. The generator still chooses only anchor subsets that force a unique placement under the same
#    consistency test as the original task. Inside the target window, every non-anchor cell is blank in
#    the input, so there are no overlaps between anchors and inherited completions.
# e) Visual observability:
#    A human sees three layers of evidence that align cleanly:
#    - the odd cell gives the 3x3 active/inactive stencil;
#    - the rest of the board shows the position-coded source templates that should be inherited;
#    - the output turns exactly the active inherited templates into their color-swapped versions.
#    This makes the combo distinct from both "copy the odd cell everywhere" and "restore the hidden
#    templates unchanged".
# f) Visual cleanliness:
#    The board uses only four colors total: divider, shared blank color, accent color, and minority
#    color. The target block is mostly blank in the input, so the completed overlay stays legible rather
#    than stacking multiple unrelated motifs in one region.
# g) Change inferability:
#    No step depends on numeric color ids. The inherited templates repeat by absolute row/col modulo 3,
#    and the swap is always a local exchange of the two colors present in the inherited template. Across
#    2-4 training pairs, solvers can see that active cells are chosen by the minority stencil but colored
#    by nearby repeated templates, then inverted.
# h) New solver assumptions and generator guarantees:
#    - Each residue class modulo 3 must have examples outside the target block; in a 7x7 board, removing
#      one contiguous 3x3 block still leaves at least 3 examples per class.
#    - Non-anchor cells in the target block must all share one blank template so the blank can be inferred
#      robustly from within-block frequencies; generator enforces that directly.
#    - Each inherited template must change under `_swap_two_colors`; the one-accent construction ensures
#      this and also keeps swapped outputs distinct from the minority pattern.
# Post-edit assumption audit:
#    - `_swap_two_colors()` is only applied to inherited position templates, never to the blank template.
#    - `minority_pat = min(counts...)` assumes the minority anchors are the rarest pattern; generator uses
#      3-4 anchors while each position template appears at least 4 times and blank appears 5-6 times.
#    - `blank_pat = _most_common(block_non_minority)` assumes every non-anchor cell in the target block is
#      blank; generator enforces that for both active-but-hidden and inactive positions.


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


def _most_common(items: list[tuple[int, ...]]) -> tuple[int, ...]:
    counts: dict[tuple[int, ...], int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    assert counts, "expected at least one pattern"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _swap_two_colors(arr: np.ndarray) -> np.ndarray:
    """Return a copy of arr with its two present colors swapped."""
    vals = [int(v) for v in np.unique(arr)]
    assert len(vals) == 2, "color-swap templates must contain exactly two colors"
    a, b = vals
    out = arr.copy()
    mask_a = out == a
    mask_b = out == b
    out[mask_a] = b
    out[mask_b] = a
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate the odd-cell stencil, inherit each active cell's hidden template, and swap that template's colors.

    Input structure:
    - The board is partitioned into equal 3x3 cell interiors by one divider color.
    - Outside one hidden 3x3 cell block, normal cells repeat a position-coded template field.
    - Inside that block, most cells are blank and only a few anchors share one rarer minority pattern.

    Rule:
    - Read the minority cell's dominant-color layout as the 3x3 active mask.
    - Place that mask uniquely on the 7x7 cell lattice so the visible minority anchors land on active sites.
    - Infer the template for each absolute cell position from matching row/col residue classes outside the block.
    - For every active site in the placed block, restore the inherited template and swap its two colors; inactive
      sites stay blank.

    Key insight:
    - The odd cell supplies only the enlarged stencil. The actual pixels of the finished overlay come from the
      repeated location-dependent templates already on the board, and the overlay acts on them like a local
      color-inversion filter.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

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

    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            p = cell_pat[i][j]
            counts[p] = counts.get(p, 0) + 1
    assert len(counts) >= 3, "need minority, blank, and inherited position templates"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

    candidates: list[tuple[int, int]] = []
    for top in range(0, n - mh + 1):
        for left in range(0, n - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique mask placement aligning existing minority cells"
    top, left = candidates[0]

    block_non_minority = [
        cell_pat[top + i][left + j]
        for i in range(mh)
        for j in range(mw)
        if cell_pat[top + i][left + j] != minority_pat
    ]
    blank_pat = _most_common(block_non_minority)

    residue_templates: dict[tuple[int, int], tuple[int, ...]] = {}
    for ri in range(mh):
        for cj in range(mw):
            samples: list[tuple[int, ...]] = []
            for i in range(n):
                for j in range(n):
                    if top <= i < top + mh and left <= j < left + mw:
                        continue
                    if i % mh != ri or j % mw != cj:
                        continue
                    p = cell_pat[i][j]
                    if p == minority_pat:
                        continue
                    samples.append(p)
            residue_templates[(ri, cj)] = _most_common(samples)

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            if mask[i, j]:
                base_pat = patterns[residue_templates[((top + i) % mh, (left + j) % mw)]]
                target_pat = _swap_two_colors(base_pat)
            else:
                target_pat = patterns[blank_pat]
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (combo D5/3 + D11/1) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 300

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_masks():
    """Enumerate all valid LR-symmetric 3x3 masks, keyed by active pixel count."""
    weights = [len(g) for g in _LR_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[i] for i in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for idx in combo:
                    for r, c in _LR_GROUPS[idx]:
                        mask[r, c] = False
                if mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_minority_mask() -> np.ndarray:
    """Pick a random valid minority mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _placement_candidates(mask: np.ndarray, subset, top: int, left: int) -> list[tuple[int, int]]:
    """Find all offsets where the chosen visible minority subset fits the mask."""
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


def _choose_given_subset(mask: np.ndarray, top: int, left: int):
    """Choose 3-4 active sites whose minority anchors uniquely determine the mask placement."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _placement_candidates(mask, subset, top, left) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of a cell interior inside the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _blank_pattern(shared: int) -> np.ndarray:
    return np.full((CELL_INNER, CELL_INNER), shared, dtype=int)


def _position_template(shared: int, accent: int, i: int, j: int) -> np.ndarray:
    arr = _blank_pattern(shared)
    arr[i % CELL_INNER, j % CELL_INNER] = accent
    return arr


def _minority_pattern(shared: int, minority_color: int, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, minority_color, shared).astype(int)


def _build_output_grid(separator: int, shared: int, accent: int, mask: np.ndarray, top: int, left: int) -> np.ndarray:
    """Outside the target block use normal templates; inside active slots use swapped inherited templates."""
    blank = _blank_pattern(shared)
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            if top <= i < top + MASK_SIZE and left <= j < left + MASK_SIZE:
                rel_i = i - top
                rel_j = j - left
                if mask[rel_i, rel_j]:
                    pat = _swap_two_colors(_position_template(shared, accent, i, j))
                else:
                    pat = blank
            else:
                pat = _position_template(shared, accent, i, j)
            paste(grid, pat, *_cell_origin(i, j))
    return grid


def _build_input_grid(
    separator: int,
    shared: int,
    accent: int,
    minority: np.ndarray,
    top: int,
    left: int,
    given_cells,
) -> np.ndarray:
    """Hide the target block behind blank cells, leaving only a few minority anchors visible."""
    blank = _blank_pattern(shared)
    given_set = set(given_cells)
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            if top <= i < top + MASK_SIZE and left <= j < left + MASK_SIZE:
                pat = minority if (i, j) in given_set else blank
            else:
                pat = _position_template(shared, accent, i, j)
            paste(grid, pat, *_cell_origin(i, j))
    return grid


def _gen_attempt():
    """Single attempt: build a fused output, derive the input, and verify with the solver."""
    separator, shared, accent, minority_color = random.sample(range(COLOR_POOL), 4)
    mask = _random_minority_mask()
    minority = _minority_pattern(shared, minority_color, mask)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None

    out = _build_output_grid(separator, shared, accent, mask, top, left)
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_input_grid(separator, shared, accent, minority, top, left, given_cells)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo D5/3+D11/1 input")
