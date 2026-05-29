# Combo variant of b99e7126 (index 21)
# combo_spec: D2/1+D11/2
# Name: 三色源图案 + 多色分形差异化处理

from functools import reduce
from itertools import combinations
from math import gcd
from pathlib import Path

import numpy as np
import random

try:
    from arc_utils import paste
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[4] / "arc2_solver_select"))
    from arc_utils import paste


# Design analysis for combo D2/1 + D11/2
# ---------------------------------------
# source_task:
# - The base task is the original b99e7126 fractal-completion puzzle.
# - D2/1 contributes a tricolor 3x3 source cell instead of a binary minority cell.
# - D11/2 contributes differentiated grid-scale actions for different source colors.
#
# a) Concrete behavior trace
# - Original task:
#   minority cell interior = binary 3x3 pattern.
#   Enlarged block behavior = active pixel -> copy minority cell, inactive pixel -> keep majority.
# - This combo:
#   source cell interior = tricolor 3x3 pattern with labels {MAIN, KEEP, ERASE}.
#   Enlarged block behavior now has three distinct outcomes:
#   1. MAIN color  -> paste another copy of the full tricolor source cell
#   2. KEEP color  -> leave the majority cell unchanged
#   3. ERASE color -> replace the cell with a solid erase cell
# - Small trace:
#   source row      = [MAIN, KEEP, ERASE]
#   original binary = [minority, majority, majority]
#   combo result    = [source,   majority, erase   ]
#   Difference:
#   the third source color is no longer collapsed into "inactive"; it produces its own
#   large-scale action and therefore changes the completed block visibly.
#
# b) Complexity calibration
# - The spatial skeleton stays fixed: a 7x7 cell lattice with 3x3 interiors and 1-pixel dividers.
# - Only three cell-scale outcomes exist inside the enlarged 3x3 block, which is enough to make
#   the fusion visible without introducing too many unrelated object classes.
# - The erase behavior is represented by one clean solid-color cell, so "erase" is visually
#   distinguishable from "keep" even though both are non-source outcomes.
#
# c) Hidden assumption audit against the original main()
# - Original assumption: the least frequent cell pattern is the single minority pattern.
#   Broken here because the finished block contains both source copies and erase cells.
#   New solver instead locates the unique 3-color cell as the source legend.
# - Original assumption: the source cell can be binarized via one active color.
#   Broken here because three source colors encode three different actions.
# - Original assumption: every non-active source pixel maps to the same majority result.
#   Broken here because one non-main color means keep while another means erase.
# - Original assumption: placement consistency only needs a boolean mask.
#   Broken here because observed non-majority cells belong to different source-color classes.
#   New placement logic matches absolute observed cells against exact source colors.
#
# d) Placement legality
# - The enlarged legend still occupies exactly one 3x3 cell block inside the 7x7 lattice, so
#   offsets remain in 0..4.
# - The generator only reveals a strict subset of non-majority positions and brute-forces that the
#   labeled subset determines a unique offset.
# - No overlaps occur because all writes happen cell-by-cell on the fixed lattice.
#
# e) Visual observability
# - A human can read the tricolor source cell as a legend:
#   one color repeats as more tricolor source copies, one color produces solid erase cells,
#   and the third color corresponds to untouched majority cells.
# - No color-id arithmetic is used. The only cues are repeated shapes and repeated color roles.
#
# f) Clean visualization
# - The final 3x3 block contains only majority cells, tricolor source copies, and solid erase
#   cells. These three patterns are visually distinct and do not stack on top of each other.
# - Majority patterns are chosen from a small asymmetric catalog to stay readable and avoid
#   accidental equality with the erase cell.
#
# g) Color-change inferability
# - Total palette size is 5: divider, majority background, shared keep color, source-main color,
#   and erase color.
# - The erase color appears both inside the source legend and in the completed erase cells, giving
#   the solver a visible structural hint instead of an arbitrary recolor.
#
# h) Quantitative non-degeneration target
# - Every sample must contain at least two ERASE positions and at least three MAIN positions in the
#   source legend.
# - Therefore the combo output differs from the binary-collapse baseline on at least two cell slots
#   inside the enlarged block.
#
# i) New solver assumptions and generator guarantees
# - Solver assumes exactly one distinct 3-color source pattern exists; generator guarantees that.
# - Solver assumes the input reveals both non-majority behaviors so the remaining unseen source
#   color can be assigned to KEEP/majority; generator enforces both labels in the revealed subset.
# - Solver assumes the erase pattern differs from the majority pattern; generator uses a solid
#   erase cell with a dedicated color, so this is always true.

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
PLACEMENT_RANGE = CELL_GRID_N - CELL_INNER + 1
GIVEN_CELL_SIZES = (4, 5)
COLOR_POOL = 10
MAX_RETRIES = 400

LABEL_MAIN = 0
LABEL_KEEP = 1
LABEL_ERASE = 2


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


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in arr.ravel())


def _extract_cells(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
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
            pat = _pattern_key(pat_arr)
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return patterns, cell_pat


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a tricolor source cell into a cell-grid fractal with copy, erase, and keep behaviors.

    Input structure:
    - The board is split into equal 3x3 cell interiors separated by one-color divider lines.
    - Most cells share one repeated majority pattern.
    - Inside one hidden 3x3 cell block, a few visible cells are replaced by non-majority patterns:
      copies of a unique tricolor source cell and copies of a solid erase cell.

    Rule:
    - Read the unique tricolor cell as a 3x3 legend whose three colors stand for three cell-scale
      actions: copy the tricolor source, erase to the solid erase cell, or keep the majority cell.
    - Use the visible non-majority cells to find the only placement of that 3x3 legend inside the
      7x7 cell lattice and to infer which observed pattern belongs to which source color.
    - Complete all 9 positions of the block according to that color-to-action mapping.

    Key insight:
    - The odd cell is not a binary mask anymore. Its three colors are three symbols controlling
      different large-scale outcomes, so the fractal overlay mixes reproduction, erasure, and
      unchanged majority cells within the same enlarged 3x3 block.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_pat = _extract_cells(g, step)

    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            p = cell_pat[i][j]
            counts[p] = counts.get(p, 0) + 1

    assert len(counts) >= 3, "variant expects majority, source, and erase cell patterns"
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]

    source_candidates = [pat for pat, arr in patterns.items() if len(np.unique(arr)) == 3]
    assert len(source_candidates) == 1, "must have exactly one distinct tricolor source pattern"
    source_pat = source_candidates[0]
    source_arr = patterns[source_pat]
    sh, sw = source_arr.shape
    assert sh == inner and sw == inner, "source cell must match cell interior size"

    existing: list[tuple[int, int, tuple[int, ...]]] = []
    for i in range(n):
        for j in range(n):
            pat = cell_pat[i][j]
            if pat != majority_pat:
                existing.append((i, j, pat))
    assert existing, "input must reveal at least one non-majority cell"

    source_colors = {int(v) for v in np.unique(source_arr)}
    assert len(source_colors) == 3, "source cell must contain exactly three colors"

    candidates: list[tuple[int, int, dict[int, tuple[int, ...]]]] = []
    for top in range(0, n - sh + 1):
        for left in range(0, n - sw + 1):
            color_to_pattern: dict[int, tuple[int, ...]] = {}
            valid = True
            for i, j, pat in existing:
                if not (top <= i < top + sh and left <= j < left + sw):
                    valid = False
                    break
                color = int(source_arr[i - top, j - left])
                prev = color_to_pattern.get(color)
                if prev is None:
                    color_to_pattern[color] = pat
                elif prev != pat:
                    valid = False
                    break
            if not valid:
                continue
            if len(set(color_to_pattern.values())) != len(color_to_pattern):
                continue
            remaining_colors = [color for color in source_colors if color not in color_to_pattern]
            if len(remaining_colors) != 1:
                continue
            full_mapping = dict(color_to_pattern)
            full_mapping[remaining_colors[0]] = majority_pat
            if len(set(full_mapping.values())) != 3:
                continue
            candidates.append((int(top), int(left), full_mapping))

    assert len(candidates) == 1, "must have a unique tricolor placement and mapping"
    top, left, color_to_pattern = candidates[0]

    out = g.copy()
    for i in range(sh):
        for j in range(sw):
            pat = patterns[color_to_pattern[int(source_arr[i, j])]]
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = pat
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = [tuple(coord) for coord in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in seen:
                continue
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _precompute_majority_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (CELL_INNER * CELL_INNER)):
        mask = np.zeros((CELL_INNER, CELL_INNER), dtype=bool)
        for idx in range(CELL_INNER * CELL_INNER):
            if bits & (1 << idx):
                mask[idx // CELL_INNER, idx % CELL_INNER] = True
        count = int(mask.sum())
        if count not in (4, 5):
            continue
        if not _is_connected(mask):
            continue
        if any(not mask[r, :].any() for r in range(CELL_INNER)):
            continue
        if any(not mask[:, c].any() for c in range(CELL_INNER)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask)):
            continue
        masks.append(mask)
    assert masks, "need majority mask catalog"
    return masks


def _precompute_source_layouts() -> list[np.ndarray]:
    layouts: list[np.ndarray] = []
    total = 3 ** (CELL_INNER * CELL_INNER)
    for bits in range(total):
        x = bits
        arr = np.empty((CELL_INNER, CELL_INNER), dtype=int)
        counts = [0, 0, 0]
        for idx in range(CELL_INNER * CELL_INNER):
            label = x % 3
            x //= 3
            arr[idx // CELL_INNER, idx % CELL_INNER] = label
            counts[label] += 1
        if counts[LABEL_MAIN] not in (3, 4):
            continue
        if counts[LABEL_KEEP] not in (2, 3, 4):
            continue
        if counts[LABEL_ERASE] not in (2, 3):
            continue
        if np.array_equal(arr, np.fliplr(arr)):
            continue
        if np.array_equal(arr, np.flipud(arr)):
            continue
        if np.array_equal(arr, np.rot90(arr)):
            continue
        if len({tuple(arr[r, :].tolist()) for r in range(CELL_INNER)}) < 3:
            continue
        if len({tuple(arr[:, c].tolist()) for c in range(CELL_INNER)}) < 2:
            continue
        layouts.append(arr)
    assert layouts, "need source layout catalog"
    return layouts


_MAJORITY_MASKS = _precompute_majority_masks()
_SOURCE_LAYOUTS = _precompute_source_layouts()


def _random_majority_pattern(shared_color: int, majority_bg: int) -> np.ndarray:
    mask = random.choice(_MAJORITY_MASKS)
    return np.where(mask, shared_color, majority_bg).astype(int)


def _random_source_pattern(main_color: int, shared_color: int, erase_color: int) -> np.ndarray:
    layout = random.choice(_SOURCE_LAYOUTS)
    palette = np.array([main_color, shared_color, erase_color], dtype=int)
    return palette[layout]


def _erase_pattern(erase_color: int) -> np.ndarray:
    return np.full((CELL_INNER, CELL_INNER), erase_color, dtype=int)


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, cells: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j, pat in cells:
        paste(grid, pat, *_cell_origin(i, j))
    return grid


def _placement_candidates(layout: np.ndarray, observed: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            ok = True
            for abs_r, abs_c, label in observed:
                if not (top <= abs_r < top + CELL_INNER and left <= abs_c < left + CELL_INNER):
                    ok = False
                    break
                if int(layout[abs_r - top, abs_c - left]) != label:
                    ok = False
                    break
            if ok:
                candidates.append((top, left))
    return candidates


def _choose_given_subset(layout: np.ndarray, top: int, left: int) -> list[tuple[int, int, int]] | None:
    informative = [
        (top + r, left + c, int(layout[r, c]))
        for r in range(CELL_INNER)
        for c in range(CELL_INNER)
        if int(layout[r, c]) != LABEL_KEEP
    ]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid: list[list[tuple[int, int, int]]] = []
        for subset in combinations(informative, size):
            labels = {label for _, _, label in subset}
            if labels != {LABEL_MAIN, LABEL_ERASE}:
                continue
            if _placement_candidates(layout, list(subset)) == [(top, left)]:
                valid.append(list(subset))
        if valid:
            return random.choice(valid)
    return None


def _collapsed_binary_baseline(
    separator: int,
    majority: np.ndarray,
    source: np.ndarray,
    layout: np.ndarray,
    top: int,
    left: int,
) -> np.ndarray:
    cells: list[tuple[int, int, np.ndarray]] = []
    for r in range(CELL_INNER):
        for c in range(CELL_INNER):
            if int(layout[r, c]) != LABEL_KEEP:
                cells.append((top + r, left + c, source))
    return _build_grid(separator, majority, cells)


def _count_changed_cells(out: np.ndarray, ref: np.ndarray) -> int:
    changed = 0
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            r0, c0 = _cell_origin(i, j)
            if not np.array_equal(
                out[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER],
                ref[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER],
            ):
                changed += 1
    return changed


def _gen_attempt() -> np.ndarray | None:
    separator, majority_bg, shared, source_main, erase_color = random.sample(range(COLOR_POOL), 5)
    majority = _random_majority_pattern(shared, majority_bg)
    source = _random_source_pattern(source_main, shared, erase_color)
    erase_cell = _erase_pattern(erase_color)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)

    layout = np.empty((CELL_INNER, CELL_INNER), dtype=int)
    full_cells: list[tuple[int, int, np.ndarray]] = []
    for r in range(CELL_INNER):
        for c in range(CELL_INNER):
            color = int(source[r, c])
            if color == source_main:
                label = LABEL_MAIN
                pat = source
            elif color == shared:
                label = LABEL_KEEP
                pat = majority
            else:
                label = LABEL_ERASE
                pat = erase_cell
            layout[r, c] = label
            if not np.array_equal(pat, majority):
                full_cells.append((top + r, left + c, pat))

    if int(np.sum(layout == LABEL_MAIN)) < 3 or int(np.sum(layout == LABEL_ERASE)) < 2:
        return None

    out = _build_grid(separator, majority, full_cells)
    baseline = _collapsed_binary_baseline(separator, majority, source, layout, top, left)
    if _count_changed_cells(out, baseline) < 2:
        return None

    subset = _choose_given_subset(layout, top, left)
    if subset is None:
        return None

    given_cells = [
        (r, c, source if label == LABEL_MAIN else erase_cell)
        for r, c, label in subset
    ]
    inp = _build_grid(separator, majority, given_cells)

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
