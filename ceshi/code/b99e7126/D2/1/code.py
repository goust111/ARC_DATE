# Variant of b99e7126
# Dimension: D2 — 源对象：异常单元格的内部图案结构 (src)
# Change: 唯一异常的3x3单元格的内部颜色像素图案，包含两种颜色（前景色和背景色）... → 异常单元格内图案使用三种颜色

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# Design analysis for this D2 variant
# -----------------------------------
# source_task:
# - None. No extra reference solver needs to be consulted beyond the original b99e7126 code.
#
# a) Concrete behavior trace
# - The original task treats the special 3x3 cell as a binary mask: one color means "copy the
#   minority cell", the other means "leave the majority cell". That relies on the source cell
#   having exactly two colors and therefore exactly two grid-scale behaviors.
# - This variant replaces that source with a true three-color 3x3 layout. The three colors now
#   encode three different grid-scale outcomes inside the placed 3x3 block:
#   1. source-main color -> paste another copy of the tricolor source cell
#   2. shared color -> keep the majority cell
#   3. accent color -> paste an auxiliary recolored cell that shares the majority cell's shape
# - Small trace:
#   source cell colors = {main, shared, accent}
#   local source row   = [main, shared, accent]
#   original logic     = [minority, majority, majority] after collapsing everything non-active
#   new logic          = [source,   majority, accent ]
#   difference         = the third color is no longer silently merged into "inactive"; it has its
#                        own repeated cell type at grid scale.
#
# b) Complexity calibration
# - The grid skeleton stays unchanged: 7x7 cell grid, cell interior 3x3, dividers 1 pixel wide.
# - Only one extra global cell pattern is introduced beyond the original pair:
#   majority, source, accent. That is enough to make the three-color source visually meaningful
#   without exploding the hypothesis space.
# - The accent cell is structurally tied to the majority cell: same binary stencil, different
#   foreground color. This makes the mapping human-readable from 2-4 examples.
#
# c) Hidden assumption audit against the original main()
# - Original assumption: "minority pattern = least frequent pattern". This is unsafe now because
#   the finished structure can contain two non-majority pattern types. New solver instead finds
#   the unique cell pattern that uses exactly three colors and uses that as the source pattern.
# - Original assumption: "most frequent color inside minority_arr is the active mask". Unsafe now
#   because two non-majority behaviors exist. New solver never binarizes the source cell.
# - Original assumption: "non-active pixels all map to majority". Unsafe now because accent pixels
#   must map to a third pattern. New solver infers a color -> cell-pattern mapping from observed
#   non-majority cells and leaves exactly one source color unmapped; that remaining color becomes
#   the majority behavior.
# - Original assumption: "placement check only needs a boolean mask". Unsafe now because positions
#   with different source colors must remain distinguishable. New solver checks consistency of
#   absolute observed cells against exact source-cell colors, not just inside/outside membership.
#
# d) Placement legality
# - The placed fractal block is still exactly 3x3 cells and must lie fully inside the 7x7 grid,
#   so offsets remain in 0..4.
# - Generator only reveals a strict subset of the non-majority positions, but it requires that the
#   subset contain both source copies and accent copies, and that this colored subset determine a
#   unique offset under the solver's matching rule.
# - No overlaps occur because every output cell occupies one cell slot in the fixed lattice.
#
# e) Visual observability
# - The rule is visible by eye: the tricolor source cell acts as a legend for the enlarged 3x3
#   block. Shared-color pixels correspond to unchanged majority cells. Accent-color pixels become
#   recolored majority-shape cells. Main-color pixels become copies of the full tricolor source.
# - No color-id ordering or hidden numeric relation is used; only visible repetition and color
#   correspondence matter.
#
# f) Clean visualization
# - There are only three cell types in the final block, and the accent cell is deliberately made
#   shape-compatible with the majority cell. This keeps the output organized instead of noisy.
# - The source layout and majority stencil are chosen from small filtered catalogs to avoid overly
#   symmetric or degenerate cases.
#
# g) Color-change inferability
# - Total palette size rises from 4 to 5, which is still compact.
# - The extra color is not arbitrary: it appears inside the source cell and reappears as the
#   foreground color of the accent cell. That repeated structure is the cue needed to infer the
#   mapping from a few examples.
#
# h) Quantitative non-degeneration target
# - Every generated sample must contain at least two accent positions in the source 3x3 layout.
# - Therefore the new output always differs from the original binary-collapse behavior on at least
#   two cell positions inside the enlarged 3x3 block.
#
# i) New solver assumptions and generator guarantees
# - Solver assumes exactly one distinct 3-color cell pattern exists; generator guarantees this by
#   using only one tricolor cell type.
# - Solver assumes the revealed subset contains both non-majority behaviors so one source color is
#   left unmatched and can be assigned to the majority behavior; generator enforces this.
# - Solver assumes the colored subset yields a unique placement; generator brute-forces subsets and
#   rejects ambiguous ones.

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
PLACEMENT_RANGE = CELL_GRID_N - CELL_INNER + 1
GIVEN_CELL_SIZES = (4, 5)
COLOR_POOL = 10
MAX_RETRIES = 400

LABEL_MAIN = 0
LABEL_SHARED = 1
LABEL_ACCENT = 2


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
    Enlarge a tricolor source cell into a 3x3 block of cell-sized copies and recolors.

    Input structure:
    - The grid is partitioned into equal 3x3 interiors separated by single-color divider lines.
    - Most cells share one majority pattern.
    - Inside one hidden 3x3 cell block, a few cells are replaced by non-majority patterns.
      One of those patterns is a unique tricolor source cell; another is a recolored auxiliary cell.

    Rule:
    - Read the tricolor source cell as a 3x3 legend.
    - Each source color maps to a different cell-scale behavior:
      one color reproduces the tricolor source cell, one color keeps the majority cell,
      and one color places the auxiliary recolored cell.
    - Use the already revealed non-majority cells to find the unique placement of that 3x3 legend
      within the 7x7 cell lattice, then complete all 9 positions of the block accordingly.

    Key insight:
    - The special cell is no longer a binary mask. It is a three-symbol template whose colors act
      as labels for three distinct repeated cell patterns at the larger scale.
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

    assert len(counts) >= 3, "variant expects majority, source, and accent cell patterns"
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
    for bits in range(3 ** (CELL_INNER * CELL_INNER)):
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
        if counts[LABEL_SHARED] not in (2, 3, 4):
            continue
        if counts[LABEL_ACCENT] not in (2, 3):
            continue
        if counts[LABEL_SHARED] + counts[LABEL_MAIN] + counts[LABEL_ACCENT] != 9:
            continue
        if np.array_equal(arr, np.fliplr(arr)):
            continue
        if np.array_equal(arr, np.flipud(arr)):
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


def _random_source_pattern(main_color: int, shared_color: int, accent_color: int) -> np.ndarray:
    layout = random.choice(_SOURCE_LAYOUTS)
    palette = np.array([main_color, shared_color, accent_color], dtype=int)
    return palette[layout]


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
        if int(layout[r, c]) != LABEL_SHARED
    ]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid: list[list[tuple[int, int, int]]] = []
        for subset in combinations(informative, size):
            labels = {label for _, _, label in subset}
            if labels != {LABEL_MAIN, LABEL_ACCENT}:
                continue
            if _placement_candidates(layout, list(subset)) == [(top, left)]:
                valid.append(list(subset))
        if valid:
            return random.choice(valid)
    return None


def _gen_attempt() -> np.ndarray | None:
    separator, majority_bg, shared, source_main, accent = random.sample(range(COLOR_POOL), 5)
    majority = _random_majority_pattern(shared, majority_bg)
    accent_cell = np.where(majority == shared, accent, majority_bg).astype(int)
    source = _random_source_pattern(source_main, shared, accent)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)

    label_to_pattern = {
        source_main: source,
        shared: majority,
        accent: accent_cell,
    }

    full_cells: list[tuple[int, int, np.ndarray]] = []
    layout = np.empty((CELL_INNER, CELL_INNER), dtype=int)
    for r in range(CELL_INNER):
        for c in range(CELL_INNER):
            color = int(source[r, c])
            layout[r, c] = LABEL_MAIN if color == source_main else LABEL_SHARED if color == shared else LABEL_ACCENT
            pat = label_to_pattern[color]
            if not np.array_equal(pat, majority):
                full_cells.append((top + r, left + c, pat))

    source_counts = {
        label: int(np.sum(layout == label))
        for label in (LABEL_MAIN, LABEL_SHARED, LABEL_ACCENT)
    }
    if source_counts[LABEL_ACCENT] < 2 or source_counts[LABEL_MAIN] < 3:
        return None

    out = _build_grid(separator, majority, full_cells)

    subset = _choose_given_subset(layout, top, left)
    if subset is None:
        return None

    given_cells = [(r, c, source if label == LABEL_MAIN else accent_cell) for r, c, label in subset]
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
    raise RuntimeError("failed to generate a valid b99e7126 D2 variant input")
