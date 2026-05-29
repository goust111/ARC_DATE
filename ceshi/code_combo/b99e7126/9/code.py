# Combo variant of b99e7126 (index 9)
# combo_spec: D1/4+D2/1
# Name: 形状匹配识别 + 三色源图案
#
# Design analysis
# ---------------
# a) Concrete behavior trace
# - Original task:
#   1. Count exact 3x3 cell bitmaps.
#   2. The least frequent exact bitmap is the source cell.
#   3. Collapse that source cell to a binary mask and enlarge it:
#      active -> paste minority bitmap, inactive -> paste one majority bitmap.
# - Fused task:
#   1. Each cell is reduced to the mask of its unique dominant connected color.
#   2. Masks are canonicalized up to rotation/flip, so many visually different majority tiles
#      merge into one shape class while the rare source class stays separate.
#   3. After identifying the source class geometrically, one source exemplar is read as a true
#      tricolor legend:
#      dominant-color pixels -> paste the source cell itself,
#      accent-color pixels -> paste a fixed auxiliary cell,
#      keep-color pixels -> leave the underlying majority variant untouched.
# - Small trace:
#   1. Input may contain 40+ majority cells whose exact pixels are all different but whose
#      dominant-shape signature is the same.
#   2. Two source cells share another dominant-shape signature.
#   3. Two accent cells repeat one exact majority-shape bitmap.
#   4. Exact-pattern counting would fail because many majority variants are singletons; shape
#      counting succeeds, and the third source color changes the enlarged 3x3 block at two
#      positions that a binary solver would leave unchanged.
#
# b) Complexity calibration
# - The macro lattice stays at 7x7 cells with 3x3 interiors, so the puzzle still reads like the
#   original family.
# - Only two geometric shape classes exist at cell level: majority-class and source-class. This
#   keeps the abstraction burden manageable.
# - Exact majority bitmaps are deliberately diverse and mostly unique, so the D1/4 change is
#   visually meaningful instead of cosmetic.
# - The source legend still has only three cell-scale behaviors: source, accent, keep. This keeps
#   the D2/1 change readable from a few examples.
#
# c) Hidden assumption audit against the original main()
# - "Minority = least frequent exact bitmap" no longer holds, because many majority variants are
#   singletons. The new solver counts canonical dominant-shape signatures instead.
# - "The source cell is binary" no longer holds. The source cell is tricolor, so the new solver
#   preserves all three colors and uses them as macro labels.
# - "Inactive source pixels all map to one majority bitmap" no longer holds, because majority
#   cells vary across the grid. The keep-color now means "do not overwrite this cell".
# - "The rare exact pattern can be used as the only clue" no longer holds. The new solver also
#   needs the repeated accent bitmap to decide which non-dominant source color means recolor.
#
# d) Placement legality
# - The enlarged legend still occupies one 3x3 block of cells, so offsets remain in 0..4.
# - Generator builds a fully valid base grid first, then overlays the full block for output and a
#   strict informative subset for input. No overlaps or out-of-bounds writes can occur.
# - Accent cells share the majority shape class, so the shape-class layer still has exactly two
#   classes in every sample.
#
# e) Visual observability
# - A human can see that the rare cells are defined by a different dominant shape, not by exact
#   pixel equality: the background decorations among the common cells vary heavily.
# - The third source color is observable because it reappears as the dominant color of the repeated
#   accent cells elsewhere in the macro block.
# - The keep-color is also visible inside many majority tiles, so it reads naturally as the "leave
#   the base alone" label rather than a hidden numeric convention.
#
# f) Clean visualization
# - There is only one overlay block, one repeated source tile, and one repeated accent tile.
# - Majority variation is confined inside each 3x3 cell; macro-scale structure remains a clean 3x3
#   block with three readable behaviors.
#
# g) Color-change inferability
# - Total palette size is 5: divider, shared dominant shape color, keep color, extra background
#   color, accent dominant color.
# - The accent color is not arbitrary: it appears inside the tricolor source tile and as the
#   dominant color of the repeated accent cells. That structural echo is the clue.
#
# h) Quantitative non-degeneration target
# - At least 3 accent positions exist in every enlarged 3x3 block.
# - In every generated input, at least 2 source cells and exactly 2 accent cells are visible, so
#   at least 1 accent completion is always hidden.
# - Majority exact bitmaps are sampled without replacement, so the repeated accent bitmap remains
#   the only repeated majority-shape exact pattern.
#
# i) New solver assumptions and generator guarantees
# - Solver assumes every cell has a unique most frequent color forming one 4-connected component;
#   generator enforces that for source, majority, and accent tiles.
# - Solver assumes exactly two canonical shape classes; generator uses one source mask and one
#   majority mask, with accent sharing the majority mask.
# - Solver assumes exactly one repeated non-source exact bitmap exists; generator makes all exposed
#   majority variants unique and reveals both accent cells.
# - Solver assumes the observed source+accent subset uniquely determines the macro offset; generator
#   brute-forces subsets and rejects ambiguous cases.

from functools import reduce
from itertools import combinations, product
from math import gcd

import numpy as np
import random

from arc_utils import paste

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
PLACEMENT_RANGE = CELL_GRID_N - CELL_INNER + 1
GIVEN_CELL_SIZES = (4, 5)
COLOR_POOL = 10
MAX_RETRIES = 500

LABEL_SOURCE = 0
LABEL_KEEP = 1
LABEL_ACCENT = 2

SOURCE_MASK = np.array(
    [
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ],
    dtype=bool,
)

MAJORITY_MASK = np.array(
    [
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
    ],
    dtype=bool,
)


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


def _symmetry_variants(mask: np.ndarray) -> list[np.ndarray]:
    """Unique rotations/reflections of a 2D boolean mask."""
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            key = tuple(int(v) for v in cand.ravel())
            if key not in seen:
                seen.add(key)
                variants.append(cand.copy())
    return variants


def _canonical_shape(mask: np.ndarray) -> tuple[int, ...]:
    """Canonical D4-invariant signature for a 3x3 shape."""
    return min(tuple(int(v) for v in cand.ravel()) for cand in _symmetry_variants(mask))


def _is_connected(mask: np.ndarray) -> bool:
    """4-connectivity for a boolean mask."""
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
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _dominant_shape(arr: np.ndarray) -> tuple[int, np.ndarray]:
    """Extract the unique most frequent color and its connected mask."""
    vals, cnts = np.unique(arr, return_counts=True)
    order = np.argsort(-cnts)
    assert len(order) >= 2 and cnts[order[0]] > cnts[order[1]], "dominant color must be unique"
    color = int(vals[order[0]])
    mask = arr == color
    assert _is_connected(mask), "dominant color must form one connected shape"
    return color, mask


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in arr.ravel())


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _extract_cells(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[np.ndarray]], list[list[tuple[int, ...]]], list[list[tuple[int, ...]]]]:
    """Return unique pattern bank, raw cell arrays, exact keys, and canonical shape signatures."""
    n = (g.shape[0] - 1) // step
    inner = step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_arrs: list[list[np.ndarray]] = []
    cell_keys: list[list[tuple[int, ...]]] = []
    shape_sigs: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row_arrs: list[np.ndarray] = []
        row_keys: list[tuple[int, ...]] = []
        row_sigs: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            key = _pattern_key(arr)
            patterns.setdefault(key, arr)
            _, mask = _dominant_shape(arr)
            row_arrs.append(arr)
            row_keys.append(key)
            row_sigs.append(_canonical_shape(mask))
        cell_arrs.append(row_arrs)
        cell_keys.append(row_keys)
        shape_sigs.append(row_sigs)
    return patterns, cell_arrs, cell_keys, shape_sigs


assert _canonical_shape(SOURCE_MASK) != _canonical_shape(MAJORITY_MASK)
MAJORITY_MASK_VARIANTS = _symmetry_variants(MAJORITY_MASK)


def _precompute_source_layouts() -> list[np.ndarray]:
    """Enumerate tricolor source legends over the fixed source dominant mask."""
    gaps = [tuple(int(v) for v in rc) for rc in np.argwhere(~SOURCE_MASK)]
    layouts: list[np.ndarray] = []
    for accent_gap_idxs in combinations(range(len(gaps)), 3):
        arr = np.full((CELL_INNER, CELL_INNER), LABEL_SOURCE, dtype=int)
        accent_idx_set = set(accent_gap_idxs)
        for idx, (r, c) in enumerate(gaps):
            arr[r, c] = LABEL_ACCENT if idx in accent_idx_set else LABEL_KEEP
        layouts.append(arr)
    assert layouts, "need source layout catalog"
    return layouts


SOURCE_LAYOUTS = _precompute_source_layouts()


def _build_source_pattern(shape_color: int, keep_color: int, accent_color: int) -> np.ndarray:
    """A tricolor source tile: source dominant shape plus keep/accent legend colors."""
    layout = random.choice(SOURCE_LAYOUTS)
    palette = np.array([shape_color, keep_color, accent_color], dtype=int)
    return palette[layout]


def _build_accent_pattern(accent_color: int, keep_color: int, bg_color: int) -> np.ndarray:
    """A fixed repeated accent tile sharing the majority shape class."""
    arr = np.full((CELL_INNER, CELL_INNER), accent_color, dtype=int)
    gaps = [tuple(int(v) for v in rc) for rc in np.argwhere(~MAJORITY_MASK)]
    for idx, (r, c) in enumerate(gaps):
        arr[r, c] = keep_color if idx % 2 == 0 else bg_color
    return arr


def _majority_variants(shape_color: int, keep_color: int, bg_color: int) -> list[np.ndarray]:
    """Many exact tricolor majority bitmaps sharing one canonical shape class."""
    variants = []
    seen = set()
    fills = (keep_color, bg_color)
    for mask in MAJORITY_MASK_VARIANTS:
        gaps = [tuple(int(v) for v in rc) for rc in np.argwhere(~mask)]
        for assignment in product(fills, repeat=len(gaps)):
            if len(set(assignment)) == 1:
                continue
            arr = np.full((CELL_INNER, CELL_INNER), shape_color, dtype=int)
            for (r, c), color in zip(gaps, assignment):
                arr[r, c] = int(color)
            key = _pattern_key(arr)
            if key not in seen:
                seen.add(key)
                variants.append(arr)
    assert variants, "need majority variants"
    return variants


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal overlay by spotting the rare cell *shape class* and then reading its tricolor legend.

    Input structure:
    - The grid is a 7x7 lattice of 3x3 cell interiors separated by one-color divider lines.
    - Every cell has one dominant connected shape color plus two background/decorative colors.
    - Most cells belong to one shape class, but many of those majority cells differ pixel-for-pixel.
    - A few source cells share another shape class, and that source tile itself uses three colors.

    Rule:
    - Canonicalize each cell by the geometry of its dominant connected color, ignoring rotation/flip and decoration.
    - The rarer canonical shape class identifies the source tile; keep one exact tricolor source exemplar.
    - Read the source exemplar as a 3x3 legend:
      its dominant color means paste another source tile,
      one non-dominant color means paste the repeated accent tile,
      the remaining color means leave the existing majority tile unchanged.
    - Use the already shown source cells and accent cells to find the unique 3x3 block placement, then finish the block.

    Key insight: The anomaly is defined first by geometry, not by exact pixels; only after that abstraction do the three colors inside the source tile become a macro-scale legend for source / accent / keep behaviors.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_arrs, cell_keys, shape_sigs = _extract_cells(g, step)

    shape_counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            sig = shape_sigs[i][j]
            shape_counts[sig] = shape_counts.get(sig, 0) + 1
    assert len(shape_counts) == 2, "expected exactly two shape classes"

    source_sig = min(shape_counts.items(), key=lambda kv: kv[1])[0]
    majority_sig = max(shape_counts.items(), key=lambda kv: kv[1])[0]

    source_arr = None
    source_key = None
    source_positions: list[tuple[int, int]] = []
    exact_counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            key = cell_keys[i][j]
            exact_counts[key] = exact_counts.get(key, 0) + 1
            if shape_sigs[i][j] == source_sig:
                source_positions.append((i, j))
                if source_arr is None:
                    source_arr = cell_arrs[i][j]
                    source_key = key
    assert source_arr is not None and source_key is not None, "need at least one source exemplar"

    source_main_color, _ = _dominant_shape(source_arr)
    source_colors = {int(v) for v in np.unique(source_arr)}
    assert len(source_colors) == 3, "source cell must be tricolor"

    accent_candidates = [
        key
        for key, count in exact_counts.items()
        if count > 1
        and key != source_key
        and _canonical_shape(_dominant_shape(patterns[key])[1]) == majority_sig
    ]
    assert len(accent_candidates) == 1, "need one repeated non-source majority-shape accent pattern"
    accent_key = accent_candidates[0]
    accent_arr = patterns[accent_key]

    accent_positions: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if cell_keys[i][j] == accent_key:
                accent_positions.append((i, j))
    assert accent_positions, "input must reveal accent cells"

    candidates: list[tuple[int, int, int]] = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            valid = True
            for i, j in source_positions:
                if not (top <= i < top + CELL_INNER and left <= j < left + CELL_INNER):
                    valid = False
                    break
                if int(source_arr[i - top, j - left]) != source_main_color:
                    valid = False
                    break
            if not valid:
                continue

            accent_color = None
            for i, j in accent_positions:
                if not (top <= i < top + CELL_INNER and left <= j < left + CELL_INNER):
                    valid = False
                    break
                color = int(source_arr[i - top, j - left])
                if color == source_main_color:
                    valid = False
                    break
                if accent_color is None:
                    accent_color = color
                elif accent_color != color:
                    valid = False
                    break
            if not valid or accent_color is None:
                continue

            if int(np.sum(source_arr == accent_color)) < len(accent_positions):
                continue
            candidates.append((int(top), int(left), int(accent_color)))

    assert len(candidates) == 1, "must have a unique placement and accent-color interpretation"
    top, left, accent_color = candidates[0]

    out = g.copy()
    for i in range(CELL_INNER):
        for j in range(CELL_INNER):
            color = int(source_arr[i, j])
            if color == source_main_color:
                target = source_arr
            elif color == accent_color:
                target = accent_arr
            else:
                continue
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target
    return out


def _build_base_grid(separator: int, cell_patterns: list[list[np.ndarray]]) -> np.ndarray:
    """Construct the full grid from already chosen per-cell 3x3 interiors."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, cell_patterns[i][j], *_cell_origin(i, j))
    return grid


def _layout_labels(source: np.ndarray, shape_color: int, accent_color: int) -> np.ndarray:
    """Compress a source tile to source/keep/accent labels."""
    labels = np.full((CELL_INNER, CELL_INNER), LABEL_KEEP, dtype=int)
    labels[source == shape_color] = LABEL_SOURCE
    labels[source == accent_color] = LABEL_ACCENT
    return labels


def _placement_candidates(layout: np.ndarray, observed: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """Find all macro offsets matching the observed source/accent subset."""
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
    """Choose an informative observed subset containing both source and accent clues."""
    informative = [
        (top + r, left + c, int(layout[r, c]))
        for r in range(CELL_INNER)
        for c in range(CELL_INNER)
        if int(layout[r, c]) != LABEL_KEEP
    ]
    accent_cells = [item for item in informative if item[2] == LABEL_ACCENT]
    source_cells = [item for item in informative if item[2] == LABEL_SOURCE]
    assert len(accent_cells) == 3, "source layout is expected to contain exactly three accent positions"

    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = []
        need_source = size - 2
        if need_source < 2 or need_source > len(source_cells):
            continue
        for accent_subset in combinations(accent_cells, 2):
            for src_subset in combinations(source_cells, need_source):
                subset = list(accent_subset) + list(src_subset)
                if _placement_candidates(layout, subset) == [(top, left)]:
                    valid.append(subset)
        if valid:
            return random.choice(valid)
    return None


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: construct output first, then reveal an informative subset."""
    separator, shape_color, keep_color, bg_color, accent_color = random.sample(range(COLOR_POOL), 5)

    source = _build_source_pattern(shape_color, keep_color, accent_color)
    source_main_color, _ = _dominant_shape(source)
    assert source_main_color == shape_color
    accent = _build_accent_pattern(accent_color, keep_color, bg_color)
    majority_bank = _majority_variants(shape_color, keep_color, bg_color)
    if len(majority_bank) < CELL_GRID_N * CELL_GRID_N:
        return None

    chosen = random.sample(majority_bank, CELL_GRID_N * CELL_GRID_N)
    cell_patterns: list[list[np.ndarray]] = []
    idx = 0
    for _ in range(CELL_GRID_N):
        row = []
        for _ in range(CELL_GRID_N):
            row.append(chosen[idx].copy())
            idx += 1
        cell_patterns.append(row)

    base = _build_base_grid(separator, cell_patterns)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    layout = _layout_labels(source, shape_color, accent_color)

    out = base.copy()
    for r in range(CELL_INNER):
        for c in range(CELL_INNER):
            label = int(layout[r, c])
            if label == LABEL_KEEP:
                continue
            pat = source if label == LABEL_SOURCE else accent
            paste(out, pat, *_cell_origin(top + r, left + c))

    subset = _choose_given_subset(layout, top, left)
    if subset is None:
        return None

    inp = base.copy()
    for abs_r, abs_c, label in subset:
        pat = source if label == LABEL_SOURCE else accent
        paste(inp, pat, *_cell_origin(abs_r, abs_c))

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    _, _, cell_keys, shape_sigs = _extract_cells(inp, CELL_STEP)
    shape_counts: dict[tuple[int, ...], int] = {}
    exact_counts: dict[tuple[int, ...], int] = {}
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            sig = shape_sigs[i][j]
            key = cell_keys[i][j]
            shape_counts[sig] = shape_counts.get(sig, 0) + 1
            exact_counts[key] = exact_counts.get(key, 0) + 1
    if sorted(shape_counts.values()) != [sum(label == LABEL_SOURCE for _, _, label in subset), CELL_GRID_N * CELL_GRID_N - sum(label == LABEL_SOURCE for _, _, label in subset)]:
        return None

    repeated_non_source = [
        key
        for key, count in exact_counts.items()
        if count > 1 and key != _pattern_key(source)
    ]
    if repeated_non_source != [_pattern_key(accent)]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
