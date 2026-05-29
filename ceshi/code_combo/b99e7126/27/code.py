# Combo variant of b99e7126 (index 27)
# combo_spec: D1/0+D6/0+D8/0
# Name: 频率排序 + 可变尺寸 + 棋盘铺贴

from itertools import combinations
import random

import numpy as np

from arc_utils import paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The original task had one repeated background tile, one rare source tile, and a
#    fixed 3x3 cell interior. This combo changes all three levers together:
#    - the divider lattice now encloses variable rectangular cell interiors
#    - the default lattice is a checkerboard of two base motifs A/B
#    - besides A/B, the board contains a source motif C and two more repeated distractor
#      motifs D/E, so the solver must rank motif frequencies instead of just finding the
#      only non-background tile
#    A concrete 2x3 example:
#    - suppose cells are 2x3, arranged on a 7x7 coarse grid
#    - parity-even cells normally show A, parity-odd cells show B
#    - the source motif C appears 3 times in the input; D appears 5 times; E appears 8
#      times; A/B fill the rest
#    - C's own 2x3 pixels form a boolean mask; after locating the unique macro placement,
#      active mask positions become C while inactive positions revert to A or B according
#      to absolute checkerboard parity
#    Relative to the original, the changed step is not only "which tile is the source"
#    but also "what counts as the background at each inactive slot" and "what cell size
#    is being magnified this time".
# b) Complexity calibration:
#    The board uses exactly five repeated motif families in every generated input:
#    base A, base B, source C, distractor D, distractor E. This is enough to force
#    frequency ranking without flooding the canvas with too many unrelated signals.
#    Source frequency is strictly minimal among repeated motifs; D/E are strictly larger;
#    A/B dominate by coverage. This keeps the task visible and learnable from examples.
# c) Hidden assumptions broken from the original solver:
#    - "one global majority pattern" is false because inactive macro slots must restore
#      checkerboard parity, not one tile
#    - "fixed 3x3 cell interiors" is false because row/column divider spacing now varies
#    - "the rarest pattern is obviously the source" is too weak because there are several
#      repeated non-base motifs; the solver must count repeated families first
#    The new solver therefore:
#    1. infers row/column cell steps from divider spacing
#    2. selects the uniquely least frequent repeated motif as source
#    3. recovers the checkerboard A/B assignment by searching for the ordered pair that
#       explains the most cells with zero parity mismatches
# d) Placement legality:
#    The source macro-block has shape equal to the source interior itself. Generator picks
#    a top-left placement that fits in the coarse grid, keeps distractors outside that
#    block, and only reveals a strict subset of source-active cells in the input. This
#    avoids overlap and keeps the completion region well-defined.
# e) Visual observability:
#    The source mask is constrained so its inactive cells hit both checkerboard parities.
#    That guarantees the completed region visibly contains both A and B, so the checkerboard
#    rule is observable inside the transformation itself instead of only in the untouched
#    background. Different samples also switch among multiple cell sizes, making D8 visible.
# f) Visual cleanliness:
#    Each cell contains exactly one motif. There is no pixel-level overlay beyond replacing
#    whole cell interiors. Distinct motif families use small two-color patterns, so the
#    board reads as a clean tiled mosaic instead of a cluttered stack of marks.
# g) Change inferability:
#    The color budget stays small: separator + shared + base-exclusive + source-exclusive
#    + distractor-exclusive colors for D and E. The structural clues do the heavy lifting:
#    alternating parity reveals A/B, repeated counts reveal C, and the source's own interior
#    reveals the macro mask.
# h) New solver assumptions and generator guarantees:
#    - source motif has a unique dominant color -> generator uses active counts above half
#    - source placement is unique -> generator only accepts subsets with one valid placement
#    - one ordered motif pair uniquely maximizes checkerboard coverage with zero mismatches
#      -> generator keeps A/B as the only large parity-consistent families
#    - source is uniquely least frequent among repeated motifs -> generator enforces strict
#      count ordering C < D < E < A/B

SIZE_CONFIGS = (
    {"inner_h": 2, "inner_w": 3, "n_rows": 7, "n_cols": 7, "active_counts": (4,), "given_sizes": (3,)},
    {"inner_h": 3, "inner_w": 2, "n_rows": 7, "n_cols": 7, "active_counts": (4,), "given_sizes": (3,)},
    {"inner_h": 4, "inner_w": 3, "n_rows": 5, "n_cols": 7, "active_counts": (7, 8), "given_sizes": (4, 5, 6)},
)
COLOR_POOL = 10
MAX_RETRIES = 500


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


def _extract_patterns(
    g: np.ndarray, row_step: int, col_step: int
) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
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


def _count_patterns(cell_pat: list[list[tuple[int, ...]]]) -> dict[tuple[int, ...], int]:
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    return counts


def _checkerboard_fit(
    cell_pat: list[list[tuple[int, ...]]], even_pat: tuple[int, ...], odd_pat: tuple[int, ...]
) -> tuple[int, int]:
    """Return (mismatches, coverage) for one ordered checkerboard assignment."""
    mismatches = 0
    coverage = 0
    for i, row in enumerate(cell_pat):
        for j, pat in enumerate(row):
            if pat != even_pat and pat != odd_pat:
                continue
            coverage += 1
            expected = even_pat if (i + j) % 2 == 0 else odd_pat
            mismatches += int(pat != expected)
    return mismatches, coverage


def _dominant_mask(arr: np.ndarray) -> np.ndarray:
    vals, cnts = np.unique(arr, return_counts=True)
    order = np.argsort(cnts)[::-1]
    assert len(order) >= 2 and cnts[order[0]] > cnts[order[1]], "source motif needs a unique active color"
    active_color = int(vals[order[0]])
    return arr == active_color


def _find_source_placement(mask: np.ndarray, existing: np.ndarray) -> tuple[int, int]:
    mh, mw = mask.shape
    n_rows, n_cols = existing.shape
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
    assert len(candidates) == 1, "source motif must admit a unique placement"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a self-similar source motif over a variable-size checkerboard cell lattice.

    Input structure:
    - The board is partitioned by one-color divider rows and columns into equal rectangular cells.
    - Most cells follow a checkerboard alternation of two base motifs, but there are also several
      repeated non-base motifs.
    - Among the repeated motif families, one source family is the uniquely least frequent repeated one.

    Rule:
    - Infer the row/column cell size from divider spacing, then count exact cell-interior motifs.
    - Choose the uniquely least frequent repeated motif as the source, read its dominant color as a
      boolean mask, and locate the unique macro placement consistent with the visible source cells.
    - Infer which two motifs form the underlying checkerboard, then repaint the source block so active
      mask positions use the source motif and inactive positions restore the parity-matching base motif.

    Key insight: The task is no longer "find the only odd tile"; the solver must first rank repeated
    motif families, while also inferring both the variable cell size and the alternating A/B background
    that the source blueprint expands over.
    """
    g = np.asarray(grid, dtype=int)
    row_step, col_step, _, _ = _infer_cell_grid_params(g)
    inner_h = row_step - 1
    inner_w = col_step - 1
    n_rows = (g.shape[0] - 1) // row_step
    n_cols = (g.shape[1] - 1) // col_step

    patterns, cell_pat = _extract_patterns(g, row_step, col_step)
    counts = _count_patterns(cell_pat)
    repeated = {pat: cnt for pat, cnt in counts.items() if cnt >= 2}
    assert len(repeated) >= 5, "need at least five repeated motif families"

    ordered = sorted(repeated.items(), key=lambda kv: kv[1])
    assert ordered[0][1] < ordered[1][1], "source motif frequency must be uniquely minimal among repeats"
    source_pat = ordered[0][0]
    source_arr = patterns[source_pat]
    assert source_arr.shape == (inner_h, inner_w)

    mask = _dominant_mask(source_arr)
    existing = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        for j in range(n_cols):
            existing[i, j] = cell_pat[i][j] == source_pat
    top, left = _find_source_placement(mask, existing)

    source_excluded = [pat for pat in repeated if pat != source_pat]
    assignments: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
    for even_pat in source_excluded:
        for odd_pat in source_excluded:
            if even_pat == odd_pat:
                continue
            mismatches, coverage = _checkerboard_fit(cell_pat, even_pat, odd_pat)
            assignments.append((mismatches, -coverage, even_pat, odd_pat))
    assignments.sort(key=lambda item: (item[0], item[1]))
    assert assignments, "need checkerboard candidates"
    assert assignments[0][0] == 0, "checkerboard base must have a zero-mismatch assignment"
    if len(assignments) > 1:
        assert assignments[0][:2] < assignments[1][:2], "checkerboard assignment must be unique"
    _, _, even_pat, odd_pat = assignments[0]
    even_arr = patterns[even_pat]
    odd_arr = patterns[odd_pat]

    out = g.copy()
    mh, mw = mask.shape
    for i in range(mh):
        for j in range(mw):
            abs_i = top + i
            abs_j = left + j
            target = source_arr if mask[i, j] else (even_arr if (abs_i + abs_j) % 2 == 0 else odd_arr)
            r0 = abs_i * row_step + 1
            c0 = abs_j * col_step + 1
            out[r0 : r0 + inner_h, c0 : c0 + inner_w] = target
    return out


def _precompute_masks() -> dict[tuple[int, int, int], list[np.ndarray]]:
    """Enumerate valid source masks keyed by (inner_h, inner_w, active_count)."""
    masks: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for cfg in SIZE_CONFIGS:
        inner_h = int(cfg["inner_h"])
        inner_w = int(cfg["inner_w"])
        total = inner_h * inner_w
        coords = list(range(total))
        for active_count in cfg["active_counts"]:
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
                inactive = np.argwhere(~mask)
                if len(inactive) < 2:
                    continue
                if len({(int(r) + int(c)) % 2 for r, c in inactive}) != 2:
                    continue
                bucket.append(mask)
            masks[(inner_h, inner_w, int(active_count))] = bucket
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


def _random_checkerboard_pair(shared: int, base_exclusive: int, inner_h: int, inner_w: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose two visibly distinct base motifs using the same two colors."""
    threshold = max(2, (inner_h * inner_w) // 3)
    for _ in range(80):
        pat_a = _random_pattern(inner_h, inner_w, shared, base_exclusive)
        pat_b = _random_pattern(inner_h, inner_w, shared, base_exclusive)
        if np.array_equal(pat_a, pat_b):
            continue
        if np.count_nonzero(pat_a != pat_b) < threshold:
            continue
        return pat_a, pat_b
    raise RuntimeError("failed to choose distinct checkerboard motifs")


def _choose_mask(inner_h: int, inner_w: int, active_counts: tuple[int, ...]) -> np.ndarray:
    active_count = int(random.choice(active_counts))
    return random.choice(MASKS_BY_KEY[(inner_h, inner_w, active_count)]).copy()


def _cell_origin(i: int, j: int, row_step: int, col_step: int) -> tuple[int, int]:
    return i * row_step + 1, j * col_step + 1


def _grid_shape(inner_h: int, inner_w: int, n_rows: int, n_cols: int) -> tuple[int, int]:
    return n_rows * (inner_h + 1) + 1, n_cols * (inner_w + 1) + 1


def _build_grid(
    separator: int,
    even_pat: np.ndarray,
    odd_pat: np.ndarray,
    overrides: dict[tuple[int, int], np.ndarray],
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Fill the checkerboard base, then overwrite selected cells."""
    inner_h, inner_w = even_pat.shape
    row_step = inner_h + 1
    col_step = inner_w + 1
    h, w = _grid_shape(inner_h, inner_w, n_rows, n_cols)
    grid = np.full((h, w), separator, dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            base_pat = even_pat if (i + j) % 2 == 0 else odd_pat
            paste(grid, base_pat, *_cell_origin(i, j, row_step, col_step))
    for (i, j), pat in overrides.items():
        paste(grid, pat, *_cell_origin(i, j, row_step, col_step))
    return grid


def _placement_candidates(
    mask: np.ndarray, given_cells: tuple[tuple[int, int], ...], n_rows: int, n_cols: int
) -> list[tuple[int, int]]:
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
    active_abs = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]
    sizes = list(given_sizes)
    random.shuffle(sizes)
    for size in sizes:
        if size >= len(active_abs):
            continue
        valid = []
        for subset in combinations(active_abs, size):
            if len({(r + c) % 2 for r, c in subset}) != 2:
                continue
            if _placement_candidates(mask, subset, n_rows, n_cols) != [(top, left)]:
                continue
            valid.append(list(subset))
        if valid:
            return random.choice(valid)
    return None


def _cells_outside_block(top: int, left: int, mh: int, mw: int, n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i in range(n_rows)
        for j in range(n_cols)
        if not (top <= i < top + mh and left <= j < left + mw)
    ]


def _sample_cells_spanning_parities(pool: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if len(pool) < count:
        return None
    for _ in range(60):
        cells = random.sample(pool, count)
        if len({(r + c) % 2 for r, c in cells}) == 2:
            return cells
    return None


def _count_signature(grid: np.ndarray) -> list[int]:
    row_step, col_step, _, _ = _infer_cell_grid_params(grid)
    _, cell_pat = _extract_patterns(grid, row_step, col_step)
    return sorted(_count_patterns(cell_pat).values())


def _gen_attempt(cfg: dict[str, object]) -> np.ndarray | None:
    inner_h = int(cfg["inner_h"])
    inner_w = int(cfg["inner_w"])
    n_rows = int(cfg["n_rows"])
    n_cols = int(cfg["n_cols"])
    active_counts = tuple(int(v) for v in cfg["active_counts"])
    given_sizes = tuple(int(v) for v in cfg["given_sizes"])

    colors = random.sample(range(COLOR_POOL), 6)
    separator, shared, base_exclusive, source_exclusive, d1_exclusive, d2_exclusive = colors

    even_pat, odd_pat = _random_checkerboard_pair(shared, base_exclusive, inner_h, inner_w)
    mask = _choose_mask(inner_h, inner_w, active_counts)
    source_pat = np.where(mask, source_exclusive, shared).astype(int)
    distractor1 = _random_pattern(inner_h, inner_w, shared, d1_exclusive)
    distractor2 = _random_pattern(inner_h, inner_w, shared, d2_exclusive)

    pats = [even_pat, odd_pat, source_pat, distractor1, distractor2]
    if any(np.array_equal(pats[i], pats[j]) for i in range(len(pats)) for j in range(i + 1, len(pats))):
        return None

    mh, mw = mask.shape
    top = random.randrange(n_rows - mh + 1)
    left = random.randrange(n_cols - mw + 1)
    active_cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]

    given_cells = _choose_given_cells(mask, top, left, n_rows, n_cols, given_sizes)
    if given_cells is None:
        return None

    source_count = len(given_cells)
    d1_count = source_count + random.choice((2, 3))
    d2_count = d1_count + random.choice((2, 3))

    free_cells = _cells_outside_block(top, left, mh, mw, n_rows, n_cols)
    d1_cells = _sample_cells_spanning_parities(free_cells, d1_count)
    if d1_cells is None:
        return None
    remaining = [cell for cell in free_cells if cell not in d1_cells]
    d2_cells = _sample_cells_spanning_parities(remaining, d2_count)
    if d2_cells is None:
        return None

    input_overrides = {(i, j): source_pat for i, j in given_cells}
    input_overrides.update({(i, j): distractor1 for i, j in d1_cells})
    input_overrides.update({(i, j): distractor2 for i, j in d2_cells})

    output_overrides = {(i, j): source_pat for i, j in active_cells}
    output_overrides.update({(i, j): distractor1 for i, j in d1_cells})
    output_overrides.update({(i, j): distractor2 for i, j in d2_cells})

    inp = _build_grid(separator, even_pat, odd_pat, input_overrides, n_rows, n_cols)
    out = _build_grid(separator, even_pat, odd_pat, output_overrides, n_rows, n_cols)

    sig = _count_signature(inp)
    if len(sig) != 5:
        return None
    if not (sig[0] == source_count and sig[0] < sig[1] < sig[2] < sig[3] <= sig[4]):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    cfg = random.choice(SIZE_CONFIGS)
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(cfg)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
