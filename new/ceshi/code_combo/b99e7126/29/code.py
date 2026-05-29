# Combo variant of b99e7126 (index 29)
# combo_spec: D4/3+D15/2
# Name: 像素级放大 + 可变放大倍数

from functools import reduce
from itertools import combinations
from math import gcd
import random

import numpy as np

from arc_utils import paste


# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Original task: the minority cell's 3x3 interior becomes a 3x3 cell-grid mask; each source
#    pixel expands to a whole cell interior filled by either the minority template or the majority
#    template. New task: the same sparse minority cells still locate the unique 3x3 cell block,
#    but after localization we magnify the *full 3x3 color pattern itself* at pixel resolution.
#    Example: with three visible minority clue cells, scale k=2, so the 3x3 source becomes a 6x6
#    patch; with four clues, k=3, so it becomes 9x9. That patch is centered inside the 11x11
#    pixel window covered by the located 3x3 cell block.
# b) Complexity calibration:
#    Only two scale values are allowed: {2, 3}. This keeps the variable zoom noticeable but easy
#    to infer across examples. The location logic still has repeated support because 3-4 minority
#    clue cells must match the active pixels of the hidden 3x3 mask, and the minority pattern
#    itself remains a two-color, high-fill shape with 6-7 active pixels.
# c) Broken original assumptions and replacements:
#    - Original assumption: an active source pixel maps to an entire cell-sized template.
#      Replaced by: every source pixel maps to a kxk pixel block in one contiguous zoomed patch.
#    - Original assumption: output is completed by filling more minority cells.
#      Replaced by: output starts from a majority-only grid and overlays the zoomed minority image.
#    - Original assumption: cell interiors remain the only editable units.
#      Replaced by: the overlay may cover divider pixels inside the 3x3 target window.
# d) Placement legality:
#    The hidden placement still uses a 3x3 mask inside the 7x7 cell grid, so top/left stay in
#    0..4. The centered zoom sizes are 6x6 or 9x9, both safely inside the 11x11 target window.
# e) Visual observability:
#    Humans can see three things directly: the minority template, the sparse clue cells, and the
#    larger 6x6/9x9 copy that appears in the matching 3x3 region. Scale is encoded by clue count
#    (3 clues -> 2x, 4 clues -> 3x), which is visible without using color ids or hidden metadata.
# f) Visual cleanliness:
#    Output is built from a clean majority grid plus one centered square patch, so there is only a
#    single overlay layer and no need to inspect multiple interacting objects in the same pixels.
# g) New solver assumptions checked against the generator:
#    - There are exactly two cell templates in the input (majority and clue minority): guaranteed.
#    - The minority template's most frequent color is its active color: guaranteed by construction.
#    - Exactly 3 or 4 minority clue cells exist, and that count encodes scale 2 or 3: guaranteed.
#    - The active clue subset yields a unique 3x3 placement: enforced by generator rejection.


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
SCALE_FROM_GIVEN = {3: 2, 4: 3}
COLOR_POOL = 10
MAX_RETRIES = 400


_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    """
    Infer a regular cell grid separated by full rows/cols of one divider color.
    Returns (step, row_divs, col_divs).
    """
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1)]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1)]
    step = reduce(gcd, drs + dcs)
    assert step > 1, "divider step must be > 1"
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must be at index 0"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must be at boundary"
    assert (h - 1) % step == 0 and (w - 1) % step == 0, "grid dimensions must be (k*step + 1)"
    return int(step), row_divs, col_divs


def _extract_cell_patterns(g: np.ndarray, step: int) -> tuple[int, dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Return inner cell size, pattern lookup, and per-cell pattern ids."""
    n = (g.shape[0] - 1) // step
    inner = step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            key = tuple(arr.ravel().tolist())
            if key not in patterns:
                patterns[key] = arr
            row.append(key)
        cell_pat.append(row)
    return inner, patterns, cell_pat


def _find_templates(cell_pat: list[list[tuple[int, ...]]]) -> tuple[tuple[int, ...], tuple[int, ...], dict[tuple[int, ...], int]]:
    """Identify minority and majority cell templates by frequency."""
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) == 2, "input must contain exactly majority and minority templates"
    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    return minority_pat, majority_pat, counts


def _active_mask_from_minority(minority_arr: np.ndarray) -> tuple[int, np.ndarray]:
    """Use the minority template's most frequent color as the active mask color."""
    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    return active_color, mask


def _find_unique_placement(existing: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Locate the single 3x3 placement whose active cells contain the existing clue cells."""
    n = existing.shape[0]
    mh, mw = mask.shape
    candidates: list[tuple[int, int]] = []
    for top in range(n - mh + 1):
        for left in range(n - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "minority clues must determine a unique placement"
    return candidates[0]


def _zoom_patch(pattern: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbor zoom of the full minority pattern."""
    k = int(scale)
    assert k >= 1
    return np.kron(pattern, np.ones((k, k), dtype=int))


def _overlay_zoom(base: np.ndarray, pattern: np.ndarray, top: int, left: int, step: int, scale: int) -> np.ndarray:
    """Overlay the zoomed minority pattern centered inside the placed 3x3 cell window."""
    out = base.copy()
    zoom = _zoom_patch(pattern, scale)
    window_r0 = top * step + 1
    window_c0 = left * step + 1
    window_h = pattern.shape[0] * step - 1
    window_w = pattern.shape[1] * step - 1
    assert zoom.shape[0] <= window_h and zoom.shape[1] <= window_w
    off_r = (window_h - zoom.shape[0]) // 2
    off_c = (window_w - zoom.shape[1]) // 2
    r0 = window_r0 + off_r
    c0 = window_c0 + off_c
    out[r0 : r0 + zoom.shape[0], c0 : c0 + zoom.shape[1]] = zoom
    return out


def _majority_grid(separator: int, majority: np.ndarray) -> np.ndarray:
    """Fill the whole 7x7 board with the majority cell template."""
    grid = np.full((GRID_SIZE, GRID_SIZE), int(separator), dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, i * CELL_STEP + 1, j * CELL_STEP + 1)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate the hidden 3x3 target block from sparse minority clues, then overlay a small pixel zoom.

    Input structure:
    - A 7x7 grid of 3x3 cell interiors separated by 1-pixel divider lines.
    - Most cells share one majority template; 3 or 4 clue cells use a distinct minority template.
    - The minority template's more frequent color marks the active positions of a hidden 3x3 mask.

    Rule:
    - Recover the minority template and its active mask from the clue cells.
    - Find the unique 3x3 cell block whose active mask positions contain all shown minority cells.
    - Let the number of shown minority cells encode the zoom factor: 3 clues -> 2x, 4 clues -> 3x.
    - Start from a majority-only grid, enlarge the full 3x3 minority pattern by that factor, and
      center the enlarged patch inside the located 3x3 cell window.

    Key insight:
    - The minority cells are not meant to be completed into more minority cells. They only pinpoint
      where the abnormal cell's own pixel pattern should be magnified, and their count sets the zoom.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    inner, patterns, cell_pat = _extract_cell_patterns(g, step)
    assert inner == CELL_INNER, "this fused variant expects 3x3 cell interiors"

    minority_pat, majority_pat, counts = _find_templates(cell_pat)
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]
    _active_color, mask = _active_mask_from_minority(minority_arr)

    n = len(cell_pat)
    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

    given = int(counts[minority_pat])
    assert given in SCALE_FROM_GIVEN, "minority clue count must encode a supported zoom"
    scale = SCALE_FROM_GIVEN[given]
    top, left = _find_unique_placement(existing, mask)

    separator = int(g[0, 0])
    base = _majority_grid(separator, majority_arr)
    return _overlay_zoom(base, minority_arr, top, left, step, scale)


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate all valid LR-symmetric 3x3 masks keyed by active count."""
    weights = [len(group) for group in _LR_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[i] for i in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for i in combo:
                    for r, c in _LR_GROUPS[i]:
                        mask[r, c] = False
                if mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Random non-uniform LR-symmetric 3x3 majority template using two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices, strict=False):
        for r, c in group:
            arr[r, c] = int(color)
    return arr


def _random_minority_mask() -> np.ndarray:
    """Pick a valid minority mask with 6 or 7 active pixels."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count]).copy()


def _placement_candidates(mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    """All placements where the given active subset is consistent with the mask."""
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
            candidates.append((int(t), int(l)))
    return candidates


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...] | None:
    """Choose 3 or 4 active mask positions that uniquely pin down the placement."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(SCALE_FROM_GIVEN.keys())
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


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    """Build one valid input/output pair and its behavior stats."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None

    given = len(subset)
    scale = SCALE_FROM_GIVEN[given]
    base = _majority_grid(separator, majority)
    inp = base.copy()
    for r, c in subset:
        paste(inp, minority, (top + r) * CELL_STEP + 1, (left + c) * CELL_STEP + 1)
    out = _overlay_zoom(base, minority, top, left, CELL_STEP, scale)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    stats = {
        "scale": int(scale),
        "given": int(given),
        "active_pixels": int(mask.sum()),
    }
    return inp, out, stats


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        built = _gen_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid fused b99e7126 input")
