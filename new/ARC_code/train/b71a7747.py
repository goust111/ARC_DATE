import itertools

import numpy as np

from arc_utils import colors_present
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the hidden two-color image by compacting the rows and columns of its complete mask.

    Input structure
    The grid contains one pair of colors whose pixels occupy every cell in the
    cross product of their active rows and active columns; all other colors form
    the surrounding filler or separators.
    Rule
    Find that unique color pair, keep exactly the rows and columns touched by
    the pair, and return the induced subgrid.
    Key insight
    The target image is not chosen by color id; it is the only two-color mask
    whose occupied pixels form a full rectangular row/column product after
    deleting unused rows and columns.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    candidates = []
    for pair in itertools.combinations(colors_present(g), 2):
        mask = np.isin(g, pair)
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        rh = int(rows.sum())
        cw = int(cols.sum())
        if 0 < rh < h and 0 < cw < w and int(mask.sum()) == rh * cw:
            out = g[np.ix_(rows, cols)]
            if set(map(int, out.ravel())) == set(pair):
                candidates.append(out)

    assert len(candidates) == 1
    return candidates[0].copy()

# --------------- generator (att2) ---------------
# Canvas and block layout bounds.
N_BLOCK_OPTIONS = (3, 4)
BLOCK_SIZE_RANGE = range(3, 6)
SEPARATOR_WIDTH_RANGE = range(1, 4)
PADDING_MIN = 2
PADDING_MAX_EXCLUSIVE = 6
INPUT_SIDE_MIN = 19
INPUT_SIDE_MAX = 28

# Hidden foreground shape density and pattern mix.
FG_DENSITY_MIN = 0.14
FG_DENSITY_MAX = 0.22
RECT_LOOP_PROB = 0.25
BRANCH_TRANSPOSE_PROB = 0.5
MIN_BRANCH_ROW_GAP = 1

# Color and retry settings.
COLOR_COUNT_OPTIONS = (4, 5)
COLOR_POOL = range(10)
GENERATION_RETRIES = 300

LAYOUT_CHOICES = [
    (n_blocks, block_size, sep, pad0, pad1, side)
    for n_blocks in N_BLOCK_OPTIONS
    for block_size in BLOCK_SIZE_RANGE
    for sep in SEPARATOR_WIDTH_RANGE
    for pad0 in range(max(PADDING_MIN, sep), PADDING_MAX_EXCLUSIVE)
    for pad1 in range(max(PADDING_MIN, sep), PADDING_MAX_EXCLUSIVE)
    for side in [n_blocks * block_size + (n_blocks - 1) * sep + pad0 + pad1]
    if INPUT_SIDE_MIN <= side <= INPUT_SIDE_MAX
]

def _gen_density_bounds(size):
    return int(np.ceil(FG_DENSITY_MIN * size * size)), int(np.floor(FG_DENSITY_MAX * size * size))


def _gen_between(a, b):
    lo, hi = sorted((a, b))
    return range(lo, hi + 1)


def _gen_rect_loop_mask(size, block_size):
    lo, hi = _gen_density_bounds(size)
    choices = []
    for r0 in range(block_size):
        for r1 in range(size - block_size, size):
            for c0 in range(block_size):
                for c1 in range(size - block_size, size):
                    if r1 <= r0 or c1 <= c0:
                        continue
                    count = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
                    if lo <= count <= hi:
                        choices.append((r0, r1, c0, c1))
    if not choices:
        return None

    r0, r1, c0, c1 = random.choice(choices)
    mask = np.zeros((size, size), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _gen_spaced_branch_rows(r0, r1, main_r):
    rows = [r for r in range(r0, r1 + 1) if abs(r - main_r) > MIN_BRANCH_ROW_GAP]
    random.shuffle(rows)
    picked = []
    for r in rows:
        if all(abs(r - old) > MIN_BRANCH_ROW_GAP for old in picked):
            picked.append(r)
    return picked


def _gen_branch_mask(size, block_size):
    left = random.randrange(block_size)
    right = random.randrange(size - block_size, size)
    r0 = random.randrange(block_size)
    r1 = random.randrange(size - block_size, size)
    trunk_c = random.randrange(left, right + 1)
    main_r = random.randrange(r0, r1 + 1)

    base = (r1 - r0 + 1) + (right - left)
    density_min, max_count = _gen_density_bounds(size)
    min_count = max(density_min, base)

    picked = _gen_spaced_branch_rows(r0, r1, main_r)
    caps = [max(trunk_c, size - 1 - trunk_c) for _ in picked]
    capacity = base + sum(caps)
    if min_count > min(max_count, capacity):
        return None
    target = random.randint(min_count, min(max_count, capacity))

    mask = np.zeros((size, size), dtype=bool)
    mask[r0 : r1 + 1, trunk_c] = True
    mask[main_r, left : right + 1] = True

    remaining = target - base
    for i, (r, cap) in enumerate(zip(picked, caps)):
        if remaining <= 0:
            break
        future = sum(caps[i + 1 :])
        lo = max(0, remaining - future)
        hi = min(cap, remaining)
        take = random.randint(lo, hi)
        if take == 0:
            continue
        dirs = []
        if trunk_c >= take:
            dirs.append(-1)
        if size - 1 - trunk_c >= take:
            dirs.append(1)
        step = random.choice(dirs)
        end = trunk_c + step * take
        mask[r, _gen_between(trunk_c, end)] = True
        remaining -= take

    if remaining != 0:
        return None
    if random.random() < BRANCH_TRANSPOSE_PROB:
        mask = mask.T
    return mask


def _gen_output(size, block_size, bg, fg):
    mask = None
    if random.random() < RECT_LOOP_PROB:
        mask = _gen_rect_loop_mask(size, block_size)
    if mask is None:
        mask = _gen_branch_mask(size, block_size)
    if mask is None:
        return None

    out = np.full((size, size), bg, dtype=int)
    out[mask] = fg
    return out


def _gen_internal_separator_mask(active_rows, active_cols):
    filler = ~(active_rows[:, None] & active_cols[None, :])
    row_ids = np.flatnonzero(active_rows)
    col_ids = np.flatnonzero(active_cols)
    internal_rows = (~active_rows).copy()
    internal_cols = (~active_cols).copy()
    internal_rows[: row_ids[0]] = False
    internal_rows[row_ids[-1] + 1 :] = False
    internal_cols[: col_ids[0]] = False
    internal_cols[col_ids[-1] + 1 :] = False

    gridline = (internal_rows[:, None] | internal_cols[None, :]) & filler
    return filler, gridline


def _gen_fill_filler(inp, active_rows, active_cols, filler_colors):
    filler, gridline = _gen_internal_separator_mask(active_rows, active_cols)
    inp[filler] = filler_colors[0]
    inp[gridline] = filler_colors[1]

    for color in filler_colors[2:]:
        for r in range(inp.shape[0]):
            cols = np.flatnonzero(filler[r])
            inp[r, random.choice(cols)] = color
        for c in range(inp.shape[1]):
            rows = np.flatnonzero(filler[:, c])
            inp[random.choice(rows), c] = color


def _gen_active_indexes(n_blocks, block_size, sep, pad0, side):
    active = np.zeros(side, dtype=bool)
    indexes = []
    for i in range(n_blocks):
        start = pad0 + i * (block_size + sep)
        indexes.extend(range(start, start + block_size))
    active[indexes] = True
    return active, indexes


def _gen_palette():
    color_count = random.choice(COLOR_COUNT_OPTIONS)
    colors = random.sample(COLOR_POOL, color_count)
    return colors[:2], colors[2:]


def _gen_attempt():
    n_blocks, block_size, sep, pad0, _pad1, side = random.choice(LAYOUT_CHOICES)
    out_size = n_blocks * block_size

    (bg, fg), filler_colors = _gen_palette()

    out = _gen_output(out_size, block_size, bg, fg)
    if out is None:
        return None

    inp = np.empty((side, side), dtype=int)
    active_rows, row_map = _gen_active_indexes(n_blocks, block_size, sep, pad0, side)
    active_cols, col_map = _gen_active_indexes(n_blocks, block_size, sep, pad0, side)

    _gen_fill_filler(inp, active_rows, active_cols, filler_colors)
    inp[np.ix_(row_map, col_map)] = out

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
