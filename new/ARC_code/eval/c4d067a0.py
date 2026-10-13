import numpy as np
from functools import reduce
from math import gcd

from arc_utils import bbox_coords, components_mask, get_bg
import random

def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a grid of colored blocks to match a minimap legend in the top-left corner.

    Input structure:
    - bg = most common color.
    - Top-left corner: a small sprite of single colored pixels forming a minimap pattern on a regular lattice.
    - Main area: uniformly colored square blocks (NxN) arranged on a regular grid; some positions are empty (bg).

    Rule:
    - The legend pixels define a color pattern; each pixel's lattice position maps to a block position in the main grid via a fixed scale factor and offset.
    - Existing colored blocks confirm the scale/offset alignment.
    - For every legend pixel whose corresponding grid position is empty, place a new block of that color.

    Key insight: The top-left pixel pattern is a scaled-down minimap of the block layout; the mapping is inferred from blocks already present.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []  # (r0,c0,size,color)
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))

    assert singles and blocks, "grid must contain legend pixels and main blocks"

    # Legend pixels: singleton colored pixels in a small top-left window.
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    # Infer legend lattice step (usually 2).
    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    stepL = reduce(gcd, drs + dcs)
    assert stepL > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))

    # Determine typical block size and step between blocks.
    sizes = [s for _, _, s, _ in blocks]
    _vals, _cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(_vals[_cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert blocks, "must contain at least one block of the dominant block size"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, s, col in blocks:
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))
    for col in by_color:
        by_color[col].sort()

    # Compute step from block top-left positions (gcd of diffs).
    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = []
    step_candidates += [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    step = reduce(gcd, step_candidates)
    assert step > 0, "block step must be positive"

    # Infer base offset: choose base that best maps legend pixels with existing block colors.
    bases: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        if col not in by_color:
            continue
        for br0, bc0 in by_color[col]:
            base_r = br0 - rr * step
            base_c = bc0 - cc * step
            bases[(base_r, base_c)] = bases.get((base_r, base_c), 0) + 1
    assert bases, "existing blocks must determine the legend-to-grid offset"
    max_votes = max(bases.values())
    best_bases = [base for base, votes in bases.items() if votes == max_votes]
    assert len(best_bases) == 1, "legend-to-grid offset must be unique"
    base = best_bases[0]
    base_r, base_c = base

    # Paint blocks for all legend pixels (usually adding missing ones).
    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, \
            "legend block projection must fit in canvas"
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out

# --------------- generator (o5x) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27  # square grid bounds
NUM_COLORS = 10                         # total ARC color palette size

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)                  # top-left pixel of the legend
LEGEND_STEP = 2                         # pixel spacing between legend dots
LEGEND_ROWS = (3, 4)                    # staircase row count options
LEGEND_COLS = (2, 3)                    # staircase column count options
LEGEND_EMPTIES = (0, 1, 2)             # how many lattice positions may be empty
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4  # distinct non-bg colors in legend
MAX_COUNT_PER_COLOR = 6                 # max legend pixels of one color

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)         # NxN block side length
GAP_RANGE = range(1, 4)               # gap between adjacent blocks
STEP_MIN, STEP_MAX = 3, 5             # block_size + gap bounds
BASE_C_MIN = 8                         # minimum block-area column (separates from legend)
BASE_R_MIN = 1                         # minimum block-area row

# === CFG: generation ===
MAX_RETRIES = 300


def _enumerate_staircase_shapes():
    """Enumerate all valid (widths) tuples for the staircase legend shape."""
    shapes = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue
                def rec(prefix, remaining, r):
                    if r == rows:
                        if remaining == 0 and max(prefix) == cols:
                            shapes.append(tuple(prefix))
                        return
                    left = rows - r - 1
                    lo = max(1, remaining - left * cols)
                    hi = min(cols, remaining - left)
                    for width in range(lo, hi + 1):
                        rec(prefix + [width], remaining - width, r + 1)
                rec([], count, 0)
    return shapes


# Valid (block_size, gap) pairs satisfying STEP_MIN <= block_size+gap <= STEP_MAX
_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE
    if STEP_MIN <= n + g <= STEP_MAX
]
# All valid staircase width tuples, precomputed once
_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    """Pick a random staircase shape; return (filled_pts, adjacent_pairs)."""
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    pairs = [(r, c) for r, w in enumerate(widths) for c in range(w - 1)]
    return pts, pairs


def _random_color_counts(total, k):
    """Distribute `total` among `k` bins, each in [1, MAX_COUNT_PER_COLOR]."""
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        i = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[i]))
        counts[i] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(pts, anchor_pair, palette):
    """Assign colors to legend pts via backtracking, ensuring unique anchor-pair identification."""
    pos_set = set(pts)
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    a, b = random.sample(palette, 2)
    counts = dict(zip(palette, _random_color_counts(len(pts), len(palette))))
    colors = {first: a, second: b}
    counts[a] -= 1
    counts[b] -= 1
    if counts[a] < 0 or counts[b] < 0:
        return None

    anchor_colors = (a, b)
    cells = [p for p in pts if p not in colors]

    def allowed(pos, col):
        r, c = pos
        left = (r, c - 1)
        if left in pos_set and left in colors:
            if (colors[left], col) == anchor_colors and (left, pos) != (first, second):
                return False
        right = (r, c + 1)
        if right in pos_set and right in colors:
            if (col, colors[right]) == anchor_colors and (pos, right) != (first, second):
                return False
        return True

    def search():
        if not cells:
            return all(v == 0 for v in counts.values())
        best_i, best_opts = None, None
        for i, pos in enumerate(cells):
            opts = [col for col in palette if counts[col] > 0 and allowed(pos, col)]
            if best_opts is None or len(opts) < len(best_opts):
                best_i, best_opts = i, opts
        if not best_opts:
            return False
        pos = cells.pop(best_i)
        random.shuffle(best_opts)
        for col in best_opts:
            colors[pos] = col
            counts[col] -= 1
            if search():
                return True
            counts[col] += 1
            del colors[pos]
        cells.insert(best_i, pos)
        return False

    if not search():
        return None
    return colors


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    # Render output: legend pixels + all blocks
    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        br, bc = base_r + rr * step, base_c + cc * step
        out[br:br + block_size, bc:bc + block_size] = col

    # Build input: keep only the 2 anchor blocks
    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        br, bc = base_r + rr * step, base_c + cc * step
        inp[br:br + block_size, bc:bc + block_size] = bg

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
    raise RuntimeError("failed to generate valid c4d067a0 input")
