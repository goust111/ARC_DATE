# Variant of c4d067a0
# Dimension: D12 — 图例中颜色的多样性 (meta)
# Change: 图例包含2-4种不同颜色的像素，每种颜色的放大色块大小相同... → 图例颜色有层级含义

# Design analysis
# ===============
# spec_variant:
#   kept_from_original:
#     - grid remains a 23-27 square with one dominant background color
#     - the legend is still a top-left sprite of singleton pixels on a step-2 lattice
#     - legend occupancy is still a connected staircase with 6-10 filled lattice cells
#     - the solver still has to infer the mapping from legend lattice coordinates to a larger block area
#   core_change_D12:
#     - legend colors now have two roles:
#       1. visible fill color of the enlarged block
#       2. global priority rank used when projected blocks overlap
#     - priority is not tied to color id; it is encoded by visible legend structure:
#       the more times a color appears in the legend, the higher its priority
#     - counts are generated to be strictly distinct, so 2-4 colors induce a strict total order
#   concrete_grid_behavior:
#     - all enlarged blocks still have the same square size N
#     - unlike the original task, the lattice step in the main area is N-1, so adjacent projected blocks overlap by 1 cell
#     - the input keeps only one horizontal adjacent anchor pair from the main area
#     - because those two anchor blocks overlap, the visible anchor is one bicolored rectangle:
#       height=N and width=(N + step) = 2N-1
#     - the wider stripe belongs to the higher-priority anchor color because it remains visible across the overlap
#   behavior_trace_small_example:
#     - original: legend row has [red][blue], main step > N, so two separate squares are visible; solver copies missing squares into empty spots
#     - variant: legend row has [red][blue], step=N-1, so the squares overlap by one column
#     - if blue appears more often in the legend than red, blue has higher priority
#     - projected anchor becomes a rectangle with a red strip of width step and a blue strip of width N
#     - solver reads that merged anchor, recovers N and step, locates the unique [red][blue] adjacency in the legend,
#       then redraws every legend pixel from low priority to high priority so later blocks cover earlier ones
#   complexity_calibration:
#     - overlap is fixed to 1 cell rather than arbitrary large overlap; this makes the effect visible without creating clutter
#     - N is restricted to 3-4, so even surrounded low-priority blocks keep a visible interior and do not disappear completely
#     - color counts are distinct and each color appears multiple times across the legend whenever possible, giving repeated support
#   implicit_assumption_audit_of_original_main:
#     - original component parsing assumed every main-area block is an isolated monochrome square
#       -> invalid now because the two anchor blocks merge into one multicolor rectangle
#     - original step inference used gcd of many block top-left coordinates
#       -> invalid now because the input no longer exposes many separate blocks
#     - original painting overwrote each square directly, which was harmless when blocks did not intersect
#       -> now overwrite order is the rule itself, so painting order must follow inferred priority
#     - original generator erased non-anchor blocks from the finished output
#       -> invalid now because deleting a higher-priority overlapping block from the final canvas would also erase pixels
#          that should reveal a kept lower-priority anchor underneath; the input must be rendered from scratch
#   placement_legality:
#     - the legend stays in the top-left corner and main-area base column stays >= 8 to avoid any contact with legend pixels
#     - every projected block fits fully inside the canvas
#     - because step=N-1 and N in {3,4}, overlaps are shallow and every block keeps some exclusive visible area
#   visual_observability:
#     - overlap produces obvious fused rectangles and stepped blended regions in the output, so the variant is visible
#     - the priority signal uses legend multiplicity, which is directly countable by a human across examples
#     - no hidden ordering by color id, no arbitrary per-sample priority table
#   new_main_assumptions_and_generator_guarantees:
#     - main assumes the input contains exactly one non-legend anchor component shaped as a fully filled bicolored horizontal rectangle
#       -> generator always keeps exactly one horizontal adjacent anchor pair and renders only that pair in the input
#     - main assumes legend color counts are strictly distinct
#       -> generator samples only distinct count profiles
#     - main assumes the ordered anchor color pair appears on exactly one horizontal adjacency in the legend
#       -> generator backtracks until that pair is unique

import itertools
import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


def _positive_gcd(vals):
    vals = [abs(int(v)) for v in vals if int(v) != 0]
    assert vals, "need at least one non-zero spacing"
    return reduce(gcd, vals)


def _extract_legend(singles, h, w):
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    stepL = _positive_gcd(drs + dcs)
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))
    return legend_pts


def _parse_anchor_component(g, pts, bg):
    r0, c0, r1, c1 = bbox_coords(pts)
    hh, ww = r1 - r0 + 1, c1 - c0 + 1
    if hh < 2 or ww <= hh:
        return None
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub == bg):
        return None
    if not np.all(sub == sub[0:1, :]):
        return None

    row = sub[0]
    changes = [i for i in range(1, ww) if int(row[i]) != int(row[i - 1])]
    if len(changes) != 1:
        return None
    split = changes[0]
    left_col = int(row[0])
    right_col = int(row[-1])
    if left_col == right_col:
        return None
    if not np.all(sub[:, :split] == left_col):
        return None
    if not np.all(sub[:, split:] == right_col):
        return None

    block_size = int(hh)
    step = int(ww - hh)
    if step <= 0:
        return None

    widths = (split, ww - split)
    if sorted(widths) != [step, block_size]:
        return None
    winner = left_col if split == block_size else right_col
    loser = right_col if winner == left_col else left_col
    return {
        "r0": int(r0),
        "c0": int(c0),
        "block_size": block_size,
        "step": step,
        "left_color": left_col,
        "right_color": right_col,
        "winner": int(winner),
        "loser": int(loser),
    }


def _priority_order(legend_pts):
    counts = {}
    for _rr, _cc, col in legend_pts:
        counts[col] = counts.get(col, 0) + 1
    freq_vals = list(counts.values())
    assert len(freq_vals) == len(set(freq_vals)), "legend color counts must be distinct"
    order = sorted(counts, key=lambda col: counts[col])
    rank = {col: i for i, col in enumerate(order)}
    return counts, order, rank


def _find_anchor_pair(legend_pts, left_color, right_color):
    legend_map = {(rr, cc): col for rr, cc, col in legend_pts}
    matches = []
    for rr, cc, col in legend_pts:
        if col != left_color:
            continue
        if legend_map.get((rr, cc + 1)) == right_color:
            matches.append((rr, cc))
    assert len(matches) == 1, "anchor color pair must identify a unique legend adjacency"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a field of enlarged blocks from a top-left legend, allowing overlaps between blocks.

    Input structure:
    - The background is the most common color.
    - The legend remains a sparse top-left lattice of single pixels.
    - In the main area, only one adjacent pair of enlarged blocks is shown; because the blocks overlap, they appear as one bicolored horizontal rectangle.

    Rule:
    - The legend pixel positions still map to a regular lattice of equal-sized square blocks in the main area.
    - A color's legend frequency defines its priority: rarer colors are painted first, more frequent colors later, so higher-priority colors cover lower-priority ones on overlap.
    - The merged anchor rectangle reveals the block size, the lattice step, and which two neighboring legend pixels determine the base offset.

    Key insight:
    - The anchor is no longer two isolated squares. It is the final visible result of two overlapping projected blocks, so decoding that fused rectangle is enough to recover the global projection before replaying the whole legend in priority order.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    singles = []
    anchors = []
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        parsed = _parse_anchor_component(g, pts, bg)
        if parsed is not None:
            anchors.append(parsed)

    assert singles and anchors, "grid must contain legend pixels and one merged anchor"
    assert len(anchors) == 1, "input must contain exactly one merged anchor component"
    anchor = anchors[0]

    legend_pts = _extract_legend(singles, h, w)
    counts, _order, rank = _priority_order(legend_pts)
    assert counts[anchor["winner"]] > counts[anchor["loser"]], \
        "anchor visibility must agree with legend priority"

    arr_rr, arr_cc = _find_anchor_pair(legend_pts, anchor["left_color"], anchor["right_color"])
    block_size = anchor["block_size"]
    step = anchor["step"]
    base_r = anchor["r0"] - arr_rr * step
    base_c = anchor["c0"] - arr_cc * step

    legend_sorted = sorted(legend_pts, key=lambda item: rank[item[2]])
    for rr, cc, col in legend_sorted:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, \
            "legend block projection must fit in canvas"
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out


# --------------- generator (o5x) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(3, 5)  # keep overlap shallow and visible
OVERLAP = 1
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 400


def _enumerate_staircase_shapes():
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


_STAIRCASE_SHAPES = _enumerate_staircase_shapes()
_STRICT_COUNT_OPTIONS = {}
for total in range(LEGEND_FILLED_MIN, LEGEND_FILLED_MAX + 1):
    for k in range(PALETTE_SIZE_MIN, PALETTE_SIZE_MAX + 1):
        opts = [
            combo for combo in itertools.combinations(range(1, MAX_COUNT_PER_COLOR + 1), k)
            if sum(combo) == total
        ]
        if opts:
            _STRICT_COUNT_OPTIONS[(total, k)] = opts


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    pairs = [(r, c) for r, w in enumerate(widths) for c in range(w - 1)]
    return pts, pairs


def _strict_color_counts(total, k):
    counts = list(random.choice(_STRICT_COUNT_OPTIONS[(total, k)]))
    random.shuffle(counts)
    return counts


def _assign_legend_colors(pts, anchor_pair, palette, count_list):
    pos_set = set(pts)
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    a, b = random.sample(palette, 2)
    counts = dict(zip(palette, count_list))
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
        right = (r, c + 1)
        if left in pos_set and left in colors:
            if (colors[left], col) == anchor_colors and (left, pos) != (first, second):
                return False
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


def _priority_from_colors(legend_colors):
    counts = {}
    for col in legend_colors.values():
        counts[col] = counts.get(col, 0) + 1
    vals = list(counts.values())
    assert len(vals) == len(set(vals))
    return sorted(counts, key=lambda col: counts[col]), counts


def _render_scene(size, bg, legend_colors, pts, base_r, base_c, block_size, step, keep=None):
    out = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col

    order, _counts = _priority_from_colors(legend_colors)
    keep_set = None if keep is None else set(keep)
    for col in order:
        for rr, cc in pts:
            if keep_set is not None and (rr, cc) not in keep_set:
                continue
            if legend_colors[(rr, cc)] != col:
                continue
            br = base_r + rr * step
            bc = base_c + cc * step
            out[br : br + block_size, bc : bc + block_size] = col
    return out


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size = random.choice(list(BLOCK_SIZE_RANGE))
    step = block_size - OVERLAP
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)

    total = len(pts)
    valid_k = [k for k in range(PALETTE_SIZE_MIN, PALETTE_SIZE_MAX + 1) if (total, k) in _STRICT_COUNT_OPTIONS]
    k = random.choice(valid_k)
    bg = random.randrange(NUM_COLORS)
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette, _strict_color_counts(total, k))
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

    out = _render_scene(size, bg, legend_colors, pts, base_r, base_c, block_size, step)
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    inp = _render_scene(size, bg, legend_colors, pts, base_r, base_c, block_size, step, keep=keep)

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
    raise RuntimeError("failed to generate valid c4d067a0 variant input")
