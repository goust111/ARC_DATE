# Combo variant of c4d067a0 (index 22)
# combo_spec: D5/2+D12/2
# Name: 图例放大叠加+颜色优先级

# Design analysis
# ===============
# spec_variant:
#   kept_from_original:
#     - the canvas is still a 23-27 square with one dominant background color
#     - the source legend is still a sparse top-left sprite of singleton pixels on a step-2 lattice
#     - the legend occupancy is still a connected staircase with 6-10 filled cells
#     - the solver still has to infer one affine projection from the legend into the main area
#   fused_change:
#     - D12/2 keeps the main-area projection overlapping: projected blocks use step=N-1, so neighboring blocks fuse
#       and color priority matters whenever two enlarged regions cover the same cell
#     - D5/2 adds a second projection of the legend itself: besides the normal block field indexed by lattice coords
#       (rr,cc), each source legend pixel is also replayed from its raw sprite coordinates (2*rr,2*cc), producing
#       a sparser enlarged watermark layer anchored at the same base
#     - both layers obey the same priority order, so overlap can happen in three ways:
#       block-vs-block, watermark-vs-watermark, and watermark-vs-block
#   behavior_trace_small_example:
#     - suppose the legend has colors A,B,C and the horizontal anchor row is [A][B]
#     - original task: A and B become two separated squares; missing squares are copied into empty lattice slots
#     - D12/2 alone: the main step becomes N-1, so [A][B] turns into one bicolored rectangle; if B appears more often
#       in the legend, B is higher priority and keeps the overlap strip
#     - this combo keeps that anchor decoding, then also projects the literal legend spacing: the source pixel at
#       lattice (1,1) is replayed again at offset (2,2) in main-area step units, creating a sparse shadow copy
#     - because the shadow uses doubled legend coordinates while the real field uses normalized lattice coordinates,
#       some watermark squares land on top of existing blocks and others land in new places; priority decides the color
#   complexity_calibration:
#     - block overlap is fixed to exactly one cell per adjacent step by using N in {3,4} and step=N-1
#     - color priority is encoded only by legend multiplicity, with strictly distinct counts, so humans can count it
#       from examples without relying on hidden color ids
#     - the watermark shares the same base as the real block field, which keeps the relationship structured instead of
#       introducing another free offset
#     - generation rejects samples where the watermark is invisible or only duplicates existing visible colors
#   implicit_assumption_audit_of_original_main:
#     - original code assumed each visible main-area object is an isolated monochrome square
#       -> invalid because the kept anchor is now one merged bicolored rectangle
#     - original code inferred the main step from many visible block top-left coordinates
#       -> invalid because the input keeps only one anchor pair
#     - original code could repaint blocks in arbitrary order because blocks did not overlap
#       -> invalid because repaint order is now the priority rule itself
#     - original code treated the legend only as a template for lattice cells
#       -> incomplete now because the same legend also generates a second watermark layer from raw sprite spacing
#   placement_legality:
#     - the main-area base column stays at least 8, so the enlarged layers never collide with the tiny source legend
#     - all actual blocks and all watermark squares must fit inside the canvas
#     - the input contains exactly one merged anchor component and no watermark layer; the output is rendered from
#       scratch so overlaps remain correct
#   visual_observability:
#     - the merged anchor exposes N, step, left/right order, and the winner of the overlap directly
#     - the watermark is a visibly sparser enlarged copy of the same top-left sprite, so the new trajectory is apparent
#     - repeated color counts across the legend give multiple supports for the priority ranking
#   cleanliness_and_inferability:
#     - only 2-4 colors are used, counts are distinct, and the anchor color pair is unique as an ordered horizontal
#       adjacency, so the solver has one stable way to recover the base
#     - generator requires both a watermark-only visible region and at least one watermark-vs-block conflict that
#       changes the final output, which keeps the combo behavior readable instead of noisy or degenerate
#   new_main_assumptions_and_generator_guarantees:
#     - main assumes the non-legend input contains exactly one fully filled two-color horizontal rectangle
#       -> generator keeps exactly one adjacent anchor pair
#     - main assumes legend color counts are strictly distinct
#       -> generator samples only strict count profiles
#     - main assumes the ordered anchor color pair occurs on exactly one legend adjacency
#       -> generator backtracks until that is true
#     - main assumes replaying watermark squares from raw legend coordinates still fits the canvas
#       -> generator computes base bounds from both the real field and the watermark field

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
    step_l = _positive_gcd(drs + dcs)
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts = []
    for r, c, v in legend:
        raw_r = int(r - rmin)
        raw_c = int(c - cmin)
        assert raw_r % step_l == 0 and raw_c % step_l == 0
        rr = raw_r // step_l
        cc = raw_c // step_l
        legend_pts.append((int(rr), int(cc), int(v), raw_r, raw_c))
    return legend_pts, int(step_l)


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
    for _rr, _cc, col, _raw_r, _raw_c in legend_pts:
        counts[col] = counts.get(col, 0) + 1
    freq_vals = list(counts.values())
    assert len(freq_vals) == len(set(freq_vals)), "legend color counts must be distinct"
    order = sorted(counts, key=lambda col: counts[col])
    rank = {col: i for i, col in enumerate(order)}
    return counts, order, rank


def _find_anchor_pair(legend_pts, left_color, right_color):
    legend_map = {(rr, cc): col for rr, cc, col, _raw_r, _raw_c in legend_pts}
    matches = []
    for rr, cc, col, _raw_r, _raw_c in legend_pts:
        if col != left_color:
            continue
        if legend_map.get((rr, cc + 1)) == right_color:
            matches.append((rr, cc))
    assert len(matches) == 1, "anchor color pair must identify a unique legend adjacency"
    return matches[0]


def _paint_layers(out, legend_pts, block_size, step, base_r, base_c, rank):
    legend_sorted = sorted(legend_pts, key=lambda item: (rank[item[2]], item[0], item[1]))
    h, w = out.shape
    for rr, cc, col, raw_r, raw_c in legend_sorted:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        wr0 = base_r + raw_r * step
        wc0 = base_c + raw_c * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size
        assert 0 <= wr0 <= h - block_size and 0 <= wc0 <= w - block_size
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
        out[wr0 : wr0 + block_size, wc0 : wc0 + block_size] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct an overlapped block field from a top-left legend, then overlay a larger sparse shadow of that same legend.

    Input structure:
    - The background is the most common color.
    - The source legend is still a sparse top-left lattice of singleton colored pixels.
    - In the main area, only one adjacent anchor pair is shown, and because the projected blocks overlap, it appears as a single horizontal two-color rectangle.

    Rule:
    - Legend color frequency defines a strict global priority: rarer colors are painted first, more frequent colors later.
    - The anchor rectangle reveals the block size, the overlapped step, and the ordered adjacent legend pair that fixes the main-area base.
    - After rebuilding the normal lattice of enlarged blocks, replay every source legend pixel a second time using its raw sprite spacing, so an enlarged watermark layer is overlaid under the same priority rule.

    Key insight:
    - The legend now drives two coordinate systems at once: normalized lattice coordinates create the real block field, while raw legend pixel offsets create the watermark shadow. The same merged anchor and the same color-count priority decode both.
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

    legend_pts, _step_l = _extract_legend(singles, h, w)
    counts, _order, rank = _priority_order(legend_pts)
    assert counts[anchor["winner"]] > counts[anchor["loser"]], \
        "anchor visibility must agree with legend priority"

    arr_rr, arr_cc = _find_anchor_pair(legend_pts, anchor["left_color"], anchor["right_color"])
    block_size = anchor["block_size"]
    step = anchor["step"]
    base_r = anchor["r0"] - arr_rr * step
    base_c = anchor["c0"] - arr_cc * step

    return _paint_layers(out, legend_pts, block_size, step, base_r, base_c, rank)


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
BLOCK_SIZE_RANGE = range(3, 5)
OVERLAP = 1
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: visibility checks ===
MIN_CROSS_DELTA = 3
MIN_SOLO_DELTA = 6

# === CFG: generation ===
MAX_RETRIES = 800


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


def _draw_small_legend(out, legend_colors):
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col


def _render_scene(size, bg, legend_colors, pts, base_r, base_c, block_size, step, keep_actual=None, add_watermark=False):
    out = np.full((size, size), bg, dtype=int)
    _draw_small_legend(out, legend_colors)

    order, _counts = _priority_from_colors(legend_colors)
    keep_set = None if keep_actual is None else set(keep_actual)
    for col in order:
        for rr, cc in pts:
            if keep_set is not None and (rr, cc) not in keep_set:
                continue
            if legend_colors[(rr, cc)] != col:
                continue
            br = base_r + rr * step
            bc = base_c + cc * step
            out[br : br + block_size, bc : bc + block_size] = col
        if add_watermark:
            for rr, cc in pts:
                if legend_colors[(rr, cc)] != col:
                    continue
                wr = base_r + rr * LEGEND_STEP * step
                wc = base_c + cc * LEGEND_STEP * step
                out[wr : wr + block_size, wc : wc + block_size] = col
    return out


def _coverage_masks(size, legend_colors, pts, base_r, base_c, block_size, step):
    actual = np.zeros((size, size), dtype=bool)
    water = np.zeros((size, size), dtype=bool)
    for rr, cc in pts:
        br = base_r + rr * step
        bc = base_c + cc * step
        actual[br : br + block_size, bc : bc + block_size] = True
        wr = base_r + rr * LEGEND_STEP * step
        wc = base_c + cc * LEGEND_STEP * step
        water[wr : wr + block_size, wc : wc + block_size] = True
    return actual, water


def _combo_metrics(size, bg, legend_colors, pts, base_r, base_c, block_size, step):
    blocks_only = _render_scene(
        size,
        bg,
        legend_colors,
        pts,
        base_r,
        base_c,
        block_size,
        step,
        keep_actual=None,
        add_watermark=False,
    )
    combo = _render_scene(
        size,
        bg,
        legend_colors,
        pts,
        base_r,
        base_c,
        block_size,
        step,
        keep_actual=None,
        add_watermark=True,
    )
    actual_mask, water_mask = _coverage_masks(size, legend_colors, pts, base_r, base_c, block_size, step)
    changed = combo != blocks_only
    cross_delta = int(np.count_nonzero(changed & actual_mask & water_mask))
    solo_delta = int(np.count_nonzero(changed & (~actual_mask) & water_mask))
    total_delta = int(np.count_nonzero(changed))
    return combo, blocks_only, cross_delta, solo_delta, total_delta


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
    max_raw_r = max_r * LEGEND_STEP
    max_raw_c = max_c * LEGEND_STEP
    max_proj_r = max(max_r, max_raw_r) * step
    max_proj_c = max(max_c, max_raw_c) * step
    r_hi = size - block_size - max_proj_r
    c_hi = size - block_size - max_proj_c
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    out, blocks_only, cross_delta, solo_delta, total_delta = _combo_metrics(
        size,
        bg,
        legend_colors,
        pts,
        base_r,
        base_c,
        block_size,
        step,
    )
    if cross_delta < MIN_CROSS_DELTA or solo_delta < MIN_SOLO_DELTA or total_delta <= 0:
        return None

    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    inp = _render_scene(
        size,
        bg,
        legend_colors,
        pts,
        base_r,
        base_c,
        block_size,
        step,
        keep_actual=keep,
        add_watermark=False,
    )

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(out, blocks_only):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
