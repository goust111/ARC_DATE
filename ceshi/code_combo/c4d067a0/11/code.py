# Combo variant of c4d067a0 (index 11)
# combo_spec: D8/1+D10/0
# Name: 干扰项过滤+单锚点

import random
import sys
from functools import reduce
from math import gcd

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The top-left sprite is still a sparse staircase legend on a step-2 lattice.
#    - The main area now shows exactly one real scaled block from that legend plus several
#      same-size distractor blocks. The distractors use legend colors but are not part of
#      the true scaled copy.
#    - Small trace:
#      Original: two adjacent real blocks at (10,10) and (10,14) directly reveal step=4
#      and the base offset, so the solver just extends the pattern.
#      Combo: only one real block might remain at (10,10), while distractors appear at
#      other plausible places such as (10,14) or (14,10) but with colors that would
#      contradict those wrong placements. The solver must test hypotheses from each visible
#      block, discard every hypothesis that clashes with at least one distractor, and keep
#      the unique hypothesis that leaves only one aligned anchor.
# b) Complexity calibration:
#    - Step stays in the original 3..5 range, so the search space is still tiny and human
#      understandable.
#    - There is exactly one true anchor block, but typically 2-4 distractors. That keeps
#      the single-anchor ambiguity visible while still giving repeated anti-evidence to
#      rule out wrong mappings.
#    - The generator requires multiple wrong zero-conflict hypotheses before distractors are
#      added, so the fusion does not collapse back to the original easier task.
# c) Implicit assumptions audited against the original main():
#    - Original gcd-based step recovery assumes multiple true blocks; that no longer holds
#      with a single anchor, so step must be enumerated instead of inferred from spacing.
#    - Original base voting treats every matching-color block as positive evidence; with
#      distractors present, that would trust the wrong blocks. The new solver therefore
#      looks for the unique hypothesis with zero projected color conflicts.
#    - Original repainting leaves off-lattice blocks in place. Here the output must erase
#      every same-size block first, then repaint only the recovered scaled copy.
# d) Placement legality:
#    - The generator keeps legend pixels, the true projected layout, and distractors inside
#      the canvas and mutually non-overlapping.
#    - Distractors are never placed on true projected slots, so the intended output remains
#      a clean completion of the legend pattern after distractor removal.
# e) Visual observability:
#    - Across examples, solvers can see that one visible block survives in the output while
#      the others disappear, and the surviving one is the only anchor compatible with a
#      conflict-free scaled copy of the legend.
#    - Distractors are placed near the block area rather than arbitrarily across the grid,
#      so the elimination step is visually attributable to the scaling rule.
# f) Visual cleanliness:
#    - All main-area objects are same-size monochrome squares, with only one layer of
#      symbols in the legend and one layer of squares in the block field. That keeps the
#      scene readable despite the added noise.
# g) Color inferability:
#    - Colors are never remapped. Every distractor also uses a legend color, so color still
#      comes entirely from the visible legend; only position-consistency determines which
#      square is genuine.
# 9) New solver assumptions and generator guarantees:
#    - Solver assumes exactly one candidate (step, base) yields zero projected conflicts and
#      exactly one aligned support block. Generator validates this before returning a sample.
#    - Solver assumes the dominant square size is shared by the true anchor and distractors.
#      Generator enforces one block size for all main-area squares.


# === Shared search constraints ===
SEARCH_STEP_MIN, SEARCH_STEP_MAX = 3, 5


def _extract_objects(g: np.ndarray, bg: int):
    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
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
    return singles, blocks


def _extract_legend(singletons: list[tuple[int, int, int]], shape: tuple[int, int]):
    h, w = shape
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singletons if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    step_l = reduce(gcd, drs + dcs)
    assert step_l > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // step_l
        cc = (c - cmin) // step_l
        legend_pts.append((int(rr), int(cc), int(v)))
    return legend_pts


def _dominant_block_size(blocks: list[tuple[int, int, int, int]]) -> int:
    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    return int(vals[cnts.argmax()])


def _project_positions(
    legend_pts: list[tuple[int, int, int]],
    step: int,
    base_r: int,
    base_c: int,
    block_size: int,
    shape: tuple[int, int],
):
    h, w = shape
    projected: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        if not (0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size):
            return None
        projected[(int(tr0), int(tc0))] = int(col)
    return projected


def _enumerate_candidates(
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int]],
    block_size: int,
    shape: tuple[int, int],
):
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, _s, col in blocks:
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))

    step_lo = max(SEARCH_STEP_MIN, block_size + 1)
    candidates: dict[tuple[int, int, int], dict[str, object]] = {}
    for step in range(step_lo, SEARCH_STEP_MAX + 1):
        for rr, cc, col in legend_pts:
            for br0, bc0 in by_color.get(col, []):
                base_r = br0 - rr * step
                base_c = bc0 - cc * step
                key = (int(step), int(base_r), int(base_c))
                if key in candidates:
                    continue
                projected = _project_positions(legend_pts, step, base_r, base_c, block_size, shape)
                if projected is None:
                    continue
                support = 0
                contradictions = 0
                for r0, c0, _s, bcol in blocks:
                    exp = projected.get((int(r0), int(c0)))
                    if exp is None:
                        continue
                    if exp == int(bcol):
                        support += 1
                    else:
                        contradictions += 1
                candidates[key] = {
                    "support": int(support),
                    "contradictions": int(contradictions),
                    "projected": projected,
                }
    return candidates


def _valid_candidates(
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int]],
    block_size: int,
    shape: tuple[int, int],
):
    candidates = _enumerate_candidates(legend_pts, blocks, block_size, shape)
    valid = {
        key: info
        for key, info in candidates.items()
        if int(info["support"]) >= 1 and int(info["contradictions"]) == 0
    }
    return candidates, valid


def _find_mapping(
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int]],
    block_size: int,
    shape: tuple[int, int],
):
    _all_candidates, valid = _valid_candidates(legend_pts, blocks, block_size, shape)
    assert valid, "at least one conflict-free mapping must exist"
    best = [key for key, info in valid.items() if int(info["support"]) == max(int(v["support"]) for v in valid.values())]
    assert len(best) == 1, "single-anchor mapping must be unique"
    step, base_r, base_c = best[0]
    assert int(valid[best[0]]["support"]) == 1, "combo variant expects exactly one true anchor block"
    return int(step), int(base_r), int(base_c), valid[best[0]]["projected"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a scaled block layout from the top-left legend when only one visible block is genuine and the others are distractors.

    Input structure:
    - The background is the grid's most frequent color.
    - A small legend of singleton pixels sits in the top-left corner on a regular lattice and specifies the intended color pattern.
    - Elsewhere there are several same-size square blocks, but only one lies on the true scaled copy of the legend; the rest are off-pattern distractors.

    Rule:
    - Treat each visible square block as a possible anchor for each same-colored legend pixel and each allowed step size.
    - Keep only the unique scale-and-offset hypothesis whose projected legend positions cause no color conflicts with any visible block that lands on those projected slots.
    - Erase every existing main-area square block, then repaint the entire projected legend layout at the recovered scale.

    Key insight:
    - With just one real anchor, positive evidence alone is insufficient; the distractors matter because wrong hypotheses make them land on projected slots with impossible colors, while the correct hypothesis leaves them as harmless off-lattice noise.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    singles, blocks = _extract_objects(g, bg)
    assert singles and blocks, "grid must contain legend pixels and main blocks"

    legend_pts = _extract_legend(singles, (h, w))
    block_size = _dominant_block_size(blocks)
    blocks = [b for b in blocks if b[2] == block_size]
    assert blocks, "must contain at least one block of the dominant block size"

    step, base_r, base_c, projected = _find_mapping(legend_pts, blocks, block_size, (h, w))
    _ = (step, base_r, base_c)

    out = g.copy()
    for r0, c0, s, _col in blocks:
        out[r0 : r0 + s, c0 : c0 + s] = bg
    for (tr0, tc0), col in projected.items():
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
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: combo behavior ===
MIN_PRE_ALT_CANDIDATES = 4
DISTRACTOR_RANGE = (2, 5)
MAX_KILL_STEPS = 6
MAX_RETRIES = 1400


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
_BLOCK_GAP_PAIRS = [
    (n, g)
    for n in BLOCK_SIZE_RANGE
    for g in GAP_RANGE
    if SEARCH_STEP_MIN <= n + g <= SEARCH_STEP_MAX
]


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    return [(r, c) for r, w in enumerate(widths) for c in range(w)]


def _random_color_counts(total, k):
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


def _assign_legend_colors(pts, palette):
    counts = dict(zip(palette, _random_color_counts(len(pts), len(palette))))
    colors = {}
    order = list(pts)
    random.shuffle(order)
    for pos in order:
        options = [col for col in palette if counts[col] > 0]
        if not options:
            return None
        col = random.choice(options)
        colors[pos] = col
        counts[col] -= 1
    return colors


def _render_output(size, bg, legend_colors, block_size, step, base_r, base_c):
    out = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        br = base_r + rr * step
        bc = base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = col
    return out


def _projected_from_colors(legend_colors, step, base_r, base_c):
    projected = {}
    for (rr, cc), col in legend_colors.items():
        projected[(int(base_r + rr * step), int(base_c + cc * step))] = int(col)
    return projected


def _make_reserved(size, legend_colors, block_size, step, base_r, base_c):
    occ = np.zeros((size, size), dtype=bool)
    lr0, lc0 = LEGEND_ORIGIN
    for rr, cc in legend_colors:
        occ[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = True
        br = base_r + rr * step
        bc = base_c + cc * step
        occ[br : br + block_size, bc : bc + block_size] = True
    return occ


def _blocks_from_input(inp, bg):
    _singles, blocks = _extract_objects(inp, bg)
    block_size = _dominant_block_size(blocks)
    return [b for b in blocks if b[2] == block_size]


def _placement_options(
    valid_wrong,
    correct_projected,
    reserved,
    palette,
    block_size,
    region,
):
    r_lo, r_hi, c_lo, c_hi = region
    options: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
    for key, info in valid_wrong.items():
        projected = info["projected"]
        for (r0, c0), exp_col in projected.items():
            if (int(r0), int(c0)) in correct_projected:
                continue
            if not (r_lo <= r0 <= r_hi and c_lo <= c0 <= c_hi):
                continue
            if reserved[r0 : r0 + block_size, c0 : c0 + block_size].any():
                continue
            for col in palette:
                if int(col) == int(exp_col):
                    continue
                opt = (int(r0), int(c0), int(col))
                options.setdefault(opt, set()).add(key)
    if not options:
        return []
    ranked = sorted(options.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1], kv[0][2]))
    return ranked


def _build_single_anchor_input(
    size,
    bg,
    legend_colors,
    anchor,
    block_size,
    step,
    base_r,
    base_c,
):
    inp = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        inp[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
    anchor_col = legend_colors[anchor]
    br = base_r + anchor[0] * step
    bc = base_c + anchor[1] * step
    inp[br : br + block_size, bc : bc + block_size] = anchor_col
    return inp


def _eliminate_wrong_candidates(
    inp,
    legend_pts,
    block_size,
    palette,
    correct_key,
    correct_projected,
    reserved,
    region,
):
    bg = get_bg(inp)
    blocks = _blocks_from_input(inp, bg)
    placed = 0
    for _ in range(MAX_KILL_STEPS):
        _cands, valid = _valid_candidates(legend_pts, blocks, block_size, inp.shape)
        valid_wrong = {
            key: info
            for key, info in valid.items()
            if key != correct_key
        }
        if not valid_wrong:
            return placed, len(valid)
        ranked = _placement_options(valid_wrong, correct_projected, reserved, palette, block_size, region)
        if not ranked:
            return None
        best_kill = len(ranked[0][1])
        best_opts = [opt for opt, killed in ranked if len(killed) == best_kill]
        r0, c0, col = random.choice(best_opts)
        inp[r0 : r0 + block_size, c0 : c0 + block_size] = col
        reserved[r0 : r0 + block_size, c0 : c0 + block_size] = True
        blocks.append((int(r0), int(c0), int(block_size), int(col)))
        placed += 1
        if placed >= DISTRACTOR_RANGE[1]:
            break

    _cands, valid = _valid_candidates(legend_pts, blocks, block_size, inp.shape)
    valid_wrong = {key: info for key, info in valid.items() if key != correct_key}
    if valid_wrong:
        return None
    return placed, len(valid)


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    pts = _pick_staircase()

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    legend_colors = _assign_legend_colors(pts, palette)
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

    out = _render_output(size, bg, legend_colors, block_size, step, base_r, base_c)
    legend_pts = [(rr, cc, col) for (rr, cc), col in legend_colors.items()]
    correct_key = (int(step), int(base_r), int(base_c))
    correct_projected = _projected_from_colors(legend_colors, step, base_r, base_c)

    anchor = random.choice(list(legend_colors))
    inp = _build_single_anchor_input(size, bg, legend_colors, anchor, block_size, step, base_r, base_c)

    base_blocks = _blocks_from_input(inp, bg)
    _base_cands, base_valid = _valid_candidates(legend_pts, base_blocks, block_size, inp.shape)
    pre_alt = sum(1 for key in base_valid if key != correct_key)
    if pre_alt < MIN_PRE_ALT_CANDIDATES:
        return None

    reserved = _make_reserved(size, legend_colors, block_size, step, base_r, base_c)
    region = (
        max(0, base_r - step),
        min(size - block_size, base_r + max_r * step + step),
        max(BASE_C_MIN, base_c - step),
        min(size - block_size, base_c + max_c * step + step),
    )
    elim = _eliminate_wrong_candidates(
        inp,
        legend_pts,
        block_size,
        palette,
        correct_key,
        correct_projected,
        reserved,
        region,
    )
    if elim is None:
        return None
    distractor_count, final_valid_count = elim
    if distractor_count < DISTRACTOR_RANGE[0]:
        return None
    if final_valid_count != 1:
        return None

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
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
