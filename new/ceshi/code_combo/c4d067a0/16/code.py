# Combo variant of c4d067a0 (index 16)
# combo_spec: D1/5+D7/0
# Name: 锚点匹配缩放+特殊标记编码缩放因子
#
# Design analysis
# ===============
# a) Concrete behavior and traced difference
#    - The original task shows a top-left sparse legend plus two already enlarged blocks.
#      Their spacing and size together reveal one affine map from legend lattice to block lattice.
#    - This combo splits those two jobs apart. The input now contains:
#      1) one unique legend anchor pixel whose color also appears as the only visible square seed
#         block in the main area, and
#      2) one different unique legend marker pixel whose legend-row position encodes the true scale.
#    - The visible seed block no longer carries the final scale. It only marks which legend cell
#      fixes the translation. The final enlargement uses a gapless lattice:
#          target(r, c) = (base_r + r * scale, base_c + c * scale)
#      with:
#          base_r = anchor_block_r0 - anchor_r * scale
#          base_c = anchor_block_c0 - anchor_c * scale
#      and:
#          scale = marker_row_number (1-based legend row count).
#    - Example trace:
#      original logic: two visible 3x3 blocks four cells apart imply one step and one size.
#      combo logic: a unique orange legend pixel matches the lone orange 2x2 seed block, so it
#      fixes the offset only; a unique teal legend pixel on the 4th legend row means scale=4;
#      therefore every legend dot becomes a 4x4 block on a pitch of 4 from that anchored origin.
#    - The crucial visible difference is that the seed block can stay 2x2 while the completed
#      output expands to 3x3 or 4x4 blocks. Scale must therefore be read from the legend marker.
#
# b) Complexity calibration
#    - The scale is restricted to {3, 4}. That is enough variation to make the row-code visible,
#      while also guaranteeing the true scale never coincides with the 2x2 anchor seed size.
#    - The legend always has 4 logical rows so the marker can visibly sit on row 3 or row 4
#      (1-based), making the encoded value human-readable across examples.
#    - Only two legend colors are special: one unique anchor color and one unique marker color.
#      The remaining cells use exactly two repeated filler colors, which keeps the pattern rich
#      enough to see the completed mosaic but avoids ambiguous extra singleton colors.
#
# c) Audit of original assumptions that break under this combo
#    - Original code infers scale from visible block size. That is false here: the only visible
#      square block is a constant-size seed and must not be reused as the final scale.
#    - Original code infers placement from multiple visible blocks and color voting. That fails
#      because only one main-area block is shown; translation must come directly from the anchor.
#    - D1/5 in isolation used the anchor block size as the scale. The combo deliberately removes
#      that assumption, so solver code must separate "which cell is anchored" from "how big blocks
#      should be".
#    - Any solver path that allows more than two singleton legend colors would misidentify the
#      marker, so the generator enforces exactly two unique legend colors: anchor and marker.
#
# d) Placement legality
#    - The legend stays inside the top-left 12x12 window on a step-2 lattice.
#    - The main-area mosaic starts at column >= 8, leaving visual separation from the legend.
#    - base_r/base_c are sampled so every scale x scale block fits the canvas.
#    - The only visible main-area component in the input is one isolated 2x2 square seed, so
#      connected-component extraction is stable and never merges with legend pixels.
#
# e) Visual observability
#    - Across examples the lone seed block always has the same small size, so it cannot explain
#      the varying completed block sizes by itself.
#    - The row of the unique non-anchor marker changes between examples and correlates directly
#      with whether the completed mosaic uses 3x3 or 4x4 blocks. This makes the symbol-decoding
#      step visible from a few training pairs.
#    - No color-id ordering is used. Special roles are exposed structurally: one unique color has
#      a matching seed block; the other unique color does not and sits on a diagnostic legend row.
#
# f) Visual cleanliness
#    - Legend pixels, the lone seed block, and the enlarged output occupy clearly separated zones.
#    - Orthogonally adjacent legend cells are forced to have different colors, so gapless scaling
#      does not create merged same-color slabs that would hide the lattice.
#    - The marker is encoded by row only; no extra border bars or overlay glyphs are added.
#
# g) Change inferability
#    - Colors are copied directly; there is no recoloring rule to guess.
#    - The only hidden-looking behavior is the marker's symbolic meaning, so generation keeps the
#      codebook tiny and repeated: row 3 -> scale 3, row 4 -> scale 4.
#    - Because the anchor seed size is always 2 and the true scale is always larger, the D7 part
#      is functionally necessary and the combo does not collapse back into the D1-only variant.
#
# New solver assumptions and generator guarantees
# -----------------------------------------------
# - Exactly one legend color appears once in the legend and once as the only visible main-area
#   square block color; this is the anchor match.
# - Exactly one other legend color appears once in the legend and nowhere among visible blocks;
#   this is the scale marker.
# - The marker lies on legend row 3 or 4 (1-based), encoding scale 3 or 4 respectively.
# - The visible anchor block is a 2x2 seed placed at the future top-left corner of the anchor's
#   enlarged block, but solver logic only depends on its top-left corner, not on its size.

from functools import reduce
from math import gcd
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


def _extract_singles_and_blocks(
    g: np.ndarray, bg: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    """Return singleton pixels and solid square blocks as component summaries."""
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


def _legend_points(
    g: np.ndarray, singles: list[tuple[int, int, int]]
) -> tuple[list[tuple[int, int, int]], int]:
    """Extract legend singleton pixels and convert them to logical lattice coordinates."""
    h, w = g.shape
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    diffs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    diffs += [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    assert diffs, "legend must span multiple rows or columns"
    step_l = reduce(gcd, diffs)
    assert step_l > 0, "legend lattice step must be positive"

    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)
    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // step_l
        cc = (c - cmin) // step_l
        legend_pts.append((int(rr), int(cc), int(v)))
    return legend_pts, step_l


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a scaled block mosaic from a legend that contains both an anchor and a scale marker.

    Input structure:
    - bg is the dominant background color.
    - The top-left corner contains isolated legend pixels on a regular lattice.
    - The main area contains exactly one visible square seed block whose color matches one unique legend pixel.
    - Another unique legend pixel has no matching visible block; its row is the symbolic scale code.

    Rule:
    - Use the color-matched legend/block pair as the anchor to recover the translation offset.
    - Decode the final enlargement factor from the non-anchor special marker's legend row number.
    - Expand every legend pixel into a same-colored square block of that decoded size on a gapless scaled lattice.

    Key insight:
    - The anchor and the scale clue are intentionally separated: the visible square only says where the mosaic is aligned,
      while the marker's row says how large every completed block must be.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    singles, blocks = _extract_singles_and_blocks(g, bg)
    assert singles and blocks, "grid must contain legend pixels and a visible anchor block"

    legend_pts, _step_l = _legend_points(g, singles)

    legend_by_color: dict[int, list[tuple[int, int]]] = {}
    for rr, cc, col in legend_pts:
        legend_by_color.setdefault(int(col), []).append((int(rr), int(cc)))

    blocks_by_color: dict[int, list[tuple[int, int, int]]] = {}
    for r0, c0, size, col in blocks:
        blocks_by_color.setdefault(int(col), []).append((int(r0), int(c0), int(size)))

    anchor_matches: list[tuple[int, int, int, int, int, int]] = []
    marker_candidates: list[tuple[int, int, int]] = []
    for col, pts in legend_by_color.items():
        if len(pts) != 1:
            continue
        rr, cc = pts[0]
        matches = blocks_by_color.get(int(col), [])
        if len(matches) == 1:
            br0, bc0, seed_size = matches[0]
            anchor_matches.append((int(col), int(rr), int(cc), int(br0), int(bc0), int(seed_size)))
        elif len(matches) == 0:
            marker_candidates.append((int(col), int(rr), int(cc)))

    assert len(anchor_matches) == 1, "grid must contain exactly one unique anchor color match"
    assert len(marker_candidates) == 1, "grid must contain exactly one unique scale marker"

    _anchor_col, ar, ac, abr, abc, seed_size = anchor_matches[0]
    _marker_col, mr, _mc = marker_candidates[0]
    assert seed_size >= 2, "visible anchor block must be at least 2x2"

    scale = mr + 1  # legend rows are read 1-based for the symbolic code
    assert scale in (3, 4), "marker row must encode scale 3 or 4"

    base_r = abr - ar * scale
    base_c = abc - ac * scale

    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * scale
        tc0 = base_c + cc * scale
        assert 0 <= tr0 <= h - scale and 0 <= tc0 <= w - scale, \
            "scaled legend projection must fit in canvas"
        out[tr0 : tr0 + scale, tc0 : tc0 + scale] = col
    return out


# --------------- generator (combo) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_ROWS = (4,)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE = 4
MAX_COUNT_PER_FILLER = 6

# === CFG: main-area scaling ===
SCALE_OPTIONS = (3, 4)
SEED_SIZE = 2
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 400


def _enumerate_staircase_shapes():
    """Enumerate all valid staircase width tuples."""
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


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    return [(r, c) for r, width in enumerate(widths) for c in range(width)]


def _neighbors4_in_pts(pts: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    pos_set = set(pts)
    neigh: dict[tuple[int, int], list[tuple[int, int]]] = {p: [] for p in pts}
    for r, c in pts:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (r + dr, c + dc)
            if q in pos_set:
                neigh[(r, c)].append(q)
    return neigh


def _filler_counts(n_remaining: int) -> tuple[int, int] | None:
    """Split remaining legend cells between two repeated filler colors."""
    if n_remaining < 4:
        return None
    c1 = 2
    c2 = 2
    extra = n_remaining - 4
    while extra:
        if c1 >= MAX_COUNT_PER_FILLER and c2 >= MAX_COUNT_PER_FILLER:
            return None
        if c1 >= MAX_COUNT_PER_FILLER:
            c2 += 1
        elif c2 >= MAX_COUNT_PER_FILLER:
            c1 += 1
        elif random.random() < 0.5:
            c1 += 1
        else:
            c2 += 1
        extra -= 1
    if random.random() < 0.5:
        c1, c2 = c2, c1
    return c1, c2


def _assign_legend_colors(
    pts: list[tuple[int, int]],
    anchor_pt: tuple[int, int],
    marker_pt: tuple[int, int],
    palette: list[int],
) -> dict[tuple[int, int], int] | None:
    """
    Color the staircase so:
    - anchor color is unique,
    - marker color is unique and distinct,
    - the other two colors repeat,
    - orthogonally adjacent cells never share a color.
    """
    neigh = _neighbors4_in_pts(pts)
    anchor_color, marker_color, fill_a, fill_b = palette
    fill_counts = _filler_counts(len(pts) - 2)
    if fill_counts is None:
        return None
    cnt_a, cnt_b = fill_counts

    colors = {
        anchor_pt: int(anchor_color),
        marker_pt: int(marker_color),
    }
    counts = {
        int(anchor_color): 1,
        int(marker_color): 1,
        int(fill_a): 0,
        int(fill_b): 0,
    }
    targets = {
        int(anchor_color): 1,
        int(marker_color): 1,
        int(fill_a): cnt_a,
        int(fill_b): cnt_b,
    }
    fillers = [int(fill_a), int(fill_b)]
    cells = [p for p in pts if p not in colors]

    def feasible(pos, col):
        if counts[col] >= targets[col]:
            return False
        for q in neigh[pos]:
            if colors.get(q) == col:
                return False
        return True

    def search():
        if not cells:
            return counts[int(fill_a)] == cnt_a and counts[int(fill_b)] == cnt_b
        best_i = None
        best_opts = None
        for i, pos in enumerate(cells):
            opts = []
            for col in fillers:
                if not feasible(pos, col):
                    continue
                rem_a = cnt_a - (counts[int(fill_a)] + (1 if col == int(fill_a) else 0))
                rem_b = cnt_b - (counts[int(fill_b)] + (1 if col == int(fill_b) else 0))
                if rem_a < 0 or rem_b < 0:
                    continue
                if rem_a + rem_b > len(cells) - 1:
                    continue
                opts.append(col)
            if best_opts is None or len(opts) < len(best_opts):
                best_i, best_opts = i, opts
        if not best_opts:
            return False
        pos = cells.pop(best_i)
        random.shuffle(best_opts)
        for col in best_opts:
            colors[pos] = col
            counts[col] += 1
            if search():
                return True
            counts[col] -= 1
            del colors[pos]
        cells.insert(best_i, pos)
        return False

    if not search():
        return None
    return colors


def _pick_special_points(
    pts: list[tuple[int, int]], scale: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Choose distinct anchor and marker points with marker row encoding the scale."""
    marker_row = scale - 1
    marker_candidates = [p for p in pts if p[0] == marker_row]
    if not marker_candidates:
        return None
    marker_pt = max(marker_candidates, key=lambda p: (p[1], -p[0]))

    anchor_candidates = [p for p in pts if p != marker_pt and p[0] != marker_row]
    if not anchor_candidates:
        anchor_candidates = [p for p in pts if p != marker_pt]
    if not anchor_candidates:
        return None
    anchor_pt = random.choice(anchor_candidates)
    return anchor_pt, marker_pt


def _gen_case():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    scale = random.choice(SCALE_OPTIONS)
    pts = _pick_staircase()
    special_pts = _pick_special_points(pts, scale)
    if special_pts is None:
        return None
    anchor_pt, marker_pt = special_pts

    bg = random.randrange(NUM_COLORS)
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], PALETTE_SIZE)
    legend_colors = _assign_legend_colors(pts, anchor_pt, marker_pt, palette)
    if legend_colors is None:
        return None

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - (max_r + 1) * scale
    c_hi = size - (max_c + 1) * scale
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        br = base_r + rr * scale
        bc = base_c + cc * scale
        out[br : br + scale, bc : bc + scale] = col

    inp = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        inp[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col

    ar, ac = anchor_pt
    anchor_color = int(legend_colors[anchor_pt])
    abr = base_r + ar * scale
    abc = base_c + ac * scale
    inp[abr : abr + SEED_SIZE, abc : abc + SEED_SIZE] = anchor_color

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    meta = {
        "scale": int(scale),
        "seed_size": int(SEED_SIZE),
        "anchor_pt": (int(ar), int(ac)),
        "marker_pt": (int(marker_pt[0]), int(marker_pt[1])),
        "marker_row_1based": int(marker_pt[0] + 1),
    }
    return inp, out, meta


def generate_input():
    for _ in range(MAX_RETRIES):
        case = _gen_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
