# Variant of 9385bd28
# Dimension: D1 — 匹配条件：标记对角线的配对方式 (match_cond)
# Change: diagonal_markers — 同色 L 形角标记位于外接矩形的对角，通过 L 形朝向判定哪条对角线是标记对角线、... → 颜色+形状联合匹配

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg


# Design analysis
# - Concrete appearance and behavior:
#   The bottom-left legend stays a vertical stack of `(marker_color, fill_color)` rows. The change is in the
#   scattered markers: a legend color no longer forms one 6-pixel L-pair. Instead, that color may have several
#   disconnected components, and exactly one pair of components shares both the color and the same small shape
#   (for example a T or a cross). Those two equal-shape components sit in opposite corners of a shared bounding
#   box. The opposite diagonal of that bbox still determines the diamond fill. A same-color singleton of another
#   shape may also appear; it is a deliberate distractor that breaks the original "group all pixels of one color"
#   assumption.
# - Small trace of the changed step:
#   Original logic: for color 3, collect all non-legend `3` pixels, make one bbox, inspect the four corners for
#   fixed L anchors, then fill the other diagonal. Variant logic: for color 3, split those pixels into connected
#   components, normalize each component's shape by translation, find the unique shape that appears exactly twice,
#   form the bbox from only that matched pair, then fill the other diagonal. The difference is exactly where D1
#   acts: matching now depends on `(color, shape)` rather than color plus a hard-coded L-corner detector.
# - Complexity calibration:
#   Shapes come from a small reusable catalog, so the task stays learnable from 2-4 examples. At least one active
#   legend entry per generated grid is forced to have a same-color wrong-shape distractor; otherwise the puzzle
#   could visually collapse back to "pair by color only". The catalog size is small enough that repetitions occur
#   across samples, giving humans evidence for the shape-matching rule.
# - Hidden assumptions audited against the original solver:
#   `_find_legend` and legend protection still hold because the legend format is unchanged. The original
#   `marker_pts = all pixels of one color` is no longer safe once same-color distractors exist, so the solver must
#   replace it with connected-component extraction. The original `_l_anchor_present(...)` and diagonal score
#   comparison rely on every active marker being an L corner; that assumption is false in this variant, so the
#   diagonal is instead recovered from which bbox corners are occupied by the two matched component bboxes. The
#   "protect current marker pixels" rule remains correct and now deliberately preserves distractors too.
# - Placement legality:
#   Each pair uses two identical components placed flush to opposite corners of a larger bbox. The bbox is chosen
#   large enough that the two components are disconnected in both row and column directions, so the solver sees two
#   components instead of one merged blob. All marker placements respect the legend exclusion mask and the occupied
#   mask, so no marker overlaps or legend corruption occurs.
# - Visual observability:
#   The crucial signal is visible: among all components of one legend color, only one shape repeats exactly twice,
#   and those two copies flank a filled diamond. This can be inferred directly from examples without relying on
#   color ids, ordering by numeric value, or invisible metadata.
# - Visual cleanliness and color inference:
#   The legend still provides the fill color, so no new color-transformation burden is introduced. Marker shapes are
#   compact and sparse, and same-color distractors are limited to single extra components, avoiding unreadable piles
#   of overlapping objects while still making the shape cue necessary.


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs at bottom-left.
    Each legend row: (marker_color at col c, fill_color at col c+1).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, marker, fill).
    """
    h, w = g.shape
    start_r = None
    start_c = None
    for r in range(h - 1, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size:
            start_r = r
            start_c = int(nz[0])
            break
    assert start_r is not None and start_c is not None and start_c < w - 1, (
        "legend must exist at bottom-left with at least 2 columns"
    )

    entries: list[tuple[int, int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1]) if c + 1 < w else bg
        entries.append((r, marker, fill))
    return c, entries


def _diamond_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """
    Mask for all cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neigh path between corners a and b:
      dist(a,p) + dist(p,b) == dist(a,b)
    """
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _component_info(pts: list[tuple[int, int]]) -> dict[str, object]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    rel = tuple(sorted((r - r0, c - c0) for r, c in pts))
    return {
        "pts": pts,
        "bbox": (r0, c0, r1, c1),
        "shape_key": rel,
        "size": (r1 - r0 + 1, c1 - c0 + 1),
    }


def _find_color_components(g: np.ndarray, color: int, legend_mask: np.ndarray) -> list[dict[str, object]]:
    comps = _components_from_mask((g == color) & (~legend_mask))
    return [_component_info(pts) for pts in comps]


def _is_solid_block(comp: dict[str, object]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    pts = comp["pts"]
    return (r1 - r0 == 1) and (c1 - c0 == 1) and len(pts) == 4


def _pair_diagonal(pair: list[dict[str, object]]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]:
    b0 = pair[0]["bbox"]
    b1 = pair[1]["bbox"]
    rmin = min(b0[0], b1[0])
    cmin = min(b0[1], b1[1])
    rmax = max(b0[2], b1[2])
    cmax = max(b0[3], b1[3])

    tl_count = sum(int(comp["bbox"][0] == rmin and comp["bbox"][1] == cmin) for comp in pair)
    br_count = sum(int(comp["bbox"][2] == rmax and comp["bbox"][3] == cmax) for comp in pair)
    tr_count = sum(int(comp["bbox"][0] == rmin and comp["bbox"][3] == cmax) for comp in pair)
    bl_count = sum(int(comp["bbox"][2] == rmax and comp["bbox"][1] == cmin) for comp in pair)

    if tl_count == 1 and br_count == 1 and tr_count == 0 and bl_count == 0:
        return (rmin, cmax), (rmax, cmin), (rmin, rmax, cmin, cmax)
    if tr_count == 1 and bl_count == 1 and tl_count == 0 and br_count == 0:
        return (rmin, cmin), (rmax, cmax), (rmin, rmax, cmin, cmax)
    raise AssertionError("matched components must occupy exactly one bbox diagonal")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the rectangle implied by a legend color and a same-shape marker pair.

    Input structure:
    - The grid has one dominant background color and a bottom-left vertical legend of `(marker_color, fill_color)` rows.
    - Away from the legend, each active legend color appears as disconnected marker components rather than fixed L-corners.
    - For a useful entry, exactly two components share both the legend color and the same small shape; optional same-color
      components of other shapes may also be present as distractors. Some entries are still simple 2x2 blocks or no-ops.

    Rule:
    - Read the legend from bottom to top and inspect only non-legend pixels of that marker color.
    - Split those pixels into 4-connected components, normalize each component by translation, and find the unique shape
      that occurs exactly twice. Those two matching components are the marker pair.
    - The pair sits on one diagonal of its combined bounding box, so fill the diamond on the opposite diagonal with the
      legend's fill color. Solid 2x2 blocks are recolored directly, and later legend markers remain protected.

    Key insight: Color chooses the candidate pool, but shape frequency identifies the actual pair; the repeated shape, not a
    fixed L orientation, tells which two components define the rectangle to fill.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True

    out = g.copy()
    marker_order = [m for _, m, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        components = _find_color_components(g, marker, legend_mask)
        if not components:
            continue

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        if len(components) == 1 and _is_solid_block(components[0]):
            block = np.zeros_like(g, dtype=bool)
            for r, c in components[0]["pts"]:
                block[r, c] = True
            out[block & (~protect)] = int(fill)
            continue

        protect |= (g == marker) & (~legend_mask)

        shape_groups: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = defaultdict(list)
        for comp in components:
            shape_groups[comp["shape_key"]].append(comp)

        pair_groups = [group for group in shape_groups.values() if len(group) == 2]
        assert len(pair_groups) == 1, "each active color must have exactly one repeated shape"
        assert all(len(group) in (1, 2) for group in shape_groups.values()), "unexpected component multiplicity"

        pair = pair_groups[0]
        a, b, (rmin, rmax, cmin, cmax) = _pair_diagonal(pair)
        if rmin == rmax or cmin == cmax:
            continue

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill)

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.25
BLOCK_PROB = 0.20
SHAPE_DISTRACTOR_PROB = 0.45

# ── CFG: pair bbox shape ───────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
PAIR_RETRIES = 180
SINGLE_RETRIES = 120
BLOCK_RETRIES = 100
GEN_RETRIES = 350
NONLEGEND_DISTRACTOR_PROB = 0.20

SHAPE_LIBRARY = {
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "PLUS": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    "ZIG": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "CUP": [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)],
}


def _shape_size(shape_name: str) -> tuple[int, int]:
    pts = SHAPE_LIBRARY[shape_name]
    return max(r for r, _ in pts) + 1, max(c for _, c in pts) + 1


def _translated_shape(shape_name: str, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr, dc in SHAPE_LIBRARY[shape_name]]


def _pair_cells(rmin: int, cmin: int, bh: int, bw: int, shape_name: str, diag: str) -> list[tuple[int, int]]:
    sh, sw = _shape_size(shape_name)
    if diag == "TLBR":
        a_top, a_left = rmin, cmin
        b_top, b_left = rmin + bh - sh, cmin + bw - sw
    else:
        a_top, a_left = rmin, cmin + bw - sw
        b_top, b_left = rmin + bh - sh, cmin
    return _translated_shape(shape_name, a_top, a_left) + _translated_shape(shape_name, b_top, b_left)


def _single_shape_cells(top: int, left: int, shape_name: str) -> list[tuple[int, int]]:
    return _translated_shape(shape_name, top, left)


def _pick_pair_bbox(H: int, W: int, shape_name: str) -> tuple[int, int] | None:
    sh, sw = _shape_size(shape_name)
    min_h = 2 * sh
    min_w = 2 * sw
    if min_h > H or min_w > W:
        return None
    bh = random.randint(min_h, min(BBOX_H_MAX, H))
    bw = random.randint(min_w, min(BBOX_W_MAX, W))
    return bh, bw


def _try_place_shape_pair(
    H: int,
    W: int,
    color: int,
    shape_name: str,
    retries: int,
    can_place,
    do_place,
) -> bool:
    for _ in range(retries):
        size = _pick_pair_bbox(H, W, shape_name)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < 0.7 else "TRBL"
        cells = _pair_cells(rmin, cmin, bh, bw, shape_name, diag)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _try_place_single_shape(
    H: int,
    W: int,
    color: int,
    shape_name: str,
    retries: int,
    can_place,
    do_place,
) -> bool:
    sh, sw = _shape_size(shape_name)
    if sh > H or sw > W:
        return False
    for _ in range(retries):
        top = random.randint(0, H - sh)
        left = random.randint(0, W - sw)
        cells = _single_shape_cells(top, left, shape_name)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _build_legend(H: int, W: int, bg: int):
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]):
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

    n_entries = len(legend_rows)
    types = ["pair"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"
    pair_indices = [i for i, t in enumerate(types) if t == "pair"]
    if not pair_indices:
        return None
    forced_shape_idx = random.choice(pair_indices)

    def can_place(cells):
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells, color):
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    used_shapes: dict[int, str] = {}
    for i, (_, marker, _) in enumerate(legend_rows):
        marker_type = types[i]
        if marker_type == "noop":
            continue
        if marker_type == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, marker)
                    placed = True
                    break
            if not placed:
                return None
            continue

        shape_name = random.choice(list(SHAPE_LIBRARY))
        used_shapes[i] = shape_name
        if not _try_place_shape_pair(H, W, marker, shape_name, PAIR_RETRIES, can_place, do_place):
            return None

        need_distractor = i == forced_shape_idx or random.random() < SHAPE_DISTRACTOR_PROB
        if need_distractor:
            distractor_shapes = [name for name in SHAPE_LIBRARY if name != shape_name]
            random.shuffle(distractor_shapes)
            placed = False
            for distractor_shape in distractor_shapes:
                if _try_place_single_shape(H, W, marker, distractor_shape, SINGLE_RETRIES, can_place, do_place):
                    placed = True
                    break
            if not placed:
                return None

    if random.random() < NONLEGEND_DISTRACTOR_PROB:
        avail_colors = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail_colors:
            distractor_color = random.choice(avail_colors)
            distractor_shape = random.choice(list(SHAPE_LIBRARY))
            _try_place_single_shape(H, W, distractor_color, distractor_shape, SINGLE_RETRIES, can_place, do_place)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    active_shape_distractors = 0
    for i, (_, marker, _) in enumerate(legend_rows):
        if types[i] != "pair":
            continue
        comps = _find_color_components(grid, marker, legend_mask)
        groups: dict[tuple[tuple[int, int], ...], int] = defaultdict(int)
        for comp in comps:
            groups[comp["shape_key"]] += 1
        if sum(v == 1 for v in groups.values()) >= 1 and sum(v == 2 for v in groups.values()) == 1:
            active_shape_distractors += 1
    if active_shape_distractors == 0:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
