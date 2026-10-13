from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import get_bg, label_mask
import random
from arc_utils import label_mask, bbox, crop_bbox, paste, random_free_location

def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes in a target object with colors from a source object's subsprites, then tight-crop.

    Input structure:
    - bg = background. Two similar 8-connected objects share an outline color.
    - Source has colored subsprites in cavities; target has empty cavities.

    Rule:
    - Find the two 8-connected objects.
    - Target = object with more hole area. Align subsprites to holes (rotation/flip if needed).
    - Fill each hole with its matched subsprite color; tight-crop.

    Key insight: Hole centers in the target are matched to source subsprite centers under the best rotation/flip transform.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _components(mask: np.ndarray) -> list[dict]:
        lab, n = label_mask(mask, connectivity=8)
        comps = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "cid": int(cid),
                    "pts": pts,
                    "size": int(len(pts)),
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
        comps.sort(key=lambda d: int(d["size"]), reverse=True)
        return comps

    def _extract_two_objects():
        comps = _components(g != int(bg))
        if len(comps) < 2:
            return [comps]
        return [[comps[0]], [comps[1]]]

    def _bbox_of(group: list[dict]) -> tuple[int, int, int, int]:
        r0 = min(int(d["bbox"][0]) for d in group)
        c0 = min(int(d["bbox"][1]) for d in group)
        r1 = max(int(d["bbox"][2]) for d in group)
        c1 = max(int(d["bbox"][3]) for d in group)
        return int(r0), int(c0), int(r1), int(c1)

    def _extract_group_subgrid(group: list[dict]) -> np.ndarray:
        r0, c0, r1, c1 = _bbox_of(group)
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for d in group:
            for rr, cc in d["pts"]:
                sub[int(rr) - int(r0), int(cc) - int(c0)] = int(g[int(rr), int(cc)])
        return sub

    def _enclosed_bg_components(sub: np.ndarray) -> list[dict]:
        m = sub == int(bg)
        lab, n = label_mask(m, connectivity=4)
        Hs, Ws = sub.shape
        holes = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if bool(r0 == 0 or c0 == 0 or r1 == int(Hs - 1) or c1 == int(Ws - 1)):
                continue
            cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
            holes.append({"pts": pts, "cent": cent, "size": int(len(pts))})
        return holes

    def _outline_col(sub: np.ndarray) -> int:
        cnt = Counter(map(int, sub.flatten()))
        opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
        return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)

    def _colored_components(sub: np.ndarray) -> list[dict]:
        outc = _outline_col(sub)
        m = (sub != int(bg)) & (sub != int(outc))
        lab, n = label_mask(m, connectivity=4)
        subs = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
            cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
            subs.append({"cent": cent, "color": int(col)})
        return subs

    def _rank_keys(pts: list[tuple[int, int]]):
        rows = sorted(set(int(r) for r, _c in pts))
        cols = sorted(set(int(c) for _r, c in pts))
        rrank = {r: i for i, r in enumerate(rows)}
        crank = {c: i for i, c in enumerate(cols)}
        keys = [(int(rrank[r]), int(crank[c])) for r, c in pts]
        return keys, rrank, crank

    def _transform_point(p: tuple[int, int], H0: int, W0: int, k: int, flip: bool) -> tuple[int, int]:
        r, c = int(p[0]), int(p[1])
        if int(k) == 0:
            rr, cc = int(r), int(c)
            H2, W2 = int(H0), int(W0)
        elif int(k) == 1:
            rr, cc = int(W0 - 1 - c), int(r)
            H2, W2 = int(W0), int(H0)
        elif int(k) == 2:
            rr, cc = int(H0 - 1 - r), int(W0 - 1 - c)
            H2, W2 = int(H0), int(W0)
        else:
            rr, cc = int(c), int(H0 - 1 - r)
            H2, W2 = int(W0), int(H0)
        if bool(flip):
            cc = int(W2 - 1 - cc)
        return int(rr), int(cc)

    groups = _extract_two_objects()
    assert not (len(groups) < 2)
    objs = [_extract_group_subgrid(grp) for grp in groups]

    holes_list = [_enclosed_bg_components(o) for o in objs]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest)
    src_subs = _colored_components(src)
    if not dest_holes or not src_subs:
        m = dest != int(bg)
        pts = np.argwhere(m)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return dest[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best = None
    for k in range(4):
        for flip in [False, True]:
            tpts = [_transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip)) for s in src_subs]
            tkeys, _s_rr, _s_cc = _rank_keys(tpts)
            if sorted(tkeys) == sorted(dest_keys):
                best = (int(k), bool(flip))
                break
        if best is not None:
            break
    if best is None:
        best = (0, False)
    k, flip = best

    # Build ordered lists and pair them deterministically.
    dest_list = []
    for h in dest_holes:
        key = (int(d_rr[int(h["cent"][0])]), int(d_cc[int(h["cent"][1])]))
        dest_list.append((key, h["cent"], h))
    dest_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    tpts_all = [_transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip)) for s in src_subs]
    _keys, s_rr, s_cc = _rank_keys(tpts_all)

    src_list = []
    for s in src_subs:
        tp = _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, int(s["color"])))
    src_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    for (_k1, _p1, col), (_k2, _p2, h) in zip(src_list, dest_list):
        for rr, cc in h["pts"]:
            dest[int(rr), int(cc)] = int(col)

    # Tight crop to non-background.
    m = dest != int(bg)
    pts = np.argwhere(m)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return dest[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

# --------------- generator (Claude#1+refactor) ---------------

CANVAS_SIZE_RANGE = (22, 30)           # canvas H/W range
OBJECT_H_RANGE = (7, 17)               # src/dst object height bounds
OBJECT_W_RANGE = (7, 13)               # src/dst object width bounds

# Cavity structure
CAVITY_COUNT_RANGE = (4, 9)            # number of cavities per object
FILL_COLOR_COUNT_RANGE = (3, 7)        # number of distinct fill colors

# Layout row patterns (weighted choices)
ROW_HEIGHT_CHOICES = [1, 2, 2, 3, 3, 3]           # per-row cavity height
LAYOUT_K_CHOICES_FULL = [1, 2, 2, 3, 3]           # k=1 centered, k=2 pair, k=3 L+C+R
LAYOUT_K_CHOICES_SMALL = [1, 2, 2]                # when remaining budget <= 2
ROW_GAP_CHOICES = [1, 1, 2]                       # empty rows between cavity rows

# Cavity slot widths per layout type
CENTERED_SLOT_WIDTHS = [3, 3, 5, 5, 7]            # k=1 (odd widths only)
PAIR_SLOT_WIDTHS = [2, 2, 3, 3, 4]                # k=2 per-slot width
SIDE_SLOT_WIDTHS = [2, 2, 3]                      # k=3 side slot width
MID_SLOT_WIDTHS = [1, 3]                          # k=3 center slot width (odd)
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]                 # k=2 empty cols at axis (forced odd)
TRIO_MID_SIDE_GAP = 1                             # k=3 empty cols between center and side

# Shape mutation
NONRECT_PROBABILITY = 0.8              # P(cavity is non-rectangular, else solid rect)
MAX_MUTATE_ATTEMPTS = 40               # iterations per shape mutation

# Color assignment
COLOR_MISMATCH_RATE = 0.10             # P(cavity breaks mirror-pair color sharing)

# Placement
PLACEMENT_BORDER = 1                   # min distance from canvas border
PLACEMENT_PADDING = 2                  # min gap between src and dst objects

# Generation retry
MAX_GEN_ATTEMPTS = 300                 # outer retry loop for generate_input()
MAX_LAYOUT_R_BUDGET = 24               # layout design vertical budget (before crop)


# =============================================================================
# Shape primitives
# =============================================================================

def _random_rect_shape(h: int, w: int) -> np.ndarray:
    """Solid rectangle h x w of booleans."""
    return np.ones((h, w), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int, symmetric: bool = False) -> np.ndarray:
    """Remove boundary cells from a rect mask while preserving 4-connectivity.
    Keeps the original (h, w) bbox; removed cells become False but array shape is unchanged.
    If symmetric=True, mirror-pair removals so the mask stays LR-symmetric.
    """
    m = mask.copy()
    H, W = m.shape
    n_remove = random.randint(1, max(1, n_remove_max))
    tried = 0
    removed = 0
    while removed < n_remove and tried < MAX_MUTATE_ATTEMPTS:
        tried += 1
        # Find boundary cells (cells with <=2 in-mask 4-neighbors).
        cand = []
        for r in range(H):
            for c in range(W):
                if not m[r, c]:
                    continue
                nb = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and m[nr, nc]:
                        nb += 1
                if nb <= 2:
                    cand.append((r, c))
        if not cand:
            break
        r, c = random.choice(cand)
        to_remove = [(r, c)]
        if symmetric:
            mc = W - 1 - c
            if (r, mc) != (r, c):
                to_remove.append((r, mc))
        m2 = m.copy()
        for rr, cc in to_remove:
            m2[rr, cc] = False
        if not m2.any():
            continue
        _, n_components = label_mask(m2, connectivity=4)
        if int(n_components) == 1:
            m = m2
            removed += len(to_remove)
    if not m.any():
        return mask.copy()
    return m


def _gen_shape(h: int, w: int, *, nonrect: bool, symmetric: bool = False) -> np.ndarray:
    """Build a cavity shape. If symmetric=True, the result is LR-symmetric."""
    m = _random_rect_shape(h, w)
    if nonrect and h * w >= 3:
        # Remove up to half the area; symmetric mutations consume cells in pairs.
        limit = max(1, (h * w) // 2)
        m = _mutate_shape(m, n_remove_max=limit, symmetric=symmetric)
    return m


def _mirror_shape(m: np.ndarray) -> np.ndarray:
    return m[:, ::-1].copy()


def _gen_dst_src_pair(h: int, w: int, *, symmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    """Independently mutate two cavity shapes (dst + src) sharing the same (h, w) bbox.
    This drives src subsprites and dst holes to have visibly different shapes, as required
    by the spec (the task matches by centroid rank, not shape).
    """
    nonrect = random.random() < NONRECT_PROBABILITY
    return (_gen_shape(h, w, nonrect=nonrect, symmetric=symmetric),
            _gen_shape(h, w, nonrect=nonrect, symmetric=symmetric))


# =============================================================================
# Layout design — LR-symmetric cavity arrangements
# =============================================================================
#
# Slot dict fields:
#   r0          : top row in layout coordinate space
#   col_offset  : leftmost column relative to the LR symmetry axis
#   shape_dst   : boolean mask for dst's version of this cavity
#   shape_src   : boolean mask for src's version (independently mutated)
#   pair_id     : mirror-pair identifier — slots sharing a pair_id are LR mirrors
#                 (used to give them matching fill colors by default)
# =============================================================================

def _add_centered_slot(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    """k=1: single cavity centered on the LR axis (odd width, LR-symmetric shape)."""
    w = random.choice(CENTERED_SLOT_WIDTHS)
    dst, src = _gen_dst_src_pair(row_h, w, symmetric=True)
    slots.append({
        'r0': cur_r, 'col_offset': -(w // 2),
        'shape_dst': dst, 'shape_src': src, 'pair_id': pair_id,
    })


def _add_mirror_pair_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    """k=2: left + right mirror pair with an odd empty-column gap at the axis."""
    w = random.choice(PAIR_SLOT_WIDTHS)
    dst_L, src_L = _gen_dst_src_pair(row_h, w, symmetric=False)
    dst_R = _mirror_shape(dst_L)
    src_R = _mirror_shape(src_L)
    axis_gap = random.choice(PAIR_AXIS_GAP_CHOICES)
    if axis_gap % 2 == 0:
        axis_gap += 1  # force odd so axis lies on a single column
    half = axis_gap // 2
    slots.append({
        'r0': cur_r, 'col_offset': -(half + w),
        'shape_dst': dst_L, 'shape_src': src_L, 'pair_id': pair_id,
    })
    slots.append({
        'r0': cur_r, 'col_offset': half + 1,
        'shape_dst': dst_R, 'shape_src': src_R, 'pair_id': pair_id,
    })


def _add_trio_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    """k=3: L + centered + R. Center slot has odd width and uses its own pair_id."""
    w_side = random.choice(SIDE_SLOT_WIDTHS)
    w_mid = random.choice(MID_SLOT_WIDTHS)
    dst_L, src_L = _gen_dst_src_pair(row_h, w_side, symmetric=False)
    dst_R = _mirror_shape(dst_L)
    src_R = _mirror_shape(src_L)
    dst_M, src_M = _gen_dst_src_pair(row_h, w_mid, symmetric=True)
    mid_half = w_mid // 2
    slots.append({
        'r0': cur_r, 'col_offset': -(mid_half + w_side + TRIO_MID_SIDE_GAP),
        'shape_dst': dst_L, 'shape_src': src_L, 'pair_id': pair_id,
    })
    slots.append({
        'r0': cur_r, 'col_offset': -mid_half,
        'shape_dst': dst_M, 'shape_src': src_M, 'pair_id': pair_id + 1,
    })
    slots.append({
        'r0': cur_r, 'col_offset': mid_half + TRIO_MID_SIDE_GAP + 1,
        'shape_dst': dst_R, 'shape_src': src_R, 'pair_id': pair_id,
    })


def _pick_layout_k(budget: int) -> int:
    """Pick the layout pattern for the current row, given remaining cavity budget."""
    if budget == 1:
        return 1
    if budget == 2:
        return random.choice(LAYOUT_K_CHOICES_SMALL)
    return random.choice(LAYOUT_K_CHOICES_FULL)


def _design_layout(n_target: int) -> list[dict]:
    """Return an LR-symmetric list of cavity slots with a total count near n_target."""
    slots: list[dict] = []
    cur_r = 0
    pair_id = 0
    budget = n_target
    while len(slots) < n_target and cur_r < MAX_LAYOUT_R_BUDGET and budget > 0:
        row_h = random.choice(ROW_HEIGHT_CHOICES)
        k = _pick_layout_k(budget)
        if k == 1:
            _add_centered_slot(slots, cur_r, pair_id, row_h)
            pair_id += 1
            budget -= 1
        elif k == 2:
            _add_mirror_pair_slots(slots, cur_r, pair_id, row_h)
            pair_id += 1
            budget -= 2
        else:
            _add_trio_slots(slots, cur_r, pair_id, row_h)
            pair_id += 2  # trio uses 2 pair_ids (outer pair + center)
            budget -= 3
        cur_r += row_h + random.choice(ROW_GAP_CHOICES)
    return slots


# =============================================================================
# Rendering — turn a layout + shapes + colors into an object patch
# =============================================================================

def _render_object(
    layout: list[dict], shapes: list[np.ndarray], colors: list[int],
    bg: int, outline_col: int, *, filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Render an object (src or dst) onto a tight canvas.

    Returns (cropped, filled_cropped):
      - cropped        : the object, with cavities painted bg if filled=False else fill-colored
      - filled_cropped : same object, ALWAYS with cavities fill-colored (used as the
                         expected main() output for dst)

    Returns (None, None) if there are no fill cells.
    """
    # Compute layout bbox across all cavity cells.
    placed_slots = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot['r0'] + rr, slot['col_offset'] + cc))
    if not placed_slots:
        return None, None

    all_rows = np.concatenate([p[1] for p in placed_slots])
    all_cols = np.concatenate([p[2] for p in placed_slots])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())

    pad = 2
    H = (r_max - r_min + 1) + 2 * pad
    W = (c_max - c_min + 1) + 2 * pad

    canvas = np.full((H, W), bg, dtype=int)
    fill_mask = np.zeros((H, W), dtype=bool)
    for slot_idx, rows, cols in placed_slots:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        canvas[rr, cc] = colors[slot_idx]
        fill_mask[rr, cc] = True

    # Vectorized 4-neighbor outline = cells adjacent to any fill cell but not fill themselves.
    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :]  |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:]  |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    canvas[outline_mask] = outline_col

    filled_grid = canvas.copy()
    if not filled:
        canvas[fill_mask] = bg

    b = bbox(canvas != bg)
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0b, c0b, r1b, c1b = b
    filled_cropped = filled_grid[r0b:r1b + 1, c0b:c1b + 1]
    return cropped, filled_cropped


# =============================================================================
# Validation helpers
# =============================================================================

def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    """Count 4-connected non-bg non-outline components (i.e. filled cavities)."""
    mask = (filled_grid != bg) & (filled_grid != outline_col)
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    """Count 4-connected bg components fully enclosed by non-bg cells (not touching border)."""
    mask = (empty_grid == bg)
    lab, n = label_mask(mask, connectivity=4)
    H, W = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 > 0 and c0 > 0 and r1 < H - 1 and c1 < W - 1:
            count += 1
    return count


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != bg
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return (OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1]
            and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1])


# =============================================================================
# Color assignment
# =============================================================================

def _assign_pair_colors(layout: list[dict], fill_pool: list[int]) -> list[int]:
    """Return one color per slot. Mirror-pair slots (same pair_id) share a color by
    default, with COLOR_MISMATCH_RATE fraction of cavities randomly broken to differ.
    """
    unique_pair_ids: list[int] = []
    for slot in layout:
        if slot['pair_id'] not in unique_pair_ids:
            unique_pair_ids.append(slot['pair_id'])
    n_pairs = len(unique_pair_ids)
    n_colors = len(fill_pool)

    # Ensure all pool colors are used when n_pairs >= n_colors; otherwise
    # use a subset of pool colors (one per pair).
    if n_pairs >= n_colors:
        pair_colors = list(fill_pool) + [random.choice(fill_pool)
                                         for _ in range(n_pairs - n_colors)]
    else:
        pair_colors = list(fill_pool)[:n_pairs]
    random.shuffle(pair_colors)

    pair_color_map = dict(zip(unique_pair_ids, pair_colors))
    colors = [pair_color_map[slot['pair_id']] for slot in layout]

    # Occasional mismatch: break LR color symmetry in a fraction of cavities.
    for i in range(len(colors)):
        if random.random() < COLOR_MISMATCH_RATE:
            colors[i] = random.choice(fill_pool)
    return colors


# =============================================================================
# Generation pipeline
# =============================================================================

def _pick_bg_and_palette() -> tuple[int, list[int]]:
    """Return (bg_color, shuffled_non_bg_palette)."""
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _build_object_pair(
    layout: list[dict], colors: list[int], bg: int, outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Render src (filled) + dst (empty) from the layout and validate both.
    Returns (src_cropped, dst_cropped, expected_output) or None on failure.
    """
    n_cavities = len(layout)
    dst_shapes = [slot['shape_dst'] for slot in layout]
    src_shapes = [slot['shape_src'] for slot in layout]

    dst_cropped, dst_filled = _render_object(
        layout, dst_shapes, colors, bg, outline_col, filled=False,
    )
    if dst_cropped is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _ = _render_object(
        layout, src_shapes, colors, bg, outline_col, filled=True,
    )
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n_cavities:
        return None
    if not _size_in_spec(src_cropped):
        return None

    # Expected output = dst with holes filled by matched colors, tight-cropped.
    return src_cropped, dst_cropped, dst_filled


def _place_on_canvas(
    src: np.ndarray, dst: np.ndarray, bg: int,
) -> np.ndarray | None:
    """Place src and dst on a random-size canvas with a mandatory gap between them."""
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    try:
        r, c = random_free_location(canvas, src, transparent=bg,
                                    border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING)
    except ValueError:
        return None
    paste(canvas, src, r, c, transparent=bg)
    try:
        r, c = random_free_location(canvas, dst, transparent=bg,
                                    border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING)
    except ValueError:
        return None
    paste(canvas, dst, r, c, transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = palette[0]

    layout = _design_layout(random.randint(*CAVITY_COUNT_RANGE))
    if len(layout) < CAVITY_COUNT_RANGE[0]:
        return None
    n_cavities = len(layout)

    fc_lo, fc_hi = FILL_COLOR_COUNT_RANGE
    max_cols = min(fc_hi, n_cavities, len(palette) - 1)
    if max_cols < fc_lo:
        return None
    n_fill_cols = random.randint(fc_lo, max_cols)
    fill_pool = palette[1:1 + n_fill_cols]
    colors = _assign_pair_colors(layout, fill_pool)

    built = _build_object_pair(layout, colors, bg, outline_col)
    if built is None:
        return None
    src, dst, expected = built

    canvas = _place_on_canvas(src, dst, bg)
    if canvas is None:
        return None

    # Oracle check against the solver.
    try:
        actual = main(canvas)
    except Exception:
        return None
    if not isinstance(actual, np.ndarray):
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError(f'generate_input: failed after {MAX_GEN_ATTEMPTS} attempts')
