# Combo variant of 6e4f6532 (index 18)
# combo_spec: D1/0+D2/0+D4/0
# Name: 颜色配对+形状定向+多色车站

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# a) Concretization and behavior trace:
#    Original task:
#    - source trains = 8-body + monochrome 9 couplers + stripe-colored markers
#    - destination stations = isolated 9-clusters
#    - matching key = coupler count; orientation key = stripe-marker directions
#    Combo task:
#    - source trains = 8-body + one multi-color coupler cluster + one attached
#      singleton label pixel
#    - destination stations = one isolated multi-color core cluster + one nearby
#      singleton label pixel of the same color as exactly one train label
#    - matching key = label color, not count
#    - orientation key = the target station core's asymmetric colored polyomino,
#      not stripe markers
#    Concrete trace on a sample with sizes (4,4,5):
#    - two size-4 station cores deliberately share the same colored coupler
#      template, so neither count nor coupler signature alone distinguishes them
#    - the train labeled color 6 must dock to the station whose nearby singleton
#      label is also 6
#    - once matched, the train's colored coupler cluster is rotated/flipped until
#      it exactly overlaps that station core under a unique D4 transform
#    - the train is erased from its scattered source and redrawn docked, while the
#      station core and station label remain in place
# b) Complexity calibration:
#    - Station sizes are drawn from repeated patterns such as (4,4), (4,4,5), and
#      (5,5,6), so the original count rule is visibly broken in every sample.
#    - Repeated sizes reuse the same multi-color coupler template, which prevents
#      the puzzle from collapsing into "just compare signatures".
#    - Multi-color cores use only two shared colors per sample to keep the grids
#      readable; the generator uses two unique label colors per sample.
# c) Hidden assumptions audit versus the original solver:
#    - Any `grid == 9` / "skip 9s" logic no longer applies because couplers and
#      stations are multi-color.
#    - Any reliance on stripe-colored train markers is invalid because markers are
#      removed entirely.
#    - Any count-based lookup `groups_by_size[len(couplers)]` is invalid because
#      multiple destinations may share the same size and even the same coupler
#      template.
#    - New parsing assumptions introduced here:
#      1. isolated non-8, non-stripe components of size >= 4 are station cores;
#      2. isolated singleton non-8, non-stripe components near a station core are
#         station labels;
#      3. inside each train component, the non-8/non-stripe pixels split into one
#         multi-cell coupler cluster and one singleton label pixel.
#      The generator explicitly constructs inputs to satisfy those assumptions.
# d) Placement legality:
#    - Output-first generation is preserved.
#    - Every docked patch fits inside the single inner region, station cores keep
#      >=2-cell clearance from stripes, station labels stay separated from station
#      cores by at least one background gap, and scattered trains keep >=1-cell
#      clearance from stations and from one another.
# e) Visual observability:
#    - The visible rule change is strong in 2-3 examples:
#      repeated station sizes break count matching, no stripe markers remain, and
#      stations are clearly multi-color rather than monochrome 9-clusters.
#    - The only stable pairing cue is the repeated singleton label color.
#    - The only stable orientation cue is the asymmetric station-core geometry.
# f) Visual cleanliness:
#    - Station cores use just two shared colors; train labels use distinct colors;
#      stripe colors remain separate. This keeps body/coupler/label/stripe roles
#      easy to parse by eye.
#    - Station labels are near their cores but not connected, so the relationship
#      is visible without destroying component separation.
# g) Change inferability:
#    - Color is used in a small, structured way: two core colors for all station
#      cores plus two unique label colors. The examples therefore expose a clear
#      "this singleton color pairs this train to this station" rule instead of an
#      unconstrained recoloring puzzle.
# 9) New solver assumptions checked against the generator:
#    - every sample has at least one repeated station size
#    - repeated-size stations reuse the same coupler template
#    - each station core has exactly one nearby singleton label
#    - each train component has exactly one singleton label and one coupler cluster
#    - each matched train/station pair has exactly one legal docking orientation


OPS = [
    (False, 0),
    (False, 1),
    (False, 2),
    (False, 3),
    (True, 0),
    (True, 1),
    (True, 2),
    (True, 3),
]
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
NON_BODY_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_point(
    r: int, c: int, shape: tuple[int, int], flip_ud: bool = False, rot_k: int = 0
) -> tuple[int, int]:
    h, w = map(int, shape)
    rr, cc = int(r), int(c)
    if flip_ud:
        rr = h - 1 - rr
    for _ in range(int(rot_k) % 4):
        rr, cc = w - 1 - cc, rr
        h, w = w, h
    return int(rr), int(cc)


def transform_points(points, shape, flip_ud: bool = False, rot_k: int = 0):
    return [transform_point(r, c, shape, flip_ud, rot_k) for r, c in points]


def transform_colored_points(points, shape, flip_ud: bool = False, rot_k: int = 0):
    return [
        (*transform_point(r, c, shape, flip_ud, rot_k), int(color))
        for r, c, color in points
    ]


def transformed_shape(shape: tuple[int, int], flip_ud: bool = False, rot_k: int = 0):
    return apply_transform(np.zeros(shape, dtype=int), flip_ud, rot_k).shape


def _bbox(coords):
    rs = [int(item[0]) for item in coords]
    cs = [int(item[1]) for item in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _dims(coords):
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _normalize_plain(coords):
    r0, c0, _, _ = _bbox(coords)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _normalize_colored(coords):
    r0, c0, _, _ = _bbox(coords)
    return frozenset((int(r - r0), int(c - c0), int(color)) for r, c, color in coords)


def _transform_shape(coords, op):
    plain = [(r, c) for r, c in coords]
    out = transform_points(plain, _dims(plain), *op)
    r0, c0, _, _ = _bbox(out)
    return {(int(r - r0), int(c - c0)) for r, c in out}


def _transform_colored_shape(coords, op):
    plain = [(r, c) for r, c, _ in coords]
    out = transform_colored_points(coords, _dims(plain), *op)
    r0, c0, _, _ = _bbox(out)
    return [(int(r - r0), int(c - c0), int(color)) for r, c, color in out]


def _adjacent8(p, cells):
    r, c = p
    return any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in cells)


def _dilate(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _path_to(body, goal, blocked, bounds):
    if goal in body:
        return True
    r0, r1, c0, c1 = bounds
    q = deque(body)
    prev = {p: None for p in body}
    dirs = DIRS4[:]
    random.shuffle(dirs)
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            p = (r + dr, c + dc)
            if p in prev or p in blocked:
                continue
            if not (r0 <= p[0] <= r1 and c0 <= p[1] <= c1):
                continue
            prev[p] = (r, c)
            if p == goal:
                cur = p
                while cur is not None:
                    body.add(cur)
                    cur = prev[cur]
                return True
            q.append(p)
    return False


def _connect_couplers_to_body(couplers, body, blocked, bounds):
    for cp in couplers:
        if _adjacent8(cp, body):
            continue
        cr, cc = cp
        choices = [
            (cr + dr, cc + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (cr + dr, cc + dc) not in blocked
        ]
        random.shuffle(choices)
        for q in choices:
            if _path_to(body, q, blocked, bounds):
                break
        else:
            return False
    return True


def _grow_body(body, blocked, target, body_min, body_max, patch_h_range, patch_w_range, iters):
    for _ in range(iters):
        all_cells = body | blocked
        h, w = _dims(all_cells)
        if len(body) >= target and h >= patch_h_range[0] and w >= patch_w_range[0]:
            break
        cands = []
        for r, c in body:
            for dr, dc in DIRS4:
                q = (r + dr, c + dc)
                if q in body or q in blocked:
                    continue
                nh, nw = _dims(all_cells | {q})
                if nh <= patch_h_range[1] and nw <= patch_w_range[1]:
                    cands.append(q)
        if not cands or len(body) >= body_max:
            break
        if h < patch_h_range[0]:
            r0, _, r1, _ = _bbox(all_cells)
            exp = [q for q in cands if q[0] < r0 or q[0] > r1]
            cands = exp or cands
        elif w < patch_w_range[0]:
            _, c0, _, c1 = _bbox(all_cells)
            exp = [q for q in cands if q[1] < c0 or q[1] > c1]
            cands = exp or cands
        body.add(random.choice(cands))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock labeled trains onto labeled multi-color stations, using the target station core to determine orientation.

    Input structure:
    - Three colored edge stripes surround one inner region.
    - Each train is a component containing color-8 body cells, one multi-color coupler cluster, and one attached singleton label pixel.
    - Each destination station consists of an isolated multi-color core cluster plus one nearby isolated singleton label pixel of the same color as exactly one train label.

    Rule:
    - Detect train components from objects containing 8, and split each train's non-body, non-stripe pixels into one multi-cell coupler cluster and one singleton label.
    - Detect station cores as isolated non-body, non-stripe components of size >= 4, and pair each core with its nearby singleton station label.
    - Match train to station by label color, then rotate/flip the train so its multi-color coupler cluster exactly overlaps the matched station core in the unique legal orientation.
    - Erase the original train and redraw only its non-coupler pixels at the docked location; station cores and station labels stay in place.

    Key insight: repeated station sizes and repeated coupler templates destroy the old count rule, so the singleton label chooses the destination while the asymmetric multi-color station core fixes the docking orientation.
    """
    grid = np.array(grid, dtype=int, copy=True)
    H, W = grid.shape
    bg = get_bg(grid)

    def find_edge_stripes():
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()

        top_h = 0
        for r in range(H):
            row = grid[r, :]
            if (row == row[0]).all() and row[0] != bg:
                top_h += 1
            else:
                break
        if top_h:
            stripe[:top_h, :] = True
            colors.add(int(grid[0, 0]))

        bot_h = 0
        for r in range(H - 1, -1, -1):
            row = grid[r, :]
            if (row == row[0]).all() and row[0] != bg:
                bot_h += 1
            else:
                break
        if bot_h:
            stripe[H - bot_h :, :] = True
            colors.add(int(grid[H - 1, 0]))

        inner_r0, inner_r1 = top_h, H - bot_h

        left_w = 0
        for c in range(W):
            col = grid[inner_r0:inner_r1, c]
            if col.size == 0:
                break
            if (col == col[0]).all() and col[0] != bg:
                left_w += 1
            else:
                break
        if left_w:
            stripe[inner_r0:inner_r1, :left_w] = True
            colors.add(int(grid[inner_r0, 0]))

        right_w = 0
        for c in range(W - 1, -1, -1):
            col = grid[inner_r0:inner_r1, c]
            if col.size == 0:
                break
            if (col == col[0]).all() and col[0] != bg:
                right_w += 1
            else:
                break
        if right_w:
            stripe[inner_r0:inner_r1, W - right_w :] = True
            colors.add(int(grid[inner_r0, W - 1]))
        return stripe, colors

    def regions_from_stripes(stripe_mask):
        non = ~stripe_mask
        lab, n = label(non)
        regs = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if (r1 - r0 + 1) >= 5 and (c1 - c0 + 1) >= 5:
                regs.append({"r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1)})
        return regs

    def isolated_nonbody_components(stripe_colors):
        mask = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                v = int(grid[r, c])
                if v == bg or v == 8 or v in stripe_colors:
                    continue
                if any(
                    0 <= r + dr < H
                    and 0 <= c + dc < W
                    and int(grid[r + dr, c + dc]) == 8
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if dr or dc
                ):
                    continue
                mask[r, c] = True
        struct = np.ones((3, 3), dtype=int)
        lab, n = label(mask, structure=struct)
        comps = []
        for k in range(1, n + 1):
            coords = [(int(r), int(c)) for r, c in np.argwhere(lab == k)]
            comps.append(
                {
                    "coords": coords,
                    "coord_set": set(coords),
                    "colors": tuple(int(grid[r, c]) for r, c in coords),
                }
            )
        return comps

    def station_label_for_core(core, singletons):
        matches = []
        for comp in singletons:
            lr, lc = comp["coords"][0]
            if any(
                abs(lr - r) + abs(lc - c) == 2 and (lr == r or lc == c)
                for r, c in core["coords"]
            ):
                matches.append(comp)
        assert len(matches) == 1, "Each station core must have exactly one nearby singleton label"
        return matches[0]

    def choose_unique_op(train, target_core):
        # D2/0 requires orientation to come from station geometry rather than
        # from any color arrangement, so we choose the unique D4 transform using
        # only the plain asymmetric shape. The later landing assertion still
        # checks that the multi-color coupler colors also match the station.
        target_sig = _normalize_plain([(r, c) for r, c, _ in target_core])
        matches = []
        for op in OPS:
            transformed = transform_colored_points(train["couplers"], train["bbox"].shape, *op)
            if _normalize_plain([(r, c) for r, c, _ in transformed]) == target_sig:
                matches.append(op)
        assert len(matches) == 1, "Target station core must determine a unique orientation"
        return matches[0]

    edge_mask, stripe_colors = find_edge_stripes()
    stripe_mask = edge_mask
    regions = regions_from_stripes(stripe_mask)
    assert len(regions) == 1, "Generator guarantees a single inner region"
    region = regions[0]

    isolated = isolated_nonbody_components(stripe_colors)
    station_cores = [comp for comp in isolated if len(comp["coords"]) >= 4]
    station_labels = [comp for comp in isolated if len(comp["coords"]) == 1]
    assert station_cores and station_labels

    stations_by_label = {}
    station_core_cells = set()
    for core in station_cores:
        label_comp = station_label_for_core(core, station_labels)
        label_color = int(grid[label_comp["coords"][0]])
        assert label_color not in stations_by_label, "Station label colors must be unique"
        stations_by_label[label_color] = {
            "core": [(r, c, int(grid[r, c])) for r, c in core["coords"]],
            "core_set": set(core["coords"]),
            "label_coord": label_comp["coords"][0],
        }
        station_core_cells.update(core["coords"])

    comp_mask = (grid != bg) & (~stripe_mask)
    lab, n = label(comp_mask)
    trains = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        vals = [int(grid[r, c]) for r, c in coords]
        if 8 not in vals:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        rel_mask = np.zeros_like(bbox, dtype=bool)
        abs_coords = [(int(r), int(c)) for r, c in coords]
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v != 8 and v != bg and v not in stripe_colors:
                rel_mask[r - r0, c - c0] = True
        sub_lab, sub_n = label(rel_mask, structure=np.ones((3, 3), dtype=int))
        pieces = []
        for kk in range(1, sub_n + 1):
            rel_coords = [(int(r), int(c)) for r, c in np.argwhere(sub_lab == kk)]
            pieces.append(rel_coords)
        big = [p for p in pieces if len(p) >= 4]
        singles = [p for p in pieces if len(p) == 1]
        assert len(big) == 1 and len(singles) == 1, "Each train needs one coupler cluster and one label"
        couplers = [(r, c, int(bbox[r, c])) for r, c in big[0]]
        lr, lc = singles[0][0]
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "couplers": couplers,
                "label_color": int(bbox[lr, lc]),
            }
        )

    out = grid.copy()
    for train in trains:
        for r, c in train["abs_coords"]:
            out[r, c] = bg

    for train in trains:
        assert train["label_color"] in stations_by_label, "Train label must match a station label"
        target = stations_by_label[train["label_color"]]
        op = choose_unique_op(train, target["core"])
        tbbox = apply_transform(train["bbox"], *op)
        transformed_couplers = transform_colored_points(train["couplers"], train["bbox"].shape, *op)
        plain_couplers = [(r, c) for r, c, _ in transformed_couplers]
        tr0, tc0, _, _ = _bbox(plain_couplers)
        sr0, sc0, _, _ = _bbox(target["core"])
        off_r, off_c = sr0 - tr0, sc0 - tc0
        landed = {(r + off_r, c + off_c, color) for r, c, color in transformed_couplers}
        assert landed == set(target["core"]), "Transformed couplers must land exactly on the station core"

        coupler_positions = {(r, c) for r, c, _ in transformed_couplers}
        h1, w1 = tbbox.shape
        ok = True
        for r in range(h1):
            for c in range(w1):
                v = int(tbbox[r, c])
                if v == bg:
                    continue
                rr, cc = r + off_r, c + off_c
                if not (region["r0"] <= rr <= region["r1"] and region["c0"] <= cc <= region["c1"]):
                    ok = False
                    break
                if stripe_mask[rr, cc]:
                    ok = False
                    break
                if (r, c) in coupler_positions:
                    if (rr, cc, v) not in landed:
                        ok = False
                        break
                elif out[rr, cc] != bg:
                    ok = False
                    break
            if not ok:
                break
        assert ok, "Docked train must fit entirely within the region"

        for r in range(h1):
            for c in range(w1):
                if int(tbbox[r, c]) != bg and (r, c) not in coupler_positions:
                    out[r + off_r, c + off_c] = int(tbbox[r, c])
    return out


# --------------- generator ---------------

GRID_H_RANGE = (13, 31)
GRID_W_RANGE = (16, 29)
GRID_MIN_AREA = 300
MIN_REGION_DIM = 16

STRIPE_THICK_RANGE = (2, 4)
STRIPE_EDGE_COUNT = 3

TRAIN_PATTERNS = [(4, 4), (5, 5)]

BODY_MIN = 5
BODY_MAX = 20
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)
BODY_GROWTH_ITERS = 160
PATH_BOUNDS_PAD = 4

LAYOUT_RETRIES = 200
DOCK_RETRIES = 30
GENERATE_RETRIES = 400

CORE_BASE_SHAPES = {
    4: {(0, 0), (1, 0), (2, 0), (2, 1)},
    5: {(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)},
    6: {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1)},
}


def _inverse_patch(patch, op):
    for inv in OPS:
        cand = apply_transform(patch, *inv)
        if np.array_equal(apply_transform(cand, *op), patch):
            return cand
    raise ValueError("no inverse transform")


def _valid_ops_for_patch(patch):
    hlo, hhi = PATCH_H_RANGE
    wlo, whi = PATCH_W_RANGE
    valid = []
    for op in OPS:
        h, w = _inverse_patch(patch, op).shape
        if hlo <= h <= hhi and wlo <= w <= whi:
            valid.append(op)
    return valid


def _layout():
    for _ in range(LAYOUT_RETRIES):
        edges = random.sample(["top", "bottom", "left", "right"], STRIPE_EDGE_COUNT)
        thick = {edge: random.randint(*STRIPE_THICK_RANGE) for edge in edges}
        top, bottom = thick.get("top", 0), thick.get("bottom", 0)
        left, right = thick.get("left", 0), thick.get("right", 0)

        candidates = []
        for h in range(*GRID_H_RANGE):
            for w in range(*GRID_W_RANGE):
                if h * w < GRID_MIN_AREA:
                    continue
                ih, iw = h - top - bottom, w - left - right
                if ih < MIN_REGION_DIM or iw < MIN_REGION_DIM:
                    continue
                candidates.append((h, w))
        if not candidates:
            continue

        h, w = random.choice(candidates)
        pool = NON_BODY_COLORS[:]
        bg = random.choice(pool)
        pool.remove(bg)
        stripe_cols = random.sample(pool, STRIPE_EDGE_COUNT)
        edge_col = dict(zip(edges, stripe_cols))
        grid = np.full((h, w), bg, dtype=int)

        if top:
            grid[:top, :] = edge_col["top"]
        if bottom:
            grid[h - bottom :, :] = edge_col["bottom"]
        if left:
            grid[top : h - bottom, :left] = edge_col["left"]
        if right:
            grid[top : h - bottom, w - right :] = edge_col["right"]

        region = {"r0": top, "r1": h - bottom - 1, "c0": left, "c1": w - right - 1}
        free_palette = [c for c in NON_BODY_COLORS if c != bg and c not in stripe_cols]
        if len(free_palette) >= 5:
            return grid, bg, [region], free_palette
    return None


def _make_proto(size, core_colors):
    coords = sorted(CORE_BASE_SHAPES[size])
    pattern = []
    for idx, (r, c) in enumerate(coords):
        color = core_colors[idx % len(core_colors)]
        pattern.append((int(r), int(c), int(color)))
    return pattern


def _build_docked_patch(bg, core_rel, label_color):
    coupler_cells = [(r, c) for r, c, _ in core_rel]
    blocked = set(coupler_cells)
    contacts = [
        q
        for r, c in coupler_cells
        for dr, dc in DIRS4
        if (q := (r + dr, c + dc)) not in blocked
    ]
    if not contacts:
        return None

    body = {random.choice(contacts)}
    all_seed = set(coupler_cells) | body
    sr0, sc0, sr1, sc1 = _bbox(all_seed)
    bounds = (sr0 - PATH_BOUNDS_PAD, sr1 + PATH_BOUNDS_PAD, sc0 - PATH_BOUNDS_PAD, sc1 + PATH_BOUNDS_PAD)

    if not _connect_couplers_to_body(coupler_cells, body, blocked, bounds):
        return None
    if len(body) > BODY_MAX:
        return None

    target = random.randint(max(BODY_MIN, len(body)), BODY_MAX)
    _grow_body(body, blocked, target, BODY_MIN, BODY_MAX, PATCH_H_RANGE, PATCH_W_RANGE, BODY_GROWTH_ITERS)

    core_and_body = body | set(coupler_cells)
    label_candidates = []
    for r, c in body:
        for dr, dc in DIRS4:
            q = (r + dr, c + dc)
            if q in core_and_body:
                continue
            if any(abs(q[0] - cr) <= 1 and abs(q[1] - cc) <= 1 for cr, cc in coupler_cells):
                continue
            h, w = _dims(core_and_body | {q})
            if PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1] and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]:
                label_candidates.append(q)
    if not label_candidates:
        return None
    label_cell = random.choice(label_candidates)

    all_cells = core_and_body | {label_cell}
    h, w = _dims(all_cells)
    if not (
        BODY_MIN <= len(body) <= BODY_MAX
        and PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1]
        and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]
    ):
        return None

    r0, c0, _, _ = _bbox(all_cells)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = 8
    patch[label_cell[0] - r0, label_cell[1] - c0] = label_color
    rel_core = []
    for r, c, color in core_rel:
        patch[r - r0, c - c0] = color
        rel_core.append((r - r0, c - c0, color))
    return patch, rel_core


def _patch_locations(region, patch, bg, occupied, coupler_rel=None, station_clearance=False, margin=0):
    h, w = occupied.shape
    ph, pw = patch.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(patch != bg)]
    locs = []
    r_start = max(region["r0"] + margin, 0)
    r_stop = min(region["r1"] - margin - ph + 2, h - ph + 1)
    c_start = max(region["c0"] + margin, 0)
    c_stop = min(region["c1"] - margin - pw + 2, w - pw + 1)
    for rr in range(r_start, r_stop):
        for cc in range(c_start, c_stop):
            ok = True
            for pr, pc in coords:
                if occupied[rr + pr, cc + pc]:
                    ok = False
                    break
            if not ok:
                continue
            if station_clearance and coupler_rel:
                for pr, pc, _ in coupler_rel:
                    ar, ac = rr + pr, cc + pc
                    if not (
                        region["r0"] + 2 <= ar <= region["r1"] - 2
                        and region["c0"] + 2 <= ac <= region["c1"] - 2
                    ):
                        ok = False
                        break
            if ok:
                locs.append((rr, cc))
    return locs


def _station_label_candidates(region, core_abs, patch_abs_mask, occupied):
    patch_cells = set(map(tuple, np.argwhere(patch_abs_mask)))
    core_plain = [(r, c) for r, c, _ in core_abs]
    core_set = set(core_plain)
    cands = set()
    for r, c in core_plain:
        for dr, dc in DIRS4:
            mid = (r + dr, c + dc)
            q = (r + 2 * dr, c + 2 * dc)
            rr, cc = q
            if not (region["r0"] <= rr <= region["r1"] and region["c0"] <= cc <= region["c1"]):
                continue
            if mid in core_set or q in core_set:
                continue
            if patch_abs_mask[rr, cc] or occupied[rr, cc]:
                continue
            if any(abs(rr - cr) <= 1 and abs(cc - cc0) <= 1 for cr, cc0 in core_plain):
                continue
            if any(abs(rr - pr) <= 1 and abs(cc - pc) <= 1 for pr, pc in patch_cells):
                continue
            cands.add(q)
    return list(cands)


def _place_docked_trains(base, bg, regions, free_palette):
    region = regions[0]
    output = base.copy()
    occupied = output != bg

    sizes = list(random.choice(TRAIN_PATTERNS))
    label_colors = random.sample(free_palette, len(sizes))
    core_colors = random.sample([c for c in free_palette if c not in label_colors], 2)
    protos = {size: _make_proto(size, core_colors) for size in set(sizes)}

    trains = []
    for size, label_color in zip(sizes, label_colors):
        placed = False
        proto = protos[size]
        for _ in range(DOCK_RETRIES):
            oriented_core = _transform_colored_shape(proto, random.choice(OPS))
            result = _build_docked_patch(bg, oriented_core, label_color)
            if result is None:
                continue
            patch, coupler_rel = result
            locs = _patch_locations(region, patch, bg, _dilate(occupied), coupler_rel, True, 0)
            random.shuffle(locs)
            for r0, c0 in locs:
                ph, pw = patch.shape
                patch_abs_mask = np.zeros_like(output, dtype=bool)
                patch_abs_mask[r0 : r0 + ph, c0 : c0 + pw] = patch != bg
                station_core = [(r0 + r, c0 + c, color) for r, c, color in coupler_rel]
                label_cands = _station_label_candidates(region, station_core, patch_abs_mask, occupied)
                if not label_cands:
                    continue
                station_label = random.choice(label_cands)
                paste(output, patch, r0, c0, transparent=bg)
                output[station_label] = label_color
                occupied = output != bg
                trains.append(
                    {
                        "patch": patch,
                        "station_core": station_core,
                        "station_label": (station_label[0], station_label[1], label_color),
                        "label_color": label_color,
                        "size": size,
                        "proto_sig": _normalize_colored(proto),
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return output, trains


def _scatter_trains(base, bg, regions, trains):
    region = regions[0]
    h, w = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((h, w), dtype=bool)
    station_label_mask = np.zeros((h, w), dtype=bool)

    for train in trains:
        for r, c, color in train["station_core"]:
            input_grid[r, c] = color
            station_mask[r, c] = True
        lr, lc, color = train["station_label"]
        input_grid[lr, lc] = color
        station_label_mask[lr, lc] = True

    train_infos = []
    for idx, train in enumerate(trains):
        valid = _valid_ops_for_patch(train["patch"])
        if not valid:
            return None
        train_infos.append((idx, train, valid))
    train_infos.sort(key=lambda item: np.count_nonzero(item[1]["patch"] != bg), reverse=True)

    base_forbidden = _dilate(base != bg) | _dilate(station_mask | station_label_mask)
    placements = [None] * len(trains)

    def dfs(pos, forbidden, used_non_identity):
        if pos == len(train_infos):
            return used_non_identity
        idx, train, valid = train_infos[pos]
        options = valid[:]
        random.shuffle(options)
        for op in options:
            in_patch = _inverse_patch(train["patch"], op)
            locs = _patch_locations(region, in_patch, bg, forbidden, None, False, 0)
            random.shuffle(locs)
            for r0, c0 in locs:
                ph, pw = in_patch.shape
                patch_mask = np.zeros((h, w), dtype=bool)
                patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
                placements[idx] = (in_patch, r0, c0)
                if dfs(pos + 1, forbidden | _dilate(patch_mask), used_non_identity or op != (False, 0)):
                    return True
        placements[idx] = None
        return False

    if not dfs(0, base_forbidden, False):
        return None

    for item in placements:
        assert item is not None
        in_patch, r0, c0 = item
        paste(input_grid, in_patch, r0, c0, transparent=bg)
    return input_grid


def _combo_metrics(trains):
    sizes = [train["size"] for train in trains]
    sigs = [train["proto_sig"] for train in trains]
    multi_color_station_count = sum(len({color for _, _, color in train["station_core"]}) >= 2 for train in trains)
    return {
        "train_count": len(trains),
        "repeated_sizes": len(sizes) - len(set(sizes)),
        "repeated_proto_signatures": len(sigs) - len(set(sigs)),
        "multicolor_stations": multi_color_station_count,
    }


def _gen_attempt():
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions, free_palette = layout

    placed = _place_docked_trains(base, bg, regions, free_palette)
    if placed is None:
        return None
    output, trains = placed

    metrics = _combo_metrics(trains)
    if metrics["repeated_sizes"] < 1:
        return None
    if metrics["repeated_proto_signatures"] < 1:
        return None
    if metrics["multicolor_stations"] != metrics["train_count"]:
        return None

    input_grid = _scatter_trains(base, bg, regions, trains)
    if input_grid is None:
        return None

    counts = Counter(map(int, input_grid.flatten()))
    if counts[bg] != max(counts.values()):
        return None
    try:
        if np.array_equal(main(input_grid), output):
            return input_grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6e4f6532 combo input")


for _size, _shape in CORE_BASE_SHAPES.items():
    assert len({_normalize_plain(_transform_shape(_shape, op)) for op in OPS}) == 8, (
        f"station shape size {_size} must have 8 distinct D4 orientations"
    )
