# Variant of 6e4f6532
# Dimension: D16 — 符号编码：条纹的双重角色 (symbol_instructions)
# Change: 条纹同时起两个作用：（1）分隔区域的视觉边界，（2）通过颜色为标记像素提供方向参照... → 条纹图案编码（非纯色条纹）
#
# Design analysis for this variant
# a) Concretization and behavior trace:
#    The original puzzle used solid divider colors. This variant replaces every
#    divider with a two-color stripe pattern. Each boundary stripe now has a
#    unique (primary, secondary) color pair. The stripe still partitions the
#    canvas, but its pair also defines marker semantics:
#    - marker in the stripe's primary color: face that boundary with the direct
#      orientation
#    - marker in the stripe's secondary color: face that boundary and also pick
#      the flipped branch among the otherwise-valid rotations
#    The output-first generator builds docked trains with markers positioned on
#    specific sides of the coupler group. After a scatter transform is chosen,
#    every marker is recolored to the primary or secondary member of its target
#    stripe pair according to the transform's flip bit. The solver therefore has
#    to recover both side direction and flip parity from the stripe pattern.
# b) Complexity calibration:
#    To keep the stripe pair readable and avoid color collisions, each instance
#    uses exactly three stripes total: either three outer-edge stripes, or two
#    edge stripes plus one internal divider. That keeps the palette within ARC's
#    0..7 non-train colors while still leaving 2-3 trains and distinct station
#    sizes. The train body remains asymmetric and irregular so the flip branch is
#    visually meaningful.
# c) Implicit assumptions review:
#    The original solver assumed "a stripe is a solid color block" in both
#    detection and marker decoding. Those assumptions are no longer valid. This
#    file therefore:
#    - detects stripe runs by uniform rows/columns, not by repeated same-color
#      slabs
#    - derives boundary rules from the full stripe run next to a region side,
#      extracting both primary and secondary colors
#    - scores train orientations against (direction, flip-bit) rules instead of
#      plain color-to-direction rules
# d) Placement legality:
#    The generator still works output-first, validates every candidate with
#    main(), enforces station clearance from stripes, keeps trains inside regions,
#    and scatters trains with one-cell spacing from stripes/stations/other trains.
# e) Visual observability:
#    The pattern change is directly visible because every stripe alternates two
#    colors. The primary/secondary choice appears again on train markers, so the
#    new rule can be inferred from a few examples without relying on hidden ids.
# f) Visual cleanliness:
#    Only three stripes are used, regions remain large rectangles, and marker
#    counts stay small. That keeps the grids readable even with two-color borders.
# g) Color-change inferability:
#    Each stripe owns exactly one pair of colors, and the marker colors come only
#    from those visible pairs. Humans can match marker colors back to stripe
#    patterns structurally instead of memorizing arbitrary remappings.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def xform_coords(coords, shape, flip_ud: bool, rot_k: int):
    result = transform_coords(coords, shape, flip_ud, rot_k)
    new_shape = apply_transform(np.zeros(shape, dtype=int), flip_ud, rot_k).shape
    return list(result), new_shape


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock train sprites onto station groups by matching coupler count, then read
    striped boundary-pair markers to recover both direction and flip state.

    Input structure:
    - Background plus exactly three divider stripes partition the canvas into one
      or two rectangular regions.
    - Every stripe is a two-color pattern rather than a solid slab; one color is
      the region-facing "primary" band and the other is the auxiliary band.
    - Color 8 forms train bodies; color 9 marks both isolated stations and
      train-attached couplers; stripe-palette pixels on trains are markers.

    Rule:
    - Stations are 8-connected 9-groups with no adjacent 8s. Train components
      are the non-stripe connected components containing color 8.
    - Match each train to the unique station group whose size equals the train's
      number of attached couplers.
    - For the target region, interpret each marker color through the bordering
      stripe pair: primary means "face this side directly", secondary means
      "face this side with the flipped branch".
    - Apply the unique flip/rotation, translate so couplers land on the station
      cells, erase the original train, and redraw only its non-bg non-9 cells.

    Key insight:
    Coupler count still selects the docking target, but orientation now comes
    from stripe-pattern decoding: the marker color identifies a boundary side
    and simultaneously says whether the correct docking transform uses the
    unflipped or flipped branch.
    """
    grid = grid.copy()
    H, W = grid.shape
    bg = get_bg(grid)

    def uniform_nonbg_row(r: int, c0: int, c1: int) -> bool:
        row = grid[r, c0:c1]
        return row.size > 0 and (row == row[0]).all() and row[0] != bg

    def uniform_nonbg_col(c: int, r0: int, r1: int) -> bool:
        col = grid[r0:r1, c]
        return col.size > 0 and (col == col[0]).all() and col[0] != bg

    def find_edge_stripes():
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()

        top_h = 0
        for r in range(H):
            if uniform_nonbg_row(r, 0, W):
                top_h += 1
                colors.add(int(grid[r, 0]))
            else:
                break
        if top_h:
            stripe[:top_h, :] = True

        bot_h = 0
        for r in range(H - 1, -1, -1):
            if uniform_nonbg_row(r, 0, W):
                bot_h += 1
                colors.add(int(grid[r, 0]))
            else:
                break
        if bot_h:
            stripe[H - bot_h :, :] = True

        inner_r0, inner_r1 = top_h, H - bot_h
        left_w = 0
        for c in range(W):
            if uniform_nonbg_col(c, inner_r0, inner_r1):
                left_w += 1
                colors.add(int(grid[inner_r0, c]))
            else:
                break
        if left_w:
            stripe[inner_r0:inner_r1, :left_w] = True

        right_w = 0
        for c in range(W - 1, -1, -1):
            if uniform_nonbg_col(c, inner_r0, inner_r1):
                right_w += 1
                colors.add(int(grid[inner_r0, c]))
            else:
                break
        if right_w:
            stripe[inner_r0:inner_r1, W - right_w :] = True

        return stripe, colors, (top_h, bot_h, left_w, right_w)

    def find_internal_stripes(edge_info):
        top_h, bot_h, left_w, right_w = edge_info
        r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()

        c = c0
        while c < c1:
            if uniform_nonbg_col(c, r0, r1):
                c2 = c
                while c2 + 1 < c1 and uniform_nonbg_col(c2 + 1, r0, r1):
                    c2 += 1
                stripe[r0:r1, c : c2 + 1] = True
                for cc in range(c, c2 + 1):
                    colors.add(int(grid[r0, cc]))
                c = c2 + 1
            else:
                c += 1

        r = r0
        while r < r1:
            if uniform_nonbg_row(r, c0, c1):
                r2 = r
                while r2 + 1 < r1 and uniform_nonbg_row(r2 + 1, c0, c1):
                    r2 += 1
                stripe[r : r2 + 1, c0:c1] = True
                for rr in range(r, r2 + 1):
                    colors.add(int(grid[rr, c0]))
                r = r2 + 1
            else:
                r += 1

        return stripe, colors

    def station_groups():
        sm = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 9:
                    continue
                if all(
                    grid[r + dr, c + dc] != 8
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (dr or dc) and 0 <= r + dr < H and 0 <= c + dc < W
                ):
                    sm[r, c] = True
        lab, n = label(sm, structure=np.ones((3, 3), dtype=int))
        groups = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            groups.append([(int(r), int(c)) for r, c in coords])
        return groups

    def regions_from_stripes(stripe_mask):
        lab, n = label(~stripe_mask)
        regs = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if (r1 - r0 + 1) >= 5 and (c1 - c0 + 1) >= 5:
                regs.append({"r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1)})
        return regs

    def read_boundary_pair(reg, direction):
        r0, r1, c0, c1 = reg["r0"], reg["r1"], reg["c0"], reg["c1"]
        if direction == 0 and r0 - 1 >= 0 and uniform_nonbg_row(r0 - 1, c0, c1 + 1):
            rows = []
            rr = r0 - 1
            while rr >= 0 and uniform_nonbg_row(rr, c0, c1 + 1):
                rows.append((rr, int(grid[rr, c0])))
                rr -= 1
            primary = rows[0][1]
            others = {color for _, color in rows if color != primary}
            if len(others) == 1:
                return primary, next(iter(others))
        if direction == 2 and r1 + 1 < H and uniform_nonbg_row(r1 + 1, c0, c1 + 1):
            rows = []
            rr = r1 + 1
            while rr < H and uniform_nonbg_row(rr, c0, c1 + 1):
                rows.append((rr, int(grid[rr, c0])))
                rr += 1
            primary = rows[0][1]
            others = {color for _, color in rows if color != primary}
            if len(others) == 1:
                return primary, next(iter(others))
        if direction == 3 and c0 - 1 >= 0 and uniform_nonbg_col(c0 - 1, r0, r1 + 1):
            cols = []
            cc = c0 - 1
            while cc >= 0 and uniform_nonbg_col(cc, r0, r1 + 1):
                cols.append((cc, int(grid[r0, cc])))
                cc -= 1
            primary = cols[0][1]
            others = {color for _, color in cols if color != primary}
            if len(others) == 1:
                return primary, next(iter(others))
        if direction == 1 and c1 + 1 < W and uniform_nonbg_col(c1 + 1, r0, r1 + 1):
            cols = []
            cc = c1 + 1
            while cc < W and uniform_nonbg_col(cc, r0, r1 + 1):
                cols.append((cc, int(grid[r0, cc])))
                cc += 1
            primary = cols[0][1]
            others = {color for _, color in cols if color != primary}
            if len(others) == 1:
                return primary, next(iter(others))
        return None

    def boundary_marker_rules_for_region(reg):
        rules = {}
        for direction in range(4):
            pair = read_boundary_pair(reg, direction)
            if pair is None:
                continue
            primary, secondary = pair
            rules[primary] = (direction, False)
            rules[secondary] = (direction, True)
        return rules

    def choose_best_op(train, marker_rules):
        bbox0 = train["bbox"]
        H0, W0 = bbox0.shape
        usable = [(v, r, c) for v, r, c in train["marker_pixels"] if v in marker_rules]
        assert usable, "Each train must expose stripe-palette markers"
        expected_flip = {marker_rules[v][1] for v, _, _ in usable}
        assert len(expected_flip) == 1, "Markers on a train must agree on the flip bit"
        expected_flip = next(iter(expected_flip))
        ref_r = sum(r for r, _ in train["body9"]) / len(train["body9"])
        ref_c = sum(c for _, c in train["body9"]) / len(train["body9"])
        best = (False, 0)
        best_score = (-1, -1)
        for flip_ud in (False, True):
            for rot_k in range(4):
                ref_t = list(
                    transform_coords(
                        [(int(round(ref_r)), int(round(ref_c)))],
                        (H0, W0),
                        flip_ud,
                        rot_k,
                    )
                )
                rr_ref, cc_ref = ref_t[0]
                dir_matches = 0
                for color, mr0, mc0 in usable:
                    m_t = list(
                        transform_coords(
                            [(int(mr0), int(mc0))],
                            (H0, W0),
                            flip_ud,
                            rot_k,
                        )
                    )
                    rr_m, cc_m = m_t[0]
                    dr, dc = rr_m - rr_ref, cc_m - cc_ref
                    direction = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
                    if direction == marker_rules[color][0]:
                        dir_matches += 1
                score = (dir_matches, int(flip_ud == expected_flip))
                if score > best_score:
                    best_score = score
                    best = (flip_ud, rot_k)
        return best

    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups()
    groups_by_size = defaultdict(list)
    for group in groups:
        groups_by_size[len(group)].append(group)
    station_set = {pt for group in groups for pt in group}

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
        abs_coords = [(int(r), int(c)) for r, c in coords]
        body9 = []
        marker_pixels = []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 9 and (r, c) not in station_set:
                body9.append((int(r - r0), int(c - c0)))
            if v in stripe_colors:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body9": body9,
                "marker_pixels": marker_pixels,
            }
        )

    out = grid.copy()
    for train in trains:
        for r, c in train["abs_coords"]:
            out[r, c] = bg

    for train in trains:
        n9 = len(train["body9"])
        assert n9 > 0, "Each train must carry at least one coupler"
        candidate_groups = groups_by_size.get(n9)
        assert candidate_groups is not None and len(candidate_groups) == 1
        group = candidate_groups[0]
        gset = set(group)
        matching_regions = [
            reg
            for reg in regions
            if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in group)
        ]
        assert len(matching_regions) == 1
        reg = matching_regions[0]
        marker_rules = boundary_marker_rules_for_region(reg)
        op = choose_best_op(train, marker_rules)
        tbbox = apply_transform(train["bbox"], *op)
        t_body9, (h1, w1) = xform_coords(train["body9"], train["bbox"].shape, *op)
        body9_set = set(t_body9)
        offsets = {
            (sr - br, sc - bc)
            for br, bc in body9_set
            for sr, sc in gset
            if all((r + sr - br, c + sc - bc) in gset for r, c in body9_set)
        }
        assert len(offsets) == 1, "Couplers must align to a unique station offset"
        off_r, off_c = next(iter(offsets))

        ok = True
        for r in range(h1):
            for c in range(w1):
                v = int(tbbox[r, c])
                if v == bg or v == 9:
                    continue
                rr, cc = r + off_r, c + off_c
                if not (reg["r0"] <= rr <= reg["r1"] and reg["c0"] <= cc <= reg["c1"]):
                    ok = False
                    break
                if stripe_mask[rr, cc] or (out[rr, cc] != bg and out[rr, cc] != 9):
                    ok = False
                    break
            if not ok:
                break
        assert ok, "Docked train must fit within its target region"

        for r in range(h1):
            for c in range(w1):
                v = int(tbbox[r, c])
                if v != bg and v != 9:
                    out[r + off_r, c + off_c] = v
    return out


# ── Generator constants ──────────────────────────────────────────────

GRID_H_RANGE = (13, 31)
GRID_W_RANGE = (16, 29)
GRID_MIN_AREA = 300
MIN_REGION_DIM = 10
INTERNAL_STRIPE_THICK = 3
EDGE_STRIPE_THICK_CHOICES = [2, 3]
INTERNAL_PROB = 0.55

TRAIN_COUNTS = [2, 3]
STATION_SIZES = [1, 2, 4]

BODY_MIN = 5
BODY_MAX = 20
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)

MARKER_DIST_RANGE = (2, 3)
MARKER_JITTER = [-1, 0, 1]
THREE_MARKER_PROB = 0.65

BODY_GROWTH_ITERS = 160
PATH_BOUNDS_PAD = 4

LAYOUT_RETRIES = 200
DOCK_RETRIES = 20
GENERATE_RETRIES = 300

COLOR_POOL = list(range(8))
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


def _apply_transform(arr, flip_ud=False, rot_k=0):
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def _inverse_patch(patch, op):
    for inv in OPS:
        cand = _apply_transform(patch, *inv)
        if np.array_equal(_apply_transform(cand, *op), patch):
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


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _dims(coords):
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


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


def _region_dirs(regions, edge_pairs, top, bottom, left, right, H, W, internal):
    for reg in regions:
        dirs = {}
        if top and reg["r0"] == top:
            dirs[0] = {"primary": edge_pairs["top"][0], "secondary": edge_pairs["top"][1]}
        if bottom and reg["r1"] == H - bottom - 1:
            dirs[2] = {"primary": edge_pairs["bottom"][0], "secondary": edge_pairs["bottom"][1]}
        if left and reg["c0"] == left:
            dirs[3] = {"primary": edge_pairs["left"][0], "secondary": edge_pairs["left"][1]}
        if right and reg["c1"] == W - right - 1:
            dirs[1] = {"primary": edge_pairs["right"][0], "secondary": edge_pairs["right"][1]}
        if internal:
            kind, pos, pair = internal
            if kind == "v":
                if reg["c1"] == pos - 1:
                    dirs[1] = {"primary": pair[0], "secondary": pair[1]}
                else:
                    dirs[3] = {"primary": pair[0], "secondary": pair[1]}
            else:
                if reg["r1"] == pos - 1:
                    dirs[2] = {"primary": pair[0], "secondary": pair[1]}
                else:
                    dirs[0] = {"primary": pair[0], "secondary": pair[1]}
        reg["dirs"] = dirs
        rh = reg["r1"] - reg["r0"] + 1
        rw = reg["c1"] - reg["c0"] + 1
        if rh < MIN_REGION_DIM or rw < MIN_REGION_DIM or len(dirs) != 3:
            return False
    return True


def _stripe_band_colors(primary, secondary, thickness, inner_at_end):
    colors = []
    for i in range(thickness):
        dist_from_inner = thickness - 1 - i if inner_at_end else i
        colors.append(primary if dist_from_inner % 2 == 0 else secondary)
    return colors


def _layout():
    for _ in range(LAYOUT_RETRIES):
        use_internal = random.random() < INTERNAL_PROB
        if use_internal:
            orient = random.choice(["v", "h"])
            edges = ["top", "bottom"] if orient == "v" else ["left", "right"]
        else:
            orient = None
            edges = random.sample(["top", "bottom", "left", "right"], 3)

        thick = {edge: random.choice(EDGE_STRIPE_THICK_CHOICES) for edge in edges}
        top, bottom = thick.get("top", 0), thick.get("bottom", 0)
        left, right = thick.get("left", 0), thick.get("right", 0)

        min_inner_for_split = MIN_REGION_DIM * 2 + INTERNAL_STRIPE_THICK
        candidates = []
        for H in range(*GRID_H_RANGE):
            for W in range(*GRID_W_RANGE):
                if H * W < GRID_MIN_AREA:
                    continue
                ih, iw = H - top - bottom, W - left - right
                if ih < MIN_REGION_DIM or iw < MIN_REGION_DIM:
                    continue
                if orient == "v" and iw < min_inner_for_split:
                    continue
                if orient == "h" and ih < min_inner_for_split:
                    continue
                candidates.append((H, W))
        if not candidates:
            continue

        H, W = random.choice(candidates)
        bg = random.choice(COLOR_POOL)
        pool = [c for c in COLOR_POOL if c != bg]
        stripe_names = edges + (["internal"] if use_internal else [])
        if len(pool) < 2 * len(stripe_names):
            continue
        sampled = random.sample(pool, 2 * len(stripe_names))
        stripe_pairs = {
            name: (sampled[2 * i], sampled[2 * i + 1])
            for i, name in enumerate(stripe_names)
        }

        grid = np.full((H, W), bg, dtype=int)

        if top:
            primary, secondary = stripe_pairs["top"]
            for r, color in enumerate(_stripe_band_colors(primary, secondary, top, True)):
                grid[r, :] = color
        if bottom:
            primary, secondary = stripe_pairs["bottom"]
            colors = _stripe_band_colors(primary, secondary, bottom, False)
            for i, color in enumerate(colors):
                grid[H - bottom + i, :] = color
        if left:
            primary, secondary = stripe_pairs["left"]
            for c, color in enumerate(_stripe_band_colors(primary, secondary, left, True)):
                grid[top : H - bottom, c] = color
        if right:
            primary, secondary = stripe_pairs["right"]
            colors = _stripe_band_colors(primary, secondary, right, False)
            for i, color in enumerate(colors):
                grid[top : H - bottom, W - right + i] = color

        regions = []
        internal = None
        if orient == "v":
            lo = left + MIN_REGION_DIM
            hi = W - right - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            c = random.randint(lo, hi)
            primary, secondary = stripe_pairs["internal"]
            band = [primary, secondary, primary]
            for i, color in enumerate(band):
                grid[top : H - bottom, c + i] = color
            internal = ("v", c, stripe_pairs["internal"])
            regions = [
                {"r0": top, "r1": H - bottom - 1, "c0": left, "c1": c - 1},
                {
                    "r0": top,
                    "r1": H - bottom - 1,
                    "c0": c + INTERNAL_STRIPE_THICK,
                    "c1": W - right - 1,
                },
            ]
        elif orient == "h":
            lo = top + MIN_REGION_DIM
            hi = H - bottom - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            r = random.randint(lo, hi)
            primary, secondary = stripe_pairs["internal"]
            band = [primary, secondary, primary]
            for i, color in enumerate(band):
                grid[r + i, left : W - right] = color
            internal = ("h", r, stripe_pairs["internal"])
            regions = [
                {"r0": top, "r1": r - 1, "c0": left, "c1": W - right - 1},
                {
                    "r0": r + INTERNAL_STRIPE_THICK,
                    "r1": H - bottom - 1,
                    "c0": left,
                    "c1": W - right - 1,
                },
            ]
        else:
            regions = [{"r0": top, "r1": H - bottom - 1, "c0": left, "c1": W - right - 1}]

        if _region_dirs(regions, stripe_pairs, top, bottom, left, right, H, W, internal):
            return grid, bg, regions
    return None


def _station_shape(size):
    if size == 1:
        return {(0, 0)}
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    return {(0, 0), (0, 1), (1, 0), (1, 1)}


def _marker_specs(dirs):
    directions = list(dirs.keys())
    if len(directions) >= 3 and random.random() < THREE_MARKER_PROB:
        return random.sample(directions, 3)
    pairs = []
    for i, a in enumerate(directions):
        for b in directions[i + 1 :]:
            if (a - b) % 4 in (1, 3):
                pairs.append([a, b])
    return random.choice(pairs) if pairs else random.sample(directions, 2)


def _marker_cell(couplers, direction):
    r0, c0, r1, c1 = _bbox(couplers)
    d = random.randint(*MARKER_DIST_RANGE)
    jitter = random.choice(MARKER_JITTER)
    if direction == 0:
        return (r0 - d, random.randint(c0, c1) + jitter)
    if direction == 2:
        return (r1 + d, random.randint(c0, c1) + jitter)
    if direction == 3:
        return (random.randint(r0, r1) + jitter, c0 - d)
    return (random.randint(r0, r1) + jitter, c1 + d)


def _inward_cell(marker, direction):
    dr, dc = DIRS4[(direction + 2) % 4]
    return marker[0] + dr, marker[1] + dc


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


def _grow_body(body, blocked, target):
    for _ in range(BODY_GROWTH_ITERS):
        all_cells = body | blocked
        h, w = _dims(all_cells)
        if len(body) >= target and h >= PATCH_H_RANGE[0] and w >= PATCH_W_RANGE[0]:
            break
        cands = []
        for r, c in body:
            for dr, dc in DIRS4:
                q = (r + dr, c + dc)
                if q in body or q in blocked:
                    continue
                nh, nw = _dims(all_cells | {q})
                if nh <= PATCH_H_RANGE[1] and nw <= PATCH_W_RANGE[1]:
                    cands.append(q)
        if not cands or len(body) >= BODY_MAX:
            break
        h, w = _dims(all_cells)
        if h < PATCH_H_RANGE[0]:
            r0, _, r1, _ = _bbox(all_cells)
            expand = [q for q in cands if q[0] < r0 or q[0] > r1]
            cands = expand or cands
        elif w < PATCH_W_RANGE[0]:
            _, c0, _, c1 = _bbox(all_cells)
            expand = [q for q in cands if q[1] < c0 or q[1] > c1]
            cands = expand or cands
        body.add(random.choice(cands))


def _build_docked_patch(bg, dirs, size):
    couplers = _station_shape(size)
    marker_dirs = _marker_specs(dirs)
    markers = []
    marker_cells = set()
    for direction in marker_dirs:
        p = _marker_cell(couplers, direction)
        if p in couplers or p in marker_cells:
            return None
        markers.append((p, direction))
        marker_cells.add(p)

    blocked = set(couplers) | marker_cells
    contacts = [
        q
        for r, c in couplers
        for dr, dc in DIRS4
        if (q := (r + dr, c + dc)) not in blocked
    ]
    if not contacts:
        return None

    body = {random.choice(contacts)}
    all_seed = set(couplers) | marker_cells | body
    sr0, sc0, sr1, sc1 = _bbox(all_seed)
    bounds = (
        sr0 - PATH_BOUNDS_PAD,
        sr1 + PATH_BOUNDS_PAD,
        sc0 - PATH_BOUNDS_PAD,
        sc1 + PATH_BOUNDS_PAD,
    )

    for p, direction in markers:
        goal = _inward_cell(p, direction)
        if goal in blocked or not _path_to(body, goal, blocked, bounds):
            return None

    if not _connect_couplers_to_body(couplers, body, blocked, bounds):
        return None

    if len(body) > BODY_MAX:
        return None
    target = random.randint(max(BODY_MIN, len(body)), BODY_MAX)
    _grow_body(body, blocked, target)

    all_cells = body | set(couplers) | marker_cells
    h, w = _dims(all_cells)
    if not (
        BODY_MIN <= len(body) <= BODY_MAX
        and PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1]
        and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]
    ):
        return None

    r0, c0, _, _ = _bbox(all_cells)
    patch = np.full((h, w), bg, dtype=int)
    marker_specs = []
    for r, c in body:
        patch[r - r0, c - c0] = 8
    for (r, c), direction in markers:
        patch[r - r0, c - c0] = dirs[direction]["primary"]
        marker_specs.append({"pos": (r - r0, c - c0), "direction": direction})
    rel_couplers = []
    for r, c in couplers:
        patch[r - r0, c - c0] = 9
        rel_couplers.append((r - r0, c - c0))
    return patch, rel_couplers, marker_specs


def _patch_locations(region, patch, bg, occupied, coupler_rel=None, station_clearance=False, margin=0):
    H, W = occupied.shape
    ph, pw = patch.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(patch != bg)]
    locs = []
    r_start = max(region["r0"] + margin, 0)
    r_stop = min(region["r1"] - margin - ph + 2, H - ph + 1)
    c_start = max(region["c0"] + margin, 0)
    c_stop = min(region["c1"] - margin - pw + 2, W - pw + 1)
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
                for pr, pc in coupler_rel:
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


def _recolor_patch(patch, marker_specs, dirs, use_secondary):
    out = patch.copy()
    for marker in marker_specs:
        r, c = marker["pos"]
        direction = marker["direction"]
        pair = dirs[direction]
        out[r, c] = pair["secondary"] if use_secondary else pair["primary"]
    return out


def _place_docked_trains(base, bg, regions):
    output = base.copy()
    occupied = _dilate(output != bg)

    n_trains = random.choice(TRAIN_COUNTS)
    sizes = random.sample(STATION_SIZES, n_trains)
    trains = []
    for size in sizes:
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            built = _build_docked_patch(bg, reg["dirs"], size)
            if built is None:
                continue
            patch, coupler_rel, marker_specs = built
            locs = _patch_locations(reg, patch, bg, occupied, coupler_rel, True, 0)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(output, patch, r0, c0, transparent=bg)
            occupied = _dilate(output != bg)
            station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
            trains.append(
                {
                    "patch": patch,
                    "station": station_abs,
                    "region": reg,
                    "dirs": reg["dirs"],
                    "marker_specs": marker_specs,
                    "out_pos": (r0, c0),
                }
            )
            break
        else:
            return None
    return output, trains


def _scatter_trains(base, output, bg, regions, trains):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for train in trains:
        for r, c in train["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True

    valid_ops = []
    for train in trains:
        valid = _valid_ops_for_patch(train["patch"])
        if not valid:
            return None
        valid_ops.append(valid)
    ops = [random.choice(valid) for valid in valid_ops]
    if all(op == (False, 0) for op in ops):
        candidates = [i for i, valid in enumerate(valid_ops) if any(op != (False, 0) for op in valid)]
        if not candidates:
            return None
        idx = random.choice(candidates)
        ops[idx] = random.choice([op for op in valid_ops[idx] if op != (False, 0)])

    output = output.copy()
    colored_patches = []
    for train, op in zip(trains, ops):
        colored = _recolor_patch(train["patch"], train["marker_specs"], train["dirs"], op[0])
        colored_patches.append(colored)
        r0, c0 = train["out_pos"]
        paste(output, colored, r0, c0, transparent=bg)

    forbidden = _dilate(base != bg) | _dilate(station_mask)
    for train, op, patch in zip(trains, ops, colored_patches):
        in_patch = _inverse_patch(patch, op)
        regs = regions[:]
        random.shuffle(regs)
        all_locs = []
        for reg in regs:
            locs = _patch_locations(reg, in_patch, bg, forbidden, None, False, 1)
            all_locs.extend((reg, loc) for loc in locs)
        if not all_locs:
            return None
        _, (r0, c0) = random.choice(all_locs)
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        ph, pw = in_patch.shape
        patch_mask = np.zeros((H, W), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
        forbidden |= _dilate(patch_mask)

    return input_grid, output


def _gen_attempt():
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions = layout

    placed = _place_docked_trains(base, bg, regions)
    if placed is None:
        return None
    output, trains = placed

    scattered = _scatter_trains(base, output, bg, regions, trains)
    if scattered is None:
        return None
    input_grid, output = scattered

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
    raise RuntimeError("failed to generate valid 6e4f6532 D16 variant input")
