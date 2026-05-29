# Variant of 6e4f6532
# Dimension: D2 — 匹配条件：方向确定机制（旋转规则） (match_cond)
# Change: 火车上的条纹色标记物相对于耦合器质心的方向，必须与目标区域对应颜色边界条纹的方位一致，以此唯一确定旋转/翻转操作... → 标记像素使用距离编码方向

from __future__ import annotations

from collections import Counter, defaultdict, deque
import numpy as np
from scipy.ndimage import label
from arc_utils import get_bg, paste
import random


# Design analysis for the D2 variant
# a) Concrete behavior trace:
#    The original task used 2-3 marker pixels whose colors matched boundary stripes of the
#    target region. The solver grouped markers by color, then asked whether each color-group
#    faced the corresponding stripe side after a dihedral transform. In this variant, that
#    mapping is removed: every marker on a train can share the same dedicated color, and the
#    side assignment is recovered only by sorting marker distances to the coupler centroid.
#    Example with a 3-sided region bordered on up/right/left: in the docked state the nearest
#    marker lies on the train's up side, the second-nearest on the right side, and the farthest
#    on the left side. After the train is undocked, rotated, and scattered, the solver must
#    test all 8 transforms, sort transformed markers by centroid distance, and keep only the
#    transform whose ordered directions become [up, right, left]. Color no longer disambiguates
#    which marker stands for which side.
# b) Complexity calibration:
#    The change stays focused on orientation resolution. Coupler-count matching, station shapes,
#    stripe layout, docking translation, and output-first generation all remain unchanged.
#    To keep the cue visible rather than degenerate, the generator uses one dedicated marker
#    color not used by stripes, emits one marker per visible boundary side of the target region,
#    and rejects samples where marker distances tie or where more than one transform satisfies
#    the ordered-side rule.
# c) Hidden-assumption audit:
#    - Old `marker_pixels` collection depended on `v in stripe_colors`; that is invalid now,
#      because markers intentionally stop sharing stripe colors. The new solver treats every
#      non-bg/non-8/non-9 train pixel as a marker.
#    - Old `choose_best_op(...)` grouped markers by color and compared each group to
#      `divider_dirs[color]`. That whole premise is gone. The replacement compares a sorted
#      distance signature against the target region's canonical side order.
#    - Old generator `_marker_specs(...)` sampled colored side-labels from `dirs.items()`.
#      The new generator instead derives the target side sequence from the region geometry and
#      picks same-color marker cells whose distances strictly increase in that sequence.
# d) Placement legality:
#    Marker search is bounded to a small neighborhood around the couplers and prunes any set
#    whose combined coupler+marker bbox would exceed the original patch size limits. Paths from
#    markers to the body still use the original BFS-based construction, so markers remain part
#    of the train component and the final patch fits inside the target region after docking.
# e) Visual observability:
#    The variant is inferable from examples because marker colors stop carrying information:
#    all markers on a train look interchangeable chromatically, while their radial ordering from
#    the coupler cluster is stable and repeated across samples. Regions still visibly expose the
#    final absolute side set (top/right/down/left), so the only usable cue is the distance rank.
# f) Visual cleanliness:
#    No extra overlays are introduced. Markers remain sparse, single-pixel annotations attached
#    to the train body. Regions, stripes, stations, and train silhouettes stay as in the base
#    task, so the only visual novelty is the "same-color but different distance" marker pattern.
# g) Color-change inferability:
#    There is no arbitrary color mapping to learn. Marker color is intentionally uniform and
#    decoupled from stripe colors, which prevents fallback to the original color-coded rule.
# New solver assumptions after the change:
#    - Each train contains exactly one marker per boundary side of its target region.
#    - Those markers have strictly increasing centroid distances in the canonical side order
#      filtered by the target region's available sides: up, right, down, left.
#    - The generator rejects ambiguous samples so exactly one dihedral transform satisfies the
#      ordered-side signature.


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each train onto the unique station with the same coupler count, orienting it by ranked marker distances.

    Input structure:
    - Colored edge/internal stripes partition the canvas into rectangular regions over a dominant background.
    - Color 8 is the train body and color 9 serves both as isolated station cells and as train-attached couplers.
    - Each train also carries several extra marker pixels of a dedicated non-stripe color; the order of their distances
      from the coupler centroid encodes which transformed side should end up facing up, right, down, then left.

    Rule:
    - Identify stations as 9-clusters not touching any 8, and identify trains as non-stripe connected components containing 8.
    - Match each train to the unique station group whose size equals its coupler count.
    - For the target region, read which absolute sides are bordered by stripes; among all flips/rotations, keep the unique one
      whose markers, when sorted from nearest to farthest from the transformed coupler centroid, point toward those sides in
      canonical order up → right → down → left, skipping missing sides.
    - Erase the train from its source location, align its couplers exactly onto the matched station cells, and redraw only the
      transformed non-background, non-9 train pixels at the docked location.

    Key insight:
    - Marker color no longer tells you which side a marker stands for. The side identity is recovered from distance rank, so
      orientation comes from combining target-region side availability with the ordered radial signature around the couplers.
    """
    inp = np.asarray(grid, dtype=int)
    grid = inp.copy()
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
        return stripe, colors, (top_h, bot_h, left_w, right_w)

    def find_internal_stripes(edge_info):
        top_h, bot_h, left_w, right_w = edge_info
        r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()

        c = c0
        while c < c1:
            col = grid[r0:r1, c]
            if col.size and (col == col[0]).all() and col[0] != bg:
                color = int(col[0])
                c2 = c
                while c2 + 1 < c1:
                    col2 = grid[r0:r1, c2 + 1]
                    if col2.size and (col2 == col2[0]).all() and int(col2[0]) == color:
                        c2 += 1
                    else:
                        break
                stripe[r0:r1, c : c2 + 1] = True
                colors.add(color)
                c = c2 + 1
            else:
                c += 1

        r = r0
        while r < r1:
            row = grid[r, c0:c1]
            if row.size and (row == row[0]).all() and row[0] != bg:
                color = int(row[0])
                r2 = r
                while r2 + 1 < r1:
                    row2 = grid[r2 + 1, c0:c1]
                    if row2.size and (row2 == row2[0]).all() and int(row2[0]) == color:
                        r2 += 1
                    else:
                        break
                stripe[r : r2 + 1, c0:c1] = True
                colors.add(color)
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
                ok = all(
                    grid[r + dr, c + dc] != 8
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (dr != 0 or dc != 0) and 0 <= r + dr < H and 0 <= c + dc < W
                )
                if ok:
                    sm[r, c] = True
        struct = np.ones((3, 3), dtype=int)
        lab, n = label(sm, structure=struct)
        groups = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            groups.append([(int(r), int(c)) for r, c in coords])
        return groups

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

    def boundary_dirs_for_region(reg, stripe_colors):
        dirs = {}
        r0, r1, c0, c1 = reg["r0"], reg["r1"], reg["c0"], reg["c1"]
        if r0 - 1 >= 0:
            mc = Counter(grid[r0 - 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 0
        if r1 + 1 < H:
            mc = Counter(grid[r1 + 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 2
        if c0 - 1 >= 0:
            mc = Counter(grid[r0:r1 + 1, c0 - 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 3
        if c1 + 1 < W:
            mc = Counter(grid[r0:r1 + 1, c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 1
        return dirs

    def xform_coords(coords, shape, flip_ud, rot_k):
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape, dtype=bool), flip_ud, rot_k).shape
        return list(result), new_shape

    def ordered_region_dirs(divider_dirs):
        present = set(divider_dirs.values())
        return [d for d in (0, 1, 2, 3) if d in present]

    def direction_from_delta(dr, dc):
        return (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)

    def marker_signature(marker_pts, coupler_pts):
        ref_r = sum(r for r, _ in coupler_pts) / len(coupler_pts)
        ref_c = sum(c for _, c in coupler_pts) / len(coupler_pts)
        scored = []
        for mr, mc in marker_pts:
            dr = mr - ref_r
            dc = mc - ref_c
            d2 = dr * dr + dc * dc
            scored.append((d2, direction_from_delta(dr, dc)))
        scored.sort(key=lambda item: item[0])
        for i in range(len(scored) - 1):
            if abs(scored[i][0] - scored[i + 1][0]) < 1e-9:
                return None
        return [dirc for _d2, dirc in scored]

    def choose_best_op(train, expected_dirs):
        bbox0 = train["bbox"]
        H0, W0 = bbox0.shape
        assert train["marker_pixels"], "Each train must expose distance-coded markers"
        assert len(train["marker_pixels"]) == len(expected_dirs), "Marker count must match target side count"
        assert train["body9"], "Each train must carry at least one coupler"

        matches = []
        for flip_ud in (False, True):
            for k in range(4):
                body9_t = list(transform_coords(train["body9"], (H0, W0), flip_ud, k))
                marker_t = list(transform_coords(train["marker_pixels"], (H0, W0), flip_ud, k))
                sig = marker_signature(marker_t, body9_t)
                if sig == expected_dirs:
                    matches.append((flip_ud, k))
        assert len(matches) == 1, "Distance-ranked markers must determine a unique orientation"
        return matches[0]

    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups()
    groups_by_size = defaultdict(list)
    for group in groups:
        groups_by_size[len(group)].append(group)
    station_set = set(pt for group in groups for pt in group)

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
        body9, marker_pixels = [], []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 9 and (r, c) not in station_set:
                body9.append((int(r - r0), int(c - c0)))
            if v not in (bg, 8, 9):
                marker_pixels.append((int(r - r0), int(c - c0)))
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
        assert candidate_groups is not None and len(candidate_groups) == 1, (
            f"Expected one station group for coupler count {n9}"
        )
        group = candidate_groups[0]
        gset = set(group)
        matching_regions = [
            reg
            for reg in regions
            if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in group)
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        reg = matching_regions[0]
        divider_dirs = boundary_dirs_for_region(reg, stripe_colors)
        expected_dirs = ordered_region_dirs(divider_dirs)
        op = choose_best_op(train, expected_dirs)
        tbbox = apply_transform(train["bbox"], *op)
        t_body9, (H1, W1) = xform_coords(train["body9"], train["bbox"].shape, *op)
        body9_set = set(t_body9)
        matching_offsets = {
            (sr - br, sc - bc)
            for br, bc in body9_set
            for sr, sc in gset
            if all((r + sr - br, c + sc - bc) in gset for r, c in body9_set)
        }
        assert len(matching_offsets) == 1, "Couplers must align to a unique station-group offset"
        off_r, off_c = next(iter(matching_offsets))

        ok = True
        for r in range(H1):
            for c in range(W1):
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
        assert ok, "Docked train must fit entirely within its target region"

        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v != bg and v != 9:
                    out[r + off_r, c + off_c] = v
    return out


# --------------- generator ---------------

GRID_H_RANGE = (13, 31)
GRID_W_RANGE = (16, 29)
GRID_MIN_AREA = 300
MIN_REGION_DIM = 10
MIN_REGION_DIRS = 3
INTERNAL_STRIPE_THICK = 2

STRIPE_THICK_RANGE = (2, 4)
INTERNAL_PROB = 0.62

TRAIN_COUNTS = [2, 3]
STATION_SIZES = [1, 2, 4]

BODY_MIN = 5
BODY_MAX = 20
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)

MARKER_SEARCH_RADIUS = 4
MARKER_MIN_D2 = 2.0

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


def _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
    for reg in regions:
        dirs = {}
        if top and reg["r0"] == top:
            dirs[edge_col["top"]] = 0
        if bottom and reg["r1"] == H - bottom - 1:
            dirs[edge_col["bottom"]] = 2
        if left and reg["c0"] == left:
            dirs[edge_col["left"]] = 3
        if right and reg["c1"] == W - right - 1:
            dirs[edge_col["right"]] = 1
        if internal:
            kind, pos, col = internal
            if kind == "v":
                dirs[col] = 1 if reg["c1"] == pos - 1 else 3
            else:
                dirs[col] = 2 if reg["r1"] == pos - 1 else 0
        reg["dirs"] = dirs
        rh = reg["r1"] - reg["r0"] + 1
        rw = reg["c1"] - reg["c0"] + 1
        if rh < MIN_REGION_DIM or rw < MIN_REGION_DIM or len(dirs) < MIN_REGION_DIRS:
            return False
    return True


def _layout():
    for _ in range(LAYOUT_RETRIES):
        use_internal = random.random() < INTERNAL_PROB
        if use_internal:
            orient = random.choice(["v", "h"])
            if orient == "v":
                edges = ["left", "right", random.choice(["top", "bottom"])]
            else:
                edges = ["top", "bottom", random.choice(["left", "right"])]
        else:
            orient = None
            edges = random.sample(["top", "bottom", "left", "right"], random.choice([3, 4]))

        thick = {e: random.randint(*STRIPE_THICK_RANGE) for e in edges}
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
        pool = COLOR_POOL[:]
        bg = random.choice(pool)
        pool.remove(bg)
        stripe_cols = random.sample(pool, len(edges) + int(use_internal))
        for color in stripe_cols:
            pool.remove(color)
        if not pool:
            continue
        marker_color = random.choice(pool)

        edge_col = dict(zip(edges, stripe_cols[: len(edges)]))
        internal_col = stripe_cols[-1] if use_internal else None
        grid = np.full((H, W), bg, dtype=int)

        if top:
            grid[:top, :] = edge_col["top"]
        if bottom:
            grid[H - bottom :, :] = edge_col["bottom"]
        if left:
            grid[top : H - bottom, :left] = edge_col["left"]
        if right:
            grid[top : H - bottom, W - right :] = edge_col["right"]

        inner = {"r0": top, "r1": H - bottom - 1, "c0": left, "c1": W - right - 1}
        regions = []
        internal = None
        if orient == "v":
            lo = left + MIN_REGION_DIM
            hi = W - right - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            c = random.randint(lo, hi)
            grid[top : H - bottom, c : c + INTERNAL_STRIPE_THICK] = internal_col
            internal = ("v", c, internal_col)
            regions = [
                {"r0": top, "r1": H - bottom - 1, "c0": left, "c1": c - 1},
                {"r0": top, "r1": H - bottom - 1, "c0": c + INTERNAL_STRIPE_THICK, "c1": W - right - 1},
            ]
        elif orient == "h":
            lo = top + MIN_REGION_DIM
            hi = H - bottom - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            r = random.randint(lo, hi)
            grid[r : r + INTERNAL_STRIPE_THICK, left : W - right] = internal_col
            internal = ("h", r, internal_col)
            regions = [
                {"r0": top, "r1": r - 1, "c0": left, "c1": W - right - 1},
                {"r0": r + INTERNAL_STRIPE_THICK, "r1": H - bottom - 1, "c0": left, "c1": W - right - 1},
            ]
        else:
            regions = [inner]

        if _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
            return grid, bg, regions, marker_color
    return None


def _station_shape(size):
    if size == 1:
        return {(0, 0)}
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    return {(0, 0), (0, 1), (1, 0), (1, 1)}


def _ordered_region_dirs_from_map(dirs):
    present = set(dirs.values())
    return [d for d in (0, 1, 2, 3) if d in present]


def _direction_from_delta(dr, dc):
    return (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)


def _marker_signature_from_points(markers, couplers):
    ref_r = sum(r for r, _ in couplers) / len(couplers)
    ref_c = sum(c for _, c in couplers) / len(couplers)
    scored = []
    for mr, mc in markers:
        dr = mr - ref_r
        dc = mc - ref_c
        d2 = dr * dr + dc * dc
        if d2 < MARKER_MIN_D2:
            return None
        scored.append((d2, _direction_from_delta(dr, dc)))
    scored.sort(key=lambda item: item[0])
    for i in range(len(scored) - 1):
        if abs(scored[i][0] - scored[i + 1][0]) < 1e-9:
            return None
    return [dirc for _d2, dirc in scored]


def _candidate_marker_positions(couplers):
    r0, c0, r1, c1 = _bbox(couplers)
    ref_r = sum(r for r, _ in couplers) / len(couplers)
    ref_c = sum(c for _, c in couplers) / len(couplers)
    out = defaultdict(list)
    for r in range(r0 - MARKER_SEARCH_RADIUS, r1 + MARKER_SEARCH_RADIUS + 1):
        for c in range(c0 - MARKER_SEARCH_RADIUS, c1 + MARKER_SEARCH_RADIUS + 1):
            if (r, c) in couplers:
                continue
            dr = r - ref_r
            dc = c - ref_c
            d2 = dr * dr + dc * dc
            if d2 < MARKER_MIN_D2:
                continue
            dirc = _direction_from_delta(dr, dc)
            out[dirc].append((d2, (r, c)))
    for dirc, items in out.items():
        random.shuffle(items)
        items.sort(key=lambda item: item[0])
        out[dirc] = items
    return out


def _select_distance_markers(couplers, expected_dirs):
    grouped = _candidate_marker_positions(couplers)
    if any(d not in grouped for d in expected_dirs):
        return None

    chosen = []
    used = set()

    def rec(idx, prev_d2):
        if idx == len(expected_dirs):
            return list(chosen)
        dirc = expected_dirs[idx]
        for d2, pos in grouped[dirc]:
            if d2 <= prev_d2 + 1e-9 or pos in used:
                continue
            trial_cells = set(couplers) | used | {pos}
            h, w = _dims(trial_cells)
            if h > PATCH_H_RANGE[1] or w > PATCH_W_RANGE[1]:
                continue
            used.add(pos)
            chosen.append((pos, dirc, d2))
            result = rec(idx + 1, d2)
            if result is not None:
                return result
            chosen.pop()
            used.remove(pos)
        return None

    return rec(0, -1.0)


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
            exp = [q for q in cands if q[0] < r0 or q[0] > r1]
            cands = exp or cands
        elif w < PATCH_W_RANGE[0]:
            _, c0, _, c1 = _bbox(all_cells)
            exp = [q for q in cands if q[1] < c0 or q[1] > c1]
            cands = exp or cands
        body.add(random.choice(cands))


def _build_docked_patch(bg, dirs, size, marker_color):
    couplers = _station_shape(size)
    expected_dirs = _ordered_region_dirs_from_map(dirs)
    selected = _select_distance_markers(couplers, expected_dirs)
    if selected is None:
        return None

    markers = [(pos, marker_color, dirc) for pos, dirc, _d2 in selected]
    marker_cells = {pos for pos, _color, _dirc in markers}
    blocked = set(couplers) | marker_cells
    contacts = [q for r, c in couplers for dr, dc in DIRS4 if (q := (r + dr, c + dc)) not in blocked]
    if not contacts:
        return None

    body = {random.choice(contacts)}
    all_seed = set(couplers) | marker_cells | body
    sr0, sc0, sr1, sc1 = _bbox(all_seed)
    bounds = (sr0 - PATH_BOUNDS_PAD, sr1 + PATH_BOUNDS_PAD, sc0 - PATH_BOUNDS_PAD, sc1 + PATH_BOUNDS_PAD)

    for p, _color, direction in markers:
        goal = _inward_cell(p, direction)
        if goal in blocked:
            return None
        if not _path_to(body, goal, blocked, bounds):
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

    sig = _marker_signature_from_points(list(marker_cells), list(couplers))
    if sig != expected_dirs:
        return None

    r0, c0, _, _ = _bbox(all_cells)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = 8
    for (r, c), color, _direction in markers:
        patch[r - r0, c - c0] = color
    rel_couplers = []
    for r, c in couplers:
        patch[r - r0, c - c0] = 9
        rel_couplers.append((r - r0, c - c0))
    return patch, rel_couplers


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


def _place_docked_trains(base, bg, regions, marker_color):
    H, W = base.shape
    output = base.copy()
    occupied = output != bg

    n_trains = random.choice(TRAIN_COUNTS)
    sizes = random.sample(STATION_SIZES, n_trains)
    trains = []
    for size in sizes:
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            result = _build_docked_patch(bg, reg["dirs"], size, marker_color)
            if result is None:
                continue
            patch, coupler_rel = result
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), coupler_rel, True, 0)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(output, patch, r0, c0, transparent=bg)
            occupied |= _dilate((output != bg) & ~occupied) | (output != bg)
            station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
            trains.append({"patch": patch, "station": station_abs})
            break
        else:
            return None
    return output, trains


def _scatter_trains(base, bg, regions, trains):
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
        non_id = [op for op in valid_ops[0] if op != (False, 0)]
        if not non_id:
            return None
        ops[0] = random.choice(non_id)

    forbidden = _dilate(base != bg) | _dilate(station_mask)
    for train, op in zip(trains, ops):
        in_patch = _inverse_patch(train["patch"], op)
        regs = regions[:]
        random.shuffle(regs)
        all_locs = []
        for reg in regs:
            locs = _patch_locations(reg, in_patch, bg, forbidden, None, False, 1)
            all_locs.extend((reg, p) for p in locs)
        if not all_locs:
            return None
        _, (r0, c0) = random.choice(all_locs)
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        ph, pw = in_patch.shape
        patch_mask = np.zeros((H, W), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
        forbidden |= _dilate(patch_mask)

    return input_grid


def _extract_marker_metrics(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    def edge_info():
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()
        top_h = 0
        for r in range(H):
            row = g[r, :]
            if (row == row[0]).all() and row[0] != bg:
                top_h += 1
            else:
                break
        if top_h:
            stripe[:top_h, :] = True
            colors.add(int(g[0, 0]))
        bot_h = 0
        for r in range(H - 1, -1, -1):
            row = g[r, :]
            if (row == row[0]).all() and row[0] != bg:
                bot_h += 1
            else:
                break
        if bot_h:
            stripe[H - bot_h :, :] = True
            colors.add(int(g[H - 1, 0]))
        inner_r0, inner_r1 = top_h, H - bot_h
        left_w = 0
        for c in range(W):
            col = g[inner_r0:inner_r1, c]
            if col.size == 0:
                break
            if (col == col[0]).all() and col[0] != bg:
                left_w += 1
            else:
                break
        if left_w:
            stripe[inner_r0:inner_r1, :left_w] = True
            colors.add(int(g[inner_r0, 0]))
        right_w = 0
        for c in range(W - 1, -1, -1):
            col = g[inner_r0:inner_r1, c]
            if col.size == 0:
                break
            if (col == col[0]).all() and col[0] != bg:
                right_w += 1
            else:
                break
        if right_w:
            stripe[inner_r0:inner_r1, W - right_w :] = True
            colors.add(int(g[inner_r0, W - 1]))
        return stripe, colors

    stripe_mask, stripe_colors = edge_info()
    comp_mask = (g != bg) & (~stripe_mask)
    lab, n = label(comp_mask)
    train_metrics = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        vals = [int(g[r, c]) for r, c in coords]
        if 8 not in vals:
            continue
        station_touch_free = []
        markers = []
        for r, c in coords:
            v = int(g[r, c])
            if v == 9:
                ok = any(
                    0 <= r + dr < H and 0 <= c + dc < W and int(g[r + dr, c + dc]) == 8
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if dr or dc
                )
                if ok:
                    station_touch_free.append((int(r), int(c)))
            elif v not in (bg, 8) and v not in stripe_colors:
                markers.append((int(r), int(c), v))
        if not station_touch_free or not markers:
            continue
        couplers = [(r, c) for r, c in station_touch_free]
        mcoords = [(r, c) for r, c, _v in markers]
        sig = _marker_signature_from_points(mcoords, couplers)
        train_metrics.append(
            {
                "marker_count": len(markers),
                "same_color": len({v for _r, _c, v in markers}) == 1,
                "unique_distances": sig is not None,
            }
        )
    return train_metrics


def variant_metrics(grid: np.ndarray) -> dict:
    trains = _extract_marker_metrics(grid)
    return {
        "train_count": len(trains),
        "all_same_color": bool(trains) and all(t["same_color"] for t in trains),
        "all_unique_distances": bool(trains) and all(t["unique_distances"] for t in trains),
        "marker_counts": sorted(t["marker_count"] for t in trains),
    }


def _gen_attempt():
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions, marker_color = layout

    placed = _place_docked_trains(base, bg, regions, marker_color)
    if placed is None:
        return None
    output, trains = placed

    input_grid = _scatter_trains(base, bg, regions, trains)
    if input_grid is None:
        return None

    counts = Counter(map(int, input_grid.flatten()))
    if counts[bg] != max(counts.values()):
        return None

    metrics = variant_metrics(input_grid)
    if not metrics["all_same_color"] or not metrics["all_unique_distances"]:
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
    raise RuntimeError("failed to generate valid 6e4f6532 D2-alt3 input")
