# Combo variant of 6e4f6532 (index 14)
# combo_spec: D2/3+D14/0
# Name: 距离编码方向+单标记极简设计

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior trace:
#    Original 6e4f6532 used 2-3 stripe-colored marker pixels. Their colors matched boundary
#    stripes, and their positions around the couplers jointly determined the docking transform.
#    D14/0 collapses that down to one marker, so a single pixel must carry both the primary
#    direction cue and the mirror-disambiguation cue. D2/3 replaces one of the original
#    multi-marker direction channels with distance encoding. The fused rule used here is:
#    marker color still names the target boundary side that the marker should attach toward.
#    Because there is only one marker, its position around the 8-body is now the only visible
#    orthogonal cue, and the marker's distance to the coupler centroid is bucketed into
#    {near, far} to choose which orthogonal branch that position must realize. Example: if the
#    marker color matches the target region's top stripe, then after the correct transform the
#    marker must attach above the body, and a near marker means its centroid position must lean
#    to the left of the body whereas a far marker means it must lean to the right. The fused
#    rule therefore really uses both D14's single-marker placement and D2's distance code.
# b) Complexity calibration:
#    The puzzle keeps the original coupler-count matching, stripe partitioning, station shapes,
#    and output-first generation. The only modified channel is train orientation. To avoid
#    trivialization, the generator only chooses marker colors whose target region exposes both
#    orthogonal boundary sides, so color plus region geometry alone is insufficient. To avoid
#    overcomplication, the distance code has only two clearly separated bins: near and far.
# c) Hidden-assumption audit:
#    - Original / D14 code assumed either multiple markers or a unique orthogonal mate could be
#      recovered directly from the region geometry. That is deliberately false here: for the
#      chosen marker color, both orthogonal boundary sides are present.
#    - Solver lines that used `orth_dirs = [...]` and expected length 1 had to be replaced with
#      an explicit distance-bucket decoder.
#    - Generator lines that sampled a single `(color_dir, pos_dir)` pair from region geometry had
#      to be replaced with specs where `pos_dir` is derived from `(color_dir, distance_bucket)`.
#    - Marker parsing still assumes exactly one stripe-colored marker per train, and generator
#      construction enforces that invariant.
# d) Placement legality:
#    The generator still builds the docked output first, then inverse-transforms and scatters
#    trains back into the input. Marker placement is legal because the marker is always a
#    4-neighbor protrusion from the 8-body, never overlaps couplers or stripes, and is rejected
#    unless the final combined bbox stays inside the original patch size limits.
# e) Visual observability:
#    Humans can see the fusion because each train has exactly one clue pixel, and samples use the
#    same cue color with visibly different marker radii. The target region also visibly offers
#    both orthogonal stripe sides for that color, so the extra distance distinction is doing real
#    work instead of restating geometry that was already obvious.
# f) Visual cleanliness:
#    The grid remains visually close to the base task: rectangular regions, sparse stations, and
#    compact train sprites. The single-marker design reduces clutter relative to the original
#    multi-marker trains, and the near/far buckets are separated enough to be noticeable without
#    sprawling into oversized sprites.
# g) Change inferability:
#    No arbitrary hidden color mapping is introduced. Marker colors still come from visible stripe
#    colors, and the only new learned code is a binary near/far distinction that repeatedly maps
#    to the two orthogonal branches. This is small enough to infer from 2-4 examples.
# New solver assumptions after the change:
#    - Each train has exactly one stripe-colored marker pixel.
#    - The marker touches the 8-body on exactly one side.
#    - For the chosen marker color, the target region contains both orthogonal boundary sides.
#    - Marker distance to the coupler centroid lands in one of two generator-controlled bins,
#      and samples outside those bins are rejected during generation.


# ── Tunable constants ────────────────────────────────────────────────

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

BODY_GROWTH_ITERS = 160
PATH_BOUNDS_PAD = 4

LAYOUT_RETRIES = 200
DOCK_RETRIES = 24
GENERATE_RETRIES = 300

COLOR_POOL = list(range(8))

# The distance code must be visually separable and stable under all D4 transforms.
NEAR_D2_MAX = 8.0
FAR_D2_MIN = 12.0


# ── Constants ────────────────────────────────────────────────────────

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


# ── Shared primitives ───────────────────────────────────────────────

def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def dominant_dir(dr: float, dc: float) -> int | None:
    if abs(dr) == abs(dc):
        return None
    if abs(dr) > abs(dc):
        return 0 if dr < 0 else 2
    return 3 if dc < 0 else 1


def transform_dir(direction: int, flip_ud: bool = False, rot_k: int = 0) -> int:
    dr, dc = DIRS4[direction]
    if flip_ud:
        dr = -dr
    for _ in range(rot_k % 4):
        dr, dc = -dc, dr
    for idx, (rr, cc) in enumerate(DIRS4):
        if (dr, dc) == (rr, cc):
            return idx
    raise ValueError("invalid transformed direction")


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


def _centroid(coords):
    return (
        sum(r for r, _ in coords) / len(coords),
        sum(c for _, c in coords) / len(coords),
    )


def _marker_d2(marker, couplers):
    cr, cc = _centroid(couplers)
    dr = marker[0] - cr
    dc = marker[1] - cc
    return dr * dr + dc * dc


def _distance_bucket_from_d2(d2: float) -> int | None:
    if d2 <= NEAR_D2_MAX:
        return 0
    if d2 >= FAR_D2_MIN:
        return 1
    return None


def _orth_pair(direction: int) -> tuple[int, int]:
    return (direction - 1) % 4, (direction + 1) % 4


def _target_pos_dir(color_dir: int, dist_code: int) -> int:
    ccw, cw = _orth_pair(color_dir)
    return ccw if dist_code == 0 else cw


def _eligible_marker_specs(dirs):
    present = set(dirs.values())
    options = []
    for color, color_dir in dirs.items():
        ccw, cw = _orth_pair(color_dir)
        if ccw in present and cw in present:
            options.append((color, color_dir, 0, ccw))
            options.append((color, color_dir, 1, cw))
    return options


def _marker_attach_dirs(marker, body):
    mr, mc = marker
    body4 = set(body)
    hits = []
    for d, (dr, dc) in enumerate(DIRS4):
        if (mr - dr, mc - dc) in body4:
            hits.append(d)
    return hits


def _marker_pos_dir(body, marker):
    br, bc = _centroid(body)
    return dominant_dir(marker[0] - br, marker[1] - bc)


def _op_pos_dirs_for_color(body, marker, bbox_shape, local_attach_dir, color_dir):
    pos_dirs = set()
    for flip_ud, rot_k in OPS:
        if transform_dir(local_attach_dir, flip_ud, rot_k) != color_dir:
            continue
        body_t = list(transform_coords(body, bbox_shape, flip_ud, rot_k))
        marker_t = list(transform_coords([marker], bbox_shape, flip_ud, rot_k))[0]
        pos_dir = _marker_pos_dir(body_t, marker_t)
        if pos_dir is not None:
            pos_dirs.add(pos_dir)
    return pos_dirs


# ── Solver ──────────────────────────────────────────────────────────

def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each train onto the unique station with the same coupler count, orienting it from one marker pixel.

    Input structure:
    - Colored edge and optional internal stripes partition the grid into rectangular regions over a dominant background.
    - Color 8 is train body, color 9 is used both for isolated station cells and for train-attached couplers.
    - Each train has exactly one extra marker pixel whose color is one of the target region's stripe colors.

    Rule:
    - Detect stations as 9-clusters that do not touch any 8, and detect trains as non-stripe connected components containing 8.
    - Match each train to the unique station group whose size equals its coupler count.
    - For the target region, interpret the marker color as the boundary side that the marker must attach toward. The single
      marker's placement around the 8-body supplies the orthogonal pose cue, and its distance to the coupler centroid chooses
      which orthogonal branch is intended (near = counterclockwise, far = clockwise). Among all flips and rotations, keep the
      unique one whose transformed attachment side and transformed marker position around the 8-body realize that pair.
    - Clear the train from its source location, align its couplers exactly onto the matched station cells, and redraw only the
      transformed non-background, non-9 pixels at the docked location.

    Key insight:
    - The train now offers only one clue pixel, so color alone is insufficient: the same color is placed in regions where both
      orthogonal interpretations are geometrically possible. Distance to the couplers breaks that ambiguity and selects the
      correct mirrored pose.
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

    def choose_best_op(train, divider_dirs):
        marker_pixels = [(v, r, c) for v, r, c in train["marker_pixels"] if v in divider_dirs]
        assert len(marker_pixels) == 1, "Each train must expose exactly one divider-colored marker"
        assert train["body9"], "Each train must carry at least one coupler"

        color, mr0, mc0 = marker_pixels[0]
        marker_neighbors = _marker_attach_dirs((mr0, mc0), train["body8"])
        assert len(marker_neighbors) == 1, "Marker must touch the 8-body on exactly one side"

        color_dir = divider_dirs[color]
        ccw, cw = _orth_pair(color_dir)
        present = set(divider_dirs.values())
        assert ccw in present and cw in present, "Target region must keep both orthogonal branches available"

        d2 = _marker_d2((mr0, mc0), train["body9"])
        dist_code = _distance_bucket_from_d2(d2)
        assert dist_code is not None, "Marker distance must land in a legal near/far bucket"
        target_pos = _target_pos_dir(color_dir, dist_code)

        local_attach_dir = marker_neighbors[0]
        matches = []
        for flip_ud in (False, True):
            for rot_k in range(4):
                body_t = list(transform_coords(train["body8"], train["bbox"].shape, flip_ud, rot_k))
                br, bc = _centroid(body_t)
                marker_t = list(transform_coords([(mr0, mc0)], train["bbox"].shape, flip_ud, rot_k))[0]
                pos_dir = dominant_dir(marker_t[0] - br, marker_t[1] - bc)
                if pos_dir != target_pos:
                    continue
                attach_dir = transform_dir(local_attach_dir, flip_ud, rot_k)
                if attach_dir == color_dir:
                    matches.append((flip_ud, rot_k))

        assert len(matches) == 1, "Single-marker cue must determine a unique transform"
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
        body8, body9, marker_pixels = [], [], []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 8:
                body8.append((int(r - r0), int(c - c0)))
            elif v == 9 and (r, c) not in station_set:
                body9.append((int(r - r0), int(c - c0)))
            elif v in stripe_colors:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body8": body8,
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
        op = choose_best_op(train, divider_dirs)
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


# ── Layout construction ──────────────────────────────────────────────

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
            return grid, bg, regions
    return None


# ── Station / body / marker construction ────────────────────────────

def _station_shape(size):
    if size == 1:
        return {(0, 0)}
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    return {(0, 0), (0, 1), (1, 0), (1, 1)}


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


def _position_strength(marker, centroid, direction):
    dr = marker[0] - centroid[0]
    dc = marker[1] - centroid[1]
    if direction == 0:
        return -dr - abs(dc)
    if direction == 2:
        return dr - abs(dc)
    if direction == 3:
        return -dc - abs(dr)
    return dc - abs(dr)


def _marker_candidates(body, couplers, color_dir, pos_dir, dist_code):
    body_ctr = _centroid(body)
    blocked = set(body) | set(couplers)
    dr, dc = DIRS4[color_dir]
    cands = []
    for br, bc in body:
        marker = (br + dr, bc + dc)
        if marker in blocked:
            continue
        neighbors = []
        for idx, (rr, cc) in enumerate(DIRS4):
            if (marker[0] - rr, marker[1] - cc) in body:
                neighbors.append(idx)
        if neighbors != [color_dir]:
            continue
        dirc = dominant_dir(marker[0] - body_ctr[0], marker[1] - body_ctr[1])
        if dirc != pos_dir:
            continue
        d2 = _marker_d2(marker, couplers)
        if _distance_bucket_from_d2(d2) != dist_code:
            continue
        all_cells = blocked | {marker}
        h, w = _dims(all_cells)
        if not (PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1] and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]):
            continue
        score = _position_strength(marker, body_ctr, pos_dir)
        if score <= 0:
            continue
        if dist_code == 0:
            pref = -d2
        else:
            pref = d2
        cands.append((score, pref, marker))
    cands.sort(reverse=True)
    return [marker for _score, _pref, marker in cands]


def _build_docked_patch(bg, dirs, size, preferred_dist_code=None):
    couplers = _station_shape(size)
    specs = _eligible_marker_specs(dirs)
    if not specs:
        return None

    blocked = set(couplers)
    contacts = [q for r, c in couplers for dr, dc in DIRS4 if (q := (r + dr, c + dc)) not in blocked]
    if not contacts:
        return None

    body = {random.choice(contacts)}
    sr0, sc0, sr1, sc1 = _bbox(set(couplers) | body)
    bounds = (sr0 - PATH_BOUNDS_PAD, sr1 + PATH_BOUNDS_PAD, sc0 - PATH_BOUNDS_PAD, sc1 + PATH_BOUNDS_PAD)

    if not _connect_couplers_to_body(couplers, body, blocked, bounds):
        return None
    if len(body) > BODY_MAX:
        return None

    target = random.randint(max(BODY_MIN, len(body)), BODY_MAX)
    _grow_body(body, blocked, target)

    all_cells = body | set(couplers)
    h, w = _dims(all_cells)
    if not (
        BODY_MIN <= len(body) <= BODY_MAX
        and PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1]
        and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]
    ):
        return None

    random.shuffle(specs)
    choices = []
    for color, color_dir, dist_code, pos_dir in specs:
        markers = _marker_candidates(body, couplers, color_dir, pos_dir, dist_code)
        for marker in markers[:4]:
            choices.append((color, color_dir, dist_code, pos_dir, marker))
    if not choices:
        return None

    if preferred_dist_code is not None:
        preferred = [choice for choice in choices if choice[2] == preferred_dist_code]
        if preferred:
            choices = preferred

    color, color_dir, dist_code, pos_dir, marker = random.choice(choices)
    all_cells = set(body) | set(couplers) | {marker}
    h, w = _dims(all_cells)
    if not (PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1] and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]):
        return None

    r0, c0, _, _ = _bbox(all_cells)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = 8
    patch[marker[0] - r0, marker[1] - c0] = color
    rel_couplers = []
    for r, c in couplers:
        patch[r - r0, c - c0] = 9
        rel_couplers.append((r - r0, c - c0))
    return patch, rel_couplers, {
        "marker_color": color,
        "color_dir": color_dir,
        "dist_code": dist_code,
        "pos_dir": pos_dir,
    }


# ── Placement search ─────────────────────────────────────────────────

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


# ── Generation pipeline ──────────────────────────────────────────────

def _place_docked_trains(base, bg, regions):
    H, W = base.shape
    output = base.copy()
    occupied = output != bg

    eligible_regions = [reg for reg in regions if _eligible_marker_specs(reg["dirs"])]
    if not eligible_regions:
        return None

    n_trains = random.choice(TRAIN_COUNTS)
    sizes = random.sample(STATION_SIZES, n_trains)
    preferred_codes = []
    if n_trains >= 2:
        preferred_codes = [0, 1]
        while len(preferred_codes) < n_trains:
            preferred_codes.append(random.randint(0, 1))
        random.shuffle(preferred_codes)
    trains = []
    for idx, size in enumerate(sizes):
        preferred_code = preferred_codes[idx] if idx < len(preferred_codes) else None
        for _ in range(DOCK_RETRIES):
            reg = random.choice(eligible_regions)
            result = _build_docked_patch(bg, reg["dirs"], size, preferred_code)
            if result is None:
                continue
            patch, coupler_rel, meta = result
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), coupler_rel, True, 0)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(output, patch, r0, c0, transparent=bg)
            occupied |= _dilate((output != bg) & ~occupied) | (output != bg)
            station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
            trains.append({"patch": patch, "station": station_abs, "meta": meta})
            break
        else:
            return None
    if n_trains >= 2 and {train["meta"]["dist_code"] for train in trains} != {0, 1}:
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


# ── Behavior metrics ─────────────────────────────────────────────────

def _extract_marker_metrics(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    def find_edge_stripes():
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
        return stripe, colors, (top_h, bot_h, left_w, right_w)

    def find_internal_stripes(edge_info):
        top_h, bot_h, left_w, right_w = edge_info
        r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()
        c = c0
        while c < c1:
            col = g[r0:r1, c]
            if col.size and (col == col[0]).all() and col[0] != bg:
                color = int(col[0])
                c2 = c
                while c2 + 1 < c1:
                    col2 = g[r0:r1, c2 + 1]
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
            row = g[r, c0:c1]
            if row.size and (row == row[0]).all() and row[0] != bg:
                color = int(row[0])
                r2 = r
                while r2 + 1 < r1:
                    row2 = g[r2 + 1, c0:c1]
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
            mc = Counter(g[r0 - 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 0
        if r1 + 1 < H:
            mc = Counter(g[r1 + 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 2
        if c0 - 1 >= 0:
            mc = Counter(g[r0:r1 + 1, c0 - 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 3
        if c1 + 1 < W:
            mc = Counter(g[r0:r1 + 1, c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 1
        return dirs

    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    station_mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if g[r, c] != 9:
                continue
            ok = all(
                g[r + dr, c + dc] != 8
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr != 0 or dc != 0) and 0 <= r + dr < H and 0 <= c + dc < W
            )
            if ok:
                station_mask[r, c] = True

    comp_mask = (g != bg) & (~stripe_mask)
    lab, n = label(comp_mask)
    metrics = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        vals = [int(g[r, c]) for r, c in coords]
        if 8 not in vals:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox_shape = (int(r1 - r0 + 1), int(c1 - c0 + 1))
        body8, couplers, markers = [], [], []
        for r, c in coords:
            v = int(g[r, c])
            if v == 8:
                body8.append((int(r - r0), int(c - c0)))
            elif v == 9 and not station_mask[r, c]:
                couplers.append((int(r - r0), int(c - c0)))
            elif v in stripe_colors:
                markers.append((int(v), int(r - r0), int(c - c0)))
        if len(markers) != 1 or not couplers:
            continue
        marker = markers[0]
        color, mr, mc = marker
        reg_hits = []
        abs_cells = [(int(r), int(c)) for r, c in coords]
        for reg in regions:
            if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in abs_cells):
                reg_hits.append(reg)
        need_distance = False
        need_position = False
        if len(reg_hits) == 1:
            dirs = boundary_dirs_for_region(reg_hits[0], stripe_colors)
            if color in dirs:
                attach_dirs = _marker_attach_dirs((mr, mc), body8)
                if len(attach_dirs) == 1:
                    pos_dirs = _op_pos_dirs_for_color(
                        body8,
                        (mr, mc),
                        bbox_shape,
                        attach_dirs[0],
                        dirs[color],
                    )
                    ccw, cw = _orth_pair(dirs[color])
                    need_position = len(pos_dirs) >= 2
                    need_distance = ccw in pos_dirs and cw in pos_dirs
        dcode = _distance_bucket_from_d2(_marker_d2((mr, mc), couplers))
        metrics.append(
            {
                "single_marker": len(markers) == 1,
                "distance_code": dcode,
                "need_distance": need_distance,
                "need_position": need_position,
            }
        )
    return metrics


def variant_metrics(grid: np.ndarray) -> dict:
    trains = _extract_marker_metrics(grid)
    return {
        "train_count": len(trains),
        "all_single_marker": bool(trains) and all(t["single_marker"] for t in trains),
        "all_need_distance": bool(trains) and all(t["need_distance"] for t in trains),
        "all_need_position": bool(trains) and all(t["need_position"] for t in trains),
        "distance_codes": sorted(t["distance_code"] for t in trains if t["distance_code"] is not None),
        "has_near": any(t["distance_code"] == 0 for t in trains),
        "has_far": any(t["distance_code"] == 1 for t in trains),
    }


def _gen_attempt():
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions = layout

    placed = _place_docked_trains(base, bg, regions)
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
    if (
        not metrics["all_single_marker"]
        or not metrics["all_need_distance"]
        or not metrics["all_need_position"]
        or (metrics["train_count"] >= 2 and not (metrics["has_near"] and metrics["has_far"]))
    ):
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
