# Combo variant of 6e4f6532 (index 6)
# combo_spec: D3/1+D5/0+D6/0
# Name: 空心火车+重新上色+幽灵轮廓
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste

# Design analysis for the fused variant:
# - Concrete example:
#   A source train is now a 1-pixel-thick hollow 8-frame, with 2-3 stripe-colored
#   marker pixels sitting inside the hole and a line-shaped 9 coupler group attached
#   just outside one side. In the base task the whole train was one connected blob, so
#   "label every non-bg component containing an 8" was enough. That breaks here because
#   the interior markers are intentionally disconnected from the frame. The solver now
#   identifies trains from 8-components first, then separately harvests outer 9-couplers
#   and inner marker pixels from the frame geometry.
# - Behavior change:
#   After docking, the moved frame stays hollow, its 8-body is repainted to the target
#   region's dominant boundary-stripe color, and the source location is not erased back
#   to background. Instead, only the source couplers+markers disappear, leaving the old
#   8-frame as a clean ghost outline. The output therefore contains three simultaneously
#   visible structures: the recolored docked train, the original station/stripe layout,
#   and the ghost frame at the source.
# - Complexity calibration / observability:
#   The train body is restricted to thin rectangular frames with 2-3 interior markers
#   so the hollow rule is obvious at a glance and the output ghost is visually crisp
#   instead of noisy. Recoloring uses the unique dominant boundary stripe of the target
#   region, so the new color has a local, human-visible explanation.
# - Implicit assumption audit:
#   The base generator used 4-cell 2x2 stations for size 4. That is physically awkward
#   for a thin external frame because some source 9s would fail the "adjacent to 8"
#   condition and would be misread as stations. This fused variant therefore keeps the
#   count vocabulary {1,2,4} but realizes size 4 as a length-4 line, which preserves the
#   count-matching logic while remaining compatible with a hollow 1-pixel frame.
# - Placement legality:
#   Source trains are scattered only into cells that are clear in both the input layout
#   and the eventual docked output, so ghost outlines never overlap stripes, stations,
#   or recolored docked bodies. Marker positions are chosen from interior cells whose
#   direction relative to the coupler centroid is actually readable by the solver's
#   orientation heuristic.


# --------------- shared helpers ---------------

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
DIR_TO_NAME = {0: "up", 1: "right", 2: "down", 3: "left"}


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def vector_to_dir(dr: float, dc: float) -> int:
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 2
    return 3 if dc < 0 else 1


def xform_coords(coords, shape, flip_ud, rot_k):
    result = transform_coords(coords, shape, flip_ud, rot_k)
    new_shape = apply_transform(np.zeros(shape, dtype=int), flip_ud, rot_k).shape
    return list(result), new_shape


def bbox_of(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def dilate8(mask: np.ndarray) -> np.ndarray:
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


def find_edge_stripes(grid: np.ndarray, bg: int):
    H, W = grid.shape
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


def find_internal_stripes(grid: np.ndarray, bg: int, edge_info):
    H, W = grid.shape
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


def regions_from_stripes(stripe_mask: np.ndarray):
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


def boundary_dirs_for_region(grid: np.ndarray, reg, stripe_colors):
    H, W = grid.shape
    dirs = {}
    r0, r1, c0, c1 = reg["r0"], reg["r1"], reg["c0"], reg["c1"]
    if r0 - 1 >= 0:
        mc = Counter(grid[r0 - 1, c0 : c1 + 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 0
    if r1 + 1 < H:
        mc = Counter(grid[r1 + 1, c0 : c1 + 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 2
    if c0 - 1 >= 0:
        mc = Counter(grid[r0 : r1 + 1, c0 - 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 3
    if c1 + 1 < W:
        mc = Counter(grid[r0 : r1 + 1, c1 + 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 1
    return dirs


def dominant_boundary_color_for_region(grid: np.ndarray, reg, stripe_colors):
    H, W = grid.shape
    counts = Counter()
    r0, r1, c0, c1 = reg["r0"], reg["r1"], reg["c0"], reg["c1"]
    width = c1 - c0 + 1
    height = r1 - r0 + 1
    if r0 - 1 >= 0:
        color = int(grid[r0 - 1, c0])
        if color in stripe_colors:
            rr = r0 - 1
            while rr >= 0 and np.all(grid[rr, c0 : c1 + 1] == color):
                counts[color] += width
                rr -= 1
    if r1 + 1 < H:
        color = int(grid[r1 + 1, c0])
        if color in stripe_colors:
            rr = r1 + 1
            while rr < H and np.all(grid[rr, c0 : c1 + 1] == color):
                counts[color] += width
                rr += 1
    if c0 - 1 >= 0:
        color = int(grid[r0, c0 - 1])
        if color in stripe_colors:
            cc = c0 - 1
            while cc >= 0 and np.all(grid[r0 : r1 + 1, cc] == color):
                counts[color] += height
                cc -= 1
    if c1 + 1 < W:
        color = int(grid[r0, c1 + 1])
        if color in stripe_colors:
            cc = c1 + 1
            while cc < W and np.all(grid[r0 : r1 + 1, cc] == color):
                counts[color] += height
                cc += 1
    assert counts, "Each target region must border at least one stripe color"
    ordered = counts.most_common()
    assert len(ordered) == 1 or ordered[0][1] > ordered[1][1], "Dominant boundary color must be unique"
    return int(ordered[0][0])


def station_groups(grid: np.ndarray):
    H, W = grid.shape
    sm = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 9:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] == 8:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                sm[r, c] = True
    lab, n = label(sm, structure=np.ones((3, 3), dtype=int))
    groups = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        groups.append([(int(r), int(c)) for r, c in coords])
    return groups


def choose_best_op(train, divider_dirs):
    bbox0 = train["bbox"]
    H0, W0 = bbox0.shape
    markers_by_color = defaultdict(list)
    for v, r, c in train["marker_pixels"]:
        if v in divider_dirs:
            markers_by_color[v].append((r, c))
    assert markers_by_color, "Each train must expose divider-colored markers"
    assert train["body9"], "Each train must carry at least one coupler"
    ref_r = sum(r for r, _ in train["body9"]) / len(train["body9"])
    ref_c = sum(c for _, c in train["body9"]) / len(train["body9"])
    best, best_score = (False, 0), -1
    for flip_ud in (False, True):
        for k in range(4):
            ref_t = list(
                transform_coords([(int(round(ref_r)), int(round(ref_c)))], (H0, W0), flip_ud, k)
            )
            rr_ref, cc_ref = ref_t[0]
            score = 0
            for color, pts in markers_by_color.items():
                mr = sum(r for r, _ in pts) / len(pts)
                mc = sum(c for _, c in pts) / len(pts)
                m_t = list(
                    transform_coords([(int(round(mr)), int(round(mc)))], (H0, W0), flip_ud, k)
                )
                rr_m, cc_m = m_t[0]
                dirc = vector_to_dir(rr_m - rr_ref, cc_m - cc_ref)
                if dirc == divider_dirs[color]:
                    score += 1
            if score > best_score:
                best_score, best = score, (flip_ud, k)
    return best


def extract_hollow_trains(
    grid: np.ndarray,
    bg: int,
    stripe_mask: np.ndarray,
    stripe_colors,
    station_set,
):
    H, W = grid.shape
    body_mask = grid == 8
    lab, n = label(body_mask, structure=np.ones((3, 3), dtype=int))
    trains = []
    used_markers = set()
    used_couplers = set()
    for k in range(1, n + 1):
        body_coords_arr = np.argwhere(lab == k)
        body_abs = [(int(r), int(c)) for r, c in body_coords_arr]
        body_set = set(body_abs)
        br0, bc0 = body_coords_arr.min(axis=0)
        br1, bc1 = body_coords_arr.max(axis=0)
        markers_abs = []
        for r in range(int(br0) + 1, int(br1)):
            for c in range(int(bc0) + 1, int(bc1)):
                v = int(grid[r, c])
                if v in stripe_colors and (r, c) not in used_markers:
                    markers_abs.append((v, r, c))
                    used_markers.add((r, c))
        couplers_abs = []
        for r in range(max(0, int(br0) - 1), min(H, int(br1) + 2)):
            for c in range(max(0, int(bc0) - 1), min(W, int(bc1) + 2)):
                if grid[r, c] != 9 or (r, c) in station_set or (r, c) in used_couplers:
                    continue
                if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in body_set):
                    couplers_abs.append((r, c))
                    used_couplers.add((r, c))
        assert couplers_abs, "Each hollow train must have external couplers"
        all_coords = body_abs + [(r, c) for _, r, c in markers_abs] + couplers_abs
        r0, c0, r1, c1 = bbox_of(all_coords)
        bbox = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        trains.append(
            {
                "bbox": bbox,
                "body_abs": body_abs,
                "marker_abs": [(r, c) for _, r, c in markers_abs],
                "extra_abs": [(r, c) for _, r, c in markers_abs] + couplers_abs,
                "body9": [(r - r0, c - c0) for r, c in couplers_abs],
                "marker_pixels": [(v, r - r0, c - c0) for v, r, c in markers_abs],
            }
        )
    return trains


def patch_locations(region, patch, bg, occupied, coupler_rel=None, station_clearance=False, margin=0):
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


# --------------- solver ---------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock hollow train frames onto matching stations, repaint the docked body, and leave a source ghost.

    Input structure:
    - Thick colored stripes still partition the board into rectangular regions around a single background.
    - Each movable train is now a hollow color-8 frame with 2-3 stripe-colored markers inside its hole.
    - Color 9 still serves double duty: isolated clusters are stations, while train couplers sit outside the frame and touch the 8-border.

    Rule:
    - Identify stations as 9-clusters not adjacent to any 8, and identify trains from their 8-frame components.
    - Match each train to the unique station whose size equals the train's coupler count, then rotate/flip until the interior markers face the same-colored boundary stripes of the destination region.
    - Dock by aligning the coupler shape exactly onto the station cells; keep the station 9s, keep the frame hollow, and recolor the docked 8-body to the dominant boundary-stripe color of that region.
    - At the source location, erase only the train's markers and couplers so the original 8-frame remains as a one-pixel ghost outline.

    Key insight: Coupler count still chooses the destination and stripe-colored markers still fix orientation, but now the train must be read as a hollow frame whose interior markers are disconnected, whose docked body changes color from the target region, and whose old shell remains behind as a ghost.
    """
    inp = grid
    grid = inp.copy()
    H, W = grid.shape
    bg = get_bg(grid)

    edge_mask, edge_colors, edge_info = find_edge_stripes(grid, bg)
    internal_mask, internal_colors = find_internal_stripes(grid, bg, edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups(grid)
    groups_by_size = defaultdict(list)
    for g in groups:
        groups_by_size[len(g)].append(g)
    station_set = set(pt for g in groups for pt in g)

    trains = extract_hollow_trains(grid, bg, stripe_mask, stripe_colors, station_set)

    out = grid.copy()
    for t in trains:
        for r, c in t["extra_abs"]:
            out[r, c] = bg

    for t in trains:
        n9 = len(t["body9"])
        assert n9 > 0, "Each train must carry at least one coupler"
        candidate_groups = groups_by_size.get(n9)
        assert candidate_groups is not None and len(candidate_groups) == 1, (
            f"Expected one station group for coupler count {n9}"
        )
        g = candidate_groups[0]
        gset = set(g)
        matching_regions = [
            rr
            for rr in regions
            if all(rr["r0"] <= r <= rr["r1"] and rr["c0"] <= c <= rr["c1"] for r, c in g)
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        reg = matching_regions[0]
        divider_dirs = boundary_dirs_for_region(grid, reg, stripe_colors)
        dock_color = dominant_boundary_color_for_region(grid, reg, stripe_colors)
        op = choose_best_op(t, divider_dirs)
        tbbox = apply_transform(t["bbox"], *op)
        t_body9, (H1, W1) = xform_coords(t["body9"], t["bbox"].shape, *op)
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
                if v == bg or v == 9:
                    continue
                out[r + off_r, c + off_c] = dock_color if v == 8 else v
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
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)
THREE_MARKER_PROB = 0.7
LAYOUT_RETRIES = 200
DOCK_RETRIES = 30
GENERATE_RETRIES = 300
COLOR_POOL = list(range(8))

LAST_GENERATION_INFO = {}
NEXT_TRAIN_COUNT_IDX = 0


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


def _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
    for reg in regions:
        dirs = {}
        weights = Counter()
        rh = reg["r1"] - reg["r0"] + 1
        rw = reg["c1"] - reg["c0"] + 1
        if top and reg["r0"] == top:
            dirs[edge_col["top"]] = 0
            weights[edge_col["top"]] += int(top * rw)
        if bottom and reg["r1"] == H - bottom - 1:
            dirs[edge_col["bottom"]] = 2
            weights[edge_col["bottom"]] += int(bottom * rw)
        if left and reg["c0"] == left:
            dirs[edge_col["left"]] = 3
            weights[edge_col["left"]] += int(left * rh)
        if right and reg["c1"] == W - right - 1:
            dirs[edge_col["right"]] = 1
            weights[edge_col["right"]] += int(right * rh)
        if internal:
            kind, pos, col = internal
            if kind == "v":
                dirs[col] = 1 if reg["c1"] == pos - 1 else 3
                weights[col] += int(INTERNAL_STRIPE_THICK * rh)
            else:
                dirs[col] = 2 if reg["r1"] == pos - 1 else 0
                weights[col] += int(INTERNAL_STRIPE_THICK * rw)
        reg["dirs"] = dirs
        ordered = weights.most_common()
        reg["dominant_color"] = (
            int(ordered[0][0])
            if ordered and (len(ordered) == 1 or ordered[0][1] > ordered[1][1])
            else None
        )
        if rh < MIN_REGION_DIM or rw < MIN_REGION_DIM or len(dirs) < MIN_REGION_DIRS:
            return False
        if reg["dominant_color"] is None:
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
        regions, internal = [], None
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
            grid[r : r + INTERNAL_STRIPE_THICK, left : W - right] = internal_col
            internal = ("h", r, internal_col)
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
            regions = [inner]

        if _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
            return grid, bg, regions
    return None


def _frame_dims_for_side(side):
    return (4, 8) if side in (0, 2) else (8, 4)


def _orthogonal_pair_exists(dirs):
    return any((a - b) % 4 in (1, 3) for a, b in combinations(dirs, 2))


def _coupler_shape(size, side, frame_h, frame_w):
    if side in (0, 2):
        run = 1 if size == 1 else size
        start_c = (frame_w - run) // 2
        row = -1 if side == 0 else frame_h
        return {(row, start_c + i) for i in range(run)}
    run = 1 if size == 1 else size
    start_r = (frame_h - run) // 2
    col = -1 if side == 3 else frame_w
    return {(start_r + i, col) for i in range(run)}


def _body_frame(frame_h, frame_w):
    body = set()
    for r in range(frame_h):
        for c in range(frame_w):
            if r in (0, frame_h - 1) or c in (0, frame_w - 1):
                body.add((r, c))
    return body


def _direction_score(cell, ref_r, ref_c, direction):
    r, c = cell
    if direction == 0:
        return (ref_r - r) * 10 - abs(ref_c - c)
    if direction == 2:
        return (r - ref_r) * 10 - abs(ref_c - c)
    if direction == 3:
        return (ref_c - c) * 10 - abs(ref_r - r)
    return (c - ref_c) * 10 - abs(ref_r - r)


def _marker_subset_and_cells(region_dirs, side, frame_h, frame_w, couplers):
    hole = [(r, c) for r in range(1, frame_h - 1) for c in range(1, frame_w - 1)]
    ref_r = sum(r for r, _ in couplers) / len(couplers)
    ref_c = sum(c for _, c in couplers) / len(couplers)
    candidates = {}
    for direction in range(4):
        cells = [
            cell
            for cell in hole
            if vector_to_dir(cell[0] - ref_r, cell[1] - ref_c) == direction
        ]
        cells.sort(key=lambda cell: _direction_score(cell, ref_r, ref_c, direction), reverse=True)
        candidates[direction] = cells

    available = [(color, direction) for color, direction in region_dirs.items() if direction != side]
    available = [(color, direction) for color, direction in available if candidates[direction]]
    random.shuffle(available)

    subsets = []
    if len(available) >= 3 and random.random() < THREE_MARKER_PROB:
        for subset in combinations(available, 3):
            dirs = [direction for _, direction in subset]
            if _orthogonal_pair_exists(dirs):
                subsets.append(subset)
    for subset in combinations(available, 2):
        dirs = [direction for _, direction in subset]
        if _orthogonal_pair_exists(dirs):
            subsets.append(subset)
    subsets.sort(key=len, reverse=True)
    for subset in subsets:
        markers = [(candidates[direction][0], color, direction) for color, direction in subset]
        return markers
    return None


def _build_hollow_patch(bg, region_dirs, size):
    sides = list(range(4))
    random.shuffle(sides)
    for side in sides:
        available_dirs = [direction for direction in region_dirs.values() if direction != side]
        if len(available_dirs) < 2 or not _orthogonal_pair_exists(available_dirs):
            continue
        frame_h, frame_w = _frame_dims_for_side(side)
        couplers = _coupler_shape(size, side, frame_h, frame_w)
        body = _body_frame(frame_h, frame_w)
        markers = _marker_subset_and_cells(region_dirs, side, frame_h, frame_w, couplers)
        if markers is None:
            continue
        all_cells = set(body) | set(couplers) | {cell for cell, _, _ in markers}
        r0, c0, r1, c1 = bbox_of(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in body:
            patch[r - r0, c - c0] = 8
        for (r, c), color, _ in markers:
            patch[r - r0, c - c0] = color
        rel_couplers = []
        for r, c in sorted(couplers):
            patch[r - r0, c - c0] = 9
            rel_couplers.append((r - r0, c - c0))
        ph, pw = patch.shape
        if not (PATCH_H_RANGE[0] <= ph <= PATCH_H_RANGE[1] and PATCH_W_RANGE[0] <= pw <= PATCH_W_RANGE[1]):
            continue
        return {
            "patch": patch,
            "coupler_rel": rel_couplers,
            "side": side,
            "marker_count": len(markers),
            "body_pixels": int(np.count_nonzero(patch == 8)),
        }
    return None


def _recolor_patch_body(patch, color):
    recolored = patch.copy()
    recolored[recolored == 8] = int(color)
    return recolored


def _place_docked_trains(base, bg, regions, n_trains):
    output = base.copy()
    occupied = output != bg
    sizes = random.sample(STATION_SIZES, n_trains)
    random.shuffle(sizes)
    trains = []
    for size in sizes:
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            if reg.get("dominant_color") is None:
                continue
            built = _build_hollow_patch(bg, reg["dirs"], size)
            if built is None:
                continue
            patch = built["patch"]
            coupler_rel = built["coupler_rel"]
            locs = patch_locations(reg, patch, bg, dilate8(occupied), coupler_rel, True, 0)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(output, _recolor_patch_body(patch, reg["dominant_color"]), r0, c0, transparent=bg)
            occupied = output != bg
            station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
            trains.append(
                {
                    "patch": patch,
                    "station": station_abs,
                    "dominant_color": reg["dominant_color"],
                    "marker_count": built["marker_count"],
                    "body_pixels": built["body_pixels"],
                    "size": size,
                    "target_region": reg,
                }
            )
            break
        else:
            return None
    return output, trains


def _scatter_trains(base, bg, regions, trains, output_docked):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True

    valid_ops = []
    for t in trains:
        valid = _valid_ops_for_patch(t["patch"])
        if not valid:
            return None
        valid_ops.append(valid)
    ops = [random.choice(valid) for valid in valid_ops]
    if all(op == (False, 0) for op in ops):
        non_id = [op for op in valid_ops[0] if op != (False, 0)]
        if not non_id:
            return None
        ops[0] = random.choice(non_id)

    forbidden = dilate8(output_docked != bg)
    source_infos = []
    for t, op in zip(trains, ops):
        in_patch = _inverse_patch(t["patch"], op)
        regs = regions[:]
        random.shuffle(regs)
        all_locs = []
        for reg in regs:
            locs = patch_locations(reg, in_patch, bg, forbidden, None, False, 1)
            all_locs.extend((reg, p) for p in locs)
        if not all_locs:
            return None
        _, (r0, c0) = random.choice(all_locs)
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        ph, pw = in_patch.shape
        patch_mask = np.zeros((H, W), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
        forbidden |= dilate8(patch_mask)
        body_cells = [(r0 + r, c0 + c) for r, c in np.argwhere(in_patch == 8)]
        source_infos.append({"body_cells": body_cells, "patch": in_patch, "op": op})
    return input_grid, source_infos


def _compose_output_with_ghosts(output_docked, source_infos):
    output = output_docked.copy()
    for info in source_infos:
        for r, c in info["body_cells"]:
            output[r, c] = 8
    return output


def _record_generation_info(trains, source_infos, output):
    global LAST_GENERATION_INFO
    LAST_GENERATION_INFO = {
        "n_trains": len(trains),
        "sizes": [int(t["size"]) for t in trains],
        "marker_counts": [int(t["marker_count"]) for t in trains],
        "body_pixels": [int(t["body_pixels"]) for t in trains],
        "dock_colors": [int(t["dominant_color"]) for t in trains],
        "ghost_pixels": int(sum(len(info["body_cells"]) for info in source_infos)),
        "recolored_pixels": int(sum(t["body_pixels"] for t in trains)),
        "output_8_pixels": int(np.count_nonzero(output == 8)),
    }


def _gen_attempt(target_n_trains):
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions = layout

    placed = _place_docked_trains(base, bg, regions, target_n_trains)
    if placed is None:
        return None
    output_docked, trains = placed

    scattered = _scatter_trains(base, bg, regions, trains, output_docked)
    if scattered is None:
        return None
    input_grid, source_infos = scattered
    output = _compose_output_with_ghosts(output_docked, source_infos)

    counts = Counter(map(int, input_grid.flatten()))
    if counts[bg] != max(counts.values()):
        return None
    if not source_infos or any(t["dominant_color"] == 8 for t in trains):
        return None
    try:
        solved = main(input_grid)
        if np.array_equal(solved, output):
            _record_generation_info(trains, source_infos, output)
            return input_grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    global NEXT_TRAIN_COUNT_IDX
    target_n_trains = TRAIN_COUNTS[NEXT_TRAIN_COUNT_IDX % len(TRAIN_COUNTS)]
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(target_n_trains)
        if grid is not None:
            NEXT_TRAIN_COUNT_IDX += 1
            return grid
    raise RuntimeError("failed to generate valid combo 6e4f6532 input")
