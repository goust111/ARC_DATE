# Combo variant of 6e4f6532 (index 17)
# combo_spec: D6/1+D16/0
# Name: 移动轨迹+分离式图例
#
# Design analysis summary
# - Concrete behavior trace: relative to the original task, stripe colors no longer tell trains how to
#   orient. A fixed corner legend now maps 2-3 colors to absolute directions, and train markers use
#   those legend colors. After a train is reoriented and docked onto the size-matched station, the
#   output also shows the train's translation path: every intermediate non-background train cell is
#   repainted to the legend color for the movement direction. The same decoded color mapping therefore
#   controls both orientation and trail coloring.
# - Small-example intuition: if the legend says red=up and blue=right, a train whose marker colors
#   demand a flip+rotation may still travel horizontally. In the output its final body keeps its own 8
#   and marker colors, while the vacated source and every intermediate slide position become a solid
#   blue trail because the top-left anchor moved rightward into the station.
# - Complexity calibration: legend directions are limited to 2-3 reusable colors across the whole
#   puzzle, and generator-side motion is restricted to a single axis per train. That keeps the new cue
#   visible and learnable from a few examples while still forcing the solver to separate "orientation"
#   from "movement color".
# - Hidden assumptions audited: the original solver assumed marker colors were stripe colors and that
#   trains disappeared from their source cells. Both fail here. The new solver reads legend colors from
#   the reserved corner region, never consults stripe colors for orientation, and repaints source/intermediate
#   train cells instead of erasing them to background.
# - Placement legality: the generator reserves the legend footprint, keeps stripes unchanged, places
#   stations with clearance, then chooses scattered source locations whose whole slide corridor avoids
#   stripes, legend cells, stations, docked trains, and previously reserved trails. This guarantees that
#   trail pixels remain unambiguous and that `main()` never needs to resolve collisions.
# - Visual observability and tidiness: legend colors are sampled disjoint from stripe colors, trail
#   colors come only from the legend palette, and every train is forced to move at least a few cells so
#   the trail is plainly visible. Trails are non-overlapping and axis-aligned, which keeps the grids
#   readable rather than noisy.
# - New solver assumptions and generator guarantees: `main()` now assumes each train's net translation
#   is non-zero, single-axis, and points along one of the legend directions. `_scatter_trains()` builds
#   exactly that geometry by accepting only source placements whose anchor displacement is horizontal or
#   vertical and whose direction exists in the legend map.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


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
LEGEND_ARM_OFFSETS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def _dir_from_delta(dr: int, dc: int) -> int:
    return (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)


def _group_markers(marker_pixels, color_to_dir):
    markers_by_color = defaultdict(list)
    for color, r, c in marker_pixels:
        if color in color_to_dir:
            markers_by_color[int(color)].append((int(r), int(c)))
    return markers_by_color


def _best_ops_for_markers(marker_pixels, body9, shape, color_to_dir):
    markers_by_color = _group_markers(marker_pixels, color_to_dir)
    assert markers_by_color, "Each train must expose legend-colored markers"
    assert body9, "Each train must carry at least one coupler"

    ref_r = sum(r for r, _ in body9) / len(body9)
    ref_c = sum(c for _, c in body9) / len(body9)
    best_score = -1
    best_ops = []
    for op in OPS:
        ref_t = list(transform_coords([(int(round(ref_r)), int(round(ref_c)))], shape, *op))
        rr_ref, cc_ref = ref_t[0]
        score = 0
        for color, pts in markers_by_color.items():
            mr = sum(r for r, _ in pts) / len(pts)
            mc = sum(c for _, c in pts) / len(pts)
            m_t = list(transform_coords([(int(round(mr)), int(round(mc)))], shape, *op))
            rr_m, cc_m = m_t[0]
            if _dir_from_delta(rr_m - rr_ref, cc_m - cc_ref) == color_to_dir[color]:
                score += 1
        if score > best_score:
            best_score = score
            best_ops = [op]
        elif score == best_score:
            best_ops.append(op)
    return best_score, best_ops, markers_by_color


def _choose_unique_op(marker_pixels, body9, shape, color_to_dir):
    best_score, best_ops, markers_by_color = _best_ops_for_markers(
        marker_pixels, body9, shape, color_to_dir
    )
    assert best_score == len(markers_by_color), "Legend markers must all agree on the chosen orientation"
    assert len(best_ops) == 1, "Legend markers must determine a unique orientation"
    return best_ops[0]


def _corner_region(regions):
    return min(regions, key=lambda reg: (reg["r0"], reg["c0"]))


def _legend_arm_cells(region):
    center_r = region["r0"] + 2
    center_c = region["c0"] + 2
    return {
        direction: (center_r + dr, center_c + dc)
        for direction, (dr, dc) in LEGEND_ARM_OFFSETS.items()
    }


def _read_corner_legend(grid: np.ndarray, bg: int, regions):
    region = _corner_region(regions)
    arm_cells = _legend_arm_cells(region)
    color_to_dir = {}
    for direction, (r, c) in arm_cells.items():
        color = int(grid[r, c])
        if color != bg:
            color_to_dir[color] = direction
    assert len(color_to_dir) >= 2, "Legend must expose at least two color-direction mappings"
    return color_to_dir


def _translation_direction(src_r0: int, src_c0: int, dst_r0: int, dst_c0: int) -> int:
    dr = int(dst_r0 - src_r0)
    dc = int(dst_c0 - src_c0)
    assert dr == 0 or dc == 0, "Trail variant expects single-axis motion"
    assert dr != 0 or dc != 0, "Trail variant expects non-zero motion"
    return _dir_from_delta(dr, dc)


def _paint_translation_trail(
    out: np.ndarray,
    bbox: np.ndarray,
    bg: int,
    src_r0: int,
    src_c0: int,
    dst_r0: int,
    dst_c0: int,
    trail_color: int,
):
    dr = int(dst_r0 - src_r0)
    dc = int(dst_c0 - src_c0)
    assert dr == 0 or dc == 0, "Trail variant expects single-axis motion"
    dist = abs(dr) + abs(dc)
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    cells = [(int(r), int(c)) for r, c in np.argwhere(bbox != bg)]
    for step in range(dist):
        rr0 = src_r0 + step * step_r
        cc0 = src_c0 + step * step_c
        for r, c in cells:
            out[rr0 + r, cc0 + c] = int(trail_color)


def _patch_marker_features(patch, legend_dirs):
    body9 = []
    marker_pixels = []
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            v = int(patch[r, c])
            if v == 9:
                body9.append((int(r), int(c)))
            if v in legend_dirs:
                marker_pixels.append((v, int(r), int(c)))
    return body9, marker_pixels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock trains onto station groups using a corner legend, then color each movement trail by the same legend.

    Input structure:
    - Bg plus colored border/internal stripes partition the canvas into regions, but stripe colors are only dividers.
    - Color 8 is the train body; color 9 appears both as isolated station groups and as couplers attached to trains.
    - A small plus-shaped legend in the corner region maps 2-3 colors to absolute directions.
    - Train marker pixels use legend colors, and those same legend colors are reused for the output trail.

    Rule:
    - Stations are 8-connected 9-clusters with no adjacent 8; trains are connected components containing 8.
    - Match each train to the unique station group whose size equals the train's coupler count.
    - Read the corner legend, rotate/flip the train so its marker colors lie on their legend-coded sides relative
      to the coupler centroid, and translate so the couplers land exactly on the station cells.
    - Remove the original train, then repaint every intermediate translated train position with the legend color of
      the movement direction; only the final docked train keeps its original non-9 colors.

    Key insight: the legend must be decoded before anything else, because the same color-direction map governs both
    orientation of marker pixels and the color assigned to the train's translation trail.
    """
    grid = np.array(grid, dtype=int, copy=True)
    H, W = grid.shape
    bg = int(get_bg(grid))

    def find_edge_stripes():
        stripe = np.zeros((H, W), dtype=bool)
        top_h = 0
        for r in range(H):
            row = grid[r, :]
            if (row == row[0]).all() and row[0] != bg:
                top_h += 1
            else:
                break
        if top_h:
            stripe[:top_h, :] = True

        bot_h = 0
        for r in range(H - 1, -1, -1):
            row = grid[r, :]
            if (row == row[0]).all() and row[0] != bg:
                bot_h += 1
            else:
                break
        if bot_h:
            stripe[H - bot_h :, :] = True

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
        return stripe, (top_h, bot_h, left_w, right_w)

    def find_internal_stripes(edge_info):
        top_h, bot_h, left_w, right_w = edge_info
        r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
        stripe = np.zeros((H, W), dtype=bool)

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
                r = r2 + 1
            else:
                r += 1
        return stripe

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

    def xform_coords(coords, shape, flip_ud, rot_k):
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape, dtype=int), flip_ud, rot_k).shape
        return list(result), new_shape

    edge_mask, edge_info = find_edge_stripes()
    internal_mask = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    regions = regions_from_stripes(stripe_mask)
    legend_dirs = _read_corner_legend(grid, bg, regions)
    dir_to_color = {direction: color for color, direction in legend_dirs.items()}

    groups = station_groups()
    groups_by_size = defaultdict(list)
    for g in groups:
        groups_by_size[len(g)].append(g)
    station_set = set(pt for g in groups for pt in g)

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
            if v in legend_dirs:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body9": body9,
                "marker_pixels": marker_pixels,
                "src_r0": int(r0),
                "src_c0": int(c0),
            }
        )

    out = grid.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
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

        op = _choose_unique_op(t["marker_pixels"], t["body9"], t["bbox"].shape, legend_dirs)
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

        move_dir = _translation_direction(t["src_r0"], t["src_c0"], off_r, off_c)
        assert move_dir in dir_to_color, "Movement direction must be present in the legend"
        _paint_translation_trail(
            out,
            t["bbox"],
            bg,
            t["src_r0"],
            t["src_c0"],
            off_r,
            off_c,
            dir_to_color[move_dir],
        )

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
                if stripe_mask[rr, cc]:
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

MARKER_DIST_RANGE = (2, 3)
MARKER_JITTER = [-1, 0, 1]
THREE_MARKER_PROB = 0.65
THREE_LEGEND_DIR_PROB = 0.7

BODY_GROWTH_ITERS = 160
PATH_BOUNDS_PAD = 4

LAYOUT_RETRIES = 200
DOCK_RETRIES = 20
GENERATE_RETRIES = 300

TRAIL_MIN_STEPS = 2
SOURCE_MARGIN = 0

COLOR_POOL = list(range(8))
ADJACENT_DIR_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]


def _apply_transform(arr, flip_ud=False, rot_k=0):
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def _inverse_patch(patch, op):
    for inv in OPS:
        cand = _apply_transform(patch, *inv)
        if np.array_equal(_apply_transform(cand, *op), patch):
            return cand
    raise ValueError("no inverse transform")


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


def _sample_legend_directions():
    if random.random() < THREE_LEGEND_DIR_PROB:
        missing = random.choice([0, 1, 2, 3])
        dirs = [d for d in range(4) if d != missing]
    else:
        dirs = list(random.choice(ADJACENT_DIR_PAIRS))
    random.shuffle(dirs)
    return dirs


def _sample_config():
    use_internal = random.random() < INTERNAL_PROB
    orient = random.choice(["v", "h"]) if use_internal else None
    n_trains = random.choice(TRAIN_COUNTS)
    return {
        "use_internal": use_internal,
        "orient": orient,
        "station_sizes": random.sample(STATION_SIZES, n_trains),
        "legend_dirs": _sample_legend_directions(),
    }


def _paint_corner_legend(grid, bg, region, legend_map):
    if region["r1"] - region["r0"] + 1 < 5 or region["c1"] - region["c0"] + 1 < 5:
        return False
    arm_cells = _legend_arm_cells(region)
    dir_to_color = {direction: color for color, direction in legend_map.items()}
    for direction, (r, c) in arm_cells.items():
        grid[r, c] = int(dir_to_color.get(direction, bg))
    return True


def _legend_reserved_mask(shape, regions):
    H, W = shape
    region = _corner_region(regions)
    mask = np.zeros((H, W), dtype=bool)
    r0 = max(0, region["r0"])
    c0 = max(0, region["c0"])
    r1 = min(H - 1, region["r0"] + 4)
    c1 = min(W - 1, region["c0"] + 4)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _layout(config):
    for _ in range(LAYOUT_RETRIES):
        use_internal = config["use_internal"]
        orient = config["orient"]
        if orient == "v":
            edges = ["left", "right", random.choice(["top", "bottom"])]
        elif orient == "h":
            edges = ["top", "bottom", random.choice(["left", "right"])]
        else:
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
        if len(pool) < len(config["legend_dirs"]):
            continue

        edge_col = dict(zip(edges, stripe_cols[: len(edges)]))
        internal_col = stripe_cols[-1] if use_internal else None
        legend_colors = random.sample(pool, len(config["legend_dirs"]))
        legend_map = dict(zip(legend_colors, config["legend_dirs"]))

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

        if not _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
            continue
        if not _paint_corner_legend(grid, bg, _corner_region(regions), legend_map):
            continue
        return grid, bg, regions, legend_map
    return None


def _station_shape(size):
    if size == 1:
        return {(0, 0)}
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    return {(0, 0), (0, 1), (1, 0), (1, 1)}


def _marker_specs(legend_map):
    items = list(legend_map.items())
    if len(items) >= 3 and random.random() < THREE_MARKER_PROB:
        return list(items)
    pairs = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if (a[1] - b[1]) % 4 in (1, 3):
                pairs.append((a, b))
    if not pairs:
        return list(items)
    return list(random.choice(pairs))


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
            exp = [q for q in cands if q[0] < r0 or q[0] > r1]
            cands = exp or cands
        elif w < PATCH_W_RANGE[0]:
            _, c0, _, c1 = _bbox(all_cells)
            exp = [q for q in cands if q[1] < c0 or q[1] > c1]
            cands = exp or cands
        body.add(random.choice(cands))


def _build_docked_patch(bg, legend_map, size):
    couplers = _station_shape(size)
    specs = _marker_specs(legend_map)
    markers, marker_cells = [], set()
    for color, direction in specs:
        p = _marker_cell(couplers, direction)
        if p in couplers or p in marker_cells:
            return None
        markers.append((p, color, direction))
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
    bounds = (sr0 - PATH_BOUNDS_PAD, sr1 + PATH_BOUNDS_PAD, sc0 - PATH_BOUNDS_PAD, sc1 + PATH_BOUNDS_PAD)

    for p, _, direction in markers:
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

    r0, c0, _, _ = _bbox(all_cells)
    patch = np.full((h, w), bg, dtype=int)
    marker_pixels = []
    rel_couplers = []
    for r, c in body:
        patch[r - r0, c - c0] = 8
    for (r, c), color, _ in markers:
        rr, cc = r - r0, c - c0
        patch[rr, cc] = color
        marker_pixels.append((int(color), rr, cc))
    for r, c in couplers:
        rr, cc = r - r0, c - c0
        patch[rr, cc] = 9
        rel_couplers.append((rr, cc))

    best_score, best_ops, markers_by_color = _best_ops_for_markers(
        marker_pixels, rel_couplers, patch.shape, legend_map
    )
    if best_score != len(markers_by_color) or best_ops != [(False, 0)]:
        return None
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


def _ops_preserving_shape(patch):
    valid = []
    for op in OPS:
        inv = _inverse_patch(patch, op)
        if inv.shape == patch.shape:
            valid.append(op)
    return valid


def _scatter_ops_for_patch(patch, legend_map):
    valid = []
    for op in _ops_preserving_shape(patch):
        inv = _inverse_patch(patch, op)
        body9_in, marker_pixels_in = _patch_marker_features(inv, legend_map)
        try:
            chosen = _choose_unique_op(marker_pixels_in, body9_in, inv.shape, legend_map)
        except AssertionError:
            continue
        if chosen == op:
            valid.append(op)
    return valid


def _trail_cells_for_patch(patch, src_rc, dst_rc, bg):
    src_r0, src_c0 = src_rc
    dst_r0, dst_c0 = dst_rc
    dr = int(dst_r0 - src_r0)
    dc = int(dst_c0 - src_c0)
    if not (dr == 0 or dc == 0) or (dr == 0 and dc == 0):
        return None
    dist = abs(dr) + abs(dc)
    if dist < TRAIL_MIN_STEPS:
        return None
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    coords = [(int(r), int(c)) for r, c in np.argwhere(patch != bg)]
    cells = set()
    for step in range(dist):
        rr0 = src_r0 + step * step_r
        cc0 = src_c0 + step * step_c
        for r, c in coords:
            cells.add((rr0 + r, cc0 + c))
    return cells


def _try_axis_direction(src_rc, dst_rc):
    dr = int(dst_rc[0] - src_rc[0])
    dc = int(dst_rc[1] - src_rc[1])
    if not (dr == 0 or dc == 0) or (dr == 0 and dc == 0):
        return None
    return _dir_from_delta(dr, dc)


def _place_docked_trains(base, bg, regions, legend_map, station_sizes, legend_reserved):
    output = base.copy()
    occupied = (output != bg) | legend_reserved

    trains = []
    for size in station_sizes:
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            result = _build_docked_patch(bg, legend_map, size)
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
            trains.append(
                {
                    "patch": patch,
                    "station": station_abs,
                    "dock_rc": (int(r0), int(c0)),
                    "region": reg,
                }
            )
            break
        else:
            return None
    return output, trains


def _scatter_trains(base, bg, regions, legend_map, docked_output, trains, legend_reserved):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True

    valid_ops = []
    for t in trains:
        valid = _scatter_ops_for_patch(t["patch"], legend_map)
        if not valid:
            return None
        valid_ops.append(valid)
    ops = [random.choice(valid) for valid in valid_ops]
    if all(op == (False, 0) for op in ops):
        for i, valid in enumerate(valid_ops):
            non_id = [op for op in valid if op != (False, 0)]
            if non_id:
                ops[i] = random.choice(non_id)
                break
        else:
            return None

    input_forbidden = (base != bg) | legend_reserved | _dilate(station_mask)
    trail_block = np.zeros((H, W), dtype=bool)
    static_trail_obstacles = (docked_output != bg) | legend_reserved
    dir_values = set(legend_map.values())

    order = list(range(len(trains)))
    random.shuffle(order)
    placed_sources = [None] * len(trains)
    actual_ops = [None] * len(trains)

    for idx in order:
        t = trains[idx]
        op = ops[idx]
        in_patch = _inverse_patch(t["patch"], op)
        if in_patch.shape != t["patch"].shape:
            return None
        actual_ops[idx] = op
        own_final_mask = np.zeros((H, W), dtype=bool)
        fr0, fc0 = t["dock_rc"]
        phf, pwf = t["patch"].shape
        own_final_mask[fr0 : fr0 + phf, fc0 : fc0 + pwf] = t["patch"] != bg
        blocked_static = static_trail_obstacles & (~own_final_mask)

        regs = regions[:]
        random.shuffle(regs)
        candidates = []
        for reg in regs:
            locs = _patch_locations(reg, in_patch, bg, input_forbidden, None, False, SOURCE_MARGIN)
            random.shuffle(locs)
            for src_rc in locs:
                move_dir = _try_axis_direction(src_rc, t["dock_rc"])
                if move_dir is None:
                    continue
                if move_dir not in dir_values:
                    continue
                trail_cells = _trail_cells_for_patch(in_patch, src_rc, t["dock_rc"], bg)
                if trail_cells is None:
                    continue
                if any(
                    rr < 0
                    or rr >= H
                    or cc < 0
                    or cc >= W
                    or blocked_static[rr, cc]
                    or trail_block[rr, cc]
                    for rr, cc in trail_cells
                ):
                    continue
                candidates.append((src_rc, trail_cells, in_patch))
        if not candidates:
            return None

        src_rc, trail_cells, chosen_patch = random.choice(candidates)
        placed_sources[idx] = {
            "src_rc": src_rc,
            "trail_cells": trail_cells,
            "patch": chosen_patch,
        }
        paste(input_grid, chosen_patch, src_rc[0], src_rc[1], transparent=bg)

        ph, pw = chosen_patch.shape
        patch_mask = np.zeros((H, W), dtype=bool)
        patch_mask[src_rc[0] : src_rc[0] + ph, src_rc[1] : src_rc[1] + pw] = chosen_patch != bg
        input_forbidden |= _dilate(patch_mask)
        for rr, cc in trail_cells:
            trail_block[rr, cc] = True

    return input_grid, placed_sources, actual_ops


def _gen_attempt(config):
    layout = _layout(config)
    if layout is None:
        return None
    base, bg, regions, legend_map = layout
    legend_reserved = _legend_reserved_mask(base.shape, regions)

    placed = _place_docked_trains(
        base, bg, regions, legend_map, config["station_sizes"], legend_reserved
    )
    if placed is None:
        return None
    docked_output, trains = placed

    scattered = _scatter_trains(base, bg, regions, legend_map, docked_output, trains, legend_reserved)
    if scattered is None:
        return None
    input_grid, _, _ = scattered

    counts = Counter(map(int, input_grid.flatten()))
    if counts[bg] != max(counts.values()):
        return None
    try:
        out = main(input_grid)
    except Exception:
        return None
    if np.array_equal(out, input_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    config = _sample_config()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(config)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6e4f6532 combo input")
