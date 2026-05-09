# Combo variant of 6e4f6532 (index 19)
# combo_spec: D1/0+D5/0+D18/1
# Name: 颜色配对+重新上色+9像素全消
#
# Design analysis for the fused variant:
# 1. Concrete behavior trace:
#    - Base task: train -> station pairing is hidden in coupler-count, docking keeps station 9s,
#      and docked body cells stay color 8.
#    - Fused task: each train and each station expose a unique non-stripe label color; that visible
#      color pairing replaces the count key. After docking, every 8/9 cell in the train footprint
#      becomes the target region's dominant boundary-stripe color, while both train/station labels
#      and all 9s disappear. A small mental example is: two trains both have two couplers, one
#      train carries label 3 and one carries label 5, and two stations also have labels 3 and 5.
#      The train labeled 3 must dock at station label 3 even though coupler counts tie; then its
#      body and aligned couplers/station cells all repaint to that region's dominant stripe color.
# 2. Complexity calibration:
#    - Repeated station sizes are mandatory, otherwise color pairing would be cosmetic and the
#      original count rule would still solve the task.
#    - Label colors stay unique within a sample and come from a small non-stripe pool, so the
#      pairing is visible without exploding the color vocabulary.
#    - Regions keep only 2-3 trains and 2-3 stripe markers per train; this preserves enough
#      variation to show the fusion while keeping each board readable.
# 3. Assumption audit against the original main():
#    - `groups_by_size[...] == 1` is no longer valid, so the solver now builds label->station.
#    - Leaving station 9s intact is no longer valid; the solver clears every station 9 before
#      redrawing because 9 must vanish entirely.
#    - Redrawing all non-bg/non-9 pixels is no longer valid because input-only labels must also
#      vanish; only stripe markers persist, while 8/9 cells are mapped to the region dominant
#      stripe color.
#    - The new solver assumes each station has exactly one adjacent label pixel, each train has
#      exactly one attached label pixel, and each region has a unique dominant boundary stripe.
#      The generator enforces all three.
# 4. Placement legality:
#    - Output-first generation is retained. Canonical docked patches still include train labels and
#      coupler 9s so the input can be obtained by inverse transforms, but output materialization
#      strips labels and recolors 8/9.
#    - Station labels are reserved even though they are absent from output; otherwise a later train
#      could occupy an input-only label cell and make the paired cue unreadable.
#    - Placement checks still enforce in-bounds, non-overlap, stripe avoidance, and station
#      clearance from boundaries.
# 5. Visual observability and cleanliness:
#    - Human-solvable cues are visible: label-color pairing in the input, stripe-marker orientation,
#      and region-based recoloring in the output.
#    - Output boards are cleaner than the input: no label colors, no 8, no 9, only bg/stripes and
#      stripe-colored docked shapes remain.
# 6. Non-degeneracy guardrails:
#    - The generator forces at least one repeated station size by using one shared station size for
#      all trains in a sample.
#    - Dominant recolor color is defined by actual bordering stripe area, not by color id.
#    - Validation later measures repeated sizes, label removal, and zero surviving 8/9 cells so the
#      fused behavior cannot silently collapse back to the base puzzle.

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock color-labeled train sprites onto color-labeled stations, then collapse the whole docked mass into stripe colors.

    Input structure:
    - Thick colored edge/internal stripes partition the board into 1-2 rectangular regions.
    - Each train component contains color-8 body cells, color-9 couplers, 2-3 stripe-colored
      orientation markers, and one extra non-stripe label pixel.
    - Each station is an isolated 9-cluster with one 4-neighbor-adjacent non-stripe label pixel
      whose color matches exactly one train label.

    Rule:
    - Detect station groups as 9-clusters whose 8-neighborhood does not touch any 8-body cell.
    - Match each train to the station with the same label color, not by coupler count; repeated
      station sizes are allowed and are intentionally uninformative.
    - Within the matched station's region, choose the flip/rotation whose stripe-colored markers
      face the same-colored boundary stripes, then translate so train couplers land exactly on the
      station cells.
    - Erase the scattered train and all station labels/9s, then redraw the docked train so every
      transformed 8 or 9 cell becomes the region's dominant boundary-stripe color, while marker
      cells keep their stripe colors and label pixels disappear.

    Key insight: Label color chooses the destination, stripe-marker direction chooses the pose, and
    the final docked object is not a mix of source colors anymore: all 8/9 structure is absorbed
    into the target region's dominant stripe color, leaving no 8 and no 9 anywhere in the output.
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

    def dominant_boundary_color_for_region(reg, stripe_colors):
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

    def xform_coords(coords, shape, flip_ud, rot_k):
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape), flip_ud, rot_k).shape
        return list(result), new_shape

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
                    transform_coords(
                        [(int(round(ref_r)), int(round(ref_c)))],
                        (H0, W0),
                        flip_ud,
                        k,
                    )
                )
                rr_ref, cc_ref = ref_t[0]
                score = 0
                for color, pts in markers_by_color.items():
                    mr = sum(r for r, _ in pts) / len(pts)
                    mc = sum(c for _, c in pts) / len(pts)
                    m_t = list(
                        transform_coords(
                            [(int(round(mr)), int(round(mc)))],
                            (H0, W0),
                            flip_ud,
                            k,
                        )
                    )
                    rr_m, cc_m = m_t[0]
                    dr, dc = rr_m - rr_ref, cc_m - cc_ref
                    dirc = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
                    if dirc == divider_dirs[color]:
                        score += 1
                if score > best_score:
                    best_score, best = score, (flip_ud, k)
        return best

    def station_label_info(group, stripe_colors):
        gset = set(group)
        label_cells = []
        label_colors = set()
        for r, c in group:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (rr, cc) in gset:
                    continue
                v = int(grid[rr, cc])
                if v in (bg, 8, 9) or v in stripe_colors:
                    continue
                label_cells.append((rr, cc))
                label_colors.add(v)
        label_cells = sorted(set(label_cells))
        assert len(label_colors) == 1, "Each station must have exactly one label color"
        assert len(label_cells) == 1, "Each station must have exactly one label cell"
        return next(iter(label_colors)), label_cells[0]

    def rendered_value(v, stripe_colors, label_color, dock_color):
        if v == bg or v == label_color:
            return None
        if v in stripe_colors:
            return int(v)
        if v in (8, 9):
            return int(dock_color)
        return None

    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups()
    stations_by_label = {}
    station_set = set()
    station_label_cells = set()
    for g in groups:
        label_color, label_cell = station_label_info(g, stripe_colors)
        assert label_color not in stations_by_label, "Station label colors must be unique"
        matching_regions = [
            rr
            for rr in regions
            if all(rr["r0"] <= r <= rr["r1"] and rr["c0"] <= c <= rr["c1"] for r, c in g)
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        stations_by_label[label_color] = {
            "group": g,
            "region": matching_regions[0],
            "label_cell": label_cell,
        }
        station_set.update(g)
        station_label_cells.add(label_cell)

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
        label_colors = set()
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 9 and (r, c) not in station_set:
                body9.append((int(r - r0), int(c - c0)))
            elif v in stripe_colors:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
            elif v not in (bg, 8):
                label_colors.add(v)
        assert len(label_colors) == 1, "Each train must carry exactly one non-stripe label color"
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body9": body9,
                "marker_pixels": marker_pixels,
                "label_color": next(iter(label_colors)),
            }
        )

    out = grid.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg
    for r, c in station_set:
        out[r, c] = bg
    for r, c in station_label_cells:
        out[r, c] = bg

    for t in trains:
        assert t["label_color"] in stations_by_label, "Train label must match some station label"
        target = stations_by_label[t["label_color"]]
        g = target["group"]
        gset = set(g)
        reg = target["region"]
        divider_dirs = boundary_dirs_for_region(reg, stripe_colors)
        dock_color = dominant_boundary_color_for_region(reg, stripe_colors)
        op = choose_best_op(t, divider_dirs)
        tbbox = apply_transform(t["bbox"], *op)
        t_body9, _ = xform_coords(t["body9"], t["bbox"].shape, *op)
        body9_set = set(t_body9)
        matching_offsets = {
            (sr - br, sc - bc)
            for br, bc in body9_set
            for sr, sc in gset
            if all((r + sr - br, c + sc - bc) in gset for r, c in body9_set)
        }
        assert len(matching_offsets) == 1, "Couplers must align to a unique station-group offset"
        off_r, off_c = next(iter(matching_offsets))

        H1, W1 = tbbox.shape
        ok = True
        for r in range(H1):
            for c in range(W1):
                mapped = rendered_value(int(tbbox[r, c]), stripe_colors, t["label_color"], dock_color)
                if mapped is None:
                    continue
                rr, cc = r + off_r, c + off_c
                if not (reg["r0"] <= rr <= reg["r1"] and reg["c0"] <= cc <= reg["c1"]):
                    ok = False
                    break
                if stripe_mask[rr, cc] or out[rr, cc] != bg:
                    ok = False
                    break
            if not ok:
                break
        assert ok, "Docked train must fit entirely within its target region"

        for r in range(H1):
            for c in range(W1):
                mapped = rendered_value(int(tbbox[r, c]), stripe_colors, t["label_color"], dock_color)
                if mapped is not None:
                    out[r + off_r, c + off_c] = mapped

    assert not np.any(out == 8), "Output must contain no color-8 cells"
    assert not np.any(out == 9), "Output must contain no color-9 cells"
    return out


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

BODY_GROWTH_ITERS = 160
PATH_BOUNDS_PAD = 4

LAYOUT_RETRIES = 220
DOCK_RETRIES = 30
GENERATE_RETRIES = 500

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
        ordered = weights.most_common()
        reg["dirs"] = dirs
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
            stripe_colors = set(edge_col.values())
            if internal_col is not None:
                stripe_colors.add(internal_col)
            label_pool = [c for c in COLOR_POOL if c not in stripe_colors and c != bg]
            if len(label_pool) >= max(TRAIN_COUNTS):
                return grid, bg, regions, label_pool
    return None


def _station_shape(size):
    if size == 1:
        return {(0, 0)}
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    return {(0, 0), (0, 1), (1, 0), (1, 1)}


def _marker_specs(dirs):
    items = list(dirs.items())
    if len(items) >= 3 and random.random() < THREE_MARKER_PROB:
        return random.sample(items, 3)
    pairs = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if (a[1] - b[1]) % 4 in (1, 3):
                pairs.append((a, b))
    return list(random.choice(pairs)) if pairs else random.sample(items, 2)


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


def _build_docked_patch(bg, dirs, size, train_label_color):
    couplers = _station_shape(size)
    specs = _marker_specs(dirs)
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

    label_candidates = []
    all_core = body | set(couplers) | marker_cells
    for r, c in body:
        for dr, dc in DIRS4:
            q = (r + dr, c + dc)
            if q in all_core:
                continue
            h, w = _dims(all_core | {q})
            if PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1] and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]:
                label_candidates.append(q)
    if not label_candidates:
        return None
    label_cell = random.choice(label_candidates)

    all_cells = all_core | {label_cell}
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
    for (r, c), color, _ in markers:
        patch[r - r0, c - c0] = color
    patch[label_cell[0] - r0, label_cell[1] - c0] = train_label_color

    rel_couplers = []
    for r, c in couplers:
        patch[r - r0, c - c0] = 9
        rel_couplers.append((r - r0, c - c0))
    return patch, rel_couplers


def _station_label_candidates(region, station_abs, patch_abs_mask, reserved):
    gset = set(station_abs)
    cands = set()
    for r, c in station_abs:
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if not (region["r0"] <= rr <= region["r1"] and region["c0"] <= cc <= region["c1"]):
                continue
            if (rr, cc) in gset or patch_abs_mask[rr, cc] or reserved[rr, cc]:
                continue
            cands.add((rr, cc))
    return list(cands)


def _materialize_output_patch(patch, dock_color, marker_colors, bg):
    out = np.full_like(patch, bg)
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            v = int(patch[r, c])
            if v == bg:
                continue
            if v in marker_colors:
                out[r, c] = v
            elif v in (8, 9):
                out[r, c] = int(dock_color)
    return out


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


def _place_docked_trains(base, bg, regions, label_pool):
    H, W = base.shape
    output = base.copy()
    reserved = output != bg

    n_trains = random.choice(TRAIN_COUNTS)
    shared_size = random.choice(STATION_SIZES)
    label_colors = random.sample(label_pool, n_trains)
    region_loads = [0] * len(regions)

    trains = []
    for label_color in label_colors:
        placed = False
        for _ in range(DOCK_RETRIES):
            min_load = min(region_loads)
            region_order = [i for i, load in enumerate(region_loads) if load == min_load]
            spillover = [i for i, load in enumerate(region_loads) if load != min_load]
            random.shuffle(region_order)
            random.shuffle(spillover)
            for reg_idx in region_order + spillover:
                reg = regions[reg_idx]
                if reg.get("dominant_color") is None:
                    continue
                result = _build_docked_patch(bg, reg["dirs"], shared_size, label_color)
                if result is None:
                    continue
                patch, coupler_rel = result
                locs = _patch_locations(reg, patch, bg, _dilate(reserved), coupler_rel, True, 0)
                random.shuffle(locs)
                for r0, c0 in locs:
                    patch_abs_mask = np.zeros((H, W), dtype=bool)
                    ph, pw = patch.shape
                    patch_abs_mask[r0 : r0 + ph, c0 : c0 + pw] = patch != bg
                    station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
                    label_cands = _station_label_candidates(reg, station_abs, patch_abs_mask, reserved)
                    if not label_cands:
                        continue
                    station_label = random.choice(label_cands)
                    out_patch = _materialize_output_patch(patch, reg["dominant_color"], set(reg["dirs"]), bg)
                    paste(output, out_patch, r0, c0, transparent=bg)
                    reserved |= patch_abs_mask
                    reserved[station_label] = True
                    region_loads[reg_idx] += 1
                    trains.append(
                        {
                            "patch": patch,
                            "station": station_abs,
                            "station_label": station_label,
                            "label_color": label_color,
                            "dominant_color": reg["dominant_color"],
                        }
                    )
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None
    return output, trains


def _scatter_trains(base, bg, regions, trains):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    station_label_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True
        lr, lc = t["station_label"]
        input_grid[lr, lc] = t["label_color"]
        station_label_mask[lr, lc] = True

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

    forbidden = _dilate(base != bg) | _dilate(station_mask | station_label_mask)
    for t, op in zip(trains, ops):
        in_patch = _inverse_patch(t["patch"], op)
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


def _variant_metrics(input_grid, output_grid, trains):
    input_colors = set(map(int, input_grid.flatten()))
    output_colors = set(map(int, output_grid.flatten()))
    station_sizes = [len(t["station"]) for t in trains]
    label_colors = {t["label_color"] for t in trains}
    return {
        "train_count": len(trains),
        "shared_size_repeats": len(station_sizes) - len(set(station_sizes)),
        "input_has_labels": int(len(label_colors & input_colors) == len(label_colors)),
        "output_has_no_labels": int(not bool(label_colors & output_colors)),
        "output_has_no_8": int(8 not in output_colors),
        "output_has_no_9": int(9 not in output_colors),
    }


def _gen_attempt():
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions, label_pool = layout

    placed = _place_docked_trains(base, bg, regions, label_pool)
    if placed is None:
        return None
    output, trains = placed

    metrics = _variant_metrics(base, output, trains)
    if metrics["shared_size_repeats"] < 1:
        return None

    input_grid = _scatter_trains(base, bg, regions, trains)
    if input_grid is None:
        return None

    metrics = _variant_metrics(input_grid, output, trains)
    if not (
        metrics["input_has_labels"]
        and metrics["output_has_no_labels"]
        and metrics["output_has_no_8"]
        and metrics["output_has_no_9"]
    ):
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
