from __future__ import annotations
from collections import Counter, defaultdict, deque
import numpy as np
from scipy.ndimage import label
from arc_utils import STRUCT4, STRUCT8, get_bg

def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    """Positional-args helper (kept local; only used by this solver)."""
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    """Positional-args helper (kept local; only used by this solver)."""
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))




def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock train sprites onto station groups by matching coupler count, then orient via divider-color markers.

    Input structure:
    - bg; colored stripes (edge + internal) partition grid into rectangular regions.
    - Color 8 forms train bodies; color 9 marks both isolated stations and train-attached couplers.
    - Stripe-colored pixels on trains are orientation markers.

    Rule:
    - Stations = 8-connected 9-clusters not adjacent to any 8. Trains = components containing 8.
    - Match train to station group where |couplers| == |station size|.
    - Rotate/flip train so marker pixels face the same-color boundary stripe of the target region.
    - Translate so couplers align exactly onto station cells; erase original position.

    Key insight: Coupler count selects the docking target; marker color encodes which boundary stripe to face, fixing the unique orientation.
    """
    inp = grid
    grid = inp.copy()
    H, W = grid.shape
    bg = get_bg(grid)

    # This puzzle's code is long because it reconstructs several "structures"
    # that are implicit in the grid:
    # - divider stripes (edge + internal) that partition the canvas into regions
    # - station groups = 9-clusters not adjacent to any 8
    # - train sprites = components containing 8 (with attached 9 "couplers")
    # Then, for each train:
    # - match target region by COUPLER COUNT (|couplers| == |station group|)
    # - choose orientation by divider-color markers (face region boundary stripes)
    # - translate so couplers land exactly onto the station group cells

    def find_edge_stripes():
        # Detect thick constant-color border stripes.
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()
        # Top edge
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
        # Bottom edge
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
        # Left edge (in inner rows)
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
        # Right edge
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
        # Detect internal constant-color stripes in the inner rectangle.
        top_h, bot_h, left_w, right_w = edge_info
        r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
        stripe = np.zeros((H, W), dtype=bool)
        colors = set()
        # Vertical internal stripes
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
        # Horizontal internal stripes
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
        # Stations are 9 pixels with no adjacent 8 in the 8-neighborhood.
        # Station groups are 8-connected components of those station pixels.
        # Station 9s have no adjacent 8 in 8-neighborhood
        sm = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 9:
                    continue
                ok = all(grid[r+dr, c+dc] != 8
                        for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                        if (dr != 0 or dc != 0) and 0 <= r+dr < H and 0 <= c+dc < W)
                if ok:
                    sm[r, c] = True
        # 8-connected clustering
        struct = np.ones((3, 3), dtype=int)
        lab, n = label(sm, structure=struct)
        groups = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            groups.append([(int(r), int(c)) for r, c in coords])
        return groups

    def regions_from_stripes(stripe_mask):
        # Regions are connected components of all non-stripe cells.
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
        # For a region bbox, infer which stripe color borders each side.
        # Returns map: stripe_color -> direction (0=up,1=right,2=down,3=left)
        # Map stripe color -> direction (0=up, 1=right, 2=down, 3=left)
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
        """Transform coords, return (list, new_shape)."""
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape), flip_ud, rot_k).shape
        return list(result), new_shape

    def choose_best_op(train, divider_dirs):
        # Pick orientation (flip/rot) that makes divider-color marker pixels
        # "face" the region boundary stripe of the same color.
        bbox0 = train["bbox"]
        H0, W0 = bbox0.shape
        markers_by_color = defaultdict(list)
        for v, r, c in train["marker_pixels"]:
            if v in divider_dirs:
                markers_by_color[v].append((r, c))
        if not markers_by_color:
            return (False, 0)
        if train["body9"]:
            ref_r = sum(r for r, _ in train["body9"]) / len(train["body9"])
            ref_c = sum(c for _, c in train["body9"]) / len(train["body9"])
        else:
            coords = np.argwhere(bbox0 != bg)
            ref_r, ref_c = coords.mean(axis=0)
        best, best_score = (False, 0), -1
        for flip_ud in (False, True):
            for k in range(4):
                ref_t = list(transform_coords([(int(round(ref_r)), int(round(ref_c)))], (H0, W0), flip_ud, k))
                rr_ref, cc_ref = ref_t[0]
                score = 0
                for color, pts in markers_by_color.items():
                    mr, mc = sum(r for r, _ in pts) / len(pts), sum(c for _, c in pts) / len(pts)
                    m_t = list(transform_coords([(int(round(mr)), int(round(mc)))], (H0, W0), flip_ud, k))
                    rr_m, cc_m = m_t[0]
                    dr, dc = rr_m - rr_ref, cc_m - cc_ref
                    dirc = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
                    if dirc == divider_dirs[color]:
                        score += 1
                if score > best_score:
                    best_score, best = score, (flip_ud, k)
        return best

    # 1) Detect stripes and partition into regions
    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    # 2) Station groups (targets)
    groups = station_groups()
    groups_by_size = defaultdict(list)
    for g in groups:
        groups_by_size[len(g)].append(g)
    station_set = set(pt for g in groups for pt in g)

    # 3) Trains (sources): CCs (excluding stripes) that contain 8.
    #    Also record train-attached 9 couplers (9s NOT part of stations).
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
        bbox = grid[r0:r1 + 1, c0:c1 + 1].copy()
        abs_coords = [(int(r), int(c)) for r, c in coords]
        body9, marker_pixels = [], []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 9 and (r, c) not in station_set:
                body9.append((int(r - r0), int(c - c0)))
            if v in stripe_colors:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
        trains.append({
            "abs_coords": abs_coords, "bbox_origin": (int(r0), int(c0)),
            "bbox": bbox, "body9": body9, "marker_pixels": marker_pixels,
        })

    out = grid.copy()
    # 4) Clear trains from original positions (we'll redraw them docked).
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    # 5) Place each train by:
    # - matching station group size by coupler count
    # - choosing orientation by marker-vs-stripe rule
    # - translating so couplers coincide exactly with station pixels
    for t in trains:
        n9 = len(t["body9"])
        if n9 == 0:
            continue
        candidate_groups = groups_by_size.get(n9, [])
        if not candidate_groups:
            continue
        for g in candidate_groups:
            gset = set(g)
            # Find region containing this station group
            reg = next((rr for rr in regions 
                       if all(rr["r0"] <= r <= rr["r1"] and rr["c0"] <= c <= rr["c1"] for r, c in g)), None)
            if reg is None:
                continue
            divider_dirs = boundary_dirs_for_region(reg, stripe_colors)
            op = choose_best_op(t, divider_dirs)
            tbbox = apply_transform(t["bbox"], *op)
            t_body9, (H1, W1) = xform_coords(t["body9"], t["bbox"].shape, *op)
            body9_set = set(t_body9)
            # Find offset by exact coupler-station alignment
            best_offset = None
            for br, bc in body9_set:
                for sr, sc in gset:
                    off = (sr - br, sc - bc)
                    if all((r + off[0], c + off[1]) in gset for r, c in body9_set):
                        best_offset = off
                        break
                if best_offset:
                    break
            if best_offset is None:
                continue
            off_r, off_c = best_offset
            # Validate and place (skip 9s - stations already exist)
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
            if not ok:
                continue
            for r in range(H1):
                for c in range(W1):
                    v = int(tbbox[r, c])
                    if v != bg and v != 9:
                        out[r + off_r, c + off_c] = v
            break
    return out


