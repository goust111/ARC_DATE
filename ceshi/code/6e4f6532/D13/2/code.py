# Variant of 6e4f6532
# Dimension: D13 — 车站大小集合 (meta)
# Change: 车站大小从{1,2,4}中选取互不相同的值，对应1x1点、1x2线段、2x2方块... → 车站形状固定但旋转不同
#
# Design analysis for the D13 variant
# ==================================
# Reference-task note:
# - This variant lists no source_task, so the implementation baseline is the
#   original 6e4f6532 solver/generator only.
#
# a) Concrete behavior and trace
# - Original behavior:
#   train A might carry one coupler, train B two, train C four; after marker-led
#   rotation each train docks to the unique station whose 9-cluster size matches
#   its coupler count.
# - Variant behavior:
#   every station is the same two-pixel line, but one station is horizontal and
#   one is vertical. Each train also carries exactly two couplers. The decisive
#   feature is now the transformed coupler orientation, not the count.
# - Example trace:
#   if a scattered train has two couplers that become a horizontal pair only
#   after a 90-degree rotation, the old size-based rule cannot choose between the
#   two size-2 stations. The new rule first uses marker colors to find the
#   docking orientation, then checks whether the rotated coupler pair is
#   horizontal or vertical, and docks onto the station with that same line
#   orientation.
#
# b) Complexity calibration
# - The generator always emits exactly two trains/stations: one horizontal target
#   and one vertical target. This keeps the new control variable visible in every
#   sample and avoids reintroducing count-based uniqueness.
# - Station size stays fixed at 2, so the old count cue is intentionally
#   destroyed. Marker colors still matter because orientation alone identifies
#   the station shape, while stripe colors identify the required rotation/flip.
#
# c) Hidden assumptions from the original solver that break here
# - `groups_by_size[len(g)]` used to be a unique key. In this variant it is
#   useless because both stations have size 2.
# - `choose_best_op(train, divider_dirs)` used to run only after the size match
#   had already chosen the target region. Now region choice and orientation match
#   are coupled, so the solver must evaluate train/station pairs jointly.
# - The original "unique offset" check remains valid, but only after filtering
#   by the transformed coupler shape.
#
# d) Placement legality
# - The generator builds docked patches with a two-cell line of 9s and then
#   scatters inverse-transformed trains exactly as before.
# - It guarantees one horizontal and one vertical station, preserves stripe
#   clearances, keeps stations non-overlapping, and verifies `main(input) ==
#   output` before accepting a sample.
#
# e) Visual observability
# - A human can see that both target stations have the same area, so "size match"
#   is no longer available. The only persistent visible difference is horizontal
#   vs vertical arrangement.
# - Marker colors still visibly explain the rotation, while the final docking
#   target is explained by the coupler line orientation.
#
# f) Visual cleanliness
# - Restricting the puzzle to two trains prevents a clutter of same-sized station
#   candidates. The two target stations remain easy to distinguish by shape
#   direction alone.
#
# g) Color inference
# - No rule depends on numeric color ids. Stripe colors only encode boundary
#   directions exactly as in the original task.
#
# Post-change assumption audit
# - `main()` now assumes there are exactly two station groups, both size-2 lines,
#   and that their orientations are one horizontal and one vertical. The
#   generator enforces that explicitly.
# - `main()` also assumes the correct train/station pairing is the unique
#   assignment where each chosen transform satisfies every marker-direction cue
#   and the transformed coupler line matches the station line orientation.

from __future__ import annotations

from collections import Counter, deque
from itertools import permutations
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste

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


H_LINE = frozenset({(0, 0), (0, 1)})
V_LINE = frozenset({(0, 0), (1, 0)})


def normalize_coords(coords) -> frozenset[tuple[int, int]]:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    r0, c0 = min(rs), min(cs)
    return frozenset((int(r) - r0, int(c) - c0) for r, c in coords)


def line_orientation(coords) -> str:
    shape = normalize_coords(coords)
    if shape == H_LINE:
        return "h"
    if shape == V_LINE:
        return "v"
    raise AssertionError(f"Expected a 2-cell horizontal/vertical line, got {sorted(shape)}")




def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock train sprites onto equal-size stations by matching coupler orientation, then orient via divider-color markers.

    Input structure:
    - A background color fills the canvas, while edge/internal divider stripes split it into rectangular regions.
    - Color 8 forms the train bodies; color 9 marks both isolated station cells and train-attached couplers.
    - Every station is a 2-cell line of 9s, and the only visible difference between stations is whether that line is horizontal or vertical.
    - Stripe-colored pixels on the trains act as orientation markers tied to the target region's boundary colors.

    Rule:
    - Detect station groups as 9-clusters not touching any 8, and detect trains as components containing 8.
    - For each possible train/station pairing, try the 8 rigid transforms of the train.
    - A transform is valid only when all marker colors face the matching boundary stripes of that station's region and the transformed coupler pair has the same horizontal/vertical orientation as the station pair.
    - Choose the unique global assignment, erase the trains from their original positions, then redraw each transformed train so its couplers land exactly on the chosen station cells.

    Key insight: Station size no longer identifies the target because every station has the same two-cell area. The decisive cue is the coupler line's orientation after the marker-implied transform.
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
    # - evaluate each station candidate jointly with each rigid transform
    # - keep only transforms whose markers face the target region's stripe colors
    # - require the transformed coupler pair to have the same line orientation
    #   as the station before translating onto that station

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

    def marker_score(train, divider_dirs, op):
        # Score one rigid transform by asking whether every marker color ends up
        # pointing toward the matching boundary stripe of the candidate region.
        bbox0 = train["bbox"]
        H0, W0 = bbox0.shape
        assert train["marker_pixels"], "Each train must expose divider-colored markers"
        assert train["body9"], "Each train must carry at least one coupler"
        ref_r = sum(r for r, _ in train["body9"]) / len(train["body9"])
        ref_c = sum(c for _, c in train["body9"]) / len(train["body9"])
        ref_t = list(
            transform_coords([(int(round(ref_r)), int(round(ref_c)))], (H0, W0), *op)
        )
        rr_ref, cc_ref = ref_t[0]
        score = 0
        for color, mr, mc in train["marker_pixels"]:
            m_t = list(transform_coords([(int(mr), int(mc))], (H0, W0), *op))
            rr_m, cc_m = m_t[0]
            dr, dc = rr_m - rr_ref, cc_m - cc_ref
            dirc = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
            if divider_dirs.get(int(color)) == dirc:
                score += 1
        return score

    def validate_placement(tbbox, reg, canvas, off_r, off_c):
        h1, w1 = tbbox.shape
        for r in range(h1):
            for c in range(w1):
                v = int(tbbox[r, c])
                if v == bg or v == 9:
                    continue
                rr, cc = r + off_r, c + off_c
                if not (reg["r0"] <= rr <= reg["r1"] and reg["c0"] <= cc <= reg["c1"]):
                    return False
                if stripe_mask[rr, cc] or (canvas[rr, cc] != bg and canvas[rr, cc] != 9):
                    return False
        return True

    def draw_train(tbbox, canvas, off_r, off_c):
        h1, w1 = tbbox.shape
        for r in range(h1):
            for c in range(w1):
                v = int(tbbox[r, c])
                if v != bg and v != 9:
                    canvas[r + off_r, c + off_c] = v

    # 1) Detect stripes and partition into regions
    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    # 2) Station groups (targets)
    groups = station_groups()
    station_set = set(pt for g in groups for pt in g)
    stations = []
    for idx, g in enumerate(groups):
        assert len(g) == 2, f"Expected only 2-cell stations, got size {len(g)}"
        matching_regions = [
            rr
            for rr in regions
            if all(rr["r0"] <= r <= rr["r1"] and rr["c0"] <= c <= rr["c1"] for r, c in g)
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        reg = matching_regions[0]
        stations.append(
            {
                "id": idx,
                "coords": g,
                "coord_set": set(g),
                "shape": normalize_coords(g),
                "orientation": line_orientation(g),
                "region": reg,
                "divider_dirs": boundary_dirs_for_region(reg, stripe_colors),
            }
        )
    assert len(stations) == 2, f"Expected exactly 2 stations, got {len(stations)}"
    assert Counter(st["orientation"] for st in stations) == Counter({"h": 1, "v": 1}), (
        "Expected exactly one horizontal and one vertical station"
    )

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
            "abs_coords": abs_coords,
            "bbox": bbox,
            "body9": body9,
            "marker_pixels": marker_pixels,
        })

    out = grid.copy()
    # 4) Clear trains from original positions (we'll redraw them docked).
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    assert len(trains) == len(stations) == 2, "Expected two trains and two stations"
    pair_best = {}
    for ti, t in enumerate(trains):
        assert len(t["body9"]) == 2, f"Each train must carry exactly two couplers, got {len(t['body9'])}"
        pair_best[ti] = {}
        full_score = len(t["marker_pixels"])
        for st in stations:
            best = None
            for op in OPS:
                tbbox = apply_transform(t["bbox"], *op)
                t_body9, _ = xform_coords(t["body9"], t["bbox"].shape, *op)
                if normalize_coords(t_body9) != st["shape"]:
                    continue
                body9_set = set(t_body9)
                matching_offsets = {
                    (sr - br, sc - bc)
                    for br, bc in body9_set
                    for sr, sc in st["coord_set"]
                    if all((r + sr - br, c + sc - bc) in st["coord_set"] for r, c in body9_set)
                }
                if len(matching_offsets) != 1:
                    continue
                score = marker_score(t, st["divider_dirs"], op)
                if score != full_score:
                    continue
                off_r, off_c = next(iter(matching_offsets))
                if not validate_placement(tbbox, st["region"], out, off_r, off_c):
                    continue
                cand = {
                    "station_id": st["id"],
                    "score": score,
                    "tbbox": tbbox,
                    "off_r": off_r,
                    "off_c": off_c,
                }
                if best is None:
                    best = cand
            if best is not None:
                pair_best[ti][st["id"]] = best

    best_canvas = None
    best_score = -1
    tied = False
    station_ids = [st["id"] for st in stations]
    for perm in permutations(station_ids, len(trains)):
        trial = out.copy()
        total = 0
        ok = True
        for ti, st_id in enumerate(perm):
            cand = pair_best.get(ti, {}).get(st_id)
            if cand is None:
                ok = False
                break
            st = next(s for s in stations if s["id"] == st_id)
            if not validate_placement(cand["tbbox"], st["region"], trial, cand["off_r"], cand["off_c"]):
                ok = False
                break
            draw_train(cand["tbbox"], trial, cand["off_r"], cand["off_c"])
            total += cand["score"]
        if not ok:
            continue
        if total > best_score:
            best_score = total
            best_canvas = trial
            tied = False
        elif total == best_score:
            tied = True
    assert best_canvas is not None, "No valid orientation-based train/station assignment found"
    assert not tied, "Expected a unique orientation-based assignment"
    return best_canvas

# --------------- generator (att2) ---------------

# ── Tunable constants ────────────────────────────────────────────────
# Canvas / grid sizing
GRID_H_RANGE = (13, 31)          # min/max+1 grid height
GRID_W_RANGE = (16, 29)          # min/max+1 grid width
GRID_MIN_AREA = 300              # minimum total cells
MIN_REGION_DIM = 10              # each region >=10 in both dims
MIN_REGION_DIRS = 3              # minimum boundary-stripe count per region
INTERNAL_STRIPE_THICK = 2        # thickness of the internal divider

# Edge stripes
STRIPE_THICK_RANGE = (2, 4)      # each edge stripe thickness

# Internal-stripe probability
INTERNAL_PROB = 0.62

# Train / station counts
TRAIN_COUNTS = [2]               # one horizontal station + one vertical station
STATION_ORIENTATIONS = ["h", "v"]

# Docked-patch body constraints
BODY_MIN = 5                     # minimum 8-cells in train body
BODY_MAX = 20                    # maximum 8-cells in train body
PATCH_H_RANGE = (4, 9)           # docked patch bbox height limits
PATCH_W_RANGE = (3, 8)           # docked patch bbox width limits

# Marker placement
MARKER_DIST_RANGE = (2, 3)       # distance from coupler bbox to marker cell
MARKER_JITTER = [-1, 0, 1]       # lateral jitter for marker position
THREE_MARKER_PROB = 1.0          # use 3 markers whenever the region exposes 3+ directions

# Growth / pathfinding
BODY_GROWTH_ITERS = 160          # max iterations for body cell growth
PATH_BOUNDS_PAD = 4              # padding around seed bbox for BFS bounds

# Retry budgets
LAYOUT_RETRIES = 200             # attempts to find a valid layout
DOCK_RETRIES = 20                # attempts per train to build+place a docked patch
GENERATE_RETRIES = 300           # outer generate_input loop budget

# Color pool (excludes 8=train body, 9=coupler/station)
COLOR_POOL = list(range(8))

# ── Constants ────────────────────────────────────────────────────────
OPS = [(False, 0), (False, 1), (False, 2), (False, 3),
       (True, 0), (True, 1), (True, 2), (True, 3)]
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# ── Primitives ───────────────────────────────────────────────────────

def _apply_transform(arr, flip_ud=False, rot_k=0):
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def _inverse_patch(patch, op):
    """Find the pre-image of *patch* under *op* (brute-force over 8 ops)."""
    for inv in OPS:
        cand = _apply_transform(patch, *inv)
        if np.array_equal(_apply_transform(cand, *op), patch):
            return cand
    raise ValueError("no inverse transform")


def _valid_ops_for_patch(patch):
    """Return ops whose inverse keeps patch within bbox size limits."""
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
    """8-connected dilation via numpy shifts."""
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
    """BFS from *body* to *goal*, adding the path into *body* (mutates)."""
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


# ── Layout construction ──────────────────────────────────────────────

def _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
    """Populate each region's 'dirs' mapping (stripe_color -> direction)."""
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
    """Build the background grid with edge/internal stripes and rectangular regions."""
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

        # Min inner dimension needed for internal stripes (2 regions of >=10 + 2-thick divider)
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
        edge_col = dict(zip(edges, stripe_cols[:len(edges)]))
        internal_col = stripe_cols[-1] if use_internal else None
        grid = np.full((H, W), bg, dtype=int)

        if top:    grid[:top, :] = edge_col["top"]
        if bottom: grid[H - bottom:, :] = edge_col["bottom"]
        if left:   grid[top:H - bottom, :left] = edge_col["left"]
        if right:  grid[top:H - bottom, W - right:] = edge_col["right"]

        inner = {"r0": top, "r1": H - bottom - 1, "c0": left, "c1": W - right - 1}
        regions, internal = [], None
        if orient == "v":
            lo, hi = left + MIN_REGION_DIM, W - right - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            c = random.randint(lo, hi)
            grid[top:H - bottom, c:c + INTERNAL_STRIPE_THICK] = internal_col
            internal = ("v", c, internal_col)
            regions = [
                {"r0": top, "r1": H - bottom - 1, "c0": left, "c1": c - 1},
                {"r0": top, "r1": H - bottom - 1, "c0": c + INTERNAL_STRIPE_THICK, "c1": W - right - 1},
            ]
        elif orient == "h":
            lo, hi = top + MIN_REGION_DIM, H - bottom - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            r = random.randint(lo, hi)
            grid[r:r + INTERNAL_STRIPE_THICK, left:W - right] = internal_col
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


# ── Station / coupler shapes ─────────────────────────────────────────

def _station_shape(orientation):
    if orientation == "h":
        return {(0, 0), (0, 1)}
    if orientation == "v":
        return {(0, 0), (1, 0)}
    raise ValueError(f"unknown orientation: {orientation}")


# ── Marker selection and placement ───────────────────────────────────

def _marker_specs(dirs):
    """Choose 2-3 (color, direction) marker specs from region boundary dirs."""
    items = list(dirs.items())
    if len(items) >= 3 and random.random() < THREE_MARKER_PROB:
        return random.sample(items, 3)
    pairs = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if (a[1] - b[1]) % 4 in (1, 3):
                pairs.append((a, b))
    return list(random.choice(pairs)) if pairs else random.sample(items, 2)


def _marker_cell(couplers, direction):
    """Pick a marker cell position at a random offset from the coupler bbox in *direction*."""
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
    """Return the cell one step inward from *marker* (toward coupler center)."""
    dr, dc = DIRS4[(direction + 2) % 4]
    return marker[0] + dr, marker[1] + dc


# ── Docked-patch construction ────────────────────────────────────────

def _connect_couplers_to_body(couplers, body, blocked, bounds):
    """Ensure every coupler is 8-adjacent to the body via BFS paths."""
    for cp in couplers:
        if _adjacent8(cp, body):
            continue
        cr, cc = cp
        choices = [(cr + dr, cc + dc)
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                   if (dr or dc) and (cr + dr, cc + dc) not in blocked]
        random.shuffle(choices)
        for q in choices:
            if _path_to(body, q, blocked, bounds):
                break
        else:
            return False
    return True


def _grow_body(body, blocked, target):
    """Grow *body* toward *target* cells while respecting bbox limits."""
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
        # Prefer expansion toward undersized dimension
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


def _build_docked_patch(bg, dirs, orientation):
    """Build a docked train patch: 8-body + 9-couplers + colored markers."""
    couplers = _station_shape(orientation)
    specs = _marker_specs(dirs)
    markers, marker_cells = [], set()
    for color, direction in specs:
        p = _marker_cell(couplers, direction)
        if p in couplers or p in marker_cells:
            return None
        markers.append((p, color, direction))
        marker_cells.add(p)

    blocked = set(couplers) | marker_cells
    contacts = [q for r, c in couplers for dr, dc in DIRS4
                if (q := (r + dr, c + dc)) not in blocked]
    if not contacts:
        return None

    body = {random.choice(contacts)}
    all_seed = set(couplers) | marker_cells | body
    sr0, sc0, sr1, sc1 = _bbox(all_seed)
    bounds = (sr0 - PATH_BOUNDS_PAD, sr1 + PATH_BOUNDS_PAD,
              sc0 - PATH_BOUNDS_PAD, sc1 + PATH_BOUNDS_PAD)

    # Connect markers inward to body
    for p, _, direction in markers:
        goal = _inward_cell(p, direction)
        if goal in blocked:
            return None
        if not _path_to(body, goal, blocked, bounds):
            return None

    # Ensure all couplers are 8-adjacent to the body
    if not _connect_couplers_to_body(couplers, body, blocked, bounds):
        return None

    if len(body) > BODY_MAX:
        return None
    target = random.randint(max(BODY_MIN, len(body)), BODY_MAX)
    _grow_body(body, blocked, target)

    all_cells = body | set(couplers) | marker_cells
    h, w = _dims(all_cells)
    if not (BODY_MIN <= len(body) <= BODY_MAX
            and PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1]
            and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]):
        return None

    # Render patch array
    r0, c0, _, _ = _bbox(all_cells)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = 8
    for (r, c), color, _ in markers:
        patch[r - r0, c - c0] = color
    rel_couplers = []
    for r, c in couplers:
        patch[r - r0, c - c0] = 9
        rel_couplers.append((r - r0, c - c0))
    return patch, rel_couplers


# ── Placement search ─────────────────────────────────────────────────

def _patch_locations(region, patch, bg, occupied, coupler_rel=None, station_clearance=False, margin=0):
    """Find all valid (r, c) placements for *patch* within *region*."""
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
                    if not (region["r0"] + 2 <= ar <= region["r1"] - 2 and
                            region["c0"] + 2 <= ac <= region["c1"] - 2):
                        ok = False
                        break
            if ok:
                locs.append((rr, cc))
    return locs


# ── Top-level generation attempt ─────────────────────────────────────

def _place_docked_trains(base, bg, regions):
    """Place N docked trains on the output grid; return (output, trains) or None."""
    H, W = base.shape
    output = base.copy()
    occupied = output != bg

    n_trains = random.choice(TRAIN_COUNTS)
    orientations = random.sample(STATION_ORIENTATIONS, n_trains)
    trains = []
    for orientation in orientations:
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            result = _build_docked_patch(bg, reg["dirs"], orientation)
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
            trains.append({"patch": patch, "station": station_abs, "orientation": orientation})
            break
        else:
            return None
    return output, trains


def _scatter_trains(base, bg, regions, trains):
    """Apply random transforms and scatter trains onto the input grid."""
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True

    # Choose a non-trivial set of transforms
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

    # Place inverse-transformed patches
    forbidden = _dilate(base != bg) | _dilate(station_mask)
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
        patch_mask[r0:r0 + ph, c0:c0 + pw] = (in_patch != bg)
        forbidden |= _dilate(patch_mask)

    return input_grid


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
    raise RuntimeError("failed to generate valid 6e4f6532 input")
