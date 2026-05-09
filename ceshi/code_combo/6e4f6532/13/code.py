# Combo variant of 6e4f6532 (index 13)
# combo_spec: D1/1+D13/2
# Name: 形状匹配+同大小不同朝向车站
#
# Design analysis
# a) Concrete behavior and trace
# - Original task:
#   station groups were distinguishable by 9-count alone, so a train first chose
#   its destination by `len(couplers)` and only then used stripe-colored markers
#   to recover the rigid transform for docking.
# - Fused variant:
#   every station has the same area: exactly two 9-cells. One station is a
#   horizontal 1x2 line and the other is a vertical 2x1 line. Every train also
#   carries exactly two couplers. Count matching is therefore useless.
# - Small trace:
#   suppose train A is scattered with a vertical 2-cell coupler pair. After the
#   marker-implied 90-degree rotation, those two 9s become horizontal. It must
#   dock onto the horizontal station, not merely "some size-2 station". Train B
#   resolves symmetrically onto the vertical station.
# - Key difference:
#   D1 contributes "match by shape rather than by count", while D13/2 removes
#   size uniqueness. Because the two candidate stations are rotated versions of
#   the same base shape, matching must use the transformed coupler shape in its
#   exact oriented form, not a rotation-invariant canonical signature.
#
# b) Complexity calibration
# - The shape family is intentionally tiny: only two same-sized station shapes
#   appear in each grid, horizontal and vertical 2-cell lines.
# - The generator always emits exactly two trains/stations, which is enough to
#   destroy size-based matching without creating redundant duplicates that would
#   make assignment ambiguous.
# - Region-marker logic is preserved so the puzzle still has a second control
#   channel: markers determine the rigid transform, and the transformed coupler
#   shape determines which equal-size station is correct.
#
# c) Hidden assumption audit
# - Original assumption `groups_by_size[len(g)]` is invalid because both station
#   groups have size 2.
# - A plain D1-style canonical shape signature is also invalid here: horizontal
#   and vertical 2-lines become the same class under rotation, which would erase
#   the only remaining distinction between stations.
# - Therefore `main()` must evaluate each train/station/op triple jointly and
#   require exact equality between the transformed coupler shape and the station
#   shape after translation normalization.
# - New solver assumptions:
#   1. there are exactly two stations and two trains,
#   2. both stations are size-2 lines with distinct orientations,
#   3. marker cues plus exact shape equality yield a unique assignment.
#   The generator enforces all three.
#
# d) Placement legality
# - Generation remains output-first: place the docked trains in legal target
#   regions, then inverse-transform and scatter them back into the input.
# - The builder keeps all trains inside regions, away from stripe cells, and
#   separated from each other and from station groups by dilated occupancy masks.
# - Solver-side placement is validated again before drawing, so malformed samples
#   are rejected during generation.
#
# e) Visual observability
# - From 2-4 examples a human can see that both stations always have the same
#   size, so counting cannot explain the mapping.
# - The only persistent visible difference is the line direction of the two 9
#   stations, which is exactly the fused rule's intended discriminant.
# - Marker colors remain visibly tied to stripe boundaries, so the examples still
#   explain why a train must be rotated/flipped before that direction test.
#
# f) Visual cleanliness
# - Restricting the puzzle to two stations prevents clutter from many same-sized
#   targets.
# - Stations keep a >=2-cell clearance from stripe borders, and scattered trains
#   keep a >=1-cell margin from forbidden structures, so the two 9-line targets
#   remain easy to read.
#
# g) Color inference
# - No step depends on numeric color ids. Divider-stripe colors only encode
#   boundary directions, exactly as in the source task.
# - Shape matching is purely geometric and only concerns the arrangement of the
#   two color-9 coupler/station cells.
#
# h) Generator guarantees for the fused rule
# - Both stations always have size 2.
# - The two station shapes are exactly one horizontal line and one vertical line.
# - At least one scattered train uses an orientation-changing transform, so the
#   marker-implied rigid motion is behaviorally active instead of degenerated
#   into an already-docked orientation.

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


def normalize_coords(coords) -> frozenset[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    assert pts, "expected non-empty coord set"
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return frozenset((r - r0, c - c0) for r, c in pts)


def exact_line_shape(coords) -> frozenset[tuple[int, int]]:
    shape = normalize_coords(coords)
    assert shape in (H_LINE, V_LINE), f"Expected a 2-cell line, got {sorted(shape)}"
    return shape


def orientation_changes_line(op) -> bool:
    return op[1] % 2 == 1


def xform_coords(coords, shape, flip_ud, rot_k):
    """Transform coords, return (list, new_shape)."""
    result = transform_coords(coords, shape, flip_ud, rot_k)
    new_shape = apply_transform(np.zeros(shape), flip_ud, rot_k).shape
    return list(result), new_shape


def _inverse_patch(patch, op):
    """Find the pre-image of *patch* under *op* (brute-force over 8 ops)."""
    for inv in OPS:
        cand = apply_transform(patch, *inv)
        if np.array_equal(apply_transform(cand, *op), patch):
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


def marker_score(train, divider_dirs, op):
    # Score one rigid transform by asking whether every marker color ends up
    # pointing toward the matching boundary stripe of the candidate region.
    bbox0 = train["bbox"]
    h0, w0 = bbox0.shape
    assert train["marker_pixels"], "Each train must expose divider-colored markers"
    assert train["body9"], "Each train must carry at least one coupler"
    ref_r = sum(r for r, _ in train["body9"]) / len(train["body9"])
    ref_c = sum(c for _, c in train["body9"]) / len(train["body9"])
    ref_t = list(transform_coords([(int(round(ref_r)), int(round(ref_c)))], (h0, w0), *op))
    rr_ref, cc_ref = ref_t[0]
    score = 0
    for color, mr, mc in train["marker_pixels"]:
        m_t = list(transform_coords([(int(mr), int(mc))], (h0, w0), *op))
        rr_m, cc_m = m_t[0]
        dr, dc = rr_m - rr_ref, cc_m - cc_ref
        dirc = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
        if divider_dirs.get(int(color)) == dirc:
            score += 1
    return score


def validate_placement(tbbox, reg, canvas, stripe_mask, bg, off_r, off_c):
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


def draw_train(tbbox, canvas, bg, off_r, off_c):
    h1, w1 = tbbox.shape
    for r in range(h1):
        for c in range(w1):
            v = int(tbbox[r, c])
            if v != bg and v != 9:
                canvas[r + off_r, c + off_c] = v


def find_edge_stripes(grid, bg):
    h, w = grid.shape
    stripe = np.zeros((h, w), dtype=bool)
    colors = set()
    top_h = 0
    for r in range(h):
        row = grid[r, :]
        if (row == row[0]).all() and row[0] != bg:
            top_h += 1
        else:
            break
    if top_h:
        stripe[:top_h, :] = True
        colors.add(int(grid[0, 0]))
    bot_h = 0
    for r in range(h - 1, -1, -1):
        row = grid[r, :]
        if (row == row[0]).all() and row[0] != bg:
            bot_h += 1
        else:
            break
    if bot_h:
        stripe[h - bot_h :, :] = True
        colors.add(int(grid[h - 1, 0]))
    inner_r0, inner_r1 = top_h, h - bot_h
    left_w = 0
    for c in range(w):
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
    for c in range(w - 1, -1, -1):
        col = grid[inner_r0:inner_r1, c]
        if col.size == 0:
            break
        if (col == col[0]).all() and col[0] != bg:
            right_w += 1
        else:
            break
    if right_w:
        stripe[inner_r0:inner_r1, w - right_w :] = True
        colors.add(int(grid[inner_r0, w - 1]))
    return stripe, colors, (top_h, bot_h, left_w, right_w)


def find_internal_stripes(grid, bg, edge_info):
    h, w = grid.shape
    top_h, bot_h, left_w, right_w = edge_info
    r0, r1, c0, c1 = top_h, h - bot_h, left_w, w - right_w
    stripe = np.zeros((h, w), dtype=bool)
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


def station_groups(grid):
    h, w = grid.shape
    sm = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 9:
                continue
            ok = all(
                grid[r + dr, c + dc] != 8
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr != 0 or dc != 0) and 0 <= r + dr < h and 0 <= c + dc < w
            )
            if ok:
                sm[r, c] = True
    lab, n = label(sm, structure=np.ones((3, 3), dtype=int))
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


def boundary_dirs_for_region(grid, reg, stripe_colors):
    h, w = grid.shape
    dirs = {}
    r0, r1, c0, c1 = reg["r0"], reg["r1"], reg["c0"], reg["c1"]
    if r0 - 1 >= 0:
        mc = Counter(grid[r0 - 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 0
    if r1 + 1 < h:
        mc = Counter(grid[r1 + 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 2
    if c0 - 1 >= 0:
        mc = Counter(grid[r0:r1 + 1, c0 - 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 3
    if c1 + 1 < w:
        mc = Counter(grid[r0:r1 + 1, c1 + 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 1
    return dirs


def extract_trains(grid, bg, stripe_mask, stripe_colors, station_set):
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
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body9": body9,
                "marker_pixels": marker_pixels,
            }
        )
    return trains


def parse_scene(grid):
    bg = get_bg(grid)
    edge_mask, edge_colors, edge_info = find_edge_stripes(grid, bg)
    internal_mask, internal_colors = find_internal_stripes(grid, bg, edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)
    groups = station_groups(grid)
    station_set = set(pt for g in groups for pt in g)
    trains = extract_trains(grid, bg, stripe_mask, stripe_colors, station_set)
    return {
        "bg": bg,
        "stripe_mask": stripe_mask,
        "stripe_colors": stripe_colors,
        "regions": regions,
        "groups": groups,
        "station_set": station_set,
        "trains": trains,
    }


def solve_from_parsed(grid, parsed):
    bg = parsed["bg"]
    stripe_mask = parsed["stripe_mask"]
    stripe_colors = parsed["stripe_colors"]
    regions = parsed["regions"]
    groups = parsed["groups"]
    station_set = parsed["station_set"]
    trains = parsed["trains"]

    stations = []
    for idx, g in enumerate(groups):
        assert len(g) == 2, f"Expected only 2-cell stations, got size {len(g)}"
        shape = exact_line_shape(g)
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
                "shape": shape,
                "region": reg,
                "divider_dirs": boundary_dirs_for_region(grid, reg, stripe_colors),
            }
        )

    assert len(stations) == 2, f"Expected exactly 2 stations, got {len(stations)}"
    assert {st["shape"] for st in stations} == {H_LINE, V_LINE}, (
        "Expected one horizontal and one vertical station"
    )

    out = grid.copy()
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
                if exact_line_shape(t_body9) != st["shape"]:
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
                if not validate_placement(tbbox, st["region"], out, stripe_mask, bg, off_r, off_c):
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
    by_id = {st["id"]: st for st in stations}
    for perm in permutations(station_ids, len(trains)):
        trial = out.copy()
        total = 0
        ok = True
        for ti, st_id in enumerate(perm):
            cand = pair_best.get(ti, {}).get(st_id)
            if cand is None:
                ok = False
                break
            st = by_id[st_id]
            if not validate_placement(
                cand["tbbox"], st["region"], trial, stripe_mask, bg, cand["off_r"], cand["off_c"]
            ):
                ok = False
                break
            draw_train(cand["tbbox"], trial, bg, cand["off_r"], cand["off_c"])
            total += cand["score"]
        if not ok:
            continue
        if total > best_score:
            best_score = total
            best_canvas = trial
            tied = False
        elif total == best_score:
            tied = True
    assert best_canvas is not None, "No valid equal-size shape-matching assignment found"
    assert not tied, "Expected a unique equal-size shape-matching assignment"
    return best_canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock train sprites onto equal-size station groups by exact coupler-shape matching, then orient them via divider-color markers.

    Input structure:
    - A background color fills the canvas, while colored edge/internal stripes divide it into rectangular regions.
    - Color 8 forms train bodies; color 9 marks both isolated station cells and train-attached couplers.
    - Every station has the same area: a 2-cell line of 9s. The only difference between stations is whether that line is horizontal or vertical.
    - Stripe-colored pixels on each train act as orientation markers tied to the target region's boundary colors.

    Rule:
    - Detect stations as 8-connected 9-clusters not touching any 8, and detect trains as connected components containing 8.
    - For every train/station candidate pair, try the 8 rigid transforms of the train.
    - A transform is valid only when all marker colors face the matching boundary stripes of that station's region and the transformed coupler shape matches that station's oriented 2-cell line exactly.
    - Choose the unique global assignment, erase the original trains, then redraw each transformed train so its couplers land on the chosen station cells.

    Key insight: Size no longer identifies the destination because all stations are equally large. The deciding clue is the exact post-transform coupler shape, while the stripe-colored markers still determine which rigid transform to apply.
    """
    return solve_from_parsed(grid.copy(), parse_scene(grid.copy()))


# --------------- generator (att2) ---------------

# ── Tunable constants ────────────────────────────────────────────────
GRID_H_RANGE = (13, 31)
GRID_W_RANGE = (16, 29)
GRID_MIN_AREA = 300
MIN_REGION_DIM = 10
MIN_REGION_DIRS = 3
INTERNAL_STRIPE_THICK = 2

STRIPE_THICK_RANGE = (2, 4)
INTERNAL_PROB = 0.62

TRAIN_COUNTS = [2]
STATION_ORIENTATIONS = ["h", "v"]

BODY_MIN = 5
BODY_MAX = 20
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)

MARKER_DIST_RANGE = (2, 3)
MARKER_JITTER = [-1, 0, 1]
THREE_MARKER_PROB = 1.0

BODY_GROWTH_ITERS = 160
PATH_BOUNDS_PAD = 4

LAYOUT_RETRIES = 200
DOCK_RETRIES = 20
SCATTER_OP_RETRIES = 80
GENERATE_RETRIES = 300

COLOR_POOL = list(range(8))


def _assign_region_dirs(regions, edge_col, top, bottom, left, right, h, w, internal):
    """Populate each region's 'dirs' mapping (stripe_color -> direction)."""
    for reg in regions:
        dirs = {}
        if top and reg["r0"] == top:
            dirs[edge_col["top"]] = 0
        if bottom and reg["r1"] == h - bottom - 1:
            dirs[edge_col["bottom"]] = 2
        if left and reg["c0"] == left:
            dirs[edge_col["left"]] = 3
        if right and reg["c1"] == w - right - 1:
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

        min_inner_for_split = MIN_REGION_DIM * 2 + INTERNAL_STRIPE_THICK
        candidates = []
        for h in range(*GRID_H_RANGE):
            for w in range(*GRID_W_RANGE):
                if h * w < GRID_MIN_AREA:
                    continue
                ih, iw = h - top - bottom, w - left - right
                if ih < MIN_REGION_DIM or iw < MIN_REGION_DIM:
                    continue
                if orient == "v" and iw < min_inner_for_split:
                    continue
                if orient == "h" and ih < min_inner_for_split:
                    continue
                candidates.append((h, w))
        if not candidates:
            continue

        h, w = random.choice(candidates)
        pool = COLOR_POOL[:]
        bg = random.choice(pool)
        pool.remove(bg)
        stripe_cols = random.sample(pool, len(edges) + int(use_internal))
        edge_col = dict(zip(edges, stripe_cols[: len(edges)]))
        internal_col = stripe_cols[-1] if use_internal else None
        grid = np.full((h, w), bg, dtype=int)

        if top:
            grid[:top, :] = edge_col["top"]
        if bottom:
            grid[h - bottom :, :] = edge_col["bottom"]
        if left:
            grid[top : h - bottom, :left] = edge_col["left"]
        if right:
            grid[top : h - bottom, w - right :] = edge_col["right"]

        inner = {"r0": top, "r1": h - bottom - 1, "c0": left, "c1": w - right - 1}
        regions, internal = [], None
        if orient == "v":
            lo = left + MIN_REGION_DIM
            hi = w - right - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            c = random.randint(lo, hi)
            grid[top : h - bottom, c : c + INTERNAL_STRIPE_THICK] = internal_col
            internal = ("v", c, internal_col)
            regions = [
                {"r0": top, "r1": h - bottom - 1, "c0": left, "c1": c - 1},
                {
                    "r0": top,
                    "r1": h - bottom - 1,
                    "c0": c + INTERNAL_STRIPE_THICK,
                    "c1": w - right - 1,
                },
            ]
        elif orient == "h":
            lo = top + MIN_REGION_DIM
            hi = h - bottom - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            r = random.randint(lo, hi)
            grid[r : r + INTERNAL_STRIPE_THICK, left : w - right] = internal_col
            internal = ("h", r, internal_col)
            regions = [
                {"r0": top, "r1": r - 1, "c0": left, "c1": w - right - 1},
                {
                    "r0": r + INTERNAL_STRIPE_THICK,
                    "r1": h - bottom - 1,
                    "c0": left,
                    "c1": w - right - 1,
                },
            ]
        else:
            regions = [inner]

        if _assign_region_dirs(regions, edge_col, top, bottom, left, right, h, w, internal):
            return grid, bg, regions
    return None


def _station_shape(orientation):
    if orientation == "h":
        return {(0, 0), (0, 1)}
    if orientation == "v":
        return {(0, 0), (1, 0)}
    raise ValueError(f"unknown orientation: {orientation}")


def _marker_specs(dirs):
    """Choose 2-3 (color, direction) marker specs from region boundary dirs."""
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


def _connect_couplers_to_body(couplers, body, blocked, bounds):
    """Ensure every coupler is 8-adjacent to the body via BFS paths."""
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
    contacts = [q for r, c in couplers for dr, dc in DIRS4 if (q := (r + dr, c + dc)) not in blocked]
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
    for r, c in body:
        patch[r - r0, c - c0] = 8
    for (r, c), color, _ in markers:
        patch[r - r0, c - c0] = color
    rel_couplers = []
    for r, c in couplers:
        patch[r - r0, c - c0] = 9
        rel_couplers.append((r - r0, c - c0))
    return patch, rel_couplers


def _patch_locations(region, patch, bg, occupied, coupler_rel=None, station_clearance=False, margin=0):
    """Find all valid (r, c) placements for *patch* within *region*."""
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


def _place_docked_trains(base, bg, regions):
    """Place both equal-size stations in one region so shape drives the pairing."""
    for target_region in random.sample(regions, len(regions)):
        output = base.copy()
        occupied = output != bg
        trains = []
        ok = True
        for orientation in random.sample(STATION_ORIENTATIONS, len(STATION_ORIENTATIONS)):
            placed = False
            for _ in range(DOCK_RETRIES):
                result = _build_docked_patch(bg, target_region["dirs"], orientation)
                if result is None:
                    continue
                patch, coupler_rel = result
                assert exact_line_shape(coupler_rel) == (H_LINE if orientation == "h" else V_LINE)
                locs = _patch_locations(target_region, patch, bg, _dilate(occupied), coupler_rel, True, 0)
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
                        "orientation": orientation,
                        "region": target_region,
                    }
                )
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return output, trains
    return None


def _choose_scatter_ops(trains):
    valid_ops = []
    for t in trains:
        valid = _valid_ops_for_patch(t["patch"])
        if not valid:
            return None
        valid_ops.append(valid)

    for _ in range(SCATTER_OP_RETRIES):
        ops = [random.choice(valid) for valid in valid_ops]
        if all(op == (False, 0) for op in ops):
            continue
        if not any(orientation_changes_line(op) for op in ops):
            continue
        return ops
    return None


def _scatter_trains(base, bg, regions, trains):
    """Apply random inverse transforms and scatter trains onto the input grid."""
    h, w = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((h, w), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True

    ops = _choose_scatter_ops(trains)
    if ops is None:
        return None

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
        patch_mask = np.zeros((h, w), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
        forbidden |= _dilate(patch_mask)

    return input_grid


def _fusion_metric_from_grid(grid):
    parsed = parse_scene(grid)
    station_sizes = sorted(len(g) for g in parsed["groups"])
    station_shapes = {exact_line_shape(g) for g in parsed["groups"]}
    trains = parsed["trains"]
    input_shapes = [exact_line_shape(t["body9"]) for t in trains]
    station_regions = []
    for g in parsed["groups"]:
        regs = [
            rr
            for rr in parsed["regions"]
            if all(rr["r0"] <= r <= rr["r1"] and rr["c0"] <= c <= rr["c1"] for r, c in g)
        ]
        assert len(regs) == 1, "Each station must belong to exactly one region"
        reg = regs[0]
        station_regions.append((reg["r0"], reg["r1"], reg["c0"], reg["c1"]))
    return {
        "same_size_station_count": sum(1 for size in station_sizes if size == 2),
        "distinct_station_orientations": len(station_shapes),
        "input_coupler_orientations": sum(1 for shape in set(input_shapes)),
        "station_region_count": len(set(station_regions)),
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

    metric = _fusion_metric_from_grid(input_grid)
    if (
        metric["same_size_station_count"] != 2
        or metric["distinct_station_orientations"] != 2
        or metric["station_region_count"] != 1
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
