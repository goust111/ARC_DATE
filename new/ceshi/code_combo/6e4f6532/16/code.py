# Combo variant of 6e4f6532 (index 16)
# combo_spec: D13/2+D17/0
# Name: 同大小车站+4连通聚类
#
# Design analysis for the combo variant
# ====================================
# Reference-task note:
# - This fusion starts from the original 6e4f6532 solver/generator and folds in
#   the single-dimension D13/2 idea (equal-size stations distinguished by
#   orientation) plus the D17/0 connectivity inversion (stations cluster by
#   4-connectivity, not 8-connectivity).
#
# a) Concrete behavior and trace
# - Original behavior:
#   trains are paired to stations by coupler count, because station sizes are
#   unique among {1, 2, 4}. After that, stripe-colored markers decide the rigid
#   transform needed before docking.
# - Fused behavior:
#   both stations are size-2 lines, one horizontal and one vertical, so size no
#   longer identifies the target. In addition, the two station lines are placed
#   as a diagonal-only contact pair such as:
#       9 9 .
#       . . 9
#       . . 9
#   Under 8-connectivity this is one size-4 cluster, but under 4-connectivity it
#   is two separate size-2 stations. The solver must therefore split stations by
#   4-connectivity first, then use marker-implied orientation plus transformed
#   coupler orientation to decide which train docks onto which line.
# - Behavioral difference:
#   an 8-connected interpretation would collapse the two stations into one 4-cell
#   blob and destroy the equal-size station matching logic. The new task only
#   works when both changes are respected together.
#
# b) Complexity calibration
# - The generator always emits exactly two trains and exactly two stations.
# - Both stations have the same visible area (2 cells), but opposite orientations
#   (horizontal vs vertical), so the D13 cue is mandatory in every sample.
# - The two stations are forced into one diagonal-touching 8-connected aggregate,
#   so the D17 cue is also mandatory in every sample instead of appearing only
#   occasionally as a weak corner case.
# - Marker colors still matter because both stations can live in the same region;
#   orientation alone picks the station, while stripe colors pick the transform.
#
# c) Hidden assumptions from the original solver that break here
# - `groups_by_size[len(g)]` is invalid because both stations have size 2.
# - 8-connected station extraction is invalid because it would merge the two
#   intended targets into one size-4 cluster.
# - Region choice and transform choice can no longer be separated from station
#   shape: the solver must evaluate train/station/transform candidates jointly.
# - The original generator tolerated couplers merely 8-adjacent to the body.
#   This combo keeps train extraction simple by generating couplers 4-adjacent to
#   the body, matching the component labeling used by the solver.
#
# d) Placement legality
# - The generator still uses output-first construction, but it now reserves one
#   region for a diagonal station pair and repeatedly rebuilds docked trains
#   until both can occupy that region without overlaps or stripe collisions.
# - Station cells keep >=3 cells of clearance from region borders so the docked
#   patches have enough room for marker protrusions and body growth.
# - The input is accepted only if `main(input) == output`.
#
# e) Visual observability
# - A human can see that the two target stations are the same size and differ
#   only by horizontal vs vertical arrangement.
# - The diagonal-touching layout makes the 4-connectivity clue visible: the four
#   station pixels look like one corner-linked shape unless one reasons in terms
#   of orthogonal connectivity.
# - Marker colors remain visibly tied to surrounding stripe colors, so the
#   rotation/flip cue is still recoverable from examples.
#
# f) Visual cleanliness
# - Restricting the task to two trains avoids clutter around the diagonal station
#   pair and keeps the orientation contrast legible.
# - The generator uses stronger station clearance and overlap checks so the two
#   docked trains do not smear together even though their station cells are close.
#
# g) Change inferability
# - No rule depends on numeric color ordering. Stripe colors are only matched by
#   visible equality between train markers and region boundaries.
# - The fused rule can be inferred from a few examples because every sample shows
#   the same two structural clues together: equal-size opposite-orientation
#   stations, and diagonal-only station contact that matters only under
#   4-connectivity.
#
# Post-change assumption audit
# - `main()` assumes there are exactly two 4-connected stations, both size-2
#   lines, with orientations `{h, v}`. The generator enforces that.
# - `main()` assumes each train has exactly two couplers and at least two marker
#   cues from the target region's stripe colors. The generator enforces that.
# - `main()` assumes the correct global assignment is the unique one where every
#   chosen transform satisfies all marker-direction cues and the transformed
#   coupler line matches the station line orientation. The generator validates
#   this by round-tripping through `main()`.

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
FOUR_CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


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


def matching_offset(src_coords, dst_coords):
    src_set = set(src_coords)
    dst_set = set(dst_coords)
    offsets = {
        (dr - sr, dc - sc)
        for sr, sc in src_set
        for dr, dc in dst_set
        if all((r + dr - sr, c + dc - sc) in dst_set for r, c in src_set)
    }
    assert len(offsets) == 1, "Expected a unique alignment offset"
    return next(iter(offsets))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock train sprites onto equal-size stations by splitting stations with 4-connectivity, then matching coupler orientation and stripe-colored markers.

    Input structure:
    - A background color fills the canvas, while edge/internal divider stripes split it into rectangular regions.
    - Color 8 forms train bodies; color 9 marks both isolated station cells and train-attached couplers.
    - The target stations are always two-cell lines, one horizontal and one vertical, and they may touch only diagonally so that 8-connectivity would merge them.
    - Stripe-colored pixels embedded in each train are orientation markers tied to the boundary stripes of the target region.

    Rule:
    - Treat station cells as 9s that are not adjacent to any 8, but cluster those station cells with 4-connectivity only.
    - Detect trains as non-stripe components containing color 8, then test every station/transform pairing.
    - A candidate is valid only when the transformed two-coupler line has the same horizontal/vertical orientation as the candidate station and every marker color points toward the matching boundary stripe of that station's region.
    - Choose the unique global assignment, erase the scattered trains, and redraw each transformed train so its couplers land exactly on the chosen station cells.

    Key insight: The visible station pixels tempt an 8-connected reading, but the correct targets appear only after splitting diagonal contacts with 4-connectivity. Once those equal-size stations are separated, the transformed coupler line orientation identifies which station each train should dock onto.
    """
    inp = grid
    grid = inp.copy()
    H, W = grid.shape
    bg = get_bg(grid)

    # This puzzle reconstructs several implicit structures:
    # - divider stripes (edge + optional internal) that define regions
    # - station cells = isolated 9s, but clustered with 4-connectivity
    # - train sprites = non-stripe components containing 8 and carrying two 9 couplers
    # For each train, the solver evaluates station candidates jointly with rigid
    # transforms, keeping only the choices whose markers face the target region's
    # stripe colors and whose transformed coupler line matches the station line.

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
        lab, n = label(sm, structure=FOUR_CONN)
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

    def xform_coords(coords, shape, flip_ud, rot_k):
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape), flip_ud, rot_k).shape
        return list(result), new_shape

    def marker_score(train, divider_dirs, op):
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

    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

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
            for op in ((False, 0), (False, 1), (False, 2), (False, 3), (True, 0), (True, 1), (True, 2), (True, 3)):
                tbbox = apply_transform(t["bbox"], *op)
                t_body9, _ = xform_coords(t["body9"], t["bbox"].shape, *op)
                if normalize_coords(t_body9) != st["shape"]:
                    continue
                off_r, off_c = matching_offset(t_body9, st["coord_set"])
                score = marker_score(t, st["divider_dirs"], op)
                if score != full_score:
                    continue
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
    assert best_canvas is not None, "No valid 4-connected station assignment found"
    assert not tied, "Expected a unique station assignment"
    return best_canvas


# --------------- generator ---------------

GRID_H_RANGE = (13, 31)
GRID_W_RANGE = (16, 29)
GRID_MIN_AREA = 300
MIN_REGION_DIM = 10
MIN_REGION_DIRS = 3
INTERNAL_STRIPE_THICK = 2

STRIPE_THICK_RANGE = (2, 4)
INTERNAL_PROB = 0.62

STATION_CLEARANCE = 3
PAIR_CANDIDATE_CAP = 96

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
PAIR_RETRIES = 80
DOCK_RETRIES = 40
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
PAIR_PATTERNS = [
    {"h": {(0, 0), (0, 1)}, "v": {(1, 2), (2, 2)}},
    {"h": {(0, 1), (0, 2)}, "v": {(1, 0), (2, 0)}},
    {"h": {(2, 0), (2, 1)}, "v": {(0, 2), (1, 2)}},
    {"h": {(2, 1), (2, 2)}, "v": {(0, 0), (1, 0)}},
]


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


def _adjacent4(p, cells):
    r, c = p
    return any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in cells)


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


def _station_shape(orientation):
    if orientation == "h":
        return {(0, 0), (0, 1)}
    if orientation == "v":
        return {(0, 0), (1, 0)}
    raise ValueError(f"unknown orientation: {orientation}")


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
        if _adjacent4(cp, body):
            continue
        cr, cc = cp
        choices = [(cr + dr, cc + dc) for dr, dc in DIRS4 if (cr + dr, cc + dc) not in blocked]
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


def _build_docked_patch(bg, dirs, orientation):
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


def _all_station_cells_with_clearance(region, rel_cells, top_left):
    rr, cc = top_left
    for dr, dc in rel_cells:
        ar, ac = rr + dr, cc + dc
        if not (region["r0"] + STATION_CLEARANCE <= ar <= region["r1"] - STATION_CLEARANCE):
            return False
        if not (region["c0"] + STATION_CLEARANCE <= ac <= region["c1"] - STATION_CLEARANCE):
            return False
    return True


def _isolated_station_mask(grid):
    H, W = grid.shape
    mask = np.zeros((H, W), dtype=bool)
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
                mask[r, c] = True
    return mask


def _station_connectivity_signature(grid):
    sm = _isolated_station_mask(grid)
    lab4, n4 = label(sm, structure=FOUR_CONN)
    groups4 = []
    for k in range(1, n4 + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab4 == k)]
        groups4.append(coords)

    lab8, n8 = label(sm, structure=np.ones((3, 3), dtype=int))
    sizes8 = sorted(int((lab8 == k).sum()) for k in range(1, n8 + 1))
    sizes4 = sorted(len(g) for g in groups4)
    orientations4 = sorted(line_orientation(g) for g in groups4) if sizes4 == [2, 2] else []
    return sizes4, sizes8, orientations4


def _pair_candidates(regions):
    candidates = []
    for reg in regions:
        for pattern in PAIR_PATTERNS:
            all_rel = pattern["h"] | pattern["v"]
            max_r = max(r for r, _ in all_rel)
            max_c = max(c for _, c in all_rel)
            for rr in range(reg["r0"], reg["r1"] - max_r + 1):
                for cc in range(reg["c0"], reg["c1"] - max_c + 1):
                    if not _all_station_cells_with_clearance(reg, all_rel, (rr, cc)):
                        continue
                    stations = {
                        orientation: [(rr + r, cc + c) for r, c in sorted(rel_cells)]
                        for orientation, rel_cells in pattern.items()
                    }
                    candidates.append({"region": reg, "stations": stations})
    random.shuffle(candidates)
    return candidates[:PAIR_CANDIDATE_CAP]


def _validate_patch_placement(canvas, region, patch, bg, off_r, off_c, station_abs):
    h, w = patch.shape
    station_set = set(station_abs)
    for r in range(h):
        for c in range(w):
            v = int(patch[r, c])
            if v == bg:
                continue
            ar, ac = off_r + r, off_c + c
            if not (region["r0"] <= ar <= region["r1"] and region["c0"] <= ac <= region["c1"]):
                return False
            if v == 9:
                if (ar, ac) not in station_set:
                    return False
                if canvas[ar, ac] not in (bg, 9):
                    return False
            else:
                if canvas[ar, ac] != bg:
                    return False
    return True


def _place_station_pair(base, bg, regions):
    candidates = _pair_candidates(regions)
    for _ in range(PAIR_RETRIES):
        if not candidates:
            break
        cand = random.choice(candidates)
        output = base.copy()
        trains = []
        order = random.sample(["h", "v"], 2)
        ok = True
        for orientation in order:
            station_abs = cand["stations"][orientation]
            placed = False
            for _ in range(DOCK_RETRIES):
                result = _build_docked_patch(bg, cand["region"]["dirs"], orientation)
                if result is None:
                    continue
                patch, coupler_rel = result
                off_r, off_c = matching_offset(coupler_rel, station_abs)
                if not _validate_patch_placement(output, cand["region"], patch, bg, off_r, off_c, station_abs):
                    continue
                paste(output, patch, off_r, off_c, transparent=bg)
                trains.append({"patch": patch, "station": station_abs, "orientation": orientation})
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and len(trains) == 2:
            return output, trains
    return None


def _scatter_trains(base, bg, regions, trains):
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

    forbidden = _dilate(base != bg) | _dilate(station_mask)
    for t, op in zip(trains, ops):
        in_patch = _inverse_patch(t["patch"], op)
        regs = regions[:]
        random.shuffle(regs)
        all_locs = []
        for reg in regs:
            locs = _patch_locations(reg, in_patch, bg, forbidden, margin=1)
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


def _patch_locations(region, patch, bg, occupied, margin=0):
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
            if ok:
                locs.append((rr, cc))
    return locs


def _gen_attempt():
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions = layout

    placed = _place_station_pair(base, bg, regions)
    if placed is None:
        return None
    output, trains = placed

    input_grid = _scatter_trains(base, bg, regions, trains)
    if input_grid is None:
        return None

    sizes4, sizes8, orientations4 = _station_connectivity_signature(input_grid)
    if sizes4 != [2, 2] or sizes8 != [4] or orientations4 != ["h", "v"]:
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
