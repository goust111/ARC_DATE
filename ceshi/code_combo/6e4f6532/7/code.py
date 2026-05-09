# Combo variant of 6e4f6532 (index 7)
# combo_spec: D3/0+D5/3
# Name: 多色车身+翻转触发颜色交换
#
# Design analysis
# a) Concrete rule and trace:
#    The original task used stripe-colored detached markers to choose the docking
#    orientation. This combo removes detached markers and instead gives every train
#    a two-color body: color 8 is the bulk, color 5 is a connected chiral "head".
#    Example trace: suppose a size-2 coupler train must dock onto a station whose
#    cells sit closest to the right boundary of its region. In the old task, marker
#    colors decided which D4 transform to use. Here, we search the 8 flip/rotation
#    transforms until the transformed color-5 head points right. If that winning
#    transform is a pure rotation, the docked train keeps body colors (8 bulk, 5
#    head). If it is a reflected transform (`flip_ud=True` before rotation), the
#    docked train uses the same geometry but swaps body colors 8<->5, so the former
#    head turns into an 8-colored tip and the former bulk turns into 5.
# b) Complexity calibration:
#    Only two body colors are involved, and their roles repeat across all trains:
#    one large region plus one small chiral subshape. This makes the visual change
#    obvious without introducing many color cases. The generator also enforces both
#    reflected and non-reflected dockings inside each sample, so the conditional
#    recoloring is supported by within-example contrast instead of appearing once.
# c) Hidden assumptions that changed:
#    - Station detection can no longer test adjacency only against color 8; color 5
#      is also train body.
#    - Train parsing can no longer treat non-bg non-9 non-body colors as markers;
#      stripe colors never appear inside trains in this combo.
#    - Orientation search no longer compares colors against stripe directions. It
#      instead relies on the color-partitioned body shape and on the station's
#      unique nearest region side.
#    - Redraw can no longer copy transformed pixels verbatim, because reflected
#      dockings recolor the transformed patch after geometry is fixed.
# d) Placement legality:
#    The generator is still output-first at the geometry level. It builds a docked
#    canonical patch whose head already faces the target side, places those patches
#    into regions with >=2-cell station clearance, then scatters inverse-transformed
#    unswapped input trains into free space with >=1-cell clearance. Output rendering
#    applies the 8<->5 swap only after the docking transform has been chosen.
# e) Visual observability:
#    A human can see that the smaller chiral subshape marks the front of each input
#    train. Across examples, trains dock so that this front points toward the unique
#    nearest border side of the matched station's region. Some docked trains keep
#    their colors while others invert the two body colors, and the only visible
#    difference between those cases is whether the necessary docking transform used a
#    reflection. Because both flipped and non-flipped cases occur in one sample, the
#    conditional recoloring is inferable from few examples.
# f) Visual cleanliness:
#    The head remains connected and extreme on one side of the body, so the two-color
#    partition reads as one object instead of noise. Stations still sit away from
#    stripes, and trains keep spacing so regions do not become cluttered.
# g) Change inferability:
#    The color change uses only the same two body colors already visible in the
#    object. There is no extra hidden key such as color ranking. The geometry tells
#    which pixels are "head" versus "bulk", and reflection alone decides whether
#    those two colors swap in the output.
#
# New solver assumptions and generator guarantees:
# - Every input train contains both body colors; the color-5 cells form exactly one
#   transformed copy of the canonical head template below.
# - Every station group belongs to exactly one region and has one strictly nearest
#   region side; the generator rejects ties.
# - For every scattered input train, exactly one D4 transform makes the head face
#   the target side, so the solver's orientation choice is unambiguous.
# - Every generated sample contains at least one reflected docking and at least one
#   non-reflected docking, so the swap rule is exercised instead of degenerating.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste

BODY_COLOR = 8
HEAD_COLOR = 5
BODY_COLORS = {BODY_COLOR, HEAD_COLOR}

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


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def _normalize(coords):
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    r0, c0 = min(rs), min(cs)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


HEAD_CANON_SHAPE = (3, 2)
HEAD_CANON_UP = {(0, 1), (1, 1), (2, 1), (2, 0)}
HEAD_TIP_UP = (0, 1)
HEAD_TEMPLATES = {}
HEAD_TIPS = {}
for _dir in range(4):
    HEAD_TEMPLATES[_dir] = _normalize(
        transform_coords(HEAD_CANON_UP, HEAD_CANON_SHAPE, False, _dir)
    )
    HEAD_TIPS[_dir] = next(
        iter(transform_coords([HEAD_TIP_UP], HEAD_CANON_SHAPE, False, _dir))
    )


def head_direction_from_pixels(coords) -> int | None:
    norm = _normalize(coords)
    for direction, templ in HEAD_TEMPLATES.items():
        if norm == templ:
            return direction
    return None


def nearest_boundary_direction(region, cells) -> int | None:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    dists = {
        0: min(rs) - region["r0"],
        1: region["c1"] - max(cs),
        2: region["r1"] - max(rs),
        3: min(cs) - region["c0"],
    }
    best = min(dists.values())
    winners = [direction for direction, dist in dists.items() if dist == best]
    return winners[0] if len(winners) == 1 else None


def swap_body_colors(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    mask_body = out == BODY_COLOR
    mask_head = out == HEAD_COLOR
    out[mask_body] = HEAD_COLOR
    out[mask_head] = BODY_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-color trains onto stations by coupler count, then swap the body colors only for reflected dockings.

    Input structure:
    - Solid edge/internal stripes partition the canvas into one or two rectangular regions.
    - Stations are isolated 9-clusters; trains are components containing colors 8 and 5 plus attached 9 couplers.
    - Inside each train, the color-5 cells form a connected directional head embedded into the larger color-8 body.

    Rule:
    - Group isolated station clusters and match each train to the unique station whose size equals its coupler count.
    - In the matched station's region, find the unique nearest boundary side of that station cluster.
    - Rotate/flip the train until the transformed color partition makes the head point toward that side, then align the couplers exactly onto the station cells.
    - Erase the source train; redraw the transformed non-9 body cells at the docked location; if the winning transform used `flip_ud=True`, swap body colors 8 and 5 before redraw.

    Key insight:
    - Coupler count still picks the destination, but orientation comes from the train's own two-color anatomy rather than detached markers. Reflection has a second consequence: it preserves geometry but inverts the two body colors, so color change depends on the transform type.
    """
    grid = grid.copy()
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

    def station_groups(stripe_mask):
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
                        if not (0 <= rr < H and 0 <= cc < W):
                            continue
                        if stripe_mask[rr, cc]:
                            continue
                        if int(grid[rr, cc]) in BODY_COLORS:
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

    def xform_coords(coords, shape, flip_ud, rot_k):
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape, dtype=int), flip_ud, rot_k).shape
        return list(result), new_shape

    def choose_best_op(train, target_dir):
        matches = []
        for flip_ud, rot_k in OPS:
            head_t = transform_coords(train["head_pixels"], train["bbox"].shape, flip_ud, rot_k)
            if head_direction_from_pixels(head_t) == target_dir:
                matches.append((flip_ud, rot_k))
        assert len(matches) == 1, "Head template should determine a unique docking orientation"
        return matches[0]

    edge_mask, _, edge_info = find_edge_stripes()
    internal_mask, _ = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups(stripe_mask)
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
        if not any(v in BODY_COLORS for v in vals):
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        abs_coords = [(int(r), int(c)) for r, c in coords]
        body9, head_pixels = [], []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 9 and (r, c) not in station_set:
                body9.append((int(r - r0), int(c - c0)))
            elif v == HEAD_COLOR:
                head_pixels.append((int(r - r0), int(c - c0)))
        assert body9, "Each train must carry at least one coupler"
        assert head_pixels, "Each train must expose a color-5 head"
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body9": body9,
                "head_pixels": head_pixels,
            }
        )

    out = grid.copy()
    for train in trains:
        for r, c in train["abs_coords"]:
            out[r, c] = bg

    for train in trains:
        coupler_count = len(train["body9"])
        candidate_groups = groups_by_size.get(coupler_count)
        assert candidate_groups is not None and len(candidate_groups) == 1, (
            f"Expected one station group for coupler count {coupler_count}"
        )
        group = candidate_groups[0]
        gset = set(group)
        matching_regions = [
            region
            for region in regions
            if all(
                region["r0"] <= r <= region["r1"] and region["c0"] <= c <= region["c1"]
                for r, c in group
            )
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        region = matching_regions[0]
        target_dir = nearest_boundary_direction(region, group)
        assert target_dir is not None, "Station must have a unique nearest boundary side"
        op = choose_best_op(train, target_dir)
        tbbox = apply_transform(train["bbox"], *op)
        if op[0]:
            tbbox = swap_body_colors(tbbox)
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
                if not (
                    region["r0"] <= rr <= region["r1"] and region["c0"] <= cc <= region["c1"]
                ):
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

HEAD_GAP_RANGE = (1, 2)
HEAD_JITTER = [-1, 0, 1]

BODY_GROWTH_ITERS = 160
PATH_BOUNDS_PAD = 4

LAYOUT_RETRIES = 200
DOCK_RETRIES = 30
GENERATE_RETRIES = 400

COLOR_POOL = [c for c in range(8) if c != HEAD_COLOR]


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


def _adjacent_body(p, cells):
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

        thick = {edge: random.randint(*STRIPE_THICK_RANGE) for edge in edges}
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


def _station_shape(size):
    if size == 1:
        return {(0, 0)}
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    return {(0, 0), (0, 1), (1, 0), (1, 1)}


def _head_cells(couplers, direction):
    templ = HEAD_TEMPLATES[direction]
    tip_r, tip_c = HEAD_TIPS[direction]
    r0, c0, r1, c1 = _bbox(couplers)
    gap = random.randint(*HEAD_GAP_RANGE)
    jitter = random.choice(HEAD_JITTER)
    if direction == 0:
        target_tip = (r0 - gap, random.randint(c0, c1) + jitter)
    elif direction == 2:
        target_tip = (r1 + gap, random.randint(c0, c1) + jitter)
    elif direction == 3:
        target_tip = (random.randint(r0, r1) + jitter, c0 - gap)
    else:
        target_tip = (random.randint(r0, r1) + jitter, c1 + gap)
    dr = target_tip[0] - tip_r
    dc = target_tip[1] - tip_c
    return {(r + dr, c + dc) for r, c in templ}


def _head_is_extreme(head_cells, body, direction):
    if direction == 0:
        return min(r for r, _ in head_cells) == min(r for r, _ in body)
    if direction == 2:
        return max(r for r, _ in head_cells) == max(r for r, _ in body)
    if direction == 3:
        return min(c for _, c in head_cells) == min(c for _, c in body)
    return max(c for _, c in head_cells) == max(c for _, c in body)


def _connect_couplers_to_body(couplers, body, blocked, bounds):
    for cp in couplers:
        if _adjacent_body(cp, body):
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


def _build_docked_patch(bg, head_dir, size):
    couplers = _station_shape(size)
    head_cells = _head_cells(couplers, head_dir)
    if head_cells & couplers:
        return None
    body = set(head_cells)
    blocked = set(couplers)
    all_seed = body | blocked
    sr0, sc0, sr1, sc1 = _bbox(all_seed)
    bounds = (
        sr0 - PATH_BOUNDS_PAD,
        sr1 + PATH_BOUNDS_PAD,
        sc0 - PATH_BOUNDS_PAD,
        sc1 + PATH_BOUNDS_PAD,
    )
    if not _connect_couplers_to_body(couplers, body, blocked, bounds):
        return None
    if len(body) > BODY_MAX:
        return None
    target = random.randint(max(BODY_MIN, len(body)), BODY_MAX)
    _grow_body(body, blocked, target)
    if len(body - head_cells) < 1:
        return None
    if not _head_is_extreme(head_cells, body, head_dir):
        return None
    all_cells = body | blocked
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
        patch[r - r0, c - c0] = BODY_COLOR
    for r, c in head_cells:
        patch[r - r0, c - c0] = HEAD_COLOR
    rel_couplers = []
    for r, c in couplers:
        patch[r - r0, c - c0] = 9
        rel_couplers.append((r - r0, c - c0))
    rel_head = [(r - r0, c - c0) for r, c in head_cells]
    if head_direction_from_pixels(rel_head) != head_dir:
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


def _input_orientation_matches(patch, target_dir):
    matches = []
    head_pixels = [(int(r), int(c)) for r, c in np.argwhere(patch == HEAD_COLOR)]
    for op in OPS:
        if head_direction_from_pixels(transform_coords(head_pixels, patch.shape, *op)) == target_dir:
            matches.append(op)
    return matches


def _place_docked_trains(base, bg, regions):
    output = base.copy()
    occupied = output != bg

    n_trains = random.choice(TRAIN_COUNTS)
    sizes = random.sample(STATION_SIZES, n_trains)
    target_dirs = random.sample([0, 1, 2, 3], n_trains)
    trains = []
    for size, target_dir in zip(sizes, target_dirs):
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            result = _build_docked_patch(bg, target_dir, size)
            if result is None:
                continue
            patch, coupler_rel = result
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), coupler_rel, True, 0)
            locs = [
                (r0, c0)
                for r0, c0 in locs
                if nearest_boundary_direction(
                    reg, [(r0 + r, c0 + c) for r, c in coupler_rel]
                )
                == target_dir
            ]
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
                    "dock_origin": (r0, c0),
                    "target_dir": target_dir,
                }
            )
            break
        else:
            return None
    return trains


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
        valid = [
            op
            for op in _valid_ops_for_patch(train["patch"])
            if len(
                _input_orientation_matches(
                    _inverse_patch(train["patch"], op), train["target_dir"]
                )
            )
            == 1
        ]
        if not valid:
            return None
        valid_ops.append(valid)

    ops = None
    for _ in range(120):
        trial_ops = [random.choice(valid) for valid in valid_ops]
        if all(op == (False, 0) for op in trial_ops):
            continue
        if {op[0] for op in trial_ops} != {False, True}:
            continue
        ops = trial_ops
        break
    if ops is None:
        return None

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

    return input_grid, ops


def _render_docked_output(base, bg, trains, ops):
    output = base.copy()
    for train in trains:
        for r, c in train["station"]:
            output[r, c] = 9
    for train, op in zip(trains, ops):
        patch = train["patch"]
        if op[0]:
            patch = swap_body_colors(patch)
        r0, c0 = train["dock_origin"]
        paste(output, patch, r0, c0, transparent=bg)
    return output


def _gen_attempt():
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions = layout

    trains = _place_docked_trains(base, bg, regions)
    if trains is None:
        return None

    scattered = _scatter_trains(base, bg, regions, trains)
    if scattered is None:
        return None
    input_grid, ops = scattered
    output = _render_docked_output(base, bg, trains, ops)

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
