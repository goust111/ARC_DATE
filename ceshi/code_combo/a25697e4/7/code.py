# Combo variant of a25697e4 (index 7)
# combo_spec: D6/2+D15/2
# Name: 距离编码旋转决定变色

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy import ndimage
from scipy.ndimage import label

from arc_utils import STRUCT4, STRUCT8, get_bg, paste, random_sprite
from arc_utils import bbox as _bbox


# Design analysis:
# - Concrete trace: if a tool's visible bbox-gap to its matched frame is d=5, then 5 % 4 = 1,
#   so the solver must rotate the tool by 90 degrees once before docking. Because the rotation
#   count is odd, the two tool colors are swapped after the placement is chosen. If d=6 instead,
#   the solver rotates by 180 degrees and keeps the colors unchanged. In the original task the
#   geometry search itself chose the orientation and color swap depended on reflection; here the
#   distance fixes the rotation first, and the rotation parity alone controls recoloring.
# - Complexity calibration: the encoded variable has only four states (0/1/2/3 quarter-turns),
#   which is enough to be visible but not so large that the examples become noisy. Plug areas stay
#   unique across pairs so the tool-to-frame pairing is still legible from shape counts, while the
#   distance band 4..11 keeps the tool visibly separated from the frame and gives each modulo class
#   at least one clean placement.
# - Hidden-assumption audit for `main()`: the original solver assumed the correct docking transform
#   could be found by trying all rotations/flips. That is no longer valid, so this variant removes
#   reflection from the intended rule and derives the single allowed rotation from `bbox_gap`.
#   This introduces a new assumption: both solver and generator must measure distance on the tight
#   foreground bbox of the free tool, not on array padding. The generator therefore crops each tool
#   before undocking and searches only placements whose tight bbox has the required modulo.
# - Legality + observability: the generator still builds the docked output first, then undocks each
#   tool by inverse-rotating it and pre-swapping colors for odd rotations so that the solver can
#   recover the target output via its parity rule. Placement search enforces in-bounds, no overlap,
#   no border contact, and padding from existing objects. Because odd rotation classes visibly swap
#   which color becomes the plug versus the handle, the fusion effect remains human-observable.


def _bbox_gap(a, b) -> int:
    """Manhattan gap between two axis-aligned bboxes (r0,c0,r1,c1)."""
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def _gap1(x0, x1, y0, y1):
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return _gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1)


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _crop_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    bb = _bbox(patch != int(bg))
    assert bb is not None
    r0, c0, r1, c1 = bb
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_anchor_groups(g: np.ndarray, bg: int):
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        pts_list = [(int(r), int(c)) for r, c in pts]
        touches = any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts_list)
        if not touches:
            continue
        fixed_components.append(
            {
                "pts": pts_list,
                "bbox": (
                    int(pts[:, 0].min()),
                    int(pts[:, 1].min()),
                    int(pts[:, 0].max()),
                    int(pts[:, 1].max()),
                ),
            }
        )
        for r, c in pts_list:
            fixed_mask[r, c] = True

    assert fixed_components, "must find at least one border-anchored frame"

    parent = list(range(len(fixed_components)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(fixed_components)):
        for j in range(i + 1, len(fixed_components)):
            if _bbox_gap(fixed_components[i]["bbox"], fixed_components[j]["bbox"]) <= 2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(fixed_components)):
        groups.setdefault(find(i), []).append(i)

    anchor_groups = []
    for idxs in groups.values():
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]
        rs = [r for r, _c in pts]
        cs = [c for _r, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        socket_mask = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(g[r, c]) != int(bg):
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        socket_mask[r, c] = True
                        break

        cavity_mask = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_c = None
        best_id = None
        for cid in range(1, int(sub_n) + 1):
            ptsc = np.argwhere(sub_lab == int(cid))
            area = int(len(ptsc))
            adj = False
            for rr, cc in ptsc:
                r = int(bbox[0] + rr)
                c = int(bbox[1] + cc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r2, c2 = r + dr, c + dc
                    if bbox[0] <= r2 <= bbox[2] and bbox[1] <= c2 <= bbox[3] and mask[r2, c2]:
                        adj = True
                        break
                if adj:
                    break
            if not adj:
                continue
            cand = (area, -int(cid))
            if best_c is None or cand > best_c:
                best_c = cand
                best_id = int(cid)

        if best_id is not None:
            ptsc = np.argwhere(sub_lab == int(best_id))
            for rr, cc in ptsc:
                cavity_mask[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        dir_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dir_counts = [0, 0, 0, 0]
        for r, c in pts:
            for di, (dr, dc) in enumerate(dir_list):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(bg):
                    dir_counts[di] += 1
        open_dir = dir_list[int(np.argmax(dir_counts))]

        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "socket_mask": socket_mask,
                "cavity_mask": cavity_mask,
                "open_dir": open_dir,
                "cavity_size": int(cavity_mask.sum()),
                "size": int(mask.sum()),
            }
        )

    anchor_groups.sort(key=lambda d: -int(d["size"]))
    return anchor_groups, fixed_mask


def _extract_movable_components(g: np.ndarray, fixed_mask: np.ndarray, bg: int):
    fg = g != int(bg)
    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == int(cid))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == int(cid)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = 0
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "size": int(len(pts)),
            }
        )
    mov_comps.sort(key=lambda d: -int(d["size"]))
    return mov_comps


def _match_components(mov_comps: list[dict], anchor_groups: list[dict]):
    comp_color_counts: list[dict[int, int]] = []
    for comp in mov_comps:
        vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]
        comp_color_counts.append(dict(Counter(vals)))

    def compatible(ci: int, gi: int) -> bool:
        cav_sz = int(anchor_groups[gi]["cavity_size"])
        return any(int(v) == cav_sz for v in comp_color_counts[ci].values())

    if len(mov_comps) == len(anchor_groups) and len(anchor_groups) > 1:
        best_perm = None
        best_score = None
        for perm in itertools.permutations(range(len(anchor_groups))):
            ok = True
            dist_sum = 0
            for ci, gi in enumerate(perm):
                if not compatible(ci, int(gi)):
                    ok = False
                    break
                dist_sum += _bbox_gap(mov_comps[ci]["bbox"], anchor_groups[int(gi)]["bbox"])
            if not ok:
                continue
            score = -int(dist_sum)
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None, "must find a compatible tool-to-frame permutation"
        group_for_comp = [int(x) for x in best_perm]
    else:
        group_for_comp = []
        for ci, comp in enumerate(mov_comps):
            best_g = None
            best_d = None
            for gi, grp in enumerate(anchor_groups):
                if not compatible(ci, gi):
                    continue
                d = _bbox_gap(comp["bbox"], grp["bbox"])
                if best_d is None or d < best_d:
                    best_d = d
                    best_g = gi
            assert best_g is not None, "every movable tool must match a cavity size"
            group_for_comp.append(int(best_g))

    return group_for_comp, comp_color_counts


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.array(grid, dtype=int)
    bg = get_bg(g)
    H, W = g.shape

    anchor_groups, fixed_mask = _extract_anchor_groups(g, bg)
    mov_comps = _extract_movable_components(g, fixed_mask, bg)
    group_for_comp, comp_color_counts = _match_components(mov_comps, anchor_groups)

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]
    meta = []

    for ci, (comp, gi) in enumerate(zip(mov_comps, group_for_comp)):
        grp = anchor_groups[int(gi)]
        placed_grp = placed_groups[int(gi)]
        socket_mask = grp["socket_mask"]
        cavity_mask = grp["cavity_mask"]
        open_dr, open_dc = grp["open_dir"]
        comp_counts = comp_color_counts[ci]

        encoded_distance = _bbox_gap(comp["bbox"], grp["bbox"])
        rot_k = int(encoded_distance % 4)
        patch = np.rot90(comp["patch"], rot_k)
        mask = np.rot90(comp["mask"], rot_k)
        coords = np.argwhere(mask)
        cav_sz = int(grp["cavity_size"])
        cols_here = sorted({int(x) for x in patch[mask].flatten() if int(x) != 0})
        plug_colors = [int(col) for col in cols_here if int(((patch == int(col)) & mask).sum()) == cav_sz]
        assert plug_colors, "distance-coded rotation must preserve exactly one cavity-sized color region"

        best_state = None
        best_score = None
        h, w = mask.shape
        for r0 in range(0, H - h + 1):
            for c0 in range(0, W - w + 1):
                ok = True
                for rr, cc in coords:
                    r = int(r0 + rr)
                    c = int(c0 + cc)
                    if placed_global[r, c]:
                        ok = False
                        break
                if not ok:
                    continue

                adj = 0
                for rr, cc in coords:
                    r = int(r0 + rr)
                    c = int(c0 + cc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        r2, c2 = r + dr, c + dc
                        if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                            adj += 1
                if adj == 0:
                    continue

                socket_hits = 0
                cavity_hits = 0
                outside = 0
                for rr, cc in coords:
                    r = int(r0 + rr)
                    c = int(c0 + cc)
                    if socket_mask[r, c]:
                        socket_hits += 1
                    if cavity_mask[r, c]:
                        cavity_hits += 1
                    br0, bc0, br1, bc1 = grp["bbox"]
                    if open_dc == -1 and c < bc0:
                        outside += 1
                    elif open_dc == 1 and c > bc1:
                        outside += 1
                    elif open_dr == -1 and r < br0:
                        outside += 1
                    elif open_dr == 1 and r > br1:
                        outside += 1

                plug_best = None
                for plug_col in plug_colors:
                    plug_hits = 0
                    plug_outside = 0
                    nonplug_in_cav = 0
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        in_cav = bool(cavity_mask[r, c])
                        col = int(patch[rr, cc])
                        if col == int(plug_col):
                            if in_cav:
                                plug_hits += 1
                            else:
                                plug_outside += 1
                        elif in_cav:
                            nonplug_in_cav += 1
                    plug_full = int(plug_hits == cav_sz and plug_outside == 0 and nonplug_in_cav == 0)
                    plug_score = (plug_full, plug_hits, -nonplug_in_cav, -plug_outside)
                    if plug_best is None or plug_score > plug_best:
                        plug_best = plug_score

                dist = _bbox_gap(grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
                score = tuple(plug_best) + (
                    int(cavity_hits),
                    int(adj),
                    int(outside),
                    int(socket_hits),
                    -int(dist),
                    -int(r0),
                    -int(c0),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_state = (patch.copy(), mask.copy(), int(r0), int(c0))

        assert best_state is not None, "must find a valid placement for every movable tool"
        placed_patch, placed_mask, r0, c0 = best_state

        swapped = False
        if rot_k % 2 == 1 and len(comp_counts) == 2:
            a, b = sorted(comp_counts.keys())
            placed_patch = _swap_colors(placed_patch, a, b)
            swapped = True

        for rr, cc in np.argwhere(placed_mask):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(placed_patch[rr, cc])
            placed_global[r, c] = True
            placed_grp[r, c] = True

        meta.append(
            {
                "group_index": int(gi),
                "distance": int(encoded_distance),
                "rotation_k": int(rot_k),
                "swap": bool(swapped),
            }
        )

    if return_meta:
        return out, meta
    return out


def analyze_fusion(grid: np.ndarray) -> list[dict]:
    """Return per-tool distance/rotation/swap metadata for this variant."""
    _out, meta = _solve(grid, return_meta=True)
    return meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock free two-color tools into border frames, with distance deciding the quarter-turn.

    Input structure:
    - bg is the dominant color.
    - Border-touching pieces cluster into monochrome frames that contain staircase-shaped bg cavities.
    - Away from the border sit movable 2-color tools; one color-region has the same area as exactly one cavity.

    Rule:
    - Pair each tool with the frame whose cavity area matches one of the tool's color-region sizes.
    - Measure the Manhattan bbox-gap from that tool to its matched frame; `gap % 4` gives the number of 90-degree rotations.
    - Rotate the tool by that amount, dock the cavity-sized region into the notch, and swap the two colors iff the rotation count is odd.

    Key insight: shape counts still identify the partner frame, but orientation no longer comes from a free search — the tool-frame distance encodes the only allowed rotation, and its parity also controls recoloring.
    """
    return _solve(grid)


# --------------- generator (output-first) ---------------

# ── CFG: canvas ──
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [1, 2]

# ── CFG: frame geometry ──
FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)

# ── CFG: tool sizing ──
HANDLE_AREA_RANGE = (4, 10)
TOOL_TOTAL_BOUNDS = (10, 15)
HANDLE_SPRITE_DIM = (2, 5)

# ── CFG: placement ──
UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4
ENCODED_DISTANCE_MIN = 4
ENCODED_DISTANCE_MAX = 11

# ── CFG: retry limits ──
PAIR_SPEC_RETRIES = 300
HANDLE_SPRITE_RETRIES = 30
GEN_ATTEMPT_RETRIES = 300

_ROTATION_CURSOR = random.randrange(4)


def _orient_left_to_side(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.fliplr(arr)
    if side == "top":
        return np.rot90(arr, 3)
    if side == "bottom":
        return np.rot90(arr, 1)
    raise ValueError(side)


def _notch_offsets(notch_n: int, fw: int, bot_h: int) -> list[tuple[int, int]]:
    if notch_n == 1:
        return [(0, fw - 1)]
    if notch_n == 2:
        return [(0, fw - 2), (0, fw - 1)]
    if notch_n == 3:
        if bot_h >= 2:
            return [(0, fw - 2), (0, fw - 1), (1, fw - 1)]
        return [(0, fw - 3), (0, fw - 2), (0, fw - 1)]
    raise ValueError(notch_n)


def _make_handle_sprite(handle_area: int, bg: int, handle_color: int) -> np.ndarray | None:
    lo, hi = HANDLE_SPRITE_DIM
    dims = [(h, w) for h in range(lo, hi) for w in range(lo, hi) if handle_area <= h * w <= handle_area * 2]
    if not dims:
        return None
    for _ in range(HANDLE_SPRITE_RETRIES):
        h, w = random.choice(dims)
        sprite = random_sprite(
            h,
            w,
            density=handle_area / (h * w),
            color_palette=[handle_color],
            connectivity=4,
            background=bg,
        )
        if int(np.sum(sprite != int(bg))) == int(handle_area):
            return sprite
    return None


def _build_left_pair(
    *,
    bg: int,
    frame_color: int,
    plug_color: int,
    handle_color: int,
    fw: int,
    top_h: int,
    bot_h: int,
    notch_n: int,
    handle_area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), int(bg), dtype=int)
    frame[:top_h, :] = int(frame_color)
    frame[top_h + 1 :, :] = int(frame_color)
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = int(bg)

    cavity = frame == int(bg)
    attach_rows = {r for r in range(fh) if cavity[r, fw - 1]}

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None

    hs_h, hs_w = handle_sprite.shape
    sprite_left_rows = {r for r in range(hs_h) if handle_sprite[r, 0] != int(bg)}
    valid_offsets = [voff for voff in range(fh) if any((voff + sr) in attach_rows for sr in sprite_left_rows)]
    if not valid_offsets:
        return None

    voff = random.choice(valid_offsets)
    total_h = max(fh, voff + hs_h)
    total_w = fw + hs_w

    frame_full = np.full((total_h, total_w), int(bg), dtype=int)
    frame_full[:fh, :fw] = frame

    tool = np.full((total_h, total_w), int(bg), dtype=int)
    tool[:fh, :fw][cavity] = int(plug_color)
    handle_mask = handle_sprite != int(bg)
    tool[voff : voff + hs_h, fw : fw + hs_w][handle_mask] = int(handle_color)
    return frame_full, tool


def _build_param_options() -> list[tuple[int, int, int, int]]:
    fw_lo, fw_hi = FRAME_W_RANGE
    nn_lo, nn_hi = NOTCH_N_RANGE
    th_lo, th_hi = TOP_H_RANGE
    bh_lo, bh_hi = BOT_H_RANGE
    pa_lo, pa_hi = PLUG_AREA_BOUNDS
    return [
        (fw, top_h, bot_h, notch_n)
        for fw in range(fw_lo, fw_hi)
        for notch_n in range(nn_lo, nn_hi)
        if pa_lo <= fw + notch_n <= pa_hi
        for top_h in range(th_lo, th_hi)
        for bot_h in range(bh_lo, bh_hi)
    ]


PARAM_OPTIONS = _build_param_options()


def _pick_pair_specs(num_pairs: int) -> list[dict]:
    ha_lo, ha_hi = HANDLE_AREA_RANGE
    tt_lo, tt_hi = TOOL_TOTAL_BOUNDS
    specs: list[dict] = []
    for _ in range(PAIR_SPEC_RETRIES):
        specs.clear()
        used_plug_areas: set[int] = set()
        for _pair in range(num_pairs):
            candidates = [x for x in PARAM_OPTIONS if x[0] + x[3] not in used_plug_areas]
            if not candidates:
                specs = []
                break
            fw, top_h, bot_h, notch_n = random.choice(candidates)
            plug_area = fw + notch_n
            specs.append(
                {
                    "fw": fw,
                    "top_h": top_h,
                    "bot_h": bot_h,
                    "notch_n": notch_n,
                    "plug_area": plug_area,
                }
            )
            used_plug_areas.add(int(plug_area))
        if len(specs) != num_pairs:
            continue
        all_plug_areas = {int(s["plug_area"]) for s in specs}
        ok = True
        for spec in specs:
            pa = int(spec["plug_area"])
            handle_choices = [
                h
                for h in range(ha_lo, ha_hi)
                if tt_lo <= pa + h <= tt_hi and h != pa and h not in all_plug_areas
            ]
            if not handle_choices:
                ok = False
                break
            spec["handle_area"] = random.choice(handle_choices)
        if ok:
            return [dict(s) for s in specs]
    return []


def _candidate_origins(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "left":
        origins = [(r0, 0) for r0 in range(H - ph + 1)]
    elif side == "right":
        origins = [(r0, W - pw) for r0 in range(H - ph + 1)]
    elif side == "top":
        origins = [(0, c0) for c0 in range(W - pw + 1)]
    elif side == "bottom":
        origins = [(H - ph, c0) for c0 in range(W - pw + 1)]
    else:
        raise ValueError(side)
    random.shuffle(origins)
    return origins


def _place_docked_pair(
    i: int,
    spec: dict,
    colors: list[int],
    side: str,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list,
    frame_bboxes: list,
) -> dict | None:
    frame_color = colors[3 * i]
    plug_color = colors[3 * i + 1]
    handle_color = colors[3 * i + 2]

    result = _build_left_pair(
        bg=bg,
        frame_color=frame_color,
        plug_color=plug_color,
        handle_color=handle_color,
        fw=spec["fw"],
        top_h=spec["top_h"],
        bot_h=spec["bot_h"],
        notch_n=spec["notch_n"],
        handle_area=spec["handle_area"],
    )
    if result is None:
        return None
    frame_left, tool_left = result
    scene_left = np.where(tool_left != int(bg), tool_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool_scene = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    frame_bb = _bbox(frame_scene != int(bg))
    assert frame_bb is not None
    fr0, fc0, fr1, fc1 = frame_bb

    origin = None
    frame_bbox = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        cand_frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if side in ("left", "right") and (cand_frame_bbox[0] == 0 or cand_frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (cand_frame_bbox[1] == 0 or cand_frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) <= UNION_BBOX_GAP_MIN - 1 for prev in union_bboxes):
            continue
        if any(_bbox_gap(cand_frame_bbox, prev) <= FRAME_BBOX_GAP_MIN - 1 for prev in frame_bboxes):
            continue
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != int(bg)) & (scene != int(bg))):
            continue
        origin = (int(r0), int(c0))
        frame_bbox = cand_frame_bbox
        break
    if origin is None or frame_bbox is None:
        return None

    r0, c0 = origin
    paste(output, scene, r0, c0, transparent=bg)
    paste(input_grid, frame_scene, r0, c0, transparent=bg)
    union_bboxes.append((int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)))
    frame_bboxes.append(frame_bbox)

    tight_tool = _crop_patch(tool_scene, bg)
    return {
        "tool": tight_tool,
        "frame_bbox": frame_bbox,
        "plug_color": int(plug_color),
        "handle_color": int(handle_color),
        "plug_area": int(spec["plug_area"]),
        "handle_area": int(spec["handle_area"]),
    }


def _prepare_input_patch(info: dict, rot_k: int) -> np.ndarray:
    patch = np.rot90(info["tool"], -int(rot_k))
    if int(rot_k) % 2 == 1:
        patch = _swap_colors(patch, int(info["plug_color"]), int(info["handle_color"]))
    return patch


def _choose_rotation_targets(num_pairs: int) -> list[int]:
    global _ROTATION_CURSOR
    targets = [int((_ROTATION_CURSOR + i) % 4) for i in range(num_pairs)]
    _ROTATION_CURSOR = int((_ROTATION_CURSOR + num_pairs) % 4)
    return targets


def _distance_encoded_origins(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    bg: int,
    target_frame_bbox: tuple[int, int, int, int],
    rot_k: int,
) -> list[tuple[int, int, int]]:
    H, W = canvas.shape
    h, w = patch.shape
    mask = patch != int(bg)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    pr0, pc0 = coords.min(axis=0)
    pr1, pc1 = coords.max(axis=0)
    pr0, pc0, pr1, pc1 = int(pr0), int(pc0), int(pr1), int(pc1)

    occ = canvas != int(bg)
    if TOOL_PADDING > 0:
        struct = STRUCT4 if TOOL_PADDING_CONN == 4 else STRUCT8
        occ = ndimage.binary_dilation(occ, iterations=TOOL_PADDING, structure=struct)

    origins = [(r0, c0) for r0 in range(H - h + 1) for c0 in range(W - w + 1)]
    random.shuffle(origins)
    valid = []
    for r0, c0 in origins:
        if r0 + pr0 < TOOL_BORDER or c0 + pc0 < TOOL_BORDER:
            continue
        if r0 + pr1 >= H - TOOL_BORDER or c0 + pc1 >= W - TOOL_BORDER:
            continue
        if bool((occ[r0 : r0 + h, c0 : c0 + w] & mask).any()):
            continue
        placed_bbox = (int(r0 + pr0), int(c0 + pc0), int(r0 + pr1), int(c0 + pc1))
        d = _bbox_gap(placed_bbox, target_frame_bbox)
        if d < ENCODED_DISTANCE_MIN or d > ENCODED_DISTANCE_MAX:
            continue
        if d % 4 != int(rot_k):
            continue
        valid.append((int(r0), int(c0), int(d)))
    return valid


def _undock_tool(
    info: dict,
    bg: int,
    input_grid: np.ndarray,
    placement_canvas: np.ndarray,
    rotation_targets: list[int],
) -> bool:
    for rot_k in rotation_targets:
        patch = _prepare_input_patch(info, rot_k)
        origins = _distance_encoded_origins(
            placement_canvas,
            patch,
            bg=bg,
            target_frame_bbox=info["frame_bbox"],
            rot_k=rot_k,
        )
        if not origins:
            continue
        r0, c0, d = random.choice(origins)
        paste(input_grid, patch, r0, c0, transparent=bg)
        paste(placement_canvas, patch, r0, c0, transparent=bg)
        info["rotation_k"] = int(rot_k)
        info["encoded_distance"] = int(d)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_pairs = random.choice(NUM_PAIRS_CHOICES)

    pair_specs = _pick_pair_specs(num_pairs)
    if len(pair_specs) != num_pairs:
        return None

    colors = [c for c in range(10) if c != int(bg)]
    random.shuffle(colors)
    if len(colors) < 3 * num_pairs:
        return None

    sides = random.sample(["left", "right", "top", "bottom"], k=num_pairs)
    output = np.full((H, W), int(bg), dtype=int)
    input_grid = np.full((H, W), int(bg), dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict] = []

    for i, spec in enumerate(pair_specs):
        info = _place_docked_pair(
            i,
            spec,
            colors,
            sides[i],
            bg,
            H,
            W,
            output,
            input_grid,
            union_bboxes,
            frame_bboxes,
        )
        if info is None:
            return None
        placed_tools.append(info)

    placement_canvas = output.copy()
    rotation_targets = _choose_rotation_targets(num_pairs)
    random.shuffle(rotation_targets)
    for info, rot_k in zip(placed_tools, rotation_targets):
        if not _undock_tool(info, bg, input_grid, placement_canvas, [int(rot_k)]):
            return None

    try:
        solved = main(input_grid)
    except Exception:
        return None
    if np.array_equal(solved, output):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
