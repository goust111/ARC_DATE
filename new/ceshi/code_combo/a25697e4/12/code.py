# Combo variant of a25697e4 (index 12)
# combo_spec: D8/2+D15/1
# Name: 对面边缘对冲放置
#
# Design analysis:
# a) Concrete behavior trace:
# - Original task: every frame lives on one border, while every movable 2-color tool is away from all borders.
#   Matching is done by cavity area, then the chosen tool is rotated/flipped into the cavity.
# - This combo changes the spatial story at two points simultaneously.
#   1. When there are 2 pairs, the frames must occupy opposite borders rather than arbitrary separate borders.
#   2. Each tool is not merely "far away"; it is attached to the border opposite its own target frame.
# - Example: if a frame is on the left border, its tool is generated on the right border as a detached 2-color
#   sprite. In the output that sprite crosses the whole grid and docks into the left frame. With two pairs on
#   left/right, each border therefore contains one frame and the other pair's tool.
#
# b) Complexity calibration:
# - The visible combo signal is not reduced to a weak distance tweak. Tools actually touch borders, so the solver
#   must distinguish "border-touching frame pieces" from "border-touching tools".
# - Pairing is still controlled by the original cavity-area rule, which keeps the task learnable. The generator
#   forces distinct cavity sizes across pairs to avoid ambiguous matches.
# - Freedom is kept moderate: 1 or 2 pairs, only opposite-side pairings, and the same frame/tool geometry family.
#
# c) Hidden-assumption audit against the original solver:
# - Original assumption broken: "all border-touching foreground belongs to fixed frames". Not true anymore because
#   tools now touch borders too.
# - Original distance fallback becomes misleading because the correct frame can be the farthest one. The combo
#   solver therefore prioritizes cavity-size compatibility and opposite-edge evidence, not nearest distance.
# - Original placement search still works once the frame mask is detected correctly, because docking geometry,
#   D4/D8 transforms, and conditional color swapping are unchanged.
#
# d) Placement legality:
# - Frames are placed output-first on valid borders and never touch corners.
# - Undocked tools are then placed on the opposite border, also away from corners and with 4-neighborhood
#   separation from every pre-existing object, so no accidental component merging occurs.
# - When there are 2 pairs, the two frame borders are guaranteed to be opposite, and each tool is guaranteed to
#   occupy the opposite border of its own frame.
#
# e) Visual observability:
# - Across samples, a human can directly see that the movable tools are sitting on borders rather than floating in
#   the interior, and that they end up docking into frames on the opposite side of the board.
# - The unchanged cavity-area rule is still recoverable because each tool has exactly one color region whose area
#   matches one frame cavity.
#
# f) Visual cleanliness:
# - Only frames are monochrome border structures; tools are always 2-color single components. This keeps the two
#   object classes separable even when they share one border in 2-pair scenes.
# - Corner placements are forbidden and same-border objects are kept separated, so border regions do not turn into
#   unreadable clusters.
#
# g) New solver assumptions and generator guarantees:
# - Every tool is a single 4-connected 2-color component.
# - Every frame is formed by 2 nearby monochrome border-touching components of one color whose bbox contains one
#   bg cavity.
# - Each tool touches exactly one border, and that border is opposite its matched frame border.
# - The generator enforces all of these conditions so the solver's classification and one-to-one matching remain
#   valid for every generated sample.

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox as _bbox, get_bg, paste, random_sprite


def _bbox_gap(a, b) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def _gap1(x0, x1, y0, y1):
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return _gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1)


def _touch_sides(pts: np.ndarray, H: int, W: int) -> set[str]:
    sides: set[str] = set()
    if np.any(pts[:, 0] == 0):
        sides.add("top")
    if np.any(pts[:, 0] == H - 1):
        sides.add("bottom")
    if np.any(pts[:, 1] == 0):
        sides.add("left")
    if np.any(pts[:, 1] == W - 1):
        sides.add("right")
    return sides


def _opposite_side(side: str) -> str:
    return {
        "left": "right",
        "right": "left",
        "top": "bottom",
        "bottom": "top",
    }[side]


def _open_dir_for_side(side: str) -> tuple[int, int]:
    return {
        "left": (0, 1),
        "right": (0, -1),
        "top": (1, 0),
        "bottom": (-1, 0),
    }[side]


def _extract_cavity(mask: np.ndarray, grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    cavity = np.zeros_like(mask, dtype=bool)
    r0, c0, r1, c1 = bbox
    sub_bg = grid[r0 : r1 + 1, c0 : c1 + 1] == int(bg)
    sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
    best = None
    best_id = None
    for cid in range(1, sub_n + 1):
        ptsc = np.argwhere(sub_lab == cid)
        area = int(len(ptsc))
        adj = False
        for rr, cc in ptsc:
            gr = int(r0 + rr)
            gc = int(c0 + cc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r2, c2 = gr + dr, gc + dc
                if r0 <= r2 <= r1 and c0 <= c2 <= c1 and mask[r2, c2]:
                    adj = True
                    break
            if adj:
                break
        if not adj:
            continue
        cand = (area, -cid)
        if best is None or cand > best:
            best = cand
            best_id = cid
    if best_id is None:
        return cavity
    ptsc = np.argwhere(sub_lab == best_id)
    for rr, cc in ptsc:
        cavity[int(r0 + rr), int(c0 + cc)] = True
    return cavity


def _gen_variants(patch: np.ndarray, mask: np.ndarray):
    base: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        base.append((p, m, False))
        base.append((np.fliplr(p), np.fliplr(m), True))
    uniq: list[tuple[np.ndarray, np.ndarray, bool]] = []
    seen = set()
    for p, m, is_flip in base:
        key = (m.tobytes(), p[m].tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m, is_flip))
    return uniq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move edge-hugging tools across the board so they dock into opposite-edge frames.

    Input structure:
    - The background is the most common color.
    - Frames are formed by 2 nearby monochrome components touching one border; when there are 2 frames, they sit on
      opposite borders.
    - Tools are separate 2-color 4-connected components; unlike the base task, they may also touch a border, and
      each tool lies on the border opposite its target frame.

    Rule:
    - Pair each tool with the frame whose cavity size matches one of the tool's two color-region sizes.
    - Rotate and optionally reflect the tool so one color region fills the frame cavity exactly while the other
      region protrudes outward from the frame.
    - If the winning docking transform uses a reflection and the tool's two color regions have different sizes,
      swap those two colors in the docked result.

    Key insight: The key distinction is no longer "border object versus interior object". Instead, border-touching
    monochrome pieces define frames, border-touching 2-color components define tools, and each tool must traverse to
    the opposite-edge frame identified by cavity-area matching.
    """
    g = grid.copy()
    bg = get_bg(g)
    H, W = g.shape

    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    raw_components = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = lab[r0 : r1 + 1, c0 : c1 + 1] == cid
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = 0
        cols = sorted({int(x) for x in patch[mask].flatten() if int(x) != 0})
        raw_components.append(
            {
                "cid": cid,
                "pts": pts,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask_local": mask,
                "patch": patch,
                "colors": cols,
                "touch_sides": _touch_sides(pts, H, W),
                "size": int(len(pts)),
            }
        )

    frame_piece_ids: set[int] = set()
    frame_pieces = []
    for comp in raw_components:
        if len(comp["colors"]) != 1:
            continue
        if len(comp["touch_sides"]) != 1:
            continue
        side = next(iter(comp["touch_sides"]))
        color = int(comp["colors"][0])
        frame_pieces.append({"cid": comp["cid"], "bbox": comp["bbox"], "pts": comp["pts"], "side": side, "color": color})
        frame_piece_ids.add(int(comp["cid"]))

    assert frame_pieces, "expected at least one frame piece"

    parent = list(range(len(frame_pieces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(frame_pieces)):
        for j in range(i + 1, len(frame_pieces)):
            pi = frame_pieces[i]
            pj = frame_pieces[j]
            if pi["side"] != pj["side"] or pi["color"] != pj["color"]:
                continue
            if _bbox_gap(pi["bbox"], pj["bbox"]) <= 2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(frame_pieces)):
        groups.setdefault(find(i), []).append(i)

    anchor_groups = []
    fixed_mask = np.zeros((H, W), dtype=bool)
    for idxs in groups.values():
        pts = [tuple(map(int, p)) for i in idxs for p in frame_pieces[i]["pts"]]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True
            fixed_mask[r, c] = True
        side = frame_pieces[idxs[0]]["side"]
        sock = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(g[r, c]) != int(bg):
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        sock[r, c] = True
                        break
        cavity = _extract_cavity(mask, g, int(bg), bbox)
        assert cavity.any(), "every frame must expose one cavity"
        cav_pts = np.argwhere(cavity)
        cr0, cc0 = cav_pts.min(axis=0)
        cr1, cc1 = cav_pts.max(axis=0)
        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "socket_mask": sock,
                "cavity_mask": cavity,
                "cavity_bbox": (int(cr0), int(cc0), int(cr1), int(cc1)),
                "cavity_size": int(cavity.sum()),
                "frame_side": side,
                "open_dir": _open_dir_for_side(side),
                "size": int(mask.sum()),
            }
        )

    anchor_groups.sort(key=lambda d: -d["size"])

    mov_comps = []
    for comp in raw_components:
        if int(comp["cid"]) in frame_piece_ids:
            continue
        if len(comp["colors"]) != 2:
            continue
        mov_comps.append(
            {
                "bbox": comp["bbox"],
                "mask": comp["mask_local"],
                "patch": comp["patch"],
                "size": comp["size"],
                "touch_sides": comp["touch_sides"],
            }
        )
    mov_comps.sort(key=lambda d: -d["size"])

    assert mov_comps, "expected at least one movable tool"
    assert len(mov_comps) == len(anchor_groups), "generator promises one tool per frame"

    comp_color_counts: list[dict[int, int]] = []
    for comp in mov_comps:
        vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]
        comp_color_counts.append(dict(Counter(vals)))

    def compatible(ci: int, gi: int) -> bool:
        cav_sz = int(anchor_groups[gi]["cavity_size"])
        return any(int(v) == cav_sz for v in comp_color_counts[ci].values())

    def opposite_match(ci: int, gi: int) -> int:
        sides = mov_comps[ci]["touch_sides"]
        if len(sides) != 1:
            return 0
        tool_side = next(iter(sides))
        return int(tool_side == _opposite_side(anchor_groups[gi]["frame_side"]))

    best_perm = None
    best_score = None
    for perm in itertools.permutations(range(len(anchor_groups))):
        ok = True
        opp_hits = 0
        dist_sum = 0
        for ci, gi in enumerate(perm):
            if not compatible(ci, gi):
                ok = False
                break
            opp_hits += opposite_match(ci, gi)
            dist_sum += _bbox_gap(mov_comps[ci]["bbox"], anchor_groups[gi]["bbox"])
        if not ok:
            continue
        score = (len(perm), opp_hits, -dist_sum)
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None, "must find a compatible tool-to-frame assignment"
    group_for_comp = [int(x) for x in best_perm]

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    for comp, gi in zip(mov_comps, group_for_comp):
        grp = anchor_groups[gi]
        placed_grp = placed_groups[gi]
        sock = grp["socket_mask"]
        cav = grp["cavity_mask"]
        odr, odc = grp["open_dir"]
        comp_vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]
        comp_counts = Counter(comp_vals)
        cav_sz = int(grp["cavity_size"])

        best_state = None
        best_score = None
        for p, m, is_flip in _gen_variants(comp["patch"], comp["mask"]):
            h, w = m.shape
            coords = np.argwhere(m)
            cols_here = sorted({int(x) for x in p[m].flatten() if int(x) != 0})
            plug_colors = [int(col) for col in cols_here if int(((p == int(col)) & m).sum()) == cav_sz]
            if not plug_colors:
                continue
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
                        if sock[r, c]:
                            socket_hits += 1
                        if cav[r, c]:
                            cavity_hits += 1
                        br0, bc0, br1, bc1 = grp["bbox"]
                        if odc == -1 and c < bc0:
                            outside += 1
                        elif odc == 1 and c > bc1:
                            outside += 1
                        elif odr == -1 and r < br0:
                            outside += 1
                        elif odr == 1 and r > br1:
                            outside += 1

                    plug_best = None
                    for plug_col in plug_colors:
                        plug_hits = 0
                        plug_outside = 0
                        nonplug_in_cav = 0
                        for rr, cc in coords:
                            r = int(r0 + rr)
                            c = int(c0 + cc)
                            in_cav = bool(cav[r, c])
                            col = int(p[rr, cc])
                            if col == int(plug_col):
                                if in_cav:
                                    plug_hits += 1
                                else:
                                    plug_outside += 1
                            elif in_cav:
                                nonplug_in_cav += 1
                        plug_full = int(plug_hits == cav_sz and plug_outside == 0 and nonplug_in_cav == 0)
                        plug_score = (int(plug_full), int(plug_hits), -int(nonplug_in_cav), -int(plug_outside))
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
                        best_state = (p, m, int(r0), int(c0), bool(is_flip))

        assert best_state is not None, "must find a valid placement for every tool"
        p, m, r0, c0, is_flip = best_state
        if is_flip and len(comp_counts) == 2 and len(set(comp_counts.values())) == 2:
            a, b = sorted(comp_counts.keys())
            q = p.copy()
            q[(p == a) & m] = b
            q[(p == b) & m] = a
            p = q
        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True
            placed_grp[r, c] = True

    return out


# --------------- generator ---------------

# Canvas.
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (26, 30)
# Slightly favor 2-pair scenes so the opposite-frame D8/2 signal is visible in a small training set,
# while still keeping the 1-pair branch active.
NUM_PAIRS_CHOICES = [1, 2, 2]

# Frame geometry.
FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)

# Tool sizing.
HANDLE_AREA_RANGE = (4, 10)
TOOL_TOTAL_BOUNDS = (10, 15)
HANDLE_SPRITE_DIM = (2, 5)

# Placement and retries.
UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
PAIR_SPEC_RETRIES = 300
HANDLE_SPRITE_RETRIES = 30
GEN_ATTEMPT_RETRIES = 300


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    if flip:
        out = np.fliplr(out)
    return out


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


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == a
    mb = out == b
    out[ma] = b
    out[mb] = a
    return out


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
        if int(np.sum(sprite != bg)) == handle_area:
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
    frame = np.full((fh, fw), bg, dtype=int)
    frame[:top_h, :] = frame_color
    frame[top_h + 1 :, :] = frame_color
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = bg

    cavity = frame == bg
    attach_rows = {r for r in range(fh) if cavity[r, fw - 1]}

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None

    hs_h, hs_w = handle_sprite.shape
    sprite_left_rows = {r for r in range(hs_h) if handle_sprite[r, 0] != bg}
    valid_offsets = [voff for voff in range(fh) if any((voff + sr) in attach_rows for sr in sprite_left_rows)]
    if not valid_offsets:
        return None

    voff = random.choice(valid_offsets)
    total_h = max(fh, voff + hs_h)
    total_w = fw + hs_w

    frame_full = np.full((total_h, total_w), bg, dtype=int)
    frame_full[:fh, :fw] = frame

    tool = np.full((total_h, total_w), bg, dtype=int)
    tool[:fh, :fw][cavity] = plug_color
    handle_mask = handle_sprite != bg
    tool[voff : voff + hs_h, fw : fw + hs_w][handle_mask] = handle_color
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
        for _j in range(num_pairs):
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
            used_plug_areas.add(plug_area)
        if len(specs) != num_pairs:
            continue
        all_plug_areas = {s["plug_area"] for s in specs}
        ok = True
        for spec in specs:
            pa = spec["plug_area"]
            handle_choices = [
                h for h in range(ha_lo, ha_hi) if tt_lo <= pa + h <= tt_hi and h != pa and h not in all_plug_areas
            ]
            if not handle_choices:
                ok = False
                break
            spec["handle_area"] = random.choice(handle_choices)
        if ok:
            return [dict(s) for s in specs]
    return []


def _candidate_frame_origins(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
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


def _candidate_tool_origins(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "left":
        origins = [(r0, 0) for r0 in range(1, H - ph)]
    elif side == "right":
        origins = [(r0, W - pw) for r0 in range(1, H - ph)]
    elif side == "top":
        origins = [(0, c0) for c0 in range(1, W - pw)]
    elif side == "bottom":
        origins = [(H - ph, c0) for c0 in range(1, W - pw)]
    else:
        raise ValueError(side)
    random.shuffle(origins)
    return origins


def _variant_choices(patch: np.ndarray, bg: int) -> list[tuple[np.ndarray, bool]]:
    seen = set()
    out: list[tuple[np.ndarray, bool]] = []
    mask = patch != bg
    for k in range(4):
        for flip in (False, True):
            q = _transform(patch, k, flip)
            m = _transform(mask.astype(np.uint8), k, flip).astype(bool)
            key = (m.tobytes(), q[m].tobytes())
            if key in seen:
                continue
            seen.add(key)
            bbox = _bbox(m)
            assert bbox is not None
            r0, c0, r1, c1 = bbox
            out.append((q[r0 : r1 + 1, c0 : c1 + 1].copy(), flip))
    random.shuffle(out)
    return out


def _can_place_tool(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    mask = patch != bg
    if np.any((canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] != bg) & mask):
        return False
    for rr, cc in np.argwhere(mask):
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r2, c2 = gr + dr, gc + dc
            if not (0 <= r2 < canvas.shape[0] and 0 <= c2 < canvas.shape[1]):
                continue
            local_rr = r2 - r0
            local_cc = c2 - c0
            if 0 <= local_rr < patch.shape[0] and 0 <= local_cc < patch.shape[1] and mask[local_rr, local_cc]:
                continue
            if int(canvas[r2, c2]) != int(bg):
                return False
    return True


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
    union_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
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
    scene_left = np.where(tool_left != bg, tool_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool_scene = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    fr_bbox = _bbox(frame_scene != bg)
    tool_bbox = _bbox(tool_scene != bg)
    assert fr_bbox is not None and tool_bbox is not None
    fr0, fc0, fr1, fc1 = fr_bbox
    tr0, tc0, tr1, tc1 = tool_bbox

    origin = None
    for r0, c0 in _candidate_frame_origins(side, ph, pw, H, W):
        union_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        frame_bbox = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) <= UNION_BBOX_GAP_MIN - 1 for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) <= FRAME_BBOX_GAP_MIN - 1 for prev in frame_bboxes):
            continue
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != bg) & (scene != bg)):
            continue
        origin = (r0, c0)
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, scene, r0, c0, transparent=bg)
    paste(input_grid, frame_scene, r0, c0, transparent=bg)
    union_bboxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    frame_bboxes.append((r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1))

    tool_trim = tool_scene[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    return {
        "tool": tool_trim,
        "plug_color": plug_color,
        "handle_color": handle_color,
        "plug_area": spec["plug_area"],
        "handle_area": spec["handle_area"],
        "frame_side": side,
        "tool_side": _opposite_side(side),
        "frame_bbox": (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1),
    }


def _undock_tool_to_opposite_edge(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    for patch, is_flip in _variant_choices(info["tool"], bg):
        q = patch
        if is_flip and info["plug_area"] != info["handle_area"]:
            q = _swap_colors(patch, info["plug_color"], info["handle_color"])
        ph, pw = q.shape
        for r0, c0 in _candidate_tool_origins(info["tool_side"], ph, pw, input_grid.shape[0], input_grid.shape[1]):
            if not _can_place_tool(placement_canvas, q, r0, c0, bg):
                continue
            paste(input_grid, q, r0, c0, transparent=bg)
            paste(placement_canvas, q, r0, c0, transparent=bg)
            return True
    return False


def _choose_frame_sides(num_pairs: int) -> list[str]:
    if num_pairs == 1:
        return [random.choice(["left", "right", "top", "bottom"])]
    if random.choice([False, True]):
        sides = ["left", "right"]
    else:
        sides = ["top", "bottom"]
    random.shuffle(sides)
    return sides


def _analyze_combo_input(grid: np.ndarray) -> dict:
    bg = get_bg(grid)
    H, W = grid.shape
    fg = grid != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    tool_sides: list[str] = []
    frame_pieces: list[dict] = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        colors = {int(grid[r, c]) for r, c in pts}
        sides = _touch_sides(pts, H, W)
        if len(colors) == 2 and len(sides) == 1:
            tool_sides.append(next(iter(sides)))
        elif len(colors) == 1 and len(sides) == 1:
            frame_pieces.append(
                {
                    "bbox": (int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())),
                    "side": next(iter(sides)),
                    "color": next(iter(colors)),
                }
            )

    parent = list(range(len(frame_pieces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(frame_pieces)):
        for j in range(i + 1, len(frame_pieces)):
            pi = frame_pieces[i]
            pj = frame_pieces[j]
            if pi["side"] != pj["side"] or pi["color"] != pj["color"]:
                continue
            if _bbox_gap(pi["bbox"], pj["bbox"]) <= 2:
                union(i, j)

    grouped_frame_sides: list[str] = []
    seen_roots: set[int] = set()
    for i in range(len(frame_pieces)):
        root = find(i)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        grouped_frame_sides.append(frame_pieces[i]["side"])

    return {"tool_sides": tool_sides, "frame_piece_sides": [p["side"] for p in frame_pieces], "frame_sides": grouped_frame_sides}


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_pairs = random.choice(NUM_PAIRS_CHOICES)

    pair_specs = _pick_pair_specs(num_pairs)
    if len(pair_specs) != num_pairs:
        return None

    colors = [c for c in range(1, 10) if c != bg]
    random.shuffle(colors)
    if len(colors) < 3 * num_pairs:
        return None

    frame_sides = _choose_frame_sides(num_pairs)
    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict] = []

    for i, spec in enumerate(pair_specs):
        info = _place_docked_pair(
            i,
            spec,
            colors,
            frame_sides[i],
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

    placement_canvas = input_grid.copy()
    for info in placed_tools:
        if not _undock_tool_to_opposite_edge(info, bg, input_grid, placement_canvas):
            return None

    if len(placed_tools) == 2:
        if {placed_tools[0]["frame_side"], placed_tools[1]["frame_side"]} not in ({"left", "right"}, {"top", "bottom"}):
            return None

    summary = _analyze_combo_input(input_grid)
    if len(summary["tool_sides"]) != num_pairs or len(summary["frame_sides"]) != num_pairs:
        return None
    if sorted(summary["tool_sides"]) != sorted(_opposite_side(side) for side in summary["frame_sides"]):
        return None
    if num_pairs == 2 and set(summary["frame_sides"]) not in ({"left", "right"}, {"top", "bottom"}):
        return None

    if np.array_equal(main(input_grid), output):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
