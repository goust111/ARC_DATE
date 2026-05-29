# Combo variant of a25697e4 (index 30)
# combo_spec: D3/2+D7/0
# Name: 分离工具复制对接
#
# Design analysis for the fused variant:
# a) Concrete behavior trace:
#    - Original: each free tool is one 4-connected 2-color component; docking removes it from
#      the source area and leaves a single docked copy near the border frame.
#    - Fused: each logical tool is split into two nearby disconnected monochrome pieces:
#      a staircase plug block and a handle block. They still undergo one shared D4 transform for
#      docking, but the source pieces stay in place and a second copy appears at the frame.
#      Output therefore shows four visible blocks per pair: source plug, source handle, docked
#      plug, docked handle.
# b) Complexity calibration:
#    - The generator keeps only 1-2 frame/tool pairs, as in the base task.
#    - Each tool still has unique plug area matching exactly one frame cavity, so pairing remains
#      learnable and non-ambiguous.
#    - Separation is local inside one tool bbox (typically one empty row/col between the two
#      pieces), while different tools are placed with larger padding. This makes "group the close
#      pair first, then dock it" visually salient.
# c) Hidden assumptions broken by the combo:
#    - The base solver assumed one movable connected component per tool. That is false here, so
#      movable pieces are first clustered by small bbox gap into 2-piece groups.
#    - The base solver started from an empty output plus fixed frames. That would wrongly erase the
#      source tool; here the output starts from the full input grid and only adds docked copies.
#    - Placement/matching still assumes one color-count equals cavity area. The generator enforces
#      plug_area != handle_area and unique plug areas across pairs so the assumption remains valid.
# d) Placement legality:
#    - Frames are border-anchored and mutually separated.
#    - Docked copies are written only into background cells around their target frame.
#    - Source tool variants are placed with padding against frames, other source tools, and already
#      present docked copies, so input and output remain non-overlapping and in bounds.
# e) Visual observability:
#    - A human sees two nearby loose pieces in the input and the same two pieces still present in
#      the output, plus a second transformed pair attached to the frame. The change from "move" to
#      "copy" is directly visible, and the separation of plug/handle is visible before and after.
# f) Visual tidiness:
#    - Only one transparent gap is introduced inside each tool; no extra markers are added.
#    - Different logical objects stay spatially separated, so the four blocks per pair are easy to
#      individuate.
# g) Inferability of color changes:
#    - The original reflection-triggered color swap is preserved, but only when the two color
#      counts differ. Because the plug shape is the only region whose size matches a cavity, the
#      structural cue for which region is the plug remains explicit even if colors swap.

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, paste, random_free_location, random_sprite
from arc_utils import bbox as _bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy separated two-piece tools onto border frames by fitting the plug into each cavity.

    Input structure:
    - The background is the most frequent color.
    - Border-touching same-color pieces cluster into frames that enclose staircase cavities.
    - Away from the border, each logical tool is split into two nearby disconnected components:
      one monochrome plug piece and one monochrome handle piece.

    Rule:
    - Pair the two nearby movable pieces into one tool, then match that tool to the frame whose
      cavity area equals exactly one of the tool's color-region sizes.
    - Apply one rotation/flip to the whole two-piece layout so the plug lands on the cavity and the
      handle lands outside the frame. If the winning docking transform includes a reflection and
      the two piece sizes differ, swap the two colors in the docked copy.
    - Keep the original source pieces in place and add the docked copy, rather than moving them.

    Key insight: proximity groups the disconnected plug+handle into one tool; cavity size picks the
    target frame; the output is the input plus one transformed duplicate per grouped tool.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    def bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        (ar0, ac0, ar1, ac1) = a
        (br0, bc0, br1, bc1) = b

        def gap1(x0: int, x1: int, y0: int, y1: int) -> int:
            if y1 < x0:
                return int(x0 - y1)
            if x1 < y0:
                return int(y0 - x1)
            return 0

        return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)

    def gen_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, bool]]:
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

    def close_parts(part_a: dict, part_b: dict) -> bool:
        if bbox_gap(part_a["bbox"], part_b["bbox"]) > 2:
            return False
        best = None
        for r0, c0 in part_a["pts"]:
            for r1, c1 in part_b["pts"]:
                d = abs(int(r0) - int(r1)) + abs(int(c0) - int(c1))
                if best is None or d < best:
                    best = d
                    if best <= 2:
                        return True
        return False

    fg = g != bg
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components: list[dict] = []
    movable_parts: list[dict] = []

    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        pts_list = [(int(r), int(c)) for r, c in pts]
        bbox = (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        )
        touches = any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts_list)
        if touches:
            fixed_components.append({"pts": pts_list, "bbox": bbox})
            for r, c in pts_list:
                fixed_mask[r, c] = True
            continue

        colors_here = sorted({int(g[r, c]) for r, c in pts_list})
        assert len(colors_here) == 1, "each separated tool piece should be monochrome"
        movable_parts.append({"pts": pts_list, "bbox": bbox, "color": int(colors_here[0]), "size": int(len(pts))})

    assert fixed_components, "at least one border frame is required"

    parent = list(range(len(fixed_components)))

    def find(parent_list: list[int], x: int) -> int:
        while parent_list[x] != x:
            parent_list[x] = parent_list[parent_list[x]]
            x = parent_list[x]
        return x

    def union(parent_list: list[int], a: int, b: int) -> None:
        ra, rb = find(parent_list, a), find(parent_list, b)
        if ra != rb:
            parent_list[rb] = ra

    # Cluster border-touching pieces into one frame when their bboxes nearly touch.
    for i in range(len(fixed_components)):
        for j in range(i + 1, len(fixed_components)):
            if bbox_gap(fixed_components[i]["bbox"], fixed_components[j]["bbox"]) <= 2:
                union(parent, i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(fixed_components)):
        groups.setdefault(find(parent, i), []).append(i)

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
                if int(g[r, c]) != bg:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        socket_mask[r, c] = True
                        break

        cavity_mask = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == bg
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_key = None
        best_id = None
        for cid in range(1, sub_n + 1):
            ptsc = np.argwhere(sub_lab == cid)
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
            key = (area, -cid)
            if best_key is None or key > best_key:
                best_key = key
                best_id = int(cid)
        assert best_id is not None, "every frame must expose a cavity"
        for rr, cc in np.argwhere(sub_lab == best_id):
            cavity_mask[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        dir_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dir_counts = [0, 0, 0, 0]
        for r, c in pts:
            for di, (dr, dc) in enumerate(dir_list):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == bg:
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

    # Group nearby disconnected internal pieces into one logical tool.
    mov_parent = list(range(len(movable_parts)))
    for i in range(len(movable_parts)):
        for j in range(i + 1, len(movable_parts)):
            if close_parts(movable_parts[i], movable_parts[j]):
                union(mov_parent, i, j)

    mov_groups: dict[int, list[int]] = {}
    for i in range(len(movable_parts)):
        mov_groups.setdefault(find(mov_parent, i), []).append(i)

    movable_tools = []
    for idxs in mov_groups.values():
        parts = [movable_parts[i] for i in idxs]
        colors = {int(p["color"]) for p in parts}
        assert len(parts) == 2 and len(colors) == 2, "each logical tool should have exactly two nearby pieces"
        rs = [r for p in parts for r, _c in p["pts"]]
        cs = [c for p in parts for _r, c in p["pts"]]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        mask = np.zeros_like(patch, dtype=bool)
        color_counts: dict[int, int] = {}
        for p in parts:
            color_counts[int(p["color"])] = int(p["size"])
            for r, c in p["pts"]:
                patch[r - r0, c - c0] = int(p["color"])
                mask[r - r0, c - c0] = True
        movable_tools.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch,
                "mask": mask,
                "size": int(mask.sum()),
                "color_counts": color_counts,
            }
        )
    movable_tools.sort(key=lambda d: -int(d["size"]))

    def compatible(tool_index: int, group_index: int) -> bool:
        cav_sz = int(anchor_groups[group_index]["cavity_size"])
        return any(int(v) == cav_sz for v in movable_tools[tool_index]["color_counts"].values())

    if len(movable_tools) == len(anchor_groups) and len(anchor_groups) > 1:
        best_perm = None
        best_score = None
        for perm in itertools.permutations(range(len(anchor_groups))):
            ok = True
            dist_sum = 0
            for ti, gi in enumerate(perm):
                if not compatible(ti, gi):
                    ok = False
                    break
                dist_sum += bbox_gap(movable_tools[ti]["bbox"], anchor_groups[gi]["bbox"])
            if not ok:
                continue
            score = -int(dist_sum)
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None, "must find a compatible tool-to-frame matching"
        group_for_tool = [int(x) for x in best_perm]
    else:
        group_for_tool = []
        for tool in movable_tools:
            best_g = None
            best_d = None
            for gi, grp in enumerate(anchor_groups):
                if not compatible(len(group_for_tool), gi):
                    continue
                d = bbox_gap(tool["bbox"], grp["bbox"])
                if best_d is None or d < best_d:
                    best_d = d
                    best_g = gi
            assert best_g is not None, "every tool must match some cavity size"
            group_for_tool.append(int(best_g))

    out = g.copy()
    placed_global = fg.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    for tool, gi in zip(movable_tools, group_for_tool):
        grp = anchor_groups[gi]
        cav = grp["cavity_mask"]
        sock = grp["socket_mask"]
        placed_grp = placed_groups[gi]
        cav_sz = int(grp["cavity_size"])
        odr, odc = grp["open_dir"]
        br0, bc0, br1, bc1 = grp["bbox"]

        best_state = None
        best_score = None
        for p, m, is_flip in gen_variants(tool["patch"], tool["mask"]):
            h, w = m.shape
            coords = np.argwhere(m)
            colors_here = sorted({int(x) for x in p[m].flatten() if int(x) != 0})
            plug_colors = [int(col) for col in colors_here if int(((p == int(col)) & m).sum()) == cav_sz]
            assert len(plug_colors) == 1, "unique cavity-sized plug color expected"
            plug_col = int(plug_colors[0])
            for r0 in range(0, H - h + 1):
                for c0 in range(0, W - w + 1):
                    if any(bool(placed_global[int(r0 + rr), int(c0 + cc)]) for rr, cc in coords):
                        continue

                    adj = 0
                    socket_hits = 0
                    cavity_hits = 0
                    outside = 0
                    plug_hits = 0
                    plug_outside = 0
                    nonplug_in_cavity = 0
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        col = int(p[rr, cc])
                        in_cavity = bool(cav[r, c])
                        if sock[r, c]:
                            socket_hits += 1
                        if in_cavity:
                            cavity_hits += 1
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                adj += 1
                        if odc == -1 and c < bc0:
                            outside += 1
                        elif odc == 1 and c > bc1:
                            outside += 1
                        elif odr == -1 and r < br0:
                            outside += 1
                        elif odr == 1 and r > br1:
                            outside += 1

                        if col == plug_col:
                            if in_cavity:
                                plug_hits += 1
                            else:
                                plug_outside += 1
                        elif in_cavity:
                            nonplug_in_cavity += 1

                    if adj == 0:
                        continue
                    plug_full = int(plug_hits == cav_sz and plug_outside == 0 and nonplug_in_cavity == 0)
                    dist = bbox_gap(grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
                    score = (
                        int(plug_full),
                        int(plug_hits),
                        -int(nonplug_in_cavity),
                        -int(plug_outside),
                        int(cavity_hits),
                        int(outside),
                        int(socket_hits),
                        int(adj),
                        -int(dist),
                        -int(r0),
                        -int(c0),
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_state = (p, m, int(r0), int(c0), bool(is_flip))

        assert best_state is not None, "must find a docked placement for every separated tool"
        p, m, r0, c0, is_flip = best_state
        color_counts = tool["color_counts"]
        if is_flip and len(color_counts) == 2 and len(set(color_counts.values())) == 2:
            a, b = sorted(color_counts.keys())
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

# Canvas
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [1, 2]

# Frame geometry
FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)

# Tool sizing
HANDLE_AREA_RANGE = (4, 10)
TOOL_TOTAL_BOUNDS = (10, 15)
HANDLE_SPRITE_DIM = (2, 5)
HANDLE_GAP = 1

# Placement
UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 4
TOOL_PADDING_CONN = 4

# Retries
PAIR_SPEC_RETRIES = 300
HANDLE_SPRITE_RETRIES = 30
GEN_ATTEMPT_RETRIES = 300


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return x0 - y1
        if x1 < y0:
            return y0 - x1
        return 0

    return _gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1)


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
    ma, mb = out == a, out == b
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
    dims = [
        (h, w)
        for h in range(lo, hi)
        for w in range(lo, hi)
        if handle_area <= h * w <= handle_area * 2
    ]
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
    attach_rows = {int(r) for r in range(fh) if bool(cavity[r, fw - 1])}

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None
    hs_h, hs_w = handle_sprite.shape
    sprite_left_rows = {int(r) for r in range(hs_h) if bool(handle_sprite[r, 0] != bg)}
    valid_offsets = [
        voff
        for voff in range(fh)
        if any((int(voff) + sr) in attach_rows for sr in sprite_left_rows)
    ]
    if not valid_offsets:
        return None
    voff = int(random.choice(valid_offsets))

    total_h = max(fh, voff + hs_h)
    total_w = fw + HANDLE_GAP + hs_w

    frame_full = np.full((total_h, total_w), bg, dtype=int)
    frame_full[:fh, :fw] = frame

    tool = np.full((total_h, total_w), bg, dtype=int)
    tool[:fh, :fw][cavity] = plug_color
    handle_mask = handle_sprite != bg
    start_c = fw + HANDLE_GAP
    tool[voff : voff + hs_h, start_c : start_c + hs_w][handle_mask] = handle_color

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
    for _ in range(PAIR_SPEC_RETRIES):
        specs: list[dict] = []
        used_plug_areas: set[int] = set()
        for _pair in range(num_pairs):
            candidates = [x for x in PARAM_OPTIONS if x[0] + x[3] not in used_plug_areas]
            if not candidates:
                specs = []
                break
            fw, top_h, bot_h, notch_n = random.choice(candidates)
            plug_area = int(fw + notch_n)
            specs.append(
                {
                    "fw": int(fw),
                    "top_h": int(top_h),
                    "bot_h": int(bot_h),
                    "notch_n": int(notch_n),
                    "plug_area": plug_area,
                }
            )
            used_plug_areas.add(plug_area)
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
            spec["handle_area"] = int(random.choice(handle_choices))
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


def _variant_choices(patch: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    seen = set()
    out: list[tuple[np.ndarray, bool]] = []
    mask = patch != 0
    for k in range(4):
        for flip in (False, True):
            q = _transform(patch, k, flip)
            m = _transform(mask.astype(np.uint8), k, flip).astype(bool)
            key = (m.tobytes(), q[m].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((q, bool(flip)))
    random.shuffle(out)
    return out


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
    frame_color = int(colors[3 * i])
    plug_color = int(colors[3 * i + 1])
    handle_color = int(colors[3 * i + 2])

    result = _build_left_pair(
        bg=bg,
        frame_color=frame_color,
        plug_color=plug_color,
        handle_color=handle_color,
        fw=int(spec["fw"]),
        top_h=int(spec["top_h"]),
        bot_h=int(spec["bot_h"]),
        notch_n=int(spec["notch_n"]),
        handle_area=int(spec["handle_area"]),
    )
    if result is None:
        return None
    frame_left, tool_left = result
    scene_left = np.where(tool_left != bg, tool_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    frame_bbox_local = _bbox(frame_scene != bg)
    assert frame_bbox_local is not None
    fr0, fc0, fr1, fc1 = frame_bbox_local

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
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
        origin = (int(r0), int(c0))
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, scene, r0, c0, transparent=bg)
    paste(input_grid, frame_scene, r0, c0, transparent=bg)
    union_bboxes.append((int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)))
    frame_bboxes.append((int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1)))
    return {
        "tool": tool,
        "plug_color": plug_color,
        "handle_color": handle_color,
        "plug_area": int(spec["plug_area"]),
        "handle_area": int(spec["handle_area"]),
    }


def _place_source_tool(info: dict, bg: int, input_grid: np.ndarray, output: np.ndarray, placement_canvas: np.ndarray) -> bool:
    for patch, is_flip in _variant_choices(info["tool"]):
        if is_flip and int(info["plug_area"]) != int(info["handle_area"]):
            patch = _swap_colors(patch, int(info["plug_color"]), int(info["handle_color"]))
        try:
            r0, c0 = random_free_location(
                placement_canvas,
                patch,
                transparent=bg,
                border=TOOL_BORDER,
                padding=TOOL_PADDING,
                padding_connectivity=TOOL_PADDING_CONN,
            )
        except ValueError:
            continue
        paste(input_grid, patch, r0, c0, transparent=bg)
        paste(output, patch, r0, c0, transparent=bg)
        paste(placement_canvas, patch, r0, c0, transparent=bg)
        return True
    return False


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = int(random.choice(list(range(10))))
    num_pairs = int(random.choice(NUM_PAIRS_CHOICES))

    pair_specs = _pick_pair_specs(num_pairs)
    if len(pair_specs) != num_pairs:
        return None

    colors = [c for c in range(1, 10) if c != bg]
    random.shuffle(colors)
    if len(colors) < 3 * num_pairs:
        return None

    sides = random.sample(["left", "right", "top", "bottom"], k=num_pairs)
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
    for info in placed_tools:
        if not _place_source_tool(info, bg, input_grid, output, placement_canvas):
            return None

    return input_grid, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        result = _gen_attempt()
        if result is None:
            continue
        input_grid, output = result
        try:
            if np.array_equal(main(input_grid), output):
                return input_grid
        except Exception:
            continue
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
