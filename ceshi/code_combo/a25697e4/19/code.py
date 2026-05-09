# Combo variant of a25697e4 (index 19)
# combo_spec: D3/2+D13/0
# Name: 分离handle填充凹槽
#
# Design analysis
# a) Concrete behavior trace:
# - Original task: a single connected 2-color tool is moved so the area-matching color block fills
#   the frame cavity and the other color remains attached outside.
# - Fused variant: each tool is already split into two nearby disconnected monochrome components.
#   Example: if a frame cavity has area 6, the input tool might be a 6-cell red block and a 5-cell
#   blue block with a 1-cell gap between them. In the output, the 6-cell block must become the
#   cavity filler, while the 5-cell block is re-positioned outside the frame and still remains
#   disconnected from the filler. The key difference is both semantic and geometric:
#   "which color fills" is inverted, and "what counts as one tool" now requires grouping two
#   components before applying one joint D4 transform.
# b) Complexity calibration:
# - The generator keeps only 1-2 frame/tool pairs and makes each pair's cavity area distinct.
# - Tool parts stay close to each other (minimum Manhattan gap 2) while different tools are placed
#   with padding, so grouping is visible but not ambiguous.
# - Plug areas are chosen different from every cavity area, so the handle-match rule is the only
#   viable area cue.
# c) Hidden assumptions audit against the original solver:
# - "each movable object is one connected component" no longer holds, so movable pieces are first
#   clustered by exact component-to-component distance.
# - "the cavity-filling color can be treated as plug" no longer holds, so the scoring logic now
#   treats the area-matching color as handle and explicitly rewards the other color being outside.
# - "docked tool may become connected at the frame" is no longer valid, so the generator builds a
#   1-cell gap between handle and plug in the docked state; the solver preserves that by moving the
#   grouped union patch as a rigid D4 object.
# d) Placement legality:
# - All frame and tool placements use bbox-gap / transparent-paste checks, stay in-bounds, and the
#   undocked tool is placed as one patch so its two components cannot drift apart.
# e) Visual observability:
# - Across examples, the cavity-sized region is visibly the inner filler even though the tool is
#   disconnected in the input. The protruding region is the wrong area to fill the notch, making the
#   inversion from the original task visible to a human.
# f) Visual cleanliness:
# - Only three colors are used per pair (frame, handle, plug), tools stay separated from frames in
#   the input, and the docked output has only one layer of frame+tool without extra markers.
# g) New solver assumptions enforced by the generator:
# - Each tool cluster contains exactly two disconnected monochrome components.
# - The two parts of one tool are the only non-border components within Manhattan distance <= 3.
# - Exactly one tool color count equals the matched cavity area.

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox as _bbox, get_bg, paste, random_free_location, random_sprite


TOOL_GROUP_DIST_MAX = 3


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


def _min_component_distance(a_pts: list[tuple[int, int]], b_pts: list[tuple[int, int]]) -> int:
    best = None
    for ar, ac in a_pts:
        for br, bc in b_pts:
            d = abs(int(ar) - int(br)) + abs(int(ac) - int(bc))
            if best is None or d < best:
                best = d
    assert best is not None
    return int(best)


def _gen_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, bool]]:
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


def _extract_anchor_groups(g: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict]]:
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        pts_list = [(int(r), int(c)) for r, c in pts]
        if not any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts_list):
            continue
        bbox = (int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max()))
        fixed_components.append({"pts": pts_list, "bbox": bbox})
        for r, c in pts_list:
            fixed_mask[r, c] = True
    assert fixed_components

    parent = list(range(len(fixed_components)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

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

    anchor_groups: list[dict] = []
    for idxs in groups.values():
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

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

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_id = None
        best_key = None
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
                best_id = cid
        assert best_id is not None
        for rr, cc in np.argwhere(sub_lab == best_id):
            cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

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
                "socket_mask": sock,
                "cavity_mask": cavity,
                "cavity_size": int(cavity.sum()),
                "open_dir": open_dir,
                "size": int(mask.sum()),
            }
        )
    anchor_groups.sort(key=lambda d: -d["size"])
    return fixed_mask, anchor_groups


def _extract_tool_groups(g: np.ndarray, bg: int, fixed_mask: np.ndarray) -> list[dict]:
    mov_mask = (g != int(bg)) & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    comps: list[dict] = []
    for cid in range(1, mov_n + 1):
        pts = np.argwhere(mov_lab == cid)
        colors = {int(g[r, c]) for r, c in pts}
        bbox = (int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max()))
        comps.append(
            {
                "cid": cid,
                "pts": [(int(r), int(c)) for r, c in pts],
                "bbox": bbox,
                "size": int(len(pts)),
                "colors": colors,
            }
        )

    parent = list(range(len(comps)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if _min_component_distance(comps[i]["pts"], comps[j]["pts"]) <= TOOL_GROUP_DIST_MAX:
                union(i, j)

    grouped: dict[int, list[int]] = {}
    for i in range(len(comps)):
        grouped.setdefault(find(i), []).append(i)

    tools: list[dict] = []
    for idxs in grouped.values():
        pts = [p for i in idxs for p in comps[i]["pts"]]
        arr = np.array(pts, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=int)
        mask = np.zeros_like(patch, dtype=bool)
        for r, c in pts:
            patch[int(r - r0), int(c - c0)] = int(g[r, c])
            mask[int(r - r0), int(c - c0)] = True
        vals = [int(x) for x in patch[mask].tolist()]
        counts = Counter(vals)
        assert len(idxs) == 2
        assert len(counts) == 2
        tools.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch,
                "mask": mask,
                "size": int(mask.sum()),
                "color_counts": dict(counts),
            }
        )
    tools.sort(key=lambda d: -d["size"])
    return tools


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock split two-part tools into border frames by using the handle block as the cavity filler.

    Input structure:
    - The background is the most common color.
    - Border-touching components cluster into monochrome frames whose interior bg cavity forms a
      staircase socket.
    - Away from the border, each movable tool is represented by two nearby disconnected monochrome
      components that must be grouped into one rigid object before transforming it.

    Rule:
    - Match each grouped tool to the frame whose cavity size equals one of the tool's two color
      counts; that matching color is the handle, not the plug.
    - Rotate and/or reflect the grouped tool so the handle component fills the cavity exactly, while
      the other component protrudes outward and stays separated from the handle.
    - If the winning docking transform uses a reflection and the two component sizes differ, swap the
      tool's two colors after docking.

    Key insight: first infer which nearby disconnected parts belong to the same tool, then use the
    cavity-sized color block as the in-frame handle and treat the other block as the outward plug.
    """
    g = grid.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    fixed_mask, anchor_groups = _extract_anchor_groups(g, bg)
    mov_groups = _extract_tool_groups(g, bg, fixed_mask)

    def compatible(ci: int, gi: int) -> bool:
        cav_sz = int(anchor_groups[gi]["cavity_size"])
        return any(int(v) == cav_sz for v in mov_groups[ci]["color_counts"].values())

    if len(mov_groups) == len(anchor_groups) and len(anchor_groups) > 1:
        best_perm = None
        best_score = None
        for perm in itertools.permutations(range(len(anchor_groups))):
            ok = True
            dist_sum = 0
            for ci, gi in enumerate(perm):
                if not compatible(ci, gi):
                    ok = False
                    break
                dist_sum += _bbox_gap(mov_groups[ci]["bbox"], anchor_groups[gi]["bbox"])
            if not ok:
                continue
            score = (-int(dist_sum),)
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None
        group_for_comp = [int(x) for x in best_perm]
    else:
        group_for_comp = []
        for ci, comp in enumerate(mov_groups):
            best_g = None
            best_d = None
            for gi, grp in enumerate(anchor_groups):
                if not compatible(ci, gi):
                    continue
                d = _bbox_gap(comp["bbox"], grp["bbox"])
                if best_d is None or d < best_d:
                    best_d = d
                    best_g = gi
            assert best_g is not None
            group_for_comp.append(int(best_g))

    out = np.full_like(g, bg)
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    for comp, gi in zip(mov_groups, group_for_comp):
        grp = anchor_groups[gi]
        placed_grp = placed_groups[gi]
        cav = grp["cavity_mask"]
        sock = grp["socket_mask"]
        cav_sz = int(grp["cavity_size"])
        odr, odc = grp["open_dir"]
        br0, bc0, br1, bc1 = grp["bbox"]
        comp_counts = Counter(comp["color_counts"])

        best_state = None
        best_score = None
        for p, m, is_flip in _gen_variants(comp["patch"], comp["mask"]):
            coords = np.argwhere(m)
            h, w = m.shape
            colors_here = sorted({int(x) for x in p[m].flatten() if int(x) != 0})
            handle_colors = [int(col) for col in colors_here if int(((p == int(col)) & m).sum()) == cav_sz]
            if not handle_colors:
                continue
            for r0 in range(0, H - h + 1):
                for c0 in range(0, W - w + 1):
                    if any(placed_global[int(r0 + rr), int(c0 + cc)] for rr, cc in coords):
                        continue
                    total_adj = 0
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                total_adj += 1
                    if total_adj == 0:
                        continue

                    dist = _bbox_gap(grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
                    for handle_col in handle_colors:
                        handle_hits = 0
                        handle_outside = 0
                        handle_adj = 0
                        plug_in_cav = 0
                        plug_outward = 0
                        plug_not_outward = 0
                        socket_hits = 0
                        for rr, cc in coords:
                            r = int(r0 + rr)
                            c = int(c0 + cc)
                            col = int(p[rr, cc])
                            in_cav = bool(cav[r, c])
                            outward = (
                                (odc == -1 and c < bc0)
                                or (odc == 1 and c > bc1)
                                or (odr == -1 and r < br0)
                                or (odr == 1 and r > br1)
                            )
                            if col == handle_col:
                                if in_cav:
                                    handle_hits += 1
                                else:
                                    handle_outside += 1
                                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    r2, c2 = r + dr, c + dc
                                    if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                        handle_adj += 1
                                if sock[r, c]:
                                    socket_hits += 1
                            else:
                                if in_cav:
                                    plug_in_cav += 1
                                elif outward:
                                    plug_outward += 1
                                else:
                                    plug_not_outward += 1
                        handle_full = int(handle_hits == cav_sz and handle_outside == 0 and plug_in_cav == 0)
                        score = (
                            int(handle_full),
                            int(handle_hits),
                            -int(handle_outside),
                            -int(plug_in_cav),
                            int(plug_outward),
                            -int(plug_not_outward),
                            int(handle_adj),
                            int(socket_hits),
                            int(total_adj),
                            -int(dist),
                            -int(r0),
                            -int(c0),
                        )
                        if best_score is None or score > best_score:
                            best_score = score
                            best_state = (p, m, int(r0), int(c0), bool(is_flip))

        assert best_state is not None
        p, m, r0, c0, is_flip = best_state
        if is_flip and len(comp_counts) == 2 and len(set(comp_counts.values())) == 2:
            a, b = sorted(comp_counts.keys())
            p = _swap_colors(p, a, b)
        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True
            placed_grp[r, c] = True

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [1, 2]

FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
CAVITY_AREA_BOUNDS = (5, 8)

PART_AREA_RANGE = (4, 10)
TOOL_TOTAL_BOUNDS = (10, 15)
PART_SPRITE_DIM = (2, 5)

UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4

PAIR_SPEC_RETRIES = 300
SPRITE_RETRIES = 30
GEN_ATTEMPT_RETRIES = 300


def _make_connected_sprite(area: int, bg: int, color: int) -> np.ndarray | None:
    lo, hi = PART_SPRITE_DIM
    dims = [(h, w) for h in range(lo, hi) for w in range(lo, hi) if area <= h * w <= area * 2]
    if not dims:
        return None
    for _ in range(SPRITE_RETRIES):
        h, w = random.choice(dims)
        sprite = random_sprite(
            h,
            w,
            density=area / (h * w),
            color_palette=[color],
            connectivity=4,
            background=bg,
        )
        if int(np.sum(sprite != bg)) != area:
            continue
        fg = sprite != bg
        lab, n = label(fg, structure=STRUCT4)
        if int(n) != 1:
            continue
        bb = _bbox(fg)
        assert bb is not None
        r0, c0, r1, c1 = bb
        return sprite[r0 : r1 + 1, c0 : c1 + 1].copy()
    return None


def _build_param_options() -> list[tuple[int, int, int, int]]:
    fw_lo, fw_hi = FRAME_W_RANGE
    nn_lo, nn_hi = NOTCH_N_RANGE
    th_lo, th_hi = TOP_H_RANGE
    bh_lo, bh_hi = BOT_H_RANGE
    ca_lo, ca_hi = CAVITY_AREA_BOUNDS
    return [
        (fw, top_h, bot_h, notch_n)
        for fw in range(fw_lo, fw_hi)
        for notch_n in range(nn_lo, nn_hi)
        if ca_lo <= fw + notch_n <= ca_hi
        for top_h in range(th_lo, th_hi)
        for bot_h in range(bh_lo, bh_hi)
    ]


PARAM_OPTIONS = _build_param_options()


def _pick_pair_specs(num_pairs: int) -> list[dict]:
    pa_lo, pa_hi = PART_AREA_RANGE
    tt_lo, tt_hi = TOOL_TOTAL_BOUNDS
    specs: list[dict] = []
    for _ in range(PAIR_SPEC_RETRIES):
        specs.clear()
        used_cavity_areas: set[int] = set()
        for _pair in range(num_pairs):
            candidates = [x for x in PARAM_OPTIONS if x[0] + x[3] not in used_cavity_areas]
            if not candidates:
                specs = []
                break
            fw, top_h, bot_h, notch_n = random.choice(candidates)
            cavity_area = fw + notch_n
            specs.append(
                {
                    "fw": fw,
                    "top_h": top_h,
                    "bot_h": bot_h,
                    "notch_n": notch_n,
                    "handle_area": cavity_area,
                }
            )
            used_cavity_areas.add(cavity_area)
        if len(specs) != num_pairs:
            continue
        all_handle_areas = {s["handle_area"] for s in specs}
        ok = True
        for spec in specs:
            ha = spec["handle_area"]
            plug_choices = [
                p
                for p in range(pa_lo, pa_hi)
                if tt_lo <= ha + p <= tt_hi and p != ha and p not in all_handle_areas
            ]
            if not plug_choices:
                ok = False
                break
            spec["plug_area"] = random.choice(plug_choices)
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
            out.append((q, flip))
    random.shuffle(out)
    return out


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
    plug_area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), bg, dtype=int)
    frame[:top_h, :] = frame_color
    frame[top_h + 1 :, :] = frame_color
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = bg

    cavity = frame == bg
    plug_sprite = _make_connected_sprite(plug_area, bg, plug_color)
    if plug_sprite is None:
        return None

    cavity_rows = [int(r) for r in np.unique(np.argwhere(cavity)[:, 0])]
    ps_h, ps_w = plug_sprite.shape
    sprite_rows = list(range(ps_h))
    valid_offsets = []
    for voff in range(0, fh + 1):
        placed_rows = {voff + sr for sr in sprite_rows}
        if any(abs(pr - cr) <= 1 for pr in placed_rows for cr in cavity_rows):
            valid_offsets.append(voff)
    if not valid_offsets:
        return None

    voff = random.choice(valid_offsets)
    total_h = max(fh, voff + ps_h)
    total_w = fw + 1 + ps_w

    frame_full = np.full((total_h, total_w), bg, dtype=int)
    frame_full[:fh, :fw] = frame

    tool = np.full((total_h, total_w), bg, dtype=int)
    tool[:fh, :fw][cavity] = handle_color
    plug_mask = plug_sprite != bg
    tool[voff : voff + ps_h, fw + 1 : fw + 1 + ps_w][plug_mask] = plug_color
    return frame_full, tool


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
    handle_color = colors[3 * i + 1]
    plug_color = colors[3 * i + 2]

    result = _build_left_pair(
        bg=bg,
        frame_color=frame_color,
        plug_color=plug_color,
        handle_color=handle_color,
        fw=spec["fw"],
        top_h=spec["top_h"],
        bot_h=spec["bot_h"],
        notch_n=spec["notch_n"],
        plug_area=spec["plug_area"],
    )
    if result is None:
        return None
    frame_left, tool_left = result
    scene_left = np.where(tool_left != bg, tool_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    fr0, fc0, fr1, fc1 = _bbox(frame_scene != bg)

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
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
    return {
        "tool": tool,
        "plug_color": plug_color,
        "handle_color": handle_color,
        "plug_area": spec["plug_area"],
        "handle_area": spec["handle_area"],
    }


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    for patch, is_flip in _variant_choices(info["tool"]):
        if is_flip and info["plug_area"] != info["handle_area"]:
            patch = _swap_colors(patch, info["plug_color"], info["handle_color"])
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
        paste(placement_canvas, patch, r0, c0, transparent=bg)
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
        if not _undock_tool(info, bg, input_grid, placement_canvas):
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
