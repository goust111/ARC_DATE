# Combo variant of a25697e4 (index 1)
# combo_spec: D1/0+D13/0
# Name: 颜色配对角色互换
#
# Design analysis:
# a) Concrete behavior trace:
#    - Canonical left-border pair uses a 6-cell staircase cavity and a 13-cell tool split into
#      a 6-cell staircase region plus a 7-cell protruding region.
#    - In the original task, the cavity-sized region is the plug and it decides both pairing and
#      insertion. Here pairing is decoupled from geometry: both frames share the same cavity
#      geometry, both tools share the same overall shape, and the only unique pairing cue is that
#      exactly one tool color equals one frame color.
#    - Docking is also inverted: the 6-cell region goes into the cavity, while the frame-colored
#      7-cell region stays outside as the protrusion. Reflection still triggers a color swap when
#      the two region sizes differ (6 vs 7 here).
# b) Complexity calibration:
#    - Two opposite-border frames are generated in every sample. Their cavities are identical, so
#      shape/area alone cannot choose which tool belongs to which frame.
#    - Tools are deliberately placed nearer the wrong frame than the right one, so a distance-based
#      fallback would cross-pair them. This keeps the D1 color-pairing change active.
# c) Hidden assumptions changed from the original solver:
#    - Matching can no longer assume "some tool color count == cavity size" identifies the target
#      frame; that only identifies which region fills the cavity after pairing is already known.
#    - The frame-colored region is not the cavity filler anymore; the cavity-sized region is.
#    - Because input tools may come from reflected undocking, visible colors can be swapped, so the
#      solver must treat color-set matching and area-role matching as separate steps.
# d) Placement legality:
#    - Output is built first with non-overlapping border scenes. Input is derived by removing the
#      docked tools and re-placing free tools strictly in-bounds, non-overlapping, non-border, and
#      with a minimum bbox gap from frames.
# e/f/g) Visual observability and cleanliness:
#    - Same-shape frames/tools isolate the fusion dimensions: color chooses the partner, while the
#      cavity-sized region chooses the inserted role. Distinct frame/handle colors avoid ambiguous
#      pairings, and rejecting samples without a "closer to wrong frame" tool makes the new rule
#      visible from a small number of examples.

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox as _bbox, get_bg, paste


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS = 2
GEN_ATTEMPT_RETRIES = 300

FRAME_W = 4
TOP_H = 2
BOT_H = 2
NOTCH_OFFSETS = [(0, 2), (0, 3)]  # notch_n = 2
CAVITY_AREA = FRAME_W + len(NOTCH_OFFSETS)  # 6
PLUG_MASK = np.array(
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 0],
    ],
    dtype=bool,
)
PLUG_AREA = int(PLUG_MASK.sum())  # 7
PLUG_VOFF = 1
MIN_TOOL_FRAME_GAP = 4


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
    out = np.rot90(arr, int(k))
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
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(),
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(),
    )


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
        pp, mm = _crop_to_mask(p, m)
        key = (mm.tobytes(), pp[mm].tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((pp, mm, is_flip))
    return uniq


def _build_left_pair(bg: int, frame_color: int, handle_color: int):
    fh = TOP_H + 1 + BOT_H
    frame = np.full((fh, FRAME_W), bg, dtype=int)
    frame[:TOP_H, :] = int(frame_color)
    frame[TOP_H + 1 :, :] = int(frame_color)
    for dr, dc in NOTCH_OFFSETS:
        frame[TOP_H + 1 + dr, dc] = int(bg)

    cavity = frame == int(bg)
    tool_full = np.full((fh, FRAME_W + PLUG_MASK.shape[1]), bg, dtype=int)
    frame_full = np.full_like(tool_full, int(bg))
    frame_full[:, :FRAME_W] = frame
    tool_full[:, :FRAME_W][cavity] = int(handle_color)
    tool_full[PLUG_VOFF : PLUG_VOFF + PLUG_MASK.shape[0], FRAME_W : FRAME_W + PLUG_MASK.shape[1]][PLUG_MASK] = int(frame_color)

    tool_mask = tool_full != int(bg)
    tool_patch, tool_patch_mask = _crop_to_mask(tool_full, tool_mask)
    return frame_full, tool_full, tool_patch, tool_patch_mask


def _collect_anchor_groups(g: np.ndarray, bg: int):
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

    assert fixed_components

    parent = list(range(len(fixed_components)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
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
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        frame_vals = [int(x) for x in g[mask].flatten() if int(x) != int(bg)]
        frame_color = int(Counter(frame_vals).most_common(1)[0][0])

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_id = None
        best_area = -1
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
            if adj and area > best_area:
                best_area = area
                best_id = int(cid)
        assert best_id is not None
        for rr, cc in np.argwhere(sub_lab == int(best_id)):
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
                "cavity_mask": cavity,
                "cavity_size": int(cavity.sum()),
                "frame_color": frame_color,
                "open_dir": open_dir,
                "size": int(mask.sum()),
            }
        )

    anchor_groups.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return anchor_groups, fixed_mask


def _collect_movable_components(g: np.ndarray, fixed_mask: np.ndarray):
    fg = g != get_bg(g)
    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == int(cid))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        m = mov_lab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(cid)
        patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
        patch[~m] = 0
        vals = [int(x) for x in patch[m].flatten()]
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": m,
                "patch": patch,
                "counts": dict(Counter(vals)),
                "colors": sorted(set(vals)),
                "size": int(len(pts)),
            }
        )
    mov_comps.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return mov_comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock loose two-color tools into border frames using color pairing and role-swapped insertion.

    Input structure:
    - bg is the most frequent color.
    - Border-touching pieces cluster into one-color frames whose interior bg cavity is a staircase notch.
    - Non-border connected components are two-color tools; one of their colors matches exactly one frame color.

    Rule:
    - Pair each tool to the frame whose color appears on that tool, regardless of cavity shape matching.
    - Rotate/flip the tool so the cavity-sized color region fills the notch, while the frame-matching color stays outside as the protrusion.
    - If the winning docking transform uses a reflection and the two color counts differ, swap the tool's two colors.
    - Remove the original free tools; output keeps only the frames with their docked tools.

    Key insight:
    - Color chooses which frame a tool belongs to, but area chooses which color region goes inside: the cavity-sized region inserts, and the frame-colored region protrudes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    anchor_groups, fixed_mask = _collect_anchor_groups(g, bg)
    mov_comps = _collect_movable_components(g, fixed_mask)

    frame_by_color = {int(grp["frame_color"]): i for i, grp in enumerate(anchor_groups)}
    group_for_comp = []
    for comp in mov_comps:
        matches = [frame_by_color[col] for col in comp["colors"] if col in frame_by_color]
        assert len(matches) == 1
        group_for_comp.append(int(matches[0]))

    if len(set(group_for_comp)) != len(group_for_comp):
        best_perm = None
        for perm in itertools.permutations(range(len(anchor_groups)), len(mov_comps)):
            ok = True
            for ci, gi in enumerate(perm):
                if int(anchor_groups[gi]["frame_color"]) not in mov_comps[ci]["colors"]:
                    ok = False
                    break
            if ok:
                best_perm = perm
                break
        assert best_perm is not None
        group_for_comp = [int(x) for x in best_perm]

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    for comp, gi in zip(mov_comps, group_for_comp):
        grp = anchor_groups[gi]
        cav = grp["cavity_mask"]
        cav_sz = int(grp["cavity_size"])
        odr, odc = grp["open_dir"]
        placed_grp = placed_groups[gi]
        frame_bbox = grp["bbox"]
        comp_counts = {int(k): int(v) for k, v in comp["counts"].items()}

        best_state = None
        best_score = None
        for p, m, is_flip in _gen_variants(comp["patch"], comp["mask"]):
            h, w = m.shape
            coords = np.argwhere(m)
            color_counts = Counter(int(p[rr, cc]) for rr, cc in coords)
            handle_candidates = [int(col) for col, cnt in color_counts.items() if int(cnt) == cav_sz]
            if not handle_candidates:
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

                    best_role = None
                    for handle_col in handle_candidates:
                        handle_hits = 0
                        handle_outside = 0
                        other_in_cav = 0
                        plug_outside = 0
                        for rr, cc in coords:
                            r = int(r0 + rr)
                            c = int(c0 + cc)
                            in_cav = bool(cav[r, c])
                            col = int(p[rr, cc])
                            if col == int(handle_col):
                                if in_cav:
                                    handle_hits += 1
                                else:
                                    handle_outside += 1
                            else:
                                if in_cav:
                                    other_in_cav += 1
                                if odc == -1 and c < frame_bbox[1]:
                                    plug_outside += 1
                                elif odc == 1 and c > frame_bbox[3]:
                                    plug_outside += 1
                                elif odr == -1 and r < frame_bbox[0]:
                                    plug_outside += 1
                                elif odr == 1 and r > frame_bbox[2]:
                                    plug_outside += 1
                        handle_full = int(handle_hits == cav_sz and handle_outside == 0 and other_in_cav == 0)
                        role_score = (
                            int(handle_full),
                            int(handle_hits),
                            -int(other_in_cav),
                            -int(handle_outside),
                            int(plug_outside),
                        )
                        if best_role is None or role_score > best_role:
                            best_role = role_score

                    dist = _bbox_gap(frame_bbox, (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
                    score = tuple(best_role) + (int(adj), -int(dist), -int(r0), -int(c0))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_state = (p.copy(), m.copy(), int(r0), int(c0), bool(is_flip))

        assert best_state is not None
        p, m, r0, c0, is_flip = best_state
        if is_flip and len(comp_counts) == 2 and len(set(comp_counts.values())) == 2:
            a, b = sorted(int(x) for x in comp_counts.keys())
            p = _swap_colors(p, a, b)
        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True
            placed_grp[r, c] = True

    return out


def _place_pair_on_side(
    side: str,
    frame_color: int,
    handle_color: int,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
):
    frame_left, tool_left, tool_patch_left, tool_mask_left = _build_left_pair(bg, frame_color, handle_color)
    frame_scene = _orient_left_to_side(frame_left, side)
    tool_scene = _orient_left_to_side(tool_left, side)
    tool_patch = _orient_left_to_side(tool_patch_left, side)
    tool_mask = tool_patch != int(bg)
    scene = np.where(tool_scene != int(bg), tool_scene, frame_scene)
    ph, pw = scene.shape
    fb = _bbox(frame_scene != int(bg))
    assert fb is not None
    fr0, fc0, fr1, fc1 = fb

    if side == "left":
        r0 = random.randint(1, H - ph - 1)
        c0 = 0
    elif side == "right":
        r0 = random.randint(1, H - ph - 1)
        c0 = W - pw
    elif side == "top":
        r0 = 0
        c0 = random.randint(1, W - pw - 1)
    elif side == "bottom":
        r0 = H - ph
        c0 = random.randint(1, W - pw - 1)
    else:
        raise ValueError(side)

    paste(output, scene, r0, c0, transparent=bg)
    paste(input_grid, frame_scene, r0, c0, transparent=bg)
    return {
        "side": side,
        "frame_color": int(frame_color),
        "handle_color": int(handle_color),
        "frame_bbox": (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1)),
        "tool_patch": tool_patch,
        "tool_mask": tool_mask,
    }


def _variant_choices(patch: np.ndarray, mask: np.ndarray):
    out = _gen_variants(patch, mask)
    random.shuffle(out)
    return out


def _candidate_origins_for_tool(side: str, ph: int, pw: int, H: int, W: int):
    if side == "left":
        rows = range(1, H - ph)
        cols = range(max(W // 2, 1), W - pw)
    elif side == "right":
        rows = range(1, H - ph)
        cols = range(1, max(2, W // 2 - pw + 1))
    elif side == "top":
        rows = range(max(H // 2, 1), H - ph)
        cols = range(1, W - pw)
    elif side == "bottom":
        rows = range(1, max(2, H // 2 - ph + 1))
        cols = range(1, W - pw)
    else:
        raise ValueError(side)
    origins = [(int(r0), int(c0)) for r0 in rows for c0 in cols]
    random.shuffle(origins)
    return origins


def _place_free_tool(
    info: dict,
    target_index: int,
    bg: int,
    H: int,
    W: int,
    input_grid: np.ndarray,
    placement_canvas: np.ndarray,
    frame_bboxes: list[tuple[int, int, int, int]],
    prefer_flip: bool,
):
    variants = _variant_choices(info["tool_patch"], info["tool_mask"])
    variants.sort(key=lambda x: int(x[2]) != int(prefer_flip))

    for patch, mask, is_flip in variants:
        patch_in = patch.copy()
        if is_flip and PLUG_AREA != CAVITY_AREA:
            patch_in = _swap_colors(patch_in, info["frame_color"], info["handle_color"])
        ph, pw = patch_in.shape
        for r0, c0 in _candidate_origins_for_tool(info["side"], ph, pw, H, W):
            bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
            if any(
                int(placement_canvas[int(r0 + rr), int(c0 + cc)]) != int(bg)
                for rr, cc in np.argwhere(mask)
            ):
                continue
            if any(_bbox_gap(bbox, fb) < MIN_TOOL_FRAME_GAP for fb in frame_bboxes):
                continue

            target_d = _bbox_gap(bbox, frame_bboxes[target_index])
            other_ds = [_bbox_gap(bbox, fb) for j, fb in enumerate(frame_bboxes) if j != target_index]
            if not other_ds or min(other_ds) >= target_d:
                continue

            paste(input_grid, patch_in, r0, c0, transparent=bg)
            paste(placement_canvas, patch_in, r0, c0, transparent=bg)
            return {
                "bbox": bbox,
                "is_flip": bool(is_flip),
                "target_index": int(target_index),
            }
    return None


def _behavior_metrics(grid: np.ndarray):
    bg = int(get_bg(grid))
    anchor_groups, fixed_mask = _collect_anchor_groups(grid, bg)
    mov_comps = _collect_movable_components(grid, fixed_mask)
    frame_by_color = {int(grp["frame_color"]): i for i, grp in enumerate(anchor_groups)}

    cross_pair_count = 0
    role_inversion_count = 0
    for comp in mov_comps:
        matches = [frame_by_color[col] for col in comp["colors"] if col in frame_by_color]
        if len(matches) != 1:
            continue
        gi = int(matches[0])
        frame_color = int(anchor_groups[gi]["frame_color"])
        cav_sz = int(anchor_groups[gi]["cavity_size"])
        frame_count = int(comp["counts"].get(frame_color, 0))
        other_counts = [int(v) for col, v in comp["counts"].items() if int(col) != frame_color]
        if other_counts and other_counts[0] == cav_sz and frame_count != cav_sz:
            role_inversion_count += 1
        target_d = _bbox_gap(comp["bbox"], anchor_groups[gi]["bbox"])
        other_ds = [_bbox_gap(comp["bbox"], grp["bbox"]) for j, grp in enumerate(anchor_groups) if j != gi]
        if other_ds and min(other_ds) < target_d:
            cross_pair_count += 1

    return {
        "num_pairs": len(anchor_groups),
        "equal_cavities": len({int(grp["cavity_size"]) for grp in anchor_groups}) == 1,
        "cross_pair_count": int(cross_pair_count),
        "role_inversion_count": int(role_inversion_count),
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))

    usable = [c for c in range(10) if c != bg]
    random.shuffle(usable)
    frame_colors = usable[:NUM_PAIRS]
    handle_colors = usable[NUM_PAIRS : 2 * NUM_PAIRS]

    family = random.choice(["horizontal", "vertical"])
    sides = ["left", "right"] if family == "horizontal" else ["top", "bottom"]

    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    pair_infos = []
    for i in range(NUM_PAIRS):
        pair_infos.append(
            _place_pair_on_side(
                sides[i],
                frame_colors[i],
                handle_colors[i],
                bg,
                H,
                W,
                output,
                input_grid,
            )
        )

    placement_canvas = input_grid.copy()
    free_infos = []
    for i, pair_info in enumerate(pair_infos):
        prefer_flip = False if i == 0 else bool(random.getrandbits(1))
        placed = _place_free_tool(
            pair_info,
            i,
            bg,
            H,
            W,
            input_grid,
            placement_canvas,
            [x["frame_bbox"] for x in pair_infos],
            prefer_flip=prefer_flip,
        )
        if placed is None:
            return None
        free_infos.append(placed)

    metrics = _behavior_metrics(input_grid)
    if not metrics["equal_cavities"]:
        return None
    if metrics["cross_pair_count"] < 1:
        return None
    if metrics["role_inversion_count"] < 1:
        return None
    if not np.array_equal(main(input_grid), output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
