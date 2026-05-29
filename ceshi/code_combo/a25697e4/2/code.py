# Combo variant of a25697e4 (index 2)
# combo_spec: D1/0+D3/0
# Name: 颜色匹配三色工具

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, paste, random_free_location

# Design analysis:
# - Concrete behavior trace:
#   The original task paired tools to frames by cavity area. Here every frame uses the same
#   staircase cavity and every tool uses the same plug geometry, so geometry no longer tells
#   the pairing apart. Instead, the tool color that matches a frame color is the plug color.
#   Each tool has three color regions: plug, hinge, handle. During docking the plug geometry
#   fills the cavity, hinge stays visible in its own color, and handle protrudes outward.
#   If the required docking transform is reflective and plug/handle areas differ, only those
#   two colors are swapped in the output; the hinge color is preserved.
# - Complexity calibration:
#   The generator always uses two frame-tool pairs, two distinct frame colors, identical cavity
#   sizes, identical plug shapes, and exactly one reflected free tool. This creates repeated
#   support for the color-match rule and also exposes the flip-triggered color swap within each
#   sample without exploding the case space.
# - Hidden assumptions audited against the original solver:
#   The original solver assumed "one tool color count equals cavity size" and that tools are
#   only two-colored. Both assumptions are invalid now. The new solver instead assumes:
#   1) each frame group has one dominant non-background color,
#   2) each movable tool is a single 4-connected three-color component,
#   3) the frame color appears in exactly one tool and identifies the plug region,
#   4) the hinge is the unique non-plug color touching both other colors,
#   5) plug and handle sizes differ, so reflection-triggered swaps are observable.
#   The generator enforces all five conditions.
# - Placement legality:
#   Output is built first from in-bounds border-anchored scenes. Input tools are then scattered
#   with padding and a border margin, so they never overlap frames, one another, or the border.
# - Visual observability and neatness:
#   Using two pairs with identical geometry but different frame colors makes color matching
#   necessary and legible. Keeping the hinge visible as a 2-pixel connector makes the third
#   color structurally meaningful instead of decorative. Exactly one reflected free tool per
#   sample ensures the swap behavior is present but not visually chaotic.


GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (26, 30)
NUM_PAIRS = 2

FRAME_W = 4
TOP_H = 1
BOT_H = 3
NOTCH_N = 2
PLUG_AREA = FRAME_W + NOTCH_N
HANDLE_AREA = 5
HINGE_AREA = 2

TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4
UNION_BBOX_GAP_MIN = 3
FRAME_BBOX_GAP_MIN = 4
GEN_ATTEMPT_RETRIES = 300


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


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(arr != int(bg))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _swap_colors(arr: np.ndarray, a: int, b: int) -> np.ndarray:
    out = arr.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


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


def _variant_choices(patch: np.ndarray, bg: int) -> list[tuple[np.ndarray, bool]]:
    seen = set()
    out: list[tuple[np.ndarray, bool]] = []
    for k in range(4):
        for flip in (False, True):
            q = _transform(patch, k, flip)
            qq = _crop_non_bg(q, bg)
            mask = qq != int(bg)
            key = (mask.tobytes(), qq[mask].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((qq, bool(flip)))
    random.shuffle(out)
    return out


def _build_left_templates(
    *,
    bg: int,
    frame_color: int,
    handle_color: int,
    hinge_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned left-border frame and unswapped docked tool templates."""
    fh = TOP_H + 1 + BOT_H
    frame = np.full((fh, FRAME_W), bg, dtype=int)
    frame[:TOP_H, :] = frame_color
    frame[TOP_H + 1 :, :] = frame_color
    frame[TOP_H + 1, FRAME_W - NOTCH_N :] = bg

    full = np.full((fh, 7), bg, dtype=int)
    full[:, :FRAME_W] = frame

    tool = np.full_like(full, bg)
    cavity = frame == bg
    tool[:, :FRAME_W][cavity] = frame_color

    hinge_cells = [(1, 4), (2, 4)]
    handle_cells = [(0, 5), (1, 5), (1, 6), (2, 5), (3, 5)]
    for r, c in hinge_cells:
        tool[r, c] = hinge_color
    for r, c in handle_cells:
        tool[r, c] = handle_color
    return full, tool


def _extract_anchor_groups(g: np.ndarray, bg: int):
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        pts_list = [(int(r), int(c)) for r, c in pts]
        touches = any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts_list)
        if not touches:
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

    anchor_groups = []
    for idxs in groups.values():
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best = None
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
            cand = (area, -cid)
            if best is None or cand > best:
                best = cand
                best_id = int(cid)
        assert best_id is not None
        ptsc = np.argwhere(sub_lab == best_id)
        for rr, cc in ptsc:
            cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        # The cavity touches the bbox edge that points toward the tool protrusion.
        cav_pts = np.argwhere(cavity)
        edge_counts = {
            (-1, 0): int(np.sum(cav_pts[:, 0] == bbox[0])),
            (1, 0): int(np.sum(cav_pts[:, 0] == bbox[2])),
            (0, -1): int(np.sum(cav_pts[:, 1] == bbox[1])),
            (0, 1): int(np.sum(cav_pts[:, 1] == bbox[3])),
        }
        open_dir = max(edge_counts.items(), key=lambda kv: kv[1])[0]

        frame_colors = [int(x) for x in np.unique(g[mask]) if int(x) != int(bg)]
        assert frame_colors
        frame_color = int(Counter(frame_colors).most_common(1)[0][0])
        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "cavity_mask": cavity,
                "cavity_size": int(cavity.sum()),
                "open_dir": open_dir,
                "frame_color": frame_color,
                "size": int(mask.sum()),
            }
        )
    anchor_groups.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["size"]))
    return fixed_mask, anchor_groups


def _extract_tools(g: np.ndarray, bg: int, fixed_mask: np.ndarray):
    fg = g != int(bg)
    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    comps = []
    for cid in range(1, mov_n + 1):
        pts = np.argwhere(mov_lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == cid
        patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), bg, dtype=int)
        patch[mask] = g[r0 : r1 + 1, c0 : c1 + 1][mask]
        colors = sorted(int(x) for x in np.unique(patch[mask]))
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "colors": colors,
                "size": int(len(pts)),
            }
        )
    comps.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["size"]))
    return comps


def _touches(a: np.ndarray, b: np.ndarray) -> bool:
    pts = np.argwhere(a)
    for r, c in pts:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < b.shape[0] and 0 <= cc < b.shape[1] and b[rr, cc]:
                return True
    return False


def _identify_hinge_and_handle(comp: dict, bg: int, plug_color: int) -> tuple[int, int]:
    colors = [int(x) for x in comp["colors"] if int(x) != int(plug_color)]
    assert len(colors) == 2
    region_masks = {col: ((comp["patch"] == int(col)) & comp["mask"]) for col in [plug_color] + colors}
    a, b = colors
    if _touches(region_masks[a], region_masks[plug_color]) and _touches(region_masks[a], region_masks[b]):
        return int(a), int(b)
    if _touches(region_masks[b], region_masks[plug_color]) and _touches(region_masks[b], region_masks[a]):
        return int(b), int(a)
    areas = {col: int(region_masks[col].sum()) for col in colors}
    hinge = min(colors, key=lambda col: areas[col])
    handle = a if hinge == b else b
    return int(hinge), int(handle)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock three-color tools into border-anchored staircase frames using color-based pairing.

    Input structure:
    - bg is the most common color.
    - Border-touching pixels form one-color frames with staircase cavities.
    - Interior movable objects are single 4-connected three-color tools: one frame-matching plug
      region, one visible hinge region, and one handle region.

    Rule:
    - Pair each tool to the frame whose color matches one tool region; that matched region is the
      plug, regardless of the other region sizes.
    - Rotate/flip the tool so the plug geometry fills the cavity exactly, while hinge and handle
      remain outside the frame and the hinge keeps its original color.
    - If the winning docking transform is reflective and plug/handle areas are unequal, swap only
      the plug and handle colors in the docked result.

    Key insight: Matching depends on the tool color that echoes a frame, not on cavity area; the
    third hinge color identifies the connector that stays visible while reflection changes the two
    functional colors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    fixed_mask, anchor_groups = _extract_anchor_groups(g, bg)
    mov_comps = _extract_tools(g, bg, fixed_mask)
    assert mov_comps

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    def compatible(ci: int, gi: int) -> bool:
        return int(anchor_groups[gi]["frame_color"]) in set(int(x) for x in mov_comps[ci]["colors"])

    group_for_comp: list[int]
    if len(mov_comps) == len(anchor_groups):
        best_perm = None
        best_score = None
        for perm in itertools.permutations(range(len(anchor_groups))):
            ok = True
            dist_sum = 0
            for ci, gi in enumerate(perm):
                if not compatible(ci, gi):
                    ok = False
                    break
                dist_sum += _bbox_gap(mov_comps[ci]["bbox"], anchor_groups[gi]["bbox"])
            if not ok:
                continue
            score = -int(dist_sum)
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None
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
            assert best_g is not None
            group_for_comp.append(int(best_g))

    for comp, gi in zip(mov_comps, group_for_comp):
        grp = anchor_groups[gi]
        placed_grp = placed_groups[gi]
        plug_color = int(grp["frame_color"])
        hinge_color, handle_color = _identify_hinge_and_handle(comp, bg, plug_color)
        plug_area = int(((comp["patch"] == plug_color) & comp["mask"]).sum())
        handle_area = int(((comp["patch"] == handle_color) & comp["mask"]).sum())
        assert plug_area == int(grp["cavity_size"])

        best_state = None
        best_score = None
        cavity = grp["cavity_mask"]
        br0, bc0, br1, bc1 = grp["bbox"]
        odr, odc = grp["open_dir"]

        for k in range(4):
            for is_flip in (False, True):
                p = _transform(comp["patch"], k, is_flip)
                m = _transform(comp["mask"].astype(np.uint8), k, is_flip).astype(bool)
                h, w = m.shape
                coords = np.argwhere(m)
                for r0 in range(0, H - h + 1):
                    for c0 in range(0, W - w + 1):
                        if any(placed_global[int(r0 + rr), int(c0 + cc)] for rr, cc in coords):
                            continue
                        adj = 0
                        plug_hits = 0
                        plug_outside = 0
                        nonplug_in_cavity = 0
                        nonplug_open = 0
                        nonplug_elsewhere = 0
                        for rr, cc in coords:
                            r = int(r0 + rr)
                            c = int(c0 + cc)
                            col = int(p[rr, cc])
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                r2, c2 = r + dr, c + dc
                                if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                    adj += 1
                            in_cavity = bool(cavity[r, c])
                            if col == plug_color:
                                if in_cavity:
                                    plug_hits += 1
                                else:
                                    plug_outside += 1
                            elif in_cavity:
                                nonplug_in_cavity += 1
                            elif odc == -1 and c < bc0:
                                nonplug_open += 1
                            elif odc == 1 and c > bc1:
                                nonplug_open += 1
                            elif odr == -1 and r < br0:
                                nonplug_open += 1
                            elif odr == 1 and r > br1:
                                nonplug_open += 1
                            else:
                                nonplug_elsewhere += 1
                        if adj == 0:
                            continue
                        plug_full = int(
                            plug_hits == int(grp["cavity_size"])
                            and plug_outside == 0
                            and nonplug_in_cavity == 0
                        )
                        dist = _bbox_gap(grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
                        score = (
                            int(plug_full),
                            int(plug_hits),
                            -int(nonplug_in_cavity),
                            -int(plug_outside),
                            int(nonplug_open),
                            -int(nonplug_elsewhere),
                            int(adj),
                            -int(dist),
                            -int(r0),
                            -int(c0),
                        )
                        if best_score is None or score > best_score:
                            best_score = score
                            best_state = (p.copy(), m.copy(), int(r0), int(c0), bool(is_flip))

        assert best_state is not None
        p, m, r0, c0, is_flip = best_state
        if is_flip and plug_area != handle_area:
            p = _swap_colors(p, plug_color, handle_color)
        for rr, cc in np.argwhere(m):
            out[int(r0 + rr), int(c0 + cc)] = int(p[rr, cc])
            placed_global[int(r0 + rr), int(c0 + cc)] = True
            placed_grp[int(r0 + rr), int(c0 + cc)] = True

    return out


def _place_pair(
    *,
    frame_scene: np.ndarray,
    tool_output_scene: np.ndarray,
    side: str,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int], tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    scene = np.where(tool_output_scene != int(bg), tool_output_scene, frame_scene)
    ph, pw = scene.shape
    frame_pts = np.argwhere(frame_scene != int(bg))
    fr0, fc0 = frame_pts.min(axis=0)
    fr1, fc1 = frame_pts.max(axis=0)

    origin = None
    best_union = None
    best_frame = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) < UNION_BBOX_GAP_MIN for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) < FRAME_BBOX_GAP_MIN for prev in frame_bboxes):
            continue
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != int(bg)) & (scene != int(bg))):
            continue
        origin = (int(r0), int(c0))
        best_union = union_bbox
        best_frame = frame_bbox
        break
    if origin is None or best_union is None or best_frame is None:
        return None

    paste(output, scene, origin[0], origin[1], transparent=bg)
    paste(input_grid, frame_scene, origin[0], origin[1], transparent=bg)
    union_bboxes.append(best_union)
    frame_bboxes.append(best_frame)
    return origin, best_union, best_frame


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    frame_colors = colors[:NUM_PAIRS]
    handle_colors = colors[NUM_PAIRS : 2 * NUM_PAIRS]
    hinge_colors = colors[2 * NUM_PAIRS : 3 * NUM_PAIRS]

    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    tool_infos: list[dict] = []

    sides = random.sample(["left", "right", "top", "bottom"], k=NUM_PAIRS)
    flip_targets = [False, True]
    random.shuffle(flip_targets)

    for i in range(NUM_PAIRS):
        frame_left, tool_left = _build_left_templates(
            bg=bg,
            frame_color=int(frame_colors[i]),
            handle_color=int(handle_colors[i]),
            hinge_color=int(hinge_colors[i]),
        )
        side = sides[i]
        frame_scene = _orient_left_to_side(frame_left, side)
        tool_input_aligned = _orient_left_to_side(tool_left, side)
        cropped_tool = _crop_non_bg(tool_input_aligned, bg)

        variants = [(patch, is_flip) for patch, is_flip in _variant_choices(cropped_tool, bg) if bool(is_flip) == bool(flip_targets[i])]
        if not variants:
            return None
        free_patch, used_flip = random.choice(variants)

        tool_output_aligned = tool_input_aligned.copy()
        if used_flip and PLUG_AREA != HANDLE_AREA:
            tool_output_aligned = _swap_colors(tool_output_aligned, int(frame_colors[i]), int(handle_colors[i]))

        placed = _place_pair(
            frame_scene=frame_scene,
            tool_output_scene=tool_output_aligned,
            side=side,
            bg=bg,
            H=H,
            W=W,
            output=output,
            input_grid=input_grid,
            union_bboxes=union_bboxes,
            frame_bboxes=frame_bboxes,
        )
        if placed is None:
            return None
        tool_infos.append({"patch": free_patch})

    placement_canvas = output.copy()
    for info in tool_infos:
        try:
            r0, c0 = random_free_location(
                placement_canvas,
                info["patch"],
                transparent=bg,
                border=TOOL_BORDER,
                padding=TOOL_PADDING,
                padding_connectivity=TOOL_PADDING_CONN,
            )
        except ValueError:
            return None
        paste(input_grid, info["patch"], r0, c0, transparent=bg)
        paste(placement_canvas, info["patch"], r0, c0, transparent=bg)

    if np.array_equal(main(input_grid), output):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
