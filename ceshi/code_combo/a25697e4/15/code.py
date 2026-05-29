# Combo variant of a25697e4 (index 15)
# combo_spec: D2/0+D6/2+D15/2
# Name: 距离决定旋转旋转决定变色

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox as _bbox, get_bg, paste, random_sprite


# Design analysis for the fused variant:
# a) Concrete behavior trace:
#    - Keep the original object family: each border frame has a staircase cavity, and each free tool
#      is a single 2-color connected component whose one color-region has the same area as exactly one
#      cavity. In the source task, the solver may rotate or flip the tool until the cavity-sized color
#      fits, and only reflections can trigger a color swap.
#    - Here the chain is explicit and coupled: first pair tool↔frame by cavity size; then measure the
#      Manhattan bbox gap d between that tool and that frame in the input; set rotation steps k=d mod 4;
#      rotate the tool by k quarter-turns only; finally, if k is odd, swap the two tool colors before
#      docking. Example: a tool 7 cells away from its frame has k=3, so it rotates 270 degrees and its
#      cavity-filling / protruding colors exchange roles in the output.
#    - The visible difference from the original is therefore not just "no flips": odd-distance classes
#      now force a recolor even though the geometry still docks by rotation alone.
# b) Complexity calibration:
#    - Every generated sample contains two pairs, with one even k in {0,2} and one odd k in {1,3}.
#      This gives repeated support for both "distance controls rotation" and "rotation parity controls
#      color swap" inside a single grid instead of relying on cross-example luck.
#    - Plug areas are distinct across the two pairs, so pairing is not ambiguous; free-tool distances
#      are limited to 4..7, which keeps the encoded control variable easy for humans to count.
# c) Hidden-assumption audit against the original solver:
#    - Any code path that enumerated reflections is invalid here; solver and generator both restrict
#      orientation space to the four rotations only.
#    - The old color-swap trigger "winning orientation used a flip" is gone. The new trigger depends on
#      the pre-docking frame/tool distance, so the solver must infer pairing before orientation.
#    - Because odd rotations swap colors regardless of region-size equality, the generator avoids equal
#      plug/handle areas; otherwise humans would get weaker visual evidence for the role exchange.
# d) Placement legality:
#    - Output is built first with non-overlapping border scenes. Free tools are then reconstructed by
#      inverse rotation and placed at exact axial gaps 4..7 from their own frames, strictly in-bounds,
#      non-border, and non-overlapping with any frame or other tool.
# e) Visual observability:
#    - The encoded variable is human-visible: the tool is placed directly inward from its target frame,
#      so the relevant distance is a simple empty-row / empty-column count, not a hidden metadata value.
#    - Odd-gap tools visibly change which color fills the cavity compared with their free appearance;
#      even-gap tools preserve that role after rotation.
# f) Visual cleanliness:
#    - Only two frame-tool pairs are present, each with distinct frame/plug/handle colors and on one
#      vertical plus one horizontal border. This avoids central clutter while keeping the fusion logic
#      legible.
# g) New solver assumptions and generator guarantees:
#    - `main()` assumes each frame cluster contains one largest adjacent bg cavity and each free tool has
#      exactly two colors with exactly one region size matching one cavity size. `generate_input()` makes
#      cavity sizes unique and keeps tools isolated from borders so that assumption always holds.
#    - `main()` assumes the measured bbox gap to the matched frame is the intended encoder. The generator
#      places each tool on the open side of its own frame with row/column overlap, so the bbox gap is a
#      single-axis distance equal to the encoded value.


GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS = 2

FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)

HANDLE_AREA_RANGE = (4, 10)
TOOL_TOTAL_BOUNDS = (10, 15)
HANDLE_SPRITE_DIM = (2, 5)

UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
SCENE_MARGIN = 1
TOOL_BORDER = 1
ENCODED_GAP_BASE = 4

PAIR_SPEC_RETRIES = 300
HANDLE_SPRITE_RETRIES = 30
GEN_ATTEMPT_RETRIES = 400

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return _gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1)


def _swap_colors(arr: np.ndarray, a: int, b: int) -> np.ndarray:
    out = arr.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    mask = arr != int(bg)
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


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


def _candidate_border_origins(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
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
    frame[:top_h, :] = int(frame_color)
    frame[top_h + 1 :, :] = int(frame_color)
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = int(bg)

    cavity = frame == int(bg)
    attach_rows = {int(r) for r in range(fh) if cavity[r, fw - 1]}
    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None

    hs_h, hs_w = handle_sprite.shape
    sprite_left_rows = {int(r) for r in range(hs_h) if handle_sprite[r, 0] != bg}
    valid_offsets = [voff for voff in range(fh) if any((voff + sr) in attach_rows for sr in sprite_left_rows)]
    if not valid_offsets:
        return None

    voff = random.choice(valid_offsets)
    total_h = max(fh, voff + hs_h)
    total_w = fw + hs_w

    frame_full = np.full((total_h, total_w), bg, dtype=int)
    frame_full[:fh, :fw] = frame

    tool = np.full((total_h, total_w), bg, dtype=int)
    tool[:fh, :fw][cavity] = int(plug_color)
    handle_mask = handle_sprite != bg
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
    for _ in range(PAIR_SPEC_RETRIES):
        specs: list[dict] = []
        chosen_params = random.sample(PARAM_OPTIONS, k=len(PARAM_OPTIONS))
        for fw, top_h, bot_h, notch_n in chosen_params:
            plug_area = fw + notch_n
            if any(spec["plug_area"] == plug_area for spec in specs):
                continue
            specs.append(
                {
                    "fw": fw,
                    "top_h": top_h,
                    "bot_h": bot_h,
                    "notch_n": notch_n,
                    "plug_area": plug_area,
                }
            )
            if len(specs) == num_pairs:
                break
        if len(specs) != num_pairs:
            continue

        plug_areas = {int(spec["plug_area"]) for spec in specs}
        for spec in specs:
            plug_area = int(spec["plug_area"])
            handle_choices = [
                h
                for h in range(ha_lo, ha_hi)
                if tt_lo <= plug_area + h <= tt_hi and h not in plug_areas
            ]
            if not handle_choices:
                specs = []
                break
            spec["handle_area"] = int(random.choice(handle_choices))
        if len(specs) == num_pairs:
            return specs
    return []


def _collect_anchor_groups(g: np.ndarray, bg: int):
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        pts_list = [(int(r), int(c)) for r, c in pts]
        if not any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts_list):
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
        rs = [r for r, _c in pts]
        cs = [c for _r, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_id = None
        best_area = -1
        for cid in range(1, int(sub_n) + 1):
            ptsc = np.argwhere(sub_lab == int(cid))
            area = int(len(ptsc))
            adjacent = False
            for rr, cc in ptsc:
                r = int(bbox[0] + rr)
                c = int(bbox[1] + cc)
                for dr, dc in DIR4:
                    r2, c2 = r + dr, c + dc
                    if bbox[0] <= r2 <= bbox[2] and bbox[1] <= c2 <= bbox[3] and mask[r2, c2]:
                        adjacent = True
                        break
                if adjacent:
                    break
            if adjacent and area > best_area:
                best_area = area
                best_id = int(cid)
        assert best_id is not None
        for rr, cc in np.argwhere(sub_lab == int(best_id)):
            cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        touches_left = any(c == 0 for _r, c in pts)
        touches_right = any(c == W - 1 for _r, c in pts)
        touches_top = any(r == 0 for r, _c in pts)
        touches_bottom = any(r == H - 1 for r, _c in pts)
        border_hits = sum(int(x) for x in [touches_left, touches_right, touches_top, touches_bottom])
        assert border_hits == 1
        if touches_left:
            open_dir = (0, 1)
        elif touches_right:
            open_dir = (0, -1)
        elif touches_top:
            open_dir = (1, 0)
        else:
            open_dir = (-1, 0)

        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "cavity_mask": cavity,
                "cavity_size": int(cavity.sum()),
                "open_dir": open_dir,
                "size": int(mask.sum()),
            }
        )

    anchor_groups.sort(key=lambda d: -d["size"])
    return anchor_groups, fixed_mask


def _collect_movable_tools(g: np.ndarray, fixed_mask: np.ndarray, bg: int):
    fg = g != int(bg)
    mov_lab, mov_n = label(fg & (~fixed_mask), structure=STRUCT4)
    comps = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == int(cid))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == int(cid)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = int(bg)
        colors = sorted({int(x) for x in patch[mask].flatten()})
        assert len(colors) == 2
        counts = Counter(int(x) for x in patch[mask].flatten())
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "colors": colors,
                "counts": {int(k): int(v) for k, v in counts.items()},
                "size": int(len(pts)),
            }
        )
    comps.sort(key=lambda d: -d["size"])
    return comps


def _assign_tools_to_groups(mov_comps: list[dict], anchor_groups: list[dict]) -> list[int]:
    comp_color_counts = [comp["counts"] for comp in mov_comps]

    def compatible(ci: int, gi: int) -> bool:
        cav_sz = int(anchor_groups[gi]["cavity_size"])
        return any(int(v) == cav_sz for v in comp_color_counts[ci].values())

    valid_perms = []
    for perm in itertools.permutations(range(len(anchor_groups)), len(mov_comps)):
        if all(compatible(ci, gi) for ci, gi in enumerate(perm)):
            valid_perms.append(perm)
            if len(valid_perms) > 1:
                break
    assert len(valid_perms) == 1
    return [int(x) for x in valid_perms[0]]


def _prepare_tool_for_group(comp: dict, grp: dict) -> tuple[np.ndarray, np.ndarray, int]:
    steps = int(_bbox_gap(comp["bbox"], grp["bbox"]) % 4)
    patch = np.rot90(comp["patch"], steps)
    mask = np.rot90(comp["mask"], steps)
    if steps % 2 == 1:
        a, b = comp["colors"]
        patch = _swap_colors(patch, a, b)
    return patch, mask, steps


def _try_place_transformed_tool(
    out: np.ndarray,
    placed_global: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    grp: dict,
    bg: int,
) -> bool:
    cavity_coords = np.argwhere(grp["cavity_mask"])
    cavity_set = {(int(r), int(c)) for r, c in cavity_coords}
    cav_sz = int(grp["cavity_size"])
    bbox = grp["bbox"]
    odr, odc = grp["open_dir"]

    colors = sorted({int(x) for x in patch[mask].flatten() if int(x) != int(bg)})
    plug_colors = [int(col) for col in colors if int(np.sum((patch == int(col)) & mask)) == cav_sz]
    assert len(plug_colors) == 1
    plug_color = plug_colors[0]

    plug_coords = np.argwhere((patch == int(plug_color)) & mask)
    nonplug_coords = np.argwhere((patch != int(plug_color)) & mask)
    all_coords = np.argwhere(mask)
    anchor_rr, anchor_cc = plug_coords[0]

    best_state = None
    best_score = None
    H, W = out.shape
    for cav_r, cav_c in cavity_coords:
        r0 = int(cav_r - anchor_rr)
        c0 = int(cav_c - anchor_cc)

        translated_all = []
        ok = True
        for rr, cc in all_coords:
            r = int(r0 + rr)
            c = int(c0 + cc)
            if not (0 <= r < H and 0 <= c < W):
                ok = False
                break
            if placed_global[r, c]:
                ok = False
                break
            translated_all.append((r, c, int(rr), int(cc)))
        if not ok:
            continue

        translated_plug = {(int(r0 + rr), int(c0 + cc)) for rr, cc in plug_coords}
        if translated_plug != cavity_set:
            continue
        if any((int(r0 + rr), int(c0 + cc)) in cavity_set for rr, cc in nonplug_coords):
            continue

        adj = 0
        outside = 0
        for r, c, _rr, _cc in translated_all:
            for dr, dc in DIR4:
                rr2, cc2 = r + dr, c + dc
                if 0 <= rr2 < H and 0 <= cc2 < W and grp["mask"][rr2, cc2]:
                    adj += 1
            if odc == -1 and c < bbox[1]:
                outside += 1
            elif odc == 1 and c > bbox[3]:
                outside += 1
            elif odr == -1 and r < bbox[0]:
                outside += 1
            elif odr == 1 and r > bbox[2]:
                outside += 1
        if adj == 0 or outside == 0:
            continue

        score = (int(outside), int(adj), -int(r0), -int(c0))
        if best_score is None or score > best_score:
            best_score = score
            best_state = translated_all

    assert best_state is not None
    for r, c, rr, cc in best_state:
        out[r, c] = int(patch[rr, cc])
        placed_global[r, c] = True
    return True


def _inspect_controls(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    anchor_groups, fixed_mask = _collect_anchor_groups(grid, bg)
    mov_comps = _collect_movable_tools(grid, fixed_mask, bg)
    assignment = _assign_tools_to_groups(mov_comps, anchor_groups)
    info = []
    for comp, gi in zip(mov_comps, assignment):
        grp = anchor_groups[gi]
        dist = int(_bbox_gap(comp["bbox"], grp["bbox"]))
        info.append({"distance": dist, "steps": dist % 4, "odd_swap": int(dist % 2 == 1)})
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock free 2-color tools into border frames, with distance encoding the only allowed rotation.

    Input structure:
    - Background is the most common color.
    - Border-touching monochrome pieces cluster into 1-2 anchor frames that contain staircase bg cavities.
    - Interior 2-color connected components are movable tools; for each one, exactly one color-region size matches exactly one frame cavity size.

    Rule:
    - Pair each tool to the frame whose cavity size matches one of the tool's two color-region sizes.
    - Measure the tool-to-frame Manhattan bbox gap d, rotate the tool by d mod 4 quarter-turns, and do not allow flips.
    - After that rotation, swap the tool's two colors iff the rotation count is odd, then place it so the cavity-sized region fills the frame cavity and the other region protrudes outward.

    Key insight: Geometry still identifies which region can fill a cavity, but the frame distance tells you how much to rotate first, and the parity of that forced rotation determines whether the two colors trade roles.
    """
    g = np.array(grid, dtype=int)
    bg = get_bg(g)

    anchor_groups, fixed_mask = _collect_anchor_groups(g, bg)
    mov_comps = _collect_movable_tools(g, fixed_mask, bg)
    assignment = _assign_tools_to_groups(mov_comps, anchor_groups)

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()

    for comp, gi in zip(mov_comps, assignment):
        grp = anchor_groups[gi]
        patch, mask, _steps = _prepare_tool_for_group(comp, grp)
        _try_place_transformed_tool(out, placed_global, patch, mask, grp, bg)

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

    built = _build_left_pair(
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
    if built is None:
        return None
    frame_left, tool_left = built
    scene_left = np.where(tool_left != bg, tool_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool_scene = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    fr0, fc0, fr1, fc1 = _bbox(frame_scene != bg)

    origin = None
    for r0, c0 in _candidate_border_origins(side, ph, pw, H, W):
        union_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        frame_bbox = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) < UNION_BBOX_GAP_MIN for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) < FRAME_BBOX_GAP_MIN for prev in frame_bboxes):
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

    frame_bbox = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)
    union_bboxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    frame_bboxes.append(frame_bbox)
    return {
        "tool_docked": tool_scene,
        "frame_bbox": frame_bbox,
        "side": side,
        "plug_color": plug_color,
        "handle_color": handle_color,
        "plug_area": int(spec["plug_area"]),
        "handle_area": int(spec["handle_area"]),
    }


def _free_tool_patch(info: dict, steps: int, bg: int) -> np.ndarray:
    docked = info["tool_docked"]
    patch = docked.copy()
    if steps % 2 == 1:
        patch = _swap_colors(patch, info["plug_color"], info["handle_color"])
    free_patch = np.rot90(patch, -steps)
    return _crop_non_bg(free_patch, bg)


def _candidate_free_origins(
    frame_bbox: tuple[int, int, int, int],
    patch_shape: tuple[int, int],
    side: str,
    gap: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    ph, pw = patch_shape
    fr0, fc0, fr1, fc1 = frame_bbox
    origins = []
    if side == "left":
        c0 = fc1 + gap
        r_lo = max(TOOL_BORDER, fr0 - ph + 1)
        r_hi = min(H - ph - TOOL_BORDER, fr1)
        if c0 + pw - 1 <= W - 1 - TOOL_BORDER and r_lo <= r_hi:
            origins = [(r0, c0) for r0 in range(r_lo, r_hi + 1)]
    elif side == "right":
        c0 = fc0 - gap - pw + 1
        r_lo = max(TOOL_BORDER, fr0 - ph + 1)
        r_hi = min(H - ph - TOOL_BORDER, fr1)
        if c0 >= TOOL_BORDER and r_lo <= r_hi:
            origins = [(r0, c0) for r0 in range(r_lo, r_hi + 1)]
    elif side == "top":
        r0 = fr1 + gap
        c_lo = max(TOOL_BORDER, fc0 - pw + 1)
        c_hi = min(W - pw - TOOL_BORDER, fc1)
        if r0 + ph - 1 <= H - 1 - TOOL_BORDER and c_lo <= c_hi:
            origins = [(r0, c0) for c0 in range(c_lo, c_hi + 1)]
    elif side == "bottom":
        r0 = fr0 - gap - ph + 1
        c_lo = max(TOOL_BORDER, fc0 - pw + 1)
        c_hi = min(W - pw - TOOL_BORDER, fc1)
        if r0 >= TOOL_BORDER and c_lo <= c_hi:
            origins = [(r0, c0) for c0 in range(c_lo, c_hi + 1)]
    else:
        raise ValueError(side)
    random.shuffle(origins)
    return origins


def _place_free_tool(
    info: dict,
    steps: int,
    bg: int,
    input_grid: np.ndarray,
    occupancy: np.ndarray,
    frame_bboxes: list[tuple[int, int, int, int]],
    tool_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    patch = _free_tool_patch(info, steps, bg)
    ph, pw = patch.shape
    gap = ENCODED_GAP_BASE + int(steps)
    origins = _candidate_free_origins(info["frame_bbox"], (ph, pw), info["side"], gap, input_grid.shape[0], input_grid.shape[1])
    for r0, c0 in origins:
        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if _bbox_gap(bbox, info["frame_bbox"]) != gap:
            continue
        if any(
            fb != info["frame_bbox"] and _bbox_gap(bbox, fb) < 2
            for fb in frame_bboxes
        ):
            continue
        if any(_bbox_gap(bbox, tb) < 2 for tb in tool_bboxes):
            continue
        region = occupancy[r0 : r0 + ph, c0 : c0 + pw]
        if np.any((patch != bg) & (region != bg)):
            continue
        paste(input_grid, patch, r0, c0, transparent=bg)
        paste(occupancy, patch, r0, c0, transparent=bg)
        tool_bboxes.append(bbox)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))

    pair_specs = _pick_pair_specs(NUM_PAIRS)
    if len(pair_specs) != NUM_PAIRS:
        return None

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    if len(colors) < 3 * NUM_PAIRS:
        return None

    sides = [random.choice(["left", "right"]), random.choice(["top", "bottom"])]
    random.shuffle(sides)
    step_choices = [random.choice([0, 2]), random.choice([1, 3])]
    random.shuffle(step_choices)

    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_infos: list[dict] = []

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
        info["steps"] = int(step_choices[i])
        placed_infos.append(info)

    occupancy = input_grid.copy()
    tool_bboxes: list[tuple[int, int, int, int]] = []
    random.shuffle(placed_infos)
    for info in placed_infos:
        if not _place_free_tool(info, int(info["steps"]), bg, input_grid, occupancy, frame_bboxes, tool_bboxes):
            return None

    controls = _inspect_controls(input_grid)
    if sorted(item["steps"] for item in controls) != sorted(step_choices):
        return None
    if not any(item["odd_swap"] for item in controls):
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
