# Combo variant of a25697e4 (index 23)
# combo_spec: D2/0+D16/2
# Name: 内嵌标记纯旋转
#
# Design analysis:
# - Concrete behavior trace:
#   Original task treats each free tool as a 2-color component. One color-region alone has area
#   equal to the frame cavity, and the solver may use any D4 transform, including reflections. In
#   this combo, each free tool is a 3-color component: plug body color + 1 marker pixel + handle
#   color. The cavity is filled by (plug body + marker) together, so no single color-region has the
#   cavity size anymore. A valid docking transform must be a rotation only, and the translated
#   placement is pinned by the marker pixel landing on the cavity cell deepest inside the socket.
# - Small example:
#   Suppose a left-border frame has a 6-cell cavity. The docked output contains 5 plug-color cells,
#   1 marker-color cell at the leftmost/topmost cavity cell, and a handle extending rightward. The
#   input keeps the frame but moves the tool elsewhere after a random rotation. The original logic
#   would search all rotations+flips and expect one monochrome region of size 6; the combo logic
#   must instead identify the singleton marker, combine it with the 5-cell plug region, forbid
#   flips, and rotate until the marker returns to that canonical socket corner.
# - Complexity calibration:
#   The marker stays a single pixel so the visual change is clear but not noisy. Plug areas remain
#   distinct across pairs, preventing pairing ambiguity. The target marker rule depends only on
#   visible geometry: it always occupies the cavity cell farthest from the opening side, so humans
#   can infer it from a few examples without needing hidden color-order conventions.
# - Hidden-assumption audit:
#   The original `compatible()` assumption "one color count == cavity size" is broken here, so the
#   new solver uses "one non-marker color count + 1 == cavity size". The original reflection-based
#   color-swap branch is removed entirely; the generator never flips detached tools. The placement
#   search no longer relies on heuristic scoring for many translations: the marker identifies one
#   absolute anchor cell, so each rotation yields at most one candidate translation.
# - Placement legality:
#   The generator still uses output-first assembly, border-anchored frames, gap checks between pair
#   bboxes, and `random_free_location()` for detached tools. Docked handle pixels are forced to lie
#   strictly outside the frame bbox in the opening direction, keeping the scene non-overlapping and
#   visually tidy.
# - Visual observability:
#   The marker is a visibly unique color inside the plug, preserved in the output. Because the plug
#   area is split into (body + marker), the combo does not collapse back to the original rule where
#   shape matching alone and a 2-color area count would suffice.
# - New solver assumptions guaranteed by the generator:
#   every movable component has exactly 3 colors; exactly one of them is a singleton marker; exactly
#   one remaining color has count `cavity_size - 1`; detached tools are produced only by rotations;
#   and the frame cavity cell selected by `_select_marker_cell_from_mask()` is unique enough to pin
#   the docking pose.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

TASK_ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = TASK_ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, get_bg, paste, random_free_location, random_sprite
from arc_utils import bbox as _bbox


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

# ── CFG: retry limits ──
PAIR_SPEC_RETRIES = 300
HANDLE_SPRITE_RETRIES = 30
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


def _candidate_origins(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "left":
        out = [(r0, 0) for r0 in range(H - ph + 1)]
    elif side == "right":
        out = [(r0, W - pw) for r0 in range(H - ph + 1)]
    elif side == "top":
        out = [(0, c0) for c0 in range(W - pw + 1)]
    elif side == "bottom":
        out = [(H - ph, c0) for c0 in range(W - pw + 1)]
    else:
        raise ValueError(side)
    random.shuffle(out)
    return out


def _rotation_variants(patch: np.ndarray, mask: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray, int]]:
    if mask is None:
        mask = patch != 0
    seen = set()
    out: list[tuple[np.ndarray, np.ndarray, int]] = []
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        key = (m.tobytes(), p[m].tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append((p, m, k))
    return out


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


def _select_marker_cell_from_mask(mask: np.ndarray, open_dir: tuple[int, int]) -> tuple[int, int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert coords
    if open_dir == (0, 1):
        return min(coords, key=lambda rc: (rc[1], rc[0]))
    if open_dir == (0, -1):
        return min(coords, key=lambda rc: (-rc[1], rc[0]))
    if open_dir == (1, 0):
        return min(coords, key=lambda rc: (rc[0], -rc[1]))
    if open_dir == (-1, 0):
        return min(coords, key=lambda rc: (-rc[0], rc[1]))
    raise ValueError(open_dir)


def _count_outside_open_dir(coords: set[tuple[int, int]], bbox, open_dir: tuple[int, int]) -> int:
    br0, bc0, br1, bc1 = bbox
    odr, odc = open_dir
    count = 0
    for r, c in coords:
        if odc == -1 and c < bc0:
            count += 1
        elif odc == 1 and c > bc1:
            count += 1
        elif odr == -1 and r < br0:
            count += 1
        elif odr == 1 and r > br1:
            count += 1
    return count


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
                h
                for h in range(ha_lo, ha_hi)
                if tt_lo <= pa + h <= tt_hi and h not in all_plug_areas and (h + 1) not in all_plug_areas
            ]
            if not handle_choices:
                ok = False
                break
            spec["handle_area"] = random.choice(handle_choices)
        if ok:
            return [dict(s) for s in specs]
    return []


def _build_left_pair(
    *,
    bg: int,
    frame_color: int,
    plug_color: int,
    marker_color: int,
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
    marker_r, marker_c = _select_marker_cell_from_mask(cavity, (0, 1))
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
    tool[marker_r, marker_c] = marker_color
    tool[voff : voff + hs_h, fw : fw + hs_w][handle_sprite != bg] = handle_color
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
    frame_color = colors[4 * i]
    plug_color = colors[4 * i + 1]
    marker_color = colors[4 * i + 2]
    handle_color = colors[4 * i + 3]

    result = _build_left_pair(
        bg=bg,
        frame_color=frame_color,
        plug_color=plug_color,
        marker_color=marker_color,
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
    return {"tool": tool}


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    variants = _rotation_variants(info["tool"])
    random.shuffle(variants)
    variants.sort(key=lambda item: item[2] == 0)
    for patch, _mask, _k in variants:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock marked movable tools into border-attached frames by restoring a rotation-coded pose.

    Input structure:
    - The background is the most common color.
    - Border-touching same-color pieces cluster into frames that contain one staircase cavity.
    - Each free tool is a single non-border 4-connected component with exactly 3 colors:
      plug body, 1-pixel direction marker, and handle.

    Rule:
    - Pair each tool with the frame whose cavity size equals (plug body count + marker count).
    - Only rotate the tool; reflections are forbidden.
    - Rotate so the marked plug pixel lands on the cavity cell farthest from the opening side, then
      place the rest of the plug into the cavity and keep the handle protruding outward.

    Key insight:
    - The singleton marker is part of the plug, so area matching uses a fused 2-part plug region,
      and the marker fixes the unique allowed rotation and translation.
    """
    g = grid.copy()
    bg = get_bg(g)
    H, W = g.shape

    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)

    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        pts_list = [(int(r), int(c)) for r, c in pts]
        touches_border = any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts_list)
        if not touches_border:
            continue
        bbox = (int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max()))
        fixed_components.append({"pts": pts_list, "bbox": bbox})
        for r, c in pts_list:
            fixed_mask[r, c] = True
    assert fixed_components, "expected at least one border frame"

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
        best_id = None
        best_area = -1
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
            if adj and area > best_area:
                best_area = area
                best_id = cid
        assert best_id is not None, "expected a frame cavity"
        for rr, cc in np.argwhere(sub_lab == best_id):
            cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        touches_left = any(c == 0 for _r, c in pts)
        touches_right = any(c == W - 1 for _r, c in pts)
        touches_top = any(r == 0 for r, _c in pts)
        touches_bottom = any(r == H - 1 for r, _c in pts)
        touch_count = sum(int(x) for x in [touches_left, touches_right, touches_top, touches_bottom])
        assert touch_count == 1, "each frame must touch exactly one border"
        if touches_left:
            open_dir = (0, 1)
        elif touches_right:
            open_dir = (0, -1)
        elif touches_top:
            open_dir = (1, 0)
        else:
            open_dir = (-1, 0)
        marker_target = _select_marker_cell_from_mask(cavity, open_dir)
        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "cavity_mask": cavity,
                "cavity_coords": {tuple(map(int, rc)) for rc in np.argwhere(cavity)},
                "cavity_size": int(cavity.sum()),
                "open_dir": open_dir,
                "marker_target": marker_target,
                "size": int(mask.sum()),
            }
        )
    anchor_groups.sort(key=lambda item: -item["size"])

    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps = []
    for cid in range(1, mov_n + 1):
        pts = np.argwhere(mov_lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == cid
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = 0
        vals = [int(x) for x in patch[mask].flatten() if int(x) != 0]
        counts = Counter(vals)
        marker_colors = [col for col, cnt in counts.items() if cnt == 1]
        assert len(counts) == 3 and len(marker_colors) == 1, "every tool must have one singleton marker"
        marker_color = int(marker_colors[0])
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "counts": counts,
                "marker_color": marker_color,
                "size": int(len(pts)),
            }
        )
    mov_comps.sort(key=lambda item: -item["size"])

    def choose_group(bbox) -> int:
        best_i = 0
        best_d = None
        for i, grp in enumerate(anchor_groups):
            d = _bbox_gap(bbox, grp["bbox"])
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        return int(best_i)

    def compatible(ci: int, gi: int) -> bool:
        cav_sz = int(anchor_groups[gi]["cavity_size"])
        comp = mov_comps[ci]
        return any(cnt + 1 == cav_sz for col, cnt in comp["counts"].items() if col != comp["marker_color"])

    if len(mov_comps) == len(anchor_groups) and len(anchor_groups) > 1:
        import itertools

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
        assert best_perm is not None, "must find a compatible permutation"
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
            group_for_comp.append(int(best_g) if best_g is not None else int(choose_group(comp["bbox"])))

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    for comp, gi in zip(mov_comps, group_for_comp):
        grp = anchor_groups[gi]
        counts = comp["counts"]
        marker_color = comp["marker_color"]
        cav_sz = int(grp["cavity_size"])
        plug_candidates = [int(col) for col, cnt in counts.items() if col != marker_color and cnt + 1 == cav_sz]
        assert len(plug_candidates) == 1, "each tool must have exactly one plug color for its frame"
        plug_color = plug_candidates[0]
        handle_color = [int(col) for col in counts if col not in {marker_color, plug_color}]
        assert len(handle_color) == 1
        handle_color = handle_color[0]

        best_state = None
        best_score = None
        target_r, target_c = grp["marker_target"]
        for p, m, k in _rotation_variants(comp["patch"], comp["mask"]):
            marker_pts = np.argwhere((p == marker_color) & m)
            assert len(marker_pts) == 1
            mr, mc = map(int, marker_pts[0])
            r0 = int(target_r - mr)
            c0 = int(target_c - mc)
            h, w = m.shape
            if not (0 <= r0 and 0 <= c0 and r0 + h <= H and c0 + w <= W):
                continue

            abs_all = {(int(r0 + rr), int(c0 + cc)) for rr, cc in np.argwhere(m)}
            if any(placed_global[r, c] for r, c in abs_all):
                continue

            plug_mask = ((p == plug_color) | (p == marker_color)) & m
            handle_mask = (p == handle_color) & m
            abs_plug = {(int(r0 + rr), int(c0 + cc)) for rr, cc in np.argwhere(plug_mask)}
            abs_handle = {(int(r0 + rr), int(c0 + cc)) for rr, cc in np.argwhere(handle_mask)}
            if abs_plug != grp["cavity_coords"]:
                continue
            if abs_handle & grp["cavity_coords"]:
                continue
            outside = _count_outside_open_dir(abs_handle, grp["bbox"], grp["open_dir"])
            if outside != len(abs_handle):
                continue

            adj = 0
            placed_grp = placed_groups[gi]
            for r, c in abs_all:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and placed_grp[rr, cc]:
                        adj += 1
            score = (adj, -k, -r0, -c0)
            if best_score is None or score > best_score:
                best_score = score
                best_state = (p, m, r0, c0)

        assert best_state is not None, "must find a valid rotation-only placement"
        p, m, r0, c0 = best_state
        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True
            placed_groups[gi][r, c] = True

    return out


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
    need = 4 * num_pairs
    if len(colors) < need:
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

    return input_grid if np.array_equal(main(input_grid), output) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
