# Combo variant of a25697e4 (index 10)
# combo_spec: D4/3+D7/3
# Name: 浮空框架对接消融

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox as _bbox, get_bg, paste, random_free_location, random_sprite


# Design analysis for the fused variant:
# a) Concrete behavior trace:
#    - Original task: a border-anchored frame stays visible in the output, and the tool docks into it.
#    - This combo keeps the same plug-into-cavity docking logic, but moves the whole frame off the border.
#      Example: if a floating frame has its cavity touching the right side of its own bbox, the tool must dock
#      from the right even though no object touches the canvas edge. After docking, the frame is deleted, so the
#      visible output is only the transformed 2-color tool suspended at the frame location.
#    - The pixel-level difference is therefore twofold: direction is inferred from the cavity opening, and the
#      output canvas starts blank instead of copying frame pixels.
# b) Complexity calibration:
#    - The geometric family is kept close to the source task: 2 same-color monochrome frame pieces, staircase
#      cavity area 5-8, one 2-color connected tool per frame, 1-2 pairs total.
#    - Ambiguity is controlled by forcing distinct plug areas across pairs and distinct frame colors across all
#      pairs, so humans can visually separate objects while still needing cavity-size matching to pair them.
# c) Hidden-assumption audit against the original solver:
#    - "Fixed components are border-touching" no longer holds, so frame discovery now starts from all connected
#      components and classifies monochrome components as frame candidates.
#    - "Open direction equals edge direction" no longer holds, so open direction is inferred from which side of
#      the frame bbox the cavity touches.
#    - "Output begins with the frame already copied" no longer holds, so the new output begins as pure background
#      and only the docked tool is pasted back.
# d) Placement legality:
#    - Generator places every frame/tool pair with internal margins, enforces pair bbox gaps, and places free
#      tools using `random_free_location` with padding so nothing overlaps or leaks out of bounds.
# e) Visual observability:
#    - The floating frame is clearly visible in the input because it is isolated from the border.
#    - The frame vanishes completely in the output, leaving a tool-shaped remnant exactly where the cavity was.
#      This makes the fused behavior visible from only a few examples.
# f) Visual cleanliness:
#    - Only one layer is present at each stage: input shows separate frames plus separate tools; output shows only
#      docked tools. No frame pixels remain underneath to create clutter.
# g) New solver assumptions and generator guarantees:
#    - `main()` assumes each true frame color contributes exactly one nearby pair of monochrome components whose
#      union bbox contains a single staircase cavity. `generate_input()` enforces that by using unique frame colors
#      and the same fixed frame template family as the source task.
#    - `main()` assumes each tool has exactly one color-region size matching one frame cavity size. The generator
#      guarantees distinct plug areas and chooses unequal plug/handle sizes, preserving the flip-based color-swap
#      rule without equal-area ambiguity.


# ---------- shared geometry ----------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [1, 2]

FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)

HANDLE_AREA_RANGE = (4, 10)
TOOL_TOTAL_BOUNDS = (10, 15)
HANDLE_SPRITE_DIM = (2, 5)

SCENE_MARGIN = 1
UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 4
TOOL_PADDING_CONN = 4

PAIR_SPEC_RETRIES = 300
HANDLE_SPRITE_RETRIES = 30
GEN_ATTEMPT_RETRIES = 300

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    if flip:
        out = np.fliplr(out)
    return out


def _orient_open_right_to(arr: np.ndarray, open_dir: str) -> np.ndarray:
    if open_dir == "right":
        return arr.copy()
    if open_dir == "left":
        return np.fliplr(arr)
    if open_dir == "down":
        return np.rot90(arr, 3)
    if open_dir == "up":
        return np.rot90(arr, 1)
    raise ValueError(open_dir)


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


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _candidate_origins_internal(ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
    origins = [
        (r0, c0)
        for r0 in range(SCENE_MARGIN, H - ph - SCENE_MARGIN + 1)
        for c0 in range(SCENE_MARGIN, W - pw - SCENE_MARGIN + 1)
    ]
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


def _build_open_right_pair(
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


def _pick_pair_specs(num_pairs: int) -> list[dict[str, int]]:
    ha_lo, ha_hi = HANDLE_AREA_RANGE
    tt_lo, tt_hi = TOOL_TOTAL_BOUNDS
    specs: list[dict[str, int]] = []
    for _ in range(PAIR_SPEC_RETRIES):
        specs.clear()
        used_plug_areas: set[int] = set()
        for _pair_idx in range(num_pairs):
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
        ok = True
        all_plug_areas = {s["plug_area"] for s in specs}
        for spec in specs:
            pa = spec["plug_area"]
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


def _place_docked_pair(
    i: int,
    spec: dict[str, int],
    colors: list[int],
    open_dir: str,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> dict[str, np.ndarray | int] | None:
    frame_color = colors[3 * i]
    plug_color = colors[3 * i + 1]
    handle_color = colors[3 * i + 2]

    result = _build_open_right_pair(
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
    frame_right, tool_right = result
    scene_right = np.where(tool_right != bg, tool_right, frame_right)

    frame_scene = _orient_open_right_to(frame_right, open_dir)
    tool = _orient_open_right_to(tool_right, open_dir)
    scene = _orient_open_right_to(scene_right, open_dir)

    ph, pw = scene.shape
    frame_bbox_local = _bbox(frame_scene != bg)
    if frame_bbox_local is None:
        return None
    fr0, fc0, fr1, fc1 = frame_bbox_local

    origin = None
    for r0, c0 in _candidate_origins_internal(ph, pw, H, W):
        union_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        frame_bbox = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)
        if any(_bbox_gap(union_bbox, prev) <= UNION_BBOX_GAP_MIN - 1 for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) <= FRAME_BBOX_GAP_MIN - 1 for prev in frame_bboxes):
            continue
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != bg) & (scene != bg)):
            continue
        if np.any((input_grid[r0 : r0 + ph, c0 : c0 + pw] != bg) & (frame_scene != bg)):
            continue
        origin = (r0, c0)
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, tool, r0, c0, transparent=bg)
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


def _undock_tool(
    info: dict[str, np.ndarray | int],
    bg: int,
    input_grid: np.ndarray,
    placement_canvas: np.ndarray,
) -> bool:
    tool = info["tool"]
    assert isinstance(tool, np.ndarray)
    for patch, is_flip in _variant_choices(tool):
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
        paste(placement_canvas, patch, r0, c0, transparent=bg)
        return True
    return False


def _extract_components(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    fg = grid != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    frame_pieces: list[dict] = []
    tools: list[dict] = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = lab[r0 : r1 + 1, c0 : c1 + 1] == cid
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = 0
        colors = sorted({int(x) for x in patch[mask].flatten()})
        info = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "mask": mask,
            "patch": patch,
            "colors": colors,
            "size": int(len(pts)),
            "pts": pts,
        }
        if len(colors) == 1:
            info["color"] = colors[0]
            frame_pieces.append(info)
        elif len(colors) == 2:
            tools.append(info)
    frame_pieces.sort(key=lambda d: -int(d["size"]))
    tools.sort(key=lambda d: -int(d["size"]))
    return frame_pieces, tools


def _build_anchor_groups(grid: np.ndarray, bg: int, frame_pieces: list[dict]) -> list[dict]:
    H, W = grid.shape
    mono_by_color: dict[int, list[dict]] = {}
    for comp in frame_pieces:
        mono_by_color.setdefault(int(comp["color"]), []).append(comp)

    anchor_groups: list[dict] = []
    used_ids: set[int] = set()
    for color, comps in mono_by_color.items():
        best_pair = None
        best_gap = None
        indexed = list(enumerate(comps))
        for (ia, a), (ib, b) in itertools.combinations(indexed, 2):
            aid = (color << 16) + ia
            bid = (color << 16) + ib
            if aid in used_ids or bid in used_ids:
                continue
            gap = _bbox_gap(a["bbox"], b["bbox"])
            if gap > 2:
                continue
            pts = np.concatenate([a["pts"], b["pts"]], axis=0)
            rs = pts[:, 0]
            cs = pts[:, 1]
            bbox = (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))
            sub_bg = grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
            sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
            if sub_n == 0:
                continue
            areas = [(int(np.sum(sub_lab == cid)), cid) for cid in range(1, sub_n + 1)]
            area, best_id = max(areas)
            if not (PLUG_AREA_BOUNDS[0] <= area <= PLUG_AREA_BOUNDS[1]):
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_pair = (aid, bid, a, b, bbox, best_id)
        if best_pair is None:
            continue
        aid, bid, a, b, bbox, best_id = best_pair
        used_ids.add(aid)
        used_ids.add(bid)

        mask = np.zeros((H, W), dtype=bool)
        for pts in (a["pts"], b["pts"]):
            for r, c in pts:
                mask[int(r), int(c)] = True

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, _sub_n = label(sub_bg, structure=STRUCT4)
        ptsc = np.argwhere(sub_lab == best_id)
        for rr, cc in ptsc:
            cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        sock = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(grid[r, c]) != int(bg):
                    continue
                for dr, dc in DIRS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        sock[r, c] = True
                        break

        side_counts = {
            "up": int(sum(cavity[bbox[0], c] for c in range(bbox[1], bbox[3] + 1))),
            "down": int(sum(cavity[bbox[2], c] for c in range(bbox[1], bbox[3] + 1))),
            "left": int(sum(cavity[r, bbox[1]] for r in range(bbox[0], bbox[2] + 1))),
            "right": int(sum(cavity[r, bbox[3]] for r in range(bbox[0], bbox[2] + 1))),
        }
        open_dir = max(side_counts.items(), key=lambda kv: kv[1])[0]
        anchor_groups.append(
            {
                "color": color,
                "mask": mask,
                "bbox": bbox,
                "socket_mask": sock,
                "cavity_mask": cavity,
                "cavity_size": int(cavity.sum()),
                "open_dir": open_dir,
                "size": int(mask.sum()),
            }
        )
    anchor_groups.sort(key=lambda d: -int(d["size"]))
    return anchor_groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock interior 2-color tools into floating staircase frames, then erase the frames.

    Input structure:
    - bg is the most common color.
    - Each frame is fully inside the grid and is split into two same-color monochrome components.
    - Each movable tool is a single 4-connected component with exactly two colors and also stays off the border.

    Rule:
    - Pair a tool with the frame whose cavity area matches one of the tool's color-region sizes.
    - Rotate/flip the tool so one color fills the cavity exactly while the other protrudes through the frame opening.
    - If the winning docking transform uses a reflection and the two color-region sizes differ, swap the two tool
      colors; then delete the frame and keep only the docked tool in the output.

    Key insight:
    - The frame is still the docking template, but because it floats, its opening direction must be read from the
      cavity shape itself; after alignment, the frame disappears, so the output is only the transformed tool.
    """
    g = grid.copy()
    bg = get_bg(g)
    H, W = g.shape

    frame_pieces, mov_comps = _extract_components(g, bg)
    anchor_groups = _build_anchor_groups(g, bg, frame_pieces)
    if not anchor_groups or not mov_comps:
        return np.full_like(g, int(bg))

    def choose_group(bbox: tuple[int, int, int, int]) -> int:
        best_i = 0
        best_d = None
        for i, grp in enumerate(anchor_groups):
            d = _bbox_gap(bbox, grp["bbox"])
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        return int(best_i)

    comp_color_counts: list[dict[int, int]] = []
    for comp in mov_comps:
        vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]
        comp_color_counts.append(dict(Counter(vals)))

    def compatible(ci: int, gi: int) -> bool:
        cav_sz = int(anchor_groups[gi]["cavity_size"])
        return any(int(v) == cav_sz for v in comp_color_counts[ci].values())

    group_for_comp: list[int]
    if len(mov_comps) == len(anchor_groups) and len(anchor_groups) > 1:
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
        if best_perm is not None:
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

    fixed_mask = np.zeros((H, W), dtype=bool)
    for grp in anchor_groups:
        fixed_mask |= grp["mask"]

    out = np.full_like(g, int(bg))
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    for comp, gi in zip(mov_comps, group_for_comp):
        grp = anchor_groups[gi]
        placed_grp = placed_groups[gi]
        sock = grp["socket_mask"]
        cav = grp["cavity_mask"]
        odir = grp["open_dir"]
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
                    socket_hits = 0
                    cavity_hits = 0
                    outside = 0
                    br0, bc0, br1, bc1 = grp["bbox"]
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        for dr, dc in DIRS4:
                            r2 = r + dr
                            c2 = c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                adj += 1
                        if sock[r, c]:
                            socket_hits += 1
                        if cav[r, c]:
                            cavity_hits += 1
                        if odir == "left" and c < bc0:
                            outside += 1
                        elif odir == "right" and c > bc1:
                            outside += 1
                        elif odir == "up" and r < br0:
                            outside += 1
                        elif odir == "down" and r > br1:
                            outside += 1
                    if adj == 0:
                        continue

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
                        best_state = (p, m, int(r0), int(c0), bool(is_flip))

        if best_state is None:
            return np.full_like(g, int(bg))
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


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_pairs = random.choice(NUM_PAIRS_CHOICES)

    pair_specs = _pick_pair_specs(num_pairs)
    if len(pair_specs) != num_pairs:
        return None

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    need = 3 * num_pairs
    if len(colors) < need:
        return None
    colors = colors[:need]

    open_dirs = random.sample(["left", "right", "up", "down"], k=num_pairs)
    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict[str, np.ndarray | int]] = []

    for i, spec in enumerate(pair_specs):
        info = _place_docked_pair(
            i,
            spec,
            colors,
            open_dirs[i],
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
        if not _undock_tool(info, bg, input_grid, placement_canvas):
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
