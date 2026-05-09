# Combo variant of a25697e4 (index 5)
# combo_spec: D1/2+D5/2
# Name: 洞数匹配T形凹槽
#
# Design analysis for the fused variant
# 1. Concrete behavior trace:
#    - Original task: a frame was paired with a tool by cavity area; one tool color filled the
#      staircase cavity and the other color stayed outside.
#    - Fused task here: the cavity itself is now a 1-cell-thick branched socket whose visible
#      branch count is 2/3/4. Tools still dock geometrically by filling that socket with their
#      plug color, but pairing is chosen by counting small marker pixels on the handle:
#      marker-count == cavity-branch-count.
#    - Small example: if two frames both have cavity area 5 but one is a bent path (2 branches)
#      and the other is a cross (4 branches), area no longer distinguishes them. The matching key
#      becomes the tool's 2-dot or 4-dot marker pattern. After the correct tool is selected, its
#      plug still has to align exactly with the socket under some D4 transform.
# 2. Complexity calibration:
#    - All cavity templates keep the same plug area (=5) so the old area-based rule is deliberately
#      non-diagnostic in multi-pair scenes.
#    - Branch counts are restricted to 2/3/4 only; this keeps the matching space small enough for
#      humans to infer while still making the new signal visible.
# 3. Implicit-assumption audit against the original solver:
#    - "one tool color count equals cavity area" is no longer safe, so compatibility now uses
#      marker-count instead of region size.
#    - "movable objects have exactly two colors" is no longer safe, because each tool carries a
#      third marker color. The solver therefore isolates the smallest color-count as the marker
#      layer and only swaps the remaining two colors under reflection.
#    - "cavity shape is staircase" is no longer safe, so branch-count is computed from the cavity's
#      4-neighbor degree-1 endpoints rather than from notch size assumptions.
# 4. Placement legality:
#    - Frames are built output-first as border-touching solid rectangles with a carved branched
#      cavity that opens only toward the docking side.
#    - Tools are 4-connected, remain away from borders in the input, and are placed with padding so
#      no overlaps or accidental merges occur.
# 5. Visual observability / neatness:
#    - Marker dots are isolated single pixels inside the handle, so humans can count them directly.
#    - The cavity shapes are sparse T/cross-like sockets with clear endpoints; frames and tools stay
#      separated by padding in the input to avoid clutter.
# 6. Non-degeneration check:
#    - Every cavity template used here has area 5. In 2-pair scenes, different frames therefore
#      share the same cavity area while differing in branch count, forcing the fused rule to matter.

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox as _bbox, get_bg, paste, random_free_location


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock marked tools into border frames whose cavities are branched T/cross sockets.

    Input structure:
    - The background is the most frequent color.
    - Border-touching monochrome components are anchor frames; each frame is a solid border slab
      with one carved bg cavity shaped like a 2/3/4-branch socket.
    - Non-border connected components are movable tools with three visible colors:
      plug, handle, and several isolated marker pixels on the handle.

    Rule:
    - Count the cavity branches of each frame and the marker pixels of each tool.
    - Match tools to frames by branch-count == marker-count.
    - Rotate/flip the chosen tool so its plug color fills the cavity exactly while handle+markers
      protrude outward from the touched border side.
    - If the winning docking transform uses a reflection and the plug/handle color counts differ,
      swap those two colors while leaving the marker color unchanged.

    Key insight: The cavity shape still controls the geometric docking, but the pairing key is the
    visible count signal: branch endpoints on the frame versus marker dots on the tool.
    """
    g = np.array(grid, dtype=int)
    bg = get_bg(g)
    H, W = g.shape

    def bbox_gap(a, b) -> int:
        (ar0, ac0, ar1, ac1) = a
        (br0, bc0, br1, bc1) = b

        def gap1(x0, x1, y0, y1):
            if y1 < x0:
                return int(x0 - y1)
            if x1 < y0:
                return int(y0 - x1)
            return 0

        return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)

    def gen_variants(patch: np.ndarray, mask: np.ndarray):
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

    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)

    fixed_components: list[dict] = []
    fixed_mask = np.zeros((H, W), dtype=bool)
    movable_components: list[dict] = []

    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        touches_border = any(
            int(r) == 0 or int(r) == H - 1 or int(c) == 0 or int(c) == W - 1
            for r, c in pts
        )
        mask = lab[r0 : r1 + 1, c0 : c1 + 1] == cid
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = 0
        info = {
            "bbox": bbox,
            "mask": mask,
            "patch": patch,
            "pts": [(int(r), int(c)) for r, c in pts],
            "size": int(len(pts)),
        }
        if touches_border:
            fixed_components.append(info)
            fixed_mask[r0 : r1 + 1, c0 : c1 + 1][mask] = True
        else:
            movable_components.append(info)

    assert fixed_components
    assert movable_components

    def extract_cavity_info(frame_info: dict) -> dict:
        r0, c0, r1, c1 = frame_info["bbox"]
        frame_mask = np.zeros((H, W), dtype=bool)
        for r, c in frame_info["pts"]:
            frame_mask[r, c] = True

        sub_bg = g[r0 : r1 + 1, c0 : c1 + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        cavity_mask = np.zeros((H, W), dtype=bool)
        best = None
        best_id = None
        for cid2 in range(1, sub_n + 1):
            pts2 = np.argwhere(sub_lab == cid2)
            area = int(len(pts2))
            adj = False
            touches_open_edge = False
            for rr, cc in pts2:
                rr2 = int(r0 + rr)
                cc2 = int(c0 + cc)
                if rr2 == r0 or rr2 == r1 or cc2 == c0 or cc2 == c1:
                    touches_open_edge = True
                for dr, dc in DIR4:
                    nr, nc = rr2 + dr, cc2 + dc
                    if 0 <= nr < H and 0 <= nc < W and frame_mask[nr, nc]:
                        adj = True
                        break
                if adj and touches_open_edge:
                    break
            if not adj:
                continue
            cand = (int(touches_open_edge), area, -cid2)
            if best is None or cand > best:
                best = cand
                best_id = cid2
        assert best_id is not None
        pts2 = np.argwhere(sub_lab == best_id)
        coords = set()
        for rr, cc in pts2:
            rr2 = int(r0 + rr)
            cc2 = int(c0 + cc)
            cavity_mask[rr2, cc2] = True
            coords.add((rr2, cc2))

        leaves = 0
        for rr2, cc2 in coords:
            deg = sum((rr2 + dr, cc2 + dc) in coords for dr, dc in DIR4)
            if deg == 1:
                leaves += 1
        assert 2 <= leaves <= 4

        socket_mask = np.zeros((H, W), dtype=bool)
        for rr2, cc2 in coords:
            for dr, dc in DIR4:
                nr, nc = rr2 + dr, cc2 + dc
                if 0 <= nr < H and 0 <= nc < W and frame_mask[nr, nc]:
                    socket_mask[rr2, cc2] = True
                    break

        if c0 == 0:
            open_dir = (0, 1)
        elif c1 == W - 1:
            open_dir = (0, -1)
        elif r0 == 0:
            open_dir = (1, 0)
        else:
            open_dir = (-1, 0)

        return {
            "mask": frame_mask,
            "bbox": frame_info["bbox"],
            "socket_mask": socket_mask,
            "cavity_mask": cavity_mask,
            "cavity_size": int(len(coords)),
            "branch_count": int(leaves),
            "open_dir": open_dir,
            "size": int(frame_info["size"]),
        }

    anchor_groups = [extract_cavity_info(comp) for comp in fixed_components]
    anchor_groups.sort(key=lambda d: -d["size"])
    movable_components.sort(key=lambda d: -d["size"])

    tool_infos = []
    for comp in movable_components:
        # `patch[mask]` already removes transparent padding; color 0 may still be a legitimate tool color.
        vals = [int(x) for x in comp["patch"][comp["mask"]].flatten()]
        counts = Counter(vals)
        ordered = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
        assert len(ordered) >= 3
        marker_color, marker_count = ordered[0]
        tool_infos.append(
            {
                **comp,
                "color_counts": counts,
                "marker_color": int(marker_color),
                "marker_count": int(marker_count),
            }
        )

    def compatible(tool: dict, grp: dict) -> bool:
        return int(tool["marker_count"]) == int(grp["branch_count"])

    if len(tool_infos) == len(anchor_groups) and len(anchor_groups) > 1:
        best_perm = None
        best_score = None
        for perm in itertools.permutations(range(len(anchor_groups))):
            ok = True
            dist_sum = 0
            for ti, gi in enumerate(perm):
                if not compatible(tool_infos[ti], anchor_groups[gi]):
                    ok = False
                    break
                dist_sum += bbox_gap(tool_infos[ti]["bbox"], anchor_groups[gi]["bbox"])
            if not ok:
                continue
            score = (-int(dist_sum),)
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None
        group_for_tool = [int(x) for x in best_perm]
    else:
        group_for_tool = []
        for tool in tool_infos:
            candidates = [
                (bbox_gap(tool["bbox"], grp["bbox"]), gi)
                for gi, grp in enumerate(anchor_groups)
                if compatible(tool, grp)
            ]
            assert candidates
            candidates.sort()
            group_for_tool.append(int(candidates[0][1]))

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    for tool, gi in zip(tool_infos, group_for_tool):
        grp = anchor_groups[gi]
        cav = grp["cavity_mask"]
        sock = grp["socket_mask"]
        placed_grp = placed_groups[gi]
        marker_color = int(tool["marker_color"])
        non_marker_colors = [int(c) for c in tool["color_counts"] if int(c) != marker_color]
        assert len(non_marker_colors) == 2
        counts_nm = {int(c): int(tool["color_counts"][c]) for c in non_marker_colors}

        best_state = None
        best_score = None
        cav_sz = int(grp["cavity_size"])
        odr, odc = grp["open_dir"]
        br0, bc0, br1, bc1 = grp["bbox"]

        for p, m, is_flip in gen_variants(tool["patch"], tool["mask"]):
            h, w = m.shape
            coords = np.argwhere(m)
            for r0 in range(0, H - h + 1):
                for c0 in range(0, W - w + 1):
                    blocked = False
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        if placed_global[r, c]:
                            blocked = True
                            break
                    if blocked:
                        continue

                    adj = 0
                    socket_hits = 0
                    cavity_hits = 0
                    outward = 0
                    nonplug_inside_bbox = 0
                    marker_outward = 0
                    marker_in_cavity = 0

                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        col = int(p[rr, cc])
                        for dr, dc in DIR4:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and placed_grp[nr, nc]:
                                adj += 1
                        if sock[r, c]:
                            socket_hits += 1
                        if cav[r, c]:
                            cavity_hits += 1

                        is_outward = False
                        if odc == 1 and c > bc1:
                            outward += 1
                            is_outward = True
                        elif odc == -1 and c < bc0:
                            outward += 1
                            is_outward = True
                        elif odr == 1 and r > br1:
                            outward += 1
                            is_outward = True
                        elif odr == -1 and r < br0:
                            outward += 1
                            is_outward = True

                        if col == marker_color:
                            if cav[r, c]:
                                marker_in_cavity += 1
                            if is_outward:
                                marker_outward += 1

                    if adj == 0:
                        continue

                    for plug_color in non_marker_colors:
                        plug_hits = 0
                        plug_outside = 0
                        nonplug_in_cavity = 0
                        marker_count_here = 0
                        handle_outward = 0
                        handle_bbox_inside = 0

                        for rr, cc in coords:
                            r = int(r0 + rr)
                            c = int(c0 + cc)
                            col = int(p[rr, cc])
                            in_cav = bool(cav[r, c])
                            in_frame_bbox = br0 <= r <= br1 and bc0 <= c <= bc1
                            is_outward = (
                                (odc == 1 and c > bc1)
                                or (odc == -1 and c < bc0)
                                or (odr == 1 and r > br1)
                                or (odr == -1 and r < br0)
                            )
                            if col == plug_color:
                                if in_cav:
                                    plug_hits += 1
                                else:
                                    plug_outside += 1
                            else:
                                if in_cav:
                                    nonplug_in_cavity += 1
                                if is_outward:
                                    handle_outward += 1
                                elif in_frame_bbox:
                                    handle_bbox_inside += 1
                                if col == marker_color:
                                    marker_count_here += 1

                        plug_full = int(
                            plug_hits == cav_sz and plug_outside == 0 and nonplug_in_cavity == 0
                        )
                        score = (
                            int(plug_full),
                            int(plug_hits),
                            -int(nonplug_in_cavity),
                            -int(plug_outside),
                            int(handle_outward),
                            -int(handle_bbox_inside),
                            int(marker_count_here == grp["branch_count"]),
                            int(marker_outward),
                            -int(marker_in_cavity),
                            int(cavity_hits),
                            int(socket_hits),
                            int(adj),
                            -int(bbox_gap((r0, c0, r0 + h - 1, c0 + w - 1), grp["bbox"])),
                            -int(r0),
                            -int(c0),
                        )
                        if best_score is None or score > best_score:
                            best_score = score
                            best_state = (p.copy(), m.copy(), int(r0), int(c0), bool(is_flip), int(plug_color))

        assert best_state is not None
        p, m, r0, c0, is_flip, plug_color = best_state
        other_colors = [c for c in non_marker_colors if c != plug_color]
        assert len(other_colors) == 1
        handle_color = int(other_colors[0])
        if is_flip and counts_nm[plug_color] != counts_nm[handle_color]:
            q = p.copy()
            plug_mask = (p == plug_color) & m
            handle_mask = (p == handle_color) & m
            q[plug_mask] = handle_color
            q[handle_mask] = plug_color
            p = q

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
NUM_PAIRS_CHOICES = [2]
UNION_BBOX_GAP_MIN = 3
FRAME_BBOX_GAP_MIN = 4
TOOL_BORDER = 1
TOOL_PADDING = 3
GEN_ATTEMPT_RETRIES = 400

# All cavity templates have area 5, so branch-count rather than area becomes the visible pairing key.
CAVITY_TEMPLATES = {
    2: [(1, 2), (2, 2), (2, 3), (2, 4), (3, 4)],
    3: [(1, 2), (2, 2), (3, 2), (2, 3), (2, 4)],
    4: [(2, 2), (1, 3), (2, 3), (3, 3), (2, 4)],
}

FRAME_H = 5
FRAME_W = 5
HANDLE_CELLS = [(2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (1, 6), (3, 6), (1, 8), (3, 8)]
MARKER_SLOTS = [(1, 6), (3, 6), (1, 8), (3, 8)]
SCENE_H = 5
SCENE_W = 10

LAST_GENERATED_META: dict | None = None


def _bbox_gap(a, b) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def gap1(x0, x1, y0, y1):
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)


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


def _swap_colors(arr: np.ndarray, a: int, b: int) -> np.ndarray:
    out = arr.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _variant_choices(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    seen = set()
    out: list[tuple[np.ndarray, bool]] = []
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


def _branch_count(mask: np.ndarray) -> int:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    leaves = 0
    for r, c in coords:
        deg = sum((r + dr, c + dc) in coords for dr, dc in DIR4)
        if deg == 1:
            leaves += 1
    return int(leaves)


def _build_left_pair(
    *,
    bg: int,
    frame_color: int,
    plug_color: int,
    handle_color: int,
    marker_color: int,
    branch_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((SCENE_H, SCENE_W), bg, dtype=int)
    frame[:, :FRAME_W] = int(frame_color)

    for r, c in CAVITY_TEMPLATES[int(branch_count)]:
        frame[r, c] = int(bg)

    tool = np.full((SCENE_H, SCENE_W), bg, dtype=int)
    for r, c in CAVITY_TEMPLATES[int(branch_count)]:
        tool[r, c] = int(plug_color)

    marker_slots = MARKER_SLOTS[: int(branch_count)]
    for r, c in HANDLE_CELLS:
        tool[r, c] = int(handle_color)
    for r, c in marker_slots:
        tool[r, c] = int(marker_color)

    return frame, tool


def _candidate_origins(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
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


def _place_pair(
    *,
    side: str,
    branch_count: int,
    colors: tuple[int, int, int, int],
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list,
    frame_bboxes: list,
) -> dict | None:
    frame_color, plug_color, handle_color, marker_color = colors
    frame_left, tool_left = _build_left_pair(
        bg=bg,
        frame_color=frame_color,
        plug_color=plug_color,
        handle_color=handle_color,
        marker_color=marker_color,
        branch_count=branch_count,
    )
    scene_left = np.where(tool_left != bg, tool_left, frame_left)
    frame_scene = _orient_left_to_side(frame_left, side)
    tool_scene = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)

    ph, pw = scene.shape
    fb = _bbox(frame_scene != bg)
    assert fb is not None
    fr0, fc0, fr1, fc1 = fb

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        frame_bbox = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)
        if any(_bbox_gap(union_bbox, prev) < UNION_BBOX_GAP_MIN for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) < FRAME_BBOX_GAP_MIN for prev in frame_bboxes):
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
    union_bboxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    frame_bboxes.append((r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1))
    return {
        "tool": tool_scene,
        "tool_mask": tool_scene != int(bg),
        "plug_color": int(plug_color),
        "handle_color": int(handle_color),
        "marker_color": int(marker_color),
        "plug_area": int(np.sum(tool_scene == int(plug_color))),
        "handle_area": int(np.sum(tool_scene == int(handle_color))),
        "marker_count": int(np.sum(tool_scene == int(marker_color))),
        "branch_count": int(branch_count),
        "frame_bbox": (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1),
        "frame_color": int(frame_color),
    }


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    for patch, is_flip in _variant_choices(info["tool"], info["tool_mask"]):
        q = patch.copy()
        if is_flip and int(info["plug_area"]) != int(info["handle_area"]):
            q = _swap_colors(q, info["plug_color"], info["handle_color"])
        try:
            r0, c0 = random_free_location(
                placement_canvas,
                q,
                transparent=bg,
                border=TOOL_BORDER,
                padding=TOOL_PADDING,
                padding_connectivity=4,
            )
        except ValueError:
            continue
        paste(input_grid, q, r0, c0, transparent=bg)
        paste(placement_canvas, q, r0, c0, transparent=bg)
        info["input_flip_used"] = bool(is_flip)
        return True
    return False


def analyze_input(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = get_bg(g)
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    frames = []
    tools = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        touches = any(
            int(r) == 0 or int(r) == H - 1 or int(c) == 0 or int(c) == W - 1
            for r, c in pts
        )
        if touches:
            sub_bg = patch == int(bg)
            sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
            best_id = None
            best = None
            for sid in range(1, sub_n + 1):
                pts2 = np.argwhere(sub_lab == sid)
                area = int(len(pts2))
                edge_touch = any(
                    int(rr) == 0 or int(rr) == patch.shape[0] - 1 or int(cc) == 0 or int(cc) == patch.shape[1] - 1
                    for rr, cc in pts2
                )
                cand = (int(edge_touch), area, -sid)
                if best is None or cand > best:
                    best = cand
                    best_id = sid
            if best_id is None:
                continue
            cav = sub_lab == best_id
            frames.append(
                {
                    "bbox": bbox,
                    "cavity_area": int(np.sum(cav)),
                    "branch_count": _branch_count(cav),
                }
            )
        else:
            vals = [int(x) for x in patch.flatten() if int(x) != int(bg)]
            counts = Counter(vals)
            marker_count = min(counts.values()) if counts else 0
            tools.append({"bbox": bbox, "marker_count": int(marker_count), "color_counts": dict(counts)})
    frames.sort(key=lambda d: d["bbox"])
    tools.sort(key=lambda d: d["bbox"])
    return {
        "num_frames": len(frames),
        "num_tools": len(tools),
        "frames": frames,
        "tools": tools,
        "ambiguous_by_area": len(frames) >= 2 and len({f["cavity_area"] for f in frames}) < len(frames),
        "distinct_by_branch": len(frames) >= 2 and len({f["branch_count"] for f in frames}) == len(frames),
    }


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_pairs = random.choice(NUM_PAIRS_CHOICES)
    branch_pool = [2, 3, 4]
    if num_pairs == 2:
        branch_counts = random.sample(branch_pool, k=2)
    else:
        branch_counts = [random.choice(branch_pool)]

    colors = [c for c in range(10) if c != bg]
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

    for i, bc in enumerate(branch_counts):
        palette = tuple(colors[4 * i : 4 * i + 4])
        info = _place_pair(
            side=sides[i],
            branch_count=int(bc),
            colors=palette,
            bg=bg,
            H=H,
            W=W,
            output=output,
            input_grid=input_grid,
            union_bboxes=union_bboxes,
            frame_bboxes=frame_bboxes,
        )
        if info is None:
            return None
        placed_tools.append(info)

    placement_canvas = output.copy()
    for info in placed_tools:
        if not _undock_tool(info, bg, input_grid, placement_canvas):
            return None

    solved = main(input_grid)
    if not np.array_equal(solved, output):
        return None

    meta = analyze_input(input_grid)
    if int(meta["num_frames"]) != 2 or int(meta["num_tools"]) != 2:
        return None
    if not meta["ambiguous_by_area"] or not meta["distinct_by_branch"]:
        return None
    meta.update(
        {
            "num_pairs": int(num_pairs),
            "bg": int(bg),
            "branch_counts": [int(x) for x in branch_counts],
            "plug_areas": [int(info["plug_area"]) for info in placed_tools],
            "handle_areas": [int(info["handle_area"]) for info in placed_tools],
            "input_flips": [bool(info.get("input_flip_used", False)) for info in placed_tools],
            "has_flip_swap_case": any(
                bool(info.get("input_flip_used", False)) and int(info["plug_area"]) != int(info["handle_area"])
                for info in placed_tools
            ),
        }
    )
    return input_grid, meta


def generate_input() -> np.ndarray:
    global LAST_GENERATED_META
    for _ in range(GEN_ATTEMPT_RETRIES):
        result = _gen_attempt()
        if result is None:
            continue
        grid, meta = result
        LAST_GENERATED_META = meta
        return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
