# Combo variant of a25697e4 (index 21)
# combo_spec: D1/0+D9/3
# Name: 干扰框架颜色排除
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Original task: each border frame was paired to one tool by cavity area, so the solver could
#      assign tools by "one tool color-count == one frame cavity size", then search a D4 docking
#      transform for that pair.
#    - Fused task here: all real frames share the same staircase cavity template, all tools share
#      the same 2-color geometry, and one extra distractor frame is added. Area and shape are now
#      deliberately non-diagnostic. The intended pair is the frame whose color matches the tool's
#      plug color. Example: if the free tools are {red+orange, green+orange} and the border frames
#      are red, green, and blue, then red-tool docks into red-frame, green-tool docks into
#      green-frame, and blue-frame stays unchanged.
#    - Difference: pairing logic moves from geometric counting to visible color agreement, while
#      the old docking geometry still matters after the pair is chosen.
# b) Complexity calibration:
#    - The scene uses only 1-2 real tools plus exactly 1 distractor frame, so the exclusion rule is
#      visible without making the grid noisy.
#    - All frames reuse one small notch template and all tools reuse one plug+handle template. This
#      ensures the new color cue is the only reliable pairing signal and prevents accidental fallback
#      to the source task's area-based rule.
# c) Hidden-assumption audit against the source solver:
#    - Broken assumption: "number of movable tools == number of frames". The fused solver must allow
#      one unmatched frame and leave it untouched.
#    - Broken assumption: "compatibility is determined by a color-region count equaling cavity area".
#      Now compatibility is "the tool contains the frame color", and the generator keeps handle
#      colors outside the frame palette so this is unambiguous.
#    - Preserved assumption: after a pair is fixed, one tool color-region still fills the cavity
#      exactly under some D4 transform, and the other region still protrudes outward.
#    - New solver assumption: every frame group is monochrome and every movable component has
#      exactly two visible colors. The generator constructs samples to satisfy this directly.
# d) Placement legality:
#    - The generator is output-first: it places docked frame+tool scenes on distinct borders,
#      records the frame-only input, then undocks each tool to an interior free location with
#      padding. This guarantees in-bounds, non-overlapping objects.
# e) Visual observability:
#    - The distractor is a full extra frame with the same notch geometry as the real frames, so a
#      human cannot solve by spotting a special shape. The only stable clue is color agreement
#      between frame and plug.
# f) Visual cleanliness:
#    - Each local region contains only frame color, plug color, one shared handle color, and
#      background. The tools are identical except for plug color, which keeps the comparison tidy.
# g) Inferability of color behavior:
#    - To keep the color-pairing cue human-readable, the generator undocks tools using rotations
#      only, so the visible plug color remains the frame-matching signal. The solver still retains
#      the original flip-triggered color-swap logic for robustness if a reflected input appears.

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox as _bbox, get_bg, paste, random_free_location


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _recolor_template(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = arr.copy()
    for src, dst in mapping.items():
        out[arr == int(src)] = int(dst)
    return out


def _build_left_templates() -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic left-border template.

    Frame color marker = 1, plug marker = 2, handle marker = 3, bg = 0.
    Cavity area is fixed at 6 for every frame, so color must drive the pairing.
    """
    core = np.zeros((5, 4), dtype=int)
    core[:2, :] = 1
    core[3:, :] = 1
    core[3, 2:] = 0

    frame = np.zeros((5, 6), dtype=int)
    frame[:, :4] = core

    tool = np.zeros((5, 6), dtype=int)
    cavity = core == 0
    tool[:5, :4][cavity] = 2
    tool[2:4, 4:6] = 3
    return frame, tool


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


def _rotation_variants(patch: np.ndarray) -> list[np.ndarray]:
    seen = set()
    out: list[np.ndarray] = []
    for k in range(4):
        q = np.rot90(patch, k)
        mask = q != 0
        key = (mask.tobytes(), q[mask].tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    random.shuffle(out)
    return out


def _extract_anchor_groups(g: np.ndarray, bg: int):
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components: list[dict] = []

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
                "color": int(g[pts_list[0][0], pts_list[0][1]]),
            }
        )
        for r, c in pts_list:
            fixed_mask[r, c] = True

    assert fixed_components, "must find border-touching frames"

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
            if (
                fixed_components[i]["color"] == fixed_components[j]["color"]
                and _bbox_gap(fixed_components[i]["bbox"], fixed_components[j]["bbox"]) <= 2
            ):
                union(i, j)

    buckets: dict[int, list[int]] = {}
    for i in range(len(fixed_components)):
        buckets.setdefault(find(i), []).append(i)

    groups: list[dict] = []
    for idxs in buckets.values():
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))

        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        frame_colors = sorted({int(x) for x in g[mask]} - {int(bg)})
        assert len(frame_colors) == 1, "each frame group must be monochrome"
        frame_color = int(frame_colors[0])

        socket = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(g[r, c]) != int(bg):
                    continue
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        socket[r, c] = True
                        break

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_id = None
        best_key = None
        for cid in range(1, int(sub_n) + 1):
            ptsc = np.argwhere(sub_lab == int(cid))
            area = int(len(ptsc))
            adj = False
            for rr, cc in ptsc:
                r = int(bbox[0] + rr)
                c = int(bbox[1] + cc)
                for dr, dc in DIR4:
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

        assert best_id is not None, "each frame must expose a cavity"
        ptsc = np.argwhere(sub_lab == int(best_id))
        for rr, cc in ptsc:
            cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        dir_counts = [0, 0, 0, 0]
        for r, c in pts:
            for idx, (dr, dc) in enumerate(DIR4):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(bg):
                    dir_counts[idx] += 1
        open_dir = DIR4[int(np.argmax(dir_counts))]

        groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "frame_color": frame_color,
                "socket_mask": socket,
                "cavity_mask": cavity,
                "cavity_size": int(cavity.sum()),
                "open_dir": open_dir,
                "size": int(mask.sum()),
            }
        )

    groups.sort(key=lambda d: -d["size"])
    return groups, fixed_mask


def _extract_movable_tools(g: np.ndarray, bg: int, fixed_mask: np.ndarray) -> list[dict]:
    fg = g != int(bg)
    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    tools: list[dict] = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == int(cid))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == int(cid)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = 0
        vals = [int(x) for x in patch[mask].flatten() if int(x) != 0]
        counts = Counter(vals)
        assert len(counts) == 2, "each movable tool must have exactly two colors"
        tools.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "colors": sorted(int(c) for c in counts),
                "counts": counts,
                "size": int(len(pts)),
            }
        )
    tools.sort(key=lambda d: -d["size"])
    return tools


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock detached color-coded tools into the matching border frames and ignore the extra frame.

    Input structure:
    - The background is the most common color.
    - Border-touching monochrome components cluster into staircase frames with interior bg cavities.
    - Non-border foreground components are movable two-color tools; all tools share the same shape,
      and one extra frame has no matching tool color.

    Rule:
    - Match a tool to the unique frame whose frame color appears on that tool.
    - Leave any frame whose color appears on no tool unchanged.
    - Rotate/flip the matched tool so that the frame-colored region fills the cavity exactly and the
      other region protrudes away from the frame.
    - If the winning docking transform uses a reflection and the two tool color-counts differ,
      swap the tool's two colors after docking.

    Key insight: Geometry only determines how a chosen tool docks; the pair itself is selected by
    frame color, and the unmatched frame is a distractor that must be excluded.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

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

    anchor_groups, fixed_mask = _extract_anchor_groups(g, bg)
    tools = _extract_movable_tools(g, bg, fixed_mask)

    color_to_group: dict[int, int] = {}
    for gi, grp in enumerate(anchor_groups):
        color_to_group[int(grp["frame_color"])] = int(gi)

    assignments: list[tuple[dict, int]] = []
    used_groups: set[int] = set()
    for comp in tools:
        matches = [color_to_group[col] for col in comp["colors"] if col in color_to_group]
        matches = [gi for gi in matches if gi not in used_groups]
        assert len(matches) == 1, "each tool must match exactly one frame color"
        gi = int(matches[0])
        used_groups.add(gi)
        assignments.append((comp, gi))

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    for comp, gi in assignments:
        grp = anchor_groups[gi]
        placed_grp = placed_groups[gi]
        target_color = int(grp["frame_color"])
        sock = grp["socket_mask"]
        cav = grp["cavity_mask"]
        odr, odc = grp["open_dir"]
        colors = list(comp["colors"])
        counts = comp["counts"]

        best_state = None
        best_score = None
        cav_sz = int(grp["cavity_size"])

        for p, m, is_flip in gen_variants(comp["patch"], comp["mask"]):
            h, w = m.shape
            coords = np.argwhere(m)

            placed_patch = p.copy()
            if is_flip and len(set(counts.values())) == 2:
                placed_patch = _swap_colors(placed_patch, colors[0], colors[1])

            for r0 in range(0, H - h + 1):
                for c0 in range(0, W - w + 1):
                    if any(placed_global[int(r0 + rr), int(c0 + cc)] for rr, cc in coords):
                        continue

                    adj = 0
                    socket_hits = 0
                    cavity_hits = 0
                    outside = 0
                    plug_hits = 0
                    plug_outside = 0
                    nonplug_in_cav = 0

                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)

                        for dr, dc in DIR4:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                adj += 1

                        if sock[r, c]:
                            socket_hits += 1
                        in_cav = bool(cav[r, c])
                        if in_cav:
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

                        col = int(placed_patch[rr, cc])
                        if col == target_color:
                            if in_cav:
                                plug_hits += 1
                            else:
                                plug_outside += 1
                        elif in_cav:
                            nonplug_in_cav += 1

                    if adj == 0:
                        continue

                    plug_full = int(plug_hits == cav_sz and plug_outside == 0 and nonplug_in_cav == 0)
                    dist = _bbox_gap(grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
                    score = (
                        int(plug_full),
                        int(plug_hits),
                        -int(nonplug_in_cav),
                        -int(plug_outside),
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
                        best_state = (placed_patch.copy(), m.copy(), int(r0), int(c0))

        assert best_state is not None, "must find a valid docking placement"
        patch, mask, r0, c0 = best_state
        for rr, cc in np.argwhere(mask):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(patch[rr, cc])
            placed_global[r, c] = True
            placed_grp[r, c] = True

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_TOOLS_CHOICES = [1, 2]
UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4
GEN_ATTEMPT_RETRIES = 300


def _place_border_patch(
    *,
    side: str,
    patch: np.ndarray,
    frame_patch: np.ndarray,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    ph, pw = patch.shape
    frame_bbox = _bbox(frame_patch != 0)
    assert frame_bbox is not None
    fr0, fc0, fr1, fc1 = frame_bbox

    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        abs_frame_bbox = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)

        if side in ("left", "right") and (abs_frame_bbox[0] == 0 or abs_frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (abs_frame_bbox[1] == 0 or abs_frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) <= UNION_BBOX_GAP_MIN - 1 for prev in union_bboxes):
            continue
        if any(_bbox_gap(abs_frame_bbox, prev) <= FRAME_BBOX_GAP_MIN - 1 for prev in frame_bboxes):
            continue
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != int(bg)) & (patch != 0)):
            continue

        paste(output, patch, int(r0), int(c0), transparent=0)
        paste(input_grid, frame_patch, int(r0), int(c0), transparent=0)
        union_bboxes.append(union_bbox)
        frame_bboxes.append(abs_frame_bbox)
        return abs_frame_bbox
    return None


def _occupied_bboxes(canvas: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    mask = canvas != int(bg)
    lab, n = label(mask, structure=STRUCT4)
    out = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        out.append(
            (
                int(pts[:, 0].min()),
                int(pts[:, 1].min()),
                int(pts[:, 0].max()),
                int(pts[:, 1].max()),
            )
        )
    return out


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    H, W = placement_canvas.shape
    occupied_bboxes = _occupied_bboxes(placement_canvas, bg)
    best = None
    best_score = None

    for patch in _rotation_variants(info["tool"]):
        ph, pw = patch.shape
        mask = patch != 0
        coords = np.argwhere(mask)

        for r0 in range(TOOL_BORDER, H - ph - TOOL_BORDER + 1):
            for c0 in range(TOOL_BORDER, W - pw - TOOL_BORDER + 1):
                ok = True
                for rr, cc in coords:
                    if int(placement_canvas[int(r0 + rr), int(c0 + cc)]) != int(bg):
                        ok = False
                        break
                if not ok:
                    continue

                bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
                if any(_bbox_gap(bbox, prev) < TOOL_PADDING for prev in occupied_bboxes):
                    continue

                dist_target = _bbox_gap(bbox, info["target_bbox"])
                dist_decoy = _bbox_gap(bbox, info["decoy_bbox"])
                score = (
                    int(dist_target > dist_decoy),
                    int(dist_target - dist_decoy),
                    -int(dist_decoy),
                    int(r0),
                    int(c0),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best = (patch.copy(), int(r0), int(c0))

    if best is None:
        for patch in _rotation_variants(info["tool"]):
            try:
                r0, c0 = random_free_location(
                    placement_canvas,
                    patch,
                    transparent=0,
                    border=TOOL_BORDER,
                    padding=TOOL_PADDING,
                    padding_connectivity=TOOL_PADDING_CONN,
                )
            except ValueError:
                continue
            best = (patch.copy(), int(r0), int(c0))
            break

    if best is None:
        return False

    patch, r0, c0 = best
    paste(input_grid, patch, int(r0), int(c0), transparent=0)
    paste(placement_canvas, patch, int(r0), int(c0), transparent=0)
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_tools = random.choice(NUM_TOOLS_CHOICES)

    colors = [c for c in range(1, 10) if c != bg]
    random.shuffle(colors)
    matched_frame_colors = colors[:num_tools]
    distractor_color = colors[num_tools]
    handle_color = colors[num_tools + 1]

    frame_template, tool_template = _build_left_templates()
    sides = random.sample(["left", "right", "top", "bottom"], k=num_tools + 1)

    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict] = []

    for idx, frame_color in enumerate(matched_frame_colors):
        frame_left = _recolor_template(frame_template, {1: int(frame_color)})
        tool_left = _recolor_template(tool_template, {2: int(frame_color), 3: int(handle_color)})
        scene_left = np.where(tool_left != 0, tool_left, frame_left)

        side = sides[idx]
        frame_scene = _orient_left_to_side(frame_left, side)
        tool_scene = _orient_left_to_side(tool_left, side)
        scene = _orient_left_to_side(scene_left, side)

        frame_bbox = _place_border_patch(
            side=side,
            patch=scene,
            frame_patch=frame_scene,
            bg=bg,
            H=H,
            W=W,
            output=output,
            input_grid=input_grid,
            union_bboxes=union_bboxes,
            frame_bboxes=frame_bboxes,
        )
        if frame_bbox is None:
            return None

        placed_tools.append(
            {
                "tool": tool_scene,
                "plug_color": int(frame_color),
                "handle_color": int(handle_color),
                "target_bbox": frame_bbox,
            }
        )

    distractor_left = _recolor_template(frame_template, {1: int(distractor_color)})
    distractor_side = sides[-1]
    distractor_scene = _orient_left_to_side(distractor_left, distractor_side)
    distractor_bbox = _place_border_patch(
        side=distractor_side,
        patch=distractor_scene,
        frame_patch=distractor_scene,
        bg=bg,
        H=H,
        W=W,
        output=output,
        input_grid=input_grid,
        union_bboxes=union_bboxes,
        frame_bboxes=frame_bboxes,
    )
    if distractor_bbox is None:
        return None

    placement_canvas = output.copy()
    for info in placed_tools:
        info["decoy_bbox"] = distractor_bbox
        if not _undock_tool(info, bg, input_grid, placement_canvas):
            return None

    if not np.array_equal(main(input_grid), output):
        return None
    return input_grid, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            grid, _ = sample
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
