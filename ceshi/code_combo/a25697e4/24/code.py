# Combo variant of a25697e4 (index 24)
# combo_spec: D3/2+D9/0+D11/0
# Name: 分离工具有序拼装

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, paste, random_free_location

# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior:
#    The original task had one connected 2-color tool per frame. Here the input has one
#    large border frame and 2-3 movable tools, but each tool is split into two nearby
#    disconnected monochrome pieces: a staircase plug and a handle. In the output those
#    same two pieces move together as one rigid patch, so the plug enters the shared
#    chamber and the handle protrudes outward. The tools are inserted into one common
#    cavity from larger total area to smaller total area, occupying successive 3-row bands.
# b) Complexity calibration:
#    The chamber is shared rather than independent per tool, so order is visible. The
#    two-piece grouping is visible because each plug/handle pair is close, while different
#    tools are generated with much larger spacing. Tool totals are strictly descending
#    (12/10/8), keeping the ranking observable and preventing tie logic.
# c) Hidden-assumption audit:
#    The original solver assumed one non-border connected component per tool and 1:1
#    frame-tool matching. This version removes both assumptions. The new solver therefore:
#    - clusters non-border components into proximity pairs,
#    - treats a paired disconnected patch as one rigid tool under D4 transforms,
#    - uses one frame with one shared cavity,
#    - places tools sequentially into cavity bands according to descending total area.
#    The generator guarantees the assumptions used by the new solver: exactly one frame
#    touches exactly one border, every tool consists of exactly two components with bbox
#    gap <= 2, tools are mutually well separated, and the cavity height is 3*k.
# d) Placement legality:
#    Output is built first in canonical left-border orientation, then input is produced by
#    removing the docked tools and scattering transformed tool patches into free space with
#    border and padding constraints. Final input/output are jointly rotated to one border
#    side, so there are no out-of-bounds placements and no overlaps.
# e) Visual observability:
#    Humans can see one big frame, several loose two-part tools, and the output stacks
#    them into the same chamber from large to small. The ranking depends on visible pixel
#    area only; no color-id ordering or other non-visual property is used.
# f) Clean visualization:
#    The frame is a single compact object. Each tool uses exactly two colors and one local
#    patch, so the scene stays readable. Padding keeps different tools from visually merging.
# g) Inferable color behavior:
#    Reflection-triggered color swap from the source task is preserved, but only as a
#    structural consequence of the chosen transform. Each tool still has a stable plug
#    shape and handle shape, so the geometry provides the clue for which color is the plug.


FRAME_W = 5
SLOT_H = 3
PLUG_W = 3
TOOL_GAP = 1
HANDLE_W = 2
TOOL_W = PLUG_W + TOOL_GAP + HANDLE_W
GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (24, 30)
NUM_TOOLS_CHOICES = [2, 3]
FRAME_TOP_BOTTOM_MARGIN = 1
TOOL_BORDER = 1
TOOL_PADDING = 4
GEN_ATTEMPT_RETRIES = 300

PLUG_MASKS = [
    np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=bool),  # area 7
    np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),  # area 6
    np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=bool),  # area 5
]
HANDLE_MASKS = [
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),  # area 5
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),  # area 4
    np.array([[1, 1], [1, 0], [0, 0]], dtype=bool),  # area 3
]


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


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    if flip:
        out = np.fliplr(out)
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


def _detect_side(grid: np.ndarray, bg: int) -> str:
    fg = grid != int(bg)
    scores = {
        "left": int(fg[:, 0].sum()),
        "right": int(fg[:, -1].sum()),
        "top": int(fg[0, :].sum()),
        "bottom": int(fg[-1, :].sum()),
    }
    side = max(scores, key=scores.get)
    if scores[side] <= 0:
        raise ValueError("no border frame detected")
    return side


def _to_left(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return grid.copy()
    if side == "right":
        return np.fliplr(grid)
    if side == "top":
        return np.rot90(grid, 1)
    if side == "bottom":
        return np.rot90(grid, 3)
    raise ValueError(side)


def _from_left(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return grid.copy()
    if side == "right":
        return np.fliplr(grid)
    if side == "top":
        return np.rot90(grid, 3)
    if side == "bottom":
        return np.rot90(grid, 1)
    raise ValueError(side)


def _extract_frame_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    fg = grid != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    frame = np.zeros_like(fg, dtype=bool)
    H, W = grid.shape
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        touches = np.any(
            (pts[:, 0] == 0) | (pts[:, 0] == H - 1) | (pts[:, 1] == 0) | (pts[:, 1] == W - 1)
        )
        if touches:
            frame |= lab == cid
    if not frame.any():
        raise ValueError("frame not found")
    return frame


def _largest_adjacent_cavity(grid: np.ndarray, bg: int, frame_mask: np.ndarray) -> np.ndarray:
    bbox = _mask_bbox(frame_mask)
    r0, c0, r1, c1 = bbox
    sub_bg = grid[r0 : r1 + 1, c0 : c1 + 1] == int(bg)
    sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
    best = None
    best_mask = None
    for cid in range(1, sub_n + 1):
        sub = sub_lab == cid
        if not sub.any():
            continue
        pts = np.argwhere(sub)
        adj = False
        for rr, cc in pts:
            gr = int(r0 + rr)
            gc = int(c0 + cc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r2 = gr + dr
                c2 = gc + dc
                if r0 <= r2 <= r1 and c0 <= c2 <= c1 and frame_mask[r2, c2]:
                    adj = True
                    break
            if adj:
                break
        if not adj:
            continue
        score = int(sub.sum())
        if best is None or score > best:
            best = score
            best_mask = sub
    if best_mask is None:
        raise ValueError("cavity not found")
    cavity = np.zeros_like(frame_mask, dtype=bool)
    cavity[r0 : r1 + 1, c0 : c1 + 1] = best_mask
    return cavity


def _extract_tool_groups(grid: np.ndarray, bg: int, frame_mask: np.ndarray) -> list[dict]:
    mov_mask = (grid != int(bg)) & (~frame_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    comps = []
    for cid in range(1, mov_n + 1):
        mask = mov_lab == cid
        pts = np.argwhere(mask)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        local_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~local_mask] = 0
        colors = sorted({int(x) for x in patch[local_mask]})
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": local_mask,
                "patch": patch,
                "size": int(local_mask.sum()),
                "colors": colors,
            }
        )
    if not comps:
        raise ValueError("tools not found")

    parent = list(range(len(comps)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if _bbox_gap(comps[i]["bbox"], comps[j]["bbox"]) <= 2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(comps)):
        groups.setdefault(find(i), []).append(i)

    tools: list[dict] = []
    for idxs in groups.values():
        if len(idxs) != 2:
            raise ValueError("each tool must have exactly two nearby pieces")
        r0 = min(comps[i]["bbox"][0] for i in idxs)
        c0 = min(comps[i]["bbox"][1] for i in idxs)
        r1 = max(comps[i]["bbox"][2] for i in idxs)
        c1 = max(comps[i]["bbox"][3] for i in idxs)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        mask = np.zeros_like(patch, dtype=bool)
        for i in idxs:
            br0, bc0, br1, bc1 = comps[i]["bbox"]
            rr0 = br0 - r0
            cc0 = bc0 - c0
            h, w = comps[i]["patch"].shape
            local_mask = comps[i]["mask"]
            patch[rr0 : rr0 + h, cc0 : cc0 + w][local_mask] = comps[i]["patch"][local_mask]
            mask[rr0 : rr0 + h, cc0 : cc0 + w] |= local_mask
        colors = sorted({int(x) for x in patch[mask]})
        if len(colors) != 2:
            raise ValueError("tool must use exactly two colors")
        tools.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch,
                "mask": mask,
                "size": int(mask.sum()),
                "color_counts": dict(Counter(int(x) for x in patch[mask])),
            }
        )
    return tools


def _solve_canonical(left_grid: np.ndarray, reverse_order: bool = False) -> np.ndarray:
    g = left_grid.copy()
    bg = get_bg(g)
    H, W = g.shape
    frame_mask = _extract_frame_mask(g, bg)
    frame_bbox = _mask_bbox(frame_mask)
    cavity = _largest_adjacent_cavity(g, bg, frame_mask)
    cav_bbox = _mask_bbox(cavity)
    cav_h = cav_bbox[2] - cav_bbox[0] + 1
    cav_w = cav_bbox[3] - cav_bbox[1] + 1
    if cav_w != PLUG_W or cav_h % SLOT_H != 0:
        raise ValueError("unexpected chamber geometry")
    num_slots = cav_h // SLOT_H

    tools = _extract_tool_groups(g, bg, frame_mask)
    if len(tools) != num_slots:
        raise ValueError("tool count must match cavity bands")
    tools.sort(key=lambda d: (-d["size"], d["bbox"][0], d["bbox"][1]))
    if reverse_order:
        tools = list(reversed(tools))

    out = np.full_like(g, int(bg))
    out[frame_mask] = g[frame_mask]
    placed = frame_mask.copy()

    for slot_idx, tool in enumerate(tools):
        target_r0 = cav_bbox[0] + SLOT_H * slot_idx
        target_c0 = cav_bbox[1]
        target_band = np.zeros_like(frame_mask, dtype=bool)
        target_band[target_r0 : target_r0 + SLOT_H, target_c0 : target_c0 + PLUG_W] = True

        best_state = None
        best_score = None
        for patch, mask, is_flip in _gen_variants(tool["patch"], tool["mask"]):
            h, w = mask.shape
            coords = np.argwhere(mask)
            colors = sorted({int(x) for x in patch[mask]})
            for r0 in range(0, H - h + 1):
                for c0 in range(0, W - w + 1):
                    overlap = False
                    for rr, cc in coords:
                        if placed[int(r0 + rr), int(c0 + cc)]:
                            overlap = True
                            break
                    if overlap:
                        continue

                    for plug_color in colors:
                        plug_inside_target = 0
                        plug_outside_target = 0
                        plug_outside_cavity = 0
                        handle_outside_right = 0
                        handle_inside_frame = 0
                        handle_in_cavity = 0
                        for rr, cc in coords:
                            gr = int(r0 + rr)
                            gc = int(c0 + cc)
                            is_plug = int(patch[rr, cc]) == int(plug_color)
                            in_target = bool(target_band[gr, gc])
                            in_cavity = bool(cavity[gr, gc])
                            if is_plug:
                                if in_target:
                                    plug_inside_target += 1
                                else:
                                    plug_outside_target += 1
                                if not in_cavity:
                                    plug_outside_cavity += 1
                            else:
                                if gc > frame_bbox[3]:
                                    handle_outside_right += 1
                                else:
                                    handle_inside_frame += 1
                                if in_cavity:
                                    handle_in_cavity += 1
                        plug_full = int(plug_outside_target == 0 and plug_outside_cavity == 0 and plug_inside_target > 0)
                        handle_full = int(handle_inside_frame == 0 and handle_in_cavity == 0 and handle_outside_right > 0)
                        score = (
                            plug_full,
                            handle_full,
                            int(plug_inside_target),
                            int(handle_outside_right),
                            -int(plug_outside_target),
                            -int(plug_outside_cavity),
                            -int(handle_inside_frame),
                            -int(handle_in_cavity),
                            -int(r0),
                            -int(c0),
                        )
                        if best_score is None or score > best_score:
                            best_score = score
                            best_state = (patch, mask, int(r0), int(c0), bool(is_flip))

        if best_state is None or best_score[0] != 1 or best_score[1] != 1:
            raise ValueError("no valid ordered placement found")

        patch, mask, r0, c0, is_flip = best_state
        color_counts = tool["color_counts"]
        if is_flip and len(color_counts) == 2 and len(set(color_counts.values())) == 2:
            a, b = sorted(color_counts)
            patch = _swap_colors(patch, a, b)
        for rr, cc in np.argwhere(mask):
            gr = int(r0 + rr)
            gc = int(c0 + cc)
            out[gr, gc] = int(patch[rr, cc])
            placed[gr, gc] = True

    return out


def solve_with_order(grid: np.ndarray, reverse_order: bool = False) -> np.ndarray:
    bg = get_bg(grid)
    side = _detect_side(grid, bg)
    left = _to_left(grid, side)
    solved_left = _solve_canonical(left, reverse_order=reverse_order)
    return _from_left(solved_left, side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble several split tools into one shared border frame.

    Input structure:
    - The dominant color is background.
    - One border-touching frame contains a single tall chamber that spans several 3-row bands.
    - Away from the border sit 2-3 movable tools; each tool is a nearby pair of disconnected
      monochrome pieces (plug + handle) that must be grouped before moving.

    Rule:
    - Treat each plug/handle pair as one rigid tool under rotation/flip.
    - Sort the tools by total area from large to small and dock them into the chamber from the
      first band to the last, with the plug inside the chamber and the handle protruding outward.
    - If the winning docking transform uses a reflection and the two tool colors have unequal
      counts, swap the two colors after docking.

    Key insight: the frame is shared, so the tools are not matched independently; first recover
    each split tool, then use visible area ranking to determine the ordered fill of the common slot.
    """
    return solve_with_order(grid, reverse_order=False)


def _build_frame_patch(num_tools: int, bg: int, frame_color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    fh = SLOT_H * num_tools + 2
    frame = np.full((fh, FRAME_W), bg, dtype=int)
    frame[0, :] = frame_color
    frame[-1, :] = frame_color
    frame[:, :2] = frame_color
    cavity_bbox = (1, 2, fh - 2, FRAME_W - 1)
    return frame, cavity_bbox


def _build_tool_patch(plug_color: int, handle_color: int, plug_mask: np.ndarray, handle_mask: np.ndarray, bg: int) -> np.ndarray:
    patch = np.full((SLOT_H, TOOL_W), bg, dtype=int)
    patch[:, :PLUG_W][plug_mask] = int(plug_color)
    patch[:, PLUG_W + TOOL_GAP :][handle_mask] = int(handle_color)
    return patch


def _unique_variants_for_patch(patch: np.ndarray, bg: int) -> list[tuple[np.ndarray, bool]]:
    mask = patch != int(bg)
    variants = []
    seen = set()
    for p, m, is_flip in _gen_variants(patch, mask):
        key = (m.tobytes(), p[m].tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append((p, bool(is_flip)))
    random.shuffle(variants)
    return variants


def _rotate_pair(left_input: np.ndarray, left_output: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    return _from_left(left_input, side), _from_left(left_output, side)


def _gen_one() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    num_tools = random.choice(NUM_TOOLS_CHOICES)
    side = random.choice(["left", "right", "top", "bottom"])

    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    need = 1 + 2 * num_tools
    if len(palette) < need:
        return None

    frame_color = palette[0]
    frame_patch, cavity_bbox = _build_frame_patch(num_tools, bg, frame_color)
    fh, fw = frame_patch.shape
    if H < fh + 2 or W < FRAME_W + TOOL_W + 3:
        return None
    frame_r0 = random.randint(FRAME_TOP_BOTTOM_MARGIN, H - fh - FRAME_TOP_BOTTOM_MARGIN)
    frame_c0 = 0

    left_output = np.full((H, W), bg, dtype=int)
    left_input = np.full((H, W), bg, dtype=int)
    paste(left_output, frame_patch, frame_r0, frame_c0, transparent=bg)
    paste(left_input, frame_patch, frame_r0, frame_c0, transparent=bg)

    tool_infos = []
    for i in range(num_tools):
        plug_color = palette[1 + 2 * i]
        handle_color = palette[1 + 2 * i + 1]
        patch = _build_tool_patch(plug_color, handle_color, PLUG_MASKS[i], HANDLE_MASKS[i], bg)
        dock_r0 = frame_r0 + 1 + SLOT_H * i
        dock_c0 = frame_c0 + 2
        paste(left_output, patch, dock_r0, dock_c0, transparent=bg)
        tool_infos.append(
            {
                "patch": patch,
                "plug_color": plug_color,
                "handle_color": handle_color,
                "plug_area": int(PLUG_MASKS[i].sum()),
                "handle_area": int(HANDLE_MASKS[i].sum()),
            }
        )

    placement_canvas = left_input.copy()
    for info in tool_infos:
        placed = False
        for patch, is_flip in _unique_variants_for_patch(info["patch"], bg):
            q = patch
            if is_flip and info["plug_area"] != info["handle_area"]:
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
            paste(left_input, q, r0, c0, transparent=bg)
            paste(placement_canvas, q, r0, c0, transparent=bg)
            placed = True
            break
        if not placed:
            return None

    inp, out = _rotate_pair(left_input, left_output, side)
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        result = _gen_one()
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError("failed to generate valid input")
