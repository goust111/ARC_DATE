# Combo variant of a25697e4 (index 9)
# combo_spec: D9/0+D11/0
# Name: 多工具有序填装
#
# Design analysis:
# - Concrete fused scene:
#   One border frame owns one large staircase cavity, not one cavity per tool.
#   The cavity is split into 2-3 row-like plug slots of strictly decreasing sizes.
#   Each free tool has exactly two colors: a bar-shaped plug and a short handle.
#   In the output all tools dock into the same frame; each plug fills one slot and
#   each handle protrudes away from the frame border.
# - Behavior trace versus the base task:
#   Base task: one tool matches one frame because one color count equals the full
#   cavity size. Order is irrelevant because each pair is independent.
#   Combo task: no tool color matches the full cavity size. Instead several tools
#   collectively cover the cavity, so the solver must reuse the same frame and place
#   tools one after another. The large tool is resolved first, then the remaining
#   cavity shrinks for the smaller tools.
# - Complexity calibration:
#   The family is intentionally narrow: one frame, 2 or 3 tools, strictly decreasing
#   plug sizes, and one visible outward direction. This keeps the rule readable while
#   still making the many:1 change obvious to a human.
# - Hidden-assumption audit:
#   The original solver assumes one cavity per anchor group and near one-to-one
#   tool-to-frame assignment. That is broken here, so `main()` keeps the original
#   frame detection but changes placement to a sequential "consume remaining cavity"
#   search on the largest anchor group.
# - Legality and cleanliness:
#   The generator builds the docked output first, then undocks each tool to a distant
#   interior location with padding. Handles are assigned to different cavity rows so
#   docked tools do not overlap and remain visually separable.
# - Observability:
#   Humans can see one frame plus multiple free tools in the input and the same frame
#   containing all tools in the output. Strictly decreasing tool areas make the
#   placement order visible without relying on color ids or any hidden metadata.

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, paste, random_free_location


GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (26, 30)
NUM_TOOLS_CHOICES = [2, 3]
TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4
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


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
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
    mask = patch != int(bg)
    for k in range(4):
        for flip in (False, True):
            q = _transform(patch, k, flip)
            m = _transform(mask.astype(np.uint8), k, flip).astype(bool)
            key = (q.shape, m.shape, m.tobytes(), q[m].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((q, bool(flip)))
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
        key = (p.shape, m.shape, m.tobytes(), p[m].tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m, bool(is_flip)))
    return uniq


def _extract_anchor_groups(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        if pts.size == 0:
            continue
        touches = any(
            int(r) == 0 or int(r) == H - 1 or int(c) == 0 or int(c) == W - 1
            for r, c in pts
        )
        if not touches:
            continue
        fixed_components.append(
            {
                "pts": [(int(r), int(c)) for r, c in pts],
                "bbox": (
                    int(pts[:, 0].min()),
                    int(pts[:, 1].min()),
                    int(pts[:, 0].max()),
                    int(pts[:, 1].max()),
                ),
            }
        )
        for r, c in pts:
            fixed_mask[int(r), int(c)] = True
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
            mask[int(r), int(c)] = True

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
        best_c = None
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
            if best_c is None or cand > best_c:
                best_c = cand
                best_id = cid
        assert best_id is not None
        ptsc = np.argwhere(sub_lab == best_id)
        for rr, cc in ptsc:
            cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        if bbox[1] == 0:
            open_dir = (0, 1)
        elif bbox[3] == W - 1:
            open_dir = (0, -1)
        elif bbox[0] == 0:
            open_dir = (1, 0)
        elif bbox[2] == H - 1:
            open_dir = (-1, 0)
        else:
            raise AssertionError("anchor group must touch exactly one outer border")

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
    return anchor_groups


def _extract_movable_components(g: np.ndarray, bg: int, fixed_mask: np.ndarray) -> list[dict]:
    mov_mask = (g != int(bg)) & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps = []
    for cid in range(1, mov_n + 1):
        pts = np.argwhere(mov_lab == cid)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        m = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == cid
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~m] = 0
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": m,
                "patch": patch,
                "size": int(len(pts)),
            }
        )
    mov_comps.sort(key=lambda d: -d["size"])
    return mov_comps


def _extract_slots(cavity: np.ndarray, open_dir: tuple[int, int]) -> list[dict]:
    H, W = cavity.shape
    slots = []
    if open_dir in ((0, -1), (0, 1)):
        for r in range(H):
            cols = np.where(cavity[r])[0]
            if len(cols) == 0:
                continue
            c0, c1 = int(cols.min()), int(cols.max())
            slot_mask = np.zeros_like(cavity, dtype=bool)
            slot_mask[r, c0 : c1 + 1] = cavity[r, c0 : c1 + 1]
            coords = [(int(r), int(c)) for c in range(c0, c1 + 1) if cavity[r, c]]
            slots.append({"mask": slot_mask, "size": len(coords), "anchor": (int(r), int(c1))})
    else:
        for c in range(W):
            rows = np.where(cavity[:, c])[0]
            if len(rows) == 0:
                continue
            r0, r1 = int(rows.min()), int(rows.max())
            slot_mask = np.zeros_like(cavity, dtype=bool)
            slot_mask[r0 : r1 + 1, c] = cavity[r0 : r1 + 1, c]
            coords = [(int(r), int(c)) for r in range(r0, r1 + 1) if cavity[r, c]]
            slots.append({"mask": slot_mask, "size": len(coords), "anchor": (int(r1), int(c))})
    slots.sort(key=lambda d: (-int(d["size"]), int(d["anchor"][0]), int(d["anchor"][1])))
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock several free tools into one border frame by consuming a shared cavity.

    Input structure:
    - The background is the most frequent color.
    - Border-touching same-color pieces cluster into one anchor frame with one large
      staircase cavity.
    - Away from the border there are 2 or 3 separate 2-color tools.

    Rule:
    - All tools belong to the same frame rather than to separate frames.
    - Process the tools from larger total area to smaller total area.
    - For each tool, rotate or flip it so one color region fills part of the still
      empty cavity and the other color region sticks outward from the frame.
    - If the chosen docking transform is a reflection and the two color counts are
      unequal, swap the two tool colors before painting the docked result.

    Key insight:
    The cavity is not matched tool-by-tool by its full size anymore; it is consumed
    sequentially. The largest tool must be placed first so later tools fit only in
    the remaining cavity cells of that same frame.
    """
    g = grid.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    anchor_groups = _extract_anchor_groups(g, bg)
    grp = anchor_groups[0]
    fixed_mask = grp["mask"].copy()
    for other in anchor_groups[1:]:
        fixed_mask |= other["mask"]

    mov_comps = _extract_movable_components(g, bg, fixed_mask)
    assert len(mov_comps) >= 2

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_grp = grp["mask"].copy()
    full_cavity = grp["cavity_mask"]
    odr, odc = grp["open_dir"]
    br0, bc0, br1, bc1 = grp["bbox"]

    for comp in mov_comps:
        remaining_cavity = full_cavity & (~placed_global)
        slots = _extract_slots(remaining_cavity, grp["open_dir"])
        assert slots
        slot = slots[0]
        slot_mask = slot["mask"]
        slot_size = int(slot["size"])
        comp_vals = [int(x) for x in comp["patch"][comp["mask"]].flatten()]
        comp_counts = Counter(comp_vals)
        colors = sorted(comp_counts)
        assert len(colors) == 2

        best_state = None
        best_score = None
        for p, m, is_flip in _gen_variants(comp["patch"], comp["mask"]):
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
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                adj += 1
                    if adj == 0:
                        continue

                    outside = 0
                    socket_hits = 0
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        if grp["socket_mask"][r, c]:
                            socket_hits += 1
                        if odc == -1 and c < bc0:
                            outside += 1
                        elif odc == 1 and c > bc1:
                            outside += 1
                        elif odr == -1 and r < br0:
                            outside += 1
                        elif odr == 1 and r > br1:
                            outside += 1

                    for plug_col in colors:
                        plug_local = (p == int(plug_col)) & m
                        plug_count = int(plug_local.sum())
                        if plug_count != slot_size:
                            continue
                        plug_hits = 0
                        plug_outside = 0
                        nonplug_in_cavity = 0
                        for rr, cc in coords:
                            r = int(r0 + rr)
                            c = int(c0 + cc)
                            in_cavity = bool(remaining_cavity[r, c])
                            if plug_local[rr, cc]:
                                if slot_mask[r, c]:
                                    plug_hits += 1
                                else:
                                    plug_outside += 1
                            elif in_cavity:
                                nonplug_in_cavity += 1

                        plug_full = int(plug_hits == slot_size and plug_outside == 0 and nonplug_in_cavity == 0)
                        bbox_here = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
                        dist = _bbox_gap(grp["bbox"], bbox_here)
                        score = (
                            int(plug_full),
                            int(plug_hits),
                            -int(nonplug_in_cavity),
                            -int(plug_outside),
                            int(outside),
                            int(socket_hits),
                            int(adj),
                            -int(dist),
                            -int(r0),
                            -int(c0),
                        )
                        if best_score is None or score > best_score:
                            best_score = score
                            best_state = {
                                "patch": p.copy(),
                                "mask": m.copy(),
                                "r0": int(r0),
                                "c0": int(c0),
                                "is_flip": bool(is_flip),
                                "counts": dict(comp_counts),
                            }

        assert best_state is not None
        p = best_state["patch"]
        m = best_state["mask"]
        if best_state["is_flip"] and len(set(best_state["counts"].values())) == 2:
            a, b = sorted(best_state["counts"])
            p = _swap_colors(p, a, b)
        for rr, cc in np.argwhere(m):
            r = int(best_state["r0"] + rr)
            c = int(best_state["c0"] + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True
            placed_grp[r, c] = True
    return out


def _slot_spec(num_tools: int) -> tuple[list[int], list[int]]:
    if num_tools == 2:
        return [7, 4], [4, 3]
    if num_tools == 3:
        return [8, 5, 4], [4, 3, 3]
    raise ValueError(num_tools)


def _build_left_scene(
    *, bg: int, frame_color: int, plug_colors: list[int], handle_colors: list[int], num_tools: int
) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    row_lengths, handle_lengths = _slot_spec(num_tools)
    fw = row_lengths[0]
    top_h = random.randint(1, 3)
    extra_bottom = random.randint(1, 2)
    bot_h = (num_tools - 1) + extra_bottom
    fh = top_h + 1 + bot_h
    max_handle = max(handle_lengths)

    frame_core = np.full((fh, fw), bg, dtype=int)
    frame_core[:top_h, :] = int(frame_color)
    frame_core[top_h + 1 :, :] = int(frame_color)
    cavity_rows = [top_h + i for i in range(num_tools)]
    frame_core[top_h, :] = int(bg)
    for i in range(1, num_tools):
        length = row_lengths[i]
        frame_core[top_h + i, fw - length : fw] = int(bg)

    frame = np.full((fh, fw + max_handle), bg, dtype=int)
    frame[:, :fw] = frame_core

    tools = []
    infos = []
    for i in range(num_tools):
        tool = np.full((fh, fw + max_handle), bg, dtype=int)
        row = cavity_rows[i]
        length = row_lengths[i]
        start = fw - length
        tool[row, start:fw] = int(plug_colors[i])
        handle_len = handle_lengths[i]
        tool[row, fw : fw + handle_len] = int(handle_colors[i])
        tools.append(tool)
        infos.append(
            {
                "tool": tool,
                "plug_color": int(plug_colors[i]),
                "handle_color": int(handle_colors[i]),
                "plug_area": int(length),
                "handle_area": int(handle_len),
                "total_area": int(length + handle_len),
            }
        )
    return frame, tools, infos


def _place_frame_scene(
    frame_left: np.ndarray,
    tools_left: list[np.ndarray],
    side: str,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
) -> list[dict] | None:
    frame_scene = _orient_left_to_side(frame_left, side)
    tool_scenes = [_orient_left_to_side(t, side) for t in tools_left]
    scene = frame_scene.copy()
    for tool in tool_scenes:
        paste(scene, tool, 0, 0, transparent=bg)
    ph, pw = scene.shape

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != bg) & (scene != bg)):
            continue
        if side in ("left", "right") and (r0 == 0 or r0 + ph - 1 == H - 1):
            continue
        if side in ("top", "bottom") and (c0 == 0 or c0 + pw - 1 == W - 1):
            continue
        origin = (r0, c0)
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, scene, r0, c0, transparent=bg)
    paste(input_grid, frame_scene, r0, c0, transparent=bg)

    result = []
    return_origin_tools = []
    for tool_scene in tool_scenes:
        patch = np.full((H, W), bg, dtype=int)
        paste(patch, tool_scene, r0, c0, transparent=bg)
        return_origin_tools.append(patch)

    for full_tool in return_origin_tools:
        rr, cc = np.argwhere(full_tool != bg).min(axis=0)
        rr1, cc1 = np.argwhere(full_tool != bg).max(axis=0)
        crop = full_tool[rr : rr1 + 1, cc : cc1 + 1].copy()
        result.append(crop)
    return result


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    for patch, is_flip in _variant_choices(info["tool"], bg):
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
    bg = random.randint(1, 9)
    num_tools = random.choice(NUM_TOOLS_CHOICES)

    colors = [c for c in range(1, 10) if c != bg]
    random.shuffle(colors)
    need = 1 + 2 * num_tools
    if len(colors) < need:
        return None
    frame_color = colors[0]
    plug_colors = colors[1 : 1 + num_tools]
    handle_colors = colors[1 + num_tools : 1 + 2 * num_tools]

    frame_left, tools_left, infos = _build_left_scene(
        bg=bg,
        frame_color=frame_color,
        plug_colors=plug_colors,
        handle_colors=handle_colors,
        num_tools=num_tools,
    )

    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    side = random.choice(["left", "right", "top", "bottom"])
    placed_crops = _place_frame_scene(frame_left, tools_left, side, bg, H, W, output, input_grid)
    if placed_crops is None:
        return None

    for crop, info in zip(placed_crops, infos):
        info["tool"] = crop

    placement_canvas = output.copy()
    order = list(range(num_tools))
    random.shuffle(order)
    for idx in order:
        if not _undock_tool(infos[idx], bg, input_grid, placement_canvas):
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
