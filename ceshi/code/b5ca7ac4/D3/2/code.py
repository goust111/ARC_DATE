# Variant of b5ca7ac4
# Dimension: D3 — 源对象——框架的结构与颜色组成 (src)
# Change: 每个源对象为5x5方框，外层1像素宽的边框为组标识色（青色8或红色2），内部3x3区域填充统一的其他颜色；所有框架尺寸相... → 矩形框架替代正方形框架

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior trace
#    - The old task had only 5x5 frames, so every detected object had the same
#      row-span (5) and the same inward shift (5 columns).
#    - This variant keeps the two frame colors and the "move to left/right edge"
#      rule, but each frame is now either a wide 5x7 rectangle or a tall 7x5
#      rectangle. The interior stays uniformly filled with a non-frame color.
#    - Example trace on the azure side:
#      old: edge square at rows 4..8 forces any later overlapping square to col 5.
#      new: if the edge frame is 5x7, the later overlapping rectangle moves to col 7;
#           if the edge frame is 7x5, the later overlapping rectangle moves to col 5.
#      The overlap test must also use the true height (5 or 7), not a hard-coded 5.
# b) Complexity calibration
#    - Only two aspect ratios are allowed: 5x7 and 7x5.
#      This is enough to make the change visible, but avoids a large shape zoo.
#    - Each sample always contains both widths 5 and 7 among edge frames, so the
#      variable inward offsets are supported by repeated evidence.
# c) Hidden assumptions from the original main()
#    - Scanning only 5x5 windows no longer works: rectangles would be missed.
#    - Using a fixed row-span of 5 for overlap is invalid once heights vary.
#    - Using a fixed inward destination of col 5 / W-10 is invalid once widths vary.
#    - For the red side, "one square-width inward" used to be W-2*5 because edge
#      width and current width were both 5. Now the correct inward placement is
#      W - overlap_edge_width - current_width.
# d) Placement legality
#    - The generator uses output-first construction.
#    - On each side there are exactly two edge frames and one inward frame.
#    - The inward frame is forced to overlap exactly one edge frame in rows, so the
#      width that controls the inward offset is unambiguous.
#    - The canvas width is increased to 28 so that a 7-wide edge frame plus a 7-wide
#      inward frame can fit on both sides without left/right groups colliding.
# e) Visual observability
#    - The rule never depends on invisible metadata such as color ids for sorting.
#      Ordering still comes only from visible horizontal positions in the input.
#    - Both 5x7 and 7x5 rectangles appear in every sample, and inward placements use
#      both controlling widths (5 and 7) across the two sides of the same sample.
# f) Visual cleanliness
#    - Input frames are generated without overlap.
#    - Output layers never exceed depth 2 per side, matching the original task's
#      conceptual simplicity while keeping the new width-dependent offsets legible.
# g) Color inference
#    - No new color mapping is introduced. Border colors still define the side, and
#      interior colors stay uniform and merely distinguish individual frames.


GRID_H, GRID_W = 22, 28
AZURE, RED = 8, 2
SHAPES = ((5, 7), (7, 5))  # (height, width)
N_AZURE = 3
N_RED = 3
INPUT_LAYOUT_RETRIES = 200
SIDE_LAYOUT_RETRIES = 200
GEN_RETRIES = 800


def _span_overlap(a0: int, ah: int, b0: int, bh: int) -> bool:
    a1 = int(a0 + ah - 1)
    b1 = int(b0 + bh - 1)
    return not (a1 < int(b0) or b1 < int(a0))


def _rects_overlap(a: dict, b: dict, pad: int = 0) -> bool:
    a_r0 = int(a["r0"]) - pad
    a_r1 = int(a["r0"]) + int(a["h"]) - 1 + pad
    b_r0 = int(b["r0"]) - pad
    b_r1 = int(b["r0"]) + int(b["h"]) - 1 + pad
    a_c0 = int(a["c0"]) - pad
    a_c1 = int(a["c0"]) + int(a["w"]) - 1 + pad
    b_c0 = int(b["c0"]) - pad
    b_c1 = int(b["c0"]) + int(b["w"]) - 1 + pad
    return not (a_r1 < b_r0 or b_r1 < a_r0 or a_c1 < b_c0 or b_c1 < a_c0)


def _any_pairwise_overlap(objs: list[dict], pad: int = 0) -> bool:
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if _rects_overlap(objs[i], objs[j], pad=pad):
                return True
    return False


def _make_patch(frame_color: int, inner_color: int, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    patch = np.full((h, w), frame_color, dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _detect_frames(g: np.ndarray) -> list[dict]:
    H, W = g.shape
    found: dict[tuple[int, int, int, int], dict] = {}
    for h, w in SHAPES:
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                patch = crop_bbox(g, (r0, c0, r0 + h - 1, c0 + w - 1))
                border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
                if int(border.min()) != int(border.max()):
                    continue
                frame_col = int(border[0])
                if frame_col not in {AZURE, RED}:
                    continue
                inner = patch[1:-1, 1:-1]
                if inner.size == 0:
                    continue
                if int(inner.min()) != int(inner.max()):
                    continue
                if int(inner[0, 0]) == frame_col:
                    continue
                found[(r0, c0, h, w)] = {
                    "frame": frame_col,
                    "r0": r0,
                    "c0": c0,
                    "h": h,
                    "w": w,
                    "patch": patch,
                }
    return list(found.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered rectangles to their designated grid edges by border color.

    Input structure:
    - bg = most common color.
    - Teal (8) and red (2) bordered rectangles are scattered across the canvas.
    - Every frame is either 5x7 or 7x5, with a 1-pixel border and a uniform interior fill.

    Rule:
    - Teal rectangles move to the left side; red rectangles move to the right side.
    - Each rectangle keeps its original top row.
    - Teal is processed by increasing input column; red is processed by decreasing input column.
    - If a rectangle row-overlaps an already placed edge rectangle of the same group, it shifts one layer inward,
      and that inward offset is determined by the overlapping edge rectangle's width.

    Key insight:
    - The side assignment still comes from border color, but the stack geometry is no longer fixed:
      row overlap depends on true height (5 or 7), and inward placement depends on the width (5 or 7)
      of the earlier edge rectangle that the current rectangle collides with vertically.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    rects = _detect_frames(g)

    azure_rects = sorted((r for r in rects if int(r["frame"]) == AZURE), key=lambda r: int(r["c0"]))
    placed_edges: list[dict] = []
    for rect in azure_rects:
        overlaps = [p for p in placed_edges if _span_overlap(int(rect["r0"]), int(rect["h"]), int(p["r0"]), int(p["h"]))]
        if overlaps:
            dest_c = max(int(p["w"]) for p in overlaps)
        else:
            dest_c = 0
        paste(out, rect["patch"], int(rect["r0"]), int(dest_c), transparent=bg)
        if not overlaps:
            placed_edges.append(rect)

    red_rects = sorted((r for r in rects if int(r["frame"]) == RED), key=lambda r: -int(r["c0"]))
    placed_edges = []
    for rect in red_rects:
        overlaps = [p for p in placed_edges if _span_overlap(int(rect["r0"]), int(rect["h"]), int(p["r0"]), int(p["h"]))]
        if overlaps:
            edge_width = max(int(p["w"]) for p in overlaps)
            dest_c = int(W - edge_width - int(rect["w"]))
        else:
            dest_c = int(W - int(rect["w"]))
        paste(out, rect["patch"], int(rect["r0"]), int(dest_c), transparent=bg)
        if not overlaps:
            placed_edges.append(rect)

    return out


def _build_side_layout(
    frame_color: int,
    side: str,
    inner_colors: list[int],
    target_edge_width: int,
) -> list[dict] | None:
    edge_shapes = [(5, 7), (7, 5)]
    random.shuffle(edge_shapes)
    target_idx = next(i for i, (_h, w) in enumerate(edge_shapes) if int(w) == int(target_edge_width))
    other_idx = 1 - target_idx
    inward_shape = random.choice(SHAPES)

    for _ in range(SIDE_LAYOUT_RETRIES):
        edge_rows = []
        ok = True
        for h, _w in edge_shapes:
            r0 = random.randint(0, GRID_H - h)
            if any(_span_overlap(r0, h, prev_r0, prev_h) for prev_r0, prev_h in edge_rows):
                ok = False
                break
            edge_rows.append((r0, h))
        if not ok:
            continue

        target_row = edge_rows[target_idx][0]
        target_h = edge_shapes[target_idx][0]
        other_row = edge_rows[other_idx][0]
        other_h = edge_shapes[other_idx][0]
        inward_h, inward_w = inward_shape
        inward_rows = [
            r0
            for r0 in range(GRID_H - inward_h + 1)
            if _span_overlap(r0, inward_h, target_row, target_h)
            and not _span_overlap(r0, inward_h, other_row, other_h)
        ]
        if not inward_rows:
            continue

        inward_row = random.choice(inward_rows)
        objects: list[dict] = []
        for i, ((h, w), (r0, _)) in enumerate(zip(edge_shapes, edge_rows)):
            if side == "azure":
                out_c = 0
            else:
                out_c = GRID_W - w
            role = "edge_target" if i == target_idx else "edge_other"
            patch = _make_patch(frame_color, inner_colors[i], (h, w))
            objects.append(
                {
                    "frame": frame_color,
                    "side": side,
                    "role": role,
                    "r0": r0,
                    "c0": out_c,
                    "out_c": out_c,
                    "h": h,
                    "w": w,
                    "patch": patch,
                }
            )

        if side == "azure":
            inward_c = target_edge_width
        else:
            inward_c = GRID_W - target_edge_width - inward_w
        patch = _make_patch(frame_color, inner_colors[2], inward_shape)
        objects.append(
            {
                "frame": frame_color,
                "side": side,
                "role": "inward",
                "r0": inward_row,
                "c0": inward_c,
                "out_c": inward_c,
                "h": inward_h,
                "w": inward_w,
                "patch": patch,
            }
        )

        if not _any_pairwise_overlap(objects):
            return objects

    return None


def _derive_input_positions(objects: list[dict], ascending: bool) -> list[dict] | None:
    target_obj = next(obj for obj in objects if obj["role"] == "edge_target")
    inward_obj = next(obj for obj in objects if obj["role"] == "inward")

    for _ in range(INPUT_LAYOUT_RETRIES):
        placed = []
        cols_used: set[int] = set()
        ok = True
        for obj in objects:
            c0 = random.randint(0, GRID_W - int(obj["w"]))
            if c0 == int(obj["out_c"]) or c0 in cols_used:
                ok = False
                break
            cols_used.add(c0)
            placed.append({**obj, "c0": c0})
        if not ok:
            continue
        if _any_pairwise_overlap(placed):
            continue

        ordered = sorted(placed, key=lambda obj: int(obj["c0"]), reverse=not ascending)
        order_lookup = {id(obj): idx for idx, obj in enumerate(ordered)}
        target_pos = next(order_lookup[id(obj)] for obj in placed if obj["role"] == "edge_target")
        inward_pos = next(order_lookup[id(obj)] for obj in placed if obj["role"] == "inward")
        if target_pos >= inward_pos:
            continue

        return placed

    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])

    # Force both controlling widths to appear in every sample.
    azure_target_width = random.choice([5, 7])
    red_target_width = 12 - azure_target_width

    azure_colors = random.sample([c for c in range(10) if c not in {bg, AZURE}], N_AZURE)
    red_colors = random.sample([c for c in range(10) if c not in {bg, RED}], N_RED)

    azure_out = _build_side_layout(AZURE, "azure", azure_colors, azure_target_width)
    if azure_out is None:
        return None
    red_out = _build_side_layout(RED, "red", red_colors, red_target_width)
    if red_out is None:
        return None

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    for obj in azure_out + red_out:
        paste(output, obj["patch"], int(obj["r0"]), int(obj["out_c"]))

    azure_input = _derive_input_positions(azure_out, ascending=True)
    if azure_input is None:
        return None
    red_input = _derive_input_positions(red_out, ascending=False)
    if red_input is None:
        return None

    all_input = azure_input + red_input
    if _any_pairwise_overlap(all_input):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for obj in all_input:
        paste(inp, obj["patch"], int(obj["r0"]), int(obj["c0"]))

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
