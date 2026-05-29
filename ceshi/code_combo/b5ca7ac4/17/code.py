# Combo variant of b5ca7ac4 (index 17)
# combo_spec: D3/2+D5/1
# Name: 矩形框架+垂直堆叠

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

for parent in Path(__file__).resolve().parents:
    candidate = parent / "solvers" / "arc_utils.py"
    if candidate.exists():
        sys.path.insert(0, str(parent / "solvers"))
        break

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    - Original task: 5x5 squares moved horizontally. Azure was sorted by input
#      column and pushed to the left; red was sorted by input column and pushed
#      to the right. Row overlap decided whether a square stayed on the edge or
#      shifted inward by a fixed 5 columns.
#    - Combo variant: objects become 5x7 or 7x5 rectangles, and motion rotates
#      to the vertical axis. The solver now sorts azure by input row ascending
#      and red by input row descending, keeps each rectangle's column fixed, and
#      pushes azure to the top / red to the bottom.
#    - Example: an azure 7x5 edge rectangle at columns 3..7 occupies rows 0..6
#      in the output. A later azure rectangle whose columns overlap 3..7 does
#      not go to row 5 just because some rectangle is involved; it goes to row 7
#      because the overlapping edge rectangle is 7 cells tall. If the controlling
#      edge rectangle were 5x7, the inward row would instead be 5.
# b) Complexity calibration
#    - Only two aspect ratios are allowed: 5x7 and 7x5. This keeps the shape
#      vocabulary small while making both width and height visibly non-constant.
#    - Each sample contains one 5-high edge rectangle and one 7-high edge
#      rectangle per color group, plus an inward rectangle. This gives repeated
#      evidence that the second vertical layer depends on height, not a fixed 5.
#    - One color group is forced to use height 5 as the controlling edge and the
#      other is forced to use height 7, so every sample exhibits both offsets.
# c) Hidden assumptions from the original main()
#    - Scanning only 5x5 windows would miss 5x7 / 7x5 frames; detection must
#      scan both allowed rectangle sizes.
#    - The original overlap check used row spans because horizontal relocation
#      preserved rows. After rotating the stacking direction, the deciding overlap
#      axis becomes columns, and the span length is the true width.
#    - The original inward destination used a constant 5-cell column shift.
#      Here the inward destination uses the overlapping edge rectangle's height,
#      producing row 5 or row 7 for azure and the corresponding bottom-aligned
#      offsets for red.
#    - A later rectangle could overlap more than one placed edge rectangle in
#      arbitrary inputs. The solver resolves that by using the maximum overlapping
#      edge height. The generator avoids ambiguity by making each inward rectangle
#      overlap exactly one edge rectangle in columns.
# d) Placement legality
#    - Output is built first. Azure edge rectangles sit at row 0; red edge
#      rectangles are bottom-aligned at row H-h. Inward rectangles are placed at
#      row controlling_edge_height for azure or H-controlling_edge_height-h for red.
#    - The canvas height is expanded to 28 so two 7-high layers can fit at the
#      top and bottom simultaneously without cross-group collisions.
#    - Within each color group, edge rectangles use disjoint columns. The inward
#      rectangle column-overlaps exactly one chosen edge rectangle and avoids the
#      other, so the controlling height is well-defined and placements never overlap.
# e) Visual observability
#    - The transformation depends only on visible properties: border color chooses
#      top vs. bottom, input row chooses processing order, visible rectangle width
#      chooses column-overlap, and visible rectangle height chooses the second
#      layer's row offset.
#    - Because every sample includes both 5x7 and 7x5 frames, a human sees that
#      the solver is not acting on identical 5x5 squares.
# f) Visual cleanliness
#    - Output depth remains 2 per color group, matching the seed task's structure.
#    - Top and bottom groups occupy separate vertical zones on the 28-row canvas,
#      preventing mixed-color clutter in the center.
#    - Input rectangles are generated without overlap, so each frame is legible.
# g) Change inferability
#    - No hidden color remapping is introduced. Interior colors merely distinguish
#      individual rectangles, while the frame colors still provide the only group
#      signal. The important new cue is geometric: 5-high and 7-high edge frames
#      create different second-layer row offsets.


GRID_H, GRID_W = 28, 22
AZURE, RED = 8, 2
SHAPES = ((5, 7), (7, 5))  # (height, width)
N_AZURE = 3
N_RED = 3
SIDE_LAYOUT_RETRIES = 300
INPUT_LAYOUT_RETRIES = 300
GEN_RETRIES = 1000


def _span_overlap(a0: int, alen: int, b0: int, blen: int) -> bool:
    a1 = int(a0 + alen - 1)
    b1 = int(b0 + blen - 1)
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
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(h),
                    "w": int(w),
                    "patch": patch,
                }
    return list(found.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered rectangles into top or bottom stacks according to border color.

    Input structure:
    - bg is the most common color.
    - Teal (8) and red (2) bordered rectangles are scattered across the grid.
    - Each frame is either 5x7 or 7x5, with a 1-cell border and a uniform interior fill.

    Rule:
    - Teal rectangles keep their original column and are pushed upward; red rectangles keep their original column and are pushed downward.
    - Process teal from top to bottom and red from bottom to top.
    - A rectangle that column-overlaps an already placed edge rectangle of the same color goes to the second vertical layer instead of the edge.
    - That second-layer row offset is the overlapping edge rectangle's height, so 5-high and 7-high edge rectangles create different inward rows.

    Key insight:
    - This is not just the old vertical-stack variant with bigger objects: visible rectangle height now controls where the inward layer lands, while visible column overlap controls whether a rectangle becomes inward at all.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    rects = _detect_frames(g)

    azure_rects = sorted((r for r in rects if int(r["frame"]) == AZURE), key=lambda r: int(r["r0"]))
    placed_edges: list[dict] = []
    for rect in azure_rects:
        overlaps = [p for p in placed_edges if _span_overlap(int(rect["c0"]), int(rect["w"]), int(p["c0"]), int(p["w"]))]
        if overlaps:
            dest_r = max(int(p["h"]) for p in overlaps)
        else:
            dest_r = 0
        paste(out, rect["patch"], int(dest_r), int(rect["c0"]), transparent=bg)
        if not overlaps:
            placed_edges.append(rect)

    red_rects = sorted((r for r in rects if int(r["frame"]) == RED), key=lambda r: -int(r["r0"]))
    placed_edges = []
    for rect in red_rects:
        overlaps = [p for p in placed_edges if _span_overlap(int(rect["c0"]), int(rect["w"]), int(p["c0"]), int(p["w"]))]
        if overlaps:
            edge_height = max(int(p["h"]) for p in overlaps)
            dest_r = int(H - edge_height - int(rect["h"]))
        else:
            dest_r = int(H - int(rect["h"]))
        paste(out, rect["patch"], int(dest_r), int(rect["c0"]), transparent=bg)
        if not overlaps:
            placed_edges.append(rect)

    return out


def _build_side_layout(
    frame_color: int,
    side: str,
    inner_colors: list[int],
    target_edge_height: int,
) -> list[dict] | None:
    edge_shapes = [(5, 7), (7, 5)]
    random.shuffle(edge_shapes)
    target_idx = next(i for i, (h, _w) in enumerate(edge_shapes) if int(h) == int(target_edge_height))
    other_idx = 1 - target_idx
    inward_shape = random.choice(SHAPES)

    for _ in range(SIDE_LAYOUT_RETRIES):
        edge_cols: list[tuple[int, int]] = []
        ok = True
        for _h, w in edge_shapes:
            c0 = random.randint(0, GRID_W - w)
            if any(_span_overlap(c0, w, prev_c0, prev_w) for prev_c0, prev_w in edge_cols):
                ok = False
                break
            edge_cols.append((c0, w))
        if not ok:
            continue

        target_c0, target_w = edge_cols[target_idx]
        other_c0, other_w = edge_cols[other_idx]
        inward_h, inward_w = inward_shape
        inward_cols = [
            c0
            for c0 in range(GRID_W - inward_w + 1)
            if _span_overlap(c0, inward_w, target_c0, target_w)
            and not _span_overlap(c0, inward_w, other_c0, other_w)
        ]
        if not inward_cols:
            continue

        inward_c0 = random.choice(inward_cols)
        objects: list[dict] = []
        for idx, ((h, w), (c0, _)) in enumerate(zip(edge_shapes, edge_cols)):
            out_r = 0 if side == "azure" else GRID_H - h
            role = "edge_target" if idx == target_idx else "edge_other"
            objects.append(
                {
                    "id": f"{side}_{role}",
                    "frame": frame_color,
                    "side": side,
                    "role": role,
                    "out_r": int(out_r),
                    "r0": int(out_r),
                    "c0": int(c0),
                    "h": int(h),
                    "w": int(w),
                    "patch": _make_patch(frame_color, inner_colors[idx], (h, w)),
                }
            )

        inward_out_r = target_edge_height if side == "azure" else GRID_H - target_edge_height - inward_h
        objects.append(
            {
                "id": f"{side}_inward",
                "frame": frame_color,
                "side": side,
                "role": "inward",
                "out_r": int(inward_out_r),
                "r0": int(inward_out_r),
                "c0": int(inward_c0),
                "h": int(inward_h),
                "w": int(inward_w),
                "patch": _make_patch(frame_color, inner_colors[2], inward_shape),
            }
        )

        if not _any_pairwise_overlap(objects):
            return objects

    return None


def _derive_input_positions(objects: list[dict], ascending: bool) -> list[dict] | None:
    edges = [obj for obj in objects if obj["role"].startswith("edge")]
    inwards = [obj for obj in objects if obj["role"] == "inward"]
    deps = {
        obj["id"]: [edge["id"] for edge in edges if _span_overlap(int(obj["c0"]), int(obj["w"]), int(edge["c0"]), int(edge["w"]))]
        for obj in inwards
    }

    for _ in range(INPUT_LAYOUT_RETRIES):
        placed: list[dict] = []
        ok = True
        for obj in objects:
            r0 = random.randint(0, GRID_H - int(obj["h"]))
            if r0 == int(obj["out_r"]):
                ok = False
                break
            placed.append({**obj, "r0": int(r0)})
        if not ok or _any_pairwise_overlap(placed):
            continue

        ordered = sorted(placed, key=lambda obj: int(obj["r0"]), reverse=not ascending)
        order_pos = {obj["id"]: idx for idx, obj in enumerate(ordered)}

        valid = True
        for inward in inwards:
            inward_pos = order_pos[inward["id"]]
            if not deps[inward["id"]]:
                valid = False
                break
            latest_dep = max(order_pos[dep_id] for dep_id in deps[inward["id"]])
            if latest_dep >= inward_pos:
                valid = False
                break
        if not valid:
            continue

        return placed

    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])

    # Force both controlling heights to appear in every sample.
    azure_target_height = random.choice([5, 7])
    red_target_height = 12 - azure_target_height

    azure_colors = random.sample([c for c in range(10) if c not in {bg, AZURE}], N_AZURE)
    red_colors = random.sample([c for c in range(10) if c not in {bg, RED}], N_RED)

    azure_out = _build_side_layout(AZURE, "azure", azure_colors, azure_target_height)
    if azure_out is None:
        return None
    red_out = _build_side_layout(RED, "red", red_colors, red_target_height)
    if red_out is None:
        return None

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    for obj in azure_out + red_out:
        paste(output, obj["patch"], int(obj["out_r"]), int(obj["c0"]))

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
