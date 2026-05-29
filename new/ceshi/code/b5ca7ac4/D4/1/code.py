# Variant of b5ca7ac4
# Dimension: D4 — 源对象——组标识方式 (src)
# Change: 通过边框颜色标识分组：青色(8)边框 vs 红色(2)边框，这两种颜色在所有示例中固定... → 通过边框厚度标识——1像素 vs 2像素

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the D4 variant
# ==================================
# a) Concrete behavior:
#    - The "left-moving" group is now identified by a 1-cell-thick frame: a 5x5 square
#      with a 3x3 solid interior.
#    - The "right-moving" group is now identified by a 2-cell-thick frame: a 7x7 square
#      whose outer two rings are frame color and whose center is still a 3x3 solid block.
#    - All frames in one sample share the same frame color, so color is no longer a usable
#      group marker. The visible cue is frame thickness / overall object size.
#    - Small example trace:
#      original logic: a teal 5x5 frame at col 3 is routed left because its border color is 8.
#      new logic: a 5x5 thin frame at col 3 is routed left because it is the thin-frame type,
#      even if its frame color matches every other object's frame color.
#      original logic: a red 5x5 frame at col 16 is routed right because its border color is 2.
#      new logic: a 7x7 thick frame at col 13 is routed right because it has a 2-cell border.
# b) Complexity calibration:
#    - Only two thickness categories are used: 1-cell and 2-cell. That keeps the rule small.
#    - The thick objects are 7x7 while the thin objects are 5x5; this makes the source-group
#      change obvious to a human from 2-3 examples.
#    - All frames share one color per sample, which actively removes the old color cue.
#    - The thick group is capped at 4 objects instead of the original 5 because 7x7 objects on a
#      22x22 canvas need more room for both valid scattering and non-overlapping output lanes.
# c) Hidden-assumption audit of the original main():
#    - The original scanner assumed every relevant object fits in a 5x5 window. That fails here
#      because 2-thick frames are 7x7.
#    - The original "frame_col in {2, 8}" filter hard-coded the old source attribute.
#    - A new pitfall appears: the center 5x5 of a 7x7 thick frame also looks like a valid thin
#      frame. The solver therefore detects thick 7x7 frames first and suppresses thin detections
#      fully contained inside them.
#    - The original inward offset equaled SQUARE_SIZE. In the variant the two groups have
#      different sizes, so inward offsets must use each group's own width (5 for thin, 7 for thick).
# d) Placement legality:
#    - The generator remains output-first and only accepts samples where output rectangles do not
#      overlap, especially because thin inward squares (cols 5-9) and thick inward squares
#      (cols 8-14) partially share horizontal territory.
#    - Input placement forbids overlap, and the final main(input) == output validation rejects any
#      incidental same-color arrangement that would create extra false detections.
# e) Visual observability:
#    - Because both groups share one frame color, the only stable way to infer the grouping is by
#      border thickness / object size, exactly matching the intended D4 change.
#    - Ordering still depends only on visible horizontal position, not on hidden color ids.
# f) Visual cleanliness:
#    - Output validation rejects cross-group overlaps.
#    - Each object's center is a solid 3x3 color block, keeping the silhouettes easy to parse.
# g) Color-rule inference:
#    - No color mapping is part of the rule. Colors only label interiors, while frame color is
#      deliberately de-emphasized by making it common across both groups.
# 9) New solver assumptions and generator guarantees:
#    - Solver assumes every valid object has a uniform border color and a uniform 3x3 center.
#      Generator constructs exactly that.
#    - Solver assumes no intended thin object sits inside a thick object except the false-positive
#      5x5 center of that thick object. Generator never intentionally constructs such containments,
#      so suppression-by-containment is safe.
#    - Solver assumes only two stack depths per side. Generator keeps the original edge/inward
#      structure and validates with main(input) == output.


GRID_H, GRID_W = 22, 22
THIN_SIZE = 5
THICK_SIZE = 7
INNER_SIZE = 3
MAX_INWARD = 2
ROW_ASSIGN_RETRIES = 60
INPUT_COL_RETRIES = 80
GEN_RETRIES = 800
N_THIN_RANGE = (3, 4)
N_THICK_RANGE = (3, 4)


def _span_overlaps(a0: int, a_len: int, b0: int, b_len: int) -> bool:
    return not (a0 + a_len - 1 < b0 or b0 + b_len - 1 < a0)


def _rects_overlap(r1: int, c1: int, s1: int, r2: int, c2: int, s2: int) -> bool:
    return _span_overlaps(r1, s1, r2, s2) and _span_overlaps(c1, s1, c2, s2)


def _uniform_non_bg(values: np.ndarray, bg: int) -> int | None:
    if values.size == 0:
        return None
    if int(values.min()) != int(values.max()):
        return None
    color = int(values.flat[0])
    if color == bg:
        return None
    return color


def _make_patch(frame_color: int, inner_color: int, size: int, thickness: int) -> np.ndarray:
    patch = np.full((size, size), frame_color, dtype=int)
    patch[thickness : thickness + INNER_SIZE, thickness : thickness + INNER_SIZE] = inner_color
    return patch


def _detect_squares(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    squares: list[dict] = []
    thick_boxes: list[tuple[int, int, int, int]] = []

    for r0 in range(H - THICK_SIZE + 1):
        for c0 in range(W - THICK_SIZE + 1):
            patch = crop_bbox(g, (r0, c0, r0 + THICK_SIZE - 1, c0 + THICK_SIZE - 1))
            border_mask = np.zeros((THICK_SIZE, THICK_SIZE), dtype=bool)
            border_mask[:2, :] = True
            border_mask[-2:, :] = True
            border_mask[:, :2] = True
            border_mask[:, -2:] = True
            frame_col = _uniform_non_bg(patch[border_mask], bg)
            if frame_col is None:
                continue

            inner = patch[2:5, 2:5]
            inner_col = _uniform_non_bg(inner, bg)
            if inner_col is None or inner_col == frame_col:
                continue

            squares.append(
                {
                    "group": "thick",
                    "size": THICK_SIZE,
                    "thickness": 2,
                    "frame": frame_col,
                    "r0": r0,
                    "c0": c0,
                    "patch": patch.copy(),
                }
            )
            thick_boxes.append((r0, c0, r0 + THICK_SIZE - 1, c0 + THICK_SIZE - 1))

    for r0 in range(H - THIN_SIZE + 1):
        for c0 in range(W - THIN_SIZE + 1):
            if any(tr0 <= r0 and tc0 <= c0 and tr1 >= r0 + THIN_SIZE - 1 and tc1 >= c0 + THIN_SIZE - 1
                   for tr0, tc0, tr1, tc1 in thick_boxes):
                continue

            patch = crop_bbox(g, (r0, c0, r0 + THIN_SIZE - 1, c0 + THIN_SIZE - 1))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            frame_col = _uniform_non_bg(border, bg)
            if frame_col is None:
                continue

            inner = patch[1:4, 1:4]
            inner_col = _uniform_non_bg(inner, bg)
            if inner_col is None or inner_col == frame_col:
                continue

            squares.append(
                {
                    "group": "thin",
                    "size": THIN_SIZE,
                    "thickness": 1,
                    "frame": frame_col,
                    "r0": r0,
                    "c0": c0,
                    "patch": patch.copy(),
                }
            )

    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered squares to their designated grid edges by border thickness.

    Input structure:
    - bg = most common color.
    - All movable objects are same-colored framed squares with solid 3×3 centers, scattered across the grid.
    - Thin-frame objects are 5×5 with a 1-pixel border; thick-frame objects are 7×7 with a 2-pixel border.

    Rule:
    - Thin 5×5 frames → left edge (col 0); thick 7×7 frames → right edge (col W−7).
    - Each object keeps its original row.
    - Process thin frames by original column ascending and thick frames by original column descending.
    - If an object's rows overlap an already-placed edge object from the same group, place it one object-width inward instead.

    Key insight: Group membership is now encoded by visible frame thickness, and the near-edge-first order still decides which overlapping object claims the edge lane.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    squares = _detect_squares(g, bg)

    thin_squares = sorted((s for s in squares if s["group"] == "thin"), key=lambda s: int(s["c0"]))
    placed_edge_rows: list[tuple[int, int]] = []
    for s in thin_squares:
        r0 = int(s["r0"])
        has_overlap = any(_span_overlaps(r0, THIN_SIZE, pr0, pr1 - pr0 + 1) for pr0, pr1 in placed_edge_rows)
        dest_c = THIN_SIZE if has_overlap else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_edge_rows.append((r0, r0 + THIN_SIZE - 1))

    thick_squares = sorted((s for s in squares if s["group"] == "thick"), key=lambda s: -int(s["c0"]))
    placed_edge_rows = []
    for s in thick_squares:
        r0 = int(s["r0"])
        has_overlap = any(_span_overlaps(r0, THICK_SIZE, pr0, pr1 - pr0 + 1) for pr0, pr1 in placed_edge_rows)
        dest_c = W - 2 * THICK_SIZE if has_overlap else W - THICK_SIZE
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == W - THICK_SIZE:
            placed_edge_rows.append((r0, r0 + THICK_SIZE - 1))

    return out


def _max_non_overlapping_rows(size: int) -> int:
    return (GRID_H + size - 1) // size


def _assign_output_rows(n: int, size: int) -> tuple[list[int], list[int]] | None:
    max_edge = _max_non_overlapping_rows(size)
    min_inward = max(1, n - max_edge)
    max_inward = min(MAX_INWARD, n - 1)
    if min_inward > max_inward:
        return None

    for _ in range(ROW_ASSIGN_RETRIES):
        n_inward = random.randint(min_inward, max_inward)
        n_edge = n - n_inward

        all_rows = list(range(GRID_H - size + 1))
        random.shuffle(all_rows)
        edge_rows: list[int] = []
        for r in all_rows:
            if all(not _span_overlaps(r, size, er, size) for er in edge_rows):
                edge_rows.append(r)
                if len(edge_rows) == n_edge:
                    break
        if len(edge_rows) < n_edge:
            continue

        random.shuffle(all_rows)
        inward_rows: list[int] = []
        for r in all_rows:
            if not any(_span_overlaps(r, size, er, size) for er in edge_rows):
                continue
            if any(_span_overlaps(r, size, ir, size) for ir in inward_rows):
                continue
            inward_rows.append(r)
            if len(inward_rows) == n_inward:
                break
        if len(inward_rows) == n_inward:
            return edge_rows, inward_rows
    return None


def _build_output_objects(
    edge_rows: list[int],
    inward_rows: list[int],
    colors: list[int],
    frame_color: int,
    size: int,
    thickness: int,
    edge_col: int,
    inward_col: int,
    group: str,
) -> list[dict]:
    objects: list[dict] = []
    color_idx = 0
    for r in edge_rows:
        patch = _make_patch(frame_color, colors[color_idx], size, thickness)
        color_idx += 1
        objects.append(
            {
                "group": group,
                "r": r,
                "c": edge_col,
                "size": size,
                "is_edge": True,
                "patch": patch,
            }
        )
    for r in inward_rows:
        patch = _make_patch(frame_color, colors[color_idx], size, thickness)
        color_idx += 1
        objects.append(
            {
                "group": group,
                "r": r,
                "c": inward_col,
                "size": size,
                "is_edge": False,
                "patch": patch,
            }
        )
    return objects


def _any_pairwise_overlap(objects: list[dict]) -> bool:
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _rects_overlap(
                int(objects[i]["r"]),
                int(objects[i]["c"]),
                int(objects[i]["size"]),
                int(objects[j]["r"]),
                int(objects[j]["c"]),
                int(objects[j]["size"]),
            ):
                return True
    return False


def _derive_input_positions(
    objects: list[dict],
    ascending: bool,
    blocked: list[dict] | None = None,
) -> list[dict] | None:
    edges = [obj for obj in objects if bool(obj["is_edge"])]
    inwards = [obj for obj in objects if not bool(obj["is_edge"])]
    size = int(objects[0]["size"])
    blocked = [] if blocked is None else blocked

    inward_deps = [
        [i for i, edge in enumerate(edges) if _span_overlaps(int(inward["r"]), size, int(edge["r"]), size)]
        for inward in inwards
    ]

    col_range = list(range(GRID_W - size + 1))
    if len(col_range) < len(objects):
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

        inward_indices = list(range(len(inwards)))
        random.shuffle(inward_indices)
        for inward_idx in inward_indices:
            latest_pos = max(
                (
                    pos
                    for dep in inward_deps[inward_idx]
                    for pos, (kind, idx) in enumerate(order)
                    if kind == "edge" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", inward_idx))

        ordered_objects = [edges[idx] if kind == "edge" else inwards[idx] for kind, idx in order]
        chosen_cols = sorted(random.sample(col_range, len(objects)))
        if not ascending:
            chosen_cols.reverse()

        result: list[dict] = []
        ok = True
        for obj, in_col in zip(ordered_objects, chosen_cols):
            if in_col == int(obj["c"]):
                ok = False
                break
            result.append(
                {
                    "group": obj["group"],
                    "r": int(obj["r"]),
                    "c": in_col,
                    "size": size,
                    "patch": obj["patch"],
                }
            )
        if not ok:
            continue
        if _any_pairwise_overlap(result):
            continue
        if any(
            _rects_overlap(int(obj["r"]), int(obj["c"]), int(obj["size"]), int(other["r"]), int(other["c"]), int(other["size"]))
            for obj in result
            for other in blocked
        ):
            continue
        return result

    return None


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.randint(0, 9)
    frame_choices = [c for c in range(10) if c not in {bg, 2, 8}]
    frame_color = random.choice(frame_choices)

    n_thin = random.randint(*N_THIN_RANGE)
    n_thick = random.randint(*N_THICK_RANGE)

    avail_inner = [c for c in range(10) if c not in {bg, frame_color}]
    if len(avail_inner) < max(n_thin, n_thick):
        return None
    thin_colors = random.sample(avail_inner, n_thin)
    thick_colors = random.sample(avail_inner, n_thick)

    for _ in range(ROW_ASSIGN_RETRIES):
        thin_assign = _assign_output_rows(n_thin, THIN_SIZE)
        thick_assign = _assign_output_rows(n_thick, THICK_SIZE)
        if thin_assign is None or thick_assign is None:
            continue
        thin_edge_rows, thin_inward_rows = thin_assign
        thick_edge_rows, thick_inward_rows = thick_assign

        thin_out = _build_output_objects(
            thin_edge_rows,
            thin_inward_rows,
            thin_colors,
            frame_color,
            THIN_SIZE,
            1,
            edge_col=0,
            inward_col=THIN_SIZE,
            group="thin",
        )
        thick_out = _build_output_objects(
            thick_edge_rows,
            thick_inward_rows,
            thick_colors,
            frame_color,
            THICK_SIZE,
            2,
            edge_col=GRID_W - THICK_SIZE,
            inward_col=GRID_W - 2 * THICK_SIZE,
            group="thick",
        )

        output_objects = thin_out + thick_out
        if _any_pairwise_overlap(output_objects):
            continue

        thick_input = _derive_input_positions(thick_out, ascending=False)
        if thick_input is None:
            continue
        thin_input = _derive_input_positions(thin_out, ascending=True, blocked=thick_input)
        if thin_input is None:
            continue

        input_objects = thin_input + thick_input
        if _any_pairwise_overlap(input_objects):
            continue

        output = np.full((GRID_H, GRID_W), bg, dtype=int)
        for obj in output_objects:
            paste(output, obj["patch"], int(obj["r"]), int(obj["c"]))

        inp = np.full((GRID_H, GRID_W), bg, dtype=int)
        for obj in input_objects:
            paste(inp, obj["patch"], int(obj["r"]), int(obj["c"]))

        try:
            if np.array_equal(main(inp), output):
                return inp, output
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            inp, _ = result
            return inp
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
