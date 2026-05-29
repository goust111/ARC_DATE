# Combo variant of b5ca7ac4 (index 18)
# combo_spec: D3/0+D4/1
# Name: 边框厚度标识+可变尺寸框架

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the fused variant
# a) Concrete behavior and example trace:
#    - The visible group marker is no longer border color. Instead, the left-moving group is always
#      a thin 5x5 frame with a 1-cell border around a 3x3 center, and the right-moving group is
#      always a thick 7x7 frame with a 2-cell border around the same 3x3 center.
#    - This fuses the two requested changes into one observable cue: thickness determines group, and
#      the same thickness also determines the object's overall size.
#    - Example: if a 5x5 thin frame sits at row 4 col 3 and another thin frame at row 6 col 11,
#      the leftmost one claims the left edge and the overlapping later one shifts to col 5.
#      Separately, if a 7x7 thick frame sits at row 2 col 16 and another thick frame at row 5
#      col 12, the rightmost one claims the right edge and the overlapping later one shifts to col 8.
#    - Compared with the original task, the moved objects are not distinguished by teal vs red and
#      the two groups now visibly have different footprints, 5x5 vs 7x7.
# b) Complexity calibration:
#    - The variant uses exactly two thickness/size categories: 1-cell border -> 5x5, 2-cell border
#      -> 7x7. This is richer than the original fixed-size task but still simple enough to infer.
#    - Every generated sample contains both groups and at least one inward object on each side, so
#      both the thickness cue and the stacking rule appear repeatedly instead of by accident.
#    - All frames in one sample share a single frame color, which removes the original color cue and
#      prevents the task from degenerating into "look for red vs teal".
# c) Implicit assumption audit against the original solver:
#    - The original 5x5-only scanner fails because thick objects are 7x7; detection must scan both
#      footprints and classify by border thickness.
#    - The original frame-color check {2, 8} fails because frame color is no longer the group label.
#    - A 7x7 thick frame contains a 5x5 subwindow that can masquerade as a thin frame, so the new
#      solver must detect thick frames first and suppress contained thin candidates.
#    - The original fixed inward offsets 5 and W-10 split into group-specific constants: thin frames
#      shift by 5 on the left, thick frames shift by 7 on the right.
# d) Placement legality:
#    - The generator is still output-first. It places thin objects only at cols 0 or 5 and thick
#      objects only at cols W-7 or W-14, while preserving rows.
#    - Output layouts reject all rectangle overlaps, including cross-side overlap between thin inward
#      frames (cols 5-9) and thick inward frames (cols 8-14).
#    - Input layouts reject overlap and also enforce a 1-cell gap between objects so same-colored
#      borders do not accidentally merge into false positives.
# e) Visual observability:
#    - A human can identify the two groups directly from visible thickness and size, because border
#      color is intentionally shared across the sample.
#    - Ordering still depends only on visible horizontal position within each group: thin objects are
#      processed by increasing input column, thick objects by decreasing input column.
# f) Visual cleanliness:
#    - The center of every object stays a solid 3x3 block, so even same-colored borders remain easy
#      to parse.
#    - The generator keeps just two stack depths per side and validates the final sample with
#      main(input) == output, which rejects cluttered accidental constructions.
# g) Change inferability:
#    - No hidden numeric property such as color id participates in grouping or ordering.
#    - The fused behavior is learnable from a few examples because "thin and small go left, thick
#      and large go right" is visible on the grid itself.
# h) New solver assumptions introduced here:
#    - Every intended object has a uniform frame color, a uniform 3x3 center, and uses one of the
#      two supported silhouettes: 5x5 with thickness 1 or 7x7 with thickness 2.
#    - Contained 5x5 detections inside thick 7x7 frames are always false positives rather than real
#      extra objects.
# i) Generator guarantees that match those assumptions:
#    - It builds only those two silhouettes, uses one shared frame color, ensures non-overlap plus a
#      1-cell input gap, and validates every sample by executing the solver.


GRID_H = 22
GRID_W = 22
INNER_SIZE = 3
THIN_SIZE = 5
THICK_SIZE = 7
THIN_THICKNESS = 1
THICK_THICKNESS = 2
MAX_INWARD = 2
N_THIN_RANGE = (3, 4)
N_THICK_RANGE = (3, 4)
ROW_ASSIGN_RETRIES = 80
INPUT_COL_RETRIES = 160
GEN_RETRIES = 1200
INPUT_GAP = 1


_THICK_BORDER_MASK = np.zeros((THICK_SIZE, THICK_SIZE), dtype=bool)
_THICK_BORDER_MASK[:THICK_THICKNESS, :] = True
_THICK_BORDER_MASK[-THICK_THICKNESS:, :] = True
_THICK_BORDER_MASK[:, :THICK_THICKNESS] = True
_THICK_BORDER_MASK[:, -THICK_THICKNESS:] = True


def _span_overlaps(a0: int, a_len: int, b0: int, b_len: int) -> bool:
    return not (int(a0 + a_len - 1) < int(b0) or int(b0 + b_len - 1) < int(a0))


def _rects_overlap(r1: int, c1: int, s1: int, r2: int, c2: int, s2: int) -> bool:
    return _span_overlaps(r1, s1, r2, s2) and _span_overlaps(c1, s1, c2, s2)


def _rects_too_close(
    r1: int,
    c1: int,
    s1: int,
    r2: int,
    c2: int,
    s2: int,
    gap: int = 0,
) -> bool:
    return not (
        int(r1 + s1 - 1 + gap) < int(r2)
        or int(r2 + s2 - 1 + gap) < int(r1)
        or int(c1 + s1 - 1 + gap) < int(c2)
        or int(c2 + s2 - 1 + gap) < int(c1)
    )


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


def _inside(box: tuple[int, int, int, int], r0: int, c0: int, size: int) -> bool:
    br0, bc0, br1, bc1 = box
    return br0 <= r0 and bc0 <= c0 and br1 >= r0 + size - 1 and bc1 >= c0 + size - 1


def _detect_objects(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    objects: list[dict] = []
    thick_boxes: list[tuple[int, int, int, int]] = []

    for r0 in range(H - THICK_SIZE + 1):
        for c0 in range(W - THICK_SIZE + 1):
            patch = crop_bbox(g, (r0, c0, r0 + THICK_SIZE - 1, c0 + THICK_SIZE - 1))
            frame_col = _uniform_non_bg(patch[_THICK_BORDER_MASK], bg)
            if frame_col is None:
                continue
            inner_col = _uniform_non_bg(patch[2:5, 2:5], bg)
            if inner_col is None or inner_col == frame_col:
                continue

            objects.append(
                {
                    "group": "thick",
                    "size": THICK_SIZE,
                    "thickness": THICK_THICKNESS,
                    "frame": int(frame_col),
                    "inner": int(inner_col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch.copy(),
                }
            )
            thick_boxes.append((int(r0), int(c0), int(r0 + THICK_SIZE - 1), int(c0 + THICK_SIZE - 1)))

    for r0 in range(H - THIN_SIZE + 1):
        for c0 in range(W - THIN_SIZE + 1):
            if any(_inside(box, r0, c0, THIN_SIZE) for box in thick_boxes):
                continue

            patch = crop_bbox(g, (r0, c0, r0 + THIN_SIZE - 1, c0 + THIN_SIZE - 1))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            frame_col = _uniform_non_bg(border, bg)
            if frame_col is None:
                continue
            inner_col = _uniform_non_bg(patch[1:4, 1:4], bg)
            if inner_col is None or inner_col == frame_col:
                continue

            objects.append(
                {
                    "group": "thin",
                    "size": THIN_SIZE,
                    "thickness": THIN_THICKNESS,
                    "frame": int(frame_col),
                    "inner": int(inner_col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch.copy(),
                }
            )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move framed squares to side stacks using border thickness, where thickness also fixes size.

    Input structure:
    - The background is the most common color.
    - All movable objects are same-style square frames with solid 3x3 centers, scattered across the grid.
    - Thin objects have a 1-cell border and are 5x5 overall; thick objects have a 2-cell border and are 7x7 overall.

    Rule:
    - Thin 5x5 frames move to the left edge and thick 7x7 frames move to the right edge.
    - Each object keeps its original top row.
    - Thin objects are processed by original column ascending; thick objects are processed by original column descending.
    - Within each group, a row-overlap with an already placed edge object pushes the later object one group-width inward.

    Key insight: The visible border thickness does double duty: it tells you which side the object belongs to and also which footprint it will occupy after stacking.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    objects = _detect_objects(g, bg)

    thin_objects = sorted((obj for obj in objects if obj["group"] == "thin"), key=lambda obj: int(obj["c0"]))
    thin_edge_rows: list[int] = []
    for obj in thin_objects:
        r0 = int(obj["r0"])
        has_overlap = any(_span_overlaps(r0, THIN_SIZE, edge_r0, THIN_SIZE) for edge_r0 in thin_edge_rows)
        dest_c = THIN_SIZE if has_overlap else 0
        paste(out, obj["patch"], r0, dest_c, transparent=bg)
        if not has_overlap:
            thin_edge_rows.append(r0)

    thick_objects = sorted((obj for obj in objects if obj["group"] == "thick"), key=lambda obj: -int(obj["c0"]))
    thick_edge_rows: list[int] = []
    for obj in thick_objects:
        r0 = int(obj["r0"])
        has_overlap = any(_span_overlaps(r0, THICK_SIZE, edge_r0, THICK_SIZE) for edge_r0 in thick_edge_rows)
        dest_c = int(W - 2 * THICK_SIZE) if has_overlap else int(W - THICK_SIZE)
        paste(out, obj["patch"], r0, dest_c, transparent=bg)
        if not has_overlap:
            thick_edge_rows.append(r0)

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
        for r0 in all_rows:
            if all(not _span_overlaps(r0, size, edge_r0, size) for edge_r0 in edge_rows):
                edge_rows.append(int(r0))
                if len(edge_rows) == n_edge:
                    break
        if len(edge_rows) < n_edge:
            continue

        random.shuffle(all_rows)
        inward_rows: list[int] = []
        for r0 in all_rows:
            if not any(_span_overlaps(r0, size, edge_r0, size) for edge_r0 in edge_rows):
                continue
            if any(_span_overlaps(r0, size, inward_r0, size) for inward_r0 in inward_rows):
                continue
            inward_rows.append(int(r0))
            if len(inward_rows) == n_inward:
                break
        if len(inward_rows) == n_inward:
            return edge_rows, inward_rows

    return None


def _build_output_objects(
    edge_rows: list[int],
    inward_rows: list[int],
    inner_colors: list[int],
    frame_color: int,
    size: int,
    thickness: int,
    group: str,
) -> list[dict]:
    objects: list[dict] = []
    color_idx = 0
    edge_col = 0 if group == "thin" else GRID_W - size
    inward_col = size if group == "thin" else GRID_W - 2 * size

    for r0 in edge_rows:
        patch = _make_patch(frame_color, inner_colors[color_idx], size, thickness)
        color_idx += 1
        objects.append(
            {
                "group": group,
                "size": size,
                "thickness": thickness,
                "frame": frame_color,
                "inner": int(patch[thickness, thickness]),
                "r0": int(r0),
                "c0": int(edge_col),
                "is_edge": True,
                "patch": patch,
            }
        )

    for r0 in inward_rows:
        patch = _make_patch(frame_color, inner_colors[color_idx], size, thickness)
        color_idx += 1
        objects.append(
            {
                "group": group,
                "size": size,
                "thickness": thickness,
                "frame": frame_color,
                "inner": int(patch[thickness, thickness]),
                "r0": int(r0),
                "c0": int(inward_col),
                "is_edge": False,
                "patch": patch,
            }
        )

    return objects


def _any_pairwise_conflict(objects: list[dict], gap: int = 0) -> bool:
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _rects_too_close(
                int(objects[i]["r0"]),
                int(objects[i]["c0"]),
                int(objects[i]["size"]),
                int(objects[j]["r0"]),
                int(objects[j]["c0"]),
                int(objects[j]["size"]),
                gap=gap,
            ):
                return True
    return False


def _conflicts_with_forbidden(objects: list[dict], forbidden: list[dict], gap: int = 0) -> bool:
    for obj in objects:
        for other in forbidden:
            if _rects_too_close(
                int(obj["r0"]),
                int(obj["c0"]),
                int(obj["size"]),
                int(other["r0"]),
                int(other["c0"]),
                int(other["size"]),
                gap=gap,
            ):
                return True
    return False


def _build_side_output(
    n: int,
    inner_colors: list[int],
    frame_color: int,
    size: int,
    thickness: int,
    group: str,
    forbidden: list[dict] | None = None,
) -> list[dict] | None:
    forbidden = [] if forbidden is None else forbidden
    for _ in range(ROW_ASSIGN_RETRIES):
        assigned = _assign_output_rows(n, size)
        if assigned is None:
            continue
        objects = _build_output_objects(
            assigned[0],
            assigned[1],
            inner_colors,
            frame_color,
            size,
            thickness,
            group,
        )
        if _any_pairwise_conflict(objects):
            continue
        if _conflicts_with_forbidden(objects, forbidden):
            continue
        return objects
    return None


def _derive_input_positions(
    objects: list[dict],
    ascending: bool,
    blocked: list[dict] | None = None,
) -> list[dict] | None:
    blocked = [] if blocked is None else blocked
    edges = [obj for obj in objects if bool(obj["is_edge"])]
    inwards = [obj for obj in objects if not bool(obj["is_edge"])]
    size = int(objects[0]["size"])

    inward_deps = [
        [idx for idx, edge in enumerate(edges) if _span_overlaps(int(inward["r0"]), size, int(edge["r0"]), size)]
        for inward in inwards
    ]
    if any(len(deps) == 0 for deps in inward_deps):
        return None

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
                pos
                for dep in inward_deps[inward_idx]
                for pos, (kind, idx) in enumerate(order)
                if kind == "edge" and idx == dep
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", inward_idx))

        ordered = [edges[idx] if kind == "edge" else inwards[idx] for kind, idx in order]
        chosen_cols = sorted(random.sample(col_range, len(objects)))
        if not ascending:
            chosen_cols.reverse()

        placed: list[dict] = []
        ok = True
        for obj, in_col in zip(ordered, chosen_cols):
            if int(in_col) == int(obj["c0"]):
                ok = False
                break

            candidate = {
                **obj,
                "c0": int(in_col),
            }
            if any(
                _rects_too_close(
                    int(candidate["r0"]),
                    int(candidate["c0"]),
                    int(candidate["size"]),
                    int(prev["r0"]),
                    int(prev["c0"]),
                    int(prev["size"]),
                    gap=INPUT_GAP,
                )
                for prev in placed
            ):
                ok = False
                break
            if any(
                _rects_too_close(
                    int(candidate["r0"]),
                    int(candidate["c0"]),
                    int(candidate["size"]),
                    int(other["r0"]),
                    int(other["c0"]),
                    int(other["size"]),
                    gap=INPUT_GAP,
                )
                for other in blocked
            ):
                ok = False
                break
            placed.append(candidate)

        if ok and not _any_pairwise_conflict(placed, gap=INPUT_GAP):
            return placed

    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    frame_choices = [c for c in range(10) if c not in {bg, 2, 8}]
    frame_color = random.choice(frame_choices)

    n_thin = random.randint(*N_THIN_RANGE)
    n_thick = random.randint(*N_THICK_RANGE)

    avail_inner = [c for c in range(10) if c not in {bg, frame_color}]
    if len(avail_inner) < n_thin + n_thick:
        return None
    inner_colors = random.sample(avail_inner, n_thin + n_thick)
    thin_colors = inner_colors[:n_thin]
    thick_colors = inner_colors[n_thin:]

    thick_out = _build_side_output(
        n_thick,
        thick_colors,
        frame_color,
        THICK_SIZE,
        THICK_THICKNESS,
        "thick",
    )
    if thick_out is None:
        return None
    thin_out = _build_side_output(
        n_thin,
        thin_colors,
        frame_color,
        THIN_SIZE,
        THIN_THICKNESS,
        "thin",
        forbidden=thick_out,
    )
    if thin_out is None:
        return None

    output_objects = thin_out + thick_out
    if _any_pairwise_conflict(output_objects):
        return None

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    for obj in output_objects:
        paste(output, obj["patch"], int(obj["r0"]), int(obj["c0"]))

    thick_input = _derive_input_positions(thick_out, ascending=False)
    if thick_input is None:
        return None
    thin_input = _derive_input_positions(thin_out, ascending=True, blocked=thick_input)
    if thin_input is None:
        return None

    input_objects = thin_input + thick_input
    if _any_pairwise_conflict(input_objects, gap=INPUT_GAP):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for obj in input_objects:
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
