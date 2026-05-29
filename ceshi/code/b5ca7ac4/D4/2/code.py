# Variant of b5ca7ac4
# Dimension: D4 — 源对象——组标识方式 (src)
# Change: 通过边框颜色标识分组：青色(8)边框 vs 红色(2)边框，这两种颜色在所有示例中固定... → 通过内部颜色是否与边框颜色相同来标识
#
# Design analysis
# - concretization:
#   All movable objects remain 5x5 border-based squares. The variant keeps the
#   original motion rule, rows, side destinations, and overlap-driven inward
#   stacking, but removes the fixed azure/red split. Every object now shares one
#   visible frame color. The left-going group is made of normal frames whose 3x3
#   interior is a different uniform color; the right-going group is made of
#   "solid" 5x5 blocks whose interior matches the frame, so the whole patch is
#   one color. A solver must separate outline-vs-solid, not color-8-vs-color-2.
# - behavior trace:
#   Original: a red-bordered square always went right, an azure-bordered square
#   always went left. Variant: two same-bordered objects can go to opposite
#   sides. Example: if both use border color 6, then a 6-frame with inner 3 goes
#   left while a full 6-colored 5x5 block goes right. Column order still decides
#   which overlapping object keeps the edge slot.
# - complexity calibration:
#   The structural cue is binary and repeated several times per grid: 3-4 frames
#   and 3-5 solids. Shared frame color makes the D4 change visually obvious.
#   Frame interiors stay uniform and mostly unique to keep scattered objects easy
#   to individuate; solids are intentionally repeated so "solidness" reads as a
#   category rather than as a special color.
# - implicit assumptions audit:
#   The original solver assumed frame colors were exactly {2, 8}; that must be
#   removed. It also assumed every detected object had an interior different from
#   its border, which would wrongly discard the solid-group objects here. The new
#   detector therefore accepts any non-background uniform border with any uniform
#   interior, then classifies via inner==border. Because solid objects of the
#   same color can visually fuse if they touch, the generator guarantees a 1-cell
#   gap in the input and forbids exact row matches between right-edge solids and
#   their inward overlaps in the output, so the step-shaped unions stay legible.
# - legality:
#   Input objects are non-overlapping and separated by at least one cell, which
#   prevents accidental larger same-color regions from creating false 5x5 scans.
#   Output uses the same two-depth stacking as the original task: edge column and
#   one inward column only.
# - observability:
#   The key control variable is visible in every example: "interior same as
#   border" vs "interior different from border". Side assignment never depends on
#   color ids, and the shared border color is allowed to vary across examples so
#   color itself cannot explain the grouping.

from __future__ import annotations

import random
import numpy as np
from arc_utils import crop_bbox, get_bg, paste


def _find_objects(grid: np.ndarray) -> list[dict[str, object]]:
    """Detect 5x5 bordered objects and classify them by inner==border."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    objects: list[dict[str, object]] = []

    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue

            frame_col = int(border[0])
            if frame_col == bg:
                continue

            inner = patch[1:-1, 1:-1]
            if int(inner.min()) != int(inner.max()):
                continue

            inner_col = int(inner[0, 0])
            objects.append(
                {
                    "frame": frame_col,
                    "inner": inner_col,
                    "is_solid": bool(inner_col == frame_col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch,
                }
            )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move scattered 5×5 same-bordered objects to the grid edges by whether they look framed or solid.

    Input structure:
    - bg = most common color.
    - Every movable 5×5 object has one uniform border color, but there are two visual subtypes:
      normal frames with a different-colored 3×3 interior, and solid-looking 5×5 blocks whose
      interior matches the border.

    Rule:
    - Normal frames → left edge (col 0); solid blocks → right edge (col W−5).
    - Each object keeps its original row.
    - Process normal frames left-to-right, solid blocks right-to-left (nearest-to-edge first).
    - If an object's rows overlap an already-placed edge object on its side, place it one square-width inward instead.

    Key insight: The grouping cue is structural rather than chromatic: identical border colors can
    send objects to opposite sides, and the edge-vs-inward slot is still determined by processing
    order plus row-span overlap.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    objects = _find_objects(g)

    frame_objects = sorted([obj for obj in objects if not bool(obj["is_solid"])], key=lambda obj: int(obj["c0"]))
    placed_at_default: list[tuple[int, int]] = []
    for obj in frame_objects:
        r0 = int(obj["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = SQ if has_overlap else 0
        paste(out, np.array(obj["patch"], dtype=int), r0, dest_c)
        if dest_c == 0:
            placed_at_default.append((r0, r1))

    solid_objects = sorted([obj for obj in objects if bool(obj["is_solid"])], key=lambda obj: -int(obj["c0"]))
    placed_at_default = []
    for obj in solid_objects:
        r0 = int(obj["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, np.array(obj["patch"], dtype=int), r0, dest_c)
        if dest_c == int(W - SQ):
            placed_at_default.append((r0, r1))

    return out


# --------------- generator (c) ---------------

# --- Configuration ---
GRID_H, GRID_W = 22, 22
SQ = 5
N_FRAME_RANGE = (3, 4)
N_SOLID_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 120
GEN_RETRIES = 1200
INPUT_GAP = 1


def _make_patch(frame_color: int, inner_color: int | None) -> np.ndarray:
    """Create a SQxSQ object; inner_color=None means a solid 5x5 block."""
    p = np.full((SQ, SQ), frame_color, dtype=int)
    if inner_color is not None:
        p[1:-1, 1:-1] = int(inner_color)
    return p


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Check whether two SQ-tall row spans overlap."""
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_conflict(r1: int, c1: int, r2: int, c2: int, gap: int = 0) -> bool:
    """Check whether two SQxSQ rectangles overlap or violate the requested gap."""
    return abs(int(r1) - int(r2)) < SQ + int(gap) and abs(int(c1) - int(c2)) < SQ + int(gap)


def _any_pairwise_conflict(positions: list[tuple], gap: int = 0) -> bool:
    """Return True if any two (r, c, ...) entries overlap or are too close."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_conflict(positions[i][0], positions[i][1], positions[j][0], positions[j][1], gap=gap):
                return True
    return False


def _assign_output_rows(n: int, *, forbid_exact_overlap_with_edge: bool = False) -> tuple[list[int], list[int]] | None:
    """Pick edge and inward row positions for one side."""
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, er) for er in edge_rows):
            edge_rows.append(int(r))
            if len(edge_rows) == n_edge:
                break
    if len(edge_rows) < n_edge:
        return None

    random.shuffle(all_rows)
    inward_rows: list[int] = []
    for r in all_rows:
        if not any(_rows_overlap(r, er) for er in edge_rows):
            continue
        if forbid_exact_overlap_with_edge and any(int(r) == int(er) for er in edge_rows):
            continue
        if any(_rows_overlap(r, ir) for ir in inward_rows):
            continue
        inward_rows.append(int(r))
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None

    return edge_rows, inward_rows


def _place_side_objects(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    patches: list[np.ndarray],
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render edge and inward objects onto output, preserving patch identity."""
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        patch = patches[i]
        paste(output, patch, r, edge_col)
        placed.append((int(r), int(edge_col), True, patch))
    for i, r in enumerate(inward_rows):
        patch = patches[len(edge_rows) + i]
        paste(output, patch, r, inward_col)
        placed.append((int(r), int(inward_col), False, patch))
    return placed


def _derive_input_positions(
    obj_list: list[tuple[int, int, bool, np.ndarray]], ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign random input columns that reproduce the solver's side-order behavior."""
    edges = [(r, oc, p) for (r, oc, is_edge, p) in obj_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in obj_list if not is_edge]
    inward_deps = [[i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)] for (ir, _, _) in inwards]

    n = len(obj_list)
    col_range = list(range(0, GRID_W - SQ + 1))
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", i) for i in edge_indices]

        inward_order = list(range(len(inwards)))
        random.shuffle(inward_order)
        for iw_idx in inward_order:
            latest_pos = max(
                (
                    pos
                    for dep in inward_deps[iw_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", iw_idx))

        ordered_objects = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]

        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, oc, p) in enumerate(ordered_objects):
            ic = int(chosen_cols[i])
            if ic == int(oc):
                ok = False
                break
            result.append((int(r), ic, p))
        if not ok:
            continue

        if not _any_pairwise_conflict(result, gap=INPUT_GAP):
            return result

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    bg = random.randint(0, 9)
    shared_frame = random.choice([c for c in range(10) if c != bg])
    n_frame = random.randint(*N_FRAME_RANGE)
    n_solid = random.randint(*N_SOLID_RANGE)

    frame_assign = _assign_output_rows(n_frame)
    if frame_assign is None:
        return None
    solid_assign = _assign_output_rows(n_solid, forbid_exact_overlap_with_edge=True)
    if solid_assign is None:
        return None
    frame_edge_rows, frame_inward_rows = frame_assign
    solid_edge_rows, solid_inward_rows = solid_assign

    avail_inner = [c for c in range(10) if c not in {bg, shared_frame}]
    if len(avail_inner) < n_frame:
        return None
    frame_inner_colors = random.sample(avail_inner, n_frame)
    frame_patches = [_make_patch(shared_frame, inner) for inner in frame_inner_colors]
    solid_patches = [_make_patch(shared_frame, None) for _ in range(n_solid)]

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    frame_out = _place_side_objects(output, frame_edge_rows, frame_inward_rows, frame_patches, edge_col=0, inward_col=SQ)
    solid_out = _place_side_objects(
        output,
        solid_edge_rows,
        solid_inward_rows,
        solid_patches,
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )

    frame_input = _derive_input_positions(frame_out, ascending=True)
    if frame_input is None:
        return None
    solid_input = _derive_input_positions(solid_out, ascending=False)
    if solid_input is None:
        return None

    all_positions = frame_input + solid_input
    if _any_pairwise_conflict(all_positions, gap=INPUT_GAP):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, p in all_positions:
        paste(inp, p, r, c)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
