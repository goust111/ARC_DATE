# Combo variant of b5ca7ac4 (index 8)
# combo_spec: D4/2+D8/2
# Name: 边框内部颜色互换+内外色同则为一组

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis
# ----------------
# a) Concrete behavior:
#    - The original task separated groups by fixed border colors: teal frames
#      always moved left and red frames always moved right, with no appearance
#      change during movement.
#    - This combo removes that chromatic grouping cue.  Every input object now
#      shares one visible border color.  A 5x5 object whose 3x3 interior differs
#      from the border is the left-going group; a 5x5 object whose interior
#      matches the border is the right-going "solid block" group.
#    - Movement order and stacking stay structurally the same: left-group
#      objects are processed by increasing input column and right-group objects
#      by decreasing input column; row overlap with an already placed edge
#      object pushes the later object one object-width inward.
#    - The new transform is fused with that regrouping: after a left-group frame
#      is moved, its border color and inner color swap roles.  Example:
#        input framed object: border 6, center 3
#        old D4-only output: border 6, center 3 at the left stack
#        combo output:      border 3, center 6 at the left stack
#      A solid 6/6 block moved to the right stack still looks the same after the
#      swap, because exchanging equal colors is a no-op.
#    - The visible difference from the original task is therefore twofold:
#      group membership is read from "framed vs solid" rather than from teal vs
#      red, and moved framed objects no longer preserve their input coloring.
#
# b) Complexity calibration:
#    - The generator keeps 3-4 framed objects and 3-5 solid blocks, so both
#      structural groups repeat enough times to be learnable from a few examples.
#    - All input borders share one color, which makes the D4 change obvious.
#      Framed interiors are unique within a grid so each left-going object stays
#      individually trackable before and after the color swap.
#    - Solid blocks intentionally repeat the shared border color.  That repeated
#      "all one color" appearance gives a strong visual cue for the right-going
#      group instead of turning each object into an arbitrary color case.
#
# c) Hidden assumptions that must change:
#    - The original solver assumed frame colors were exactly {2, 8}; this no
#      longer holds, so detection must accept any non-background uniform border.
#    - The original detector rejected inner==border; that would wrongly discard
#      the solid-group objects here, so solids must now be accepted as valid
#      movable objects.
#    - The original paste step reused the input patch unchanged.  That is no
#      longer correct for framed objects, because movement now swaps border and
#      interior colors.
#    - Because many objects share one border color, touching inputs could create
#      accidental larger regions and false 5x5 scans.  The generator therefore
#      enforces a one-cell input gap between all objects.
#    - Exact row matches between edge and inward solid blocks would form a 5x10
#      same-color bar on the right side, which is legal but visually muddy.  The
#      generator avoids those exact matches to keep the stacked solids legible.
#
# d) Placement legality:
#    - Objects remain fixed-size 5x5 patches on a 22x22 canvas and are placed
#      entirely in bounds.
#    - Input placement is output-first: choose valid edge/inward rows per side,
#      render the intended output, then derive non-overlapping input columns that
#      reproduce the solver's processing order constraints.
#    - Input objects are pairwise separated by at least one cell, and the
#      finished grid is validated by running main() on it.
#
# e) Visual observability:
#    - A human can see the new grouping cue directly: same border color, but
#      some objects are hollow-looking frames while others are solid blocks.
#    - The color-change rule is also visible: left-going framed objects arrive
#      with their old center color on the border and the shared border color in
#      the center.  This happens several times per grid with different interior
#      colors, giving repeated evidence for the swap.
#    - No hidden numeric color ordering is used; only visible row overlap, input
#      column order, and the framed-vs-solid structure control the behavior.
#
# f) Visual cleanliness:
#    - The output still uses only two depths per side, preserving the neat
#      stacked layout of the original task.
#    - Right-side solids are row-staggered instead of perfectly aligned, which
#      prevents them from merging into ambiguous long bars.
#    - Left-side frames keep distinct border colors after swapping, so adjacent
#      stacked frames remain easy to segment visually.
#
# g) Change inferability:
#    - The swap is tightly anchored to object structure: only moved framed
#      objects visibly change, and they all change in the same way.
#    - The generator forbids left-group interiors from equaling the shared input
#      border color, so every framed object truly changes appearance after the
#      swap.
#    - A useful anti-degeneracy check is to compare the combo output against the
#      D4-only "move but do not swap" baseline on the same input.  For valid
#      generated samples they must differ, because every left-going framed object
#      contributes changed cells after movement.
#
# Post-change solver assumptions reviewed against the generator:
#    - Every movable object is a 5x5 patch with a uniform border and a uniform
#      3x3 interior: guaranteed.
#    - Input objects may share a border color, so classification must depend
#      only on inner==border vs inner!=border: guaranteed by construction.
#    - There is at least one framed object and at least one solid object in each
#      sample because the count ranges are 3-4 and 3-5 respectively.
#    - Input objects are separated enough that the window scan finds each object
#      exactly once in practice: guaranteed by the one-cell gap rule.
#    - Every framed object changes under the border/inner swap, so the combo
#      does not collapse to the D4-only variant on generated samples.


GRID_H, GRID_W = 22, 22
SQ = 5
N_FRAME_RANGE = (3, 4)
N_SOLID_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 160
GEN_RETRIES = 1500
INPUT_GAP = 1


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _swap_border_and_inner(patch: np.ndarray) -> np.ndarray:
    """Swap the uniform border color with the uniform 3x3 interior color."""
    frame_color = int(patch[0, 0])
    inner_color = int(patch[1, 1])
    out = np.full((SQ, SQ), inner_color, dtype=int)
    out[1:-1, 1:-1] = frame_color
    return out


def _find_objects(grid: np.ndarray) -> list[dict[str, object]]:
    """Detect valid 5x5 objects and classify them by whether inner==border."""
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

            frame_color = int(border[0])
            if frame_color == bg:
                continue

            inner = patch[1:-1, 1:-1]
            if int(inner.min()) != int(inner.max()):
                continue

            inner_color = int(inner[0, 0])
            objects.append(
                {
                    "frame": frame_color,
                    "inner": inner_color,
                    "is_solid": bool(inner_color == frame_color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch,
                }
            )

    return objects


def _solve(grid: np.ndarray, *, swap_on_move: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    objects = _find_objects(g)

    framed_objects = sorted(
        [obj for obj in objects if not bool(obj["is_solid"])],
        key=lambda obj: int(obj["c0"]),
    )
    placed_at_edge: list[tuple[int, int]] = []
    for obj in framed_objects:
        r0 = int(obj["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        patch = np.array(obj["patch"], dtype=int)
        if swap_on_move:
            patch = _swap_border_and_inner(patch)
        paste(out, patch, r0, dest_c)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    solid_objects = sorted(
        [obj for obj in objects if bool(obj["is_solid"])],
        key=lambda obj: -int(obj["c0"]),
    )
    placed_at_edge = []
    for obj in solid_objects:
        r0 = int(obj["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        patch = np.array(obj["patch"], dtype=int)
        if swap_on_move:
            patch = _swap_border_and_inner(patch)
        paste(out, patch, r0, dest_c)
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move same-bordered 5×5 objects to side stacks, then swap border and center colors on arrival.

    Input structure:
    - The background is the most common color.
    - Every movable object is a 5×5 patch with a uniform outer ring and a uniform 3×3 center.
    - In a typical puzzle instance all objects share one input border color, but they split into
      two visual subtypes: true frames whose center differs from the border, and solid-looking
      blocks whose center matches the border.

    Rule:
    - Framed objects move to the left stack and solid-looking objects move to the right stack.
    - Each object keeps its original row.
    - Process framed objects by increasing input column and solid blocks by decreasing input
      column; a row-overlap with an already placed edge object pushes the later one one square
      inward on that side.
    - When an object is pasted into the output, exchange its border color with its center color.
      Solid blocks therefore look unchanged because both regions already have the same color.

    Key insight: grouping is determined by structure rather than by a fixed color id, and the
    moved copy is not identical to the input object.  Solve the task by first separating framed
    objects from solid blocks, then applying the original edge-stacking logic while remembering
    to swap border and interior colors for the pasted result.
    """
    return _solve(grid, swap_on_move=True)


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_conflict(r1: int, c1: int, r2: int, c2: int, gap: int = 0) -> bool:
    return abs(int(r1) - int(r2)) < SQ + int(gap) and abs(int(c1) - int(c2)) < SQ + int(gap)


def _any_pairwise_conflict(positions: list[tuple], gap: int = 0) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_conflict(positions[i][0], positions[i][1], positions[j][0], positions[j][1], gap=gap):
                return True
    return False


def _assign_output_rows(
    n: int,
    *,
    forbid_exact_overlap_with_edge: bool = False,
) -> tuple[list[int], list[int]] | None:
    """Pick edge and inward rows for one side of the output."""
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
    input_patches: list[np.ndarray],
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render transformed output objects and keep input patches for back-solving."""
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        input_patch = input_patches[i]
        paste(output, _swap_border_and_inner(input_patch), r, edge_col)
        placed.append((int(r), int(edge_col), True, input_patch))
    for i, r in enumerate(inward_rows):
        input_patch = input_patches[len(edge_rows) + i]
        paste(output, _swap_border_and_inner(input_patch), r, inward_col)
        placed.append((int(r), int(inward_col), False, input_patch))
    return placed


def _derive_input_positions(
    obj_list: list[tuple[int, int, bool, np.ndarray]],
    *,
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign input columns that reproduce the solver's edge-vs-inward choices."""
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
    bg = random.randint(0, 9)
    shared_frame = random.choice([c for c in range(10) if c != bg])
    n_frame = random.randint(*N_FRAME_RANGE)
    n_solid = random.randint(*N_SOLID_RANGE)

    frame_assign = _assign_output_rows(n_frame, forbid_exact_overlap_with_edge=True)
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
    frame_input_patches = [_make_patch(shared_frame, inner) for inner in frame_inner_colors]
    solid_input_patches = [_make_patch(shared_frame, shared_frame) for _ in range(n_solid)]

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    frame_out = _place_side_objects(
        output,
        frame_edge_rows,
        frame_inward_rows,
        frame_input_patches,
        edge_col=0,
        inward_col=SQ,
    )
    solid_out = _place_side_objects(
        output,
        solid_edge_rows,
        solid_inward_rows,
        solid_input_patches,
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
