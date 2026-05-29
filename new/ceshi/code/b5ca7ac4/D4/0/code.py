# Variant of b5ca7ac4
# Dimension: D4 — 源对象——组标识方式 (src)
# Change: 通过边框颜色标识分组：青色(8)边框 vs 红色(2)边框，这两种颜色在所有示例中固定... → 通过边框线型标识——实线 vs 虚线边框

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis
# a) Concrete behavior and trace:
#    - Objects remain 5x5 framed squares with a uniform 3x3 interior, so the
#      motion rule and output geometry stay unchanged.
#    - Both groups now share one border color (fixed teal/8). The grouping cue
#      is the perimeter pattern:
#        solid:
#          8 8 8 8 8
#          8 i i i 8
#          8 i i i 8
#          8 i i i 8
#          8 8 8 8 8
#        dashed:
#          8 . 8 . 8
#          . i i i .
#          8 i i i 8
#          . i i i .
#          8 . 8 . 8
#      Here "." means background. The dashed frame is still visibly square, but
#      its border continuity differs from the solid frame.
#    - Example trace: if two squares at rows 4 and 6 overlap vertically and the
#      left-moving group is processed by increasing input column, then the first
#      one takes col 0 and the second takes col 5. In the original task, the
#      solver decided the group from frame color 8 vs 2. In this variant, both
#      use color 8, so the solver must first classify "solid" vs "dashed" and
#      only then apply the same left/right placement logic.
# b) Complexity calibration:
#    - The change is visible in every object because each example contains
#      several solid and several dashed frames using the same border color.
#    - Only two border styles exist, preventing combinatorial ambiguity.
#    - The 5x5 size is kept, so the variant does not introduce extra object
#      scales or new placement freedoms.
# c) Original assumptions that break under the variant:
#    - `border.min() == border.max()` only works for solid borders; dashed
#      borders intentionally mix frame color and background on the perimeter.
#    - `frame_col in {2, 8}` is obsolete because color no longer identifies the
#      group.
#    - Filtering detected squares by `frame == 8` vs `frame == 2` must be
#      replaced by filtering by border style.
#    - `paste(..., transparent=bg)` remains correct and important: dashed gaps
#      are literal background cells and should stay transparent when moved.
# d) Placement legality:
#    - Generation still treats every object as occupying its full 5x5 bbox.
#      This avoids overlapping detection windows even though dashed borders
#      contain background holes.
#    - Output columns remain {0, 5} for the left-moving group and {W-5, W-10}
#      for the right-moving group, so no new geometric conflicts are added.
# e) Visual observability:
#    - Humans can infer the grouping cue because all frames share one color but
#      split into clearly continuous vs interrupted outlines across 6-9 objects.
#    - Sorting never depends on hidden values such as color id or object index;
#      it still depends only on visible input columns.
# f) Visual neatness:
#    - The grid keeps the original sparse layout and only uses one extra visual
#      distinction (line continuity), so the examples remain easy to parse.
# g) Color inference:
#    - Interior colors are decorative identifiers only. They stay uniform and
#      avoid the shared frame color and the background, but they do not encode
#      grouping or ordering.
# h) Variant anti-degeneration check:
#    - Every generated input contains both solid and dashed frames while all
#      detected frames share the same border color, so the original color-based
#      rule cannot explain the outputs.
# i) New solver assumptions and generator guarantees:
#    - The solver assumes each valid object is an exact 5x5 solid or dashed
#      template with a uniform non-background 3x3 interior.
#    - The generator enforces that exact template, keeps 5x5 bboxes disjoint,
#      and validates each sample by re-running `main`.


GRID_H, GRID_W = 22, 22
SQ = 5
FRAME_COLOR = 8
N_SOLID_RANGE = (3, 4)
N_DASHED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500


def _border_mask() -> np.ndarray:
    mask = np.zeros((SQ, SQ), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


BORDER_MASK = _border_mask()
PARITY_MASK = (np.indices((SQ, SQ)).sum(axis=0) % 2) == 0
DASH_COLOR_MASK = BORDER_MASK & PARITY_MASK
DASH_GAP_MASK = BORDER_MASK & (~PARITY_MASK)


def _classify_patch(patch: np.ndarray, bg: int) -> tuple[str, int] | None:
    inner = patch[1:-1, 1:-1]
    if int(inner.min()) != int(inner.max()):
        return None
    inner_color = int(inner[0, 0])
    if inner_color == int(bg):
        return None

    solid_border = patch[BORDER_MASK]
    if int(solid_border.min()) == int(solid_border.max()):
        frame_color = int(solid_border[0])
        if frame_color != int(bg) and frame_color != inner_color:
            return "solid", frame_color

    dash_cells = patch[DASH_COLOR_MASK]
    frame_color = int(dash_cells[0])
    if (
        int(dash_cells.min()) == int(dash_cells.max())
        and frame_color != int(bg)
        and frame_color != inner_color
        and bool(np.all(patch[DASH_GAP_MASK] == int(bg)))
    ):
        return "dashed", frame_color

    return None


def _make_patch(frame_color: int, inner_color: int, bg: int, *, solid: bool) -> np.ndarray:
    if solid:
        patch = np.full((SQ, SQ), frame_color, dtype=int)
    else:
        patch = np.full((SQ, SQ), bg, dtype=int)
        patch[DASH_COLOR_MASK] = frame_color
    patch[1:-1, 1:-1] = inner_color
    return patch


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, er) for er in edge_rows):
            edge_rows.append(r)
            if len(edge_rows) == n_edge:
                break
    if len(edge_rows) < n_edge:
        return None

    random.shuffle(all_rows)
    inward_rows: list[int] = []
    for r in all_rows:
        if not any(_rows_overlap(r, er) for er in edge_rows):
            continue
        if any(_rows_overlap(r, ir) for ir in inward_rows):
            continue
        inward_rows.append(r)
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None
    return edge_rows, inward_rows


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    colors: list[int],
    *,
    solid: bool,
    bg: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        patch = _make_patch(FRAME_COLOR, colors[i], bg, solid=solid)
        paste(output, patch, r, edge_col, transparent=bg)
        placed.append((r, edge_col, True, patch))
    for i, r in enumerate(inward_rows):
        patch = _make_patch(FRAME_COLOR, colors[len(edge_rows) + i], bg, solid=solid)
        paste(output, patch, r, inward_col, transparent=bg)
        placed.append((r, inward_col, False, patch))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    *,
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]
    inward_deps = [[i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)] for (ir, _, _) in inwards]

    n = len(sq_list)
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
                    for d in inward_deps[iw_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == d
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", iw_idx))

        ordered_squares = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]
        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result = []
        ok = True
        for i, (r, oc, patch) in enumerate(ordered_squares):
            ic = chosen_cols[i]
            if ic == oc:
                ok = False
                break
            result.append((r, ic, patch))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered 5×5 squares to opposite grid edges by border continuity.

    Input structure:
    - bg = most common color.
    - Several 5×5 teal-framed squares with uniform 3×3 interiors are scattered across the grid.
    - Some frames are continuous solid outlines; others are interrupted dashed outlines using bg-colored gaps.

    Rule:
    - Solid-frame squares move to the left edge (col 0); dashed-frame squares move to the right edge (col W−5).
    - Each square keeps its original row.
    - Process solid squares by original column ascending; process dashed squares by original column descending.
    - If a square's row span overlaps a square already placed at that side's edge column, place it one square-width inward instead.

    Key insight:
    - Color no longer separates the groups: both groups share the same frame color, so the solver must read the border's continuity pattern before applying the original edge-stacking rule.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            classified = _classify_patch(patch, bg)
            if classified is None:
                continue
            style, frame_color = classified
            squares.append(
                {
                    "style": style,
                    "frame": frame_color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch,
                }
            )

    solid_squares = sorted([s for s in squares if s["style"] == "solid"], key=lambda s: int(s["c0"]))
    placed_at_default: list[tuple[int, int]] = []
    for s in solid_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_default.append((r0, r1))

    dashed_squares = sorted([s for s in squares if s["style"] == "dashed"], key=lambda s: -int(s["c0"]))
    placed_at_default = []
    for s in dashed_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_default.append((r0, r1))

    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != FRAME_COLOR])
    n_solid = random.randint(*N_SOLID_RANGE)
    n_dashed = random.randint(*N_DASHED_RANGE)

    solid_assign = _assign_output_rows(n_solid)
    if solid_assign is None:
        return None
    dashed_assign = _assign_output_rows(n_dashed)
    if dashed_assign is None:
        return None
    solid_edge_rows, solid_inward_rows = solid_assign
    dashed_edge_rows, dashed_inward_rows = dashed_assign

    avail_solid = [c for c in range(10) if c not in {bg, FRAME_COLOR}]
    avail_dashed = [c for c in range(10) if c not in {bg, FRAME_COLOR}]
    if len(avail_solid) < n_solid or len(avail_dashed) < n_dashed:
        return None
    solid_colors = random.sample(avail_solid, n_solid)
    dashed_colors = random.sample(avail_dashed, n_dashed)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    solid_out = _place_side_squares(
        output,
        solid_edge_rows,
        solid_inward_rows,
        solid_colors,
        solid=True,
        bg=bg,
        edge_col=0,
        inward_col=SQ,
    )
    dashed_out = _place_side_squares(
        output,
        dashed_edge_rows,
        dashed_inward_rows,
        dashed_colors,
        solid=False,
        bg=bg,
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )

    solid_input = _derive_input_positions(solid_out, ascending=True)
    if solid_input is None:
        return None
    dashed_input = _derive_input_positions(dashed_out, ascending=False)
    if dashed_input is None:
        return None

    all_positions = solid_input + dashed_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in all_positions:
        paste(inp, patch, r, c, transparent=bg)
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
