# Variant of b5ca7ac4
# Dimension: D1 — 分组标准——边框颜色二分法 (match_cond)
# Change: 通过边框颜色将框架分为两组：边框为青色(8)的归为左堆叠组，边框为红色(2)的归为右堆叠组... → 按内部填充颜色的冷暖色调分组

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the D1 variant
# ----------------------------------
# variant_spec:
#   objects:
#     - The visible objects stay identical to the seed task: scattered 5x5 bordered frames.
#     - Border colors are still restricted to teal(8) and red(2), because the task is only
#       changing the grouping criterion, not the object geometry.
#     - The 3x3 interior remains uniform, but its color now carries the grouping signal.
#   semantic_split:
#     - cool interior colors {1 blue, 3 green, 5 gray} -> move to the left stack.
#     - warm interior colors {2 red, 4 yellow, 6 magenta, 7 orange} -> move to the right stack.
#     - Maroon(9) and black(0) are deliberately excluded from interiors so the warm/cool bins
#       are visually cleaner and do not ask the solver to guess ambiguous edge cases.
#   behavior_trace:
#     - Original seed: a red-framed square with blue interior still moved right, because only the
#       frame mattered.
#     - Variant: the same red-framed / blue-filled square moves left, because blue is a cool fill.
#     - Likewise, a teal-framed square with yellow interior now moves right.
#     - This isolates D1 cleanly: the placement, row preservation, overlap test, and inward shift
#       are unchanged; only the left/right membership test is replaced.
#   complexity_calibration:
#     - Generator forces both border colors to appear on both output sides, so frame color is no
#       longer predictive.
#     - Each group uses multiple interior colors from its palette, giving repeated evidence for the
#       warm/cool split instead of a one-off color lookup.
#     - The task still has only two semantic bins, which keeps the inductive burden reasonable.
#   hidden_assumption_audit:
#     - Window scan still assumes every real object is exactly 5x5 with a uniform one-cell frame.
#       Generator preserves this.
#     - Group classification now depends on the interior being a uniform 3x3 fill. Generator keeps
#       interiors uniform, and the solver rejects non-uniform candidate windows.
#     - Ordering logic still depends on input columns: left-group squares are processed by input
#       column ascending, right-group squares by descending. Generator derives input columns so this
#       ordering reproduces the intended edge/inward split.
#     - Overlap logic still assumes at most two horizontal depths per side (edge plus one inward
#       column). Row assignment preserves that invariant.
#   placement_legality:
#     - Output is built first with non-overlapping edge squares and inward squares that row-overlap
#       at least one edge square on the same side.
#     - Input positions are then back-solved with distinct columns and a global non-overlap check.
#     - Final validation runs both the new solver and the original frame-based solver; samples are
#       rejected unless the new solver matches the target output and the old rule would differ.
#   observability:
#     - A human can see red-framed squares ending up on the left and teal-framed squares ending up
#       on the right, so the seed rule is visibly broken.
#     - Across examples, blue/green/gray consistently travel left while red/yellow/magenta/orange
#       travel right, making the semantic split recoverable from 2-4 training pairs.


# --------------- solver ---------------

SQ = 5
AZURE = 8
RED = 2
COOL_INNER_COLORS = (1, 3, 5)
WARM_INNER_COLORS = (2, 4, 6, 7)
ALL_GROUPABLE_INNER_COLORS = set(COOL_INNER_COLORS) | set(WARM_INNER_COLORS)


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (int(a1) < int(b0) or int(b1) < int(a0))


def _tone_side(inner_color: int) -> str | None:
    if int(inner_color) in COOL_INNER_COLORS:
        return "left"
    if int(inner_color) in WARM_INNER_COLORS:
        return "right"
    return None


def _frame_side(frame_color: int) -> str | None:
    if int(frame_color) == AZURE:
        return "left"
    if int(frame_color) == RED:
        return "right"
    return None


def _detect_squares(g: np.ndarray) -> list[dict[str, int | np.ndarray]]:
    H, W = g.shape
    squares: list[dict[str, int | np.ndarray]] = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue

            frame_col = int(border[0])
            if frame_col not in {AZURE, RED}:
                continue

            inner = patch[1:-1, 1:-1]
            inner_vals = np.unique(inner)
            if len(inner_vals) != 1:
                continue

            inner_col = int(inner_vals[0])
            if inner_col == frame_col or inner_col not in ALL_GROUPABLE_INNER_COLORS:
                continue

            squares.append(
                {
                    "frame": frame_col,
                    "inner": inner_col,
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch,
                }
            )
    return squares


def _solve_with_grouping(
    g: np.ndarray,
    side_of_square,
) -> np.ndarray:
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    squares = _detect_squares(g)
    left_squares = sorted(
        [s for s in squares if side_of_square(s) == "left"],
        key=lambda s: int(s["c0"]),
    )
    placed_at_default: list[tuple[int, int]] = []
    for s in left_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(_overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_default.append((r0, r1))

    right_squares = sorted(
        [s for s in squares if side_of_square(s) == "right"],
        key=lambda s: -int(s["c0"]),
    )
    placed_at_default = []
    for s in right_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(_overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_default.append((r0, r1))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack bordered 5×5 squares by the warm/cool category of their interior color.

    Input structure:
    - bg = most common color.
    - Red (2) and teal (8) framed 5×5 squares are scattered across the grid.
    - Each square has a uniform 3×3 interior whose color belongs to either a cool set
      (blue/green/gray) or a warm set (red/yellow/magenta/orange).

    Rule:
    - Cool-filled squares move to the left stack; warm-filled squares move to the right stack.
    - The frame color is preserved but does not decide the side.
    - Each square keeps its original row.
    - Left-group squares are processed by input column ascending; right-group squares by
      input column descending.
    - If a square row-overlaps an already placed edge square on its side, it shifts one
      square-width inward instead of occupying the edge column.

    Key insight: The geometry is unchanged from the seed task, but the decisive feature has
    moved from the border to the interior: identical frame colors can now land on opposite
    sides depending on whether the 3×3 fill reads as cool or warm.
    """
    g = np.array(grid, dtype=int)
    return _solve_with_grouping(g, lambda s: _tone_side(int(s["inner"])))


def _frame_based_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    return _solve_with_grouping(g, lambda s: _frame_side(int(s["frame"])))


# --------------- generator ---------------

# Generator configuration mirrors the seed task geometry and only changes the grouping semantics.
GRID_H, GRID_W = 22, 22
N_COOL_RANGE = (3, 4)
N_WARM_RANGE = (3, 4)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500
BG_CHOICES = (0, 9)


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(r_a - r_b) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(r1 - r2) < SQ and abs(c1 - c2) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, object]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(
                int(positions[i][0]),
                int(positions[i][1]),
                int(positions[j][0]),
                int(positions[j][1]),
            ):
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


def _sample_frames(n: int) -> list[int]:
    frames = [AZURE, RED]
    while len(frames) < n:
        frames.append(random.choice([AZURE, RED]))
    random.shuffle(frames)
    return frames


def _sample_inner_colors(palette: tuple[int, ...], n: int) -> list[int]:
    colors = list(palette)
    random.shuffle(colors)
    result = colors[: min(len(colors), n)]
    while len(result) < n:
        result.append(random.choice(palette))
    random.shuffle(result)
    return result


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    inner_colors: list[int],
    frame_colors: list[int],
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray, int, int]]:
    placed: list[tuple[int, int, bool, np.ndarray, int, int]] = []
    for i, r in enumerate(edge_rows):
        frame = int(frame_colors[i])
        inner = int(inner_colors[i])
        p = _make_patch(frame, inner)
        paste(output, p, r, edge_col)
        placed.append((r, edge_col, True, p, frame, inner))
    for i, r in enumerate(inward_rows):
        idx = len(edge_rows) + i
        frame = int(frame_colors[idx])
        inner = int(inner_colors[idx])
        p = _make_patch(frame, inner)
        paste(output, p, r, inward_col)
        placed.append((r, inward_col, False, p, frame, inner))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray, int, int]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, oc, p) for (r, oc, is_edge, p, _, _) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p, _, _) in sq_list if not is_edge]
    inward_deps = [
        [i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)]
        for (ir, _, _) in inwards
    ]

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

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, oc, p) in enumerate(ordered_squares):
            ic = int(chosen_cols[i])
            if ic == int(oc):
                ok = False
                break
            result.append((int(r), ic, p))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _mismatch_count(groups: list[tuple[int, int]]) -> int:
    return sum(1 for frame, inner in groups if _frame_side(frame) != _tone_side(inner))


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    n_cool = random.randint(*N_COOL_RANGE)
    n_warm = random.randint(*N_WARM_RANGE)

    cool_assign = _assign_output_rows(n_cool)
    if cool_assign is None:
        return None
    warm_assign = _assign_output_rows(n_warm)
    if warm_assign is None:
        return None
    cool_edge_rows, cool_inward_rows = cool_assign
    warm_edge_rows, warm_inward_rows = warm_assign

    cool_frames = _sample_frames(n_cool)
    warm_frames = _sample_frames(n_warm)
    cool_inner_colors = _sample_inner_colors(COOL_INNER_COLORS, n_cool)
    warm_inner_colors = _sample_inner_colors(WARM_INNER_COLORS, n_warm)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    cool_out = _place_side_squares(
        output,
        cool_edge_rows,
        cool_inward_rows,
        cool_inner_colors,
        cool_frames,
        edge_col=0,
        inward_col=SQ,
    )
    warm_out = _place_side_squares(
        output,
        warm_edge_rows,
        warm_inward_rows,
        warm_inner_colors,
        warm_frames,
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )

    all_group_specs = [(frame, inner) for (_, _, _, _, frame, inner) in cool_out + warm_out]
    if _mismatch_count(all_group_specs) < 2:
        return None
    if len({inner for (_, inner) in all_group_specs if inner in COOL_INNER_COLORS}) < 2:
        return None
    if len({inner for (_, inner) in all_group_specs if inner in WARM_INNER_COLORS}) < 2:
        return None

    cool_input = _derive_input_positions(cool_out, ascending=True)
    if cool_input is None:
        return None
    warm_input = _derive_input_positions(warm_out, ascending=False)
    if warm_input is None:
        return None

    all_positions = cool_input + warm_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, p in all_positions:
        paste(inp, p, r, c)

    try:
        if not np.array_equal(main(inp), output):
            return None
        if np.array_equal(_frame_based_main(inp), output):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
