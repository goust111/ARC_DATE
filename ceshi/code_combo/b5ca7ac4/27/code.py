# Combo variant of b5ca7ac4 (index 27)
# combo_spec: D1/0+D8/0
# Name: 冷暖分组+边框统一化

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the combo variant
# -------------------------------------
# concretization_and_trace:
#   - Visible objects remain the seed task's scattered 5x5 one-cell-thick frames with a
#     uniform 3x3 interior.
#   - The first change is semantic: side membership is no longer tied to frame color.
#     Example: a red-framed square with blue interior used to move right; now it moves left
#     because blue is in the cool bin. A teal-framed square with yellow interior used to move
#     left; now it moves right because yellow is warm.
#   - The second change applies after the move: every moved frame is recolored to one common
#     border color. So the output no longer preserves the original red/teal markers at all.
#     The interior colors stay unchanged, rows stay unchanged, and the left/right stacking
#     geometry stays unchanged.
# complexity_calibration:
#   - Only two semantic groups are used, keeping the induction burden manageable.
#   - Interior colors come from small repeated palettes:
#       cool = {1 blue, 3 green, 5 gray}
#       warm = {2 red, 4 yellow, 6 magenta, 7 orange}
#   - The generator forces both input frame colors to appear on both output sides, so border
#     color is visibly unreliable as a grouping cue.
#   - The output frame recolor is a single global color, which makes the D8 change obvious
#     without adding another many-to-many color mapping puzzle.
# hidden_assumption_audit:
#   - Window scanning still assumes true objects are exactly 5x5 with a uniform border and
#     a uniform 3x3 center. The generator preserves that exactly.
#   - Group classification now depends on the interior being one of the allowed warm/cool
#     colors. The solver rejects ambiguous interiors; the generator only emits supported ones.
#   - Ordering still depends on input columns: left-group squares are processed ascending and
#     right-group squares descending. Input columns are back-solved to reproduce edge vs inward
#     placement under that order.
#   - Recoloring introduces a new assumption: the unified border color must differ from both
#     the background and every allowed interior color, otherwise a frame could visually vanish.
#     The generator therefore reserves maroon(9) for the output border and black(0) for bg.
# placement_legality_and_observability:
#   - Output is built first, with non-overlapping edge squares and inward squares that overlap
#     an edge square only in row-span, never in occupied cells.
#   - Input positions are then assigned with distinct columns and a global non-overlap check.
#   - Samples are rejected unless the warm/cool solver matches the target, the old frame-based
#     grouping would produce a different output, and the target really has a single border color.
#   - From 2-4 training pairs, a human can see two stable facts: cool interiors consistently
#     aggregate on the left while warm interiors aggregate on the right, and every output frame
#     has the same border color regardless of its input border.


SQ = 5
GRID_H = 22
GRID_W = 22
AZURE = 8
RED = 2
UNIFIED_FRAME = 9
COOL_INNER_COLORS = (1, 3, 5)
WARM_INNER_COLORS = (2, 4, 6, 7)
ALL_GROUPABLE_INNER_COLORS = set(COOL_INNER_COLORS) | set(WARM_INNER_COLORS)
N_COOL_RANGE = (3, 4)
N_WARM_RANGE = (3, 4)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500
BG_CHOICES = (0,)


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (int(a1) < int(b0) or int(b1) < int(a0))


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


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


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _recolor_patch_border(patch: np.ndarray, frame_color: int) -> np.ndarray:
    out = np.array(patch, copy=True)
    out[0, :] = int(frame_color)
    out[-1, :] = int(frame_color)
    out[:, 0] = int(frame_color)
    out[:, -1] = int(frame_color)
    return out


def _detect_squares(
    g: np.ndarray,
    allowed_frames: set[int],
) -> list[dict[str, int | np.ndarray]]:
    H, W = g.shape
    squares: list[dict[str, int | np.ndarray]] = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue

            frame_col = int(border[0])
            if frame_col not in allowed_frames:
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
    output_frame: int,
) -> np.ndarray:
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)
    squares = _detect_squares(g, {AZURE, RED})

    left_squares = sorted(
        [s for s in squares if side_of_square(s) == "left"],
        key=lambda s: int(s["c0"]),
    )
    placed_at_edge: list[tuple[int, int]] = []
    for s in left_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(_overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, _recolor_patch_border(s["patch"], output_frame), r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    right_squares = sorted(
        [s for s in squares if side_of_square(s) == "right"],
        key=lambda s: -int(s["c0"]),
    )
    placed_at_edge = []
    for s in right_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(_overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, _recolor_patch_border(s["patch"], output_frame), r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack scattered 5x5 frames by interior temperature, then normalize every border color.

    Input structure:
    - The background is the most common color.
    - The foreground objects are red- or teal-bordered 5x5 frames with a uniform 3x3 center.
    - The center color, not the border, belongs to a warm set or a cool set.

    Rule:
    - Cool-centered frames move into the left stack; warm-centered frames move into the right stack.
    - Every frame keeps its original row.
    - Left-group frames are processed by input column ascending; right-group frames by input
      column descending.
    - If a frame row-overlaps an already placed edge frame on its side, it shifts one frame-width
      inward.
    - After placement, all frame borders are recolored to the same unified color.

    Key insight: The input still shows two possible border colors, but they are now decoys.
    Side membership comes from the interior's warm/cool category, and the output erases the
    original border labels by repainting every frame edge identically.
    """
    g = np.array(grid, dtype=int)
    return _solve_with_grouping(g, lambda s: _tone_side(int(s["inner"])), UNIFIED_FRAME)


def _frame_based_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    return _solve_with_grouping(g, lambda s: _frame_side(int(s["frame"])), UNIFIED_FRAME)


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
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
        if any(_rows_overlap(r, ir) for ir in inward_rows):
            continue
        inward_rows.append(int(r))
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None
    return edge_rows, inward_rows


def _sample_inner_colors(palette: tuple[int, ...], n: int) -> list[int]:
    colors = list(palette)
    random.shuffle(colors)
    result = colors[: min(len(colors), n)]
    while len(result) < n:
        result.append(int(random.choice(palette)))
    random.shuffle(result)
    return result


def _assign_frames_for_group(inner_colors: list[int], side: str) -> list[int]:
    n = len(inner_colors)
    frames: list[int | None] = [None] * n

    if side == "left":
        mismatch_idx = random.randrange(n)
        frames[mismatch_idx] = RED
        other_idx = random.choice([i for i in range(n) if i != mismatch_idx])
        frames[other_idx] = AZURE
        for i in range(n):
            if frames[i] is None:
                frames[i] = int(random.choice([AZURE, RED]))
    else:
        azure_idx = random.randrange(n)
        frames[azure_idx] = AZURE
        red_candidates = [i for i, inner in enumerate(inner_colors) if inner != RED and i != azure_idx]
        if not red_candidates:
            red_candidates = [i for i, inner in enumerate(inner_colors) if inner != RED]
        red_idx = random.choice(red_candidates)
        frames[red_idx] = RED
        for i, inner in enumerate(inner_colors):
            if frames[i] is not None:
                continue
            valid_frames = [AZURE] if inner == RED else [AZURE, RED]
            frames[i] = int(random.choice(valid_frames))

    final_frames = [int(f) for f in frames]
    assert any(f == AZURE for f in final_frames)
    assert any(f == RED for f in final_frames)
    assert all(inner != frame for inner, frame in zip(inner_colors, final_frames))
    return final_frames


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
        inner = int(inner_colors[i])
        frame = int(frame_colors[i])
        input_patch = _make_patch(frame, inner)
        paste(output, _make_patch(UNIFIED_FRAME, inner), r, edge_col)
        placed.append((int(r), int(edge_col), True, input_patch, frame, inner))
    for i, r in enumerate(inward_rows):
        idx = len(edge_rows) + i
        inner = int(inner_colors[idx])
        frame = int(frame_colors[idx])
        input_patch = _make_patch(frame, inner)
        paste(output, _make_patch(UNIFIED_FRAME, inner), r, inward_col)
        placed.append((int(r), int(inward_col), False, input_patch, frame, inner))
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
        for i, (r, oc, patch) in enumerate(ordered_squares):
            ic = int(chosen_cols[i])
            if ic == int(oc):
                ok = False
                break
            result.append((int(r), ic, patch))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _group_mismatch_count(groups: list[tuple[int, int]]) -> int:
    return sum(1 for frame, inner in groups if _frame_side(frame) != _tone_side(inner))


def _count_unified_output_frames(grid: np.ndarray) -> int:
    squares = _detect_squares(np.array(grid, dtype=int), {UNIFIED_FRAME})
    return len({int(s["frame"]) for s in squares})


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(BG_CHOICES))
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

    cool_inner_colors = _sample_inner_colors(COOL_INNER_COLORS, n_cool)
    warm_inner_colors = _sample_inner_colors(WARM_INNER_COLORS, n_warm)
    cool_frames = _assign_frames_for_group(cool_inner_colors, "left")
    warm_frames = _assign_frames_for_group(warm_inner_colors, "right")

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

    all_specs = [(frame, inner) for (_, _, _, _, frame, inner) in cool_out + warm_out]
    if _group_mismatch_count(all_specs) < 2:
        return None
    if not any(frame == RED for frame, inner in all_specs if _tone_side(inner) == "left"):
        return None
    if not any(frame == AZURE for frame, inner in all_specs if _tone_side(inner) == "right"):
        return None
    if len(set(cool_inner_colors)) < 2 or len(set(warm_inner_colors)) < 2:
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
    for r, c, patch in all_positions:
        paste(inp, patch, r, c)

    try:
        if not np.array_equal(main(inp), output):
            return None
        if np.array_equal(_frame_based_main(inp), output):
            return None
        if _count_unified_output_frames(output) != 1:
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
