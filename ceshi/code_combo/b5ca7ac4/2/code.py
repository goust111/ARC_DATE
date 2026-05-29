# Combo variant of b5ca7ac4 (index 2)
# combo_spec: D1/0+D2/2
# Name: 冷暖色调分组+内部颜色排序处理

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the combo variant
# -------------------------------------
# fused_rule:
#   - Visible objects stay close to the seed task: scattered 5x5 bordered squares with a
#     uniform 3x3 interior. The fusion only changes the semantic control signals.
#   - Grouping is now by the interior tone family, not by the border:
#       cool {1 blue, 3 green, 5 gray} -> left stack
#       warm {2 red, 4 yellow, 6 magenta, 7 orange} -> right stack
#   - Within each tone group, processing order is no longer spatial. Squares are traversed by
#     interior color ascending, which is equivalent here to the visible fixed sequence
#       blue -> green -> gray and red -> yellow -> magenta -> orange.
# behavior_trace:
#   - Example left group: blue(1), green(3), gray(5). Suppose gray row-overlaps blue while
#     green is separate. In the seed task, whichever matching square is nearer the left edge in
#     the input claims the edge. In this combo, blue is always processed before gray because
#     1 < 5, so blue keeps the edge and gray moves inward regardless of input column.
#   - Example right group: red(2), yellow(4), magenta(6), orange(7). If magenta overlaps red,
#     the edge/inward choice depends on the fixed color sequence, not on who starts farther right.
#   - This makes the two edits cooperate: the interior color decides both the side (warm/cool)
#     and the stack priority (ordered color sequence).
# complexity_calibration:
#   - Interiors are restricted to a small, repeated warm/cool palette so the semantic split is
#     learnable from a few examples.
#   - Colors are unique within each tone group for each sample, avoiding tie ambiguity and making
#     the per-group order visually recoverable.
#   - Each side uses exactly one inward square. That is enough for ordering to matter while keeping
#     the output tidy and easy to parse.
# hidden_assumption_audit:
#   - Window scanning still assumes true objects are exactly 5x5 with a one-cell uniform border.
#     The generator preserves this.
#   - The solver assumes the 3x3 interior is uniform and belongs to the supported warm/cool
#     palette. The generator enforces that and excludes ambiguous colors 0 and 9 from interiors.
#   - Sorting by interior color alone is only unambiguous when colors are unique within a tone
#     group. The generator guarantees uniqueness; the solver also adds stable tie-breakers.
#   - The two-column stacking logic still assumes at most one inward depth per side. The generator
#     builds outputs with exactly one inward square per side, so no triple stacking is needed.
# placement_legality:
#   - Output rows are chosen first: edge rows on a side do not overlap each other, and the single
#     inward row overlaps exactly one edge row on that side.
#   - Input columns are then back-solved from disjoint left/right column pools so cross-side input
#     collisions are impossible. We still verify pairwise non-overlap before accepting a sample.
# observability_and_cleanliness:
#   - Both border colors are forced to appear on both output sides, so frame color is visibly
#     unreliable and the warm/cool grouping must be inferred from the interior.
#   - Input columns are deliberately arranged so a tone-only spatial solver would disagree with the
#     combo solver, ensuring the color-order component is behaviorally active rather than cosmetic.
#   - Because each side has just one inward square, the final pattern remains visually clean: every
#     square is still individually legible and no region accumulates more than two horizontal depths.


# --------------- solver ---------------

SQ = 5
AZURE = 8
RED = 2

COOL_INNER_COLORS = (1, 3, 5)
WARM_INNER_COLORS = (2, 4, 6, 7)
ALL_GROUPABLE_INNER_COLORS = set(COOL_INNER_COLORS) | set(WARM_INNER_COLORS)


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (int(a1) < int(b0) or int(b1) < int(a0))


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


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


def _color_order_key(s: dict[str, int | np.ndarray]) -> tuple[int, int, int]:
    return (int(s["inner"]), int(s["c0"]), int(s["r0"]))


def _left_spatial_key(s: dict[str, int | np.ndarray]) -> tuple[int, int, int]:
    return (int(s["c0"]), int(s["inner"]), int(s["r0"]))


def _right_spatial_key(s: dict[str, int | np.ndarray]) -> tuple[int, int, int]:
    return (-int(s["c0"]), int(s["inner"]), int(s["r0"]))


def _solve_with_rules(
    g: np.ndarray,
    side_of_square,
    left_sort_key,
    right_sort_key,
) -> np.ndarray:
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    squares = _detect_squares(g)

    left_squares = sorted(
        [s for s in squares if side_of_square(s) == "left"],
        key=left_sort_key,
    )
    placed_at_edge: list[tuple[int, int]] = []
    for s in left_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(_overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    right_squares = sorted(
        [s for s in squares if side_of_square(s) == "right"],
        key=right_sort_key,
    )
    placed_at_edge = []
    for s in right_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(_overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack bordered 5x5 squares by interior color semantics instead of border identity.

    Input structure:
    - bg is the most common color.
    - The grid contains scattered red(2) or teal(8) framed 5x5 squares.
    - Each square has a uniform 3x3 fill drawn from a cool palette (blue/green/gray) or a warm
      palette (red/yellow/magenta/orange).

    Rule:
    - Cool interiors form the left-moving group; warm interiors form the right-moving group.
    - Every square keeps its original row, and the frame color is preserved but ignored for side
      selection.
    - Within each tone group, process squares by interior color ascending rather than by location.
    - If a square row-overlaps any already placed edge square on its side, place it one square
      inward instead of on the edge column.

    Key insight:
    - The same interior attribute drives both decisions: warm/cool determines which side a square
      belongs to, and the fixed order among those interior colors determines who gets edge priority
      when overlapping rows compete on the same side.
    """
    g = np.array(grid, dtype=int)
    return _solve_with_rules(g, lambda s: _tone_side(int(s["inner"])), _color_order_key, _color_order_key)


def _frame_based_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    return _solve_with_rules(g, lambda s: _frame_side(int(s["frame"])), _left_spatial_key, _right_spatial_key)


def _tone_spatial_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    return _solve_with_rules(g, lambda s: _tone_side(int(s["inner"])), _left_spatial_key, _right_spatial_key)


# --------------- generator ---------------

GRID_H, GRID_W = 22, 22
BG_CHOICES = (0, 9)
GEN_RETRIES = 500

COOL_COLORS = COOL_INNER_COLORS
WARM_COLORS = WARM_INNER_COLORS

# Disjoint input-column pools keep the two tone groups separated in the input as well.
LEFT_INPUT_COLS = (1, 2, 3, 4, 6)
RIGHT_INPUT_COLS = (11, 13, 14, 15, 16)


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


def _assign_output_rows(n: int) -> tuple[list[int], int, int] | None:
    all_rows = list(range(GRID_H - SQ + 1))
    for _ in range(80):
        random.shuffle(all_rows)
        edge_rows: list[int] = []
        for r in all_rows:
            if all(not _rows_overlap(r, edge_r) for edge_r in edge_rows):
                edge_rows.append(int(r))
                if len(edge_rows) == n - 1:
                    break
        if len(edge_rows) != n - 1:
            continue

        random.shuffle(all_rows)
        for inward_row in all_rows:
            overlap_indices = [i for i, edge_r in enumerate(edge_rows) if _rows_overlap(inward_row, edge_r)]
            if len(overlap_indices) == 1:
                return edge_rows, int(inward_row), int(overlap_indices[0])
    return None


def _sample_frames(inner_colors: tuple[int, ...]) -> list[int] | None:
    allowed_per_square: list[list[int]] = []
    for inner in inner_colors:
        allowed = [AZURE, RED]
        if int(inner) == RED:
            allowed = [AZURE]
        allowed_per_square.append(allowed)

    for _ in range(80):
        frames = [random.choice(allowed) for allowed in allowed_per_square]
        if AZURE in frames and RED in frames:
            return [int(frame) for frame in frames]
    return None


def _place_group_output(
    output: np.ndarray,
    inner_colors: tuple[int, ...],
    frame_colors: list[int],
    edge_rows: list[int],
    inward_row: int,
    dep_edge_slot: int,
    edge_col: int,
    inward_col: int,
) -> list[dict[str, int | np.ndarray]] | None:
    ordered_specs = sorted(
        [(int(inner), int(frame)) for inner, frame in zip(inner_colors, frame_colors)],
        key=lambda pair: pair[0],
    )
    inward_idx = random.randint(1, len(ordered_specs) - 1)
    inward_inner, inward_frame = ordered_specs[inward_idx]
    edge_specs = ordered_specs[:inward_idx] + ordered_specs[inward_idx + 1 :]
    lower_edge_specs = [spec for spec in edge_specs if spec[0] < inward_inner]
    if not lower_edge_specs:
        return None

    dep_spec = random.choice(lower_edge_specs)
    remaining_specs = [spec for spec in edge_specs if spec != dep_spec]
    random.shuffle(remaining_specs)

    row_specs: list[tuple[int, int] | None] = [None] * len(edge_rows)
    row_specs[dep_edge_slot] = dep_spec
    for row_idx in range(len(edge_rows)):
        if row_specs[row_idx] is None:
            row_specs[row_idx] = remaining_specs.pop()

    placed: list[dict[str, int | np.ndarray]] = []
    for row, spec in zip(edge_rows, row_specs):
        inner, frame = spec
        patch = _make_patch(frame, inner)
        paste(output, patch, int(row), int(edge_col))
        placed.append(
            {
                "row": int(row),
                "out_col": int(edge_col),
                "is_edge": 1,
                "frame": int(frame),
                "inner": int(inner),
                "patch": patch,
            }
        )

    inward_patch = _make_patch(inward_frame, inward_inner)
    paste(output, inward_patch, int(inward_row), int(inward_col))
    placed.append(
        {
            "row": int(inward_row),
            "out_col": int(inward_col),
            "is_edge": 0,
            "frame": int(inward_frame),
            "inner": int(inward_inner),
            "patch": inward_patch,
        }
    )
    return placed


def _derive_input_positions(
    entries: list[dict[str, int | np.ndarray]],
    ascending: bool,
    candidate_cols: tuple[int, ...],
) -> list[tuple[int, int, np.ndarray]] | None:
    inward_idx = next(i for i, entry in enumerate(entries) if int(entry["is_edge"]) == 0)
    inward_inner = int(entries[inward_idx]["inner"])
    dep_candidates = [
        i
        for i, entry in enumerate(entries)
        if int(entry["is_edge"]) == 1
        and int(entry["inner"]) < inward_inner
        and _rows_overlap(int(entry["row"]), int(entries[inward_idx]["row"]))
    ]
    if not dep_candidates:
        return None
    dep_idx = dep_candidates[0]

    middle = [i for i in range(len(entries)) if i not in {inward_idx, dep_idx}]
    random.shuffle(middle)
    spatial_order = [inward_idx, *middle, dep_idx]

    col_combos = list(itertools.combinations(candidate_cols, len(entries)))
    random.shuffle(col_combos)
    for combo in col_combos:
        ordered_cols = sorted(combo, reverse=not ascending)
        assigned_cols = {idx: int(ordered_cols[pos]) for pos, idx in enumerate(spatial_order)}
        result = [
            (int(entry["row"]), int(assigned_cols[i]), entry["patch"])
            for i, entry in enumerate(entries)
        ]
        if not _any_pairwise_overlap(result):
            return result
    return None


def _count_frame_tone_mismatches(entries: list[dict[str, int | np.ndarray]]) -> int:
    return sum(
        1
        for entry in entries
        if _frame_side(int(entry["frame"])) != _tone_side(int(entry["inner"]))
    )


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(BG_CHOICES))

    cool_frames = _sample_frames(COOL_COLORS)
    warm_frames = _sample_frames(WARM_COLORS)
    if cool_frames is None or warm_frames is None:
        return None

    cool_rows = _assign_output_rows(len(COOL_COLORS))
    warm_rows = _assign_output_rows(len(WARM_COLORS))
    if cool_rows is None or warm_rows is None:
        return None

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    cool_out = _place_group_output(output, COOL_COLORS, cool_frames, *cool_rows, edge_col=0, inward_col=SQ)
    warm_out = _place_group_output(
        output,
        WARM_COLORS,
        warm_frames,
        *warm_rows,
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )
    if cool_out is None or warm_out is None:
        return None

    all_entries = cool_out + warm_out
    if _count_frame_tone_mismatches(all_entries) < 2:
        return None

    cool_input = _derive_input_positions(cool_out, ascending=True, candidate_cols=LEFT_INPUT_COLS)
    warm_input = _derive_input_positions(warm_out, ascending=False, candidate_cols=RIGHT_INPUT_COLS)
    if cool_input is None or warm_input is None:
        return None

    all_positions = cool_input + warm_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for row, col, patch in all_positions:
        paste(inp, patch, int(row), int(col))

    try:
        combo_out = main(inp)
        if not np.array_equal(combo_out, output):
            return None
        if np.array_equal(_frame_based_main(inp), output):
            return None
        if np.array_equal(_tone_spatial_main(inp), output):
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
