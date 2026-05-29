# Combo variant of b5ca7ac4 (index 29)
# combo_spec: D1/0+D2/2+D8/0
# Name: 冷暖分组+颜色排序+边框统一

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the combo variant
# -------------------------------------
# fused_rule:
#   - Objects stay as scattered 5x5 bordered squares with a uniform 3x3 fill.
#   - D1 changes grouping: side is decided by the interior tone, not by the border.
#     cool fills {1 blue, 3 green, 5 gray} -> left stack
#     warm fills {2 red, 4 yellow, 6 magenta, 7 orange} -> right stack
#   - D2 changes ordering: inside each tone-group, squares are processed by inner color id
#     ascending, not by any spatial coordinate. A concrete trace:
#       * red frame + blue fill now goes left because blue is cool.
#       * teal frame + yellow fill now goes right because yellow is warm.
#       * if left-side blue(1) and gray(5) squares overlap in rows, blue is handled first
#         and keeps the edge slot; gray is pushed inward even if gray starts farther left.
#   - D8 changes appearance after motion: once a square is placed, its border is rewritten
#     to one shared color (maroon 9). The output no longer preserves the input frame colors.
#
# complexity_calibration:
#   - The generator keeps the tone palettes small and reusable, so the warm/cool split and the
#     per-group color order are learnable from a few examples instead of requiring many colors.
#   - Cool side always uses the full ordered trio 1<3<5; warm side uses 3-4 distinct colors
#     from 2<4<6<7. This gives repeated pairwise ordering evidence.
#   - Each side contains at least one inward square, so every sample includes row-overlap cases
#     where processing order visibly matters.
#
# hidden_assumption_audit:
#   - The 5x5 sliding-window detector still assumes every real object is exactly a one-cell frame
#     around a uniform 3x3 fill. The generator preserves that and rejects ambiguous interiors.
#   - Grouping now depends on the interior being one of the declared warm/cool colors, so the
#     generator excludes black(0) and maroon(9) from interiors.
#   - Ordering no longer depends on input columns. Therefore the generator is rewritten as a
#     color-order-first construction: output rows are assigned in color order, and input columns
#     are only required to avoid overlap while also making the old spatial-order solver disagree.
#   - Border unification introduces a new assumption that output border color must stay distinct
#     from background and interior fills. The generator fixes bg=0 and output border=9.
#
# placement_legality_and_cleanliness:
#   - Edge squares on the same side never row-overlap.
#   - Inward squares must row-overlap an earlier edge square and never row-overlap another inward
#     square on the same side, so each side still uses only two visible columns.
#   - Input squares are sampled with a global non-overlap check.
#   - Validation compares the fused solver against three ablations that isolate D1, D2, and D8;
#     samples are rejected unless all three altered mechanisms actually matter on that input.


SQ = 5
GRID_H = 22
GRID_W = 22
AZURE = 8
RED = 2
UNIFIED_FRAME = 9
BG_COLOR = 0
COOL_INNER_COLORS = (1, 3, 5)
WARM_INNER_COLORS = (2, 4, 6, 7)
ALL_GROUPABLE_INNER_COLORS = set(COOL_INNER_COLORS) | set(WARM_INNER_COLORS)
MAX_INWARD = 2
INPUT_COL_RETRIES = 120
ROW_PLAN_RETRIES = 120
GEN_RETRIES = 500
N_COOL_RANGE = (3, 3)
N_WARM_RANGE = (3, 4)


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


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


def _detect_squares(g: np.ndarray) -> list[dict[str, int]]:
    H, W = g.shape
    squares: list[dict[str, int]] = []
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
                }
            )
    return squares


def _solve_with_controls(
    grid: np.ndarray,
    side_of_square,
    left_sort_key,
    right_sort_key,
    output_frame: int | None,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def render_square(square: dict[str, int]) -> np.ndarray:
        frame = int(square["frame"]) if output_frame is None else int(output_frame)
        return _make_patch(frame, int(square["inner"]))

    def place_side(squares: list[dict[str, int]], edge_col: int, inward_col: int) -> None:
        placed_edge_rows: list[tuple[int, int]] = []
        for sq in squares:
            r0 = int(sq["r0"])
            r1 = int(r0 + SQ - 1)
            has_overlap = any(
                not (r1 < pr0 or pr1 < r0)
                for pr0, pr1 in placed_edge_rows
            )
            dest_c = int(inward_col) if has_overlap else int(edge_col)
            paste(out, render_square(sq), r0, dest_c, transparent=bg)
            if dest_c == int(edge_col):
                placed_edge_rows.append((r0, r1))

    squares = _detect_squares(g)
    left_squares = sorted(
        [sq for sq in squares if side_of_square(sq) == "left"],
        key=left_sort_key,
    )
    right_squares = sorted(
        [sq for sq in squares if side_of_square(sq) == "right"],
        key=right_sort_key,
    )

    place_side(left_squares, edge_col=0, inward_col=SQ)
    place_side(right_squares, edge_col=W - SQ, inward_col=W - 2 * SQ)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack bordered 5x5 squares by interior tone, break ties by interior color order, and
    erase the original border coding in the output.

    Input structure:
    - Background is the dominant color.
    - The foreground objects are red(2)-framed and teal(8)-framed 5x5 squares with uniform 3x3
      interiors chosen from a cool palette {1,3,5} or a warm palette {2,4,6,7}.
    - Border color is still visible in the input, but it is no longer the semantic key.

    Rule:
    - Cool interiors move to the left stack; warm interiors move to the right stack.
    - Within each side, process squares by their interior color value ascending.
    - Each square keeps its original row.
    - A square that row-overlaps any already placed edge square on its side is shifted one square
      width inward instead of taking the edge column.
    - After moving, every square border is recolored to the same maroon value, so output borders
      no longer indicate the original group or frame color.

    Key insight: The output depends on a chained color interpretation: interior tone chooses the
    side, interior color order chooses which overlapping square wins the edge slot, and the final
    recolor removes the input border cue so the grouping evidence must be read from the examples.
    """
    return _solve_with_controls(
        grid,
        side_of_square=lambda sq: _tone_side(int(sq["inner"])),
        left_sort_key=lambda sq: (int(sq["inner"]), int(sq["r0"]), int(sq["c0"])),
        right_sort_key=lambda sq: (int(sq["inner"]), int(sq["r0"]), int(sq["c0"])),
        output_frame=UNIFIED_FRAME,
    )


def _frame_group_color_order_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_controls(
        grid,
        side_of_square=lambda sq: _frame_side(int(sq["frame"])),
        left_sort_key=lambda sq: (int(sq["inner"]), int(sq["r0"]), int(sq["c0"])),
        right_sort_key=lambda sq: (int(sq["inner"]), int(sq["r0"]), int(sq["c0"])),
        output_frame=UNIFIED_FRAME,
    )


def _tone_group_column_order_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_controls(
        grid,
        side_of_square=lambda sq: _tone_side(int(sq["inner"])),
        left_sort_key=lambda sq: (int(sq["c0"]), int(sq["r0"]), int(sq["inner"])),
        right_sort_key=lambda sq: (-int(sq["c0"]), int(sq["r0"]), int(sq["inner"])),
        output_frame=UNIFIED_FRAME,
    )


def _tone_group_color_order_preserve_frame_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_controls(
        grid,
        side_of_square=lambda sq: _tone_side(int(sq["inner"])),
        left_sort_key=lambda sq: (int(sq["inner"]), int(sq["r0"]), int(sq["c0"])),
        right_sort_key=lambda sq: (int(sq["inner"]), int(sq["r0"]), int(sq["c0"])),
        output_frame=None,
    )


def _assign_side_specs(
    inner_colors: list[int],
    side: str,
) -> list[dict[str, int | str | bool]] | None:
    edge_col = 0 if side == "left" else GRID_W - SQ
    inward_col = SQ if side == "left" else GRID_W - 2 * SQ
    colors = sorted(int(c) for c in inner_colors)
    n = len(colors)
    max_inward = min(MAX_INWARD, n - 1)
    if max_inward < 1:
        return None

    for _ in range(ROW_PLAN_RETRIES):
        n_inward = random.randint(1, max_inward)
        inward_indices = set(random.sample(range(1, n), n_inward))
        edge_rows: list[int] = []
        inward_rows: list[int] = []
        specs: list[dict[str, int | str | bool]] = []
        ok = True
        for idx, inner in enumerate(colors):
            all_rows = list(range(GRID_H - SQ + 1))
            random.shuffle(all_rows)
            if idx in inward_indices:
                candidates = [
                    r
                    for r in all_rows
                    if any(_rows_overlap(r, er) for er in edge_rows)
                    and all(not _rows_overlap(r, ir) for ir in inward_rows)
                ]
                if not candidates:
                    ok = False
                    break
                row = int(candidates[0])
                inward_rows.append(row)
                specs.append(
                    {
                        "side": side,
                        "inner": int(inner),
                        "row": row,
                        "out_col": int(inward_col),
                        "is_edge": False,
                    }
                )
            else:
                candidates = [
                    r
                    for r in all_rows
                    if all(not _rows_overlap(r, er) for er in edge_rows)
                ]
                if not candidates:
                    ok = False
                    break
                row = int(candidates[0])
                edge_rows.append(row)
                specs.append(
                    {
                        "side": side,
                        "inner": int(inner),
                        "row": row,
                        "out_col": int(edge_col),
                        "is_edge": True,
                    }
                )
        if ok:
            return specs
    return None


def _sample_frames_for_side(n: int, target_side: str) -> list[int]:
    frames = [AZURE, RED]
    while len(frames) < n:
        if target_side == "left":
            frames.append(random.choice([AZURE, RED, RED]))
        else:
            frames.append(random.choice([AZURE, AZURE, RED]))
    random.shuffle(frames)
    return [int(f) for f in frames]


def _render_output(specs: list[dict[str, int | str | bool]]) -> np.ndarray:
    out = np.full((GRID_H, GRID_W), BG_COLOR, dtype=int)
    for spec in specs:
        paste(
            out,
            _make_patch(UNIFIED_FRAME, int(spec["inner"])),
            int(spec["row"]),
            int(spec["out_col"]),
            transparent=BG_COLOR,
        )
    return out


def _build_input_grid(specs: list[dict[str, int | str | bool]]) -> np.ndarray:
    inp = np.full((GRID_H, GRID_W), BG_COLOR, dtype=int)
    for spec in specs:
        paste(
            inp,
            _make_patch(int(spec["frame"]), int(spec["inner"])),
            int(spec["row"]),
            int(spec["in_col"]),
            transparent=BG_COLOR,
        )
    return inp


def _assign_input_columns(
    specs: list[dict[str, int | str | bool]],
) -> list[dict[str, int | str | bool]] | None:
    left_specs = sorted(
        [dict(spec) for spec in specs if spec["side"] == "left"],
        key=lambda spec: int(spec["inner"]),
    )
    right_specs = sorted(
        [dict(spec) for spec in specs if spec["side"] == "right"],
        key=lambda spec: int(spec["inner"]),
    )
    if not left_specs or not right_specs:
        return None

    col_range = list(range(0, GRID_W - SQ + 1))
    for _ in range(INPUT_COL_RETRIES):
        if len(col_range) < len(left_specs) or len(col_range) < len(right_specs):
            return None

        left_cols = sorted(random.sample(col_range, len(left_specs)), reverse=True)
        right_cols = sorted(random.sample(col_range, len(right_specs)))
        for spec, col in zip(left_specs, left_cols):
            spec["in_col"] = int(col)
        for spec, col in zip(right_specs, right_cols):
            spec["in_col"] = int(col)

        all_specs = left_specs + right_specs
        if any(int(spec["in_col"]) == int(spec["out_col"]) for spec in all_specs):
            continue

        positions = [(int(spec["row"]), int(spec["in_col"]), spec) for spec in all_specs]
        if _any_pairwise_overlap(positions):
            continue
        return all_specs
    return None


def _mismatch_count(specs: list[dict[str, int | str | bool]]) -> int:
    return sum(
        1
        for spec in specs
        if _frame_side(int(spec["frame"])) != _tone_side(int(spec["inner"]))
    )


def _cross_side_mismatch_count(specs: list[dict[str, int | str | bool]]) -> tuple[int, int]:
    cool_red = sum(
        1
        for spec in specs
        if spec["side"] == "left" and int(spec["frame"]) == RED
    )
    warm_azure = sum(
        1
        for spec in specs
        if spec["side"] == "right" and int(spec["frame"]) == AZURE
    )
    return int(cool_red), int(warm_azure)


def _gen_attempt() -> np.ndarray | None:
    n_cool = random.randint(*N_COOL_RANGE)
    n_warm = random.randint(*N_WARM_RANGE)
    cool_colors = list(COOL_INNER_COLORS[:n_cool])
    warm_colors = random.sample(list(WARM_INNER_COLORS), n_warm)

    cool_specs = _assign_side_specs(cool_colors, side="left")
    warm_specs = _assign_side_specs(warm_colors, side="right")
    if cool_specs is None or warm_specs is None:
        return None

    for spec, frame in zip(cool_specs, _sample_frames_for_side(len(cool_specs), "left")):
        spec["frame"] = int(frame)
    for spec, frame in zip(warm_specs, _sample_frames_for_side(len(warm_specs), "right")):
        spec["frame"] = int(frame)

    all_specs = cool_specs + warm_specs
    if _mismatch_count(all_specs) < 2:
        return None
    cool_red, warm_azure = _cross_side_mismatch_count(all_specs)
    if cool_red < 1 or warm_azure < 1:
        return None

    output = _render_output(all_specs)
    input_specs = _assign_input_columns(all_specs)
    if input_specs is None:
        return None
    inp = _build_input_grid(input_specs)

    try:
        detected = _detect_squares(inp)
        if len(detected) != len(input_specs):
            return None
        if not np.array_equal(main(inp), output):
            return None
        if np.array_equal(_frame_group_color_order_main(inp), output):
            return None
        if np.array_equal(_tone_group_column_order_main(inp), output):
            return None
        if np.array_equal(_tone_group_color_order_preserve_frame_main(inp), output):
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
