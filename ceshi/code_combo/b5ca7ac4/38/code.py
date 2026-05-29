# Combo variant of b5ca7ac4 (index 38)
# combo_spec: D15/2+D16/0
# Name: 分隔线干扰+输出背景变色

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste

# Design analysis
# ---------------
# fusion_summary:
#   - Keep the original task's object vocabulary: scattered 5x5 red/teal framed squares.
#   - Add exactly one full-grid solid distractor line in the input. It overwrites one row or one
#     column across the whole canvas, including any squares it crosses.
#   - Couple the two dimensions instead of treating them independently: the distractor line's
#     color is also the output background color. The old background only exists in the input.
#
# behavior_trace:
#   - Original task:
#     - read each intact 5x5 frame directly from input
#     - move teal frames left and red frames right
#     - keep the same background color in output
#   - This combo:
#     - the line may repaint one border row/col of a true frame, so raw 5x5 scanning is no longer
#       reliable
#     - detect the full-width/full-height solid non-bg line first, treat its cells as masked-out
#       evidence, and reconstruct each square from the surviving border/interior cells
#     - after reconstruction, apply the same left/right stacking rule
#     - fill the output with the line color, not the input background color, so the line disappears
#       as a separate object and instead becomes the new empty-space color
#   - Concrete example:
#     - suppose input bg is 0, line color is 6, and a teal square at rows 4..8 gets its top border
#       overwritten by a row-line at r=4
#     - old code rejects that 5x5 window because the border is no longer uniformly teal
#     - new code ignores row 4 while testing that window, recovers a canonical teal frame with its
#       uniform interior, then pastes it into an output whose background is 6 everywhere else
#
# complexity_calibration:
#   - The background change is not an arbitrary color mapping. It is anchored to the unique global
#     line color, which is visible in every input and reused everywhere empty in the output.
#   - Generator reserves the line color so it does not appear inside any square. This prevents
#     ambiguous cases where an interior would vanish into the new background.
#   - Only one distractor line is used per sample. That keeps the cue strong and the target output
#     unambiguous while still requiring masked reconstruction.
#
# hidden_assumption_audit:
#   - Original `get_bg` assumption still holds because a single 22-cell line cannot outnumber the
#     remaining background cells on a 22x22 board.
#   - Original "every non-bg pixel belongs to a frame" assumption is false now; the solver must
#     explicitly detect and remove the global line.
#   - Original "detected patch can be pasted verbatim" assumption is weakened because some cells may
#     semantically mean "background" under D16. Pasting onto an output pre-filled with the new
#     background and using transparent=`input_bg` preserves that meaning.
#   - Sorting by input column remains valid because the distractor changes colors only, never square
#     top-left coordinates.
#
# placement_legality:
#   - Output is still generated first using the original edge/inward row constraints.
#   - Input positions are derived exactly as before, ensuring the solver's nearest-edge-first order
#     reproduces the chosen edge/inward stacking.
#   - The distractor line is chosen only from rows/cols that hit at least one square border, so the
#     added object is behaviorally active rather than decorative.
#
# observability_and_tidy_layout:
#   - Humans can still see the original left-vs-right grouping by frame color.
#   - The distractor is globally straight and spans the entire grid, so it reads as a separator-like
#     nuisance rather than a candidate square.
#   - Because the output background becomes the distractor color everywhere, examples provide a
#     repeated structural clue for the color change rule instead of an arbitrary palette remap.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), frame_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
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


def _find_distractor_lines(g: np.ndarray, bg: int) -> tuple[set[int], set[int], int]:
    h, w = g.shape
    row_lines = {
        r
        for r in range(h)
        if len(set(map(int, g[r, :]))) == 1 and int(g[r, 0]) != bg
    }
    col_lines = {
        c
        for c in range(w)
        if len(set(map(int, g[:, c]))) == 1 and int(g[0, c]) != bg
    }
    line_colors = {int(g[r, 0]) for r in row_lines} | {int(g[0, c]) for c in col_lines}
    line_color = int(next(iter(line_colors))) if line_colors else int(bg)
    return row_lines, col_lines, line_color


def _recover_square_patch(
    g: np.ndarray,
    r0: int,
    c0: int,
    row_lines: set[int],
    col_lines: set[int],
) -> np.ndarray | None:
    border_colors: list[int] = []
    inner_colors: list[int] = []

    for dr in range(SQ):
        for dc in range(SQ):
            rr = int(r0 + dr)
            cc = int(c0 + dc)
            if rr in row_lines or cc in col_lines:
                continue
            color = int(g[rr, cc])
            if dr in {0, SQ - 1} or dc in {0, SQ - 1}:
                border_colors.append(color)
            else:
                inner_colors.append(color)

    if not border_colors or not inner_colors:
        return None
    if len(set(border_colors)) != 1:
        return None

    frame_color = int(border_colors[0])
    if frame_color not in {AZURE, RED}:
        return None
    if len(set(inner_colors)) != 1:
        return None

    inner_color = int(inner_colors[0])
    if inner_color == frame_color:
        return None
    return _make_patch(frame_color, inner_color)


def _detect_squares(
    g: np.ndarray,
    row_lines: set[int],
    col_lines: set[int],
) -> list[dict[str, object]]:
    h, w = g.shape
    squares: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for r0 in range(h - SQ + 1):
        for c0 in range(w - SQ + 1):
            patch = _recover_square_patch(g, int(r0), int(c0), row_lines, col_lines)
            if patch is None:
                continue
            key = (int(r0), int(c0))
            if key in seen:
                continue
            seen.add(key)
            squares.append(
                {
                    "frame": int(patch[0, 0]),
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch,
                }
            )
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover line-damaged 5x5 frames, then restack them while changing the output background.

    Input structure:
    - The most common color is the input background.
    - Teal (8) and red (2) 5x5 framed squares with uniform 3x3 interiors are scattered on the grid.
    - One solid full-width or full-height non-background line may cut through those squares as a
      distractor; its color is also the output background color.

    Rule:
    - Detect any solid non-bg row/column spanning the entire grid and ignore those cells when
      identifying squares.
    - Reconstruct each square from the remaining visible border and interior cells.
    - Move teal squares to the left edge and red squares to the right edge, preserving each square's
      original row and using original input-column order to decide which overlapping square stays on
      the edge versus shifting one square inward.
    - Fill all empty output cells with the distractor line color instead of the input background.

    Key insight: The full-grid line serves two roles at once: it is noise that must be masked out to
    recover the true squares, and it reveals the new background color that replaces the old one in
    the output.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    input_bg = int(get_bg(g))

    row_lines, col_lines, line_color = _find_distractor_lines(g, input_bg)
    output_bg = int(line_color) if (row_lines or col_lines) else int(input_bg)
    squares = _detect_squares(g, row_lines, col_lines)

    out = np.full((h, w), output_bg, dtype=int)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    azure_squares = sorted(
        [s for s in squares if int(s["frame"]) == AZURE],
        key=lambda s: int(s["c0"]),
    )
    placed_at_edge: list[tuple[int, int]] = []
    for s in azure_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], r0, int(dest_c), transparent=input_bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    red_squares = sorted(
        [s for s in squares if int(s["frame"]) == RED],
        key=lambda s: -int(s["c0"]),
    )
    placed_at_edge = []
    for s in red_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(w - 2 * SQ) if has_overlap else int(w - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=input_bg)
        if dest_c == int(w - SQ):
            placed_at_edge.append((r0, r1))

    return out


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


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    colors: list[int],
    frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        patch = _make_patch(frame_color, colors[i])
        paste(output, patch, int(r), int(edge_col))
        placed.append((int(r), int(edge_col), True, patch))
    for i, r in enumerate(inward_rows):
        patch = _make_patch(frame_color, colors[len(edge_rows) + i])
        paste(output, patch, int(r), int(inward_col))
        placed.append((int(r), int(inward_col), False, patch))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]

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

        ordered_squares = [
            (edges[idx] if typ == "edge" else inwards[idx])
            for typ, idx in order
        ]

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
        if not ok:
            continue
        if not _any_pairwise_overlap(result):
            return result
    return None


def _line_stats_for_row(
    square_positions: list[tuple[int, int, np.ndarray]],
    row: int,
) -> tuple[int, int]:
    hits = 0
    border_hits = 0
    for r, _c, _patch in square_positions:
        if int(r) <= int(row) < int(r) + SQ:
            hits += 1
            if int(row) in {int(r), int(r) + SQ - 1}:
                border_hits += 1
    return hits, border_hits


def _line_stats_for_col(
    square_positions: list[tuple[int, int, np.ndarray]],
    col: int,
) -> tuple[int, int]:
    hits = 0
    border_hits = 0
    for _r, c, _patch in square_positions:
        if int(c) <= int(col) < int(c) + SQ:
            hits += 1
            if int(col) in {int(c), int(c) + SQ - 1}:
                border_hits += 1
    return hits, border_hits


def _choose_distractor_line(
    square_positions: list[tuple[int, int, np.ndarray]],
) -> tuple[str, int]:
    candidates: list[tuple[int, int, str, int]] = []

    for row in range(GRID_H):
        hits, border_hits = _line_stats_for_row(square_positions, int(row))
        if border_hits > 0:
            candidates.append((hits, border_hits, "row", int(row)))
    for col in range(GRID_W):
        hits, border_hits = _line_stats_for_col(square_positions, int(col))
        if border_hits > 0:
            candidates.append((hits, border_hits, "col", int(col)))

    if not candidates:
        raise RuntimeError("No valid distractor line candidate")

    best_hits = max(hits for hits, _border_hits, _axis, _idx in candidates)
    target_hits = 2 if best_hits >= 2 else 1
    filtered = [cand for cand in candidates if cand[0] >= target_hits]
    best_border_hits = max(border_hits for _hits, border_hits, _axis, _idx in filtered)
    filtered = [cand for cand in filtered if cand[1] == best_border_hits]
    _hits, _border_hits, axis, idx = random.choice(filtered)
    return axis, idx


def _gen_attempt() -> np.ndarray | None:
    input_bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    line_color = random.choice([c for c in range(10) if c not in {input_bg, AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_assign = _assign_output_rows(n_azure)
    if azure_assign is None:
        return None
    red_assign = _assign_output_rows(n_red)
    if red_assign is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_assign
    red_edge_rows, red_inward_rows = red_assign

    avail_inner = [c for c in range(10) if c not in {input_bg, line_color, AZURE, RED}]
    if len(avail_inner) < max(n_azure, n_red):
        return None
    azure_colors = random.sample(avail_inner, n_azure)
    red_colors = random.sample(avail_inner, n_red)

    output = np.full((GRID_H, GRID_W), line_color, dtype=int)
    azure_out = _place_side_squares(
        output,
        azure_edge_rows,
        azure_inward_rows,
        azure_colors,
        AZURE,
        edge_col=0,
        inward_col=SQ,
    )
    red_out = _place_side_squares(
        output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )

    azure_input = _derive_input_positions(azure_out, ascending=True)
    if azure_input is None:
        return None
    red_input = _derive_input_positions(red_out, ascending=False)
    if red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    axis, idx = _choose_distractor_line(all_positions)

    inp = np.full((GRID_H, GRID_W), input_bg, dtype=int)
    for r, c, patch in all_positions:
        paste(inp, patch, int(r), int(c))

    if axis == "row":
        if _line_stats_for_row(all_positions, int(idx))[1] <= 0:
            return None
        inp[int(idx), :] = int(line_color)
    else:
        if _line_stats_for_col(all_positions, int(idx))[1] <= 0:
            return None
        inp[:, int(idx)] = int(line_color)

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
