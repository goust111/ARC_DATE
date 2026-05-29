# Variant of b5ca7ac4
# Dimension: D15 — 干扰项——输入中的非目标对象 (src)
# Change: 输入中除框架和背景外无其他对象，所有非背景像素都属于某个框架... → 增加分隔线作为视觉干扰

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste

# Design analysis
# ---------------
# variant:
#   added_object:
#     - exactly one full-width or full-height solid distractor line is drawn in the input
#     - its color is neither bg nor frame color
#     - unlike a harmless separator placed on empty space, this line is chosen to cut through
#       at least one square border, and usually several squares, so the original solver's
#       "raw 5x5 patch is already correct" assumption no longer holds
#   input_effect:
#     - a row-line overwrites one row of every square it crosses
#     - a col-line overwrites one column of every square it crosses
#   output_effect:
#     - the line disappears completely
#     - every square must be restored to its canonical 5x5 frame + uniform 3x3 interior
#
# trace_example:
#   original:
#     - a teal frame at rows 4..8, cols 9..13 is scanned as a perfect 5x5 patch
#   variant:
#     - a horizontal distractor at row 4 repaints that frame's top border with some third color
#     - the old border==uniform test fails, so the square would be missed
#     - the new solver first detects row 4 as a full-grid solid line, then while testing each
#       5x5 window it ignores masked cells on that line, verifies the remaining border cells are
#       all teal, verifies the remaining inner cells share one non-teal color, and reconstructs
#       the original 5x5 patch before applying the old stacking rule
#
# hidden_assumption_audit:
#   - old code assumed every relevant square can be read verbatim from the input patch
#     -> broken once a divider overwrites border or interior cells
#   - old code assumed every non-bg foreground pixel either belongs to a frame or can be ignored
#     -> technically still true for output reconstruction, but insufficient for detection because
#        the distractor can corrupt true squares
#   - old code used strict border uniformity on the raw patch
#     -> replaced with masked uniformity after removing global divider rows/cols from evidence
#
# placement_legality:
#   - generator still builds the intended output first
#   - then it derives non-overlapping input square positions exactly as before
#   - only after positions are fixed does it choose a distractor row/col that intersects at
#     least one square border; this guarantees the variant is behaviorally active
#   - the line spans the whole grid, so it is visually obvious and solver-detectable via the
#     "solid non-bg full row/col" signature
#
# observability:
#   - humans can see the same left/right stacking rule as the original task
#   - the extra line is globally straight, independent of any one square, and disappears in the
#     output, so its role as a distractor is inferable from a few examples


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500


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


def _find_distractor_lines(g: np.ndarray, bg: int) -> tuple[set[int], set[int]]:
    h, w = g.shape
    row_lines = {
        r for r in range(h)
        if len(set(map(int, g[r, :]))) == 1 and int(g[r, 0]) != bg
    }
    col_lines = {
        c for c in range(w)
        if len(set(map(int, g[:, c]))) == 1 and int(g[0, c]) != bg
    }
    return row_lines, col_lines


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
    Rebuild 5x5 framed squares after ignoring a full-grid distractor line, then stack them by frame color.

    Input structure:
    - bg = most common color.
    - Azure (8) and red (2) framed 5x5 squares are scattered as before.
    - One or more solid full-width/full-height non-bg divider lines may overwrite parts of those squares.

    Rule:
    - First identify any solid non-bg row/column that spans the whole grid and treat it as a distractor.
    - Recover each 5x5 square by ignoring divider-covered cells; the surviving border cells fix the frame color,
      and the surviving inner cells fix the uniform 3x3 fill.
    - Move azure squares to the left edge and red squares to the right edge, preserving the original row and the
      original nearest-edge-first ordering; row-overlapping later squares shift one square inward.

    Key insight: The global straight divider is not an object to move; it is a mask to discard so each damaged
    square can be reconstructed before applying the original edge-stacking logic.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    row_lines, col_lines = _find_distractor_lines(g, bg)
    squares = _detect_squares(g, row_lines, col_lines)

    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    azure_squares = sorted(
        [s for s in squares if int(s["frame"]) == AZURE],
        key=lambda s: int(s["c0"]),
    )
    placed_at_default: list[tuple[int, int]] = []
    for s in azure_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], r0, int(dest_c), transparent=bg)
        if dest_c == 0:
            placed_at_default.append((r0, r1))

    red_squares = sorted(
        [s for s in squares if int(s["frame"]) == RED],
        key=lambda s: -int(s["c0"]),
    )
    placed_at_default = []
    for s in red_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = int(w - 2 * SQ) if has_overlap else int(w - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(w - SQ):
            placed_at_default.append((r0, r1))

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
    frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        p = _make_patch(frame_color, colors[i])
        paste(output, p, int(r), int(edge_col))
        placed.append((int(r), int(edge_col), True, p))
    for i, r in enumerate(inward_rows):
        p = _make_patch(frame_color, colors[len(edge_rows) + i])
        paste(output, p, int(r), int(inward_col))
        placed.append((int(r), int(inward_col), False, p))
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
        for i, (r, oc, p) in enumerate(ordered_squares):
            ic = int(chosen_cols[i])
            if ic == int(oc):
                ok = False
                break
            result.append((int(r), ic, p))
        if not ok:
            continue
        if not _any_pairwise_overlap(result):
            return result
    return None


def _line_stats_for_row(square_positions: list[tuple[int, int, np.ndarray]], row: int) -> tuple[int, int]:
    hits = 0
    border_hits = 0
    for r, _c, _p in square_positions:
        if int(r) <= row < int(r) + SQ:
            hits += 1
            if row in {int(r), int(r) + SQ - 1}:
                border_hits += 1
    return hits, border_hits


def _line_stats_for_col(square_positions: list[tuple[int, int, np.ndarray]], col: int) -> tuple[int, int]:
    hits = 0
    border_hits = 0
    for _r, c, _p in square_positions:
        if int(c) <= col < int(c) + SQ:
            hits += 1
            if col in {int(c), int(c) + SQ - 1}:
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
    filtered = [
        cand for cand in candidates
        if cand[0] >= target_hits
    ]
    best_border_hits = max(border_hits for _hits, border_hits, _axis, _idx in filtered)
    filtered = [
        cand for cand in filtered
        if cand[1] == best_border_hits
    ]
    _hits, _border_hits, axis, idx = random.choice(filtered)
    return axis, idx


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
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

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
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
    distractor_color = random.choice([c for c in range(10) if c not in {bg, AZURE, RED}])

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, p in all_positions:
        paste(inp, p, int(r), int(c))

    if axis == "row":
        if not any(_line_stats_for_row(all_positions, idx)):
            return None
        inp[int(idx), :] = distractor_color
    else:
        if not any(_line_stats_for_col(all_positions, idx)):
            return None
        inp[:, int(idx)] = distractor_color

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
