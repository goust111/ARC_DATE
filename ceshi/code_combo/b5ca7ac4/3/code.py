# Combo variant of b5ca7ac4 (index 3)
# combo_spec: D2/1+D5/1
# Name: 按列排序+垂直堆叠

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

for parent in Path(__file__).resolve().parents:
    candidate = parent / "solvers" / "arc_utils.py"
    if candidate.exists():
        sys.path.insert(0, str(parent / "solvers"))
        break

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   objects:
#     - The visible objects stay the same as the seed task: scattered 5x5 framed squares with border
#       color 8 or 2 and a solid 3x3 interior.
#     - Only the relocation logic changes: squares now keep their input column and move vertically.
#   traced_example:
#     - Suppose teal squares are at input columns 2, 6, 4 and red squares are at columns 16, 11, 14.
#     - In the fused rule, teal are processed by column left->right: 2, 4, 6. The square at col 2 reaches
#       the top edge row 0. If the square at col 4 overlaps that edge square in its 5-column span, it is
#       pushed to row 5 instead; otherwise it also lands at row 0. Red mirrors this by column right->left,
#       landing at row H-5 or, when column-overlapping an earlier edge square, at row H-10.
#     - Compared with the seed's horizontal motion, both the movement axis and the overlap axis rotate:
#       keep-column / move-row replaces keep-row / move-column.
# complexity_calibration:
#   - The task still uses 3-4 teal and 3-5 red squares on a 22x22 board with only two stack depths.
#   - Each color group is forced to have at least one inward square so the second vertical band is visible.
#   - Ordering uses only visible geometry (input columns), not color ids or hidden metadata.
# implicit_assumption_audit:
#   - Scanning all 5x5 windows is still valid because object shape, colors, and size are unchanged.
#   - The seed solver stored row spans of already-placed edge squares; that assumption breaks after the
#     vertical flip, so the fused solver stores column spans instead.
#   - Destination bands change from columns {0, 5, W-10, W-5} to rows {0, 5, H-10, H-5}.
#   - Because ordering and stacking axes are now orthogonal, tie handling on equal input columns would
#     become an unmotivated hidden rule; the generator therefore avoids same-column squares within a color.
# placement_legality:
#   - Output uses four non-overlapping row bands: teal at rows 0/5, red at rows H-10/H-5.
#   - Within a row band, columns are chosen so same-band squares do not overlap; inward squares must
#     column-overlap at least one edge square so the second band is logically justified.
#   - Input rows are sampled separately and validated against rectangle overlap, so scattered inputs stay legal.
# observability_and_cleanliness:
#   - The vertical relocation is easy to see because every square preserves x-position and snaps to the
#     top or bottom region.
#   - The column-sorting component is also made observable: generated inputs are rejected if a row-ordered
#     vertical solver would produce the same output, so examples really require column order.
# generator_guarantees_for_main:
#   - Any square labeled inward in the synthesized output is placed at a column that overlaps an earlier
#     edge square under the fused column order.
#   - Input rows are validated so `main(input)` matches the intended output and a row-ordered vertical
#     alternative disagrees, preventing silent fallback to the simpler D5-only behavior.


# --------------- solver ---------------

GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
COL_ASSIGN_RETRIES = 120
INPUT_ROW_RETRIES = 300
GEN_RETRIES = 500


def _spans_overlap(start_a: int, start_b: int) -> bool:
    """Whether two SQ-wide intervals starting at the given coordinates overlap."""
    return abs(int(start_a) - int(start_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    """Whether two SQxSQ rectangles overlap."""
    return _spans_overlap(r1, r2) and _spans_overlap(c1, c2)


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    """Return True iff any two (row, col, patch) placements overlap as rectangles."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create one bordered SQxSQ square with a uniform interior."""
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _detect_squares(grid: np.ndarray) -> list[dict[str, int | np.ndarray]]:
    """Find framed SQxSQ objects by scanning every candidate window."""
    g = np.array(grid, dtype=int)
    h, w = g.shape
    squares: list[dict[str, int | np.ndarray]] = []
    for r0 in range(h - SQ + 1):
        for c0 in range(w - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame = int(border[0])
            if frame not in {AZURE, RED}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame).any()):
                continue
            squares.append({"frame": frame, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def _render_vertical(
    squares: list[dict[str, int | np.ndarray]],
    shape: tuple[int, int],
    bg: int,
    order_key: str,
) -> np.ndarray:
    """Render the vertical top/bottom stacking rule using the chosen ordering axis."""
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)

    teal = sorted(
        [s for s in squares if int(s["frame"]) == AZURE],
        key=lambda s: (int(s[order_key]), int(s["r0"]) if order_key == "c0" else int(s["c0"])),
    )
    placed_edge_cols: list[int] = []
    for sq in teal:
        c0 = int(sq["c0"])
        has_overlap = any(_spans_overlap(c0, ec) for ec in placed_edge_cols)
        dest_r = SQ if has_overlap else 0
        paste(out, sq["patch"], dest_r, c0, transparent=bg)
        if dest_r == 0:
            placed_edge_cols.append(c0)

    red = sorted(
        [s for s in squares if int(s["frame"]) == RED],
        key=lambda s: (-int(s[order_key]), int(s["r0"]) if order_key == "c0" else int(s["c0"])),
    )
    placed_edge_cols = []
    for sq in red:
        c0 = int(sq["c0"])
        has_overlap = any(_spans_overlap(c0, ec) for ec in placed_edge_cols)
        dest_r = int(h - 2 * SQ) if has_overlap else int(h - SQ)
        paste(out, sq["patch"], dest_r, c0, transparent=bg)
        if dest_r == int(h - SQ):
            placed_edge_cols.append(c0)

    return out


def _solve_with_order(grid: np.ndarray, order_key: str) -> np.ndarray:
    """Shared solver core used by the generator for non-degeneracy checks."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    squares = _detect_squares(g)
    return _render_vertical(squares, g.shape, bg, order_key)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move framed 5x5 squares into top or bottom bands chosen by border color.

    Input structure:
    - The background is the most frequent color.
    - Teal (8) and red (2) framed 5x5 squares with non-frame interiors are scattered across the grid.

    Rule:
    - Keep each square's input column fixed.
    - Process teal squares from smaller input columns to larger ones and push them toward the top edge; process red squares from larger input columns to smaller ones and push them toward the bottom edge.
    - A square uses the edge band first (row 0 for teal, row H-5 for red); if its 5-column span overlaps any earlier edge-band square of the same color, it goes one band inward instead (row 5 for teal, row H-10 for red).

    Key insight: Both decisive ingredients are column-based now: the input column order decides who claims the edge band first, and column-span overlap decides who must drop into the second vertical band.
    """
    return _solve_with_order(np.array(grid, dtype=int), order_key="c0")


# --------------- generator ---------------

def _assign_group_columns(n: int, ascending: bool) -> tuple[list[int], list[int]] | None:
    """Pick output columns whose order reproduces edge vs inward placement."""
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward
    all_cols = list(range(GRID_W - SQ + 1))

    for _ in range(COL_ASSIGN_RETRIES):
        random.shuffle(all_cols)
        edge_cols: list[int] = []
        for c in all_cols:
            if all(not _spans_overlap(c, ec) for ec in edge_cols):
                edge_cols.append(c)
                if len(edge_cols) == n_edge:
                    break
        if len(edge_cols) < n_edge:
            continue

        random.shuffle(all_cols)
        inward_cols: list[int] = []
        for c in all_cols:
            if any(_spans_overlap(c, ic) for ic in inward_cols):
                continue
            if ascending:
                ok_dir = any(_spans_overlap(c, ec) and c > ec for ec in edge_cols)
            else:
                ok_dir = any(_spans_overlap(c, ec) and c < ec for ec in edge_cols)
            if not ok_dir:
                continue
            inward_cols.append(c)
            if len(inward_cols) == n_inward:
                break
        if len(inward_cols) < n_inward:
            continue

        ordered = sorted([(c, True) for c in edge_cols] + [(c, False) for c in inward_cols], key=lambda t: t[0], reverse=not ascending)
        placed_edge_cols: list[int] = []
        valid = True
        for c, is_edge in ordered:
            has_overlap = any(_spans_overlap(c, ec) for ec in placed_edge_cols)
            if is_edge == has_overlap:
                valid = False
                break
            if not has_overlap:
                placed_edge_cols.append(c)
        if valid:
            return edge_cols, inward_cols

    return None


def _place_band_squares(
    output: np.ndarray,
    edge_cols: list[int],
    inward_cols: list[int],
    colors: list[int],
    frame_color: int,
    edge_row: int,
    inward_row: int,
) -> list[dict[str, int | bool | np.ndarray]]:
    """Render one color group's output bands and keep metadata for input synthesis."""
    placed: list[dict[str, int | bool | np.ndarray]] = []
    for i, c in enumerate(edge_cols):
        patch = _make_patch(frame_color, colors[i])
        paste(output, patch, edge_row, c)
        placed.append({"frame": frame_color, "col": int(c), "out_row": int(edge_row), "is_edge": True, "patch": patch})
    for i, c in enumerate(inward_cols):
        patch = _make_patch(frame_color, colors[len(edge_cols) + i])
        paste(output, patch, inward_row, c)
        placed.append({"frame": frame_color, "col": int(c), "out_row": int(inward_row), "is_edge": False, "patch": patch})
    return placed


def _collect_row_order_constraints(
    groups: list[tuple[list[dict[str, int | bool | np.ndarray]], bool]]
) -> list[tuple[int, int, bool]]:
    """Pick witness inward squares that must violate row-based ordering."""
    constraints: list[tuple[int, int, bool]] = []
    next_id = 0
    for entries, ascending in groups:
        for entry in entries:
            entry["input_id"] = next_id
            next_id += 1

        edges = [e for e in entries if bool(e["is_edge"])]
        witnesses = [e for e in entries if not bool(e["is_edge"])]
        random.shuffle(witnesses)
        for witness in witnesses:
            deps = [edge for edge in edges if _spans_overlap(int(witness["col"]), int(edge["col"]))]
            if not deps:
                continue
            for dep in deps:
                constraints.append((int(witness["input_id"]), int(dep["input_id"]), bool(ascending)))
            break
    return constraints


def _derive_input_rows(
    azure_entries: list[dict[str, int | bool | np.ndarray]],
    red_entries: list[dict[str, int | bool | np.ndarray]],
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign scattered input rows while making row-order fail on at least one witness chain."""
    entries = azure_entries + red_entries
    constraints = _collect_row_order_constraints([(azure_entries, True), (red_entries, False)])
    if not constraints:
        return None

    row_pool = list(range(GRID_H - SQ + 1))
    for _ in range(INPUT_ROW_RETRIES):
        chosen_rows = random.sample(row_pool, len(entries))
        positions: list[tuple[int, int, np.ndarray]] = []
        by_id: dict[int, int] = {}
        valid = True
        for entry, row in zip(entries, chosen_rows):
            out_row = int(entry["out_row"])
            if row == out_row:
                valid = False
                break
            by_id[int(entry["input_id"])] = int(row)
            positions.append((int(row), int(entry["col"]), entry["patch"]))
        if not valid or _any_pairwise_overlap(positions):
            continue

        if not all((by_id[a] < by_id[b]) if asc else (by_id[a] > by_id[b]) for a, b, asc in constraints):
            continue
        return positions

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt for the fused column-order + vertical-stack rule."""
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_assign = _assign_group_columns(n_azure, ascending=True)
    if azure_assign is None:
        return None
    red_assign = _assign_group_columns(n_red, ascending=False)
    if red_assign is None:
        return None
    azure_edge_cols, azure_inward_cols = azure_assign
    red_edge_cols, red_inward_cols = red_assign

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    output = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    azure_entries = _place_band_squares(output, azure_edge_cols, azure_inward_cols, azure_colors, AZURE, edge_row=0, inward_row=SQ)
    red_entries = _place_band_squares(
        output,
        red_edge_cols,
        red_inward_cols,
        red_colors,
        RED,
        edge_row=GRID_H - SQ,
        inward_row=GRID_H - 2 * SQ,
    )

    input_positions = _derive_input_rows(azure_entries, red_entries)
    if input_positions is None:
        return None

    inp = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for row, col, patch in input_positions:
        paste(inp, patch, row, col)

    try:
        col_output = main(inp)
        row_output = _solve_with_order(inp, order_key="r0")
    except Exception:
        return None
    if not np.array_equal(col_output, output):
        return None
    if np.array_equal(row_output, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
