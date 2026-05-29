# Combo variant of b5ca7ac4 (index 20)
# combo_spec: D2/2+D13/2
# Name: 框架内优先级标记+颜色值排序

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import crop_bbox, get_bg, paste
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the fused variant
# ------------------------------------
# a) Pixel-level behavior:
#    Each 5x5 frame still has border color 8 or 2, but its 3x3 interior is no longer uniform.
#    The interior uses one fixed "badge" cell at the top-left of the 3x3 area to store a marker
#    color, while the other eight cells share one main fill color. Example on one side:
#      frame A: marker 5, main 7
#      frame B: marker 9, main 1
#      frame C: marker 9, main 3
#      frame D: marker 9, main 6
#    The solver sorts by marker first, then by main color, so the order is A,B,C,D even though
#    B has a smaller main color than A. This creates an explicit "marker beats color" conflict,
#    while C before D demonstrates the secondary tie-break inside one marker tier.
# b) Complexity calibration:
#    The marker palette is kept to two colors and the main-color palette to five fixed colors.
#    That is enough to create repeated evidence across examples without exploding the hypothesis
#    space. The badge position is fixed, so the visual cue is strong and consistent.
# c) Hidden assumptions reviewed:
#    The original solver assumed a uniform 3x3 interior and used input columns to encode order.
#    That no longer holds. The new solver assumes:
#      - a 5x5 border of color 2 or 8,
#      - exactly one badge cell at a fixed interior coordinate,
#      - the remaining eight interior cells are a single main color,
#      - badge colors come from a small known order and main colors come from a fixed visible
#        sequence. The generator enforces all of these.
# d) Placement legality:
#    Rows are chosen first. Edge rows on one side never overlap each other. Each inward row
#    overlaps exactly one anchor edge row and avoids the other edge rows, so stacking depth stays
#    at two columns just like the source task. Input columns are then sampled randomly while
#    preserving row positions and preventing rectangle overlap.
# e) Visual observability:
#    The rule is inferable from a few examples because the badge is a literal visible pixel, and
#    the secondary order uses the same small color sequence repeatedly. The generator forces at
#    least one cross-marker conflict and one same-marker tie case per sample, so both cues matter.
# f) Visual cleanliness:
#    The badge stays in the same corner inside every frame, frames remain 5x5, and side stacking
#    is still limited to edge vs one-step inward. This keeps the output readable instead of messy.
# g) Degeneracy checks:
#    The generator rejects any sample whose output matches the original column-ordered behavior.
#    That prevents accidental fallback to the source task even when the new solver itself is valid.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
BG = 0
MARKER_POS = (0, 0)  # inside the 3x3 interior
MARKER_ORDER = {5: 0, 9: 1}
MAIN_COLOR_ORDER = {1: 0, 3: 1, 4: 2, 6: 3, 7: 4}
MAIN_COLORS = [1, 3, 4, 6, 7]
INPUT_COL_RETRIES = 80
GEN_RETRIES = 600
ROW_ASSIGN_RETRIES = 200
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, object]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _make_patch(frame_color: int, main_color: int, marker_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), frame_color, dtype=int)
    patch[1:-1, 1:-1] = main_color
    patch[1 + MARKER_POS[0], 1 + MARKER_POS[1]] = marker_color
    return patch


def _detect_squares(g: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    squares: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for r0 in range(h - SQ + 1):
        for c0 in range(w - SQ + 1):
            patch = crop_bbox(g, (r0, c0, r0 + SQ - 1, c0 + SQ - 1))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {AZURE, RED}:
                continue

            inner = patch[1:-1, 1:-1]
            marker_color = int(inner[MARKER_POS])
            rest = inner.copy().flatten().tolist()
            marker_flat_idx = MARKER_POS[0] * 3 + MARKER_POS[1]
            del rest[marker_flat_idx]
            if len(set(map(int, rest))) != 1:
                continue
            main_color = int(rest[0])
            if marker_color == main_color:
                continue
            if marker_color not in MARKER_ORDER or main_color not in MAIN_COLOR_ORDER:
                continue

            if (r0, c0) in seen:
                continue
            seen.add((r0, c0))
            squares.append(
                {
                    "frame": frame_col,
                    "r0": r0,
                    "c0": c0,
                    "marker": marker_color,
                    "main": main_color,
                    "patch": patch,
                }
            )
    return squares


def _sort_key(square: dict[str, object]) -> tuple[int, int]:
    return (MARKER_ORDER[int(square["marker"])], MAIN_COLOR_ORDER[int(square["main"])])


def _place_by_order(
    squares: list[dict[str, object]],
    edge_col: int,
    inward_col: int,
    out: np.ndarray,
    bg: int,
) -> None:
    placed_edge_rows: list[tuple[int, int]] = []
    for square in sorted(squares, key=_sort_key):
        r0 = int(square["r0"])
        r1 = r0 + SQ - 1
        has_overlap = any(not (r1 < pr0 or pr1 < r0) for pr0, pr1 in placed_edge_rows)
        dest_c = inward_col if has_overlap else edge_col
        paste(out, square["patch"], r0, dest_c, transparent=bg)
        if not has_overlap:
            placed_edge_rows.append((r0, r1))


def _baseline_column_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)
    squares = _detect_squares(g)
    azure = sorted([s for s in squares if int(s["frame"]) == AZURE], key=lambda s: int(s["c0"]))
    red = sorted([s for s in squares if int(s["frame"]) == RED], key=lambda s: -int(s["c0"]))
    for side_squares, edge_col, inward_col in (
        (azure, 0, SQ),
        (red, g.shape[1] - SQ, g.shape[1] - 2 * SQ),
    ):
        placed_edge_rows: list[tuple[int, int]] = []
        for square in side_squares:
            r0 = int(square["r0"])
            r1 = r0 + SQ - 1
            has_overlap = any(not (r1 < pr0 or pr1 < r0) for pr0, pr1 in placed_edge_rows)
            dest_c = inward_col if has_overlap else edge_col
            paste(out, square["patch"], r0, dest_c, transparent=bg)
            if not has_overlap:
                placed_edge_rows.append((r0, r1))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move 5x5 framed tiles to the side edges using an interior badge-driven priority.

    Input structure:
    - Background is the most common color.
    - Each object is a 5x5 teal or red frame. Inside each frame, one fixed badge pixel sits in
      the top-left cell of the 3x3 interior, and the other eight interior cells share one main
      fill color.

    Rule:
    - Teal-framed tiles move to the left edge; red-framed tiles move to the right edge.
    - Each tile keeps its original row.
    - Within each frame-color group, process tiles by badge priority first, then by main fill
      color as a tie-break.
    - A tile goes at the edge unless its 5-row span overlaps an earlier edge tile on that side,
      in which case it shifts inward by one tile width.

    Key insight:
    - Input position no longer defines the order. The fixed interior badge is the primary
      instruction, and the visible main-color sequence only breaks ties among tiles with the same
      badge.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)
    squares = _detect_squares(g)
    _place_by_order([s for s in squares if int(s["frame"]) == AZURE], 0, SQ, out, bg)
    _place_by_order(
        [s for s in squares if int(s["frame"]) == RED],
        g.shape[1] - SQ,
        g.shape[1] - 2 * SQ,
        out,
        bg,
    )
    return out


def _pattern_for_side(n: int, mode: str) -> list[dict[str, object]]:
    if n >= 4:
        pattern = [
            {"label": "A_edge", "marker": 5, "main": 7},
            {"label": "A_inward", "marker": 9, "main": 1},
            {"label": "B_edge", "marker": 9, "main": 3},
            {"label": "B_inward", "marker": 9, "main": 6},
        ]
        if n == 5:
            pattern = [{"label": "C_edge", "marker": 5, "main": 4}] + pattern
        return pattern
    if mode == "primary":
        return [
            {"label": "A_edge", "marker": 5, "main": 7},
            {"label": "A_inward", "marker": 9, "main": 1},
            {"label": "C_edge", "marker": 9, "main": 4},
        ]
    return [
        {"label": "C_edge", "marker": 5, "main": 7},
        {"label": "B_edge", "marker": 9, "main": 3},
        {"label": "B_inward", "marker": 9, "main": 6},
    ]


def _choose_modes(n_azure: int, n_red: int) -> tuple[str, str]:
    if n_azure >= 4 and n_red >= 4:
        return "both", "both"
    if n_azure >= 4:
        return "both", random.choice(["primary", "secondary"])
    if n_red >= 4:
        return random.choice(["primary", "secondary"]), "both"
    if random.random() < 0.5:
        return "primary", "secondary"
    return "secondary", "primary"


def _assign_rows(pattern: list[dict[str, object]]) -> dict[str, int] | None:
    edge_labels: list[str] = []
    for item in pattern:
        label = str(item["label"])
        if label.endswith("_edge") and label not in edge_labels:
            edge_labels.append(label)
    all_rows = list(range(GRID_H - SQ + 1))

    for _ in range(ROW_ASSIGN_RETRIES):
        random.shuffle(all_rows)
        edge_rows: dict[str, int] = {}
        for row in all_rows:
            if all(not _rows_overlap(row, used) for used in edge_rows.values()):
                edge_rows[edge_labels[len(edge_rows)]] = row
                if len(edge_rows) == len(edge_labels):
                    break
        if len(edge_rows) != len(edge_labels):
            continue

        assigned = dict(edge_rows)
        used_inward_rows: list[int] = []
        ok = True
        for item in pattern:
            label = str(item["label"])
            if not label.endswith("_inward"):
                continue
            anchor = label.replace("_inward", "_edge")
            candidates = [
                row
                for row in all_rows
                if _rows_overlap(row, assigned[anchor])
                and all(
                    label2 == anchor or not _rows_overlap(row, row2)
                    for label2, row2 in edge_rows.items()
                )
                and all(not _rows_overlap(row, prev) for prev in used_inward_rows)
            ]
            if not candidates:
                ok = False
                break
            chosen = random.choice(candidates)
            assigned[label] = chosen
            used_inward_rows.append(chosen)
        if ok:
            return assigned
    return None


def _simulate_output_side(
    pattern: list[dict[str, object]],
    frame_color: int,
    edge_col: int,
    inward_col: int,
    row_map: dict[str, int],
    out: np.ndarray,
) -> list[dict[str, object]] | None:
    ordered = sorted(pattern, key=lambda item: (MARKER_ORDER[int(item["marker"])], MAIN_COLOR_ORDER[int(item["main"])]))
    placed_edge_rows: list[int] = []
    items: list[dict[str, object]] = []
    for item in ordered:
        label = str(item["label"])
        row = int(row_map[label])
        expect_inward = label.endswith("_inward")
        has_overlap = any(_rows_overlap(row, prev) for prev in placed_edge_rows)
        dest_col = inward_col if has_overlap else edge_col
        if has_overlap != expect_inward:
            return None
        patch = _make_patch(frame_color, int(item["main"]), int(item["marker"]))
        paste(out, patch, row, dest_col)
        items.append(
            {
                "row": row,
                "out_col": dest_col,
                "patch": patch,
                "frame": frame_color,
                "marker": int(item["marker"]),
                "main": int(item["main"]),
                "label": label,
            }
        )
        if not has_overlap:
            placed_edge_rows.append(row)
    return items


def _assign_input_columns(items: list[dict[str, object]]) -> list[tuple[int, int, np.ndarray]] | None:
    col_range = list(range(GRID_W - SQ + 1))
    for _ in range(INPUT_COL_RETRIES):
        placed: list[tuple[int, int, np.ndarray]] = []
        order = items[:]
        random.shuffle(order)
        ok = True
        for item in order:
            row = int(item["row"])
            out_col = int(item["out_col"])
            candidates = [
                col
                for col in col_range
                if col != out_col
                and all(not _rects_overlap(row, col, prev_r, prev_c) for prev_r, prev_c, _ in placed)
            ]
            if not candidates:
                ok = False
                break
            chosen_col = random.choice(candidates)
            placed.append((row, chosen_col, item["patch"]))
        if ok:
            return placed
    return None


def _feature_counts(items: list[dict[str, object]]) -> tuple[int, int]:
    primary = 0
    secondary = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a = items[i]
            b = items[j]
            if not _rows_overlap(int(a["row"]), int(b["row"])):
                continue
            a_marker = int(a["marker"])
            b_marker = int(b["marker"])
            a_main = int(a["main"])
            b_main = int(b["main"])
            if a_marker != b_marker and MARKER_ORDER[a_marker] < MARKER_ORDER[b_marker] and MAIN_COLOR_ORDER[a_main] > MAIN_COLOR_ORDER[b_main]:
                primary += 1
            if a_marker == b_marker and MAIN_COLOR_ORDER[a_main] != MAIN_COLOR_ORDER[b_main]:
                secondary += 1
    return primary, secondary


def _gen_attempt() -> np.ndarray | None:
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)
    azure_mode, red_mode = _choose_modes(n_azure, n_red)

    azure_pattern = _pattern_for_side(n_azure, azure_mode)
    red_pattern = _pattern_for_side(n_red, red_mode)
    azure_rows = _assign_rows(azure_pattern)
    red_rows = _assign_rows(red_pattern)
    if azure_rows is None or red_rows is None:
        return None

    output = np.full((GRID_H, GRID_W), BG, dtype=int)
    azure_items = _simulate_output_side(azure_pattern, AZURE, 0, SQ, azure_rows, output)
    if azure_items is None:
        return None
    red_items = _simulate_output_side(red_pattern, RED, GRID_W - SQ, GRID_W - 2 * SQ, red_rows, output)
    if red_items is None:
        return None

    all_items = azure_items + red_items
    primary_count, secondary_count = _feature_counts(all_items)
    if primary_count == 0 or secondary_count == 0:
        return None

    input_positions = _assign_input_columns(all_items)
    if input_positions is None or _any_pairwise_overlap(input_positions):
        return None

    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    for row, col, patch in input_positions:
        paste(inp, patch, row, col)

    try:
        expected = main(inp)
    except Exception:
        return None
    if not np.array_equal(expected, output):
        return None
    if np.array_equal(_baseline_column_output(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
