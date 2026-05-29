# Combo variant of b5ca7ac4 (index 23)
# combo_spec: D7/2+D12/0
# Name: 等间距行分配+极不对称分组

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the fused variant
# -------------------------------------
# 1. Concrete behavior:
#    - The original task keeps each square's input row and only changes columns.
#    - This variant keeps the original left/right grouping and the same per-color
#      column ordering, but row placement is no longer copied from input.
#    - After sorting a color-group by its processing order, rows are reassigned to
#      evenly spaced slots from top to bottom. Example for H=22 and n=6:
#      rows -> [0, 3, 7, 10, 14, 17]. With SQ=5 this creates a visible pattern
#      where ranks 0,2,4 can stay on the edge while 1,3,5 overlap those edge rows
#      and are pushed one square inward. A singleton group has only row 0.
#    - D12/0 forces a strong 1-vs-5/6 asymmetry, so one side of the output has a
#      single top-aligned frame while the other side shows the evenly distributed,
#      deeper stack.
#
# 2. Complexity calibration:
#    - The row slots are deterministic and few, so the rule is visible instead of
#      noisy. Using 5-6 items keeps repeated evidence for the equal-spacing rule.
#    - We preserve the original two-depth stacking logic only; this avoids a messy
#      triple-overlap layout while still showing interaction between the new rows
#      and the old edge/inward rule.
#
# 3. Implicit-assumption audit:
#    - Original code assumes output row = input row; this is deliberately removed.
#      Generator must therefore guarantee that the input columns alone determine
#      group order, while input rows can be arbitrary.
#    - The overlap test now depends on reassigned output rows, not input rows.
#    - The square detector still assumes valid non-overlapping 5x5 bordered frames;
#      generator enforces non-overlap so the scan finds one window per object.
#    - Unique interior colors per frame type remain useful, because they let a
#      solver visually track that objects changed rows rather than only columns.
#
# 4. Placement legality and observability:
#    - Output rows are generated inside [0, H-SQ] by linspace-like spacing, so no
#      square leaves the grid.
#    - Input squares are scattered with rejection sampling on fixed columns and
#      random rows, guaranteeing non-overlap and ensuring every square changes row.
#    - Generator rejects samples that would match the legacy row-preserving output,
#      so the produced training examples visibly demonstrate the fused behavior.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
MIN_GROUP_SIZE = 1
MAJORITY_RANGE = (5, 6)
GEN_RETRIES = 600
INPUT_ROW_RETRIES = 200
INPUT_COL_POOL = [
    c for c in range(GRID_W - SQ + 1)
    if c not in {0, SQ, GRID_W - 2 * SQ, GRID_W - SQ}
]


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _target_rows(height: int, n: int) -> list[int]:
    if n <= 0:
        return []
    if n == 1:
        return [0]
    rows = np.linspace(0, int(height - SQ), int(n))
    result = [int(round(x)) for x in rows]
    for i in range(1, len(result)):
        if result[i] <= result[i - 1]:
            result[i] = result[i - 1] + 1
    if result[-1] > int(height - SQ):
        shift = result[-1] - int(height - SQ)
        result = [r - shift for r in result]
    return result


def _find_squares(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.array(grid, dtype=int)
    height, width = g.shape
    squares: list[dict[str, object]] = []
    for r0 in range(height - SQ + 1):
        for c0 in range(width - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {AZURE, RED}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue
            squares.append(
                {
                    "frame": frame_col,
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch.copy(),
                }
            )
    return squares


def _render_group(
    out: np.ndarray,
    squares: list[dict[str, object]],
    *,
    edge_col: int,
    inward_col: int,
    bg: int,
    reassign_rows: bool,
) -> list[dict[str, object]]:
    if not squares:
        return []
    height = int(out.shape[0])
    target_rows = _target_rows(height, len(squares)) if reassign_rows else [int(s["r0"]) for s in squares]
    placed_edge_rows: list[int] = []
    placed: list[dict[str, object]] = []
    for idx, sq in enumerate(squares):
        row = int(target_rows[idx])
        has_overlap = any(_rows_overlap(row, prev_row) for prev_row in placed_edge_rows)
        col = int(inward_col if has_overlap else edge_col)
        paste(out, sq["patch"], row, col, transparent=bg)
        if col == int(edge_col):
            placed_edge_rows.append(row)
        placed.append(
            {
                "frame": int(sq["frame"]),
                "patch": np.array(sq["patch"], dtype=int),
                "src_row": int(sq["r0"]),
                "src_col": int(sq["c0"]),
                "row": row,
                "col": col,
                "is_edge": col == int(edge_col),
                "rank": idx,
            }
        )
    return placed


def _solve(grid: np.ndarray, *, reassign_rows: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    height, width = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)
    squares = _find_squares(g)

    azure = sorted(
        [sq for sq in squares if int(sq["frame"]) == AZURE],
        key=lambda sq: int(sq["c0"]),
    )
    red = sorted(
        [sq for sq in squares if int(sq["frame"]) == RED],
        key=lambda sq: -int(sq["c0"]),
    )

    _render_group(out, azure, edge_col=0, inward_col=SQ, bg=bg, reassign_rows=reassign_rows)
    _render_group(
        out,
        red,
        edge_col=int(width - SQ),
        inward_col=int(width - 2 * SQ),
        bg=bg,
        reassign_rows=reassign_rows,
    )
    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, reassign_rows=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack the bordered 5x5 squares by frame color, but assign their output rows by group rank.

    Input structure:
    - Background is the most common color.
    - The foreground contains scattered 5x5 framed squares with border color 8 (azure) or 2 (red).
    - One border-color group has only 1 square; the other has 5 or 6 squares.

    Rule:
    - Azure squares are processed by increasing input column and placed toward the left side.
    - Red squares are processed by decreasing input column and placed toward the right side.
    - Within each color-group, the kth processed square is moved to the kth evenly spaced row slot:
      top slot for rank 0, then progressively lower slots down to the last legal 5x5 top row.
    - After receiving that reassigned row, a square sits on its side edge unless its 5-row span
      overlaps an already edge-placed square of the same group; overlapping ones shift inward by 5 cells.

    Key insight:
    - The decisive information is the per-color ordering by horizontal position plus the extreme 1-vs-5/6
      group-size imbalance. Input rows are not preserved; they are replaced by a deterministic evenly
      spaced row pattern, and that new pattern then drives the edge-versus-inward stacking.
    """
    return _solve(grid, reassign_rows=True)


def _group_layout(
    *,
    frame_color: int,
    count: int,
    inner_colors: list[int],
    edge_col: int,
    inward_col: int,
    ascending: bool,
    bg: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    output = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    ordered_squares = [
        {"frame": int(frame_color), "r0": -1, "c0": -1, "patch": _make_patch(frame_color, inner_colors[i])}
        for i in range(count)
    ]
    placed = _render_group(
        output,
        ordered_squares,
        edge_col=edge_col,
        inward_col=inward_col,
        bg=int(bg),
        reassign_rows=True,
    )
    for idx, item in enumerate(placed):
        item["ascending"] = bool(ascending)
        item["out_row"] = int(item["row"])
        item["out_col"] = int(item["col"])
        item["order_index"] = idx
    return placed, output


def _assign_input_columns(count: int, *, ascending: bool) -> list[int]:
    cols = sorted(random.sample(INPUT_COL_POOL, count))
    return cols if ascending else cols[::-1]


def _assign_input_positions(items: list[dict[str, object]]) -> list[dict[str, object]] | None:
    row_candidates = list(range(GRID_H - SQ + 1))
    indexed = list(range(len(items)))
    indexed.sort(
        key=lambda i: sum(abs(int(items[i]["in_col"]) - int(items[j]["in_col"])) < SQ for j in range(len(items))),
        reverse=True,
    )

    for _ in range(INPUT_ROW_RETRIES):
        placed_rows: dict[int, int] = {}
        ok = True
        for idx in indexed:
            item = items[idx]
            in_col = int(item["in_col"])
            out_row = int(item["out_row"])
            candidates = [r for r in row_candidates if r != out_row]
            random.shuffle(candidates)
            chosen = None
            for r in candidates:
                if any(
                    _rects_overlap(r, in_col, placed_rows[other], int(items[other]["in_col"]))
                    for other in placed_rows
                ):
                    continue
                chosen = r
                break
            if chosen is None:
                ok = False
                break
            placed_rows[idx] = chosen
        if ok:
            for idx, row in placed_rows.items():
                items[idx]["in_row"] = int(row)
            return items
    return None


def _sample_group_colors(bg: int, frame_color: int, count: int) -> list[int]:
    choices = [c for c in range(10) if c not in {int(bg), int(frame_color)}]
    return random.sample(choices, count)


def _build_variant_instance() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    minority_frame = random.choice([AZURE, RED])
    majority_frame = RED if minority_frame == AZURE else AZURE
    counts = {
        minority_frame: MIN_GROUP_SIZE,
        majority_frame: random.randint(*MAJORITY_RANGE),
    }

    azure_layout, azure_output = _group_layout(
        frame_color=AZURE,
        count=counts.get(AZURE, 0),
        inner_colors=_sample_group_colors(bg, AZURE, counts.get(AZURE, 0)),
        edge_col=0,
        inward_col=SQ,
        ascending=True,
        bg=bg,
    )
    red_layout, red_output = _group_layout(
        frame_color=RED,
        count=counts.get(RED, 0),
        inner_colors=_sample_group_colors(bg, RED, counts.get(RED, 0)),
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
        ascending=False,
        bg=bg,
    )

    output = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    paste(output, azure_output, 0, 0, transparent=bg)
    paste(output, red_output, 0, 0, transparent=bg)

    all_items = azure_layout + red_layout
    for frame_color, ascending in ((AZURE, True), (RED, False)):
        group = [item for item in all_items if int(item["frame"]) == frame_color]
        cols = _assign_input_columns(len(group), ascending=ascending)
        for item, col in zip(group, cols):
            item["in_col"] = int(col)

    placed = _assign_input_positions(all_items)
    if placed is None:
        return None

    inp = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for item in placed:
        paste(inp, item["patch"], int(item["in_row"]), int(item["in_col"]))

    if not np.array_equal(main(inp), output):
        return None
    if np.array_equal(_legacy_main(inp), output):
        return None
    return inp, output


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    squares = _find_squares(np.array(grid, dtype=int))
    by_frame = {
        AZURE: sorted([sq for sq in squares if int(sq["frame"]) == AZURE], key=lambda sq: int(sq["c0"])),
        RED: sorted([sq for sq in squares if int(sq["frame"]) == RED], key=lambda sq: -int(sq["c0"])),
    }
    changed_rows = 0
    for frame_color in (AZURE, RED):
        rows = _target_rows(grid.shape[0], len(by_frame[frame_color]))
        for sq, row in zip(by_frame[frame_color], rows):
            changed_rows += int(int(sq["r0"]) != int(row))
    legacy_same = int(np.array_equal(main(grid), _legacy_main(grid)))
    return {
        "azure_count": len(by_frame[AZURE]),
        "red_count": len(by_frame[RED]),
        "changed_rows": changed_rows,
        "legacy_same": legacy_same,
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        instance = _build_variant_instance()
        if instance is not None:
            return instance[0]
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
