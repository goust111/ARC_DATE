# Variant of b5ca7ac4
# Dimension: D6 — 目标位置——堆叠时的重叠处理 (dst)
# Change: 当同组框架在行方向上有垂直重叠时，后处理的框架沿堆叠方向偏移5列（一个框架宽度），使相邻框架紧密排列不覆盖... → 重叠时后来者覆盖前者

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the D6 variant
# a) Concrete behavior and trace:
#    - The objects stay the same: isolated 5x5 framed squares, frame color teal(8) or red(2),
#      with a uniform 3x3 interior whose color is unique within that frame color.
#    - The destination side also stays the same: teal squares go to the left edge, red squares
#      go to the right edge, and every square keeps its original row.
#    - The changed step is only the overlap reaction during stacking. Example on the teal side:
#      an earlier teal square at rows 2..6 is pasted to cols 0..4. A later teal square at rows
#      4..8 overlaps those rows. Original task: the later square moved to cols 5..9. Variant:
#      the later square is still pasted at cols 0..4, so rows 4..6 of the earlier square are
#      overwritten, leaving only its non-overlapped remnant visible.
# b) Complexity calibration:
#    - To keep the difference visible but not chaotic, the generator creates 1-2 overlap pairs
#      per side and forbids triple-depth chains. So each row band shows at most two same-side
#      squares fighting for one edge lane.
#    - Rows inside an overlap pair are always distinct, so one square never disappears completely.
# c) Implicit assumptions from the original main():
#    - Square detection by scanning 5x5 windows is still valid because input squares remain
#      non-overlapping and individually intact.
#    - The old `has_overlap -> inward column` logic depended on the existence of a second lane.
#      That assumption is false in D6; destination column must now ignore overlap and stay at
#      the edge column.
#    - The old `placed_at_default` bookkeeping is no longer needed for placement, because overlap
#      affects only draw priority, not lane choice.
# d) Placement legality:
#    - Input squares are generated as non-overlapping 5x5 rectangles inside the 22x22 grid.
#    - Output overlaps are allowed only within a color side and only because the new rule says
#      later pasted squares cover earlier ones.
# e) Visual observability:
#    - Each side has at least one overlapping pair, so the new overwrite behavior appears in
#      every example. The ordering signal remains visible because it is determined by input
#      columns, not by hidden metadata.
# f) Visual cleanliness:
#    - Overlaps are restricted to disjoint pairs. This avoids tangled three-square merges that
#      would make the surviving remnants hard to parse by eye.
# g) Color inference:
#    - No color-id arithmetic or hidden ordering is introduced. Colors still only identify side
#      (frame color) and square identity (interior color).
# 9) New solver assumptions to match in the generator:
#    - Solver assumes the input still contains standalone 5x5 framed squares, so generator never
#      overlaps squares in the input.
#    - Solver assumes ordering is recoverable from input columns alone, so generator assigns
#      distinct columns per side in strict processing order.
#    - Solver relies on ordinary paste overwrite semantics at the destination edge, so generator
#      intentionally creates same-side row overlaps to exercise that behavior.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move framed 5x5 squares to the edge chosen by their border color.

    Input structure:
    - bg is the most frequent color in the grid.
    - The non-background objects are teal (8) and red (2) 5x5 border squares with colored 3x3 interiors, scattered without overlap.

    Rule:
    - Teal squares move to the left edge (col 0); red squares move to the right edge (col W-5).
    - Every square keeps its original top row.
    - Teal squares are processed by increasing input column, red squares by decreasing input column.
    - If later same-color squares overlap earlier ones in rows, they are still pasted in the same edge lane and cover the shared rows.

    Key insight:
    - Input-column order still determines priority, but overlap no longer creates a second column; it creates overwrite priority within a single edge column.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    out = np.full_like(g, bg)

    square_size = 5

    # Detect framed squares by scanning 5x5 windows. This stays reliable because the
    # input generator keeps the source squares non-overlapping even though outputs can merge.
    squares = []
    for r0 in range(H - square_size + 1):
        for c0 in range(W - square_size + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + square_size - 1), int(c0 + square_size - 1)))

            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {2, 8}:
                continue

            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue

            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})

    teal_squares = sorted([s for s in squares if int(s["frame"]) == 8], key=lambda s: int(s["c0"]))
    for s in teal_squares:
        paste(out, s["patch"], int(s["r0"]), 0, transparent=bg)

    red_squares = sorted([s for s in squares if int(s["frame"]) == 2], key=lambda s: -int(s["c0"]))
    for s in red_squares:
        paste(out, s["patch"], int(s["r0"]), int(W - square_size), transparent=bg)

    return out


# --------------- generator (c) ---------------

# --- Configuration ---
GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_OVERLAP_PAIRS = 2
INPUT_COL_RETRIES = 100
LAYOUT_RETRIES = 200
GEN_RETRIES = 500


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a SQxSQ bordered patch with uniform interior."""
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Check whether two SQ-tall row spans overlap."""
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1, c1, r2, c2) -> bool:
    """Check whether two SQxSQ rectangles overlap."""
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple]) -> bool:
    """Return True if any two (r, c, ...) entries overlap as SQxSQ rectangles."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _count_row_overlaps(rows: list[int]) -> int:
    """Count overlapping SQ-tall row-span pairs."""
    total = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if _rows_overlap(rows[i], rows[j]):
                total += 1
    return total


def _distribute_slack(total: int, slots: int) -> list[int]:
    """Randomly distribute *total* non-negative units across *slots* buckets."""
    gaps = [0] * slots
    for _ in range(int(total)):
        gaps[random.randrange(slots)] += 1
    return gaps


def _choose_pair_offsets(n_pairs: int, n_singles: int) -> list[int]:
    """Choose pair offsets in [1, SQ-1] while keeping the combined layout in-bounds."""
    min_total = SQ * (n_pairs + n_singles) + n_pairs
    if min_total > GRID_H:
        raise ValueError("Not enough height for requested overlap layout")

    offsets = [1] * n_pairs
    budget = GRID_H - (SQ * (n_pairs + n_singles) + sum(offsets))
    while budget > 0:
        expandable = [i for i, off in enumerate(offsets) if off < SQ - 1]
        if not expandable:
            break
        idx = random.choice(expandable)
        offsets[idx] += 1
        budget -= 1
    random.shuffle(offsets)
    return offsets


def _build_side_plan(
    n: int,
    frame_color: int,
    inner_colors: list[int],
) -> list[tuple[int, np.ndarray]]:
    """Build one side's output plan in processing order, with 1-2 overlap pairs and no triples."""
    max_pairs = min(MAX_OVERLAP_PAIRS, n // 2)
    n_pairs = random.randint(1, max_pairs)
    n_singles = n - 2 * n_pairs
    offsets = _choose_pair_offsets(n_pairs, n_singles)

    components = [{"kind": "pair", "offset": off, "length": SQ + off} for off in offsets]
    components.extend({"kind": "single", "length": SQ} for _ in range(n_singles))
    random.shuffle(components)

    total_len = sum(int(comp["length"]) for comp in components)
    slack = GRID_H - total_len
    gaps = _distribute_slack(slack, len(components) + 1)

    cursor = gaps[0]
    for i, comp in enumerate(components):
        comp["start"] = int(cursor)
        cursor += int(comp["length"]) + gaps[i + 1]

    comp_order = list(range(len(components)))
    random.shuffle(comp_order)

    plan: list[tuple[int, np.ndarray]] = []
    color_idx = 0
    for comp_idx in comp_order:
        comp = components[comp_idx]
        start = int(comp["start"])
        if comp["kind"] == "single":
            rows_in_order = [start]
        else:
            off = int(comp["offset"])
            if random.choice([True, False]):
                rows_in_order = [start, start + off]
            else:
                rows_in_order = [start + off, start]
        for row in rows_in_order:
            patch = _make_patch(frame_color, inner_colors[color_idx])
            plan.append((int(row), patch))
            color_idx += 1

    rows = [row for row, _ in plan]
    if _count_row_overlaps(rows) < n_pairs:
        raise AssertionError("Planned overlaps were not realized")
    for i, row_a in enumerate(rows):
        deg = sum(1 for j, row_b in enumerate(rows) if i != j and _rows_overlap(row_a, row_b))
        if deg > 1:
            raise AssertionError("Triple-overlap chain created")
    return plan


def _assign_input_positions(
    side_plan: list[tuple[int, np.ndarray]],
    ascending: bool,
    output_col: int,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign non-overlapping input columns that realize the requested processing order."""
    n = len(side_plan)
    col_range = [c for c in range(0, GRID_W - SQ + 1) if c != int(output_col)]
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result = [(int(row), int(chosen_cols[i]), patch) for i, (row, patch) in enumerate(side_plan)]
        if not _any_pairwise_overlap(result):
            return result
    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None

    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    try:
        azure_plan = _build_side_plan(n_azure, AZURE, azure_colors)
        red_plan = _build_side_plan(n_red, RED, red_colors)
    except (AssertionError, ValueError):
        return None

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    for row, patch in azure_plan:
        paste(output, patch, row, 0)
    for row, patch in red_plan:
        paste(output, patch, row, GRID_W - SQ)

    azure_input = _assign_input_positions(azure_plan, ascending=True, output_col=0)
    if azure_input is None:
        return None
    red_input = _assign_input_positions(red_plan, ascending=False, output_col=GRID_W - SQ)
    if red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for row, col, patch in all_positions:
        paste(inp, patch, row, col)

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
