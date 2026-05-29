# Combo variant of b5ca7ac4 (index 22)
# combo_spec: D5/2+D6/2
# Name: 对角线堆叠+覆盖重叠

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the D5/2 + D6/2 combo
# a) Concrete behavior and trace:
#    - Objects stay the same as the original task: isolated 5x5 framed squares with frame color
#      teal(8) or red(2), and a uniform 3x3 interior whose color is unique within that frame
#      color. Input squares remain intact and non-overlapping.
#    - The fused destination rule is instantiated as a diagonal projection that keeps each
#      square's top row but changes its destination column:
#        teal at input row r goes to output top-left (r, r),
#        red at input row r goes to output top-left (r, W-5-r).
#      This realizes the two requested diagonals directly on the 22x22 canvas while remaining
#      feasible for up to five red squares. A literal "corner anchor + add (5,5) every time"
#      chain would not fit five 5x5 squares vertically, so the row-indexed projection is the
#      concrete diagonal form used here.
#    - Example on the teal side: a teal square at (row 3, any input col) moves to (3,3); another
#      teal square at row 6 moves to (6,6). Their row difference is 3 and their column difference
#      is also 3, so the two 5x5 outputs overlap in a 2x2 area. Original task: the second square
#      would have moved to a separate horizontal lane near the left edge. This combo keeps both on
#      the same diagonal and lets the later-processed square overwrite the shared cells.
# b) Complexity calibration:
#    - The generator always creates at least one same-color overlap pair per side, so both the
#      diagonal projection and the overwrite priority are visible in every sample.
#    - Overlaps are limited to disjoint pairs, with no triple-overlap chains. That keeps the
#      visual effect readable: the solver must notice partial remnants, but the output does not
#      devolve into an unreadable knot.
# c) Implicit assumptions from the original main():
#    - Scanning 5x5 windows to detect framed squares still works because the input generator keeps
#      source squares non-overlapping.
#    - The original `has_overlap -> shift inward to a second lane` assumption is invalid here.
#      Destination is now determined only by frame color and original row, while overlap changes
#      only draw priority through ordinary paste overwrite semantics.
#    - The original bookkeeping of which rows already occupied the edge lane is no longer relevant.
#      Overlap is not prevented; it is required and resolved by processing order.
# d) Placement legality:
#    - Input placement is constrained to top rows in [0, 17], which guarantees both diagonal
#      formulas produce in-bounds destination columns in [0, 17].
#    - Input squares are rejected if any pair overlaps. Output plans are also rejected if a teal
#      rectangle would overlap a red rectangle, so the only overwrites in the final output are the
#      intended same-color, same-rule overlaps.
# e) Visual observability:
#    - A human can see that teal objects land on the main diagonal and red objects on the opposite
#      diagonal. Because overlap pairs have row differences 1..4, their column differences match
#      and the partial covering pattern becomes visible in both axes, which is the distinctive cue
#      of the combo.
#    - Ordering is inferred from visible input columns only: teal uses left-to-right priority and
#      red uses right-to-left priority, exactly as in the source task.
# f) Visual cleanliness:
#    - Same-color overlaps are pairwise only, and cross-color output collisions are forbidden.
#      This keeps each diagonal legible and prevents hidden dependencies on the order between color
#      groups.
# g) Change inferability:
#    - No new color mapping is introduced. Frame color still defines the destination diagonal, and
#      the interior colors only distinguish individual squares within a group.
# 9) New solver assumptions to match in the generator:
#    - `main()` assumes each detected square's top row is also a valid diagonal column index, so
#      the generator only emits rows 0..17.
#    - `main()` assumes overwrite order comes only from sorted input columns, so the generator
#      assigns strictly ordered, distinct input columns per color group.
#    - `main()` assumes same-color destination overlaps may occur and should be preserved, so the
#      generator deliberately constructs 1-2 overlapping pairs per side.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_OVERLAP_PAIRS = 2
INPUT_COL_RETRIES = 200
LAYOUT_RETRIES = 300
GEN_RETRIES = 500


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a SQxSQ framed square with a uniform 3x3 interior."""
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Return True iff two SQ-tall spans overlap."""
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    """Return True iff two SQxSQ rectangles overlap."""
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    """Return True iff any placed SQxSQ rectangles overlap."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _output_col(frame_color: int, row: int, width: int = GRID_W) -> int:
    """Destination column on the frame color's diagonal."""
    row = int(row)
    if int(frame_color) == AZURE:
        return row
    return int(width - SQ - row)


def _count_row_overlaps(rows: list[int]) -> int:
    """Count overlapping row-span pairs in a plan."""
    total = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if _rows_overlap(rows[i], rows[j]):
                total += 1
    return total


def _distribute_slack(total: int, slots: int) -> list[int]:
    """Distribute non-negative slack across layout gaps."""
    gaps = [0] * int(slots)
    for _ in range(int(total)):
        gaps[random.randrange(slots)] += 1
    return gaps


def _choose_pair_offsets(n_pairs: int, n_singles: int) -> list[int]:
    """Choose overlap offsets in [1, SQ-1] while staying inside the grid height."""
    for _ in range(200):
        offsets = [random.randint(1, SQ - 1) for _ in range(int(n_pairs))]
        if SQ * (n_pairs + n_singles) + sum(offsets) <= GRID_H:
            return offsets
    raise ValueError("Not enough height for requested overlap layout")


def _build_side_plan(
    n: int,
    frame_color: int,
    inner_colors: list[int],
) -> list[tuple[int, np.ndarray]]:
    """Build one side's output plan in processing order."""
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

    cursor = int(gaps[0])
    for i, comp in enumerate(components):
        comp["start"] = cursor
        cursor += int(comp["length"]) + int(gaps[i + 1])

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
            if not 0 <= int(row) <= GRID_H - SQ:
                raise AssertionError("Planned row is out of bounds")
            patch = _make_patch(frame_color, inner_colors[color_idx])
            plan.append((int(row), patch))
            color_idx += 1

    rows = [row for row, _ in plan]
    if _count_row_overlaps(rows) < n_pairs:
        raise AssertionError("Planned overlaps were not realized")
    for i, row_a in enumerate(rows):
        degree = sum(1 for j, row_b in enumerate(rows) if i != j and _rows_overlap(row_a, row_b))
        if degree > 1:
            raise AssertionError("Triple-overlap chain created")
    return plan


def _output_positions(
    side_plan: list[tuple[int, np.ndarray]],
    frame_color: int,
) -> list[tuple[int, int, np.ndarray]]:
    """Convert a row plan into diagonal destination positions."""
    return [(int(row), _output_col(frame_color, int(row)), patch) for row, patch in side_plan]


def _cross_side_output_overlap(
    azure_plan: list[tuple[int, np.ndarray]],
    red_plan: list[tuple[int, np.ndarray]],
) -> bool:
    """Reject output layouts where teal and red rectangles would collide."""
    azure_pos = _output_positions(azure_plan, AZURE)
    red_pos = _output_positions(red_plan, RED)
    for r1, c1, _ in azure_pos:
        for r2, c2, _ in red_pos:
            if _rects_overlap(r1, c1, r2, c2):
                return True
    return False


def _assign_input_positions(
    side_plan: list[tuple[int, np.ndarray]],
    ascending: bool,
    frame_color: int,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign non-overlapping input columns that realize the side's processing order."""
    n = len(side_plan)
    col_range = list(range(0, GRID_W - SQ + 1))

    for _ in range(INPUT_COL_RETRIES):
        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result = []
        valid = True
        for i, (row, patch) in enumerate(side_plan):
            input_col = int(chosen_cols[i])
            output_col = _output_col(frame_color, int(row))
            if input_col == output_col:
                valid = False
                break
            result.append((int(row), input_col, patch))
        if not valid:
            continue
        if not _any_pairwise_overlap(result):
            return result
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reposition framed 5x5 squares onto color-specific diagonals, resolving clashes by overwrite.

    Input structure:
    - bg is the most frequent color in the grid.
    - The non-background objects are scattered teal (8) and red (2) 5x5 frame squares with
      non-background, non-frame 3x3 interiors.

    Rule:
    - Each square keeps its original top row.
    - Teal squares are projected onto the main diagonal: a square from row r goes to col r.
    - Red squares are projected onto the opposite diagonal: a square from row r goes to col W-5-r.
    - Teal squares are processed by increasing input column; red squares by decreasing input column.
      If projected same-color squares overlap, the later processed one simply covers the earlier one.

    Key insight:
    - Row determines the diagonal landing spot, while input-column order determines overwrite
      priority inside overlapping diagonal pairs.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    out = np.full_like(g, bg)

    squares = []
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
            if not bool((inner != frame_col).any()):
                continue

            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})

    teal_squares = sorted([s for s in squares if int(s["frame"]) == AZURE], key=lambda s: int(s["c0"]))
    for square in teal_squares:
        row = int(square["r0"])
        dest_col = _output_col(AZURE, row, W)
        paste(out, square["patch"], row, dest_col, transparent=bg)

    red_squares = sorted([s for s in squares if int(s["frame"]) == RED], key=lambda s: -int(s["c0"]))
    for square in red_squares:
        row = int(square["r0"])
        dest_col = _output_col(RED, row, W)
        paste(out, square["patch"], row, dest_col, transparent=bg)

    return out


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

    azure_plan = None
    red_plan = None
    for _ in range(LAYOUT_RETRIES):
        try:
            candidate_azure = _build_side_plan(n_azure, AZURE, azure_colors)
            candidate_red = _build_side_plan(n_red, RED, red_colors)
        except (AssertionError, ValueError):
            continue
        if _cross_side_output_overlap(candidate_azure, candidate_red):
            continue
        azure_plan = candidate_azure
        red_plan = candidate_red
        break
    if azure_plan is None or red_plan is None:
        return None

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    for row, patch in azure_plan:
        paste(output, patch, int(row), _output_col(AZURE, int(row)))
    for row, patch in red_plan:
        paste(output, patch, int(row), _output_col(RED, int(row)))

    azure_input = _assign_input_positions(azure_plan, ascending=True, frame_color=AZURE)
    if azure_input is None:
        return None
    red_input = _assign_input_positions(red_plan, ascending=False, frame_color=RED)
    if red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for row, col, patch in all_positions:
        paste(inp, patch, int(row), int(col))

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
