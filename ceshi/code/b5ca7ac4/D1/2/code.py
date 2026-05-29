# Variant of b5ca7ac4
# Dimension: D1 — 分组标准——边框颜色二分法 (match_cond)
# Change: 通过边框颜色将框架分为两组：边框为青色(8)的归为左堆叠组，边框为红色(2)的归为右堆叠组
#   → 按框架在网格中的位置分组：中心在网格水平中线以上的归左组，以下的归右组

"""
Design analysis
===============

source_task:
  none

variant_spec:
  preserved_structure:
    grid: 22x22
    objects: 5x5 bordered squares with frame color 2 or 8 and non-frame 3x3 interiors
    motion:
      - destination side still depends on a binary grouping
      - original row is preserved
      - left group still uses input-column ascending order
      - right group still uses input-column descending order
      - row-overlap with an already placed edge square still triggers one-step inward stacking
  core_change_D1:
    old_grouping: frame color 8 -> left, frame color 2 -> right
    new_grouping: square center above the horizontal midline -> left, below the midline -> right
    exact_test:
      - solver uses center_row = r0 + 2 for a 5x5 square
      - left group iff center_row < H / 2
      - on a 22x22 grid this means r0 in 0..8 goes left, r0 in 9..17 goes right
  generator_adjustments_required:
    - Because each group must stay inside one half of the grid, per-group counts are reduced to 3..4.
      With 5x5 objects, this keeps the original edge/inward stacking pattern physically realizable.
    - Every generated example mixes frame colors inside both spatial groups.
      This prevents degeneration back into the original color-based task.

behavior_trace:
  original_logic_example:
    - top red square at row 2, col 3
    - bottom teal square at row 14, col 15
    - old solver sends the red square right and the teal square left because only frame color matters
  variant_logic_example:
    - the same top red square is now in the upper half, so it joins the left group
    - the same bottom teal square is now in the lower half, so it joins the right group
    - input-column ordering still only decides edge vs inward placement within each spatial group
  visible_difference:
    - red frames can appear on the left in the output
    - teal frames can appear on the right in the output
    - this is directly visible from a few training pairs, so the grouping cue is spatial rather than chromatic

complexity_calibration:
  - The change is not simplified into a trivial top/bottom sort-and-pack task.
    The original two-layer stacking mechanism remains intact.
  - The generator still creates multiple objects per group, with repeated overlap cases,
    so the solver must infer both grouping and placement order.
  - Frame colors remain only as object appearance; they are deliberately mixed across groups
    so they cannot be used as a hidden shortcut.

implicit_assumption_audit_for_main:
  - Square detection by 5x5 scan still assumes frame colors are exactly 2 or 8 and borders are uniform.
    Generator preserves that.
  - The old `frame_col not in {2, 8}` filter remains valid because D1 changes grouping, not object vocabulary.
  - The old split `[s for s in squares if s["frame"] == 8]` / `== 2` would be wrong now because it hard-coded
    the obsolete color grouping. It must be replaced by a row-center test.
  - The ordering keys based on input column remain valid because D1 does not change the tie-breaking mechanism.
  - The overlap bookkeeping assumes at most two output columns per side. Generator still guarantees that.

placement_legality:
  - Top-group rows are generated only from starts whose centers are above the midline.
  - Bottom-group rows are generated only from starts whose centers are below the midline.
  - Within each group, edge rows are pairwise non-overlapping; inward rows each overlap at least one edge row
    and are pairwise non-overlapping themselves.
  - Input placements are sampled afterward and rejected unless all 5x5 squares are pairwise non-overlapping.
  - Final validation is still `np.array_equal(main(inp), output)`.

visual_observability:
  - Each example contains both red and teal frames in the upper half and in the lower half.
  - Therefore at least one red frame must move left and at least one teal frame must move right in every sample.
  - This guarantees the new grouping rule produces outputs that differ from the original color rule.

tidiness_and_color_inference:
  - Output columns remain the same two clean stacks on each side, so the image stays legible.
  - No new color remapping is introduced; colors are only identity cues for the squares.

note_on_baseline:
  - The textual summary in the prompt mentions top-to-bottom / bottom-to-top handling,
    but the provided solver and spec actually use input-column ordering.
    This variant preserves the provided code/spec behavior and changes only the grouping criterion.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import crop_bbox, get_bg, paste


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_TOP_RANGE = (3, 4)
N_BOTTOM_RANGE = (3, 4)
MAX_INWARD = 2
INPUT_COL_RETRIES = 80
ROW_ASSIGN_RETRIES = 200
FRAME_ASSIGN_RETRIES = 100
GEN_RETRIES = 600


def _goes_left(r0: int, height: int, square_size: int = SQ) -> bool:
    """Return True iff the square center lies above the grid's horizontal midline."""
    center_row = int(r0) + square_size // 2
    return float(center_row) < float(height) / 2.0


def _detect_squares(g: np.ndarray) -> list[dict[str, int | np.ndarray]]:
    """Detect all 5x5 bordered squares with frame color 2 or 8."""
    H, W = g.shape
    squares: list[dict[str, int | np.ndarray]] = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {RED, AZURE}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue
            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered 5x5 squares to grid edges by whether they start above or below the midline.

    Input structure:
    - bg is the most common color.
    - Red (2) and teal (8) framed 5x5 squares with colored interiors are scattered across the board.

    Rule:
    - A square whose center is above the horizontal midline moves to the left edge; otherwise it moves to the right edge.
    - Each square keeps its original row.
    - Left-group squares are processed by input column ascending; right-group squares by input column descending.
    - If a square's rows overlap a square already placed on that side's outer edge, it is placed one square-width inward instead.

    Key insight:
    - Frame color no longer decides the destination side; only the square's vertical position does.
      Input-column ordering and row-overlap still determine which member of each spatial group claims the edge slot.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares = _detect_squares(g)

    left_group = sorted(
        [s for s in squares if _goes_left(int(s["r0"]), H)],
        key=lambda s: int(s["c0"]),
    )
    placed_at_default: list[tuple[int, int]] = []
    for s in left_group:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_default.append((r0, r1))

    right_group = sorted(
        [s for s in squares if not _goes_left(int(s["r0"]), H)],
        key=lambda s: -int(s["c0"]),
    )
    placed_at_default = []
    for s in right_group:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_default.append((r0, r1))

    return out


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a 5x5 framed square with a uniform 3x3 interior."""
    patch = np.full((SQ, SQ), frame_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
    return patch


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Check whether two SQ-tall row spans overlap."""
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    """Check whether two SQxSQ rectangles overlap."""
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, object]]) -> bool:
    """Return True if any two (row, col, ...) entries overlap as SQxSQ rectangles."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _candidate_rows(go_left: bool) -> list[int]:
    """All legal row starts for one spatial group."""
    return [r for r in range(GRID_H - SQ + 1) if _goes_left(r, GRID_H) == go_left]


TOP_ROWS = _candidate_rows(True)
BOTTOM_ROWS = _candidate_rows(False)


def _assign_output_rows(n: int, row_pool: list[int]) -> tuple[list[int], list[int]] | None:
    """Pick edge and inward row positions for one side inside the given row pool."""
    for _ in range(ROW_ASSIGN_RETRIES):
        min_edge = max(1, n - MAX_INWARD)
        max_edge = min(2, n - 1)
        if min_edge > max_edge:
            return None
        n_edge = random.randint(min_edge, max_edge)
        n_inward = n - n_edge

        all_rows = row_pool[:]
        random.shuffle(all_rows)
        edge_rows: list[int] = []
        for r in all_rows:
            if all(not _rows_overlap(r, er) for er in edge_rows):
                edge_rows.append(r)
                if len(edge_rows) == n_edge:
                    break
        if len(edge_rows) < n_edge:
            continue

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
            continue
        return edge_rows, inward_rows
    return None


def _assign_frames(n_top: int, n_bottom: int) -> tuple[list[int], list[int]] | None:
    """Assign frame colors so both spatial groups visibly mix red and teal."""
    total = n_top + n_bottom
    red_low = max(3, 2)
    red_high = min(5, total - 2)
    if red_low > red_high:
        return None

    for _ in range(FRAME_ASSIGN_RETRIES):
        n_red = random.randint(red_low, red_high)
        top_red_low = max(1, n_red - (n_bottom - 1))
        top_red_high = min(n_top - 1, n_red - 1)
        if top_red_low > top_red_high:
            continue
        top_red = random.randint(top_red_low, top_red_high)
        bottom_red = n_red - top_red
        if not (1 <= bottom_red <= n_bottom - 1):
            continue

        top_frames = [RED] * top_red + [AZURE] * (n_top - top_red)
        bottom_frames = [RED] * bottom_red + [AZURE] * (n_bottom - bottom_red)
        random.shuffle(top_frames)
        random.shuffle(bottom_frames)
        return top_frames, bottom_frames
    return None


def _assign_all_inner_colors(
    top_frames: list[int],
    bottom_frames: list[int],
    bg: int,
) -> tuple[list[int], list[int]] | None:
    """Assign interior colors globally uniquely within each frame-color family."""
    all_frames = top_frames + bottom_frames
    red_needed = sum(frame == RED for frame in all_frames)
    teal_needed = sum(frame == AZURE for frame in all_frames)
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    avail_teal = [c for c in range(10) if c not in {bg, AZURE}]
    if red_needed > len(avail_red) or teal_needed > len(avail_teal):
        return None

    red_colors = iter(random.sample(avail_red, red_needed))
    teal_colors = iter(random.sample(avail_teal, teal_needed))
    top_inner: list[int] = []
    bottom_inner: list[int] = []
    for frame in top_frames:
        top_inner.append(next(red_colors) if frame == RED else next(teal_colors))
    for frame in bottom_frames:
        bottom_inner.append(next(red_colors) if frame == RED else next(teal_colors))
    return top_inner, bottom_inner


def _place_group_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    frames: list[int],
    inner_colors: list[int],
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render one group's squares onto output and return placement metadata."""
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    idx = 0
    for r in edge_rows:
        patch = _make_patch(frames[idx], inner_colors[idx])
        paste(output, patch, r, edge_col)
        placed.append((int(r), int(edge_col), True, patch))
        idx += 1
    for r in inward_rows:
        patch = _make_patch(frames[idx], inner_colors[idx])
        paste(output, patch, r, inward_col)
        placed.append((int(r), int(inward_col), False, patch))
        idx += 1
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign random input columns that reproduce the solver's edge/inward placement."""
    edges = [(r, oc, patch) for (r, oc, is_edge, patch) in sq_list if is_edge]
    inwards = [(r, oc, patch) for (r, oc, is_edge, patch) in sq_list if not is_edge]
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
        order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

        inward_indices = list(range(len(inwards)))
        random.shuffle(inward_indices)
        for iw_idx in inward_indices:
            latest_pos = max(
                (
                    pos
                    for dep in inward_deps[iw_idx]
                    for pos, (kind, idx) in enumerate(order)
                    if kind == "edge" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", iw_idx))

        ordered_squares = [
            edges[idx] if kind == "edge" else inwards[idx]
            for kind, idx in order
        ]

        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, oc, patch) in enumerate(ordered_squares):
            ic = chosen_cols[i]
            if ic == oc:
                ok = False
                break
            result.append((int(r), int(ic), patch))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid for the spatial-grouping variant."""
    bg = random.choice([c for c in range(10) if c not in {RED, AZURE}])
    n_top = random.randint(*N_TOP_RANGE)
    n_bottom = random.randint(*N_BOTTOM_RANGE)

    top_assign = _assign_output_rows(n_top, TOP_ROWS)
    bottom_assign = _assign_output_rows(n_bottom, BOTTOM_ROWS)
    if top_assign is None or bottom_assign is None:
        return None
    top_edge_rows, top_inward_rows = top_assign
    bottom_edge_rows, bottom_inward_rows = bottom_assign

    frame_assign = _assign_frames(n_top, n_bottom)
    if frame_assign is None:
        return None
    top_frames, bottom_frames = frame_assign

    inner_assign = _assign_all_inner_colors(top_frames, bottom_frames, bg)
    if inner_assign is None:
        return None
    top_inner, bottom_inner = inner_assign

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    top_out = _place_group_squares(
        output,
        top_edge_rows,
        top_inward_rows,
        top_frames,
        top_inner,
        edge_col=0,
        inward_col=SQ,
    )
    bottom_out = _place_group_squares(
        output,
        bottom_edge_rows,
        bottom_inward_rows,
        bottom_frames,
        bottom_inner,
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )

    top_input = _derive_input_positions(top_out, ascending=True)
    bottom_input = _derive_input_positions(bottom_out, ascending=False)
    if top_input is None or bottom_input is None:
        return None

    all_positions = top_input + bottom_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in all_positions:
        paste(inp, patch, r, c)

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
