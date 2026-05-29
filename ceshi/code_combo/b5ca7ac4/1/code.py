# Combo variant of b5ca7ac4 (index 1)
# combo_spec: D1/2+D5/0
# Name: 位置分组+方向反转

"""
Design analysis
===============

source_task:
  base_rule:
    - detect scattered 5x5 framed squares with frame color 2 or 8
    - original solver grouped by frame color
    - original left side used input-column ascending order and right side used descending order
    - row-overlap with an already placed edge square forced one-step inward stacking

fused_variant:
  grouping_change_D1_2:
    - grouping no longer depends on frame color
    - use the square center row instead: center_row = r0 + 2
    - upper group iff center_row < H / 2
    - on a 22x22 grid this means upper starts are r0 in 0..8, lower starts are r0 in 9..17
  direction_change_D5_0:
    - reverse the side assignment associated with the two groups
    - upper-half group now stacks on the right edge
    - lower-half group now stacks on the left edge
    - therefore upper group must use the right-side processing pattern (input col descending)
      and lower group must use the left-side pattern (input col ascending)

behavior_trace:
  concrete_example:
    - square A: red frame, r0=2, c0=4 -> center row 4, so it belongs to the upper group
    - square B: teal frame, r0=6, c0=14 -> center row 8, also upper group
    - square C: red frame, r0=12, c0=3 -> center row 14, lower group
    - original task would send A and C right because they are red, and B left because it is teal
    - fused task sends A and B right because they are in the upper half, and sends C left because it is in the lower half
    - within the upper group, B is processed before A if its input column is farther right; that determines
      which one claims the outer right column and which one is pushed inward when their row spans overlap
  visible_difference:
    - red squares can now appear on the left if they start in the lower half
    - teal squares can now appear on the right if they start in the upper half
    - the destination side is explained by spatial position, while the exact outer-vs-inner slot is still
      explained by side-specific input-column order plus overlap

complexity_calibration:
  - The task still contains two coupled inferences instead of a trivial top/bottom sort:
    first infer the spatial grouping, then infer that the grouping maps to the opposite side.
  - The two-layer stacking mechanism is preserved, so the solver still has to reason about processing order.
  - Generator examples keep multiple objects in each half and deliberately create overlap cases on both sides.
  - Both halves contain both frame colors, preventing a shortcut back to the original color rule.

implicit_assumption_audit_for_main:
  - 5x5 window detection still assumes borders are uniform and frame colors are exactly 2 or 8.
    Generator preserves that vocabulary.
  - Any code path that split objects by `frame == 8` or `frame == 2` would now be wrong and is replaced
    by a center-row test.
  - The old left-side ascending and right-side descending orderings are still valid, but they must now be
    attached to destination side rather than frame color.
  - The overlap bookkeeping still assumes at most two output columns per side; generator keeps max depth at 2.
  - The midline test introduces a new assumption that no generated square sits ambiguously on both sides of the
    threshold. With size 5 and even grid height 22, `center_row < 11` cleanly partitions starts into 0..8 and 9..17.

placement_legality:
  - Upper-group rows are sampled only from starts 0..8; lower-group rows only from 9..17.
  - Within each group, edge rows are pairwise non-overlapping.
  - Each inward row overlaps at least one edge row and inward rows are pairwise non-overlapping.
  - Input placements are sampled afterward and rejected unless every 5x5 square is pairwise non-overlapping.
  - Final generator validation is still `np.array_equal(main(inp), output)`.

visual_observability:
  - Every generated example mixes red and teal frames in the upper half and in the lower half.
  - That guarantees at least one teal square moves right and at least one red square moves left in every sample.
  - A human can therefore see that color is not the grouping key, and also see that the upper group maps to the
    opposite side from the original task's teal-left / red-right convention.

tidiness_and_inference_control:
  - Output remains restricted to two clean columns on each side, so the pictures stay legible.
  - No invisible attribute such as color index ordering is used for grouping or tie-breaking; only visible position
    and visible input-column order matter.
  - Counts are reduced to 3..4 objects per half so the half-grid geometry remains feasible without producing clutter.

generator_non_degeneracy_check:
  - The important failure mode would be samples whose fused output accidentally matches the original color-based
    output. Mixing colors in both halves rules this out structurally.
  - In post-write validation, compare the fused solver output against the original color-based policy across many
    generated inputs; a healthy generator should show large disagreement, not just successful execution.
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
GEN_RETRIES = 700


def _is_upper_half(r0: int, height: int, square_size: int = SQ) -> bool:
    """Return True iff the square center lies above the horizontal midline."""
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
    Move bordered 5x5 squares to opposite grid edges based on whether they begin above or below the midline.

    Input structure:
    - bg is the most common color.
    - Red (2) and teal (8) framed 5x5 squares with colored interiors are scattered across the board.

    Rule:
    - A square whose center is above the horizontal midline moves to the right edge; otherwise it moves to the left edge.
    - Each square keeps its original row.
    - Right-destination squares are processed by input column descending; left-destination squares by input column ascending.
    - If a square's rows overlap a square already placed on that side's outer edge, it is placed one square-width inward instead.

    Key insight:
    - Destination side is now determined by vertical position rather than frame color, and the side mapping is reversed
      relative to the original color rule. Input-column order and row-overlap still decide which member of a spatial
      group takes the edge slot and which one gets bumped inward.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares = _detect_squares(g)

    upper_group = sorted(
        [s for s in squares if _is_upper_half(int(s["r0"]), H)],
        key=lambda s: -int(s["c0"]),
    )
    placed_at_edge: list[tuple[int, int]] = []
    for s in upper_group:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

    lower_group = sorted(
        [s for s in squares if not _is_upper_half(int(s["r0"]), H)],
        key=lambda s: int(s["c0"]),
    )
    placed_at_edge = []
    for s in lower_group:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

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


def _candidate_rows(is_upper: bool) -> list[int]:
    """All legal row starts for one spatial group."""
    return [r for r in range(GRID_H - SQ + 1) if _is_upper_half(r, GRID_H) == is_upper]


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
    """Assign frame colors so both halves visibly mix red and teal."""
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
    """Single attempt to generate a valid input grid for the fused variant."""
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
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )
    bottom_out = _place_group_squares(
        output,
        bottom_edge_rows,
        bottom_inward_rows,
        bottom_frames,
        bottom_inner,
        edge_col=0,
        inward_col=SQ,
    )

    top_input = _derive_input_positions(top_out, ascending=False)
    bottom_input = _derive_input_positions(bottom_out, ascending=True)
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
