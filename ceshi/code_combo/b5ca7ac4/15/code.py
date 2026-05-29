# Combo variant of b5ca7ac4 (index 15)
# combo_spec: D1/3+D17/2
# Name: 距离分组+最近边缘堆叠

"""
Design analysis
===============

source_task:
  - The original task detects scattered 5x5 framed squares with frame colors 8 and 2.
  - It then splits them by frame color: teal always stacks on the left, red always stacks on the right.
  - Within each side, the square closer to that side in the input is processed first and claims the outer edge slot.
  - Any later square whose 5-row span overlaps an already placed outer-edge square is pushed one square-width inward.

fused_rule:
  - Grouping no longer depends on frame color at all.
  - For each square, compute distance to the left edge as `c0` and distance to the right edge as `W - SQ - c0`.
  - The square joins the side with the smaller distance and also stacks toward that same nearest edge.
  - Therefore the grouping key, movement direction, and per-side processing order all use the same visible distance signal.

behavior_trace:
  - Example on a 22x22 grid with SQ=5:
    a square at c0=2 has distances (left=2, right=15), so it joins the left group and is processed by ascending c0.
    a square at c0=14 has distances (left=14, right=3), so it joins the right group and is processed by descending c0.
  - If those two squares are teal and red respectively, the fused task matches the original for those objects.
  - The important changed case is a red square at c0=1 or a teal square at c0=16:
    the original task would still send them by color, but the fused task sends them by edge proximity instead.
  - If two left-group squares overlap in rows, the one with smaller input c0 takes col 0 and the later one goes to col 5.
    The same logic mirrors on the right using larger input c0 first, with cols W-5 and W-10.

complexity_calibration:
  - This is not a trivial simplification, because the solver still must infer two coupled ideas:
    1. left vs right membership comes from horizontal distance, not color
    2. outer-vs-inner placement still depends on per-side input order and row overlap
  - To keep the difference visually obvious, the generator forces both frame colors to appear on both output sides.
    That prevents accidental regression to "teal-left / red-right" examples.
  - The rule still uses only two stack depths per side, which keeps the images readable and avoids clutter.

implicit_assumption_audit_for_main:
  - Detection still assumes every object is a clean 5x5 frame with uniform border color 2 or 8 and a nontrivial interior.
    The generator preserves exactly that object vocabulary.
  - The original split by `frame == 8` versus `frame == 2` is no longer valid; every place that depended on it is replaced
    by a nearest-edge computation from the square's input column.
  - Because all objects have identical width, comparing left and right distances using the top-left column is sufficient.
  - On the fixed 22x22 canvas with SQ=5, `c0` is integral and `W - SQ = 17`, so exact left/right ties cannot occur.
    The solver still includes a deterministic tie break to the left for robustness, but the generator never relies on it.
  - The overlap bookkeeping still assumes only two legal output columns per side; the generator caps depth at edge + inward.

placement_legality_and_generator_guards:
  - Output rows are generated side-by-side exactly as in the base task: outer-edge rows never overlap each other, while each
    inward row overlaps at least one outer-edge row and no other inward row on that side.
  - Input columns are not sampled blindly. Instead, the generator first decides a valid solver processing order for each side,
    then backtracks over side-compatible input columns until all 5x5 input squares are pairwise non-overlapping.
  - The left-group input columns are always chosen from the left-nearer half and the right-group columns from the right-nearer
    half, so the intended grouping is guaranteed by construction.

visual_observability_and_non_degeneracy:
  - Every generated sample contains at least one red square nearer the left edge and at least one teal square nearer the right
    edge. A human can see from a few examples that color no longer determines the destination side.
  - A useful quantitative degeneration check is:
    1. count "cross-color" objects = red-left plus teal-right in the generated input
    2. compare fused output with the original color-based solver on the same input
    Healthy samples should have cross-color count >= 2 and nonzero pixel difference from the original output.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_ORDER_RETRIES = 120
GEN_RETRIES = 600


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, Any]]) -> bool:
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


def _nearest_side(c0: int, width: int) -> str:
    left_dist = int(c0)
    right_dist = int(width - SQ - c0)
    return "left" if left_dist <= right_dist else "right"


def _detect_squares(g: np.ndarray) -> list[dict[str, Any]]:
    H, W = g.shape
    squares: list[dict[str, Any]] = []
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
            squares.append(
                {
                    "frame": int(frame_col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch,
                }
            )
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered 5x5 squares to the side that is horizontally closest to where they start.

    Input structure:
    - The background is the most common color.
    - Several 5x5 framed squares with border color 8 or 2 are scattered across the grid, each with a colored 3x3 interior.

    Rule:
    - For each square, compare its distance to the left edge and to the right edge of the grid.
    - Squares nearer the left edge stack on the left using input-column ascending order; squares nearer the right edge stack on the right using input-column descending order.
    - Each square keeps its original row.
    - If a square's 5-row span overlaps a square already placed on that side's outer edge, it is placed one square-width inward instead.

    Key insight:
    - The same distance-to-edge measurement now drives both the left/right grouping and the destination edge itself, so color only identifies the square type and no longer determines where it moves.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares = _detect_squares(g)

    left_group = sorted(
        [s for s in squares if _nearest_side(int(s["c0"]), W) == "left"],
        key=lambda s: (int(s["c0"]), int(s["r0"])),
    )
    placed_at_edge: list[tuple[int, int]] = []
    for s in left_group:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    right_group = sorted(
        [s for s in squares if _nearest_side(int(s["c0"]), W) == "right"],
        key=lambda s: (-int(s["c0"]), int(s["r0"])),
    )
    placed_at_edge = []
    for s in right_group:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

    return out


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    max_inward = min(MAX_INWARD, int(n) - 1)
    if max_inward < 1:
        return None

    n_inward = random.randint(1, max_inward)
    n_edge = int(n) - n_inward

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


def _build_side_slots(side: str, n: int) -> list[dict[str, Any]] | None:
    row_assign = _assign_output_rows(n)
    if row_assign is None:
        return None
    edge_rows, inward_rows = row_assign
    edge_col = 0 if side == "left" else GRID_W - SQ
    inward_col = SQ if side == "left" else GRID_W - 2 * SQ

    slots: list[dict[str, Any]] = []
    for r in edge_rows:
        slots.append({"side": side, "r0": int(r), "out_col": int(edge_col), "is_edge": True})
    for r in inward_rows:
        slots.append({"side": side, "r0": int(r), "out_col": int(inward_col), "is_edge": False})
    return slots


def _choose_side_color_mix(n_azure: int, n_red: int) -> tuple[int, int, int, int, int, int] | None:
    total = int(n_azure + n_red)
    feasible: list[tuple[int, int, int, int, int, int]] = []
    for n_left in range(2, total - 1):
        n_right = total - n_left
        left_azure_lo = max(1, n_left - (n_red - 1))
        left_azure_hi = min(n_azure - 1, n_left - 1)
        for left_azure in range(left_azure_lo, left_azure_hi + 1):
            left_red = n_left - left_azure
            right_azure = n_azure - left_azure
            right_red = n_red - left_red
            if min(left_azure, left_red, right_azure, right_red) < 1:
                continue
            feasible.append(
                (
                    int(n_left),
                    int(n_right),
                    int(left_azure),
                    int(left_red),
                    int(right_azure),
                    int(right_red),
                )
            )
    if not feasible:
        return None
    return random.choice(feasible)


def _decorate_slots(
    left_slots: list[dict[str, Any]],
    right_slots: list[dict[str, Any]],
    bg: int,
    left_azure: int,
    left_red: int,
    right_azure: int,
    right_red: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]] | None:
    n_azure = left_azure + right_azure
    n_red = left_red + right_red
    azure_choices = [c for c in range(10) if c not in {bg, AZURE}]
    red_choices = [c for c in range(10) if c not in {bg, RED}]
    if len(azure_choices) < n_azure or len(red_choices) < n_red:
        return None

    azure_colors = random.sample(azure_choices, n_azure)
    red_colors = random.sample(red_choices, n_red)
    azure_idx = 0
    red_idx = 0

    output = np.full((GRID_H, GRID_W), int(bg), dtype=int)

    def assign_side(slots: list[dict[str, Any]], n_side_azure: int, n_side_red: int) -> list[dict[str, Any]]:
        nonlocal azure_idx, red_idx
        decorated: list[dict[str, Any]] = [dict(slot) for slot in slots]
        frames = [AZURE] * int(n_side_azure) + [RED] * int(n_side_red)
        random.shuffle(frames)
        random.shuffle(decorated)
        for slot, frame in zip(decorated, frames):
            if int(frame) == AZURE:
                patch = _make_patch(AZURE, int(azure_colors[azure_idx]))
                azure_idx += 1
            else:
                patch = _make_patch(RED, int(red_colors[red_idx]))
                red_idx += 1
            slot["frame"] = int(frame)
            slot["patch"] = patch
            paste(output, patch, int(slot["r0"]), int(slot["out_col"]))
        return decorated

    left_decorated = assign_side(left_slots, left_azure, left_red)
    right_decorated = assign_side(right_slots, right_azure, right_red)
    return output, left_decorated, right_decorated


def _sample_processing_order(side_slots: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    edges = [slot for slot in side_slots if bool(slot["is_edge"])]
    inwards = [slot for slot in side_slots if not bool(slot["is_edge"])]
    if not edges:
        return None

    inward_deps: list[list[int]] = []
    for inward in inwards:
        deps = [i for i, edge in enumerate(edges) if _rows_overlap(int(inward["r0"]), int(edge["r0"]))]
        if not deps:
            return None
        inward_deps.append(deps)

    edge_indices = list(range(len(edges)))
    random.shuffle(edge_indices)
    order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

    inward_indices = list(range(len(inwards)))
    random.shuffle(inward_indices)
    for iw_idx in inward_indices:
        latest_pos = max(
            pos
            for pos, (kind, idx) in enumerate(order)
            if kind == "edge" and idx in inward_deps[iw_idx]
        )
        insert_pos = random.randint(latest_pos + 1, len(order))
        order.insert(insert_pos, ("inward", iw_idx))

    resolved: list[dict[str, Any]] = []
    for kind, idx in order:
        resolved.append(edges[idx] if kind == "edge" else inwards[idx])
    return resolved


def _assign_columns_for_order(
    ordered_slots: list[dict[str, Any]],
    side: str,
    occupied: list[tuple[int, int, np.ndarray]],
) -> list[tuple[int, int, np.ndarray]] | None:
    if side == "left":
        candidates = list(range(0, 9))
    else:
        candidates = list(range(9, GRID_W - SQ + 1))

    candidate_lists: list[list[int]] = []
    for slot in ordered_slots:
        cols = [c for c in candidates if int(c) != int(slot["out_col"])]
        random.shuffle(cols)
        candidate_lists.append(cols)

    placed: list[tuple[int, int, np.ndarray]] = []

    def backtrack(i: int) -> bool:
        if i == len(ordered_slots):
            return True

        slot = ordered_slots[i]
        row = int(slot["r0"])
        out_col = int(slot["out_col"])
        prev_col = None if not placed else int(placed[-1][1])

        for col in candidate_lists[i]:
            if int(col) == out_col:
                continue
            if prev_col is not None:
                if side == "left" and int(col) <= prev_col:
                    continue
                if side == "right" and int(col) >= prev_col:
                    continue

            if any(_rects_overlap(row, int(col), pr, pc) for pr, pc, _ in placed):
                continue
            if any(_rects_overlap(row, int(col), pr, pc) for pr, pc, _ in occupied):
                continue

            placed.append((row, int(col), slot["patch"]))
            if backtrack(i + 1):
                return True
            placed.pop()
        return False

    return placed if backtrack(0) else None


def _derive_input_positions(
    left_slots: list[dict[str, Any]],
    right_slots: list[dict[str, Any]],
) -> list[tuple[int, int, np.ndarray]] | None:
    first_side, second_side = ("left", "right")
    if len(right_slots) > len(left_slots):
        first_side, second_side = second_side, first_side

    side_slots = {"left": left_slots, "right": right_slots}

    for _ in range(INPUT_ORDER_RETRIES):
        first_order = _sample_processing_order(side_slots[first_side])
        second_order = _sample_processing_order(side_slots[second_side])
        if first_order is None or second_order is None:
            continue

        first_positions = _assign_columns_for_order(first_order, first_side, occupied=[])
        if first_positions is None:
            continue

        second_positions = _assign_columns_for_order(second_order, second_side, occupied=first_positions)
        if second_positions is None:
            continue

        all_positions = first_positions + second_positions
        if not _any_pairwise_overlap(all_positions):
            return all_positions
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    mix = _choose_side_color_mix(n_azure, n_red)
    if mix is None:
        return None
    n_left, n_right, left_azure, left_red, right_azure, right_red = mix

    left_slots = _build_side_slots("left", n_left)
    right_slots = _build_side_slots("right", n_right)
    if left_slots is None or right_slots is None:
        return None

    decorated = _decorate_slots(
        left_slots,
        right_slots,
        bg,
        left_azure,
        left_red,
        right_azure,
        right_red,
    )
    if decorated is None:
        return None
    output, left_slots, right_slots = decorated

    input_positions = _derive_input_positions(left_slots, right_slots)
    if input_positions is None:
        return None

    inp = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for r, c, patch in input_positions:
        paste(inp, patch, int(r), int(c))

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
