# Combo variant of b5ca7ac4 (index 37)
# combo_spec: D2/1+D17/2
# Name: 列位置排序+最近边缘堆叠

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the fused variant:
# - Concrete behavior: every 5x5 framed square now chooses its destination side from
#   geometry alone. A square at input col 2 is closer to the left target column 0 than
#   to the right target column W-5, so it joins the left stack; a square at col 14 is
#   closer to the right target, so it joins the right stack. After that split, the left
#   stack is processed by input column ascending, the right stack by input column
#   descending. Row overlap still decides edge vs inward placement on that side.
# - Difference from the source task: color no longer determines side. The important
#   visible change is that teal and red frames can both appear on both output edges.
# - Complexity control: stacking depth stays at two columns per side, exactly as in the
#   source task, so the variant remains readable. The generator enforces color mixing on
#   both sides to prevent accidental collapse back to the original color-to-side rule.
# - Implicit assumptions to preserve:
#   1. Solver assumes every relevant object is a non-overlapping 5x5 framed square.
#   2. Solver assumes no three mutually row-overlapping squares land on the same side,
#      otherwise two inward placements could collide. The generator preserves the old
#      "edge rows disjoint, inward rows disjoint, inward overlaps an edge" structure.
#   3. Solver uses the square's left column to measure distance to target edge columns
#      0 and W-5; with W=22 and SQ=5 this yields a clean split: cols 0..8 go left,
#      cols 9..17 go right, with no tie case.
# - Placement legality: the generator first builds a legal output, then maps left-edge /
#   left-inward / right-inward / right-edge squares back to four fixed input lanes
#   (1, 6, 11, 16). These lanes preserve nearest-edge classification, preserve the
#   processing order needed for stacking, and keep different side/type bands from
#   overlapping horizontally.


SQ = 5
AZURE = 8
RED = 2
GRID_H, GRID_W = 22, 22
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
GEN_RETRIES = 500

LEFT_EDGE_IN_COL = 1
LEFT_INWARD_IN_COL = 6
RIGHT_INWARD_IN_COL = 11
RIGHT_EDGE_IN_COL = 16


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack scattered 5x5 framed squares against whichever horizontal edge is nearer.

    Input structure:
    - The background is the most frequent color.
    - The foreground consists of non-overlapping 5x5 bordered squares with frame color
      teal (8) or red (2), each with a solid 3x3 interior of a different color.

    Rule:
    - Keep each square's row fixed.
    - Send each square to the nearer output edge: col 0 if it is closer to the left
      target, or col W-5 if it is closer to the right target.
    - On the left side, process squares by original input column from left to right; on
      the right side, process them from right to left.
    - Within each side, a square that row-overlaps an already placed edge square is
      shifted one square-width inward instead of taking the edge slot.

    Key insight:
    - The decisive grouping is spatial, not color-based: first infer the square's nearer
      edge from its input column, then use that side's column-order priority to decide
      which overlapping square stays on the outer edge.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    def target_side(c0: int) -> str:
        left_dist = int(c0)
        right_dist = int((w - SQ) - c0)
        return "left" if left_dist < right_dist else "right"

    squares: list[dict[str, object]] = []
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
            squares.append(
                {
                    "frame": frame,
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch,
                    "side": target_side(int(c0)),
                }
            )

    left_squares = sorted(
        [s for s in squares if s["side"] == "left"],
        key=lambda s: int(s["c0"]),
    )
    right_squares = sorted(
        [s for s in squares if s["side"] == "right"],
        key=lambda s: -int(s["c0"]),
    )

    placed_left_edge_rows: list[tuple[int, int]] = []
    for s in left_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_left_edge_rows)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_left_edge_rows.append((r0, r1))

    placed_right_edge_rows: list[tuple[int, int]] = []
    for s in right_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_right_edge_rows)
        dest_c = int(w - 2 * SQ) if has_overlap else int(w - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(w - SQ):
            placed_right_edge_rows.append((r0, r1))

    return out


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), frame_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
    return patch


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _choose_side_split(n_azure: int, n_red: int) -> tuple[int, int] | None:
    total = int(n_azure + n_red)
    options: list[tuple[int, int]] = []
    for azure_left in range(1, n_azure):
        for red_left in range(1, n_red):
            n_left = azure_left + red_left
            n_right = total - n_left
            if 3 <= n_left <= 6 and 3 <= n_right <= 6:
                options.append((azure_left, red_left))
    if not options:
        return None
    return random.choice(options)


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    valid_inward_counts = [
        inward
        for inward in range(1, min(MAX_INWARD, n - 1) + 1)
        if (n - inward) <= 4
    ]
    if not valid_inward_counts:
        return None

    n_inward = random.choice(valid_inward_counts)
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


def _emit_side(
    output: np.ndarray,
    frame_colors: list[int],
    edge_rows: list[int],
    inward_rows: list[int],
    inner_color_pool: dict[int, list[int]],
    side: str,
) -> list[dict[str, object]]:
    random.shuffle(frame_colors)
    squares: list[dict[str, object]] = []
    rows_and_flags = [(r, True) for r in edge_rows] + [(r, False) for r in inward_rows]
    random.shuffle(rows_and_flags)

    for (r0, is_edge), frame in zip(rows_and_flags, frame_colors):
        inner = inner_color_pool[frame].pop()
        patch = _make_patch(frame, inner)
        if side == "left":
            out_col = 0 if is_edge else SQ
            input_col = LEFT_EDGE_IN_COL if is_edge else LEFT_INWARD_IN_COL
        else:
            out_col = GRID_W - SQ if is_edge else GRID_W - 2 * SQ
            input_col = RIGHT_EDGE_IN_COL if is_edge else RIGHT_INWARD_IN_COL
        paste(output, patch, int(r0), int(out_col))
        squares.append(
            {
                "frame": int(frame),
                "r0": int(r0),
                "c0": int(input_col),
                "out_c": int(out_col),
                "side": side,
                "is_edge": bool(is_edge),
                "patch": patch,
            }
        )
    return squares


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    split = _choose_side_split(n_azure, n_red)
    if split is None:
        return None
    azure_left, red_left = split
    azure_right = n_azure - azure_left
    red_right = n_red - red_left

    left_rows = _assign_output_rows(azure_left + red_left)
    right_rows = _assign_output_rows(azure_right + red_right)
    if left_rows is None or right_rows is None:
        return None
    left_edge_rows, left_inward_rows = left_rows
    right_edge_rows, right_inward_rows = right_rows

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None

    inner_color_pool = {
        AZURE: random.sample(avail_azure, n_azure),
        RED: random.sample(avail_red, n_red),
    }

    output = np.full((GRID_H, GRID_W), bg, dtype=int)

    left_frames = [AZURE] * azure_left + [RED] * red_left
    right_frames = [AZURE] * azure_right + [RED] * red_right

    left_squares = _emit_side(
        output,
        left_frames,
        left_edge_rows,
        left_inward_rows,
        inner_color_pool,
        "left",
    )
    right_squares = _emit_side(
        output,
        right_frames,
        right_edge_rows,
        right_inward_rows,
        inner_color_pool,
        "right",
    )

    all_squares = left_squares + right_squares
    input_positions = [(int(s["r0"]), int(s["c0"]), s["patch"]) for s in all_squares]
    if _any_pairwise_overlap(input_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r0, c0, patch in input_positions:
        paste(inp, patch, int(r0), int(c0))

    if np.array_equal(main(inp), output):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
