# Combo variant of b5ca7ac4 (index 10)
# combo_spec: D6/0+D11/1
# Name: 扩展近邻检测+间隙堆叠
#
# Design analysis
# a) Concrete behavior trace:
#    Take teal rows [r, r+6, r+1] in processing order.
#    - Original task:
#      first -> edge, second -> still edge (gap 1 row is not overlap),
#      third -> inward only once because it overlaps the first edge square.
#    - Fused task here:
#      first -> edge, second -> lane 1 because gap 1 row now counts as a blocker,
#      third -> lane 2 because lane 0 is blocked by overlap and lane 1 is blocked by
#      the near-neighbor row at r+6. This is the intended synergy: the wider blocker
#      test creates extra shifts, and each shift leaves a visible 1-cell gutter.
# b) Complexity calibration:
#    The rule still uses a small, human-recoverable parameter set:
#    blocker if vertical gap < 2 rows, lane offset = 6 columns. The output remains
#    framed-square stacking, but samples visibly differ from the original through
#    one-column gutters and occasional third lanes.
# c) Hidden assumptions broken from the seed solver:
#    - "Only edge squares matter" is no longer valid; every occupied lane can block.
#    - "Overlap means row intersection only" changes to "intersection or gap 0/1".
#    - "At most two depths" no longer holds logically, so the solver now does
#      first-fit lane assignment across all lanes that fit in the grid.
# d) Placement legality:
#    Generator uses only in-bounds lane columns (left: 0/6/12, right: 17/11/5),
#    checks all output rectangles globally, and backtracks on input columns so the
#    scattered source squares never overlap.
# e) Visual observability:
#    Every generated instance contains at least one near-only trigger, so the
#    D11/1 change is not latent. The 1-column gutter makes the D6/0 change visible.
# f) Visual cleanliness:
#    Objects remain plain 5x5 framed squares with unique interiors per frame color.
#    Extra depth is capped by grid width, and global overlap checks prevent muddled
#    cross-side collisions.
# g) New solver assumptions audited against the generator:
#    The solver assumes detected objects are genuine 5x5 red/teal frames and that
#    the required lane count fits the width. The generator guarantees both by
#    construction and validates each sample with main(input) == output.

from __future__ import annotations

import os
import random
import sys

import numpy as np

SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import crop_bbox, get_bg, paste


GRID_H, GRID_W = 22, 22
SQ = 5
STEP = 6
NEAR_GAP_ROWS = 2
MAX_LANES = 1 + (GRID_W - SQ) // STEP
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
SIDE_BUILD_RETRIES = 200
INPUT_ASSIGN_RETRIES = 60
GEN_RETRIES = 500


def _scan_squares(grid: np.ndarray) -> list[dict[str, object]]:
    squares: list[dict[str, object]] = []
    h, w = grid.shape
    for r0 in range(h - SQ + 1):
        for c0 in range(w - SQ + 1):
            patch = crop_bbox(grid, (r0, c0, r0 + SQ - 1, c0 + SQ - 1))
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


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return max(int(r_a), int(r_b)) <= min(int(r_a) + SQ - 1, int(r_b) + SQ - 1)


def _rows_need_offset(r_a: int, r_b: int) -> bool:
    a0 = int(r_a)
    b0 = int(r_b)
    a1 = a0 + SQ - 1
    b1 = b0 + SQ - 1
    gap = max(max(a0, b0) - min(a1, b1) - 1, 0)
    return gap < NEAR_GAP_ROWS


def _lane_col(width: int, left_side: bool, depth: int) -> int:
    return int(depth * STEP if left_side else width - SQ - depth * STEP)


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, object]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _stack_rows(rows: list[int]) -> tuple[list[int], int] | None:
    lanes: list[list[int]] = [[] for _ in range(MAX_LANES)]
    depths: list[int] = []
    near_only_hits = 0
    for row in rows:
        depth = 0
        while depth < MAX_LANES:
            blockers = [prev for prev in lanes[depth] if _rows_need_offset(row, prev)]
            if not blockers:
                break
            if any(not _rows_overlap(row, prev) for prev in blockers):
                near_only_hits += 1
            depth += 1
        if depth >= MAX_LANES:
            return None
        lanes[depth].append(int(row))
        depths.append(depth)
    return depths, near_only_hits


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), frame_color, dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _build_side_plan(
    n: int,
    prefix_rows: list[int],
    candidate_rows: list[int],
    *,
    require_depth2: bool,
) -> tuple[list[int], list[int], int] | None:
    for _ in range(SIDE_BUILD_RETRIES):
        rows = [int(r) for r in prefix_rows]
        stacked = _stack_rows(rows)
        if stacked is None:
            continue
        depths, near_only_hits = stacked

        while len(rows) < n:
            found = False
            for _ in range(80):
                trial_row = int(random.choice(candidate_rows))
                trial_rows = rows + [trial_row]
                trial_stacked = _stack_rows(trial_rows)
                if trial_stacked is None:
                    continue
                rows = trial_rows
                depths, near_only_hits = trial_stacked
                found = True
                break
            if not found:
                break

        if len(rows) != n:
            continue
        if require_depth2 and max(depths) < 2:
            continue
        if near_only_hits < 1:
            continue
        return rows, depths, near_only_hits
    return None


def _make_side_entries(
    rows: list[int],
    depths: list[int],
    colors: list[int],
    frame_color: int,
    *,
    left_side: bool,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for row, depth, color in zip(rows, depths, colors):
        entries.append(
            {
                "row": int(row),
                "depth": int(depth),
                "out_col": _lane_col(GRID_W, left_side, int(depth)),
                "patch": _make_patch(frame_color, color),
            }
        )
    return entries


def _search_input_cols(
    rows: list[int],
    out_cols: list[int],
    ascending: bool,
    blocked: list[tuple[int, int, object]],
) -> list[int] | None:
    n = len(rows)
    limit = GRID_W - SQ
    for _ in range(INPUT_ASSIGN_RETRIES):
        chosen: list[int] = []

        def rec(i: int) -> bool:
            if i == n:
                return True

            if ascending:
                low = 0 if i == 0 else chosen[-1] + 1
                high = limit - (n - i - 1)
                domain = list(range(low, high + 1))
            else:
                high = limit if i == 0 else chosen[-1] - 1
                low = n - i - 1
                domain = list(range(low, high + 1))

            random.shuffle(domain)
            for col in domain:
                if col == int(out_cols[i]):
                    continue
                if i > 0:
                    if ascending and col <= chosen[-1]:
                        continue
                    if (not ascending) and col >= chosen[-1]:
                        continue
                if any(_rects_overlap(rows[i], col, rows[j], chosen[j]) for j in range(i)):
                    continue
                if any(_rects_overlap(rows[i], col, br, bc) for br, bc, _ in blocked):
                    continue
                chosen.append(int(col))
                if rec(i + 1):
                    return True
                chosen.pop()
            return False

        if rec(0):
            return chosen
    return None


def _derive_input_positions(
    azure_entries: list[dict[str, object]],
    red_entries: list[dict[str, object]],
) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]] | None:
    azure_rows = [int(e["row"]) for e in azure_entries]
    azure_out_cols = [int(e["out_col"]) for e in azure_entries]
    azure_cols = _search_input_cols(azure_rows, azure_out_cols, True, [])
    if azure_cols is None:
        return None
    azure_positions = [
        (int(e["row"]), int(col), e["patch"]) for e, col in zip(azure_entries, azure_cols)
    ]

    red_rows = [int(e["row"]) for e in red_entries]
    red_out_cols = [int(e["out_col"]) for e in red_entries]
    red_cols = _search_input_cols(red_rows, red_out_cols, False, azure_positions)
    if red_cols is None:
        return None
    red_positions = [
        (int(e["row"]), int(col), e["patch"]) for e, col in zip(red_entries, red_cols)
    ]

    if _any_pairwise_overlap(azure_positions + red_positions):
        return None
    return azure_positions, red_positions


def _generate_input_and_output() -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_anchor = random.randint(0, 3)
    azure_plan = _build_side_plan(
        n_azure,
        [azure_anchor, azure_anchor + 6, azure_anchor + 1],
        list(range(0, 10)),
        require_depth2=True,
    )
    if azure_plan is None:
        return None
    azure_rows, azure_depths, azure_near_hits = azure_plan

    red_anchor = random.randint(10, 11)
    red_plan = _build_side_plan(
        n_red,
        [red_anchor, red_anchor + 6],
        list(range(10, 18)),
        require_depth2=False,
    )
    if red_plan is None:
        return None
    red_rows, red_depths, red_near_hits = red_plan

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    azure_entries = _make_side_entries(azure_rows, azure_depths, azure_colors, AZURE, left_side=True)
    red_entries = _make_side_entries(red_rows, red_depths, red_colors, RED, left_side=False)

    output_positions = [
        (int(e["row"]), int(e["out_col"]), e["patch"]) for e in azure_entries + red_entries
    ]
    if _any_pairwise_overlap(output_positions):
        return None

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    for row, col, patch in output_positions:
        paste(output, patch, row, col)

    input_positions = _derive_input_positions(azure_entries, red_entries)
    if input_positions is None:
        return None
    azure_input, red_input = input_positions

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for row, col, patch in azure_input + red_input:
        paste(inp, patch, row, col)

    if not np.array_equal(main(inp), output):
        return None

    stats = {
        "azure_near_hits": int(azure_near_hits),
        "red_near_hits": int(red_near_hits),
        "azure_max_depth": int(max(azure_depths)),
        "red_max_depth": int(max(red_depths)),
    }
    return inp, output, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather the red and teal framed 5x5 squares against their matching side edges.

    Input structure:
    - The background is the grid's most common color.
    - Non-background objects are 5x5 hollow frames with border color 8 (teal) or 2 (red),
      each enclosing a solid 3x3 interior color.

    Rule:
    - Teal frames are processed from smaller input column to larger and stack from the left edge;
      red frames are processed from larger input column to smaller and stack from the right edge.
    - Each frame keeps its original top row.
    - A later frame cannot use a lane if an earlier same-color frame in that lane either overlaps
      its 5-row span or sits within 1 blank row of it vertically; it shifts one lane inward instead.
    - Each inward shift moves by 6 columns, so adjacent lanes are separated by a 1-column gap.

    Key insight:
    - The output is determined by edge-directed ordering plus first-fit lane assignment under the
      expanded vertical-neighborhood test, not just by exact row overlap.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)
    max_lanes = 1 + (w - SQ) // STEP

    def place_group(group: list[dict[str, object]], *, left_side: bool) -> None:
        lanes: list[list[int]] = [[] for _ in range(max_lanes)]
        for square in group:
            row = int(square["r0"])
            depth = 0
            while depth < max_lanes and any(_rows_need_offset(row, prev) for prev in lanes[depth]):
                depth += 1
            if depth >= max_lanes:
                raise ValueError("stack requires more lanes than fit in the grid")
            col = _lane_col(w, left_side, depth)
            if col < 0 or col + SQ > w:
                raise ValueError("computed lane is out of bounds")
            paste(out, square["patch"], row, col, transparent=bg)
            lanes[depth].append(row)

    squares = _scan_squares(g)
    azure_squares = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    red_squares = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))
    place_group(azure_squares, left_side=True)
    place_group(red_squares, left_side=False)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _generate_input_and_output()
        if built is not None:
            return built[0]
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
