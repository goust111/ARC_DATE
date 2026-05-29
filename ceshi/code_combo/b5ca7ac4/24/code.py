# Combo variant of b5ca7ac4 (index 24)
# combo_spec: D1/2+D15/0
# Name: 噪声像素+位置分组

"""
Design analysis
===============

fusion_summary:
  preserved_core:
    - Grid stays 22x22.
    - Real objects are still 5x5 bordered squares with frame color 2 or 8 and a uniform 3x3 interior.
    - Motion still has the same two-layer stacking pattern on each side:
      first claim the outer edge, later row-overlapping objects on that side move one square inward.
    - Input-column order still decides the edge-vs-inward winner within each side.
  fused_changes:
    - D15 adds isolated singleton noise pixels on background cells; these must be erased before object detection.
    - D1 changes grouping from frame color to spatial position:
      center above the horizontal midline -> left stack, otherwise -> right stack.
    - On a 22x22 grid with 5x5 squares, center_row = r0 + 2, so legal left-group starts are 0..8
      and legal right-group starts are 9..17.

behavior_trace:
  concrete_example:
    - Suppose a red frame starts at row 2, col 14 and a teal frame starts at row 13, col 3.
    - In the source task, the red frame would go right and the teal frame would go left because color decides.
    - In this combo, singleton specks are removed first; then the red frame is grouped left because its center
      is above the midline, and the teal frame is grouped right because its center is below the midline.
    - If another top-half frame with overlapping rows has a smaller input column, it keeps the left edge slot and
      the later one shifts to col 5 exactly as in the base task.
  visible_difference:
    - Red frames can land on the left and teal frames can land on the right.
    - Extra lone pixels disappear entirely, so the solver must both denoise and regroup by position.

complexity_calibration:
  - The task is not simplified into a plain top/bottom sort: each side still needs multiple frames and repeated
    overlap cases so column order remains relevant.
  - Noise is restricted to 2-5 isolated pixels, which is enough to visibly falsify the shortcut
    "every non-background pixel belongs to a square" without cluttering the grid.
  - The generator forces both spatial groups to contain both frame colors, guaranteeing that color is visibly
    insufficient as a grouping rule.

implicit_assumption_audit_for_main:
  - `bg = get_bg(g)` assumes background is dominant. The generator keeps only 6-8 squares plus sparse noise, so this holds.
  - The denoiser assumes each distractor is a singleton in its 8-neighborhood. The generator places noise only on cells whose
    surrounding 3x3 block is pure background.
  - The square detector assumes target objects are complete 5x5 windows with a uniform border in {2,8} and a uniform interior
    not equal to the frame or background. The generator preserves exactly that.
  - The old split by `frame == 8` versus `frame == 2` is invalid after D1, so grouping must instead depend on `r0 + 2 < H/2`.
  - The edge bookkeeping assumes at most two output columns per side. The generator still creates only edge and one-step inward placements.

placement_legality:
  - Top-group output rows are drawn only from starts 0..8; bottom-group output rows only from 9..17.
  - Within each group, edge rows are pairwise non-overlapping and inward rows each overlap at least one edge row
    while remaining pairwise non-overlapping themselves.
  - Input columns are sampled afterward and rejected unless all 5x5 frames are pairwise non-overlapping.
  - Noise pixels are added only after all frames are placed and only onto isolated background cells, so they cannot touch
    each other or any frame.

visual_observability_and_tidiness:
  - Every generated example contains both red and teal frames in the upper half and in the lower half, so at least one
    frame of each color violates the original color-based destination rule.
  - Noise disappears in the output but never touches frames, keeping the scene readable.
  - Output remains two clean vertical stack zones on the left and right, which makes the positional grouping easy to compare
    against the input.

anti_degeneracy_intent:
  - A sample counts as meaningfully fused only if the spatial-group output differs from a color-group output on the same
    denoised frames. The generator's forced color mixing across both halves is what makes that difference systematic.
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
FRAME_ASSIGN_RETRIES = 120
GEN_RETRIES = 700
NOISE_RANGE = (2, 5)


def _goes_left(r0: int, height: int, square_size: int = SQ) -> bool:
    """Return True iff the square center lies above the grid's horizontal midline."""
    center_row = int(r0) + square_size // 2
    return float(center_row) < float(height) / 2.0


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
            if _rects_overlap(
                int(positions[i][0]),
                int(positions[i][1]),
                int(positions[j][0]),
                int(positions[j][1]),
            ):
                return True
    return False


def _remove_isolated_noise(grid: np.ndarray, bg: int) -> np.ndarray:
    """Erase non-bg pixels that are singletons in their 8-neighborhood."""
    src = np.array(grid, dtype=int)
    cleaned = src.copy()
    H, W = src.shape
    for r, c in np.argwhere(src != bg):
        r0 = max(0, int(r) - 1)
        r1 = min(H - 1, int(r) + 1)
        c0 = max(0, int(c) - 1)
        c1 = min(W - 1, int(c) + 1)
        if int(np.count_nonzero(src[r0 : r1 + 1, c0 : c1 + 1] != bg)) == 1:
            cleaned[int(r), int(c)] = bg
    return cleaned


def _detect_squares(grid: np.ndarray, bg: int) -> list[dict[str, int | np.ndarray]]:
    """Detect complete 5x5 framed squares after noise removal."""
    H, W = grid.shape
    squares: list[dict[str, int | np.ndarray]] = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(grid, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {RED, AZURE}:
                continue

            inner = patch[1:-1, 1:-1]
            if int(inner.min()) != int(inner.max()):
                continue
            inner_col = int(inner[0, 0])
            if inner_col in {bg, frame_col}:
                continue

            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def _render_grouped_output(
    squares: list[dict[str, int | np.ndarray]],
    bg: int,
    shape: tuple[int, int],
) -> np.ndarray:
    """Apply the spatial grouping rule to already detected squares."""
    H, W = shape
    out = np.full(shape, int(bg), dtype=int)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    left_group = sorted(
        [s for s in squares if _goes_left(int(s["r0"]), H)],
        key=lambda s: int(s["c0"]),
    )
    placed_at_edge: list[tuple[int, int]] = []
    for s in left_group:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, np.array(s["patch"], dtype=int), r0, int(dest_c), transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    right_group = sorted(
        [s for s in squares if not _goes_left(int(s["r0"]), H)],
        key=lambda s: -int(s["c0"]),
    )
    placed_at_edge = []
    for s in right_group:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, np.array(s["patch"], dtype=int), r0, int(dest_c), transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack real 5x5 framed squares after stripping singleton noise and grouping by height.

    Input structure:
    - bg is the most common color.
    - True objects are 5x5 red (2) or teal (8) bordered squares with a uniform 3x3 interior.
    - Extra isolated one-cell colored specks may appear on background cells; they are distractors.

    Rule:
    - First erase isolated non-bg singleton dots, then detect only complete 5x5 framed squares.
    - A square whose center lies above the horizontal midline moves to the left stack; the others move to the right stack.
    - Each square keeps its original row.
    - Left-group squares are processed by increasing input column and right-group squares by decreasing input column; row-overlap with an already edge-placed square pushes the later one one square-width inward.

    Key insight:
    - Color no longer decides the destination side, and not every colored pixel is meaningful.
      The solver must ignore lone specks, recover the full frames, then use vertical position plus the old column-order/row-overlap logic to determine the final stacks.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cleaned = _remove_isolated_noise(g, bg)
    squares = _detect_squares(cleaned, bg)
    return _render_grouped_output(squares, bg, g.shape)


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a 5x5 framed square with a uniform 3x3 interior."""
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


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
                edge_rows.append(int(r))
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
            inward_rows.append(int(r))
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
    """Render one spatial group's squares and return placement metadata."""
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    idx = 0
    for r in edge_rows:
        patch = _make_patch(frames[idx], inner_colors[idx])
        paste(output, patch, int(r), int(edge_col))
        placed.append((int(r), int(edge_col), True, patch))
        idx += 1
    for r in inward_rows:
        patch = _make_patch(frames[idx], inner_colors[idx])
        paste(output, patch, int(r), int(inward_col))
        placed.append((int(r), int(inward_col), False, patch))
        idx += 1
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign input columns that reproduce the target edge/inward placements."""
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]
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
            ic = int(chosen_cols[i])
            if ic == int(oc):
                ok = False
                break
            result.append((int(r), ic, patch))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _can_place_noise(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    """Check that a noise dot would stay isolated from all other non-bg pixels."""
    if int(grid[r, c]) != bg:
        return False
    H, W = grid.shape
    r0 = max(0, int(r) - 1)
    r1 = min(H - 1, int(r) + 1)
    c0 = max(0, int(c) - 1)
    c1 = min(W - 1, int(c) + 1)
    return not bool((grid[r0 : r1 + 1, c0 : c1 + 1] != bg).any())


def _add_noise(inp: np.ndarray, bg: int) -> bool:
    """Add 2-5 isolated singleton distractors, always including red and teal examples."""
    n_noise = random.randint(*NOISE_RANGE)
    extra_colors = [c for c in range(10) if c not in {bg, RED, AZURE}]
    forced_colors = [RED, AZURE]
    while len(forced_colors) < n_noise:
        pool = [RED, AZURE] + extra_colors
        forced_colors.append(random.choice(pool))

    for color in forced_colors:
        candidates = [
            (r, c)
            for r in range(GRID_H)
            for c in range(GRID_W)
            if _can_place_noise(inp, r, c, bg)
        ]
        if not candidates:
            return False
        r, c = random.choice(candidates)
        inp[int(r), int(c)] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid fused-variant input grid."""
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
        paste(inp, patch, int(r), int(c))

    if not _add_noise(inp, bg):
        return None

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
