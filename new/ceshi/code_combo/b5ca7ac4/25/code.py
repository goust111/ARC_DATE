# Combo variant of b5ca7ac4 (index 25)
# combo_spec: D3/3+D16/1
# Name: 背景色框架+三组分类

from __future__ import annotations

import random
import sys

import numpy as np

SOLVER_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS_DIR not in sys.path:
    sys.path.append(SOLVER_UTILS_DIR)

from arc_utils import crop_bbox, get_bg, paste

# Design analysis
# ---------------
# variant_spec:
#   preserved_structure:
#     grid: still 22x22
#     source_objects: every real object still occupies a 5x5 footprint with a 3x3 uniform interior
#     row_motion: every detected object keeps its original top row after moving
#   D3_change:
#     visible_groups:
#       teal_8: left band with original edge/inward arbitration
#       red_2: right band with mirrored edge/inward arbitration
#       green_3: a third visible frame color routed to the centered anchor
#   D16_change:
#     hidden_group:
#       bg_frame: some 5x5 objects use the background color as their border
#       appearance: they look like isolated 3x3 solid blocks because the border disappears into the canvas
#       destination: these hidden objects join the same centered group as green frames
#   concrete_output_layout:
#     left band: cols 0 and 5
#     center band: col 8 only because (22 - 5) // 2 = 8
#     right band: cols 17 and 12
#
# behavior_trace:
#   original_flow:
#     - scan 5x5 windows and keep only teal or red uniform-border frames
#     - teal is processed by input column ascending; row overlap decides col 0 vs col 5
#     - red is processed by input column descending; row overlap decides col 17 vs col 12
#   fused_flow:
#     - the same scan now also accepts green frames and background-colored borders
#     - green frames and bg-framed objects are merged into one center-bound family
#     - side groups still use visible horizontal proximity to decide which overlapping square claims the true edge
#   small_example:
#     - suppose bg is yellow 4
#     - a teal frame at row 3 / col 1 and another teal frame at row 5 / col 13 still become left-edge then left-inward
#     - a green frame at row 10 / col 2 moves to row 10 / col 8
#     - a hidden object with outer border 4 and inner 3x3 maroon 9 at row 14 / col 15 looks like a plain 3x3 maroon
#       block in the input, but it also moves to row 14 / col 8
#   difference_from_single_changes:
#     - compared with the original task, there is now a third destination class
#     - compared with plain D3/3, some center-bound objects have no visible frame color, so the solver must infer that
#       a 3x3 block surrounded by one cell of background is really another 5x5 framed object
#
# complexity_calibration:
#   - The fused rule is still low-entropy: the output has only three landing zones.
#   - Every sample contains at least one visible green center object and at least one hidden background-framed center
#     object, so the extra concept is repeatedly supported instead of appearing as a one-off oddity.
#   - The hidden objects are not routed to a separate fourth zone. Making four destinations on a 22x22 board would
#     create avoidable visual clutter; merging green and hidden objects into one center group keeps the pattern legible.
#
# hidden_assumptions_review:
#   - Fixed-size 5x5 window scanning still works, but the old assumption "frame color is in {2, 8}" is broken.
#     The new detector must accept {2, 3, 8, bg}.
#   - The original solver only checked "inner differs from frame somewhere". That is too weak for hidden objects,
#     because arbitrary background windows would pass too often. The fused solver strengthens detection to require a
#     uniform 3x3 interior whose color differs from the uniform border color.
#   - A single center anchor cannot safely host row-overlapping objects. The generator therefore guarantees that every
#     green or hidden center-bound object has a row-span disjoint from every other center-bound object.
#   - Center objects at col 8 would horizontally collide with side inward bands at cols 5 and 12 if their rows
#     overlapped. The generator therefore forbids center rows from overlapping any side-inward row.
#   - Because bg-framed objects visually collapse to 3x3 blocks, the generator must avoid accidental extra 3x3 islands;
#     it does this by constructing only the intended objects and validating with the solver.
#
# placement_legality:
#   - Input placement uses non-overlapping 5x5 rectangles.
#   - Output placement is collision-free because:
#       * teal inward rows are pairwise disjoint and only overlap teal edge rows
#       * red inward rows are pairwise disjoint and only overlap red edge rows
#       * all center rows are pairwise disjoint and avoid every inward row
#   - Output-first generation is preserved, then random input columns are derived and validated by checking
#     main(input) == output.
#
# observability_and_cleanliness:
#   - Humans can still see the left-band and right-band rules from visible teal/red frames.
#   - The new center rule is inferable because visible green frames and visually frameless 3x3 blocks both migrate to
#     the same middle band, demonstrating that "looks like a 3x3 block" is not a separate object type.
#   - No ordering depends on invisible facts such as color id ranking; only the side groups use visibly meaningful
#     horizontal ordering, and the center group has no hidden tie-breaker at all.
#   - The generator forces hidden objects to appear in every sample, so "ignore the plain 3x3 blocks" is falsified by
#     the examples.


GRID_H, GRID_W = 22, 22
SQ = 5
TEAL, RED, GREEN = 8, 2, 3
LEFT_COL = 0
LEFT_INNER_COL = SQ
CENTER_COL = (GRID_W - SQ) // 2
RIGHT_INNER_COL = GRID_W - 2 * SQ
RIGHT_COL = GRID_W - SQ
N_TEAL_RANGE = (2, 3)
N_GREEN_RANGE = (1, 2)
N_RED_RANGE = (2, 3)
N_HIDDEN_MAX = 2
MAX_TOTAL_OBJECTS = 8
MAX_INWARD = 2
INPUT_COL_RETRIES = 80
GEN_RETRIES = 900


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


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


def _scan_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    allowed_frames = {int(bg), TEAL, GREEN, RED}
    for r0 in range(grid.shape[0] - SQ + 1):
        for c0 in range(grid.shape[1] - SQ + 1):
            patch = crop_bbox(grid, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in allowed_frames:
                continue

            inner = patch[1:-1, 1:-1]
            if int(inner.min()) != int(inner.max()):
                continue
            inner_col = int(inner[0, 0])
            if inner_col == frame_col:
                continue

            objects.append(
                {
                    "frame": frame_col,
                    "inner": inner_col,
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch.copy(),
                }
            )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack three visible frame groups plus background-framed hidden squares into left, center, and right bands.

    Input structure:
    - bg is the most common color in the grid.
    - Every true object still occupies a 5×5 footprint with a uniform-colored 3×3 interior.
    - Teal (8), green (3), and red (2) objects have visible 1-cell borders; some additional objects use bg itself
      as the border, so they visually appear as standalone 3×3 blocks.

    Rule:
    - Teal squares keep their row and stack on the left: the first non-overlapping teal row-span claims col 0, and
      any later teal square whose row-span overlaps an already edge-placed teal square moves to col 5 instead.
    - Red squares mirror that on the right: the first non-overlapping red row-span claims col W−5, and later
      overlapping red squares move to col W−10.
    - Green squares and bg-framed hidden squares form one shared third group and all move to the centered anchor at
      col (W−5)//2.

    Key insight: The solver must recognize that some apparent 3×3 color blocks are actually 5×5 objects whose border
    matches the background; those hidden objects join the same middle destination group as the visible green frames.
    """
    g = np.array(grid, dtype=int)
    _, W = g.shape
    bg = int(get_bg(g))

    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    objects = _scan_objects(g, bg)

    teal_objects = sorted([obj for obj in objects if int(obj["frame"]) == TEAL], key=lambda obj: int(obj["c0"]))
    placed_left_rows: list[tuple[int, int]] = []
    for obj in teal_objects:
        r0 = int(obj["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_left_rows)
        dest_c = LEFT_INNER_COL if has_overlap else LEFT_COL
        paste(out, np.array(obj["patch"], dtype=int), r0, dest_c, transparent=bg)
        if dest_c == LEFT_COL:
            placed_left_rows.append((r0, r1))

    center_objects = sorted(
        [obj for obj in objects if int(obj["frame"]) in {GREEN, bg}],
        key=lambda obj: (int(obj["r0"]), int(obj["c0"]), 0 if int(obj["frame"]) == GREEN else 1),
    )
    for obj in center_objects:
        paste(out, np.array(obj["patch"], dtype=int), int(obj["r0"]), CENTER_COL, transparent=bg)

    red_objects = sorted([obj for obj in objects if int(obj["frame"]) == RED], key=lambda obj: -int(obj["c0"]))
    placed_right_rows: list[tuple[int, int]] = []
    for obj in red_objects:
        r0 = int(obj["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_right_rows)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, np.array(obj["patch"], dtype=int), r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_right_rows.append((r0, r1))

    return out


def _assign_side_rows(n: int) -> tuple[list[int], list[int]] | None:
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, er) for er in edge_rows):
            edge_rows.append(r)
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
        inward_rows.append(r)
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None
    return edge_rows, inward_rows


def _assign_center_rows(n: int, forbidden_rows: list[int]) -> list[int] | None:
    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    rows: list[int] = []
    for r in all_rows:
        if any(_rows_overlap(r, fr) for fr in forbidden_rows):
            continue
        if any(_rows_overlap(r, rr) for rr in rows):
            continue
        rows.append(r)
        if len(rows) == n:
            return rows
    return None


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    colors: list[int],
    frame_color: int,
    edge_col: int,
    inward_col: int,
    bg: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        patch = _make_patch(frame_color, colors[i])
        paste(output, patch, r, edge_col, transparent=bg)
        placed.append((r, edge_col, True, patch))
    for i, r in enumerate(inward_rows):
        patch = _make_patch(frame_color, colors[len(edge_rows) + i])
        paste(output, patch, r, inward_col, transparent=bg)
        placed.append((r, inward_col, False, patch))
    return placed


def _place_center_squares(
    output: np.ndarray,
    rows: list[int],
    green_colors: list[int],
    hidden_colors: list[int],
) -> list[tuple[int, int, np.ndarray]]:
    placed: list[tuple[int, int, np.ndarray]] = []
    row_iter = iter(rows)
    bg = int(get_bg(output))
    for color in green_colors:
        r = int(next(row_iter))
        patch = _make_patch(GREEN, color)
        paste(output, patch, r, CENTER_COL, transparent=bg)
        placed.append((r, CENTER_COL, patch))
    for color in hidden_colors:
        r = int(next(row_iter))
        patch = _make_patch(bg, color)
        paste(output, patch, r, CENTER_COL, transparent=bg)
        placed.append((r, CENTER_COL, patch))
    return placed


def _derive_side_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, oc, patch) for (r, oc, is_edge, patch) in sq_list if is_edge]
    inwards = [(r, oc, patch) for (r, oc, is_edge, patch) in sq_list if not is_edge]

    inward_deps = [[i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)] for (ir, _, _) in inwards]
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
        for inward_idx in inward_indices:
            latest_pos = max(
                (
                    pos
                    for dep in inward_deps[inward_idx]
                    for pos, (kind, idx) in enumerate(order)
                    if kind == "edge" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", inward_idx))

        ordered = [edges[idx] if kind == "edge" else inwards[idx] for kind, idx in order]

        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, out_c, patch) in enumerate(ordered):
            in_c = int(chosen_cols[i])
            if in_c == out_c:
                ok = False
                break
            result.append((int(r), in_c, patch))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _derive_center_input_positions(
    sq_list: list[tuple[int, int, np.ndarray]],
) -> list[tuple[int, int, np.ndarray]] | None:
    col_range = [c for c in range(0, GRID_W - SQ + 1) if c != CENTER_COL]
    if len(col_range) < len(sq_list):
        return None

    for _ in range(INPUT_COL_RETRIES):
        chosen_cols = random.sample(col_range, len(sq_list))
        result = [(int(r), int(chosen_cols[i]), patch) for i, (r, _, patch) in enumerate(sq_list)]
        if not _any_pairwise_overlap(result):
            return result
    return None


def _pick_unique_colors(count: int, forbidden: set[int], used: set[int]) -> list[int] | None:
    pool = [c for c in range(10) if c not in forbidden and c not in used]
    if len(pool) < count:
        return None
    chosen = random.sample(pool, count)
    used.update(chosen)
    return chosen


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {TEAL, GREEN, RED}])
    n_teal = random.randint(*N_TEAL_RANGE)
    n_green = random.randint(*N_GREEN_RANGE)
    n_red = random.randint(*N_RED_RANGE)
    max_hidden = min(N_HIDDEN_MAX, MAX_TOTAL_OBJECTS - n_teal - n_green - n_red)
    if max_hidden < 1:
        return None
    n_hidden = random.randint(1, max_hidden)

    teal_assign = _assign_side_rows(n_teal)
    if teal_assign is None:
        return None
    red_assign = _assign_side_rows(n_red)
    if red_assign is None:
        return None
    teal_edge_rows, teal_inward_rows = teal_assign
    red_edge_rows, red_inward_rows = red_assign

    center_rows = _assign_center_rows(n_green + n_hidden, forbidden_rows=teal_inward_rows + red_inward_rows)
    if center_rows is None:
        return None
    random.shuffle(center_rows)

    used_colors: set[int] = set()
    teal_colors = _pick_unique_colors(n_teal, {bg, TEAL}, used_colors)
    if teal_colors is None:
        return None
    green_colors = _pick_unique_colors(n_green, {bg, GREEN}, used_colors)
    if green_colors is None:
        return None
    red_colors = _pick_unique_colors(n_red, {bg, RED}, used_colors)
    if red_colors is None:
        return None
    hidden_colors = _pick_unique_colors(n_hidden, {bg}, used_colors)
    if hidden_colors is None:
        return None

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    teal_out = _place_side_squares(
        output,
        teal_edge_rows,
        teal_inward_rows,
        teal_colors,
        TEAL,
        edge_col=LEFT_COL,
        inward_col=LEFT_INNER_COL,
        bg=bg,
    )
    center_out = _place_center_squares(output, center_rows, green_colors, hidden_colors)
    red_out = _place_side_squares(
        output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        edge_col=RIGHT_COL,
        inward_col=RIGHT_INNER_COL,
        bg=bg,
    )

    teal_input = _derive_side_input_positions(teal_out, ascending=True)
    if teal_input is None:
        return None
    red_input = _derive_side_input_positions(red_out, ascending=False)
    if red_input is None:
        return None

    center_input = None
    for _ in range(INPUT_COL_RETRIES):
        candidate = _derive_center_input_positions(center_out)
        if candidate is None:
            return None
        all_positions = teal_input + candidate + red_input
        if not _any_pairwise_overlap(all_positions):
            center_input = candidate
            break
    if center_input is None:
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in teal_input + center_input + red_input:
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
