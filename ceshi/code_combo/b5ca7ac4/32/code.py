# Combo variant of b5ca7ac4 (index 32)
# combo_spec: D1/2+D5/0+D13/0
# Name: 边缘标记+位置分组+方向反转

"""
Design analysis
===============

variant_spec:
  preserved_structure:
    - Grid stays 22x22.
    - Objects are still scattered 5x5 framed squares with frame colors 2 and 8.
    - A square keeps its original row in the output.
    - Each destination side still has only two depths: edge column and one inward column.
    - Edge-vs-inward still depends on row-overlap with already placed edge squares.
  fused_changes:
    - D1/2: grouping is no longer frame-color-based. A square belongs to the upper group iff its center row
      is above the horizontal midline; otherwise it belongs to the lower group.
    - D13/0: direction is made explicit by border control pixels, but once grouping becomes spatial, marker-to-group
      binding can no longer use frame color. The clean resolution is: one marker lives in the upper half and controls
      the upper group; one marker lives in the lower half and controls the lower group.
    - D5/0: the direction indicated by a marker is reversed. A marker on the left border makes that spatial group
      stack to the right; a marker on the right border makes it stack to the left.
    - Processing order mirrors the true destination side:
        actual left  -> input columns ascending
        actual right -> input columns descending

behavior_trace:
  small_example:
    - Suppose an upper red square is at row 2, col 4 and an upper teal square is at row 4, col 13.
      Their row spans overlap. A lower red square is at row 14, col 6.
    - Original task:
        teal always goes left; red always goes right.
        So the upper red goes right, the upper teal goes left.
    - After D1/2 alone:
        both upper squares join the same upper spatial group, so both would go to the same side.
    - Fused task with an upper marker on the left border:
        the upper marker controls the upper group, but D5 reverses it, so the upper group actually stacks right.
        Therefore the upper squares are processed by descending input column: col 13 claims the outer right slot,
        col 4 shifts inward because it overlaps in rows.
    - If the lower marker sits on the right border, D5 flips that too, so the lower group stacks left.
  visible_difference:
    - Frame color is visibly decoupled from destination side because both halves mix red and teal frames.
    - Marker side matters, but only after learning the reversal: the pile appears on the opposite side from the marker.

complexity_calibration:
  - The task still requires three inferences in sequence: identify the two spatial groups, read the two control pixels,
    then apply the original two-layer stacking logic with side-dependent ordering.
  - Marker positions are binary left/right and repeated once per half, which keeps the instruction vocabulary small.
  - The generator forces both frame colors to appear in both halves, so color cannot collapse the task back to the seed.
  - The generator also varies which half ends up on which side, so the control pixels are not decorative.

implicit_assumption_audit_for_main:
  - Detecting 5x5 uniform-border squares with frame color 2 or 8 is still valid; the generator preserves that vocabulary.
  - The old split by `frame == 8` / `frame == 2` is invalid and must be replaced by a center-row test.
  - The old assumption "left marker means left output" is invalid under D5; the solver must explicitly reverse it.
  - Once grouping is spatial, marker color cannot identify a square group anymore. The solver instead assumes there is
    exactly one non-square border marker in the upper half and one in the lower half.
  - Sorting direction can no longer be hard-coded by color; it must be derived from the actual post-reversal side.
  - Control pixels must be copied to the output because they are explicit visible symbols in the input.

placement_legality:
  - Upper-group square starts are restricted to rows 0..8, which gives centers 2..10 and keeps them above the midline.
  - Lower-group square starts are restricted to rows 9..17, which gives centers 11..19 and keeps them below the midline.
  - Input squares never touch the left or right border, so border pixels are unambiguous control symbols.
  - Upper and lower groups always stack to opposite sides in a sample, which keeps the output legible and collision-free.
  - Upper marker rows are sampled from 0..8 and lower marker rows from 13..21; with the opposite-side output design,
    those bands keep markers away from any edge square that could otherwise overwrite them.

visual_observability:
  - A human can see that the top-half squares always move together and the bottom-half squares always move together.
  - The control pixel for a half is the lone border pixel in that same half, so the binding between marker and group is
    spatially local rather than hidden.
  - Across samples, the destination side is opposite the marker side, making the reversal learnable from 2-3 examples.
  - No ordering step uses invisible numeric properties; only row half, marker side, and input column are used.

visual_cleanliness_and_inference:
  - Output remains organized into two side bands, exactly like the seed task, so the images stay easy to parse.
  - Control pixels are isolated single cells, visually distinct from the 5x5 framed squares.
  - Marker colors are fixed but not semantically tied to frame colors; the generator mixes frame colors inside each half
    so the solver cannot mistake the marker color for a group-color key.

new_main_assumptions_checked_against_generator:
  - main() assumes there are exactly two border markers outside squares, one in each half; generator constructs that.
  - main() assumes top and bottom groups each have at least one square; generator uses 3-4 per half.
  - main() assumes every group needs at most two output columns; generator still builds edge rows plus inward rows only.
  - main() assumes the true stack side is the opposite of the marker side; generator validates every sample by rerunning main().
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
FRAME_COLORS = (AZURE, RED)
N_TOP_RANGE = (3, 4)
N_BOTTOM_RANGE = (3, 4)
MAX_INWARD = 2
INPUT_COL_RETRIES = 100
ROW_ASSIGN_RETRIES = 200
FRAME_ASSIGN_RETRIES = 120
GEN_RETRIES = 800
TOP_MARKER_ROWS = list(range(0, 9))
BOTTOM_MARKER_ROWS = list(range(13, GRID_H))


def _is_top_group(r0: int, height: int, square_size: int = SQ) -> bool:
    """Return True iff the square center lies above the horizontal midline."""
    center_row = int(r0) + square_size // 2
    return float(center_row) < float(height) / 2.0


def _marker_half(r: int, height: int) -> str:
    """Classify a control-pixel row as belonging to the upper or lower half."""
    return "top" if float(int(r)) < float(height) / 2.0 else "bottom"


def _reverse_side(side: str) -> str:
    """D5 reversal: the true stack side is opposite the marker side."""
    return "right" if side == "left" else "left"


def _target_cols(side: str, width: int) -> tuple[int, int]:
    """Return (edge_col, inward_col) for a leftward or rightward stack."""
    if side == "left":
        return 0, SQ
    return int(width - SQ), int(width - 2 * SQ)


def _sort_descending(side: str) -> bool:
    """Rightward stacks process squares nearer the right edge first."""
    return side == "right"


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


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a 5x5 framed square with a uniform 3x3 interior."""
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _detect_squares(g: np.ndarray) -> list[dict[str, object]]:
    """Detect all framed 5x5 squares while ignoring lone control pixels."""
    H, W = g.shape
    squares: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in FRAME_COLORS:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue
            key = (int(r0), int(c0))
            if key in seen:
                continue
            seen.add(key)
            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack framed 5x5 squares by top/bottom spatial grouping, using reversed edge markers as direction cues.

    Input structure:
    - bg is the most common color.
    - Red (2) and teal (8) framed 5x5 squares with colored interiors are scattered away from the left/right borders.
    - There are two lone border markers: one in the upper half and one in the lower half, each placed on the left or right edge.

    Rule:
    - A square belongs to the upper group if its center is above the horizontal midline; otherwise it belongs to the lower group.
    - The border marker in the same half chooses that group's nominal side, but the actual stack side is reversed:
      left-edge marker -> stack right, right-edge marker -> stack left.
    - Each square keeps its original row; leftward groups use input-column ascending order, rightward groups descending.
    - If a square's 5-row span overlaps an already placed edge square from the same group, it shifts one square-width inward instead.

    Key insight:
    - The grouping cue is spatial, not chromatic, and the control pixel must be read together with the reversal:
      each half follows its own marker, but piles up on the opposite side from that marker.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    squares = _detect_squares(g)
    square_mask = np.zeros((H, W), dtype=bool)
    for s in squares:
        r0 = int(s["r0"])
        c0 = int(s["c0"])
        square_mask[int(r0) : int(r0) + SQ, int(c0) : int(c0) + SQ] = True

    control_points = [
        (int(r), int(c), int(g[int(r), int(c)]))
        for r, c in np.argwhere((g != int(bg)) & ~square_mask)
        if int(c) == 0 or int(c) == int(W - 1)
    ]
    if len(control_points) != 2:
        raise ValueError(f"Expected exactly two control pixels, found {len(control_points)}")

    marker_by_half: dict[str, tuple[int, int, int]] = {}
    for r, c, color in control_points:
        half = _marker_half(r, H)
        if half in marker_by_half:
            raise ValueError(f"Expected one control pixel in {half} half, found multiple")
        marker_by_half[half] = (int(r), int(c), int(color))
    if set(marker_by_half) != {"top", "bottom"}:
        raise ValueError("Expected one upper-half marker and one lower-half marker")

    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    group_specs = [
        ("top", [s for s in squares if _is_top_group(int(s["r0"]), H)]),
        ("bottom", [s for s in squares if not _is_top_group(int(s["r0"]), H)]),
    ]
    for half, group in group_specs:
        _marker_r, marker_c, _marker_color = marker_by_half[half]
        marker_side = "left" if int(marker_c) == 0 else "right"
        actual_side = _reverse_side(marker_side)
        edge_col, inward_col = _target_cols(actual_side, W)
        ordered_group = sorted(
            group,
            key=lambda s: int(s["c0"]),
            reverse=_sort_descending(actual_side),
        )
        placed_at_edge: list[tuple[int, int]] = []
        for s in ordered_group:
            r0 = int(s["r0"])
            r1 = int(r0 + SQ - 1)
            has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
            dest_c = int(inward_col) if has_overlap else int(edge_col)
            paste(out, s["patch"], r0, dest_c, transparent=bg)
            if dest_c == int(edge_col):
                placed_at_edge.append((r0, r1))

    for r, c, color in control_points:
        out[int(r), int(c)] = int(color)

    return out


def _candidate_rows(go_top: bool) -> list[int]:
    """All legal square start rows for one spatial group."""
    return [r for r in range(GRID_H - SQ + 1) if _is_top_group(r, GRID_H) == go_top]


TOP_ROWS = _candidate_rows(True)
BOTTOM_ROWS = _candidate_rows(False)


def _assign_output_rows(n: int, row_pool: list[int]) -> tuple[list[int], list[int]] | None:
    """Pick edge and inward row starts for one group inside its legal half."""
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
    """Assign interior colors globally uniquely within each frame family."""
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
    side: str,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render one spatial group's squares onto the chosen output side."""
    edge_col, inward_col = _target_cols(side, output.shape[1])
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
    side: str,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign random interior columns that reproduce the requested side-dependent stacking."""
    edges = [(r, oc, patch) for (r, oc, is_edge, patch) in sq_list if is_edge]
    inwards = [(r, oc, patch) for (r, oc, is_edge, patch) in sq_list if not is_edge]
    inward_deps = [
        [i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)]
        for (ir, _, _) in inwards
    ]

    n = len(sq_list)
    col_range = list(range(1, GRID_W - SQ))
    if len(col_range) < n:
        return None

    descending = _sort_descending(side)
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

        chosen_cols = sorted(random.sample(col_range, n), reverse=descending)
        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, oc, patch) in enumerate(ordered_squares):
            ic = int(chosen_cols[i])
            if ic == int(oc):
                ok = False
                break
            result.append((int(r), int(ic), patch))
        if ok and not _any_pairwise_overlap(result):
            return result
    return None


def _choose_marker_rows() -> tuple[int, int] | None:
    """Choose safe marker rows in disjoint vertical bands."""
    top_candidates = TOP_MARKER_ROWS[:]
    bottom_candidates = BOTTOM_MARKER_ROWS[:]
    if not top_candidates or not bottom_candidates:
        return None
    return int(random.choice(top_candidates)), int(random.choice(bottom_candidates))


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid fused-variant input grid."""
    bg = random.choice([c for c in range(10) if c not in FRAME_COLORS])
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

    top_actual_side = random.choice(["left", "right"])
    bottom_actual_side = "right" if top_actual_side == "left" else "left"
    top_marker_side = _reverse_side(top_actual_side)
    bottom_marker_side = _reverse_side(bottom_actual_side)

    output = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    top_out = _place_group_squares(
        output,
        top_edge_rows,
        top_inward_rows,
        top_frames,
        top_inner,
        top_actual_side,
    )
    bottom_out = _place_group_squares(
        output,
        bottom_edge_rows,
        bottom_inward_rows,
        bottom_frames,
        bottom_inner,
        bottom_actual_side,
    )

    marker_rows = _choose_marker_rows()
    if marker_rows is None:
        return None
    top_marker_row, bottom_marker_row = marker_rows
    control_points = [
        (int(top_marker_row), 0 if top_marker_side == "left" else GRID_W - 1, AZURE),
        (int(bottom_marker_row), 0 if bottom_marker_side == "left" else GRID_W - 1, RED),
    ]
    for r, c, color in control_points:
        output[int(r), int(c)] = int(color)

    top_input = _derive_input_positions(top_out, top_actual_side)
    bottom_input = _derive_input_positions(bottom_out, bottom_actual_side)
    if top_input is None or bottom_input is None:
        return None

    all_positions = top_input + bottom_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for r, c, patch in all_positions:
        paste(inp, patch, int(r), int(c))
    for r, c, color in control_points:
        inp[int(r), int(c)] = int(color)

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
