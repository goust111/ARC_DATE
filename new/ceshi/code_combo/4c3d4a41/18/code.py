# Combo variant of 4c3d4a41 (index 18)
# combo_spec: D8/0+D9/2+D12/2
# Name: 全随机泛化测试

"""
design_analysis:
  fused_goal:
    summary:
      - The original task had three fixed crutches: exactly four protrusions, fixed slot-column lengths
        [3,4,4,3], and a permanent left-bar/right-frame layout on an 8x20 board.
      - This combo removes all three supports at once: the bar now has 2-5 protrusions, each frame column
        has an independent random length in 1..6, and the frame/bar can appear at different absolute positions.
      - The intended solver must therefore detect the two structural objects by shape, count the protrusions,
        match them to the colored columns, and then apply the same insertion logic without relying on constants.

  concrete_behavior_and_trace:
    example_trace:
      - Example parameters: 3 protrusions, slot lengths [6,2,5], colored counts [4,1,3], protrusion heights
        [4,2,3], frame at rows 1..11 cols 9..17, loose bar elsewhere.
      - Input: three active frame columns appear at cols 11,13,15. Their visible stacks are grey+color runs
        of heights 6,2,5 starting from the frame interior top, while the separate bar has width 5 and three
        upward teeth at its even local columns.
      - Output: the same bar is translated into the frame so its base lands one row above the interior floor and
        its three tooth columns align with cols 11,13,15. Then each column's original colored run is repainted
        immediately above that tooth, clipped if the tooth leaves less headroom than the input count.
      - Difference from the original: nothing about the geometry can be read from fixed absolute coordinates or a
        fixed "4 columns / pyramid lengths" template anymore; only the structural correspondence survives.

  complexity_calibration:
    why_this_is_not_too_simple:
      - Keeping one active column per protrusion preserves a clean one-to-one mapping, so the task is still
        inferable by eye from a few examples.
      - Column lengths vary independently but stay in the small discrete range 1..6, which makes the change
        visually obvious without creating an unbounded zoo of column shapes.
      - The frame keeps the same semi-open silhouette and the bar keeps the same comb silhouette, so humans still
        have strong object cues even though positions and counts vary.
    ambiguity_controls:
      - Colors remain one solid color per column and are sampled without repetition, preventing ambiguous
        many-to-one column matching.
      - Protrusions stay evenly spaced every two columns, matching the frame's active columns, so alignment is
        still a single translation problem instead of a combinatorial matching problem.

  implicit_assumption_audit:
    original_assumptions_that_break:
      - "Largest grey component = frame, second-largest = bar" is unsafe here because with only 2 protrusions the
        bar can be smaller than a long internal grey cap.
      - Fixed destination columns {11,13,15,17}, fixed bar width 7, fixed dx=+10, and fixed row anchors all fail
        once protrusion count and positions are randomized.
      - Fixed slot lengths [3,4,4,3] are gone, so the solver cannot infer how many colored cells survive from any
        template; it must read each input column directly.
    new_assumptions_in_main:
      - The frame is the unique grey component whose bbox is a hollow rectangle with a two-cell opening in its
        left wall and no grey interior pixels belonging to that same component.
      - The bar is the unique non-frame grey component whose bbox bottom row is completely filled and whose tooth
        columns are exactly the evenly spaced columns 0,2,4,... across the bar width.
      - Inside the frame, colored pixels occur only in the active columns and each such column corresponds to one
        tooth after a single horizontal translation.
    generator_guarantees:
      - The generator enforces those shapes exactly, keeps frame and loose bar as separate components with at least
        one blank cell between their bboxes, and validates `main(input) == expected` before returning a sample.

  placement_legality:
    guarantees:
      - Grid size is fixed to 20x26 so both a 11-row frame and a detached bar can move around without clipping,
        including genuinely variable above/below layouts instead of row-locked placements.
      - For `n` protrusions, frame width is `2*n+3`, bar width is `2*n-1`, active frame columns are
        `frame_c0 + 2 + 2*i`, and the inserted bar origin is `frame_c0 + 2`; this preserves one blank column
        between active columns and one blank interior margin on both sides.
      - The frame opening rows are the two rows immediately above and at the inserted bar base level, matching the
        original "semi-open frame" affordance.

  visual_observability_and_cleanliness:
    human_view:
      - The frame and bar remain visually distinct grey objects: one is a hollow container with a side gap, the
        other is a comb with a solid base. Random placement does not create overlap or merge them.
      - The active columns are sparse and vertically clean: each column is monochrome color under a grey cap, with
        empty spacer columns between them.
      - No hidden color arithmetic is used. Every output color comes from the same input column, so the rule is a
        visible geometric relocation rather than a symbolic remapping.

  degeneration_check_plan:
    metric:
      - After implementation, measure the tuple `(protrusion_count, slot_lengths, frame_origin, bar_origin)` on
        many generated samples.
      - The combo is non-degenerate only if those tuples actually spread across multiple counts, multiple length
        patterns, and multiple placements rather than collapsing to the original `(4, [3,4,4,3], fixed, fixed)`.
      - Generator calls intentionally rotate through protrusion counts and the four relative-layout modes before
        falling back to unconstrained resampling, so branch coverage does not silently collapse onto left/right cases.
"""

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, paste


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _local_mask(pts: np.ndarray):
    b = _bbox(pts)
    r0, c0, r1, c1 = b
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[int(r) - r0, int(c) - c0] = True
    return b, mask


def _is_frame_component(pts: np.ndarray) -> bool:
    bbox, mask = _local_mask(pts)
    _r0, _c0, r1, c1 = bbox
    h = r1 - bbox[0] + 1
    w = c1 - bbox[1] + 1
    if h < 7 or w < 7:
        return False
    if not mask[0, :].all() or not mask[-1, :].all():
        return False
    if not mask[:, -1].all():
        return False
    left_gap_rows = np.flatnonzero(~mask[:, 0])
    if left_gap_rows.size != 2 or left_gap_rows[1] != left_gap_rows[0] + 1:
        return False
    if left_gap_rows[0] == 0 or left_gap_rows[-1] == h - 1:
        return False
    if mask[1:-1, 1:-1].any():
        return False
    return True


def _is_bar_component(pts: np.ndarray) -> bool:
    bbox, mask = _local_mask(pts)
    _r0, _c0, r1, c1 = bbox
    h = r1 - bbox[0] + 1
    w = c1 - bbox[1] + 1
    if h < 2 or w < 3 or w > 9 or w % 2 == 0:
        return False
    if not mask[-1, :].all():
        return False
    proj_cols = np.flatnonzero(mask[:-1, :].any(axis=0))
    expected = np.arange(0, w, 2)
    if proj_cols.size < 2 or proj_cols.size > 5:
        return False
    if not np.array_equal(proj_cols, expected):
        return False
    if mask[:-1, 1::2].any():
        return False
    return True


def _extract_components(g: np.ndarray, grey: int):
    frame = None
    bar = None
    for pts in _cc4(g == grey):
        area = len(pts)
        if _is_frame_component(pts):
            if frame is None or area > frame[0]:
                frame = (area, _bbox(pts), pts)
        elif _is_bar_component(pts):
            if bar is None or area > bar[0]:
                bar = (area, _bbox(pts), pts)
    return frame, bar


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a detached grey comb into a semi-open grey frame, align every tooth with a colored frame column, and restack the colors above it.

    Input structure:
    - The background contains one hollow grey frame with a two-cell opening on its left wall, plus one separate grey
      horizontal bar whose full bottom edge is solid and whose evenly spaced upward teeth define the active columns.
    - Inside the frame, the active columns are sparse vertical runs made from a grey cap above a single repeated
      color; the number of such columns matches the number of teeth and can vary from instance to instance.

    Rule:
    - Identify the frame by its semi-open rectangular outline and the bar by its solid-base comb shape.
    - Translate the bar so its base lands one row above the frame interior floor and its tooth columns line up with
      the colored columns inside the frame.
    - For each aligned column, copy that column's original colored run and place it as a vertical stack ending
      immediately above the tooth, clipping only if the tooth leaves insufficient headroom.

    Key insight:
    - Absolute geometry is no longer reliable: the count of teeth, the column lengths, and the frame/bar positions
      all vary, so the invariant is purely structural matching between a comb-shaped key and sparse colored lock columns.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

    bg = get_bg(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    frame_info, bar_info = _extract_components(g, grey)
    if frame_info is None or bar_info is None:
        return g.copy()

    _frame_area, frame_bbox, _frame_pts = frame_info
    _bar_area, _bar_bbox, bar_pts = bar_info
    sr0, sc0, sr1, sc1 = frame_bbox
    interior_top = sr0 + 1

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = max(int(r) for r, _ in bar_pts)
    target_base_row = sr1 - 2
    dy = target_base_row - bar_base_row

    col_min_row = {}
    for r, c in bar_pts:
        c = int(c)
        r = int(r)
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)

    dst_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )

    dx = None
    if src_proj_cols and len(src_proj_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_proj_cols, dst_cols)]
        if len(set(deltas)) == 1:
            dx = deltas[0]
    if dx is None:
        best = (-1, 0)
        dst_set = set(dst_cols)
        for cand_dx in range(-w, w + 1):
            score = sum((c + cand_dx) in dst_set for c in src_proj_cols)
            if score > best[0]:
                best = (score, cand_dx)
        dx = best[1]

    placed = []
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col = {}
    for rr, cc in placed:
        placed_by_col.setdefault(cc, []).append(rr)

    for col in dst_cols:
        if col not in placed_by_col:
            continue
        tooth_top = min(placed_by_col[col])
        desired_bottom = tooth_top - 1
        if desired_bottom < interior_top:
            continue

        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]

        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = desired_bottom - len(colors) + 1
        for i, color in enumerate(colors):
            out[start_row + i, col] = color

    return out


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

GRID_H = 20
GRID_W = 26
FRAME_H = 11
BAR_H = 4
OPEN_ROWS = (FRAME_H - 4, FRAME_H - 3)
LENGTH_RANGE = (1, 6)
HEIGHT_RANGE = (2, 4)
PROTRUSION_RANGE = (2, 5)
LAYOUT_MODES = ("left", "right", "above", "below")
MAX_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 200
_GENERATOR_CALL_COUNT = 0


def _frame_patch(protrusion_count: int) -> np.ndarray:
    width = 2 * protrusion_count + 3
    patch = np.zeros((FRAME_H, width), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[list(OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    width = 2 * len(heights) - 1
    patch = np.zeros((BAR_H, width), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip(range(0, width, 2), heights):
        patch[BAR_H - height : BAR_H, local_c] = GREY
    return patch


def _bar_output_origin(frame_r0: int, frame_c0: int) -> tuple[int, int]:
    return frame_r0 + FRAME_H - BAR_H - 2, frame_c0 + 2


def _slot_cols(frame_c0: int, protrusion_count: int) -> list[int]:
    return [frame_c0 + 2 + 2 * i for i in range(protrusion_count)]


def _paint_input_column(
    grid: np.ndarray, top_row: int, col: int, length: int, color: int, count: int
) -> None:
    cap = length - count
    grid[top_row : top_row + cap, col] = GREY
    grid[top_row + cap : top_row + length, col] = color


def _paint_output_column(
    grid: np.ndarray,
    top_row: int,
    col: int,
    length: int,
    color: int,
    count: int,
    tooth_top: int,
) -> None:
    cap = length - count
    grid[top_row : top_row + cap, col] = GREY
    desired_bottom = tooth_top - 1
    visible = min(count, max(0, desired_bottom - top_row + 1))
    if visible <= 0:
        return
    start_row = desired_bottom - visible + 1
    grid[start_row : desired_bottom + 1, col] = color


def _rects_overlap(a, b, pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _eligible_layout_modes(frame_w: int, bar_w: int) -> list[str]:
    modes = []
    if frame_w + 1 + bar_w <= GRID_W:
        modes.extend(["left", "right"])
    if FRAME_H + 1 + BAR_H <= GRID_H:
        modes.extend(["above", "below"])
    return modes


def _sample_layout(
    rng: np.random.Generator, frame_w: int, bar_w: int, preferred_mode: str | None = None
) -> tuple[int, int, int, int] | None:
    modes = _eligible_layout_modes(frame_w, bar_w)
    if preferred_mode is not None:
        modes = [preferred_mode] if preferred_mode in modes else []
    if not modes:
        return None

    for _ in range(LAYOUT_ATTEMPTS):
        mode = str(rng.choice(modes))
        if mode == "left":
            frame_c0 = int(rng.integers(bar_w + 1, GRID_W - frame_w + 1))
            bar_c0 = int(rng.integers(0, frame_c0 - bar_w))
            frame_r0 = int(rng.integers(0, GRID_H - FRAME_H + 1))
            bar_r0 = int(rng.integers(0, GRID_H - BAR_H + 1))
        elif mode == "right":
            frame_c0 = int(rng.integers(0, GRID_W - frame_w - bar_w))
            bar_c0 = int(rng.integers(frame_c0 + frame_w + 1, GRID_W - bar_w + 1))
            frame_r0 = int(rng.integers(0, GRID_H - FRAME_H + 1))
            bar_r0 = int(rng.integers(0, GRID_H - BAR_H + 1))
        elif mode == "above":
            frame_r0 = int(rng.integers(BAR_H + 1, GRID_H - FRAME_H + 1))
            bar_r0 = int(rng.integers(0, frame_r0 - BAR_H))
            frame_c0 = int(rng.integers(0, GRID_W - frame_w + 1))
            bar_c0 = int(rng.integers(0, GRID_W - bar_w + 1))
        else:
            frame_r0 = int(rng.integers(0, GRID_H - FRAME_H - BAR_H))
            bar_r0 = int(rng.integers(frame_r0 + FRAME_H + 1, GRID_H - BAR_H + 1))
            frame_c0 = int(rng.integers(0, GRID_W - frame_w + 1))
            bar_c0 = int(rng.integers(0, GRID_W - bar_w + 1))

        frame_rect = (frame_r0, frame_c0, frame_r0 + FRAME_H - 1, frame_c0 + frame_w - 1)
        bar_rect = (bar_r0, bar_c0, bar_r0 + BAR_H - 1, bar_c0 + bar_w - 1)
        if not _rects_overlap(frame_rect, bar_rect, pad=1):
            return frame_r0, frame_c0, bar_r0, bar_c0
    return None


def _sample_params(
    rng: np.random.Generator,
    protrusion_count: int | None = None,
    layout_mode: str | None = None,
):
    if protrusion_count is None:
        protrusion_count = int(rng.integers(PROTRUSION_RANGE[0], PROTRUSION_RANGE[1] + 1))
    heights = [int(rng.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1)) for _ in range(protrusion_count)]
    lengths = [int(rng.integers(LENGTH_RANGE[0], LENGTH_RANGE[1] + 1)) for _ in range(protrusion_count)]
    counts = [int(rng.integers(1, length + 1)) for length in lengths]
    colors = [int(v) for v in rng.choice(PALETTE, size=protrusion_count, replace=False)]
    frame_w = 2 * protrusion_count + 3
    bar_w = 2 * protrusion_count - 1
    layout = _sample_layout(rng, frame_w, bar_w, preferred_mode=layout_mode)
    if layout is None:
        return None
    frame_r0, frame_c0, bar_r0, bar_c0 = layout
    return protrusion_count, heights, lengths, counts, colors, frame_r0, frame_c0, bar_r0, bar_c0


def _build_case(
    protrusion_count: int,
    heights: list[int],
    lengths: list[int],
    counts: list[int],
    colors: list[int],
    frame_r0: int,
    frame_c0: int,
    bar_r0: int,
    bar_c0: int,
    *,
    output_mode: bool,
) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    frame_patch = _frame_patch(protrusion_count)
    bar_patch = _bar_patch(heights)
    paste(grid, frame_patch, frame_r0, frame_c0)
    if output_mode:
        target_r0, target_c0 = _bar_output_origin(frame_r0, frame_c0)
        paste(grid, bar_patch, target_r0, target_c0)
    else:
        paste(grid, bar_patch, bar_r0, bar_c0)

    top_row = frame_r0 + 1
    out_bar_r0, out_bar_c0 = _bar_output_origin(frame_r0, frame_c0)
    for i, (col, length, count, color, height) in enumerate(
        zip(_slot_cols(frame_c0, protrusion_count), lengths, counts, colors, heights)
    ):
        if output_mode:
            tooth_top = out_bar_r0 + BAR_H - height
            _paint_output_column(grid, top_row, col, length, color, count, tooth_top)
        else:
            _paint_input_column(grid, top_row, col, length, color, count)
    return grid


def _gen_attempt(
    rng: np.random.Generator,
    protrusion_count: int | None = None,
    layout_mode: str | None = None,
) -> np.ndarray | None:
    params = _sample_params(rng, protrusion_count=protrusion_count, layout_mode=layout_mode)
    if params is None:
        return None
    (
        protrusion_count,
        heights,
        lengths,
        counts,
        colors,
        frame_r0,
        frame_c0,
        bar_r0,
        bar_c0,
    ) = params

    grid = _build_case(
        protrusion_count,
        heights,
        lengths,
        counts,
        colors,
        frame_r0,
        frame_c0,
        bar_r0,
        bar_c0,
        output_mode=False,
    )
    expected = _build_case(
        protrusion_count,
        heights,
        lengths,
        counts,
        colors,
        frame_r0,
        frame_c0,
        bar_r0,
        bar_c0,
        output_mode=True,
    )

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GENERATOR_CALL_COUNT
    rng = np.random.default_rng()
    protrusion_values = list(range(PROTRUSION_RANGE[0], PROTRUSION_RANGE[1] + 1))
    base_idx = _GENERATOR_CALL_COUNT
    _GENERATOR_CALL_COUNT += 1
    target_n = protrusion_values[base_idx % len(protrusion_values)]
    target_mode = LAYOUT_MODES[(base_idx // len(protrusion_values)) % len(LAYOUT_MODES)]
    for _ in range(MAX_ATTEMPTS):
        for _ in range(80):
            grid = _gen_attempt(rng, protrusion_count=target_n, layout_mode=target_mode)
            if grid is not None:
                return grid
        target_n = int(rng.choice(protrusion_values))
        target_mode = str(rng.choice(LAYOUT_MODES))
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
