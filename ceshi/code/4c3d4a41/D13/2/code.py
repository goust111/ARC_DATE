# Variant of 4c3d4a41
# Dimension: D13 — 干扰项与噪声 (meta)
# Change: 无干扰项，网格中只有框架、杆和彩色像素三类对象... → 第二个未使用的杆

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, paste
from arc_utils import get_bg as _bg_color


# Design analysis
# ----------------
# a) Concrete behavior trace
#    - The original task has one frame on the right and one bar on the left. The bar is moved into the frame,
#      aligning its upward projections with the four colored slot columns.
#    - This variant adds a second grey bar as a distractor. The frame still has exactly four colored slot columns,
#      so only the bar whose projection pattern can be translated onto those four columns should be used.
#    - Example of the changed decision point:
#      original: the second-largest grey component is always the only bar, so the solver can move it directly.
#      variant: there are now two bar components. One may have 3 or 5 projections, or 4 projections with wrong gaps.
#      The solver must compare each bar's projection columns against the colored columns in the frame and select
#      the bar with an exact count-and-spacing match before applying the original translation/stacking logic.
#
# b) Complexity calibration
#    - The distractor is not just a duplicate placed elsewhere; it differs by a small, visible structural property:
#      wrong projection count or wrong spacing. That keeps the D13 change easy to notice while preserving the
#      original core mechanic.
#    - The generator randomizes whether the correct bar is the left or right candidate, so a trivial positional
#      heuristic cannot solve the task.
#    - Distractor families are drawn from a short menu to keep the visual vocabulary small and learnable.
#
# c) Implicit assumptions in the original solver that break here
#    - "grey_comps[1] is the bar" is no longer safe: there are now three grey components.
#    - "the chosen bar can be found by size ranking" is no longer safe, because the distractor may be comparable
#      in area or even have more projections.
#    - The horizontal shift computation itself remains valid, but only after a compatible bar has been selected.
#
# d) Placement legality
#    - The canvas is widened from 20 to 28 columns so that two 7-cell bars can sit side-by-side with a background
#      gap and still remain disconnected from the frame.
#    - The frame keeps its original 8x11 geometry and the bars keep their original 4x7 footprint, so no object
#      overlaps or leaves the grid.
#
# e) Visual observability
#    - Across a few examples, a human can see: there is always one frame, always two bars, and only the bar whose
#      bumps line up with the four colored columns is moved into the frame. The unused bar disappears in the output.
#    - Both distractor branches are visually explicit: one has the wrong number of bumps, the other has four bumps
#      but non-uniform spacing, so the difference is inferable without hidden metadata.
#
# f) Visual cleanliness
#    - The two bars are confined to the left half of the grid and never overlap each other or the frame.
#    - Inside the frame, the output remains identical in style to the original task, so the added complexity is
#      localized to the bar-selection step rather than cluttering the transformed region.
#
# g) Color inference
#    - Colors are unchanged by the variant. The same four slot colors are stacked above the inserted bar, so there
#      is no new color mapping burden.
#
# h) Generator guarantees for the new solver
#    - Exactly one non-frame grey component has four visible projections whose columns are a constant-shift match
#      to the four colored slot columns.
#    - The other grey component has either a wrong projection count or wrong inter-projection gaps.
#    - The correct bar position is randomized between the two input slots.
#
# i) Post-change assumption audit for main()
#    - Component ranking: the largest grey component is assumed to be the frame; the generator preserves this.
#    - Destination extraction: colored pixels relevant to alignment are assumed to lie inside the frame bbox only.
#    - Candidate scoring: the intended bar is assumed to be the unique exact translation match to the frame columns.
#    - Restacking loop: each destination column is assumed to have one monochrome input stack, exactly as in the
#      original task; the generator keeps that invariant unchanged.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _projection_signature(pts) -> tuple[int, list[int]]:
    base_row = max(int(r) for r, _ in pts)
    col_min_row: dict[int, int] = {}
    for r, c in pts:
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)
    proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < base_row)
    return base_row, proj_cols


def _best_alignment(src_cols: list[int], dst_cols: list[int], span: int) -> tuple[bool, int, int]:
    exact = False
    dx = 0
    if src_cols and len(src_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_cols, dst_cols)]
        if len(set(deltas)) == 1:
            exact = True
            dx = deltas[0]

    best_overlap = -1
    best_dx = 0
    dst_set = set(dst_cols)
    for cand_dx in range(-span, span + 1):
        overlap = sum((c + cand_dx) in dst_set for c in src_cols)
        if overlap > best_overlap:
            best_overlap = overlap
            best_dx = cand_dx

    return exact, dx if exact else best_dx, best_overlap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the grey bar whose projection pattern fits the frame, then place it and restack the colors.

    Input structure:
    - bg plus a grey structural color forming three disconnected components: one frame on the right and two candidate bars on the left.
    - Colored pixels already sit inside the frame and identify the destination columns that the chosen bar must match.

    Rule:
    - Keep the frame, erase the original bars, and inspect both bars' upward projection columns.
    - Choose the only bar whose projection count and spacing can be translated onto the colored frame columns.
    - Move that bar into the frame so its base sits one cell above the frame interior bottom, then stack each colored column immediately above the matched bar column.

    Key insight:
    - The extra bar is a distractor: compatibility with the frame is determined by the visible projection pattern, not by bar position or size.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    grey_comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(g == grey)]
    if len(grey_comps) < 3:
        return g.copy()
    grey_comps.sort(key=lambda t: (-t[0], t[1]))

    (_, slot_bbox, _), *bar_comps = grey_comps
    sr0, sc0, sr1, sc1 = slot_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    dst_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )
    if not dst_cols:
        return out

    best_key = None
    chosen = None
    for _size, _bbox_unused, bar_pts in bar_comps:
        bar_base_row, src_proj_cols = _projection_signature(bar_pts)
        exact, dx, overlap = _best_alignment(src_proj_cols, dst_cols, w)
        key = (
            1 if exact else 0,
            overlap,
            -abs(len(src_proj_cols) - len(dst_cols)),
            -(max(src_proj_cols) - min(src_proj_cols)) if src_proj_cols else -10**9,
        )
        if best_key is None or key > best_key:
            best_key = key
            chosen = (bar_pts, bar_base_row, dx)

    if chosen is None:
        return out

    bar_pts, bar_base_row, dx = chosen
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    placed: list[tuple[int, int]] = []
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(cc, []).append(rr)

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1
    for col in dst_cols:
        if col not in placed_by_col:
            continue
        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
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

        start_row = max(desired_bottom - len(colors) + 1, interior_top)
        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = colv

    return out


# --------------- generator ---------------

BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 28

FRAME_R0, FRAME_C0 = 0, 17
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

SLOT_COLS = [19, 21, 23, 25]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1

BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_INPUT_C0S = [1, 9]
BAR_OUTPUT_C0 = SLOT_COLS[0]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (2, 5)
MAX_ATTEMPTS = 400

DISTRACTOR_PATTERNS = (
    ("count3", (0, 2, 6)),
    ("count5", (0, 1, 3, 5, 6)),
    ("spacing4", (0, 1, 4, 6)),
    ("spacing4", (0, 2, 5, 6)),
)


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(proj_cols: list[int] | tuple[int, ...], heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip(proj_cols, heights):
        patch[BAR_H - height : BAR_H, int(local_c)] = GREY
    return patch


def _paint_slot_column(
    grid: np.ndarray,
    col: int,
    length: int,
    color: int,
    count: int,
    height: int | None,
) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    if height is None:
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
        return

    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible > 0:
        row_end = BAR_BASE_ABS_ROW - height
        grid[row_end - visible + 1 : row_end + 1, col] = color


def _sample_target_heights(rng: np.random.Generator) -> list[int]:
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(BAR_LOCAL_PROJ_COLS) - 1))
    if all(h == 4 for h in heights):
        heights[int(rng.integers(0, len(heights)))] = 3
    return heights


def _sample_distractor(rng: np.random.Generator) -> tuple[str, list[int], list[int]]:
    family, proj_cols = DISTRACTOR_PATTERNS[int(rng.integers(0, len(DISTRACTOR_PATTERNS)))]
    heights = [int(rng.integers(2, 5)) for _ in proj_cols]
    return family, list(proj_cols), heights


def _build_input(
    target_heights: list[int],
    colors: list[int],
    colored_counts: list[int],
    target_pos_index: int,
    distractor_proj_cols: list[int],
    distractor_heights: list[int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(BAR_LOCAL_PROJ_COLS, target_heights), BAR_R0, BAR_INPUT_C0S[target_pos_index])
    other_index = 1 - target_pos_index
    paste(grid, _bar_patch(distractor_proj_cols, distractor_heights), BAR_R0, BAR_INPUT_C0S[other_index])
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_slot_column(grid, col, length, color, count, None)
    return grid


def _build_expected(target_heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(BAR_LOCAL_PROJ_COLS, target_heights), BAR_R0, BAR_OUTPUT_C0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, target_heights):
        _paint_slot_column(grid, col, length, color, count, height)
    return grid


def _sample_params(rng: np.random.Generator):
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    target_heights = _sample_target_heights(rng)
    distractor_family, distractor_proj_cols, distractor_heights = _sample_distractor(rng)
    target_pos_index = int(rng.integers(0, 2))
    return (
        colors,
        colored_counts,
        target_heights,
        distractor_family,
        distractor_proj_cols,
        distractor_heights,
        target_pos_index,
    )


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    (
        colors,
        colored_counts,
        target_heights,
        _distractor_family,
        distractor_proj_cols,
        distractor_heights,
        target_pos_index,
    ) = _sample_params(rng)

    grid = _build_input(
        target_heights,
        colors,
        colored_counts,
        target_pos_index,
        distractor_proj_cols,
        distractor_heights,
    )
    expected = _build_expected(target_heights, colors, colored_counts)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    grey_comps = list(_cc4(grid == GREY))
    if len(grey_comps) != 3:
        return None

    slot_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((grid != BG) & (grid != GREY))
            if FRAME_R0 <= int(r) < FRAME_R0 + FRAME_H and FRAME_C0 <= int(c) < FRAME_C0 + FRAME_W
        }
    )
    exact_matches = 0
    for pts in grey_comps:
        bb = _bbox(pts)
        if bb[1] >= FRAME_C0:
            continue
        _base, proj_cols = _projection_signature(pts)
        exact, _dx, _overlap = _best_alignment(proj_cols, slot_cols, W)
        exact_matches += int(exact)
    if exact_matches != 1:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
