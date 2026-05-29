# Combo variant of 4c3d4a41 (index 19)
# combo_spec: D13/2+D16/1
# Name: 双杆选择+贪心匹配

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, paste
from arc_utils import get_bg as _bg_color


# Design analysis
# ----------------
# a) Concrete behavior trace
#    - Original task: one grey frame, one grey bar, and four colored slot columns. The bar is translated rigidly so
#      its visible upward projections land under the colored columns, then each column's colors are restacked above
#      the inserted bar.
#    - Combo task: there are now two candidate grey bars but still one frame. Neither candidate is an exact 4-column
#      fit to the slot. For each bar, the solver must search horizontal shifts and count how many visible projections
#      land on colored slot columns; then it must choose the bar with the larger maximum overlap.
#    - Example trace: suppose slot columns are at 19/21/23/25. Bar A has projections at local cols [0,1,4,6], whose
#      best rigid translation hits 3 slot columns. Bar B has projections at [0,1,3,4], whose best rigid translation
#      hits only 2 slot columns. The output uses Bar A, translated by its best dx, so only the three covered slot
#      columns receive restacked colors; the uncovered slot column loses its colored pixels.
#
# b) Complexity calibration
#    - Both bars keep the same 7-wide base and both keep four visible projections, so the distractor is not trivially
#      rejected by component size or projection count.
#    - The correct-bar family is restricted to a short menu of wrong-spacing patterns with best overlap exactly 3.
#      The distractor family is restricted to patterns with best overlap exactly 2. This makes the greedy criterion
#      visible, stable, and unambiguous across samples.
#    - Colors and slot geometry stay close to the original task so the new difficulty is localized to bar selection
#      plus partial matching, rather than to unrelated layout changes.
#
# c) Implicit assumptions in the original solver that break here
#    - "The second-largest grey component is the bar" is invalid because there are three grey components.
#    - "A compatible bar can be identified by exact constant-delta matching to all destination columns" is invalid:
#      the chosen bar is intentionally only a partial fit.
#    - "Every colored slot column will still receive output colors" is invalid: unmatched slot columns remain unfilled.
#    - The original restacking loop assumes colors are still taken from the original slot columns in place; that part
#      remains valid and is preserved.
#
# d) Placement legality
#    - The canvas is widened to 28 columns. The frame occupies cols 17-27, while the two 7-wide bars are staged at
#      cols 1 and 9. This leaves at least one background column between every object, so components stay disconnected.
#    - All sampled projection columns stay inside the 7-wide bar patch and all projection heights are in {2,3,4}, so
#      every declared projection is visibly above the base and remains a true matching feature.
#    - The greedy dx search uses a secondary preference for keeping more translated bar pixels inside the frame bbox,
#      so tied overlap scores still produce a legal-looking insertion.
#
# e) Visual observability
#    - Across a few examples, a human can see two similar candidate bars and only one is moved into the frame.
#    - The deciding feature is visible geometry: which bar can place more bumps directly under the colored columns.
#      No color ids or hidden metadata are used in the ranking.
#    - Because the chosen bar only matches 3 of the 4 slot columns, the output visibly drops one colored column,
#      making the "partial greedy fit" behavior distinct from the original exact-fit task.
#
# f) Visual cleanliness
#    - The left side contains only the two staged bars; the right side contains only the frame and slot columns.
#      The added complexity is easy to isolate visually.
#    - Inside the frame, the output still uses the original layering vocabulary: grey frame, grey inserted bar, and
#      monochrome vertical color stacks. There are no extra markers or annotations.
#
# g) Color inference
#    - Colors are not remapped. Each surviving output stack still comes from the same input slot column as in the
#      original task. The only new inference is whether that slot column is actually covered by the selected bar.
#
# h) Generator guarantees for the combo behavior
#    - Exactly one candidate bar has greedy best overlap 3; the other has greedy best overlap 2.
#    - No candidate bar is an exact four-column fit to the frame, so the D16 change is always active instead of
#      collapsing back to the original or to the single D13 exact-match variant.
#    - The better bar is randomized between the two left-side staging positions, preventing a positional shortcut.
#
# i) Post-change assumption audit for main()
#    - Largest grey component = frame: preserved by construction because the frame area is larger than either bar.
#    - Colored destination columns live only inside the frame bbox: preserved exactly as in the original task.
#    - Best-overlap bar is unique: guaranteed by the 3-vs-2 overlap family split in the generator.
#    - Projection extraction uses only visible bumps above the base row: guaranteed by sampling heights >= 2.


def _bbox(pts):
    arr = pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int)
    return bbox_coords(arr)


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
    proj_cols = sorted(c for c, top in col_min_row.items() if top < base_row)
    return int(base_row), proj_cols


def _slot_color_columns(g: np.ndarray, bg: int, grey: int, slot_bbox: tuple[int, int, int, int]) -> list[int]:
    sr0, sc0, sr1, sc1 = slot_bbox
    return sorted(
        {
            int(c)
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )


def _best_shift_for_bar(
    bar_pts,
    dy: int,
    src_proj_cols: list[int],
    dst_cols: list[int],
    slot_bbox: tuple[int, int, int, int],
    span: int,
) -> tuple[int, int, int, float]:
    sr0, sc0, sr1, sc1 = slot_bbox
    dst_set = set(dst_cols)
    dst_center = float(sum(dst_cols) / len(dst_cols)) if dst_cols else 0.0

    best_key = None
    best_dx = 0
    best_overlap = -1
    best_inside = -1
    best_center_cost = float("inf")

    for cand_dx in range(-span, span + 1):
        shifted = [c + cand_dx for c in src_proj_cols]
        overlap = sum(col in dst_set for col in shifted)
        inside = sum(
            sr0 <= int(r) + dy <= sr1 and sc0 <= int(c) + cand_dx <= sc1
            for r, c in bar_pts
        )
        center = float(sum(shifted) / len(shifted)) if shifted else 0.0
        center_cost = abs(center - dst_center)
        key = (overlap, inside, -center_cost, -abs(cand_dx))
        if best_key is None or key > best_key:
            best_key = key
            best_dx = int(cand_dx)
            best_overlap = int(overlap)
            best_inside = int(inside)
            best_center_cost = float(center_cost)

    return best_dx, best_overlap, best_inside, best_center_cost


def _render_with_bar(
    g: np.ndarray,
    bg: int,
    grey: int,
    slot_bbox: tuple[int, int, int, int],
    bar_pts,
    dy: int,
    dx: int,
) -> np.ndarray:
    sr0, sc0, sr1, sc1 = slot_bbox
    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    placed: list[tuple[int, int]] = []
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1
    for col in _slot_color_columns(g, bg, grey, slot_bbox):
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
                out[rr, col] = int(colv)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two candidate grey bars by greedy overlap with the frame columns, then insert the better one.

    Input structure:
    - The background is plain, while grey forms three disconnected components: one frame on the right and two staged bars on the left.
    - Inside the frame, colored pixels occupy four vertical slot columns and mark the only columns that can receive restacked colors.

    Rule:
    - Erase both input bars from the output canvas, keep only the frame area, and examine each bar's visible upward projections.
    - For each bar, search rigid horizontal shifts and score how many projections land on colored slot columns; choose the bar with the highest best score.
    - Move the chosen bar upward into the frame so its base sits one row above the frame's interior floor, using that bar's best horizontal shift.
    - For every frame column that ends up directly above a placed bar column, restack that column's original colors immediately on top of the bar; unmatched colored columns stay empty.

    Key insight:
    - This is no longer about finding an exact key. The correct bar is the one with the strongest partial fit under a greedy maximum-overlap translation.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

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
    dst_cols = _slot_color_columns(g, bg, grey, slot_bbox)
    if not dst_cols:
        sr0, sc0, sr1, sc1 = slot_bbox
        out = np.full_like(g, bg)
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
        return out

    sr0, _sc0, sr1, _sc1 = slot_bbox
    target_base_row = (sr1 - 1) - 1

    chosen = None
    best_key = None
    for _size, _bar_bbox, bar_pts in bar_comps:
        base_row, src_proj_cols = _projection_signature(bar_pts)
        dy = int(target_base_row - base_row)
        dx, overlap, inside, center_cost = _best_shift_for_bar(bar_pts, dy, src_proj_cols, dst_cols, slot_bbox, w)
        key = (overlap, inside, -center_cost, -abs(len(src_proj_cols) - len(dst_cols)), -abs(dx))
        if best_key is None or key > best_key:
            best_key = key
            chosen = (bar_pts, dy, dx)

    if chosen is None:
        return g.copy()

    bar_pts, dy, dx = chosen
    return _render_with_bar(g, bg, grey, slot_bbox, bar_pts, dy, dx)


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
BAR_INPUT_C0S = [1, 9]
BAR_HEIGHT_RANGE = (2, 5)
MAX_ATTEMPTS = 500

GOOD_PATTERNS = (
    (0, 1, 2, 6),
    (0, 1, 4, 6),
    (0, 2, 3, 6),
    (0, 2, 5, 6),
    (0, 3, 4, 6),
    (0, 4, 5, 6),
)

BAD_PATTERNS = (
    (0, 1, 3, 4),
    (0, 1, 5, 6),
    (1, 2, 4, 5),
    (2, 3, 5, 6),
    (0, 3, 4, 5),
    (1, 2, 3, 6),
)


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(proj_cols: tuple[int, ...], heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip(proj_cols, heights):
        patch[BAR_H - int(height) : BAR_H, int(local_c)] = GREY
    return patch


def _paint_input_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = int(length - count)
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = int(color)


def _paint_output_slot_cap(grid: np.ndarray, col: int, length: int, count: int) -> None:
    cap = int(length - count)
    if cap > 0:
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY


def _sample_heights(rng: np.random.Generator, n: int) -> list[int]:
    return [int(rng.integers(*BAR_HEIGHT_RANGE)) for _ in range(n)]


def _bar_points(proj_cols: tuple[int, ...], heights: list[int], c0: int) -> np.ndarray:
    patch = _bar_patch(proj_cols, heights)
    pts = np.argwhere(patch == GREY).astype(int)
    pts[:, 0] += BAR_R0
    pts[:, 1] += int(c0)
    return pts


def _build_input(
    good_pattern: tuple[int, ...],
    good_heights: list[int],
    bad_pattern: tuple[int, ...],
    bad_heights: list[int],
    colors: list[int],
    counts: list[int],
    good_pos_index: int,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(good_pattern, good_heights), BAR_R0, BAR_INPUT_C0S[good_pos_index])
    paste(grid, _bar_patch(bad_pattern, bad_heights), BAR_R0, BAR_INPUT_C0S[1 - good_pos_index])
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, counts):
        _paint_input_slot_column(grid, int(col), int(length), int(color), int(count))
    return grid


def _build_expected(
    good_pattern: tuple[int, ...],
    good_heights: list[int],
    colors: list[int],
    counts: list[int],
    good_pos_index: int,
) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    paste(out, _frame_patch(), FRAME_R0, FRAME_C0)
    for col, length, count in zip(SLOT_COLS, SLOT_LENS, counts):
        _paint_output_slot_cap(out, int(col), int(length), int(count))

    slot_bbox = (FRAME_R0, FRAME_C0, FRAME_R0 + FRAME_H - 1, FRAME_C0 + FRAME_W - 1)
    good_c0 = BAR_INPUT_C0S[good_pos_index]
    bar_pts = _bar_points(good_pattern, good_heights, good_c0)
    base_row, src_proj_cols = _projection_signature(bar_pts)
    target_base_row = (slot_bbox[2] - 1) - 1
    dy = int(target_base_row - base_row)
    dx, _overlap, _inside, _center_cost = _best_shift_for_bar(bar_pts, dy, src_proj_cols, SLOT_COLS, slot_bbox, W)

    placed_by_col: dict[int, list[int]] = {}
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if slot_bbox[0] <= rr <= slot_bbox[2] and slot_bbox[1] <= cc <= slot_bbox[3]:
            out[rr, cc] = GREY
            placed_by_col.setdefault(int(cc), []).append(int(rr))

    interior_top = FRAME_R0 + 1
    for col, color, count in zip(SLOT_COLS, colors, counts):
        col = int(col)
        if col not in placed_by_col:
            continue
        desired_bottom = min(placed_by_col[col]) - 1
        visible = min(int(count), int(desired_bottom - interior_top + 1))
        if visible <= 0:
            continue
        start_row = desired_bottom - visible + 1
        out[start_row : desired_bottom + 1, col] = int(color)

    return out


def _sample_params(rng: np.random.Generator):
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    good_pattern = GOOD_PATTERNS[int(rng.integers(0, len(GOOD_PATTERNS)))]
    bad_pattern = BAD_PATTERNS[int(rng.integers(0, len(BAD_PATTERNS)))]
    good_heights = _sample_heights(rng, len(good_pattern))
    bad_heights = _sample_heights(rng, len(bad_pattern))
    good_pos_index = int(rng.integers(0, 2))
    return colors, counts, good_pattern, good_heights, bad_pattern, bad_heights, good_pos_index


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    colors, counts, good_pattern, good_heights, bad_pattern, bad_heights, good_pos_index = _sample_params(rng)
    grid = _build_input(good_pattern, good_heights, bad_pattern, bad_heights, colors, counts, good_pos_index)
    expected = _build_expected(good_pattern, good_heights, colors, counts, good_pos_index)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None

    grey_comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(grid == GREY)]
    if len(grey_comps) != 3:
        return None
    grey_comps.sort(key=lambda t: (-t[0], t[1]))
    slot_bbox = grey_comps[0][1]
    sr0, _sc0, sr1, _sc1 = slot_bbox
    target_base_row = (sr1 - 1) - 1
    dst_cols = _slot_color_columns(grid, BG, GREY, slot_bbox)

    overlaps = []
    exact_like = 0
    for _size, _bbox_unused, pts in grey_comps[1:]:
        base_row, src_proj_cols = _projection_signature(pts)
        dy = int(target_base_row - base_row)
        dx, overlap, _inside, _center_cost = _best_shift_for_bar(pts, dy, src_proj_cols, dst_cols, slot_bbox, W)
        overlaps.append(int(overlap))
        shifted = [c + dx for c in src_proj_cols]
        exact_like += int(len(src_proj_cols) == len(dst_cols) and shifted == dst_cols)

    overlaps.sort(reverse=True)
    if overlaps != [3, 2]:
        return None
    if exact_like != 0:
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
