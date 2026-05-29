# Combo variant of 4c3d4a41 (index 26)
# combo_spec: D6/1+D14/1
# Name: 框架内部输出+清除灰帽

import numpy as np

from arc_utils import bbox_coords, components_mask, colors_present, get_bg, paste


# Design analysis for the fused variant.
# a) Concrete trace:
#    Keep the original 8x20 input geometry. Suppose the bar heights are [2,4,1,3] and the colored counts are
#    [1,2,4,1]. In the base task, after docking the bar into the frame, the first target column would still retain
#    grey cap cells above its lone color because the solver copies the slot region before repainting. Here the
#    output is only the frame interior (6x9), and those inherited cap cells are dropped to bg; only the moved bar
#    and the recolored stacks remain.
# b) Complexity calibration:
#    The frame, bar, four destination columns, and the dx-alignment logic are preserved. This keeps repeated
#    structural evidence in every sample while making the fused change visually obvious: the border disappears and
#    some former grey hats turn into empty bg cells.
# c) Hidden assumptions audited:
#    - Largest grey component is still the frame; second largest is still the bar.
#    - Colored pixels inside the frame still identify the destination columns.
#    - A single horizontal shift aligns all bar projections with those columns.
#    The broken base-task assumption is that copying the slot patch is safe. It is no longer safe because that
#    would preserve both the border and the old grey hats, so the fused solver rebuilds the interior from scratch.
# d) Placement legality:
#    Generator keeps the original fixed frame bbox and bar geometry. The translated bar still lands at rows 2..5,
#    cols 11..17, which lies fully inside the frame interior rows 1..6, cols 10..18.
# e) Visual observability:
#    D6/1 is visible because output size shrinks from 8x20 to 6x9. D14/1 is visible because at least some
#    columns show bg above the colors where grey hats would have survived in the original task.
# f) Cleanliness:
#    The cropped output contains only bg, the moved bar's grey pixels, and the column colors. No frame border or
#    inherited slot filler remains, so the result stays readable.
# g) Color inference:
#    Colors are not remapped; each slot column keeps one distinct color. The rule is positional, not numeric.
# h) Generator anti-degeneration guard:
#    Samples are rejected unless the fused output erases at least MIN_ERASED_HATS grey pixels relative to the
#    cropped original-style output, so this does not collapse into a pure crop-only variant.
# i) New solver assumptions vs generator:
#    Solver assumes every destination column has at least one colored pixel and that projection columns are spaced
#    like the slot columns. Generator enforces the original four-column layout, distinct colors, and valid counts.


def _bbox(pts):
    arr = pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int)
    return bbox_coords(arr)


def _cc4(mask: np.ndarray):
    return components_mask(mask, connectivity=4)


def _most_common(arr) -> int:
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the grey bar into the frame, then return only the cleaned frame interior.

    Input structure:
    - The dominant color is background; grey forms two disconnected structures: a large half-open frame and a
      smaller horizontal bar with upward projections.
    - Inside the frame, four colored columns mark the x-positions that the bar projections must align with.

    Rule:
    - Translate the bar so its base sits one row above the interior floor and its projections line up with the
      colored columns.
    - Re-stack each column's original colors immediately above the translated projection in that column.
    - Output only the frame's interior rectangle, removing the border and clearing every pre-existing grey hat so
      that only bg, the translated bar, and the moved colored cells remain.

    Key insight:
    - The frame still identifies the target cavity, but the final answer must be reconstructed from the aligned
      bar geometry and the slot colors rather than copied from the original slot contents.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    grey_comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(g == grey)]
    grey_comps.sort(key=lambda t: (-t[0], t[1]))
    if len(grey_comps) < 2:
        return g.copy()

    (_, slot_bbox, _), (_, _bar_bbox, bar_pts) = grey_comps[0], grey_comps[1]
    sr0, sc0, sr1, sc1 = map(int, slot_bbox)
    ir0, ic0, ir1, ic1 = sr0 + 1, sc0 + 1, sr1 - 1, sc1 - 1

    out = np.full((ir1 - ir0 + 1, ic1 - ic0 + 1), bg, dtype=int)

    bar_base_row = max(int(r) for r, _ in bar_pts)
    target_base_row = ir1 - 1
    dy = target_base_row - bar_base_row

    col_min_row: dict[int, int] = {}
    for r, c in bar_pts:
        rr = int(r)
        cc = int(c)
        col_min_row[cc] = min(col_min_row.get(cc, rr), rr)
    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)

    dst_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((g[ir0 : ir1 + 1, ic0 : ic1 + 1] != bg) & (g[ir0 : ir1 + 1, ic0 : ic1 + 1] != grey))
        }
    )
    dst_cols = [ic0 + c for c in dst_cols]

    dx = None
    if src_proj_cols and len(src_proj_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_proj_cols, dst_cols)]
        if len(set(deltas)) == 1:
            dx = deltas[0]
    if dx is None:
        best = (-1, 0)
        dst_set = set(dst_cols)
        width = g.shape[1]
        for cand_dx in range(-width, width + 1):
            score = sum((c + cand_dx) in dst_set for c in src_proj_cols)
            if score > best[0]:
                best = (score, cand_dx)
        dx = best[1]

    placed_by_col: dict[int, list[int]] = {}
    for r, c in bar_pts:
        rr = int(r) + int(dy)
        cc = int(c) + int(dx)
        if ir0 <= rr <= ir1 and ic0 <= cc <= ic1:
            out[rr - ir0, cc - ic0] = grey
            placed_by_col.setdefault(cc, []).append(rr)

    for col in dst_cols:
        if col not in placed_by_col:
            continue
        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
        if desired_bottom < ir0:
            continue

        rows = [r for r in range(ir0, ir1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]
        max_len = desired_bottom - ir0 + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]
        start_row = desired_bottom - len(colors) + 1
        for i, colv in enumerate(colors):
            rr = start_row + i
            if ir0 <= rr <= ir1:
                out[rr - ir0, col - ic0] = colv

    return out


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

INTERIOR_R0 = FRAME_R0 + 1
INTERIOR_C0 = FRAME_C0 + 1
INTERIOR_H = FRAME_H - 2
INTERIOR_W = FRAME_W - 2

SLOT_COLS = [11, 13, 15, 17]
SLOT_COLS_LOCAL = [c - INTERIOR_C0 for c in SLOT_COLS]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1
SLOT_TOP_ROW_LOCAL = SLOT_TOP_ROW - INTERIOR_R0

BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1
BAR_OUTPUT_LOCAL_R0 = BAR_R0 - INTERIOR_R0
BAR_OUTPUT_LOCAL_C0 = BAR_OUTPUT_C0 - INTERIOR_C0
BAR_BASE_LOCAL_ROW = BAR_BASE_ABS_ROW - INTERIOR_R0

HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 300
MIN_ERASED_HATS = 2


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[BAR_H - height : BAR_H, local_c] = GREY
    return patch


def _paint_input_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color


def _build_input_grid(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_output_interior(
    heights: list[int], colors: list[int], colored_counts: list[int], preserve_hats: bool
) -> np.ndarray:
    if preserve_hats:
        out = _build_input_grid(heights, colors, colored_counts)[
            INTERIOR_R0 : INTERIOR_R0 + INTERIOR_H,
            INTERIOR_C0 : INTERIOR_C0 + INTERIOR_W,
        ].copy()
        out[(out != BG) & (out != GREY)] = BG
    else:
        out = np.zeros((INTERIOR_H, INTERIOR_W), dtype=int)

    paste(out, _bar_patch(heights), BAR_OUTPUT_LOCAL_R0, BAR_OUTPUT_LOCAL_C0, overwrite=True, clip=True)

    for local_col, color, count, height in zip(SLOT_COLS_LOCAL, colors, colored_counts, heights):
        bar_top = BAR_BASE_LOCAL_ROW - height + 1
        desired_bottom = bar_top - 1
        if desired_bottom < 0:
            continue
        visible = min(count, desired_bottom + 1)
        if visible <= 0:
            continue
        start = desired_bottom - visible + 1
        out[start : desired_bottom + 1, local_col] = color

    return out


def _erased_hat_count(heights: list[int], colors: list[int], colored_counts: list[int]) -> int:
    with_hats = _build_output_interior(heights, colors, colored_counts, preserve_hats=True)
    without_hats = _build_output_interior(heights, colors, colored_counts, preserve_hats=False)
    return int(np.count_nonzero((with_hats == GREY) & (without_hats == BG)))


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    if _erased_hat_count(heights, colors, colored_counts) < MIN_ERASED_HATS:
        return None

    grid = _build_input_grid(heights, colors, colored_counts)
    expected = _build_output_interior(heights, colors, colored_counts, preserve_hats=False)

    if int(get_bg(grid)) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
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
