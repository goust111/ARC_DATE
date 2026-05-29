# Combo variant of 136b0064 (index 25)
# combo_spec: D1/0+D5/3+D8/3
# Name: 行主序渐变色稀疏折线

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace:
#    The visible input layout still matches the seed task: a 7-column left canvas
#    holds 3x3 glyphs in two tile columns, a full-height divider sits at col 7, and
#    a single gray start pixel lies on the right. The fused changes all happen in how
#    those glyphs are decoded into the output walk.
#    Small example with N=3 and row slots:
#      row 0: A at (0,0), B at (0,4)
#      row 1: C at (4,0), D at (4,4)
#      row 2: E at (8,0), F at (8,4)
#    Original order: A, C, E, B, D, F. New order: A, B, C, D, E, F.
#    Original step: before each segment, move down 1 row. New step: move down 2 rows.
#    Original color: segment color copied the glyph's own color. New color: segment
#    index 0..k-1 is mapped to a fixed light-to-dark rainbow palette, so color is
#    determined by position in the decoded sequence rather than by glyph identity.
#    Therefore the same six glyphs now produce a row-major, vertically sparser walk
#    whose colors progress along the path.
#
# b) Complexity calibration:
#    The combo can degenerate in three independent ways:
#    - row-major may accidentally match column-major on a symmetric area sequence;
#    - step-2 may be visually hidden if the compared path is too short or trivial;
#    - gradient coloring may become uninformative if samples are too short or if the
#      produced output accidentally matches the old glyph-bound coloring.
#    Generation therefore rejects samples unless the new output differs from
#    (1) row-major + step-1 + gradient, (2) column-major + step-2 + gradient, and
#    (3) row-major + step-2 + glyph-bound color. This keeps all three fused changes
#    active in the training distribution. It also biases the path sampler toward
#    underused areas and rejects paths with too few distinct glyph types so the
#    four motion branches remain visibly represented instead of collapsing mostly
#    to the easy right/down cases.
#
# c) Implicit-assumption audit:
#    - Divider detection as a full-height constant non-background column still holds.
#    - Component extraction still assumes each intended glyph is a 3x3 monochromatic
#      8-connected component; the generator preserves the original tile spacing.
#    - The old solver assumed glyph color was the segment color. That assumption is
#      false now, so the solver must ignore glyph color after component detection.
#    - The old solver also assumed `glyphs.sort(key=(c0,r0))` and `r_cur += 1`; both
#      assumptions are false in this combo and are replaced by `(r0,c0)` sorting and
#      `r_cur += 2`.
#    - Because every glyph now consumes at least 2 rows of output progress, the old
#      height formula `H = 4*N - 1` is no longer safe. Generation must use a taller
#      canvas so the sparse walk never exits the bottom.
#
# d) Placement legality:
#    The input still uses fixed tile slots `(r0, c0) = (4*t, 0 or 4)` so glyphs do
#    not overlap each other, the divider, or the gray start. Output-first sampling is
#    done with the new sparse deltas `{4:(3,0), 5:(2,2), 6:(2,-3), 7:(2,-1)}`, which
#    guarantees every chosen instruction stays within the 7-column canvas and below
#    the computed height.
#
# e) Visual observability:
#    A human can infer the rule from a few examples because the three changes are all
#    visible in pixel space: paired glyphs in the same tile row are consumed before
#    dropping to the next tile row, consecutive segments start two rows lower instead
#    of one, and output colors no longer match glyph colors but progress along the
#    path. No hidden ordering cue such as raw color-id sorting is used.
#
# f) Visual cleanliness:
#    The grid remains tidy because only one polyline is drawn, the start marker stays
#    isolated at row 0, the sparse step inserts blank rows between segments, and the
#    glyph area is separated from the output area by the standard divider. No extra
#    markers are added.
#
# g) Color inferability:
#    The gradient uses a short fixed ordered palette `(8, 1, 4, 7, 6, 2, 9)` sampled
#    from early to late path positions. This keeps the number of output colors small,
#    repeated across examples, and tightly tied to a visible structural cue: segment
#    order along the walk. The generator keeps at least four glyphs, so the gradient
#    is not collapsed to a single ambiguous color.
#
# New solver assumptions checked against generation:
# - There is exactly one full-height constant divider column.
# - The gray start shifts into columns 0..6 after removing the divider.
# - Every intended glyph is a 3x3 monochromatic component with area in {4,5,6,7}.
# - Glyphs are placed in row-major tile order when the generator stamps the grid.
# - The chosen height `H = 6*N + 1` is sufficient for the sparse step-2 walk.
# - Generated samples make the combo observable by changing all of order, color
#   mode, and step size relative to their corresponding ablations.
# - Generated samples contain at least three distinct glyph areas including the
#   long-left area-6 branch, so the walk grammar does not collapse to a narrow
#   subset of instructions.


N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0
STEP_DOWN = 2

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREA_TO_DIR = {5: 1, 6: -1, 7: -1}
_AREAS = (4, 5, 6, 7)
_SPARSE_AREA_DELTA = {4: (3, 0), 5: (2, 2), 6: (2, -3), 7: (2, -1)}
_DENSE_AREA_DELTA = {4: (2, 0), 5: (1, 2), 6: (1, -3), 7: (1, -1)}
_GRADIENT_PALETTE = (8, 1, 4, 7, 6, 2, 9)

MIN_DIRECTION_CHANGES = 2
MIN_DISTINCT_AREAS = 3
REQUIRE_AREA6 = True
MAX_ATTEMPTS = 600


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    _H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def _collect_glyphs(left: np.ndarray, *, bg: int, div_color: int | None) -> list[dict[str, int]]:
    ignore = {int(bg), GRAY}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        box = comp["bbox"]
        if bbox_h(box) == 3 and bbox_w(box) == 3:
            r0, c0, _r1, _c1 = map(int, box)
            glyphs.append({"color": color, "r0": r0, "c0": c0, "area": int(comp["area"])})
    return glyphs


def _gradient_color(idx: int, total: int) -> int:
    if total <= 1:
        return int(_GRADIENT_PALETTE[0])
    pal_idx = round(idx * (len(_GRADIENT_PALETTE) - 1) / (total - 1))
    return int(_GRADIENT_PALETTE[int(pal_idx)])


def _solve_with_modes(
    grid: np.ndarray,
    *,
    row_major: bool,
    step_down: int,
    gradient_color: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    left = g[:, : int(div_c)]
    glyphs = _collect_glyphs(left, bg=bg, div_color=div_color)
    if row_major:
        glyphs.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
    else:
        glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    total = len(glyphs)
    for idx, glyph in enumerate(glyphs):
        area = int(glyph["area"])
        color = _gradient_color(idx, total) if gradient_color else int(glyph["color"])
        seg_len = int(_AREA_TO_LEN[area])

        r_cur = int(r_cur + step_down)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = color

        if area == 4:
            end_r = int(r_cur + (seg_len - 1))
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = color
        else:
            dir_h = int(_AREA_TO_DIR[area])
            end_c = int(c_cur) + dir_h * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[r_cur, c_cur] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode paired 3x3 glyphs into a sparse rainbow polyline on the left canvas.

    Input structure:
    - Left of a full-height divider: monochromatic 3x3 glyphs stamped into two tile columns on a black background.
    - Right of the divider: exactly one gray (5) start pixel, which becomes the walk origin after the divider is removed.

    Rule:
    - Remove the divider and shift the gray start into the 7-column left canvas.
    - Read glyphs in row-major tile order: top tile row first, with the left glyph before the right glyph in each row.
    - Each glyph's pixel area still chooses the segment geometry: area 4 -> down 2, area 5 -> right 3, area 6 -> left 4, area 7 -> left 2.
    - Before drawing each segment, move down 2 rows; color the segment by its position in the decoded sequence using a fixed early-to-late gradient palette rather than the glyph's own color.

    Key insight:
    - Shape controls only the motion, while traversal order controls both when a segment is executed and which gradient color it receives, so the solution must jointly infer row-wise decoding, sparse vertical stepping, and order-based coloring.
    """
    return _solve_with_modes(grid, row_major=True, step_down=STEP_DOWN, gradient_color=True)


def _compute_height(N: int) -> int:
    # Worst case is all area-4 glyphs: each contributes +3 rows after the step-2 rule.
    return 6 * N + 1


def _pick_area(opts: list[int], counts: dict[int, int]) -> int:
    # Prefer locally underused instruction types so the generator exercises the
    # full walk grammar instead of overproducing the easiest moves.
    weights: list[float] = []
    for area in opts:
        weight = 3.0 / (1 + counts.get(area, 0))
        if area == 6:
            weight *= 1.6
        if area not in counts:
            weight *= 1.3
        weights.append(weight)
    return int(random.choices(opts, weights=weights, k=1)[0])


def _path_is_rich(areas: list[int]) -> bool:
    distinct = set(areas)
    if len(distinct) < MIN_DISTINCT_AREAS:
        return False
    if REQUIRE_AREA6 and 6 not in distinct:
        return False
    return True


def _sample_path(N: int, start_col: int) -> list[int] | None:
    H = _compute_height(N)
    r, c = GRAY_START_ROW, start_col
    areas: list[int] = []
    counts: dict[int, int] = {}
    for _ in range(2 * N):
        opts = [a for a in _AREAS if r + _SPARSE_AREA_DELTA[a][0] < H and 0 <= c + _SPARSE_AREA_DELTA[a][1] < LEFT_W]
        if not opts:
            return None
        area = _pick_area(opts, counts)
        areas.append(area)
        counts[area] = counts.get(area, 0) + 1
        dr, dc = _SPARSE_AREA_DELTA[area]
        r += dr
        c += dc

    types = [a == 4 for a in areas]
    changes = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])
    if changes < MIN_DIRECTION_CHANGES:
        return None
    if not _path_is_rich(areas):
        return None
    return areas


def _render_grid(N: int, start_col: int, areas: list[int]) -> np.ndarray:
    H = _compute_height(N)
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i % 2]
        r0 = TILE_ROW_STRIDE * (i // 2)
        grid[r0:r0 + 3, tile_col:tile_col + 3] = _TEMPLATES[area] * _AREA_TO_COLOR[area]
    return grid


def _column_major_permutation(areas: list[int]) -> list[int]:
    N = len(areas) // 2
    left_col = [areas[2 * r] for r in range(N)]
    right_col = [areas[2 * r + 1] for r in range(N)]
    return left_col + right_col


def _variant_observable(inp: np.ndarray, areas: list[int]) -> bool:
    new_out = main(inp)

    if areas == _column_major_permutation(areas):
        return False

    order_ablation = _solve_with_modes(inp, row_major=False, step_down=STEP_DOWN, gradient_color=True)
    step_ablation = _solve_with_modes(inp, row_major=True, step_down=1, gradient_color=True)
    color_ablation = _solve_with_modes(inp, row_major=True, step_down=STEP_DOWN, gradient_color=False)

    if np.array_equal(new_out, order_ablation):
        return False
    if np.array_equal(new_out, step_ablation):
        return False
    if np.array_equal(new_out, color_ablation):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    areas = _sample_path(N, start_col)
    if areas is None:
        return None

    inp = _render_grid(N, start_col, areas)
    if not _variant_observable(inp, areas):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is None:
                continue
            out = main(inp)
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
