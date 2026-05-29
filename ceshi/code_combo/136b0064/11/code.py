# Combo variant of 136b0064 (index 11)
# combo_spec: D6/3+D10/2
# Name: 自适应画布尺寸输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior trace:
#    Keep the visible input exactly in the original 15-column layout:
#    glyphs occupy the 7-column left panel, divider is still the full-height
#    column 7, and the gray start still sits in cols 8..14.
#    The change happens in the *latent drawing canvas* used by the solver.
#
#    Example with N=4:
#      visible left panel width = 7, number of glyphs = 2*N = 8
#      fused latent width = num_glyphs + 3 = 11
#      gray at input col 10 still decodes to start_col = 2
#      if the first three glyphs are area-5, area-5, area-4 then the traced
#      polyline reaches columns 2..6..8, which would violate the old 7-column
#      canvas but is valid on the new 11-column canvas.
#      The final output is not 15x7 or 15x11; it is cropped to the bbox of the
#      drawn trace, e.g. rows 0..9 and cols 2..8 -> shape 10x7.
#
#    Difference from the original:
#      - old solver: "divider column index == drawable width", output fixed Hx7
#      - fused solver: divider only decodes the gray start shift; drawable width
#        comes from glyph count, then the result is tight-cropped.
#
# b) Complexity calibration:
#    Only one new width rule is introduced: latent_width = num_glyphs + 3.
#    Solver supports widths 7, 9, 11, 13 for N in 2..5. Generator intentionally
#    samples N in 3..5 so every produced training example visibly differs from
#    the original fixed-width task; otherwise the N=2 case collapses back to 7.
#    Tight crop is visually obvious because generated outputs are wider than 7
#    even though the visible left input panel never is.
#
# c) Hidden assumptions broken by the fusion:
#    - Original `out = zeros((H, div_c))` is no longer valid: divider position
#      is not the true drawing width anymore.
#    - Original horizontal bound check `0 <= end_c < div_c` must instead use the
#      latent width derived from glyph count.
#    - Returning the full latent canvas would leak unused blank columns and blur
#      the D6/3 behavior, so a final tight crop is required.
#    - Because the start pixel is still encoded by the right-side 7-column zone,
#      generator must keep start_col in 0..6 even when latent width is larger.
#
# d) Placement legality:
#    Generator still places exactly 2*N non-overlapping 3x3 glyphs into the two
#    original tile columns and exactly one gray pixel on the right.
#    Path search enforces row bounds, latent-width bounds, and the original gray
#    encoding range, so solver asserts are never hit on generated inputs.
#
# e/f/g) Observability, cleanliness, inferability:
#    The input remains clean because no new markers are added; the only visible
#    clue is that more glyphs imply a more permissive output width. To keep that
#    clue observable, generator rejects most N>=3 samples unless the traced path
#    actually uses columns beyond 6 and the cropped output becomes wider than 7.
#    That avoids degeneration back to the original task.
# ---------------------------------------------------------------------------


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """Find a full-height constant non-background column."""
    _h, w = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(w)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(grid != int(bg))
    if pts.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _extract_glyphs(left: np.ndarray, *, bg: int, div_color: int | None) -> list[dict[str, int]]:
    ignore = {int(bg), 5}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        bbox = comp["bbox"]
        if bbox_h(bbox) == 3 and bbox_w(bbox) == 3:
            r0, c0, _r1, _c1 = map(int, bbox)
            glyphs.append({"color": color, "r0": r0, "c0": c0, "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))
    return glyphs


def _latent_width(num_glyphs: int) -> int:
    return int(num_glyphs) + 3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the glyph program, draw it on a hidden width that grows with glyph count, then tight-crop the result.

    Input structure:
    - The visible input keeps the original divided layout: two columns of 3x3 glyphs on the left of a full-height divider, and one gray start pixel on the right.
    - The left visible panel is still only 7 columns wide even though the true drawing canvas may be wider.

    Rule:
    - Read the 3x3 glyphs in column-major order and decode each shape by pixel area as before.
    - Let `num_glyphs` be the number of decoded glyphs; the real drawing width is `num_glyphs + 3`.
    - Shift the gray start into this latent canvas, trace the colored polyline there, then crop the output to the minimal bbox containing all non-background cells.

    Key insight:
    - Glyph shape still determines segment direction and length, but glyph count now also determines the hidden horizontal budget. The divider no longer tells you the final output width.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    left = g[:, : int(div_c)]
    glyphs = _extract_glyphs(left, bg=bg, div_color=div_color)
    draw_w = _latent_width(len(glyphs))
    out = np.full((int(h), int(draw_w)), int(bg), dtype=int)

    gray = np.argwhere(g == 5)
    if gray.size == 0:
        return _tight_crop(out, bg)

    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    assert 0 <= c_cur < 7
    assert c_cur < int(draw_w)
    out[int(r_cur), int(c_cur)] = 5

    area_to_len = {4: 2, 5: 3, 6: 4, 7: 2}
    area_to_dir = {5: 1, 6: -1, 7: -1}

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(area_to_len[area])

        r_cur += 1
        assert r_cur < h, f"segment exits bottom: r_cur={r_cur}, H={h}"
        out[int(r_cur), int(c_cur)] = color

        if area == 4:
            end_r = int(r_cur + (seg_len - 1))
            assert end_r < h, f"vertical segment exits bottom: end_r={end_r}, H={h}"
            for _ in range(seg_len - 1):
                r_cur += 1
                out[int(r_cur), int(c_cur)] = color
        else:
            dir_h = int(area_to_dir[area])
            end_c = int(c_cur + dir_h * (seg_len - 1))
            assert 0 <= end_c < int(draw_w), (
                f"horizontal segment exits latent canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, draw_w={draw_w}"
            )
            for _ in range(seg_len - 1):
                c_cur += dir_h
                out[int(r_cur), int(c_cur)] = color

    return _tight_crop(out, bg)


# --------------- generator ---------------

N_MIN, N_MAX = 3, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
VISIBLE_LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_TO_TOTAL_DELTA = {4: (2, 0), 5: (1, 2), 6: (1, -3), 7: (1, -1)}
_AREAS = (4, 5, 6, 7)
MAX_ATTEMPTS = 400


def _trace_columns(area: int, c: int) -> tuple[int, int]:
    dc = int(_AREA_TO_TOTAL_DELTA[int(area)][1])
    c2 = int(c + dc)
    return int(min(c, c2)), int(max(c, c2))


def _sample_area_budget(num_glyphs: int) -> dict[int, int]:
    if int(num_glyphs) == 6:
        required = {4, 5, random.choice((6, 7))}
    else:
        required = set(_AREAS)
    return {area: int(area in required) for area in _AREAS}


def _search_areas(
    *,
    n: int,
    start_col: int,
    target_hi: int,
    target_lo: int,
    area_budget: dict[int, int],
) -> list[int] | None:
    h = 4 * int(n) - 1
    num_glyphs = 2 * int(n)
    draw_w = _latent_width(num_glyphs)
    need_extended = draw_w > VISIBLE_LEFT_W
    path: list[int] = []

    def rec(
        i: int,
        r: int,
        c: int,
        min_c: int,
        max_c: int,
        changes: int,
        prev_vertical: bool | None,
        missing: frozenset[int],
    ) -> bool:
        rem = int(num_glyphs - i)
        if i == num_glyphs:
            if changes < 2:
                return False
            if missing:
                return False
            if min_c > target_lo or max_c < target_hi:
                return False
            if need_extended and max_c <= VISIBLE_LEFT_W - 1:
                return False
            return True

        if r + rem > h - 1:
            return False
        if len(missing) > rem:
            return False
        if max(max_c, c + 2 * rem) < target_hi:
            return False
        if min(min_c, c - 3 * rem) > target_lo:
            return False
        if 5 in missing and max_c < target_hi and max(c + 2 * rem, max_c) < target_hi:
            return False
        if (6 in missing and 7 in missing) and min_c > target_lo and min(c - 3 * rem, min_c) > target_lo:
            return False

        options = list(_AREAS)
        random.shuffle(options)

        def score(area: int) -> tuple[int, int, int]:
            _dr, dc = _AREA_TO_TOTAL_DELTA[int(area)]
            lo, hi = _trace_columns(int(area), int(c))
            toward_hi = int(hi >= target_hi)
            toward_lo = int(lo <= target_lo and dc < 0)
            type_change = int(prev_vertical is not None and (area == 4) != prev_vertical)
            needed = int(area in missing)
            if max_c < target_hi:
                return (toward_hi, needed, type_change)
            if min_c > target_lo:
                return (toward_lo, needed, type_change)
            return (needed, type_change, toward_hi + toward_lo)

        options.sort(key=score, reverse=True)

        for area in options:
            dr, dc = _AREA_TO_TOTAL_DELTA[int(area)]
            new_r = int(r + dr)
            new_c = int(c + dc)
            if not (0 <= new_c < draw_w):
                continue
            if not (new_r < h):
                continue

            lo, hi = _trace_columns(int(area), int(c))
            new_min_c = int(min(min_c, lo))
            new_max_c = int(max(max_c, hi))
            is_vertical = area == 4
            new_changes = int(changes + (prev_vertical is not None and is_vertical != prev_vertical))
            new_missing = missing.difference((int(area),))

            path.append(int(area))
            ok = rec(
                i + 1,
                new_r,
                new_c,
                new_min_c,
                new_max_c,
                new_changes,
                is_vertical,
                new_missing,
            )
            if ok:
                return True
            path.pop()
        return False

    if rec(
        0,
        GRAY_START_ROW,
        int(start_col),
        int(start_col),
        int(start_col),
        0,
        None,
        frozenset(area for area, need in area_budget.items() if need > 0),
    ):
        return path.copy()
    return None


def _render_grid(n: int, start_col: int, areas: list[int]) -> np.ndarray:
    h = 4 * int(n) - 1
    grid = np.zeros((int(h), GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // int(n)]
        r0 = TILE_ROW_STRIDE * (i % int(n))
        color = _AREA_TO_COLOR[int(area)]
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[int(area)] * color
    return grid


def _glyph_count_from_input(grid: np.ndarray) -> int:
    bg = int(get_bg(grid))
    div_c, div_color = find_full_height_constant_col(grid, bg=bg)
    assert div_c is not None
    left = grid[:, : int(div_c)]
    return len(_extract_glyphs(left, bg=bg, div_color=div_color))


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(N_MIN, N_MAX)
    num_glyphs = 2 * int(n)
    draw_w = _latent_width(num_glyphs)
    area_budget = _sample_area_budget(num_glyphs)

    if draw_w > VISIBLE_LEFT_W:
        start_hi = 2 if n == 3 else 1
    else:
        start_hi = VISIBLE_LEFT_W - 1
    start_col = random.randint(0, start_hi)

    target_hi = random.randint(max(VISIBLE_LEFT_W, draw_w - 2), draw_w - 1)
    target_lo = 0 if start_col == 0 else random.randint(0, min(1, start_col))
    areas = _search_areas(
        n=n,
        start_col=start_col,
        target_hi=target_hi,
        target_lo=target_lo,
        area_budget=area_budget,
    )
    if areas is None:
        return None
    return _render_grid(n, start_col, areas)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is None:
                continue

            out = main(inp)
            glyph_count = _glyph_count_from_input(inp)
            draw_w = _latent_width(glyph_count)
            if draw_w > VISIBLE_LEFT_W and out.shape[1] <= VISIBLE_LEFT_W:
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
