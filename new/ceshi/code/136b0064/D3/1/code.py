# Variant of 136b0064
# Dimension: D3 — 源对象——字形的视觉形状与结构 (src)
# Change: 4种固定的3x3单色字形模板（plus、diamond、cross、U），每种形状严格绑定一种颜色... → 字形颜色与形状解绑——颜色独立采样

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the D3 variant
# a) Concretization and behavior trace:
#    - The left side still contains exactly 2*N monochromatic 3x3 glyphs in two tile columns.
#      The four masks stay fixed, so shape still determines motion:
#      area 4 -> down 2, area 5 -> right 3, area 6 -> left 4, area 7 -> left 2.
#    - The change is purely in glyph color assignment: any glyph may now use any sampled
#      paint color from a small palette, independently of its mask. The output polyline still
#      follows the same geometry, but each segment is painted with the observed glyph color,
#      not a shape-bound canonical color.
#    - Small trace:
#      old task: cross(area 6) was always color 3, so "cross -> left 4 green" looked like one rule.
#      new task: a cross can be color 8 and still move left 4, while a plus can also be color 8
#      and move down 2. The solver must therefore split "shape -> motion" from "color -> paint".
# b) Complexity calibration:
#    - Keep only the original four masks and original layout; this avoids inventing extra
#      freedoms unrelated to D3.
#    - Use a restrained glyph palette {1,2,3,6,7,8,9}. This is enough to visibly break the old
#      binding while remaining easy to compare across 2-4 examples.
#    - Generator enforces both "same shape, different colors" and "same color, different shapes"
#      so the decoupling is visible inside individual examples.
# c) Hidden-assumption review of the original main():
#    - Component finding by monochromatic 8-connectivity is still valid because each glyph remains
#      a single-color 3x3 connected object.
#    - Ignoring all components whose color equals divider/start color is no longer safe in general;
#      color is now independent. The new solver therefore reads every 3x3 component on the left
#      canvas and does not filter glyphs by color.
#    - Using glyph["color"] directly for painting remains correct; it is now the explicit rule.
# d) Placement legality:
#    - Tile slots are unchanged: c0 in {0,4}, row stride 4, so 3x3 glyphs never overlap.
#    - The path sampler still guarantees the shifted start and every segment endpoint stay inside
#      the 7-column left canvas and below row H-1.
# e) Visual observability:
#    - Every generated puzzle contains at least one repeated shape with two different colors and
#      at least one shared color used by two different shapes.
#    - At least one glyph uses a non-canonical color (7/8/9), so the change is visible against
#      the original task's color inventory.
# f) Visual cleanliness:
#    - Output segments never overwrite earlier segments: row position strictly increases after each
#      instruction, exactly as in the base task. This keeps color evidence legible.
# g) Color-rule inferability:
#    - Colors are sampled from a small palette and reused, not from many unique colors.
#    - Repetition patterns provide the needed clues: motion repeats with shape, paint repeats with
#      color, and those two repetitions cut across one another.
# Solver/generator contract after the change:
#    - main() assumes the left half contains only valid 3x3 glyph components on background.
#    - generate_input() guarantees all glyph colors avoid the background and gray start; it also
#      rejects any sample whose output would match the old canonical shape->color binding.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """Find a full-height constant non-background column."""
    H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode 3x3 glyphs into a colored polyline across the left canvas.

    Input structure:
    - A full-height divider splits the grid into a left glyph panel and a right start-marker panel.
    - The left panel contains monochromatic 3x3 glyphs on background; the right panel contains one gray (5) start cell.

    Rule:
    - Remove the divider, shift the gray start into the left canvas, and read glyphs in column-major order.
    - Each glyph's visual shape determines the segment geometry: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
    - Each glyph's own color determines the paint color of that segment, independently of the shape.

    Key insight:
    - Shape and color now carry different parts of the instruction: the 3x3 mask gives the motion, while the observed glyph color gives the segment color.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))

    div_c, _div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)

    gray = np.argwhere(g == 5)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = 5

    left = g[:, : int(div_c)]
    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"color": int(comp["color"]), "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    area_to_len = {4: 2, 5: 3, 6: 4, 7: 2}
    area_to_dir = {5: 1, 6: -1, 7: -1}

    for glyph in glyphs:
        area = int(glyph["area"])
        color = int(glyph["color"])
        assert area in area_to_len, f"unexpected glyph area: {area}"
        seg_len = int(area_to_len[area])

        r_cur = int(r_cur + 1)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = color

        if area == 4:
            end_r = r_cur + (seg_len - 1)
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = color
        else:
            dir_h = int(area_to_dir[area])
            end_c = int(c_cur) + dir_h * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[r_cur, c_cur] = color

    return out


N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0
MAX_ATTEMPTS = 500
PATH_ATTEMPTS = 80
COLOR_ATTEMPTS = 200
MIN_DIRECTION_CHANGES = 2

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREA_TO_DIR = {5: 1, 6: -1, 7: -1}
_AREA_DELTA = {4: (2, 0), 5: (1, 2), 6: (1, -3), 7: (1, -1)}
_AREAS = (4, 5, 6, 7)
_OLD_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_GLYPH_COLORS = (1, 2, 3, 6, 7, 8, 9)
_NOVEL_COLORS = (7, 8, 9)


def _direction_change_count(areas: list[int]) -> int:
    kinds = [a == 4 for a in areas]
    return sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1])


def _sample_path(N: int, start_col: int) -> list[int] | None:
    """Sample a valid area sequence that stays inside the left canvas."""
    H = 4 * N - 1
    for _ in range(PATH_ATTEMPTS):
        r, c = GRAY_START_ROW, start_col
        areas: list[int] = []
        for _step in range(2 * N):
            opts = [a for a in _AREAS if r + _AREA_DELTA[a][0] < H and 0 <= c + _AREA_DELTA[a][1] < LEFT_W]
            if not opts:
                break
            area = random.choice(opts)
            areas.append(area)
            dr, dc = _AREA_DELTA[area]
            r += dr
            c += dc

        if len(areas) != 2 * N:
            continue
        if _direction_change_count(areas) < MIN_DIRECTION_CHANGES:
            continue
        if len(set(areas)) < 2:
            continue
        if not any(areas.count(area) >= 2 for area in set(areas)):
            continue
        return areas
    return None


def _color_assignment_is_informative(areas: list[int], colors: list[int]) -> bool:
    if len(set(colors)) < 2:
        return False
    if not any(color in _NOVEL_COLORS for color in colors):
        return False
    if sum(color != _OLD_AREA_TO_COLOR[area] for area, color in zip(areas, colors)) < 2:
        return False

    same_shape_diff_color = any(len({colors[i] for i, area2 in enumerate(areas) if area2 == area}) >= 2 for area in set(areas))
    if not same_shape_diff_color:
        return False

    same_color_diff_shape = any(len({areas[i] for i, color2 in enumerate(colors) if color2 == color}) >= 2 for color in set(colors))
    return same_color_diff_shape


def _sample_colors(areas: list[int]) -> list[int] | None:
    groups: dict[int, list[int]] = {}
    for idx, area in enumerate(areas):
        groups.setdefault(area, []).append(idx)

    repeated_areas = [area for area, idxs in groups.items() if len(idxs) >= 2]
    if not repeated_areas or len(groups) < 2:
        return None

    for _ in range(COLOR_ATTEMPTS):
        shared_color = random.choice(_NOVEL_COLORS)
        alternate_color = random.choice([c for c in _GLYPH_COLORS if c != shared_color])

        repeat_area = random.choice(repeated_areas)
        i0, i1 = random.sample(groups[repeat_area], 2)
        other_area = random.choice([area for area in groups if area != repeat_area])
        i2 = random.choice(groups[other_area])

        palette = set(random.sample(_GLYPH_COLORS, k=random.randint(2, min(4, len(_GLYPH_COLORS)))))
        palette.update({shared_color, alternate_color})
        palette_list = sorted(palette)

        colors: list[int | None] = [None] * len(areas)
        colors[i0] = shared_color
        colors[i1] = alternate_color
        colors[i2] = shared_color

        for idx in range(len(areas)):
            if colors[idx] is None:
                colors[idx] = random.choice(palette_list)

        final_colors = [int(c) for c in colors]
        if _color_assignment_is_informative(areas, final_colors):
            return final_colors
    return None


def _simulate_output(H: int, start_col: int, areas: list[int], colors: list[int], *, canonical_colors: bool) -> np.ndarray:
    out = np.zeros((H, LEFT_W), dtype=int)
    out[GRAY_START_ROW, start_col] = GRAY

    r_cur, c_cur = GRAY_START_ROW, start_col
    for area, color in zip(areas, colors):
        paint = _OLD_AREA_TO_COLOR[area] if canonical_colors else color
        seg_len = _AREA_TO_LEN[area]

        r_cur += 1
        out[r_cur, c_cur] = paint
        if area == 4:
            for _ in range(seg_len - 1):
                r_cur += 1
                out[r_cur, c_cur] = paint
        else:
            dir_h = _AREA_TO_DIR[area]
            for _ in range(seg_len - 1):
                c_cur += dir_h
                out[r_cur, c_cur] = paint
    return out


def _render_grid(N: int, start_col: int, areas: list[int], colors: list[int]) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, (area, color) in enumerate(zip(areas, colors)):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[area] * color
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    N = random.randint(N_MIN, N_MAX)
    H = 4 * N - 1
    start_col = random.randint(0, LEFT_W - 1)

    areas = _sample_path(N, start_col)
    if areas is None:
        return None

    colors = _sample_colors(areas)
    if colors is None:
        return None

    expected = _simulate_output(H, start_col, areas, colors, canonical_colors=False)
    canonical = _simulate_output(H, start_col, areas, colors, canonical_colors=True)
    if np.array_equal(expected, canonical):
        return None

    return _render_grid(N, start_col, areas, colors), expected, canonical


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            attempt = _gen_attempt()
            if attempt is None:
                continue
            inp, expected, canonical = attempt
            out = main(inp)
            if out.shape != expected.shape:
                continue
            if not np.array_equal(out, expected):
                continue
            if np.array_equal(out, canonical):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
