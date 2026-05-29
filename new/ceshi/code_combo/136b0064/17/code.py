# Combo variant of 136b0064 (index 17)
# combo_spec: D5/2+D8/0
# Name: 长度定色与面积定步进

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    Keep the original 3x3 glyph inventory and read order, but split the old
#    "area -> (direction, length, color, fixed step=1)" bundle into two branches:
#      area -> direction/length, then length -> output color
#      area -> downward step size before drawing
#    Example sequence [area4, area7, area5, area6] from the same start point:
#      original: step sizes [1,1,1,1], lengths [2,2,3,4], colors [6,2,1,3]
#      fused:    step sizes [1,2,1,2], lengths [2,2,3,4], colors [7,7,8,9]
#    The important witness is area4 vs area7: both make length-2 segments and
#    therefore share output color 7, but area7 starts one row lower because its
#    step size is larger. That makes the two fused changes visibly coupled.
#
# b) Complexity calibration:
#    Use only 3 output colors (one per length) and 2 step sizes (1 or 2). This
#    is enough to create a new inference chain without exploding the hypothesis
#    space. The generator forces every sample to include area4, area7, area6,
#    and at least two area5 glyphs so that:
#      - all three output colors appear;
#      - both step sizes appear;
#      - the "same length => same color" pattern is repeated by area4 and area7.
#
# c) Implicit assumptions changed in main():
#    - Glyph input color is no longer the segment color, so the solver must stop
#      using comp["color"] as drawing color and rely on area -> length -> color.
#    - Step size is no longer constant, so each loop iteration must fetch a
#      per-area step before placing the next segment.
#    - Because area4 and area7 now share segment color, the solver cannot use
#      output color to reconstruct shape identity; it must decode area first.
#
# d) Placement legality in the generator:
#    - Tile placement stays fixed in the original two-column 3x3 layout, so the
#      glyph panel cannot overlap internally.
#    - Path sampling tracks every drawn output cell and rejects candidates that
#      would exit the 7-column canvas, run below the last row, or overwrite an
#      earlier segment cell. This keeps the output legible.
#
# e/f/g) Observability, cleanliness, and inferability:
#    - Output colors 7/8/9 are distinct from the glyph colors and from divider
#      color 4 / start color 5, so the new color rule is visually obvious.
#    - Every generated sample contains repeated length-2 evidence from two
#      different glyph areas (4 and 7), making the color-by-length rule
#      inferable instead of looking like arbitrary recoloring.
#    - The path remains monotone in row direction and non-overlapping, so the
#      final canvas stays clean enough for a human to read from a few examples.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


# --- canvas geometry ---
N_MIN, N_MAX = 3, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

# --- glyph grammar kept from the base task ---
_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_INPUT_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}

# --- fused rule ---
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREA_TO_DIR = {5: 1, 6: -1, 7: -1}
_AREA_TO_STEP = {4: 1, 5: 1, 6: 2, 7: 2}
_LEN_TO_OUT_COLOR = {2: 7, 3: 8, 4: 9}
_AREAS = (4, 5, 6, 7)

# Total endpoint displacement after applying one glyph under the fused rule.
_AREA_TO_DELTA = {4: (2, 0), 5: (1, 2), 6: (2, -3), 7: (2, -1)}

# --- generation constraints ---
_MIN_COUNTS = {4: 1, 5: 2, 6: 1, 7: 1}
MIN_DIRECTION_CHANGES = 3
MAX_ATTEMPTS = 400


def _collect_glyphs(left: np.ndarray, *, bg: int, div_color: int | None) -> list[dict[str, int]]:
    ignore = {int(bg), GRAY}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))
    return glyphs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode 3x3 glyphs into a colored polyline whose color depends on segment length.

    Input structure:
    - Left of a full-height divider: fixed 3x3 glyphs placed in two tile columns on a black background.
    - Right of the divider: exactly one gray start pixel whose shifted column becomes the polyline origin.

    Rule:
    - Remove the divider and relocate the gray start into the 7-column left canvas.
    - Read glyphs in column-major order (left tile column top-to-bottom, then right column top-to-bottom).
    - Each glyph area still determines the segment geometry, but now it also chooses the pre-segment downward step:
      area 4 -> step 1, down len 2; area 5 -> step 1, right len 3; area 6 -> step 2, left len 4; area 7 -> step 2, left len 2.
    - Draw each segment in the color assigned to its length only: len 2 -> 7, len 3 -> 8, len 4 -> 9.

    Key insight:
    The glyph area no longer maps straight to a colored segment. Instead it fans out into two attributes:
    area decides both the step size and the segment length/direction, and the segment length then decides color.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.full((H, int(div_c)), bg, dtype=int)

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

    for glyph in glyphs:
        area = int(glyph["area"])
        assert area in _AREA_TO_LEN and area in _AREA_TO_STEP, f"unexpected glyph area: {area}"

        step = int(_AREA_TO_STEP[area])
        seg_len = int(_AREA_TO_LEN[area])
        seg_color = int(_LEN_TO_OUT_COLOR[seg_len])

        r_cur = int(r_cur + step)
        assert r_cur < H, f"segment exits bottom after step: r_cur={r_cur}, H={H}, step={step}"
        out[r_cur, int(c_cur)] = seg_color

        if area == 4:
            end_r = r_cur + (seg_len - 1)
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = seg_color
        else:
            dir_h = int(_AREA_TO_DIR[area])
            end_c = int(c_cur) + dir_h * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[r_cur, c_cur] = seg_color

    return out


def _count_direction_changes(areas: list[int]) -> int:
    kinds = [int(a == 4) for a in areas]  # 1 = vertical, 0 = horizontal
    return sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1])


def _required_slots_remaining(counts: dict[int, int]) -> int:
    return sum(max(0, _MIN_COUNTS[a] - counts[a]) for a in _AREAS)


def _min_remaining_row_delta(counts: dict[int, int], slots_left: int) -> int:
    need4 = max(0, _MIN_COUNTS[4] - counts[4])
    need5 = max(0, _MIN_COUNTS[5] - counts[5])
    need6 = max(0, _MIN_COUNTS[6] - counts[6])
    need7 = max(0, _MIN_COUNTS[7] - counts[7])
    needed = need4 + need5 + need6 + need7
    if needed > slots_left:
        return 10**9
    forced = need4 * 2 + need5 * 1 + need6 * 2 + need7 * 2
    extra = slots_left - needed
    return forced + extra * 1


def _segment_cells(r: int, c: int, area: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    step = int(_AREA_TO_STEP[area])
    seg_len = int(_AREA_TO_LEN[area])
    start_r = int(r + step)

    if area == 4:
        cells = [(int(start_r + k), int(c)) for k in range(seg_len)]
        end = (int(start_r + seg_len - 1), int(c))
        return cells, end

    dir_h = int(_AREA_TO_DIR[area])
    cells = [(int(start_r), int(c + dir_h * k)) for k in range(seg_len)]
    end = (int(start_r), int(c + dir_h * (seg_len - 1)))
    return cells, end


def _sample_path(N: int, start_col: int) -> list[int] | None:
    H = 4 * N - 1
    total = 2 * N
    counts = {a: 0 for a in _AREAS}
    visited = {(GRAY_START_ROW, start_col)}

    def dfs(idx: int, r: int, c: int, areas: list[int]) -> list[int] | None:
        if idx == total:
            if _required_slots_remaining(counts) != 0:
                return None
            if _count_direction_changes(areas) < MIN_DIRECTION_CHANGES:
                return None
            return list(areas)

        slots_left = total - idx
        if slots_left < _required_slots_remaining(counts):
            return None
        if r + _min_remaining_row_delta(counts, slots_left) >= H:
            return None

        options: list[tuple[int, list[tuple[int, int]], tuple[int, int]]] = []
        for area in _AREAS:
            cells, end = _segment_cells(r, c, area)
            if any(not (0 <= rr < H and 0 <= cc < LEFT_W) for rr, cc in cells):
                continue
            if any((rr, cc) in visited for rr, cc in cells):
                continue
            options.append((area, cells, end))

        random.shuffle(options)
        options.sort(key=lambda item: counts[item[0]] >= _MIN_COUNTS[item[0]])

        for area, cells, end in options:
            counts[area] += 1
            if total - (idx + 1) < _required_slots_remaining(counts):
                counts[area] -= 1
                continue

            for cell in cells:
                visited.add(cell)
            areas.append(area)

            res = dfs(idx + 1, int(end[0]), int(end[1]), areas)
            if res is not None:
                return res

            areas.pop()
            for cell in cells:
                visited.remove(cell)
            counts[area] -= 1

        return None

    return dfs(0, GRAY_START_ROW, start_col, [])


def _render_grid(N: int, start_col: int, areas: list[int]) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        color = int(_AREA_TO_INPUT_COLOR[area])
        grid[r0:r0 + 3, tile_col:tile_col + 3] = _TEMPLATES[area] * color
    return grid


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    areas = _sample_path(N, start_col)
    if areas is None:
        return None
    return _render_grid(N, start_col, areas)


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
