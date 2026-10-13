from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLBOX_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, paste


BG = 0
BLUE = 1
RED = 2
MAGENTA = 6
PAYLOAD_COLORS = (3, 4, 5, 7, 8, 9)
FORBIDDEN = {BG, BLUE, RED, MAGENTA}

GEN_ATTEMPTS = 800


def _axis_info(grid: np.ndarray) -> tuple[int, int, int, int, bool]:
    marker_cells = np.argwhere(grid == MAGENTA)
    assert marker_cells.shape[0] == 1, "expected a unique axis marker"
    axis_r, axis_c = map(int, marker_cells[0])

    row_colors = sorted(set(map(int, grid[axis_r, :])) - {BG, MAGENTA})
    col_colors = sorted(set(map(int, grid[:, axis_c])) - {BG, MAGENTA})
    assert len(row_colors) == 1 and len(col_colors) == 1, "axis lines must be monochrome"

    h_color = int(row_colors[0])
    v_color = int(col_colors[0])
    assert {h_color, v_color} == {BLUE, RED}, "expected blue/red axis pair"
    main_vertical = v_color == BLUE
    return axis_r, axis_c, h_color, v_color, main_vertical


def _find_palette(grid: np.ndarray) -> dict:
    h, w = map(int, grid.shape)
    candidates: list[dict] = []
    for row in (0, h - 1):
        c = 0
        while c < w:
            if int(grid[row, c]) in FORBIDDEN:
                c += 1
                continue
            c0 = c
            colors: list[int] = []
            while c < w and int(grid[row, c]) not in FORBIDDEN:
                colors.append(int(grid[row, c]))
                c += 1
            if len(colors) >= 4 and len(set(colors)) == len(colors):
                candidates.append({"row": int(row), "c0": int(c0), "colors": np.array(colors, dtype=int)})
    assert len(candidates) == 1, "expected a unique edge palette strip"
    return candidates[0]


def _rect_key(box: tuple[int, int, int, int], axis_r: int, axis_c: int, main_vertical: bool) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    center_r2 = int(r0 + r1)
    center_c2 = int(c0 + c1)
    if main_vertical:
        return abs(center_c2 - 2 * int(axis_c)), abs(center_r2 - 2 * int(axis_r))
    return abs(center_r2 - 2 * int(axis_r)), abs(center_c2 - 2 * int(axis_c))


def _find_rectangles(grid: np.ndarray, palette: dict) -> list[dict]:
    mask = ~np.isin(grid, list(FORBIDDEN))
    row = int(palette["row"])
    c0 = int(palette["c0"])
    width = int(len(palette["colors"]))
    mask[row, c0 : c0 + width] = False

    rects: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        patch = crop_bbox(grid, box)
        colors = sorted(set(map(int, patch.flatten())) - {BG})
        assert len(colors) == 1, "each rectangle must be monochrome"
        assert int(colors[0]) not in FORBIDDEN
        assert int(coords.shape[0]) == int(patch.shape[0] * patch.shape[1]), "rectangles must be solid"
        rects.append(
            {
                "box": tuple(map(int, box)),
                "color": int(colors[0]),
                "arr": patch.copy(),
            }
        )

    assert 3 <= len(rects) <= 4, "expected 3-4 interior rectangles"
    return rects


def _stack_and_recolor(rect_arrays: list[np.ndarray], palette_colors: np.ndarray) -> np.ndarray:
    out_h = int(sum(int(arr.shape[0]) for arr in rect_arrays))
    out_w = int(max(int(arr.shape[1]) for arr in rect_arrays))
    out = np.zeros((out_h, out_w), dtype=int)
    cursor = 0
    for arr in rect_arrays:
        paste(out, arr, int(cursor), 0)
        cursor += int(arr.shape[0])

    palette_len = int(len(palette_colors))
    for c in range(out_w):
        color = int(palette_colors[c % palette_len])
        col_mask = out[:, c] != BG
        out[col_mask, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort rectangles by distance to the blue axis, optionally rotate them, then recolor the stacked result column-by-column with the edge palette.

    Input structure:
    - Black background with one horizontal axis line and one vertical axis line crossing near a corner.
    - The crossing pixel is magenta; the two axis colors are blue and red, with blue defining the main axis.
    - The interior contains 3-4 disjoint solid rectangles, plus a 1-row multicolor palette strip attached to the top or bottom edge.

    Rule:
    - Locate the magenta crossing, determine which axis is blue, and read the edge palette from the unique 1-row multicolor strip.
    - Compute each rectangle's perpendicular center-distance to the blue axis; break ties by the perpendicular center-distance to the red axis.
    - If the blue axis is vertical, rotate every rectangle 90 degrees counterclockwise; otherwise keep them as-is.
    - Stack the ordered rectangles top-to-bottom with left edges aligned, then recolor every non-black cell in column i with palette[i mod palette_width].

    Key insight:
    - The blue/red axis pair supplies an oriented coordinate system: it decides both the sorting distances and whether rotation happens, while the final stacked geometry decides how the palette cycles across output columns.
    """
    g = np.asarray(grid, dtype=int)
    axis_r, axis_c, _h_color, _v_color, main_vertical = _axis_info(g)
    palette = _find_palette(g)
    rects = _find_rectangles(g, palette)

    keys = [_rect_key(rect["box"], axis_r, axis_c, main_vertical) for rect in rects]
    assert len(keys) == len(set(keys)), "rectangle ordering must be unique"

    rects = sorted(rects, key=lambda rect: _rect_key(rect["box"], axis_r, axis_c, main_vertical))
    rect_arrays: list[np.ndarray] = []
    for rect in rects:
        arr = rect["arr"]
        if main_vertical:
            arr = np.rot90(arr, k=1)
        rect_arrays.append(arr)
    return _stack_and_recolor(rect_arrays, palette["colors"])


def _apply_flips(grid: np.ndarray, flip_v: bool, flip_h: bool) -> np.ndarray:
    out = np.array(grid, copy=True)
    if flip_v:
        out = np.flipud(out)
    if flip_h:
        out = np.fliplr(out)
    return out


def _transform_rect_specs(rects: list[dict], h: int, w: int, flip_v: bool, flip_h: bool) -> list[dict]:
    out: list[dict] = []
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        rh = int(rect["h"])
        rw = int(rect["w"])
        if flip_v:
            r0 = int(h - r0 - rh)
        if flip_h:
            c0 = int(w - c0 - rw)
        out.append({"r0": r0, "c0": c0, "h": rh, "w": rw, "color": int(rect["color"])})
    return out


def _expected_from_specs(
    rects: list[dict],
    axis_r: int,
    axis_c: int,
    main_vertical: bool,
    palette_colors: np.ndarray,
) -> np.ndarray:
    ordered = sorted(
        rects,
        key=lambda rect: _rect_key(
            (int(rect["r0"]), int(rect["c0"]), int(rect["r0"] + rect["h"] - 1), int(rect["c0"] + rect["w"] - 1)),
            axis_r,
            axis_c,
            main_vertical,
        ),
    )
    arrays: list[np.ndarray] = []
    for rect in ordered:
        arr = np.full((int(rect["h"]), int(rect["w"])), int(rect["color"]), dtype=int)
        if main_vertical:
            arr = np.rot90(arr, k=1)
        arrays.append(arr)
    return _stack_and_recolor(arrays, palette_colors)


def _band_partition(rect_count: int, tie_case: bool) -> list[int]:
    if rect_count == 3:
        options = ([2, 1], [1, 1, 1])
    else:
        options = ([2, 2], [2, 1, 1], [1, 1, 1, 1])
    eligible = [list(option) for option in options if (max(option) > 1) == tie_case]
    return random.choice(eligible)


def _build_horizontal_rects(h: int, w: int, axis_r: int, axis_c: int, rect_colors: list[int], palette_len: int, wrap_case: bool, tie_case: bool) -> list[dict] | None:
    band_sizes = _band_partition(len(rect_colors), tie_case)
    specs: list[dict] = []
    r_cursor = int(axis_r + 2 + random.randint(0, 1))
    idx = 0
    wrap_used = not wrap_case

    for band_size in band_sizes:
        rect_h = random.randint(2, 4)
        widths = [random.randint(2, 5) for _ in range(int(band_size))]
        if wrap_case and not wrap_used:
            widths[random.randrange(int(band_size))] = 5
            wrap_used = True

        gaps = [random.randint(2, 3) for _ in range(int(band_size) - 1)]
        total_w = int(sum(widths) + sum(gaps))
        min_c = int(axis_c + 2)
        max_c = int(w - 2 - total_w)
        if max_c < min_c or r_cursor + rect_h - 1 > h - 2:
            return None

        band_c = random.randint(min_c, max_c)
        c = int(band_c)
        for j in range(int(band_size)):
            specs.append({"r0": int(r_cursor), "c0": int(c), "h": int(rect_h), "w": int(widths[j]), "color": int(rect_colors[idx])})
            c += int(widths[j])
            if j < int(band_size) - 1:
                c += int(gaps[j])
            idx += 1
        r_cursor += int(rect_h + random.randint(2, 3))

    if idx != len(rect_colors) or not wrap_used:
        return None
    return specs


def _build_vertical_rects(h: int, w: int, axis_r: int, axis_c: int, rect_colors: list[int], tie_case: bool) -> list[dict] | None:
    band_sizes = _band_partition(len(rect_colors), tie_case)
    specs: list[dict] = []
    c_cursor = int(axis_c + 2 + random.randint(0, 1))
    idx = 0

    for band_size in band_sizes:
        rect_w = random.randint(2, 5)
        heights = [random.randint(2, 4) for _ in range(int(band_size))]
        gaps = [random.randint(2, 3) for _ in range(int(band_size) - 1)]
        total_h = int(sum(heights) + sum(gaps))
        min_r = int(axis_r + 2)
        max_r = int(h - 2 - total_h)
        if max_r < min_r or c_cursor + rect_w - 1 > w - 2:
            return None

        band_r = random.randint(min_r, max_r)
        r = int(band_r)
        for j in range(int(band_size)):
            specs.append({"r0": int(r), "c0": int(c_cursor), "h": int(heights[j]), "w": int(rect_w), "color": int(rect_colors[idx])})
            r += int(heights[j])
            if j < int(band_size) - 1:
                r += int(gaps[j])
            idx += 1
        c_cursor += int(rect_w + random.randint(2, 3))

    if idx != len(rect_colors):
        return None
    if all(int(rect["h"]) == int(rect["w"]) for rect in specs):
        return None
    return specs


def _render_canonical(
    h: int,
    w: int,
    axis_r: int,
    axis_c: int,
    main_vertical: bool,
    palette_row: int,
    palette_c0: int,
    palette_colors: np.ndarray,
    rects: list[dict],
) -> np.ndarray:
    grid = np.zeros((int(h), int(w)), dtype=int)
    h_axis_color = RED if main_vertical else BLUE
    v_axis_color = BLUE if main_vertical else RED
    grid[int(axis_r), :] = int(h_axis_color)
    grid[:, int(axis_c)] = int(v_axis_color)
    grid[int(axis_r), int(axis_c)] = MAGENTA
    grid[int(palette_row), int(palette_c0) : int(palette_c0) + int(len(palette_colors))] = palette_colors
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        rh = int(rect["h"])
        rw = int(rect["w"])
        grid[r0 : r0 + rh, c0 : c0 + rw] = int(rect["color"])
    return grid


def _metadata_consistent(
    grid: np.ndarray,
    rects: list[dict],
    axis_r: int,
    axis_c: int,
    main_vertical: bool,
    palette_row: int,
    palette_c0: int,
    palette_colors: np.ndarray,
) -> bool:
    try:
        parsed_axis_r, parsed_axis_c, _h_color, _v_color, parsed_main_vertical = _axis_info(grid)
        parsed_palette = _find_palette(grid)
        parsed_rects = _find_rectangles(grid, parsed_palette)
    except Exception:
        return False

    if parsed_axis_r != int(axis_r) or parsed_axis_c != int(axis_c):
        return False
    if parsed_main_vertical != bool(main_vertical):
        return False
    if int(parsed_palette["row"]) != int(palette_row) or int(parsed_palette["c0"]) != int(palette_c0):
        return False
    if not np.array_equal(parsed_palette["colors"], palette_colors):
        return False

    expected_boxes = {
        int(rect["color"]): (int(rect["r0"]), int(rect["c0"]), int(rect["r0"] + rect["h"] - 1), int(rect["c0"] + rect["w"] - 1))
        for rect in rects
    }
    parsed_boxes = {int(rect["color"]): tuple(map(int, rect["box"])) for rect in parsed_rects}
    return expected_boxes == parsed_boxes


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    axis_r = random.randint(1, 3)
    axis_c = random.randint(1, 3)
    main_vertical = random.choice([False, True])
    rect_count = random.choice([3, 4])
    tie_case = random.random() < 0.65
    wrap_case = (not main_vertical) and (random.random() < 0.65)
    palette_len = 4 if wrap_case else random.randint(4, 6)
    palette_colors = np.array(random.sample(PAYLOAD_COLORS, palette_len), dtype=int)
    rect_colors = random.sample(PAYLOAD_COLORS, rect_count)
    palette_row = random.choice([0, h - 1])

    min_palette_c = int(axis_c + 2)
    max_palette_c = int(w - palette_len - 1)
    if max_palette_c < min_palette_c:
        return None
    palette_c0 = random.randint(min_palette_c, max_palette_c)

    if main_vertical:
        rects = _build_vertical_rects(h, w, axis_r, axis_c, rect_colors, tie_case)
    else:
        rects = _build_horizontal_rects(h, w, axis_r, axis_c, rect_colors, palette_len, wrap_case, tie_case)
    if rects is None:
        return None

    canonical = _render_canonical(h, w, axis_r, axis_c, main_vertical, palette_row, palette_c0, palette_colors, rects)
    flip_v = random.choice([False, True])
    flip_h = random.choice([False, True])
    grid = _apply_flips(canonical, flip_v, flip_h)

    final_axis_r = int(h - 1 - axis_r) if flip_v else int(axis_r)
    final_axis_c = int(w - 1 - axis_c) if flip_h else int(axis_c)
    final_palette_row = int(h - 1 - palette_row) if flip_v else int(palette_row)
    final_palette_c0 = int(w - palette_c0 - palette_len) if flip_h else int(palette_c0)
    final_palette = palette_colors[::-1].copy() if flip_h else palette_colors.copy()
    final_rects = _transform_rect_specs(rects, h, w, flip_v, flip_h)

    keys = [
        _rect_key(
            (int(rect["r0"]), int(rect["c0"]), int(rect["r0"] + rect["h"] - 1), int(rect["c0"] + rect["w"] - 1)),
            final_axis_r,
            final_axis_c,
            main_vertical,
        )
        for rect in final_rects
    ]
    if len(keys) != len(set(keys)):
        return None

    expected = _expected_from_specs(final_rects, final_axis_r, final_axis_c, main_vertical, final_palette)
    if max(expected.shape) > 30:
        return None
    if wrap_case and expected.shape[1] <= palette_len:
        return None

    if not _metadata_consistent(
        grid,
        final_rects,
        final_axis_r,
        final_axis_c,
        main_vertical,
        final_palette_row,
        final_palette_c0,
        final_palette,
    ):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
