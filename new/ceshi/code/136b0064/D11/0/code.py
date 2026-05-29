# Variant of 136b0064
# Dimension: D11 — 符号编码——字形区域的辅助信息 (symbol_instructions)
# Change: 无额外符号编码；字形形状是唯一的指令来源，颜色严格绑定形状... → 右侧区域包含图例（legend）映射表

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the D11 variant
# ----------------------------------
# a) Concreteization and behavior trace
#    - Left side stays the same visual family as the source task: two columns of 3x3 glyph tiles on a 7-column canvas.
#    - The core change is on the right side. Instead of "only one gray start pixel", the right side now contains:
#      1. the gray start pixel on row 0, and
#      2. a 4-row legend table below it. Each legend entry occupies one 3x7 strip:
#         cols 8..10 = one 3x3 glyph sample, col 11 = spacer, cols 12..14 = one 3x3 arrow sample.
#    - The glyph sample tells "which shape this row is about"; the arrow sample tells the direction currently assigned
#      to that shape in this instance. Length still comes from the shape's area, so the legend only reassigns direction.
#    - Example trace:
#      original behavior: a blue diamond (area 5) always means "go right length 3".
#      variant behavior: if the right legend shows the diamond next to a left-arrow sample, then the same blue diamond
#      now means "go left length 3". The glyph color still becomes the segment color in the output.
#    - Flow impact:
#      input construction changes because the generator must synthesize a legend;
#      solver changes because `area_to_dir` can no longer be hard-coded;
#      output drawing mostly stays the same after direction lookup is replaced.
#
# b) Complexity calibration
#    - Keep the old four glyph shapes and their old area-derived lengths {4:2, 5:3, 6:4, 7:2}.
#    - Only directions are remapped, using the multiset {down, right, left, left}. This keeps the path language close
#      to the source task and avoids exploding the space of behaviors.
#    - The legend always contains exactly four entries and the path uses all four shapes at least once, so the new
#      mechanism is visible and repeated within each instance.
#
# c) Hidden assumption audit against the original `main()`
#    - Divider detection still works: the full-height constant non-bg column is unchanged.
#    - Gray start handling still works: there is still exactly one gray pixel on the right.
#    - The old line `area_to_dir = {5: 1, 6: -1, 7: -1}` is invalid now because direction is no longer fixed by area.
#    - The old strict shape-color binding is invalid now, so left-glyph color must be treated as draw color only.
#    - Any solver logic that tries to infer meaning from glyph color would be wrong in this variant.
#
# d) Placement legality
#    - To fit four 3x7 legend rows cleanly inside the original 15-column grid, the generator restricts N to 4..5,
#      so H = 15 or 19. This is the smallest adaptation needed to make the legend readable without widening the grid.
#    - Legend rows start at 2, 5, 8, 11; row 0 is reserved for the gray start and row 1 is blank.
#    - Left glyph tiles remain non-overlapping because the original 3x3-at-(c0 in {0,4}, r0 every 4 rows) layout stays.
#
# e) Visual observability
#    - The same glyph shape can point to different directions across examples, and the legend shows that explicitly.
#    - Left glyph colors are sampled independently from shape, so the old "pink plus / blue diamond / green cross / red U"
#      shortcut is visibly broken.
#    - Legend row order is shuffled per sample so the mapping is read from the table itself, not from row position.
#
# f) Visual cleanliness
#    - The right side uses fixed strips with an empty spacer column between shape and arrow, so the two symbol systems do
#      not merge into one blob.
#    - The legend uses small 3x3 arrow exemplars rather than extra text-like noise, keeping the scene readable.
#
# g) Color-change inferability
#    - Output color still comes directly from the left glyph's own color, so the new rule does not add a second opaque
#      color transform. The only new indirection is shape -> direction via the legend.
#    - The generator enforces non-trivial color/shape mixing so the dataset does not accidentally collapse back to the
#      source task's fixed binding.
#
# h) New implicit assumptions introduced by this solver
#    - The solver assumes the legend occupies the fixed four row bands listed in `LEGEND_ROW0S`.
#    - It assumes each legend strip contains exactly one 3x3 glyph patch at local cols 0..2 and one 3x3 arrow patch at
#      local cols 4..6, with bg elsewhere in the strip.
#    - The generator guarantees those assumptions directly.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper: find a column where all values are identical and != bg.
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


# --- shared geometry ---
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
RIGHT_W = GRID_W - DIVIDER_COL - 1
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0
N_MIN, N_MAX = 4, 5

# Right-side legend layout: one 3x3 glyph sample + spacer + one 3x3 arrow sample.
LEGEND_ROW0S = (2, 5, 8, 11)
LEGEND_SHAPE_C0 = 0
LEGEND_ARROW_C0 = 4

# Shape grammar from the source task. Area still determines segment length.
_SHAPE_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREAS = (4, 5, 6, 7)

# Legend arrow exemplars. Only the mask matters to the solver; colors are just for readability.
_DIR_TEMPLATES = {
    "D": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
    "R": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int),
    "L": np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
}
_DIR_TO_STEP = {"D": (1, 0), "R": (0, 1), "L": (0, -1)}

# Fixed legend styling; left-glyph colors are sampled independently.
_LEGEND_SHAPE_COLORS = {4: 6, 5: 7, 6: 8, 7: 9}
_LEGEND_ARROW_COLORS = {"D": 8, "R": 1, "L": 2}

# Original task's color binding, kept only to prevent generator collapse back to the source behavior.
_ORIG_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_ORIG_AREA_TO_DIR = {4: "D", 5: "R", 6: "L", 7: "L"}
_DRAW_COLORS = (1, 2, 3, 6, 7, 8, 9)

MAX_ATTEMPTS = 400
MIN_DIRECTION_CHANGES = 2


def _decode_dir_patch(patch: np.ndarray, bg: int) -> str:
    mask = (np.asarray(patch, dtype=int) != int(bg)).astype(int)
    for name, tmpl in _DIR_TEMPLATES.items():
        if np.array_equal(mask, tmpl):
            return name
    raise AssertionError(f"unrecognized legend arrow patch:\n{mask}")


def _read_legend(right: np.ndarray, bg: int) -> dict[int, str]:
    area_to_dir: dict[int, str] = {}
    for r0 in LEGEND_ROW0S:
        if r0 + 2 >= right.shape[0]:
            continue
        shape_patch = right[r0 : r0 + 3, LEGEND_SHAPE_C0 : LEGEND_SHAPE_C0 + 3]
        arrow_patch = right[r0 : r0 + 3, LEGEND_ARROW_C0 : LEGEND_ARROW_C0 + 3]
        area = int(np.count_nonzero(shape_patch != bg))
        if area == 0:
            continue
        assert area in _AREAS, f"legend shape area must be one of {_AREAS}, got {area}"
        area_to_dir[area] = _decode_dir_patch(arrow_patch, bg)
    assert set(area_to_dir) == set(_AREAS), f"incomplete legend: got keys {sorted(area_to_dir)}"
    return area_to_dir


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a polyline by decoding 3x3 glyphs through a right-side legend.

    Input structure:
    - Left of a full-height divider: 3x3 colored glyphs on bg, arranged in the same two tile columns as the source task.
    - Right of the divider: one gray start pixel plus a four-entry legend; each legend row shows a glyph sample and a 3x3 arrow sample.

    Rule:
    - Remove the divider and shift the gray start into the left canvas.
    - Read the legend first: a glyph's area still fixes its segment length, but the legend arrow for that shape now decides whether it goes down, right, or left.
    - Read left-side glyphs in column-major order (left→right, top→bottom); for each glyph step down one cell, then draw its segment in the glyph's own color.

    Key insight: Shape is no longer the whole instruction. The right-side legend reassigns directions per instance, so the same glyph can mean a different move in different inputs.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None and int(div_c) == DIVIDER_COL

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
    right = g[:, int(div_c) + 1 :]
    area_to_dir = _read_legend(right, bg)

    ignore = {int(bg), GRAY}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"color": color, "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(_AREA_TO_LEN[area])
        direction = area_to_dir[area]

        r_cur = int(r_cur + 1)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = color

        if direction == "D":
            end_r = r_cur + (seg_len - 1)
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = color
        else:
            dir_h = int(_DIR_TO_STEP[direction][1])
            end_c = int(c_cur) + dir_h * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={direction}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[r_cur, c_cur] = color

    return out


def _sample_dir_map() -> dict[int, str]:
    while True:
        dirs = ["D", "R", "L", "L"]
        random.shuffle(dirs)
        area_to_dir = {area: dirs[i] for i, area in enumerate(_AREAS)}
        dist = sum(int(area_to_dir[a] != _ORIG_AREA_TO_DIR[a]) for a in _AREAS)
        if dist >= 2:
            return area_to_dir


def _area_delta(area: int, direction: str) -> tuple[int, int]:
    seg_len = int(_AREA_TO_LEN[area])
    if direction == "D":
        return seg_len, 0
    if direction == "R":
        return 1, seg_len - 1
    return 1, -(seg_len - 1)


def _sample_path(N: int, start_col: int, area_to_dir: dict[int, str]) -> list[int] | None:
    H = 4 * N - 1
    r, c = GRAY_START_ROW, start_col
    areas: list[int] = []
    counts = {a: 0 for a in _AREAS}
    prev_kind: str | None = None
    changes = 0

    for _ in range(2 * N):
        opts = []
        for area in _AREAS:
            dr, dc = _area_delta(area, area_to_dir[area])
            if r + dr < H and 0 <= c + dc < LEFT_W:
                opts.append(area)
        if not opts:
            return None

        def score(area: int) -> tuple[int, int, float]:
            dir_kind = area_to_dir[area]
            missing_bonus = 0 if counts[area] else 1
            change_bonus = 0 if prev_kind is None or dir_kind == prev_kind else 1
            return (missing_bonus, change_bonus, random.random())

        area = max(opts, key=score)
        areas.append(area)
        counts[area] += 1
        dir_kind = area_to_dir[area]
        if prev_kind is not None and dir_kind != prev_kind:
            changes += 1
        prev_kind = dir_kind
        dr, dc = _area_delta(area, dir_kind)
        r += dr
        c += dc

    if any(counts[a] == 0 for a in _AREAS):
        return None
    if changes < MIN_DIRECTION_CHANGES:
        return None
    return areas


def _sample_glyph_colors(areas: list[int]) -> list[int] | None:
    for _ in range(80):
        colors = [random.choice(_DRAW_COLORS) for _ in areas]
        shape_to_colors: dict[int, set[int]] = {a: set() for a in _AREAS}
        color_to_shapes: dict[int, set[int]] = {}
        mismatch_count = 0
        for area, color in zip(areas, colors):
            shape_to_colors[area].add(int(color))
            color_to_shapes.setdefault(int(color), set()).add(int(area))
            if int(color) != int(_ORIG_AREA_TO_COLOR[area]):
                mismatch_count += 1
        diversified_shape = any(len(cols) >= 2 for cols in shape_to_colors.values())
        shared_color = any(len(shapes) >= 2 for shapes in color_to_shapes.values())
        if mismatch_count >= len(areas) // 2 and (diversified_shape or shared_color):
            return colors
    return None


def _render_grid(
    N: int,
    start_col: int,
    areas: list[int],
    glyph_colors: list[int],
    area_to_dir: dict[int, str],
    legend_order: list[int],
) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _SHAPE_TEMPLATES[area] * int(glyph_colors[i])

    for idx, area in enumerate(legend_order):
        r0 = LEGEND_ROW0S[idx]
        shape_color = int(_LEGEND_SHAPE_COLORS[area])
        direction = area_to_dir[area]
        arrow_color = int(_LEGEND_ARROW_COLORS[direction])
        grid[r0 : r0 + 3, DIVIDER_COL + 1 + LEGEND_SHAPE_C0 : DIVIDER_COL + 1 + LEGEND_SHAPE_C0 + 3] = (
            _SHAPE_TEMPLATES[area] * shape_color
        )
        grid[r0 : r0 + 3, DIVIDER_COL + 1 + LEGEND_ARROW_C0 : DIVIDER_COL + 1 + LEGEND_ARROW_C0 + 3] = (
            _DIR_TEMPLATES[direction] * arrow_color
        )

    return grid


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    area_to_dir = _sample_dir_map()
    areas = _sample_path(N, start_col, area_to_dir)
    if areas is None:
        return None
    glyph_colors = _sample_glyph_colors(areas)
    if glyph_colors is None:
        return None
    legend_order = random.sample(list(_AREAS), k=len(_AREAS))
    return _render_grid(N, start_col, areas, glyph_colors, area_to_dir, legend_order)


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
