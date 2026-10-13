from __future__ import annotations
import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg
import random


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a polyline by decoding 3x3 glyph instructions read across a divided grid.

    Input structure:
    - Left of a full-height vertical divider: 3x3 colored glyphs on bg.
    - Right of divider: a single gray (5) start pixel.

    Rule:
    - Remove the divider; shift the gray start into the left canvas.
    - Read glyphs in column-major order (left→right, top→bottom).
    - Each glyph's shape (pixel area) encodes a segment: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
    - Append segments sequentially, stepping down one cell between each.

    Key insight: Glyph pixel count (area) uniquely determines both segment length and direction — the shape itself is the instruction.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # Divider is a full-height constant non-bg column.
    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)

    # Move gray start from right side into left canvas (remove divider col too).
    gray = np.argwhere(g == 5)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = 5

    # Collect 3x3 glyph components on the left.
    left = g[:, : int(div_c)]
    ignore = {int(bg), 5}
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
            glyphs.append({"color": int(col), "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    area_to_len = {4: 2, 5: 3, 6: 4, 7: 2}
    area_to_dir = {5: 1, 6: -1, 7: -1}  # diamond right; cross/U left

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(area_to_len.get(area, 2))

        r_cur = int(r_cur + 1)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = color

        if area == 4:  # vertical (down)
            end_r = r_cur + (seg_len - 1)
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = color
        else:  # horizontal
            dir_h = int(area_to_dir.get(area, 1))
            end_c = int(c_cur) + dir_h * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[r_cur, c_cur] = color

    return out

# --------------- generator (attempt 1 (Claude)) ---------------

# --- canvas geometry ---
N_MIN, N_MAX = 2, 5            # number of tile rows; H = 4*N - 1, num_glyphs = 2*N
GRID_W = 15                    # full input width
DIVIDER_COL = 7                # divider position (col), full-height
DIVIDER_COLOR = 4              # divider fill color
LEFT_W = 7                     # left canvas width (cols 0..6)
TILE_COLS = (0, 4)             # 3x3 tile c0 positions on the left
TILE_ROW_STRIDE = 4            # row spacing between successive tile slots
GRAY = 5                       # start-pixel color
GRAY_START_ROW = 0             # gray start always sits at row 0 [observed]

# --- shape grammar (area -> 3x3 template, color, and (dr, dc) walk delta) ---
# Walk delta = total (row, col) displacement caused by one glyph instruction:
#   step down 1 then segment of (len, dir). E.g. area 4 (vertical len 2) => (+2, 0).
_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),  # plus    -> down 2,  color 6
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),  # diamond -> right 3, color 1
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),  # cross   -> left 4,  color 3
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),  # U       -> left 2,  color 2
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_DELTA = {4: (2, 0), 5: (1, 2), 6: (1, -3), 7: (1, -1)}
_AREAS = (4, 5, 6, 7)

# --- generation constraints ---
MIN_DIRECTION_CHANGES = 2      # min vertical<->horizontal flips across the walk
MAX_ATTEMPTS = 300             # outer retry budget


def _sample_path(N: int, start_col: int) -> list[int] | None:
    """Random-walk a sequence of 2*N area instructions that stays in the left canvas.

    Returns None if the walk dead-ends or fails the direction-change constraint.
    """
    H = 4 * N - 1
    r, c = GRAY_START_ROW, start_col
    areas: list[int] = []
    for _ in range(2 * N):
        opts = [a for a in _AREAS
                if r + _AREA_DELTA[a][0] < H and 0 <= c + _AREA_DELTA[a][1] < LEFT_W]
        if not opts:
            return None
        area = random.choice(opts)
        areas.append(area)
        dr, dc = _AREA_DELTA[area]
        r += dr
        c += dc

    types = [a == 4 for a in areas]  # True = vertical, False = horizontal
    changes = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])
    if changes < MIN_DIRECTION_CHANGES:
        return None
    return areas


def _render_grid(N: int, start_col: int, areas: list[int]) -> np.ndarray:
    """Paint divider, gray start, and all 2*N tiled glyphs into the input grid."""
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    # Column-major tiling: first N glyphs into TILE_COLS[0], next N into TILE_COLS[1].
    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        grid[r0:r0 + 3, tile_col:tile_col + 3] = _TEMPLATES[area] * _AREA_TO_COLOR[area]
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
            out = main(inp)  # oracle check
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
