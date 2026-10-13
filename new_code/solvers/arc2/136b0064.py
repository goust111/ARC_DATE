from __future__ import annotations
import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg


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

        if int(r_cur) + 1 >= H:
            break
        r_cur = int(r_cur + 1)
        out[int(r_cur), int(c_cur)] = int(color)

        if area == 4:  # vertical
            for _ in range(int(seg_len - 1)):
                if int(r_cur) + 1 >= H:
                    break
                r_cur = int(r_cur + 1)
                out[int(r_cur), int(c_cur)] = int(color)
        else:  # horizontal
            dir_h = int(area_to_dir.get(area, 1))
            if not (0 <= int(c_cur) + int(dir_h) * int(seg_len - 1) < int(div_c)):
                dir_h = int(-dir_h)
            for _ in range(int(seg_len - 1)):
                c_cur = int(c_cur + dir_h)
                out[int(r_cur), int(c_cur)] = int(color)

    return out

