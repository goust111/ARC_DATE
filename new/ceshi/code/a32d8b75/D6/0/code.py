# Variant of a32d8b75
# Dimension: D6 — 符号编码：位置标记的编码方式 (symbol_instructions)
# Change: 5x5边框6方块内3x3区域中的单个颜色4像素，行列位置(0-2)编码锚点对齐（起始/中间/结束）... → 二进制位置编码：多个标记像素的排列组合

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for this D6 variant
# -----------------------------------
# spec:
#   mutation_core:
#     marker_square remains a 5x5 color-6 border, but its 3x3 inner area no longer
#     contains a single location pixel. Instead:
#       - inner center (1,1) is always color 4 as a visible reference/sentinel
#       - inner left column (rows 0,1,2) stores row bits 1,2,4
#       - inner right column (rows 0,1,2) stores col bits 1,2,4
#       - inner top-middle and bottom-middle stay 0 for cleanliness
#     Therefore a marker can contain 1-7 colored cells inside the border.
#   position_decode:
#     row_code = sum(bit_i * 2^i for the three left-column cells)
#     col_code = sum(bit_i * 2^i for the three right-column cells)
#     The codes directly mean the composite's top-left row/col offset inside the
#     output canvas. Generator guarantees 0 <= row_code <= grid_h - comp_h and
#     0 <= col_code <= canvas_w - comp_w, so placement is exact and unclipped.
#   unchanged_parts:
#     unit-sprite extraction, color swap, frame tiling, rotation clock-hand, canvas
#     cropping, and non-overlap rules are preserved from the source task.
#
# a) concretization + behavior trace:
#   Example marker:
#     6 6 6 6 6
#     6 4 0 0 6   left-top bit => +1 row
#     6 0 4 4 6   center sentinel + right-mid bit => +2 col
#     6 4 0 4 6   left-bot bit => +4 row, right-bot bit => +4 col
#     6 6 6 6 6
#   This decodes to row_code = 1+4 = 5, col_code = 2+4 = 6, so the composite is
#   placed with top-left at (5, 6). Under the original logic, only the first nonzero
#   inner pixel would have been read, collapsing this marker to a coarse 3-state anchor.
#
# b) complexity calibration:
#   Three bits per axis yields 8 values, which is visibly richer than the original
#   3-way anchor without exploding the search space. Two legends remain common so the
#   train set can show repeated evidence for the same encoding convention.
#
# c) hidden-assumption audit against the old main():
#   - _position_indices_from_border_square assumed marker uniqueness; invalid now.
#   - _anchor_to_top_left assumed 3 semantic anchors; invalid now.
#   - The write loop assumed computed top/left already fit; still valid only if the
#     generator constrains binary codes against composite size. The new generator does so.
#   - Marker-panel detection needed a nonempty inner 3x3. The always-on center sentinel
#     preserves that even when both binary codes are zero.
#
# d) placement legality:
#   The generator computes each composite size first, then enumerates only binary codes
#   that keep the composite within bounds. For two legends it searches non-overlapping
#   code pairs before committing them to the marker square.
#
# e) visual observability:
#   The row and col bits live in separate vertical stacks, so a solver can correlate the
#   left stack with vertical displacement and the right stack with horizontal displacement.
#   Preferred 2-legend samples often share one axis and differ on the other, making the
#   axis-specific effect easier to spot from a few examples.
#
# f) visual cleanliness:
#   Only color 4 appears inside the marker square, and the middle column stays sparse.
#   The marker remains easy to distinguish from the unit sprite, frame, and rotation mark.
#
# g) color-change audit:
#   No new color mapping rule is introduced. The only color behavior remains the original
#   unit-sprite swap, so D6 is isolated to symbol encoding rather than palette logic.
#
# h) post-change main() assumptions that the generator must satisfy:
#   - center sentinel at inner (1,1) is present
#   - row/col codes do not exceed legal top-left offsets
#   - 2-legend placements do not overlap after exact-offset decoding
#   The generator enforces all three explicitly.


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    h, w = g.shape
    cols6 = [c for c in range(w) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
    cols6.sort()
    runs: list[tuple[int, int]] = []
    s = e = cols6[0]
    for c in cols6[1:]:
        if c == e + 1:
            e = c
        else:
            runs.append((s, e))
            s = e = c
    runs.append((s, e))
    return runs


def _split_canvas(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """
    Identify a left separator run (between left legend and canvas), and optionally a
    right separator run (between canvas and a right-side legend panel).
    """
    h, w = g.shape
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= w - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 5x5 border-6 squares that look like the top of an 8x5 marker panel."""
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (
                np.all(sub[0, :] == 6)
                and np.all(sub[4, :] == 6)
                and np.all(sub[:, 0] == 6)
                and np.all(sub[:, 4] == 6)
            ):
                continue
            inner = sub[1:4, 1:4]
            if not np.any((inner != 0) & (inner != 6)):
                continue
            rot = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
            if rot.shape != (3, 3) or not np.any((rot != 0) & (rot != 6)):
                continue
            out.append((int(r0), int(c0)))
    out.sort()
    return out


def _extract_unit_sprite(legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    # Unit sprite is the top-most 4-connected component of pixels not in {0,6}.
    mask = (legend != 0) & (legend != 6)
    pts_all = np.argwhere(mask)
    if pts_all.size == 0:
        return None
    rmin = int(pts_all[:, 0].min())
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])
    h, w = legend.shape
    vis = np.zeros((h, w), dtype=bool)
    st = [(rmin, cseed)]
    vis[rmin, cseed] = True
    pts: list[tuple[int, int]] = []
    while st:
        r, c = st.pop()
        pts.append((r, c))
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not vis[rr, cc]:
                vis[rr, cc] = True
                st.append((rr, cc))
    r0, c0, r1, c1 = bbox_coords(np.asarray(pts))
    spr = legend[r0 : r1 + 1, c0 : c1 + 1].copy()
    cols = sorted({int(x) for x in np.unique(spr) if int(x) not in (0, 6)})
    if len(cols) == 2:
        a, b = cols
        spr2 = spr.copy()
        spr2[spr == a] = -1
        spr2[spr == b] = a
        spr2[spr2 == -1] = b
        spr = spr2
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
    # Frame pixels live between the unit sprite and the border6 square.
    r_start = unit_bb[2] + 1
    r_end = border6_r0 - 1
    if r_start > r_end:
        return None
    sub = legend[r_start : r_end + 1, :]
    mask = (sub != 0) & (sub != 6)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    pts[:, 0] += r_start
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return (r0, c0, r1, c1)


def _decode_binary_offsets_from_border_square(
    g: np.ndarray, border6_r0: int, border6_c0: int
) -> tuple[int, int]:
    sq = g[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    center = int(inner[1, 1])
    assert center not in (0, 6), "binary marker square must contain a center sentinel"
    row_code = 0
    col_code = 0
    for bit in range(3):
        if int(inner[bit, 0]) not in (0, 6):
            row_code |= 1 << bit
        if int(inner[bit, 2]) not in (0, 6):
            col_code |= 1 << bit
    return row_code, col_code


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Rotation marker is in the bottom 3x3 at rows (border6_r0+5 .. border6_r0+7) and cols (1..3),
    drawn with non-{0,6} pixels (typically 7). Use the furthest pixel from the center (1,1)
    to determine direction:
      up -> 0, right -> 1, down -> 2, left -> 3.
    """
    h, w = legend.shape
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = legend[r0 : r0 + 3, c0 : c0 + 3]
    pts = np.argwhere((sub != 0) & (sub != 6))
    assert pts.size > 0, "no rotation marker pixel found"
    center = np.array([1, 1], dtype=int)
    dists = np.abs(pts - center).sum(axis=1)
    p = pts[int(dists.argmax())]
    dr, dc = int(p[0] - center[0]), int(p[1] - center[1])
    if abs(dr) >= abs(dc):
        if dr < 0:
            return 0
        if dr > 0:
            return 2
    else:
        if dc > 0:
            return 1
        if dc < 0:
            return 3
    raise AssertionError("rotation marker pixel at center, direction ambiguous")


def _offset_from_binary_code(code: int, size: int, place_size: int) -> int:
    """
    Binary code directly stores the top-left offset in pixels.

    The generator guarantees code <= size - place_size, so solver placement remains exact.
    """
    max_offset = size - place_size
    assert max_offset >= 0, "composite larger than canvas"
    assert 0 <= code <= min(7, max_offset), "binary offset outside legal range"
    return code


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble and place enlarged sprites on a canvas using legend-panel instructions with binary offsets.

    Input structure:
    - Color-6 separator columns split the grid into a left legend and a main canvas, with an optional right legend.
    - Each legend panel still contains a unit sprite, a frame template, a 5x5 bordered marker square, and a 3x3 rotation clock-hand.
    - The marker square's 3x3 interior now uses multiple color-4 pixels: the left column encodes a 3-bit row offset, the right column encodes a 3-bit column offset, and the center is a fixed reference pixel.

    Rule:
    - Tile the unit sprite into every active frame cell, then rotate that composite according to the clock-hand.
    - Decode the binary marker bits into exact top-left row/column offsets inside the output canvas.
    - Stamp the rotated composite at that decoded offset, preserving any color-6 separator cells in the canvas.

    Key insight:
    - The position marker is no longer "which of 9 cells is filled"; it is a pair of binary stacks whose on/off combinations specify the row and column offsets independently.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main_canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = main_canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(main_canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(main_canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = main_canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    for marker_r0, marker_c0 in marker_squares:
        if canvas_c0 <= marker_c0 < canvas_c1:
            continue
        if marker_c0 + 5 > w:
            continue
        panel = g[:, marker_c0 : marker_c0 + 5]

        unit = _extract_unit_sprite(panel)
        if unit is None:
            continue
        unit_sprite, unit_bb = unit

        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)

        m_unit = (unit_sprite != 0) & (unit_sprite != 6)
        uh, uw = unit_sprite.shape
        fh, fw = frame_mask.shape
        comp = np.zeros((fh * uh, fw * uw), dtype=int)
        for i in range(fh):
            for j in range(fw):
                if not frame_mask[i, j]:
                    continue
                rr0 = i * uh
                cc0 = j * uw
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][m_unit] = unit_sprite[m_unit]

        row_code, col_code = _decode_binary_offsets_from_border_square(g, marker_r0, marker_c0)
        top = _offset_from_binary_code(row_code, eff_h, comp.shape[0])
        left = _offset_from_binary_code(col_code, eff_w, comp.shape[1])

        ch, cw = comp.shape
        assert top + ch <= eff_h and left + cw <= eff_w, "binary placement exceeds effective canvas"
        for r in range(ch):
            rr = top + r
            assert 0 <= rr < out.shape[0], "sprite row out of canvas bounds"
            for c in range(cw):
                cc = left + c
                assert 0 <= cc < out.shape[1], "sprite col out of canvas bounds"
                if out[rr, cc] == 6:
                    continue
                val = int(comp[r, c])
                if val != 0 and val != 6:
                    assert not composite_written[rr, cc], "composites must not overlap"
                    composite_written[rr, cc] = True
                    out[rr, cc] = val
    return out


# --------------- generator (binary_offset_markers) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 1200


def _allocate_colors(num_legends: int):
    """Pick unit colors (2 per legend), 2 background colors, and 1 frame color per legend."""
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets = []
    used_unit = set()
    for _ in range(num_legends):
        if len(pool) < 2:
            return None
        c1, c2 = pool.pop(), pool.pop()
        unit_color_sets.append((c1, c2))
        used_unit.update([c1, c2])

    bg_cands = [c for c in pool if c not in used_unit]
    if len(bg_cands) < 2:
        return None
    random.shuffle(bg_cands)
    bg_colors = (bg_cands[0], bg_cands[1])

    frame_colors = []
    all_set = set(USABLE_COLORS)
    for uc in unit_color_sets:
        candidates = list(all_set - set(uc))
        frame_colors.append(random.choice(candidates))

    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _design_legend(
    leg: dict,
    unit_colors,
    frame_color,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
    prefer_small: bool = False,
) -> bool:
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((n, n), uc1, dtype=int)
    num_c2 = random.randint(1, n * n - 1)
    for p in random.sample(range(n * n), num_c2):
        sprite[p // n, p % n] = uc2
    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (n + 1)
    combos = []
    for f_h in range(*FRAME_H_RANGE):
        if avail < f_h + 1:
            continue
        for f_w in range(*FRAME_W_RANGE):
            for rot_k in range(4):
                r_h, r_w = (f_h, f_w) if rot_k % 2 == 0 else (f_w, f_h)
                comp_h = r_h * n
                comp_w = r_w * n
                if comp_h <= grid_h and comp_w <= canvas_w:
                    combos.append((f_h, f_w, rot_k, comp_h * comp_w))
    if not combos:
        return False

    if prefer_small:
        combos.sort(key=lambda x: x[3])
        keep = max(1, len(combos) // 3)
        combos = combos[:keep]
    frame_h, frame_w, rot_k, _ = random.choice(combos)
    frame_mask = random_sprite(
        frame_h,
        frame_w,
        density=FRAME_DENSITY,
        connectivity=8,
        color_palette=[1],
        symmetry="not_symmetric",
    ) != 0
    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)

    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_h,
        frame_W=frame_w,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    """Return (height, width) of the composite after rotation."""
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _code_options(max_offset: int) -> list[int]:
    return list(range(min(7, max_offset) + 1))


def _code_complexity(code: int) -> int:
    return bin(code).count("1")


def _single_legend_code_score(row_code: int, col_code: int) -> tuple[int, int]:
    multi = int(_code_complexity(row_code) >= 2) + int(_code_complexity(col_code) >= 2)
    nonzero = int(row_code > 0) + int(col_code > 0)
    return (multi, nonzero)


def _pair_code_score(pair: tuple[int, int, int, int]) -> tuple[int, int, int]:
    ar, ac, br, bc = pair
    same_one_axis = int((ar == br) ^ (ac == bc))
    multi = sum(int(_code_complexity(code) >= 2) for code in pair)
    spread = abs(ar - br) + abs(ac - bc)
    return (same_one_axis, multi, spread)


def _assign_binary_offsets(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign binary-coded top-left offsets, avoiding overlap for the 2-legend case."""
    if len(legends) == 1:
        leg = legends[0]
        ch, cw = _comp_size(leg)
        row_opts = _code_options(grid_h - ch)
        col_opts = _code_options(canvas_w - cw)
        if not row_opts or not col_opts:
            return False
        choices = [(r, c) for r in row_opts for c in col_opts if r > 0 or c > 0]
        if not choices:
            choices = [(r, c) for r in row_opts for c in col_opts]
        best_score = max(_single_legend_code_score(r, c) for r, c in choices)
        top_choices = [(r, c) for r, c in choices if _single_legend_code_score(r, c) == best_score]
        row_code, col_code = random.choice(top_choices)
        leg.update(row_code=row_code, col_code=col_code, comp_top=row_code, comp_left=col_code)
        return True

    a, b = legends[0], legends[1]
    a_ch, a_cw = _comp_size(a)
    b_ch, b_cw = _comp_size(b)
    a_rows = _code_options(grid_h - a_ch)
    a_cols = _code_options(canvas_w - a_cw)
    b_rows = _code_options(grid_h - b_ch)
    b_cols = _code_options(canvas_w - b_cw)
    if not a_rows or not a_cols or not b_rows or not b_cols:
        return False

    valid_pairs = []
    for ar in a_rows:
        for ac in a_cols:
            at, al = ar, ac
            for br in b_rows:
                for bc in b_cols:
                    bt, bl = br, bc
                    if at < bt + b_ch and bt < at + a_ch and al < bl + b_cw and bl < al + a_cw:
                        continue
                    valid_pairs.append((ar, ac, br, bc))
    if not valid_pairs:
        return False

    best_score = max(_pair_code_score(pair) for pair in valid_pairs)
    preferred = [pair for pair in valid_pairs if _pair_code_score(pair) == best_score]
    ar, ac, br, bc = random.choice(preferred)
    a.update(row_code=ar, col_code=ac, comp_top=ar, comp_left=ac)
    b.update(row_code=br, col_code=bc, comp_top=br, comp_left=bc)
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
    """Build a canvas with alternating horizontal or vertical color bands."""
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    band_dir = random.choice(["h", "v"])
    dim = grid_h if band_dir == "h" else canvas_w
    bw = random.randint(BAND_WIDTH_MIN, max(BAND_WIDTH_MIN, dim // 2))
    band_indices = np.arange(dim) // bw % 2
    colors = np.array([bg1, bg2])
    if band_dir == "h":
        canvas[:] = colors[band_indices][:, None]
    else:
        canvas[:] = colors[band_indices][None, :]
    return canvas


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Stamp each legend's tiled composite onto the canvas (non-{0,6} pixels overwrite)."""
    output = canvas.copy()
    for leg in legends:
        sp, rf = leg["sprite"], leg["rotated_frame"]
        n = leg["N"]
        t, l = leg["comp_top"], leg["comp_left"]
        fh, fw = rf.shape

        comp = np.zeros((fh * n, fw * n), dtype=int)
        sprite_mask = (sp != 0) & (sp != 6)
        for i in range(fh):
            for j in range(fw):
                if rf[i, j]:
                    r0, c0 = i * n, j * n
                    comp[r0 : r0 + n, c0 : c0 + n][sprite_mask] = sp[sprite_mask]

        region = output[t : t + fh * n, l : l + fw * n]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool):
    """Draw one legend panel (separator, marker square, rotation patch, sprite, frame)."""
    n = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, n - 4)) if not is_right else (sep_col + max(0, n - 3))

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 2, mc0 + 2] = 4
    for bit in range(3):
        if leg["row_code"] & (1 << bit):
            grid[marker_r0 + 1 + bit, mc0 + 1] = 4
        if leg["col_code"] & (1 << bit):
            grid[marker_r0 + 1 + bit, mc0 + 3] = 4

    rr0 = marker_r0 + 5
    grid[rr0 : rr0 + 3, mc0 : mc0 + 5] = 0
    grid[rr0 : rr0 + 3, mc0] = 6
    grid[rr0 : rr0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rr0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = 7
    grid[cx_r + dr, cx_c + dc] = 7

    sp_col = col_start + 1 - mc0
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["legend_sprite"]

    f_h, f_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - f_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - f_w))
    frame_region = grid[fr_start : fr_start + f_h, mc0 + f_col : mc0 + f_col + f_w]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        if num_legends == 2:
            n = random.choices([2, 3, 4, 5], weights=[6, 5, 2, 1])[0]
        else:
            n = random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": n, "legend_w": n + 2})

    left_lw = legends[0]["legend_w"]
    left_total = left_lw + 1
    right_total = (1 + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    for li in range(num_legends):
        if not _design_legend(
            legends[li],
            unit_color_sets[li],
            frame_colors[li],
            marker_r0,
            grid_h,
            canvas_w,
            prefer_small=(num_legends == 2),
        ):
            return None

    if not _assign_binary_offsets(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, marker_r0=marker_r0, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0, is_right=True)

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
