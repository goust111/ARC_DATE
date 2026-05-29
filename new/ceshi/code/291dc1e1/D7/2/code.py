# Variant of 291dc1e1
# Dimension: D7 — 输出网格尺寸与裁剪 (grid_shape)
# Change: 输出尺寸紧密适配：高度=所有旋转后矩形高度之和，宽度=最大旋转后矩形宽度... → 输出放入方形网格（取长边为边长）

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis for the D7 variant
#
# a) Concrete behavior trace
# - Input stays the same task family: one marked corner, one horizontal edge line, one vertical edge line,
#   and several solid interior rectangles. The rectangles may contain a few minority-color cells so that
#   rotation remains visible.
# - Original output logic:
#   1. detect the corner + the two edge lines,
#   2. sort interior rectangles by distance to the "blue" axis then the "red" axis,
#   3. rotate every rectangle 90 degrees CCW iff the blue line is vertical,
#   4. stack them from top to bottom in a tightly cropped rectangle whose width is the maximum item width.
# - Variant output logic changes only step 4:
#   after the same ordering and optional rotation, compute the tight stack height H and width W, then place
#   the stack into a square canvas of side S=max(H, W). Rectangles still start at row 0 and remain
#   horizontally centered; any extra area on the right or bottom is background.
# - Small trace:
#   if the rotated sequence has shapes (2x6), (2x4), (2x4), the original output is 6x6 and the variant
#   happens to match it. If the sequence has shapes (2x10), (2x2), (2x4), the original output is 6x10,
#   while the variant becomes 10x10 with the 6x10 stack aligned to the top and 4 blank rows added below.
#   If the sequence has shapes (6x2), (4x2), (2x2), the original output is 12x2, while the variant is
#   12x12 with blank columns on both sides because the stacked rectangles stay horizontally centered.
#
# b) Complexity calibration
# - The new rule must be visible from only a few examples, so the generator rejects degenerate samples where
#   the tight output is already square.
# - To make both branches of S=max(H, W) observable, the generator creates:
#   - height-driven samples where H>W,
#   - width-driven samples where W>H.
# - The input structure is otherwise unchanged: two row bands of rectangles, consistent gaps, and at least
#   one rectangle in each band. This keeps the sorting cue identical to the source task instead of adding
#   unrelated new mechanisms.
#
# c) Implicit-assumption audit against the original main()
# - Corner detection assumes exactly one non-background corner cell whose neighbors are the two axis colors.
#   The generator preserves this.
# - Line extraction assumes the two guide lines occupy the entire corner row and column. Preserved.
# - Rectangle extraction assumes interior objects are 4-connected, non-overlapping, and have no background
#   holes inside their bounding box. Preserved by construction.
# - Sorting by (distance to blue edge, distance to red edge, tie-break along axes) does not depend on the
#   old output crop shape, so it remains valid unchanged.
# - The old code implicitly assumed the final canvas size is exactly (sum heights, max widths). That is the
#   only assumption that must change here. We now treat those as intermediate tight dimensions and then pad
#   to a square side length.
# - A subtle original "harmless" pattern was `out = full((outH, outW))`, which silently encoded the old D7
#   behavior. In the variant this would erase the entire change, so the square side must be computed after
#   the full ordered sequence is known.
#
# d) Placement legality
# - Generator rectangles are placed in two separated horizontal bands with fixed 2-cell column gaps. This
#   prevents 4-connectivity merges.
# - Width choices are constrained so every band fits in the canonical grid, then whole-grid flips/rotations
#   move that legal layout to other corners/orientations without introducing overlap.
# - Because the solver masks only the full corner row and column, every rectangle must stay strictly inside
#   the interior. The generator keeps a blank margin between each band and the axes.
#
# e) Visual observability
# - A human comparing examples can still infer the original stacking rule from object positions and the
#   optional rotation. The new clue is the persistent square output canvas with background-only padding on
#   the right or bottom.
# - Width-driven and height-driven samples ensure the square is not just "always wider" or "always taller";
#   the max-of-two-dimensions rule is therefore supported by repeated evidence.
#
# f) Visual cleanliness
# - Only three element types coexist in the input: guide lines, marker, and separated rectangles. There is
#   no overlapping markup.
# - Output remains a single clean stack; the variant adds only blank padding, so the visual scene stays easy
#   to parse.
#
# g) Color predictability
# - No new color rule is introduced. Colors remain object-local texture only; the transformation is entirely
#   spatial. Minority cells are used sparingly so they reveal rotation without creating a separate mapping
#   puzzle.
#
# h) New implicit assumptions introduced by this variant implementation
# - The helper that measures the tight stack assumes there is at least one extracted rectangle. The generator
#   always creates 3-6 rectangles, so that is guaranteed.
# - Width-driven visibility assumes blue stays horizontal; otherwise the mandatory rotation turns width-driven
#   canonical layouts back into height-driven outputs. The generator enforces blue-orientation choices that
#   match the intended branch.
# - The centering logic assumes every rectangle width is <= square side, which is true because the side is
#   defined as the maximum between total stack height and maximum rectangle width.
#
# i) Source-task note
# - The variant spec lists no source_task, so no extra reference solver is consulted.


def _extract_ordered_sequence(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == H - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == W - 1:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != int(g[r, c]) for n in nbr):
            corner = (int(r), int(c))
            break
    assert corner is not None

    cr, cc = map(int, corner)
    marker = int(g[cr, cc])

    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == 1 or vcol == 1:
        blue_col = 1
        red_col = int(vcol if hcol == 1 else hcol)
    else:
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < W and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(g[r, cc]) == int(vcol):
            vlen += 1
            r += int(vstep)
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": sub.copy()})

    assert rects

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            col = int(edge_idx)
            if int(c0) > col:
                return int(c0 - col - 1)
            if int(c1) < col:
                return int(col - c1 - 1)
            return 0
        row = int(edge_idx)
        if int(r0) > row:
            return int(r0 - row - 1)
        if int(r1) < row:
            return int(row - r1 - 1)
        return 0

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        rect["db"] = int(dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(dist_to_edge(rect, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            along_from_corner(rect, blue_orient),
            along_from_corner(rect, red_orient),
        )
    )

    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    return bg, seq


def _tight_stack_shape(seq: list[np.ndarray]) -> tuple[int, int]:
    height = int(sum(int(arr.shape[0]) for arr in seq))
    width = int(max(int(arr.shape[1]) for arr in seq))
    return height, width


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two colored edge lines, then place the stack
    into a square output whose side is max(tight_stack_height, tight_stack_width).
    """
    bg, seq = _extract_ordered_sequence(np.array(grid, dtype=int))
    tight_h, tight_w = _tight_stack_shape(seq)
    side = int(max(tight_h, tight_w))
    out = np.full((side, side), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((side - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)

    return out


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid
        if corner == "tr":
            return np.fliplr(grid)
        if corner == "bl":
            return np.flipud(grid)
        return np.rot90(grid, 2)

    base = np.rot90(grid, 1)
    if corner == "bl":
        return base
    if corner == "tl":
        return np.flipud(base)
    if corner == "tr":
        return np.rot90(grid, 3)
    return np.flipud(np.rot90(grid, 3))


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.55:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _sample_band_pair(square_driver: str) -> tuple[int, int]:
    if square_driver == "width":
        choices = [(1, 2), (2, 1), (2, 2)]
    else:
        choices = [(2, 3), (3, 2), (3, 3)]
    idx = int(np.random.randint(len(choices)))
    return tuple(map(int, choices[idx]))


def _sample_band_widths(count: int, square_driver: str, force_wide: bool) -> list[int]:
    widths = []
    for i in range(count):
        if force_wide and i == 0:
            widths.append(int(np.random.choice([10, 12])))
        elif square_driver == "width":
            widths.append(int(np.random.choice([2, 4])))
        else:
            widths.append(int(np.random.choice([2, 4, 6])))
    return widths


def _band_used_width(widths: list[int]) -> int:
    return int(sum(int(w) for w in widths) + 2 * max(0, len(widths) - 1))


def _build_canonical(square_driver: str) -> np.ndarray:
    n0, n1 = _sample_band_pair(square_driver)
    wide_band = int(np.random.randint(2))

    widths0 = _sample_band_widths(n0, square_driver, force_wide=(square_driver == "width" and wide_band == 0))
    widths1 = _sample_band_widths(n1, square_driver, force_wide=(square_driver == "width" and wide_band == 1))

    used0 = _band_used_width(widths0)
    used1 = _band_used_width(widths1)
    long_dim = int(max(used0, used1) + 4 + np.random.randint(0, 3))
    assert long_dim <= 30

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, widths in ((2, widths0), (5, widths1)):
        c = 2
        for w in widths:
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, int(w)), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, int(r0), int(c))
            c += int(w) + 2

    return grid


def _gen_attempt() -> np.ndarray:
    square_driver = str(np.random.choice(["height", "width"], p=[0.6, 0.4]))
    canonical = _build_canonical(square_driver)
    if square_driver == "width":
        blue_orient = "h"
    else:
        blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def _variant_stats(grid: np.ndarray) -> tuple[int, int, int]:
    _bg, seq = _extract_ordered_sequence(grid)
    tight_h, tight_w = _tight_stack_shape(seq)
    side = int(max(tight_h, tight_w))
    return tight_h, tight_w, side * side - tight_h * tight_w


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        tight_h, tight_w, extra_area = _variant_stats(grid)
        if extra_area <= 0:
            continue
        if tight_h == tight_w:
            continue
        _ = main(grid)
        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D7 variant")
