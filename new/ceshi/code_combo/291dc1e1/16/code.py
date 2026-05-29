# Combo variant of 291dc1e1 (index 16)
# combo_spec: D6/3+D7/2
# Name: 水平排列+方形输出

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis
#
# a) Concrete behavior and trace
# - The input family stays close to the source task: one marked corner, one blue edge
#   line, one red edge line, and several solid rectangles in the interior.
# - The original solver pipeline is:
#   1. detect the special corner and infer which adjacent line is blue/red,
#   2. mask the corner row/column and extract solid interior rectangles,
#   3. sort rectangles by (distance to blue, distance to red, tie-breaks along axes),
#   4. rotate every rectangle 90 degrees CCW if the blue edge is vertical,
#   5. stack them top-to-bottom in a tight canvas.
# - This combo keeps steps 1-4 unchanged and replaces step 5 with two linked changes:
#   first build the D6 horizontal strip, then apply the D7 square wrapper to that strip.
# - Small trace:
#   suppose the ordered post-rotation sequence is A: 2x6, B: 4x4, C: 2x2 with bg 8.
#   Original task:
#     tight vertical canvas = 8x6, paste A/B/C from row 0 downward, each centered in x.
#   D6-only:
#     tight horizontal canvas = 4x12, paste A/B/C from col 0 rightward, each centered in y.
#   This fused variant:
#     keep that 4x12 horizontal strip, then place it into a 12x12 square at the top-left.
#     The extra 8 rows are background. B stays taller than A/C, so y-centering remains visible.
# - The interaction matters: the square size is computed from the new horizontal layout
#   (`side = max(total_width, max_height)`), not from the original vertical stack.
#
# b) Complexity calibration
# - A degenerate generator could leave all rectangles with height 2, making the square
#   change visible only as a shallow blank footer and making vertical centering inside
#   the horizontal strip nearly invisible. To avoid that, the generator uses two bands
#   with different rectangle heights chosen from a small set {2,4,6}.
# - The rectangle count stays moderate (4-8 total) with repeated widths from {2,4,6},
#   so the ordering cue is still easy to infer and the new layout is visually obvious.
# - The rule does not add extra grouping, recoloring, or spacing conventions, so the
#   puzzle remains about spatial transformation rather than new symbolic bookkeeping.
#
# c) Implicit-assumption review of the original main()
# - Corner detection assumes exactly one corner marker whose orthogonal neighbors are
#   the two edge-line colors. The generator preserves that.
# - Blue/red inference assumes those lines occupy the full corner row and column. Kept.
# - Rectangle extraction assumes interior objects are 4-connected and hole-free inside
#   their bounding boxes. The generator constructs only solid rectangles with gaps.
# - Sorting by distances to the two edge lines does not depend on the old output shape,
#   so that logic remains valid unchanged.
# - The original output stage assumed:
#     outH = sum(heights), outW = max(widths), row accumulation, x-centering.
#   Those assumptions are false after D6/3. They become:
#     tight_h = max(heights), tight_w = sum(widths), column accumulation, y-centering.
# - D7/2 then adds one more change: the tight horizontal strip is no longer final.
#   It must be embedded into a square canvas of side max(tight_h, tight_w). Reusing
#   the old `np.full((outH, outW))` pattern would silently drop the square behavior.
#
# d) Placement legality
# - In the generator, rectangles are placed in two horizontal bands inside a canonical
#   grid with a 1-cell gap from the blue edge, a 1-2 cell gap between the bands, and
#   2-cell gaps between neighboring rectangles inside each band.
# - Each band uses a single fixed height, so objects in the same band share a clean top
#   edge and cannot overlap. Distinct band heights are allowed because the row bands are
#   separated by explicit background rows.
# - Whole-grid flips/rotations preserve non-overlap and keep every rectangle away from
#   the masked corner row/column.
# - Output placement is legal because the square side is at least both the strip height
#   and strip width, so the strip always fits when pasted at (0,0).
#
# e) Visual observability
# - Across a few samples, a human should see the same order cue as the source task but
#   the output is no longer a vertical stack. It is a left-to-right strip whose shorter
#   rectangles float vertically within the strip, and that strip sits inside a square.
# - The generator rejects samples where the horizontal strip already equals the original
#   vertical output or where the square wrapper adds no padding, because those samples
#   would hide one of the two fused changes.
#
# f) Visual neatness
# - Inputs contain only four clean element types: background, the two guide lines, the
#   corner marker, and separated solid rectangles. No object overlaps another object.
# - Outputs contain one composed horizontal strip plus blank square padding; no extra
#   marks or borders are added, so the transformed scene stays easy to parse.
#
# g) Color-change inference
# - There is no new color mapping. Rectangle colors are copied through unchanged.
# - Minority pixels are used sparingly inside some rectangles only to make rotation
#   observable when blue is vertical; they do not encode extra rules.
#
# h) New assumptions introduced by the fused main() and how the generator supports them
# - `_assemble_horizontal_strip()` assumes there is at least one extracted rectangle.
#   The generator always creates two non-empty bands with 2-4 rectangles each.
# - Vertical centering inside the strip is only interesting when at least one rectangle
#   is shorter than `max_height`. The generator enforces different heights for the two
#   bands, so that condition always holds.
# - The square wrapper is only visible when `side > tight_h` or `side > tight_w`.
#   The generator filters for positive extra area, so the square does real work.
# - The fused behavior should differ from the original task, not just be a no-op square
#   around an unchanged vertical stack. The generator filters for that as well.


def _ordered_sequence(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    corners = [(0, 0), (0, int(w - 1)), (int(h - 1), 0), (int(h - 1), int(w - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbrs = []
        if r == 0:
            nbrs.append(int(g[r + 1, c]))
        if r == h - 1:
            nbrs.append(int(g[r - 1, c]))
        if c == 0:
            nbrs.append(int(g[r, c + 1]))
        if c == w - 1:
            nbrs.append(int(g[r, c - 1]))
        if nbrs and all(int(n) != int(g[r, c]) for n in nbrs):
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
        while 0 <= c < w and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < h and int(g[r, cc]) == int(vcol):
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

    line_mask = np.zeros((h, w), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": arr.copy()})

    assert rects

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            if int(c0) > int(edge_idx):
                return int(c0 - edge_idx - 1)
            if int(c1) < int(edge_idx):
                return int(edge_idx - c1 - 1)
            return 0
        if int(r0) > int(edge_idx):
            return int(r0 - edge_idx - 1)
        if int(r1) < int(edge_idx):
            return int(edge_idx - r1 - 1)
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

    return int(bg), seq


def _assemble_vertical_stack(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _assemble_horizontal_strip(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = int(max(int(arr.shape[0]) for arr in seq))
    out_w = int(sum(int(arr.shape[1]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_c = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_r = int((out_h - h) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_c += int(w)
    return out


def _assemble_combo(seq: list[np.ndarray], bg: int) -> np.ndarray:
    strip = _assemble_horizontal_strip(seq, bg)
    side = int(max(int(strip.shape[0]), int(strip.shape[1])))
    out = np.full((side, side), int(bg), dtype=int)
    paste(out, strip, 0, 0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder rectangles by proximity to the two edge lines, rotate if blue is vertical,
    arrange them left-to-right with vertical centering, then embed that strip in a
    square canvas whose side is max(total_strip_width, strip_height).
    """
    bg, seq = _ordered_sequence(np.array(grid, dtype=int))
    return _assemble_combo(seq, int(bg))


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
HEIGHT_CHOICES = (2, 4, 6)
WIDTH_CHOICES = (2, 4, 6)
BAND_COUNT_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2))


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
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != int(dominant)]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _band_used_width(widths: list[int]) -> int:
    return int(sum(int(w) for w in widths) + 2 * max(0, len(widths) - 1))


def _sample_band_heights() -> tuple[int, int]:
    h0 = int(np.random.choice(HEIGHT_CHOICES))
    choices = [int(h) for h in HEIGHT_CHOICES if int(h) != int(h0)]
    h1 = int(np.random.choice(choices))
    return h0, h1


def _sample_band_widths(count: int) -> list[int]:
    widths = [int(np.random.choice(WIDTH_CHOICES)) for _ in range(count)]
    if np.random.rand() < 0.45:
        idx = int(np.random.randint(count))
        widths[idx] = int(np.random.choice([4, 6]))
    return widths


def _build_canonical() -> np.ndarray:
    for _ in range(400):
        n0, n1 = BAND_COUNT_CHOICES[int(np.random.randint(len(BAND_COUNT_CHOICES)))]
        h0, h1 = _sample_band_heights()
        gap_rows = int(np.random.choice([1, 2]))
        widths0 = _sample_band_widths(int(n0))
        widths1 = _sample_band_widths(int(n1))
        total_strip_w = int(sum(widths0) + sum(widths1))

        used0 = _band_used_width(widths0)
        used1 = _band_used_width(widths1)
        max_used = int(max(used0, used1))
        if max_used > 22:
            continue
        if total_strip_w > 30:
            continue

        band0_r = 2
        band1_r = int(band0_r + h0 + gap_rows)
        grid_h = int(band1_r + h1 + 1)
        if grid_h > 18:
            continue

        trailing = int(np.random.choice([1, 2, 3]))
        grid_w = int(2 + max_used + trailing)
        if grid_w > 28:
            continue

        grid = np.full((grid_h, grid_w), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        for r0, band_h, widths in ((band0_r, h0, widths0), (band1_r, h1, widths1)):
            c = 2
            for w in widths:
                color = int(np.random.choice(DOMINANT_COLORS))
                patch = np.full((int(band_h), int(w)), color, dtype=int)
                patch = _sprinkle_minorities(patch, color)
                paste(grid, patch, int(r0), int(c))
                c += int(w) + 2

        return grid

    raise RuntimeError("failed to build canonical combo input")


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def _variant_metrics(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    bg, seq = _ordered_sequence(grid)
    vertical = _assemble_vertical_stack(seq, bg)
    strip = _assemble_horizontal_strip(seq, bg)
    combo = _assemble_combo(seq, bg)
    extra_area = int(combo.size - strip.size)
    differs_from_original = int(
        vertical.shape != combo.shape or not np.array_equal(vertical, combo)
    )
    differs_from_d6_only = int(
        strip.shape != combo.shape or not np.array_equal(strip, combo)
    )
    return (
        int(strip.shape[0]),
        int(strip.shape[1]),
        int(combo.shape[0]),
        int(extra_area),
        int(differs_from_original and differs_from_d6_only),
    )


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        strip_h, strip_w, side, extra_area, visible = _variant_metrics(grid)
        if extra_area <= 0:
            continue
        if visible == 0:
            continue
        if strip_h >= strip_w:
            continue
        _ = main(grid)
        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant")
