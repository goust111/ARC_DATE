# Variant of 291dc1e1
# Dimension: D3 — 源对象形状与识别 (src)
# Change: 实心矩形（thin_dim=2, thick_dim 为偶数 2/4/6/8），分布在两条平行band中... → 单像素宽的线段替代矩形
#
# Design analysis
# a) Concrete behavior and trace
# - The border system is unchanged: one marked corner, one horizontal edge line, one vertical edge line.
# - The source objects are no longer 2xh solid rectangles. Each object is now a solid 1xL line
#   segment in canonical form, with L in {2, 4, 6, 8}. The two parallel bands remain at canonical
#   rows 2 and 5, so the ordering cue remains “distance to blue edge first, then distance to red edge”.
# - Small trace in canonical orientation:
#   top band has lengths [4, 2], bottom band has lengths [6, 4]. Distances to blue edge are 1 for
#   the top band and 4 for the bottom band. Distances to red edge increase left-to-right because
#   segments are placed with 2-cell gaps. So the order is top-left, top-right, bottom-left, bottom-right.
#   If the whole input is transformed so blue becomes vertical, those horizontal 1xL segments become
#   vertical Lx1 segments in the input. The solver must still sort by the same distances, then rotate
#   each extracted segment 90 degrees CCW, producing horizontal 1xL segments in the stacked output.
# - Difference from the original task at the critical step:
#   original extraction accepted thick 2-row rectangles; this variant must reject them and only keep
#   components whose bbox has min dimension exactly 1. The rotation branch becomes visually stronger
#   because a line changes orientation completely.
#
# b) Complexity calibration
# - Keeping two bands and even lengths preserves enough combinatorics for 4-6 objects per sample.
# - Colors are drawn without replacement from a 6-color pool, so distinct objects stay visually easy
#   to track while the sorting rule is still nontrivial.
# - Lengths remain in a small discrete set {2,4,6,8}; this is enough variety for visible differences
#   without introducing arbitrary free-form geometry.
#
# c) Hidden assumption audit against the original solver
# - Corner detection still works unchanged because the marker/edge arrangement is unchanged.
# - Blue/red line selection still works unchanged because edge colors and lengths follow the same scheme.
# - The original object extraction relied on “no background inside bbox”, which is true for both solid
#   rectangles and solid 1-pixel-wide segments. However that alone would also accept rectangles, so this
#   variant adds the explicit condition min(height, width) == 1.
# - Distance computation uses bbox extents only; that remains valid for line segments.
# - The rotation step uses np.rot90 on cropped objects; for a line segment this flips 1xL <-> Lx1,
#   which is exactly the intended visible effect.
#
# d) Placement legality
# - Canonical construction places segments starting at column 2 with 2-cell gaps, so components cannot
#   touch each other or the edge lines.
# - The two bands are at rows 2 and 5 of an 8-row canonical grid, matching the original separation.
# - Whole-grid flips/rotations preserve non-overlap and keep every segment fully inside bounds.
#
# e) Visual observability
# - In blue-horizontal cases the input already shows horizontal segments, so the output is a reordered
#   stack of horizontal lines.
# - In blue-vertical cases the input shows vertical segments, while the output shows horizontal segments.
#   With 1-pixel thickness, this orientation change is immediately visible from 2-3 examples.
#
# f) Visual cleanliness
# - No internal minority pixels are used, because they would fragment or visually clutter thin segments.
# - Distinct colors, one-pixel thickness, and 2-cell gaps keep object boundaries unambiguous.
#
# g) Color inference
# - Colors are identity-preserving only; there is no color remapping rule. The limited palette is used
#   purely to help track objects from input to output.
#
# i) New assumptions introduced by the modified solver
# - Every non-edge object component must be a full solid line segment with bbox min dimension 1.
# - The generator guarantees this by only drawing canonical 1xL segments, preserving 2-cell spacing,
#   and validating that the transformed grid re-extracts as 4-6 such segments before returning it.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


BG = 8
BLUE = 1
RED = 2
MARKER = 0
LINE_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
LINE_LENGTHS = np.array([2, 4, 6, 8], dtype=int)


def _locate_axes(g: np.ndarray, bg: int) -> tuple[int, int, int, int, int, str, str]:
    H, W = g.shape
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

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
        red_col = int(vcol if hcol == BLUE else hcol)
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
    return int(cr), int(cc), int(marker), int(hcol), int(vcol), blue_orient, red_orient


def _extract_segments(g: np.ndarray, bg: int, marker: int, cr: int, cc: int) -> list[dict]:
    H, W = g.shape
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    segments = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        if int(min(sub.shape)) != 1:
            continue
        segments.append({"box": tuple(map(int, b)), "arr": sub.copy()})
    return segments


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
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


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack 1-pixel-wide line segments by proximity to two colored edge lines.

    The coordinate-system rule is unchanged from the original task; only the source-object shape
    changes from 2xh solid rectangles to solid line segments whose bbox has min dimension 1.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, hcol, _vcol, blue_orient, red_orient = _locate_axes(g, bg)

    segments = _extract_segments(g, bg, marker, cr, cc)
    assert segments

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for seg in segments:
        seg["db"] = int(_dist_to_edge(seg["box"], blue_orient, blue_edge_idx))
        seg["dr"] = int(_dist_to_edge(seg["box"], red_orient, red_edge_idx))

    segments.sort(
        key=lambda seg: (
            int(seg["db"]),
            int(seg["dr"]),
            _along_from_corner(seg["box"], blue_orient, cr, cc),
            _along_from_corner(seg["box"], red_orient, cr, cc),
        )
    )

    seq = []
    for seg in segments:
        arr = seg["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    outW = int(max(int(arr.shape[1]) for arr in seq))
    outH = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((outH, outW), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((outW - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)

    return out


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


def _sample_band_lengths(count: int) -> list[int]:
    return [int(x) for x in np.random.choice(LINE_LENGTHS, size=count, replace=True)]


def _build_canonical() -> np.ndarray:
    for _ in range(500):
        n0 = int(np.random.randint(2, 4))
        n1 = int(np.random.randint(2, 4))
        lengths0 = _sample_band_lengths(n0)
        lengths1 = _sample_band_lengths(n1)
        if max(lengths0 + lengths1) < 6:
            continue

        used0 = int(sum(lengths0) + 2 * (n0 - 1))
        used1 = int(sum(lengths1) + 2 * (n1 - 1))
        max_used = int(max(used0, used1))
        if max_used > 27:
            continue

        trailing = int(np.random.randint(1, 4))
        long_dim = int(max_used + 2 + trailing)
        grid = np.full((8, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        colors = [int(c) for c in np.random.choice(LINE_COLORS, size=n0 + n1, replace=False)]
        color_idx = 0
        for r0, lengths in ((2, lengths0), (5, lengths1)):
            c = 2
            for length in lengths:
                grid[r0, c : c + int(length)] = int(colors[color_idx])
                color_idx += 1
                c += int(length) + 2

        return grid

    raise RuntimeError("failed to build canonical line-segment layout")


def _is_valid_variant(grid: np.ndarray) -> bool:
    out = main(grid)
    bg = int(get_bg(grid))
    cr, cc, marker, hcol, _vcol, blue_orient, _red_orient = _locate_axes(grid, bg)
    segments = _extract_segments(grid, bg, marker, cr, cc)
    if not (4 <= len(segments) <= 6):
        return False
    if not all(int(min(seg["arr"].shape)) == 1 for seg in segments):
        return False
    if blue_orient == "v" and not all(int(seg["arr"].shape[1]) == 1 for seg in segments):
        return False
    if blue_orient == "h" and not all(int(seg["arr"].shape[0]) == 1 for seg in segments):
        return False
    if out.size == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        canonical = _build_canonical()
        blue_orient = str(np.random.choice(["h", "v"]))
        corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
        grid = _transform(canonical, blue_orient, corner)
        if _is_valid_variant(grid):
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D3 variant")
