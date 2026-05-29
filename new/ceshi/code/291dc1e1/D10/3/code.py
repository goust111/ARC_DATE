# Variant of 291dc1e1
# Dimension: D10 — 堆叠顺序与多步处理 (multi_step)
# Change: 一次性排序所有矩形后按序堆叠，无迭代... → 先按蓝线距离分组，组内按红线距离排序后水平拼接，再将各组垂直堆叠

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the D10 variant:
# - Concreteized behavior:
#   In the canonical orientation the blue line is the top row and the red line is
#   the left column. Rectangles live in 2-cell-tall horizontal bands such as
#   rows 2-3, 5-6, 8-9. Old behavior sorted every rectangle globally by
#   (blue-distance, red-distance), so a sample with bands [A,B] and [C,D,E]
#   became a vertical stack A / B / C / D / E. The new behavior first groups by
#   blue-distance: band 1 becomes [A|B], band 2 becomes [C|D|E], then these rows
#   are vertically stacked into a small grid. This is the only rule change; axis
#   detection, tie-breaking, and the "rotate 90 deg CCW when blue is vertical"
#   rule are preserved.
# - Complexity calibration:
#   The generator always creates 2-4 blue-distance bands and 2-4 rectangles per
#   band. This guarantees repeated support for the grouping cue and makes the
#   output visibly different from the original one-step stack. Widths vary across
#   rectangles so each row is nontrivial, but all rectangles share post-rotation
#   height 2 to keep the row-concatenation rule visually clean.
# - Implicit-assumption audit against the original solver:
#   `line_mask` still assumes the two guide lines occupy the full corner row and
#   column; the generator preserves that. Component extraction still assumes each
#   rectangle is a single 4-connected non-background block with no background
#   holes; the generator preserves that as well, even when minority colors are
#   sprinkled inside a rectangle. The old global `rects.sort(...)` encoded the
#   whole output order; that assumption is exactly what changes here, so the new
#   solver only uses blue-distance for inter-band order and red-distance for
#   intra-band order. The rotation step stays before output placement, which is
#   important because vertical-blue inputs would otherwise create incompatible row
#   heights after grouping.
# - Placement legality:
#   Canonical inputs use a top blue line, a left red line, then place each
#   rectangle band at row `2 + 3*i` and each rectangle inside a band with a
#   2-cell gap. Grid width is derived from the widest band plus trailing
#   background, so no rectangle overlaps or exits the canvas. Random flips/rot90
#   only move whole valid constructions.
# - Visual observability / neatness:
#   The output now has 2-4 rows instead of one long vertical stack, and each row
#   contains multiple blocks. That is visible from 2-3 examples without needing
#   hidden arithmetic. Rectangles keep small asymmetric minority pixels so the
#   blue-vertical rotation branch remains learnable, but the interiors stay sparse
#   enough that the rectangles are still read as coherent blocks.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_OPTIONS = np.array([2, 4, 6, 8], dtype=int)


def _locate_axes(g: np.ndarray, bg: int) -> tuple[int, int, int, str, str]:
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
    return int(cr), int(cc), int(marker), blue_orient, red_orient


def _dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
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


def _along_from_corner(rect: dict, orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _extract_rects(g: np.ndarray) -> tuple[list[dict], int, str, str]:
    H, W = g.shape
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _locate_axes(g, bg)

    line_mask = np.zeros((H, W), dtype=bool)
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

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect, red_orient, red_edge_idx))
        rect["ab"] = int(_along_from_corner(rect, blue_orient, cr, cc))
        rect["ar"] = int(_along_from_corner(rect, red_orient, cr, cc))

    return rects, bg, blue_orient, red_orient


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - detect rectangles as in the source task
    - group them by distance to the blue edge
    - within each blue-distance band, sort by distance to the red edge
    - if blue edge is vertical, rotate each rectangle 90 deg CCW
    - concatenate each band horizontally, then stack those rows vertically
    """
    g = np.array(grid, dtype=int)
    rects, bg, blue_orient, _red_orient = _extract_rects(g)

    bands: dict[int, list[dict]] = {}
    for rect in rects:
        bands.setdefault(int(rect["db"]), []).append(rect)

    rows = []
    for db in sorted(bands):
        items = sorted(bands[int(db)], key=lambda r: (int(r["dr"]), int(r["ar"]), int(r["ab"])))
        seq = []
        for rect in items:
            arr = rect["arr"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            seq.append(arr)

        row_h = int(max(int(arr.shape[0]) for arr in seq))
        row_w = int(sum(int(arr.shape[1]) for arr in seq))
        row = np.full((row_h, row_w), int(bg), dtype=int)
        out_c = 0
        for arr in seq:
            h, w = map(int, arr.shape)
            out_r = int((row_h - h) // 2)
            paste(row, arr, int(out_r), int(out_c))
            out_c += int(w)
        rows.append(row)

    out_h = int(sum(int(row.shape[0]) for row in rows))
    out_w = int(max(int(row.shape[1]) for row in rows))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for row in rows:
        h, w = map(int, row.shape)
        paste(out, row, int(out_r), 0)
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


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.35:
        return patch

    max_minor = min(3, area // 3)
    if max_minor < 1:
        return patch

    n_minor = int(np.random.randint(1, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    palette = [int(c) for c in MINORITY_COLORS if int(c) != int(dominant)]
    colors = np.random.choice(palette, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _build_canonical() -> np.ndarray:
    for _ in range(200):
        band_count = int(np.random.randint(2, 5))
        band_widths = []
        band_used = []
        for _band in range(band_count):
            rect_count = int(np.random.randint(2, 5))
            widths = [int(np.random.choice(WIDTH_OPTIONS)) for _ in range(rect_count)]
            used = int(sum(widths) + 2 * (rect_count - 1))
            band_widths.append(widths)
            band_used.append(used)

        if max(band_used) > 24:
            continue
        if len(set(band_used)) == 1 and len(set(len(w) for w in band_widths)) == 1:
            continue

        trailing_w = int(np.random.randint(1, 4))
        trailing_h = int(np.random.randint(1, 3))
        H = int(3 * band_count + 1 + trailing_h)
        W = int(2 + max(band_used) + trailing_w)
        if H > 30 or W > 30:
            continue

        grid = np.full((H, W), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        for band_idx, widths in enumerate(band_widths):
            r0 = int(2 + 3 * band_idx)
            c0 = 2
            color_order = np.random.permutation(DOMINANT_COLORS)
            for rect_idx, width in enumerate(widths):
                dominant = int(color_order[rect_idx % len(color_order)])
                patch = np.full((2, int(width)), dominant, dtype=int)
                patch = _sprinkle_minorities(patch, dominant)
                paste(grid, patch, int(r0), int(c0))
                c0 += int(width) + 2

        return grid
    raise RuntimeError("failed to build canonical layout")


def _variant_metrics(grid: np.ndarray) -> dict:
    rects, _bg, blue_orient, _red_orient = _extract_rects(grid)
    band_sizes: dict[int, int] = {}
    for rect in rects:
        band_sizes[int(rect["db"])] = band_sizes.get(int(rect["db"]), 0) + 1
    sizes = sorted(int(v) for v in band_sizes.values())
    return {
        "band_count": int(len(sizes)),
        "max_band_size": int(max(sizes)),
        "rect_count": int(len(rects)),
        "blue_orient": str(blue_orient),
    }


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        metrics = _variant_metrics(grid)
        if metrics["band_count"] < 2 or metrics["max_band_size"] < 2:
            continue
        _ = main(grid)
        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D10 variant")
