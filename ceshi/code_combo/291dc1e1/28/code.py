# Combo variant of 291dc1e1 (index 28)
# combo_spec: D5/4+D11/2
# Name: 镜像翻转+band间颜色互换

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis
# - Concrete behavior: the original task sorts rectangles by distance to the blue
#   and red edge-lines, then stacks them. When blue is vertical it rotates every
#   rectangle 90 degrees CCW. This combo keeps the same extraction and ordering,
#   but changes two transform stages at once:
#   1) vertical-blue inputs use a left-right mirror (`fliplr`) instead of
#      rotation, so a 2x6 rectangle stays 2x6 rather than becoming 6x2.
#   2) the near-blue band (db smaller) and far-blue band (db larger) exchange
#      dominant colors pairwise after band-local sorting by the same ordering key.
#      Rectangle i in the near band takes rectangle i's dominant color from the
#      far band, and vice versa.
# - Small example in canonical orientation:
#     near band rectangles: [color 3 width 4], [color 4 width 2]
#     far band rectangles:  [color 6 width 6], [color 7 width 4]
#   Original horizontal-blue output body colors would stack as 3,4,6,7.
#   Combo output stacks the same shapes in the same order but with body colors
#   6,7,3,4. If blue becomes vertical after input transformation, original output
#   would rotate each rectangle; combo output mirrors each rectangle instead, so
#   both geometry and colors differ.
# - Complexity calibration: the generator keeps exactly two bands and equal band
#   counts (2 or 3 each). This avoids ambiguous pairing while still leaving
#   multiple rectangles per band so the color-swap rule has repeated evidence.
#   Dominant colors in a pair are forced to differ, preventing degeneration back
#   to the original task. Some rectangles get minority pixels to make "mirror
#   vs rotate" visually inspectable.
# - Hidden-assumption audit against the original solver:
#   * Corner / line detection is unchanged, so the original edge-line assumptions
#     still hold.
#   * Rectangle extraction still assumes each object fully fills its bounding box.
#     The generator preserves this by placing solid 2xh rectangles and changing
#     only interior colors, never adding background holes.
#   * New dominant-color swapping assumes each rectangle has a unique dominant
#     color. The generator decorates only area >= 8 rectangles and keeps
#     minorities <= 3 pixels, so one color remains strictly most frequent.
#   * Pairwise swap is most natural when both bands have the same length. The
#     generator guarantees equal counts; the solver still degrades safely by
#     swapping only the first min(len(near), len(far)) pairs if given odd inputs.
# - Placement legality: canonical placement uses fixed rows at db=1 and db=4,
#   starts each band at dr=1, separates rectangles by 2 background cells, and
#   chooses widths so the longer band fits inside long_dim <= 28. The later
#   orientation/corner transform is a rigid symmetry, so it preserves
#   non-overlap and in-bounds placement.
# - Visual observability: humans can infer the rule from a few examples because
#   rectangles still come in two clean depth bands and keep their order. The new
#   output simultaneously shows (a) near/far band colors crossing over by rank
#   and (b) vertical-blue cases producing mirrored, not rotated, sprites.
# - Color-change interpretability: minority colors are restricted to {1, 2},
#   while swappable body colors come from {3, 4, 5, 6, 7, 9}. This separates the
#   "body color moved across bands" signal from incidental decorative pixels.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
BODY_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2], dtype=int)


def _dominant_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    order = np.lexsort((vals, -counts))
    return int(vals[int(order[0])])


def _recolor_dominant(arr: np.ndarray, src: int, dst: int) -> np.ndarray:
    if int(src) == int(dst):
        return arr.copy()
    out = arr.copy()
    out[out == int(src)] = int(dst)
    return out


def _locate_corner_and_lines(g: np.ndarray, bg: int) -> tuple[int, int, int, int, int, str, str]:
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
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cr, cc, marker, _hcol, _vcol, blue_orient, red_orient = _locate_corner_and_lines(g, bg)

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict[str, object]] = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": box, "arr": sub.copy()})

    assert rects

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for idx, rect in enumerate(rects):
        box = rect["box"]
        rect["idx"] = int(idx)
        rect["db"] = int(_dist_to_edge(box, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(box, red_orient, red_edge_idx))
        rect["ab"] = int(_along_from_corner(box, blue_orient, cr, cc))
        rect["ar"] = int(_along_from_corner(box, red_orient, cr, cc))
        rect["dom"] = int(_dominant_color(np.asarray(rect["arr"], dtype=int)))

    rects.sort(key=lambda r: (int(r["db"]), int(r["dr"]), int(r["ab"]), int(r["ar"])))

    by_db: dict[int, list[dict[str, object]]] = {}
    for rect in rects:
        by_db.setdefault(int(rect["db"]), []).append(rect)

    swap_dom_by_idx: dict[int, int] = {}
    if len(by_db) >= 2:
        dbs = sorted(by_db)
        near = by_db[int(dbs[0])]
        far = by_db[int(dbs[-1])]
        pair_count = min(len(near), len(far))
        for i in range(pair_count):
            swap_dom_by_idx[int(near[i]["idx"])] = int(far[i]["dom"])
            swap_dom_by_idx[int(far[i]["idx"])] = int(near[i]["dom"])

    seq = []
    for rect in rects:
        arr = np.asarray(rect["arr"], dtype=int)
        if blue_orient == "v":
            arr = np.fliplr(arr)
        src_dom = int(rect["dom"])
        dst_dom = int(swap_dom_by_idx.get(int(rect["idx"]), src_dom))
        arr = _recolor_dominant(arr, src_dom, dst_dom)
        seq.append(arr)

    out_w = int(max(int(a.shape[1]) for a in seq))
    out_h = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
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


def _decorate_patch(patch: np.ndarray, dominant: int, force: bool = False) -> np.ndarray:
    area = int(patch.size)
    if area < 8:
        return patch
    if not force and np.random.rand() < 0.45:
        return patch

    flat = patch.reshape(-1).copy()
    target = int(np.random.randint(2, 4))
    cells = np.random.choice(area, size=target, replace=False)

    for cell in cells:
        flat[int(cell)] = int(np.random.choice(MINORITY_COLORS))

    candidate = flat.reshape(patch.shape)
    if patch.shape[1] == 2 and np.array_equal(np.fliplr(candidate), np.rot90(candidate, 1)):
        return _decorate_patch(patch, dominant, force=True)
    if patch.shape[1] > 2 and np.array_equal(np.fliplr(candidate), candidate):
        return _decorate_patch(patch, dominant, force=True)
    return candidate


def _sample_band_widths(count: int) -> list[int]:
    return [int(np.random.choice([2, 4, 6, 8])) for _ in range(count)]


def _build_canonical() -> np.ndarray:
    while True:
        count = int(np.random.choice([2, 3]))
        near_widths = _sample_band_widths(count)
        far_widths = _sample_band_widths(count)
        if max(near_widths + far_widths) == 2:
            continue

        used_near = int(sum(near_widths) + 2 * (count - 1))
        used_far = int(sum(far_widths) + 2 * (count - 1))
        max_used = max(used_near, used_far)
        trailing = int(np.random.choice([1, 2]))
        long_dim = int(2 + max_used + trailing)
        if not (8 <= long_dim <= 28):
            continue

        colors = np.random.permutation(BODY_COLORS)
        near_colors = [int(c) for c in colors[:count]]
        far_colors = [int(c) for c in colors[count : 2 * count]]
        if any(int(a) == int(b) for a, b in zip(near_colors, far_colors)):
            continue

        grid = np.full((8, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        eligible_slots = []
        for band_id, widths in enumerate((near_widths, far_widths)):
            for i, width in enumerate(widths):
                if int(width) > 2:
                    eligible_slots.append(int(band_id * count + i))
        decorate_slot = int(np.random.choice(eligible_slots))
        slot_idx = 0
        for r0, widths, colors_band in ((2, near_widths, near_colors), (5, far_widths, far_colors)):
            c = 2
            for width, color in zip(widths, colors_band):
                patch = np.full((2, int(width)), int(color), dtype=int)
                force = bool(slot_idx == decorate_slot)
                patch = _decorate_patch(patch, int(color), force=force)
                paste(grid, patch, int(r0), int(c))
                c += int(width) + 2
                slot_idx += 1

        return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        canonical = _build_canonical()
        blue_orient = str(np.random.choice(["h", "v"]))
        corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
        grid = _transform(canonical, blue_orient, corner)
        try:
            _ = main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate combo input for 291dc1e1/28")
