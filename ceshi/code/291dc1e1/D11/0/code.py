# Variant of 291dc1e1
# Dimension: D11 — 颜色变换——堆叠时的颜色处理 (transform)
# Change: 矩形颜色在堆叠过程中保持不变，直接复制到输出... → 重着色为统一颜色

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis for the D11 variant:
# a) Concrete behavior and trace:
#    The original task takes filled interior rectangles, sorts them by distance to two edge
#    lines, optionally rotates them, then stacks them while preserving each rectangle's colors.
#    This variant keeps the same rectangle geometry, sort order, and rotation, but changes the
#    final rendering step: every stacked rectangle is recolored to the blue-axis color.
#    Example: if three 2xh rectangles contain colors {4}, {9 with a 5 speckle}, {3}, the
#    original output would stack [4-block, 9/5-block, 3-block]; the variant stacks the same
#    shapes in the same order, but all three blocks become solid color 1.
# b) Complexity calibration:
#    Recoloring every rectangle to a single color is intentionally local to the final stacking
#    stage. To keep the effect visible, the generator preserves multiple input object colors and
#    occasional interior speckles; otherwise recolor would be hard to notice from 2-4 examples.
# c) Hidden-assumption audit for main():
#    - Corner detection assumes exactly one corner marker distinct from adjacent line colors.
#      This still holds unchanged.
#    - Blue/red axis inference assumes axis colors are encoded by the two edge lines; D11 does
#      not change that.
#    - Rectangle extraction uses "bbox has no background inside" as the rectangle test. This
#      remains valid because the generator still creates solid rectangular silhouettes; only
#      their pixel colors vary.
#    - Sorting uses only geometry (distance to blue/red edges plus along-edge tie-breaks), so
#      color removal cannot affect order.
#    - The original sequence construction copied each rectangle array directly. That line relied
#      on the old property "rectangle colors should survive into output". Under D11 this is the
#      one unsafe assumption, so we replace it with explicit recoloring before stacking.
# d) Placement legality:
#    The generator still builds disjoint 2xw rectangles in two horizontal bands separated by
#    background, then applies global flips/rotations that preserve non-overlap and edge-line
#    legality. Rectangles never touch the corner row/column, so solver masking remains correct.
# e) Visual observability:
#    A human now sees: input rectangles have several object colors, output rectangles keep the
#    same widths/heights/order but collapse to the blue-line color. Because the axis color also
#    exists in the input as a salient edge line, the recolor target is inferable without adding
#    a new legend.
# f) Visual neatness:
#    Only one large structural cue changes: object colors collapse. Geometry stays clean because
#    rectangles are still separated by 2-cell gaps and stacked without overlap.
# g) Color-rule inferability:
#    Object colors are restricted to a small non-axis palette {3,4,5,6,7,9}; the blue axis is
#    always color 1. This avoids accidental "already blue" rectangles and makes the mapping
#    "all objects -> blue-axis color" easy to infer from few examples.
# 9) New assumptions introduced by the variant:
#    The solver itself only assumes a valid blue axis exists, because that is the target color.
#    The generator additionally guarantees at least two distinct object colors and rejects any
#    sample where recolored output equals original-color output, so the D11 behavior is present
#    rather than silently degenerating to the base task.


def _solve(grid: np.ndarray, *, recolor: bool) -> np.ndarray:
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
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, b)), "arr": sub.copy()})

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

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    for rct in rects:
        rct["db"] = int(dist_to_edge(rct, blue_orient, blue_edge_idx))
        rct["dr"] = int(dist_to_edge(rct, red_orient, red_edge_idx))

    rects.sort(
        key=lambda r: (
            int(r["db"]),
            int(r["dr"]),
            along_from_corner(r, blue_orient),
            along_from_corner(r, red_orient),
        )
    )

    seq = []
    for rct in rects:
        a = rct["arr"]
        if blue_orient == "v":
            a = np.rot90(a, k=1)
        if recolor:
            a = np.full(a.shape, int(blue_col), dtype=int)
        seq.append(a)

    outW = int(max(int(a.shape[1]) for a in seq))
    outH = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((outH, outW), int(bg), dtype=int)

    out_r = 0
    for a in seq:
        h, w = map(int, a.shape)
        out_c = int((outW - w) // 2)
        paste(out, a, int(out_r), int(out_c))
        out_r += int(h)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Same geometric rule as 291dc1e1, but every stacked rectangle is recolored
    to the blue edge-line color before being pasted into the output.
    """
    return _solve(grid, recolor=True)


BG = 8
BLUE = 1
RED = 2
MARKER = 0
OBJECT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)


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


def _distribute_band_widths(count: int, used_target: int) -> list[int]:
    extra_units = (used_target - (4 * count - 2)) // 2
    order = np.random.permutation(count)
    units = np.zeros(count, dtype=int)
    remaining = int(extra_units)
    for i, idx in enumerate(order):
        slots_left = count - i - 1
        lo = max(0, remaining - 3 * slots_left)
        hi = min(3, remaining)
        take = int(np.random.randint(lo, hi + 1))
        units[int(idx)] = take
        remaining -= take
    return [2 + 2 * int(u) for u in units]


def _sprinkle_minorities(patch: np.ndarray, dominant: int, *, force: bool = False) -> np.ndarray:
    area = int(patch.size)
    if area < 8:
        return patch
    if not force and np.random.rand() < 0.6:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(c) for c in OBJECT_COLORS if int(c) != int(dominant)]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _build_canonical() -> np.ndarray:
    long_dim = int(np.random.randint(9, 29))
    options = []
    for trailing in (1, 2):
        max_used = long_dim - 2 - trailing
        if max_used < 6 or max_used % 2:
            continue
        max_count = min(5, (max_used + 2) // 4)
        pairs = []
        for a in range(2, max_count + 1):
            for b in range(2, max_count + 1):
                if a + b > 9:
                    continue
                eligible = []
                if max_used <= 10 * a - 2:
                    eligible.append(0)
                if max_used <= 10 * b - 2:
                    eligible.append(1)
                if eligible:
                    pairs.append((a, b, eligible))
        if pairs:
            options.append((max_used, pairs))

    max_used, pairs = options[int(np.random.randint(len(options)))]
    n0, n1, eligible = pairs[int(np.random.randint(len(pairs)))]
    max_band = int(np.random.choice(eligible))
    max0 = min(max_used, 10 * n0 - 2)
    max1 = min(max_used, 10 * n1 - 2)
    used0 = int(max_used if max_band == 0 else np.random.choice(np.arange(4 * n0 - 2, max0 + 1, 2)))
    used1 = int(max_used if max_band == 1 else np.random.choice(np.arange(4 * n1 - 2, max1 + 1, 2)))
    band0 = _distribute_band_widths(n0, used0)
    band1 = _distribute_band_widths(n1, used1)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    widths = band0 + band1
    force_speckle_idx = None
    large_candidates = [i for i, w in enumerate(widths) if int(2 * w) >= 8]
    if large_candidates:
        force_speckle_idx = int(np.random.choice(large_candidates))

    patch_colors = np.random.choice(OBJECT_COLORS, size=len(widths), replace=True)
    if len(set(map(int, patch_colors))) < 2:
        alt_idx = int(np.random.randint(len(widths)))
        alternatives = [int(c) for c in OBJECT_COLORS if int(c) != int(patch_colors[0])]
        patch_colors[alt_idx] = int(np.random.choice(alternatives))

    patch_idx = 0
    for r0, widths_in_band in ((2, band0), (5, band1)):
        c = 2
        for w in widths_in_band:
            color = int(patch_colors[patch_idx])
            patch = np.full((2, int(w)), color, dtype=int)
            patch = _sprinkle_minorities(patch, color, force=patch_idx == force_speckle_idx)
            paste(grid, patch, r0, c)
            c += int(w) + 2
            patch_idx += 1

    return grid


def _count_object_colors(grid: np.ndarray) -> int:
    vals = {
        int(v)
        for v in grid.flatten()
        if int(v) not in {int(BG), int(BLUE), int(RED), int(MARKER)}
    }
    return int(len(vals))


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if _count_object_colors(grid) < 2:
            continue
        new_out = _solve(grid, recolor=True)
        old_out = _solve(grid, recolor=False)
        if np.array_equal(new_out, old_out):
            continue
        assert set(map(int, np.unique(new_out))) <= {int(BG), int(BLUE)}
        return grid
    raise RuntimeError("failed to generate D11 input for 291dc1e1")
