# Combo variant of 291dc1e1 (index 18)
# combo_spec: D9/0+D14/1
# Name: 无标记像素+装饰线

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

"""
Design analysis for combo D9/0 + D14/1
--------------------------------------

a) Concrete behavior and trace
   - Original task:
     one full border row and one full border column meet at a corner marker.
     The solver finds that marker, reads the two adjacent line colors, then
     treats every remaining solid component as a sortable rectangle.
   - This combo removes the marker and adds one extra decorative thin line.
     Therefore the entry-point logic changes in two coupled places:
     1. Functional axes can no longer be found by a unique corner pixel.
        Instead, the solver must find the unique all-non-background border row
        and border column, infer their dominant colors, and assign blue/red by
        line length: longer line = blue.
     2. After those axes are masked out, not every remaining component is a
        rectangle anymore, because the decorative line is also a foreground
        component. The solver must ignore 1-pixel-thick components.
   - Small trace in canonical orientation:
     top row = all color 2, left column = [2,1,1,1,1,1,1,1], and the far-right
     border also contains a vertical color-5 segment from rows 2..6.
     Old solver fails immediately because no corner marker exists.
     D9-only logic can find the axes but will also extract the color-5 segment
     as a 5x1 "rectangle" and stack it into the output.
     Combo logic:
       detect top row + left column as the two functional axes,
       conclude blue is horizontal because its visible length is longer,
       mask that row/column,
       keep only solid components with height >= 2 and width >= 2,
       so the 5x1 decorative line is discarded and only the true rectangles
       participate in sorting.

b) Complexity calibration
   - The line-role ambiguity is real but controlled: only two candidate line
     colors exist, and exactly one geometric cue resolves them (longer line).
   - The noise is also controlled: exactly one decorative line, always 1-pixel
     thick and length >= 4, so it is visually salient but structurally simple.
   - Decorative color is chosen from body colors but removed from rectangle
     palettes for that sample, preventing clutter where the distractor color is
     also heavily reused inside rectangles.

c) Hidden-assumption audit of the original solver
   - Corner scan depended on a unique non-background marker at a corner whose
     color differed from both neighbors. This is invalid after D9/0.
   - "If color 1 is present, it is blue" is also invalid. In this combo, either
     color 1 or color 2 may be the longer line, so blue must be assigned by
     visible line length.
   - "Every non-line foreground component is a rectangle" becomes invalid after
     D14/1 because the decorative line is also a component. We repair this by
     requiring extracted sortable objects to be solid and at least 2x2.
   - Masking the full border row and full border column still remains valid,
     because the functional axes keep the same geometry as the base task.

d) Placement legality
   - Canonical construction remains 8 x long_dim, with long_dim in 9..28.
   - Rectangles still occupy the two interior 2-row bands, start 1 cell away
     from the red axis, and keep 2-cell gaps along the blue direction.
   - The decorative line is pasted on the opposite border edge with gaps to both
     corners, so it never overlaps a functional axis corner and never becomes a
     second full border row/column.
   - Because objects are separated by background before the global transforms,
     flips/rotations preserve non-overlap and non-adjacency.

e) Visual observability
   - Across examples, humans can see that outputs contain only stacked
     rectangles, while one extra thin border segment from the input disappears.
   - They can also see there is no special corner marker anymore, so the solver
     must use the two long border lines themselves as the coordinate system.
   - Examples where color 2 is the longer line make it visible that line role is
     geometric, not tied to a fixed color id.

f) Visual cleanliness
   - The decorative line lives on an otherwise unused opposite edge strip, so it
     does not weave through the rectangle field or split components.
   - Rectangles remain separated by background and maintain their simple 2xN
     structure, which keeps the scene legible despite the added distractor.

g) Color inferability
   - No arbitrary color remapping is introduced. Colors 1 and 2 still denote
     the two functional axes; only their roles are inferred from length.
   - Decorative color varies across samples, but its structural signature is
     stable: a lone 1-pixel-thick line that never appears in the output.

h) Quantitative behavior targets for this combo
   - Every generated input should satisfy:
     1. no marker color 0 anywhere,
     2. exactly one full non-background border row and one full non-background
        border column,
     3. exactly one extra 1-pixel-thick solid component of length >= 4 after
        masking the functional axes.
   - On the same generated input, a D9-only solver that lacks thin-line
     filtering should usually produce a different output, proving the decorative
     line is behaviorally active rather than decorative in name only.

i) New solver assumptions and generator guarantees
   - Assumption: there is exactly one all-non-background border row and one
     all-non-background border column. The generator guarantees this by keeping
     only the functional axes full-length and leaving gaps at the decorative
     line's endpoints.
   - Assumption: the dominant color of each functional axis is unique and the
     two axis lengths differ. The canonical layout guarantees width > height-1,
     so the long line is always strictly distinguishable.
   - Assumption: sortable objects are solid components with both dimensions >= 2.
     The generator guarantees rectangles are 2x{2,4,6,8}, while the decorative
     line is always 1-pixel thick and disconnected from rectangles.
"""


BG = 8
LINE_COLORS = np.array([1, 2], dtype=int)
BODY_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_BASE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _majority_color(vec: np.ndarray) -> int:
    vals, counts = np.unique(vec.astype(int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _locate_lines(g: np.ndarray, bg: int) -> tuple[int, int, str, str]:
    H, W = map(int, g.shape)
    edge_rows = [r for r in (0, H - 1) if bool(np.all(g[r, :] != int(bg)))]
    edge_cols = [c for c in (0, W - 1) if bool(np.all(g[:, c] != int(bg)))]
    assert len(edge_rows) == 1 and len(edge_cols) == 1

    cr = int(edge_rows[0])
    cc = int(edge_cols[0])
    row = g[cr, :]
    col = g[:, cc]

    hcol = int(_majority_color(row))
    vcol = int(_majority_color(col))
    assert hcol != vcol

    hlen = int(np.count_nonzero(row == int(hcol)))
    vlen = int(np.count_nonzero(col == int(vcol)))
    assert hlen != vlen

    blue_orient = "h" if hlen > vlen else "v"
    red_orient = "v" if blue_orient == "h" else "h"
    return cr, cc, blue_orient, red_orient


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two functional edge lines.

    Combo behavior:
    - No corner marker exists.
    - The unique full border row and full border column define the functional axes.
    - The longer of those two lines is the blue line; the shorter is the red line.
    - One extra thin decorative line may appear elsewhere and must be ignored.
    - If the blue line is vertical, rotate each rectangle 90 degrees CCW.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cr, cc, blue_orient, red_orient = _locate_lines(g, bg)

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask)
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        if min(map(int, sub.shape)) < 2:
            continue
        rects.append({"box": tuple(map(int, b)), "arr": sub.copy()})

    assert rects

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            if c0 > edge_idx:
                return int(c0 - edge_idx - 1)
            if c1 < edge_idx:
                return int(edge_idx - c1 - 1)
            return 0
        if r0 > edge_idx:
            return int(r0 - edge_idx - 1)
        if r1 < edge_idx:
            return int(edge_idx - r1 - 1)
        return 0

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

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


def _sprinkle_minorities(patch: np.ndarray, dominant: int, deco_color: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.6:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    minority_choices = [int(c) for c in MINORITY_BASE if int(c) != dominant and int(c) != deco_color]
    if not minority_choices:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor, len(minority_choices)) + 1))
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _add_decorative_line(grid: np.ndarray, deco_color: int) -> None:
    # Opposite border from the corner, with endpoint gaps so it is never a full edge line.
    grid[2:7, -1] = int(deco_color)


def _build_canonical(long_col: int, short_col: int) -> np.ndarray:
    long_dim = int(np.random.randint(9, 29))
    deco_color = int(np.random.choice(BODY_COLORS))
    body_choices = np.array([int(c) for c in BODY_COLORS if int(c) != deco_color], dtype=int)

    options = []
    for trailing in (2,):
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
    grid[0, :] = int(long_col)
    grid[:, 0] = int(short_col)
    grid[0, 0] = int(long_col)

    for r0, widths in ((2, band0), (5, band1)):
        c = 2
        for w in widths:
            color = int(np.random.choice(body_choices))
            patch = np.full((2, w), color, dtype=int)
            patch = _sprinkle_minorities(patch, color, deco_color)
            paste(grid, patch, r0, c)
            c += w + 2

    _add_decorative_line(grid, deco_color)
    return grid


def _gen_attempt() -> np.ndarray:
    long_col = int(np.random.choice(LINE_COLORS))
    short_col = int(LINE_COLORS[0] + LINE_COLORS[1] - long_col)
    canonical = _build_canonical(long_col, short_col)
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            return _gen_attempt()
        except Exception:
            continue
    raise RuntimeError("failed to generate input for 291dc1e1 combo D9/0+D14/1")
