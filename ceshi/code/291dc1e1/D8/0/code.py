# Variant of 291dc1e1
# Dimension: D8 — 多色矩形的处理 (transform)
# Change: 矩形可能有2-3个少数色像素（<=25%面积），直接保持原样参与堆叠... → 去除少数色像素（统一为主色）

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# ---------------------------------------------------------------------------
# Variant analysis / design notes
# ---------------------------------------------------------------------------
#
# a) Concrete behavior trace
# - Original task: each interior object is a filled 2xh rectangle. In the D8
#   version, some rectangles are no longer perfectly monochrome: they may carry
#   2-3 stray pixels of other colors, but the rectangle boundary is still solid,
#   so the bbox remains a full rectangle.
# - Original solver behavior on one 2x6 rectangle like
#       3 3 3 4 3 3
#       3 3 5 3 3 3
#   was: keep that exact pixel pattern, then maybe rotate, then stack it.
# - New behavior: first infer dominant color 3, repaint every minority pixel to
#   3, producing a pure 2x6 block, then rotate if needed, then stack it.
# - The only logical change is therefore inside the rectangle-extraction stage:
#   after cropping each rectangle but before stacking, run a "majority recolor"
#   cleanup. Ordering by distances to the two edge lines is unchanged.
#
# b) Complexity calibration
# - The variant should be visible, so the generator guarantees at least one
#   rectangle with width >= 4 (area >= 8), because only then can 2 minority
#   pixels still satisfy the <=25% rule.
# - Not every rectangle is dirtied. Keeping a mix of clean and dirty rectangles
#   makes the recolor rule visually inferable instead of looking like a global
#   palette remap.
# - Minority colors are limited to 1-2 colors per rectangle and 2-3 pixels
#   total, matching the stated difficulty and avoiding ambiguous "patterns".
#
# c) Hidden-assumption audit for the solver
# - Corner detection only assumes the marker color differs from the adjacent edge
#   line colors. Minority pixels are strictly interior to rectangles, so this is
#   unchanged.
# - Blue/red line orientation detection only reads the corner row/column.
#   Interior noise does not affect it.
# - Component extraction originally relied on rectangles having no background
#   inside the bbox. This still holds: the variant changes colors, not geometry.
# - Sorting relies on bbox coordinates, not on rectangle interior colors.
# - Rotation is shape-based; recoloring before or after rotation is equivalent.
# - The only place that depended on the old "keep minority pixels" rule was the
#   rectangle array stored for output. That is exactly where we now clean.
#
# d) Placement legality in the generator
# - The generator keeps the original two-band layout and the same edge-line /
#   corner transforms, so rectangles never overlap and always stay away from the
#   marker row/column.
# - Minority pixels are written only inside already allocated rectangle cells, so
#   they cannot create new components or change object geometry.
#
# e) Visual observability
# - Each generated input contains at least one dirty rectangle, and the output is
#   all pure rectangles. This guarantees the new behavior changes the answer on
#   the same input, instead of silently degenerating to the original task.
# - The generator favors at least one larger rectangle so the cleanup can be
#   spotted by eye in 2-4 examples.
#
# f) Visual cleanliness
# - Rectangles remain axis-aligned blocks with spacing between them in the input.
# - Adjacent rectangles in the canonical output order are assigned different
#   dominant colors to reduce accidental visual merging after stacking.
#
# g) Color-change inferability
# - There is no arbitrary color mapping. The rule is local and structural:
#   "inside each rectangle, minority colors collapse to the rectangle's majority
#   color". That is easier to infer than a free-form recolor table.
# - This follows the same complexity baseline as reference task 20270e3b: first
#   use a special color arrangement for recognition, then erase that specialness
#   by recoloring to the main object color in the output.


def _clean_to_dominant(rect: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(rect, return_counts=True)
    dominant = int(vals[int(np.argmax(counts))])
    out = rect.copy()
    out[out != dominant] = dominant
    return out


def _solve(grid: np.ndarray, *, clean_rectangles: bool) -> np.ndarray:
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

    # Assumption preserved under D8: each component's bbox is still a full
    # rectangle; the only variation is internal minority recolor noise.
    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        if clean_rectangles:
            sub = _clean_to_dominant(sub)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two colored edge lines.

    D8 variant change:
    - Rectangles may contain 2-3 minority-color pixels.
    - Before stacking, recolor every minority pixel to the rectangle's dominant
      color, so the final output contains only solid rectangles.
    """
    return _solve(grid, clean_rectangles=True)


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


def _sprinkle_minorities(patch: np.ndarray, dominant: int, *, force: bool) -> tuple[np.ndarray, int]:
    area = int(patch.size)
    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch, 0
    if not force and np.random.rand() < 0.65:
        return patch, 0

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch, n_minor


def _build_canonical() -> tuple[np.ndarray, dict] | None:
    # Slightly larger minimum width than the seed generator so the D8 change is
    # visually observable: we need at least one rectangle with area >= 8.
    long_dim = int(np.random.randint(13, 29))
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
    if not options:
        return None

    max_used, pairs = options[int(np.random.randint(len(options)))]
    n0, n1, eligible = pairs[int(np.random.randint(len(pairs)))]
    max_band = int(np.random.choice(eligible))
    max0 = min(max_used, 10 * n0 - 2)
    max1 = min(max_used, 10 * n1 - 2)
    used0 = int(max_used if max_band == 0 else np.random.choice(np.arange(4 * n0 - 2, max0 + 1, 2)))
    used1 = int(max_used if max_band == 1 else np.random.choice(np.arange(4 * n1 - 2, max1 + 1, 2)))
    band0 = _distribute_band_widths(n0, used0)
    band1 = _distribute_band_widths(n1, used1)

    specs = []
    for r0, widths in ((2, band0), (5, band1)):
        c = 2
        for w in widths:
            specs.append({"r": int(r0), "c": int(c), "w": int(w)})
            c += int(w) + 2

    eligible_dirty = [i for i, spec in enumerate(specs) if int(spec["w"]) >= 4]
    if not eligible_dirty:
        return None

    dirty_target = int(np.random.randint(1, min(3, len(eligible_dirty)) + 1))
    dirty_indices = set(map(int, np.random.choice(eligible_dirty, size=dirty_target, replace=False)))

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    prev_color = None
    dirty_pixels = 0
    for idx, spec in enumerate(specs):
        choices = [int(c) for c in DOMINANT_COLORS if int(c) != prev_color]
        color = int(np.random.choice(choices))
        patch = np.full((2, int(spec["w"])), color, dtype=int)
        patch, n_minor = _sprinkle_minorities(patch, color, force=idx in dirty_indices)
        dirty_pixels += int(n_minor)
        paste(grid, patch, int(spec["r"]), int(spec["c"]))
        prev_color = color

    meta = {
        "rectangles": int(len(specs)),
        "dirty_rectangles": int(len(dirty_indices)),
        "dirty_pixels": int(dirty_pixels),
        "eligible_dirty_rectangles": int(len(eligible_dirty)),
    }
    return grid, meta


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    built = _build_canonical()
    if built is None:
        return None
    canonical, meta = built
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner), meta


def generate_input() -> np.ndarray:
    for _ in range(400):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        grid, meta = attempt
        try:
            cleaned = main(grid)
            legacy = _solve(grid, clean_rectangles=False)
        except Exception:
            continue

        # Generator-side guarantees:
        # - at least one rectangle truly uses the D8 behavior
        # - the new solver changes the output on this input
        # - the cleaned output no longer contains the old minority-noise pattern
        if int(meta["dirty_rectangles"]) < 1 or int(meta["dirty_pixels"]) < 2:
            continue
        if np.array_equal(cleaned, legacy):
            continue
        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D8")
