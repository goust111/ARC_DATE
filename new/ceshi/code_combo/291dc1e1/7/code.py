# Combo variant of 291dc1e1 (index 7)
# combo_spec: D3/2+D10/0
# Name: 色组矩形+分band分步堆叠

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for the combo variant

a) Concrete behavior and trace
- Base task: detect the corner marker plus blue/red edge lines, extract the interior
  rectangles, sort all rectangles globally by `(db, dr, along_blue, along_red)`,
  optionally rotate if the blue edge is vertical, then stack in one uninterrupted
  pile.
- Combo task: preserve the same geometry, but introduce two organizing axes:
  `db` still defines the processing steps (near blue band first, far blue band
  second), and inside each band, same-colored rectangles become groups.
- Small canonical trace with top-left corner:
  near band by `dr`: `[green, yellow, green]`
  far band by `dr`: `[yellow, green, yellow]`
  Original output: `green, yellow, green, yellow, green, yellow`
  D10-only output: `green, yellow, green | sep | yellow, green, yellow`
  D3-only output: `green, green, green, yellow, yellow, yellow`
  Combo output: `green, green, yellow | sep | yellow, yellow, green`
- The difference is therefore not a simple concatenation of two rules: band
  membership controls the macro step order, while color controls the micro order
  inside each step.

b) Complexity calibration
- The generator keeps exactly two blue-distance bands, because the separator row is
  the visual cue for the multi-step half of the rule.
- Each band contains 3 or 4 rectangles, and each band uses exactly two colors in an
  interleaved pattern such as `A,B,A` or `A,B,A,B`. This gives repeated evidence for
  color grouping without creating too many color hypotheses.
- Rectangles stay as simple solid `2 x {2,4,6,8}` blocks, matching the original
  object language and keeping the fused rule visually legible.

c) Implicit-assumption audit of the original `main()`
- Corner detection still assumes a unique non-background marker in one corner whose
  adjacent edge cells have different colors. The generator preserves that exactly.
- Blue/red orientation detection still assumes full edge lines through the corner.
  The combo variant does not change the axes.
- `line_mask[cr, :]` and `line_mask[:, cc]` assume rectangles never touch the axes.
  Canonical placement keeps a one-cell gap from both lines before any transform.
- Original rectangle extraction only needed bbox solidity. Color grouping adds a new
  dependency: each rectangle must have a stable dominant color. The generator makes
  every rectangle monochrome, and the solver still computes the dominant color
  defensively.
- The original single global sort no longer matches the task. In the combo solver,
  the global key is retained only as reference logic; the actual order is:
  partition by `db`, then within each band group by dominant color, then sort items
  inside each color group by the band-local red-distance key.
- Rotation after ordering still holds because rotation changes only output shape, not
  band membership or color-group identity.

d) Placement legality
- Canonical construction uses the original 8-row layout: blue edge on row 0, red edge
  on col 0, near band on rows `2:4`, far band on rows `5:7`.
- Rectangles within a band are pasted left-to-right with a 2-cell background gap, so
  components never touch. Band rows are separated by one full background row.
- Sampled total band span is capped so `long_dim <= 28`, matching the original
  physical regime and preventing overflow after pasting.

e) Visual observability
- A human looking at 2-4 examples sees two recurring facts at once:
  first, the output always has a blank separator row splitting near and far bands;
  second, within each half, same-colored rectangles that were interleaved in the
  input become contiguous blocks.
- Because both bands are non-empty and both contain repeated colors, neither cue is
  accidental or one-off.

f) Visual cleanliness
- The input still contains only the corner marker, two edge lines, and clean solid
  rectangles on a uniform background.
- No minority pixels are added in this combo variant, because color identity is now a
  core reasoning signal and visual noise would make the grouping cue weaker.

g) Color-change inferability
- There is no recoloring. Rectangle colors persist from input to output and serve only
  as grouping labels.
- Only two rectangle colors are used per sample, and both repeat multiple times. This
  is enough support for humans to infer that equal colors should stay together.

9) New assumptions introduced by the combo solver and generator guarantees
- Assumption: there are exactly two distinct `db` bands worth separating.
  Guarantee: canonical construction always uses the original near/far rows only.
- Assumption: color grouping matters inside at least one band.
  Guarantee: each band is generated with an interleaved two-color pattern, so combo
  order differs from pure D10 order.
- Assumption: the fused rule is not degenerate into the original or either single
  variant.
  Guarantee: generation rejects any sample where combo output equals original,
  D3-only, or D10-only output.
- Assumption: group ordering is deterministic.
  Guarantee: within a band, groups are ordered by the first member's band-local key,
  with color as a final tie-breaker.
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
RECT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_OPTIONS = np.array([2, 4, 6, 8], dtype=int)


def _locate_axes(grid: np.ndarray, bg: int) -> tuple[int, int, int, str, str]:
    h, w = grid.shape
    corners = [(0, 0), (0, int(w - 1)), (int(h - 1), 0), (int(h - 1), int(w - 1))]
    corner = None
    for r, c in corners:
        if int(grid[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(grid[r + 1, c]))
        if r == h - 1:
            nbr.append(int(grid[r - 1, c]))
        if c == 0:
            nbr.append(int(grid[r, c + 1]))
        if c == w - 1:
            nbr.append(int(grid[r, c - 1]))
        if nbr and all(int(n) != int(grid[r, c]) for n in nbr):
            corner = (int(r), int(c))
            break
    assert corner is not None

    cr, cc = map(int, corner)
    marker = int(grid[cr, cc])
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(grid[cr, cc + hstep])
    vcol = int(grid[cr + vstep, cc])

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
        red_col = int(vcol if hcol == BLUE else hcol)
    else:
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < w and int(grid[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < h and int(grid[r, cc]) == int(vcol):
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


def _extract_rectangles(grid: np.ndarray, bg: int, cr: int, cc: int, marker: int) -> list[dict]:
    h, w = grid.shape
    line_mask = np.zeros((h, w), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (grid != int(bg)) & (~line_mask) & (grid != int(marker))
    rects: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(grid, box)
        if bool((sub == int(bg)).any()):
            continue
        vals = [int(v) for v in sub.flatten() if int(v) != int(bg)]
        assert vals
        dominant = int(Counter(vals).most_common(1)[0][0])
        rects.append({"box": box, "arr": sub.copy(), "color": dominant})

    assert rects
    return rects


def _analyze_rectangles(grid: np.ndarray) -> tuple[int, str, str, int, int, list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _locate_axes(g, bg)
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    rects = _extract_rectangles(g, bg, cr, cc, marker)
    for rect in rects:
        box = rect["box"]
        db = int(_dist_to_edge(box, blue_orient, blue_edge_idx))
        dr = int(_dist_to_edge(box, red_orient, red_edge_idx))
        along_blue = int(_along_from_corner(box, blue_orient, cr, cc))
        along_red = int(_along_from_corner(box, red_orient, cr, cc))
        rect["db"] = db
        rect["dr"] = dr
        rect["global_key"] = (db, dr, along_blue, along_red)
        rect["band_key"] = (dr, along_red, along_blue)

    return bg, blue_orient, red_orient, cr, cc, rects


def _order_color_groups(rects: list[dict], key_name: str) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for rect in rects:
        grouped.setdefault(int(rect["color"]), []).append(rect)

    groups = []
    for color, items in grouped.items():
        ordered_items = sorted(items, key=lambda rect: rect[key_name])
        groups.append((ordered_items[0][key_name], int(color), ordered_items))
    groups.sort(key=lambda item: (item[0], int(item[1])))

    ordered: list[dict] = []
    for _lead_key, _color, items in groups:
        ordered.extend(items)
    return ordered


def _plan_bands(grid: np.ndarray, *, group_by_color: bool, separate_bands: bool) -> tuple[int, str, list[list[dict]]]:
    bg, blue_orient, _red_orient, _cr, _cc, rects = _analyze_rectangles(grid)
    if not separate_bands:
        if group_by_color:
            return bg, blue_orient, [_order_color_groups(rects, "global_key")]
        return bg, blue_orient, [sorted(rects, key=lambda rect: rect["global_key"])]

    bands: list[list[dict]] = []
    for db in sorted({int(rect["db"]) for rect in rects}):
        band_rects = [rect for rect in rects if int(rect["db"]) == int(db)]
        if group_by_color:
            bands.append(_order_color_groups(band_rects, "band_key"))
        else:
            bands.append(sorted(band_rects, key=lambda rect: rect["band_key"]))
    return bg, blue_orient, bands


def _stack_bands(bg: int, blue_orient: str, bands: list[list[dict]]) -> np.ndarray:
    arr_bands: list[list[np.ndarray]] = []
    for band in bands:
        arrs = []
        for rect in band:
            arr = rect["arr"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            arrs.append(arr)
        arr_bands.append(arrs)

    out_w = int(max(int(arr.shape[1]) for band in arr_bands for arr in band))
    out_h = int(sum(int(arr.shape[0]) for band in arr_bands for arr in band) + max(0, len(arr_bands) - 1))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for band_idx, band in enumerate(arr_bands):
        if band_idx > 0:
            out_r += 1
        for arr in band:
            hh, ww = map(int, arr.shape)
            out_c = int((out_w - ww) // 2)
            paste(out, arr, int(out_r), int(out_c))
            out_r += int(hh)
    return out


def _solve(grid: np.ndarray, *, group_by_color: bool, separate_bands: bool) -> np.ndarray:
    bg, blue_orient, bands = _plan_bands(grid, group_by_color=group_by_color, separate_bands=separate_bands)
    return _stack_bands(bg, blue_orient, bands)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Detect the original rectangles and compute their blue/red edge distances.
    - Process rectangles in two blue-distance steps: near band first, far band second.
    - Within each band, group rectangles by dominant color.
    - Sort rectangles inside each color group by `(dr, along_red, along_blue)`.
    - Order the color groups in the band by the first member's band-local key.
    - Rotate every rectangle 90 degrees CCW if the blue edge is vertical.
    - Stack the near-band pile, insert one full background separator row, then stack
      the far-band pile.
    """
    return _solve(grid, group_by_color=True, separate_bands=True)


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


def _sample_band_counts() -> tuple[int, int]:
    choices = [(3, 3), (3, 4), (4, 3), (4, 4)]
    idx = int(np.random.randint(len(choices)))
    near_count, far_count = choices[idx]
    return int(near_count), int(far_count)


def _sample_band_widths(count: int) -> tuple[list[int], int]:
    for _ in range(200):
        widths = [int(np.random.choice(WIDTH_OPTIONS)) for _ in range(count)]
        used = int(sum(widths) + 2 * (count - 1))
        if used <= 24:
            return widths, used
    raise RuntimeError("failed to sample band widths")


def _interleaved_band_colors(count: int, colors: tuple[int, int], start_idx: int) -> list[int]:
    assert count in (3, 4)
    seq = []
    for i in range(count):
        seq.append(int(colors[(start_idx + i) % 2]))
    if count == 3:
        seq[2] = int(seq[0])
    return seq


def _build_canonical() -> np.ndarray:
    near_count, far_count = _sample_band_counts()
    near_widths, near_used = _sample_band_widths(near_count)
    far_widths, far_used = _sample_band_widths(far_count)
    long_dim = int(2 + max(near_used, far_used) + np.random.choice([1, 2]))
    assert long_dim <= 28

    chosen = np.random.choice(RECT_COLORS, size=2, replace=False)
    colors = (int(chosen[0]), int(chosen[1]))
    near_start = int(np.random.randint(2))
    far_start = 1 - near_start
    near_colors = _interleaved_band_colors(near_count, colors, near_start)
    far_colors = _interleaved_band_colors(far_count, colors, far_start)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, widths, band_colors in ((2, near_widths, near_colors), (5, far_widths, far_colors)):
        c = 2
        for width, color in zip(widths, band_colors, strict=True):
            patch = np.full((2, int(width)), int(color), dtype=int)
            paste(grid, patch, int(r0), int(c))
            c += int(width) + 2

    return grid


def _band_color_profile(grid: np.ndarray) -> list[list[int]]:
    _bg, _bo, _ro, _cr, _cc, rects = _analyze_rectangles(grid)
    by_band: dict[int, list[int]] = {}
    for rect in rects:
        by_band.setdefault(int(rect["db"]), []).append(int(rect["color"]))
    return [by_band[db] for db in sorted(by_band)]


def _has_internal_separator(out: np.ndarray, bg: int) -> bool:
    for r in range(1, int(out.shape[0]) - 1):
        if bool((out[r, :] == int(bg)).all()):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    try:
        profiles = _band_color_profile(grid)
        if len(profiles) != 2:
            return None
        for colors in profiles:
            counts = Counter(colors)
            if len(colors) < 3 or len(counts) != 2 or max(counts.values()) < 2:
                return None

        combo = _solve(grid, group_by_color=True, separate_bands=True)
        original = _solve(grid, group_by_color=False, separate_bands=False)
        group_only = _solve(grid, group_by_color=True, separate_bands=False)
        band_only = _solve(grid, group_by_color=False, separate_bands=True)

        if np.array_equal(combo, original):
            return None
        if np.array_equal(combo, group_only):
            return None
        if np.array_equal(combo, band_only):
            return None
        if not _has_internal_separator(combo, BG):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant")
