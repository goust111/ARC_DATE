# Combo variant of 291dc1e1 (index 34)
# combo_spec: D3/2+D10/0+D11/0
# Name: 色组矩形+分步堆叠+统一着色

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for combo D3/2 + D10/0 + D11/0

a) Concrete behavior and trace
- Base task: detect interior filled rectangles, sort every rectangle globally by
  `(distance_to_blue, distance_to_red, along_blue, along_red)`, rotate each one
  90 degrees CCW when the blue axis is vertical, then stack all rectangles
  continuously.
- Combo task: keep the same geometry detection and optional rotation, but change
  the sequencing/rendering pipeline in three linked ways.
  1. Partition rectangles by blue-distance band, so the near-blue band is output
     first and the far band later.
  2. Inside each band, group rectangles by their dominant input color.
  3. Inside each color group, sort by the original geometric key; sort groups by
     the first member's geometric key; then stack the groups contiguously.
  4. Insert one full background separator row between bands.
  5. Recolor every pasted rectangle to the blue-axis color, so the output keeps
     only geometry/order evidence and loses the input color labels.
- Small canonical trace:
  near band slots by old order: `A(red,w=2), B(green,w=6), C(red,w=8), D(green,w=4)`
  far band slots by old order: `E(yellow,w=2), F(blue,w=6), G(yellow,w=4)`
  Base output order: `A B C D E F G`.
  Combo output order: near band becomes `(red: A,C) (green: B,D)`, then one bg
  row, then far band becomes `(yellow: E,G) (blue: F)`. After recoloring, the
  output keeps widths/order/separator but all blocks become the same blue color.
  The human must therefore infer grouping from the input colors and verify it via
  the changed width sequence in the stacked output.

b) Complexity calibration
- Colors are restricted to 2 groups per band and each group has repeated evidence.
  This keeps the grouping hypothesis small enough for ARC while still forcing a
  real nontrivial inference.
- Band structure stays the original clean 2-band layout, so the D10 separator is
  visually obvious and does not compete with noisy placement.
- Generator rejects cases where combo output collapses to `stepwise+recolor` with
  no grouping effect, or to `grouped+recolor` with no band-step effect.

c) Implicit-assumption audit of the original main()
- Corner detection assumed a unique marker at a corner adjacent to two different
  line colors. Preserved.
- Blue/red axis inference assumed intact full edge lines. Preserved.
- Component extraction assumed objects do not touch the corner row/column and that
  each component's bbox has no background holes. Preserved by generation.
- Original `rects.sort(...)` assumed a single flat total order was the final
  output order. No longer valid: now that key becomes a per-rectangle geometric
  key used inside color groups and to order groups inside each band.
- Original copy-through of rectangle arrays assumed colors survive into output. No
  longer valid: combo explicitly overwrites all rectangle colors with the blue
  axis color.
- Original height computation assumed no separator rows. No longer valid because
  the multi-step output reserves one background row between processed bands.

d) Placement legality
- Canonical generation still uses the original non-overlapping scaffold:
  edge row/column, one-cell gap from each edge, rectangles only in rows `2:4` and
  `5:7`, and two background columns between neighbors along the band direction.
- Width budgets are sampled so each band's used span is at most 25, making
  `long_dim = 2 + max(used0, used1) + trailing` stay within 30.
- After optional flip/rotation of the whole input, objects remain disjoint and the
  two distinct blue-distance bands remain recoverable.

e) Visual observability
- The input shows repeated colors within each band, typically interleaved.
- The output shows two processing phases separated by a blank row and, within each
  phase, a width/height sequence consistent with color grouping rather than with
  the naive red-distance order.
- Because the output is recolored uniformly, the only way to explain the changed
  within-band order is to use the input colors as grouping labels. That is exactly
  the intended fusion behavior.

f) Visual cleanliness
- Rectangles are monochrome in the input; no speckles are used here because the
  fused task already asks the solver to combine grouping, phased stacking, and
  recoloring. Clean monochrome components make the grouping cue unambiguous.
- Output adds only one clean separator row between the two band piles.

g) Color-change inferability
- Input object colors come from a small non-axis palette `{3,4,5,6,7,9}` and the
  output color is always the blue axis color `1`. This keeps the recolor mapping
  stable and easy to infer.
- Because colors disappear in the output, the generator guarantees repeated colors
  per band and rejects samples where grouped ordering would be indistinguishable
  after recoloring.

9) New assumptions introduced by the combo and generator guarantees
- Assumption: each component has a stable dominant color. Guarantee: rectangles are
  generated as solid monochrome patches.
- Assumption: at least one band contains an interleaving color pattern whose
  grouped order differs from the plain red-distance order. Guarantee: generator
  rejects samples where combo output equals `stepwise+recolor` without grouping.
- Assumption: there are exactly two nonempty blue-distance bands. Guarantee:
  generation always instantiates both canonical bands and rejects malformed cases.
- Assumption: group ordering is deterministic. Guarantee: groups are ordered by the
  first member's geometric key with color as a final tie-breaker.
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
RECT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_OPTIONS = np.array([2, 4, 6, 8], dtype=int)


def _locate_axes(grid: np.ndarray) -> tuple[int, int, int, int, int, str, str]:
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
    return int(bg), int(cr), int(cc), int(marker), int(blue_col), blue_orient, red_orient


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
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


def _along_from_corner(
    box: tuple[int, int, int, int], orient: str, corner_r: int, corner_c: int
) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if corner_r == 0 else -r1)
    return int(c0 if corner_c == 0 else -c1)


def _extract_rectangles(grid: np.ndarray) -> tuple[int, int, str, str, list[dict]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, cr, cc, marker, _blue_col, blue_orient, red_orient = _locate_axes(g)

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict] = []
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        vals = [int(v) for v in sub.flatten() if int(v) != int(bg)]
        assert vals
        color = int(Counter(vals).most_common(1)[0][0])
        sort_key = (
            int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
            int(_dist_to_edge(box, red_orient, red_edge_idx)),
            int(_along_from_corner(box, blue_orient, cr, cc)),
            int(_along_from_corner(box, red_orient, cr, cc)),
        )
        rects.append(
            {
                "box": box,
                "arr": sub.copy(),
                "color": int(color),
                "db": int(sort_key[0]),
                "dr": int(sort_key[1]),
                "sort_key": sort_key,
            }
        )

    assert rects
    return int(bg), int(cr), blue_orient, red_orient, rects


def _ordered_rectangles(
    rects: list[dict], *, group_by_color: bool, bandwise: bool
) -> list[list[dict]]:
    if bandwise:
        band_values = sorted({int(rect["db"]) for rect in rects})
        bands = [[rect for rect in rects if int(rect["db"]) == int(db)] for db in band_values]
    else:
        bands = [list(rects)]

    ordered_bands: list[list[dict]] = []
    for band in bands:
        if group_by_color:
            grouped: dict[int, list[dict]] = {}
            for rect in band:
                grouped.setdefault(int(rect["color"]), []).append(rect)

            groups: list[tuple[tuple[int, int, int, int], int, list[dict]]] = []
            for color, items in grouped.items():
                items = sorted(items, key=lambda rect: rect["sort_key"])
                groups.append((items[0]["sort_key"], int(color), items))
            groups.sort(key=lambda item: (item[0], int(item[1])))

            ordered: list[dict] = []
            for _first_key, _color, items in groups:
                ordered.extend(items)
        else:
            ordered = sorted(band, key=lambda rect: rect["sort_key"])
        ordered_bands.append(ordered)
    return ordered_bands


def _solve(
    grid: np.ndarray, *, group_by_color: bool, bandwise: bool, recolor: bool
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, _cr, _blue_orient, _red_orient, rects = _extract_rectangles(g)
    _bg2, _cr2, _cc2, _marker2, blue_col, blue_orient, _red_orient2 = _locate_axes(g)
    bands = _ordered_rectangles(rects, group_by_color=group_by_color, bandwise=bandwise)

    out_bands: list[list[np.ndarray]] = []
    for band in bands:
        arrs: list[np.ndarray] = []
        for rect in band:
            arr = rect["arr"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            if recolor:
                arr = np.full(arr.shape, int(blue_col), dtype=int)
            arrs.append(arr)
        out_bands.append(arrs)

    out_w = int(max(int(arr.shape[1]) for band in out_bands for arr in band))
    out_h = int(sum(int(arr.shape[0]) for band in out_bands for arr in band))
    if bandwise:
        out_h += max(0, len(out_bands) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for band_idx, band in enumerate(out_bands):
        if bandwise and band_idx > 0:
            out_r += 1
        for arr in band:
            h, w = map(int, arr.shape)
            out_c = int((out_w - w) // 2)
            paste(out, arr, int(out_r), int(out_c))
            out_r += int(h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - detect the original rectangles and their blue/red edge distances
    - process rectangles in ascending blue-distance bands
    - inside each band, group same-colored rectangles together
    - sort within each color group by the original geometric key
    - sort groups by their leading member's geometric key
    - rotate rectangles 90 degrees CCW if the blue edge is vertical
    - recolor every pasted rectangle to the blue edge color
    - insert one full background row between consecutive bands
    """
    return _solve(grid, group_by_color=True, bandwise=True, recolor=True)


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


def _sample_band_count() -> int:
    return int(np.random.choice([3, 4]))


def _sample_band_widths(count: int) -> tuple[list[int], int]:
    for _ in range(300):
        widths = [int(np.random.choice(WIDTH_OPTIONS)) for _ in range(count)]
        used = int(sum(widths) + 2 * (count - 1))
        if used <= 25:
            return widths, used
    raise RuntimeError("failed to sample band widths")


def _group_sizes_for_band(count: int) -> tuple[int, int]:
    if count == 3:
        return (2, 1)
    if np.random.rand() < 0.6:
        return (2, 2)
    return (3, 1)


def _interleaved_colors(count: int, colors: tuple[int, int]) -> list[int]:
    if count == 3:
        return [int(colors[0]), int(colors[1]), int(colors[0])]
    if np.random.rand() < 0.5:
        return [int(colors[0]), int(colors[1]), int(colors[0]), int(colors[1])]
    return [int(colors[1]), int(colors[0]), int(colors[1]), int(colors[0])]


def _build_canonical() -> np.ndarray:
    near_count = _sample_band_count()
    far_count = _sample_band_count()
    near_widths, near_used = _sample_band_widths(near_count)
    far_widths, far_used = _sample_band_widths(far_count)
    long_dim = int(2 + max(near_used, far_used) + np.random.choice([1, 2, 3]))
    assert long_dim <= 30

    color_pool = list(map(int, np.random.choice(RECT_COLORS, size=4, replace=False)))
    near_colors = (int(color_pool[0]), int(color_pool[1]))
    far_colors = (int(color_pool[2]), int(color_pool[3]))
    near_seq = _interleaved_colors(near_count, near_colors)
    far_seq = _interleaved_colors(far_count, far_colors)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, widths, colors in ((2, near_widths, near_seq), (5, far_widths, far_seq)):
        c = 2
        for width, color in zip(widths, colors, strict=True):
            patch = np.full((2, int(width)), int(color), dtype=int)
            paste(grid, patch, int(r0), int(c))
            c += int(width) + 2

    return grid


def _band_profile(grid: np.ndarray) -> list[int]:
    _bg, _cr, _blue_orient, _red_orient, rects = _extract_rectangles(grid)
    counts: dict[int, int] = {}
    for rect in rects:
        db = int(rect["db"])
        counts[db] = counts.get(db, 0) + 1
    return [counts[key] for key in sorted(counts)]


def _group_run_count(grid: np.ndarray) -> tuple[int, int]:
    _bg, _cr, _blue_orient, _red_orient, rects = _extract_rectangles(grid)
    combo_bands = _ordered_rectangles(rects, group_by_color=True, bandwise=True)
    flat_bands = _ordered_rectangles(rects, group_by_color=False, bandwise=True)

    def run_count(bands: list[list[dict]]) -> int:
        total = 0
        for band in bands:
            if not band:
                continue
            runs = 1
            for i in range(1, len(band)):
                if int(band[i]["color"]) != int(band[i - 1]["color"]):
                    runs += 1
            total += runs
        return total

    return run_count(flat_bands), run_count(combo_bands)


def _combo_behavior_present(grid: np.ndarray) -> bool:
    combo = _solve(grid, group_by_color=True, bandwise=True, recolor=True)
    no_group = _solve(grid, group_by_color=False, bandwise=True, recolor=True)
    no_band = _solve(grid, group_by_color=True, bandwise=False, recolor=True)
    no_recolor = _solve(grid, group_by_color=True, bandwise=True, recolor=False)
    if np.array_equal(combo, no_group):
        return False
    if np.array_equal(combo, no_band):
        return False
    if np.array_equal(combo, no_recolor):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    try:
        band_sizes = _band_profile(grid)
        if len(band_sizes) != 2 or min(band_sizes) < 3:
            return None
        flat_runs, combo_runs = _group_run_count(grid)
        if combo_runs >= flat_runs:
            return None
        if not _combo_behavior_present(grid):
            return None
        out = main(grid)
        if not any(bool((out[r, :] == BG).all()) for r in range(1, int(out.shape[0]) - 1)):
            return None
        if set(map(int, np.unique(out))) != {int(BG), int(BLUE)}:
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo input for 291dc1e1/34")
