# Variant of 291dc1e1
# Dimension: D11 — 颜色变换——堆叠时的颜色处理 (transform)
# Change: 矩形颜色在堆叠过程中保持不变，直接复制到输出... → 按排序顺序渐变着色

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for the D11 variant
===================================

a) Concrete behavior and trace
------------------------------
Input still uses the same spatial grammar as the source task:
- background color 8
- one marked corner (color 0)
- one horizontal edge line and one vertical edge line touching the corner
- several solid 2-row rectangles in the interior, separated by background gaps

The original task keeps each rectangle's interior colors after sorting/stacking.
This variant changes only the stack-time coloring:
- sort rectangles exactly as before
- rotate them exactly as before when the blue axis is vertical
- then overwrite the whole rectangle with a rank color:
  rank 1 -> color 3
  rank 2 -> color 4
  rank 3 -> color 5
  rank 4 -> color 6
  rank 5 -> color 7
  rank 6 -> color 9

Small trace:
- Suppose the sorted rectangles are widths [4, 2, 6].
- Original output for blue-horizontal would stack the original three patches.
- Variant output stacks shapes [2x4, 2x2, 2x6], but the first becomes all-3,
  the second all-4, the third all-5.
- If blue is vertical, shapes become [4x2, 2x2, 6x2] first, then receive
  colors [3, 4, 5].

The changed stage is therefore the final "build seq and paste into output"
block. Extraction, ranking and rotation stay unchanged.

b) Complexity calibration
-------------------------
The rank-color mapping must be inferable from 2-4 training pairs, so the task
should not use too many distinct rank buckets. The generator therefore keeps
the number of rectangles in [4, 6], matching the six fixed rank colors above.
That avoids ambiguous color cycling and ensures repeated evidence for the first
few ranks across examples.

To make the recoloring visibly different from the source task, input rectangles
use colors chosen to differ from their assigned output rank color, and many of
them contain a few minority pixels. This preserves the original "multicolor
patch copied verbatim" expectation while making the overwrite easy to notice.

c) Hidden-assumption audit against the source solver
----------------------------------------------------
Relevant source-solver assumptions and whether they still hold:

- `sub = crop_bbox(g, b)` + `if (sub == bg).any(): continue`
  Assumption: each object is a filled rectangle with no background holes.
  Still required. Generator preserves that.

- `mask = (g != bg) & (~line_mask) & (g != marker)`
  Assumption: rectangle interiors never use the marker color 0, otherwise the
  component mask would develop holes or split components. The source generator
  already avoided this accidentally; the variant now enforces it deliberately.

- `rects.sort(...)`
  Assumption: order is determined only by geometry, not by rectangle colors.
  Still valid, and this is exactly what D11 needs because colors now encode rank
  only in the output.

- `a = np.rot90(a, k=1)` when blue is vertical
  Assumption: rotation happens before stacking. Still valid. Recoloring is
  applied after this rotation so the output color depends only on rank, not on
  original pigment placement.

- `paste(out, a, ...)`
  Source assumption: `a` already contains the final colors to display. This is
  the one place where the old task depended on color preservation. The variant
  must replace `a` with a uniform rank-colored patch before pasting.

d) Placement legality
---------------------
Generator guarantees:
- rectangles are placed only at rows 2 and 5 in canonical form
- rectangles start at column 2, leaving one background column from the axes
- rectangles in the same band are separated by 2 background columns
- both bands fit inside the chosen width
- all rectangles are 4-connected and mutually disconnected
- transforms are only flips/rotations of the canonical legal layout, so no
  overlap or out-of-bounds issues are introduced afterward

e) Visual observability
-----------------------
With 4-6 rectangles per sample and a fixed color ladder [3, 4, 5, 6, 7, 9],
the solver sees repeated evidence that "first stacked rectangle is always 3,
second is always 4, ..." even though input colors vary. Because the input often
contains minority pixels, the source-task explanation "it just copies colors"
stops matching the training pairs.

f) Visual cleanliness
---------------------
The layout remains sparse and banded exactly like the source task: two line
axes plus two rows of separated rectangles. Minority pixels are capped at a few
cells per rectangle, so they add evidence for overwrite without turning the
input into noise.

g) Color-change inferability
----------------------------
The mapping is intentionally small, fixed and position-based. Colors are not
conditioned on rectangle size or source color, only on sorted rank. That gives
humans a compact rule to infer from few examples.

New assumptions introduced by this variant
------------------------------------------
1. The number of rectangles must not exceed the number of rank colors.
   Generator enforces 4-6 rectangles; solver asserts the bound.
2. Marker color 0 must not appear inside rectangles.
   Generator enforces this because the component mask excludes marker cells.
3. Input rectangle dominant colors should differ from their assigned output
   rank color if we want the D11 change to be observable.
   Generator enforces this so the variant does not degenerate back to the
   source behavior on sampled tasks.
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
RANK_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
INPUT_DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_CHOICES = np.array([2, 4, 6, 8], dtype=int)


def _locate_layout(g: np.ndarray) -> tuple[int, int, int, int, str, str]:
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
    return int(bg), int(cr), int(cc), int(marker), blue_orient, red_orient


def _sorted_rectangles(g: np.ndarray) -> tuple[int, str, list[dict]]:
    H, W = g.shape
    bg, cr, cc, marker, blue_orient, red_orient = _locate_layout(g)

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
    return int(bg), blue_orient, rects


def _stack_rectangles(arrays: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(a.shape[1]) for a in arrays))
    out_h = int(sum(int(a.shape[0]) for a in arrays))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for a in arrays:
        h, w = map(int, a.shape)
        out_c = int((out_w - w) // 2)
        paste(out, a, int(out_r), int(out_c))
        out_r += int(h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to the two edge lines.

    Variant change: after the source task determines the rectangle order
    (and optional 90-degree CCW rotation), rectangle colors are overwritten
    by a fixed rank-color ladder instead of being preserved from the input.
    """
    g = np.array(grid, dtype=int)
    bg, blue_orient, rects = _sorted_rectangles(g)
    assert len(rects) <= len(RANK_COLORS)

    seq = []
    for idx, rect in enumerate(rects):
        a = rect["arr"]
        if blue_orient == "v":
            a = np.rot90(a, k=1)
        recolored = np.full(a.shape, int(RANK_COLORS[idx]), dtype=int)
        seq.append(recolored)

    return _stack_rectangles(seq, bg)


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


def _sprinkle_minorities(patch: np.ndarray, dominant: int, forbidden: set[int]) -> np.ndarray:
    area = int(patch.size)
    if area < 6 or np.random.rand() < 0.35:
        return patch

    max_minor = min(3, area // 3)
    if max_minor < 1:
        return patch

    n_minor = int(np.random.randint(1, max_minor + 1))
    choices = [int(c) for c in MINORITY_COLORS if int(c) not in forbidden and int(c) != dominant]
    if not choices:
        return patch

    cells = np.random.choice(area, size=n_minor, replace=False)
    colors = np.random.choice(choices, size=n_minor, replace=True)
    flat = patch.reshape(-1)
    for cell, color in zip(cells, colors):
        flat[int(cell)] = int(color)
    return patch


def _band_used(widths: list[int]) -> int:
    return int(sum(widths) + 2 * (len(widths) - 1))


def _build_patch(width: int, rank_color: int) -> np.ndarray:
    dominant_choices = [int(c) for c in INPUT_DOMINANT_COLORS if int(c) != int(rank_color)]
    dominant = int(np.random.choice(dominant_choices))
    patch = np.full((2, int(width)), dominant, dtype=int)
    return _sprinkle_minorities(patch, dominant, forbidden={BG, MARKER, int(rank_color)})


def _expected_variant_output(rect_specs: list[dict]) -> np.ndarray:
    seq = []
    for spec in rect_specs:
        width = int(spec["width"])
        color = int(spec["rank_color"])
        seq.append(np.full((2, width), color, dtype=int))
    return _stack_rectangles(seq, BG)


def _expected_legacy_output(grid: np.ndarray) -> np.ndarray:
    bg, blue_orient, rects = _sorted_rectangles(np.array(grid, dtype=int))
    seq = []
    for rect in rects:
        a = np.array(rect["arr"], copy=True)
        if blue_orient == "v":
            a = np.rot90(a, k=1)
        seq.append(a)
    return _stack_rectangles(seq, bg)


def _build_canonical() -> tuple[np.ndarray, list[dict]]:
    top_count, bottom_count = [(2, 2), (2, 3), (3, 2), (3, 3)][int(np.random.randint(4))]
    widths0 = [int(np.random.choice(WIDTH_CHOICES)) for _ in range(top_count)]
    widths1 = [int(np.random.choice(WIDTH_CHOICES)) for _ in range(bottom_count)]

    if all(w == 2 for w in widths0 + widths1):
        if np.random.rand() < 0.5:
            widths0[int(np.random.randint(len(widths0)))] = 4
        else:
            widths1[int(np.random.randint(len(widths1)))] = 4

    used0 = _band_used(widths0)
    used1 = _band_used(widths1)
    trailing = int(np.random.randint(1, 4))
    long_dim = int(2 + max(used0, used1) + trailing)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    rect_specs: list[dict] = []
    rank_idx = 0
    for r0, widths in ((2, widths0), (5, widths1)):
        c = 2
        for width in widths:
            rank_color = int(RANK_COLORS[rank_idx])
            patch = _build_patch(int(width), rank_color)
            paste(grid, patch, int(r0), int(c))
            rect_specs.append(
                {
                    "width": int(width),
                    "rank_color": rank_color,
                    "patch": patch.copy(),
                }
            )
            c += int(width) + 2
            rank_idx += 1

    return grid, rect_specs


def _gen_attempt() -> np.ndarray | None:
    canonical, rect_specs = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    expected = _expected_variant_output(rect_specs)
    legacy = _expected_legacy_output(grid)
    actual = main(grid)

    if np.array_equal(expected, legacy):
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D11 variant")
