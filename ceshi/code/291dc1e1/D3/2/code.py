# Variant of 291dc1e1
# Dimension: D3 — 源对象形状与识别 (src)
# Change: 实心矩形（thin_dim=2, thick_dim 为偶数 2/4/6/8），分布在两条平行band中... → 带颜色分组的矩形（同色矩形为一组）

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for the D3 variant

Reference-task note:
- The workspace does not contain a solver file for source task `b5ca7ac4`; only its
  official I/O exists. Per instruction, the solver-read step is skipped because the
  solver file is absent. The only borrowed idea is "color grouping".

a) Concrete behavior and trace
- Original input: two colored edge lines define a coordinate system; every interior
  object is a 2xh solid rectangle. Output stacks rectangles after sorting them by
  `(distance to blue edge, distance to red edge, tie-breaks along the corner)`.
- Variant input: rectangles are still 2xh solid rectangles, but colors now repeat.
  Same-colored rectangles form a group. Example in canonical orientation:
  slot order by the old rule is `[green, yellow, green, yellow]`.
  Original output would stack `green, yellow, green, yellow`.
  Variant output first sorts inside each color group by the old key, then sorts the
  groups by the leading member's old key, so the result becomes
  `green, green, yellow, yellow`.
- Therefore the changed step is not object detection geometry, but object identity:
  color is now a grouping feature that overrides the old fully-flat ordering.

b) Complexity calibration
- The variant uses only 2 or 3 repeated rectangle colors, and every color appears at
  least twice. This is enough repetition for a human to notice grouping, while keeping
  the hypothesis space small.
- Widths still come from the old discrete set `{2, 4, 6, 8}` and objects stay in two
  parallel bands, so the puzzle remains recognizably close to the source task.
- The generator rejects degenerate samples where color-grouped stacking would produce
  the same output as the original per-rectangle stacking.

c) Implicit-assumption audit of the original `main()`
- Corner detection assumes a unique marker at one corner adjacent to two different line
  colors. This still holds.
- Blue/red orientation detection assumes the two edge lines are intact. This still holds.
- `line_mask` removing the full corner row and column assumes objects never touch those
  lines. The generator preserves a one-cell gap.
- Rectangle extraction previously only required "no background inside bbox"; that was
  enough because rectangles were solid or nearly solid. In this variant, grouping depends
  on color identity, so the solver must explicitly attach a dominant color to each bbox.
- The old flat `rects.sort(...)` encoded the full output order. That assumption no longer
  holds; now the flat sort becomes an intra-group sort and also the key used to order
  whole groups by their first member.
- Rotation after sorting is still valid because grouping depends on input color, and
  rotating a rectangle does not alter its group membership.

d) Placement legality
- The generator places rectangles only in rows `2:4` and `5:7` of a canonical 8-row
  grid, with a two-column gap between neighboring rectangles and a one-cell gap from
  the edge lines. This prevents overlap, avoids line contact, and keeps component
  extraction trivial.
- Grid width is derived from actual band usage and capped at 30 columns, so all pastes
  are in-bounds.

e) Visual observability
- Repeated colors are the only new cue, so every sample must contain at least two
  same-colored rectangles and at least one interleaving pattern such as `A,B,A,B`.
- Because the output becomes contiguous color blocks instead of an interleaved sequence,
  the new rule is visible in 2-3 examples without requiring hidden metadata.

f) Visual cleanliness
- Objects remain plain solid rectangles, spacing is uniform, and there are no internal
  decorations. The only non-background elements are the two axis lines, the corner
  marker, and the rectangles, so regions stay easy to parse.

g) Color-change inferability
- No recoloring occurs. Colors are stable identifiers of groups, not arbitrary mappings.
  This avoids the usual ambiguity of color remapping tasks.

9) New assumptions introduced by the variant and how the generator guarantees them
- Assumption: each meaningful component has a stable dominant color. Guarantee: every
  generated rectangle is monochrome.
- Assumption: at least two color groups exist, and each group size is >= 2. Guarantee:
  group sizes are sampled from that constrained family only.
- Assumption: grouping materially changes the output. Guarantee: generation is rejected
  if grouped solving and original flat solving produce identical outputs.
- Assumption: tie-breaks are deterministic. Guarantee: the solver uses the old geometric
  key plus color as a final group-level tie-breaker.
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
RECT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_OPTIONS = np.array([2, 4, 6, 8], dtype=int)


def _locate_axes(grid: np.ndarray, bg: int) -> tuple[int, int, int, str, str]:
    H, W = grid.shape
    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(grid[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(grid[r + 1, c]))
        if r == H - 1:
            nbr.append(int(grid[r - 1, c]))
        if c == 0:
            nbr.append(int(grid[r, c + 1]))
        if c == W - 1:
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
        while 0 <= c < W and int(grid[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(grid[r, cc]) == int(vcol):
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


def _along_from_corner(
    box: tuple[int, int, int, int], orient: str, corner_r: int, corner_c: int
) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if corner_r == 0 else -r1)
    return int(c0 if corner_c == 0 else -c1)


def _extract_rectangles(grid: np.ndarray, bg: int, corner_r: int, corner_c: int, marker: int) -> list[dict]:
    H, W = grid.shape
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[corner_r, :] = True
    line_mask[:, corner_c] = True

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


def _solve(grid: np.ndarray, *, group_by_color: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _locate_axes(g, bg)
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    rects = _extract_rectangles(g, bg, cr, cc, marker)
    for rect in rects:
        box = rect["box"]
        rect["sort_key"] = (
            int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
            int(_dist_to_edge(box, red_orient, red_edge_idx)),
            int(_along_from_corner(box, blue_orient, cr, cc)),
            int(_along_from_corner(box, red_orient, cr, cc)),
        )

    if group_by_color:
        grouped: dict[int, list[dict]] = {}
        for rect in rects:
            grouped.setdefault(int(rect["color"]), []).append(rect)

        ordered: list[dict] = []
        groups = []
        for color, items in grouped.items():
            items.sort(key=lambda rect: rect["sort_key"])
            groups.append((items[0]["sort_key"], int(color), items))
        groups.sort(key=lambda item: (item[0], int(item[1])))
        for _group_key, _color, items in groups:
            ordered.extend(items)
    else:
        ordered = sorted(rects, key=lambda rect: rect["sort_key"])

    seq = []
    for rect in ordered:
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
    Variant rule:
    - Detect the same rectangles as in the original task.
    - Compute the original geometric sort key for every rectangle.
    - Group rectangles by their fill color.
    - Sort inside each color group by the original key.
    - Sort groups by the first rectangle of each group under that same key.
    - If the blue edge is vertical, rotate every rectangle 90 degrees CCW.
    - Stack the resulting sequence top-to-bottom, centered horizontally.
    """
    return _solve(grid, group_by_color=True)


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
    for _ in range(200):
        total = int(np.random.randint(4, 8))
        n0 = int(np.random.randint(2, min(4, total - 2) + 1))
        n1 = int(total - n0)
        if 2 <= n1 <= 4:
            return n0, n1
    raise RuntimeError("failed to sample band counts")


def _sample_band_widths(count: int) -> tuple[list[int], int]:
    for _ in range(200):
        widths = [int(np.random.choice(WIDTH_OPTIONS)) for _ in range(count)]
        used = int(sum(widths) + 2 * (count - 1))
        if used <= 25:
            return widths, used
    raise RuntimeError("failed to sample band widths")


def _sample_group_sizes(total: int) -> list[int]:
    group_count = int(np.random.choice([2, 3] if total >= 6 else [2]))
    sizes = np.full(group_count, 2, dtype=int)
    remaining = int(total - 2 * group_count)
    while remaining > 0:
        idx = int(np.random.randint(group_count))
        sizes[idx] += 1
        remaining -= 1
    sizes = np.array(sizes, dtype=int)
    np.random.shuffle(sizes)
    return [int(v) for v in sizes]


def _interleaved_color_sequence(group_sizes: list[int], colors: np.ndarray) -> list[int]:
    order = np.arange(len(group_sizes), dtype=int)
    np.random.shuffle(order)
    remaining = np.array(group_sizes, dtype=int)
    seq: list[int] = []
    while int(remaining.sum()) > 0:
        active = [int(idx) for idx in order if int(remaining[int(idx)]) > 0]
        for idx in active:
            seq.append(int(colors[int(idx)]))
            remaining[int(idx)] -= 1
    return seq


def _build_canonical() -> np.ndarray:
    n0, n1 = _sample_band_counts()
    band0, used0 = _sample_band_widths(n0)
    band1, used1 = _sample_band_widths(n1)
    long_dim = int(2 + max(used0, used1) + np.random.choice([1, 2, 3]))
    assert long_dim <= 30

    total = int(n0 + n1)
    group_sizes = _sample_group_sizes(total)
    colors = np.random.choice(RECT_COLORS, size=len(group_sizes), replace=False)
    slot_colors = _interleaved_color_sequence(group_sizes, colors)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    slot_idx = 0
    for r0, widths in ((2, band0), (5, band1)):
        c = 2
        for width in widths:
            patch = np.full((2, int(width)), int(slot_colors[slot_idx]), dtype=int)
            paste(grid, patch, int(r0), int(c))
            c += int(width) + 2
            slot_idx += 1

    assert slot_idx == len(slot_colors)
    return grid


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    grouped = _solve(grid, group_by_color=True)
    flat = _solve(grid, group_by_color=False)
    if np.array_equal(grouped, flat):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D3 variant")
