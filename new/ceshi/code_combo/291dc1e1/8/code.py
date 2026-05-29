# Combo variant of 291dc1e1 (index 8)
# combo_spec: D3/2+D11/0
# Name: 色组矩形+统一重着色

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for the fused D3/2 + D11/0 variant

a) Concrete behavior and trace
- Base task: detect interior rectangles, sort them by the geometric key
  `(distance_to_blue, distance_to_red, along_blue, along_red)`, optionally rotate
  each rectangle if the blue axis is vertical, then stack while preserving colors.
- Fused task: the same geometric key is still computed, but it becomes a two-level
  ordering signal. Rectangles of the same input color form a group; each group is
  internally sorted by the old geometric key, and whole groups are ordered by the
  first rectangle of each group under that same key. After this grouped ordering is
  fixed, every output rectangle is recolored to the blue-axis color.
- Small canonical example. Suppose the old flat order is:
  `[(green, w=2), (yellow, w=6), (green, w=8), (yellow, w=4)]`.
  Base output keeps this sequence and keeps colors.
  Fused output first groups by color to get
  `[(green,2), (green,8), (yellow,6), (yellow,4)]`,
  then recolors every block to blue, so the visible output widths are `2, 8, 6, 4`
  and the input colors disappear. The solver must therefore infer grouping from the
  input, but the output preserves only geometry and the blue recolor.

b) Complexity calibration
- The generator uses only 2 or 3 rectangle colors, and each chosen color appears at
  least twice. This keeps grouping visually obvious instead of creating many singleton
  colors that would collapse back to the base task.
- Rectangle widths still come from the small discrete set `{2, 4, 6, 8}` and the
  two-band layout from the source task is preserved, so the puzzle remains close to
  291dc1e1 rather than becoming a new unrelated task.
- Degenerate samples are rejected unless grouped+recolored output differs from the
  recolor-only output. This forces the grouping rule to have a visible geometric effect
  even though output colors are unified.

c) Implicit-assumption audit of `main()`
- Corner detection assumes a unique corner marker adjacent to two full edge lines.
  The generator keeps that unchanged.
- Axis inference assumes one line is blue or, failing that, the longer line is blue.
  The generator uses the original fixed colors, so this still holds.
- Masking the entire corner row and column assumes rectangles keep a one-cell gap from
  both axes. The generator preserves the original band structure, so line masking is safe.
- Rectangle extraction assumes each object is a filled bbox without background holes.
  The generator uses solid rectangles, so component extraction remains valid.
- New fused assumption: a rectangle has a stable group identity derivable from input
  color. The solver therefore records a dominant color per rectangle. The generator
  guarantees monochrome rectangles so this dominant color is unambiguous.
- The base task assumed one global flat sort was the full output order. That no longer
  holds; the new code must first sort within groups and then sort groups themselves.
- Recoloring invalidates the old assumption that copying rectangle pixels directly is
  correct. The fused solver must explicitly replace each rectangle array by a solid
  blue-colored rectangle after rotation.

d) Placement legality
- In canonical orientation, rectangles occupy rows `2:4` and `5:7`, start one cell
  away from the red axis, and keep two background columns between neighbors. This
  prevents overlap, keeps all rectangles off the edge lines, and preserves easy
  connected-component extraction.
- Grid width is derived from actual band usage plus a small trailing margin, so every
  paste remains in bounds before and after global flips/rotations.

e) Visual observability
- Across examples, the input shows repeated rectangle colors in an interleaved pattern
  such as `A,B,A,B` or `A,B,C,A,B,C`, while the output shows a blue-only stack whose
  width order matches grouped-by-color collection rather than the original flat order.
- Because output colors are removed, the generator must make the grouped order visibly
  different in shape sequence; that is why it rejects samples where grouped+recolored
  and recolor-only outputs match.

f) Visual cleanliness
- The scene still contains only background, two edge lines, one corner marker, and
  solid rectangles. No extra overlays or internal decorations are added, so the new
  reasoning burden comes from grouping/recolor, not from visual clutter.

g) Color-change inferability
- Input rectangle colors are chosen only from `{3,4,5,6,7,9}`; blue remains the axis
  color `1` and never appears as an object color. This makes the recolor target easy to
  infer and avoids accidental samples where an object already matches the output color.
- Because only 2-3 object colors appear and each repeats, a human can use those colors
  as grouping cues in the input without being overwhelmed by many arbitrary mappings.

9) New assumptions introduced by the fused solver and generator guarantees
- Assumption: at least two color groups exist and matter. Guarantee: group sizes are
  sampled so every group has size >= 2, and at least two groups are always present.
- Assumption: grouping changes the visible stacked geometry after recolor. Guarantee:
  generation rejects any sample where grouped+recolored output equals flat+recolored.
- Assumption: recoloring is actually exercised. Guarantee: object colors exclude blue
  and at least two object colors are present, so grouped+original-color output differs
  from grouped+recolored output.
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
RECT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_OPTIONS = np.array([2, 4, 6, 8], dtype=int)


def _locate_axes(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, str, str]:
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
    return int(cr), int(cc), int(marker), int(blue_col), blue_orient, red_orient


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
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


def _along_from_corner(
    box: tuple[int, int, int, int], orient: str, corner_r: int, corner_c: int
) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if corner_r == 0 else -r1)
    return int(c0 if corner_c == 0 else -c1)


def _extract_rectangles(
    grid: np.ndarray, bg: int, corner_r: int, corner_c: int, marker: int
) -> list[dict]:
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


def _order_rectangles(
    rects: list[dict], blue_orient: str, red_orient: str, corner_r: int, corner_c: int
) -> list[dict]:
    blue_edge_idx = int(corner_c if blue_orient == "v" else corner_r)
    red_edge_idx = int(corner_c if red_orient == "v" else corner_r)

    for rect in rects:
        box = rect["box"]
        rect["sort_key"] = (
            int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
            int(_dist_to_edge(box, red_orient, red_edge_idx)),
            int(_along_from_corner(box, blue_orient, corner_r, corner_c)),
            int(_along_from_corner(box, red_orient, corner_r, corner_c)),
        )

    grouped: dict[int, list[dict]] = {}
    for rect in rects:
        grouped.setdefault(int(rect["color"]), []).append(rect)

    groups = []
    for color, items in grouped.items():
        items.sort(key=lambda rect: rect["sort_key"])
        groups.append((items[0]["sort_key"], int(color), items))
    groups.sort(key=lambda item: (item[0], int(item[1])))

    ordered: list[dict] = []
    for _group_key, _color, items in groups:
        ordered.extend(items)
    return ordered


def _solve(grid: np.ndarray, *, group_by_color: bool, recolor: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, blue_col, blue_orient, red_orient = _locate_axes(g, bg)
    rects = _extract_rectangles(g, bg, cr, cc, marker)

    if group_by_color:
        ordered = _order_rectangles(rects, blue_orient, red_orient, cr, cc)
    else:
        blue_edge_idx = int(cc if blue_orient == "v" else cr)
        red_edge_idx = int(cc if red_orient == "v" else cr)
        for rect in rects:
            box = rect["box"]
            rect["sort_key"] = (
                int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
                int(_dist_to_edge(box, red_orient, red_edge_idx)),
                int(_along_from_corner(box, blue_orient, cr, cc)),
                int(_along_from_corner(box, red_orient, cr, cc)),
            )
        ordered = sorted(rects, key=lambda rect: rect["sort_key"])

    seq = []
    for rect in ordered:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        if recolor:
            arr = np.full(arr.shape, int(blue_col), dtype=int)
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
    Fused rule:
    - detect the same rectangles as the source task;
    - use input rectangle color as a grouping cue;
    - sort inside each color group by the original geometric key;
    - order whole groups by the first member under that same key;
    - rotate each rectangle 90 degrees CCW if the blue axis is vertical;
    - recolor every stacked rectangle to the blue-axis color.
    """
    return _solve(grid, group_by_color=True, recolor=True)


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
        total = int(np.random.randint(4, 9))
        n0 = int(np.random.randint(2, min(5, total - 2) + 1))
        n1 = int(total - n0)
        if 2 <= n1 <= 5:
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
        sizes[int(np.random.randint(group_count))] += 1
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
    for _ in range(400):
        grid = _gen_attempt()
        if _count_object_colors(grid) < 2:
            continue

        fused = _solve(grid, group_by_color=True, recolor=True)
        recolor_only = _solve(grid, group_by_color=False, recolor=True)
        group_only = _solve(grid, group_by_color=True, recolor=False)

        if np.array_equal(fused, recolor_only):
            continue
        if np.array_equal(fused, group_only):
            continue

        assert set(map(int, np.unique(fused))) <= {int(BG), int(BLUE)}
        return grid
    raise RuntimeError("failed to generate combo input for 291dc1e1")
