# Variant of 291dc1e1
# Dimension: D10 — 堆叠顺序与多步处理 (multi_step)
# Change: 一次性排序所有矩形后按序堆叠，无迭代... → 交替band堆叠
#
# Design analysis for this variant
# a) Concrete behavior / trace:
#    The original task already forms two visible horizontal bands in canonical form:
#    one band close to the blue edge and one farther away. Originally all rectangles
#    were globally sorted and then stacked from the top. Here the input stays almost
#    the same, but the output stage is split into two passes:
#    - near-blue band: stack from the top downward;
#    - far-blue band: stack from the bottom upward.
#    Example in canonical orientation:
#      near band order by red distance = [A, B, C]
#      far band order by red distance = [D, E]
#      original output order top->bottom = [A, B, C, D, E]
#      variant output layout top->bottom = [A, B, C, E, D]
#    The changed logic therefore lives entirely in the output placement stage; edge
#    detection, rectangle extraction, and per-band ordering stay compatible with the
#    original mechanism.
#
# b) Complexity calibration:
#    If a band had only one object, bottom-up vs top-down would be invisible there.
#    The generator therefore guarantees at least two rectangles in each band and uses
#    distinct widths inside a band so reversing the far band changes the visible stack.
#
# c) Old-code dependency audit:
#    - Corner detection assumes a unique marker at a grid corner: still true.
#    - Blue/red line inference assumes one horizontal + one vertical edge line: still true.
#    - Rectangle extraction assumes compact rectangular components away from the lines:
#      still true.
#    - The original `rects.sort(...)` was harmless because output consumed the whole list
#      in one direction. In the variant we cannot directly paste that single list top-down;
#      we must preserve the sorted order inside each band, then place the second band from
#      the bottom. This is the main semantic change.
#    - The original output loop assumed one monotonically increasing cursor `out_r`. That
#      no longer holds; we need both a top cursor and a bottom cursor.
#
# d) Placement legality:
#    The generator keeps two 2-row bands separated by background rows, with 2-cell gaps
#    between rectangles. Output uses total height = sum of object heights, so top-stack
#    and bottom-stack meet exactly without overlap or overflow.
#
# e) Visual observability:
#    The variant must be inferable from 2-4 examples, so the generator avoids degenerate
#    cases where reversing the far band would look identical to the original behavior.
#    Distinct widths in each band make the directional difference visible even if colors
#    repeat.
#
# f) Visual cleanliness:
#    The layout remains sparse and axis-aligned: one corner marker, two straight edge
#    lines, two separated interior bands, centered output stack. No extra markers or
#    overlaid signals are introduced.
#
# g) Color inferability:
#    Colors still identify rectangles only; no new color mapping is added. A small palette
#    and optional minor interior pixels give object identity without forcing the solver to
#    infer a separate color rule.
#
# h) New solver assumptions and generator guarantees:
#    The variant solver assumes exactly two distinct blue-distance bands. The generator
#    enforces that by constructing exactly two rows of rectangles in canonical space.
#    It also guarantees at least two rectangles per band and validates that the produced
#    sample does not collapse to the original one-direction output.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


def _locate_corner(g: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = map(int, g.shape)
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == h - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == w - 1:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != int(g[r, c]) for n in nbr):
            return int(r), int(c)
    raise AssertionError("corner marker not found")


def _infer_edge_orientations(g: np.ndarray, corner: tuple[int, int]) -> tuple[str, str, int]:
    h, w = map(int, g.shape)
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
        while 0 <= c < w and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < h and int(g[r, cc]) == int(vcol):
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
    return blue_orient, red_orient, marker


def _extract_rectangles(
    g: np.ndarray, bg: int, corner: tuple[int, int], marker: int
) -> list[dict[str, object]]:
    h, w = map(int, g.shape)
    cr, cc = map(int, corner)
    line_mask = np.zeros((h, w), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict[str, object]] = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": sub.copy()})
    if not rects:
        raise AssertionError("no rectangles found")
    return rects


def _dist_to_edge(rect: dict[str, object], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        col = int(edge_idx)
        if c0 > col:
            return int(c0 - col - 1)
        if c1 < col:
            return int(col - c1 - 1)
        return 0
    row = int(edge_idx)
    if r0 > row:
        return int(r0 - row - 1)
    if r1 < row:
        return int(row - r1 - 1)
    return 0


def _along_from_corner(rect: dict[str, object], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _rotate_if_needed(arr: np.ndarray, blue_orient: str) -> np.ndarray:
    return np.rot90(arr, k=1) if blue_orient == "v" else arr


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = map(int, g.shape)
    bg = int(get_bg(g))

    corner = _locate_corner(g, bg)
    cr, cc = map(int, corner)
    blue_orient, red_orient, marker = _infer_edge_orientations(g, corner)
    rects = _extract_rectangles(g, bg, corner, marker)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            _along_from_corner(rect, blue_orient, cr, cc),
            _along_from_corner(rect, red_orient, cr, cc),
        )
    )

    blue_bands = sorted({int(rect["db"]) for rect in rects})
    if len(blue_bands) != 2:
        raise AssertionError("variant expects exactly two blue-distance bands")
    near_db, far_db = map(int, blue_bands)

    near_seq = [_rotate_if_needed(np.array(rect["arr"], dtype=int), blue_orient) for rect in rects if int(rect["db"]) == near_db]
    far_seq = [_rotate_if_needed(np.array(rect["arr"], dtype=int), blue_orient) for rect in rects if int(rect["db"]) == far_db]

    out_w = int(max(int(arr.shape[1]) for arr in near_seq + far_seq))
    out_h = int(sum(int(arr.shape[0]) for arr in near_seq + far_seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    top_r = 0
    for arr in near_seq:
        ah, aw = map(int, arr.shape)
        paste(out, arr, int(top_r), int((out_w - aw) // 2))
        top_r += int(ah)

    bottom_r = int(out_h)
    for arr in far_seq:
        ah, aw = map(int, arr.shape)
        bottom_r -= int(ah)
        paste(out, arr, int(bottom_r), int((out_w - aw) // 2))

    return out


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_CHOICES = np.array([2, 4, 6, 8], dtype=int)


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
    if area < 8 or np.random.rand() < 0.55:
        return patch
    max_minor = min(3, area // 4)
    if max_minor < 1:
        return patch
    n_minor = int(np.random.randint(1, max_minor + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    cells = np.random.choice(area, size=n_minor, replace=False)
    colors = np.random.choice(minority_choices, size=n_minor, replace=True)
    flat = patch.reshape(-1)
    for cell, color in zip(cells, colors):
        flat[int(cell)] = int(color)
    return patch


def _make_band(count: int) -> list[np.ndarray]:
    widths = list(map(int, np.random.choice(WIDTH_CHOICES, size=count, replace=False)))
    colors = list(
        map(
            int,
            np.random.choice(
                DOMINANT_COLORS,
                size=count,
                replace=bool(count > len(DOMINANT_COLORS)),
            ),
        )
    )
    patches: list[np.ndarray] = []
    for width, color in zip(widths, colors):
        patch = np.full((2, int(width)), int(color), dtype=int)
        patches.append(_sprinkle_minorities(patch, int(color)))
    return patches


def _used_width(patches: list[np.ndarray]) -> int:
    return int(sum(int(p.shape[1]) for p in patches) + 2 * (len(patches) - 1))


def _stack_output(
    near_band: list[np.ndarray], far_band: list[np.ndarray], blue_orient: str, bg: int
) -> np.ndarray:
    def rotate_band(band: list[np.ndarray]) -> list[np.ndarray]:
        return [_rotate_if_needed(arr.copy(), blue_orient) for arr in band]

    near_seq = rotate_band(near_band)
    far_seq = rotate_band(far_band)
    out_w = int(max(int(arr.shape[1]) for arr in near_seq + far_seq))
    out_h = int(sum(int(arr.shape[0]) for arr in near_seq + far_seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    top_r = 0
    for arr in near_seq:
        ah, aw = map(int, arr.shape)
        paste(out, arr, int(top_r), int((out_w - aw) // 2))
        top_r += int(ah)

    bottom_r = int(out_h)
    for arr in far_seq:
        ah, aw = map(int, arr.shape)
        bottom_r -= int(ah)
        paste(out, arr, int(bottom_r), int((out_w - aw) // 2))

    return out


def _stack_output_original(
    near_band: list[np.ndarray], far_band: list[np.ndarray], blue_orient: str, bg: int
) -> np.ndarray:
    seq = [_rotate_if_needed(arr.copy(), blue_orient) for arr in near_band + far_band]
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for arr in seq:
        ah, aw = map(int, arr.shape)
        paste(out, arr, int(out_r), int((out_w - aw) // 2))
        out_r += int(ah)
    return out


def _build_canonical() -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    near_count = int(np.random.randint(2, 5))
    far_count = int(np.random.randint(2, 5))
    near_band = _make_band(near_count)
    far_band = _make_band(far_count)

    used_near = _used_width(near_band)
    used_far = _used_width(far_band)
    long_dim = int(max(used_near, used_far) + 3 + np.random.randint(1, 5))

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, band in ((2, near_band), (5, far_band)):
        c = 2
        for patch in band:
            paste(grid, patch, int(r0), int(c))
            c += int(patch.shape[1]) + 2

    return grid, near_band, far_band


def _gen_attempt() -> np.ndarray | None:
    canonical, near_band, far_band = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    expected = _stack_output(near_band, far_band, blue_orient, BG)
    original = _stack_output_original(near_band, far_band, blue_orient, BG)
    if np.array_equal(expected, original):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D10 variant")
