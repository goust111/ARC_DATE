# Variant of 291dc1e1
# Dimension: D12 — 角落与方向编码 (symbol_instructions)
# Change: 标记像素(色0)位于蓝线与红线交叉的网格角落，蓝线颜色固定为1，方向由网格宽高比隐含... → 标记像素颜色编码旋转方向

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

"""
Design analysis for this D12 variant
====================================

Source-task review:
- The variant description cites no separate source task, so the baseline is the
  original 291dc1e1 solver/generator only.

a) Concretization and behavior trace
------------------------------------
- Input still has two edge lines meeting at a corner, plus interior full
  rectangles. Blue remains color 1. Red remains color 2. Background remains 8.
- The only intended rule change is how output rotation is chosen:
  old rule: if the blue line is vertical, rotate every rectangle 90 deg CCW.
  new rule: read the single corner marker color instead:
    3 -> rotate every rectangle 90 deg CW
    4 -> rotate every rectangle 90 deg CCW
    5 -> do not rotate
- Sorting by distance to the blue and red edge-lines is unchanged.
- Concrete trace:
  Suppose the corner is top-left, blue is horizontal, and two 2x4 rectangles are
  found. In the original task the output would keep them unrotated, because blue
  is horizontal. In this variant, if the corner marker is 3, those same 2x4
  rectangles are first sorted exactly as before, but then each becomes 4x2 by CW
  rotation before stacking. The changed step is only the final per-rectangle
  orientation transform.

b) Complexity calibration
-------------------------
- The marker uses exactly three colors and three actions. That is enough to be a
  real symbolic code, but still small enough for ARC-style induction.
- The generator keeps the familiar edge-line geometry and rectangle layout, so
  the task does not become a different puzzle.
- Every rectangle receives at least one off-color accent pixel. This prevents the
  CW and CCW cases from collapsing visually on symmetric solid blocks.

c) Implicit assumption audit against the original main()
-------------------------------------------------------
- Corner detection originally assumed only that the marker is a unique corner
  pixel different from its adjacent line cells. That still holds.
- The old mask used `(g != marker)`, which was harmless only because the original
  marker color 0 never appeared inside rectangles. After D12, marker colors are
  ordinary palette colors, so excluding every cell of that color would delete
  legitimate rectangle pixels. The new solver removes only the actual corner cell
  by index, not every cell sharing the marker's color.
- The old rotation branch depended on `blue_orient == "v"`. That assumption is
  replaced by an explicit marker-color lookup.

d) Placement legality
---------------------
- The generator still builds a canonical non-overlapping layout: top edge line,
  left edge line, then two separated horizontal bands of rectangles placed with
  at least one background column between them.
- Spatial legality is preserved by reusing the original width-distribution logic,
  which guarantees the total used width fits the chosen canvas.
- The final `_transform()` only flips/rotates the whole legal grid, so objects
  remain in-bounds and non-overlapping in every corner/orientation case.

e) Visual observability
-----------------------
- A human can compare examples and notice that the same kind of corner symbol now
  predicts the output orientation. Because rectangles contain accent pixels, CW,
  CCW, and no-rotation are distinguishable even when rectangle sizes coincide.
- Blue/red edge lines still explain the ordering, while the corner pixel is the
  only new degree of freedom, so the new mechanism stays readable.

f) Visual cleanliness
---------------------
- Marker colors 3/4/5 are reserved for the single corner cell only.
- Rectangle dominant colors avoid 3/4/5, preventing the corner code from being
  drowned out by unrelated uses of those colors.
- Objects remain axis-aligned rectangles with clear spacing, just like the base
  task.

g) Color-change inferability
----------------------------
- The mapping uses exactly three reserved colors at the same structural location:
  the unique corner marker. This gives repeated positional evidence for the code.
- No other part of the input changes color based on that rule, so the inference
  burden is limited to "marker color -> global rotation mode".

h) Post-change solver assumptions and generator guarantees
---------------------------------------------------------
- Solver assumption: exactly one corner marker exists and its color is in
  `MARKER_TO_ROT_K`. Generator guarantee: canonical input writes exactly one such
  marker at the edge-line intersection before any whole-grid transform.
- Solver assumption: interior objects are full rectangles with no background
  holes. Generator guarantee: each patch is a filled 2xW rectangle whose accents
  use non-background colors.
- Solver assumption: at least one rectangle exists. Generator guarantee: both
  interior bands contain 2..5 rectangles.
"""

BG = 8
BLUE = 1
RED = 2

MARKER_TO_ROT_K = {
    3: -1,  # 90 deg clockwise
    4: 1,   # 90 deg counterclockwise
    5: 0,   # no rotation
}
MARKER_COLORS = np.array(sorted(MARKER_TO_ROT_K), dtype=int)

DOMINANT_COLORS = np.array([6, 7, 9], dtype=int)
ACCENT_COLORS = np.array([0, 1, 2], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two colored edge lines.

    Variant rule:
    - The corner marker color explicitly encodes how every rectangle is rotated.
    - Sorting by distance to the blue and red edge-lines is unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        val = int(g[r, c])
        if val == int(bg) or val not in MARKER_TO_ROT_K:
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
        if nbr and all(int(n) != val for n in nbr):
            corner = (int(r), int(c))
            break
    assert corner is not None

    cr, cc = map(int, corner)
    rot_k = int(MARKER_TO_ROT_K[int(g[cr, cc])])

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

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask)
    mask[cr, cc] = False
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
        arr = rct["arr"]
        if rot_k:
            arr = np.rot90(arr, k=rot_k)
        seq.append(arr)

    outW = int(max(int(a.shape[1]) for a in seq))
    outH = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((outH, outW), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((outW - w) // 2)
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


def _make_patch(width: int, dominant: int) -> np.ndarray:
    patch = np.full((2, int(width)), int(dominant), dtype=int)
    n_accents = 1 + int(width >= 4 and np.random.rand() < 0.65)
    if width >= 6 and np.random.rand() < 0.25:
        n_accents += 1

    cells = np.random.choice(2 * int(width), size=n_accents, replace=False)
    flat = patch.reshape(-1)
    for i, cell in enumerate(cells):
        flat[int(cell)] = int(ACCENT_COLORS[i % len(ACCENT_COLORS)])
    return patch


def _build_canonical(marker_color: int) -> np.ndarray:
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
    grid[0, 0] = int(marker_color)

    for r0, widths in ((2, band0), (5, band1)):
        c = 2
        for w in widths:
            color = int(np.random.choice(DOMINANT_COLORS))
            paste(grid, _make_patch(int(w), color), r0, c)
            c += int(w) + 2

    return grid


def _gen_attempt() -> np.ndarray:
    marker_color = int(np.random.choice(MARKER_COLORS))
    canonical = _build_canonical(marker_color)
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        try:
            _ = main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D12 variant")
