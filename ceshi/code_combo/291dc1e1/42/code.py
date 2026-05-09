# Combo variant of 291dc1e1 (index 42)
# combo_spec: D5/1+D6/3+D7/2
# Name: 水平排列+方形输出+始终旋转

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for combo D5/1 + D6/3 + D7/2
=============================================

a) 具体化与行为追踪
- Input family stays the same as the source task: one marked corner, one full horizontal edge line,
  one full vertical edge line, and several separated filled rectangles in the interior.
- Source-task behavior:
  1. sort rectangles by (distance to blue edge, distance to red edge, along-blue, along-red),
  2. rotate each rectangle 90 degrees CCW only when the blue edge is vertical,
  3. stack rectangles top-to-bottom,
  4. crop tightly to the stack's bounding box.
- Combo behavior:
  1. keep the same rectangle extraction and ordering,
  2. rotate every rectangle 90 degrees CCW regardless of blue orientation,
  3. place the rotated rectangles left-to-right with no gaps,
  4. make the final canvas square, side = max(horizontal-strip height, horizontal-strip width).
- Concrete trace:
  suppose blue is horizontal and the sorted rectangles are A=2x6, B=2x4, C=2x2.
  Source task:
    no rotation, vertical stack -> tight size = (2+2+2) x max(6,4,2) = 6x6.
  This combo:
    always rotate -> A=6x2, B=4x2, C=2x2.
    horizontal strip -> tight size = max(6,4,2) x (2+2+2) = 6x6.
    square side = 6, so layout differs even though the canvas size matches.
  Another trace with four rectangles A=2x12, B=2x4, C=2x2, D=2x2 and blue horizontal:
    combo rotation -> 12x2, 4x2, 2x2, 2x2.
    horizontal strip tight size = 12x8.
    square output = 12x12, with the strip left-aligned and each rectangle vertically centered.

b) 复杂度校准
- The task still has one stable cue system: two guide axes determine ordering. The combo only alters
  transform, packing direction, and final crop shape.
- To make the fused behavior visible, the generator enforces:
  - at least four rectangles, so left-to-right composition is obvious,
  - many blue-horizontal samples, so unconditional rotation is actually testable,
  - non-square horizontal strips, so the square-padding rule is visible,
  - at least one non-2x2 rectangle in every kept sample, so rotation is visually observable.
- Width choices stay on a small even-valued set; only a rare 10 or 12 wide rectangle is introduced
  for height-driven square cases, because with the original 4-9 rectangle count and widths <= 8,
  blue-horizontal outputs cannot visibly satisfy `strip_height > strip_width` except the degenerate 8=8 case.

c) 隐式假设审查
- Corner detection assumes exactly one corner marker that differs from both adjacent edge-line colors.
  Preserved by generator.
- Blue/red line inference assumes color 1 identifies blue when present, else the longer adjacent edge
  is blue. Preserved.
- Rectangle extraction assumes interior objects are disconnected 4-connected filled rectangles with no
  background holes inside their bbox. Preserved.
- The source-task branch `if blue_orient == "v": rotate` is invalid under this combo; reusing it would
  silently erase D5/1. The solver therefore rotates unconditionally.
- The source-task crop `height=sum(h), width=max(w)` and row accumulation are invalid under D6/3.
  They must become horizontal-strip measurements `height=max(h), width=sum(w)` and column accumulation.
- Applying D7/2 after D6/3 means the square side must be computed from the horizontal strip, not from
  the legacy vertical stack. Reusing the original tight-size formula would implement the wrong square.

d) 放置合法性
- Canonical inputs use two interior bands separated from the guide axes and from each other.
- Rectangles within a band are placed with fixed 2-cell gaps, so 4-connectivity never merges them.
- The chosen long dimension is always at least `max(band_used_widths) + 4`, so every rectangle stays
  inside bounds with a one-cell margin from the red edge and 1-2 trailing background cells.
- Whole-grid flips/rotations preserve legality.

e) 视觉可观察性
- Across a few samples, a human sees the same ordering cue as the base task, but the output now has
  three consistent visual signatures:
  1. every object appears rotated relative to the blue-horizontal inputs,
  2. objects line up left-to-right rather than top-to-bottom,
  3. the canvas is square, with background padding only on the right or on top/bottom depending on
     which strip dimension is shorter.
- The generator rejects samples where these cues collapse into a legacy-equivalent result.

f) 可视化整洁度
- Inputs stay sparse: only guide lines, one marker, and separated rectangles.
- Outputs contain one clean horizontal composition on uniform background; no overlaps or extra markers.

g) 颜色变化可推断性
- No color remapping is introduced. Colors remain local rectangle texture only.
- Minority pixels are sparse and only serve to make rotation visually detectable.

h) 融合后 main() 的新隐式假设
- Horizontal square assembly assumes there is at least one extracted rectangle. Generator guarantees 4-8.
- Vertical centering on the square canvas assumes every rotated rectangle height is <= square side.
  This is guaranteed because the side is the max of strip height and strip width.
- Left-to-right packing assumes the sum of rotated widths equals the strip width and no object needs gaps.
  Generator uses rectangles only; no post-transform overlap can occur.

i) generator 对 solver 新假设的保障
- Every retained sample is validated against the combo solver itself.
- The generator also compares the combo output to three nearby non-combo alternatives:
  legacy original, conditional-rotation horizontal-square, and always-rotate horizontal-tight.
  Samples that match any relevant alternative are rejected when that would hide the intended rule.
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _locate_corner(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    H, W = g.shape
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
    cr, cc = corner
    return int(cr), int(cc), int(g[cr, cc])


def _line_orientations(g: np.ndarray, cr: int, cc: int) -> tuple[str, str]:
    H, W = g.shape
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
    return blue_orient, red_orient


def _ordered_sequence(grid: np.ndarray, *, always_rotate: bool) -> tuple[int, str, list[np.ndarray]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cr, cc, marker = _locate_corner(g, bg)
    blue_orient, red_orient = _line_orientations(g, cr, cc)

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": arr.copy()})

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

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

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
        if always_rotate or blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    return int(bg), str(blue_orient), seq


def _horizontal_strip_shape(seq: list[np.ndarray]) -> tuple[int, int]:
    height = int(max(int(arr.shape[0]) for arr in seq))
    width = int(sum(int(arr.shape[1]) for arr in seq))
    return height, width


def _vertical_stack_shape(seq: list[np.ndarray]) -> tuple[int, int]:
    height = int(sum(int(arr.shape[0]) for arr in seq))
    width = int(max(int(arr.shape[1]) for arr in seq))
    return height, width


def _assemble_horizontal(seq: list[np.ndarray], bg: int, *, square: bool) -> np.ndarray:
    tight_h, tight_w = _horizontal_strip_shape(seq)
    out_h = int(max(tight_h, tight_w) if square else tight_h)
    out_w = int(max(tight_h, tight_w) if square else tight_w)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_c = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_r = int((out_h - h) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_c += int(w)
    return out


def _assemble_vertical(seq: list[np.ndarray], bg: int, *, square: bool) -> np.ndarray:
    tight_h, tight_w = _vertical_stack_shape(seq)
    out_h = int(max(tight_h, tight_w) if square else tight_h)
    out_w = int(max(tight_h, tight_w) if square else tight_w)
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
    Reorder rectangles by the source-task edge-distance rule, rotate every rectangle
    90 degrees CCW, then place them left-to-right inside a square output canvas.
    """
    bg, _blue_orient, seq = _ordered_sequence(grid, always_rotate=True)
    return _assemble_horizontal(seq, bg, square=True)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    bg, _blue_orient, seq = _ordered_sequence(grid, always_rotate=False)
    return _assemble_vertical(seq, bg, square=False)


def _conditional_horizontal_square(grid: np.ndarray) -> np.ndarray:
    bg, _blue_orient, seq = _ordered_sequence(grid, always_rotate=False)
    return _assemble_horizontal(seq, bg, square=True)


def _always_rotate_horizontal_tight(grid: np.ndarray) -> np.ndarray:
    bg, _blue_orient, seq = _ordered_sequence(grid, always_rotate=True)
    return _assemble_horizontal(seq, bg, square=False)


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
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _band_used_width(widths: list[int]) -> int:
    return int(sum(int(w) for w in widths) + 2 * max(0, len(widths) - 1))


def _sample_band_pair(driver: str) -> tuple[int, int]:
    if driver == "height":
        choices = [(2, 2), (2, 3), (3, 2)]
    else:
        choices = [(3, 3), (3, 4), (4, 4)]
    idx = int(np.random.randint(len(choices)))
    return tuple(map(int, choices[idx]))


def _sample_widths(count: int, driver: str, *, force_wide: bool) -> list[int]:
    widths: list[int] = []
    for idx in range(count):
        if force_wide and idx == 0:
            widths.append(int(np.random.choice([10, 12])))
        elif driver == "height":
            widths.append(int(np.random.choice([2, 4, 6])))
        else:
            widths.append(int(np.random.choice([2, 4, 6, 8])))
    return widths


def _build_canonical(driver: str) -> np.ndarray:
    while True:
        n0, n1 = _sample_band_pair(driver)
        wide_band = int(np.random.randint(2))

        widths0 = _sample_widths(n0, driver, force_wide=(driver == "height" and wide_band == 0))
        widths1 = _sample_widths(n1, driver, force_wide=(driver == "height" and wide_band == 1))
        total = int(n0 + n1)

        used0 = _band_used_width(widths0)
        used1 = _band_used_width(widths1)
        long_dim = int(max(used0, used1) + 4 + np.random.randint(0, 3))
        if long_dim > 30:
            continue

        max_width = int(max(widths0 + widths1))
        if driver == "height" and max_width <= 2 * total:
            continue
        if driver == "width" and max_width >= 2 * total:
            continue

        grid = np.full((8, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        for r0, widths in ((2, widths0), (5, widths1)):
            c = 2
            for width in widths:
                color = int(np.random.choice(DOMINANT_COLORS))
                patch = np.full((2, int(width)), color, dtype=int)
                patch = _sprinkle_minorities(patch, color)
                paste(grid, patch, int(r0), int(c))
                c += int(width) + 2

        return grid


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    out = main(grid)
    bg, blue_orient, seq = _ordered_sequence(grid, always_rotate=True)
    tight_h, tight_w = _horizontal_strip_shape(seq)
    conditional = _conditional_horizontal_square(grid)
    legacy = _legacy_main(grid)
    tight = _always_rotate_horizontal_tight(grid)
    return {
        "blue_horizontal": int(blue_orient == "h"),
        "tight_h": int(tight_h),
        "tight_w": int(tight_w),
        "side": int(out.shape[0]),
        "extra_area": int(out.size - tight_h * tight_w),
        "d5_visible": int(not np.array_equal(out, conditional)),
        "d6_visible": int(not np.array_equal(out, legacy)),
        "d7_visible": int(not np.array_equal(out, tight)),
    }


def _gen_attempt() -> np.ndarray:
    driver = str(np.random.choice(["height", "width"], p=[0.35, 0.65]))
    canonical = _build_canonical(driver)
    blue_orient = "h" if np.random.rand() < 0.85 else "v"
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        metrics = _fusion_metrics(grid)
        if metrics["extra_area"] <= 0:
            continue
        if metrics["d6_visible"] == 0 or metrics["d7_visible"] == 0:
            continue
        if metrics["blue_horizontal"] == 1 and metrics["d5_visible"] == 0:
            continue
        if metrics["blue_horizontal"] == 0 and np.random.rand() < 0.7:
            continue
        _ = main(grid)
        return grid
    raise RuntimeError("failed to generate input for combo variant 291dc1e1/42")
