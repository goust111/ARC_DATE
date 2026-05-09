# Variant of 291dc1e1
# Dimension: D1 — 排序主键——距离度量方式 (match_cond)
# Change: 按 (到蓝线的距离, 到红线的距离) 升序排序，距离为矩形边缘到边线的像素间隔... → 按面积大小排序

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the D1 variant
# a) Concrete behavior and trace:
#    The original task uses the corner marker plus the blue/red edge lines as a
#    coordinate frame, then orders the interior solid rectangles by
#    (distance-to-blue, distance-to-red). In this variant, the frame is still
#    needed to detect whether the blue line is horizontal or vertical, because
#    that still controls the global 90 degree CCW rotation step. The changed
#    step is only the ordering: rectangles are stacked by area from small to
#    large. A concrete canonical example is:
#      top band:    2x5, 2x2
#      bottom band: 3x4, 3x2
#    Original order in the top-left/horizontal-blue frame would be
#      2x5 -> 2x2 -> 3x4 -> 3x2
#    because the top band is closer to the blue line.
#    Variant order is
#      2x2(area 4) -> 3x2(area 6) -> 3x4(area 12) -> 2x5(area 10) sorted as
#      2x2 -> 3x2 -> 2x5 -> 3x4
#    so the changed behavior is directly visible in the stack.
# b) Complexity calibration:
#    If all rectangles had the same height, area would collapse to width and the
#    visual change would be weaker. The generator therefore keeps two horizontal
#    bands but gives them different fixed heights (2 and 3), so the solver must
#    reason about cell count rather than one-axis proximity. Areas are kept
#    unique per sample to avoid ambiguous ties.
# c) Implicit-assumption audit on the original solver:
#    - Corner detection still assumes exactly one corner marker distinct from
#      both adjacent line colors. Generator preserves that.
#    - Blue-line detection still prefers color 1, else longer line. Generator
#      still uses color 1 for the blue line.
#    - Rectangle extraction assumes each object fills its bbox with no
#      background holes. Generator keeps every object as a full rectangle; only
#      non-background accent cells are added inside.
#    - The old `dist_to_edge` values were the main ordering signal. That
#      assumption is removed. Only orientation-dependent rotation and a stable
#      spatial tiebreak remain.
# d) Placement legality:
#    Rectangles are placed in two separated bands with at least 2 columns of gap
#    inside a larger background canvas, so components never touch, overlap, or
#    cross the masked edge row/column. The transformed variants are produced by
#    whole-grid flips/rotations, which preserve non-overlap.
# e) Visual observability:
#    The generator rejects samples whose area order matches the original
#    distance-based order on the same transformed input. That forces the D1
#    change to matter in the observed examples.
# f) Visual cleanliness:
#    The scene stays close to the source task: one corner marker, two edge
#    lines, and two neat interior rows of rectangles. The only richer detail is
#    a few accent cells inside each rectangle so orientation changes remain
#    visible after rotation.
# g) Color inferability:
#    Colors are not remapped by the rule. Rectangle dominant colors are kept
#    distinct within a sample to make tracking objects easy, and accent colors
#    are sparse so they do not introduce a second hidden rule.
# h) Generator validation:
#    For each attempted sample, the generator constructs the intended area-sorted
#    output from metadata, compares it with `main(input)`, and retries on any
#    mismatch or degeneration.
# i) New solver assumptions after the edit:
#    The solver now assumes only that every target object is a solid rectangle
#    whose area is measurable from its cropped bbox. The generator guarantees
#    that, and also guarantees unique areas for visual clarity, though the
#    solver itself remains deterministic via spatial tiebreaks if equal areas
#    ever occur.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
ACCENT_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

TOP_ROW = 2
BOTTOM_ROW = 6
TOP_H = 2
BOTTOM_H = 3


def _locate_frame(g: np.ndarray) -> tuple[int, tuple[int, int], int, str, str]:
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
        blue_col = int(vcol if vlen >= hlen else hcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "v" if blue_orient == "h" else "h"
    return int(bg), (int(cr), int(cc)), int(marker), blue_orient, red_orient


def _extract_rectangles(g: np.ndarray) -> tuple[int, tuple[int, int], str, str, list[dict]]:
    bg, corner, marker, blue_orient, red_orient = _locate_frame(g)
    cr, cc = corner
    H, W = g.shape

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append(
            {
                "box": tuple(map(int, box)),
                "arr": sub.copy(),
                "area": int(np.count_nonzero(sub != int(bg))),
            }
        )

    assert rects
    return int(bg), (int(cr), int(cc)), blue_orient, red_orient, rects


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _stack_sequence(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(a.shape[1]) for a in seq))
    out_h = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for a in seq:
        h, w = map(int, a.shape)
        out_c = int((out_w - w) // 2)
        paste(out, a, int(out_r), int(out_c))
        out_r += int(h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by area.

    Input structure:
    - bg = most common color.
    - Two edge lines meet at a marked corner.
    - The blue line is color 1 if present, else the longer line.
    - Solid interior rectangles occupy full bounding boxes.

    Rule:
    - Sort rectangles by area, from small to large.
    - If the blue edge is vertical, rotate each rectangle 90 degrees CCW.
    - Stack top-to-bottom, centered horizontally.
    """
    g = np.array(grid, dtype=int)
    bg, (cr, cc), blue_orient, red_orient, rects = _extract_rectangles(g)

    rects.sort(
        key=lambda r: (
            int(r["area"]),
            _along_from_corner(r["box"], blue_orient, cr, cc),
            _along_from_corner(r["box"], red_orient, cr, cc),
            tuple(map(int, r["box"])),
        )
    )

    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    return _stack_sequence(seq, bg)


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


def _decorate_patch(h: int, w: int, dominant: int) -> np.ndarray:
    patch = np.full((h, w), int(dominant), dtype=int)
    candidates = [(0, 0), (h - 1, min(1, w - 1)), (min(1, h - 1), w - 1)]
    n_accents = 1 + int((h * w >= 8) and (np.random.rand() < 0.75))
    n_accents += int((h * w >= 12) and (np.random.rand() < 0.4))
    n_accents = min(n_accents, len(candidates))
    colors = [int(c) for c in ACCENT_COLORS if int(c) not in {int(dominant), int(BG)}]
    np.random.shuffle(colors)
    for idx, (r, c) in enumerate(candidates[:n_accents]):
        patch[int(r), int(c)] = int(colors[idx])
    return patch


def _rect_specs_for_band(height: int, count: int) -> list[tuple[int, int]]:
    widths = np.arange(2, 7, dtype=int)
    order = np.random.permutation(len(widths))
    return [(int(height), int(widths[idx])) for idx in order[:count]]


def _place_band(grid: np.ndarray, row: int, specs: list[tuple[int, int]], colors: np.ndarray) -> list[dict]:
    rects = []
    c = 2
    for idx, (h, w) in enumerate(specs):
        patch = _decorate_patch(h, w, int(colors[idx]))
        paste(grid, patch, int(row), int(c))
        rects.append(
            {
                "patch": patch,
                "area": int(h * w),
                "box": (int(row), int(c), int(row + h - 1), int(c + w - 1)),
            }
        )
        c += int(w + 2)
    return rects


def _render_single_rect(
    patch: np.ndarray,
    box: tuple[int, int, int, int],
    canonical_shape: tuple[int, int],
    blue_orient: str,
    corner: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.full(canonical_shape, BG, dtype=int)
    r0, c0, _r1, _c1 = map(int, box)
    paste(canvas, patch, int(r0), int(c0))
    transformed = _transform(canvas, blue_orient, corner)
    coords = np.argwhere(transformed != BG)
    bbox = bbox_coords(coords)
    return crop_bbox(transformed, bbox), tuple(map(int, bbox))


def _corner_coords(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = map(int, shape)
    mapping = {
        "tl": (0, 0),
        "tr": (0, int(w - 1)),
        "bl": (int(h - 1), 0),
        "br": (int(h - 1), int(w - 1)),
    }
    return mapping[corner]


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
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


def _old_distance_order_ids(rects: list[dict], input_shape: tuple[int, int], blue_orient: str, corner: str) -> list[int]:
    cr, cc = _corner_coords(input_shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    ordered = sorted(
        rects,
        key=lambda r: (
            _dist_to_edge(r["input_box"], blue_orient, blue_edge_idx),
            _dist_to_edge(r["input_box"], red_orient, red_edge_idx),
            _along_from_corner(r["input_box"], blue_orient, cr, cc),
            _along_from_corner(r["input_box"], red_orient, cr, cc),
        ),
    )
    return [int(r["id"]) for r in ordered]


def _area_order_ids(rects: list[dict], input_shape: tuple[int, int], blue_orient: str, corner: str) -> list[int]:
    cr, cc = _corner_coords(input_shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    ordered = sorted(
        rects,
        key=lambda r: (
            int(r["area"]),
            _along_from_corner(r["input_box"], blue_orient, cr, cc),
            _along_from_corner(r["input_box"], red_orient, cr, cc),
        ),
    )
    return [int(r["id"]) for r in ordered]


def _build_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    count_options = [(2, 2), (2, 3), (3, 2)]
    n_top, n_bottom = count_options[int(np.random.randint(len(count_options)))]

    top_specs = _rect_specs_for_band(TOP_H, n_top)
    bottom_specs = _rect_specs_for_band(BOTTOM_H, n_bottom)
    all_areas = [h * w for h, w in top_specs + bottom_specs]
    if len(set(all_areas)) != len(all_areas):
        return None

    need_w_top = 2 + sum(w for _h, w in top_specs) + 2 * (len(top_specs) - 1) + int(np.random.randint(1, 4))
    need_w_bottom = 2 + sum(w for _h, w in bottom_specs) + 2 * (len(bottom_specs) - 1) + int(np.random.randint(1, 4))
    width = int(max(need_w_top, need_w_bottom))
    height = 11

    canonical = np.full((height, width), BG, dtype=int)
    canonical[0, :] = BLUE
    canonical[:, 0] = RED
    canonical[0, 0] = MARKER

    colors = np.random.permutation(DOMINANT_COLORS)
    rects = []
    rects.extend(_place_band(canonical, TOP_ROW, top_specs, colors[:n_top]))
    rects.extend(_place_band(canonical, BOTTOM_ROW, bottom_specs, colors[n_top : n_top + n_bottom]))

    # Avoid all-near-square sets; the rotation branch should stay visible.
    if sum(int(r["patch"].shape[0] != r["patch"].shape[1]) for r in rects) < 2:
        return None

    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    inp = _transform(canonical, blue_orient, corner)

    transformed_rects = []
    for idx, rect in enumerate(rects):
        input_patch, input_box = _render_single_rect(rect["patch"], rect["box"], canonical.shape, blue_orient, corner)
        transformed_rects.append(
            {
                "id": int(idx),
                "area": int(rect["area"]),
                "input_patch": input_patch,
                "input_box": input_box,
            }
        )

    old_ids = _old_distance_order_ids(transformed_rects, inp.shape, blue_orient, corner)
    area_ids = _area_order_ids(transformed_rects, inp.shape, blue_orient, corner)
    if old_ids == area_ids:
        return None

    rect_by_id = {int(r["id"]): r for r in transformed_rects}
    seq = []
    for rid in area_ids:
        arr = rect_by_id[int(rid)]["input_patch"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    expected = _stack_sequence(seq, BG)

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    meta = {
        "blue_orient": blue_orient,
        "corner": corner,
        "rect_count": len(rects),
        "areas": sorted(int(r["area"]) for r in rects),
        "old_ids": old_ids,
        "area_ids": area_ids,
    }
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(500):
        built = _build_attempt()
        if built is not None:
            inp, _expected, _meta = built
            return inp
    raise RuntimeError("failed to generate input for 291dc1e1 D1 area-order variant")
