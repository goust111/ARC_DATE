# Combo variant of 291dc1e1 (index 39)
# combo_spec: D1/0+D6/3+D10/3
# Name: 面积排序+水平排列+band内水平拼接

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# a) Concreteized behavior and trace:
#    The source task uses the corner marker and the blue/red guide lines to build
#    a coordinate frame, then sorts all rectangles globally by
#    (distance-to-blue, distance-to-red), rotates every rectangle 90 degrees CCW
#    when the blue line is vertical, and stacks the full sequence vertically.
#    This combo changes two linked stages. First, rectangles are still grouped by
#    distance to the blue line, so each blue-distance band remains a meaningful
#    row-group. Second, the ordering inside each band is no longer by red-line
#    distance but by rectangle area. Example in the canonical top-blue layout:
#      band db=1:  A=2x5(area10), B=3x3(area9), C=2x2(area4) placed left-to-right
#      band db=6:  D=3x4(area12), E=2x3(area6)
#    Source task output order:
#      A, B, C, D, E   because each band is traversed by red-distance.
#    Combo output:
#      row1 = C | B | A   sorted by area inside db=1
#      row2 = E | D       sorted by area inside db=6
#      final output = row1 stacked above row2.
#    If blue is vertical, each cropped rectangle is rotated first, and the same
#    grouping/row-concatenation logic is applied to the rotated pieces.
# b) Complexity calibration:
#    To make "area" a real cue rather than just width ranking, the generator uses
#    both 2-cell-tall and 3-cell-tall rectangles and forces at least one band to
#    mix those heights. At the same time it keeps only 2-3 bands and 2-4 objects
#    per band, so the scene remains readable and the grouping cue has repeated
#    support across examples.
# c) Implicit-assumption audit:
#    - Corner detection still assumes one marker in a grid corner, distinct from
#      its adjacent guide-line colors. Generator preserves that.
#    - Blue-line inference still prefers color 1, else the longer line. Generator
#      always uses color 1 for blue and color 2 for red, so the fallback is only
#      defensive.
#    - Component extraction still assumes each target object fills its bounding
#      box with no background holes. Generator uses solid rectangles with a few
#      non-background accent pixels only.
#    - The original global `rects.sort(...)` assumption is invalid here. It is
#      replaced by a two-stage ordering: band by `db`, then inside each band by
#      `area`.
#    - The original one-dimensional vertical stack assumption is also invalid.
#      The new output builder must assemble horizontal rows first and then stack
#      those rows vertically.
# d) Placement legality:
#    Canonical inputs place every band at one shared top row, leave a 2-cell gap
#    between rectangles inside a band, and leave a 2-row gap between bands. Grid
#    width is derived from the widest band, and grid height from the cumulative
#    band heights plus trailing border, so no rectangle overlaps another or exits
#    the canvas. Whole-grid flips/rotations preserve that legality.
# e) Visual observability:
#    A solver looking at 2-4 examples can see that objects near the same blue
#    distance stay together in one output row, while left-to-right order inside a
#    row follows small-to-large area rather than input position. The output is a
#    compact grid, not a one-column stack, so both D10 and D6 effects are visible.
# f) Visual cleanliness:
#    The input keeps one marker, two full edge lines, large background gaps, and
#    only rectangular objects. Varying heights within a band are limited to 2 or
#    3 so the picture stays structured instead of noisy.
# g) Color inferability:
#    No color remapping is introduced. Dominant colors help the viewer track each
#    rectangle; sparse accent cells only make orientation changes visible.
# h) Generator validation:
#    Each attempt builds a canonical scene, transforms it to a random corner and
#    blue orientation, independently constructs the intended combo output from the
#    transformed rectangle metadata, and rejects any attempt where `main(input)`
#    disagrees or where the combo output collapses to the source-task or D10-only
#    output on the same input.
# i) New solver assumptions:
#    The new solver assumes blue-distance bands are distinguishable and that area
#    is measurable per rectangle. The generator guarantees at least two bands,
#    at least one multi-rectangle band, unique areas within every band, and at
#    least one band whose area order differs from its red-distance order.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
ACCENT_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
RECT_SPECS = [
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 3),
    (3, 4),
    (3, 5),
]


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
    return int(bg), (int(cr), int(cc)), int(marker), blue_orient, red_orient


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


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _extract_rectangles(g: np.ndarray) -> tuple[int, tuple[int, int], str, str, list[dict]]:
    bg, (cr, cc), marker, blue_orient, red_orient = _locate_frame(g)
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
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        area = int(np.count_nonzero(arr != int(bg)))
        rects.append({"box": tuple(map(int, box)), "arr": arr.copy(), "area": area})

    assert rects

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect["box"], blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect["box"], red_orient, red_edge_idx))
        rect["ab"] = int(_along_from_corner(rect["box"], blue_orient, cr, cc))
        rect["ar"] = int(_along_from_corner(rect["box"], red_orient, cr, cc))

    return int(bg), (int(cr), int(cc)), blue_orient, red_orient, rects


def _rotate_if_needed(arr: np.ndarray, blue_orient: str) -> np.ndarray:
    return np.rot90(arr, k=1) if blue_orient == "v" else arr


def _assemble_horizontal(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = int(max(int(arr.shape[0]) for arr in seq))
    out_w = int(sum(int(arr.shape[1]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_c = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_r = int((out_h - h) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_c += int(w)
    return out


def _assemble_vertical(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for arr in seq:
        h, _w = map(int, arr.shape)
        paste(out, arr, int(out_r), 0)
        out_r += int(h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group rectangles by distance to the blue guide line, sort by area inside each
    group, concatenate each group horizontally, then stack the rows vertically.
    """
    g = np.array(grid, dtype=int)
    bg, (_cr, _cc), blue_orient, _red_orient, rects = _extract_rectangles(g)

    bands: dict[int, list[dict]] = {}
    for rect in rects:
        bands.setdefault(int(rect["db"]), []).append(rect)

    rows = []
    for db in sorted(bands):
        ordered = sorted(
            bands[int(db)],
            key=lambda rect: (
                int(rect["area"]),
                int(rect["dr"]),
                int(rect["ar"]),
                int(rect["ab"]),
                tuple(map(int, rect["box"])),
            ),
        )
        seq = [_rotate_if_needed(rect["arr"], blue_orient) for rect in ordered]
        rows.append(_assemble_horizontal(seq, bg))

    return _assemble_vertical(rows, bg)


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
    candidates = [(0, 0), (h - 1, w - 1), (0, w - 1)]
    n_accents = 1 + int((h * w >= 8) and (np.random.rand() < 0.8))
    n_accents += int((h * w >= 12) and (np.random.rand() < 0.35))
    n_accents = min(n_accents, len(candidates))
    palette = [int(c) for c in ACCENT_COLORS if int(c) not in {int(dominant), int(BG), int(MARKER)}]
    np.random.shuffle(palette)
    for idx, (r, c) in enumerate(candidates[:n_accents]):
        patch[int(r), int(c)] = int(palette[idx])
    return patch


def _build_band_specs(rect_count: int, force_mixed_heights: bool) -> list[tuple[int, int]]:
    for _ in range(200):
        chosen = [RECT_SPECS[idx] for idx in np.random.choice(len(RECT_SPECS), size=rect_count, replace=False)]
        areas = [int(h * w) for h, w in chosen]
        if len(set(areas)) != rect_count:
            continue
        if force_mixed_heights and len({int(h) for h, _w in chosen}) < 2:
            continue
        return list(chosen)
    raise RuntimeError("failed to build a valid band")


def _place_band(grid: np.ndarray, r0: int, specs: list[tuple[int, int]], colors: np.ndarray) -> tuple[list[dict], int]:
    rects = []
    c0 = 2
    band_h = max(int(h) for h, _w in specs)
    for idx, (h, w) in enumerate(specs):
        patch = _decorate_patch(int(h), int(w), int(colors[idx % len(colors)]))
        paste(grid, patch, int(r0), int(c0))
        rects.append(
            {
                "patch": patch,
                "box": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
                "area": int(h * w),
                "height": int(h),
                "width": int(w),
            }
        )
        c0 += int(w) + 2
    used_w = int(c0 - 4)
    return rects, int(max(used_w, 0)), int(band_h)


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


def _enrich_transformed_rects(
    rects: list[dict],
    input_shape: tuple[int, int],
    blue_orient: str,
    corner: str,
) -> list[dict]:
    cr, cc = _corner_coords(input_shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    enriched = []
    for idx, rect in enumerate(rects):
        box = tuple(map(int, rect["input_box"]))
        enriched.append(
            {
                "id": int(idx),
                "input_patch": rect["input_patch"],
                "input_box": box,
                "area": int(rect["area"]),
                "height": int(rect["height"]),
                "width": int(rect["width"]),
                "db": int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
                "dr": int(_dist_to_edge(box, red_orient, red_edge_idx)),
                "ab": int(_along_from_corner(box, blue_orient, cr, cc)),
                "ar": int(_along_from_corner(box, red_orient, cr, cc)),
            }
        )
    return enriched


def _output_original(rects: list[dict], bg: int, blue_orient: str) -> np.ndarray:
    ordered = sorted(
        rects,
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            int(rect["ab"]),
            int(rect["ar"]),
            int(rect["id"]),
        ),
    )
    seq = [_rotate_if_needed(rect["input_patch"], blue_orient) for rect in ordered]
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _output_d10(rects: list[dict], bg: int, blue_orient: str) -> np.ndarray:
    bands: dict[int, list[dict]] = {}
    for rect in rects:
        bands.setdefault(int(rect["db"]), []).append(rect)

    rows = []
    for db in sorted(bands):
        ordered = sorted(
            bands[int(db)],
            key=lambda rect: (
                int(rect["dr"]),
                int(rect["ar"]),
                int(rect["ab"]),
                int(rect["id"]),
            ),
        )
        seq = [_rotate_if_needed(rect["input_patch"], blue_orient) for rect in ordered]
        rows.append(_assemble_horizontal(seq, bg))
    return _assemble_vertical(rows, bg)


def _output_combo(rects: list[dict], bg: int, blue_orient: str) -> np.ndarray:
    bands: dict[int, list[dict]] = {}
    for rect in rects:
        bands.setdefault(int(rect["db"]), []).append(rect)

    rows = []
    for db in sorted(bands):
        ordered = sorted(
            bands[int(db)],
            key=lambda rect: (
                int(rect["area"]),
                int(rect["dr"]),
                int(rect["ar"]),
                int(rect["ab"]),
                int(rect["id"]),
            ),
        )
        seq = [_rotate_if_needed(rect["input_patch"], blue_orient) for rect in ordered]
        rows.append(_assemble_horizontal(seq, bg))
    return _assemble_vertical(rows, bg)


def _build_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    band_count = int(np.random.choice([2, 3]))
    for _ in range(200):
        rect_counts = [int(np.random.randint(2, 5)) for _band in range(band_count)]
        total_rects = int(sum(rect_counts))
        if not (4 <= total_rects <= 9):
            continue

        mixed_band_idx = int(np.random.randint(band_count))
        band_specs = []
        band_infos = []
        used_widths = []
        cursor_r = 2
        max_used_w = 0

        for band_idx, rect_count in enumerate(rect_counts):
            specs = _build_band_specs(rect_count, force_mixed_heights=(band_idx == mixed_band_idx))
            areas = [int(h * w) for h, w in specs]
            area_order = np.argsort(areas).tolist()
            if area_order == list(range(rect_count)):
                swap_idx = int(np.random.randint(1, rect_count))
                specs[0], specs[swap_idx] = specs[swap_idx], specs[0]
                areas = [int(h * w) for h, w in specs]

            band_h = max(int(h) for h, _w in specs)
            band_used_w = int(sum(int(w) for _h, w in specs) + 2 * (rect_count - 1))
            band_specs.append(specs)
            band_infos.append({"top": int(cursor_r), "band_h": int(band_h), "areas": areas})
            used_widths.append(int(band_used_w))
            max_used_w = max(int(max_used_w), int(band_used_w))
            cursor_r += int(band_h) + 2

        if max_used_w > 25:
            continue

        trailing_h = int(np.random.randint(1, 3))
        trailing_w = int(np.random.randint(1, 4))
        height = int(cursor_r + trailing_h)
        width = int(2 + max_used_w + trailing_w)
        if height > 30 or width > 30:
            continue

        canonical = np.full((height, width), BG, dtype=int)
        canonical[0, :] = BLUE
        canonical[:, 0] = RED
        canonical[0, 0] = MARKER

        all_rects = []
        reordered_band_count = 0
        for band_idx, specs in enumerate(band_specs):
            color_order = np.random.permutation(DOMINANT_COLORS)
            placed, used_w, band_h = _place_band(canonical, band_infos[band_idx]["top"], specs, color_order)
            if used_w != used_widths[band_idx] or band_h != band_infos[band_idx]["band_h"]:
                return None
            areas = [int(rect["area"]) for rect in placed]
            if list(np.argsort(areas)) != list(range(len(areas))):
                reordered_band_count += 1
            all_rects.extend(placed)

        if reordered_band_count < 1:
            continue
        if sum(int(rect["height"] != rect["width"]) for rect in all_rects) < 2:
            continue

        blue_orient = str(np.random.choice(["h", "v"]))
        corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
        inp = _transform(canonical, blue_orient, corner)

        transformed_rects = []
        for rect in all_rects:
            input_patch, input_box = _render_single_rect(rect["patch"], rect["box"], canonical.shape, blue_orient, corner)
            transformed_rects.append(
                {
                    "input_patch": input_patch,
                    "input_box": input_box,
                    "area": int(rect["area"]),
                    "height": int(rect["height"]),
                    "width": int(rect["width"]),
                }
            )

        rects = _enrich_transformed_rects(transformed_rects, inp.shape, blue_orient, corner)
        band_sizes: dict[int, int] = {}
        combo_diff_bands = 0
        for rect in rects:
            band_sizes[int(rect["db"])] = band_sizes.get(int(rect["db"]), 0) + 1
        for db in sorted(band_sizes):
            band = [rect for rect in rects if int(rect["db"]) == int(db)]
            by_area = [int(rect["id"]) for rect in sorted(band, key=lambda rect: (int(rect["area"]), int(rect["dr"]), int(rect["ar"]), int(rect["ab"]), int(rect["id"])))]
            by_red = [int(rect["id"]) for rect in sorted(band, key=lambda rect: (int(rect["dr"]), int(rect["ar"]), int(rect["ab"]), int(rect["id"])))]
            if by_area != by_red:
                combo_diff_bands += 1

        if len(band_sizes) < 2 or max(band_sizes.values()) < 2 or combo_diff_bands < 1:
            continue

        expected = _output_combo(rects, BG, blue_orient)
        original = _output_original(rects, BG, blue_orient)
        d10_only = _output_d10(rects, BG, blue_orient)
        if np.array_equal(expected, original) or np.array_equal(expected, d10_only):
            continue

        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue

        meta = {
            "band_count": int(len(band_sizes)),
            "max_band_size": int(max(band_sizes.values())),
            "combo_diff_bands": int(combo_diff_bands),
            "blue_orient": blue_orient,
            "corner": corner,
            "rect_count": int(len(rects)),
            "shape": tuple(map(int, inp.shape)),
        }
        return inp, expected, meta

    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        built = _build_attempt()
        if built is not None:
            inp, _expected, _meta = built
            return inp
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant 39")


def _variant_metrics(grid: np.ndarray) -> dict:
    bg, (_cr, _cc), blue_orient, _red_orient, rects = _extract_rectangles(np.array(grid, dtype=int))
    normalized = [
        {
            "id": int(i),
            "input_patch": rect["arr"],
            "input_box": rect["box"],
            "area": int(rect["area"]),
            "height": int(rect["arr"].shape[0]),
            "width": int(rect["arr"].shape[1]),
            "db": int(rect["db"]),
            "dr": int(rect["dr"]),
            "ab": int(rect["ab"]),
            "ar": int(rect["ar"]),
            "box": rect["box"],
            "arr": rect["arr"],
        }
        for i, rect in enumerate(rects)
    ]
    combo = _output_combo(normalized, bg, blue_orient)
    d10_only = _output_d10(normalized, bg, blue_orient)
    original = _output_original(normalized, bg, blue_orient)

    band_sizes: dict[int, int] = {}
    combo_diff_bands = 0
    mixed_height_bands = 0
    for rect in rects:
        band_sizes[int(rect["db"])] = band_sizes.get(int(rect["db"]), 0) + 1
    for db in sorted(band_sizes):
        band = [rect for rect in rects if int(rect["db"]) == int(db)]
        if len({int(rect["arr"].shape[0]) for rect in band}) > 1:
            mixed_height_bands += 1
        by_area = [tuple(map(int, rect["box"])) for rect in sorted(band, key=lambda rect: (int(rect["area"]), int(rect["dr"]), int(rect["ar"]), int(rect["ab"]), tuple(map(int, rect["box"])))) ]
        by_red = [tuple(map(int, rect["box"])) for rect in sorted(band, key=lambda rect: (int(rect["dr"]), int(rect["ar"]), int(rect["ab"]), tuple(map(int, rect["box"])))) ]
        if by_area != by_red:
            combo_diff_bands += 1

    return {
        "band_count": int(len(band_sizes)),
        "max_band_size": int(max(band_sizes.values())),
        "combo_diff_bands": int(combo_diff_bands),
        "mixed_height_bands": int(mixed_height_bands),
        "combo_diff_vs_d10": bool(not np.array_equal(combo, d10_only)),
        "combo_diff_vs_original": bool(not np.array_equal(combo, original)),
    }
