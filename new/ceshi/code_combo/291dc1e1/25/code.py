# Combo variant of 291dc1e1 (index 25)
# combo_spec: D1/0+D10/0
# Name: 面积排序+分band分步堆叠

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "solvers" / "arc_utils.py").exists():
            return parent
    raise RuntimeError("workspace root with solvers/arc_utils.py not found")


ROOT = _find_root()
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the combo D1/0 + D10/0 variant
# a) Concrete behavior and trace:
#    The source task finds the corner marker plus blue/red edge lines, extracts
#    interior solid rectangles, and globally sorts them by
#    (distance-to-blue, distance-to-red). This combo keeps the same frame
#    detection and the same orientation-controlled 90 degree CCW rotation when
#    the blue edge is vertical, but changes the ordering in two coupled ways:
#    first partition by blue-distance band, then sort by area inside each band.
#    Example in the canonical top-left / horizontal-blue layout:
#      near band: 2x5 rectangle A at dr=1, 2x2 rectangle B at dr=8
#      far band:  3x4 rectangle C at dr=1, 3x2 rectangle D at dr=7
#    Original 291dc1e1 output order is A,B,C,D because blue-distance is primary
#    and red-distance is the within-band order.
#    D1-only would ignore band boundaries and sort globally by area:
#      B(4), D(6), C(12), A(10) -> B,D,A,C after full tie-breaking.
#    This combo instead does two explicit steps:
#      near band by area: B then A
#      separator row
#      far band by area: D then C
#    If blue is vertical, each rectangle is rotated after its order is chosen.
#    The visible effect is therefore not just "different order", but "area
#    reasoning repeated per band, with a full blank row between stages."
# b) Complexity calibration:
#    If both bands used the same rectangle height, area within a band would
#    reduce to width. The generator therefore uses two fixed band heights
#    (2 cells for the near band, 3 for the far band), so area remains a genuine
#    count-based property and the solver cannot collapse the combo to one-axis
#    geometry. Each sample still keeps only two bands and 2-3 rectangles per
#    band, which is enough repeated evidence for humans while staying visually
#    clean.
# c) Implicit-assumption audit on the original solver:
#    - Corner detection assumes one corner marker distinct from both adjacent
#      line colors. The generator preserves that exactly.
#    - Blue edge detection prefers color 1 or falls back to the longer line.
#      The generator keeps color 1 as blue, so this remains valid.
#    - Rectangle extraction assumes each object fills its bbox with no
#      background holes. The generator keeps solid rectangular patches and only
#      adds sparse non-background accent cells inside them, so area can still be
#      measured by counting non-background cells in the crop.
#    - The original `rects.sort(...)` assumes one global order. That no longer
#      matches the combo rule, so ordering is moved inside blue-distance groups.
#    - The original output height assumes no gaps. The combo needs separator
#      rows, so output height now adds one background row between successive
#      bands.
# d) Placement legality:
#    The canonical generator places the near band at db=1 and the far band at a
#    larger blue-distance, with at least two background columns between adjacent
#    rectangles inside a band and at least two background rows between bands.
#    Whole-grid flips/rotations preserve non-overlap and keep the two bands
#    disjoint, so component extraction stays reliable.
# e) Visual observability:
#    A human looking at 2-4 examples can see two stable cues:
#    - output is processed in two steps because a full background row separates
#      the piles
#    - inside each step, object order follows size rather than left-to-right /
#      red-distance order
#    The generator rejects samples where area order accidentally matches the
#    red-distance order in both bands, because those would hide the D1 part.
# f) Visual cleanliness:
#    The scene remains the original clean motif: one marker, two border lines,
#    and two interior bands of rectangles on uniform background. Distinct
#    dominant colors per rectangle plus sparse accents make object identity easy
#    to track without introducing extra symbols.
# g) Color-change inferability:
#    No color remapping is part of the rule. Colors stay attached to objects, so
#    the inference burden is purely about grouping, area comparison, and the
#    separator row. Sparse accents are only there to keep rotated objects
#    visually traceable.
# h) Generator validation:
#    Each attempt constructs rectangle metadata, transforms the full scene,
#    computes the intended combo output from metadata, compares it with
#    `main(input)`, and retries on any mismatch. It also rejects attempts whose
#    combo output equals the D10-only output, because that would mean the
#    area-sorting control variable is not visibly active.
# i) New solver assumptions after the edit:
#    The solver now assumes rectangles can be grouped by discrete blue-distance
#    values and that their areas are recoverable from solid bbox crops. The
#    generator guarantees exactly two non-empty bands and keeps each band's
#    rectangle areas distinct, while the solver still contains spatial
#    tie-breakers so it remains deterministic even if equal areas appeared.


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
CANONICAL_H = 11


def _locate_frame(g: np.ndarray) -> tuple[int, tuple[int, int], int, str, str]:
    H, W = g.shape
    bg = int(get_bg(g))
    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        neighbors = []
        if r == 0:
            neighbors.append(int(g[r + 1, c]))
        if r == H - 1:
            neighbors.append(int(g[r - 1, c]))
        if c == 0:
            neighbors.append(int(g[r, c + 1]))
        if c == W - 1:
            neighbors.append(int(g[r, c - 1]))
        if neighbors and all(int(n) != int(g[r, c]) for n in neighbors):
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
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append(
            {
                "box": box,
                "arr": sub.copy(),
                "area": int(np.count_nonzero(sub != int(bg))),
                "db": int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
                "dr": int(_dist_to_edge(box, red_orient, red_edge_idx)),
            }
        )

    assert rects
    return int(bg), (int(cr), int(cc)), blue_orient, red_orient, rects


def _within_band_key(rect: dict, red_orient: str, blue_orient: str, cr: int, cc: int) -> tuple[int, int, int, tuple[int, int, int, int]]:
    box = tuple(map(int, rect["box"]))
    return (
        int(rect["area"]),
        _along_from_corner(box, red_orient, cr, cc),
        _along_from_corner(box, blue_orient, cr, cc),
        box,
    )


def _stack_bands(bands: list[list[np.ndarray]], bg: int) -> np.ndarray:
    out_w = int(max(int(arr.shape[1]) for band in bands for arr in band))
    out_h = int(sum(int(arr.shape[0]) for band in bands for arr in band) + max(0, len(bands) - 1))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for band_idx, band in enumerate(bands):
        if band_idx > 0:
            out_r += 1
        for arr in band:
            h, w = map(int, arr.shape)
            out_c = int((out_w - w) // 2)
            paste(out, arr, int(out_r), int(out_c))
            out_r += int(h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack rectangles in blue-distance stages, using area order inside each stage.

    Rule:
    - find the marked corner and the two edge lines as in 291dc1e1
    - group rectangles by distance to the blue edge, near band first
    - sort rectangles by area inside each band
    - rotate each rectangle 90 degrees CCW if the blue edge is vertical
    - stack the processed bands top-to-bottom with one full background separator
      row between successive bands
    """
    g = np.array(grid, dtype=int)
    bg, (cr, cc), blue_orient, red_orient, rects = _extract_rectangles(g)

    bands: list[list[np.ndarray]] = []
    for db in sorted({int(rect["db"]) for rect in rects}):
        group = [rect for rect in rects if int(rect["db"]) == int(db)]
        group.sort(key=lambda rect: _within_band_key(rect, red_orient, blue_orient, cr, cc))
        band_arrays = []
        for rect in group:
            arr = rect["arr"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            band_arrays.append(arr)
        bands.append(band_arrays)

    return _stack_bands(bands, bg)


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
    n_accents = 1 + int((h * w >= 8) and (np.random.rand() < 0.8))
    n_accents += int((h * w >= 12) and (np.random.rand() < 0.35))
    n_accents = min(n_accents, len(candidates))
    accent_pool = [int(c) for c in ACCENT_COLORS if int(c) not in {int(dominant), int(BG)}]
    np.random.shuffle(accent_pool)
    for idx, (r, c) in enumerate(candidates[:n_accents]):
        patch[int(r), int(c)] = int(accent_pool[idx])
    return patch


def _place_band(grid: np.ndarray, row: int, height: int, widths: list[int], colors: np.ndarray) -> list[dict]:
    rects = []
    c = 2
    for idx, w in enumerate(widths):
        patch = _decorate_patch(int(height), int(w), int(colors[idx]))
        paste(grid, patch, int(row), int(c))
        rects.append(
            {
                "patch": patch,
                "area": int(height * w),
                "box": (int(row), int(c), int(row + height - 1), int(c + w - 1)),
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


def _metadata_band_lists(
    rects: list[dict], input_shape: tuple[int, int], blue_orient: str, corner: str, mode: str
) -> list[list[int]]:
    cr, cc = _corner_coords(input_shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    enriched = []
    for rect in rects:
        box = tuple(map(int, rect["input_box"]))
        enriched.append(
            {
                "id": int(rect["id"]),
                "area": int(rect["area"]),
                "box": box,
                "db": int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
                "dr": int(_dist_to_edge(box, red_orient, red_edge_idx)),
            }
        )

    bands = []
    for db in sorted({int(rect["db"]) for rect in enriched}):
        group = [rect for rect in enriched if int(rect["db"]) == int(db)]
        if mode == "combo":
            group.sort(
                key=lambda rect: (
                    int(rect["area"]),
                    _along_from_corner(rect["box"], red_orient, cr, cc),
                    _along_from_corner(rect["box"], blue_orient, cr, cc),
                    rect["box"],
                )
            )
        elif mode == "d10":
            group.sort(
                key=lambda rect: (
                    int(rect["dr"]),
                    _along_from_corner(rect["box"], red_orient, cr, cc),
                    _along_from_corner(rect["box"], blue_orient, cr, cc),
                    rect["box"],
                )
            )
        else:
            raise ValueError(mode)
        bands.append([int(rect["id"]) for rect in group])
    return bands


def _build_output_from_band_ids(
    band_ids: list[list[int]], rect_by_id: dict[int, dict], blue_orient: str
) -> np.ndarray:
    bands = []
    for ids in band_ids:
        arrs = []
        for rid in ids:
            arr = rect_by_id[int(rid)]["input_patch"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            arrs.append(arr)
        bands.append(arrs)
    return _stack_bands(bands, BG)


def _build_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    count_options = [(2, 2), (2, 3), (3, 2), (3, 3)]
    near_count, far_count = count_options[int(np.random.randint(len(count_options)))]

    width_pool = np.arange(2, 7, dtype=int)
    near_widths = [int(w) for w in np.random.choice(width_pool, size=near_count, replace=False)]
    far_widths = [int(w) for w in np.random.choice(width_pool, size=far_count, replace=False)]

    near_area_order = [TOP_H * int(w) for w in near_widths]
    far_area_order = [BOTTOM_H * int(w) for w in far_widths]
    if len(set(near_area_order)) != near_count or len(set(far_area_order)) != far_count:
        return None

    need_w_near = 2 + sum(near_widths) + 2 * (near_count - 1) + int(np.random.randint(1, 4))
    need_w_far = 2 + sum(far_widths) + 2 * (far_count - 1) + int(np.random.randint(1, 4))
    width = int(max(need_w_near, need_w_far))

    canonical = np.full((CANONICAL_H, width), BG, dtype=int)
    canonical[0, :] = BLUE
    canonical[:, 0] = RED
    canonical[0, 0] = MARKER

    colors = np.random.permutation(DOMINANT_COLORS)
    rects = []
    rects.extend(_place_band(canonical, TOP_ROW, TOP_H, near_widths, colors[:near_count]))
    rects.extend(_place_band(canonical, BOTTOM_ROW, BOTTOM_H, far_widths, colors[near_count : near_count + far_count]))

    if sum(int(rect["patch"].shape[0] != rect["patch"].shape[1]) for rect in rects) < 2:
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

    combo_bands = _metadata_band_lists(transformed_rects, inp.shape, blue_orient, corner, mode="combo")
    d10_bands = _metadata_band_lists(transformed_rects, inp.shape, blue_orient, corner, mode="d10")
    if len(combo_bands) != 2 or min(len(band) for band in combo_bands) < 2:
        return None
    if all(combo_bands[idx] == d10_bands[idx] for idx in range(len(combo_bands))):
        return None

    rect_by_id = {int(rect["id"]): rect for rect in transformed_rects}
    expected = _build_output_from_band_ids(combo_bands, rect_by_id, blue_orient)
    d10_output = _build_output_from_band_ids(d10_bands, rect_by_id, blue_orient)
    if np.array_equal(expected, d10_output):
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    meta = {
        "blue_orient": blue_orient,
        "corner": corner,
        "near_count": near_count,
        "far_count": far_count,
        "combo_bands": combo_bands,
        "d10_bands": d10_bands,
        "mismatch_bands": int(sum(combo_bands[idx] != d10_bands[idx] for idx in range(len(combo_bands)))),
    }
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(600):
        built = _build_attempt()
        if built is not None:
            inp, _expected, _meta = built
            return inp
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant D1/0+D10/0")
