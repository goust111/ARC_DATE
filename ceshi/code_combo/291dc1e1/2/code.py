# Combo variant of 291dc1e1 (index 2)
# combo_spec: D1/0+D14/0
# Name: 面积排序+噪声过滤

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

"""
Design analysis for the combo variant D1/0 + D14/0

a) Concrete behavior and trace
- Original task:
  1. Detect the marked corner and infer which edge line is the blue axis.
  2. Treat every interior solid component as a rectangle.
  3. Sort rectangles by (distance to blue line, distance to red line).
  4. If blue is vertical, rotate every rectangle 90 degrees CCW, then stack.
- Combo task:
  1. Detect the same corner and edge-line frame.
  2. Scan all interior components, but keep only components that are at least 2x2
     and fully fill their bounding boxes. Jagged blobs are rejected as noise.
  3. Compute each kept rectangle's area and sort by area ascending instead of by
     distance to the two axes.
  4. Keep the original orientation-dependent rotation and vertical stacking.
- Small concrete example in canonical top-left form:
    top band:    2x5 rectangle, 2x2 rectangle
    bottom band: 3x4 rectangle, 3x2 rectangle
    noise:       one 3-cell L shape in the middle corridor
  Original order would be 2x5 -> 2x2 -> 3x4 -> 3x2 because the top band is
  closer to the blue line. The combo order is 2x2(area 4) -> 3x2(area 6) ->
  2x5(area 10) -> 3x4(area 12), while the L-shape disappears completely.
  The difference is therefore two-step and visible: filter first, then reorder.

b) Complexity calibration
- The generator keeps two rectangle bands with different fixed heights (2 and 3).
  That prevents area from collapsing into a simple "width-only" cue.
- Rectangle counts are limited to 4-6 total and areas are forced unique per
  sample, which keeps ordering deterministic and visually readable.
- Noise uses a small set of 3-5 cell irregular templates. This is enough to make
  anomaly detection visible without turning the scene into clutter.

c) Implicit-assumption audit of the original solver
- Original assumption: every interior non-background component can be interpreted
  as a rectangle if its raw crop has no background cells. This fails once noise
  blobs are added. The combo solver instead checks the component occupancy mask.
- Original assumption: distance-to-blue and distance-to-red are the main sorting
  signal. This is removed; distance is retained only inside the activity metric
  used to detect generator degeneration.
- Original assumption: orientation detection comes from the corner marker and
  edge-line colors. That still holds and is preserved unchanged.
- New interaction between changes: a noise blob must not accidentally be accepted
  as a valid rectangle. The solver therefore requires full bbox occupancy and a
  minimum size of 2x2 before area is even considered.

d) Placement legality
- Rectangles are placed first in two non-touching bands on a background canvas.
- Noise is added only onto cells that are empty and at least one 4-neighbor step
  away from rectangles, the corner marker, and the two edge lines.
- Whole-grid flips and rotations preserve all non-overlap and separation facts.
- Every accepted sample is validated by checking that `main(input)` matches the
  generator's intended output.

e) Visual observability
- Across a few examples, the human sees that jagged extra blobs never appear in
  the output, so they are distractors.
- The surviving objects do not remain in top-band-first order; instead smaller
  rectangles rise to the top of the stack even when they are farther from the
  blue line. This makes the area-based rule visible rather than latent.

f) Visual cleanliness
- Noise is restricted to background corridors above, between, or below the two
  rectangle bands.
- Dominant rectangle colors are distinct inside each sample, which makes object
  tracking easy after reordering.
- Sparse interior accent pixels are kept, but only inside valid rectangles, so
  they support rotation visibility without competing with the noise story.

g) Color-change inferability
- There is no hidden color remapping rule.
- Noise reuses ordinary non-background colors rather than one dedicated "trash"
  color, so filtering must be based on shape, not palette.
- Color count stays small and structured enough that the true rule is still
  attributable to geometry and area.

h) Generator-side activity metrics
- `rejected_noise_components`: how many interior connected components are seen
  but rejected by the rectangle-structure test.
- `order_changed`: whether stacking by area produces a different output from
  stacking the same filtered rectangles by the original distance rule.
- A sample is accepted only when both metrics are active.

i) New solver assumptions and how the generator enforces them
- Valid targets are 4-connected, bbox-filling rectangles of size at least 2x2.
- Noise must stay disconnected from valid rectangles under 4-connectivity.
- Area ties are possible in principle, but the generator rejects them for visual
  clarity; the solver still has stable spatial tie-breaks.
- The blue-line orientation still controls the global 90 degree CCW rotation.
"""

BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
ACCENT_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
NOISE_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
NOISE_TEMPLATES = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]

TOP_ROW = 2
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


def _extract_rectangles(
    g: np.ndarray,
) -> tuple[int, tuple[int, int], str, str, list[dict], int]:
    bg, corner, marker, blue_orient, red_orient = _locate_frame(g)
    cr, cc = corner
    H, W = g.shape

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict] = []
    rejected = 0

    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 2 or w < 2:
            rejected += 1
            continue

        occ = np.zeros((h, w), dtype=bool)
        occ[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if not bool(occ.all()):
            rejected += 1
            continue

        rects.append(
            {
                "box": (int(r0), int(c0), int(r1), int(c1)),
                "arr": crop_bbox(g, (int(r0), int(c0), int(r1), int(c1))),
                "area": int(h * w),
            }
        )

    if not rects:
        raise AssertionError("no rectangles found")
    return int(bg), (int(cr), int(cc)), blue_orient, red_orient, rects, int(rejected)


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


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


def _stack_sequence(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(a.shape[1]) for a in seq))
    out_h = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        paste(out, arr, int(out_r), int((out_w - w) // 2))
        out_r += int(h)
    return out


def _sorted_rects_by_area(rects: list[dict], blue_orient: str, red_orient: str, cr: int, cc: int) -> list[dict]:
    return sorted(
        rects,
        key=lambda rect: (
            int(rect["area"]),
            _along_from_corner(rect["box"], blue_orient, cr, cc),
            _along_from_corner(rect["box"], red_orient, cr, cc),
            tuple(map(int, rect["box"])),
        ),
    )


def _sorted_rects_by_original_distance(
    rects: list[dict], blue_orient: str, red_orient: str, cr: int, cc: int
) -> list[dict]:
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    return sorted(
        rects,
        key=lambda rect: (
            _dist_to_edge(rect["box"], blue_orient, blue_edge_idx),
            _dist_to_edge(rect["box"], red_orient, red_edge_idx),
            _along_from_corner(rect["box"], blue_orient, cr, cc),
            _along_from_corner(rect["box"], red_orient, cr, cc),
            tuple(map(int, rect["box"])),
        ),
    )


def _build_output_from_rects(rects: list[dict], blue_orient: str, bg: int) -> np.ndarray:
    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    return _stack_sequence(seq, bg)


def activity_metrics(grid: np.ndarray) -> dict:
    """
    Quantify whether the combo behavior is active on a concrete input.

    Returns:
    - rejected_noise_components: number of interior components rejected as noise
    - order_changed: 1 iff area-order output differs from original distance-order
    - rect_count: number of accepted rectangles
    """
    g = np.array(grid, dtype=int)
    bg, (cr, cc), blue_orient, red_orient, rects, rejected = _extract_rectangles(g)
    area_sorted = _sorted_rects_by_area(rects, blue_orient, red_orient, cr, cc)
    dist_sorted = _sorted_rects_by_original_distance(rects, blue_orient, red_orient, cr, cc)
    area_out = _build_output_from_rects(area_sorted, blue_orient, bg)
    dist_out = _build_output_from_rects(dist_sorted, blue_orient, bg)
    return {
        "rejected_noise_components": int(rejected),
        "order_changed": int(not np.array_equal(area_out, dist_out)),
        "rect_count": int(len(rects)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter non-rectangular noise, sort the remaining rectangles by area, rotate
    them if the blue edge is vertical, then stack them.
    """
    g = np.array(grid, dtype=int)
    bg, (cr, cc), blue_orient, red_orient, rects, _rejected = _extract_rectangles(g)
    ordered = _sorted_rects_by_area(rects, blue_orient, red_orient, cr, cc)
    return _build_output_from_rects(ordered, blue_orient, bg)


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid.copy()
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
                "patch": patch.copy(),
                "area": int(h * w),
                "box": (int(row), int(c), int(row + h - 1), int(c + w - 1)),
            }
        )
        c += int(w + 2)
    return rects


def _neighbor_block(mask: np.ndarray) -> np.ndarray:
    blocked = mask.copy()
    blocked[1:, :] |= mask[:-1, :]
    blocked[:-1, :] |= mask[1:, :]
    blocked[:, 1:] |= mask[:, :-1]
    blocked[:, :-1] |= mask[:, 1:]
    return blocked


def _paint_noise_patch(shape: np.ndarray) -> np.ndarray:
    patch = np.full(shape.shape, BG, dtype=int)
    n_colors = int(np.random.randint(1, 3))
    palette = np.random.choice(NOISE_COLORS, size=n_colors, replace=False)
    coords = np.argwhere(shape)
    for idx, (r, c) in enumerate(coords):
        patch[int(r), int(c)] = int(palette[idx % len(palette)])
    return patch


def _add_noise(grid: np.ndarray, occupied_seed: np.ndarray) -> int:
    H, W = grid.shape
    occupied = occupied_seed.copy()
    noise_count = 0
    target = int(np.random.randint(1, 4))

    for _ in range(target):
        blocked = _neighbor_block(occupied)
        candidates: list[tuple[int, int, np.ndarray]] = []
        shape_order = np.random.permutation(len(NOISE_TEMPLATES))
        for shape_idx in shape_order:
            shape = NOISE_TEMPLATES[int(shape_idx)]
            h, w = shape.shape
            for r in range(1, H - h + 1):
                for c in range(1, W - w + 1):
                    cells = np.argwhere(shape) + np.array([[r, c]])
                    if np.any(blocked[cells[:, 0], cells[:, 1]]):
                        continue
                    candidates.append((int(r), int(c), shape))
        if not candidates:
            break
        r, c, shape = candidates[int(np.random.randint(len(candidates)))]
        patch = _paint_noise_patch(shape)
        paste(grid, patch, int(r), int(c), transparent=BG)
        cells = np.argwhere(shape) + np.array([[r, c]])
        occupied[cells[:, 0], cells[:, 1]] = True
        noise_count += 1

    return int(noise_count)


def _render_single_patch(
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


def _old_distance_order_ids(rects: list[dict], input_shape: tuple[int, int], blue_orient: str, corner: str) -> list[int]:
    cr, cc = _corner_coords(input_shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    ordered = sorted(
        rects,
        key=lambda rect: (
            _dist_to_edge(rect["input_box"], blue_orient, blue_edge_idx),
            _dist_to_edge(rect["input_box"], red_orient, red_edge_idx),
            _along_from_corner(rect["input_box"], blue_orient, cr, cc),
            _along_from_corner(rect["input_box"], red_orient, cr, cc),
            int(rect["id"]),
        ),
    )
    return [int(rect["id"]) for rect in ordered]


def _area_order_ids(rects: list[dict], input_shape: tuple[int, int], blue_orient: str, corner: str) -> list[int]:
    cr, cc = _corner_coords(input_shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    ordered = sorted(
        rects,
        key=lambda rect: (
            int(rect["area"]),
            _along_from_corner(rect["input_box"], blue_orient, cr, cc),
            _along_from_corner(rect["input_box"], red_orient, cr, cc),
            int(rect["id"]),
        ),
    )
    return [int(rect["id"]) for rect in ordered]


def _build_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    count_options = [(2, 2), (2, 3), (3, 2), (3, 3)]
    n_top, n_bottom = count_options[int(np.random.randint(len(count_options)))]

    top_specs = _rect_specs_for_band(TOP_H, n_top)
    bottom_specs = _rect_specs_for_band(BOTTOM_H, n_bottom)
    all_areas = [h * w for h, w in top_specs + bottom_specs]
    if len(set(all_areas)) != len(all_areas):
        return None

    gap_rows = int(np.random.randint(2, 5))
    bottom_row = int(TOP_ROW + TOP_H + gap_rows)
    bottom_margin = int(np.random.randint(2, 5))
    height = int(bottom_row + BOTTOM_H + bottom_margin)

    need_w_top = 2 + sum(w for _h, w in top_specs) + 2 * (len(top_specs) - 1) + int(np.random.randint(2, 5))
    need_w_bottom = 2 + sum(w for _h, w in bottom_specs) + 2 * (len(bottom_specs) - 1) + int(np.random.randint(2, 5))
    width = int(max(need_w_top, need_w_bottom))

    canonical = np.full((height, width), BG, dtype=int)
    canonical[0, :] = BLUE
    canonical[:, 0] = RED
    canonical[0, 0] = MARKER

    rect_mask = np.zeros_like(canonical, dtype=bool)
    frame_mask = np.zeros_like(canonical, dtype=bool)
    frame_mask[0, :] = True
    frame_mask[:, 0] = True

    colors = np.random.permutation(DOMINANT_COLORS)
    rects = []
    rects.extend(_place_band(canonical, TOP_ROW, top_specs, colors[:n_top]))
    rects.extend(_place_band(canonical, bottom_row, bottom_specs, colors[n_top : n_top + n_bottom]))

    for rect in rects:
        r0, c0, r1, c1 = map(int, rect["box"])
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    if sum(int(rect["patch"].shape[0] != rect["patch"].shape[1]) for rect in rects) < 2:
        return None

    noise_count = _add_noise(canonical, rect_mask | frame_mask)
    if noise_count < 1:
        return None

    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    inp = _transform(canonical, blue_orient, corner)

    transformed_rects = []
    for idx, rect in enumerate(rects):
        input_patch, input_box = _render_single_patch(rect["patch"], rect["box"], canonical.shape, blue_orient, corner)
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

    rect_by_id = {int(rect["id"]): rect for rect in transformed_rects}
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

    metrics = activity_metrics(inp)
    if int(metrics["rejected_noise_components"]) < 1 or int(metrics["order_changed"]) != 1:
        return None

    meta = {
        "blue_orient": blue_orient,
        "corner": corner,
        "rect_count": len(rects),
        "noise_count": int(noise_count),
        "rejected_noise_components": int(metrics["rejected_noise_components"]),
        "old_ids": old_ids,
        "area_ids": area_ids,
    }
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(600):
        built = _build_attempt()
        if built is not None:
            inp, _expected, _meta = built
            return inp
    raise RuntimeError("failed to generate input for 291dc1e1 combo D1/0+D14/0")
