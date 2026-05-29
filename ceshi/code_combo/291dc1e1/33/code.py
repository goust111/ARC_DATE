# Combo variant of 291dc1e1 (index 33)
# combo_spec: D1/0+D8/1+D14/0
# Name: 面积排序+少数色优先+噪声过滤

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste
except ImportError:
    ROOT = Path(__file__).resolve().parents[5]
    for helper_dir in (ROOT / "solvers", ROOT / "arc2_solver_select"):
        helper_str = str(helper_dir)
        if helper_str not in sys.path:
            sys.path.append(helper_str)
    from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for combo D1/0 + D8/1 + D14/0

a) Concrete behavior and trace
- Base scene stays recognizable: one marked corner, a full blue edge line, a full red
  edge line, and several interior filled rectangles.
- Fusion change 1 (D14): interior non-background components are no longer guaranteed
  to all be valid rectangles, because 1-3 detached jagged noise blobs are added.
- Fusion change 2 (D8): among valid rectangles, some are pure solid blocks and some
  contain a small minority-color sprinkle while still fully filling the bbox.
- Fusion change 3 (D1): after filtering and classifying, valid rectangles are stacked
  by a new composite key `(has_minority desc, area asc)` rather than by edge distance.
- Small concrete canonical example:
    top band:    pure 2x4 area 8, minority 2x8 area 16
    bottom band: minority 3x4 area 12, pure 3x6 area 18
    plus one detached L-shaped noise blob elsewhere
  Original solver path:
    1. Treat every interior component as a candidate.
    2. Sort surviving rectangles by distance to blue/red lines:
       top-left pure -> top-right minority -> bottom-left minority -> bottom-right pure.
  Fused path:
    1. Reject the L-shape because its component does not fill its bbox.
    2. Classify the four rectangles by whether they contain minority pixels.
    3. Sort minority rectangles first, by area: 12 then 16.
    4. Then sort pure rectangles by area: 8 then 18.
  Difference:
    - Noise vanishes from the output entirely.
    - A farther minority rectangle can jump ahead of a nearer pure rectangle.
    - Within the minority group, area order can reverse the original near/far order.

b) Complexity calibration
- The minority criterion is a single boolean, not a many-way texture taxonomy, so the
  rule remains inferable from 2-4 demonstrations.
- Areas are chosen from a small discrete set and kept distinct within a sample so the
  area ordering is visible instead of collapsing into ties.
- Noise uses only a few 3-5 cell templates, enough to force filtering but not enough
  to clutter the canvas or create an uncontrolled object zoo.

c) Implicit-assumption audit of the original solver
- Original `mask = (g != bg) & (~line_mask) & (g != marker)` is still fine, but the
  later assumption "every component with no background inside crop is a rectangle" is
  no longer sufficient once non-rectangular noise exists.
- The fused solver therefore validates each component structurally: bbox dimensions
  must both be at least 2 and the component occupancy must fill the entire bbox.
- Original sorting depended on `db`/`dr`; that assumption is removed. Distances now
  matter only in activity diagnostics, not in the actual output rule.
- New color-composition logic assumes valid rectangles have one dominant color and at
  most 25% minority cells. The generator enforces exactly that.
- Rotation-by-blue-orientation remains valid, so frame detection is preserved.

d) Placement legality
- Rectangles are placed in two separated bands, starting one cell away from the red
  edge and never touching each other.
- Noise is placed only on currently empty cells and kept 4-neighbor-separated from
  rectangles, other noise, and the edge lines by using a dilated occupancy mask.
- Whole-grid flips/rotations preserve non-overlap and all required gaps.

e) Visual observability
- A human sees three repeated cues at once:
  1. jagged blobs disappear, so they are distractors;
  2. multi-color rectangles are grouped before pure rectangles;
  3. inside each group, smaller rectangles stack above larger ones.
- The generator forces samples where both the minority priority and the area key alter
  the order relative to simpler baselines, so the fusion does not collapse to a single
  visible cue.

f) Visual cleanliness
- Noise is restricted to empty corridors above, between, or below the two bands.
- Rectangle dominant colors are sampled without replacement, which makes object
  tracking easy even after stacking and optional rotation.
- Minority sprinkles are sparse, so rectangles still read as rectangles rather than
  as noisy mosaics.

g) Color-change inferability
- No color mapping rule exists. Colors only serve object identity plus the visible
  "this rectangle has a few off-color cells" marker.
- The palette stays small and familiar; noise reuses ordinary object colors so the
  solver cannot rely on a dedicated "ignore this color" shortcut.

h) Generator validation and fusion-activity metric
- Each generated candidate is transformed, then its intended output is reconstructed
  from per-rectangle metadata and compared against `main(input)`.
- The generator additionally measures:
  `rejected_noise_components`,
  whether fused order differs from priority+distance order,
  whether fused order differs from area-only order,
  and whether fused order differs from the original distance order.
- Samples are accepted only when all three fusion levers are active.

i) New solver assumptions after the edit
- Legal target objects are 4-connected, bbox-filling rectangles with height >= 2 and
  width >= 2.
- Minority-bearing rectangles remain bbox-filling because minority cells never use the
  background color.
- At least one rectangle exists in each minority class for visual evidence; generator
  guarantees two minority and two pure rectangles.
- Noise remains detached by one 4-neighbor gap so structural extraction is stable.
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
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


def _locate_corner(g: np.ndarray, bg: int) -> tuple[int, int]:
    H, W = map(int, g.shape)
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
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
            return int(r), int(c)
    raise AssertionError("corner marker not found")


def _infer_axes(g: np.ndarray, cr: int, cc: int) -> tuple[int, str, str]:
    H, W = map(int, g.shape)
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
    return int(marker), blue_orient, red_orient


def _has_minority_pixels(arr: np.ndarray) -> bool:
    vals, counts = np.unique(arr, return_counts=True)
    if len(vals) <= 1:
        return False
    area = int(arr.size)
    majority = int(counts.max())
    minority_total = int(area - majority)
    return 1 <= minority_total <= area // 4


def _extract_rectangles(g: np.ndarray, bg: int, marker: int, cr: int, cc: int) -> tuple[list[dict], int]:
    H, W = map(int, g.shape)
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict] = []
    rejected_noise = 0

    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 2 or w < 2:
            rejected_noise += 1
            continue

        occ = np.zeros((h, w), dtype=bool)
        occ[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if not bool(occ.all()):
            rejected_noise += 1
            continue

        box = (int(r0), int(c0), int(r1), int(c1))
        arr = crop_bbox(g, box)
        rects.append(
            {
                "box": box,
                "arr": arr.copy(),
                "area": int(h * w),
                "has_minority": bool(_has_minority_pixels(arr)),
            }
        )

    if not rects:
        raise AssertionError("no valid rectangles found")
    return rects, int(rejected_noise)


def _analyze_grid(grid: np.ndarray) -> tuple[int, int, int, str, str, list[dict], int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc = _locate_corner(g, bg)
    marker, blue_orient, red_orient = _infer_axes(g, cr, cc)
    rects, rejected_noise = _extract_rectangles(g, bg, marker, cr, cc)
    return int(bg), int(cr), int(cc), blue_orient, red_orient, rects, int(rejected_noise)


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


def _sorted_rectangles(rects: list[dict], cr: int, cc: int, blue_orient: str, red_orient: str) -> list[dict]:
    return sorted(
        rects,
        key=lambda rect: (
            0 if bool(rect["has_minority"]) else 1,
            int(rect["area"]),
            _along_from_corner(rect["box"], blue_orient, cr, cc),
            _along_from_corner(rect["box"], red_orient, cr, cc),
            tuple(map(int, rect["box"])),
        ),
    )


def _distance_sorted_rectangles(rects: list[dict], cr: int, cc: int, blue_orient: str, red_orient: str) -> list[dict]:
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


def _priority_distance_sorted_rectangles(
    rects: list[dict], cr: int, cc: int, blue_orient: str, red_orient: str
) -> list[dict]:
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    return sorted(
        rects,
        key=lambda rect: (
            0 if bool(rect["has_minority"]) else 1,
            _dist_to_edge(rect["box"], blue_orient, blue_edge_idx),
            _dist_to_edge(rect["box"], red_orient, red_edge_idx),
            _along_from_corner(rect["box"], blue_orient, cr, cc),
            _along_from_corner(rect["box"], red_orient, cr, cc),
            tuple(map(int, rect["box"])),
        ),
    )


def _area_only_sorted_rectangles(rects: list[dict], cr: int, cc: int, blue_orient: str, red_orient: str) -> list[dict]:
    return sorted(
        rects,
        key=lambda rect: (
            int(rect["area"]),
            _along_from_corner(rect["box"], blue_orient, cr, cc),
            _along_from_corner(rect["box"], red_orient, cr, cc),
            tuple(map(int, rect["box"])),
        ),
    )


def _stack_sequence(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        paste(out, arr, int(out_r), int((out_w - w) // 2))
        out_r += int(h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Ignore non-rectangular interior noise components.
    2. Among valid rectangles, minority-bearing ones have higher priority.
    3. Inside each priority class, sort by area ascending.
    4. If the blue edge is vertical, rotate every rectangle 90 degrees CCW.
    5. Stack the cropped rectangles top-to-bottom, centered horizontally.
    """
    bg, cr, cc, blue_orient, red_orient, rects, _rejected_noise = _analyze_grid(grid)
    ordered = _sorted_rectangles(rects, cr, cc, blue_orient, red_orient)

    seq = []
    for rect in ordered:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    return _stack_sequence(seq, bg)


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


def _minority_cells(h: int, w: int, n_minor: int) -> list[tuple[int, int]]:
    candidates = [
        (0, 0),
        (h - 1, w - 1),
        (0, w - 1),
        (h - 1, 0),
        (h // 2, w // 2),
        (0, max(0, w // 2)),
        (min(h - 1, 1), max(0, w // 2 - 1)),
    ]
    chosen: list[tuple[int, int]] = []
    used = set()
    for cell in candidates:
        rr, cc = map(int, cell)
        if (rr, cc) in used:
            continue
        used.add((rr, cc))
        chosen.append((rr, cc))
        if len(chosen) == int(n_minor):
            return chosen
    return chosen


def _make_patch(h: int, w: int, dominant: int, has_minority: bool) -> np.ndarray:
    patch = np.full((int(h), int(w)), int(dominant), dtype=int)
    if not has_minority:
        return patch

    area = int(h * w)
    max_minor = min(3, area // 4)
    if max_minor < 2:
        raise AssertionError("minority rectangle too small")
    n_minor = 2 if max_minor == 2 else int(np.random.choice([2, 3]))
    colors = [int(c) for c in MINORITY_COLORS if int(c) != int(dominant)]
    chosen = np.random.choice(colors, size=int(np.random.choice([1, min(2, n_minor)])), replace=False)
    for idx, (r, c) in enumerate(_minority_cells(int(h), int(w), int(n_minor))):
        patch[int(r), int(c)] = int(chosen[idx % len(chosen)])
    return patch


def _neighbor_block(mask: np.ndarray) -> np.ndarray:
    blocked = mask.copy()
    blocked[1:, :] |= mask[:-1, :]
    blocked[:-1, :] |= mask[1:, :]
    blocked[:, 1:] |= mask[:, :-1]
    blocked[:, :-1] |= mask[:, 1:]
    return blocked


def _paint_noise_patch(shape: np.ndarray) -> np.ndarray:
    patch = np.full(shape.shape, BG, dtype=int)
    palette = np.random.choice(DOMINANT_COLORS, size=int(np.random.randint(1, 3)), replace=False)
    coords = np.argwhere(shape)
    for idx, (r, c) in enumerate(coords):
        patch[int(r), int(c)] = int(palette[idx % len(palette)])
    return patch


def _add_noise(grid: np.ndarray, occupied: np.ndarray) -> int:
    H, W = map(int, grid.shape)
    count = 0
    current = occupied.copy()

    for _ in range(int(np.random.randint(1, 4))):
        blocked = _neighbor_block(current)
        candidates: list[tuple[int, int, np.ndarray]] = []
        for shape in NOISE_TEMPLATES:
            h, w = map(int, shape.shape)
            for r in range(1, H - h):
                for c in range(1, W - w):
                    cells = np.argwhere(shape) + np.array([[r, c]])
                    if np.any(blocked[cells[:, 0], cells[:, 1]]):
                        continue
                    candidates.append((int(r), int(c), shape))
        if not candidates:
            break
        r0, c0, shape = candidates[int(np.random.randint(len(candidates)))]
        patch = _paint_noise_patch(shape)
        paste(grid, patch, int(r0), int(c0), transparent=BG)
        cells = np.argwhere(shape) + np.array([[r0, c0]])
        current[cells[:, 0], cells[:, 1]] = True
        count += 1
    return int(count)


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
    bbox = tuple(map(int, bbox_coords(coords)))
    return crop_bbox(transformed, bbox), bbox


def _stack_expected(rects: list[dict], blue_orient: str) -> np.ndarray:
    seq = []
    for rect in rects:
        arr = rect["input_patch"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    return _stack_sequence(seq, BG)


def _activity_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    bg, cr, cc, blue_orient, red_orient, rects, rejected_noise = _analyze_grid(grid)
    del bg
    fused = _sorted_rectangles(list(rects), cr, cc, blue_orient, red_orient)
    by_distance = _distance_sorted_rectangles(list(rects), cr, cc, blue_orient, red_orient)
    by_priority_distance = _priority_distance_sorted_rectangles(list(rects), cr, cc, blue_orient, red_orient)
    by_area_only = _area_only_sorted_rectangles(list(rects), cr, cc, blue_orient, red_orient)

    minority_count = sum(int(bool(rect["has_minority"])) for rect in rects)
    pure_count = len(rects) - minority_count
    fused_ids = [tuple(rect["box"]) for rect in fused]
    distance_ids = [tuple(rect["box"]) for rect in by_distance]
    priority_distance_ids = [tuple(rect["box"]) for rect in by_priority_distance]
    area_only_ids = [tuple(rect["box"]) for rect in by_area_only]

    return {
        "rect_count": int(len(rects)),
        "minority_count": int(minority_count),
        "pure_count": int(pure_count),
        "rejected_noise_components": int(rejected_noise),
        "diff_vs_distance": bool(fused_ids != distance_ids),
        "diff_vs_priority_distance": bool(fused_ids != priority_distance_ids),
        "diff_vs_area_only": bool(fused_ids != area_only_ids),
    }


def _build_canonical() -> tuple[np.ndarray, list[dict], int]:
    bottom_row = int(TOP_ROW + TOP_H + np.random.randint(2, 5))
    height = int(bottom_row + BOTTOM_H + np.random.randint(2, 5))

    top_specs = [
        {"h": TOP_H, "w": 4, "has_minority": False},
        {"h": TOP_H, "w": 8, "has_minority": True},
    ]
    bottom_specs = [
        {"h": BOTTOM_H, "w": 4, "has_minority": True},
        {"h": BOTTOM_H, "w": 6, "has_minority": False},
    ]
    top_specs = [top_specs[idx] for idx in np.random.permutation(len(top_specs))]
    bottom_specs = [bottom_specs[idx] for idx in np.random.permutation(len(bottom_specs))]

    top_used = sum(int(spec["w"]) for spec in top_specs) + 2 * (len(top_specs) - 1)
    bottom_used = sum(int(spec["w"]) for spec in bottom_specs) + 2 * (len(bottom_specs) - 1)
    long_dim = int(2 + max(top_used, bottom_used) + np.random.randint(2, 5))

    grid = np.full((height, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    colors = np.random.choice(DOMINANT_COLORS, size=4, replace=False)
    color_idx = 0
    rects: list[dict] = []
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[0, :] = True
    occupied[:, 0] = True

    for row, specs in ((TOP_ROW, top_specs), (bottom_row, bottom_specs)):
        c = 2
        for spec in specs:
            dominant = int(colors[color_idx])
            color_idx += 1
            patch = _make_patch(int(spec["h"]), int(spec["w"]), dominant, bool(spec["has_minority"]))
            paste(grid, patch, int(row), int(c))
            box = (int(row), int(c), int(row + spec["h"] - 1), int(c + spec["w"] - 1))
            rects.append(
                {
                    "patch": patch.copy(),
                    "box": box,
                    "area": int(spec["h"] * spec["w"]),
                    "has_minority": bool(spec["has_minority"]),
                }
            )
            occupied[int(row) : int(row + spec["h"]), int(c) : int(c + spec["w"])] = True
            c += int(spec["w"]) + 2

    noise_count = _add_noise(grid, occupied)
    return grid, rects, int(noise_count)


def _gen_attempt() -> tuple[np.ndarray, dict[str, int | bool]] | None:
    canonical, rects, noise_count = _build_canonical()
    if noise_count < 1:
        return None

    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    inp = _transform(canonical, blue_orient, corner)

    transformed_rects = []
    for rect in rects:
        input_patch, input_box = _render_single_rect(rect["patch"], rect["box"], canonical.shape, blue_orient, corner)
        transformed_rects.append(
            {
                "input_patch": input_patch,
                "box": input_box,
                "area": int(rect["area"]),
                "has_minority": bool(rect["has_minority"]),
            }
        )

    cr_bg, cr, cc, blue_inp_orient, red_inp_orient, analyzed_rects, rejected_noise = _analyze_grid(inp)
    del cr_bg
    if rejected_noise < 1 or len(analyzed_rects) != 4:
        return None

    ordered = _sorted_rectangles(list(transformed_rects), cr, cc, blue_inp_orient, red_inp_orient)
    expected = _stack_expected(ordered, blue_inp_orient)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    metrics = _activity_metrics(inp)
    if metrics["minority_count"] != 2 or metrics["pure_count"] != 2:
        return None
    if not metrics["diff_vs_distance"]:
        return None
    if not metrics["diff_vs_priority_distance"]:
        return None
    if not metrics["diff_vs_area_only"]:
        return None
    return inp, metrics


def generate_input() -> np.ndarray:
    for _ in range(600):
        built = _gen_attempt()
        if built is not None:
            inp, _metrics = built
            return inp
    raise RuntimeError("failed to generate input for 291dc1e1 combo 33")
