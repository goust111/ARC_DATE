# Combo variant of 2ba387bc (index 36)
# combo_spec: D1/2+D2/2+D3/0
# Name: 面积分类+距离排序+可变尺寸

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVERS_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the combo variant D1/2 + D2/2 + D3/0

a) Concrete behavior and trace
   - Input objects are still isolated monochrome squares on a plain background, but
     their size is no longer fixed: each object is a 3x3, 4x4, or 5x5 square.
   - There are still exactly two semantic kinds:
     1. frame: a 1-pixel border of one color with a background-filled interior.
     2. solid: a fully filled square of one color.
   - Original task:
     * accepts only 4x4 objects,
     * classifies frame vs solid by checking the 2x2 interior,
     * sorts each group by (row, col),
     * stacks into fixed 4-row bands and fixed 4+4 output width.
   - Combo task:
     * accepts any square with side n in {3,4,5},
     * classifies by non-background area:
         frame area = n^2 - (n-2)^2 = 4n-4,
         solid area = n^2,
     * sorts each group by distance from object center to grid center,
     * outputs two adaptive columns of width max_size, with band heights equal to the
       larger of the paired frame/solid sizes in that output row.
   - Small trace:
     suppose a 3x3 frame at (2,2), a 5x5 frame at (12,10), a 4x4 solid at (5,16),
     and a 3x3 solid at (15,4) in a 24x24 grid.
     * Original code would ignore the 3x3 and 5x5 objects entirely because it hard-codes
       4x4, then sort the remaining 4x4 solid by row and emit a fixed 8-wide output.
     * Combo code keeps all four squares. The 3x3 frame has area 8, the 5x5 frame area
       16, the 4x4 solid area 16, and the 3x3 solid area 9. The two frames are grouped
       together because each matches the size-dependent border area, even though their
       areas differ. Inside each group, the object whose center is closer to the whole-grid
       center is stacked first. Output row starts are cumulative band heights rather than
       multiples of 4, and each column uses width 5 because the largest object is 5x5.

b) Complexity calibration
   - Sizes are limited to three values {3,4,5}. This is enough to make "area grouping"
     genuinely size-adaptive, while keeping the rule visually learnable.
   - Generator requires both groups to have at least two objects, at least one non-4
     frame and one non-4 solid, and at least two distinct sizes overall. This prevents
     collapse back to the original fixed-size regime.
   - Generator also requires at least one group's distance ordering to differ from simple
     (row, col) ordering, so D2 is not visually degenerate.

c) Hidden-assumption audit of the original main()
   - `bbox_h(b) != 4 or bbox_w(b) != 4`: no longer valid. We must accept square bboxes
     of size 3..5 and reject only non-squares / other sizes.
   - `sub[1:3, 1:3] == bg`: no longer valid. D1 says classification must use counted
     non-background area rather than a fixed interior slice.
   - `frames.sort(key=(row,col))` / `solids.sort(key=(row,col))`: no longer valid.
     D2 replaces this with center-distance sorting.
   - `outH = 4 * max(...)`, `outW = 8`, and `paste(..., i*4, ...)`: no longer valid.
     D3 makes both band heights and column widths depend on the realized object sizes.
   - New tie-related assumption from D2: if two objects in the same group are exactly
     equidistant from grid center, pure distance ordering is ambiguous. Solver therefore
     uses `(dist2, row, col, size)` for determinism; generator avoids equal distances
     within each group so the primary rule stays human-visible.
   - New classification assumption from D1+D3: an accepted object must be a monochrome
     square whose foreground area is either `4n-4` or `n^2`. Generator constructs only
     such objects.

d) Placement legality
   - Generator uses `random_free_location(..., border=1, padding=1, padding_connectivity=4)`.
     This guarantees objects fit inside the canvas, do not overlap, and do not touch
     orthogonally, so `find_components(... connectivity=4, monochromatic=False)` recovers
     each square as exactly one component.
   - Colors are unique per object, keeping components visually separate and preventing
     accidental merges through same-colored contact.

e) Visual observability
   - All control variables are visible in the pixels:
     * size is visible from the bbox,
     * area-based frame/solid grouping is inferable because frames are hollow and solids
       are filled, but the solver specifically has to adapt the counted area to size,
     * distance-to-center ordering is a visible spatial relation.
   - The generator avoids samples where distance ordering accidentally matches old row
     ordering for both groups, because such samples would hide the D2 change.

f) Visual cleanliness
   - No extra markers are added; objects remain isolated monochrome squares on empty
     background. This keeps the scene readable despite combining three dimensions.
   - Output still uses exactly two columns, so the only added visual complexity comes
     from object-size variation and ordering, not from clutter.

g) Change inferability
   - No hidden use of raw color ids occurs. Colors only distinguish objects.
   - Because both frame and solid groups include non-4 examples across the dataset,
     the observed frame areas are not a single constant 12 and the observed solid areas
     are not a single constant 16. This makes the size-adaptive area rule materially
     different from the original fixed-size heuristic.

h) Main's new assumptions and generator guarantees
   - Solver assumptions:
     * every relevant object is an isolated monochrome square of size 3..5;
     * its foreground area is either `4n-4` (frame) or `n^2` (solid);
     * same-group center distances are preferably distinct for visual clarity.
   - Generator guarantees these by construction and rejects attempts unless:
     * both groups have >= 2 objects,
     * each group has unique center distances,
     * at least one group is reordered by the distance rule,
     * at least one frame and one solid have size != 4,
     * solver output matches the generator-side assembly exactly.
"""


def _make_frame(size: int, col: int, bg: int = 0) -> np.ndarray:
    a = np.full((int(size), int(size)), int(col), dtype=int)
    a[1:-1, 1:-1] = int(bg)
    return a


def _make_solid(size: int, col: int) -> np.ndarray:
    return np.full((int(size), int(size)), int(col), dtype=int)


def _distance2_to_center(r0: int, c0: int, size: int, H: int, W: int) -> int:
    """
    Squared Euclidean distance on doubled coordinates.

    Object center:
      (r0 + (size-1)/2, c0 + (size-1)/2)
    Doubled:
      (2*r0 + size - 1, 2*c0 + size - 1)

    Grid center doubled:
      (H - 1, W - 1)
    """
    dr2 = int(2 * int(r0) + int(size) - 1 - (int(H) - 1))
    dc2 = int(2 * int(c0) + int(size) - 1 - (int(W) - 1))
    return int(dr2 * dr2 + dc2 * dc2)


def _foreground_area(sub: np.ndarray, bg: int) -> int:
    return int(np.count_nonzero(sub != int(bg)))


def _uniform_non_bg_color(sub: np.ndarray, bg: int) -> bool:
    vals = {int(v) for v in sub.flatten().tolist() if int(v) != int(bg)}
    return len(vals) == 1


def _classify_square(sub: np.ndarray, bg: int) -> str | None:
    h, w = map(int, sub.shape)
    if h != w or h < 3 or h > 5:
        return None
    if not _uniform_non_bg_color(sub, bg):
        return None

    area = _foreground_area(sub, bg)
    solid_area = int(h * h)
    frame_area = int(solid_area - max(h - 2, 0) * max(h - 2, 0))

    if area == frame_area:
        return "frame"
    if area == solid_area:
        return "solid"
    return None


def _collect_objects(g: np.ndarray) -> tuple[int, list[dict]]:
    bg = int(get_bg(g))
    H, W = map(int, g.shape)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    objects: list[dict] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        if h != w or h < 3 or h > 5:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        sub = crop_bbox(g, b)
        kind = _classify_square(sub, bg)
        if kind is None:
            continue
        objects.append(
            {
                "kind": kind,
                "size": int(h),
                "r0": int(r0),
                "c0": int(c0),
                "area": _foreground_area(sub, bg),
                "dist2": _distance2_to_center(int(r0), int(c0), int(h), int(H), int(W)),
                "sprite": sub.copy(),
            }
        )
    return int(bg), objects


def _distance_sorted(group: list[dict]) -> list[dict]:
    return sorted(group, key=lambda obj: (int(obj["dist2"]), int(obj["r0"]), int(obj["c0"]), int(obj["size"])))


def _position_signature(group: list[dict]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(group, key=lambda obj: (int(obj["r0"]), int(obj["c0"]), int(obj["size"])))
    return tuple((int(obj["r0"]), int(obj["c0"])) for obj in ordered)


def _distance_signature(group: list[dict]) -> tuple[tuple[int, int], ...]:
    ordered = _distance_sorted(group)
    return tuple((int(obj["r0"]), int(obj["c0"])) for obj in ordered)


def _assemble_output(objects: list[dict], bg: int) -> np.ndarray:
    assert objects
    frames = _distance_sorted([obj for obj in objects if obj["kind"] == "frame"])
    solids = _distance_sorted([obj for obj in objects if obj["kind"] == "solid"])

    max_size = max(int(obj["size"]) for obj in objects)
    row_count = max(len(frames), len(solids))
    row_heights = [
        max(
            int(frames[i]["size"]) if i < len(frames) else 0,
            int(solids[i]["size"]) if i < len(solids) else 0,
        )
        for i in range(row_count)
    ]

    out = np.full((int(sum(row_heights)), int(2 * max_size)), int(bg), dtype=int)
    r = 0
    for i, band_h in enumerate(row_heights):
        if i < len(frames):
            paste(out, frames[i]["sprite"], int(r), 0)
        if i < len(solids):
            paste(out, solids[i]["sprite"], int(r), int(max_size))
        r += int(band_h)
    return out


def _fusion_metrics_from_objects(objects: list[dict]) -> dict:
    frames = [obj for obj in objects if obj["kind"] == "frame"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    groups = [frames, solids]
    reordered_groups = sum(int(_position_signature(group) != _distance_signature(group)) for group in groups)
    return {
        "frame_count": int(len(frames)),
        "solid_count": int(len(solids)),
        "size_set": tuple(sorted({int(obj["size"]) for obj in objects})),
        "frame_non4": int(sum(int(obj["size"]) != 4 for obj in frames)),
        "solid_non4": int(sum(int(obj["size"]) != 4 for obj in solids)),
        "frame_areas": tuple(sorted({int(obj["area"]) for obj in frames})),
        "solid_areas": tuple(sorted({int(obj["area"]) for obj in solids})),
        "reordered_groups": int(reordered_groups),
        "frame_unique_dist": int(len({int(obj["dist2"]) for obj in frames}) == len(frames)),
        "solid_unique_dist": int(len({int(obj["dist2"]) for obj in solids}) == len(solids)),
    }


def _variant_visible(objects: list[dict]) -> bool:
    if not objects:
        return False

    metrics = _fusion_metrics_from_objects(objects)
    if metrics["frame_count"] < 2 or metrics["solid_count"] < 2:
        return False
    if not metrics["frame_unique_dist"] or not metrics["solid_unique_dist"]:
        return False
    if metrics["reordered_groups"] < 1:
        return False
    if metrics["frame_non4"] < 1 or metrics["solid_non4"] < 1:
        return False
    if len(metrics["size_set"]) < 2:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered variable-size squares into two columns.

    Combo rule:
    - Accept 3x3 to 5x5 monochrome square components.
    - Classify them by counted non-background area:
      size n with area 4n-4 -> frame, area n^2 -> solid.
    - Sort each group by distance from object center to grid center.
    - Output uses width = 2 * max_size and adaptive row-band heights.
    """
    g = np.array(grid, dtype=int)
    bg, objects = _collect_objects(g)
    assert objects
    return _assemble_output(objects, bg)


def _sample_kind_sizes(n: int) -> list[int]:
    sizes = [int(np.random.randint(3, 6)) for _ in range(int(n))]
    if all(size == 4 for size in sizes):
        sizes[int(np.random.randint(0, n))] = int(np.random.choice([3, 5]))
    return sizes


def _sample_specs() -> list[dict]:
    n_frames = int(np.random.randint(2, 5))
    n_solids = int(np.random.randint(2, 5))
    frame_sizes = _sample_kind_sizes(n_frames)
    solid_sizes = _sample_kind_sizes(n_solids)
    all_sizes = frame_sizes + solid_sizes
    if len(set(all_sizes)) < 2:
        all_sizes[0] = 3
        all_sizes[-1] = 5
        frame_sizes = all_sizes[:n_frames]
        solid_sizes = all_sizes[n_frames:]

    colors = list(np.random.permutation(np.arange(1, 10))[: n_frames + n_solids].astype(int))
    specs = []
    for i in range(n_frames):
        specs.append({"kind": "frame", "size": int(frame_sizes[i]), "color": int(colors[i])})
    for i in range(n_solids):
        specs.append({"kind": "solid", "size": int(solid_sizes[i]), "color": int(colors[n_frames + i])})
    np.random.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(28, 39))
    W = int(np.random.randint(28, 39))
    g = np.full((H, W), int(bg), dtype=int)

    specs = _sample_specs()
    objects: list[dict] = []
    try:
        for spec in specs:
            size = int(spec["size"])
            color = int(spec["color"])
            sprite = _make_frame(size, color, bg) if spec["kind"] == "frame" else _make_solid(size, color)
            r0, c0 = random_free_location(
                g,
                sprite,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, sprite, int(r0), int(c0), transparent=bg)
            objects.append(
                {
                    "kind": spec["kind"],
                    "size": int(size),
                    "r0": int(r0),
                    "c0": int(c0),
                    "area": _foreground_area(sprite, bg),
                    "dist2": _distance2_to_center(int(r0), int(c0), int(size), int(H), int(W)),
                    "sprite": sprite.copy(),
                }
            )
    except Exception:
        return None

    if not _variant_visible(objects):
        return None

    expected = _assemble_output(objects, bg)
    actual = main(g)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(1000):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 1000 attempts")
