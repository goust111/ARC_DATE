# Combo variant of 2ba387bc (index 4)
# combo_spec: D1/2+D3/0
# Name: 面积分类+可变尺寸方块

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVERS_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D1/2 + D3/0
=====================================

a) Concrete behavior and trace
   - Input still contains isolated monochrome square objects on one background.
   - Objects now have side length n in {3, 4, 5}.
   - Two semantic types remain:
     * frame: exactly one colored border ring around a bg interior.
       Its non-bg area is the perimeter pixel count `4*n - 4`.
     * solid: fully filled square.
       Its non-bg area is the total area `n*n`.
   - Output grouping is unchanged semantically: frames go to the left column,
     solids to the right column, and each group is sorted by input `(row, col)`.
   - Layout is no longer fixed at 4 rows per object. The output uses:
     * column width = maximum square size seen in the input
     * one shared row band per index across the two groups
     * band height = max(size of left object, size of right object)

   Small trace:
   - Suppose the input contains:
     * a 3x3 frame at row 2 -> area 8, so it is a frame
     * a 5x5 solid at row 9 -> area 25, so it is a solid
     * a 4x4 frame at row 14 -> area 12, so it is a frame
   - Original solver:
     * rejects 3x3 and 5x5 because it hard-codes `4x4`
     * classifies 4x4 by checking the 2x2 interior
     * uses fixed 4-row bands and output width 8
   - Combo solver:
     * accepts all three because each bbox is a square in 3..5
     * computes `area = count(sub != bg)`
     * uses `area == 4*n-4` to label frames and `area == n*n` to label solids
     * sorts frames as [3x3, 4x4], solids as [5x5]
     * uses `max_size = 5`, so output width is 10
     * first band height is `max(3, 5) = 5`; second band height is `max(4, 0) = 4`
   - Difference:
     both the classification threshold and the packing geometry now depend on size.

b) Complexity calibration
   - Size freedom is limited to three observable cases only: 3, 4, 5.
     That is enough to make the rule visibly different without making inference noisy.
   - Generator enforces at least one non-4 object and at least two distinct sizes overall,
     so samples do not collapse back to the original fixed-size task.
   - Both groups are always present, which keeps the left/right column rule visible.

c) Hidden-assumption audit of the original main()
   - `bbox_h(b) != 4 or bbox_w(b) != 4`:
     invalid now; objects must instead be accepted for any square size in 3..5.
   - `sub[1:3, 1:3] == bg`:
     invalid now because the interior slice changes with size; the combo variant also
     changes the semantic cue entirely from interior inspection to area counting.
   - Treating every non-frame 4x4 as solid:
     invalid now because the solver must distinguish using size-aware area thresholds.
   - `outH = 4 * max(...)`, `outW = 8`, and row starts at `i * 4`:
     invalid now because output geometry depends on the largest size and on per-band
     maxima between the paired frame/solid objects.

d) Placement legality
   - Generator uses `random_free_location(..., padding=1, connectivity=4)` for each
     square, so objects do not overlap, do not touch orthogonally, and stay in bounds.
   - Every generated object is either a proper square ring or a filled square, so each
     connected component has a square bounding box compatible with the solver logic.

e) Visual observability
   - A human can see size variation directly from the input because squares visibly vary
     among 3x3, 4x4, and 5x5.
   - A human can also infer the new classification cue from examples: for each size,
     hollow squares have visibly fewer colored cells than full squares, and the needed
     threshold changes with size rather than being tied to a fixed 4x4 center test.
   - Output column width and shared band heights expose the size dependence again in the
     packed result, especially when a 3x3 object shares a band with a 5x5 object.

f) Visual cleanliness
   - Colors remain unique per object, which prevents merged same-color ambiguities.
   - Padding keeps components separated, so no frame overlaps or touching artifacts
     obscure the intended object boundaries.
   - No extra markers are added; the scene is still a plain scatter-and-sort puzzle.

g) Inferability of the change
   - There is no hidden use of absolute color values or non-visual metadata.
   - The critical variable is size, and the generator guarantees repeated support:
     every sample has at least two sizes and at least one size outside 4.
   - Because the original task never contained non-4 objects, these samples visibly force
     the solver to infer a size-aware rule instead of memorizing a 4x4 template.

h) New solver assumptions after modification
   - Every relevant component is an isolated square of size 3..5.
   - Each such square is monochrome in its non-bg cells.
   - Its non-bg area is either `4*n - 4` (frame) or `n*n` (solid).
   - The generator explicitly constructs only those cases and rejects samples that fail
     a round-trip check with `main()`.
"""


def _make_frame(size: int, col: int, bg: int = 0) -> np.ndarray:
    a = np.full((size, size), int(col), dtype=int)
    a[1:-1, 1:-1] = int(bg)
    return a


def _make_solid(size: int, col: int) -> np.ndarray:
    return np.full((size, size), int(col), dtype=int)


def _classify_by_area(sub: np.ndarray, bg: int) -> str | None:
    h, w = map(int, sub.shape)
    if h != w or h < 3 or h > 5:
        return None

    non_bg = sub[sub != int(bg)]
    if non_bg.size == 0:
        return None
    if len(set(map(int, non_bg.tolist()))) != 1:
        return None

    size = int(h)
    area = int(non_bg.size)
    if area == int(size * size):
        return "solid"
    if area == int(4 * size - 4):
        return "frame"
    return None


def _assemble_output(objects: list[dict], bg: int) -> np.ndarray:
    frames = [obj for obj in objects if obj["kind"] == "frame"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    frames.sort(key=lambda obj: (int(obj["r0"]), int(obj["c0"])))
    solids.sort(key=lambda obj: (int(obj["r0"]), int(obj["c0"])))

    if not frames and not solids:
        return np.zeros((0, 0), dtype=int)

    max_size = max(int(obj["size"]) for obj in objects)
    band_count = max(len(frames), len(solids))
    band_heights = [
        max(
            int(frames[i]["size"]) if i < len(frames) else 0,
            int(solids[i]["size"]) if i < len(solids) else 0,
        )
        for i in range(band_count)
    ]

    out = np.full((int(sum(band_heights)), int(2 * max_size)), int(bg), dtype=int)
    r = 0
    for i, band_h in enumerate(band_heights):
        if i < len(frames):
            paste(out, frames[i]["sprite"], int(r), 0)
        if i < len(solids):
            paste(out, solids[i]["sprite"], int(r), int(max_size))
        r += int(band_h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered variable-size squares into two columns by size-aware area class.

    - Accept only square components of size 3..5.
    - Count non-background pixels inside each square.
    - `area == 4*n-4` -> frame, `area == n*n` -> solid.
    - Sort each group by input `(row, col)`.
    - Stack frames on the left and solids on the right.
    - Each column has width `max_size`, and each shared row band has height equal to the
      larger of the two objects placed in that band.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

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
        kind = _classify_by_area(sub, bg)
        if kind is None:
            continue
        objects.append(
            {
                "kind": kind,
                "size": int(h),
                "r0": int(r0),
                "c0": int(c0),
                "sprite": sub,
            }
        )

    return _assemble_output(objects, bg)


def _sample_sizes(total: int) -> list[int]:
    while True:
        sizes = [int(np.random.randint(3, 6)) for _ in range(total)]
        if any(size != 4 for size in sizes) and len(set(sizes)) >= 2:
            return sizes


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(24, 37))
    W = int(np.random.randint(24, 37))
    g = np.full((H, W), bg, dtype=int)

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    total = n_frames + n_solids
    sizes = _sample_sizes(total)
    colors = list(np.random.permutation(np.arange(1, 10))[:total].astype(int))

    frame_specs = [{"kind": "frame", "size": sizes[i], "color": colors[i]} for i in range(n_frames)]
    solid_specs = [
        {"kind": "solid", "size": sizes[n_frames + i], "color": colors[n_frames + i]} for i in range(n_solids)
    ]
    specs = frame_specs + solid_specs
    np.random.shuffle(specs)

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
                    "sprite": sprite,
                }
            )
    except Exception:
        return None

    frames = [obj for obj in objects if obj["kind"] == "frame"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    frame_rows = [int(obj["r0"]) for obj in frames]
    solid_rows = [int(obj["r0"]) for obj in solids]
    if len(frame_rows) != len(set(frame_rows)) or len(solid_rows) != len(set(solid_rows)):
        return None

    expected = _assemble_output(objects, bg)
    actual = main(g)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
