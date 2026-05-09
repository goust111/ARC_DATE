# Variant of 2ba387bc
# Dimension: D3 — 源对象的形状与尺寸 (src)
# Change: 所有对象均为固定的4x4正方形，框架为1像素宽边框+2x2背景内部，实心为全色填充... → 可变尺寸的方块（3x3到5x5）

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVERS_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the D3 variant
==================================

1. Concrete behavior
   - Input objects are still isolated monochrome squares on a plain background.
   - There are only two semantic types:
     * frame: an n x n square with a 1-pixel-thick border of one color and a fully
       background-filled (n-2) x (n-2) interior.
     * solid: an n x n square fully filled with one non-background color.
   - The changed degree of freedom is n in {3, 4, 5}. The original solver hard-coded
     n = 4 and therefore also hard-coded:
     * bbox height/width == 4
     * interior slice == sub[1:3, 1:3]
     * output band height == 4
     * output width == 8
   - In the variant, classification is unchanged, but layout must now depend on size.
     We keep the same semantic grouping and same top-to-bottom ordering within each
     group, then place frames in the left column and solids in the right column.

   Small trace:
   - Original:
     frame 4x4 at input row 2, solid 4x4 at input row 7 -> output uses a 4-row band.
   - Variant:
     frame 3x3 at input row 2, solid 5x5 at input row 7 -> both are the first item of
     their group, so they share the first output band, but that band must be 5 rows high.
     The frame is pasted at the top-left of its 5-row band, leaving background below it.
   - Difference:
     output row starts are no longer simple multiples of 4; they are cumulative maxima
     of paired object heights.

2. Complexity calibration
   - Allowed sizes are restricted to 3, 4, 5 only. This keeps the rule inferable from
     few examples while still making the spatial change visible.
   - Generator forces at least one non-4 object and at least two distinct sizes overall,
     so samples do not silently collapse back to the original fixed-size regime.

3. Hidden-assumption audit of the original main()
   - `if bbox_h(b) != 4 or bbox_w(b) != 4`: invalid now; must accept any square in 3..5.
   - `sub[1:3, 1:3] == bg`: invalid now; interior size depends on n, so it becomes
     `sub[1:-1, 1:-1] == bg`.
   - `outH = 4 * max(...)`: invalid now; row heights vary per paired band.
   - `outW = 8`: invalid now; each output column must be as wide as the largest square.
   - `paste(..., i * 4, ...)`: invalid now; row offsets are cumulative band heights.

4. Placement legality
   - Generator uses `random_free_location(..., padding=1, connectivity=4)` so objects
     neither overlap nor touch orthogonally.
   - Every generated object is a true 3..5 square, so the solver's square-bbox test
     matches the generator's construction exactly.

5. Visual observability
   - The changed behavior is visible without relying on hidden color semantics:
     different object sizes are directly observable in both input and output.
   - Output rows with mixed sizes show background under the smaller object inside a
     shared band, making the variable-height packing rule readable from examples.

6. Visual cleanliness
   - No extra markers or overlays are introduced.
   - Colors remain unique per object, which keeps component boundaries obvious.

7. New solver assumptions after modification
   - Every relevant object is an isolated square component with side length in 3..5.
   - Frames have a uniform border and a fully background interior.
   - Solids are uniformly colored.
   - The generator explicitly constructs only such objects and verifies that the solver
     output matches the ground-truth output assembled from those same objects.
"""


def _make_frame(size: int, col: int, bg: int = 0) -> np.ndarray:
    a = np.full((size, size), int(col), dtype=int)
    a[1:-1, 1:-1] = int(bg)
    return a


def _make_solid(size: int, col: int) -> np.ndarray:
    return np.full((size, size), int(col), dtype=int)


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    h, w = map(int, sub.shape)
    if h != w or h < 3 or h > 5:
        return False
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_cols = set(map(int, border.tolist()))
    if len(border_cols) != 1:
        return False
    if int(next(iter(border_cols))) == int(bg):
        return False
    return bool((sub[1:-1, 1:-1] == int(bg)).all())


def _is_solid(sub: np.ndarray, bg: int) -> bool:
    h, w = map(int, sub.shape)
    if h != w or h < 3 or h > 5:
        return False
    vals = set(map(int, sub.flatten().tolist()))
    return len(vals) == 1 and int(next(iter(vals))) != int(bg)


def _assemble_output(objects: list[dict], bg: int) -> np.ndarray:
    frames = [obj for obj in objects if obj["kind"] == "frame"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    frames.sort(key=lambda obj: (int(obj["r0"]), int(obj["c0"])))
    solids.sort(key=lambda obj: (int(obj["r0"]), int(obj["c0"])))

    if not frames and not solids:
        return np.zeros((0, 0), dtype=int)

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered variable-size squares into two columns by type.

    Variant rule:
    - Objects are 3x3 to 5x5 squares.
    - Frames have a 1-pixel border and bg interior; solids are fully filled.
    - Sort frames and solids independently by input (row, col).
    - Stack frames in the left output column and solids in the right output column.
    - Both output columns use width = max object size in the whole input.
    - Output row bands are paired by index; each band's height is the larger size of the
      two objects placed in that band.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    objects: list[dict] = []

    for comp in comps:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        if h != w or h < 3 or h > 5:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        if _is_frame(sub, bg):
            kind = "frame"
        elif _is_solid(sub, bg):
            kind = "solid"
        else:
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
                    "size": size,
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
