# Combo variant of 2ba387bc (index 20)
# combo_spec: D6/3+D9/1
# Name: 正方形输出+极不对称分组
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location
except ModuleNotFoundError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D6/3 + D9/1

a) Concrete behavior and trace
- Original task:
  1. detect every scattered 4x4 object on the dominant background,
  2. classify it as frame if the 2x2 center is background and the outer border is one color,
  3. otherwise classify it as solid if the whole 4x4 block is one non-background color,
  4. sort each type by input row,
  5. stack frames in cols 0..3 and solids in cols 4..7,
  6. output shape = (4 * max(n_frames, n_solids), 8).
- This combo keeps steps 1..5 but changes two coupled parts:
  1. object counts become extremely asymmetric: one group has exactly 1 object and the other has 4 or 5;
  2. after building the original 8-wide stacked arrangement, the final canvas is expanded to a square
     with side length `max(base_height, 8)`, padding the unused area with background.
- Example trace:
  suppose input has 1 frame and 4 solids, all in different colors, with solid top rows increasing.
  Original output would be a 16x8 grid:
    left column: 1 frame in rows 0..3, rows 4..15 left as background
    right column: 4 solids stacked through rows 0..15
  Combo output becomes 16x16:
    the same 16x8 arrangement occupies the left half,
    cols 8..15 are pure background.
  If counts are reversed (4 frames, 1 solid), the square is still 16x16, but now the right object column
  is short and the left column is tall.

b) Complexity calibration
- Making the output square without changing counts would be too mild: for balanced groups, the only
  difference is extra padding on the right.
- Making counts asymmetric without changing geometry would still leave the familiar 8-wide rectangle.
- Combining them makes the change visually obvious in every sample:
  the tall column forces height 16 or 20, which then forces a 16x16 or 20x20 square output.
- Counts are restricted to only four cases: (1,4), (1,5), (4,1), (5,1). This keeps the rule learnable.
- Colors remain unique per object, so the only reasoning burden is shape grouping + square-size inference.

c) Hidden-assumption review of main()
- `bg = get_bg(g)`: still valid because the generator keeps a large empty canvas; background remains dominant.
- `find_components(..., monochromatic=False)`: still valid, but it now relies on a stronger generator guarantee:
  objects must not touch under 4-connectivity, even if they are the same shape class. The generator enforces
  this with padded placement.
- `bbox_h(b) != 4 or bbox_w(b) != 4`: still valid because every relevant object remains exactly 4x4.
- Frame detection from a monochrome border plus bg-filled center: unchanged and still the intended cue.
- Solid detection must now be explicit rather than `else -> solid`, so malformed components are ignored.
- Sorting by `(row, col)` introduces a tie-break assumption. To keep the rule human-visible as "top to bottom",
  the generator rejects samples where two frames share a top row or two solids share a top row.
- The original implicit width assumption `outW = 8` no longer determines the final shape by itself.
  The new solver must first compute the original stacked rectangle, then square-pad it.

d) Placement legality
- Every object is placed with `random_free_location(..., border=1, padding=1, padding_connectivity=4)`.
- This guarantees no overlap, no out-of-bounds writes, and at least one background cell of 4-connected
  separation between objects, so component extraction stays correct.
- Frames are pasted with `transparent=bg`, preserving their hollow centers.

e) Visual observability
- Humans can still see the same two shape types as in the source task.
- The new behavior is obvious from 2-3 examples because:
  1. one output column is always much longer than the other,
  2. the canvas is always square, even though only the leftmost 8 columns contain objects,
  3. the amount of empty padding changes with whether the tall column has 4 or 5 objects.
- No rule depends on numeric color ids, parity, or any hidden metadata.

f) Visual cleanliness
- Input remains a sparse scatter of clean 4x4 objects on blank background.
- Output remains a simple two-column stack, only embedded in a larger square canvas.
- The extra area is uniform background rather than new markings, so it is readable instead of cluttered.

g) Change inferability
- There is no color remapping at all.
- The new inferable variables are only:
  1. which objects belong to the frame column vs solid column,
  2. how many objects are in the taller group,
  3. that the final side length is the max of original width 8 and stacked height.
- Because the generator always produces a much taller-than-wide base rectangle, the square rule is visible
  as "append background columns until width matches height".

Post-test behavior metric
- For each generated sample we can measure:
  `minority_count = min(n_frames, n_solids)`,
  `majority_count = max(n_frames, n_solids)`,
  `side = main(inp).shape[0]`,
  `padding_cols = side - 8`.
- A non-degenerate combo sample must satisfy:
  `minority_count == 1`,
  `majority_count in {4, 5}`,
  `main(inp).shape[0] == main(inp).shape[1]`,
  `padding_cols in {8, 12}`.
- Comparing against the original solver on the same input should always differ in shape:
  original outputs 16x8 or 20x8, while this combo outputs 16x16 or 20x20.

New implicit assumptions after modification
- `main()` assumes every counted component is an exact 4x4 frame or exact 4x4 solid; the generator guarantees it.
- `main()` assumes within each type there is a strict visual top-to-bottom order; the generator guarantees distinct
  top rows per type.
- `main()` assumes the square canvas can be formed by padding the original stacked rectangle with background only;
  this is guaranteed because the combo does not alter the relative placement of the two output columns.
"""


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_colors = set(map(int, border.flatten()))
    return len(border_colors) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _is_solid(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    vals = set(map(int, sub.flatten()))
    return len(vals) == 1 and int(next(iter(vals))) != int(bg)


def _collect_objects(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        if _is_frame(sub, bg):
            frames.append((int(r0), int(c0), sub))
        elif _is_solid(sub, bg):
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids


def _build_square_output(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    base_h = 4 * max(int(len(frames)), int(len(solids)))
    base_w = 8
    side = max(int(base_h), int(base_w))
    out = np.full((side, side), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares by type as in the source task, then square-pad the result.

    - Frames stack in the left 4 columns.
    - Solids stack in the next 4 columns.
    - The underlying stacked rectangle is the original 8-wide layout.
    - Final output is a background-filled square with side length = max(8, stacked_height).
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)
    return _build_square_output(frames, solids, bg)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _sample_counts() -> tuple[int, int]:
    major = int(np.random.choice([4, 5]))
    if bool(np.random.randint(0, 2)):
        return 1, major
    return major, 1


def _top_rows_are_distinct(objects: list[tuple[int, int, np.ndarray]]) -> bool:
    rows = [int(r0) for r0, _c0, _sub in objects]
    return len(rows) == len(set(rows))


def _analyze_input(g: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)
    out = _build_square_output(frames, solids, bg)
    return {
        "n_frames": int(len(frames)),
        "n_solids": int(len(solids)),
        "minority_count": int(min(len(frames), len(solids))),
        "majority_count": int(max(len(frames), len(solids))),
        "square_side": int(out.shape[0]),
        "padding_cols": int(out.shape[1] - 8),
        "frame_rows_distinct": int(_top_rows_are_distinct(frames)),
        "solid_rows_distinct": int(_top_rows_are_distinct(solids)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(24, 33))
    w = int(np.random.randint(24, 33))
    g = np.full((h, w), bg, dtype=int)

    n_frames, n_solids = _sample_counts()
    total = int(n_frames + n_solids)
    colors = list(map(int, np.random.choice(np.arange(1, 10), size=total, replace=False)))

    objects: list[np.ndarray] = []
    for col in colors[:n_frames]:
        objects.append(_make_frame(col, bg))
    for col in colors[n_frames:]:
        objects.append(_make_solid(col))
    np.random.shuffle(objects)

    try:
        for obj in objects:
            r0, c0 = random_free_location(
                g,
                obj,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, obj, int(r0), int(c0), transparent=bg)
    except Exception:
        return None

    info = _analyze_input(g)
    if info["n_frames"] != n_frames or info["n_solids"] != n_solids:
        return None
    if info["minority_count"] != 1 or info["majority_count"] not in {4, 5}:
        return None
    if info["frame_rows_distinct"] != 1 or info["solid_rows_distinct"] != 1:
        return None
    if info["square_side"] not in {16, 20}:
        return None
    if info["padding_cols"] not in {8, 12}:
        return None

    main(g)
    return g


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
