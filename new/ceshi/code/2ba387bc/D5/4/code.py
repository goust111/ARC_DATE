# Variant of 2ba387bc
# Dimension: D5 — 目标布局——两列排列方式 (dst)
# Change: 框架放入左列（cols 0-3），实心放入右列（cols 4-7），两列紧邻无间隙... → 交错排列——框架和实心在同一列中交替放置

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the D5 variant

a) Concrete behavior and traced example
   - Input stays the same: several separated 4x4 objects on the background.
     Frames have a 1-pixel colored border and a bg-colored 2x2 center; solids are
     fully filled 4x4 monochrome blocks.
   - Original output had two adjacent columns: frames stacked in cols 0-3 and
     solids stacked in cols 4-7.
   - Variant output has width 4 only. We still sort frames by their input
     (row, col), and solids by their input (row, col), but we no longer place the
     two groups side-by-side. Instead we interleave their sorted lists in a single
     column, always starting with the first frame: F1, S1, F2, S2, ... then any
     remainder of the longer list.
   - Example:
       frames at rows [2, 12], solids at rows [5, 18, 22]
       original: left column = [F1, F2], right column = [S1, S2, S3]
       variant:  [F1, S1, F2, S2, S3] in one 4-wide column
     The change only affects the destination layout step; detection and
     classification remain unchanged.

b) Complexity calibration
   - The variant must be visibly different from the seed task, so output width is
     reduced from 8 to 4 and both types must appear multiple times.
   - To avoid ambiguous training examples, the generator only uses 2-4 frames and
     2-4 solids. This guarantees a real alternation pattern instead of trivial
     one-object cases.
   - The generator rejects placements where the global top-to-bottom order of all
     objects already matches the interleaved output order, because those samples
     would make "alternate by type" hard to distinguish from "just sort all
     objects globally by row".

c) Implicit-assumption audit of the original main()
   - `find_components(... monochromatic=False)`: assumes each object is a single
     4-connected component even for frames with holes. Still true.
   - `bbox_h(b) == 4 and bbox_w(b) == 4`: assumes every relevant object is 4x4.
     Still true; generator preserves this exactly.
   - Frame test checks single-color border plus bg-filled 2x2 interior. Still
     true; D5 does not alter object appearance.
   - Original `outH = 4 * max(len(frames), len(solids))` assumed two columns could
     share rows. This is false in the variant because every object now needs its
     own 4-row slot.
   - Original `outW = 8` and the two `paste(..., col=0/4)` calls hard-coded the
     old left/right layout. These are the lines that must change.

d) Placement legality
   - Generator still uses `random_free_location(... border=1, padding=1,
     padding_connectivity=4)` so 4x4 objects do not overlap, do not touch, and
     stay in bounds.
   - Because objects are separated, `find_components` will recover exactly the
     intended number of 4x4 components.

e) Visual observability
   - Humans can see the rule from 2-4 examples because output becomes a narrow
     single stack and neighboring 4x4 slots alternate frame/solid/frame/solid.
   - The generator makes sure examples are diagnostic: the input's global
     top-to-bottom order is not allowed to coincide with the output type order.
     This forces the solver to infer grouping + alternation rather than naive
     global sorting.

f) Visual cleanliness
   - Objects remain isolated in the input and perfectly stacked in the output.
     No overlaps, no markers, no extra guide pixels.

g) Color inferability
   - Colors are unchanged: each object keeps its own unique visible color.
     No hidden dependence on numeric color ids is introduced.
"""


def _extract_objects(grid: np.ndarray) -> tuple[int, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue

        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        border_colors = set(map(int, border.flatten()))

        if len(border_colors) == 1 and bool((sub[1:3, 1:3] == bg).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return bg, frames, solids


def _interleave_objects(
    frames: list[tuple[int, int, np.ndarray]], solids: list[tuple[int, int, np.ndarray]]
) -> list[tuple[int, int, np.ndarray]]:
    ordered: list[tuple[int, int, np.ndarray]] = []
    for i in range(max(len(frames), len(solids))):
        if i < len(frames):
            ordered.append(frames[i])
        if i < len(solids):
            ordered.append(solids[i])
    return ordered


def _build_output(
    bg: int, frames: list[tuple[int, int, np.ndarray]], solids: list[tuple[int, int, np.ndarray]]
) -> np.ndarray:
    ordered = _interleave_objects(frames, solids)
    out = np.full((4 * len(ordered), 4), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(ordered):
        paste(out, sub, int(i * 4), 0)
    return out


def _global_type_order(
    frames: list[tuple[int, int, np.ndarray]], solids: list[tuple[int, int, np.ndarray]]
) -> list[str]:
    tagged = [("frame", *item) for item in frames] + [("solid", *item) for item in solids]
    tagged.sort(key=lambda t: (int(t[1]), int(t[2])))
    return [str(t[0]) for t in tagged]


def _interleaved_type_order(n_frames: int, n_solids: int) -> list[str]:
    order: list[str] = []
    for i in range(max(int(n_frames), int(n_solids))):
        if i < n_frames:
            order.append("frame")
        if i < n_solids:
            order.append("solid")
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 objects by type, then place them into one single column.

    - Frames are sorted by input (row, col).
    - Solids are sorted by input (row, col).
    - Output width is 4.
    - Objects are interleaved frame, solid, frame, solid, ... starting with the
      first frame; if one group is longer, its leftovers continue at the bottom.
    """
    bg, frames, solids = _extract_objects(np.array(grid, dtype=int))
    return _build_output(bg, frames, solids)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _sample_counts() -> tuple[int, int]:
    options = [
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4),
    ]
    idx = int(np.random.randint(0, len(options)))
    return options[idx]


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n_frames, n_solids = _sample_counts()
    total = int(n_frames + n_solids)

    H = int(np.random.randint(24, 35))
    W = int(np.random.randint(24, 35))
    g = np.full((H, W), int(bg), dtype=int)

    colors = np.random.permutation(np.arange(1, 10))[:total]
    objects: list[tuple[str, np.ndarray]] = []

    for i in range(n_frames):
        objects.append(("frame", _make_frame(int(colors[i]), bg)))
    for i in range(n_solids):
        objects.append(("solid", _make_solid(int(colors[n_frames + i]))))

    order = np.random.permutation(len(objects))
    try:
        for idx in order:
            _kind, sprite = objects[int(idx)]
            r0, c0 = random_free_location(
                g, sprite, transparent=bg, border=1, padding=1, padding_connectivity=4
            )
            paste(g, sprite, int(r0), int(c0), transparent=bg)
    except Exception:
        return None

    bg2, frames, solids = _extract_objects(g)
    if len(frames) != n_frames or len(solids) != n_solids:
        return None

    global_types = _global_type_order(frames, solids)
    interleaved_types = _interleaved_type_order(n_frames, n_solids)
    if global_types == interleaved_types:
        return None

    expected = _build_output(bg2, frames, solids)
    actual = main(g)
    if actual.shape != (4 * total, 4):
        return None
    if not np.array_equal(actual, expected):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
