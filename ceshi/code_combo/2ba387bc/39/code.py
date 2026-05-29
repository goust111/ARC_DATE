# Combo variant of 2ba387bc (index 39)
# combo_spec: D1/1+D9/0+D11/1
# Name: 颜色计数分类+同色方块+图案干扰

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
Design analysis for combo D1/1 + D9/0 + D11/1
==============================================

Source-task reminder
- The original task scatters several 4x4 objects on a uniform background.
- True frames are 4x4 hollow squares: one non-bg border color, bg 2x2 center.
- True solids are 4x4 single-color filled squares.
- Output keeps the same two-column layout from the evaluation examples: frames stacked on the
  left, solids stacked on the right, both sorted by input `(row, col)`.

a) Concrete behavior and trace
- This combo keeps the original output transform but changes the reasoning burden:
  1. D1: classification should start from color count rather than only "look at the center".
  2. D9: colors are no longer unique; a frame and a solid may share the same color.
  3. D11: extra 4x4 connected distractors appear and must be ignored.
- The important fusion point is that "2 colors = frame" is not literally sufficient once
  distractors exist. We therefore use a two-stage interpretation:
  - exactly 1 non-bg color across the 4x4 -> solid
  - exactly 2 colors, one of them is bg, and the arrangement is a true hollow frame -> frame
  - everything else -> distractor / ignore
- Small trace:
  Input objects by top row:
  - blue solid: one color -> keep in right column
  - blue frame: two colors `{blue, bg}` with bg 2x2 center -> keep in left column
  - red "false frame": two colors `{red, bg}` but only one center cell is bg -> ignore
  - yellow/green checkerboard: two non-bg colors -> ignore
  Original main():
  - blue solid -> right
  - blue frame -> left
  - red false frame -> not a frame, therefore incorrectly sent right
  - checkerboard -> not a frame, therefore incorrectly sent right
  Combo main():
  - solid stays right
  - frame stays left
  - both distractors fail the strict color-count + structure tests and disappear
  Difference:
  - D9 makes color identity useless because blue can appear in both columns.
  - D11 makes the old `else -> solid` shortcut invalid.
  - D1 remains observable because the first split is by number of colors, but D11 forces the
    solver to define the 2-color case precisely rather than stopping at `len(colors) == 2`.

b) Complexity calibration
- If duplicate colors were merely allowed, many random samples would still accidentally have all
  distinct colors. To make D9 visible, every sample contains at least one color shared by a true
  frame and a true solid.
- If distractors were merely optional, many samples would collapse back to the original task.
  To make D11 visible, every sample contains at least:
  - one 2-color distractor that includes bg but is not a true frame
  - one additional patterned distractor with either 2 non-bg colors or 3 colors
- Primary object colors are drawn from a small palette so repeats are common, but the total
  palette stays limited enough for humans to visually track.

c) Implicit-assumption audit of the original main()
- `get_bg(g)`: still valid because the canvas is much larger than the foreground occupancy.
- `find_components(..., monochromatic=False)`: still essential. Same-colored objects can exist,
  so component separation must come from geometry and placement padding, not color identity.
- `bbox_h(b) == bbox_w(b) == 4`: still valid for all relevant objects, including distractors.
- Original frame test:
  `len(border_colors) == 1 and center == bg`
  This still identifies true frames.
- Original fallback:
  `else: solids.append(...)`
  This no longer holds. In the combo, some 4x4 components are distractors and must be ignored.
- New explicit assumptions:
  - a true solid is exactly one non-bg color over all 16 cells
  - a true frame is exactly two colors, one is bg, border is monochrome non-bg, center is all bg
  - any other 4x4 connected component is not part of the output

d) Placement legality
- All objects are placed with `random_free_location(..., border=1, padding=1,
  padding_connectivity=4)`.
- This guarantees no overlap, no out-of-bounds placement, and at least one background cell of
  4-connectivity separation, preventing accidental component merging even when colors repeat.
- Objects that contain bg cells are pasted with `transparent=bg` so their intended holes remain bg.

e) Visual observability
- A human can infer the rule from 2-4 examples because each sample visibly shows:
  - same-colored frame/solid pairs that split into different output columns
  - extra 4x4 patterns that vanish from the output
  - at least one "misleading" 2-color `{color, bg}` distractor, proving that "has two colors"
    is not by itself enough to be a frame
- No hidden numeric-color ordering is used in sorting or classification.

f) Visual cleanliness
- All objects remain clean 4x4 patches on a blank background.
- Distractors come from a small library of readable patterns rather than noisy mosaics.
- Padding keeps components visually separable, which matters because repeated colors are allowed.

g) Color-change inferability
- There is no color remapping at all. Colors are copied through unchanged for kept objects.
- The only color-related logic is counting distinct colors inside each 4x4 crop and preserving
  enough repeated colors that humans can see color identity is not the grouping key.

h) Behavior-validation metric
- `shared_cross_type_colors >= 1`: proves D9 is expressed.
- `bg_two_color_distractors >= 1`: proves the sample contains a 2-color non-frame that would trap
  an over-literal "2 colors => frame" heuristic.
- `legacy_output_differs = True`: we also compare against the original solver logic; because every
  sample has distractors, the legacy output must differ from the combo output on the same input.

i) New solver assumptions and generator guarantees
- `main()` assumes every relevant kept object is a connected 4x4 component. The generator enforces
  this by construction and padded placement.
- `main()` assumes same-colored objects remain disconnected. The generator enforces this via
  `padding=1`.
- `main()` assumes the 2-color frame case must include bg and have a hollow 2x2 center. The
  generator guarantees all true frames satisfy this and all mandatory misleading distractors violate
  it in a visible way.
"""


def _unique_colors(sub: np.ndarray) -> set[int]:
    return set(map(int, sub.flatten()))


def _is_solid(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    cols = _unique_colors(sub)
    return len(cols) == 1 and int(bg) not in cols


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    cols = _unique_colors(sub)
    if len(cols) != 2 or int(bg) not in cols:
        return False
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_cols = set(map(int, border.flatten()))
    if len(border_cols) != 1:
        return False
    frame_col = int(next(iter(border_cols)))
    return frame_col != int(bg) and bool((sub[1:3, 1:3] == int(bg)).all())


def _classify_4x4(sub: np.ndarray, bg: int) -> str | None:
    if _is_solid(sub, bg):
        return "solid"
    if _is_frame(sub, bg):
        return "frame"
    return None


def _collect_objects(
    grid: np.ndarray,
    bg: int,
) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []
    distractors: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(grid, b)
        kind = _classify_4x4(sub, bg)
        if kind == "frame":
            frames.append((int(r0), int(c0), sub))
        elif kind == "solid":
            solids.append((int(r0), int(c0), sub))
        else:
            distractors.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    distractors.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids, distractors


def _build_output(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
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
        border_cols = set(map(int, border.flatten()))
        if len(border_cols) == 1 and bool((sub[1:3, 1:3] == int(bg)).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return _build_output(frames, solids, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo behavior:
    - 1-color 4x4 non-bg blocks are solids and stack on the right.
    - 2-color 4x4 blocks are frames only when they are exactly {border_color, bg} with a hollow
      2x2 center; otherwise they are distractors and disappear.
    - Same colors may appear across multiple objects and across both kept categories.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids, _distractors = _collect_objects(g, bg)
    return _build_output(frames, solids, bg)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _make_bg_false_frame(kind: str, col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    if kind == "one_hole":
        a[1, 1] = int(bg)
        return a
    if kind == "diag_holes":
        a[1, 1] = int(bg)
        a[2, 2] = int(bg)
        return a
    if kind == "three_holes":
        a[1, 1] = int(bg)
        a[1, 2] = int(bg)
        a[2, 1] = int(bg)
        return a
    raise ValueError(f"unknown bg_false_frame kind: {kind}")


def _make_pattern_distractor(kind: str, colors: list[int]) -> np.ndarray:
    if kind == "checker":
        col_a, col_b = map(int, colors[:2])
        a = np.full((4, 4), col_a, dtype=int)
        for r in range(4):
            for c in range(4):
                if (r + c) % 2 == 1:
                    a[r, c] = col_b
        return a
    if kind == "hstripe":
        col_a, col_b = map(int, colors[:2])
        a = np.full((4, 4), col_a, dtype=int)
        a[2:, :] = col_b
        return a
    if kind == "vstripe":
        col_a, col_b = map(int, colors[:2])
        a = np.full((4, 4), col_a, dtype=int)
        a[:, 2:] = col_b
        return a
    if kind == "filled_frame":
        col_a, col_b = map(int, colors[:2])
        a = np.full((4, 4), col_a, dtype=int)
        a[1:3, 1:3] = col_b
        return a
    if kind == "tricolor_rows":
        col_a, col_b, col_c = map(int, colors[:3])
        a = np.full((4, 4), col_a, dtype=int)
        a[1:3, :] = col_b
        a[3, :] = col_c
        return a
    raise ValueError(f"unknown pattern distractor kind: {kind}")


def _object_color(sub: np.ndarray, bg: int) -> int:
    if _is_frame(sub, bg):
        return int(sub[0, 0])
    return int(sub[0, 0])


def _analyze_input(g: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(g))
    frames, solids, distractors = _collect_objects(g, bg)
    frame_cols = [_object_color(sub, bg) for _r0, _c0, sub in frames]
    solid_cols = [_object_color(sub, bg) for _r0, _c0, sub in solids]

    two_color_distractors = 0
    bg_two_color_distractors = 0
    non_bg_two_color_distractors = 0
    multicolor_distractors = 0
    for _r0, _c0, sub in distractors:
        cols = _unique_colors(sub)
        if len(cols) == 2:
            two_color_distractors += 1
            if int(bg) in cols:
                bg_two_color_distractors += 1
            else:
                non_bg_two_color_distractors += 1
        if len(cols) >= 3:
            multicolor_distractors += 1

    legacy_differs = int(not np.array_equal(main(g), _legacy_main(g)))
    return {
        "n_frames": int(len(frames)),
        "n_solids": int(len(solids)),
        "n_distractors": int(len(distractors)),
        "shared_cross_type_colors": int(len(set(frame_cols) & set(solid_cols))),
        "two_color_distractors": int(two_color_distractors),
        "bg_two_color_distractors": int(bg_two_color_distractors),
        "non_bg_two_color_distractors": int(non_bg_two_color_distractors),
        "multicolor_distractors": int(multicolor_distractors),
        "legacy_output_differs": int(legacy_differs),
    }


def _choose_primary_colors(n_frames: int, n_solids: int) -> tuple[list[int], list[int], list[int]]:
    total = int(n_frames + n_solids)
    palette_size = int(np.random.randint(2, min(4, total, 9) + 1))
    palette = list(map(int, np.random.choice(np.arange(1, 10), size=palette_size, replace=False)))
    shared = int(np.random.choice(palette))

    frame_cols = [shared]
    solid_cols = [shared]
    for _ in range(n_frames - 1):
        frame_cols.append(int(np.random.choice(palette)))
    for _ in range(n_solids - 1):
        solid_cols.append(int(np.random.choice(palette)))
    return frame_cols, solid_cols, palette


def _pick_colors(pool: list[int], k: int) -> list[int]:
    colors = list(map(int, pool))
    np.random.shuffle(colors)
    if len(colors) >= k:
        return colors[:k]
    extra_pool = [c for c in range(1, 10) if c not in colors]
    np.random.shuffle(extra_pool)
    need = k - len(colors)
    return colors + extra_pool[:need]


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(24, 35))
    w = int(np.random.randint(24, 35))
    g = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    n_extra_distractors = int(np.random.randint(1, 3))

    frame_cols, solid_cols, palette = _choose_primary_colors(n_frames, n_solids)
    records: list[tuple[str, int, int, np.ndarray]] = []

    pattern_kinds = ["checker", "hstripe", "vstripe", "filled_frame", "tricolor_rows"]
    false_frame_kinds = ["one_hole", "diag_holes", "three_holes"]

    objects: list[tuple[str, np.ndarray]] = []
    for col in frame_cols:
        objects.append(("frame", _make_frame(col, bg)))
    for col in solid_cols:
        objects.append(("solid", _make_solid(col)))

    false_frame_color = int(np.random.choice(palette))
    false_frame = _make_bg_false_frame(str(np.random.choice(false_frame_kinds)), false_frame_color, bg)
    if _classify_4x4(false_frame, bg) is not None:
        return None
    objects.append(("distractor", false_frame))

    first_pattern_kind = str(np.random.choice(pattern_kinds))
    first_pattern_colors = _pick_colors(palette, 3 if first_pattern_kind == "tricolor_rows" else 2)
    first_pattern = _make_pattern_distractor(first_pattern_kind, first_pattern_colors)
    if _classify_4x4(first_pattern, bg) is not None:
        return None
    objects.append(("distractor", first_pattern))

    for _ in range(n_extra_distractors):
        kind = str(np.random.choice(pattern_kinds))
        cols = _pick_colors(palette, 3 if kind == "tricolor_rows" else 2)
        patch = _make_pattern_distractor(kind, cols)
        if _classify_4x4(patch, bg) is not None:
            return None
        objects.append(("distractor", patch))

    np.random.shuffle(objects)

    try:
        for kind, patch in objects:
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, int(r0), int(c0), transparent=bg)
            records.append((kind, int(r0), int(c0), patch))
    except Exception:
        return None

    expected_frames = [(r0, c0, patch) for kind, r0, c0, patch in records if kind == "frame"]
    expected_solids = [(r0, c0, patch) for kind, r0, c0, patch in records if kind == "solid"]
    expected_frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    expected_solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    expected = _build_output(expected_frames, expected_solids, bg)

    got = main(g)
    if not np.array_equal(got, expected):
        return None

    info = _analyze_input(g)
    expected_distractors = sum(1 for kind, _r0, _c0, _patch in records if kind == "distractor")
    if info["n_frames"] != n_frames or info["n_solids"] != n_solids:
        return None
    if info["n_distractors"] != expected_distractors:
        return None
    if info["shared_cross_type_colors"] < 1:
        return None
    if info["bg_two_color_distractors"] < 1:
        return None
    if info["two_color_distractors"] < 1:
        return None
    if info["non_bg_two_color_distractors"] + info["multicolor_distractors"] < 1:
        return None
    if info["legacy_output_differs"] != 1:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
