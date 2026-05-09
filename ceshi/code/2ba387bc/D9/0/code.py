# Variant of 2ba387bc
# Dimension: D9 — 对象数量与比例约束 (meta)
# Change: 框架数量1-4个，实心数量1-4个，总数2-8个，每个方块使用不同颜色... → 允许同色方块存在
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
Design analysis for the D9 variant

a) Concrete behavior and trace
- Pixel form is unchanged: every object is still exactly 4x4 on a uniform background.
- Frames are 1-pixel hollow squares: a uniform colored border with a 2x2 background interior.
- Solids are 4x4 uniform filled squares.
- The variant changes only color multiplicity: different objects may now reuse colors, including
  the important case "same color, different shape" (for example a blue frame and a blue solid).
- Small trace:
  Original generator might place frame(color=3) and solid(color=7); a weak solver could separate
  them by color identity and still look correct.
  Variant generator may place frame(color=3) and solid(color=3). The output still stacks the frame
  in the left column and the solid in the right column, so classification must come from the 2x2
  interior structure, not from color uniqueness.

b) Complexity calibration
- If duplicate colors were merely "allowed", many random samples would still accidentally use all
  distinct colors and the variant would be visually invisible.
- To make D9 observable, every generated sample contains at least one color shared by a frame and a
  solid. Additional objects draw from a small palette of 1..min(4, total_objects-1) colors.
- This keeps enough repetition to force shape-based reasoning without making the scene unreadable.

c) Hidden assumptions review of main()
- `get_bg(g)`: still valid because background remains the dominant color.
- `find_components(..., monochromatic=False)`: crucial. Objects are found by geometry, not by color,
  so same-colored but disconnected objects remain separate components. The generator must preserve at
  least 1-cell 4-connectivity separation.
- `bbox_h/bbox_w == 4`: still valid because all legal objects remain 4x4.
- Frame test: depends only on a uniform border and background-filled 2x2 center. This remains the
  intended discriminator and is exactly the structure we want solvers to use.
- Solid test: in this variant we make it explicit that a solid is a 4x4 single-color non-background
  patch. This avoids relying on the original loose `else -> solid` shortcut.
- Sorting by `(row, col)` and output layout are unchanged because D9 only changes color reuse.

d) Placement legality
- Objects are placed with `random_free_location(..., border=1, padding=1, padding_connectivity=4)`.
- This guarantees no overlap, no out-of-bounds placement, and no 4-connected touching between
  objects, which prevents `find_components` from merging same-colored neighbors.
- Frames are pasted with `transparent=bg` so their hollow centers remain background.

e) Visual observability
- Every sample has at least one cross-type shared color: some frame color also appears on some solid.
- A human comparing input/output pairs can see that same-colored objects split into different output
  columns, so color cannot be the deciding feature.

f) Visual cleanliness
- All objects keep the original simple 4x4 geometry.
- At least one background cell separates objects, avoiding clutter and accidental unions.
- The output format is unchanged, so the only new visual signal is deliberate color reuse.

g) Color-change inferability
- There is no color remapping rule in the transformation.
- Colors are copied through unchanged; only duplicate-color generation changes. The solver therefore
  needs no hidden palette ordering or numeric-color logic.

Source-task note
- No separate `source_task` was provided for this variant, so there was nothing extra to inspect.

Post-test behavior check
- The key metric is `shared_cross_type_colors = |frame_colors ∩ solid_colors|`.
- A sample counts as expressing the variant only if this value is at least 1.

New implicit assumptions after modification
- `main()` assumes every relevant component is either an exact frame or an exact solid; the generator
  guarantees this.
- `main()` also assumes same-colored objects are disconnected under 4-connectivity; the generator
  guarantees this via padded placement.
"""


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    return len(set(map(int, border.flatten()))) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _is_solid(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    vals = set(map(int, sub.flatten()))
    return len(vals) == 1 and int(next(iter(vals))) != int(bg)


def _color_of_object(sub: np.ndarray, bg: int) -> int:
    if _is_frame(sub, bg):
        return int(sub[0, 0])
    return int(sub[0, 0])


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into two columns by type: frames left, solids right.

    Variant detail:
    - Colors are no longer unique across objects.
    - Classification must therefore use shape structure only.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)

    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)

    return out


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _analyze_input(g: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)
    frame_cols = [_color_of_object(sub, bg) for _r0, _c0, sub in frames]
    solid_cols = [_color_of_object(sub, bg) for _r0, _c0, sub in solids]
    shared = set(frame_cols) & set(solid_cols)
    return {
        "n_frames": int(len(frames)),
        "n_solids": int(len(solids)),
        "total_objects": int(len(frames) + len(solids)),
        "distinct_colors": int(len(set(frame_cols + solid_cols))),
        "shared_cross_type_colors": int(len(shared)),
    }


def _choose_object_colors(n_frames: int, n_solids: int) -> tuple[list[int], list[int]]:
    total = int(n_frames + n_solids)
    max_distinct = min(4, total - 1, 9)
    palette_size = int(np.random.randint(1, max_distinct + 1))
    palette = list(map(int, np.random.choice(np.arange(1, 10), size=palette_size, replace=False)))
    shared = int(np.random.choice(palette))

    frame_cols = [shared]
    solid_cols = [shared]

    for _ in range(n_frames - 1):
        frame_cols.append(int(np.random.choice(palette)))
    for _ in range(n_solids - 1):
        solid_cols.append(int(np.random.choice(palette)))

    return frame_cols, solid_cols


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    g = np.full((h, w), bg, dtype=int)
    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))

    frame_cols, solid_cols = _choose_object_colors(n_frames, n_solids)
    objects: list[np.ndarray] = []
    for col in frame_cols:
        objects.append(_make_frame(col, bg))
    for col in solid_cols:
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
    if info["shared_cross_type_colors"] < 1:
        return None
    if info["distinct_colors"] >= info["total_objects"]:
        return None

    main(g)
    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
