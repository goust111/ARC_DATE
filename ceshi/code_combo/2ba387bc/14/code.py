# Combo variant of 2ba387bc (index 14)
# combo_spec: D3/3+D12/1
# Name: 可变边框宽度+框架内部第三色

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D3/3 + D12/1
======================================

a) Concrete behavior trace
   - Original task objects are always 4x4. A frame is:
         a a a a
         a 0 0 a
         a 0 0 a
         a a a a
     and a solid is a monochrome 4x4 block. The original solver therefore does:
       1. reject any component whose bbox is not 4x4
       2. inspect the center 2x2
       3. if the center is background and the border is uniform -> frame
          otherwise -> solid
   - Combo task changes two things at once:
       1. the square can be 4x4 or 6x6, and a 6x6 frame may have border width 1 or 2
       2. frame interiors are no longer background; they are filled with a third color
   - Example object:
         6x6 thick frame with third-color interior
         b b b b b b
         b b b b b b
         b b c c b b
         b b c c b b
         b b b b b b
         b b b b b b
     Original logic fails twice:
       - it rejects the object because bbox != 4x4
       - even if size were accepted, the interior is not background, so it would not
         be recognized as a frame
     Combo logic instead:
       - accepts any square bbox
       - tests whether the square is exactly representable as a uniform outer band of
         thickness t and a uniform inner square of a different non-background color
       - classifies that object as a frame with t=2
   - Output rule remains the same at the semantic level: frames go to the left column,
     solids to the right column, both groups sorted top-to-bottom by input row. The
     physical layout must now preserve real object sizes, so row heights and column
     widths are derived from the objects instead of being fixed at 4 and 8.

b) Complexity calibration
   - Sizes are limited to {4, 6}. This makes the size change obvious without making
     the task underconstrained.
   - Border widths are limited to {1, 2}, with width 2 only appearing on 6x6 frames.
   - Every generated sample includes:
       one 6x6 thin third-color frame,
       one 6x6 thick third-color frame,
       one 4x4 third-color frame,
       one 6x6 solid,
       one 4x4 solid.
     This defeats size-only heuristics in both directions and gives repeated evidence
     that frames are defined by structure rather than by a single size or color.
   - Some frame interior colors are intentionally reused by solids, so "this color
     means inside of a frame" is not a viable shortcut.

c) Implicit-assumption audit of the original main()
   - `bbox_h(b) != 4 or bbox_w(b) != 4`
     Assumption: every relevant object is exactly 4x4. Broken by 6x6 objects.
   - `sub[1:3, 1:3] == bg`
     Assumption: a frame is identified by a background hole. Broken because combo
     frames have non-background interiors.
   - Border test using only outermost cells
     Assumption: 1-pixel border width is enough to characterize the frame. Broken by
     6x6 thick frames because classification must search over possible border widths.
   - `outH = 4 * max(...)` and `outW = 8`
     Assumption: both columns are fixed 4 cells wide and every stacked row is exactly
     4 cells tall. Broken once object size varies.
   - Multi-color connectivity
     Original generator used monochrome objects or background-holed rings. Combo frames
     are two-color filled squares, so the solver must treat connected non-background
     pixels as one multicolor component. `find_components(... monochromatic=False)`
     remains valid for that.

d) Placement legality
   - The generator uses `random_free_location(..., padding=1, connectivity=4)` before
     every paste, so objects cannot overlap and cannot touch orthogonally.
   - Every sprite is a full square with no transparent interior, so the occupied area
     exactly matches its bbox and placement remains easy to reason about.
   - Reusing colors across separate objects is safe because padding prevents accidental
     mergers into one component.

e) Visual observability
   - Humans can see three distinct clues in just a few examples:
       1. some frames are 6x6 and some are 4x4
       2. some frames have a visibly thicker border
       3. frame interiors are filled, not hollow, yet they still sort with frames
   - A 6x6 solid disproves "large squares are frames".
   - A 4x4 third-color frame disproves "4x4 means original rule still applies".
   - Reused interior/solid colors disprove a pure color-id rule.

f) Visual cleanliness
   - Objects are isolated by padding and use simple square geometry, so the scene is
     still easy to parse.
   - Output uses per-rank row heights and per-column max widths, preventing overlap
     while keeping the original two-column organization visually clean.

g) Color-change audit and inferability
   - Colors do not encode ordering or category directly.
   - The only meaningful color relation is local: a frame has one uniform border color
     and a different uniform inner color. That relation is visually explicit.
   - The palette size stays small because object sizes and frame/solid structure carry
     most of the explanatory burden.

h) Anti-degeneration check
   - The generator rejects any sample whose output matches:
       1. the original 4x4-only/background-center solver, or
       2. a size-only baseline that allows larger squares but still requires the frame
          interior to be background.
     This guarantees both combo dimensions are active on generated samples.

i) New assumptions introduced by the combo solver
   - Every relevant object is still a square.
   - A valid frame must have exactly two non-background colors: one for a uniform outer
     band and one for a uniform inner square.
   - A valid solid must be uniformly filled by one non-background color.
   - The generator enforces all of these assumptions directly.
"""


def _make_frame(side: int, border: int, border_col: int, inner_col: int) -> np.ndarray:
    side = int(side)
    border = int(border)
    border_col = int(border_col)
    inner_col = int(inner_col)
    if side <= 0 or border <= 0 or side - 2 * border <= 0:
        raise ValueError("invalid frame specification")
    if border_col == inner_col:
        raise ValueError("frame border and inner colors must differ")
    sprite = np.full((side, side), border_col, dtype=int)
    sprite[border : side - border, border : side - border] = inner_col
    return sprite


def _make_solid(side: int, col: int) -> np.ndarray:
    side = int(side)
    col = int(col)
    if side <= 0:
        raise ValueError("invalid solid size")
    return np.full((side, side), col, dtype=int)


def _frame_signature(sub: np.ndarray, bg: int) -> tuple[int, int, int] | None:
    h, w = sub.shape
    if h != w or h < 3:
        return None

    colors = sorted(set(map(int, sub.flatten())))
    if int(bg) in colors or len(colors) != 2:
        return None

    border_col = int(sub[0, 0])
    inner_candidates = [c for c in colors if c != border_col]
    if len(inner_candidates) != 1:
        return None
    inner_col = int(inner_candidates[0])

    for border in range(1, h // 2 + 1):
        inner = h - 2 * border
        if inner <= 0:
            break
        expected = np.full((h, w), border_col, dtype=int)
        expected[border : h - border, border : w - border] = inner_col
        if np.array_equal(sub, expected):
            return int(border), border_col, inner_col
    return None


def _classify_square(sub: np.ndarray, bg: int) -> tuple[str, dict[str, int]] | None:
    h, w = sub.shape
    if h != w:
        return None

    colors = sorted(set(map(int, sub.flatten())))
    if len(colors) == 1 and int(colors[0]) != int(bg):
        return ("solid", {"size": int(h), "color": int(colors[0]), "border": 0, "inner": int(colors[0])})

    signature = _frame_signature(sub, bg)
    if signature is not None:
        border, border_col, inner_col = signature
        return (
            "frame",
            {
                "size": int(h),
                "color": int(border_col),
                "border": int(border),
                "inner": int(inner_col),
            },
        )
    return None


def _extract_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[dict] = []
    solids: list[dict] = []

    for comp in comps:
        r0, c0, _r1, _c1 = comp["bbox"]
        sub = crop_bbox(grid, comp["bbox"])
        kind = _classify_square(sub, bg)
        if kind is None:
            continue
        label, info = kind
        item = {
            "r0": int(r0),
            "c0": int(c0),
            "sub": sub,
            "size": int(info["size"]),
            "border": int(info["border"]),
            "border_col": int(info["color"]),
            "inner_col": int(info["inner"]),
        }
        if label == "frame":
            frames.append(item)
        else:
            solids.append(item)

    frames.sort(key=lambda t: (int(t["r0"]), int(t["c0"])))
    solids.sort(key=lambda t: (int(t["r0"]), int(t["c0"])))
    return frames, solids


def _build_output(frames: list[dict], solids: list[dict], bg: int) -> np.ndarray:
    left_w = max((int(obj["size"]) for obj in frames), default=0)
    right_w = max((int(obj["size"]) for obj in solids), default=0)
    n_rows = max(len(frames), len(solids))
    row_heights = [
        max(
            int(frames[i]["size"]) if i < len(frames) else 0,
            int(solids[i]["size"]) if i < len(solids) else 0,
        )
        for i in range(n_rows)
    ]

    out_h = int(sum(row_heights))
    out_w = int(left_w + right_w)
    if out_h == 0 and out_w == 0:
        return np.full((0, 0), int(bg), dtype=int)

    out = np.full((out_h, out_w), int(bg), dtype=int)
    y = 0
    for i, row_h in enumerate(row_heights):
        if i < len(frames):
            paste(out, frames[i]["sub"], y, 0)
        if i < len(solids):
            paste(out, solids[i]["sub"], y, left_w)
        y += int(row_h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered square objects into two columns by structure.

    Combo rule:
    - Solid: a uniformly filled non-background square.
    - Frame: a square made of a uniform outer border of thickness 1 or 2 and a
      uniform inner square of a different non-background color.
    - Frame size may be 4x4 or 6x6. Solid size may also be 4x4 or 6x6.
    - Frames stack on the left, solids on the right, each group sorted by input row.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _extract_objects(g, bg)
    return _build_output(frames, solids, bg)


def _legacy_main_original(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if int(r1 - r0 + 1) != 4 or int(c1 - c0 + 1) != 4:
            continue
        sub = crop_bbox(g, comp["bbox"])
        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        if len(set(map(int, border.flatten()))) == 1 and bool((sub[1:3, 1:3] == bg).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (t[0], t[1]))
    solids.sort(key=lambda t: (t[0], t[1]))
    out_h = 4 * max(len(frames), len(solids))
    out = np.full((out_h, 8), bg, dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, i * 4, 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, i * 4, 4)
    return out


def _bg_center_variable_size_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[dict] = []
    solids: list[dict] = []

    for comp in comps:
        r0, c0, _r1, _c1 = comp["bbox"]
        sub = crop_bbox(g, comp["bbox"])
        h, w = sub.shape
        if h != w:
            continue

        colors = sorted(set(map(int, sub.flatten())))
        if len(colors) == 1 and int(colors[0]) != int(bg):
            solids.append({"r0": int(r0), "c0": int(c0), "sub": sub, "size": int(h)})
            continue

        if len(colors) != 2 or int(bg) not in colors:
            solids.append({"r0": int(r0), "c0": int(c0), "sub": sub, "size": int(h)})
            continue

        border_col = int(sub[0, 0])
        ok = False
        for border in range(1, h // 2 + 1):
            inner = h - 2 * border
            if inner <= 0:
                break
            expected = np.full((h, w), border_col, dtype=int)
            expected[border : h - border, border : w - border] = int(bg)
            if np.array_equal(sub, expected):
                ok = True
                break

        item = {"r0": int(r0), "c0": int(c0), "sub": sub, "size": int(h)}
        if ok:
            frames.append(item)
        else:
            solids.append(item)

    frames.sort(key=lambda t: (int(t["r0"]), int(t["c0"])))
    solids.sort(key=lambda t: (int(t["r0"]), int(t["c0"])))
    return _build_output(frames, solids, bg)


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(grid))
    frames, solids = _extract_objects(np.array(grid, dtype=int), bg)
    return {
        "frame_count": len(frames),
        "solid_count": len(solids),
        "third_color_frame_count": sum(int(obj["inner_col"] != bg) for obj in frames),
        "thick_frame_count": sum(int(obj["border"] >= 2) for obj in frames),
        "large_frame_count": sum(int(obj["size"] > 4) for obj in frames),
        "large_solid_count": sum(int(obj["size"] > 4) for obj in solids),
    }


def _assign_colors(specs: list[dict]) -> list[np.ndarray]:
    frame_specs = [spec for spec in specs if spec["kind"] == "frame"]
    solid_specs = [spec for spec in specs if spec["kind"] == "solid"]

    palette = list(map(int, np.random.permutation(np.arange(1, 10))))
    border_colors = palette[: len(frame_specs)]
    solid_colors = palette[len(frame_specs) : len(frame_specs) + len(solid_specs)]

    if len(border_colors) < len(frame_specs) or len(solid_colors) < len(solid_specs):
        raise ValueError("not enough palette colors for mandatory objects")

    reusable_inner_pool = solid_colors + palette[len(frame_specs) + len(solid_specs) :]
    if not reusable_inner_pool:
        reusable_inner_pool = solid_colors[:]

    sprites: list[np.ndarray] = []
    frame_idx = 0
    solid_idx = 0
    for spec in specs:
        if spec["kind"] == "frame":
            border_col = int(border_colors[frame_idx])
            if solid_colors:
                inner_col = int(solid_colors[frame_idx % len(solid_colors)])
            else:
                choices = [col for col in reusable_inner_pool if int(col) != border_col]
                inner_col = int(choices[0])
            if inner_col == border_col:
                choices = [col for col in reusable_inner_pool if int(col) != border_col]
                inner_col = int(choices[0])
            sprites.append(_make_frame(spec["side"], spec["border"], border_col, inner_col))
            frame_idx += 1
        else:
            sprites.append(_make_solid(spec["side"], int(solid_colors[solid_idx])))
            solid_idx += 1
    return sprites


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    specs = [
        {"kind": "frame", "side": 6, "border": 1},
        {"kind": "frame", "side": 6, "border": 2},
        {"kind": "frame", "side": 4, "border": 1},
        {"kind": "solid", "side": 6, "border": 0},
        {"kind": "solid", "side": 4, "border": 0},
    ]

    extra_frames = int(np.random.randint(0, 2))
    extra_solids = int(np.random.randint(0, 2))
    for _ in range(extra_frames):
        side, border = [(4, 1), (6, 1), (6, 2)][int(np.random.randint(0, 3))]
        specs.append({"kind": "frame", "side": int(side), "border": int(border)})
    for _ in range(extra_solids):
        side = int([4, 6][int(np.random.randint(0, 2))])
        specs.append({"kind": "solid", "side": side, "border": 0})

    H = int(np.random.randint(28, 39))
    W = int(np.random.randint(28, 39))
    g = np.full((H, W), bg, dtype=int)

    try:
        sprites = _assign_colors(specs)
        order = list(np.random.permutation(len(sprites)))
        for idx in order:
            sprite = sprites[idx]
            r0, c0 = random_free_location(
                g,
                sprite,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, sprite, r0, c0, transparent=bg)
    except Exception:
        return None

    metrics = _variant_metrics(g)
    frame_target = sum(int(spec["kind"] == "frame") for spec in specs)
    solid_target = sum(int(spec["kind"] == "solid") for spec in specs)
    if metrics["frame_count"] != frame_target or metrics["solid_count"] != solid_target:
        return None
    if metrics["third_color_frame_count"] != frame_target:
        return None
    if metrics["thick_frame_count"] < 1:
        return None
    if metrics["large_frame_count"] < 2:
        return None
    if metrics["large_solid_count"] < 1:
        return None

    new_out = main(g)
    legacy_out = _legacy_main_original(g)
    bg_center_out = _bg_center_variable_size_main(g)
    if new_out.size == 0:
        return None
    if np.array_equal(new_out, legacy_out):
        return None
    if np.array_equal(new_out, bg_center_out):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
