# Variant of 2ba387bc
# Dimension: D3 — 源对象的形状与尺寸 (src)
# Change: 所有对象均为固定的4x4正方形，框架为1像素宽边框+2x2背景内部，实心为全色填充... → 空心框架的边框宽度可变

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the D3 variant
==================================

a) Concrete behavior trace
   - Old task: every object is exactly 4x4. A frame is always:
         c c c c
         c 0 0 c
         c 0 0 c
         c c c c
     and a solid is a monochrome 4x4 block. Because size never changes, the old
     solver can classify by checking whether the center 2x2 is background.
   - New task: objects remain monochrome squares on a uniform background, but the
     square size is no longer fixed. Solids may be 4x4 or 6x6. Frames may be 6x6
     with either a 1-pixel or 2-pixel border, e.g.
         6x6 thin frame:
         c c c c c c
         c 0 0 0 0 c
         c 0 0 0 0 c
         c 0 0 0 0 c
         c 0 0 0 0 c
         c c c c c c
         6x6 thick frame:
         c c c c c c
         c c c c c c
         c c 0 0 c c
         c c 0 0 c c
         c c c c c c
         c c c c c c
   - Flow difference on one object:
       old logic on a 6x6 thick frame -> rejects it immediately because bbox != 4x4
       new logic on the same object -> accepts it as a square, searches for a valid
       border thickness t, finds t=2, and classifies it as a frame.
   - Output behavior remains "frames left, solids right, each group sorted by input
     row". The only necessary adaptation is that output width/height are now driven
     by the actual object sizes rather than a hard-coded 4x4 cell.

b) Complexity calibration
   - Sizes are limited to {4, 6}. This is enough to make the new behavior visible
     without creating too many free degrees.
   - Frame border thickness is limited to {1, 2}, and 2 only appears on 6x6 frames.
   - Each generated sample is forced to contain:
       one 6x6 frame with border 1,
       one 6x6 frame with border 2,
       one 6x6 solid,
       one 4x4 solid.
     This gives repeated support for the rule and prevents size-only heuristics.

c) Implicit-assumption audit of the original main()
   - `if bbox_h(b) != 4 or bbox_w(b) != 4: continue`
     Assumption: every relevant object is 4x4. Broken by the variant.
   - `sub[1:3, 1:3] == bg`
     Assumption: every frame has exactly a 2x2 hollow center. Broken by 6x6 frames.
   - `outH = 4 * max(...)` and `outW = 8`
     Assumption: every pasted object is 4x4, so two columns are always width 4.
     Broken once objects can be 6x6.
   - The old border-color test `len(bcols) == 1` is still useful, but it is not
     sufficient alone; now the whole square must match a ring of some thickness.

d) Placement legality
   - Generator still uses `random_free_location(..., padding=1, connectivity=4)` so
     objects neither overlap nor touch orthogonally.
   - All sprites are square and monochrome apart from intentional background holes.
   - The solver reads connected components of all non-background pixels, which remains
     valid because every object is one 4-connected colored component even with holes.

e) Visual observability
   - The key new clue is visible: some frames are visibly larger and have a thicker
     border, yet they still go to the same left column.
   - A 6x6 solid is also present, so "large square -> left column" is disproved.
   - Both thin and thick 6x6 frames are present in the same sample, so the solver
     must infer that border width does not change the category.

f) Visual cleanliness
   - Objects are separated by padding and use unique colors, so component boundaries
     stay clear.
   - Output rows are aligned by per-rank row heights, which avoids overlap while
     keeping the two-column arrangement tidy.

g) Color-change audit
   - Colors carry no hidden meaning. They only distinguish objects visually.
   - Sorting depends only on input row position, never on color ids.

h) Anti-degeneration check
   - Generator rejects any sample whose output under the new solver is identical to a
     faithful 4x4-only legacy solver. This guarantees the D3 change is actually active
     on the produced input instead of silently collapsing back to the original task.

i) New assumptions introduced by the variant solver
   - Every relevant object is still a square monochrome component.
   - A frame must be exactly representable as a uniform outer band of thickness t and
     a fully background inner square.
   - Generator enforces these assumptions directly through sprite construction.
"""


def _make_frame(side: int, border: int, col: int, bg: int = 0) -> np.ndarray:
    side = int(side)
    border = int(border)
    if side <= 0 or border <= 0 or side - 2 * border <= 0:
        raise ValueError("invalid frame specification")
    a = np.full((side, side), int(col), dtype=int)
    a[border : side - border, border : side - border] = int(bg)
    return a


def _make_solid(side: int, col: int) -> np.ndarray:
    side = int(side)
    if side <= 0:
        raise ValueError("invalid solid size")
    return np.full((side, side), int(col), dtype=int)


def _frame_thickness(sub: np.ndarray, bg: int) -> int | None:
    h, w = sub.shape
    if h != w or h < 3:
        return None
    colors = set(map(int, sub.flatten()))
    non_bg = sorted(colors - {int(bg)})
    if len(non_bg) != 1:
        return None

    col = int(non_bg[0])
    for border in range(1, h // 2 + 1):
        inner = h - 2 * border
        if inner <= 0:
            break
        expected = np.full((h, w), col, dtype=int)
        expected[border : h - border, border : w - border] = int(bg)
        if np.array_equal(sub, expected):
            return int(border)
    return None


def _classify_square(sub: np.ndarray, bg: int) -> tuple[str, int] | None:
    h, w = sub.shape
    if h != w:
        return None
    colors = set(map(int, sub.flatten()))
    non_bg = sorted(colors - {int(bg)})
    if len(non_bg) != 1:
        return None
    if np.all(sub != int(bg)):
        return ("solid", 0)
    border = _frame_thickness(sub, bg)
    if border is not None:
        return ("frame", int(border))
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
        item = {
            "r0": int(r0),
            "c0": int(c0),
            "sub": sub,
            "size": int(sub.shape[0]),
            "border": int(kind[1]),
        }
        if kind[0] == "frame":
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
    if out_h == 0 or out_w == 0:
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
    Sort scattered monochrome square objects into two columns by type.

    Variant rule:
    - Objects are square solids or square hollow frames.
    - Solids may be 4x4 or 6x6.
    - Frames may be 6x6 with border width 1 or 2, and may also include 4x4 thin
      frames. Border width does not affect classification.
    - Sort frames and solids independently by input row, then stack frames left and
      solids right, preserving each object's actual size.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _extract_objects(g, bg)
    return _build_output(frames, solids, bg)


def _legacy_main_4x4_only(grid: np.ndarray) -> np.ndarray:
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


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(grid))
    frames, solids = _extract_objects(np.array(grid, dtype=int), bg)
    return {
        "frame_count": len(frames),
        "solid_count": len(solids),
        "thick_frame_count": sum(int(obj["border"] >= 2) for obj in frames),
        "large_frame_count": sum(int(obj["size"] > 4) for obj in frames),
        "large_solid_count": sum(int(obj["size"] > 4) for obj in solids),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    mandatory = [
        {"kind": "frame", "side": 6, "border": 1},
        {"kind": "frame", "side": 6, "border": 2},
        {"kind": "solid", "side": 6, "border": 0},
        {"kind": "solid", "side": 4, "border": 0},
    ]

    extra_frames = int(np.random.randint(0, 2))
    extra_solids = int(np.random.randint(0, 3))
    for _ in range(extra_frames):
        side, border = [(4, 1), (6, 1), (6, 2)][int(np.random.randint(0, 3))]
        mandatory.append({"kind": "frame", "side": int(side), "border": int(border)})
    for _ in range(extra_solids):
        side = int([4, 6][int(np.random.randint(0, 2))])
        mandatory.append({"kind": "solid", "side": side, "border": 0})

    total = len(mandatory)
    H = int(np.random.randint(26, 37))
    W = int(np.random.randint(26, 37))
    g = np.full((H, W), bg, dtype=int)

    if total > 9:
        return None

    colors = list(map(int, np.random.permutation(np.arange(1, 10))[:total]))
    sprites: list[np.ndarray] = []
    for spec, col in zip(mandatory, colors):
        if spec["kind"] == "frame":
            sprites.append(_make_frame(spec["side"], spec["border"], col, bg))
        else:
            sprites.append(_make_solid(spec["side"], col))

    try:
        order = list(np.random.permutation(total))
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
    if metrics["frame_count"] != sum(spec["kind"] == "frame" for spec in mandatory):
        return None
    if metrics["solid_count"] != sum(spec["kind"] == "solid" for spec in mandatory):
        return None
    if metrics["thick_frame_count"] < 1:
        return None
    if metrics["large_frame_count"] < 2:
        return None
    if metrics["large_solid_count"] < 1:
        return None

    new_out = main(g)
    legacy_out = _legacy_main_4x4_only(g)
    if new_out.size == 0 or np.array_equal(new_out, legacy_out):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
