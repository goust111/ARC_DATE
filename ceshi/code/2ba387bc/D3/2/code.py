# Variant of 2ba387bc
# Dimension: D3 — 源对象的形状与尺寸 (src)
# Change: 所有对象均为固定的4x4正方形，框架为1像素宽边框+2x2背景内部，实心为全色填充... → 三种类型的方块（框架、实心、十字形）
#
# Design analysis
# a) Concrete behavior:
#    - Keep every source object as a 4x4 square on a uniform background.
#    - Type 1 frame: border is one object color, center 2x2 is all background.
#    - Type 2 solid: all 16 cells are the same non-background color.
#    - Type 3 cross: border is one object color, center 2x2 is a fixed diagonal
#      motif [[color, bg], [bg, color]]. This adds a third visually distinct
#      internal structure while preserving the old outer silhouette.
#    - Output changes from 2 columns to 3 columns. Frames stay in cols 0:4,
#      solids stay in cols 4:8, and crosses occupy the new cols 8:12. Inside
#      each type, objects are still sorted by input row, then input column.
#    - Small trace:
#      Original: a non-frame 4x4 object fell through the else-branch and was
#      treated as solid.
#      Variant: a cross object has a uniform border like a frame, but its 2x2
#      center is neither all background nor all color, so it must be detected
#      explicitly and routed to a third list/column.
# b) Complexity calibration:
#    - Only one new category is added, and its center pattern is fixed. This
#      makes the change obvious in 2-4 examples without introducing orientation
#      ambiguity or a large template family.
#    - Generator guarantees repeated support for all three categories so the
#      third column is not a rare corner case.
# c) Implicit assumptions in the original main():
#    - `bbox_h(b) != 4 or bbox_w(b) != 4`: still valid; objects remain 4x4.
#    - `border` has one color: still valid for frame/cross and trivially true for
#      solid, so it remains a useful discriminator.
#    - `if border uniform and center all bg -> frame else solid`: no longer valid,
#      because cross objects would be misclassified as solid. The variant must
#      replace this binary fallback with explicit 3-way recognition.
#    - Output width `8` and `max(len(frames), len(solids))`: no longer valid; a
#      third type requires width `12` and max over three counts.
# d) Placement legality:
#    - Objects are still placed by `random_free_location` with `border=1` and
#      `padding=1`, which prevents touching/overlap and keeps components
#      separable under 4-connectivity.
#    - Colors are unique across objects in each generated grid, avoiding visual
#      confusion while not participating in the rule.
# e) Visual observability:
#    - The third type is visible from the center 2x2 alone: empty center, full
#      center, or diagonal center. The solver never depends on numeric color ids.
#    - Every generated sample contains at least one object of each type, so the
#      3-column output is learnable from a few examples.
# f) Visual cleanliness:
#    - No overlays, no markers, no extra annotations. The scene remains a sparse
#      scatter of separated 4x4 squares exactly like the original task.
# g) Color-change inferability:
#    - No color mapping is introduced. Colors are preserved object-by-object, so
#      the only new inference burden is shape/type recognition.
# 9) New main() assumptions and generator guarantees:
#    - Assumption: valid task objects are exactly 4x4 connected components.
#      Generator guarantees only such objects are placed.
#    - Assumption: cross orientation is the fixed main diagonal pattern.
#      Generator uses only that template.
#    - Assumption: objects are separated enough to remain distinct components.
#      Generator enforces this with padded placement.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _make_cross(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = np.array([[int(col), int(bg)], [int(bg), int(col)]], dtype=int)
    return a


def _classify_square(sub: np.ndarray, bg: int) -> str | None:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_colors = set(map(int, border.flatten()))
    whole_colors = set(map(int, sub.flatten()))
    center = sub[1:3, 1:3]

    if len(whole_colors) == 1 and int(sub[0, 0]) != int(bg):
        return "solid"

    if len(border_colors) != 1:
        return None

    col = int(next(iter(border_colors)))
    if col == int(bg):
        return None

    if bool((center == int(bg)).all()):
        return "frame"

    cross_center = np.array([[col, int(bg)], [int(bg), col]], dtype=int)
    if bool((center == cross_center).all()):
        return "cross"

    return None


def _build_output(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    crosses: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    out_h = 4 * max(len(frames), len(solids), len(crosses))
    out = np.full((out_h, 12), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    for i, (_r0, _c0, sub) in enumerate(crosses):
        paste(out, sub, int(i * 4), 8)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into three columns by internal structure:
    frames left, solids middle, diagonal-cross squares right.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []
    crosses: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        kind = _classify_square(sub, bg)

        if kind == "frame":
            frames.append((int(r0), int(c0), sub))
        elif kind == "solid":
            solids.append((int(r0), int(c0), sub))
        elif kind == "cross":
            crosses.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    crosses.sort(key=lambda t: (int(t[0]), int(t[1])))

    return _build_output(frames, solids, crosses, bg)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(22, 33))
    w = int(np.random.randint(22, 33))
    g = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(1, 4))
    n_solids = int(np.random.randint(1, 4))
    n_crosses = int(np.random.randint(1, 4))

    colors = list(np.random.permutation(np.arange(1, 10)))
    need = n_frames + n_solids + n_crosses
    if need > len(colors):
        return None
    colors = [int(c) for c in colors[:need]]

    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []
    crosses: list[tuple[int, int, np.ndarray]] = []

    try:
        for _ in range(n_frames):
            col = colors.pop()
            sub = _make_frame(col, bg)
            r0, c0 = random_free_location(
                g,
                sub,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, sub, r0, c0, transparent=bg)
            frames.append((int(r0), int(c0), sub))

        for _ in range(n_solids):
            col = colors.pop()
            sub = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                sub,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, sub, r0, c0, transparent=bg)
            solids.append((int(r0), int(c0), sub))

        for _ in range(n_crosses):
            col = colors.pop()
            sub = _make_cross(col, bg)
            r0, c0 = random_free_location(
                g,
                sub,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, sub, r0, c0, transparent=bg)
            crosses.append((int(r0), int(c0), sub))

        frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        solids.sort(key=lambda t: (int(t[0]), int(t[1])))
        crosses.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _build_output(frames, solids, crosses, bg)
        if np.array_equal(main(g), expected):
            return g
    except Exception:
        return None

    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
