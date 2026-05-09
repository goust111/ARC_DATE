# Variant of 269e22fb
# Dimension: D17 — 图案的拓扑结构 (meta)
# Change: 平面拓扑，图案有明确边界，裁剪窗口不跨越边界... → 圆柱拓扑（仅水平循环）

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import colors_present


# Design analysis for the D17 variant
# -----------------------------------
# spec:
#   preserved_structure:
#     - The hidden target is still the same fixed 20x20 binary motif.
#     - The motif can still appear in any of its 8 dihedral orientations.
#     - The visible input is still a two-color rectangular crop and the output is
#       still the full 20x20 motif repainted with those same two colors.
#   core_topology_change:
#     - Original task: the crop was a standard planar rectangle fully contained
#       inside columns [0, 19].
#     - New task: rows are still planar, but columns live on a horizontal cycle.
#       A crop may start near the right edge and continue from column 0.
#   behavior_trace:
#     - Example with W=8 and horizontal start c=17:
#       old planar crop used columns [17, 18, 19] only if W<=3, so a width-8
#       crop could never start there.
#       new cylindrical crop uses columns [17, 18, 19, 0, 1, 2, 3, 4].
#       The solver therefore must replace contiguous slicing with modular column
#       indexing, while keeping the vertical offset logic unchanged.
#   complexity_calibration:
#     - To make D17 visible instead of silently degenerating to the planar task,
#       the generator always chooses a wrapping crop.
#     - For this particular motif, every wrapping crop with H in [8, 16] and
#       W in [8, 12] is uniquely identifiable across orientation, offset, and
#       fg/bg polarity, so the original size range can be preserved.
#   implicit_assumption_audit:
#     - `H <= 20 and W <= 20` remains valid.
#     - `len(colors) == 2` remains valid.
#     - The row loop `r in [0, 20-H]` still depends on no vertical wrap; the
#       generator keeps that invariant.
#     - The old column loop `c in [0, 20-W]` no longer matches the task because
#       starts near the right edge are now legal; it must become `c in [0, 19]`.
#     - The old slice `pattern[r:r+H, c:c+W]` is no longer sound; it must use
#       wrapped column extraction.
#   placement_legality:
#     - Generator picks `r` from `[0, 20-H]`, so rows never wrap.
#     - Generator picks `c` from `[21-W, 19]`, guaranteeing `c+W > 20`, so the
#       crop always crosses the horizontal boundary and exercises the variant.
#     - No extra objects are introduced, so overlap/clutter risks stay unchanged.
#   visual_observability:
#     - The leftmost and rightmost columns of the input can be adjacent pieces of
#       the same hidden motif neighborhood. That is visible in the pixels and is
#       exactly the new rule a human must infer.
#   color_reasoning:
#     - Colors still only encode fg/bg polarity. No ordering, sorting, or hidden
#       dependence on color ids is introduced.
#   new_solver_assumptions:
#     - Valid inputs must have exactly one match among
#       8 orientations x (20-H+1) row starts x 20 wrapped column starts x
#       2 fg/bg polarities.
#     - The generator enforces that assumption by sampling only wrapping crops in
#       the proven-unique size range and verifying `main(input) == expected`.


_BASE_PATTERN = (
    "11000000000000011000",
    "11111100000000011000",
    "11111110000000011100",
    "11000111000000011100",
    "00000011100000111110",
    "00000001110000100110",
    "00000000110011100110",
    "00000000010010100110",
    "00000000011110100110",
    "01111111110110100110",
    "01000000011110100110",
    "01111111110010100110",
    "01010000010011100110",
    "11101111110000100110",
    "01011000010000111110",
    "11101000010001101100",
    "01011000010001000110",
    "11101000010001100011",
    "01011000101000101110",
    "11100001000100111000",
)


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _extract_crop(pattern: np.ndarray, r: int, c: int, h: int, w: int, *, wrap_horizontal: bool) -> np.ndarray:
    rows = pattern[int(r) : int(r) + int(h)]
    if not wrap_horizontal:
        return rows[:, int(c) : int(c) + int(w)]
    cols = (np.arange(int(w)) + int(c)) % pattern.shape[1]
    return rows[:, cols]


def _find_matches(grid: np.ndarray, *, wrap_horizontal: bool) -> list[tuple[int, int, np.ndarray, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    colors = colors_present(g)
    base = _pattern_mask()

    matches: list[tuple[int, int, np.ndarray, int, int]] = []
    col_starts = range(20) if wrap_horizontal else range(20 - w + 1)
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible = g == int(fg)
        for pattern in _dihedral_variants(base):
            for r in range(20 - h + 1):
                for c in col_starts:
                    window = _extract_crop(pattern, r, c, h, w, wrap_horizontal=wrap_horizontal)
                    if np.array_equal(window, visible):
                        matches.append((int(fg), int(bg), pattern, int(r), int(c)))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full 20x20 motif from a horizontally wrapped crop of one dihedral variant.

    Input structure:
    - The grid is a two-color rectangular crop of one rotation/reflection of the shared 20x20 binary motif.
    - The crop can cross the left/right boundary of that 20x20 canvas, so columns wrap horizontally, but rows remain within ordinary top/bottom bounds.

    Rule:
    - Test each input color as the motif foreground.
    - Compare the resulting boolean crop against every dihedral orientation, every legal vertical offset, and every horizontal start column using wrapped column indexing.
    - When exactly one wrapped placement matches, repaint the full 20x20 motif with the recovered foreground/background colors.

    Key insight:
    - The crop still comes from one fixed binary template; the only extra freedom is that the visible rectangle may splice together the motif's right edge and left edge.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h <= 20 and w <= 20

    colors = colors_present(g)
    assert len(colors) == 2

    matches = _find_matches(g, wrap_horizontal=True)
    assert len(matches) == 1
    fg, bg, pattern, _r, _c = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _sample_case(*, return_meta: bool = False):
    base = _pattern_mask()
    pattern = random.choice(_dihedral_variants(base))
    fg, bg = random.sample(list(range(10)), 2)
    output = np.where(pattern, fg, bg).astype(int)

    h = random.randint(8, 16)
    w = random.randint(8, 12)
    r = random.randint(0, 20 - h)
    c = random.randint(21 - w, 19)
    assert c + w > 20

    grid = _extract_crop(output, r, c, h, w, wrap_horizontal=True).astype(int)

    matches = _find_matches(grid, wrap_horizontal=True)
    assert len(matches) == 1
    assert len(_find_matches(grid, wrap_horizontal=False)) == 0
    assert np.array_equal(main(grid), output)

    if return_meta:
        return grid, output, {"h": h, "w": w, "r": r, "c": c, "wrap_span": c + w - 20}
    return grid, output


def generate_input() -> np.ndarray:
    return _sample_case(return_meta=False)[0]
