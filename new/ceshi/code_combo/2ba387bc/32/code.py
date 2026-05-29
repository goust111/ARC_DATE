# Combo variant of 2ba387bc (index 32)
# combo_spec: D1/0+D13/2
# Name: 空间分离陷阱+边框分类

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo variant D1/0 + D13/2
===============================================

a) Concrete behavior trace
--------------------------
- Original task:
  - each 4x4 object is either a frame (uniform border + bg 2x2 interior) or a solid
    (all 16 cells same color);
  - solver groups by the interior test and stacks the two groups into left/right columns.
- Fused variant:
  - every object is still a single 4x4 connected component, but the semantic split is now:
    - left-column class: border is monochrome, interior is non-background;
    - right-column class: border uses multiple colors.
  - generator additionally places monochrome-border objects only in the upper half and
    multi-border objects only in the lower half to create the D13/2 spatial trap.
- Small trace:
  - example upper tile:
      [[2,2,2,2],
       [2,7,7,2],
       [2,7,7,2],
       [2,2,2,2]]
    Original logic: border is uniform but center is not bg, so it is treated as "solid".
    New logic: border is uniform, so it belongs to the left group.
  - example lower tile:
      [[3,8,3,8],
       [8,3,3,3],
       [3,3,3,8],
       [8,3,8,3]]
    Original logic: border is not uniform, so it is also treated as "solid".
    New logic: border is multi-color, so it belongs to the right group.
  - Difference:
    the original solver would collapse both examples into the right column, while the
    fused solver splits them by border consistency. Spatial position is correlated with
    the class but is not the rule used by `main()`.

b) Complexity calibration
-------------------------
- The classification signal is visually obvious and low-branching: only border-uniform
  vs border-mixed.
- The trap is also obvious: upper-half tiles consistently belong to one class and lower
  half tiles to the other. This creates a tempting but incorrect location-based shortcut.
- To avoid ambiguity and excessive color noise, each tile uses only two colors and both
  classes repeat multiple times per sample (2..4 objects each).

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- `find_components(... monochromatic=False)`:
  still valid. Variant tiles are multi-color but remain one 4-connected object.
- `bbox_h(b) == 4 and bbox_w(b) == 4`:
  still valid. Generator preserves exact 4x4 object size.
- `sub[1:3,1:3] == bg` as the frame test:
  invalid. Upper-class tiles deliberately have non-bg interiors, so this test would
  misclassify them and erase the D1/0 change.
- `len(border_colors) == 1 and center_is_bg`:
  invalid because the center condition is removed.
- Sorting by `(row, col)` and output width 8:
  still valid and unchanged.

d) Placement legality
---------------------
- Objects are placed with `random_free_location(... border=1, padding=1)`, so they never
  overlap, never leave the grid, and never touch by 4-neighborhood.
- Placement happens in two disjoint row bands separated by blank rows, so the intended
  upper/lower split is guaranteed and components cannot merge across the midline.

e) Visual observability
-----------------------
- Across a few examples a human can see:
  - upper-half objects all have monochrome borders;
  - lower-half objects all have mixed borders;
  - output still stacks one class left and the other right.
- The rule relies only on visible border patterns, not on hidden numeric color ordering.

f) Visual neatness
------------------
- The scene stays sparse: only isolated 4x4 squares on a plain background.
- No legends or overlays are added, so each object remains easy to parse individually.
- Two clean spatial bands make the trap legible without cluttering the grid.

g) Color-change inferability
----------------------------
- There is no cross-sample color remapping. Colors merely decorate individual objects.
- Classification depends on whether the outer ring is uniform, a property humans can see
  immediately. Interior colors are distractors for D1/0 but are still visually simple.

h) Quantitative anti-degeneration metric
----------------------------------------
- Metric 1: `diverges_from_original` compares this solver with an internal copy of the
  original 2ba387bc classification rule on the same generated input. Because all variant
  tiles have non-bg interiors, the original rule sends every tile to the right column.
  Valid samples must therefore disagree with the original output.
- Metric 2: `separation_gap = min(lower_rows) - max(upper_rows)` measures the spatial
  gap between the two classes. Valid samples require a positive gap, proving D13/2 is
  actually present in the generated grid.

i) New assumptions introduced by the new main()
-----------------------------------------------
- Each valid object must still be a 4x4 component.
- The border must contain only non-background colors.
- Classification is defined only by the number of distinct border colors:
  exactly 1 => left class, at least 2 => right class.
- `generate_input()` enforces these assumptions directly when constructing tiles.
"""


BG = 0


def _border_cells(sub: np.ndarray) -> np.ndarray:
    return np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])


def _classify_border(sub: np.ndarray, bg: int) -> str | None:
    border = _border_cells(sub)
    bcols = {int(v) for v in border.flatten()}
    if int(bg) in bcols:
        return None
    if len(bcols) == 1:
        return "uniform"
    if len(bcols) >= 2:
        return "mixed"
    return None


def _original_like_classify(sub: np.ndarray, bg: int) -> str | None:
    border = _border_cells(sub)
    bcols = {int(v) for v in border.flatten()}
    if int(bg) in bcols:
        return None
    if len(bcols) == 1 and bool((sub[1:3, 1:3] == int(bg)).all()):
        return "frame"
    return "solid"


def _assemble_output(
    left_tiles: list[tuple[int, int, np.ndarray]],
    right_tiles: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    out_h = 4 * max(len(left_tiles), len(right_tiles))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(left_tiles):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(right_tiles):
        paste(out, sub, int(i * 4), 4)

    return out


def _collect_tiles(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    bg = int(get_bg(grid))
    tiles: list[tuple[int, int, np.ndarray]] = []
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        tiles.append((int(r0), int(c0), crop_bbox(grid, b)))
    tiles.sort(key=lambda t: (int(t[0]), int(t[1])))
    return tiles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 tiles into two columns by border color consistency.

    - monochrome border => left column
    - multi-color border => right column
    - within each group, preserve top-to-bottom order by input row
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    left_tiles: list[tuple[int, int, np.ndarray]] = []
    right_tiles: list[tuple[int, int, np.ndarray]] = []

    for r0, c0, sub in _collect_tiles(g):
        kind = _classify_border(sub, bg)
        if kind == "uniform":
            left_tiles.append((int(r0), int(c0), sub))
        elif kind == "mixed":
            right_tiles.append((int(r0), int(c0), sub))

    return _assemble_output(left_tiles, right_tiles, bg)


def _baseline_original_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for r0, c0, sub in _collect_tiles(g):
        kind = _original_like_classify(sub, bg)
        if kind == "frame":
            frames.append((int(r0), int(c0), sub))
        elif kind == "solid":
            solids.append((int(r0), int(c0), sub))

    return _assemble_output(frames, solids, bg)


def _make_uniform_border_tile(border_col: int, center_col: int) -> np.ndarray:
    tile = np.full((4, 4), int(border_col), dtype=int)
    tile[1:3, 1:3] = int(center_col)
    return tile


def _make_mixed_border_tile(base_col: int, accent_col: int) -> np.ndarray:
    tile = np.full((4, 4), int(base_col), dtype=int)
    tile[0, :] = np.array([base_col, accent_col, base_col, accent_col], dtype=int)
    tile[3, :] = np.array([accent_col, base_col, accent_col, base_col], dtype=int)
    tile[1, 0] = accent_col
    tile[2, 0] = base_col
    tile[1, 3] = base_col
    tile[2, 3] = accent_col
    return tile


def _pick_two_colors() -> tuple[int, int]:
    cols = np.random.choice(np.arange(1, 10), size=2, replace=False)
    return int(cols[0]), int(cols[1])


def _place_in_band(
    grid: np.ndarray,
    patch: np.ndarray,
    row_start: int,
    row_end: int,
    bg: int,
) -> tuple[int, int]:
    band = grid[int(row_start) : int(row_end), :]
    r_local, c0 = random_free_location(
        band,
        patch,
        transparent=bg,
        border=1,
        padding=1,
        padding_connectivity=4,
    )
    return int(row_start + r_local), int(c0)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    uniform_rows: list[int] = []
    mixed_rows: list[int] = []

    for r0, _c0, sub in _collect_tiles(g):
        kind = _classify_border(sub, bg)
        if kind == "uniform":
            uniform_rows.append(int(r0))
        elif kind == "mixed":
            mixed_rows.append(int(r0))

    separation_gap = min(mixed_rows) - max(uniform_rows) if uniform_rows and mixed_rows else -999
    return {
        "uniform_count": len(uniform_rows),
        "mixed_count": len(mixed_rows),
        "separation_gap": int(separation_gap),
        "diverges_from_original": not np.array_equal(main(g), _baseline_original_main(g)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = BG
    h = int(np.random.randint(24, 31))
    w = int(np.random.randint(22, 31))
    g = np.full((h, w), bg, dtype=int)

    n_uniform = int(np.random.randint(2, 5))
    n_mixed = int(np.random.randint(2, 5))
    mid = h // 2
    top_end = mid - 1
    bottom_start = mid + 1

    if top_end < 8 or h - bottom_start < 8:
        return None

    placed_uniform: list[tuple[int, int, np.ndarray]] = []
    placed_mixed: list[tuple[int, int, np.ndarray]] = []

    try:
        for _ in range(n_uniform):
            border_col, center_col = _pick_two_colors()
            patch = _make_uniform_border_tile(border_col, center_col)
            r0, c0 = _place_in_band(g, patch, 0, top_end, bg)
            paste(g, patch, r0, c0, transparent=bg)
            placed_uniform.append((int(r0), int(c0), patch.copy()))

        for _ in range(n_mixed):
            base_col, accent_col = _pick_two_colors()
            patch = _make_mixed_border_tile(base_col, accent_col)
            r0, c0 = _place_in_band(g, patch, bottom_start, h, bg)
            paste(g, patch, r0, c0, transparent=bg)
            placed_mixed.append((int(r0), int(c0), patch.copy()))

        placed_uniform.sort(key=lambda t: (int(t[0]), int(t[1])))
        placed_mixed.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _assemble_output(placed_uniform, placed_mixed, bg)
        actual = main(g)
        metrics = _behavior_metrics(g)

        if not np.array_equal(actual, expected):
            return None
        if int(metrics["uniform_count"]) != n_uniform:
            return None
        if int(metrics["mixed_count"]) != n_mixed:
            return None
        if int(metrics["separation_gap"]) <= 0:
            return None
        if not bool(metrics["diverges_from_original"]):
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
