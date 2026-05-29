# Combo variant of 71e489b6 (index 27)
# combo_spec: D1/2+D3/2+D10/2
# Name: 填充缺陷+无毛刺+前景背景互换
from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask


# Design analysis for the fused variant
# a) Pixel-level trace:
#    - Original: blue foreground on black background; black defects stay black and receive an
#      orange Chebyshev-1 halo; some blue spurs inside black are absorbed first.
#    - Fused: black is now the large foreground color, blue is the background color, and the
#      only targets are tiny blue components trapped inside black regions. Those target pixels
#      themselves become orange. There is no spur-absorption stage at all.
#    - Example: a 1-cell blue island inside a black block changes 1 -> 7 in place; the 8
#      surrounding black cells remain 0. This is visually different from the original halo.
# b) Complexity calibration:
#    - Keep the original partitioned large-region layout so the task still contains competing
#      large blue/black masses.
#    - Keep 2-9 defects of size 1-3 so samples show repetition and multiple sizes, but avoid
#      extra target classes such as leaf tips that would muddy the fused rule.
# c) Implicit-assumption audit for main():
#    - The original STEP 0 assumed blue spurs may exist and should disappear. That assumption is
#      removed, so the fused solver must work directly on the raw grid.
#    - The original component scan treated color 0 as the target class. After the color-role
#      swap, target candidates must instead be blue components.
#    - The original halo painter assumed target pixels remain black. That assumption is inverted:
#      now the target pixels are recolored to orange directly.
#    - A new assumption appears: every true defect is a small blue 4-component fully enclosed by
#      black. The generator enforces this by placing defects only on strict black interiors.
# d) Placement legality:
#    - Defects are sampled from black interior pixels only, never on the border, never
#      overlapping, and kept Chebyshev-separated so they do not merge.
#    - Every non-defect 4-neighbor around a defect remains black, so each defect is a distinct
#      enclosed blue island.
# e) Visual observability:
#    - Humans can infer the rule because only tiny blue islands inside black flip to orange,
#      while the large blue background regions stay blue. The changed cells are the islands
#      themselves, not surrounding rings.
# f) Visual cleanliness:
#    - No spurs and no leaf targets means a single semantic layer: "small enclosed blue pieces".
#      This avoids mixed annotations on region boundaries.
# g) Change inferability:
#    - The color change is supported structurally: blue cells inside black regions turn orange,
#      but other blue areas do not. There are only three colors and one repeated object type.
# i) Generator/solver contract:
#    - main() assumes non-defect blue components are either large or touch non-black exterior.
#      generate_input() guarantees this by validating large region sizes and placing defects only
#      inside strict black interiors.


# --------------- solver ---------------


def _find_enclosed_blue_defects(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return all size-1..3 blue components fully enclosed by black in 4-neighborhood."""
    H, W = grid.shape
    defects: list[list[tuple[int, int]]] = []
    for coords in components_mask(grid == 1, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) > 3:
            continue

        comp_set = set(pts)
        enclosed = True
        for r, c in pts:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    enclosed = False
                    break
                if (rr, cc) in comp_set:
                    continue
                if int(grid[rr, cc]) != 0:
                    enclosed = False
                    break
            if not enclosed:
                break

        if enclosed:
            defects.append(pts)
    return defects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor enclosed blue defects inside black foreground regions.

    Input structure:
    - Blue (1) is the background color and may form one or more large regions.
    - Black (0) forms the large foreground masses.
    - Some black regions contain tiny blue 4-connected islands of size 1-3; there are no blue
      spurs to absorb and no separate leaf-tip target class.

    Rule:
    - Scan blue connected components.
    - If a blue component has size at most 3 and every 4-neighbor outside that component is
      black and in-bounds, treat it as a defect.
    - Fill those defect pixels themselves with orange (7). Leave all surrounding cells unchanged.

    Key insight: The orange marks the small blue islands trapped inside black regions, so the
    changed pixels are the defects themselves rather than a halo around them.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    for pts in _find_enclosed_blue_defects(g):
        for r, c in pts:
            out[r, c] = 7

    return out


# --------------- generator ---------------

# Tunable generation parameters
GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MIN_BLUE_REGION = 24
MIN_BLACK_REGION = 30
PARTITION_STYLES = ["h_stripes", "rect_inset", "l_shape"]
STRIPE_CUTS_RANGE = (1, 3)
CUT_MARGIN = 3
INSET_MARGIN = 2
INNER_INSET_MIN_GAP = 6
INNER_INSET_PROB = 0.5
BLACK_BAND_HEIGHT_RANGE = (3, 5)
SIDE_MARGIN_RANGE = (1, 2)
DEFECT_COUNT_RANGE = (2, 9)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 60
MIN_DEFECTS = 2
MAX_GEN_ATTEMPTS = 300


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(
    pixels: list[tuple[int, int]] | set[tuple[int, int]], dist: int
) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create a blue background with one or more large black foreground regions."""
    grid = np.ones((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        max_bands = STRIPE_CUTS_RANGE[1]
        n_bands = random.randint(STRIPE_CUTS_RANGE[0], max_bands)
        placed: list[tuple[int, int, int, int]] = []
        for _ in range(n_bands):
            height = random.randint(*BLACK_BAND_HEIGHT_RANGE)
            side_margin = random.randint(*SIDE_MARGIN_RANGE)
            if H - 2 * CUT_MARGIN - height < 0 or W - 2 * side_margin < MIN_BLACK_REGION // height:
                continue
            for _ in range(30):
                r0 = random.randint(CUT_MARGIN, H - CUT_MARGIN - height)
                r1 = r0 + height
                if any(not (r1 + 2 <= pr0 or r0 >= pr1 + 2) for pr0, pr1, _, _ in placed):
                    continue
                c0 = side_margin
                c1 = W - side_margin
                grid[r0:r1, c0:c1] = 0
                placed.append((r0, r1, c0, c1))
                break

        if not placed:
            side_margin = 1
            height = min(max(BLACK_BAND_HEIGHT_RANGE[0], H // 3), H - 2 * CUT_MARGIN)
            r0 = (H - height) // 2
            grid[r0 : r0 + height, side_margin : W - side_margin] = 0

    elif style == "rect_inset":
        r0 = random.randint(INSET_MARGIN, H // 3)
        r1 = random.randint(2 * H // 3, H - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, W // 3)
        c1 = random.randint(2 * W // 3, W - CUT_MARGIN)
        grid[r0 : r1 + 1, c0 : c1 + 1] = 0
        if (
            r1 - r0 > INNER_INSET_MIN_GAP
            and c1 - c0 > INNER_INSET_MIN_GAP
            and random.random() < INNER_INSET_PROB
        ):
            ir0 = random.randint(r0 + 2, r0 + (r1 - r0) // 3)
            ir1 = random.randint(r0 + 2 * (r1 - r0) // 3, r1 - 2)
            ic0 = random.randint(c0 + 2, c0 + (c1 - c0) // 3)
            ic1 = random.randint(c0 + 2 * (c1 - c0) // 3, c1 - 2)
            grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = 1

    else:
        r0 = random.randint(INSET_MARGIN, H // 3)
        r1 = random.randint(2 * H // 3, H - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, W // 3)
        c1 = random.randint(2 * W // 3, W - CUT_MARGIN)
        thickness = random.randint(
            3, min(5, max(3, (r1 - r0 + 1) // 2), max(3, (c1 - c0 + 1) // 2))
        )
        corner = random.choice(["tl", "tr", "bl", "br"])
        if corner == "tl":
            grid[r0 : r1 + 1, c0 : c0 + thickness] = 0
            grid[r0 : r0 + thickness, c0 : c1 + 1] = 0
        elif corner == "tr":
            grid[r0 : r1 + 1, c1 - thickness + 1 : c1 + 1] = 0
            grid[r0 : r0 + thickness, c0 : c1 + 1] = 0
        elif corner == "bl":
            grid[r0 : r1 + 1, c0 : c0 + thickness] = 0
            grid[r1 - thickness + 1 : r1 + 1, c0 : c1 + 1] = 0
        else:
            grid[r0 : r1 + 1, c1 - thickness + 1 : c1 + 1] = 0
            grid[r1 - thickness + 1 : r1 + 1, c0 : c1 + 1] = 0

    return grid


def _find_interior_black(grid: np.ndarray) -> list[tuple[int, int]]:
    """Strict black interior pixels: all four neighbors are in-bounds and black."""
    black = grid == 0
    eroded = black.copy()
    eroded[:-1, :] &= black[1:, :]
    eroded[1:, :] &= black[:-1, :]
    eroded[:, :-1] &= black[:, 1:]
    eroded[:, 1:] &= black[:, :-1]
    eroded[0, :] = False
    eroded[-1, :] = False
    eroded[:, 0] = False
    eroded[:, -1] = False
    rs, cs = np.where(eroded)
    coords = list(zip(rs.tolist(), cs.tolist()))
    random.shuffle(coords)
    return coords


def _place_defects(
    grid: np.ndarray, interior_black: list[tuple[int, int]]
) -> tuple[set[tuple[int, int]], int]:
    """Place 2-9 small blue defects inside strict black interiors."""
    H, W = grid.shape
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    defect_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_black)
    defects_placed = 0

    for _ in range(n_defects):
        if not interior_black:
            break

        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
        for _try in range(DEFECT_PLACE_TRIES):
            if not interior_black:
                break

            seed_r, seed_c = random.choice(interior_black)
            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    if _nearby(nr, nc, DEFECT_CLEARANCE, defect_pixels):
                        continue
                    defect.append((nr, nc))
                    if len(defect) >= defect_size:
                        break

            defect_set = set(defect)
            valid = len(defect_set) == defect_size
            for r, c in defect:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < H and 0 <= cc < W):
                        valid = False
                        break
                    if (rr, cc) in defect_set:
                        continue
                    if grid[rr, cc] != 0 or (rr, cc) in defect_pixels:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                defect_pixels.update(defect_set)
                defects_placed += 1
                remove = _exclusion_zone(list(defect_set), DEFECT_CLEARANCE)
                interior_black[:] = [p for p in interior_black if p not in remove]
                interior_set -= remove
                break

    return defect_pixels, defects_placed


def _paint_defects(grid: np.ndarray, defect_pixels: set[tuple[int, int]]) -> np.ndarray:
    output = grid.copy()
    for r, c in defect_pixels:
        output[r, c] = 7
    return output


def _is_clean_combo_input(
    grid: np.ndarray, defect_pixels: set[tuple[int, int]]
) -> bool:
    if not defect_pixels:
        return False
    if np.any(grid[0, :] != 1) or np.any(grid[-1, :] != 1):
        return False
    if np.any(grid[:, 0] != 1) or np.any(grid[:, -1] != 1):
        return False

    for r, c in defect_pixels:
        if int(grid[r, c]) != 1:
            return False

    found = _find_enclosed_blue_defects(grid)
    found_pixels = {p for comp in found for p in comp}
    if found_pixels != defect_pixels:
        return False

    for comp in components_mask(grid == 1, connectivity=4):
        pts = {(int(r), int(c)) for r, c in comp}
        if len(pts) <= 3 and not pts.issubset(defect_pixels):
            return False

    return True


def _valid_partition(grid: np.ndarray) -> bool:
    blue_comps = components_mask(grid == 1, connectivity=4)
    black_comps = components_mask(grid == 0, connectivity=4)
    if not blue_comps or not black_comps:
        return False
    if any(len(comp) < MIN_BLUE_REGION for comp in blue_comps):
        return False
    if any(len(comp) < MIN_BLACK_REGION for comp in black_comps):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    grid = _build_partition(H, W)
    if not _valid_partition(grid):
        return None

    interior_black = _find_interior_black(grid)
    defect_pixels, defects_placed = _place_defects(grid, interior_black)
    if defects_placed < MIN_DEFECTS:
        return None

    inp = grid.copy()
    for r, c in defect_pixels:
        inp[r, c] = 1

    output = _paint_defects(inp, defect_pixels)

    try:
        if _is_clean_combo_input(inp, defect_pixels) and np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
