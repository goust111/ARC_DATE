# Combo variant of 71e489b6 (index 18)
# combo_spec: D2/2+D3/0
# Name: 泛化异常检测+毛刺品红标记

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask

# Design analysis for the combo variant D2/2 + D3/0
# a) Concrete behavior and trace:
#    - D2/2 replaces the old black-only source rule with a broader one: any embedded non-blue component inside blue
#      territory is an anomaly, regardless of whether its color is black, red, or grey.
#    - D3/0 flips the old absorption transform: blue spur pixels in black territory are preserved and recolored
#      magenta instead of being deleted.
#    - Decision matrix:
#      * blue-region + blue pixel -> ordinary foreground unless covered by an orange halo.
#      * blue-region + non-blue pixel whose whole 4-connected component is surrounded by blue -> anomaly center.
#      * black-region + blue spur pixel with >=3 effective black 4-neighbors -> magenta mark, never absorbed.
#      * black-region + ordinary black pixel -> unchanged unless touched by an orange halo.
#    - Small trace:
#         input                 output
#         1 1 1                 7 7 7
#         1 5 1   anomaly ->    7 5 7
#         1 1 1                 7 7 7
#
#         0 0 0                 0 0 0
#         0 1 0   spur ->       0 6 0
#         0 0 0                 0 0 0
# b) Complexity calibration:
#    - The visual language stays compact: orange always means "embedded mismatch in blue", magenta always means
#      "blue spur inside black".
#    - The generator forces at least one black embedded anomaly, at least one colored embedded anomaly, and at least
#      two spurs in every accepted grid so both combo branches are repeatedly visible.
# c) Hidden assumptions audit:
#    - Target centers are no longer assumed black; anomaly centers keep their original color.
#    - Source detection is no longer topology-specific to black components; it is color-agnostic over embedded
#      non-blue components.
#    - Spur handling is separated from anomaly marking so magenta never disappears under the original absorption step.
# d) Placement legality and neatness:
#    - Embedded anomaly components are placed only on strict blue interiors and kept Chebyshev-separated.
#    - Base partitions with a large black island fully enclosed by blue are rejected, so the human-visible rule stays
#      focused on the planted anomaly components rather than macro-region accidents.
#    - Spurs are buffered away from anomaly halos to avoid orange/magenta semantic collisions.


BLUE = 1
BLACK = 0
RED = 2
GREY = 5
MAGENTA = 6
ORANGE = 7

ANOMALY_COLORS = (BLACK, RED, GREY)
COLORED_ANOMALY_COLORS = (RED, GREY)

GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MIN_BLUE_REGION = 30
MIN_BLACK_REGION = 20
PARTITION_STYLES = ["h_stripes", "rect_inset", "l_shape"]
STRIPE_CUTS_RANGE = (1, 3)
CUT_MARGIN = 3
INSET_MARGIN = 2
INNER_INSET_MIN_GAP = 6
INNER_INSET_PROB = 0.5
DEFECT_COUNT_RANGE = (3, 9)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 60
SPUR_COUNT_RANGE = (2, 6)
SPUR_CLEARANCE = 1
SPUR_TARGET_BUFFER = 2
MIN_DEFECTS = 3
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(pixels: set[tuple[int, int]] | list[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((int(r + dr), int(c + dc)))
    return zone


def _effective_black_count(grid: np.ndarray, r: int, c: int) -> int:
    """Count black 4-neighbors, adding out-of-bounds only for corners."""
    h, w = grid.shape
    black_n = 0
    oob = 0
    for rr, cc in _neighbors4(r, c):
        if not (0 <= rr < h and 0 <= cc < w):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_n += 1
    if oob >= 2:
        black_n += oob
    return int(black_n)


def _find_spurs(grid: np.ndarray) -> set[tuple[int, int]]:
    """Blue pixels in black regions that the source task would absorb."""
    h, w = grid.shape
    spurs: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLUE:
                continue
            if _effective_black_count(grid, r, c) >= 3:
                spurs.add((int(r), int(c)))
    return spurs


def _collect_embedded_non_blue_targets(grid: np.ndarray) -> set[tuple[int, int]]:
    """Find all non-blue 4-components whose full external 4-boundary is blue."""
    h, w = grid.shape
    mask = grid != BLUE
    targets: set[tuple[int, int]] = set()

    for coords in components_mask(mask, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        comp_set = set(pts)
        valid = True
        for r, c in pts:
            for rr, cc in _neighbors4(r, c):
                if (rr, cc) in comp_set:
                    continue
                if not (0 <= rr < h and 0 <= cc < w):
                    valid = False
                    break
                if int(grid[rr, cc]) != BLUE:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            targets.update(comp_set)

    return targets


def _paint_output(
    grid: np.ndarray,
    anomaly_targets: set[tuple[int, int]],
    spurs: set[tuple[int, int]],
) -> np.ndarray:
    """Render orange halos around blue-region anomalies and magenta on spurs."""
    h, w = grid.shape
    out = np.array(grid, dtype=int)

    for r, c in anomaly_targets:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in anomaly_targets or (rr, cc) in spurs:
                    continue
                out[rr, cc] = ORANGE

    for r, c in spurs:
        out[r, c] = MAGENTA

    return out


def _collect_anomaly_targets(grid: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return (all anomaly targets, spur marks) for the combo rule."""
    spurs = _find_spurs(grid)
    embedded_targets = _collect_embedded_non_blue_targets(grid)
    return embedded_targets, spurs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark two anomaly families at once: any blue-region color mismatch gets an orange halo, while black-region blue
    spurs stay visible as magenta.

    Input structure:
    - Large blue foreground regions and large black regions partition the board.
    - Inside blue regions there may be small embedded anomaly components of several colors, including black, red,
      or grey.
    - Inside black regions there may also be isolated blue spur pixels that are almost surrounded by black.

    Rule:
    - First identify spur pixels with the original absorption test: blue cells with at least three effective black
      4-neighbors, counting out-of-bounds only for corner cells. These are analysis-only black but output magenta.
    - Then detect every non-blue 4-connected component whose external 4-neighbors are all blue; the whole
      component is an anomaly target regardless of its color.
    - Paint an orange Chebyshev-1 halo around every targeted anomaly cell, keep anomaly centers in their original
      colors, and finally recolor spur pixels magenta.

    Key insight:
    The hidden abstraction is "mark any embedded mismatch inside the blue territory," while handling blue
    mismatches inside black territory differently by preserving and recoloring them.
    """
    g = np.array(grid, dtype=int)
    anomaly_targets, spurs = _collect_anomaly_targets(g)
    return _paint_output(g, anomaly_targets, spurs)


def _build_partition(h: int, w: int) -> np.ndarray:
    """Create an h x w grid partitioned into blue and black macro-regions."""
    grid = np.zeros((h, w), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cuts = sorted(random.sample(range(CUT_MARGIN, h - CUT_MARGIN), min(n_cuts, h - 2 * CUT_MARGIN)))
        bounds = [0] + cuts + [h]
        first_color = random.randint(0, 1)
        for i in range(len(bounds) - 1):
            grid[bounds[i] : bounds[i + 1], :] = (first_color + i) % 2

    elif style == "rect_inset":
        bg = random.randint(0, 1)
        grid[:, :] = bg
        r0 = random.randint(INSET_MARGIN, h // 3)
        r1 = random.randint(2 * h // 3, h - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, w // 3)
        c1 = random.randint(2 * w // 3, w - CUT_MARGIN)
        grid[r0 : r1 + 1, c0 : c1 + 1] = 1 - bg
        if (
            r1 - r0 > INNER_INSET_MIN_GAP
            and c1 - c0 > INNER_INSET_MIN_GAP
            and random.random() < INNER_INSET_PROB
        ):
            ir0 = random.randint(r0 + 2, r0 + (r1 - r0) // 3)
            ir1 = random.randint(r0 + 2 * (r1 - r0) // 3, r1 - 2)
            ic0 = random.randint(c0 + 2, c0 + (c1 - c0) // 3)
            ic1 = random.randint(c0 + 2 * (c1 - c0) // 3, c1 - 2)
            grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg

    else:
        bg = random.randint(0, 1)
        grid[:, :] = bg
        vc = random.randint(w // 3, 2 * w // 3)
        hc = random.randint(h // 3, 2 * h // 3)
        if random.random() < 0.5:
            grid[:, :vc] = 1 - bg
            grid[:hc, vc:] = 1 - bg
        else:
            grid[:, vc:] = 1 - bg
            grid[hc:, :vc] = 1 - bg

    return grid


def _find_interior_blue(grid: np.ndarray) -> list[tuple[int, int]]:
    """Blue cells whose in-bounds 4-neighbors are all blue."""
    blue = grid == BLUE
    eroded = blue.copy()
    eroded[:-1, :] &= blue[1:, :]
    eroded[1:, :] &= blue[:-1, :]
    eroded[:, :-1] &= blue[:, 1:]
    eroded[:, 1:] &= blue[:, :-1]
    rs, cs = np.where(eroded)
    coords = list(zip(rs.tolist(), cs.tolist()))
    random.shuffle(coords)
    return coords


def _place_anomaly_components(
    grid: np.ndarray,
    interior_blue: list[tuple[int, int]],
) -> list[dict]:
    """Place small blue-region anomaly components of colors 0/2/5."""
    h, w = grid.shape
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    colors = [BLACK, random.choice(COLORED_ANOMALY_COLORS)]
    while len(colors) < n_defects:
        colors.append(random.choice(ANOMALY_COLORS))
    random.shuffle(colors)

    occupied: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)
    placed: list[dict] = []

    for color in colors:
        if not interior_blue:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]

        for _ in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = interior_blue[random.randint(0, len(interior_blue) - 1)]
            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, occupied):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = _neighbors4(seed_r, seed_c)
                random.shuffle(dirs)
                for nr, nc in dirs:
                    if (nr, nc) not in interior_set or (nr, nc) in occupied:
                        continue
                    ok = True
                    for rr, cc in _neighbors4(nr, nc):
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < h and 0 <= cc < w:
                            if int(grid[rr, cc]) != BLUE or (rr, cc) in occupied:
                                ok = False
                                break
                    if ok and not _nearby(nr, nc, DEFECT_CLEARANCE, occupied):
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for rr, cc in _neighbors4(r, c):
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < h and 0 <= cc < w:
                        if int(grid[rr, cc]) != BLUE or (rr, cc) in occupied:
                            valid = False
                            break
                if not valid:
                    break

            if not valid:
                continue

            occupied.update(defect_set)
            for r, c in defect_set:
                grid[r, c] = color
            placed.append({"pixels": defect_set, "color": color})
            remove = _exclusion_zone(defect_set, DEFECT_CLEARANCE)
            interior_blue[:] = [p for p in interior_blue if p not in remove]
            interior_set -= remove
            break

    return placed


def _find_spur_candidates(grid: np.ndarray, forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black pixels eligible to become blue spur marks."""
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK:
                continue
            if (r, c) in forbidden:
                continue
            probe = np.array(grid, dtype=int)
            probe[r, c] = BLUE
            if _effective_black_count(probe, r, c) >= 3:
                candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _baseline_output(grid: np.ndarray) -> np.ndarray:
    """Original 71e489b6 behavior, used only for generator anti-degeneracy checks."""
    g = np.array(grid, dtype=int)
    h, w = g.shape
    base = g.copy()
    for r in range(h):
        for c in range(w):
            if int(base[r, c]) != BLUE:
                continue
            if _effective_black_count(base, r, c) >= 3:
                base[r, c] = BLACK

    preserve: set[tuple[int, int]] = set()
    comps = components_mask(base == BLACK, connectivity=4)
    for coords in comps:
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        comp_set = set(pts)
        for r, c in pts:
            degree = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) in comp_set:
                    degree += 1
            if degree <= 1:
                preserve.add((int(r), int(c)))

    out = base.copy()
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in preserve:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    grid = _build_partition(h, w)

    for comp in components_mask(grid == BLUE, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in components_mask(grid == BLACK, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return None
    if _collect_embedded_non_blue_targets(grid):
        return None

    interior_blue = _find_interior_blue(grid)
    anomalies = _place_anomaly_components(grid, interior_blue)
    if len(anomalies) < MIN_DEFECTS:
        return None

    colors_used = {item["color"] for item in anomalies}
    if BLACK not in colors_used or not (colors_used & set(COLORED_ANOMALY_COLORS)):
        return None

    target_pixels = set().union(*(item["pixels"] for item in anomalies))

    forbidden = _exclusion_zone(target_pixels, SPUR_TARGET_BUFFER)
    candidates = _find_spur_candidates(grid, forbidden)
    if len(candidates) < MIN_SPURS:
        return None

    inp = np.array(grid, dtype=int)
    spurs: set[tuple[int, int]] = set()
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    for r, c in candidates:
        if len(spurs) >= n_spurs:
            break
        if _nearby(r, c, SPUR_CLEARANCE, spurs):
            continue
        inp[r, c] = BLUE
        spurs.add((int(r), int(c)))

    if len(spurs) < MIN_SPURS:
        return None

    expected = _paint_output(inp, target_pixels, spurs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    combo_delta = int(np.count_nonzero(actual != _baseline_output(inp)))
    colored_target_cells = sum(
        len(item["pixels"]) for item in anomalies if item["color"] in COLORED_ANOMALY_COLORS
    )
    if combo_delta <= 0 or colored_target_cells <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
