# Combo variant of 71e489b6 (index 5)
# combo_spec: D3/0+D5/0
# Name: 毛刺品红标记+缺陷类型分色
from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask  # noqa: E402


# Design analysis for the fused rule:
# - Concrete trace: in the source task, a blue pixel with >=3 effective black 4-neighbors
#   is first absorbed into black, then black components are split into tiny defects (size <= 3)
#   and leaf pixels of large components, and all detected targets get orange halos.
#   Here that first blue pixel is no longer "deleted" in the final picture. We still detect it
#   with the same criterion, but only on a hidden repaired grid used for geometry; the visible
#   output turns that pixel magenta as a single point. Example: a black basin with one injected
#   blue cell now yields a magenta point, while a one-cell black hole inside blue still yields an
#   orange 3x3 ring, and a one-cell black leaf on a large black region yields a magenta 3x3 ring.
# - Complexity calibration: only three visible behaviors remain, each supported repeatedly:
#   orange ring = tiny internal black defect, magenta ring = leaf of a large black mass,
#   magenta point = blue spur in black. The color count stays low (0/1/6/7), and the two magenta
#   meanings are separated by shape so the fusion is visible but not underconstrained.
# - Implicit assumptions audited:
#   1. The original STEP 0 repaired black-region topology before component analysis. That
#      assumption still matters, so solver logic now records spur coordinates while repairing a
#      private `base` grid instead of letting the visible output topology drive detection.
#   2. "Small black component" still means size <= 3 on the repaired grid; otherwise inserted
#      blue spurs could incorrectly fragment large black regions into fake small defects.
#   3. Leaf detection still depends on degree <= 1 inside a large black 4-component; only the
#      halo color changes from orange to magenta.
#   4. Halo painting must not recolor the target black pixels themselves; spur points are painted
#      last so they remain single magenta markers even if a halo footprint passes over them.
# - Placement legality and cleanliness: generator keeps defects, leaves, and spurs separated by
#   Chebyshev clearance so halos rarely overlap, forbids accidental spur-like blue pixels in the
#   clean base grid, and validates with the solver oracle.
# - Visual observability: every generated sample is forced to contain at least one orange defect,
#   one magenta leaf halo, and at least two magenta spur points, so a human can infer both the
#   color split and the shape split from only a few examples.


BLACK = 0
BLUE = 1
MAGENTA = 6
ORANGE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _effective_black_neighbors(grid: np.ndarray, r: int, c: int) -> int:
    """Count black 4-neighbors; corner out-of-bounds sides also count as black."""
    h, w = grid.shape
    black_n = 0
    oob = 0
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if not (0 <= rr < h and 0 <= cc < w):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_n += 1
    if oob >= 2:
        black_n += oob
    return black_n


def _extract_spurs(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    """Return (repaired_grid, spur_points) using the original in-place absorption scan."""
    base = np.array(grid, dtype=int)
    spur_points: set[tuple[int, int]] = set()
    h, w = base.shape
    for r in range(h):
        for c in range(w):
            if int(base[r, c]) != BLUE:
                continue
            if _effective_black_neighbors(base, r, c) >= 3:
                spur_points.add((int(r), int(c)))
                base[r, c] = BLACK
    return base, spur_points


def _classify_black_targets(base: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Split repaired black pixels into internal defects and leaf pixels."""
    internal_defects: set[tuple[int, int]] = set()
    leaf_pixels: set[tuple[int, int]] = set()

    for coords in components_mask(base == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            internal_defects.update(pts)
            continue
        comp_set = set(pts)
        for r, c in pts:
            deg = sum((r + dr, c + dc) in comp_set for dr, dc in DIR4)
            if deg <= 1:
                leaf_pixels.add((r, c))
    return internal_defects, leaf_pixels


def _paint_halo(
    out: np.ndarray,
    centers: set[tuple[int, int]],
    color: int,
    blocked: set[tuple[int, int]],
) -> None:
    """Paint a Chebyshev-1 halo but leave blocked center pixels unchanged."""
    h, w = out.shape
    for r, c in centers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in blocked:
                    out[rr, cc] = color


def _expected_output(
    input_grid: np.ndarray,
    defect_pixels: set[tuple[int, int]],
    leaf_pixels: set[tuple[int, int]],
    spur_points: set[tuple[int, int]],
) -> np.ndarray:
    """Paint orange halos for internal defects, magenta halos for leaves, and magenta spur dots."""
    out = np.array(input_grid, dtype=int)
    blocked = set(defect_pixels) | set(leaf_pixels)
    _paint_halo(out, defect_pixels, ORANGE, blocked)
    _paint_halo(out, leaf_pixels, MAGENTA, blocked)
    for r, c in blocked:
        out[r, c] = BLACK
    for r, c in spur_points:
        out[r, c] = MAGENTA
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark two different kinds of black-region anomalies while also preserving blue spurs as magenta dots.

    Input structure:
    - Blue (1) foreground regions and black (0) regions partition the grid.
    - Tiny black defects may sit fully inside blue regions, and large black regions may have dangling black leaf pixels.
    - Blue spur pixels can also appear inside black regions at spots that the source task would have absorbed.

    Rule:
    - First detect every blue pixel with at least three effective black 4-neighbors (corner out-of-bounds counts as black), but use it only as a recorded spur marker while repairing a hidden black base for later analysis.
    - On that repaired base, treat black components of size at most three as internal defects, and treat degree-0/1 pixels of larger black components as leaves.
    - Paint an orange (7) Chebyshev-1 halo around internal defects, a magenta (6) Chebyshev-1 halo around leaves, keep all target black pixels black, and finally paint each recorded spur itself as a single magenta point.

    Key insight: The same local geometry now drives two visibly different outputs: type decides halo color for black anomalies, while the old "absorbed" blue spur is no longer merged away and instead survives as a magenta singleton.
    """
    g = np.array(grid, dtype=int)
    base, spur_points = _extract_spurs(g)
    defect_pixels, leaf_pixels = _classify_black_targets(base)
    return _expected_output(g, defect_pixels, leaf_pixels, spur_points)


# --------------- generator ---------------

# Tuned to keep the three visible behaviors present and visually separated.
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
DEFECT_COUNT_RANGE = (2, 8)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 50
LEAF_COUNT_RANGE = (1, 4)
LEAF_CLEARANCE = 3
SPUR_COUNT_RANGE = (2, 5)
SPUR_CLEARANCE = 2
MIN_DEFECTS = 2
MIN_LEAVES = 1
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 500


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
                zone.add((r + dr, c + dc))
    return zone


def _build_partition(h: int, w: int) -> np.ndarray:
    """Create a grid split into large blue and black regions."""
    grid = np.zeros((h, w), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cut_candidates = list(range(CUT_MARGIN, h - CUT_MARGIN))
        cuts = sorted(random.sample(cut_candidates, min(n_cuts, len(cut_candidates))))
        boundaries = [0] + cuts + [h]
        first_color = random.randint(0, 1)
        for i in range(len(boundaries) - 1):
            grid[boundaries[i] : boundaries[i + 1], :] = (first_color + i) % 2

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
    """Blue pixels whose in-bounds 4-neighbors are all blue."""
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


def _place_defects(grid: np.ndarray, interior_blue: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], int]:
    """Place 2-8 isolated small black defects strictly inside blue interiors."""
    h, w = grid.shape
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    defect_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)
    defects_placed = 0

    for _ in range(n_defects):
        if not interior_blue:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
        for _try in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = random.choice(interior_blue)
            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = DIR4.copy()
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    ok = True
                    for dr2, dc2 in DIR4:
                        rr = nr + dr2
                        cc = nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < h and 0 <= cc < w:
                            if grid[rr, cc] != BLUE or (rr, cc) in defect_pixels:
                                ok = False
                                break
                    if ok and not _nearby(nr, nc, DEFECT_CLEARANCE, defect_pixels):
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < h and 0 <= cc < w:
                        if grid[rr, cc] != BLUE or (rr, cc) in defect_pixels:
                            valid = False
                            break
                if not valid:
                    break

            if valid:
                defect_pixels.update(defect_set)
                defects_placed += 1
                remove = _exclusion_zone(defect_set, DEFECT_CLEARANCE)
                interior_blue[:] = [p for p in interior_blue if p not in remove]
                interior_set -= remove
                break

    return defect_pixels, defects_placed


def _place_leaves(grid: np.ndarray, defect_pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Place spaced single-cell black leaves protruding from large black regions into blue."""
    h, w = grid.shape
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    while len(leaf_pixels) < n_leaves:
        candidates: list[tuple[int, int]] = []
        for comp in components_mask(grid == BLACK, connectivity=4):
            if len(comp) <= 3:
                continue
            comp_set = {(int(r), int(c)) for r, c in comp}
            for r, c in comp_set:
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if grid[rr, cc] != BLUE:
                        continue
                    if _nearby(rr, cc, LEAF_CLEARANCE, defect_pixels | leaf_pixels):
                        continue
                    black_n = 0
                    for dr2, dc2 in DIR4:
                        r2 = rr + dr2
                        c2 = cc + dc2
                        if 0 <= r2 < h and 0 <= c2 < w and grid[r2, c2] == BLACK:
                            black_n += 1
                    if black_n == 1:
                        candidates.append((rr, cc))
        if not candidates:
            break
        random.shuffle(candidates)
        lr, lc = candidates[0]
        leaf_pixels.add((lr, lc))
        grid[lr, lc] = BLACK

    return leaf_pixels


def _find_spur_candidates(
    clean_grid: np.ndarray,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Black cells that can be flipped to blue and then re-detected as spurs."""
    h, w = clean_grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(clean_grid[r, c]) != BLACK:
                continue
            if (r, c) in forbidden:
                continue
            temp = clean_grid.copy()
            temp[r, c] = BLUE
            if _effective_black_neighbors(temp, r, c) >= 3:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _choose_spurs(
    clean_grid: np.ndarray,
    defect_pixels: set[tuple[int, int]],
    leaf_pixels: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    """Pick isolated spur locations away from all halos and verify exact re-detection."""
    forbidden = _exclusion_zone(defect_pixels | leaf_pixels, 2)
    candidates = _find_spur_candidates(clean_grid, forbidden)
    chosen: set[tuple[int, int]] = set()
    target_count = random.randint(*SPUR_COUNT_RANGE)

    for r, c in candidates:
        if _nearby(r, c, SPUR_CLEARANCE, chosen):
            continue
        chosen.add((r, c))
        if len(chosen) >= target_count:
            break

    if len(chosen) < MIN_SPURS:
        return None

    inp = clean_grid.copy()
    for r, c in chosen:
        inp[r, c] = BLUE
    repaired, detected = _extract_spurs(inp)
    if not np.array_equal(repaired, clean_grid):
        return None
    if detected != chosen:
        return None
    return chosen


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to build a valid fused-rule input."""
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = _build_partition(h, w)

    for comp in components_mask(grid == BLUE, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in components_mask(grid == BLACK, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return None

    interior_blue = _find_interior_blue(grid)
    defect_pixels, defects_placed = _place_defects(grid, interior_blue)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, defect_pixels)
    if len(leaf_pixels) < MIN_LEAVES:
        return None

    clean_grid = grid.copy()
    repaired, accidental_spurs = _extract_spurs(clean_grid)
    if accidental_spurs:
        return None
    if not np.array_equal(repaired, clean_grid):
        return None

    spur_points = _choose_spurs(clean_grid, defect_pixels, leaf_pixels)
    if spur_points is None:
        return None

    inp = clean_grid.copy()
    for r, c in spur_points:
        inp[r, c] = BLUE

    output = _expected_output(inp, defect_pixels, leaf_pixels, spur_points)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
