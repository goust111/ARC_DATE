# Combo variant of 71e489b6 (index 9)
# combo_spec: D7/2+D12/0
# Name: 光晕仅覆写蓝色像素+缺陷也涂橙色
from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import components_mask


# Design analysis for the fused variant
# a) Concrete trace:
#    - Original isolated 1-cell defect inside solid blue: output was an orange 3x3 ring
#      with a black center. Fused rule makes the center orange too, so that same local
#      pattern becomes a solid orange 3x3 block.
#    - Original leaf pixel sticking out from a large black region: its halo could repaint
#      both nearby blue and nearby black cells orange, while the leaf itself stayed black.
#      Fused rule still marks the leaf center orange, but only those halo cells that were
#      blue in the absorbed base grid may turn orange; neighboring black body cells stay
#      black, creating a visibly clipped halo.
#    - The two edits therefore interact: isolated defects show the D12 center-fill change,
#      while leaves expose the D7 conditional-overwrite change.
# b) Complexity calibration:
#    - The underlying object vocabulary stays the same as the base task: small black defects,
#      leaf tips of large black components, and absorbable blue spurs.
#    - To keep the new rule observable from few examples, the generator forces at least one
#      leaf in every sample, so the clipped-halo behavior is repeatedly demonstrated.
# c) Hidden assumptions review:
#    - STEP 0 still assumes inputs only use colors 0/1 and that spur absorption happens
#      before target detection; the generator preserves both conditions.
#    - Target detection still relies on "small black component" and "leaf of a larger black
#      component." That logic remains valid because the fusion changes rendering only.
#    - The original paint loop assumed every halo write may overwrite black pixels and that
#      targets are restored to black afterwards. Both assumptions are false now: painting
#      must consult the absorbed base color, and there is no final black restoration.
# d) Placement legality:
#    - Defects are placed strictly inside blue interiors and kept Chebyshev-2 apart.
#    - Leaves are placed one pixel at a time on blue cells adjacent to large black regions,
#      also kept apart from other targets so their orange marks remain readable.
#    - Spurs are inserted only on cells that STEP 0 deterministically absorbs back to black.
# e) Visual observability:
#    - Every sample contains multiple ordinary defects plus at least one leaf, so both
#      "center turns orange" and "halo stops at black cells" can be inferred by eye.
# f) Visual cleanliness:
#    - Targets keep the original small sizes and are spaced out; clipped halos reduce orange
#      spill inside black masses, which helps separate marked anomalies from region bodies.
# g) Rule inferability:
#    - Only one output color is introduced. Its behavior is explained structurally: orange
#      covers the defect itself and expands one Chebyshev step outward, but only across blue.

HOLE = 0
BLUE = 1
ORANGE = 7


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
    """Apply the original blue-spur absorption rule."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    base = g.copy()
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            black_neighbors = 0
            oob = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == HOLE:
                    black_neighbors += 1
            if oob >= 2:
                black_neighbors += oob
            if black_neighbors >= 3:
                base[r, c] = HOLE
    return base


def _collect_targets(base: np.ndarray) -> set[tuple[int, int]]:
    """Collect all black pixels that should be marked orange in the fused output."""
    H, W = base.shape
    targets: set[tuple[int, int]] = set()
    comps = components_mask(base == HOLE, connectivity=4)
    for coords in comps:
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            targets.update(pts)
            continue
        for r, c in pts:
            black_neighbors = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == HOLE:
                    black_neighbors += 1
            if black_neighbors <= 1:
                targets.add((r, c))
    return targets


def _build_output(base: np.ndarray, targets: set[tuple[int, int]]) -> np.ndarray:
    """Paint orange on each target plus blue cells in its Chebyshev-1 neighborhood."""
    H, W = base.shape
    out = base.copy()
    for r, c in targets:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (rr, cc) == (r, c) or int(base[rr, cc]) == BLUE:
                    out[rr, cc] = ORANGE
    return out


def _build_original_style_output(base: np.ndarray, targets: set[tuple[int, int]]) -> np.ndarray:
    """Reference renderer for degeneration checks against the base task behavior."""
    H, W = base.shape
    out = base.copy()
    for r, c in targets:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in targets:
                    out[rr, cc] = ORANGE
    for r, c in targets:
        out[r, c] = HOLE
    return out


def _build_d7_only_output(base: np.ndarray, targets: set[tuple[int, int]]) -> np.ndarray:
    """Reference renderer for blue-only halos whose target centers remain black."""
    H, W = base.shape
    out = base.copy()
    for r, c in targets:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == BLUE:
                    out[rr, cc] = ORANGE
    return out


def _build_d12_only_output(base: np.ndarray, targets: set[tuple[int, int]]) -> np.ndarray:
    """Reference renderer for full-overwrite halos whose target centers turn orange."""
    H, W = base.shape
    out = base.copy()
    for r, c in targets:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = ORANGE
    return out


def _count_clipped_black_cells(base: np.ndarray, targets: set[tuple[int, int]]) -> int:
    """Count black non-target cells that lie in a target halo and stay black only in this variant."""
    clipped: set[tuple[int, int]] = set()
    H, W = base.shape
    for r, c in targets:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (rr, cc) not in targets and int(base[rr, cc]) == HOLE:
                    clipped.add((rr, cc))
    return len(clipped)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black anomalies in blue regions with solid-orange centers and blue-only halos.

    Input structure:
    - The canvas is partitioned into large blue (1) and black (0) regions.
    - Small black components of size 1-3 can sit fully inside blue regions, and larger black
      regions can have single-pixel leaf tips poking into blue.
    - Sparse blue spur pixels may also appear inside black areas and are not true defects.

    Rule:
    - First absorb any blue spur that has at least three effective black 4-neighbors, using
      the same corner handling as the base task.
    - Then identify all size-1/2/3 black components, plus every leaf pixel of any larger
      black component, as the target defect pixels.
    - Recolor each target pixel itself orange (7), and also recolor every blue cell in its
      Chebyshev-1 neighborhood orange. Black non-target neighbors are never overwritten.

    Key insight:
    - The marked objects are the same structural anomalies as before, but the rendering rule
      has flipped from "orange ring around a black center" to "solid orange defect with a
      halo that only spreads through blue terrain."
    """
    base = _absorb_spurs(np.array(grid, dtype=int))
    targets = _collect_targets(base)
    return _build_output(base, targets)


# --------------- generator ---------------

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
DEFECT_COUNT_RANGE = (2, 9)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 50
LEAF_COUNT_RANGE = (1, 3)
MIN_LEAVES = 1
SPUR_COUNT_RANGE = (2, 6)
MIN_DEFECTS = 2
MIN_SPURS = 2
MIN_CLIPPED_BLACK_CELLS = 1
MAX_GEN_ATTEMPTS = 300


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    """Return True if any pixel in `ref` is within Chebyshev distance `dist`."""
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(pixels: list[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    """Return the set of coords within Chebyshev distance `dist` of any pixel."""
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((int(r + dr), int(c + dc)))
    return zone


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create an HxW grid partitioned into blue and black regions."""
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cuts = sorted(random.sample(
            range(CUT_MARGIN, H - CUT_MARGIN), min(n_cuts, H - 2 * CUT_MARGIN)))
        boundaries = [0] + cuts + [H]
        first_color = random.randint(0, 1)
        for i in range(len(boundaries) - 1):
            grid[boundaries[i]:boundaries[i + 1], :] = (first_color + i) % 2

    elif style == "rect_inset":
        bg = random.randint(0, 1)
        grid[:, :] = bg
        r0 = random.randint(INSET_MARGIN, H // 3)
        r1 = random.randint(2 * H // 3, H - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, W // 3)
        c1 = random.randint(2 * W // 3, W - CUT_MARGIN)
        grid[r0:r1 + 1, c0:c1 + 1] = 1 - bg
        if (r1 - r0 > INNER_INSET_MIN_GAP and c1 - c0 > INNER_INSET_MIN_GAP
                and random.random() < INNER_INSET_PROB):
            ir0 = random.randint(r0 + 2, r0 + (r1 - r0) // 3)
            ir1 = random.randint(r0 + 2 * (r1 - r0) // 3, r1 - 2)
            ic0 = random.randint(c0 + 2, c0 + (c1 - c0) // 3)
            ic1 = random.randint(c0 + 2 * (c1 - c0) // 3, c1 - 2)
            grid[ir0:ir1 + 1, ic0:ic1 + 1] = bg

    else:
        bg = random.randint(0, 1)
        grid[:, :] = bg
        vc = random.randint(W // 3, 2 * W // 3)
        if random.random() < 0.5:
            grid[:, :vc] = 1 - bg
            hc = random.randint(H // 3, 2 * H // 3)
            grid[:hc, vc:] = 1 - bg
        else:
            grid[:, vc:] = 1 - bg
            hc = random.randint(H // 3, 2 * H // 3)
            grid[hc:, :vc] = 1 - bg

    return grid


def _find_interior_blue(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return blue pixels whose in-bounds 4-neighbors are all blue."""
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
    """Place small black defects strictly inside blue interiors."""
    H, W = grid.shape
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
            seed_r, seed_c = interior_blue[random.randint(0, len(interior_blue) - 1)]

            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = int(seed_r + dr)
                    nc = int(seed_c + dc)
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    ok = True
                    for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rr = int(nr + dr2)
                        cc = int(nc + dc2)
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < H and 0 <= cc < W:
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
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W:
                        if grid[rr, cc] != BLUE or (rr, cc) in defect_pixels:
                            valid = False
                            break
                if not valid:
                    break

            if valid:
                defect_pixels.update(defect)
                defects_placed += 1
                remove = _exclusion_zone(defect, DEFECT_CLEARANCE)
                interior_blue[:] = [p for p in interior_blue if p not in remove]
                interior_set -= remove
                break

    return defect_pixels, defects_placed


def _place_leaves(grid: np.ndarray, defect_pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Place separated leaf protrusions from large black regions into blue regions."""
    H, W = grid.shape
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    black_comps = components_mask(grid == HOLE, connectivity=4)
    large_black = [c for c in black_comps if len(c) > 3]

    for _ in range(n_leaves):
        if not large_black:
            break
        comp_set = set((int(r), int(c)) for r, c in random.choice(large_black))

        candidates: list[tuple[int, int]] = []
        for r, c in comp_set:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if not (0 <= rr < H and 0 <= cc < W) or grid[rr, cc] != BLUE:
                    continue
                if (rr, cc) in defect_pixels or (rr, cc) in leaf_pixels:
                    continue
                black_n = sum(
                    1 for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= rr + dr2 < H and 0 <= cc + dc2 < W
                    and (grid[rr + dr2, cc + dc2] == HOLE or (rr + dr2, cc + dc2) in leaf_pixels)
                )
                if black_n == 1:
                    candidates.append((rr, cc))

        random.shuffle(candidates)
        for lr, lc in candidates:
            if _nearby(lr, lc, DEFECT_CLEARANCE, defect_pixels | leaf_pixels):
                continue
            leaf_pixels.add((lr, lc))
            grid[lr, lc] = HOLE
            break

    return leaf_pixels


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black pixels eligible to become blue spurs that STEP 0 will absorb."""
    H, W = grid.shape
    black = grid == HOLE

    black_neighbors = np.zeros((H, W), dtype=int)
    black_neighbors[:-1, :] += black[1:, :]
    black_neighbors[1:, :] += black[:-1, :]
    black_neighbors[:, :-1] += black[:, 1:]
    black_neighbors[:, 1:] += black[:, :-1]

    oob = np.zeros((H, W), dtype=int)
    oob[0, :] += 1
    oob[-1, :] += 1
    oob[:, 0] += 1
    oob[:, -1] += 1

    effective = black_neighbors.copy()
    effective[oob >= 2] += oob[oob >= 2]

    rs, cs = np.where(black & (effective >= 3))
    candidates = [(int(r), int(c)) for r, c in zip(rs, cs) if (int(r), int(c)) not in exclude]
    random.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input for the fused variant."""
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    grid = _build_partition(H, W)

    for comp in components_mask(grid == BLUE, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in components_mask(grid == HOLE, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return None

    interior_blue = _find_interior_blue(grid)
    defect_pixels, defects_placed = _place_defects(grid, interior_blue)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = HOLE

    leaf_pixels = _place_leaves(grid, defect_pixels)
    if len(leaf_pixels) < MIN_LEAVES:
        return None

    targets = defect_pixels | leaf_pixels
    if _count_clipped_black_cells(grid, targets) < MIN_CLIPPED_BLACK_CELLS:
        return None

    output = _build_output(grid, targets)
    if np.array_equal(output, _build_original_style_output(grid, targets)):
        return None
    if np.array_equal(output, _build_d7_only_output(grid, targets)):
        return None
    if np.array_equal(output, _build_d12_only_output(grid, targets)):
        return None

    inp = grid.copy()
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    candidates = _find_spur_candidates(inp, targets)
    for r, c in candidates[:n_spurs]:
        inp[r, c] = BLUE
    if min(n_spurs, len(candidates)) < MIN_SPURS:
        return None

    try:
        if np.array_equal(main(inp), output):
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
