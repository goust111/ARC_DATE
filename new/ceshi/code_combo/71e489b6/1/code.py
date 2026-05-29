# Combo variant of 71e489b6 (index 1)
# combo_spec: D1/0+D5/0
# Name: 十字光晕+类型分色
from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_ROOT = ROOT / "solvers"
if str(SOLVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLVER_ROOT))

from arc_utils import components_mask


BLACK = 0
BLUE = 1
MAGENTA = 6
ORANGE = 7

CROSS_OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0)]


# Design analysis for the fused variant.
# a) Concrete behavior trace:
#    - Original: a 1-cell black hole inside blue becomes a 3x3 orange ring; a 1-pixel black
#      leaf attached to a large black area gets the same 3x3 orange ring.
#    - Fused: the same hole paints only its orthogonal neighbors in orange, while the leaf
#      paints only its orthogonal neighbors in magenta. Both centers stay black, and blue
#      spurs in black regions are still absorbed before any marking.
#    - Example around a single defect at the center:
#        input            original output      fused output
#        111              171                 171
#        101              707                 101
#        111              171                 171
#      The geometric cue changes from square halo to cross, and leaf-vs-hole color now differs.
# b) Complexity calibration:
#    - Only two target classes are used: small internal components and leaf pixels of large
#      black components. This is enough to support the new color split without introducing
#      extra ambiguous categories.
#    - The generator forces both classes to appear and keeps target footprints apart, so the
#      cross geometry and the type-color mapping are both repeated within a small set of samples.
# c) Implicit assumptions reviewed:
#    - Spur absorption assumes blue noise cells have at least three effective black neighbors;
#      generation preserves that exact condition.
#    - Internal defects are still identified solely by black 4-components of size <= 3; the
#      generator keeps them strictly inside blue so they do not merge with large black regions.
#    - Leaf detection assumes large black components can contain degree-0/1 tips; the generator
#      creates attached 1-pixel protrusions and prevents adjacent leaves that would raise degree.
#    - Cross painting assumes target centers should remain black and should not be overwritten
#      by another target's halo; the painter excludes all target centers, and the generator keeps
#      target footprints disjoint for visual cleanliness.
# d) Placement legality:
#    - Defects are sampled only from blue interior pixels whose non-defect 4-neighbors remain blue.
#    - Leaves are sampled only from blue cells adjacent to a large black component, while checking
#      that they have exactly one black attachment and do not collide with reserved footprints.
#    - Spurs are added only at black cells that the solver's absorption pass will deterministically
#      flip back to black.
# e) Visual observability:
#    - The fused rule is human-visible because shape and color both change: crosses replace squares,
#      and magenta appears only around boundary tips while orange stays around internal holes.
# f) Visual cleanliness:
#    - Target centers and halo footprints are separated with a reserved-footprint test, which avoids
#      dense overlap where orange and magenta crosses would become hard to attribute.
# g) Color-rule inferability:
#    - Only two marking colors appear, and their mapping is anchored to a structural distinction
#      visible in the input: isolated internal holes vs tips sticking out of a large black mass.
# Additional solver/generator contract:
#    - `main()` assumes at least one internal defect and one leaf may appear; `generate_input()`
#      guarantees that, so the color split is not left unsupported by the training examples.


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Apply the original spur-absorption pass in-place order."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            black_nb = 0
            oob = 0
            for dr, dc in CROSS_OFFSETS:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == BLACK:
                    black_nb += 1
            if oob >= 2:
                black_nb += oob
            if black_nb >= 3:
                base[r, c] = BLACK
    return base


def _classify_targets(base: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return (internal_defects, leaf_pixels) after spur absorption."""
    H, W = base.shape
    internal: set[tuple[int, int]] = set()
    leaves: set[tuple[int, int]] = set()

    for coords in components_mask(base == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            internal.update(pts)
            continue
        for r, c in pts:
            deg = 0
            for dr, dc in CROSS_OFFSETS:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == BLACK:
                    deg += 1
            if deg <= 1:
                leaves.add((r, c))
    return internal, leaves


def _paint_crosses(base: np.ndarray, internal: set[tuple[int, int]], leaves: set[tuple[int, int]]) -> np.ndarray:
    """Paint orange or magenta crosses while preserving all target centers as black."""
    out = np.array(base, dtype=int)
    preserve = set(internal) | set(leaves)
    for color, pixels in ((ORANGE, internal), (MAGENTA, leaves)):
        for r, c in pixels:
            for dr, dc in CROSS_OFFSETS:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and (rr, cc) not in preserve:
                    out[rr, cc] = color
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black anomalies in blue territory with type-specific crosses instead of square halos.

    Input structure:
    - Blue (1) foreground regions sit beside large black (0) regions.
    - Blue areas may contain tiny black internal defects, and large black regions may poke 1-cell leaf tips into blue.
    - Black regions may also contain thin blue spur pixels that should disappear before marking.

    Rule:
    - First absorb any blue spur whose orthogonal neighborhood is mostly black, using the original corner treatment.
    - Then detect two kinds of black targets: small 4-connected black components of size <= 3, and degree-0/1 leaf pixels on larger black components.
    - Paint a 4-neighbor cross around each target center while leaving the center black: internal defects use orange (7), leaf pixels use magenta (6).

    Key insight: The puzzle still isolates structural black irregularities, but the output now encodes both geometry and defect type: isolated holes get orange crosses, boundary tips get magenta crosses.
    """
    base = _absorb_blue_spurs(np.array(grid, dtype=int))
    internal, leaves = _classify_targets(base)
    return _paint_crosses(base, internal, leaves)


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
DEFECT_COUNT_RANGE = (3, 8)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 80
LEAF_COUNT_RANGE = (1, 4)
LEAF_PLACE_TRIES = 80
SPUR_COUNT_RANGE = (2, 6)
MIN_DEFECTS = 2
MIN_LEAVES = 1
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 400


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _cross_footprint(pixels: set[tuple[int, int]] | list[tuple[int, int]]) -> set[tuple[int, int]]:
    footprint: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr, dc in CROSS_OFFSETS:
            footprint.add((r + dr, c + dc))
    return footprint


def _build_partition(H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        candidates = range(CUT_MARGIN, H - CUT_MARGIN)
        cuts = sorted(random.sample(candidates, min(n_cuts, H - 2 * CUT_MARGIN)))
        boundaries = [0] + cuts + [H]
        first_color = random.randint(0, 1)
        for i in range(len(boundaries) - 1):
            grid[boundaries[i] : boundaries[i + 1], :] = (first_color + i) % 2

    elif style == "rect_inset":
        bg = random.randint(0, 1)
        grid[:, :] = bg
        r0 = random.randint(INSET_MARGIN, H // 3)
        r1 = random.randint(2 * H // 3, H - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, W // 3)
        c1 = random.randint(2 * W // 3, W - CUT_MARGIN)
        grid[r0 : r1 + 1, c0 : c1 + 1] = 1 - bg
        if (r1 - r0 > INNER_INSET_MIN_GAP and c1 - c0 > INNER_INSET_MIN_GAP
                and random.random() < INNER_INSET_PROB):
            ir0 = random.randint(r0 + 2, r0 + (r1 - r0) // 3)
            ir1 = random.randint(r0 + 2 * (r1 - r0) // 3, r1 - 2)
            ic0 = random.randint(c0 + 2, c0 + (c1 - c0) // 3)
            ic1 = random.randint(c0 + 2 * (c1 - c0) // 3, c1 - 2)
            grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg

    else:  # l_shape
        bg = random.randint(0, 1)
        grid[:, :] = bg
        vc = random.randint(W // 3, 2 * W // 3)
        hc = random.randint(H // 3, 2 * H // 3)
        if random.random() < 0.5:
            grid[:, :vc] = 1 - bg
            grid[:hc, vc:] = 1 - bg
        else:
            grid[:, vc:] = 1 - bg
            grid[hc:, :vc] = 1 - bg

    return grid


def _find_interior_blue(grid: np.ndarray) -> list[tuple[int, int]]:
    H, W = grid.shape
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


def _footprint_is_clear(
    pixels: list[tuple[int, int]] | set[tuple[int, int]],
    reserved_centers: set[tuple[int, int]],
    reserved_halo: set[tuple[int, int]],
) -> bool:
    centers = set(pixels)
    halo = _cross_footprint(centers)
    if centers & reserved_centers:
        return False
    if centers & reserved_halo:
        return False
    if halo & reserved_centers:
        return False
    if halo & reserved_halo:
        return False
    return True


def _place_defects(
    grid: np.ndarray,
    interior_blue: list[tuple[int, int]],
    reserved_centers: set[tuple[int, int]],
    reserved_halo: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], int]:
    H, W = grid.shape
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    defect_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)
    defects_placed = 0

    for _ in range(n_defects):
        if not interior_blue:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
        for _ in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = random.choice(interior_blue)
            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = CROSS_OFFSETS[:]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    if _nearby(nr, nc, DEFECT_CLEARANCE, defect_pixels):
                        continue
                    ok = True
                    for dr2, dc2 in CROSS_OFFSETS:
                        rr = nr + dr2
                        cc = nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < H and 0 <= cc < W:
                            if grid[rr, cc] != BLUE or (rr, cc) in defect_pixels:
                                ok = False
                                break
                    if ok:
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for dr, dc in CROSS_OFFSETS:
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W:
                        if grid[rr, cc] != BLUE or (rr, cc) in defect_pixels:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                continue
            if not _footprint_is_clear(defect, reserved_centers, reserved_halo):
                continue

            defect_pixels.update(defect)
            defects_placed += 1
            reserved_centers.update(defect_set)
            reserved_halo.update(_cross_footprint(defect_set))
            remove = set()
            for r, c in defect:
                for dr in range(-DEFECT_CLEARANCE, DEFECT_CLEARANCE + 1):
                    for dc in range(-DEFECT_CLEARANCE, DEFECT_CLEARANCE + 1):
                        remove.add((r + dr, c + dc))
            interior_blue[:] = [p for p in interior_blue if p not in remove]
            interior_set -= remove
            break

    return defect_pixels, defects_placed


def _place_leaves(
    grid: np.ndarray,
    defect_pixels: set[tuple[int, int]],
    reserved_centers: set[tuple[int, int]],
    reserved_halo: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    H, W = grid.shape
    desired = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    for _ in range(LEAF_PLACE_TRIES):
        if len(leaf_pixels) >= desired:
            break
        black_comps = components_mask(grid == BLACK, connectivity=4)
        large_black = [set((int(r), int(c)) for r, c in comp) for comp in black_comps if len(comp) > 3]
        if not large_black:
            break

        candidates: list[tuple[int, int]] = []
        random.shuffle(large_black)
        for comp_set in large_black:
            for r, c in list(comp_set):
                dirs = CROSS_OFFSETS[:]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < H and 0 <= cc < W):
                        continue
                    if grid[rr, cc] != BLUE:
                        continue
                    if (rr, cc) in defect_pixels or (rr, cc) in leaf_pixels:
                        continue
                    black_n = 0
                    for dr2, dc2 in CROSS_OFFSETS:
                        nr = rr + dr2
                        nc = cc + dc2
                        if 0 <= nr < H and 0 <= nc < W:
                            if grid[nr, nc] == BLACK or (nr, nc) in leaf_pixels:
                                black_n += 1
                    if black_n != 1:
                        continue
                    if _nearby(rr, cc, DEFECT_CLEARANCE, defect_pixels):
                        continue
                    if not _footprint_is_clear({(rr, cc)}, reserved_centers, reserved_halo):
                        continue
                    candidates.append((rr, cc))
            if candidates:
                break

        if not candidates:
            break
        lr, lc = random.choice(candidates)
        leaf_pixels.add((lr, lc))
        grid[lr, lc] = BLACK
        reserved_centers.add((lr, lc))
        reserved_halo.update(_cross_footprint({(lr, lc)}))

    return leaf_pixels


def _build_variant_output(
    grid: np.ndarray,
    defect_pixels: set[tuple[int, int]],
    leaf_pixels: set[tuple[int, int]],
) -> np.ndarray:
    return _paint_crosses(grid, defect_pixels, leaf_pixels)


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    H, W = grid.shape
    black = grid == BLACK

    nb = np.zeros((H, W), dtype=int)
    nb[:-1, :] += black[1:, :]
    nb[1:, :] += black[:-1, :]
    nb[:, :-1] += black[:, 1:]
    nb[:, 1:] += black[:, :-1]

    oob = np.zeros((H, W), dtype=int)
    oob[0, :] += 1
    oob[-1, :] += 1
    oob[:, 0] += 1
    oob[:, -1] += 1

    effective = nb.copy()
    effective[oob >= 2] += oob[oob >= 2]

    rs, cs = np.where(black & (effective >= 3))
    candidates = [(int(r), int(c)) for r, c in zip(rs, cs) if (int(r), int(c)) not in exclude]
    random.shuffle(candidates)
    return candidates


def _combo_visibility_metrics(inp: np.ndarray) -> dict[str, int]:
    combo = main(inp)
    base = _absorb_blue_spurs(inp)
    internal, leaves = _classify_targets(base)

    original_like = np.array(base, dtype=int)
    preserve = set(internal) | set(leaves)
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < original_like.shape[0] and 0 <= cc < original_like.shape[1] and (rr, cc) not in preserve:
                    original_like[rr, cc] = ORANGE
    for r, c in preserve:
        original_like[r, c] = BLACK

    return {
        "internal_pixels": len(internal),
        "leaf_pixels": len(leaves),
        "orange_cells": int(np.count_nonzero(combo == ORANGE)),
        "magenta_cells": int(np.count_nonzero(combo == MAGENTA)),
        "combo_vs_original_diff": int(np.count_nonzero(combo != original_like)),
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)

    for comp in components_mask(grid == BLUE, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in components_mask(grid == BLACK, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return None

    interior_blue = _find_interior_blue(grid)
    reserved_centers: set[tuple[int, int]] = set()
    reserved_halo: set[tuple[int, int]] = set()

    defect_pixels, defects_placed = _place_defects(grid, interior_blue, reserved_centers, reserved_halo)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, defect_pixels, reserved_centers, reserved_halo)
    if len(leaf_pixels) < MIN_LEAVES:
        return None

    output = _build_variant_output(grid, defect_pixels, leaf_pixels)

    inp = grid.copy()
    candidates = _find_spur_candidates(inp, defect_pixels | leaf_pixels)
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    placed_spurs = min(n_spurs, len(candidates))
    if placed_spurs < MIN_SPURS:
        return None
    for r, c in candidates[:placed_spurs]:
        inp[r, c] = BLUE

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None

    metrics = _combo_visibility_metrics(inp)
    if metrics["leaf_pixels"] < MIN_LEAVES:
        return None
    if metrics["magenta_cells"] <= 0:
        return None
    if metrics["combo_vs_original_diff"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid combo sample after {MAX_GEN_ATTEMPTS} attempts")
