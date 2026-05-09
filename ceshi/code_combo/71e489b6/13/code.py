# Combo variant of 71e489b6 (index 13)
# combo_spec: D1/0+D7/1
# Name: 十字光晕+互斥光晕
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
ORANGE = 7

CROSS_OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0)]


# Design analysis for combo D1/0 + D7/1
# a) Concrete behavior trace:
#    - Original: after blue spurs are absorbed, every small black component (size <= 3)
#      and every degree-0/1 leaf pixel of a large black component receives a full 3x3
#      orange halo. Overlap is harmless because later paints simply keep cells orange.
#    - Fused: the detected target classes stay the same, but each defect object now emits
#      only a 4-neighbor cross. Small components are treated as one object whose halo is the
#      union of the crosses around its pixels; large-component leaves are still singleton
#      objects. If two different objects want the same halo cell, that cell is left untouched.
#    - Example with two 1-cell defects two columns apart inside blue:
#         input            old 3x3 rule        fused rule
#         11111            77777               17171
#         10101            70707               10101
#         11111            77777               17171
#      Under the fused rule, the shared middle arm cell is not painted, so the two crosses
#      visibly repel each other.
# b) Complexity calibration:
#    - Only two structural target categories remain: tiny isolated black holes and leaf tips
#      of large black masses. The output change is concentrated in one visual channel:
#      cross geometry plus overlap erasure. This keeps the rule learnable.
#    - The generator forces at least one true halo conflict between distinct objects, so the
#      D7 change is not a latent no-op. Other objects are kept spaced out so the conflict is
#      easy to inspect.
# c) Implicit assumptions audit:
#    - Original STEP 3 assumed sequential overwriting was enough. That breaks here because
#      overlap cells must keep the pre-paint color, so the solver must count per-object halo
#      coverage before writing any orange.
#    - Original generator kept defects Chebyshev-separated by 2, which suppresses most cross
#      conflicts. The combo generator intentionally introduces one distance-2 aligned pair and
#      verifies that its exclusive output differs from simple additive-cross painting.
#    - Small black components are still recognized by 4-connectivity and size <= 3, so the
#      generator keeps them strictly inside blue and away from large black regions.
#    - Leaf detection still relies on black degree <= 1 inside large components, so leaf
#      placement is limited to blue cells with exactly one black attachment.
# d) Placement legality:
#    - The forced conflict pair uses two blue-interior singleton defects with one blue cell
#      between them, keeping them disconnected but guaranteeing one shared cross arm.
#    - Additional defects and leaves are accepted only if they stay in bounds, preserve the
#      intended connectivity class, and do not collide with reserved target or halo cells.
#    - Spurs are inserted only at black cells that the solver's absorption pass deterministically
#      flips back to black.
# e) Visual observability:
#    - Across samples, humans can see both recurring cues: orange markers are crosses rather
#      than squares, and neighboring defects sometimes leave a conspicuous unpainted gap exactly
#      where two cross arms would otherwise meet.
# f) Visual cleanliness:
#    - Only one object pair is required to conflict. All other targets are halo-separated,
#      preventing the output from collapsing into dense orange patches.
# g) Inferability of the change:
#    - The color vocabulary is unchanged from the source task. The only new evidence is spatial:
#      orthogonal-only reach and overlap erasure. Because the generator repeats both behaviors,
#      a human can infer the combo without relying on hidden metadata such as color ids.
# Additional solver/generator contract:
#    - `main()` assumes overlap is evaluated between defect objects, not between the individual
#      pixels of one multi-cell small component. `generate_input()` builds and scores objects
#      with the same object-level definition.


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Mirror the original spur-absorption pass exactly."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            black_n = 0
            oob = 0
            for dr, dc in CROSS_OFFSETS:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == BLACK:
                    black_n += 1
            if oob >= 2:
                black_n += oob
            if black_n >= 3:
                base[r, c] = BLACK
    return base


def _black_degree(grid: np.ndarray, r: int, c: int) -> int:
    H, W = grid.shape
    deg = 0
    for dr, dc in CROSS_OFFSETS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == BLACK:
            deg += 1
    return deg


def _collect_defect_objects(base: np.ndarray) -> tuple[list[set[tuple[int, int]]], set[tuple[int, int]]]:
    """
    Collect target objects after spur absorption.

    Small black components (size <= 3) are one object per component.
    Leaf pixels on larger black components are singleton objects.
    """
    objects: list[set[tuple[int, int]]] = []
    preserve: set[tuple[int, int]] = set()

    for coords in components_mask(base == BLACK, connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        if len(pts) <= 3:
            objects.append(pts)
            preserve.update(pts)
            continue
        for r, c in sorted(pts):
            if _black_degree(base, r, c) <= 1:
                objects.append({(r, c)})
                preserve.add((r, c))
    return objects, preserve


def _cross_footprint(
    shape: tuple[int, int],
    obj: set[tuple[int, int]],
    preserve: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Return the in-bounds cross halo cells for one defect object."""
    H, W = shape
    halo: set[tuple[int, int]] = set()
    for r, c in obj:
        for dr, dc in CROSS_OFFSETS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                halo.add((rr, cc))
    return halo


def _paint_exclusive_crosses(
    base: np.ndarray,
    objects: list[set[tuple[int, int]]],
    preserve: set[tuple[int, int]],
) -> np.ndarray:
    """Paint orange on halo cells covered by exactly one defect object."""
    out = np.array(base, dtype=int)
    cover_count: dict[tuple[int, int], int] = {}
    for obj in objects:
        for cell in _cross_footprint(base.shape, obj, preserve):
            cover_count[cell] = cover_count.get(cell, 0) + 1
    for (r, c), count in cover_count.items():
        if count == 1:
            out[r, c] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black anomalies with cross halos that cancel at shared cells.

    Input structure:
    - Blue (1) foreground regions are interleaved with large black (0) regions.
    - Blue interiors may contain tiny black components, and large black regions may have
      1-pixel leaf tips protruding into adjacent blue.
    - Black regions may also contain thin blue spur pixels that should be absorbed first.

    Rule:
    - First absorb any blue spur whose effective 4-neighborhood is mostly black, using the
      same corner treatment as the source task.
    - Then detect two target classes: each small 4-connected black component of size <= 3,
      and each degree-0/1 leaf pixel belonging to a larger black component.
    - For every target object, form an orange (7) cross halo on the four orthogonal neighbor
      cells of its pixels, but if a halo cell is claimed by two or more target objects, do not
      paint that shared cell. All target pixels themselves stay black.

    Key insight:
    - The solver still finds the same structural black defects as the original task, but the
      marking geometry is now orthogonal-only and conflict-aware: orange appears only on unique
      cross arms, while contested halo cells remain in their original color.
    """
    base = _absorb_blue_spurs(np.array(grid, dtype=int))
    objects, preserve = _collect_defect_objects(base)
    return _paint_exclusive_crosses(base, objects, preserve)


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
LEAF_PLACE_TRIES = 100
SPUR_COUNT_RANGE = (2, 6)

MIN_DEFECT_OBJECTS = 3
MIN_LEAVES = 1
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 500


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _expand_zone(cells: set[tuple[int, int]] | list[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _build_partition(H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        choices = list(range(CUT_MARGIN, H - CUT_MARGIN))
        cuts = sorted(random.sample(choices, min(n_cuts, len(choices))))
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
        hc = random.randint(H // 3, 2 * H // 3)
        if random.random() < 0.5:
            grid[:, :vc] = 1 - bg
            grid[:hc, vc:] = 1 - bg
        else:
            grid[:, vc:] = 1 - bg
            grid[hc:, :vc] = 1 - bg

    return grid


def _find_interior_blue(grid: np.ndarray) -> list[tuple[int, int]]:
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


def _validate_region_sizes(grid: np.ndarray) -> bool:
    blue_comps = components_mask(grid == BLUE, connectivity=4)
    black_comps = components_mask(grid == BLACK, connectivity=4)
    return (
        len(blue_comps) > 0
        and len(black_comps) > 0
        and all(len(comp) >= MIN_BLUE_REGION for comp in blue_comps)
        and all(len(comp) >= MIN_BLACK_REGION for comp in black_comps)
    )


def _force_overlap_pair(grid: np.ndarray) -> list[set[tuple[int, int]]] | None:
    """Place two singleton defects at Manhattan distance 2 so their cross halos share one cell."""
    interior = _find_interior_blue(grid)
    interior_set = set(interior)
    for r, c in interior:
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            rr = r + dr
            cc = c + dc
            mr = r + dr // 2
            mc = c + dc // 2
            if (rr, cc) not in interior_set or (mr, mc) not in interior_set:
                continue
            pair = [{(r, c)}, {(rr, cc)}]
            preserve = {(r, c), (rr, cc)}
            halos = [_cross_footprint(grid.shape, obj, preserve) for obj in pair]
            if halos[0] & halos[1]:
                return pair
    return None


def _candidate_defect_shapes(
    grid: np.ndarray,
    interior_blue: list[tuple[int, int]],
    blocked_targets: set[tuple[int, int]],
    occupied_targets: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    """Enumerate a few legal small-component candidates grown from interior seeds."""
    H, W = grid.shape
    interior_set = set(interior_blue)
    candidates: list[set[tuple[int, int]]] = []

    sample = interior_blue[:]
    random.shuffle(sample)
    for seed_r, seed_c in sample[:80]:
        if (seed_r, seed_c) in blocked_targets or (seed_r, seed_c) in occupied_targets:
            continue
        for defect_size in random.sample([1, 2, 3], k=3):
            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = CROSS_OFFSETS[:]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in occupied_targets or (nr, nc) in blocked_targets:
                        continue
                    defect.append((nr, nc))
                    if len(defect) >= defect_size:
                        break
            defect_set = set(defect)
            if len(defect_set) != defect_size:
                continue
            valid = True
            for r, c in defect_set:
                if (r, c) in blocked_targets:
                    valid = False
                    break
                for dr, dc in CROSS_OFFSETS:
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W and (grid[rr, cc] != BLUE or (rr, cc) in occupied_targets):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                candidates.append(defect_set)
    random.shuffle(candidates)
    return candidates


def _place_additional_defects(
    grid: np.ndarray,
    objects: list[set[tuple[int, int]]],
    preserve: set[tuple[int, int]],
    reserved_halos: set[tuple[int, int]],
) -> int:
    """Add extra small black components while avoiding accidental halo clutter."""
    target_total = random.randint(*DEFECT_COUNT_RANGE)
    blocked_targets = _expand_zone(preserve, DEFECT_CLEARANCE) | _expand_zone(reserved_halos, 1)
    placed = len(objects)

    while placed < target_total:
        interior_blue = _find_interior_blue(grid)
        candidates = _candidate_defect_shapes(grid, interior_blue, blocked_targets, preserve)
        accepted = False
        for defect in candidates[:DEFECT_PLACE_TRIES]:
            candidate_preserve = preserve | defect
            halo = _cross_footprint(grid.shape, defect, candidate_preserve)
            if halo & reserved_halos:
                continue
            for r, c in defect:
                grid[r, c] = BLACK
            objects.append(defect)
            preserve.update(defect)
            reserved_halos.update(halo)
            blocked_targets |= _expand_zone(defect, DEFECT_CLEARANCE)
            blocked_targets |= _expand_zone(halo, 1)
            placed += 1
            accepted = True
            break
        if not accepted:
            break
    return placed


def _place_leaves(
    grid: np.ndarray,
    objects: list[set[tuple[int, int]]],
    preserve: set[tuple[int, int]],
    reserved_halos: set[tuple[int, int]],
) -> int:
    """Place singleton leaf pixels on large black regions, keeping their halos separate."""
    blocked_targets = _expand_zone(preserve, DEFECT_CLEARANCE) | _expand_zone(reserved_halos, 1)
    leaf_count = 0
    n_leaves = random.randint(*LEAF_COUNT_RANGE)

    for _ in range(n_leaves):
        placed = False
        for _try in range(LEAF_PLACE_TRIES):
            large_black = [
                {(int(r), int(c)) for r, c in comp}
                for comp in components_mask(grid == BLACK, connectivity=4)
                if len(comp) > 3
            ]
            random.shuffle(large_black)
            if not large_black:
                break
            comp_set = large_black[0]
            candidates: list[tuple[int, int]] = []
            H, W = grid.shape
            for r, c in comp_set:
                for dr, dc in CROSS_OFFSETS:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < H and 0 <= cc < W) or grid[rr, cc] != BLUE:
                        continue
                    if (rr, cc) in preserve or (rr, cc) in blocked_targets:
                        continue
                    black_n = 0
                    for dr2, dc2 in CROSS_OFFSETS:
                        nr = rr + dr2
                        nc = cc + dc2
                        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == BLACK:
                            black_n += 1
                    if black_n == 1:
                        candidates.append((rr, cc))
            random.shuffle(candidates)
            for cell in candidates:
                obj = {cell}
                candidate_preserve = preserve | obj
                halo = _cross_footprint(grid.shape, obj, candidate_preserve)
                if halo & reserved_halos:
                    continue
                lr, lc = cell
                grid[lr, lc] = BLACK
                objects.append(obj)
                preserve.add(cell)
                reserved_halos.update(halo)
                blocked_targets |= _expand_zone(obj, DEFECT_CLEARANCE)
                blocked_targets |= _expand_zone(halo, 1)
                leaf_count += 1
                placed = True
                break
            if placed:
                break
        if not placed:
            break
    return leaf_count


def _build_output_and_overlap(
    grid: np.ndarray,
    objects: list[set[tuple[int, int]]],
    preserve: set[tuple[int, int]],
) -> tuple[np.ndarray, int]:
    """Render output and count exclusive-rule conflict cells."""
    out = np.array(grid, dtype=int)
    cover_count: dict[tuple[int, int], int] = {}
    for obj in objects:
        for cell in _cross_footprint(grid.shape, obj, preserve):
            cover_count[cell] = cover_count.get(cell, 0) + 1
    for (r, c), count in cover_count.items():
        if count == 1:
            out[r, c] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    overlap_cells = sum(1 for count in cover_count.values() if count >= 2)
    return out, overlap_cells


def _build_additive_cross_output(
    grid: np.ndarray,
    objects: list[set[tuple[int, int]]],
    preserve: set[tuple[int, int]],
) -> np.ndarray:
    """Reference output for the non-exclusive D1-only behavior."""
    out = np.array(grid, dtype=int)
    for obj in objects:
        for r, c in _cross_footprint(grid.shape, obj, preserve):
            out[r, c] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Find black cells eligible to become blue spurs and later be re-absorbed.

    This mirrors the solver's absorption criterion.
    """
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
    candidates = [(int(r), int(c)) for r, c in zip(rs, cs) if (r, c) not in exclude]
    random.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)
    if not _validate_region_sizes(grid):
        return None

    objects = _force_overlap_pair(grid)
    if objects is None:
        return None

    preserve: set[tuple[int, int]] = set()
    for obj in objects:
        preserve.update(obj)
        for r, c in obj:
            grid[r, c] = BLACK

    reserved_halos: set[tuple[int, int]] = set()
    for obj in objects:
        reserved_halos.update(_cross_footprint(grid.shape, obj, preserve))

    defect_objects = _place_additional_defects(grid, objects, preserve, reserved_halos)
    if defect_objects < MIN_DEFECT_OBJECTS:
        return None

    leaf_count = _place_leaves(grid, objects, preserve, reserved_halos)
    if leaf_count < MIN_LEAVES:
        return None

    output, overlap_cells = _build_output_and_overlap(grid, objects, preserve)
    if overlap_cells < 1:
        return None

    additive = _build_additive_cross_output(grid, objects, preserve)
    if np.array_equal(additive, output):
        return None

    inp = grid.copy()
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    candidates = _find_spur_candidates(inp, preserve)
    for r, c in candidates[:n_spurs]:
        inp[r, c] = BLUE
    if min(n_spurs, len(candidates)) < MIN_SPURS:
        return None

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
