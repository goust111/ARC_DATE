# Combo variant of 71e489b6 (index 11)
# combo_spec: D2/0+D4/0
# Name: 仅单像素缺陷+仅边界缺陷

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask


# Design analysis
# ---------------
# a) Concrete behavior:
#    - Original 71e489b6 marks two target families after absorbing blue spurs:
#      every black component of size 1-3 anywhere inside blue, plus leaf pixels
#      from larger black components.
#    - This combo takes the intersection of "single-pixel defect" and
#      "boundary defect".  In practice that removes the interior 1-cell holes:
#      they have blue neighbors but no black 4-neighbor, so they are not on a
#      blue-black boundary.  It also removes 2-3 cell holes because they are
#      not single-pixel defects.
#    - The surviving targets are one-cell black protrusions attached to a large
#      black region.  They have exactly one black 4-neighbor from the parent
#      region and at least one blue 4-neighbor on the outside.
#    - Small trace:
#        old scene: `11111 / 11011 / 11111` gives a 1-cell interior hole at the
#        center, so the old solver halos it.
#        combo scene: the same hole is ignored, but a `0` that sticks one cell
#        out of a large black block into blue is haloed because it is both
#        singular and boundary-touching.
#      The visible difference is "not every tiny black dot is marked anymore;
#      only the tip attached to a black mass is."
#
# b) Complexity calibration:
#    - The generator always includes both marked boundary leaves and unmarked
#      interior black holes.  That prevents the variant from collapsing into
#      "mark all single black pixels" or into the original behavior.
#    - Interior distractors mix size-1 and size-2/3 holes.  This gives the
#      solver repeated evidence that size alone is insufficient and boundary
#      contact is the deciding factor.
#    - Boundary targets stay one cell wide and reasonably separated, so the
#      orange halos remain legible instead of turning into a noisy band.
#
# c) Implicit-assumption review:
#    - Spur absorption is kept unchanged; the combo does not modify that part.
#    - The original branch `if len(component) <= 3: preserve whole component`
#      is no longer valid.  Small components are now always ignored, even when
#      they are 1-cell holes, because they lack the required black neighbor.
#    - The original leaf rule `black_neighbors <= 1` is too broad for this
#      combo if applied to every component.  We restrict it to large black
#      components only, which excludes isolated 2-cell and 3-cell black holes.
#    - The new solver assumes valid targets are one-cell stubs on large black
#      regions.  The generator guarantees this by only adding leaves through a
#      single blue cell with exactly one black neighbor.
#
# d) Placement legality:
#    - Interior distractors are grown only from strictly interior blue cells,
#      and every non-defect 4-neighbor around them remains blue, so they stay
#      isolated from large black regions.
#    - Boundary targets are placed by flipping a blue cell adjacent to a large
#      black component only when that cell has exactly one black 4-neighbor.
#      This keeps each target in-bounds, non-overlapping, and still a 1-cell
#      protrusion after insertion.
#    - Spurs are added only on black cells that the solver's absorption step
#      will deterministically turn back to black.
#
# e) Visual observability:
#    - A human can compare two nearby motifs in the same task family:
#      interior black holes inside blue are left untouched, while black tips
#      attached to a region boundary receive orange halos.
#    - No hidden ordering by numeric color is used.  The inference depends only
#      on visible adjacency and connectedness.
#
# f) Visual cleanliness:
#    - Targets and distractors are spaced so that most distractors are outside
#      the orange halos of true targets.
#    - The partition styles remain large and rectilinear, which preserves the
#      original task's clean region structure and makes the protruding tips easy
#      to spot.
#
# g) Change inferability:
#    - The rule can be inferred from 2-4 examples because the generator repeats
#      the same contrast: black holes fully enclosed by blue are ignored, but
#      one-cell black boundary tips are highlighted.
#    - Color usage remains minimal: input uses only black/blue and output adds
#      orange exactly as a local halo marker.
#
# Post-change solver assumptions checked against the generator:
#    - Every marked target belongs to a black component of size > 3: enforced.
#    - Every marked target has exactly one black 4-neighbor and at least one
#      blue 4-neighbor after spur absorption: enforced.
#    - Every generated grid contains at least one ignored singleton interior
#      hole and at least one ignored 2-3 cell interior hole, so the combo does
#      not degenerate back to the original "mark all small black components".


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

DISTRACTOR_COUNT_RANGE = (3, 6)
DISTRACTOR_CLEARANCE = 2
DISTRACTOR_PLACE_TRIES = 60
TARGET_COUNT_RANGE = (2, 5)
TARGET_CLEARANCE = 2
TARGET_PLACE_TRIES = 80
SPUR_COUNT_RANGE = (2, 6)

MIN_TARGETS = 2
MIN_DISTRACTORS = 3
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 400

HOLE_COL = 0
BLUE_COL = 1
HALO_COL = 7


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(
    pixels: set[tuple[int, int]] | list[tuple[int, int]], dist: int
) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _build_partition(H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        choices = range(CUT_MARGIN, H - CUT_MARGIN)
        cuts = sorted(random.sample(choices, min(n_cuts, H - 2 * CUT_MARGIN)))
        bounds = [0] + cuts + [H]
        first_color = random.randint(0, 1)
        for i in range(len(bounds) - 1):
            grid[bounds[i] : bounds[i + 1], :] = (first_color + i) % 2

    elif style == "rect_inset":
        bg = random.randint(0, 1)
        grid[:, :] = bg
        r0 = random.randint(INSET_MARGIN, H // 3)
        r1 = random.randint(2 * H // 3, H - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, W // 3)
        c1 = random.randint(2 * W // 3, W - CUT_MARGIN)
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
    blue = grid == BLUE_COL
    eroded = blue.copy()
    eroded[:-1, :] &= blue[1:, :]
    eroded[1:, :] &= blue[:-1, :]
    eroded[:, :-1] &= blue[:, 1:]
    eroded[:, 1:] &= blue[:, :-1]
    rs, cs = np.where(eroded)
    coords = list(zip(rs.tolist(), cs.tolist()))
    random.shuffle(coords)
    return coords


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
    base = np.array(grid, dtype=int).copy()
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE_COL:
                continue
            black_n = 0
            oob = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == HOLE_COL:
                    black_n += 1
            if oob >= 2:
                black_n += oob
            if black_n >= 3:
                base[r, c] = HOLE_COL
    return base


def _neighbor_counts(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    H, W = grid.shape
    black_n = 0
    blue_n = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if not (0 <= rr < H and 0 <= cc < W):
            continue
        if int(grid[rr, cc]) == HOLE_COL:
            black_n += 1
        elif int(grid[rr, cc]) == BLUE_COL:
            blue_n += 1
    return black_n, blue_n


def _find_targets(grid: np.ndarray) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    for coords in components_mask(grid == HOLE_COL, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            continue
        for r, c in pts:
            black_n, blue_n = _neighbor_counts(grid, r, c)
            if black_n == 1 and blue_n >= 1:
                targets.add((r, c))
    return targets


def _build_halo(grid: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    H, W = out.shape
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    out[rr, cc] = HALO_COL
    for r, c in preserve:
        out[r, c] = HOLE_COL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Highlight only boundary-attached one-pixel black tips in blue regions.

    Input structure:
    - Blue (1) and black (0) still form a large partitioned scene with wide
      regions, plus a few local anomalies.
    - Some blue pixels inside black areas are fragile spurs that should be
      absorbed back into black before defect detection.
    - Black anomalies come in two visible styles: isolated holes fully inside
      blue, and one-cell protruding tips attached to larger black regions.

    Rule:
    - First absorb blue spurs exactly as in the source task when a blue pixel is
      mostly surrounded by black in the 4-neighborhood.
    - Ignore all small isolated black components, even single-pixel holes.
      Instead, only mark black pixels that belong to a larger black component,
      have exactly one black 4-neighbor, and also touch blue by 4-neighborhood.
    - For each such target pixel, paint an orange (7) Chebyshev-1 halo around
      it while keeping the target pixel itself black.

    Key insight: the new target is not "tiny black means mark it".  The marked
    pixel must be the exposed one-cell tip of a larger black structure, so
    attached boundary position matters as much as local size.
    """
    base = _absorb_spurs(np.array(grid, dtype=int))
    preserve = _find_targets(base)
    return _build_halo(base, preserve)


def _place_interior_holes(
    grid: np.ndarray,
    interior_blue: list[tuple[int, int]],
) -> tuple[list[set[tuple[int, int]]], dict[int, int]]:
    H, W = grid.shape
    placed: list[set[tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)

    count = random.randint(*DISTRACTOR_COUNT_RANGE)
    requested_sizes = [1, random.choice([2, 3])]
    while len(requested_sizes) < count:
        requested_sizes.append(random.choices([1, 2, 3], weights=[4, 2, 1])[0])
    random.shuffle(requested_sizes)

    size_counts = {1: 0, 2: 0, 3: 0}
    for defect_size in requested_sizes:
        if not interior_blue:
            break
        for _ in range(DISTRACTOR_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = interior_blue[random.randint(0, len(interior_blue) - 1)]
            if _nearby(seed_r, seed_c, DISTRACTOR_CLEARANCE, occupied):
                continue

            defect = [(seed_r, seed_c)]
            defect_set = {(seed_r, seed_c)}
            frontier = [(seed_r, seed_c)]
            while frontier and len(defect) < defect_size:
                fr, fc = frontier.pop(random.randrange(len(frontier)))
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = fr + dr
                    nc = fc + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_set:
                        continue
                    if _nearby(nr, nc, DISTRACTOR_CLEARANCE, occupied):
                        continue
                    defect.append((nr, nc))
                    defect_set.add((nr, nc))
                    frontier.append((nr, nc))
                    if len(defect) >= defect_size:
                        break

            if len(defect) != defect_size:
                continue

            valid = True
            for r, c in defect:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if not (0 <= rr < H and 0 <= cc < W):
                        valid = False
                        break
                    if grid[rr, cc] != BLUE_COL or (rr, cc) in occupied:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            placed_set = set(defect)
            placed.append(placed_set)
            occupied.update(placed_set)
            size_counts[defect_size] += 1
            for r, c in placed_set:
                grid[r, c] = HOLE_COL
            remove = _exclusion_zone(placed_set, DISTRACTOR_CLEARANCE)
            interior_blue[:] = [p for p in interior_blue if p not in remove]
            interior_set -= remove
            break

    return placed, size_counts


def _place_boundary_targets(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    H, W = grid.shape
    targets: set[tuple[int, int]] = set()
    goal = random.randint(*TARGET_COUNT_RANGE)

    for _ in range(goal):
        placed = False
        for _ in range(TARGET_PLACE_TRIES):
            large_black = [
                {(int(r), int(c)) for r, c in comp}
                for comp in components_mask(grid == HOLE_COL, connectivity=4)
                if len(comp) > 3
            ]
            random.shuffle(large_black)
            if not large_black:
                return targets

            candidates: list[tuple[int, int]] = []
            for comp_set in large_black:
                for r, c in comp_set:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rr = r + dr
                        cc = c + dc
                        if not (0 <= rr < H and 0 <= cc < W):
                            continue
                        if grid[rr, cc] != BLUE_COL:
                            continue
                        if (rr, cc) in blocked or (rr, cc) in targets:
                            continue
                        if _nearby(rr, cc, TARGET_CLEARANCE, targets):
                            continue
                        if _nearby(rr, cc, TARGET_CLEARANCE + 1, blocked):
                            continue
                        black_n, _ = _neighbor_counts(grid, rr, cc)
                        if black_n == 1:
                            candidates.append((rr, cc))
                if candidates:
                    break

            if not candidates:
                break

            tr, tc = random.choice(candidates)
            grid[tr, tc] = HOLE_COL
            black_n, blue_n = _neighbor_counts(grid, tr, tc)
            if black_n == 1 and blue_n >= 1:
                targets.add((tr, tc))
                placed = True
                break
            grid[tr, tc] = BLUE_COL

        if not placed:
            break

    return targets


def _find_spur_candidates(
    grid: np.ndarray,
    exclude: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    black = grid == HOLE_COL
    nb = np.zeros(grid.shape, dtype=int)
    nb[:-1, :] += black[1:, :]
    nb[1:, :] += black[:-1, :]
    nb[:, :-1] += black[:, 1:]
    nb[:, 1:] += black[:, :-1]

    oob = np.zeros(grid.shape, dtype=int)
    oob[0, :] += 1
    oob[-1, :] += 1
    oob[:, 0] += 1
    oob[:, -1] += 1

    effective = nb.copy()
    effective[oob >= 2] += oob[oob >= 2]

    rs, cs = np.where(black & (effective >= 3))
    candidates = [
        (int(r), int(c))
        for r, c in zip(rs, cs)
        if (int(r), int(c)) not in exclude
    ]
    random.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)

    for comp in components_mask(grid == BLUE_COL, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in components_mask(grid == HOLE_COL, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return None

    interior_blue = _find_interior_blue(grid)
    distractor_components, size_counts = _place_interior_holes(grid, interior_blue)
    distractor_pixels = set().union(*distractor_components) if distractor_components else set()
    if len(distractor_components) < MIN_DISTRACTORS:
        return None
    if size_counts[1] < 1 or (size_counts[2] + size_counts[3]) < 1:
        return None

    targets = _place_boundary_targets(grid, distractor_pixels)
    if len(targets) < MIN_TARGETS:
        return None

    output = _build_halo(grid, targets)

    inp = grid.copy()
    spur_goal = random.randint(*SPUR_COUNT_RANGE)
    spur_candidates = _find_spur_candidates(inp, distractor_pixels | targets)
    spurs = spur_candidates[:spur_goal]
    for r, c in spurs:
        inp[r, c] = BLUE_COL
    if len(spurs) < MIN_SPURS:
        return None

    base = _absorb_spurs(inp)
    recovered_targets = _find_targets(base)
    if recovered_targets != targets:
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
