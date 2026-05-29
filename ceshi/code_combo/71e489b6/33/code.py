# Combo variant of 71e489b6 (index 33)
# combo_spec: D1/1+D11/0
# Name: 可变半径光晕+迭代检测

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask


"""
Design analysis for combo D1/1 + D11/0
======================================

a) Concrete behavior and trace
  - Original task:
    absorb blue spur pixels inside black regions; detect every small black component of size <= 3
    plus leaf pixels of larger black components; paint a fixed Chebyshev-1 orange halo in one shot.
  - Fused rule used here:
    keep the spur-absorption step, but replace one-shot marking with three recomputed rounds:
    size-3 defects first with radius 3, then current size-2 defects with radius 2, then current
    size-1 defects and large-component leaves with radius 1.
  - Why the iteration is real:
    if a 3-cell defect sits near a 2-cell defect, round 1 may paint orange onto one pixel of the
    2-cell defect because that smaller defect is not protected yet. The 2-cell defect then becomes
    a singleton and is only detected in the later small-radius round. This makes D1/1 and D11/0
    causally interact instead of merely coexisting.
  - Small concrete trace:
    a horizontal 3-cell defect at row r and a vertical 2-cell defect with one pixel at Chebyshev
    distance 3 and the other at distance 4 from it behave as follows.
    1. Original solver: both defects are preserved immediately; both only receive radius-1 halos.
    2. Fused solver round k=3: the 3-cell component is preserved; its radius-3 halo overwrites the
       nearer pixel of the 2-cell component.
    3. Fused solver round k=2: that former 2-cell defect no longer has size 2, so nothing happens.
    4. Fused solver round k=1: the remaining singleton is finally detected and gets only radius 1.

b) Complexity calibration
  - Halo radii are limited to exactly {1, 2, 3}; this keeps the mapping visible and learnable.
  - Every generated sample contains:
    one size-3 defect, one isolated size-2 defect, one isolated singleton, and one forced
    "shrink pair" whose size-2 member is reduced by the earlier radius-3 round.
  - That gives repeated support for all three radii while still exposing the iterative effect.
  - The generator keeps only a few total defects, so the output remains readable rather than turning
    into dense orange coverage.

c) Implicit-assumption audit of the original main()
  - Original assumption: all target pixels can be collected before any painting happens.
    Broken here: later rounds must see the orange written by earlier rounds.
  - Original assumption: a defect's size is measured on the raw post-cleaning black mask once.
    Broken here: the measured size is round-dependent for small defects.
  - Original assumption: any black pixel inside a small component that is once identified is always
    preserved from every other target's halo in the same pass.
    Broken here on purpose: only the current round's targets are protected, so earlier large halos
    can erase not-yet-detected smaller defects.
  - New solver-side assumption:
    only pixels that belonged to an initial small black component are allowed to become later-round
    small-defect targets. This prevents orange splitting of a bulk black region from creating fake
    "new defects". The generator guarantees a clear separation between true blue-hole defects and
    large black regions.

d) Placement legality
  - Defect placement checks that every defect pixel is currently blue and every non-defect 4-neighbor
    around it is also blue, so each defect is genuinely a black hole inside blue.
  - A Chebyshev exclusion zone keeps ordinary defects far apart; only the forced shrink pair violates
    wide spacing, and it does so with a hand-crafted geometry that still avoids 4-connectivity.
  - Blue spurs are inserted only at positions that the absorption rule provably removes.

e) Visual observability
  - Radius differences are visible directly in the output: singleton, size-2, and size-3 defects
    produce three distinct square halo extents.
  - The iterative part is also visible: one apparent "would-be" 2-cell defect ends up showing a
    mixed pattern, with one cell consumed by a larger halo and only the leftover singleton marked
    later by a smaller ring.

f) Visual cleanliness
  - The forced interaction uses exactly one local pair; all other defects are isolated.
  - Spurs stay in black regions and disappear during solving, so they do not add output clutter.
  - Leaves from large black regions are still supported by the solver, but the generator does not
    rely on them for the core combo signal, which keeps the sample easier to read.

g) Post-change generator guarantees tied to main()
  - `overwritten_small_components >= 1`:
    at least one initial small defect loses a black pixel to an earlier larger-radius round.
  - `round_component_counts[3] >= 1`, `round_component_counts[2] >= 1`, `round_component_counts[1] >= 2`:
    all three radii are behaviorally active in accepted samples.
  - `main(inp) != baseline_original_main(inp)`:
    accepted samples do not collapse back to the original fixed-radius one-shot task.
"""


HOLE_COL = 0
BLUE_COL = 1
HALO_COL = 7

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
SPUR_COUNT_RANGE = (2, 5)
LEAF_COUNT_RANGE = (1, 1)
MIN_SPURS = 2
DEFECT_CLEARANCE = 4
DEFECT_PLACE_TRIES = 80
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Apply the original blue-spur absorption step."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE_COL:
                continue
            black_nb = 0
            oob = 0
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == HOLE_COL:
                    black_nb += 1
            if oob >= 2:
                black_nb += oob
            if black_nb >= 3:
                base[r, c] = HOLE_COL
    return base


def _initial_black_partition(base: np.ndarray) -> tuple[list[set[tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]], dict[tuple[int, int], int]]:
    small_components: list[set[tuple[int, int]]] = []
    small_pool: set[tuple[int, int]] = set()
    large_pool: set[tuple[int, int]] = set()
    owner: dict[tuple[int, int], int] = {}

    comps = components_mask(base == HOLE_COL, connectivity=4)
    for comp in comps:
        pts = {(int(r), int(c)) for r, c in comp}
        if len(pts) <= 3:
            idx = len(small_components)
            small_components.append(pts)
            small_pool |= pts
            for pt in pts:
                owner[pt] = idx
        else:
            large_pool |= pts
    return small_components, small_pool, large_pool, owner


def _solve_combo(grid: np.ndarray, *, return_trace: bool = False):
    """
    Highlight black defects in blue regions with size-dependent halos applied iteratively.

    Input structure:
    - Blue (1) foreground regions and large black (0) regions share the grid.
    - Small black hole-like defects of size 1..3 may sit strictly inside blue regions.
    - Some black regions may contain thin blue spur pixels that should be absorbed first.

    Rule:
    - First absorb blue spur pixels that are surrounded by black on at least three effective sides.
    - Then run three marking rounds on the current black mask: size-3 defects with radius 3, then
      current size-2 defects with radius 2, then current size-1 defects and leaf pixels of large
      black regions with radius 1.
    - In each round, only that round's targets are protected; earlier large halos may therefore erase
      pixels from smaller nearby defects before those smaller defects are re-detected.

    Key insight:
    - A defect is not judged only once. The orange written by larger defects changes what smaller
      defects still look like, so both halo extent and processing order matter.
    """
    base = _absorb_blue_spurs(grid)
    H, W = base.shape

    small_components, small_pool, large_pool, small_owner = _initial_black_partition(base)
    state = base.copy()
    locked_black: set[tuple[int, int]] = set()

    trace = {
        "initial_small_sizes": [len(comp) for comp in small_components],
        "remaining_before_round": {idx: {} for idx in range(len(small_components))},
        "round_component_counts": {1: 0, 2: 0, 3: 0},
        "round_pixel_counts": {1: 0, 2: 0, 3: 0},
        "round_leaf_counts": {1: 0, 2: 0, 3: 0},
        "overwritten_small_pixels": 0,
        "overwritten_small_components": set(),
        "used_radii": set(),
    }

    for k in (3, 2, 1):
        black_now = {(int(r), int(c)) for r, c in np.argwhere(state == HOLE_COL)}
        for idx, comp in enumerate(small_components):
            trace["remaining_before_round"][idx][k] = len(comp & black_now)

        targets: list[tuple[set[tuple[int, int]], int]] = []
        leaf_targets_this_round = 0
        for comp in components_mask(state == HOLE_COL, connectivity=4):
            pts = {(int(r), int(c)) for r, c in comp}
            if pts <= locked_black:
                continue
            if pts.issubset(small_pool):
                if len(pts) == k and not (pts & locked_black):
                    targets.append((pts, k))
                continue
            if k != 1 or not (pts & large_pool):
                continue
            for r, c in pts:
                if (r, c) not in large_pool or (r, c) in locked_black:
                    continue
                deg = sum((rr, cc) in pts for rr, cc in _neighbors4(r, c))
                if deg <= 1:
                    targets.append(({(r, c)}, 1))
                    leaf_targets_this_round += 1

        if not targets:
            continue

        current_targets = set().union(*(pixels for pixels, _radius in targets))
        locked_black |= current_targets
        trace["round_component_counts"][k] += len(targets)
        trace["round_pixel_counts"][k] += sum(len(pixels) for pixels, _radius in targets)
        trace["round_leaf_counts"][k] += leaf_targets_this_round
        trace["used_radii"].add(k)

        for pixels, radius in targets:
            for r, c in pixels:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        rr = r + dr
                        cc = c + dc
                        if not (0 <= rr < H and 0 <= cc < W):
                            continue
                        if (rr, cc) in locked_black:
                            continue
                        if state[rr, cc] == HOLE_COL and (rr, cc) in small_pool:
                            trace["overwritten_small_pixels"] += 1
                            trace["overwritten_small_components"].add(small_owner[(rr, cc)])
                        state[rr, cc] = HALO_COL

        for r, c in locked_black:
            state[r, c] = HOLE_COL

    trace["used_radii"] = tuple(sorted(trace["used_radii"]))
    trace["overwritten_small_components"] = len(trace["overwritten_small_components"])
    trace["shrunk_size2_to1"] = sum(
        1
        for idx, size in enumerate(trace["initial_small_sizes"])
        if size == 2 and trace["remaining_before_round"][idx].get(2) == 1
    )

    if return_trace:
        return state, trace
    return state


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black defects in blue regions with size-coded halos and iterative re-detection.

    Input structure:
    - The grid mixes large blue foreground regions with large black regions.
    - Small black hole-like defects of size 1..3 can sit inside blue, and some large black regions
      may have one-cell leaf tips protruding into adjacent blue.
    - A few blue spur pixels may also appear inside black regions; those are noise to be absorbed.

    Rule:
    - First absorb any blue spur pixel that has at least three effective black sides under the
      original 4-neighbor rule.
    - Then run three ordered marking rounds on the evolving black mask: current size-3 defects get
      radius-3 orange halos, current size-2 defects get radius-2 halos, and current size-1 defects
      plus leaf pixels of large black components get radius-1 halos.
    - Earlier rounds paint immediately and only protect their own current target centers, so a large
      halo can partially overwrite a nearby smaller defect before that smaller defect is re-detected.

    Key insight:
    - Halo size depends on defect size, but defect size itself can change across rounds because the
      orange painted by earlier larger defects alters what black pixels remain for later detection.
    """
    return _solve_combo(np.array(grid, dtype=int), return_trace=False)


def _baseline_original_main(grid: np.ndarray) -> np.ndarray:
    """Exact fixed-radius one-shot baseline used only to reject degenerate generated samples."""
    base = _absorb_blue_spurs(grid)
    H, W = base.shape
    preserve: set[tuple[int, int]] = set()
    comps = components_mask(base == HOLE_COL, connectivity=4)
    for coords in comps:
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        for r, c in pts:
            deg = sum(
                1
                for rr, cc in _neighbors4(r, c)
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == HOLE_COL
            )
            if deg <= 1:
                preserve.add((r, c))

    out = base.copy()
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


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(pixels: set[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
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
        cut_space = list(range(CUT_MARGIN, H - CUT_MARGIN))
        cuts = sorted(random.sample(cut_space, min(n_cuts, len(cut_space)))) if cut_space else []
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
        if random.random() < 0.5:
            grid[:, :vc] = 1 - bg
            hc = random.randint(H // 3, 2 * H // 3)
            grid[:hc, vc:] = 1 - bg
        else:
            grid[:, vc:] = 1 - bg
            hc = random.randint(H // 3, 2 * H // 3)
            grid[hc:, :vc] = 1 - bg

    return grid


def _valid_partition(grid: np.ndarray) -> bool:
    blue_comps = components_mask(grid == BLUE_COL, connectivity=4)
    black_comps = components_mask(grid == HOLE_COL, connectivity=4)
    return (
        blue_comps
        and black_comps
        and all(len(comp) >= MIN_BLUE_REGION for comp in blue_comps)
        and all(len(comp) >= MIN_BLACK_REGION for comp in black_comps)
    )


def _shape_ok(grid: np.ndarray, cells: list[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    H, W = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if grid[r, c] != BLUE_COL or (r, c) in reserved:
            return False
    for r, c in cells:
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in cell_set and grid[rr, cc] != BLUE_COL:
                return False
    return True


def _place_pattern(
    grid: np.ndarray,
    offsets: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    H, W = grid.shape
    max_r = max(r for r, _ in offsets)
    max_c = max(c for _, c in offsets)
    anchors = [(r, c) for r in range(H - max_r) for c in range(W - max_c)]
    random.shuffle(anchors)
    for ar, ac in anchors[:DEFECT_PLACE_TRIES * 6]:
        cells = [(ar + dr, ac + dc) for dr, dc in offsets]
        if _shape_ok(grid, cells, reserved):
            for r, c in cells:
                grid[r, c] = HOLE_COL
            return set(cells)
    return None


def _place_forced_combo_pair(grid: np.ndarray, reserved: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    size3_offsets = [(0, 0), (0, 1), (0, 2)]
    shrinking_size2_offsets = [(3, 5), (4, 5)]

    H, W = grid.shape
    anchors = [(r, c) for r in range(H - 4) for c in range(W - 5)]
    random.shuffle(anchors)
    for ar, ac in anchors[:DEFECT_PLACE_TRIES * 8]:
        s3 = {(ar + dr, ac + dc) for dr, dc in size3_offsets}
        s2 = {(ar + dr, ac + dc) for dr, dc in shrinking_size2_offsets}
        all_cells = list(s3 | s2)
        if not _shape_ok(grid, all_cells, reserved):
            continue
        for r, c in all_cells:
            grid[r, c] = HOLE_COL
        return s3, s2
    return None


def _place_leaves(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    count_range: tuple[int, int] = LEAF_COUNT_RANGE,
) -> set[tuple[int, int]] | None:
    H, W = grid.shape
    leaf_pixels: set[tuple[int, int]] = set()
    large_comps = [
        {(int(r), int(c)) for r, c in comp}
        for comp in components_mask(grid == HOLE_COL, connectivity=4)
        if len(comp) > 3
    ]
    if not large_comps:
        return None

    target_count = random.randint(*count_range)
    for _ in range(target_count):
        candidates: list[tuple[int, int]] = []
        for comp in large_comps:
            for r, c in comp:
                for rr, cc in _neighbors4(r, c):
                    if not (0 <= rr < H and 0 <= cc < W):
                        continue
                    if grid[rr, cc] != BLUE_COL or (rr, cc) in reserved or (rr, cc) in leaf_pixels:
                        continue
                    black_deg = sum(
                        1
                        for nr, nc in _neighbors4(rr, cc)
                        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == HOLE_COL
                    )
                    if black_deg == 1:
                        candidates.append((rr, cc))
        if not candidates:
            return None
        lr, lc = random.choice(candidates)
        grid[lr, lc] = HOLE_COL
        leaf_pixels.add((lr, lc))
        reserved |= _exclusion_zone({(lr, lc)}, 2)
        large_comps = [
            {(int(r), int(c)) for r, c in comp}
            for comp in components_mask(grid == HOLE_COL, connectivity=4)
            if len(comp) > 3
        ]
        if not large_comps:
            return None

    return leaf_pixels


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    H, W = grid.shape
    black = grid == HOLE_COL

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


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)
    if not _valid_partition(grid):
        return None

    reserved: set[tuple[int, int]] = set()
    defects: list[set[tuple[int, int]]] = []

    forced = _place_forced_combo_pair(grid, reserved)
    if forced is None:
        return None
    forced_s3, forced_s2 = forced
    defects.extend([forced_s3, forced_s2])
    reserved |= _exclusion_zone(forced_s3 | forced_s2, DEFECT_CLEARANCE)

    isolated_size2 = _place_pattern(grid, [(0, 0), (1, 0)] if random.random() < 0.5 else [(0, 0), (0, 1)], reserved)
    if isolated_size2 is None:
        return None
    defects.append(isolated_size2)
    reserved |= _exclusion_zone(isolated_size2, DEFECT_CLEARANCE)

    isolated_size1 = _place_pattern(grid, [(0, 0)], reserved)
    if isolated_size1 is None:
        return None
    defects.append(isolated_size1)
    reserved |= _exclusion_zone(isolated_size1, DEFECT_CLEARANCE)

    leaf_pixels = _place_leaves(grid, reserved)
    if leaf_pixels is None or len(leaf_pixels) < LEAF_COUNT_RANGE[0]:
        return None
    reserved |= _exclusion_zone(leaf_pixels, 2)

    inp = grid.copy()
    all_target_pixels = set().union(*defects) | leaf_pixels
    spur_candidates = _find_spur_candidates(inp, all_target_pixels)
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    if min(n_spurs, len(spur_candidates)) < MIN_SPURS:
        return None
    for r, c in spur_candidates[:n_spurs]:
        inp[r, c] = BLUE_COL

    out, trace = _solve_combo(inp, return_trace=True)
    if np.array_equal(out, _baseline_original_main(inp)):
        return None
    if trace["overwritten_small_components"] < 1:
        return None
    if trace["shrunk_size2_to1"] < 1:
        return None
    if trace["round_component_counts"][3] < 1:
        return None
    if trace["round_component_counts"][2] < 1:
        return None
    if trace["round_component_counts"][1] < 2:
        return None
    if trace["round_leaf_counts"][1] < LEAF_COUNT_RANGE[0]:
        return None
    if tuple(trace["used_radii"]) != (1, 2, 3):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"Failed to generate valid combo sample after {MAX_GEN_ATTEMPTS} attempts")
