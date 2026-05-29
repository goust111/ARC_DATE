# Combo variant of 71e489b6 (index 17)
# combo_spec: D1/1+D12/1
# Name: 可变半径光晕+缺陷中心红色标记

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask

# Design analysis for the combo variant D1/1 + D12/1
# a) Concrete behavior and trace:
#    - Original task:
#      1. Absorb blue spur pixels that sit in black zones.
#      2. Detect two target families on the cleaned black layout:
#         small black components of size 1..3, and leaf pixels of large black components.
#      3. Paint an orange Chebyshev-1 halo while keeping the target pixels black.
#    - Fused task:
#      1. The same spur-absorption step is kept unchanged, so the defect topology is still read
#         from the cleaned grid.
#      2. Small black components now carry a size code:
#         size-1 component -> orange radius 1, size-2 component -> orange radius 2,
#         size-3 component -> orange radius 3.
#      3. Target pixels themselves become red instead of remaining black.
#      4. Leaf pixels of large black components are still valid targets, but each leaf is treated
#         as its own singleton center, so it naturally receives radius 1 and a red center.
#    - Small trace:
#         clean input patch               fused output patch
#         1 1 1 1 1 1 1                  7 7 7 7 7 7 7
#         1 1 0 0 1 1 1   size-2 hole -> 7 7 2 2 7 7 7
#         1 1 1 1 1 1 1                  7 7 7 7 7 7 7
#      The same shape in the base task would only have a 3x3 orange neighborhood and black centers.
# b) Complexity calibration:
#    - The rule is intentionally limited to three radius levels {1, 2, 3}; no larger size buckets
#      are introduced, so the mapping is visible and learnable.
#    - The generator forces every accepted sample to contain true size-1, size-2, and size-3
#      interior defects. That prevents the combo from collapsing back to the base puzzle or to
#      a pure "red-center recolor" with no visible radius variation.
# c) Broken original assumptions:
#    - "All halos are radius 1" is false. The write step must inspect the connected size of each
#      small defect component before painting.
#    - "Preserved target pixels stay black" is false. Centers must be written last in red so halo
#      overlap cannot erase them.
#    - "Every marked target is already a singleton" is false for size-2/3 defects. The solver must
#      keep the component grouping so the shared size can control every center pixel in that group.
# d) Placement legality:
#    - Interior defects are still placed strictly inside blue regions and never 4-connect to the
#      surrounding large black regions or to each other.
#    - Leaves are only grown from large black regions into adjacent blue cells and are buffered away
#      from the larger defects enough that radius-2/3 halos remain readable.
#    - Spurs are only injected at cells that the solver's absorption rule will deterministically
#      flip back to black.
# e) Visual observability:
#    - A human can infer the fused rule from a few examples because the red centers expose the
#      defect component size directly, and the orange spread repeats at three visibly different
#      radii around 1-cell, 2-cell, and 3-cell red groups.
#    - Repetition matters, so accepted samples always include all three small-defect sizes.
# f) Visual neatness:
#    - The generator rejects crowded layouts where size-2 or size-3 defects fail to show any
#      exclusive orange area beyond radius 1. This keeps the size coding visible instead of letting
#      overlaps turn the board into an indistinct orange cloud.
# g) Color inference:
#    - Only four colors appear: black/blue input colors plus red centers and orange halos.
#      Red always marks the defect pixels themselves; orange always marks their surrounding halo.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes every true radius-bearing multi-pixel target is a size-2 or size-3 black
#      component isolated inside blue; `generate_input()` constructs them exactly that way.
#    - `main()` assumes leaves from large black masses should still be handled as singleton targets;
#      the generator keeps that original object family alive, but never lets it be the only target type.
#    - `main()` assumes center precedence over halo paint. The generator permits halo overlap, but
#      red centers are always restored last, matching the intended layering.


BLUE = 1
BLACK = 0
RED = 2
ORANGE = 7

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
REQUIRED_DEFECT_SIZES = [1, 2, 3]
EXTRA_DEFECT_RANGE = (0, 2)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 60
LEAF_COUNT_RANGE = (0, 3)
LEAF_DEFECT_BUFFER = 3
SPUR_COUNT_RANGE = (2, 6)
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _effective_black_count(grid: np.ndarray, r: int, c: int) -> int:
    """Count black 4-neighbors, with out-of-bounds counted only at corners."""
    h, w = grid.shape
    black_neighbors = 0
    oob = 0
    for rr, cc in _neighbors4(r, c):
        if not (0 <= rr < h and 0 <= cc < w):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_neighbors += 1
    if oob >= 2:
        black_neighbors += oob
    return int(black_neighbors)


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
    """Apply the base task's blue-spur absorption step."""
    base = np.array(grid, dtype=int)
    h, w = base.shape
    for r in range(h):
        for c in range(w):
            if int(base[r, c]) != BLUE:
                continue
            if _effective_black_count(base, r, c) >= 3:
                base[r, c] = BLACK
    return base


def _collect_targets(base: np.ndarray) -> list[dict]:
    """Collect small black components and leaf pixels from large black components."""
    h, w = base.shape
    targets: list[dict] = []
    for coords in components_mask(base == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            targets.append({"pixels": set(pts), "size": len(pts), "kind": "defect"})
            continue
        comp_set = set(pts)
        for r, c in pts:
            black_deg = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) in comp_set:
                    black_deg += 1
            if black_deg <= 1:
                targets.append({"pixels": {(r, c)}, "size": 1, "kind": "leaf"})
    return targets


def _halo_footprint(pixels: set[tuple[int, int]], radius: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    """All in-bounds cells in the Chebyshev-radius footprint around the given pixels."""
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((int(rr), int(cc)))
    return out


def _paint_output(base: np.ndarray, targets: list[dict]) -> np.ndarray:
    """Paint orange halos first, then restore target centers in red."""
    out = np.array(base, dtype=int)
    centers = set().union(*(item["pixels"] for item in targets)) if targets else set()
    for item in targets:
        radius = max(1, min(3, int(item["size"])))
        for rr, cc in _halo_footprint(set(item["pixels"]), radius, base.shape):
            if (rr, cc) not in centers:
                out[rr, cc] = ORANGE
    for r, c in centers:
        out[r, c] = RED
    return out


def _baseline_output(base: np.ndarray, targets: list[dict]) -> np.ndarray:
    """Base-task output, used only for generator-side anti-degeneration checks."""
    out = np.array(base, dtype=int)
    centers = set().union(*(item["pixels"] for item in targets)) if targets else set()
    for item in targets:
        for rr, cc in _halo_footprint(set(item["pixels"]), 1, base.shape):
            if (rr, cc) not in centers:
                out[rr, cc] = ORANGE
    for r, c in centers:
        out[r, c] = BLACK
    return out


def _clean_targets_and_output(grid: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Shared oracle pipeline for both solver and generator."""
    base = _absorb_spurs(grid)
    targets = _collect_targets(base)
    return base, targets, _paint_output(base, targets)


def _visibility_metrics(base: np.ndarray, targets: list[dict], output: np.ndarray) -> dict:
    """Quantify whether the fused behavior is visibly present in a sample."""
    all_footprints = []
    center_set = set().union(*(item["pixels"] for item in targets)) if targets else set()
    for item in targets:
        radius = max(1, min(3, int(item["size"])))
        all_footprints.append(_halo_footprint(set(item["pixels"]), radius, base.shape))

    size_counts = {1: 0, 2: 0, 3: 0}
    red_defect_pixels = 0
    extended_visible = {2: 0, 3: 0}
    for idx, item in enumerate(targets):
        if item["kind"] == "defect":
            size = int(item["size"])
            size_counts[size] += 1
            red_defect_pixels += len(item["pixels"])
            if size >= 2:
                full = all_footprints[idx]
                inner = _halo_footprint(set(item["pixels"]), 1, base.shape)
                extended = full - inner - center_set
                others = set().union(*(fp for j, fp in enumerate(all_footprints) if j != idx)) if len(all_footprints) > 1 else set()
                visible = sum(1 for cell in extended if cell not in others and int(output[cell]) == ORANGE)
                extended_visible[size] += int(visible)

    baseline = _baseline_output(base, targets)
    delta_vs_baseline = int(np.count_nonzero(output != baseline))
    return {
        "size_counts": size_counts,
        "red_defect_pixels": red_defect_pixels,
        "extended_visible": extended_visible,
        "delta_vs_baseline": delta_vs_baseline,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black defects in blue regions with size-coded orange halos and red centers.

    Input structure:
    - Blue (1) foreground regions sit against black (0) background regions.
    - Inside blue areas there are small black defect components of size 1..3, and large black areas may
      also contribute dangling leaf pixels that poke into blue.
    - Some black areas additionally contain blue spur pixels that should first be absorbed back into black.

    Rule:
    - First absorb any blue pixel that has at least three effective black 4-neighbors, using the same
      corner-counting rule as the base task.
    - On that cleaned grid, treat every black component of size <= 3 as one defect group, and treat every
      degree-0/1 leaf pixel of larger black components as its own singleton target.
    - Paint each target with an orange Chebyshev halo whose radius equals its target size (1, 2, or 3),
      then overwrite the target pixels themselves in red (2).

    Key insight: The red center shape tells you the connected defect size, and that same size controls how
    far the surrounding orange halo expands.
    """
    _, targets, out = _clean_targets_and_output(grid)
    return out


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    """Return True if any pixel in `ref` is within Chebyshev distance `dist`."""
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(pixels: set[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    """All coordinates within Chebyshev distance `dist` of any given pixel."""
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((int(r + dr), int(c + dc)))
    return zone


def _build_partition(h: int, w: int) -> np.ndarray:
    """Create a grid partitioned into large blue and black regions."""
    grid = np.zeros((h, w), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cuts = sorted(random.sample(range(CUT_MARGIN, h - CUT_MARGIN), min(n_cuts, h - 2 * CUT_MARGIN)))
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
        if (r1 - r0 > INNER_INSET_MIN_GAP and c1 - c0 > INNER_INSET_MIN_GAP and random.random() < INNER_INSET_PROB):
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
    """Blue pixels with four in-bounds blue neighbors, excluding the outer border."""
    h, w = grid.shape
    coords = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLUE:
                continue
            if all(int(grid[rr, cc]) == BLUE for rr, cc in _neighbors4(r, c)):
                coords.append((int(r), int(c)))
    random.shuffle(coords)
    return coords


def _footprint_in_bounds(pixels: set[tuple[int, int]], radius: int, shape: tuple[int, int]) -> bool:
    """Return True when the full Chebyshev-radius halo fits inside the canvas."""
    h, w = shape
    for r, c in pixels:
        if r - radius < 0 or r + radius >= h or c - radius < 0 or c + radius >= w:
            return False
    return True


def _valid_defect_shape(grid: np.ndarray, defect: set[tuple[int, int]], existing: set[tuple[int, int]]) -> bool:
    """Every non-defect 4-neighbor must still be blue and unused."""
    h, w = grid.shape
    radius = len(defect)
    if not _footprint_in_bounds(defect, radius, grid.shape):
        return False
    for r, c in defect:
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in defect:
                continue
            if 0 <= rr < h and 0 <= cc < w:
                if int(grid[rr, cc]) != BLUE or (rr, cc) in existing:
                    return False
    return True


def _grow_defect(
    grid: np.ndarray,
    seed: tuple[int, int],
    size: int,
    interior_set: set[tuple[int, int]],
    existing: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    """Backtrack an exact-size connected defect of up to 3 cells."""
    if _nearby(seed[0], seed[1], DEFECT_CLEARANCE, existing):
        return None

    def rec(current: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
        if len(current) == size:
            return set(current) if _valid_defect_shape(grid, current, existing) else None
        frontier: set[tuple[int, int]] = set()
        for r, c in current:
            for rr, cc in _neighbors4(r, c):
                if (rr, cc) in current or (rr, cc) not in interior_set or (rr, cc) in existing:
                    continue
                if _nearby(rr, cc, DEFECT_CLEARANCE, existing):
                    continue
                frontier.add((int(rr), int(cc)))
        candidates = list(frontier)
        random.shuffle(candidates)
        for cand in candidates:
            result = rec(current | {cand})
            if result is not None:
                return result
        return None

    return rec({(int(seed[0]), int(seed[1]))})


def _place_defects(grid: np.ndarray, interior_blue: list[tuple[int, int]]) -> tuple[list[set[tuple[int, int]]], set[tuple[int, int]]]:
    """Place mandatory size-1/2/3 defects plus a few extras."""
    interior_set = set(interior_blue)
    defect_groups: list[set[tuple[int, int]]] = []
    defect_pixels: set[tuple[int, int]] = set()

    sizes = list(REQUIRED_DEFECT_SIZES)
    extra_count = random.randint(*EXTRA_DEFECT_RANGE)
    sizes.extend(random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS, k=extra_count))
    random.shuffle(sizes)

    for size in sizes:
        placed = None
        for _ in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed = interior_blue[random.randint(0, len(interior_blue) - 1)]
            placed = _grow_defect(grid, seed, size, interior_set, defect_pixels)
            if placed is not None:
                break
        if placed is None:
            return [], set()
        defect_groups.append(set(placed))
        defect_pixels.update(placed)
        remove = _exclusion_zone(set(placed), DEFECT_CLEARANCE)
        interior_blue[:] = [p for p in interior_blue if p not in remove]
        interior_set -= remove

    return defect_groups, defect_pixels


def _place_leaves(grid: np.ndarray, defect_pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Place a few singleton leaves on the boundary of large black regions."""
    h, w = grid.shape
    leaf_pixels: set[tuple[int, int]] = set()
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    black_comps = components_mask(grid == BLACK, connectivity=4)
    large_black = [set((int(r), int(c)) for r, c in comp) for comp in black_comps if len(comp) > 3]

    for _ in range(n_leaves):
        if not large_black:
            break
        comp_set = random.choice(large_black)
        candidates = []
        for r, c in comp_set:
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if int(grid[rr, cc]) != BLUE or (rr, cc) in leaf_pixels:
                    continue
                if _nearby(rr, cc, LEAF_DEFECT_BUFFER, defect_pixels):
                    continue
                black_n = sum(
                    1
                    for r2, c2 in _neighbors4(rr, cc)
                    if 0 <= r2 < h and 0 <= c2 < w and (int(grid[r2, c2]) == BLACK or (r2, c2) in leaf_pixels)
                )
                if black_n == 1:
                    candidates.append((int(rr), int(cc)))
        if candidates:
            lr, lc = random.choice(candidates)
            leaf_pixels.add((int(lr), int(lc)))
            grid[lr, lc] = BLACK

    return leaf_pixels


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black cells that can safely become blue spurs."""
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK or (r, c) in exclude:
                continue
            trial = np.array(grid, dtype=int)
            trial[r, c] = BLUE
            if _effective_black_count(trial, r, c) >= 3:
                candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _sample_ok(base: np.ndarray, targets: list[dict], output: np.ndarray) -> bool:
    """Reject samples where the combo behavior is not visibly present."""
    metrics = _visibility_metrics(base, targets, output)
    if metrics["size_counts"][1] < 1 or metrics["size_counts"][2] < 1 or metrics["size_counts"][3] < 1:
        return False
    if metrics["extended_visible"][2] <= 0 or metrics["extended_visible"][3] <= 0:
        return False
    if metrics["red_defect_pixels"] < 6:
        return False
    if metrics["delta_vs_baseline"] <= 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid fused-combo input."""
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
    defect_groups, defect_pixels = _place_defects(grid, interior_blue)
    if not defect_groups:
        return None
    for group in defect_groups:
        for r, c in group:
            grid[r, c] = BLACK

    _place_leaves(grid, defect_pixels)

    base, targets, output = _clean_targets_and_output(grid)
    if not _sample_ok(base, targets, output):
        return None

    inp = np.array(grid, dtype=int)
    spur_candidates = _find_spur_candidates(inp, set().union(*(item["pixels"] for item in targets)) if targets else set())
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    if min(n_spurs, len(spur_candidates)) < MIN_SPURS:
        return None
    for r, c in spur_candidates[:n_spurs]:
        inp[r, c] = BLUE

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
