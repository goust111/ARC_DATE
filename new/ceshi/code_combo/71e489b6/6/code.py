# Combo variant of 71e489b6 (index 6)
# combo_spec: D1/1+D5/1
# Name: 可变半径光晕+大小编码颜色
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
#    - Original: every size-1/2/3 defect and every large-black leaf got the same orange
#      Chebyshev-1 halo, so a 3-cell defect looked almost the same as a 1-cell defect.
#    - Fused: first absorb blue spurs exactly as before, then detect two target kinds:
#      small black components of size 1..3, plus leaf pixels from larger black components.
#      Each target object receives one size bucket. Bucket 1 uses radius 1 + orange (7),
#      bucket 2 uses radius 2 + yellow (4), bucket 3 uses radius 3 + magenta (6).
#      Large-component leaves are single pixels, so they are bucket 1.
#    - Example: an isolated 2-cell domino defect at {(5,5),(5,6)} used to produce one
#      orange 4x3-ish union halo. Now it produces a yellow union of Chebyshev-2 squares,
#      while a 3-cell tromino produces a larger magenta footprint.
# b) Complexity calibration:
#    - Only three size buckets are used, matching the original defect-size range.
#      That keeps the rule inferable while making the output visibly different.
#    - The generator forces repeated support for buckets 1/2/3 so the radius-color
#      correspondence is observable from a few examples instead of being a rare corner case.
# c) Hidden assumptions review:
#    - The original STEP 1 assumed every small component can be handled uniformly.
#      Here we still group by component, but now component size is an active parameter.
#    - The original painting loop assumed one halo color and radius. That assumption is
#      removed: painting now consumes a per-target bucket.
#    - Overlapping halos of different buckets would create order-sensitive color conflicts.
#      The generator therefore keeps halo footprints disjoint; if arbitrary inputs violate
#      that, the solver resolves ties by painting larger buckets later.
# d) Placement legality:
#    - Defects are placed as exact 1/2/3-cell connected polyominoes strictly inside blue.
#    - Their full intended halo footprint must stay in bounds and avoid existing halo zones.
#    - Leaves are added one pixel at a time next to large black regions, also with disjoint
#      halo footprints. Spurs are inserted only in black cells that solver STEP 0 absorbs.
# e) Visual observability:
#    - Every generated puzzle contains at least one size-1, one size-2, and one size-3
#      defect, so the solver must explain both the color change and the halo-radius change.
# f) Visual cleanliness:
#    - Wide-radius halos are the main clutter risk, so object counts are reduced relative to
#      the base task and halo zones are kept separate to preserve readable silhouettes.
# g) Rule inferability:
#    - The only changing attribute is object size, and both affected channels (radius, color)
#      vary in lockstep with that size. This gives a strong structural cue instead of an
#      arbitrary color remapping.

HOLE = 0
BLUE = 1
MAGENTA = 6
ORANGE = 7
YELLOW = 4
HALO_BY_SIZE = {1: ORANGE, 2: YELLOW, 3: MAGENTA}

# --------------- solver helpers ---------------


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
    """Apply the original blue-spur absorption rule."""
    H, W = grid.shape
    out = grid.copy()
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != BLUE:
                continue
            black_neighbors = 0
            oob = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(grid[rr, cc]) == HOLE:
                    black_neighbors += 1
            if oob >= 2:
                black_neighbors += oob
            if black_neighbors >= 3:
                out[r, c] = HOLE
    return out


def _collect_targets(base: np.ndarray) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    """Return target descriptors plus all black pixels that must remain black."""
    H, W = base.shape
    targets: list[dict[str, object]] = []
    preserve: set[tuple[int, int]] = set()
    comps = components_mask(base == HOLE, connectivity=4)
    for coords in comps:
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            bucket = len(pts)
            targets.append({"pixels": pts, "bucket": bucket, "kind": "small_component"})
            preserve.update(pts)
            continue
        for r, c in pts:
            black_neighbors = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == HOLE:
                    black_neighbors += 1
            if black_neighbors <= 1:
                targets.append({"pixels": [(r, c)], "bucket": 1, "kind": "leaf"})
                preserve.add((r, c))
    return targets, preserve


def _paint_halos(base: np.ndarray, targets: list[dict[str, object]], preserve: set[tuple[int, int]]) -> np.ndarray:
    """Render size-coded halos while keeping the target black pixels untouched."""
    H, W = base.shape
    out = base.copy()
    for target in sorted(targets, key=lambda item: int(item["bucket"])):
        bucket = int(target["bucket"])
        radius = bucket
        color = int(HALO_BY_SIZE[bucket])
        pixels = target["pixels"]
        for r, c in pixels:
            rr0 = max(0, int(r - radius))
            rr1 = min(H - 1, int(r + radius))
            cc0 = max(0, int(c - radius))
            cc1 = min(W - 1, int(c + radius))
            for rr in range(rr0, rr1 + 1):
                for cc in range(cc0, cc1 + 1):
                    if (rr, cc) not in preserve:
                        out[rr, cc] = color
    for r, c in preserve:
        out[r, c] = HOLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect black defects inside blue regions and mark them with size-coded halos.

    Input structure:
    - Blue (1) and black (0) form large partitioned regions.
    - Small black components of size 1-3 may sit inside blue areas, and large black regions
      may have single-pixel leaf tips extending into blue.
    - Some isolated blue spur pixels can also appear inside black regions.

    Rule:
    - First absorb any blue spur whose effective black 4-neighborhood count is at least 3.
    - Then treat each black component of size 1/2/3 as one defect object; treat each leaf
      pixel of any larger black component as a separate size-1 target.
    - Paint a Chebyshev halo around each target object using both size channels:
      size 1 -> orange radius 1, size 2 -> yellow radius 2, size 3 -> magenta radius 3.
      The target pixels themselves remain black.

    Key insight: the same structural anomaly test as the base task is still used, but defect
    size now controls two visible outputs at once: halo extent and halo color.
    """
    g = np.array(grid, dtype=int)
    base = _absorb_spurs(g)
    targets, preserve = _collect_targets(base)
    return _paint_halos(base, targets, preserve)


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
MANDATORY_DEFECT_SIZES = (1, 2, 3)
EXTRA_DEFECT_RANGE = (0, 1)
EXTRA_DEFECT_WEIGHTS = [4, 3, 2]
LEAF_COUNT_RANGE = (0, 2)
SPUR_COUNT_RANGE = (2, 5)
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 400

DEFECT_SHAPES = {
    1: [[(0, 0)]],
    2: [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
    ],
    3: [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ],
}


def _halo_zone(pixels: list[tuple[int, int]], radius: int) -> set[tuple[int, int]]:
    """All cells in the Chebyshev-radius halo around an object."""
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                zone.add((int(r + dr), int(c + dc)))
    return zone


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create an HxW grid split into alternating blue/black regions."""
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        options = list(range(CUT_MARGIN, H - CUT_MARGIN))
        if not options:
            return grid
        cuts = sorted(random.sample(options, min(n_cuts, len(options))))
        bounds = [0] + cuts + [H]
        first_color = random.randint(0, 1)
        for i in range(len(bounds) - 1):
            grid[bounds[i]:bounds[i + 1], :] = (first_color + i) % 2

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


def _regions_large_enough(grid: np.ndarray) -> bool:
    """Ensure the partition still resembles the base task."""
    blue_ok = any(len(comp) >= MIN_BLUE_REGION for comp in components_mask(grid == BLUE, connectivity=4))
    black_ok = any(len(comp) >= MIN_BLACK_REGION for comp in components_mask(grid == HOLE, connectivity=4))
    return blue_ok and black_ok


def _valid_defect_shape(
    grid: np.ndarray,
    pixels: list[tuple[int, int]],
    bucket: int,
    used_halo: set[tuple[int, int]],
) -> bool:
    """Check a candidate defect against color, adjacency, and halo-spacing rules."""
    H, W = grid.shape
    pixel_set = set(pixels)
    zone = _halo_zone(pixels, bucket)
    for rr, cc in zone:
        if not (0 <= rr < H and 0 <= cc < W):
            return False
    if zone & used_halo:
        return False
    for r, c in pixels:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != BLUE:
            return False
    for r, c in pixels:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if (rr, cc) in pixel_set:
                continue
            if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != BLUE:
                return False
    return True


def _place_defects(grid: np.ndarray) -> tuple[list[dict[str, object]], set[tuple[int, int]], bool]:
    """Place exact-size black components while keeping halo footprints separate."""
    H, W = grid.shape
    used_halo: set[tuple[int, int]] = set()
    targets: list[dict[str, object]] = []
    requested = list(MANDATORY_DEFECT_SIZES)
    for _ in range(random.randint(*EXTRA_DEFECT_RANGE)):
        requested.append(random.choices([1, 2, 3], weights=EXTRA_DEFECT_WEIGHTS)[0])
    random.shuffle(requested)

    anchors = [(r, c) for r in range(H) for c in range(W)]
    for bucket in requested:
        placed = False
        random.shuffle(anchors)
        shapes = list(DEFECT_SHAPES[bucket])
        for ar, ac in anchors:
            random.shuffle(shapes)
            for shape in shapes:
                pixels = [(int(ar + dr), int(ac + dc)) for dr, dc in shape]
                if not _valid_defect_shape(grid, pixels, bucket, used_halo):
                    continue
                for r, c in pixels:
                    grid[r, c] = HOLE
                used_halo.update(_halo_zone(pixels, bucket))
                targets.append({"pixels": pixels, "bucket": bucket, "kind": "small_component"})
                placed = True
                break
            if placed:
                break
        if not placed:
            return [], set(), False

    return targets, used_halo, True


def _place_leaves(
    grid: np.ndarray,
    used_halo: set[tuple[int, int]],
) -> list[dict[str, object]]:
    """Add a few single-pixel leaves with bucket-1 halos."""
    H, W = grid.shape
    leaf_targets: list[dict[str, object]] = []
    desired = random.randint(*LEAF_COUNT_RANGE)
    if desired == 0:
        return leaf_targets

    for _ in range(desired):
        large_black = [comp for comp in components_mask(grid == HOLE, connectivity=4) if len(comp) > 3]
        if not large_black:
            break
        random.shuffle(large_black)
        placed = False
        for comp in large_black:
            comp_set = {(int(r), int(c)) for r, c in comp}
            candidates: list[tuple[int, int]] = []
            for r, c in comp_set:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if not (0 <= rr < H and 0 <= cc < W):
                        continue
                    if int(grid[rr, cc]) != BLUE:
                        continue
                    black_neighbors = 0
                    for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        r2 = int(rr + dr2)
                        c2 = int(cc + dc2)
                        if 0 <= r2 < H and 0 <= c2 < W and int(grid[r2, c2]) == HOLE:
                            black_neighbors += 1
                    if black_neighbors != 1:
                        continue
                    zone = _halo_zone([(rr, cc)], 1)
                    if any(not (0 <= zr < H and 0 <= zc < W) for zr, zc in zone):
                        continue
                    if zone & used_halo:
                        continue
                    candidates.append((rr, cc))
            if not candidates:
                continue
            lr, lc = random.choice(candidates)
            grid[lr, lc] = HOLE
            used_halo.update(_halo_zone([(lr, lc)], 1))
            leaf_targets.append({"pixels": [(lr, lc)], "bucket": 1, "kind": "leaf"})
            placed = True
            break
        if not placed:
            break
    return leaf_targets


def _build_expected_output(grid: np.ndarray, targets: list[dict[str, object]]) -> np.ndarray:
    """Render the target-coded output from the clean base grid."""
    preserve = {tuple(p) for target in targets for p in target["pixels"]}
    return _paint_halos(grid, targets, preserve)


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Black cells that become blue spurs and will be absorbed back by STEP 0."""
    H, W = grid.shape
    black = (grid == HOLE)
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
    out = [(int(r), int(c)) for r, c in zip(rs, cs) if (int(r), int(c)) not in exclude]
    random.shuffle(out)
    return out


def _gen_attempt() -> np.ndarray | None:
    """One output-first generation attempt."""
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)
    if not _regions_large_enough(grid):
        return None

    defect_targets, used_halo, ok = _place_defects(grid)
    if not ok:
        return None
    leaf_targets = _place_leaves(grid, used_halo)
    all_targets = defect_targets + leaf_targets

    # Guard against accidental regressions: the three encoded buckets must be present.
    present_buckets = {int(target["bucket"]) for target in defect_targets}
    if present_buckets != {1, 2, 3}:
        return None

    output = _build_expected_output(grid, all_targets)

    inp = grid.copy()
    exclude = set(used_halo)
    exclude.update(tuple(p) for target in all_targets for p in target["pixels"])
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    candidates = _find_spur_candidates(inp, exclude)
    if min(n_spurs, len(candidates)) < MIN_SPURS:
        return None
    for r, c in candidates[:n_spurs]:
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
