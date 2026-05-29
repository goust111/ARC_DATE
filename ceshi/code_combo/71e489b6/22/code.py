# Combo variant of 71e489b6 (index 22)
# combo_spec: D1/1+D5/1+D12/1
# Name: 可变半径+大小编码颜色+红色中心
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
# a) Concrete behavior trace:
#    - Original pipeline: absorb blue spurs inside black regions, then detect two target
#      families: (1) black 4-components of size 1..3 strictly inside blue, and (2) leaf
#      pixels of larger black components. Every target got the same orange Chebyshev-1 halo
#      while the target pixels themselves stayed black.
#    - Fused pipeline: the target detection logic stays the same, but target size now drives
#      three visible channels at once:
#        size 1 -> orange halo radius 1, red center of size 1
#        size 2 -> yellow halo radius 2, red center of size 2
#        size 3 -> magenta halo radius 3, red center of size 3
#      Leaves are still single pixels, so they always use the size-1 code.
#    - Local example for a 2-cell horizontal defect:
#        input:
#          1111
#          1001
#          1111
#        original output:
#          7777
#          7007
#          7777
#        fused output:
#          4444
#          4224
#          4444
#      The same hidden variable "defect size" is now visible as halo radius, halo color, and
#      the length of the red center component itself.
# b) Complexity calibration:
#    - The size vocabulary is capped at three buckets because the source task already limits
#      isolated defects to sizes 1, 2, and 3. This keeps the mapping discrete and inferable.
#    - The generator forces all three buckets to appear in every sample and usually adds one
#      extra defect, so the size-coded output does not collapse into a one-off coincidence.
# c) Implicit-assumption review of the original solver:
#    - Spur absorption relied on the exact in-place scan rule "blue with >=3 effective black
#      4-neighbors becomes black". That assumption still holds, so the generator preserves it.
#    - The original branch "all small components share one halo radius/color" is no longer
#      valid. The solver must carry component size forward into painting.
#    - The original final restoration loop forced targets back to black. That assumption breaks
#      under D12/1: targets must instead become red after halo painting.
#    - The original painter was safe under arbitrary overlap because every halo used the same
#      color. Here overlap between size buckets would create color-order ambiguity, so the
#      generator reserves each target's full halo footprint and keeps them disjoint.
# d) Placement legality:
#    - Every planted size-1/2/3 defect is a connected black polyomino whose non-defect
#      4-neighbors remain blue, so it is detected as a valid small black component.
#    - Every leaf is created by flipping a blue cell adjacent to exactly one black cell of a
#      large component, keeping it in-bounds and preserving leaf degree <= 1.
#    - Every target's full halo footprint stays in-bounds and avoids all earlier footprints.
#    - Spurs are added only on black cells that the solver's cleanup pass deterministically
#      absorbs back to black.
# e) Visual observability:
#    - A human sees the same structural anomaly class repeated with three coupled effects:
#      bigger black defects become bigger red centers, wider halos, and different halo colors.
#    - Because the generator includes size-1, size-2, and size-3 defects in every sample,
#      the size-to-radius and size-to-color correspondences are repeatedly supported.
# f) Visual cleanliness:
#    - Radius-3 halos are large and visually dominant, so target count is reduced relative to
#      the base task and halo zones are kept disjoint to avoid unreadable color collisions.
#    - Leaves remain optional and sparse; when present they reinforce the size-1 code without
#      flooding the grid with extra orange.
# g) Change inferability:
#    - The output palette stays small: red centers plus three halo colors keyed only by visible
#      defect size. No hidden use of numeric color ids, scan order, or non-visual tie-breakers
#      is required for a solver to infer the rule.
# Additional solver/generator contract:
#    - `main()` assumes size buckets are meaningful even when multiple targets exist. The
#      generator guarantees each bucket appears and keeps their footprints separated, so the
#      three-channel coding remains legible instead of degenerating to the source behavior.


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
MAGENTA = 6
ORANGE = 7

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
HALO_BY_SIZE = {1: ORANGE, 2: YELLOW, 3: MAGENTA}

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
EXTRA_DEFECT_RANGE = (1, 1)
EXTRA_DEFECT_WEIGHTS = [5, 4, 2]
LEAF_COUNT_RANGE = (1, 1)
MIN_LEAF_TARGETS = 1
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


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
    """Apply the source task's in-place blue-spur absorption rule."""
    H, W = grid.shape
    base = np.array(grid, dtype=int)
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            black_n = 0
            oob = 0
            for dr, dc in DIRS4:
                rr = int(r + dr)
                cc = int(c + dc)
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


def _collect_targets(base: np.ndarray) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    """Return target descriptors plus the union of all target-center pixels."""
    H, W = base.shape
    targets: list[dict[str, object]] = []
    centers: set[tuple[int, int]] = set()
    for coords in components_mask(base == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            targets.append({"pixels": pts, "bucket": len(pts), "kind": "small_component"})
            centers.update(pts)
            continue
        for r, c in pts:
            degree = 0
            for dr, dc in DIRS4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == BLACK:
                    degree += 1
            if degree <= 1:
                targets.append({"pixels": [(r, c)], "bucket": 1, "kind": "leaf"})
                centers.add((r, c))
    return targets, centers


def _paint_output(
    base: np.ndarray,
    targets: list[dict[str, object]],
    centers: set[tuple[int, int]],
) -> np.ndarray:
    """Paint size-coded halos, then paint every target center red."""
    H, W = base.shape
    out = np.array(base, dtype=int)
    for target in sorted(targets, key=lambda item: int(item["bucket"])):
        bucket = int(target["bucket"])
        color = int(HALO_BY_SIZE[bucket])
        radius = bucket
        for r, c in target["pixels"]:
            rr0 = max(0, int(r - radius))
            rr1 = min(H - 1, int(r + radius))
            cc0 = max(0, int(c - radius))
            cc1 = min(W - 1, int(c + radius))
            for rr in range(rr0, rr1 + 1):
                for cc in range(cc0, cc1 + 1):
                    if (rr, cc) not in centers:
                        out[rr, cc] = color
    for r, c in centers:
        out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect black anomalies in blue regions and encode their size with halo radius, halo color,
    and a red center marker.

    Input structure:
    - The input uses blue (1) and black (0) only.
    - Large blue and black regions form a rectilinear partition of the board.
    - Small black components of size 1-3 can lie inside blue, large black regions may have
      single-pixel leaves protruding into blue, and some blue spur pixels may sit inside black.

    Rule:
    - First absorb any blue spur whose effective black 4-neighbor count is at least 3, using
      the same cleanup pass as the source task.
    - Then detect every black component of size 1, 2, or 3, plus every degree-0/1 leaf pixel
      belonging to a larger black component.
    - Mark each target with size-coded rendering:
      size 1 -> orange Chebyshev-1 halo, size 2 -> yellow Chebyshev-2 halo,
      size 3 -> magenta Chebyshev-3 halo. The target pixels themselves become red.

    Key insight: the anomaly detector is almost unchanged from the original puzzle, but the
    output now exposes target size redundantly through three aligned cues: center size, halo
    color, and halo extent.
    """
    g = np.array(grid, dtype=int)
    base = _absorb_spurs(g)
    targets, centers = _collect_targets(base)
    return _paint_output(base, targets, centers)


def _halo_zone(pixels: list[tuple[int, int]], radius: int) -> set[tuple[int, int]]:
    """Return the full Chebyshev-radius footprint around an object."""
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                zone.add((int(r + dr), int(c + dc)))
    return zone


def _partition_valid(grid: np.ndarray) -> bool:
    """Keep only partitions with sufficiently large blue and black regions."""
    for comp in components_mask(grid == BLUE, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return False
    for comp in components_mask(grid == BLACK, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return False
    return True


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create a source-style partition of alternating blue and black regions."""
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


def _valid_defect_shape(
    grid: np.ndarray,
    pixels: list[tuple[int, int]],
    bucket: int,
    used_halo: set[tuple[int, int]],
) -> bool:
    """Check whether a candidate size bucket can be carved from blue safely."""
    H, W = grid.shape
    pixel_set = set(pixels)
    zone = _halo_zone(pixels, bucket)
    for rr, cc in zone:
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if int(grid[rr, cc]) != BLUE:
            return False
    if zone & used_halo:
        return False
    for r, c in pixels:
        if int(grid[r, c]) != BLUE:
            return False
    for r, c in pixels:
        for dr, dc in DIRS4:
            rr = int(r + dr)
            cc = int(c + dc)
            if (rr, cc) in pixel_set:
                continue
            if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != BLUE:
                return False
    return True


def _place_defects(grid: np.ndarray) -> tuple[list[dict[str, object]], set[tuple[int, int]], bool]:
    """Plant exact-size black defects while reserving each future halo footprint."""
    H, W = grid.shape
    anchors = [(r, c) for r in range(H) for c in range(W)]
    used_halo: set[tuple[int, int]] = set()
    targets: list[dict[str, object]] = []
    requested = list(MANDATORY_DEFECT_SIZES)
    for _ in range(random.randint(*EXTRA_DEFECT_RANGE)):
        requested.append(random.choices([1, 2, 3], weights=EXTRA_DEFECT_WEIGHTS)[0])
    random.shuffle(requested)

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
                    grid[r, c] = BLACK
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
    """Optionally add a few degree-1 leaf pixels, always using the size-1 code."""
    H, W = grid.shape
    leaf_targets: list[dict[str, object]] = []
    desired = random.randint(*LEAF_COUNT_RANGE)
    for _ in range(desired):
        large_black = [comp for comp in components_mask(grid == BLACK, connectivity=4) if len(comp) > 3]
        if not large_black:
            break
        random.shuffle(large_black)
        placed = False
        for comp in large_black:
            comp_set = {(int(r), int(c)) for r, c in comp}
            candidates: list[tuple[int, int]] = []
            for r, c in comp_set:
                for dr, dc in DIRS4:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if not (0 <= rr < H and 0 <= cc < W):
                        continue
                    if int(grid[rr, cc]) != BLUE:
                        continue
                    black_n = 0
                    for dr2, dc2 in DIRS4:
                        r2 = int(rr + dr2)
                        c2 = int(cc + dc2)
                        if 0 <= r2 < H and 0 <= c2 < W and int(grid[r2, c2]) == BLACK:
                            black_n += 1
                    if black_n != 1:
                        continue
                    zone = _halo_zone([(rr, cc)], 1)
                    if zone & used_halo:
                        continue
                    if any(not (0 <= zr < H and 0 <= zc < W) for zr, zc in zone):
                        continue
                    candidates.append((rr, cc))
            if not candidates:
                continue
            lr, lc = random.choice(candidates)
            grid[lr, lc] = BLACK
            used_halo.update(_halo_zone([(lr, lc)], 1))
            leaf_targets.append({"pixels": [(lr, lc)], "bucket": 1, "kind": "leaf"})
            placed = True
            break
        if not placed:
            break
    return leaf_targets


def _build_expected_output(grid: np.ndarray, targets: list[dict[str, object]]) -> np.ndarray:
    centers = {tuple(pixel) for target in targets for pixel in target["pixels"]}
    return _paint_output(grid, targets, centers)


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black cells that can temporarily become absorbable blue spurs."""
    H, W = grid.shape
    black = (grid == BLACK)
    black_n = np.zeros((H, W), dtype=int)
    black_n[:-1, :] += black[1:, :]
    black_n[1:, :] += black[:-1, :]
    black_n[:, :-1] += black[:, 1:]
    black_n[:, 1:] += black[:, :-1]

    oob = np.zeros((H, W), dtype=int)
    oob[0, :] += 1
    oob[-1, :] += 1
    oob[:, 0] += 1
    oob[:, -1] += 1

    effective = black_n.copy()
    effective[oob >= 2] += oob[oob >= 2]

    rs, cs = np.where(black & (effective >= 3))
    out = [(int(r), int(c)) for r, c in zip(rs, cs) if (int(r), int(c)) not in exclude]
    random.shuffle(out)
    return out


def _gen_attempt() -> np.ndarray | None:
    """Build one valid input by constructing a clean board, then injecting spurs."""
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)
    if not _partition_valid(grid):
        return None

    defect_targets, used_halo, ok = _place_defects(grid)
    if not ok:
        return None
    leaf_targets = _place_leaves(grid, used_halo)
    if len(leaf_targets) < MIN_LEAF_TARGETS:
        return None
    all_targets = defect_targets + leaf_targets

    present_buckets = {int(target["bucket"]) for target in defect_targets}
    if present_buckets != {1, 2, 3}:
        return None

    output = _build_expected_output(grid, all_targets)

    inp = np.array(grid, dtype=int)
    exclude = set(used_halo)
    for target in all_targets:
        for pixel in target["pixels"]:
            exclude.add(tuple(pixel))

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
