# Combo variant of 71e489b6 (index 37)
# combo_spec: D11/2+D5/1
# Name: 层级优先级处理+大小编码颜色
from __future__ import annotations

# Design analysis for the fusion:
# a) Concrete behavior:
#    - Source task: absorb blue spurs in black, detect every black 4-component of size <= 3 and
#      every degree-0/1 pixel of larger black components, then paint one orange 3x3 halo.
#    - Fusion: keep the same detection, but classify each target by effective size:
#      size-3 component -> magenta halo, size-2 component -> yellow halo, size-1 target
#      (singleton defect or leaf pixel) -> orange halo. Rendering order now matters:
#      first size 3, then 2, then 1, so smaller targets overwrite larger-target halo cells.
#      Example chain:
#        00000 cleaned target layout by object sizes:  [3-column]  [2-column]  [1-dot]
#      If a cell belongs to both the size-3 and size-1 halos, the final color is orange, not
#      magenta. That visible overwrite is the D11/2 contribution; the color-size mapping is D5/1.
# b) Complexity calibration:
#    - Only three size buckets are used, matching the original <= 3 detection bound, so the rule is
#      learnable from a few examples. The generator forces repeated support for all three buckets.
# c) Hidden-assumption audit:
#    - Original STEP 3 assumed one global border color and order-independence. That no longer holds,
#      so target collection must retain per-object sizes and painting must preserve explicit order.
#    - Large black regions still contribute only leaf pixels, and each leaf is treated as a size-1
#      target. This keeps the old structural detector valid under the fused rule.
# d) Placement legality:
#    - Planned defects are carved only from blue 4-interior cells, remain mutually disconnected, and
#      stay away from later random defects/leaves. Spurs are added only to black cells that the same
#      absorption logic will deterministically flip back to black.
# e) Visual observability:
#    - Every accepted sample contains at least one size-3/size-2/size-1 overlap chain, so both the
#      three-color mapping and the "small overwrites large" priority are visible without hidden cues.
# f) Visual cleanliness:
#    - Only the planned chain is allowed to create mixed-size halo overlap. Extra defects and leaves
#      are buffered away with Chebyshev clearance so boards do not collapse into unreadable color soup.
# g) Inferability of color change:
#    - The palette is capped at three halo colors for the three legal sizes, and each accepted board
#      must show all three sizes at least once. This gives repeated evidence for the size->color rule.

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask  # noqa: E402


BLACK = 0
BLUE = 1
YELLOW = 4
MAGENTA = 6
ORANGE = 7

HALO_BY_SIZE = {1: ORANGE, 2: YELLOW, 3: MAGENTA}
PRIORITY_ORDER = (3, 2, 1)
REVERSE_PRIORITY_ORDER = (1, 2, 3)

# --------------- generator config ---------------

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

# The fused rule needs all three size buckets visible, so the generator places a fixed overlap chain
# first and only adds a few extra isolated defects afterwards.
EXTRA_DEFECT_COUNT_RANGE = (1, 4)
EXTRA_DEFECT_SIZE_WEIGHTS = [4, 3, 2]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 60
LEAF_COUNT_RANGE = (0, 2)
SPUR_COUNT_RANGE = (2, 5)
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 400

# Planned trio: size-3 column, size-2 column, size-1 singleton. Their halos overlap left-to-right,
# so order reversal visibly changes the output.
PRIORITY_PATTERN = {
    3: [(0, 0), (1, 0), (2, 0)],
    2: [(0, 2), (1, 2)],
    1: [(1, 4)],
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


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


def _halo_cells(
    shape: tuple[int, int],
    pixels: set[tuple[int, int]],
    preserve: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    H, W = shape
    halo: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    halo.add((rr, cc))
    return halo


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
    """Apply the original blue-spur absorption step."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            black_nb = 0
            oob = 0
            for rr, cc in _neighbors4(r, c):
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


def _collect_targets(base: np.ndarray) -> tuple[list[dict], set[tuple[int, int]]]:
    """Collect small black components and leaf pixels of larger black components."""
    targets: list[dict] = []
    preserve: set[tuple[int, int]] = set()

    for coords in components_mask(base == BLACK, connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        if len(pts) <= 3:
            targets.append({"pixels": pts, "size": len(pts), "kind": "small"})
            preserve.update(pts)
            continue

        for r, c in pts:
            degree = sum((rr, cc) in pts for rr, cc in _neighbors4(r, c))
            if degree <= 1:
                targets.append({"pixels": {(r, c)}, "size": 1, "kind": "leaf"})
                preserve.add((r, c))

    return targets, preserve


def _paint_targets(
    base: np.ndarray,
    targets: list[dict],
    preserve: set[tuple[int, int]],
    *,
    order: tuple[int, int, int] = PRIORITY_ORDER,
    monochrome: bool = False,
) -> np.ndarray:
    """Paint Chebyshev-1 halos with explicit size priority."""
    out = np.array(base, dtype=int)
    H, W = out.shape
    for size in order:
        color = ORANGE if monochrome else HALO_BY_SIZE[size]
        for item in targets:
            if int(item["size"]) != size:
                continue
            for r, c in item["pixels"]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                            out[rr, cc] = color

    for r, c in preserve:
        out[r, c] = BLACK
    return out


def _compute_metrics(
    base: np.ndarray,
    targets: list[dict],
    preserve: set[tuple[int, int]],
    fused_out: np.ndarray,
    reversed_out: np.ndarray,
    mono_out: np.ndarray,
) -> dict:
    coverage: dict[tuple[int, int], set[int]] = {}
    color_hist = {ORANGE: 0, YELLOW: 0, MAGENTA: 0}
    size_hist = {1: 0, 2: 0, 3: 0}

    for item in targets:
        size = int(item["size"])
        size_hist[size] += 1
        for cell in _halo_cells(base.shape, set(item["pixels"]), preserve):
            coverage.setdefault(cell, set()).add(size)

    for r in range(base.shape[0]):
        for c in range(base.shape[1]):
            val = int(fused_out[r, c])
            if val in color_hist:
                color_hist[val] += 1

    mixed_overlap_cells = sum(len(sizes) >= 2 for sizes in coverage.values())
    priority_difference_cells = int(np.count_nonzero(fused_out != reversed_out))
    color_difference_cells = int(np.count_nonzero(fused_out != mono_out))

    return {
        "size_hist": size_hist,
        "halo_color_hist": color_hist,
        "mixed_overlap_cells": mixed_overlap_cells,
        "priority_difference_cells": priority_difference_cells,
        "color_difference_cells": color_difference_cells,
    }


def behavior_metrics(grid: np.ndarray) -> dict:
    """Public helper used for post-generation degeneration checks."""
    base = _absorb_spurs(np.array(grid, dtype=int))
    targets, preserve = _collect_targets(base)
    fused = _paint_targets(base, targets, preserve, order=PRIORITY_ORDER)
    reversed_out = _paint_targets(base, targets, preserve, order=REVERSE_PRIORITY_ORDER)
    mono_out = _paint_targets(base, targets, preserve, order=PRIORITY_ORDER, monochrome=True)
    return _compute_metrics(base, targets, preserve, fused, reversed_out, mono_out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black defects in blue territory with size-coded halos whose overlaps resolve by priority.

    Input structure:
    - Blue (1) foreground regions are embedded in black (0) regions, with tiny black defects inside the blue and optional blue spur pixels inside black.
    - Defects can be singleton holes, 2-cell holes, 3-cell holes, or leaf tips of larger black components.

    Rule:
    - First absorb any blue pixel that is effectively surrounded by black on at least three 4-neighbor sides, using the same corner rule as the source task.
    - On that cleaned grid, detect every black 4-connected component of size 1-3 and every degree-0/1 pixel of larger black components.
    - Paint a Chebyshev-1 halo around those targets in three passes: size 3 uses magenta (6), size 2 uses yellow (4), and size 1 uses orange (7). The passes run in descending size order, so later smaller halos overwrite earlier larger halos; target pixels themselves stay black.

    Key insight:
    - The detector is still purely structural, but the output now encodes target size twice: by halo color and by overwrite precedence where small defects win on shared halo cells.
    """
    g = np.array(grid, dtype=int)
    base = _absorb_spurs(g)
    targets, preserve = _collect_targets(base)
    return _paint_targets(base, targets, preserve, order=PRIORITY_ORDER)


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create an HxW grid partitioned into large blue and black regions."""
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cuts = sorted(
            random.sample(
                range(CUT_MARGIN, H - CUT_MARGIN),
                min(n_cuts, H - 2 * CUT_MARGIN),
            )
        )
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

    else:  # l_shape
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


def _region_sizes_ok(grid: np.ndarray) -> bool:
    for comp in components_mask(grid == BLUE, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return False
    for comp in components_mask(grid == BLACK, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return False
    return True


def _find_interior_blue(grid: np.ndarray) -> list[tuple[int, int]]:
    """Blue pixels whose four in-bounds neighbors are also blue."""
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


def _valid_defect(
    grid: np.ndarray,
    defect: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    for r, c in defect:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if grid[r, c] != BLUE or (r, c) in occupied:
            return False
    for r, c in defect:
        for rr, cc in _neighbors4(r, c):
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            if (rr, cc) in defect:
                continue
            if grid[rr, cc] != BLUE or (rr, cc) in occupied:
                return False
    return True


def _place_priority_chain(grid: np.ndarray) -> list[dict] | None:
    """Place the mandatory 3->2->1 overlap chain inside one blue region."""
    H, W = grid.shape
    interior = set(_find_interior_blue(grid))
    anchors = [(r, c) for r in range(H - 2) for c in range(W - 4)]
    random.shuffle(anchors)

    for ar, ac in anchors:
        groups: list[dict] = []
        all_pixels: set[tuple[int, int]] = set()
        ok = True
        for size in PRIORITY_ORDER:
            pts = {(ar + dr, ac + dc) for dr, dc in PRIORITY_PATTERN[size]}
            if not pts <= interior:
                ok = False
                break
            if pts & all_pixels:
                ok = False
                break
            if not _valid_defect(grid, pts, all_pixels):
                ok = False
                break
            all_pixels.update(pts)
            groups.append({"pixels": pts, "size": size, "kind": "small"})

        if ok:
            for item in groups:
                for r, c in item["pixels"]:
                    grid[r, c] = BLACK
            return groups
    return None


def _grow_defect(
    seed: tuple[int, int],
    desired_size: int,
    interior: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    defect = {seed}
    frontier = [seed]
    while frontier and len(defect) < desired_size:
        current = frontier.pop(random.randrange(len(frontier)))
        nbrs = list(_neighbors4(*current))
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            if (nr, nc) in defect or (nr, nc) not in interior:
                continue
            if (nr, nc) in blocked or (nr, nc) in occupied:
                continue
            defect.add((nr, nc))
            frontier.append((nr, nc))
            if len(defect) >= desired_size:
                break
    return defect


def _place_extra_defects(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> list[dict]:
    interior_list = _find_interior_blue(grid)
    interior = set(interior_list)
    groups: list[dict] = []
    count = random.randint(*EXTRA_DEFECT_COUNT_RANGE)

    for _ in range(count):
        desired_size = random.choices([1, 2, 3], weights=EXTRA_DEFECT_SIZE_WEIGHTS, k=1)[0]
        placed = False
        for _ in range(DEFECT_PLACE_TRIES):
            candidates = [pt for pt in interior_list if pt not in blocked and pt not in occupied]
            if not candidates:
                break
            seed = random.choice(candidates)
            defect = _grow_defect(seed, desired_size, interior, blocked, occupied)
            if len(defect) != desired_size:
                continue
            if not _valid_defect(grid, defect, occupied):
                continue
            for r, c in defect:
                grid[r, c] = BLACK
            occupied.update(defect)
            blocked |= _exclusion_zone(defect, DEFECT_CLEARANCE)
            groups.append({"pixels": defect, "size": desired_size, "kind": "small"})
            placed = True
            break
        if not placed:
            continue

    return groups


def _place_leaves(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> list[dict]:
    leaf_groups: list[dict] = []
    desired = random.randint(*LEAF_COUNT_RANGE)

    for _ in range(desired):
        large_black = [set((int(r), int(c)) for r, c in comp) for comp in components_mask(grid == BLACK, connectivity=4) if len(comp) > 3]
        random.shuffle(large_black)
        placed = False
        for comp in large_black:
            candidates: list[tuple[int, int]] = []
            for r, c in comp:
                for rr, cc in _neighbors4(r, c):
                    if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                        continue
                    if grid[rr, cc] != BLUE or (rr, cc) in occupied or (rr, cc) in blocked:
                        continue
                    black_n = sum(
                        1
                        for nr, nc in _neighbors4(rr, cc)
                        if 0 <= nr < grid.shape[0]
                        and 0 <= nc < grid.shape[1]
                        and grid[nr, nc] == BLACK
                    )
                    if black_n == 1:
                        candidates.append((rr, cc))
            random.shuffle(candidates)
            for lr, lc in candidates:
                if _nearby(lr, lc, DEFECT_CLEARANCE, occupied):
                    continue
                grid[lr, lc] = BLACK
                occupied.add((lr, lc))
                blocked |= _exclusion_zone({(lr, lc)}, DEFECT_CLEARANCE)
                leaf_groups.append({"pixels": {(lr, lc)}, "size": 1, "kind": "leaf"})
                placed = True
                break
            if placed:
                break

    return leaf_groups


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black pixels that the absorption step will flip back to black after turning them blue."""
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
    clean = _build_partition(H, W)
    if not _region_sizes_ok(clean):
        return None

    planned_groups = _place_priority_chain(clean)
    if planned_groups is None:
        return None

    occupied = set().union(*(item["pixels"] for item in planned_groups))
    blocked = _exclusion_zone(occupied, DEFECT_CLEARANCE)

    extra_groups = _place_extra_defects(clean, blocked, occupied)
    leaf_groups = _place_leaves(clean, blocked, occupied)

    expected = main(clean)
    metrics = behavior_metrics(clean)
    if any(metrics["size_hist"][size] == 0 for size in (1, 2, 3)):
        return None
    if any(metrics["halo_color_hist"][color] == 0 for color in (ORANGE, YELLOW, MAGENTA)):
        return None
    if metrics["mixed_overlap_cells"] == 0:
        return None
    if metrics["priority_difference_cells"] == 0:
        return None
    if metrics["color_difference_cells"] == 0:
        return None

    inp = np.array(clean, dtype=int)
    spur_exclude = blocked | occupied
    candidates = _find_spur_candidates(inp, spur_exclude)
    n_spurs = min(random.randint(*SPUR_COUNT_RANGE), len(candidates))
    if n_spurs < MIN_SPURS:
        return None
    for r, c in candidates[:n_spurs]:
        inp[r, c] = BLUE

    if not np.array_equal(main(inp), expected):
        return None

    # Guard against accidental degradation after spur injection too.
    inp_metrics = behavior_metrics(inp)
    if inp_metrics["priority_difference_cells"] == 0 or inp_metrics["mixed_overlap_cells"] == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
