# Combo variant of 71e489b6 (index 20)
# combo_spec: D2/3+D13/0
# Name: 仅叶子检测+8连通

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask

# Design analysis:
# a) Behavioral trace:
#    Original task: absorb thin blue spur pixels inside black, then mark two black
#    anomaly types: every small black component of size <= 3, plus every leaf
#    pixel of larger black components, all under 4-connectivity. This combo keeps
#    the blue-spur absorption pass, but flips detection to "leafs only" and
#    changes both component grouping and leaf degree counting to 8-connectivity.
#    A concrete example is a black rectangle inside blue with an extra black pixel
#    placed diagonally outside one corner. Under the source rule that diagonal
#    pixel is a separate 1-cell defect, so it is marked because it is small. Under
#    the fused rule it joins the rectangle's 8-connected component, becomes a
#    degree-1 endpoint, and is marked specifically as a leaf. Meanwhile unrelated
#    isolated 1-cell black dots elsewhere in blue are no longer marked at all.
# b) Complexity calibration:
#    If the generator only produced orthogonal leaf stubs, the 8-connectivity
#    change would be hard to notice because those pixels would also look like leafs
#    under 4-connectivity. To make the combo visible, accepted samples must
#    include diagonal-only leaf attachments that are targets only in the 8-connected
#    interpretation, plus several isolated tiny black defects that remain unmarked.
# c) Hidden assumptions audit:
#    The original STEP 1/2 assumed "size <= 3 implies target"; that premise is
#    invalid and must be removed entirely. The original leaf test counted black
#    4-neighbors in the full grid; here we must instead count same-component
#    8-neighbors, otherwise diagonal endpoints would be missed. The large-component
#    threshold (> 3) still matters, but it now applies after 8-connected merging,
#    so diagonal attachments can convert what looked like a tiny defect into part
#    of a large structure. STEP 0 still uses 4-neighbor black-side counting because
#    the combo specification only changes connectivity for black components and
#    leaf detection, not for blue-spur absorption.
# d) Placement legality:
#    Large black shapes are carved fully inside a blue field so they have clear
#    convex corners. Diagonal leaf candidates are chosen only on in-bounds blue
#    cells whose two orthogonal bridge cells are blue, guaranteeing a diagonal-only
#    8-connection to one large black component. Ignored tiny defects are placed on
#    blue cells whose full 8-neighborhood is blue, so they stay isolated and do
#    not merge with black regions or with each other.
# e) Visual observability:
#    Every accepted board contains both visible ignored tiny defects and at least
#    two diagonal-only leafs that appear only under 8-connectivity. Across a few
#    examples, a human can see that isolated black dots in blue are not highlighted
#    while black pixels hanging from large regions by a corner are highlighted.
# f) Visual cleanliness:
#    The palette remains the source palette {0, 1, 7}. Targets are spaced apart by
#    Chebyshev clearance so orange halos do not dissolve into one giant blob, and
#    the large black region count is kept low to preserve readable structure.
# g) Inferability:
#    The rule change is structural rather than color-based. The repeated cue is
#    stable: diagonal corner-attached black pixels near a large black mass get an
#    orange halo, but isolated tiny black dots elsewhere do not. This avoids hidden
#    numeric conventions and keeps the transformation recoverable from geometry.

GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
PARTITION_STYLES = ["rect_inset", "l_shape"]
MIN_BLUE_REGION = 90
MIN_BLACK_REGION = 24
IGNORED_DEFECT_COUNT_RANGE = (2, 6)
LEAF_COUNT_RANGE = (2, 4)
SPUR_COUNT_RANGE = (2, 6)
MIN_IGNORED_DEFECTS = 2
MIN_DIAGONAL_LEAVES = 2
TARGET_CLEARANCE = 2
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Mirror the unchanged source-task spur absorption pass."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != 1:
                continue
            black_n = 0
            oob = 0
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == 0:
                    black_n += 1
            if oob >= 2:
                black_n += int(oob)
            if black_n >= 3:
                base[r, c] = 0
    return base


def _collect_leaf_targets(grid: np.ndarray, connectivity: int) -> set[tuple[int, int]]:
    """Return leaf pixels of large black components under the chosen connectivity."""
    H, W = grid.shape
    neighbor_fn = _neighbors8 if int(connectivity) == 8 else _neighbors4
    preserve: set[tuple[int, int]] = set()

    for coords in components_mask(grid == 0, connectivity=connectivity):
        pts = {(int(r), int(c)) for r, c in coords}
        if len(pts) <= 3:
            continue
        for r, c in pts:
            same_n = 0
            for rr, cc in neighbor_fn(r, c):
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) in pts:
                    same_n += 1
            if same_n <= 1:
                preserve.add((r, c))

    return preserve


def _paint_halo(grid: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    """Paint orange on each target's Chebyshev-1 halo, keeping targets black."""
    out = np.array(grid, dtype=int)
    H, W = out.shape
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    out[rr, cc] = 7
    for r, c in preserve:
        out[r, c] = 0
    return out


def _count_small_components(grid: np.ndarray, connectivity: int = 8) -> int:
    return sum(1 for comp in components_mask(grid == 0, connectivity=connectivity) if len(comp) <= 3)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark only the endpoint pixels of large black regions after interpreting black connectivity diagonally.

    Input structure:
    - A 16-19 by 16-19 board contains one or two large black (0) regions inside a broad blue (1) field.
    - Some black pixels appear as tiny isolated dots inside blue, and some black pixels touch a large black region only diagonally at its corners; black regions may also contain thin blue spur pixels.

    Rule:
    - First absorb any blue pixel that has at least three effective black 4-neighbor sides, using the same corner treatment as the source task.
    - Then form black connected components with 8-connectivity and ignore every component of size <= 3.
    - Inside each larger 8-connected black component, find pixels with at most one black 8-neighbor and paint orange (7) on their Chebyshev-1 halo; the leaf pixels themselves stay black.

    Key insight:
    - The highlighted cells are not "all small black defects" anymore; they are only the dangling endpoints of large 8-connected black structures, including diagonal corner attachments that would not qualify under the original 4-connected reading.
    """
    base = _absorb_blue_spurs(grid)
    preserve = _collect_leaf_targets(base, connectivity=8)
    return _paint_halo(base, preserve)


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create a large blue field containing one large black shape with convex corners."""
    grid = np.ones((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "rect_inset":
        r0 = random.randint(3, max(3, H // 3))
        r1 = random.randint(max(r0 + 4, 2 * H // 3), H - 4)
        c0 = random.randint(3, max(3, W // 3))
        c1 = random.randint(max(c0 + 4, 2 * W // 3), W - 4)
        grid[r0:r1 + 1, c0:c1 + 1] = 0
        return grid

    r0 = random.randint(3, max(3, H // 3))
    c0 = random.randint(3, max(3, W // 3))
    h1 = random.randint(max(5, H // 2), H - r0 - 3)
    w1 = random.randint(3, max(3, W // 4))
    h2 = random.randint(3, max(3, H // 4))
    w2 = random.randint(max(5, W // 2), W - c0 - 3)
    grid[r0:r0 + h1, c0:c0 + w1] = 0
    grid[r0:r0 + h2, c0:c0 + w2] = 0

    if random.random() < 0.5:
        grid = np.fliplr(grid)
    if random.random() < 0.5:
        grid = np.flipud(grid)
    return grid.copy()


def _find_isolated_blue_sites(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return blue cells whose full 8-neighborhood is blue."""
    H, W = grid.shape
    coords: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if grid[r, c] != 1:
                continue
            ok = True
            for rr, cc in _neighbors8(r, c):
                if grid[rr, cc] != 1:
                    ok = False
                    break
            if ok:
                coords.append((r, c))
    random.shuffle(coords)
    return coords


def _place_ignored_defects(grid: np.ndarray, blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Place isolated tiny black dots that the combo solver must ignore."""
    defect_pixels: set[tuple[int, int]] = set()
    target = random.randint(*IGNORED_DEFECT_COUNT_RANGE)
    candidates = _find_isolated_blue_sites(grid)
    for r, c in candidates:
        if len(defect_pixels) >= target:
            break
        if _nearby(r, c, TARGET_CLEARANCE, blocked | defect_pixels):
            continue
        defect_pixels.add((r, c))
        grid[r, c] = 0
    return defect_pixels


def _find_diagonal_leaf_candidates(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Find blue cells that would become diagonal-only leaf endpoints of large black regions."""
    H, W = grid.shape
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []

    for coords in components_mask(grid == 0, connectivity=8):
        pts = {(int(r), int(c)) for r, c in coords}
        if len(pts) <= 3:
            continue
        for r, c in pts:
            for dr in (-1, 1):
                for dc in (-1, 1):
                    rr = r + dr
                    cc = c + dc
                    if not (1 <= rr < H - 1 and 1 <= cc < W - 1):
                        continue
                    if grid[rr, cc] != 1 or (rr, cc) in seen:
                        continue
                    if _nearby(rr, cc, TARGET_CLEARANCE, blocked):
                        continue
                    if grid[r, cc] != 1 or grid[rr, c] != 1:
                        continue
                    black_n = 0
                    for ar, ac in _neighbors8(rr, cc):
                        if 0 <= ar < H and 0 <= ac < W and grid[ar, ac] == 0:
                            black_n += 1
                    if black_n != 1:
                        continue
                    seen.add((rr, cc))
                    candidates.append((rr, cc))

    random.shuffle(candidates)
    return candidates


def _place_diagonal_leaves(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Add black pixels that become leaf targets only under 8-connectivity."""
    leaf_pixels: set[tuple[int, int]] = set()
    wanted = random.randint(*LEAF_COUNT_RANGE)

    for _ in range(wanted):
        candidates = _find_diagonal_leaf_candidates(grid, blocked | leaf_pixels)
        if not candidates:
            break
        pick = candidates[0]
        leaf_pixels.add(pick)
        grid[pick] = 0

    return leaf_pixels


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black cells that become removable blue spur pixels in the input."""
    H, W = grid.shape
    black = grid == 0

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
    coords = [
        (int(r), int(c))
        for r, c in zip(rs, cs)
        if (int(r), int(c)) not in exclude
    ]
    random.shuffle(coords)
    return coords


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    clean = _build_partition(H, W)
    blue_comps = list(components_mask(clean == 1, connectivity=4))
    black_comps = list(components_mask(clean == 0, connectivity=8))
    if not blue_comps or not black_comps:
        return None
    if max(len(comp) for comp in blue_comps) < MIN_BLUE_REGION:
        return None
    if max(len(comp) for comp in black_comps) < MIN_BLACK_REGION:
        return None

    leaf_pixels = _place_diagonal_leaves(clean, set())
    if len(leaf_pixels) < MIN_DIAGONAL_LEAVES:
        return None

    ignored_defects = _place_ignored_defects(clean, leaf_pixels)
    if len(ignored_defects) < MIN_IGNORED_DEFECTS:
        return None

    preserve8 = _collect_leaf_targets(clean, connectivity=8)
    preserve4 = _collect_leaf_targets(clean, connectivity=4)
    diagonal_only = preserve8 - preserve4
    if len(diagonal_only) < MIN_DIAGONAL_LEAVES:
        return None
    if not leaf_pixels.issubset(diagonal_only):
        return None
    if _count_small_components(clean, connectivity=8) < MIN_IGNORED_DEFECTS:
        return None

    expected = _paint_halo(clean, preserve8)

    inp = clean.copy()
    spur_count = random.randint(*SPUR_COUNT_RANGE)
    candidates = _find_spur_candidates(inp, leaf_pixels | ignored_defects)
    if len(candidates) < 2:
        return None
    for r, c in candidates[:spur_count]:
        inp[r, c] = 1

    try:
        if np.array_equal(main(inp), expected):
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
