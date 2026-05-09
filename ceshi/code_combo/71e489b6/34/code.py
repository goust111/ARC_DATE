# Combo variant of 71e489b6 (index 34)
# combo_spec: D4/1+D8/1
# Name: 距离检测+大网格稀疏

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import components_mask


"""
Design analysis for combo D4/1 + D8/1
=====================================

a) Concrete behavior and trace:
   - Source behavior:
     * First absorb thin blue spurs inside black regions.
     * Then mark every small black component (size <= 3) and every leaf pixel of a larger
       black component.
     * Finally paint an orange Chebyshev-1 halo around those marked black cells.
   - Fused behavior:
     * Spurs are still absorbed first; this part remains useful because the input still
       contains distracting blue specks inside the large black regions.
     * The marked set is no longer "small components + leaves". A black cell is now marked
       only if it belongs to a black island embedded in blue and its Chebyshev distance to
       that island's blue boundary is at least 2.
     * On a pixel grid, distance >= 2 means the full 3x3 neighborhood around that black cell
       is still black. So a 3x3 defect contributes only its center; a 4x4 block contributes
       a 2x2 core; a 1x1 or 2x2 hole contributes nothing.
   - Concrete example:
     * Original task on a lone 1x1 black hole inside blue: that single black cell is marked.
     * This combo on a 3x3 black hole inside blue: only the center is marked, and the outer
       ring of that 3x3 block gets overwritten by the orange halo.
     * On a 5x4 defect block, the retained black target region becomes a 3x2 inner core.
   - Difference made visible by D8/1:
     * The grid grows to 22-26 in each dimension and the number of planted defects drops to
       2-5, so samples look sparse and require distance reasoning rather than local spotting
       of tiny black dots.

b) Complexity calibration:
   - If internal defects stayed size 1-3, the distance rule would be invisible because no
     pixel could possibly have Chebyshev distance >= 2 from blue.
   - Therefore the generator uses only compact solid black rectangles of size at least 3x3.
     The allowed palette of sizes is small, so the rule is still inferable from 2-4 examples.
   - Defects are kept sparse with several cells of clearance, making the large-grid change
     visually obvious instead of collapsing into cluttered overlapping halos.

c) Implicit-assumption audit against the source main():
   - Source Step 0 assumed blue spurs can be recognized by 4-neighbor black counts. This
     remains valid and is preserved unchanged.
   - Source Step 1 assumed every tiny black component is a defect. Broken by D4/1: a 1x1,
     1x2, 2x2, or 3-cell component is now intentionally ignored because none of its pixels
     are distance-2 cores.
   - Source Step 1 also assumed large black regions contribute leaf pixels. Broken by D4/1:
     leaves are boundary phenomena, but the new rule only marks deep interior black cells.
   - A new assumption is introduced: legitimate target pixels live inside black islands whose
     4-neighbor exterior is all blue and whose component does not touch the frame edge.
     Otherwise the main black regions would also contain distance-2 interiors.
   - The generator therefore guarantees that all macroscopic black regions are edge-anchored,
     while the actual defects are isolated black islands planted inside blue.

d) Placement legality:
   - Defects are placed only where a 2-cell blue buffer exists around the whole defect box.
     This guarantees in-bounds placement, prevents merging with large black regions, and keeps
     the intended distance-core unambiguous.
   - Defects are separated by a larger exclusion zone so their halos and black cores do not
     collapse into one another.
   - Spurs are inserted only at black cells that the preserved absorption rule provably turns
     back to black, and never on top of defect cells or near them.

e) Visual observability:
   - In input/output pairs, a human can see that large black blobs inside blue are not entirely
     preserved: only their inner core stays black while the surrounding black rim becomes orange.
   - At the same time, thin black edge protrusions are no longer highlighted, distinguishing the
     new rule from the source task's leaf detection.
   - The larger canvas and reduced count create a visible "search the big board for a few deep
     cores" pattern rather than a dense field of tiny defects.

f) Visual cleanliness:
   - The scene still uses only 0/1 in input and adds 7 in output.
   - Large black regions sit on the border structure, while defect islands sit in blue interiors.
     This separates object roles spatially and keeps the board readable.
   - Halo overlap is allowed by the rule but discouraged by generator spacing, so outputs stay
     tidy instead of becoming one orange mass.

g) Variant metric for non-degeneracy:
   - For generated samples, compare the fused target set against the source-task target set
     ("small component or leaf pixel"). The generator rejects any sample where the two target
     sets are identical.
   - It also rejects samples with zero fused targets. In accepted samples, the new distance-core
     rule must do real work that the original rule would miss.

h) New solver assumptions after the edit:
   - Any non-border black component whose 4-neighbor exterior is not all blue is treated as
     structural background, not as a defect island.
   - Defect components are assumed to be compact enough that "distance >= 2" is captured by
     checking whether the full 3x3 neighborhood stays inside the same black component.
   - The generator explicitly enforces both assumptions.
"""


BLUE = 1
BLACK = 0
ORANGE = 7

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
PARTITION_STYLES = ("stripes", "frame", "l_shape")
MIN_BLUE_REGION = 90
MIN_BLACK_REGION = 30

DEFECT_COUNT_RANGE = (2, 5)
DEFECT_SHAPES = ((3, 3), (3, 4), (4, 3), (4, 4), (3, 5), (5, 3))
DEFECT_SHAPE_WEIGHTS = (6, 4, 4, 3, 2, 2)
DEFECT_BLUE_MARGIN = 2
DEFECT_CLEARANCE = 4
DEFECT_PLACE_TRIES = 200

SPUR_COUNT_RANGE = (2, 6)
MIN_SPURS = 2
SPUR_CLEARANCE = 1

MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _box_zone(r0: int, c0: int, h: int, w: int, margin: int, H: int, W: int) -> set[tuple[int, int]]:
    zone = set()
    rr0 = max(0, int(r0 - margin))
    rr1 = min(H - 1, int(r0 + h - 1 + margin))
    cc0 = max(0, int(c0 - margin))
    cc1 = min(W - 1, int(c0 + w - 1 + margin))
    for r in range(rr0, rr1 + 1):
        for c in range(cc0, cc1 + 1):
            zone.add((int(r), int(c)))
    return zone


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (int(r + dr), int(c + dc)) in ref:
                return True
    return False


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
    base = np.array(grid, dtype=int).copy()
    H, W = map(int, base.shape)
    snap = base.copy()
    for r in range(H):
        for c in range(W):
            if int(snap[r, c]) != BLUE:
                continue
            black_n = 0
            oob = 0
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(snap[rr, cc]) == BLACK:
                    black_n += 1
            if oob >= 2:
                black_n += oob
            if black_n >= 3:
                base[r, c] = BLACK
    return base


def _is_internal_blue_island(base: np.ndarray, comp_set: set[tuple[int, int]]) -> bool:
    H, W = map(int, base.shape)
    if any(r in (0, H - 1) or c in (0, W - 1) for r, c in comp_set):
        return False
    for r, c in comp_set:
        for rr, cc in _neighbors4(r, c):
            if (int(rr), int(cc)) in comp_set:
                continue
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            if int(base[rr, cc]) != BLUE:
                return False
    return True


def _distance_targets(base: np.ndarray) -> set[tuple[int, int]]:
    H, W = map(int, base.shape)
    targets: set[tuple[int, int]] = set()
    for coords in components_mask(base == BLACK, connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        if not _is_internal_blue_island(base, pts):
            continue
        for r, c in pts:
            deep = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if not (0 <= rr < H and 0 <= cc < W and (rr, cc) in pts):
                        deep = False
                        break
                if not deep:
                    break
            if deep:
                targets.add((int(r), int(c)))
    return targets


def _original_targets(base: np.ndarray) -> set[tuple[int, int]]:
    preserve: set[tuple[int, int]] = set()
    H, W = map(int, base.shape)
    for coords in components_mask(base == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        comp_set = set(pts)
        for r, c in pts:
            deg = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and (int(rr), int(cc)) in comp_set:
                    deg += 1
            if deg <= 1:
                preserve.add((int(r), int(c)))
    return preserve


def _paint_halo(base: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(base, dtype=int).copy()
    H, W = map(int, out.shape)
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[int(r), int(c)] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Highlight only the deep black cores of blue-embedded defects, then halo them in orange.

    Input structure:
    - The canvas is a larger 22-26 by 22-26 board partitioned into large blue and black regions.
    - Large black regions stay attached to the border structure, while a few sparse compact black
      islands are planted inside blue areas.
    - Some blue single-pixel spurs also appear inside black regions and should be absorbed first.

    Rule:
    - Convert blue spur pixels back to black when they have at least three effective black
      4-neighbors, counting out-of-bounds only for corners exactly as in the source task.
    - Among black components, keep only those embedded in blue, and within each such island mark
      only the black pixels whose full 3x3 neighborhood remains black; these are exactly the
      pixels at Chebyshev distance >= 2 from the island's blue boundary.
    - Paint orange on the Chebyshev-1 halo around those marked pixels, while the marked pixels
      themselves remain black.

    Key insight:
    - The output is no longer driven by tiny black anomalies or leaf tips. It is driven by depth:
      a black pixel matters only if it sits at least two Chebyshev layers inside a blue-enclosed
      black island.
    """
    base = _absorb_spurs(np.array(grid, dtype=int))
    preserve = _distance_targets(base)
    return _paint_halo(base, preserve)


def _build_stripes(H: int, W: int) -> np.ndarray:
    grid = np.ones((H, W), dtype=int)
    pattern = random.choice(("blue-black-blue", "black-blue-black-blue", "blue-black-blue-black"))

    if pattern == "blue-black-blue":
        black_h = random.randint(2, 4)
        top_blue = random.randint(8, H - black_h - 8)
        bottom_blue = H - top_blue - black_h
        heights = [top_blue, black_h, bottom_blue]
        colors = [BLUE, BLACK, BLUE]
    elif pattern == "black-blue-black-blue":
        black1 = random.randint(2, 4)
        black2 = random.randint(2, 4)
        blue1 = random.randint(7, H - black1 - black2 - 7)
        blue2 = H - black1 - blue1 - black2
        heights = [black1, blue1, black2, blue2]
        colors = [BLACK, BLUE, BLACK, BLUE]
    else:
        black1 = random.randint(2, 4)
        black2 = random.randint(2, 4)
        blue1 = random.randint(7, H - black1 - black2 - 7)
        blue2 = H - blue1 - black1 - black2
        heights = [blue1, black1, blue2, black2]
        colors = [BLUE, BLACK, BLUE, BLACK]

    r = 0
    for h, col in zip(heights, colors):
        grid[r : r + h, :] = int(col)
        r += int(h)
    return grid


def _build_frame(H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    thickness = random.randint(2, 4)
    grid[thickness : H - thickness, thickness : W - thickness] = BLUE
    if random.random() < 0.5:
        bar_t = random.randint(2, 3)
        if random.random() < 0.5:
            bar_w = random.randint(W // 4, W // 2)
            c0 = random.randint(thickness + 2, W - thickness - bar_w - 2)
            grid[thickness : thickness + bar_t, c0 : c0 + bar_w] = BLACK
        else:
            bar_h = random.randint(H // 4, H // 2)
            r0 = random.randint(thickness + 2, H - thickness - bar_h - 2)
            grid[r0 : r0 + bar_h, thickness : thickness + bar_t] = BLACK
    return grid


def _build_l_shape(H: int, W: int) -> np.ndarray:
    grid = np.ones((H, W), dtype=int)
    thickness = random.randint(2, 4)
    h_len = random.randint(W // 2, W - 2)
    v_len = random.randint(H // 2, H - 2)
    corner = random.choice(("tl", "tr", "bl", "br"))

    if corner == "tl":
        grid[:thickness, :h_len] = BLACK
        grid[:v_len, :thickness] = BLACK
    elif corner == "tr":
        grid[:thickness, W - h_len :] = BLACK
        grid[:v_len, W - thickness :] = BLACK
    elif corner == "bl":
        grid[H - thickness :, :h_len] = BLACK
        grid[H - v_len :, :thickness] = BLACK
    else:
        grid[H - thickness :, W - h_len :] = BLACK
        grid[H - v_len :, W - thickness :] = BLACK

    if random.random() < 0.5:
        stripe_t = random.randint(2, 3)
        if random.random() < 0.5:
            r0 = 0 if corner in {"bl", "br"} else H - stripe_t
            c0 = random.randint(W // 5, max(W // 5, W - W // 3))
            c1 = min(W, c0 + random.randint(W // 4, W // 3))
            grid[r0 : r0 + stripe_t, c0:c1] = BLACK
        else:
            c0 = 0 if corner in {"tr", "br"} else W - stripe_t
            r0 = random.randint(H // 5, max(H // 5, H - H // 3))
            r1 = min(H, r0 + random.randint(H // 4, H // 3))
            grid[r0:r1, c0 : c0 + stripe_t] = BLACK
    return grid


def _build_partition(H: int, W: int) -> np.ndarray:
    style = random.choice(PARTITION_STYLES)
    if style == "stripes":
        return _build_stripes(H, W)
    if style == "frame":
        return _build_frame(H, W)
    return _build_l_shape(H, W)


def _placement_is_blue(grid: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int) -> bool:
    H, W = map(int, grid.shape)
    rr0 = r0 - margin
    rr1 = r0 + h + margin - 1
    cc0 = c0 - margin
    cc1 = c0 + w + margin - 1
    if rr0 < 0 or cc0 < 0 or rr1 >= H or cc1 >= W:
        return False
    sub = grid[rr0 : rr1 + 1, cc0 : cc1 + 1]
    return bool(np.all(sub == BLUE))


def _place_defects(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    H, W = map(int, grid.shape)
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    blocked: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []

    for _ in range(n_defects):
        placed = False
        for _try in range(DEFECT_PLACE_TRIES):
            h, w = random.choices(DEFECT_SHAPES, weights=DEFECT_SHAPE_WEIGHTS, k=1)[0]
            r0 = random.randint(DEFECT_BLUE_MARGIN, H - h - DEFECT_BLUE_MARGIN)
            c0 = random.randint(DEFECT_BLUE_MARGIN, W - w - DEFECT_BLUE_MARGIN)
            if not _placement_is_blue(grid, r0, c0, h, w, DEFECT_BLUE_MARGIN):
                continue
            box_zone = _box_zone(r0, c0, h, w, DEFECT_CLEARANCE, H, W)
            if any(cell in blocked for cell in box_zone):
                continue
            defect = {(int(r), int(c)) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
            for r, c in defect:
                grid[r, c] = BLACK
            blocked |= box_zone
            components.append(defect)
            placed = True
            break
        if not placed:
            break
    return components


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    H, W = map(int, grid.shape)
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
    coords = [(int(r), int(c)) for r, c in zip(rs, cs) if (int(r), int(c)) not in exclude]
    random.shuffle(coords)
    return coords


def _sample_spurs(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    chosen: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in candidates:
        if any(_nearby(r, c, SPUR_CLEARANCE, {(rr, cc)}) for rr, cc in chosen):
            continue
        chosen.append((int(r), int(c)))
        seen.add((int(r), int(c)))
        if len(chosen) >= count:
            break
    return chosen


def _region_sizes_ok(grid: np.ndarray) -> bool:
    blue_comps = components_mask(grid == BLUE, connectivity=4)
    black_comps = components_mask(grid == BLACK, connectivity=4)
    if not blue_comps or not black_comps:
        return False
    if max(len(comp) for comp in blue_comps) < MIN_BLUE_REGION:
        return False
    if max(len(comp) for comp in black_comps) < MIN_BLACK_REGION:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    clean = _build_partition(H, W)
    if not _region_sizes_ok(clean):
        return None

    defect_components = _place_defects(clean)
    if len(defect_components) < DEFECT_COUNT_RANGE[0]:
        return None

    fused_targets = _distance_targets(clean)
    if not fused_targets:
        return None
    if fused_targets == _original_targets(clean):
        return None

    output = _paint_halo(clean, fused_targets)

    exclude: set[tuple[int, int]] = set()
    for comp in defect_components:
        for r, c in comp:
            exclude |= _box_zone(int(r), int(c), 1, 1, DEFECT_BLUE_MARGIN, H, W)
    spur_candidates = _find_spur_candidates(clean, exclude)
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    chosen_spurs = _sample_spurs(spur_candidates, n_spurs)
    if len(chosen_spurs) < MIN_SPURS:
        return None

    inp = clean.copy()
    for r, c in chosen_spurs:
        inp[r, c] = BLUE

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
