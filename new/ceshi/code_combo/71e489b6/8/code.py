# Combo variant of 71e489b6 (index 8)
# combo_spec: D2/3+D12/1
# Name: 仅叶子检测+缺陷中心红色标记

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
#    Original rule marks two kinds of black anomalies after absorbing blue spurs:
#    (1) every small black component of size <= 3, and (2) every leaf pixel inside
#    a large black component. The combo removes branch (1), keeps only branch (2),
#    and changes the marked leaf pixel itself from black to red while its 3x3 halo
#    stays orange. Example: a 1-cell black hole inside solid blue used to become
#    orange ring + black center; now it stays plain black. A 1-pixel black tip
#    sticking out of a large black region still gets a 3x3 orange halo, but the
#    tip itself becomes red.
# b) Complexity calibration:
#    To make the difference visible, generated inputs deliberately contain both
#    ignored small black defects and at least one true leaf. This avoids a trivial
#    "center recolor only" dataset and shows the logical inversion directly.
# c) Hidden assumptions audit:
#    The original STEP 1 assumed every target pixel should remain black. That is
#    invalid now, so target pixels must be painted red after the halo pass. The
#    original generator also assumed defect_pixels == marked pixels; that no longer
#    holds, so expected outputs must be built from detected leaf targets only.
# d) Placement legality:
#    Defects are kept strictly inside blue interiors, leaves are added only on blue
#    cells adjacent to large black components, and spacing rules prevent overlap,
#    halo collisions, or accidental merging between defect classes.
# e) Visual observability:
#    Each sample is forced to include repeated evidence for both sides of the rule:
#    multiple small black holes that stay unmarked and one or more red-centered
#    leaf endpoints that do get orange halos.
# f) Visual cleanliness:
#    Leaves are kept apart from each other and from small defects, so red centers
#    and orange rings remain readable instead of collapsing into clutter.
# g) Inferability:
#    Only two output marker colors are introduced for a single structural cause:
#    red always means the endpoint itself, orange means its Chebyshev-1 halo.

# --------------- solver ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
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
DEFECT_COUNT_RANGE = (2, 7)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 60
LEAF_COUNT_RANGE = (1, 4)
LEAF_CLEARANCE = 2
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


def _exclusion_zone(pixels: set[tuple[int, int]] | list[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Mirror the original spur-absorption pass exactly."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != 1:
                continue
            black_n = 0
            oob = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
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


def _find_leaf_targets(grid: np.ndarray) -> set[tuple[int, int]]:
    """Return black pixels in large components with <= 1 black 4-neighbor."""
    H, W = grid.shape
    targets: set[tuple[int, int]] = set()
    for coords in components_mask(grid == 0, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            continue
        for r, c in pts:
            black_n = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == 0:
                    black_n += 1
            if black_n <= 1:
                targets.add((r, c))
    return targets


def _paint_leaf_halos(grid: np.ndarray, targets: set[tuple[int, int]]) -> np.ndarray:
    """Paint orange halos and red centers for the selected leaf targets."""
    H, W = grid.shape
    out = np.array(grid, dtype=int)
    for r, c in targets:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in targets:
                    out[rr, cc] = 7
    for r, c in targets:
        out[r, c] = 2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark only dangling endpoints of large black regions, using red centers and orange halos.

    Input structure:
    - Blue (1) and black (0) regions partition the grid; blue interiors may contain small black holes, and black regions may contain thin blue spur pixels that should disappear before detection.

    Rule:
    - First absorb blue spur pixels into black when they are surrounded by black on at least three effective 4-neighbor sides, using the same corner treatment as the source task.
    - Then inspect only black components larger than 3 cells, find their 4-neighbor leaf pixels (black degree <= 1), color each such leaf red (2), and color its Chebyshev-1 neighborhood orange (7).
    - Small black components inside blue are ignored entirely and remain black.

    Key insight:
    - The transform no longer treats every tiny black hole as a defect; the marked objects are specifically graph endpoints of large black structures, with a two-layer code: red for the endpoint itself and orange for its local halo.
    """
    base = _absorb_blue_spurs(grid)
    targets = _find_leaf_targets(base)
    return _paint_leaf_halos(base, targets)


# --------------- generator ---------------

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
    blue = grid == 1
    eroded = blue.copy()
    eroded[:-1, :] &= blue[1:, :]
    eroded[1:, :] &= blue[:-1, :]
    eroded[:, :-1] &= blue[:, 1:]
    eroded[:, 1:] &= blue[:, :-1]
    rs, cs = np.where(eroded)
    coords = list(zip(rs.tolist(), cs.tolist()))
    random.shuffle(coords)
    return coords


def _place_defects(grid: np.ndarray, blocked: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], int]:
    """Place small black holes inside blue that should remain unmarked."""
    H, W = grid.shape
    interior_blue = _find_interior_blue(grid)
    interior_set = set(interior_blue)
    defect_pixels: set[tuple[int, int]] = set()
    defects_placed = 0
    n_defects = random.randint(*DEFECT_COUNT_RANGE)

    for _ in range(n_defects):
        if not interior_blue:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
        for _try in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = interior_blue[random.randint(0, len(interior_blue) - 1)]
            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels | blocked):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr, nc = seed_r + dr, seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    if _nearby(nr, nc, DEFECT_CLEARANCE, blocked):
                        continue
                    ok = True
                    for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rr, cc = nr + dr2, nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < H and 0 <= cc < W:
                            if grid[rr, cc] != 1 or (rr, cc) in defect_pixels:
                                ok = False
                                break
                    if ok:
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W:
                        if grid[rr, cc] != 1 or (rr, cc) in defect_pixels:
                            valid = False
                            break
                if not valid:
                    break

            if valid:
                defect_pixels.update(defect)
                defects_placed += 1
                remove = _exclusion_zone(defect, DEFECT_CLEARANCE)
                interior_blue[:] = [p for p in interior_blue if p not in remove]
                interior_set -= remove
                break

    return defect_pixels, defects_placed


def _leaf_candidate_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    """Blue cells adjacent to large black components that can become leaves."""
    H, W = grid.shape
    candidates: set[tuple[int, int]] = set()
    for coords in components_mask(grid == 0, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            continue
        comp_set = set(pts)
        for r, c in pts:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if grid[rr, cc] != 1:
                    continue
                black_n = 0
                for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr2, cc + dc2
                    if 0 <= nr < H and 0 <= nc < W and (nr, nc) in comp_set:
                        black_n += 1
                if black_n == 1:
                    candidates.add((rr, cc))
    out = list(candidates)
    random.shuffle(out)
    return out


def _place_leaves(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], int]:
    """Add isolated 1-cell leaf protrusions that remain true leaves."""
    leaf_pixels: set[tuple[int, int]] = set()
    n_goal = random.randint(*LEAF_COUNT_RANGE)

    for _ in range(n_goal):
        candidates = _leaf_candidate_cells(grid)
        placed = False
        for r, c in candidates:
            if (r, c) in blocked or (r, c) in leaf_pixels:
                continue
            if _nearby(r, c, LEAF_CLEARANCE, blocked | leaf_pixels):
                continue
            grid[r, c] = 0
            new_targets = _find_leaf_targets(grid)
            if (r, c) in new_targets and not _nearby(r, c, 1, leaf_pixels):
                leaf_pixels.add((r, c))
                placed = True
                break
            grid[r, c] = 1
        if not placed:
            break

    return leaf_pixels, len(leaf_pixels)


def _build_output(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    """Return expected output and the leaf targets that drive it."""
    targets = _find_leaf_targets(grid)
    return _paint_leaf_halos(grid, targets), targets


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
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

    for comp in components_mask(grid == 1, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in components_mask(grid == 0, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return None

    leaf_pixels, n_leaves = _place_leaves(grid, blocked=set())
    if n_leaves < MIN_LEAVES:
        return None

    defect_pixels, defects_placed = _place_defects(grid, blocked=_exclusion_zone(leaf_pixels, 1))
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = 0

    output, targets = _build_output(grid)
    if len(targets) < MIN_LEAVES:
        return None
    if not leaf_pixels.issubset(targets):
        return None

    inp = grid.copy()
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    exclude = defect_pixels | targets | _exclusion_zone(defect_pixels | targets, 1)
    candidates = _find_spur_candidates(inp, exclude)
    if min(n_spurs, len(candidates)) < MIN_SPURS:
        return None
    for r, c in candidates[:n_spurs]:
        inp[r, c] = 1

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None

    absorbed = _absorb_blue_spurs(inp)
    small_comps = [coords for coords in components_mask(absorbed == 0, connectivity=4) if len(coords) <= 3]
    if len(small_comps) < MIN_DEFECTS:
        return None
    if np.count_nonzero(output == 2) < MIN_LEAVES:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
