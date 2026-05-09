# Combo variant of 71e489b6 (index 7)
# combo_spec: D3/0+D9/0
# Name: 毛刺品红标记+灰色噪点干扰

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import components_mask
except ImportError:
    for parent in Path(__file__).resolve().parents:
        solver_dir = parent / "solvers"
        if (solver_dir / "arc_utils.py").exists():
            sys.path.insert(0, str(solver_dir))
            break
    from arc_utils import components_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - The original pipeline had one nuisance channel: blue spur pixels inside
#      black regions were absorbed into black before any defect haloing.
#    - This combo splits the nuisances into two visible roles while keeping the
#      original orange-defect rule intact:
#      1) black defects inside blue still get orange halos,
#      2) blue spur pixels inside black are kept and recolored magenta,
#      3) grey pixels inside black are removed to black and never haloed.
#    - Small trace:
#        input local patch
#          0 0 0        1 1 1
#          0 1 0   and  1 0 1
#          0 5 0        1 1 1
#      becomes
#          0 6 0        7 7 7
#          0 0 0   and  7 0 7
#          0 0 0        7 7 7
#      The left patch shows D3/0: the blue spur is not absorbed, it is marked
#      magenta. The grey dot underneath simply disappears to black. The right
#      patch shows the unchanged defect-to-orange-halo behavior.
#
# b) Complexity calibration:
#    - Keep only two new colors in total: grey (input-only nuisance) and
#      magenta (output-only spur mark). Orange remains reserved for black
#      defects. This is enough to make the fusion visible without creating a
#      palette-guessing task.
#    - Generator always places multiple spurs and multiple grey noise dots, so
#      examples cannot silently collapse back to the original "only orange" task.
#    - Spurs are kept as singleton dots for visual cleanliness. The combo is
#      already multi-branch, so forcing 1-2 cell spur morphology would add
#      little insight while increasing collision risk with halos and leaves.
#
# c) Implicit-assumption audit against the original solver:
#    - Original STEP 0 assumed every blue pixel with >=3 effective black
#      neighbors should be rewritten to black. That assumption is now false:
#      such pixels must be remembered as spur marks and preserved structurally.
#    - Original code also assumed the input used only colors 0 and 1. That is
#      now false because grey 5 is legal input noise and must be normalized to 0
#      before component analysis.
#    - The black-component pass still assumes that all black pixels relevant to
#      haloing are truly black in the working grid. The new solver restores this
#      by deleting grey noise before computing components.
#    - A new assumption appears: every grey pixel is noise, not a meaningful
#      foreground object. The generator guarantees this by placing grey only on
#      black-region interior cells, away from defects and blue regions.
#
# d) Placement legality:
#    - Defects are still placed strictly inside blue interiors, with clearance so
#      they stay separate 4-components and never merge into large black regions.
#    - Leaf pixels are still carved from large black components into neighboring
#      blue cells.
#    - Spur candidates are sampled only from black cells that satisfy the solver
#      predicate and are kept separated from each other and from orange-halo
#      zones, so they do not accidentally alter defect structure.
#    - Grey noise is sampled from black interior cells with four black
#      neighbors, again separated from spurs and defect halos, so deleting it
#      cannot create or remove a halo target.
#
# e) Visual observability:
#    - Across 2-4 examples, a human sees three stable local grammars:
#      black-in-blue => orange halo, blue-in-black => magenta center, grey-in-
#      black => disappearance. These categories are spatially anchored by the
#      surrounding region colors rather than by hidden metadata.
#    - The same blue spur predicate repeats several times per grid, so magenta is
#      inferable as a structure-based transformation, not as an arbitrary recolor.
#
# f) Visual cleanliness:
#    - The generator leaves a one-cell safety margin around orange halos when
#      placing spurs and grey noise. This keeps magenta dots from sitting inside
#      orange rings and keeps grey dots from being mistaken for defect centers.
#    - Grey noise is input-only and disappears completely, so the output remains
#      dominated by orange rings plus a few isolated magenta dots.
#
# g) Change inferability:
#    - The fused behavior does not rely on color ids, scan order, or component
#      ordering. Only visible local context matters: "small black in blue",
#      "blue mostly enclosed by black", and "grey inside black".
#    - Samples are rejected unless both magenta and grey-noise branches are
#      active, preventing degeneration to the base task or to a single-branch
#      variant.


BLUE = 1
BLACK = 0
GREY = 5
MAGENTA = 6
ORANGE = 7


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
DEFECT_COUNT_RANGE = (2, 9)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 50
LEAF_COUNT_RANGE = (0, 4)
SPUR_COUNT_RANGE = (2, 6)
NOISE_COUNT_RANGE = (2, 5)
MIN_DEFECTS = 2
MIN_SPURS = 2
MIN_NOISE = 2
SAFETY_MARGIN = 2
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


def _effective_black_neighbors(grid: np.ndarray, r: int, c: int) -> int:
    H, W = grid.shape
    black_neighbors = 0
    oob = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if not (0 <= rr < H and 0 <= cc < W):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_neighbors += 1
    if oob >= 2:
        black_neighbors += oob
    return black_neighbors


def _find_spur_marks(grid: np.ndarray) -> list[tuple[int, int]]:
    H, W = grid.shape
    marks: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != BLUE:
                continue
            if _effective_black_neighbors(grid, r, c) >= 3:
                marks.append((r, c))
    return marks


def _collect_preserve_pixels(grid: np.ndarray) -> set[tuple[int, int]]:
    H, W = grid.shape
    preserve: set[tuple[int, int]] = set()
    for comp in components_mask(grid == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in comp]
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        comp_set = set(pts)
        for r, c in pts:
            degree = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) in comp_set:
                    degree += 1
            if degree <= 1:
                preserve.add((r, c))
    return preserve


def _paint_orange_halo(base: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(base, dtype=int)
    H, W = out.shape
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black defects with orange halos, blue spurs with magenta, and erase grey noise.

    Input structure:
    - Large blue (1) and black (0) regions partition the canvas.
    - Blue regions may contain small black defects or leaf-like intrusions from larger black areas.
    - Black regions may also contain isolated blue spur pixels and grey (5) nuisance pixels.

    Rule:
    - First treat every grey pixel as removable black-region noise and normalize it to black.
    - Then find blue pixels with at least three effective black 4-neighbors; keep them as spur locations and mark them magenta (6) in the output instead of absorbing them.
    - On the cleaned black mask, detect all black defect targets exactly as in the base task: every small black component (size <= 3) and every leaf pixel of a larger black component stays black and gets an orange (7) Chebyshev-1 halo.

    Key insight: The output now distinguishes three visually different anomaly roles by local context: black-in-blue gets an orange ring, blue-in-black becomes a magenta dot, and grey-in-black is simply removed.
    """
    g = np.array(grid, dtype=int)
    work = g.copy()
    work[work == GREY] = BLACK

    spur_marks = _find_spur_marks(work)
    preserve = _collect_preserve_pixels(work)
    out = _paint_orange_halo(work, preserve)

    for r, c in spur_marks:
        out[r, c] = MAGENTA

    return out


def _build_partition(H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cut_count = min(n_cuts, H - 2 * CUT_MARGIN)
        cuts = sorted(random.sample(range(CUT_MARGIN, H - CUT_MARGIN), cut_count))
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


def _place_defects(grid: np.ndarray, interior_blue: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], int]:
    H, W = grid.shape
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    defect_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)
    defects_placed = 0

    for _ in range(n_defects):
        if not interior_blue:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
        for _try in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = interior_blue[random.randint(0, len(interior_blue) - 1)]
            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    ok = True
                    for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rr = nr + dr2
                        cc = nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < H and 0 <= cc < W:
                            if int(grid[rr, cc]) != BLUE or (rr, cc) in defect_pixels:
                                ok = False
                                break
                    if ok and not _nearby(nr, nc, DEFECT_CLEARANCE, defect_pixels):
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W:
                        if int(grid[rr, cc]) != BLUE or (rr, cc) in defect_pixels:
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


def _place_leaves(grid: np.ndarray, defect_pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    H, W = grid.shape
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    black_comps = components_mask(grid == BLACK, connectivity=4)
    large_black = [comp for comp in black_comps if len(comp) > 3]

    for _ in range(n_leaves):
        if not large_black:
            break
        comp_set = {(int(r), int(c)) for r, c in random.choice(large_black)}
        candidates = []
        for r, c in comp_set:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < H and 0 <= cc < W) or int(grid[rr, cc]) != BLUE:
                    continue
                if (rr, cc) in defect_pixels or (rr, cc) in leaf_pixels:
                    continue
                black_n = 0
                for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = rr + dr2
                    nc = cc + dc2
                    if 0 <= nr < H and 0 <= nc < W:
                        if int(grid[nr, nc]) == BLACK or (nr, nc) in leaf_pixels:
                            black_n += 1
                if black_n == 1:
                    candidates.append((rr, cc))

        if candidates:
            lr, lc = random.choice(candidates)
            if not _nearby(lr, lc, DEFECT_CLEARANCE, defect_pixels):
                leaf_pixels.add((lr, lc))
                grid[lr, lc] = BLACK

    return leaf_pixels


def _find_spur_candidates(grid: np.ndarray, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != BLACK or (r, c) in blocked:
                continue
            if _effective_black_neighbors(grid, r, c) >= 3:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_spurs(base: np.ndarray, blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    work = np.array(base, dtype=int)
    target = random.randint(*SPUR_COUNT_RANGE)
    spurs: set[tuple[int, int]] = set()
    blocked_zone = set(blocked)

    while len(spurs) < target:
        candidates = _find_spur_candidates(work, blocked_zone)
        if not candidates:
            break
        r, c = candidates[0]
        work[r, c] = BLUE
        spurs.add((r, c))
        blocked_zone |= _exclusion_zone({(r, c)}, SAFETY_MARGIN)

    return spurs


def _find_noise_candidates(grid: np.ndarray, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != BLACK or (r, c) in blocked:
                continue
            if all(int(grid[r + dr, c + dc]) == BLACK for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_grey_noise(base: np.ndarray, blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    target = random.randint(*NOISE_COUNT_RANGE)
    noise: set[tuple[int, int]] = set()
    blocked_zone = set(blocked)
    for r, c in _find_noise_candidates(base, blocked_zone):
        noise.add((r, c))
        blocked_zone |= _exclusion_zone({(r, c)}, SAFETY_MARGIN)
        if len(noise) >= target:
            break
    return noise


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)

    for comp in components_mask(grid == BLUE, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in components_mask(grid == BLACK, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return None

    interior_blue = _find_interior_blue(grid)
    defect_pixels, defects_placed = _place_defects(grid, interior_blue)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, defect_pixels)
    preserve = defect_pixels | leaf_pixels
    blocked = _exclusion_zone(preserve, SAFETY_MARGIN)

    spurs = _place_spurs(grid, blocked)
    if len(spurs) < MIN_SPURS:
        return None

    noise = _place_grey_noise(grid, blocked | _exclusion_zone(spurs, SAFETY_MARGIN))
    if len(noise) < MIN_NOISE:
        return None

    output = _paint_orange_halo(grid, preserve)
    for r, c in spurs:
        output[r, c] = MAGENTA

    inp = grid.copy()
    for r, c in spurs:
        inp[r, c] = BLUE
    for r, c in noise:
        inp[r, c] = GREY

    try:
        solver_out = main(inp)
    except Exception:
        return None

    if not np.array_equal(solver_out, output):
        return None

    if not np.any(inp == GREY):
        return None
    if not np.any(solver_out == MAGENTA):
        return None
    if np.any(solver_out == GREY):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
