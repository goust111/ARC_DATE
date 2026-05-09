# Combo variant of 71e489b6 (index 32)
# combo_spec: D3/0+D5/0+D9/1
# Name: 毛刺品红+类型分色+多色干扰

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
#    Original task flow: blue spur pixels inside black regions are absorbed into
#    black, then the cleaned black mask is scanned for two anomaly types:
#    detached tiny black components (size <= 3) and leaf pixels of larger black
#    components. All anomalies then receive the same orange Chebyshev-1 halo.
#    Fused flow: the "cleaned black mask" is still the analytic substrate, but
#    the pixels that would have been absorbed are now any non-blue/non-black
#    single-pixel color specks inside black regions. They are treated as black
#    only for structural analysis, yet rendered back as magenta markers in the
#    output. Small detached black defects still get orange halos, while leaf
#    pixels of large black components get magenta halos. Example: a red pixel
#    inside a black stripe with three black 4-neighbors becomes a single magenta
#    output dot; a one-cell black tip jutting into blue remains black and gets a
#    magenta ring; a 1x1 black hole fully inside blue remains black and gets an
#    orange ring.
# b) Complexity calibration:
#    Magenta now serves two roles, so morphology must disambiguate them. The
#    generator therefore keeps spur markers isolated inside black regions and
#    leaves at least one-cell spacing between any spur and any halo footprint.
#    Each accepted sample must show all three visible behaviors together:
#    orange halos around internal defects, magenta halos around leaf tips, and
#    isolated magenta spur dots sourced from at least two different input colors.
# c) Hidden assumptions audit:
#    The original STEP 0 assumed only blue spur pixels needed absorption; that
#    assumption is invalid here, so spur detection explicitly targets every
#    non-blue/non-black pixel with >= 3 effective black 4-neighbors. The
#    original component scan also assumed the post-STEP0 grid contains only
#    black and blue; here we restore detected spur cells to black in a separate
#    analysis grid before computing components. Finally, the original renderer
#    used a single halo color; the fused renderer must classify targets first so
#    it can paint internal defects orange and leaf tips magenta, then restore
#    spur cells as magenta singletons after halo painting.
# d) Placement legality:
#    Internal defects are placed only on strict blue interiors and kept apart by
#    Chebyshev clearance so they never merge with each other or with black
#    regions. Leaf tips are inserted only on blue cells adjacent to exactly one
#    black neighbor of a large black component, guaranteeing they stay leaves.
#    Spur noise is created only from black cells whose effective black-neighbor
#    count is >= 3, so the solver can soundly reinterpret them as black during
#    analysis. Clearance zones keep all three object types disjoint.
# e) Visual observability:
#    Across a few examples, a human can infer the rule because the three marker
#    motifs repeat with clear structure: orange 3x3 rings always surround tiny
#    detached black defects in blue; magenta 3x3 rings always surround black
#    leaf tips attached to large black masses; isolated input specks of several
#    unrelated colors all collapse to the same magenta single-pixel output mark.
# f) Visual cleanliness:
#    The generator rejects crowded boards where halos of different types would
#    overlap or where spur dots would touch magenta leaf halos. That preserves
#    a readable separation between "ring-shaped magenta around black centers"
#    and "standalone magenta dots in black regions".
# g) Inferability:
#    The color logic stays small and structural: orange means detached internal
#    defect halo, magenta means either leaf halo or spur marker, with shape and
#    context distinguishing those two magenta cases. Input noise colors come
#    from a small palette of non-blue/non-black colors and are repeated so the
#    inferred rule is "arbitrary foreign color specks" rather than a fixed
#    mapping from one source color.

# --------------- solver ---------------

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
DEFECT_PLACE_TRIES = 60
LEAF_COUNT_RANGE = (1, 3)
SPUR_COUNT_RANGE = (3, 6)
TARGET_CLEARANCE = 3
SPUR_TARGET_CLEARANCE = 3
SPUR_SPUR_CLEARANCE = 2
MIN_DEFECT_GROUPS = 2
MIN_LEAVES = 1
MIN_SPURS = 3
MAX_GEN_ATTEMPTS = 400

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
TEAL = 8
MAROON = 9

NOISE_PALETTE = [RED, GREEN, YELLOW, TEAL, MAROON]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(
    pixels: set[tuple[int, int]] | list[tuple[int, int]],
    dist: int,
) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _effective_black_count(grid: np.ndarray, r: int, c: int) -> int:
    """Count black 4-neighbors, treating corners outside the board as black."""
    H, W = grid.shape
    black_n = 0
    oob = 0
    for rr, cc in _neighbors4(r, c):
        if not (0 <= rr < H and 0 <= cc < W):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_n += 1
    if oob >= 2:
        black_n += int(oob)
    return black_n


def _detect_spurs(grid: np.ndarray) -> set[tuple[int, int]]:
    """Find foreign-color spur pixels that sit inside black territory."""
    H, W = grid.shape
    spurs: set[tuple[int, int]] = set()
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color in (BLACK, BLUE):
                continue
            if _effective_black_count(grid, r, c) >= 3:
                spurs.add((r, c))
    return spurs


def _analysis_base(grid: np.ndarray, spurs: set[tuple[int, int]]) -> np.ndarray:
    """Return the grid used for black-structure analysis, with spurs absorbed."""
    base = np.array(grid, dtype=int)
    for r, c in spurs:
        base[r, c] = BLACK
    return base


def _collect_targets(
    base: np.ndarray,
) -> tuple[list[set[tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Split black anomalies into detached tiny components vs large-component leaves."""
    H, W = base.shape
    internal_groups: list[set[tuple[int, int]]] = []
    leaf_pixels: set[tuple[int, int]] = set()
    all_targets: set[tuple[int, int]] = set()

    for coords in components_mask(base == BLACK, connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        if len(pts) <= 3:
            internal_groups.append(pts)
            all_targets.update(pts)
            continue
        for r, c in pts:
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == BLACK:
                    black_n += 1
            if black_n <= 1:
                leaf_pixels.add((r, c))
                all_targets.add((r, c))

    return internal_groups, leaf_pixels, all_targets


def _halo_cells(
    targets: set[tuple[int, int]] | list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    """Return the Chebyshev-1 halo footprint excluding target centers."""
    H, W = shape
    halo: set[tuple[int, int]] = set()
    for r, c in targets:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in blocked:
                    halo.add((rr, cc))
    return halo


def _render_output(
    base: np.ndarray,
    internal_groups: list[set[tuple[int, int]]],
    leaf_pixels: set[tuple[int, int]],
    spur_pixels: set[tuple[int, int]],
) -> np.ndarray:
    """Render type-split halos plus magenta spur markers."""
    out = np.array(base, dtype=int)
    all_targets = set().union(*internal_groups) if internal_groups else set()
    all_targets |= set(leaf_pixels)

    for group in internal_groups:
        for rr, cc in _halo_cells(group, all_targets, base.shape):
            out[rr, cc] = ORANGE

    for rr, cc in _halo_cells(leaf_pixels, all_targets, base.shape):
        out[rr, cc] = MAGENTA

    for r, c in all_targets:
        out[r, c] = BLACK
    for r, c in spur_pixels:
        out[r, c] = MAGENTA
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether a generated board visibly exercises the fused behavior."""
    spur_pixels = _detect_spurs(grid)
    base = _analysis_base(grid, spur_pixels)
    internal_groups, leaf_pixels, all_targets = _collect_targets(base)

    internal_targets = set().union(*internal_groups) if internal_groups else set()
    orange_halo = _halo_cells(internal_targets, all_targets, base.shape)
    leaf_halo = _halo_cells(leaf_pixels, all_targets, base.shape)
    fused = _render_output(base, internal_groups, leaf_pixels, spur_pixels)

    original_like = np.array(base, dtype=int)
    all_halo = _halo_cells(all_targets, all_targets, base.shape)
    for r, c in all_halo:
        original_like[r, c] = ORANGE
    for r, c in all_targets:
        original_like[r, c] = BLACK

    input_noise_colors = {
        int(grid[r, c])
        for r, c in spur_pixels
        if int(grid[r, c]) not in (BLACK, BLUE)
    }
    return {
        "internal_groups": len(internal_groups),
        "leaf_pixels": len(leaf_pixels),
        "spur_pixels": len(spur_pixels),
        "orange_halo_cells": len(orange_halo),
        "leaf_halo_cells": len(leaf_halo),
        "input_noise_colors": len(input_noise_colors),
        "delta_vs_original": int(np.count_nonzero(fused != original_like)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark foreign-color spur specks and split black-defect halos by anomaly type.

    Input structure:
    - Large blue (1) and black (0) regions tile the board, with a few tiny black defects fully embedded in blue and a few black leaf tips sticking out from larger black regions.
    - Black regions may also contain isolated single-pixel noise specks in several other colors (for example red, green, yellow, teal, or maroon) rather than only blue.

    Rule:
    - First detect every non-blue/non-black pixel that has at least three effective black 4-neighbor sides, using the same corner-outside-as-black convention as the source task; treat those spur pixels as black only for structural analysis.
    - On that cleaned black mask, classify every black 4-connected component of size <= 3 as an internal defect, and every pixel of a larger black component with black 4-degree <= 1 as a leaf tip.
    - Paint a Chebyshev-1 halo around each anomaly: internal defects use orange (7), leaf tips use magenta (6), anomaly centers stay black, and the detected spur pixels themselves are rendered as magenta singletons.

    Key insight:
    - The task still depends on reconstructing the black geometry before classifying anomalies, but now two distinct magenta motifs coexist: isolated foreign-color specks inside black regions become standalone magenta dots, while black leaf tips attached to large regions get magenta rings around black centers; tiny detached black defects are the only anomalies with orange rings.
    """
    spurs = _detect_spurs(grid)
    base = _analysis_base(grid, spurs)
    internal_groups, leaf_pixels, _all_targets = _collect_targets(base)
    return _render_output(base, internal_groups, leaf_pixels, spurs)


# --------------- generator ---------------

def _build_partition(H: int, W: int) -> np.ndarray:
    """Create an HxW board partitioned into large blue and black regions."""
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        population = list(range(CUT_MARGIN, H - CUT_MARGIN))
        if not population:
            return grid
        cuts = sorted(random.sample(population, min(n_cuts, len(population))))
        boundaries = [0] + cuts + [H]
        first_color = random.randint(0, 1)
        for idx in range(len(boundaries) - 1):
            grid[boundaries[idx] : boundaries[idx + 1], :] = (first_color + idx) % 2

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
    """Return blue pixels whose 4-neighbors are all blue or out-of-bounds."""
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


def _place_internal_defects(
    grid: np.ndarray,
    interior_blue: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], int]:
    """Place detached 1-3 cell black components inside blue interiors."""
    H, W = grid.shape
    defect_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    placed_groups = 0

    for _ in range(n_defects):
        if not interior_blue:
            break
        size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]

        for _try in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = random.choice(interior_blue)
            if _nearby(seed_r, seed_c, TARGET_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            frontier = [(seed_r, seed_c)]
            while len(defect) < size and frontier:
                cr, cc = frontier.pop(0)
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels or (nr, nc) in defect:
                        continue
                    if _nearby(nr, nc, TARGET_CLEARANCE, defect_pixels):
                        continue
                    defect.append((nr, nc))
                    frontier.append((nr, nc))
                    if len(defect) >= size:
                        break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for rr, cc in _neighbors4(r, c):
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W:
                        if int(grid[rr, cc]) != BLUE or (rr, cc) in defect_pixels:
                            valid = False
                            break
                if not valid:
                    break

            if valid:
                defect_pixels.update(defect_set)
                placed_groups += 1
                remove = _exclusion_zone(defect_set, TARGET_CLEARANCE)
                interior_blue[:] = [p for p in interior_blue if p not in remove]
                interior_set -= remove
                break

    return defect_pixels, placed_groups


def _leaf_candidates(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Find blue cells that can become one-step leaf tips of large black components."""
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for comp in components_mask(grid == BLACK, connectivity=4):
        pts = {(int(r), int(c)) for r, c in comp}
        if len(pts) <= 3:
            continue
        for r, c in pts:
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W) or int(grid[rr, cc]) != BLUE:
                    continue
                if (rr, cc) in blocked or _nearby(rr, cc, TARGET_CLEARANCE, blocked):
                    continue
                black_n = 0
                for ar, ac in _neighbors4(rr, cc):
                    if 0 <= ar < H and 0 <= ac < W and int(grid[ar, ac]) == BLACK:
                        black_n += 1
                if black_n == 1:
                    candidates.append((rr, cc))
    random.shuffle(candidates)
    return candidates


def _place_leaves(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Place 1-3 single-pixel black leaf tips on large black-region boundaries."""
    leaf_pixels: set[tuple[int, int]] = set()
    wanted = random.randint(*LEAF_COUNT_RANGE)

    for _ in range(wanted):
        candidates = _leaf_candidates(grid, blocked | leaf_pixels)
        if not candidates:
            break
        r, c = candidates[0]
        grid[r, c] = BLACK
        leaf_pixels.add((r, c))
    return leaf_pixels


def _find_spur_candidates(
    grid: np.ndarray,
    exclude: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Find black cells eligible to become foreign-color single-pixel spur noise."""
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != BLACK:
                continue
            if (r, c) in exclude:
                continue
            if _effective_black_count(grid, r, c) >= 3:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _assign_spur_colors(n_spurs: int) -> list[int]:
    """Choose 2-3 distinct foreign colors and repeat them across spur pixels."""
    if n_spurs <= 0:
        return []
    n_colors = min(random.randint(2, 3), n_spurs, len(NOISE_PALETTE))
    chosen = random.sample(NOISE_PALETTE, n_colors)
    colors = chosen.copy()
    while len(colors) < n_spurs:
        colors.append(random.choice(chosen))
    random.shuffle(colors)
    return colors[:n_spurs]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)

    blue_comps = components_mask(grid == BLUE, connectivity=4)
    black_comps = components_mask(grid == BLACK, connectivity=4)
    if not blue_comps or not black_comps:
        return None
    if any(len(comp) < MIN_BLUE_REGION for comp in blue_comps):
        return None
    if any(len(comp) < MIN_BLACK_REGION for comp in black_comps):
        return None

    interior_blue = _find_interior_blue(grid)
    defect_pixels, placed_groups = _place_internal_defects(grid, interior_blue)
    if placed_groups < MIN_DEFECT_GROUPS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, defect_pixels)
    if len(leaf_pixels) < MIN_LEAVES:
        return None

    internal_groups, detected_leafs, all_targets = _collect_targets(grid)
    if len(internal_groups) < MIN_DEFECT_GROUPS or len(detected_leafs) < MIN_LEAVES:
        return None
    if leaf_pixels != detected_leafs:
        return None

    protected = _exclusion_zone(all_targets, SPUR_TARGET_CLEARANCE)
    spur_candidates = _find_spur_candidates(grid, protected)
    wanted_spurs = random.randint(*SPUR_COUNT_RANGE)
    chosen_spurs: list[tuple[int, int]] = []
    chosen_set: set[tuple[int, int]] = set()
    for r, c in spur_candidates:
        if _nearby(r, c, SPUR_SPUR_CLEARANCE, chosen_set):
            continue
        chosen_spurs.append((r, c))
        chosen_set.add((r, c))
        if len(chosen_spurs) >= wanted_spurs:
            break
    n_spurs = len(chosen_spurs)
    if n_spurs < MIN_SPURS:
        return None

    inp = np.array(grid, dtype=int)
    spur_pixels = chosen_spurs
    spur_colors = _assign_spur_colors(n_spurs)
    if len(set(spur_colors)) < 2:
        return None
    for (r, c), color in zip(spur_pixels, spur_colors):
        inp[r, c] = color

    metrics = _behavior_metrics(inp)
    if metrics["orange_halo_cells"] == 0:
        return None
    if metrics["leaf_halo_cells"] == 0:
        return None
    if metrics["spur_pixels"] < MIN_SPURS:
        return None
    if metrics["input_noise_colors"] < 2:
        return None
    if metrics["delta_vs_original"] == 0:
        return None

    output = _render_output(grid, internal_groups, detected_leafs, set(spur_pixels))
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
