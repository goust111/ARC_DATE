# Combo variant of 71e489b6 (index 21)
# combo_spec: D1/0+D3/0+D5/0
# Name: 十字光晕+毛刺品红+类型分色
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Small 1-3 cell black components inside blue still count as "internal defects", and leaf pixels
#    of larger black components still count as "leaves". The changes are:
#    - the old 3x3 orange halo becomes a 4-neighbor cross;
#    - internal defects use orange crosses;
#    - leaves use magenta crosses;
#    - blue spur pixels that would previously be absorbed now stay visible as single magenta dots.
#    Example around one target pixel:
#      input      old output      fused output
#      111        777             171
#      101   ->   707        vs   707   for an internal defect
#      111        777             171
#    For a leaf pixel the fused cross is the same geometry but magenta (6), and for a spur the
#    center pixel itself becomes 6 with no surrounding halo.
#
# b) Complexity calibration:
#    The fusion would become ambiguous if magenta were used for both leaves and spurs without a
#    reliable shape cue. The generator therefore enforces both a magenta cross and isolated magenta
#    dot(s) in each sample, with spacing that prevents accidental merging into larger magenta blobs.
#
# c) Implicit assumptions changed in main():
#    - STEP 0 in the original solver destroyed spur pixels. That assumption is gone; spur detection
#      must be read-only.
#    - A single halo color is no longer valid; target classification must retain "internal" vs "leaf".
#    - A square Chebyshev footprint is no longer valid; only orthogonal neighbors may be painted.
#    - Painting order could matter if marks overlap. The generator avoids overlaps, so the solver can
#      stay simple and deterministic.
#
# d) Placement legality:
#    - Defects are placed strictly inside blue interiors, as before.
#    - Leaves are added only on blue cells with exactly one black 4-neighbor, so after conversion
#      they are genuine degree-1 pixels of a large black component.
#    - Spurs are inserted only on black cells whose effective black-neighbor count is >=3, matching
#      the original "absorbable spur" predicate, and they are kept away from cross footprints.
#
# e) Visual observability:
#    Every generated sample contains all three visible phenomena:
#    orange cross(es), magenta cross(es), and isolated magenta dot(s). A human can infer that color
#    separates internal-vs-leaf defects, while morphology separates leaf marks from spur marks.
#
# f) Visual cleanliness:
#    The generator enforces clearance between targets so orange/magenta crosses rarely overlap and
#    spur dots do not attach to magenta crosses. This keeps the output legible instead of noisy.
#
# g) Color-rule inferability:
#    Only two output colors are introduced beyond the input palette: orange for internal defects and
#    magenta for both leaves and spurs. The shared magenta is still inferable because leaves always
#    appear as 5-cell crosses around black centers, while spurs appear as isolated single pixels.
#
# Post-change assumption audit:
# - main() assumes spur dots do not sit on top of cross centers or cross arms when the puzzle is
#   meant to be human-solvable. generate_input() guarantees this by excluding a buffer around all
#   defect/leaf targets before inserting spurs.
# - main() assumes there is at least one internal defect, one leaf, and one spur if the sample is to
#   display the fused rule fully. generate_input() enforces minima for all three classes.

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVERS = ROOT / "solvers"
for path in (ROOT, SOLVERS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arc_utils import components_mask  # noqa: E402


BLUE = 1
BLACK = 0
MAGENTA = 6
ORANGE = 7
CROSS_OFFSETS = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _effective_black_neighbors(grid: np.ndarray, r: int, c: int) -> int:
    """Black 4-neighbors, with outside counted as black only at corners."""
    h, w = grid.shape
    black_n = 0
    oob = 0
    for dr, dc in N4:
        rr = r + dr
        cc = c + dc
        if not (0 <= rr < h and 0 <= cc < w):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_n += 1
    if oob >= 2:
        black_n += oob
    return black_n


def _is_spur(grid: np.ndarray, r: int, c: int) -> bool:
    return int(grid[r, c]) == BLUE and _effective_black_neighbors(grid, r, c) >= 3


def _collect_targets(grid: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Split black targets into small internal defects vs large-component leaves."""
    internal: set[tuple[int, int]] = set()
    leaves: set[tuple[int, int]] = set()
    for coords in components_mask(grid == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            internal.update(pts)
            continue
        comp_set = set(pts)
        for r, c in pts:
            deg = 0
            for dr, dc in N4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and (rr, cc) in comp_set:
                    deg += 1
            if deg <= 1:
                leaves.add((r, c))
    return internal, leaves


def _paint_crosses(
    out: np.ndarray,
    targets: set[tuple[int, int]],
    color: int,
    preserve_black: set[tuple[int, int]],
) -> None:
    for r, c in targets:
        for dr, dc in CROSS_OFFSETS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and (rr, cc) not in preserve_black:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark structural black defects inside blue regions with typed crosses, while keeping blue spurs as magenta dots.

    Input structure:
    - Large blue (1) and black (0) regions partition the board.
    - Blue regions may contain small black defects, and large black regions may poke one-cell leaves into blue.
    - Black regions may also contain thin blue spur pixels that satisfy the old "would be absorbed" condition.

    Rule:
    - Read blue spur pixels non-destructively: if a blue pixel has at least three effective black 4-neighbors
      (counting two out-of-bounds sides as black at corners), recolor only that pixel to magenta (6).
    - Detect black targets on the unchanged black structure: every black component of size 1-3 is an internal
      defect, while degree-0/1 pixels of larger black components are leaves.
    - Paint a cross-shaped halo on each target: internal defects get orange (7) crosses, leaves get magenta (6)
      crosses, and every target pixel itself stays black.

    Key insight: The output fuses two classification axes at once: color distinguishes internal defects from
    leaf defects, while shape distinguishes magenta leaf-crosses from magenta spur dots.
    """
    g = np.array(grid, dtype=int)

    internal, leaves = _collect_targets(g)
    preserve_black = internal | leaves
    spur_marks = {(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if _is_spur(g, r, c)}

    out = g.copy()
    _paint_crosses(out, internal, ORANGE, preserve_black)
    _paint_crosses(out, leaves, MAGENTA, preserve_black)

    for r, c in preserve_black:
        out[r, c] = BLACK
    for r, c in spur_marks:
        out[r, c] = MAGENTA

    return out


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
DEFECT_COUNT_RANGE = (3, 8)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 60
LEAF_COUNT_RANGE = (1, 4)
LEAF_CLEARANCE = 2
SPUR_COUNT_RANGE = (2, 5)
SPUR_CLEARANCE = 2
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


def _cross_footprint(pixels: set[tuple[int, int]] | list[tuple[int, int]]) -> set[tuple[int, int]]:
    footprint: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr, dc in CROSS_OFFSETS:
            footprint.add((r + dr, c + dc))
    return footprint


def _build_partition(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cuts = sorted(
            random.sample(range(CUT_MARGIN, h - CUT_MARGIN), min(n_cuts, h - 2 * CUT_MARGIN))
        )
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
    h, w = grid.shape
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    defect_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)
    defects_placed = 0

    for _ in range(n_defects):
        if not interior_blue:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
        for _ in range(DEFECT_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = random.choice(interior_blue)
            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, defect_pixels):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = N4[:]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in defect_pixels:
                        continue
                    ok = True
                    for dr2, dc2 in N4:
                        rr = nr + dr2
                        cc = nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < h and 0 <= cc < w:
                            if grid[rr, cc] != BLUE or (rr, cc) in defect_pixels:
                                ok = False
                                break
                    if ok and not _nearby(nr, nc, DEFECT_CLEARANCE, defect_pixels):
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for dr, dc in N4:
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if 0 <= rr < h and 0 <= cc < w:
                        if grid[rr, cc] != BLUE or (rr, cc) in defect_pixels:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                continue

            defect_pixels.update(defect)
            defects_placed += 1
            remove = _exclusion_zone(defect, DEFECT_CLEARANCE)
            interior_blue[:] = [p for p in interior_blue if p not in remove]
            interior_set -= remove
            break

    return defect_pixels, defects_placed


def _leaf_candidate_ok(
    grid: np.ndarray,
    r: int,
    c: int,
    blocked: set[tuple[int, int]],
) -> bool:
    if grid[r, c] != BLUE or _nearby(r, c, LEAF_CLEARANCE, blocked):
        return False
    black_n = 0
    for dr, dc in N4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and grid[rr, cc] == BLACK:
            black_n += 1
    return black_n == 1


def _place_leaves(grid: np.ndarray, defect_pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    for _ in range(n_leaves):
        blocked = defect_pixels | leaf_pixels
        candidates: list[tuple[int, int]] = []
        for comp in components_mask(grid == BLACK, connectivity=4):
            pts = [(int(r), int(c)) for r, c in comp]
            if len(pts) <= 3:
                continue
            comp_set = set(pts)
            for r, c in pts:
                for dr, dc in N4:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                        continue
                    if (rr, cc) in blocked:
                        continue
                    if (rr, cc) in comp_set:
                        continue
                    if _leaf_candidate_ok(grid, rr, cc, blocked):
                        candidates.append((rr, cc))
        random.shuffle(candidates)
        chosen = None
        for rr, cc in candidates:
            if _leaf_candidate_ok(grid, rr, cc, defect_pixels | leaf_pixels):
                chosen = (rr, cc)
                break
        if chosen is None:
            break
        leaf_pixels.add(chosen)
        grid[chosen] = BLACK

    return leaf_pixels


def _render_output(
    inp: np.ndarray,
    internal: set[tuple[int, int]],
    leaves: set[tuple[int, int]],
    spurs: set[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    preserve_black = internal | leaves
    _paint_crosses(out, internal, ORANGE, preserve_black)
    _paint_crosses(out, leaves, MAGENTA, preserve_black)
    for r, c in preserve_black:
        out[r, c] = BLACK
    for r, c in spurs:
        out[r, c] = MAGENTA
    return out


def _spur_block_zone(defect_pixels: set[tuple[int, int]], leaf_pixels: set[tuple[int, int]]) -> set[tuple[int, int]]:
    target_pixels = defect_pixels | leaf_pixels
    footprint = _cross_footprint(target_pixels)
    return _exclusion_zone(footprint, 1)


def _find_spur_candidates(grid: np.ndarray, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != BLACK or (r, c) in blocked or _nearby(r, c, SPUR_CLEARANCE, blocked):
                continue
            if _effective_black_neighbors(grid, r, c) >= 3:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = _build_partition(h, w)

    blue_comps = components_mask(grid == BLUE, connectivity=4)
    black_comps = components_mask(grid == BLACK, connectivity=4)
    if not blue_comps or not black_comps:
        return None
    for comp in blue_comps:
        if len(comp) < MIN_BLUE_REGION:
            return None
    for comp in black_comps:
        if len(comp) < MIN_BLACK_REGION:
            return None

    interior_blue = _find_interior_blue(grid)
    defect_pixels, defects_placed = _place_defects(grid, interior_blue)
    if defects_placed < MIN_DEFECTS:
        return None
    for r, c in defect_pixels:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, defect_pixels)
    if len(leaf_pixels) < MIN_LEAVES:
        return None

    blocked = _spur_block_zone(defect_pixels, leaf_pixels)
    inp = grid.copy()
    n_spurs_goal = random.randint(*SPUR_COUNT_RANGE)
    spur_pixels: set[tuple[int, int]] = set()

    while len(spur_pixels) < n_spurs_goal:
        candidates = _find_spur_candidates(inp, blocked | spur_pixels)
        if not candidates:
            break
        placed = False
        for r, c in candidates:
            if _nearby(r, c, SPUR_CLEARANCE, spur_pixels):
                continue
            test = inp.copy()
            test[r, c] = BLUE
            if not _is_spur(test, r, c):
                continue
            inp[r, c] = BLUE
            spur_pixels.add((r, c))
            placed = True
            break
        if not placed:
            break

    if len(spur_pixels) < MIN_SPURS:
        return None

    expected = _render_output(inp, defect_pixels, leaf_pixels, spur_pixels)
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
