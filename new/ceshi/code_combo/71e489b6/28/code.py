# Combo variant of 71e489b6 (index 28)
# combo_spec: D1/0+D4/0+D13/0
# Name: 十字光晕+仅边界缺陷+8连通
#
# Design analysis
# a) Concrete behavior:
#    - Original: absorb blue spurs, treat every <=3 black component as a defect,
#      treat >3 black components by 4-neighbor leaf pixels, then paint a 3x3 halo.
#    - Fused: absorb the same spurs, but split black with 8-connectivity, keep
#      only boundary-touching pixels from small components, keep boundary leaves
#      from large components, and paint only the orthogonal cross.
#    - Example: a 3-cell horizontal black chain plus one diagonal black pixel is
#      two components under 4-connectivity but one size-4 component under
#      8-connectivity. The fused solver therefore switches from "mark all cells
#      of a small component" to "mark only leaf-like boundary cells of a large
#      component", which visibly changes the preserved black centers.
# b) Complexity calibration:
#    - The generator always includes repeated support for three regimes:
#      ignored 1-cell holes, marked 2-3 cell boundary defects, and at least one
#      diagonal 8-connected merge. This keeps the rule visible in 2-4 examples.
# c) Implicit assumptions checked:
#    - Spur absorption still assumes small blue pixels sit in black-heavy
#      4-neighborhoods.
#    - Small-defect targeting now assumes a target has both a blue 4-neighbor
#      and a black 4-neighbor, so the generator must include size-2/3 defects
#      and isolated singletons separately.
#    - Large-component targeting still uses 4-neighbor degree, even though the
#      component itself is built with 8-connectivity; the generator therefore
#      constructs diagonal merges on purpose and keeps them visually separated
#      from unrelated black regions.
# d) Placement legality:
#    - Interior defect patterns are placed only on blue cells whose surrounding
#      8-neighborhood is blue, preventing accidental merges or overlap.
#    - Leaf pixels are added only on blue cells adjacent to a large black
#      component and are spaced away from other structures.
# e) Visual observability:
#    - Singletons stay black with no orange, 2-3 cell defects get orange
#      orthogonal arms, and diagonal merges alter which black cells remain
#      centers. These are directly inspectable without hidden numeric cues.
# f) Cleanliness:
#    - Defects are kept several cells apart so cross halos do not collapse into
#      one orange mass.
# g) Inferability:
#    - Colors remain the original three-color palette (0/1/7). The new pattern
#      is conveyed by geometry only: cross vs square, boundary-only vs all-small,
#      and 8-connected regrouping via diagonal contact.

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask  # noqa: E402


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
DEFECT_CLEARANCE = 2
SPUR_COUNT_RANGE = (2, 6)
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 400

MARKED_PATTERNS = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
]

DIAGONAL_MERGE_PATTERNS = [
    [(0, 0), (0, 1), (0, 2), (-1, -1)],
    [(0, 0), (0, 1), (0, 2), (-1, 3)],
    [(0, 0), (0, 1), (0, 2), (1, -1)],
    [(0, 0), (0, 1), (0, 2), (1, 3)],
    [(0, 0), (1, 0), (2, 0), (-1, -1)],
    [(0, 0), (1, 0), (2, 0), (3, -1)],
    [(0, 0), (1, 0), (2, 0), (-1, 1)],
    [(0, 0), (1, 0), (2, 0), (3, 1)],
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(pixels: set[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
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
                black_n += oob
            if black_n >= 3:
                base[r, c] = 0
    return base


def _target_details(
    base: np.ndarray,
    *,
    connectivity: int,
    boundary_only_small: bool,
) -> tuple[set[tuple[int, int]], dict[str, int]]:
    H, W = base.shape
    preserve: set[tuple[int, int]] = set()
    stats = {
        "small_components": 0,
        "small_target_cells": 0,
        "ignored_singletons": 0,
        "large_components": 0,
        "large_target_cells": 0,
    }

    for coords in components_mask(base == 0, connectivity=connectivity):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            stats["small_components"] += 1
            boundary_pts: list[tuple[int, int]] = []
            for r, c in pts:
                blue_adj = False
                black_adj = False
                for rr, cc in _neighbors4(r, c):
                    if not (0 <= rr < H and 0 <= cc < W):
                        continue
                    if int(base[rr, cc]) == 1:
                        blue_adj = True
                    elif int(base[rr, cc]) == 0:
                        black_adj = True
                if blue_adj and black_adj:
                    boundary_pts.append((r, c))
            targets = boundary_pts if boundary_only_small else pts
            if boundary_only_small and len(pts) == 1 and not boundary_pts:
                stats["ignored_singletons"] += 1
            preserve.update(targets)
            stats["small_target_cells"] += len(targets)
            continue

        stats["large_components"] += 1
        for r, c in pts:
            blue_adj = False
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if int(base[rr, cc]) == 1:
                    blue_adj = True
                elif int(base[rr, cc]) == 0:
                    black_n += 1
            if blue_adj and black_n <= 1:
                preserve.add((r, c))
                stats["large_target_cells"] += 1

    return preserve, stats


def _paint_halo(
    base: np.ndarray,
    preserve: set[tuple[int, int]],
    *,
    halo_shape: str,
) -> np.ndarray:
    out = base.copy()
    if halo_shape == "cross":
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        ]
    H, W = base.shape
    for r, c in preserve:
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                out[rr, cc] = 7
    for r, c in preserve:
        out[r, c] = 0
    return out


def _solve_variant(
    grid: np.ndarray,
    *,
    connectivity: int,
    boundary_only_small: bool,
    halo_shape: str,
) -> np.ndarray:
    base = _absorb_spurs(grid)
    preserve, _ = _target_details(
        base,
        connectivity=connectivity,
        boundary_only_small=boundary_only_small,
    )
    return _paint_halo(base, preserve, halo_shape=halo_shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark only boundary-facing black anomalies inside the blue-vs-black partition.

    Input structure:
    - Large blue and black regions tile the board; blue regions may contain small
      black holes, while black regions may contain thin blue spur noise.
    - Some black anomalies are isolated single pixels; others are short 2-3 cell
      chains or diagonal-touching groups that look connected only in 8-neighborhood.

    Rule:
    - First absorb blue spur pixels that are surrounded by black on at least
      three effective 4-neighbor sides.
    - Then find black connected components using 8-connectivity. In components of
      size 1-3, keep only pixels that touch both blue and black in 4-neighborhood;
      in larger components, keep only boundary leaf pixels with at most one black
      4-neighbor.
    - Paint orange on the four orthogonal neighbors of each kept pixel, while the
      kept black pixels themselves remain black.

    Key insight:
    - The important changes are geometric, not chromatic: diagonal contact can
      merge components, single-pixel holes stop being marked, and the orange
      marking is a cross instead of a full 3x3 halo.
    """
    return _solve_variant(
        np.array(grid, dtype=int),
        connectivity=8,
        boundary_only_small=True,
        halo_shape="cross",
    )


def _build_partition(H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cut_pool = list(range(CUT_MARGIN, H - CUT_MARGIN))
        if not cut_pool:
            return grid
        cuts = sorted(random.sample(cut_pool, min(n_cuts, len(cut_pool))))
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
        hc = random.randint(H // 3, 2 * H // 3)
        if random.random() < 0.5:
            grid[:, :vc] = 1 - bg
            grid[:hc, vc:] = 1 - bg
        else:
            grid[:, vc:] = 1 - bg
            grid[hc:, :vc] = 1 - bg

    return grid


def _valid_large_regions(grid: np.ndarray) -> bool:
    blue_comps = list(components_mask(grid == 1, connectivity=4))
    black_comps = list(components_mask(grid == 0, connectivity=4))
    return (
        blue_comps
        and black_comps
        and all(len(comp) >= MIN_BLUE_REGION for comp in blue_comps)
        and all(len(comp) >= MIN_BLACK_REGION for comp in black_comps)
    )


def _can_place_in_blue(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if grid[r, c] != 1 or (r, c) in forbidden:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W) or (rr, cc) in cell_set:
                    continue
                if grid[rr, cc] != 1 or (rr, cc) in forbidden:
                    return False
    return True


def _choose_interior_pattern(
    grid: np.ndarray,
    pattern_pool: list[list[tuple[int, int]]],
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    H, W = grid.shape
    candidates: list[set[tuple[int, int]]] = []
    for pattern in pattern_pool:
        for r in range(H):
            for c in range(W):
                cells = [(r + dr, c + dc) for dr, dc in pattern]
                if _can_place_in_blue(grid, cells, forbidden):
                    candidates.append(set(cells))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_pattern(
    grid: np.ndarray,
    pattern_pool: list[list[tuple[int, int]]],
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    cells = _choose_interior_pattern(grid, pattern_pool, forbidden)
    if cells is None:
        return None
    for r, c in cells:
        grid[r, c] = 0
    return cells


def _place_leaf_pixels(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    *,
    min_count: int,
    max_count: int,
) -> set[tuple[int, int]] | None:
    H, W = grid.shape
    leaf_pixels: set[tuple[int, int]] = set()
    target = random.randint(min_count, max_count)

    black_comps = list(components_mask(grid == 0, connectivity=4))
    large_black = [set((int(r), int(c)) for r, c in comp) for comp in black_comps if len(comp) > 3]
    random.shuffle(large_black)

    for comp_set in large_black:
        if len(leaf_pixels) >= target:
            break
        candidates = []
        for r, c in comp_set:
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if grid[rr, cc] != 1 or (rr, cc) in forbidden:
                    continue
                black_n = 0
                for r2, c2 in _neighbors4(rr, cc):
                    if 0 <= r2 < H and 0 <= c2 < W and grid[r2, c2] == 0:
                        black_n += 1
                if black_n == 1:
                    candidates.append((rr, cc))
        random.shuffle(candidates)
        if candidates:
            rr, cc = candidates[0]
            grid[rr, cc] = 0
            leaf_pixels.add((rr, cc))
            forbidden.add((rr, cc))

    if len(leaf_pixels) < min_count:
        return None
    return leaf_pixels


def _find_spur_candidates(
    grid: np.ndarray,
    exclude: set[tuple[int, int]],
) -> list[tuple[int, int]]:
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
    candidates = []
    for r, c in zip(rs.tolist(), cs.tolist()):
        if not _nearby(r, c, 1, exclude):
            candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    base = _build_partition(H, W)
    if not _valid_large_regions(base):
        return None

    forbidden: set[tuple[int, int]] = set()
    structural_black: set[tuple[int, int]] = set()

    diag_cells = _place_pattern(base, DIAGONAL_MERGE_PATTERNS, forbidden)
    if diag_cells is None:
        return None
    structural_black |= diag_cells
    forbidden |= _exclusion_zone(diag_cells, DEFECT_CLEARANCE)

    singleton_count = random.randint(1, 2)
    singletons: set[tuple[int, int]] = set()
    for _ in range(singleton_count):
        cells = _place_pattern(base, [[(0, 0)]], forbidden)
        if cells is None:
            return None
        singletons |= cells
        structural_black |= cells
        forbidden |= _exclusion_zone(cells, DEFECT_CLEARANCE)

    extra_marked = random.randint(1, 3)
    marked_cells: set[tuple[int, int]] = set(diag_cells)
    for _ in range(extra_marked):
        cells = _place_pattern(base, MARKED_PATTERNS, forbidden)
        if cells is None:
            return None
        marked_cells |= cells
        structural_black |= cells
        forbidden |= _exclusion_zone(cells, DEFECT_CLEARANCE)

    leaf_pixels = _place_leaf_pixels(base, forbidden, min_count=1, max_count=3)
    if leaf_pixels is None:
        return None
    structural_black |= leaf_pixels
    forbidden |= _exclusion_zone(leaf_pixels, DEFECT_CLEARANCE)

    inp = base.copy()
    spur_count = random.randint(*SPUR_COUNT_RANGE)
    spur_candidates = _find_spur_candidates(inp, structural_black)
    if min(spur_count, len(spur_candidates)) < MIN_SPURS:
        return None
    for r, c in spur_candidates[:spur_count]:
        inp[r, c] = 1

    fused = main(inp)
    d1less = _solve_variant(inp, connectivity=8, boundary_only_small=True, halo_shape="square")
    d4less = _solve_variant(inp, connectivity=8, boundary_only_small=False, halo_shape="cross")
    d13less = _solve_variant(inp, connectivity=4, boundary_only_small=True, halo_shape="cross")
    if np.array_equal(fused, d1less) or np.array_equal(fused, d4less) or np.array_equal(fused, d13less):
        return None

    absorbed = _absorb_spurs(inp)
    _, stats = _target_details(absorbed, connectivity=8, boundary_only_small=True)
    if stats["ignored_singletons"] < 1:
        return None
    if stats["small_target_cells"] < 2:
        return None
    if stats["large_target_cells"] < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
