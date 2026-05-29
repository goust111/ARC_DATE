# Combo variant of 71e489b6 (index 12)
# combo_spec: D5/2+D6/2
# Name: 同心矩形布局+区域位置分色光晕

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask


# Design analysis for combo D5/2 + D6/2
# a) Concrete layout and behavior trace
#    The original task treated all blue regions as equivalent and always painted the same
#    orange halo. In this combo the layout is no longer a generic partition: the whole
#    grid becomes nested axis-aligned rectangles that alternate blue and black, with the
#    outermost layer blue. Example trace:
#      clean layout:        add a defect in outer blue:      add one in inner blue:
#        11111111             11101111                         11101111
#        10000001             10000001                         10000001
#        10111101             10111101                         10110101
#        10111101   --->      10111101          --->           10110101
#        10000001             10000001                         10000001
#        11111111             11111111                         11111111
#    After solving, both black defects still receive Chebyshev-1 halos, but the outer
#    one is orange while the deeper one is magenta or yellow according to which blue ring
#    contains it. The spur-absorption step remains unchanged and happens before ring
#    analysis, so fake blue pixels inside black rings disappear first.
# b) Complexity calibration
#    Using too many ring colors would be unlearnable from 2-4 examples, while using only
#    one colored ring would collapse back to the original task. The design therefore fixes
#    three ordered blue-ring colors at most: outer -> orange (7), middle -> magenta (6),
#    inner -> yellow (4). The generator always creates three blue regions and guarantees
#    at least one defect per region, so the mapping repeats inside each sample instead of
#    appearing as a one-off coincidence.
# c) Implicit assumptions review
#    - Original assumption "all preserve pixels share one halo color" is invalid now;
#      `main()` must assign color per target pixel after identifying blue-ring ownership.
#    - Original assumption "any region partition is acceptable" is invalid; ordering blue
#      components by bounding-box containment is only safe because the generator enforces
#      strictly nested rectangular rings.
#    - Original assumption "leaf detection alone is enough" still holds, but each leaf now
#      needs a second inference step: inspect adjacent blue cells to determine which ring
#      the protrusion extends into.
#    - Spur absorption must still run before component analysis; otherwise temporary blue
#      specks inside black rings would split the ring structure and break the nesting sort.
# d) Placement legality
#    The generator uses only nested rectangles with positive thickness, keeps every planned
#    target at Chebyshev distance >= 3 from every other target, and inserts leaves only on
#    blue cells that currently touch exactly one black neighbor. That prevents overlap,
#    out-of-bounds growth, cross-color halo collisions, and accidental merging between tiny
#    defects and large black rings.
# e) Visual observability
#    A human can see three concentric blue zones separated by black frames, then observe
#    that anomalies in different zones get different halo colors while shapes stay the same.
#    The color rule is tied to visible position, not to hidden numeric ids or scan order.
# f) Visual cleanliness
#    Thin-ring clutter is avoided by forcing only one or two leaves and by reserving a full
#    3x3 halo buffer between targets. This keeps each colored patch isolated and readable.
# g) Change inferability
#    The color vocabulary is capped at three and the ordering is spatial: farther outward
#    means orange, deeper inward means magenta, deepest means yellow. Because all three
#    regions are present in the generated samples, the rule can be inferred from structure.
# h) Fusion-specific degeneration check
#    If every sample had defects only in one ring, the output could look identical to the
#    original task except for a palette swap. The generator prevents this by placing at
#    least one black defect in each blue ring before any extra random targets are added.
# i) New solver assumptions vs generator guarantees
#    `main()` assumes every meaningful small black component is enclosed by exactly one blue
#    ring and every meaningful leaf has adjacent blue neighbors from exactly one ring. The
#    nested-frame generator guarantees both by construction and keeps the central black mass
#    large enough that it can never be mistaken for a size<=3 defect.


HALO_COLORS = (7, 6, 4)  # outer, middle, inner
GRID_H_RANGE = (17, 19)
GRID_W_RANGE = (17, 19)
RING_PATTERNS = (
    (2, 2, 2, 1),  # blue, black, blue, black, then inner blue center
    (2, 1, 2, 2),
)
MIN_INNER_SIDE = 3
TARGET_CLEARANCE = 3
MANDATORY_DEFECTS_PER_RING = 1
EXTRA_DEFECT_COUNT_RANGE = (0, 3)
LEAF_COUNT_RANGE = (0, 2)
SPUR_COUNT_RANGE = (2, 5)
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 400

DEFECT_TEMPLATES = (
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (-1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (0, -1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (-1, 0), (-2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, -1), (0, -2)),
    ((0, 0), (1, 0), (0, 1)),
    ((0, 0), (-1, 0), (0, 1)),
    ((0, 0), (1, 0), (0, -1)),
    ((0, 0), (-1, 0), (0, -1)),
)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _ring_sort_key(coords: np.ndarray) -> tuple[int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return (-int(area), int(r0 + c0), int(len(coords)))


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) != 1:
                continue
            black_n = 0
            oob = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(g[rr, cc]) == 0:
                    black_n += 1
            if oob >= 2:
                black_n += oob
            if black_n >= 3:
                out[r, c] = 0
    return out


def _blue_region_map(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    comps = components_mask(grid == 1, connectivity=4)
    comps = sorted(comps, key=_ring_sort_key)
    region_ids = np.full(grid.shape, -1, dtype=int)
    for idx, coords in enumerate(comps):
        for r, c in coords:
            region_ids[int(r), int(c)] = idx
    return region_ids, comps


def _dominant_adjacent_region(
    base: np.ndarray, region_ids: np.ndarray, cells: list[tuple[int, int]]
) -> int | None:
    counts: Counter[int] = Counter()
    H, W = base.shape
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == 1:
                rid = int(region_ids[rr, cc])
                if rid >= 0:
                    counts[rid] += 1
    if not counts:
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == 1:
                        rid = int(region_ids[rr, cc])
                        if rid >= 0:
                            counts[rid] += 1
    if not counts:
        return None
    return int(counts.most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black defects inside concentric blue rings with ring-dependent halo colors.

    Input structure:
    - The canvas is a stack of nested rectangular layers: outer blue, then black, then
      blue, then black, and finally an inner blue rectangle.
    - Small black defects sit inside blue rings; large black rings may sprout one-pixel
      black leaves into adjacent blue rings; occasional blue spur pixels lie inside black.

    Rule:
    - First absorb any blue spur whose 4-neighborhood is mostly black, exactly as in the
      base task.
    - Then detect target black pixels: every size<=3 black component, plus every leaf
      pixel of a larger black component.
    - Paint a Chebyshev-1 halo around each target, but choose the halo color from the
      containing blue ring: outer blue -> orange (7), next blue -> magenta (6), deepest
      blue -> yellow (4). The target pixels themselves remain black.

    Key insight: defect shape detection is unchanged, but the output color is no longer
    global; it is determined by which nested blue region the defect or protruding leaf
    belongs to.
    """
    base = _absorb_spurs(np.asarray(grid, dtype=int))
    H, W = base.shape

    region_ids, _blue_regions = _blue_region_map(base)
    out = base.copy()
    preserve: set[tuple[int, int]] = set()
    halo_jobs: list[tuple[int, int, int, int]] = []

    for comp in components_mask(base == 0, connectivity=4):
        pts = [(int(r), int(c)) for r, c in comp]
        if len(pts) <= 3:
            rid = _dominant_adjacent_region(base, region_ids, pts)
            if rid is None:
                continue
            color = int(HALO_COLORS[min(rid, len(HALO_COLORS) - 1)])
            for r, c in pts:
                preserve.add((r, c))
                halo_jobs.append((rid, r, c, color))
            continue

        comp_set = set(pts)
        for r, c in pts:
            black_n = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) in comp_set:
                    black_n += 1
            if black_n > 1:
                continue
            rid = _dominant_adjacent_region(base, region_ids, [(r, c)])
            if rid is None:
                continue
            color = int(HALO_COLORS[min(rid, len(HALO_COLORS) - 1)])
            preserve.add((r, c))
            halo_jobs.append((rid, r, c, color))

    halo_jobs.sort()
    for _rid, r, c, color in halo_jobs:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    out[rr, cc] = color

    for r, c in preserve:
        out[r, c] = 0

    return out


def _build_nested_layout(H: int, W: int) -> np.ndarray:
    valid = [
        widths for widths in RING_PATTERNS
        if H - 2 * sum(widths) >= MIN_INNER_SIDE and W - 2 * sum(widths) >= MIN_INNER_SIDE
    ]
    if not valid:
        raise ValueError("grid too small for the required concentric layout")

    b0, k0, b1, k1 = random.choice(valid)
    grid = np.ones((H, W), dtype=int)

    inset = b0
    grid[inset:H - inset, inset:W - inset] = 0
    inset += k0
    grid[inset:H - inset, inset:W - inset] = 1
    inset += b1
    grid[inset:H - inset, inset:W - inset] = 0
    inset += k1
    grid[inset:H - inset, inset:W - inset] = 1
    return grid


def _shape_ok(
    cells: list[tuple[int, int]],
    region_ids: np.ndarray,
    region_id: int,
    occupied: set[tuple[int, int]],
) -> bool:
    H, W = region_ids.shape
    shape = set(cells)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(region_ids[r, c]) != int(region_id):
            return False
        if (r, c) in occupied or _nearby(r, c, TARGET_CLEARANCE, occupied):
            return False

    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in shape:
                continue
            if int(region_ids[rr, cc]) != int(region_id):
                return False
    return True


def _place_one_defect(
    region_ids: np.ndarray,
    region_id: int,
    occupied: set[tuple[int, int]],
    require_any: bool,
) -> list[tuple[int, int]] | None:
    cells = [(int(r), int(c)) for r, c in np.argwhere(region_ids == region_id)]
    random.shuffle(cells)

    templates = list(DEFECT_TEMPLATES)
    if require_any:
        templates = [tpl for tpl in templates if len(tpl) == 1] + templates

    for sr, sc in cells:
        if _nearby(sr, sc, TARGET_CLEARANCE, occupied):
            continue
        random.shuffle(templates)
        for tpl in templates:
            shape = [(sr + dr, sc + dc) for dr, dc in tpl]
            if _shape_ok(shape, region_ids, region_id, occupied):
                return shape
    return None


def _place_defects(grid: np.ndarray, region_ids: np.ndarray) -> dict[tuple[int, int], int] | None:
    targets: dict[tuple[int, int], int] = {}
    occupied: set[tuple[int, int]] = set()
    n_regions = int(region_ids.max()) + 1

    for rid in range(n_regions):
        for _ in range(MANDATORY_DEFECTS_PER_RING):
            defect = _place_one_defect(region_ids, rid, occupied, require_any=True)
            if defect is None:
                return None
            for cell in defect:
                occupied.add(cell)
                targets[cell] = int(HALO_COLORS[min(rid, len(HALO_COLORS) - 1)])
                grid[cell] = 0

    extra = random.randint(*EXTRA_DEFECT_COUNT_RANGE)
    region_order = list(range(n_regions))
    for _ in range(extra):
        random.shuffle(region_order)
        placed = False
        for rid in region_order:
            defect = _place_one_defect(region_ids, rid, occupied, require_any=False)
            if defect is None:
                continue
            for cell in defect:
                occupied.add(cell)
                targets[cell] = int(HALO_COLORS[min(rid, len(HALO_COLORS) - 1)])
                grid[cell] = 0
            placed = True
            break
        if not placed:
            break

    return targets


def _candidate_leaves(
    grid: np.ndarray,
    region_ids: np.ndarray,
    occupied: set[tuple[int, int]],
    comp_set: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    H, W = grid.shape
    out: list[tuple[int, int, int]] = []
    for r, c in comp_set:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if int(grid[rr, cc]) != 1:
                continue
            rid = int(region_ids[rr, cc])
            if rid < 0 or (rr, cc) in occupied or _nearby(rr, cc, TARGET_CLEARANCE, occupied):
                continue
            black_n = 0
            for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r2 = rr + dr2
                c2 = cc + dc2
                if 0 <= r2 < H and 0 <= c2 < W and int(grid[r2, c2]) == 0:
                    black_n += 1
            if black_n == 1:
                out.append((rr, cc, rid))
    random.shuffle(out)
    return out


def _place_leaves(
    grid: np.ndarray,
    region_ids: np.ndarray,
    targets: dict[tuple[int, int], int],
) -> None:
    occupied = set(targets)
    black_comps = components_mask(grid == 0, connectivity=4)
    large = [set((int(r), int(c)) for r, c in comp) for comp in black_comps if len(comp) > 6]
    random.shuffle(large)
    n_leaves = random.randint(*LEAF_COUNT_RANGE)

    placed = 0
    for comp_set in large:
        if placed >= n_leaves:
            break
        for rr, cc, rid in _candidate_leaves(grid, region_ids, occupied, comp_set):
            grid[rr, cc] = 0
            occupied.add((rr, cc))
            targets[(rr, cc)] = int(HALO_COLORS[min(rid, len(HALO_COLORS) - 1)])
            placed += 1
            break


def _build_output(grid: np.ndarray, targets: dict[tuple[int, int], int]) -> np.ndarray:
    out = grid.copy()
    preserve = set(targets)
    H, W = grid.shape
    for (r, c), color in sorted(targets.items(), key=lambda item: (item[1], item[0][0], item[0][1])):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    out[rr, cc] = int(color)
    for r, c in preserve:
        out[r, c] = 0
    return out


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

    clean = _build_nested_layout(H, W)
    region_ids, blue_regions = _blue_region_map(clean)
    if len(blue_regions) != 3:
        return None

    work = clean.copy()
    targets = _place_defects(work, region_ids)
    if targets is None:
        return None

    _place_leaves(work, region_ids, targets)
    output = _build_output(work, targets)

    inp = work.copy()
    candidates = _find_spur_candidates(inp, set(targets))
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    if len(candidates) < MIN_SPURS:
        return None
    for r, c in candidates[:n_spurs]:
        inp[r, c] = 1

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
