# Combo variant of 71e489b6 (index 38)
# combo_spec: D4/2+D8/2
# Name: 凸性检测+更大缺陷

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior:
#    Original task marked two defect types:
#    - any tiny black component of size <= 3 inside blue
#    - leaf pixels on large black regions
#    This combo removes the leaf logic and replaces the detector with a convexity
#    test on blue regions. On an ARC grid, we operationalize "more convex" as
#    "more row-convex and more column-convex": within the affected blue component,
#    the number of split blue intervals across rows and columns goes down after
#    filling a black component back to blue.
#    Example:
#      111111     row/col interval excess before fill > after fill
#      110011  -> the 2x2 black block splits rows and columns, so it is marked
#      110011     even though its size is 4, which the original solver ignored.
#      111111
#
# b) Complexity calibration:
#    If we accepted every interior black pixel, the task would collapse back to
#    "find holes". To keep the convexity cue visible, the generator only builds
#    orthogonally convex blue regions (rectangles, separated rectangles, L-shapes)
#    and then carves 1-6 pixel connected black defects out of them. At least one
#    defect per sample is size 4-6 and nontrivial (L/T/etc.), so the larger-defect
#    dimension is consistently visible.
#
# c) Hidden assumptions reviewed against the original main():
#    - Spur absorption still assumes isolated blue pixels in black regions have
#      >= 3 effective black 4-neighbors; the generator still guarantees that.
#    - The old "all small black components are defects" assumption no longer
#      holds. We now require a candidate to be fully 4-neighbor-surrounded by
#      blue and to reduce both row and column split counts when filled.
#    - The old large-black leaf logic is removed entirely; the generator therefore
#      stops placing leaf protrusions so that no unmarked branch tips compete with
#      the new rule.
#
# d/e/f/g) Placement legality and visual observability:
#    - Every defect is kept strictly inside blue, away from borders and from other
#      defects by a Chebyshev clearance buffer.
#    - Blue base regions remain large and visually simple, so the black carved
#      chunks are the only visible source of non-convexity.
#    - Orange still marks a Chebyshev-1 halo, preserving continuity with the
#      original task while making the new larger defects easy to spot.


HOLE = 0
BLUE = 1
ORANGE = 7


GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
PARTITION_STYLES = ["single_rect", "double_rects", "single_l"]
MIN_BLUE_COMPONENT = 30
DEFECT_COUNT_RANGE = (2, 6)
DEFECT_SIZE_WEIGHTS = {1: 2, 2: 3, 3: 3, 4: 4, 5: 3, 6: 3}
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 6
MIN_DEFECTS = 2
MIN_LARGE_DEFECTS = 1
SPUR_COUNT_RANGE = (2, 6)
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 400


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _shape_orientations(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    pts = list(shape)
    for reflect in (False, True):
        cur = [(-r, c) if reflect else (r, c) for r, c in pts]
        for _ in range(4):
            cur = [(-c, r) for r, c in cur]
            seen.add(_normalize_shape(tuple(cur)))
    return sorted(seen)


CANONICAL_SHAPES = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1)),
]


IRREGULAR_LARGE_CANONICAL = [
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1)),
]


def _build_shape_buckets() -> tuple[dict[int, list[tuple[tuple[int, int], ...]]], list[tuple[tuple[int, int], ...]]]:
    by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {}
    for shape in CANONICAL_SHAPES:
        for oriented in _shape_orientations(shape):
            by_size.setdefault(len(oriented), set()).add(oriented)
    irregular = set()
    for shape in IRREGULAR_LARGE_CANONICAL:
        irregular.update(_shape_orientations(shape))
    return {k: sorted(v) for k, v in by_size.items()}, sorted(irregular)


SHAPES_BY_SIZE, IRREGULAR_LARGE_SHAPES = _build_shape_buckets()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _exclusion_zone(pixels: set[tuple[int, int]] | list[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    zone = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _line_excess(vec: np.ndarray) -> int:
    idx = np.flatnonzero(vec)
    if idx.size == 0:
        return 0
    return int(np.sum(np.diff(idx) > 1))


def _interval_excess(mask: np.ndarray) -> tuple[int, int]:
    row_extra = sum(_line_excess(mask[r, :]) for r in range(mask.shape[0]))
    col_extra = sum(_line_excess(mask[:, c]) for c in range(mask.shape[1]))
    return int(row_extra), int(col_extra)


def _blue_component_mask(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    H, W = mask.shape
    sr, sc = start
    out = np.zeros_like(mask, dtype=bool)
    stack = [(int(sr), int(sc))]
    out[int(sr), int(sc)] = True
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not out[rr, cc]:
                out[rr, cc] = True
                stack.append((rr, cc))
    return out


def _absorb_spurs(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    base = g.copy()
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            black_n = 0
            oob = 0
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == HOLE:
                    black_n += 1
            if oob >= 2:
                black_n += oob
            if black_n >= 3:
                base[r, c] = HOLE
    return base


def _strictly_inside_blue(grid: np.ndarray, pts: list[tuple[int, int]]) -> bool:
    H, W = grid.shape
    pts_set = set(pts)
    for r, c in pts:
        for rr, cc in _neighbors4(r, c):
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            if (rr, cc) in pts_set:
                continue
            if int(grid[rr, cc]) != BLUE:
                return False
    return True


def _improves_convexity(grid: np.ndarray, pts: list[tuple[int, int]]) -> bool:
    filled = np.array(grid, dtype=int)
    for r, c in pts:
        filled[r, c] = BLUE
    comp_after = _blue_component_mask(filled == BLUE, pts[0])
    comp_before = comp_after & (grid == BLUE)
    row_before, col_before = _interval_excess(comp_before)
    row_after, col_after = _interval_excess(comp_after)
    return row_after < row_before and col_after < col_before


def _collect_convexity_defects(grid: np.ndarray) -> set[tuple[int, int]]:
    preserve: set[tuple[int, int]] = set()
    for coords in components_mask(grid == HOLE, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if not (1 <= len(pts) <= 6):
            continue
        if not _strictly_inside_blue(grid, pts):
            continue
        if _improves_convexity(grid, pts):
            preserve.update(pts)
    return preserve


def _build_halo(grid: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    H, W = grid.shape
    out = np.array(grid, dtype=int)
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = HOLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Highlight interior black chunks whose removal makes blue regions more convex.

    Input structure:
    - Large blue (1) regions sit on a black (0) background.
    - Some black connected chunks of size 1-6 are carved strictly inside blue.
    - Small blue spurs may also appear inside black regions as disposable noise.

    Rule:
    - First absorb thin blue spur pixels back into black when they have at least
      three effective black 4-neighbors.
    - Then inspect each 1-6 pixel black component that is fully enclosed by blue.
      If filling that component would reduce row-splits and column-splits in the
      surrounding blue component, it is a defect.
    - Paint orange (7) on the Chebyshev-1 halo of every defect, while the defect
      pixels themselves remain black.

    Key insight:
    The marked black cells are not chosen by raw size alone; they are exactly the
    small carved chunks that make an otherwise simple blue region less convex.
    """
    base = _absorb_spurs(np.array(grid, dtype=int))
    preserve = _collect_convexity_defects(base)
    return _build_halo(base, preserve)


def _build_partition(H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "single_rect":
        h = random.randint(max(8, H - 8), H - 3)
        w = random.randint(max(8, W - 8), W - 3)
        r0 = random.randint(1, H - h - 1)
        c0 = random.randint(1, W - w - 1)
        grid[r0:r0 + h, c0:c0 + w] = BLUE

    elif style == "double_rects":
        if random.random() < 0.5:
            gap = random.randint(2, 3)
            rect_h = random.randint(5, 7)
            total_h = rect_h * 2 + gap
            if total_h >= H - 1:
                return _build_partition(H, W)
            w = random.randint(max(7, W - 10), W - 3)
            r0 = random.randint(1, H - total_h - 1)
            c0 = random.randint(1, W - w - 1)
            grid[r0:r0 + rect_h, c0:c0 + w] = BLUE
            grid[r0 + rect_h + gap:r0 + 2 * rect_h + gap, c0:c0 + w] = BLUE
        else:
            gap = random.randint(2, 3)
            rect_w = random.randint(5, 7)
            total_w = rect_w * 2 + gap
            if total_w >= W - 1:
                return _build_partition(H, W)
            h = random.randint(max(7, H - 10), H - 3)
            r0 = random.randint(1, H - h - 1)
            c0 = random.randint(1, W - total_w - 1)
            grid[r0:r0 + h, c0:c0 + rect_w] = BLUE
            grid[r0:r0 + h, c0 + rect_w + gap:c0 + 2 * rect_w + gap] = BLUE

    else:
        box_h = random.randint(max(10, H - 7), H - 2)
        box_w = random.randint(max(10, W - 7), W - 2)
        r0 = random.randint(1, H - box_h)
        c0 = random.randint(1, W - box_w)
        arm_h = random.randint(4, max(4, box_h - 2))
        arm_w = random.randint(4, max(4, box_w - 2))
        orient = random.choice(["tl", "tr", "bl", "br"])
        if orient == "tl":
            grid[r0:r0 + box_h, c0:c0 + arm_w] = BLUE
            grid[r0:r0 + arm_h, c0:c0 + box_w] = BLUE
        elif orient == "tr":
            grid[r0:r0 + box_h, c0 + box_w - arm_w:c0 + box_w] = BLUE
            grid[r0:r0 + arm_h, c0:c0 + box_w] = BLUE
        elif orient == "bl":
            grid[r0:r0 + box_h, c0:c0 + arm_w] = BLUE
            grid[r0 + box_h - arm_h:r0 + box_h, c0:c0 + box_w] = BLUE
        else:
            grid[r0:r0 + box_h, c0 + box_w - arm_w:c0 + box_w] = BLUE
            grid[r0 + box_h - arm_h:r0 + box_h, c0:c0 + box_w] = BLUE

    return grid


def _shape_cells(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + dr, c0 + dc) for dr, dc in shape]


def _enumerate_placements(
    grid: np.ndarray,
    shapes: list[tuple[tuple[int, int], ...]],
    blocked: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    placements: list[list[tuple[int, int]]] = []
    for shape in shapes:
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        for r0 in range(H - max_r):
            for c0 in range(W - max_c):
                pts = _shape_cells(shape, r0, c0)
                if any((r, c) in blocked or int(grid[r, c]) != BLUE for r, c in pts):
                    continue
                if _strictly_inside_blue(grid, pts):
                    placements.append(pts)
    random.shuffle(placements)
    return placements


def _weighted_size() -> int:
    sizes = sorted(DEFECT_SIZE_WEIGHTS)
    weights = [DEFECT_SIZE_WEIGHTS[s] for s in sizes]
    return int(random.choices(sizes, weights=weights)[0])


def _place_defects(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], bool]:
    work = np.array(grid, dtype=int)
    target = random.randint(*DEFECT_COUNT_RANGE)
    blocked: set[tuple[int, int]] = set()
    defects: list[list[tuple[int, int]]] = []
    large_count = 0

    first_options = _enumerate_placements(work, IRREGULAR_LARGE_SHAPES, blocked)
    if not first_options:
        return [], False
    first = random.choice(first_options)
    defects.append(first)
    large_count += int(len(first) >= 4)
    for r, c in first:
        work[r, c] = HOLE
    blocked |= _exclusion_zone(first, DEFECT_CLEARANCE)

    for _ in range(target - 1):
        placed = False
        for _try in range(DEFECT_PLACE_TRIES):
            size = _weighted_size()
            options = _enumerate_placements(work, SHAPES_BY_SIZE[size], blocked)
            if not options:
                continue
            choice = random.choice(options)
            defects.append(choice)
            large_count += int(len(choice) >= 4)
            for r, c in choice:
                work[r, c] = HOLE
            blocked |= _exclusion_zone(choice, DEFECT_CLEARANCE)
            placed = True
            break
        if not placed:
            break

    return defects, large_count >= MIN_LARGE_DEFECTS


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    H, W = grid.shape
    black = grid == HOLE
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

    blue_comps = components_mask(clean == BLUE, connectivity=4)
    if not blue_comps or min(len(comp) for comp in blue_comps) < MIN_BLUE_COMPONENT:
        return None

    defects, large_ok = _place_defects(clean)
    if not large_ok or len(defects) < MIN_DEFECTS:
        return None

    base = clean.copy()
    preserve: set[tuple[int, int]] = set()
    for pts in defects:
        for r, c in pts:
            base[r, c] = HOLE
            preserve.add((r, c))

    if preserve != _collect_convexity_defects(base):
        return None

    output = _build_halo(base, preserve)

    inp = base.copy()
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    candidates = _find_spur_candidates(inp, preserve)
    used = candidates[:n_spurs]
    if len(used) < MIN_SPURS:
        return None
    for r, c in used:
        inp[r, c] = BLUE

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
