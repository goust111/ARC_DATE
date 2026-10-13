import random
from collections import Counter

import numpy as np


BG = 0
FENCE = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS.values():
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _solve_gap_for_component(
    g: np.ndarray, comp: list[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    h, w = g.shape
    comp_set = set(comp)
    candidates: set[tuple[int, int]] = set()

    for r, c in comp:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == BG:
                candidates.add((nr, nc))

    solutions: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
    for gr, gc in candidates:
        grey_neighbors = 0
        for nr, nc in _neighbors4(gr, gc):
            if (nr, nc) in comp_set:
                grey_neighbors += 1
        if grey_neighbors < 2:
            continue

        closed = g == FENCE
        closed = closed.copy()
        closed[gr, gc] = True
        passable = ~closed

        for region in _components(passable):
            if _touches_border(region, h, w):
                continue
            region_set = set(region)
            inside_dirs = []
            for dr, dc in DIRS.values():
                nr, nc = gr + dr, gc + dc
                if (nr, nc) in region_set:
                    inside_dirs.append((dr, dc))
            if len(inside_dirs) != 1:
                continue

            seed_vals = [int(g[r, c]) for r, c in region if int(g[r, c]) not in (BG, FENCE)]
            if not seed_vals:
                continue
            counts = Counter(seed_vals)
            if len(counts) != 1:
                continue
            seed_color, seed_count = next(iter(counts.items()))
            inside_dr, inside_dc = inside_dirs[0]
            out_dir = (-inside_dr, -inside_dc)
            solutions.append(((gr, gc), out_dir, int(seed_color), int(seed_count)))

    assert len(solutions) == 1, "each grey fence must have one unambiguous seeded gap"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow seed-colored branches outward from the single gap of each grey fence.

    Input structure:
    - Black is the background and grey pixels form one or more 4-connected fence components.
    - Each fence is a closed enclosure except for exactly one single-cell black gap on its boundary.
    - Inside every fence are one or more seed pixels, all sharing one non-black, non-grey color; different fences use different seed colors.
    - The exterior of the fences is black, so the only colored pixels outside the grey fences are absent.

    Rule:
    - Keep all original grey fence pixels and all original seed pixels unchanged.
    - For each grey fence, find the unique black boundary cell which would close the fence and isolate its seeded interior.
    - The side of that gap adjacent to the isolated interior gives the inward direction, so the branch grows in the opposite outward direction.
    - Starting at the gap cell, paint a straight branch in the seed color for as many pixels as there are seed pixels inside that fence, clipping at the grid boundary.

    Key insight: The gap determines the branch direction, while the count of same-colored interior seed pixels determines the branch length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()

    grey_components = _components(g == FENCE)
    assert grey_components, "expected at least one grey fence"

    for comp in grey_components:
        (gr, gc), (dr, dc), seed_color, seed_count = _solve_gap_for_component(g, comp)
        for step in range(seed_count):
            rr = gr + dr * step
            cc = gc + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if int(g[rr, cc]) == BG:
                out[rr, cc] = seed_color

    return out


def _branch_cells(gap: tuple[int, int], direction: str, length: int, h: int, w: int) -> set[tuple[int, int]]:
    dr, dc = DIRS[direction]
    cells: set[tuple[int, int]] = set()
    gr, gc = gap
    for step in range(length):
        r = gr + dr * step
        c = gc + dc * step
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.add((r, c))
    return cells


def _rect_fence(
    r0: int, c0: int, fh: int, fw: int, direction: str, gap_offset: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    ring: set[tuple[int, int]] = set()
    for c in range(c0, c0 + fw):
        ring.add((r0, c))
        ring.add((r0 + fh - 1, c))
    for r in range(r0, r0 + fh):
        ring.add((r, c0))
        ring.add((r, c0 + fw - 1))

    if direction == "up":
        gap = (r0, c0 + gap_offset)
    elif direction == "down":
        gap = (r0 + fh - 1, c0 + gap_offset)
    elif direction == "left":
        gap = (r0 + gap_offset, c0)
    else:
        gap = (r0 + gap_offset, c0 + fw - 1)
    ring.remove(gap)

    interior = {
        (r, c)
        for r in range(r0 + 1, r0 + fh - 1)
        for c in range(c0 + 1, c0 + fw - 1)
    }
    return ring, interior, gap


def _add_bumps(
    rng: random.Random,
    ring: set[tuple[int, int]],
    r0: int,
    c0: int,
    fh: int,
    fw: int,
    gap: tuple[int, int],
    branch: set[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    bumped = set(ring)
    for _ in range(rng.randint(0, 3)):
        side = rng.choice(list(DIRS))
        if side in ("up", "down"):
            c = rng.randint(c0 + 1, c0 + fw - 2)
            r = r0 if side == "up" else r0 + fh - 1
        else:
            r = rng.randint(r0 + 1, r0 + fh - 2)
            c = c0 if side == "left" else c0 + fw - 1

        if abs(r - gap[0]) + abs(c - gap[1]) <= 1:
            continue
        dr, dc = DIRS[side]
        p1 = (r + dr, c + dc)
        if not (0 <= p1[0] < h and 0 <= p1[1] < w):
            continue
        if p1 == gap or p1 in branch:
            continue
        bumped.add(p1)

        if rng.random() < 0.25:
            p2 = (p1[0] + dr, p1[1] + dc)
            if 0 <= p2[0] < h and 0 <= p2[1] < w and p2 not in branch and p2 != gap:
                bumped.add(p2)
    return bumped


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _available_branch(gap: tuple[int, int], direction: str, h: int, w: int) -> int:
    r, c = gap
    if direction == "up":
        return r + 1
    if direction == "down":
        return h - r
    if direction == "left":
        return c + 1
    return w - c


def _draw_expected(
    grid: np.ndarray,
    specs: list[tuple[tuple[int, int], str, int, int]],
) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()
    for gap, direction, color, length in specs:
        dr, dc = DIRS[direction]
        gr, gc = gap
        for step in range(length):
            r = gr + dr * step
            c = gc + dc * step
            if not (0 <= r < h and 0 <= c < w):
                break
            if int(grid[r, c]) == BG:
                out[r, c] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 30)
    w = rng.randint(14, 30)
    max_objects = min(4, len(SEED_COLORS), max(1, (h * w) // 150 + 1))
    n_objects = rng.randint(1, max_objects)
    colors = rng.sample(SEED_COLORS, n_objects)

    grid = np.full((h, w), BG, dtype=int)
    blocked: set[tuple[int, int]] = set()
    specs: list[tuple[tuple[int, int], str, int, int]] = []

    for color in colors:
        placed = False
        for _ in range(250):
            fh = rng.randint(5, min(9, h - 2))
            fw = rng.randint(5, min(9, w - 2))
            if h - fh - 1 < 1 or w - fw - 1 < 1:
                continue

            r0 = rng.randint(1, h - fh - 1)
            c0 = rng.randint(1, w - fw - 1)
            direction = rng.choice(list(DIRS))
            if direction in ("up", "down"):
                gap_offset = rng.randint(1, fw - 2)
            else:
                gap_offset = rng.randint(1, fh - 2)

            ring, interior, gap = _rect_fence(r0, c0, fh, fw, direction, gap_offset)
            seed_count = rng.randint(1, min(8, len(interior)))
            available = _available_branch(gap, direction, h, w)
            want_clip = rng.random() < 0.35
            if want_clip and seed_count <= available:
                continue
            if not want_clip and seed_count > available:
                continue

            branch = _branch_cells(gap, direction, seed_count, h, w)
            ring = _add_bumps(rng, ring, r0, c0, fh, fw, gap, branch, h, w)
            if ring & branch:
                continue

            inward = OPPOSITE[direction]
            idr, idc = DIRS[inward]
            near_gap = (gap[0] + idr, gap[1] + idc)
            seed_pool = sorted(interior - {near_gap})
            if len(seed_pool) < seed_count:
                seed_pool = sorted(interior)
            seed_cells = set(rng.sample(seed_pool, seed_count))

            footprint = set(ring) | set(interior) | set(branch)
            if any(cell in blocked for cell in footprint):
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint):
                continue

            for r, c in ring:
                grid[r, c] = FENCE
            for r, c in seed_cells:
                grid[r, c] = color

            blocked.update(_halo(footprint, h, w))
            specs.append((gap, direction, color, seed_count))
            placed = True
            break

        if not placed:
            return None

    expected = _draw_expected(grid, specs)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(600):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid seeded-gap fence grid")
