import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_GRID_SIZE = 16
MAX_GRID_SIZE = 30
MIN_INTERIOR = 4
MAX_INTERIOR = 10
MAX_REGIONS = 5
GEN_RETRIES = 400
PLACE_RETRIES = 160


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _extract_regions(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = g != GREY
    regions: list[dict[str, int]] = []

    for pts in _components(mask):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in pts):
            continue

        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(pts) != area:
            continue

        if not (
            np.all(g[r0 - 1, c0 - 1 : c1 + 2] == GREY)
            and np.all(g[r1 + 1, c0 - 1 : c1 + 2] == GREY)
            and np.all(g[r0 - 1 : r1 + 2, c0 - 1] == GREY)
            and np.all(g[r0 - 1 : r1 + 2, c1 + 1] == GREY)
        ):
            continue

        inner = g[r0 : r1 + 1, c0 : c1 + 1]
        seed_mask = (inner != BLACK) & (inner != GREY)
        if int(seed_mask.sum()) != 1:
            continue

        seed_local = np.argwhere(seed_mask)[0]
        sr = int(r0 + seed_local[0])
        sc = int(c0 + seed_local[1])
        regions.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "seed_r": sr,
                "seed_c": sc,
                "color": int(g[sr, sc]),
            }
        )

    return regions


def _region_orientation(region: dict[str, int]) -> str:
    top = region["seed_r"] - region["r0"]
    bottom = region["r1"] - region["seed_r"]
    left = region["seed_c"] - region["c0"]
    right = region["c1"] - region["seed_c"]
    return "horizontal" if min(top, bottom) <= min(left, right) else "vertical"


def _paint_region(out: np.ndarray, region: dict[str, int]) -> None:
    r0, c0, r1, c1 = region["r0"], region["c0"], region["r1"], region["c1"]
    sr, sc, color = region["seed_r"], region["seed_c"], region["color"]

    top = sr - r0
    bottom = r1 - sr
    left = sc - c0
    right = c1 - sc
    d = min(top, bottom, left, right)
    horizontal = min(top, bottom) <= min(left, right)

    if horizontal:
        if d == 0:
            out[r0 : r1 + 1, c0 : c1 + 1] = color
            return
        for rr in range(sr, r1 + 1, d):
            out[rr, c0 : c1 + 1] = color
        for rr in range(sr - d, r0 - 1, -d):
            out[rr, c0 : c1 + 1] = color
        return

    if d == 0:
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        return
    for cc in range(sc, c1 + 1, d):
        out[r0 : r1 + 1, cc] = color
    for cc in range(sc - d, c0 - 1, -d):
        out[r0 : r1 + 1, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed rectangular region with seed-colored stripes.

    The stripe direction is determined by the nearest inner boundary to the
    region's unique seed, with top/bottom ties taking precedence over left/right.
    The stripe spacing equals that nearest-boundary distance.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for region in _extract_regions(g):
        _paint_region(out, region)
    return out


def _branch_for_seed(local_r: int, local_c: int, inner_h: int, inner_w: int) -> str:
    top = local_r
    bottom = inner_h - 1 - local_r
    left = local_c
    right = inner_w - 1 - local_c
    tb = min(top, bottom)
    lr = min(left, right)
    if tb < lr:
        return "h_strict"
    if tb == lr:
        return "h_tie"
    return "v_strict"


def _seed_candidates(inner_h: int, inner_w: int, branch: str) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r in range(1, inner_h - 1):
        for c in range(1, inner_w - 1):
            if _branch_for_seed(r, c, inner_h, inner_w) == branch:
                coords.append((r, c))
    return coords


def _min_seed_distance(local_r: int, local_c: int, inner_h: int, inner_w: int) -> int:
    return min(local_r, inner_h - 1 - local_r, local_c, inner_w - 1 - local_c)


def _sample_branch_plan(n_regions: int, rng: random.Random) -> list[str]:
    if n_regions == 1:
        return ["h_tie" if rng.random() < 0.2 else ("h_strict" if rng.random() < 0.5 else "v_strict")]

    plan = ["h_strict", "v_strict"]
    while len(plan) < n_regions:
        if rng.random() < 0.5:
            plan.append("h_tie" if rng.random() < 0.3 else "h_strict")
        else:
            plan.append("v_strict")

    if "h_tie" not in plan and rng.random() < 0.35:
        horiz = [i for i, branch in enumerate(plan) if branch == "h_strict"]
        if horiz:
            plan[rng.choice(horiz)] = "h_tie"

    rng.shuffle(plan)
    return plan


def _sample_geometry(branch: str, h: int, w: int, rng: random.Random):
    max_h = min(MAX_INTERIOR, h - 4)
    max_w = min(MAX_INTERIOR, w - 4)
    if max_h < MIN_INTERIOR or max_w < MIN_INTERIOR:
        return None

    for _ in range(40):
        if branch == "v_strict":
            inner_h = rng.randint(max(5, MIN_INTERIOR), max_h)
            inner_w = rng.randint(MIN_INTERIOR, min(max_w, 8))
        elif branch == "h_strict":
            inner_h = rng.randint(MIN_INTERIOR, min(max_h, 8))
            inner_w = rng.randint(max(5, MIN_INTERIOR), max_w)
        else:
            inner_h = rng.randint(MIN_INTERIOR, max_h)
            inner_w = rng.randint(MIN_INTERIOR, max_w)

        candidates = _seed_candidates(inner_h, inner_w, branch)
        if not candidates:
            continue

        spacious = [
            pos
            for pos in candidates
            if _min_seed_distance(pos[0], pos[1], inner_h, inner_w) >= 2
        ]
        if spacious and rng.random() < 0.6:
            candidates = spacious

        return inner_h, inner_w, candidates

    return None


def _can_place(used: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(used[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_used(used: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_region(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    local_seed: tuple[int, int],
    color: int,
) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY

    seed_r = top + 1 + local_seed[0]
    seed_c = left + 1 + local_seed[1]
    grid[seed_r, seed_c] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    grid = np.zeros((h, w), dtype=int)
    used = np.zeros((h, w), dtype=bool)

    max_regions = min(MAX_REGIONS, max(2, (h * w) // 90))
    n_regions = rng.randint(2, max_regions)
    branches = _sample_branch_plan(n_regions, rng)

    palette = list(SEED_COLORS)
    rng.shuffle(palette)
    palette = palette[: rng.randint(2, min(5, len(palette)))]

    for branch in branches:
        placed = False
        geom = _sample_geometry(branch, h, w, rng)
        if geom is None:
            return None
        inner_h, inner_w, candidates = geom
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if h - outer_h < 2 or w - outer_w < 2:
            return None

        for _ in range(PLACE_RETRIES):
            top = rng.randint(1, h - outer_h - 1)
            left = rng.randint(1, w - outer_w - 1)
            bottom = top + outer_h - 1
            right = left + outer_w - 1
            if not _can_place(used, top, left, bottom, right, pad=1):
                continue

            local_seed = rng.choice(candidates)
            color = rng.choice(palette)
            _draw_region(grid, top, left, inner_h, inner_w, local_seed, color)
            _mark_used(used, top, left, bottom, right, pad=1)
            placed = True
            break

        if not placed:
            return None

    regions = _extract_regions(grid)
    if len(regions) != n_regions:
        return None

    orientations = {_region_orientation(region) for region in regions}
    if n_regions >= 2 and orientations != {"horizontal", "vertical"}:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
