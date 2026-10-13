from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GENERATION_ATTEMPTS = 400
LOCAL_REGION_ATTEMPTS = 140
SEED_SEARCH_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 220

_GEN_COUNTER = 0

Cell = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Cell]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[Cell]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Cell]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[Cell] = deque([(r, c)])
            comp: list[Cell] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _shape_is_connected(cells: set[Cell]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    queue: deque[Cell] = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _bbox_span(cells: set[Cell]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _manhattan_path(a: Cell, b: Cell, rng: random.Random) -> list[Cell]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    axes = [0, 1]
    rng.shuffle(axes)
    for axis in axes:
        if axis == 0:
            step = 1 if tr >= r else -1
            while r != tr:
                r += step
                path.append((r, c))
        else:
            step = 1 if tc >= c else -1
            while c != tc:
                c += step
                path.append((r, c))
    return path


def _random_shape(height: int, width: int, rng: random.Random) -> set[Cell] | None:
    cells: set[Cell] = set()
    rect_count = rng.choice((2, 2, 3))

    for idx in range(rect_count):
        min_h = max(2, height // 3)
        min_w = max(2, width // 3)
        rect_h = rng.randint(min_h, height)
        rect_w = rng.randint(min_w, width)
        top = rng.randint(0, height - rect_h)
        left = rng.randint(0, width - rect_w)
        rect = {(r, c) for r in range(top, top + rect_h) for c in range(left, left + rect_w)}

        if idx == 0 or not cells:
            cells |= rect
            continue

        attach = rng.choice(tuple(cells))
        target = (rng.randint(top, top + rect_h - 1), rng.randint(left, left + rect_w - 1))
        cells |= rect
        cells |= set(_manhattan_path(attach, target, rng))

    frontier = {
        (nr, nc)
        for r, c in cells
        for nr, nc in _neighbors4(r, c)
        if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells
    }
    extra_steps = rng.randint(0, max(1, (height * width) // 7))
    for _ in range(extra_steps):
        if not frontier:
            break
        choices = list(frontier)
        weights = []
        for cell in choices:
            support = sum((nr, nc) in cells for nr, nc in _neighbors4(*cell))
            weights.append(1 + 3 * support)
        chosen = rng.choices(choices, weights=weights, k=1)[0]
        cells.add(chosen)
        frontier = {
            (nr, nc)
            for r, c in cells
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells
        }

    carve_steps = rng.randint(0, max(1, (height + width) // 5))
    for _ in range(carve_steps):
        removable = []
        for cell in cells:
            if len(cells) <= 7:
                break
            if sum((nr, nc) in cells for nr, nc in _neighbors4(*cell)) >= 4:
                continue
            trial = cells - {cell}
            if _shape_is_connected(trial):
                removable.append(cell)
        if not removable:
            break
        cells.remove(rng.choice(removable))

    if len(cells) < max(6, (height * width) // 4):
        return None
    span_h, span_w = _bbox_span(cells)
    if span_h < min(3, height) or span_w < min(3, width):
        return None
    return cells


def _region_features(shape: set[Cell], seeds: tuple[Cell, ...]) -> dict[str, bool]:
    radius = len(seeds)
    seed_set = set(seeds)
    grey = shape - seed_set
    cover_counts: dict[Cell, int] = {}
    overlap = False
    boundary_truncated = False

    for r, c in seeds:
        for dr, dc in DIRS4:
            for step in range(1, radius + 1):
                nr = r + dr * step
                nc = c + dc * step
                if (nr, nc) not in shape:
                    boundary_truncated = True
                    break
                if (nr, nc) in seed_set:
                    break
                if (nr, nc) not in grey:
                    break
                cover_counts[(nr, nc)] = cover_counts.get((nr, nc), 0) + 1
                if cover_counts[(nr, nc)] >= 2:
                    overlap = True

    return {
        "changed": bool(cover_counts),
        "overlap": overlap,
        "boundary_truncated": boundary_truncated,
    }


def _cells_are_separated(cells: tuple[Cell, ...]) -> bool:
    for i, (r1, c1) in enumerate(cells):
        for r2, c2 in cells[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _choose_seeds(
    shape: set[Cell], seed_count: int, profile: dict[str, int | bool], rng: random.Random
) -> list[Cell] | None:
    if seed_count == 0:
        return []

    cells = list(shape)
    rng.shuffle(cells)
    if len(cells) < seed_count:
        return None

    combo_cells = cells[: min(len(cells), 32 if seed_count == 3 else 40)]
    for combo in combinations(combo_cells, seed_count):
        if not _cells_are_separated(combo):
            continue
        info = _region_features(shape, combo)
        if not info["changed"]:
            continue
        if profile.get("overlap") and not info["overlap"]:
            continue
        if profile.get("truncate") and not info["boundary_truncated"]:
            continue
        return list(combo)

    for _ in range(SEED_SEARCH_ATTEMPTS):
        rng.shuffle(cells)
        chosen: list[Cell] = []
        for cell in cells:
            if any(max(abs(cell[0] - r), abs(cell[1] - c)) <= 1 for r, c in chosen):
                continue
            chosen.append(cell)
            if len(chosen) == seed_count:
                break
        if len(chosen) != seed_count:
            continue
        info = _region_features(shape, tuple(chosen))
        if not info["changed"]:
            continue
        if profile.get("overlap") and not info["overlap"]:
            continue
        if profile.get("truncate") and not info["boundary_truncated"]:
            continue
        return chosen
    return None


def _build_local_region(profile: dict[str, int | bool], rng: random.Random) -> np.ndarray | None:
    seed_count = int(profile["seed_count"])
    min_dim = 4
    if seed_count >= 1:
        min_dim = 5
    if seed_count >= 3 or profile.get("overlap"):
        min_dim = max(min_dim, 7)
    if profile.get("truncate"):
        min_dim = max(min_dim, 6)
    max_dim = min(10, min_dim + 3)

    for _ in range(LOCAL_REGION_ATTEMPTS):
        height = rng.randint(min_dim, max_dim)
        width = rng.randint(min_dim, max_dim)
        shape = _random_shape(height, width, rng)
        if shape is None:
            continue
        seeds = _choose_seeds(shape, seed_count, profile, rng)
        if seeds is None:
            continue

        grid = np.full((height, width), BLACK, dtype=int)
        for r, c in shape:
            grid[r, c] = GREY

        if seed_count:
            color = rng.choice(SEED_COLORS)
            for r, c in seeds:
                grid[r, c] = color
        return grid
    return None


def _profile_list(gen_index: int, rng: random.Random) -> list[dict[str, int | bool]]:
    mode = gen_index % 4
    if mode == 0:
        profiles = [
            {"seed_count": 0},
            {"seed_count": 1},
            {"seed_count": 2, "overlap": True},
        ]
    elif mode == 1:
        profiles = [
            {"seed_count": 0},
            {"seed_count": 1},
            {"seed_count": 2, "truncate": True},
        ]
    elif mode == 2:
        profiles = [
            {"seed_count": 0},
            {"seed_count": 2, "overlap": True},
            {"seed_count": 3, "truncate": True},
        ]
    else:
        profiles = [
            {"seed_count": 1},
            {"seed_count": 2, "overlap": True},
            {"seed_count": 3, "truncate": True},
        ]

    extra_count = rng.randint(0, 2)
    for _ in range(extra_count):
        seed_count = rng.choice((0, 1, 2, 3))
        profile: dict[str, int | bool] = {"seed_count": seed_count}
        if seed_count >= 2 and rng.random() < 0.35:
            profile["truncate"] = True
        if seed_count >= 2 and rng.random() < 0.25:
            profile["overlap"] = True
        profiles.append(profile)

    rng.shuffle(profiles)
    return profiles


def _place_regions(regions: list[np.ndarray], height: int, width: int, rng: random.Random) -> np.ndarray | None:
    grid = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    ordered = sorted(regions, key=lambda arr: int(np.count_nonzero(arr != BLACK)), reverse=True)
    for region in ordered:
        rh, rw = region.shape
        if rh > height or rw > width:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(0, height - rh)
            left = rng.randint(0, width - rw)
            mt = max(0, top - 1)
            ml = max(0, left - 1)
            mb = min(height, top + rh + 1)
            mr = min(width, left + rw + 1)
            if reserved[mt:mb, ml:mr].any():
                continue

            submask = region != BLACK
            patch = grid[top : top + rh, left : left + rw]
            patch[submask] = region[submask]
            reserved[mt:mb, ml:mr] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _grid_is_valid(grid: np.ndarray) -> bool:
    comps = _components(grid != BLACK)
    if not comps:
        return False

    seen_seedless = False
    seen_seeded = False
    for comp in comps:
        values = [int(grid[r, c]) for r, c in comp]
        if GREY not in values:
            return False
        seeds = [(r, c) for r, c in comp if int(grid[r, c]) not in (BLACK, GREY)]
        if not seeds:
            seen_seedless = True
            continue
        seen_seeded = True
        if len({int(grid[r, c]) for r, c in seeds}) != 1:
            return False
        if not _cells_are_separated(tuple(seeds)):
            return False

    if not seen_seeded:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if seen_seedless and not np.any(out == GREY):
        return False
    return True


def _gen_attempt(gen_index: int) -> np.ndarray | None:
    rng = random.Random()
    profiles = _profile_list(gen_index, rng)
    min_dim = max(14, 8 + 4 * len(profiles))
    height = rng.randint(min_dim, 30)
    width = rng.randint(min_dim, 30)

    regions: list[np.ndarray] = []
    for profile in profiles:
        region = _build_local_region(profile, rng)
        if region is None:
            return None
        regions.append(region)

    grid = _place_regions(regions, height, width, rng)
    if grid is None:
        return None
    if not _grid_is_valid(grid):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed into a same-colored cross inside its own grey soil region, with radius set by the region's seed count.

    Input structure:
    - Black is the background color.
    - Grey cells form one or more 4-connected soil regions separated by black.
    - Some soil regions contain one or more single-pixel colored seeds; seed colors are neither black nor grey.
    - Within one soil region, all seeds share the same color, while different regions may use different seed colors.

    Rule:
    - Treat each black-separated non-black component as one soil region together with any seeds embedded in it.
    - For every soil region that contains seeds, count its seeds as N.
    - From each seed, extend up, down, left, and right for up to N steps, recoloring only the cells that were originally grey.
    - Stop a direction as soon as the next cell is not originally grey; seedless soil regions and the black background stay unchanged.

    Key insight:
    - The growth radius is a group property of the whole soil region: first group seeds by region, then count them, then use that count to drive every seed's cross-shaped expansion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for comp in _components(g != BLACK):
        seeds = [(r, c) for r, c in comp if int(g[r, c]) not in (BLACK, GREY)]
        if not seeds:
            continue

        seed_colors = {int(g[r, c]) for r, c in seeds}
        assert len(seed_colors) == 1, "seeds inside one soil region must share one color"
        radius = len(seeds)

        for r, c in seeds:
            color = int(g[r, c])
            for dr, dc in DIRS4:
                for step in range(1, radius + 1):
                    nr = r + dr * step
                    nc = c + dc * step
                    if not _in_bounds(g.shape, nr, nc):
                        break
                    if int(g[nr, nc]) != GREY:
                        break
                    out[nr, nc] = color
    return out


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(_GEN_COUNTER)
        _GEN_COUNTER += 1
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
