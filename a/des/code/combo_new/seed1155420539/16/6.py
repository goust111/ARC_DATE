from __future__ import annotations

from collections import defaultdict

import numpy as np


BG_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SCENARIOS = ("open", "collision", "merge", "mixed")
MAX_ATTEMPTS = 500

RNG = np.random.default_rng()
GEN_SCENARIO_INDEX = int(RNG.integers(len(SCENARIOS)))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _corner_positions(h: int, w: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]


def _priority_order(grid: np.ndarray, bg: int) -> list[int]:
    h, w = grid.shape
    order: list[int] = []
    seen: set[int] = set()
    for r, c in _corner_positions(h, w):
        color = int(grid[r, c])
        if color == bg or color in seen:
            continue
        seen.add(color)
        order.append(color)

    for r, c in np.argwhere(grid != bg):
        rr, cc = int(r), int(c)
        if (rr, cc) in set(_corner_positions(h, w)):
            continue
        color = int(grid[rr, cc])
        if color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _extract_seeds(grid: np.ndarray, bg: int) -> dict[int, list[tuple[int, int]]]:
    h, w = grid.shape
    corner_set = set(_corner_positions(h, w))
    seeds: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(grid != bg):
        rr, cc = int(r), int(c)
        if (rr, cc) in corner_set:
            continue
        seeds[int(grid[rr, cc])].append((rr, cc))
    return dict(seeds)


def _simulate_growth(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    order = _priority_order(g, bg)
    seeds = _extract_seeds(g, bg)

    out = g.copy()
    meta = {"collision": False, "merge": False}

    for color in order:
        positions = seeds.get(color, [])
        arm_len = len(positions)
        if arm_len == 0:
            continue
        for r, c in positions:
            for dr, dc in DIRS:
                for step in range(1, arm_len + 1):
                    rr = r + dr * step
                    cc = c + dc * step
                    if not (0 <= rr < h and 0 <= cc < w):
                        break
                    cell = int(out[rr, cc])
                    if cell == bg:
                        out[rr, cc] = color
                        continue
                    if cell == color:
                        meta["merge"] = True
                        continue
                    meta["collision"] = True
                    break

    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into a priority-resolved cross whose arm length equals the global seed count of that color.

    Input structure:
    - The background is grey.
    - Four single-pixel corner markers encode the color priority from highest to lowest in clockwise order starting at the top-left corner.
    - Every other non-background pixel is a single-cell colored seed, and several seeds may share the same color.

    Rule:
    - Count how many seeds use each color.
    - For each seed, extend branches of that seed's color upward, downward, leftward, and rightward by exactly that global count, stopping at the grid boundary.
    - When a branch reaches a cell already claimed by a different color, the higher-priority color stays and the lower-priority branch stops just before that cell.
    - When branches of the same color meet, they merge naturally and continue sharing the same cross structure.

    Key insight:
    - The branch length is determined by a global color frequency, while branch conflicts are resolved by a separate clockwise legend in the corners, so the solver must infer both global counting and external priority before drawing locally.
    """
    return _simulate_growth(grid)


# --------------- generator ---------------


def _seed_ok(
    pos: tuple[int, int],
    h: int,
    w: int,
    existing: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    r, c = pos
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if pos in reserved:
        return False
    for rr, cc in existing:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    for rr, cc in _corner_positions(h, w):
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _sample_seed_cell(h: int, w: int) -> tuple[int, int]:
    if RNG.random() < 0.35:
        side = int(RNG.integers(4))
        if side == 0:
            return 1, int(RNG.integers(1, w - 1))
        if side == 1:
            return h - 2, int(RNG.integers(1, w - 1))
        if side == 2:
            return int(RNG.integers(1, h - 1)), 1
        return int(RNG.integers(1, h - 1)), w - 2
    return int(RNG.integers(1, h - 1)), int(RNG.integers(1, w - 1))


def _choose_counts(active_colors: list[int], min_counts: dict[int, int]) -> dict[int, int]:
    counts = {
        color: int(RNG.integers(min_counts.get(color, 1), 5))
        for color in active_colors
    }
    while sum(counts.values()) > 8:
        reducible = [c for c in active_colors if counts[c] > min_counts.get(c, 1)]
        if not reducible:
            break
        color = int(RNG.choice(reducible))
        counts[color] -= 1
    return counts


def _place_merge_anchor(
    h: int,
    w: int,
    arm_len: int,
    existing: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(120):
        axis = str(RNG.choice(["row", "col"]))
        d1 = int(RNG.integers(1, arm_len + 1))
        d2 = int(RNG.integers(1, arm_len + 1))
        if axis == "row":
            r = int(RNG.integers(1, h - 1))
            c0 = 1 + d1
            c1 = w - 2 - d2
            if c0 > c1:
                continue
            c = int(RNG.integers(c0, c1 + 1))
            seeds = [(r, c - d1), (r, c + d2)]
            center = (r, c)
        else:
            c = int(RNG.integers(1, w - 1))
            r0 = 1 + d1
            r1 = h - 2 - d2
            if r0 > r1:
                continue
            r = int(RNG.integers(r0, r1 + 1))
            seeds = [(r - d1, c), (r + d2, c)]
            center = (r, c)

        if center in reserved:
            continue
        if any(not _seed_ok(seed, h, w, existing + seeds[:idx], reserved) for idx, seed in enumerate(seeds)):
            continue
        return seeds, center
    return None


def _place_collision_anchor(
    h: int,
    w: int,
    len_a: int,
    len_b: int,
    existing: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    for _ in range(180):
        dx = int(RNG.integers(1, len_a + 1))
        dy = int(RNG.integers(1, len_b + 1))
        if max(dx, dy) < 2:
            continue

        pr = int(RNG.integers(1, h - 1))
        pc = int(RNG.integers(1, w - 1))
        sx = int(RNG.choice([-1, 1]))
        sy = int(RNG.choice([-1, 1]))

        seed_a = (pr, pc - sx * dx)
        seed_b = (pr - sy * dy, pc)
        collision = (pr, pc)

        if collision in reserved or seed_a == seed_b:
            continue
        if not _seed_ok(seed_a, h, w, existing, reserved):
            continue
        if not _seed_ok(seed_b, h, w, existing + [seed_a], reserved):
            continue
        if max(abs(seed_a[0] - seed_b[0]), abs(seed_a[1] - seed_b[1])) <= 1:
            continue
        return seed_a, seed_b, collision
    return None


def _build_grid(
    h: int,
    w: int,
    priority_colors: list[int],
    seeds: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((h, w), BG_COLOR, dtype=int)
    for (r, c), color in zip(_corner_positions(h, w), priority_colors):
        grid[r, c] = int(color)
    for color, pts in seeds.items():
        for r, c in pts:
            grid[r, c] = int(color)
    return grid


def _paths_clear(
    h: int,
    w: int,
    priority_colors: list[int],
    seeds: dict[int, list[tuple[int, int]]],
) -> bool:
    occupied: dict[tuple[int, int], int] = {}
    for (r, c), color in zip(_corner_positions(h, w), priority_colors):
        occupied[(r, c)] = int(color)
    for color, pts in seeds.items():
        for pos in pts:
            occupied[pos] = int(color)

    for color, pts in seeds.items():
        arm_len = len(pts)
        for r, c in pts:
            for dr, dc in DIRS:
                for step in range(1, arm_len + 1):
                    rr = r + dr * step
                    cc = c + dc * step
                    if not (0 <= rr < h and 0 <= cc < w):
                        break
                    other = occupied.get((rr, cc))
                    if other is None or other == color:
                        continue
                    return False
    return True


def _scenario_ok(meta: dict[str, bool], scenario: str) -> bool:
    if scenario == "open":
        return not meta["collision"] and not meta["merge"]
    if scenario == "collision":
        return meta["collision"] and not meta["merge"]
    if scenario == "merge":
        return meta["merge"] and not meta["collision"]
    return meta["collision"] and meta["merge"]


def _gen_attempt(scenario: str) -> np.ndarray | None:
    h = int(RNG.integers(10, 21))
    w = int(RNG.integers(10, 21))

    priority_colors = list(map(int, RNG.choice(PALETTE, size=4, replace=False)))
    min_active = 3 if scenario == "mixed" else 2
    num_active = int(RNG.integers(min_active, 5))
    active_colors = list(map(int, RNG.choice(priority_colors, size=num_active, replace=False)))

    merge_color: int | None = None
    collision_pair: tuple[int, int] | None = None
    min_counts: dict[int, int] = {}

    if scenario in {"merge", "mixed"}:
        merge_color = int(RNG.choice(active_colors))
        min_counts[merge_color] = 2

    if scenario in {"collision", "mixed"}:
        pool = [c for c in active_colors if c != merge_color] if scenario == "mixed" else active_colors
        a, b = map(int, RNG.choice(pool, size=2, replace=False))
        collision_pair = (a, b)
        driver = int(RNG.choice([a, b]))
        min_counts[driver] = max(min_counts.get(driver, 1), 2)

    counts = _choose_counts(active_colors, min_counts)
    seeds: dict[int, list[tuple[int, int]]] = {color: [] for color in active_colors}
    reserved: set[tuple[int, int]] = set()

    if merge_color is not None:
        merge_anchor = _place_merge_anchor(h, w, counts[merge_color], [], reserved)
        if merge_anchor is None:
            return None
        merge_seeds, merge_center = merge_anchor
        seeds[merge_color].extend(merge_seeds)
        reserved.add(merge_center)

    if collision_pair is not None:
        existing = [pos for pts in seeds.values() for pos in pts]
        collision_anchor = _place_collision_anchor(
            h,
            w,
            counts[collision_pair[0]],
            counts[collision_pair[1]],
            existing,
            reserved,
        )
        if collision_anchor is None:
            return None
        seed_a, seed_b, collision_cell = collision_anchor
        seeds[collision_pair[0]].append(seed_a)
        seeds[collision_pair[1]].append(seed_b)
        reserved.add(collision_cell)

    for color in sorted(active_colors, key=lambda c: counts[c] - len(seeds[c]), reverse=True):
        while len(seeds[color]) < counts[color]:
            existing = [pos for pts in seeds.values() for pos in pts]
            placed = False
            for _ in range(120):
                pos = _sample_seed_cell(h, w)
                if _seed_ok(pos, h, w, existing, reserved):
                    seeds[color].append(pos)
                    placed = True
                    break
            if not placed:
                return None

    if not _paths_clear(h, w, priority_colors, seeds):
        return None

    grid = _build_grid(h, w, priority_colors, seeds)
    out, meta = _simulate_growth(grid, return_meta=True)
    if not _scenario_ok(meta, scenario):
        return None
    if np.array_equal(grid, out):
        return None
    if np.count_nonzero(out != BG_COLOR) > (h * w) // 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    global GEN_SCENARIO_INDEX
    scenario = SCENARIOS[GEN_SCENARIO_INDEX % len(SCENARIOS)]
    GEN_SCENARIO_INDEX += 1
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
