from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import itertools
import random


def _seed_color(g: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        if coords.size and np.all(coords[:, 1] == 0):
            candidates.append(int(color))

    assert len(candidates) == 1, "expected one non-background color used only as left-edge beam seeds"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each left-edge beam marker rightward, letting obstacle pixels create side wakes and split branches.

    Input structure
    - The grid has a dominant background, one seed color whose pixels all lie in the leftmost column, and one obstacle color.
    - Seed pixels are horizontal emitters; all other non-background pixels are fixed one-pixel obstacles.
    - Obstacles may appear on a beam row or immediately diagonally in front of another obstacle.

    Rule
    - From every seed, paint background cells to the right with the seed color.
    - When a beam hits an obstacle, keep the obstacle unchanged and paint the available three-cell side wakes above and below it.
    - A side wake is blocked if its left entry cell is an unactivated obstacle; obstacles reached inside wakes paint only their open side wakes and the cell immediately to their right.
    - If a hit obstacle has obstacles at both forward diagonals, the original beam stops and full beams continue on those two adjacent rows without painting back through the parent obstacle.

    Key insight
    - The output is a directed ray simulation: obstacle pixels are fixed blockers that locally thicken the beam, and paired forward-diagonal blockers are the only places where the beam splits into new horizontal paths.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    beam = _seed_color(g, bg)
    obstacle_colors = colors_present(g, exclude={bg, beam})
    assert len(obstacle_colors) == 1, "expected one fixed obstacle color"

    H, W = g.shape
    out = g.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < H and 0 <= c < W

    def is_obstacle(r: int, c: int) -> bool:
        return in_bounds(r, c) and int(g[r, c]) not in (bg, beam)

    def paint(r: int, c: int) -> None:
        if in_bounds(r, c) and int(g[r, c]) == bg:
            out[r, c] = beam

    active_obstacles: set[tuple[int, int]] = set()
    local_hits: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    full_hits: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    full_beams: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()

    def hit_obstacle(
        r: int,
        c: int,
        *,
        full: bool,
        blocked_entries: frozenset[tuple[int, int]] = frozenset(),
    ) -> list[tuple[int, int, frozenset[tuple[int, int]]]]:
        assert is_obstacle(r, c), "beam hit must be an obstacle"

        if full:
            full_key = (r, c, tuple(sorted(blocked_entries)))
            if full_key in full_hits:
                return []
            full_hits.add(full_key)
        else:
            local_key = (r, c, tuple(sorted(blocked_entries)))
            if local_key in local_hits:
                return []
            local_hits.add(local_key)

        active_obstacles.add((r, c))
        side_obstacles: list[tuple[int, int]] = []
        side_open = False

        for dr in (-1, 1):
            rr = r + dr
            if not 0 <= rr < H:
                continue

            entry = (rr, c - 1)
            if entry in blocked_entries:
                continue
            if c - 1 >= 0 and is_obstacle(*entry) and entry not in active_obstacles:
                continue

            side_open = True
            for cc in range(max(0, c - 1), min(W, c + 2)):
                paint(rr, cc)
                if is_obstacle(rr, cc) and (rr, cc) != (r, c):
                    side_obstacles.append((rr, cc))

        if not full and side_open:
            paint(r, c + 1)

        if full:
            up = is_obstacle(r - 1, c + 1)
            down = is_obstacle(r + 1, c + 1)
            if up and down:
                parent = frozenset({(r, c)})
                return [(r - 1, c, parent), (r + 1, c, parent)]

        for rr, cc in side_obstacles:
            hit_obstacle(rr, cc, full=False, blocked_entries=frozenset({(r, c)}))

        return []

    def run_full_beam(
        r: int,
        start_c: int,
        blocked_entries: frozenset[tuple[int, int]] = frozenset(),
    ) -> None:
        assert 0 <= r < H, "beam row must be inside the grid"
        beam_key = (r, start_c, tuple(sorted(blocked_entries)))
        if beam_key in full_beams:
            return
        full_beams.add(beam_key)

        c = start_c + 1
        while c < W:
            if int(g[r, c]) == bg:
                out[r, c] = beam
                c += 1
                continue
            if int(g[r, c]) == beam:
                c += 1
                continue

            branches = hit_obstacle(r, c, full=True, blocked_entries=blocked_entries)
            if branches:
                for rr, branch_start_c, branch_blocked in branches:
                    run_full_beam(rr, branch_start_c, branch_blocked)
                return
            c += 1

    seeds = np.argwhere(g == beam)
    assert seeds.size > 0, "expected at least one beam seed"
    for r_raw, c_raw in seeds:
        r = int(r_raw)
        c = int(c_raw)
        assert c == 0, "beam seeds must be on the left edge"
        run_full_beam(r, c)

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_SIZE_MIN, GRID_SIZE_MAX = 6, 26
BACKGROUND_COLOR, SEED_COLOR = 0, 1
OBSTACLE_COLORS = (5, 8)

# Seed layout.
SEED_COUNT_MAX = 3
SEED_ROW_SPACING = 4
PREFERRED_SEED_EDGE_MARGIN = 1

# Obstacle density and count bounds.
OBSTACLE_DENSITY_MIN, OBSTACLE_DENSITY_MAX = 0.05, 0.16
OBSTACLE_COUNT_MIN, OBSTACLE_COUNT_MAX = 2, 80

# Branch-triangle placement.
BRANCH_PROBABILITY = 0.3
BRANCH_EXTRA_OBSTACLES = 2
BRANCH_PARENT_MIN_COL = 2

# Required obstacle placement and retries.
REQUIRED_ROW_OBSTACLE_MIN_COL = 2
GENERATION_RETRY_LIMIT = 300
CARDINAL_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_seed_rows(n: int, k: int) -> tuple[int, ...]:
    for pool in (range(PREFERRED_SEED_EDGE_MARGIN, n - PREFERRED_SEED_EDGE_MARGIN), range(n)):
        choices = [
            rows
            for rows in itertools.combinations(pool, k)
            if all(b - a >= SEED_ROW_SPACING for a, b in zip(rows, rows[1:]))
        ]
        if choices:
            return random.choice(choices)
    raise ValueError("no feasible seed rows")


def _gen_group_is_open(g, coords, obstacle, allowed_branches) -> bool:
    h, w = g.shape
    coord_set = set(coords)
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or g[r, c] != BACKGROUND_COLOR:
            return False
        for dr, dc in CARDINAL_DELTAS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and g[rr, cc] != BACKGROUND_COLOR and (rr, cc) not in coord_set:
                return False

    def occupied(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w and (g[r, c] == obstacle or (r, c) in coord_set)

    possible_parents = set()
    for r, c in coords:
        possible_parents.update(((r, c), (r + 1, c - 1), (r - 1, c - 1)))

    for r, c in possible_parents:
        if not (1 <= r < h - 1 and 0 <= c < w - 1):
            continue
        if occupied(r, c) and occupied(r - 1, c + 1) and occupied(r + 1, c + 1):
            if (r, c) not in allowed_branches:
                return False
    return True


def _gen_place_one(g, coords, obstacle) -> None:
    for r, c in coords:
        g[r, c] = obstacle


def _gen_open_singletons(g, obstacle, allowed_branches, *, row=None, min_col=0):
    h, w = g.shape
    rows = range(h) if row is None else (row,)
    return [
        (r, c)
        for r in rows
        for c in range(min_col, w)
        if _gen_group_is_open(g, ((r, c),), obstacle, allowed_branches)
    ]


def _gen_branch_options(g, seed_rows, obstacle):
    h, w = g.shape
    options = []
    for r in seed_rows:
        if not 1 <= r < h - 1:
            continue
        for c in range(BRANCH_PARENT_MIN_COL, w - 1):
            parent = (r, c)
            coords = (parent, (r - 1, c + 1), (r + 1, c + 1))
            if _gen_group_is_open(g, coords, obstacle, {parent}):
                options.append((parent, coords))
    random.shuffle(options)
    return options


def _gen_try_place_branch(g, seed_rows, obstacle, target_obstacles, seed_count):
    allowed_branches = set()
    if target_obstacles >= seed_count + BRANCH_EXTRA_OBSTACLES and random.random() < BRANCH_PROBABILITY:
        branch_options = _gen_branch_options(g, seed_rows, obstacle)
        if branch_options:
            parent, coords = random.choice(branch_options)
            _gen_place_one(g, coords, obstacle)
            allowed_branches.add(parent)
    return allowed_branches


def _gen_seed_rows_have_obstacles(g, seed_rows, obstacle, allowed_branches) -> bool:
    for r in seed_rows:
        if np.any(g[r, 1:] == obstacle):
            continue
        choices = _gen_open_singletons(g, obstacle, allowed_branches, row=r, min_col=REQUIRED_ROW_OBSTACLE_MIN_COL)
        if not choices:
            return False
        _gen_place_one(g, (random.choice(choices),), obstacle)
    return True


def _gen_fill_obstacles(g, obstacle, target_obstacles, allowed_branches) -> bool:
    while int(np.count_nonzero(g == obstacle)) < target_obstacles:
        choices = _gen_open_singletons(g, obstacle, allowed_branches)
        if not choices:
            return False
        _gen_place_one(g, (random.choice(choices),), obstacle)
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    g = np.zeros((n, n), dtype=int)
    obstacle = random.choice(OBSTACLE_COLORS)

    min_obstacles = max(OBSTACLE_COUNT_MIN, int(np.ceil(OBSTACLE_DENSITY_MIN * n * n)))
    max_obstacles = min(OBSTACLE_COUNT_MAX, int(np.floor(OBSTACLE_DENSITY_MAX * n * n)))
    target_obstacles = random.randint(min_obstacles, max_obstacles)

    max_seeds = min(SEED_COUNT_MAX, (n + SEED_ROW_SPACING - 1) // SEED_ROW_SPACING, target_obstacles)
    seed_count = random.randint(1, max_seeds)
    seed_rows = _gen_seed_rows(n, seed_count)
    for r in seed_rows:
        g[r, 0] = SEED_COLOR

    allowed_branches = _gen_try_place_branch(g, seed_rows, obstacle, target_obstacles, seed_count)
    if not _gen_seed_rows_have_obstacles(g, seed_rows, obstacle, allowed_branches):
        return None
    if not _gen_fill_obstacles(g, obstacle, target_obstacles, allowed_branches):
        return None

    out = main(g)
    if np.array_equal(out, g):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
