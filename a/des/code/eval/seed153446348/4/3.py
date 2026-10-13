from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREEN = 3
SEED_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PERP = {
    (-1, 0): ((0, -1), (0, 1)),
    (1, 0): ((0, -1), (0, 1)),
    (0, -1): ((-1, 0), (1, 0)),
    (0, 1): ((-1, 0), (1, 0)),
}
OPPOSITE = {
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
    (0, -1): (0, 1),
    (0, 1): (0, -1),
}

GRID_MIN = 12
GRID_MAX = 20
GEN_RETRIES = 400

EXTRA_COUNT_PATTERNS = (
    (1,),
    (2,),
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 1, 1),
)


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _distance_to_edge(pos: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = pos
    dr, dc = direction
    if dr == -1:
        return int(r)
    if dr == 1:
        return int(h - 1 - r)
    if dc == -1:
        return int(c)
    return int(w - 1 - c)


def _ray_cells(pos: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = direction
    return [(int(r + dr * step), int(c + dc * step)) for step in range(1, steps + 1)]


def _footprint(seed: dict) -> set[tuple[int, int]]:
    r, c = seed["pos"]
    cells = {(int(r), int(c))}
    for dr, dc in seed["dirs"]:
        cells.add((int(r + dr), int(c + dc)))
    return cells


def _occupied_cells(seeds: list[dict]) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for seed in seeds:
        occupied.update(_footprint(seed))
    return occupied


def _layout_valid(seeds: list[dict], h: int, w: int) -> bool:
    seed_positions: dict[tuple[int, int], int] = {}
    green_owners: dict[tuple[int, int], int] = {}

    for idx, seed in enumerate(seeds):
        r, c = map(int, seed["pos"])
        if not _inside(h, w, r, c):
            return False
        if (r, c) in seed_positions:
            return False
        seed_positions[(r, c)] = int(idx)

    for idx, seed in enumerate(seeds):
        dirs = tuple(dict.fromkeys(seed["dirs"]))
        if not dirs:
            return False
        if len(dirs) != len(seed["dirs"]):
            return False
        r, c = map(int, seed["pos"])
        for dr, dc in dirs:
            gr, gc = int(r + dr), int(c + dc)
            if not _inside(h, w, gr, gc):
                return False
            if (gr, gc) in seed_positions or (gr, gc) in green_owners:
                return False
            green_owners[(gr, gc)] = int(idx)

    for idx, seed in enumerate(seeds):
        r, c = map(int, seed["pos"])
        own_greens = {(int(r + dr), int(c + dc)) for dr, dc in seed["dirs"]}
        for dr, dc in DIRS4:
            nr, nc = int(r + dr), int(c + dc)
            if not _inside(h, w, nr, nc):
                continue
            owner = green_owners.get((nr, nc))
            if (nr, nc) in own_greens:
                if owner != idx:
                    return False
            elif owner is not None:
                return False

    for green_cell, owner in green_owners.items():
        touches = 0
        gr, gc = green_cell
        for dr, dc in DIRS4:
            neighbor = (int(gr + dr), int(gc + dc))
            if neighbor in seed_positions:
                touches += 1
        if touches != 1:
            return False

    return True


def _render_input(h: int, w: int, seeds: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for seed in seeds:
        r, c = map(int, seed["pos"])
        grid[r, c] = int(seed["color"])
    for seed in seeds:
        r, c = map(int, seed["pos"])
        for dr, dc in seed["dirs"]:
            grid[int(r + dr), int(c + dc)] = GREEN
    return grid


def _extract_seeds(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seeds: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, GREEN):
                continue
            dirs = []
            for dr, dc in DIRS4:
                nr, nc = int(r + dr), int(c + dc)
                if _inside(h, w, nr, nc) and int(g[nr, nc]) == GREEN:
                    dirs.append((int(dr), int(dc)))
            if dirs:
                seeds.append({"color": color, "pos": (int(r), int(c)), "dirs": tuple(dirs)})
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color seeds, remove the green direction markers, and grow each
    seed along every marked cardinal direction by that color's global seed count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    seeds = _extract_seeds(g)
    counts = Counter(int(seed["color"]) for seed in seeds)

    out = g.copy()
    out[out == GREEN] = BG
    base = out.copy()

    ordered = sorted(seeds, key=lambda seed: (seed["pos"][0], seed["pos"][1], seed["color"]))
    for seed in ordered:
        color = int(seed["color"])
        r, c = map(int, seed["pos"])
        steps = int(counts[color])
        for dr, dc in seed["dirs"]:
            rr, cc = int(r), int(c)
            for _ in range(steps):
                rr += int(dr)
                cc += int(dc)
                if not _inside(h, w, rr, cc):
                    break
                if int(base[rr, cc]) != BG:
                    break
                out[rr, cc] = int(color)

    return out


def _try_add_seed(
    seeds: list[dict],
    h: int,
    w: int,
    color: int,
    pos: tuple[int, int],
    dirs: tuple[tuple[int, int], ...],
    forbidden: set[tuple[int, int]],
) -> list[dict] | None:
    new_seed = {"color": int(color), "pos": (int(pos[0]), int(pos[1])), "dirs": tuple(dirs)}
    if any(cell in forbidden for cell in _footprint(new_seed)):
        return None
    candidate = seeds + [new_seed]
    if not _layout_valid(candidate, h, w):
        return None
    return candidate


def _reserve_path(
    forbidden: set[tuple[int, int]],
    pos: tuple[int, int],
    direction: tuple[int, int],
    clear_steps: int,
) -> None:
    for cell in _ray_cells(pos, direction, clear_steps)[1:]:
        forbidden.add(cell)


def _path_is_clear(
    pos: tuple[int, int],
    direction: tuple[int, int],
    clear_steps: int,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    return all(cell not in occupied and cell not in forbidden for cell in _ray_cells(pos, direction, clear_steps))


def _place_edge_multidir(
    seeds: list[dict],
    h: int,
    w: int,
    color: int,
    count: int,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[list[dict], tuple[int, int], dict[tuple[int, int], int]] | None:
    occupied = _occupied_cells(seeds)
    edge_dirs = list(DIRS4)
    rng.shuffle(edge_dirs)
    for edge_dir in edge_dirs:
        open_dirs = list(PERP[edge_dir])
        rng.shuffle(open_dirs)
        for open_dir in open_dirs:
            candidates = []
            for r in range(h):
                for c in range(w):
                    pos = (int(r), int(c))
                    edge_dist = _distance_to_edge(pos, edge_dir, h, w)
                    open_dist = _distance_to_edge(pos, open_dir, h, w)
                    if not (1 <= edge_dist < count):
                        continue
                    if open_dist < count:
                        continue
                    if not _path_is_clear(pos, edge_dir, edge_dist, occupied, forbidden):
                        continue
                    if not _path_is_clear(pos, open_dir, count, occupied, forbidden):
                        continue
                    candidate = _try_add_seed(seeds, h, w, color, pos, (edge_dir, open_dir), forbidden)
                    if candidate is None:
                        continue
                    candidates.append((candidate, pos, {edge_dir: edge_dist, open_dir: count}))
            if candidates:
                return rng.choice(candidates)
    return None


def _place_block_pair(
    seeds: list[dict],
    h: int,
    w: int,
    color: int,
    count: int,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[list[dict], tuple[int, int], dict[tuple[int, int], int], tuple[int, int], dict[tuple[int, int], int]] | None:
    occupied = _occupied_cells(seeds)
    shoot_dirs = list(DIRS4)
    rng.shuffle(shoot_dirs)
    distances = list(range(3, count + 1))
    rng.shuffle(distances)

    for shoot_dir in shoot_dirs:
        for distance in distances:
            candidates = []
            for r in range(h):
                for c in range(w):
                    shooter_pos = (int(r), int(c))
                    blocker_pos = (int(r + shoot_dir[0] * distance), int(c + shoot_dir[1] * distance))
                    if not _inside(h, w, blocker_pos[0], blocker_pos[1]):
                        continue
                    if blocker_pos in occupied or blocker_pos in forbidden:
                        continue
                    if not _path_is_clear(shooter_pos, shoot_dir, distance - 1, occupied, forbidden):
                        continue
                    with_shooter = _try_add_seed(seeds, h, w, color, shooter_pos, (shoot_dir,), forbidden)
                    if with_shooter is None:
                        continue
                    blocker_dirs = [direction for direction in DIRS4 if direction != OPPOSITE[shoot_dir]]
                    rng.shuffle(blocker_dirs)
                    for blocker_dir in blocker_dirs:
                        candidate = _try_add_seed(
                            with_shooter,
                            h,
                            w,
                            color,
                            blocker_pos,
                            (blocker_dir,),
                            forbidden,
                        )
                        if candidate is None:
                            continue
                        candidates.append(
                            (
                                candidate,
                                shooter_pos,
                                {shoot_dir: distance - 1},
                                blocker_pos,
                                {blocker_dir: min(count, _distance_to_edge(blocker_pos, blocker_dir, h, w))},
                            )
                        )
            if candidates:
                return rng.choice(candidates)
    return None


def _dirs_for_mode(
    pos: tuple[int, int],
    count: int,
    mode: str,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], int]]:
    valid = []
    for direction in DIRS4:
        edge_dist = _distance_to_edge(pos, direction, h, w)
        if mode == "open":
            if edge_dist < count:
                continue
            clear_steps = int(count)
        elif mode == "edge":
            if not (1 <= edge_dist < count):
                continue
            clear_steps = int(edge_dist)
        else:
            if edge_dist < 1:
                continue
            clear_steps = int(min(count, edge_dist))
        if not _path_is_clear(pos, direction, clear_steps, occupied, forbidden):
            continue
        valid.append((direction, clear_steps))
    return valid


def _place_random_seed(
    seeds: list[dict],
    h: int,
    w: int,
    color: int,
    count: int,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
    allow_multidir: bool,
) -> tuple[list[dict], tuple[int, int], dict[tuple[int, int], int]] | None:
    occupied = _occupied_cells(seeds)
    positions = [(int(r), int(c)) for r in range(h) for c in range(w)]
    rng.shuffle(positions)

    modes = ["open"]
    if count >= 2:
        modes.append("edge")
    rng.shuffle(modes)

    for mode in modes:
        for pos in positions:
            dirs_with_steps = _dirs_for_mode(pos, count, mode, h, w, occupied, forbidden)
            if not dirs_with_steps:
                continue
            rng.shuffle(dirs_with_steps)
            if allow_multidir and len(dirs_with_steps) >= 2 and rng.random() < 0.35:
                chosen = dirs_with_steps[:2]
                candidate = _try_add_seed(
                    seeds,
                    h,
                    w,
                    color,
                    pos,
                    tuple(direction for direction, _steps in chosen),
                    forbidden,
                )
                if candidate is not None:
                    return candidate, pos, {direction: steps for direction, steps in chosen}
            direction, steps = dirs_with_steps[0]
            candidate = _try_add_seed(seeds, h, w, color, pos, (direction,), forbidden)
            if candidate is not None:
                return candidate, pos, {direction: steps}
    return None


def _classify_rays(grid: np.ndarray) -> tuple[list[str], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seeds = _extract_seeds(g)
    counts = Counter(int(seed["color"]) for seed in seeds)
    base = g.copy()
    base[base == GREEN] = BG

    branches: list[str] = []
    painted: list[tuple[int, int]] = []
    seen_targets: set[tuple[int, int]] = set()

    for seed in seeds:
        color = int(seed["color"])
        steps = int(counts[color])
        r, c = map(int, seed["pos"])
        for dr, dc in seed["dirs"]:
            rr, cc = int(r), int(c)
            branch = "open"
            ray_cells: list[tuple[int, int]] = []
            for _ in range(steps):
                rr += int(dr)
                cc += int(dc)
                if not _inside(h, w, rr, cc):
                    branch = "edge"
                    break
                if int(base[rr, cc]) != BG:
                    branch = "seed"
                    break
                ray_cells.append((int(rr), int(cc)))
            branches.append(branch)
            for cell in ray_cells:
                if cell in seen_targets:
                    return [], []
                seen_targets.add(cell)
                painted.append(cell)

    return branches, painted


def _scene_ok(grid: np.ndarray, expected_counts: dict[int, int]) -> bool:
    if grid.shape[0] < GRID_MIN or grid.shape[1] < GRID_MIN:
        return False
    if int(np.count_nonzero(grid == GREEN)) == 0:
        return False

    seeds = _extract_seeds(grid)
    if not seeds:
        return False
    parsed_counts = Counter(int(seed["color"]) for seed in seeds)
    if dict(parsed_counts) != dict(expected_counts):
        return False
    if max(parsed_counts.values()) < 3:
        return False
    if max(len(seed["dirs"]) for seed in seeds) < 2:
        return False

    branches, painted = _classify_rays(grid)
    if not branches or not painted:
        return False
    if not {"open", "edge", "seed"}.issubset(set(branches)):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)

    focus_color = int(rng.choice(SEED_COLORS))
    focus_count = int(rng.choice((3, 4)))
    extra_pattern = tuple(int(x) for x in rng.choice(EXTRA_COUNT_PATTERNS))
    extra_colors = rng.sample([color for color in SEED_COLORS if color != focus_color], len(extra_pattern))

    counts = {focus_color: focus_count}
    for color, count in zip(extra_colors, extra_pattern):
        counts[int(color)] = int(count)

    seeds: list[dict] = []
    forbidden: set[tuple[int, int]] = set()

    edge_seed = _place_edge_multidir(seeds, h, w, focus_color, focus_count, forbidden, rng)
    if edge_seed is None:
        return None
    seeds, pos, reservations = edge_seed
    for direction, steps in reservations.items():
        _reserve_path(forbidden, pos, direction, steps)

    block_pair = _place_block_pair(seeds, h, w, focus_color, focus_count, forbidden, rng)
    if block_pair is None:
        return None
    seeds, shooter_pos, shooter_res, blocker_pos, blocker_res = block_pair
    for direction, steps in shooter_res.items():
        _reserve_path(forbidden, shooter_pos, direction, steps)
    for direction, steps in blocker_res.items():
        _reserve_path(forbidden, blocker_pos, direction, steps)

    if focus_count == 4:
        extra_focus = _place_random_seed(
            seeds,
            h,
            w,
            focus_color,
            focus_count,
            forbidden,
            rng,
            allow_multidir=True,
        )
        if extra_focus is None:
            return None
        seeds, pos, reservations = extra_focus
        for direction, steps in reservations.items():
            _reserve_path(forbidden, pos, direction, steps)

    extra_seed_colors: list[int] = []
    for color, count in counts.items():
        if color == focus_color:
            continue
        extra_seed_colors.extend([int(color)] * int(count))
    rng.shuffle(extra_seed_colors)

    for idx, color in enumerate(extra_seed_colors):
        placed = _place_random_seed(
            seeds,
            h,
            w,
            color,
            counts[color],
            forbidden,
            rng,
            allow_multidir=bool(idx % 2 == 0 or rng.random() < 0.25),
        )
        if placed is None:
            return None
        seeds, pos, reservations = placed
        for direction, steps in reservations.items():
            _reserve_path(forbidden, pos, direction, steps)

    grid = _render_input(h, w, seeds)
    if not _scene_ok(grid, counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated attempts")
