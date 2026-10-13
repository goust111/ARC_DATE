from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
NONZERO_COLORS = tuple(range(1, 10))
ODD_BOUNCE_LENGTHS = (3, 5, 7)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _sorted_nonzero_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
    cells = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != BG)]
    cells.sort()
    return cells


def _find_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    counts = Counter(map(int, grid.ravel()))
    seeds: list[tuple[int, int, int]] = []
    for color, count in sorted(counts.items()):
        if color == BG or count != 1:
            continue
        locs = np.argwhere(grid == int(color))
        if locs.shape[0] != 1:
            raise ValueError("singleton seed color must occupy exactly one cell")
        r, c = locs[0]
        seeds.append((int(r), int(c), int(color)))
    seeds.sort()
    if not 2 <= len(seeds) <= 3:
        raise ValueError("expected 2-3 singleton-color seeds")
    return seeds


def _wall_positions(grid: np.ndarray, seeds: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    seed_pos = {(r, c) for r, c, _ in seeds}
    return {(r, c) for r, c, _ in _sorted_nonzero_cells(grid) if (r, c) not in seed_pos}


def _simulate(
    grid: np.ndarray,
    *,
    detect_every_step: bool = False,
    bounce: bool = True,
    reverse_order: bool = False,
    return_meta: bool = False,
) -> tuple[np.ndarray, list[dict]] | np.ndarray:
    g = np.asarray(grid, dtype=int)
    seeds = _find_seeds(g)
    if reverse_order:
        seeds = list(reversed(seeds))
    seed_pos = {(r, c) for r, c, _ in seeds}
    walls = _wall_positions(g, seeds)
    out = g.copy()
    meta: list[dict] = []

    for sr, sc, start_color in seeds:
        r, c = sr, sc
        dr = 1
        color = int(start_color)
        step = 0
        bounce_count = 0
        ray = {
            "seed": (sr, sc, start_color),
            "blind_hits": [],
            "wall_hits": [],
            "trail": [],
            "post_bounce_trail": [],
            "stop": "boundary",
        }
        while True:
            nr = int(r + dr)
            nc = int(c + 1)
            if not _in_bounds(g.shape, nr, nc):
                ray["stop"] = "boundary"
                break
            step += 1
            r, c = nr, nc
            should_detect = detect_every_step or (step % 2 == 1)

            if not should_detect:
                if (r, c) in walls:
                    ray["blind_hits"].append((r, c, int(g[r, c])))
                continue

            if (r, c) in seed_pos and (r, c) != (sr, sc):
                ray["stop"] = "seed"
                break

            if (r, c) in walls:
                wall_color = int(g[r, c])
                ray["wall_hits"].append((r, c, wall_color))
                if bounce:
                    color = wall_color
                    dr = -dr
                    step = 0
                    bounce_count += 1
                    continue
                ray["stop"] = "wall"
                break

            out[r, c] = int(color)
            ray["trail"].append((r, c, int(color)))
            if bounce_count > 0:
                ray["post_bounce_trail"].append((r, c, int(color)))

        meta.append(ray)

    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace period-2 diagonal rays that bounce only on sampled wall hits and inherit wall colors.

    Input structure:
    - The grid has a black background with 2-3 isolated seed pixels and 4-8 isolated wall pixels.
    - Seed colors are singleton colors: each appears exactly once.
    - Wall colors are disjoint from the seed colors and repeat across multiple wall pixels.

    Rule:
    - From each seed, fire a ray toward down-right, advancing one cell diagonally per step.
    - Only odd-numbered steps are sampled: odd steps place trail pixels and can hit walls or other seeds; even steps are skipped entirely.
    - If an odd sampled position is a wall, keep the wall pixel unchanged, flip only the row direction, inherit that wall's color, reset the odd/even counter, and continue.
    - If an odd sampled position is another seed, stop that ray. Later rays overwrite earlier trail pixels.

    Key insight:
    - The decisive behavior comes from the sampling period: walls on even steps are invisible, so the ray can pass through them and bounce only at later odd-step wall cells, which changes both geometry and color.
    """
    return _simulate(grid)


def _pairwise_seed_spacing_ok(seeds: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(seeds):
        for r2, c2 in seeds[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) < 5:
                return False
    return True


def _can_place_seed(existing: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    r, c = candidate
    return all(abs(r - er) + abs(c - ec) >= 5 for er, ec in existing)


def _choose_seed_colors(n: int) -> list[int]:
    return random.sample(NONZERO_COLORS, n)


def _build_overlap_pair(shape: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]] | None:
    h, w = shape
    row_choices = list(range(0, h - 7))
    col_choices = list(range(0, w - 9))
    if not row_choices or not col_choices:
        return None
    r = random.choice(row_choices)
    c = random.choice(col_choices)
    seed1 = (r, c)
    seed2 = (r + 4, c + 4)
    walls = [
        (r + 2, c + 2, "even"),
        (r + 6, c + 6, "even"),
        (r + 7, c + 7, "odd"),
        (r + 5, c + 9, "even"),
    ]
    return [seed1, seed2], walls


def _future_bounce_exists(cur: tuple[int, int], dr: int, shape: tuple[int, int]) -> bool:
    r, c = cur
    for length in ODD_BOUNCE_LENGTHS:
        rr = int(r + dr * length)
        cc = int(c + length)
        if not _in_bounds(shape, rr, cc):
            continue
        if not _in_bounds(shape, rr - dr, cc + 1):
            continue
        return True
    return False


def _plan_independent_seed(
    seed: tuple[int, int],
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    extra_bounce: bool,
) -> list[tuple[int, int, str]] | None:
    sr, sc = seed
    for _ in range(120):
        walls: list[tuple[int, int, str]] = []
        used = set(forbidden)
        cur = (sr, sc)
        dr = 1
        segments = 2 if extra_bounce else 1
        ok = True

        for seg_idx in range(segments):
            candidates: list[tuple[int, int]] = []
            for length in ODD_BOUNCE_LENGTHS:
                br = int(cur[0] + dr * length)
                bc = int(cur[1] + length)
                if not _in_bounds(shape, br, bc):
                    continue
                if (br, bc) in used:
                    continue
                if seg_idx == segments - 1:
                    if not _in_bounds(shape, br - dr, bc + 1):
                        continue
                else:
                    if not _future_bounce_exists((br, bc), -dr, shape):
                        continue
                evens = []
                for even_len in range(2, length, 2):
                    er = int(cur[0] + dr * even_len)
                    ec = int(cur[1] + even_len)
                    if (er, ec) in used or not _in_bounds(shape, er, ec):
                        continue
                    evens.append(even_len)
                if evens:
                    candidates.append((length, random.choice(evens)))

            if not candidates:
                ok = False
                break

            length, even_len = random.choice(candidates)
            er = int(cur[0] + dr * even_len)
            ec = int(cur[1] + even_len)
            br = int(cur[0] + dr * length)
            bc = int(cur[1] + length)
            walls.append((er, ec, "even"))
            walls.append((br, bc, "odd"))
            used.add((er, ec))
            used.add((br, bc))
            cur = (br, bc)
            dr = -dr

        if ok:
            return walls
    return None


def _assign_wall_colors(seed_colors: list[int], wall_positions: list[tuple[int, int]]) -> dict[tuple[int, int], int] | None:
    wall_count = len(wall_positions)
    remaining = [c for c in NONZERO_COLORS if c not in set(seed_colors)]
    max_palette = min(3, wall_count // 2, len(remaining))
    if max_palette < 2:
        return None
    palette_size = random.randint(2, max_palette)
    palette = random.sample(remaining, palette_size)

    counts = [2] * palette_size
    for _ in range(wall_count - 2 * palette_size):
        counts[random.randrange(palette_size)] += 1

    colors: list[int] = []
    for color, count in zip(palette, counts):
        colors.extend([int(color)] * int(count))
    random.shuffle(colors)

    mapping = {}
    for pos, color in zip(wall_positions, colors):
        mapping[pos] = int(color)
    return mapping


def _build_grid(
    shape: tuple[int, int],
    seeds: list[tuple[int, int]],
    seed_colors: list[int],
    walls: list[tuple[int, int, str]],
) -> np.ndarray | None:
    wall_positions = [(r, c) for r, c, _ in walls]
    if len(set(wall_positions)) != len(wall_positions):
        return None
    if not 4 <= len(wall_positions) <= 8:
        return None

    wall_colors = _assign_wall_colors(seed_colors, wall_positions)
    if wall_colors is None:
        return None

    grid = np.zeros(shape, dtype=int)
    for (r, c), color in zip(seeds, seed_colors):
        grid[r, c] = int(color)
    for r, c in wall_positions:
        if grid[r, c] != BG:
            return None
        grid[r, c] = int(wall_colors[(r, c)])
    return grid


def _path_overlap_count(meta: list[dict]) -> int:
    hits: Counter[tuple[int, int]] = Counter()
    for ray in meta:
        for r, c, _ in ray["trail"]:
            hits[(r, c)] += 1
    return sum(v >= 2 for v in hits.values())


def _validate_instance(grid: np.ndarray, *, want_overlap: bool) -> bool:
    try:
        seeds = _find_seeds(grid)
        out, meta = _simulate(grid, return_meta=True)
    except Exception:
        return False

    seed_positions = [(r, c) for r, c, _ in seeds]
    if not _pairwise_seed_spacing_ok(seed_positions):
        return False

    counts = Counter(map(int, grid.ravel()))
    if any(counts[color] != 1 for _r, _c, color in seeds):
        return False

    wall_colors = [color for color, count in counts.items() if color != BG and count >= 2]
    if len(wall_colors) < 2:
        return False

    wall_cells = int(np.count_nonzero(grid) - len(seeds))
    if not 4 <= wall_cells <= 8:
        return False

    if np.array_equal(out, grid):
        return False

    for ray in meta:
        if len(ray["wall_hits"]) < 1:
            return False
        if len(ray["blind_hits"]) < 1:
            return False
        if len(ray["post_bounce_trail"]) < 1:
            return False

    every_step = _simulate(grid, detect_every_step=True, bounce=True)
    if np.array_equal(every_step, out):
        return False

    no_bounce = _simulate(grid, detect_every_step=False, bounce=False)
    if np.array_equal(no_bounce, out):
        return False

    if want_overlap:
        reversed_out = _simulate(grid, reverse_order=True)
        if np.array_equal(reversed_out, out):
            return False
        if _path_overlap_count(meta) < 1:
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    shape = (h, w)
    seed_count = random.randint(2, 3)
    want_overlap = seed_count >= 2 and random.random() < 0.45
    extra_bounce_budget = 1 if (2 * seed_count + 2 <= 8 and random.random() < 0.45) else 0

    seed_positions: list[tuple[int, int]] = []
    walls: list[tuple[int, int, str]] = []

    if want_overlap:
        overlap = _build_overlap_pair(shape)
        if overlap is None:
            return None
        seed_positions.extend(overlap[0])
        walls.extend(overlap[1])

    while len(seed_positions) < seed_count:
        for _ in range(120):
            r = random.randint(0, h - 6)
            c = random.randint(0, w - 8)
            candidate = (r, c)
            if _can_place_seed(seed_positions, candidate):
                seed_positions.append(candidate)
                break
        else:
            return None

    forbidden = set(seed_positions)
    forbidden.update((r, c) for r, c, _kind in walls)
    for idx, seed in enumerate(seed_positions):
        if want_overlap and idx < 2:
            continue
        use_extra = extra_bounce_budget > 0 and random.random() < 0.6
        planned = _plan_independent_seed(seed, shape, forbidden, extra_bounce=use_extra)
        if planned is None:
            return None
        walls.extend(planned)
        forbidden.update((r, c) for r, c, _kind in planned)
        if use_extra:
            extra_bounce_budget -= 1

    if len(walls) < 4 or len(walls) > 8:
        return None

    seed_colors = _choose_seed_colors(seed_count)
    grid = _build_grid(shape, seed_positions, seed_colors, walls)
    if grid is None:
        return None

    if not _validate_instance(grid, want_overlap=want_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-bounce ray instance")
