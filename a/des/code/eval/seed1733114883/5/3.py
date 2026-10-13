from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

TOWER_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA)

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)

# For a 2x2 tower, the four launch cells are the canonical discrete representatives
# of the four edge midpoints described in the puzzle statement.
EMITTERS = (
    (UP, (-1, 0)),
    (RIGHT, (0, 2)),
    (DOWN, (2, 1)),
    (LEFT, (1, -1)),
)

BACKSLASH_REFLECT = {
    RIGHT: DOWN,
    DOWN: RIGHT,
    LEFT: UP,
    UP: LEFT,
}

SLASH_REFLECT = {
    RIGHT: UP,
    UP: RIGHT,
    LEFT: DOWN,
    DOWN: LEFT,
}

MIRROR_HELPERS = {
    "\\": [(-1, -1), (1, 1)],
    "/": [(-1, 1), (1, -1)],
}

GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _find_towers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    special = {BLACK, GREY, AZURE, ORANGE}
    towers: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in special or bool(visited[r, c]):
                continue

            stack = [(int(r), int(c))]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in (UP, RIGHT, DOWN, LEFT):
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if bool(visited[nr, nc]) or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            if len(coords) != 4:
                continue

            rows = [int(rr) for rr, _ in coords]
            cols = [int(cc) for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue

            ok = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if int(g[rr, cc]) != color:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            towers.append(
                {
                    "color": int(color),
                    "top_left": (int(r0), int(c0)),
                    "cells": {(int(rr), int(cc)) for rr, cc in coords},
                }
            )

    towers.sort(key=lambda tower: (int(tower["top_left"][0]), int(tower["top_left"][1]), int(tower["color"])))
    return towers


def _find_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    g = np.asarray(grid, dtype=int)
    mirrors: dict[tuple[int, int], str] = {}

    for r, c in np.argwhere(g == GREY):
        r = int(r)
        c = int(c)

        has_backslash = (
            _in_bounds(g.shape, r - 1, c - 1)
            and _in_bounds(g.shape, r + 1, c + 1)
            and int(g[r - 1, c - 1]) == AZURE
            and int(g[r + 1, c + 1]) == AZURE
        )
        has_slash = (
            _in_bounds(g.shape, r - 1, c + 1)
            and _in_bounds(g.shape, r + 1, c - 1)
            and int(g[r - 1, c + 1]) == AZURE
            and int(g[r + 1, c - 1]) == AZURE
        )

        if has_backslash and not has_slash:
            mirrors[(int(r), int(c))] = "\\"
        elif has_slash and not has_backslash:
            mirrors[(int(r), int(c))] = "/"

    return mirrors


def _find_receivers(grid: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(np.asarray(grid, dtype=int) == ORANGE)]


def _trace_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    towers = _find_towers(g)
    mirrors = _find_mirrors(g)

    tower_cells: dict[tuple[int, int], int] = {}
    for idx, tower in enumerate(towers):
        for cell in tower["cells"]:
            tower_cells[cell] = int(idx)

    coverage = np.zeros((h, w), dtype=int)
    reflected = np.zeros((h, w), dtype=bool)
    cell_colors: dict[tuple[int, int], list[int]] = defaultdict(list)
    ray_infos: list[dict] = []

    for tower in towers:
        color = int(tower["color"])
        r0, c0 = map(int, tower["top_left"])

        for initial_dir, launch_offset in EMITTERS:
            dr, dc = map(int, initial_dir)
            r = int(r0 + launch_offset[0])
            c = int(c0 + launch_offset[1])
            reflections = 0
            used_reflection = False
            path: list[tuple[int, int, bool]] = []

            while 0 <= r < h and 0 <= c < w:
                if (int(r), int(c)) in tower_cells:
                    break

                after_reflection = bool(reflections > 0)
                path.append((int(r), int(c), bool(after_reflection)))
                coverage[r, c] += 1
                if after_reflection:
                    reflected[r, c] = True
                cell_colors[(int(r), int(c))].append(int(color))

                mirror_kind = mirrors.get((int(r), int(c)))
                if mirror_kind is not None:
                    if reflections == 1:
                        break
                    if mirror_kind == "\\":
                        dr, dc = BACKSLASH_REFLECT[(int(dr), int(dc))]
                    else:
                        dr, dc = SLASH_REFLECT[(int(dr), int(dc))]
                    reflections = 1
                    used_reflection = True

                r += int(dr)
                c += int(dc)

            ray_infos.append({"color": int(color), "path": path, "reflected": bool(used_reflection)})

    return {
        "coverage": coverage,
        "reflected": reflected,
        "cell_colors": cell_colors,
        "ray_infos": ray_infos,
        "towers": towers,
        "mirrors": mirrors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Signal towers broadcast four orthogonal rays. Grey mirror pixels reflect a ray at
    most once, and receivers recolor based on how many rays traverse their cells.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    scene = _trace_scene(g)

    if not scene["towers"]:
        return out

    for r, c in _find_receivers(g):
        hits = scene["cell_colors"].get((int(r), int(c)), [])
        if len(hits) == 1:
            out[r, c] = int(hits[0])
        elif len(hits) >= 2:
            out[r, c] = MAROON

    return out


def _can_place_tower(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if r < 0 or c < 0 or r + 1 >= h or c + 1 >= w:
        return False

    r0 = max(0, int(r - 1))
    r1 = min(h - 1, int(r + 2))
    c0 = max(0, int(c - 1))
    c1 = min(w - 1, int(c + 2))
    return bool((grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK).all())


def _paint_tower(grid: np.ndarray, r: int, c: int, color: int) -> None:
    grid[int(r) : int(r) + 2, int(c) : int(c) + 2] = int(color)


def _can_place_mirror(grid: np.ndarray, r: int, c: int, kind: str) -> bool:
    cells = [(int(r), int(c))]
    for dr, dc in MIRROR_HELPERS[kind]:
        rr = int(r + dr)
        cc = int(c + dc)
        if not _in_bounds(grid.shape, rr, cc):
            return False
        cells.append((int(rr), int(cc)))

    for rr, cc in cells:
        if int(grid[rr, cc]) != BLACK:
            return False

    rows = [rr for rr, _ in cells]
    cols = [cc for _, cc in cells]
    r0 = max(0, min(rows) - 1)
    r1 = min(grid.shape[0] - 1, max(rows) + 1)
    c0 = max(0, min(cols) - 1)
    c1 = min(grid.shape[1] - 1, max(cols) + 1)
    cell_set = set(cells)

    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if (rr, cc) in cell_set:
                continue
            if int(grid[rr, cc]) != BLACK:
                return False

    return True


def _paint_mirror(grid: np.ndarray, r: int, c: int, kind: str) -> None:
    grid[int(r), int(c)] = GREY
    for dr, dc in MIRROR_HELPERS[kind]:
        grid[int(r + dr), int(c + dc)] = AZURE


def _pick_with_spacing(
    rng: random.Random,
    pool: list[tuple[int, int]],
    chosen: list[tuple[int, int]],
    min_chebyshev: int = 1,
) -> tuple[int, int] | None:
    shuffled = list(pool)
    rng.shuffle(shuffled)

    for cell in shuffled:
        if all(max(abs(int(cell[0]) - int(other[0])), abs(int(cell[1]) - int(other[1]))) > min_chebyshev for other in chosen):
            return (int(cell[0]), int(cell[1]))

    if shuffled:
        cell = shuffled[0]
        return (int(cell[0]), int(cell[1]))

    return None


def _choose_mirror_target(grid: np.ndarray, scene: dict, rng: random.Random) -> np.ndarray | None:
    coverage = np.asarray(scene["coverage"], dtype=int)
    primary: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []

    for r, c in np.argwhere(coverage > 0):
        r = int(r)
        c = int(c)
        if int(grid[r, c]) != BLACK:
            continue
        if int(coverage[r, c]) == 1:
            primary.append((int(r), int(c)))
        else:
            fallback.append((int(r), int(c)))

    rng.shuffle(primary)
    rng.shuffle(fallback)

    for r, c in primary + fallback:
        kinds = ["\\", "/"]
        rng.shuffle(kinds)
        for kind in kinds:
            if not _can_place_mirror(grid, r, c, kind):
                continue
            trial = grid.copy()
            _paint_mirror(trial, r, c, kind)
            trial_scene = _trace_scene(trial)
            reflected_hits = np.argwhere(
                (trial == BLACK)
                & (trial_scene["coverage"] > 0)
                & trial_scene["reflected"]
            )
            if int(reflected_hits.shape[0]) == 0:
                continue
            return trial

    return None


def _add_distractor_mirrors(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    extra_count = rng.choice([0, 0, 1, 1, 2])
    out = grid.copy()

    for _ in range(extra_count):
        scene = _trace_scene(out)
        coverage = np.asarray(scene["coverage"], dtype=int)
        candidates: list[tuple[int, int, str]] = []

        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                if int(coverage[r, c]) != 0 or int(out[r, c]) != BLACK:
                    continue
                for kind in ("\\", "/"):
                    if not _can_place_mirror(out, r, c, kind):
                        continue
                    helpers_ok = True
                    for dr, dc in MIRROR_HELPERS[kind]:
                        if int(coverage[r + dr, c + dc]) != 0:
                            helpers_ok = False
                            break
                    if helpers_ok:
                        candidates.append((int(r), int(c), kind))

        if not candidates:
            break

        r, c, kind = rng.choice(candidates)
        _paint_mirror(out, r, c, kind)

    return out


def _sample_tower_colors(rng: random.Random, count: int) -> list[int]:
    colors = [int(rng.choice(TOWER_COLORS)) for _ in range(int(count))]
    if len(set(colors)) == 1 and count > 1:
        idx = int(rng.randrange(count))
        choices = [color for color in TOWER_COLORS if color != colors[idx]]
        colors[idx] = int(rng.choice(choices))
    return colors


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = int(rng.randint(12, 18))
    w = int(rng.randint(12, 18))
    grid = np.full((h, w), BLACK, dtype=int)

    tower_count = int(rng.choice([3, 3, 4, 4]))
    colors = _sample_tower_colors(rng, tower_count)

    placed = 0
    for color in colors:
        success = False
        for _ in range(200):
            r = int(rng.randrange(h - 1))
            c = int(rng.randrange(w - 1))
            if not _can_place_tower(grid, r, c):
                continue
            _paint_tower(grid, r, c, int(color))
            placed += 1
            success = True
            break
        if not success:
            return None

    if placed != tower_count:
        return None

    base_scene = _trace_scene(grid)
    if len(base_scene["towers"]) != tower_count:
        return None

    mirrored = _choose_mirror_target(grid, base_scene, rng)
    if mirrored is None:
        return None

    grid = _add_distractor_mirrors(mirrored, rng)
    scene = _trace_scene(grid)
    coverage = np.asarray(scene["coverage"], dtype=int)
    reflected = np.asarray(scene["reflected"], dtype=bool)
    black_mask = grid == BLACK

    zero_candidates = [(int(r), int(c)) for r, c in np.argwhere(black_mask & (coverage == 0))]
    single_candidates = [(int(r), int(c)) for r, c in np.argwhere(black_mask & (coverage == 1))]
    multi_candidates = [(int(r), int(c)) for r, c in np.argwhere(black_mask & (coverage >= 2))]
    single_reflected = [(int(r), int(c)) for r, c in np.argwhere(black_mask & (coverage == 1) & reflected)]
    multi_reflected = [(int(r), int(c)) for r, c in np.argwhere(black_mask & (coverage >= 2) & reflected)]

    if not zero_candidates or not single_candidates or not multi_candidates:
        return None
    if not single_reflected and not multi_reflected:
        return None

    chosen: list[tuple[int, int]] = []

    single_pos = _pick_with_spacing(
        rng,
        single_reflected if single_reflected else single_candidates,
        chosen,
    )
    if single_pos is None:
        return None
    chosen.append(single_pos)

    multi_pool = multi_reflected if (multi_reflected and not single_reflected) else multi_candidates
    multi_pos = _pick_with_spacing(rng, multi_pool, chosen)
    if multi_pos is None:
        return None
    chosen.append(multi_pos)

    zero_pos = _pick_with_spacing(rng, zero_candidates, chosen)
    if zero_pos is None:
        return None
    chosen.append(zero_pos)

    extra_receivers = int(rng.choice([0, 1, 2]))
    category_pools = {
        0: zero_candidates,
        1: single_candidates,
        2: multi_candidates,
    }
    for _ in range(extra_receivers):
        category = int(rng.choice([0, 1, 2]))
        extra = _pick_with_spacing(rng, category_pools[category], chosen)
        if extra is not None and extra not in chosen:
            chosen.append(extra)

    for r, c in chosen:
        grid[int(r), int(c)] = ORANGE

    out = main(grid)

    if int(out[zero_pos]) != ORANGE:
        return None
    if int(out[multi_pos]) != MAROON:
        return None

    single_hits = scene["cell_colors"].get(single_pos, [])
    if len(single_hits) != 1:
        return None
    if int(out[single_pos]) != int(single_hits[0]):
        return None

    if len(_find_receivers(grid)) < 3:
        return None

    final_scene = _trace_scene(grid)
    if len(final_scene["towers"]) != tower_count:
        return None
    if not any(bool(ray["reflected"]) for ray in final_scene["ray_infos"]):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle input")
