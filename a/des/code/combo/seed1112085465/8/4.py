from __future__ import annotations

import random

import numpy as np


AZURE = 8
PALETTE = (0, 1, 2, 3, 4, 5, 6, 7, 9)
MIN_SIZE = 9
MAX_SIZE = 16
MAX_RECOLORERS = 5
MAX_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == AZURE):
        return AZURE
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _emitter_direction(r: int, c: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    return 0, -1


def _find_markers(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int], int]]:
    h, w = grid.shape
    emitters: list[tuple[int, int, int]] = []
    recolorers: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg:
                continue
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                emitters.append((r, c, color))
            else:
                recolorers[(r, c)] = color

    emitters.sort(key=lambda item: (int(item[0]), int(item[1])))
    return emitters, recolorers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast rays inward from edge emitters. The ray color switches after passing over
    any interior recolorer pixel, while earlier rays keep priority at crossings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background_color(g)
    emitters, recolorers = _find_markers(g, bg)

    out = g.copy()
    claimed = np.zeros((h, w), dtype=bool)

    for r0, c0, start_color in emitters:
        dr, dc = _emitter_direction(int(r0), int(c0), g.shape)
        color = int(start_color)
        r, c = int(r0), int(c0)

        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break

            is_recolorer = (r, c) in recolorers
            if not claimed[r, c] and (is_recolorer or int(g[r, c]) == bg):
                out[r, c] = color
                claimed[r, c] = True

            if is_recolorer:
                color = int(recolorers[(r, c)])

            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                break

    return out


def _choose_edges() -> tuple[list[str], list[str]]:
    scenario = random.choice(("vertical_only", "horizontal_only", "mixed"))
    if scenario == "vertical_only":
        return random.choice((["top"], ["bottom"], ["top", "bottom"])), []
    if scenario == "horizontal_only":
        return [], random.choice((["left"], ["right"], ["left", "right"]))
    return (
        random.choice((["top"], ["bottom"], ["top", "bottom"])),
        random.choice((["left"], ["right"], ["left", "right"])),
    )


def _sample_emitters(h: int, w: int) -> list[tuple[str, int, int]]:
    vertical_edges, horizontal_edges = _choose_edges()
    assert vertical_edges or horizontal_edges

    emitters: list[tuple[str, int, int]] = []
    used_cols: set[int] = set()
    used_rows: set[int] = set()

    for edge in vertical_edges:
        choices = [c for c in range(1, w - 1) if c not in used_cols]
        if not choices:
            raise ValueError("no free column for vertical emitter")
        c = random.choice(choices)
        used_cols.add(int(c))
        r = 0 if edge == "top" else h - 1
        emitters.append((edge, int(r), int(c)))

    for edge in horizontal_edges:
        choices = [r for r in range(1, h - 1) if r not in used_rows]
        if not choices:
            raise ValueError("no free row for horizontal emitter")
        r = random.choice(choices)
        used_rows.add(int(r))
        c = 0 if edge == "left" else w - 1
        emitters.append((edge, int(r), int(c)))

    return emitters


def _candidate_pools(
    h: int, w: int, emitters: list[tuple[str, int, int]]
) -> tuple[dict[int, list[tuple[int, int]]], list[tuple[int, int]]]:
    vertical_cols = [c for _edge, r, c in emitters if r in (0, h - 1)]
    horizontal_rows = [r for _edge, r, c in emitters if c in (0, w - 1)]

    dedicated: dict[int, list[tuple[int, int]]] = {}
    for idx, (_edge, r, c) in enumerate(emitters):
        if r in (0, h - 1):
            pool = [(rr, c) for rr in range(1, h - 1) if rr not in horizontal_rows]
        else:
            pool = [(r, cc) for cc in range(1, w - 1) if cc not in vertical_cols]
        dedicated[idx] = pool

    intersections = [(r, c) for r in horizontal_rows for c in vertical_cols]
    return dedicated, intersections


def _sample_recolorers(
    h: int, w: int, emitters: list[tuple[str, int, int]]
) -> list[tuple[int, int]]:
    dedicated, intersections = _candidate_pools(h, w, emitters)
    picks: list[tuple[int, int]] = []

    for idx in range(len(emitters)):
        pool = dedicated[idx][:]
        random.shuffle(pool)
        target = random.choice((0, 1, 2))
        picks.extend(pool[: min(target, len(pool))])

    if intersections:
        pool = intersections[:]
        random.shuffle(pool)
        target = random.choice((0, 1, 2))
        picks.extend(pool[: min(target, len(pool))])

    seen: set[tuple[int, int]] = set()
    unique_picks: list[tuple[int, int]] = []
    for pos in picks:
        if pos not in seen:
            seen.add(pos)
            unique_picks.append(pos)

    if not unique_picks:
        fallback = []
        for pool in dedicated.values():
            fallback.extend(pool)
        fallback.extend(intersections)
        if not fallback:
            raise ValueError("no valid recolorer position")
        unique_picks = [random.choice(fallback)]

    if len(unique_picks) > MAX_RECOLORERS:
        random.shuffle(unique_picks)
        unique_picks = unique_picks[:MAX_RECOLORERS]

    return unique_picks


def _paint_input(
    h: int,
    w: int,
    emitters: list[tuple[str, int, int]],
    recolorers: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), AZURE, dtype=int)
    total_markers = len(emitters) + len(recolorers)
    colors = random.sample(PALETTE, total_markers)

    for (_edge, r, c), color in zip(emitters, colors[: len(emitters)]):
        grid[int(r), int(c)] = int(color)

    for (r, c), color in zip(recolorers, colors[len(emitters) :]):
        grid[int(r), int(c)] = int(color)

    return grid


def _ray_positions(
    emitters: list[tuple[str, int, int]], shape: tuple[int, int]
) -> list[list[tuple[int, int]]]:
    positions: list[list[tuple[int, int]]] = []
    h, w = shape
    for _edge, r0, c0 in emitters:
        dr, dc = _emitter_direction(int(r0), int(c0), shape)
        ray: list[tuple[int, int]] = []
        r, c = int(r0), int(c0)
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            ray.append((r, c))
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                break
        positions.append(ray)
    return positions


def _is_interesting(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    emitters, recolorers = _find_markers(grid, bg)
    if not emitters or not recolorers:
        return False

    rays = _ray_positions([(f"e{i}", r, c) for i, (r, c, _color) in enumerate(emitters)], grid.shape)
    recolorer_hits = 0
    multi_hit_ray = False
    crossing_cells: set[tuple[int, int]] = set()

    ray_sets = [set(ray) for ray in rays]
    for ray in rays:
        hits = sum((r, c) in recolorers for r, c in ray)
        recolorer_hits += hits
        if hits >= 2:
            multi_hit_ray = True

    for i in range(len(ray_sets)):
        for j in range(i + 1, len(ray_sets)):
            crossing_cells |= ray_sets[i] & ray_sets[j]

    if recolorer_hits == 0:
        return False
    if len(emitters) >= 2 and not multi_hit_ray and not (crossing_cells & recolorers.keys()):
        return random.random() < 0.4
    return True


def _generate_attempt() -> np.ndarray:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    emitters = _sample_emitters(h, w)
    recolorers = _sample_recolorers(h, w, emitters)
    grid = _paint_input(h, w, emitters, recolorers)

    out = main(grid)
    if np.array_equal(out, grid):
        raise ValueError("degenerate sample")
    if not _is_interesting(grid):
        raise ValueError("not enough structure")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            return _generate_attempt()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid ray-recolor puzzle")
