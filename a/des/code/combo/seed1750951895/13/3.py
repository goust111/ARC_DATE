from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _find_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if grid[nr, nc] != BG and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _ordered_path(component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = set(component)
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in component:
        nbrs = [(nr, nc) for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if (nr, nc) in cells]
        adjacency[(r, c)] = nbrs

    endpoints = [cell for cell, nbrs in adjacency.items() if len(nbrs) == 1]
    assert len(component) >= 2
    assert len(endpoints) == 2
    assert all(len(nbrs) in (1, 2) for nbrs in adjacency.values())

    path = [endpoints[0]]
    prev = None
    current = endpoints[0]
    while True:
        next_cells = [cell for cell in adjacency[current] if cell != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1
        prev, current = current, next_cells[0]
        path.append(current)

    assert len(path) == len(component)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the color order along each foreground path object while leaving the path geometry unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for component in _find_components(g):
        path = _ordered_path(component)
        colors = [int(g[r, c]) for r, c in path]
        assert len(set(colors)) >= 2
        rev = colors[::-1]
        for (r, c), color in zip(path, rev, strict=False):
            out[r, c] = color
    return out


def _forbidden_mask(occupied: np.ndarray) -> np.ndarray:
    h, w = occupied.shape
    blocked = occupied.copy()
    for r in range(h):
        for c in range(w):
            if not occupied[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        blocked[nr, nc] = True
    return blocked


def _walk_candidates(path: list[tuple[int, int]], blocked: np.ndarray) -> list[tuple[int, int]]:
    h, w = blocked.shape
    last = path[-1]
    prev = path[-2] if len(path) >= 2 else None
    candidates: list[tuple[int, int]] = []
    for nr, nc in _neighbors4(last[0], last[1], h, w):
        if blocked[nr, nc] or (nr, nc) in path:
            continue
        if prev is not None:
            back_neighbors = 0
            for ar, ac in _neighbors4(nr, nc, h, w):
                if (ar, ac) in path and (ar, ac) != last:
                    back_neighbors += 1
            if back_neighbors:
                continue
        candidates.append((nr, nc))
    return candidates


def _sample_path(blocked: np.ndarray, min_len: int, max_len: int) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    free = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
    random.shuffle(free)
    target_len = random.randint(min_len, max_len)

    for start in free:
        path = [start]
        while len(path) < target_len:
            candidates = _walk_candidates(path, blocked)
            if not candidates:
                break
            prev = path[-2] if len(path) >= 2 else None
            if prev is None:
                chosen = random.choice(candidates)
            else:
                straight: list[tuple[int, int]] = []
                turn: list[tuple[int, int]] = []
                dr0 = path[-1][0] - prev[0]
                dc0 = path[-1][1] - prev[1]
                for cand in candidates:
                    dr1 = cand[0] - path[-1][0]
                    dc1 = cand[1] - path[-1][1]
                    if (dr0, dc0) == (dr1, dc1):
                        straight.append(cand)
                    else:
                        turn.append(cand)
                if straight and turn:
                    chosen = random.choice(turn if random.random() < 0.5 else straight)
                else:
                    chosen = random.choice(candidates)
            path.append(chosen)

        if len(path) >= min_len:
            return path
    return None


def _sample_colors(length: int) -> list[int]:
    palette_size = random.randint(2, min(5, len(COLOR_POOL)))
    palette = random.sample(COLOR_POOL, palette_size)
    while True:
        colors = [random.choice(palette) for _ in range(length)]
        if len(set(colors)) >= 2 and colors != colors[::-1]:
            return colors


def _validate_paths(grid: np.ndarray) -> bool:
    try:
        components = _find_components(grid)
        if not components:
            return False
        for component in components:
            path = _ordered_path(component)
            colors = [int(grid[r, c]) for r, c in path]
            if len(set(colors)) < 2:
                return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 20)
    w = random.randint(10, 20)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    object_count = random.randint(1, 4)
    lengths = [random.randint(3, min(10, max(3, (h * w) // 8))) for _ in range(object_count)]
    random.shuffle(lengths)

    for length in lengths:
        blocked = _forbidden_mask(occupied)
        path = _sample_path(blocked, min_len=3, max_len=length)
        if path is None:
            return None
        colors = _sample_colors(len(path))
        for (r, c), color in zip(path, colors, strict=False):
            grid[r, c] = color
            occupied[r, c] = True

    if not _validate_paths(grid):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
