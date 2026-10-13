"""ARC task 6: counterfactual gravity followed by intersection-only overlay."""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


YELLOW = 4
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _drop_distance(coords: np.ndarray, terrain: np.ndarray, height: int) -> int:
    """Maximum rigid downward translation avoiding terrain and the lower edge."""
    occupied = {(int(r), int(c)) for r, c in terrain}
    distance = 0
    while True:
        candidate = distance + 1
        if any(int(r) + candidate >= height or (int(r) + candidate, int(c)) in occupied
               for r, c in coords):
            return distance
        distance = candidate


def _moved_set(coords: np.ndarray, distance: int) -> set[tuple[int, int]]:
    return {(int(r) + distance, int(c)) for r, c in coords}


def main(grid: np.ndarray) -> np.ndarray:
    """Drop two sprites independently and retain only their yellow overlap.

    Input structure:
    A dominant background surrounds two differently colored, suspended,
    4-connected sprites.  A third foreground color forms a contiguous uneven
    terrain region touching the bottom edge.

    Rule:
    Treating the other sprite as absent, translate each sprite rigidly downward
    as far as possible without crossing the bottom edge or a terrain cell.  Erase
    both original sprites, preserve the terrain, and paint the cellwise
    intersection of the two independently settled shapes yellow.

    Key insight:
    The sprites never support one another: both falls are counterfactual and
    independent, and only the overlap of their two parallel-world destinations
    survives in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    non_bg = set(map(int, g.ravel())) - {bg}
    bottom_colors = set(map(int, g[-1])) - {bg}
    assert len(bottom_colors) == 1, "terrain must be the unique color touching the bottom"
    terrain_color = next(iter(bottom_colors))
    sprite_colors = non_bg - {terrain_color}
    assert len(sprite_colors) == 2, "expected exactly two differently colored sprites"

    terrain = np.argwhere(g == terrain_color)
    assert terrain.size and len(_components(g == terrain_color)) == 1
    sprites: list[np.ndarray] = []
    for color in sprite_colors:
        parts = _components(g == color)
        assert len(parts) == 1, "each sprite color must form one 4-connected object"
        coords = parts[0]
        assert int(coords[:, 0].max()) < int(terrain[:, 0].min()), "sprites must begin above terrain"
        sprites.append(coords)

    landed = [
        _moved_set(coords, _drop_distance(coords, terrain, h))
        for coords in sprites
    ]
    overlap = landed[0] & landed[1]

    out = g.copy()
    for color in sprite_colors:
        out[out == color] = bg
    for r, c in overlap:
        out[r, c] = YELLOW
    return out


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    queue = deque(seen)
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen == cells


def _attempt() -> np.ndarray | None:
    # The landed construction deliberately contains both stopping branches:
    # A is stopped by the rightmost terrain column; B reaches the bottom.
    h = random.randint(13, 22)
    w = random.randint(13, 24)
    core_len = random.randint(1, 3)
    terrain_end = random.randint(2, w - core_len - 6)
    terrain_start = random.randint(0, max(0, terrain_end - 4))
    row = h - 4

    core = {(row, terrain_end + 2 + j) for j in range(core_len)}
    a = {(row, c) for c in range(terrain_end, terrain_end + 2 + core_len)}
    # Shape A gets visible, random upper structure without changing its blocker.
    for c in random.sample(sorted(c for _r, c in a), random.randint(1, min(3, len(a)))):
        a.add((row - 1, c))
        if random.random() < 0.35:
            a.add((row - 2, c))

    stem_col = terrain_end + 2 + core_len
    b = set(core)
    b.add((row, stem_col))
    b.update((r, stem_col) for r in range(row, h))
    if stem_col + 1 < w and random.random() < 0.7:
        b.add((h - 2, stem_col + 1))
        if random.random() < 0.45:
            b.add((h - 3, stem_col + 1))
    if random.random() < 0.55:
        b.add((row - 1, min(c for _r, c in core)))

    if not (_connected(a) and _connected(b) and a & b == core):
        return None

    terrain_cells: set[tuple[int, int]] = set()
    heights: dict[int, int] = {}
    for c in range(terrain_start, terrain_end + 1):
        if c == terrain_end:
            height = 3
        else:
            height = random.randint(1, 5)
            if c > terrain_start:
                height = max(1, min(5, heights[c - 1] + random.choice((-1, 0, 1))))
        heights[c] = height
        terrain_cells.update((r, c) for r in range(h - height, h))
    if a & terrain_cells or b & terrain_cells:
        return None

    max_shift = row - 3
    if max_shift < 4:
        return None
    da = random.randint(2, min(6, max_shift))
    db_choices = [d for d in range(2, min(8, max_shift) + 1) if d != da]
    if not db_choices:
        return None
    db = random.choice(db_choices)
    original_a = {(r - da, c) for r, c in a}
    original_b = {(r - db, c) for r, c in b}
    if original_a & original_b or min(r for r, _c in original_a | original_b) < 0:
        return None
    # Make the sprites visibly suspended and prevent same-color adjacency to terrain.
    if max(r for r, _c in original_a | original_b) >= min(r for r, _c in terrain_cells) - 1:
        return None

    palette = [0, 1, 2, 3, 5, 6, 7, 8, 9]
    bg, terrain_color, color_a, color_b = random.sample(palette, 4)
    grid = np.full((h, w), bg, dtype=int)
    for r, c in terrain_cells:
        grid[r, c] = terrain_color
    for r, c in original_a:
        grid[r, c] = color_a
    for r, c in original_b:
        grid[r, c] = color_b

    terrain_arr = np.asarray(sorted(terrain_cells), dtype=int)
    arr_a = np.asarray(sorted(original_a), dtype=int)
    arr_b = np.asarray(sorted(original_b), dtype=int)
    got_da = _drop_distance(arr_a, terrain_arr, h)
    got_db = _drop_distance(arr_b, terrain_arr, h)
    landed_a = _moved_set(arr_a, got_da)
    landed_b = _moved_set(arr_b, got_db)

    # Semantic validation, including anti-shortcut counterexamples.
    if got_da != da or got_db != db or got_da == got_db:
        return None
    if landed_a != a or landed_b != b or landed_a & landed_b != core:
        return None
    if not (0 < len(core) < len(a) and len(core) < len(b)):
        return None
    if (original_a & original_b) == core:  # no-drop overlay must not solve the task
        return None
    if max(r for r, _c in b) != h - 1:  # B is bottom-limited.
        return None
    if not any((r + 1, c) in terrain_cells for r, c in a):  # A is terrain-limited.
        return None
    # Sequentially dropping B onto already-landed A must change its destination.
    stacked_obstacles = np.asarray(sorted(terrain_cells | a), dtype=int)
    if _drop_distance(arr_b, stacked_obstacles, h) == got_db:
        return None

    expected = grid.copy()
    expected[np.isin(expected, [color_a, color_b])] = bg
    for r, c in core:
        expected[r, c] = YELLOW
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate diverse, non-degenerate two-world gravity scenes."""
    for _ in range(500):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid interference-shadow scene")
