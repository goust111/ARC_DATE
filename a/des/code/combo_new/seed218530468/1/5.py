import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
GROUND = 9
WINDOW_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_ground_row(grid: np.ndarray) -> tuple[int, int]:
    candidates = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color != BG and np.all(grid[r] == color):
            candidates.append((r, color))
    assert candidates, "expected a full-width non-black ground line"
    return max(candidates, key=lambda item: item[0])


def _analyze_buildings(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int, int, int, int]]]:
    ground_row, ground_color = _find_ground_row(grid)
    above_ground = grid[:ground_row]
    mask = (above_ground != BG) & (above_ground != ground_color)
    components = _components(mask)
    assert components, "expected at least one building"

    buildings = []
    spans = []
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 + 1 < ground_row, "building must not touch the ground line"

        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert coords.shape[0] == area, "building component must fill its bbox"

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors, counts = np.unique(sub, return_counts=True)
        order = np.argsort(-counts)
        body_color = int(colors[order[0]])
        assert body_color not in (BG, ground_color), "building body color must be non-background"
        if colors.size > 1:
            assert counts[order[0]] > counts[order[1]], "building body must be the unique majority"

        windows = sub[sub != body_color]
        assert windows.size > 0, "each building needs at least one window"
        win_colors, win_counts = np.unique(windows, return_counts=True)
        best = int(win_counts.max())
        assert int(np.sum(win_counts == best)) == 1, "window majority color must be unique"
        projection_color = int(win_colors[np.argmax(win_counts)])

        buildings.append((r0, c0, r1, c1, projection_color))
        spans.append((c0, c1))

    spans.sort()
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert end < next_start, "building spans must be horizontally disjoint"

    buildings.sort(key=lambda item: item[1])
    return ground_row, ground_color, buildings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each grey building straight down to the ground using its most common window color.

    Input structure:
    - The background is black.
    - A single full-width horizontal maroon line marks the ground near the bottom.
    - Above that line are several non-overlapping solid rectangular buildings whose dominant body color is grey.
    - Each building contains one-pixel colored windows, and the window colors are neither black nor grey nor maroon.

    Rule:
    - Keep every existing pixel unchanged.
    - For each building, count the colors of its window pixels and pick the unique most frequent one.
    - Take the building's full column span and fill the rectangle from one row below the building down to one row above the ground with that color.
    - Only black cells in that projection corridor are repainted.

    Key insight:
    - The relevant objects are full colored rectangles above a unique horizontal ground line, and the projection color comes from a frequency summary of the windows rather than from the grey building body.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ground_row, _ground_color, buildings = _analyze_buildings(g)
    out = g.copy()
    for _r0, c0, r1, c1, projection_color in buildings:
        strip = out[r1 + 1 : ground_row, c0 : c1 + 1]
        strip[strip == BG] = projection_color
    return out


def _sample_segments(rng: random.Random, width: int, count: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        widths = [rng.randint(3, 8) for _ in range(count)]
        gaps = [rng.randint(1, 4) for _ in range(count - 1)]
        left_margin = rng.randint(1, 3)
        right_margin = rng.randint(1, 3)
        total = left_margin + right_margin + sum(widths) + sum(gaps)
        if total > width:
            continue

        slack = width - total
        spaces = [left_margin] + gaps + [right_margin]
        for _ in range(slack):
            spaces[rng.randrange(len(spaces))] += 1

        segments = []
        c = spaces[0]
        inner_gaps = spaces[1:-1]
        for i, building_width in enumerate(widths):
            segments.append((c, c + building_width - 1))
            c += building_width
            if i < len(inner_gaps):
                c += inner_gaps[i]
        return segments
    return None


def _window_plan(rng: random.Random, interior_size: int, building_area: int) -> dict[int, int]:
    max_windows = min(interior_size, max(1, building_area // 2 - 1))
    palette_sizes = [1]
    if interior_size >= 2 and max_windows >= 3:
        palette_sizes.append(2)
    if interior_size >= 3 and max_windows >= 4:
        palette_sizes.append(3)

    n_colors = rng.choice(palette_sizes)
    palette = rng.sample(WINDOW_COLORS, n_colors)
    if n_colors == 1:
        return {palette[0]: rng.randint(1, max_windows)}

    dominant_count = rng.randint(2, max_windows - (n_colors - 1))
    other_counts = [1] * (n_colors - 1)
    extra = rng.randint(0, max_windows - dominant_count - (n_colors - 1))
    while extra > 0:
        candidates = [i for i, count in enumerate(other_counts) if count + 1 < dominant_count]
        if not candidates:
            break
        idx = rng.choice(candidates)
        other_counts[idx] += 1
        extra -= 1

    counts = [dominant_count] + other_counts
    return {color: count for color, count in zip(palette, counts)}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(14, 30)
    width = rng.randint(14, 30)
    ground_row = rng.randint(max(8, height - 6), height - 2)

    feasible_counts = [n for n in range(1, 5) if 2 + 3 * n + (n - 1) <= width]
    if not feasible_counts:
        return None
    building_count = rng.choice(feasible_counts)
    segments = _sample_segments(rng, width, building_count)
    if segments is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[ground_row, :] = GROUND

    for c0, c1 in segments:
        building_width = c1 - c0 + 1
        max_height = min(10, ground_row - 2)
        if max_height < 3:
            return None
        building_height = rng.randint(3, max_height)
        top_max = ground_row - building_height - 2
        if top_max < 0:
            return None
        r0 = rng.randint(0, top_max)
        r1 = r0 + building_height - 1

        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        if not interior:
            return None
        plan = _window_plan(rng, len(interior), building_height * building_width)
        total_windows = sum(plan.values())
        if total_windows > len(interior):
            return None
        positions = rng.sample(interior, total_windows)

        idx = 0
        for color, count in plan.items():
            for r, c in positions[idx : idx + count]:
                grid[r, c] = color
            idx += count

    try:
        _ground_row, _ground_color, buildings = _analyze_buildings(grid)
    except AssertionError:
        return None
    if len(buildings) != building_count:
        return None
    if not np.any(main(grid) != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
