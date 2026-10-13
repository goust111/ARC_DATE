from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


THRESHOLD = 3
GRID_SHAPES = [(3, 4), (3, 5), (4, 4), (4, 5), (5, 4), (5, 5)]
COLOR_POOL = tuple(range(1, 10))
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _collect_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append({"color": color, "coords": coords, "area": len(coords)})
    return components


def _component_sizes_by_cell(grid: np.ndarray) -> np.ndarray:
    sizes = np.zeros(grid.shape, dtype=int)
    for comp in _collect_components(grid):
        area = int(comp["area"])
        for r, c in comp["coords"]:
            sizes[r, c] = area
    return sizes


def _self_masks(grid: np.ndarray) -> dict[int, np.ndarray]:
    return {
        color: np.where(grid == color, color, 0).astype(int)
        for color in map(int, np.unique(grid))
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand only the pixels from large same-color components into their self-mask blocks.

    Input structure:
    - The input is a small fully colored grid with 2-3 non-black colors and no black background.
    - Same-color cells form one or more 4-connected components, and at least one color has both large and small components.
    - The output is assembled from one input-shaped block per input cell.

    Rule:
    - Find every 4-connected component separately within each color and measure its area on the original grid.
    - Mark a pixel as large iff its component area is at least 3; otherwise mark it as small.
    - Replace each large pixel by the self-mask block of its own color, and replace each small pixel by an all-black block.

    Key insight:
    - Two equal-colored pixels can map to either the same nonzero block or to a blank block depending only on the size of their original connected component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.all(g != 0), "input must not contain black"

    h, w = g.shape
    assert h * h <= 30 and w * w <= 30, "expanded output must fit ARC limits"

    component_sizes = _component_sizes_by_cell(g)
    masks = _self_masks(g)
    out = np.zeros((h * h, w * w), dtype=int)

    for r in range(h):
        for c in range(w):
            if int(component_sizes[r, c]) < THRESHOLD:
                continue
            color = int(g[r, c])
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = masks[color]
    return out


def _random_connected_subset(
    allowed: np.ndarray, area: int, rng: random.Random, start_cells: list[tuple[int, int]] | None = None
) -> list[tuple[int, int]] | None:
    h, w = allowed.shape
    if int(np.sum(allowed)) < area:
        return None

    starts = start_cells or [tuple(map(int, cell)) for cell in np.argwhere(allowed)]
    rng.shuffle(starts)

    for start in starts[: min(len(starts), 20)]:
        if not allowed[start]:
            continue
        cells = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc]:
                frontier.add((nr, nc))

        while len(cells) < area and frontier:
            candidates = list(frontier)
            frontier_scores = []
            for cand in candidates:
                score = sum((nbr in cells) for nbr in _neighbors4(*cand))
                frontier_scores.append((score, rng.random(), cand))
            _, _, chosen = max(frontier_scores)
            frontier.remove(chosen)
            cells.add(chosen)
            for nr, nc in _neighbors4(*chosen):
                if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) == area:
            return list(cells)
    return None


def _place_focus_components(
    h: int, w: int, areas: list[int], rng: random.Random
) -> np.ndarray | None:
    focus_mask = np.zeros((h, w), dtype=bool)
    forbidden = np.zeros((h, w), dtype=bool)

    for area in sorted(areas, reverse=True):
        component = _random_connected_subset(~forbidden, area, rng)
        if component is None:
            return None
        for r, c in component:
            focus_mask[r, c] = True
        for r, c in component:
            forbidden[r, c] = True
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True
    return focus_mask


def _add_accent_color(
    grid: np.ndarray, occupied: np.ndarray, accent: int, rng: random.Random
) -> bool:
    h, w = grid.shape
    free = ~occupied
    free_count = int(np.sum(free))
    if free_count < 2:
        return False

    area1 = rng.randint(2, min(5, free_count - 1))
    patch1 = _random_connected_subset(free, area1, rng)
    if patch1 is None:
        return False
    for r, c in patch1:
        grid[r, c] = accent
        occupied[r, c] = True

    if free_count - area1 < 1 or rng.random() >= 0.45:
        return True

    separated = ~occupied
    for r, c in patch1:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                separated[nr, nc] = False
    if int(np.sum(separated)) == 0:
        return True

    area2 = rng.randint(1, min(2, int(np.sum(separated))))
    patch2 = _random_connected_subset(separated, area2, rng)
    if patch2 is None:
        return True
    for r, c in patch2:
        grid[r, c] = accent
        occupied[r, c] = True
    return True


def _focus_component_areas(cell_count: int, rng: random.Random) -> list[int]:
    areas = [rng.choice([3, 3, 4, 5]), rng.choice([1, 2])]
    if cell_count >= 16 and rng.random() < 0.55:
        areas.append(rng.choice([1, 2]))
    if cell_count >= 20 and rng.random() < 0.35:
        areas.append(rng.choice([3, 4]))
    while sum(areas) > cell_count - 2:
        areas.pop()
    return areas


def _input_meets_spec(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if (h, w) not in GRID_SHAPES:
        return False
    if np.any(grid == 0):
        return False

    colors = list(map(int, np.unique(grid)))
    if not 2 <= len(colors) <= 3:
        return False
    if any(int(np.sum(grid == color)) < 2 for color in colors):
        return False

    components = _collect_components(grid)
    if not any(comp["area"] >= THRESHOLD for comp in components):
        return False
    if not any(comp["area"] < THRESHOLD for comp in components):
        return False

    by_color: dict[int, set[str]] = defaultdict(set)
    for comp in components:
        by_color[int(comp["color"])].add("large" if int(comp["area"]) >= THRESHOLD else "small")
    if not any(statuses == {"large", "small"} for statuses in by_color.values()):
        return False

    out = main(grid)
    zero_block = np.zeros(grid.shape, dtype=int)
    sizes = _component_sizes_by_cell(grid)
    masks = _self_masks(grid)
    saw_blank = False
    saw_retained = False
    saw_same_color_split = False

    for color in colors:
        color_has_large = False
        color_has_small = False
        for r, c in np.argwhere(grid == color):
            block = out[r * h : (r + 1) * h, c * w : (c + 1) * w]
            if int(sizes[r, c]) >= THRESHOLD:
                if not np.array_equal(block, masks[int(color)]):
                    return False
                saw_retained = True
                color_has_large = True
            else:
                if not np.array_equal(block, zero_block):
                    return False
                saw_blank = True
                color_has_small = True
        saw_same_color_split |= color_has_large and color_has_small

    return saw_blank and saw_retained and saw_same_color_split


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    h, w = rng.choice(GRID_SHAPES)
    color_count = rng.choices([2, 3], weights=[3, 2], k=1)[0]
    colors = rng.sample(COLOR_POOL, color_count)
    focus, filler = colors[0], colors[1]

    focus_areas = _focus_component_areas(h * w, rng)
    focus_mask = _place_focus_components(h, w, focus_areas, rng)
    if focus_mask is None:
        return None

    grid = np.full((h, w), filler, dtype=int)
    grid[focus_mask] = focus
    occupied = focus_mask.copy()

    if color_count == 3:
        accent = colors[2]
        if not _add_accent_color(grid, occupied, accent, rng):
            return None

    return grid if _input_meets_spec(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gated pixel-scale ARC input")
