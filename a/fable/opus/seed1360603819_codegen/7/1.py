from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components_by_color(scene: np.ndarray) -> dict[int, list[list[tuple[int, int]]]]:
    """Return 4-connected components for each non-black color in the scene."""
    h, w = scene.shape
    seen = np.zeros((h, w), dtype=bool)
    groups: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)

    for r in range(h):
        for c in range(w):
            color = int(scene[r, c])
            if color == BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(scene[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            groups[color].append(comp)
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each scene color by using its object count as a 1-based index into the left ruler.

    Input structure:
    - Column 0 is a vertical ruler: several distinct non-black pixels ordered from top to bottom.
    - Column 1 is all black and separates the ruler from the scene.
    - Columns 2.. are the black-background scene with monochrome 4-connected objects.

    Rule:
    - For each non-black scene color, count how many 4-connected objects of that color appear.
    - Use that count k as a 1-based index into the ruler.
    - Repaint every pixel of that scene color with the kth ruler color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 3

    ruler = [int(g[r, 0]) for r in range(g.shape[0]) if int(g[r, 0]) != BLACK]
    assert ruler, "expected at least one ruler color"
    assert len(ruler) == len(set(ruler)), "ruler colors must be unique"
    assert np.all(g[:, 1] == BLACK), "expected a black separator column"

    scene = g[:, 2:]
    by_color = _components_by_color(scene)

    out = g.copy()
    out_scene = out[:, 2:]
    for color, comps in by_color.items():
        count = len(comps)
        assert 1 <= count <= len(ruler), "scene color count must fit inside the ruler"
        out_scene[scene == color] = ruler[count - 1]
    return out


# --------------- generator ---------------

GEN_MAX_ATTEMPTS = 400
RULER_LEN_RANGE = (3, 6)
SCENE_COLOR_COUNT_RANGE = (2, 4)
MAX_TOTAL_OBJECTS = 10
GRID_HEIGHT_RANGE = (8, 16)
SCENE_WIDTH_RANGE = (8, 20)
SHAPE_AREA_RANGE = (1, 4)
PLACEMENT_ATTEMPTS = 250


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _random_shape() -> list[tuple[int, int]]:
    """Grow a small connected polyomino to keep object shapes varied."""
    area = random.randint(*SHAPE_AREA_RANGE)
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if -3 <= nr <= 3 and -3 <= nc <= 3 and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
    return _normalize(cells)


def _shape_size(shape: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _sample_counts(scene_color_count: int, ruler_len: int) -> list[int]:
    prefer_duplicates = scene_color_count >= 3 and random.random() < 0.5
    for _ in range(200):
        counts = [random.randint(1, ruler_len) for _ in range(scene_color_count)]
        if sum(counts) < scene_color_count + 1 or sum(counts) > MAX_TOTAL_OBJECTS:
            continue
        if len(set(counts)) == 1:
            continue
        if prefer_duplicates and len(set(counts)) == scene_color_count:
            continue
        if not prefer_duplicates and scene_color_count <= ruler_len and len(set(counts)) != scene_color_count:
            continue
        return counts

    if scene_color_count == 2:
        return [1, min(2, ruler_len)]
    if scene_color_count == 3:
        return [1, 2, 2] if ruler_len >= 2 else [1, 1, 2]
    return [1, 2, 2, 3]


def _choose_profile() -> tuple[list[int], list[int], list[int]]:
    ruler_len = random.randint(*RULER_LEN_RANGE)
    max_scene_colors = min(SCENE_COLOR_COUNT_RANGE[1], ruler_len, len(PALETTE) - ruler_len)
    scene_color_count = random.randint(SCENE_COLOR_COUNT_RANGE[0], max_scene_colors)

    ruler_and_scene = random.sample(PALETTE, ruler_len + scene_color_count)
    ruler_colors = ruler_and_scene[:ruler_len]
    scene_colors = ruler_and_scene[ruler_len:]
    counts = _sample_counts(scene_color_count, ruler_len)
    return ruler_colors, scene_colors, counts


def _place_objects(
    height: int,
    width: int,
    objects: list[tuple[int, list[tuple[int, int]]]],
) -> np.ndarray | None:
    scene = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    # Place larger footprints first to reduce rejection failures.
    ordered = sorted(objects, key=lambda item: len(item[1]), reverse=True)
    for color, shape in ordered:
        sh, sw = _shape_size(shape)
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - sh)
            c0 = random.randint(0, width - sw)
            cells = [(r0 + r, c0 + c) for r, c in shape]
            if any(blocked[r, c] for r, c in cells):
                continue
            for r, c in cells:
                scene[r, c] = color
            for r, c in cells:
                for rr in range(max(0, r - 1), min(height, r + 2)):
                    for cc in range(max(0, c - 1), min(width, c + 2)):
                        blocked[rr, cc] = True
            placed = True
            break
        if not placed:
            return None
    return scene


def _build_grid(ruler_colors: list[int], scene: np.ndarray) -> np.ndarray:
    h, w = scene.shape
    grid = np.zeros((h, w + 2), dtype=int)
    for r, color in enumerate(ruler_colors):
        grid[r, 0] = color
    grid[:, 2:] = scene
    return grid


def _is_clear_scene(scene: np.ndarray) -> bool:
    h, w = scene.shape
    for r in range(h):
        for c in range(w):
            color = int(scene[r, c])
            if color == BLACK:
                continue
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    neighbor = int(scene[rr, cc])
                    if (rr, cc) != (r, c) and neighbor != BLACK and neighbor != color:
                        return False
    return True


def _valid_candidate(grid: np.ndarray, ruler_colors: list[int], scene_colors: list[int], counts: list[int]) -> bool:
    if grid.shape[1] < 3 or not np.all(grid[:, 1] == BLACK):
        return False
    if any(int(grid[r, 0]) != color for r, color in enumerate(ruler_colors)):
        return False
    if np.any(grid[len(ruler_colors) :, 0] != BLACK):
        return False

    scene = grid[:, 2:]
    if not _is_clear_scene(scene):
        return False

    comp_map = _components_by_color(scene)
    if sorted(comp_map) != sorted(scene_colors):
        return False
    actual_counts = [len(comp_map[color]) for color in scene_colors]
    if actual_counts != counts:
        return False
    if len(set(actual_counts)) < 2:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    for color, count in zip(scene_colors, counts):
        target = ruler_colors[count - 1]
        mask = scene == color
        if not mask.any():
            return False
        if np.any(out[:, 2:][mask] != target):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    ruler_colors, scene_colors, counts = _choose_profile()
    total_objects = sum(counts)

    min_height = max(GRID_HEIGHT_RANGE[0], len(ruler_colors))
    height = random.randint(min_height, GRID_HEIGHT_RANGE[1])
    min_scene_width = min(SCENE_WIDTH_RANGE[1], max(SCENE_WIDTH_RANGE[0], 6 + total_objects))
    if min_scene_width > SCENE_WIDTH_RANGE[1]:
        return None
    width = random.randint(min_scene_width, SCENE_WIDTH_RANGE[1])

    objects: list[tuple[int, list[tuple[int, int]]]] = []
    for color, count in zip(scene_colors, counts):
        for _ in range(count):
            objects.append((color, _random_shape()))

    scene = _place_objects(height, width, objects)
    if scene is None:
        return None

    grid = _build_grid(ruler_colors, scene)
    if not _valid_candidate(grid, ruler_colors, scene_colors, counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_MAX_ATTEMPTS} attempts")
