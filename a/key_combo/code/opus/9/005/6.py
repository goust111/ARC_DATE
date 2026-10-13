from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
INPUT_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _trim_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BACKGROUND)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_connected_patch(color: int, area: int, max_side: int) -> np.ndarray:
    color = int(color)
    area = int(area)
    feasible = []
    for h in range(1, max_side + 1):
        for w in range(1, max_side + 1):
            box_area = h * w
            if box_area < area:
                continue
            if box_area > area + 4:
                continue
            if area >= 4 and min(h, w) == 1 and random.random() < 0.7:
                continue
            feasible.append((h, w))
    if not feasible:
        feasible = [(area, 1), (1, area)]

    height, width = random.choice(feasible)
    start = (random.randrange(height), random.randrange(width))
    coords = {start}

    while len(coords) < area:
        frontier = []
        for r, c in coords:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in coords:
                    frontier.append((nr, nc))
        coords.add(random.choice(frontier))

    patch = np.zeros((height, width), dtype=int)
    for r, c in coords:
        patch[r, c] = color
    patch = _trim_patch(patch)

    if random.random() < 0.5:
        patch = np.flipud(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    patch = np.rot90(patch, random.randrange(4))
    return _trim_patch(np.asarray(patch, dtype=int))


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    height, width = grid.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > height or c0 + pw > width:
        return False

    coords = np.argwhere(patch != BACKGROUND)
    for dr, dc in coords:
        r = int(r0 + dr)
        c = int(c0 + dc)
        if grid[r, c] != BACKGROUND:
            return False
        rr0 = max(0, r - padding)
        rr1 = min(height, r + padding + 1)
        cc0 = max(0, c - padding)
        cc1 = min(width, c + padding + 1)
        if np.any(grid[rr0:rr1, cc0:cc1] != BACKGROUND):
            return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, padding: int) -> bool:
    height, width = grid.shape
    ph, pw = patch.shape
    candidates = [(r0, c0) for r0 in range(height - ph + 1) for c0 in range(width - pw + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if not _can_place_patch(grid, patch, r0, c0, padding):
            continue
        mask = patch != BACKGROUND
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        return True
    return False


def _analyze(grid: np.ndarray) -> dict:
    grid = np.asarray(grid, dtype=int)
    clean = np.zeros_like(grid)
    valid_colors: list[int] = []
    distractor_colors: list[int] = []
    component_counts: dict[int, int] = {}
    component_areas: dict[int, list[int]] = {}
    total_areas: dict[int, int] = {}

    for color in _colors_present(grid, exclude={BACKGROUND}):
        comps = _components_mask(grid == color)
        component_counts[color] = len(comps)
        component_areas[color] = [int(comp.shape[0]) for comp in comps]
        total_areas[color] = int(sum(component_areas[color]))
        if len(comps) == 1:
            valid_colors.append(color)
            coords = comps[0]
            clean[coords[:, 0], coords[:, 1]] = color
        else:
            distractor_colors.append(color)

    return {
        "clean": clean,
        "valid_colors": valid_colors,
        "distractor_colors": distractor_colors,
        "component_counts": component_counts,
        "component_areas": component_areas,
        "total_areas": total_areas,
    }


def _build_output(clean: np.ndarray, border_width: int) -> np.ndarray:
    clean = np.asarray(clean, dtype=int)
    side_h, side_w = clean.shape
    assert side_h == side_w, "this task uses square input grids"
    k = int(border_width)
    out_side = 2 * side_h + 3 * k
    out = np.full((out_side, out_side), FRAME_COLOR, dtype=int)

    rot0 = clean
    rot90 = np.rot90(clean, -1)
    rot180 = np.rot90(clean, 2)
    rot270 = np.rot90(clean, 1)

    out[k : k + side_h, k : k + side_w] = rot0
    out[k : k + side_h, 2 * k + side_w : 2 * k + 2 * side_w] = rot90
    out[2 * k + side_h : 2 * k + 2 * side_h, k : k + side_w] = rot270
    out[2 * k + side_h : 2 * k + 2 * side_h, 2 * k + side_w : 2 * k + 2 * side_w] = rot180
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only single-component colors, then rotate the cleaned square into a four-panel collage whose gray border width equals the number of kept colors.

    Input structure:
    - The input is a square black-background grid with 3 to 5 non-black colors.
    - Each non-black color forms one or more 4-connected monochrome components.
    - Some colors appear as exactly one component, while at least one distractor color is split across multiple components.

    Rule:
    - Count the 4-connected components of every non-black color.
    - Preserve colors with exactly one component and erase every color with multiple components to black, producing a cleaned grid G'.
    - Let K be the number of preserved colors.
    - Rotate G' by 0, 90, 180, and 270 degrees clockwise, place them in a 2x2 arrangement, and separate the copies plus the outer margin with gray strips of width K.

    Key insight:
    - The same component-count analysis has two consequences: it filters the visible content and also determines the geometric border thickness used in the final rotated collage.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square grid"

    analysis = _analyze(g)
    clean = analysis["clean"]
    valid_colors = analysis["valid_colors"]
    assert 1 <= len(valid_colors) <= 3, "expected between 1 and 3 preserved colors"
    assert analysis["distractor_colors"], "expected at least one distractor color"

    return _build_output(clean, len(valid_colors))


# --------------- generator ---------------

GRID_SIDE_RANGE = (8, 10)
VALID_COLOR_COUNT_RANGE = (1, 3)
MAX_TOTAL_COLORS = 5
MIN_TOTAL_COLORS = 3
VALID_AREA_RANGE = (3, 7)
DISTRACTOR_COMPONENT_COUNT_CHOICES = (2, 2, 3)
DISTRACTOR_FRAGMENT_MAX_AREA = 3
PLACEMENT_PADDING = 1
GEN_ATTEMPTS = 500


def _sample_valid_areas(k: int) -> list[int]:
    return [random.randint(*VALID_AREA_RANGE) for _ in range(k)]


def _sample_distractor_sizes(total_area: int, component_count: int) -> list[int]:
    sizes = [1] * component_count
    remaining = total_area - component_count
    while remaining > 0:
        idx = random.randrange(component_count)
        if sizes[idx] >= DISTRACTOR_FRAGMENT_MAX_AREA:
            continue
        sizes[idx] += 1
        remaining -= 1
    random.shuffle(sizes)
    return sizes


def _rotation_distinct(clean: np.ndarray) -> bool:
    return all(not np.array_equal(clean, np.rot90(clean, k)) for k in (1, 2, 3))


def _semantic_ok(grid: np.ndarray) -> bool:
    analysis = _analyze(grid)
    valid_colors = analysis["valid_colors"]
    distractor_colors = analysis["distractor_colors"]
    clean = analysis["clean"]
    foreground_colors = _colors_present(grid, exclude={BACKGROUND})

    if not (MIN_TOTAL_COLORS <= len(foreground_colors) <= MAX_TOTAL_COLORS):
        return False
    if not (1 <= len(valid_colors) <= 3):
        return False
    if not distractor_colors:
        return False
    if any(analysis["component_counts"][color] != 1 for color in valid_colors):
        return False
    if any(analysis["component_counts"][color] < 2 for color in distractor_colors):
        return False
    if any(analysis["component_areas"][color][0] < 3 for color in valid_colors):
        return False
    if np.array_equal(clean, grid):
        return False
    if not _rotation_distinct(clean):
        return False

    valid_areas = [analysis["total_areas"][color] for color in valid_colors]
    distractor_areas = [analysis["total_areas"][color] for color in distractor_colors]
    if max(distractor_areas) < min(valid_areas):
        return False
    if max(valid_areas) < min(distractor_areas):
        return False

    expected = _build_output(clean, len(valid_colors))
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if actual.shape[0] > 30 or actual.shape[1] > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    side = random.randint(*GRID_SIDE_RANGE)
    grid = np.zeros((side, side), dtype=int)

    k = random.randint(*VALID_COLOR_COUNT_RANGE)
    min_distractors = max(1, MIN_TOTAL_COLORS - k)
    max_distractors = MAX_TOTAL_COLORS - k
    distractor_count = random.randint(min_distractors, max_distractors)
    total_colors = k + distractor_count
    colors = random.sample(INPUT_COLOR_POOL, total_colors)
    valid_colors = colors[:k]
    distractor_colors = colors[k:]

    valid_areas = _sample_valid_areas(k)
    max_valid_area = max(valid_areas)
    min_valid_area = min(valid_areas)

    distractor_specs: list[tuple[int, list[int]]] = []
    for color in distractor_colors:
        component_count = random.choice(DISTRACTOR_COMPONENT_COUNT_CHOICES)
        lo = max(component_count, min_valid_area - 1)
        hi = max(lo, min(component_count * DISTRACTOR_FRAGMENT_MAX_AREA, max_valid_area + 1))
        total_area = random.randint(lo, hi)
        sizes = _sample_distractor_sizes(total_area, component_count)
        distractor_specs.append((color, sizes))

    pieces: list[tuple[int, np.ndarray]] = []
    max_patch_side = min(5, side - 2)
    for color, area in zip(valid_colors, valid_areas):
        pieces.append((area, _random_connected_patch(color, area, max_patch_side)))
    for color, sizes in distractor_specs:
        for size in sizes:
            pieces.append((size, _random_connected_patch(color, size, 3)))

    pieces.sort(key=lambda item: (-item[0], random.random()))
    for _area, patch in pieces:
        if not _place_patch(grid, patch, PLACEMENT_PADDING):
            return None

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
