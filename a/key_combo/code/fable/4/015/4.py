from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREEN = 3
TARGET_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)

GRID_MIN_SIZE = 16
GRID_MAX_SIZE = 30
RECT_MIN_HEIGHT = 7
RECT_MAX_HEIGHT = 14
RECT_MIN_WIDTH = 8
RECT_MAX_WIDTH = 16
SHAPE_BBOX_MAX = 4
SHAPE_AREAS = tuple(range(2, 8))
CAVITY_AREAS = tuple(range(3, 7))
MATCH_COUNT_RANGE = (1, 4)
SAME_AREA_DISTRACTOR_RANGE = (1, 3)
EXTRA_DISTRACTOR_RANGE = (0, 2)
GEN_RETRY_LIMIT = 400

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
]

Shape = tuple[tuple[int, int], ...]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> Shape:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _shape_hw(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _shape_mask(shape: Shape, color: int = 1) -> np.ndarray:
    h, w = _shape_hw(shape)
    mask = np.zeros((h, w), dtype=int)
    for r, c in shape:
        mask[r, c] = int(color)
    return mask


def _shape_coords(shape: Shape, r0: int, c0: int) -> np.ndarray:
    return np.array([(r0 + r, c0 + c) for r, c in shape], dtype=int)


def _connected(shape: Shape, connectivity: int) -> bool:
    cells = set(shape)
    stack = [next(iter(cells))]
    seen = {stack[0]}
    deltas = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8
    while stack:
        r, c = stack.pop()
        for dr, dc in deltas:
            nbr = (r + dr, c + dc)
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(cells)


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    deltas = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8
    components: list[np.ndarray] = []
    height, width = mask.shape

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _dihedral_variants(shape: Shape) -> list[Shape]:
    pts = np.asarray(shape, dtype=int)
    variants: list[Shape] = []
    seen: set[Shape] = set()

    for rotation in range(4):
        transformed = pts.copy()
        for _ in range(rotation):
            transformed = np.column_stack((-transformed[:, 1], transformed[:, 0]))
        for reflected in (False, True):
            candidate = transformed.copy()
            if reflected:
                candidate[:, 1] = -candidate[:, 1]
            normalized = _normalize(candidate)
            if normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)

    return variants


def _shape_catalog(connectivity: int) -> dict[int, list[Shape]]:
    cells = [(r, c) for r in range(SHAPE_BBOX_MAX) for c in range(SHAPE_BBOX_MAX)]
    by_area = {area: [] for area in SHAPE_AREAS}
    seen: dict[int, set[Shape]] = {area: set() for area in SHAPE_AREAS}

    for area in SHAPE_AREAS:
        for combo in itertools.combinations(cells, area):
            shape = _normalize(np.array(combo, dtype=int))
            if shape in seen[area]:
                continue
            h, w = _shape_hw(shape)
            if h > SHAPE_BBOX_MAX or w > SHAPE_BBOX_MAX:
                continue
            if _connected(shape, connectivity):
                seen[area].add(shape)
                by_area[area].append(shape)

    return by_area


SHAPES4 = _shape_catalog(connectivity=4)
SHAPES8 = _shape_catalog(connectivity=8)
CAVITY_SHAPES = [
    shape
    for area in CAVITY_AREAS
    for shape in SHAPES4[area]
    if min(_shape_hw(shape)) >= 2 and len(_dihedral_variants(shape)) >= 2
]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the rectangle's hidden cavity as the template and recolor matching green components.

    Input structure:
    - A black background contains one non-green rectangular block with a single 4-connected black cavity strictly inside it.
    - All other foreground pixels are green and form separated 8-connected components.
    - Some green components match the cavity shape only after rotation or reflection, while others are distractors.

    Rule:
    - Extract the cavity mask, normalize it inside its own bounding box, and enumerate all unique rotations and reflections.
    - Compare every green 8-connected component by its normalized shape against those template variants.
    - Recolor matching green components to the rectangle color, erase the rest, and fill the original cavity with that same color.

    Key insight:
    The shape key is not drawn as foreground at all; it is the negative space carved out of the rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = sorted(set(map(int, g.ravel())) - {BLACK})
    assert GREEN in colors, "expected green components"
    target_colors = [color for color in colors if color != GREEN]
    assert len(target_colors) == 1, "expected exactly one non-green foreground color"
    target_color = int(target_colors[0])

    rect_mask = g == target_color
    rect_components = _components(rect_mask, connectivity=4)
    assert len(rect_components) == 1, "expected one rectangular block"
    rect_coords = rect_components[0]
    r0, c0, r1, c1 = _bbox_coords(rect_coords)

    rect_patch = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((rect_patch == target_color) | (rect_patch == BLACK)), "block bbox must contain only the block and its cavity"

    cavity_mask = rect_patch == BLACK
    assert cavity_mask.any(), "expected one cavity"
    assert not cavity_mask[0].any() and not cavity_mask[-1].any()
    assert not cavity_mask[:, 0].any() and not cavity_mask[:, -1].any()

    cavity_components = _components(cavity_mask, connectivity=4)
    assert len(cavity_components) == 1, "expected exactly one cavity"
    cavity_coords = cavity_components[0]
    template = _normalize(cavity_coords)
    variants = set(_dihedral_variants(template))

    out = g.copy()
    green_mask = g == GREEN
    out[green_mask] = BLACK

    green_components = _components(green_mask, connectivity=8)
    assert green_components, "expected at least one green component"
    for comp in green_components:
        if _normalize(comp) in variants:
            out[comp[:, 0], comp[:, 1]] = target_color

    cavity_global = cavity_coords + np.array([r0, c0], dtype=int)
    out[cavity_global[:, 0], cavity_global[:, 1]] = target_color
    return out


def _sample_match_shapes(template: Shape, count: int) -> list[Shape]:
    variants = _dihedral_variants(template)
    non_template = [shape for shape in variants if shape != template]
    matches: list[Shape] = []

    if count > 0 and non_template and random.random() < 0.85:
        matches.append(random.choice(non_template))
        count -= 1

    for _ in range(count):
        matches.append(random.choice(variants))

    random.shuffle(matches)
    return matches


def _sample_same_area_distractors(template: Shape, count: int) -> list[Shape]:
    variants = set(_dihedral_variants(template))
    area = len(template)
    candidates = [shape for shape in SHAPES8[area] if shape not in variants]
    assert candidates, "expected at least one same-area distractor candidate"
    if count <= len(candidates):
        return random.sample(candidates, count)
    return [random.choice(candidates) for _ in range(count)]


def _sample_extra_distractors(template_area: int, count: int) -> list[Shape]:
    areas = [area for area in SHAPE_AREAS if area != template_area]
    shapes: list[Shape] = []
    for _ in range(count):
        area = random.choice(areas)
        shapes.append(random.choice(SHAPES8[area]))
    return shapes


def _can_place_shape(grid: np.ndarray, shape: Shape, r0: int, c0: int, *, border: int = 1) -> bool:
    height, width = grid.shape
    h, w = _shape_hw(shape)
    if r0 < border or c0 < border or r0 + h > height - border or c0 + w > width - border:
        return False

    cells = {(r0 + r, c0 + c) for r, c in shape}
    for rr, cc in cells:
        if int(grid[rr, cc]) != BLACK:
            return False
        for dr, dc in NEIGHBORS8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width:
                if (nr, nc) not in cells and int(grid[nr, nc]) != BLACK:
                    return False
    return True


def _placement_options(grid: np.ndarray, shape: Shape) -> list[tuple[int, int]]:
    h, w = _shape_hw(shape)
    height, width = grid.shape
    options: list[tuple[int, int]] = []
    for r0 in range(1, height - h):
        for c0 in range(1, width - w):
            if _can_place_shape(grid, shape, r0, c0):
                options.append((r0, c0))
    return options


def _paint_shape(grid: np.ndarray, shape: Shape, r0: int, c0: int, color: int) -> np.ndarray:
    coords = _shape_coords(shape, r0, c0)
    grid[coords[:, 0], coords[:, 1]] = int(color)
    return coords


def _choose_rectangle_dims(template: Shape, height: int, width: int) -> tuple[int, int] | None:
    th, tw = _shape_hw(template)
    min_h = max(RECT_MIN_HEIGHT, th + 3)
    min_w = max(RECT_MIN_WIDTH, tw + 3)
    max_h = min(RECT_MAX_HEIGHT, height - 2)
    max_w = min(RECT_MAX_WIDTH, width - 2)
    if min_h > max_h or min_w > max_w:
        return None
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    width = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    template = random.choice(CAVITY_SHAPES)
    dims = _choose_rectangle_dims(template, height, width)
    if dims is None:
        return None
    rect_h, rect_w = dims
    th, tw = _shape_hw(template)

    grid = np.full((height, width), BLACK, dtype=int)
    target_color = random.choice(TARGET_COLORS)

    rect_r0 = random.randint(1, height - rect_h - 1)
    rect_c0 = random.randint(1, width - rect_w - 1)
    rect_r1 = rect_r0 + rect_h - 1
    rect_c1 = rect_c0 + rect_w - 1
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = target_color

    hole_r0 = random.randint(rect_r0 + 1, rect_r1 - th)
    hole_c0 = random.randint(rect_c0 + 1, rect_c1 - tw)
    cavity_coords = _shape_coords(template, hole_r0, hole_c0)
    grid[cavity_coords[:, 0], cavity_coords[:, 1]] = BLACK

    match_count = random.randint(*MATCH_COUNT_RANGE)
    same_area_count = random.randint(*SAME_AREA_DISTRACTOR_RANGE)
    extra_count = random.randint(*EXTRA_DISTRACTOR_RANGE)

    objects = [
        {"kind": "match", "shape": shape}
        for shape in _sample_match_shapes(template, match_count)
    ]
    objects += [
        {"kind": "distractor", "shape": shape}
        for shape in _sample_same_area_distractors(template, same_area_count)
    ]
    objects += [
        {"kind": "distractor", "shape": shape}
        for shape in _sample_extra_distractors(len(template), extra_count)
    ]
    random.shuffle(objects)
    objects.sort(
        key=lambda obj: (_shape_hw(obj["shape"])[0] * _shape_hw(obj["shape"])[1], len(obj["shape"])),
        reverse=True,
    )

    placed: list[tuple[str, np.ndarray]] = []
    for obj in objects:
        options = _placement_options(grid, obj["shape"])
        if not options:
            return None
        r0, c0 = random.choice(options)
        coords = _paint_shape(grid, obj["shape"], r0, c0, GREEN)
        placed.append((obj["kind"], coords))

    expected = grid.copy()
    expected[expected == GREEN] = BLACK
    for kind, coords in placed:
        if kind == "match":
            expected[coords[:, 0], coords[:, 1]] = target_color
    expected[cavity_coords[:, 0], cavity_coords[:, 1]] = target_color

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
