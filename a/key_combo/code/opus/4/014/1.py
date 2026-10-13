from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

HALO_BY_HOLE_COUNT = {0: BLUE, 1: RED, 2: GREEN}

GENERATION_RETRIES = 400
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 600


@dataclass(frozen=True)
class Placement:
    shape: np.ndarray
    hole_count: int
    top: int
    left: int


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _hole_regions(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            touches_boundary = r in (0, h - 1) or c in (0, w - 1)
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_boundary = True
                        queue.append((nr, nc))
            if not touches_boundary:
                holes.append(np.array(coords, dtype=int))
    return holes


def _hole_count(mask: np.ndarray) -> int:
    return len(_hole_regions(mask))


def _single_component(mask: np.ndarray) -> bool:
    return len(_components_from_mask(mask)) == 1


def _component_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each orange object's one-cell halo according to how many enclosed holes it contains.

    Input structure:
    - The grid is a black background with 4-connected orange objects.
    - Objects are separated so their one-cell expanded bounding boxes do not overlap.
    - Some objects are solid, some enclose one black hole, and some enclose two black holes.

    Rule:
    - Find every orange 4-connected component and count the black 4-connected regions fully enclosed inside its tight bbox.
    - Map hole counts to halo colors with the fixed codebook 0 -> blue, 1 -> red, 2 -> green.
    - Expand each component's tight bbox by one cell in every direction, clip to the canvas, and recolor only black pixels in that region to the halo color.
    - Keep the orange component unchanged and leave enclosed hole pixels black.

    Key insight:
    - The halo color is determined by a topological invariant of the component, not by its size, position, or silhouette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for coords in _components_from_mask(g == ORANGE):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

        holes = _hole_regions(local_mask)
        hole_count = len(holes)
        assert hole_count in HALO_BY_HOLE_COUNT, "unexpected hole count"

        pr0 = max(0, r0 - 1)
        pc0 = max(0, c0 - 1)
        pr1 = min(g.shape[0] - 1, r1 + 1)
        pc1 = min(g.shape[1] - 1, c1 + 1)

        region_in = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
        region_out = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        hole_mask = np.zeros(region_in.shape, dtype=bool)
        row_offset = r0 - pr0
        col_offset = c0 - pc0
        for hole in holes:
            hole_mask[hole[:, 0] + row_offset, hole[:, 1] + col_offset] = True

        fill_mask = (region_in == BLACK) & ~hole_mask
        region_out[fill_mask] = HALO_BY_HOLE_COUNT[hole_count]

    return out


def _shape_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(list(cells), dtype=int)
    coords[:, 0] -= coords[:, 0].min()
    coords[:, 1] -= coords[:, 1].min()
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, k=random.randrange(4))
    return _trim_mask(out)


def _exterior_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))
    return exterior


def _add_bumps(mask: np.ndarray, target_holes: int, max_bumps: int = 2) -> np.ndarray:
    working = np.pad(mask.astype(bool), 1)
    for _ in range(random.randint(0, max_bumps)):
        exterior = _exterior_background(working)
        candidates = []
        for r, c in np.argwhere(exterior):
            if any(
                0 <= nr < working.shape[0]
                and 0 <= nc < working.shape[1]
                and working[nr, nc]
                for nr, nc in _neighbors4(int(r), int(c))
            ):
                candidates.append((int(r), int(c)))
        random.shuffle(candidates)

        placed = False
        for r, c in candidates:
            trial = working.copy()
            trial[r, c] = True
            trimmed = _trim_mask(trial)
            if _single_component(trimmed) and _hole_count(trimmed) == target_holes:
                working = trial
                placed = True
                break
        if not placed:
            break
    return _trim_mask(working)


def _random_solid_shape() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        area = random.randint(3, 8)
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < area:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(*cell):
                if nb not in cells:
                    frontier.add(nb)
        mask = _random_transform(_shape_from_cells(cells))
        if _single_component(mask) and _hole_count(mask) == 0:
            return mask
    raise RuntimeError("failed to build a solid shape")


def _one_hole_shape() -> np.ndarray:
    hole_h = random.randint(1, 2)
    hole_w = random.randint(1, 2)
    mask = np.ones((hole_h + 2, hole_w + 2), dtype=bool)
    mask[1:-1, 1:-1] = False
    mask = _add_bumps(mask, target_holes=1, max_bumps=2)
    mask = _random_transform(mask)
    assert _single_component(mask) and _hole_count(mask) == 1
    return mask


def _two_hole_shape() -> np.ndarray:
    if random.random() < 0.5:
        hole_w = random.randint(1, 2)
        top_h = random.randint(1, 2)
        bottom_h = random.randint(1, 2)
        height = top_h + bottom_h + 3
        width = hole_w + 2
        mask = np.ones((height, width), dtype=bool)
        mask[1 : 1 + top_h, 1:-1] = False
        split = 1 + top_h
        mask[split + 1 : split + 1 + bottom_h, 1:-1] = False
    else:
        hole_h = random.randint(1, 2)
        left_w = random.randint(1, 2)
        right_w = random.randint(1, 2)
        height = hole_h + 2
        width = left_w + right_w + 3
        mask = np.ones((height, width), dtype=bool)
        mask[1:-1, 1 : 1 + left_w] = False
        split = 1 + left_w
        mask[1:-1, split + 1 : split + 1 + right_w] = False
    mask = _add_bumps(mask, target_holes=2, max_bumps=2)
    mask = _random_transform(mask)
    assert _single_component(mask) and _hole_count(mask) == 2
    return mask


def _random_shape_for_holes(hole_count: int) -> np.ndarray:
    if hole_count == 0:
        return _random_solid_shape()
    if hole_count == 1:
        return _one_hole_shape()
    if hole_count == 2:
        return _two_hole_shape()
    raise ValueError(f"unsupported hole count: {hole_count}")


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_box(shape: np.ndarray, top: int, left: int) -> tuple[int, int, int, int]:
    h, w = shape.shape
    return top - 1, left - 1, top + h, left + w


def _choose_grid_shape(shapes: list[np.ndarray]) -> tuple[int, int] | None:
    padded_area = sum((shape.shape[0] + 2) * (shape.shape[1] + 2) for shape in shapes)
    base = max(18, int(np.ceil((padded_area * random.uniform(1.45, 1.7)) ** 0.5)))
    min_dim = max(base, max(max(shape.shape) for shape in shapes) + 3)
    if min_dim > 28:
        return None
    max_dim = min(28, min_dim + 4)
    return random.randint(min_dim, max_dim), random.randint(min_dim, max_dim)


def _place_shapes(shapes: list[np.ndarray], hole_counts: list[int], shape: tuple[int, int]) -> list[Placement] | None:
    height, width = shape
    order = sorted(range(len(shapes)), key=lambda i: (shapes[i].shape[0] + 2) * (shapes[i].shape[1] + 2), reverse=True)
    boxes: list[tuple[int, int, int, int]] = []
    placements: list[Placement | None] = [None] * len(shapes)

    for idx in order:
        sprite = shapes[idx]
        h, w = sprite.shape
        if h + 2 > height or w + 2 > width:
            return None
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(1, height - h - 1)
            left = random.randint(1, width - w - 1)
            box = _expanded_box(sprite, top, left)
            if any(_boxes_overlap(box, other) for other in boxes):
                continue
            boxes.append(box)
            placements[idx] = Placement(sprite.copy(), int(hole_counts[idx]), int(top), int(left))
            break
        else:
            return None

    return [placement for placement in placements if placement is not None]


def _render_input(shape: tuple[int, int], placements: list[Placement]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for placement in placements:
        h, w = placement.shape.shape
        region = grid[placement.top : placement.top + h, placement.left : placement.left + w]
        region[placement.shape] = ORANGE
    return grid


def _expected_output_from_metadata(grid: np.ndarray, placements: list[Placement]) -> np.ndarray:
    out = grid.copy()
    for placement in placements:
        pr0, pc0, pr1, pc1 = _expanded_box(placement.shape, placement.top, placement.left)
        region_in = grid[pr0 : pr1 + 1, pc0 : pc1 + 1]
        region_out = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        fill_mask = region_in == BLACK
        hole_mask = np.zeros(region_in.shape, dtype=bool)
        row_offset = placement.top - pr0
        col_offset = placement.left - pc0
        for hole in _hole_regions(placement.shape):
            hole_mask[hole[:, 0] + row_offset, hole[:, 1] + col_offset] = True
        region_out[fill_mask & ~hole_mask] = HALO_BY_HOLE_COUNT[placement.hole_count]
    return out


def _valid_input_semantics(grid: np.ndarray, placements: list[Placement]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.all((grid == BLACK) | (grid == ORANGE)):
        return False
    if not (4 <= len(placements) <= 6):
        return False
    if {placement.hole_count for placement in placements} != {0, 1, 2}:
        return False

    boxes = [_expanded_box(placement.shape, placement.top, placement.left) for placement in placements]
    for i in range(len(boxes)):
        r0, c0, r1, c1 = boxes[i]
        if r0 < 0 or c0 < 0 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
            return False
        for j in range(i + 1, len(boxes)):
            if _boxes_overlap(boxes[i], boxes[j]):
                return False

    components = _components_from_mask(grid == ORANGE)
    actual_hole_counts = sorted(_hole_count(_component_mask_from_coords(coords)) for coords in components)
    expected_hole_counts = sorted(placement.hole_count for placement in placements)
    if actual_hole_counts != expected_hole_counts:
        return False

    expected = _expected_output_from_metadata(grid, placements)
    actual = main(grid)
    return np.array_equal(actual, expected)


def _generate_attempt() -> np.ndarray | None:
    n_objects = random.randint(4, 6)
    hole_counts = [0, 1, 2] + [random.choice((0, 1, 2)) for _ in range(n_objects - 3)]
    random.shuffle(hole_counts)

    shapes = [_random_shape_for_holes(hole_count) for hole_count in hole_counts]
    canvas_shape = _choose_grid_shape(shapes)
    if canvas_shape is None:
        return None

    placements = _place_shapes(shapes, hole_counts, canvas_shape)
    if placements is None:
        return None

    grid = _render_input(canvas_shape, placements)
    if not _valid_input_semantics(grid, placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
