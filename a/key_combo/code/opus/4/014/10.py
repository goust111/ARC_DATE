from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "new/solvers/arc_utils.py").is_file():
        _ROOT = _parent
        break
if _ROOT is None:
    raise ImportError("could not locate new/solvers/arc_utils.py from this file")
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from new.solvers.arc_utils import Color, bbox_coords, components_mask


BACKGROUND = Color.BLACK
OBJECT_COLOR = Color.GREY
MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
GENERATION_RETRIES = 500
GRID_SIDE_MIN = 13
GRID_SIDE_MAX = 18
OBJECT_COUNT_CHOICES = (3, 4)
OBJECT_MARGIN = 2
OBJECT_AREA_MIN = 4
OBJECT_AREA_MAX = 10
OBJECT_BBOX_MAX = 4


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _label_gray_objects(grid: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    comps = components_mask(grid == OBJECT_COLOR, connectivity=4)
    labels = np.full(grid.shape, -1, dtype=int)
    for idx, coords in enumerate(comps):
        labels[coords[:, 0], coords[:, 1]] = idx
    return comps, labels


def _trace_first_hit(labels: np.ndarray, start_r: int, start_c: int, dr: int, dc: int) -> int | None:
    h, w = labels.shape
    r, c = start_r + dr, start_c + dc
    while 0 <= r < h and 0 <= c < w:
        idx = int(labels[r, c])
        if idx >= 0:
            return idx
        r += dr
        c += dc
    return None


def _iter_boundary_markers(grid: np.ndarray):
    h, w = grid.shape
    for c in range(1, w - 1):
        top_color = int(grid[0, c])
        if top_color != BACKGROUND:
            yield 0, c, top_color, 1, 0
        bottom_color = int(grid[h - 1, c])
        if bottom_color != BACKGROUND:
            yield h - 1, c, bottom_color, -1, 0
    for r in range(1, h - 1):
        left_color = int(grid[r, 0])
        if left_color != BACKGROUND:
            yield r, 0, left_color, 0, 1
        right_color = int(grid[r, w - 1])
        if right_color != BACKGROUND:
            yield r, w - 1, right_color, 0, -1


def _recolor_from_hit_set(colors: set[int]) -> int:
    assert colors, "each object must be hit by at least one marker"
    if len(colors) == 1:
        return next(iter(colors))
    if len(colors) == 2:
        return Color.ORANGE
    return Color.PINK


def _solve_by_projection(grid: np.ndarray) -> tuple[list[np.ndarray], list[set[int]]]:
    g = np.asarray(grid, dtype=int)
    comps, labels = _label_gray_objects(g)
    hits = [set() for _ in comps]
    for r, c, color, dr, dc in _iter_boundary_markers(g):
        assert color in MARKER_COLORS, "edge markers must use blue/red/green/yellow"
        target = _trace_first_hit(labels, r, c, dr, dc)
        assert target is not None, "every marker must hit a gray object"
        hits[target].add(int(color))
    return comps, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray object from the diversity of boundary markers that project into it.

    Input structure:
    - The grid is a black background with 3-4 separate gray 4-connected objects in the interior.
    - Single-pixel blue/red/green/yellow markers sit on the outer boundary, never in the corners.
    - Each marker shares a row or column with at least one gray object along its inward direction.

    Rule:
    - Project every boundary marker straight inward: top goes down, bottom goes up, left goes right, right goes left.
    - The first gray pixel met on that ray determines which gray connected component the marker hits.
    - Collect the set of distinct marker colors hitting each object, delete the markers, then recolor the whole object:
      one distinct color -> that color, two distinct colors -> orange, three or more -> pink.

    Key insight:
    - The output color is determined by a many-to-one aggregation over all rays hitting an object, so the crucial signal is
      the size of each object's distinct-color hit set rather than any single marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(g[0, 0]) == BACKGROUND
    assert int(g[0, -1]) == BACKGROUND
    assert int(g[-1, 0]) == BACKGROUND
    assert int(g[-1, -1]) == BACKGROUND
    interior = g[1:-1, 1:-1]
    assert not np.any((interior != BACKGROUND) & (interior != OBJECT_COLOR)), "only gray objects belong in the interior"

    comps, hits = _solve_by_projection(g)
    assert 3 <= len(comps) <= 4, "expected 3-4 gray objects"

    out = np.zeros_like(g)
    for idx, coords in enumerate(comps):
        fill = _recolor_from_hit_set(hits[idx])
        out[coords[:, 0], coords[:, 1]] = fill
    return out


def _random_polyomino(area: int) -> np.ndarray:
    for _ in range(300):
        cells = {(0, 0)}
        frontier = set(_neighbors4(0, 0))
        while len(cells) < area:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(*cell):
                if nb not in cells:
                    frontier.add(nb)

        coords = np.array(sorted(cells), dtype=int)
        coords -= coords.min(axis=0)
        patch = np.zeros((coords[:, 0].max() + 1, coords[:, 1].max() + 1), dtype=int)
        patch[coords[:, 0], coords[:, 1]] = 1

        if random.random() < 0.5:
            patch = np.flipud(patch)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        patch = np.rot90(patch, k=random.randrange(4))

        out = np.argwhere(patch > 0)
        r0, c0, _r1, _c1 = bbox_coords(out)
        out = out - np.array([r0, c0], dtype=int)
        h = int(out[:, 0].max()) + 1
        w = int(out[:, 1].max()) + 1
        if h <= OBJECT_BBOX_MAX and w <= OBJECT_BBOX_MAX:
            return out.astype(int)
    raise RuntimeError("failed to sample a compact polyomino")


def _can_place(coords: np.ndarray, top: int, left: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    placed = coords + np.array([top, left], dtype=int)
    for r, c in placed:
        rr = int(r)
        cc = int(c)
        if rr < OBJECT_MARGIN or cc < OBJECT_MARGIN or rr >= h - OBJECT_MARGIN or cc >= w - OBJECT_MARGIN:
            return False
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_objects(height: int, width: int, shapes: list[np.ndarray]) -> list[np.ndarray] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placed_objects: list[np.ndarray] = []
    for coords in sorted(shapes, key=len, reverse=True):
        shape_h = int(coords[:, 0].max()) + 1
        shape_w = int(coords[:, 1].max()) + 1
        candidates = []
        for top in range(OBJECT_MARGIN, height - shape_h - OBJECT_MARGIN + 1):
            for left in range(OBJECT_MARGIN, width - shape_w - OBJECT_MARGIN + 1):
                if _can_place(coords, top, left, occupied):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placed = coords + np.array([top, left], dtype=int)
        occupied[placed[:, 0], placed[:, 1]] = True
        placed_objects.append(placed)
    return placed_objects


def _boundary_slots(placed_objects: list[np.ndarray], shape: tuple[int, int]) -> tuple[dict[int, list[tuple[int, int, int, int]]], np.ndarray]:
    labels = np.full(shape, -1, dtype=int)
    for idx, coords in enumerate(placed_objects):
        labels[coords[:, 0], coords[:, 1]] = idx

    h, w = shape
    slots = {idx: [] for idx in range(len(placed_objects))}
    for c in range(1, w - 1):
        target = _trace_first_hit(labels, 0, c, 1, 0)
        if target is not None:
            slots[target].append((0, c, 1, 0))
        target = _trace_first_hit(labels, h - 1, c, -1, 0)
        if target is not None:
            slots[target].append((h - 1, c, -1, 0))
    for r in range(1, h - 1):
        target = _trace_first_hit(labels, r, 0, 0, 1)
        if target is not None:
            slots[target].append((r, 0, 0, 1))
        target = _trace_first_hit(labels, r, w - 1, 0, -1)
        if target is not None:
            slots[target].append((r, w - 1, 0, -1))
    return slots, labels


def _choose_unique_counts(n_objects: int) -> list[int]:
    if n_objects == 3:
        template = random.choice(([1, 2, 3], [1, 2, 4]))
    else:
        template = random.choice(([1, 1, 2, 3], [1, 1, 2, 4]))
    counts = list(template)
    random.shuffle(counts)
    return counts


def _marker_plan(
    slots_by_object: dict[int, list[tuple[int, int, int, int]]],
    unique_counts: list[int],
) -> list[tuple[int, int, int]]:
    chosen_slots: dict[int, list[tuple[int, int, int, int]]] = {}
    color_sets: dict[int, set[int]] = {}
    plan: list[tuple[int, int, int]] = []

    for idx, count in enumerate(unique_counts):
        slots = slots_by_object[idx]
        if len(slots) < count:
            raise ValueError("not enough marker lanes for the requested color set")
        picked_slots = random.sample(slots, count)
        picked_colors = random.sample(MARKER_COLORS, count)
        chosen_slots[idx] = list(picked_slots)
        color_sets[idx] = set(picked_colors)
        for (r, c, _dr, _dc), color in zip(picked_slots, picked_colors):
            plan.append((r, c, color))

    total_markers = len(plan)
    if total_markers < 8 and random.random() < 0.8:
        extra_budget = min(8 - total_markers, 1 + int(total_markers <= 6 and random.random() < 0.5))
        for _ in range(extra_budget):
            eligible = [
                idx
                for idx, slots in slots_by_object.items()
                if len(slots) > len(chosen_slots[idx])
            ]
            if not eligible:
                break
            idx = random.choice(eligible)
            remaining = [slot for slot in slots_by_object[idx] if slot not in chosen_slots[idx]]
            if not remaining:
                continue
            slot = random.choice(remaining)
            color = random.choice(tuple(color_sets[idx]))
            chosen_slots[idx].append(slot)
            plan.append((slot[0], slot[1], color))
    random.shuffle(plan)
    return plan


def _draw_input(shape: tuple[int, int], placed_objects: list[np.ndarray], markers: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for coords in placed_objects:
        grid[coords[:, 0], coords[:, 1]] = OBJECT_COLOR
    for r, c, color in markers:
        grid[r, c] = color
    return grid


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    comps = components_mask(g == OBJECT_COLOR, connectivity=4)
    labels = np.full(g.shape, -1, dtype=int)
    for idx, coords in enumerate(comps):
        labels[coords[:, 0], coords[:, 1]] = idx

    hits = [set() for _ in comps]
    h, w = g.shape
    for c in range(1, w - 1):
        color = int(g[0, c])
        if color != BACKGROUND:
            target = _trace_first_hit(labels, 0, c, 1, 0)
            assert target is not None
            hits[target].add(color)
        color = int(g[h - 1, c])
        if color != BACKGROUND:
            target = _trace_first_hit(labels, h - 1, c, -1, 0)
            assert target is not None
            hits[target].add(color)
    for r in range(1, h - 1):
        color = int(g[r, 0])
        if color != BACKGROUND:
            target = _trace_first_hit(labels, r, 0, 0, 1)
            assert target is not None
            hits[target].add(color)
        color = int(g[r, w - 1])
        if color != BACKGROUND:
            target = _trace_first_hit(labels, r, w - 1, 0, -1)
            assert target is not None
            hits[target].add(color)

    out = np.zeros_like(g)
    for idx, coords in enumerate(comps):
        out[coords[:, 0], coords[:, 1]] = _recolor_from_hit_set(hits[idx])
    return out


def _validate_generated_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert int(g[0, 0]) == BACKGROUND
    assert int(g[0, -1]) == BACKGROUND
    assert int(g[-1, 0]) == BACKGROUND
    assert int(g[-1, -1]) == BACKGROUND

    comps = components_mask(g == OBJECT_COLOR, connectivity=4)
    assert 3 <= len(comps) <= 4
    for coords in comps:
        assert OBJECT_AREA_MIN <= len(coords) <= OBJECT_AREA_MAX

    labels = np.full(g.shape, -1, dtype=int)
    for idx, coords in enumerate(comps):
        labels[coords[:, 0], coords[:, 1]] = idx

    marker_cells = []
    h, w = g.shape
    for r, c, color, dr, dc in _iter_boundary_markers(g):
        assert color in MARKER_COLORS
        marker_cells.append((r, c, color, dr, dc))
    assert 5 <= len(marker_cells) <= 8

    interior = g[1:-1, 1:-1]
    assert not np.any((interior != BACKGROUND) & (interior != OBJECT_COLOR))

    hit_sets = [set() for _ in comps]
    hit_counts = [0 for _ in comps]
    for r, c, color, dr, dc in marker_cells:
        target = _trace_first_hit(labels, r, c, dr, dc)
        assert target is not None
        hit_sets[target].add(color)
        hit_counts[target] += 1
    assert all(count >= 1 for count in hit_counts)
    assert len({len(colors) for colors in hit_sets}) >= 2
    assert any(hit_counts[idx] > len(hit_sets[idx]) for idx in range(len(hit_sets))) or len(marker_cells) == sum(len(s) for s in hit_sets)

    occupied = [set(map(tuple, coords)) for coords in comps]
    for i in range(len(occupied)):
        for j in range(i + 1, len(occupied)):
            for r, c in occupied[i]:
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        assert (rr, cc) not in occupied[j], "objects should not touch, even diagonally"

    expected = _independent_expected(g)
    actual = main(g)
    assert np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    width = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    n_objects = random.choice(OBJECT_COUNT_CHOICES)

    shapes = [_random_polyomino(random.randint(OBJECT_AREA_MIN, OBJECT_AREA_MAX)) for _ in range(n_objects)]
    placed = _place_objects(height, width, shapes)
    if placed is None:
        return None

    unique_counts = _choose_unique_counts(n_objects)
    slots_by_object, _labels = _boundary_slots(placed, (height, width))
    if any(len(slots_by_object[idx]) < unique_counts[idx] for idx in range(n_objects)):
        return None

    try:
        markers = _marker_plan(slots_by_object, unique_counts)
    except ValueError:
        return None

    grid = _draw_input((height, width), placed, markers)
    try:
        _validate_generated_grid(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection-hit ARC input")
