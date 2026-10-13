from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


ALLOWED_COLORS = (1, 2, 3, 4, 7)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GEN_ATTEMPTS = 600

_CANONICAL_PATCHES = {
    2: (
        np.array([[1, 1]], dtype=bool),
    ),
    3: (
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
    ),
    4: (
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    ),
    5: (
        np.array([[1, 1, 1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    ),
}


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _patch_to_coords(patch: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalize_coords([(int(r), int(c)) for r, c in np.argwhere(patch)])


def _build_shape_library() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    library: dict[int, set[tuple[tuple[int, int], ...]]] = {k: set() for k in _CANONICAL_PATCHES}
    for size, patches in _CANONICAL_PATCHES.items():
        for patch in patches:
            for k in range(4):
                rotated = np.rot90(patch, k)
                for candidate in (rotated, np.fliplr(rotated)):
                    library[size].add(_patch_to_coords(candidate))
    return {size: sorted(shapes) for size, shapes in library.items()}


SHAPE_LIBRARY = _build_shape_library()


def _foreground_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != 0)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _count_components_by_color(grid: np.ndarray) -> dict[int, int]:
    return {color: len(_component_coords(grid == color)) for color in _foreground_colors(grid)}


def _component_sizes_by_color(grid: np.ndarray) -> dict[int, list[int]]:
    return {
        color: sorted(int(comp.shape[0]) for comp in _component_coords(grid == color))
        for color in _foreground_colors(grid)
    }


def _areas_by_color(grid: np.ndarray) -> dict[int, int]:
    return {color: int(np.count_nonzero(grid == color)) for color in _foreground_colors(grid)}


def _ascending_unique_order(values: dict[int, int]) -> list[int]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    ranks = [value for _, value in ordered]
    assert len(set(ranks)) == len(ranks), "expected unique ranks"
    return [color for color, _ in ordered]


def _apply_cycle(grid: np.ndarray, ordered_colors: list[int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    mapping = {
        int(ordered_colors[0]): int(ordered_colors[1]),
        int(ordered_colors[1]): int(ordered_colors[2]),
        int(ordered_colors[2]): int(ordered_colors[0]),
    }
    for src, dst in mapping.items():
        out[grid == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor every foreground pixel by a count-ranked 3-cycle over the input colors.

    Input structure:
    - The grid has black background 0 and exactly three non-black foreground colors.
    - Each foreground color appears in one or more 4-connected monochrome objects.
    - Object positions, shapes, and sizes vary, but the colors are the only values that may change.

    Rule:
    - Count the 4-connected objects of each foreground color.
    - Sort the three colors by object count in ascending order, calling them c1, c2, c3.
    - Apply the cycle c1 -> c2, c2 -> c3, c3 -> c1 to every foreground pixel, leaving background 0 unchanged.

    Key insight:
    - The recoloring is not fixed by color identity; the current input's per-color object counts define the cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = _foreground_colors(g)
    assert len(colors) == 3, "expected exactly three foreground colors"

    count_by_color = _count_components_by_color(g)
    ordered = _ascending_unique_order(count_by_color)
    return _apply_cycle(g, ordered)


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    return h, w


def _sample_size_for_rank(rank: int) -> int:
    weights = {
        0: [1, 2, 4, 5],
        1: [2, 4, 3, 2],
        2: [5, 4, 2, 1],
    }
    return random.choices([2, 3, 4, 5], weights=weights[rank], k=1)[0]


def _sample_shape(size: int) -> tuple[tuple[int, int], ...]:
    return random.choice(SHAPE_LIBRARY[size])


def _is_non_rectangular(shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = _shape_bbox(shape)
    return len(shape) != h * w


def _pick_counts_and_colors() -> tuple[dict[int, int], list[int]]:
    colors = random.sample(ALLOWED_COLORS, 3)
    counts = random.sample([1, 2, 3, 4, 5], 3)
    return {color: count for color, count in zip(colors, counts)}, colors


def _choose_object_shapes(count_by_color: dict[int, int]) -> tuple[list[tuple[int, tuple[tuple[int, int], ...]]], bool]:
    colors_by_count = _ascending_unique_order(count_by_color)
    rank_by_color = {color: rank for rank, color in enumerate(colors_by_count)}
    objects: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    saw_non_rect = False
    for color, count in count_by_color.items():
        for _ in range(count):
            size = _sample_size_for_rank(rank_by_color[color])
            shape = _sample_shape(size)
            saw_non_rect = saw_non_rect or _is_non_rectangular(shape)
            objects.append((color, shape))
    random.shuffle(objects)
    return objects, saw_non_rect


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr, dc in DIR8 + [(0, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_objects(
    h: int,
    w: int,
    objects: list[tuple[int, tuple[tuple[int, int], ...]]],
) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    ordered = sorted(
        objects,
        key=lambda item: (
            -len(item[1]),
            -(_shape_bbox(item[1])[0] * _shape_bbox(item[1])[1]),
            random.random(),
        ),
    )
    for color, shape in ordered:
        sh, sw = _shape_bbox(shape)
        candidates: list[list[tuple[int, int]]] = []
        for r0 in range(1, h - sh):
            for c0 in range(1, w - sw):
                cells = [(r0 + r, c0 + c) for r, c in shape]
                if any(blocked[r, c] for r, c in cells):
                    continue
                candidates.append(cells)
        if not candidates:
            return None
        cells = random.choice(candidates)
        for r, c in cells:
            grid[r, c] = color
        _mark_blocked(blocked, cells)
    return grid


def _alternative_color_order_output(grid: np.ndarray) -> np.ndarray:
    return _apply_cycle(grid, sorted(_foreground_colors(grid)))


def _alternative_area_order_output(grid: np.ndarray) -> np.ndarray:
    return _apply_cycle(grid, _ascending_unique_order(_areas_by_color(grid)))


def _valid_component_geometry(grid: np.ndarray) -> bool:
    h, w = grid.shape
    components: list[set[tuple[int, int]]] = []
    for color in _foreground_colors(grid):
        for comp in _component_coords(grid == color):
            if not (2 <= comp.shape[0] <= 5):
                return False
            cells = {(int(r), int(c)) for r, c in comp}
            for r, c in comp:
                rr = int(r)
                cc = int(c)
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    return False
            components.append(cells)
    for i, comp_a in enumerate(components):
        for comp_b in components[i + 1 :]:
            for r1, c1 in comp_a:
                for r2, c2 in comp_b:
                    if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                        return False
    return True


def _has_area_counterexample(grid: np.ndarray) -> bool:
    area_by_color = _areas_by_color(grid)
    if len(set(area_by_color.values())) != 3:
        return False
    return not np.array_equal(main(grid), _alternative_area_order_output(grid))


def _has_numeric_order_counterexample(grid: np.ndarray) -> bool:
    return not np.array_equal(main(grid), _alternative_color_order_output(grid))


def _validate_instance(grid: np.ndarray) -> bool:
    colors = _foreground_colors(grid)
    if len(colors) != 3 or any(color not in ALLOWED_COLORS for color in colors):
        return False

    counts = _count_components_by_color(grid)
    if sorted(counts.values()) != sorted(set(counts.values())):
        return False
    if any(not (1 <= count <= 5) for count in counts.values()):
        return False

    if not _valid_component_geometry(grid):
        return False
    if not _has_area_counterexample(grid):
        return False
    if not _has_numeric_order_counterexample(grid):
        return False

    out = main(grid)
    if out.shape != grid.shape or np.any(out[grid == 0] != 0):
        return False
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _alternative_area_order_output(grid)):
        return False
    if np.array_equal(out, _alternative_color_order_output(grid)):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    count_by_color, _colors = _pick_counts_and_colors()
    total_objects = sum(count_by_color.values())
    min_side = max(10, 9 + (total_objects + 1) // 3)
    h = random.randint(min_side, min(14, min_side + 2))
    w = random.randint(min_side, min(14, min_side + 2))

    objects, saw_non_rect = _choose_object_shapes(count_by_color)
    if not saw_non_rect:
        return None

    grid = _place_objects(h, w, objects)
    if grid is None:
        return None
    if _count_components_by_color(grid) != count_by_color:
        return None
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid count-ranked color-cycle input")
