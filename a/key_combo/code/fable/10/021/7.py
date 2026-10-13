from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
GREY = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


SHAPE_LIBRARY: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in FOUR_DIRS:
        yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    colors: Iterable[int] | None = None,
) -> list[dict]:
    h, w = grid.shape
    if colors is None:
        allowed = set(int(v) for v in np.unique(grid) if int(v) != bg)
    else:
        allowed = {int(v) for v in colors if int(v) != bg}

    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in allowed:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(sorted(pts), dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_area(shape: tuple[tuple[int, int], ...]) -> int:
    return len(shape)


def _classify_input(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _most_common_color(grid)
    h, _w = grid.shape
    components = _find_components(grid, bg=bg)
    bars: list[dict] = []
    objects: list[dict] = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        width = c1 - c0 + 1
        height = r1 - r0 + 1
        is_bar = r1 == h - 1 and width == 1 and comp["area"] == height
        if is_bar:
            bars.append(comp)
        else:
            objects.append(comp)
    return bg, bars, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray shape family by the legend bar whose height equals that family's population.

    Input structure:
    - The background contains several solid one-column vertical bars standing on the bottom edge.
    - Every non-bar object elsewhere is gray, 4-connected, and families are defined by translation-equivalent shapes.
    - Different shape families have different member counts, and bar heights are pairwise distinct.

    Rule:
    - Treat each bottom bar of height h and color c as a legend entry mapping family size h -> color c.
    - Group the gray objects by normalized shape, count the number of members in each family, and recolor the whole family if its count matches some bar height.
    - Erase every family whose count matches no bar height, while leaving the bars unchanged.

    Key insight:
    - The bars are not a histogram to summarize the gray objects; they are a count-to-color legend, so the solver must group by shape first and only then compare family frequency to bar height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, bars, objects = _classify_input(g)
    assert bars, "expected at least one legend bar"
    assert objects, "expected at least one gray object"

    legend: dict[int, int] = {}
    for bar in bars:
        r0, _c0, r1, _c1 = bar["bbox"]
        height = r1 - r0 + 1
        assert height not in legend, "legend bar heights must be distinct"
        legend[height] = int(bar["color"])

    object_colors = {int(obj["color"]) for obj in objects}
    assert len(object_colors) == 1, "all non-bar objects must share one gray source color"

    families: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for obj in objects:
        families[_shape_key(obj["coords"])].append(obj)

    counts = [len(group) for group in families.values()]
    assert len(counts) == len(set(counts)), "shape-family populations must be pairwise distinct"

    out = g.copy()
    for group in families.values():
        family_size = len(group)
        target = legend.get(family_size, bg)
        for obj in group:
            rr, cc = obj["coords"].T
            out[rr, cc] = target
    return out


def _choose_shapes(family_sizes: list[int]) -> list[tuple[tuple[int, int], ...]]:
    for _ in range(400):
        shapes = random.sample(list(SHAPE_LIBRARY), len(family_sizes))
        areas = [_shape_area(shape) for shape in shapes]
        if any(a == b for i, a in enumerate(areas) for b in areas[i + 1 :]):
            if all(size != area for size, area in zip(family_sizes, areas)):
                return shapes
    raise RuntimeError("failed to pick non-degenerate family shapes")


def _choose_bar_columns(width: int, count: int) -> list[int]:
    base = sorted(random.sample(range(width - 2 * (count - 1)), count))
    return [col + 2 * idx for idx, col in enumerate(base)]


def _can_place(coords: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return False
    return True


def _paint_coords(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _build_expected(
    grid: np.ndarray,
    family_members: list[list[list[tuple[int, int]]]],
    family_sizes: list[int],
    legend: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    bg = BACKGROUND
    for members, size in zip(family_members, family_sizes):
        color = legend.get(size, bg)
        for coords in members:
            _paint_coords(out, coords, color)
    return out


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    family_sizes: list[int],
    family_shapes: list[tuple[tuple[int, int], ...]],
    legend: dict[int, int],
) -> None:
    assert grid.shape == expected.shape
    assert np.array_equal(main(grid), expected), "solver and generator disagree"

    bg, bars, objects = _classify_input(grid)
    assert bg == BACKGROUND
    assert len(bars) == len(legend)
    assert len({int(bar["color"]) for bar in bars}) == len(bars)
    assert sorted((bar["bbox"][2] - bar["bbox"][0] + 1) for bar in bars) == sorted(legend)

    extracted = defaultdict(list)
    for obj in objects:
        extracted[_shape_key(obj["coords"])].append(obj)
    extracted_sizes = sorted(len(group) for group in extracted.values())
    assert extracted_sizes == sorted(family_sizes)
    assert len(extracted_sizes) == len(set(extracted_sizes))

    shape_areas = [_shape_area(shape) for shape in family_shapes]
    assert any(a == b for i, a in enumerate(shape_areas) for b in shape_areas[i + 1 :])
    assert all(size != area for size, area in zip(family_sizes, shape_areas))

    for bar in bars:
        rr, cc = bar["coords"].T
        assert np.array_equal(grid[rr, cc], expected[rr, cc])

    matched = 0
    erased = 0
    for group in extracted.values():
        size = len(group)
        target = legend.get(size, bg)
        if size in legend:
            matched += 1
        else:
            erased += 1
        for obj in group:
            rr, cc = obj["coords"].T
            assert np.all(expected[rr, cc] == target)
    assert matched >= 2
    assert matched + erased == len(family_sizes)
    assert np.any(expected != grid)


def _gen_attempt() -> np.ndarray | None:
    matched_count = random.randint(2, 4)
    max_extra = min(2, 5 - matched_count)
    if max_extra == 0:
        unmatched_count = 0
    else:
        unmatched_count = random.choices(
            list(range(max_extra + 1)),
            weights=[4, 5, 3][: max_extra + 1],
            k=1,
        )[0]

    matched_sizes = random.sample([2, 3, 4, 5], matched_count)
    remaining_sizes = [size for size in [1, 2, 3, 4, 5] if size not in matched_sizes]
    unmatched_sizes = random.sample(remaining_sizes, unmatched_count)
    family_sizes = matched_sizes + unmatched_sizes
    random.shuffle(family_sizes)

    total_objects = sum(family_sizes)
    family_shapes = _choose_shapes(family_sizes)
    max_shape_h = max(_shape_dims(shape)[0] for shape in family_shapes)

    object_rows = min(18, max(8, 5 + total_objects // 2) + random.randint(0, 2))
    height = object_rows + max(matched_sizes) + 3
    if height > 30:
        return None

    width_min = max(16, 8 + total_objects)
    width_max = min(29, width_min + 5)
    if width_min > width_max:
        return None
    width = random.randint(width_min, width_max)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    legend_colors = random.sample(list(BAR_COLORS), matched_count)
    bar_specs = list(zip(matched_sizes, legend_colors))
    random.shuffle(bar_specs)
    legend = {size: color for size, color in bar_specs}
    for col, (bar_height, color) in zip(_choose_bar_columns(width, matched_count), bar_specs):
        grid[height - bar_height :, col] = color

    occupied = np.zeros((object_rows, width), dtype=bool)
    family_members: list[list[list[tuple[int, int]]]] = [[] for _ in family_sizes]
    instance_order = []
    for family_id, size in enumerate(family_sizes):
        shape = family_shapes[family_id]
        sh, sw = _shape_dims(shape)
        for _ in range(size):
            instance_order.append((-(sh * sw), -len(shape), family_id))
    random.shuffle(instance_order)
    instance_order.sort()

    for _priority_a, _priority_b, family_id in instance_order:
        shape = family_shapes[family_id]
        sh, sw = _shape_dims(shape)
        placed = False
        for _ in range(250):
            top = random.randint(0, object_rows - sh)
            left = random.randint(0, width - sw)
            coords = [(top + dr, left + dc) for dr, dc in shape]
            if not _can_place(coords, occupied):
                continue
            for r, c in coords:
                occupied[r, c] = True
                grid[r, c] = GREY
            family_members[family_id].append(coords)
            placed = True
            break
        if not placed:
            return None

    expected = _build_expected(grid, family_members, family_sizes, legend)
    _validate_case(grid, expected, family_sizes, family_shapes, legend)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid histogram-legend input")
