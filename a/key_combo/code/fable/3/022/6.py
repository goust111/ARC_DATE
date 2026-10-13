from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


ALL_COLORS = tuple(range(10))
GEN_RETRIES = 500


@dataclass(frozen=True)
class Container:
    left: int
    right: int
    top: int
    bottom: int

    @property
    def inner_width(self) -> int:
        return self.right - self.left - 1

    @property
    def depth(self) -> int:
        return self.bottom - self.top

    @property
    def capacity(self) -> int:
        return self.inner_width * self.depth


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    color, _ = counts.most_common(1)[0]
    max_count = counts[color]
    assert sum(1 for value in counts.values() if value == max_count) == 1, "background must be uniquely dominant"
    return int(color)


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    excluded = set() if exclude is None else {int(c) for c in exclude}
    return [int(c) for c in np.unique(grid) if int(c) not in excluded]


def _find_sand_column(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in _colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        if coords.size == 0:
            continue
        cols = sorted({int(c) for _, c in coords})
        if len(cols) != 1:
            continue
        rows = sorted(int(r) for r, _ in coords)
        if rows and rows == list(range(rows[-1] + 1)):
            candidates.append((int(color), int(cols[0]), len(rows)))

    assert len(candidates) == 1, "expected exactly one top-hanging sand column"
    sand_color, sand_col, grain_count = candidates[0]
    assert grain_count > 0, "the sand column must contain at least one grain"
    return sand_color, sand_col, grain_count


def _bottom_runs(g: np.ndarray, wall: int) -> list[tuple[int, int]]:
    bottom = g.shape[0] - 1
    runs: list[tuple[int, int]] = []
    c = 0
    while c < g.shape[1]:
        if int(g[bottom, c]) != int(wall):
            c += 1
            continue
        start = c
        while c < g.shape[1] and int(g[bottom, c]) == int(wall):
            c += 1
        runs.append((start, c - 1))
    return runs


def _find_containers(g: np.ndarray, wall: int) -> list[Container]:
    H, W = g.shape
    runs = _bottom_runs(g, wall)
    assert runs, "expected U-shaped containers on the bottom edge"

    frame = np.zeros_like(g, dtype=bool)
    containers: list[Container] = []
    for left, right in runs:
        assert right - left + 1 >= 3, "each U bottom must have two walls and at least one interior column"

        depth = 0
        r = H - 2
        while r >= 0 and int(g[r, left]) == int(wall) and int(g[r, right]) == int(wall):
            assert not np.any(g[r, left + 1 : right] == int(wall)), "container interiors cannot contain wall pixels"
            depth += 1
            r -= 1
        assert depth >= 1, "each U must have positive internal depth"

        top = H - 1 - depth
        container = Container(int(left), int(right), int(top), H - 1)
        containers.append(container)

        frame[top:H, left] = True
        frame[top:H, right] = True
        frame[H - 1, left : right + 1] = True

    containers.sort(key=lambda c: c.left)
    for prev, cur in zip(containers, containers[1:]):
        assert prev.right + 1 < cur.left, "neighboring containers must be separated by background columns"
    assert np.array_equal(g == int(wall), frame), "all wall pixels must belong to the parsed U frames"
    assert all(c.inner_width > 0 and c.depth > 0 for c in containers)
    return containers


def _find_inlet(containers: list[Container], sand_col: int) -> int:
    hits = [i for i, container in enumerate(containers) if container.left < sand_col < container.right]
    assert len(hits) == 1, "the sand column must point into exactly one container interior"
    return hits[0]


def _allocate(containers: list[Container], start: int, grains: int) -> list[int]:
    amounts = [0] * len(containers)
    remaining = int(grains)
    for idx in range(start, len(containers)):
        container = containers[idx]
        take = min(remaining, container.capacity)
        assert take % container.inner_width == 0, "final sand levels must occupy whole rows"
        amounts[idx] = take
        remaining -= take
        if remaining == 0:
            break

    assert remaining == 0, "sand must fit into the inlet container and containers to its right"
    return amounts


def _paint_amount(out: np.ndarray, container: Container, amount: int, sand: int) -> None:
    assert amount % container.inner_width == 0
    rows = amount // container.inner_width
    assert 0 <= rows <= container.depth
    if rows == 0:
        return
    r0 = container.bottom - rows
    r1 = container.bottom - 1
    out[r0 : r1 + 1, container.left + 1 : container.right] = int(sand)


def main(grid: np.ndarray) -> np.ndarray:
    """Pour a hanging sand column into bottom U containers with rightward cascade overflow.

    Input structure:
    - A dominant background surrounds several bottom-aligned U-shaped containers of one wall color.
    - Each U has a solid bottom bar, two equal-height side walls, and a rectangular open interior.
    - A single vertical column of one sand color starts at the top edge and points into exactly one container interior.

    Rule:
    - Count the pixels in the hanging sand column and remove that column from the output.
    - Starting with the container directly below the column, fill each container interior from bottom to top in whole rows.
    - When a container reaches its full inner capacity, pass the remaining sand to the nearest container on its right.
    - Keep every wall pixel and every background pixel outside the final sand fill unchanged.

    Key insight:
    - A container's level is determined by the total grain count after subtracting the capacities of all earlier containers in the rightward cascade.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    sand, sand_col, grains = _find_sand_column(g, bg)
    other_colors = [c for c in _colors_present(g, exclude={bg, sand})]
    assert len(other_colors) == 1, "expected exactly one container wall color"
    wall = int(other_colors[0])

    containers = _find_containers(g, wall)
    start = _find_inlet(containers, sand_col)
    amounts = _allocate(containers, start, grains)

    out = g.copy()
    out[out == int(sand)] = int(bg)
    for container, amount in zip(containers, amounts):
        _paint_amount(out, container, amount, sand)
    return out


# --------------- generator ---------------


def _choose_dimensions() -> tuple[list[int], list[int]] | None:
    count = random.randint(3, 5)
    widths = [random.randint(2, 5) for _ in range(count)]
    depths = [random.randint(2, 5) for _ in range(count)]
    if len(set(widths)) < 2 or len(set(depths)) < 2:
        return None
    return widths, depths


def _choose_distribution(widths: list[int], depths: list[int]) -> tuple[int, list[int], str] | None:
    n = len(widths)
    modes = ("overflow_partial", "overflow_exact", "no_overflow")
    weights = (5, 3, 4)

    for _ in range(80):
        mode = random.choices(modes, weights=weights, k=1)[0]
        levels = [0] * n

        if mode == "no_overflow":
            candidates = [i for i, depth in enumerate(depths) if depth >= 2]
            if not candidates:
                continue
            start = random.choice(candidates)
            levels[start] = random.randint(1, depths[start] - 1)
        elif mode == "overflow_partial":
            candidates = [i for i in range(n - 1) if any(d >= 2 for d in depths[i + 1 :])]
            if not candidates:
                continue
            start = random.choice(candidates)
            final_options = [i for i in range(start + 1, n) if depths[i] >= 2]
            final = random.choice(final_options)
            for idx in range(start, final):
                levels[idx] = depths[idx]
            levels[final] = random.randint(1, depths[final] - 1)
        else:
            start = random.randint(0, n - 2)
            final = random.randint(start + 1, n - 1)
            for idx in range(start, final + 1):
                levels[idx] = depths[idx]

        grains = sum(levels[i] * widths[i] for i in range(n))
        if 0 < grains <= 24:
            return start, levels, mode
    return None


def _place_containers(widths: list[int], depths: list[int]) -> tuple[int, list[Container]] | None:
    spans = [w + 2 for w in widths]
    gaps = [random.randint(1, 2) for _ in range(len(widths) - 1)]
    used_width = sum(spans) + sum(gaps)
    if used_width + 2 > 30:
        return None

    grid_width = random.randint(used_width + 2, 30)
    left_margin = random.randint(1, grid_width - used_width - 1)

    containers: list[Container] = []
    left = left_margin
    bottom = -1
    for idx, (inner_width, depth) in enumerate(zip(widths, depths)):
        right = left + inner_width + 1
        containers.append(Container(left, right, bottom - depth, bottom))
        left = right + 1
        if idx < len(gaps):
            left += gaps[idx]
    return grid_width, containers


def _with_height(containers: list[Container], source: int, grains: int) -> tuple[int, list[Container]] | None:
    max_depth = max(c.depth for c in containers)
    source_depth = containers[source].depth
    min_height = max(12, max_depth + 4, grains + source_depth + 1)
    if min_height > 30:
        return None

    height = random.randint(min_height, 30)
    bottom = height - 1
    shifted = [
        Container(c.left, c.right, bottom - c.depth, bottom)
        for c in containers
    ]
    assert grains <= shifted[source].top
    return height, shifted


def _render_input(
    shape: tuple[int, int],
    containers: list[Container],
    bg: int,
    wall: int,
    sand: int,
    source: int,
    grains: int,
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for container in containers:
        grid[container.top : container.bottom + 1, container.left] = int(wall)
        grid[container.top : container.bottom + 1, container.right] = int(wall)
        grid[container.bottom, container.left : container.right + 1] = int(wall)

    inlet_col = containers[source].left + 1 + (containers[source].inner_width - 1) // 2
    assert grains <= containers[source].top
    grid[:grains, inlet_col] = int(sand)
    return grid


def _render_expected(inp: np.ndarray, containers: list[Container], levels: list[int], bg: int, sand: int) -> np.ndarray:
    out = inp.copy()
    out[out == int(sand)] = int(bg)
    for container, rows in zip(containers, levels):
        assert 0 <= rows <= container.depth
        amount = rows * container.inner_width
        _paint_amount(out, container, amount, sand)
    return out


def _semantic_accepts(
    inp: np.ndarray,
    expected: np.ndarray,
    containers: list[Container],
    widths: list[int],
    depths: list[int],
    source: int,
    levels: list[int],
    mode: str,
    bg: int,
    wall: int,
    sand: int,
) -> bool:
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False
    if _most_common_color(inp) != int(bg):
        return False
    if len(containers) < 3 or len(set(widths)) < 2 or len(set(depths)) < 2:
        return False
    if wall == bg or sand == bg or wall == sand:
        return False
    if any(levels[i] != 0 for i in range(source)):
        return False

    first_filled = next((i for i, level in enumerate(levels) if level > 0), None)
    if first_filled != source:
        return False

    source_capacity = containers[source].capacity
    grains = sum(levels[i] * containers[i].inner_width for i in range(len(containers)))
    if mode == "no_overflow":
        if not (0 < grains < source_capacity):
            return False
        if any(levels[i] != 0 for i in range(source + 1, len(containers))):
            return False
    elif mode == "overflow_partial":
        if not (grains > source_capacity):
            return False
        right_nonempty = [i for i in range(source + 1, len(containers)) if levels[i] > 0]
        if not right_nonempty:
            return False
        final = right_nonempty[-1]
        if levels[final] >= depths[final]:
            return False
        if any(levels[i] != depths[i] for i in range(source, final)):
            return False
    elif mode == "overflow_exact":
        if not (grains > source_capacity):
            return False
        nonempty = [i for i, level in enumerate(levels) if level > 0]
        if not nonempty or nonempty[0] != source:
            return False
        if any(levels[i] != depths[i] for i in nonempty):
            return False
    else:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected) and not np.array_equal(inp, expected)


def _gen_attempt() -> np.ndarray | None:
    dims = _choose_dimensions()
    if dims is None:
        return None
    widths, depths = dims

    distribution = _choose_distribution(widths, depths)
    if distribution is None:
        return None
    source, levels, mode = distribution
    grains = sum(levels[i] * widths[i] for i in range(len(widths)))

    placed = _place_containers(widths, depths)
    if placed is None:
        return None
    grid_width, base_containers = placed

    sized = _with_height(base_containers, source, grains)
    if sized is None:
        return None
    height, containers = sized

    bg = random.choice(ALL_COLORS)
    wall = random.choice([c for c in ALL_COLORS if c != bg])
    sand = random.choice([c for c in ALL_COLORS if c not in {bg, wall}])

    inp = _render_input((height, grid_width), containers, bg, wall, sand, source, grains)
    expected = _render_expected(inp, containers, levels, bg, sand)
    if not _semantic_accepts(inp, expected, containers, widths, depths, source, levels, mode, bg, wall, sand):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascade-overflow input")
