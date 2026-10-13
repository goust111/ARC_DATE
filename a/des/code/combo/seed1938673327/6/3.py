from __future__ import annotations

from collections import Counter

import numpy as np
import random


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(int(r), int(c))]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
    return count


def _separator_candidates(g: np.ndarray) -> list[tuple[int, list[int], list[int]]]:
    h, w = g.shape
    candidates: list[tuple[int, list[int], list[int]]] = []

    for color in np.unique(g):
        color = int(color)
        rows = [r for r in range(h) if bool(np.all(g[r, :] == color))]
        cols = [c for c in range(w) if bool(np.all(g[:, c] == color))]
        if len(rows) < 2 or len(cols) < 2:
            continue
        if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
            continue
        if any(rows[i] + 1 >= rows[i + 1] for i in range(len(rows) - 1)):
            continue
        if any(cols[i] + 1 >= cols[i + 1] for i in range(len(cols) - 1)):
            continue

        expected = len(rows) * w + len(cols) * h - len(rows) * len(cols)
        if int(np.sum(g == color)) != expected:
            continue

        region_count = (len(rows) - 1) * (len(cols) - 1)
        if region_count >= 2:
            candidates.append((color, rows, cols))

    return candidates


def _regions(rows: list[int], cols: list[int]) -> list[tuple[int, int, int, int]]:
    result: list[tuple[int, int, int, int]] = []
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r0, r1 = rows[i] + 1, rows[i + 1]
            c0, c1 = cols[j] + 1, cols[j + 1]
            assert r0 < r1 and c0 < c1, "each region must have positive area"
            result.append((int(r0), int(r1), int(c0), int(c1)))
    return result


def _region_core_color(region: np.ndarray) -> int:
    h, w = region.shape
    assert h >= 2 and w >= 2, "regions must be nontrivial"

    border = np.concatenate(
        [region[0, :], region[h - 1, :], region[:, 0], region[:, w - 1]]
    )
    border_colors = np.unique(border)
    assert border_colors.size == 1, "region border must be a single background color"
    bg = int(border_colors[0])

    obj_mask = region != bg
    assert bool(obj_mask.any()), "each region must contain a foreground object"
    assert _component_count(obj_mask) == 1, "expected exactly one foreground object per region"

    r0, c0, r1, c1 = _bbox(obj_mask)
    crop = region[r0 : r1 + 1, c0 : c1 + 1]
    counts = Counter(map(int, crop[crop != bg].reshape(-1)))
    assert len(counts) == 2, "object must use one shell color and one core color"

    core_colors = [color for color, count in counts.items() if count == 1]
    assert len(core_colors) == 1, "expected exactly one unique core pixel color"
    core = int(core_colors[0])

    shell_colors = [color for color, count in counts.items() if color != core and count >= 2]
    assert len(shell_colors) == 1, "expected one repeated shell color"
    return core


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse each separator-bounded region to the unique core color of its object,
    ordered by region area from small to large.

    Input structure:
    - One separator color forms an outer frame plus full internal separator rows
      and columns, dividing the grid into rectangular regions.
    - Every region contains one foreground object on a uniform local background.
    - Each object uses exactly two colors: one shell color repeated several
      times, and one core color appearing on exactly one pixel.

    Rule:
    - Find every rectangular region enclosed by the separator lines.
    - For each region, identify the object's unique singleton color.
    - Sort regions by area and output the core colors as an N x 1 column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    candidates = _separator_candidates(g)
    assert len(candidates) == 1, "expected exactly one separator color"
    _sep, rows, cols = candidates[0]

    records: list[tuple[int, int, int, int]] = []
    for r0, r1, c0, c1 in _regions(rows, cols):
        region = g[r0:r1, c0:c1]
        area = int(region.shape[0] * region.shape[1])
        core = _region_core_color(region)
        records.append((area, int(r0), int(c0), core))

    records.sort(key=lambda item: (item[0], item[1], item[2]))
    return np.array([[core] for _area, _r0, _c0, core in records], dtype=int)


# --------------- generator ---------------

PALETTE = tuple(range(10))
LAYOUTS = (
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (4, 1),
)
REGION_SIZE_RANGE = (5, 9)
MAX_GRID_SIDE = 30
MIN_OBJECT_AREA = 4
MAX_OBJECT_AREA = 10
MAX_OBJECT_SPAN = 5
GENERATION_ATTEMPTS = 400


def _sample_region_sizes(count: int) -> list[int]:
    return [random.randint(*REGION_SIZE_RANGE) for _ in range(count)]


def _sample_partition() -> tuple[list[int], list[int]] | None:
    row_count, col_count = random.choice(LAYOUTS)

    for _ in range(200):
        heights = _sample_region_sizes(row_count)
        widths = _sample_region_sizes(col_count)
        total_h = sum(heights) + row_count + 1
        total_w = sum(widths) + col_count + 1
        if total_h > MAX_GRID_SIDE or total_w > MAX_GRID_SIDE:
            continue

        areas = [h * w for h in heights for w in widths]
        if len(set(areas)) != len(areas):
            continue

        return heights, widths
    return None


def _rectangle_shape(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w)}


def _frame_shape(h: int, w: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }


def _plus_shape(h: int, w: int) -> set[tuple[int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    cells = {(mid_r, c) for c in range(w)}
    cells |= {(r, mid_c) for r in range(h)}
    return cells


def _polyomino_shape(h: int, w: int, target_area: int) -> set[tuple[int, int]] | None:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _sample_shape(cell_h: int, cell_w: int) -> set[tuple[int, int]] | None:
    max_h = min(MAX_OBJECT_SPAN, cell_h - 2)
    max_w = min(MAX_OBJECT_SPAN, cell_w - 2)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(120):
        obj_h = random.randint(2, max_h)
        obj_w = random.randint(2, max_w)
        area_limit = min(MAX_OBJECT_AREA, (cell_h * cell_w) // 2 - 1, obj_h * obj_w)
        if area_limit < MIN_OBJECT_AREA:
            continue

        options: list[set[tuple[int, int]]] = []

        rect = _rectangle_shape(obj_h, obj_w)
        if len(rect) <= area_limit:
            options.append(rect)

        if obj_h >= 3 and obj_w >= 3:
            frame = _frame_shape(obj_h, obj_w)
            if MIN_OBJECT_AREA <= len(frame) <= area_limit:
                options.append(frame)

            plus = _plus_shape(obj_h, obj_w)
            if MIN_OBJECT_AREA <= len(plus) <= area_limit:
                options.append(plus)

        target_area = random.randint(MIN_OBJECT_AREA, area_limit)
        poly = _polyomino_shape(obj_h, obj_w, target_area)
        if poly is not None:
            rows = {r for r, _c in poly}
            cols = {c for _r, c in poly}
            if len(rows) >= 2 and len(cols) >= 2:
                options.append(poly)

        if not options:
            continue

        cells = random.choice(options)
        return cells
    return None


def _place_object(
    grid: np.ndarray,
    region: tuple[int, int, int, int],
    shell: int,
    core: int,
) -> None:
    r0, r1, c0, c1 = region
    cell_h = int(r1 - r0)
    cell_w = int(c1 - c0)

    cells = _sample_shape(cell_h, cell_w)
    assert cells is not None, "failed to sample a valid object"

    obj_r0 = random.randint(r0 + 1, r1 - 2)
    obj_c0 = random.randint(c0 + 1, c1 - 2)

    max_r = max(r for r, _c in cells)
    max_c = max(c for _r, c in cells)
    obj_r0 = min(obj_r0, r1 - 1 - max_r - 1)
    obj_c0 = min(obj_c0, c1 - 1 - max_c - 1)
    assert obj_r0 >= r0 + 1 and obj_c0 >= c0 + 1

    translated = [(obj_r0 + r, obj_c0 + c) for r, c in cells]
    for rr, cc in translated:
        grid[rr, cc] = int(shell)

    core_r, core_c = random.choice(translated)
    grid[core_r, core_c] = int(core)


def _expected_output(
    heights: list[int],
    widths: list[int],
    core_by_region: dict[tuple[int, int], int],
) -> np.ndarray:
    records: list[tuple[int, int, int]] = []
    for i, h in enumerate(heights):
        for j, w in enumerate(widths):
            records.append((int(h * w), int(i), int(j)))
    records.sort()
    return np.array([[core_by_region[(i, j)]] for _area, i, j in records], dtype=int)


def _build_grid(
    heights: list[int],
    widths: list[int],
    bg: int,
    sep: int,
    shells: dict[tuple[int, int], int],
    cores: dict[tuple[int, int], int],
) -> np.ndarray:
    total_h = sum(heights) + len(heights) + 1
    total_w = sum(widths) + len(widths) + 1
    grid = np.full((total_h, total_w), sep, dtype=int)

    row_starts = []
    pos = 1
    for h in heights:
        row_starts.append(pos)
        pos += h + 1

    col_starts = []
    pos = 1
    for w in widths:
        col_starts.append(pos)
        pos += w + 1

    for i, r0 in enumerate(row_starts):
        for j, c0 in enumerate(col_starts):
            h = heights[i]
            w = widths[j]
            region = (r0, r0 + h, c0, c0 + w)
            grid[r0 : r0 + h, c0 : c0 + w] = int(bg)
            _place_object(grid, region, shells[(i, j)], cores[(i, j)])

    return grid


def _gen_attempt() -> np.ndarray | None:
    partition = _sample_partition()
    if partition is None:
        return None
    heights, widths = partition

    cell_count = len(heights) * len(widths)
    bg, sep = random.sample(PALETTE, 2)
    remaining = [c for c in PALETTE if c not in (bg, sep)]
    core_colors = random.sample(remaining, cell_count)

    shells: dict[tuple[int, int], int] = {}
    cores: dict[tuple[int, int], int] = {}
    idx = 0
    for i in range(len(heights)):
        for j in range(len(widths)):
            core = int(core_colors[idx])
            shell_choices = [c for c in remaining if c != core]
            shells[(i, j)] = int(random.choice(shell_choices))
            cores[(i, j)] = core
            idx += 1

    try:
        grid = _build_grid(heights, widths, bg, sep, shells, cores)
        expected = _expected_output(heights, widths, cores)
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
