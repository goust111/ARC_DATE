from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 500
LAYOUT_ATTEMPTS = 200
RING_COUNT_RANGE = (2, 5)
OUTER_MAX_BY_COUNT = {2: 20, 3: 18, 4: 16, 5: 14}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _ring_info(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    values = np.unique(grid[coords[:, 0], coords[:, 1]])
    if values.size != 1 or int(values[0]) == BLACK:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = int(values[0])

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if int(coords.shape[0]) != int(border.sum()):
        return None
    if not bool(np.all(sub[border] == color)):
        return None
    if not bool(np.all(sub[~border] == BLACK)):
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "height": h,
        "width": w,
        "area": h * w,
    }


def _extract_rings(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rings: list[dict] = []

    for coords in _components(g != BLACK):
        info = _ring_info(g, coords)
        assert info is not None, "every foreground component must be a monochrome rectangular ring"
        rings.append(info)

    assert rings, "expected at least one ring"
    areas = [int(ring["area"]) for ring in rings]
    assert len(areas) == len(set(areas)), "ring areas must be unique"
    rings.sort(key=lambda ring: int(ring["area"]), reverse=True)
    return rings


def _draw_ring(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered rectangular rings into a centered largest-to-smallest nesting.

    Input structure:
    - The background is black.
    - The foreground consists of multiple single-color rectangular rings with 1-pixel-thick borders.
    - Each ring is isolated from the others, and each ring encloses only black cells.
    - The rings have distinct outer rectangle areas, so their size order is unique.

    Rule:
    - Detect every monochrome rectangular ring and measure its outer bounding-box area.
    - Sort the rings from largest to smallest by that outer area.
    - Create an output whose size matches the largest ring's outer dimensions.
    - Place the largest ring at the outside, then place each next smaller ring centered inside the previous ring's interior; when perfect centering is impossible, bias upward and leftward.

    Key insight:
    - The input locations are irrelevant after recognition: the task is to recover the size order of the rings and rebuild them as a concentric nesting chain.
    """
    rings = _extract_rings(np.asarray(grid, dtype=int))
    outer = rings[0]
    out = np.zeros((int(outer["height"]), int(outer["width"])), dtype=int)

    current_top = 0
    current_left = 0
    current_height = int(outer["height"])
    current_width = int(outer["width"])

    for index, ring in enumerate(rings):
        height = int(ring["height"])
        width = int(ring["width"])
        if index == 0:
            top = 0
            left = 0
        else:
            inner_height = current_height - 2
            inner_width = current_width - 2
            assert height <= inner_height and width <= inner_width, "nested ring must fit inside parent interior"
            top = current_top + 1 + (inner_height - height) // 2
            left = current_left + 1 + (inner_width - width) // 2
        _draw_ring(out, top, left, height, width, int(ring["color"]))
        current_top = top
        current_left = left
        current_height = height
        current_width = width

    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts
    remaining = total
    result: list[int] = []
    for i in range(parts - 1):
        value = random.randint(1, remaining - (parts - i - 1))
        result.append(value)
        remaining -= value
    result.append(remaining)
    return result


def _nonnegative_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= 0
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1)) if parts > 1 else []
    result: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        result.append(cut - prev - 1)
        prev = cut
    return result


def _sample_dimensions() -> list[tuple[int, int]]:
    for _ in range(GEN_ATTEMPTS):
        count = random.randint(*RING_COUNT_RANGE)
        outer_min = 3 + 2 * (count - 1)
        outer_max = OUTER_MAX_BY_COUNT[count]
        outer_h = random.randint(outer_min, outer_max)
        outer_w = random.randint(outer_min, outer_max)

        dims = [(outer_h, outer_w)]
        current_h, current_w = outer_h, outer_w
        ok = True
        for remaining_children in range(count - 1, 0, -1):
            child_min = 3 + 2 * (remaining_children - 1)
            max_shrink_h = current_h - child_min
            max_shrink_w = current_w - child_min
            if max_shrink_h < 2 or max_shrink_w < 2:
                ok = False
                break
            shrink_h = random.randint(2, max_shrink_h)
            shrink_w = random.randint(2, max_shrink_w)
            current_h -= shrink_h
            current_w -= shrink_w
            dims.append((current_h, current_w))
        if not ok:
            continue

        areas = [h * w for h, w in dims]
        if len(areas) != len(set(areas)):
            continue
        if sum(areas) > 520:
            continue
        return dims
    raise RuntimeError("failed to sample a valid ring-dimension chain")


def _layout_rows(
    rings: list[tuple[int, int, int]]
) -> tuple[int, int, list[tuple[int, int, int, int, int]]] | None:
    count = len(rings)
    for _ in range(LAYOUT_ATTEMPTS):
        shuffled = rings[:]
        random.shuffle(shuffled)
        row_count = random.randint(1, min(count, 3))
        row_sizes = _positive_composition(count, row_count)

        rows: list[list[tuple[int, int, int]]] = []
        start = 0
        for size in row_sizes:
            rows.append(shuffled[start : start + size])
            start += size

        horizontal_gaps: list[list[int]] = []
        row_widths: list[int] = []
        row_heights: list[int] = []
        for row in rows:
            gaps = [random.randint(1, 3) for _ in range(len(row) - 1)]
            width = sum(int(item[1]) for item in row) + sum(gaps)
            height = max(int(item[0]) for item in row)
            horizontal_gaps.append(gaps)
            row_widths.append(width)
            row_heights.append(height)

        vertical_gaps = [random.randint(1, 3) for _ in range(row_count - 1)]
        min_h = sum(row_heights) + sum(vertical_gaps)
        min_w = max(row_widths)
        if min_h > 30 or min_w > 30:
            continue

        height = random.randint(min_h, 30)
        width = random.randint(min_w, 30)
        vertical_slack = _nonnegative_composition(height - min_h, row_count + 1)

        placements: list[tuple[int, int, int, int, int]] = []
        y = vertical_slack[0]
        for row_index, row in enumerate(rows):
            row_height = row_heights[row_index]
            row_width = row_widths[row_index]
            slack = _nonnegative_composition(width - row_width, len(row) + 1)
            x = slack[0]
            for item_index, (ring_h, ring_w, color) in enumerate(row):
                top = y + random.randint(0, row_height - ring_h)
                placements.append((top, x, ring_h, ring_w, color))
                x += ring_w
                if item_index < len(row) - 1:
                    x += horizontal_gaps[row_index][item_index] + slack[item_index + 1]
            y += row_height
            if row_index < row_count - 1:
                y += vertical_gaps[row_index] + vertical_slack[row_index + 1]

        return height, width, placements
    return None


def _build_grid(
    placements: list[tuple[int, int, int, int, int]], shape: tuple[int, int]
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for top, left, height, width, color in placements:
        _draw_ring(grid, top, left, height, width, color)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        dims = _sample_dimensions()
        colors = random.sample(COLORS, len(dims))
        ring_specs = [(h, w, color) for (h, w), color in zip(dims, colors)]
        layout = _layout_rows(ring_specs)
        if layout is None:
            continue
        height, width, placements = layout
        grid = _build_grid(placements, (height, width))
        rings = _extract_rings(grid)
        if len(rings) != len(dims):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
