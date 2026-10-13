import random

import numpy as np


BACKGROUND = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GENERATION_ATTEMPTS = 400
GROUP_ATTEMPTS = 500

SHAPE_OFFSETS: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((0, 0),),
    2: ((0, -1), (0, 0), (0, 1)),
    3: ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
    4: tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)),
}


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _shape_cells(count: int, anchor_r: int, anchor_c: int, height: int, width: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for dr, dc in SHAPE_OFFSETS[count]:
        rr = anchor_r + dr
        cc = anchor_c + dc
        if 0 <= rr < height and 0 <= cc < width:
            cells.add((rr, cc))
    return cells


def _expand_cells(cells: set[tuple[int, int]], height: int, width: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    expanded.add((rr, cc))
    return expanded


def _cluster_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _cluster_buffer(points: list[tuple[int, int]], height: int, width: int, pad: int = 1) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _cluster_bbox(points)
    buffered: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - pad), min(height, r1 + pad + 1)):
        for c in range(max(0, c0 - pad), min(width, c1 + pad + 1)):
            buffered.add((r, c))
    return buffered


def _max_manhattan(points: list[tuple[int, int]]) -> int:
    best = 0
    for i in range(len(points)):
        r1, c1 = points[i]
        for j in range(i + 1, len(points)):
            r2, c2 = points[j]
            best = max(best, abs(r1 - r2) + abs(c1 - c2))
    return best


def _anchor_of(points: np.ndarray) -> tuple[int, int]:
    return _round_half_up(points[:, 0].mean()), _round_half_up(points[:, 1].mean())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Regroup each color's scattered pixels into a count-coded shape centered at that color's visual centroid.

    Input structure:
    - The background is black.
    - Each non-black color appears as a small loose cluster of 1 to 4 pixels.
    - Different colors use different pixel counts and are spatially separated from one another.

    Rule:
    - For each non-black color, count its pixels and compute the rounded mean row and column of those pixels.
    - Erase the foreground back to a black grid of the same size.
    - Redraw that color at the rounded centroid using this map:
      1 pixel -> single point, 2 pixels -> horizontal 3-cell segment,
      3 pixels -> 5-cell cross, 4 pixels -> filled 3x3 square.
    - Clip any part of a shape that would fall outside the grid.

    Key insight:
    - Two summaries drive the result independently: the pixel count selects the geometry, and the cluster centroid selects the placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = map(int, g.shape)
    out = np.full((height, width), BACKGROUND, dtype=int)

    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    for color in sorted(colors):
        points = np.argwhere(g == color)
        count = int(points.shape[0])
        assert 1 <= count <= 4, "each foreground color must occur 1 to 4 times"
        anchor_r, anchor_c = _anchor_of(points)
        for rr, cc in _shape_cells(count, anchor_r, anchor_c, height, width):
            out[rr, cc] = color

    return out


def _sample_group(
    count: int,
    color: int,
    grid: np.ndarray,
    occupied_input: set[tuple[int, int]],
    forbidden_input: set[tuple[int, int]],
    forbidden_output: set[tuple[int, int]],
) -> bool:
    height, width = map(int, grid.shape)
    full_shape_size = len(SHAPE_OFFSETS[count])

    for _ in range(GROUP_ATTEMPTS):
        box_h = random.randint(1 if count <= 2 else 2, 5)
        box_w = random.randint(1 if count <= 2 else 2, 5)
        if box_h * box_w < count:
            continue

        edge_bias = random.random() < 0.3
        if edge_bias:
            side = random.choice(("top", "bottom", "left", "right"))
            if side == "top":
                r0 = 0
                c0 = random.randint(0, width - box_w)
            elif side == "bottom":
                r0 = height - box_h
                c0 = random.randint(0, width - box_w)
            elif side == "left":
                r0 = random.randint(0, height - box_h)
                c0 = 0
            else:
                r0 = random.randint(0, height - box_h)
                c0 = width - box_w
        else:
            r0 = random.randint(0, height - box_h)
            c0 = random.randint(0, width - box_w)

        cells = [
            (r, c)
            for r in range(r0, r0 + box_h)
            for c in range(c0, c0 + box_w)
            if (r, c) not in occupied_input
        ]
        if len(cells) < count:
            continue

        points = random.sample(cells, count)
        points.sort()

        if count > 1 and _max_manhattan(points) < 2:
            continue

        cluster_guard = _cluster_buffer(points, height, width, pad=1)
        if cluster_guard & forbidden_input:
            continue

        point_array = np.array(points, dtype=int)
        anchor_r, anchor_c = _anchor_of(point_array)
        shape = _shape_cells(count, anchor_r, anchor_c, height, width)
        if len(shape) != full_shape_size and random.random() < 0.7:
            # Keep clipping as an occasional case rather than the default.
            continue

        shape_guard = _expand_cells(shape, height, width, pad=1)
        if shape_guard & forbidden_output:
            continue

        for r, c in points:
            grid[r, c] = color
            occupied_input.add((r, c))
        forbidden_input.update(cluster_guard)
        forbidden_output.update(shape_guard)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(15, 20)
    width = random.randint(15, 20)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    color_count = random.randint(2, 4)
    counts = random.sample([1, 2, 3, 4], color_count)
    colors = random.sample(COLOR_POOL, color_count)

    occupied_input: set[tuple[int, int]] = set()
    forbidden_input: set[tuple[int, int]] = set()
    forbidden_output: set[tuple[int, int]] = set()

    order = list(zip(colors, counts))
    random.shuffle(order)
    for color, count in order:
        ok = _sample_group(count, color, grid, occupied_input, forbidden_input, forbidden_output)
        if not ok:
            return None

    seen_colors = [int(color) for color in np.unique(grid) if int(color) != BACKGROUND]
    if len(seen_colors) != color_count:
        return None
    if len({int(np.count_nonzero(grid == color)) for color in seen_colors}) != color_count:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
