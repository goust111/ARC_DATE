from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
LEGEND_SEPARATOR = 5
LEGEND_H = 4
LEGEND_W = 10
GEN_RETRIES = 400
LEGEND_GAP_RANGE = (1, 3)
OBJECT_COUNT_OPTIONS = (3, 4)
PADDING = 2
COLOR_POOL = tuple(range(1, 10))

DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

PATTERN_TEMPLATES = (
    np.array([[0, 0], [0, 0]], dtype=int),
    np.array([[0, 1], [0, 1]], dtype=int),
    np.array([[0, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 0]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[0, 0], [0, 1]], dtype=int),
)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _legend_separator_mask() -> np.ndarray:
    mask = np.zeros((LEGEND_H, LEGEND_W), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, 3] = True
    mask[:, 6] = True
    mask[:, 9] = True
    return mask


LEGEND_SEPARATOR_MASK = _legend_separator_mask()


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, np.ndarray]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, dict[int, np.ndarray]]] = []
    for r0 in range(h - LEGEND_H + 1):
        for c0 in range(w - LEGEND_W + 1):
            patch = grid[r0 : r0 + LEGEND_H, c0 : c0 + LEGEND_W]
            sep = int(patch[0, 0])
            if sep == BG:
                continue
            if not np.all(patch[LEGEND_SEPARATOR_MASK] == sep):
                continue
            if np.any(patch[~LEGEND_SEPARATOR_MASK] == sep):
                continue
            if np.any(patch[~LEGEND_SEPARATOR_MASK] == BG):
                continue
            patterns = {
                3: patch[1:3, 1:3].copy(),
                4: patch[1:3, 4:6].copy(),
                5: patch[1:3, 7:9].copy(),
            }
            if len({tuple(pattern.flatten()) for pattern in patterns.values()}) != 3:
                continue
            candidates.append((r0, c0, patterns))
    assert candidates, "legend not found"
    r0, c0, patterns = max(candidates, key=lambda item: (item[0], item[1]))
    return (int(r0), int(c0), int(r0 + LEGEND_H - 1), int(c0 + LEGEND_W - 1)), patterns


def _polygon_vertices(coords: np.ndarray) -> np.ndarray:
    points = sorted({(int(c), int(r)) for r, c in np.asarray(coords, dtype=int)})
    assert len(points) >= 3

    def cross(
        origin: tuple[int, int],
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> int:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[int, int]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    assert len(hull) >= 3
    return np.asarray([(r, c) for c, r in hull], dtype=int)


def _point_in_polygon(row: int, col: int, vertices: np.ndarray) -> bool:
    x = float(col)
    y = float(row)
    inside = False
    n = len(vertices)
    for i in range(n):
        y1, x1 = map(float, vertices[i])
        y2, x2 = map(float, vertices[(i + 1) % n])
        if (y1 > y) != (y2 > y):
            cross_x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
            if x < cross_x:
                inside = not inside
    return inside


def _fill_component(out: np.ndarray, coords: np.ndarray, pattern: np.ndarray) -> None:
    cells = np.asarray(coords, dtype=int)
    r0, c0 = map(int, cells.min(axis=0))
    r1, c1 = map(int, cells.max(axis=0))
    border = {tuple(map(int, cell)) for cell in cells}
    vertices = _polygon_vertices(cells)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in border or out[r, c] != BG:
                continue
            if _point_in_polygon(r, c, vertices):
                out[r, c] = int(pattern[(r - r0) % 2, (c - c0) % 2])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed polygon container with the legend pattern indexed by its corner count.

    Input structure:
    - The background is black.
    - The bottom of the grid contains a legend framed by one separator color, split into three 2x2 pattern cells.
    - Above the legend sit several one-pixel-thick closed outlines in distinct colors; each outline is a simple loop shaped like a triangle, rectangle, or pentagon.

    Rule:
    - Locate the three legend cells and read them left-to-right as the fill patterns for 3-corner, 4-corner, and 5-corner containers.
    - Trace every outline loop, count its corners by counting direction-change vertices along the loop, and choose the matching 2x2 legend pattern.
    - Fill only the enclosed background pixels of that container by periodically tiling the chosen pattern, aligned to the outline bbox's top-left corner; keep the outlines and legend unchanged.

    Key insight:
    - Color does not identify the fill; the decisive feature is the outline geometry, specifically how many straight edge runs the closed loop has.
    """
    g = np.asarray(grid, dtype=int)
    legend_bbox, patterns = _find_legend(g)
    r0, c0, r1, c1 = legend_bbox

    work = (g != BG).copy()
    work[r0 : r1 + 1, c0 : c1 + 1] = False

    out = g.copy()
    for component in _connected_components(work):
        corners = len(_polygon_vertices(component))
        assert corners in patterns, f"unsupported corner count: {corners}"
        _fill_component(out, component, patterns[corners])
    return out


def _walk_trace(segments: list[tuple[tuple[int, int], int]]) -> np.ndarray:
    r = c = 0
    points = [(0, 0)]
    for (dr, dc), length in segments:
        for _ in range(int(length)):
            r += int(dr)
            c += int(dc)
            points.append((r, c))
    assert (r, c) == (0, 0)
    points = points[:-1]
    assert len(points) == len(set(points)), "trace self-intersects"
    return np.asarray(points, dtype=int)


def _rotate90(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=int)
    return np.column_stack((arr[:, 1], -arr[:, 0]))


def _transform_trace(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=int).copy()
    for _ in range(random.randrange(4)):
        arr = _rotate90(arr)
    if random.choice([False, True]):
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return arr


def _triangle_trace() -> np.ndarray:
    side = random.randint(4, 7)
    trace = _walk_trace([((1, 1), side), ((0, -1), side), ((-1, 0), side)])
    return _transform_trace(trace)


def _rectangle_trace() -> np.ndarray:
    height = random.randint(4, 8)
    width = random.randint(5, 9)
    trace = _walk_trace(
        [
            ((0, 1), width - 1),
            ((1, 0), height - 1),
            ((0, -1), width - 1),
            ((-1, 0), height - 1),
        ]
    )
    return _transform_trace(trace)


def _pentagon_trace() -> np.ndarray:
    top = random.randint(4, 7)
    slope = random.randint(2, 4)
    trace = _walk_trace(
        [
            ((0, 1), top),
            ((1, 1), slope),
            ((1, -1), slope),
            ((0, -1), top),
            ((-1, 0), 2 * slope),
        ]
    )
    return _transform_trace(trace)


def _sample_pattern(colors: list[int], used: set[tuple[int, ...]]) -> np.ndarray:
    templates = list(PATTERN_TEMPLATES)
    random.shuffle(templates)
    for template in templates:
        slots = int(template.max()) + 1
        chosen = random.sample(colors, slots)
        pattern = np.asarray(chosen, dtype=int)[template]
        key = tuple(map(int, pattern.flatten()))
        if key not in used:
            used.add(key)
            return pattern
    raise AssertionError("failed to sample a distinct legend pattern")


def _build_legend(patterns: dict[int, np.ndarray]) -> np.ndarray:
    legend = np.full((LEGEND_H, LEGEND_W), LEGEND_SEPARATOR, dtype=int)
    legend[1:3, 1:3] = patterns[3]
    legend[1:3, 4:6] = patterns[4]
    legend[1:3, 7:9] = patterns[5]
    return legend


def _sprite_mask_from_trace(trace: np.ndarray) -> np.ndarray:
    r0, c0 = map(int, trace.min(axis=0))
    r1, c1 = map(int, trace.max(axis=0))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = trace - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _place_traces(
    traces: list[tuple[int, np.ndarray, int]],
    height: int,
    width: int,
) -> list[tuple[np.ndarray, int, int]]:
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[tuple[np.ndarray, int, int]] = []
    order = traces[:]
    random.shuffle(order)
    for color, trace, corners in order:
        sprite = _sprite_mask_from_trace(trace)
        sh, sw = sprite.shape
        candidates = [
            (r0, c0)
            for r0 in range(1, height - sh)
            for c0 in range(1, width - sw)
        ]
        random.shuffle(candidates)
        spot = None
        for r0, c0 in candidates:
            rr0 = max(0, r0 - PADDING)
            cc0 = max(0, c0 - PADDING)
            rr1 = min(height, r0 + sh + PADDING)
            cc1 = min(width, c0 + sw + PADDING)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            spot = (r0, c0)
            occupied[rr0:rr1, cc0:cc1] = True
            break
        if spot is None:
            raise ValueError("could not place all objects")
        abs_trace = trace + np.array(spot, dtype=int)
        placed.append((abs_trace, int(color), int(corners)))
    return placed


def _oracle_output(grid: np.ndarray, placed: list[tuple[np.ndarray, int, int]], patterns: dict[int, np.ndarray]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for trace, _color, corners in placed:
        _fill_component(out, trace, patterns[corners])
    return out


def _gen_attempt() -> np.ndarray | None:
    object_count = random.choice(OBJECT_COUNT_OPTIONS)
    corner_counts = [3, 4, 5]
    if object_count == 4:
        corner_counts.append(random.choice([3, 4, 5]))
    random.shuffle(corner_counts)

    available = [color for color in COLOR_POOL if color != LEGEND_SEPARATOR]
    border_colors = random.sample(available, len(corner_counts))
    pattern_colors = [color for color in available if color not in border_colors]

    used_patterns: set[tuple[int, ...]] = set()
    patterns = {
        3: _sample_pattern(pattern_colors, used_patterns),
        4: _sample_pattern(pattern_colors, used_patterns),
        5: _sample_pattern(pattern_colors, used_patterns),
    }
    legend = _build_legend(patterns)

    traces: list[tuple[int, np.ndarray, int]] = []
    for color, corners in zip(border_colors, corner_counts, strict=False):
        if corners == 3:
            trace = _triangle_trace()
        elif corners == 4:
            trace = _rectangle_trace()
        else:
            trace = _pentagon_trace()
        assert len(_polygon_vertices(trace)) == corners
        traces.append((int(color), trace, int(corners)))

    legend_gap = random.randint(*LEGEND_GAP_RANGE)
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    top_height = height - LEGEND_H - legend_gap
    if top_height < 8 or width < LEGEND_W + 2:
        return None
    if top_height <= max(int(trace[:, 0].max()) + 2 for _color, trace, _corners in traces):
        return None
    if width <= max(int(trace[:, 1].max()) + 2 for _color, trace, _corners in traces):
        return None

    legend_r0 = height - LEGEND_H
    legend_c0 = random.randint(0, width - LEGEND_W)
    try:
        placed = _place_traces(traces, top_height, width)
    except ValueError:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[legend_r0 : legend_r0 + LEGEND_H, legend_c0 : legend_c0 + LEGEND_W] = legend
    for trace, color, _corners in placed:
        grid[trace[:, 0], trace[:, 1]] = int(color)

    expected = _oracle_output(grid, placed, patterns)
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
