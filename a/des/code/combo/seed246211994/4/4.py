from __future__ import annotations

from collections import deque
import random

import numpy as np


PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 300
MIN_RECTANGLES = 2
MAX_RECTANGLES = 7
MAX_SIDE = 7
MAX_AREA = 24
MIN_GAP = 2
MAX_GAP = 4
GRID_MIN = 12
GRID_MAX = 30


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], tuple[int, int, int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[tuple[list[tuple[int, int]], tuple[int, int, int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)

                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comps.append((cells, (int(r0), int(c0), int(r1), int(c1))))
    return comps


def _extract_rectangles(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    rectangles: list[tuple[int, int]] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for cells, (r0, c0, r1, c1) in _components(g == color):
            area = len(cells)
            box_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            if area != box_area:
                raise AssertionError("each foreground object must be a solid rectangle")
            rectangles.append((area, color))

    if not rectangles:
        raise AssertionError("expected at least one foreground rectangle")

    areas = [area for area, _color in rectangles]
    if len(areas) != len(set(areas)):
        raise AssertionError("rectangle areas must be globally unique")

    return rectangles


def _build_output(colors_in_rank_order: list[int]) -> np.ndarray:
    n = len(colors_in_rank_order)
    size = 2 * n - 1
    out = np.empty((size, size), dtype=int)
    for layer, color in enumerate(colors_in_rank_order):
        out[layer : size - layer, layer : size - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather the scattered rectangles, sort them by area, and redraw them as nested square rings.
    """
    rectangles = _extract_rectangles(np.asarray(grid, dtype=int))
    rectangles.sort(key=lambda item: int(item[0]), reverse=True)
    return _build_output([color for _area, color in rectangles])


def _random_split(total: int, parts: int) -> list[int]:
    pieces = [0] * parts
    for _ in range(total):
        pieces[random.randrange(parts)] += 1
    return pieces


def _area_to_shapes() -> dict[int, list[tuple[int, int]]]:
    by_area: dict[int, list[tuple[int, int]]] = {}
    for h in range(1, MAX_SIDE + 1):
        for w in range(1, MAX_SIDE + 1):
            area = h * w
            if area > MAX_AREA:
                continue
            by_area.setdefault(area, []).append((h, w))
    return by_area


AREA_TO_SHAPES = _area_to_shapes()
CANDIDATE_AREAS = tuple(sorted(AREA_TO_SHAPES))


def _sample_rectangles() -> tuple[list[dict[str, int]], list[int], int] | None:
    n_rectangles = random.randint(MIN_RECTANGLES, MAX_RECTANGLES)
    if len(CANDIDATE_AREAS) < n_rectangles:
        return None

    bg = random.choice(PALETTE)
    colors = random.sample([c for c in PALETTE if c != bg], 2)
    start_index = random.randrange(2)
    rank_colors = [colors[(start_index + i) % 2] for i in range(n_rectangles)]
    areas = sorted(random.sample(CANDIDATE_AREAS, n_rectangles), reverse=True)

    rectangles: list[dict[str, int]] = []
    for area, color in zip(areas, rank_colors):
        h, w = random.choice(AREA_TO_SHAPES[int(area)])
        rectangles.append({"area": int(area), "color": int(color), "h": int(h), "w": int(w)})
    return rectangles, rank_colors, bg


def _arrange_rows(rectangles: list[dict[str, int]]) -> list[list[dict[str, int]]]:
    order = rectangles[:]
    random.shuffle(order)
    row_count = random.randint(1, min(4, len(order)))
    rows: list[list[dict[str, int]]] = [[] for _ in range(row_count)]

    for i, rect in enumerate(order):
        if i < row_count:
            rows[i].append(rect)
            continue
        widths = [sum(item["w"] for item in row) + MIN_GAP * max(0, len(row) - 1) for row in rows]
        min_width = min(widths)
        candidates = [idx for idx, width in enumerate(widths) if width == min_width]
        rows[random.choice(candidates)].append(rect)

    for row in rows:
        random.shuffle(row)
    return rows


def _place_rows(
    rectangles: list[dict[str, int]],
    bg: int,
) -> np.ndarray | None:
    rows = _arrange_rows(rectangles)
    row_heights = [max(rect["h"] for rect in row) for row in rows]
    row_gaps = [[random.randint(MIN_GAP, MAX_GAP) for _ in range(len(row) - 1)] for row in rows]
    row_widths = [
        sum(rect["w"] for rect in row) + sum(gaps)
        for row, gaps in zip(rows, row_gaps)
    ]
    vertical_gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(len(rows) - 1)]

    core_h = sum(row_heights) + sum(vertical_gaps)
    core_w = max(row_widths)
    base_h = core_h + 2
    base_w = core_w + 2
    if base_h > GRID_MAX or base_w > GRID_MAX:
        return None

    h = max(base_h + random.randint(0, 6), random.randint(GRID_MIN, GRID_MAX))
    w = max(base_w + random.randint(0, 8), random.randint(GRID_MIN, GRID_MAX))
    if h > GRID_MAX or w > GRID_MAX:
        return None

    total_area = sum(rect["area"] for rect in rectangles)
    color_totals: dict[int, int] = {}
    for rect in rectangles:
        color_totals[rect["color"]] = color_totals.get(rect["color"], 0) + rect["area"]
    max_color_total = max(color_totals.values())

    while h * w - total_area <= max_color_total:
        if w < GRID_MAX and (w <= h or h == GRID_MAX):
            w += 1
        elif h < GRID_MAX:
            h += 1
        else:
            return None

    grid = np.full((h, w), int(bg), dtype=int)

    extra_vertical = h - base_h
    vertical_extra = _random_split(extra_vertical, len(rows) + 1)
    y = 1 + vertical_extra[0]

    for row_index, row in enumerate(rows):
        row_height = row_heights[row_index]
        row_width = row_widths[row_index]
        slack_x = w - (row_width + 2)
        start_x = 1 + random.randint(0, slack_x)
        x = start_x

        for rect_index, rect in enumerate(row):
            offset_y = random.randint(0, row_height - rect["h"])
            r0 = y + offset_y
            c0 = x
            grid[r0 : r0 + rect["h"], c0 : c0 + rect["w"]] = int(rect["color"])
            x += rect["w"]
            if rect_index < len(row) - 1:
                x += row_gaps[row_index][rect_index]

        y += row_height
        if row_index < len(rows) - 1:
            y += vertical_gaps[row_index] + vertical_extra[row_index + 1]

    return grid


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_rectangles()
    if sampled is None:
        return None
    rectangles, rank_colors, bg = sampled
    grid = _place_rows(rectangles, bg)
    if grid is None:
        return None

    try:
        if not np.array_equal(main(grid), _build_output(rank_colors)):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
