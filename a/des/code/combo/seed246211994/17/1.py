from __future__ import annotations

import random

import numpy as np


COLOR_VALUES = tuple(range(10))
GRID_HEIGHT_BOUNDS = (15, 30)
GRID_WIDTH_BOUNDS = (15, 30)
RECT_COUNT_BOUNDS = (2, 6)
MAX_RECT_SIDE = 7
MAX_RECT_AREA = 24
AREA_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    seen = np.zeros((h, w), dtype=bool)
    rectangles: list[dict[str, int | tuple[int, int, int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0 = int(pts[:, 0].min())
            c0 = int(pts[:, 1].min())
            r1 = int(pts[:, 0].max())
            c1 = int(pts[:, 1].max())
            rect_h = int(r1 - r0 + 1)
            rect_w = int(c1 - c0 + 1)
            area = int(rect_h * rect_w)

            if int(pts.shape[0]) != area:
                raise AssertionError("each foreground object must be a solid rectangle")
            if not bool((g[r0 : r1 + 1, c0 : c1 + 1] == color).all()):
                raise AssertionError("rectangle interior must be monochrome")

            rectangles.append(
                {
                    "color": color,
                    "area": area,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    if not rectangles:
        raise AssertionError("expected at least one foreground rectangle")
    return rectangles


def _render_nested(colors: list[int]) -> np.ndarray:
    size = int(2 * len(colors) - 1)
    out = np.empty((size, size), dtype=int)
    for layer, color in enumerate(colors):
        end = int(size - layer - 1)
        out[layer : end + 1, layer] = int(color)
        out[layer : end + 1, end] = int(color)
        out[layer, layer : end + 1] = int(color)
        out[end, layer : end + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge scattered solid rectangles into nested square rings ordered by area.

    Rule:
    - Find all non-background solid rectangular components.
    - Sort them by area from large to small.
    - Build a square of side 2*n-1, where n is the rectangle count.
    - Draw one-cell-thick nested square rings from outside to inside, using the
      sorted rectangle colors; the smallest rectangle becomes the center pixel.
    """
    rectangles = _extract_rectangles(np.asarray(grid, dtype=int))
    rectangles.sort(key=lambda rect: (-int(rect["area"]), int(rect["bbox"][0]), int(rect["bbox"][1])))
    ordered_colors = [int(rect["color"]) for rect in rectangles]
    return _render_nested(ordered_colors)


def _factor_pairs(area: int, max_h: int, max_w: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for rect_h in range(1, min(max_h, area) + 1):
        if area % rect_h != 0:
            continue
        rect_w = area // rect_h
        if 1 <= rect_w <= max_w:
            pairs.append((int(rect_h), int(rect_w)))
    return pairs


def _orientation(rect_h: int, rect_w: int) -> str:
    if rect_h == rect_w:
        return "square"
    return "wide" if rect_w > rect_h else "tall"


def _sample_rectangles(h: int, w: int, colors: list[int]) -> list[dict[str, int]] | None:
    max_h = min(MAX_RECT_SIDE, h - 2)
    max_w = min(MAX_RECT_SIDE, w - 2)
    available_areas = [
        area
        for area in range(2, MAX_RECT_AREA + 1)
        if _factor_pairs(area, max_h, max_w)
    ]
    if len(available_areas) < len(colors):
        return None

    require_mixed_orientations = len(colors) >= 2 and random.random() < 0.7
    for _ in range(AREA_ATTEMPTS):
        areas = random.sample(available_areas, len(colors))
        if sum(areas) > (h * w) // 4:
            continue
        if h * w - sum(areas) <= max(areas):
            continue

        rectangles: list[dict[str, int]] = []
        orientations: set[str] = set()
        for color, area in zip(colors, areas):
            rect_h, rect_w = random.choice(_factor_pairs(area, max_h, max_w))
            rectangles.append(
                {
                    "color": int(color),
                    "area": int(area),
                    "h": int(rect_h),
                    "w": int(rect_w),
                }
            )
            orientations.add(_orientation(rect_h, rect_w))

        if require_mixed_orientations and len(orientations) < 2:
            continue

        padded_area = sum((rect["h"] + 2) * (rect["w"] + 2) for rect in rectangles)
        if padded_area > h * w:
            continue

        random.shuffle(rectangles)
        return rectangles
    return None


def _place_rectangles(
    h: int,
    w: int,
    bg: int,
    rectangles: list[dict[str, int]],
) -> tuple[np.ndarray, list[dict[str, int | tuple[int, int, int, int]]]] | None:
    grid = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, int | tuple[int, int, int, int]]] = []

    for rect in sorted(rectangles, key=lambda item: (-int(item["area"]), -max(int(item["h"]), int(item["w"])))):
        rect_h = int(rect["h"])
        rect_w = int(rect["w"])
        positions: list[tuple[int, int]] = []

        if h - rect_h <= 1 or w - rect_w <= 1:
            return None

        for r0 in range(1, h - rect_h):
            for c0 in range(1, w - rect_w):
                if bool(occupied[r0 - 1 : r0 + rect_h + 1, c0 - 1 : c0 + rect_w + 1].any()):
                    continue
                positions.append((int(r0), int(c0)))

        if not positions:
            return None

        r0, c0 = random.choice(positions)
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(rect["color"])
        occupied[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        placed.append(
            {
                **rect,
                "bbox": (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1),
            }
        )

    return grid, placed


def _expected_output(rectangles: list[dict[str, int | tuple[int, int, int, int]]]) -> np.ndarray:
    ordered_colors = [
        int(rect["color"])
        for rect in sorted(rectangles, key=lambda item: -int(item["area"]))
    ]
    return _render_nested(ordered_colors)


def _valid_input(grid: np.ndarray, expected_count: int) -> bool:
    rectangles = _extract_rectangles(grid)
    if len(rectangles) != expected_count:
        return False
    areas = [int(rect["area"]) for rect in rectangles]
    if len(set(areas)) != len(areas):
        return False
    bg = _background_color(grid)
    bg_count = int(np.count_nonzero(grid == bg))
    return bg_count > max(areas)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_BOUNDS)
    w = random.randint(*GRID_WIDTH_BOUNDS)
    count = random.randint(*RECT_COUNT_BOUNDS)

    bg = random.choice(COLOR_VALUES)
    palette = [color for color in COLOR_VALUES if color != bg]
    colors = random.sample(palette, count)

    rectangles = _sample_rectangles(h, w, colors)
    if rectangles is None:
        return None

    placed_result = _place_rectangles(h, w, bg, rectangles)
    if placed_result is None:
        return None
    grid, placed = placed_result

    if not _valid_input(grid, count):
        return None

    expected = _expected_output(placed)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
