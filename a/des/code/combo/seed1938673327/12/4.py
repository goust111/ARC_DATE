from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == 0):
        return 0
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict[str, np.ndarray | int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, np.ndarray | int]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            components.append({"color": color, "coords": coords})

    return components


def _solid_odd_rectangle_bbox(grid: np.ndarray, component: dict[str, np.ndarray | int]) -> tuple[int, int, int, int]:
    color = int(component["color"])
    coords = np.asarray(component["coords"], dtype=int)
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    assert height % 2 == 1 and width % 2 == 1, "rectangle sides must be odd"
    assert coords.shape[0] == height * width, "foreground objects must be solid rectangles"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be fully filled"
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each solid rectangle by the midpoints of its four edges.

    Input structure
    - A black background contains several solid monochrome rectangles.
    - Each rectangle has odd height and odd width.

    Rule
    - For each rectangle, erase the whole object.
    - Keep only the midpoint pixel of the top, bottom, left, and right edges.
    - Preserve the object's original color at those four midpoint pixels.

    Key insight
    - The surviving pixels are edge midpoints, not corners.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    components = _foreground_components(g, bg)
    assert components, "expected at least one foreground rectangle"

    out = np.full_like(g, bg)
    for component in components:
        color = int(component["color"])
        r0, c0, r1, c1 = _solid_odd_rectangle_bbox(g, component)
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2

        out[r0, mid_c] = color
        out[r1, mid_c] = color
        out[mid_r, c0] = color
        out[mid_r, c1] = color

    return out


# --------------- generator ---------------

BACKGROUND = 0
FOREGROUND_COLORS = tuple(range(1, 10))
GRID_HEIGHT_RANGE = (10, 22)
GRID_WIDTH_RANGE = (10, 22)
OBJECT_COUNT_RANGE = (1, 5)
RECT_SIDE_MIN = 3
RECT_SIDE_MAX = 11
GRID_BORDER = 1
OBJECT_GAP = 1
MIN_BACKGROUND_RATIO = 0.45
MAX_GENERATION_ATTEMPTS = 400


def _odd_sizes(limit: int) -> list[int]:
    upper = min(limit, RECT_SIDE_MAX)
    return [size for size in range(RECT_SIDE_MIN, upper + 1, 2)]


def _weighted_odd_size(limit: int) -> int | None:
    choices = _odd_sizes(limit)
    if not choices:
        return None
    weights = [limit - size + 2 for size in choices]
    return int(random.choices(choices, weights=weights, k=1)[0])


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _render_input(shape: tuple[int, int], specs: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        r0 = int(spec["top"])
        c0 = int(spec["left"])
        r1 = r0 + int(spec["height"])
        c1 = c0 + int(spec["width"])
        grid[r0:r1, c0:c1] = int(spec["color"])
    return grid


def _render_expected(shape: tuple[int, int], specs: list[dict[str, int]]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        r0 = int(spec["top"])
        c0 = int(spec["left"])
        r1 = r0 + int(spec["height"]) - 1
        c1 = c0 + int(spec["width"]) - 1
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2
        out[r0, mid_c] = color
        out[r1, mid_c] = color
        out[mid_r, c0] = color
        out[mid_r, c1] = color
    return out


def _sample_specs(height: int, width: int) -> list[dict[str, int]] | None:
    max_objects = min(
        OBJECT_COUNT_RANGE[1],
        len(FOREGROUND_COLORS),
        max(OBJECT_COUNT_RANGE[0], (height * width) // 45 + 1),
    )
    if max_objects < OBJECT_COUNT_RANGE[0]:
        return None

    count = random.randint(OBJECT_COUNT_RANGE[0], max_objects)
    colors = random.sample(FOREGROUND_COLORS, count)
    specs: list[dict[str, int]] = []

    rect_h_limit = height - 2 * GRID_BORDER
    rect_w_limit = width - 2 * GRID_BORDER
    for color in colors:
        rect_h = _weighted_odd_size(rect_h_limit)
        rect_w = _weighted_odd_size(rect_w_limit)
        if rect_h is None or rect_w is None:
            return None
        specs.append({"color": color, "height": rect_h, "width": rect_w})

    random.shuffle(specs)
    specs.sort(key=lambda spec: spec["height"] * spec["width"], reverse=True)
    return specs


def _place_specs(height: int, width: int, specs: list[dict[str, int]]) -> list[dict[str, int]] | None:
    placed: list[dict[str, int]] = []
    boxes: list[tuple[int, int, int, int]] = []

    for spec in specs:
        rect_h = int(spec["height"])
        rect_w = int(spec["width"])
        top_range = range(GRID_BORDER, height - rect_h - GRID_BORDER + 1)
        left_range = range(GRID_BORDER, width - rect_w - GRID_BORDER + 1)
        positions = [(top, left) for top in top_range for left in left_range]
        random.shuffle(positions)

        for top, left in positions:
            box = (top, left, top + rect_h - 1, left + rect_w - 1)
            if all(_boxes_separated(box, other, OBJECT_GAP) for other in boxes):
                placed.append(
                    {
                        "color": int(spec["color"]),
                        "height": rect_h,
                        "width": rect_w,
                        "top": int(top),
                        "left": int(left),
                    }
                )
                boxes.append(box)
                break
        else:
            return None

    return placed


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    specs = _sample_specs(height, width)
    if specs is None:
        return None

    placed = _place_specs(height, width, specs)
    if placed is None:
        return None

    grid = _render_input((height, width), placed)
    bg_count = int(np.count_nonzero(grid == BACKGROUND))
    largest_object = max(spec["height"] * spec["width"] for spec in placed)
    if bg_count <= largest_object or bg_count / grid.size < MIN_BACKGROUND_RATIO:
        return None

    expected = _render_expected((height, width), placed)
    return grid if np.array_equal(main(grid), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {MAX_GENERATION_ATTEMPTS} attempts")
