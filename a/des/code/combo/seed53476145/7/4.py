from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return components


def _assert_solid_rectangle(grid: np.ndarray, component: dict[str, object]) -> None:
    r0, c0, r1, c1 = component["bbox"]  # type: ignore[index]
    color = int(component["color"])
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert int(component["area"]) == height * width, "each object must be a solid rectangle"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be fully filled"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode each solid rectangle's width and height as adjacent grey ruler bars.

    For every monochromatic 4-connected solid rectangle on a black background:
    - draw a grey vertical bar immediately to its left with the same height,
    - draw a grey horizontal bar immediately above it with the same width,
    - keep the original rectangle unchanged,
    - clip bars that would fall outside the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g)
    assert components, "expected at least one foreground rectangle"

    out = g.copy()
    for component in components:
        _assert_solid_rectangle(g, component)
        r0, c0, r1, c1 = component["bbox"]  # type: ignore[index]

        if c0 > 0:
            out[r0 : r1 + 1, c0 - 1] = GREY
        if r0 > 0:
            out[r0 - 1, c0 : c1 + 1] = GREY

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (8, 18)
GRID_WIDTH_RANGE = (8, 18)
MAX_OBJECT_COUNT = 5
RECT_HEIGHT_RANGE = (1, 5)
RECT_WIDTH_RANGE = (1, 6)
RECT_AREA_RANGE = (1, 18)
GENERATION_RETRIES = 400

MODE_PRIORITY = {"corner": 0, "top": 1, "left": 1, "interior": 2}


def _sample_object_count(height: int, width: int) -> int:
    max_count = min(MAX_OBJECT_COUNT, max(1, (height * width) // 28))
    return random.randint(1, max_count)


def _choose_modes(count: int) -> list[str]:
    available_specials = ["top", "left"] if count == 1 else ["top", "left", "corner"]
    special_count = random.randint(0, min(2, count, len(available_specials)))
    specials = random.sample(available_specials, k=special_count)
    modes = specials + ["interior"] * (count - special_count)
    random.shuffle(modes)
    return modes


def _position_ranges(
    height: int,
    width: int,
    rect_h: int,
    rect_w: int,
    mode: str,
) -> tuple[range | tuple[int, ...], range | tuple[int, ...]]:
    if rect_h > height or rect_w > width:
        return range(0), range(0)

    if mode == "interior":
        return range(1, height - rect_h + 1), range(1, width - rect_w + 1)
    if mode == "top":
        return (0,), range(1, width - rect_w + 1)
    if mode == "left":
        return range(1, height - rect_h + 1), (0,)
    assert mode == "corner"
    return (0,), (0,)


def _reserve_bounds(
    top: int,
    left: int,
    rect_h: int,
    rect_w: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, top - 1),
        min(height - 1, top + rect_h),
        max(0, left - 1),
        min(width - 1, left + rect_w),
    )


def _valid_positions(
    occupied: np.ndarray,
    rect_h: int,
    rect_w: int,
    mode: str,
) -> list[tuple[int, int]]:
    height, width = map(int, occupied.shape)
    row_range, col_range = _position_ranges(height, width, rect_h, rect_w, mode)
    positions: list[tuple[int, int]] = []

    for top in row_range:
        for left in col_range:
            rr0, rr1, cc0, cc1 = _reserve_bounds(top, left, rect_h, rect_w, height, width)
            if not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                positions.append((top, left))

    return positions


def _size_options(occupied: np.ndarray, mode: str) -> list[tuple[int, int, list[tuple[int, int]]]]:
    height, width = map(int, occupied.shape)
    options: list[tuple[int, int, list[tuple[int, int]]]] = []

    for rect_h in range(RECT_HEIGHT_RANGE[0], min(RECT_HEIGHT_RANGE[1], height) + 1):
        for rect_w in range(RECT_WIDTH_RANGE[0], min(RECT_WIDTH_RANGE[1], width) + 1):
            area = rect_h * rect_w
            if not (RECT_AREA_RANGE[0] <= area <= RECT_AREA_RANGE[1]):
                continue
            positions = _valid_positions(occupied, rect_h, rect_w, mode)
            if positions:
                options.append((rect_h, rect_w, positions))

    return options


def _place_rectangles(height: int, width: int, count: int) -> list[dict[str, int | str]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    rectangles: list[dict[str, int | str]] = []
    modes = sorted(_choose_modes(count), key=lambda mode: MODE_PRIORITY[mode])

    for mode in modes:
        options = _size_options(occupied, mode)
        if not options:
            return None

        rect_h, rect_w, positions = random.choice(options)
        top, left = random.choice(positions)
        rr0, rr1, cc0, cc1 = _reserve_bounds(top, left, rect_h, rect_w, height, width)
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        rectangles.append(
            {
                "mode": mode,
                "top": top,
                "left": left,
                "height": rect_h,
                "width": rect_w,
            }
        )

    return rectangles


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    count = _sample_object_count(height, width)
    rectangles = _place_rectangles(height, width, count)
    if rectangles is None:
        return None

    colors = random.sample(RECT_COLORS, k=count)
    grid = np.full((height, width), BLACK, dtype=int)

    for rect, color in zip(rectangles, colors):
        top = int(rect["top"])
        left = int(rect["left"])
        rect_h = int(rect["height"])
        rect_w = int(rect["width"])
        grid[top : top + rect_h, left : left + rect_w] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(_find_components(grid)) != count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
