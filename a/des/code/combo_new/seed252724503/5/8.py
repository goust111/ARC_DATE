from __future__ import annotations

import random

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")
SIZES = (1, 2, 3, 4)
GEN_ATTEMPTS = 500
PLACEMENT_TRIES = 180
GRID_HEIGHT_RANGE = (16, 28)
GRID_WIDTH_RANGE = (16, 28)
OBJECT_COUNT_RANGE = (4, 6)
EDGE_BIAS_PROB = 0.35
INPUT_PADDING = 1
OUTPUT_PADDING = 1


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append({"color": color, "coords": coords})
    return comps


def _square_object(grid: np.ndarray, comp: dict) -> dict:
    coords = comp["coords"]
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    size = r1 - r0 + 1

    assert size == c1 - c0 + 1, "every foreground component must be square"
    assert 1 <= size <= 4, "square side length must be in [1, 4]"
    assert len(coords) == size * size, "square component area must match its bbox"
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert bool((patch == int(comp["color"])).all()), "square bbox must be solid"

    return {
        "color": int(comp["color"]),
        "r0": int(r0),
        "c0": int(c0),
        "size": int(size),
    }


def _layer_widths(size: int) -> list[int]:
    start = 1 if size % 2 == 1 else 2
    return list(range(start, size + 1, 2))


def _nearest_side(shape: tuple[int, int], r0: int, c0: int, size: int) -> str:
    side = _nearest_side_or_none(shape, r0, c0, size)
    assert side is not None, "each square must have a unique nearest edge"
    return side


def _nearest_side_or_none(shape: tuple[int, int], r0: int, c0: int, size: int) -> str | None:
    h, w = shape
    center_r = r0 + (size - 1) / 2.0
    center_c = c0 + (size - 1) / 2.0
    distances = {
        "up": float(center_r),
        "down": float(h - 1 - center_r),
        "left": float(center_c),
        "right": float(w - 1 - center_c),
    }
    best = min(distances.values())
    winners = [name for name, value in distances.items() if abs(value - best) < 1e-9]
    if len(winners) != 1:
        return None
    return winners[0]


def _wedge_cells(r0: int, c0: int, size: int, direction: str) -> set[tuple[int, int]]:
    center2_r = 2 * r0 + size - 1
    center2_c = 2 * c0 + size - 1
    widths = _layer_widths(size)
    max_offset = len(widths) - 1
    cells: set[tuple[int, int]] = set()

    for depth, width in enumerate(widths):
        offset = max_offset - depth
        span_start = (center2_c - (width - 1)) // 2
        span_top = (center2_r - (width - 1)) // 2

        if direction == "up":
            row2 = center2_r - 2 * offset
            row = (row2 + 1) // 2
            for col in range(span_start, span_start + width):
                cells.add((int(row), int(col)))
        elif direction == "down":
            row2 = center2_r + 2 * offset
            row = row2 // 2
            for col in range(span_start, span_start + width):
                cells.add((int(row), int(col)))
        elif direction == "left":
            col2 = center2_c - 2 * offset
            col = (col2 + 1) // 2
            for row in range(span_top, span_top + width):
                cells.add((int(row), int(col)))
        else:
            assert direction == "right"
            col2 = center2_c + 2 * offset
            col = col2 // 2
            for row in range(span_top, span_top + width):
                cells.add((int(row), int(col)))

    return cells


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    h, w = grid.shape
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each solid square with a triangular wedge pointing toward its unique nearest grid edge.

    Input structure:
    - The background is the dominant color.
    - The foreground consists of several disjoint monochrome solid squares.
    - Each square has side length from 1 to 4 pixels.
    - For every square, the center of its bounding box has exactly one nearest side among the four grid edges.

    Rule:
    - Treat each foreground component as one square object and determine which edge of the grid is uniquely closest to its center.
    - Erase the original square.
    - Redraw that object in the same color as a wedge pointing toward the nearest edge: its base is perpendicular to the pointing direction, its widest row or column has width equal to the square side, and narrower rows or columns taper toward the tip.
    - If any part of the wedge would fall outside the grid, clip it at the boundary.

    Key insight:
    - The square's position encodes direction via the nearest border, while the square's own side length encodes the wedge size, so direction and shape scale must be inferred independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    bg = _mode_color(g)
    comps = _find_components(g, bg)
    assert comps, "expected at least one square"

    objects = [_square_object(g, comp) for comp in comps]
    objects.sort(key=lambda obj: (obj["r0"], obj["c0"], obj["color"]))

    out = np.full_like(g, bg)
    for obj in objects:
        direction = _nearest_side(g.shape, obj["r0"], obj["c0"], obj["size"])
        cells = _wedge_cells(obj["r0"], obj["c0"], obj["size"], direction)
        _paint_cells(out, cells, obj["color"])
    return out


def _square_cells(r0: int, c0: int, size: int) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r in range(r0, r0 + size)
        for c in range(c0, c0 + size)
    }


def _expanded_cells(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((int(rr), int(cc)))
    return out


def _balanced_directions(count: int, rng: random.Random) -> list[str]:
    directions = list(DIRECTIONS)
    while len(directions) < count:
        directions.append(rng.choice(DIRECTIONS))
    rng.shuffle(directions)
    return directions


def _balanced_sizes(count: int, rng: random.Random) -> list[int]:
    sizes = list(SIZES)
    while len(sizes) < count:
        sizes.append(rng.choice(SIZES))
    rng.shuffle(sizes)
    return sizes


def _directional_position(shape: tuple[int, int], size: int, direction: str, rng: random.Random) -> tuple[int, int]:
    h, w = shape
    max_r = h - size
    max_c = w - size
    edge_band_h = max(0, h // 4)
    edge_band_w = max(0, w // 4)

    if direction == "up":
        if rng.random() < EDGE_BIAS_PROB:
            r0 = rng.randint(0, min(max_r, 1))
        else:
            r0 = rng.randint(0, min(max_r, edge_band_h))
        c0 = rng.randint(0, max_c)
    elif direction == "down":
        if rng.random() < EDGE_BIAS_PROB:
            r0 = rng.randint(max(0, max_r - 1), max_r)
        else:
            r0 = rng.randint(max(0, max_r - edge_band_h), max_r)
        c0 = rng.randint(0, max_c)
    elif direction == "left":
        r0 = rng.randint(0, max_r)
        if rng.random() < EDGE_BIAS_PROB:
            c0 = rng.randint(0, min(max_c, 1))
        else:
            c0 = rng.randint(0, min(max_c, edge_band_w))
    else:
        assert direction == "right"
        r0 = rng.randint(0, max_r)
        if rng.random() < EDGE_BIAS_PROB:
            c0 = rng.randint(max(0, max_c - 1), max_c)
        else:
            c0 = rng.randint(max(0, max_c - edge_band_w), max_c)

    return int(r0), int(c0)


def _place_objects(shape: tuple[int, int], items: list[dict], rng: random.Random) -> list[dict] | None:
    used_input: set[tuple[int, int]] = set()
    used_output: set[tuple[int, int]] = set()
    placed: list[dict] = []

    for item in sorted(items, key=lambda obj: (-obj["size"], obj["direction"], obj["color"])):
        success = False
        for _ in range(PLACEMENT_TRIES):
            r0, c0 = _directional_position(shape, item["size"], item["direction"], rng)
            if _nearest_side_or_none(shape, r0, c0, item["size"]) != item["direction"]:
                continue

            square_cells = _square_cells(r0, c0, item["size"])
            wedge_cells = {
                cell
                for cell in _wedge_cells(r0, c0, item["size"], item["direction"])
                if 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]
            }
            if not wedge_cells:
                continue

            if _expanded_cells(square_cells, shape, INPUT_PADDING) & used_input:
                continue
            if _expanded_cells(wedge_cells, shape, OUTPUT_PADDING) & used_output:
                continue

            placed_obj = {
                "color": int(item["color"]),
                "size": int(item["size"]),
                "direction": item["direction"],
                "r0": int(r0),
                "c0": int(c0),
            }
            placed.append(placed_obj)
            used_input |= _expanded_cells(square_cells, shape, INPUT_PADDING)
            used_output |= _expanded_cells(wedge_cells, shape, OUTPUT_PADDING)
            success = True
            break

        if not success:
            return None

    return placed


def _build_input(shape: tuple[int, int], bg: int, placed: list[dict]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for obj in placed:
        grid[obj["r0"] : obj["r0"] + obj["size"], obj["c0"] : obj["c0"] + obj["size"]] = int(obj["color"])
    return grid


def _build_expected(shape: tuple[int, int], bg: int, placed: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for obj in placed:
        cells = _wedge_cells(obj["r0"], obj["c0"], obj["size"], obj["direction"])
        _paint_cells(out, cells, obj["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1])
    w = rng.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])
    count = rng.randint(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1])

    bg = rng.randrange(10)
    colors = rng.sample([c for c in range(10) if c != bg], count)
    directions = _balanced_directions(count, rng)
    sizes = _balanced_sizes(count, rng)
    items = [
        {"color": int(color), "direction": direction, "size": int(size)}
        for color, direction, size in zip(colors, directions, sizes)
    ]

    placed = _place_objects((h, w), items, rng)
    if placed is None:
        return None

    grid = _build_input((h, w), bg, placed)
    if _mode_color(grid) != bg:
        return None

    expected = _build_expected((h, w), bg, placed)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
