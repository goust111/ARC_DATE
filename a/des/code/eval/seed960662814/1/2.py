from collections import defaultdict, deque
import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
PAIR_COUNT_RANGE = (1, 4)
GRID_SIDE_MAX = 22
SHAPE_HEIGHT_RANGE = (1, 4)
SHAPE_WIDTH_RANGE = (1, 4)
SHAPE_AREA_MAX = 6
PLACEMENT_MARGIN = 1
GENERATION_RETRIES = 500
SHAPE_RETRIES = 120


def _neighbors(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    height = 1 + max(r for r, _ in shape)
    width = 1 + max(c for _, c in shape)
    return height, width


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors(rr, cc, height, width):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            shape = _normalize(coords)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "shape": shape,
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Swap colors between every pair of translation-identical monochrome fragments."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    assert components
    assert len(components) % 2 == 0

    groups: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for comp in components:
        groups[comp["shape"]].append(comp)

    out = g.copy()
    for same_shape in groups.values():
        assert len(same_shape) == 2
        first, second = same_shape
        color_a = int(first["color"])
        color_b = int(second["color"])
        assert color_a != color_b

        for r, c in first["coords"]:
            out[r, c] = color_b
        for r, c in second["coords"]:
            out[r, c] = color_a

    return out


def _grow_shape(height: int, width: int, area: int) -> tuple[tuple[int, int], ...] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set(_neighbors(start[0], start[1], height, width))

    while len(cells) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nbr in _neighbors(cell[0], cell[1], height, width):
            if nbr not in cells:
                frontier.add(nbr)

    return _normalize(cells)


def _random_shape(used: set[tuple[tuple[int, int], ...]]) -> tuple[tuple[int, int], ...] | None:
    for _ in range(SHAPE_RETRIES):
        height = random.randint(*SHAPE_HEIGHT_RANGE)
        width = random.randint(*SHAPE_WIDTH_RANGE)
        area_hi = min(SHAPE_AREA_MAX, height * width)
        area_lo = 1 if area_hi == 1 else 2
        area = random.randint(area_lo, area_hi)
        shape = _grow_shape(height, width, area)
        if shape is None or shape in used:
            continue
        return shape
    return None


def _choose_grid_size(pair_count: int) -> tuple[int, int]:
    min_side = 8 + 2 * pair_count
    rows = random.randint(min_side, GRID_SIDE_MAX)
    cols = random.randint(min_side, GRID_SIDE_MAX)
    return rows, cols


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    row0 = max(0, r0 - PLACEMENT_MARGIN)
    col0 = max(0, c0 - PLACEMENT_MARGIN)
    row1 = min(blocked.shape[0], r0 + height + PLACEMENT_MARGIN)
    col1 = min(blocked.shape[1], c0 + width + PLACEMENT_MARGIN)
    blocked[row0:row1, col0:col1] = True


def _place_components(rows: int, cols: int, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.full((rows, cols), BACKGROUND, dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)

    pending = []
    for pair_id, pair in enumerate(pairs):
        shape = pair["shape"]
        height, width = _shape_size(shape)
        for color in pair["colors"]:
            pending.append(
                {
                    "pair_id": pair_id,
                    "shape": shape,
                    "color": int(color),
                    "height": height,
                    "width": width,
                    "area": len(shape),
                    "tie": random.random(),
                }
            )

    pending.sort(key=lambda item: (-item["area"], -item["height"] * item["width"], item["tie"]))
    placements: dict[int, list[dict]] = defaultdict(list)

    for spec in pending:
        height = spec["height"]
        width = spec["width"]
        candidates: list[tuple[int, int]] = []

        for r0 in range(rows - height + 1):
            for c0 in range(cols - width + 1):
                if blocked[r0 : r0 + height, c0 : c0 + width].any():
                    continue
                candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        coords = [(r0 + dr, c0 + dc) for dr, dc in spec["shape"]]
        for r, c in coords:
            grid[r, c] = int(spec["color"])
        _mark_blocked(blocked, r0, c0, height, width)
        placements[spec["pair_id"]].append({"coords": coords, "color": int(spec["color"])})

    expected = grid.copy()
    for pair_items in placements.values():
        assert len(pair_items) == 2
        first, second = pair_items
        for r, c in first["coords"]:
            expected[r, c] = int(second["color"])
        for r, c in second["coords"]:
            expected[r, c] = int(first["color"])

    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    pairs: list[dict] = []

    for _ in range(pair_count):
        shape = _random_shape(used_shapes)
        if shape is None:
            return None
        used_shapes.add(shape)
        color_a, color_b = random.sample(PALETTE, 2)
        pairs.append({"shape": shape, "colors": (color_a, color_b)})

    rows, cols = _choose_grid_size(pair_count)
    placed = _place_components(rows, cols, pairs)
    if placed is None:
        return None

    grid, expected = placed
    try:
        actual = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid paired-fragment grid")
