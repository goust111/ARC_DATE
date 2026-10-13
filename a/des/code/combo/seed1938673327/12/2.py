from __future__ import annotations

import numpy as np


Coord = tuple[int, int]


# --------------- solver ---------------


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _foreground_positions(grid: np.ndarray, bg: int) -> dict[int, list[Coord]]:
    positions: dict[int, list[Coord]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        positions[color] = coords
    return positions


def _bridge_cells(a: Coord, b: Coord) -> list[Coord]:
    (r0, c0), (r1, c1) = sorted((tuple(map(int, a)), tuple(map(int, b))))

    if r0 == r1:
        return [(r0, c) for c in range(c0 + 1, c1)]
    if c0 == c1:
        return [(r, c0) for r in range(r0 + 1, r1)]

    step_c = 1 if c1 > c0 else -1
    step_r = 1 if r1 > r0 else -1
    cells = [(r0, c) for c in range(c0 + step_c, c1 + step_c, step_c)]
    cells.extend((r, c1) for r in range(r0 + step_r, r1, step_r))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the singleton marker color as the bridge deck color, remove that marker,
    and connect every paired color either horizontally, vertically, or with a
    horizontal-then-vertical L shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    positions = _foreground_positions(g, bg)
    assert positions, "expected at least one non-background color"

    counts = {color: len(coords) for color, coords in positions.items()}
    bridge_colors = [color for color, count in counts.items() if count == 1]
    assert len(bridge_colors) == 1, "expected exactly one singleton bridge-color marker"
    assert all(count in (1, 2) for count in counts.values()), "all non-bg colors must be singleton or paired"

    bridge_color = int(bridge_colors[0])
    paired = [(color, coords) for color, coords in positions.items() if color != bridge_color]
    assert paired, "expected at least one bridge-pier pair"
    assert all(len(coords) == 2 for _color, coords in paired), "all non-marker colors must appear exactly twice"

    out = g.copy()
    marker_r, marker_c = positions[bridge_color][0]
    out[marker_r, marker_c] = bg

    for _color, coords in paired:
        a, b = coords
        for r, c in _bridge_cells(a, b):
            out[r, c] = bridge_color

    return out


# --------------- generator ---------------


PALETTE = tuple(range(10))
PAIR_TYPES = ("h", "v", "l")

PAIR_COUNT_RANGE = (3, 6)
GRID_HEIGHT_RANGE = (14, 22)
GRID_WIDTH_RANGE = (14, 22)
BOX_SPAN_RANGE = (3, 7)
PLACEMENT_MARGIN = 1
PAIR_PLACE_TRIES = 200
GENERATION_TRIES = 400


def _sample_pair_types(rng: np.random.Generator, pair_count: int) -> list[str]:
    pair_types = list(PAIR_TYPES)
    pair_types.extend(PAIR_TYPES[int(rng.integers(0, len(PAIR_TYPES)))] for _ in range(pair_count - len(PAIR_TYPES)))
    rng.shuffle(pair_types)
    return pair_types


def _sample_colors(rng: np.random.Generator, pair_count: int) -> tuple[int, int, list[int]]:
    sampled = [int(x) for x in rng.choice(PALETTE, size=pair_count + 2, replace=False)]
    bg = sampled[0]
    bridge_color = sampled[1]
    pair_colors = sampled[2:]
    return bg, bridge_color, pair_colors


def _sample_box(rng: np.random.Generator, pair_type: str, color: int) -> dict[str, int]:
    if pair_type == "h":
        return {"type": pair_type, "color": color, "height": 1, "width": int(rng.integers(*BOX_SPAN_RANGE))}
    if pair_type == "v":
        return {"type": pair_type, "color": color, "height": int(rng.integers(*BOX_SPAN_RANGE)), "width": 1}
    return {
        "type": pair_type,
        "color": color,
        "height": int(rng.integers(*BOX_SPAN_RANGE)),
        "width": int(rng.integers(*BOX_SPAN_RANGE)),
        "flip": int(rng.integers(0, 2)),
    }


def _spec_area(spec: dict[str, int]) -> int:
    return (int(spec["height"]) + 2 * PLACEMENT_MARGIN) * (int(spec["width"]) + 2 * PLACEMENT_MARGIN)


def _endpoints_for_box(spec: dict[str, int], top: int, left: int) -> tuple[Coord, Coord]:
    height = int(spec["height"])
    width = int(spec["width"])
    pair_type = spec["type"]

    if pair_type == "h":
        return (top, left), (top, left + width - 1)
    if pair_type == "v":
        return (top, left), (top + height - 1, left)
    if int(spec["flip"]) == 0:
        return (top, left), (top + height - 1, left + width - 1)
    return (top, left + width - 1), (top + height - 1, left)


def _try_place_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    used_rows: set[int],
    used_cols: set[int],
    spec: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, object] | None:
    height, width = grid.shape
    box_h = int(spec["height"])
    box_w = int(spec["width"])
    row_hi = height - box_h - PLACEMENT_MARGIN
    col_hi = width - box_w - PLACEMENT_MARGIN
    if row_hi <= PLACEMENT_MARGIN or col_hi <= PLACEMENT_MARGIN:
        return None

    for _ in range(PAIR_PLACE_TRIES):
        top = int(rng.integers(PLACEMENT_MARGIN, row_hi + 1))
        left = int(rng.integers(PLACEMENT_MARGIN, col_hi + 1))
        reserve = blocked[
            top - PLACEMENT_MARGIN : top + box_h + PLACEMENT_MARGIN,
            left - PLACEMENT_MARGIN : left + box_w + PLACEMENT_MARGIN,
        ]
        if bool(np.any(reserve)):
            continue

        a, b = _endpoints_for_box(spec, top, left)
        endpoint_rows = {a[0], b[0]}
        endpoint_cols = {a[1], b[1]}
        if endpoint_rows & used_rows or endpoint_cols & used_cols:
            continue

        bridge_cells = _bridge_cells(a, b)
        if not bridge_cells:
            continue

        blocked[
            top - PLACEMENT_MARGIN : top + box_h + PLACEMENT_MARGIN,
            left - PLACEMENT_MARGIN : left + box_w + PLACEMENT_MARGIN,
        ] = True
        grid[a] = int(spec["color"])
        grid[b] = int(spec["color"])
        used_rows.update(endpoint_rows)
        used_cols.update(endpoint_cols)
        return {"color": int(spec["color"]), "a": a, "b": b}

    return None


def _place_marker(
    grid: np.ndarray,
    blocked: np.ndarray,
    used_rows: set[int],
    used_cols: set[int],
    bridge_color: int,
    bg: int,
    rng: np.random.Generator,
) -> Coord | None:
    preferred = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if grid[r, c] == bg and not blocked[r, c] and r not in used_rows and c not in used_cols
    ]
    candidates = preferred or [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if grid[r, c] == bg and not blocked[r, c]
    ]
    if not candidates:
        return None

    marker = candidates[int(rng.integers(0, len(candidates)))]
    grid[marker] = bridge_color
    return marker


def _expected_output(grid: np.ndarray, bg: int, bridge_color: int, marker: Coord, placements: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    out[marker] = bg
    for placement in placements:
        for r, c in _bridge_cells(placement["a"], placement["b"]):
            out[r, c] = bridge_color
    return out


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(*PAIR_COUNT_RANGE))
    bg, bridge_color, pair_colors = _sample_colors(rng, pair_count)
    specs = [_sample_box(rng, pair_type, color) for pair_type, color in zip(_sample_pair_types(rng, pair_count), pair_colors)]
    specs.sort(key=_spec_area, reverse=True)

    grid_h = int(rng.integers(*GRID_HEIGHT_RANGE))
    grid_w = int(rng.integers(*GRID_WIDTH_RANGE))
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    placements: list[dict[str, object]] = []
    for spec in specs:
        placed = _try_place_pair(grid, blocked, used_rows, used_cols, spec, rng)
        if placed is None:
            return None
        placements.append(placed)

    marker = _place_marker(grid, blocked, used_rows, used_cols, bridge_color, bg, rng)
    if marker is None:
        return None

    expected = _expected_output(grid, bg, bridge_color, marker, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_TRIES):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-pair ARC input")
