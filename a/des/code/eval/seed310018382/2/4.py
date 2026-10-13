from __future__ import annotations

import random

import numpy as np


YELLOW = 4
ORANGE = 7
SOURCE_COLORS = (1, 2, 3, 5, 6, 8, 9)

GEN_ATTEMPTS = 400
GRID_H_RANGE = (12, 19)
GRID_W_RANGE = (12, 19)
LINE_MARGIN = 4
LINE_GAP = 3

DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}

# 8-connected source shapes with one unique 8-neighbor endpoint whose outward
# direction is cardinal. The generator uses only these, while main infers the
# endpoint from the component itself.
TEMPLATES = {
    "right": [
        {"cells": ((0, 0), (0, 1), (0, 2), (1, 0)), "endpoint": (0, 2)},
        {"cells": ((0, 0), (1, 0), (1, 1), (1, 2)), "endpoint": (1, 2)},
    ],
    "left": [
        {"cells": ((0, 0), (0, 1), (0, 2), (1, 2)), "endpoint": (0, 0)},
        {"cells": ((0, 2), (1, 0), (1, 1), (1, 2)), "endpoint": (1, 0)},
    ],
    "down": [
        {"cells": ((0, 0), (0, 1), (1, 0), (2, 0)), "endpoint": (2, 0)},
        {"cells": ((0, 0), (0, 1), (1, 1), (2, 1)), "endpoint": (2, 1)},
    ],
    "up": [
        {"cells": ((0, 0), (1, 0), (2, 0), (2, 1)), "endpoint": (0, 0)},
        {"cells": ((0, 1), (1, 1), (2, 0), (2, 1)), "endpoint": (0, 1)},
    ],
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = grid.shape

    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue

        stack = [(r0, c0)]
        seen[r0, c0] = True
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        comps.append(comp)

    return comps


def _endpoint_and_direction(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    coord_set = set(coords)
    endpoints: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in coords:
        neighbors = [(nr, nc) for nr, nc in _neighbors8(r, c) if (nr, nc) in coord_set]
        if len(neighbors) == 1:
            endpoints.append(((r, c), neighbors[0]))

    assert len(endpoints) == 1, "each source object must have exactly one 8-neighbor endpoint"
    endpoint, anchor = endpoints[0]
    dr = int(endpoint[0] - anchor[0])
    dc = int(endpoint[1] - anchor[1])
    assert abs(dr) + abs(dc) == 1, "endpoint must protrude in a cardinal direction"
    return endpoint, (dr, dc)


def _trace_ray(
    shape: tuple[int, int],
    endpoint: tuple[int, int],
    direction: tuple[int, int],
    absorbers: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r = int(endpoint[0] + dr)
    c = int(endpoint[1] + dc)
    path: list[tuple[int, int]] = []

    while 0 <= r < h and 0 <= c < w:
        if (r, c) in absorbers:
            break
        path.append((r, c))
        r += dr
        c += dc

    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow one ray from each small colored source object.

    - Multi-cell same-color 8-components are the source objects.
    - Singleton same-color components are absorber points.
    - A source's unique 8-neighbor endpoint determines the ray direction.
    - Rays stop at the boundary or at a same-color absorber, and only repaint
      original background cells.
    - Cells visited by two rays turn yellow; cells visited by three or more
      rays turn orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    absorbers_by_color: dict[int, set[tuple[int, int]]] = {}
    sources: list[dict[str, object]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _color_components(g, color):
            if len(comp) == 1:
                absorbers_by_color.setdefault(int(color), set()).add(comp[0])
                continue

            assert 2 <= len(comp) <= 4, "source objects must have 2-4 cells"
            endpoint, direction = _endpoint_and_direction(comp)
            sources.append(
                {
                    "color": int(color),
                    "coords": set(comp),
                    "endpoint": endpoint,
                    "direction": direction,
                }
            )

    assert sources, "expected at least one multi-cell source object"

    visits: dict[tuple[int, int], set[int]] = {}
    for idx, source in enumerate(sources):
        path = _trace_ray(
            g.shape,
            source["endpoint"],  # type: ignore[arg-type]
            source["direction"],  # type: ignore[arg-type]
            absorbers_by_color.get(int(source["color"]), set()),
        )
        for cell in path:
            visits.setdefault(cell, set()).add(int(idx))

    out = g.copy()
    for (r, c), ray_ids in visits.items():
        if int(g[r, c]) != bg:
            continue
        if len(ray_ids) == 1:
            ray_id = next(iter(ray_ids))
            out[r, c] = int(sources[ray_id]["color"])
        elif len(ray_ids) == 2:
            out[r, c] = YELLOW
        else:
            out[r, c] = ORANGE

    return out


def _pick_spaced_values(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(int(low), int(high) + 1))
    random.shuffle(values)
    chosen: list[int] = []
    for value in values:
        if all(abs(value - other) >= int(min_gap) for other in chosen):
            chosen.append(int(value))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _place_source(
    grid: np.ndarray,
    occupied: np.ndarray,
    direction: str,
    line: int,
    color: int,
) -> dict[str, object] | None:
    h, w = grid.shape
    template = random.choice(TEMPLATES[direction])
    endpoint_r, endpoint_c = template["endpoint"]

    if direction == "right":
        top = int(line - endpoint_r)
        left = 0
        orientation = "H"
    elif direction == "left":
        top = int(line - endpoint_r)
        left = int(w - 3)
        orientation = "H"
    elif direction == "down":
        top = 0
        left = int(line - endpoint_c)
        orientation = "V"
    else:
        assert direction == "up"
        top = int(h - 3)
        left = int(line - endpoint_c)
        orientation = "V"

    coords: list[tuple[int, int]] = []
    for dr, dc in template["cells"]:
        r = int(top + dr)
        c = int(left + dc)
        if not (0 <= r < h and 0 <= c < w):
            return None
        if occupied[r, c]:
            return None
        coords.append((r, c))

    for r, c in coords:
        occupied[r, c] = True
        grid[r, c] = int(color)

    endpoint = (int(top + endpoint_r), int(left + endpoint_c))
    return {
        "color": int(color),
        "coords": coords,
        "endpoint": endpoint,
        "direction_name": direction,
        "direction": DIRS[direction],
        "orientation": orientation,
        "line": int(line),
    }


def _absorber_candidates(
    source: dict[str, object],
    required_lines: list[int],
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    endpoint_r, endpoint_c = source["endpoint"]  # type: ignore[misc]
    dr, dc = source["direction"]  # type: ignore[misc]

    if source["orientation"] == "H":
        assert required_lines
        start_c = (max(required_lines) + 1) if dc == 1 else (min(required_lines) - 1)
        r = int(endpoint_r)
        c = int(start_c)
    else:
        assert required_lines
        start_r = (max(required_lines) + 1) if dr == 1 else (min(required_lines) - 1)
        r = int(start_r)
        c = int(endpoint_c)

    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if occupied[r, c]:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _make_specs(mode: str, rows: list[int], cols: list[int]) -> list[tuple[str, int]]:
    if mode == "single_cross":
        return [
            (random.choice(("right", "left")), rows[0]),
            (random.choice(("down", "up")), cols[0]),
        ]
    if mode == "yellow_grid":
        specs: list[tuple[str, int]] = []
        for row in rows:
            specs.append((random.choice(("right", "left")), row))
        for col in cols:
            specs.append((random.choice(("down", "up")), col))
        return specs
    if mode == "orange_row":
        specs = [("right", rows[0]), ("left", rows[0])]
        for col in cols:
            specs.append((random.choice(("down", "up")), col))
        return specs
    assert mode == "orange_col"
    specs = [("down", cols[0]), ("up", cols[0])]
    for row in rows:
        specs.append((random.choice(("right", "left")), row))
    return specs


def _mode_layout(mode: str, h: int, w: int) -> tuple[list[int], list[int]] | None:
    row_low = LINE_MARGIN
    row_high = h - LINE_MARGIN - 1
    col_low = LINE_MARGIN
    col_high = w - LINE_MARGIN - 1

    if row_high < row_low or col_high < col_low:
        return None

    if mode == "single_cross":
        rows = _pick_spaced_values(row_low, row_high, 1, LINE_GAP)
        cols = _pick_spaced_values(col_low, col_high, 1, LINE_GAP)
    elif mode == "yellow_grid":
        h_count, v_count = random.choice(((2, 1), (1, 2), (2, 2)))
        rows = _pick_spaced_values(row_low, row_high, h_count, LINE_GAP)
        cols = _pick_spaced_values(col_low, col_high, v_count, LINE_GAP)
    elif mode == "orange_row":
        rows = _pick_spaced_values(row_low, row_high, 1, LINE_GAP)
        cols = _pick_spaced_values(col_low, col_high, random.choice((1, 2)), LINE_GAP)
    else:
        assert mode == "orange_col"
        rows = _pick_spaced_values(row_low, row_high, random.choice((1, 2)), LINE_GAP)
        cols = _pick_spaced_values(col_low, col_high, 1, LINE_GAP)

    if rows is None or cols is None:
        return None
    return rows, cols


def _decorate_absorbers(
    grid: np.ndarray,
    occupied: np.ndarray,
    sources: list[dict[str, object]],
) -> bool:
    h_rows = [int(source["line"]) for source in sources if source["orientation"] == "H"]
    v_cols = [int(source["line"]) for source in sources if source["orientation"] == "V"]
    candidate_map: dict[int, list[tuple[int, int]]] = {}

    for idx, source in enumerate(sources):
        required = v_cols if source["orientation"] == "H" else h_rows
        candidates = _absorber_candidates(source, required, occupied)
        if candidates:
            candidate_map[idx] = candidates

    if not candidate_map:
        return True

    chosen = [idx for idx in candidate_map if random.random() < 0.5]
    keys = list(candidate_map)
    if len(keys) >= 2:
        if not chosen:
            chosen = [random.choice(keys)]
        elif len(chosen) == len(keys):
            chosen.remove(random.choice(chosen))

    for idx in chosen:
        r, c = random.choice(candidate_map[idx])
        color = int(sources[idx]["color"])
        if occupied[r, c]:
            return False
        occupied[r, c] = True
        grid[r, c] = color

    return True


def _validate_generated(grid: np.ndarray, mode: str) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False

    has_yellow = bool(np.any(out == YELLOW))
    has_orange = bool(np.any(out == ORANGE))

    if mode in {"single_cross", "yellow_grid"}:
        return has_yellow and not has_orange
    return has_orange


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    mode = random.choice(("single_cross", "yellow_grid", "orange_row", "orange_col"))
    layout = _mode_layout(mode, h, w)
    if layout is None:
        return None
    rows, cols = layout

    specs = _make_specs(mode, rows, cols)
    n_sources = len(specs)
    if n_sources > len(SOURCE_COLORS):
        return None

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = random.sample(list(SOURCE_COLORS), n_sources)
    random.shuffle(specs)

    sources: list[dict[str, object]] = []
    for (direction, line), color in zip(specs, colors):
        source = _place_source(grid, occupied, direction, int(line), int(color))
        if source is None:
            return None
        sources.append(source)

    if not _decorate_absorbers(grid, occupied, sources):
        return None
    if not _validate_generated(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-intersection task")
