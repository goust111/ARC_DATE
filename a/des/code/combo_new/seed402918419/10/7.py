from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 5
PALETTE = (0, 1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
TOWER_PATTERNS = {
    "up": (
        np.array([[0, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [0, 1, 1], [0, 1, 1]], dtype=int),
    ),
    "down": (
        np.array([[1, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 1], [0, 1, 1], [0, 1, 0]], dtype=int),
    ),
    "left": (
        np.array([[0, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[0, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=int),
    ),
    "right": (
        np.array([[1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[0, 0, 0], [1, 1, 1], [1, 1, 0]], dtype=int),
    ),
}

GRID_MIN = 20
GRID_MAX = 30
MIN_PATH_LEN = 9
LANE_GAP = 4
PRISM_GAP = 2
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values = grid.astype(int, copy=False).ravel()
    return int(Counter(values).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            result.append(comp)
    return result


def _classify_tower(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(coords) != 5:
        return None

    color_values = {int(grid[r, c]) for r, c in coords}
    if len(color_values) != 1:
        return None
    color = next(iter(color_values))

    cells = set(coords)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    for r0 in range(min(rows), max(rows)):
        for c0 in range(min(cols), max(cols)):
            square = {
                (r0, c0),
                (r0 + 1, c0),
                (r0, c0 + 1),
                (r0 + 1, c0 + 1),
            }
            if not square.issubset(cells):
                continue

            extra = list(cells - square)
            if len(extra) != 1:
                continue
            er, ec = extra[0]
            if er < r0 and c0 <= ec <= c0 + 1:
                direction = "up"
            elif er > r0 + 1 and c0 <= ec <= c0 + 1:
                direction = "down"
            elif ec < c0 and r0 <= er <= r0 + 1:
                direction = "left"
            elif ec > c0 + 1 and r0 <= er <= r0 + 1:
                direction = "right"
            else:
                continue

            return {
                "type": "tower",
                "color": color,
                "coords": tuple(sorted(coords)),
                "antenna": (er, ec),
                "direction": direction,
            }
    return None


def _classify_prism(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(coords) != 3:
        return None

    coords_sorted = sorted(coords)
    rows = sorted({r for r, _ in coords_sorted})
    cols = sorted({c for _, c in coords_sorted})
    if len(rows) == 1 and cols == list(range(cols[0], cols[0] + 3)):
        ordered = sorted(coords_sorted, key=lambda rc: rc[1])
        colors = [int(grid[r, c]) for r, c in ordered]
        if colors[0] == colors[2] and colors[0] != colors[1]:
            return {
                "type": "prism",
                "orientation": "h",
                "incident": colors[0],
                "emitted": colors[1],
                "center": ordered[1],
                "coords": tuple(ordered),
            }
    if len(cols) == 1 and rows == list(range(rows[0], rows[0] + 3)):
        ordered = sorted(coords_sorted, key=lambda rc: rc[0])
        colors = [int(grid[r, c]) for r, c in ordered]
        if colors[0] == colors[2] and colors[0] != colors[1]:
            return {
                "type": "prism",
                "orientation": "v",
                "incident": colors[0],
                "emitted": colors[1],
                "center": ordered[1],
                "coords": tuple(ordered),
            }
    return None


def _parse_objects(grid: np.ndarray) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    towers: list[dict[str, object]] = []
    prisms: list[dict[str, object]] = []
    for comp in _components(g != bg):
        tower = _classify_tower(g, comp)
        if tower is not None:
            towers.append(tower)
            continue
        prism = _classify_prism(g, comp)
        if prism is not None:
            prisms.append(prism)
            continue
        raise AssertionError(f"unrecognized non-background component: {sorted(comp)}")

    assert towers, "expected at least one tower"
    assert prisms, "expected at least one prism"
    return bg, towers, prisms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each signal tower ray to the grid edge and let matching prisms recolor the ray after the crossing.

    Input structure:
    - A dominant grey-like background surrounds several monochromatic signal towers and several 3-cell prisms.
    - Each tower is a connected 5-pixel object made from a solid 2x2 body plus a 1-pixel antenna extending on one side.
    - Each prism is a straight 1x3 or 3x1 strip whose two end pixels share an incident color and whose middle pixel has a different emitted color.

    Rule:
    - For every tower, start from the cell just beyond the antenna and march straight in the antenna direction until the boundary.
    - The ray starts in the tower color; background cells on the path are painted with the current ray color, while tower and prism cells themselves stay unchanged.
    - When the ray crosses the center of a perpendicular prism and the current ray color matches that prism's end color, the ray color changes to the prism's middle color for all later cells on that path.

    Key insight:
    - The puzzle is a ray-casting task with state: the tower gives the direction, but the visible ray color after each prism depends on the current carried color rather than on a one-time lookup.
    """
    g = np.asarray(grid, dtype=int)
    bg, towers, prisms = _parse_objects(g)
    occupied = g != bg
    prism_by_center = {tuple(prism["center"]): prism for prism in prisms}

    out = g.copy()
    for tower in sorted(towers, key=lambda item: (*item["antenna"], str(item["direction"]))):
        color = int(tower["color"])
        dr, dc = DIRS[str(tower["direction"])]
        r, c = tower["antenna"]
        r += dr
        c += dc
        while 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            if not occupied[r, c]:
                out[r, c] = color

            prism = prism_by_center.get((r, c))
            if prism is not None:
                orientation = str(prism["orientation"])
                if ((dr != 0 and orientation == "h") or (dc != 0 and orientation == "v")) and color == int(prism["incident"]):
                    color = int(prism["emitted"])

            r += dr
            c += dc

    return out


def _pick_color(exclude: set[int]) -> int:
    choices = [color for color in PALETTE if color not in exclude]
    assert choices
    return int(random.choice(choices))


def _sample_spaced(values: list[int], count: int, min_gap: int, attempts: int = 200) -> list[int] | None:
    if count == 0:
        return []
    for _ in range(attempts):
        chosen: list[int] = []
        for value in random.sample(values, len(values)):
            if all(abs(value - prev) >= min_gap for prev in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _sample_prism_indices(path_len: int, count: int) -> list[int] | None:
    candidates = list(range(1, path_len - 1))
    return _sample_spaced(candidates, count, PRISM_GAP)


def _scenario_pairs(start_color: int, scenario: str) -> list[tuple[int, int]]:
    if scenario == "plain":
        return []
    if scenario == "match":
        return [(start_color, _pick_color({start_color, BG}))]
    if scenario == "mismatch":
        incident = _pick_color({start_color, BG})
        return [(incident, _pick_color({incident, BG}))]

    patterns = (
        (True, True),
        (False, True),
        (True, False),
        (True, True, False),
        (False, True, True),
    )
    current = start_color
    pairs: list[tuple[int, int]] = []
    for should_match in random.choice(patterns):
        if should_match:
            emitted = _pick_color({current, BG})
            pairs.append((current, emitted))
            current = emitted
        else:
            incident = _pick_color({current, BG})
            emitted = _pick_color({incident, BG})
            pairs.append((incident, emitted))
    return pairs


def _stamp_tower(grid: np.ndarray, direction: str, antenna: tuple[int, int], color: int) -> None:
    pattern = random.choice(TOWER_PATTERNS[direction])
    ar, ac = antenna
    if direction == "up":
        r0, c0 = ar, ac - 1
    elif direction == "down":
        r0, c0 = ar - 2, ac - 1
    elif direction == "left":
        r0, c0 = ar - 1, ac
    else:
        r0, c0 = ar - 1, ac - 2

    for dr in range(3):
        for dc in range(3):
            if pattern[dr, dc]:
                grid[r0 + dr, c0 + dc] = color


def _stamp_vertical_prism(grid: np.ndarray, row: int, col: int, incident: int, emitted: int) -> None:
    grid[row - 1, col] = incident
    grid[row, col] = emitted
    grid[row + 1, col] = incident


def _stamp_horizontal_prism(grid: np.ndarray, row: int, col: int, incident: int, emitted: int) -> None:
    grid[row, col - 1] = incident
    grid[row, col] = emitted
    grid[row, col + 1] = incident


def _build_horizontal_sample(height: int, width: int) -> np.ndarray | None:
    lane_values = list(range(1, height - 1))
    max_lanes = 1 + (lane_values[-1] - lane_values[0]) // LANE_GAP
    tower_count = min(random.randint(4, 5), max_lanes)
    rows = _sample_spaced(lane_values, tower_count, LANE_GAP)
    if rows is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    scenarios = ["plain", "match", "mismatch", "chain"]
    while len(scenarios) < tower_count:
        scenarios.append(random.choice(["plain", "match", "mismatch", "chain"]))
    random.shuffle(scenarios)

    directions = ["left", "right"]
    while len(directions) < tower_count:
        directions.append(random.choice(["left", "right"]))
    random.shuffle(directions)

    tower_colors = random.sample(PALETTE, tower_count)
    for ray_row, direction, color, scenario in zip(rows, directions, tower_colors, scenarios):
        if direction == "left":
            antenna_col = random.randint(MIN_PATH_LEN, width - 3)
            path_len = antenna_col
        else:
            antenna_col = random.randint(2, width - MIN_PATH_LEN - 1)
            path_len = width - antenna_col - 1

        pairs = _scenario_pairs(color, scenario)
        indices = _sample_prism_indices(path_len, len(pairs))
        if indices is None:
            return None

        _stamp_tower(grid, direction, (ray_row, antenna_col), color)
        for idx, (incident, emitted) in zip(indices, pairs):
            col = antenna_col - 1 - idx if direction == "left" else antenna_col + 1 + idx
            _stamp_vertical_prism(grid, ray_row, col, incident, emitted)

    return grid


def _build_vertical_sample(height: int, width: int) -> np.ndarray | None:
    lane_values = list(range(1, width - 1))
    max_lanes = 1 + (lane_values[-1] - lane_values[0]) // LANE_GAP
    tower_count = min(random.randint(4, 5), max_lanes)
    cols = _sample_spaced(lane_values, tower_count, LANE_GAP)
    if cols is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    scenarios = ["plain", "match", "mismatch", "chain"]
    while len(scenarios) < tower_count:
        scenarios.append(random.choice(["plain", "match", "mismatch", "chain"]))
    random.shuffle(scenarios)

    directions = ["up", "down"]
    while len(directions) < tower_count:
        directions.append(random.choice(["up", "down"]))
    random.shuffle(directions)

    tower_colors = random.sample(PALETTE, tower_count)
    for ray_col, direction, color, scenario in zip(cols, directions, tower_colors, scenarios):
        if direction == "up":
            antenna_row = random.randint(MIN_PATH_LEN, height - 3)
            path_len = antenna_row
        else:
            antenna_row = random.randint(2, height - MIN_PATH_LEN - 1)
            path_len = height - antenna_row - 1

        pairs = _scenario_pairs(color, scenario)
        indices = _sample_prism_indices(path_len, len(pairs))
        if indices is None:
            return None

        _stamp_tower(grid, direction, (antenna_row, ray_col), color)
        for idx, (incident, emitted) in zip(indices, pairs):
            row = antenna_row - 1 - idx if direction == "up" else antenna_row + 1 + idx
            _stamp_horizontal_prism(grid, row, ray_col, incident, emitted)

    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    mode = random.choice(["horizontal", "vertical"])

    grid = _build_horizontal_sample(height, width) if mode == "horizontal" else _build_vertical_sample(height, width)
    if grid is None:
        return None

    try:
        bg, towers, prisms = _parse_objects(grid)
    except AssertionError:
        return None

    if bg != BG or len(towers) < 4 or len(prisms) < 3:
        return None

    out = main(grid)
    painted = (grid == BG) & (out != BG)
    if int(painted.sum()) < 12:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower prism sample")
