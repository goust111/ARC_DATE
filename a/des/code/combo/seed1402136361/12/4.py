import random

import numpy as np


BLUE = 1
RED = 2
ALL_COLORS = tuple(range(10))
NEUTRAL_COLORS = tuple(color for color in ALL_COLORS if color not in {BLUE, RED})

GRID_MIN_SIZE = 16
GRID_MAX_SIZE = 24
GRID_BORDER = 1

PATH_COUNT_MIN = 1
PATH_COUNT_MAX = 3
PRIMARY_TURN_RANGE = (2, 5)
SECONDARY_TURN_RANGE = (0, 4)
SEGMENT_LENGTH_RANGE = (1, 4)

PATH_SHAPE_ATTEMPTS = 160
PLACEMENT_ATTEMPTS = 160
GRID_LAYOUT_ATTEMPTS = 40
GENERATION_ATTEMPTS = 400

DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIRS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))

    return comps


def _ordered_path(coords: np.ndarray) -> list[tuple[int, int]]:
    cells = [tuple(map(int, coord)) for coord in coords]
    cell_set = set(cells)
    assert len(cells) >= 2, "each path must have two endpoints"

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cell_set:
        nbrs = []
        for dr, dc in DIRS4:
            candidate = (r + dr, c + dc)
            if candidate in cell_set:
                nbrs.append(candidate)
        assert 1 <= len(nbrs) <= 2, "path must be one-pixel-wide and unbranched"
        neighbors[(r, c)] = nbrs

    endpoints = sorted(cell for cell, nbrs in neighbors.items() if len(nbrs) == 1)
    assert len(endpoints) == 2, "expected exactly two endpoints"

    start = endpoints[0]
    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [candidate for candidate in neighbors[cur] if candidate != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "simple path traversal must be unique"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt

    assert len(ordered) == len(cells), "path must not contain loops"
    return ordered


def _turn_sign(
    prev: tuple[int, int],
    cur: tuple[int, int],
    nxt: tuple[int, int],
) -> int:
    dr1 = cur[0] - prev[0]
    dc1 = cur[1] - prev[1]
    dr2 = nxt[0] - cur[0]
    dc2 = nxt[1] - cur[1]
    assert abs(dr1) + abs(dc1) == 1
    assert abs(dr2) + abs(dc2) == 1

    cross = dr1 * dc2 - dc1 * dr2
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each path's turn pixels by traversal-relative turn direction.

    Input structure
    - The background is the dominant color.
    - Each foreground connected component is a single-color, one-pixel-wide 4-connected path.
    - Every path has exactly two endpoints.

    Rule
    - Traverse each path from the lexicographically smaller endpoint.
    - At a 90-degree turn, recolor the turn pixel red for a left turn and blue for a right turn.
    - Keep endpoints and straight segments unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    components = _components(g != bg)
    assert components, "expected at least one path"

    out = g.copy()
    for coords in components:
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each path must be monochromatic"

        ordered = _ordered_path(coords)
        for idx in range(1, len(ordered) - 1):
            sign = _turn_sign(ordered[idx - 1], ordered[idx], ordered[idx + 1])
            if sign > 0:
                out[ordered[idx]] = RED
            elif sign < 0:
                out[ordered[idx]] = BLUE

    return out


def _turn_index(dir_idx: int, turn: str) -> int:
    if turn == "L":
        return (dir_idx - 1) % 4
    if turn == "R":
        return (dir_idx + 1) % 4
    raise AssertionError(turn)


def _build_relative_path(turns: list[str], lengths: list[int], initial_dir: int) -> list[tuple[int, int]] | None:
    assert len(lengths) == len(turns) + 1
    points = [(0, 0)]
    used = {points[0]}
    dir_idx = initial_dir

    for seg_idx, length in enumerate(lengths):
        dr, dc = DIRS4[dir_idx]
        for _ in range(length):
            nr = points[-1][0] + dr
            nc = points[-1][1] + dc
            candidate = (nr, nc)
            if candidate in used:
                return None
            neighbor_count = sum((nr + rr, nc + cc) in used for rr, cc in DIRS4)
            if neighbor_count != 1:
                return None
            points.append(candidate)
            used.add(candidate)

        if seg_idx < len(turns):
            dir_idx = _turn_index(dir_idx, turns[seg_idx])

    return points


def _normalize_path(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in points)
    min_c = min(c for _r, c in points)
    return [(r - min_r, c - min_c) for r, c in points]


def _path_bbox(points: list[tuple[int, int]]) -> tuple[int, int]:
    height = max(r for r, _c in points) + 1
    width = max(c for _r, c in points) + 1
    return height, width


def _turn_labels_from_points(points: list[tuple[int, int]]) -> list[str]:
    ordered = points if points[0] <= points[-1] else list(reversed(points))
    labels: list[str] = []
    for idx in range(1, len(ordered) - 1):
        sign = _turn_sign(ordered[idx - 1], ordered[idx], ordered[idx + 1])
        if sign > 0:
            labels.append("L")
        elif sign < 0:
            labels.append("R")
    return labels


def _sample_turns(rng: random.Random, turn_range: tuple[int, int], require_both: bool) -> list[str]:
    turn_count = rng.randint(turn_range[0], turn_range[1])
    turns = [rng.choice(["L", "R"]) for _ in range(turn_count)]
    if require_both and len(turns) >= 2 and len(set(turns)) == 1:
        flip_idx = rng.randrange(len(turns))
        turns[flip_idx] = "L" if turns[0] == "R" else "R"
    return turns


def _generate_path_shape(rng: random.Random, require_both: bool) -> list[tuple[int, int]] | None:
    turn_range = PRIMARY_TURN_RANGE if require_both else SECONDARY_TURN_RANGE

    for _ in range(PATH_SHAPE_ATTEMPTS):
        turns = _sample_turns(rng, turn_range, require_both)
        lengths = [rng.randint(*SEGMENT_LENGTH_RANGE) for _ in range(len(turns) + 1)]
        points = _build_relative_path(turns, lengths, rng.randrange(4))
        if points is None:
            continue

        points = _normalize_path(points)
        height, width = _path_bbox(points)
        if height > GRID_MAX_SIZE - 2 * GRID_BORDER or width > GRID_MAX_SIZE - 2 * GRID_BORDER:
            continue

        labels = _turn_labels_from_points(points)
        if require_both and set(labels) != {"L", "R"}:
            continue

        return points

    return None


def _place_paths(
    rng: random.Random,
    path_specs: list[dict[str, object]],
    height: int,
    width: int,
    bg: int,
) -> np.ndarray | None:
    grid = np.full((height, width), bg, dtype=int)
    blocked: set[tuple[int, int]] = set()

    specs = sorted(
        path_specs,
        key=lambda spec: (
            int(spec["bbox"][0]) * int(spec["bbox"][1]),
            len(spec["points"]),
        ),
        reverse=True,
    )

    for spec in specs:
        points = spec["points"]
        color = int(spec["color"])
        box_h, box_w = spec["bbox"]

        top_min = GRID_BORDER
        left_min = GRID_BORDER
        top_max = height - box_h - GRID_BORDER
        left_max = width - box_w - GRID_BORDER
        if top_min > top_max or left_min > left_max:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(top_min, top_max)
            left = rng.randint(left_min, left_max)
            shifted = [(r + top, c + left) for r, c in points]
            if any(cell in blocked for cell in shifted):
                continue

            for r, c in shifted:
                grid[r, c] = color
            for r, c in shifted:
                for dr, dc in DIRS8:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        blocked.add((nr, nc))

            placed = True
            break

        if not placed:
            return None

    return grid


def _dominant_background(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    if int(values[order[0]]) != bg:
        return False
    if len(order) == 1:
        return True
    return int(counts[order[0]]) > int(counts[order[1]])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(NEUTRAL_COLORS)
    path_count = rng.randint(PATH_COUNT_MIN, PATH_COUNT_MAX)
    palette = [color for color in NEUTRAL_COLORS if color != bg]
    colors = rng.sample(palette, path_count)

    path_specs: list[dict[str, object]] = []
    primary = _generate_path_shape(rng, require_both=True)
    if primary is None:
        return None
    path_specs.append({"points": primary, "bbox": _path_bbox(primary), "color": colors[0]})

    for idx in range(1, path_count):
        shape = _generate_path_shape(rng, require_both=False)
        if shape is None:
            return None
        path_specs.append({"points": shape, "bbox": _path_bbox(shape), "color": colors[idx]})

    for _ in range(GRID_LAYOUT_ATTEMPTS):
        height = rng.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
        width = rng.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
        grid = _place_paths(rng, path_specs, height, width, bg)
        if grid is None:
            continue
        if not _dominant_background(grid, bg):
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if not np.any(out == RED):
            continue
        if not np.any(out == BLUE):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid turn-coding path puzzle")
