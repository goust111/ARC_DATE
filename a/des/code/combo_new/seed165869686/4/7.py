import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_COLOR = 0

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
PAIR_ATTEMPTS = 500
GRID_MIN = 12
GRID_MAX = 30
EDGE_BIAS = 0.35
PAIR_EDGE_BIAS = 0.15
LENGTH_RANGE = (1, 5)
OVERLAP_LENGTH_RANGE = (2, 5)
COLOR_POOL = tuple(range(1, 10))
MODES = ("isolated", "strong_overlap", "tie_overlap")


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _tower_pixels(center_r: int, center_c: int, length: int) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    for delta in range(-length, length + 1):
        pixels.add((int(center_r + delta), int(center_c)))
        pixels.add((int(center_r), int(center_c + delta)))
    return pixels


def _make_tower(color: int, center_r: int, center_c: int, length: int) -> dict:
    return {
        "color": int(color),
        "center": (int(center_r), int(center_c)),
        "length": int(length),
        "pixels": _tower_pixels(int(center_r), int(center_c), int(length)),
    }


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r, c in map(tuple, np.argwhere(grid != bg)):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue

        color = int(grid[r, c])
        stack = [(r, c)]
        visited[r, c] = True
        coords: list[tuple[int, int]] = []

        while stack:
            rr, cc = stack.pop()
            coords.append((int(rr), int(cc)))
            for dr, dc in DIRS4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

        components.append({"color": int(color), "coords": coords})

    return components


def _tower_from_component(component: dict) -> dict:
    color = int(component["color"])
    coords = {tuple(map(int, cell)) for cell in component["coords"]}
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)

    assert height == width and height % 2 == 1 and height >= 3, "tower bbox must be odd square"
    length = height // 2
    center_r = r0 + length
    center_c = c0 + length
    expected = _tower_pixels(center_r, center_c, length)
    assert coords == expected, "foreground components must be perfect plus towers"
    return _make_tower(color, center_r, center_c, length)


def _extract_towers(grid: np.ndarray, bg: int) -> list[dict]:
    towers = [_tower_from_component(component) for component in _components(grid, bg)]
    towers.sort(key=lambda tower: (tower["center"][0], tower["center"][1], tower["color"]))
    return towers


def _broadcast(grid: np.ndarray, towers: list[dict], bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best_len = np.full((h, w), -1, dtype=int)
    best_color = np.full((h, w), int(bg), dtype=int)
    tied = np.zeros((h, w), dtype=bool)

    for tower in towers:
        center_r, center_c = tower["center"]
        length = int(tower["length"])
        color = int(tower["color"])

        for dr in range(-length, length + 1):
            rr = int(center_r + dr)
            if not (0 <= rr < h):
                continue
            span = int(length - abs(dr))
            c_start = max(0, center_c - span)
            c_end = min(w - 1, center_c + span)
            for cc in range(c_start, c_end + 1):
                if int(g[rr, cc]) != int(bg):
                    continue
                if length > int(best_len[rr, cc]):
                    best_len[rr, cc] = length
                    best_color[rr, cc] = color
                    tied[rr, cc] = False
                elif length == int(best_len[rr, cc]) and int(best_color[rr, cc]) != color:
                    tied[rr, cc] = True

    out = g.copy()
    mask = (g == bg) & (best_len >= 0) & (~tied)
    out[mask] = best_color[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each plus-shaped signal tower into a same-color Manhattan diamond, with longer arms winning overlaps.

    Input structure:
    - The background is a single dominant color.
    - Each foreground object is a monochrome plus shape made from one center cell and four equal straight arms.
    - Different towers are separated by background and can have different colors and arm lengths.
    - No other foreground structure is present.

    Rule:
    - For every tower, use its arm length L as a broadcast radius around the tower center.
    - Recolor every background cell whose Manhattan distance from that center is at most L to the tower color.
    - If several towers cover the same background cell, keep only the color of the tower with the largest L; if that largest L is tied, leave the cell as background.
    - The original plus-shaped tower cells remain unchanged.

    Key insight:
    - The visible arm length is the numeric parameter of the transformation: it defines a diamond broadcast region, and overlap conflicts are resolved by comparing those arm lengths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    bg = _dominant_color(g)
    towers = _extract_towers(g, bg)
    assert towers, "expected at least one signal tower"
    return _broadcast(g, towers, bg)


def _paint_towers(shape: tuple[int, int], towers: list[dict], bg: int = BG_COLOR) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for tower in towers:
        for r, c in tower["pixels"]:
            grid[r, c] = int(tower["color"])
    return grid


def _random_center(height: int, width: int, length: int, edge_bias: float) -> tuple[int, int]:
    row = random.randint(length, height - 1 - length)
    col = random.randint(length, width - 1 - length)

    if random.random() < edge_bias:
        side = random.choice(("top", "bottom", "left", "right"))
        if side == "top":
            row = length
        elif side == "bottom":
            row = height - 1 - length
        elif side == "left":
            col = length
        else:
            col = width - 1 - length

    return int(row), int(col)


def _towers_touch(first: dict, second: dict) -> bool:
    for r1, c1 in first["pixels"]:
        for r2, c2 in second["pixels"]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _diamonds_overlap(first: dict, second: dict) -> bool:
    r1, c1 = first["center"]
    r2, c2 = second["center"]
    dist = abs(r1 - r2) + abs(c1 - c2)
    return dist <= int(first["length"]) + int(second["length"])


def _coverage_summary(grid: np.ndarray, towers: list[dict], bg: int) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    unique_overlap = False
    tie_overlap = False

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != int(bg):
                continue
            covering = [
                tower
                for tower in towers
                if abs(r - tower["center"][0]) + abs(c - tower["center"][1]) <= int(tower["length"])
            ]
            if len(covering) < 2:
                continue
            top_length = max(int(tower["length"]) for tower in covering)
            winners = sum(int(tower["length"]) == top_length for tower in covering)
            if winners == 1:
                unique_overlap = True
            else:
                tie_overlap = True

    return {"unique_overlap": unique_overlap, "tie_overlap": tie_overlap}


def _random_shape(lengths: list[int], count: int) -> tuple[int, int]:
    max_length = max(lengths)
    min_side = max(GRID_MIN, 2 * max_length + 2 + 3 * max(0, count - 1))
    high_side = max(min_side, GRID_MAX - random.randint(0, 4))
    height = random.randint(min_side, min(GRID_MAX, high_side))
    width = random.randint(min_side, min(GRID_MAX, high_side))
    return int(height), int(width)


def _try_place_isolated_tower(
    height: int,
    width: int,
    color: int,
    length: int,
    towers: list[dict],
) -> dict | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        center_r, center_c = _random_center(height, width, length, EDGE_BIAS)
        tower = _make_tower(color, center_r, center_c, length)
        if any(_towers_touch(tower, other) for other in towers):
            continue
        if any(_diamonds_overlap(tower, other) for other in towers):
            continue
        return tower
    return None


def _try_place_overlap_pair(
    height: int,
    width: int,
    color_a: int,
    color_b: int,
    length_a: int,
    length_b: int,
    mode: str,
) -> list[dict] | None:
    for _ in range(PAIR_ATTEMPTS):
        center_a = _random_center(height, width, length_a, PAIR_EDGE_BIAS)
        center_b = _random_center(height, width, length_b, PAIR_EDGE_BIAS)
        tower_a = _make_tower(color_a, center_a[0], center_a[1], length_a)
        tower_b = _make_tower(color_b, center_b[0], center_b[1], length_b)

        if tower_a["center"] == tower_b["center"]:
            continue
        if tower_a["center"][0] == tower_b["center"][0] or tower_a["center"][1] == tower_b["center"][1]:
            continue
        if _towers_touch(tower_a, tower_b):
            continue
        if not _diamonds_overlap(tower_a, tower_b):
            continue

        pair_grid = _paint_towers((height, width), [tower_a, tower_b])
        summary = _coverage_summary(pair_grid, [tower_a, tower_b], BG_COLOR)
        if mode == "strong_overlap" and summary["unique_overlap"] and not summary["tie_overlap"]:
            return [tower_a, tower_b]
        if mode == "tie_overlap" and summary["tie_overlap"] and not summary["unique_overlap"]:
            return [tower_a, tower_b]

    return None


def _sample_lengths(mode: str, count: int) -> list[int]:
    if mode == "strong_overlap":
        first_two = random.sample(range(OVERLAP_LENGTH_RANGE[0], OVERLAP_LENGTH_RANGE[1] + 1), 2)
        lengths = [int(first_two[0]), int(first_two[1])]
    elif mode == "tie_overlap":
        shared = random.randint(OVERLAP_LENGTH_RANGE[0], OVERLAP_LENGTH_RANGE[1])
        lengths = [int(shared), int(shared)]
    else:
        lengths = []

    while len(lengths) < count:
        lengths.append(random.randint(LENGTH_RANGE[0], LENGTH_RANGE[1]))

    if all(length == 1 for length in lengths):
        lengths[random.randrange(len(lengths))] = 2
    if len(lengths) > 2:
        extras = lengths[2:]
        random.shuffle(extras)
        lengths = lengths[:2] + extras
    return lengths


def _build_candidate(mode: str) -> np.ndarray | None:
    count = random.randint(2, 5 if mode == "isolated" else 4)
    colors = random.sample(COLOR_POOL, count)
    lengths = _sample_lengths(mode, count)
    height, width = _random_shape(lengths, count)
    towers: list[dict] = []

    if mode == "isolated":
        for color, length in zip(colors, lengths, strict=False):
            tower = _try_place_isolated_tower(height, width, color, length, towers)
            if tower is None:
                return None
            towers.append(tower)
    else:
        pair = _try_place_overlap_pair(
            height,
            width,
            colors[0],
            colors[1],
            lengths[0],
            lengths[1],
            mode,
        )
        if pair is None:
            return None
        towers.extend(pair)
        for idx in range(2, count):
            tower = _try_place_isolated_tower(height, width, colors[idx], lengths[idx], towers)
            if tower is None:
                return None
            towers.append(tower)

    grid = _paint_towers((height, width), towers)
    bg = _dominant_color(grid)
    parsed = _extract_towers(grid, bg)
    if len(parsed) != len(towers):
        return None

    summary = _coverage_summary(grid, parsed, bg)
    if mode == "isolated" and (summary["unique_overlap"] or summary["tie_overlap"]):
        return None
    if mode == "strong_overlap" and (not summary["unique_overlap"] or summary["tie_overlap"]):
        return None
    if mode == "tie_overlap" and (not summary["tie_overlap"] or summary["unique_overlap"]):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    mode = random.choice(MODES)
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid {mode} sample")
