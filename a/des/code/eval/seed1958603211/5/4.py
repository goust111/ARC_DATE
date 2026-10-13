import random
from collections import deque

import numpy as np


AZURE = 8
BLACK = 0
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
AXIS = {"U": "V", "D": "V", "L": "H", "R": "H"}
ARROW_SHAPES = {
    "U": [(-1, 0), (0, -1), (0, 0), (0, 1)],
    "D": [(0, -1), (0, 0), (0, 1), (1, 0)],
    "L": [(-1, 0), (0, -1), (0, 0), (1, 0)],
    "R": [(-1, 0), (0, 0), (0, 1), (1, 0)],
}
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
GEN_SCENARIOS = ["no_cross", "h_shorter", "v_shorter", "tie"]
GEN_RETRIES = 400
PAIR_TRIES = 240


def _neighbors4(cell):
    r, c = cell
    for dr, dc in DIRS.values():
        yield r + dr, c + dc


def _components(mask: np.ndarray):
    remaining = {tuple(map(int, coord)) for coord in np.argwhere(mask)}
    comps = []
    while remaining:
        start = remaining.pop()
        q = deque([start])
        comp = [start]
        while q:
            cell = q.popleft()
            for nxt in _neighbors4(cell):
                if nxt in remaining:
                    remaining.remove(nxt)
                    q.append(nxt)
                    comp.append(nxt)
        comps.append(comp)
    return comps


def _arrow_cells(center, direction):
    r, c = center
    return {(r + dr, c + dc) for dr, dc in ARROW_SHAPES[direction]}


def _detect_arrow(component):
    cells = {tuple(map(int, cell)) for cell in component}
    if len(cells) != 4:
        return None

    center = None
    present_dirs = None
    for r, c in cells:
        here = []
        for name, (dr, dc) in DIRS.items():
            if (r + dr, c + dc) in cells:
                here.append(name)
        if len(here) == 3:
            if center is not None:
                return None
            center = (r, c)
            present_dirs = set(here)
        elif len(here) != 1:
            return None

    if center is None or present_dirs is None:
        return None

    missing = (set(DIRS) - present_dirs)
    if len(missing) != 1:
        return None
    direction = OPPOSITE[missing.pop()]
    if _arrow_cells(center, direction) != cells:
        return None
    return center, direction


def _pair_records(grid: np.ndarray):
    records = []
    colors = sorted(int(color) for color in np.unique(grid) if int(color) not in (AZURE, BLACK))
    for color in colors:
        comps = _components(grid == color)
        if len(comps) != 2:
            raise AssertionError("each foreground color must form exactly two arrows")
        arrows = []
        for comp in comps:
            detected = _detect_arrow(comp)
            if detected is None:
                raise AssertionError("every component must be a valid arrow sprite")
            center, direction = detected
            arrows.append({"center": center, "direction": direction})

        (r1, c1), (r2, c2) = arrows[0]["center"], arrows[1]["center"]
        if r1 == r2:
            left, right = sorted(arrows, key=lambda item: item["center"][1])
            if left["direction"] != "R" or right["direction"] != "L":
                raise AssertionError("horizontal arrow pairs must face each other")
            row = left["center"][0]
            c_left = left["center"][1]
            c_right = right["center"][1]
            records.append(
                {
                    "color": color,
                    "axis": "H",
                    "distance": c_right - c_left,
                    "path": [(row, c) for c in range(c_left + 2, c_right - 1)],
                }
            )
        elif c1 == c2:
            top, bottom = sorted(arrows, key=lambda item: item["center"][0])
            if top["direction"] != "D" or bottom["direction"] != "U":
                raise AssertionError("vertical arrow pairs must face each other")
            col = top["center"][1]
            r_top = top["center"][0]
            r_bottom = bottom["center"][0]
            records.append(
                {
                    "color": color,
                    "axis": "V",
                    "distance": r_bottom - r_top,
                    "path": [(r, col) for r in range(r_top + 2, r_bottom - 1)],
                }
            )
        else:
            raise AssertionError("paired arrows must share a row or a column")
    return records


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    out = g.copy()
    claims = {}
    for record in _pair_records(g):
        for cell in record["path"]:
            if g[cell] != AZURE:
                continue
            claims.setdefault(cell, []).append(record)

    for cell, options in claims.items():
        best = min(options, key=lambda item: (item["distance"], 0 if item["axis"] == "H" else 1))
        out[cell] = best["color"]
    return out


def _make_horizontal_pair(row, c_left, c_right, color):
    sprites = _arrow_cells((row, c_left), "R") | _arrow_cells((row, c_right), "L")
    path = {(row, c) for c in range(c_left + 2, c_right - 1)}
    return {
        "color": color,
        "axis": "H",
        "distance": c_right - c_left,
        "sprite_cells": sprites,
        "path_cells": path,
    }


def _make_vertical_pair(col, r_top, r_bottom, color):
    sprites = _arrow_cells((r_top, col), "D") | _arrow_cells((r_bottom, col), "U")
    path = {(r, col) for r in range(r_top + 2, r_bottom - 1)}
    return {
        "color": color,
        "axis": "V",
        "distance": r_bottom - r_top,
        "sprite_cells": sprites,
        "path_cells": path,
    }


def _halo(cells, height, width):
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _in_bounds(cells, height, width):
    return all(0 <= r < height and 0 <= c < width for r, c in cells)


def _split_options(total, max_a, max_b):
    options = []
    for a in range(2, max_a + 1):
        b = total - a
        if 2 <= b <= max_b:
            options.append((a, b))
    return options


def _totals(max_a, max_b):
    totals = []
    for total in range(5, max_a + max_b + 1):
        if _split_options(total, max_a, max_b):
            totals.append(total)
    return totals


def _can_place(pair, height, width, occupied_sprites, occupied_paths, sprite_halo):
    if not _in_bounds(pair["sprite_cells"], height, width):
        return False
    if pair["sprite_cells"] & sprite_halo:
        return False
    if pair["sprite_cells"] & occupied_paths:
        return False
    if _halo(pair["sprite_cells"], height, width) & occupied_paths:
        return False
    if pair["path_cells"] & occupied_sprites:
        return False
    if pair["path_cells"] & occupied_paths:
        return False
    if pair["path_cells"] & sprite_halo:
        return False
    return True


def _make_cross_pairs(height, width, colors, scenario, rng):
    for _ in range(PAIR_TRIES):
        cross_r = rng.randint(3, height - 4)
        cross_c = rng.randint(3, width - 4)
        max_left = cross_c - 1
        max_right = width - 2 - cross_c
        max_up = cross_r - 1
        max_down = height - 2 - cross_r

        h_totals = _totals(max_left, max_right)
        v_totals = _totals(max_up, max_down)
        if not h_totals or not v_totals:
            continue

        if scenario == "h_shorter":
            choices = [(dh, dv) for dh in h_totals for dv in v_totals if dh < dv]
        elif scenario == "v_shorter":
            choices = [(dh, dv) for dh in h_totals for dv in v_totals if dv < dh]
        else:
            choices = [(d, d) for d in h_totals if d in set(v_totals)]
        if not choices:
            continue

        dist_h, dist_v = rng.choice(choices)
        left_off, right_off = rng.choice(_split_options(dist_h, max_left, max_right))
        up_off, down_off = rng.choice(_split_options(dist_v, max_up, max_down))

        h_pair = _make_horizontal_pair(cross_r, cross_c - left_off, cross_c + right_off, colors[0])
        v_pair = _make_vertical_pair(cross_c, cross_r - up_off, cross_r + down_off, colors[1])
        if h_pair["path_cells"] & v_pair["path_cells"] != {(cross_r, cross_c)}:
            continue
        if h_pair["sprite_cells"] & v_pair["sprite_cells"]:
            continue
        if h_pair["sprite_cells"] & v_pair["path_cells"]:
            continue
        if v_pair["sprite_cells"] & h_pair["path_cells"]:
            continue
        if _halo(h_pair["sprite_cells"], height, width) & v_pair["sprite_cells"]:
            continue
        return [h_pair, v_pair]
    return None


def _random_pair(height, width, color, axis, rng):
    if axis == "H":
        if width < 8:
            return None
        dist = rng.randint(5, width - 3)
        start_max = width - 2 - dist
        if start_max < 1:
            return None
        row = rng.randint(1, height - 2)
        c_left = rng.randint(1, start_max)
        return _make_horizontal_pair(row, c_left, c_left + dist, color)

    if height < 8:
        return None
    dist = rng.randint(5, height - 3)
    start_max = height - 2 - dist
    if start_max < 1:
        return None
    col = rng.randint(1, width - 2)
    r_top = rng.randint(1, start_max)
    return _make_vertical_pair(col, r_top, r_top + dist, color)


def _render_pairs(height, width, pairs):
    grid = np.full((height, width), AZURE, dtype=int)
    for pair in pairs:
        for r, c in pair["sprite_cells"]:
            grid[r, c] = pair["color"]
    return grid


def _pair_intersections(pairs):
    hits = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            inter = pairs[i]["path_cells"] & pairs[j]["path_cells"]
            if inter:
                hits.append((i, j, inter))
    return hits


def _generate_attempt(rng):
    height = rng.randint(12, 18)
    width = rng.randint(12, 18)
    scenario = rng.choice(GEN_SCENARIOS)
    pair_count = rng.randint(2, 4)
    colors = rng.sample(GEN_COLORS, pair_count)

    pairs = []
    occupied_sprites = set()
    occupied_paths = set()
    sprite_halo = set()

    def register(pair):
        pairs.append(pair)
        occupied_sprites.update(pair["sprite_cells"])
        occupied_paths.update(pair["path_cells"])
        sprite_halo.update(_halo(pair["sprite_cells"], height, width))

    color_idx = 0
    if scenario != "no_cross":
        cross_pairs = _make_cross_pairs(height, width, colors[:2], scenario, rng)
        if cross_pairs is None:
            return None
        for pair in cross_pairs:
            register(pair)
        color_idx = 2

    remaining = pair_count - len(pairs)
    axes = []
    if scenario == "no_cross" and remaining >= 2 and rng.random() < 0.5:
        axes.extend(rng.sample(["H", "V"], 2))
    while len(axes) < remaining:
        axes.append(rng.choice(["H", "V"]))
    rng.shuffle(axes)

    for axis in axes:
        placed = None
        for _ in range(PAIR_TRIES):
            candidate = _random_pair(height, width, colors[color_idx], axis, rng)
            if candidate is None:
                continue
            if _can_place(candidate, height, width, occupied_sprites, occupied_paths, sprite_halo):
                placed = candidate
                break
        if placed is None:
            return None
        register(placed)
        color_idx += 1

    intersections = _pair_intersections(pairs)
    if scenario == "no_cross":
        if intersections:
            return None
    else:
        if len(intersections) != 1:
            return None
        i, j, _ = intersections[0]
        dist_i = pairs[i]["distance"]
        dist_j = pairs[j]["distance"]
        axes_i = pairs[i]["axis"]
        axes_j = pairs[j]["axis"]
        if {axes_i, axes_j} != {"H", "V"}:
            return None
        horiz = dist_i if axes_i == "H" else dist_j
        vert = dist_i if axes_i == "V" else dist_j
        if scenario == "h_shorter" and not (horiz < vert):
            return None
        if scenario == "v_shorter" and not (vert < horiz):
            return None
        if scenario == "tie" and horiz != vert:
            return None

    grid = _render_pairs(height, width, pairs)
    detected = _pair_records(grid)
    if len(detected) != pair_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
