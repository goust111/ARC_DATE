from __future__ import annotations

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _direction_for_seed(shape: tuple[int, int], r: int, c: int) -> Coord:
    h, w = shape
    touches = [r == 0, r == h - 1, c == 0, c == w - 1]
    assert sum(touches) == 1, "seed pixels must lie on exactly one border side"
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    return 0, -1


def _place_positions(count: int, lo: int, hi: int, min_gap: int, rng: np.random.Generator) -> list[int] | None:
    span = hi - lo + 1
    needed = count + (count - 1) * min_gap
    if span < needed:
        return None
    compressed = span - (count - 1) * min_gap
    picks = sorted(int(v) for v in rng.choice(compressed, size=count, replace=False))
    return [int(lo + picks[i] + i * min_gap) for i in range(count)]


def _parse_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    full_rows = [r for r in range(h) if np.all(g[r, :] != bg)]
    full_cols = [c for c in range(w) if np.all(g[:, c] != bg)]
    assert bool(full_rows) ^ bool(full_cols), "expected one parallel-wall orientation"

    if full_cols:
        wall_positions = sorted(int(c) for c in full_cols)
        wall_color = None
        door_center_by_wall: dict[int, int] = {}
        door_color_by_wall: dict[int, int] = {}
        for c in wall_positions:
            values, counts = np.unique(g[:, c], return_counts=True)
            counts_by_color = {int(v): int(n) for v, n in zip(values, counts)}
            door_colors = [color for color, n in counts_by_color.items() if n == 3]
            wall_colors = [color for color, n in counts_by_color.items() if n == h - 3]
            assert len(door_colors) == 1 and len(wall_colors) == 1, "vertical wall must have one 3-cell door"
            door_color = int(door_colors[0])
            body_color = int(wall_colors[0])
            door_rows = np.flatnonzero(g[:, c] == door_color)
            assert door_rows.size == 3 and np.all(np.diff(door_rows) == 1), "door must be contiguous"
            door_center_by_wall[c] = int(door_rows[1])
            door_color_by_wall[c] = door_color
            if wall_color is None:
                wall_color = body_color
            else:
                assert wall_color == body_color, "all walls must share one wall color"
        orientation = "vertical"
    else:
        wall_positions = sorted(int(r) for r in full_rows)
        wall_color = None
        door_center_by_wall = {}
        door_color_by_wall = {}
        for r in wall_positions:
            values, counts = np.unique(g[r, :], return_counts=True)
            counts_by_color = {int(v): int(n) for v, n in zip(values, counts)}
            door_colors = [color for color, n in counts_by_color.items() if n == 3]
            wall_colors = [color for color, n in counts_by_color.items() if n == w - 3]
            assert len(door_colors) == 1 and len(wall_colors) == 1, "horizontal wall must have one 3-cell door"
            door_color = int(door_colors[0])
            body_color = int(wall_colors[0])
            door_cols = np.flatnonzero(g[r, :] == door_color)
            assert door_cols.size == 3 and np.all(np.diff(door_cols) == 1), "door must be contiguous"
            door_center_by_wall[r] = int(door_cols[1])
            door_color_by_wall[r] = door_color
            if wall_color is None:
                wall_color = body_color
            else:
                assert wall_color == body_color, "all walls must share one wall color"
        orientation = "horizontal"

    role_colors = {int(bg), int(wall_color), *map(int, door_color_by_wall.values())}
    remaining = set(int(v) for v in np.unique(g)) - role_colors
    assert len(remaining) == 1, "expected exactly one seed color"
    seed_color = int(next(iter(remaining)))

    seeds = [tuple(map(int, rc)) for rc in np.argwhere(g == seed_color)]
    assert seeds, "expected border seed pixels"
    for r, c in seeds:
        assert r in (0, h - 1) or c in (0, w - 1), "seeds must be on the border"

    return {
        "bg": int(bg),
        "seed_color": int(seed_color),
        "wall_color": int(wall_color),
        "orientation": orientation,
        "wall_positions": wall_positions,
        "door_center_by_wall": door_center_by_wall,
        "door_color_by_wall": door_color_by_wall,
        "seeds": seeds,
    }


def _trace_rays(grid: np.ndarray, layout: dict) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(layout["bg"])
    seed_color = int(layout["seed_color"])
    orientation = layout["orientation"]
    door_center_by_wall = {int(k): int(v) for k, v in layout["door_center_by_wall"].items()}
    door_color_by_wall = {int(k): int(v) for k, v in layout["door_color_by_wall"].items()}
    wall_positions = set(int(v) for v in layout["wall_positions"])

    claims: dict[Coord, int] = {}
    for seed_r, seed_c in layout["seeds"]:
        dr, dc = _direction_for_seed(g.shape, seed_r, seed_c)
        cur_r, cur_c = int(seed_r), int(seed_c)
        color = seed_color

        while True:
            nxt_r = cur_r + dr
            nxt_c = cur_c + dc
            if not _in_bounds(g.shape, nxt_r, nxt_c):
                break

            if orientation == "vertical" and nxt_c in wall_positions:
                if nxt_r != door_center_by_wall[nxt_c]:
                    break
                cur_r, cur_c = nxt_r, nxt_c
                color = door_color_by_wall[nxt_c]
                continue

            if orientation == "horizontal" and nxt_r in wall_positions:
                if nxt_c != door_center_by_wall[nxt_r]:
                    break
                cur_r, cur_c = nxt_r, nxt_c
                color = door_color_by_wall[nxt_r]
                continue

            base = int(g[nxt_r, nxt_c])
            assert base in {bg, seed_color, color}, "rays may only paint empty cells or same-color traces"
            previous = claims.get((nxt_r, nxt_c))
            assert previous is None or previous == color, "ray overlaps must be color-consistent"
            claims[(nxt_r, nxt_c)] = color
            cur_r, cur_c = nxt_r, nxt_c

    out = g.copy()
    for (r, c), color in claims.items():
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each border seed inward, letting only door-center hits cross the walls and recolor the downstream ray.

    Input structure:
    - The grid has a dominant background, several parallel full-span one-cell-thick walls, and one three-cell door segment embedded in each wall.
    - All door segments lie along the wall direction, each wall has its own door color, and the middle cell of that segment is the only passable point.
    - Several border seed pixels share one color and emit rays perpendicular to the side they touch.

    Rule:
    - Starting from each seed, extend a straight ray cell by cell into the grid using the current ray color.
    - If the next cell is wall body or a non-center door cell, stop on the previous cell.
    - If the ray reaches a door center, cross that wall without painting over it, switch the ray color to that door's color, and continue on the far side.
    - Keep tracing until the ray reaches the grid edge; preserve the original walls, doors, and seeds.

    Key insight:
    - A three-cell door is not an open three-cell gap: only its middle cell transmits the ray, and every successful crossing also rewrites the color carried into later corridors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    layout = _parse_layout(g)
    return _trace_rays(g, layout)


def _render_input(spec: dict) -> np.ndarray:
    h, w = spec["shape"]
    grid = np.full((h, w), int(spec["bg"]), dtype=int)
    orientation = spec["orientation"]
    wall_color = int(spec["wall_color"])

    if orientation == "vertical":
        for c, center, door_color in zip(spec["wall_positions"], spec["door_centers"], spec["door_colors"]):
            grid[:, int(c)] = wall_color
            grid[int(center) - 1 : int(center) + 2, int(c)] = int(door_color)
    else:
        for r, center, door_color in zip(spec["wall_positions"], spec["door_centers"], spec["door_colors"]):
            grid[int(r), :] = wall_color
            grid[int(r), int(center) - 1 : int(center) + 2] = int(door_color)

    for r, c in spec["seeds"]:
        assert grid[int(r), int(c)] == int(spec["bg"]), "seed placement must avoid walls"
        grid[int(r), int(c)] = int(spec["seed_color"])
    return grid


def _expected_output(spec: dict) -> np.ndarray:
    inp = _render_input(spec)
    layout = {
        "bg": int(spec["bg"]),
        "seed_color": int(spec["seed_color"]),
        "wall_color": int(spec["wall_color"]),
        "orientation": spec["orientation"],
        "wall_positions": [int(v) for v in spec["wall_positions"]],
        "door_center_by_wall": {
            int(pos): int(center) for pos, center in zip(spec["wall_positions"], spec["door_centers"])
        },
        "door_color_by_wall": {
            int(pos): int(color) for pos, color in zip(spec["wall_positions"], spec["door_colors"])
        },
        "seeds": [tuple(map(int, seed)) for seed in spec["seeds"]],
    }
    return _trace_rays(inp, layout)


def _sample_distinct(pool: list[int], count: int, rng: np.random.Generator) -> list[int]:
    picks = rng.choice(pool, size=count, replace=False)
    return [int(v) for v in picks]


def _build_seed_list(
    orientation: str,
    axis_span: int,
    run_start: int,
    run_end: int,
    rng: np.random.Generator,
) -> list[tuple[str, int]]:
    axis_pool = list(range(1, axis_span - 1))
    used: set[int] = set()
    seeds: list[tuple[str, int]] = []

    side_a, side_b = ("left", "right") if orientation == "vertical" else ("top", "bottom")
    seeds.append((side_a, int(run_start)))
    used.add(int(run_start))
    if run_end != run_start:
        seeds.append((side_b, int(run_end)))
        used.add(int(run_end))

    target = int(rng.integers(3, 6))
    filler_pool = [axis for axis in axis_pool if axis not in used]
    rng.shuffle(filler_pool)
    sides = [side_b, side_a] if run_end == run_start else [side_a, side_b]
    side_index = 0
    while len(seeds) < target and filler_pool:
        axis = int(filler_pool.pop())
        side = sides[side_index % len(sides)]
        side_index += 1
        if side == side_a and axis == run_start:
            continue
        if side == side_b and axis == run_end:
            continue
        seeds.append((side, axis))
    return seeds


def _seed_coords_from_axes(shape: tuple[int, int], orientation: str, seed_axes: list[tuple[str, int]]) -> list[Coord]:
    h, w = shape
    coords: list[Coord] = []
    used_axis: set[int] = set()
    for side, axis in seed_axes:
        assert axis not in used_axis, "parallel rays must use distinct rows or columns"
        used_axis.add(int(axis))
        if orientation == "vertical":
            coords.append((int(axis), 0 if side == "left" else w - 1))
        else:
            coords.append((0 if side == "top" else h - 1, int(axis)))
    return coords


def _build_spec(rng: np.random.Generator) -> dict | None:
    orientation = "vertical" if bool(rng.integers(0, 2)) else "horizontal"
    wall_count = int(rng.integers(3, 6))

    if orientation == "vertical":
        h = int(rng.integers(10, 19))
        w = int(rng.integers(max(12, wall_count * 3 + 3), 29))
        wall_positions = _place_positions(wall_count, 2, w - 3, 2, rng)
        axis_span = h
    else:
        h = int(rng.integers(max(12, wall_count * 3 + 3), 29))
        w = int(rng.integers(10, 19))
        wall_positions = _place_positions(wall_count, 2, h - 3, 2, rng)
        axis_span = w
    if wall_positions is None:
        return None

    centers_pool = list(range(2, axis_span - 2))
    if len(centers_pool) < 2:
        return None
    full_pass = bool(rng.integers(0, 100) < 40)
    if full_pass:
        through_axis = int(rng.choice(centers_pool))
        door_centers = [through_axis] * wall_count
    else:
        first_axis, last_axis = _sample_distinct(centers_pool, 2, rng)
        split = int(rng.integers(1, wall_count))
        door_centers = [first_axis] * split + [last_axis] * (wall_count - split)

    seed_axes = _build_seed_list(orientation, axis_span, door_centers[0], door_centers[-1], rng)
    seed_coords = _seed_coords_from_axes((h, w), orientation, seed_axes)

    palette = list(range(10))
    bg, seed_color, wall_color = _sample_distinct(palette, 3, rng)
    remaining = [color for color in palette if color not in {bg, seed_color, wall_color}]
    if len(remaining) < wall_count:
        return None
    door_colors = _sample_distinct(remaining, wall_count, rng)

    return {
        "shape": (h, w),
        "bg": int(bg),
        "seed_color": int(seed_color),
        "wall_color": int(wall_color),
        "orientation": orientation,
        "wall_positions": [int(v) for v in wall_positions],
        "door_centers": [int(v) for v in door_centers],
        "door_colors": [int(v) for v in door_colors],
        "seeds": seed_coords,
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        spec = _build_spec(rng)
        if spec is None:
            continue
        inp = _render_input(spec)
        expected = _expected_output(spec)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid gate-ray puzzle")
