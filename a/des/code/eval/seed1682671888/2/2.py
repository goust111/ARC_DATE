from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (int(a[0] + b[0]), int(a[1] + b[1]))


def _mul(d: tuple[int, int], k: int) -> tuple[int, int]:
    return (int(d[0] * k), int(d[1] * k))


def _in_bounds(rc: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = rc
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_perpendicular(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] * b[0] + a[1] * b[1] == 0


def _most_common_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    order = np.argsort(-counts)
    assert len(order) > 0
    if len(order) > 1:
        assert int(counts[order[0]]) > int(counts[order[1]])
    return int(values[order[0]])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _rail_data(g: np.ndarray, rail_color: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    rails: list[dict] = []
    cell_to_rail: dict[tuple[int, int], int] = {}
    for comp in _find_components(g == int(rail_color)):
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        assert len(comp) in (2, 3, 4), "rails must be length 2-4"
        if len(rows) == 1:
            row = rows[0]
            c0, c1 = cols[0], cols[-1]
            expect = [(row, c) for c in range(c0, c1 + 1)]
            assert comp and len(expect) == len(comp), "horizontal rail must be contiguous"
            cells = expect
            endpoint_dirs = {
                cells[0]: RIGHT,
                cells[-1]: LEFT,
            }
        else:
            assert len(cols) == 1, "rail must be straight"
            col = cols[0]
            r0, r1 = rows[0], rows[-1]
            expect = [(r, col) for r in range(r0, r1 + 1)]
            assert comp and len(expect) == len(comp), "vertical rail must be contiguous"
            cells = expect
            endpoint_dirs = {
                cells[0]: DOWN,
                cells[-1]: UP,
            }
        rail_idx = len(rails)
        for cell in cells:
            assert cell not in cell_to_rail
            cell_to_rail[cell] = rail_idx
        rails.append({"cells": cells, "endpoint_dirs": endpoint_dirs})
    return rails, cell_to_rail


def _seed_specs(g: np.ndarray, bg: int, rail_color: int) -> list[dict]:
    h, w = g.shape
    counts = Counter(map(int, g.reshape(-1)))
    seed_specs: list[dict] = []
    for color, count in counts.items():
        if color in (int(bg), int(rail_color)):
            continue
        assert count == 1, "each seed color must appear exactly once"
        rc = tuple(map(int, np.argwhere(g == int(color))[0]))
        assert 0 <= rc[0] < h and 0 <= rc[1] < w
        seed_specs.append({"color": int(color), "seed": rc})
    seed_specs.sort(key=lambda spec: (spec["seed"][0], spec["seed"][1]))
    return seed_specs


def _trace_vine(
    g: np.ndarray,
    rail_color: int,
    rails: list[dict],
    cell_to_rail: dict[tuple[int, int], int],
    seed: tuple[int, int],
    init_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = g.shape
    path = [seed]
    occupied = {seed}
    pos = seed
    direction = init_dir

    for _ in range(h * w * 4):
        nxt = _add(pos, direction)
        if not _in_bounds(nxt, g.shape):
            break
        if nxt in occupied:
            break
        if int(g[nxt]) == int(rail_color):
            rail = rails[cell_to_rail[nxt]]
            assert nxt in rail["endpoint_dirs"], "turning rail must be hit at an endpoint"
            new_dir = rail["endpoint_dirs"][nxt]
            assert _is_perpendicular(direction, new_dir), "turn must change axis"
            direction = new_dir
            continue
        pos = nxt
        occupied.add(pos)
        path.append(pos)
    else:
        raise AssertionError("vine tracing exceeded safety bound")

    return path


def _solve_paths(g: np.ndarray) -> tuple[list[dict], int]:
    grid = np.asarray(g, dtype=int)
    bg = _most_common_color(grid)
    counts = Counter(map(int, grid.reshape(-1)))
    non_bg = [int(color) for color in counts if int(color) != int(bg)]
    rail_candidates = [color for color in non_bg if counts[color] > 1]
    assert len(rail_candidates) == 1, "expected one repeated rail color"
    rail_color = int(rail_candidates[0])

    rails, cell_to_rail = _rail_data(grid, rail_color)
    seeds = _seed_specs(grid, bg, rail_color)

    vines: list[dict] = []
    for order, spec in enumerate(seeds):
        seed = spec["seed"]
        candidates: list[tuple[int, int]] = []
        for dr, dc in DIRS:
            endpoint = (seed[0] + dr, seed[1] + dc)
            if endpoint not in cell_to_rail:
                continue
            rail = rails[cell_to_rail[endpoint]]
            if endpoint not in rail["endpoint_dirs"]:
                continue
            init_dir = rail["endpoint_dirs"][endpoint]
            if _is_perpendicular((dr, dc), init_dir):
                candidates.append(init_dir)
        assert len(candidates) == 1, "seed must identify exactly one rail endpoint"
        path = _trace_vine(grid, rail_color, rails, cell_to_rail, seed, candidates[0])
        vines.append(
            {
                "order": int(order),
                "color": int(spec["color"]),
                "seed": seed,
                "path": path,
                "length": int(len(path)),
            }
        )
    return vines, rail_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored vines from seed pixels. A seed reads the orientation of the
    adjacent grey rail endpoint to obtain its initial direction. While moving,
    a vine turns whenever its next cell is a grey rail endpoint, adopting that
    rail's axis, and it stops on boundary exit or before re-entering its own
    path. Grey rails remain unchanged, and crossing vine pixels are resolved by
    preferring the shorter total vine; equal lengths keep the earlier seed in
    row-major order.
    """
    g = np.asarray(grid, dtype=int)
    vines, rail_color = _solve_paths(g)

    winners: dict[tuple[int, int], tuple[int, int, int]] = {}
    for vine in vines:
        key = (int(vine["length"]), int(vine["order"]))
        for cell in vine["path"]:
            current = winners.get(cell)
            if current is None or key < current[:2]:
                winners[cell] = (key[0], key[1], int(vine["color"]))

    out = g.copy()
    for (r, c), (_length, _order, color) in winners.items():
        if int(g[r, c]) != int(rail_color):
            out[r, c] = int(color)
    return out


def _path_from_segments(
    start: tuple[int, int],
    segments: list[tuple[tuple[int, int], int]],
) -> list[tuple[int, int]]:
    path = [start]
    pos = start
    seen = {start}
    for direction, steps in segments:
        assert steps >= 1
        for _ in range(steps):
            pos = _add(pos, direction)
            assert pos not in seen, "generator paths must stay simple before the terminal loop check"
            path.append(pos)
            seen.add(pos)
    return path


def _rails_from_vine_spec(vine: dict) -> list[list[tuple[int, int]]]:
    start = vine["seed"]
    init_dir = vine["segments"][0][0]
    init_normal = vine["init_normal"]
    init_len = int(vine["init_rail_len"])
    rails = [[_add(_add(start, init_normal), _mul(init_dir, k)) for k in range(init_len)]]

    pos = start
    segments = vine["segments"]
    turn_lengths = vine["turn_rail_lens"]
    assert len(turn_lengths) == len(segments) - 1
    for i, (direction, steps) in enumerate(segments):
        pos = _add(pos, _mul(direction, steps))
        if i + 1 == len(segments):
            continue
        next_dir = segments[i + 1][0]
        endpoint = _add(pos, direction)
        rail_len = int(turn_lengths[i])
        rails.append([_add(endpoint, _mul(next_dir, k)) for k in range(rail_len)])
    return rails


def _render_input(
    shape: tuple[int, int],
    bg: int,
    rail_color: int,
    vines: list[dict],
    extra_rails: list[list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    inp = np.full((h, w), int(bg), dtype=int)
    rail_cells: set[tuple[int, int]] = set()
    path_cells_by_vine: list[list[tuple[int, int]]] = []
    occupied_seeds: set[tuple[int, int]] = set()

    for vine in vines:
        path = _path_from_segments(vine["seed"], vine["segments"])
        for cell in path:
            assert _in_bounds(cell, shape)
        path_cells_by_vine.append(path)
        assert vine["seed"] not in occupied_seeds
        occupied_seeds.add(vine["seed"])

    for vine, path in zip(vines, path_cells_by_vine):
        for rail in _rails_from_vine_spec(vine):
            for cell in rail:
                assert _in_bounds(cell, shape)
                assert cell not in rail_cells
                rail_cells.add(cell)
    for rail in extra_rails:
        for cell in rail:
            assert _in_bounds(cell, shape)
            assert cell not in rail_cells
            rail_cells.add(cell)

    for cell in rail_cells:
        for path in path_cells_by_vine:
            assert cell not in path, "rails may not occupy vine path cells"

    for cell in rail_cells:
        inp[cell] = int(rail_color)
    for vine in vines:
        seed = vine["seed"]
        assert inp[seed] == int(bg)
        inp[seed] = int(vine["color"])

    expected = np.full_like(inp, int(bg))
    for cell in rail_cells:
        expected[cell] = int(rail_color)
    winners: dict[tuple[int, int], tuple[int, int, int]] = {}
    for order, (vine, path) in enumerate(zip(vines, path_cells_by_vine)):
        for cell in path:
            key = (len(path), order)
            current = winners.get(cell)
            if current is None or key < current[:2]:
                winners[cell] = (key[0], key[1], int(vine["color"]))
    for cell, (_length, _order, color) in winners.items():
        if cell not in rail_cells:
            expected[cell] = int(color)

    return inp, expected


def _seed_order(vines: list[dict]) -> list[dict]:
    return sorted(vines, key=lambda vine: (vine["seed"][0], vine["seed"][1]))


def _validated_scene(
    shape: tuple[int, int],
    bg: int,
    rail_color: int,
    vines: list[dict],
    extra_rails: list[list[tuple[int, int]]],
) -> np.ndarray | None:
    vines = _seed_order(vines)
    for vine in vines:
        seen = {vine["seed"]}
        pos = vine["seed"]
        for direction, steps in vine["segments"]:
            for _ in range(steps):
                pos = _add(pos, direction)
                if pos in seen:
                    return None
                seen.add(pos)

        final_dir = vine["segments"][-1][0]
        final_path = _path_from_segments(vine["seed"], vine["segments"])
        final_next = _add(final_path[-1], final_dir)
        if vine["stop"] == "boundary":
            if _in_bounds(final_next, shape):
                return None
        else:
            if final_next not in set(final_path):
                return None

    try:
        inp, expected = _render_input(shape, bg, rail_color, vines, extra_rails)
    except AssertionError:
        return None

    try:
        got = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(got, expected) else None


def _random_palette(seed_count: int) -> tuple[int, int, list[int]]:
    colors = list(range(10))
    bg = random.choice(colors)
    colors.remove(bg)
    rail = random.choice(colors)
    colors.remove(rail)
    seed_colors = random.sample(colors, seed_count)
    return int(bg), int(rail), [int(c) for c in seed_colors]


def _free_distractor_rails(
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    count: int,
) -> list[list[tuple[int, int]]]:
    h, w = shape
    rails: list[list[tuple[int, int]]] = []
    blocked = set(forbidden)
    for _ in range(count):
        placed = False
        for _ in range(80):
            direction = random.choice((RIGHT, DOWN))
            length = random.randint(2, 4)
            if direction == RIGHT:
                r = random.randint(0, h - 1)
                c = random.randint(0, w - length)
                rail = [(r, c + k) for k in range(length)]
            else:
                r = random.randint(0, h - length)
                c = random.randint(0, w - 1)
                rail = [(r + k, c) for k in range(length)]
            halo = set()
            for rr, cc in rail:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            halo.add((nr, nc))
            if halo & blocked:
                continue
            rails.append(rail)
            blocked |= halo
            placed = True
            break
        if not placed:
            break
    return rails


def _cross_scene(equal: bool) -> np.ndarray | None:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    max_len = min(h - 2, w - 2)
    if max_len < 6:
        return None

    if equal:
        length_h = length_v = random.randint(6, max_len)
    else:
        length_h = random.randint(6, max_len)
        choices = [n for n in range(6, max_len + 1) if n != length_h]
        if not choices:
            return None
        length_v = random.choice(choices)

    start_h_col = w - length_h
    start_v_row = h - length_v
    if start_h_col > w - 3 or start_v_row > h - 3:
        return None

    cross_r_lo = max(1, start_v_row + 1)
    cross_r_hi = h - 2
    cross_c_lo = max(1, start_h_col + 1)
    cross_c_hi = w - 2
    if cross_r_lo > cross_r_hi or cross_c_lo > cross_c_hi:
        return None
    cross_r = random.randint(cross_r_lo, cross_r_hi)
    cross_c = random.randint(cross_c_lo, cross_c_hi)

    seed_rows = [cross_r, start_v_row]
    seed_cols = [start_h_col, cross_c]
    if any(v <= 0 for v in seed_rows + seed_cols):
        return None
    if any(v >= lim - 1 for v, lim in ((cross_r, h), (start_v_row, h), (start_h_col, w), (cross_c, w))):
        return None

    bg, rail_color, seed_colors = _random_palette(2)
    vines = [
        {
            "color": seed_colors[0],
            "seed": (cross_r, start_h_col),
            "segments": [(RIGHT, w - 1 - start_h_col)],
            "init_normal": random.choice((UP, DOWN)),
            "init_rail_len": random.randint(2, 4),
            "turn_rail_lens": [],
            "stop": "boundary",
        },
        {
            "color": seed_colors[1],
            "seed": (start_v_row, cross_c),
            "segments": [(DOWN, h - 1 - start_v_row)],
            "init_normal": random.choice((LEFT, RIGHT)),
            "init_rail_len": random.randint(2, 4),
            "turn_rail_lens": [],
            "stop": "boundary",
        },
    ]

    forbidden = set()
    for vine in vines:
        forbidden.add(vine["seed"])
        forbidden.update(_path_from_segments(vine["seed"], vine["segments"]))
        for rail in _rails_from_vine_spec(vine):
            forbidden.update(rail)
    extra_rails = _free_distractor_rails((h, w), forbidden, random.randint(0, 2))
    return _validated_scene((h, w), bg, rail_color, vines, extra_rails)


def _loop_scene() -> np.ndarray | None:
    h = random.randint(13, 20)
    w = random.randint(13, 20)
    a = random.randint(3, min(6, w - 6))
    b = random.randint(3, min(6, h - 6))
    start_r = random.randint(2, h - b - 3)
    start_c = random.randint(2, w - a - 3)

    bg, rail_color, seed_colors = _random_palette(2)
    loop_vine = {
        "color": seed_colors[0],
        "seed": (start_r, start_c),
        "segments": [(RIGHT, a), (DOWN, b), (LEFT, a), (UP, b - 1)],
        "init_normal": random.choice((UP, DOWN)),
        "init_rail_len": random.randint(2, 4),
        "turn_rail_lens": [random.randint(2, 4) for _ in range(3)],
        "stop": "loop",
    }

    row2 = random.randint(1, h - 2)
    start2 = random.randint(1, w - 6)
    second_vine = {
        "color": seed_colors[1],
        "seed": (row2, start2),
        "segments": [(RIGHT, w - 1 - start2)],
        "init_normal": random.choice((UP, DOWN)),
        "init_rail_len": random.randint(2, 4),
        "turn_rail_lens": [],
        "stop": "boundary",
    }

    vines = [loop_vine, second_vine]
    forbidden = set()
    for vine in vines:
        forbidden.add(vine["seed"])
        forbidden.update(_path_from_segments(vine["seed"], vine["segments"]))
        for rail in _rails_from_vine_spec(vine):
            forbidden.update(rail)
    extra_rails = _free_distractor_rails((h, w), forbidden, random.randint(0, 3))
    return _validated_scene((h, w), bg, rail_color, vines, extra_rails)


def _turn_scene() -> np.ndarray | None:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    start_r = random.randint(1, h - 5)
    start_c = random.randint(1, w - 6)
    first_steps = random.randint(2, w - start_c - 4)
    second_steps = h - 1 - start_r
    if first_steps < 2 or second_steps < 3:
        return None

    bg, rail_color, seed_colors = _random_palette(3)
    vines = [
        {
            "color": seed_colors[0],
            "seed": (start_r, start_c),
            "segments": [(RIGHT, first_steps), (DOWN, second_steps)],
            "init_normal": random.choice((UP, DOWN)),
            "init_rail_len": random.randint(2, 4),
            "turn_rail_lens": [random.randint(2, 4)],
            "stop": "boundary",
        },
        {
            "color": seed_colors[1],
            "seed": (random.randint(1, h - 2), random.randint(1, max(1, w - 7))),
            "segments": [(RIGHT, random.randint(4, 6))],
            "init_normal": random.choice((UP, DOWN)),
            "init_rail_len": random.randint(2, 4),
            "turn_rail_lens": [],
            "stop": "boundary",
        },
        {
            "color": seed_colors[2],
            "seed": (random.randint(1, max(1, h - 7)), random.randint(1, w - 2)),
            "segments": [(DOWN, random.randint(4, 6))],
            "init_normal": random.choice((LEFT, RIGHT)),
            "init_rail_len": random.randint(2, 4),
            "turn_rail_lens": [],
            "stop": "boundary",
        },
    ]

    adjusted_vines: list[dict] = []
    for vine in vines:
        seed_r, seed_c = vine["seed"]
        direction, steps = vine["segments"][0]
        if direction == RIGHT:
            steps = min(steps, w - 1 - seed_c)
            if steps < 2:
                return None
            vine = {**vine, "segments": [(RIGHT, steps)], "stop": "boundary"}
        elif direction == DOWN and len(vine["segments"]) == 1:
            steps = min(steps, h - 1 - seed_r)
            if steps < 2:
                return None
            vine = {**vine, "segments": [(DOWN, steps)], "stop": "boundary"}
        adjusted_vines.append(vine)
    adjusted_vines[0] = vines[0]

    forbidden = set()
    for vine in adjusted_vines:
        forbidden.add(vine["seed"])
        forbidden.update(_path_from_segments(vine["seed"], vine["segments"]))
        for rail in _rails_from_vine_spec(vine):
            forbidden.update(rail)
    extra_rails = _free_distractor_rails((h, w), forbidden, random.randint(0, 3))
    return _validated_scene((h, w), bg, rail_color, adjusted_vines, extra_rails)


SCENARIOS = (_cross_scene, _loop_scene, _turn_scene, lambda: _cross_scene(equal=True))


def generate_input() -> np.ndarray:
    for _ in range(400):
        scene_builder = random.choice(SCENARIOS)
        grid = scene_builder() if scene_builder is not _cross_scene else _cross_scene(equal=False)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid vine-rail input")
