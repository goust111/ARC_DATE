from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

BASE_COLORS = (RED, BLUE, YELLOW)
ALL_RECIPES = (
    (RED,),
    (BLUE,),
    (YELLOW,),
    (RED, BLUE),
    (RED, YELLOW),
    (BLUE, YELLOW),
    (RED, BLUE, YELLOW),
)
MIX_MAP = {
    frozenset((RED,)): RED,
    frozenset((BLUE,)): BLUE,
    frozenset((YELLOW,)): YELLOW,
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
    frozenset((RED, BLUE, YELLOW)): GREY,
}


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _border_coords(top: int, left: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(left, left + 5):
        cells.append((top, c))
        cells.append((top + 4, c))
    for r in range(top + 1, top + 4):
        cells.append((r, left))
        cells.append((r, left + 4))
    return cells


def _find_containers(grid: np.ndarray, bg: int):
    h, w = grid.shape
    candidates = []
    border_mask = np.ones((5, 5), dtype=bool)
    border_mask[1:4, 1:4] = False
    for top in range(h - 4):
        for left in range(w - 4):
            patch = grid[top : top + 5, left : left + 5]
            border_color = int(patch[0, 0])
            if border_color == bg:
                continue
            if not np.all(patch[border_mask] == border_color):
                continue
            fill = int(patch[1, 1])
            if not np.all(patch[1:4, 1:4] == fill):
                continue
            if fill != bg and fill not in BASE_COLORS:
                continue
            candidates.append((top, left, border_color, fill))
    assert candidates, "expected at least one 5x5 container"

    frame_color = Counter(border_color for _, _, border_color, _ in candidates).most_common(1)[0][0]
    containers = []
    seen = set()
    for top, left, border_color, fill in candidates:
        if border_color != frame_color or (top, left) in seen:
            continue
        seen.add((top, left))
        containers.append(
            {
                "top": top,
                "left": left,
                "fill": fill,
            }
        )
    assert containers, "expected at least one container with the shared frame color"
    return frame_color, containers


def _extract_ports(grid: np.ndarray, frame_color: int, containers):
    h, w = grid.shape
    border_mask = np.zeros_like(grid, dtype=bool)
    for container in containers:
        for r, c in _border_coords(container["top"], container["left"]):
            border_mask[r, c] = True
    pipe_mask = (grid == frame_color) & ~border_mask

    endpoint_to_ports: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    ports = []
    for idx, container in enumerate(containers):
        for br, bc in _border_coords(container["top"], container["left"]):
            outside = [(nr, nc) for nr, nc in _neighbors4(br, bc, h, w) if pipe_mask[nr, nc]]
            for endpoint in outside:
                port = {"container": idx, "border": (br, bc), "endpoint": endpoint}
                ports.append(port)
                endpoint_to_ports[endpoint].append(port)

    for endpoint, endpoint_ports in endpoint_to_ports.items():
        assert len(endpoint_ports) == 1, f"ambiguous endpoint {endpoint}"
    return pipe_mask, ports, endpoint_to_ports


def _trace_partner(start_port, pipe_mask: np.ndarray, endpoint_to_ports):
    start_endpoint = start_port["endpoint"]
    start_container = int(start_port["container"])
    prev = start_port["border"]
    curr = start_endpoint
    h, w = pipe_mask.shape
    seen = set()

    while True:
        state = (prev, curr)
        assert state not in seen, "pipe loop detected"
        seen.add(state)

        if curr != start_endpoint and curr in endpoint_to_ports:
            partner = endpoint_to_ports[curr][0]
            if int(partner["container"]) != start_container:
                return partner

        neighbors = [(nr, nc) for nr, nc in _neighbors4(curr[0], curr[1], h, w) if pipe_mask[nr, nc]]
        forward = [cell for cell in neighbors if cell != prev]
        assert forward, "pipe ended without reaching a container"
        if len(forward) == 1:
            nxt = forward[0]
        else:
            dr = curr[0] - prev[0]
            dc = curr[1] - prev[1]
            straight = (curr[0] + dr, curr[1] + dc)
            assert straight in forward, "ambiguous pipe crossing"
            nxt = straight
        prev, curr = curr, nxt


def _fill_terminals(grid: np.ndarray, containers, connections, bg: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    terminal_colors: dict[int, set[int]] = defaultdict(set)
    for left_port, right_port in connections:
        li = int(left_port["container"])
        ri = int(right_port["container"])
        left_fill = int(containers[li]["fill"])
        right_fill = int(containers[ri]["fill"])
        left_is_source = left_fill != bg
        right_is_source = right_fill != bg
        if left_is_source and not right_is_source:
            terminal_colors[ri].add(left_fill)
        elif right_is_source and not left_is_source:
            terminal_colors[li].add(right_fill)
        else:
            raise AssertionError("every pipe should connect exactly one source to one terminal")

    for idx, colors in terminal_colors.items():
        color = MIX_MAP[frozenset(colors)]
        top = int(containers[idx]["top"])
        left = int(containers[idx]["left"])
        out[top + 1 : top + 4, left + 1 : left + 4] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = Counter(map(int, g.flat)).most_common(1)[0][0]
    frame_color, containers = _find_containers(g, bg)
    pipe_mask, ports, endpoint_to_ports = _extract_ports(g, frame_color, containers)
    _ = pipe_mask
    used_endpoints = set()
    connections = []
    for port in ports:
        endpoint = port["endpoint"]
        if endpoint in used_endpoints:
            continue
        partner = _trace_partner(port, pipe_mask, endpoint_to_ports)
        used_endpoints.add(endpoint)
        used_endpoints.add(partner["endpoint"])
        connections.append((port, partner))
    return _fill_terminals(g, containers, connections, bg)


def _draw_container(grid: np.ndarray, top: int, left: int, fill: int) -> None:
    grid[top : top + 5, left] = GREY
    grid[top : top + 5, left + 4] = GREY
    grid[top, left : left + 5] = GREY
    grid[top + 4, left : left + 5] = GREY
    grid[top + 1 : top + 4, left + 1 : left + 4] = fill


def _paint_pipe_cell(grid: np.ndarray, r: int, c: int) -> None:
    value = int(grid[r, c])
    if value not in (BLACK, GREY):
        raise ValueError("pipe would overlap a container interior")
    grid[r, c] = GREY


def _paint_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            _paint_pipe_cell(grid, r0, c)
    elif c0 == c1:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            _paint_pipe_cell(grid, r, c0)
    else:
        raise ValueError("segments must be axis-aligned")


def _sample_recipes(rng: random.Random, terminal_count: int) -> list[tuple[int, ...]]:
    for _ in range(200):
        recipes = rng.sample(list(ALL_RECIPES), terminal_count)
        covered = set()
        for recipe in recipes:
            covered.update(recipe)
        if covered == set(BASE_COLORS):
            return recipes
    raise RuntimeError("failed to cover all source colors")


def _transform_grid(grid: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        out = grid
    elif code == 1:
        out = np.flipud(grid)
    elif code == 2:
        out = np.fliplr(grid)
    elif code == 3:
        out = np.flipud(np.fliplr(grid))
    elif code == 4:
        out = grid.T
    elif code == 5:
        out = np.flipud(grid.T)
    elif code == 6:
        out = np.fliplr(grid.T)
    elif code == 7:
        out = np.flipud(np.fliplr(grid.T))
    else:
        raise ValueError("unknown transform code")
    return np.array(out, copy=True)


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    terminal_count = rng.choice((2, 3))
    recipes = _sample_recipes(rng, terminal_count)
    total_wires = sum(len(recipe) for recipe in recipes)

    total_containers = 3 + terminal_count
    slot_count = total_containers + rng.randint(0, 1)
    height = 6 * slot_count + rng.randint(4, 7)
    source_left = rng.randint(1, 3)
    width = rng.randint(max(24, total_wires + 18), max(30, total_wires + 24))
    terminal_left = width - 5 - rng.randint(1, 3)
    lane_candidates = list(range(source_left + 7, terminal_left - 1))
    if len(lane_candidates) < total_wires:
        raise RuntimeError("not enough routing lanes")

    candidate_tops = [1 + 6 * i for i in range(slot_count)]
    chosen_tops = rng.sample(candidate_tops, total_containers)
    rng.shuffle(chosen_tops)
    source_tops = chosen_tops[:3]
    terminal_tops = chosen_tops[3:]

    grid = np.full((height, width), BLACK, dtype=int)
    expected = np.full((height, width), BLACK, dtype=int)

    colors = list(BASE_COLORS)
    rng.shuffle(colors)
    sources = {}
    for color, top in zip(colors, source_tops):
        _draw_container(grid, top, source_left, color)
        _draw_container(expected, top, source_left, color)
        rows = [top + 1, top + 2, top + 3]
        rng.shuffle(rows)
        sources[color] = {"top": top, "rows": rows}

    terminals = []
    for recipe, top in zip(recipes, terminal_tops):
        _draw_container(grid, top, terminal_left, BLACK)
        mixed = MIX_MAP[frozenset(recipe)]
        _draw_container(expected, top, terminal_left, mixed)
        terminals.append({"recipe": tuple(recipe), "top": top})

    wires = []
    for terminal in terminals:
        rows = [terminal["top"] + 1, terminal["top"] + 2, terminal["top"] + 3]
        terminal_rows = rng.sample(rows, len(terminal["recipe"]))
        recipe_colors = list(terminal["recipe"])
        rng.shuffle(recipe_colors)
        for color, end_row in zip(recipe_colors, terminal_rows):
            start_row = sources[color]["rows"].pop()
            wires.append(
                {
                    "start": (start_row, source_left + 5),
                    "end": (end_row, terminal_left - 1),
                }
            )

    rng.shuffle(wires)
    lanes = rng.sample(lane_candidates, len(wires))
    for wire, lane in zip(wires, lanes):
        sr, sc = wire["start"]
        tr, tc = wire["end"]
        _paint_segment(grid, (sr, sc), (sr, lane))
        _paint_segment(grid, (sr, lane), (tr, lane))
        _paint_segment(grid, (tr, lane), (tr, tc))
        _paint_segment(expected, (sr, sc), (sr, lane))
        _paint_segment(expected, (sr, lane), (tr, lane))
        _paint_segment(expected, (tr, lane), (tr, tc))

    transform_code = rng.randrange(8)
    grid = _transform_grid(grid, transform_code)
    expected = _transform_grid(expected, transform_code)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        try:
            grid, expected = _generate_case(rng)
            actual = main(grid)
            if np.array_equal(actual, expected):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
