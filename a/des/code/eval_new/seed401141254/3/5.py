from __future__ import annotations

from collections import deque
import random

import numpy as np


GREY = 5
MAROON = 9
CARDINAL_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_CHOICES = (0, 1, 2, 3, 4, 6, 7, 8)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_path_cell(grid: np.ndarray, bg: int, r: int, c: int) -> bool:
    return _in_bounds(grid.shape, r, c) and int(grid[r, c]) != bg


def _bridge_channels(grid: np.ndarray, bg: int, r: int, c: int) -> tuple[str, ...]:
    if int(grid[r, c]) != MAROON:
        return ()
    horiz = _is_path_cell(grid, bg, r, c - 1) and _is_path_cell(grid, bg, r, c + 1)
    vert = _is_path_cell(grid, bg, r - 1, c) and _is_path_cell(grid, bg, r + 1, c)
    channels: list[str] = []
    if horiz:
        channels.append("h")
    if vert:
        channels.append("v")
    return tuple(channels)


def _visible_node(info: dict[str, tuple[int, int, str]]) -> tuple[int, int, str]:
    if "h" in info:
        return info["h"]
    if "o" in info:
        return info["o"]
    return info["v"]


def _incident_node(info: dict[str, tuple[int, int, str]], dr: int, dc: int) -> tuple[int, int, str] | None:
    if "o" in info:
        return info["o"]
    return info.get("h" if dc else "v")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each path network by tracing endpoint connectivity through connected crossings and bridge crossings.

    Input structure:
    - A dominant background color surrounds one-pixel-wide orthogonal paths.
    - Grey path pixels form the ordinary road network, while some crossing pixels are Maroon bridges.
    - A Maroon bridge cell has both a horizontal and a vertical continuation; the two directions pass through without connecting.
    - Colored non-background pixels sit at path terminals and mark the endpoint colors to propagate.

    Rule:
    - Treat every non-background cell as part of the path graph, but split each Maroon bridge into an independent horizontal channel and vertical channel.
    - Ordinary Grey crossings stay fully connected, so any endpoint can spread through them into every reachable branch of that component.
    - Within each semantic component, propagate colors outward from all endpoint markers by shortest-path distance; when two colors meet, the earlier source in scan order wins the tie.
    - Render every ordinary path cell with its propagated color, and render each Maroon bridge cell with the horizontal channel color because that bridge is the visible overpass.

    Key insight:
    - The Maroon marker changes topology rather than color: it looks like a crossing, but it must be split into two non-connecting channels before endpoint colors can be propagated correctly.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape

    node_neighbors: dict[tuple[int, int, str], set[tuple[int, int, str]]] = {}
    cell_nodes: dict[tuple[int, int], dict[str, tuple[int, int, str]]] = {}
    endpoint_colors: dict[tuple[int, int, str], int] = {}

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg:
                continue
            channels = _bridge_channels(g, bg, r, c)
            info: dict[str, tuple[int, int, str]] = {}
            if channels:
                for ch in channels:
                    node = (r, c, ch)
                    node_neighbors[node] = set()
                    info[ch] = node
            else:
                node = (r, c, "o")
                node_neighbors[node] = set()
                info["o"] = node
            cell_nodes[(r, c)] = info
            if color not in (GREY, MAROON):
                endpoint_colors[_visible_node(info)] = color

    for r in range(h):
        for c in range(w):
            if (r, c) not in cell_nodes:
                continue
            here = cell_nodes[(r, c)]
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cell_nodes:
                    continue
                there = cell_nodes[(nr, nc)]
                a = _incident_node(here, dr, dc)
                b = _incident_node(there, dr, dc)
                if a is None or b is None:
                    continue
                node_neighbors[a].add(b)
                node_neighbors[b].add(a)

    node_colors: dict[tuple[int, int, str], int] = {}
    seen: set[tuple[int, int, str]] = set()
    for start in node_neighbors:
        if start in seen:
            continue
        component: list[tuple[int, int, str]] = []
        q = deque([start])
        seen.add(start)
        while q:
            node = q.popleft()
            component.append(node)
            for nxt in node_neighbors[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)

        sources = sorted(
            [(node, endpoint_colors[node]) for node in component if node in endpoint_colors],
            key=lambda item: (item[0][0], item[0][1], item[0][2], item[1]),
        )
        if not sources:
            continue

        dist: dict[tuple[int, int, str], int] = {}
        owner: dict[tuple[int, int, str], int] = {}
        bfs = deque()
        for idx, (node, _color) in enumerate(sources):
            if node in dist:
                continue
            dist[node] = 0
            owner[node] = idx
            bfs.append(node)

        while bfs:
            node = bfs.popleft()
            for nxt in node_neighbors[node]:
                if nxt in dist:
                    continue
                dist[nxt] = dist[node] + 1
                owner[nxt] = owner[node]
                bfs.append(nxt)

        for node, idx in owner.items():
            node_colors[node] = int(sources[idx][1])

    out = np.full_like(g, bg)
    for (r, c), info in cell_nodes.items():
        node = _visible_node(info)
        out[r, c] = int(node_colors.get(node, g[r, c]))
    return out


def _sample_spaced_positions(
    rng: random.Random, low: int, high: int, count: int, min_gap: int
) -> list[int] | None:
    if high < low:
        return None
    values = list(range(low, high + 1))
    if len(values) < count:
        return None
    for _ in range(200):
        picks = sorted(rng.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _palette(bg: int) -> list[int]:
    return [color for color in range(10) if color not in (bg, GREY, MAROON)]


def _set_endpoint_colors(
    route: dict, color_a: int, color_b: int
) -> None:
    cells = route["cells"]
    route["endpoint_colors"] = {
        cells[0]: int(color_a),
        cells[-1]: int(color_b),
    }


def _assign_route_colors(rng: random.Random, routes: list[dict], bg: int) -> None:
    colors = _palette(bg)
    same_idx = rng.randrange(len(routes))
    diff_candidates = [i for i, route in enumerate(routes) if len(route["cells"]) % 2 == 0 and i != same_idx]
    diff_idx = rng.choice(diff_candidates) if diff_candidates else None

    for i, route in enumerate(routes):
        if i == same_idx:
            color = rng.choice(colors)
            _set_endpoint_colors(route, color, color)
            continue
        if i == diff_idx or (len(route["cells"]) % 2 == 0 and rng.random() < 0.5):
            color_a, color_b = rng.sample(colors, 2)
            _set_endpoint_colors(route, color_a, color_b)
        else:
            color = rng.choice(colors)
            _set_endpoint_colors(route, color, color)


def _component_cells(component: dict) -> set[tuple[int, int]]:
    return set(component["cells"])


def _can_place_component(
    shape: tuple[int, int], cells: set[tuple[int, int]], occupied: set[tuple[int, int]]
) -> bool:
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in occupied:
            return False
    for r, c in cells:
        for dr, dc in CARDINAL_DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied and (nr, nc) not in cells:
                return False
    return True


def _make_plus(
    center: tuple[int, int],
    left_steps: int,
    right_steps: int,
    up_steps: int,
    down_steps: int,
    color: int,
) -> dict:
    r, c = center
    cells = {(r, c)}
    for step in range(1, left_steps + 1):
        cells.add((r, c - step))
    for step in range(1, right_steps + 1):
        cells.add((r, c + step))
    for step in range(1, up_steps + 1):
        cells.add((r - step, c))
    for step in range(1, down_steps + 1):
        cells.add((r + step, c))
    endpoints = {
        (r, c - left_steps): color,
        (r, c + right_steps): color,
        (r - up_steps, c): color,
        (r + down_steps, c): color,
    }
    return {
        "kind": "plus",
        "cells": sorted(cells),
        "endpoint_colors": endpoints,
    }


def _try_place_plus(
    rng: random.Random, shape: tuple[int, int], occupied: set[tuple[int, int]], bg: int
) -> dict | None:
    colors = _palette(bg)
    h, w = shape
    for _ in range(120):
        left_steps = rng.randint(2, 4)
        right_steps = rng.randint(2, 4)
        up_steps = rng.randint(2, 4)
        down_steps = rng.randint(2, 4)
        r_low, r_high = up_steps + 1, h - down_steps - 2
        c_low, c_high = left_steps + 1, w - right_steps - 2
        if r_low > r_high or c_low > c_high:
            continue
        center = (rng.randint(r_low, r_high), rng.randint(c_low, c_high))
        color = rng.choice(colors)
        plus = _make_plus(center, left_steps, right_steps, up_steps, down_steps, color)
        if _can_place_component(shape, _component_cells(plus), occupied):
            return plus
    return None


def _make_elbow(
    bend: tuple[int, int],
    dir_to_bend: tuple[int, int],
    dir_from_bend: tuple[int, int],
    len_to_bend: int,
    len_from_bend: int,
    color_a: int,
    color_b: int,
) -> dict:
    br, bc = bend
    sr = br - (len_to_bend - 1) * dir_to_bend[0]
    sc = bc - (len_to_bend - 1) * dir_to_bend[1]
    cells = []
    for step in range(len_to_bend):
        cells.append((sr + step * dir_to_bend[0], sc + step * dir_to_bend[1]))
    for step in range(1, len_from_bend):
        cells.append((br + step * dir_from_bend[0], bc + step * dir_from_bend[1]))
    return {
        "kind": "poly",
        "cells": cells,
        "endpoint_colors": {
            cells[0]: int(color_a),
            cells[-1]: int(color_b),
        },
    }


def _try_place_elbow(
    rng: random.Random, shape: tuple[int, int], occupied: set[tuple[int, int]], bg: int
) -> dict | None:
    colors = _palette(bg)
    options = [
        ((-1, 0), (0, 1)),
        ((-1, 0), (0, -1)),
        ((1, 0), (0, 1)),
        ((1, 0), (0, -1)),
        ((0, -1), (1, 0)),
        ((0, -1), (-1, 0)),
        ((0, 1), (1, 0)),
        ((0, 1), (-1, 0)),
    ]
    h, w = shape
    for _ in range(160):
        dir_to_bend, dir_from_bend = rng.choice(options)
        len_to_bend = rng.randint(3, 6)
        len_from_bend = rng.randint(3, 6)
        min_r = 1 + max(0, (len_to_bend - 1) * dir_to_bend[0], -(len_from_bend - 1) * dir_from_bend[0])
        max_r = h - 2 + min(0, (len_to_bend - 1) * dir_to_bend[0], -(len_from_bend - 1) * dir_from_bend[0])
        min_c = 1 + max(0, (len_to_bend - 1) * dir_to_bend[1], -(len_from_bend - 1) * dir_from_bend[1])
        max_c = w - 2 + min(0, (len_to_bend - 1) * dir_to_bend[1], -(len_from_bend - 1) * dir_from_bend[1])
        if min_r > max_r or min_c > max_c:
            continue
        bend = (rng.randint(min_r, max_r), rng.randint(min_c, max_c))
        if (len_to_bend + len_from_bend - 1) % 2 == 0 and rng.random() < 0.5:
            color_a, color_b = rng.sample(colors, 2)
        else:
            color_a = color_b = rng.choice(colors)
        elbow = _make_elbow(bend, dir_to_bend, dir_from_bend, len_to_bend, len_from_bend, color_a, color_b)
        if _can_place_component(shape, _component_cells(elbow), occupied):
            return elbow
    return None


def _render_grid(shape: tuple[int, int], bg: int, components: list[dict]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    occupants: dict[tuple[int, int], list[tuple[dict, int]]] = {}
    for component in components:
        for idx, cell in enumerate(component["cells"]):
            occupants.setdefault(cell, []).append((component, idx))

    for (r, c), items in occupants.items():
        if len(items) == 1:
            component, _idx = items[0]
            grid[r, c] = int(component["endpoint_colors"].get((r, c), GREY))
            continue

        kinds = {component["kind"] for component, _idx in items}
        if (
            len(items) == 2
            and kinds == {"h", "v"}
            and all((r, c) not in component["endpoint_colors"] for component, _idx in items)
        ):
            grid[r, c] = MAROON
            continue
        raise ValueError("unexpected overlap pattern")

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    bg = rng.choice(BG_CHOICES)

    top = rng.randint(1, max(2, h // 5))
    left = rng.randint(1, max(2, w // 5))
    bottom_choices = [r for r in range(max(top + 8, (3 * h) // 5), h - 1) if (r - top) % 2 == 1]
    right_choices = [c for c in range(max(left + 8, (3 * w) // 5), w - 1) if (c - left) % 2 == 1]
    if not bottom_choices or not right_choices:
        return None
    bottom = rng.choice(bottom_choices)
    right = rng.choice(right_choices)

    n_h = rng.randint(1, 2)
    n_v = rng.randint(1, 2)
    rows = _sample_spaced_positions(rng, top + 1, bottom - 1, n_h, 3)
    cols = _sample_spaced_positions(rng, left + 1, right - 1, n_v, 3)
    if rows is None or cols is None:
        return None

    routes: list[dict] = []
    for r in rows:
        routes.append(
            {
                "kind": "h",
                "cells": [(r, c) for c in range(left, right + 1)],
                "endpoint_colors": {},
            }
        )
    for c in cols:
        routes.append(
            {
                "kind": "v",
                "cells": [(r, c) for r in range(top, bottom + 1)],
                "endpoint_colors": {},
            }
        )
    _assign_route_colors(rng, routes, bg)

    components = list(routes)
    occupied = set().union(*(_component_cells(route) for route in routes))

    if rng.random() < 0.5:
        plus = _try_place_plus(rng, (h, w), occupied, bg)
        if plus is not None:
            components.append(plus)
            occupied |= _component_cells(plus)

    if rng.random() < 0.5:
        elbow = _try_place_elbow(rng, (h, w), occupied, bg)
        if elbow is not None:
            components.append(elbow)
            occupied |= _component_cells(elbow)

    grid = _render_grid((h, w), bg, components)
    out = main(grid)

    if np.array_equal(out, grid):
        return None
    if np.any(np.isin(out, (GREY, MAROON))):
        return None
    if not np.any(grid == MAROON):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid bridge-network input")
