from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
PIPE = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SCALE_CHOICES = (4, 5)
PADDING_RANGE = (0, 4)
MAX_ATTEMPTS = 400
TRANSFORMS = (
    ("id", False),
    ("rot90", True),
    ("rot180", False),
    ("rot270", True),
    ("flip_h", False),
    ("flip_v", False),
    ("diag", True),
    ("anti", True),
)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _dir_index(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    for idx, (rr, cc) in enumerate(DIRS):
        if (dr, dc) == (rr, cc):
            return idx
    raise ValueError(f"cells are not 4-neighbors: {a} -> {b}")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _render_output(
    shape: tuple[int, int],
    seed_colors: dict[tuple[int, int], int],
    neighbor_map: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for pos, color in seed_colors.items():
        out[pos] = int(color)

    h, w = shape
    for pos, entries in neighbor_map.items():
        ordered = sorted(entries, key=lambda item: int(item[0]))
        for i, (_port_dir, other) in enumerate(ordered):
            rr = int(pos[0] + DIRS[i][0])
            cc = int(pos[1] + DIRS[i][1])
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(seed_colors[other])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    seeds = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere((g != BG) & (g != PIPE))
    }
    neighbor_map: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {
        pos: [] for pos in seeds
    }

    for comp in _components(g == PIPE):
        adjacent: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        for cell in comp:
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in seeds:
                    adjacent[(nr, nc)].add(cell)

        if len(adjacent) != 2:
            raise AssertionError(f"expected each pipe to touch exactly two seeds, got {adjacent}")

        items = list(adjacent.items())
        (seed_a, cells_a), (seed_b, cells_b) = items[0], items[1]
        if len(cells_a) != 1 or len(cells_b) != 1:
            raise AssertionError("pipe endpoint must touch each seed at exactly one side")

        pipe_a = next(iter(cells_a))
        pipe_b = next(iter(cells_b))
        dir_a = _dir_index(seed_a, pipe_a)
        dir_b = _dir_index(seed_b, pipe_b)
        neighbor_map[seed_a].append((dir_a, seed_b))
        neighbor_map[seed_b].append((dir_b, seed_a))

    return _render_output(g.shape, seeds, neighbor_map)


FAMILIES = {
    "pair_straight": {
        "nodes": {0: (0, 0), 1: (0, 2)},
        "edges": [(0, 1, None)],
    },
    "pair_L": {
        "nodes": {0: (0, 0), 1: (2, 2)},
        "edges": [(0, 1, "either")],
    },
    "chain3": {
        "nodes": {0: (0, 0), 1: (2, 2), 2: (4, 2)},
        "edges": [(0, 1, "vertical"), (1, 2, None)],
    },
    "chain4": {
        "nodes": {0: (0, 2), 1: (2, 2), 2: (2, 4), 3: (4, 4)},
        "edges": [(0, 1, None), (1, 2, None), (2, 3, None)],
    },
    "star3": {
        "nodes": {0: (2, 2), 1: (0, 0), 2: (0, 4), 3: (4, 2)},
        "edges": [(0, 1, "vertical"), (0, 2, "horizontal"), (0, 3, None)],
    },
    "star4": {
        "nodes": {0: (2, 2), 1: (0, 0), 2: (0, 4), 3: (4, 4), 4: (4, 0)},
        "edges": [
            (0, 1, "vertical"),
            (0, 2, "horizontal"),
            (0, 3, "vertical"),
            (0, 4, "horizontal"),
        ],
    },
    "cycle4": {
        "nodes": {0: (0, 0), 1: (0, 4), 2: (4, 4), 3: (4, 0)},
        "edges": [(0, 1, None), (1, 2, None), (2, 3, None), (3, 0, None)],
    },
    "tadpole5": {
        "nodes": {0: (2, 2), 1: (0, 2), 2: (2, 4), 3: (4, 4), 4: (4, 0)},
        "edges": [(0, 1, None), (0, 2, None), (2, 3, None), (0, 4, "horizontal")],
    },
}

BRANCH_BUCKETS = {
    1: ("pair_straight", "pair_L"),
    2: ("chain3", "chain4", "cycle4"),
    3: ("star3", "tadpole5"),
    4: ("star4",),
}


def _transform_point(pt: tuple[int, int], name: str, span: int) -> tuple[int, int]:
    r, c = pt
    last = span - 1
    if name == "id":
        return r, c
    if name == "rot90":
        return c, last - r
    if name == "rot180":
        return last - r, last - c
    if name == "rot270":
        return last - c, r
    if name == "flip_h":
        return r, last - c
    if name == "flip_v":
        return last - r, c
    if name == "diag":
        return c, r
    if name == "anti":
        return last - c, last - r
    raise ValueError(name)


def _swap_style(style: str | None, swap_axes: bool) -> str | None:
    if style not in {"vertical", "horizontal"} or not swap_axes:
        return style
    return "horizontal" if style == "vertical" else "vertical"


def _route_edge(
    a: tuple[int, int],
    b: tuple[int, int],
    style: str | None,
) -> tuple[list[tuple[int, int]], int, int]:
    ra, ca = a
    rb, cb = b

    if ra == rb:
        step = 1 if cb > ca else -1
        path = [(ra, c) for c in range(ca + step, cb, step)]
        return path, (1 if step > 0 else 3), (3 if step > 0 else 1)

    if ca == cb:
        step = 1 if rb > ra else -1
        path = [(r, ca) for r in range(ra + step, rb, step)]
        return path, (2 if step > 0 else 0), (0 if step > 0 else 2)

    if style not in {"vertical", "horizontal"}:
        raise ValueError("diagonal edges need an L-shape style")

    if style == "vertical":
        dr = 1 if rb > ra else -1
        port_a = (ra + dr, ca)
        port_b = (rb - dr, cb)
        bend = (port_b[0], port_a[1])
        path: list[tuple[int, int]] = []
        for r in range(port_a[0], bend[0] + dr, dr):
            path.append((r, port_a[1]))
        dc = 1 if port_b[1] > bend[1] else -1
        for c in range(bend[1] + dc, port_b[1] + dc, dc):
            path.append((bend[0], c))
        return path, (2 if dr > 0 else 0), (0 if dr > 0 else 2)

    dc = 1 if cb > ca else -1
    port_a = (ra, ca + dc)
    port_b = (rb, cb - dc)
    bend = (port_a[0], port_b[1])
    path = []
    for c in range(port_a[1], bend[1] + dc, dc):
        path.append((port_a[0], c))
    dr = 1 if port_b[0] > bend[0] else -1
    for r in range(bend[0] + dr, port_b[0] + dr, dr):
        path.append((r, bend[1]))
    return path, (1 if dc > 0 else 3), (3 if dc > 0 else 1)


def _sample_family() -> tuple[dict[int, tuple[int, int]], list[tuple[int, int, str | None]]]:
    bucket = random.choice((1, 2, 3, 4))
    family = FAMILIES[random.choice(BRANCH_BUCKETS[bucket])]
    nodes = dict(family["nodes"])

    edges: list[tuple[int, int, str | None]] = []
    for u, v, style in family["edges"]:
        if style == "either":
            style = random.choice(("vertical", "horizontal"))
        edges.append((u, v, style))

    transform_name, swap_axes = random.choice(TRANSFORMS)
    span = 5
    nodes = {idx: _transform_point(pt, transform_name, span) for idx, pt in nodes.items()}
    edges = [(u, v, _swap_style(style, swap_axes)) for u, v, style in edges]

    min_r = min(r for r, _c in nodes.values())
    min_c = min(c for _r, c in nodes.values())
    nodes = {idx: (r - min_r, c - min_c) for idx, (r, c) in nodes.items()}
    return nodes, edges


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    nodes, edges = _sample_family()
    scale = random.choice(SCALE_CHOICES)
    pad_top = random.randint(*PADDING_RANGE)
    pad_bottom = random.randint(*PADDING_RANGE)
    pad_left = random.randint(*PADDING_RANGE)
    pad_right = random.randint(*PADDING_RANGE)

    max_r = max(r for r, _c in nodes.values())
    max_c = max(c for _r, c in nodes.values())
    height = pad_top + pad_bottom + scale * max_r + 5
    width = pad_left + pad_right + scale * max_c + 5

    positions = {
        idx: (pad_top + 2 + scale * r, pad_left + 2 + scale * c)
        for idx, (r, c) in nodes.items()
    }
    seed_set = set(positions.values())
    colors = random.sample(PALETTE, k=len(positions))
    seed_colors_by_node = {idx: int(color) for idx, color in zip(sorted(positions), colors)}
    seed_colors_by_pos = {positions[idx]: seed_colors_by_node[idx] for idx in positions}
    neighbor_map_by_node: dict[int, list[tuple[int, int]]] = {idx: [] for idx in positions}

    grid = np.full((height, width), BG, dtype=int)
    for idx, (r, c) in positions.items():
        grid[r, c] = seed_colors_by_node[idx]

    occupied_pipe: set[tuple[int, int]] = set()
    used_dirs: dict[int, set[int]] = {idx: set() for idx in positions}

    for u, v, style in edges:
        path, dir_u, dir_v = _route_edge(positions[u], positions[v], style)
        if not path:
            raise ValueError("pipe path must contain at least one grey cell")
        if dir_u in used_dirs[u] or dir_v in used_dirs[v]:
            raise ValueError("seed port reused")

        end_touch_u = sum(abs(r - positions[u][0]) + abs(c - positions[u][1]) == 1 for r, c in path)
        end_touch_v = sum(abs(r - positions[v][0]) + abs(c - positions[v][1]) == 1 for r, c in path)
        if end_touch_u != 1 or end_touch_v != 1:
            raise ValueError("pipe must touch each endpoint seed exactly once")

        path_set = set(path)
        for cell in path:
            if cell in seed_set or cell in occupied_pipe:
                raise ValueError("pipe overlap")
            r, c = cell
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in occupied_pipe:
                    raise ValueError("pipes must stay disconnected")
                if (nr, nc) in seed_set and (nr, nc) not in {positions[u], positions[v]}:
                    raise ValueError("pipe may not touch a third seed")
            grid[r, c] = PIPE

        occupied_pipe.update(path_set)
        used_dirs[u].add(dir_u)
        used_dirs[v].add(dir_v)
        neighbor_map_by_node[u].append((dir_u, v))
        neighbor_map_by_node[v].append((dir_v, u))

    for idx, dirs in used_dirs.items():
        if not (1 <= len(dirs) <= 4):
            raise ValueError(f"invalid degree for node {idx}: {dirs}")

    neighbor_map_by_pos = {
        positions[idx]: [(port_dir, positions[other]) for port_dir, other in neighbor_map_by_node[idx]]
        for idx in positions
    }
    expected = _render_output(grid.shape, seed_colors_by_pos, neighbor_map_by_pos)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, expected = _build_instance()
            if np.array_equal(main(grid), expected):
                return grid
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
