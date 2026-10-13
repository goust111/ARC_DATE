import random
from typing import Iterable

import numpy as np


BG = 0
DIRS = ("h", "v", "d1", "d2")
MIN_LINE_COUNT = 3
MAX_LINE_COUNT = 4
MIN_SIDE = 8
MAX_SIDE = 15
MIN_LINE_LEN = 4
MAX_ATTEMPTS = 500


def _line_coords(shape: tuple[int, int], direction: str, key: int) -> list[tuple[int, int]]:
    h, w = shape
    coords: list[tuple[int, int]] = []

    if direction == "h":
        r = key
        coords = [(r, c) for c in range(w)]
    elif direction == "v":
        c = key
        coords = [(r, c) for r in range(h)]
    elif direction == "d1":
        d = key
        r = max(0, d)
        c = r - d
        while 0 <= r < h and 0 <= c < w:
            coords.append((r, c))
            r += 1
            c += 1
    elif direction == "d2":
        s = key
        r = max(0, s - (w - 1))
        c = s - r
        while 0 <= r < h and 0 <= c < w:
            coords.append((r, c))
            r += 1
            c -= 1
    else:
        raise AssertionError(f"unknown direction: {direction}")

    return coords


def _candidate_lines(shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    lines: list[dict] = []

    for r in range(h):
        coords = _line_coords(shape, "h", r)
        if len(coords) >= MIN_LINE_LEN:
            lines.append({"dir": "h", "key": r, "coords": coords})
    for c in range(w):
        coords = _line_coords(shape, "v", c)
        if len(coords) >= MIN_LINE_LEN:
            lines.append({"dir": "v", "key": c, "coords": coords})
    for d in range(-(w - 1), h):
        coords = _line_coords(shape, "d1", d)
        if len(coords) >= MIN_LINE_LEN:
            lines.append({"dir": "d1", "key": d, "coords": coords})
    for s in range(h + w - 1):
        coords = _line_coords(shape, "d2", s)
        if len(coords) >= MIN_LINE_LEN:
            lines.append({"dir": "d2", "key": s, "coords": coords})

    return lines


def _infer_line(shape: tuple[int, int], coords: np.ndarray) -> dict:
    rows = coords[:, 0]
    cols = coords[:, 1]
    candidates: list[dict] = []

    if np.all(rows == rows[0]):
        candidates.append({"dir": "h", "key": int(rows[0])})
    if np.all(cols == cols[0]):
        candidates.append({"dir": "v", "key": int(cols[0])})
    if np.all(rows - cols == rows[0] - cols[0]):
        candidates.append({"dir": "d1", "key": int(rows[0] - cols[0])})
    if np.all(rows + cols == rows[0] + cols[0]):
        candidates.append({"dir": "d2", "key": int(rows[0] + cols[0])})

    assert len(candidates) == 1, "each color must reveal a unique supporting line"
    candidate = candidates[0]
    full_coords = _line_coords(shape, candidate["dir"], candidate["key"])
    assert len(full_coords) >= len(coords)
    candidate["coords"] = full_coords
    return candidate


def _build_edges(lines: dict[int, dict], grid: np.ndarray) -> set[tuple[int, int]]:
    colors = sorted(lines)
    edges: set[tuple[int, int]] = set()

    for i, a in enumerate(colors):
        coords_a = set(lines[a]["coords"])
        for b in colors[i + 1 :]:
            inter = coords_a.intersection(lines[b]["coords"])
            assert len(inter) <= 1, "distinct valid lines can intersect at most once"
            if not inter:
                continue
            r, c = next(iter(inter))
            top = int(grid[r, c])
            assert top in (a, b), "crossing must show one of the two line colors"
            if top == a:
                edges.add((b, a))
            else:
                edges.add((a, b))

    return edges


def _unique_toposort(nodes: Iterable[int], edges: set[tuple[int, int]]) -> list[int]:
    order: list[int] = []
    nodes = list(nodes)
    children = {node: set() for node in nodes}
    indeg = {node: 0 for node in nodes}

    for u, v in edges:
        if v not in children[u]:
            children[u].add(v)
            indeg[v] += 1

    while indeg:
        zeros = sorted(node for node, deg in indeg.items() if deg == 0)
        assert len(zeros) == 1, "occlusion constraints must determine a unique order"
        node = zeros[0]
        order.append(node)
        for nxt in children[node]:
            indeg[nxt] -= 1
        del indeg[node]

    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each full straight line, derive pairwise occlusion constraints from
    visible crossing colors, and output the unique bottom-to-top color order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = sorted(int(color) for color in np.unique(g) if color != BG)
    assert colors, "expected at least one visible line color"

    lines: dict[int, dict] = {}
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) >= 2, "each line must retain at least two visible pixels"
        lines[color] = _infer_line(g.shape, coords)

    edges = _build_edges(lines, g)
    order = _unique_toposort(colors, edges)
    return np.array(order, dtype=int).reshape(-1, 1)


def _crosses(a: dict, b: dict) -> bool:
    return bool(set(a["coords"]).intersection(b["coords"]))


def _render(lines: list[dict], colors_in_order: list[int], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for line, color in zip(lines, colors_in_order):
        for r, c in line["coords"]:
            grid[r, c] = color
    return grid


def _expected_output(colors_in_order: list[int]) -> np.ndarray:
    return np.array(colors_in_order, dtype=int).reshape(-1, 1)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_SIDE, MAX_SIDE)
    w = rng.randint(MIN_SIDE, MAX_SIDE)
    shape = (h, w)
    all_lines = _candidate_lines(shape)
    by_dir = {direction: [line for line in all_lines if line["dir"] == direction] for direction in DIRS}

    line_count = rng.randint(MIN_LINE_COUNT, MAX_LINE_COUNT)
    dirs = rng.sample(DIRS, line_count)
    chosen = [dict(rng.choice(by_dir[direction])) for direction in dirs]
    rng.shuffle(chosen)

    if any(not _crosses(chosen[i], chosen[i + 1]) for i in range(line_count - 1)):
        return None

    palette = [color for color in range(1, 10)]
    colors_in_order = rng.sample(palette, line_count)
    grid = _render(chosen, colors_in_order, shape)

    visible_counts = {color: int(np.sum(grid == color)) for color in colors_in_order}
    if any(count < 2 for count in visible_counts.values()):
        return None

    expected = _expected_output(colors_in_order)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
