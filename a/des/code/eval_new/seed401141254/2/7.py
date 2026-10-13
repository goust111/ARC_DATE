from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

PAIR_COLOR = {
    RED: GREEN,
    GREEN: RED,
    BLUE: YELLOW,
    YELLOW: BLUE,
    ORANGE: MAGENTA,
    MAGENTA: ORANGE,
}

PAIR_GROUPS = ((RED, GREEN), (BLUE, YELLOW), (ORANGE, MAGENTA))
EXTRA_SOURCE_COLORS = (CYAN, MAROON)

ROTATIONS = (0, 1, 2, 3)
BASE_COMPONENT_KINDS = ("straight", "t_chain", "cross_t")
EXTRA_COMPONENT_KINDS = ("straight", "t_chain")

MAX_GENERATION_ATTEMPTS = 400


Coord = Tuple[int, int]


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[Coord]:
    if r > 0:
        yield (r - 1, c)
    if r + 1 < h:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < w:
        yield (r, c + 1)


def _find_separator(g: np.ndarray) -> int:
    candidates = [
        r
        for r in range(1, g.shape[0] - 1)
        if bool(np.all(g[r, :] == GREY)) and bool(np.all(g[r + 1 :, :] == BLACK))
    ]
    assert len(candidates) == 1, f"expected one separator row, got {candidates}"
    return int(candidates[0])


def _build_adjacency(signal: np.ndarray) -> Dict[Coord, List[Coord]]:
    h, w = signal.shape
    cells = [tuple(map(int, rc)) for rc in np.argwhere(signal != BLACK)]
    adjacency: Dict[Coord, List[Coord]] = {}
    for r, c in cells:
        adjacency[(r, c)] = [
            (rr, cc)
            for rr, cc in _neighbors4(r, c, h, w)
            if int(signal[rr, cc]) != BLACK
        ]
        deg = len(adjacency[(r, c)])
        assert 1 <= deg <= 4, f"invalid wire degree {deg} at {(r, c)}"
    return adjacency


def _is_node(signal: np.ndarray, adjacency: Dict[Coord, List[Coord]], cell: Coord) -> bool:
    if int(signal[cell]) != GREY:
        return False
    deg = len(adjacency[cell])
    if deg not in (3, 4):
        return False
    r, c = cell
    dirs = {(rr - r, cc - c) for rr, cc in adjacency[cell]}
    has_h = (0, -1) in dirs or (0, 1) in dirs
    has_v = (-1, 0) in dirs or (1, 0) in dirs
    return has_h and has_v


def _propagate_from_source(
    signal: np.ndarray,
    adjacency: Dict[Coord, List[Coord]],
    source: Coord,
) -> List[Tuple[int, int, int]]:
    source_color = int(signal[source])
    start_neighbors = adjacency[source]
    assert len(start_neighbors) == 1, f"source must be a free end: {source}"

    stack = [(source, start_neighbors[0], source_color, 0)]
    seen: set[Tuple[Coord, Coord, int]] = set()
    lights: List[Tuple[int, int, int]] = []
    while stack:
        prev, cur, color, node_count = stack.pop()
        state = (prev, cur, color)
        if state in seen:
            continue
        seen.add(state)

        deg = len(adjacency[cur])
        if deg == 1:
            lights.append((int(cur[1]), int(node_count), int(color)))
            continue

        next_cells = [nxt for nxt in adjacency[cur] if nxt != prev]
        assert next_cells, f"dead end inside wire at {cur}"

        if _is_node(signal, adjacency, cur):
            next_color = int(PAIR_COLOR.get(color, color)) if deg == 4 else int(color)
            next_count = int(node_count + 1)
            branches = next_cells
        else:
            assert len(next_cells) == 1, f"non-node wire must not branch at {cur}"
            next_color = int(color)
            next_count = int(node_count)
            branches = next_cells

        for nxt in branches:
            stack.append((cur, nxt, next_color, next_count))

    return lights


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored source pulses through the grey wire network and light the canvas row indexed by node count at each terminal column.

    Input structure:
    - The background is black, and one full grey row separates an upper signal area from a lower canvas area.
    - In the signal area, grey pixels form one-cell-wide horizontal and vertical wires, while some wire free ends are replaced by unique colored source pixels.
    - Other wire free ends may stay grey and act as terminals; the canvas area below the separator is initially all black.

    Rule:
    - From each colored source, propagate a pulse through the connected wire without recoloring the wire itself.
    - When the pulse reaches a T-junction (degree 3), it splits into every forward direction and keeps its color; when it reaches a cross-junction (degree 4), it also splits forward but swaps to its fixed paired color if one exists.
    - Whenever a pulse arrives at a different free end, light the canvas pixel in the same column and in the row equal to the number of traversed junctions so far, keeping the signal area unchanged.

    Key insight:
    - The output is encoded by graph traversal state: wire topology controls where pulses can branch, cross-junctions alone cause color flips, and the number of visited junctions becomes the canvas row index.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, f"grid exceeds ARC bounds: {g.shape}"

    sep = _find_separator(g)
    signal = g[:sep, :]
    canvas = g[sep + 1 :, :]
    assert signal.size > 0 and canvas.size > 0, "expected nonempty signal and canvas areas"
    assert bool(np.all(canvas == BLACK)), "canvas must start blank"

    adjacency = _build_adjacency(signal)
    assert adjacency, "expected at least one wire component"

    source_cells = [
        cell
        for cell in sorted(adjacency)
        if int(signal[cell]) not in (BLACK, GREY)
    ]
    assert source_cells, "expected colored sources"
    source_colors = [int(signal[cell]) for cell in source_cells]
    assert len(set(source_colors)) == len(source_colors), "source colors must be distinct"
    for cell in source_cells:
        assert len(adjacency[cell]) == 1, f"source must sit on a free end: {cell}"

    out = g.copy()
    canvas_top = sep + 1
    for source in source_cells:
        for col, row, color in _propagate_from_source(signal, adjacency, source):
            assert 0 <= row < canvas.shape[0], f"node-count row {row} exceeds canvas height {canvas.shape[0]}"
            target = (canvas_top + row, col)
            existing = int(out[target])
            assert existing in (BLACK, color), f"conflicting canvas writes at {target}: {existing} vs {color}"
            out[target] = int(color)

    return out


def _normalize_points(points: Dict[str, Coord]) -> Dict[str, Coord]:
    min_r = min(r for r, _c in points.values())
    min_c = min(c for _r, c in points.values())
    return {name: (r - min_r, c - min_c) for name, (r, c) in points.items()}


def _rotate_points(points: Dict[str, Coord], turns: int) -> Dict[str, Coord]:
    rotated = _normalize_points(points)
    for _ in range(turns % 4):
        h = max(r for r, _c in rotated.values()) + 1
        rotated = {name: (c, h - 1 - r) for name, (r, c) in rotated.items()}
        rotated = _normalize_points(rotated)
    return rotated


def _segment_cells(a: Coord, b: Coord) -> List[Coord]:
    r0, c0 = a
    r1, c1 = b
    assert r0 == r1 or c0 == c1, f"edge must be axis-aligned: {a} -> {b}"
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    lo, hi = sorted((r0, r1))
    return [(r, c0) for r in range(lo, hi + 1)]


def _render_component(
    points: Dict[str, Coord],
    edges: List[Tuple[str, str]],
    source_name: str,
    source_color: int,
) -> np.ndarray:
    coords = set()
    for a_name, b_name in edges:
        coords.update(_segment_cells(points[a_name], points[b_name]))
    for coord in points.values():
        coords.add(coord)

    h = max(r for r, _c in coords) + 1
    w = max(c for _r, c in coords) + 1
    patch = np.full((h, w), BLACK, dtype=int)
    for r, c in coords:
        patch[r, c] = GREY
    sr, sc = points[source_name]
    patch[sr, sc] = int(source_color)
    return patch


def _straight_component(source_color: int) -> np.ndarray:
    length = random.randint(3, 6)
    points = {"S": (0, 0), "T": (0, length)}
    points = _rotate_points(points, random.choice(ROTATIONS))
    return _render_component(points, [("S", "T")], "S", source_color)


def _t_chain_component(source_color: int) -> np.ndarray:
    first_run = random.randint(2, 4)
    second_run = random.randint(2, 4)
    end_run = random.randint(2, 4)
    branch1 = random.randint(2, 4)
    branch2 = random.randint(2, 4)
    dir1 = random.choice((-1, 1))
    dir2 = random.choice((-1, 1))
    has_second = random.random() < 0.7

    points: Dict[str, Coord] = {
        "S": (0, 0),
        "N1": (0, first_run),
        "B1": (dir1 * branch1, first_run),
    }
    edges: List[Tuple[str, str]] = [("S", "N1"), ("N1", "B1")]
    if has_second:
        n2_col = first_run + second_run
        end_col = n2_col + end_run
        points["N2"] = (0, n2_col)
        points["B2"] = (dir2 * branch2, n2_col)
        points["R"] = (0, end_col)
        edges.extend([("N1", "N2"), ("N2", "B2"), ("N2", "R")])
    else:
        points["R"] = (0, first_run + second_run + end_run)
        edges.append(("N1", "R"))

    points = _rotate_points(points, random.choice(ROTATIONS))
    return _render_component(points, edges, "S", source_color)


def _cross_t_component(source_color: int) -> np.ndarray:
    source_run = random.randint(2, 4)
    right_run = random.randint(2, 4)
    up_run = random.randint(2, 4)
    down_to_t = random.randint(2, 3)
    down_run = random.randint(2, 4)
    side_run = random.randint(2, 4)
    side_dir = random.choice((-1, 1))

    cross_col = source_run
    t_row = down_to_t
    points = {
        "S": (0, 0),
        "X": (0, cross_col),
        "R": (0, cross_col + right_run),
        "U": (-up_run, cross_col),
        "T": (t_row, cross_col),
        "D": (t_row + down_run, cross_col),
        "H": (t_row, cross_col + side_dir * side_run),
    }
    edges = [
        ("S", "X"),
        ("X", "R"),
        ("U", "X"),
        ("X", "T"),
        ("T", "D"),
        ("T", "H"),
    ]
    points = _rotate_points(points, random.choice(ROTATIONS))
    return _render_component(points, edges, "S", source_color)


def _build_component(kind: str, source_color: int) -> np.ndarray:
    if kind == "straight":
        return _straight_component(source_color)
    if kind == "t_chain":
        return _t_chain_component(source_color)
    if kind == "cross_t":
        return _cross_t_component(source_color)
    raise ValueError(f"unknown component kind: {kind}")


def _choose_component_specs() -> List[Tuple[str, int]]:
    pair_colors = [random.choice(group) for group in PAIR_GROUPS]
    random.shuffle(pair_colors)
    specs = list(zip(BASE_COMPONENT_KINDS, pair_colors))
    if random.random() < 0.5:
        specs.append((random.choice(EXTRA_COMPONENT_KINDS), random.choice(EXTRA_SOURCE_COLORS)))
    random.shuffle(specs)
    return specs


def _assemble_signal_area(components: List[np.ndarray]) -> np.ndarray | None:
    widths = [patch.shape[1] for patch in components]
    heights = [patch.shape[0] for patch in components]

    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    gaps = [random.randint(2, 4) for _ in range(len(components) - 1)]
    width = left_margin + right_margin + sum(widths) + sum(gaps)
    if width > 30:
        return None

    min_signal_h = max(heights) + 2
    max_signal_h = min(14, 29 - 4)
    if min_signal_h > max_signal_h:
        return None
    signal_h = random.randint(min_signal_h, max_signal_h)

    canvas_h = random.randint(4, min(8, 29 - signal_h))
    if signal_h + 1 + canvas_h > 30:
        return None

    signal = np.full((signal_h, width), BLACK, dtype=int)
    x = left_margin
    for i, patch in enumerate(components):
        h, w = patch.shape
        top = random.randint(1, signal_h - h - 1)
        mask = patch != BLACK
        region = signal[top : top + h, x : x + w]
        region[mask] = patch[mask]
        x += w
        if i < len(gaps):
            x += gaps[i]

    full = np.full((signal_h + 1 + canvas_h, width), BLACK, dtype=int)
    full[:signal_h, :] = signal
    full[signal_h, :] = GREY
    return full


def _valid_generated_sample(inp: np.ndarray) -> bool:
    sep = _find_separator(inp)
    signal = inp[:sep, :]
    source_colors = {
        int(signal[r, c])
        for r, c in np.argwhere((signal != BLACK) & (signal != GREY))
    }
    if len(source_colors) < 3:
        return False

    out = main(inp)
    canvas = out[sep + 1 :, :]
    lit_rows = {int(r) for r, _c in np.argwhere(canvas != BLACK)}
    lit_colors = {int(v) for v in np.unique(canvas) if int(v) != BLACK}
    if np.count_nonzero(canvas != BLACK) < 5:
        return False
    if not {0, 1, 2}.issubset(lit_rows):
        return False
    if not (lit_colors - source_colors):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _choose_component_specs()
    components = [_build_component(kind, color) for kind, color in specs]
    grid = _assemble_signal_area(components)
    if grid is None:
        return None
    if not _valid_generated_sample(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-network ARC input")
