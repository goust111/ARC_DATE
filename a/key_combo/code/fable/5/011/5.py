from __future__ import annotations

from collections import Counter, defaultdict
from math import lcm
import random

import numpy as np


Coord = tuple[int, int]


def _neighbors4(r: int, c: int):
    """Yield 4-neighbors in the common ARC (row, column) convention."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    h, w = mask.shape
    unseen = mask.astype(bool).copy()
    result: list[np.ndarray] = []
    for r0, c0 in np.argwhere(unseen):
        r0, c0 = int(r0), int(c0)
        if not unseen[r0, c0]:
            continue
        unseen[r0, c0] = False
        todo = [(r0, c0)]
        coords: list[Coord] = []
        while todo:
            r, c = todo.pop()
            coords.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and unseen[rr, cc]:
                    unseen[rr, cc] = False
                    todo.append((rr, cc))
        result.append(np.asarray(coords, dtype=int))
    return result


def _solid_square_components(grid: np.ndarray, bg: int) -> list[dict]:
    """Find isolated monochrome solid squares and their inclusive bboxes."""
    squares: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            height, width = r1 - r0 + 1, c1 - c0 + 1
            if height != width or coords.shape[0] != height * width:
                continue
            if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                squares.append(
                    {
                        "color": color,
                        "size": height,
                        "bbox": (r0, c0, r1, c1),
                        "coords": coords,
                    }
                )
    return squares


def _simple_paths(adjacency: dict[int, list[int]], start: int, goal: int, limit: int) -> list[list[int]]:
    """Enumerate node-simple paths in a small graph."""
    paths: list[list[int]] = []

    def visit(node: int, path: list[int], used: set[int]) -> None:
        if len(paths) > limit:
            return
        if node == goal:
            paths.append(path.copy())
            return
        for nxt in adjacency.get(node, []):
            if nxt not in used:
                used.add(nxt)
                path.append(nxt)
                visit(nxt, path, used)
                path.pop()
                used.remove(nxt)

    visit(start, [start], {start})
    assert len(paths) <= limit, "graph has too many simple paths for this ARC task"
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the endpoint path whose node count equals the color-legend length, then color its nodes in legend order and its tracks grey.

    Input structure:
    - The dominant color is background. Solid equal-sized squares are graph nodes and one-cell-wide segments are graph edges.
    - Two endpoint nodes have special colors; all other graph nodes share one station color.
    - A corner contains a horizontal row of separated 2x2 legend blocks; its first and last colors equal the two endpoint colors.

    Rule:
    - Enumerate simple graph paths between the endpoint nodes and select the unique path containing exactly as many nodes as the legend has blocks.
    - Orient that path from the endpoint matching the legend's first color, recolor its ith node with the ith legend color, and recolor precisely its edge segments grey.
    - Preserve the legend, every off-path node, every off-path track, and the background.

    Key insight:
    - The legend carries two indispensable graph instructions: its length chooses a non-shortest path, while its endpoint colors determine the path direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    h, w = g.shape
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    squares = _solid_square_components(g, bg)

    # Legend blocks are the only 2x2 solid components and form one horizontal row.
    legend_blocks = [sq for sq in squares if int(sq["size"]) == 2]
    assert len(legend_blocks) >= 4, "expected at least four 2x2 legend blocks"
    legend_rows = {int(sq["bbox"][0]) for sq in legend_blocks}
    assert len(legend_rows) == 1, "legend blocks must share a row"
    legend_blocks.sort(key=lambda sq: int(sq["bbox"][1]))
    legend = [int(sq["color"]) for sq in legend_blocks]
    assert legend[0] != legend[-1], "endpoint colors must orient the selected path uniquely"

    # Infer the common node-square size without relying on any color id.
    node_candidates = [sq for sq in squares if int(sq["size"]) >= 3]
    size_counts = Counter(int(sq["size"]) for sq in node_candidates)
    assert size_counts, "expected solid graph-node squares"
    node_size, node_count = max(size_counts.items(), key=lambda item: (item[1], item[0]))
    nodes_raw = [sq for sq in node_candidates if int(sq["size"]) == int(node_size)]
    assert len(nodes_raw) == node_count >= len(legend), "insufficient graph nodes"

    nodes: list[dict] = []
    pixel_to_node: dict[Coord, int] = {}
    occupied = np.zeros((h, w), dtype=bool)
    for node_id, sq in enumerate(nodes_raw):
        coords = [(int(r), int(c)) for r, c in sq["coords"]]
        nodes.append({"color": int(sq["color"]), "coords": coords})
        for coord in coords:
            assert coord not in pixel_to_node
            pixel_to_node[coord] = node_id
            occupied[coord] = True

    for sq in legend_blocks:
        for r, c in sq["coords"]:
            occupied[int(r), int(c)] = True

    starts = [i for i, node in enumerate(nodes) if int(node["color"]) == legend[0]]
    goals = [i for i, node in enumerate(nodes) if int(node["color"]) == legend[-1]]
    assert len(starts) == len(goals) == 1, "legend endpoint colors must identify exactly one node each"
    start, goal = starts[0], goals[0]
    assert start != goal

    # Everything foreground outside the square objects must be track. Each
    # 4-connected track component is one edge touching exactly two nodes.
    track_mask = (g != bg) & ~occupied
    track_components = _components(track_mask)
    assert track_components, "expected graph tracks"
    adjacency_sets: dict[int, set[int]] = defaultdict(set)
    edge_pixels: dict[frozenset[int], list[Coord]] = {}
    track_color: int | None = None
    for coords in track_components:
        colors = {int(g[int(r), int(c)]) for r, c in coords}
        assert len(colors) == 1, "a track segment must be monochrome"
        component_color = next(iter(colors))
        if track_color is None:
            track_color = component_color
        assert component_color == track_color, "all graph tracks must share one color"

        touched: set[int] = set()
        pixels: list[Coord] = []
        for r, c in coords:
            r, c = int(r), int(c)
            pixels.append((r, c))
            for rr, cc in _neighbors4(r, c):
                node_id = pixel_to_node.get((rr, cc))
                if node_id is not None:
                    touched.add(node_id)
        assert len(touched) == 2, "each track segment must connect exactly two node squares"
        a, b = sorted(touched)
        key = frozenset((a, b))
        assert key not in edge_pixels, "parallel edge segments would make path selection ambiguous"
        edge_pixels[key] = pixels
        adjacency_sets[a].add(b)
        adjacency_sets[b].add(a)

    adjacency = {node: sorted(neighbors) for node, neighbors in adjacency_sets.items()}
    all_paths = _simple_paths(adjacency, start, goal, limit=256)
    matching = [path for path in all_paths if len(path) == len(legend)]
    assert len(matching) == 1, "legend length must select exactly one endpoint path"
    selected = matching[0]

    out = g.copy()
    for color, node_id in zip(legend, selected):
        for r, c in nodes[node_id]["coords"]:
            out[r, c] = color
    for a, b in zip(selected, selected[1:]):
        for r, c in edge_pixels[frozenset((a, b))]:
            out[r, c] = 5
    return out


# ------------------------------ generator ------------------------------

_NODE_SIZE = 3
_GREY = 5
_MAX_ATTEMPTS = 200


def _draw_track(canvas: np.ndarray, a: Coord, b: Coord, color: int) -> list[Coord]:
    """Draw a straight center-to-center segment between aligned node bboxes."""
    n = _NODE_SIZE
    ra, ca = a
    rb, cb = b
    pixels: list[Coord] = []
    if ra == rb:
        row = ra + n // 2
        lo, hi = sorted((ca, cb))
        pixels = [(row, c) for c in range(lo + n, hi)]
    elif ca == cb:
        col = ca + n // 2
        lo, hi = sorted((ra, rb))
        pixels = [(r, col) for r in range(lo + n, hi)]
    else:
        raise ValueError("generator edges must be axis-aligned")
    assert pixels, "nodes require a visible gap for their track"
    for r, c in pixels:
        assert int(canvas[r, c]) == -1, "generated track segments must not overlap"
        canvas[r, c] = color
    return pixels


def _paint_node(canvas: np.ndarray, pos: Coord, color: int) -> list[Coord]:
    r0, c0 = pos
    coords = [
        (r, c)
        for r in range(r0, r0 + _NODE_SIZE)
        for c in range(c0, c0 + _NODE_SIZE)
    ]
    assert all(int(canvas[r, c]) == -1 for r, c in coords)
    canvas[r0 : r0 + _NODE_SIZE, c0 : c0 + _NODE_SIZE] = color
    return coords


def _build_patch(legend: list[int], station: int, track: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a three-route graph and an independent expected-output patch."""
    k = len(legend)
    # The second outer route is randomly one node shorter or longer than the
    # target. Thus selecting by fixed rank (always longest / always middle) fails.
    other_length = random.choice((k - 1, k + 1))
    target_intervals = k - 3
    other_intervals = other_length - 3
    common = lcm(target_intervals, other_intervals)
    min_multiplier = (
        (_NODE_SIZE + 2) * max(target_intervals, other_intervals) + common - 1
    ) // common
    max_multiplier = 24 // common
    multiplier = random.randint(min_multiplier, min(max_multiplier, min_multiplier + 1))
    span = common * multiplier
    vertical_step = _NODE_SIZE + random.randint(2, 3)
    patch_h = 2 * vertical_step + _NODE_SIZE
    patch_w = span + _NODE_SIZE
    inp = np.full((patch_h, patch_w), -1, dtype=int)

    target_on_bottom = bool(random.getrandbits(1))
    target_row = 2 * vertical_step if target_on_bottom else 0
    other_row = 0 if target_on_bottom else 2 * vertical_step
    middle_row = vertical_step

    start, goal = 0, k - 1
    positions: dict[int, Coord] = {
        start: (middle_row, 0),
        goal: (middle_row, span),
    }
    # The target has k-2 internal nodes, including one below/above each endpoint.
    for i in range(k - 2):
        positions[i + 1] = (target_row, i * span // target_intervals)

    other_ids = list(range(k, k + other_length - 2))
    for i, node_id in enumerate(other_ids):
        positions[node_id] = (other_row, i * span // other_intervals)

    short_id = k + other_length - 2
    positions[short_id] = (middle_row, span // 2)

    target_path = [start] + list(range(1, k - 1)) + [goal]
    other_path = [start] + other_ids + [goal]
    decoy_path = [start, short_id, goal]
    target_edges = list(zip(target_path, target_path[1:]))
    other_edges = list(zip(other_path, other_path[1:]))
    decoy_edges = list(zip(decoy_path, decoy_path[1:]))
    edges = target_edges + other_edges + decoy_edges

    edge_pixels: dict[frozenset[int], list[Coord]] = {}
    for a, b in edges:
        edge_pixels[frozenset((a, b))] = _draw_track(inp, positions[a], positions[b], track)

    node_pixels: dict[int, list[Coord]] = {}
    for node_id, pos in positions.items():
        color = legend[0] if node_id == start else legend[-1] if node_id == goal else station
        node_pixels[node_id] = _paint_node(inp, pos, color)

    expected = inp.copy()
    for color, node_id in zip(legend, target_path):
        for r, c in node_pixels[node_id]:
            expected[r, c] = color
    for edge in target_edges:
        for r, c in edge_pixels[frozenset(edge)]:
            expected[r, c] = _GREY

    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    paths = _simple_paths(adjacency, start, goal, limit=10)

    metadata = {
        "paths": paths,
        "target_path": target_path,
        "other_path": other_path,
        "decoy_path": decoy_path,
        "target_edges": target_edges,
        "other_edges": other_edges,
        "decoy_edges": decoy_edges,
        "node_pixels": node_pixels,
        "edge_pixels": edge_pixels,
        "legend": legend,
        "station": station,
        "track": track,
    }
    return inp, expected, metadata


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    """Reject self-consistent but semantically degenerate generated puzzles."""
    legend = meta["legend"]
    paths = meta["paths"]
    target = meta["target_path"]
    other = meta["other_path"]
    decoy = meta["decoy_path"]
    node_pixels = meta["node_pixels"]
    edge_pixels = meta["edge_pixels"]

    # Exactly three alternatives exist; length uniquely selects the target,
    # while a direct route is shorter and the other outer route alternates
    # between shorter and longer than the target.
    if len(paths) != 3:
        return False
    if sum(len(path) == len(legend) for path in paths) != 1:
        return False
    if target not in paths or other not in paths or decoy not in paths:
        return False
    if not len(decoy) < min(len(target), len(other)):
        return False
    if abs(len(other) - len(target)) != 1:
        return False

    # Direction cannot be ignored: all legend colors are distinct, so reversal
    # changes every corresponding node assignment (including the endpoints).
    reversed_rule = inp.copy()
    for color, node_id in zip(reversed(legend), target):
        for r, c in node_pixels[node_id]:
            reversed_rule[r, c] = color
    for edge in meta["target_edges"]:
        for r, c in edge_pixels[frozenset(edge)]:
            reversed_rule[r, c] = _GREY
    if np.array_equal(reversed_rule, expected):
        return False

    # Neither recoloring layer is cosmetic.
    nodes_only = inp.copy()
    for color, node_id in zip(legend, target):
        for r, c in node_pixels[node_id]:
            nodes_only[r, c] = color
    tracks_only = inp.copy()
    for edge in meta["target_edges"]:
        for r, c in edge_pixels[frozenset(edge)]:
            tracks_only[r, c] = _GREY
    if np.array_equal(nodes_only, expected) or np.array_equal(tracks_only, expected):
        return False

    # A shortest-path shortcut must visibly alter the wrong branch, and both
    # off-path station and track colors survive in the true output.
    shortest_wrong = inp.copy()
    for color, node_id in zip(legend, decoy):
        for r, c in node_pixels[node_id]:
            shortest_wrong[r, c] = color
    for edge in meta["decoy_edges"]:
        for r, c in edge_pixels[frozenset(edge)]:
            shortest_wrong[r, c] = _GREY
    if np.array_equal(shortest_wrong, expected):
        return False

    fixed_rank_wrong = inp.copy()
    for color, node_id in zip(legend, other):
        for r, c in node_pixels[node_id]:
            fixed_rank_wrong[r, c] = color
    for edge in meta["other_edges"]:
        for r, c in edge_pixels[frozenset(edge)]:
            fixed_rank_wrong[r, c] = _GREY
    if np.array_equal(fixed_rank_wrong, expected):
        return False

    off_path_nodes = decoy[1:-1] + other[1:-1]
    off_path_edges = meta["decoy_edges"] + meta["other_edges"]
    if not any(expected[r, c] == meta["station"] for node in off_path_nodes for r, c in node_pixels[node]):
        return False
    if not any(expected[r, c] == meta["track"] for edge in off_path_edges for r, c in edge_pixels[frozenset(edge)]):
        return False
    return not np.array_equal(inp, expected)


def _transform(array: np.ndarray, turns: int, mirror: bool) -> np.ndarray:
    out = np.rot90(array, turns)
    if mirror:
        out = np.fliplr(out)
    return out.copy()


def _gen_attempt() -> np.ndarray | None:
    k = random.choice((5, 6))

    # Grey is reserved for selected output tracks. Every visible input role is
    # structurally assigned a distinct, randomly permuted non-grey color.
    bg = random.choice([color for color in range(10) if color != _GREY])
    available = [color for color in range(10) if color not in (bg, _GREY)]
    random.shuffle(available)
    legend = available[:k]
    station, track = available[k], available[k + 1]

    patch_in, patch_out, meta = _build_patch(legend, station, track)
    if not _validate_semantics(patch_in, patch_out, meta):
        return None

    turns = random.randrange(4)
    mirror = bool(random.getrandbits(1))
    patch_in = _transform(patch_in, turns, mirror)
    patch_out = _transform(patch_out, turns, mirror)
    ph, pw = patch_in.shape

    legend_width = 3 * k - 1
    gap = random.randint(2, 4)
    min_h = ph + gap + 4
    min_w = max(pw, legend_width) + 2
    if min_h > 30 or min_w > 30:
        return None
    height = random.randint(min_h, 30)
    width = random.randint(min_w, min(30, min_w + 5))

    legend_at_top = bool(random.getrandbits(1))
    legend_at_left = bool(random.getrandbits(1))
    legend_r = 1 if legend_at_top else height - 3
    legend_c = 1 if legend_at_left else width - legend_width - 1

    if legend_at_top:
        graph_r = random.randint(3 + gap, height - ph - 1)
    else:
        graph_r = random.randint(1, height - ph - gap - 3)
    graph_c = random.randint(1, width - pw - 1)

    inp = np.full((height, width), bg, dtype=int)
    expected = np.full((height, width), bg, dtype=int)
    inp[graph_r : graph_r + ph, graph_c : graph_c + pw] = np.where(patch_in == -1, bg, patch_in)
    expected[graph_r : graph_r + ph, graph_c : graph_c + pw] = np.where(patch_out == -1, bg, patch_out)
    for i, color in enumerate(legend):
        c0 = legend_c + 3 * i
        inp[legend_r : legend_r + 2, c0 : c0 + 2] = color
        expected[legend_r : legend_r + 2, c0 : c0 + 2] = color

    # Visible-structure invariants, independent of the solver oracle.
    if Counter(map(int, inp.ravel())).most_common(1)[0][0] != bg:
        return None
    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return None
    if len(set(legend)) != k or legend[0] == legend[-1]:
        return None
    try:
        actual = main(inp)
    except (AssertionError, ValueError, IndexError):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance of the legend-selected path task."""
    for _ in range(_MAX_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"generate_input: failed after {_MAX_ATTEMPTS} attempts")
