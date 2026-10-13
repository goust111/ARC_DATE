from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg, neighbors4


BG = int(Color.BLACK)
TRACK = int(Color.GREY)
SINK = int(Color.RED)
BEAD_COLORS = [
    int(Color.BLUE),
    int(Color.GREEN),
    int(Color.YELLOW),
    int(Color.PINK),
    int(Color.ORANGE),
    int(Color.TEAL),
    int(Color.MAROON),
]

MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _find_sink_mask(grid: np.ndarray) -> np.ndarray:
    red = grid == SINK
    squares: list[np.ndarray] = []
    for coords in components_mask(red, connectivity=4):
        if int(coords.shape[0]) != 4:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1 - r0) == 1 and int(c1 - c0) == 1:
            squares.append(coords)
    assert len(squares) == 1, "expected one red 2x2 sink square"
    mask = np.zeros_like(grid, dtype=bool)
    coords = squares[0]
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _track_graph(
    path_mask: np.ndarray, sink_mask: np.ndarray
) -> tuple[dict[tuple[int, int], list[tuple[int, int]]], tuple[int, int], tuple[int, int], list[tuple[int, int]], dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
    comps = components_mask(path_mask, connectivity=4)
    assert len(comps) == 1, "track plus beads must form one connected component"
    track = {tuple(map(int, rc)) for rc in comps[0]}

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    degrees: dict[tuple[int, int], int] = {}
    for cell in track:
        adj = [
            (int(nr), int(nc))
            for nr, nc in neighbors4(*cell)
            if (int(nr), int(nc)) in track
        ]
        neighbors[cell] = adj
        degrees[cell] = len(adj)

    assert all(deg in {1, 2, 3} for deg in degrees.values()), "Y track must be 1-cell wide"
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    junctions = [cell for cell, deg in degrees.items() if deg == 3]
    assert len(endpoints) == 3 and len(junctions) == 1, "expected one junction and three endpoints"
    junction = junctions[0]

    sink_endpoints = []
    for cell in endpoints:
        for nr, nc in neighbors4(*cell):
            if _in_bounds(path_mask.shape, nr, nc) and bool(sink_mask[int(nr), int(nc)]):
                sink_endpoints.append(cell)
                break
    assert len(sink_endpoints) == 1, "exactly one endpoint must touch the sink square"
    sink_endpoint = sink_endpoints[0]

    dist = {sink_endpoint: 0}
    prev: dict[tuple[int, int], tuple[int, int] | None] = {sink_endpoint: None}
    q = deque([sink_endpoint])
    while q:
        cell = q.popleft()
        for nxt in neighbors[cell]:
            if nxt in dist:
                continue
            dist[nxt] = int(dist[cell]) + 1
            prev[nxt] = cell
            q.append(nxt)

    assert len(dist) == len(track), "track must be a tree reachable from the sink"
    return neighbors, sink_endpoint, junction, endpoints, dist, prev


def _trunk_from_prev(
    prev: dict[tuple[int, int], tuple[int, int] | None], junction: tuple[int, int]
) -> list[tuple[int, int]]:
    trunk: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = junction
    while cur is not None:
        trunk.append(cur)
        cur = prev[cur]
    trunk.reverse()
    return trunk


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge uniquely ordered beads from a Y track onto the trunk toward the red sink square.

    Input structure:
    - The background is black.
    - A one-cell-wide non-self-intersecting grey Y track carries several single-pixel beads of pairwise distinct colors; bead cells overwrite the grey track but still belong to it.
    - One endpoint of the track is orthogonally adjacent to a red 2x2 sink square, while the opposite side reaches a unique junction that splits into two branches.
    - The total bead count does not exceed the number of trunk cells from the sink-adjacent endpoint to the junction, and all bead path distances to that endpoint are unique.

    Rule:
    - Treat the visible grey cells together with the bead cells as one Y-shaped track tree.
    - Measure each bead's distance by the number of steps along that track to the sink-adjacent endpoint.
    - Restore the whole track to grey, then sort beads from nearest to farthest by that path distance and place them on successive trunk cells starting at the sink-adjacent endpoint and moving one step at a time toward the junction.
    - Keep the sink square and the track geometry unchanged.

    Key insight:
    - The merge order is the one-dimensional order along the track tree, not visual closeness to the red sink or branch membership.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG, "background must be black"

    sink_mask = _find_sink_mask(g)
    path_mask = (g != bg) & (~sink_mask)
    neighbors, sink_endpoint, junction, _endpoints, dist, prev = _track_graph(path_mask, sink_mask)
    trunk = _trunk_from_prev(prev, junction)

    bead_cells = [cell for cell in dist if int(g[cell]) != TRACK]
    bead_colors = [int(g[cell]) for cell in bead_cells]
    assert bead_cells, "expected at least one bead"
    assert len(set(bead_colors)) == len(bead_colors), "bead colors must be unique"

    bead_order = sorted((int(dist[cell]), cell, int(g[cell])) for cell in bead_cells)
    bead_dists = [item[0] for item in bead_order]
    assert len(set(bead_dists)) == len(bead_dists), "bead path distances must be unique"
    assert len(bead_order) <= len(trunk), "merged queue must fit on the trunk"

    out = np.full_like(g, bg)
    out[sink_mask] = SINK
    out[path_mask] = TRACK
    for i, (_d, _cell, color) in enumerate(bead_order):
        r, c = trunk[i]
        out[int(r), int(c)] = int(color)
    return out


def _rotate(coord: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = coord
    turns = int(turns) % 4
    if turns == 0:
        return int(r), int(c)
    if turns == 1:
        return int(c), int(-r)
    if turns == 2:
        return int(-r), int(-c)
    return int(-c), int(r)


def _transform(coords: list[tuple[int, int]], turns: int, dr: int, dc: int) -> list[tuple[int, int]]:
    out = []
    for coord in coords:
        r, c = _rotate(coord, turns)
        out.append((int(r + dr), int(c + dc)))
    return out


def _track_degrees(coords: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    track = set(coords)
    return {
        cell: sum((int(nr), int(nc)) in track for nr, nc in neighbors4(*cell))
        for cell in track
    }


def _build_canonical_layout() -> dict | None:
    trunk_len = random.randint(5, 10)
    upper_v = random.randint(2, 4)
    lower_v = random.randint(2, 4)
    upper_h = random.randint(2, 5)
    lower_h = random.randint(2, 5)
    upper_dir = random.choice([-1, 1])
    lower_dir = random.choice([-1, 1])
    if upper_dir == 1 and lower_dir == 1:
        if random.random() < 0.5:
            upper_dir = -1
        else:
            lower_dir = -1

    trunk = [(0, c) for c in range(trunk_len + 1)]
    upper = [(-k, trunk_len) for k in range(1, upper_v + 1)]
    upper += [(-upper_v, trunk_len + upper_dir * k) for k in range(1, upper_h + 1)]
    lower = [(k, trunk_len) for k in range(1, lower_v + 1)]
    lower += [(lower_v, trunk_len + lower_dir * k) for k in range(1, lower_h + 1)]
    sink = [(-1, -2), (-1, -1), (0, -2), (0, -1)]

    track = trunk + upper + lower
    if len(set(track)) != len(track):
        return None
    degrees = _track_degrees(track)
    if sorted(degrees.values()).count(3) != 1 or sorted(degrees.values()).count(1) != 3:
        return None
    if any(deg not in {1, 2, 3} for deg in degrees.values()):
        return None

    return {
        "sink": sink,
        "trunk": trunk,
        "upper": upper,
        "lower": lower,
    }


def _choose_interleaved_branch_offsets(upper_len: int, lower_len: int) -> tuple[list[int], list[int], str] | None:
    patterns = [
        pattern
        for pattern in ("ABA", "BAB", "ABAB", "BABA")
        if pattern.count("A") <= upper_len and pattern.count("B") <= lower_len
    ]
    random.shuffle(patterns)
    for pattern in patterns:
        offsets: list[int] = []
        prev = 0
        ok = True
        for idx, label in enumerate(pattern):
            limit = upper_len if label == "A" else lower_len
            remaining = len(pattern) - idx - 1
            hi = limit - remaining
            lo = prev + 1
            if hi < lo:
                ok = False
                break
            offsets.append(random.randint(lo, hi))
            prev = offsets[-1]
        if not ok:
            continue
        upper_offsets = [offsets[i] for i, label in enumerate(pattern) if label == "A"]
        lower_offsets = [offsets[i] for i, label in enumerate(pattern) if label == "B"]
        return upper_offsets, lower_offsets, pattern
    return None


def _assign_beads(layout: dict) -> dict | None:
    trunk = layout["trunk"]
    upper = layout["upper"]
    lower = layout["lower"]

    branch_pick = _choose_interleaved_branch_offsets(len(upper), len(lower))
    if branch_pick is None:
        return None
    upper_offsets, lower_offsets, pattern = branch_pick

    branch_beads: list[dict] = []
    for offset in upper_offsets:
        branch_beads.append(
            {
                "coord": upper[offset - 1],
                "distance": len(trunk) - 1 + offset,
                "branch": "A",
            }
        )
    for offset in lower_offsets:
        branch_beads.append(
            {
                "coord": lower[offset - 1],
                "distance": len(trunk) - 1 + offset,
                "branch": "B",
            }
        )
    branch_beads.sort(key=lambda bead: bead["distance"])
    labels = "".join(bead["branch"] for bead in branch_beads)
    switches = sum(labels[i] != labels[i - 1] for i in range(1, len(labels)))
    if switches < 2:
        return None

    max_extra = min(2, len(trunk) - len(branch_beads))
    trunk_count = random.randint(0, max_extra)
    trunk_offsets = sorted(random.sample(range(len(trunk)), trunk_count))
    trunk_beads = [
        {
            "coord": trunk[offset],
            "distance": offset,
            "branch": "T",
        }
        for offset in trunk_offsets
    ]

    beads = trunk_beads + branch_beads
    if len(beads) < 3 or len(beads) > len(trunk) or len(beads) > len(BEAD_COLORS):
        return None

    colors = random.sample(BEAD_COLORS, len(beads))
    random.shuffle(colors)
    for bead, color in zip(beads, colors):
        bead["color"] = int(color)

    layout = dict(layout)
    layout["beads"] = beads
    layout["pattern"] = pattern
    return layout


def _place_layout(layout: dict) -> dict | None:
    all_coords = layout["sink"] + layout["trunk"] + layout["upper"] + layout["lower"]
    turns = random.randint(0, 3)
    rotated = [_rotate(coord, turns) for coord in all_coords]
    rows = [r for r, _c in rotated]
    cols = [c for _r, c in rotated]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    top = random.randint(1, 4)
    left = random.randint(1, 4)
    bottom = random.randint(1, 4)
    right = random.randint(1, 4)

    height = int(max_r - min_r + 1 + top + bottom)
    width = int(max_c - min_c + 1 + left + right)
    if max(height, width) > 30:
        return None

    dr = top - min_r
    dc = left - min_c
    placed = {
        "shape": (height, width),
        "sink": _transform(layout["sink"], turns, dr, dc),
        "trunk": _transform(layout["trunk"], turns, dr, dc),
        "upper": _transform(layout["upper"], turns, dr, dc),
        "lower": _transform(layout["lower"], turns, dr, dc),
        "beads": [],
    }
    for bead in layout["beads"]:
        coord = _transform([bead["coord"]], turns, dr, dc)[0]
        placed["beads"].append(
            {
                "coord": coord,
                "distance": int(bead["distance"]),
                "branch": bead["branch"],
                "color": int(bead["color"]),
            }
        )
    return placed


def _paint_input(spec: dict) -> np.ndarray:
    grid = np.full(spec["shape"], BG, dtype=int)
    for r, c in spec["sink"]:
        grid[int(r), int(c)] = SINK
    for r, c in spec["trunk"] + spec["upper"] + spec["lower"]:
        grid[int(r), int(c)] = TRACK
    for bead in spec["beads"]:
        r, c = bead["coord"]
        grid[int(r), int(c)] = int(bead["color"])
    return grid


def _paint_output(spec: dict) -> np.ndarray:
    grid = np.full(spec["shape"], BG, dtype=int)
    for r, c in spec["sink"]:
        grid[int(r), int(c)] = SINK
    for r, c in spec["trunk"] + spec["upper"] + spec["lower"]:
        grid[int(r), int(c)] = TRACK

    ordered = sorted(spec["beads"], key=lambda bead: int(bead["distance"]))
    for i, bead in enumerate(ordered):
        r, c = spec["trunk"][i]
        grid[int(r), int(c)] = int(bead["color"])
    return grid


def _min_sink_manhattan(coord: tuple[int, int], sink: list[tuple[int, int]]) -> int:
    r, c = coord
    return min(abs(int(r) - int(sr)) + abs(int(c) - int(sc)) for sr, sc in sink)


def _semantic_checks(spec: dict, inp: np.ndarray, out: np.ndarray) -> bool:
    beads = spec["beads"]
    bead_coords = [tuple(bead["coord"]) for bead in beads]
    bead_colors = [int(bead["color"]) for bead in beads]
    if len(set(bead_coords)) != len(bead_coords):
        return False
    if len(set(bead_colors)) != len(bead_colors):
        return False

    branch_labels = [bead["branch"] for bead in sorted(beads, key=lambda bead: bead["distance"]) if bead["branch"] != "T"]
    if len(branch_labels) < 3:
        return False
    if set(branch_labels) != {"A", "B"}:
        return False
    if sum(branch_labels[i] != branch_labels[i - 1] for i in range(1, len(branch_labels))) < 2:
        return False

    dists = [int(bead["distance"]) for bead in beads]
    if len(set(dists)) != len(dists):
        return False
    if len(beads) > len(spec["trunk"]):
        return False

    path_order = [tuple(bead["coord"]) for bead in sorted(beads, key=lambda bead: bead["distance"])]
    manhattan_order = [
        tuple(bead["coord"])
        for bead in sorted(
            beads,
            key=lambda bead: (_min_sink_manhattan(tuple(bead["coord"]), spec["sink"]), bead["coord"][0], bead["coord"][1]),
        )
    ]
    if path_order == manhattan_order:
        return False

    if np.array_equal(inp, out):
        return False

    out_bead_positions = [tuple(coord) for coord in spec["trunk"][: len(beads)]]
    if len(set(out_bead_positions)) != len(out_bead_positions):
        return False

    if np.any((inp == SINK) != (out == SINK)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        layout = _build_canonical_layout()
        if layout is None:
            continue
        layout = _assign_beads(layout)
        if layout is None:
            continue
        spec = _place_layout(layout)
        if spec is None:
            continue

        inp = _paint_input(spec)
        out = _paint_output(spec)
        if not _semantic_checks(spec, inp, out):
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    raise RuntimeError("failed to generate a valid branch-merge sample")
