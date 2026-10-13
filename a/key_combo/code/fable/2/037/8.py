from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
SOURCE_COLORS = [1, 2, 3, 6, 7, 9]
WIRE = 4
OFF = 5
FRAME = 8

PITCH = 5
BASE = 4
MAX_RETRIES = 600

DIR_ORDER = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

PATTERN_U = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=int,
)
PATTERN_D = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=int,
)
PATTERN_L = np.array(
    [
        [1, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 1, 0],
    ],
    dtype=int,
)
PATTERN_R = np.array(
    [
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 1],
    ],
    dtype=int,
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def direction_name(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == -1 and dc == 0:
        return "U"
    if dr == 1 and dc == 0:
        return "D"
    if dr == 0 and dc == -1:
        return "L"
    if dr == 0 and dc == 1:
        return "R"
    raise AssertionError("direction_name expects orthogonal adjacent cells")


def fine_coord(node: tuple[int, int]) -> tuple[int, int]:
    return BASE + PITCH * node[0], BASE + PITCH * node[1]


def orth_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise AssertionError("orth_line requires axis-aligned endpoints")


def bulb_descriptor(socket: tuple[int, int], side: str) -> dict:
    r, c = socket
    if side == "U":
        ring_r0, ring_c0 = r - 4, c - 1
        neck = (r - 1, c)
        interior = (r - 3, c)
    elif side == "D":
        ring_r0, ring_c0 = r + 2, c - 1
        neck = (r + 1, c)
        interior = (r + 3, c)
    elif side == "L":
        ring_r0, ring_c0 = r - 1, c - 4
        neck = (r, c - 1)
        interior = (r, c - 3)
    elif side == "R":
        ring_r0, ring_c0 = r - 1, c + 2
        neck = (r, c + 1)
        interior = (r, c + 3)
    else:
        raise AssertionError("unknown bulb side")

    ring: set[tuple[int, int]] = set()
    for rr in range(ring_r0, ring_r0 + 3):
        for cc in range(ring_c0, ring_c0 + 3):
            if rr in (ring_r0, ring_r0 + 2) or cc in (ring_c0, ring_c0 + 2):
                ring.add((rr, cc))
    frame_cells = ring | {neck}

    all_cells = frame_cells | {interior}
    rows = [rr for rr, _ in all_cells]
    cols = [cc for _, cc in all_cells]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    margin = {
        (rr, cc)
        for rr in range(r0 - 1, r1 + 2)
        for cc in range(c0 - 1, c1 + 2)
    }
    return {
        "socket": socket,
        "side": side,
        "frame_cells": frame_cells,
        "interior": interior,
        "bbox": (r0, c0, r1, c1),
        "margin": margin,
    }


def parse_bulb_component(coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = bbox_of(coords)
    crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        crop[r - r0, c - c0] = 1

    if crop.shape == PATTERN_U.shape and np.array_equal(crop, PATTERN_U):
        return {"socket": (r1 + 1, c0 + 1), "interior": (r0 + 1, c0 + 1), "side": "U"}
    if crop.shape == PATTERN_D.shape and np.array_equal(crop, PATTERN_D):
        return {"socket": (r0 - 1, c0 + 1), "interior": (r0 + 2, c0 + 1), "side": "D"}
    if crop.shape == PATTERN_L.shape and np.array_equal(crop, PATTERN_L):
        return {"socket": (r0 + 1, c1 + 1), "interior": (r0 + 1, c0 + 1), "side": "L"}
    if crop.shape == PATTERN_R.shape and np.array_equal(crop, PATTERN_R):
        return {"socket": (r0 + 1, c0 - 1), "interior": (r0 + 1, c0 + 2), "side": "R"}
    return None


def graph_neighbors(grid: np.ndarray, source: tuple[int, int], cell: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    h, w = grid.shape
    for nr, nc in neighbors4(*cell):
        if not in_bounds((h, w), nr, nc):
            continue
        if (nr, nc) == source or int(grid[nr, nc]) == WIRE:
            out.append((nr, nc))
    return out


def trace_signal_states(grid: np.ndarray, source: tuple[int, int]) -> dict[tuple[int, int], dict]:
    wire_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == WIRE)]
    start_neighbors = [cell for cell in neighbors4(*source) if in_bounds(grid.shape, *cell) and int(grid[cell]) == WIRE]
    assert len(start_neighbors) == 1, "source must feed exactly one wire"
    start = start_neighbors[0]

    degree = {cell: len(graph_neighbors(grid, source, cell)) for cell in wire_cells}
    assert all(degree[cell] >= 1 for cell in wire_cells)

    info: dict[tuple[int, int], dict] = {
        start: {"parity": 0, "turns": 0, "after_branch": False, "depth": 1}
    }
    visited = {start}
    stack = [(start, source, direction_name(source, start), 0, False, 1)]

    while stack:
        cell, parent, incoming_dir, turns, after_branch, depth = stack.pop()
        branch_here = degree[cell] >= 3
        for nxt in graph_neighbors(grid, source, cell):
            if nxt == parent or nxt == source:
                continue
            if nxt in visited:
                raise AssertionError("wire must form a tree")
            ndir = direction_name(cell, nxt)
            nturns = turns + int(ndir != incoming_dir)
            nafter = after_branch or branch_here
            visited.add(nxt)
            info[nxt] = {"parity": nturns % 2, "turns": nturns, "after_branch": nafter, "depth": depth + 1}
            stack.append((nxt, cell, ndir, nturns, nafter, depth + 1))

    assert len(visited) == len(wire_cells), "all wire cells must be reachable from the source"
    return info


def candidate_sides(grid: np.ndarray, source: tuple[int, int], cell: tuple[int, int]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dirs = tuple(sorted(direction_name(cell, nbr) for nbr in graph_neighbors(grid, source, cell)))
    if not dirs or len(dirs) == 4:
        return dirs, ()
    return dirs, tuple(side for side in "UDLR" if side not in dirs)


def bulb_clear(grid: np.ndarray, source: tuple[int, int], bulb: dict) -> bool:
    h, w = grid.shape
    for r, c in bulb["frame_cells"] | {bulb["interior"]}:
        if not in_bounds((h, w), r, c):
            return False
        if int(grid[r, c]) != BG:
            return False

    allowed_touch = {bulb["socket"], bulb["interior"]} | bulb["frame_cells"]
    for r, c in bulb["frame_cells"]:
        for nr, nc in neighbors4(r, c):
            if not in_bounds((h, w), nr, nc):
                return False
            if (nr, nc) in allowed_touch:
                continue
            if int(grid[nr, nc]) == WIRE or (nr, nc) == source:
                return False
    return True


def count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(2, len(path)):
        if direction_name(path[i - 2], path[i - 1]) != direction_name(path[i - 1], path[i]):
            turns += 1
    return turns


def coarse_neighbors(node: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = node
    out = []
    for nr, nc in neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    random.shuffle(out)
    return out


def grow_path(
    start: tuple[int, int],
    length: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]] | None = None,
    min_turns: int = 0,
) -> list[tuple[int, int]] | None:
    base_occupied = set() if occupied is None else set(occupied)
    for _ in range(120):
        if start in base_occupied:
            return None
        path = [start]
        used = set(base_occupied)
        used.add(start)
        while len(path) < length:
            cands = [nbr for nbr in coarse_neighbors(path[-1], h, w) if nbr not in used]
            if not cands:
                break
            if len(path) >= 2:
                prev_dir = direction_name(path[-2], path[-1])
                turn_cands = [nbr for nbr in cands if direction_name(path[-1], nbr) != prev_dir]
                straight_cands = [nbr for nbr in cands if direction_name(path[-1], nbr) == prev_dir]
                if turn_cands and (not straight_cands or random.random() < 0.65):
                    cands = turn_cands
                else:
                    cands = straight_cands or cands
            nxt = random.choice(cands)
            path.append(nxt)
            used.add(nxt)
        if len(path) == length and count_turns(path) >= min_turns:
            return path
    return None


def build_tree() -> tuple[np.ndarray, tuple[int, int], dict]:
    h_coarse = random.choice([4, 5])
    w_coarse = random.choice([4, 5])
    source_color = random.choice(SOURCE_COLORS)
    need_branch = random.random() < 0.75

    trunk_len = random.randint(5, 7)
    if h_coarse * w_coarse < trunk_len:
        trunk_len = h_coarse * w_coarse

    trunk = None
    for _ in range(120):
        start = (random.randrange(h_coarse), random.randrange(w_coarse))
        min_trunk_turns = 2 if not need_branch else 1
        trunk = grow_path(start, trunk_len, h_coarse, w_coarse, min_turns=min_trunk_turns)
        if trunk is not None:
            break
    if trunk is None:
        raise RuntimeError("failed to build trunk")

    occupied = set(trunk)
    branches: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    n_branches = random.choice([1, 2]) if need_branch else 0

    anchor_indices = list(range(1, len(trunk) - 1))
    random.shuffle(anchor_indices)
    for anchor_idx in anchor_indices:
        if len(branches) >= n_branches:
            break
        anchor = trunk[anchor_idx]
        prev_node = trunk[anchor_idx - 1]
        next_node = trunk[anchor_idx + 1]
        used_dirs = {direction_name(anchor, prev_node), direction_name(anchor, next_node)}
        starters = [
            nbr
            for nbr in coarse_neighbors(anchor, h_coarse, w_coarse)
            if nbr not in occupied and direction_name(anchor, nbr) not in used_dirs
        ]
        random.shuffle(starters)
        made_branch = False
        for starter in starters:
            branch_len = random.randint(2, 4)
            min_branch_turns = 1 if branch_len >= 4 and random.random() < 0.5 else 0
            branch = grow_path(
                starter,
                branch_len,
                h_coarse,
                w_coarse,
                occupied=occupied | {anchor},
                min_turns=min_branch_turns,
            )
            if branch is None:
                continue
            branches.append((anchor, branch))
            occupied.update(branch)
            made_branch = True
            break
        if made_branch and not need_branch:
            break

    if need_branch and not branches:
        raise RuntimeError("branch required but not created")

    grid = np.full((h_coarse * PITCH + 4, w_coarse * PITCH + 4), BG, dtype=int)
    source = fine_coord(trunk[0])
    grid[source] = source_color

    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for a, b in zip(trunk, trunk[1:], strict=False):
        edges.append((a, b))
    for anchor, branch in branches:
        edges.append((anchor, branch[0]))
        for a, b in zip(branch, branch[1:], strict=False):
            edges.append((a, b))

    for a, b in edges:
        for rr, cc in orth_line(fine_coord(a), fine_coord(b)):
            if (rr, cc) == source:
                continue
            grid[rr, cc] = WIRE

    meta = {
        "source_color": source_color,
        "need_branch": need_branch,
        "trunk": trunk,
        "branches": branches,
    }
    return grid, source, meta


def choose_bulbs(base_grid: np.ndarray, source: tuple[int, int], need_branch: bool) -> list[dict]:
    info = trace_signal_states(base_grid, source)
    candidates: list[dict] = []
    for cell, state in info.items():
        dirs, sides = candidate_sides(base_grid, source, cell)
        if not sides:
            continue
        for side in sides:
            bulb = bulb_descriptor(cell, side)
            if not bulb_clear(base_grid, source, bulb):
                continue
            candidates.append(
                {
                    "socket": cell,
                    "side": side,
                    "dirs": dirs,
                    "signature": (dirs, side),
                    "turns": int(state["turns"]),
                    "parity": int(state["parity"]),
                    "after_branch": bool(state["after_branch"]),
                    "depth": int(state["depth"]),
                    "bulb": bulb,
                }
            )

    if len(candidates) < 4:
        raise RuntimeError("not enough bulb candidates")

    def conflict(a: dict, b: dict) -> bool:
        return not a["bulb"]["margin"].isdisjoint(b["bulb"]["margin"])

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for cand in candidates:
        grouped[cand["signature"]].append(cand)

    pair: tuple[dict, dict] | None = None
    pair_score: tuple[int, int, int] | None = None
    for items in grouped.values():
        evens = [item for item in items if item["parity"] == 0]
        odds = [item for item in items if item["parity"] == 1]
        for a in evens:
            for b in odds:
                if conflict(a, b):
                    continue
                cats = {turn_bucket(a["turns"]), turn_bucket(b["turns"])}
                score = (
                    len(cats),
                    int(a["after_branch"] or b["after_branch"]),
                    abs(a["depth"] - b["depth"]),
                )
                if pair is None or score > pair_score:
                    pair = (a, b)
                    pair_score = score
    if pair is None:
        raise RuntimeError("missing same-signature parity counterexample")

    target_count = random.randint(4, min(7, len(candidates)))
    selected = [pair[0], pair[1]]
    used_margin = set(pair[0]["bulb"]["margin"]) | set(pair[1]["bulb"]["margin"])
    used_sockets = {pair[0]["socket"], pair[1]["socket"]}

    def can_add(cand: dict) -> bool:
        return cand["socket"] not in used_sockets and used_margin.isdisjoint(cand["bulb"]["margin"])

    categories = {turn_bucket(item["turns"]) for item in selected}
    remaining = [cand for cand in candidates if cand not in selected]
    random.shuffle(remaining)

    required = {"zero", "odd", "even_positive"}
    for wanted in sorted(required - categories):
        pick = next((cand for cand in remaining if turn_bucket(cand["turns"]) == wanted and can_add(cand)), None)
        if pick is None:
            raise RuntimeError("missing turn bucket coverage")
        selected.append(pick)
        used_margin |= pick["bulb"]["margin"]
        used_sockets.add(pick["socket"])
        categories.add(wanted)

    if need_branch and not any(item["after_branch"] for item in selected):
        pick = next((cand for cand in remaining if cand["after_branch"] and can_add(cand)), None)
        if pick is None:
            raise RuntimeError("missing bulb beyond branch")
        selected.append(pick)
        used_margin |= pick["bulb"]["margin"]
        used_sockets.add(pick["socket"])

    priority = sorted(
        remaining,
        key=lambda cand: (
            int(cand["after_branch"]),
            int(turn_bucket(cand["turns"]) == "even_positive"),
            cand["depth"],
        ),
        reverse=True,
    )
    for cand in priority:
        if len(selected) >= target_count:
            break
        if can_add(cand):
            selected.append(cand)
            used_margin |= cand["bulb"]["margin"]
            used_sockets.add(cand["socket"])

    if len(selected) < 4:
        raise RuntimeError("too few selected bulbs")
    if {"zero", "odd", "even_positive"} - {turn_bucket(item["turns"]) for item in selected}:
        raise RuntimeError("selected bulbs lost turn coverage")
    if need_branch and not any(item["after_branch"] for item in selected):
        raise RuntimeError("selected bulbs lost branch coverage")

    signature_groups = defaultdict(set)
    for item in selected:
        signature_groups[item["signature"]].add(item["parity"])
    if not any(parities == {0, 1} for parities in signature_groups.values()):
        raise RuntimeError("selected bulbs lost parity counterexample")

    return selected


def turn_bucket(turns: int) -> str:
    if turns == 0:
        return "zero"
    if turns % 2 == 1:
        return "odd"
    return "even_positive"


def paint_bulbs(grid: np.ndarray, bulbs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in bulbs:
        for r, c in item["bulb"]["frame_cells"]:
            out[r, c] = FRAME
        rr, cc = item["bulb"]["interior"]
        assert int(out[rr, cc]) == BG
    return out


def expected_output(inp: np.ndarray, bulbs: list[dict], source_color: int) -> np.ndarray:
    out = inp.copy()
    for item in bulbs:
        rr, cc = item["bulb"]["interior"]
        out[rr, cc] = source_color if item["parity"] == 0 else OFF
    return out


def validate_instance(inp: np.ndarray, source: tuple[int, int], source_color: int, bulbs: list[dict], need_branch: bool) -> None:
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert int(inp[source]) == source_color
    assert np.count_nonzero(inp == source_color) == 1

    for item in bulbs:
        assert int(inp[item["socket"]]) == WIRE
        rr, cc = item["bulb"]["interior"]
        assert int(inp[rr, cc]) == BG

    out = main(inp)
    exp = expected_output(inp, bulbs, source_color)
    assert np.array_equal(out, exp), "solver/output mismatch"

    changed = np.argwhere(out != inp)
    assert len(changed) == len(bulbs), "only bulb interiors should change"

    fill_colors = {int(out[item["bulb"]["interior"]]) for item in bulbs}
    assert source_color in fill_colors and OFF in fill_colors, "both bright and dim bulbs must appear"

    categories = {turn_bucket(item["turns"]) for item in bulbs}
    assert categories == {"zero", "odd", "even_positive"}

    signature_groups = defaultdict(set)
    for item in bulbs:
        signature_groups[item["signature"]].add(item["parity"])
    assert any(parities == {0, 1} for parities in signature_groups.values()), "need same-local-shape opposite outputs"

    if need_branch:
        assert any(item["after_branch"] for item in bulbs), "branching examples need downstream bulbs"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each bulb interior with the signal color determined by the cumulative number of right-angle turns from the source.

    Input structure:
    - The grid has a dominant background, one single-pixel source, one single-cell-wide orthogonal wire tree, and several hollow bulb frames with short sockets.
    - Every bulb frame is a 3x3 hollow square plus a one-cell neck that points toward exactly one wire cell.
    - The source color varies, while the wire and bulb-frame colors stay fixed across instances.

    Rule:
    - Start a signal at the source and propagate it along the wire.
    - The signal begins bright in the source color and flips between source color and grey whenever the path makes a 90-degree turn.
    - When the signal reaches a wire cell that a bulb socket points to, fill that bulb's interior with the current signal color.
    - At branch points, the current state is copied into every outgoing branch, and each branch keeps counting only its own later turns.
    - Keep the source pixel, the wire, and the bulb frames unchanged.

    Key insight:
    - A bulb depends on the total turn parity along the full source-to-bulb route, so bulbs with the same local attachment shape can still end up with opposite fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    source_colors = [color for color in colors if int(np.count_nonzero(g == color)) == 1]
    assert len(source_colors) == 1, "expected exactly one single-pixel source color"
    source_color = source_colors[0]
    source = tuple(map(int, np.argwhere(g == source_color)[0]))

    wire_neighbor_colors = {
        int(g[nr, nc])
        for nr, nc in neighbors4(*source)
        if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) != bg
    }
    wire_neighbor_colors.discard(source_color)
    assert len(wire_neighbor_colors) == 1, "source must touch exactly one wire color"
    wire_color = next(iter(wire_neighbor_colors))
    assert wire_color == WIRE, "wire color must match the puzzle encoding"

    frame_colors = [color for color in colors if color not in {source_color, wire_color}]
    assert len(frame_colors) == 1, "expected exactly one bulb-frame color"
    frame_color = frame_colors[0]
    assert frame_color == FRAME, "frame color must match the puzzle encoding"

    bulbs_by_socket: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for comp in connected_components(g == frame_color):
        bulb = parse_bulb_component(comp)
        assert bulb is not None, "every frame-color component must be a valid bulb frame"
        socket = bulb["socket"]
        assert in_bounds(g.shape, *socket) and int(g[socket]) == wire_color
        rr, cc = bulb["interior"]
        assert int(g[rr, cc]) == bg
        bulbs_by_socket[socket].append((rr, cc))

    states = trace_signal_states(g, source)
    out = g.copy()
    for socket, interiors in bulbs_by_socket.items():
        parity = int(states[socket]["parity"])
        fill = source_color if parity == 0 else OFF
        for rr, cc in interiors:
            out[rr, cc] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    try:
        base_grid, source, meta = build_tree()
        bulbs = choose_bulbs(base_grid, source, meta["need_branch"])
        inp = paint_bulbs(base_grid, bulbs)
        validate_instance(inp, source, meta["source_color"], bulbs, meta["need_branch"])
        return inp
    except (AssertionError, RuntimeError):
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after retries")
