import itertools
import random
from collections import deque

import numpy as np


BG = 0
WIRE = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _runs(mask) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_legend(grid: np.ndarray, bg: int, wire: int) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[list[int], set[tuple[int, int]]]] = []
    for r in (0, h - 1):
        mask = (grid[r] != bg) & (grid[r] != wire)
        for c0, c1 in _runs(mask.tolist()):
            length = c1 - c0 + 1
            if length not in (3, 4):
                continue
            if c0 not in (0, w - length):
                continue
            colors = [int(x) for x in grid[r, c0 : c1 + 1]]
            if bg in colors or wire in colors or len(set(colors)) != length:
                continue
            inward_r = 1 if r == 0 else h - 2
            if not np.all(grid[inward_r, c0 : c1 + 1] == bg):
                continue
            cells = {(r, c) for c in range(c0, c1 + 1)}
            candidates.append((colors, cells))
    assert len(candidates) == 1, f"expected one legend run, got {len(candidates)}"
    return candidates[0]


def _find_blocks(
    grid: np.ndarray, bg: int, wire: int, legend_cells: set[tuple[int, int]]
) -> list[dict]:
    h, w = grid.shape
    blocks: list[dict] = []
    for r in range(h - 1):
        for c in range(w - 1):
            cells = {(r + dr, c + dc) for dr in range(2) for dc in range(2)}
            if cells & legend_cells:
                continue
            window = grid[r : r + 2, c : c + 2]
            vals = {int(x) for x in window.ravel()}
            if wire in vals:
                continue
            non_bg = vals - {bg}
            if len(non_bg) != 1:
                continue
            color = int(next(iter(non_bg)))
            color_count = int(np.sum(window == color))
            bg_count = int(np.sum(window == bg))
            if not ((color_count == 4 and bg_count == 0) or (color_count == 3 and bg_count == 1)):
                continue

            extends = False
            for rr, cc in cells:
                if int(grid[rr, cc]) != color:
                    continue
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if (nr, nc) not in cells and int(grid[nr, nc]) == color:
                        extends = True
                        break
                if extends:
                    break
            if extends:
                continue

            adjacent_wire: set[tuple[int, int]] = set()
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if (nr, nc) not in cells and int(grid[nr, nc]) == wire:
                        adjacent_wire.add((nr, nc))
            if len(adjacent_wire) != 1:
                continue

            blocks.append(
                {
                    "top_left": (r, c),
                    "cells": cells,
                    "color": color,
                    "is_source": color_count == 3,
                    "endpoint": next(iter(adjacent_wire)),
                }
            )

    endpoints = [block["endpoint"] for block in blocks]
    assert len(set(endpoints)) == len(endpoints), "blocks must attach to distinct wire endpoints"
    assert sum(block["is_source"] for block in blocks) == 1, "expected exactly one source block"
    assert any(not block["is_source"] for block in blocks), "expected at least one receiver block"
    return blocks


def _wire_structure(grid: np.ndarray, wire: int) -> tuple[dict, set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = grid.shape
    wire_cells = {tuple(map(int, pt)) for pt in np.argwhere(grid == wire)}
    assert wire_cells, "wire network must be present"

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {cell: [] for cell in wire_cells}
    for r, c in wire_cells:
        neighbors = []
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in wire_cells:
                neighbors.append((nr, nc))
        assert neighbors, "wire network cannot contain isolated pixels"
        adj[(r, c)] = neighbors

    start = next(iter(wire_cells))
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    assert seen == wire_cells, "wire network must be a single component"

    endpoints = {cell for cell, neighbors in adj.items() if len(neighbors) == 1}
    relay_nodes: set[tuple[int, int]] = set()
    for r, c in wire_cells:
        has_h = (r, c - 1) in wire_cells or (r, c + 1) in wire_cells
        has_v = (r - 1, c) in wire_cells or (r + 1, c) in wire_cells
        if len(adj[(r, c)]) >= 3 and has_h and has_v:
            relay_nodes.add((r, c))

    return adj, relay_nodes, endpoints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the hollow 2x2 signal source, follow the grey wire tree, and recolor
    each receiver by the signal color after one cycle-step per relay node.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors, "grid must contain foreground colors"
    wire = max(colors, key=lambda c: int(np.sum(g == c)))

    cycle, legend_cells = _find_legend(g, bg=bg, wire=wire)
    blocks = _find_blocks(g, bg=bg, wire=wire, legend_cells=legend_cells)
    source = next(block for block in blocks if block["is_source"])
    receivers = [block for block in blocks if not block["is_source"]]

    adj, relay_nodes, endpoints = _wire_structure(g, wire=wire)
    block_endpoints = {block["endpoint"] for block in blocks}
    assert endpoints == block_endpoints, "every wire endpoint must have exactly one attached 2x2 block"
    assert source["color"] in cycle, "source color must belong to the legend cycle"

    source_ep = source["endpoint"]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {source_ep: None}
    relay_count = {source_ep: 0}
    stack = [source_ep]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt == parent[cur]:
                continue
            assert nxt not in parent, "wire network must not contain cycles"
            parent[nxt] = cur
            relay_count[nxt] = relay_count[cur] + (1 if nxt in relay_nodes else 0)
            stack.append(nxt)
    assert len(parent) == len(adj), "source must reach the whole network"

    cycle_index = {color: i for i, color in enumerate(cycle)}
    start_idx = cycle_index[source["color"]]
    out = g.copy()
    for block in receivers:
        steps = relay_count[block["endpoint"]]
        color = cycle[(start_idx + steps) % len(cycle)]
        r0, c0 = block["top_left"]
        out[r0 : r0 + 2, c0 : c0 + 2] = color
    return out


def _block_top_left(endpoint: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = endpoint
    if side == "L":
        return r - 1, c - 2
    if side == "R":
        return r, c + 1
    if side == "U":
        return r - 2, c - 1
    if side == "D":
        return r + 1, c
    raise ValueError(f"unknown side {side}")


def _contact_local(side: str) -> tuple[int, int]:
    if side in ("L", "U"):
        return 1, 1
    if side in ("R", "D"):
        return 0, 0
    raise ValueError(f"unknown side {side}")


def _legend_cells(h: int, w: int, cycle_len: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        row, start = 0, 0
    elif corner == "tr":
        row, start = 0, w - cycle_len
    elif corner == "bl":
        row, start = h - 1, 0
    elif corner == "br":
        row, start = h - 1, w - cycle_len
    else:
        raise ValueError(f"unknown corner {corner}")
    return [(row, start + i) for i in range(cycle_len)]


def _to_rc(orientation: str, a: int, b: int) -> tuple[int, int]:
    if orientation == "h":
        return b, a
    return a, b


def _random_segment_lengths(relay_count: int, axis_len: int) -> list[int] | None:
    parts = relay_count + 1
    if relay_count == 0:
        lo, hi = 8, min(12, axis_len - 9)
        if lo > hi:
            return None
        return [random.randint(lo, hi)]

    lo, hi = 3, 6
    max_total = axis_len - 9
    min_total = lo * parts
    max_feasible = min(hi * parts, max_total)
    if min_total > max_feasible:
        return None
    total = random.randint(min_total, max_feasible)
    options = [xs for xs in itertools.product(range(lo, hi + 1), repeat=parts) if sum(xs) == total]
    if not options:
        return None
    return list(random.choice(options))


def _render_attempt(
    h: int,
    w: int,
    orientation: str,
    source_at_min: bool,
    cycle: list[int],
    source_color: int,
    receiver_colors: list[int],
    corner: str,
    relay_count: int,
) -> np.ndarray | None:
    a_size = w if orientation == "h" else h
    b_size = h if orientation == "h" else w

    lengths = _random_segment_lengths(relay_count, a_size)
    if lengths is None:
        return None
    total_span = sum(lengths)
    if a_size - 5 - total_span < 4:
        return None
    a0 = random.randint(4, a_size - 5 - total_span)
    positions = [a0]
    for length in lengths:
        positions.append(positions[-1] + length)

    if relay_count == 0:
        if b_size < 9:
            return None
        b_trunk = random.randint(4, b_size - 5)
        branch_specs: list[tuple[int, int]] = []
    else:
        if b_size < 15:
            return None
        b_trunk = random.randint(7, b_size - 8)
        branch_specs = []
        relay_positions = positions[1:-1]
        for a in relay_positions:
            sign = random.choice([-1, 1])
            max_len = b_trunk - 4 if sign < 0 else (b_size - 5) - b_trunk
            if max_len < 3:
                return None
            length = random.randint(3, max_len)
            branch_specs.append((a, b_trunk + sign * length))

    if source_at_min:
        source_a, far_a = positions[0], positions[-1]
    else:
        source_a, far_a = positions[-1], positions[0]

    if orientation == "h":
        source_side = "L" if source_at_min else "R"
        far_side = "R" if source_at_min else "L"
        branch_side = lambda b_end: "U" if b_end < b_trunk else "D"
    else:
        source_side = "U" if source_at_min else "D"
        far_side = "D" if source_at_min else "U"
        branch_side = lambda b_end: "L" if b_end < b_trunk else "R"

    source_endpoint = _to_rc(orientation, source_a, b_trunk)
    receiver_endpoints = [_to_rc(orientation, far_a, b_trunk)]
    receiver_sides = [far_side]
    for a, b_end in branch_specs:
        receiver_endpoints.append(_to_rc(orientation, a, b_end))
        receiver_sides.append(branch_side(b_end))

    if len(receiver_endpoints) != len(receiver_colors):
        return None

    legend_coords = _legend_cells(h, w, len(cycle), corner)
    legend_set = set(legend_coords)
    wire_cells: set[tuple[int, int]] = set()

    a_min, a_max = positions[0], positions[-1]
    for a in range(a_min, a_max + 1):
        wire_cells.add(_to_rc(orientation, a, b_trunk))
    for a, b_end in branch_specs:
        lo, hi = sorted((b_trunk, b_end))
        for b in range(lo, hi + 1):
            wire_cells.add(_to_rc(orientation, a, b))

    if wire_cells & legend_set:
        return None

    g = np.full((h, w), BG, dtype=int)
    for (r, c), color in zip(legend_coords, cycle):
        g[r, c] = color
    for r, c in wire_cells:
        g[r, c] = WIRE

    used_block_cells: set[tuple[int, int]] = set()

    def place_block(endpoint: tuple[int, int], side: str, color: int, hollow: bool) -> bool:
        r0, c0 = _block_top_left(endpoint, side)
        cells = [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in cells):
            return False
        if any((rr, cc) in legend_set or (rr, cc) in wire_cells or (rr, cc) in used_block_cells for rr, cc in cells):
            return False
        contact = _contact_local(side)
        hole = None
        if hollow:
            choices = [(dr, dc) for dr in range(2) for dc in range(2) if (dr, dc) != contact]
            hole = random.choice(choices)
        for dr in range(2):
            for dc in range(2):
                rr, cc = r0 + dr, c0 + dc
                g[rr, cc] = BG if hollow and (dr, dc) == hole else color
                used_block_cells.add((rr, cc))
        return True

    if not place_block(source_endpoint, source_side, source_color, hollow=True):
        return None
    for endpoint, side, color in zip(receiver_endpoints, receiver_sides, receiver_colors):
        if not place_block(endpoint, side, color, hollow=False):
            return None

    non_bg_counts = {int(color): int(np.sum(g == color)) for color in np.unique(g) if int(color) != BG}
    if max(non_bg_counts, key=non_bg_counts.get) != WIRE:
        return None

    out = main(g)
    if np.array_equal(out, g):
        return None
    return g


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(["h", "v"])
    h = random.randint(17, 24)
    w = random.randint(18, 28)

    a_size = w if orientation == "h" else h
    max_relays = min(3, max(0, (a_size - 12) // 3))
    branched = max_relays >= 1 and random.random() < 0.5
    relay_count = random.randint(1, max_relays) if branched else 0

    cycle = random.sample(PALETTE, random.choice([3, 4]))
    source_color = random.choice(cycle)
    receiver_count = relay_count + 1
    receiver_pool = [c for c in PALETTE if c not in cycle]
    if receiver_count > len(receiver_pool):
        return None
    receiver_colors = random.sample(receiver_pool, receiver_count)

    return _render_attempt(
        h=h,
        w=w,
        orientation=orientation,
        source_at_min=random.choice([True, False]),
        cycle=cycle,
        source_color=source_color,
        receiver_colors=receiver_colors,
        corner=random.choice(["tl", "tr", "bl", "br"]),
        relay_count=relay_count,
    )


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-network ARC input")
