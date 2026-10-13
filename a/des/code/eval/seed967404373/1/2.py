from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np

BG = 0
PIPE = 5
RELAY = 9
SOURCE_PALETTE = [1, 2, 3, 4, 6, 7, 8]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400

Coord = tuple[int, int]


def _neighbors4(r: int, c: int, h: int, w: int) -> list[Coord]:
    out: list[Coord] = []
    for dr, dc in DIRS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _classify_scene(g: np.ndarray) -> tuple[set[Coord], set[Coord], dict[Coord, int], dict[Coord, int]]:
    h, w = g.shape
    pipes: set[Coord] = set()
    relays: set[Coord] = set()
    colored: dict[Coord, int] = {}
    for r in range(h):
        for c in range(w):
            value = int(g[r, c])
            if value == PIPE:
                pipes.add((r, c))
            elif value == RELAY:
                relays.add((r, c))
            elif value != BG:
                colored[(r, c)] = value

    sources: dict[Coord, int] = {}
    gates: dict[Coord, int] = {}
    structural = pipes | relays
    for coord, color in colored.items():
        relay_neighbors = [nbr for nbr in _neighbors4(coord[0], coord[1], h, w) if nbr in relays]
        structural_neighbors = [nbr for nbr in _neighbors4(coord[0], coord[1], h, w) if nbr in structural]
        if relay_neighbors:
            gates[coord] = color
        elif len(structural_neighbors) == 1:
            sources[coord] = color
        else:
            gates[coord] = color
    return pipes, relays, sources, gates


def _relay_gate_colors(relays: set[Coord], gates: dict[Coord, int]) -> dict[Coord, int]:
    relay_colors: dict[Coord, int] = {}
    for relay in relays:
        best_dist: int | None = None
        best_colors: list[int] = []
        for gate, color in gates.items():
            dist = abs(relay[0] - gate[0]) + abs(relay[1] - gate[1])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_colors = [color]
            elif dist == best_dist:
                best_colors.append(color)
        if best_dist is not None and best_dist <= 2:
            unique_colors = set(best_colors)
            if len(unique_colors) == 1:
                relay_colors[relay] = next(iter(unique_colors))
    return relay_colors


def _propagate_from_source(
    graph: dict[Coord, tuple[Coord, ...]],
    pipes: set[Coord],
    relay_colors: dict[Coord, int],
    source_positions: set[Coord],
    start: Coord,
    start_color: int,
) -> tuple[dict[Coord, int], dict[Coord, set[int]]]:
    sentinel = (-1, -1)
    queue = deque([(start, sentinel, start_color, 0)])
    seen: dict[tuple[Coord, Coord, int], int] = {(start, sentinel, start_color): 0}
    best_dist: dict[Coord, int] = {}
    best_colors: dict[Coord, set[int]] = defaultdict(set)

    while queue:
        pos, prev, color, dist = queue.popleft()
        if pos in pipes:
            current_best = best_dist.get(pos)
            if current_best is None or dist < current_best:
                best_dist[pos] = dist
                best_colors[pos] = {color}
            elif dist == current_best:
                best_colors[pos].add(color)

        if pos in source_positions and prev != sentinel:
            continue

        for nxt in graph[pos]:
            if nxt == prev:
                continue
            next_color = relay_colors.get(nxt, color)
            state = (nxt, pos, next_color)
            next_dist = dist + 1
            old = seen.get(state)
            if old is None or next_dist < old:
                seen[state] = next_dist
                queue.append((nxt, pos, next_color, next_dist))

    return best_dist, best_colors


def _transform(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pipes, relays, sources, gates = _classify_scene(g)
    if not pipes or not sources:
        return g.copy()

    h, w = g.shape
    network = pipes | relays | set(sources)
    graph = {
        coord: tuple(nbr for nbr in _neighbors4(coord[0], coord[1], h, w) if nbr in network)
        for coord in network
    }
    relay_colors = _relay_gate_colors(relays, gates)
    source_positions = set(sources)

    arrivals: dict[Coord, list[tuple[int, set[int]]]] = defaultdict(list)
    for source in sorted(sources):
        dist_map, color_map = _propagate_from_source(
            graph=graph,
            pipes=pipes,
            relay_colors=relay_colors,
            source_positions=source_positions,
            start=source,
            start_color=int(sources[source]),
        )
        for cell, dist in dist_map.items():
            arrivals[cell].append((dist, color_map[cell]))

    out = g.copy()
    for cell in pipes:
        options = arrivals.get(cell, [])
        if not options:
            continue
        best_dist = min(dist for dist, _colors in options)
        winners = [colors for dist, colors in options if dist == best_dist]
        if len(winners) == 1 and len(winners[0]) == 1:
            out[cell] = next(iter(winners[0]))
        else:
            out[cell] = PIPE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate every source signal through the grey pipe network, update the carried
    color when a relay sees a nearby gate, and color each grey pipe cell by the
    earliest-arriving source; equal-time conflicts stay grey.
    """
    return _transform(np.asarray(grid, dtype=int))


def _sample_spaced_numbers(candidates: list[int], count: int, min_gap: int, rng: random.Random) -> list[int] | None:
    for _ in range(80):
        pool = candidates[:]
        rng.shuffle(pool)
        chosen: list[int] = []
        for value in pool:
            if all(abs(value - other) >= min_gap for other in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _sample_spaced_cells(
    candidates: list[Coord], count: int, min_dist: int, rng: random.Random
) -> list[Coord] | None:
    for _ in range(80):
        pool = candidates[:]
        rng.shuffle(pool)
        chosen: list[Coord] = []
        for cell in pool:
            if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= min_dist for other in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _degrees(network: set[Coord], h: int, w: int) -> dict[Coord, int]:
    return {
        cell: sum(1 for nbr in _neighbors4(cell[0], cell[1], h, w) if nbr in network)
        for cell in network
    }


def _is_tree(network: set[Coord], h: int, w: int) -> bool:
    if not network:
        return False
    start = next(iter(network))
    seen = {start}
    queue = deque([start])
    edge_count = 0
    while queue:
        cell = queue.popleft()
        neighbors = [nbr for nbr in _neighbors4(cell[0], cell[1], h, w) if nbr in network]
        edge_count += len(neighbors)
        for nbr in neighbors:
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    edge_count //= 2
    return len(seen) == len(network) and edge_count == len(network) - 1


def _build_layout(
    h: int,
    w: int,
    orientation: str,
    branch_count: int,
    force_tie: bool,
    rng: random.Random,
) -> tuple[set[Coord], list[Coord], dict[str, object]] | None:
    path_cells: set[Coord] = set()
    sources: list[Coord] = []
    meta: dict[str, object] = {"orientation": orientation}

    if orientation == "horizontal":
        row = rng.randint(4, h - 5)
        min_span = max(6, branch_count * 2 + 4)
        max_span = min(w - 3, 12)
        parity = 0 if force_tie else 1
        spans = [span for span in range(min_span, max_span + 1) if span % 2 == parity]
        if not spans:
            return None
        span = rng.choice(spans)
        c0 = rng.randint(1, w - span - 2)
        c1 = c0 + span
        sources.extend([(row, c0), (row, c1)])
        for c in range(c0 + 1, c1):
            path_cells.add((row, c))

        midpoint = (row, (c0 + c1) // 2) if force_tie else None
        half_span = span // 2
        choices = list(range(c0 + 2, c1 - 1))
        if midpoint is not None:
            mid_col = midpoint[1]
            choices = [c for c in choices if abs(c - mid_col) >= 2]
        attachments = _sample_spaced_numbers(choices, branch_count, 2, rng)
        if attachments is None:
            return None

        branch_specs = []
        for col in attachments:
            options: list[tuple[int, int, int]] = []
            for side in (-1, 1):
                max_len = row - 1 if side == -1 else h - row - 2
                min_len = 2
                if midpoint is not None:
                    min_len = max(min_len, half_span - abs(col - midpoint[1]) + 1)
                if max_len >= min_len:
                    options.append((side, min_len, max_len))
            if not options:
                return None
            side, min_len, max_len = rng.choice(options)
            upper = max_len if min_len > 5 else min(max_len, 5)
            if upper < min_len:
                upper = max_len
            length = rng.randint(min_len, upper)
            for step in range(1, length):
                path_cells.add((row + side * step, col))
            source = (row + side * length, col)
            sources.append(source)
            branch_specs.append((col, side, length))

        meta["midpoint"] = midpoint
        meta["branch_specs"] = branch_specs
        return path_cells, sources, meta

    col = rng.randint(4, w - 5)
    min_span = max(6, branch_count * 2 + 4)
    max_span = min(h - 3, 12)
    parity = 0 if force_tie else 1
    spans = [span for span in range(min_span, max_span + 1) if span % 2 == parity]
    if not spans:
        return None
    span = rng.choice(spans)
    r0 = rng.randint(1, h - span - 2)
    r1 = r0 + span
    sources.extend([(r0, col), (r1, col)])
    for r in range(r0 + 1, r1):
        path_cells.add((r, col))

    midpoint = ((r0 + r1) // 2, col) if force_tie else None
    half_span = span // 2
    choices = list(range(r0 + 2, r1 - 1))
    if midpoint is not None:
        mid_row = midpoint[0]
        choices = [r for r in choices if abs(r - mid_row) >= 2]
    attachments = _sample_spaced_numbers(choices, branch_count, 2, rng)
    if attachments is None:
        return None

    branch_specs = []
    for row in attachments:
        options = []
        for side in (-1, 1):
            max_len = col - 1 if side == -1 else w - col - 2
            min_len = 2
            if midpoint is not None:
                min_len = max(min_len, half_span - abs(row - midpoint[0]) + 1)
            if max_len >= min_len:
                options.append((side, min_len, max_len))
        if not options:
            return None
        side, min_len, max_len = rng.choice(options)
        upper = max_len if min_len > 5 else min(max_len, 5)
        if upper < min_len:
            upper = max_len
        length = rng.randint(min_len, upper)
        for step in range(1, length):
            path_cells.add((row, col + side * step))
        source = (row, col + side * length)
        sources.append(source)
        branch_specs.append((row, side, length))

    meta["midpoint"] = midpoint
    meta["branch_specs"] = branch_specs
    return path_cells, sources, meta


def _gate_slots(relay: Coord, network: set[Coord], occupied_gates: set[Coord], h: int, w: int) -> list[Coord]:
    slots: list[Coord] = []
    for cell in _neighbors4(relay[0], relay[1], h, w):
        if cell in network or cell in occupied_gates:
            continue
        touching = [nbr for nbr in _neighbors4(cell[0], cell[1], h, w) if nbr in network]
        if len(touching) == 1 and touching[0] == relay:
            slots.append(cell)
    return slots


def _build_grid(
    h: int,
    w: int,
    grey_cells: set[Coord],
    relays: set[Coord],
    sources: dict[Coord, int],
    gates: dict[Coord, int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r, c in grey_cells:
        grid[r, c] = PIPE
    for r, c in relays:
        grid[r, c] = RELAY
    for (r, c), color in sources.items():
        grid[r, c] = int(color)
    for (r, c), color in gates.items():
        grid[r, c] = int(color)
    return grid


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 18)
    w = rng.randint(12, 18)
    orientation = rng.choice(["horizontal", "vertical"])
    branch_count = rng.randint(1, 3)
    force_tie = rng.random() < 0.5

    layout = _build_layout(h, w, orientation, branch_count, force_tie, rng)
    if layout is None:
        return None
    path_cells, source_positions, meta = layout

    if len(source_positions) > len(SOURCE_PALETTE):
        return None
    if len(set(source_positions)) != len(source_positions):
        return None

    source_colors = rng.sample(SOURCE_PALETTE, len(source_positions))
    source_map = {coord: color for coord, color in zip(source_positions, source_colors)}
    network = set(path_cells) | set(source_map)
    if not _is_tree(network, h, w):
        return None

    midpoint = meta.get("midpoint")
    degrees = _degrees(network, h, w)
    relay_candidates = [
        cell
        for cell in path_cells
        if degrees[cell] == 2
        and all(nbr not in source_map for nbr in _neighbors4(cell[0], cell[1], h, w))
        and cell != midpoint
    ]
    if not relay_candidates:
        return None

    if len(relay_candidates) >= 2 and rng.random() < 0.65:
        relay_count = rng.randint(2, min(3, len(relay_candidates)))
    else:
        relay_count = 1
    relays = _sample_spaced_cells(relay_candidates, relay_count, 3, rng)
    if relays is None:
        return None
    relays = set(relays)
    grey_cells = set(path_cells) - relays

    if relay_count == 1:
        gated_relays = set(relays) if rng.random() < 0.5 else set()
    else:
        gated_count = rng.randint(1, relay_count - 1)
        gated_relays = set(rng.sample(sorted(relays), gated_count))

    gates: dict[Coord, int] = {}
    for relay in sorted(gated_relays):
        slots = _gate_slots(relay, network, set(gates), h, w)
        if not slots:
            return None
        gate = rng.choice(slots)
        gates[gate] = rng.choice(source_colors)

    grid = _build_grid(h, w, grey_cells, relays, source_map, gates)

    det_pipes, det_relays, det_sources, det_gates = _classify_scene(grid)
    if det_pipes != grey_cells or det_relays != relays or det_sources != source_map or det_gates != gates:
        return None

    out = _transform(grid)
    painted_colors = {int(out[r, c]) for r, c in grey_cells if int(out[r, c]) != PIPE}
    if len(painted_colors) < 2:
        return None
    if not np.any((grid == PIPE) & (out != PIPE)):
        return None

    if midpoint is not None and int(out[midpoint]) != PIPE:
        return None

    if gates:
        no_gate_grid = grid.copy()
        for gate in gates:
            no_gate_grid[gate] = BG
        if np.array_equal(out, _transform(no_gate_grid)):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
