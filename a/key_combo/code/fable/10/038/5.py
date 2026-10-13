from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PIPE = 5
SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _simple_path_order(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert coords, "expected a non-empty pipe path"
    cells = set(coords)
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in coords:
        adjacency[cell] = [nbr for nbr in _neighbors4(*cell) if nbr in cells]
    degrees = {cell: len(nbrs) for cell, nbrs in adjacency.items()}
    assert all(deg in (1, 2) for deg in degrees.values()), "pipe must be branchless and one-cell wide"
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "pipe must have exactly two endpoints"

    order = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while True:
        nxts = [nbr for nbr in adjacency[cur] if nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "pipe walk must be deterministic"
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
    assert len(order) == len(coords), "pipe must be a single connected path"
    return order


def _orientation_info(grid: np.ndarray, order: list[tuple[int, int]], pipe_color: int):
    specials = [
        (idx, int(grid[r, c]))
        for idx, (r, c) in enumerate(order)
        if int(grid[r, c]) != pipe_color
    ]
    if len(specials) < 2:
        return None
    if specials[0][1] != specials[1][1]:
        return None

    seed_color = specials[0][1]
    first_seed = specials[0][0]
    second_seed = specials[1][0]
    period = second_seed - first_seed
    if period <= 0 or first_seed >= period:
        return None

    checkpoints: dict[int, int] = {}
    for idx, color in specials[2:]:
        if idx < second_seed:
            return None
        checkpoints[int(idx)] = int(color)

    return {
        "order": order,
        "seed_color": int(seed_color),
        "seed_indices": (int(first_seed), int(second_seed)),
        "period": int(period),
        "checkpoints": checkpoints,
    }


def _extract_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    mask = g != bg
    assert np.any(mask), "expected a visible pipe path"

    non_bg = [int(v) for v in g[mask]]
    pipe_color = Counter(non_bg).most_common(1)[0][0]
    order = _simple_path_order(mask)

    candidates = []
    for orient in (order, list(reversed(order))):
        info = _orientation_info(g, orient, pipe_color)
        if info is not None:
            candidates.append(info)
    assert len(candidates) == 1, "seed-side orientation must be uniquely determined"

    info = candidates[0]
    seed0, seed1 = info["seed_indices"]
    assert seed1 + info["period"] < len(info["order"]), "need at least one beat after the second seed"
    return {
        "grid": g,
        "bg": int(bg),
        "pipe_color": int(pipe_color),
        **info,
    }


def _render_output(scene: dict) -> np.ndarray:
    g = scene["grid"]
    out = g.copy()
    order = scene["order"]
    seed_color = scene["seed_color"]
    seed0, seed1 = scene["seed_indices"]
    period = scene["period"]
    checkpoints = scene["checkpoints"]

    current_color = int(seed_color)
    for idx in range(seed1 + 1, len(order)):
        if idx in checkpoints:
            current_color = int(checkpoints[idx])
        if (idx - seed1) % period == 0 and idx not in checkpoints:
            r, c = order[idx]
            out[r, c] = int(current_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint periodic beats along a gray snake pipe, handing the beat color off at each checkpoint.

    Input structure:
    - The background contains a one-cell-wide, 4-connected, branchless pipe whose cells form a single gray path with two endpoints.
    - Near one endpoint, two pipe cells already share one non-gray color; they are the first two beat seeds, and their path distance defines the period.
    - Additional single-cell checkpoints replace some later pipe cells with other colors; they may or may not lie on beat positions.

    Rule:
    - Unroll the pipe into a 1D path from the endpoint nearest the seeds.
    - Starting with the seed color, paint every later pipe cell whose path index stays congruent to the second seed modulo the period.
    - Whenever traversal reaches a checkpoint, immediately switch the current beat color to that checkpoint's color for all subsequent beat cells.
    - Checkpoints keep their own colors, non-beat pipe cells stay gray, and all off-pipe pixels remain unchanged.

    Key insight:
    - The geometry is only a carrier: the real pattern lives on the pipe's 1D traversal order, where the first two seeds reveal both the start side and the modular beat spacing.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    return _render_output(scene)


def _coarse_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((int(nr), int(nc)))
    return out


def _build_coarse_walk(rows: int, cols: int, length: int) -> list[tuple[int, int]] | None:
    for _ in range(120):
        start = (random.randrange(rows), random.randrange(cols))
        path = [start]
        visited = {start}

        def dfs() -> bool:
            if len(path) == length:
                return True
            nbrs = [nbr for nbr in _coarse_neighbors(path[-1], rows, cols) if nbr not in visited]
            random.shuffle(nbrs)
            nbrs.sort(key=lambda cell: sum(1 for nxt in _coarse_neighbors(cell, rows, cols) if nxt not in visited))
            for nbr in nbrs:
                path.append(nbr)
                visited.add(nbr)
                if dfs():
                    return True
                visited.remove(nbr)
                path.pop()
            return False

        if dfs():
            return path
    return None


def _expand_path(nodes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for i, (r, c) in enumerate(nodes):
        fr, fc = 2 * r + 1, 2 * c + 1
        if i == 0:
            order.append((fr, fc))
            continue
        pr, pc = order[-1]
        mr, mc = (pr + fr) // 2, (pc + fc) // 2
        order.append((mr, mc))
        order.append((fr, fc))
    return order


def _turn_count(nodes: list[tuple[int, int]]) -> int:
    if len(nodes) < 3:
        return 0
    turns = 0
    for i in range(1, len(nodes) - 1):
        a = (nodes[i][0] - nodes[i - 1][0], nodes[i][1] - nodes[i - 1][1])
        b = (nodes[i + 1][0] - nodes[i][0], nodes[i + 1][1] - nodes[i][1])
        if a != b:
            turns += 1
    return turns


def _valid_seed_params(path_len: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for period in range(2, min(8, path_len - 1)):
        for first_seed in range(0, min(period, 5)):
            if first_seed + 3 * period < path_len:
                pairs.append((int(first_seed), int(period)))
    return pairs


def _choose_checkpoint_positions(second_seed: int, period: int, path_len: int):
    future_beats = list(range(second_seed + period, path_len, period))
    if len(future_beats) < 2:
        return None
    beat_set = set(future_beats)
    available = list(range(second_seed + 1, path_len - 1))
    offbeat = [idx for idx in available if idx not in beat_set]
    before_first = [idx for idx in offbeat if idx < future_beats[0]]
    after_first = [idx for idx in offbeat if idx > future_beats[0]]
    onbeat_usable = future_beats[:-1]

    modes = []
    if before_first:
        modes.append("early")
    if after_first:
        modes.append("late")
    if onbeat_usable:
        modes.append("onbeat")
    if not modes:
        return None

    mode = random.choice(modes)
    max_count = min(4, len(available) - 1)
    if max_count < 1:
        return None

    positions: set[int] = set()
    if mode == "early":
        positions.add(random.choice(before_first))
    elif mode == "late":
        positions.add(random.choice(after_first))
    else:
        positions.add(random.choice(onbeat_usable))

    remaining = [idx for idx in available if idx not in positions]
    extra_count = random.randint(0, min(3, len(remaining)))
    while len(positions) < 1 + extra_count:
        positions.add(random.choice(remaining))
        remaining = [idx for idx in available if idx not in positions]

    checkpoints = sorted(positions)
    if all(idx in beat_set for idx in checkpoints):
        return None
    if not any(any(beat > idx for beat in future_beats) for idx in checkpoints):
        return None
    if not any(beat not in positions for beat in future_beats):
        return None
    return mode, checkpoints


def _planned_output(
    grid: np.ndarray,
    order: list[tuple[int, int]],
    seed_indices: tuple[int, int],
    period: int,
    checkpoint_colors: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    current_color = int(grid[order[seed_indices[0]]])
    for idx in range(seed_indices[1] + 1, len(order)):
        if idx in checkpoint_colors:
            current_color = int(checkpoint_colors[idx])
        if (idx - seed_indices[1]) % period == 0 and idx not in checkpoint_colors:
            r, c = order[idx]
            out[r, c] = int(current_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    coarse_rows = random.randint(5, 14)
    coarse_cols = random.randint(5, 14)
    max_nodes = min(coarse_rows * coarse_cols, 24)
    min_nodes = 9
    if max_nodes < min_nodes:
        return None

    node_count = random.randint(min_nodes, max_nodes)
    nodes = _build_coarse_walk(coarse_rows, coarse_cols, node_count)
    if nodes is None or _turn_count(nodes) < 2:
        return None

    order = _expand_path(nodes)
    path_len = len(order)
    params = _valid_seed_params(path_len)
    if not params:
        return None

    weighted = [pair for pair in params if pair[0] > 0]
    first_seed, period = random.choice(weighted or params)
    second_seed = first_seed + period

    checkpoint_choice = _choose_checkpoint_positions(second_seed, period, path_len)
    if checkpoint_choice is None:
        return None
    mode, checkpoint_positions = checkpoint_choice

    palette = random.sample(SPECIAL_COLORS, 1 + len(checkpoint_positions))
    seed_color = palette[0]
    checkpoint_colors = {
        int(idx): int(color)
        for idx, color in zip(checkpoint_positions, palette[1:])
    }

    h = 2 * coarse_rows + 1
    w = 2 * coarse_cols + 1
    grid = np.full((h, w), BG, dtype=int)
    for r, c in order:
        grid[r, c] = PIPE
    grid[order[first_seed]] = seed_color
    grid[order[second_seed]] = seed_color
    for idx, color in checkpoint_colors.items():
        r, c = order[idx]
        grid[r, c] = color

    expected = _planned_output(grid, order, (first_seed, second_seed), period, checkpoint_colors)

    scene = _extract_scene(grid)
    if scene["pipe_color"] != PIPE:
        return None
    if scene["seed_indices"] != (first_seed, second_seed):
        return None
    if scene["period"] != period:
        return None
    if scene["checkpoints"] != checkpoint_colors:
        return None
    if not np.array_equal(_render_output(scene), expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    beat_indices = list(range(second_seed + period, path_len, period))
    painted_future = [idx for idx in beat_indices if idx not in checkpoint_colors]
    if len(painted_future) < 1:
        return None

    if mode == "early":
        if checkpoint_positions[0] >= beat_indices[0]:
            return None
        first_painted = painted_future[0]
        if expected[order[first_painted]] != checkpoint_colors[checkpoint_positions[0]]:
            return None
    elif mode == "late":
        if checkpoint_positions[0] <= beat_indices[0]:
            return None
        if expected[order[beat_indices[0]]] != seed_color:
            return None
    else:
        onbeat = [idx for idx in checkpoint_positions if idx in beat_indices]
        if not onbeat:
            return None
        cp = onbeat[0]
        later = [idx for idx in painted_future if idx > cp]
        if not later:
            return None
        if expected[order[later[0]]] != checkpoint_colors[cp]:
            return None

    pad_top = random.randint(0, max(0, (30 - grid.shape[0]) // 2))
    pad_bottom = random.randint(0, 30 - grid.shape[0] - pad_top)
    pad_left = random.randint(0, max(0, (30 - grid.shape[1]) // 2))
    pad_right = random.randint(0, 30 - grid.shape[1] - pad_left)
    padded = np.pad(grid, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=BG)

    if max(padded.shape) > 30:
        return None
    if not np.array_equal(main(padded), np.pad(expected, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=BG)):
        return None
    return padded


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-beat input")
