from __future__ import annotations

from collections import deque
from itertools import permutations

import numpy as np


BG = 0
PIPE = 5
SLOT = 9
BLOCK_COLORS = [1, 2, 3, 4, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _find_components(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    excluded = set() if exclude is None else {int(x) for x in exclude}
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color in excluded:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(cr, cc, g.shape):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _extract_slot(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for comp in _find_components(g, exclude={BG}):
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        color = int(comp["color"])
        hh, ww = patch.shape
        if hh < 3 or ww < 3:
            continue
        if not np.all(patch[0, :] == color) or not np.all(patch[-1, :] == color):
            continue
        if not np.all(patch[:, 0] == color) or not np.all(patch[:, -1] == color):
            continue
        inner = patch[1:-1, 1:-1]
        if inner.size == 0 or not np.all(inner == BG):
            continue
        if comp["area"] != int(hh * ww - inner.size):
            continue
        if inner.shape[0] == 1 and 3 <= inner.shape[1] <= 5:
            fill = [(r0 + 1, c0 + 1 + i) for i in range(inner.shape[1])]
            orientation = "horizontal"
        elif inner.shape[1] == 1 and 3 <= inner.shape[0] <= 5:
            fill = [(r0 + 1 + i, c0 + 1) for i in range(inner.shape[0])]
            orientation = "vertical"
        else:
            continue
        candidates.append(
            {
                "color": color,
                "coords": set(comp["coords"]),
                "bbox": comp["bbox"],
                "fill": fill,
                "orientation": orientation,
            }
        )
    assert len(candidates) == 1, "expected exactly one slot"
    return candidates[0]


def _is_solid_square(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h not in {2, 3}:
        return False
    if comp["area"] != side_h * side_w:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])))


def _extract_blocks(grid: np.ndarray, slot_color: int) -> list[dict]:
    blocks = [comp for comp in _find_components(grid, exclude={BG, slot_color}) if _is_solid_square(grid, comp)]
    assert 2 <= len(blocks) <= 5, "expected 2-5 blocks"
    colors = [int(comp["color"]) for comp in blocks]
    assert len(set(colors)) == len(colors), "block colors must be distinct"
    return blocks


def _build_network(grid: np.ndarray, slot: dict, blocks: list[dict]) -> tuple[set[tuple[int, int]], dict, dict]:
    h, w = grid.shape
    block_cells = {coord for comp in blocks for coord in comp["coords"]}
    network_cells = set()
    valve_colors: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or (r, c) in slot["coords"] or (r, c) in block_cells:
                continue
            network_cells.add((r, c))
            if color != PIPE:
                valve_colors[(r, c)] = color

    graph = {cell: [] for cell in network_cells}
    for r, c in network_cells:
        for nr, nc in _neighbors4(r, c, grid.shape):
            if (nr, nc) in network_cells:
                graph[(r, c)].append((nr, nc))
    return network_cells, graph, valve_colors


def _slot_entry(slot: dict, network_cells: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int]:
    adjacent = set()
    for r, c in slot["coords"]:
        for nr, nc in _neighbors4(r, c, shape):
            if (nr, nc) in network_cells:
                adjacent.add((nr, nc))
    assert len(adjacent) == 1, "slot must have exactly one pipe connection"
    return next(iter(adjacent))


def _dist_to_slot(graph: dict, slot_entry: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {slot_entry: 0}
    q = deque([slot_entry])
    while q:
        cell = q.popleft()
        for nxt in graph[cell]:
            if nxt in dist:
                continue
            dist[nxt] = dist[cell] + 1
            q.append(nxt)
    return dist


def _assign_blocks_to_endpoints(blocks: list[dict], endpoints: list[tuple[int, int]]) -> list[tuple[dict, tuple[int, int]]]:
    assert len(blocks) == len(endpoints), "block/open-end count mismatch"
    distance = []
    for block in blocks:
        coords = block["coords"]
        distance.append(
            [
                min(abs(er - br) + abs(ec - bc) for br, bc in coords)
                for er, ec in endpoints
            ]
        )

    best = None
    best_perm = None
    for perm in permutations(range(len(endpoints))):
        total = sum(distance[i][perm[i]] for i in range(len(blocks)))
        key = (total, tuple(perm))
        if best is None or key < best:
            best = key
            best_perm = perm
    assert best_perm is not None

    pairs = []
    for block_index, endpoint_index in enumerate(best_perm):
        pairs.append((blocks[block_index], endpoints[endpoint_index]))
    return pairs


def _best_progress_choice(
    current: tuple[int, int],
    previous: tuple[int, int] | None,
    candidates: list[tuple[int, int]],
    dist: dict[tuple[int, int], int],
) -> tuple[int, int]:
    straight = None
    if previous is not None:
        dr = current[0] - previous[0]
        dc = current[1] - previous[1]
        trial = (current[0] + dr, current[1] + dc)
        if trial in candidates:
            straight = trial
    return min(
        candidates,
        key=lambda cell: (dist.get(cell, 10**9), 0 if cell == straight else 1, cell[0], cell[1]),
    )


def _trace_length(
    start: tuple[int, int],
    slot_entry: tuple[int, int],
    token_color: int,
    graph: dict[tuple[int, int], list[tuple[int, int]]],
    valve_colors: dict[tuple[int, int], int],
    dist: dict[tuple[int, int], int],
) -> int:
    previous = None
    current = start
    steps = 0
    seen = set()
    limit = max(10, len(graph) * 4)

    while current != slot_entry:
        state = (previous, current)
        assert state not in seen, "routing loop detected"
        seen.add(state)

        candidates = [nxt for nxt in graph[current] if nxt != previous]
        assert candidates, "route hit a dead end"

        if current in valve_colors and previous is not None:
            dr = current[0] - previous[0]
            dc = current[1] - previous[1]
            straight = (current[0] + dr, current[1] + dc)
            turn_candidates = [cell for cell in candidates if cell != straight]
            if valve_colors[current] == token_color and turn_candidates:
                nxt = min(
                    turn_candidates,
                    key=lambda cell: (dist.get(cell, 10**9), cell[0], cell[1]),
                )
            elif straight in candidates:
                nxt = straight
            else:
                nxt = _best_progress_choice(current, previous, candidates, dist)
        else:
            nxt = _best_progress_choice(current, previous, candidates, dist)

        previous, current = current, nxt
        steps += 1
        assert steps <= limit, "route exceeded step budget"

    return steps


def main(grid: np.ndarray) -> np.ndarray:
    """Route colored blocks through a pipe network into the maroon slot.

    Input structure:
    - Black background.
    - A one-pixel-wide orthogonal pipe network, mostly gray, with one maroon
      rectangular slot whose interior is black.
    - Several monochrome 2x2 or 3x3 blocks near open pipe ends.
    - Some junction centers are recolored as single-pixel valves.

    Rule:
    - For each block, trace from its nearby open end toward the slot.
    - At a valve, if the valve color matches the block color, take the turning
      branch; otherwise keep going straight when possible.
    - Erase the original blocks and place one 1x1 pixel per block into the slot,
      ordered by total traveled pipe length from short to long.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    slot = _extract_slot(g)
    blocks = _extract_blocks(g, slot["color"])
    network_cells, graph, valve_colors = _build_network(g, slot, blocks)
    entry = _slot_entry(slot, network_cells, g.shape)
    dist = _dist_to_slot(graph, entry)

    endpoints = [cell for cell, nbrs in graph.items() if len(nbrs) == 1 and cell != entry]
    pairs = _assign_blocks_to_endpoints(blocks, endpoints)

    routed = []
    for block, start in pairs:
        assert start in dist, "open end must connect to the slot"
        length = _trace_length(start, entry, int(block["color"]), graph, valve_colors, dist)
        routed.append((length, start, block))

    routed.sort(key=lambda item: (item[0], item[1][0], item[1][1], item[2]["bbox"]))
    assert len(routed) <= len(slot["fill"]), "slot interior too small"

    out = g.copy()
    for _length, _start, block in routed:
        for r, c in block["coords"]:
            out[r, c] = BG

    for (r, c), (_length, _start, block) in zip(slot["fill"], routed):
        out[r, c] = int(block["color"])
    for r, c in slot["fill"][len(routed) :]:
        out[r, c] = BG

    return out


def _place_square(grid: np.ndarray, top: int, left: int, size: int, color: int) -> None:
    grid[top : top + size, left : left + size] = int(color)


def _draw_slot(grid: np.ndarray, trunk_row: int, slot_len: int) -> tuple[int, int]:
    grid[trunk_row - 1 : trunk_row + 2, 0 : slot_len + 2] = SLOT
    grid[trunk_row, 1 : slot_len + 1] = BG
    return trunk_row, slot_len + 2


def _draw_detour(grid: np.ndarray, trunk_row: int, rejoin_col: int, valve_col: int, depth: int) -> None:
    top_row = trunk_row - depth
    grid[top_row : trunk_row + 1, rejoin_col] = PIPE
    grid[top_row : trunk_row + 1, valve_col] = PIPE
    grid[top_row, rejoin_col : valve_col + 1] = PIPE


def _transform_grid(grid: np.ndarray, rotate_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, int(rotate_k))
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _expected_order(block_specs: list[dict], entry_col: int, valve_specs: list[dict]) -> list[int]:
    routed = []
    for spec in block_specs:
        extra = sum(2 * valve["depth"] for valve in valve_specs if valve["color"] == spec["color"])
        length = int(spec["stub_len"] + (spec["attach_col"] - entry_col) + extra)
        routed.append((length, spec["color"]))
    lengths = [item[0] for item in routed]
    if len(set(lengths)) != len(lengths):
        return []
    routed.sort()
    return [color for _length, color in routed]


def _make_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_blocks = int(rng.integers(2, 6))
    slot_len = int(rng.integers(max(3, n_blocks), 6))
    n_valves = int(rng.integers(1, min(3, n_blocks) + 1))

    block_colors = rng.choice(BLOCK_COLORS, size=n_blocks, replace=False).astype(int).tolist()
    block_sizes = rng.choice([2, 3], size=n_blocks, replace=True).astype(int).tolist()
    stub_lens = rng.integers(2, 6, size=n_blocks).astype(int).tolist()
    anchor_sides = rng.integers(0, 2, size=n_blocks).astype(int).tolist()

    for _ in range(80):
        valve_colors = [block_colors[int(i)] for i in rng.integers(0, n_blocks, size=n_valves)]
        matched = {color for color in valve_colors}
        if 0 < len(matched) < n_blocks:
            break
    else:
        return None

    valve_specs = []
    cursor = slot_len + 5
    for _ in range(n_valves):
        span = int(rng.integers(3, 7))
        depth = int(rng.integers(2, 5))
        rejoin_col = cursor
        valve_col = rejoin_col + span
        valve_specs.append({"rejoin_col": rejoin_col, "valve_col": valve_col, "depth": depth})
        cursor = valve_col + int(rng.integers(2, 5))

    block_specs = []
    cursor += int(rng.integers(2, 5))
    for idx in range(n_blocks):
        size = block_sizes[idx]
        block_left = cursor
        attach_col = block_left if anchor_sides[idx] == 0 else block_left + size - 1
        block_specs.append(
            {
                "color": int(block_colors[idx]),
                "size": int(size),
                "stub_len": int(stub_lens[idx]),
                "attach_col": int(attach_col),
                "block_left": int(block_left),
            }
        )
        cursor = block_left + size + int(rng.integers(3, 7))

    max_depth = max(spec["depth"] for spec in valve_specs)
    trunk_row = max_depth + 2
    block_bottom = max(trunk_row + spec["stub_len"] + 1 + spec["size"] for spec in block_specs)
    width = max(spec["block_left"] + spec["size"] for spec in block_specs) + 1
    grid = np.full((block_bottom + 1, width), BG, dtype=int)

    _slot_center_row, entry_col = _draw_slot(grid, trunk_row, slot_len)
    trunk_end = max(spec["attach_col"] for spec in block_specs)
    grid[trunk_row, entry_col : trunk_end + 1] = PIPE

    for valve, color in zip(valve_specs, valve_colors):
        _draw_detour(grid, trunk_row, valve["rejoin_col"], valve["valve_col"], valve["depth"])
        grid[trunk_row, valve["valve_col"]] = int(color)
        valve["color"] = int(color)

    for spec in block_specs:
        attach_col = spec["attach_col"]
        endpoint_row = trunk_row + spec["stub_len"]
        grid[trunk_row : endpoint_row + 1, attach_col] = PIPE
        block_top = endpoint_row + 2
        _place_square(grid, block_top, spec["block_left"], spec["size"], spec["color"])

    expected = _expected_order(block_specs, entry_col, valve_specs)
    if not expected:
        return None

    top_pad = int(rng.integers(1, 5))
    bottom_pad = int(rng.integers(1, 5))
    left_pad = int(rng.integers(1, 5))
    right_pad = int(rng.integers(1, 5))
    padded = np.full((grid.shape[0] + top_pad + bottom_pad, grid.shape[1] + left_pad + right_pad), BG, dtype=int)
    padded[top_pad : top_pad + grid.shape[0], left_pad : left_pad + grid.shape[1]] = grid

    transformed = _transform_grid(padded, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))

    try:
        slot = _extract_slot(transformed)
        blocks = _extract_blocks(transformed, slot["color"])
        if len(blocks) != n_blocks:
            return None
        out = main(transformed)
    except Exception:
        return None

    fill_colors = [int(out[r, c]) for r, c in slot["fill"]]
    if fill_colors[: len(expected)] != expected:
        return None
    if any(color != BG for color in fill_colors[len(expected) :]):
        return None
    if any(int(out[r, c]) != BG for comp in blocks for r, c in comp["coords"]):
        return None

    return transformed


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-routing ARC input")
