from __future__ import annotations

import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --------------- solver helpers ---------------


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _inside(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _step(cell: tuple[int, int], direction: int) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return cell[0] + dr, cell[1] + dc


def _rot_cw(direction: int) -> int:
    return (direction + 1) % 4


def _rot_ccw(direction: int) -> int:
    return (direction - 1) % 4


def _opposite(direction: int) -> int:
    return (direction + 2) % 4


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _line_color_mask(grid: np.ndarray) -> np.ndarray:
    return np.isin(grid, [GREY, RED, BLUE])


def _parse_arrows(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray, set[tuple[int, int]]]:
    arrows: list[dict[str, object]] = []
    green_comps = _components(grid == GREEN)
    assert green_comps, "expected at least one green arrow"

    centers: set[tuple[int, int]] = set()
    prepared: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
    for comp in green_comps:
        assert len(comp) == 3, "green marker must have area 3"
        rows = {r for r, _c in comp}
        cols = {c for _r, c in comp}
        if len(rows) == 1:
            ordered = sorted(comp, key=lambda rc: rc[1])
        elif len(cols) == 1:
            ordered = sorted(comp, key=lambda rc: rc[0])
        else:
            raise AssertionError("green marker must be a straight 3-cell segment")
        center = ordered[1]
        centers.add(center)
        prepared.append((comp, center))

    network_mask = _line_color_mask(grid)
    for center in centers:
        network_mask[center] = True

    for comp, center in prepared:
        comp_set = set(comp)
        inward_dirs: list[int] = []
        for direction in range(4):
            nxt = _step(center, direction)
            if not _inside(grid.shape, nxt) or nxt in comp_set:
                continue
            if int(grid[nxt]) in (GREY, RED, BLUE):
                inward_dirs.append(direction)
        assert len(inward_dirs) == 1, "each arrow center must touch exactly one line cell"
        arrows.append({"center": center, "dir": inward_dirs[0]})

    return arrows, network_mask, centers


def _degree_map(network_mask: np.ndarray) -> np.ndarray:
    h, w = network_mask.shape
    degrees = np.zeros((h, w), dtype=int)
    for r, c in np.argwhere(network_mask):
        deg = 0
        for nr, nc in _neighbors4((int(r), int(c))):
            if 0 <= nr < h and 0 <= nc < w and network_mask[nr, nc]:
                deg += 1
        degrees[int(r), int(c)] = deg
    return degrees


def _trace_signal(
    grid: np.ndarray,
    network_mask: np.ndarray,
    degrees: np.ndarray,
    arrow_centers: set[tuple[int, int]],
    start: tuple[int, int],
    direction: int,
) -> dict[str, object]:
    pos = start
    cur_dir = direction
    last_node_color: int | None = None
    seen: set[tuple[int, int, int]] = set()

    while True:
        state = (pos[0], pos[1], cur_dir)
        assert state not in seen, "signal path must not cycle"
        seen.add(state)

        nxt = _step(pos, cur_dir)
        assert _inside(grid.shape, nxt) and network_mask[nxt], "signal left the line network"
        pos = nxt
        assert pos not in arrow_centers, "signal should not terminate at another green start arrow"

        color = int(grid[pos])
        if color == GREY:
            if int(degrees[pos]) == 1:
                assert last_node_color in (RED, BLUE), "terminal arrow color needs a prior node"
                return {"start": start, "end": pos, "dir": cur_dir, "color": last_node_color}
        elif color == RED:
            last_node_color = RED
            cur_dir = _rot_cw(cur_dir)
        elif color == BLUE:
            last_node_color = BLUE
            cur_dir = _rot_ccw(cur_dir)
        else:
            raise AssertionError("encountered invalid line color while tracing")


def _analyze_signals(grid: np.ndarray) -> list[dict[str, object]]:
    arrows, network_mask, arrow_centers = _parse_arrows(grid)
    degrees = _degree_map(network_mask)
    results = []
    for arrow in arrows:
        center = arrow["center"]
        direction = int(arrow["dir"])
        assert degrees[center] == 1, "green arrow center must sit on a line endpoint"
        results.append(_trace_signal(grid, network_mask, degrees, arrow_centers, center, direction))
    return results


def _arrow_wings(center: tuple[int, int], direction: int) -> list[tuple[int, int]]:
    if direction % 2 == 0:
        wing_dirs = [1, 3]
    else:
        wing_dirs = [0, 2]
    return [_step(center, wing_dir) for wing_dir in wing_dirs]


def _paint_terminal_arrow(out: np.ndarray, center: tuple[int, int], direction: int, color: int) -> None:
    out[center] = color
    for wing in _arrow_wings(center, direction):
        assert _inside(out.shape, wing), "terminal arrow wings must stay in bounds"
        out[wing] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate green arrow signals through a grey line network, rotating at red/blue
    nodes, then stamp a terminal T-arrow at the reached empty endpoint using the
    final node color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    seen_endpoints: set[tuple[int, int]] = set()
    for result in _analyze_signals(g):
        endpoint = result["end"]
        assert endpoint not in seen_endpoints, "two signals should not share a terminal endpoint"
        seen_endpoints.add(endpoint)
        _paint_terminal_arrow(out, endpoint, int(result["dir"]), int(result["color"]))
    return out


# --------------- generator helpers ---------------

BG_POOL = [0, 4, 6, 7, 8, 9]
GRID_H_RANGE = (14, 23)
GRID_W_RANGE = (14, 23)
PATH_COUNT_CHOICES = [1, 2, 3]
PATH_COUNT_WEIGHTS = [0.3, 0.45, 0.25]
SEG_LEN_RANGE = (2, 5)
BRANCH_LEN_RANGE = (1, 4)
PATH_RETRIES = 120
MAX_GENERATION_ATTEMPTS = 400
NODE_DECORATION_CHOICES = ["none", "entry", "exit_opp", "both"]
NODE_DECORATION_WEIGHTS = [0.45, 0.2, 0.2, 0.15]


def _sample_node_count(path_count: int) -> int:
    if path_count == 1:
        return random.choice([2, 3, 4])
    if path_count == 2:
        return random.choice([1, 2, 3])
    return random.choice([1, 2])


def _sample_turns(n_nodes: int) -> list[bool]:
    turns = [bool(random.getrandbits(1)) for _ in range(n_nodes)]  # True => cw/red
    if n_nodes >= 2 and len(set(turns)) == 1 and random.random() < 0.75:
        turns[random.randrange(n_nodes)] = not turns[0]
    return turns


def _guard_cells(center: tuple[int, int], direction: int, shape: tuple[int, int]) -> set[tuple[int, int]] | None:
    guards = {center}
    wings = _arrow_wings(center, direction)
    if not all(_inside(shape, wing) for wing in wings):
        return None
    for wing in wings:
        guards.add(wing)
        for nbr in _neighbors4(wing):
            if _inside(shape, nbr) and nbr != center:
                guards.add(nbr)
    return guards


def _free_start(
    shape: tuple[int, int],
    direction: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    h, w = shape
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(candidates)
    for start in candidates:
        if start in occupied or start in reserved:
            continue
        if any(nbr in occupied for nbr in _neighbors4(start)):
            continue
        guards = _guard_cells(start, direction, shape)
        if guards is None:
            continue
        if any(cell in occupied or cell in reserved for cell in guards if cell != start):
            continue
        return start, guards
    return None


def _can_add_network_cell(
    cell: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    allowed_neighbors: set[tuple[int, int]],
) -> bool:
    if not _inside(shape, cell) or cell in occupied or cell in reserved:
        return False
    for nbr in _neighbors4(cell):
        if nbr in occupied and nbr not in allowed_neighbors:
            return False
    return True


def _grow_branch(
    start: tuple[int, int],
    direction: int,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    lengths = list(range(BRANCH_LEN_RANGE[0], BRANCH_LEN_RANGE[1] + 1))
    random.shuffle(lengths)
    for length in lengths:
        cur = start
        branch: list[tuple[int, int]] = []
        branch_occupied = set(occupied)
        ok = True
        for _ in range(length):
            nxt = _step(cur, direction)
            if not _can_add_network_cell(nxt, shape, branch_occupied, reserved, {cur}):
                ok = False
                break
            branch.append(nxt)
            branch_occupied.add(nxt)
            cur = nxt
        if ok:
            return branch
    return None


def _generate_path(
    shape: tuple[int, int],
    global_occupied: set[tuple[int, int]],
    global_reserved: set[tuple[int, int]],
    path_count: int,
) -> dict[str, object] | None:
    for _ in range(PATH_RETRIES):
        start_dir = random.randrange(4)
        start_info = _free_start(shape, start_dir, global_occupied, global_reserved)
        if start_info is None:
            continue
        start, start_guards = start_info

        n_nodes = _sample_node_count(path_count)
        turns = _sample_turns(n_nodes)
        dirs = [start_dir]
        for is_cw in turns:
            dirs.append(_rot_cw(dirs[-1]) if is_cw else _rot_ccw(dirs[-1]))
        seg_lengths = [random.randint(*SEG_LEN_RANGE) for _ in range(n_nodes + 1)]

        occupied = set(global_occupied)
        reserved = set(global_reserved) | start_guards
        local_network = {start}
        path_cells = [start]
        node_records: list[dict[str, object]] = []
        cur = start
        ok = True

        for seg_idx, (direction, seg_len) in enumerate(zip(dirs, seg_lengths)):
            for _ in range(seg_len):
                nxt = _step(cur, direction)
                if not _can_add_network_cell(nxt, shape, occupied | local_network, reserved, {cur}):
                    ok = False
                    break
                local_network.add(nxt)
                path_cells.append(nxt)
                cur = nxt
            if not ok:
                break
            if seg_idx < n_nodes:
                outgoing = dirs[seg_idx + 1]
                node_records.append(
                    {
                        "cell": cur,
                        "incoming": direction,
                        "outgoing": outgoing,
                        "color": RED if outgoing == _rot_cw(direction) else BLUE,
                    }
                )
        if not ok:
            continue

        end = cur
        end_dir = dirs[-1]
        end_guards = _guard_cells(end, end_dir, shape)
        if end_guards is None:
            continue
        if any(cell in global_occupied or cell in global_reserved for cell in end_guards if cell != end):
            continue
        if any(cell in local_network and cell != end for cell in end_guards):
            continue

        branches: set[tuple[int, int]] = set()
        for node in node_records:
            decoration = random.choices(NODE_DECORATION_CHOICES, weights=NODE_DECORATION_WEIGHTS, k=1)[0]
            extra_dirs: list[int] = []
            if decoration in ("entry", "both"):
                extra_dirs.append(int(node["incoming"]))
            if decoration in ("exit_opp", "both"):
                extra_dirs.append(_opposite(int(node["outgoing"])))
            random.shuffle(extra_dirs)

            for extra_dir in extra_dirs:
                branch = _grow_branch(
                    start=node["cell"],
                    direction=extra_dir,
                    shape=shape,
                    occupied=global_occupied | local_network | branches,
                    reserved=global_reserved | start_guards | end_guards,
                )
                if branch is None:
                    continue
                branches.update(branch)

        all_network = local_network | branches
        if any(cell in end_guards and cell != end for cell in all_network):
            continue

        return {
            "start": start,
            "start_dir": start_dir,
            "end": end,
            "end_dir": end_dir,
            "path_cells": path_cells,
            "network_cells": all_network,
            "node_colors": {tuple(node["cell"]): int(node["color"]) for node in node_records},
            "last_color": int(node_records[-1]["color"]),
            "start_guards": start_guards,
            "end_guards": end_guards,
        }
    return None


def _render_grid(shape: tuple[int, int], bg: int, paths: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for path in paths:
        for cell in path["network_cells"]:
            grid[cell] = GREY
        for cell, color in path["node_colors"].items():
            grid[cell] = color
        start = path["start"]
        start_dir = int(path["start_dir"])
        grid[start] = GREEN
        for wing in _arrow_wings(start, start_dir):
            grid[wing] = GREEN
    return grid


def _validate_generated_grid(grid: np.ndarray, paths: list[dict[str, object]], bg: int) -> bool:
    if _get_bg(grid) != bg:
        return False

    specs = {
        tuple(path["start"]): {
            "end": tuple(path["end"]),
            "dir": int(path["end_dir"]),
            "color": int(path["last_color"]),
        }
        for path in paths
    }
    if len(specs) != len(paths):
        return False

    try:
        results = _analyze_signals(grid)
    except AssertionError:
        return False

    if len(results) != len(paths):
        return False
    if len({tuple(result["end"]) for result in results}) != len(results):
        return False

    by_start = {tuple(result["start"]): result for result in results}
    if set(by_start) != set(specs):
        return False
    for start, spec in specs.items():
        result = by_start[start]
        if tuple(result["end"]) != spec["end"]:
            return False
        if int(result["dir"]) != spec["dir"]:
            return False
        if int(result["color"]) != spec["color"]:
            return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    if out.shape != grid.shape:
        return False

    for spec in specs.values():
        if int(out[spec["end"]]) != spec["color"]:
            return False
        for wing in _arrow_wings(spec["end"], spec["dir"]):
            if int(out[wing]) != spec["color"]:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1]),
        random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1]),
    )
    bg = random.choice(BG_POOL)
    path_count = random.choices(PATH_COUNT_CHOICES, weights=PATH_COUNT_WEIGHTS, k=1)[0]

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    paths: list[dict[str, object]] = []

    for _ in range(path_count):
        path = _generate_path(shape, occupied, reserved, path_count)
        if path is None:
            return None
        paths.append(path)
        occupied |= set(path["network_cells"])
        reserved |= set(path["start_guards"]) | set(path["end_guards"])

    grid = _render_grid(shape, bg, paths)
    if not _validate_generated_grid(grid, paths, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 7")
