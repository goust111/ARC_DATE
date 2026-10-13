from __future__ import annotations

from collections import deque
import math
import random

import numpy as np


BG = 0
CHANNEL = 5
MOVER = 1
TARGET = 2

HRECT = (2, 3)
VRECT = (3, 2)
DIR_DELTA = {"R": (0, 1), "L": (0, -1), "D": (1, 0), "U": (-1, 0)}
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _solid_rect_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    mask = grid == int(color)
    box = _bbox(mask)
    if box is None:
        raise AssertionError(f"missing color {color}")
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch == int(color)):
        raise AssertionError(f"color {color} is not a solid rectangle")
    if int(mask.sum()) != patch.size:
        raise AssertionError(f"color {color} must form one solid rectangle")
    return r0, c0, r1, c1


def _dims_from_box(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _rect_mask(h: int, w: int, r0: int, c0: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=bool)
    out[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = True
    return out


def _state_cells(state: tuple[int, int, int], dims_pair: tuple[tuple[int, int], tuple[int, int]]) -> set[tuple[int, int]]:
    r, c, ori = state
    h, w = dims_pair[int(ori)]
    return {(rr, cc) for rr in range(int(r), int(r + h)) for cc in range(int(c), int(c + w))}


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = int(color)


def _feasible_positions(traversable: np.ndarray, dims: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = dims
    H, W = traversable.shape
    out: set[tuple[int, int]] = set()
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            if bool(traversable[r : r + h, c : c + w].all()):
                out.add((int(r), int(c)))
    return out


def _rotation_targets(
    state: tuple[int, int, int],
    dims_pair: tuple[tuple[int, int], tuple[int, int]],
    feasible0: set[tuple[int, int]],
    feasible1: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    r, c, ori = state
    h, w = dims_pair[int(ori)]
    hh, ww = dims_pair[1 - int(ori)]
    center2_r = int(2 * r + h - 1)
    center2_c = int(2 * c + w - 1)
    base_r = (center2_r - (hh - 1)) / 2.0
    base_c = (center2_c - (ww - 1)) / 2.0
    cand_rs = {math.floor(base_r), math.ceil(base_r)}
    cand_cs = {math.floor(base_c), math.ceil(base_c)}
    feasible = feasible1 if int(ori) == 0 else feasible0

    current_cells = _state_cells(state, dims_pair)
    best_key: tuple[int, int, int, int] | None = None
    best: list[tuple[int, int, int]] = []
    for rr in cand_rs:
        for cc in cand_cs:
            pos = (int(rr), int(cc))
            if pos not in feasible:
                continue
            center_dev = abs(int(2 * rr + hh - 1) - center2_r) + abs(int(2 * cc + ww - 1) - center2_c)
            other_cells = {
                (r2, c2)
                for r2 in range(int(rr), int(rr + hh))
                for c2 in range(int(cc), int(cc + ww))
            }
            overlap = len(current_cells & other_cells)
            key = (int(center_dev), int(-overlap), int(rr), int(cc))
            if best_key is None or key < best_key:
                best_key = key
                best = [(int(rr), int(cc), 1 - int(ori))]
            elif key == best_key:
                best.append((int(rr), int(cc), 1 - int(ori)))
    return best


def _build_state_graph(
    traversable: np.ndarray,
    dims_pair: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[set[tuple[int, int, int]], dict[tuple[int, int, int], set[tuple[int, int, int]]]]:
    feasible0 = _feasible_positions(traversable, dims_pair[0])
    feasible1 = _feasible_positions(traversable, dims_pair[1])
    states = {(r, c, 0) for r, c in feasible0} | {(r, c, 1) for r, c in feasible1}
    adj = {state: set() for state in states}

    for r, c, ori in states:
        for dr, dc in DIR_DELTA.values():
            nxt = (int(r + dr), int(c + dc), int(ori))
            if nxt in states:
                adj[(r, c, ori)].add(nxt)
        for nxt in _rotation_targets((r, c, ori), dims_pair, feasible0, feasible1):
            if nxt in states:
                adj[(r, c, ori)].add(nxt)
                adj[nxt].add((r, c, ori))
    return states, adj


def _bfs_distances(
    adj: dict[tuple[int, int, int], set[tuple[int, int, int]]],
    start: tuple[int, int, int],
) -> tuple[dict[tuple[int, int, int], int], dict[tuple[int, int, int], tuple[int, int, int] | None]]:
    dist = {start: 0}
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in sorted(adj[cur]):
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            parent[nxt] = cur
            queue.append(nxt)
    return dist, parent


def _shortest_path_count(
    adj: dict[tuple[int, int, int], set[tuple[int, int, int]]],
    start: tuple[int, int, int],
    target: tuple[int, int, int],
) -> tuple[int | None, int]:
    dist = {start: 0}
    ways = {start: 1}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in sorted(adj[cur]):
            nd = dist[cur] + 1
            if nxt not in dist:
                dist[nxt] = nd
                ways[nxt] = ways[cur]
                queue.append(nxt)
            elif dist[nxt] == nd:
                ways[nxt] = min(2, ways[nxt] + ways[cur])
    if target not in dist:
        return None, 0
    return dist[target], ways[target]


def _path_from_parent(
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None],
    target: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    path: list[tuple[int, int, int]] = []
    cur: tuple[int, int, int] | None = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _render_output(
    grid: np.ndarray,
    start_state: tuple[int, int, int],
    final_state: tuple[int, int, int],
    target_state: tuple[int, int, int],
    dims_pair: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    sr, sc, sori = start_state
    sh, sw = dims_pair[int(sori)]
    _paint_rect(out, sr, sc, sh, sw, CHANNEL)
    if final_state == target_state:
        tr, tc, tori = target_state
        th, tw = dims_pair[int(tori)]
        _paint_rect(out, tr, tc, th, tw, CHANNEL)
    fr, fc, fori = final_state
    fh, fw = dims_pair[int(fori)]
    _paint_rect(out, fr, fc, fh, fw, MOVER)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a non-square rectangle through a winding channel, rotating at narrow turns so it can cover the target.

    Input structure:
    - Background is black.
    - Grey pixels form one unbranched orthogonal channel whose width alternates between
      straight 2-cell corridors and larger turn pockets.
    - One blue solid rectangle is the mover and one red solid rectangle is the target;
      they have the same area and are both fully inside the grey channel.

    Rule:
    - Treat every top-left placement where the whole rectangle fits in grey/blue/red as a valid state.
    - The rectangle may translate by one cell in the channel while keeping orientation, or rotate 90 degrees
      to the best center-preserving placement of the transposed rectangle when the turn pocket allows it.
    - Follow the connected state graph from the blue start toward the red target; if the target state is
      reachable, place blue there, otherwise stop at the farthest reachable state. Replace the original blue
      position with grey, and leave the red target only when it is not reached.

    Key insight:
    The channel geometry itself encodes when rotation is mandatory: straight corridor placements allow only one
    orientation, while the asymmetric turn pocket admits exactly one center-consistent rotated state.
    """
    g = np.asarray(grid, dtype=int)
    mover_box = _solid_rect_bbox(g, MOVER)
    target_box = _solid_rect_bbox(g, TARGET)
    mover_dims = _dims_from_box(mover_box)
    target_dims = _dims_from_box(target_box)
    if mover_dims[0] * mover_dims[1] != target_dims[0] * target_dims[1]:
        raise AssertionError("mover and target must have equal area")
    if mover_dims[0] == mover_dims[1]:
        raise AssertionError("mover must be non-square")

    dims_pair = (mover_dims, (int(mover_dims[1]), int(mover_dims[0])))
    if target_dims not in dims_pair:
        raise AssertionError("target orientation must match the mover or its 90-degree rotation")

    traversable = (g == CHANNEL) | (g == MOVER) | (g == TARGET)
    states, adj = _build_state_graph(traversable, dims_pair)
    start_state = (int(mover_box[0]), int(mover_box[1]), 0)
    target_ori = 0 if target_dims == dims_pair[0] else 1
    target_state = (int(target_box[0]), int(target_box[1]), int(target_ori))
    if start_state not in states or target_state not in states:
        raise AssertionError("start/target placement must be feasible inside the channel")

    dist, _parent = _bfs_distances(adj, start_state)
    if target_state in dist:
        final_state = target_state
    else:
        target_center = (
            target_box[0] + (target_dims[0] - 1) / 2.0,
            target_box[1] + (target_dims[1] - 1) / 2.0,
        )

        def key(state: tuple[int, int, int]) -> tuple[int, float, int, int, int]:
            r, c, ori = state
            h, w = dims_pair[int(ori)]
            center = (r + (h - 1) / 2.0, c + (w - 1) / 2.0)
            center_dist = abs(center[0] - target_center[0]) + abs(center[1] - target_center[1])
            return (dist[state], -center_dist, -r, -c, -ori)

        final_state = max(dist, key=key)
    return _render_output(g, start_state, final_state, target_state, dims_pair)


def _move_state(state: tuple[int, int, int], direction: str) -> tuple[int, int, int]:
    r, c, ori = state
    dr, dc = DIR_DELTA[direction]
    return int(r + dr), int(c + dc), int(ori)


def _turn_state(state: tuple[int, int, int], direction: str, next_direction: str) -> tuple[int, int, int]:
    r, c, ori = state
    if direction in "RL":
        if ori != 0 or next_direction not in "UD":
            raise AssertionError("invalid horizontal->vertical turn")
        rr = int(r - 1) if next_direction == "U" else int(r)
        cc = int(c + 1) if direction == "R" else int(c)
        return rr, cc, 1
    if ori != 1 or next_direction not in "RL":
        raise AssertionError("invalid vertical->horizontal turn")
    rr = int(r + 1) if direction == "D" else int(r)
    cc = int(c - 1) if next_direction == "L" else int(c)
    return rr, cc, 0


def _build_route_states(directions: list[str], lengths: list[int]) -> list[tuple[int, int, int]]:
    orientation = 0 if directions[0] in "RL" else 1
    state = (0, 0, orientation)
    states = [state]
    for idx, direction in enumerate(directions):
        for _ in range(int(lengths[idx]) - 1):
            state = _move_state(state, direction)
            states.append(state)
        if idx + 1 < len(directions):
            state = _turn_state(state, direction, directions[idx + 1])
            states.append(state)
    return states


def _normalized_states(states: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    cells: set[tuple[int, int]] = set()
    for state in states:
        cells |= _state_cells(state, (HRECT, VRECT))
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return [(int(r - min_r), int(c - min_c), int(ori)) for r, c, ori in states]


def _states_to_mask(states: list[tuple[int, int, int]]) -> np.ndarray:
    cells: set[tuple[int, int]] = set()
    for state in states:
        cells |= _state_cells(state, (HRECT, VRECT))
    height = max(r for r, _c in cells) + 1
    width = max(c for _r, c in cells) + 1
    mask = np.zeros((int(height), int(width)), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _route_directions() -> list[str]:
    segment_count = random.choice([2, 3, 4])
    start_axis = random.choice(["H", "V"])
    horiz = random.choice(["R", "L"])
    vert = random.choice(["D", "U"])
    directions: list[str] = []
    for idx in range(segment_count):
        if (start_axis == "H" and idx % 2 == 0) or (start_axis == "V" and idx % 2 == 1):
            directions.append(horiz)
        else:
            directions.append(vert)
    return directions


def _validate_generated_instance(
    grid: np.ndarray,
    start_state: tuple[int, int, int],
    target_state: tuple[int, int, int],
) -> bool:
    traversable = (grid == CHANNEL) | (grid == MOVER) | (grid == TARGET)
    dims_pair = (HRECT, VRECT)
    states, adj = _build_state_graph(traversable, dims_pair)
    if start_state not in states or target_state not in states:
        return False

    dist, parent = _bfs_distances(adj, start_state)
    if target_state not in dist:
        return False
    if any(len(adj[state]) > 2 for state in dist):
        return False

    shortest_len, path_count = _shortest_path_count(adj, start_state, target_state)
    if shortest_len is None or path_count != 1:
        return False

    path = _path_from_parent(parent, target_state)
    if len(path) != shortest_len + 1:
        return False
    if not any(path[i][2] != path[i + 1][2] for i in range(len(path) - 1)):
        return False

    expected = _render_output(grid, start_state, target_state, target_state, dims_pair)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    directions = _route_directions()
    lengths = [random.randint(3, 6) for _ in directions]
    states = _normalized_states(_build_route_states(directions, lengths))
    mask = _states_to_mask(states)

    top = random.randint(1, 4)
    left = random.randint(1, 4)
    bottom = random.randint(1, 4)
    right = random.randint(1, 4)
    H = int(mask.shape[0] + top + bottom)
    W = int(mask.shape[1] + left + right)
    if max(H, W) > 30:
        return None

    grid = np.full((H, W), BG, dtype=int)
    route_r0 = int(top)
    route_c0 = int(left)
    grid[route_r0 : route_r0 + mask.shape[0], route_c0 : route_c0 + mask.shape[1]][mask] = CHANNEL

    start_r, start_c, start_ori = states[0]
    target_r, target_c, target_ori = states[-1]
    sh, sw = (HRECT, VRECT)[int(start_ori)]
    th, tw = (HRECT, VRECT)[int(target_ori)]
    start_state = (int(start_r + route_r0), int(start_c + route_c0), int(start_ori))
    target_state = (int(target_r + route_r0), int(target_c + route_c0), int(target_ori))
    _paint_rect(grid, start_state[0], start_state[1], sh, sw, MOVER)
    _paint_rect(grid, target_state[0], target_state[1], th, tw, TARGET)

    if np.count_nonzero(grid == BG) <= np.count_nonzero(grid != BG):
        return None
    if not _validate_generated_instance(grid, start_state, target_state):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-rotation channel sample")
