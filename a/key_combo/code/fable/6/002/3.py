from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, find_components, get_bg, in_bounds, neighbors4, paste, random_free_location


Coord = tuple[int, int]
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_POOL = tuple(range(10))

GRID_HEIGHT_RANGE = (20, 30)
GRID_WIDTH_RANGE = (20, 30)
COMPONENT_COUNT_RANGE = (2, 4)
CHAIN_LENGTH_RANGE = (8, 14)
COMPONENT_BUILD_ATTEMPTS = 220
GRID_BUILD_ATTEMPTS = 500


def _perp_dirs(direction: Coord) -> tuple[Coord, Coord]:
    dr, dc = map(int, direction)
    return (-dc, dr), (dc, -dr)


def _step(src: Coord, dst: Coord) -> Coord:
    return int(dst[0] - src[0]), int(dst[1] - src[1])


def _turn_count(chain: list[Coord]) -> int:
    dirs = [_step(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    return sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _gate_posts(center: Coord, inward: Coord) -> tuple[Coord, Coord]:
    perps = _perp_dirs(_step(center, inward))
    return (
        (int(center[0] + perps[0][0]), int(center[1] + perps[0][1])),
        (int(center[0] + perps[1][0]), int(center[1] + perps[1][1])),
    )


def _grow_walk(path: list[Coord], used: set[Coord], target_len: int) -> bool:
    if len(path) == int(target_len):
        return True

    tail = path[-1]
    candidates = list(DIR4)
    random.shuffle(candidates)
    if len(path) >= 2 and random.random() < 0.55:
        prev_dir = _step(path[-2], path[-1])
        if prev_dir in candidates:
            candidates.remove(prev_dir)
            candidates.insert(0, prev_dir)

    for dr, dc in candidates:
        nxt = (int(tail[0] + dr), int(tail[1] + dc))
        if nxt in used:
            continue
        if any((int(nr), int(nc)) in used and (int(nr), int(nc)) != tail for nr, nc in neighbors4(*nxt)):
            continue
        path.append(nxt)
        used.add(nxt)
        if _grow_walk(path, used, target_len):
            return True
        used.remove(nxt)
        path.pop()
    return False


def _walk_has_valid_gates(path: list[Coord]) -> bool:
    used = set(path)
    all_posts: set[Coord] = set()
    for center, inward in ((path[0], path[1]), (path[-1], path[-2])):
        posts = _gate_posts(center, inward)
        for post in posts:
            if post in used or post in all_posts:
                return False
            neighbors = {(int(nr), int(nc)) for nr, nc in neighbors4(*post) if (int(nr), int(nc)) in used}
            if neighbors != {center}:
                return False
        all_posts.update(posts)
    for post in all_posts:
        if any((int(nr), int(nc)) in all_posts and (int(nr), int(nc)) != post for nr, nc in neighbors4(*post)):
            return False
    return True


def _build_walk(length: int, require_turn: bool) -> list[Coord]:
    for _ in range(COMPONENT_BUILD_ATTEMPTS):
        path = [(0, 0)]
        used = {path[0]}
        if not _grow_walk(path, used, int(length)):
            continue
        if require_turn and _turn_count(path) == 0:
            continue
        if not _walk_has_valid_gates(path):
            continue
        return path
    raise RuntimeError("failed to build a valid path walk")


def _marker_options(chain_len: int, target_side: int, authorized_near: bool) -> list[int]:
    options: list[int] = []
    for idx in range(2, int(chain_len) - 2):
        if target_side == 0:
            target_dist = int(idx - 1)
            other_dist = int(chain_len - 2 - idx)
        else:
            target_dist = int(chain_len - 2 - idx)
            other_dist = int(idx - 1)
        if target_dist < 1 or other_dist < 1 or target_dist == other_dist:
            continue
        if authorized_near and target_dist < other_dist:
            options.append(int(idx))
        if not authorized_near and target_dist > other_dist:
            options.append(int(idx))
    return options


def _build_component_patch(bg: int, authorized_near: bool, require_turn: bool) -> np.ndarray:
    for _ in range(COMPONENT_BUILD_ATTEMPTS):
        chain_len = random.randint(*CHAIN_LENGTH_RANGE)
        chain = _build_walk(chain_len, require_turn=require_turn)
        target_side = random.randint(0, 1)
        marker_choices = _marker_options(len(chain), target_side, authorized_near)
        if not marker_choices:
            continue
        marker_idx = random.choice(marker_choices)

        colors = random.sample([c for c in COLOR_POOL if c != int(bg)], 3)
        path_color = int(colors[0])
        authorized_color = int(colors[1])
        other_gate_color = int(colors[2])

        left_posts = _gate_posts(chain[0], chain[1])
        right_posts = _gate_posts(chain[-1], chain[-2])
        left_color = authorized_color if target_side == 0 else other_gate_color
        right_color = authorized_color if target_side == 1 else other_gate_color

        all_cells = list(chain) + list(left_posts) + list(right_posts)
        bbox = bbox_coords(np.asarray(all_cells, dtype=int))
        r0, c0, r1, c1 = map(int, bbox)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)

        def norm(cell: Coord) -> Coord:
            return int(cell[0] - r0), int(cell[1] - c0)

        for cell in chain:
            patch[norm(cell)] = path_color
        patch[norm(chain[marker_idx])] = authorized_color
        for post in left_posts:
            patch[norm(post)] = left_color
        for post in right_posts:
            patch[norm(post)] = right_color

        info = _parse_component(patch, np.argwhere(patch != int(bg)))
        if info["turn_count"] < 1 and require_turn:
            continue
        if bool(info["target_distance"] < info["other_distance"]) != bool(authorized_near):
            continue
        return patch
    raise RuntimeError("failed to build a valid component patch")


def _component_degree(coords_set: set[Coord], cell: Coord) -> int:
    return sum((int(nr), int(nc)) in coords_set for nr, nc in neighbors4(*cell))


def _detect_gates(grid: np.ndarray, coords_set: set[Coord]) -> list[dict[str, object]]:
    gates_by_center: dict[Coord, dict[str, object]] = {}
    degrees = {cell: _component_degree(coords_set, cell) for cell in coords_set}
    for r, c in coords_set:
        center = (int(r), int(c))
        center_color = int(grid[center])
        for dr, dc in ((1, 0), (0, 1)):
            post_a = (int(r - dr), int(c - dc))
            post_b = (int(r + dr), int(c + dc))
            if post_a not in coords_set or post_b not in coords_set:
                continue
            post_color = int(grid[post_a])
            if post_color != int(grid[post_b]) or post_color == center_color:
                continue
            if degrees[post_a] != 1 or degrees[post_b] != 1:
                continue
            inward = [
                (int(nr), int(nc))
                for nr, nc in neighbors4(r, c)
                if (int(nr), int(nc)) in coords_set and (int(nr), int(nc)) not in {post_a, post_b}
            ]
            if len(inward) != 1:
                continue
            gate = {
                "center": center,
                "posts": {post_a, post_b},
                "post_color": post_color,
                "inward": inward[0],
                "axis": (int(dr), int(dc)),
            }
            prev = gates_by_center.get(center)
            if prev is not None and prev["posts"] != gate["posts"]:
                raise AssertionError("ambiguous gate center")
            gates_by_center[center] = gate
    gates = sorted(gates_by_center.values(), key=lambda gate: gate["center"])
    assert len(gates) == 2, "expected exactly two gates per component"
    assert int(gates[0]["post_color"]) != int(gates[1]["post_color"]), "gate colors must differ"
    return gates


def _ordered_chain(coords_set: set[Coord], gate_posts: set[Coord], gate_centers: set[Coord]) -> list[Coord]:
    chain_nodes = coords_set - gate_posts
    adjacency: dict[Coord, list[Coord]] = {}
    for cell in chain_nodes:
        nbrs = [(int(nr), int(nc)) for nr, nc in neighbors4(*cell) if (int(nr), int(nc)) in chain_nodes]
        assert len(nbrs) in (1, 2), "component skeleton must be a simple path"
        adjacency[cell] = nbrs
    endpoints = {cell for cell, nbrs in adjacency.items() if len(nbrs) == 1}
    assert endpoints == gate_centers, "path endpoints must be the two gate centers"

    start = min(gate_centers)
    order = [start]
    prev: Coord | None = None
    cur = start
    while True:
        nxts = [cell for cell in adjacency[cur] if cell != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "simple path expected"
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == len(chain_nodes), "chain walk must cover the whole component path"
    assert order[-1] in gate_centers and order[-1] != start
    return order


def _parse_component(grid: np.ndarray, coords: np.ndarray) -> dict[str, object]:
    coords_set = {tuple(map(int, coord)) for coord in np.asarray(coords, dtype=int)}
    gates = _detect_gates(grid, coords_set)
    gate_posts = set().union(*(gate["posts"] for gate in gates))
    chain = _ordered_chain(coords_set, gate_posts, {gate["center"] for gate in gates})

    chain_colors = [int(grid[cell]) for cell in chain]
    path_color = int(Counter(chain_colors).most_common(1)[0][0])
    marker_cells = [cell for cell in chain if int(grid[cell]) != path_color]
    assert len(marker_cells) == 1, "expected exactly one marker on the path"
    marker = marker_cells[0]
    marker_color = int(grid[marker])
    positions = {cell: idx for idx, cell in enumerate(chain)}

    left_gate = None
    right_gate = None
    for gate in gates:
        gate["center_idx"] = int(positions[gate["center"]])
        gate["parking_idx"] = int(positions[gate["inward"]])
        assert abs(int(gate["center_idx"]) - int(gate["parking_idx"])) == 1
        if int(gate["center_idx"]) == 0:
            assert int(gate["parking_idx"]) == 1
            left_gate = gate
        else:
            assert int(gate["center_idx"]) == len(chain) - 1
            assert int(gate["parking_idx"]) == len(chain) - 2
            right_gate = gate
    assert left_gate is not None and right_gate is not None

    matching = [gate for gate in gates if int(gate["post_color"]) == marker_color]
    assert len(matching) == 1, "marker color must authorize exactly one gate"
    target_gate = matching[0]
    other_gate = right_gate if target_gate is left_gate else left_gate

    marker_idx = int(positions[marker])
    target_distance = abs(int(marker_idx) - int(target_gate["parking_idx"]))
    other_distance = abs(int(marker_idx) - int(other_gate["parking_idx"]))
    assert target_distance != other_distance, "generator avoids equal-distance gates"

    return {
        "coords_set": coords_set,
        "chain": chain,
        "path_color": path_color,
        "marker": marker,
        "marker_color": marker_color,
        "marker_idx": marker_idx,
        "positions": positions,
        "gates": gates,
        "left_gate": left_gate,
        "right_gate": right_gate,
        "target_gate": target_gate,
        "other_gate": other_gate,
        "target_distance": int(target_distance),
        "other_distance": int(other_distance),
        "turn_count": int(_turn_count(chain)),
    }


def _apply_component_transform(out: np.ndarray, info: dict[str, object], bg: int) -> None:
    chain = info["chain"]
    marker_idx = int(info["marker_idx"])
    parking_idx = int(info["target_gate"]["parking_idx"])
    if marker_idx == parking_idx:
        return
    lo = min(marker_idx, parking_idx)
    hi = max(marker_idx, parking_idx)
    for idx in range(lo, hi + 1):
        out[chain[idx]] = int(bg)
    out[info["target_gate"]["center"]] = int(info["path_color"])
    out[info["target_gate"]["inward"]] = int(info["marker_color"])


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    for comp in comps:
        info = _parse_component(g, comp["coords"])
        _apply_component_transform(out, info, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marker along its path to the gate whose posts share the marker's color, erasing only the traversed half.

    Input structure:
    - The dominant background contains several disconnected 4-connected multicolor components.
    - Inside each component, one monochrome path has a single differently colored marker on the path itself.
    - Each path endpoint is a gate center: a path-colored cell flanked by two opposite post pixels of one color; the two endpoint gates use different post colors.

    Rule:
    - In each component, identify the unique gate whose post color matches the marker color.
    - Move the marker along the path to the gate's inner adjacent path cell, the one touching the gate center but not lying between the two posts.
    - Erase the marker's original cell and every traversed path cell on that side; keep the untouched half-path plus both gate centers and all gate posts.

    Key insight: The competition between the two endpoint gates is decided only by color authorization, not by which gate is closer, straighter, or encountered first along the path.
    """
    return _solve(np.asarray(grid, dtype=int))


def _validate_generated_grid(grid: np.ndarray, expected_components: int) -> bool:
    bg = int(get_bg(grid))
    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != bg:
        return False

    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    if len(comps) != int(expected_components):
        return False

    near_seen = False
    far_seen = False
    turn_seen = False
    expected = grid.copy()

    for comp in comps:
        info = _parse_component(grid, comp["coords"])
        near_seen |= int(info["target_distance"]) < int(info["other_distance"])
        far_seen |= int(info["target_distance"]) > int(info["other_distance"])
        turn_seen |= int(info["turn_count"]) > 0

        chain = info["chain"]
        marker_idx = int(info["marker_idx"])
        parking_idx = int(info["target_gate"]["parking_idx"])
        if marker_idx in {0, len(chain) - 1} or parking_idx in {0, len(chain) - 1}:
            return False
        if int(info["target_distance"]) < 1 or int(info["other_distance"]) < 1:
            return False

        _apply_component_transform(expected, info, bg)

    if expected_components >= 2 and not (near_seen and far_seen):
        return False
    if not turn_seen:
        return False

    if np.array_equal(grid, expected):
        return False
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(COLOR_POOL)
    component_count = random.randint(*COMPONENT_COUNT_RANGE)

    branch_flags = [True, False]
    while len(branch_flags) < component_count:
        branch_flags.append(bool(random.getrandbits(1)))
    random.shuffle(branch_flags)

    turn_flags = [True]
    while len(turn_flags) < component_count:
        turn_flags.append(random.random() < 0.7)
    random.shuffle(turn_flags)

    try:
        patches = [
            _build_component_patch(bg, authorized_near=branch_flags[i], require_turn=turn_flags[i])
            for i in range(component_count)
        ]
    except RuntimeError:
        return None

    patches.sort(key=lambda patch: int(np.count_nonzero(patch != bg)), reverse=True)

    h_min = max(GRID_HEIGHT_RANGE[0], max(int(p.shape[0]) for p in patches) + 2)
    w_min = max(GRID_WIDTH_RANGE[0], max(int(p.shape[1]) for p in patches) + 2)
    if h_min > GRID_HEIGHT_RANGE[1] or w_min > GRID_WIDTH_RANGE[1]:
        return None

    grid = np.full(
        (random.randint(h_min, GRID_HEIGHT_RANGE[1]), random.randint(w_min, GRID_WIDTH_RANGE[1])),
        int(bg),
        dtype=int,
    )

    for patch in patches:
        try:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, patch, r0, c0, transparent=bg)

    if not _validate_generated_grid(grid, component_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GRID_BUILD_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-gate ARC input")
