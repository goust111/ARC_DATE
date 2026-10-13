from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
TEAL = 8

FRAME_SIZE = 4
ROOM_SIZE = 5
ROOM_STRIDE = ROOM_SIZE + 1
GEN_RETRIES = 600


Coord = tuple[int, int]


@dataclass(frozen=True)
class Frame:
    top: int
    left: int
    color: int
    gap: Coord
    direction: Coord


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _frame_border_positions(top: int, left: int) -> list[Coord]:
    cells = []
    for r in range(top, top + FRAME_SIZE):
        for c in range(left, left + FRAME_SIZE):
            if r in (top, top + FRAME_SIZE - 1) or c in (left, left + FRAME_SIZE - 1):
                cells.append((r, c))
    return cells


def _frame_direction(top: int, left: int, gap: Coord) -> Coord:
    center_r = top + 1.5
    center_c = left + 1.5
    dr = -1 if gap[0] < center_r else 1
    dc = -1 if gap[1] < center_c else 1
    return dr, dc


def _find_frames(grid: np.ndarray) -> list[Frame]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames: list[Frame] = []
    for top in range(h - FRAME_SIZE + 1):
        for left in range(w - FRAME_SIZE + 1):
            block = g[top : top + FRAME_SIZE, left : left + FRAME_SIZE]
            inner = block[1:3, 1:3]
            if inner.shape != (2, 2):
                continue
            if not np.all(inner == inner[0, 0]):
                continue
            color = int(inner[0, 0])
            if color not in (RED, BLUE):
                continue

            border_vals = []
            gap: Coord | None = None
            ok = True
            for r, c in _frame_border_positions(top, left):
                v = int(g[r, c])
                border_vals.append(v)
                if v == BLACK:
                    if gap is not None:
                        ok = False
                        break
                    gap = (r, c)
                elif v != GREEN:
                    ok = False
                    break
            if not ok or gap is None:
                continue
            if border_vals.count(BLACK) != 1 or border_vals.count(GREEN) != 11:
                continue
            frames.append(Frame(top, left, color, gap, _frame_direction(top, left, gap)))

    dedup: list[Frame] = []
    seen: set[tuple[int, int]] = set()
    for frame in frames:
        key = (frame.top, frame.left)
        if key not in seen:
            seen.add(key)
            dedup.append(frame)
    dedup.sort(key=lambda f: (f.top, f.left))
    return dedup


def _find_seeds(grid: np.ndarray, frames: list[Frame]) -> dict[int, Coord]:
    occupied = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        occupied[frame.top : frame.top + FRAME_SIZE, frame.left : frame.left + FRAME_SIZE] = True

    seeds: dict[int, Coord] = {}
    for color in (RED, BLUE):
        coords = np.argwhere((grid == color) & (~occupied))
        assert coords.shape == (1, 2), f"expected exactly one seed of color {color}"
        seeds[color] = (int(coords[0, 0]), int(coords[0, 1]))
    return seeds


def _simulate_rays(grid: np.ndarray, frames: list[Frame]) -> tuple[np.ndarray, set[Coord], list[list[Coord]]]:
    g = np.array(grid, dtype=int, copy=True)
    carved: set[Coord] = set()
    per_ray: list[list[Coord]] = []
    for frame in sorted(frames, key=lambda f: (f.top, f.left)):
        dr, dc = frame.direction
        r, c = frame.gap[0] + dr, frame.gap[1] + dc
        erased: list[Coord] = []
        while _in_bounds(g.shape, r, c):
            pos = (r, c)
            if pos in carved:
                break
            value = int(g[r, c])
            if value == BLACK:
                r += dr
                c += dc
                continue
            if value == TEAL:
                g[r, c] = BLACK
                carved.add(pos)
                erased.append(pos)
                r += dr
                c += dc
                continue
            break
        per_ray.append(erased)
    return g, carved, per_ray


def _traversable(grid: np.ndarray, pos: Coord, seeds: set[Coord]) -> bool:
    r, c = pos
    return int(grid[r, c]) == BLACK or pos in seeds


def _bfs_distances(grid: np.ndarray, start: Coord, seeds: set[Coord]) -> dict[Coord, int]:
    q = deque([start])
    dist = {start: 0}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if not _in_bounds(grid.shape, nr, nc) or nxt in dist:
                continue
            if not _traversable(grid, nxt, seeds):
                continue
            dist[nxt] = dist[(r, c)] + 1
            q.append(nxt)
    return dist


def _paint_after_rays(grid: np.ndarray, post_ray_grid: np.ndarray, seeds_by_color: dict[int, Coord]) -> np.ndarray:
    red_seed = seeds_by_color[RED]
    blue_seed = seeds_by_color[BLUE]
    seed_set = {red_seed, blue_seed}
    red_dist = _bfs_distances(post_ray_grid, red_seed, seed_set)
    blue_dist = _bfs_distances(post_ray_grid, blue_seed, seed_set)

    out = np.array(post_ray_grid, copy=True)
    all_cells = set(red_dist) | set(blue_dist)
    for pos in all_cells:
        r, c = pos
        if int(post_ray_grid[r, c]) != BLACK and pos not in seed_set:
            continue
        d_red = red_dist.get(pos, 10**9)
        d_blue = blue_dist.get(pos, 10**9)
        if d_red < d_blue:
            use_red = True
        elif d_blue < d_red:
            use_red = False
        else:
            use_red = red_seed < blue_seed
        if use_red:
            out[r, c] = RED if d_red % 2 == 0 else BLUE
        else:
            out[r, c] = BLUE if d_blue % 2 == 0 else RED
    return out


def _transform(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    assert len(frames) == 2, "expected exactly two framed blocks"
    assert {frame.color for frame in frames} == {RED, BLUE}, "expected one red and one blue framed block"
    seeds = _find_seeds(g, frames)
    carved_grid, carved, per_ray = _simulate_rays(g, frames)
    out = _paint_after_rays(g, carved_grid, seeds)
    info = {
        "frames": frames,
        "seeds": seeds,
        "carved_grid": carved_grid,
        "carved": carved,
        "per_ray": per_ray,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """Carve maze walls with diagonal rays from frame gaps, then parity-fill the modified corridors from two seeds.

    Input structure:
    - Black background cells are maze corridors, while teal cells form 1-pixel maze walls.
    - Two 4x4 green frames contain monochrome 2x2 inner blocks, one red and one blue; each frame has exactly one missing corner-border pixel that acts as a gap.
    - One red seed pixel and one blue seed pixel lie on corridor cells outside the frames, initially in different corridor components.

    Rule:
    - For each frame, use the gap's corner relative to the inner 2x2 block to choose one diagonal ray direction.
    - March the ray outward cell by cell: skip black corridor cells, erase teal wall cells to black, and stop at any green frame cell, any non-black/non-teal cell, any previously carved wall cell, or the grid edge.
    - After both rays finish, run BFS from the red seed and from the blue seed on the modified 4-connected corridor network; a cell takes the parity coloring of its nearer seed, with ties broken by the lexicographically smaller seed coordinate.

    Key insight:
    - The rays do not draw visible paths; they only remove wall pixels, so the real effect is a topology change in the maze, and the final red/blue pattern comes from shortest-path parity on that changed topology.
    """
    out, _ = _transform(np.asarray(grid, dtype=int))
    return out


def _room_start(idx: int) -> int:
    return 1 + idx * ROOM_STRIDE


def _room_bounds(rr: int, cc: int) -> tuple[int, int, int, int]:
    r0 = _room_start(rr)
    c0 = _room_start(cc)
    return r0, c0, r0 + ROOM_SIZE - 1, c0 + ROOM_SIZE - 1


def _wall_positions(count: int) -> list[int]:
    return [i * ROOM_STRIDE for i in range(count + 1)]


def _make_room_grid(room_rows: int, room_cols: int) -> np.ndarray:
    h = room_rows * ROOM_SIZE + room_rows + 1
    w = room_cols * ROOM_SIZE + room_cols + 1
    grid = np.full((h, w), BLACK, dtype=int)
    for r in _wall_positions(room_rows):
        grid[r, :] = TEAL
    for c in _wall_positions(room_cols):
        grid[:, c] = TEAL
    return grid


def _adjacent_rooms(rooms: list[Coord]) -> list[tuple[Coord, Coord]]:
    room_set = set(rooms)
    edges = []
    for rr, cc in rooms:
        for dr, dc in ((1, 0), (0, 1)):
            nbr = (rr + dr, cc + dc)
            if nbr in room_set:
                edges.append(((rr, cc), nbr))
    return edges


def _random_connected_edges(rooms: list[Coord], rng: random.Random) -> list[tuple[Coord, Coord]]:
    if len(rooms) <= 1:
        return []
    adj: dict[Coord, list[Coord]] = {room: [] for room in rooms}
    for a, b in _adjacent_rooms(rooms):
        adj[a].append(b)
        adj[b].append(a)

    start = rng.choice(rooms)
    visited = {start}
    frontier = []
    for nbr in adj[start]:
        frontier.append((start, nbr))
    chosen: list[tuple[Coord, Coord]] = []
    while len(visited) < len(rooms):
        frontier = [edge for edge in frontier if edge[1] not in visited]
        if not frontier:
            return []
        a, b = rng.choice(frontier)
        chosen.append((a, b))
        visited.add(b)
        for nbr in adj[b]:
            if nbr not in visited:
                frontier.append((b, nbr))

    extras = []
    chosen_set = {tuple(sorted(edge)) for edge in chosen}
    for a, b in _adjacent_rooms(rooms):
        key = tuple(sorted((a, b)))
        if key not in chosen_set and rng.random() < 0.35:
            extras.append((a, b))
    return chosen + extras


def _door_candidates(a: Coord, b: Coord) -> list[Coord]:
    if a[0] == b[0]:
        rr = a[0]
        left = min(a[1], b[1])
        r0 = _room_start(rr)
        wall_c = _room_start(left) + ROOM_SIZE
        return [(r0 + i, wall_c) for i in range(ROOM_SIZE)]
    cc = a[1]
    top = min(a[0], b[0])
    c0 = _room_start(cc)
    wall_r = _room_start(top) + ROOM_SIZE
    return [(wall_r, c0 + i) for i in range(ROOM_SIZE)]


def _open_room_edges(grid: np.ndarray, edges: list[tuple[Coord, Coord]], rng: random.Random) -> None:
    for a, b in edges:
        candidates = _door_candidates(a, b)
        rng.shuffle(candidates)
        door_count = 1 + int(rng.random() < 0.25)
        for cell in candidates[:door_count]:
            grid[cell] = BLACK


@dataclass(frozen=True)
class FrameConfig:
    room: Coord
    top: int
    left: int
    gap_name: str
    gap: Coord
    direction: Coord
    barrier_hits: tuple[Coord, ...]
    erased: tuple[Coord, ...]


def _place_frame(grid: np.ndarray, top: int, left: int, color: int, gap_name: str) -> None:
    grid[top : top + FRAME_SIZE, left : left + FRAME_SIZE] = GREEN
    grid[top + 1 : top + 3, left + 1 : left + 3] = color
    gap_map = {
        "nw": (top, left),
        "ne": (top, left + 3),
        "sw": (top + 3, left),
        "se": (top + 3, left + 3),
    }
    grid[gap_map[gap_name]] = BLACK


def _frame_config_candidates(
    room_rows: int,
    room_cols: int,
    barrier_axis: str,
    barrier_index: int,
    side: str,
) -> list[FrameConfig]:
    base = _make_room_grid(room_rows, room_cols)
    wall_rows = set(_wall_positions(room_rows))
    wall_cols = set(_wall_positions(room_cols))
    barrier_cells: set[Coord]
    if barrier_axis == "vertical":
        barrier_c = _wall_positions(room_cols)[barrier_index]
        barrier_cells = {(r, barrier_c) for r in range(base.shape[0]) if r not in wall_rows}
        valid_rooms = [
            (rr, cc)
            for rr in range(room_rows)
            for cc in range(room_cols)
            if (cc < barrier_index) == (side == "left")
        ]
    else:
        barrier_r = _wall_positions(room_rows)[barrier_index]
        barrier_cells = {(barrier_r, c) for c in range(base.shape[1]) if c not in wall_cols}
        valid_rooms = [
            (rr, cc)
            for rr in range(room_rows)
            for cc in range(room_cols)
            if (rr < barrier_index) == (side == "top")
        ]

    gap_specs = {
        "nw": lambda top, left: (top, left),
        "ne": lambda top, left: (top, left + 3),
        "sw": lambda top, left: (top + 3, left),
        "se": lambda top, left: (top + 3, left + 3),
    }

    out: list[FrameConfig] = []
    for room in valid_rooms:
        room_r0, room_c0, _, _ = _room_bounds(*room)
        for off_r in (0, 1):
            for off_c in (0, 1):
                top = room_r0 + off_r
                left = room_c0 + off_c
                for gap_name, gap_fn in gap_specs.items():
                    temp = np.array(base, copy=True)
                    _place_frame(temp, top, left, RED, gap_name)
                    frame = _find_frames(temp)
                    if len(frame) != 1:
                        continue
                    carved_grid, _carved, per_ray = _simulate_rays(temp, frame)
                    erased = tuple(per_ray[0])
                    barrier_hits = tuple(cell for cell in erased if cell in barrier_cells)
                    if len(erased) < 2 or not barrier_hits:
                        continue
                    if barrier_axis == "vertical":
                        if any(r in wall_rows for r, _ in barrier_hits):
                            continue
                    else:
                        if any(c in wall_cols for _, c in barrier_hits):
                            continue
                    gap = gap_fn(top, left)
                    out.append(
                        FrameConfig(
                            room=room,
                            top=top,
                            left=left,
                            gap_name=gap_name,
                            gap=gap,
                            direction=_frame_direction(top, left, gap),
                            barrier_hits=barrier_hits,
                            erased=erased,
                        )
                    )
    return out


def _component_map(grid: np.ndarray, passable: set[int]) -> tuple[np.ndarray, dict[int, list[Coord]]]:
    labels = np.full(grid.shape, -1, dtype=int)
    components: dict[int, list[Coord]] = {}
    next_label = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if labels[r, c] != -1 or int(grid[r, c]) not in passable:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_label
            cells: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if labels[nr, nc] != -1 or int(grid[nr, nc]) not in passable:
                        continue
                    labels[nr, nc] = next_label
                    q.append((nr, nc))
            components[next_label] = cells
            next_label += 1
    return labels, components


def _bbox_distance(cell: Coord, bbox: tuple[int, int, int, int]) -> int:
    r, c = cell
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return dr + dc


def _choose_seed(cells: list[Coord], frame_boxes: list[tuple[int, int, int, int]], rng: random.Random) -> Coord | None:
    candidates = [
        cell
        for cell in cells
        if all(_bbox_distance(cell, box) >= 2 for box in frame_boxes)
        and all(cell[0] not in (box[0], box[2]) and cell[1] not in (box[1], box[3]) for box in frame_boxes)
    ]
    if not candidates:
        candidates = [cell for cell in cells if all(_bbox_distance(cell, box) >= 1 for box in frame_boxes)]
    if not candidates:
        return None
    candidates.sort(key=lambda rc: (rc[0], rc[1]))
    thirds = candidates[len(candidates) // 3 : len(candidates) - len(candidates) // 3]
    pool = thirds if thirds else candidates
    return rng.choice(pool)


def _fill_without_carving(grid: np.ndarray) -> np.ndarray:
    frames = _find_frames(grid)
    seeds = _find_seeds(grid, frames)
    return _paint_after_rays(grid, np.asarray(grid, dtype=int), seeds)


def _semantic_ok(inp: np.ndarray) -> bool:
    try:
        out, info = _transform(inp)
    except Exception:
        return False

    frames: list[Frame] = info["frames"]  # type: ignore[assignment]
    seeds: dict[int, Coord] = info["seeds"]  # type: ignore[assignment]
    carved_grid: np.ndarray = info["carved_grid"]  # type: ignore[assignment]
    per_ray: list[list[Coord]] = info["per_ray"]  # type: ignore[assignment]
    if any(len(ray) == 0 for ray in per_ray):
        return False
    if sum(len(ray) for ray in per_ray) < 3:
        return False
    if len(set(cell for ray in per_ray for cell in ray)) != sum(len(ray) for ray in per_ray):
        return False

    seed_set = {seeds[RED], seeds[BLUE]}
    pre_red = _bfs_distances(inp, seeds[RED], seed_set)
    if seeds[BLUE] in pre_red:
        return False
    post_red = _bfs_distances(carved_grid, seeds[RED], seed_set)
    if seeds[BLUE] not in post_red:
        return False

    no_carve = _fill_without_carving(inp)
    diff = int(np.count_nonzero(out != no_carve))
    if diff < 8:
        return False

    corridor_colors = out[(out == RED) | (out == BLUE)]
    if int(np.count_nonzero(corridor_colors == RED)) < 8:
        return False
    if int(np.count_nonzero(corridor_colors == BLUE)) < 8:
        return False

    frame_mask = np.zeros_like(inp, dtype=bool)
    for frame in frames:
        frame_mask[frame.top : frame.top + FRAME_SIZE, frame.left : frame.left + FRAME_SIZE] = True
    if np.any(out[frame_mask & (inp == GREEN)] != GREEN):
        return False
    if np.any(out[(inp == TEAL) & (carved_grid == TEAL)] != TEAL):
        return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.choice((3, 4))
    room_cols = rng.choice((3, 4))
    barrier_axis = rng.choice(("vertical", "horizontal"))
    if barrier_axis == "vertical":
        barrier_index = rng.randrange(1, room_cols)
        side_a, side_b = "left", "right"
    else:
        barrier_index = rng.randrange(1, room_rows)
        side_a, side_b = "top", "bottom"

    cand_a = _frame_config_candidates(room_rows, room_cols, barrier_axis, barrier_index, side_a)
    cand_b = _frame_config_candidates(room_rows, room_cols, barrier_axis, barrier_index, side_b)
    if not cand_a or not cand_b:
        return None

    frame_a = rng.choice(cand_a)
    frame_b = rng.choice(cand_b)
    if frame_a.room == frame_b.room:
        return None

    grid = _make_room_grid(room_rows, room_cols)
    frame_colors = [RED, BLUE]
    rng.shuffle(frame_colors)
    _place_frame(grid, frame_a.top, frame_a.left, frame_colors[0], frame_a.gap_name)
    _place_frame(grid, frame_b.top, frame_b.left, frame_colors[1], frame_b.gap_name)

    if barrier_axis == "vertical":
        rooms_a = [(rr, cc) for rr in range(room_rows) for cc in range(barrier_index)]
        rooms_b = [(rr, cc) for rr in range(room_rows) for cc in range(barrier_index, room_cols)]
    else:
        rooms_a = [(rr, cc) for rr in range(barrier_index) for cc in range(room_cols)]
        rooms_b = [(rr, cc) for rr in range(barrier_index, room_rows) for cc in range(room_cols)]

    edges_a = _random_connected_edges(rooms_a, rng)
    edges_b = _random_connected_edges(rooms_b, rng)
    if rooms_a and not edges_a and len(rooms_a) > 1:
        return None
    if rooms_b and not edges_b and len(rooms_b) > 1:
        return None
    _open_room_edges(grid, edges_a, rng)
    _open_room_edges(grid, edges_b, rng)

    frames = _find_frames(grid)
    if len(frames) != 2:
        return None
    carved_grid, _carved, per_ray = _simulate_rays(grid, frames)
    if any(len(ray) == 0 for ray in per_ray):
        return None

    labels, comps = _component_map(grid, {BLACK})
    if len(comps) < 2:
        return None

    if barrier_axis == "vertical":
        barrier_col = _wall_positions(room_cols)[barrier_index]
        comps_a = [cells for cells in comps.values() if any(c < barrier_col for _, c in cells)]
        comps_b = [cells for cells in comps.values() if any(c > barrier_col for _, c in cells)]
    else:
        barrier_row = _wall_positions(room_rows)[barrier_index]
        comps_a = [cells for cells in comps.values() if any(r < barrier_row for r, _ in cells)]
        comps_b = [cells for cells in comps.values() if any(r > barrier_row for r, _ in cells)]
    if not comps_a or not comps_b:
        return None

    frame_boxes = [
        (frame.top, frame.left, frame.top + FRAME_SIZE - 1, frame.left + FRAME_SIZE - 1)
        for frame in frames
    ]
    seed_red = _choose_seed(max(comps_a, key=len), frame_boxes, rng)
    seed_blue = _choose_seed(max(comps_b, key=len), frame_boxes, rng)
    if seed_red is None or seed_blue is None or seed_red == seed_blue:
        return None

    grid[seed_red] = RED
    grid[seed_blue] = BLUE
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-carve parity-maze input")
