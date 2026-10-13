import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

WALL = GREY
PASSAGE = AZURE
DOOR = MAROON
KEY_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GEN_ATTEMPTS = 600
MAZE_CELL_RANGE = (4, 8)
OUTER_MARGIN_RANGE = (1, 3)


@dataclass(frozen=True)
class Entrance:
    pos: tuple[int, int]
    key_pos: tuple[int, int]
    dir_idx: int
    hand: str
    color: int


@dataclass(frozen=True)
class MazeSpec:
    grid: np.ndarray
    local: np.ndarray
    offset: tuple[int, int]
    entrances: tuple[Entrance, ...]
    door_positions: frozenset[tuple[int, int]]


def _step(pos: tuple[int, int], dir_idx: int) -> tuple[int, int]:
    dr, dc = DIRS[dir_idx]
    return pos[0] + dr, pos[1] + dc


def _turn_left(dir_idx: int) -> int:
    return (dir_idx - 1) % 4


def _turn_right(dir_idx: int) -> int:
    return (dir_idx + 1) % 4


def _in_bounds(arr: np.ndarray, pos: tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]


def _is_open(arr: np.ndarray, pos: tuple[int, int]) -> bool:
    return _in_bounds(arr, pos) and int(arr[pos]) in {PASSAGE, DOOR}


def _open_neighbors(arr: np.ndarray, pos: tuple[int, int]) -> list[tuple[int, int]]:
    return [_step(pos, d) for d in range(4) if _is_open(arr, _step(pos, d))]


def _boundary_openings(local: np.ndarray) -> list[tuple[int, int]]:
    h, w = local.shape
    openings: list[tuple[int, int]] = []
    for c in range(w):
        if int(local[0, c]) != WALL:
            openings.append((0, c))
        if int(local[h - 1, c]) != WALL:
            openings.append((h - 1, c))
    for r in range(1, h - 1):
        if int(local[r, 0]) != WALL:
            openings.append((r, 0))
        if int(local[r, w - 1]) != WALL:
            openings.append((r, w - 1))
    return openings


def _opening_sides(local: np.ndarray, pos: tuple[int, int]) -> list[int]:
    r, c = pos
    h, w = local.shape
    sides = []
    if r == 0:
        sides.append(0)
    if c == w - 1:
        sides.append(1)
    if r == h - 1:
        sides.append(2)
    if c == 0:
        sides.append(3)
    return sides


def _infer_hand(local: np.ndarray, entrance_pos: tuple[int, int], dir_idx: int) -> str:
    first = _step(entrance_pos, dir_idx)
    assert _is_open(local, first), "entrance must lead to an interior corridor cell"
    left_wall = not _is_open(local, _step(first, _turn_left(dir_idx)))
    right_wall = not _is_open(local, _step(first, _turn_right(dir_idx)))
    assert left_wall ^ right_wall, "exactly one entrance side must expose a wall for hand-rule inference"
    return "left" if left_wall else "right"


def _parse_maze(grid: np.ndarray) -> MazeSpec:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    structure_mask = np.isin(g, (WALL, PASSAGE, DOOR))
    coords = np.argwhere(structure_mask)
    assert len(coords) > 0, "maze structure must exist"
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())

    local = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(np.isin(local, (WALL, PASSAGE, DOOR))), "maze bbox must be fully occupied by walls/passages/doors"

    entrances: list[Entrance] = []
    key_positions: set[tuple[int, int]] = set()
    key_colors: set[int] = set()

    for pos in _boundary_openings(local):
        sides = _opening_sides(local, pos)
        assert len(sides) == 1, "entrance gaps cannot sit on corners"
        outward = sides[0]
        inward = (outward + 2) % 4
        gr = r0 + pos[0]
        gc = c0 + pos[1]
        key_pos = (gr + DIRS[outward][0], gc + DIRS[outward][1])
        assert _in_bounds(g, key_pos), "key must fit inside the canvas"
        key_color = int(g[key_pos])
        assert key_color not in {BLACK, WALL, PASSAGE, DOOR}, "entrance must have one colored key outside"
        assert key_pos not in key_positions, "each entrance must have a unique key cell"
        assert key_color not in key_colors, "keys must use distinct colors"
        hand = _infer_hand(local, pos, inward)
        entrances.append(Entrance(pos=pos, key_pos=key_pos, dir_idx=inward, hand=hand, color=key_color))
        key_positions.add(key_pos)
        key_colors.add(key_color)

    assert 2 <= len(entrances) <= 4, "expected 2 to 4 entrances"

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    colored_outside = {
        tuple(map(int, pt))
        for pt in np.argwhere(outside != BLACK)
    }
    assert colored_outside == key_positions, "outside the maze bbox only the keys may be colored"

    door_positions = frozenset(
        (int(r - r0), int(c - c0))
        for r, c in np.argwhere(local == DOOR)
    )
    return MazeSpec(
        grid=g,
        local=local.copy(),
        offset=(r0, c0),
        entrances=tuple(entrances),
        door_positions=door_positions,
    )


def _next_dir(local: np.ndarray, pos: tuple[int, int], heading: int, hand: str) -> int:
    if hand == "left":
        order = (_turn_left(heading), heading, _turn_right(heading), (heading + 2) % 4)
    else:
        order = (_turn_right(heading), heading, _turn_left(heading), (heading + 2) % 4)
    for dir_idx in order:
        nxt = _step(pos, dir_idx)
        if _is_open(local, nxt):
            return dir_idx
    raise AssertionError("walker must always have a valid continuation")


def _trace_key(spec: MazeSpec, entrance: Entrance) -> dict[tuple[int, int], int]:
    entrance_set = {e.pos for e in spec.entrances}
    visits: dict[tuple[int, int], int] = {}

    pos = entrance.pos
    heading = entrance.dir_idx
    step_count = 1
    visits[pos] = step_count

    pos = _step(pos, heading)
    step_count += 1
    assert _is_open(spec.local, pos), "entrance must connect to an interior path"
    visits.setdefault(pos, step_count)

    seen_states: set[tuple[tuple[int, int], int]] = set()
    while True:
        if pos in entrance_set and pos != entrance.pos:
            return visits
        state = (pos, heading)
        assert state not in seen_states, "wall-following path should terminate at another entrance"
        seen_states.add(state)
        heading = _next_dir(spec.local, pos, heading, entrance.hand)
        pos = _step(pos, heading)
        step_count += 1
        visits.setdefault(pos, step_count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keys enter a rectangular perfect maze, infer left-vs-right hand-following
    from the first interior wall beside the entrance, and recolor only the
    maroon door cells they reach first.
    """
    spec = _parse_maze(grid)

    arrivals: dict[tuple[int, int], list[tuple[int, int]]] = {pos: [] for pos in spec.door_positions}
    for entrance in spec.entrances:
        visits = _trace_key(spec, entrance)
        for door_pos in spec.door_positions:
            step_count = visits.get(door_pos)
            if step_count is not None:
                arrivals[door_pos].append((step_count, entrance.color))

    out = np.asarray(grid, dtype=int).copy()
    for entrance in spec.entrances:
        out[entrance.key_pos] = BLACK

    for door_pos in spec.door_positions:
        hits = arrivals[door_pos]
        if not hits:
            continue
        best = min(step_count for step_count, _color in hits)
        colors = {color for step_count, color in hits if step_count == best}
        fill = ORANGE if len(colors) > 1 else next(iter(colors))
        gr = spec.offset[0] + door_pos[0]
        gc = spec.offset[1] + door_pos[1]
        out[gr, gc] = fill

    return out


def _carve_perfect_maze(cell_h: int, cell_w: int) -> np.ndarray:
    maze = np.full((2 * cell_h + 1, 2 * cell_w + 1), WALL, dtype=int)
    visited = np.zeros((cell_h, cell_w), dtype=bool)
    stack = [(random.randrange(cell_h), random.randrange(cell_w))]
    visited[stack[0]] = True
    maze[2 * stack[0][0] + 1, 2 * stack[0][1] + 1] = PASSAGE

    while stack:
        cr, cc = stack[-1]
        neighbors = []
        for dr, dc in DIRS:
            nr = cr + dr
            nc = cc + dc
            if 0 <= nr < cell_h and 0 <= nc < cell_w and not visited[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            stack.pop()
            continue
        nr, nc = random.choice(neighbors)
        visited[nr, nc] = True
        maze[2 * nr + 1, 2 * nc + 1] = PASSAGE
        maze[cr + nr + 1, cc + nc + 1] = PASSAGE
        stack.append((nr, nc))
    return maze


def _entrance_candidates(maze: np.ndarray) -> list[dict[str, object]]:
    h, w = maze.shape
    candidates: list[dict[str, object]] = []
    for c in range(1, w - 1, 2):
        if maze[1, c] == PASSAGE:
            pos = (0, c)
            dir_idx = 2
            first = (1, c)
            left_open = _is_open(maze, _step(first, _turn_left(dir_idx)))
            right_open = _is_open(maze, _step(first, _turn_right(dir_idx)))
            if left_open ^ right_open:
                candidates.append({"pos": pos, "dir_idx": dir_idx, "hand": "left" if not left_open else "right"})
        if maze[h - 2, c] == PASSAGE:
            pos = (h - 1, c)
            dir_idx = 0
            first = (h - 2, c)
            left_open = _is_open(maze, _step(first, _turn_left(dir_idx)))
            right_open = _is_open(maze, _step(first, _turn_right(dir_idx)))
            if left_open ^ right_open:
                candidates.append({"pos": pos, "dir_idx": dir_idx, "hand": "left" if not left_open else "right"})
    for r in range(1, h - 1, 2):
        if maze[r, 1] == PASSAGE:
            pos = (r, 0)
            dir_idx = 1
            first = (r, 1)
            left_open = _is_open(maze, _step(first, _turn_left(dir_idx)))
            right_open = _is_open(maze, _step(first, _turn_right(dir_idx)))
            if left_open ^ right_open:
                candidates.append({"pos": pos, "dir_idx": dir_idx, "hand": "left" if not left_open else "right"})
        if maze[r, w - 2] == PASSAGE:
            pos = (r, w - 1)
            dir_idx = 3
            first = (r, w - 2)
            left_open = _is_open(maze, _step(first, _turn_left(dir_idx)))
            right_open = _is_open(maze, _step(first, _turn_right(dir_idx)))
            if left_open ^ right_open:
                candidates.append({"pos": pos, "dir_idx": dir_idx, "hand": "left" if not left_open else "right"})
    return candidates


def _far_enough(pos: tuple[int, int], chosen: list[dict[str, object]]) -> bool:
    for other in chosen:
        orow, ocol = other["pos"]
        if abs(pos[0] - orow) + abs(pos[1] - ocol) < 4:
            return False
    return True


def _pick_entrances(candidates: list[dict[str, object]], target_mode: str) -> list[dict[str, object]] | None:
    lefts = [cand for cand in candidates if cand["hand"] == "left"]
    rights = [cand for cand in candidates if cand["hand"] == "right"]
    if not lefts or not rights:
        return None

    max_count = min(4, len(candidates))
    if max_count < 2:
        return None
    if target_mode in {"tie", "mixed"} and max_count >= 4:
        target_count = 4
    elif target_mode in {"tie", "mixed"} and max_count >= 3:
        target_count = 3
    else:
        target_count = random.randint(2, max_count)

    for _ in range(200):
        selected: list[dict[str, object]] = []
        first_left = random.choice(lefts)
        selected.append(first_left)
        right_options = [cand for cand in rights if cand["pos"] != first_left["pos"] and _far_enough(cand["pos"], selected)]
        if not right_options:
            continue
        selected.append(random.choice(right_options))

        pool = [cand for cand in candidates if cand["pos"] not in {item["pos"] for item in selected}]
        random.shuffle(pool)
        for cand in pool:
            if len(selected) >= target_count:
                break
            if _far_enough(cand["pos"], selected):
                selected.append(cand)
        if len(selected) == target_count:
            return selected
    return None


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _segments(local: np.ndarray) -> list[list[tuple[int, int]]]:
    open_cells = [tuple(map(int, pt)) for pt in np.argwhere(local != WALL)]
    degree = {pos: len(_open_neighbors(local, pos)) for pos in open_cells}
    vertices = {pos for pos, deg in degree.items() if deg != 2}

    segments: list[list[tuple[int, int]]] = []
    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for vertex in sorted(vertices):
        for nbr in _open_neighbors(local, vertex):
            edge = _edge_key(vertex, nbr)
            if edge in seen_edges:
                continue
            path = [vertex, nbr]
            seen_edges.add(edge)
            prev = vertex
            cur = nbr
            while cur not in vertices:
                nxts = [nxt for nxt in _open_neighbors(local, cur) if nxt != prev]
                assert len(nxts) == 1, "degree-2 corridor cells must continue uniquely"
                nxt = nxts[0]
                path.append(nxt)
                seen_edges.add(_edge_key(cur, nxt))
                prev, cur = cur, nxt
            segments.append(path)
    return segments


def _door_candidates(spec: MazeSpec, visit_maps: list[dict[tuple[int, int], int]]) -> list[dict[str, object]]:
    entrance_set = {ent.pos for ent in spec.entrances}
    candidates: list[dict[str, object]] = []

    for seg_idx, path in enumerate(_segments(spec.local)):
        for pos in path[1:-1]:
            if pos in entrance_set:
                continue
            hits = [(visit[pos], key_idx) for key_idx, visit in enumerate(visit_maps) if pos in visit]
            if not hits:
                continue
            hits.sort()
            best = hits[0][0]
            winners = [key_idx for step_count, key_idx in hits if step_count == best]
            if len(hits) == 1:
                kind = "solo"
            elif len(winners) == 1:
                kind = "winner"
            else:
                kind = "tie"
            candidates.append(
                {
                    "pos": pos,
                    "segment": seg_idx,
                    "kind": kind,
                    "hits": hits,
                }
            )
    return candidates


def _take_candidate(
    chosen: list[dict[str, object]],
    used_segments: set[int],
    pool: list[dict[str, object]],
) -> bool:
    options = [cand for cand in pool if cand["segment"] not in used_segments]
    if not options:
        return False
    cand = random.choice(options)
    chosen.append(cand)
    used_segments.add(int(cand["segment"]))
    return True


def _select_doors(spec: MazeSpec, visit_maps: list[dict[tuple[int, int], int]], target_mode: str) -> list[tuple[int, int]] | None:
    candidates = _door_candidates(spec, visit_maps)
    if not candidates:
        return None

    by_kind = {
        kind: [cand for cand in candidates if cand["kind"] == kind]
        for kind in ("solo", "winner", "tie")
    }
    chosen: list[dict[str, object]] = []
    used_segments: set[int] = set()

    if target_mode == "solo":
        if not _take_candidate(chosen, used_segments, by_kind["solo"]):
            return None
    elif target_mode == "winner":
        if not _take_candidate(chosen, used_segments, by_kind["winner"]):
            return None
    elif target_mode == "tie":
        if not _take_candidate(chosen, used_segments, by_kind["tie"]):
            return None
    else:
        if not _take_candidate(chosen, used_segments, by_kind["winner"]):
            return None
        if not _take_candidate(chosen, used_segments, by_kind["tie"]):
            return None

    max_segments = len({int(cand["segment"]) for cand in candidates})
    target_count = random.randint(len(chosen), min(5, max_segments))
    while len(chosen) < target_count:
        available = [cand for cand in candidates if int(cand["segment"]) not in used_segments]
        if not available:
            break
        if target_mode == "solo":
            priority = by_kind["solo"] + by_kind["winner"] + by_kind["tie"]
        elif target_mode == "winner":
            priority = by_kind["winner"] + by_kind["solo"] + by_kind["tie"]
        elif target_mode == "tie":
            priority = by_kind["tie"] + by_kind["winner"] + by_kind["solo"]
        else:
            priority = by_kind["winner"] + by_kind["tie"] + by_kind["solo"]
        if not _take_candidate(chosen, used_segments, priority):
            break
    return [tuple(cand["pos"]) for cand in chosen]


def _embed_maze(local_maze: np.ndarray, openings: list[dict[str, object]]) -> np.ndarray:
    top = random.randint(*OUTER_MARGIN_RANGE)
    bottom = random.randint(*OUTER_MARGIN_RANGE)
    left = random.randint(*OUTER_MARGIN_RANGE)
    right = random.randint(*OUTER_MARGIN_RANGE)
    h, w = local_maze.shape
    grid = np.full((top + h + bottom, left + w + right), BLACK, dtype=int)
    grid[top : top + h, left : left + w] = local_maze

    colors = random.sample(KEY_COLORS, len(openings))
    for color, opening in zip(colors, openings):
        r, c = opening["pos"]
        side = (int(opening["dir_idx"]) + 2) % 4
        gr = top + r + DIRS[side][0]
        gc = left + c + DIRS[side][1]
        grid[gr, gc] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    target_mode = random.choice(("solo", "winner", "tie", "mixed"))
    cell_h = random.randint(*MAZE_CELL_RANGE)
    cell_w = random.randint(*MAZE_CELL_RANGE)
    maze = _carve_perfect_maze(cell_h, cell_w)

    candidates = _entrance_candidates(maze)
    openings = _pick_entrances(candidates, target_mode)
    if openings is None:
        return None

    local_maze = maze.copy()
    for opening in openings:
        r, c = opening["pos"]
        local_maze[r, c] = PASSAGE

    base_grid = _embed_maze(local_maze, openings)
    spec = _parse_maze(base_grid)

    hands = {ent.hand for ent in spec.entrances}
    if hands != {"left", "right"}:
        return None

    visit_maps = [_trace_key(spec, ent) for ent in spec.entrances]
    door_positions = _select_doors(spec, visit_maps, target_mode)
    if not door_positions:
        return None

    grid = base_grid.copy()
    for r, c in door_positions:
        gr = spec.offset[0] + r
        gc = spec.offset[1] + c
        grid[gr, gc] = DOOR

    final_spec = _parse_maze(grid)
    if len(final_spec.door_positions) != len(door_positions):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid maze input")
