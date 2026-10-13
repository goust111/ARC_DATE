from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


AZURE = 8
RED = 2
GREEN = 3
WALL = 5
BG_POOL = [0, 1, 4, 6, 7, 9]

UP, RIGHT, DOWN, LEFT = range(4)
DIRS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}
TURN_LEFT = {UP: LEFT, RIGHT: UP, DOWN: RIGHT, LEFT: DOWN}
TURN_RIGHT = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
PRISM_TURN = {RED: "L", GREEN: "R"}


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _edge_direction(h: int, w: int, r: int, c: int) -> int | None:
    sides = []
    if r == 0:
        sides.append(DOWN)
    if r == h - 1:
        sides.append(UP)
    if c == 0:
        sides.append(RIGHT)
    if c == w - 1:
        sides.append(LEFT)
    if len(sides) != 1:
        return None
    return sides[0]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(h, w, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_prisms(grid: np.ndarray) -> tuple[dict[tuple[int, int], tuple[int, str]], dict[tuple[int, int], tuple[int, int, str]]]:
    prisms: dict[tuple[int, int], tuple[int, str]] = {}
    by_cell: dict[tuple[int, int], tuple[int, int, str]] = {}
    for color in (RED, GREEN):
        turn = PRISM_TURN[color]
        for comp in _components_of_color(grid, color):
            if len(comp) != 4:
                continue
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue
            expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            if set(comp) != expected:
                continue
            prisms[(r0, c0)] = (color, turn)
            for cell in expected:
                by_cell[cell] = (r0, c0, turn)
    return prisms, by_cell


def _child_launches(pr: int, pc: int, direction: int, turn: str, offset: int) -> list[tuple[int, int, int]]:
    if direction == RIGHT:
        straight = (pr + offset, pc + 2, RIGHT)
        turned = (pr - 1, pc + offset, UP) if turn == "L" else (pr + 2, pc + offset, DOWN)
    elif direction == LEFT:
        straight = (pr + offset, pc - 1, LEFT)
        turned = (pr + 2, pc + offset, DOWN) if turn == "L" else (pr - 1, pc + offset, UP)
    elif direction == DOWN:
        straight = (pr + 2, pc + offset, DOWN)
        turned = (pr + offset, pc + 2, RIGHT) if turn == "L" else (pr + offset, pc - 1, LEFT)
    else:
        straight = (pr - 1, pc + offset, UP)
        turned = (pr + offset, pc - 1, LEFT) if turn == "L" else (pr + offset, pc + 2, RIGHT)
    return [straight, turned]


def _split_from_prism(pr: int, pc: int, direction: int, hit_r: int, hit_c: int, turn: str) -> list[tuple[int, int, int]]:
    offset: int | None = None
    if direction == RIGHT and hit_c == pc and hit_r in (pr, pr + 1):
        offset = hit_r - pr
    elif direction == LEFT and hit_c == pc + 1 and hit_r in (pr, pr + 1):
        offset = hit_r - pr
    elif direction == DOWN and hit_r == pr and hit_c in (pc, pc + 1):
        offset = hit_c - pc
    elif direction == UP and hit_r == pr + 1 and hit_c in (pc, pc + 1):
        offset = hit_c - pc
    if offset is None:
        return []
    return _child_launches(pr, pc, direction, turn, offset)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate beam splitters encoded as 2x2 red/green prisms.

    - Azure edge pixels are beam sources.
    - Red prisms split into straight + left-turn beams.
    - Green prisms split into straight + right-turn beams.
    - Beams only paint background cells and stop on walls, boundaries, existing
      beam paths, or invalid objects. Prism cells stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _dominant_color(g)
    _, prism_by_cell = _find_prisms(g)

    sources: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != AZURE:
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if (r, c) in prism_by_cell:
                continue
            direction = _edge_direction(h, w, r, c)
            if direction is not None:
                sources.append((r, c, direction))
    sources.sort()

    out = g.copy()
    launches = deque((r, c, d, False) for r, c, d in sources)
    seen_launches: set[tuple[int, int, int, bool]] = set()

    while launches:
        r, c, direction, paint_first = launches.popleft()
        state = (r, c, direction, paint_first)
        if state in seen_launches:
            continue
        seen_launches.add(state)

        if not _in_bounds(h, w, r, c):
            continue
        if paint_first:
            if int(out[r, c]) == AZURE or int(g[r, c]) != bg:
                continue
            out[r, c] = AZURE

        rr, cc = r, c
        dr, dc = DIRS[direction]
        while True:
            nr, nc = rr + dr, cc + dc
            if not _in_bounds(h, w, nr, nc):
                break
            if int(out[nr, nc]) == AZURE:
                break
            prism = prism_by_cell.get((nr, nc))
            if prism is not None:
                pr, pc, turn = prism
                for sr, sc, sd in _split_from_prism(pr, pc, direction, nr, nc, turn):
                    if not _in_bounds(h, w, sr, sc):
                        continue
                    if int(out[sr, sc]) == AZURE or int(g[sr, sc]) != bg:
                        continue
                    launches.append((sr, sc, sd, True))
                break
            if int(g[nr, nc]) != bg:
                break
            out[nr, nc] = AZURE
            rr, cc = nr, nc

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
SOURCE_GAP = 4
TARGET_PRISM_MAX = 5
MAX_DEPTH_RANGE = (2, 3)
DISTRACTOR_WALL_MAX = 3
TERMINAL_WALL_PROB = 0.45
MAX_GENERATION_RETRIES = 400

EDGE_TOP = "top"
EDGE_RIGHT = "right"
EDGE_BOTTOM = "bottom"
EDGE_LEFT = "left"


def _choose_spaced_positions(length: int, count: int, gap: int) -> list[int] | None:
    choices = list(range(1, length - 1))
    if len(choices) < count:
        return None
    picked: list[int] = []
    for _ in range(count):
        valid = [x for x in choices if all(abs(x - y) >= gap for y in picked)]
        if not valid:
            return None
        picked.append(random.choice(valid))
    picked.sort()
    return picked


def _source_cell(edge: str, pos: int, h: int, w: int) -> tuple[int, int, int]:
    if edge == EDGE_TOP:
        return 0, pos, DOWN
    if edge == EDGE_BOTTOM:
        return h - 1, pos, UP
    if edge == EDGE_LEFT:
        return pos, 0, RIGHT
    return pos, w - 1, LEFT


def _segment_before_prism(r: int, c: int, direction: int, dist: int) -> list[tuple[int, int]]:
    if direction == RIGHT:
        return [(r, x) for x in range(c + 1, c + dist)]
    if direction == LEFT:
        return [(r, x) for x in range(c - 1, c - dist, -1)]
    if direction == DOWN:
        return [(y, c) for y in range(r + 1, r + dist)]
    return [(y, c) for y in range(r - 1, r - dist, -1)]


def _prism_top_left_from_dist(r: int, c: int, direction: int, dist: int, offset: int) -> tuple[int, int]:
    if direction == RIGHT:
        return r - offset, c + dist
    if direction == LEFT:
        return r - offset, c - dist - 1
    if direction == DOWN:
        return r + dist, c - offset
    return r - dist - 1, c - offset


def _clear_line_to_edge(
    h: int,
    w: int,
    r: int,
    c: int,
    direction: int,
    path_reserved: set[tuple[int, int]],
    solid_reserved: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    dr, dc = DIRS[direction]
    line: list[tuple[int, int]] = []
    nr, nc = r + dr, c + dc
    while _in_bounds(h, w, nr, nc):
        if (nr, nc) in path_reserved or (nr, nc) in solid_reserved:
            return None
        line.append((nr, nc))
        nr += dr
        nc += dc
    return line


def _prism_has_buffer(prism_cells: set[tuple[int, int]], solid_reserved: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in prism_cells:
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) in solid_reserved and (nr, nc) not in prism_cells:
                return False
    return True


def _try_place_split(
    h: int,
    w: int,
    r: int,
    c: int,
    direction: int,
    depth: int,
    ctx: dict[str, object],
) -> list[tuple[int, int, int]] | None:
    path_reserved: set[tuple[int, int]] = ctx["path_reserved"]  # type: ignore[assignment]
    solid_reserved: set[tuple[int, int]] = ctx["solid_reserved"]  # type: ignore[assignment]
    color_counts: Counter[str] = ctx["color_counts"]  # type: ignore[assignment]

    preferred_turns = ["L", "R"]
    if color_counts["L"] == 0 and color_counts["R"] > 0:
        preferred_turns = ["L", "R"]
    elif color_counts["R"] == 0 and color_counts["L"] > 0:
        preferred_turns = ["R", "L"]
    else:
        random.shuffle(preferred_turns)

    max_dist = max(h, w)
    distances = list(range(2, max_dist + 1))
    random.shuffle(distances)
    offsets = [0, 1]
    random.shuffle(offsets)

    for turn in preferred_turns:
        for dist in distances:
            for offset in offsets:
                pr, pc = _prism_top_left_from_dist(r, c, direction, dist, offset)
                if not _in_bounds(h, w, pr, pc) or not _in_bounds(h, w, pr + 1, pc + 1):
                    continue
                prism_cells = {(pr, pc), (pr, pc + 1), (pr + 1, pc), (pr + 1, pc + 1)}
                segment = _segment_before_prism(r, c, direction, dist)
                if any(cell in path_reserved or cell in solid_reserved for cell in segment):
                    continue
                if any(cell in path_reserved or cell in solid_reserved for cell in prism_cells):
                    continue
                if not _prism_has_buffer(prism_cells, solid_reserved, h, w):
                    continue

                children = _child_launches(pr, pc, direction, turn, offset)
                if len({(sr, sc) for sr, sc, _ in children}) != 2:
                    continue
                valid = True
                for sr, sc, _ in children:
                    if not _in_bounds(h, w, sr, sc):
                        valid = False
                        break
                    if (sr, sc) in path_reserved or (sr, sc) in solid_reserved:
                        valid = False
                        break
                if not valid:
                    continue

                for cell in segment:
                    path_reserved.add(cell)
                for cell in prism_cells:
                    solid_reserved.add(cell)
                for sr, sc, _ in children:
                    path_reserved.add((sr, sc))

                color = RED if turn == "L" else GREEN
                prisms: list[tuple[int, int, int]] = ctx["prisms"]  # type: ignore[assignment]
                prisms.append((pr, pc, color))
                prism_depths: list[int] = ctx["prism_depths"]  # type: ignore[assignment]
                prism_depths.append(depth)
                color_counts[turn] += 1
                ctx["visible_splits"] = int(ctx["visible_splits"]) + 1
                return children
    return None


def _terminate_branch(h: int, w: int, r: int, c: int, direction: int, ctx: dict[str, object]) -> bool:
    path_reserved: set[tuple[int, int]] = ctx["path_reserved"]  # type: ignore[assignment]
    solid_reserved: set[tuple[int, int]] = ctx["solid_reserved"]  # type: ignore[assignment]
    source_edges: set[int] = ctx["source_edges"]  # type: ignore[assignment]
    line = _clear_line_to_edge(h, w, r, c, direction, path_reserved, solid_reserved)
    if line is None:
        return False
    if not line:
        return True

    target_edge = {
        UP: EDGE_TOP,
        RIGHT: EDGE_RIGHT,
        DOWN: EDGE_BOTTOM,
        LEFT: EDGE_LEFT,
    }[direction]
    if target_edge not in source_edges and random.random() < TERMINAL_WALL_PROB:
        wall_cell = line[-1]
        if wall_cell not in path_reserved and wall_cell not in solid_reserved:
            for cell in line[:-1]:
                path_reserved.add(cell)
            solid_reserved.add(wall_cell)
            walls: set[tuple[int, int]] = ctx["walls"]  # type: ignore[assignment]
            walls.add(wall_cell)
            ctx["has_terminal_wall"] = True
            return True

    for cell in line:
        path_reserved.add(cell)
    return True


def _grow_branch(h: int, w: int, r: int, c: int, direction: int, depth: int, ctx: dict[str, object]) -> bool:
    target_prisms = int(ctx["target_prisms"])
    max_depth = int(ctx["max_depth"])
    prism_count = len(ctx["prisms"])  # type: ignore[arg-type]

    if depth < max_depth and prism_count < target_prisms:
        children = _try_place_split(h, w, r, c, direction, depth, ctx)
        if children is not None:
            for sr, sc, sd in children:
                if not _grow_branch(h, w, sr, sc, sd, depth + 1, ctx):
                    return False
            return True
    return _terminate_branch(h, w, r, c, direction, ctx)


def _place_distractor_walls(h: int, w: int, ctx: dict[str, object]) -> None:
    path_reserved: set[tuple[int, int]] = ctx["path_reserved"]  # type: ignore[assignment]
    solid_reserved: set[tuple[int, int]] = ctx["solid_reserved"]  # type: ignore[assignment]
    source_edges: set[int] = ctx["source_edges"]  # type: ignore[assignment]
    walls: set[tuple[int, int]] = ctx["walls"]  # type: ignore[assignment]

    candidates: list[tuple[int, int]] = []
    if EDGE_TOP not in source_edges:
        candidates.extend((0, c) for c in range(1, w - 1))
    if EDGE_BOTTOM not in source_edges:
        candidates.extend((h - 1, c) for c in range(1, w - 1))
    if EDGE_LEFT not in source_edges:
        candidates.extend((r, 0) for r in range(1, h - 1))
    if EDGE_RIGHT not in source_edges:
        candidates.extend((r, w - 1) for r in range(1, h - 1))

    random.shuffle(candidates)
    extra = random.randint(0, DISTRACTOR_WALL_MAX)
    for cell in candidates:
        if extra == 0:
            break
        if cell in path_reserved or cell in solid_reserved:
            continue
        solid_reserved.add(cell)
        walls.add(cell)
        extra -= 1


def _build_grid(
    h: int,
    w: int,
    bg: int,
    sources: list[tuple[int, int, int]],
    prisms: list[tuple[int, int, int]],
    walls: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for pr, pc, color in prisms:
        grid[pr : pr + 2, pc : pc + 2] = color
    for r, c in walls:
        grid[r, c] = WALL
    for r, c, _ in sources:
        grid[r, c] = AZURE
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice(BG_POOL)

    side_count = 1 if random.random() < 0.6 else 2
    source_sides = random.sample([EDGE_TOP, EDGE_RIGHT, EDGE_BOTTOM, EDGE_LEFT], k=side_count)
    num_sources = random.choices([1, 2, 3], weights=[4, 4, 1], k=1)[0]
    num_sources = max(num_sources, side_count)

    counts = {side: 1 for side in source_sides}
    for _ in range(num_sources - side_count):
        counts[random.choice(source_sides)] += 1

    sources: list[tuple[int, int, int]] = []
    for side in source_sides:
        length = w if side in (EDGE_TOP, EDGE_BOTTOM) else h
        positions = _choose_spaced_positions(length, counts[side], SOURCE_GAP)
        if positions is None:
            return None
        for pos in positions:
            sources.append(_source_cell(side, pos, h, w))
    sources.sort()

    min_prisms = max(2, len(sources) + 1)
    target_prisms = random.randint(min_prisms, min(TARGET_PRISM_MAX, min_prisms + 1))
    ctx: dict[str, object] = {
        "path_reserved": {(r, c) for r, c, _ in sources},
        "solid_reserved": {(r, c) for r, c, _ in sources},
        "prisms": [],
        "walls": set(),
        "prism_depths": [],
        "color_counts": Counter(),
        "visible_splits": 0,
        "target_prisms": target_prisms,
        "max_depth": random.randint(*MAX_DEPTH_RANGE),
        "source_edges": set(source_sides),
        "has_terminal_wall": False,
    }

    for r, c, direction in sources:
        if not _grow_branch(h, w, r, c, direction, 0, ctx):
            return None

    prisms: list[tuple[int, int, int]] = ctx["prisms"]  # type: ignore[assignment]
    prism_depths: list[int] = ctx["prism_depths"]  # type: ignore[assignment]
    color_counts: Counter[str] = ctx["color_counts"]  # type: ignore[assignment]
    if len(prisms) < min_prisms:
        return None
    if max(prism_depths, default=-1) < 1:
        return None
    if color_counts["L"] == 0 or color_counts["R"] == 0:
        return None
    if int(ctx["visible_splits"]) < 2:
        return None

    _place_distractor_walls(h, w, ctx)
    walls: set[tuple[int, int]] = ctx["walls"]  # type: ignore[assignment]
    grid = _build_grid(h, w, bg, sources, prisms, walls)

    if _dominant_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out == AZURE)) <= int(np.count_nonzero(grid == AZURE)):
        return None
    for pr, pc, color in prisms:
        if not np.all(out[pr : pr + 2, pc : pc + 2] == color):
            return None
    for r, c in walls:
        if int(out[r, c]) != WALL:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid splitter puzzle after {MAX_GENERATION_RETRIES} attempts")
