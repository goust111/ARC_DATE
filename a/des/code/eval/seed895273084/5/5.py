from __future__ import annotations

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
CYAN = 8
MAROON = 9

PRISM_COLORS = (BLUE, RED, GREEN, MAGENTA, CYAN, MAROON)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

TURN_LEFT = {
    DIRS["up"]: DIRS["left"],
    DIRS["left"]: DIRS["down"],
    DIRS["down"]: DIRS["right"],
    DIRS["right"]: DIRS["up"],
}

TURN_RIGHT = {
    DIRS["up"]: DIRS["right"],
    DIRS["right"]: DIRS["down"],
    DIRS["down"]: DIRS["left"],
    DIRS["left"]: DIRS["up"],
}

MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _add(a: tuple[int, int], b: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return a[0] + b[0] * k, a[1] + b[1] * k


def _find_source_and_direction(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    source_cells = np.argwhere(g == YELLOW)
    assert len(source_cells) == 1, "expected exactly one yellow source"
    sr, sc = map(int, source_cells[0])

    orange_neighbors = []
    for dr, dc in DIRS.values():
        nr, nc = sr + dr, sc + dc
        if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == ORANGE:
            orange_neighbors.append((dr, dc))
    assert len(orange_neighbors) == 1, "expected one adjacent orange direction marker"
    return (sr, sc), orange_neighbors[0]


def _is_prism(cell: int) -> bool:
    return int(cell) not in {BLACK, GREY, YELLOW, ORANGE}


def _turn_options(
    g: np.ndarray,
    pos: tuple[int, int],
    direction: tuple[int, int],
    wall: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = pos
    wr, wc = wall
    if direction[0] == 0:
        options = [(-1, 0), (1, 0)]
    else:
        options = [(0, -1), (0, 1)]

    valid: list[tuple[int, int]] = []
    for nd in options:
        side_from_wall = (wr + nd[0], wc + nd[1])
        side_from_pos = (r + nd[0], c + nd[1])
        if not _in_bounds(g.shape, side_from_wall[0], side_from_wall[1]):
            continue
        if not _in_bounds(g.shape, side_from_pos[0], side_from_pos[1]):
            continue
        wall_neighbor = int(g[side_from_wall])
        pos_neighbor = int(g[side_from_pos])
        if wall_neighbor != GREY and pos_neighbor != GREY:
            valid.append(nd)
    return valid


@dataclass
class TraceResult:
    out: np.ndarray
    painted: list[tuple[int, int]]
    prism_hits: list[tuple[int, int, int]]
    reflections: list[tuple[int, int, tuple[int, int], tuple[int, int]]]
    exit_reason: str


def _trace_beam(grid: np.ndarray) -> TraceResult:
    g = np.asarray(grid, dtype=int)
    source, direction = _find_source_and_direction(g)
    current_color = YELLOW
    out = g.copy()
    painted: list[tuple[int, int]] = []
    prism_hits: list[tuple[int, int, int]] = []
    reflections: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []

    r, c = source
    seen: set[tuple[int, int, int, int, int]] = set()

    while True:
        state = (r, c, direction[0], direction[1], current_color)
        if state in seen:
            return TraceResult(out, painted, prism_hits, reflections, "loop")
        seen.add(state)

        nr, nc = r + direction[0], c + direction[1]
        if not _in_bounds(g.shape, nr, nc):
            return TraceResult(out, painted, prism_hits, reflections, "out")
        if (nr, nc) == source:
            return TraceResult(out, painted, prism_hits, reflections, "source")

        cell = int(g[nr, nc])
        if cell == GREY:
            choices = _turn_options(g, (r, c), direction, (nr, nc))
            if len(choices) != 1:
                return TraceResult(out, painted, prism_hits, reflections, "blocked")
            new_direction = choices[0]
            reflections.append((nr, nc, direction, new_direction))
            direction = new_direction
            continue

        r, c = nr, nc
        if cell == BLACK:
            out[r, c] = current_color
            painted.append((r, c))
            continue
        if _is_prism(cell):
            current_color = cell
            prism_hits.append((r, c, cell))
            continue
        # Orange is only a transparent direction marker; yellow is handled by the source check above.
        assert cell == ORANGE, "unexpected non-wall cell type on the beam path"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace an orthogonal light ray through grey walls and embedded colored prisms.

    Input structure
    - black background
    - grey horizontal/vertical wall segments
    - one yellow source with one adjacent orange direction marker
    - 1-3 colored wall prisms

    Rule
    - Emit from the yellow source in the direction indicated by the adjacent orange marker.
    - Paint every visited black cell with the current beam color.
    - Passing through a prism keeps the direction but changes the beam color to the prism color.
    - Hitting a grey wall turns the beam by 90 degrees toward the unique open side determined
      by the wall pixel's perpendicular neighbor cells; if the choice is not unique, the beam stops.
    - Stop when the beam leaves the grid or returns to the source.
    """
    return _trace_beam(np.asarray(grid, dtype=int)).out


@dataclass
class Segment:
    direction: tuple[int, int]
    cells: list[tuple[int, int]]


@dataclass
class Reflector:
    wall_cell: tuple[int, int]
    turn_dir: tuple[int, int]
    cells: list[tuple[int, int]]


def _segment_lengths_for_turns(turn_count: int) -> list[int]:
    base = [random.randint(3, 6) for _ in range(turn_count)]
    if random.random() < 0.45:
        idx = random.randrange(turn_count)
        base[idx] += random.randint(1, 3)
    return base


def _available_run(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cells = []
    r, c = start
    while True:
        r += direction[0]
        c += direction[1]
        if not _in_bounds(shape, r, c):
            break
        if (r, c) in blocked:
            break
        cells.append((r, c))
    return cells


def _wall_candidates(
    shape: tuple[int, int],
    endpoint: tuple[int, int],
    direction: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], Reflector]]:
    wall = _add(endpoint, direction)
    if not _in_bounds(shape, wall[0], wall[1]) or wall in blocked:
        return []

    candidates = []
    for turn_dir in (TURN_LEFT[direction], TURN_RIGHT[direction]):
        first_next = _add(endpoint, turn_dir)
        open_by_wall = _add(wall, turn_dir)
        blocked_by_wall = _add(wall, (-turn_dir[0], -turn_dir[1]))
        if not _in_bounds(shape, first_next[0], first_next[1]) or first_next in blocked:
            continue
        if not _in_bounds(shape, open_by_wall[0], open_by_wall[1]) or open_by_wall in blocked:
            continue
        if not _in_bounds(shape, blocked_by_wall[0], blocked_by_wall[1]) or blocked_by_wall in blocked:
            continue

        max_len = 1
        while True:
            nxt = _add(wall, (-turn_dir[0], -turn_dir[1]), max_len)
            if not _in_bounds(shape, nxt[0], nxt[1]) or nxt in blocked:
                break
            max_len += 1
        if max_len < 2:
            continue

        wall_len = random.randint(2, min(max_len, 5))
        cells = [_add(wall, (-turn_dir[0], -turn_dir[1]), k) for k in range(wall_len)]
        reflector = Reflector(wall_cell=wall, turn_dir=turn_dir, cells=cells)
        candidates.append((turn_dir, reflector))
    random.shuffle(candidates)
    return candidates


def _build_route(
    shape: tuple[int, int],
    source: tuple[int, int],
    direction: tuple[int, int],
    turn_count: int,
) -> tuple[list[Segment], list[Reflector]] | None:
    planned_lengths = _segment_lengths_for_turns(turn_count)
    segments: list[Segment] = []
    reflectors: list[Reflector] = []
    blocked: set[tuple[int, int]] = {source}

    def rec(
        pos: tuple[int, int],
        current_dir: tuple[int, int],
        turn_idx: int,
        blocked_now: set[tuple[int, int]],
    ) -> bool:
        if turn_idx == turn_count:
            final_cells = _available_run(shape, pos, current_dir, blocked_now)
            if len(final_cells) < 2:
                return False
            segments.append(Segment(direction=current_dir, cells=final_cells))
            return True

        min_len = planned_lengths[turn_idx]
        forward = _available_run(shape, pos, current_dir, blocked_now)
        if len(forward) < min_len + 1:
            return False

        max_choice = min(len(forward) - 1, min_len + 4)
        lengths = list(range(min_len, max_choice + 1))
        random.shuffle(lengths)

        for seg_len in lengths:
            cells = forward[:seg_len]
            endpoint = cells[-1]
            trial_blocked = set(blocked_now)
            trial_blocked.update(cells)

            for next_dir, reflector in _wall_candidates(shape, endpoint, current_dir, trial_blocked):
                future_blocked = set(trial_blocked)
                future_blocked.update(reflector.cells)
                next_pos = endpoint

                segments.append(Segment(direction=current_dir, cells=cells))
                reflectors.append(reflector)
                if rec(next_pos, next_dir, turn_idx + 1, future_blocked):
                    return True
                reflectors.pop()
                segments.pop()
        return False

    if not rec(source, direction, 0, blocked):
        return None
    return segments, reflectors


def _candidate_prism_slots(
    shape: tuple[int, int],
    source: tuple[int, int],
    first_marker: tuple[int, int],
    segments: list[Segment],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int, tuple[int, int], tuple[int, int]]]:
    candidates = []
    for seg_idx, seg in enumerate(segments):
        if len(seg.cells) < 4:
            continue
        if seg.direction[0] == 0:
            perp = (-1, 0)
        else:
            perp = (0, -1)

        for i, cell in enumerate(seg.cells):
            if cell == first_marker:
                continue
            if seg_idx == len(segments) - 1 and i >= len(seg.cells) - 1:
                continue
            if i == len(seg.cells) - 1:
                continue
            if i == 0 and cell == first_marker:
                continue
            if i < 1 or i > len(seg.cells) - 2:
                continue

            a = _add(cell, perp)
            b = _add(cell, (-perp[0], -perp[1]))
            if not _in_bounds(shape, a[0], a[1]) or not _in_bounds(shape, b[0], b[1]):
                continue
            if a in reserved or b in reserved:
                continue
            candidates.append((seg_idx, i, cell, perp))
    random.shuffle(candidates)
    return candidates


def _choose_prisms(
    shape: tuple[int, int],
    source: tuple[int, int],
    first_marker: tuple[int, int],
    segments: list[Segment],
    reserved: set[tuple[int, int]],
    prism_count: int,
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    slots = _candidate_prism_slots(shape, source, first_marker, segments, reserved)
    used_cells: set[tuple[int, int]] = set()
    used_neigh: set[tuple[int, int]] = set()
    chosen: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    colors = list(PRISM_COLORS)
    random.shuffle(colors)

    for _seg_idx, _i, cell, perp in slots:
        a = _add(cell, perp)
        b = _add(cell, (-perp[0], -perp[1]))
        if cell in used_cells or a in used_neigh or b in used_neigh:
            continue
        used_cells.add(cell)
        used_neigh.add(a)
        used_neigh.add(b)
        chosen.append((cell, perp, colors[len(chosen) % len(colors)]))
        if len(chosen) == prism_count:
            break
    return chosen


def _render_input(
    shape: tuple[int, int],
    source: tuple[int, int],
    direction: tuple[int, int],
    segments: list[Segment],
    reflectors: list[Reflector],
    prisms: list[tuple[tuple[int, int], tuple[int, int], int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    grid[source] = YELLOW
    first_marker = _add(source, direction)
    grid[first_marker] = ORANGE

    for reflector in reflectors:
        for cell in reflector.cells:
            grid[cell] = GREY

    prism_cells = {cell: (perp, color) for cell, perp, color in prisms}
    for cell, (perp, color) in prism_cells.items():
        a = _add(cell, perp)
        b = _add(cell, (-perp[0], -perp[1]))
        grid[a] = GREY
        grid[b] = GREY
        grid[cell] = color

    return grid


def _beam_path_color_count(out: np.ndarray, inp: np.ndarray) -> int:
    changed = (out != inp) & (inp == BLACK)
    return int(np.count_nonzero(changed))


def _apply_random_dihedral(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    rot = random.randint(0, 3)
    g = np.rot90(g, k=rot)
    if random.random() < 0.5:
        g = np.fliplr(g)
    return np.array(g, copy=True)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 24)
    w = random.randint(11, 24)
    shape = (h, w)

    turn_count = random.choice([1, 2, 3, 4, 5])
    prism_count = random.choice([1, 2, 3])
    direction = random.choice(list(DIRS.values()))

    margin = 2
    source = (
        random.randint(margin, h - 1 - margin),
        random.randint(margin, w - 1 - margin),
    )
    marker = _add(source, direction)
    if not _in_bounds(shape, marker[0], marker[1]):
        return None

    route = _build_route(shape, source, direction, turn_count)
    if route is None:
        return None
    segments, reflectors = route

    reserved: set[tuple[int, int]] = {source, marker}
    for seg in segments:
        reserved.update(seg.cells)
    for reflector in reflectors:
        reserved.update(reflector.cells)

    prisms = _choose_prisms(shape, source, marker, segments, reserved, prism_count)
    if len(prisms) != prism_count:
        return None

    inp = _render_input(shape, source, direction, segments, reflectors, prisms)
    trace = _trace_beam(inp)
    if trace.exit_reason != "out":
        return None
    if len(trace.reflections) != turn_count:
        return None
    if len(trace.prism_hits) != prism_count:
        return None
    if _beam_path_color_count(trace.out, inp) < 5:
        return None
    if len({int(color) for _r, _c, color in trace.prism_hits}) != prism_count:
        return None

    # Keep the light route readable: no wall should touch the source except the marker.
    sr, sc = source
    for dr, dc in DIRS.values():
        nr, nc = sr + dr, sc + dc
        if not _in_bounds(shape, nr, nc):
            continue
        if (nr, nc) == marker:
            continue
        if int(inp[nr, nc]) == GREY or _is_prism(int(inp[nr, nc])):
            return None

    transformed = _apply_random_dihedral(inp)
    transformed_trace = _trace_beam(transformed)
    if transformed_trace.exit_reason != "out":
        return None
    if len(transformed_trace.reflections) != turn_count:
        return None
    if len(transformed_trace.prism_hits) != prism_count:
        return None
    return transformed


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-room instance")
