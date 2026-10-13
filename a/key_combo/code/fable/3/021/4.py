import random
from collections import deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GRAY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


SEED_COLORS = [Color.BLUE, Color.GREEN, Color.GRAY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
STAMP_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
GEN_ATTEMPTS = 3000


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _red_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(grid == Color.RED):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if _in_bounds(grid.shape, rr, cc) and not seen[rr, cc] and int(grid[rr, cc]) == Color.RED:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        components.append(comp)
    return components


def _seed_and_direction(grid: np.ndarray) -> tuple[int, tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    seeds = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere((grid != Color.BLACK) & (grid != Color.RED))
    ]
    assert len(seeds) == 1, "expected exactly one non-red seed pixel"
    r, c, color = seeds[0]
    assert color != Color.YELLOW, "seed color must be distinct from the yellow stamp center"
    assert r in (0, h - 1) and c in (0, w - 1), "seed must be in a corner"
    dr = 1 if r == 0 else -1
    dc = 1 if c == 0 else -1
    return color, (r, c), (dr, dc)


def _mirror_orientations(grid: np.ndarray) -> dict[tuple[int, int], str]:
    orientations: dict[tuple[int, int], str] = {}
    for comp in _red_components(grid):
        rows = {r for r, _c in comp}
        cols = {c for _r, c in comp}
        if len(rows) == 1 and len(comp) >= 2:
            orientation = "horizontal"
        elif len(cols) == 1 and len(comp) >= 2:
            orientation = "vertical"
        else:
            raise AssertionError("each red mirror must be a horizontal or vertical segment of length at least two")
        for cell in comp:
            orientations[cell] = orientation
    return orientations


def _trace_events(grid: np.ndarray) -> list[tuple[int, int, int, int, str, int, int]]:
    _seed_color, (r, c), (dr, dc) = _seed_and_direction(grid)
    orientations = _mirror_orientations(grid)
    events: list[tuple[int, int, int, int, str, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray path must not loop"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            break
        if int(grid[nr, nc]) == Color.RED:
            orientation = orientations.get((nr, nc))
            assert orientation is not None, "red hit cell must belong to a valid mirror segment"
            events.append((r, c, nr, nc, orientation, dr, dc))
            if orientation == "horizontal":
                dr = -dr
            else:
                dc = -dc
            continue

        r, c = nr, nc

    return events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp only the invisible diagonal ray's red-mirror reflection turns.

    Input structure:
    - The grid has a black background, one non-red seed pixel in a corner, and red mirror segments.
    - Each red mirror component is a one-cell-thick horizontal or vertical segment.
    - Some red mirror segments may be decoys that are never reached by the diagonal flight.

    Rule:
    - Start at the seed and move in the only diagonal direction that enters the grid.
    - If the next cell is a red horizontal mirror, do not enter it and flip the row direction; if it
      is a vertical mirror, flip the column direction instead.
    - At every reflection turn, stamp a 3x3 mark centered on the current cell: yellow at the center
      and the seed color at the four diagonal corners, painting only cells that are still black.
    - Keep the original seed and all red mirror cells unchanged, and stop when the flight exits.

    Key insight:
    - The visible output is not the flight path; the whole diagonal bounce route must be reconstructed
      to identify exactly which mirrors are struck and where the hidden turns occur.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a 2D ARC grid"
    seed_color, _seed, _direction = _seed_and_direction(g)
    events = _trace_events(g)

    out = g.copy()
    for turn_r, turn_c, _hit_r, _hit_c, _orientation, _dr, _dc in events:
        if out[turn_r, turn_c] == Color.BLACK:
            out[turn_r, turn_c] = Color.YELLOW
        for rr_off, cc_off in STAMP_OFFSETS:
            rr, cc = turn_r + rr_off, turn_c + cc_off
            if _in_bounds(out.shape, rr, cc) and out[rr, cc] == Color.BLACK:
                out[rr, cc] = seed_color

    return out


def _corner_state(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    choice = random.randrange(4)
    if choice == 0:
        return (0, 0), (1, 1)
    if choice == 1:
        return (0, w - 1), (1, -1)
    if choice == 2:
        return (h - 1, 0), (-1, 1)
    return (h - 1, w - 1), (-1, -1)


def _segment_cells(hit: tuple[int, int], orientation: str, length: int, offset: int) -> list[tuple[int, int]]:
    hr, hc = hit
    if orientation == "horizontal":
        c0 = hc - offset
        return [(hr, c0 + i) for i in range(length)]
    r0 = hr - offset
    return [(r0 + i, hc) for i in range(length)]


def _segment_is_clear(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    blocked_stamp_cells: set[tuple[int, int]] | None = None,
) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != Color.BLACK:
            return False
        if blocked_stamp_cells is not None and (r, c) in blocked_stamp_cells:
            return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in cell_set:
                    continue
                if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == Color.RED:
                    return False
    return True


def _candidate_segments(
    grid: np.ndarray,
    hit: tuple[int, int],
    turn: tuple[int, int],
    incoming: tuple[int, int],
    orientation: str,
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    max_len = min(6, w if orientation == "horizontal" else h)
    hit_r, hit_c = hit
    turn_r, turn_c = turn
    dr, dc = incoming
    other_stamp_corners = {
        (turn_r + rr, turn_c + cc)
        for rr, cc in STAMP_OFFSETS
        if (turn_r + rr, turn_c + cc) != (hit_r, hit_c)
    }
    other_stamp_corners.add(turn)

    options: list[list[tuple[int, int]]] = []
    for length in range(2, max_len + 1):
        for offset in range(length):
            cells = _segment_cells(hit, orientation, length, offset)
            if not _segment_is_clear(grid, cells, other_stamp_corners):
                continue
            # Require the hit cell to be the only segment cell on the current diagonal approach.
            before_hit = {(hit_r - k * dr, hit_c - k * dc) for k in range(1, length + 2)}
            if any(cell in before_hit for cell in cells if cell != hit):
                continue
            options.append(cells)
    random.shuffle(options)
    return options


def _path_cells_until_block(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    step = 1
    while True:
        nr, nc = r + step * dr, c + step * dc
        if not _in_bounds(grid.shape, nr, nc):
            return cells
        if int(grid[nr, nc]) == Color.RED:
            return cells
        cells.append((nr, nc))
        step += 1


def _place_planned_mirror(
    grid: np.ndarray,
    state: tuple[int, int, int, int],
    orientation: str,
) -> tuple[int, int, int, int] | None:
    r, c, dr, dc = state
    candidates: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
    clear_path = _path_cells_until_block(grid, r, c, dr, dc)

    for distance, hit in enumerate(clear_path, start=1):
        if distance < 3:
            continue
        hit_r, hit_c = hit
        turn = (hit_r - dr, hit_c - dc)
        turn_r, turn_c = turn
        if not (1 <= turn_r < grid.shape[0] - 1 and 1 <= turn_c < grid.shape[1] - 1):
            continue
        if int(grid[turn_r, turn_c]) != Color.BLACK:
            continue
        path_before = clear_path[: distance - 1]
        if any(int(grid[rr, cc]) != Color.BLACK for rr, cc in path_before):
            continue
        for segment in _candidate_segments(grid, hit, turn, (dr, dc), orientation):
            candidates.append((hit, turn, segment))

    if not candidates:
        return None

    random.shuffle(candidates)
    _hit, (turn_r, turn_c), segment = candidates[0]
    for rr, cc in segment:
        grid[rr, cc] = Color.RED

    if orientation == "horizontal":
        dr = -dr
    else:
        dc = -dc
    return turn_r, turn_c, dr, dc


def _trace_signature(events: list[tuple[int, int, int, int, str, int, int]]) -> tuple[tuple[int, int, int, int, str], ...]:
    return tuple((tr, tc, hr, hc, orientation) for tr, tc, hr, hc, orientation, _dr, _dc in events)


def _component_index(grid: np.ndarray) -> dict[tuple[int, int], int]:
    index: dict[tuple[int, int], int] = {}
    for i, comp in enumerate(_red_components(grid)):
        for cell in comp:
            index[cell] = i
    return index


def _try_add_decoys(grid: np.ndarray, base_signature: tuple[tuple[int, int, int, int, str], ...]) -> bool:
    wanted = random.randint(1, 3)
    added = 0
    h, w = grid.shape
    for _ in range(250):
        if added >= wanted:
            break
        orientation = random.choice(["horizontal", "vertical"])
        limit = w if orientation == "horizontal" else h
        length = random.randint(2, min(6, limit))
        if orientation == "horizontal":
            r = random.randrange(h)
            c0 = random.randrange(0, w - length + 1)
            cells = [(r, c0 + i) for i in range(length)]
        else:
            r0 = random.randrange(0, h - length + 1)
            c = random.randrange(w)
            cells = [(r0 + i, c) for i in range(length)]
        if not _segment_is_clear(grid, cells):
            continue

        trial = grid.copy()
        for rr, cc in cells:
            trial[rr, cc] = Color.RED
        try:
            if _trace_signature(_trace_events(trial)) != base_signature:
                continue
        except AssertionError:
            continue
        grid[:, :] = trial
        added += 1
    return added >= 1


def _semantic_check(grid: np.ndarray, target_hits: int) -> bool:
    try:
        seed_color, _seed, _direction = _seed_and_direction(grid)
        events = _trace_events(grid)
        if len(events) != target_hits:
            return False
        orientations = [event[4] for event in events]
        if "horizontal" not in orientations or "vertical" not in orientations:
            return False

        comp_index = _component_index(grid)
        hit_components = {comp_index[(hr, hc)] for _tr, _tc, hr, hc, _ori, _dr, _dc in events}
        if len(hit_components) != len(events):
            return False
        if len(_red_components(grid)) <= len(hit_components):
            return False

        out = main(grid)
        if not np.array_equal(out[grid == Color.RED], grid[grid == Color.RED]):
            return False
        seed_cells = np.argwhere(grid == seed_color)
        if seed_cells.shape[0] != 1:
            return False

        centers = set()
        for turn_r, turn_c, hit_r, hit_c, _ori, _dr, _dc in events:
            if (turn_r, turn_c) in centers:
                return False
            centers.add((turn_r, turn_c))
            if int(grid[turn_r, turn_c]) != Color.BLACK or int(out[turn_r, turn_c]) != Color.YELLOW:
                return False
            diagonals = [(turn_r + rr, turn_c + cc) for rr, cc in STAMP_OFFSETS]
            if not all(_in_bounds(grid.shape, rr, cc) for rr, cc in diagonals):
                return False
            values = [int(out[rr, cc]) for rr, cc in diagonals]
            if values.count(seed_color) != 3 or values.count(Color.RED) != 1:
                return False
            if int(out[hit_r, hit_c]) != Color.RED:
                return False

        return bool(np.any(out != grid))
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    grid = np.zeros((h, w), dtype=int)
    seed, (dr, dc) = _corner_state((h, w))
    seed_color = random.choice(SEED_COLORS)
    grid[seed] = seed_color

    target_hits = random.randint(3, 5)
    orientations = ["horizontal", "vertical"]
    orientations.extend(random.choice(["horizontal", "vertical"]) for _ in range(target_hits - 2))
    random.shuffle(orientations)

    state = (seed[0], seed[1], dr, dc)
    seen_states = {state}
    for orientation in orientations:
        new_state = _place_planned_mirror(grid, state, orientation)
        if new_state is None:
            return None
        state = new_state
        if state in seen_states:
            return None
        seen_states.add(state)

    try:
        base_events = _trace_events(grid)
    except AssertionError:
        return None
    if len(base_events) != target_hits:
        return None
    base_signature = _trace_signature(base_events)
    if not _try_add_decoys(grid, base_signature):
        return None
    if not _semantic_check(grid, target_hits):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible stamped bounce input")
