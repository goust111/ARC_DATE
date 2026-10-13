import random
from dataclasses import dataclass

import numpy as np


BG = 0
RED = 2
YELLOW = 4
PIPE_COLOR_POOL = [1, 3, 5, 6, 7, 8, 9]

SHIFT_BY_DIRECTION = {
    "up": (-2, 0),
    "down": (2, 0),
    "left": (0, -2),
    "right": (0, 2),
}


@dataclass
class PipeItem:
    axis: int
    row: int
    col: int
    kind: str
    direction: str | None = None
    marker_row: int | None = None
    marker_col: int | None = None


@dataclass(frozen=True)
class GenValve:
    axis: int
    direction: str


@dataclass(frozen=True)
class GenPipe:
    color: int
    orientation: str
    fixed: int
    start: int
    end: int
    valves: tuple[GenValve, ...]


def _bg_color(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _horizontal_marker_direction(grid: np.ndarray, row: int, col: int) -> str | None:
    up = row > 0 and int(grid[row - 1, col]) == YELLOW
    down = row + 1 < grid.shape[0] and int(grid[row + 1, col]) == YELLOW
    if up == down:
        return None
    return "up" if up else "down"


def _vertical_marker_direction(grid: np.ndarray, row: int, col: int) -> str | None:
    left = col > 0 and int(grid[row, col - 1]) == YELLOW
    right = col + 1 < grid.shape[1] and int(grid[row, col + 1]) == YELLOW
    if left == right:
        return None
    return "left" if left else "right"


def _parse_pipe(grid: np.ndarray, color: int) -> tuple[str, list[PipeItem]]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, "pipe color must exist"
    rows = sorted({int(r) for r, _ in coords})
    cols = sorted({int(c) for _, c in coords})
    h, w = grid.shape

    if len(rows) == 1:
        row = rows[0]
        valves: dict[int, str] = {}
        for rr, cc in np.argwhere(grid == RED):
            rr = int(rr)
            cc = int(cc)
            if rr != row:
                continue
            direction = _horizontal_marker_direction(grid, rr, cc)
            if direction is None:
                continue
            if (cc > 0 and int(grid[rr, cc - 1]) == color) or (cc + 1 < w and int(grid[rr, cc + 1]) == color):
                valves[cc] = direction

        axis_values = set(cols) | set(valves)
        start = min(axis_values)
        end = max(axis_values)
        items: list[PipeItem] = []
        for axis in range(start, end + 1):
            if axis in valves:
                marker_row = row - 1 if valves[axis] == "up" else row + 1
                items.append(
                    PipeItem(
                        axis=axis,
                        row=row,
                        col=axis,
                        kind="valve",
                        direction=valves[axis],
                        marker_row=marker_row,
                        marker_col=axis,
                    )
                )
            else:
                assert int(grid[row, axis]) == color, "horizontal pipe must be contiguous except for valves"
                items.append(PipeItem(axis=axis, row=row, col=axis, kind="cell"))
        return "h", items

    if len(cols) == 1:
        col = cols[0]
        valves = {}
        for rr, cc in np.argwhere(grid == RED):
            rr = int(rr)
            cc = int(cc)
            if cc != col:
                continue
            direction = _vertical_marker_direction(grid, rr, cc)
            if direction is None:
                continue
            if (rr > 0 and int(grid[rr - 1, cc]) == color) or (rr + 1 < h and int(grid[rr + 1, cc]) == color):
                valves[rr] = direction

        axis_values = set(rows) | set(valves)
        start = min(axis_values)
        end = max(axis_values)
        items = []
        for axis in range(start, end + 1):
            if axis in valves:
                marker_col = col - 1 if valves[axis] == "left" else col + 1
                items.append(
                    PipeItem(
                        axis=axis,
                        row=axis,
                        col=col,
                        kind="valve",
                        direction=valves[axis],
                        marker_row=axis,
                        marker_col=marker_col,
                    )
                )
            else:
                assert int(grid[axis, col]) == color, "vertical pipe must be contiguous except for valves"
                items.append(PipeItem(axis=axis, row=axis, col=col, kind="cell"))
        return "v", items

    raise AssertionError("each pipe color must form a straight line")


def _copy_items(items: list[PipeItem]) -> list[PipeItem]:
    return [
        PipeItem(
            axis=item.axis,
            row=item.row,
            col=item.col,
            kind=item.kind,
            direction=item.direction,
            marker_row=item.marker_row,
            marker_col=item.marker_col,
        )
        for item in items
    ]


def _shift_items(items: list[PipeItem], dr: int, dc: int) -> None:
    for item in items:
        item.row += dr
        item.col += dc
        if item.kind == "valve":
            assert item.marker_row is not None and item.marker_col is not None
            item.marker_row += dr
            item.marker_col += dc


def _prefix_stays(direction: str) -> bool:
    # The text leaves the two collinear halves symmetric; the generator and solver
    # use the natural screen-order convention: up/left pins the prefix half,
    # down/right pins the suffix half.
    return direction in ("up", "left")


def _simulate_pipe_cells(items: list[PipeItem]) -> list[tuple[int, int]]:
    active = _copy_items(items)
    settled: list[tuple[int, int]] = []

    while True:
        valve_index = next((i for i, item in enumerate(active) if item.kind == "valve"), None)
        if valve_index is None:
            settled.extend((item.row, item.col) for item in active if item.kind == "cell")
            return settled

        prefix = active[:valve_index]
        valve = active[valve_index]
        suffix = active[valve_index + 1 :]
        assert valve.direction is not None
        dr, dc = SHIFT_BY_DIRECTION[valve.direction]

        if _prefix_stays(valve.direction):
            settled.extend((item.row, item.col) for item in prefix if item.kind == "cell")
            _shift_items(suffix, dr, dc)
        else:
            _shift_items(prefix, dr, dc)
            settled.extend((item.row, item.col) for item in prefix if item.kind == "cell")

        active = suffix


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut each straight colored pipe at its red valves and offset one side by two
    cells in the yellow-marked direction, processing valves from the start end.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = np.full((h, w), bg, dtype=int)

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (bg, RED, YELLOW)):
        _orientation, items = _parse_pipe(g, color)
        for r, c in _simulate_pipe_cells(items):
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = color

    return out


def _max_valve_count(length: int, gap: int = 1) -> int:
    span = length - 2
    best = 0
    for count in range(1, 4):
        if span >= count + (count - 1) * gap:
            best = count
    return best


def _sample_spaced_positions(count: int, lo: int, hi: int, gap: int, rng: random.Random) -> list[int] | None:
    span = hi - lo + 1
    need = count + (count - 1) * gap
    if span < need:
        return None
    compressed = span - (count - 1) * gap
    picks = sorted(rng.sample(range(compressed), count))
    return [lo + picks[i] + i * gap for i in range(count)]


def _pipe_items_from_spec(pipe: GenPipe) -> list[PipeItem]:
    items: list[PipeItem] = []
    valve_map = {valve.axis: valve.direction for valve in pipe.valves}
    if pipe.orientation == "h":
        row = pipe.fixed
        for axis in range(pipe.start, pipe.end + 1):
            if axis in valve_map:
                direction = valve_map[axis]
                marker_row = row - 1 if direction == "up" else row + 1
                items.append(
                    PipeItem(
                        axis=axis,
                        row=row,
                        col=axis,
                        kind="valve",
                        direction=direction,
                        marker_row=marker_row,
                        marker_col=axis,
                    )
                )
            else:
                items.append(PipeItem(axis=axis, row=row, col=axis, kind="cell"))
    else:
        col = pipe.fixed
        for axis in range(pipe.start, pipe.end + 1):
            if axis in valve_map:
                direction = valve_map[axis]
                marker_col = col - 1 if direction == "left" else col + 1
                items.append(
                    PipeItem(
                        axis=axis,
                        row=axis,
                        col=col,
                        kind="valve",
                        direction=direction,
                        marker_row=axis,
                        marker_col=marker_col,
                    )
                )
            else:
                items.append(PipeItem(axis=axis, row=axis, col=col, kind="cell"))
    return items


def _pipe_input_coords(pipe: GenPipe) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    valve_map = {valve.axis: valve.direction for valve in pipe.valves}
    if pipe.orientation == "h":
        row = pipe.fixed
        for axis in range(pipe.start, pipe.end + 1):
            coords.append((row, axis))
            direction = valve_map.get(axis)
            if direction is not None:
                coords.append((row - 1 if direction == "up" else row + 1, axis))
    else:
        col = pipe.fixed
        for axis in range(pipe.start, pipe.end + 1):
            coords.append((axis, col))
            direction = valve_map.get(axis)
            if direction is not None:
                coords.append((axis, col - 1 if direction == "left" else col + 1))
    return coords


def _pipe_output_coords(pipe: GenPipe, h: int, w: int) -> list[tuple[int, int]]:
    coords = []
    for r, c in _simulate_pipe_cells(_pipe_items_from_spec(pipe)):
        if 0 <= r < h and 0 <= c < w:
            coords.append((r, c))
    return coords


def _coords_are_free(blocked: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _reserve_coords(blocked: np.ndarray, coords: list[tuple[int, int]], pad: int) -> None:
    h, w = blocked.shape
    for r, c in coords:
        r0 = max(0, r - pad)
        c0 = max(0, c - pad)
        r1 = min(h, r + pad + 1)
        c1 = min(w, c + pad + 1)
        blocked[r0:r1, c0:c1] = True


def _render_input(grid: np.ndarray, pipe: GenPipe) -> None:
    valve_map = {valve.axis: valve.direction for valve in pipe.valves}
    if pipe.orientation == "h":
        row = pipe.fixed
        for axis in range(pipe.start, pipe.end + 1):
            direction = valve_map.get(axis)
            grid[row, axis] = RED if direction is not None else pipe.color
            if direction is not None:
                grid[row - 1 if direction == "up" else row + 1, axis] = YELLOW
    else:
        col = pipe.fixed
        for axis in range(pipe.start, pipe.end + 1):
            direction = valve_map.get(axis)
            grid[axis, col] = RED if direction is not None else pipe.color
            if direction is not None:
                grid[axis, col - 1 if direction == "left" else col + 1] = YELLOW


def _render_output(grid: np.ndarray, pipe: GenPipe) -> None:
    for r, c in _pipe_output_coords(pipe, *grid.shape):
        grid[r, c] = pipe.color


def _sample_pipe(color: int, h: int, w: int, need_multi: bool, rng: random.Random) -> GenPipe | None:
    orientation = rng.choice(["h", "v"])
    if orientation == "h":
        max_len = min(10, w - 2)
        if max_len < 6:
            return None
        length = rng.randint(6, max_len)
        max_valves = _max_valve_count(length)
        if max_valves == 0:
            return None
        if need_multi:
            if max_valves < 2:
                return None
            valve_count = rng.randint(2, max_valves)
        else:
            valve_count = rng.randint(1, min(2, max_valves))
        positions = _sample_spaced_positions(valve_count, 1, length - 2, gap=1, rng=rng)
        if positions is None:
            return None

        directions = [rng.choice(["up", "down"]) for _ in positions]

        start = rng.randint(1, w - length - 1)
        fixed = rng.randint(1, h - 2)
        valves = tuple(GenValve(axis=start + pos, direction=direction) for pos, direction in zip(positions, directions))
        return GenPipe(color=color, orientation=orientation, fixed=fixed, start=start, end=start + length - 1, valves=valves)

    max_len = min(10, h - 2)
    if max_len < 6:
        return None
    length = rng.randint(6, max_len)
    max_valves = _max_valve_count(length)
    if max_valves == 0:
        return None
    if need_multi:
        if max_valves < 2:
            return None
        valve_count = rng.randint(2, max_valves)
    else:
        valve_count = rng.randint(1, min(2, max_valves))
    positions = _sample_spaced_positions(valve_count, 1, length - 2, gap=1, rng=rng)
    if positions is None:
        return None

    directions = [rng.choice(["left", "right"]) for _ in positions]

    start = rng.randint(1, h - length - 1)
    fixed = rng.randint(1, w - 2)
    valves = tuple(GenValve(axis=start + pos, direction=direction) for pos, direction in zip(positions, directions))
    return GenPipe(color=color, orientation=orientation, fixed=fixed, start=start, end=start + length - 1, valves=valves)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 20)
    w = rng.randint(14, 20)
    num_pipes = rng.randint(2, 4)
    colors = rng.sample(PIPE_COLOR_POOL, num_pipes)

    input_blocked = np.zeros((h, w), dtype=bool)
    output_blocked = np.zeros((h, w), dtype=bool)
    pipes: list[GenPipe] = []
    multi_index = rng.randrange(num_pipes) if rng.random() < 0.7 else -1

    for i, color in enumerate(colors):
        placed = False
        for _ in range(300):
            pipe = _sample_pipe(color, h, w, need_multi=(i == multi_index), rng=rng)
            if pipe is None:
                continue
            input_coords = _pipe_input_coords(pipe)
            output_coords = _pipe_output_coords(pipe, h, w)
            if len(output_coords) < 3:
                continue
            if not _coords_are_free(input_blocked, input_coords):
                continue
            if not _coords_are_free(output_blocked, output_coords):
                continue
            _reserve_coords(input_blocked, input_coords, pad=1)
            _reserve_coords(output_blocked, output_coords, pad=0)
            pipes.append(pipe)
            placed = True
            break
        if not placed:
            return None

    if multi_index >= 0 and all(len(pipe.valves) < 2 for pipe in pipes):
        return None

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    for pipe in pipes:
        _render_input(grid, pipe)
        _render_output(expected, pipe)

    if not np.any(grid == RED) or not np.any(grid == YELLOW):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-valve input")
