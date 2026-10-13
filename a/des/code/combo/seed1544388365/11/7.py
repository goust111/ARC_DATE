from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {delta: idx for idx, delta in enumerate(DIRS)}

MAX_TRACE_MULTIPLIER = 4
MAX_GENERATION_ATTEMPTS = 300


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _neighbors(cell: tuple[int, int], passable: np.ndarray) -> list[tuple[int, int]]:
    h, w = passable.shape
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(passable[nr, nc]):
            out.append((nr, nc))
    return out


def _direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return b[0] - a[0], b[1] - a[1]


def _choose_next_dir(
    cell: tuple[int, int],
    travel_dir: tuple[int, int],
    passable: np.ndarray,
) -> tuple[int, int] | None:
    back = (-travel_dir[0], -travel_dir[1])
    choices: list[tuple[int, int]] = []
    for nxt in _neighbors(cell, passable):
        step = _direction(cell, nxt)
        if step != back:
            choices.append(step)

    if not choices:
        return None
    if len(choices) == 1:
        return choices[0]

    idx = DIR_TO_INDEX[travel_dir]
    priority = [
        DIRS[idx],
        DIRS[(idx + 1) % 4],
        DIRS[(idx - 1) % 4],
    ]
    for cand in priority:
        if cand in choices:
            return cand
    return choices[0]


def _trace_path(
    start: tuple[int, int],
    initial_dir: tuple[int, int],
    passable: np.ndarray,
) -> list[tuple[int, int]]:
    limit = int(passable.sum()) * MAX_TRACE_MULTIPLIER + 4
    path = [start]
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    cell = start
    travel_dir = initial_dir
    for _ in range(limit):
        state = (cell, travel_dir)
        if state in seen_states:
            break
        seen_states.add(state)

        next_dir = _choose_next_dir(cell, travel_dir, passable)
        if next_dir is None:
            break
        cell = _add(cell, next_dir)
        path.append(cell)
        travel_dir = next_dir
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored sources along gray rails, resolving intersections by
    straight/right/left priority and first-arrival ownership.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    passable = g != BLACK
    assert np.any(g == GRAY), "expected gray rails"
    assert np.any(passable), "expected a non-empty rail network"

    source_coords = np.argwhere((g != BLACK) & (g != GRAY))
    assert len(source_coords) > 0, "expected at least one source"

    source_records: list[dict[str, object]] = []
    seen_colors: set[int] = set()
    for r_raw, c_raw in source_coords:
        cell = (int(r_raw), int(c_raw))
        color = int(g[cell])
        assert color not in seen_colors, "source colors must be unique"
        seen_colors.add(color)

        nbrs = _neighbors(cell, passable)
        assert len(nbrs) == 1, "each source must sit at a rail endpoint"
        initial_dir = _direction(cell, nbrs[0])
        path = _trace_path(cell, initial_dir, passable)
        source_records.append(
            {
                "start": cell,
                "color": color,
                "path": path,
            }
        )

    source_records.sort(key=lambda item: tuple(item["start"]))  # deterministic tie-break only

    winners: dict[tuple[int, int], tuple[int, int]] = {}
    for priority, record in enumerate(source_records):
        for step, cell in enumerate(record["path"]):
            current = winners.get(cell)
            score = (step, priority)
            if current is None or score < current:
                winners[cell] = score

    out = g.copy()
    out[g == GRAY] = GRAY
    for priority, record in enumerate(source_records):
        color = int(record["color"])
        for step, cell in enumerate(record["path"]):
            if winners[cell] != (step, priority):
                break
            out[cell] = color
    return out


def _sample_spaced_positions(
    candidates: list[int],
    count: int,
    min_gap: int,
) -> list[int] | None:
    if count == 0:
        return []
    if len(candidates) < count:
        return None
    for _ in range(120):
        chosen = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _winner_distance(horizontal_len: int, vertical_len: int, row: int) -> int:
    return horizontal_len + (vertical_len - row)


def _build_canonical_pair() -> tuple[np.ndarray, np.ndarray] | None:
    horizontal_source_count = random.randint(1, 2)
    vertical_source_count = random.randint(0, 1)

    min_horizontal = max(5, 2 * horizontal_source_count + 3)
    horizontal_len = random.randint(min_horizontal, 8)
    vertical_len = random.randint(4, 8)

    horizontal_positions = list(range(1, horizontal_len - 1))
    horizontal_merges = _sample_spaced_positions(horizontal_positions, horizontal_source_count, 2)
    if horizontal_merges is None:
        return None

    vertical_merges: list[int] = []
    if vertical_source_count:
        row_candidates = list(range(max(1, vertical_len - 3), vertical_len))
        vertical_merges = _sample_spaced_positions(row_candidates, vertical_source_count, 2)
        if vertical_merges is None:
            return None

    remaining_h = [x for x in range(1, horizontal_len) if x not in horizontal_merges]
    if not remaining_h:
        return None
    horizontal_decor_count = random.randint(1, min(2, len(remaining_h)))
    horizontal_decor = sorted(random.sample(remaining_h, horizontal_decor_count))

    remaining_v = [y for y in range(1, vertical_len) if y not in vertical_merges]
    vertical_decor_count = 0 if not remaining_v else random.randint(0, min(2, len(remaining_v)))
    vertical_decor = sorted(random.sample(remaining_v, vertical_decor_count))

    down_lengths: dict[int, int] = {}
    for x in horizontal_merges:
        down_lengths[x] = random.randint(x + 1, x + 4)

    east_lengths: dict[int, int] = {}
    for y in vertical_merges:
        dist = _winner_distance(horizontal_len, vertical_len, y)
        east_lengths[y] = random.randint(dist + 1, dist + 4)

    north_decor: dict[int, int] = {}
    south_decor: dict[int, int] = {}
    for x in horizontal_decor:
        mode = random.choice(["north", "south", "both"])
        if mode in {"north", "both"}:
            north_decor[x] = random.randint(1, min(3, vertical_len))
        if mode in {"south", "both"}:
            south_decor[x] = random.randint(1, 3)

    east_decor: dict[int, int] = {}
    for y in vertical_decor:
        east_decor[y] = random.randint(1, 3)

    max_down = max([0, *down_lengths.values(), *south_decor.values()])
    max_east = max([0, *east_lengths.values(), *east_decor.values()])
    base_h = vertical_len + max_down + 1
    base_w = horizontal_len + max_east + 1

    inp = np.full((base_h, base_w), BLACK, dtype=int)
    out = np.full((base_h, base_w), BLACK, dtype=int)

    backbone: list[tuple[int, int]] = []
    for x in range(horizontal_len + 1):
        backbone.append((vertical_len, x))
    for y in range(vertical_len):
        backbone.append((y, horizontal_len))
    _paint_cells(inp, backbone, GRAY)
    _paint_cells(out, backbone, GRAY)

    for x, length in north_decor.items():
        cells = [(vertical_len - d, x) for d in range(1, length + 1)]
        _paint_cells(inp, cells, GRAY)
        _paint_cells(out, cells, GRAY)

    for x, length in south_decor.items():
        cells = [(vertical_len + d, x) for d in range(1, length + 1)]
        _paint_cells(inp, cells, GRAY)
        _paint_cells(out, cells, GRAY)

    for y, length in east_decor.items():
        cells = [(y, horizontal_len + d) for d in range(1, length + 1)]
        _paint_cells(inp, cells, GRAY)
        _paint_cells(out, cells, GRAY)

    source_count = 1 + len(horizontal_merges) + len(vertical_merges)
    colors = random.sample(PALETTE, source_count)
    winner_color = colors[0]
    next_color_idx = 1

    winner_source = (vertical_len, 0)
    inp[winner_source] = winner_color
    _paint_cells(out, backbone, winner_color)

    for x in horizontal_merges:
        length = down_lengths[x]
        branch = [(vertical_len + d, x) for d in range(length + 1)]
        color = colors[next_color_idx]
        next_color_idx += 1

        _paint_cells(inp, branch, GRAY)
        _paint_cells(out, branch[1:], GRAY)
        inp[branch[-1]] = color
        _paint_cells(out, branch[1:], color)

    for y in vertical_merges:
        length = east_lengths[y]
        branch = [(y, horizontal_len + d) for d in range(length + 1)]
        color = colors[next_color_idx]
        next_color_idx += 1

        _paint_cells(inp, branch, GRAY)
        _paint_cells(out, branch[1:], GRAY)
        inp[branch[-1]] = color
        _paint_cells(out, branch[1:], color)

    for r, c in np.argwhere(inp == winner_color):
        assert (int(r), int(c)) == winner_source
    assert np.any(inp == GRAY), "generator must leave gray rails"
    return inp, out


def _transform_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randrange(4)
    if k:
        inp = np.rot90(inp, k)
        out = np.rot90(out, k)
    if random.random() < 0.5:
        inp = np.fliplr(inp)
        out = np.fliplr(out)
    return np.asarray(inp, dtype=int).copy(), np.asarray(out, dtype=int).copy()


def _embed_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pad_top = random.randint(0, 3)
    pad_bottom = random.randint(0, 3)
    pad_left = random.randint(0, 3)
    pad_right = random.randint(0, 3)

    h, w = inp.shape
    canvas_h = h + pad_top + pad_bottom
    canvas_w = w + pad_left + pad_right
    canvas_in = np.full((canvas_h, canvas_w), BLACK, dtype=int)
    canvas_out = np.full((canvas_h, canvas_w), BLACK, dtype=int)
    canvas_in[pad_top : pad_top + h, pad_left : pad_left + w] = inp
    canvas_out[pad_top : pad_top + h, pad_left : pad_left + w] = out
    return canvas_in, canvas_out


def _gen_attempt() -> np.ndarray | None:
    pair = _build_canonical_pair()
    if pair is None:
        return None
    inp, expected = pair
    inp, expected = _transform_pair(inp, expected)
    inp, expected = _embed_pair(inp, expected)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rail-routing input")
