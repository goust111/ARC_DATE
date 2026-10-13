import itertools
import random

import numpy as np


SEGMENT_COLORS = (1, 2, 3, 4)
START_COLOR = 7
BACKGROUND = 0
GRID_SIZES = (11, 12, 13)
MAX_TRACE_STEPS = 20
HIT_SELECTION_STEPS = 15
SEGMENT_COUNT_OPTIONS = (3, 4)
SEGMENT_LENGTH_RANGE = (3, 7)
GENERATION_ATTEMPTS = 500


def _simulate_ray(shape: tuple[int, int], start: tuple[int, int], limit: int = MAX_TRACE_STEPS) -> list[tuple[int, int]]:
    """Return the visited cells after leaving start, stopping on return or after `limit` moves."""
    h, w = shape
    r, c = map(int, start)
    dr, dc = -1, 1
    path: list[tuple[int, int]] = []

    for _ in range(limit):
        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= h:
            dr = -dr
            nr = r + dr
        if nc < 0 or nc >= w:
            dc = -dc
            nc = c + dc
        r, c = nr, nc
        path.append((int(r), int(c)))
        if (r, c) == start:
            break
    return path


def _crossing_sequence(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    starts = np.argwhere(g == START_COLOR)
    assert starts.shape[0] == 1, "expected exactly one orange start pixel"
    start = tuple(map(int, starts[0]))

    sequence: list[int] = []
    for r, c in _simulate_ray(g.shape, start):
        color = int(g[r, c])
        if color not in (BACKGROUND, START_COLOR):
            if not sequence or sequence[-1] != color:
                sequence.append(color)
    return sequence


def _build_nested_output(sequence: list[int]) -> np.ndarray:
    layers = len(sequence)
    size = 2 * layers + 1
    out = np.full((size, size), START_COLOR, dtype=int)
    for depth, color in enumerate(sequence):
        out[depth : size - depth, depth : size - depth] = int(color)
    out[layers, layers] = START_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate the bouncing diagonal ray and convert its wall-crossing color order into concentric square frames.

    Input structure:
    - The grid has a black background with one orange start pixel at the lower-left corner.
    - There are 3-4 distinct colored wall segments, each a straight one-cell-thick horizontal or vertical line.
    - Wall segments do not intersect and the ray path only uses boundary reflections.

    Rule:
    - Start from the orange pixel and move one cell at a time along the up-right diagonal.
    - Reflect the horizontal direction at the left/right edges and the vertical direction at the top/bottom edges.
    - Whenever the ray steps onto a colored wall cell, record that wall color in encounter order; stop if the ray returns to the start or after 20 moves.
    - Build a tight odd-sized output of concentric square layers whose outer-to-inner colors follow the recorded sequence, with the final 1x1 center colored orange.

    Key insight:
    - The walls themselves are only markers; the real information is the temporal order in which the bouncing ray visits them.
    """
    sequence = _crossing_sequence(np.asarray(grid, dtype=int))
    return _build_nested_output(sequence)


def _segment_cells(hit: tuple[int, int], orientation: str, lo: int, hi: int) -> list[tuple[int, int]]:
    r, c = hit
    if orientation == "h":
        return [(int(r), cc) for cc in range(int(lo), int(hi) + 1)]
    return [(rr, int(c)) for rr in range(int(lo), int(hi) + 1)]


def _enumerate_placements(
    h: int,
    w: int,
    path: list[tuple[int, int]],
    hit_index: int,
) -> list[tuple[str, int, int]]:
    hit = path[hit_index]
    r, c = hit
    visited = set(path)
    placements: list[tuple[str, int, int]] = []

    row_blockers = sorted(cc for rr, cc in visited if rr == r and cc != c)
    left = max([cc for cc in row_blockers if cc < c], default=-1)
    right = min([cc for cc in row_blockers if cc > c], default=w)
    row_lo = max(left + 1, 1)
    row_hi = min(right - 1, w - 2)
    max_len = min(SEGMENT_LENGTH_RANGE[1], row_hi - row_lo + 1)
    for length in range(SEGMENT_LENGTH_RANGE[0], max_len + 1):
        start_lo = max(row_lo, c - length + 1)
        start_hi = min(c, row_hi - length + 1)
        for start in range(start_lo, start_hi + 1):
            placements.append(("h", int(start), int(start + length - 1)))

    col_blockers = sorted(rr for rr, cc in visited if cc == c and rr != r)
    up = max([rr for rr in col_blockers if rr < r], default=-1)
    down = min([rr for rr in col_blockers if rr > r], default=h)
    col_lo = max(up + 1, 1)
    col_hi = min(down - 1, h - 2)
    max_len = min(SEGMENT_LENGTH_RANGE[1], col_hi - col_lo + 1)
    for length in range(SEGMENT_LENGTH_RANGE[0], max_len + 1):
        start_lo = max(col_lo, r - length + 1)
        start_hi = min(r, col_hi - length + 1)
        for start in range(start_lo, start_hi + 1):
            placements.append(("v", int(start), int(start + length - 1)))

    random.shuffle(placements)
    return placements


def _touches_existing(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return True
    return False


def _choose_hit_indices(h: int, w: int, path: list[tuple[int, int]], n_segments: int) -> tuple[int, ...] | None:
    candidate_ids = []
    for idx in range(min(HIT_SELECTION_STEPS, len(path))):
        if _enumerate_placements(h, w, path, idx):
            candidate_ids.append(idx)

    valid = [
        combo
        for combo in itertools.combinations(candidate_ids, n_segments)
        if all(combo[i + 1] - combo[i] >= 2 for i in range(len(combo) - 1))
    ]
    return random.choice(valid) if valid else None


def _place_segments(
    h: int,
    w: int,
    path: list[tuple[int, int]],
    hit_indices: tuple[int, ...],
) -> list[tuple[tuple[int, int], str, int, int]] | None:
    occupied: set[tuple[int, int]] = {(h - 1, 0)}
    placements_by_hit = {idx: _enumerate_placements(h, w, path, idx) for idx in hit_indices}
    ordered_hits = sorted(hit_indices, key=lambda idx: len(placements_by_hit[idx]))
    chosen: dict[int, tuple[tuple[int, int], str, int, int]] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(ordered_hits):
            return True

        hit_idx = ordered_hits[pos]
        hit = path[hit_idx]
        for orientation, lo, hi in placements_by_hit[hit_idx]:
            cells = _segment_cells(hit, orientation, lo, hi)
            if _touches_existing(cells, occupied):
                continue
            for cell in cells:
                occupied.add(cell)
            chosen[hit_idx] = (hit, orientation, lo, hi)
            if backtrack(pos + 1):
                return True
            del chosen[hit_idx]
            for cell in cells:
                occupied.remove(cell)
        return False

    if not backtrack(0):
        return None
    return [chosen[idx] for idx in hit_indices]


def _render_input(
    h: int,
    w: int,
    segments: list[tuple[tuple[int, int], str, int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[h - 1, 0] = START_COLOR
    for color, (hit, orientation, lo, hi) in zip(colors, segments):
        for r, c in _segment_cells(hit, orientation, lo, hi):
            grid[r, c] = int(color)
    return grid


def _generation_attempt() -> np.ndarray | None:
    h = random.choice(GRID_SIZES)
    w = random.choice([size for size in GRID_SIZES if size != h])
    n_segments = random.choice(SEGMENT_COUNT_OPTIONS)
    colors = random.sample(list(SEGMENT_COLORS), n_segments)

    path = _simulate_ray((h, w), (h - 1, 0))
    if len(set(path)) != len(path):
        return None

    hit_indices = _choose_hit_indices(h, w, path, n_segments)
    if hit_indices is None:
        return None

    segments = _place_segments(h, w, path, hit_indices)
    if segments is None:
        return None

    grid = _render_input(h, w, segments, colors)
    sequence = _crossing_sequence(grid)
    expected = _build_nested_output(colors)

    if sequence != colors:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
