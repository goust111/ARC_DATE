from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]

BG = 0
TRIGGER = 5
PATH_COLORS = (1, 2, 3, 4, 7)

GRID_H_RANGE = (10, 17)
GRID_W_RANGE = (12, 19)
PATH_LEN_RANGE = (10, 20)
SEGMENT_COUNT_RANGE = (3, 7)  # at least two turns
SEGMENT_LEN_MAX = 6
MARGIN = 1

MAX_BUILD_ATTEMPTS = 500
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

_SEEN_INPUTS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()
_GENERATION_COUNT = 0


def neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _ordered_path(coords: np.ndarray) -> list[Coord]:
    cells = {tuple(map(int, coord)) for coord in coords}
    assert cells, "expected a non-empty path"

    degrees = {
        cell: sum((neighbor in cells) for neighbor in neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    path: list[Coord] = []
    prev = None
    cur = endpoints[0]
    while True:
        path.append(cur)
        next_cells = [
            neighbor
            for neighbor in neighbors4(*cur)
            if neighbor in cells and neighbor != prev
        ]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]

    assert len(path) == len(cells), "path traversal must visit every foreground pixel"
    return path


def _analyze_grid(grid: np.ndarray) -> tuple[list[Coord], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    coords = np.argwhere(g != BG)
    path = _ordered_path(coords)
    trigger_indices = [
        i for i, (r, c) in enumerate(path) if int(g[r, c]) == TRIGGER
    ]
    assert len(trigger_indices) == 1, "expected exactly one gray trigger"

    trigger_index = trigger_indices[0]
    assert 0 < trigger_index < len(path) - 1, "trigger must be strictly internal"

    left = path[:trigger_index]
    right = path[trigger_index + 1 :]
    assert left and right, "trigger must split the path into two non-empty segments"
    assert len(left) != len(right), "segment lengths must differ"
    return path, trigger_index


def _apply_shorter_segment_rule(grid: np.ndarray, path: list[Coord], trigger_index: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    left = path[:trigger_index]
    right = path[trigger_index + 1 :]
    target = left if len(left) < len(right) else right
    target_colors = [int(out[r, c]) for r, c in target]
    for (r, c), color in zip(target, reversed(target_colors)):
        out[r, c] = color
    return out


def _reverse_path_colors(grid: np.ndarray, path: list[Coord]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    colors = [int(out[r, c]) for r, c in path]
    for (r, c), color in zip(path, reversed(colors)):
        out[r, c] = color
    return out


def _reverse_segment(grid: np.ndarray, segment: list[Coord]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    colors = [int(out[r, c]) for r, c in segment]
    for (r, c), color in zip(segment, reversed(colors)):
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse only the color sequence on the shorter of the two path segments split by the unique gray trigger.

    Input structure:
    - The background is black.
    - All non-black pixels form one 4-connected one-pixel-wide path with exactly two endpoints and no branches.
    - Exactly one internal gray pixel lies on that path and acts as the trigger.

    Rule:
    - Traverse the full path from one endpoint to the other.
    - Use the unique gray pixel as the split point; it belongs to neither side segment.
    - Count the number of path cells on each side of the trigger.
    - Reverse the color sequence in place only on the shorter side segment.
    - Keep the longer side unchanged, and keep the gray trigger unchanged.

    Key insight:
    - The gray trigger does not recolor anything by itself; it uniquely decides where the path is cut, and the segment-length comparison decides which local color subsequence must be reversed.
    """
    g = np.asarray(grid, dtype=int)
    path, trigger_index = _analyze_grid(g)
    return _apply_shorter_segment_rule(g, path, trigger_index)


def _turn_dirs(prev_dir: Coord | None) -> list[Coord]:
    if prev_dir is None:
        dirs = list(DIRS)
    else:
        dr, dc = prev_dir
        dirs = [(-dc, dr), (dc, -dr)]
    random.shuffle(dirs)
    return dirs


def _sample_run_lengths(total_steps: int) -> list[int]:
    valid_counts = [
        segments
        for segments in range(SEGMENT_COUNT_RANGE[0], SEGMENT_COUNT_RANGE[1] + 1)
        if segments <= total_steps <= segments * SEGMENT_LEN_MAX
    ]
    segments = random.choice(valid_counts)
    remaining = total_steps
    runs: list[int] = []
    for i in range(segments):
        slots_left = segments - i - 1
        lo = max(1, remaining - SEGMENT_LEN_MAX * slots_left)
        hi = min(SEGMENT_LEN_MAX, remaining - slots_left)
        runs.append(random.randint(lo, hi))
        remaining -= runs[-1]
    random.shuffle(runs)
    return runs


def _extend_segment(
    path: list[Coord],
    used: set[Coord],
    direction: Coord,
    steps: int,
    shape: tuple[int, int],
) -> tuple[list[Coord], set[Coord]] | None:
    h, w = map(int, shape)
    dr, dc = direction
    r, c = path[-1]
    new_path = list(path)
    new_used = set(used)

    for _ in range(steps):
        nr, nc = r + dr, c + dc
        if not (MARGIN <= nr < h - MARGIN and MARGIN <= nc < w - MARGIN):
            return None
        if (nr, nc) in new_used:
            return None
        if any((ar, ac) in new_used and (ar, ac) != (r, c) for ar, ac in neighbors4(nr, nc)):
            return None

        new_path.append((nr, nc))
        new_used.add((nr, nc))
        r, c = nr, nc
    return new_path, new_used


def _build_path(shape: tuple[int, int], target_len: int) -> list[Coord] | None:
    run_lengths = _sample_run_lengths(target_len - 1)
    h, w = map(int, shape)
    starts = [
        (r, c)
        for r in range(MARGIN, h - MARGIN)
        for c in range(MARGIN, w - MARGIN)
    ]
    random.shuffle(starts)

    def search(
        seg_i: int,
        prev_dir: Coord | None,
        path: list[Coord],
        used: set[Coord],
    ) -> list[Coord] | None:
        if seg_i == len(run_lengths):
            return path
        for direction in _turn_dirs(prev_dir):
            extended = _extend_segment(path, used, direction, run_lengths[seg_i], shape)
            if extended is None:
                continue
            found = search(seg_i + 1, direction, extended[0], extended[1])
            if found is not None:
                return found
        return None

    for start in starts:
        found = search(0, None, [start], {start})
        if found is not None:
            return found
    return None


def _choose_trigger_index(path_len: int, prefer_left_shorter: bool) -> int | None:
    left_shorter = [
        i
        for i in range(2, path_len - 2)
        if i < path_len - i - 1 and path_len - i - 1 >= 2
    ]
    right_shorter = [
        i
        for i in range(2, path_len - 2)
        if i > path_len - i - 1 and path_len - i - 1 >= 2
    ]
    primary = left_shorter if prefer_left_shorter else right_shorter
    secondary = right_shorter if prefer_left_shorter else left_shorter
    if primary:
        return random.choice(primary)
    if secondary:
        return random.choice(secondary)
    return None


def _sample_path_colors(path_len: int, trigger_index: int) -> list[int] | None:
    palette = random.sample(PATH_COLORS, random.randint(3, len(PATH_COLORS)))
    colors = [random.choice(palette) for _ in range(path_len)]
    colors[trigger_index] = TRIGGER

    shorter_start = 0
    shorter_end = trigger_index
    if trigger_index > path_len - trigger_index - 1:
        shorter_start = trigger_index + 1
        shorter_end = path_len

    shorter_seq = colors[shorter_start:shorter_end]
    if shorter_seq == shorter_seq[::-1]:
        return None
    if len(set(colors) - {TRIGGER}) < 3:
        return None
    return colors


def _paint_path(shape: tuple[int, int], path: list[Coord], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for (r, c), color in zip(path, colors):
        grid[r, c] = color
    return grid


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    g = np.asarray(grid, dtype=int)
    return tuple(map(int, g.shape)), g.tobytes()


def _semantic_checks(inp: np.ndarray) -> np.ndarray | None:
    path, trigger_index = _analyze_grid(inp)
    left = path[:trigger_index]
    right = path[trigger_index + 1 :]
    shorter = left if len(left) < len(right) else right
    longer = right if shorter is left else left

    out = main(inp)
    expected = _apply_shorter_segment_rule(inp, path, trigger_index)
    if not np.array_equal(out, expected):
        return None
    if np.array_equal(out, inp):
        return None

    trigger_cell = path[trigger_index]
    if int(out[trigger_cell]) != TRIGGER:
        return None
    if any(int(out[r, c]) != int(inp[r, c]) for r, c in longer):
        return None

    shorter_in = [int(inp[r, c]) for r, c in shorter]
    shorter_out = [int(out[r, c]) for r, c in shorter]
    if shorter_out != shorter_in[::-1]:
        return None

    if np.array_equal(out, _reverse_path_colors(inp, path)):
        return None
    if np.array_equal(out, _reverse_segment(inp, longer)):
        return None
    if np.array_equal(out, _reverse_segment(inp, left)) and shorter is not left:
        return None
    if np.array_equal(out, _reverse_segment(inp, right)) and shorter is not right:
        return None

    return out


def _gen_attempt(prefer_left_shorter: bool) -> tuple[np.ndarray, np.ndarray] | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    target_len = random.randint(*PATH_LEN_RANGE)
    path = _build_path(shape, target_len)
    if path is None:
        return None

    canonical_path = _ordered_path(np.array(path, dtype=int))
    trigger_index = _choose_trigger_index(len(canonical_path), prefer_left_shorter)
    if trigger_index is None:
        return None

    colors = _sample_path_colors(len(canonical_path), trigger_index)
    if colors is None:
        return None

    inp = _paint_path(shape, canonical_path, colors)
    out = _semantic_checks(inp)
    if out is None:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    global _GENERATION_COUNT

    for _ in range(MAX_BUILD_ATTEMPTS):
        prefer_left_shorter = (_GENERATION_COUNT % 2) == 0
        generated = _gen_attempt(prefer_left_shorter)
        if generated is None:
            continue

        inp, out = generated
        inp_sig = _signature(inp)
        out_sig = _signature(out)
        if inp_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
            continue

        _SEEN_INPUTS.add(inp_sig)
        _SEEN_OUTPUTS.add(out_sig)
        _GENERATION_COUNT += 1
        return inp
    raise RuntimeError("failed to generate a valid local-reversal ARC input")
