import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5


def _ring_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    coords: list[tuple[int, int]] = []
    for c in range(w):
        coords.append((0, c))
    for r in range(1, h - 1):
        coords.append((r, w - 1))
    for c in range(w - 1, -1, -1):
        coords.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _find_red_run(g: np.ndarray) -> tuple[int, int]:
    ring = _ring_coords(g.shape)
    vals = [int(g[r, c]) for r, c in ring]
    starts = [i for i, v in enumerate(vals) if v == RED and vals[i - 1] != RED]
    assert len(starts) == 1, "expected one contiguous red border segment"
    start = starts[0]
    length = 0
    while vals[(start + length) % len(vals)] == RED:
        length += 1
    assert 2 <= length <= 4, "red border segment length must define L"
    return start, length


def _reference_height(inner: np.ndarray) -> int:
    h = 0
    for r in range(inner.shape[0]):
        if np.all(inner[r] != BLACK):
            h += 1
        else:
            break
    assert 1 <= h <= 2, "reference row group must be 1-2 rows tall"
    return h


def _group_prefix_width(block: np.ndarray) -> int:
    nz = np.argwhere(block != BLACK)
    assert nz.size > 0, "incomplete group must contain a non-empty prefix"
    return int(nz[:, 1].max()) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the border's cyclic red segments and use their length to extend each inner prefix group by that many green columns.

    Input structure:
    - The outer border is gray with one contiguous red segment of length L on a single edge.
    - Inside the border, the top row group is a full-width blue/yellow reference pattern of height 1 or 2.
    - Lower row groups have the same height, are separated by black rows, and show only a left prefix of the reference pattern.

    Rule:
    - Read the red border segment length as L and traverse the border clockwise.
    - Starting at that segment, repeat a border cycle of L red cells followed by L gray cells around the whole ring.
    - Keep the reference group unchanged, and for every lower partial group append exactly L green columns immediately after its visible prefix.

    Key insight:
    - The border segment length is both a geometric border period and a hidden numeric instruction that controls how far each inner row group continues.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5

    out = g.copy()
    ring = _ring_coords(g.shape)
    start, period_len = _find_red_run(g)
    cycle = 2 * period_len
    for i, (r, c) in enumerate(ring):
        if ((i - start) % cycle) < period_len:
            out[r, c] = RED
        else:
            out[r, c] = GREY

    inner = g[1:-1, 1:-1]
    inner_out = out[1:-1, 1:-1]
    ref_h = _reference_height(inner)
    ref = inner[:ref_h].copy()
    assert np.all(ref != BLACK), "reference group must span the full inner width"

    r = ref_h
    while r < inner.shape[0]:
        if np.all(inner[r] == BLACK):
            r += 1
            continue
        assert r + ref_h <= inner.shape[0], "partial group truncated"
        block = inner[r : r + ref_h]
        prefix_w = _group_prefix_width(block)
        assert prefix_w + period_len <= inner.shape[1], "green continuation must stay in bounds"
        assert np.array_equal(block[:, :prefix_w], ref[:, :prefix_w]), "partial group must match a reference prefix"
        assert np.all(block[:, prefix_w:] == BLACK), "partial group suffix must be blank"
        inner_out[r : r + ref_h, prefix_w : prefix_w + period_len] = GREEN
        r += ref_h

    return out


def _side_segment_cells(h: int, w: int, length: int) -> list[list[tuple[int, int]]]:
    options: list[list[tuple[int, int]]] = []
    for c in range(1, w - length):
        options.append([(0, c + k) for k in range(length)])
    for r in range(1, h - length):
        options.append([(r + k, w - 1) for k in range(length)])
    for c in range(w - 2, length - 1, -1):
        options.append([(h - 1, c - k) for k in range(length)])
    for r in range(h - 2, length - 1, -1):
        options.append([(r - k, 0) for k in range(length)])
    return options


def _random_reference(ref_h: int, inner_w: int) -> tuple[np.ndarray, int]:
    period = random.randint(2, min(5, inner_w))
    while True:
        states = [
            tuple(random.choice((BLUE, YELLOW)) for _ in range(ref_h))
            for _ in range(period)
        ]
        ref = np.zeros((ref_h, inner_w), dtype=int)
        for c in range(inner_w):
            state = states[c % period]
            for r in range(ref_h):
                ref[r, c] = state[r]
        if len(set(states)) >= 2 and all(len(set(ref[r])) >= 2 for r in range(ref_h)):
            return ref, period


def _choose_prefixes(group_count: int, inner_w: int, length: int, period: int) -> list[int]:
    choices = list(range(1, inner_w - length + 1))
    assert len(choices) >= group_count
    prefixes: list[int] = []
    wrap_like = [p for p in choices if (p % period) + length > period]
    if wrap_like:
        prefixes.append(random.choice(wrap_like))
    remaining = [p for p in choices if p not in prefixes]
    random.shuffle(remaining)
    prefixes.extend(remaining[: group_count - len(prefixes)])
    random.shuffle(prefixes)
    return prefixes


def _build_input() -> np.ndarray:
    group_count = random.randint(2, 3)
    ref_h = random.randint(1, 2)
    min_inner_h = ref_h * (group_count + 1) + group_count
    inner_h = random.randint(min_inner_h, min_inner_h + 3)
    inner_w = random.randint(8, 16)
    h = inner_h + 2
    w = inner_w + 2

    length = random.randint(2, min(4, inner_w - 2))
    ref, period = _random_reference(ref_h, inner_w)
    prefixes = _choose_prefixes(group_count, inner_w, length, period)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    for r, c in random.choice(_side_segment_cells(h, w, length)):
        grid[r, c] = RED

    inner = grid[1:-1, 1:-1]
    inner[:ref_h] = ref
    r = ref_h + 1
    for i, prefix_w in enumerate(prefixes):
        inner[r : r + ref_h, :prefix_w] = ref[:, :prefix_w]
        r += ref_h
        if i != len(prefixes) - 1:
            r += 1

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_input()
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input")
