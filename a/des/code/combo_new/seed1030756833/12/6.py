from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
ORANGE = 7
SEED_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 300


def _separator_rows(g: np.ndarray) -> list[int]:
    return [int(r) for r in range(g.shape[0]) if bool(np.all(g[r, :] == ORANGE))]


def _separator_cols(g: np.ndarray) -> list[int]:
    return [int(c) for c in range(g.shape[1]) if bool(np.all(g[:, c] == ORANGE))]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for sep in sorted(separators):
        assert 0 <= sep < length, "separator index out of bounds"
        start, end = prev + 1, sep - 1
        if start <= end:
            spans.append((start, end))
        prev = sep
    start, end = prev + 1, length - 1
    if start <= end:
        spans.append((start, end))
    return spans


def _room_spans(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sep_rows = _separator_rows(g)
    sep_cols = _separator_cols(g)
    assert sep_rows and sep_cols, "expected complete orange row and column separators"

    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    assert row_spans and col_spans, "separators must leave nonempty rooms"
    return row_spans, col_spans


def _cross_cells(
    bounds: tuple[int, int, int, int],
    seed: tuple[int, int],
    radius: int,
) -> set[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    sr, sc = seed
    cells = {(int(sr), int(sc))}
    for dr, dc in DIR4:
        for dist in range(1, int(radius) + 1):
            rr = int(sr + dr * dist)
            cc = int(sc + dc * dist)
            if rr < r0 or rr > r1 or cc < c0 or cc > c1:
                break
            cells.add((rr, cc))
    return cells


def _paint_cross(
    out: np.ndarray,
    bounds: tuple[int, int, int, int],
    seed: tuple[int, int],
    radius: int,
    color: int,
) -> None:
    for r, c in _cross_cells(bounds, seed, radius):
        out[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into a bounded cross whose radius is counted by grey dots in its room.

    Input structure:
    - The background is black, and complete one-pixel orange rows and columns
      split the grid into rectangular rooms.
    - Each room contains zero or one colored seed pixel; seed colors are not
      black, orange, or grey.
    - Grey singleton pixels inside a room are count markers for that room.

    Rule:
    - For every room with a seed, count its grey pixels N.
    - Clear all grey pixels to black.
    - Paint the seed color from the seed outward N cells in the four cardinal
      directions, stopping each arm at the room boundary while other arms continue.
    - Orange separator lines remain unchanged, and rooms without a seed become black.

    Key insight:
    - The grey pixels are not obstacles or decoration; they encode the cross
      growth radius for the seed in the same rectangular room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_spans, col_spans = _room_spans(g)
    out = g.copy()
    out[g == GRAY] = BLACK

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            seed_mask = (region != BLACK) & (region != GRAY) & (region != ORANGE)
            seeds = np.argwhere(seed_mask)
            if len(seeds) == 0:
                continue

            assert len(seeds) == 1, "each room may contain at most one seed"
            local_r, local_c = map(int, seeds[0])
            sr, sc = int(r0 + local_r), int(c0 + local_c)
            color = int(g[sr, sc])
            radius = int(np.count_nonzero(region == GRAY))
            _paint_cross(out, (int(r0), int(r1), int(c0), int(c1)), (sr, sc), radius, color)

    return out


# --------------- generator ---------------


def _line_positions(room_sizes: list[int], bordered: bool) -> tuple[int, list[int]]:
    if bordered:
        length = int(sum(room_sizes) + len(room_sizes) + 1)
        positions = [0]
        pos = 1
        for size in room_sizes:
            pos += int(size)
            positions.append(int(pos))
            pos += 1
        return length, positions

    length = int(sum(room_sizes) + len(room_sizes) - 1)
    positions = []
    pos = 0
    for size in room_sizes[:-1]:
        pos += int(size)
        positions.append(int(pos))
        pos += 1
    return length, positions


def _make_separator_grid(
    room_heights: list[int],
    room_widths: list[int],
    bordered: bool,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]] | None:
    height, sep_rows = _line_positions(room_heights, bordered)
    width, sep_cols = _line_positions(room_widths, bordered)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for r in sep_rows:
        grid[int(r), :] = ORANGE
    for c in sep_cols:
        grid[:, int(c)] = ORANGE

    row_spans = _spans(height, sep_rows)
    col_spans = _spans(width, sep_cols)
    if len(row_spans) != len(room_heights) or len(col_spans) != len(room_widths):
        return None
    return grid, row_spans, col_spans


def _room_cells(bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _valid_seed_choices(
    bounds: tuple[int, int, int, int],
    branch: str,
) -> list[tuple[tuple[int, int], int]]:
    r0, r1, c0, c1 = bounds
    all_cells = _room_cells(bounds)
    choices: list[tuple[tuple[int, int], int]] = []

    if branch == "zero":
        return [(random.choice(all_cells), 0)]

    for seed in all_cells:
        sr, sc = seed
        min_dist = min(sr - r0, r1 - sr, sc - c0, c1 - sc)
        for radius in range(1, 6):
            cross = _cross_cells(bounds, seed, radius)
            if len(all_cells) - len(cross) < radius:
                continue
            truncated = radius > min_dist
            if branch == "truncated" and not truncated:
                continue
            if branch == "untruncated" and truncated:
                continue
            choices.append((seed, radius))

    if branch == "truncated":
        larger = [choice for choice in choices if choice[1] >= 2]
        if larger:
            choices = larger
    if branch == "untruncated":
        larger = [choice for choice in choices if choice[1] >= 2]
        if larger and random.random() < 0.65:
            choices = larger
    return choices


def _place_seed_room(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
    color: int,
    branch: str,
) -> dict | None:
    choices = _valid_seed_choices(bounds, branch)
    if not choices:
        return None

    seed, radius = random.choice(choices)
    cross = _cross_cells(bounds, seed, radius)
    candidates = [cell for cell in _room_cells(bounds) if cell not in cross]
    if len(candidates) < radius:
        return None

    grid[seed] = int(color)
    for r, c in random.sample(candidates, int(radius)):
        grid[int(r), int(c)] = GRAY

    return {
        "bounds": bounds,
        "seed": seed,
        "radius": int(radius),
        "color": int(color),
    }


def _place_gray_only_room(grid: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    cells = _room_cells(bounds)
    count = random.randint(1, min(3, len(cells)))
    for r, c in random.sample(cells, count):
        grid[int(r), int(c)] = GRAY


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    out[grid == GRAY] = BLACK
    for spec in specs:
        _paint_cross(
            out,
            spec["bounds"],
            spec["seed"],
            spec["radius"],
            spec["color"],
        )
    return out


def _branch_sequence(count: int) -> list[str]:
    branches = ["truncated", "untruncated"]
    if count >= 3 and random.random() < 0.55:
        branches.append("zero")
    while len(branches) < count:
        branches.append(random.choice(["truncated", "untruncated", "random"]))
    random.shuffle(branches)
    return branches


def _gen_attempt() -> np.ndarray | None:
    bordered = random.random() < 0.5
    room_rows = random.randint(2, 4)
    room_cols = random.randint(2, 4)
    room_heights = [random.randint(4, 7) for _ in range(room_rows)]
    room_widths = [random.randint(4, 7) for _ in range(room_cols)]

    made = _make_separator_grid(room_heights, room_widths, bordered)
    if made is None:
        return None
    grid, row_spans, col_spans = made

    rooms = [
        (r0, r1, c0, c1)
        for r0, r1 in row_spans
        for c0, c1 in col_spans
    ]
    if len(rooms) < 4:
        return None

    seed_count = random.randint(2, min(6, len(rooms) - 1, len(SEED_COLORS)))
    seed_rooms = random.sample(range(len(rooms)), seed_count)
    colors = random.sample(list(SEED_COLORS), seed_count)
    branches = _branch_sequence(seed_count)

    specs: list[dict] = []
    for idx, color, branch in zip(seed_rooms, colors, branches):
        spec = _place_seed_room(grid, rooms[idx], int(color), branch)
        if spec is None:
            return None
        specs.append(spec)

    empty_rooms = [i for i in range(len(rooms)) if i not in seed_rooms]
    if empty_rooms and random.random() < 0.45:
        for idx in random.sample(empty_rooms, random.randint(1, min(2, len(empty_rooms)))):
            _place_gray_only_room(grid, rooms[idx])

    expected = _expected_from_specs(grid, specs)
    if np.array_equal(expected, grid):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1030756833/12/6 input")
