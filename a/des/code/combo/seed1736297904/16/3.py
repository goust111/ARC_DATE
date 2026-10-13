import random
from typing import Iterable

import numpy as np


BLACK = 0
MAROON = 9
MAX_GENERATION_ATTEMPTS = 200


PATTERN_LIBRARY = (
    np.array([[1, 0], [0, 1]], dtype=int),
    np.array([[1, 1], [0, 0]], dtype=int),
    np.array([[1, 0], [1, 0]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 0], [0, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
)


def _full_separator_indices(lines: np.ndarray) -> list[int]:
    return [int(i) for i, is_sep in enumerate(lines) if bool(is_sep)]


def _room_spans(size: int, separators: Iterable[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for idx in list(separators) + [size]:
        if idx - prev > 1:
            spans.append((prev + 1, idx - 1))
        prev = idx
    return spans


def _tile_to_shape(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    reps_r = (shape[0] + tile.shape[0] - 1) // tile.shape[0]
    reps_c = (shape[1] + tile.shape[1] - 1) // tile.shape[1]
    return np.tile(tile, (reps_r, reps_c))[: shape[0], : shape[1]]


def _minimal_period_tile(room: np.ndarray) -> np.ndarray:
    h, w = room.shape
    best: np.ndarray | None = None
    best_area = None
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            area = ph * pw
            if best_area is not None and area > best_area:
                continue
            tile = room[:ph, :pw]
            if np.array_equal(_tile_to_shape(tile, room.shape), room):
                best = tile.copy()
                best_area = area
    assert best is not None, "legend room must contain a repeatable texture"
    return best


def _render_binary_pattern(mask: np.ndarray, primary: int, secondary: int, shape: tuple[int, int]) -> np.ndarray:
    colored_tile = np.where(mask == 1, primary, secondary)
    return _tile_to_shape(colored_tile, shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each non-legend room with the tiled texture selected by its seed color.

    Input structure
    - Maroon full-row/full-column separators split the grid into rectangular rooms.
    - Rooms in the leftmost room column form the legend: each is fully textured with
      a two-color repeating pattern whose top-left pixel is the pattern index color.
    - Every other room is black except for one non-black, non-maroon seed pixel.

    Rule
    - Read the texture dictionary from the leftmost room column.
    - Match each seed color to the legend room whose top-left pixel has that color.
    - Tile the matched texture over that room from the room-local top-left origin.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    sep_rows = _full_separator_indices(np.all(g == MAROON, axis=1))
    sep_cols = _full_separator_indices(np.all(g == MAROON, axis=0))
    row_spans = _room_spans(g.shape[0], sep_rows)
    col_spans = _room_spans(g.shape[1], sep_cols)
    assert row_spans and len(col_spans) >= 2, "expected a legend column plus at least one target column"

    legend_c0, legend_c1 = col_spans[0]
    patterns: dict[int, np.ndarray] = {}

    for r0, r1 in row_spans:
        room = g[r0 : r1 + 1, legend_c0 : legend_c1 + 1]
        assert room.shape[0] >= 3 and room.shape[1] >= 3
        assert not np.any(room == MAROON), "rooms must exclude separator pixels"
        colors = [int(color) for color in np.unique(room)]
        assert len(colors) == 2, "legend rooms must be two-color textures"
        index_color = int(room[0, 0])
        assert index_color != BLACK and index_color != MAROON, "legend index color must be visible"
        assert index_color not in patterns, "legend index colors must be unique"
        patterns[index_color] = _minimal_period_tile(room)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans[1:]:
            room = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(room == MAROON), "rooms must exclude separator pixels"
            seed_positions = np.argwhere((room != BLACK) & (room != MAROON))
            assert len(seed_positions) == 1, "non-legend rooms must contain exactly one seed"
            seed_r, seed_c = map(int, seed_positions[0])
            seed_color = int(room[seed_r, seed_c])
            assert set(map(int, np.unique(room))).issubset({BLACK, seed_color}), "target rooms must be black plus the seed"
            assert seed_color in patterns, "seed color must reference a legend texture"
            out[r0 : r1 + 1, c0 : c1 + 1] = _tile_to_shape(patterns[seed_color], room.shape)

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray:
    room_rows = rng.randint(2, 4)
    room_cols = rng.randint(2, 4)

    row_heights = [rng.randint(3, 6) for _ in range(room_rows)]
    col_widths = [rng.randint(3, 6) for _ in range(room_cols)]
    if len(set(row_heights)) == 1 and len(set(col_widths)) == 1:
        row_heights[rng.randrange(room_rows)] += 1

    palette = rng.sample(list(range(1, 9)), 2 * room_rows)
    index_colors = palette[:room_rows]
    secondary_colors = palette[room_rows:]
    pattern_masks = rng.sample(PATTERN_LIBRARY, room_rows)

    height = sum(row_heights) + room_rows + 1
    width = sum(col_widths) + room_cols + 1
    grid = np.full((height, width), MAROON, dtype=int)

    row_spans: list[tuple[int, int]] = []
    cursor = 1
    for room_h in row_heights:
        row_spans.append((cursor, cursor + room_h - 1))
        cursor += room_h + 1

    col_spans: list[tuple[int, int]] = []
    cursor = 1
    for room_w in col_widths:
        col_spans.append((cursor, cursor + room_w - 1))
        cursor += room_w + 1

    legend_w = col_widths[0]
    for i, (r0, r1) in enumerate(row_spans):
        room_h = r1 - r0 + 1
        legend = _render_binary_pattern(
            pattern_masks[i],
            index_colors[i],
            secondary_colors[i],
            (room_h, legend_w),
        )
        grid[r0 : r1 + 1, col_spans[0][0] : col_spans[0][1] + 1] = legend

    assignments = list(index_colors)
    total_targets = room_rows * (room_cols - 1)
    while len(assignments) < total_targets:
        assignments.append(rng.choice(index_colors))
    rng.shuffle(assignments)

    assign_iter = iter(assignments)
    for r0, r1 in row_spans:
        for c0, c1 in col_spans[1:]:
            grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            seed_color = next(assign_iter)
            seed_r = rng.randint(r0, r1)
            seed_c = rng.randint(c0, c1)
            grid[seed_r, seed_c] = seed_color

    assert len(set(assignments)) == room_rows, "every legend texture should be used at least once"
    solved = main(grid)
    assert solved.shape == grid.shape
    assert not np.array_equal(solved, grid), "generated input should require a transformation"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _gen_attempt(rng)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
