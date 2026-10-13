from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
NON_BACKGROUND_COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 800
MAX_THICKNESS_PER_COLOR = 4
MIN_CENTER_REMAINDER = 3


def _extract_palette(grid: np.ndarray) -> tuple[int, ...]:
    assert grid.ndim == 2, "expected a 2D grid"
    assert max(grid.shape) <= 30, "ARC grids must fit within 30x30"
    assert grid.shape[0] >= 2 and grid.shape[1] >= 4, "expected a palette row and a board"

    top = grid[0, :]
    palette: list[int] = []
    for value in top:
        color = int(value)
        if color == BACKGROUND:
            break
        palette.append(color)

    assert palette, "expected a non-empty left-justified palette"
    assert len(set(palette)) == len(palette), "palette colors must not repeat"
    assert all(color != BACKGROUND for color in palette), "palette colors must be non-black"
    assert np.all(top[len(palette) :] == BACKGROUND), "palette row must be black after the palette"
    return tuple(palette)


def _count_board_pixels(grid: np.ndarray, palette: tuple[int, ...]) -> tuple[int, ...]:
    board = grid[1:, :]
    assert board.size > 0, "expected a non-empty board area"

    allowed = set(palette) | {BACKGROUND}
    present = set(map(int, np.unique(board)))
    assert present.issubset(allowed), "board may contain only black and palette colors"

    counts = tuple(int(np.count_nonzero(board == color)) for color in palette)
    assert all(count > 0 for count in counts), "each palette color must appear as scattered pixels"
    assert 2 * sum(counts) <= min(board.shape), "total thickness must fit in the board"
    return counts


def _draw_nested_frames(out: np.ndarray, palette: tuple[int, ...], counts: tuple[int, ...]) -> None:
    assert len(palette) == len(counts), "palette and count lists must align"

    r0, c0 = 1, 0
    r1, c1 = out.shape[0] - 1, out.shape[1] - 1
    for color, thickness in zip(palette, counts):
        t = int(thickness)
        assert t > 0, "frame thickness must be positive"
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 2 * t and width >= 2 * t, "nested frame does not fit"

        out[r0 : r0 + t, c0 : c1 + 1] = int(color)
        out[r1 - t + 1 : r1 + 1, c0 : c1 + 1] = int(color)
        out[r0 : r1 + 1, c0 : c0 + t] = int(color)
        out[r0 : r1 + 1, c1 - t + 1 : c1 + 1] = int(color)

        r0 += t
        c0 += t
        r1 -= t
        c1 -= t


def _render_from_spec(
    shape: tuple[int, int],
    top_row: np.ndarray,
    palette: tuple[int, ...],
    counts: tuple[int, ...],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[0, :] = top_row
    _draw_nested_frames(out, palette, counts)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the lower board as nested thick rectangular frames in top-row palette order.

    Input structure:
    - The top row starts at the left with a contiguous palette of distinct non-black
      colors, and the rest of that row is black.
    - The board below the palette row contains only black cells and scattered pixels
      whose colors come from the palette.
    - Every palette color appears at least once, and the total scatter count leaves
      enough room for all nested frames inside the board.

    Rule:
    - Keep the top palette row unchanged and erase all scattered pixels below it.
    - For each palette color from left to right, count its scattered pixels in the
      board and draw the next concentric rectangular frame with that color.
    - The first frame hugs the board boundary, each later frame hugs the previous
      frame's inside edge, and each frame thickness equals that color's scatter count.
    - Any central rectangle left after all layers are drawn remains black.

    Key insight:
    - Scatter positions are decoys: palette writing order alone sets outside-to-inside
      nesting, while each color's pixel count supplies only that layer's thickness.
    """
    g = np.asarray(grid, dtype=int)
    palette = _extract_palette(g)
    counts = _count_board_pixels(g, palette)
    return _render_from_spec(g.shape, g[0, :], palette, counts)


def _board_depth(br: int, c: int, board_h: int, width: int) -> int:
    return int(min(br, c, board_h - 1 - br, width - 1 - c))


def _cell_is_clear(cell: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    r, c = cell
    for ur, uc in used:
        if abs(r - ur) <= 1 and abs(c - uc) <= 1:
            return False
    return True


def _candidate_cells(
    board_h: int,
    width: int,
    used: set[tuple[int, int]],
    *,
    min_depth: int | None = None,
    max_depth: int | None = None,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for br in range(board_h):
        for c in range(width):
            depth = _board_depth(br, c, board_h, width)
            if min_depth is not None and depth < min_depth:
                continue
            if max_depth is not None and depth > max_depth:
                continue
            cell = (br, c)
            if cell in used or not _cell_is_clear(cell, used):
                continue
            cells.append(cell)
    return cells


def _sample_cells(
    board_h: int,
    width: int,
    count: int,
    used: set[tuple[int, int]],
    *,
    min_depth: int | None = None,
    max_depth: int | None = None,
) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    for _ in range(count):
        options = _candidate_cells(
            board_h,
            width,
            used,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if not options:
            return None
        cell = random.choice(options)
        used.add(cell)
        chosen.append(cell)
    return chosen


def _generate_shape() -> tuple[int, int]:
    mode = random.choice(("wide", "tall", "balanced"))
    if mode == "wide":
        board_h = random.randint(13, 23)
        width = random.randint(board_h + 2, min(30, board_h + 7))
    elif mode == "tall":
        width = random.randint(13, 23)
        board_h = random.randint(width + 2, min(29, width + 7))
    else:
        board_h = random.randint(13, 29)
        width = min(30, max(13, board_h + random.randint(-3, 3)))
    return board_h + 1, width


def _generate_counts(layer_count: int, max_total: int) -> tuple[int, ...] | None:
    extra_capacity = max_total - layer_count
    if extra_capacity < 1:
        return None

    extra = random.randint(1, min(extra_capacity, layer_count * (MAX_THICKNESS_PER_COLOR - 1)))
    counts = [1] * layer_count
    while extra > 0:
        eligible = [i for i, count in enumerate(counts) if count < MAX_THICKNESS_PER_COLOR]
        if not eligible:
            return None
        idx = random.choice(eligible)
        counts[idx] += 1
        extra -= 1
    return tuple(counts)


def _place_scatter(
    board_h: int,
    width: int,
    palette: tuple[int, ...],
    counts: tuple[int, ...],
) -> dict[int, list[tuple[int, int]]] | None:
    used: set[tuple[int, int]] = set()
    positions: dict[int, list[tuple[int, int]]] = {}
    short_side = min(board_h, width)

    central_min_depth = max(2, short_side // 3)
    first_cells = _sample_cells(
        board_h,
        width,
        counts[0],
        used,
        min_depth=central_min_depth,
    )
    if first_cells is None:
        return None
    positions[palette[0]] = first_cells

    last_cells = _sample_cells(
        board_h,
        width,
        counts[-1],
        used,
        max_depth=1,
    )
    if last_cells is None:
        return None
    positions[palette[-1]] = last_cells

    middle_indices = list(range(1, len(palette) - 1))
    random.shuffle(middle_indices)
    for idx in middle_indices:
        cells = _sample_cells(board_h, width, counts[idx], used)
        if cells is None:
            return None
        positions[palette[idx]] = cells

    return positions


def _build_input(
    shape: tuple[int, int],
    palette: tuple[int, ...],
    positions: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[0, : len(palette)] = np.array(palette, dtype=int)
    for color, cells in positions.items():
        for br, c in cells:
            grid[br + 1, c] = int(color)
    return grid


def _mean_depth(
    cells: list[tuple[int, int]],
    board_h: int,
    width: int,
) -> float:
    return float(sum(_board_depth(r, c, board_h, width) for r, c in cells) / len(cells))


def _semantically_valid(
    grid: np.ndarray,
    palette: tuple[int, ...],
    counts: tuple[int, ...],
    positions: dict[int, list[tuple[int, int]]],
) -> bool:
    try:
        parsed_palette = _extract_palette(grid)
        parsed_counts = _count_board_pixels(grid, parsed_palette)
        assert parsed_palette == palette
        assert parsed_counts == counts

        expected = _render_from_spec(grid.shape, grid[0, :], palette, counts)
        actual = main(grid)
        assert np.array_equal(actual, expected)

        total = sum(counts)
        center = actual[1 + total : grid.shape[0] - total, total : grid.shape[1] - total]
        assert center.shape[0] >= MIN_CENTER_REMAINDER
        assert center.shape[1] >= MIN_CENTER_REMAINDER
        assert np.all(center == BACKGROUND)

        board_h = grid.shape[0] - 1
        width = grid.shape[1]
        first_mean = _mean_depth(positions[palette[0]], board_h, width)
        last_mean = _mean_depth(positions[palette[-1]], board_h, width)
        assert first_mean > last_mean + 1.0

        depth_rank = sorted(
            (
                (_mean_depth(positions[color], board_h, width), idx, color)
                for idx, color in enumerate(palette)
            )
        )
        depth_palette = tuple(color for _depth, _idx, color in depth_rank)
        assert depth_palette != palette
        count_by_color = dict(zip(palette, counts))
        depth_counts = tuple(count_by_color[color] for color in depth_palette)
        depth_order_output = _render_from_spec(grid.shape, grid[0, :], depth_palette, depth_counts)
        assert not np.array_equal(depth_order_output, expected)

        swapped = grid.copy()
        swapped[0, 0], swapped[0, 1] = int(swapped[0, 1]), int(swapped[0, 0])
        assert not np.array_equal(main(swapped), expected)

        thick_colors = [color for color, count in zip(palette, counts) if count > 1]
        assert thick_colors
        color = random.choice(thick_colors)
        br, c = positions[color][0]
        thinner = grid.copy()
        thinner[br + 1, c] = BACKGROUND
        assert not np.array_equal(main(thinner), expected)
    except AssertionError:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height, width = _generate_shape()
    board_h = height - 1
    short_side = min(board_h, width)
    max_total = (short_side - MIN_CENTER_REMAINDER) // 2
    max_layers = min(5, max_total - 1)
    if max_layers < 2:
        return None

    layer_count = random.randint(2, max_layers)
    counts = _generate_counts(layer_count, max_total)
    if counts is None:
        return None

    palette = tuple(random.sample(NON_BACKGROUND_COLORS, layer_count))
    positions = _place_scatter(board_h, width, palette, counts)
    if positions is None:
        return None

    grid = _build_input((height, width), palette, positions)
    if _semantically_valid(grid, palette, counts, positions):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid palette nesting input")
