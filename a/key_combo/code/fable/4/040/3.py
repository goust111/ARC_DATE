from __future__ import annotations

from collections import Counter
import random

import numpy as np


Direction = int  # 0=top, 1=right, 2=bottom, 3=left
TilePos = tuple[int, int]


def _separator_grid(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    """Return the separator color and the inclusive spans between its lines."""
    height, width = map(int, g.shape)
    candidates: list[tuple[int, list[int], list[int]]] = []
    for value in np.unique(g):
        color = int(value)
        rows = [r for r in range(height) if bool(np.all(g[r, :] == color))]
        cols = [c for c in range(width) if bool(np.all(g[:, c] == color))]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one visible separator-grid color"

    separator, sep_rows, sep_cols = candidates[0]

    def spans(length: int, lines: list[int]) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        start = 0
        for line in lines:
            assert start < line, "separator lines must be one cell wide"
            result.append((start, line - 1))
            start = line + 1
        assert start < length, "separator grid must end with a nonempty tile"
        result.append((start, length - 1))
        return result

    return separator, spans(height, sep_rows), spans(width, sep_cols)


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
    return int(ranked[0][0])


def _head_direction(tile: np.ndarray, bg: int, head_color: int) -> Direction:
    head = np.argwhere(tile == head_color)
    assert head.shape == (1, 2), "a source sprite must have exactly one head pixel"
    foreground = np.argwhere(tile != bg)
    assert len(foreground) >= 3
    r0, c0 = foreground.min(axis=0)
    r1, c1 = foreground.max(axis=0)
    hr, hc = map(int, head[0])

    candidates: list[Direction] = []
    if hr == int(r0) and 2 * hc == int(c0 + c1):
        candidates.append(0)
    if hc == int(c1) and 2 * hr == int(r0 + r1):
        candidates.append(1)
    if hr == int(r1) and 2 * hc == int(c0 + c1):
        candidates.append(2)
    if hc == int(c0) and 2 * hr == int(r0 + r1):
        candidates.append(3)
    assert len(candidates) == 1, "the head must uniquely identify one sprite side"
    return candidates[0]


def _marker_direction(tile: np.ndarray, bg: int) -> tuple[int, Direction]:
    coords = np.argwhere(tile != bg)
    assert coords.shape == (1, 2), "a target tile must contain one marker pixel"
    r, c = map(int, coords[0])
    size = int(tile.shape[0])
    mid = size // 2
    positions = ((0, mid), (mid, size - 1), (size - 1, mid), (mid, 0))
    matches = [direction for direction, pos in enumerate(positions) if (r, c) == pos]
    assert len(matches) == 1, "a marker must touch exactly one tile edge at its midpoint"
    return int(tile[r, c]), matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Select one of two headed sprites by marker color, rotate it toward the marker's edge, and stamp it into that tile.

    Input structure:
    - Full rows and columns of one separator color divide the grid into equal square tiles.
    - Exactly two source tiles contain multicolor sprites; each has a repeated body color and one uniquely colored head pixel on one bbox side's central axis.
    - Every target tile contains one head-colored marker at the inner midpoint of its top, right, bottom, or left edge; other tiles are background-only.

    Rule:
    - Match each marker color to the source whose single head has that color.
    - Rotate that complete square source tile so its head points toward the marker's edge, then replace the target tile with the rotated source tile, erasing the marker.
    - Preserve both source tiles, all separator lines, and all unmarked tiles.

    Key insight:
    - Marker color chooses the template while marker side chooses its requested heading, and the source head supplies the reference heading needed to determine the rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    separator, row_spans, col_spans = _separator_grid(g)

    heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(heights) == len(widths) == 1
    tile_h, tile_w = heights.pop(), widths.pop()
    assert tile_h == tile_w and tile_h % 2 == 1, "expected equal odd square tiles"

    bg = _unique_mode(g[g != separator])
    assert bg != separator

    sources: list[tuple[np.ndarray, int, Direction]] = []
    singleton_tiles: list[tuple[int, int, np.ndarray]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            non_bg = tile[tile != bg]
            if non_bg.size == 0:
                continue
            colors, counts = np.unique(non_bg, return_counts=True)
            if non_bg.size == 1:
                singleton_tiles.append((r0, c0, tile))
                continue
            assert len(colors) == 2, "each source contains one body color and one head color"
            head_candidates = [int(color) for color, count in zip(colors, counts) if int(count) == 1]
            body_counts = [int(count) for count in counts if int(count) > 1]
            assert len(head_candidates) == 1 and len(body_counts) == 1
            head_color = head_candidates[0]
            sources.append((tile.copy(), head_color, _head_direction(tile, bg, head_color)))

    assert len(sources) == 2, "expected exactly two source sprites"
    by_head = {head_color: (tile, direction) for tile, head_color, direction in sources}
    assert len(by_head) == 2, "source head colors must be different"
    assert singleton_tiles, "expected at least one marked target tile"

    out = g.copy()
    for r0, c0, tile in singleton_tiles:
        marker_color, requested = _marker_direction(tile, bg)
        assert marker_color in by_head, "marker color must select one source head"
        source_tile, native = by_head[marker_color]
        quarter_turns_ccw = (native - requested) % 4
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = np.rot90(
            source_tile, quarter_turns_ccw
        )
    return out


# ------------------------------ generator ------------------------------

_COLORS = tuple(range(10))
_TILE_SIZES = (5, 7)
_MAX_ATTEMPTS = 300


def _tile_origin(pos: TilePos, size: int) -> tuple[int, int]:
    return pos[0] * (size + 1), pos[1] * (size + 1)


def _base_grid(tile_rows: int, tile_cols: int, size: int, bg: int, separator: int) -> np.ndarray:
    height = tile_rows * size + tile_rows - 1
    width = tile_cols * size + tile_cols - 1
    grid = np.full((height, width), bg, dtype=int)
    grid[size :: size + 1, :] = separator
    grid[:, size :: size + 1] = separator
    return grid


def _put_tile(grid: np.ndarray, pos: TilePos, tile: np.ndarray, size: int) -> None:
    r0, c0 = _tile_origin(pos, size)
    grid[r0 : r0 + size, c0 : c0 + size] = tile


def _north_sprite(size: int, bg: int, body_color: int, head_color: int) -> np.ndarray:
    """Build a random connected sprite whose unique head points upward."""
    mid = size // 2
    radius = 1 if size == 5 else random.choice((1, 2))
    body: set[tuple[int, int]] = {(r, mid) for r in range(2, mid + 1)}
    body.update((mid, c) for c in range(mid - radius, mid + radius + 1))

    desired = random.randint(len(body) + 1, min(len(body) + size, (size - 3) * (2 * radius + 1)))
    allowed = {
        (r, c)
        for r in range(2, size - 1)
        for c in range(mid - radius, mid + radius + 1)
    }
    while len(body) < desired:
        frontier = [
            cell
            for cell in allowed - body
            if any(
                (cell[0] + dr, cell[1] + dc) in body
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
        ]
        if not frontier:
            break
        body.add(random.choice(frontier))

    tile = np.full((size, size), bg, dtype=int)
    for r, c in body:
        tile[r, c] = body_color
    tile[1, mid] = head_color
    assert _head_direction(tile, bg, head_color) == 0
    return tile


def _target_assignments(count: int) -> list[tuple[Direction, int]]:
    # These six jointly defeat side-only and color-only shortcuts while covering all sides.
    assignments: list[tuple[Direction, int]] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 1),
    ]
    while len(assignments) < count:
        assignments.append((random.randrange(4), random.randrange(2)))
    random.shuffle(assignments)
    return assignments


def _generated_semantics_hold(
    inp: np.ndarray,
    expected: np.ndarray,
    *,
    size: int,
    separator: int,
    bg: int,
    source_positions: list[TilePos],
    source_tiles: list[np.ndarray],
    native_headings: list[Direction],
    target_positions: list[TilePos],
    assignments: list[tuple[Direction, int]],
) -> bool:
    """Check the stated semantics and the dependencies that rule out easy shortcuts."""
    try:
        actual = main(inp)
    except (AssertionError, ValueError):
        return False
    if not np.array_equal(actual, expected) or np.array_equal(inp, expected):
        return False
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False

    # Separator lines and source tiles must remain exactly unchanged.
    if not np.array_equal(actual[inp == separator], inp[inp == separator]):
        return False
    for index, pos in enumerate(source_positions):
        r0, c0 = _tile_origin(pos, size)
        if not np.array_equal(inp[r0 : r0 + size, c0 : c0 + size], source_tiles[index]):
            return False
        if not np.array_equal(actual[r0 : r0 + size, c0 : c0 + size], source_tiles[index]):
            return False
        # The exact head color is recovered below without relying on its numeric id.
        singles = [
            int(color)
            for color, n in zip(*np.unique(source_tiles[index][source_tiles[index] != bg], return_counts=True))
            if int(n) == 1
        ]
        if len(singles) != 1 or _head_direction(source_tiles[index], bg, singles[0]) != native_headings[index]:
            return False

    directions = {direction for direction, _source in assignments}
    selected = {source for _direction, source in assignments}
    if directions != {0, 1, 2, 3} or selected != {0, 1}:
        return False
    for source_index in (0, 1):
        if len({d for d, s in assignments if s == source_index}) < 2:
            return False

    rendered: list[np.ndarray] = []
    for pos, (requested, source_index) in zip(target_positions, assignments):
        r0, c0 = _tile_origin(pos, size)
        input_tile = inp[r0 : r0 + size, c0 : c0 + size]
        marker_coords = np.argwhere(input_tile != bg)
        marker_expected = ((0, size // 2), (size // 2, size - 1), (size - 1, size // 2), (size // 2, 0))[requested]
        if marker_coords.shape != (1, 2) or tuple(map(int, marker_coords[0])) != marker_expected:
            return False
        stamp = np.rot90(source_tiles[source_index], (native_headings[source_index] - requested) % 4)
        if not np.array_equal(actual[r0 : r0 + size, c0 : c0 + size], stamp):
            return False
        rendered.append(stamp)

    # A same-side/different-source pair proves color selection affects the result.
    color_witness = any(
        d1 == d2 and s1 != s2 and not np.array_equal(rendered[i], rendered[j])
        for i, (d1, s1) in enumerate(assignments)
        for j, (d2, s2) in enumerate(assignments[i + 1 :], start=i + 1)
    )
    # A same-source/different-side pair proves side-based rotation affects the result.
    side_witness = any(
        s1 == s2 and d1 != d2 and not np.array_equal(rendered[i], rendered[j])
        for i, (d1, s1) in enumerate(assignments)
        for j, (d2, s2) in enumerate(assignments[i + 1 :], start=i + 1)
    )
    return color_witness and side_witness


def _generate_attempt() -> np.ndarray | None:
    size = random.choice(_TILE_SIZES)
    if size == 7:
        tile_rows = tile_cols = 3
    else:
        tile_rows = random.randint(3, 5)
        tile_cols = random.randint(3, 5)

    separator, bg, body0, body1, head0, head1 = random.sample(_COLORS, 6)
    body_colors = (body0, body1)
    head_colors = (head0, head1)
    native_headings = random.sample(range(4), 2)

    north_tiles: list[np.ndarray] = []
    for index in range(2):
        for _ in range(50):
            candidate = _north_sprite(size, bg, body_colors[index], head_colors[index])
            candidate_mask = candidate != bg
            if not north_tiles or not np.array_equal(candidate_mask, north_tiles[0] != bg):
                north_tiles.append(candidate)
                break
        else:
            return None
    source_tiles = [
        np.rot90(tile, (-native_headings[index]) % 4)
        for index, tile in enumerate(north_tiles)
    ]

    positions = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]
    random.shuffle(positions)
    source_positions = positions[:2]
    max_targets = min(10, len(positions) - 3)  # Always retain at least one blank tile.
    target_count = random.randint(6, max_targets)
    target_positions = positions[2 : 2 + target_count]
    assignments = _target_assignments(target_count)

    inp = _base_grid(tile_rows, tile_cols, size, bg, separator)
    expected = inp.copy()
    for pos, tile in zip(source_positions, source_tiles):
        _put_tile(inp, pos, tile, size)
        _put_tile(expected, pos, tile, size)

    marker_positions = ((0, size // 2), (size // 2, size - 1), (size - 1, size // 2), (size // 2, 0))
    for pos, (requested, source_index) in zip(target_positions, assignments):
        marker_tile = np.full((size, size), bg, dtype=int)
        mr, mc = marker_positions[requested]
        marker_tile[mr, mc] = head_colors[source_index]
        _put_tile(inp, pos, marker_tile, size)

        stamp = np.rot90(
            source_tiles[source_index],
            (native_headings[source_index] - requested) % 4,
        )
        _put_tile(expected, pos, stamp, size)

    if not _generated_semantics_hold(
        inp,
        expected,
        size=size,
        separator=separator,
        bg=bg,
        source_positions=source_positions,
        source_tiles=source_tiles,
        native_headings=native_headings,
        target_positions=target_positions,
        assignments=assignments,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        result = _generate_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid dual-source stamp puzzle")
