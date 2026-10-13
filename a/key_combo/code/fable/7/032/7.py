"""ARC task 7: every marked tile states its own destination slot."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
YELLOW = 4
SEPARATOR = 5
MACRO_SIZE = 3
TILE_SIZE = 3
BODY_COLORS = (1, 2, 3, 6, 7, 8, 9)


def _bands(separators: list[int], length: int) -> list[tuple[int, int]]:
    """Return half-open bands between separator indices."""
    bounds = [-1, *separators, length]
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]


def _layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Recover the tile bands from the two full grey separator lines."""
    h, w = grid.shape
    row_seps = [r for r in range(h) if np.all(grid[r, :] == SEPARATOR)]
    col_seps = [c for c in range(w) if np.all(grid[:, c] == SEPARATOR)]
    assert len(row_seps) == len(col_seps) == MACRO_SIZE - 1

    row_bands = _bands(row_seps, h)
    col_bands = _bands(col_seps, w)
    assert len(row_bands) == len(col_bands) == MACRO_SIZE
    assert all(hi - lo == TILE_SIZE for lo, hi in row_bands)
    assert all(hi - lo == TILE_SIZE for lo, hi in col_bands)

    non_separator = np.ones(grid.shape, dtype=bool)
    non_separator[row_seps, :] = False
    non_separator[:, col_seps] = False
    assert not np.any(grid[non_separator] == SEPARATOR)
    return row_bands, col_bands


def main(grid: np.ndarray) -> np.ndarray:
    """Simultaneously relocate every sprite to the tile encoded by its yellow pixel.

    Input structure:
    - Two full grey rows and two full grey columns divide a black canvas into a
      3x3 array of 3x3 tiles. Each occupied tile contains one yellow marker and
      a colored sprite; unoccupied tiles are entirely black.

    Rule:
    - In every occupied tile, read the yellow pixel's local (row, column) as the
      destination tile's macro (row, column). Clear all tile interiors, then
      copy every complete 3x3 source tile to its distinct destination at once,
      preserving the grey separator lines.

    Key insight:
    - Each yellow pixel is both part of its sprite and a local-coordinate
      address, so occupied destinations must be read before any tile is erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    row_bands, col_bands = _layout(g)

    moves: list[tuple[tuple[int, int], np.ndarray]] = []
    destinations: set[tuple[int, int]] = set()
    for macro_r, (r0, r1) in enumerate(row_bands):
        for macro_c, (c0, c1) in enumerate(col_bands):
            tile = g[r0:r1, c0:c1]
            if np.all(tile == BACKGROUND):
                continue
            markers = np.argwhere(tile == YELLOW)
            assert markers.shape == (1, 2), (
                f"occupied tile {(macro_r, macro_c)} must have exactly one yellow marker"
            )
            destination = tuple(map(int, markers[0]))
            assert destination not in destinations, "sprite destinations must be distinct"
            destinations.add(destination)
            moves.append((destination, tile.copy()))

    assert moves, "expected at least one occupied tile"
    out = np.full_like(g, BACKGROUND)
    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR)]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR)]
    out[row_seps, :] = SEPARATOR
    out[:, col_seps] = SEPARATOR

    # Every source patch was saved before this loop: this is simultaneous motion.
    for (dst_r, dst_c), tile in moves:
        r0, r1 = row_bands[dst_r]
        c0, c1 = col_bands[dst_c]
        out[r0:r1, c0:c1] = tile
    return out


def _empty_grid() -> np.ndarray:
    grid = np.zeros((11, 11), dtype=int)
    grid[(3, 7), :] = SEPARATOR
    grid[:, (3, 7)] = SEPARATOR
    return grid


def _tile_slice(slot: tuple[int, int]) -> tuple[slice, slice]:
    macro_r, macro_c = slot
    r0 = macro_r * (TILE_SIZE + 1)
    c0 = macro_c * (TILE_SIZE + 1)
    return slice(r0, r0 + TILE_SIZE), slice(c0, c0 + TILE_SIZE)


def _connected_sprite(marker: tuple[int, int], color: int) -> np.ndarray:
    """Grow a clear connected polyomino containing the address marker."""
    target_area = random.randint(3, 7)
    cells = {marker}
    while len(cells) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < TILE_SIZE and 0 <= nc < TILE_SIZE and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))

    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    for cell in cells:
        tile[cell] = color
    tile[marker] = YELLOW
    return tile


def _choose_mapping() -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]], str]:
    """Choose a swap, long cycle, or cycle-plus-empty-destinations mapping."""
    slots = [(r, c) for r in range(MACRO_SIZE) for c in range(MACRO_SIZE)]
    mode = random.choice(("swap", "long_cycle", "mixed"))

    if mode == "swap":
        count = random.randint(3, 5)
        sources = random.sample(slots, count)
        cycle_len = 2
    elif mode == "long_cycle":
        count = random.randint(3, 8)
        sources = random.sample(slots, count)
        cycle_len = count
    else:
        count = random.randint(4, 7)
        sources = random.sample(slots, count)
        minimum_cycle = max(2, 2 * count - len(slots))
        cycle_len = random.randint(minimum_cycle, count - 1)

    random.shuffle(sources)
    cycle = sources[:cycle_len]
    mapping = {cycle[i]: cycle[(i + 1) % cycle_len] for i in range(cycle_len)}

    remaining_sources = sources[cycle_len:]
    empty_slots = [slot for slot in slots if slot not in sources]
    random.shuffle(empty_slots)
    for source, destination in zip(remaining_sources, empty_slots):
        mapping[source] = destination

    assert set(mapping) == set(sources)
    assert len(set(mapping.values())) == len(mapping)
    assert all(source != destination for source, destination in mapping.items())
    return sources, mapping, mode


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    sources: list[tuple[int, int]],
    mapping: dict[tuple[int, int], tuple[int, int]],
    mode: str,
) -> bool:
    """Check address decoding, exact copying, erasure, and non-degenerate motion."""
    if not np.array_equal(main(inp), expected) or np.array_equal(inp, expected):
        return False
    occupied = set(sources)
    if not any(dst in occupied for dst in mapping.values()):
        return False
    if mode == "mixed" and not any(dst not in occupied for dst in mapping.values()):
        return False

    for source, destination in mapping.items():
        src_slice = _tile_slice(source)
        dst_slice = _tile_slice(destination)
        tile = inp[src_slice]
        marker = np.argwhere(tile == YELLOW)
        if marker.shape != (1, 2) or tuple(map(int, marker[0])) != destination:
            return False
        if not np.array_equal(expected[dst_slice], tile):
            return False

    incoming = set(mapping.values())
    for source in occupied - incoming:
        if np.any(expected[_tile_slice(source)] != BACKGROUND):
            return False
    return True


def generate_input() -> np.ndarray:
    """Generate varied, oracle-checked simultaneous tile relocations."""
    for _ in range(300):
        sources, mapping, mode = _choose_mapping()
        inp = _empty_grid()
        expected = _empty_grid()
        colors = random.sample(BODY_COLORS, min(len(sources), len(BODY_COLORS)))
        if len(colors) < len(sources):
            colors.extend(random.choices(BODY_COLORS, k=len(sources) - len(colors)))
        random.shuffle(colors)

        source_tiles: dict[tuple[int, int], np.ndarray] = {}
        for source, color in zip(sources, colors):
            destination = mapping[source]
            tile = _connected_sprite(destination, color)
            source_tiles[source] = tile
            inp[_tile_slice(source)] = tile
        for source, destination in mapping.items():
            expected[_tile_slice(destination)] = source_tiles[source]

        if _semantic_check(inp, expected, sources, mapping, mode):
            return inp
    raise RuntimeError("failed to generate a valid simultaneous-relocation grid")

