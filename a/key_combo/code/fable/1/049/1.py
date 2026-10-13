from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(set(map(int, np.asarray(grid, dtype=int).ravel())))


def _open_runs(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    separator_set = set(map(int, separators))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(int(length)):
        if idx in separator_set:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, int(length) - 1))
    return runs


def _candidate_layout(grid: np.ndarray, separator: int) -> dict | None:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)

    row_lines = np.flatnonzero(np.all(g == int(separator), axis=1))
    col_lines = np.flatnonzero(np.all(g == int(separator), axis=0))
    if len(row_lines) == 0 or len(col_lines) == 0:
        return None

    row_cells = _open_runs(height, row_lines)
    col_cells = _open_runs(width, col_lines)
    n = len(row_cells)
    if n < 2 or n != len(col_cells):
        return None

    row_heights = {int(r1 - r0 + 1) for r0, r1 in row_cells}
    col_widths = {int(c1 - c0 + 1) for c0, c1 in col_cells}
    if len(row_heights) != 1 or len(col_widths) != 1:
        return None

    tile_h = row_heights.pop()
    tile_w = col_widths.pop()
    if tile_h != tile_w or tile_h < n:
        return None

    row_keep = np.ones(height, dtype=bool)
    col_keep = np.ones(width, dtype=bool)
    row_keep[row_lines] = False
    col_keep[col_lines] = False
    interior = g[np.ix_(row_keep, col_keep)]
    interior_values = [int(v) for v in interior.ravel() if int(v) != int(separator)]
    if not interior_values:
        return None
    bg = int(Counter(interior_values).most_common(1)[0][0])
    if bg == int(separator):
        return None

    entities: dict[Coord, np.ndarray] = {}
    pointers: dict[Coord, Coord] = {}
    pointer_count = 0

    for tile_r, (r0, r1) in enumerate(row_cells):
        for tile_c, (c0, c1) in enumerate(col_cells):
            tile = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
            non_bg = np.argwhere(tile != int(bg))
            if len(non_bg) == 0:
                continue
            if len(non_bg) == 1:
                pr, pc = map(int, non_bg[0])
                if (
                    int(tile[pr, pc]) == int(separator)
                    and pr < n
                    and pc < n
                    and (pr, pc) != (tile_r, tile_c)
                ):
                    pointers[(tile_r, tile_c)] = (pr, pc)
                    pointer_count += 1
                    continue
            entities[(tile_r, tile_c)] = tile.copy()

    if not entities or not pointers:
        return None

    resolved: dict[Coord, Coord] = {}
    visiting: set[Coord] = set()

    def resolve(pos: Coord) -> Coord:
        if pos in resolved:
            return resolved[pos]
        if pos in visiting:
            raise ValueError("pointer cycle")
        visiting.add(pos)
        if pos in entities:
            end = pos
        elif pos in pointers:
            target = pointers[pos]
            if target == pos:
                raise ValueError("self pointer")
            end = resolve(target)
        else:
            raise ValueError("pointer must resolve to an entity")
        visiting.remove(pos)
        resolved[pos] = end
        return end

    try:
        for pos in pointers:
            resolve(pos)
    except ValueError:
        return None

    return {
        "separator": int(separator),
        "bg": int(bg),
        "n": int(n),
        "tile_size": int(tile_h),
        "row_cells": row_cells,
        "col_cells": col_cells,
        "entities": entities,
        "pointers": pointers,
        "resolved": resolved,
        "pointer_count": int(pointer_count),
    }


def _infer_layout(grid: np.ndarray) -> dict:
    candidates = []
    for color in _colors_present(grid):
        parsed = _candidate_layout(grid, int(color))
        if parsed is not None:
            candidates.append(parsed)
    assert candidates, "failed to infer separator grid layout"
    candidates.sort(
        key=lambda item: (int(item["pointer_count"]), len(item["entities"])),
        reverse=True,
    )
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each pointer tile with the sprite tile reached by following chained tile addresses.

    Input structure:
    - A separator color forms full rows and columns that partition the grid into an n by n array of equal square tiles.
    - Some tiles contain full multi-pixel sprite patterns, while empty tiles contain only the background.
    - Every other non-empty tile contains exactly one separator-colored pixel whose local coordinate is a tile address.

    Rule:
    - Read a pointer tile's single pixel at local coordinate (r, c) as an address to tile-row r and tile-column c in the tile array.
    - If the addressed tile is itself a pointer tile, keep following pointer tiles until an actual sprite tile is reached.
    - Replace each pointer tile by a copy of the final sprite tile it resolves to.
    - Leave original sprite tiles, empty tiles, and all separator lines unchanged.

    Key insight:
    - The lone separator-colored pixel inside a tile is not a decoration but a coordinate in tile-local space; only by tracing that coordinate chain to its terminal sprite can the correct stamp for each tile be determined.
    """
    g = np.asarray(grid, dtype=int)
    layout = _infer_layout(g)
    out = g.copy()

    for pos, end in layout["resolved"].items():
        r0, r1 = layout["row_cells"][pos[0]]
        c0, c1 = layout["col_cells"][pos[1]]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = layout["entities"][end]

    return out


GEN_BG = 0
GEN_N_RANGE = (3, 5)
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _blank_partition(n: int, separator: int) -> np.ndarray:
    side = n * n + n - 1
    grid = np.full((side, side), GEN_BG, dtype=int)
    for k in range(1, n):
        idx = k * (n + 1) - 1
        grid[idx, :] = int(separator)
        grid[:, idx] = int(separator)
    return grid


def _tile_origin(n: int, pos: Coord) -> Coord:
    return int(pos[0] * (n + 1)), int(pos[1] * (n + 1))


def _write_tile(grid: np.ndarray, n: int, pos: Coord, tile: np.ndarray) -> None:
    r0, c0 = _tile_origin(n, pos)
    grid[r0 : r0 + n, c0 : c0 + n] = tile


def _random_connected_mask(size: int, area: int) -> np.ndarray | None:
    mask = np.zeros((size, size), dtype=bool)
    seed = (random.randrange(size), random.randrange(size))
    mask[seed] = True
    frontier = {
        (nr, nc)
        for nr, nc in _neighbors4(*seed)
        if 0 <= nr < size and 0 <= nc < size
    }

    while int(mask.sum()) < int(area):
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _random_sprite(size: int, separator: int) -> np.ndarray:
    palette = [c for c in range(10) if c not in (GEN_BG, int(separator))]
    max_area = min(size * size - 1, size + 2)
    for _ in range(80):
        area = random.randint(3, max_area)
        mask = _random_connected_mask(size, area)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        if len(set(map(int, coords[:, 0]))) < 2 or len(set(map(int, coords[:, 1]))) < 2:
            continue

        color_count = random.randint(2, min(3, area))
        colors = random.sample(palette, color_count)
        coord_list = [tuple(map(int, coord)) for coord in coords]
        random.shuffle(coord_list)

        tile = np.full((size, size), GEN_BG, dtype=int)
        for idx, (r, c) in enumerate(coord_list):
            if idx < color_count:
                tile[r, c] = int(colors[idx])
            else:
                tile[r, c] = int(random.choice(colors))

        if len(set(map(int, tile[mask]))) >= 2:
            return tile
    raise RuntimeError("failed to generate a sprite tile")


def _pointer_patch(n: int, separator: int, target: Coord) -> np.ndarray:
    tile = np.full((n, n), GEN_BG, dtype=int)
    tile[int(target[0]), int(target[1])] = int(separator)
    return tile


def _resolve_generated(pos: Coord, pointers: dict[Coord, Coord]) -> Coord:
    seen: set[Coord] = set()
    cur = pos
    while cur in pointers:
        if cur in seen:
            raise ValueError("generator created a cycle")
        seen.add(cur)
        cur = pointers[cur]
    return cur


def _sample_counts(total_tiles: int, n: int) -> tuple[int, int]:
    entity_count = random.randint(2, min(4, n))
    remaining = total_tiles - entity_count
    empty_lo = max(1, remaining // 4)
    empty_hi = max(empty_lo, remaining // 2)
    empty_count = random.randint(empty_lo, empty_hi)
    pointer_count = remaining - empty_count
    return entity_count, pointer_count


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*GEN_N_RANGE)
    separator = random.randint(1, 9)
    total_tiles = n * n
    entity_count, pointer_count = _sample_counts(total_tiles, n)
    if pointer_count < 3:
        return None

    all_positions = [(r, c) for r in range(n) for c in range(n)]
    entity_positions = random.sample(all_positions, entity_count)
    remaining_positions = [pos for pos in all_positions if pos not in entity_positions]
    pointer_positions = random.sample(remaining_positions, pointer_count)

    sprites: dict[Coord, np.ndarray] = {}
    seen_sprites: set[bytes] = set()
    for pos in entity_positions:
        for _ in range(100):
            sprite = _random_sprite(n, separator)
            sig = sprite.tobytes()
            if sig not in seen_sprites:
                seen_sprites.add(sig)
                sprites[pos] = sprite
                break
        else:
            return None

    random.shuffle(pointer_positions)
    chain_len = random.randint(2, min(4, pointer_count - 1))
    chain_nodes = pointer_positions[:chain_len]
    other_pointers = pointer_positions[chain_len:]
    root_entity, alt_entity = random.sample(entity_positions, 2)

    pointers: dict[Coord, Coord] = {}
    pointers[chain_nodes[-1]] = root_entity
    for idx in range(chain_len - 2, -1, -1):
        pointers[chain_nodes[idx]] = chain_nodes[idx + 1]

    direct_pointer = other_pointers.pop()
    pointers[direct_pointer] = alt_entity
    random.shuffle(other_pointers)
    for pos in other_pointers:
        pointer_targets = list(pointers)
        if pointer_targets and random.random() < 0.55:
            pointers[pos] = random.choice(pointer_targets)
        else:
            pointers[pos] = random.choice(entity_positions)

    output = _blank_partition(n, separator)
    for pos, sprite in sprites.items():
        _write_tile(output, n, pos, sprite)
    for pos in pointer_positions:
        end = _resolve_generated(pos, pointers)
        if end not in sprites:
            return None
        _write_tile(output, n, pos, sprites[end])

    inp = _blank_partition(n, separator)
    for pos, sprite in sprites.items():
        _write_tile(inp, n, pos, sprite)
    for pos in pointer_positions:
        _write_tile(inp, n, pos, _pointer_patch(n, separator, pointers[pos]))

    if np.array_equal(inp, output):
        return None
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain tile grid")
