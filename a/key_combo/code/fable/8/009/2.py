import random
from typing import Iterable

import numpy as np


GEN_MAX_ATTEMPTS = 500
GEN_LAYOUTS = ((1, 5), (5, 1), (1, 7), (7, 1), (3, 3))
GEN_LAYOUT_WEIGHTS = (3, 3, 2, 2, 5)
GEN_MIN_TILE_SIDE = 3
GEN_MAX_GRID_SIDE = 30


def _swap_colors(tile: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = tile.copy()
    mask_a = tile == int(color_a)
    mask_b = tile == int(color_b)
    out[mask_a] = int(color_b)
    out[mask_b] = int(color_a)
    return out


def _partner_tile(tile: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    return _swap_colors(np.fliplr(tile), color_a, color_b)


def _signature(tile: np.ndarray) -> bytes:
    return tile.tobytes()


def _extract_spans(separator_mask: np.ndarray) -> list[tuple[int, int]] | None:
    cuts = np.flatnonzero(separator_mask)
    if len(cuts) < 2 or int(cuts[0]) != 0 or int(cuts[-1]) != len(separator_mask) - 1:
        return None
    if np.any(np.diff(cuts) <= 1):
        return None

    spans: list[tuple[int, int]] = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        spans.append((int(left) + 1, int(right) - 1))
    lengths = {right - left + 1 for left, right in spans}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        return None
    return spans


def _parse_tiled_grid(grid: np.ndarray) -> tuple[int, tuple[int, int], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    candidates: list[tuple[int, tuple[int, int], list[np.ndarray]]] = []
    colors = sorted(set(int(x) for x in g.flat))
    for separator in colors:
        row_spans = _extract_spans(np.all(g == separator, axis=1))
        col_spans = _extract_spans(np.all(g == separator, axis=0))
        if row_spans is None or col_spans is None:
            continue

        fg_colors = sorted(set(colors) - {0, int(separator)})
        if len(fg_colors) != 2:
            continue

        tiles: list[np.ndarray] = []
        ok = True
        allowed = {0, fg_colors[0], fg_colors[1]}
        for r0, r1 in row_spans:
            for c0, c1 in col_spans:
                patch = g[r0 : r1 + 1, c0 : c1 + 1]
                if np.any(patch == separator):
                    ok = False
                    break
                if set(int(x) for x in patch.flat) - allowed:
                    ok = False
                    break
                tiles.append(patch.copy())
            if not ok:
                break
        if ok and len(tiles) >= 3 and len(tiles) % 2 == 1:
            candidates.append((int(separator), (fg_colors[0], fg_colors[1]), tiles))

    assert len(candidates) == 1, "expected one valid separator layout"
    return candidates[0]


def _find_orphan_index(tiles: list[np.ndarray], color_a: int, color_b: int) -> int:
    sig_to_index: dict[bytes, int] = {}
    for idx, tile in enumerate(tiles):
        sig = _signature(tile)
        assert sig not in sig_to_index, "tiles must be unique"
        sig_to_index[sig] = idx

    orphan: list[int] = []
    for idx, tile in enumerate(tiles):
        partner_idx = sig_to_index.get(_signature(_partner_tile(tile, color_a, color_b)))
        if partner_idx is None:
            orphan.append(idx)
        else:
            assert partner_idx != idx, "a tile cannot be its own partner"
            back = _partner_tile(tiles[partner_idx], color_a, color_b)
            assert np.array_equal(back, tile), "pairing must be symmetric"

    assert len(orphan) == 1, "expected exactly one orphan tile"
    return orphan[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Synthesize the absent mirrored color-swapped partner of the unique orphan tile.

    Input structure:
    - The grid is partitioned by separator lines into an odd number of equal-size tiles.
    - Inside every tile, black is the shared background and exactly two foreground colors are used.
    - Most tiles appear in pairs under one composite transform, but one tile's partner is missing.

    Rule:
    - Parse the tile lattice and recover the shared foreground color pair.
    - Two tiles are partners iff one equals the other's horizontal mirror after swapping the two foreground colors.
    - Find the unique tile with no such partner, apply that same composite transform to it, and output the synthesized partner tile alone.

    Key insight:
    - The same "mirror first, then swap the two foreground colors" rule is both the pairing test and the construction rule for the missing answer tile.
    """
    _separator, (color_a, color_b), tiles = _parse_tiled_grid(grid)
    orphan_idx = _find_orphan_index(tiles, color_a, color_b)
    return _partner_tile(tiles[orphan_idx], color_a, color_b)


def _neighbors4(cell: tuple[int, int], h: int, w: int) -> Iterable[tuple[int, int]]:
    r, c = cell
    if r > 0:
        yield (r - 1, c)
    if r + 1 < h:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < w:
        yield (r, c + 1)


def _grow_connected_subset(
    h: int,
    w: int,
    count: int,
    allowed: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    pool = allowed if allowed is not None else {(r, c) for r in range(h) for c in range(w)}
    if count < 1 or len(pool) < count:
        return None

    seed = random.choice(tuple(pool))
    chosen = {seed}
    frontier = {nbr for nbr in _neighbors4(seed, h, w) if nbr in pool}
    while len(chosen) < count:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        for nbr in _neighbors4(cell, h, w):
            if nbr in pool and nbr not in chosen:
                frontier.add(nbr)
    return chosen


def _tile_counts(tile: np.ndarray, color_a: int, color_b: int) -> tuple[int, int]:
    return int(np.count_nonzero(tile == color_a)), int(np.count_nonzero(tile == color_b))


def _make_random_tile(
    h: int,
    w: int,
    color_a: int,
    color_b: int,
    counts: tuple[int, int] | None = None,
) -> np.ndarray | None:
    area = h * w
    for _ in range(120):
        if counts is None:
            occ = random.randint(max(6, area // 2), area - 1)
            count_a = random.randint(2, occ - 2)
            count_b = occ - count_a
        else:
            count_a, count_b = counts
            occ = count_a + count_b
        if min(count_a, count_b) < 2 or occ >= area:
            continue

        occupied = _grow_connected_subset(h, w, occ)
        if occupied is None:
            continue
        a_cells = _grow_connected_subset(h, w, count_a, allowed=occupied)
        if a_cells is None:
            continue

        tile = np.zeros((h, w), dtype=int)
        for r, c in a_cells:
            tile[r, c] = color_a
        for r, c in occupied - a_cells:
            tile[r, c] = color_b

        if np.count_nonzero(tile == 0) == 0:
            continue
        if np.array_equal(tile, np.fliplr(tile)):
            continue
        if np.array_equal(tile, _swap_colors(tile, color_a, color_b)):
            continue
        if np.array_equal(tile, _partner_tile(tile, color_a, color_b)):
            continue
        return tile
    return None


def _has_distractor_pair(tiles: list[np.ndarray], color_a: int, color_b: int) -> bool:
    for i, left in enumerate(tiles):
        counts_left = _tile_counts(left, color_a, color_b)
        for j, right in enumerate(tiles):
            if i >= j:
                continue
            counts_right = _tile_counts(right, color_a, color_b)
            if counts_left != (counts_right[1], counts_right[0]):
                continue
            if np.array_equal(right, _partner_tile(left, color_a, color_b)):
                continue
            return True
    return False


def _semantic_instance_ok(tiles: list[np.ndarray], color_a: int, color_b: int) -> bool:
    sigs = {_signature(tile) for tile in tiles}
    if len(sigs) != len(tiles):
        return False

    orphan_count = 0
    for tile in tiles:
        partner = _partner_tile(tile, color_a, color_b)
        if np.array_equal(tile, partner):
            return False
        if _signature(partner) not in sigs:
            orphan_count += 1
        if _signature(np.fliplr(tile)) in sigs:
            return False
        if _signature(_swap_colors(tile, color_a, color_b)) in sigs:
            return False
    if orphan_count != 1:
        return False
    return _has_distractor_pair(tiles, color_a, color_b)


def _choose_layout() -> tuple[int, int, int, int]:
    while True:
        rows, cols = random.choices(GEN_LAYOUTS, weights=GEN_LAYOUT_WEIGHTS, k=1)[0]
        max_h = (GEN_MAX_GRID_SIDE - (rows + 1)) // rows
        max_w = (GEN_MAX_GRID_SIDE - (cols + 1)) // cols
        if max_h < GEN_MIN_TILE_SIDE or max_w < GEN_MIN_TILE_SIDE:
            continue
        h = random.randint(GEN_MIN_TILE_SIDE, max_h)
        w = random.randint(GEN_MIN_TILE_SIDE, max_w)
        return rows, cols, h, w


def _build_tile_set(h: int, w: int, color_a: int, color_b: int, pair_count: int) -> list[np.ndarray] | None:
    used: set[bytes] = set()
    seeds: list[np.ndarray] = []

    area = h * w
    for _ in range(80):
        occ = random.randint(max(8, area // 2), area - 1)
        count_a = random.randint(2, occ - 2)
        count_b = occ - count_a
        if count_a == count_b:
            continue

        first = _make_random_tile(h, w, color_a, color_b, counts=(count_a, count_b))
        if first is None:
            continue
        second = _make_random_tile(h, w, color_a, color_b, counts=(count_b, count_a))
        if second is None:
            continue

        first_pair = _partner_tile(first, color_a, color_b)
        second_pair = _partner_tile(second, color_a, color_b)
        first_sigs = {
            _signature(first),
            _signature(first_pair),
        }
        second_sigs = {
            _signature(second),
            _signature(second_pair),
        }
        if first_sigs & second_sigs:
            continue
        seeds.extend([first, second])
        used |= first_sigs | second_sigs
        break
    if len(seeds) != 2:
        return None

    while len(seeds) < pair_count:
        tile = _make_random_tile(h, w, color_a, color_b)
        if tile is None:
            return None
        partner = _partner_tile(tile, color_a, color_b)
        sigs = {_signature(tile), _signature(partner)}
        if sigs & used:
            continue
        seeds.append(tile)
        used |= sigs

    orphan = None
    for _ in range(120):
        tile = _make_random_tile(h, w, color_a, color_b)
        if tile is None:
            continue
        partner = _partner_tile(tile, color_a, color_b)
        if _signature(tile) in used or _signature(partner) in used:
            continue
        orphan = tile
        break
    if orphan is None:
        return None

    tiles: list[np.ndarray] = []
    for seed in seeds:
        tiles.append(seed)
        tiles.append(_partner_tile(seed, color_a, color_b))
    tiles.append(orphan)
    return tiles if _semantic_instance_ok(tiles, color_a, color_b) else None


def _render_grid(
    rows: int,
    cols: int,
    h: int,
    w: int,
    tiles: list[np.ndarray],
    separator: int,
) -> np.ndarray:
    grid = np.full((rows * h + rows + 1, cols * w + cols + 1), int(separator), dtype=int)
    idx = 0
    for tile_r in range(rows):
        for tile_c in range(cols):
            top = 1 + tile_r * (h + 1)
            left = 1 + tile_c * (w + 1)
            grid[top : top + h, left : left + w] = tiles[idx]
            idx += 1
    return grid


def _generate_one() -> np.ndarray | None:
    rows, cols, h, w = _choose_layout()
    pair_count = (rows * cols - 1) // 2

    palette = random.sample(range(1, 10), 3)
    color_a, color_b, separator = palette
    tiles = _build_tile_set(h, w, color_a, color_b, pair_count)
    if tiles is None:
        return None

    random.shuffle(tiles)
    grid = _render_grid(rows, cols, h, w, tiles, separator)

    try:
        parsed_separator, parsed_colors, parsed_tiles = _parse_tiled_grid(grid)
        assert parsed_separator == separator
        assert set(parsed_colors) == {color_a, color_b}
        assert _semantic_instance_ok(parsed_tiles, color_a, color_b)
        orphan_idx = _find_orphan_index(parsed_tiles, color_a, color_b)
        expected = _partner_tile(parsed_tiles[orphan_idx], color_a, color_b)
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-partner ARC instance")
