from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, get_bg


Coord = tuple[int, int]
SourceSpec = dict[str, object]


def _candidate_n(size: int) -> int | None:
    for n in range(2, 6):
        if n * n + n - 1 == size:
            return n
    return None


def _separator_indices(n: int) -> list[int]:
    return [k * (n + 1) - 1 for k in range(1, n)]


def _tile_origin(tile_index: int, n: int) -> int:
    return tile_index * (n + 1)


def _tile_bounds(tile_r: int, tile_c: int, n: int) -> tuple[int, int, int, int]:
    r0 = _tile_origin(tile_r, n)
    c0 = _tile_origin(tile_c, n)
    return r0, c0, r0 + n, c0 + n


def _tile_view(grid: np.ndarray, tile_r: int, tile_c: int, n: int) -> np.ndarray:
    r0, c0, r1, c1 = _tile_bounds(tile_r, tile_c, n)
    return grid[r0:r1, c0:c1]


def _infer_layout(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square grid"

    n = _candidate_n(int(g.shape[0]))
    assert n is not None, "grid side does not match n*n+n-1 for any valid n"

    bg = int(get_bg(g))
    for idx in _separator_indices(n):
        assert bool(np.all(g[idx, :] == bg)), "separator rows must be background"
        assert bool(np.all(g[:, idx] == bg)), "separator cols must be background"
    return bg, n


def _extract_sources(grid: np.ndarray) -> tuple[int, int, list[SourceSpec]]:
    g = np.asarray(grid, dtype=int)
    bg, n = _infer_layout(g)

    sources: list[SourceSpec] = []
    for tile_r in range(n):
        for tile_c in range(n):
            tile = _tile_view(g, tile_r, tile_c, n)
            fg_mask = tile != bg
            if not np.any(fg_mask):
                continue

            hole_coords = np.argwhere(tile == bg)
            assert hole_coords.shape[0] == 1, "each non-empty tile must contain exactly one hole"
            assert int(fg_mask.sum()) == n * n - 1, "non-empty tiles must be solid except for one hole"

            colors = sorted(set(map(int, tile[fg_mask].tolist())))
            assert len(colors) == 1, "each non-empty tile must be monochromatic"

            hole_r, hole_c = map(int, hole_coords[0])
            sources.append(
                {
                    "source_tile": (tile_r, tile_c),
                    "target_tile": (hole_r, hole_c),
                    "color": int(colors[0]),
                }
            )

    assert sources, "expected at least one non-empty tile"
    targets = [tuple(spec["target_tile"]) for spec in sources]
    assert len(set(targets)) == len(targets), "hole coordinates must point to distinct targets"
    return bg, n, sources


def _build_expected_from_specs(bg: int, n: int, sources: list[SourceSpec]) -> np.ndarray:
    side = n * n + n - 1
    out = np.full((side, side), bg, dtype=int)
    for spec in sources:
        target_r, target_c = spec["target_tile"]
        color = int(spec["color"])
        tile = _tile_view(out, int(target_r), int(target_c), n)
        tile[:, :] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each holed source tile's color block to the destination tile encoded by the hole position.

    Input structure:
    - The grid is a square tile array split by one-cell background separator rows and columns.
    - Its side length is n*n+n-1, so it contains n rows by n columns of n×n tiles.
    - Each non-empty tile is a solid monochrome n×n block with exactly one background hole at local coordinate (r,c).
    - Empty tiles and all separator rows/columns are entirely background.

    Rule:
    - Read each hole's local coordinate (r,c) as a tile-array destination address.
    - Send the tile's color to array tile (r,c), filling that destination tile completely with no hole.
    - Clear the source tile to background unless it is also the addressed destination, in which case it becomes fully filled.
    - Distinct non-empty tiles always point to distinct destinations, so each output tile is written at most once.

    Key insight:
    - The hole marks where the whole colored tile should be forwarded, not merely which cell inside the same tile should be repaired.
    """
    g = np.asarray(grid, dtype=int)
    bg, n, sources = _extract_sources(g)
    return _build_expected_from_specs(bg, n, sources)


def _paint_source_tile(grid: np.ndarray, tile_r: int, tile_c: int, n: int, color: int, target: Coord, bg: int) -> None:
    tile = _tile_view(grid, tile_r, tile_c, n)
    tile[:, :] = int(color)
    hole_r, hole_c = map(int, target)
    tile[hole_r, hole_c] = bg


def _sample_mapping(n: int) -> list[SourceSpec]:
    all_tiles = [(r, c) for r in range(n) for c in range(n)]
    max_sources = min(len(all_tiles) - 1, 6)
    count = random.randint(2, max_sources)
    source_tiles = random.sample(all_tiles, count)
    source_set = set(source_tiles)

    mapping: dict[Coord, Coord] = {}
    used_targets: set[Coord] = set()
    remaining = source_tiles[:]
    random.shuffle(remaining)

    if len(remaining) >= 3 and random.random() < 0.65:
        self_count = random.randint(1, min(2, len(remaining) - 2))
        for src in remaining[:self_count]:
            mapping[src] = src
            used_targets.add(src)
        remaining = remaining[self_count:]

    if len(remaining) >= 2 and random.random() < 0.7:
        cycle_len = random.randint(2, min(3, len(remaining)))
        cycle_nodes = random.sample(remaining, cycle_len)
        for idx, src in enumerate(cycle_nodes):
            dst = cycle_nodes[(idx + 1) % cycle_len]
            mapping[src] = dst
            used_targets.add(dst)
            remaining.remove(src)

    empty_targets = [tile for tile in all_tiles if tile not in source_set and tile not in used_targets]
    random.shuffle(empty_targets)
    fallback_targets = [tile for tile in all_tiles if tile not in used_targets]
    random.shuffle(fallback_targets)

    for src in remaining:
        choices = [tile for tile in empty_targets if tile != src]
        if choices:
            dst = choices.pop()
            empty_targets.remove(dst)
        else:
            choices = [tile for tile in fallback_targets if tile != src]
            assert choices, "expected at least one unused target for every remaining source"
            dst = choices.pop()
        mapping[src] = dst
        used_targets.add(dst)
        if dst in fallback_targets:
            fallback_targets.remove(dst)

    non_self = [src for src, dst in mapping.items() if src != dst]
    if not non_self:
        src = source_tiles[0]
        candidates = [tile for tile in all_tiles if tile != src and tile not in used_targets]
        if not candidates:
            candidates = [tile for tile in all_tiles if tile != src]
        mapping[src] = random.choice(candidates)

    colors = random.sample([c for c in Color.NOT_BLACK], len(source_tiles))
    random.shuffle(colors)
    specs: list[SourceSpec] = []
    for src, color in zip(source_tiles, colors):
        specs.append(
            {
                "source_tile": src,
                "target_tile": mapping[src],
                "color": int(color),
            }
        )
    return specs


def _build_input(bg: int, n: int, specs: list[SourceSpec]) -> np.ndarray:
    side = n * n + n - 1
    grid = np.full((side, side), bg, dtype=int)
    for spec in specs:
        src_r, src_c = spec["source_tile"]
        color = int(spec["color"])
        target = tuple(spec["target_tile"])
        _paint_source_tile(grid, int(src_r), int(src_c), n, color, target, bg)
    return grid


def _mapping_features(specs: list[SourceSpec]) -> tuple[bool, bool, bool]:
    source_set = {tuple(spec["source_tile"]) for spec in specs}
    has_self = False
    has_occupied_target = False
    has_empty_target = False
    for spec in specs:
        src = tuple(spec["source_tile"])
        dst = tuple(spec["target_tile"])
        if src == dst:
            has_self = True
        elif dst in source_set:
            has_occupied_target = True
        else:
            has_empty_target = True
    return has_self, has_occupied_target, has_empty_target


def _valid_generated_instance(grid: np.ndarray, specs: list[SourceSpec]) -> bool:
    try:
        bg, n, parsed = _extract_sources(grid)
    except AssertionError:
        return False
    if len(parsed) != len(specs):
        return False
    expected = _build_expected_from_specs(bg, n, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    target_colors = {tuple(spec["target_tile"]): int(spec["color"]) for spec in specs}
    for tile_r in range(n):
        for tile_c in range(n):
            tile = _tile_view(actual, tile_r, tile_c, n)
            if (tile_r, tile_c) in target_colors:
                if not bool(np.all(tile == target_colors[(tile_r, tile_c)])):
                    return False
            elif not bool(np.all(tile == bg)):
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        n = random.randint(2, 5)
        specs = _sample_mapping(n)
        bg_choices = [c for c in range(10) if c not in {int(spec["color"]) for spec in specs}]
        bg = int(random.choice(bg_choices))
        grid = _build_input(bg, n, specs)

        has_self, has_occupied_target, has_empty_target = _mapping_features(specs)
        if len(specs) < 2:
            continue
        if not has_empty_target and n >= 3:
            continue
        if not has_self and random.random() < 0.45:
            continue
        if not has_occupied_target and random.random() < 0.35:
            continue
        if _valid_generated_instance(grid, specs):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
