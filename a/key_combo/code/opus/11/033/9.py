from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
TILE_SIZE = 4
GRID_SIZE = TILE_SIZE * 2 + 1
REGION_ORDER = (
    ("lu", 0, 0, 0),
    ("ru", 0, 1, 1),
    ("ld", 1, 0, 3),
    ("rd", 1, 1, 2),
)
PAYLOAD_COLORS = (1, 2, 3, 6, 8, 9)
BASE_AREA_RANGE = (5, 8)
GEN_ATTEMPTS = 500


def _bands_from_separator(separators: list[int], size: int) -> list[tuple[int, int]]:
    assert len(separators) == 1
    sep = separators[0]
    bands = [(0, sep), (sep + 1, size)]
    assert all(hi - lo == TILE_SIZE for lo, hi in bands)
    return bands


def _rotate_cw(tile: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(tile, -int(turns))


def _aligned_to_base(patch: np.ndarray, turns_cw: int) -> np.ndarray:
    return np.rot90(patch, int(turns_cw))


def _coords_of(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _unique_rotations(tile: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = _rotate_cw(tile, k)
        if not any(np.array_equal(rotated, old) for old in variants):
            variants.append(rotated)
    return variants


def _is_valid_base_tile(tile: np.ndarray) -> bool:
    if tile.shape != (TILE_SIZE, TILE_SIZE):
        return False
    mask = tile != BACKGROUND
    area = int(mask.sum())
    if area < BASE_AREA_RANGE[0] or area > BASE_AREA_RANGE[1]:
        return False
    colors = {int(v) for v in tile[mask]}
    if len(colors) < 2 or len(colors) > 3:
        return False
    if not _is_connected(mask):
        return False
    if len(_unique_rotations(tile)) != 4:
        return False
    return True


def _parse_regions(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[str, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape == (GRID_SIZE, GRID_SIZE)

    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR)]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR)]
    assert len(row_seps) == len(col_seps) == 1

    row_bands = _bands_from_separator(row_seps, g.shape[0])
    col_bands = _bands_from_separator(col_seps, g.shape[1])

    regions: dict[str, np.ndarray] = {}
    for name, br, bc, _rot in REGION_ORDER:
        r0, r1 = row_bands[br]
        c0, c1 = col_bands[bc]
        patch = g[r0:r1, c0:c1]
        assert patch.shape == (TILE_SIZE, TILE_SIZE)
        assert not np.any(patch == SEPARATOR)
        regions[name] = patch
    return row_bands, col_bands, regions


def _reconstruct_base(regions: dict[str, np.ndarray]) -> np.ndarray:
    base = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    for name, _br, _bc, rot in REGION_ORDER:
        aligned = _aligned_to_base(regions[name], rot)
        for r in range(TILE_SIZE):
            for c in range(TILE_SIZE):
                color = int(aligned[r, c])
                if color == BACKGROUND:
                    continue
                if base[r, c] == BACKGROUND:
                    base[r, c] = color
                else:
                    assert base[r, c] == color, "cross-region evidence must agree"

    assert _is_valid_base_tile(base), "merged base tile must satisfy the puzzle constraints"

    for name, _br, _bc, rot in REGION_ORDER:
        full_region = _rotate_cw(base, rot)
        patch = regions[name]
        missing = (full_region != BACKGROUND) & (patch == BACKGROUND)
        visible = patch != BACKGROUND
        assert np.all(full_region[visible] == patch[visible]), "visible pixels must already match the rotated tile"
        assert 2 <= int(missing.sum()) <= 3, "each region must erase 2-3 tile pixels"
    return base


def main(grid: np.ndarray) -> np.ndarray:
    """Complete four partially erased rotated tile copies by cross-referencing them in a shared base orientation.

    Input structure
    - A 9x9 grid has black background and one full grey separator row plus one full grey separator column, splitting the board into four 4x4 regions.
    - The four regions contain different rotations of the same colored 4x4 tile on black, with some non-black pixels erased to black in each region.
    - The tile uses 2-3 foreground colors, has 5-8 non-black cells, and its non-black mask is 4-connected and not invariant under any non-trivial rotation.

    Rule
    - Interpret the quadrants with fixed orientations: upper-left = 0 degrees, upper-right = 90 degrees clockwise, lower-left = 270 degrees clockwise, lower-right = 180 degrees.
    - Reverse-rotate each region back into the common base orientation and merge all visible non-black evidence to reconstruct the complete base tile.
    - Rotate that reconstructed tile forward into each quadrant's orientation and fill only the erased black gaps, while preserving existing non-black pixels and the grey separators.

    Key insight
    - No quadrant is complete by itself: the missing pixels line up only after undoing each quadrant's rotation, so the solution requires a shared coordinate system across all four regions.
    """
    g = np.asarray(grid, dtype=int)
    row_bands, col_bands, regions = _parse_regions(g)
    base = _reconstruct_base(regions)

    out = g.copy()
    for name, br, bc, rot in REGION_ORDER:
        rotated = _rotate_cw(base, rot)
        r0, r1 = row_bands[br]
        c0, c1 = col_bands[bc]
        patch = out[r0:r1, c0:c1]
        fill_mask = (patch == BACKGROUND) & (rotated != BACKGROUND)
        patch[fill_mask] = rotated[fill_mask]
    return out


def _random_connected_mask() -> np.ndarray | None:
    area = random.randint(*BASE_AREA_RANGE)
    cells = {(random.randrange(TILE_SIZE), random.randrange(TILE_SIZE))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < TILE_SIZE and 0 <= nc < TILE_SIZE and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(sorted(frontier)))

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 or len(cols) < 2:
        return None

    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if not _is_connected(mask):
        return None
    return mask


def _color_base_mask(mask: np.ndarray) -> np.ndarray:
    cells = _coords_of(mask)
    color_count = 2 if random.random() < 0.55 else 3
    color_count = min(color_count, len(cells))
    palette = random.sample(PAYLOAD_COLORS, color_count)
    assigned = list(palette)
    assigned.extend(random.choice(palette) for _ in range(len(cells) - color_count))
    random.shuffle(assigned)

    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    for (r, c), color in zip(cells, assigned):
        tile[r, c] = int(color)
    return tile


def _make_base_tile() -> np.ndarray | None:
    mask = _random_connected_mask()
    if mask is None:
        return None
    tile = _color_base_mask(mask)
    if not _is_valid_base_tile(tile):
        return None
    return tile


def _region_removal_sets(base: np.ndarray) -> list[set[tuple[int, int]]]:
    cells = _coords_of(base != BACKGROUND)
    witnesses = random.sample(cells, 4)
    removals: list[set[tuple[int, int]]] = []
    for i in range(4):
        removals.append({witnesses[j] for j in range(4) if j != i})
    return removals


def _apply_removals(base: np.ndarray, removals: list[set[tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[TILE_SIZE, :] = SEPARATOR
    grid[:, TILE_SIZE] = SEPARATOR

    for idx, (_name, br, bc, rot) in enumerate(REGION_ORDER):
        tile = _rotate_cw(base, rot).copy()
        for r, c in removals[idx]:
            rotated_r, rotated_c = _rotate_coord_cw(r, c, rot, TILE_SIZE)
            tile[rotated_r, rotated_c] = BACKGROUND
        r0 = br * (TILE_SIZE + 1)
        c0 = bc * (TILE_SIZE + 1)
        grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = tile
    return grid


def _rotate_coord_cw(r: int, c: int, turns: int, size: int) -> tuple[int, int]:
    turns %= 4
    if turns == 0:
        return r, c
    if turns == 1:
        return c, size - 1 - r
    if turns == 2:
        return size - 1 - r, size - 1 - c
    return size - 1 - c, r


def _aligned_visible_sets(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    _, _, regions = _parse_regions(grid)
    visible_sets: list[set[tuple[int, int]]] = []
    for _name, _br, _bc, rot in REGION_ORDER:
        aligned = _aligned_to_base(regions[_name], rot)
        visible_sets.append(set(_coords_of(aligned != BACKGROUND)))
    return visible_sets


def _check_non_degenerate(inp: np.ndarray, base: np.ndarray, removals: list[set[tuple[int, int]]]) -> None:
    out = main(inp)
    expected = _apply_removals(base, [set() for _ in REGION_ORDER])
    assert np.array_equal(out, expected), "solver must reconstruct the complete four-region board"
    assert not np.array_equal(inp, out), "input must require filling"

    visible_sets = _aligned_visible_sets(inp)
    all_base_cells = set(_coords_of(base != BACKGROUND))
    assert set().union(*visible_sets) == all_base_cells, "all tile pixels must be recoverable across regions"
    assert len(removals) == 4
    assert len({frozenset(s) for s in removals}) == 4, "each region needs a distinct erasure pattern"
    assert all(len(s) == 3 for s in removals), "each region intentionally erases three witnesses"

    for visible in visible_sets:
        assert visible < all_base_cells, "no single region may already reveal the whole tile"

    for region_index in range(4):
        union_without = set()
        for other_index, visible in enumerate(visible_sets):
            if other_index != region_index:
                union_without |= visible
        assert union_without < all_base_cells, "every region must contribute indispensable information"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        base = _make_base_tile()
        if base is None:
            continue

        removals = _region_removal_sets(base)
        inp = _apply_removals(base, removals)

        try:
            _check_non_degenerate(inp, base, removals)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a valid input for puzzle 9")
