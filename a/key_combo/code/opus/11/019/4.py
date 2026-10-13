import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

LEFT_W = 5
SEPARATOR_W = 1
SPRITE_SIZE = 3
SPRITE_GAP = 1
SPRITE_ROWS = 3
SPRITE_COLS = 3
LATTICE_SIZE = SPRITE_ROWS * SPRITE_SIZE + (SPRITE_ROWS - 1) * SPRITE_GAP
GEN_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _build_pattern_catalog() -> list[np.ndarray]:
    patterns = []
    for area in range(4, 8):
        for cells in itertools.combinations(range(9), area):
            mask = np.zeros((3, 3), dtype=int)
            for idx in cells:
                mask[idx // 3, idx % 3] = 1
            if _bbox(mask.astype(bool)) != (0, 0, 2, 2):
                continue
            if _is_connected(mask.astype(bool)):
                patterns.append(mask)
    return patterns


PATTERN_CATALOG = _build_pattern_catalog()


def _ray_collision_count(left: np.ndarray, r: int, c: int) -> int:
    total = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        while 0 <= rr < left.shape[0] and 0 <= cc < left.shape[1]:
            if int(left[rr, cc]) == GREEN:
                total += 1
                break
            rr += dr
            cc += dc
    return total


def _shift_row_tiles(row_band: np.ndarray, steps: int) -> np.ndarray:
    tiles = []
    for j in range(SPRITE_COLS):
        c0 = j * (SPRITE_SIZE + SPRITE_GAP)
        tiles.append(row_band[:, c0 : c0 + SPRITE_SIZE].copy())
    steps %= SPRITE_COLS
    ordered = tiles[steps:] + tiles[:steps]
    out = np.zeros_like(row_band)
    for j, tile in enumerate(ordered):
        c0 = j * (SPRITE_SIZE + SPRITE_GAP)
        out[:, c0 : c0 + SPRITE_SIZE] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cycle-shift each sprite row left by the obstacle-hit count seen by its aligned marker ray cast.

    Input structure:
    - The grid has a left control area, one full grey separator column, and a right sprite area.
    - The right side contains three rows of blue 3x3 sprites arranged in a 3-by-3 lattice with 1-cell gaps.
    - The left side contains three red markers aligned with the centers of the three sprite rows, plus scattered green obstacle pixels.

    Rule:
    - For each red marker, cast rays up, down, left, and right within the left control area only.
    - Each ray counts one collision if its first hit is a green obstacle; a boundary hit counts nothing.
    - Sum the four directional collisions to get C, then cycle-shift the aligned sprite row left by C positions among its three sprites.
    - Return only the cropped right-side sprite lattice, with blue sprites on a black background.

    Key insight:
    - The obstacle layout in the left control area encodes each row's permutation: ray collisions become the row's cyclic shift amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert separator_cols, "expected a full grey separator column"
    sep = int(separator_cols[0])

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    sprite_bbox = _bbox(right == BLUE)
    assert sprite_bbox is not None, "expected blue sprites on the right side"
    sr0, sc0, sr1, sc1 = sprite_bbox
    sprite = right[sr0 : sr1 + 1, sc0 : sc1 + 1]
    assert sprite.shape == (LATTICE_SIZE, LATTICE_SIZE)

    markers = {int(r): int(c) for r, c in np.argwhere(left == RED)}
    assert len(markers) == 3, "expected exactly three markers"

    output = np.zeros((LATTICE_SIZE, LATTICE_SIZE), dtype=int)
    for i in range(SPRITE_ROWS):
        row_center = int(sr0 + i * (SPRITE_SIZE + SPRITE_GAP) + 1)
        assert row_center in markers, "marker rows must align with sprite row centers"
        hit_count = _ray_collision_count(left, row_center, markers[row_center])
        r0 = i * (SPRITE_SIZE + SPRITE_GAP)
        row_band = sprite[r0 : r0 + SPRITE_SIZE, :]
        output[r0 : r0 + SPRITE_SIZE, :] = _shift_row_tiles(row_band, hit_count)
    return output


def _safe_ray_slots(
    row: int,
    col: int,
    direction: str,
    height: int,
    marker_rows: set[int],
    marker_cols: set[int],
) -> list[tuple[int, int]]:
    if direction == "U":
        return [(r, col) for r in range(0, row) if r not in marker_rows]
    if direction == "D":
        return [(r, col) for r in range(row + 1, height) if r not in marker_rows]
    if direction == "L":
        return [(row, c) for c in range(0, col) if c not in marker_cols]
    if direction == "R":
        return [(row, c) for c in range(col + 1, LEFT_W) if c not in marker_cols]
    raise ValueError(direction)


def _build_sprite_lattice(patterns: list[np.ndarray]) -> np.ndarray:
    lattice = np.zeros((LATTICE_SIZE, LATTICE_SIZE), dtype=int)
    k = 0
    for i in range(SPRITE_ROWS):
        for j in range(SPRITE_COLS):
            r0 = i * (SPRITE_SIZE + SPRITE_GAP)
            c0 = j * (SPRITE_SIZE + SPRITE_GAP)
            lattice[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE] = patterns[k] * BLUE
            k += 1
    return lattice


def _choose_actual_counts() -> list[int]:
    counts = [random.choice((0, 3)), random.choice((1, 4)), 2]
    random.shuffle(counts)
    return counts


def _gen_attempt() -> np.ndarray | None:
    top_pad = random.randint(1, 4)
    bottom_pad = random.randint(1, 4)
    right_left_pad = random.randint(0, 2)
    right_right_pad = random.randint(0, 2)

    height = top_pad + LATTICE_SIZE + bottom_pad
    width = LEFT_W + SEPARATOR_W + right_left_pad + LATTICE_SIZE + right_right_pad
    if max(height, width) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:, LEFT_W] = GREY

    patterns = random.sample(PATTERN_CATALOG, 9)
    sprite_lattice = _build_sprite_lattice(patterns)
    sprite_r0 = top_pad
    sprite_c0 = LEFT_W + SEPARATOR_W + right_left_pad
    grid[sprite_r0 : sprite_r0 + LATTICE_SIZE, sprite_c0 : sprite_c0 + LATTICE_SIZE] = sprite_lattice

    marker_rows = [sprite_r0 + 1 + 4 * i for i in range(SPRITE_ROWS)]
    marker_cols = random.sample([1, 2, 3], 3)
    marker_row_set = set(marker_rows)
    marker_col_set = set(marker_cols)
    for r, c in zip(marker_rows, marker_cols):
        grid[r, c] = RED

    desired_counts = _choose_actual_counts()
    for r, c, count in zip(marker_rows, marker_cols, desired_counts):
        directions = random.sample(["U", "D", "L", "R"], count)
        for direction in directions:
            slots = _safe_ray_slots(r, c, direction, height, marker_row_set, marker_col_set)
            if not slots:
                return None
            rr, cc = random.choice(slots)
            if grid[rr, cc] != BLACK:
                return None
            grid[rr, cc] = GREEN

    base_obstacles = int(np.count_nonzero(grid == GREEN))
    if base_obstacles > 10:
        return None
    target_total = random.randint(max(5, base_obstacles), 10)
    decoy_slots = [
        (r, c)
        for r in range(height)
        for c in range(LEFT_W)
        if r not in marker_row_set and c not in marker_col_set and grid[r, c] == BLACK
    ]
    if len(decoy_slots) < target_total - base_obstacles:
        return None
    for rr, cc in random.sample(decoy_slots, target_total - base_obstacles):
        grid[rr, cc] = GREEN

    counts_seen = [_ray_collision_count(grid[:, :LEFT_W], r, c) for r, c in zip(marker_rows, marker_cols)]
    if counts_seen != desired_counts:
        return None
    if sorted(c % 3 for c in counts_seen) != [0, 1, 2]:
        return None

    out = main(grid)
    if out.shape != (LATTICE_SIZE, LATTICE_SIZE):
        return None
    if np.array_equal(out, sprite_lattice):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
