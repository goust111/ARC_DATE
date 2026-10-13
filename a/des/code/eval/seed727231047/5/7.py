from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 7
ROOM_SIZE = 5
ROOM_STEP = ROOM_SIZE + 1
PATTERN_SIZE = 3
GEN_RETRIES = 400
COLOR_POOL = [1, 2, 3, 4, 5, 6, 8, 9]

_QUADRANT_CHOICES = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 3), (0, 4), (1, 3), (1, 4)),
    ((3, 3), (3, 4), (4, 3), (4, 4)),
    ((3, 0), (3, 1), (4, 0), (4, 1)),
)


def _mask(rows: tuple[str, str, str]) -> np.ndarray:
    return np.array([[1 if ch == "1" else 0 for ch in row] for row in rows], dtype=int)


PATTERN_CATALOG = [
    _mask(("100", "110", "011")),
    _mask(("111", "100", "010")),
    _mask(("010", "111", "001")),
    _mask(("101", "110", "010")),
    _mask(("110", "011", "001")),
    _mask(("001", "111", "010")),
    _mask(("010", "110", "101")),
    _mask(("110", "101", "001")),
    _mask(("100", "111", "001")),
    _mask(("011", "110", "100")),
    _mask(("101", "011", "010")),
    _mask(("010", "101", "110")),
]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _room_origins(grid: np.ndarray, wall: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    origins: list[tuple[int, int]] = []
    for r0 in range(1, h - ROOM_SIZE):
        for c0 in range(1, w - ROOM_SIZE):
            if not np.all(grid[r0 - 1, c0 - 1 : c0 + ROOM_SIZE + 1] == wall):
                continue
            if not np.all(grid[r0 + ROOM_SIZE, c0 - 1 : c0 + ROOM_SIZE + 1] == wall):
                continue
            if not np.all(grid[r0 : r0 + ROOM_SIZE, c0 - 1] == wall):
                continue
            if not np.all(grid[r0 : r0 + ROOM_SIZE, c0 + ROOM_SIZE] == wall):
                continue
            if np.any(grid[r0 : r0 + ROOM_SIZE, c0 : c0 + ROOM_SIZE] == wall):
                continue
            origins.append((r0, c0))
    assert origins, "expected at least one 5x5 room"
    return sorted(origins)


def _library_bbox(grid: np.ndarray, wall: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for coords in _components(grid == wall):
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1) <= ROOM_SIZE + 2 and (c1 - c0 + 1) <= ROOM_SIZE + 2:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == wall):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == wall):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == wall):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == wall):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == wall):
            continue
        candidates.append((r1 + c1, r1, c1, r0, c0))
    assert candidates, "failed to locate the isolated pattern library"
    _, r1, c1, r0, c0 = max(candidates)
    return int(r0), int(c0), int(r1), int(c1)


def _library_patterns(
    grid: np.ndarray, library_bbox: tuple[int, int, int, int], bg: int, wall: int
) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = library_bbox
    library = grid[r0 + 1 : r1, c0 + 1 : c1]
    patterns: dict[int, np.ndarray] = {}
    for color in np.unique(library):
        color = int(color)
        if color in {bg, wall}:
            continue
        coords = np.argwhere(library == color)
        pr0, pc0 = coords.min(axis=0)
        pr1, pc1 = coords.max(axis=0)
        patch = library[int(pr0) : int(pr1) + 1, int(pc0) : int(pc1) + 1]
        assert patch.shape == (PATTERN_SIZE, PATTERN_SIZE), "library entries must be 3x3"
        assert set(map(int, np.unique(patch))).issubset({bg, color})
        patterns[color] = patch.copy()
    assert patterns, "expected at least one library pattern"
    return patterns


def _rotation_k(seed_r: int, seed_c: int) -> int:
    if seed_r <= 1 and seed_c <= 1:
        return 0
    if seed_r <= 1 and seed_c >= 3:
        return 3
    if seed_r >= 3 and seed_c >= 3:
        return 2
    if seed_r >= 3 and seed_c <= 1:
        return 1
    raise AssertionError("seed must lie in one of the four room quadrants")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select a 3x3 library pattern by seed color, rotate it by the seed quadrant, and stamp it into each room center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    colors, counts = np.unique(g[g != bg], return_counts=True)
    assert counts.size > 0, "expected foreground structure"
    wall = int(colors[int(np.argmax(counts))])

    room_origins = _room_origins(g, wall)
    library_bbox = _library_bbox(g, wall)
    patterns = _library_patterns(g, library_bbox, bg, wall)

    out = g.copy()
    for r0, c0 in room_origins:
        room = g[r0 : r0 + ROOM_SIZE, c0 : c0 + ROOM_SIZE]
        seeds = np.argwhere((room != bg) & (room != wall))
        assert len(seeds) == 1, "each room must contain exactly one seed"
        seed_r, seed_c = map(int, seeds[0])
        color = int(room[seed_r, seed_c])
        assert color in patterns, "every seed color must exist in the library"

        rotated = np.rot90(patterns[color], k=_rotation_k(seed_r, seed_c))
        center = out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4]
        mask = rotated != bg
        center[mask] = rotated[mask]

    return out


def _colored_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask != 0, int(color), BG).astype(int)


def _room_assignments(rng: random.Random, count: int, values: list[int]) -> list[int]:
    out = values.copy()
    while len(out) < count:
        out.append(int(rng.choice(values)))
    rng.shuffle(out)
    return out


def _validate_grid(grid: np.ndarray, expected_rooms: int, expected_colors: set[int]) -> bool:
    if _most_common_color(grid) != BG:
        return False
    colors, counts = np.unique(grid[grid != BG], return_counts=True)
    if counts.size == 0:
        return False
    if int(colors[int(np.argmax(counts))]) != WALL:
        return False
    try:
        room_origins = _room_origins(grid, WALL)
        if len(room_origins) != expected_rooms:
            return False
        library_bbox = _library_bbox(grid, WALL)
        patterns = _library_patterns(grid, library_bbox, BG, WALL)
        if set(patterns) != set(expected_colors):
            return False
        for r0, c0 in room_origins:
            room = grid[r0 : r0 + ROOM_SIZE, c0 : c0 + ROOM_SIZE]
            if np.count_nonzero((room != BG) & (room != WALL)) != 1:
                return False
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.randint(2, 4)
    room_cols = rng.randint(2, 4)
    room_count = room_rows * room_cols
    color_count = rng.randint(2, min(5, room_count))
    colors = rng.sample(COLOR_POOL, color_count)

    room_h = room_rows * ROOM_STEP + 1
    room_w = room_cols * ROOM_STEP + 1

    lib_cols = rng.randint(1, min(3, color_count))
    lib_rows = (color_count + lib_cols - 1) // lib_cols
    pad_r = rng.randint(1, 2)
    pad_c = rng.randint(1, 2)
    inner_h = 2 * pad_r + lib_rows * PATTERN_SIZE + max(0, lib_rows - 1)
    inner_w = 2 * pad_c + lib_cols * PATTERN_SIZE + max(0, lib_cols - 1)
    lib_h = inner_h + 2
    lib_w = inner_w + 2

    top_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    gap_v = rng.randint(2, 5)
    gap_h = rng.randint(2, 5)

    grid_h = top_margin + room_h + gap_v + lib_h + bottom_margin
    grid_w = left_margin + room_w + gap_h + lib_w + right_margin
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    room_top = top_margin
    room_left = left_margin
    library_top = grid_h - bottom_margin - lib_h
    library_left = grid_w - right_margin - lib_w

    for k in range(room_rows + 1):
        grid[room_top + k * ROOM_STEP, room_left : room_left + room_w] = WALL
    for k in range(room_cols + 1):
        grid[room_top : room_top + room_h, room_left + k * ROOM_STEP] = WALL

    grid[library_top, library_left : library_left + lib_w] = WALL
    grid[library_top + lib_h - 1, library_left : library_left + lib_w] = WALL
    grid[library_top : library_top + lib_h, library_left] = WALL
    grid[library_top : library_top + lib_h, library_left + lib_w - 1] = WALL

    quadrant_ids = _room_assignments(rng, room_count, list(range(4)))
    room_colors = _room_assignments(rng, room_count, [int(color) for color in colors])

    room_index = 0
    for rr in range(room_rows):
        for cc in range(room_cols):
            r0 = room_top + 1 + rr * ROOM_STEP
            c0 = room_left + 1 + cc * ROOM_STEP
            quadrant = quadrant_ids[room_index]
            seed_dr, seed_dc = rng.choice(_QUADRANT_CHOICES[quadrant])
            grid[r0 + seed_dr, c0 + seed_dc] = room_colors[room_index]
            room_index += 1

    chosen_masks = rng.sample(PATTERN_CATALOG, color_count)
    pattern_map: dict[int, np.ndarray] = {}
    for color, base_mask in zip(colors, chosen_masks, strict=False):
        rotated_mask = np.rot90(base_mask, k=rng.randrange(4))
        pattern_map[int(color)] = _colored_patch(rotated_mask, int(color))

    library_colors = [int(color) for color in colors]
    rng.shuffle(library_colors)
    for idx, color in enumerate(library_colors):
        pr = idx // lib_cols
        pc = idx % lib_cols
        top = library_top + 1 + pad_r + pr * (PATTERN_SIZE + 1)
        left = library_left + 1 + pad_c + pc * (PATTERN_SIZE + 1)
        grid[top : top + PATTERN_SIZE, left : left + PATTERN_SIZE] = pattern_map[color]

    if not _validate_grid(grid, room_count, set(colors)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
