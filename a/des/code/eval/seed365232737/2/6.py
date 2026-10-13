from __future__ import annotations

import random

import numpy as np


BG = 0
HAND_COLOR = 2
RING_COLOR = 5
TICK_COLOR = 7
SPRITE_COLORS = (1, 3, 4, 6, 8, 9)

DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
K_BY_DIRECTION = {
    (-1, 0): 1,  # up
    (0, 1): 2,   # right
    (1, 0): 3,   # down
    (0, -1): 4,  # left
}

CLOCK_TEMPLATES = {
    7: (
        ".#####.",
        "##...##",
        "#.....#",
        "#.....#",
        "#.....#",
        "##...##",
        ".#####.",
    ),
    9: (
        ".#######.",
        "##.....##",
        "#.......#",
        "#.......#",
        "#.......#",
        "#.......#",
        "#.......#",
        "##.....##",
        ".#######.",
    ),
}

_RNG = random.Random()
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120
SHAPE_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _crop_component(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, r1, c0, c1 = _bbox(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    shifted = coords - np.array([r0, c0], dtype=int)
    out[shifted[:, 0], shifted[:, 1]] = grid[coords[:, 0], coords[:, 1]]
    return out


def _find_clock_and_sprite(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    comps = _components(grid != BG)
    assert len(comps) >= 2, "expected a clock and a sprite"

    clock_idx = None
    for idx, comp in enumerate(comps):
        colors = {int(grid[r, c]) for r, c in comp}
        if {HAND_COLOR, RING_COLOR, TICK_COLOR} <= colors:
            orange_count = sum(int(grid[r, c] == TICK_COLOR) for r, c in comp)
            if orange_count == 4:
                assert clock_idx is None, "clock must be unique"
                clock_idx = idx
    assert clock_idx is not None, "clock component not found"

    sprite_comps = [comp for idx, comp in enumerate(comps) if idx != clock_idx]
    assert len(sprite_comps) == 1, "expected exactly one sprite component"
    sprite = sprite_comps[0]
    sprite_colors = {int(grid[r, c]) for r, c in sprite}
    assert sprite_colors.isdisjoint({HAND_COLOR, RING_COLOR, TICK_COLOR}), "sprite must avoid control colors"
    return comps[clock_idx], sprite


def _read_k(grid: np.ndarray, clock_coords: np.ndarray) -> int:
    clock_mask = _mask_from_coords(grid.shape, clock_coords)
    tick_coords = np.argwhere(clock_mask & (grid == TICK_COLOR))
    assert tick_coords.shape[0] == 4, "clock must have four tick marks"

    top = tick_coords[np.argmin(tick_coords[:, 0])]
    bottom = tick_coords[np.argmax(tick_coords[:, 0])]
    left = tick_coords[np.argmin(tick_coords[:, 1])]
    right = tick_coords[np.argmax(tick_coords[:, 1])]

    center_r = (int(top[0]) + int(bottom[0])) // 2
    center_c = (int(left[1]) + int(right[1])) // 2
    center = np.array([center_r, center_c], dtype=int)

    assert int(top[1]) == center_c and int(bottom[1]) == center_c
    assert int(left[0]) == center_r and int(right[0]) == center_r
    assert grid[center_r, center_c] == HAND_COLOR, "clock center must be red"

    red_coords = np.argwhere(clock_mask & (grid == HAND_COLOR))
    arm = red_coords[np.any(red_coords != center, axis=1)]
    assert arm.shape[0] >= 1, "clock hand must extend away from the center"

    if np.all(arm[:, 1] == center_c):
        direction = (-1, 0) if int(arm[:, 0].min()) < center_r else (1, 0)
    else:
        assert np.all(arm[:, 0] == center_r), "clock hand must be axis-aligned"
        direction = (0, -1) if int(arm[:, 1].min()) < center_c else (0, 1)
    return K_BY_DIRECTION[direction]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read K from the clock hand direction, rotate the sprite clockwise K times,
    and tile the rotated sprite into a tight K x K array.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    clock_coords, sprite_coords = _find_clock_and_sprite(g)
    k = _read_k(g, clock_coords)

    sprite = _crop_component(g, sprite_coords)
    rotated = np.rot90(sprite, -k)
    return np.tile(rotated, (k, k)).astype(int)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, r1, c0, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _has_four_orientations(mask: np.ndarray) -> bool:
    seen = set()
    cur = mask.copy()
    for _ in range(4):
        norm = _normalize_mask(cur)
        seen.add(norm.tobytes() + str(norm.shape).encode())
        cur = np.rot90(cur, -1)
    return len(seen) == 4


def _random_sprite_mask() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = _RNG.randint(3, 4)
        box_w = _RNG.randint(3, 4)
        target_cells = _RNG.randint(4, min(6, box_h * box_w - 1))

        start = (_RNG.randrange(box_h), _RNG.randrange(box_w))
        cells = {start}
        while len(cells) < target_cells:
            r, c = _RNG.choice(tuple(cells))
            dr, dc = _RNG.choice(DIRS4)
            nr = r + dr
            nc = c + dc
            if 0 <= nr < box_h and 0 <= nc < box_w:
                cells.add((nr, nc))

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _normalize_mask(mask)

        h, w = mask.shape
        if max(h, w) < 3:
            continue
        if not _has_four_orientations(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric sprite")


def _paint_clock(size: int, direction: tuple[int, int]) -> np.ndarray:
    rows = CLOCK_TEMPLATES[size]
    clock = np.zeros((size, size), dtype=int)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                clock[r, c] = RING_COLOR

    center = size // 2
    ticks = {
        (-1, 0): (0, center),
        (0, 1): (center, size - 1),
        (1, 0): (size - 1, center),
        (0, -1): (center, 0),
    }
    for pos in ticks.values():
        clock[pos] = TICK_COLOR

    dr, dc = direction
    radius = center
    clock[center, center] = HAND_COLOR
    for step in range(1, radius):
        clock[center + dr * step, center + dc * step] = HAND_COLOR
    return clock


def _sprite_array(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = color
    return out


def _boxes_separated(
    a_top: int,
    a_left: int,
    a_h: int,
    a_w: int,
    b_top: int,
    b_left: int,
    b_h: int,
    b_w: int,
    gap: int = 1,
) -> bool:
    a_r0 = a_top - gap
    a_r1 = a_top + a_h - 1 + gap
    a_c0 = a_left - gap
    a_c1 = a_left + a_w - 1 + gap
    b_r0 = b_top
    b_r1 = b_top + b_h - 1
    b_c0 = b_left
    b_c1 = b_left + b_w - 1
    return a_r1 < b_r0 or b_r1 + gap < a_top or a_c1 < b_c0 or b_c1 + gap < a_left


def _place_on_canvas(clock: np.ndarray, sprite: np.ndarray) -> np.ndarray | None:
    clock_h, clock_w = clock.shape
    sprite_h, sprite_w = sprite.shape
    grid_h = _RNG.randint(18, 26)
    grid_w = _RNG.randint(18, 26)

    if clock_h + 2 > grid_h or clock_w + 2 > grid_w or sprite_h + 2 > grid_h or sprite_w + 2 > grid_w:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        clock_r = _RNG.randint(1, grid_h - clock_h - 1)
        clock_c = _RNG.randint(1, grid_w - clock_w - 1)
        sprite_r = _RNG.randint(1, grid_h - sprite_h - 1)
        sprite_c = _RNG.randint(1, grid_w - sprite_w - 1)
        if not _boxes_separated(clock_r, clock_c, clock_h, clock_w, sprite_r, sprite_c, sprite_h, sprite_w, gap=1):
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        clock_mask = clock != BG
        sprite_mask = sprite != BG
        grid[clock_r : clock_r + clock_h, clock_c : clock_c + clock_w][clock_mask] = clock[clock_mask]
        grid[sprite_r : sprite_r + sprite_h, sprite_c : sprite_c + sprite_w][sprite_mask] = sprite[sprite_mask]
        return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    direction = _RNG.choice(DIRS4)
    clock_size = _RNG.choice((7, 9))
    clock = _paint_clock(clock_size, direction)

    sprite_mask = _random_sprite_mask()
    sprite_color = _RNG.choice(SPRITE_COLORS)
    sprite = _sprite_array(sprite_mask, sprite_color)
    sprite = np.rot90(sprite, -_RNG.randrange(4))

    grid = _place_on_canvas(clock, sprite)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.size == 0 or out.shape[0] > 20 or out.shape[1] > 20:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid clock-and-sprite input")
