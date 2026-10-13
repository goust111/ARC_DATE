from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

LEGEND = np.array([[RED, YELLOW], [GREEN, BLUE]], dtype=int)
SWAP = {RED: YELLOW, YELLOW: RED, GREEN: BLUE, BLUE: GREEN}
INVERSE_SWAP = {dst: src for src, dst in SWAP.items()}

# The fixed legend position gives each color a visible sign-flip code.
COLOR_TO_SIGN = {
    RED: (-1, -1),
    YELLOW: (-1, 1),
    GREEN: (1, -1),
    BLUE: (1, 1),
}

ALL_DIRECTIONS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

VERTICAL_DIRECTIONS = [(-1, 0), (1, 0)]
HORIZONTAL_DIRECTIONS = [(0, -1), (0, 1)]
DIAGONAL_DIRECTIONS = [d for d in ALL_DIRECTIONS if d[0] != 0 and d[1] != 0]

GRID_SIZE_RANGE = (15, 18)
SPRITE_COUNT_RANGE = (4, 8)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or visited[r0, c0]:
                continue

            stack = [(r0, c0)]
            visited[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c, h, w):
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            boxes.append((min(rows), min(cols), max(rows), max(cols)))

    return boxes


def _transform_direction(base_direction: tuple[int, int], swapped_color: int) -> tuple[int, int]:
    dr, dc = base_direction
    sr, sc = COLOR_TO_SIGN[swapped_color]
    out_r = 0 if dr == 0 else dr * sr
    out_c = 0 if dc == 0 else dc * sc
    return int(out_r), int(out_c)


def _slot_origin(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dr + 1) * 3, (dc + 1) * 3


def _find_sprites(grid: np.ndarray) -> list[np.ndarray]:
    assert grid.ndim == 2 and grid.shape[0] >= 5 and grid.shape[1] >= 5
    assert np.array_equal(grid[:2, :2], LEGEND), "expected the fixed legend in the upper-left corner"

    fg_mask = grid != BLACK
    fg_mask[:2, :2] = False
    boxes = _component_bboxes(fg_mask)

    sprites: list[np.ndarray] = []
    for r0, c0, r1, c1 in sorted(boxes):
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each sprite must occupy a 3x3 bbox"
        block = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        assert np.all(block != BLACK), "sprite bbox must be fully filled"
        sprites.append(block)

    assert SPRITE_COUNT_RANGE[0] <= len(sprites) <= SPRITE_COUNT_RANGE[1]
    return sprites


def _solve_from_sprites(sprites: list[np.ndarray]) -> np.ndarray:
    out = np.full((9, 9), GREY, dtype=int)
    blank_count = 0
    target_directions: set[tuple[int, int]] = set()
    source_directions: set[tuple[int, int]] = {(0, 0)}

    for block in sprites:
        colored = np.argwhere(block != GREY)
        if colored.shape[0] == 0:
            blank_count += 1
            continue

        assert colored.shape[0] == 1, "nonblank sprite must contain exactly one marker cell"
        mr, mc = map(int, colored[0])
        base_direction = (mr - 1, mc - 1)
        assert base_direction != (0, 0), "the colored marker must not sit at the center"
        assert base_direction not in source_directions, "sprite marker directions must be unique"
        source_directions.add(base_direction)

        color_before = int(block[mr, mc])
        assert color_before in SWAP, "marker color must come from the legend palette"
        color_after = SWAP[color_before]

        rewritten_direction = _transform_direction(base_direction, color_after)
        assert rewritten_direction != (0, 0)
        assert rewritten_direction not in target_directions, "rewritten directions must be unique"
        target_directions.add(rewritten_direction)

        transformed = block.copy()
        transformed[mr, mc] = color_after
        r0, c0 = _slot_origin(rewritten_direction)
        out[r0 : r0 + 3, c0 : c0 + 3] = transformed

    assert blank_count == 1, "expected exactly one blank sprite"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each sprite's direction marker through the legend, then route the whole sprite into the matching 3x3 slot of a gray 9x9 board.

    Input structure:
    - The grid has a black background and a fixed 2x2 legend in the upper-left corner: top row [red, yellow], bottom row [green, blue].
    - Elsewhere there are 4-8 isolated 3x3 gray sprites. Every nonblank sprite has exactly one non-gray marker cell in one of the eight neighbor positions around the gray center, and exactly one sprite is fully gray.
    - Marker colors use only the four legend colors.

    Rule:
    - Swap each marker color with its legend row-mate: red <-> yellow and green <-> blue.
    - Read the marker offset from the sprite center as a base direction. Then use the recolored marker's fixed position in the 2x2 legend as a sign-flip code that rewrites that direction.
    - Start from a 9x9 all-gray output viewed as a 3x3 array of 3x3 slots. Ignore the blank sprite and paste every recolored nonblank sprite into the slot named by its rewritten direction.

    Key insight:
    - The legend is not a cosmetic recolor: after the swap, the marker color changes how the same offset should be interpreted, so routing depends on both the offset and the recolored legend position.
    """
    g = np.asarray(grid, dtype=int)
    sprites = _find_sprites(g)
    return _solve_from_sprites(sprites)


def _direction_category(direction: tuple[int, int]) -> str:
    dr, dc = direction
    if dr == 0:
        return "horizontal"
    if dc == 0:
        return "vertical"
    return "diagonal"


def _pick_targets(source_directions: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int]]:
    groups = {
        "vertical": VERTICAL_DIRECTIONS,
        "horizontal": HORIZONTAL_DIRECTIONS,
        "diagonal": DIAGONAL_DIRECTIONS,
    }
    chosen: dict[tuple[int, int], tuple[int, int]] = {}
    for category, pool in groups.items():
        members = [d for d in source_directions if _direction_category(d) == category]
        if not members:
            continue
        targets = random.sample(pool, len(members))
        random.shuffle(members)
        random.shuffle(targets)
        for src, dst in zip(members, targets, strict=True):
            chosen[src] = dst
    return chosen


def _candidate_swapped_colors(
    base_direction: tuple[int, int],
    target_direction: tuple[int, int],
) -> list[int]:
    candidates = []
    for color_after in (RED, YELLOW, GREEN, BLUE):
        if _transform_direction(base_direction, color_after) == target_direction:
            candidates.append(color_after)
    assert candidates
    return candidates


def _assign_original_colors(
    source_directions: list[tuple[int, int]],
    target_map: dict[tuple[int, int], tuple[int, int]],
) -> dict[tuple[int, int], int]:
    order = source_directions[:]
    random.shuffle(order)

    used_swapped: set[int] = set()
    colors: dict[tuple[int, int], int] = {}
    for direction in order:
        target = target_map[direction]
        candidates = _candidate_swapped_colors(direction, target)
        fresh = [color for color in candidates if color not in used_swapped]
        color_after = random.choice(fresh or candidates)
        used_swapped.add(color_after)
        colors[direction] = INVERSE_SWAP[color_after]
    return colors


def _make_sprite(base_direction: tuple[int, int] | None, color_before: int | None) -> np.ndarray:
    sprite = np.full((3, 3), GREY, dtype=int)
    if base_direction is None:
        return sprite
    dr, dc = base_direction
    sprite[1 + dr, 1 + dc] = int(color_before)
    return sprite


def _can_place(top: int, left: int, blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + 4)
    c1 = min(w, left + 4)
    return not blocked[r0:r1, c0:c1].any()


def _mark_blocked(top: int, left: int, blocked: np.ndarray) -> None:
    h, w = blocked.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + 4)
    c1 = min(w, left + 4)
    blocked[r0:r1, c0:c1] = True


def _place_sprites(blocks: list[np.ndarray], h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), BLACK, dtype=int)
    grid[:2, :2] = LEGEND

    blocked = np.zeros((h, w), dtype=bool)
    _mark_blocked(0, 0, blocked)

    order = blocks[:]
    random.shuffle(order)
    for block in order:
        candidates = [
            (r, c)
            for r in range(h - 2)
            for c in range(w - 2)
            if _can_place(r, c, blocked)
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        grid[top : top + 3, left : left + 3] = block
        _mark_blocked(top, left, blocked)

    return grid


def _oracle_output(
    source_directions: list[tuple[int, int]],
    colors_before: dict[tuple[int, int], int],
    target_map: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    out = np.full((9, 9), GREY, dtype=int)
    for direction in source_directions:
        color_before = colors_before[direction]
        color_after = SWAP[color_before]
        assert _transform_direction(direction, color_after) == target_map[direction]
        sprite = _make_sprite(direction, color_before)
        rr, cc = 1 + direction[0], 1 + direction[1]
        sprite[rr, cc] = color_after
        top, left = _slot_origin(target_map[direction])
        out[top : top + 3, left : left + 3] = sprite
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    total_sprites = random.randint(*SPRITE_COUNT_RANGE)
    nonblank_count = total_sprites - 1

    source_directions = random.sample(ALL_DIRECTIONS, nonblank_count)
    target_map = _pick_targets(source_directions)
    colors_before = _assign_original_colors(source_directions, target_map)

    blocks = [_make_sprite(None, None)]
    for direction in source_directions:
        blocks.append(_make_sprite(direction, colors_before[direction]))

    grid = _place_sprites(blocks, h, w)
    if grid is None:
        return None

    expected = _oracle_output(source_directions, colors_before, target_map)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-5 input")
