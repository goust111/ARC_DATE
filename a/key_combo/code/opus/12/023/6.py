from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

FRAME_COLORS = (BLUE, RED, GREEN, YELLOW)
MOVE_DIRS = {
    RED: (-1, 0),
    GREEN: (1, 0),
    BLUE: (0, -1),
    YELLOW: (0, 1),
}
STEP1_PRIORITY = {
    BLUE: 4,
    RED: 3,
    YELLOW: 2,
    GREEN: 1,
}
COLOR_ORDER = [BLUE, RED, GREEN, YELLOW]


def _bbox_of_nonblack(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BLACK)
    if coords.size == 0:
        raise AssertionError("expected non-black frame pixels")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _bbox_of_nonblack(g)
    assert r1 - r0 >= 7 and c1 - c0 >= 7, "frame interior must be at least 6x6"
    assert int(g[r0, c0]) == BLACK and int(g[r0, c1]) == BLACK
    assert int(g[r1, c0]) == BLACK and int(g[r1, c1]) == BLACK
    assert np.all(g[r0, c0 + 1 : c1] == RED)
    assert np.all(g[r1, c0 + 1 : c1] == GREEN)
    assert np.all(g[r0 + 1 : r1, c0] == BLUE)
    assert np.all(g[r0 + 1 : r1, c1] == YELLOW)
    return r0, c0, r1, c1


def _is_shifted_row(r: int, box: tuple[int, int, int, int]) -> bool:
    r0, _c0, r1, _c1 = box
    bottom = r1 - 1
    assert r0 < r < r1
    return (bottom - r) % 2 == 1


def _step1_destination(
    r: int,
    c: int,
    color: int,
    box: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    dr, dc = MOVE_DIRS[color]
    nr, nc = r + dr, c + dc
    if nr in (r0, r1) or nc in (c0, c1):
        return r, c
    return nr, nc


def _apply_step1(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    return_groups: bool = False,
):
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = box
    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = BLACK

    groups: dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]] = defaultdict(list)
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(g[r, c])
            if color == BLACK:
                continue
            if color in MOVE_DIRS:
                nr, nc = _step1_destination(r, c, color, box)
                groups[(nr, nc)].append((STEP1_PRIORITY[color], color, (r, c)))
            else:
                groups[(r, c)].append((0, color, (r, c)))

    for (r, c), contenders in groups.items():
        best = max(contenders, key=lambda item: (item[0], -item[1]))
        out[r, c] = best[1]

    if return_groups:
        return out, groups
    return out


def _apply_wave(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = box
    for r in range(r0 + 1, r1):
        if not _is_shifted_row(r, box):
            continue
        row = out[r, c0 + 1 : c1].copy()
        out[r, c0 + 1 : c1] = np.roll(row, 1)
    return out


def _transform(grid: np.ndarray) -> np.ndarray:
    box = _find_frame(grid)
    return _apply_wave(_apply_step1(grid, box), box)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each frame-colored interior pixel one step toward its matching side, then wave-shift alternating interior rows from the bottom.

    Input structure:
    - A black background contains one 1-cell-thick rectangular frame whose corners are black, top side is red, bottom side is green, left side is blue, and right side is yellow.
    - The frame interior is mostly black and contains sparse single-pixel foreground marks colored blue, red, green, or yellow.
    - The meaningful transform region is the area strictly inside that colored frame; the frame itself stays fixed.

    Rule:
    - First, move every interior pixel of a frame-side color one cell toward the matching side: red up, green down, blue left, yellow right; if that step would land on the frame border, the pixel stays put.
    - Compute those moves simultaneously from the pre-move snapshot; when multiple pixels target the same interior cell, keep only the highest-priority color in the order blue, red, yellow, green.
    - Then, on the post-move interior, keep the bottom interior row fixed and circularly shift every other row above it one cell to the right, alternating upward from the bottom.
    - Leave all black cells outside the moved/shifted interior content and the whole rectangular frame unchanged.

    Key insight:
    - A pixel's color decides the first translation, but the second wave depends on the row reached after that first move, so the final position comes from the ordered composition of both steps.
    """
    return _transform(np.asarray(grid, dtype=int))


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 + 1 : c1] = RED
    grid[r1, c0 + 1 : c1] = GREEN
    grid[r0 + 1 : r1, c0] = BLUE
    grid[r0 + 1 : r1, c1] = YELLOW


def _reverse_source(
    dest: tuple[int, int],
    color: int,
) -> tuple[int, int]:
    dr, dc = MOVE_DIRS[color]
    return int(dest[0] - dr), int(dest[1] - dc)


def _in_interior(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    r, c = cell
    return r0 < r < r1 and c0 < c < c1


def _source_is_valid(cell: tuple[int, int], color: int, box: tuple[int, int, int, int]) -> bool:
    if not _in_interior(cell, box):
        return False
    r0, c0, r1, c1 = box
    r, c = cell
    if color == RED:
        return r > r0 + 1
    if color == GREEN:
        return r < r1 - 1
    if color == BLUE:
        return c > c0 + 1
    if color == YELLOW:
        return c < c1 - 1
    return False


def _iter_valid_sources(box: tuple[int, int, int, int], color: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            cell = (r, c)
            if _source_is_valid(cell, color, box):
                cells.append(cell)
    return cells


def _step1_collision_cells(groups: dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]]) -> list[tuple[int, int]]:
    return [cell for cell, contenders in groups.items() if len(contenders) > 1]


def _count_step1_pixels(grid: np.ndarray, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    return int(np.count_nonzero(interior))


def _validate_generated(grid: np.ndarray, *, require_collision: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    try:
        box = _find_frame(g)
    except AssertionError:
        return False

    r0, c0, r1, c1 = box
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    if np.count_nonzero(outside) != 0:
        return False

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    colors = [int(v) for v in interior.flatten() if int(v) != BLACK]
    if not (8 <= len(colors) <= 12):
        return False
    if not set(COLOR_ORDER).issubset(colors):
        return False

    nonblack = np.argwhere(interior != BLACK)
    for rr, cc in nonblack:
        color = int(interior[rr, cc])
        if color not in MOVE_DIRS:
            return False
        if not _source_is_valid((r0 + 1 + int(rr), c0 + 1 + int(cc)), color, box):
            return False

    step1, groups = _apply_step1(g, box, return_groups=True)
    collisions = _step1_collision_cells(groups)
    if require_collision and not collisions:
        return False

    if require_collision:
        winner_seen = False
        for cell in collisions:
            contenders = groups[cell]
            best_color = max(contenders, key=lambda item: item[0])[1]
            if int(step1[cell]) != best_color:
                return False
            winner_seen = True
        if not winner_seen:
            return False

    shifted_with_pixels = 0
    fixed_with_pixels = 0
    vertical_on_shifted = 0
    for r in range(r0 + 1, r1):
        row = step1[r, c0 + 1 : c1]
        has_pixels = bool(np.any(row != BLACK))
        if _is_shifted_row(r, box):
            shifted_with_pixels += int(has_pixels)
        else:
            fixed_with_pixels += int(has_pixels)
    if shifted_with_pixels == 0 or fixed_with_pixels == 0:
        return False

    for dest, contenders in groups.items():
        for _priority, color, _src in contenders:
            if color in (RED, GREEN) and _is_shifted_row(dest[0], box):
                vertical_on_shifted += 1
    if vertical_on_shifted == 0:
        return False

    out = _apply_wave(step1, box)
    if np.array_equal(out, g):
        return False
    if np.array_equal(out, step1):
        return False

    wave_first = _apply_step1(_apply_wave(g, box), box)
    if np.array_equal(out, wave_first):
        return False

    if _count_step1_pixels(step1, box) > len(colors):
        return False

    return True


def _random_frame_box(rng: random.Random) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    inner_h = rng.randint(6, 9)
    inner_w = rng.randint(6, 10)
    top = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    left = rng.randint(1, 3)
    right = rng.randint(1, 3)
    h = top + inner_h + 2 + bottom
    w = left + inner_w + 2 + right
    box = (top, left, top + inner_h + 1, left + inner_w + 1)
    return box, (h, w)


def _choose_collision_pattern(
    rng: random.Random,
    box: tuple[int, int, int, int],
) -> tuple[tuple[int, int], list[int]] | None:
    patterns = [
        [BLUE, RED],
        [RED, YELLOW],
        [YELLOW, GREEN],
        [BLUE, GREEN],
        [RED, GREEN],
        [BLUE, YELLOW],
        [BLUE, RED, YELLOW],
        [RED, YELLOW, GREEN],
    ]
    rng.shuffle(patterns)
    r0, c0, r1, c1 = box
    interior_cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
    ]
    rng.shuffle(interior_cells)
    for colors in patterns:
        for dest in interior_cells:
            sources = [_reverse_source(dest, color) for color in colors]
            if len(set(sources)) != len(sources):
                continue
            if not all(_source_is_valid(src, color, box) for src, color in zip(sources, colors)):
                continue
            return dest, colors
    return None


def _too_close(cell: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    r, c = cell
    for ur, uc in used:
        if max(abs(r - ur), abs(c - uc)) <= 1:
            return True
    return False


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    box, shape = _random_frame_box(rng)
    grid = np.zeros(shape, dtype=int)
    _draw_frame(grid, box)

    total_pixels = rng.randint(8, 12)
    require_collision = rng.random() < 0.7

    placed: list[tuple[tuple[int, int], int]] = []
    used_sources: set[tuple[int, int]] = set()
    used_dests: dict[tuple[int, int], int] = {}

    if require_collision:
        collision = _choose_collision_pattern(rng, box)
        if collision is None:
            return None
        collision_dest, collision_colors = collision
        for color in collision_colors:
            src = _reverse_source(collision_dest, color)
            placed.append((src, color))
            used_sources.add(src)
            used_dests[collision_dest] = used_dests.get(collision_dest, 0) + 1
    else:
        collision_dest = None

    mandatory = COLOR_ORDER.copy()
    for _src, color in placed:
        if color in mandatory:
            mandatory.remove(color)

    while len(placed) < total_pixels:
        if mandatory:
            color = mandatory.pop(rng.randrange(len(mandatory)))
        else:
            color = rng.choice(COLOR_ORDER)

        candidates = _iter_valid_sources(box, color)
        rng.shuffle(candidates)
        chosen = None
        for src in candidates:
            if src in used_sources:
                continue
            if _too_close(src, used_sources):
                continue
            dest = _step1_destination(src[0], src[1], color, box)
            if collision_dest is not None and dest == collision_dest:
                continue
            if dest in used_dests:
                continue
            chosen = (src, dest)
            break
        if chosen is None:
            return None

        src, dest = chosen
        placed.append((src, color))
        used_sources.add(src)
        used_dests[dest] = 1

    rng.shuffle(placed)
    for (r, c), color in placed:
        grid[r, c] = color

    if not _validate_generated(grid, require_collision=require_collision):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        seed = random.randrange(1 << 30)
        grid = _generate_attempt(random.Random(seed))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid border-shift wave puzzle")
