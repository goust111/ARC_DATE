from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
SPRITE_COLOR_POOL = (4, 6, 7, 8, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)

MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _arrow_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> set[tuple[int, int]]:
    sr, sc = start
    dr, dc = direction
    cells = {(sr + k * dr, sc + k * dc) for k in range(int(length))}
    er = sr + (length - 1) * dr
    ec = sc + (length - 1) * dc
    if direction == RIGHT:
        cells.add((er - 1, ec - 1))
        cells.add((er + 1, ec - 1))
    elif direction == LEFT:
        cells.add((er - 1, ec + 1))
        cells.add((er + 1, ec + 1))
    elif direction == DOWN:
        cells.add((er - 1, ec - 1))
        cells.add((er - 1, ec + 1))
    else:
        cells.add((er + 1, ec - 1))
        cells.add((er + 1, ec + 1))
    return cells


def _transform_stamp(stamp: np.ndarray, trigger_color: int) -> np.ndarray:
    if trigger_color == RED:
        return np.rot90(stamp, -1)
    if trigger_color == BLUE:
        return np.fliplr(stamp)
    if trigger_color == GREEN:
        return np.flipud(stamp)
    raise AssertionError("unknown trigger color")


def _find_arrow(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], int]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size > 0, "arrow not found"
    grey_set = {tuple(map(int, xy)) for xy in grey_coords}
    length = int(len(grey_set) - 2)
    assert length >= 3

    matches = []
    for start in grey_set:
        for direction in DIRECTIONS:
            expected = _arrow_cells(start, direction, length)
            if expected == grey_set:
                matches.append((start, direction, length))
    assert matches, "arrow direction is ambiguous"

    viable = []
    for start, direction, length in matches:
        try:
            _find_sprite(grid, start, direction)
        except AssertionError:
            continue
        viable.append((start, direction, length))

    assert len(viable) == 1, "arrow direction is ambiguous"
    return viable[0]


def _find_sprite(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    sr, sc = start
    candidates = []
    for h in range(3, 6):
        for w in range(3, 6):
            if direction == RIGHT:
                c1 = sc - 1
                c0 = c1 - w + 1
                for r0 in range(sr - h + 1, sr + 1):
                    r1 = r0 + h - 1
                    if r0 < 0 or c0 < 0:
                        continue
                    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
                    if patch.shape == (h, w) and np.all((patch != BLACK) & (patch != GREY)):
                        candidates.append((r0, c0, r1, c1))
            elif direction == LEFT:
                c0 = sc + 1
                c1 = c0 + w - 1
                for r0 in range(sr - h + 1, sr + 1):
                    r1 = r0 + h - 1
                    if r0 < 0 or c1 >= grid.shape[1]:
                        continue
                    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
                    if patch.shape == (h, w) and np.all((patch != BLACK) & (patch != GREY)):
                        candidates.append((r0, c0, r1, c1))
            elif direction == DOWN:
                r1 = sr - 1
                r0 = r1 - h + 1
                for c0 in range(sc - w + 1, sc + 1):
                    c1 = c0 + w - 1
                    if r0 < 0 or c0 < 0:
                        continue
                    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
                    if patch.shape == (h, w) and np.all((patch != BLACK) & (patch != GREY)):
                        candidates.append((r0, c0, r1, c1))
            else:
                r0 = sr + 1
                r1 = r0 + h - 1
                for c0 in range(sc - w + 1, sc + 1):
                    c1 = c0 + w - 1
                    if r1 >= grid.shape[0] or c0 < 0:
                        continue
                    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
                    if patch.shape == (h, w) and np.all((patch != BLACK) & (patch != GREY)):
                        candidates.append((r0, c0, r1, c1))

    assert candidates, "sprite not found"
    sprite_bbox = min(candidates, key=lambda b: (b[0], b[1], b[2] - b[0], b[3] - b[1]))
    r0, c0, r1, c1 = sprite_bbox
    return sprite_bbox, grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _stamp_anchor(
    sprite_bbox: tuple[int, int, int, int],
    step_index: int,
    direction: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = sprite_bbox
    sprite_h = r1 - r0 + 1
    sprite_w = c1 - c0 + 1
    if direction == RIGHT:
        return r0, c0 + step_index * sprite_w
    if direction == LEFT:
        return r0, c0 - step_index * sprite_w
    if direction == DOWN:
        return r0 + step_index * sprite_h, c0
    return r0 - step_index * sprite_h, c0


def _sorted_triggers(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> list[int]:
    colors = []
    for rr in range(r0, r0 + h):
        for cc in range(c0, c0 + w):
            color = int(grid[rr, cc])
            if color in (RED, BLUE, GREEN):
                colors.append((cc, rr, color))
    colors.sort()
    return [color for _cc, _rr, color in colors]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly stamp the top-left sprite along the arrow direction.

    The sprite shown in the input is the exemplar. Additional copies are placed
    one original sprite-width/height apart in the arrow direction. Triggers
    found inside a stamped copy transform the stamp for subsequent copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape

    start, direction, _length = _find_arrow(g)
    sprite_bbox, stamp0 = _find_sprite(g, start, direction)

    out = g.copy()
    out[out == GREY] = BLACK

    current_stamp = stamp0.copy()

    step_index = 1
    while True:
        rr, cc = _stamp_anchor(sprite_bbox, step_index, direction)
        stamp_h, stamp_w = map(int, current_stamp.shape)
        if rr < 0 or cc < 0 or rr + stamp_h > h or cc + stamp_w > w:
            break

        trigger_colors = _sorted_triggers(out, rr, cc, stamp_h, stamp_w)
        out[rr : rr + stamp_h, cc : cc + stamp_w] = current_stamp
        for color in trigger_colors:
            current_stamp = _transform_stamp(current_stamp, color)
        step_index += 1

    return out


def _pattern_signature(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in arr.tolist())


def _make_sprite() -> np.ndarray:
    for _ in range(200):
        size = random.randint(3, 5)
        n_colors = random.randint(2, min(4, len(SPRITE_COLOR_POOL)))
        palette = random.sample(SPRITE_COLOR_POOL, n_colors)
        stamp = np.array(
            [[random.choice(palette) for _ in range(size)] for _ in range(size)],
            dtype=int,
        )
        if len(set(map(int, stamp.ravel()))) < 2:
            continue
        variants = {
            _pattern_signature(stamp),
            _pattern_signature(np.rot90(stamp, -1)),
            _pattern_signature(np.fliplr(stamp)),
            _pattern_signature(np.flipud(stamp)),
        }
        if len(variants) < 4:
            continue
        return stamp
    raise RuntimeError("failed to create asymmetric sprite")


def _choose_position_range(lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _sample_layout(
    stamp: np.ndarray,
    direction: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int], int, list[tuple[int, int]], int] | None:
    size = int(stamp.shape[0])
    height = random.randint(24, 36)
    width = random.randint(24, 36)
    min_copies = 4

    if direction == RIGHT:
        r0 = _choose_position_range(0, height - size)
        c0 = _choose_position_range(0, min(6, width - (min_copies + 1) * size))
    elif direction == LEFT:
        r0 = _choose_position_range(0, height - size)
        c0 = _choose_position_range(min_copies * size, min(width - size, min_copies * size + 6))
    elif direction == DOWN:
        r0 = _choose_position_range(0, min(6, height - (min_copies + 1) * size))
        c0 = _choose_position_range(0, width - size)
    else:
        r0 = _choose_position_range(min_copies * size, min(height - size, min_copies * size + 6))
        c0 = _choose_position_range(0, width - size)

    if r0 is None or c0 is None:
        return None

    bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
    anchors = []
    step = 1
    while True:
        rr, cc = _stamp_anchor(bbox, step, direction)
        if rr < 0 or cc < 0 or rr + size > height or cc + size > width:
            break
        anchors.append((rr, cc))
        step += 1
    if len(anchors) < min_copies:
        return None

    if direction in (LEFT, RIGHT):
        line_offset = random.randint(1, size - 2)
        start = (r0 + line_offset, c0 + size) if direction == RIGHT else (r0 + line_offset, c0 - 1)
        max_len = width - start[1] if direction == RIGHT else start[1] + 1
    else:
        line_offset = random.randint(1, size - 2)
        start = (r0 + size, c0 + line_offset) if direction == DOWN else (r0 - 1, c0 + line_offset)
        max_len = height - start[0] if direction == DOWN else start[0] + 1

    if max_len < 3:
        return None

    arrow_len = random.randint(3, min(10, max_len))
    grid = np.full((height, width), BLACK, dtype=int)
    grid[r0 : r0 + size, c0 : c0 + size] = stamp
    arrow = _arrow_cells(start, direction, arrow_len)
    for rr, cc in arrow:
        if not (0 <= rr < height and 0 <= cc < width):
            return None
        if int(grid[rr, cc]) != BLACK:
            return None
        grid[rr, cc] = GREY

    return grid, bbox, start, arrow_len, anchors, line_offset


def _multi_trigger_positions(
    anchor: tuple[int, int],
    direction: tuple[int, int],
    line_offset: int,
    size: int,
    count: int,
) -> list[tuple[int, int]]:
    ar, ac = anchor
    if direction in (LEFT, RIGHT):
        row = ar + ((line_offset + 1) % size)
        if count == 2:
            cols = [ac, ac + size - 1]
        else:
            cols = [ac, ac + 2, ac + 4]
        return [(row, col) for col in cols]

    col = ac + ((line_offset + 1) % size)
    if count == 2:
        rows = [ar, ar + size - 1]
    else:
        rows = [ar, ar + 2, ar + 4]
    return [(row, col) for row in rows]


def _single_trigger_position(
    anchor: tuple[int, int],
    direction: tuple[int, int],
    line_offset: int,
    occupied: set[tuple[int, int]],
    grid: np.ndarray,
    size: int,
) -> tuple[int, int] | None:
    ar, ac = anchor
    candidates = []
    for rr in range(ar, ar + size):
        for cc in range(ac, ac + size):
            if (rr, cc) in occupied:
                continue
            if int(grid[rr, cc]) != BLACK:
                continue
            if direction in (LEFT, RIGHT) and rr == ar + line_offset:
                continue
            if direction in (UP, DOWN) and cc == ac + line_offset:
                continue
            candidates.append((rr, cc))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_triggers(
    grid: np.ndarray,
    anchors: list[tuple[int, int]],
    direction: tuple[int, int],
    line_offset: int,
    size: int,
) -> bool:
    mode_pool = ["single", "double"]
    if size == 5:
        mode_pool.append("triple")
    mode = random.choice(mode_pool)

    schedule: dict[int, list[int]] = {}
    colors = [RED, BLUE, GREEN]
    random.shuffle(colors)
    usable = list(range(len(anchors)))
    random.shuffle(usable)

    if mode == "single":
        if len(usable) < 3:
            return False
        for idx, color in zip(usable[:3], colors):
            schedule[idx] = [color]
        if len(usable) >= 4 and random.random() < 0.5:
            schedule[usable[3]] = [random.choice((RED, BLUE, GREEN))]
    elif mode == "double":
        if len(usable) < 2:
            return False
        schedule[usable[0]] = colors[:2]
        schedule[usable[1]] = [colors[2]]
        if len(usable) >= 3 and random.random() < 0.5:
            schedule[usable[2]] = [random.choice((RED, BLUE, GREEN))]
    else:
        schedule[usable[0]] = colors[:]
        if len(usable) >= 2 and random.random() < 0.5:
            schedule[usable[1]] = [random.choice((RED, BLUE, GREEN))]

    occupied: set[tuple[int, int]] = set()
    for idx, trigger_colors in schedule.items():
        anchor = anchors[idx]
        if len(trigger_colors) == 1:
            pos = _single_trigger_position(anchor, direction, line_offset, occupied, grid, size)
            if pos is None:
                return False
            rr, cc = pos
            grid[rr, cc] = trigger_colors[0]
            occupied.add(pos)
            continue

        positions = _multi_trigger_positions(anchor, direction, line_offset, size, len(trigger_colors))
        for rr, cc in positions:
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                return False
            if int(grid[rr, cc]) != BLACK:
                return False
            if (rr, cc) in occupied:
                return False
        for (rr, cc), color in zip(positions, trigger_colors):
            grid[rr, cc] = color
            occupied.add((rr, cc))

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        stamp = _make_sprite()
        direction = random.choice(DIRECTIONS)
        sampled = _sample_layout(stamp, direction)
        if sampled is None:
            continue
        grid, sprite_bbox, _start, _arrow_len, anchors, line_offset = sampled
        if not _place_triggers(grid, anchors, direction, line_offset, int(stamp.shape[0])):
            continue

        trigger_colors = {int(color) for color in grid.ravel() if int(color) in (RED, BLUE, GREEN)}
        if trigger_colors != {RED, BLUE, GREEN}:
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if np.count_nonzero(grid == GREY) == 0:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
