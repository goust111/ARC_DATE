import random
from collections import Counter

import numpy as np


BG = 5
RED = 2
PALETTE = tuple(range(10))
GEN_MAIN_COLORS = (1, 3, 4, 6, 7, 8, 9)
GEN_PARTNER_COLORS = (0, 1, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")
_GEN_COUNTER = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _compass_partner(grid: np.ndarray, coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    cell_set = {tuple(map(int, xy)) for xy in coords}
    color_counts = Counter(int(grid[r, c]) for r, c in cell_set)
    red_cells = [xy for xy in cell_set if int(grid[xy]) == RED]
    assert len(red_cells) == 1
    red_cell = red_cells[0]
    dominant_color = max(color_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    candidates = []
    for nr, nc in _neighbors4(*red_cell):
        if (nr, nc) not in cell_set:
            continue
        color = int(grid[nr, nc])
        if color == RED:
            continue
        score = (color_counts[color], 1 if color == dominant_color else 0, nr, nc)
        candidates.append((score, (nr, nc)))

    assert candidates
    candidates.sort(key=lambda item: item[0])
    return red_cell, candidates[0][1]


def _move_delta(direction: str, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return -r0, 0
    if direction == "down":
        return h - 1 - r1, 0
    if direction == "left":
        return 0, -c0
    if direction == "right":
        return 0, w - 1 - c1
    raise ValueError(direction)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each connected foreground object to the edge indicated by its internal two-pixel compass.

    The compass is a red pixel adjacent to one other non-background pixel of a different color.
    The direction from the other compass pixel to the red pixel is the movement direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = np.full_like(g, bg)

    for coords in _components(g, bg):
        red_cell, partner = _compass_partner(g, coords)
        dr = int(red_cell[0] - partner[0])
        dc = int(red_cell[1] - partner[1])
        assert abs(dr) + abs(dc) == 1
        if dr == -1:
            direction = "up"
        elif dr == 1:
            direction = "down"
        elif dc == -1:
            direction = "left"
        else:
            direction = "right"

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        shift_r, shift_c = _move_delta(direction, (int(r0), int(c0), int(r1), int(c1)), (h, w))
        for r, c in coords:
            out[int(r) + shift_r, int(c) + shift_c] = int(g[int(r), int(c)])

    return out


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _make_shape(direction: str) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    h = random.randint(4, 6)
    w = random.randint(4, 6)
    mask = np.ones((h, w), dtype=bool)

    if direction == "up":
        red = (random.randint(1, h - 3), random.randint(1, w - 2))
        partner = (red[0] + 1, red[1])
    elif direction == "down":
        red = (random.randint(2, h - 2), random.randint(1, w - 2))
        partner = (red[0] - 1, red[1])
    elif direction == "left":
        red = (random.randint(1, h - 2), random.randint(1, w - 3))
        partner = (red[0], red[1] + 1)
    else:
        red = (random.randint(1, h - 2), random.randint(2, w - 2))
        partner = (red[0], red[1] - 1)

    deletable = [(r, c) for r in range(h) for c in range(w) if (r, c) not in {red, partner}]
    random.shuffle(deletable)
    max_deletions = random.randint(1, min(5, h * w // 4))
    deletions = 0
    for r, c in deletable:
        if deletions >= max_deletions:
            break
        if 0 < r < h - 1 and 0 < c < w - 1:
            continue
        trial = mask.copy()
        trial[r, c] = False
        rows = np.any(trial, axis=1)
        cols = np.any(trial, axis=0)
        if not (rows[0] and rows[-1] and cols[0] and cols[-1]):
            continue
        if not _is_connected(trial):
            continue
        mask = trial
        deletions += 1

    return mask, red, partner


def _cells_from(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in np.argwhere(mask)}


def _clear_of_neighbors(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _paint_object(grid: np.ndarray, mask: np.ndarray, top: int, left: int, main_color: int, partner_color: int, red: tuple[int, int], partner: tuple[int, int]) -> None:
    for r, c in np.argwhere(mask):
        grid[top + int(r), left + int(c)] = main_color
    grid[top + red[0], left + red[1]] = RED
    grid[top + partner[0], left + partner[1]] = partner_color


def _direction_mix(count: int) -> list[str]:
    global _GEN_COUNTER
    primary = DIRECTIONS[_GEN_COUNTER % len(DIRECTIONS)]
    _GEN_COUNTER += 1
    chosen = [primary]
    while len(chosen) < count:
        chosen.append(random.choice(DIRECTIONS))
    random.shuffle(chosen)
    return chosen


def _place_objects(h: int, w: int, specs: list[dict]) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    input_occupied: set[tuple[int, int]] = set()
    output_occupied: set[tuple[int, int]] = set()

    for spec in specs:
        mask = spec["mask"]
        obj_h, obj_w = mask.shape
        direction = spec["direction"]
        placed = False

        for _ in range(200):
            if direction in ("up", "down"):
                if w - obj_w - 2 < 1:
                    return None
                final_left = random.randint(1, w - obj_w - 2)
                final_top = 0 if direction == "up" else h - obj_h
                max_gap = h - obj_h - 1
                if max_gap < 1:
                    return None
                gap = random.randint(1, max_gap)
                input_top = final_top + gap if direction == "up" else final_top - gap
                input_left = final_left
            else:
                if h - obj_h - 2 < 1:
                    return None
                final_top = random.randint(1, h - obj_h - 2)
                final_left = 0 if direction == "left" else w - obj_w
                max_gap = w - obj_w - 1
                if max_gap < 1:
                    return None
                gap = random.randint(1, max_gap)
                input_left = final_left + gap if direction == "left" else final_left - gap
                input_top = final_top

            input_cells = _cells_from(mask, input_top, input_left)
            output_cells = _cells_from(mask, final_top, final_left)
            if not _clear_of_neighbors(input_cells, input_occupied):
                continue
            if not _clear_of_neighbors(output_cells, output_occupied):
                continue

            _paint_object(
                grid,
                mask,
                input_top,
                input_left,
                spec["main_color"],
                spec["partner_color"],
                spec["red"],
                spec["partner"],
            )
            input_occupied |= input_cells
            output_occupied |= output_cells
            placed = True
            break

        if not placed:
            return None

    return grid


def _validate(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    bg = _get_bg(grid)
    comps = _components(grid, bg)
    if not (2 <= len(comps) <= 5):
        return False
    for coords in comps:
        try:
            red_cell, partner = _compass_partner(grid, coords)
        except AssertionError:
            return False
        if abs(red_cell[0] - partner[0]) + abs(red_cell[1] - partner[1]) != 1:
            return False
    out = main(grid)
    if np.count_nonzero(grid != bg) != np.count_nonzero(out != bg):
        return False
    out_bg = _get_bg(out)
    for coords in _components(out, out_bg):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if not (r0 == 0 or c0 == 0 or r1 == out.shape[0] - 1 or c1 == out.shape[1] - 1):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 18)
    w = random.randint(11, 18)
    count = random.randint(2, 5)
    directions = _direction_mix(count)
    main_colors = random.sample(GEN_MAIN_COLORS, count)

    specs = []
    for idx in range(count):
        direction = directions[idx]
        mask, red, partner = _make_shape(direction)
        partner_choices = [c for c in GEN_PARTNER_COLORS if c != main_colors[idx]]
        specs.append(
            {
                "direction": direction,
                "mask": mask,
                "red": red,
                "partner": partner,
                "main_color": main_colors[idx],
                "partner_color": random.choice(partner_choices),
            }
        )

    random.shuffle(specs)
    grid = _place_objects(h, w, specs)
    if grid is None:
        return None
    if not _validate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compass-slide input")
