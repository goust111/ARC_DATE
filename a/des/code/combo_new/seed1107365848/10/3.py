from __future__ import annotations

import random

import numpy as np


BG_COLOR = 7
SEED_COLOR = 3
BORDER_COLORS = [0, 1, 2, 4, 5, 6, 8, 9]

GRID_MIN = 14
GRID_MAX = 30
ROOM_COUNT_RANGE = (1, 4)
ROOM_SIZE_RANGE = (5, 12)
ROOM_GAP = 1
OUTER_MARGIN = 1
PLACEMENT_TRIES = 120
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    if len(points) != 2 * (height + width) - 4:
        return None

    pset = set(points)
    for c in range(c0, c1 + 1):
        if (r0, c) not in pset or (r1, c) not in pset:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in pset or (r, c1) not in pset:
            return None
    return r0, c0, r1, c1


def _majority_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _paint_outline(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _max_inset_layers(height: int, width: int) -> int:
    return min((height - 1) // 2, (width - 1) // 2)


def _detect_rooms(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int, int, int], int]]:
    rooms: list[tuple[tuple[int, int, int, int], int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            bbox = _is_rectangular_frame(comp)
            if bbox is not None:
                rooms.append((bbox, color))
    rooms.sort(key=lambda item: item[0])
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the seed pixels inside each colored room and add that many inset copies of its border.

    Input structure:
    - The background is a dominant color.
    - Several non-overlapping one-pixel-thick rectangular border rooms appear in different colors.
    - Inside each room, the area is background except for a small number of seed pixels away from the border.

    Rule:
    - Detect every rectangular room border and count the non-background seed pixels inside it.
    - Erase those seed pixels back to the background color.
    - For a room with count N, draw up to N inward-shifted copies of the same rectangular border color,
      each one pixel farther from the original border than the previous copy.
    - Stop early for a room if the shrinking rectangle runs out of space.

    Key insight:
    - The interior seed pixels are only a counter: their positions do not matter, only how many there are,
      and that count determines how many concentric inset border outlines are generated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _majority_color(g)
    rooms = _detect_rooms(g, bg)
    assert rooms, "expected at least one rectangular room"

    out = g.copy()
    for (r0, c0, r1, c1), color in rooms:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = inner != bg
        n_seeds = int(seed_mask.sum())
        assert n_seeds >= 1, "each room must contain at least one seed"

        cleared = out[r0 + 1 : r1, c0 + 1 : c1]
        cleared[seed_mask] = bg

        for inset in range(1, n_seeds + 1):
            rr0 = r0 + inset
            cc0 = c0 + inset
            rr1 = r1 - inset
            cc1 = c1 - inset
            if rr0 > rr1 or cc0 > cc1:
                break
            _paint_outline(out, rr0, cc0, rr1, cc1, color)

    return out


def _seed_capacity(height: int, width: int) -> int:
    return max(0, height - 4) * max(0, width - 4)


def _room_spec_candidates(max_h: int, max_w: int, truncated: bool) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    for height in range(ROOM_SIZE_RANGE[0], min(ROOM_SIZE_RANGE[1], max_h) + 1):
        for width in range(ROOM_SIZE_RANGE[0], min(ROOM_SIZE_RANGE[1], max_w) + 1):
            seed_cap = _seed_capacity(height, width)
            if seed_cap <= 0:
                continue
            max_layers = _max_inset_layers(height, width)
            max_seeds = min(4, seed_cap)
            if truncated:
                low = max_layers + 1
                high = max_seeds
            else:
                low = 1
                high = min(max_layers, max_seeds)
            if low > high:
                continue
            for n_seeds in range(low, high + 1):
                specs.append((height, width, n_seeds))
    return specs


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_rooms(
    height: int, width: int, specs: list[tuple[int, int, int]]
) -> list[tuple[int, int, int, int, int]] | None:
    ordered = specs[:]
    random.shuffle(ordered)
    ordered.sort(key=lambda item: item[0] * item[1], reverse=True)

    placed: list[tuple[int, int, int, int, int]] = []
    for room_h, room_w, n_seeds in ordered:
        max_r0 = height - room_h - OUTER_MARGIN
        max_c0 = width - room_w - OUTER_MARGIN
        if max_r0 < OUTER_MARGIN or max_c0 < OUTER_MARGIN:
            return None

        success = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(OUTER_MARGIN, max_r0)
            c0 = random.randint(OUTER_MARGIN, max_c0)
            rect = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
            if all(_rects_separated(rect, prev[:4], ROOM_GAP) for prev in placed):
                placed.append((*rect, n_seeds))
                success = True
                break
        if not success:
            return None
    return placed


def _draw_room(grid: np.ndarray, room: tuple[int, int, int, int, int], color: int) -> None:
    r0, c0, r1, c1, n_seeds = room
    _paint_outline(grid, r0, c0, r1, c1, color)

    seed_cells = [
        (r, c)
        for r in range(r0 + 2, r1 - 1)
        for c in range(c0 + 2, c1 - 1)
    ]
    assert len(seed_cells) >= n_seeds, "room must have enough seed slots"
    for r, c in random.sample(seed_cells, n_seeds):
        grid[r, c] = SEED_COLOR


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)

    max_room_h = height - 2 * OUTER_MARGIN
    max_room_w = width - 2 * OUTER_MARGIN
    normal_specs = _room_spec_candidates(max_room_h, max_room_w, truncated=False)
    truncated_specs = _room_spec_candidates(max_room_h, max_room_w, truncated=True)
    if not normal_specs:
        return None

    room_count = random.randint(*ROOM_COUNT_RANGE)
    flags: list[bool] = []
    if room_count == 1:
        flags = [bool(truncated_specs) and random.choice([False, True])]
    else:
        if truncated_specs:
            flags = [True, False] + [random.choice([False, True]) for _ in range(room_count - 2)]
            random.shuffle(flags)
        else:
            flags = [False] * room_count

    specs: list[tuple[int, int, int]] = []
    for truncated in flags:
        pool = truncated_specs if truncated else normal_specs
        if not pool:
            return None
        specs.append(random.choice(pool))

    rooms = _place_rooms(height, width, specs)
    if rooms is None:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    colors = random.sample(BORDER_COLORS, len(rooms))
    for room, color in zip(rooms, colors):
        _draw_room(grid, room, color)

    detected = _detect_rooms(grid, BG_COLOR)
    if len(detected) != len(rooms):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-seed grid")
