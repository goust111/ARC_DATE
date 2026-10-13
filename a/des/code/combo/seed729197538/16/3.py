from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(10))

# --------------- solver ---------------


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_frame_bbox(component: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    pts = set(component)
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(pts) != expected:
        return None

    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return None
    for r in range(r0 + 1, r1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return None

    if any(r0 < r < r1 and c0 < c < c1 for r, c in pts):
        return None
    return (r0, c0, r1, c1)


def _find_rooms(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    bg = _background_color(grid)
    candidates: list[tuple[int, list[dict[str, int]]]] = []

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue

        components = _components4(grid == color)
        if not components:
            continue

        rooms: list[dict[str, int]] = []
        ok = True
        for comp in components:
            bbox = _rect_frame_bbox(comp)
            if bbox is None:
                ok = False
                break

            r0, c0, r1, c1 = bbox
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or np.any(interior == color):
                ok = False
                break

            seed_positions = np.argwhere((interior != bg) & (interior != color))
            if seed_positions.shape[0] != 1:
                ok = False
                break

            seed_r, seed_c = map(int, seed_positions[0])
            seed_r += r0 + 1
            seed_c += c0 + 1
            rooms.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "seed_r": seed_r,
                    "seed_c": seed_c,
                    "seed_color": int(grid[seed_r, seed_c]),
                    "ray_len": int(np.count_nonzero(interior == bg)),
                }
            )

        if ok and rooms:
            candidates.append((color, rooms))

    assert len(candidates) == 1, "expected exactly one frame color"
    frame_color, rooms = candidates[0]

    covered = np.zeros_like(grid, dtype=bool)
    for room in rooms:
        covered[room["seed_r"], room["seed_c"]] = True
    assert np.array_equal((grid != bg) & (grid != frame_color), covered), "unexpected extra foreground pixels"

    return int(frame_color), rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each colored seed emits four axis-aligned rays whose maximum length equals
    the background area inside its enclosing rectangular room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    bg = _background_color(g)
    frame_color, rooms = _find_rooms(g)

    out = g.copy()
    for room in rooms:
        seed_r = room["seed_r"]
        seed_c = room["seed_c"]
        seed_color = room["seed_color"]

        for dr, dc in DIR4:
            for step in range(1, room["ray_len"] + 1):
                nr = seed_r + dr * step
                nc = seed_c + dc * step
                if not (0 <= nr < height and 0 <= nc < width):
                    break
                if int(g[nr, nc]) == frame_color:
                    break
                if int(out[nr, nc]) == bg:
                    out[nr, nc] = seed_color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
ROOM_COUNT_OPTIONS = [1, 2, 3, 4, 5]
ROOM_COUNT_WEIGHTS = [1, 3, 3, 2, 1]
MAX_ROOM_INNER = 7
ROOM_PAD = 1
PLACEMENT_TRIES = 80
GEN_TRIES = 400


def _room_count_for_size(height: int, width: int) -> int:
    capacity = max(1, (height - 2) * (width - 2) // 18)
    options = [count for count in ROOM_COUNT_OPTIONS if count <= capacity]
    weights = [ROOM_COUNT_WEIGHTS[ROOM_COUNT_OPTIONS.index(count)] for count in options]
    return random.choices(options, weights=weights, k=1)[0]


def _sample_inner_size(height: int, width: int) -> tuple[int, int] | None:
    max_inner_h = min(MAX_ROOM_INNER, height - 4)
    max_inner_w = min(MAX_ROOM_INNER, width - 4)
    if max_inner_h < 1 or max_inner_w < 1:
        return None

    choices: list[tuple[int, int]] = []
    weights: list[int] = []
    for inner_h in range(1, max_inner_h + 1):
        for inner_w in range(1, max_inner_w + 1):
            area = inner_h * inner_w
            if area > 20:
                continue
            weight = 2
            if inner_h == 1 or inner_w == 1:
                weight += 2
            if area <= 9:
                weight += 2
            choices.append((inner_h, inner_w))
            weights.append(weight)
    return random.choices(choices, weights=weights, k=1)[0] if choices else None


def _can_place(occupied: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int) -> bool:
    height, width = occupied.shape
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    rr0 = max(0, r0 - ROOM_PAD)
    cc0 = max(0, c0 - ROOM_PAD)
    rr1 = min(height, r1 + ROOM_PAD + 1)
    cc1 = min(width, c1 + ROOM_PAD + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int) -> None:
    height, width = occupied.shape
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    rr0 = max(0, r0 - ROOM_PAD)
    cc0 = max(0, c0 - ROOM_PAD)
    rr1 = min(height, r1 + ROOM_PAD + 1)
    cc1 = min(width, c1 + ROOM_PAD + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _draw_room(
    grid: np.ndarray,
    *,
    frame_color: int,
    seed_color: int,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    seed_r: int,
    seed_c: int,
) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    grid[seed_r, seed_c] = seed_color


def _generate_once() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    bg = random.choice(PALETTE)
    frame_color = random.choice([color for color in PALETTE if color != bg])

    room_count = _room_count_for_size(height, width)
    occupied = np.zeros((height, width), dtype=bool)
    rooms: list[dict[str, int]] = []

    for _ in range(room_count):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            size = _sample_inner_size(height, width)
            if size is None:
                return None
            inner_h, inner_w = size
            outer_h = inner_h + 2
            outer_w = inner_w + 2

            if outer_h > height - 2 or outer_w > width - 2:
                continue

            r0 = random.randint(1, height - outer_h - 1)
            c0 = random.randint(1, width - outer_w - 1)
            if not _can_place(occupied, r0, c0, outer_h, outer_w):
                continue

            seed_r = r0 + 1 + random.randint(0, inner_h - 1)
            seed_c = c0 + 1 + random.randint(0, inner_w - 1)
            rooms.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "seed_r": seed_r,
                    "seed_c": seed_c,
                }
            )
            _mark_occupied(occupied, r0, c0, outer_h, outer_w)
            placed = True
            break
        if not placed:
            return None

    seed_choices = [color for color in PALETTE if color not in (bg, frame_color)]
    palette_size = random.randint(1, min(4, len(seed_choices), room_count))
    seed_palette = random.sample(seed_choices, palette_size)

    grid = np.full((height, width), bg, dtype=int)
    for room in rooms:
        seed_color = random.choice(seed_palette)
        _draw_room(
            grid,
            frame_color=frame_color,
            seed_color=seed_color,
            r0=room["r0"],
            c0=room["c0"],
            inner_h=room["inner_h"],
            inner_w=room["inner_w"],
            seed_r=room["seed_r"],
            seed_c=room["seed_c"],
        )

    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != bg:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-ray puzzle")
