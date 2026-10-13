from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
GEN_WALL = 5
GEN_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_GRID_RANGE = (26, 35)
GEN_ROOM_COUNT_CHOICES = (4, 5)
GEN_SHORT_SIDE_CHOICES = (5, 7, 9)
GEN_LONG_EXTRA_CHOICES = (0, 2, 4)
GEN_PLACEMENT_GAP = 2
GEN_PLACEMENT_RETRIES = 200
GEN_ATTEMPTS = 400

RNG = np.random.default_rng()


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _wall_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    non_bg = [(count, color) for color, count in counts.items() if color != BG]
    assert non_bg, "expected at least one non-background color"
    return max(non_bg)[1]


def _door_side(patch: np.ndarray, wall: int) -> str:
    top = [c for c in range(1, patch.shape[1] - 1) if int(patch[0, c]) != wall]
    bottom = [c for c in range(1, patch.shape[1] - 1) if int(patch[-1, c]) != wall]
    left = [r for r in range(1, patch.shape[0] - 1) if int(patch[r, 0]) != wall]
    right = [r for r in range(1, patch.shape[0] - 1) if int(patch[r, -1]) != wall]
    sides = {"N": top, "S": bottom, "W": left, "E": right}
    hits = [side for side, values in sides.items() if len(values) == 1]
    assert len(hits) == 1, "each room must have exactly one single-cell doorway"
    assert sum(len(values) for values in sides.values()) == 1, "unexpected extra border gaps"
    return hits[0]


def _extract_rooms(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int] | np.ndarray | str]]:
    g = np.asarray(grid, dtype=int)
    wall = _wall_color(g)
    rooms: list[dict[str, int | tuple[int, int, int, int] | np.ndarray | str]] = []
    for comp in _connected_components(g == wall):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "room frame must enclose an interior"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert not np.any((patch == wall) & ~border), "wall pixels must stay on the rectangular border"
        side = _door_side(patch, wall)

        interior = patch[1:-1, 1:-1]
        inner_colors = {int(color) for color in np.unique(interior) if int(color) != BG}
        assert len(inner_colors) == 1, "each room must contain seeds of exactly one color"
        seed_color = inner_colors.pop()
        seed_count = int(np.count_nonzero(interior == seed_color))
        assert 1 <= seed_count <= 4, "seed count must be between 1 and 4"
        assert np.all((interior == BG) | (interior == seed_color)), "room interior must contain only background and seeds"

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "door": side,
                "color": seed_color,
                "count": seed_count,
                "interior": interior.copy(),
            }
        )
    rooms.sort(key=lambda room: tuple(room["bbox"]))  # deterministic painting order
    return rooms


def _anchor_and_basis(height: int, width: int, door: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if door == "N":
        return (height - 1, width // 2), (-1, 0), (0, 1)
    if door == "S":
        return (0, width // 2), (1, 0), (0, -1)
    if door == "W":
        return (height // 2, width - 1), (0, -1), (-1, 0)
    if door == "E":
        return (height // 2, 0), (0, 1), (1, 0)
    raise AssertionError(f"unknown doorway side: {door}")


def _shape_cells(height: int, width: int, seed_count: int, door: str) -> set[tuple[int, int]]:
    size = max(1, min(height, width) // 2)
    anchor, forward, right = _anchor_and_basis(height, width, door)
    cells: set[tuple[int, int]] = set()

    def add(base_r: int, base_c: int, a: int, b: int) -> None:
        r = base_r + a * forward[0] + b * right[0]
        c = base_c + a * forward[1] + b * right[1]
        if 0 <= r < height and 0 <= c < width:
            cells.add((r, c))

    if seed_count == 3:
        for step in range(size + 1):
            add(anchor[0], anchor[1], step, 0)
            add(anchor[0], anchor[1], 0, step)
        return cells

    center_r = anchor[0] + size * forward[0]
    center_c = anchor[1] + size * forward[1]

    if seed_count == 1:
        for a in range(-size, size + 1):
            span = size - abs(a)
            for b in range(-span, span + 1):
                add(center_r, center_c, a, b)
    elif seed_count == 2:
        for a in range(-size, size + 1):
            add(center_r, center_c, a, 0)
        for b in range(-size, size + 1):
            add(center_r, center_c, 0, b)
    elif seed_count == 4:
        for a in range(-size, size + 1):
            for b in range(-size, size + 1):
                add(center_r, center_c, a, b)
    else:
        raise AssertionError(f"unsupported seed count: {seed_count}")

    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a room-specific figure from the wall opposite each doorway.

    Input structure:
    - Black background with several gray rectangular room frames.
    - Each room has one single-cell doorway on exactly one wall.
    - Inside each room, 1-4 same-colored seed pixels indicate which figure to draw.

    Rule:
    - Seed count 1 -> solid diamond.
    - Seed count 2 -> cross.
    - Seed count 3 -> L shape.
    - Seed count 4 -> solid square.
    - Figure size is half of the room's shorter interior side, floored.
    - The figure is anchored against the wall opposite the doorway and grows toward the doorway.
    - The L shape uses a fixed clockwise elbow relative to the growth direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for room in _extract_rooms(g):
        r0, c0, r1, c1 = room["bbox"]
        color = int(room["color"])
        count = int(room["count"])
        door = str(room["door"])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        h, w = interior.shape
        for rr, cc in _shape_cells(h, w, count, door):
            interior[rr, cc] = color
    return out


def _sample_room_dims() -> tuple[int, int]:
    short_side = int(RNG.choice(GEN_SHORT_SIDE_CHOICES))
    long_side = short_side + int(RNG.choice(GEN_LONG_EXTRA_CHOICES))
    if RNG.random() < 0.5:
        return short_side, long_side
    return long_side, short_side


def _can_place(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - GEN_PLACEMENT_GAP)
    cc0 = max(0, c0 - GEN_PLACEMENT_GAP)
    rr1 = min(h, r0 + height + GEN_PLACEMENT_GAP)
    cc1 = min(w, c0 + width + GEN_PLACEMENT_GAP)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _draw_room(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    door: str,
    color: int,
    seed_count: int,
) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    grid[top, left : right + 1] = GEN_WALL
    grid[bottom, left : right + 1] = GEN_WALL
    grid[top : bottom + 1, left] = GEN_WALL
    grid[top : bottom + 1, right] = GEN_WALL

    if door == "N":
        grid[top, left + 1 + inner_w // 2] = BG
    elif door == "S":
        grid[bottom, left + 1 + inner_w // 2] = BG
    elif door == "W":
        grid[top + 1 + inner_h // 2, left] = BG
    elif door == "E":
        grid[top + 1 + inner_h // 2, right] = BG
    else:
        raise AssertionError(f"unknown doorway side: {door}")

    positions = [(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)]
    chosen = RNG.choice(len(positions), size=seed_count, replace=False)
    for index in np.atleast_1d(chosen):
        r, c = positions[int(index)]
        grid[r, c] = color


def _validate_generated(grid: np.ndarray) -> bool:
    rooms = _extract_rooms(grid)
    if len(rooms) < 4:
        return False
    if {int(room["count"]) for room in rooms} != {1, 2, 3, 4}:
        return False
    if {str(room["door"]) for room in rooms} != {"N", "S", "E", "W"}:
        return False
    out = main(grid)
    if not np.any(out != grid):
        return False
    wall = _wall_color(grid)
    return bool(np.all(out[grid == wall] == wall))


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(*GEN_GRID_RANGE))
    width = int(RNG.integers(*GEN_GRID_RANGE))
    room_count = int(RNG.choice(GEN_ROOM_COUNT_CHOICES))
    if room_count > len(GEN_PALETTE):
        return None

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    counts = [1, 2, 3, 4]
    while len(counts) < room_count:
        counts.append(int(RNG.integers(1, 5)))
    RNG.shuffle(counts)

    doors = ["N", "S", "E", "W"]
    while len(doors) < room_count:
        doors.append(str(RNG.choice(["N", "S", "E", "W"])))
    RNG.shuffle(doors)

    colors = list(RNG.choice(np.array(GEN_PALETTE, dtype=int), size=room_count, replace=False))

    specs = []
    for i in range(room_count):
        inner_h, inner_w = _sample_room_dims()
        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "count": counts[i],
                "door": doors[i],
                "color": int(colors[i]),
            }
        )

    for spec in sorted(specs, key=lambda item: item["outer_h"] * item["outer_w"], reverse=True):
        placed = False
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        if height - outer_h <= 1 or width - outer_w <= 1:
            return None
        for _ in range(GEN_PLACEMENT_RETRIES):
            top = int(RNG.integers(1, height - outer_h))
            left = int(RNG.integers(1, width - outer_w))
            if not _can_place(occupied, top, left, outer_h, outer_w):
                continue
            _draw_room(
                grid,
                top,
                left,
                int(spec["inner_h"]),
                int(spec["inner_w"]),
                str(spec["door"]),
                int(spec["color"]),
                int(spec["count"]),
            )
            occupied[top : top + outer_h, left : left + outer_w] = True
            placed = True
            break
        if not placed:
            return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
