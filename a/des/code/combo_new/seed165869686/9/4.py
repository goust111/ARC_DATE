import random
from collections import Counter

import numpy as np


DIR_VECS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

DIR_TURNS = {"up": 0, "right": 1, "down": 2, "left": 3}

# Base shapes are defined with the tail at (0, 0) and the sprite body extending
# away from the top border, so the tail direction is "up" before rotation.
BASE_SHAPES = [
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (1, -1), (1, 0)},
    {(0, 0), (1, -1), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (1, 0), (2, -1), (2, 0)},
    {(0, 0), (1, 0), (2, -1), (2, 0), (2, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
    {(0, 0), (1, 0), (2, -2), (2, -1), (2, 0)},
]

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 200
PALETTE = list(range(10))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _rotate_cw(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(c, -r) for r, c in coords}


def _rotate_shape(coords: set[tuple[int, int]], turns: int) -> set[tuple[int, int]]:
    out = set(coords)
    for _ in range(turns % 4):
        out = _rotate_cw(out)
    return out


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        cell = stack.pop()
        for nxt in _neighbors4(cell):
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _tail_from_side(sprite_cells: set[tuple[int, int]], side: str) -> tuple[int, int]:
    rows = [r for r, _ in sprite_cells]
    cols = [c for _, c in sprite_cells]
    if side == "up":
        edge = min(rows)
        candidates = [cell for cell in sprite_cells if cell[0] == edge]
    elif side == "down":
        edge = max(rows)
        candidates = [cell for cell in sprite_cells if cell[0] == edge]
    elif side == "left":
        edge = min(cols)
        candidates = [cell for cell in sprite_cells if cell[1] == edge]
    else:
        edge = max(cols)
        candidates = [cell for cell in sprite_cells if cell[1] == edge]

    assert len(candidates) == 1, "expected a unique tail pixel on the marker-facing side"
    tail = candidates[0]
    degree = sum(nei in sprite_cells for nei in _neighbors4(tail))
    assert degree == 1, "tail must be a one-pixel protrusion"
    return tail


def _border_side(marker: tuple[int, int], h: int, w: int) -> str:
    r, c = marker
    sides = []
    if r == 0:
        sides.append("up")
    if r == h - 1:
        sides.append("down")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, "marker must sit on exactly one border side"
    return sides[0]


def _line_between(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored sprite along its tail direction until it reaches its matching border marker, drawing the tail trail behind it.

    Input structure:
    - The background is the dominant color.
    - Each non-background color appears as exactly one small monochrome interior sprite plus one same-color single-pixel marker on the border.
    - The sprite is a 4-connected L-like or T-like shape with 3 to 5 pixels and has a unique protruding tail on the side that faces its marker.
    - The marker lies on one non-corner border cell, and the sprite's tail points directly toward that border.

    Rule:
    - For each color, separate the unique border marker from the interior sprite.
    - Use the marker's border side to identify the motion direction, then locate the sprite's tail as the unique extreme pixel on that side.
    - Translate the whole sprite along that direction until the tail becomes orthogonally adjacent to the marker.
    - Clear the original sprite location, keep the border marker, and paint the straight tail trajectory from the original tail position to the new tail position with the sprite color.

    Key insight:
    - Color gives the destination border marker, while the sprite geometry identifies which pixel is the moving tail that must stop next to that marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h >= 4 and w >= 4

    bg = get_bg(g)
    out = np.full_like(g, int(bg))

    for color in sorted(int(x) for x in np.unique(g) if int(x) != int(bg)):
        coords = {(int(r), int(c)) for r, c in np.argwhere(g == color)}
        border_cells = {cell for cell in coords if cell[0] in (0, h - 1) or cell[1] in (0, w - 1)}
        assert len(border_cells) == 1, "each color must have exactly one border marker"
        marker = next(iter(border_cells))
        sprite = coords - border_cells
        assert 3 <= len(sprite) <= 5, "sprite must have 3 to 5 pixels"
        assert _is_connected(sprite), "sprite must be 4-connected"

        side = _border_side(marker, h, w)
        tail = _tail_from_side(sprite, side)
        move_r, move_c = DIR_VECS[side]
        assert marker[0] == tail[0] if side in ("left", "right") else marker[1] == tail[1]

        new_tail = (marker[0] - move_r, marker[1] - move_c)
        delta_r = new_tail[0] - tail[0]
        delta_c = new_tail[1] - tail[1]
        moved = {(r + delta_r, c + delta_c) for r, c in sprite}
        assert all(0 <= r < h and 0 <= c < w for r, c in moved)

        path = _line_between(tail, new_tail)
        for r, c in path:
            assert out[r, c] == bg
            out[r, c] = color
        for r, c in moved:
            assert out[r, c] in (bg, color)
            out[r, c] = color
        mr, mc = marker
        assert out[mr, mc] == bg
        out[mr, mc] = color

    return out


def _margin(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _shift(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _touches_border(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)


def _max_slide_distance(final_cells: set[tuple[int, int]], direction: str, h: int, w: int) -> int:
    dr, dc = DIR_VECS[direction]
    distance = 0
    while True:
        trial = _shift(final_cells, -dr * (distance + 1), -dc * (distance + 1))
        if not _in_bounds(trial, h, w):
            return distance
        distance += 1


def _candidate_marker(direction: str, h: int, w: int) -> tuple[int, int]:
    if direction == "up":
        return 0, random.randint(1, w - 2)
    if direction == "down":
        return h - 1, random.randint(1, w - 2)
    if direction == "left":
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _place_one(
    h: int,
    w: int,
    direction: str,
    occupied_input: set[tuple[int, int]],
    reserved_corridors: set[tuple[int, int]],
    forbidden_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
) -> dict | None:
    turns = DIR_TURNS[direction]
    dr, dc = DIR_VECS[direction]

    for _ in range(PLACE_ATTEMPTS):
        shape = _rotate_shape(random.choice(BASE_SHAPES), turns)
        marker = _candidate_marker(direction, h, w)
        final_tail = (marker[0] - dr, marker[1] - dc)
        final_cells = _shift(shape, final_tail[0], final_tail[1])
        if not _in_bounds(final_cells, h, w):
            continue

        max_dist = _max_slide_distance(final_cells, direction, h, w)
        if max_dist < 2:
            continue
        dist = random.randint(2, max_dist)
        input_cells = _shift(final_cells, -dr * dist, -dc * dist)
        input_tail = (final_tail[0] - dr * dist, final_tail[1] - dc * dist)
        path = set(_line_between(input_tail, final_tail))
        corridor = path | {marker}
        output_cells = path | final_cells | {marker}

        if _touches_border(input_cells, h, w):
            continue
        if path & (input_cells - {input_tail}):
            continue
        if (input_cells | {marker}) & forbidden_input:
            continue
        if input_cells & reserved_corridors:
            continue
        if corridor & occupied_input:
            continue
        if output_cells & occupied_output:
            continue

        return {
            "marker": marker,
            "input_cells": input_cells,
            "output_cells": output_cells,
            "corridor": corridor,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(2, 4)
    min_side = 9 + 2 * n_objects
    max_side = min(22, min_side + 5)
    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)
    bg = random.choice(PALETTE)
    colors = random.sample([c for c in PALETTE if c != bg], n_objects)
    directions = random.sample(list(DIR_VECS), n_objects)
    random.shuffle(colors)

    grid = np.full((h, w), bg, dtype=int)
    occupied_input: set[tuple[int, int]] = set()
    reserved_corridors: set[tuple[int, int]] = set()
    forbidden_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()

    for color, direction in zip(colors, directions):
        placement = _place_one(h, w, direction, occupied_input, reserved_corridors, forbidden_input, occupied_output)
        if placement is None:
            return None

        for r, c in placement["input_cells"]:
            grid[r, c] = color
        mr, mc = placement["marker"]
        grid[mr, mc] = color

        occupied_input |= placement["input_cells"] | {placement["marker"]}
        reserved_corridors |= placement["corridor"]
        forbidden_input |= _margin(placement["input_cells"] | {placement["marker"]}, h, w)
        occupied_output |= placement["output_cells"]

    counts = Counter(map(int, grid.ravel()))
    if counts[bg] <= max(v for k, v in counts.items() if k != bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-tail sprite input")
