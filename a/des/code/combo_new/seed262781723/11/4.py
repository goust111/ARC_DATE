import random
from typing import Iterable

import numpy as np


BG = 0
POINTER_RED = 2
FRAME_GREY = 5
PALETTE = [1, 3, 4, 6, 7, 8, 9]
SIZE_CHOICES = (5, 7)
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_SHIFT = {direction: idx for idx, direction in enumerate(DIRECTIONS)}

GRID_MIN = 22
GRID_MAX = 28
COUNT_MIN = 2
COUNT_MAX = 4
EDGE_MARGIN = 1
PLACEMENT_GAP = 2
GEN_ATTEMPTS = 400

_GEN_CALL_INDEX = 0


def _border_track(r0: int, c0: int, size: int) -> list[tuple[int, int]]:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    track: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        track.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        track.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        track.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        track.append((r, c0))
    return track


def _dial_at(grid: np.ndarray, r0: int, c0: int, size: int) -> dict | None:
    h, w = grid.shape
    if r0 < 0 or c0 < 0 or r0 + size > h or c0 + size > w:
        return None

    r1 = r0 + size - 1
    c1 = c0 + size - 1
    center = (r0 + size // 2, c0 + size // 2)
    hub = int(grid[center])
    if hub in (BG, POINTER_RED, FRAME_GREY):
        return None

    track = _border_track(r0, c0, size)
    markers: list[tuple[int, int]] = []
    for idx, (r, c) in enumerate(track):
        val = int(grid[r, c])
        if val == FRAME_GREY:
            continue
        if val in (BG, POINTER_RED) or val == hub:
            return None
        markers.append((idx, val))
    if not markers:
        return None

    pointer = None
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            val = int(grid[r, c])
            if (r, c) == center:
                continue
            if val == BG:
                continue
            if val == POINTER_RED and abs(r - center[0]) + abs(c - center[1]) == 1:
                if pointer is not None:
                    return None
                pointer = (r, c)
                continue
            return None
    if pointer is None:
        return None

    direction = (pointer[0] - center[0], pointer[1] - center[1])
    if direction not in DIR_TO_SHIFT:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "size": size,
        "center": center,
        "pointer": pointer,
        "shift": DIR_TO_SHIFT[direction],
        "track": track,
        "markers": markers,
    }


def _find_dials(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    found: list[dict] = []
    occupied = np.zeros(g.shape, dtype=bool)
    for size in sorted(SIZE_CHOICES, reverse=True):
        for r0 in range(g.shape[0] - size + 1):
            for c0 in range(g.shape[1] - size + 1):
                dial = _dial_at(g, r0, c0, size)
                if dial is None:
                    continue
                rr0, cc0, rr1, cc1 = dial["bbox"]
                if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
                found.append(dial)
    found.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["size"]))
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each dial's border markers according to the red pointer direction inside the dial.

    Input structure:
    - The background is black, and the grid contains several hollow grey square dials of side length 5 or 7.
    - Each dial border is mostly grey but includes some non-black, non-grey colored tick pixels on the border track.
    - Each dial has a non-black, non-grey hub pixel at its center and exactly one adjacent red pointer pixel inside the hollow interior.

    Rule:
    - Read the pointer direction relative to the hub: up means 0 quarter-turn steps, right means 1 clockwise step, down means 2, and left means 3.
    - Move every colored border tick clockwise along the dial border track by that many quarter-turn steps, where one step equals side length minus one pixels on the perimeter sequence.
    - Restore the pointer pixel to black while keeping the grey frame and the hub pixel unchanged.

    Key insight:
    - The red interior pixel is not decoration; it encodes a 0/90/180/270-degree cyclic shift of the colored border ticks around the square ring.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    dials = _find_dials(g)
    assert dials, "expected at least one dial"

    for dial in dials:
        track = dial["track"]
        for r, c in track:
            out[r, c] = FRAME_GREY
        shift_cells = dial["shift"] * (dial["size"] - 1)
        for idx, color in dial["markers"]:
            nr, nc = track[(idx + shift_cells) % len(track)]
            out[nr, nc] = color
        pr, pc = dial["pointer"]
        out[pr, pc] = BG

    return out


def _sample_directions(rng: random.Random, count: int, required: tuple[int, int]) -> list[tuple[int, int]]:
    directions = rng.sample(DIRECTIONS, k=count)
    if required not in directions:
        directions[rng.randrange(count)] = required
    rng.shuffle(directions)
    return directions


def _reserve(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    h, w = mask.shape
    rr0 = max(0, r0 - PLACEMENT_GAP)
    cc0 = max(0, c0 - PLACEMENT_GAP)
    rr1 = min(h, r1 + PLACEMENT_GAP + 1)
    cc1 = min(w, c1 + PLACEMENT_GAP + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _paint_dial(
    grid: np.ndarray,
    r0: int,
    c0: int,
    size: int,
    direction: tuple[int, int],
    rng: random.Random,
) -> None:
    track = _border_track(r0, c0, size)
    for r, c in track:
        grid[r, c] = FRAME_GREY

    marker_count = rng.randint(2, min(6, len(track) // 3 + 1))
    marker_positions = rng.sample(range(len(track)), k=marker_count)
    colors = rng.sample(PALETTE, k=marker_count + 1)
    marker_colors = colors[:marker_count]
    hub_color = colors[-1]

    for idx, color in zip(marker_positions, marker_colors):
        r, c = track[idx]
        grid[r, c] = color

    center = (r0 + size // 2, c0 + size // 2)
    grid[center] = hub_color
    pr = center[0] + direction[0]
    pc = center[1] + direction[1]
    grid[pr, pc] = POINTER_RED


def _positions_match(dials: Iterable[dict], placed: Iterable[tuple[int, int, int]]) -> bool:
    detected = sorted((d["bbox"][0], d["bbox"][1], d["size"]) for d in dials)
    expected = sorted(placed)
    return detected == expected


def _gen_attempt(required_dir: tuple[int, int], required_size: int, rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    dial_count = rng.randint(COUNT_MIN, COUNT_MAX)

    sizes = [required_size]
    while len(sizes) < dial_count:
        sizes.append(rng.choice(SIZE_CHOICES))
    directions = _sample_directions(rng, dial_count, required_dir)

    specs = list(zip(sizes, directions))
    specs.sort(key=lambda item: item[0], reverse=True)

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    placed: list[tuple[int, int, int]] = []

    for size, direction in specs:
        placed_here = False
        row_min = EDGE_MARGIN
        row_max = h - size - EDGE_MARGIN
        col_min = EDGE_MARGIN
        col_max = w - size - EDGE_MARGIN
        if row_min > row_max or col_min > col_max:
            return None

        for _ in range(160):
            r0 = rng.randint(row_min, row_max)
            c0 = rng.randint(col_min, col_max)
            if reserved[r0 : r0 + size, c0 : c0 + size].any():
                continue
            _paint_dial(grid, r0, c0, size, direction, rng)
            _reserve(reserved, (r0, c0, r0 + size - 1, c0 + size - 1))
            placed.append((r0, c0, size))
            placed_here = True
            break
        if not placed_here:
            return None

    dials = _find_dials(grid)
    if len(dials) != dial_count or not _positions_match(dials, placed):
        return None

    out = main(grid)
    if out.shape != grid.shape or not np.any(out != grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX

    call_idx = _GEN_CALL_INDEX
    required_dir = DIRECTIONS[call_idx % len(DIRECTIONS)]
    required_size = SIZE_CHOICES[call_idx % len(SIZE_CHOICES)]
    rng = random.Random()

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(required_dir, required_size, rng)
        if grid is not None:
            _GEN_CALL_INDEX += 1
            return grid
    raise RuntimeError("failed to generate a valid dial-rotation input")
