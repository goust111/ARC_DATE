import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
BLOCK_SIZE = 3
BLOCK_COLORS = (1, 2, 3, 6, 7, 8, 9)
SIDE_TO_DIR = {
    "top": (1, 0),
    "right": (0, -1),
    "bottom": (-1, 0),
    "left": (0, 1),
}
GEN_ATTEMPTS = 500


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _clockwise_border_cells(h: int, w: int) -> list[tuple[int, int, str]]:
    cells: list[tuple[int, int, str]] = []
    for c in range(w):
        cells.append((0, c, "top"))
    for r in range(1, h):
        cells.append((r, w - 1, "right"))
    for c in range(w - 2, -1, -1):
        cells.append((h - 1, c, "bottom"))
    for r in range(h - 2, 0, -1):
        cells.append((r, 0, "left"))
    return cells


def _read_instructions(grid: np.ndarray) -> list[tuple[int, tuple[int, int]]]:
    h, w = grid.shape
    instructions: list[tuple[int, tuple[int, int]]] = []
    for r, c, side in _clockwise_border_cells(h, w):
        color = int(grid[r, c])
        if color == GREY:
            continue
        instructions.append((color, SIDE_TO_DIR[side]))
    return instructions


def _find_blocks(grid: np.ndarray, colors: list[int]) -> dict[int, tuple[int, int]]:
    inner = grid[1:-1, 1:-1]
    blocks: dict[int, tuple[int, int]] = {}
    for color in colors:
        pts = np.argwhere(inner == int(color))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        blocks[int(color)] = (int(r0 + 1), int(c0 + 1))
    return blocks


def _move_one(
    blocks: dict[int, tuple[int, int]],
    color: int,
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> int:
    h, w = shape
    r, c = blocks[int(color)]
    dr, dc = direction

    # Contact is measured on occupied rows/cols, so a stopped move can create a 1-cell overlap strip.
    if dr == 1:
        distance = (h - 2) - (r + BLOCK_SIZE - 1)
        for other_color, (orow, ocol) in blocks.items():
            if int(other_color) == int(color):
                continue
            if orow <= r:
                continue
            if not _intervals_overlap(c, c + BLOCK_SIZE - 1, ocol, ocol + BLOCK_SIZE - 1):
                continue
            gap = int(orow - (r + BLOCK_SIZE - 1))
            if 0 < gap < distance:
                distance = gap
        blocks[int(color)] = (int(r + distance), int(c))
        return int(distance)

    if dr == -1:
        distance = r - 1
        for other_color, (orow, ocol) in blocks.items():
            if int(other_color) == int(color):
                continue
            if orow >= r:
                continue
            if not _intervals_overlap(c, c + BLOCK_SIZE - 1, ocol, ocol + BLOCK_SIZE - 1):
                continue
            gap = int(r - (orow + BLOCK_SIZE - 1))
            if 0 < gap < distance:
                distance = gap
        blocks[int(color)] = (int(r - distance), int(c))
        return int(distance)

    if dc == 1:
        distance = (w - 2) - (c + BLOCK_SIZE - 1)
        for other_color, (orow, ocol) in blocks.items():
            if int(other_color) == int(color):
                continue
            if ocol <= c:
                continue
            if not _intervals_overlap(r, r + BLOCK_SIZE - 1, orow, orow + BLOCK_SIZE - 1):
                continue
            gap = int(ocol - (c + BLOCK_SIZE - 1))
            if 0 < gap < distance:
                distance = gap
        blocks[int(color)] = (int(r), int(c + distance))
        return int(distance)

    distance = c - 1
    for other_color, (orow, ocol) in blocks.items():
        if int(other_color) == int(color):
            continue
        if ocol >= c:
            continue
        if not _intervals_overlap(r, r + BLOCK_SIZE - 1, orow, orow + BLOCK_SIZE - 1):
            continue
        gap = int(c - (ocol + BLOCK_SIZE - 1))
        if 0 < gap < distance:
            distance = gap
    blocks[int(color)] = (int(r), int(c - distance))
    return int(distance)


def _simulate(
    blocks: dict[int, tuple[int, int]],
    instructions: list[tuple[int, tuple[int, int]]],
    shape: tuple[int, int],
) -> tuple[dict[int, tuple[int, int]], list[int]]:
    state = {int(color): (int(r), int(c)) for color, (r, c) in blocks.items()}
    distances: list[int] = []
    for color, direction in instructions:
        distances.append(_move_one(state, int(color), direction, shape))
    return state, distances


def _occupancy_counts(shape: tuple[int, int], blocks: dict[int, tuple[int, int]]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for r, c in blocks.values():
        counts[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] += 1
    return counts


def _render(shape: tuple[int, int], blocks: dict[int, tuple[int, int]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[0, :] = GREY
    out[-1, :] = GREY
    out[:, 0] = GREY
    out[:, -1] = GREY

    counts = np.zeros(shape, dtype=int)
    first_color = np.full(shape, -1, dtype=int)
    for color, (r, c) in blocks.items():
        patch_counts = counts[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE]
        patch_first = first_color[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE]
        fresh = patch_counts == 0
        patch_first[fresh] = int(color)
        patch_counts += 1

    occupied = counts > 0
    out[occupied] = first_color[occupied]
    out[counts > 1] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read colored frame gaps clockwise, move the matching 3x3 blocks, and paint overlaps yellow.
    """
    g = np.asarray(grid, dtype=int)
    instructions = _read_instructions(g)
    colors = [int(color) for color, _ in instructions]
    blocks = _find_blocks(g, colors)
    final_blocks, _ = _simulate(blocks, instructions, g.shape)
    return _render(g.shape, final_blocks)


def _boxes_too_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    return not (
        ar + BLOCK_SIZE - 1 < br - 1
        or br + BLOCK_SIZE - 1 < ar - 1
        or ac + BLOCK_SIZE - 1 < bc - 1
        or bc + BLOCK_SIZE - 1 < ac - 1
    )


def _choose_sides(count: int, rng: random.Random) -> list[str]:
    sides: list[str] = []
    base = ["top", "right", "bottom", "left"]
    while len(sides) < count:
        cycle = base[:]
        rng.shuffle(cycle)
        sides.extend(cycle)
    return sides[:count]


def _sample_edge_positions(span: int, count: int, rng: random.Random) -> list[int] | None:
    candidates = list(range(1, span - 1))
    for _ in range(80):
        rng.shuffle(candidates)
        chosen: list[int] = []
        for pos in candidates:
            if all(abs(int(pos) - int(prev)) >= 2 for prev in chosen):
                chosen.append(int(pos))
                if len(chosen) == count:
                    return chosen
    if count <= len(candidates):
        return rng.sample(candidates, count)
    return None


def _place_blocks(
    h: int,
    w: int,
    colors: list[int],
    sides: list[str],
    rng: random.Random,
) -> dict[int, tuple[int, int]] | None:
    blocks: dict[int, tuple[int, int]] = {}
    for color, side in zip(colors, sides):
        r_lo, r_hi = 1, h - BLOCK_SIZE - 1
        c_lo, c_hi = 1, w - BLOCK_SIZE - 1
        if side == "top":
            r_hi -= 1
        elif side == "bottom":
            r_lo += 1
        elif side == "left":
            c_hi -= 1
        else:
            c_lo += 1
        if r_lo > r_hi or c_lo > c_hi:
            return None
        found = False
        for _ in range(120):
            pos = (int(rng.randint(r_lo, r_hi)), int(rng.randint(c_lo, c_hi)))
            if any(_boxes_too_close(pos, other) for other in blocks.values()):
                continue
            blocks[int(color)] = pos
            found = True
            break
        if not found:
            return None
    return blocks


def _build_grid(
    h: int,
    w: int,
    blocks: dict[int, tuple[int, int]],
    gaps: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY
    for color, (r, c) in blocks.items():
        grid[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] = int(color)
    for (r, c), color in gaps.items():
        grid[int(r), int(c)] = int(color)
    return grid


def _make_gaps(
    h: int,
    w: int,
    colors: list[int],
    sides: list[str],
    rng: random.Random,
) -> dict[tuple[int, int], int] | None:
    grouped = {"top": [], "right": [], "bottom": [], "left": []}
    for color, side in zip(colors, sides):
        grouped[side].append(int(color))

    gaps: dict[tuple[int, int], int] = {}
    for side, side_colors in grouped.items():
        if not side_colors:
            continue
        span = w if side in {"top", "bottom"} else h
        positions = _sample_edge_positions(span, len(side_colors), rng)
        if positions is None:
            return None
        rng.shuffle(side_colors)
        for pos, color in zip(positions, side_colors):
            if side == "top":
                gaps[(0, int(pos))] = int(color)
            elif side == "bottom":
                gaps[(h - 1, int(pos))] = int(color)
            elif side == "left":
                gaps[(int(pos), 0)] = int(color)
            else:
                gaps[(int(pos), w - 1)] = int(color)
    return gaps


def _has_overlap(blocks: dict[int, tuple[int, int]], shape: tuple[int, int]) -> bool:
    counts = _occupancy_counts(shape, blocks)
    return bool((counts > 1).any())


def _has_triple_overlap(blocks: dict[int, tuple[int, int]], shape: tuple[int, int]) -> bool:
    counts = _occupancy_counts(shape, blocks)
    return bool((counts > 2).any())


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        h = int(rng.randint(11, 18))
        w = int(rng.randint(11, 18))
        max_blocks = min(6, len(BLOCK_COLORS), max(3, ((h - 2) * (w - 2)) // 20))
        block_count = int(rng.randint(3, max_blocks))
        want_overlap = bool(rng.randint(0, 1))

        colors = rng.sample(BLOCK_COLORS, block_count)
        sides = _choose_sides(block_count, rng)
        blocks = _place_blocks(h, w, colors, sides, rng)
        if blocks is None:
            continue

        gaps = _make_gaps(h, w, colors, sides, rng)
        if gaps is None:
            continue

        grid = _build_grid(h, w, blocks, gaps)
        instructions = _read_instructions(grid)
        if len(instructions) != block_count:
            continue
        if {int(color) for color, _ in instructions} != set(colors):
            continue

        final_blocks, distances = _simulate(blocks, instructions, grid.shape)
        if any(int(distance) <= 0 for distance in distances):
            continue
        if want_overlap != _has_overlap(final_blocks, grid.shape):
            continue
        if _has_triple_overlap(final_blocks, grid.shape):
            continue
        if np.array_equal(grid, _render(grid.shape, final_blocks)):
            continue
        return grid
    raise RuntimeError("generate_input: exhausted attempts")
