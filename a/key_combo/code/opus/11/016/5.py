import random
from collections import deque

import numpy as np


BG = 5
WALL = 0
MOVABLE = 2
GREEN = 3
BLUE = 1
ORANGE = 7

FRAME_OUTER = 8
ROOM_SIZE = 6
LEGEND_SIZE = 2
MAX_GENERATION_ATTEMPTS = 500

DIR_RIGHT = (0, 1)
DIR_LEFT = (0, -1)
DIR_DOWN = (1, 0)
DIR_UP = (-1, 0)
ALL_DIRS = (DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT)

LEGEND_TO_DIR = {
    ((BG, BG), (WALL, WALL)): DIR_RIGHT,
    ((WALL, WALL), (BG, BG)): DIR_LEFT,
    ((BG, WALL), (BG, WALL)): DIR_DOWN,
    ((WALL, BG), (WALL, BG)): DIR_UP,
}

SAFE_CELLS = [(r, c) for r in range(1, 5) for c in range(1, 5)]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _find_frame(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - FRAME_OUTER + 1):
        for c0 in range(w - FRAME_OUTER + 1):
            patch = grid[r0 : r0 + FRAME_OUTER, c0 : c0 + FRAME_OUTER]
            if not np.all(patch[0, :] == WALL):
                continue
            if not np.all(patch[-1, :] == WALL):
                continue
            if not np.all(patch[:, 0] == WALL):
                continue
            if not np.all(patch[:, -1] == WALL):
                continue
            if np.any(patch[1:-1, 1:-1] == WALL):
                continue
            candidates.append((r0, c0))
    assert len(candidates) == 1, f"expected one {FRAME_OUTER}x{FRAME_OUTER} hollow frame, got {len(candidates)}"
    return candidates[0]


def _decode_legend(grid: np.ndarray) -> tuple[int, int]:
    patch = tuple(tuple(int(v) for v in row) for row in grid[:LEGEND_SIZE, -LEGEND_SIZE:])
    assert patch in LEGEND_TO_DIR, "unrecognized 2x2 legend pattern"
    return LEGEND_TO_DIR[patch]


def _compress_movers(red_mask: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    out = np.zeros_like(red_mask, dtype=bool)
    dr, dc = direction
    if (dr, dc) == DIR_DOWN:
        counts = red_mask.sum(axis=0)
        for c, count in enumerate(counts):
            count = int(count)
            if count:
                out[ROOM_SIZE - count :, c] = True
        return out
    if (dr, dc) == DIR_UP:
        counts = red_mask.sum(axis=0)
        for c, count in enumerate(counts):
            count = int(count)
            if count:
                out[:count, c] = True
        return out
    if (dr, dc) == DIR_RIGHT:
        counts = red_mask.sum(axis=1)
        for r, count in enumerate(counts):
            count = int(count)
            if count:
                out[r, ROOM_SIZE - count :] = True
        return out
    if (dr, dc) == DIR_LEFT:
        counts = red_mask.sum(axis=1)
        for r, count in enumerate(counts):
            count = int(count)
            if count:
                out[r, :count] = True
        return out
    raise AssertionError("direction must be cardinal")


def _recolor_components(red_mask: np.ndarray) -> np.ndarray:
    out = np.full((ROOM_SIZE, ROOM_SIZE), BG, dtype=int)
    for comp in _components(red_mask):
        area = len(comp)
        color = GREEN if area == 1 else BLUE if area == 2 else ORANGE
        for r, c in comp:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the red pixels inside the framed room according to the corner legend, then recolor each final cluster by area.

    Input structure:
    - The grid has a gray background plus one black hollow frame enclosing a 6x6 room.
    - A small 2x2 black legend sits in the top-right corner and contains exactly two gray cells.
    - Red movable pixels are inside the room, are not adjacent to the frame wall, and are pairwise non-adjacent in the input.

    Rule:
    - Decode the 2x2 legend by the placement of its two gray cells: top row = move right, bottom row = move left, left column = move down, right column = move up.
    - Move every red pixel through the room in that direction until it would hit the black frame; pixels stack in the same row or column as they compress.
    - Find the 4-connected red clusters after the move and recolor them by area: size 1 -> green, size 2 -> blue, size >= 3 -> orange.
    - Keep the frame, the gray background, and the legend unchanged.

    Key insight:
    - The legend chooses which room axis is compressed, and that compression changes which red pixels become connected, so the final color depends on the post-move cluster sizes rather than the input positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    frame_r0, frame_c0 = _find_frame(g)
    direction = _decode_legend(g)

    room = g[frame_r0 + 1 : frame_r0 + 1 + ROOM_SIZE, frame_c0 + 1 : frame_c0 + 1 + ROOM_SIZE]
    red_mask = room == MOVABLE
    final_red = _compress_movers(red_mask, direction)
    recolored_room = _recolor_components(final_red)

    out = g.copy()
    out_room = out[frame_r0 + 1 : frame_r0 + 1 + ROOM_SIZE, frame_c0 + 1 : frame_c0 + 1 + ROOM_SIZE]
    out_room[:] = BG
    out_room[recolored_room != BG] = recolored_room[recolored_room != BG]
    return out


def _nonadjacent(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set:
                return False
    return True


def _build_safe_configs() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    by_size = {n: [] for n in range(4, 9)}
    n_cells = len(SAFE_CELLS)
    for mask in range(1 << n_cells):
        size = mask.bit_count()
        if size < 4 or size > 8:
            continue
        cells = tuple(SAFE_CELLS[i] for i in range(n_cells) if (mask >> i) & 1)
        if _nonadjacent(cells):
            by_size[size].append(cells)
    return by_size


SAFE_CONFIGS_BY_SIZE = _build_safe_configs()


def _sizes_for_config(cells: tuple[tuple[int, int], ...], direction: tuple[int, int]) -> list[int]:
    mask = np.zeros((ROOM_SIZE, ROOM_SIZE), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    final_mask = _compress_movers(mask, direction)
    return sorted(len(comp) for comp in _components(final_mask))


def _branch_ok(sizes: list[int], branch: int) -> bool:
    has1 = 1 in sizes
    has2 = 2 in sizes
    has3 = any(size >= 3 for size in sizes)
    if len(sizes) < 2:
        return False
    if branch == 0:
        return has1 and has2
    if branch == 1:
        return has1 and has3
    if branch == 2:
        return has2 and has3
    return has1 and has2 and has3


def _sample_start_config(rng: random.Random, direction: tuple[int, int]) -> tuple[tuple[int, int], ...] | None:
    branch = rng.randrange(4)
    counts = list(range(4, 9))
    rng.shuffle(counts)
    fallback: tuple[tuple[int, int], ...] | None = None
    for count in counts:
        configs = SAFE_CONFIGS_BY_SIZE[count][:]
        rng.shuffle(configs)
        for cells in configs:
            sizes = _sizes_for_config(cells, direction)
            palette = {GREEN if s == 1 else BLUE if s == 2 else ORANGE for s in sizes}
            if len(palette) < 2:
                continue
            if max(sizes) == sum(sizes):
                continue
            if _branch_ok(sizes, branch):
                return cells
            if fallback is None and len(sizes) >= 2:
                fallback = cells
    return fallback


def _legend_patch(direction: tuple[int, int]) -> np.ndarray:
    for pattern, vec in LEGEND_TO_DIR.items():
        if vec == direction:
            return np.array(pattern, dtype=int)
    raise AssertionError("unknown direction")


def _valid_frame_positions(h: int, w: int) -> list[tuple[int, int]]:
    positions = []
    legend_rows = {0, 1}
    legend_cols = {w - 2, w - 1}
    for r0 in range(h - FRAME_OUTER + 1):
        for c0 in range(w - FRAME_OUTER + 1):
            frame_rows = set(range(r0, r0 + FRAME_OUTER))
            frame_cols = set(range(c0, c0 + FRAME_OUTER))
            if frame_rows & legend_rows and frame_cols & legend_cols:
                continue
            positions.append((r0, c0))
    return positions


def _build_input_grid(
    h: int,
    w: int,
    frame_pos: tuple[int, int],
    start_cells: tuple[tuple[int, int], ...],
    direction: tuple[int, int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    r0, c0 = frame_pos

    grid[r0, c0 : c0 + FRAME_OUTER] = WALL
    grid[r0 + FRAME_OUTER - 1, c0 : c0 + FRAME_OUTER] = WALL
    grid[r0 : r0 + FRAME_OUTER, c0] = WALL
    grid[r0 : r0 + FRAME_OUTER, c0 + FRAME_OUTER - 1] = WALL

    grid[:LEGEND_SIZE, -LEGEND_SIZE:] = _legend_patch(direction)

    for rr, cc in start_cells:
        grid[r0 + 1 + rr, c0 + 1 + cc] = MOVABLE
    return grid


def _generator_accepts(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    if np.array_equal(out, grid):
        return False
    colors = set(int(v) for v in np.unique(out))
    if len(colors & {GREEN, BLUE, ORANGE}) < 2:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(10, 13)
    w = rng.randint(10, 13)
    frame_positions = _valid_frame_positions(h, w)
    if not frame_positions:
        return None

    direction = rng.choice(ALL_DIRS)
    start_cells = _sample_start_config(rng, direction)
    if start_cells is None:
        return None

    rng.shuffle(frame_positions)
    for frame_pos in frame_positions:
        grid = _build_input_grid(h, w, frame_pos, start_cells, direction)
        if _generator_accepts(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
