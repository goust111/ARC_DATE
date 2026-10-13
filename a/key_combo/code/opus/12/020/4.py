import random
from collections import deque

import numpy as np


BG = 0
FILL = 2
WALL = 5

GEN_ATTEMPTS = 200
PATCH_SIZE = 4

_GEN_COUNTER = 0


def _mirror_expand(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return np.block([[np.flipud(np.fliplr(g)), np.flipud(g)], [np.fliplr(g), g]])


def _flood_fill_from_seeds(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    q: deque[tuple[int, int]] = deque()
    seen = np.zeros((h, w), dtype=bool)

    for r, c in np.argwhere(out == FILL):
        rr, cc = int(r), int(c)
        q.append((rr, cc))
        seen[rr, cc] = True

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if seen[nr, nc] or out[nr, nc] == WALL:
                continue
            if out[nr, nc] not in (BG, FILL):
                continue
            seen[nr, nc] = True
            if out[nr, nc] == BG:
                out[nr, nc] = FILL
            q.append((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror a partial wall pattern into four quadrants, then flood-fill from the mirrored seeds.

    Input structure:
    - The grid is a small black background containing one grey wall line and one red seed pixel.
    - The wall is a short L/U-like partial enclosure; the seed sits in the half-enclosed side of that wall.
    - The pattern is interpreted as the lower-right source quadrant of a larger mirrored construction.

    Rule:
    - First perform the four-way mirror expansion used in ARC task 833dafe3:
      lower-right = input, lower-left = left-right reflection, upper-right = top-bottom reflection,
      upper-left = reflection across both axes.
    - Then treat every red pixel in the mirrored output as a flood-fill seed and spread red through
      4-connected black cells only; grey wall cells block the fill and stay unchanged.
    - Regions that become closed only after mirroring are filled completely, while still-open regions
      keep expanding until they reach the border of their mirrored background component.

    Key insight:
    - The mirror step changes the topology first, and the flood fill only reveals that new topology.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _flood_fill_from_seeds(_mirror_expand(g))


BASE_WALL = ((0, 2), (1, 2), (2, 2), (3, 0), (3, 1), (3, 2))
BASE_SEED_CLOSED = (1, 1)
BASE_SEED_OPEN = (2, 3)


def _transpose_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((c, r) for r, c in coords)


def _transpose_coord(coord: tuple[int, int]) -> tuple[int, int]:
    return coord[1], coord[0]


def _rot180_coord(coord: tuple[int, int], size: int = PATCH_SIZE) -> tuple[int, int]:
    return size - 1 - coord[0], size - 1 - coord[1]


def _rot180_coords(coords: tuple[tuple[int, int], ...], size: int = PATCH_SIZE) -> tuple[tuple[int, int], ...]:
    return tuple(_rot180_coord(coord, size=size) for coord in coords)


def _component_from_seed(grid: np.ndarray, start: tuple[int, int]) -> tuple[set[tuple[int, int]], bool]:
    h, w = grid.shape
    q: deque[tuple[int, int]] = deque([start])
    cells = {start}
    touches_border = False

    while q:
        r, c = q.popleft()
        if r in (0, h - 1) or c in (0, w - 1):
            touches_border = True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cells or grid[nr, nc] not in (BG, FILL):
                continue
            cells.add((nr, nc))
            q.append((nr, nc))
    return cells, touches_border


def _seed_components(grid: np.ndarray) -> list[tuple[int, bool]]:
    seen = set()
    out: list[tuple[int, bool]] = []
    for r, c in np.argwhere(grid == FILL):
        start = (int(r), int(c))
        if start in seen:
            continue
        cells, touches_border = _component_from_seed(grid, start)
        seen |= cells
        out.append((len(cells), touches_border))
    return out


def _wall_component_size(grid: np.ndarray) -> int:
    wall_cells = np.argwhere(grid == WALL)
    assert wall_cells.size > 0
    start = (int(wall_cells[0, 0]), int(wall_cells[0, 1]))
    h, w = grid.shape
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen or grid[nr, nc] != WALL:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return len(seen)


def _build_input(height: int, width: int, *, mode: str, variant: str) -> np.ndarray:
    assert 5 <= height <= 9 and 5 <= width <= 9

    wall = BASE_WALL
    seed_closed = BASE_SEED_CLOSED
    seed_open = BASE_SEED_OPEN
    if variant == "transpose":
        wall = _transpose_coords(wall)
        seed_closed = _transpose_coord(seed_closed)
        seed_open = _transpose_coord(seed_open)

    grid = np.zeros((height, width), dtype=int)
    if mode == "closed":
        wall_coords = wall
        seed = seed_closed
        top, left = 0, 0
    else:
        wall_coords = _rot180_coords(wall)
        seed = _rot180_coord(seed_open)
        top, left = height - PATCH_SIZE, width - PATCH_SIZE

    for r, c in wall_coords:
        grid[top + r, left + c] = WALL
    grid[top + seed[0], left + seed[1]] = FILL
    return grid


def _semantic_ok(inp: np.ndarray, mode: str) -> bool:
    if inp.shape[0] < PATCH_SIZE or inp.shape[1] < PATCH_SIZE:
        return False
    if set(map(int, np.unique(inp))) != {BG, FILL, WALL}:
        return False
    if int((inp == FILL).sum()) != 1 or int((inp == WALL).sum()) != 6:
        return False
    if _wall_component_size(inp) != 6:
        return False

    seed = tuple(map(int, np.argwhere(inp == FILL)[0]))
    input_component, input_touches_border = _component_from_seed(inp, seed)
    if not input_touches_border or len(input_component) <= 1:
        return False

    mirrored = _mirror_expand(inp)
    out = main(inp)
    if int((mirrored == FILL).sum()) != 4:
        return False

    seed_components = _seed_components(mirrored)
    new_fill = int((out == FILL).sum() - (mirrored == FILL).sum())

    if mode == "closed":
        return (
            len(seed_components) == 1
            and seed_components[0][1] is False
            and new_fill == 20
        )

    red_fraction = float((out == FILL).sum()) / float(out.size)
    return (
        len(seed_components) == 1
        and seed_components[0][1] is True
        and new_fill >= 48
        and red_fraction <= 0.76
    )


def _next_mode_and_variant() -> tuple[str, str]:
    global _GEN_COUNTER
    idx = _GEN_COUNTER
    _GEN_COUNTER += 1
    mode = "closed" if idx % 2 == 0 else "open"
    variant = "base" if (idx // 2) % 2 == 0 else "transpose"
    return mode, variant


def _gen_attempt() -> np.ndarray | None:
    mode, variant = _next_mode_and_variant()
    height = random.randint(5, 9)
    width = random.randint(5, 9)
    inp = _build_input(height, width, mode=mode, variant=variant)
    return inp if _semantic_ok(inp, mode) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for mirrored enclosure flood-fill")
