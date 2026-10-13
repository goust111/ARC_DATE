import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
PALETTE = (2, 3, 4, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 500
GRID_SIZE_CHOICES = (6, 7)
_N_BAG = []


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=sprite.dtype))


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _foreground_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return _mask_components((grid != BLACK) & (grid != BLUE))


def _count_foreground_components(grid: np.ndarray) -> int:
    return len(_foreground_components(grid))


def _seed_coord(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == BLUE)
    if coords.shape != (1, 2):
        raise ValueError("expected exactly one blue seed pixel")
    return int(coords[0, 0]), int(coords[0, 1])


def _upscaled_seed_slices(seed: tuple[int, int], factor: int) -> tuple[slice, slice]:
    r, c = seed
    return slice(r * factor, (r + 1) * factor), slice(c * factor, (c + 1) * factor)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    factor = _count_foreground_components(g)
    if factor < 1:
        raise ValueError("expected at least one non-blue foreground component")

    seed = _seed_coord(g)
    out = scale_sprite(g, factor)
    seed_rs, seed_cs = _upscaled_seed_slices(seed, factor)

    seen = np.zeros(out.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for rr in range(seed_rs.start, seed_rs.stop):
        for cc in range(seed_cs.start, seed_cs.stop):
            seen[rr, cc] = True
            q.append((rr, cc))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(out.shape, nr, nc) or seen[nr, nc] or out[nr, nc] != BLACK:
                continue
            seen[nr, nc] = True
            out[nr, nc] = BLUE
            q.append((nr, nc))

    out[seed_rs, seed_cs] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the non-blue foreground components, upscale the full grid by that count, then flood-fill from the enlarged seed and erase the seed block.

    Input structure:
    - The grid uses black as background and contains one blue seed pixel.
    - Every other non-black object is a small monochromatic 4-connected shape.
    - The non-blue shapes are mutually separated as distinct foreground components.

    Rule:
    - Count the 4-connected components formed by all non-black, non-blue pixels; call this number N.
    - Replace every input cell by an N x N block of the same color.
    - From the enlarged blue seed block, flood through 4-connected black cells only, recoloring each reached black cell blue.
    - Erase the original enlarged seed block back to black, while keeping the enlarged foreground blockers unchanged.

    Key insight:
    - The component count is itself the geometric control parameter: it simultaneously sets the upscale factor and changes the topology of the flood-fill domain.
    """
    return _solve(grid)


def _bagged_component_count() -> int:
    global _N_BAG
    if not _N_BAG:
        _N_BAG = [2, 3, 4]
        random.shuffle(_N_BAG)
    return _N_BAG.pop()


def _step(coord: tuple[int, int], direction: tuple[int, int], times: int = 1) -> tuple[int, int]:
    r, c = coord
    dr, dc = direction
    return r + dr * times, c + dc * times


def _turns(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return (dc, -dr), (-dc, dr)


def _line_capacity(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = start
    dr, dc = direction
    k = 0
    while 0 <= r < h and 0 <= c < w:
        k += 1
        r += dr
        c += dc
    return k


def _is_component_valid(shape: tuple[int, int], cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    if not all(in_bounds(shape, r, c) for r, c in cells):
        return False
    if cell_set & occupied:
        return False
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in occupied and (nr, nc) not in cell_set:
                return False
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return len(_mask_components(mask)) == 1


def _build_component(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    outward: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    capacity = _line_capacity(shape, anchor, outward)
    if capacity < 2:
        return None

    candidates = []
    line_only_max = min(4, capacity)
    for area in range(2, line_only_max + 1):
        cells = [_step(anchor, outward, k) for k in range(area)]
        candidates.append(cells)

    if capacity >= 2:
        far2 = _step(anchor, outward, 1)
        for side in _turns(outward):
            bend = _step(far2, side)
            candidates.append([anchor, far2, bend])

    if capacity >= 3:
        far3 = _step(anchor, outward, 2)
        for side in _turns(outward):
            bend = _step(far3, side)
            candidates.append([anchor, _step(anchor, outward, 1), far3, bend])

    random.shuffle(candidates)
    for cells in candidates:
        if 2 <= len(cells) <= 4 and _is_component_valid(shape, cells, occupied):
            return cells
    return None


def _topology_spec(h: int, w: int, count: int):
    if count == 2:
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            pocket = (0, 0)
            return pocket, [((0, 1), (0, 1)), ((1, 0), (1, 0))]
        if corner == "tr":
            pocket = (0, w - 1)
            return pocket, [((0, w - 2), (0, -1)), ((1, w - 1), (1, 0))]
        if corner == "bl":
            pocket = (h - 1, 0)
            return pocket, [((h - 1, 1), (0, 1)), ((h - 2, 0), (-1, 0))]
        pocket = (h - 1, w - 1)
        return pocket, [((h - 1, w - 2), (0, -1)), ((h - 2, w - 1), (-1, 0))]

    if count == 3:
        edge = random.choice(("top", "bottom", "left", "right"))
        if edge == "top":
            c = random.randint(1, w - 2)
            pocket = (0, c)
            return pocket, [((0, c - 1), (0, -1)), ((0, c + 1), (0, 1)), ((1, c), (1, 0))]
        if edge == "bottom":
            c = random.randint(1, w - 2)
            pocket = (h - 1, c)
            return pocket, [((h - 1, c - 1), (0, -1)), ((h - 1, c + 1), (0, 1)), ((h - 2, c), (-1, 0))]
        if edge == "left":
            r = random.randint(1, h - 2)
            pocket = (r, 0)
            return pocket, [((r - 1, 0), (-1, 0)), ((r + 1, 0), (1, 0)), ((r, 1), (0, 1))]
        r = random.randint(1, h - 2)
        pocket = (r, w - 1)
        return pocket, [((r - 1, w - 1), (-1, 0)), ((r + 1, w - 1), (1, 0)), ((r, w - 2), (0, -1))]

    r = random.randint(1, h - 2)
    c = random.randint(1, w - 2)
    pocket = (r, c)
    return pocket, [((r - 1, c), (-1, 0)), ((r + 1, c), (1, 0)), ((r, c - 1), (0, -1)), ((r, c + 1), (0, 1))]


def _largest_background_component(grid: np.ndarray) -> set[tuple[int, int]]:
    comps = _mask_components(grid == BLACK)
    if not comps:
        return set()
    return set(max(comps, key=len))


def _seed_candidates(grid: np.ndarray) -> list[tuple[int, int]]:
    largest_bg = _largest_background_component(grid)
    candidates = []
    for r, c in sorted(largest_bg):
        if sum(1 for nr, nc in neighbors4(r, c) if in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK) < 2:
            continue
        if any(in_bounds(grid.shape, nr, nc) and grid[nr, nc] not in (BLACK, BLUE) for nr, nc in neighbors4(r, c)):
            continue
        candidates.append((r, c))
    return candidates


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    h, w = inp.shape
    if inp.ndim != 2 or out.ndim != 2 or max(h, w) > 30 or max(out.shape) > 30:
        return False

    seed_coords = np.argwhere(inp == BLUE)
    if seed_coords.shape != (1, 2):
        return False

    nonzero_nonblue = [int(v) for v in np.unique(inp) if v not in (BLACK, BLUE)]
    if len(nonzero_nonblue) < 2 or len(nonzero_nonblue) > 4:
        return False

    comps = _foreground_components(inp)
    if len(comps) != len(nonzero_nonblue):
        return False

    colors = []
    for comp in comps:
        vals = {int(inp[r, c]) for r, c in comp}
        if len(vals) != 1:
            return False
        color = vals.pop()
        if color in colors:
            return False
        colors.append(color)
        if not (2 <= len(comp) <= 4):
            return False

    sr, sc = map(int, seed_coords[0])
    for nr, nc in neighbors4(sr, sc):
        if in_bounds(inp.shape, nr, nc) and inp[nr, nc] not in (BLACK, BLUE):
            return False

    factor = len(comps)
    if out.shape != (h * factor, w * factor):
        return False

    expected = _solve(inp)
    if not np.array_equal(out, expected):
        return False

    seed_rs, seed_cs = _upscaled_seed_slices((sr, sc), factor)
    if np.any(out[seed_rs, seed_cs] != BLACK):
        return False

    blue_count = int(np.count_nonzero(out == BLUE))
    if blue_count < factor:
        return False

    blockers = scale_sprite(((inp != BLACK) & (inp != BLUE)).astype(int), factor).astype(bool)
    seed_mask = np.zeros(out.shape, dtype=bool)
    seed_mask[seed_rs, seed_cs] = True
    remaining_black = (out == BLACK) & (~seed_mask) & (~blockers)
    if not remaining_black.any():
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(GRID_SIZE_CHOICES)
    w = random.choice(GRID_SIZE_CHOICES)
    count = _bagged_component_count()
    if max(h * count, w * count) > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    pocket, specs = _topology_spec(h, w, count)
    occupied: set[tuple[int, int]] = set()
    components = []

    for anchor, outward in specs:
        cells = _build_component(grid.shape, anchor, outward, occupied)
        if cells is None:
            return None
        occupied.update(cells)
        components.append(cells)

    colors = random.sample(PALETTE, count)
    for color, cells in zip(colors, components):
        for r, c in cells:
            grid[r, c] = color

    candidates = [cell for cell in _seed_candidates(grid) if cell != pocket]
    if not candidates:
        return None
    seed = random.choice(candidates)
    grid[seed] = BLUE

    out = _solve(grid)
    if not _semantic_checks(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
