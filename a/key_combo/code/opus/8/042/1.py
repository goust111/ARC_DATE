import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7

RANK_COLORS = [BLUE, GREEN, YELLOW, ORANGE, PINK]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _red_neighbor_count(grid: np.ndarray, r: int, c: int) -> int:
    return sum(int(grid[rr, cc] == RED) for rr, cc in _neighbors4(grid.shape, r, c))


def _filter_unsupported(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    red_coords = np.argwhere(out == RED)
    for r0, c0 in red_coords:
        r, c = int(r0), int(c0)
        if _red_neighbor_count(grid, r, c) < 2:
            out[r, c] = BLACK
    return out


def _components_of_red(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != RED:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(grid.shape, rr, cc):
                    if not seen[nr, nc] and int(grid[nr, nc]) == RED:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove weakly supported red noise, then recolor the surviving components by size rank.

    Input structure:
    - The grid has a black background and a single red foreground color.
    - Red pixels form several 4-connected regions, mixing dense main bodies with weak noise.
    - Noise appears as isolated red pixels, detached 2-pixel fragments, or 1-neighbor leaf pixels attached to a main body.

    Rule:
    - For every red pixel, count its red 4-neighbors in the original input and erase the pixel if that count is below 2.
    - Find the remaining red 4-connected components after this filtering step.
    - Sort those surviving components by pixel count from largest to smallest.
    - Recolor rank 1..5 to blue, green, yellow, orange, and pink respectively; erased noise stays black.

    Key insight:
    - The ranking is defined on the post-filter components, so the solver must denoise first and only then measure and recolor what remains.
    """
    g = np.asarray(grid, dtype=int)
    filtered = _filter_unsupported(g)
    comps = sorted(_components_of_red(filtered), key=len, reverse=True)

    out = g.copy()
    out[g == RED] = BLACK
    for color, comp in zip(RANK_COLORS, comps):
        _paint_cells(out, comp, color)
    return out


SAFE_SHAPES = [
    np.array([[1, 1, 1], [1, 1, 1]], dtype=int),  # 6
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),  # 8
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int),  # 9
    np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]], dtype=int),  # 10
    np.array([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=int),  # 11
    np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=int),  # 12
    np.array([[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=int),  # 14
    np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=int),  # 15
]


def _transform_shape(shape: np.ndarray) -> np.ndarray:
    out = np.array(shape, copy=True)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out


def _shape_cells(shape: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    cells = []
    h, w = shape.shape
    for r in range(h):
        for c in range(w):
            if int(shape[r, c]) == 1:
                cells.append((top + r, left + c))
    return cells


def _all_cells_with_padding(cells: list[tuple[int, int]], pad: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                out.add((r + dr, c + dc))
    return out


def _place_shapes(
    height: int, width: int, shapes: list[np.ndarray], pad: int
) -> list[list[tuple[int, int]]] | None:
    occupied: set[tuple[int, int]] = set()
    padded: set[tuple[int, int]] = set()
    placements: list[list[tuple[int, int]]] = []
    for shape in shapes:
        sh, sw = shape.shape
        placed = False
        for _ in range(150):
            top = random.randint(1, height - sh - 1)
            left = random.randint(1, width - sw - 1)
            cells = _shape_cells(shape, top, left)
            if any(cell in occupied for cell in cells):
                continue
            if any(cell in padded for cell in cells):
                continue
            placements.append(cells)
            occupied.update(cells)
            padded.update(_all_cells_with_padding(cells, pad))
            placed = True
            break
        if not placed:
            return None
    return placements


def _place_shapes_in_zones(
    height: int,
    width: int,
    shapes: list[np.ndarray],
    zones: list[tuple[int, int, int, int]],
    pad: int,
) -> list[list[tuple[int, int]]] | None:
    occupied: set[tuple[int, int]] = set()
    padded: set[tuple[int, int]] = set()
    placements: list[list[tuple[int, int]]] = []
    for shape, (r0, r1, c0, c1) in zip(shapes, zones):
        sh, sw = shape.shape
        if (r1 - r0) < sh or (c1 - c0) < sw:
            return None
        placed = False
        for _ in range(100):
            top = random.randint(r0, r1 - sh)
            left = random.randint(c0, c1 - sw)
            cells = _shape_cells(shape, top, left)
            if any(cell in occupied for cell in cells):
                continue
            if any(cell in padded for cell in cells):
                continue
            placements.append(cells)
            occupied.update(cells)
            padded.update(_all_cells_with_padding(cells, pad))
            placed = True
            break
        if not placed:
            return None
    return placements


def _leaf_candidates(grid: np.ndarray, core_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates = []
    for r, c in core_cells:
        for rr, cc in _neighbors4((h, w), r, c):
            if int(grid[rr, cc]) != BLACK:
                continue
            neigh = [(nr, nc) for nr, nc in _neighbors4((h, w), rr, cc) if int(grid[nr, nc]) == RED]
            if len(neigh) == 1:
                candidates.append((rr, cc))
    random.shuffle(candidates)
    return candidates


def _add_attached_leaf_noise(grid: np.ndarray, core_components: list[list[tuple[int, int]]]) -> int:
    added = 0
    order = list(range(len(core_components)))
    random.shuffle(order)
    target = random.randint(1, min(3, len(core_components)))
    for idx in order:
        core_cells = set(core_components[idx])
        for rr, cc in _leaf_candidates(grid, core_cells):
            if int(grid[rr, cc]) == BLACK:
                grid[rr, cc] = RED
                added += 1
                break
        if added >= target:
            break
    return added


def _can_place_isolated_single(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BLACK:
        return False
    return _red_neighbor_count(grid, r, c) == 0


def _can_place_domino(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    (r0, c0), (r1, c1) = a, b
    if int(grid[r0, c0]) != BLACK or int(grid[r1, c1]) != BLACK:
        return False
    temp = grid.copy()
    temp[r0, c0] = RED
    temp[r1, c1] = RED
    return _red_neighbor_count(temp, r0, c0) == 1 and _red_neighbor_count(temp, r1, c1) == 1


def _add_fragment_noise(grid: np.ndarray) -> int:
    h, w = grid.shape
    added = 0
    for _ in range(random.randint(1, 3)):
        if random.random() < 0.55:
            placed = False
            for _ in range(80):
                r = random.randint(0, h - 1)
                c = random.randint(0, w - 1)
                if _can_place_isolated_single(grid, r, c):
                    grid[r, c] = RED
                    added += 1
                    placed = True
                    break
            if not placed:
                continue
        else:
            placed = False
            for _ in range(80):
                r = random.randint(0, h - 1)
                c = random.randint(0, w - 1)
                dr, dc = random.choice(DIRS4)
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and _can_place_domino(grid, (r, c), (rr, cc)):
                    grid[r, c] = RED
                    grid[rr, cc] = RED
                    added += 2
                    placed = True
                    break
            if not placed:
                continue
    return added


def _add_near_bridge_noise(grid: np.ndarray, components: list[list[tuple[int, int]]]) -> int:
    """Add facing 1-neighbor leaves to create bridge-like clutter without changing the filtered components."""
    h, w = grid.shape
    core_sets = [set(comp) for comp in components]
    for i in range(len(core_sets)):
        for j in range(i + 1, len(core_sets)):
            a = core_sets[i]
            b = core_sets[j]
            for ar, ac in a:
                for br, bc in b:
                    if ar == br and 3 <= abs(ac - bc) <= 5:
                        step = 1 if ac < bc else -1
                        leaf_a = (ar, ac + step)
                        leaf_b = (br, bc - step)
                        if not (0 <= leaf_a[0] < h and 0 <= leaf_a[1] < w):
                            continue
                        if not (0 <= leaf_b[0] < h and 0 <= leaf_b[1] < w):
                            continue
                        if leaf_a == leaf_b:
                            continue
                        if _can_place_isolated_single(grid, *leaf_a) and _can_place_isolated_single(grid, *leaf_b):
                            grid[leaf_a] = RED
                            grid[leaf_b] = RED
                            return 2
                    if ac == bc and 3 <= abs(ar - br) <= 5:
                        step = 1 if ar < br else -1
                        leaf_a = (ar + step, ac)
                        leaf_b = (br - step, bc)
                        if not (0 <= leaf_a[0] < h and 0 <= leaf_a[1] < w):
                            continue
                        if not (0 <= leaf_b[0] < h and 0 <= leaf_b[1] < w):
                            continue
                        if leaf_a == leaf_b:
                            continue
                        if _can_place_isolated_single(grid, *leaf_a) and _can_place_isolated_single(grid, *leaf_b):
                            grid[leaf_a] = RED
                            grid[leaf_b] = RED
                            return 2
    return 0


def _component_areas(grid: np.ndarray) -> list[int]:
    return sorted((len(comp) for comp in _components_of_red(grid)), reverse=True)


def _valid_core_shape(shape: np.ndarray) -> bool:
    coords = np.argwhere(shape == 1)
    if coords.size == 0:
        return False
    for r0, c0 in coords:
        r, c = int(r0), int(c0)
        deg = 0
        for rr, cc in _neighbors4(shape.shape, r, c):
            deg += int(shape[rr, cc] == 1)
        if deg < 2:
            return False
    return True


def _validate_generated(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not set(np.unique(inp)).issubset({BLACK, RED}):
        return False

    filtered = _filter_unsupported(inp)
    comps = _components_of_red(filtered)
    if not (3 <= len(comps) <= 5):
        return False

    areas = [len(comp) for comp in comps]
    if len(set(areas)) != len(areas):
        return False
    if any(area < 5 or area > 15 for area in areas):
        return False

    if np.array_equal(inp, filtered):
        return False
    deleted = int(np.sum((inp == RED) & (filtered == BLACK)))
    if deleted < 3:
        return False

    for r0, c0 in np.argwhere(filtered == RED):
        if _red_neighbor_count(filtered, int(r0), int(c0)) < 2:
            return False

    out = main(inp)
    used = [int(v) for v in np.unique(out) if int(v) != BLACK]
    if len(used) != len(comps):
        return False
    if set(used) != set(RANK_COLORS[: len(comps)]):
        return False

    if len(_component_areas(inp)) > len(areas) + 3:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    count = random.randint(3, 5)
    if count == 5:
        height = random.randint(19, 23)
        width = random.randint(19, 23)
        shape_pool = random.sample(SAFE_SHAPES[:6], count)
        pad = 1
    elif count == 4:
        height = random.randint(16, 21)
        width = random.randint(16, 21)
        shape_pool = random.sample(SAFE_SHAPES[:7], count)
        pad = 2
    else:
        height = random.randint(15, 20)
        width = random.randint(15, 20)
        shape_pool = random.sample(SAFE_SHAPES, count)
        pad = 2

    shapes = [_transform_shape(shape) for shape in shape_pool]
    if not all(_valid_core_shape(shape) for shape in shapes):
        return None

    if count == 5:
        mid_r = height // 2
        mid_c = width // 2
        zones = [
            (1, mid_r - 2, 1, mid_c - 2),
            (1, mid_r - 2, mid_c + 1, width - 2),
            (mid_r - 2, mid_r + 3, mid_c - 3, mid_c + 4),
            (mid_r + 1, height - 2, 1, mid_c - 2),
            (mid_r + 1, height - 2, mid_c + 1, width - 2),
        ]
        random.shuffle(zones)
        placements = _place_shapes_in_zones(height, width, shapes, zones, pad)
    else:
        placements = _place_shapes(height, width, shapes, pad)
    if placements is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for cells in placements:
        _paint_cells(grid, cells, RED)

    leaf_added = _add_attached_leaf_noise(grid, placements)
    fragment_added = _add_fragment_noise(grid)
    bridge_like_added = _add_near_bridge_noise(grid, placements)
    if leaf_added == 0 or fragment_added == 0:
        return None
    if leaf_added + fragment_added + bridge_like_added < 3:
        return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 042/1")
