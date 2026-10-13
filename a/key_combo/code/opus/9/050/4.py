from __future__ import annotations

import random
from itertools import product

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

INPUT_OBJECT_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
RECOLOR_FOR_PARITY = {0: BLUE, 1: RED}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
COUNT_PATTERNS = [
    counts
    for counts in product(range(1, 5), repeat=4)
    if 8 <= sum(counts) <= 12
    and any(v % 2 == 0 for v in counts)
    and any(v % 2 == 1 for v in counts)
]
RICH_COUNT_PATTERNS = [counts for counts in COUNT_PATTERNS if 1 in counts or 4 in counts]
QUADRANT_ORDER = ("ul", "ur", "ll", "lr")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask as global (r, c) coords."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
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
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_center_block(grid: np.ndarray) -> tuple[int, int]:
    grey_components = _components(grid == GREY)
    matches: list[tuple[int, int]] = []
    for comp in grey_components:
        if len(comp) != 9:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            matches.append((r0 + 1, c0 + 1))
    if len(matches) != 1:
        raise ValueError("expected exactly one solid 3x3 grey block")
    return matches[0]


def _quadrant_bounds(shape: tuple[int, int], cr: int, cc: int) -> dict[str, tuple[int, int, int, int]]:
    h, w = shape
    return {
        "ul": (0, cr - 1, 0, cc - 1),
        "ur": (0, cr - 1, cc + 1, w - 1),
        "ll": (cr + 1, h - 1, 0, cc - 1),
        "lr": (cr + 1, h - 1, cc + 1, w - 1),
    }


def _object_components_in_bounds(
    grid: np.ndarray, bounds: tuple[int, int, int, int]
) -> list[list[tuple[int, int]]]:
    r0, r1, c0, c1 = bounds
    if r0 > r1 or c0 > c1:
        return []
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = (sub != BLACK) & (sub != GREY)
    parts = _components(mask)
    return [[(r + r0, c + c0) for r, c in comp] for comp in parts]


def _analyze_grid(grid: np.ndarray) -> tuple[int, int, dict[str, list[list[tuple[int, int]]]]]:
    cr, cc = _find_center_block(grid)
    quadrant_objects = {
        name: _object_components_in_bounds(grid, bounds)
        for name, bounds in _quadrant_bounds(grid.shape, cr, cc).items()
    }
    return cr, cc, quadrant_objects


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    cr, cc, quadrant_objects = _analyze_grid(g)
    out = g.copy()

    out[cr, :] = GREY
    out[:, cc] = GREY
    out[cr - 1 : cr + 2, cc - 1 : cc + 2] = GREY

    for name in QUADRANT_ORDER:
        objects = quadrant_objects[name]
        color = RECOLOR_FOR_PARITY[len(objects) % 2]
        for comp in objects:
            for r, c in comp:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a grey cross from the center block, then recolor each quadrant by the parity of its object count.

    Input structure:
    - The background is black.
    - Exactly one solid 3x3 grey block marks the center reference point.
    - Non-grey foreground objects are small 4-connected monochrome regions placed strictly inside the four quadrants and away from the center row, center column, and grey block.

    Rule:
    - Find the grey 3x3 block and take its center cell as (cr, cc).
    - Paint a grey cross through the full center row and full center column while keeping the 3x3 block grey.
    - Treat cells with row < cr and/or row > cr together with col < cc and/or col > cc as the four quadrants, excluding the center row and center column themselves.
    - Count the non-black, non-grey 4-connected objects independently inside each quadrant.
    - Recolor every object in an odd-count quadrant red and every object in an even-count quadrant blue, preserving shapes, the grey block, the grey cross, and the black background.

    Key insight:
    - The emitted grey cross is both a visible output feature and the partition that defines which objects are counted together, so the final recoloring depends on quadrant object parity rather than on an object's own color, size, or shape.
    """
    return _solve(grid)


def _sample_count_pattern() -> tuple[int, int, int, int]:
    if random.random() < 0.7:
        return random.choice(RICH_COUNT_PATTERNS)
    return random.choice(COUNT_PATTERNS)


def _random_connected_shape(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                frontier.add((r + dr, c + dc))
        frontier -= cells
        candidate = random.choice(sorted(frontier))
        cells.add(candidate)

    if random.random() < 0.35:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if max(rs) == min(rs) or max(cs) == min(cs):
            extra = random.choice(sorted(cells))
            for dr, dc in DIRS4:
                if len(cells) >= area:
                    break
                probe = (extra[0] + dr, extra[1] + dc)
                if probe not in cells:
                    cells.add(probe)
            while len(cells) > area:
                removable = []
                for cell in sorted(cells):
                    rest = cells - {cell}
                    if rest and _is_connected_cell_set(rest):
                        removable.append(cell)
                if not removable:
                    break
                cells.remove(random.choice(removable))

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True

    transform_id = random.randrange(8)
    if transform_id >= 4:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, transform_id % 4)
    return mask.copy()


def _is_connected_cell_set(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _occupied_with_halo(occupied: np.ndarray) -> np.ndarray:
    h, w = occupied.shape
    halo = occupied.copy()
    for r, c in np.argwhere(occupied):
        for dr, dc in DIRS8:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w:
                halo[nr, nc] = True
    return halo


def _quadrant_generation_regions(
    shape: tuple[int, int], cr: int, cc: int
) -> dict[str, tuple[int, int, int, int]]:
    h, w = shape
    return {
        "ul": (0, cr - 3, 0, cc - 3),
        "ur": (0, cr - 3, cc + 3, w - 1),
        "ll": (cr + 3, h - 1, 0, cc - 3),
        "lr": (cr + 3, h - 1, cc + 3, w - 1),
    }


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    region: tuple[int, int, int, int],
    shape: np.ndarray,
    color: int,
) -> list[tuple[int, int]] | None:
    r0, r1, c0, c1 = region
    sh, sw = shape.shape
    if r1 - r0 + 1 < sh or c1 - c0 + 1 < sw:
        return None

    forbidden = _occupied_with_halo(occupied)
    candidates: list[list[tuple[int, int]]] = []
    local_cells = [(int(r), int(c)) for r, c in np.argwhere(shape)]
    for top in range(r0, r1 - sh + 2):
        for left in range(c0, c1 - sw + 2):
            cells = [(top + r, left + c) for r, c in local_cells]
            if any(forbidden[r, c] for r, c in cells):
                continue
            candidates.append(cells)
    if not candidates:
        return None

    cells = random.choice(candidates)
    for r, c in cells:
        grid[r, c] = color
        occupied[r, c] = True
    return cells


def _build_input() -> np.ndarray | None:
    h = random.randint(17, 22)
    w = random.randint(17, 22)
    cr = max(6, min(h - 7, h // 2 + random.randint(-1, 1)))
    cc = max(6, min(w - 7, w // 2 + random.randint(-1, 1)))

    grid = np.zeros((h, w), dtype=int)
    grid[cr - 1 : cr + 2, cc - 1 : cc + 2] = GREY
    occupied = grid != BLACK

    count_pattern = _sample_count_pattern()
    regions = _quadrant_generation_regions(grid.shape, cr, cc)

    for name, count in zip(QUADRANT_ORDER, count_pattern):
        region = regions[name]
        for _ in range(count):
            placed = None
            for _shape_try in range(60):
                area = random.randint(2, 5)
                shape = _random_connected_shape(area)
                color = random.choice(INPUT_OBJECT_COLORS)
                placed = _place_shape(grid, occupied, region, shape, color)
                if placed is not None:
                    break
            if placed is None:
                return None
    return grid


def _quadrant_name_for_cell(r: int, c: int, cr: int, cc: int) -> str | None:
    if r < cr and c < cc:
        return "ul"
    if r < cr and c > cc:
        return "ur"
    if r > cr and c < cc:
        return "ll"
    if r > cr and c > cc:
        return "lr"
    return None


def _independent_expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    cr, cc = _find_center_block(g)
    out = g.copy()
    out[:, cc] = GREY
    out[cr, :] = GREY
    out[cr - 1 : cr + 2, cc - 1 : cc + 2] = GREY

    quadrant_parts = {name: [] for name in QUADRANT_ORDER}
    for comp in _components((g != BLACK) & (g != GREY)):
        names = {_quadrant_name_for_cell(r, c, cr, cc) for r, c in comp}
        if len(names) != 1 or None in names:
            raise ValueError("foreground object crosses a forbidden boundary")
        quadrant_parts[names.pop()].append(comp)

    for name in QUADRANT_ORDER:
        fill = RED if len(quadrant_parts[name]) % 2 == 1 else BLUE
        for comp in quadrant_parts[name]:
            for r, c in comp:
                out[r, c] = fill
    return out


def _validate_input(grid: np.ndarray) -> bool:
    try:
        cr, cc = _find_center_block(grid)
    except ValueError:
        return False

    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    object_mask = (grid != BLACK) & (grid != GREY)
    objects = _components(object_mask)
    if not (8 <= len(objects) <= 12):
        return False

    counts = {name: 0 for name in QUADRANT_ORDER}
    area_sums = {name: 0 for name in QUADRANT_ORDER}
    colors_by_quadrant = {name: set() for name in QUADRANT_ORDER}

    for comp in objects:
        if not (2 <= len(comp) <= 5):
            return False
        comp_colors = {int(grid[r, c]) for r, c in comp}
        if len(comp_colors) != 1 or next(iter(comp_colors)) not in INPUT_OBJECT_COLORS:
            return False

        quad_names = {_quadrant_name_for_cell(r, c, cr, cc) for r, c in comp}
        if len(quad_names) != 1 or None in quad_names:
            return False
        quad = quad_names.pop()

        for r, c in comp:
            if cr - 2 <= r <= cr + 2 and cc - 2 <= c <= cc + 2:
                return False
        counts[quad] += 1
        area_sums[quad] += len(comp)
        colors_by_quadrant[quad].update(comp_colors)

    if min(counts.values()) < 1:
        return False
    if not any(v % 2 == 0 for v in counts.values()):
        return False
    if not any(v % 2 == 1 for v in counts.values()):
        return False
    if not any((area_sums[name] % 2) != (counts[name] % 2) for name in QUADRANT_ORDER):
        return False

    odd_colors = set().union(*(colors_by_quadrant[name] for name in QUADRANT_ORDER if counts[name] % 2 == 1))
    even_colors = set().union(*(colors_by_quadrant[name] for name in QUADRANT_ORDER if counts[name] % 2 == 0))
    if not (odd_colors & even_colors):
        return False

    expected = _independent_expected_output(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    out_colors = set(int(v) for v in np.unique(actual))
    if RED not in out_colors or BLUE not in out_colors:
        return False
    if not np.all(actual[cr, :] == GREY) or not np.all(actual[:, cc] == GREY):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_input()
        if grid is not None and _validate_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid quadrant-parity ray puzzle input")
