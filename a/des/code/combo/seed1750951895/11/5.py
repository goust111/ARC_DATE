import numpy as np


BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAROON = 9

DIRS = {
    RED: (-1, 0),
    GREEN: (1, 0),
    BLUE: (0, -1),
    YELLOW: (0, 1),
}
MARKER_COLORS = tuple(DIRS)
OBJECT_COLORS = (0, 5, 6, 7)
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _point_symmetric_center(coords: list[tuple[int, int]]) -> tuple[int, int] | None:
    n = len(coords)
    if n <= 1:
        return None
    sr = sum(r for r, _ in coords)
    sc = sum(c for _, c in coords)
    if sr % n != 0 or sc % n != 0:
        return None
    cr, cc = sr // n, sc // n
    cells = set(coords)
    if (cr, cc) not in cells:
        return None
    for r, c in coords:
        if (2 * cr - r, 2 * cc - c) not in cells:
            return None
    return cr, cc


def _find_object(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    object_mask = (grid != BG) & ~np.isin(grid, MARKER_COLORS)
    comps = _components(object_mask)
    candidates: list[tuple[int, set[tuple[int, int]], tuple[int, int]]] = []
    for comp in comps:
        center = _point_symmetric_center(comp)
        if center is not None:
            candidates.append((len(comp), set(comp), center))
    assert len(candidates) == 1, "expected exactly one symmetric foreground object"
    _area, cells, center = candidates[0]
    return cells, center


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    object_cells, center = _find_object(g)
    out = g.copy()
    out[center] = MAROON

    proposals: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for r, c in map(tuple, np.argwhere(np.isin(g, MARKER_COLORS))):
        color = int(g[r, c])
        dr, dc = DIRS[color]
        nr, nc = int(r + dr), int(c + dc)
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] == BG:
            proposals.append(((int(r), int(c)), (nr, nc), color))

    dest_counts: dict[tuple[int, int], int] = {}
    for _src, dst, _color in proposals:
        dest_counts[dst] = dest_counts.get(dst, 0) + 1

    for src, dst, color in proposals:
        if dest_counts[dst] != 1:
            continue
        out[src] = BG
        out[dst] = color

    return out


def _shape_orbit(anchor: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    r, c = anchor
    return {(r, c), (h - 1 - r, w - 1 - c)}


def _generate_symmetric_shape(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    center = (h // 2, w // 2)
    shape = {center}
    target = int(rng.choice(np.arange(5, min(h * w, 19) + 1, 2)))
    canonical: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            rr, cc = h - 1 - r, w - 1 - c
            if (r, c) == center:
                continue
            if (r < rr) or (r == rr and c < cc):
                canonical.append((r, c))

    while len(shape) < target:
        candidates: list[set[tuple[int, int]]] = []
        for cell in canonical:
            orbit = _shape_orbit(cell, h, w)
            if orbit & shape:
                continue
            if len(shape) + len(orbit) > target:
                continue
            for r, c in orbit:
                if any((r + dr, c + dc) in shape for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                    candidates.append(orbit)
                    break
        if not candidates:
            return None
        shape |= set(rng.choice(candidates))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    if len(_components(mask)) != 1:
        return None
    return mask


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _place_free_marker(
    rng: np.random.Generator,
    grid: np.ndarray,
    color: int,
    used_dests: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    dr, dc = DIRS[color]
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    for r, c in cells:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if grid[r, c] != BG or grid[nr, nc] != BG or (nr, nc) in used_dests:
            continue
        if any(0 <= rr < h and 0 <= cc < w and grid[rr, cc] != BG for rr, cc in _neighbors4(r, c)):
            continue
        grid[r, c] = color
        used_dests.add((nr, nc))
        return True
    return False


def _place_blocked_marker(
    rng: np.random.Generator,
    grid: np.ndarray,
    object_cells: set[tuple[int, int]],
    color: int,
) -> bool:
    h, w = grid.shape
    dr, dc = DIRS[color]
    targets = list(object_cells)
    rng.shuffle(targets)
    for tr, tc in targets:
        sr, sc = tr - dr, tc - dc
        if not (0 <= sr < h and 0 <= sc < w):
            continue
        if grid[sr, sc] != BG:
            continue
        grid[sr, sc] = color
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(10, 17))
    w = int(rng.integers(10, 17))
    obj_h = int(rng.choice([3, 5, 7]))
    obj_w = int(rng.choice([3, 5, 7]))
    if obj_h >= h - 2 or obj_w >= w - 2:
        return None

    shape = _generate_symmetric_shape(rng, obj_h, obj_w)
    if shape is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    color = int(rng.choice(OBJECT_COLORS))

    base_r = (h - obj_h) // 2
    base_c = (w - obj_w) // 2
    top = int(np.clip(base_r + rng.integers(-2, 3), 1, h - obj_h - 1))
    left = int(np.clip(base_c + rng.integers(-2, 3), 1, w - obj_w - 1))

    object_cells: set[tuple[int, int]] = set()
    for rr, cc in map(tuple, np.argwhere(shape)):
        ar, ac = top + int(rr), left + int(cc)
        grid[ar, ac] = color
        object_cells.add((ar, ac))

    marker_specs = [(RED, "free"), (GREEN, "free"), (BLUE, "blocked"), (YELLOW, "blocked")]
    extra_count = int(rng.integers(0, 5))
    for _ in range(extra_count):
        marker_specs.append((int(rng.choice(MARKER_COLORS)), str(rng.choice(["free", "blocked"]))))
    rng.shuffle(marker_specs)

    used_dests: set[tuple[int, int]] = set()
    free_count = 0
    blocked_count = 0
    for color, mode in marker_specs:
        ok = (
            _place_free_marker(rng, grid, color, used_dests)
            if mode == "free"
            else _place_blocked_marker(rng, grid, object_cells, color)
        )
        if not ok:
            return None
        if mode == "free":
            free_count += 1
        else:
            blocked_count += 1

    if free_count == 0 or blocked_count == 0:
        return None

    marker_count = int(np.count_nonzero(np.isin(grid, MARKER_COLORS)))
    if marker_count < 4:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    moved = 0
    stayed = 0
    for r, c in map(tuple, np.argwhere(np.isin(grid, MARKER_COLORS))):
        if out[r, c] == BG:
            moved += 1
        else:
            stayed += 1
    if moved == 0 or stayed == 0:
        return None
    if out[tuple(np.argwhere(shape)[0] + np.array([top, left]))] == MAROON and len(object_cells) == 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
