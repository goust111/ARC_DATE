import numpy as np


FOREGROUND = 2
BACKGROUND = 0
INPUT_SHAPE = (4, 5)
GEN_RETRY_LIMIT = 600
GEN_TOTAL_RANGE = range(6, 11)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _mirror_both_axes(grid: np.ndarray) -> np.ndarray:
    top = np.concatenate((grid, np.fliplr(grid)), axis=1)
    bottom = np.concatenate((np.flipud(grid), np.flipud(np.fliplr(grid))), axis=1)
    return np.concatenate((top, bottom), axis=0)


def _keep_largest_red_components(quadrant: np.ndarray) -> np.ndarray:
    out = quadrant.copy()
    red_mask = out == FOREGROUND
    comps = _components(red_mask)
    if not comps:
        return out

    best_area = max(len(comp) for comp in comps)
    keep = np.zeros_like(red_mask, dtype=bool)
    for comp in comps:
        if len(comp) == best_area:
            for r, c in comp:
                keep[r, c] = True

    out[red_mask & ~keep] = BACKGROUND
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the input across both axes, then keep only the largest red connected component(s) inside each quadrant.

    Input structure:
    - The input is a small black-background grid whose meaningful foreground cells are red.
    - Red cells may form several disconnected 4-connected regions.
    - The generator uses a 4x5 source tile, but the transformation itself is size-agnostic.

    Rule:
    - Build a 2x2 reflected layout: upper-left = original, upper-right = left-right flip,
      lower-left = up-down flip, lower-right = 180-degree rotation.
    - Treat the four reflected tiles as four independent quadrants of equal size.
    - In each quadrant, find the 4-connected red components, keep every component whose area
      is tied for the maximum in that quadrant, and erase all smaller red components to black.

    Key insight:
    - The task is a composition of whole-grid mirroring and per-quadrant area ranking of red connected components.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    out = _mirror_both_axes(g)
    quadrants = [
        (slice(0, h), slice(0, w)),
        (slice(0, h), slice(w, 2 * w)),
        (slice(h, 2 * h), slice(0, w)),
        (slice(h, 2 * h), slice(w, 2 * w)),
    ]

    for rs, cs in quadrants:
        out[rs, cs] = _keep_largest_red_components(out[rs, cs])
    return out


def _all_size_patterns() -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    unique: set[tuple[int, ...]] = set()
    tied: set[tuple[int, ...]] = set()

    def rec(total_left: int, parts_left: int, max_part: int, prefix: tuple[int, ...]):
        if parts_left == 0:
            if total_left == 0:
                parts = tuple(sorted(prefix, reverse=True))
                if parts[0] > parts[1]:
                    unique.add(parts)
                elif parts.count(parts[0]) >= 2 and any(p < parts[0] for p in parts):
                    tied.add(parts)
            return

        min_rest = parts_left - 1
        high = min(max_part, total_left - min_rest, 4)
        for part in range(high, 0, -1):
            rec(total_left - part, parts_left - 1, part, prefix + (part,))

    for total in GEN_TOTAL_RANGE:
        for count in range(2, 5):
            rec(total, count, 4, ())

    return sorted(unique), sorted(tied)


UNIQUE_PATTERNS, TIED_PATTERNS = _all_size_patterns()


def _random_connected_shape(size: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        nr, nc = frontier[np.random.randint(len(frontier))]
        cells.add((nr, nc))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    norm = [(r - r0, c - c0) for r, c in cells]
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True

    k = int(np.random.randint(4))
    if k:
        mask = np.rot90(mask, k)
    if bool(np.random.randint(2)):
        mask = np.fliplr(mask)
    if bool(np.random.randint(2)):
        mask = np.flipud(mask)
    return mask.copy()


def _can_place(grid: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> bool:
    h, w = grid.shape
    sh, sw = shape.shape
    if r0 < 0 or c0 < 0 or r0 + sh > h or c0 + sw > w:
        return False

    coords = np.argwhere(shape)
    for dr, dc in coords:
        r = int(r0 + dr)
        c = int(c0 + dc)
        if grid[r, c] != BACKGROUND:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == FOREGROUND:
                return False
    return True


def _place_shape(grid: np.ndarray, shape: np.ndarray) -> bool:
    h, w = grid.shape
    sh, sw = shape.shape
    positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    np.random.shuffle(positions)
    for r0, c0 in positions:
        if _can_place(grid, shape, r0, c0):
            coords = np.argwhere(shape)
            for dr, dc in coords:
                grid[r0 + int(dr), c0 + int(dc)] = FOREGROUND
            return True
    return False


def _component_sizes(grid: np.ndarray) -> list[int]:
    return sorted((len(comp) for comp in _components(grid == FOREGROUND)), reverse=True)


def _touches_inner_mirror_edge(grid: np.ndarray) -> bool:
    return bool(np.any(grid[-1, :] == FOREGROUND) or np.any(grid[:, -1] == FOREGROUND))


def _valid_input(grid: np.ndarray, expect_tie: bool) -> bool:
    if grid.shape != INPUT_SHAPE:
        return False
    if np.count_nonzero(grid == FOREGROUND) not in GEN_TOTAL_RANGE:
        return False
    if np.array_equal(grid, np.fliplr(grid)) or np.array_equal(grid, np.flipud(grid)):
        return False
    if not _touches_inner_mirror_edge(grid):
        return False

    sizes = _component_sizes(grid)
    if not (2 <= len(sizes) <= 4):
        return False
    if sizes[0] < 2:
        return False
    if expect_tie:
        if sizes.count(sizes[0]) < 2 or not any(s < sizes[0] for s in sizes):
            return False
    else:
        if len(sizes) < 2 or sizes[0] <= sizes[1]:
            return False

    mirrored = _mirror_both_axes(grid)
    if np.array_equal(main(grid), mirrored):
        return False
    return True


def _generate_attempt(expect_tie: bool) -> np.ndarray | None:
    patterns = TIED_PATTERNS if expect_tie else UNIQUE_PATTERNS
    sizes = list(patterns[np.random.randint(len(patterns))])
    np.random.shuffle(sizes)

    grid = np.zeros(INPUT_SHAPE, dtype=int)
    for size in sorted(sizes, reverse=True):
        shape = _random_connected_shape(size)
        if not _place_shape(grid, shape):
            return None

    if not _valid_input(grid, expect_tie=expect_tie):
        return None
    if not np.array_equal(main(grid), main(grid.copy())):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        expect_tie = bool(np.random.randint(2))
        grid = _generate_attempt(expect_tie)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
