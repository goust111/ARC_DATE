import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

EVEN_BLOCK = np.array([[BLUE, RED], [RED, BLUE]], dtype=int)
ODD_BLOCK = np.array([[RED, BLUE], [BLUE, RED]], dtype=int)

GEN_ATTEMPTS = 500
MIN_H = 5
MAX_H = 7
MIN_W = 5
MAX_W = 8
MIN_FG = 10
MAX_FG = 15


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _count_foreground_neighbors(grid: np.ndarray, r: int, c: int, *, conn8: bool = False) -> int:
    total = 0
    nbrs = _neighbors8(r, c) if conn8 else _neighbors4(r, c)
    for nr, nc in nbrs:
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
            total += 1
    return total


def _connected_component_size(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return 0
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen)


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and _connected_component_size(mask) == int(mask.sum())


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    bg_seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not bg_seen[r, c]:
                bg_seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not bg_seen[r, c]:
                bg_seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and not mask[nr, nc] and not bg_seen[nr, nc]:
                bg_seen[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any((~mask) & (~bg_seen)))


def _bbox_area(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _matches_spatial_proxy(grid: np.ndarray, proxy) -> bool:
    fg = np.argwhere(grid == GREY)
    if len(fg) == 0:
        return False
    same = True
    inverse = True
    for r, c in fg:
        parity = _count_foreground_neighbors(grid, int(r), int(c)) % 2
        val = proxy(int(r), int(c))
        if parity != val:
            same = False
        if parity != 1 - val:
            inverse = False
    return same or inverse


def _semantic_ok(grid: np.ndarray) -> bool:
    if grid.ndim != 2:
        return False
    if set(map(int, np.unique(grid))) != {BLACK, GREY}:
        return False

    fg_mask = grid == GREY
    fg_count = int(fg_mask.sum())
    if not (MIN_FG <= fg_count <= MAX_FG):
        return False
    if not _is_connected(fg_mask):
        return False
    if _has_hole(fg_mask):
        return False
    if fg_count >= grid.size:
        return False
    if _bbox_area(fg_mask) == fg_count:
        return False

    counts4 = []
    counts8 = []
    for r, c in np.argwhere(fg_mask):
        counts4.append(_count_foreground_neighbors(grid, int(r), int(c)))
        counts8.append(_count_foreground_neighbors(grid, int(r), int(c), conn8=True))

    parities4 = {count % 2 for count in counts4}
    if parities4 != {0, 1}:
        return False
    if 4 not in counts4:
        return False
    if not any(count % 2 == 1 for count in counts4):
        return False
    if not any((a % 2) != (b % 2) for a, b in zip(counts4, counts8)):
        return False

    proxies = (
        lambda r, c: r % 2,
        lambda r, c: c % 2,
        lambda r, c: (r + c) % 2,
    )
    if any(_matches_spatial_proxy(grid, proxy) for proxy in proxies):
        return False

    out = main(grid)
    seen_even = False
    seen_odd = False
    for r, c in np.argwhere(fg_mask):
        block = out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2]
        if np.array_equal(block, EVEN_BLOCK):
            seen_even = True
        elif np.array_equal(block, ODD_BLOCK):
            seen_odd = True
        else:
            return False

    if not (seen_even and seen_odd):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each grey pixel into a parity-chosen 2x2 checkerboard block.

    Input structure:
    - The grid is a small black background with grey foreground pixels.
    - Grey pixels form one irregular 4-connected blob with no holes.
    - The local 4-neighborhood in the original-resolution grid is the only context that matters.

    Rule:
    - Double the height and width to build the output.
    - Each black pixel becomes a black 2x2 block.
    - For each grey pixel, count its grey 4-neighbors in the input grid.
    - If that count is even, write [[blue, red], [red, blue]] in the corresponding 2x2 block.
    - If that count is odd, write [[red, blue], [blue, red]] in the corresponding 2x2 block.

    Key insight:
    - The checkerboard orientation is not global and not position-based; it is chosen separately for each source pixel from the parity of its original 4-neighbor count.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((2 * h, 2 * w), BLACK, dtype=int)

    for r, c in np.argwhere(g == GREY):
        rr = int(r)
        cc = int(c)
        count = _count_foreground_neighbors(g, rr, cc)
        out[2 * rr : 2 * rr + 2, 2 * cc : 2 * cc + 2] = EVEN_BLOCK if count % 2 == 0 else ODD_BLOCK

    return out


def _weighted_choice(candidates: list[tuple[int, int]], shape: set[tuple[int, int]]) -> tuple[int, int]:
    weights = []
    for r, c in candidates:
        touch = sum((nr, nc) in shape for nr, nc in _neighbors4(r, c))
        if touch == 1:
            weights.append(5)
        elif touch == 2:
            weights.append(3)
        else:
            weights.append(1)
    return random.choices(candidates, weights=weights, k=1)[0]


def _grow_shape(h: int, w: int, target: int) -> set[tuple[int, int]]:
    r0 = random.randint(0, h - 3)
    c0 = random.randint(0, w - 3)
    shape = {(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)}

    while len(shape) < target:
        frontier = set()
        for r, c in shape:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds((h, w), nr, nc) and (nr, nc) not in shape:
                    frontier.add((nr, nc))
        if not frontier:
            break
        shape.add(_weighted_choice(list(frontier), shape))

    return shape


def _attempt() -> np.ndarray | None:
    h = random.randint(MIN_H, MAX_H)
    w = random.randint(MIN_W, MAX_W)
    target = random.randint(MIN_FG, min(MAX_FG, h * w - 4))

    shape = _grow_shape(h, w, target)
    if len(shape) != target:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, c in shape:
        grid[r, c] = GREY

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
