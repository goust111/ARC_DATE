import random
from collections import Counter, deque

import numpy as np


BLACK = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CORNER_TO_DIR = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _decode_arrow(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    side = r1 - r0 + 1
    assert side >= 2 and c1 - c0 + 1 == side, "arrowhead bbox must be a square"
    mask = np.zeros((side, side), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    holes = np.argwhere(~mask)
    assert holes.shape == (1, 2), "arrowhead must be a full square with one missing corner"
    hr, hc = map(int, holes[0])
    assert (hr, hc) in {
        (0, 0),
        (0, side - 1),
        (side - 1, 0),
        (side - 1, side - 1),
    }, "missing cell must be a corner"

    if (hr, hc) == (0, 0):
        direction = (-1, -1)
        start = (r0 - 1, c0 - 1)
    elif (hr, hc) == (0, side - 1):
        direction = (-1, 1)
        start = (r0 - 1, c1 + 1)
    elif (hr, hc) == (side - 1, 0):
        direction = (1, -1)
        start = (r1 + 1, c0 - 1)
    else:
        direction = (1, 1)
        start = (r1 + 1, c1 + 1)

    steps = len(cells)
    return direction, start, steps


def _trace_bouncing_ray(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int], steps: int
) -> tuple[list[tuple[int, int]], int]:
    h, w = map(int, shape)
    r, c = map(int, start)
    dr, dc = map(int, direction)
    assert 0 <= r < h and 0 <= c < w, "ray must start from an in-bounds outer cell"

    path: list[tuple[int, int]] = []
    bounces = 0
    for _ in range(steps):
        path.append((r, c))
        nr, nc = r + dr, c + dc
        bounced = False
        if nr < 0 or nr >= h:
            dr = -dr
            nr = r + dr
            bounced = True
        if nc < 0 or nc >= w:
            dc = -dc
            nc = c + dc
            bounced = True
        if bounced:
            bounces += 1
        r, c = nr, nc
    return path, bounces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each missing-corner square arrowhead by a same-color diagonal ray whose length equals the arrowhead area.

    Input structure:
    - The grid has a black background with several disconnected monochrome arrowhead sprites.
    - Each sprite is a solid square with exactly one missing corner, and that missing corner indicates a diagonal direction.
    - The layout keeps arrowheads and launched rays from interfering with one another.

    Rule:
    - Detect each monochrome component and verify it is a square missing exactly one corner.
    - Use the missing-corner position to infer the diagonal launch direction, and count the component pixels to get the step budget.
    - Starting from the outer cell just beyond the missing corner, walk that many diagonal steps, reflecting off grid boundaries by flipping the blocked direction component, and paint every visited cell with the arrow color.

    Key insight:
    - The arrowhead's area is not decoration: it is the ray length budget, while the missing corner simultaneously encodes both launch corner and diagonal direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"

    out = g.copy()
    for color, cells in _find_components(g, bg):
        direction, start, steps = _decode_arrow(cells)
        path, _bounces = _trace_bouncing_ray(g.shape, start, direction, steps)
        for r, c in path:
            out[r, c] = color
    return out


def _arrow_cells(top: int, left: int, side: int, missing_corner: str) -> list[tuple[int, int]]:
    hole = {
        "tl": (0, 0),
        "tr": (0, side - 1),
        "bl": (side - 1, 0),
        "br": (side - 1, side - 1),
    }[missing_corner]
    cells = []
    for rr in range(side):
        for cc in range(side):
            if (rr, cc) != hole:
                cells.append((top + rr, left + cc))
    return cells


def _launch_from_bbox(top: int, left: int, side: int, missing_corner: str):
    bottom = top + side - 1
    right = left + side - 1
    if missing_corner == "tl":
        return (-1, -1), (top - 1, left - 1)
    if missing_corner == "tr":
        return (-1, 1), (top - 1, right + 1)
    if missing_corner == "bl":
        return (1, -1), (bottom + 1, left - 1)
    return (1, 1), (bottom + 1, right + 1)


def _mark_chebyshev(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                mask[nr, nc] = True


def _place_one_arrow(
    shape: tuple[int, int],
    grid: np.ndarray,
    object_exact: np.ndarray,
    object_pad: np.ndarray,
    ray_exact: np.ndarray,
    ray_pad: np.ndarray,
    require_bounce: bool | None,
) -> bool:
    h, w = map(int, shape)
    corners = ["tl", "tr", "bl", "br"]
    for _ in range(250):
        if require_bounce is True:
            side = random.choice([3, 4, 5, 5])
        elif require_bounce is False:
            side = random.choice([2, 3, 3, 4])
        else:
            side = random.choice([2, 3, 4, 5])
        if side + 2 > h or side + 2 > w:
            continue

        top = random.randint(1, h - side - 1)
        left = random.randint(1, w - side - 1)
        missing_corner = random.choice(corners)
        color = random.choice(PALETTE)
        cells = _arrow_cells(top, left, side, missing_corner)
        if any(object_pad[r, c] or ray_pad[r, c] for r, c in cells):
            continue

        direction, start = _launch_from_bbox(top, left, side, missing_corner)
        steps = side * side - 1
        path, bounces = _trace_bouncing_ray(shape, start, direction, steps)
        if require_bounce is True and bounces == 0:
            continue
        if require_bounce is False and bounces != 0:
            continue
        if len(set(path)) != len(path):
            continue
        if any(object_exact[r, c] or object_pad[r, c] or ray_pad[r, c] for r, c in path):
            continue
        own_pad = np.zeros_like(object_exact)
        _mark_chebyshev(own_pad, cells)
        if any(own_pad[r, c] for r, c in path[1:]):
            continue

        for r, c in cells:
            grid[r, c] = color
            object_exact[r, c] = True
        _mark_chebyshev(object_pad, cells)
        for r, c in path:
            ray_exact[r, c] = True
        _mark_chebyshev(ray_pad, path)
        return True
    return False


def _branch_requirements(count: int) -> list[bool | None]:
    mode = random.choices(
        ["mixed", "bounce", "straight", "free"],
        weights=[5, 3, 2, 2],
        k=1,
    )[0]
    if mode == "bounce":
        return [True] * count
    if mode == "straight":
        return [False] * count
    if mode == "mixed":
        reqs: list[bool | None] = [True, False]
        while len(reqs) < count:
            reqs.append(random.choice([None, True, False]))
        random.shuffle(reqs)
        return reqs
    return [None] * count


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 24)
    w = random.randint(15, 24)
    count = random.randint(2, 4)

    grid = np.zeros((h, w), dtype=int)
    object_exact = np.zeros((h, w), dtype=bool)
    object_pad = np.zeros((h, w), dtype=bool)
    ray_exact = np.zeros((h, w), dtype=bool)
    ray_pad = np.zeros((h, w), dtype=bool)

    requirements = _branch_requirements(count)
    for require_bounce in requirements:
        if not _place_one_arrow((h, w), grid, object_exact, object_pad, ray_exact, ray_pad, require_bounce):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-bounce arrow grid")
