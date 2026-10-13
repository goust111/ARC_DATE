import random
from collections import deque

import numpy as np


AZURE = 8
MAROON = 9
NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == int(bg):
                continue

            color = int(grid[sr, sc])
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords = []

            while q:
                r, c = q.popleft()
                coords.append((int(r), int(c)))
                for dr, dc in NEIGHBORS_4:
                    nr, nc = int(r + dr), int(c + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each foreground object to its point-reflected quadrant and recolor nearby objects.

    Input structure:
    - The grid is an odd-sized square with Azure background.
    - Foreground objects are monochrome 4-connected components in non-Azure colors.
    - Every object lies fully inside one quadrant; the center row and center column are Azure.

    Rule:
    - Clear the input canvas to Azure.
    - For every foreground object, rotate all of its cells 180 degrees around the grid center.
    - If the object's bounding-box center has Manhattan distance greater than half the side length from the grid center, keep its color.
    - Otherwise draw the reflected object in Maroon, and keep the center row and center column Azure.

    Key insight:
    - The same point reflection is applied to every object; only the visible near-versus-far distance from the center controls the recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    n, w = g.shape
    assert n == w and n % 2 == 1

    center = n // 2
    assert np.all(g[center, :] == AZURE)
    assert np.all(g[:, center] == AZURE)

    out = np.full_like(g, AZURE)
    for comp in _components_by_color(g, AZURE):
        r0, c0, r1, c1 = comp["bbox"]
        rows = [r for r, _c in comp["coords"]]
        cols = [c for _r, c in comp["coords"]]
        assert all(r < center for r in rows) or all(r > center for r in rows)
        assert all(c < center for c in cols) or all(c > center for c in cols)

        # Double all distances so half-cell bbox centers stay exact.
        distance2 = abs((r0 + r1) - 2 * center) + abs((c0 + c1) - 2 * center)
        color = int(comp["color"]) if int(distance2) > int(n) else MAROON

        for r, c in comp["coords"]:
            rr = int(n - 1 - r)
            cc = int(n - 1 - c)
            assert 0 <= rr < n and 0 <= cc < n
            assert rr != center and cc != center
            out[rr, cc] = int(color)

    return out


# --------------- generator ---------------

GRID_SIDES = (13, 15, 17, 19, 21, 23, 25)
COLOR_POOL = (0, 1, 2, 3, 4, 5, 6, 7)
QUADRANTS = ("UL", "UR", "LL", "LR")
GEN_ATTEMPTS = 300
SHAPE_ATTEMPTS = 80


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _random_connected_mask(rng: random.Random, side: int) -> np.ndarray:
    max_dim = min(5, max(2, side // 5 + 1))
    h = rng.randint(1, max_dim)
    w = rng.randint(1, max_dim)
    if h == 1 and w == 1 and rng.random() < 0.85:
        h, w = rng.choice([(1, 2), (2, 1), (2, 2)])

    target = rng.randint(1 if h * w == 1 else 2, min(h * w, 8))
    cells = {(rng.randrange(h), rng.randrange(w))}
    misses = 0
    while len(cells) < target and misses < 100:
        r, c = rng.choice(tuple(cells))
        dr, dc = rng.choice(NEIGHBORS_4)
        nr, nc = int(r + dr), int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
        else:
            misses += 1

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    mask = _crop_mask(mask)

    for _ in range(rng.randrange(4)):
        mask = np.rot90(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    return _crop_mask(mask)


def _ranges_for_quadrant(side: int, shape: tuple[int, int], quadrant: str) -> tuple[range, range] | None:
    center = side // 2
    h, w = shape
    if quadrant == "UL":
        rows = range(0, center - h + 1)
        cols = range(0, center - w + 1)
    elif quadrant == "UR":
        rows = range(0, center - h + 1)
        cols = range(center + 1, side - w + 1)
    elif quadrant == "LL":
        rows = range(center + 1, side - h + 1)
        cols = range(0, center - w + 1)
    else:
        rows = range(center + 1, side - h + 1)
        cols = range(center + 1, side - w + 1)

    if len(rows) == 0 or len(cols) == 0:
        return None
    return rows, cols


def _distance_branch(side: int, top: int, left: int, h: int, w: int) -> str:
    center = side // 2
    distance2 = abs((2 * top + h - 1) - 2 * center) + abs((2 * left + w - 1) - 2 * center)
    return "far" if int(distance2) > int(side) else "near"


def _has_clear_padding(occupied: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    side = occupied.shape[0]
    for r, c in coords:
        for rr in range(max(0, int(r) - 1), min(side, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(side, int(c) + 2)):
                if occupied[rr, cc]:
                    return False
    return True


def _place_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    quadrant: str,
    branch: str,
    rng: random.Random,
) -> bool:
    side = grid.shape[0]
    ranges = _ranges_for_quadrant(side, mask.shape, quadrant)
    if ranges is None:
        return False
    row_range, col_range = ranges
    mh, mw = mask.shape

    candidates = []
    for top in row_range:
        for left in col_range:
            if _distance_branch(side, int(top), int(left), int(mh), int(mw)) == branch:
                candidates.append((int(top), int(left)))
    rng.shuffle(candidates)

    mask_pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    for top, left in candidates:
        coords = [(int(top + r), int(left + c)) for r, c in mask_pts]
        if not _has_clear_padding(occupied, coords):
            continue
        for r, c in coords:
            grid[r, c] = int(color)
            occupied[r, c] = True
        return True
    return False


def _object_count_for_side(side: int, rng: random.Random) -> int:
    if side <= 13:
        return 4
    if side <= 17:
        return rng.randint(4, 5)
    if side <= 21:
        return rng.randint(4, 6)
    return rng.randint(5, 7)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(GRID_SIDES)
    count = _object_count_for_side(side, rng)
    colors = rng.sample(COLOR_POOL, count)

    extra_near = rng.randrange(2)
    near_count = count // 2 + (1 if count % 2 and extra_near else 0)
    far_count = count - near_count
    branches = ["near"] * near_count + ["far"] * far_count
    rng.shuffle(branches)

    quadrants = list(QUADRANTS)
    rng.shuffle(quadrants)
    while len(quadrants) < count:
        quadrants.append(rng.choice(QUADRANTS))
    rng.shuffle(quadrants)

    grid = np.full((side, side), AZURE, dtype=int)
    occupied = np.zeros((side, side), dtype=bool)

    specs = list(zip(colors, quadrants, branches))
    rng.shuffle(specs)
    for color, quadrant, branch in specs:
        placed = False
        for _ in range(SHAPE_ATTEMPTS):
            mask = _random_connected_mask(rng, side)
            if _place_object(grid, occupied, mask, int(color), quadrant, branch, rng):
                placed = True
                break
        if not placed:
            return None

    center = side // 2
    if np.any(grid[center, :] != AZURE) or np.any(grid[:, center] != AZURE):
        return None

    actual_branches = set()
    for comp in _components_by_color(grid, AZURE):
        r0, c0, r1, c1 = comp["bbox"]
        actual_branches.add(_distance_branch(side, r0, c0, r1 - r0 + 1, c1 - c0 + 1))
    if actual_branches != {"near", "far"}:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid point-reflection distance input")
