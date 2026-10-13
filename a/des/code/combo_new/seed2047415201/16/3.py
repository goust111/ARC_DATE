from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
FILL_COLORS = tuple(c for c in range(1, 10) if c != GREY)
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GENERATION_ATTEMPTS = 500


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((1, 0), (-1, 0), (0, 1), (0, -1))
    return tuple(
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    )


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    deltas = _neighbors(connectivity)
    comps: list[np.ndarray] = []

    for sr in range(height):
        for sc in range(width):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in deltas:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        if not seen[rr, cc] and bool(mask[rr, cc]):
                            seen[rr, cc] = True
                            q.append((rr, cc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _u_cells(direction: str, r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if direction == "up":
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
    elif direction == "down":
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
        cells.update((r0, c) for c in range(c0, c1 + 1))
    elif direction == "left":
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
    elif direction == "right":
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
        cells.update((r, c0) for r in range(r0, r1 + 1))
    return cells


def _interior_cells(direction: str, r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if direction == "up":
        return [(r, c) for r in range(r0, r1) for c in range(c0 + 1, c1)]
    if direction == "down":
        return [(r, c) for r in range(r0 + 1, r1 + 1) for c in range(c0 + 1, c1)]
    if direction == "left":
        return [(r, c) for r in range(r0 + 1, r1) for c in range(c0, c1)]
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1 + 1)]


def _classify_container(coords: np.ndarray) -> tuple[str, tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    actual = {(int(r), int(c)) for r, c in coords}
    matches = [
        direction
        for direction in ("up", "down", "left", "right")
        if actual == _u_cells(direction, r0, c0, r1, c1)
    ]
    if len(matches) != 1:
        return None
    return matches[0], (r0, c0, r1, c1)


def _find_containers(grid: np.ndarray) -> list[dict[str, object]]:
    containers: list[dict[str, object]] = []
    for comp in _components(grid == GREY, connectivity=4):
        classified = _classify_container(comp)
        if classified is None:
            continue
        direction, bbox = classified
        r0, c0, r1, c1 = bbox
        interior = _interior_cells(direction, r0, c0, r1, c1)
        fill = [
            (r, c, int(grid[r, c]))
            for r, c in interior
            if int(grid[r, c]) not in (BLACK, GREY)
        ]
        containers.append(
            {
                "direction": direction,
                "bbox": bbox,
                "interior": interior,
                "fill": fill,
            }
        )
    return containers


def _container_sort_key(info: dict[str, object]) -> tuple[int, int, int]:
    direction = str(info["direction"])
    r0, c0, r1, c1 = info["bbox"]  # type: ignore[misc]
    if direction == "up":
        return (0, int(r0), int(c0))
    if direction == "down":
        return (1, -int(r1), int(c0))
    if direction == "left":
        return (2, int(c0), int(r0))
    return (3, -int(c1), int(r0))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each colored filling out of its Grey U-shaped container through the open side.

    Input structure:
    - The grid background is Black.
    - Grey pixels form separated one-cell-thick rectangular U containers.
    - Each container opens up, down, left, or right, and its interior may contain one or more rows or columns of colored filling pixels.

    Rule:
    - Identify every Grey U component and infer its opening direction from the missing side of the U.
    - Clear the container interiors to Black while preserving all Grey wall pixels.
    - Translate each container's colored interior pixels as one block along the opening direction until the next step would leave the grid or hit another non-Black pixel.
    - Repaint the translated filling in its original relative arrangement and colors.

    Key insight:
    The missing side of each Grey U is a direction code: the filling is not erased, it slides out through that opening while keeping its stripe order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    containers = _find_containers(g)
    out = np.array(g, dtype=int, copy=True)

    for info in containers:
        for r, c in info["interior"]:  # type: ignore[assignment]
            out[int(r), int(c)] = BLACK

    for info in sorted(containers, key=_container_sort_key):
        fill = info["fill"]  # type: ignore[assignment]
        if not fill:
            continue
        direction = str(info["direction"])
        dr, dc = DIRECTIONS[direction]

        step = 0
        while True:
            nxt = step + 1
            ok = True
            for r, c, _value in fill:
                rr = int(r) + nxt * dr
                cc = int(c) + nxt * dc
                if not (0 <= rr < height and 0 <= cc < width):
                    ok = False
                    break
                if int(out[rr, cc]) != BLACK:
                    ok = False
                    break
            if not ok:
                break
            step = nxt

        for r, c, value in fill:
            rr = int(r) + step * dr
            cc = int(c) + step * dc
            if 0 <= rr < height and 0 <= cc < width and int(out[rr, cc]) == BLACK:
                out[rr, cc] = int(value)

    return out


def _stripe_colors(depth: int, rng: random.Random) -> list[int]:
    if depth <= 0:
        return []
    if depth == 1 or rng.random() < 0.3:
        palette = [rng.choice(FILL_COLORS)]
    else:
        palette = rng.sample(FILL_COLORS, rng.randint(2, min(4, depth, len(FILL_COLORS))))
    colors = [rng.choice(palette) for _ in range(depth)]
    if len(palette) >= 2 and len(set(colors)) == 1:
        colors[-1] = rng.choice([c for c in palette if c != colors[0]])
    return colors


def _make_container_patch(direction: str, rng: random.Random) -> tuple[np.ndarray, int]:
    if direction in ("up", "down"):
        h = rng.randint(4, 7)
        w = rng.randint(4, 8)
        max_depth = min(4, h - 1)
    else:
        h = rng.randint(4, 8)
        w = rng.randint(4, 7)
        max_depth = min(4, w - 1)

    depth = rng.randint(1, max_depth)
    colors = _stripe_colors(depth, rng)
    patch = np.zeros((h, w), dtype=int)

    if direction == "up":
        patch[:, 0] = GREY
        patch[:, -1] = GREY
        patch[-1, :] = GREY
        for i, r in enumerate(range(h - 1 - depth, h - 1)):
            patch[r, 1:-1] = colors[i]
    elif direction == "down":
        patch[:, 0] = GREY
        patch[:, -1] = GREY
        patch[0, :] = GREY
        for i, r in enumerate(range(1, 1 + depth)):
            patch[r, 1:-1] = colors[i]
    elif direction == "left":
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, -1] = GREY
        for i, c in enumerate(range(w - 1 - depth, w - 1)):
            patch[1:-1, c] = colors[i]
    else:
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        for i, c in enumerate(range(1, 1 + depth)):
            patch[1:-1, c] = colors[i]

    return patch, depth


def _placement_ranges(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    direction: str,
    depth: int,
) -> tuple[range, range] | None:
    height, width = grid_shape
    ph, pw = patch_shape
    r_low, r_high = 0, height - ph
    c_low, c_high = 0, width - pw

    if direction == "up":
        r_low = max(r_low, depth)
    elif direction == "down":
        r_high = min(r_high, height - depth - ph)
    elif direction == "left":
        c_low = max(c_low, depth)
    else:
        c_high = min(c_high, width - depth - pw)

    if r_low > r_high or c_low > c_high:
        return None
    return range(r_low, r_high + 1), range(c_low, c_high + 1)


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    ph, pw = patch.shape
    mask = patch != BLACK
    if bool((grid[r0 : r0 + ph, c0 : c0 + pw][mask] != BLACK).any()):
        return False

    height, width = map(int, grid.shape)
    rr0, rr1 = max(0, r0 - 1), min(height, r0 + ph + 1)
    cc0, cc1 = max(0, c0 - 1), min(width, c0 + pw + 1)
    expanded = np.zeros((rr1 - rr0, cc1 - cc0), dtype=bool)
    pr0, pc0 = r0 - rr0, c0 - cc0
    expanded[pr0 : pr0 + ph, pc0 : pc0 + pw] = mask
    dilated = expanded.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rs0 = max(0, dr)
            rs1 = expanded.shape[0] + min(0, dr)
            cs0 = max(0, dc)
            cs1 = expanded.shape[1] + min(0, dc)
            td0 = max(0, -dr)
            td1 = td0 + (rs1 - rs0)
            tc0 = max(0, -dc)
            tc1 = tc0 + (cs1 - cs0)
            dilated[td0:td1, tc0:tc1] |= expanded[rs0:rs1, cs0:cs1]
    return not bool((grid[rr0:rr1, cc0:cc1] != BLACK)[dilated].any())


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    mask = patch != BLACK
    view = grid[r0 : r0 + ph, c0 : c0 + pw]
    view[mask] = patch[mask]


def _black_is_dominant(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts[BLACK]) > max(int(v) for i, v in enumerate(counts) if i != BLACK)


def _generated_grid_is_valid(grid: np.ndarray) -> bool:
    containers = _find_containers(grid)
    if not containers:
        return False
    fill_count = sum(len(info["fill"]) for info in containers)  # type: ignore[arg-type]
    if fill_count == 0 or not _black_is_dominant(grid):
        return False

    try:
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape or not _black_is_dominant(out):
        return False
    if np.array_equal(out, grid):
        return False
    if int(np.sum(out == GREY)) != int(np.sum(grid == GREY)):
        return False
    if int(np.sum((out != BLACK) & (out != GREY))) != int(fill_count):
        return False

    for info in containers:
        for r, c in info["interior"]:  # type: ignore[assignment]
            if int(out[int(r), int(c)]) != BLACK:
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_containers = rng.randint(1, 4)
    min_side = 12 + 2 * n_containers
    height = rng.randint(min_side, 30)
    width = rng.randint(min_side, 30)
    grid = np.full((height, width), BLACK, dtype=int)

    directions = list(DIRECTIONS)
    rng.shuffle(directions)
    while len(directions) < n_containers:
        directions.append(rng.choice(tuple(DIRECTIONS)))
    directions = directions[:n_containers]
    rng.shuffle(directions)

    for direction in directions:
        placed = False
        for _ in range(200):
            patch, depth = _make_container_patch(direction, rng)
            ranges = _placement_ranges(grid.shape, patch.shape, direction, depth)
            if ranges is None:
                continue
            rows, cols = ranges
            r0 = rng.choice(tuple(rows))
            c0 = rng.choice(tuple(cols))
            if not _can_place(grid, patch, r0, c0):
                continue

            trial = grid.copy()
            _paste(trial, patch, r0, c0)
            if _generated_grid_is_valid(trial):
                grid = trial
                placed = True
                break
        if not placed:
            return None

    if not _generated_grid_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-pouring input")
