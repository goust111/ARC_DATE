from __future__ import annotations

import random

import numpy as np


BLACK = 0
ORANGE = 7
AZURE = 8
TOWER_COLORS = (1, 2, 3, 4, 5, 6, 9)
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _flood_same_color(
    grid: np.ndarray,
    start: tuple[int, int],
    visited: np.ndarray,
) -> list[tuple[int, int]]:
    color = int(grid[start])
    stack = [start]
    visited[start] = True
    coords: list[tuple[int, int]] = []

    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for dr, dc in CARDINALS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if not visited[nr, nc] and int(grid[nr, nc]) == color:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

    return coords


def _find_towers(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    visited = np.zeros(g.shape, dtype=bool)
    towers: list[dict[str, int]] = []

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color in (BLACK, ORANGE, AZURE) or visited[r, c]:
                continue

            coords = _flood_same_color(g, (r, c), visited)
            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0 = min(rows)
            c0 = min(cols)
            r1 = max(rows)
            c1 = max(cols)

            assert len(coords) == 9, "each tower must contain exactly 9 cells"
            assert r1 - r0 == 2 and c1 - c0 == 2, "each tower must have a 3x3 bbox"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "each tower must be solid"

            towers.append({"color": color, "r0": r0, "c0": c0, "r": r0 + 1, "c": c0 + 1})

    assert towers, "expected at least one tower"
    towers.sort(key=lambda tower: (tower["r0"], tower["c0"]))

    rows = [tower["r"] for tower in towers]
    cols = [tower["c"] for tower in towers]
    assert len(set(rows)) == len(rows), "tower centers must use distinct rows"
    assert len(set(cols)) == len(cols), "tower centers must use distinct columns"
    return towers


def _tower_mask(shape: tuple[int, int], towers: list[dict[str, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for tower in towers:
        r0 = tower["r0"]
        c0 = tower["c0"]
        mask[r0 : r0 + 3, c0 : c0 + 3] = True
    return mask


def _edge_rays(tower: dict[str, int]) -> list[tuple[int, int, int, int]]:
    r0 = tower["r0"]
    c0 = tower["c0"]
    return [
        (r0, c0 + 1, -1, 0),
        (r0 + 2, c0 + 1, 1, 0),
        (r0 + 1, c0, 0, -1),
        (r0 + 1, c0 + 2, 0, 1),
    ]


def _simulate(
    grid: np.ndarray,
    *,
    reverse: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    towers = _find_towers(g)
    if reverse:
        towers = list(reversed(towers))
    tower_mask = _tower_mask(g.shape, towers)

    out = g.copy()
    stats = {"orange": 0, "azure": 0, "tower": 0, "colored": 0, "edge": 0}

    for tower in towers:
        color = int(tower["color"])
        for sr, sc, dr, dc in _edge_rays(tower):
            r = sr + dr
            c = sc + dc
            stopped = False

            while 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                if tower_mask[r, c]:
                    stats["tower"] += 1
                    stopped = True
                    break

                value = int(out[r, c])
                if value == BLACK:
                    out[r, c] = color
                    r += dr
                    c += dc
                    continue

                if value == ORANGE:
                    out[r, c] = color
                    stats["orange"] += 1
                    r += dr
                    c += dc
                    continue

                if value == AZURE:
                    out[r, c] = color
                    stats["azure"] += 1
                    stopped = True
                    break

                stats["colored"] += 1
                stopped = True
                break

            if not stopped:
                stats["edge"] += 1

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit colored rays from 3x3 towers, with receiver color controlling pass-through versus absorption.

    Input structure:
    - Background is black.
    - Several solid 3x3 monochrome squares are signal towers; each tower color is distinct and is also the color of its rays.
    - Single-pixel receivers are orange or azure and sit in black space between towers.
    - Towers do not overlap, and different towers use distinct center rows and center columns.

    Rule:
    - Sort towers from top to bottom, then from left to right.
    - From each tower's four side-center pixels, cast one ray outward in the four cardinal directions.
    - Each ray recolors black cells to the tower color as it advances.
    - When a ray reaches an orange receiver, recolor that receiver and keep going; when it reaches an azure receiver, recolor that receiver and stop.
    - Rays also stop when they reach another tower, any earlier non-black paint left by a previous ray, or the grid boundary.

    Key insight:
    - Orange and azure receivers encode two different interactions, and earlier towers claim cells first, so the tower processing order changes which later rays can continue.
    """
    return _simulate(grid)[0]


def _bbox_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _towers_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) <= 2 and abs(a[1] - b[1]) <= 2


def _ray_cells_to_tower(
    shape: tuple[int, int],
    center: tuple[int, int],
    direction: tuple[int, int],
    tower_mask: np.ndarray,
) -> list[tuple[int, int]]:
    r, c = center
    dr, dc = direction

    if dr == -1:
        r -= 2
    elif dr == 1:
        r += 2
    elif dc == -1:
        c -= 2
    else:
        c += 2

    cells: list[tuple[int, int]] = []
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        if tower_mask[r, c]:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _render_scene(
    shape: tuple[int, int],
    centers: list[tuple[int, int]],
    colors: list[int],
    receivers: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for (r, c), color in zip(centers, colors, strict=True):
        grid[r - 1 : r + 2, c - 1 : c + 2] = color
    for (r, c), color in receivers.items():
        grid[r, c] = color
    return grid


def _place_extra_centers(
    rng: random.Random,
    shape: tuple[int, int],
    centers: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    target_count: int,
) -> list[tuple[int, int]] | None:
    rows = {r for r, _c in centers}
    cols = {c for _r, c in centers}
    placed = list(centers)

    for _ in range(400):
        if len(placed) == target_count:
            return placed

        r = rng.randint(2, shape[0] - 3)
        c = rng.randint(2, shape[1] - 3)
        candidate = (r, c)

        if r in rows or c in cols:
            continue
        if any(_towers_overlap(candidate, other) for other in placed):
            continue
        if _bbox_cells(candidate) & protected:
            continue

        placed.append(candidate)
        rows.add(r)
        cols.add(c)

    return None


def _candidate_receiver_cells(
    shape: tuple[int, int],
    centers: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    tower_mask = np.zeros(shape, dtype=bool)
    for center in centers:
        for cell in _bbox_cells(center):
            tower_mask[cell] = True

    all_cells: set[tuple[int, int]] = set()
    continuable: set[tuple[int, int]] = set()
    for center in centers:
        for direction in CARDINALS:
            cells = _ray_cells_to_tower(shape, center, direction, tower_mask)
            all_cells.update(cells)
            continuable.update(cells[:-1])
    return all_cells, continuable


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(18, 27), rng.randint(18, 27))
    need_tower_hit = rng.random() < 0.5
    tower_count = 2 + int(need_tower_hit) + rng.randint(0, 1)

    r_a_low = 5
    r_a_high = shape[0] - 6
    c_a_low = 6 if need_tower_hit else 3
    c_a_high = shape[1] - 6
    if r_a_low > r_a_high or c_a_low > c_a_high:
        return None

    r_a = rng.randint(r_a_low, r_a_high)
    c_a = rng.randint(c_a_low, c_a_high)

    r_b_low = r_a + 3
    r_b_high = min(shape[0] - 3, r_a + 8)
    c_b_low = c_a + 3
    c_b_high = min(shape[1] - 3, c_a + 8)
    if r_b_low > r_b_high or c_b_low > c_b_high:
        return None

    r_b = rng.randint(r_b_low, r_b_high)
    c_b = rng.randint(c_b_low, c_b_high)

    centers = [(r_a, c_a), (r_b, c_b)]
    if need_tower_hit:
        c_c_high = c_a - 4
        if c_c_high < 2:
            return None
        c_c = rng.randint(2, c_c_high)
        centers.append((r_a + 1, c_c))

    protected: set[tuple[int, int]] = set()
    protected.update((r_a, c) for c in range(c_a + 2, c_b + 1))
    protected.update((r, c_b) for r in range(r_a, r_b - 1))
    protected.update((r, c_a) for r in range(0, r_a - 1))
    if need_tower_hit:
        c_c = centers[2][1]
        protected.update((r_a, c) for c in range(c_c + 1, c_a - 1))

    placed = _place_extra_centers(rng, shape, centers, protected, tower_count)
    if placed is None:
        return None
    centers = placed

    colors = rng.sample(list(TOWER_COLORS), len(centers))
    tower_mask = np.zeros(shape, dtype=bool)
    for center in centers:
        for cell in _bbox_cells(center):
            tower_mask[cell] = True

    a_up_cells = _ray_cells_to_tower(shape, centers[0], (-1, 0), tower_mask)
    other_rows = {r for r, _c in centers[1:]}
    orange_candidates = [
        cell
        for cell in a_up_cells[:-1]
        if cell[0] not in other_rows
    ]
    if not orange_candidates:
        return None
    orange_receiver = rng.choice(orange_candidates)

    azure_receiver = (r_a, c_b)
    receivers: dict[tuple[int, int], int] = {
        orange_receiver: ORANGE,
        azure_receiver: AZURE,
    }

    all_path_cells, continuable_cells = _candidate_receiver_cells(shape, centers)
    extra_candidates = list(all_path_cells - protected - set(receivers))
    rng.shuffle(extra_candidates)
    extra_receivers = rng.randint(0, min(3, len(extra_candidates)))
    for cell in extra_candidates[:extra_receivers]:
        if cell in continuable_cells and rng.random() < 0.5:
            receivers[cell] = ORANGE
        else:
            receivers[cell] = AZURE

    grid = _render_scene(shape, centers, colors, receivers)

    try:
        forward, stats = _simulate(grid)
        reverse, _ = _simulate(grid, reverse=True)
    except AssertionError:
        return None

    if np.array_equal(forward, reverse):
        return None
    if stats["orange"] < 1 or stats["azure"] < 1:
        return None
    if stats["colored"] < 1 or stats["edge"] < 1:
        return None
    if need_tower_hit and stats["tower"] < 1:
        return None
    if not np.any(forward != grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
