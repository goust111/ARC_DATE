import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2


GRID_SIDE_RANGE = (13, 22)
MARGIN_RANGE = (1, 2)
TARGET_FILL_RANGE = (0.18, 0.34)
MIN_BBOX_SPAN_FRAC = 0.55
MIN_OCCLUDERS = 2
MAX_OCCLUDERS = 5
OCCLUDER_SIZE_RANGE = (2, 8)
OCCLUDER_RADIUS_RANGE = (2, 4)
GENERATION_ATTEMPTS = 400

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove red occluders, then restore the blue figure by 90-degree rotational symmetry.

    Input structure
    - Black background.
    - One large central blue pattern that should be invariant under 90-degree rotation.
    - Several red irregular blobs occlude parts of the blue pattern and nearby background.

    Rule
    - Delete all red pixels.
    - Let every visible blue pixel vote for its full 90-degree rotation orbit.
    - Fill every cell in any orbit that contains at least one visible blue pixel.

    Key insight
    - The hidden figure is redundantly encoded four times by quarter-turn symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w, "90-degree rotational closure requires a square grid"

    cleaned = g.copy()
    cleaned[cleaned == RED] = BLACK
    blue_mask = cleaned == BLUE

    repaired = blue_mask.copy()
    for k in range(1, 4):
        repaired |= np.rot90(blue_mask, k)

    out = np.zeros_like(g)
    out[repaired] = BLUE
    return out


def _rotate90(cell: tuple[int, int], n: int) -> tuple[int, int]:
    r, c = cell
    return c, n - 1 - r


def _orbit(cell: tuple[int, int], n: int) -> tuple[tuple[int, int], ...]:
    coords = []
    cur = cell
    while cur not in coords:
        coords.append(cur)
        cur = _rotate90(cur, n)
    return tuple(sorted(coords))


def _all_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orb = _orbit((r, c), n)
            seen.update(orb)
            orbits.append(orb)
    return orbits


def _neighbors4(cell: tuple[int, int], n: int):
    r, c = cell
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            yield rr, cc


def _count_components(mask: np.ndarray, connectivity: int = 4) -> int:
    n, m = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    offsets = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        count += 1
        queue = deque([(r, c)])
        visited[r, c] = True
        while queue:
            rr, cc = queue.popleft()
            for dr, dc in offsets:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < n and 0 <= nc < m and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
    return count


def _bbox(mask: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(mask)
    rows = pts[:, 0]
    cols = pts[:, 1]
    return int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)


def _distance_to_center(orbit_coords: tuple[tuple[int, int], ...], n: int) -> float:
    center = (n - 1) / 2.0
    return float(sum((r - center) ** 2 + (c - center) ** 2 for r, c in orbit_coords) / len(orbit_coords))


def _make_blue_pattern(rng: random.Random, n: int) -> np.ndarray | None:
    margin = rng.randint(*MARGIN_RANGE)
    all_orbits = _all_orbits(n)
    usable = [
        orb
        for orb in all_orbits
        if all(margin <= r < n - margin and margin <= c < n - margin for r, c in orb)
    ]
    if not usable:
        return None

    orbit_set = {orb for orb in usable}
    orbit_to_neighbors: dict[tuple[tuple[int, int], ...], set[tuple[tuple[int, int], ...]]] = {orb: set() for orb in usable}
    for orb in usable:
        for cell in orb:
            for neigh in _neighbors4(cell, n):
                neigh_orb = _orbit(neigh, n)
                if neigh_orb in orbit_set and neigh_orb != orb:
                    orbit_to_neighbors[orb].add(neigh_orb)

    start = min(usable, key=lambda orb: _distance_to_center(orb, n))
    chosen = {start}
    mask = np.zeros((n, n), dtype=bool)
    for r, c in start:
        mask[r, c] = True

    target = rng.randint(int(n * n * TARGET_FILL_RANGE[0]), int(n * n * TARGET_FILL_RANGE[1]))
    max_steps = len(usable) * 3
    steps = 0
    while int(mask.sum()) < target and steps < max_steps:
        steps += 1
        frontier = []
        for orb in chosen:
            frontier.extend(candidate for candidate in orbit_to_neighbors[orb] if candidate not in chosen)
        if not frontier:
            break

        unique_frontier = list(dict.fromkeys(frontier))
        weights = []
        for orb in unique_frontier:
            touching = sum(1 for neighbor in orbit_to_neighbors[orb] if neighbor in chosen)
            jitter = rng.uniform(0.2, 1.2)
            weights.append((touching + 1.0) * jitter)
        orb = rng.choices(unique_frontier, weights=weights, k=1)[0]
        chosen.add(orb)
        for r, c in orb:
            mask[r, c] = True

    area = int(mask.sum())
    if area < int(n * n * TARGET_FILL_RANGE[0]):
        return None
    if _count_components(mask, connectivity=4) != 1:
        return None

    span_r, span_c = _bbox(mask)
    if span_r < int(np.ceil(n * MIN_BBOX_SPAN_FRAC)) or span_c < int(np.ceil(n * MIN_BBOX_SPAN_FRAC)):
        return None

    if np.array_equal(mask, np.rot90(mask)):
        return mask
    return None


def _boundary_candidates(blue_mask: np.ndarray) -> list[tuple[int, int]]:
    n = blue_mask.shape[0]
    candidates = []
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            has_blue = False
            has_black = False
            for dr, dc in NEIGHBORS8 + [(0, 0)]:
                rr, cc = r + dr, c + dc
                if blue_mask[rr, cc]:
                    has_blue = True
                else:
                    has_black = True
            if has_blue and has_black:
                candidates.append((r, c))
    return candidates


def _touches_existing_blob(cell: tuple[int, int], red_mask: np.ndarray) -> bool:
    n = red_mask.shape[0]
    r, c = cell
    for dr, dc in NEIGHBORS8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n and red_mask[rr, cc]:
            return True
    return False


def _grow_blob(
    rng: random.Random,
    blue_mask: np.ndarray,
    red_mask: np.ndarray,
    orbit_caps: dict[tuple[tuple[int, int], ...], int],
    start: tuple[int, int],
    target_size: int,
    radius_cap: int,
) -> set[tuple[int, int]] | None:
    n = blue_mask.shape[0]
    blob = {start}
    if blue_mask[start]:
        orb = _orbit(start, n)
        if orbit_caps[orb] <= 0:
            return None
        orbit_caps[orb] -= 1

    while len(blob) < target_size:
        frontier = set()
        for cell in blob:
            for neigh in _neighbors4(cell, n):
                if neigh in blob or red_mask[neigh]:
                    continue
                if _touches_existing_blob(neigh, red_mask):
                    continue
                if abs(neigh[0] - start[0]) + abs(neigh[1] - start[1]) > radius_cap:
                    continue
                frontier.add(neigh)
        if not frontier:
            break

        frontier_list = list(frontier)
        weights = []
        for cell in frontier_list:
            boundary_bonus = 3.0 if blue_mask[cell] != blue_mask[start] else 1.0
            weights.append(boundary_bonus * rng.uniform(0.4, 1.6))
        cell = rng.choices(frontier_list, weights=weights, k=1)[0]
        if blue_mask[cell]:
            orb = _orbit(cell, n)
            if orbit_caps[orb] <= 0:
                continue
            orbit_caps[orb] -= 1
        blob.add(cell)

    if len(blob) < 2:
        return None
    return blob


def _make_occluders(rng: random.Random, blue_mask: np.ndarray) -> np.ndarray | None:
    n = blue_mask.shape[0]
    candidates = _boundary_candidates(blue_mask)
    if len(candidates) < MIN_OCCLUDERS:
        return None

    orbit_caps: dict[tuple[tuple[int, int], ...], int] = {}
    for orb in _all_orbits(n):
        if blue_mask[orb[0]]:
            orbit_caps[orb] = len(orb) - 1

    red_mask = np.zeros_like(blue_mask, dtype=bool)
    occluder_count = rng.randint(MIN_OCCLUDERS, MAX_OCCLUDERS)
    blobs = []

    shuffled = candidates[:]
    rng.shuffle(shuffled)
    attempts = 0
    while len(blobs) < occluder_count and attempts < len(candidates) * 10:
        attempts += 1
        if not shuffled:
            shuffled = candidates[:]
            rng.shuffle(shuffled)
        start = shuffled.pop()
        if red_mask[start] or _touches_existing_blob(start, red_mask):
            continue

        local_caps = orbit_caps.copy()
        blob = _grow_blob(
            rng,
            blue_mask,
            red_mask,
            local_caps,
            start,
            target_size=rng.randint(*OCCLUDER_SIZE_RANGE),
            radius_cap=rng.randint(*OCCLUDER_RADIUS_RANGE),
        )
        if blob is None:
            continue

        blob_blue = any(blue_mask[cell] for cell in blob)
        blob_black = any(not blue_mask[cell] for cell in blob)
        if not (blob_blue and blob_black):
            continue

        trial = red_mask.copy()
        for cell in blob:
            trial[cell] = True
        if _count_components(trial, connectivity=8) != len(blobs) + 1:
            continue

        red_mask = trial
        orbit_caps = local_caps
        blobs.append(blob)

    if len(blobs) < MIN_OCCLUDERS:
        return None

    if not np.any(red_mask & blue_mask):
        return None
    if not np.any(red_mask & ~blue_mask):
        return None

    visible_blue = blue_mask & ~red_mask
    for orb, cap in orbit_caps.items():
        if blue_mask[orb[0]] and not any(visible_blue[r, c] for r, c in orb):
            return None

    hidden_blue = blue_mask & red_mask
    if not np.any(hidden_blue):
        return None

    return red_mask


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n = rng.randint(*GRID_SIDE_RANGE)

    blue_mask = _make_blue_pattern(rng, n)
    if blue_mask is None:
        return None

    red_mask = _make_occluders(rng, blue_mask)
    if red_mask is None:
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[blue_mask] = BLUE
    grid[red_mask] = RED

    if np.array_equal(grid, main(grid)):
        return None

    expected = np.zeros_like(grid)
    expected[blue_mask] = BLUE
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotational-occlusion puzzle")
