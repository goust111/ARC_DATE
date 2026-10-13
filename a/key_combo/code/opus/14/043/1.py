import hashlib
from collections import deque

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
FLOOR = 5

GRID_H_RANGE = (13, 19)
GRID_W_RANGE = (13, 19)
SPRITE_SIZE_RANGE = (3, 6)
OBJECT_COUNT_RANGE = (4, 5)
MAX_GENERATION_ATTEMPTS = 800

_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _coords_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _stamp(shape: tuple[int, int], objects: list[np.ndarray]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for coords in objects:
        grid[coords[:, 0], coords[:, 1]] = BLUE
    grid[-1, :] = FLOOR
    return grid


def _settle_objects(objects: list[np.ndarray], shape: tuple[int, int], floor_mask: np.ndarray) -> list[np.ndarray]:
    settled = [coords.copy() for coords in objects]
    stopped = [False] * len(settled)
    h, w = shape

    while not all(stopped):
        stopped_mask = np.zeros((h, w), dtype=bool)
        for idx, coords in enumerate(settled):
            if stopped[idx]:
                stopped_mask[coords[:, 0], coords[:, 1]] = True

        active = [i for i, s in enumerate(stopped) if not s]
        active.sort(key=lambda i: (-int(settled[i][:, 0].max()), -int(settled[i][:, 0].min()), int(settled[i][:, 1].min())))

        for idx in active:
            shifted = settled[idx] + np.array([1, 0], dtype=int)
            blocked = (
                np.any(shifted[:, 0] >= h)
                or np.any(shifted[:, 1] < 0)
                or np.any(shifted[:, 1] >= w)
                or np.any(floor_mask[shifted[:, 0], shifted[:, 1]])
                or np.any(stopped_mask[shifted[:, 0], shifted[:, 1]])
            )
            if blocked:
                stopped[idx] = True
                stopped_mask[settled[idx][:, 0], settled[idx][:, 1]] = True
            else:
                settled[idx] = shifted

    return settled


def _closed_blue_labels(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], set[int]]:
    blue_labels = np.zeros(grid.shape, dtype=int)
    blue_parts = _components(grid == BLUE)
    for idx, coords in enumerate(blue_parts, start=1):
        blue_labels[coords[:, 0], coords[:, 1]] = idx

    closed: set[int] = set()
    for hole in _components(grid == BG):
        touches_edge = (
            np.any(hole[:, 0] == 0)
            or np.any(hole[:, 0] == grid.shape[0] - 1)
            or np.any(hole[:, 1] == 0)
            or np.any(hole[:, 1] == grid.shape[1] - 1)
        )
        if touches_edge:
            continue
        adjacent = set()
        for r, c in hole:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    label = int(blue_labels[nr, nc])
                    if label > 0:
                        adjacent.add(label)
        closed.update(adjacent)
    return blue_labels, blue_parts, closed


def _recolor_closed_components(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    labels, _, closed = _closed_blue_labels(grid)
    for label in closed:
        out[labels == label] = GREEN
    return out


def _pixelwise_gravity(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    floor_mask = grid == FLOOR
    objects = [coords.copy() for coords in _components(grid == BLUE)]
    pixel_objects = [coords[[i]] for coords in objects for i in range(coords.shape[0])]
    settled = _settle_objects(pixel_objects, (h, w), floor_mask)
    return _stamp((h, w), settled)


def _object_gravity(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    floor_mask = grid == FLOOR
    objects = _components(grid == BLUE)
    settled = _settle_objects(objects, (h, w), floor_mask)
    return _stamp((h, w), settled)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the blue sprites onto the grey floor, then recolor only the settled groups that trap black interior cells.

    Input structure:
    - Black background contains a single full-width grey floor row at the bottom and 3-5 disconnected blue sprites above it.
    - Each blue sprite is a small 4-connected rigid object with 3-6 cells.
    - Different sprites may share columns, but they start separated in space and never overlap the floor.

    Rule:
    - Treat each input blue connected component as one rigid sprite.
    - Repeatedly process the still-falling sprites from lowest to highest: move a sprite down by one row iff its translated cells would avoid the grey floor, the grid bottom, and any blue pixels that have already stopped; otherwise mark that sprite stopped.
    - After all sprites stop, recompute the 4-connected blue components in the settled state.
    - Any settled blue component adjacent to at least one black region that cannot reach the grid boundary through 4-connected black cells is enclosed, so recolor that whole component green; keep all other blue pixels blue.
    - Leave the grey floor and all black background cells unchanged.

    Key insight:
    - Hollow-ness is not an intrinsic property of any input sprite: gravity changes the topology first, and only the post-settling merged components can create enclosed black holes.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    settled = _object_gravity(g)
    return _recolor_closed_components(settled)


def _random_polyomino(rng: np.random.Generator, area: int, max_h: int = 4, max_w: int = 5) -> np.ndarray | None:
    for _ in range(200):
        cells = {(0, 0)}
        frontier = set(_neighbors4(0, 0))
        while len(cells) < area:
            options = list(frontier)
            rr, cc = options[int(rng.integers(len(options)))]
            frontier.discard((rr, cc))
            cells.add((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        coords = np.array(norm, dtype=int)
        height = int(coords[:, 0].max()) + 1
        width = int(coords[:, 1].max()) + 1
        if height <= max_h and width <= max_w:
            return coords
    return None


def _touches_existing(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if occupied[rr, cc]:
            return True
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return True
    return False


def _place_open_objects(
    rng: np.random.Generator,
    h: int,
    w: int,
    settled_objects: list[np.ndarray],
    open_count: int,
) -> list[np.ndarray] | None:
    occupied = np.zeros((h, w), dtype=bool)
    for coords in settled_objects:
        occupied[coords[:, 0], coords[:, 1]] = True

    result = [coords.copy() for coords in settled_objects]
    for _obj_idx in range(open_count):
        placed = False
        for _ in range(250):
            area = int(rng.integers(SPRITE_SIZE_RANGE[0], SPRITE_SIZE_RANGE[1] + 1))
            rel = _random_polyomino(rng, area)
            if rel is None:
                continue
            height = int(rel[:, 0].max()) + 1
            width = int(rel[:, 1].max()) + 1
            top = h - 1 - height
            if top < 1 or width >= w:
                continue
            left = int(rng.integers(0, w - width + 1))
            coords = rel + np.array([top, left], dtype=int)
            if _touches_existing(coords, occupied):
                continue
            occupied[coords[:, 0], coords[:, 1]] = True
            result.append(coords)
            placed = True
            break
        if not placed:
            return None
    return result


def _choose_cluster(rng: np.random.Generator, h: int, w: int) -> tuple[list[np.ndarray], list[int]] | None:
    mode = str(rng.choice(["single_cap", "split_cap"]))
    max_pillar_len = min(6, h - 6)
    if max_pillar_len < 3:
        return None
    pillar_len = int(rng.integers(3, max_pillar_len + 1))
    top_row = h - 2 - pillar_len
    if top_row < 2:
        return None

    if mode == "single_cap":
        cap_len = int(rng.integers(3, min(6, w - 4) + 1))
        open_count = int(rng.integers(1, 3))
    else:
        cap_len = 6
        if w < 10:
            return None
        open_count = 1

    max_left = w - cap_len - 1
    if max_left < 1:
        return None
    left = int(rng.integers(1, max_left + 1))
    right = left + cap_len - 1

    left_pillar = np.array([(top_row + 1 + dr, left) for dr in range(pillar_len)], dtype=int)
    right_pillar = np.array([(top_row + 1 + dr, right) for dr in range(pillar_len)], dtype=int)
    cluster = [left_pillar, right_pillar]

    if mode == "single_cap":
        cap = np.array([(top_row, left + dc) for dc in range(cap_len)], dtype=int)
        cluster.append(cap)
    else:
        cap_left = np.array([(top_row, left + dc) for dc in range(3)], dtype=int)
        cap_right = np.array([(top_row, left + 3 + dc) for dc in range(3)], dtype=int)
        cluster.extend([cap_left, cap_right])

    all_objects = _place_open_objects(rng, h, w, cluster, open_count)
    if all_objects is None:
        return None
    return all_objects, list(range(len(cluster)))


def _sample_input_positions(
    rng: np.random.Generator,
    settled_objects: list[np.ndarray],
    cluster_indices: list[int],
    h: int,
) -> list[np.ndarray] | None:
    shifts = [0] * len(settled_objects)

    pillar_shift = int(rng.integers(2, 5))
    for idx in cluster_indices[:2]:
        max_shift = int(settled_objects[idx][:, 0].min())
        if max_shift < pillar_shift:
            return None
        shifts[idx] = pillar_shift

    if len(cluster_indices) == 3:
        cap_idx = cluster_indices[2]
        max_shift = int(settled_objects[cap_idx][:, 0].min())
        low = pillar_shift + 2
        if max_shift < low:
            return None
        shifts[cap_idx] = int(rng.integers(low, max_shift + 1))
    else:
        cap_a, cap_b = cluster_indices[2], cluster_indices[3]
        max_a = int(settled_objects[cap_a][:, 0].min())
        max_b = int(settled_objects[cap_b][:, 0].min())
        low = pillar_shift + 2
        if max_a < low or max_b < low:
            return None
        choices_a = list(range(low, max_a + 1))
        choices_b = list(range(low, max_b + 1))
        valid = [(a, b) for a in choices_a for b in choices_b if a != b]
        if not valid:
            return None
        a_shift, b_shift = valid[int(rng.integers(len(valid)))]
        shifts[cap_a] = int(a_shift)
        shifts[cap_b] = int(b_shift)

    for idx, coords in enumerate(settled_objects):
        if idx in cluster_indices:
            continue
        max_shift = int(coords[:, 0].min())
        low = 2
        if max_shift < low:
            return None
        shifts[idx] = int(rng.integers(low, max_shift + 1))

    lifted = [coords - np.array([shifts[i], 0], dtype=int) for i, coords in enumerate(settled_objects)]
    return lifted


def _validate_semantics(input_grid: np.ndarray, settled_objects: list[np.ndarray], cluster_indices: list[int]) -> bool:
    settled_plain = _stamp(input_grid.shape, settled_objects)
    settled_colored = _recolor_closed_components(settled_plain)
    solved = main(input_grid)
    if not np.array_equal(solved, settled_colored):
        return False

    input_parts = _components(input_grid == BLUE)
    if len(input_parts) != len(settled_objects):
        return False
    if any(not (SPRITE_SIZE_RANGE[0] <= len(part) <= SPRITE_SIZE_RANGE[1]) for part in input_parts):
        return False

    if np.any(input_grid[-1, :] != FLOOR):
        return False

    input_closed = _closed_blue_labels(input_grid)[2]
    if input_closed:
        return False

    labels, parts, closed = _closed_blue_labels(settled_plain)
    green_count = len(closed)
    if green_count < 1 or green_count >= len(parts):
        return False
    if GREEN not in solved or BLUE not in solved:
        return False

    if np.array_equal(_pixelwise_gravity(input_grid), settled_plain):
        return False

    for idx in cluster_indices:
        reduced = [coords.copy() for j, coords in enumerate(settled_objects) if j != idx]
        reduced_grid = _stamp(input_grid.shape, reduced)
        if _closed_blue_labels(reduced_grid)[2]:
            return False

    initial_sig = {_coords_signature(part) for part in input_parts}
    target_sig = {_coords_signature(coords) for coords in settled_objects}
    if initial_sig == target_sig:
        return False

    return True


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))

    cluster_data = _choose_cluster(rng, h, w)
    if cluster_data is None:
        return None
    settled_objects, cluster_indices = cluster_data
    if not (OBJECT_COUNT_RANGE[0] <= len(settled_objects) <= OBJECT_COUNT_RANGE[1]):
        return None

    lifted = _sample_input_positions(rng, settled_objects, cluster_indices, h)
    if lifted is None:
        return None

    input_grid = _stamp((h, w), lifted)
    if len(_components(input_grid == BLUE)) != len(settled_objects):
        return None

    settled_plain = _stamp((h, w), settled_objects)
    if not _validate_semantics(input_grid, settled_objects, cluster_indices):
        return None

    out_sig = hashlib.md5(settled_plain.tobytes() + str(settled_plain.shape).encode()).hexdigest()
    if out_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(out_sig)
    return input_grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-hole input")
