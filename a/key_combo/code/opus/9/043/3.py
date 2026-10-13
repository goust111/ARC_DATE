from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
PINK = 6
TEAL = 8
TRAIL_LEN = 4

DIRS4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_GEN_BRANCH_TOGGLE = False
_GEN_RETRIES = 400


def _neighbors8(r: int, c: int, size: int):
    for dr, dc in DIRS8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc, h):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _diag_symmetric(mask: np.ndarray) -> bool:
    return mask.shape[0] == mask.shape[1] and np.array_equal(mask, mask.T)


def _tile_bounds(grid: np.ndarray, tile_r: int, tile_c: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    tile_h = h // 2
    tile_w = w // 2
    r0 = tile_r * tile_h
    c0 = tile_c * tile_w
    return r0, c0, r0 + tile_h, c0 + tile_w


def _tile_index_for_coord(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = grid.shape
    return int(r >= h // 2), int(c >= w // 2)


def _find_trail_direction(grid: np.ndarray, red: tuple[int, int]) -> tuple[int, int]:
    pink = {tuple(map(int, rc)) for rc in np.argwhere(grid == PINK)}
    assert len(pink) == TRAIL_LEN, "expected exactly four pink trail cells"
    rr, cc = red
    matches = []
    for dr, dc in DIRS4:
        want = {(rr + step * dr, cc + step * dc) for step in range(1, TRAIL_LEN + 1)}
        if want == pink:
            matches.append((dr, dc))
    assert len(matches) == 1, "pink cells must form one straight trail from the red marker"
    return matches[0]


def _locate_target_tile(grid: np.ndarray, red: tuple[int, int], reverse_dir: tuple[int, int]) -> tuple[int, int]:
    h, w = grid.shape
    start_tile = _tile_index_for_coord(grid, *red)
    dr, dc = reverse_dir
    r, c = red
    while True:
        r += dr
        c += dc
        assert 0 <= r < h and 0 <= c < w, "reverse ray must enter another tile before leaving the grid"
        tile = _tile_index_for_coord(grid, r, c)
        if tile != start_tile:
            return tile


def main(grid: np.ndarray) -> np.ndarray:
    """Locate the tile hit by the reverse pink trajectory, then erase clusters or singletons according to that tile's diagonal symmetry.

    Input structure:
    - The grid is a black canvas split conceptually into a 2x2 arrangement of equal square tiles.
    - Every tile contains a teal pattern made of isolated pixels and 8-connected clusters.
    - Exactly one non-target tile also contains one red marker and a four-cell pink straight trail in a cardinal direction.

    Rule:
    - Read the pink trail direction relative to the red marker, then cast a ray from the red marker in the opposite direction.
    - The first tile entered after the ray crosses a tile boundary is the target tile.
    - Check whether the teal pattern inside the target tile is invariant under main-diagonal transpose in tile-local coordinates.
    - If the target is asymmetric, erase every size-1 teal component and keep every larger 8-connected component.
    - If the target is symmetric, erase every teal component of size greater than 1 and keep only the size-1 teal components.

    Key insight:
    - The pink trajectory does not point to a pixel to edit; it only identifies which tile to inspect, and that tile's own symmetry decides which kind of teal noise gets erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0 and h // 2 == w // 2, "expected four equal square tiles"

    red_coords = np.argwhere(g == RED)
    assert len(red_coords) == 1, "expected exactly one red marker"
    red = tuple(map(int, red_coords[0]))

    trail_dir = _find_trail_direction(g, red)
    reverse_dir = (-trail_dir[0], -trail_dir[1])
    target_tile = _locate_target_tile(g, red, reverse_dir)

    r0, c0, r1, c1 = _tile_bounds(g, *target_tile)
    tile = g[r0:r1, c0:c1]
    teal_mask = tile == TEAL
    comps = _component_coords(teal_mask)
    assert comps, "target tile must contain teal pixels"

    keep_singletons = _diag_symmetric(teal_mask)

    out = g.copy()
    for comp in comps:
        size = int(len(comp))
        erase = (size > 1) if keep_singletons else (size == 1)
        if erase:
            rr = comp[:, 0] + r0
            cc = comp[:, 1] + c0
            out[rr, cc] = BLACK

    return out


def _halo(cells: set[tuple[int, int]], size: int) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors8(r, c, size):
            expanded.add((nr, nc))
    return expanded


def _choose_red_and_reservations(size: int, trail_dir: tuple[int, int]) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    dr, dc = trail_dir
    if dr == 0:
        red_r = random.randrange(size)
        if dc > 0:
            red_c = random.randrange(1, size - TRAIL_LEN)
        else:
            red_c = random.randrange(TRAIL_LEN, size - 1)
    else:
        red_c = random.randrange(size)
        if dr > 0:
            red_r = random.randrange(1, size - TRAIL_LEN)
        else:
            red_r = random.randrange(TRAIL_LEN, size - 1)

    reserved = {(red_r, red_c)}
    for step in range(1, TRAIL_LEN + 1):
        reserved.add((red_r + step * dr, red_c + step * dc))

    ray_dr, ray_dc = -dr, -dc
    rr, cc = red_r + ray_dr, red_c + ray_dc
    while 0 <= rr < size and 0 <= cc < size:
        reserved.add((rr, cc))
        rr += ray_dr
        cc += ray_dc

    return (red_r, red_c), reserved


def _random_cluster(size: int, cluster_size: int, forbidden: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    cells = [(r, c) for r in range(size) for c in range(size) if (r, c) not in forbidden]
    random.shuffle(cells)
    for start in cells:
        cluster = {start}
        for _ in range(cluster_size - 1):
            frontier = []
            for r, c in cluster:
                for nr, nc in _neighbors8(r, c, size):
                    if (nr, nc) not in forbidden and (nr, nc) not in cluster:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cluster.add(random.choice(frontier))
        if len(cluster) == cluster_size:
            return cluster
    return None


def _paint_components(size: int, components: list[set[tuple[int, int]]]) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for comp in components:
        for r, c in comp:
            mask[r, c] = True
    return mask


def _component_size_signature(mask: np.ndarray) -> tuple[int, int]:
    comps = _component_coords(mask)
    singletons = sum(len(comp) == 1 for comp in comps)
    big_clusters = sum(len(comp) >= 3 for comp in comps)
    return singletons, big_clusters


def _gen_symmetric_tile(size: int, reserved: set[tuple[int, int]], require_target_mix: bool) -> np.ndarray | None:
    motifs = [
        {(0, 0), (0, 1), (1, 0)},
        {(0, 0), (0, 1), (1, 0), (1, 1)},
    ]
    for _ in range(200):
        components: list[set[tuple[int, int]]] = []
        blocked = set(reserved)

        motif = random.choice(motifs)
        anchors = []
        for base in range(size - 1):
            placed = {(base + r, base + c) for r, c in motif}
            if all(0 <= r < size and 0 <= c < size and (r, c) not in blocked for r, c in placed):
                anchors.append(placed)
        if not anchors:
            continue
        cluster = random.choice(anchors)
        components.append(cluster)
        blocked |= _halo(cluster, size)

        pair_options = []
        for r in range(size):
            for c in range(size):
                if r >= c - 1:
                    continue
                pair = {(r, c), (c, r)}
                if any(cell in blocked for cell in pair):
                    continue
                if max(abs(r - c), abs(c - r)) <= 1:
                    continue
                pair_options.append(pair)
        if not pair_options:
            continue
        singletons = random.choice(pair_options)
        components.extend([{cell} for cell in sorted(singletons)])
        blocked |= _halo(singletons, size)

        if random.random() < 0.45:
            extra_diag = [d for d in range(size) if (d, d) not in blocked]
            if extra_diag:
                d = random.choice(extra_diag)
                components.append({(d, d)})
                blocked |= _halo({(d, d)}, size)

        mask = _paint_components(size, components)
        if not _diag_symmetric(mask):
            continue
        singletons_count, big_clusters = _component_size_signature(mask)
        if require_target_mix and not (singletons_count >= 2 and big_clusters >= 1):
            continue
        if not require_target_mix and not (singletons_count >= 1 and big_clusters >= 1):
            continue
        return mask.astype(int)
    return None


def _gen_asymmetric_tile(size: int, reserved: set[tuple[int, int]], require_target_mix: bool) -> np.ndarray | None:
    for _ in range(250):
        components: list[set[tuple[int, int]]] = []
        blocked = set(reserved)

        cluster_size = random.randint(3, 5) if require_target_mix else random.randint(2, 5)
        first_cluster = _random_cluster(size, cluster_size, blocked)
        if first_cluster is None:
            continue
        components.append(first_cluster)
        blocked |= _halo(first_cluster, size)

        if random.random() < 0.35:
            extra_cluster = _random_cluster(size, random.randint(2, 4), blocked)
            if extra_cluster is not None:
                components.append(extra_cluster)
                blocked |= _halo(extra_cluster, size)

        needed_singletons = 2 if require_target_mix else random.randint(1, 2)
        free = [(r, c) for r in range(size) for c in range(size) if (r, c) not in blocked]
        random.shuffle(free)
        if len(free) < needed_singletons:
            continue
        for idx in range(needed_singletons):
            cell = free[idx]
            components.append({cell})
            blocked |= _halo({cell}, size)

        mask = _paint_components(size, components)
        if _diag_symmetric(mask):
            continue
        singletons_count, big_clusters = _component_size_signature(mask)
        if require_target_mix and not (singletons_count >= 2 and big_clusters >= 1):
            continue
        if not require_target_mix and not (singletons_count >= 1 and big_clusters >= 1):
            continue
        return mask.astype(int)
    return None


def _build_tile_mask(size: int, symmetric: bool, reserved: set[tuple[int, int]], is_target: bool) -> np.ndarray | None:
    if symmetric:
        return _gen_symmetric_tile(size, reserved, require_target_mix=is_target)
    return _gen_asymmetric_tile(size, reserved, require_target_mix=is_target)


def _reverse_tile_neighbors(tile_r: int, tile_c: int) -> list[tuple[int, int]]:
    dirs = []
    if tile_r > 0:
        dirs.append((-1, 0))
    if tile_r < 1:
        dirs.append((1, 0))
    if tile_c > 0:
        dirs.append((0, -1))
    if tile_c < 1:
        dirs.append((0, 1))
    return dirs


def _build_symmetry_layout(target_tile: tuple[int, int], target_symmetric: bool) -> dict[tuple[int, int], bool]:
    tiles = [(0, 0), (0, 1), (1, 0), (1, 1)]
    layout = {tile: False for tile in tiles}
    if target_symmetric:
        layout[target_tile] = True
        other = random.choice([tile for tile in tiles if tile != target_tile])
        layout[other] = True
    else:
        others = [tile for tile in tiles if tile != target_tile]
        for tile in random.sample(others, 2):
            layout[tile] = True
    return layout


def _validate_instance(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    h, w = inp.shape
    size = h // 2
    red = tuple(map(int, np.argwhere(inp == RED)[0]))
    trail_dir = _find_trail_direction(inp, red)
    target_tile = _locate_target_tile(inp, red, (-trail_dir[0], -trail_dir[1]))
    marker_tile = _tile_index_for_coord(inp, *red)
    if target_tile == marker_tile:
        return False

    sym_count = 0
    for tile_r in range(2):
        for tile_c in range(2):
            r0, c0, r1, c1 = _tile_bounds(inp, tile_r, tile_c)
            if _diag_symmetric(inp[r0:r1, c0:c1] == TEAL):
                sym_count += 1
    if sym_count != 2:
        return False

    r0, c0, r1, c1 = _tile_bounds(inp, *target_tile)
    target_mask = inp[r0:r1, c0:c1] == TEAL
    comps = _component_coords(target_mask)
    if sum(len(comp) == 1 for comp in comps) < 2:
        return False
    if sum(len(comp) >= 3 for comp in comps) < 1:
        return False

    changed = inp != out
    if not changed.any():
        return False
    target_change = np.zeros_like(changed)
    target_change[r0:r1, c0:c1] = True
    if np.any(changed & ~target_change):
        return False

    keep_singletons = _diag_symmetric(target_mask)
    wrong_out = inp.copy()
    for comp in comps:
        size_comp = len(comp)
        wrong_erase = (size_comp == 1) if keep_singletons else (size_comp > 1)
        if wrong_erase:
            rr = comp[:, 0] + r0
            cc = comp[:, 1] + c0
            wrong_out[rr, cc] = BLACK
    if np.array_equal(wrong_out, out):
        return False

    return True


def _gen_attempt(target_symmetric: bool) -> np.ndarray | None:
    size = random.randint(6, 8)
    grid = np.zeros((2 * size, 2 * size), dtype=int)

    marker_tile = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    reverse_dir = random.choice(_reverse_tile_neighbors(*marker_tile))
    target_tile = (marker_tile[0] + reverse_dir[0], marker_tile[1] + reverse_dir[1])
    trail_dir = (-reverse_dir[0], -reverse_dir[1])

    sym_layout = _build_symmetry_layout(target_tile, target_symmetric)

    red_local, reserved = _choose_red_and_reservations(size, trail_dir)
    reserved = _halo(reserved, size)

    for tile_r in range(2):
        for tile_c in range(2):
            tile = (tile_r, tile_c)
            tile_reserved = reserved if tile == marker_tile else set()
            mask = _build_tile_mask(size, sym_layout[tile], tile_reserved, is_target=(tile == target_tile))
            if mask is None:
                return None
            r0, c0, r1, c1 = _tile_bounds(grid, tile_r, tile_c)
            grid[r0:r1, c0:c1][mask.astype(bool)] = TEAL

    marker_r0, marker_c0, _marker_r1, _marker_c1 = _tile_bounds(grid, *marker_tile)
    rr, cc = red_local
    grid[marker_r0 + rr, marker_c0 + cc] = RED
    for step in range(1, TRAIL_LEN + 1):
        pr = marker_r0 + rr + step * trail_dir[0]
        pc = marker_c0 + cc + step * trail_dir[1]
        grid[pr, pc] = PINK

    if _validate_instance(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    global _GEN_BRANCH_TOGGLE
    _GEN_BRANCH_TOGGLE = not _GEN_BRANCH_TOGGLE
    branch_order = [_GEN_BRANCH_TOGGLE, not _GEN_BRANCH_TOGGLE]
    for preferred in branch_order:
        for _ in range(_GEN_RETRIES):
            grid = _gen_attempt(preferred)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 3")
