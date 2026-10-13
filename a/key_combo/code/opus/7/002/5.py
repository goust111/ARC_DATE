from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PATH_COLOR = 7
SEED_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
SEGMENT_DIRS = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}
RAY_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 14
GRID_MAX = 19
PATH_MIN_LEN = 5
PATH_MAX_LEN = 12
GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 250
SEED_COMBO_ATTEMPTS = 250


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _dilate8(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(iterations):
        nxt = out.copy()
        for r, c in np.argwhere(out):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = int(r) + dr, int(c) + dc
                    if _in_bounds(out.shape, nr, nc):
                        nxt[nr, nc] = True
        out = nxt
    return out


def _path_degrees(cells: set[tuple[int, int]]) -> list[int]:
    return [sum((r + dr, c + dc) in cells for dr, dc in DIRS4) for r, c in cells]


def _is_simple_path(coords: np.ndarray) -> bool:
    cells = {tuple(map(int, rc)) for rc in coords}
    if len(cells) != len(coords):
        return False
    degs = _path_degrees(cells)
    if degs.count(1) != 2 or any(d not in (1, 2) for d in degs):
        return False
    endpoints = [cell for cell in cells if sum((cell[0] + dr, cell[1] + dc) in cells for dr, dc in DIRS4) == 1]
    ordered = [endpoints[0]]
    unused = cells - {ordered[0]}
    while unused:
        cur = ordered[-1]
        nbrs = [cell for cell in unused if abs(cell[0] - cur[0]) + abs(cell[1] - cur[1]) == 1]
        if len(nbrs) != 1:
            return False
        ordered.append(nbrs[0])
        unused.remove(nbrs[0])
    return len(ordered) == len(coords)


def _normalize_coords(cells: list[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(cells, dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _generate_path_shape() -> np.ndarray:
    for _ in range(PATH_ATTEMPTS):
        length = random.randint(PATH_MIN_LEN, PATH_MAX_LEN)
        segments = random.randint(2, min(4, length - 1))
        seg_lengths = _random_composition(length - 1, segments)
        cur = (0, 0)
        cells = [cur]
        direction = random.choice(DIRS4)
        used = {cur}
        turns = 0
        ok = True

        for i, seg_len in enumerate(seg_lengths):
            if i > 0:
                direction = random.choice(SEGMENT_DIRS[direction])
                turns += 1
            for _ in range(seg_len):
                cur = (cur[0] + direction[0], cur[1] + direction[1])
                if cur in used:
                    ok = False
                    break
                cells.append(cur)
                used.add(cur)
            if not ok:
                break

        if not ok or turns == 0:
            continue

        coords = _normalize_coords(cells)
        if coords[:, 0].ptp() == 0 or coords[:, 1].ptp() == 0:
            continue
        if _is_simple_path(coords):
            return coords
    raise RuntimeError("failed to generate a simple path")


def _coords_to_mask(coords: np.ndarray, shape: tuple[int, int], offset: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    rr = coords[:, 0] + offset[0]
    cc = coords[:, 1] + offset[1]
    mask[rr, cc] = True
    return mask


def _place_paths(shape: tuple[int, int]) -> list[np.ndarray] | None:
    h, w = shape
    occupied = np.zeros(shape, dtype=bool)
    forbidden = np.zeros(shape, dtype=bool)
    placed: list[np.ndarray] = []

    for _ in range(2):
        local = _generate_path_shape()
        ph = int(local[:, 0].max()) + 1
        pw = int(local[:, 1].max()) + 1
        if ph > h - 2 or pw > w - 2:
            return None

        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                rr = local[:, 0] + r0
                cc = local[:, 1] + c0
                if forbidden[rr, cc].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        mask = _coords_to_mask(local, shape, (r0, c0))
        if (mask & occupied).any():
            return None
        occupied |= mask
        forbidden |= _dilate8(mask, iterations=2)
        placed.append(mask)

    return placed


def _trace_ray(
    start: tuple[int, int],
    direction: tuple[int, int],
    blocker_mask: np.ndarray,
    occupied: np.ndarray,
) -> tuple[list[tuple[int, int]], str]:
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(blocker_mask.shape, r, c):
            return cells, "edge"
        if blocker_mask[r, c]:
            return cells, "halo"
        if occupied[r, c]:
            return cells, "occupied"
        cells.append((r, c))


def _solve_from_parts(seed_points: list[tuple[int, int, int]], blocker_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(blocker_mask.shape, dtype=int)
    for r, c, color in seed_points:
        out[r, c] = color
    for r, c, color in sorted(seed_points):
        for direction in RAY_DIRS:
            cells, _ = _trace_ray((r, c), direction, blocker_mask, out != 0)
            for rr, cc in cells:
                out[rr, cc] = color
    return out


def _extract_paths_and_seeds(grid: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    path_mask = grid == PATH_COLOR
    path_components = _components(path_mask, connectivity=4)
    seed_points = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere((grid != BG) & (grid != PATH_COLOR))]
    return path_components, seed_points


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the orange paths and keep only seed-colored rays clipped by the paths' invisible 8-neighborhood halos.

    Input structure:
    - The grid has black background, exactly two orange 4-connected one-pixel-wide path components,
      and 2-4 singleton seed pixels in other colors.
    - The seed pixels lie on background cells outside the orange paths.
    - The paths do not branch; each is a simple chain whose surrounding 8-neighborhood halo acts as
      a hidden geometric constraint.

    Rule:
    - Compute the union of the two paths' 8-neighborhood dilations by one cell; this halo is a blocker
      but is never drawn.
    - Start from an all-black canvas, copy the seed pixels, and extend same-colored rays from each seed
      upward, downward, leftward, and rightward.
    - Each ray stops before the first halo cell, before leaving the grid, or before hitting a pixel already
      occupied by another seed or earlier ray.
    - Do not draw the orange paths or the halo in the output.

    Key insight:
    - The ray lengths are not limited by the visible paths themselves, but by an invisible one-cell halo
      around each path, so the missing orange objects matter only through a hidden intermediate dilation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    path_components, seed_points = _extract_paths_and_seeds(g)
    assert len(path_components) == 2, "expected exactly two orange path components"
    assert 2 <= len(seed_points) <= 4, "expected 2-4 seed pixels"

    path_mask = g == PATH_COLOR
    halo = np.zeros_like(path_mask, dtype=bool)
    for comp in path_components:
        comp_mask = np.zeros_like(path_mask, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        halo |= _dilate8(comp_mask, iterations=1)

    return _solve_from_parts(seed_points, halo)


def _seed_candidate_info(
    pos: tuple[int, int],
    halo: np.ndarray,
    path_mask: np.ndarray,
) -> dict | None:
    if halo[pos] or path_mask[pos]:
        return None

    empty = np.zeros(halo.shape, dtype=bool)
    rays: dict[tuple[int, int], list[tuple[int, int]]] = {}
    stops: dict[tuple[int, int], str] = {}
    raw_rays: dict[tuple[int, int], list[tuple[int, int]]] = {}
    total = 0
    halo_blocked = 0
    differs_from_raw = False
    for direction in RAY_DIRS:
        cells, stop = _trace_ray(pos, direction, halo, empty)
        raw_cells, _ = _trace_ray(pos, direction, path_mask, empty)
        rays[direction] = cells
        raw_rays[direction] = raw_cells
        stops[direction] = stop
        total += len(cells)
        halo_blocked += int(stop == "halo" and len(cells) >= 1)
        differs_from_raw |= cells != raw_cells

    if total < 4:
        return None
    if halo_blocked == 0:
        return None
    if not differs_from_raw:
        return None

    return {
        "pos": pos,
        "rays": rays,
        "stops": stops,
        "ray_union": {cell for cells in rays.values() for cell in cells},
    }


def _compatible_seed_choice(chosen: list[dict], cand: dict) -> bool:
    pos = cand["pos"]
    for prev in chosen:
        if abs(prev["pos"][0] - pos[0]) + abs(prev["pos"][1] - pos[1]) < 3:
            return False
        if pos in prev["ray_union"] or prev["pos"] in cand["ray_union"]:
            return False
        if prev["ray_union"] & cand["ray_union"]:
            return False
    return True


def _find_compatible_combo(candidates: list[dict], target_k: int) -> list[dict] | None:
    for _ in range(40):
        pool = candidates[:]
        random.shuffle(pool)
        chosen: list[dict] = []
        for cand in pool:
            if _compatible_seed_choice(chosen, cand):
                chosen.append(cand)
            if len(chosen) == target_k:
                return chosen
    return None


def _choose_seeds(path_masks: list[np.ndarray]) -> list[tuple[int, int, int]] | None:
    path_mask = np.zeros_like(path_masks[0], dtype=bool)
    halo = np.zeros_like(path_masks[0], dtype=bool)
    individual_halos: list[np.ndarray] = []
    for mask in path_masks:
        path_mask |= mask
        comp_halo = _dilate8(mask, iterations=1)
        individual_halos.append(comp_halo)
        halo |= comp_halo

    candidates: list[dict] = []
    h, w = path_mask.shape
    for r in range(h):
        for c in range(w):
            info = _seed_candidate_info((r, c), halo, path_mask)
            if info is not None:
                candidates.append(info)
    if len(candidates) < 2:
        return None

    for _ in range(SEED_COMBO_ATTEMPTS):
        target_order = [4, 3, 2]
        if random.random() < 0.5:
            random.shuffle(target_order)
        chosen = None
        for k in target_order:
            chosen = _find_compatible_combo(candidates, k)
            if chosen is not None:
                break
        if chosen is None:
            continue

        colors = random.sample(SEED_COLORS, len(chosen))
        seeds = [(cand["pos"][0], cand["pos"][1], color) for cand, color in zip(chosen, colors)]

        full_out = _solve_from_parts(seeds, halo)
        raw_out = _solve_from_parts(seeds, path_mask)
        if np.array_equal(full_out, raw_out):
            continue

        if np.array_equal(full_out, _solve_from_parts(seeds, individual_halos[0])):
            continue
        if np.array_equal(full_out, _solve_from_parts(seeds, individual_halos[1])):
            continue

        return seeds
    return None


def _validate_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(np.unique(grid)).issubset(set(range(10)))

    path_components, seed_points = _extract_paths_and_seeds(grid)
    assert len(path_components) == 2
    assert 2 <= len(seed_points) <= 4
    assert len({color for _, _, color in seed_points}) == len(seed_points)

    path_mask = grid == PATH_COLOR
    halos: list[np.ndarray] = []
    halo = np.zeros_like(path_mask, dtype=bool)

    for comp in path_components:
        assert PATH_MIN_LEN <= len(comp) <= PATH_MAX_LEN
        assert _is_simple_path(comp)
        comp_mask = np.zeros_like(path_mask, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        comp_halo = _dilate8(comp_mask, iterations=1)
        halos.append(comp_halo)
        halo |= comp_halo

    for r, c, _ in seed_points:
        assert grid[r, c] != PATH_COLOR
        assert not halo[r, c]

    seed_infos = [_seed_candidate_info((r, c), halo, path_mask) for r, c, _ in seed_points]
    assert all(info is not None for info in seed_infos)
    chosen_infos = [info for info in seed_infos if info is not None]
    for i, info_a in enumerate(chosen_infos):
        for info_b in chosen_infos[i + 1 :]:
            assert _compatible_seed_choice([info_a], info_b)

    out = main(grid)
    expected = _solve_from_parts(seed_points, halo)
    assert np.array_equal(out, expected)
    assert out.shape == grid.shape
    assert not np.any(out == PATH_COLOR)
    for r, c, color in seed_points:
        assert out[r, c] == color
    assert np.all(out[path_mask] == BG)

    raw_out = _solve_from_parts(seed_points, path_mask)
    assert not np.array_equal(out, raw_out)
    assert not np.array_equal(out, _solve_from_parts(seed_points, halos[0]))
    assert not np.array_equal(out, _solve_from_parts(seed_points, halos[1]))


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    path_masks = _place_paths(shape)
    if path_masks is None:
        return None

    seeds = _choose_seeds(path_masks)
    if seeds is None:
        return None

    grid = np.zeros(shape, dtype=int)
    for mask in path_masks:
        grid[mask] = PATH_COLOR
    for r, c, color in seeds:
        if grid[r, c] != BG:
            return None
        grid[r, c] = color

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for hidden halo ray puzzle")
