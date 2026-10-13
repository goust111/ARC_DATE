from __future__ import annotations

from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SHAPE_LIBRARY: dict[int, list[np.ndarray]] = {
    1: [np.array([[1]], dtype=int)],
    2: [np.array([[1, 1]], dtype=int)],
    3: [
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
    ],
    4: [
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    ],
    5: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
        np.array([[1, 1], [1, 1], [1, 0]], dtype=int),
    ],
}

GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _find_seed_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            components.append({"color": color, "coords": coords, "area": len(coords)})
    return components


def _grow_by_area(
    grid: np.ndarray,
    *,
    reverse: bool = False,
    collect_stats: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    components = _find_seed_components(g)
    order = sorted(range(len(components)), key=lambda idx: int(components[idx]["area"]), reverse=reverse)

    out = g.copy()
    occupied = g != 0
    stats = {
        "painted": 0,
        "grown_components": 0,
        "border_stops": 0,
        "other_seed_stops": 0,
        "prior_ray_stops": 0,
        "internal_seed_stops": 0,
    }

    for idx in order:
        comp = components[idx]
        color = int(comp["color"])
        coords = list(comp["coords"])
        coord_set = set(coords)
        new_cells: set[tuple[int, int]] = set()

        for r, c in coords:
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                while 0 <= rr < h and 0 <= cc < w and not occupied[rr, cc]:
                    new_cells.add((int(rr), int(cc)))
                    rr += dr
                    cc += dc

                if not collect_stats:
                    continue
                if not (0 <= rr < h and 0 <= cc < w):
                    stats["border_stops"] += 1
                elif int(g[rr, cc]) != 0:
                    if (int(rr), int(cc)) in coord_set:
                        stats["internal_seed_stops"] += 1
                    else:
                        stats["other_seed_stops"] += 1
                else:
                    stats["prior_ray_stops"] += 1

        if new_cells:
            stats["grown_components"] += 1
        stats["painted"] += len(new_cells)
        for rr, cc in new_cells:
            out[rr, cc] = color
            occupied[rr, cc] = True

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed cluster into cross-shaped rays, with smaller clusters claiming space first.

    Input structure:
    - The background is black.
    - Several distinct colors each form one compact 4-connected seed cluster.
    - Seed clusters are separated by black space and have pairwise distinct areas.

    Rule:
    - Measure the area of every seed cluster and process the clusters from smallest area to largest.
    - When processing one cluster, launch rays from each of its pixels in the four cardinal directions.
    - A ray paints black cells in the cluster color until the next step would leave the grid or hit any non-black cell; obstacle cells are not overwritten.
    - Rays drawn by earlier clusters remain as obstacles for later clusters, while the original seed pixels stay unchanged.

    Key insight:
    - The visible pattern is a staged ray-casting process: every seed emits local cross arms, but the final arm lengths are determined by global area order because earlier growth blocks later growth.
    """
    out, _stats = _grow_by_area(grid, reverse=False, collect_stats=False)
    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask != 0)
    return mask[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]


def _mask_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask != 0)]


def _random_mask(area: int, rng: np.random.Generator) -> np.ndarray:
    mask = SHAPE_LIBRARY[area][int(rng.integers(len(SHAPE_LIBRARY[area])))].copy()
    if int(rng.integers(2)):
        mask = np.fliplr(mask)
    mask = np.rot90(mask, int(rng.integers(4)))
    return _normalize_mask(mask)


def _well_separated(coords: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _inside(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _place_with_anchor(
    area: int,
    anchor_pos: tuple[int, int],
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
    validator,
) -> list[tuple[int, int]] | None:
    for _ in range(PLACEMENT_RETRIES):
        mask = _random_mask(area, rng)
        cells = _mask_cells(mask)
        anchor_cell = cells[int(rng.integers(len(cells)))]
        coords = [
            (int(anchor_pos[0] + r - anchor_cell[0]), int(anchor_pos[1] + c - anchor_cell[1]))
            for r, c in cells
        ]
        if not _inside(coords, h, w):
            continue
        if not _well_separated(coords, occupied):
            continue
        if validator(coords):
            return coords
    return None


def _place_random(
    area: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    for _ in range(PLACEMENT_RETRIES):
        mask = _random_mask(area, rng)
        cells = _mask_cells(mask)
        mh, mw = mask.shape
        r0 = int(rng.integers(0, h - mh + 1))
        c0 = int(rng.integers(0, w - mw + 1))
        coords = [(int(r0 + r), int(c0 + c)) for r, c in cells]
        if not _well_separated(coords, occupied):
            continue
        if any((r, c) in protected for r, c in coords):
            continue
        return coords
    return None


def _stamp(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(15, 27))
    w = int(rng.integers(15, 27))
    seed_count = int(rng.choice([3, 4, 5], p=[0.45, 0.35, 0.20]))
    areas = sorted(int(x) for x in rng.choice(np.arange(1, 6), size=seed_count, replace=False))
    colors = [int(x) for x in rng.choice(np.arange(1, 10), size=seed_count, replace=False)]

    area_a = areas[0]
    area_b = areas[1]
    area_c = areas[-1]
    extra_areas = areas[2:-1]

    for _ in range(PLACEMENT_RETRIES):
        r_cross = int(rng.integers(2, h - 3))
        c_cross = int(rng.integers(3, w - 4))
        if c_cross - 3 < 1 or w - (c_cross + 3) < 2 or h - (r_cross + 3) < 2:
            continue

        c_a = int(rng.integers(1, c_cross - 2))
        c_c = int(rng.integers(c_cross + 3, w - 1))
        r_b = int(rng.integers(r_cross + 3, h - 1))

        occupied: set[tuple[int, int]] = set()

        coords_a = _place_with_anchor(
            area_a,
            (r_cross, c_a),
            h,
            w,
            occupied,
            rng,
            lambda pts: max(c for _r, c in pts) <= c_cross - 2,
        )
        if coords_a is None:
            continue
        occupied.update(coords_a)

        coords_b = _place_with_anchor(
            area_b,
            (r_b, c_cross),
            h,
            w,
            occupied,
            rng,
            lambda pts: min(r for r, _c in pts) >= r_cross + 2,
        )
        if coords_b is None:
            continue
        occupied.update(coords_b)

        coords_c = _place_with_anchor(
            area_c,
            (r_cross, c_c),
            h,
            w,
            occupied,
            rng,
            lambda pts: min(c for _r, c in pts) >= c_cross + 2,
        )
        if coords_c is None:
            continue
        occupied.update(coords_c)

        right_start = max(c for _r, c in coords_a) + 1
        right_stop = min(c for _r, c in coords_c)
        down_start = r_cross + 1
        down_stop = min(r for r, _c in coords_b)
        if right_start >= right_stop or down_start >= down_stop:
            continue

        protected = {(r_cross, c) for c in range(right_start, right_stop)}
        protected.update((r, c_cross) for r in range(down_start, down_stop))

        extra_coords: list[list[tuple[int, int]]] = []
        extra_ok = True
        for area in extra_areas:
            coords = _place_random(area, h, w, occupied, protected, rng)
            if coords is None:
                extra_ok = False
                break
            extra_coords.append(coords)
            occupied.update(coords)
        if not extra_ok:
            continue

        grid = np.zeros((h, w), dtype=int)
        all_coords = [coords_a, coords_b, coords_c, *extra_coords]
        rng.shuffle(colors)
        for color, coords in zip(colors, all_coords):
            _stamp(grid, coords, color)

        components = _find_seed_components(grid)
        if len(components) != seed_count:
            continue
        if sorted(int(comp["area"]) for comp in components) != areas:
            continue

        out, stats = _grow_by_area(grid, reverse=False, collect_stats=True)
        reverse_out, _ = _grow_by_area(grid, reverse=True, collect_stats=False)

        fill_ratio = float(np.count_nonzero(out)) / float(out.size)
        if stats["painted"] < 18:
            continue
        if stats["grown_components"] < 2:
            continue
        if stats["other_seed_stops"] < 1:
            continue
        if stats["prior_ray_stops"] < 1:
            continue
        if stats["border_stops"] < 4:
            continue
        if np.array_equal(out, grid):
            continue
        if np.array_equal(out, reverse_out):
            continue
        if not (0.15 <= fill_ratio <= 0.72):
            continue

        return grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        out = main(grid)
        if grid.ndim == 2 and out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
