from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BG = 0
CENTER_COLOR = 8
GUIDE_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
AREA_TO_DIMS = {
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(2, 2), (1, 4), (4, 1)],
    5: [(1, 5), (5, 1)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _all_in_bounds(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                comp.add((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)


def _is_rectangle(cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _candidate_vectors(
    center_cells: set[tuple[int, int]],
    guide_cells: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    for sr, sc in center_cells:
        for tr, tc in guide_cells:
            vec = (tr - sr, tc - sc)
            if vec == (0, 0):
                continue
            dr, dc = vec
            back_shifted = {(r - dr, c - dc) for r, c in guide_cells}
            if back_shifted.issubset(center_cells):
                candidates.add(vec)
    return candidates


def _parse_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    center_comps = _components(g == CENTER_COLOR)
    assert len(center_comps) == 1
    center_cells = center_comps[0]
    assert 5 <= len(center_cells) <= 7
    assert _connected(center_cells)
    assert not _is_rectangle(center_cells)

    guide_comps = _components(g == GUIDE_COLOR)
    assert len(guide_comps) == 3
    for comp in guide_comps:
        assert 1 <= len(comp) <= 3
        assert len(comp) < len(center_cells)
        assert _connected(comp)

    seed_comps: list[dict[str, object]] = []
    for color in sorted(set(int(x) for x in np.unique(g)) - {BG, CENTER_COLOR, GUIDE_COLOR}):
        comps = _components(g == color)
        for comp in comps:
            assert _touches_border(comp, (h, w))
            assert _is_rectangle(comp)
            seed_comps.append({"color": color, "cells": comp, "area": len(comp)})
    assert len(seed_comps) == 3
    assert len({int(seed["area"]) for seed in seed_comps}) == 3

    guides: list[dict[str, object]] = []
    for comp in guide_comps:
        candidates = _candidate_vectors(center_cells, comp)
        assert len(candidates) == 1
        vec = next(iter(candidates))
        guides.append(
            {
                "cells": comp,
                "vector": vec,
                "distance": abs(vec[0]) + abs(vec[1]),
            }
        )

    assert len({int(guide["distance"]) for guide in guides}) == 3
    return {"center": center_cells, "guides": guides, "seeds": seed_comps}


def _render_from_rankings(
    grid: np.ndarray,
    center_cells: set[tuple[int, int]],
    guides: list[dict[str, object]],
    seeds: list[dict[str, object]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    ordered_guides = sorted(guides, key=lambda item: int(item["distance"]))
    ordered_seeds = sorted(seeds, key=lambda item: -int(item["area"]))
    assert len(ordered_guides) == len(ordered_seeds) == 3

    for guide, seed in zip(ordered_guides, ordered_seeds):
        dr, dc = guide["vector"]
        color = int(seed["color"])
        step = 1
        while True:
            placed = False
            for r, c in center_cells:
                rr = r + step * dr
                cc = c + step * dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color
                    placed = True
            if not placed:
                break
            step += 1

    for r, c in center_cells:
        out[r, c] = CENTER_COLOR
    for seed in seeds:
        for r, c in seed["cells"]:
            out[r, c] = int(seed["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend full copies of the center sprite along guide displacements, recoloring each ray by seed-size rank.

    Input structure:
    - Black background with one cyan 4-connected center sprite near the middle.
    - Three gray guide components around it, each being a strict pixel subset of the center sprite under one translation.
    - Three colored rectangular seed blocks touch the border; their colors differ and their areas are all distinct.

    Rule:
    - For each gray guide, recover the translation vector that makes some subset of the center land exactly on that guide.
    - Rank the three guide vectors by Manhattan distance from nearest to farthest, and rank the three seed blocks by area from largest to smallest.
    - Pair equal ranks: the k-th nearest guide ray takes the color of the k-th largest seed block, then repeatedly place the full center sprite at integer multiples of that guide vector.
    - Keep the original center sprite cyan and keep the seed blocks unchanged; if rays overlap, the farther-ranked ray overwrites the nearer-ranked one.

    Key insight:
    - Geometry provides the ray directions while independent seed-area counting provides the colors; only the rank correspondence binds them.
    """
    parsed = _parse_grid(np.asarray(grid, dtype=int))
    return _render_from_rankings(
        np.asarray(grid, dtype=int),
        parsed["center"],
        parsed["guides"],
        parsed["seeds"],
    )


def _random_center_shape(rng: random.Random) -> set[tuple[int, int]]:
    for _ in range(200):
        target = rng.randint(5, 7)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(rng.choice(frontier))
        cells = _normalize(cells)
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 + 1 > 4 or c1 - c0 + 1 > 4:
            continue
        if _is_rectangle(cells):
            continue
        if len({r for r, _c in cells}) < 2 or len({c for _r, c in cells}) < 2:
            continue
        return cells
    raise RuntimeError("failed to build center shape")


def _center_offset(
    shape: set[tuple[int, int]],
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[int, int]:
    _r0, _c0, r1, c1 = _bbox(shape)
    bh = r1 + 1
    bw = c1 + 1
    center_r = h // 2 - bh // 2
    center_c = w // 2 - bw // 2
    row_options = [r for r in range(max(2, center_r - 2), min(h - bh - 1, center_r + 2) + 1)]
    col_options = [c for c in range(max(2, center_c - 2), min(w - bw - 1, center_c + 2) + 1)]
    assert row_options and col_options
    return rng.choice(row_options), rng.choice(col_options)


def _direction_key(vec: tuple[int, int]) -> tuple[int, int]:
    return _sign(vec[0]), _sign(vec[1])


def _has_gap(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in NEIGHBORS8:
            if (r + dr, c + dc) in b:
                return False
    return a.isdisjoint(b)


def _ray_geometry(
    center_cells: set[tuple[int, int]],
    vec: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[list[set[tuple[int, int]]], set[tuple[int, int]]]:
    copies: list[set[tuple[int, int]]] = []
    covered: set[tuple[int, int]] = set()
    step = 1
    while True:
        shifted = _translate(center_cells, step * vec[0], step * vec[1])
        clipped = {cell for cell in shifted if 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]}
        if not clipped:
            break
        copies.append(shifted)
        covered |= clipped
        step += 1
    return copies, covered


def _choose_vectors(
    center_cells: set[tuple[int, int]],
    shape: tuple[int, int],
    want_overlap: bool,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], bool] | None:
    pool: list[tuple[int, int]] = []
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            if (dr, dc) == (0, 0):
                continue
            dist = abs(dr) + abs(dc)
            if dist < 3 or dist > 8:
                continue
            shifted1 = _translate(center_cells, dr, dc)
            shifted2 = _translate(center_cells, 2 * dr, 2 * dc)
            if not _all_in_bounds(shifted1, shape) or not _all_in_bounds(shifted2, shape):
                continue
            if not _has_gap(center_cells, shifted1):
                continue
            pool.append((dr, dc))

    rng.shuffle(pool)
    if len(pool) < 3:
        return None

    # Exhaustively enumerating every 3-vector combination is unnecessary here and
    # makes random validation slow. Sample enough distinct triples to preserve the
    # same constraints while keeping generate_input responsive.
    seen: set[tuple[tuple[int, int], ...]] = set()
    max_trials = min(2500, max(400, len(pool) * 20))
    for _ in range(max_trials):
        combo = tuple(rng.sample(pool, 3))
        key = tuple(sorted(combo))
        if key in seen:
            continue
        seen.add(key)
        dists = [abs(dr) + abs(dc) for dr, dc in combo]
        if len(set(dists)) != 3:
            continue
        if len({_direction_key(vec) for vec in combo}) != 3:
            continue
        count = Counter()
        for vec in combo:
            _copies, covered = _ray_geometry(center_cells, vec, shape)
            for cell in covered:
                count[cell] += 1
        overlap = any(v > 1 for v in count.values())
        if want_overlap and not overlap:
            continue
        return list(combo), overlap
    return None


def _connected_subsets(cells: set[tuple[int, int]], size: int) -> list[set[tuple[int, int]]]:
    out: list[set[tuple[int, int]]] = []
    ordered = sorted(cells)
    for combo in combinations(ordered, size):
        subset = set(combo)
        if _connected(subset):
            out.append(subset)
    return out


def _choose_guides(
    center_cells: set[tuple[int, int]],
    vectors: list[tuple[int, int]],
    rng: random.Random,
) -> list[dict[str, object]] | None:
    size_patterns = [
        [1, 2, 3],
        [1, 1, 3],
        [1, 2, 2],
        [2, 2, 3],
    ]
    candidates_by_size = {size: _connected_subsets(center_cells, size) for size in (1, 2, 3)}
    for _ in range(200):
        order = list(range(3))
        rng.shuffle(order)
        sizes = list(rng.choice(size_patterns))
        rng.shuffle(sizes)
        guides: list[dict[str, object]] = []
        occupied = set(center_cells)
        ok = True
        for idx, size in zip(order, sizes):
            vec = vectors[idx]
            subset_options = candidates_by_size[size][:]
            rng.shuffle(subset_options)
            picked = None
            for subset in subset_options:
                guide_cells = _translate(subset, vec[0], vec[1])
                if guide_cells & occupied:
                    continue
                if not _has_gap(guide_cells, occupied):
                    continue
                if _candidate_vectors(center_cells, guide_cells) != {vec}:
                    continue
                picked = guide_cells
                break
            if picked is None:
                ok = False
                break
            guides.append(
                {
                    "cells": picked,
                    "vector": vec,
                    "distance": abs(vec[0]) + abs(vec[1]),
                    "size": len(picked),
                }
            )
            occupied |= picked
        if ok and len(guides) == 3:
            return guides
    return None


def _buffer_ok(rect: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    if rect & occupied:
        return False
    for r, c in rect:
        for dr, dc in NEIGHBORS8:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _edge_positions(edge: str, rh: int, rw: int, h: int, w: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    if edge == "top":
        for c in range(1, w - rw):
            positions.append((0, c))
    elif edge == "bottom":
        for c in range(1, w - rw):
            positions.append((h - rh, c))
    elif edge == "left":
        for r in range(1, h - rh):
            positions.append((r, 0))
    else:
        for r in range(1, h - rh):
            positions.append((r, w - rw))
    return positions


def _choose_seeds(
    h: int,
    w: int,
    occupied_input: set[tuple[int, int]],
    ray_union: set[tuple[int, int]],
    rng: random.Random,
) -> list[dict[str, object]] | None:
    colors = rng.sample(list(SEED_COLORS), 3)
    areas = rng.sample(sorted(AREA_TO_DIMS), 3)
    if rng.random() < 0.7:
        edges = rng.sample(["top", "bottom", "left", "right"], 3)
    else:
        edges = [rng.choice(["top", "bottom", "left", "right"]) for _ in range(3)]
    occupied = set(occupied_input)
    seeds: list[dict[str, object]] = []
    for color, area, edge in zip(colors, areas, edges):
        dims = AREA_TO_DIMS[area][:]
        rng.shuffle(dims)
        placed = None
        for rh, rw in dims:
            positions = _edge_positions(edge, rh, rw, h, w)
            rng.shuffle(positions)
            for r0, c0 in positions:
                rect = {(r0 + dr, c0 + dc) for dr in range(rh) for dc in range(rw)}
                if rect & ray_union:
                    continue
                if not _buffer_ok(rect, occupied):
                    continue
                placed = {"color": color, "cells": rect, "area": area}
                break
            if placed is not None:
                break
        if placed is None:
            return None
        seeds.append(placed)
        occupied |= placed["cells"]
    return seeds


def _instance_to_grid(
    h: int,
    w: int,
    center_cells: set[tuple[int, int]],
    guides: list[dict[str, object]],
    seeds: list[dict[str, object]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c in center_cells:
        grid[r, c] = CENTER_COLOR
    for guide in guides:
        for r, c in guide["cells"]:
            grid[r, c] = GUIDE_COLOR
    for seed in seeds:
        for r, c in seed["cells"]:
            grid[r, c] = int(seed["color"])
    return grid


def _render_expected(instance: dict[str, object]) -> np.ndarray:
    return _render_from_rankings(
        instance["grid"],
        instance["center"],
        instance["guides"],
        instance["seeds"],
    )


def _validate_instance(instance: dict[str, object]) -> None:
    grid = instance["grid"]
    parsed = _parse_grid(grid)
    expected = _render_expected(instance)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, grid)

    distances = sorted(abs(v[0]) + abs(v[1]) for v in instance["vectors"])
    assert len(set(distances)) == 3
    for vec in instance["vectors"]:
        assert _all_in_bounds(_translate(instance["center"], vec[0], vec[1]), grid.shape)
        assert _all_in_bounds(_translate(instance["center"], 2 * vec[0], 2 * vec[1]), grid.shape)

    all_ray_cells = Counter()
    for vec in instance["vectors"]:
        _copies, covered = _ray_geometry(instance["center"], vec, grid.shape)
        for cell in covered:
            all_ray_cells[cell] += 1
    overlap = any(v > 1 for v in all_ray_cells.values())
    if instance["want_overlap"]:
        assert overlap

    seed_areas = sorted((int(seed["area"]) for seed in parsed["seeds"]), reverse=True)
    assert len(set(seed_areas)) == 3
    ray_colors = sorted(set(int(x) for x in np.unique(actual)) - {BG, CENTER_COLOR} - set(int(seed["color"]) for seed in parsed["seeds"]))
    assert not ray_colors


def _build_instance(rng: random.Random) -> dict[str, object] | None:
    h = rng.randint(17, 24)
    w = rng.randint(17, 24)
    center_local = _random_center_shape(rng)
    off_r, off_c = _center_offset(center_local, h, w, rng)
    center_cells = _translate(center_local, off_r, off_c)

    want_overlap = rng.random() < 0.45
    chosen = _choose_vectors(center_cells, (h, w), want_overlap, rng)
    if chosen is None:
        return None
    vectors, actual_overlap = chosen

    guides = _choose_guides(center_cells, vectors, rng)
    if guides is None:
        return None

    ray_union: set[tuple[int, int]] = set()
    for vec in vectors:
        _copies, covered = _ray_geometry(center_cells, vec, (h, w))
        ray_union |= covered
    occupied_input = set(center_cells)
    for guide in guides:
        occupied_input |= guide["cells"]

    seeds = _choose_seeds(h, w, occupied_input, ray_union, rng)
    if seeds is None:
        return None

    grid = _instance_to_grid(h, w, center_cells, guides, seeds)
    instance = {
        "grid": grid,
        "center": center_cells,
        "guides": guides,
        "seeds": seeds,
        "vectors": vectors,
        "want_overlap": want_overlap,
        "actual_overlap": actual_overlap,
    }
    _validate_instance(instance)
    return instance


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        instance = _build_instance(rng)
        if instance is not None:
            return instance["grid"]
    raise RuntimeError("failed to generate a valid puzzle instance")
