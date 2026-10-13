import itertools
import random

import numpy as np


BG = 0
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
RING_OFFSETS = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
]
CORNER_OFFSETS = [(0, 0), (0, 2), (2, 2), (2, 0)]
SIDE_OFFSETS = [(0, 1), (1, 2), (2, 1), (1, 0)]
GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _ring_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr, dc in RING_OFFSETS}


def _is_ring_at(grid: np.ndarray, r0: int, c0: int) -> bool:
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + 2 >= H or c0 + 2 >= W:
        return False
    if int(grid[r0 + 1, c0 + 1]) != BG:
        return False
    return all(int(grid[r0 + dr, c0 + dc]) != BG for dr, dc in RING_OFFSETS)


def _detect_rings(grid: np.ndarray) -> list[dict]:
    rings = []
    seen_cells: set[tuple[int, int]] = set()
    for r0 in range(grid.shape[0] - 2):
        for c0 in range(grid.shape[1] - 2):
            if not _is_ring_at(grid, r0, c0):
                continue
            cells = _ring_cells(r0, c0)
            if seen_cells & cells:
                raise AssertionError("overlapping rings are not allowed")
            seen_cells |= cells
            rings.append(
                {
                    "top_left": (r0, c0),
                    "cells": cells,
                    "corners": [(r0 + dr, c0 + dc) for dr, dc in CORNER_OFFSETS],
                    "sides": [(r0 + dr, c0 + dc) for dr, dc in SIDE_OFFSETS],
                }
            )
    return rings


def _source_cells(grid: np.ndarray, ring_cells: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    sources = []
    for r, c in np.argwhere(grid != BG):
        rr = int(r)
        cc = int(c)
        if (rr, cc) in ring_cells:
            continue
        sources.append((rr, cc, int(grid[rr, cc])))
    return sources


def _farther_ring_exists(
    ring_by_cell: dict[tuple[int, int], int],
    shape: tuple[int, int],
    ring_idx: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> bool:
    rr = r + dr
    cc = c + dc
    while _in_bounds(shape, rr, cc):
        other = ring_by_cell.get((rr, cc))
        if other is not None and other != ring_idx:
            return True
        rr += dr
        cc += dc
    return False


def _trace_ray(
    shape: tuple[int, int],
    start: tuple[int, int],
    dr: int,
    dc: int,
    ring_by_cell: dict[tuple[int, int], int],
    ring_cells_list: list[set[tuple[int, int]]],
) -> dict:
    r, c = start
    path = []
    while True:
        nr = r + dr
        nc = c + dc
        if not _in_bounds(shape, nr, nc):
            return {
                "endpoint": (r, c),
                "path": path,
                "ring_idx": None,
                "counted": False,
                "shadow": False,
            }
        ring_idx = ring_by_cell.get((nr, nc))
        if ring_idx is not None:
            counted = any((ar, ac) in ring_cells_list[ring_idx] for ar, ac in _neighbors4(r, c))
            return {
                "endpoint": (r, c),
                "path": path,
                "ring_idx": ring_idx,
                "counted": counted,
                "shadow": _farther_ring_exists(ring_by_cell, shape, ring_idx, nr, nc, dr, dc),
            }
        path.append((nr, nc))
        r, c = nr, nc


def _simulate_scene(grid: np.ndarray, rings: list[dict], sources: list[tuple[int, int, int]]) -> tuple[np.ndarray, list[int], list[dict]]:
    out = np.array(grid, dtype=int, copy=True)
    ring_by_cell: dict[tuple[int, int], int] = {}
    ring_cells_list = [ring["cells"] for ring in rings]
    for idx, ring in enumerate(rings):
        for cell in ring["cells"]:
            ring_by_cell[cell] = idx

    touch_counts = [0] * len(rings)
    ray_records = []
    for r, c, color in sources:
        for dr, dc in DIAGONALS:
            ray = _trace_ray(grid.shape, (r, c), dr, dc, ring_by_cell, ring_cells_list)
            for rr, cc in ray["path"]:
                prev = int(out[rr, cc])
                if prev not in (BG, color):
                    raise AssertionError("ray colors conflict")
                out[rr, cc] = color
            if ray["counted"]:
                touch_counts[int(ray["ring_idx"])] += 1
            ray_records.append({**ray, "start": (r, c), "color": color, "dir": (dr, dc)})
    return out, touch_counts, ray_records


def _rotate_ring_in_place(out: np.ndarray, grid: np.ndarray, ring: dict, k: int) -> None:
    k %= 4
    if k == 0:
        return
    corner_vals = [int(grid[r, c]) for r, c in ring["corners"]]
    side_vals = [int(grid[r, c]) for r, c in ring["sides"]]
    rotated_corners = np.roll(np.array(corner_vals, dtype=int), k)
    rotated_sides = np.roll(np.array(side_vals, dtype=int), -k)
    for (r, c), value in zip(ring["corners"], rotated_corners):
        out[r, c] = int(value)
    for (r, c), value in zip(ring["sides"], rotated_sides):
        out[r, c] = int(value)


def main(grid: np.ndarray) -> np.ndarray:
    """Draw diagonal source rays, count which 3x3 rings they reach first, and rotate each ring by that touch count.

    Input structure:
    - The grid is black background with several complete 3x3 ring objects: eight non-black
      cells around a black center.
    - Additional non-black cells are isolated single-pixel ray sources, separate from the rings.
    - Rings do not overlap, and source pixels are not part of any 3x3 ring.

    Rule:
    - From every isolated source, extend four rays along the diagonal directions using the
      source color.
    - A ray stops before leaving the canvas or before stepping into the first ring cell on
      that diagonal; the source pixel and every traversed background cell remain colored.
    - For each ring, count how many ray endpoints are 4-neighbor-adjacent to that same first-hit
      ring. Rotate the four corner colors clockwise by k steps and the four side-center colors
      counterclockwise by k steps, where k is the ring's touch count.

    Key insight:
    - The diagonal geometry determines which ring each ray reaches first, so nearer rings can
      shield farther rings; those touch counts are then converted into opposite-direction color cycles
      on the ring's corner and side positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rings = _detect_rings(g)
    assert rings, "expected at least one 3x3 ring"
    all_ring_cells = set().union(*(ring["cells"] for ring in rings))
    sources = _source_cells(g, all_ring_cells)
    assert sources, "expected at least one isolated source pixel"

    out, touch_counts, _ray_records = _simulate_scene(g, rings, sources)
    for ring, count in zip(rings, touch_counts):
        _rotate_ring_in_place(out, g, ring, count)
    return out


def _boxes_separated(a: tuple[int, int], b: tuple[int, int], pad: int = 1) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 2, ac0 + 2
    br1, bc1 = br0 + 2, bc0 + 2
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _random_nonperiodic_cycle(palette: list[int]) -> list[int]:
    for _ in range(200):
        if random.random() < 0.5:
            values = random.sample(palette, 4)
        else:
            values = [random.choice(palette) for _ in range(4)]
        arr = np.array(values, dtype=int)
        if len(set(values)) < 2:
            continue
        if any(np.array_equal(arr, np.roll(arr, k)) for k in (1, 2, 3)):
            continue
        return values
    raise RuntimeError("failed to sample a visible 4-cycle")


def _paint_ring(grid: np.ndarray, top_left: tuple[int, int], palette: list[int]) -> None:
    r0, c0 = top_left
    corners = _random_nonperiodic_cycle(palette)
    sides = _random_nonperiodic_cycle(palette)
    patch = np.zeros((3, 3), dtype=int)
    for value, (dr, dc) in zip(corners, CORNER_OFFSETS):
        patch[dr, dc] = value
    for value, (dr, dc) in zip(sides, SIDE_OFFSETS):
        patch[dr, dc] = value
    grid[r0 : r0 + 3, c0 : c0 + 3] = patch


def _choose_ring_positions(H: int, W: int, count: int, require_shadow: bool) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = []
    if require_shadow and count >= 2:
        dr, dc = random.choice(DIAGONALS)
        candidates = []
        for r0 in range(1, H - 3):
            for c0 in range(1, W - 3):
                other = (r0 + 4 * dr, c0 + 4 * dc)
                if not (1 <= other[0] <= H - 4 and 1 <= other[1] <= W - 4):
                    continue
                if _boxes_separated((r0, c0), other, pad=1):
                    candidates.append(((r0, c0), other))
        if not candidates:
            return None
        first, second = random.choice(candidates)
        positions.extend([first, second])

    pool = [(r0, c0) for r0 in range(1, H - 3) for c0 in range(1, W - 3)]
    random.shuffle(pool)
    for pos in pool:
        if pos in positions:
            continue
        if all(_boxes_separated(pos, other, pad=1) for other in positions):
            positions.append(pos)
            if len(positions) == count:
                return positions
    return None


def _cell_is_isolated_from_rings(cell: tuple[int, int], ring_cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    r, c = cell
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if _in_bounds(shape, rr, cc) and (rr, cc) in ring_cells:
                return False
    return True


def _candidate_sources(rings: list[dict], shape: tuple[int, int]) -> list[dict]:
    H, W = shape
    ring_cells = set().union(*(ring["cells"] for ring in rings))
    ring_by_cell: dict[tuple[int, int], int] = {}
    ring_cells_list = [ring["cells"] for ring in rings]
    for idx, ring in enumerate(rings):
        for cell in ring["cells"]:
            ring_by_cell[cell] = idx

    candidates = []
    for r in range(H):
        for c in range(W):
            if (r, c) in ring_cells:
                continue
            if not _cell_is_isolated_from_rings((r, c), ring_cells, shape):
                continue
            rays = [_trace_ray(shape, (r, c), dr, dc, ring_by_cell, ring_cells_list) for dr, dc in DIAGONALS]
            if not any(ray["counted"] for ray in rays):
                continue
            occupied = {(r, c)}
            counts = [0] * len(rings)
            for ray in rays:
                occupied.update(ray["path"])
                if ray["counted"]:
                    counts[int(ray["ring_idx"])] += 1
            candidates.append(
                {
                    "pos": (r, c),
                    "rays": rays,
                    "occupied": frozenset(occupied),
                    "counts": tuple(counts),
                    "has_shadow": any(ray["counted"] and ray["shadow"] for ray in rays),
                }
            )
    random.shuffle(candidates)
    if len(candidates) > 80:
        candidates = candidates[:80]
    return candidates


def _sources_compatible(a: dict, b: dict) -> bool:
    (r1, c1) = a["pos"]
    (r2, c2) = b["pos"]
    if r1 == r2 or c1 == c2 or r1 - c1 == r2 - c2 or r1 + c1 == r2 + c2:
        return False
    return a["occupied"].isdisjoint(b["occupied"])


def _combine_counts(combo: tuple[dict, ...], ring_count: int) -> list[int]:
    total = [0] * ring_count
    for cand in combo:
        for idx, value in enumerate(cand["counts"]):
            total[idx] += int(value)
    return total


def _combo_ok(
    combo: tuple[dict, ...],
    ring_count: int,
    require_shadow: bool,
    require_untouched: bool,
) -> bool:
    counts = _combine_counts(combo, ring_count)
    if max(counts) < 2:
        return False
    if require_untouched and ring_count >= 3 and 0 not in counts:
        return False
    if len(set(counts)) < 2:
        return False
    if any(count > 0 and count % 4 == 0 for count in counts):
        return False
    if require_shadow and not any(cand["has_shadow"] for cand in combo):
        return False
    return True


def _choose_source_combo(rings: list[dict], shape: tuple[int, int], source_count: int, require_shadow: bool) -> list[tuple[int, int]] | None:
    candidates = _candidate_sources(rings, shape)
    if len(candidates) < source_count:
        return None

    require_untouched = random.random() < 0.6
    if source_count == 2:
        for a_idx in range(len(candidates)):
            a = candidates[a_idx]
            for b_idx in range(a_idx + 1, len(candidates)):
                b = candidates[b_idx]
                if not _sources_compatible(a, b):
                    continue
                combo = (a, b)
                if _combo_ok(combo, len(rings), require_shadow, require_untouched):
                    return [a["pos"], b["pos"]]
        return None

    for a_idx in range(len(candidates)):
        a = candidates[a_idx]
        for b_idx in range(a_idx + 1, len(candidates)):
            b = candidates[b_idx]
            if not _sources_compatible(a, b):
                continue
            for c_idx in range(b_idx + 1, len(candidates)):
                c = candidates[c_idx]
                if not _sources_compatible(a, c) or not _sources_compatible(b, c):
                    continue
                combo = (a, b, c)
                if _combo_ok(combo, len(rings), require_shadow, require_untouched):
                    return [a["pos"], b["pos"], c["pos"]]
    return None


def _build_input(H: int, W: int, ring_positions: list[tuple[int, int]], source_positions: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    source_count = len(source_positions)
    all_colors = list(range(1, 10))
    source_colors = random.sample(all_colors, source_count)
    ring_palette = random.sample([color for color in all_colors if color not in source_colors], 4)

    for top_left in ring_positions:
        _paint_ring(grid, top_left, ring_palette)
    for color, (r, c) in zip(source_colors, source_positions):
        grid[r, c] = color
    return grid


def _semantic_checks(grid: np.ndarray) -> None:
    rings = _detect_rings(grid)
    assert 2 <= len(rings) <= 4
    ring_cells = set().union(*(ring["cells"] for ring in rings))
    sources = _source_cells(grid, ring_cells)
    assert 2 <= len(sources) <= 3

    for i, (r1, c1, col1) in enumerate(sources):
        assert col1 != BG
        for j in range(i + 1, len(sources)):
            r2, c2, _ = sources[j]
            assert r1 != r2 and c1 != c2
            assert r1 - c1 != r2 - c2 and r1 + c1 != r2 + c2

    _, counts, ray_records = _simulate_scene(grid, rings, sources)
    by_source: dict[tuple[int, int], int] = {}
    for ray in ray_records:
        if ray["counted"]:
            by_source[ray["start"]] = by_source.get(ray["start"], 0) + 1
    assert len(by_source) == len(sources)
    assert max(counts) >= 2
    assert all(count == 0 or count % 4 != 0 for count in counts)
    assert len(set(counts)) >= 2


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(14, 18)
    W = random.randint(14, 18)
    ring_count = random.choice([2, 3, 3, 4])
    source_count = random.choice([2, 3])
    require_shadow = random.random() < 0.55

    ring_positions = _choose_ring_positions(H, W, ring_count, require_shadow)
    if ring_positions is None:
        return None
    source_positions = _choose_source_combo(
        [
            {"cells": _ring_cells(r0, c0), "top_left": (r0, c0)}
            for r0, c0 in ring_positions
        ],
        (H, W),
        source_count,
        require_shadow,
    )
    if source_positions is None:
        return None

    grid = _build_input(H, W, ring_positions, source_positions)
    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-ring counting scene")
