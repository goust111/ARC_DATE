from __future__ import annotations

from collections import Counter, deque
import hashlib
import itertools
import random

import numpy as np


GEN_ATTEMPTS = 400
GRID_MIN_SIZE = 20
GRID_MAX_SIZE = 30
SOLID_MIN_SIZE = 11
SOLID_MAX_SIZE = 19
MIN_GRID_MARGIN = 3
MIN_INTERIOR_DEPTH = 5
MIN_GAUGES = 2
MAX_GAUGES = 4
MIN_NOTCHES = 1
MAX_NOTCHES = 4
MAX_NOTCH_DEPTH = 4
MIN_NOTCH_SPAN = 2

_SEEN_INPUT_SIGNATURES: set[bytes] = set()
_SEEN_OUTPUT_SIGNATURES: set[bytes] = set()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _neighbors8(r: int, c: int):
    rr = int(r)
    cc = int(c)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield rr + dr, cc + dc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    use_diag = int(connectivity) == 8

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                nbrs = _neighbors8(rr, cc) if use_diag else _neighbors4(rr, cc)
                for nr, nc in nbrs:
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _outside_mask(wall_mask: np.ndarray) -> np.ndarray:
    wall_mask = np.asarray(wall_mask, dtype=bool)
    h, w = wall_mask.shape
    outside = np.zeros_like(wall_mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not wall_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(wall_mask.shape, nr, nc) and not wall_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _distance_from_wall(wall_mask: np.ndarray) -> np.ndarray:
    wall_mask = np.asarray(wall_mask, dtype=bool)
    dist = np.full(wall_mask.shape, -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in np.argwhere(wall_mask):
        rr = int(r)
        cc = int(c)
        dist[rr, cc] = 0
        q.append((rr, cc))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(wall_mask.shape, nr, nc) and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _wall_from_solid(solid: np.ndarray) -> np.ndarray:
    solid = np.asarray(solid, dtype=bool)
    wall = np.zeros_like(solid, dtype=bool)
    for r, c in np.argwhere(solid):
        rr = int(r)
        cc = int(c)
        if any(not _in_bounds(solid.shape, nr, nc) or not solid[nr, nc] for nr, nc in _neighbors4(rr, cc)):
            wall[rr, cc] = True
    return wall


def _scene_from_wall_candidate(grid: np.ndarray, bg: int, wall_mask: np.ndarray):
    outside = _outside_mask(wall_mask)
    interior = (~wall_mask) & (~outside)
    if not interior.any():
        return None
    if len(_components(interior, connectivity=4)) != 1:
        return None

    other_non_bg = (grid != int(bg)) & (~wall_mask)
    if not other_non_bg.any():
        return None
    if np.any(other_non_bg & interior):
        return None

    dist = _distance_from_wall(wall_mask)
    depth_to_color: dict[int, int] = {}
    used_colors: set[int] = set()

    for r, c in np.argwhere(other_non_bg):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors8(rr, cc):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != int(bg):
                return None
        d = int(dist[rr, cc])
        if d <= 0 or d in depth_to_color:
            return None
        if not np.any(interior & (dist == d)):
            return None
        color = int(grid[rr, cc])
        if color in used_colors:
            return None
        used_colors.add(color)
        depth_to_color[d] = color

    if not depth_to_color:
        return None

    return wall_mask, outside, interior, dist, depth_to_color


def _identify_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    candidates = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for coords in _components(g == color, connectivity=4):
            if len(coords) <= 1:
                continue
            wall_mask = np.zeros_like(g, dtype=bool)
            wall_mask[coords[:, 0], coords[:, 1]] = True
            scene = _scene_from_wall_candidate(g, bg, wall_mask)
            if scene is not None:
                candidates.append(scene)

    assert len(candidates) == 1, "expected exactly one closed wall with isolated outside gauges"
    wall_mask, outside, interior, dist, depth_to_color = candidates[0]
    return int(bg), wall_mask, outside, interior, dist, depth_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each outside gauge pixel's wall distance to recolor the matching interior ring.

    Input structure:
    - The grid has one dominant background color, one irregular monochrome closed wall,
      and several isolated colored singleton pixels in the exterior region.
    - The wall partitions the canvas into one enclosed interior region and the outside.
    - The interior starts entirely as background, while the exterior gauge colors are all distinct.

    Rule:
    - Identify the unique closed wall and partition non-wall cells into outside and enclosed interior.
    - Measure 4-neighbor distance from every cell to the wall; inside the enclosure this creates
      depth-1, depth-2, ... concentric bands.
    - For each exterior singleton gauge pixel, compute its 4-neighbor distance d to the wall and
      recolor the entire interior band at depth d with that gauge pixel's color.
    - Leave the wall, the outside background, and the gauge pixels themselves unchanged.

    Key insight:
    - The exterior pixels are distance gauges, not local paint seeds: each one selects a full
      interior ring by matching wall-distance depth across the inside/outside partition.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall_mask, _outside, interior, dist, depth_to_color = _identify_scene(g)

    out = g.copy()
    for depth, color in sorted(depth_to_color.items()):
        out[interior & (dist == int(depth))] = int(color)
    return out


def _carve_notch(local: np.ndarray, side: str, depth: int, start: int, span: int) -> None:
    if side == "top":
        local[:depth, start : start + span] = False
    elif side == "bottom":
        local[-depth:, start : start + span] = False
    elif side == "left":
        local[start : start + span, :depth] = False
    else:
        local[start : start + span, -depth:] = False


def _random_local_solid(height: int, width: int, rng: random.Random) -> np.ndarray:
    local = np.ones((height, width), dtype=bool)
    sides = ["top", "bottom", "left", "right"]
    notch_count = rng.randint(MIN_NOTCHES, MAX_NOTCHES)
    for _ in range(notch_count):
        side = rng.choice(sides)
        if side in {"top", "bottom"}:
            max_depth = min(MAX_NOTCH_DEPTH, max(1, height // 3))
            max_span = width - 3
            if max_span < MIN_NOTCH_SPAN:
                continue
            depth = rng.randint(1, max_depth)
            span = rng.randint(MIN_NOTCH_SPAN, max_span)
            start = rng.randint(1, width - span - 1)
        else:
            max_depth = min(MAX_NOTCH_DEPTH, max(1, width // 3))
            max_span = height - 3
            if max_span < MIN_NOTCH_SPAN:
                continue
            depth = rng.randint(1, max_depth)
            span = rng.randint(MIN_NOTCH_SPAN, max_span)
            start = rng.randint(1, height - span - 1)
        _carve_notch(local, side, depth, start, span)
    return local


def _embed_local_solid(local: np.ndarray, grid_shape: tuple[int, int], rng: random.Random) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=bool)
    h, w = map(int, grid_shape)
    lh, lw = local.shape
    r0 = rng.randint(MIN_GRID_MARGIN, h - lh - MIN_GRID_MARGIN)
    c0 = rng.randint(MIN_GRID_MARGIN, w - lw - MIN_GRID_MARGIN)
    grid[r0 : r0 + lh, c0 : c0 + lw] = local
    return grid


def _safe_outside_candidates(wall_mask: np.ndarray, outside: np.ndarray, dist: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    h, w = wall_mask.shape
    candidates: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if not outside[r, c]:
                continue
            d = int(dist[r, c])
            if d < 2:
                continue
            if any(_in_bounds(wall_mask.shape, nr, nc) and wall_mask[nr, nc] for nr, nc in _neighbors8(r, c)):
                continue
            candidates.setdefault(d, []).append((r, c))
    return candidates


def _valid_depth_subsets(interior_depths: list[int], outside_candidates: dict[int, list[tuple[int, int]]]) -> list[tuple[int, ...]]:
    eligible = [d for d in interior_depths if d >= 2 and outside_candidates.get(d)]
    max_count = min(MAX_GAUGES, len(eligible))
    subsets: list[tuple[int, ...]] = []
    for count in range(MIN_GAUGES, max_count + 1):
        for subset in itertools.combinations(eligible, count):
            if len({d % 2 for d in subset}) < 2:
                continue
            if not any(b - a > 1 for a, b in zip(subset, subset[1:])):
                continue
            subsets.append(tuple(int(d) for d in subset))
    return subsets


def _pick_gauge_positions(
    depth_subset: tuple[int, ...],
    outside_candidates: dict[int, list[tuple[int, int]]],
    rng: random.Random,
) -> dict[int, tuple[int, int]] | None:
    order = sorted(depth_subset, key=lambda d: len(outside_candidates[d]))
    picked: dict[int, tuple[int, int]] = {}
    used: list[tuple[int, int]] = []

    def search(i: int) -> bool:
        if i == len(order):
            return True
        depth = order[i]
        options = outside_candidates[depth][:]
        rng.shuffle(options)
        for r, c in options:
            if any(max(abs(r - pr), abs(c - pc)) <= 1 for pr, pc in used):
                continue
            picked[depth] = (r, c)
            used.append((r, c))
            if search(i + 1):
                return True
            used.pop()
            del picked[depth]
        return False

    if search(0):
        return picked
    return None


def _expected_output(
    grid: np.ndarray,
    interior: np.ndarray,
    dist: np.ndarray,
    depth_to_color: dict[int, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for depth, color in depth_to_color.items():
        out[interior & (dist == int(depth))] = int(color)
    return out


def _signature(grid: np.ndarray) -> bytes:
    data = grid.tobytes() + str(tuple(map(int, grid.shape))).encode()
    return hashlib.md5(data).digest()


def _validate_generated_case(
    grid: np.ndarray,
    wall_mask: np.ndarray,
    outside: np.ndarray,
    interior: np.ndarray,
    dist: np.ndarray,
    depth_to_color: dict[int, int],
) -> bool:
    if len(_components(wall_mask, connectivity=4)) != 1:
        return False
    if len(_components(interior, connectivity=4)) != 1:
        return False
    if not outside.any() or not interior.any():
        return False

    selected_depths = sorted(depth_to_color)
    if len(selected_depths) < MIN_GAUGES:
        return False
    if len({d % 2 for d in selected_depths}) < 2:
        return False
    if not any(b - a > 1 for a, b in zip(selected_depths, selected_depths[1:])):
        return False

    interior_depths = sorted({int(v) for v in dist[interior]})
    if max(interior_depths) < MIN_INTERIOR_DEPTH:
        return False
    if 1 not in interior_depths:
        return False
    if any(not np.any(interior & (dist == d)) for d in selected_depths):
        return False
    if not any(d not in depth_to_color for d in interior_depths):
        return False

    gauges = np.argwhere((grid != _get_bg(grid)) & (~wall_mask))
    if len(gauges) != len(depth_to_color):
        return False
    seen_depths: set[int] = set()
    seen_colors: set[int] = set()
    for r, c in gauges:
        rr = int(r)
        cc = int(c)
        if not outside[rr, cc]:
            return False
        for nr, nc in _neighbors8(rr, cc):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != int(_get_bg(grid)):
                return False
        d = int(dist[rr, cc])
        color = int(grid[rr, cc])
        if d in seen_depths or color in seen_colors or depth_to_color.get(d) != color:
            return False
        seen_depths.add(d)
        seen_colors.add(color)

    expected = _expected_output(grid, interior, dist, depth_to_color)
    if np.array_equal(grid, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    for depth, color in depth_to_color.items():
        mask = interior & (dist == int(depth))
        if not mask.any() or not np.all(expected[mask] == int(color)):
            return False
    for depth in interior_depths:
        if depth in depth_to_color:
            continue
        mask = interior & (dist == int(depth))
        if not np.all(expected[mask] == _get_bg(grid)):
            return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    width = rng.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)

    max_local_h = min(SOLID_MAX_SIZE, height - 2 * MIN_GRID_MARGIN)
    max_local_w = min(SOLID_MAX_SIZE, width - 2 * MIN_GRID_MARGIN)
    if max_local_h < SOLID_MIN_SIZE or max_local_w < SOLID_MIN_SIZE:
        return None

    local_h = rng.randint(SOLID_MIN_SIZE, max_local_h)
    local_w = rng.randint(SOLID_MIN_SIZE, max_local_w)
    local = _random_local_solid(local_h, local_w, rng)
    solid = _embed_local_solid(local, (height, width), rng)
    if len(_components(solid, connectivity=4)) != 1:
        return None

    wall_mask = _wall_from_solid(solid)
    if len(_components(wall_mask, connectivity=4)) != 1:
        return None

    outside = _outside_mask(wall_mask)
    interior = (~wall_mask) & (~outside)
    if not interior.any():
        return None
    if len(_components(interior, connectivity=4)) != 1:
        return None

    dist = _distance_from_wall(wall_mask)
    interior_depths = sorted({int(v) for v in dist[interior]})
    if not interior_depths or max(interior_depths) < MIN_INTERIOR_DEPTH:
        return None

    outside_candidates = _safe_outside_candidates(wall_mask, outside, dist)
    depth_subsets = _valid_depth_subsets(interior_depths, outside_candidates)
    if not depth_subsets:
        return None

    counts = sorted({len(subset) for subset in depth_subsets})
    weights = [3 if count == 3 else 2 for count in counts]
    count_order = rng.choices(counts, weights=weights, k=len(counts))
    for count in counts:
        if count not in count_order:
            count_order.append(count)

    chosen_positions = None
    chosen_depths: tuple[int, ...] | None = None
    for count in count_order:
        same_count = [subset for subset in depth_subsets if len(subset) == count]
        rng.shuffle(same_count)
        for depth_subset in same_count:
            positions = _pick_gauge_positions(depth_subset, outside_candidates, rng)
            if positions is not None:
                chosen_positions = positions
                chosen_depths = depth_subset
                break
        if chosen_positions is not None:
            break
    if chosen_positions is None or chosen_depths is None:
        return None

    palette = list(range(10))
    bg = rng.choice(palette)
    remaining = [c for c in palette if c != bg]
    wall_color = rng.choice(remaining)
    remaining = [c for c in remaining if c != wall_color]
    gauge_colors = rng.sample(remaining, len(chosen_depths))
    depth_to_color = {depth: color for depth, color in zip(chosen_depths, gauge_colors)}

    grid = np.full((height, width), bg, dtype=int)
    grid[wall_mask] = wall_color
    for depth, (r, c) in chosen_positions.items():
        grid[r, c] = depth_to_color[depth]

    if not _validate_generated_case(grid, wall_mask, outside, interior, dist, depth_to_color):
        return None

    expected = _expected_output(grid, interior, dist, depth_to_color)
    in_sig = _signature(grid)
    out_sig = _signature(expected)
    if in_sig in _SEEN_INPUT_SIGNATURES or out_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_INPUT_SIGNATURES.add(in_sig)
    _SEEN_OUTPUT_SIGNATURES.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid distance-selects-depth ARC input")
