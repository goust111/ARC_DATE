from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
import random

import numpy as np


MIN_COMPONENT_DISTANCE = 5
GENERATE_ATTEMPTS = 600
CANDIDATE_SAMPLE_LIMIT = 220


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _normalize_cells(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _variants_from_pattern(lines: tuple[str, ...]) -> list[tuple[tuple[int, int], ...]]:
    arr = np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)
    variants: dict[tuple[tuple[int, int], ...], None] = {}
    for k in range(4):
        rotated = np.rot90(arr, k)
        for variant in (rotated, np.fliplr(rotated)):
            coords = tuple((int(r), int(c)) for r, c in np.argwhere(variant))
            variants[_normalize_cells(coords)] = None
    return list(variants)


def _build_shape_library() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    base_patterns = {
        1: [("#",)],
        2: [("##",)],
        3: [("###",), ("##", "#.")],
        4: [
            ("####",),
            ("##", "##"),
            ("###", ".#."),
            ("###", "#.."),
            ("##.", ".##"),
        ],
    }
    shapes: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for size, patterns in base_patterns.items():
        bucket: dict[tuple[tuple[int, int], ...], None] = {}
        for pattern in patterns:
            for variant in _variants_from_pattern(pattern):
                bucket[variant] = None
        shapes[size] = list(bucket)
    return shapes


SHAPES_BY_SIZE = _build_shape_library()


def _seed_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append({"color": color, "coords": tuple(coords), "size": len(coords)})
    return components


def _growth_region(bg_mask: np.ndarray, seed_coords: tuple[tuple[int, int], ...], steps: int) -> set[tuple[int, int]]:
    h, w = bg_mask.shape
    seen = set(seed_coords)
    queue = deque((r, c, 0) for r, c in seed_coords)
    grown: set[tuple[int, int]] = set()

    while queue:
        r, c, dist = queue.popleft()
        if dist >= steps:
            continue
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen or not bool(bg_mask[nr, nc]):
                continue
            seen.add((nr, nc))
            grown.add((nr, nc))
            queue.append((nr, nc, dist + 1))
    return grown


def _claim_summary(grid: np.ndarray) -> tuple[
    list[dict[str, object]],
    dict[tuple[int, int], list[int]],
    dict[tuple[int, int], list[int]],
    set[tuple[int, int]],
]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    components = _seed_components(g, bg)
    bg_mask = g == bg
    claimants: dict[tuple[int, int], list[int]] = defaultdict(list)

    for idx, comp in enumerate(components):
        reach = _growth_region(bg_mask, comp["coords"], int(comp["size"]))
        for cell in reach:
            claimants[cell].append(idx)

    conflict_cells = {cell: owners for cell, owners in claimants.items() if len(owners) >= 2}
    conflict_pairs: set[tuple[int, int]] = set()
    for owners in conflict_cells.values():
        for a, b in combinations(sorted(owners), 2):
            conflict_pairs.add((int(a), int(b)))
    return components, claimants, conflict_cells, conflict_pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each 4-connected monochrome seed by as many 4-neighbor layers as its area.
    Cells claimed by multiple seeds stay as background.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    components, claimants, _conflict_cells, _conflict_pairs = _claim_summary(g)

    out = g.copy()
    for cell, owners in claimants.items():
        r, c = cell
        if len(owners) == 1:
            out[r, c] = int(components[owners[0]]["color"])
        else:
            out[r, c] = bg
    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _absolute_shape(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple((int(r0 + r), int(c0 + c)) for r, c in shape)


def _all_placements(
    shape: tuple[tuple[int, int], ...], height: int, width: int, preferred_pad: int
) -> list[tuple[tuple[int, int], ...]]:
    sh, sw = _shape_dims(shape)
    for pad in (preferred_pad, 1, 0):
        r_lo = pad
        c_lo = pad
        r_hi = height - sh - pad
        c_hi = width - sw - pad
        if r_lo > r_hi or c_lo > c_hi:
            continue
        placements = [
            _absolute_shape(shape, r0, c0)
            for r0 in range(r_lo, r_hi + 1)
            for c0 in range(c_lo, c_hi + 1)
        ]
        random.shuffle(placements)
        return placements
    return []


def _min_manhattan_distance(
    a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]
) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def _sample_sizes(mode: str, seed_count: int) -> list[int]:
    if mode == "none":
        return [random.randint(1, 4) for _ in range(seed_count)]
    if mode == "one":
        pair_options = [(a, b) for a in range(1, 5) for b in range(1, 5) if a + b >= MIN_COMPONENT_DISTANCE]
        a, b = random.choice(pair_options)
        return [a, b] + [random.randint(1, 4) for _ in range(seed_count - 2)]
    return [random.randint(2, 4) for _ in range(3)] + [random.randint(1, 4) for _ in range(seed_count - 3)]


def _relation_overrides(mode: str) -> dict[tuple[int, int], str]:
    if mode == "none":
        return {}
    if mode == "one":
        return {(0, 1): "conflict"}
    if random.random() < 0.5:
        return {(0, 1): "conflict", (1, 2): "conflict", (0, 2): "free"}
    return {(0, 1): "conflict", (1, 2): "conflict", (0, 2): "conflict"}


def _pair_relation(
    i: int, j: int, mode: str, overrides: dict[tuple[int, int], str]
) -> str:
    if i > j:
        i, j = j, i
    return overrides.get((i, j), "separate")


def _place_seeds(
    height: int,
    width: int,
    shapes: list[tuple[tuple[int, int], ...]],
    sizes: list[int],
    mode: str,
    overrides: dict[tuple[int, int], str],
) -> list[tuple[tuple[int, int], ...]] | None:
    preferred_pad = 2 if min(height, width) >= 18 else 1
    candidates = [_all_placements(shape, height, width, preferred_pad) for shape in shapes]
    if any(not opts for opts in candidates):
        return None

    required_degree = [0] * len(shapes)
    for (a, b), rel in overrides.items():
        if rel == "conflict":
            required_degree[a] += 1
            required_degree[b] += 1

    order = list(range(len(shapes)))
    random.shuffle(order)
    order.sort(key=lambda idx: (-required_degree[idx], -sizes[idx]))

    placed: dict[int, tuple[tuple[int, int], ...]] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True

        idx = order[pos]
        options = candidates[idx]
        if len(options) > CANDIDATE_SAMPLE_LIMIT:
            options = random.sample(options, CANDIDATE_SAMPLE_LIMIT)
        else:
            options = list(options)
            random.shuffle(options)

        for coords in options:
            ok = True
            for other_idx, other_coords in placed.items():
                dist = _min_manhattan_distance(coords, other_coords)
                if dist < MIN_COMPONENT_DISTANCE:
                    ok = False
                    break
                relation = _pair_relation(idx, other_idx, mode, overrides)
                if relation == "conflict" and dist > sizes[idx] + sizes[other_idx]:
                    ok = False
                    break
                if relation == "separate" and dist <= sizes[idx] + sizes[other_idx]:
                    ok = False
                    break
            if not ok:
                continue
            placed[idx] = coords
            if backtrack(pos + 1):
                return True
            del placed[idx]
        return False

    if not backtrack(0):
        return None
    return [placed[idx] for idx in range(len(shapes))]


def _build_grid(
    height: int,
    width: int,
    bg: int,
    colors: list[int],
    placements: list[tuple[tuple[int, int], ...]],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for color, coords in zip(colors, placements):
        for r, c in coords:
            grid[r, c] = int(color)
    return grid


def _mode_matches(mode: str, conflict_pairs: set[tuple[int, int]], conflict_cells: dict[tuple[int, int], list[int]]) -> bool:
    if mode == "none":
        return len(conflict_pairs) == 0
    if mode == "one":
        return len(conflict_pairs) == 1 and bool(conflict_cells)
    return len(conflict_pairs) >= 2 and bool(conflict_cells)


def _generate_attempt() -> np.ndarray | None:
    mode = random.choice(("none", "one", "many"))
    if mode == "none":
        seed_count = random.randint(2, 5)
    elif mode == "one":
        seed_count = random.randint(2, 4)
    else:
        seed_count = random.randint(3, 5)

    sizes = _sample_sizes(mode, seed_count)
    shapes = [random.choice(SHAPES_BY_SIZE[size]) for size in sizes]
    overrides = _relation_overrides(mode)

    base = 8 + 3 * seed_count
    height = random.randint(base, base + 4)
    width = random.randint(base, base + 5)

    palette = random.sample(range(10), seed_count + 1)
    bg = palette[0]
    colors = palette[1:]

    placements = _place_seeds(height, width, shapes, sizes, mode, overrides)
    if placements is None:
        return None

    grid = _build_grid(height, width, bg, colors, placements)

    # Re-validate with the exact solver logic instead of relying only on distance heuristics.
    components, claimants, conflict_cells, conflict_pairs = _claim_summary(grid)
    if len(components) != seed_count:
        return None
    if sorted(int(comp["size"]) for comp in components) != sorted(sizes):
        return None
    if not _mode_matches(mode, conflict_pairs, conflict_cells):
        return None

    out = main(grid)
    if not np.any(out != grid):
        return None

    values, counts = np.unique(grid, return_counts=True)
    if int(counts[int(np.argmax(counts))]) != int(np.sum(grid == bg)):
        return None
    if max(int(np.sum(grid == color)) for color in colors) > 4:
        return None
    if any(len(owners) > 3 for owners in conflict_cells.values()) and mode == "one":
        return None
    if not claimants:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
