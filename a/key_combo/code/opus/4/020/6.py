from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
Pair = tuple[Coord, Coord]


_BG = 0
_NON_BG_COLORS = tuple(range(1, 10))
_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MAX_GEN_ATTEMPTS = 500
_GEN_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    mask = grid != bg
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            color_order: list[int] = []
            color_seen: set[int] = set()

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                color = int(grid[rr, cc])
                if color not in color_seen:
                    color_seen.add(color)
                    color_order.append(color)

                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "colors": tuple(color_order),
                }
            )

    return comps


def _crop_non_bg(grid: np.ndarray, bg: int = _BG) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    if coords.size == 0:
        return grid.copy()
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _canonical_pair(a: Coord, b: Coord) -> Pair:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def _pair_key(a: Coord, b: Coord) -> tuple[int, int, int, int, int, int, int]:
    (r0, c0), (r1, c1) = _canonical_pair(a, b)
    return (
        abs(r0 - r1) + abs(c0 - c1),
        r0 + r1,
        c0 + c1,
        r0,
        c0,
        r1,
        c1,
    )


def _greedy_pairs(coords: list[Coord]) -> list[Pair]:
    remaining = set(coords)
    pairs: list[Pair] = []

    while remaining:
        ordered = sorted(remaining)
        best_pair: Pair | None = None
        best_key: tuple[int, int, int, int, int, int, int] | None = None

        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                key = _pair_key(a, b)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = _canonical_pair(a, b)

        assert best_pair is not None
        remaining.remove(best_pair[0])
        remaining.remove(best_pair[1])
        pairs.append(best_pair)

    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Use greedily paired isolated pixels to define global color swaps, then apply them inside matching two-color objects.

    Input structure:
    - The grid has black background and several 4-connected foreground components.
    - Some foreground components are isolated single pixels whose colors are all distinct.
    - Every remaining multi-pixel foreground object is 4-connected, irregular, and uses exactly two non-background colors.

    Rule:
    - Find all size-1 foreground components and greedily pair them by minimum Manhattan distance, breaking ties by smaller row-sum then smaller col-sum.
    - Each isolated-pixel pair defines one global A<->B swap rule from its two colors.
    - Remove the isolated pixels, then for every multi-pixel object whose two colors match one rule, swap those two colors inside that object only.

    Key insight:
    - The isolated pixels are not part of the final picture; they act as remote code markers whose greedy spatial pairing encodes which color pairs must be exchanged elsewhere.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    comps = _find_components(g, bg)
    isolated = [tuple(map(int, comp["coords"][0])) for comp in comps if comp["area"] == 1]
    objects = [comp for comp in comps if comp["area"] > 1]

    assert isolated and len(isolated) % 2 == 0

    out = g.copy()
    rules: dict[frozenset[int], dict[int, int]] = {}
    for a, b in _greedy_pairs(isolated):
        ca = int(g[a])
        cb = int(g[b])
        rules[frozenset((ca, cb))] = {ca: cb, cb: ca}
        out[a] = bg
        out[b] = bg

    for comp in objects:
        if len(comp["colors"]) != 2:
            continue
        key = frozenset(comp["colors"])
        if key not in rules:
            continue
        mapping = rules[key]
        coords = comp["coords"]
        values = g[coords[:, 0], coords[:, 1]]
        swapped = np.array([mapping[int(v)] for v in values], dtype=int)
        out[coords[:, 0], coords[:, 1]] = swapped

    return out


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _connected_mask(area: int) -> np.ndarray:
    for _ in range(120):
        dims = [
            (h, w)
            for h in range(2, 6)
            for w in range(2, 6)
            if h * w >= area + 1 and h * w <= area + 7
        ]
        h, w = random.choice(dims)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True

        while int(mask.sum()) < area:
            frontier: list[Coord] = []
            for rr, cc in zip(*np.where(mask)):
                for dr, dc in _NEIGHBORS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break

            unique_frontier = list(dict.fromkeys(frontier))
            random.shuffle(unique_frontier)

            def score(cell: Coord) -> float:
                return sum(
                    1
                    for dr, dc in _NEIGHBORS4
                    if 0 <= cell[0] + dr < h and 0 <= cell[1] + dc < w and mask[cell[0] + dr, cell[1] + dc]
                ) + random.random()

            chosen = max(unique_frontier, key=score)
            mask[chosen] = True

        if int(mask.sum()) != area:
            continue

        crop = _crop_non_bg(mask.astype(int), 0).astype(bool)
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue
        if int(crop.sum()) >= crop.shape[0] * crop.shape[1]:
            continue
        return crop

    raise RuntimeError("failed to sample connected mask")


def _connected_subset(mask: np.ndarray, size: int) -> np.ndarray:
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    start = random.choice(cells)
    chosen = {start}

    while len(chosen) < size:
        frontier: list[Coord] = []
        for rr, cc in chosen:
            for dr, dc in _NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        if not frontier:
            break
        chosen.add(random.choice(frontier))

    out = np.zeros_like(mask, dtype=bool)
    for rr, cc in chosen:
        out[rr, cc] = True
    return out


def _swap_patch_colors(patch: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    swapped = patch.copy()
    swapped[patch == a] = b
    swapped[patch == b] = a
    return swapped


def _make_object_patch(colors: tuple[int, int]) -> np.ndarray:
    area = random.randint(5, 12)
    mask = _connected_mask(area)
    minority_size = random.randint(2, area - 2)
    minority = _connected_subset(mask, minority_size)

    minority_color, majority_color = colors
    if random.random() < 0.5:
        minority_color, majority_color = majority_color, minority_color

    patch = np.full(mask.shape, _BG, dtype=int)
    patch[mask] = majority_color
    patch[minority] = minority_color

    k = random.randrange(4)
    if k:
        patch = np.rot90(patch, k)
    if random.random() < 0.5:
        patch = np.fliplr(patch)

    return _crop_non_bg(patch, _BG)


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False

    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(grid.shape[0], r0 + h + padding)
    cc1 = min(grid.shape[1], c0 + w + padding)
    return not np.any(grid[rr0:rr1, cc0:cc1] != _BG)


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != _BG
    region[mask] = patch[mask]


def _sample_pair_coords(height: int, width: int, band: str, distance: int) -> Pair:
    if band == "top":
        anchor_r = 1
        offsets = [(0, distance)]
        if distance >= 2:
            offsets.append((-1, distance - 1))
    else:
        anchor_r = height - 2
        offsets = [(0, distance)]
        if distance >= 2:
            offsets.append((1, distance - 1))

    dr, dc = random.choice(offsets)
    if random.random() < 0.5:
        dc = -dc

    c_min = 1 - min(0, dc)
    c_max = width - 2 - max(0, dc)
    anchor_c = random.randint(c_min, c_max)
    return _canonical_pair((anchor_r, anchor_c), (anchor_r + dr, anchor_c + dc))


def _branch_config(branch: int) -> tuple[int, list[int]]:
    if branch == 0:
        return 1, [0] * random.randint(1, 3)
    if branch == 1:
        order = [0, 1]
        random.shuffle(order)
        return 2, order
    if branch == 2:
        count = random.randint(3, 4)
        uses = [0, 1] + [random.randint(0, 1) for _ in range(count - 2)]
        random.shuffle(uses)
        return 2, uses

    pair_count = 1 if random.random() < 0.4 else 2
    if pair_count == 1:
        return 1, [0] * random.randint(2, 4)
    count = random.randint(2, 4)
    uses = [0, 1] + [random.randint(0, 1) for _ in range(count - 2)]
    random.shuffle(uses)
    return 2, uses


def _choose_shape(object_count: int) -> tuple[int, int]:
    if object_count >= 4:
        return random.randint(17, 20), random.randint(19, 24)
    if object_count == 3:
        return random.randint(16, 19), random.randint(18, 23)
    return random.randint(15, 19), random.randint(15, 22)


def _min_distance_to_objects(code_coords: list[Coord], object_coords: list[np.ndarray]) -> int:
    best = 10**9
    for coord in code_coords:
        for arr in object_coords:
            for rr, cc in arr:
                best = min(best, _manhattan(coord, (int(rr), int(cc))))
    return best


def _instance_valid(
    grid: np.ndarray,
    expected: np.ndarray,
    intended_pairs: list[Pair],
) -> bool:
    comps = _find_components(grid, _BG)
    isolated = [tuple(map(int, comp["coords"][0])) for comp in comps if comp["area"] == 1]
    objects = [comp for comp in comps if comp["area"] > 1]

    if len(isolated) not in (2, 4) or len(isolated) % 2 != 0:
        return False
    if not objects:
        return False

    for coord in isolated:
        distances = sorted(_manhattan(coord, other) for other in isolated if other != coord)
        if len(distances) > 1 and distances[0] == distances[1]:
            return False

    actual_pairs = _greedy_pairs(isolated)
    if {frozenset(pair) for pair in actual_pairs} != {frozenset(pair) for pair in intended_pairs}:
        return False

    pair_color_sets = {frozenset((int(grid[a]), int(grid[b]))) for a, b in actual_pairs}
    isolated_colors = {int(grid[r, c]) for r, c in isolated}
    object_colors: set[int] = set()

    object_coords = []
    for comp in objects:
        if not (5 <= comp["area"] <= 12):
            return False
        if len(comp["colors"]) != 2:
            return False
        color_set = frozenset(comp["colors"])
        if color_set not in pair_color_sets:
            return False
        object_colors.update(comp["colors"])
        object_coords.append(comp["coords"])

    if isolated_colors != object_colors:
        return False
    if _min_distance_to_objects(isolated, object_coords) < 3:
        return False
    return np.array_equal(main(grid), expected)


def _generate_attempt(branch: int) -> np.ndarray | None:
    pair_count, object_pair_ids = _branch_config(branch)
    height, width = _choose_shape(len(object_pair_ids))

    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)

    colors = random.sample(list(_NON_BG_COLORS), 2 * pair_count)
    pair_colors = [tuple(colors[i * 2 : i * 2 + 2]) for i in range(pair_count)]

    intended_pairs: list[Pair] = []
    code_specs: list[tuple[Coord, int]] = []
    if pair_count == 1:
        distance = random.choice((2, 3, 4, 5))
        band = random.choice(("top", "bottom"))
        coords = _sample_pair_coords(height, width, band, distance)
        intended_pairs.append(coords)
        for coord, color in zip(coords, pair_colors[0]):
            code_specs.append((coord, color))
    else:
        distances = random.sample((2, 3, 4, 5), 2)
        for band, distance, colors_for_pair in zip(("top", "bottom"), distances, pair_colors):
            coords = _sample_pair_coords(height, width, band, distance)
            intended_pairs.append(coords)
            for coord, color in zip(coords, colors_for_pair):
                code_specs.append((coord, color))

    object_specs: list[tuple[np.ndarray, np.ndarray, tuple[int, int], int, int]] = []
    for pair_id in object_pair_ids:
        patch = _make_object_patch(pair_colors[pair_id])
        swapped = _swap_patch_colors(patch, pair_colors[pair_id])
        ph, pw = patch.shape
        row_min = 4
        row_max = height - ph - 5
        if row_max < row_min:
            return None

        placed = False
        for _ in range(200):
            r0 = random.randint(row_min, row_max)
            c0 = random.randint(1, width - pw - 2)
            if not _can_place(grid, patch, r0, c0, padding=1):
                continue
            _paste_patch(grid, patch, r0, c0)
            _paste_patch(expected, swapped, r0, c0)
            object_specs.append((patch, swapped, pair_colors[pair_id], r0, c0))
            placed = True
            break
        if not placed:
            return None

    for coord, color in code_specs:
        if grid[coord] != _BG:
            return None
        grid[coord] = color

    if not _instance_valid(grid, expected, intended_pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    branch = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1

    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _generate_attempt(branch)
        if grid is not None:
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
