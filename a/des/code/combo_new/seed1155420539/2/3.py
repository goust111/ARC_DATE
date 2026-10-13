from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


GREY = 5
COLOR_COUNT = 10
GENERATION_ATTEMPTS = 600
PAIR_COUNTS = (2, 3)
GRID_RANGE_BY_PAIRS = {
    2: (18, 27),
    3: (24, 31),
}
OUTER_SIDE_RANGE = (5, 7)
PLACEMENT_GAP = 1
SEGMENT_TOLERANCE = 1.25


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(counts.most_common(1)[0][0])


def _dominant_background(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    bg_count = int(counts.get(int(bg), 0))
    return all(bg_count > int(count) for color, count in counts.items() if int(color) != int(bg))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(np.array(cells, dtype=int))

    return comps


def _frame_bbox_from_component(
    grid: np.ndarray, coords: np.ndarray, color: int
) -> tuple[int, int, int, int] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    expected = int(2 * height + 2 * width - 4)
    if int(coords.shape[0]) != expected:
        return None

    local = np.zeros((height, width), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if bool(np.any(interior == int(color))):
        return None
    return r0, c0, r1, c1


def _container_center(container: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, container["bbox"])
    return np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0], dtype=float)


def _detect_containers(grid: np.ndarray, bg: int) -> list[dict]:
    containers: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (int(bg), GREY):
            continue
        for coords in _components(grid == color):
            bbox = _frame_bbox_from_component(grid, coords, color)
            if bbox is not None:
                containers.append({"color": int(color), "bbox": bbox})

    assert containers and len(containers) % 2 == 0, "expected paired rectangular containers"
    containers.sort(key=lambda d: (int(d["color"]), tuple(map(int, d["bbox"]))))

    grouped: dict[int, list[dict]] = defaultdict(list)
    for container in containers:
        grouped[int(container["color"])].append(container)
    assert all(len(items) == 2 for items in grouped.values()), "each border color must occur in exactly two containers"
    return containers


def _extract_pattern(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    interior = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    mask = interior != int(bg)
    assert bool(mask.any()), "each container must contain an internal pattern"
    rows, cols = np.where(mask)
    pr0, pr1 = int(rows.min()), int(rows.max())
    pc0, pc1 = int(cols.min()), int(cols.max())
    return interior[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _clear_interior(out: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)


def _paste_centered(out: np.ndarray, patch: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    ph, pw = map(int, patch.shape)
    assert ph <= inner_h and pw <= inner_w, "rotated pattern must fit in target container"
    top = int(r0 + 1 + (inner_h - ph) // 2)
    left = int(c0 + 1 + (inner_w - pw) // 2)
    mask = patch != int(bg)
    target = out[top : top + ph, left : left + pw]
    target[mask] = patch[mask]


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    vec = b - a
    denom = float(np.dot(vec, vec))
    if denom == 0.0:
        return float("inf"), 0.0
    t = float(np.dot(point - a, vec) / denom)
    nearest = a + t * vec
    return float(np.linalg.norm(point - nearest)), t


def _inside_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int], margin: int = 0) -> bool:
    r, c = map(int, point)
    r0, c0, r1, c1 = map(int, bbox)
    return r0 - margin <= r <= r1 + margin and c0 - margin <= c <= c1 + margin


def _relay_count(grid: np.ndarray, first: dict, second: dict, all_bboxes: list[tuple[int, int, int, int]]) -> int:
    a = _container_center(first)
    b = _container_center(second)
    count = 0
    for r, c in np.argwhere(grid == GREY):
        point = (int(r), int(c))
        if any(_inside_bbox(point, bbox, margin=0) for bbox in all_bboxes):
            continue
        dist, t = _point_segment_distance(np.array(point, dtype=float), a, b)
        if 0.0 < t < 1.0 and dist <= SEGMENT_TOLERANCE:
            count += 1
    assert 0 <= count <= 2, "relay count must be 0, 1, or 2"
    return int(count)


def _solve_with_containers(grid: np.ndarray, containers: list[dict], bg: int) -> np.ndarray:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for container in containers:
        grouped[int(container["color"])].append(container)

    all_bboxes = [tuple(map(int, d["bbox"])) for d in containers]
    out = np.array(grid, dtype=int, copy=True)

    patterns: dict[int, np.ndarray] = {}
    for index, container in enumerate(containers):
        patterns[index] = _extract_pattern(grid, tuple(map(int, container["bbox"])), bg)
        _clear_interior(out, tuple(map(int, container["bbox"])), bg)

    index_by_bbox = {tuple(map(int, d["bbox"])): i for i, d in enumerate(containers)}
    for color in sorted(grouped):
        first, second = grouped[color]
        k = _relay_count(grid, first, second, all_bboxes)
        first_idx = index_by_bbox[tuple(map(int, first["bbox"]))]
        second_idx = index_by_bbox[tuple(map(int, second["bbox"]))]
        first_pat = np.rot90(patterns[first_idx], -k).copy()
        second_pat = np.rot90(patterns[second_idx], -k).copy()
        _paste_centered(out, second_pat, tuple(map(int, first["bbox"])), bg)
        _paste_centered(out, first_pat, tuple(map(int, second["bbox"])), bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap paired rectangular containers' internal patterns, rotating them by the number of grey relay markers between the pair.

    Input structure:
    - The most common color is the background.
    - Several one-pixel-thick rectangular containers sit on the background; each
      container has a colored border and a small non-background pattern inside.
    - Every border color appears on exactly two containers, creating visible
      pairs, and grey single-pixel relay markers may lie between paired centers.

    Rule:
    - Pair the two containers sharing each border color.
    - For each pair, count the grey relay pixels lying on the open segment
      between the two container centers; this count K is 0, 1, or 2.
    - Clear both interiors, rotate each extracted internal pattern clockwise by
      K times 90 degrees, and place the two rotated patterns into the opposite
      container, centered in that interior.
    - Container borders and grey relay markers remain fixed.

    Key insight:
    - Border color gives the container pairing, while the number of grey pixels
      along the pair's connecting line encodes the rotation applied during the
      swap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    containers = _detect_containers(g, bg)
    return _solve_with_containers(g, containers, bg)


# --------------- generator ---------------


BASE_PATTERNS = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
)


def _random_mask() -> np.ndarray:
    mask = random.choice(BASE_PATTERNS).copy()
    mask = np.rot90(mask, random.randrange(4)).copy()
    if random.random() < 0.5:
        mask = np.fliplr(mask).copy()
    return mask


def _make_pattern(color: int, bg: int) -> np.ndarray:
    mask = _random_mask()
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _pattern_key(pattern: np.ndarray, bg: int) -> tuple:
    return tuple(map(int, pattern.ravel())) + tuple(map(int, pattern.shape)) + (int(bg),)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _expanded_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    return max(0, r0 - margin), max(0, c0 - margin), min(h - 1, r1 + margin), min(w - 1, c1 + margin)


def _can_place_bbox(blocked: np.ndarray, bbox: tuple[int, int, int, int], margin: int) -> bool:
    h, w = map(int, blocked.shape)
    r0, c0, r1, c1 = map(int, bbox)
    if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
        return False
    er0, ec0, er1, ec1 = _expanded_bbox(bbox, blocked.shape, margin)
    return not bool(blocked[er0 : er1 + 1, ec0 : ec1 + 1].any())


def _reserve_bbox(blocked: np.ndarray, bbox: tuple[int, int, int, int], margin: int) -> None:
    er0, ec0, er1, ec1 = _expanded_bbox(bbox, blocked.shape, margin)
    blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True


def _place_containers(specs: list[dict], h: int, w: int) -> bool:
    blocked = np.zeros((h, w), dtype=bool)
    order = list(range(len(specs)))
    random.shuffle(order)
    order.sort(key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"], reverse=True)

    for index in order:
        spec = specs[index]
        oh, ow = int(spec["outer_h"]), int(spec["outer_w"])
        candidates = [
            (r, c)
            for r in range(1, h - oh)
            for c in range(1, w - ow)
        ]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates[:250]:
            bbox = (int(r0), int(c0), int(r0 + oh - 1), int(c0 + ow - 1))
            if not _can_place_bbox(blocked, bbox, PLACEMENT_GAP):
                continue
            specs[index]["bbox"] = bbox
            _reserve_bbox(blocked, bbox, PLACEMENT_GAP)
            placed = True
            break
        if not placed:
            return False
    return True


def _candidate_relay_cell(
    target: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    grid: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = map(int, grid.shape)
    choices: list[tuple[float, int, int]] = []
    base_r, base_c = int(round(float(target[0]))), int(round(float(target[1])))
    for radius in range(3):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                r, c = base_r + dr, base_c + dc
                if not (0 <= r < h and 0 <= c < w):
                    continue
                if (r, c) in used or int(grid[r, c]) != int(_most_common_color(grid)):
                    continue
                if any(_inside_bbox((r, c), bbox, margin=1) for bbox in bboxes):
                    continue
                if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in used):
                    continue
                dist, t = _point_segment_distance(np.array([r, c], dtype=float), a, b)
                if 0.0 < t < 1.0 and dist <= SEGMENT_TOLERANCE:
                    choices.append((float(dist + np.linalg.norm(np.array([r, c], dtype=float) - target) * 0.05), r, c))
        if choices:
            choices.sort()
            return int(choices[0][1]), int(choices[0][2])
    return None


def _place_relays(grid: np.ndarray, pairs: list[tuple[dict, dict, int]], bg: int) -> bool:
    bboxes = [tuple(map(int, spec["bbox"])) for pair in pairs for spec in pair[:2]]
    used: set[tuple[int, int]] = set()

    for first, second, k in pairs:
        if int(k) == 0:
            continue
        a = _container_center(first)
        b = _container_center(second)
        pair_cells: list[tuple[int, int]] = []
        for i in range(1, int(k) + 1):
            target = a + (b - a) * (i / float(k + 1))
            cell = _candidate_relay_cell(target, a, b, grid, bboxes, used)
            if cell is None:
                return False
            pair_cells.append(cell)
            used.add(cell)
        for r, c in pair_cells:
            if int(grid[r, c]) != int(bg):
                return False
            grid[r, c] = GREY

    all_bboxes = [tuple(map(int, spec["bbox"])) for pair in pairs for spec in pair[:2]]
    for first, second, k in pairs:
        try:
            actual = _relay_count(grid, first, second, all_bboxes)
        except AssertionError:
            return False
        if actual != int(k):
            return False
    return True


def _sample_specs(pair_count: int, bg: int) -> tuple[list[dict], list[tuple[dict, dict, int]]] | None:
    palette = [c for c in range(COLOR_COUNT) if c not in (int(bg), GREY)]
    if len(palette) < int(pair_count):
        return None
    random.shuffle(palette)
    border_colors = palette[:pair_count]
    pattern_pool = [c for c in palette[pair_count:] if c not in border_colors]
    if not pattern_pool:
        return None

    if pair_count == 3:
        relay_counts = [0, 1, 2]
        random.shuffle(relay_counts)
    else:
        relay_counts = random.sample([0, 1, 2], 2)

    specs: list[dict] = []
    pairs: list[tuple[dict, dict, int]] = []
    for pair_index, border in enumerate(border_colors):
        pair_specs: list[dict] = []
        pair_keys: set[tuple] = set()
        for _ in range(2):
            for _attempt in range(100):
                outer_h = random.randint(*OUTER_SIDE_RANGE)
                outer_w = random.randint(*OUTER_SIDE_RANGE)
                pat_color = random.choice(pattern_pool)
                pattern = _make_pattern(pat_color, bg)
                ph, pw = map(int, pattern.shape)
                if ph > outer_h - 2 or pw > outer_w - 2:
                    continue
                key = _pattern_key(pattern, bg)
                if key in pair_keys:
                    continue
                pair_keys.add(key)
                spec = {
                    "color": int(border),
                    "outer_h": int(outer_h),
                    "outer_w": int(outer_w),
                    "pattern": pattern,
                }
                pair_specs.append(spec)
                specs.append(spec)
                break
            else:
                return None
        pairs.append((pair_specs[0], pair_specs[1], int(relay_counts[pair_index])))
    random.shuffle(specs)
    return specs, pairs


def _draw_input_grid(
    h: int, w: int, specs: list[dict], pairs: list[tuple[dict, dict, int]], bg: int
) -> np.ndarray | None:
    if not _place_containers(specs, h, w):
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    for spec in specs:
        bbox = tuple(map(int, spec["bbox"]))
        _draw_frame(grid, bbox, int(spec["color"]))
        _paste_centered(grid, spec["pattern"], bbox, bg)

    if not _place_relays(grid, pairs, bg):
        return None
    return grid


def _expected_from_specs(grid: np.ndarray, specs: list[dict], pairs: list[tuple[dict, dict, int]], bg: int) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        _clear_interior(out, tuple(map(int, spec["bbox"])), bg)

    for first, second, k in pairs:
        first_pattern = _extract_pattern(grid, tuple(map(int, first["bbox"])), bg)
        second_pattern = _extract_pattern(grid, tuple(map(int, second["bbox"])), bg)
        _paste_centered(out, np.rot90(second_pattern, -int(k)).copy(), tuple(map(int, first["bbox"])), bg)
        _paste_centered(out, np.rot90(first_pattern, -int(k)).copy(), tuple(map(int, second["bbox"])), bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNTS)
    lo, hi = GRID_RANGE_BY_PAIRS[pair_count]
    h = random.randint(lo, hi - 1)
    w = random.randint(lo, hi - 1)
    bg = random.choice([c for c in range(COLOR_COUNT) if c != GREY])

    sampled = _sample_specs(pair_count, bg)
    if sampled is None:
        return None
    specs, pairs = sampled

    grid = _draw_input_grid(h, w, specs, pairs, bg)
    if grid is None:
        return None
    if not _dominant_background(grid, bg):
        return None

    expected = _expected_from_specs(grid, specs, pairs, bg)
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(produced, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for seed1155420539/2/3")
