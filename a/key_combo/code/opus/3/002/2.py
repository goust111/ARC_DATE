import itertools
import random
from collections import deque

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
QUADRANT_COLORS = {
    (0, 0): RED,
    (0, 1): GREEN,
    (1, 0): YELLOW,
    (1, 1): BLUE,
}
PAIR_ORDER = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
QUADRANT_TO_PAIR_IDS = {
    0: (0, 1, 2),
    1: (0, 3, 4),
    2: (1, 3, 5),
    3: (2, 4, 5),
}

QUADRANT_SIDE_RANGE = (8, 15)
BOX_SIZE_OPTIONS = (6, 7)
MASK_AREA_RANGE = (8, 20)
GEN_ATTEMPTS = 500


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _colors_present(grid: np.ndarray, exclude=()) -> list[int]:
    blocked = set(int(x) for x in exclude)
    return sorted(int(x) for x in np.unique(grid) if int(x) not in blocked)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                cell = (int(rr), int(cc))
                if cell not in seen:
                    seen.add(cell)
                    queue.append(cell)
    return len(seen) == int(mask.sum())


def _mask_from_coords(side: int, coords: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _analyze_quadrant(region: np.ndarray) -> dict:
    colors = _colors_present(region, exclude=[BACKGROUND, GREY])
    assert len(colors) == 1, "each quadrant must contain exactly one mask color"
    color = colors[0]
    mask = region == color
    marker_count = int(np.count_nonzero(region == GREY))
    assert 1 <= marker_count <= 4, "marker counts must stay in 1..4"
    assert MASK_AREA_RANGE[0] <= int(mask.sum()) <= MASK_AREA_RANGE[1], "mask area out of range"
    assert _is_connected(mask), "quadrant mask must be 4-connected"
    return {"color": color, "mask": mask, "count": marker_count}


def _split_quadrants(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "input must be a square grid"
    side = int(g.shape[0])
    assert side % 2 == 0 and side > 0, "input side must be a positive even number"
    q = side // 2
    parts: list[dict] = []
    for qr, qc in QUADRANTS:
        region = g[qr * q : (qr + 1) * q, qc * q : (qc + 1) * q]
        info = _analyze_quadrant(region)
        info["quadrant"] = (qr, qc)
        parts.append(info)
    counts = [part["count"] for part in parts]
    assert len(set(counts)) == 4, "quadrant marker counts must be distinct"
    return q, parts


def _overlay(parts: list[dict], order: tuple[int, ...], q: int) -> np.ndarray:
    out = np.zeros((q, q), dtype=int)
    for idx in order:
        part = parts[int(idx)]
        out[part["mask"]] = int(part["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the four quadrant masks on the shared half-size canvas in ascending grey-marker count order.

    Input structure:
    - The input is a square black grid split into four equal quadrants.
    - Each quadrant contains one 4-connected monochrome mask in a distinct non-grey color.
    - Each quadrant also contains 1-4 grey marker pixels placed on that quadrant's background cells.

    Rule:
    - Count the grey marker pixels in each quadrant.
    - Sort the quadrants from fewest markers to most markers.
    - Fold the four colored masks onto one quadrant-sized black canvas and stamp them in that sorted order, with later masks overwriting earlier ones.

    Key insight:
    - The grey pixels are invisible controls: they disappear from the output, but their counts encode the mask layer stack.
    """
    q, parts = _split_quadrants(np.asarray(grid, dtype=int))
    order = tuple(sorted(range(4), key=lambda idx: parts[idx]["count"]))
    return _overlay(parts, order, q)


def _local_corner_clusters(box_size: int) -> dict[int, set[tuple[int, int]]]:
    s = int(box_size)
    return {
        0: {(0, 0), (0, 1), (1, 0)},
        1: {(0, s - 1), (0, s - 2), (1, s - 1)},
        2: {(s - 1, 0), (s - 2, 0), (s - 1, 1)},
        3: {(s - 1, s - 1), (s - 2, s - 1), (s - 1, s - 2)},
    }


def _local_pair_cells(box_size: int) -> dict[int, tuple[int, int]]:
    s = int(box_size)
    mid0 = s // 2 - 1
    mid1 = s // 2
    return {
        0: (1, mid0),       # pair (0,1): top center
        1: (mid0, 1),       # pair (0,2): left center
        2: (mid0, mid0),    # pair (0,3): main diagonal center
        3: (mid0, mid1),    # pair (1,2): anti-diagonal center
        4: (mid1, s - 2),   # pair (1,3): right center
        5: (s - 2, mid1),   # pair (2,3): bottom center
    }


def _translate(cells: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(int(r + top), int(c + left)) for r, c in cells}


def _random_shortest_path(
    side: int,
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if start in goals:
        return [start]

    queue = deque([start])
    parent = {start: None}
    goal_hit = None

    while queue and goal_hit is None:
        cell = queue.popleft()
        nbrs = list(neighbors4(*cell))
        random.shuffle(nbrs)
        for rr, cc in nbrs:
            nxt = (int(rr), int(cc))
            if nxt in parent:
                continue
            if not (0 <= rr < side and 0 <= cc < side):
                continue
            if nxt not in allowed:
                continue
            if nxt in blocked and nxt not in goals:
                continue
            parent[nxt] = cell
            if nxt in goals:
                goal_hit = nxt
                break
            queue.append(nxt)

    if goal_hit is None:
        return None

    path = [goal_hit]
    cur = goal_hit
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _connect_required_points(
    side: int,
    start_cells: set[tuple[int, int]],
    targets: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    shape = set(start_cells)
    pending = set(targets)
    while pending:
        choices: list[tuple[int, tuple[int, int]]] = []
        for target in pending:
            dist = min(abs(target[0] - cell[0]) + abs(target[1] - cell[1]) for cell in shape)
            choices.append((dist, target))
        min_dist = min(dist for dist, _ in choices)
        target = random.choice([cell for dist, cell in choices if dist == min_dist])
        path = _random_shortest_path(side, target, shape, allowed, blocked - shape)
        if path is None:
            return None
        shape.update(path)
        pending.remove(target)
    return shape


def _grow_irregular_tail(
    side: int,
    shape: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    min_area: int,
    max_area: int,
) -> set[tuple[int, int]] | None:
    target_area = random.randint(max(int(min_area), len(shape)), min(int(max_area), len(shape) + 4))
    while len(shape) < target_area:
        frontier = []
        for r, c in shape:
            for rr, cc in neighbors4(r, c):
                cell = (int(rr), int(cc))
                if not (0 <= rr < side and 0 <= cc < side):
                    continue
                if cell not in allowed or cell in blocked or cell in shape:
                    continue
                frontier.append(cell)
        if not frontier:
            return None
        weights = []
        center = (side - 1) / 2.0
        for r, c in frontier:
            weights.append(1.0 + 0.15 * (abs(r - center) + abs(c - center)))
        shape.add(random.choices(frontier, weights=weights, k=1)[0])
    return shape


def _build_masks(q: int) -> dict[int, np.ndarray] | None:
    size_choices = [s for s in BOX_SIZE_OPTIONS if s <= q - 2]
    if not size_choices:
        return None
    box_size = random.choice(size_choices)
    top = random.randint(1, q - box_size - 1)
    left = random.randint(1, q - box_size - 1)

    corner_clusters = {
        idx: _translate(cells, top, left)
        for idx, cells in _local_corner_clusters(box_size).items()
    }
    pair_cells = {
        pair_id: tuple(np.array(cell) + np.array((top, left)))
        for pair_id, cell in _local_pair_cells(box_size).items()
    }
    allowed = {
        (r, c)
        for r in range(top, top + box_size)
        for c in range(left, left + box_size)
    }
    all_pair_cells = set(pair_cells.values())

    masks: dict[int, set[tuple[int, int]]] = {}
    occupied: set[tuple[int, int]] = set()
    build_order = list(range(4))
    random.shuffle(build_order)

    for idx in build_order:
        required_pairs = {pair_cells[pair_id] for pair_id in QUADRANT_TO_PAIR_IDS[idx]}
        shape = set(corner_clusters[idx])
        forbidden_pair_cells = all_pair_cells - required_pairs
        blocked = (occupied - required_pairs) | forbidden_pair_cells
        shape = _connect_required_points(q, shape, required_pairs, allowed, blocked)
        if shape is None:
            return None
        if len(shape) > MASK_AREA_RANGE[1]:
            return None
        shape = _grow_irregular_tail(
            q,
            shape,
            allowed,
            blocked,
            min_area=MASK_AREA_RANGE[0],
            max_area=MASK_AREA_RANGE[1],
        )
        if shape is None or len(shape) > MASK_AREA_RANGE[1]:
            return None
        masks[idx] = shape
        occupied.update(shape)

    out = {idx: _mask_from_coords(q, cells) for idx, cells in masks.items()}
    if not all(_is_connected(mask) for mask in out.values()):
        return None
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
                out[rr, cc] = True
    return out


def _place_markers(mask: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    side = int(mask.shape[0])
    blocked = _dilate(mask)
    candidates = [(r, c) for r in range(side) for c in range(side) if not blocked[r, c]]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - rr) + abs(c - cc) >= 2 for rr, cc in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) == int(count):
                return chosen
    return None


def _render_input(masks: dict[int, np.ndarray], counts: dict[int, int]) -> np.ndarray | None:
    q = int(next(iter(masks.values())).shape[0])
    grid = np.zeros((2 * q, 2 * q), dtype=int)
    for idx, (qr, qc) in enumerate(QUADRANTS):
        region = grid[qr * q : (qr + 1) * q, qc * q : (qc + 1) * q]
        region[masks[idx]] = QUADRANT_COLORS[(qr, qc)]
        markers = _place_markers(masks[idx], counts[idx])
        if markers is None:
            return None
        for r, c in markers:
            region[r, c] = GREY
    return grid


def _pairwise_overlap_count(masks: dict[int, np.ndarray]) -> int:
    pairs = 0
    for a, b in PAIR_ORDER:
        if np.any(masks[a] & masks[b]):
            pairs += 1
    return pairs


def _semantic_signature(parts: list[dict], q: int) -> tuple[int, tuple[bytes, ...]]:
    outputs = []
    for order in itertools.permutations(range(4)):
        out = _overlay(parts, order, q)
        outputs.append(out.tobytes())
    actual_order = tuple(sorted(range(4), key=lambda idx: parts[idx]["count"]))
    actual = _overlay(parts, actual_order, q).tobytes()
    return sum(sig == actual for sig in outputs), tuple(outputs)


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        q, parts = _split_quadrants(grid)
    except AssertionError:
        return False

    actual_order = tuple(sorted(range(4), key=lambda idx: parts[idx]["count"]))
    expected = _overlay(parts, actual_order, q)
    if not np.array_equal(main(grid), expected):
        return False

    colors = {part["color"] for part in parts}
    if colors != {BLUE, RED, GREEN, YELLOW}:
        return False
    if sorted(part["count"] for part in parts) != [1, 2, 3, 4]:
        return False

    visible_colors = set(int(x) for x in np.unique(expected) if int(x) != BACKGROUND)
    if visible_colors != colors:
        return False

    masks = {idx: part["mask"] for idx, part in enumerate(parts)}
    if _pairwise_overlap_count(masks) < 2:
        return False

    equal_count, _ = _semantic_signature(parts, q)
    if equal_count != 1:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    q = random.randint(*QUADRANT_SIDE_RANGE)
    masks = _build_masks(q)
    if masks is None:
        return None

    counts_perm = random.sample([1, 2, 3, 4], 4)
    counts = {idx: counts_perm[idx] for idx in range(4)}
    grid = _render_input(masks, counts)
    if grid is None:
        return None
    if _validate_generated(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-overlay ARC input")
