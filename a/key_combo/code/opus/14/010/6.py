import numpy as np


BG = 0
SPRITE = 1
MARKER = 2

GRID_MIN = 20
GRID_MAX_EXCLUSIVE = 29
MARKER_COUNT_CHOICES = (3, 4, 5)
PAIR_DISTANCE_MIN = 4
CURRENT_DISTANCE_MIN = 4
CURRENT_DISTANCE_MAX = 10
EARLIER_DISTANCE_MARGIN = 1
MAX_GENERATION_ATTEMPTS = 400

# All templates use an odd-sized bbox so the bbox center is an integer cell.
SPRITE_TEMPLATES = (
    np.array([(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)], dtype=int),   # plus
    np.array([(0, -1), (0, 0), (0, 1)], dtype=int),                     # horizontal line
    np.array([(-1, 0), (0, 0), (1, 0)], dtype=int),                     # vertical line
    np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1)], dtype=int),
    np.array([(-1, -1), (-1, 0), (0, 0), (0, 1), (1, 1)], dtype=int),
    np.array([(-1, 1), (0, -1), (0, 0), (0, 1), (1, -1)], dtype=int),
)


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _in_bbox(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _bbox_point_distance(box: tuple[int, int, int, int], point: tuple[int, int]) -> int:
    r, c = point
    r0, c0, r1, c1 = box
    if r < r0:
        dr = r0 - r
    elif r > r1:
        dr = r - r1
    else:
        dr = 0
    if c < c0:
        dc = c0 - c
    elif c > c1:
        dc = c - c1
    else:
        dc = 0
    return dr + dc


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _min_distance_to_coords(point: tuple[int, int], coords: set[tuple[int, int]]) -> int:
    return min(_manhattan(point, cell) for cell in coords)


def _shift(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _all_in_bounds(coords: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _expand_once(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = shape
    expanded = set(coords)
    for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            expanded.add((r, c))
    return expanded


def _rank_markers(
    box: tuple[int, int, int, int], markers: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]]:
    ranked = []
    for idx, point in enumerate(markers):
        dist = _bbox_point_distance(box, point)
        ranked.append((dist, point[0], point[1], idx))
    ranked.sort()
    return ranked


def _simulate(
    sprite_coords: set[tuple[int, int]],
    markers: list[tuple[int, int]],
    shape: tuple[int, int],
    *,
    require_unique: bool,
    require_no_accidental_cover: bool,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    coords = set(sprite_coords)
    remaining = list(markers)
    order: list[tuple[int, int]] = []

    while remaining:
        box = _bbox(coords)
        ranked = _rank_markers(box, remaining)
        if require_unique and len(ranked) >= 2 and ranked[0][0] == ranked[1][0]:
            raise ValueError("nearest marker is not unique")

        _, _, _, pick_idx = ranked[0]
        target = remaining.pop(pick_idx)
        order.append(target)

        center_r, center_c = _bbox_center(box)
        moved = _shift(coords, target[0] - center_r, target[1] - center_c)
        if not _all_in_bounds(moved, shape):
            raise ValueError("moved sprite left the grid")

        moved_box = _bbox(moved)
        if require_no_accidental_cover and any(_in_bbox(p, moved_box) for p in remaining):
            raise ValueError("movement bbox covered an unconsumed marker")

        expanded = _expand_once(moved, shape)
        expanded_box = _bbox(expanded)
        if require_no_accidental_cover and any(_in_bbox(p, expanded_box) for p in remaining):
            raise ValueError("expanded bbox covered an unconsumed marker")

        coords = expanded

    return coords, order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the blue sprite to the nearest red marker, consuming it and expanding the sprite one bbox-ring each time.

    Input structure:
    - A black background contains one blue 4-connected sprite and several isolated red single-pixel markers.
    - The sprite is monochrome, the markers are monochrome, and all coordinates use the grid's row/column geometry.

    Rule:
    - Repeatedly compute each remaining marker's Manhattan distance to the sprite's current bounding box and pick the unique nearest marker.
    - Translate the whole sprite so the bbox center lands on that marker, consume the marker, then expand the moved bbox by one cell on every side.
    - Every cell in that new outer ring becomes blue by copying the nearest bbox-edge sprite pixel; repeat until no red markers remain.

    Key insight:
    - The greedy target order depends on the sprite's current bbox size, and each consumption enlarges that bbox before the next distance comparison.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprite_coords = {tuple(map(int, rc)) for rc in np.argwhere(g == SPRITE)}
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]

    assert sprite_coords, "expected a blue sprite"
    final_coords, _ = _simulate(
        sprite_coords,
        markers,
        g.shape,
        require_unique=False,
        require_no_accidental_cover=False,
    )

    out = np.zeros_like(g)
    for r, c in final_coords:
        out[r, c] = SPRITE
    return out


def _template_cells(template: np.ndarray, center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + int(dr), cc + int(dc)) for dr, dc in template}


def _candidate_next_state(
    current: set[tuple[int, int]], target: tuple[int, int], shape: tuple[int, int]
) -> set[tuple[int, int]] | None:
    box = _bbox(current)
    center_r, center_c = _bbox_center(box)
    moved = _shift(current, target[0] - center_r, target[1] - center_c)
    if not _all_in_bounds(moved, shape):
        return None
    return _expand_once(moved, shape)


def _valid_initial_distances(
    initial_sprite: set[tuple[int, int]], markers: list[tuple[int, int]]
) -> bool:
    if not (3 <= len(markers) <= 5):
        return False
    for i, point in enumerate(markers):
        if _min_distance_to_coords(point, initial_sprite) < PAIR_DISTANCE_MIN:
            return False
        for other in markers[i + 1 :]:
            if _manhattan(point, other) < PAIR_DISTANCE_MIN:
                return False
    return True


def _build_candidate_list(
    initial_sprite: set[tuple[int, int]],
    states: list[set[tuple[int, int]]],
    chosen_markers: list[tuple[int, int]],
    chosen_distances: list[int],
    shape: tuple[int, int],
) -> list[tuple[tuple[int, int], set[tuple[int, int]], int]]:
    h, w = shape
    current = states[-1]
    current_box = _bbox(current)
    current_center = _bbox_center(current_box)
    state_boxes = [_bbox(state) for state in states]
    candidates: list[tuple[tuple[int, int], set[tuple[int, int]], int]] = []

    for r in range(h):
        for c in range(w):
            point = (r, c)
            if point in chosen_markers:
                continue
            if _min_distance_to_coords(point, initial_sprite) < PAIR_DISTANCE_MIN:
                continue
            if any(_manhattan(point, other) < PAIR_DISTANCE_MIN for other in chosen_markers):
                continue

            dist = _bbox_point_distance(current_box, point)
            if dist < CURRENT_DISTANCE_MIN or dist > CURRENT_DISTANCE_MAX:
                continue
            if any(_in_bbox(point, box) for box in state_boxes):
                continue

            earlier_ok = True
            for box, chosen_dist in zip(state_boxes[:-1], chosen_distances):
                if _bbox_point_distance(box, point) <= chosen_dist + EARLIER_DISTANCE_MARGIN:
                    earlier_ok = False
                    break
            if not earlier_ok:
                continue

            next_state = _candidate_next_state(current, point, shape)
            if next_state is None:
                continue

            next_box = _bbox(next_state)
            if any(_in_bbox(other, next_box) for other in chosen_markers):
                continue

            # Prefer varied directions around the current center, then varied distances.
            direction_key = (
                int(point[0] > current_center[0]) - int(point[0] < current_center[0]),
                int(point[1] > current_center[1]) - int(point[1] < current_center[1]),
            )
            score = (
                abs(direction_key[0]) + abs(direction_key[1]),
                -abs(point[0] - current_center[0] - (point[1] - current_center[1])),
                -dist,
            )
            candidates.append((point, next_state, score))

    return candidates


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX_EXCLUSIVE))
    w = int(rng.integers(GRID_MIN, GRID_MAX_EXCLUSIVE))
    shape = (h, w)
    marker_count = int(rng.choice(MARKER_COUNT_CHOICES))
    template = SPRITE_TEMPLATES[int(rng.integers(0, len(SPRITE_TEMPLATES)))]

    center = (int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))
    initial_sprite = _template_cells(template, center)
    if not _all_in_bounds(initial_sprite, shape):
        return None

    states = [initial_sprite]
    chosen_markers: list[tuple[int, int]] = []
    chosen_distances: list[int] = []

    for _ in range(marker_count):
        candidates = _build_candidate_list(
            initial_sprite,
            states,
            chosen_markers,
            chosen_distances,
            shape,
        )
        if not candidates:
            return None

        rng.shuffle(candidates)
        candidates.sort(key=lambda item: item[2], reverse=True)
        top = candidates[: min(12, len(candidates))]
        point, next_state, _ = top[int(rng.integers(0, len(top)))]

        chosen_markers.append(point)
        chosen_distances.append(_bbox_point_distance(_bbox(states[-1]), point))
        states.append(next_state)

    if not _valid_initial_distances(initial_sprite, chosen_markers):
        return None

    grid = np.zeros(shape, dtype=int)
    for r, c in initial_sprite:
        grid[r, c] = SPRITE
    for r, c in chosen_markers:
        grid[r, c] = MARKER

    try:
        final_coords, order = _simulate(
            initial_sprite,
            chosen_markers,
            shape,
            require_unique=True,
            require_no_accidental_cover=True,
        )
    except ValueError:
        return None

    if len(order) != marker_count:
        return None

    expected = np.zeros_like(grid)
    for r, c in final_coords:
        expected[r, c] = SPRITE
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
