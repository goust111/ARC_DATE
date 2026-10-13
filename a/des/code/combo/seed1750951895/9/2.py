import random
from collections import deque

import numpy as np


CHECKER_MASK = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=int,
)
RING_MASK = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=int,
)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_ORDER = ["up", "right", "down", "left"]
SIDE_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
CONNECTOR_OFFSET = {
    "up": (-1, 1),
    "right": (1, 3),
    "down": (3, 1),
    "left": (1, -1),
}


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_object_window(g: np.ndarray, r: int, c: int, mask: np.ndarray, bg: int) -> bool:
    if r < 0 or c < 0 or r + 3 > g.shape[0] or c + 3 > g.shape[1]:
        return False
    win = g[r : r + 3, c : c + 3]
    fg_vals = {int(v) for v in win[mask == 1]}
    if len(fg_vals) != 1 or bg in fg_vals:
        return False
    return np.array_equal(win != bg, mask.astype(bool))


def _find_windows(g: np.ndarray, mask: np.ndarray, bg: int) -> list[tuple[int, int]]:
    hits = []
    for r in range(g.shape[0] - 2):
        for c in range(g.shape[1] - 2):
            if _is_object_window(g, r, c, mask, bg):
                hits.append((r, c))
    return hits


def _adjacent_sites(top: int, left: int) -> list[tuple[int, int]]:
    sites = []
    for dc in range(3):
        sites.append((top - 1, left + dc))
        sites.append((top + 3, left + dc))
    for dr in range(3):
        sites.append((top + dr, left - 1))
        sites.append((top + dr, left + 3))
    return sites


def _mark_objects(shape: tuple[int, int], positions: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in positions:
        mask[r : r + 3, c : c + 3] = True
    return mask


def _guide_component(
    g: np.ndarray, object_mask: np.ndarray, start: tuple[int, int], color: int
) -> set[tuple[int, int]]:
    assert int(g[start]) == color
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                continue
            if object_mask[nr, nc] or int(g[nr, nc]) != color or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the terminal 3x3 ring reached by repeatedly following same-color guide paths
    from the unique 3x3 checkerboard source.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    checkers = _find_windows(g, CHECKER_MASK, bg)
    rings = _find_windows(g, RING_MASK, bg)
    assert len(checkers) == 1, "expected exactly one checkerboard source"
    assert rings, "expected at least one ring object"

    all_objects = checkers + rings
    object_mask = _mark_objects(g.shape, all_objects)

    current = checkers[0]
    previous_component: set[tuple[int, int]] = set()

    while True:
        guides = []
        for rr, cc in _adjacent_sites(*current):
            if not _in_bounds(g.shape, rr, cc):
                continue
            if object_mask[rr, cc] or int(g[rr, cc]) == bg:
                continue
            if (rr, cc) in previous_component:
                continue
            guides.append((rr, cc))

        if current in rings and not guides:
            r, c = current
            return g[r : r + 3, c : c + 3].copy()

        assert len(guides) == 1, "each traversed object must expose one usable outgoing guide"
        start = guides[0]
        color = int(g[start])
        component = _guide_component(g, object_mask, start, color)

        targets = []
        for ring in rings:
            if ring == current:
                continue
            if any(site in component for site in _adjacent_sites(*ring)):
                targets.append(ring)
        assert len(targets) == 1, "guide segment must terminate at exactly one ring"

        previous_component = component
        current = targets[0]


def _connector(top_left: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = top_left
    dr, dc = CONNECTOR_OFFSET[side]
    return r + dr, c + dc


def _object_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = top_left
    return {(r + dr, c + dc) for dr in range(3) for dc in range(3)}


def _halo_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = top_left
    return {(rr, cc) for rr in range(r - 1, r + 4) for cc in range(c - 1, c + 4)}


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 2 if bc > ac else -2
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 2 if br > ar else -2
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("segment must be axis-aligned")


def _paint_checker(grid: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r, c = top_left
    for dr in range(3):
        for dc in range(3):
            if CHECKER_MASK[dr, dc]:
                grid[r + dr, c + dc] = color


def _paint_ring(grid: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r, c = top_left
    for dr in range(3):
        for dc in range(3):
            if RING_MASK[dr, dc]:
                grid[r + dr, c + dc] = color


def _can_place_object(
    top_left: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    halo_taken: set[tuple[int, int]],
) -> bool:
    r, c = top_left
    if r < 1 or c < 1 or r + 4 >= shape[0] or c + 4 >= shape[1]:
        return False
    cells = _object_cells(top_left)
    halo = _halo_cells(top_left)
    return not (cells & occupied) and not (halo & halo_taken)


def _path_has_clearance(
    cells: list[tuple[int, int]],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    object_cells: set[tuple[int, int]],
    allow_adjacent: set[tuple[int, int]],
) -> bool:
    for idx, cell in enumerate(cells):
        r, c = cell
        if not _in_bounds(shape, r, c) or cell in occupied or cell in object_cells:
            return False
        if idx in (0, len(cells) - 1):
            continue
        for dr, dc in DIRS4:
            nbr = (r + dr, c + dc)
            if nbr in object_cells and nbr not in allow_adjacent:
                return False
    return True


def _add_noise_segments(
    grid: np.ndarray,
    bg: int,
    occupied: set[tuple[int, int]],
    halo_taken: set[tuple[int, int]],
    count: int,
) -> None:
    h, w = grid.shape
    palette = [v for v in range(1, 10) if v != bg]
    for _ in range(count):
        for _ in range(40):
            horizontal = bool(random.getrandbits(1))
            length = random.choice([1, 2, 3])
            color = random.choice(palette)
            if horizontal:
                r = random.randrange(1, h - 1, 2)
                c0 = random.randrange(1, w - 1, 2)
                cells = [(r, c0 + 2 * i) for i in range(length)]
            else:
                r0 = random.randrange(1, h - 1, 2)
                c = random.randrange(1, w - 1, 2)
                cells = [(r0 + 2 * i, c) for i in range(length)]
            if any(not _in_bounds(grid.shape, r, c) for r, c in cells):
                continue
            if any(cell in occupied or cell in halo_taken for cell in cells):
                continue
            for r, c in cells:
                grid[r, c] = color
                occupied.add((r, c))
            break


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = random.randint(19, 29)
    w = random.randint(19, 29)
    grid = np.full((h, w), bg, dtype=int)

    checker_color = random.randint(1, 9)
    ring_colors = random.sample([v for v in range(1, 10) if v != checker_color], random.randint(2, 4))
    distractor_count = random.randint(1, 4)
    path_colors = random.sample(
        [v for v in range(1, 10) if v not in {checker_color, *ring_colors}],
        len(ring_colors),
    )

    occupied: set[tuple[int, int]] = set()
    halo_taken: set[tuple[int, int]] = set()
    object_cells_taken: set[tuple[int, int]] = set()

    even_rows = list(range(2, h - 4, 2))
    even_cols = list(range(2, w - 4, 2))
    if not even_rows or not even_cols:
        return None
    start = (random.choice(even_rows), random.choice(even_cols))
    if not _can_place_object(start, grid.shape, occupied, halo_taken):
        return None

    chain = [start]
    incoming = None
    segment_sides: list[str] = []

    for idx in range(len(ring_colors)):
        current = chain[-1]
        dirs = SIDE_ORDER[:]
        random.shuffle(dirs)
        if incoming is not None:
            rel_groups = {
                "straight": [incoming],
                "turn": [d for d in dirs if d not in (incoming, OPPOSITE[incoming])],
            }
            preferred = "straight" if idx % 2 == 0 else "turn"
            dirs = rel_groups[preferred] + [d for d in dirs if d not in rel_groups[preferred]]

        placed = False
        for side in dirs:
            step_r, step_c = SIDE_DELTA[side]
            distances = [6, 8, 10, 12]
            random.shuffle(distances)
            for dist in distances:
                nxt = (current[0] + step_r * dist, current[1] + step_c * dist)
                if not _can_place_object(nxt, grid.shape, occupied, halo_taken):
                    continue
                cells = _segment_cells(_connector(current, side), _connector(nxt, OPPOSITE[side]))
                allow_adjacent = _object_cells(current) | _object_cells(nxt)
                if not _path_has_clearance(cells, grid.shape, occupied, object_cells_taken, allow_adjacent):
                    continue
                chain.append(nxt)
                segment_sides.append(side)
                incoming = side
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    _paint_checker(grid, chain[0], checker_color)
    for cell in _object_cells(chain[0]):
        occupied.add(cell)
        object_cells_taken.add(cell)
    halo_taken |= _halo_cells(chain[0])

    for top_left, color in zip(chain[1:], ring_colors):
        _paint_ring(grid, top_left, color)
        for cell in _object_cells(top_left):
            occupied.add(cell)
            object_cells_taken.add(cell)
        halo_taken |= _halo_cells(top_left)

    for idx, side in enumerate(segment_sides):
        cells = _segment_cells(_connector(chain[idx], side), _connector(chain[idx + 1], OPPOSITE[side]))
        color = path_colors[idx]
        for r, c in cells:
            grid[r, c] = color
            occupied.add((r, c))

    distractor_palette = [v for v in range(1, 10) if v != checker_color]
    random.shuffle(distractor_palette)
    for i in range(distractor_count):
        color = distractor_palette[i % len(distractor_palette)]
        placed = False
        for _ in range(60):
            top_left = (random.choice(even_rows), random.choice(even_cols))
            if not _can_place_object(top_left, grid.shape, occupied, halo_taken):
                continue
            _paint_ring(grid, top_left, color)
            for cell in _object_cells(top_left):
                occupied.add(cell)
                object_cells_taken.add(cell)
            halo_taken |= _halo_cells(top_left)
            placed = True
            break
        if not placed:
            return None

    _add_noise_segments(grid, bg, occupied, halo_taken, count=random.randint(0, 3))

    expected = grid[chain[-1][0] : chain[-1][0] + 3, chain[-1][1] : chain[-1][1] + 3].copy()
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
