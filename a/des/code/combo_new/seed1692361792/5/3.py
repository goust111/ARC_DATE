import random

import numpy as np


_DIR_TO_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_DIR_TO_PATTERN = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}
_PATTERN_TO_DIR = {pattern: direction for direction, pattern in _DIR_TO_PATTERN.items()}

_BG = 0
_COLORS = tuple(range(1, 10))
_GRID_MIN = 12
_GRID_MAX = 24
_OBJECT_COUNT_MIN = 2
_OBJECT_COUNT_MAX = 5
_MAX_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        stack = [(r, c)]
        visited[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in _DIR_TO_STEP.values():
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_direction(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int]]:
    assert len(coords) == 5, "arrow must contain five pixels"
    r0, c0, r1, c1 = _bbox(coords)
    assert (r1 - r0, c1 - c0) == (2, 2), "arrow must fit in a 3x3 box"
    norm = frozenset((r - r0, c - c0) for r, c in coords)
    assert norm in _PATTERN_TO_DIR, "unknown arrow shape"
    return _PATTERN_TO_DIR[norm], (r0, c0)


def _square_top_left(coords: list[tuple[int, int]]) -> tuple[int, int]:
    assert len(coords) == 4, "square must contain four pixels"
    r0, c0, r1, c1 = _bbox(coords)
    assert (r1 - r0, c1 - c0) == (1, 1), "square must be 2x2"
    expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    assert set(coords) == expected, "square component must be solid 2x2"
    return r0, c0


def _slide_square(
    occupied: np.ndarray, start_r: int, start_c: int, dr: int, dc: int
) -> tuple[int, int, str, int]:
    h, w = occupied.shape
    r = int(start_r)
    c = int(start_c)
    steps = 0
    while True:
        nr = r + dr
        nc = c + dc
        if nr < 0 or nc < 0 or nr + 1 >= h or nc + 1 >= w:
            return r, c, "boundary", steps
        if bool(occupied[nr : nr + 2, nc : nc + 2].any()):
            return r, c, "block", steps
        r = nr
        c = nc
        steps += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored 2x2 block along the direction encoded by its same-color arrow, then erase the arrows.

    Input structure:
    - The background is black and the non-background colors are all distinct instruction/object pairs.
    - Each color appears in exactly two disconnected 4-connected components: one 5-pixel arrow and one solid 2x2 square.
    - Every arrow is a 3-pixel horizontal or vertical shaft with two side pixels at the pointed end, so the whole arrow fits inside a 3x3 box.

    Rule:
    - Decode every arrow's direction from its 5-pixel shape and match it to the same-color 2x2 square.
    - Process arrows in row-major scan order by their top-left 3x3 bounding-box corner.
    - Move each matched square step by step in the arrow direction on an otherwise empty output canvas, stopping only at the grid boundary or a square placed earlier in this order.

    Key insight:
    - The arrows are command glyphs rather than output content: color links each command to one square, and only the scan-order sequence creates the blocking interactions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    records: list[tuple[int, int, str, int, int, int]] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        components = _components_of_color(g, color)
        assert len(components) == 2, "each non-background color must form one arrow and one square"

        arrow = None
        square = None
        for coords in components:
            if len(coords) == 5:
                arrow = coords
            elif len(coords) == 4:
                square = coords
            else:
                raise AssertionError("unexpected component size")
        assert arrow is not None and square is not None, "missing arrow or square for a color"

        direction, (arrow_r, arrow_c) = _arrow_direction(arrow)
        square_r, square_c = _square_top_left(square)
        records.append((arrow_r, arrow_c, direction, color, square_r, square_c))

    records.sort(key=lambda item: (item[0], item[1]))

    out = np.full_like(g, bg)
    occupied = np.zeros(g.shape, dtype=bool)
    for _arrow_r, _arrow_c, direction, color, square_r, square_c in records:
        dr, dc = _DIR_TO_STEP[direction]
        final_r, final_c, _reason, _steps = _slide_square(occupied, square_r, square_c, dr, dc)
        occupied[final_r : final_r + 2, final_c : final_c + 2] = True
        out[final_r : final_r + 2, final_c : final_c + 2] = color

    return out


def _box_clear(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(reserved[r0:r1, c0:c1].any())


def _reserve_box(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    reserved[top : top + height, left : left + width] = True


def _square_candidates(
    reserved: np.ndarray, occupied: np.ndarray, direction: str
) -> list[tuple[int, int, int, int, str, int]]:
    h, w = reserved.shape
    dr, dc = _DIR_TO_STEP[direction]
    candidates: list[tuple[int, int, int, int, str, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if not _box_clear(reserved, r, c, 2, 2, pad=1):
                continue
            final_r, final_c, reason, steps = _slide_square(occupied, r, c, dr, dc)
            if steps == 0:
                continue
            candidates.append((r, c, final_r, final_c, reason, steps))
    return candidates


def _choose_direction(
    rng: random.Random,
    candidates_by_dir: dict[str, list[tuple[int, int, int, int, str, int]]],
    used_counts: dict[str, int],
) -> str:
    min_count = min(used_counts[direction] for direction in candidates_by_dir)
    choices = [direction for direction in candidates_by_dir if used_counts[direction] == min_count]
    return rng.choice(choices)


def _choose_candidate(
    rng: random.Random, candidates: list[tuple[int, int, int, int, str, int]]
) -> tuple[int, int, int, int, str, int]:
    long_moves = [candidate for candidate in candidates if candidate[5] >= 2]
    pool = long_moves or candidates
    return rng.choice(pool)


def _arrow_positions(reserved: np.ndarray, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    local_reserved = reserved.copy()
    positions: list[tuple[int, int]] = []
    for _ in range(count):
        h, w = local_reserved.shape
        candidates = []
        for r in range(h - 2):
            for c in range(w - 2):
                if _box_clear(local_reserved, r, c, 3, 3, pad=1):
                    candidates.append((r, c))
        if not candidates:
            return None
        pos = rng.choice(candidates)
        positions.append(pos)
        _reserve_box(local_reserved, pos[0], pos[1], 3, 3)
    positions.sort()
    return positions


def _paint_arrow(grid: np.ndarray, top: int, left: int, direction: str, color: int) -> None:
    for dr, dc in _DIR_TO_PATTERN[direction]:
        grid[top + dr, left + dc] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(_GRID_MIN, _GRID_MAX)
    width = rng.randint(_GRID_MIN, _GRID_MAX)
    count = rng.randint(_OBJECT_COUNT_MIN, _OBJECT_COUNT_MAX)
    colors = rng.sample(_COLORS, count)

    grid = np.full((height, width), _BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    occupied = np.zeros((height, width), dtype=bool)
    expected = np.full((height, width), _BG, dtype=int)

    records: list[dict[str, object]] = []
    used_counts = {direction: 0 for direction in _DIR_TO_STEP}
    must_block = count >= 2 and rng.random() < 0.5
    block_index = rng.randrange(1, count) if must_block else -1
    block_seen = False

    for index, color in enumerate(colors):
        require_block_here = index == block_index and not block_seen
        candidates_by_dir: dict[str, list[tuple[int, int, int, int, str, int]]] = {}
        for direction in _DIR_TO_STEP:
            candidates = _square_candidates(reserved, occupied, direction)
            if require_block_here:
                candidates = [candidate for candidate in candidates if candidate[4] == "block"]
            if candidates:
                candidates_by_dir[direction] = candidates
        if not candidates_by_dir:
            return None

        direction = _choose_direction(rng, candidates_by_dir, used_counts)
        start_r, start_c, final_r, final_c, reason, steps = _choose_candidate(rng, candidates_by_dir[direction])

        _reserve_box(reserved, start_r, start_c, 2, 2)
        grid[start_r : start_r + 2, start_c : start_c + 2] = color
        occupied[final_r : final_r + 2, final_c : final_c + 2] = True
        expected[final_r : final_r + 2, final_c : final_c + 2] = color
        used_counts[direction] += 1
        if reason == "block":
            block_seen = True

        records.append(
            {
                "color": color,
                "direction": direction,
                "square": (start_r, start_c),
                "final": (final_r, final_c),
                "reason": reason,
                "steps": steps,
            }
        )

    if must_block and not block_seen:
        return None

    arrow_positions = _arrow_positions(reserved, count, rng)
    if arrow_positions is None:
        return None

    for record, (arrow_r, arrow_c) in zip(records, arrow_positions):
        _paint_arrow(grid, arrow_r, arrow_c, str(record["direction"]), int(record["color"]))

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the arrow-slide task")
