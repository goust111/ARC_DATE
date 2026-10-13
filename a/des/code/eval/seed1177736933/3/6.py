import random

import numpy as np


BG = 0
OBJ_SIZE = 3
ARROW_SIZE = 3
HEIGHT_RANGE = (15, 21)
WIDTH_RANGE = (16, 22)  # includes one blank spacer column and the sequence column
NUM_ARROWS_RANGE = (3, 4)
MAX_DISTANCE = 4
GEN_ATTEMPTS = 600
PALETTE = tuple(range(1, 10))
DIRECTIONS = ("up", "down", "left", "right")
HORIZONTAL_DIRS = {"left", "right"}
VERTICAL_DIRS = {"up", "down"}
OFFSET_CHOICES = (-2, -1, 0, 1, 2)

DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}

OBJECT_BASE_MASKS = (
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
)


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR_VECS.values():
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append(comp)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _arrow_direction(coords: list[tuple[int, int]]) -> str:
    assert len(coords) == 5, "arrow must contain exactly 5 cells"
    norm = _normalize(coords)
    for direction, pattern in ARROW_PATTERNS.items():
        if norm == pattern:
            return direction
    raise AssertionError("unrecognized arrow shape")


def _movement_distance(
    object_bbox: tuple[int, int, int, int],
    arrow_bbox: tuple[int, int, int, int],
    direction: str,
) -> int:
    or0, oc0, or1, oc1 = object_bbox
    ar0, ac0, ar1, ac1 = arrow_bbox
    if direction == "right":
        assert _intervals_overlap(or0, or1, ar0, ar1), "right arrow must overlap the object's rows"
        dist = int(oc0 - ac1 - 1)
    elif direction == "left":
        assert _intervals_overlap(or0, or1, ar0, ar1), "left arrow must overlap the object's rows"
        dist = int(ac0 - oc1 - 1)
    elif direction == "down":
        assert _intervals_overlap(oc0, oc1, ac0, ac1), "down arrow must overlap the object's columns"
        dist = int(or0 - ar1 - 1)
    else:
        assert direction == "up"
        assert _intervals_overlap(oc0, oc1, ac0, ac1), "up arrow must overlap the object's columns"
        dist = int(ar0 - or1 - 1)
    assert dist >= 1, "gap-encoded move distance must be positive"
    return dist


def _translate_bbox(
    bbox: tuple[int, int, int, int],
    dr: int,
    dc: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 4

    sequence = [int(g[r, w - 1]) for r in range(h) if int(g[r, w - 1]) != BG]
    assert sequence, "expected a non-empty sequence bar in the rightmost column"
    assert len(set(sequence)) == len(sequence), "sequence bar colors must be unique"

    colors = {int(v) for v in np.unique(g) if int(v) != BG}
    object_candidates = sorted(color for color in colors if color not in set(sequence))
    assert len(object_candidates) == 1, "expected exactly one target-object color"
    object_color = int(object_candidates[0])

    object_components = _components_for_color(g, object_color)
    assert len(object_components) == 1, "target object must be a single connected component"
    object_coords = object_components[0]
    object_bbox = _bbox(object_coords)
    obj_r0, obj_c0, obj_r1, obj_c1 = object_bbox
    object_rel = [(int(r - obj_r0), int(c - obj_c0)) for r, c in object_coords]

    current_bbox = object_bbox
    for arrow_color in sequence:
        color_cells = [(int(r), int(c)) for r, c in np.argwhere(g == int(arrow_color))]
        seq_cells = [(r, c) for r, c in color_cells if int(c) == int(w - 1)]
        arrow_cells = [(r, c) for r, c in color_cells if int(c) != int(w - 1)]
        assert len(seq_cells) == 1, "each instruction color must appear exactly once in the sequence bar"
        assert len(arrow_cells) == 5, "each instruction color must form one 5-cell arrow"

        direction = _arrow_direction(arrow_cells)
        arrow_bbox = _bbox(arrow_cells)
        dist = _movement_distance(current_bbox, arrow_bbox, direction)
        dr, dc = DIR_VECS[direction]
        current_bbox = _translate_bbox(current_bbox, dr * dist, dc * dist)

    out = np.zeros_like(g)
    final_r0, final_c0, _final_r1, _final_c1 = current_bbox
    for rr, cc in object_rel:
        tr = int(final_r0 + rr)
        tc = int(final_c0 + cc)
        assert 0 <= tr < h and 0 <= tc < w
        out[tr, tc] = object_color
    return out


def _rect_cells(top: int, left: int) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r in range(int(top), int(top + OBJ_SIZE))
        for c in range(int(left), int(left + OBJ_SIZE))
    }


def _random_object_cells() -> list[tuple[int, int]]:
    mask = np.array(random.choice(OBJECT_BASE_MASKS), copy=True)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, random.randrange(4))
    coords = [(int(r), int(c)) for r in range(OBJ_SIZE) for c in range(OBJ_SIZE) if bool(mask[r, c])]
    r0, c0, r1, c1 = _bbox(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (OBJ_SIZE, OBJ_SIZE)
    return coords


def _paint_cells(
    grid: np.ndarray,
    origin: tuple[int, int],
    rel_cells: list[tuple[int, int]] | frozenset[tuple[int, int]],
    color: int,
) -> None:
    r0, c0 = origin
    for rr, cc in rel_cells:
        grid[int(r0 + rr), int(c0 + cc)] = int(color)


def _swept_rect_cells(pos: tuple[int, int], direction: str, distance: int) -> set[tuple[int, int]]:
    dr, dc = DIR_VECS[direction]
    swept: set[tuple[int, int]] = set()
    for step in range(int(distance) + 1):
        swept |= _rect_cells(int(pos[0] + dr * step), int(pos[1] + dc * step))
    return swept


def _arrow_origin(
    object_pos: tuple[int, int],
    direction: str,
    distance: int,
    offset: int,
) -> tuple[int, int]:
    r, c = object_pos
    if direction == "right":
        return int(r + offset), int(c - distance - ARROW_SIZE)
    if direction == "left":
        return int(r + offset), int(c + OBJ_SIZE + distance)
    if direction == "down":
        return int(r - distance - ARROW_SIZE), int(c + offset)
    assert direction == "up"
    return int(r + OBJ_SIZE + distance), int(c + offset)


def _step_options(
    current: tuple[int, int],
    direction: str,
    height: int,
    play_width: int,
    blocked_by_past: set[tuple[int, int]],
    blocked_by_arrows: set[tuple[int, int]],
) -> list[dict[str, object]]:
    r, c = current
    if direction == "right":
        limit = min(MAX_DISTANCE, c - ARROW_SIZE, play_width - OBJ_SIZE - c)
    elif direction == "left":
        limit = min(MAX_DISTANCE, c, play_width - (OBJ_SIZE + ARROW_SIZE) - c)
    elif direction == "down":
        limit = min(MAX_DISTANCE, r - ARROW_SIZE, height - OBJ_SIZE - r)
    else:
        assert direction == "up"
        limit = min(MAX_DISTANCE, r, height - (OBJ_SIZE + ARROW_SIZE) - r)

    if limit < 1:
        return []

    dists = list(range(1, int(limit) + 1))
    random.shuffle(dists)
    offsets = list(OFFSET_CHOICES)
    options: list[dict[str, object]] = []
    for distance in dists:
        dr, dc = DIR_VECS[direction]
        next_pos = (int(r + dr * distance), int(c + dc * distance))
        random.shuffle(offsets)
        for offset in offsets:
            ar, ac = _arrow_origin(current, direction, int(distance), int(offset))
            if not (0 <= ar <= height - ARROW_SIZE and 0 <= ac <= play_width - ARROW_SIZE):
                continue
            arrow_block = _rect_cells(ar, ac)
            if arrow_block & blocked_by_past:
                continue
            if arrow_block & blocked_by_arrows:
                continue
            options.append(
                {
                    "distance": int(distance),
                    "next_pos": next_pos,
                    "arrow_origin": (int(ar), int(ac)),
                    "path_block": _swept_rect_cells(current, direction, int(distance)),
                }
            )
    return options


def _sample_directions(count: int) -> list[str]:
    while True:
        directions = [random.choice(DIRECTIONS) for _ in range(int(count))]
        if not any(d in HORIZONTAL_DIRS for d in directions):
            continue
        if not any(d in VERTICAL_DIRS for d in directions):
            continue
        if len(set(directions)) < 2:
            continue
        return directions


def _sample_sequence_rows(height: int, count: int) -> list[int]:
    rows = random.sample(range(int(height)), int(count))
    rows.sort()
    return rows


def _plan_scene(height: int, width: int, directions: list[str]) -> dict[str, object] | None:
    play_width = int(width - 2)
    start_rows = list(range(4, height - 6))
    start_cols = list(range(4, play_width - 6))
    if not start_rows or not start_cols:
        return None

    current = (random.choice(start_rows), random.choice(start_cols))
    positions = [current]
    past_block = _rect_cells(current[0], current[1])
    arrow_block = set()
    arrows: list[dict[str, object]] = []

    for direction in directions:
        options = _step_options(current, direction, height, play_width, past_block, arrow_block)
        if not options:
            return None
        choice = random.choice(options)
        arrows.append(
            {
                "direction": direction,
                "origin": choice["arrow_origin"],
                "distance": choice["distance"],
            }
        )
        arrow_block |= _rect_cells(choice["arrow_origin"][0], choice["arrow_origin"][1])
        past_block |= set(choice["path_block"])
        current = choice["next_pos"]
        positions.append(current)

    return {"positions": positions, "arrows": arrows, "play_width": play_width}


def _render_scene(
    height: int,
    width: int,
    object_cells: list[tuple[int, int]],
    object_color: int,
    arrow_colors: list[int],
    sequence_rows: list[int],
    plan: dict[str, object],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    positions = plan["positions"]
    arrows = plan["arrows"]

    _paint_cells(grid, positions[0], object_cells, object_color)

    for color, arrow in zip(arrow_colors, arrows):
        _paint_cells(grid, arrow["origin"], ARROW_PATTERNS[arrow["direction"]], color)

    for row, color in zip(sequence_rows, arrow_colors):
        grid[int(row), int(width - 1)] = int(color)

    return grid


def _render_expected(
    height: int,
    width: int,
    object_cells: list[tuple[int, int]],
    object_color: int,
    final_pos: tuple[int, int],
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    _paint_cells(out, final_pos, object_cells, object_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    count = random.randint(*NUM_ARROWS_RANGE)

    directions = _sample_directions(count)
    plan = _plan_scene(height, width, directions)
    if plan is None:
        return None

    colors = random.sample(PALETTE, count + 1)
    object_color = int(colors[0])
    arrow_colors = [int(color) for color in colors[1:]]
    object_cells = _random_object_cells()
    sequence_rows = _sample_sequence_rows(height, count)

    grid = _render_scene(height, width, object_cells, object_color, arrow_colors, sequence_rows, plan)
    expected = _render_expected(height, width, object_cells, object_color, plan["positions"][-1])

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
