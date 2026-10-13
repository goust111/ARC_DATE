from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_2_036_9", "/home/lixinyu/snap/new/solvers/arc_utils.py"
).load_module()

Color = _ARC_UTILS.Color
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds


BLACK = Color.BLACK
RED = Color.RED
GREY = Color.GREY

CORNER_OFFSETS = {
    "nw": (0, 0),
    "ne": (0, 1),
    "sw": (1, 0),
    "se": (1, 1),
}
OFFSET_TO_CORNER = {offset: corner for corner, offset in CORNER_OFFSETS.items()}
OPPOSITE_CORNER = {
    "nw": "se",
    "ne": "sw",
    "sw": "ne",
    "se": "nw",
}
DIRECTION_TO_MARKER = {
    (1, 1): "nw",
    (1, -1): "ne",
    (-1, 1): "sw",
    (-1, -1): "se",
}
DIRECTIONS = tuple(DIRECTION_TO_MARKER)

CHAIN_RELAY_RANGE = (2, 4)
CHAIN_GAP_RANGE = (1, 3)
TAIL_LENGTH_RANGE = (2, 5)
CANVAS_MARGIN_RANGE = (1, 4)
DISTRACTOR_COUNT_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 500


def _corner_coord(top_left: tuple[int, int], corner: str) -> tuple[int, int]:
    dr, dc = CORNER_OFFSETS[corner]
    return int(top_left[0] + dr), int(top_left[1] + dc)


def _top_left_from_corner(coord: tuple[int, int], corner: str) -> tuple[int, int]:
    dr, dc = CORNER_OFFSETS[corner]
    return int(coord[0] - dr), int(coord[1] - dc)


def _square_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = map(int, top_left)
    return {
        (r0, c0),
        (r0, c0 + 1),
        (r0 + 1, c0),
        (r0 + 1, c0 + 1),
    }


def _square_halo(top_left: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    r0, c0 = map(int, top_left)
    return {
        (rr, cc)
        for rr in range(r0 - pad, r0 + 2 + pad)
        for cc in range(c0 - pad, c0 + 2 + pad)
    }


def _laterals_for_direction(direction: tuple[int, int]) -> tuple[str, str]:
    straight = DIRECTION_TO_MARKER[direction]
    trailing = OPPOSITE_CORNER[straight]
    laterals = [corner for corner in CORNER_OFFSETS if corner not in {straight, trailing}]
    laterals.sort()
    return laterals[0], laterals[1]


def _parse_squares(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BLACK
    assert set(map(int, np.unique(g))).issubset({BLACK, GREY, RED})

    components = find_components(g, bg=BLACK, connectivity=8, monochromatic=False)
    assert components, "expected at least one 2x2 square"

    squares: list[dict] = []
    seed_count = 0
    for idx, comp in enumerate(components):
        bbox = tuple(map(int, comp["bbox"]))
        area = int(comp["area"])
        assert area == 4
        assert bbox_h(bbox) == 2 and bbox_w(bbox) == 2

        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape == (2, 2)
        assert np.all(patch != BLACK)
        colors = set(map(int, patch.ravel()))
        assert colors.issubset({GREY, RED})

        red_positions = np.argwhere(patch == RED)
        is_seed = len(red_positions) == 1
        if is_seed:
            seed_count += 1
            rr, cc = map(int, red_positions[0])
            marker_corner = OFFSET_TO_CORNER[(rr, cc)]
        else:
            assert np.all(patch == GREY)
            marker_corner = None

        squares.append(
            {
                "id": idx,
                "tl": (r0, c0),
                "seed": is_seed,
                "marker_corner": marker_corner,
            }
        )

    assert seed_count == 1, "expected exactly one seed square"
    return squares


def _trace_ray(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    squares = _parse_squares(g)
    seed = next(square for square in squares if square["seed"])
    marker_corner = seed["marker_corner"]
    marker = _corner_coord(seed["tl"], marker_corner)
    emit = _corner_coord(seed["tl"], OPPOSITE_CORNER[marker_corner])
    direction = (int(emit[0] - marker[0]), int(emit[1] - marker[1]))
    assert direction in DIRECTION_TO_MARKER

    corner_map: dict[tuple[int, int], tuple[dict, str]] = {}
    for square in squares:
        for corner in CORNER_OFFSETS:
            corner_map[_corner_coord(square["tl"], corner)] = (square, corner)

    out = g.copy()
    painted: list[tuple[int, int]] = []
    hits: list[dict] = []
    current = emit
    relay_states: set[tuple[int, str, tuple[int, int]]] = set()
    budget = int(g.size * 6)

    while True:
        budget -= 1
        assert budget >= 0, "ray tracing should leave the grid"
        nr = int(current[0] + direction[0])
        nc = int(current[1] + direction[1])
        if not in_bounds(g.shape, nr, nc):
            break

        value = int(g[nr, nc])
        if value == BLACK:
            out[nr, nc] = RED
            painted.append((nr, nc))
            current = (nr, nc)
            continue

        square, hit_corner = corner_map[(nr, nc)]
        assert not square["seed"], "ray must not re-enter the seed square"
        state = (int(square["id"]), str(hit_corner), tuple(map(int, current)))
        assert state not in relay_states, "ray relay cycle detected"
        relay_states.add(state)

        exit_corner = OPPOSITE_CORNER[hit_corner]
        current = _corner_coord(square["tl"], exit_corner)
        hits.append(
            {
                "square_id": int(square["id"]),
                "tl": tuple(map(int, square["tl"])),
                "hit_corner": str(hit_corner),
                "exit_corner": str(exit_corner),
                "hit_coord": (int(nr), int(nc)),
            }
        )

    return {
        "output": out,
        "painted": painted,
        "hits": hits,
        "squares": squares,
        "seed": seed,
        "direction": direction,
        "straight_corner": DIRECTION_TO_MARKER[direction],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the red diagonal relay beam from the marked seed square through grey 2x2 blocks.

    Input structure:
    - A black background contains several isolated grey 2x2 squares.
    - Exactly one square is the seed: one of its corner cells is red and the other three cells are grey.
    - The intended beam meets other squares exactly on square-corner cells.

    Rule:
    - Keep every input square unchanged.
    - The seed emits a red diagonal ray in the direction from the red corner toward the seed's opposite corner, extended outward.
    - Paint every visited background cell red.
    - When the ray reaches a square corner, do not paint the square; instead, jump to that square's diagonally opposite corner and continue in the same direction until the ray leaves the grid.

    Key insight:
    - Each 2x2 square is a relay, not an obstacle: a corner hit teleports the beam to the opposite corner, so side-corner hits shift the downstream diagonal while straight-corner hits preserve it.
    """
    return _trace_ray(grid)["output"]


def _can_place_square(
    top_left: tuple[int, int],
    existing: list[tuple[int, int]],
    forbidden_cells: set[tuple[int, int]],
) -> bool:
    cells = _square_cells(top_left)
    if cells & forbidden_cells:
        return False
    halo = _square_halo(top_left)
    for other in existing:
        if halo & _square_cells(other):
            return False
    return True


def _pick_relay_corners(direction: tuple[int, int], relay_count: int) -> list[str]:
    straight = DIRECTION_TO_MARKER[direction]
    lateral_a, lateral_b = _laterals_for_direction(direction)

    lateral_cap = min(3, relay_count)
    lateral_count = min(
        relay_count,
        random.choices(
            list(range(1, lateral_cap + 1)),
            weights=[3, 5, 4][:lateral_cap],
            k=1,
        )[0],
    )
    lateral_slots = set(random.sample(range(relay_count), lateral_count))

    corners = []
    for idx in range(relay_count):
        if idx in lateral_slots:
            corners.append(random.choice((lateral_a, lateral_b)))
        else:
            corners.append(straight)
    return corners


def _build_chain() -> dict | None:
    direction = random.choice(DIRECTIONS)
    marker_corner = DIRECTION_TO_MARKER[direction]
    relay_count = random.randint(*CHAIN_RELAY_RANGE)
    relay_corners = _pick_relay_corners(direction, relay_count)
    relay_gaps = [random.randint(*CHAIN_GAP_RANGE) for _ in range(relay_count)]

    square_tls = [(0, 0)]
    occupied = set(_square_cells((0, 0)))
    painted_cells: set[tuple[int, int]] = set()
    emit = _corner_coord((0, 0), OPPOSITE_CORNER[marker_corner])

    for gap, hit_corner in zip(relay_gaps, relay_corners):
        segment = {
            (int(emit[0] + step * direction[0]), int(emit[1] + step * direction[1]))
            for step in range(1, gap + 1)
        }
        hit = (
            int(emit[0] + (gap + 1) * direction[0]),
            int(emit[1] + (gap + 1) * direction[1]),
        )
        top_left = _top_left_from_corner(hit, hit_corner)
        if not _can_place_square(top_left, square_tls, occupied | painted_cells | segment):
            return None

        if segment & occupied:
            return None
        painted_cells |= segment
        square_tls.append(top_left)
        occupied |= _square_cells(top_left)
        emit = _corner_coord(top_left, OPPOSITE_CORNER[hit_corner])

    if not any(corner != DIRECTION_TO_MARKER[direction] for corner in relay_corners):
        return None

    rows = [r for tl in square_tls for r, _ in _square_cells(tl)]
    cols = [c for tl in square_tls for _, c in _square_cells(tl)]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    tail_len = random.randint(*TAIL_LENGTH_RANGE)
    row_base = max_r - emit[0] if direction[0] == 1 else emit[0] - min_r
    col_base = max_c - emit[1] if direction[1] == 1 else emit[1] - min_c

    row_slack = random.randint(0, 2)
    col_slack = random.randint(0, 2)
    if random.random() < 0.5:
        row_slack = 0
    else:
        col_slack = 0

    top = random.randint(*CANVAS_MARGIN_RANGE)
    bottom = random.randint(*CANVAS_MARGIN_RANGE)
    left = random.randint(*CANVAS_MARGIN_RANGE)
    right = random.randint(*CANVAS_MARGIN_RANGE)

    if direction[0] == 1:
        bottom = tail_len - row_base + row_slack
    else:
        top = tail_len - row_base + row_slack

    if direction[1] == 1:
        right = tail_len - col_base + col_slack
    else:
        left = tail_len - col_base + col_slack

    if min(top, bottom, left, right) < 0:
        return None

    height = int(max_r - min_r + 1 + top + bottom)
    width = int(max_c - min_c + 1 + left + right)
    if max(height, width) > 30:
        return None

    shift = (int(top - min_r), int(left - min_c))
    return {
        "direction": direction,
        "marker_corner": marker_corner,
        "relay_corners": relay_corners,
        "relay_gaps": relay_gaps,
        "square_tls": square_tls,
        "shift": shift,
        "shape": (height, width),
    }


def _render_chain(chain: dict) -> np.ndarray:
    height, width = map(int, chain["shape"])
    shift_r, shift_c = map(int, chain["shift"])
    grid = np.zeros((height, width), dtype=int)
    for top_left in chain["square_tls"]:
        r0 = int(top_left[0] + shift_r)
        c0 = int(top_left[1] + shift_c)
        grid[r0 : r0 + 2, c0 : c0 + 2] = GREY

    seed_tl = chain["square_tls"][0]
    marker = _corner_coord(
        (int(seed_tl[0] + shift_r), int(seed_tl[1] + shift_c)),
        chain["marker_corner"],
    )
    grid[marker] = RED
    return grid


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if in_bounds(shape, rr, cc):
                    expanded.add((rr, cc))
    return expanded


def _place_distractors(grid: np.ndarray, path_cells: set[tuple[int, int]]) -> int:
    g = grid.copy()
    existing_tls = [tuple(map(int, square["tl"])) for square in _parse_squares(g)]
    buffered_path = _expand_cells(path_cells, g.shape, pad=1)
    occupied_cells = {tuple(map(int, rc)) for rc in np.argwhere(g != BLACK)}
    target = random.randint(*DISTRACTOR_COUNT_RANGE)

    candidates = [(r, c) for r in range(g.shape[0] - 1) for c in range(g.shape[1] - 1)]
    random.shuffle(candidates)

    placed = 0
    for top_left in candidates:
        cells = _square_cells(top_left)
        if not _can_place_square(top_left, existing_tls, buffered_path | occupied_cells):
            continue
        if cells & path_cells:
            continue
        r0, c0 = map(int, top_left)
        g[r0 : r0 + 2, c0 : c0 + 2] = GREY
        existing_tls.append(top_left)
        occupied_cells |= cells
        placed += 1
        if placed == target:
            break

    if placed == 0:
        return 0
    grid[:, :] = g
    return placed


def _naive_straight_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    squares = _parse_squares(g)
    seed = next(square for square in squares if square["seed"])
    marker_corner = seed["marker_corner"]
    marker = _corner_coord(seed["tl"], marker_corner)
    emit = _corner_coord(seed["tl"], OPPOSITE_CORNER[marker_corner])
    direction = (int(emit[0] - marker[0]), int(emit[1] - marker[1]))

    out = g.copy()
    current = emit
    while True:
        nr = int(current[0] + direction[0])
        nc = int(current[1] + direction[1])
        if not in_bounds(g.shape, nr, nc):
            break
        if g[nr, nc] == BLACK:
            out[nr, nc] = RED
        current = (nr, nc)
    return out


def _erase_square(grid: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    for r, c in _square_cells(top_left):
        g[r, c] = BLACK
    return g


def _validate_case(grid: np.ndarray, planned: dict) -> bool:
    trace = _trace_ray(grid)
    hits = trace["hits"]
    shift_r, shift_c = map(int, planned["shift"])
    planned_relays = [
        (int(tl[0] + shift_r), int(tl[1] + shift_c))
        for tl in planned["square_tls"][1:]
    ]

    if [hit["tl"] for hit in hits] != planned_relays:
        return False
    if [hit["hit_corner"] for hit in hits] != list(planned["relay_corners"]):
        return False
    if len(_parse_squares(grid)) - 1 <= len(hits):
        return False
    if not any(hit["hit_corner"] != trace["straight_corner"] for hit in hits):
        return False
    if np.array_equal(trace["output"], _naive_straight_output(grid)):
        return False
    if len(trace["painted"]) < 6:
        return False

    for hit in hits:
        if hit["hit_corner"] == trace["straight_corner"]:
            continue
        altered = _erase_square(grid, hit["tl"])
        if np.array_equal(main(altered), trace["output"]):
            return False

    return True


def _generate_case() -> np.ndarray | None:
    chain = _build_chain()
    if chain is None:
        return None

    grid = _render_chain(chain)
    initial_trace = _trace_ray(grid)
    if len(initial_trace["hits"]) != len(chain["relay_corners"]):
        return None

    distractors = _place_distractors(grid, set(initial_trace["painted"]))
    if distractors == 0:
        return None
    if not _validate_case(grid, chain):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-ray ARC input")
