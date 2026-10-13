import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
TEAL = 8

COLOR_CYCLE = (RED, TEAL, YELLOW)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)

LEFT_TURN = {
    UP: LEFT,
    LEFT: DOWN,
    DOWN: RIGHT,
    RIGHT: UP,
}

RIGHT_TURN = {
    UP: RIGHT,
    RIGHT: DOWN,
    DOWN: LEFT,
    LEFT: UP,
}

MIRROR_REFLECT = {
    "\\": {
        UP: LEFT,
        LEFT: UP,
        DOWN: RIGHT,
        RIGHT: DOWN,
    },
    "/": {
        UP: RIGHT,
        RIGHT: UP,
        DOWN: LEFT,
        LEFT: DOWN,
    },
}

DIAGONAL_STEPS = {
    "\\": ((-1, -1), (1, 1)),
    "/": ((-1, 1), (1, -1)),
}

GRID_MIN = 12
GRID_MAX = 24
TURN_COUNT_CHOICES = (2, 3, 3, 4, 4, 5)
MIN_SEGMENT_LEN = 2
MAX_TURN_SEGMENT_LEN = 7
MAX_MIRROR_LEN = 5
GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 60


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _add(p: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return p[0] + d[0] * k, p[1] + d[1] * k


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    if connectivity == 4:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _entry_direction(entry: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = entry
    h, w = shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert on_top + on_bottom + on_left + on_right == 1, "entry must lie on exactly one edge"
    if on_top:
        return DOWN
    if on_bottom:
        return UP
    if on_left:
        return RIGHT
    return LEFT


def _steps_to_exit(pos: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    steps = 0
    cur = pos
    while True:
        cur = _add(cur, direction)
        if not in_bounds(shape, cur[0], cur[1]):
            return steps
        steps += 1


def _mirror_for_turn(direction: tuple[int, int], next_direction: tuple[int, int]) -> str:
    for orientation, mapping in MIRROR_REFLECT.items():
        if mapping[direction] == next_direction:
            return orientation
    raise AssertionError("invalid 90-degree turn")


def _mirror_orientation(component: np.ndarray) -> str:
    assert component.shape[0] >= 2, "mirror segments must be at least length 2"
    diag_main = component[:, 0] - component[:, 1]
    diag_anti = component[:, 0] + component[:, 1]
    if np.all(diag_main == diag_main[0]):
        return "\\"
    if np.all(diag_anti == diag_anti[0]):
        return "/"
    raise AssertionError("gray component is not a straight 45-degree segment")


def _parse_mirrors(grid: np.ndarray) -> tuple[dict[tuple[int, int], str], list[np.ndarray]]:
    comps = _components_mask(grid == GRAY, connectivity=8)
    assert comps, "expected at least one mirror segment"
    mirror_map: dict[tuple[int, int], str] = {}
    for comp in comps:
        orientation = _mirror_orientation(comp)
        for r, c in comp:
            coord = (int(r), int(c))
            assert coord not in mirror_map
            mirror_map[coord] = orientation
    return mirror_map, comps


def _trace_ray(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, GREEN, GRAY}

    entries = np.argwhere(g == GREEN)
    assert entries.shape[0] == 1, "expected one green entry pixel"
    entry = tuple(map(int, entries[0]))
    direction = _entry_direction(entry, g.shape)
    mirror_map, mirror_components = _parse_mirrors(g)

    visited_pairs: list[tuple[tuple[int, int], int]] = []
    segment_lengths: list[int] = [0]
    turn_points: list[tuple[int, int]] = []
    hit_mirrors: list[tuple[int, int]] = []
    visited_cells: set[tuple[int, int]] = set()

    pos = entry
    segment_idx = 0
    for _ in range(g.shape[0] * g.shape[1] * 8):
        nxt = _add(pos, direction)
        if not in_bounds(g.shape, nxt[0], nxt[1]):
            break
        if nxt in mirror_map:
            turn_points.append(pos)
            hit_mirrors.append(nxt)
            direction = MIRROR_REFLECT[mirror_map[nxt]][direction]
            segment_idx += 1
            segment_lengths.append(0)
            continue

        assert int(g[nxt]) == BLACK, "ray may only traverse background cells"
        assert nxt != entry, "ray must not revisit the entry pixel"
        assert nxt not in visited_cells, "ray must not revisit a path cell"
        visited_cells.add(nxt)
        visited_pairs.append((nxt, segment_idx))
        segment_lengths[segment_idx] += 1
        pos = nxt
    else:
        raise AssertionError("ray did not exit the grid")

    return {
        "entry": entry,
        "mirror_map": mirror_map,
        "mirror_components": mirror_components,
        "visited_pairs": visited_pairs,
        "segment_lengths": segment_lengths,
        "turn_points": turn_points,
        "hit_mirrors": hit_mirrors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot the edge ray through gray diagonal mirrors and color each straight run by turn order.

    Input structure:
    - The background is black.
    - Exactly one green pixel lies on a non-corner edge cell and marks the ray entry.
    - One or more gray 45-degree diagonal segments lie inside the grid; each segment is a straight "\" or "/" run of length at least 2.

    Rule:
    - Start from the green entry and move orthogonally inward.
    - If the next cell in the current direction belongs to a gray mirror segment, do not enter that mirror cell; instead turn 90 degrees on the current cell according to the segment orientation.
    - Color the first straight path segment red, the second teal, the third yellow, and keep cycling red-teal-yellow until the ray leaves the grid.
    - Keep all gray mirror cells gray and keep the green entry pixel unchanged.

    Key insight:
    - The mirror geometry determines where the ray bends, but the paint color depends only on the ordinal index of the straight segment between bends.
    """
    g = np.asarray(grid, dtype=int)
    trace = _trace_ray(g)
    out = g.copy()
    for (r, c), segment_idx in trace["visited_pairs"]:
        out[r, c] = COLOR_CYCLE[segment_idx % len(COLOR_CYCLE)]
    return out


def _random_entry(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return 0, random.randint(2, w - 3)
    if side == "bottom":
        return h - 1, random.randint(2, w - 3)
    if side == "left":
        return random.randint(2, h - 3), 0
    return random.randint(2, h - 3), w - 1


def _candidate_turn_lengths(edge_steps: int) -> list[int]:
    max_len = edge_steps - 1
    if max_len < MIN_SEGMENT_LEN:
        return []
    upper = min(max_len, MAX_TURN_SEGMENT_LEN)
    lengths = list(range(MIN_SEGMENT_LEN, upper + 1))
    if max_len > upper:
        lengths.append(max_len)
    random.shuffle(lengths)
    return lengths


def _plan_path(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    turns_left: int,
    entry: tuple[int, int],
    used_cells: set[tuple[int, int]],
    mirror_heads: set[tuple[int, int]],
) -> dict | None:
    edge_steps = _steps_to_exit(pos, direction, shape)
    if turns_left == 0:
        if edge_steps < MIN_SEGMENT_LEN:
            return None
        final_cells = [_add(pos, direction, k) for k in range(1, edge_steps + 1)]
        if any(cell == entry or cell in used_cells or cell in mirror_heads for cell in final_cells):
            return None
        return {
            "segments": [final_cells],
            "mirror_specs": [],
        }

    for length in _candidate_turn_lengths(edge_steps):
        segment_cells = [_add(pos, direction, k) for k in range(1, length + 1)]
        if any(cell == entry or cell in used_cells or cell in mirror_heads for cell in segment_cells):
            continue

        pivot = segment_cells[-1]
        head = _add(pivot, direction)
        if head == entry or head in used_cells or head in mirror_heads:
            continue
        if any(_chebyshev(head, other) <= 1 for other in mirror_heads):
            continue

        next_options = [LEFT_TURN[direction], RIGHT_TURN[direction]]
        random.shuffle(next_options)
        for next_direction in next_options:
            first_after_turn = _add(pivot, next_direction)
            if not in_bounds(shape, first_after_turn[0], first_after_turn[1]):
                continue
            if first_after_turn == entry or first_after_turn in used_cells or first_after_turn in mirror_heads:
                continue
            orientation = _mirror_for_turn(direction, next_direction)
            rest = _plan_path(
                shape,
                pivot,
                next_direction,
                turns_left - 1,
                entry,
                used_cells | set(segment_cells),
                mirror_heads | {head},
            )
            if rest is None:
                continue
            return {
                "segments": [segment_cells] + rest["segments"],
                "mirror_specs": [{"head": head, "orientation": orientation}] + rest["mirror_specs"],
            }
    return None


def _mirror_options(
    shape: tuple[int, int],
    head: tuple[int, int],
    orientation: str,
    blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    other_heads: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    options: list[list[tuple[int, int]]] = []
    for step in DIAGONAL_STEPS[orientation]:
        segment = [head]
        cur = head
        for _ in range(MAX_MIRROR_LEN - 1):
            nxt = _add(cur, step)
            if not in_bounds(shape, nxt[0], nxt[1]):
                break
            if nxt in blocked or nxt in occupied or nxt in other_heads:
                break
            if any(_chebyshev(nxt, other) <= 1 for other in occupied | other_heads):
                break
            segment.append(nxt)
            cur = nxt
            if len(segment) >= 2:
                options.append(segment.copy())
    random.shuffle(options)
    return options


def _place_mirror_segments(
    shape: tuple[int, int],
    entry: tuple[int, int],
    plan: dict,
) -> set[tuple[int, int]] | None:
    all_heads = {spec["head"] for spec in plan["mirror_specs"]}
    if any(_chebyshev(a, b) <= 1 for a in all_heads for b in all_heads if a != b):
        return None

    path_cells = {cell for segment in plan["segments"] for cell in segment}
    blocked = set(path_cells)
    blocked.add(entry)

    for _ in range(PLACEMENT_ATTEMPTS):
        occupied: set[tuple[int, int]] = set()
        chosen: list[list[tuple[int, int]]] = []
        order = list(range(len(plan["mirror_specs"])))
        random.shuffle(order)
        ok = True
        for idx in order:
            spec = plan["mirror_specs"][idx]
            head = spec["head"]
            other_heads = all_heads - {head}
            if any(_chebyshev(head, other) <= 1 for other in occupied | other_heads):
                ok = False
                break
            options = _mirror_options(shape, head, spec["orientation"], blocked, occupied, other_heads)
            if not options:
                ok = False
                break
            cells = random.choice(options)
            chosen.append(cells)
            occupied.update(cells)
        if ok:
            return occupied
    return None


def _build_grid(
    shape: tuple[int, int],
    entry: tuple[int, int],
    mirror_cells: Iterable[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    grid[entry] = GREEN
    for cell in mirror_cells:
        grid[cell] = GRAY
    return grid


def _validate_generated(grid: np.ndarray, plan: dict) -> bool:
    try:
        trace = _trace_ray(grid)
        out = main(grid)
    except AssertionError:
        return False

    expected_pairs = []
    for segment_idx, segment in enumerate(plan["segments"]):
        for cell in segment:
            expected_pairs.append((cell, segment_idx))
    expected_heads = [spec["head"] for spec in plan["mirror_specs"]]
    expected_path = {cell for segment in plan["segments"] for cell in segment}

    if trace["visited_pairs"] != expected_pairs:
        return False
    if trace["hit_mirrors"] != expected_heads:
        return False
    if trace["segment_lengths"] != [len(segment) for segment in plan["segments"]]:
        return False
    if len(trace["mirror_components"]) != len(plan["mirror_specs"]):
        return False

    changed = set(map(tuple, np.argwhere(out != grid)))
    if changed != expected_path:
        return False
    if int(out[plan["segments"][0][0]]) != RED:
        return False

    segment_colors = {COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(len(plan["segments"]))}
    if not {RED, TEAL, YELLOW}.issubset(segment_colors):
        return False

    for segment_idx, segment in enumerate(plan["segments"]):
        color = COLOR_CYCLE[segment_idx % len(COLOR_CYCLE)]
        if any(int(out[cell]) != color for cell in segment):
            return False

    if int(out[trace["entry"]]) != GREEN:
        return False
    if not np.array_equal(out[grid == GRAY], grid[grid == GRAY]):
        return False

    orientations = {spec["orientation"] for spec in plan["mirror_specs"]}
    if len(plan["mirror_specs"]) >= 3 and len(orientations) < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
        entry = _random_entry(shape)
        turn_count = random.choice(TURN_COUNT_CHOICES)
        plan = _plan_path(
            shape,
            entry,
            _entry_direction(entry, shape),
            turn_count,
            entry,
            set(),
            set(),
        )
        if plan is None:
            continue
        mirror_cells = _place_mirror_segments(shape, entry, plan)
        if mirror_cells is None:
            continue
        grid = _build_grid(shape, entry, mirror_cells)
        if _validate_generated(grid, plan):
            return grid
    raise RuntimeError("failed to generate a valid fold-mirror ray puzzle")
