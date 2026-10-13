import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

NORTH = (-1, 0)
EAST = (0, 1)
SOUTH = (1, 0)
WEST = (0, -1)
CARDINALS = [NORTH, EAST, SOUTH, WEST]

ACTION_TO_COLOR = {"L": GREEN, "R": BLUE, "S": YELLOW}

MAX_GENERATION_ATTEMPTS = 300
MAX_COMPONENT_ATTEMPTS = 200
MAX_ACTION_ATTEMPTS = 200
MAX_BRANCH_LEN = 3
MIN_COMPONENT_SIDE = 8
MAX_COMPONENT_SIDE = 14


def _in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _left_of(direction: tuple[int, int]) -> tuple[int, int]:
    if direction == NORTH:
        return WEST
    if direction == WEST:
        return SOUTH
    if direction == SOUTH:
        return EAST
    return NORTH


def _right_of(direction: tuple[int, int]) -> tuple[int, int]:
    if direction == NORTH:
        return EAST
    if direction == EAST:
        return SOUTH
    if direction == SOUTH:
        return WEST
    return NORTH


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return (-direction[0], -direction[1])


def _turn(direction: tuple[int, int], action: str) -> tuple[int, int]:
    if action == "L":
        return _left_of(direction)
    if action == "R":
        return _right_of(direction)
    return direction


def _pipe_neighbors(grid: np.ndarray, pos: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for nbr in _neighbors4(pos):
        if _in_bounds(grid.shape, nbr) and int(grid[nbr]) != BG:
            out.append(nbr)
    return out


def _trace_signal(
    grid: np.ndarray, source: tuple[int, int]
) -> tuple[set[tuple[int, int]], tuple[int, int] | None]:
    neighbors = _pipe_neighbors(grid, source)
    if len(neighbors) != 1:
        return set(), None

    prev = source
    curr = neighbors[0]
    traversed_grey: set[tuple[int, int]] = set()
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while True:
        direction = (curr[0] - prev[0], curr[1] - prev[1])
        state = (curr, direction)
        if state in seen_states:
            return traversed_grey, None
        seen_states.add(state)

        color = int(grid[curr])
        if color == GREY:
            traversed_grey.add(curr)
            neighbors = _pipe_neighbors(grid, curr)
            if len(neighbors) <= 1:
                return traversed_grey, curr
            forward = [nbr for nbr in neighbors if nbr != prev]
            if len(forward) != 1:
                return traversed_grey, None
            prev, curr = curr, forward[0]
            continue

        if color == GREEN:
            next_dir = _left_of(direction)
        elif color == BLUE:
            next_dir = _right_of(direction)
        elif color == YELLOW:
            next_dir = direction
        else:
            return traversed_grey, None

        nxt = (curr[0] + next_dir[0], curr[1] + next_dir[1])
        if not _in_bounds(grid.shape, nxt) or int(grid[nxt]) == BG:
            return traversed_grey, None
        prev, curr = curr, nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each edge source through a grey pipe network, turning at colored junctions and lighting the reached terminals.

    Input structure:
    - Background is black.
    - Grey pixels form one-cell-wide horizontal and vertical pipe segments.
    - Some 3-way or 4-way intersections are recolored as green, blue, or yellow junction pixels.
    - One or two red edge pixels mark pipe sources; other dead-end grey endpoints are terminals.

    Rule:
    - Start a signal from every red source and move along the unique adjacent pipe branch.
    - Whenever the signal enters a colored junction, green means turn left, blue means turn right, and yellow means continue straight relative to the incoming direction.
    - If the chosen outgoing direction has no pipe branch, that signal stops at the junction; otherwise it continues until it reaches a dead-end terminal or repeats a directed state.
    - Every traversed grey pipe pixel becomes orange, and any terminal grey endpoint reached by a signal becomes red; source pixels and colored junction pixels keep their original colors.

    Key insight:
    - The colored junction pixel is not recolored data but a routing instruction: decoding left/right/straight from junction color is necessary before tracing which terminals receive the signal.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    sources = sorted(
        (int(r), int(c))
        for r, c in np.argwhere(g == RED)
        if int(r) in (0, H - 1) or int(c) in (0, W - 1)
    )

    out = g.copy()
    traversed: set[tuple[int, int]] = set()
    terminals: set[tuple[int, int]] = set()

    for source in sources:
        grey_cells, terminal = _trace_signal(g, source)
        traversed.update(grey_cells)
        if terminal is not None and int(g[terminal]) == GREY:
            terminals.add(terminal)

    for r, c in traversed:
        if int(g[r, c]) == GREY:
            out[r, c] = ORANGE
    for r, c in terminals:
        out[r, c] = RED

    return out


def _choice(seq):
    return random.choice(list(seq))


def _allowed_actions(direction: tuple[int, int]) -> list[str]:
    if direction == EAST:
        return ["L", "R", "S"]
    if direction == NORTH:
        return ["R", "S"]
    if direction == SOUTH:
        return ["L", "S"]
    return []


def _sample_actions(n_nodes: int) -> list[str] | None:
    target = {"L", "R", "S"}
    for _ in range(MAX_ACTION_ATTEMPTS):
        actions: list[str] = []
        direction = EAST
        seen: set[str] = set()
        for i in range(n_nodes):
            options = _allowed_actions(direction)
            remaining = n_nodes - i
            missing = [action for action in options if action not in seen]
            if missing and len(target - seen) >= remaining:
                action = _choice(missing)
            elif missing and random.random() < 0.7:
                action = _choice(missing)
            else:
                action = _choice(options)
            actions.append(action)
            seen.add(action)
            direction = _turn(direction, action)
        if seen == target:
            return actions
    return None


def _ray_cells(
    start: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    return [
        (start[0] + direction[0] * step, start[1] + direction[1] * step)
        for step in range(1, length + 1)
    ]


def _can_add_cells(
    occupied: set[tuple[int, int]],
    start: tuple[int, int],
    cells: list[tuple[int, int]],
    *,
    keep_source_edge_clear: bool = False,
) -> bool:
    new_cells = set(cells)
    for idx, cell in enumerate(cells):
        r, c = cell
        if cell in occupied:
            return False
        if keep_source_edge_clear and c <= 0:
            return False

        allowed = {start if idx == 0 else cells[idx - 1]}
        if idx + 1 < len(cells):
            allowed.add(cells[idx + 1])

        for nbr in _neighbors4(cell):
            if nbr in occupied and nbr not in allowed:
                return False
            if nbr in new_cells and nbr not in allowed:
                return False
    return True


def _feasible_lengths(
    occupied: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    *,
    keep_source_edge_clear: bool = False,
    max_len: int = MAX_BRANCH_LEN,
) -> list[int]:
    feasible: list[int] = []
    cells: list[tuple[int, int]] = []
    for step in range(1, max_len + 1):
        cell = (start[0] + direction[0] * step, start[1] + direction[1] * step)
        cells.append(cell)
        if not _can_add_cells(
            occupied,
            start,
            cells,
            keep_source_edge_clear=keep_source_edge_clear,
        ):
            break
        feasible.append(step)
    return feasible


def _degree_map(grid: np.ndarray) -> dict[tuple[int, int], int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid != BG)]
    coord_set = set(coords)
    degrees: dict[tuple[int, int], int] = {}
    for pos in coords:
        degrees[pos] = sum(nbr in coord_set for nbr in _neighbors4(pos))
    return degrees


def _validate_component(patch: np.ndarray) -> bool:
    if patch.ndim != 2 or not (1 <= patch.shape[0] <= MAX_COMPONENT_SIDE and 1 <= patch.shape[1] <= MAX_COMPONENT_SIDE):
        return False

    sources = [tuple(map(int, rc)) for rc in np.argwhere(patch == RED)]
    if len(sources) != 1:
        return False
    source = sources[0]
    if source[1] != 0:
        return False

    foreground = {tuple(map(int, rc)) for rc in np.argwhere(patch != BG)}
    if sum(c == 0 for _r, c in foreground) != 1:
        return False

    degrees = _degree_map(patch)
    for pos, degree in degrees.items():
        color = int(patch[pos])
        if pos == source:
            if degree != 1:
                return False
        elif color in (GREEN, BLUE, YELLOW):
            if degree not in (3, 4):
                return False
        elif color == GREY:
            if degree not in (1, 2):
                return False
        else:
            return False

    return True


def _build_component(stop_case: bool) -> np.ndarray | None:
    for _ in range(MAX_COMPONENT_ATTEMPTS):
        n_nodes = random.randint(4, 5) if stop_case else random.randint(3, 5)
        actions = _sample_actions(n_nodes)
        if actions is None:
            continue

        occupied: set[tuple[int, int]] = {(0, 0)}
        source = (0, 0)
        current = source
        direction = EAST
        nodes: list[dict] = []
        ok = True

        for idx, action in enumerate(actions):
            seg_len = random.randint(2, 4)
            cells = _ray_cells(current, direction, seg_len)
            if not _can_add_cells(occupied, current, cells):
                ok = False
                break
            occupied.update(cells)
            node_pos = cells[-1]
            chosen_dir = _turn(direction, action)
            is_stop = stop_case and idx == n_nodes - 1
            node = {
                "pos": node_pos,
                "action": action,
                "incoming": direction,
                "back": _opposite(direction),
                "chosen": chosen_dir,
                "out": None if is_stop else chosen_dir,
                "stop": is_stop,
            }
            nodes.append(node)
            current = node_pos
            if is_stop:
                break
            direction = chosen_dir

        if not ok:
            continue

        if not stop_case:
            term_len = random.randint(2, 4)
            cells = _ray_cells(current, direction, term_len)
            if not _can_add_cells(occupied, current, cells):
                continue
            occupied.update(cells)

        random.shuffle(nodes)
        for node in nodes:
            pos = node["pos"]
            if node["stop"]:
                extra_dirs = [
                    d for d in CARDINALS if d not in {node["back"], node["chosen"]}
                ]
            else:
                extras = [
                    d for d in CARDINALS if d not in {node["back"], node["out"]}
                ]
                feasible_extras = [
                    d
                    for d in extras
                    if _feasible_lengths(
                        occupied,
                        pos,
                        d,
                        keep_source_edge_clear=(d == WEST),
                    )
                ]
                if not feasible_extras:
                    ok = False
                    break
                if len(feasible_extras) == 2 and random.random() < 0.5:
                    extra_dirs = feasible_extras
                else:
                    extra_dirs = [_choice(feasible_extras)]

            for extra_dir in extra_dirs:
                lengths = _feasible_lengths(
                    occupied,
                    pos,
                    extra_dir,
                    keep_source_edge_clear=(extra_dir == WEST),
                )
                if not lengths:
                    ok = False
                    break
                branch_len = _choice(lengths)
                occupied.update(_ray_cells(pos, extra_dir, branch_len))
            if not ok:
                break

        if not ok:
            continue

        min_r = min(r for r, _c in occupied)
        max_r = max(r for r, _c in occupied)
        min_c = min(c for _r, c in occupied)
        max_c = max(c for _r, c in occupied)
        if min_c != 0:
            continue

        height = max_r - min_r + 1
        width = max_c - min_c + 1
        if not (MIN_COMPONENT_SIDE <= max(height, width) <= MAX_COMPONENT_SIDE):
            continue
        if height > MAX_COMPONENT_SIDE or width > MAX_COMPONENT_SIDE:
            continue

        patch = np.full((height, width), BG, dtype=int)
        row_shift = -min_r
        for r, c in occupied:
            patch[r + row_shift, c] = GREY
        patch[source[0] + row_shift, source[1]] = RED
        for node in nodes:
            rr, cc = node["pos"]
            patch[rr + row_shift, cc] = ACTION_TO_COLOR[node["action"]]

        if _validate_component(patch):
            return patch

    return None


def _source_edge(patch: np.ndarray) -> str:
    source = tuple(map(int, np.argwhere(patch == RED)[0]))
    r, c = source
    if c == 0:
        return "left"
    if c == patch.shape[1] - 1:
        return "right"
    if r == 0:
        return "top"
    return "bottom"


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    H, W = canvas.shape
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    for dr, dc in np.argwhere(patch != BG):
        r = int(r0 + dr)
        c = int(c0 + dc)
        r1 = max(0, r - 1)
        r2 = min(H, r + 2)
        c1 = max(0, c - 1)
        c2 = min(W, c + 2)
        if np.any(canvas[r1:r2, c1:c2] != BG):
            return False
    return True


def _place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _candidate_positions(
    canvas_shape: tuple[int, int], patch: np.ndarray
) -> list[tuple[int, int]]:
    H, W = canvas_shape
    h, w = patch.shape
    edge = _source_edge(patch)
    if edge == "left":
        return [(r0, 0) for r0 in range(H - h + 1)]
    if edge == "right":
        return [(r0, W - w) for r0 in range(H - h + 1)]
    if edge == "top":
        return [(0, c0) for c0 in range(W - w + 1)]
    return [(H - h, c0) for c0 in range(W - w + 1)]


def _generate_scene() -> np.ndarray | None:
    num_components = random.randint(1, 2)
    stop_index = None
    if num_components == 2 and random.random() < 0.5:
        stop_index = random.randrange(num_components)

    patches: list[np.ndarray] = []
    for idx in range(num_components):
        patch = _build_component(stop_case=(idx == stop_index))
        if patch is None:
            return None
        patch = np.rot90(patch, random.randrange(4)).copy()
        patches.append(patch)

    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    min_h = max(14 if num_components == 1 else 18, max_h + 1)
    min_w = max(14 if num_components == 1 else 18, max_w + 1)
    if min_h > 30 or min_w > 30:
        return None

    H = random.randint(min_h, 30)
    W = random.randint(min_w, 30)
    canvas = np.full((H, W), BG, dtype=int)

    order = sorted(
        range(len(patches)),
        key=lambda i: int(np.count_nonzero(patches[i] != BG)),
        reverse=True,
    )

    for idx in order:
        patch = patches[idx]
        positions = _candidate_positions(canvas.shape, patch)
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if _can_place(canvas, patch, r0, c0):
                _place_patch(canvas, patch, r0, c0)
                placed = True
                break
        if not placed:
            return None

    source_count = int(np.sum(canvas == RED))
    if source_count != num_components:
        return None
    if not np.any(canvas == GREEN) or not np.any(canvas == BLUE) or not np.any(canvas == YELLOW):
        return None
    if not np.any(canvas == GREY):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_scene()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid pipe-routing sample")
