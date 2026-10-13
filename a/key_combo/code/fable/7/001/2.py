import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """ARC toolbox-compatible bounds check using (row, col) coordinates."""
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive bbox of an (N,2) coordinate array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _gray_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected one grey occluder"
    bbox = bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey cells must form a solid rectangle"
    assert int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1), "grey occluder must be rectangular"
    return bbox


def _green_neighbors(grid: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREEN:
            out.append((nr, nc))
    return out


def _adjacent_to_color(grid: np.ndarray, cell: tuple[int, int], color: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == color:
            out.append((nr, nc))
    return out


def _component_contact(grid: np.ndarray, comp: np.ndarray) -> tuple[int, int]:
    contacts = []
    for r, c in comp:
        if _adjacent_to_color(grid, (int(r), int(c)), GREY):
            contacts.append((int(r), int(c)))
    assert len(contacts) == 1, "each visible green segment must touch the occluder once"
    return contacts[0]


def _component_touches_red(grid: np.ndarray, comp: np.ndarray) -> bool:
    return any(_adjacent_to_color(grid, (int(r), int(c)), RED) for r, c in comp)


def _contact_dir_into_gray(grid: np.ndarray, cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    dirs = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
            dirs.append((dr, dc))
    assert len(dirs) == 1, "contact cell must face exactly one grey cell"
    return dirs[0]


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise AssertionError("segment endpoints must share a row or column")


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        assert in_bounds(grid.shape, r, c)
        grid[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the path segment hidden by the grey rectangle and recover the hidden turn-trigger obstacle.

    Input structure:
    - A black background contains scattered teal obstacle pixels, one solid grey occluding rectangle, a green start-side path fragment, and a red end marker line.
    - The full green path obeys the kinematic rule "go straight, and turn only when the next cell ahead is teal".
    - The grey rectangle hides one middle path segment and, if the hidden segment turns, the teal obstacle pixel that forced that turn.

    Rule:
    - Find the two visible green path fragments that touch the grey rectangle.
    - Use the fragment that reaches the red marker to identify the exit side; the other fragment is the entry side.
    - Read the entry direction from the entry contact toward grey, and the exit direction as the opposite of the exit contact's grey-facing direction.
    - If the two directions match, fill the hidden straight segment between them.
    - If they are perpendicular, join them with the unique L-turn at the intersection of the entry ray and exit ray, then place one teal obstacle one step ahead from the turn along the entry direction.
    - Replace every remaining grey cell with black.

    Key insight:
    - The missing pixels are determined by local path kinematics: visible trajectory endpoints reveal both the hidden path geometry and the unseen causal obstacle that must have triggered the turn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _gray_bbox(g)
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    green_components = components_mask(g == GREEN)
    assert len(green_components) == 2, "the occluder should split the path into two green fragments"

    exit_flags = [_component_touches_red(g, comp) for comp in green_components]
    assert sum(exit_flags) == 1, "exactly one visible green fragment must reach the red marker"
    exit_comp = green_components[exit_flags.index(True)]
    entry_comp = green_components[1 - exit_flags.index(True)]

    entry_contact = _component_contact(g, entry_comp)
    exit_contact = _component_contact(g, exit_comp)

    entry_dir = _contact_dir_into_gray(g, entry_contact)
    exit_gray_dir = _contact_dir_into_gray(g, exit_contact)
    exit_dir = (-exit_gray_dir[0], -exit_gray_dir[1])

    hidden_start = (entry_contact[0] + entry_dir[0], entry_contact[1] + entry_dir[1])
    hidden_end = (exit_contact[0] - exit_dir[0], exit_contact[1] - exit_dir[1])
    assert r0 <= hidden_start[0] <= r1 and c0 <= hidden_start[1] <= c1
    assert r0 <= hidden_end[0] <= r1 and c0 <= hidden_end[1] <= c1

    hidden_path: list[tuple[int, int]]
    if entry_dir == exit_dir:
        hidden_path = _segment(hidden_start, hidden_end)
    else:
        assert entry_dir[0] != exit_dir[0] and entry_dir[1] != exit_dir[1], "hidden segment must be straight or a single right-angle turn"
        if entry_dir[0] == 0:
            turn = (hidden_start[0], hidden_end[1])
        else:
            turn = (hidden_end[0], hidden_start[1])
        hidden_path = _segment(hidden_start, turn) + _segment(turn, hidden_end)[1:]
        barrier = (turn[0] + entry_dir[0], turn[1] + entry_dir[1])
        assert r0 <= barrier[0] <= r1 and c0 <= barrier[1] <= c1
        assert barrier not in hidden_path, "barrier must sit ahead of the turning point, not on the path"
        out[barrier] = TEAL

    _paint(out, hidden_path, GREEN)
    return out


def _dihedral(grid: np.ndarray, op: int) -> np.ndarray:
    out = np.rot90(grid, k=op % 4)
    if op >= 4:
        out = np.fliplr(out)
    return out.copy()


def _rect_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _manhattan_halo(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) > radius:
                    continue
                nr, nc = r + dr, c + dc
                if in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def _ordered_green_path(grid: np.ndarray) -> list[tuple[int, int]]:
    green_components = components_mask(grid == GREEN)
    assert len(green_components) == 1, "expected one connected green path in the full output"
    green_cells = {tuple(map(int, cell)) for cell in green_components[0]}
    degrees = {cell: len([n for n in _green_neighbors(grid, cell) if n in green_cells]) for cell in green_cells}
    assert all(deg <= 2 for deg in degrees.values()), "green path must be non-branching"
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "green path must have exactly two endpoints"

    start_candidates = [cell for cell in endpoints if not _adjacent_to_color(grid, cell, RED)]
    end_candidates = [cell for cell in endpoints if _adjacent_to_color(grid, cell, RED)]
    assert len(start_candidates) == 1 and len(end_candidates) == 1, "path endpoints must separate into start side and red-facing side"

    ordered = [start_candidates[0]]
    prev = None
    cur = start_candidates[0]
    while True:
        nxt = [cell for cell in _green_neighbors(grid, cur) if cell in green_cells and cell != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path must be a single chain"
        prev, cur = cur, nxt[0]
        ordered.append(cur)

    assert len(ordered) == len(green_cells), "trace must visit every green cell exactly once"
    assert ordered[-1] == end_candidates[0], "trace must end at the red-facing endpoint"
    return ordered


def _validate_full_output(grid: np.ndarray) -> bool:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert not np.any(grid == GREY), "full output must not contain grey"

    ordered = _ordered_green_path(grid)
    dirs = [
        (ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1])
        for i in range(len(ordered) - 1)
    ]
    assert dirs, "green path must contain at least two cells"

    turn_count = 0
    for i in range(1, len(ordered) - 1):
        incoming = (ordered[i][0] - ordered[i - 1][0], ordered[i][1] - ordered[i - 1][1])
        outgoing = (ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1])
        forward = (ordered[i][0] + incoming[0], ordered[i][1] + incoming[1])
        if incoming == outgoing:
            assert ordered[i + 1] == forward, "straight motion must continue to the next green cell"
        else:
            turn_count += 1
            assert int(grid[forward]) == TEAL, "a turn must be caused by one teal obstacle directly ahead"
            assert incoming[0] != outgoing[0] and incoming[1] != outgoing[1], "turns must be right-angle"

    last_dir = dirs[-1]
    first_red = (ordered[-1][0] + last_dir[0], ordered[-1][1] + last_dir[1])
    reds = _adjacent_to_color(grid, ordered[-1], RED)
    assert len(reds) == 1 and reds[0] == first_red, "the red marker must continue the final path direction"
    second_red = (first_red[0] + last_dir[0], first_red[1] + last_dir[1])
    assert in_bounds(grid.shape, second_red[0], second_red[1]) and int(grid[second_red]) == RED
    assert turn_count in (0, 1), "generated outputs should contain at most one turn"
    return True


def _validate_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    assert inp.ndim == 2 and expected.ndim == 2
    assert inp.shape == expected.shape
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30

    r0, c0, r1, c1 = _gray_bbox(inp)
    assert np.all(inp[r0 : r1 + 1, c0 : c1 + 1] == GREY)

    mask = np.ones(inp.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    assert np.array_equal(inp[mask], expected[mask]), "only the occluder area may differ"
    assert not np.any(expected == GREY), "expected output cannot retain grey"
    _validate_full_output(expected)
    solved = main(inp)
    assert np.array_equal(solved, expected), "solver output must match the independently generated truth"
    return True


def _add_noise_teal(grid: np.ndarray, rect_bbox: tuple[int, int, int, int], path_cells: list[tuple[int, int]], red_cells: list[tuple[int, int]]) -> None:
    occupied = {tuple(map(int, cell)) for cell in np.argwhere(grid != BLACK)}
    protected = occupied | _manhattan_halo(set(path_cells) | set(red_cells), grid.shape, radius=1)
    protected |= _manhattan_halo(_rect_cells(rect_bbox), grid.shape, radius=1)

    candidates = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if int(grid[r, c]) == BLACK and (r, c) not in protected
    ]
    random.shuffle(candidates)
    count = min(len(candidates), random.randint(2, 8))
    for r, c in candidates[:count]:
        grid[r, c] = TEAL


def _build_canonical_straight() -> tuple[np.ndarray, np.ndarray]:
    rect_h = random.randint(3, 7)
    rect_w = random.randint(4, 8)
    row_offset = random.randint(1, rect_h - 2)
    left_vis = random.randint(4, 6)
    right_vis = random.randint(3, 6)
    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)

    h = top_pad + rect_h + bottom_pad
    w = left_pad + left_vis + rect_w + right_vis + 2 + right_pad
    expected = np.zeros((h, w), dtype=int)

    r0 = top_pad
    c0 = left_pad + left_vis
    rect_bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    path_r = r0 + row_offset

    last_green_c = c0 + rect_w + right_vis - 1
    green_path = [(path_r, c) for c in range(left_pad, last_green_c + 1)]
    red_cells = [(path_r, last_green_c + 1), (path_r, last_green_c + 2)]

    _paint(expected, green_path, GREEN)
    _paint(expected, red_cells, RED)
    _add_noise_teal(expected, rect_bbox, green_path, red_cells)

    inp = expected.copy()
    rr0, cc0, rr1, cc1 = rect_bbox
    inp[rr0 : rr1 + 1, cc0 : cc1 + 1] = GREY
    return inp, expected


def _build_canonical_turn() -> tuple[np.ndarray, np.ndarray]:
    rect_h = random.randint(3, 7)
    rect_w = random.randint(3, 7)
    turn_row_offset = random.randint(1, rect_h - 2)
    turn_col_offset = random.randint(1, rect_w - 2)
    left_vis = random.randint(4, 6)
    down_vis = random.randint(3, 6)
    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)

    h = top_pad + rect_h + down_vis + 2 + bottom_pad
    w = left_pad + left_vis + rect_w + right_pad
    expected = np.zeros((h, w), dtype=int)

    r0 = top_pad
    c0 = left_pad + left_vis
    rect_bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    turn_r = r0 + turn_row_offset
    turn_c = c0 + turn_col_offset

    last_green_r = rect_bbox[2] + down_vis
    green_path = [(turn_r, c) for c in range(left_pad, turn_c + 1)]
    green_path += [(r, turn_c) for r in range(turn_r + 1, last_green_r + 1)]
    red_cells = [(last_green_r + 1, turn_c), (last_green_r + 2, turn_c)]
    barrier = (turn_r, turn_c + 1)

    _paint(expected, green_path, GREEN)
    expected[barrier] = TEAL
    _paint(expected, red_cells, RED)
    _add_noise_teal(expected, rect_bbox, green_path, red_cells)

    inp = expected.copy()
    rr0, cc0, rr1, cc1 = rect_bbox
    inp[rr0 : rr1 + 1, cc0 : cc1 + 1] = GREY
    return inp, expected


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(400):
        if random.random() < 0.5:
            inp, expected = _build_canonical_straight()
        else:
            inp, expected = _build_canonical_turn()

        op = random.randint(0, 7)
        inp = _dihedral(inp, op)
        expected = _dihedral(expected, op)

        try:
            _validate_case(inp, expected)
        except AssertionError:
            continue
        return inp, expected
    raise RuntimeError("failed to generate a valid ARC case")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
