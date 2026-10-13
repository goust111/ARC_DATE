import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
Shape = frozenset[Coord]

GREEN = 3
TEAL = 8

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [0, 1, 2, 4, 5, 6, 7, 9]

BASE_LEFT_SHAPE: Shape = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude:
        colors -= set(exclude)
    return sorted(colors)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def left_of(dr: int, dc: int) -> Coord:
    return -dc, dr


def right_of(dr: int, dc: int) -> Coord:
    return dc, -dr


def normalize(cells: set[Coord] | Shape) -> Shape:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def rotate90(cells: set[Coord] | Shape) -> Shape:
    return normalize({(c, -r) for r, c in cells})


def mirror_lr(cells: set[Coord] | Shape) -> Shape:
    return normalize({(r, -c) for r, c in cells})


def rotation_variants(cells: set[Coord] | Shape) -> list[Shape]:
    out: list[Shape] = []
    seen: set[Shape] = set()
    cur = normalize(cells)
    for _ in range(4):
        if cur not in seen:
            out.append(cur)
            seen.add(cur)
        cur = rotate90(cur)
    return out


LEFT_VARIANTS = set(rotation_variants(BASE_LEFT_SHAPE))
RIGHT_VARIANTS = set(rotation_variants(mirror_lr(BASE_LEFT_SHAPE)))


def components_of_mask(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[Coord] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def classify_chirality(cells: list[Coord] | set[Coord] | Shape) -> str:
    shape = normalize(set(cells))
    if shape in LEFT_VARIANTS:
        return "left"
    if shape in RIGHT_VARIANTS:
        return "right"
    raise AssertionError("blocker must be an L tetromino or its mirror")


def find_start_bar(grid: np.ndarray, bg: int) -> tuple[int, Coord, Coord]:
    candidates: list[tuple[int, Coord, Coord]] = []
    h, w = grid.shape
    for color in colors_present(grid, exclude={bg}):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(coords) != 2:
            continue
        (r0, c0), (r1, c1) = coords
        if abs(r0 - r1) + abs(c0 - c1) != 1:
            continue
        border = [p for p in coords if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
        if len(border) != 1:
            continue
        border_cell = border[0]
        inner_cell = coords[1] if coords[0] == border_cell else coords[0]
        candidates.append((int(color), border_cell, inner_cell))
    assert len(candidates) == 1, "expected a unique 2-cell border-touching start bar"
    return candidates[0]


def extract_blockers(grid: np.ndarray, bg: int, start_color: int) -> tuple[int, dict[Coord, str], list[list[Coord]]]:
    blocker_colors = colors_present(grid, exclude={bg, start_color})
    assert len(blocker_colors) == 1, "expected exactly one blocker color"
    blocker_color = blocker_colors[0]
    comps = components_of_mask(grid == blocker_color)
    assert comps, "expected blocker sprites"
    hit_map: dict[Coord, str] = {}
    for comp in comps:
        assert len(comp) == 4, "each blocker must contain four cells"
        chirality = classify_chirality(comp)
        for cell in comp:
            assert cell not in hit_map
            hit_map[cell] = chirality
    return blocker_color, hit_map, comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the ray from the edge-touching green bar, turning left or right according to each hit L blocker’s chirality.

    Input structure:
    - The background is the most common color.
    - Foreground cells consist of one 2-cell green start bar with exactly one border cell and several disjoint 4-cell L-shaped blockers.
    - Each blocker is a rotation or mirror of the same tetromino, so rotations preserve chirality while reflections swap the two chirality classes.

    Rule:
    - Start at the border cell of the green bar, pass through the other green cell, and continue straight into the grid.
    - Paint every traversed background cell green; blockers stay unchanged.
    - Whenever the next cell straight ahead belongs to a blocker, turn at the current cell: the base chirality class turns the ray left relative to travel, and the mirrored chirality class turns it right. Stop when the next step would leave the grid.

    Key insight:
    - Local corridor geometry does not choose the turn: the blocker must be classified up to rotation but not up to reflection, because only mirror changes left into right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    start_color, border_cell, inner_cell = find_start_bar(g, bg)
    blocker_color, hit_map, _ = extract_blockers(g, bg, start_color)

    dr = inner_cell[0] - border_cell[0]
    dc = inner_cell[1] - border_cell[1]
    assert abs(dr) + abs(dc) == 1

    out = g.copy()
    r, c = inner_cell
    seen_states: set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "ray path must be finite and non-cyclic"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(g.shape, nr, nc):
            break

        if (nr, nc) in hit_map:
            turn = hit_map[(nr, nc)]
            if turn == "left":
                dr, dc = left_of(dr, dc)
            else:
                dr, dc = right_of(dr, dc)
            tr, tc = r + dr, c + dc
            assert in_bounds(g.shape, tr, tc), "turn must stay inside the grid"
            assert g[tr, tc] != blocker_color, "turn direction must lead to a walkable cell"
            continue

        out[nr, nc] = start_color
        r, c = nr, nc

    return out


GRID_MIN = 12
GRID_MAX = 20
TURN_MIN = 4
TURN_MAX = 7
SEGMENT_MIN = 1
SEGMENT_MAX = 5
NEW_PATH_MIN = 10
NEW_PATH_MAX = 24
GEN_RETRIES = 600


def steps_to_edge(h: int, w: int, r: int, c: int, dr: int, dc: int) -> int:
    steps = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        steps += 1
        rr += dr
        cc += dc
    return steps


def make_entry(h: int, w: int) -> tuple[Coord, Coord, Coord]:
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        c = random.randint(2, w - 3)
        return (0, c), (1, c), (1, 0)
    if side == "bottom":
        c = random.randint(2, w - 3)
        return (h - 1, c), (h - 2, c), (-1, 0)
    if side == "left":
        r = random.randint(2, h - 3)
        return (r, 0), (r, 1), (0, 1)
    r = random.randint(2, h - 3)
    return (r, w - 1), (r, w - 2), (0, -1)


def random_turn_sequence(n_turns: int) -> list[str]:
    seq = [random.choice(["left", "right"]) for _ in range(n_turns)]
    if "left" not in seq:
        seq[random.randrange(n_turns)] = "left"
    if "right" not in seq:
        seq[random.randrange(n_turns)] = "right"
    return seq


def choose_run_options(
    h: int,
    w: int,
    current: Coord,
    direction: Coord,
    desired_turn: str,
    occupied: set[Coord],
    need_future_turn: bool,
) -> list[int]:
    dr, dc = direction
    max_forward = steps_to_edge(h, w, current[0], current[1], dr, dc)
    options: list[int] = []
    for run_len in range(SEGMENT_MIN, min(SEGMENT_MAX, max_forward - 1) + 1):
        new_cells = [(current[0] + dr * step, current[1] + dc * step) for step in range(1, run_len + 1)]
        if any(cell in occupied for cell in new_cells):
            continue
        p = new_cells[-1]
        hit = (p[0] + dr, p[1] + dc)
        if not in_bounds((h, w), hit[0], hit[1]):
            continue
        left_cell = (p[0] - dc, p[1] + dr)
        right_cell = (p[0] + dc, p[1] - dr)
        if not in_bounds((h, w), left_cell[0], left_cell[1]):
            continue
        if not in_bounds((h, w), right_cell[0], right_cell[1]):
            continue
        if left_cell in occupied or right_cell in occupied:
            continue
        new_dir = left_of(dr, dc) if desired_turn == "left" else right_of(dr, dc)
        next_cell = (p[0] + new_dir[0], p[1] + new_dir[1])
        if next_cell in occupied or not in_bounds((h, w), next_cell[0], next_cell[1]):
            continue
        if need_future_turn and steps_to_edge(h, w, p[0], p[1], new_dir[0], new_dir[1]) < 2:
            continue
        options.append(run_len)
    return options


def build_path_case(h: int, w: int) -> tuple[list[Coord], list[dict[str, object]]] | None:
    border_cell, inner_cell, direction = make_entry(h, w)
    n_turns = random.randint(TURN_MIN, TURN_MAX)
    turn_seq = random_turn_sequence(n_turns)

    path = [border_cell, inner_cell]
    occupied = {border_cell, inner_cell}
    current = inner_cell
    dr, dc = direction
    turns: list[dict[str, object]] = []

    for idx, desired_turn in enumerate(turn_seq):
        options = choose_run_options(
            h,
            w,
            current,
            (dr, dc),
            desired_turn,
            occupied,
            need_future_turn=idx < n_turns - 1,
        )
        if not options:
            return None
        run_len = random.choice(options)
        segment_cells = [(current[0] + dr * step, current[1] + dc * step) for step in range(1, run_len + 1)]
        path.extend(segment_cells)
        occupied.update(segment_cells)
        p = segment_cells[-1]
        turns.append(
            {
                "cell": p,
                "direction": (dr, dc),
                "turn": desired_turn,
                "hit": (p[0] + dr, p[1] + dc),
            }
        )
        dr, dc = left_of(dr, dc) if desired_turn == "left" else right_of(dr, dc)
        current = p

    final_steps = steps_to_edge(h, w, current[0], current[1], dr, dc)
    if final_steps < 1:
        return None
    final_cells = [(current[0] + dr * step, current[1] + dc * step) for step in range(1, final_steps + 1)]
    if any(cell in occupied for cell in final_cells):
        return None
    path.extend(final_cells)

    new_cells = len(path) - 2
    if not (NEW_PATH_MIN <= new_cells <= NEW_PATH_MAX):
        return None
    return path, turns


def blocker_candidate_offsets(chirality: str) -> list[tuple[Shape, Shape]]:
    base = LEFT_VARIANTS if chirality == "left" else RIGHT_VARIANTS
    out: list[tuple[Shape, Shape]] = []
    seen: set[tuple[Shape, Shape]] = set()
    for orient in base:
        for anchor in orient:
            offsets = frozenset((r - anchor[0], c - anchor[1]) for r, c in orient)
            key = (orient, offsets)
            if key not in seen:
                seen.add(key)
                out.append((orient, offsets))
    return out


LEFT_CANDIDATES = blocker_candidate_offsets("left")
RIGHT_CANDIDATES = blocker_candidate_offsets("right")


def blocker_neighbors(cells: set[Coord]) -> set[Coord]:
    halo: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                halo.add((r + dr, c + dc))
    return halo


def candidate_shapes_for_turn(
    h: int,
    w: int,
    turn_info: dict[str, object],
    path_set: set[Coord],
    blocker_cells: set[Coord],
) -> list[tuple[set[Coord], Shape]]:
    p = turn_info["cell"]
    dr, dc = turn_info["direction"]
    hit = turn_info["hit"]
    chirality = turn_info["turn"]
    assert isinstance(p, tuple)
    assert isinstance(hit, tuple)
    templates = LEFT_CANDIDATES if chirality == "left" else RIGHT_CANDIDATES
    lr = left_of(dr, dc)
    rr = right_of(dr, dc)
    forbidden_local = {
        p,
        (p[0] + lr[0], p[1] + lr[1]),
        (p[0] + rr[0], p[1] + rr[1]),
    }
    blocker_halo = blocker_neighbors(blocker_cells)

    out: list[tuple[set[Coord], Shape]] = []
    for orient, offsets in templates:
        cells = {(hit[0] + off_r, hit[1] + off_c) for off_r, off_c in offsets}
        if any(not in_bounds((h, w), r, c) for r, c in cells):
            continue
        if cells & path_set:
            continue
        if cells & forbidden_local:
            continue
        if cells & blocker_halo:
            continue
        out.append((cells, orient))
    return out


def place_blockers(
    h: int,
    w: int,
    turns: list[dict[str, object]],
    path_set: set[Coord],
) -> tuple[list[set[Coord]], list[Shape]] | None:
    chosen: list[set[Coord] | None] = [None] * len(turns)
    orientations: list[Shape | None] = [None] * len(turns)
    chirality_orients = {"left": set(), "right": set()}
    global_orients: set[Shape] = set()

    def backtrack(done: set[int], blocker_cells: set[Coord]) -> bool:
        if len(done) == len(turns):
            return True
        order = []
        for idx, info in enumerate(turns):
            if idx in done:
                continue
            candidates = candidate_shapes_for_turn(h, w, info, path_set, blocker_cells)
            if not candidates:
                return False
            order.append((len(candidates), idx, candidates))
        order.sort(key=lambda item: item[0])
        _, idx, candidates = order[0]
        info = turns[idx]
        chirality = info["turn"]

        def score(item: tuple[set[Coord], Shape]) -> tuple[int, int, float]:
            _cells, orient = item
            return (
                0 if orient not in chirality_orients[chirality] else 1,
                0 if orient not in global_orients else 1,
                random.random(),
            )

        for cells, orient in sorted(candidates, key=score):
            chosen[idx] = cells
            orientations[idx] = orient
            prev_chiral = set(chirality_orients[chirality])
            prev_global = set(global_orients)
            chirality_orients[chirality].add(orient)
            global_orients.add(orient)
            if backtrack(done | {idx}, blocker_cells | cells):
                return True
            chosen[idx] = None
            orientations[idx] = None
            chirality_orients[chirality] = prev_chiral
            global_orients.clear()
            global_orients.update(prev_global)
        return False

    if not backtrack(set(), set()):
        return None
    return [set(cells) for cells in chosen if cells is not None], [orient for orient in orientations if orient is not None]


def paint_expected_output(inp: np.ndarray, path: list[Coord]) -> np.ndarray:
    out = inp.copy()
    for r, c in path:
        out[r, c] = GREEN
    return out


def trace_path_states(grid: np.ndarray) -> tuple[list[Coord], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    start_color, border_cell, inner_cell = find_start_bar(g, bg)
    blocker_color, hit_map, _ = extract_blockers(g, bg, start_color)

    dr = inner_cell[0] - border_cell[0]
    dc = inner_cell[1] - border_cell[1]
    r, c = inner_cell
    path = [border_cell, inner_cell]
    turns: list[dict[str, object]] = []
    seen_states: set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states
        seen_states.add(state)
        nr, nc = r + dr, c + dc
        if not in_bounds(g.shape, nr, nc):
            break
        if (nr, nc) in hit_map:
            turn = hit_map[(nr, nc)]
            turns.append({"cell": (r, c), "direction": (dr, dc), "turn": turn, "hit": (nr, nc)})
            if turn == "left":
                dr, dc = left_of(dr, dc)
            else:
                dr, dc = right_of(dr, dc)
            continue
        assert g[nr, nc] != blocker_color
        path.append((nr, nc))
        r, c = nr, nc
    return path, turns


def validate_case(inp: np.ndarray, expected_path: list[Coord], expected_turns: list[dict[str, object]]) -> None:
    out = main(inp)
    expected = paint_expected_output(inp, expected_path)
    assert np.array_equal(out, expected), "main() must match the independently constructed path"

    path2, turns2 = trace_path_states(inp)
    assert path2 == expected_path
    assert len(turns2) == len(expected_turns)
    assert [t["turn"] for t in turns2] == [t["turn"] for t in expected_turns]

    bg = get_bg(inp)
    start_color, _border, _inner = find_start_bar(inp, bg)
    blocker_color, _hit_map, comps = extract_blockers(inp, bg, start_color)
    assert start_color == GREEN
    assert blocker_color == TEAL

    chirality_counts = {"left": 0, "right": 0}
    orientation_sets = {"left": set(), "right": set()}
    for comp in comps:
        chirality = classify_chirality(comp)
        chirality_counts[chirality] += 1
        orientation_sets[chirality].add(normalize(set(comp)))
    assert chirality_counts["left"] >= 1 and chirality_counts["right"] >= 1
    assert sum(len(v) for v in orientation_sets.values()) >= 3
    assert max(len(orientation_sets["left"]), len(orientation_sets["right"])) >= 2

    blocker_cells = {cell for comp in comps for cell in comp}
    for info in turns2:
        p = info["cell"]
        dr, dc = info["direction"]
        left_cell = (p[0] - dc, p[1] + dr)
        right_cell = (p[0] + dc, p[1] - dr)
        assert in_bounds(inp.shape, left_cell[0], left_cell[1])
        assert in_bounds(inp.shape, right_cell[0], right_cell[1])
        assert left_cell not in blocker_cells
        assert right_cell not in blocker_cells

    new_green = int(np.count_nonzero((out == GREEN) & (inp != GREEN)))
    assert NEW_PATH_MIN <= new_green <= NEW_PATH_MAX


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice(PALETTE)
    path_case = build_path_case(h, w)
    if path_case is None:
        return None
    path, turns = path_case
    path_set = set(path)
    placed = place_blockers(h, w, turns, path_set)
    if placed is None:
        return None
    blockers, _orientations = placed

    grid = np.full((h, w), bg, dtype=int)
    grid[path[0]] = GREEN
    grid[path[1]] = GREEN
    for cells in blockers:
        for r, c in cells:
            grid[r, c] = TEAL

    try:
        validate_case(grid, path, turns)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chirality-turning ARC input")
