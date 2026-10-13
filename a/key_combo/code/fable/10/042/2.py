import random

import numpy as np


BG = 0
WALL = 5
TAIL = 8
HEAD_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIR4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
MAX_GENERATE_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _rot_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _rot_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _advance(cell: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * steps, cell[1] + direction[1] * steps


def _edge_steps(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    steps = 0
    r, c = start
    while True:
        r += direction[0]
        c += direction[1]
        if not _in_bounds(shape, r, c):
            return steps
        steps += 1


def _free_steps(shape: tuple[int, int], blocked: set[tuple[int, int]], start: tuple[int, int], direction: tuple[int, int]) -> int:
    steps = 0
    r, c = start
    while True:
        r += direction[0]
        c += direction[1]
        if not _in_bounds(shape, r, c) or (r, c) in blocked:
            return steps
        steps += 1


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    used_heads: set[tuple[int, int]] = set()
    tails = [tuple(map(int, rc)) for rc in np.argwhere(grid == TAIL)]
    assert tails, "expected at least one cyan tail pixel"

    arrow_cells: set[tuple[int, int]] = set()
    for tail in tails:
        neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(*tail)
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in (BG, WALL)
        ]
        assert len(neighbors) == 1, "each cyan tail must touch exactly one non-wall foreground head"
        head = neighbors[0]
        assert head not in used_heads, "two tails cannot share one head"
        head_color = int(grid[head])
        assert head_color not in (BG, WALL, TAIL), "head pixel must be a non-wall non-cyan color"

        back_neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(*head)
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in (BG, WALL)
        ]
        assert back_neighbors == [tail] or sorted(back_neighbors) == sorted([tail]), "arrow object must be exactly two pixels"

        direction = (head[0] - tail[0], head[1] - tail[1])
        arrows.append({"tail": tail, "head": head, "color": head_color, "direction": direction})
        used_heads.add(head)
        arrow_cells.add(tail)
        arrow_cells.add(head)

    observed_fg = {
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if int(grid[r, c]) not in (BG, WALL)
    }
    assert observed_fg == arrow_cells, "all non-wall foreground pixels must belong to arrows"
    arrows.sort(key=lambda item: (item["head"][0], item["head"][1], item["tail"][0], item["tail"][1]))
    return arrows


def _measure_run(grid: np.ndarray, painted: set[tuple[int, int]], start: tuple[int, int], direction: tuple[int, int]) -> int:
    steps = 0
    r, c = start
    while True:
        r += direction[0]
        c += direction[1]
        if not _in_bounds(grid.shape, r, c):
            return steps
        if int(grid[r, c]) != BG or (r, c) in painted:
            return steps
        steps += 1


def _trace_arrow(
    grid: np.ndarray,
    head: tuple[int, int],
    color: int,
    direction: tuple[int, int],
    painted: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[dict]]:
    cur = head
    path: list[tuple[int, int]] = []
    bends: list[dict] = []

    for _ in range(grid.size * 4):
        nxt = _advance(cur, direction)
        if not _in_bounds(grid.shape, *nxt):
            return path, bends

        cell = int(grid[nxt])
        if cell == BG:
            assert nxt not in painted, "rays must not touch each other"
            painted.add(nxt)
            path.append(nxt)
            cur = nxt
            continue

        if cell == WALL:
            left_dir = _rot_left(direction)
            right_dir = _rot_right(direction)
            left_space = _measure_run(grid, painted, cur, left_dir)
            right_space = _measure_run(grid, painted, cur, right_dir)
            assert left_space != right_space, "wall turns must be uniquely decided by unequal free runs"
            chosen_dir = left_dir if left_space > right_space else right_dir
            chosen_turn = "L" if chosen_dir == left_dir else "R"
            assert max(left_space, right_space) > 0, "wall turn must leave at least one free background cell"
            bends.append(
                {
                    "at": cur,
                    "hit": nxt,
                    "incoming": direction,
                    "turn": chosen_turn,
                    "left_space": left_space,
                    "right_space": right_space,
                }
            )
            direction = chosen_dir
            continue

        raise AssertionError("ray may only encounter background or wall cells")

    raise AssertionError("ray failed to terminate at the edge")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each two-pixel arrow into a colored ray that folds at gray walls.

    Input structure:
    - The background is black (0), and gray (5) cells form wall barriers.
    - Every arrow is a 4-connected two-pixel object made of one cyan tail pixel (8)
      and one colored head pixel; the tail-to-head orientation gives the ray direction.
    - Apart from walls, all non-background pixels belong to such arrow markers.

    Rule:
    - From each head pixel, extend a ray in the tail-to-head direction and paint only
      background cells with the head color.
    - When the next step would leave the grid, the ray stops.
    - When the next step is a wall, do not enter the wall; instead turn left or right
      by 90 degrees and continue along the side with the longer contiguous background run.
      The construction guarantees those two runs are unequal at every turn.
    - Walls and arrow pixels stay unchanged, and different rays never touch.

    Key insight:
    - A wall is not an endpoint but a hinge: each arrow supplies an initial direction,
      then every collision locally chooses the more spacious side, producing a folded wire.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    arrows = _find_arrows(g)

    out = g.copy()
    painted: set[tuple[int, int]] = set()
    for arrow in arrows:
        path, _bends = _trace_arrow(g, arrow["head"], arrow["color"], arrow["direction"], painted)
        for r, c in path:
            out[r, c] = int(arrow["color"])
    return out


def _build_after_bend(
    shape: tuple[int, int],
    current: tuple[int, int],
    incoming_dir: tuple[int, int],
    bends_left: int,
    blocked: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[dict], list[str]] | None:
    left_dir = _rot_left(incoming_dir)
    right_dir = _rot_right(incoming_dir)
    turn_options = [("L", left_dir, right_dir), ("R", right_dir, left_dir)]
    random.shuffle(turn_options)

    for turn_label, next_dir, other_dir in turn_options:
        other_space = _free_steps(shape, blocked, current, other_dir)
        next_space = _free_steps(shape, blocked, current, next_dir)

        if bends_left == 1:
            if next_space != _edge_steps(shape, current, next_dir):
                continue
            if next_space <= other_space or next_space < 1:
                continue
            segment = [_advance(current, next_dir, step) for step in range(1, next_space + 1)]
            if any(cell in forbidden for cell in segment):
                continue
            return segment, [], [turn_label]

        max_len = next_space - 1
        min_len = max(1, other_space + 1)
        if max_len < min_len:
            continue

        candidate_lengths = list(range(min_len, max_len + 1))
        random.shuffle(candidate_lengths)
        candidate_lengths.sort(reverse=bool(random.getrandbits(1)))

        for seg_len in candidate_lengths:
            segment = [_advance(current, next_dir, step) for step in range(1, seg_len + 1)]
            next_wall = _advance(current, next_dir, seg_len + 1)
            if any(cell in forbidden for cell in segment) or next_wall in forbidden:
                continue
            new_blocked = blocked | set(segment) | {next_wall}
            child = _build_after_bend(
                shape=shape,
                current=segment[-1],
                incoming_dir=next_dir,
                bends_left=bends_left - 1,
                blocked=new_blocked,
                forbidden=forbidden,
            )
            if child is None:
                continue
            child_path, child_walls, child_turns = child
            wall_plan = {"hit": next_wall, "incoming": next_dir}
            return segment + child_path, [wall_plan] + child_walls, [turn_label] + child_turns

    return None


def _plan_arrow(
    shape: tuple[int, int],
    blocked_global: set[tuple[int, int]],
    forbidden_global: set[tuple[int, int]],
    color: int,
    bends: int,
) -> dict | None:
    h, w = shape
    for _ in range(250):
        direction = random.choice(DIR4)
        head = (random.randrange(h), random.randrange(w))
        tail = _advance(head, (-direction[0], -direction[1]))
        if not _in_bounds(shape, *tail):
            continue
        if head in forbidden_global or tail in forbidden_global:
            continue

        blocked = set(blocked_global)
        blocked.add(head)
        blocked.add(tail)

        if bends == 0:
            final_len = _free_steps(shape, blocked, head, direction)
            if final_len != _edge_steps(shape, head, direction) or final_len < 2:
                continue
            path = [_advance(head, direction, step) for step in range(1, final_len + 1)]
            if any(cell in forbidden_global for cell in path):
                continue
            return {
                "tail": tail,
                "head": head,
                "color": color,
                "path": path,
                "walls": [],
                "turns": [],
            }

        forward_space = _free_steps(shape, blocked, head, direction)
        max_initial_len = forward_space - 1
        if max_initial_len < 1:
            continue

        initial_lengths = list(range(1, max_initial_len + 1))
        random.shuffle(initial_lengths)
        initial_lengths.sort(reverse=bool(random.getrandbits(1)))

        for initial_len in initial_lengths:
            initial_path = [_advance(head, direction, step) for step in range(1, initial_len + 1)]
            first_wall = _advance(head, direction, initial_len + 1)
            if any(cell in forbidden_global for cell in initial_path) or first_wall in forbidden_global:
                continue
            start_blocked = blocked | set(initial_path) | {first_wall}
            child = _build_after_bend(
                shape=shape,
                current=initial_path[-1],
                incoming_dir=direction,
                bends_left=bends,
                blocked=start_blocked,
                forbidden=forbidden_global,
            )
            if child is None:
                continue
            child_path, child_walls, child_turns = child
            return {
                "tail": tail,
                "head": head,
                "color": color,
                "path": initial_path + child_path,
                "walls": [{"hit": first_wall, "incoming": direction}] + child_walls,
                "turns": child_turns,
            }

    return None


def _expand_walls(shape: tuple[int, int], records: list[dict]) -> set[tuple[int, int]]:
    arrow_cells = {record["tail"] for record in records} | {record["head"] for record in records}
    path_cells = {cell for record in records for cell in record["path"]}
    wall_cells = {wall["hit"] for record in records for wall in record["walls"]}
    occupied = set(arrow_cells) | set(path_cells) | set(wall_cells)

    for wall in [item for record in records for item in record["walls"]]:
        hit = wall["hit"]
        incoming = wall["incoming"]
        axis_dirs = [(-1, 0), (1, 0)] if incoming[0] == 0 else [(0, -1), (0, 1)]
        random.shuffle(axis_dirs)

        added = 0
        for axis in axis_dirs:
            max_take = 0
            for dist in range(1, 3):
                cell = _advance(hit, axis, dist)
                if not _in_bounds(shape, *cell) or cell in occupied:
                    break
                max_take += 1
            take = random.randint(0, max_take)
            for dist in range(1, take + 1):
                cell = _advance(hit, axis, dist)
                wall_cells.add(cell)
                occupied.add(cell)
                added += 1

        if added == 0:
            for axis in axis_dirs:
                cell = _advance(hit, axis, 1)
                if _in_bounds(shape, *cell) and cell not in occupied:
                    wall_cells.add(cell)
                    occupied.add(cell)
                    break

    return wall_cells


def _buffer_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    buffered = set(cells)
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    buffered.add((rr, cc))
    return buffered


def _analyze(grid: np.ndarray) -> list[dict]:
    arrows = _find_arrows(grid)
    painted: set[tuple[int, int]] = set()
    analysis: list[dict] = []
    for arrow in arrows:
        path, bends = _trace_arrow(grid, arrow["head"], arrow["color"], arrow["direction"], painted)
        analysis.append(
            {
                "tail": arrow["tail"],
                "head": arrow["head"],
                "color": arrow["color"],
                "path": path,
                "bends": bends,
            }
        )
    return analysis


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze(grid)
        out = main(grid)
    except Exception:
        return False

    if len(analysis) < 2:
        return False

    bend_counts = [len(item["bends"]) for item in analysis]
    if max(bend_counts) < 2:
        return False
    if not any(count <= 1 for count in bend_counts):
        return False

    turn_set = {bend["turn"] for item in analysis for bend in item["bends"]}
    if sum(bend_counts) >= 2 and turn_set != {"L", "R"}:
        return False

    total_path = sum(len(item["path"]) for item in analysis)
    if total_path < 10:
        return False

    if np.array_equal(out, grid):
        return False

    diff_cells = [tuple(map(int, rc)) for rc in np.argwhere(out != grid)]
    if not diff_cells:
        return False
    for r, c in diff_cells:
        if int(grid[r, c]) != BG:
            return False
        if int(out[r, c]) in (BG, WALL, TAIL):
            return False

    outputs = {tuple(item["path"]) for item in analysis}
    if len(outputs) != len(analysis):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    arrow_count = random.randint(2, 3)
    colors = random.sample(HEAD_COLORS, arrow_count)

    bend_counts = [random.choice((2, 3)), random.choice((0, 1))]
    if arrow_count == 3:
        bend_counts.append(random.choice((1, 2)))

    records: list[dict] = []
    blocked_global: set[tuple[int, int]] = set()
    forbidden_global: set[tuple[int, int]] = set()

    for color, bends in zip(colors, bend_counts):
        record = _plan_arrow(
            shape=(h, w),
            blocked_global=blocked_global,
            forbidden_global=forbidden_global,
            color=color,
            bends=bends,
        )
        if record is None:
            return None
        records.append(record)

        actual_cells = {record["tail"], record["head"]} | set(record["path"]) | {wall["hit"] for wall in record["walls"]}
        blocked_global |= actual_cells
        forbidden_global |= _buffer_cells((h, w), actual_cells)

    wall_cells = _expand_walls((h, w), records)
    grid = np.zeros((h, w), dtype=int)
    for r, c in wall_cells:
        grid[r, c] = WALL
    for record in records:
        grid[record["tail"]] = TAIL
        grid[record["head"]] = record["color"]

    if not _semantic_ok(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folded-wall wire instance")
