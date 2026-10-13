import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
ORANGE = 7

SPRITE_COLOR_PAIRS = [
    (2, 4),
    (3, 6),
    (9, 8),
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _lex_key(cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    return int(c), int(r)


def _separator_indices(grid: np.ndarray, axis: int) -> list[int]:
    n = grid.shape[axis]
    out = []
    for idx in range(n):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if bool(np.all(line == BLACK)):
            out.append(int(idx))
    return out


def _regular_intervals(separators: list[int], size: int) -> list[tuple[int, int]]:
    cuts = [-1] + list(separators) + [int(size)]
    intervals = []
    for left, right in zip(cuts, cuts[1:]):
        if right - left <= 1:
            continue
        intervals.append((int(left + 1), int(right)))
    assert len(intervals) >= 2, "expected at least a 2x2 block grid"
    lengths = {int(b - a) for a, b in intervals}
    assert len(lengths) == 1, "block grid must have uniform block size"
    return intervals


def _block_signature(block: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in block.ravel())


def _infer_sprite(blocks: list[np.ndarray]) -> np.ndarray:
    mixed = [b for b in blocks if len(np.unique(b)) > 1]
    assert mixed, "expected at least one sprite-coded wall block"
    counts = Counter(_block_signature(b) for b in mixed)
    sprite_sig, _ = counts.most_common(1)[0]
    sprite = np.array(sprite_sig, dtype=int).reshape(mixed[0].shape)
    for block in mixed:
        assert np.array_equal(block, sprite), "all wall blocks must share one repeated sprite"
    return sprite


def _infer_blank_color(sprite: np.ndarray) -> int:
    counts = Counter(int(x) for x in sprite.ravel())
    assert len(counts) == 2, "sprite must use exactly two visible colors"
    ordered = counts.most_common()
    assert ordered[0][1] != ordered[1][1], "sprite colors must have distinct frequencies"
    return int(ordered[-1][0])


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    row_intervals = _regular_intervals(_separator_indices(g, axis=0), g.shape[0])
    col_intervals = _regular_intervals(_separator_indices(g, axis=1), g.shape[1])

    block_shapes = {(r1 - r0, c1 - c0) for r0, r1 in row_intervals for c0, c1 in col_intervals}
    assert len(block_shapes) == 1
    block_h, block_w = next(iter(block_shapes))
    assert block_h == block_w, "blocks must be square"

    blocks = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            blocks.append(g[r0:r1, c0:c1])
    sprite = _infer_sprite(blocks)
    blank_color = _infer_blank_color(sprite)

    macro_h = len(row_intervals)
    macro_w = len(col_intervals)
    passable = np.zeros((macro_h, macro_w), dtype=bool)
    kinds = np.full((macro_h, macro_w), "wall", dtype=object)
    start = None
    goal = None

    for r, (r0, r1) in enumerate(row_intervals):
        for c, (c0, c1) in enumerate(col_intervals):
            block = g[r0:r1, c0:c1]
            uniq = np.unique(block)
            if len(uniq) > 1:
                assert np.array_equal(block, sprite), "mixed-color blocks must be the repeated wall sprite"
                continue

            color = int(uniq[0])
            if color == BLUE:
                assert start is None, "expected one mover block"
                start = (int(r), int(c))
                passable[r, c] = True
                kinds[r, c] = "start"
            elif color == ORANGE:
                assert goal is None, "expected one destination block"
                goal = (int(r), int(c))
                passable[r, c] = True
                kinds[r, c] = "goal"
            else:
                assert color == blank_color, "uniform non-marker blocks must use the blank style"
                passable[r, c] = True
                kinds[r, c] = "blank"

    assert start is not None and goal is not None and start != goal
    return {
        "grid": g,
        "row_intervals": row_intervals,
        "col_intervals": col_intervals,
        "block_size": int(block_h),
        "sprite": sprite,
        "blank_color": int(blank_color),
        "passable": passable,
        "kinds": kinds,
        "start": start,
        "goal": goal,
    }


def _distance_from_goal(passable: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    h, w = passable.shape
    dist = -np.ones((h, w), dtype=int)
    q = deque([goal])
    dist[goal] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = int(dist[r, c] + 1)
                q.append((int(nr), int(nc)))
    return dist


def _lexicographic_shortest_path(
    passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    dist = _distance_from_goal(passable, goal)
    assert dist[start] >= 0, "mover must be able to reach the destination through blank blocks"
    path = [start]
    cur = start
    while cur != goal:
        r, c = cur
        candidates = []
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < passable.shape[0] and 0 <= nc < passable.shape[1]:
                if passable[nr, nc] and dist[nr, nc] == dist[r, c] - 1:
                    candidates.append((int(nr), int(nc)))
        assert candidates, "shortest-path reconstruction failed"
        cur = min(candidates, key=_lex_key)
        path.append(cur)
    return path


def _fill_block(out: np.ndarray, row_intervals, col_intervals, cell: tuple[int, int], patch: np.ndarray) -> None:
    r, c = cell
    r0, r1 = row_intervals[r]
    c0, c1 = col_intervals[c]
    out[r0:r1, c0:c1] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse a sprite-vs-blank block code while exempting the mover's shortest corridor to the target.

    Input structure:
    - The grid is partitioned by black separator rows and columns into equal square blocks.
    - Every block is either one repeated two-color sprite (a wall), one uniform blank-style block, one uniform blue mover block, or one uniform orange target block.
    - The mover and target occupy blocks that otherwise belong to the blank/passable state.

    Rule:
    - Treat sprite blocks as walls and uniform blank, blue, and orange blocks as passable macro-cells.
    - Find the 4-neighbor shortest path from mover to target in block coordinates; among equal-length paths, compare each path as a sequence of block coordinates ordered left-to-right then top-to-bottom and keep the lexicographically smallest one.
    - Replace the mover block with the blank style and the target block with a blue block.
    - Invert the encoding everywhere else: sprite blocks become uniform blank blocks, while non-path blank-style blocks become the repeated sprite.
    - Keep every separator line black, and exempt every block on the chosen path from inversion so the whole path stays blank except for the blue target endpoint.

    Key insight:
    - The sprite/blank code serves two roles at once: it defines the maze for pathfinding and it is the code that gets globally inverted, so the chosen shortest path survives as the only untouched tunnel through the reversed maze.
    """
    parsed = _parse_grid(np.asarray(grid, dtype=int))
    path = _lexicographic_shortest_path(parsed["passable"], parsed["start"], parsed["goal"])
    path_set = set(path)

    blank_patch = np.full((parsed["block_size"], parsed["block_size"]), parsed["blank_color"], dtype=int)
    blue_patch = np.full((parsed["block_size"], parsed["block_size"]), BLUE, dtype=int)
    out = np.zeros_like(parsed["grid"])

    macro_h, macro_w = parsed["passable"].shape
    for r in range(macro_h):
        for c in range(macro_w):
            cell = (int(r), int(c))
            if cell in path_set:
                patch = blue_patch if cell == parsed["goal"] else blank_patch
            else:
                patch = blank_patch if parsed["kinds"][r, c] == "wall" else parsed["sprite"]
            _fill_block(out, parsed["row_intervals"], parsed["col_intervals"], cell, patch)

    return out


def _choose_sprite(block_size: int, wall_color: int, blank_color: int) -> np.ndarray:
    mask_bank = []
    diag = np.eye(block_size, dtype=bool)
    mask_bank.append(diag)
    mask_bank.append(np.fliplr(diag))

    center_col = np.zeros((block_size, block_size), dtype=bool)
    center_col[:, block_size // 2] = True
    mask_bank.append(center_col)

    center_row = np.zeros((block_size, block_size), dtype=bool)
    center_row[block_size // 2, :] = True
    mask_bank.append(center_row)

    corners = np.zeros((block_size, block_size), dtype=bool)
    corners[0, 0] = True
    corners[0, -1] = True
    corners[-1, 0] = True
    corners[-1, -1] = True
    mask_bank.append(corners)

    mask = random.choice(mask_bank)
    sprite = np.full((block_size, block_size), int(wall_color), dtype=int)
    sprite[mask] = int(blank_color)
    return sprite


def _render_input(
    passable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    sprite: np.ndarray,
    blank_color: int,
) -> np.ndarray:
    block_size = int(sprite.shape[0])
    macro_h, macro_w = passable.shape
    height = macro_h * block_size + (macro_h - 1)
    width = macro_w * block_size + (macro_w - 1)
    out = np.zeros((height, width), dtype=int)
    blank_patch = np.full((block_size, block_size), int(blank_color), dtype=int)
    blue_patch = np.full((block_size, block_size), BLUE, dtype=int)
    orange_patch = np.full((block_size, block_size), ORANGE, dtype=int)

    for r in range(macro_h):
        for c in range(macro_w):
            r0 = r * (block_size + 1)
            c0 = c * (block_size + 1)
            if (r, c) == start:
                patch = blue_patch
            elif (r, c) == goal:
                patch = orange_patch
            elif passable[r, c]:
                patch = blank_patch
            else:
                patch = sprite
            out[r0 : r0 + block_size, c0 : c0 + block_size] = patch
    return out


def _render_output(
    passable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    sprite: np.ndarray,
    blank_color: int,
) -> np.ndarray:
    path = _lexicographic_shortest_path(passable, start, goal)
    path_set = set(path)
    block_size = int(sprite.shape[0])
    macro_h, macro_w = passable.shape
    height = macro_h * block_size + (macro_h - 1)
    width = macro_w * block_size + (macro_w - 1)
    out = np.zeros((height, width), dtype=int)
    blank_patch = np.full((block_size, block_size), int(blank_color), dtype=int)
    blue_patch = np.full((block_size, block_size), BLUE, dtype=int)

    for r in range(macro_h):
        for c in range(macro_w):
            r0 = r * (block_size + 1)
            c0 = c * (block_size + 1)
            if (r, c) in path_set:
                patch = blue_patch if (r, c) == goal else blank_patch
            else:
                patch = blank_patch if not passable[r, c] else sprite
            out[r0 : r0 + block_size, c0 : c0 + block_size] = patch
    return out


def _count_shortest_paths(passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int], limit: int = 3) -> int:
    dist = _distance_from_goal(passable, goal)
    if dist[start] < 0:
        return 0
    nodes = [(int(dist[r, c]), int(r), int(c)) for r in range(passable.shape[0]) for c in range(passable.shape[1]) if dist[r, c] >= 0]
    nodes.sort()
    counts = np.zeros_like(dist, dtype=int)
    counts[goal] = 1
    for d, r, c in nodes[1:]:
        total = 0
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < passable.shape[0] and 0 <= nc < passable.shape[1]:
                if passable[nr, nc] and dist[nr, nc] == d - 1:
                    total += int(counts[nr, nc])
        counts[r, c] = min(int(limit), int(total))
    return int(counts[start])


def _enumerate_shortest_paths(
    passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int], limit: int = 2
) -> list[list[tuple[int, int]]]:
    dist = _distance_from_goal(passable, goal)
    if dist[start] < 0:
        return []
    paths = []

    def dfs(cur: tuple[int, int], path: list[tuple[int, int]]) -> None:
        if len(paths) >= limit:
            return
        if cur == goal:
            paths.append(path.copy())
            return
        r, c = cur
        candidates = []
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < passable.shape[0] and 0 <= nc < passable.shape[1]:
                if passable[nr, nc] and dist[nr, nc] == dist[r, c] - 1:
                    candidates.append((int(nr), int(nc)))
        candidates.sort(key=_lex_key)
        for nxt in candidates:
            path.append(nxt)
            dfs(nxt, path)
            path.pop()

    dfs(start, [start])
    return paths


def _add_spurs(passable: np.ndarray, attempts: int = 20) -> int:
    added = 0
    for _ in range(attempts):
        passable_cells = list(zip(*np.where(passable)))
        if not passable_cells:
            break
        origin = random.choice(passable_cells)
        dirs = list(_neighbors4(0, 0))
        random.shuffle(dirs)
        for dr, dc in dirs:
            length = random.randint(1, 2)
            cells = []
            pr, pc = origin
            ok = True
            for _step in range(length):
                rr = int(pr + dr)
                cc = int(pc + dc)
                if not (0 <= rr < passable.shape[0] and 0 <= cc < passable.shape[1]) or passable[rr, cc]:
                    ok = False
                    break
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < passable.shape[0] and 0 <= nc < passable.shape[1]):
                        continue
                    if (nr, nc) == (pr, pc):
                        continue
                    if passable[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
                cells.append((rr, cc))
                pr, pc = rr, cc
            if ok and cells:
                for rr, cc in cells:
                    passable[rr, cc] = True
                added += len(cells)
                break
    return added


def _snake_path(macro_h: int, macro_w: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    horizontal = random.random() < 0.5
    h, w = (macro_h, macro_w) if horizontal else (macro_w, macro_h)

    start_r = random.randrange(h)
    goal_r = random.randrange(h)
    max_turns = min(4, max(1, w - 2))
    turn_count = random.randint(1, max_turns)
    turn_cols = sorted(random.sample(range(1, w - 1), turn_count))

    rows = [start_r]
    for _ in range(turn_count - 1):
        choices = [r for r in range(h) if r != rows[-1]]
        rows.append(random.choice(choices))
    if goal_r == rows[-1]:
        goal_r = (goal_r + random.randint(1, h - 1)) % h
    rows.append(goal_r)

    coords = [(start_r, 0)]
    cur_r, cur_c = start_r, 0
    for idx, turn_c in enumerate(turn_cols):
        step = 1 if turn_c >= cur_c else -1
        for c in range(cur_c + step, turn_c + step, step):
            coords.append((cur_r, int(c)))
        next_r = rows[idx + 1]
        if next_r != cur_r:
            step_r = 1 if next_r > cur_r else -1
            for r in range(cur_r + step_r, next_r + step_r, step_r):
                coords.append((int(r), turn_c))
        cur_r, cur_c = next_r, turn_c

    for c in range(cur_c + 1, w):
        coords.append((cur_r, int(c)))

    passable = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        passable[r, c] = True

    start = coords[0]
    goal = coords[-1]
    if horizontal:
        return passable, start, goal

    transposed = passable.T.copy()
    return transposed, (start[1], start[0]), (goal[1], goal[0])


def _extend_from_corner(
    passable: np.ndarray, endpoint: tuple[int, int], corner: tuple[int, int], anchor: tuple[int, int]
) -> tuple[int, int]:
    local_r = endpoint[0] - anchor[0]
    local_c = endpoint[1] - anchor[1]
    outward = {
        (0, 0): [(-1, 0), (0, -1)],
        (0, 2): [(-1, 0), (0, 1)],
        (2, 0): [(1, 0), (0, -1)],
        (2, 2): [(1, 0), (0, 1)],
    }[(local_r, local_c)]
    random.shuffle(outward)
    for dr, dc in outward:
        length = random.randint(0, 2)
        cur = endpoint
        ok = True
        cells = []
        for _ in range(length):
            rr = int(cur[0] + dr)
            cc = int(cur[1] + dc)
            if not (0 <= rr < passable.shape[0] and 0 <= cc < passable.shape[1]) or passable[rr, cc]:
                ok = False
                break
            for nr, nc in _neighbors4(rr, cc):
                if not (0 <= nr < passable.shape[0] and 0 <= nc < passable.shape[1]):
                    continue
                if (nr, nc) == cur:
                    continue
                if passable[nr, nc]:
                    ok = False
                    break
            if not ok:
                break
            cells.append((rr, cc))
            cur = (rr, cc)
        if ok:
            for rr, cc in cells:
                passable[rr, cc] = True
            return cur
    return endpoint


def _ring_choice(macro_h: int, macro_w: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    passable = np.zeros((macro_h, macro_w), dtype=bool)
    anchor_r = random.randint(0, macro_h - 3)
    anchor_c = random.randint(0, macro_w - 3)
    anchor = (anchor_r, anchor_c)
    for dr in range(3):
        for dc in range(3):
            if (dr, dc) != (1, 1):
                passable[anchor_r + dr, anchor_c + dc] = True

    corner_pairs = [
        ((0, 0), (2, 2)),
        ((0, 2), (2, 0)),
        ((2, 0), (0, 2)),
        ((2, 2), (0, 0)),
    ]
    s_local, g_local = random.choice(corner_pairs)
    start = (anchor_r + s_local[0], anchor_c + s_local[1])
    goal = (anchor_r + g_local[0], anchor_c + g_local[1])
    start = _extend_from_corner(passable, start, s_local, anchor)
    goal = _extend_from_corner(passable, goal, g_local, anchor)
    return passable, start, goal


def _non_path_blank_count(passable: np.ndarray, path: list[tuple[int, int]]) -> int:
    return int(passable.sum() - len(set(path)))


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    passable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    mode: str,
) -> bool:
    parsed = _parse_grid(grid)
    if not np.array_equal(parsed["passable"], passable):
        return False
    if parsed["start"] != start or parsed["goal"] != goal:
        return False

    chosen_path = _lexicographic_shortest_path(passable, start, goal)
    if len(chosen_path) < 4:
        return False
    if _non_path_blank_count(passable, chosen_path) < 1:
        return False
    if int((~passable).sum()) < 2:
        return False

    shortest_count = _count_shortest_paths(passable, start, goal)
    if mode == "tie":
        if shortest_count < 2:
            return False
        alts = _enumerate_shortest_paths(passable, start, goal, limit=2)
        if len(alts) < 2:
            return False
        if alts[0] != chosen_path:
            return False
        alt_path = alts[1]
        if set(alt_path) == set(chosen_path):
            return False
    else:
        if shortest_count != 1:
            return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    block_size = random.choice([3, 4])
    max_dim = min(7, 31 // (block_size + 1))
    macro_h = random.randint(4, max_dim)
    macro_w = random.randint(4, max_dim)

    wall_color, blank_color = random.choice(SPRITE_COLOR_PAIRS)
    sprite = _choose_sprite(block_size, wall_color, blank_color)

    mode = "tie" if random.random() < 0.45 else "unique"
    if mode == "tie":
        passable, start, goal = _ring_choice(macro_h, macro_w)
    else:
        passable, start, goal = _snake_path(macro_h, macro_w)

    added = _add_spurs(passable, attempts=random.randint(4, 10))
    if added == 0:
        return None

    if start == goal:
        return None
    if not passable[start] or not passable[goal]:
        return None

    expected = _render_output(passable, start, goal, sprite, blank_color)
    grid = _render_input(passable, start, goal, sprite, blank_color)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _validate_instance(grid, expected, passable, start, goal, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-exemption inversion puzzle")
