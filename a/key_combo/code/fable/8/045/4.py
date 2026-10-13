import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
PATTERNS = [
    ("G", "E"),
    ("E", "G"),
    ("G", "E", "G"),
    ("E", "G", "E"),
]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_corner(r: int, c: int, h: int, w: int) -> bool:
    return (r in {0, h - 1}) and (c in {0, w - 1})


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            out.append(comp)
    return out


def _parse_launcher(grid: np.ndarray) -> tuple[list[int], tuple[int, int], tuple[int, int]]:
    seq_mask = (grid != BG) & (grid != GREY)
    comps = _components8(seq_mask)
    assert len(comps) == 1, "expected exactly one colored launcher sequence"
    cells = {tuple(map(int, p)) for p in comps[0]}
    assert len(cells) >= 3

    endpoints = []
    for r, c in cells:
        diag_neighbors = sum((r + dr, c + dc) in cells for dr, dc in DIAGONALS)
        assert diag_neighbors in {1, 2}, "launcher must be a single diagonal chain"
        if diag_neighbors == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2

    candidates = []
    for start, end in ((endpoints[0], endpoints[1]), (endpoints[1], endpoints[0])):
        dr = 1 if end[0] > start[0] else -1
        dc = 1 if end[1] > start[1] else -1
        length = abs(end[0] - start[0]) + 1
        if abs(end[0] - start[0]) != abs(end[1] - start[1]):
            continue
        ordered = [(start[0] + i * dr, start[1] + i * dc) for i in range(length)]
        if set(ordered) != cells:
            continue
        pr, pc = start[0] - dr, start[1] - dc
        ar, ac = end[0] + dr, end[1] + dc
        blocked_before = (not _in_bounds(grid.shape, pr, pc)) or int(grid[pr, pc]) == GREY
        open_after = _in_bounds(grid.shape, ar, ac) and int(grid[ar, ac]) == BG
        if blocked_before and open_after:
            candidates.append((ordered, end, (dr, dc)))
    assert len(candidates) == 1, "launcher direction must be uniquely readable"
    ordered, launch_end, direction = candidates[0]
    seq = [int(grid[r, c]) for r, c in ordered]
    return seq, launch_end, direction


def _step_after_reflection(
    shape: tuple[int, int],
    grey_mask: np.ndarray,
    pos: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], bool, bool]:
    r, c = pos
    dr, dc = direction
    nr, nc = r + dr, c + dc
    hit_grey = _in_bounds(shape, nr, nc) and bool(grey_mask[nr, nc])
    if _in_bounds(shape, nr, nc) and not hit_grey:
        return (nr, nc), direction, False, False

    if hit_grey:
        side_r_block = _in_bounds(shape, r + dr, c) and bool(grey_mask[r + dr, c])
        side_c_block = _in_bounds(shape, r, c + dc) and bool(grey_mask[r, c + dc])
        if not side_r_block and not side_c_block:
            side_r_block = True
            side_c_block = True
        new_dr = -dr if side_r_block else dr
        new_dc = -dc if side_c_block else dc
        nxt = (r + new_dr, c + new_dc)
        assert _in_bounds(shape, nxt[0], nxt[1]) and not grey_mask[nxt]
        return nxt, (new_dr, new_dc), True, False

    side_r_block = not _in_bounds(shape, r + dr, c)
    side_c_block = not _in_bounds(shape, r, c + dc)
    new_dr = -dr if side_r_block else dr
    new_dc = -dc if side_c_block else dc
    nxt = (r + new_dr, c + new_dc)
    assert _in_bounds(shape, nxt[0], nxt[1])
    return nxt, (new_dr, new_dc), False, True


def _simulate_with_sequence(
    grid: np.ndarray,
    seq: list[int],
    launch_end: tuple[int, int],
    direction: tuple[int, int],
    *,
    toggle_rewind: bool = True,
) -> tuple[np.ndarray, dict]:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    grey_mask = out == GREY
    current = launch_end
    idx = 0
    idx_step = 1
    seen: set[tuple[tuple[int, int], tuple[int, int], int, int]] = set()
    path: list[tuple[int, int]] = []
    grey_hits = 0
    edge_hits = 0
    reverse_steps = 0

    while not _is_corner(current[0], current[1], h, w):
        state = (current, direction, idx, idx_step)
        assert state not in seen, "ribbon entered a loop instead of reaching a corner"
        seen.add(state)

        nxt, direction, hit_grey, hit_edge = _step_after_reflection(out.shape, grey_mask, current, direction)
        if hit_grey:
            grey_hits += 1
            if toggle_rewind:
                idx_step *= -1
        if hit_edge:
            edge_hits += 1
        current = nxt
        out[current] = seq[idx]
        path.append(current)
        reverse_steps += int(idx_step == -1)
        idx = (idx + idx_step) % len(seq)

    return out, {
        "path": path,
        "grey_hits": grey_hits,
        "edge_hits": edge_hits,
        "reverse_steps": reverse_steps,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch the diagonal color cycle, mirror it off edges, and rewind its order whenever it bounces off grey reflector blocks.

    Input structure:
    - The background is black.
    - One multi-color 45-degree diagonal chain is the launcher sequence; its blocked end is against a grey block or the grid boundary, and its open end is the launch tip.
    - One or more grey solid blocks sit elsewhere as reflectors.

    Rule:
    - Read the launcher colors from the blocked end to the open end; the open end emits the ribbon along the same diagonal direction.
    - Each newly visited cell is colored by cycling through that sequence; boundary bounces mirror the diagonal direction but do not reset the color phase.
    - If the next diagonal step would enter a grey block, reflect as if off the contacted block face and, from that bounced step onward, continue the same phase in the reversed cyclic order; a later grey bounce flips the order back again.
    - Stop as soon as the ribbon reaches a grid corner, and keep the grey blocks unchanged.

    Key insight:
    - Geometry alone is insufficient: edge mirrors and grey reflectors bend the path similarly, but only grey bounces toggle whether the ongoing color cycle is read forward or backward.
    """
    g = np.asarray(grid, dtype=int)
    seq, launch_end, direction = _parse_launcher(g)
    out, _meta = _simulate_with_sequence(g, seq, launch_end, direction, toggle_rewind=True)
    return out


def _compatible_launcher_starts(h: int, w: int, direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    dr, dc = direction
    starts = []
    for r in range(h):
        for c in range(w):
            pr, pc = r - dr, c - dc
            if _in_bounds((h, w), pr, pc):
                continue
            coords = [(r + i * dr, c + i * dc) for i in range(length + 1)]
            if not all(_in_bounds((h, w), rr, cc) for rr, cc in coords):
                continue
            if _is_corner(r, c, h, w):
                continue
            starts.append((r, c))
    random.shuffle(starts)
    return starts


def _travel_cells(pos: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = direction
    return [(r + i * dr, c + i * dc) for i in range(1, steps + 1)]


def _distances_to_edges(pos: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    r, c = pos
    dr, dc = direction
    row_dist = (h - 1 - r) if dr > 0 else r
    col_dist = (w - 1 - c) if dc > 0 else c
    return row_dist, col_dist


def _boundary_event(
    pos: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    row_dist, col_dist = _distances_to_edges(pos, direction, h, w)
    if row_dist == col_dist:
        return None
    steps = min(row_dist, col_dist)
    cells = _travel_cells(pos, direction, steps)
    if any(cell in reserved for cell in cells):
        return None
    dr, dc = direction
    if row_dist < col_dist:
        new_direction = (-dr, dc)
    else:
        new_direction = (dr, -dc)
    end_pos = pos if steps == 0 else cells[-1]
    return end_pos, new_direction, cells


def _grey_candidates(
    pos: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], set[tuple[int, int]]]]:
    row_dist, col_dist = _distances_to_edges(pos, direction, h, w)
    max_steps = min(row_dist, col_dist)
    dr, dc = direction
    candidates = []
    for steps in range(1, max_steps):
        prefix = _travel_cells(pos, direction, steps)
        if any(cell in reserved for cell in prefix):
            continue
        q = prefix[-1]
        target = (q[0] + dr, q[1] + dc)
        side_r = (q[0] + dr, q[1])
        side_c = (q[0], q[1] + dc)

        for axis, block, other, new_direction in (
            ("r", {side_r, target}, side_c, (-dr, dc)),
            ("c", {side_c, target}, side_r, (dr, -dc)),
        ):
            if not all(_in_bounds((h, w), rr, cc) for rr, cc in block):
                continue
            if not _in_bounds((h, w), other[0], other[1]):
                continue
            if other in reserved or other in block:
                continue
            if block & reserved:
                continue
            nxt = (q[0] + new_direction[0], q[1] + new_direction[1])
            if not _in_bounds((h, w), nxt[0], nxt[1]) or nxt in reserved or nxt in block:
                continue
            candidates.append((q, new_direction, prefix, set(block)))
    random.shuffle(candidates)
    return candidates


def _final_corner_cells(
    pos: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    row_dist, col_dist = _distances_to_edges(pos, direction, h, w)
    if row_dist != col_dist or row_dist == 0:
        return None
    cells = _travel_cells(pos, direction, row_dist)
    if any(cell in reserved for cell in cells):
        return None
    return cells


def _search_pattern(
    pos: tuple[int, int],
    direction: tuple[int, int],
    pattern: tuple[str, ...],
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    if not pattern:
        final_cells = _final_corner_cells(pos, direction, h, w, reserved)
        if final_cells is None:
            return None
        return final_cells, set()

    token = pattern[0]
    if token == "E":
        event = _boundary_event(pos, direction, h, w, reserved)
        if event is None:
            return None
        new_pos, new_direction, cells = event
        result = _search_pattern(new_pos, new_direction, pattern[1:], h, w, reserved | set(cells))
        if result is None:
            return None
        suffix_cells, grey_cells = result
        return cells + suffix_cells, grey_cells

    candidates = _grey_candidates(pos, direction, h, w, reserved)
    for q, new_direction, prefix, block in candidates[:24]:
        result = _search_pattern(q, new_direction, pattern[1:], h, w, reserved | set(prefix) | block)
        if result is None:
            continue
        suffix_cells, grey_cells = result
        return prefix + suffix_cells, block | grey_cells
    return None


def _build_input_attempt() -> np.ndarray | None:
    h = random.randint(10, 18)
    w = random.randint(10, 18)
    length = random.randint(3, 5)
    direction = random.choice(DIAGONALS)
    starts = _compatible_launcher_starts(h, w, direction, length)
    if not starts:
        return None
    start = random.choice(starts)
    seq_colors = random.sample(PALETTE, length)

    grid = np.zeros((h, w), dtype=int)
    launcher_cells = []
    for i, color in enumerate(seq_colors):
        r = start[0] + i * direction[0]
        c = start[1] + i * direction[1]
        grid[r, c] = color
        launcher_cells.append((r, c))
    launch_end = launcher_cells[-1]

    reserved = set(launcher_cells)
    patterns = PATTERNS[:]
    random.shuffle(patterns)
    solution = None
    for pattern in patterns:
        result = _search_pattern(launch_end, direction, pattern, h, w, reserved)
        if result is not None:
            path_cells, grey_cells = result
            solution = (pattern, path_cells, grey_cells)
            break
    if solution is None:
        return None

    pattern, path_cells, grey_cells = solution
    for r, c in grey_cells:
        grid[r, c] = GREY

    try:
        parsed_seq, parsed_launch, parsed_direction = _parse_launcher(grid)
        out, meta = _simulate_with_sequence(grid, parsed_seq, parsed_launch, parsed_direction, toggle_rewind=True)
        no_rewind, _ = _simulate_with_sequence(grid, parsed_seq, parsed_launch, parsed_direction, toggle_rewind=False)
    except AssertionError:
        return None

    if parsed_seq != seq_colors or parsed_launch != launch_end or parsed_direction != direction:
        return None
    if set(meta["path"]) != set(path_cells):
        return None
    if len(meta["path"]) != len(path_cells):
        return None
    if meta["grey_hits"] != pattern.count("G"):
        return None
    if meta["edge_hits"] < pattern.count("E"):
        return None
    if meta["grey_hits"] < 1 or meta["edge_hits"] < 1:
        return None
    if np.array_equal(out, no_rewind):
        return None
    if np.array_equal(out, grid):
        return None
    if len(set(meta["path"])) != len(meta["path"]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rewind-ribbon input")
