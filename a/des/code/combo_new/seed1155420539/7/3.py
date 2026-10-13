from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
AZURE = 8
SEED_COLORS = (1, 3, 4, 6, 7, 9)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
REL_ORDER = ("R", "S", "L")
REL_TO_DELTA = {"R": 1, "S": 0, "L": -1}

GRID_BOUNDS = (16, 30)
SEQ_LEN_BOUNDS = (3, 6)
PATH_LEN_BOUNDS = (8, 18)
SEED_FRAME_HEIGHTS = (3, 4)
SIDE_BRANCH_LEN_BOUNDS = (1, 3)
EXTRA_BRANCH_PROB = 0.45
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _dir_index(src: tuple[int, int], dst: tuple[int, int]) -> int:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    for idx, (rr, cc) in enumerate(DIRS):
        if (dr, dc) == (rr, cc):
            return idx
    raise AssertionError("cells are not orthogonally adjacent")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _component_from(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = start
    assert 0 <= sr < h and 0 <= sc < w and bool(mask[sr, sc])
    comp = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in comp:
                comp.add((nr, nc))
                stack.append((nr, nc))
    return comp


def _find_seed_sequence(g: np.ndarray) -> list[int]:
    h, w = g.shape
    candidates: list[tuple[int, list[int]]] = []
    for comp in _components(g == AZURE):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 5:
            continue
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if not touches_corner:
            continue
        if not np.all(g[r0, c0 : c1 + 1] == AZURE):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == AZURE):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == AZURE):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == AZURE):
            continue
        if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == AZURE):
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        best_row = None
        best_count = -1
        for rr in range(interior.shape[0]):
            row = interior[rr]
            count = int(np.count_nonzero(row != BLACK))
            if count > best_count:
                best_count = count
                best_row = rr
        if best_row is None or best_count <= 0:
            continue
        seq = [int(v) for v in interior[best_row] if int(v) != BLACK]
        if len(seq) < 2:
            continue
        candidates.append((len(comp), seq))

    assert len(candidates) == 1, "expected exactly one azure seed frame"
    return candidates[0][1]


def _choose_next(
    current: tuple[int, int],
    prev: tuple[int, int],
    dir_idx: int,
    channel: set[tuple[int, int]],
    visited: set[tuple[int, int]],
) -> tuple[tuple[int, int], int] | None:
    for offset in (1, 0, -1):
        ndir = (dir_idx + offset) % 4
        dr, dc = DIRS[ndir]
        nxt = (current[0] + dr, current[1] + dc)
        if nxt in channel and nxt != prev and nxt not in visited:
            return nxt, ndir
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Grow a repeating color band from the red anchor through the grey-walled channel using the seed-row palette and right-hand routing.

    Input structure:
    - The background is black, and a one-pixel-wide channel network is isolated by grey walls.
    - One red pixel marks the anchor at a channel endpoint.
    - A small azure-framed seed box sits in a corner and contains one horizontal row of palette colors.

    Rule:
    - Read the seed-row colors from left to right as a cyclic coloring sequence.
    - Starting from the red anchor, move into its unique adjacent channel cell and continue through the channel.
    - At each step keep the anchor red, color later channel cells in sequence order, and choose the next direction by right-hand priority: right, then straight, then left.
    - Stop when the channel has no unvisited forward option; keep the seed box and grey walls unchanged.

    Key insight:
    - The azure corner box encodes the repeated color order, while the actual route is determined locally by the anchor direction plus right-hand turn priority inside the channel graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sequence = _find_seed_sequence(g)

    red_pts = np.argwhere(g == RED)
    assert len(red_pts) == 1, "expected one red anchor"
    anchor = tuple(map(int, red_pts[0]))

    passable = (g == BLACK) | (g == RED)
    channel = _component_from(passable, anchor)

    anchor_neighbors = [pos for pos in _neighbors4(*anchor) if pos in channel]
    assert len(anchor_neighbors) == 1, "anchor must be a channel endpoint"

    out = g.copy()
    visited = {anchor}
    prev = anchor
    current = anchor_neighbors[0]
    dir_idx = _dir_index(anchor, current)
    step = 0

    while True:
        out[current] = sequence[step % len(sequence)]
        visited.add(current)

        choice = _choose_next(current, prev, dir_idx, channel, visited)
        if choice is None:
            break
        prev, current = current, choice[0]
        dir_idx = choice[1]
        step += 1

    out[anchor] = RED
    return out


def _seed_bbox(corner: str, height: int, width: int, grid_h: int, grid_w: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, height - 1, width - 1
    if corner == "tr":
        return 0, grid_w - width, height - 1, grid_w - 1
    if corner == "bl":
        return grid_h - height, 0, grid_h - 1, width - 1
    if corner == "br":
        return grid_h - height, grid_w - width, grid_h - 1, grid_w - 1
    raise ValueError(corner)


def _paint_seed_box(
    grid: np.ndarray,
    corner: str,
    sequence: list[int],
    box_h: int,
    seq_row_offset: int,
) -> tuple[int, int, int, int]:
    box_w = len(sequence) + 2
    r0, c0, r1, c1 = _seed_bbox(corner, box_h, box_w, grid.shape[0], grid.shape[1])
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    grid[r0, c0 : c1 + 1] = AZURE
    grid[r1, c0 : c1 + 1] = AZURE
    grid[r0 : r1 + 1, c0] = AZURE
    grid[r0 : r1 + 1, c1] = AZURE
    seq_row = r0 + seq_row_offset
    grid[seq_row, c0 + 1 : c1] = np.array(sequence, dtype=int)
    return r0, c0, r1, c1


def _rel_dir(in_dir: int, rel: str) -> int:
    return (in_dir + REL_TO_DELTA[rel]) % 4


def _valid_new_cell(
    cell: tuple[int, int],
    parent: tuple[int, int],
    allowed: np.ndarray,
    occupied: set[tuple[int, int]],
) -> bool:
    r, c = cell
    h, w = allowed.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if not bool(allowed[r, c]) or cell in occupied:
        return False
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) in occupied and (nr, nc) != parent:
            return False
    return True


def _extend_main_path(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    allowed: np.ndarray,
    seen_rels: set[str],
    target_len: int,
    min_len: int,
) -> list[tuple[int, int]] | None:
    if len(path) == target_len:
        return path.copy() if seen_rels == set(REL_ORDER) else None

    current = path[-1]
    prev = path[-2]
    in_dir = _dir_index(prev, current)
    remaining = target_len - len(path)
    missing = set(REL_ORDER) - seen_rels
    if remaining < len(missing):
        return None

    options = []
    for rel in REL_ORDER:
        ndir = _rel_dir(in_dir, rel)
        dr, dc = DIRS[ndir]
        nxt = (current[0] + dr, current[1] + dc)
        if _valid_new_cell(nxt, current, allowed, occupied):
            priority = 0 if rel in missing else 1
            options.append((priority, random.random(), rel, nxt))

    options.sort()
    for _priority, _rand, rel, nxt in options:
        path.append(nxt)
        occupied.add(nxt)
        result = _extend_main_path(path, occupied, allowed, seen_rels | {rel}, target_len, min_len)
        if result is not None:
            return result
        occupied.remove(nxt)
        path.pop()

    if len(path) >= min_len and seen_rels == set(REL_ORDER):
        return path.copy()
    return None


def _build_main_path(allowed: np.ndarray) -> list[tuple[int, int]] | None:
    free_cells = [tuple(map(int, pt)) for pt in np.argwhere(allowed)]
    random.shuffle(free_cells)
    target_len = random.randint(*PATH_LEN_BOUNDS)
    min_len = max(PATH_LEN_BOUNDS[0], target_len - 4)

    for start in free_cells[:120]:
        dir_order = list(range(4))
        random.shuffle(dir_order)
        for dir_idx in dir_order:
            dr, dc = DIRS[dir_idx]
            nxt = (start[0] + dr, start[1] + dc)
            occupied = {start}
            if not _valid_new_cell(nxt, start, allowed, occupied):
                continue
            path = [start, nxt]
            occupied.add(nxt)
            result = _extend_main_path(path, occupied, allowed, set(), target_len, min_len)
            if result is not None:
                return result
    return None


def _turn_infos(path: list[tuple[int, int]]) -> list[dict[str, int | str]]:
    infos = []
    for i in range(1, len(path) - 1):
        in_dir = _dir_index(path[i - 1], path[i])
        out_dir = _dir_index(path[i], path[i + 1])
        delta = (out_dir - in_dir) % 4
        rel = {1: "R", 0: "S", 3: "L"}[delta]
        infos.append({"index": i, "in_dir": in_dir, "rel": rel})
    return infos


def _add_stub(
    root: tuple[int, int],
    dir_idx: int,
    allowed: np.ndarray,
    occupied: set[tuple[int, int]],
    max_len: int,
) -> list[tuple[int, int]]:
    dr, dc = DIRS[dir_idx]
    branch = []
    prev = root
    current = (root[0] + dr, root[1] + dc)
    for _ in range(max_len):
        if not _valid_new_cell(current, prev, allowed, occupied):
            break
        branch.append(current)
        occupied.add(current)
        prev = current
        current = (current[0] + dr, current[1] + dc)
    return branch


def _add_side_branches(
    path: list[tuple[int, int]],
    allowed: np.ndarray,
) -> set[tuple[int, int]] | None:
    occupied = set(path)
    infos = _turn_infos(path)
    random.shuffle(infos)
    branches_added = 0

    right_nodes = [info for info in infos if info["rel"] == "R"]
    random.shuffle(right_nodes)
    for info in right_nodes[:2]:
        root = path[int(info["index"])]
        in_dir = int(info["in_dir"])
        rel_choices = ["S", "L"]
        random.shuffle(rel_choices)
        for rel in rel_choices:
            branch = _add_stub(
                root,
                _rel_dir(in_dir, rel),
                allowed,
                occupied,
                random.randint(*SIDE_BRANCH_LEN_BOUNDS),
            )
            if branch:
                branches_added += 1
                break
        if branches_added:
            break

    straight_nodes = [info for info in infos if info["rel"] == "S"]
    random.shuffle(straight_nodes)
    for info in straight_nodes[:2]:
        root = path[int(info["index"])]
        branch = _add_stub(
            root,
            _rel_dir(int(info["in_dir"]), "L"),
            allowed,
            occupied,
            random.randint(*SIDE_BRANCH_LEN_BOUNDS),
        )
        if branch:
            branches_added += 1
            break

    random.shuffle(infos)
    for info in infos:
        root = path[int(info["index"])]
        in_dir = int(info["in_dir"])
        rel = str(info["rel"])
        extras = ["S", "L"] if rel == "R" else (["L"] if rel == "S" else [])
        random.shuffle(extras)
        for extra_rel in extras:
            if random.random() >= EXTRA_BRANCH_PROB:
                continue
            branch = _add_stub(
                root,
                _rel_dir(in_dir, extra_rel),
                allowed,
                occupied,
                random.randint(*SIDE_BRANCH_LEN_BOUNDS),
            )
            if branch:
                branches_added += 1

    return occupied if branches_added >= 1 else None


def _reserved_mask(shape: tuple[int, int], seed_box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    allowed = np.zeros((h, w), dtype=bool)
    allowed[2 : h - 2, 2 : w - 2] = True
    r0, c0, r1, c1 = seed_box
    gap = 2
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h - 1, r1 + gap)
    cc1 = min(w - 1, c1 + gap)
    allowed[rr0 : rr1 + 1, cc0 : cc1 + 1] = False
    return allowed


def _render_input(
    shape: tuple[int, int],
    seed_corner: str,
    sequence: list[int],
    box_h: int,
    seq_row_offset: int,
    main_path: list[tuple[int, int]],
    all_channel_cells: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    _paint_seed_box(grid, seed_corner, sequence, box_h, seq_row_offset)

    for r, c in all_channel_cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < shape[0] and 0 <= nc < shape[1] and (nr, nc) not in all_channel_cells and int(grid[nr, nc]) == BLACK:
                grid[nr, nc] = GREY

    for r, c in all_channel_cells:
        grid[r, c] = BLACK
    ar, ac = main_path[0]
    grid[ar, ac] = RED
    return grid


def _expected_output(inp: np.ndarray, main_path: list[tuple[int, int]], sequence: list[int]) -> np.ndarray:
    out = inp.copy()
    for i, (r, c) in enumerate(main_path[1:]):
        out[r, c] = sequence[i % len(sequence)]
    out[main_path[0]] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_BOUNDS)
    w = random.randint(*GRID_BOUNDS)
    seq_len = random.randint(*SEQ_LEN_BOUNDS)
    sequence = random.sample(SEED_COLORS, seq_len)
    seed_corner = random.choice(["tl", "tr", "bl", "br"])
    box_h = random.choice(SEED_FRAME_HEIGHTS)
    seq_row_offset = 1 if box_h == 3 else random.choice([1, 2])

    probe = np.full((h, w), BLACK, dtype=int)
    seed_box = _paint_seed_box(probe, seed_corner, sequence, box_h, seq_row_offset)
    allowed = _reserved_mask((h, w), seed_box)
    if int(np.count_nonzero(allowed)) < PATH_LEN_BOUNDS[0] + 8:
        return None

    main_path = _build_main_path(allowed)
    if main_path is None:
        return None

    channel_cells = _add_side_branches(main_path, allowed)
    if channel_cells is None:
        return None

    inp = _render_input((h, w), seed_corner, sequence, box_h, seq_row_offset, main_path, channel_cells)
    expected = _expected_output(inp, main_path, sequence)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
