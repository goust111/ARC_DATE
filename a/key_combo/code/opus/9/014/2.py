from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREEN = 3
TEAL = 8
NOISE_PALETTE = (1, 4, 5, 6, 7, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ENTRY_SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_start_bar(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    greens = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    assert len(greens) == 2, "expected exactly one 2-pixel green start bar"
    (r0, c0), (r1, c1) = greens
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "start bar must be 4-connected"

    h, w = g.shape

    def is_border(cell: tuple[int, int]) -> bool:
        r, c = cell
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    border_cells = [cell for cell in greens if is_border(cell)]
    assert len(border_cells) == 1, "exactly one green pixel must touch the border"
    border = border_cells[0]
    inner = greens[1] if greens[0] == border else greens[0]
    dr = inner[0] - border[0]
    dc = inner[1] - border[1]
    return border, inner, (int(dr), int(dc))


def _find_red_block(g: np.ndarray) -> tuple[int, int]:
    reds = np.argwhere(g == RED)
    assert reds.shape[0] == 4, "expected exactly one 2x2 red marker block"
    r0, c0 = reds.min(axis=0)
    r1, c1 = reds.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert (r1 - r0, c1 - c0) == (1, 1), "red marker must be a 2x2 block"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == RED)
    return r0, c0


def _trace_path(
    g: np.ndarray,
    border: tuple[int, int],
    inner: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = inner
    path = [border, inner]
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "path walk should not cycle"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(g.shape, nr, nc):
            break

        if int(g[nr, nc]) != TEAL:
            r, c = int(nr), int(nc)
            path.append((r, c))
            continue

        turns: list[tuple[int, int]] = []
        for tdr, tdc in ((-dc, dr), (dc, -dr)):
            tr, tc = r + tdr, c + tdc
            if in_bounds(g.shape, tr, tc) and int(g[tr, tc]) != TEAL:
                turns.append((int(tdr), int(tdc)))
        assert len(turns) == 1, "each blocker collision must have one valid perpendicular turn"
        dr, dc = turns[0]

    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the green corridor path, then erase a red-block cross while sparing traced path pixels.

    Input structure:
    - A black background is mixed with ordinary colored noise pixels, one 2x2 red marker block, teal blocker pixels, and a 2-pixel green start bar with exactly one border-touching pixel.
    - The teal blockers force a single one-cell-wide ray route that starts at the border green pixel, passes through the other green pixel, and turns only when the next cell is teal.
    - The red marker block does not overlap the route, but the route shares at least one of the block's two rows or two columns.

    Rule:
    - Starting from the border green pixel, continue through the other green pixel in the same direction.
    - Whenever the next cell is not teal, move into it and treat it as part of the route; whenever the next cell is teal, turn to the unique traversable perpendicular direction; stop when the next step would leave the grid.
    - Paint every visited route cell green. Then use the 2x2 red block's two rows and two columns as a cross-shaped erase region.
    - Inside that cross, reset every pixel that is neither a traced green path pixel, nor a teal blocker, nor a red marker pixel to black; outside the cross, keep everything else unchanged.

    Key insight:
    - The protected pixels are defined dynamically by the traced path positions, so the red-block cross can only be resolved after the corridor walk has been computed in the same coordinate frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    red_r0, red_c0 = _find_red_block(g)
    border, inner, direction = _find_start_bar(g)
    path = _trace_path(g, border, inner, direction)
    path_set = set(path)

    out = g.copy()
    for r, c in path:
        out[r, c] = GREEN

    cross_rows = {red_r0, red_r0 + 1}
    cross_cols = {red_c0, red_c0 + 1}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if r not in cross_rows and c not in cross_cols:
                continue
            if (r, c) in path_set or int(g[r, c]) in (TEAL, RED):
                continue
            out[r, c] = BG
    return out


def _choose_entry_bar(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    side = random.choice(ENTRY_SIDES)
    if side == "top":
        c = random.randint(1, w - 2)
        return (0, c), (1, c), (1, 0)
    if side == "bottom":
        c = random.randint(1, w - 2)
        return (h - 1, c), (h - 2, c), (-1, 0)
    if side == "left":
        r = random.randint(1, h - 2)
        return (r, 0), (r, 1), (0, 1)
    r = random.randint(1, h - 2)
    return (r, w - 1), (r, w - 2), (0, -1)


def _max_forward(r: int, c: int, dr: int, dc: int, h: int, w: int) -> int:
    steps = 0
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return steps
        steps += 1


def _extend_segment(
    path_cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    r: int,
    c: int,
    dr: int,
    dc: int,
    seg_len: int | None,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    steps = 0
    while True:
        if seg_len is not None and steps == seg_len:
            return r, c
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return (r, c) if seg_len is None else None
        if (nr, nc) in occupied:
            return None
        path_cells.append((nr, nc))
        occupied.add((nr, nc))
        r, c = nr, nc
        steps += 1


def _place_turn_blockers(
    blockers: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    r: int,
    c: int,
    dr: int,
    dc: int,
    remaining_turns: int,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    ahead = (r + dr, c + dc)
    if ahead in occupied or not in_bounds((h, w), ahead[0], ahead[1]):
        return None

    options = [(-dc, dr), (dc, -dr)]
    random.shuffle(options)
    for tdr, tdc in options:
        tr, tc = r + tdr, c + tdc
        wr, wc = r - tdr, c - tdc
        if not in_bounds((h, w), tr, tc):
            continue
        if (tr, tc) in occupied or (tr, tc) in blockers:
            continue
        forward_room = _max_forward(r, c, tdr, tdc, h, w)
        if forward_room < 1:
            continue
        if remaining_turns > 0 and forward_room < 2:
            continue
        if in_bounds((h, w), wr, wc):
            if (wr, wc) in occupied or (wr, wc) in blockers:
                continue
            blockers.add((wr, wc))
        blockers.add(ahead)
        return int(tdr), int(tdc)
    return None


def _build_path_spec() -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int], int] | None:
    turns = random.randint(1, 4)
    min_dim = max(12, 6 + turns * 2)
    h = random.randint(min_dim, 18)
    w = random.randint(min_dim, 18)
    grid = np.full((h, w), BG, dtype=int)

    border, inner, direction = _choose_entry_bar(h, w)
    dr, dc = direction
    r, c = inner

    occupied = {border, inner}
    blockers: set[tuple[int, int]] = set()
    path_cells: list[tuple[int, int]] = []

    for seg_i in range(turns + 1):
        final_segment = seg_i == turns
        if final_segment:
            pos = _extend_segment(path_cells, occupied, r, c, dr, dc, None, h, w)
            if pos is None:
                return None
            r, c = pos
            break

        max_fwd = _max_forward(r, c, dr, dc, h, w)
        if max_fwd < 2:
            return None
        max_seg = min(5, max_fwd - 1)
        min_seg = 1 if seg_i > 0 else 2
        if max_seg < min_seg:
            return None
        seg_len = random.randint(min_seg, max_seg)
        pos = _extend_segment(path_cells, occupied, r, c, dr, dc, seg_len, h, w)
        if pos is None:
            return None
        r, c = pos

        turn_dir = _place_turn_blockers(blockers, occupied, r, c, dr, dc, turns - seg_i - 1, h, w)
        if turn_dir is None:
            return None
        dr, dc = turn_dir

    path = [border, inner] + path_cells
    if len(path) < 9:
        return None

    for cell in blockers:
        grid[cell] = TEAL
    grid[border] = GREEN
    grid[inner] = GREEN
    return grid, path, border, turns


def _all_free_block_positions(
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            cells = {(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)}
            if cells & forbidden:
                continue
            candidates.append((r0, c0))
    return candidates


def _choose_red_block(
    shape: tuple[int, int],
    path: Iterable[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    path_list = list(path)
    candidates = []
    for r0, c0 in _all_free_block_positions(shape, forbidden):
        rows = {r0, r0 + 1}
        cols = {c0, c0 + 1}
        in_cross = [(r, c) for r, c in path_list if r in rows or c in cols]
        if not (2 <= len(in_cross) <= len(path_list) - 2):
            continue
        row_hits = sum(r in rows for r, _ in path_list)
        col_hits = sum(c in cols for _, c in path_list)
        if row_hits == 0 and col_hits == 0:
            continue
        candidates.append((len(in_cross), abs(row_hits - col_hits), random.random(), r0, c0))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, _, _, r0, c0 = candidates[0]
    return r0, c0


def _scatter_extra_blockers(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    count: int,
) -> None:
    h, w = grid.shape
    eligible = []
    for r in range(h):
        for c in range(w):
            if (r, c) in reserved or grid[r, c] != BG:
                continue
            if any((r + dr, c + dc) in reserved for dr, dc in DIRS4 if in_bounds(grid.shape, r + dr, c + dc)):
                continue
            eligible.append((r, c))
    random.shuffle(eligible)
    for r, c in eligible[:count]:
        grid[r, c] = TEAL


def _expected_output(
    inp: np.ndarray,
    path: list[tuple[int, int]],
    red_r0: int,
    red_c0: int,
) -> np.ndarray:
    out = inp.copy()
    path_set = set(path)
    for r, c in path:
        out[r, c] = GREEN
    rows = {red_r0, red_r0 + 1}
    cols = {red_c0, red_c0 + 1}
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            if r not in rows and c not in cols:
                continue
            if (r, c) in path_set or int(inp[r, c]) in (RED, TEAL):
                continue
            out[r, c] = BG
    return out


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    path: list[tuple[int, int]],
    border: tuple[int, int],
    red_r0: int,
    red_c0: int,
    turns: int,
) -> bool:
    rows = {red_r0, red_r0 + 1}
    cols = {red_c0, red_c0 + 1}
    path_set = set(path)
    start_pixels = {border, path[1]}

    protected_inside = [
        (r, c)
        for r, c in path
        if (r in rows or c in cols) and (r, c) not in start_pixels and int(inp[r, c]) not in (GREEN, RED, TEAL)
    ]
    erased_inside = [
        (r, c)
        for r in range(inp.shape[0])
        for c in range(inp.shape[1])
        if (r in rows or c in cols)
        and (r, c) not in path_set
        and int(inp[r, c]) not in (BG, RED, TEAL, GREEN)
        and int(out[r, c]) == BG
    ]
    path_outside = [(r, c) for r, c in path if r not in rows and c not in cols]
    changed_to_green = [(r, c) for r, c in path if int(inp[r, c]) != GREEN and int(out[r, c]) == GREEN]

    return (
        turns >= 1
        and len(protected_inside) >= 2
        and len(erased_inside) >= 3
        and len(path_outside) >= 3
        and len(changed_to_green) >= 5
        and np.count_nonzero(inp == RED) == 4
        and np.count_nonzero(inp == GREEN) == 2
    )


def _generate_candidate() -> np.ndarray | None:
    built = _build_path_spec()
    if built is None:
        return None
    grid, path, border, turns = built

    reserved = set(path) | set(map(tuple, np.argwhere(grid == TEAL)))
    red_block = _choose_red_block(grid.shape, path, reserved)
    if red_block is None:
        return None
    red_r0, red_c0 = red_block
    red_cells = {(red_r0, red_c0), (red_r0 + 1, red_c0), (red_r0, red_c0 + 1), (red_r0 + 1, red_c0 + 1)}

    inp = grid.copy()
    noise_colors = random.sample(NOISE_PALETTE, k=random.randint(3, 5))
    fill_prob = random.uniform(0.30, 0.40)

    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            if (r, c) in red_cells or int(inp[r, c]) in (GREEN, TEAL):
                continue
            if random.random() < fill_prob:
                inp[r, c] = random.choice(noise_colors)

    for r, c in red_cells:
        inp[r, c] = RED

    reserved_all = reserved | red_cells | set(path[:2])
    _scatter_extra_blockers(inp, reserved_all, random.randint(0, 3))

    expected = _expected_output(inp, path, red_r0, red_c0)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(inp, actual, path, border, red_r0, red_c0, turns):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
