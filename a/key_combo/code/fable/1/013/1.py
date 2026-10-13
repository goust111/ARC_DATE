from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREEN = 3
PINK = 6
CYAN = 8
PALETTE_COLORS = (1, 2, 4, 5, 7, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CORNERS = (
    ((0, 0), (1, 1)),
    ((0, -1), (1, -1)),
    ((-1, 0), (-1, 1)),
    ((-1, -1), (-1, -1)),
)
GEN_RETRY_LIMIT = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _single_coord(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    assert coords.shape == (1, 2), "expected exactly one matching cell"
    return int(coords[0, 0]), int(coords[0, 1])


def _first_non_bg(
    grid: np.ndarray, start: tuple[int, int], d_idx: int, bg: int
) -> tuple[int, int] | None:
    r, c = start
    dr, dc = DIRS[int(d_idx)]
    while True:
        r += dr
        c += dc
        if not _in_bounds(grid.shape, r, c):
            return None
        if int(grid[r, c]) != bg:
            return int(r), int(c)


def _read_palette(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    palette_mask = ~np.isin(grid, [bg, GREEN, PINK, CYAN])
    coords = np.argwhere(palette_mask)
    assert coords.size > 0, "expected diagonal palette markers"

    palette_cells = {(int(r), int(c)) for r, c in coords}
    palette_colors = [int(grid[r, c]) for r, c in coords]
    assert len(set(palette_colors)) == len(palette_colors), "palette colors must be unique"
    for color in palette_colors:
        assert int(np.sum(grid == color)) == 1, "palette colors must be singleton markers"

    h, w = grid.shape
    candidates = []
    for (cr, cc), (dr, dc) in CORNERS:
        r = 0 if cr == 0 else h - 1
        c = 0 if cc == 0 else w - 1
        if (r, c) not in palette_cells:
            continue
        seq = []
        rr, cc2 = r, c
        while _in_bounds(grid.shape, rr, cc2) and (rr, cc2) in palette_cells:
            seq.append((int(rr), int(cc2), int(grid[rr, cc2])))
            rr += dr
            cc2 += dc
        if {(sr, sc) for sr, sc, _ in seq} == palette_cells:
            candidates.append(seq)

    assert len(candidates) == 1, "expected one corner-anchored diagonal palette"
    return candidates[0]


def _initial_direction(grid: np.ndarray, start: tuple[int, int], bg: int) -> int:
    candidates = []
    for d_idx in range(4):
        first = _first_non_bg(grid, start, d_idx, bg)
        if first is None:
            continue
        if int(grid[first]) in (PINK, CYAN):
            candidates.append(int(d_idx))
    assert len(candidates) == 1, "expected exactly one visible initial control direction"
    return candidates[0]


def _trace_ray(grid: np.ndarray) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    start = _single_coord(g == GREEN)
    d_idx = _initial_direction(g, start, bg)

    r, c = start
    visited: list[tuple[int, int]] = []
    path_cells = {start}
    seen_states = set()

    while True:
        state = (int(r), int(c), int(d_idx))
        assert state not in seen_states, "ray must eventually leave the grid"
        seen_states.add(state)

        dr, dc = DIRS[d_idx]
        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            break

        value = int(g[nr, nc])
        if value == PINK:
            visited.append((int(nr), int(nc)))
            d_idx = (d_idx + 1) % 4
            continue
        if value == CYAN:
            visited.append((int(nr), int(nc)))
            d_idx = (d_idx - 1) % 4
            continue

        assert value == bg, "ray path cells must stay background"
        r, c = int(nr), int(nc)
        path_cells.add((r, c))

    assert visited, "expected at least one visited control"
    return visited, path_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the control pixels hit by an invisible turning ray using the corner diagonal palette order.

    Input structure:
    - The background is black, with exactly one green start pixel.
    - Pink and cyan singleton pixels act as turn controls; only some of them are
      encountered by the ray and the rest are distractors.
    - A corner contains a singleton diagonal palette whose colors are all unique
      and distinct from green, pink, and cyan.

    Rule:
    - Find the unique cardinal direction from the green start whose first visible
      non-background pixel is a control.
    - Move an invisible ray cell by cell in that direction; if the next cell is
      pink, turn right at the current cell, and if the next cell is cyan, turn
      left at the current cell. The control cell itself is never occupied.
    - Record the visited controls in encounter order, read the palette colors
      from the corner inward along the diagonal, and recolor the k-th visited
      control with the k-th palette color.

    Key insight:
    - The ray itself leaves no trace, so the task is to recover the hidden visit
      order of turn controls and use the diagonal corner markers as an ordered palette.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    palette = _read_palette(g, bg)
    visited, _path_cells = _trace_ray(g)

    assert len(visited) <= len(palette), "palette must cover every visited control"
    out = g.copy()
    for idx, (r, c) in enumerate(visited):
        out[r, c] = int(palette[idx][2])
    return out


def _palette_positions(corner_idx: int, length: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if corner_idx == 0:
        return [(i, i) for i in range(length)]
    if corner_idx == 1:
        return [(i, w - 1 - i) for i in range(length)]
    if corner_idx == 2:
        return [(h - 1 - i, i) for i in range(length)]
    return [(h - 1 - i, w - 1 - i) for i in range(length)]


def _sample_turns(count: int) -> list[str]:
    turns = [random.choice(("R", "L")) for _ in range(count)]
    if count >= 2 and random.random() < 0.8 and len(set(turns)) == 1:
        turns[random.randrange(count)] = "L" if turns[0] == "R" else "R"
    return turns


def _free_run(
    shape: tuple[int, int],
    start: tuple[int, int],
    d_idx: int,
    occupied_path: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = DIRS[d_idx]
    cells = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cell = (int(r), int(c))
        if cell in occupied_path or cell in blocked:
            return []
        cells.append(cell)


def _extend_path(
    shape: tuple[int, int],
    current: tuple[int, int],
    d_idx: int,
    turns: list[str],
    turn_index: int,
    occupied_path: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[dict] | None:
    if turn_index == len(turns):
        final_cells = _free_run(shape, current, d_idx, occupied_path, blocked)
        if len(final_cells) >= 1:
            return []
        return None

    dr, dc = DIRS[d_idx]
    max_len = 0
    r, c = current
    while _in_bounds(shape, r + (max_len + 2) * dr, c + (max_len + 2) * dc):
        max_len += 1
    if max_len < 1:
        return None

    lengths = list(range(1, max_len + 1))
    random.shuffle(lengths)
    preferred_min = 2 if max_len >= 2 and random.random() < 0.7 else 1

    for length in lengths:
        if length < preferred_min:
            continue
        seg_cells = [(int(r + step * dr), int(c + step * dc)) for step in range(1, length + 1)]
        control = (int(r + (length + 1) * dr), int(c + (length + 1) * dc))
        if any(cell in occupied_path or cell in blocked for cell in seg_cells):
            continue
        if control in occupied_path or control in blocked:
            continue

        next_d_idx = (d_idx + 1) % 4 if turns[turn_index] == "R" else (d_idx - 1) % 4
        new_current = seg_cells[-1]
        sub = _extend_path(
            shape,
            new_current,
            next_d_idx,
            turns,
            turn_index + 1,
            occupied_path | set(seg_cells),
            blocked | {control},
        )
        if sub is None:
            continue
        color = PINK if turns[turn_index] == "R" else CYAN
        head = {
            "control": control,
            "color": color,
            "turn_cell": new_current,
            "d_idx": d_idx,
        }
        return [head] + sub

    if preferred_min == 2:
        for length in lengths:
            if length != 1:
                continue
            seg_cells = [(int(r + dr), int(c + dc))]
            control = (int(r + 2 * dr), int(c + 2 * dc))
            if seg_cells[0] in occupied_path or seg_cells[0] in blocked:
                continue
            if control in occupied_path or control in blocked:
                continue
            next_d_idx = (d_idx + 1) % 4 if turns[turn_index] == "R" else (d_idx - 1) % 4
            sub = _extend_path(
                shape,
                seg_cells[-1],
                next_d_idx,
                turns,
                turn_index + 1,
                occupied_path | set(seg_cells),
                blocked | {control},
            )
            if sub is None:
                continue
            color = PINK if turns[turn_index] == "R" else CYAN
            head = {
                "control": control,
                "color": color,
                "turn_cell": seg_cells[-1],
                "d_idx": d_idx,
            }
            return [head] + sub

    return None


def _start_candidates(
    shape: tuple[int, int], corner_idx: int, palette_len: int
) -> list[tuple[int, int]]:
    h, w = shape
    candidates = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if corner_idx == 0 and (r <= palette_len + 1 or c <= palette_len + 1):
                continue
            if corner_idx == 1 and (r <= palette_len + 1 or c >= w - palette_len - 2):
                continue
            if corner_idx == 2 and (r >= h - palette_len - 2 or c <= palette_len + 1):
                continue
            if corner_idx == 3 and (r >= h - palette_len - 2 or c >= w - palette_len - 2):
                continue
            candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _chebyshev_far_enough(
    cell: tuple[int, int], others: list[tuple[int, int]], min_gap: int = 1
) -> bool:
    r, c = cell
    for orow, ocol in others:
        if max(abs(r - orow), abs(c - ocol)) <= min_gap:
            return False
    return True


def _expected_output(
    grid: np.ndarray, visited: list[tuple[int, int]], palette: list[tuple[int, int, int]]
) -> np.ndarray:
    out = grid.copy()
    for idx, (r, c) in enumerate(visited):
        out[r, c] = int(palette[idx][2])
    return out


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(16, 26)
    w = random.randint(16, 26)
    turn_count = random.randint(2, len(PALETTE_COLORS))
    corner_idx = random.randrange(4)
    palette_colors = random.sample(PALETTE_COLORS, turn_count)
    palette_positions = _palette_positions(corner_idx, turn_count, (h, w))
    palette = [
        (int(r), int(c), int(color))
        for (r, c), color in zip(palette_positions, palette_colors)
    ]
    palette_cells = {(r, c) for r, c, _ in palette}

    turns = _sample_turns(turn_count)
    for start in _start_candidates((h, w), corner_idx, turn_count):
        start_directions = list(range(4))
        random.shuffle(start_directions)
        for d_idx in start_directions:
            path = _extend_path(
                (h, w),
                start,
                d_idx,
                turns,
                0,
                {start},
                set(palette_cells),
            )
            if path is None:
                continue

            grid = np.full((h, w), BG, dtype=int)
            grid[start] = GREEN
            for r, c, color in palette:
                grid[r, c] = color
            for item in path:
                cr, cc = item["control"]
                grid[cr, cc] = int(item["color"])

            try:
                visited, path_cells = _trace_ray(grid)
            except AssertionError:
                continue

            intended_controls = [tuple(item["control"]) for item in path]
            if visited != intended_controls:
                continue

            visible_points = [start] + intended_controls
            free_cells = [
                (r, c)
                for r in range(1, h - 1)
                for c in range(1, w - 1)
                if (r, c) not in path_cells
                and (r, c) not in palette_cells
                and grid[r, c] == BG
                and r != start[0]
                and c != start[1]
                and _chebyshev_far_enough((r, c), visible_points, min_gap=1)
            ]
            random.shuffle(free_cells)
            distractor_count = random.randint(1, min(4, len(free_cells)))
            chosen = []
            for cell in free_cells:
                if _chebyshev_far_enough(cell, chosen, min_gap=1):
                    chosen.append(cell)
                if len(chosen) == distractor_count:
                    break
            if len(chosen) != distractor_count:
                continue
            for cell in chosen:
                grid[cell] = random.choice((PINK, CYAN))

            try:
                actual_palette = _read_palette(grid, BG)
                actual_visited, _ = _trace_ray(grid)
            except AssertionError:
                continue
            if actual_visited != intended_controls:
                continue
            if [color for _, _, color in actual_palette] != palette_colors:
                continue

            expected = _expected_output(grid, intended_controls, actual_palette)
            try:
                actual = main(grid)
            except AssertionError:
                continue
            if np.array_equal(actual, expected):
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
