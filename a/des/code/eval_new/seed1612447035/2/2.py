from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SOURCE = 2
PATH_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
SIZE_RANGE = (18, 24)
MAIN_RETRIES = 80
CASE_RETRIES = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate(cell: tuple[int, int], size: int, k: int) -> tuple[int, int]:
    r, c = cell
    if k == 0:
        return r, c
    if k == 1:
        return c, size - 1 - r
    if k == 2:
        return size - 1 - r, size - 1 - c
    return size - 1 - c, r


def _transform_cells(cells: list[tuple[int, int]], size: int, k: int) -> list[tuple[int, int]]:
    return [_rotate(cell, size, k) for cell in cells]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_source_and_legend(
    grid: np.ndarray,
) -> tuple[tuple[int, int], set[tuple[int, int]], dict[int, int]]:
    h, w = grid.shape
    red_border = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] == SOURCE and (r == 0 or c == 0 or r == h - 1 or c == w - 1)
    ]
    assert len(red_border) == 1, "expected a unique red source pixel on the grid edge"
    source = red_border[0]

    non_bg = grid != _most_common_color(grid)
    candidates = []
    for comp in _components(non_bg):
        if source in comp:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh * ww != len(comp):
            continue
        if ww == 2 or hh == 2:
            candidates.append((comp, r0, c0, r1, c1))

    assert len(candidates) == 1, "expected exactly one disconnected legend rectangle"
    comp, r0, c0, r1, c1 = candidates[0]
    legend_cells = set(comp)
    mapping: dict[int, int] = {}

    if c1 - c0 + 1 == 2:
        for r in range(r0, r1 + 1):
            left = int(grid[r, c0])
            right = int(grid[r, c0 + 1])
            assert left != right, "legend rows must encode two different colors"
            assert left not in mapping, "legend left colors must be unique"
            mapping[left] = right
    else:
        for c in range(c0, c1 + 1):
            left = int(grid[r0, c])
            right = int(grid[r0 + 1, c])
            assert left != right, "legend columns must encode two different colors"
            assert left not in mapping, "legend left colors must be unique"
            mapping[left] = right

    return source, legend_cells, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate the edge signal through the path network, then grow a terminal bar whose length equals the traveled step count.

    Input structure:
    - The background is the dominant color.
    - A single red border pixel is the signal source; it 4-touches exactly one path endpoint.
    - The colored paths are one-pixel-wide 4-connected chains with distinct colors, and some paths share one pixel with another path to create intersections.
    - A disconnected 2-cell-wide legend rectangle lists color pairs: each row encodes current-path color on the left and preferred turn color on the right.

    Rule:
    - Start at the path endpoint adjacent to the red source and move one path pixel at a time without stepping back.
    - At each intersection, first check the legend entry for the current path color; if the mapped color is present among the forward neighbors, switch onto that color's branch, otherwise continue along the current color when possible.
    - Count every visited path pixel until the signal reaches a dead end.
    - Keep the input unchanged and extend a straight bar from the terminal pixel in the last travel direction; the bar's color is the terminal path color and its total length equals the visited-step count.

    Key insight:
    - The legend does not place new pixels directly; it only decides which colored branch the signal will follow at a shared pixel, and the eventual bar length is the trace length of that simulated walk.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    source, legend_cells, mapping = _extract_source_and_legend(g)
    bg = _most_common_color(g)

    path_mask = (g != bg)
    path_mask[source] = False
    for r, c in legend_cells:
        path_mask[r, c] = False

    start_neighbors = [
        cell
        for cell in _neighbors4(*source)
        if _in_bounds((h, w), cell) and path_mask[cell]
    ]
    assert len(start_neighbors) == 1, "source must touch exactly one path endpoint"

    prev = source
    cur = start_neighbors[0]
    current_color = int(g[cur])
    visited: list[tuple[int, int]] = []
    seen_states: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    while True:
        state = (prev, cur, current_color)
        assert state not in seen_states, "signal walk must not enter a loop"
        seen_states.add(state)
        visited.append(cur)
        nbrs = [
            cell
            for cell in _neighbors4(*cur)
            if _in_bounds((h, w), cell) and path_mask[cell] and cell != prev
        ]
        if not nbrs:
            break

        nxt = None
        target_color = mapping.get(current_color)
        if target_color is not None:
            target_nbrs = [cell for cell in nbrs if int(g[cell]) == target_color]
            if len(target_nbrs) == 1:
                nxt = target_nbrs[0]
                current_color = target_color
            else:
                assert len(target_nbrs) <= 1, "target branch must be unique when present"

        if nxt is None:
            same_nbrs = [cell for cell in nbrs if int(g[cell]) == current_color]
            if len(same_nbrs) == 1:
                nxt = same_nbrs[0]
            elif len(same_nbrs) == 0 and len(nbrs) == 1:
                nxt = nbrs[0]
                current_color = int(g[nxt])
            else:
                dr = cur[0] - prev[0]
                dc = cur[1] - prev[1]
                straight = [
                    cell
                    for cell in nbrs
                    if (cell[0] - cur[0], cell[1] - cur[1]) == (dr, dc)
                ]
                assert len(straight) == 1, "path continuation must be unique"
                nxt = straight[0]
                current_color = int(g[nxt])

        prev, cur = cur, nxt

    assert visited, "the signal must visit at least one path cell"
    terminal = visited[-1]
    terminal_color = current_color
    dr = terminal[0] - prev[0]
    dc = terminal[1] - prev[1]
    assert (dr, dc) != (0, 0)

    out = g.copy()
    for step in range(len(visited)):
        rr = terminal[0] + dr * step
        cc = terminal[1] + dc * step
        assert 0 <= rr < h and 0 <= cc < w, "terminal bar must fit inside the canvas"
        if step > 0:
            assert out[rr, cc] == bg, "terminal bar may only extend through background"
        out[rr, cc] = terminal_color
    return out


def _sample_main(size: int) -> list[tuple[int, int]] | None:
    for _ in range(MAIN_RETRIES):
        shape = random.choices(
            ["straight", "turn", "zigzag"], weights=[0.3, 0.3, 0.4], k=1
        )[0]
        first_len = random.randint(4, 6)
        if shape == "straight":
            row = random.randint(4, size - 5)
            cells = [(row, c) for c in range(1, 1 + first_len)]
            if all(_in_bounds((size, size), cell) for cell in cells):
                return cells
            continue

        second_len = random.randint(3, 5)
        sign = random.choice([-1, 1])
        if sign < 0:
            row = random.randint(second_len, size - 5)
        else:
            row = random.randint(4, size - second_len - 1)

        cells = [(row, c) for c in range(1, 1 + first_len)]
        turn = cells[-1]
        cells.extend((turn[0] + sign * d, turn[1]) for d in range(1, second_len))

        if shape == "zigzag":
            third_len = random.randint(3, 5)
            turn2 = cells[-1]
            cells.extend((turn2[0], turn2[1] + d) for d in range(1, third_len))

        if len(set(cells)) != len(cells):
            continue
        if all(_in_bounds((size, size), cell) for cell in cells):
            return cells
    return None


def _branch_directions(main_cells: list[tuple[int, int]], idx: int) -> list[tuple[int, int]]:
    prev = main_cells[idx - 1]
    cur = main_cells[idx]
    nxt = main_cells[idx + 1]
    dr1 = cur[0] - prev[0]
    dc1 = cur[1] - prev[1]
    dr2 = nxt[0] - cur[0]
    dc2 = nxt[1] - cur[1]
    if (dr1, dc1) != (dr2, dc2):
        return []
    if dr1 == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _candidate_indices(main_cells: list[tuple[int, int]]) -> list[int]:
    return [
        idx
        for idx in range(2, len(main_cells) - 2)
        if _branch_directions(main_cells, idx)
    ]


def _make_branch(
    size: int,
    main_cells: list[tuple[int, int]],
    idx: int,
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]] | None:
    start = main_cells[idx]
    cells = [start]
    for step in range(1, length):
        cell = (start[0] + direction[0] * step, start[1] + direction[1] * step)
        if not _in_bounds((size, size), cell):
            return None
        cells.append(cell)
    return cells


def _tail_cells(
    terminal: tuple[int, int], direction: tuple[int, int], steps: int
) -> list[tuple[int, int]]:
    return [
        (terminal[0] + direction[0] * delta, terminal[1] + direction[1] * delta)
        for delta in range(1, steps)
    ]


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    size = random.randint(*SIZE_RANGE)
    main_cells = _sample_main(size)
    if main_cells is None or len(main_cells) < 5:
        return None

    candidate_idxs = _candidate_indices(main_cells)
    if not candidate_idxs:
        return None

    switch_case = random.random() < 0.5
    occupied = set(main_cells)
    branches: list[dict] = []
    selected_attach = None
    selected_branch_index = None
    reserved_tail: set[tuple[int, int]] = set()
    active_route = list(main_cells)

    if switch_case:
        attach_order = sorted(random.sample(candidate_idxs, len(candidate_idxs)))
        random.shuffle(attach_order)
        chosen = None
        for idx in attach_order:
            options = _branch_directions(main_cells, idx)
            random.shuffle(options)
            for direction in options:
                lengths = list(range(3, 6))
                random.shuffle(lengths)
                for length in lengths:
                    branch_cells = _make_branch(size, main_cells, idx, direction, length)
                    if branch_cells is None:
                        continue
                    if any(cell in occupied for cell in branch_cells[1:]):
                        continue
                    prefix_len = idx + 1
                    steps = prefix_len + length - 1
                    tail = _tail_cells(branch_cells[-1], direction, steps)
                    if not all(_in_bounds((size, size), cell) for cell in tail):
                        continue
                    if any(cell in occupied for cell in tail):
                        continue
                    chosen = (idx, direction, branch_cells, tail)
                    break
                if chosen is not None:
                    break
            if chosen is not None:
                break
        if chosen is None:
            return None

        idx, direction, branch_cells, tail = chosen
        occupied.update(branch_cells[1:])
        branches.append({"attach": idx, "cells": branch_cells})
        selected_attach = idx
        selected_branch_index = 0
        reserved_tail = set(tail)
        active_route = main_cells[: idx + 1] + branch_cells[1:]
    else:
        final_dir = (
            main_cells[-1][0] - main_cells[-2][0],
            main_cells[-1][1] - main_cells[-2][1],
        )
        tail = _tail_cells(main_cells[-1], final_dir, len(main_cells))
        if not all(_in_bounds((size, size), cell) for cell in tail):
            return None
        if any(cell in occupied for cell in tail):
            return None
        reserved_tail = set(tail)

    extra_candidates = [idx for idx in candidate_idxs if idx != selected_attach]
    random.shuffle(extra_candidates)
    extra_goal = random.randint(0, min(2, len(extra_candidates)))

    for idx in extra_candidates:
        if len(branches) >= (1 if switch_case else 0) + extra_goal:
            break
        options = _branch_directions(main_cells, idx)
        random.shuffle(options)
        placed = False
        for direction in options:
            lengths = list(range(3, 6))
            random.shuffle(lengths)
            for length in lengths:
                branch_cells = _make_branch(size, main_cells, idx, direction, length)
                if branch_cells is None:
                    continue
                extra_cells = set(branch_cells[1:])
                if extra_cells & occupied:
                    continue
                if extra_cells & reserved_tail:
                    continue
                branches.append({"attach": idx, "cells": branch_cells})
                occupied.update(extra_cells)
                placed = True
                break
            if placed:
                break

    if not branches:
        return None

    color_pool = random.sample(PATH_COLORS, 1 + len(branches))
    main_color = color_pool[0]
    for branch, color in zip(branches, color_pool[1:]):
        branch["color"] = color

    legend_rows: list[tuple[int, int]] = []
    if switch_case:
        assert selected_branch_index is not None
        selected_branch = branches[selected_branch_index]
        legend_rows.append((main_color, int(selected_branch["color"])))
        distractor_lefts = [int(branch["color"]) for i, branch in enumerate(branches) if i != selected_branch_index]
        random.shuffle(distractor_lefts)
        for left in distractor_lefts[: random.randint(0, min(2, len(distractor_lefts)))]:
            rights = [main_color] + [int(branch["color"]) for branch in branches if int(branch["color"]) != left]
            legend_rows.append((left, random.choice(rights)))
        terminal_color = int(selected_branch["color"])
    else:
        lefts = [int(branch["color"]) for branch in branches]
        random.shuffle(lefts)
        take = random.randint(1, min(3, len(lefts)))
        for left in lefts[:take]:
            rights = [main_color] + [int(branch["color"]) for branch in branches if int(branch["color"]) != left]
            legend_rows.append((left, random.choice(rights)))
        terminal_color = main_color

    random.shuffle(legend_rows)
    assert legend_rows, "legend must contain at least one row"

    line_dir = (
        active_route[-1][0] - active_route[-2][0],
        active_route[-1][1] - active_route[-2][1],
    )
    output_line = [
        (active_route[-1][0] + line_dir[0] * step, active_route[-1][1] + line_dir[1] * step)
        for step in range(len(active_route))
    ]
    if not all(_in_bounds((size, size), cell) for cell in output_line):
        return None

    rot = random.randrange(4)
    source = _rotate((main_cells[0][0], 0), size, rot)
    main_rot = _transform_cells(main_cells, size, rot)
    line_rot = _transform_cells(output_line, size, rot)
    branches_rot = [
        {"attach": branch["attach"], "cells": _transform_cells(branch["cells"], size, rot), "color": int(branch["color"])}
        for branch in branches
    ]

    reserved = set(main_rot)
    reserved.add(source)
    for branch in branches_rot:
        reserved.update(branch["cells"][1:])
    reserved.update(line_rot[1:])

    legend_h = len(legend_rows)
    corner_positions = [
        (1, 1),
        (1, size - 3),
        (size - legend_h - 1, 1),
        (size - legend_h - 1, size - 3),
    ]
    positions = []
    for pos in corner_positions:
        if pos not in positions:
            positions.append(pos)
    for r in range(1, size - legend_h):
        for c in range(1, size - 2):
            positions.append((r, c))
    random.shuffle(positions[4:])

    legend_top_left = None
    for r0, c0 in positions:
        if r0 < 0 or c0 < 0 or r0 + legend_h > size or c0 + 2 > size:
            continue
        block = {(r, c) for r in range(r0, r0 + legend_h) for c in range(c0, c0 + 2)}
        moat = {
            (r, c)
            for r in range(r0 - 1, r0 + legend_h + 1)
            for c in range(c0 - 1, c0 + 3)
            if 0 <= r < size and 0 <= c < size
        }
        if block & reserved:
            continue
        if moat & (reserved - block):
            continue
        legend_top_left = (r0, c0)
        break

    if legend_top_left is None:
        return None

    inp = np.full((size, size), BG, dtype=int)
    for r, c in main_rot:
        inp[r, c] = main_color
    for branch in branches_rot:
        for r, c in branch["cells"][1:]:
            inp[r, c] = int(branch["color"])
    inp[source] = SOURCE

    lr0, lc0 = legend_top_left
    for offset, (left, right) in enumerate(legend_rows):
        inp[lr0 + offset, lc0] = left
        inp[lr0 + offset, lc0 + 1] = right

    out = inp.copy()
    for r, c in line_rot:
        out[r, c] = terminal_color

    try:
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(CASE_RETRIES):
        built = _sample_case()
        if built is not None:
            return built[0]
    raise RuntimeError(f"failed to generate a valid input after {CASE_RETRIES} attempts")
