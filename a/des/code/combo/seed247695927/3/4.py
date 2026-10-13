from __future__ import annotations

import random
from collections import Counter

import numpy as np


END_MARKER = 5
HINGE_MARKER = 0
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATE_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _component_degree(coord_set: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in coord_set for dr, dc in DIR4)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    comps = _components(g != bg)
    assert comps, "expected at least one rod"

    out = g.copy()
    clear_cells: list[tuple[int, int]] = []
    paint_cells: list[tuple[int, int, int]] = []

    for comp in comps:
        coord_set = set(comp)
        colors = Counter(int(g[r, c]) for r, c in comp)
        max_count = max(colors.values())
        line_colors = [color for color, count in colors.items() if count == max_count]
        assert max_count >= 2 and len(line_colors) == 1, "line color must be uniquely most common"
        line_color = int(line_colors[0])

        marker_cells = [(r, c) for r, c in comp if int(g[r, c]) != line_color]
        assert len(marker_cells) == 2, "each rod must have exactly two markers"

        endpoint_candidates = [p for p in marker_cells if _component_degree(coord_set, p) == 1]
        hinge_candidates = [p for p in marker_cells if _component_degree(coord_set, p) == 2]
        assert len(endpoint_candidates) == 1 and len(hinge_candidates) == 1
        endpoint = endpoint_candidates[0]
        hinge = hinge_candidates[0]

        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        assert len(rows) == 1 or len(cols) == 1, "rod must be straight"

        if len(rows) == 1:
            fixed_sign = int(np.sign(endpoint[1] - hinge[1]))
            assert fixed_sign != 0
            tail_step = (0, -fixed_sign)
            up_space = hinge[0]
            down_space = h - 1 - hinge[0]
            assert up_space != down_space, "fold direction must be unique"
            fold_step = (-1, 0) if up_space > down_space else (1, 0)
        else:
            fixed_sign = int(np.sign(endpoint[0] - hinge[0]))
            assert fixed_sign != 0
            tail_step = (-fixed_sign, 0)
            left_space = hinge[1]
            right_space = w - 1 - hinge[1]
            assert left_space != right_space, "fold direction must be unique"
            fold_step = (0, -1) if left_space > right_space else (0, 1)

        tail: list[tuple[int, int]] = []
        step = 1
        while True:
            rr = hinge[0] + tail_step[0] * step
            cc = hinge[1] + tail_step[1] * step
            if (rr, cc) not in coord_set:
                break
            assert int(g[rr, cc]) == line_color
            tail.append((rr, cc))
            step += 1
        assert tail, "hinge must have a movable tail"

        for idx, (rr, cc) in enumerate(tail, start=1):
            del rr, cc
            nr = hinge[0] + fold_step[0] * idx
            nc = hinge[1] + fold_step[1] * idx
            assert 0 <= nr < h and 0 <= nc < w
            paint_cells.append((nr, nc, line_color))
        clear_cells.extend(tail)

    clear_set = set(clear_cells)
    assert len(clear_set) == len(clear_cells), "tails must not overlap before folding"
    paint_positions = [(r, c) for r, c, _ in paint_cells]
    assert len(set(paint_positions)) == len(paint_positions), "folded tails must not overlap"
    for r, c, color in paint_cells:
        assert int(g[r, c]) == bg or (r, c) in clear_set or int(g[r, c]) == color

    for r, c in clear_cells:
        out[r, c] = bg
    for r, c, color in paint_cells:
        out[r, c] = color
    return out


def _rand_bg() -> int:
    return random.choice([1, 2, 3, 4, 6, 7, 8, 9])


def _choose_line_colors(bg: int, count: int) -> list[int]:
    choices = [c for c in PALETTE if c != bg and c not in {END_MARKER, HINGE_MARKER}]
    random.shuffle(choices)
    return choices[:count]


def _rod_cells(
    orient: str,
    hinge: tuple[int, int],
    fixed_sign: int,
    fixed_len: int,
    tail_len: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    hr, hc = hinge
    if orient == "h":
        fixed_step = (0, fixed_sign)
        tail_step = (0, -fixed_sign)
    else:
        fixed_step = (fixed_sign, 0)
        tail_step = (-fixed_sign, 0)

    fixed_cells = [(hr + fixed_step[0] * k, hc + fixed_step[1] * k) for k in range(1, fixed_len)]
    endpoint = fixed_cells[-1]
    tail_cells = [(hr + tail_step[0] * k, hc + tail_step[1] * k) for k in range(1, tail_len + 1)]
    all_cells = [hinge] + fixed_cells + tail_cells
    return all_cells, endpoint, fixed_cells[:-1], tail_cells


def _folded_cells(shape: tuple[int, int], orient: str, hinge: tuple[int, int], tail_len: int) -> list[tuple[int, int]] | None:
    h, w = shape
    hr, hc = hinge
    if orient == "h":
        up_space = hr
        down_space = h - 1 - hr
        if up_space == down_space:
            return None
        step = (-1, 0) if up_space > down_space else (1, 0)
    else:
        left_space = hc
        right_space = w - 1 - hc
        if left_space == right_space:
            return None
        step = (0, -1) if left_space > right_space else (0, 1)
    return [(hr + step[0] * k, hc + step[1] * k) for k in range(1, tail_len + 1)]


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _try_place_rod(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    orient: str,
    branch_bias: int,
    line_color: int,
) -> bool:
    h, w = grid.shape
    fixed_sign = random.choice([-1, 1])
    fixed_len = random.randint(2, 5)
    tail_len = random.randint(1, 4)
    if fixed_len + tail_len < 4:
        return False

    row_choices = list(range(h))
    col_choices = list(range(w))
    random.shuffle(row_choices)
    random.shuffle(col_choices)

    for hr in row_choices:
        for hc in col_choices:
            if orient == "h":
                up_space = hr
                down_space = h - 1 - hr
                if up_space == down_space:
                    continue
                if branch_bias > 0 and down_space <= up_space:
                    continue
                if branch_bias < 0 and up_space <= down_space:
                    continue
            else:
                left_space = hc
                right_space = w - 1 - hc
                if left_space == right_space:
                    continue
                if branch_bias > 0 and right_space <= left_space:
                    continue
                if branch_bias < 0 and left_space <= right_space:
                    continue

            all_cells, endpoint, fixed_red, tail_cells = _rod_cells(
                orient, (hr, hc), fixed_sign, fixed_len, tail_len
            )
            folded = _folded_cells(grid.shape, orient, (hr, hc), tail_len)
            if folded is None:
                continue
            envelope = set(all_cells) | set(folded)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in envelope):
                continue
            if _dilate(envelope, grid.shape) & occupied:
                continue

            for r, c in fixed_red + tail_cells:
                grid[r, c] = line_color
            grid[endpoint] = END_MARKER
            grid[hr, hc] = HINGE_MARKER
            occupied.update(envelope)
            return True
    return False


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(8, 16)
    w = random.randint(8, 16)
    bg = _rand_bg()
    grid = np.full((h, w), bg, dtype=int)

    max_rods = min(5, max(2, (h * w) // 28))
    rod_count = random.randint(1, max_rods)
    line_colors = _choose_line_colors(bg, rod_count)
    if len(line_colors) < rod_count:
        return None

    occupied: set[tuple[int, int]] = set()
    rod_specs = []
    for i in range(rod_count):
        orient = random.choice(["h", "v"])
        branch_bias = random.choice([-1, 1])
        rod_specs.append((orient, branch_bias, line_colors[i]))

    if all(orient == "h" for orient, _, _ in rod_specs):
        rod_specs[0] = ("v", rod_specs[0][1], rod_specs[0][2])
    if all(orient == "v" for orient, _, _ in rod_specs):
        rod_specs[0] = ("h", rod_specs[0][1], rod_specs[0][2])

    for orient, branch_bias, line_color in rod_specs:
        if not _try_place_rod(grid, occupied, orient, branch_bias, line_color):
            return None

    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-rod input")
