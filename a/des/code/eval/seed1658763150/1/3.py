from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
DIVIDER_COLOR = 9
PALETTE = (1, 2, 3, 6, 7, 8)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {delta: idx for idx, delta in enumerate(DIRS)}

GEN_ATTEMPTS = 300
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (24, 36)
LEFT_MIN_WIDTH = 10
RIGHT_MIN_WIDTH = 11
LEGEND_COLOR_RANGE = (2, 4)
FIRST_LEN_RANGE = (2, 5)
RIGHT_EXTRA_SEEDS_RANGE = (0, 2)
PLACEMENT_ATTEMPTS = 200


def _turn(direction: int, turn_left: bool) -> int:
    return (direction - 1) % 4 if turn_left else (direction + 1) % 4


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _in_bounds(r: int, c: int, row_lo: int, row_hi: int, col_lo: int, col_hi: int) -> bool:
    return row_lo <= r < row_hi and col_lo <= c < col_hi


def _grow_path(seed: tuple[int, int], initial_dir: int, first_len: int) -> list[tuple[int, int]]:
    r, c = seed
    cells = [(r, c)]
    direction = int(initial_dir)
    turn_left = True
    for seg_len in range(int(first_len), 0, -1):
        dr, dc = DIRS[direction]
        for _ in range(seg_len):
            r += dr
            c += dc
            cells.append((r, c))
        if seg_len > 1:
            direction = _turn(direction, turn_left)
            turn_left = not turn_left
    return cells


def _normalize(cells: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, min_c = min(rows), min(cols)
    shifted = [(r - min_r, c - min_c) for r, c in cells]
    height = max(r for r, _ in shifted) + 1
    width = max(c for _, c in shifted) + 1
    return shifted, height, width


def _clip_to_region(
    cells: list[tuple[int, int]], row_lo: int, row_hi: int, col_lo: int, col_hi: int
) -> list[tuple[int, int]]:
    return [(r, c) for r, c in cells if _in_bounds(r, c, row_lo, row_hi, col_lo, col_hi)]


def _halo(
    cells: set[tuple[int, int]], row_lo: int, row_hi: int, col_lo: int, col_hi: int
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(rr, cc, row_lo, row_hi, col_lo, col_hi):
                    out.add((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _trace_path(start: tuple[int, int], cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [pt for pt in _neighbors4(*cur) if pt in cells and pt != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            return []
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def _path_signature(path: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    if len(path) < 2:
        return [], []
    dirs = [
        DIR_TO_INDEX[(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])]
        for i in range(len(path) - 1)
    ]
    run_dirs = [dirs[0]]
    run_lengths = [1]
    for direction in dirs[1:]:
        if direction == run_dirs[-1]:
            run_lengths[-1] += 1
        else:
            run_dirs.append(direction)
            run_lengths.append(1)
    return run_dirs, run_lengths


def _matches_rule(run_dirs: list[int], run_lengths: list[int]) -> bool:
    if not run_lengths:
        return False
    first_len = run_lengths[0]
    if run_lengths != list(range(first_len, 0, -1)):
        return False
    for idx in range(1, len(run_dirs)):
        expect = _turn(run_dirs[idx - 1], turn_left=(idx % 2 == 1))
        if run_dirs[idx] != expect:
            return False
    return True


def _find_divider(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates = [c for c in range(w) if int(grid[0, c]) != BG and np.all(grid[:, c] == grid[0, c])]
    candidates = [
        c
        for c in candidates
        if 0 < c < w - 1 and np.any(grid[:, :c] != BG) and np.any(grid[:, c + 1 :] != BG)
    ]
    assert len(candidates) == 1, "expected exactly one full-height divider line"
    col = candidates[0]
    return col, int(grid[0, col])


def _learn_first_lengths(left: np.ndarray) -> dict[int, int]:
    mask = left != BG
    comps = _components(mask)
    learned: dict[int, int] = {}
    for comp in comps:
        colors = {int(left[r, c]) for r, c in comp}
        assert len(colors) == 1, "legend groups must be monochrome"
        color = next(iter(colors))
        cell_set = set(comp)
        degrees: dict[tuple[int, int], int] = {}
        endpoints: list[tuple[int, int]] = []
        for r, c in comp:
            deg = sum((nr, nc) in cell_set for nr, nc in _neighbors4(r, c))
            assert deg in (1, 2), "legend group must be a simple path"
            degrees[(r, c)] = deg
            if deg == 1:
                endpoints.append((r, c))
        assert len(endpoints) == 2, "legend group must have exactly two endpoints"

        candidates: list[int] = []
        for endpoint in endpoints:
            path = _trace_path(endpoint, cell_set)
            if len(path) != len(comp):
                continue
            run_dirs, run_lengths = _path_signature(path)
            if _matches_rule(run_dirs, run_lengths):
                candidates.append(run_lengths[0])
        assert candidates, "failed to infer first segment length from legend"

        first_len = max(candidates)
        if color in learned:
            assert learned[color] == first_len
        learned[color] = first_len
    assert learned, "legend must contain at least one example group"
    return learned


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn a first-segment length per color from the left legend, then grow
    same-color staircase branches from the seeds on the right.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    divider_col, divider_color = _find_divider(g)
    left = g[:, :divider_col]
    learned = _learn_first_lengths(left)

    out = g.copy()
    out[out == YELLOW] = BG

    row_lo, row_hi = 0, h
    col_lo, col_hi = divider_col + 1, w
    for yr, yc in np.argwhere(g == YELLOW):
        if int(yc) <= divider_col:
            continue
        candidates: list[tuple[int, int, int, int]] = []
        for direction, (dr, dc) in enumerate(DIRS):
            sr = int(yr) - dr
            sc = int(yc) - dc
            if not _in_bounds(sr, sc, row_lo, row_hi, col_lo, col_hi):
                continue
            color = int(g[sr, sc])
            if color in learned:
                candidates.append((sr, sc, direction, color))
        assert len(candidates) == 1, "each yellow marker must point to exactly one seed"
        sr, sc, direction, color = candidates[0]
        for r, c in _clip_to_region(_grow_path((sr, sc), direction, learned[color]), row_lo, row_hi, col_lo, col_hi):
            out[r, c] = color

    out[:, divider_col] = divider_color
    return out


def _place_legend_groups(
    grid: np.ndarray, divider_col: int, color_to_len: dict[int, int]
) -> bool:
    h, _ = grid.shape
    blocked: set[tuple[int, int]] = set()
    colors = list(color_to_len)
    random.shuffle(colors)
    for color in colors:
        initial_dir = random.randrange(4)
        base_cells = _grow_path((0, 0), initial_dir, color_to_len[color])
        norm_cells, box_h, box_w = _normalize(base_cells)
        max_top = h - box_h
        max_left = divider_col - box_w
        if max_top < 0 or max_left < 0:
            return False
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, max_top)
            left = random.randint(0, max_left)
            cells = {(top + r, left + c) for r, c in norm_cells}
            if cells & blocked:
                continue
            for r, c in cells:
                grid[r, c] = color
            blocked |= _halo(cells, 0, h, 0, divider_col)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_canvas_seeds(
    grid: np.ndarray, divider_col: int, color_to_len: dict[int, int]
) -> bool:
    h, w = grid.shape
    row_lo, row_hi = 0, h
    col_lo, col_hi = divider_col + 1, w

    seed_colors = list(color_to_len)
    extras = [random.choice(seed_colors) for _ in range(random.randint(*RIGHT_EXTRA_SEEDS_RANGE))]
    plan = seed_colors + extras
    random.shuffle(plan)

    require_clip = random.random() < 0.5
    forced_clip_index = random.randrange(len(plan)) if require_clip else -1

    specs = []
    for idx, color in enumerate(plan):
        specs.append(
            {
                "color": int(color),
                "first_len": int(color_to_len[color]),
                "direction": random.randrange(4),
                "want_clip": idx == forced_clip_index,
            }
        )
    specs.sort(key=lambda item: (0 if item["want_clip"] else 1, -item["first_len"]))

    blocked: set[tuple[int, int]] = set()
    placed_any_clip = False

    for spec in specs:
        color = int(spec["color"])
        first_len = int(spec["first_len"])
        direction = int(spec["direction"])
        want_clip = bool(spec["want_clip"])
        dr, dc = DIRS[direction]
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            seed_r = random.randint(row_lo, row_hi - 1)
            seed_c = random.randint(col_lo, col_hi - 1)
            marker_r = seed_r + dr
            marker_c = seed_c + dc
            if not _in_bounds(marker_r, marker_c, row_lo, row_hi, col_lo, col_hi):
                continue

            full_path = _grow_path((seed_r, seed_c), direction, first_len)
            clipped = _clip_to_region(full_path, row_lo, row_hi, col_lo, col_hi)
            did_clip = len(clipped) != len(full_path)
            if want_clip != did_clip:
                continue

            cells = set(clipped)
            if cells & blocked:
                continue

            grid[seed_r, seed_c] = color
            grid[marker_r, marker_c] = YELLOW
            blocked |= _halo(cells, row_lo, row_hi, col_lo, col_hi)
            placed = True
            placed_any_clip = placed_any_clip or did_clip
            break
        if not placed:
            return False

    if require_clip and not placed_any_clip:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    if w < LEFT_MIN_WIDTH + RIGHT_MIN_WIDTH + 1:
        return None

    divider_col = random.randint(LEFT_MIN_WIDTH, w - RIGHT_MIN_WIDTH - 1)
    grid = np.zeros((h, w), dtype=int)
    grid[:, divider_col] = DIVIDER_COLOR

    max_colors = 4 if h >= 22 and divider_col >= 12 else 3
    num_colors = random.randint(LEGEND_COLOR_RANGE[0], min(LEGEND_COLOR_RANGE[1], max_colors))
    colors = random.sample(PALETTE, num_colors)
    color_to_len = {color: random.randint(*FIRST_LEN_RANGE) for color in colors}

    if not _place_legend_groups(grid, divider_col, color_to_len):
        return None
    if not _place_canvas_seeds(grid, divider_col, color_to_len):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.any(grid[:, divider_col] != DIVIDER_COLOR):
        return None
    if not np.any(grid == YELLOW):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
