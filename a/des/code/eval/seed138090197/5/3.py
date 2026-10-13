from __future__ import annotations

import random

import numpy as np


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_L_PATTERNS = [
    frozenset(((-1, 0), (0, -1))),
    frozenset(((-1, 0), (0, 1))),
    frozenset(((1, 0), (0, -1))),
    frozenset(((1, 0), (0, 1))),
]
_T_PATTERNS = [frozenset(d for d in _DIRS if d != missing) for missing in _DIRS]
_CROSS_PATTERN = frozenset(_DIRS)


def _mode_color(values: np.ndarray) -> int:
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[np.argmax(counts)])


def _partition_color(g: np.ndarray) -> int:
    row_colors = [int(g[r, 0]) for r in range(g.shape[0]) if np.all(g[r, :] == g[r, 0])]
    col_colors = [int(g[0, c]) for c in range(g.shape[1]) if np.all(g[:, c] == g[0, c])]
    candidates = set(row_colors) & set(col_colors)
    assert candidates, "expected at least one full-row/full-column partition color"
    return max(candidates, key=lambda color: row_colors.count(color) + col_colors.count(color))


def _spans(line_positions: list[int], size: int) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(line_positions) + [size]
    spans = []
    for left, right in zip(bounds, bounds[1:]):
        if left + 1 <= right - 1:
            spans.append((left + 1, right - 1))
    return spans


def _partition_spans(g: np.ndarray, line_color: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    line_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == line_color)]
    line_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == line_color)]
    assert line_rows and line_cols, "expected both horizontal and vertical partition lines"
    row_spans = _spans(line_rows, g.shape[0])
    col_spans = _spans(line_cols, g.shape[1])
    assert row_spans and col_spans
    return row_spans, col_spans


def _pattern_from_cells(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    cell_set = {tuple(map(int, cell)) for cell in coords}
    candidates: list[frozenset[tuple[int, int]]] = []
    for sr, sc in cell_set:
        built = {(sr, sc)}
        dirs = []
        for dr, dc in _DIRS:
            k = 1
            advanced = False
            while (sr + dr * k, sc + dc * k) in cell_set:
                built.add((sr + dr * k, sc + dc * k))
                advanced = True
                k += 1
            if advanced:
                dirs.append((dr, dc))
        if built == cell_set and 2 <= len(dirs) <= 4:
            candidates.append(frozenset(dirs))
    unique = list(dict.fromkeys(candidates))
    assert len(unique) == 1, "legend example must encode a unique direction set"
    return unique[0]


def _legend_patterns(legend: np.ndarray, bg: int) -> dict[int, frozenset[tuple[int, int]]]:
    patterns: dict[int, frozenset[tuple[int, int]]] = {}
    for color in np.unique(legend):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(legend == color)
        patterns[color] = _pattern_from_cells(coords)
    assert patterns, "expected at least one legend example"
    return patterns


def _grow_region(
    sub: np.ndarray,
    seed_pos: tuple[int, int],
    color: int,
    directions: frozenset[tuple[int, int]],
    bg: int,
    obstacle_color: int | None,
) -> np.ndarray:
    out = sub.copy()
    sr, sc = seed_pos
    for dr, dc in directions:
        rr, cc = sr + dr, sc + dc
        while 0 <= rr < sub.shape[0] and 0 <= cc < sub.shape[1]:
            if obstacle_color is not None and int(sub[rr, cc]) == obstacle_color:
                break
            assert int(sub[rr, cc]) == bg, "growth path may only cross background"
            out[rr, cc] = color
            rr += dr
            cc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top-left legend region, convert each color's sample shape into a set of growth
    directions, then grow each matching seed inside its own rectangular partition until it hits
    a grey wall or a maroon obstacle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    line_color = _partition_color(g)
    bg = _mode_color(g[g != line_color])
    row_spans, col_spans = _partition_spans(g, line_color)

    lr0, lr1 = row_spans[0]
    lc0, lc1 = col_spans[0]
    legend = g[lr0 : lr1 + 1, lc0 : lc1 + 1]
    patterns = _legend_patterns(legend, bg)

    out = g.copy()
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            if ri == 0 and ci == 0:
                continue

            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = {int(v) for v in np.unique(sub)} - {bg}
            seed_colors = colors & set(patterns)
            assert len(seed_colors) == 1, "each non-legend region must contain exactly one seed color"
            seed_color = next(iter(seed_colors))
            obstacle_colors = colors - {seed_color}
            assert len(obstacle_colors) <= 1, "expected at most one obstacle color in a region"
            obstacle_color = next(iter(obstacle_colors)) if obstacle_colors else None

            seed_cells = np.argwhere(sub == seed_color)
            assert seed_cells.shape == (1, 2), "each region must contain exactly one seed pixel"
            seed_pos = tuple(map(int, seed_cells[0]))

            out[r0 : r1 + 1, c0 : c1 + 1] = _grow_region(
                sub=sub,
                seed_pos=seed_pos,
                color=seed_color,
                directions=patterns[seed_color],
                bg=bg,
                obstacle_color=obstacle_color,
            )

    return out


_BG = 0
_LINE = 5
_OBSTACLE = 9
_SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)
_LAYOUTS = ((2, 3), (3, 2), (3, 3))
_LEGEND_SLOT = 5
_LEGEND_MARGIN = 1
_OTHER_SPAN_MIN = 5
_OTHER_SPAN_MAX = 8
_GEN_RETRIES = 400


def _region_boxes(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    boxes = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            boxes.append((r0, c0, r1, c1))
    return boxes


def _legend_size() -> tuple[int, int]:
    base = 2 * _LEGEND_SLOT + 3 * _LEGEND_MARGIN
    return random.randint(base, base + 2), random.randint(base, base + 2)


def _choose_pattern_map() -> dict[int, frozenset[tuple[int, int]]]:
    colors = random.sample(list(_SEED_COLORS), 4)
    l_pattern = random.choice(_L_PATTERNS)
    t_pattern = random.choice(_T_PATTERNS)
    extra_choices = [p for p in [*_L_PATTERNS, *_T_PATTERNS] if p not in {l_pattern, t_pattern}]
    chosen = [l_pattern, t_pattern, _CROSS_PATTERN, random.choice(extra_choices)]
    random.shuffle(chosen)
    return {color: pattern for color, pattern in zip(colors, chosen)}


def _draw_legend(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    patterns: dict[int, frozenset[tuple[int, int]]],
) -> None:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    row_origins = [r0 + _LEGEND_MARGIN, r0 + h - _LEGEND_SLOT - _LEGEND_MARGIN]
    col_origins = [c0 + _LEGEND_MARGIN, c0 + w - _LEGEND_SLOT - _LEGEND_MARGIN]
    slots = [(rr, cc) for rr in row_origins for cc in col_origins]
    random.shuffle(slots)

    for (color, directions), (top, left) in zip(patterns.items(), slots):
        seed_r = top + _LEGEND_SLOT // 2
        seed_c = left + _LEGEND_SLOT // 2
        grid[seed_r, seed_c] = color
        for dr, dc in directions:
            arm_length = random.randint(1, 2)
            for step in range(1, arm_length + 1):
                grid[seed_r + dr * step, seed_c + dc * step] = color


def _distance_to_edge(
    box: tuple[int, int, int, int],
    seed: tuple[int, int],
    direction: tuple[int, int],
) -> int:
    r0, c0, r1, c1 = box
    sr, sc = seed
    dr, dc = direction
    if dr == -1:
        return sr - r0
    if dr == 1:
        return r1 - sr
    if dc == -1:
        return sc - c0
    return c1 - sc


def _place_region_contents(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    seed_color: int,
    directions: frozenset[tuple[int, int]],
) -> tuple[tuple[int, int], int]:
    r0, c0, r1, c1 = box
    candidate_rows = range(r0 + 1, r1)
    candidate_cols = range(c0 + 1, c1)
    seeds = [(r, c) for r in candidate_rows for c in candidate_cols]
    assert seeds, "region must have an interior cell for the seed"
    seed = random.choice(seeds)
    sr, sc = seed
    grid[sr, sc] = seed_color

    mode = random.choice(("none", "ray_blocks", "mixed"))
    obstacle_count = 0
    for direction in directions:
        dist = _distance_to_edge(box, seed, direction)
        if dist < 2:
            continue
        if mode == "none":
            continue
        if mode == "ray_blocks" and random.random() < 0.65:
            stop = random.randint(2, dist)
        elif mode == "mixed" and random.random() < 0.45:
            stop = random.randint(2, dist)
        else:
            continue
        dr, dc = direction
        br = sr + dr * stop
        bc = sc + dc * stop
        grid[br, bc] = _OBSTACLE
        obstacle_count += 1

    extra_candidates = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if grid[r, c] == _BG
    ]
    random.shuffle(extra_candidates)
    extra_budget = random.randint(0, 1 if mode == "none" else 2)
    for r, c in extra_candidates[:extra_budget]:
        grid[r, c] = _OBSTACLE
        obstacle_count += 1

    return seed, obstacle_count


def _expected_output(
    inp: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    placements: list[tuple[tuple[int, int, int, int], tuple[int, int], int, frozenset[tuple[int, int]]]],
) -> np.ndarray:
    out = inp.copy()
    for box, seed, color, directions in placements:
        r0, c0, r1, c1 = box
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        local_seed = (seed[0] - r0, seed[1] - c0)
        grown = _grow_region(sub.copy(), local_seed, color, directions, _BG, _OBSTACLE)
        out[r0 : r1 + 1, c0 : c1 + 1] = grown
    return out


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    row_count, col_count = random.choice(_LAYOUTS)
    legend_h, legend_w = _legend_size()
    heights = [legend_h] + [random.randint(_OTHER_SPAN_MIN, _OTHER_SPAN_MAX) for _ in range(row_count - 1)]
    widths = [legend_w] + [random.randint(_OTHER_SPAN_MIN, _OTHER_SPAN_MAX) for _ in range(col_count - 1)]

    grid_h = sum(heights) + row_count - 1
    grid_w = sum(widths) + col_count - 1
    grid = np.full((grid_h, grid_w), _BG, dtype=int)

    row_cursor = 0
    for h in heights[:-1]:
        row_cursor += h
        grid[row_cursor, :] = _LINE
        row_cursor += 1
    col_cursor = 0
    for w in widths[:-1]:
        col_cursor += w
        grid[:, col_cursor] = _LINE
        col_cursor += 1

    row_spans, col_spans = _partition_spans(grid, _LINE)
    boxes = _region_boxes(row_spans, col_spans)
    legend_box = boxes[0]

    pattern_map = _choose_pattern_map()
    _draw_legend(grid, legend_box, pattern_map)

    ordinary_boxes = boxes[1:]
    colors = list(pattern_map)
    assignments = colors[:]
    while len(assignments) < len(ordinary_boxes):
        assignments.append(random.choice(colors))
    random.shuffle(assignments)

    placements = []
    total_obstacles = 0
    for box, color in zip(ordinary_boxes, assignments):
        seed, obstacle_count = _place_region_contents(grid, box, color, pattern_map[color])
        placements.append((box, seed, color, pattern_map[color]))
        total_obstacles += obstacle_count

    if total_obstacles == 0 and random.random() < 0.5:
        box, seed, color, directions = random.choice(placements)
        viable = [direction for direction in directions if _distance_to_edge(box, seed, direction) >= 2]
        if viable:
            dr, dc = random.choice(viable)
            stop = random.randint(2, _distance_to_edge(box, seed, (dr, dc)))
            grid[seed[0] + dr * stop, seed[1] + dc * stop] = _OBSTACLE
            total_obstacles = 1

    expected = _expected_output(grid, row_spans, col_spans, placements)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        built = _build_candidate()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
