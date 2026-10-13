from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

GEN_HEIGHT_RANGE = (20, 28)
GEN_WIDTH_RANGE = (20, 28)
GEN_COLOR_COUNT_RANGE = (2, 4)
GEN_LAYOUT_ATTEMPTS = 120
GEN_MAX_ATTEMPTS = 500
GEN_COVERAGE_RANGE = (0.08, 0.58)

COUNT_PAIRS_BY_TOTAL = {
    2: ((1, 1),),
    3: ((1, 2), (2, 1)),
    4: ((1, 3), (2, 2), (3, 1)),
    5: ((2, 3), (3, 2)),
    6: ((3, 3),),
}
TIE_TOTALS = tuple(total for total, pairs in COUNT_PAIRS_BY_TOTAL.items() if len(pairs) >= 2)


def _axis_groups(grid: np.ndarray) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    left: dict[int, list[int]] = defaultdict(list)
    top: dict[int, list[int]] = defaultdict(list)

    for r in range(1, h):
        color = int(g[r, 0])
        if color != BG:
            left[color].append(r)

    for c in range(1, w):
        color = int(g[0, c])
        if color != BG:
            top[color].append(c)

    return dict(left), dict(top)


def _sorted_specs_from_groups(
    rows_by_color: dict[int, list[int]], cols_by_color: dict[int, list[int]]
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for color in sorted(set(rows_by_color) & set(cols_by_color)):
        rows = sorted(int(r) for r in rows_by_color[color])
        cols = sorted(int(c) for c in cols_by_color[color])
        size = len(rows) + len(cols)
        specs.append(
            {
                "color": int(color),
                "rows": tuple(rows),
                "cols": tuple(cols),
                "size": int(size),
                "first_row": int(rows[0]),
            }
        )

    specs.sort(key=lambda spec: (-int(spec["size"]), -int(spec["first_row"])))
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use matching edge-axis markers to stamp overlapping solid squares whose size is set by repeated marker counts.

    Input structure:
    - The background is black.
    - The top row and left column act as two colored tick axes, while the remaining cells are the drawing canvas.
    - Each non-black pixel on the left axis marks one row, and each non-black pixel on the top axis marks one column.
    - Every axis color appears at least once on both axes, and a color may repeat on either axis.

    Rule:
    - For each color C that appears on both axes, count its left-axis marks Nr and top-axis marks Nc.
    - Let the square side length be S = Nr + Nc.
    - For every combination of a left-axis row r of color C and a top-axis column c of color C, draw a solid S x S square of color C with top-left corner at (r, c).
    - If a square extends beyond the canvas, keep only the in-bounds part.
    - Draw colors in descending S order; if two colors have the same S, draw the one whose first left-axis occurrence is lower in the grid first. Later squares overwrite earlier ones.

    Key insight:
    - Same-color row ticks and column ticks define square anchors by Cartesian product, repeated marks on both axes add together to determine square size, and overlap is resolved by a size-first then left-axis-order painting priority.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    rows_by_color, cols_by_color = _axis_groups(g)
    for spec in _sorted_specs_from_groups(rows_by_color, cols_by_color):
        color = int(spec["color"])
        size = int(spec["size"])
        rows = spec["rows"]
        cols = spec["cols"]
        for r in rows:
            r1 = min(h, int(r) + size)
            for c in cols:
                c1 = min(w, int(c) + size)
                out[int(r) : r1, int(c) : c1] = color

    return out


def _safe_positions(limit: int, size: int) -> list[int]:
    return list(range(1, limit - size + 1))


def _clip_positions(limit: int, size: int) -> list[int]:
    return list(range(max(1, limit - size + 1), limit))


def _intervals_overlap(p0: int, size0: int, p1: int, size1: int) -> bool:
    return p0 < p1 + size1 and p1 < p0 + size0


def _choose_focus_pair(
    colors: list[int], counts_by_color: dict[int, tuple[int, int]], force_tie: bool
) -> tuple[int, int]:
    if force_tie:
        return colors[0], colors[1]

    sizes = {color: counts_by_color[color][0] + counts_by_color[color][1] for color in colors}
    ordered = sorted(colors, key=lambda color: (sizes[color], color))
    return ordered[-1], ordered[0]


def _sample_count_pairs(
    colors: list[int], force_tie: bool, rng: random.Random
) -> dict[int, tuple[int, int]]:
    counts_by_color: dict[int, tuple[int, int]] = {}
    if force_tie:
        tie_total = rng.choice(TIE_TOTALS)
        tie_pairs = list(COUNT_PAIRS_BY_TOTAL[tie_total])
        pair_a, pair_b = rng.sample(tie_pairs, 2)
        counts_by_color[colors[0]] = pair_a
        counts_by_color[colors[1]] = pair_b

        remaining_totals = [total for total in COUNT_PAIRS_BY_TOTAL if total != tie_total]
        rng.shuffle(remaining_totals)
        for color, total in zip(colors[2:], remaining_totals):
            counts_by_color[color] = rng.choice(COUNT_PAIRS_BY_TOTAL[total])
        return counts_by_color

    totals = rng.sample(sorted(COUNT_PAIRS_BY_TOTAL), len(colors))
    for color, total in zip(colors, totals):
        counts_by_color[color] = rng.choice(COUNT_PAIRS_BY_TOTAL[total])
    return counts_by_color


def _choose_overlapping_focus_positions(
    limit: int, size_a: int, size_b: int, rng: random.Random
) -> tuple[int, int] | None:
    safe_a = _safe_positions(limit, size_a)
    safe_b = _safe_positions(limit, size_b)
    for _ in range(80):
        pos_a = rng.choice(safe_a)
        candidates_b = [
            pos_b
            for pos_b in safe_b
            if pos_b != pos_a and _intervals_overlap(pos_a, size_a, pos_b, size_b)
        ]
        if candidates_b:
            return pos_a, rng.choice(candidates_b)
    return None


def _clip_choices(
    colors: list[int],
    counts_by_color: dict[int, tuple[int, int]],
    sizes: dict[int, int],
    focus_pair: tuple[int, int],
    h: int,
    w: int,
) -> list[tuple[str, int]]:
    choices: list[tuple[str, int]] = []
    focus_set = set(focus_pair)
    for color in colors:
        nr, nc = counts_by_color[color]
        row_reserved = 1 if color in focus_set else 0
        col_reserved = 1 if color in focus_set else 0
        if _clip_positions(h, sizes[color]) and nr > row_reserved:
            choices.append(("row", color))
        if _clip_positions(w, sizes[color]) and nc > col_reserved:
            choices.append(("col", color))
    return choices


def _place_axis_positions(
    *,
    limit: int,
    counts: dict[int, int],
    sizes: dict[int, int],
    colors: list[int],
    focus_pair: tuple[int, int],
    clip_assignment: tuple[str, int] | None,
    rng: random.Random,
    is_row_axis: bool,
) -> dict[int, list[int]] | None:
    positions: dict[int, list[int]] = {color: [] for color in colors}
    used: set[int] = set()
    focus_first: dict[int, int] = {}

    color_a, color_b = focus_pair
    pair = _choose_overlapping_focus_positions(limit, sizes[color_a], sizes[color_b], rng)
    if pair is None:
        return None
    pos_a, pos_b = pair
    positions[color_a].append(pos_a)
    positions[color_b].append(pos_b)
    used.update((pos_a, pos_b))
    focus_first[color_a] = pos_a
    focus_first[color_b] = pos_b

    clip_dim = None if clip_assignment is None else clip_assignment[0]
    clip_color = None if clip_assignment is None else clip_assignment[1]
    axis_name = "row" if is_row_axis else "col"

    for color in colors:
        need = counts[color] - len(positions[color])
        if need < 0:
            return None

        if clip_dim == axis_name and color == clip_color:
            if need <= 0:
                return None
            clip_candidates = [pos for pos in _clip_positions(limit, sizes[color]) if pos not in used]
            if is_row_axis and color in focus_first:
                clip_candidates = [pos for pos in clip_candidates if pos > focus_first[color]]
            if not clip_candidates:
                return None
            chosen = rng.choice(clip_candidates)
            positions[color].append(chosen)
            used.add(chosen)
            need -= 1

        safe_candidates = [pos for pos in _safe_positions(limit, sizes[color]) if pos not in used]
        if is_row_axis and color in focus_first:
            safe_candidates = [pos for pos in safe_candidates if pos > focus_first[color]]
        if len(safe_candidates) < need:
            return None
        if need:
            picked = rng.sample(safe_candidates, need)
            positions[color].extend(picked)
            used.update(picked)

    for color in colors:
        positions[color].sort()

    if len(used) != sum(counts.values()):
        return None

    return positions


def _build_grid(
    h: int,
    w: int,
    rows_by_color: dict[int, list[int]],
    cols_by_color: dict[int, list[int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for color, rows in rows_by_color.items():
        for r in rows:
            grid[int(r), 0] = int(color)
    for color, cols in cols_by_color.items():
        for c in cols:
            grid[0, int(c)] = int(color)
    return grid


def _coverage_sets(specs: list[dict[str, object]], h: int, w: int) -> list[list[set[int]]]:
    cover = [[set() for _ in range(w)] for _ in range(h)]
    for spec in specs:
        color = int(spec["color"])
        size = int(spec["size"])
        for r in spec["rows"]:
            r1 = min(h, int(r) + size)
            for c in spec["cols"]:
                c1 = min(w, int(c) + size)
                for rr in range(int(r), r1):
                    for cc in range(int(c), c1):
                        cover[rr][cc].add(color)
    return cover


def _has_clipped_square(specs: list[dict[str, object]], h: int, w: int) -> bool:
    for spec in specs:
        size = int(spec["size"])
        for r in spec["rows"]:
            for c in spec["cols"]:
                if int(r) + size > h or int(c) + size > w:
                    return True
    return False


def _is_valid_instance(
    grid: np.ndarray,
    *,
    counts_by_color: dict[int, tuple[int, int]],
    focus_pair: tuple[int, int],
    force_tie: bool,
    force_clip: bool,
) -> bool:
    h, w = grid.shape
    rows_by_color, cols_by_color = _axis_groups(grid)
    if set(rows_by_color) != set(cols_by_color):
        return False

    specs = _sorted_specs_from_groups(rows_by_color, cols_by_color)
    if len(specs) != len(counts_by_color):
        return False

    size_by_color: dict[int, int] = {}
    for spec in specs:
        color = int(spec["color"])
        rows = tuple(int(r) for r in spec["rows"])
        cols = tuple(int(c) for c in spec["cols"])
        if counts_by_color[color] != (len(rows), len(cols)):
            return False
        size_by_color[color] = int(spec["size"])

    if force_tie:
        if size_by_color[focus_pair[0]] != size_by_color[focus_pair[1]]:
            return False
    elif len(set(size_by_color.values())) != len(size_by_color):
        return False

    clipped = _has_clipped_square(specs, h, w)
    if clipped != force_clip:
        return False

    cover = _coverage_sets(specs, h, w)
    if not any({focus_pair[0], focus_pair[1]} <= cover[r][c] for r in range(1, h) for c in range(1, w)):
        return False

    canvas_overlap = any(len(cover[r][c]) >= 2 for r in range(1, h) for c in range(1, w))
    if not canvas_overlap:
        return False

    out = main(grid)
    canvas = out[1:, 1:]
    if not np.any(canvas != BG):
        return False

    for color in counts_by_color:
        if not np.any(canvas == int(color)):
            return False

    coverage = float(np.count_nonzero(canvas)) / float(canvas.size)
    if not (GEN_COVERAGE_RANGE[0] <= coverage <= GEN_COVERAGE_RANGE[1]):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GEN_HEIGHT_RANGE)
    w = rng.randint(*GEN_WIDTH_RANGE)
    force_tie = rng.random() < 0.5
    force_clip = rng.random() < 0.5

    color_count = rng.randint(*GEN_COLOR_COUNT_RANGE)
    colors = rng.sample(PALETTE, color_count)
    counts_by_color = _sample_count_pairs(colors, force_tie, rng)
    focus_pair = _choose_focus_pair(colors, counts_by_color, force_tie)
    sizes = {color: counts_by_color[color][0] + counts_by_color[color][1] for color in colors}

    clip_assignment = None
    if force_clip:
        choices = _clip_choices(colors, counts_by_color, sizes, focus_pair, h, w)
        if not choices:
            return None
        clip_assignment = rng.choice(choices)

    row_counts = {color: counts_by_color[color][0] for color in colors}
    col_counts = {color: counts_by_color[color][1] for color in colors}

    for _ in range(GEN_LAYOUT_ATTEMPTS):
        rows_by_color = _place_axis_positions(
            limit=h,
            counts=row_counts,
            sizes=sizes,
            colors=colors,
            focus_pair=focus_pair,
            clip_assignment=clip_assignment,
            rng=rng,
            is_row_axis=True,
        )
        if rows_by_color is None:
            continue

        cols_by_color = _place_axis_positions(
            limit=w,
            counts=col_counts,
            sizes=sizes,
            colors=colors,
            focus_pair=focus_pair,
            clip_assignment=clip_assignment,
            rng=rng,
            is_row_axis=False,
        )
        if cols_by_color is None:
            continue

        grid = _build_grid(h, w, rows_by_color, cols_by_color)
        if _is_valid_instance(
            grid,
            counts_by_color=counts_by_color,
            focus_pair=focus_pair,
            force_tie=force_tie,
            force_clip=force_clip,
        ):
            return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid input grid")
