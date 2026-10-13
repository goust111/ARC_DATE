from __future__ import annotations

import random

import numpy as np


def _color_counts(grid: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(grid, return_counts=True)
    return {int(v): int(n) for v, n in zip(values, counts)}


def _background_color(grid: np.ndarray) -> int:
    counts = _color_counts(grid)
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "expected a unique dominant background color"
    return int(winners[0])


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = _background_color(g)
    counts = _color_counts(g)

    non_bg = [color for color in counts if color != bg]
    assert len(non_bg) >= 3, "expected start marker plus at least two instruction colors"

    start_colors = [color for color in non_bg if counts[color] == 1]
    assert len(start_colors) == 1, "expected one unique single-pixel start color"
    start_color = int(start_colors[0])
    start_pos = tuple(map(int, np.argwhere(g == start_color)[0]))

    instruction_colors = set(non_bg) - {start_color}
    palette_candidates: list[tuple[int, list[int], dict[int, int]]] = []
    for palette_r in (0, height - 1):
        order: list[int] = []
        palette_cols: dict[int, int] = {}
        for c in range(width):
            color = int(g[palette_r, c])
            if color == bg:
                continue
            if color in palette_cols:
                order = []
                break
            palette_cols[color] = c
            order.append(color)
        if len(order) < 2:
            continue
        if set(order) != instruction_colors:
            continue
        palette_candidates.append((int(palette_r), order, palette_cols))

    assert len(palette_candidates) == 1, "expected exactly one palette row on the top or bottom edge"
    palette_r, palette_order, palette_cols = palette_candidates[0]

    instructions: list[tuple[int, int, int, int]] = []
    for color in palette_order:
        palette_c = int(palette_cols[color])
        coords = [
            (int(r), int(c))
            for r, c in np.argwhere(g == color)
            if not (int(r) == palette_r and int(c) == palette_c)
        ]
        assert 2 <= len(coords) <= 5, "each instruction segment must have length 2..5"

        rows = sorted({r for r, _c in coords})
        cols = sorted({c for _r, c in coords})
        if len(rows) == 1:
            seg_cols = sorted(c for _r, c in coords)
            assert seg_cols == list(range(seg_cols[0], seg_cols[-1] + 1))
            if seg_cols[-1] < palette_c:
                dr, dc = 0, -1
            elif seg_cols[0] > palette_c:
                dr, dc = 0, 1
            else:
                raise AssertionError("horizontal segment must lie strictly left or right of its palette cell")
            length = len(seg_cols)
        elif len(cols) == 1:
            seg_rows = sorted(r for r, _c in coords)
            assert seg_rows == list(range(seg_rows[0], seg_rows[-1] + 1))
            if seg_rows[-1] < palette_r:
                dr, dc = -1, 0
            elif seg_rows[0] > palette_r:
                dr, dc = 1, 0
            else:
                raise AssertionError("vertical segment must lie strictly above or below its palette cell")
            length = len(seg_rows)
        else:
            raise AssertionError("instruction object must be a straight line segment")

        instructions.append((int(color), int(dr), int(dc), int(length)))

    return {
        "shape": (int(height), int(width)),
        "bg": int(bg),
        "start_color": int(start_color),
        "start_pos": (int(start_pos[0]), int(start_pos[1])),
        "palette_row": int(palette_r),
        "instructions": instructions,
    }


def _render_path(
    shape: tuple[int, int],
    bg: int,
    start_color: int,
    start_pos: tuple[int, int],
    instructions: list[tuple[int, int, int, int]],
) -> np.ndarray:
    height, width = shape
    out = np.full((height, width), bg, dtype=int)
    cur_r, cur_c = map(int, start_pos)
    out[cur_r, cur_c] = int(start_color)

    for color, dr, dc, length in instructions:
        for _ in range(int(length)):
            cur_r += int(dr)
            cur_c += int(dc)
            assert 0 <= cur_r < height and 0 <= cur_c < width, "decoded path must stay inside the grid"
            if int(out[cur_r, cur_c]) == int(bg):
                out[cur_r, cur_c] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode scattered colored segments into a palette-ordered drawing program.

    Each palette color names one segment. The segment orientation chooses whether
    the brush moves horizontally or vertically, its length gives the step count,
    and its position relative to the palette cell gives the sign of the motion.
    The brush starts at the unique singleton marker. Already painted cells are
    never recolored, but the brush still moves through them.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    return _render_path(
        scene["shape"],
        scene["bg"],
        scene["start_color"],
        scene["start_pos"],
        scene["instructions"],
    )


GRID_H_RANGE = (15, 26)
GRID_W_RANGE = (16, 30)
INSTRUCTION_COUNT_RANGE = (4, 6)
SEGMENT_LENGTH_RANGE = (2, 5)
MAX_VERTICAL_SEGMENTS = 2
PALETTE_MIN_SEP = 2
GEN_RETRIES = 500


def _step(axis: str, sign: int) -> tuple[int, int]:
    return (0, int(sign)) if axis == "H" else (int(sign), 0)


def _trace_offsets(specs: list[tuple[str, int, int]]) -> list[tuple[int, int]]:
    cur_r = 0
    cur_c = 0
    visited = [(0, 0)]
    for axis, sign, length in specs:
        dr, dc = _step(axis, sign)
        for _ in range(length):
            cur_r += dr
            cur_c += dc
            visited.append((cur_r, cur_c))
    return visited


def _sample_specs(top_palette: bool) -> list[tuple[str, int, int]]:
    vertical_sign = 1 if top_palette else -1
    n_instructions = random.randint(*INSTRUCTION_COUNT_RANGE)
    retrace = random.random() < 0.5

    specs: list[tuple[str, int, int]] = []
    if retrace:
        sign = random.choice([-1, 1])
        length_a = random.randint(3, SEGMENT_LENGTH_RANGE[1])
        length_b = random.randint(2, length_a)
        specs.extend(
            [
                ("H", sign, length_a),
                ("H", -sign, length_b),
                ("V", vertical_sign, random.randint(*SEGMENT_LENGTH_RANGE)),
            ]
        )
    else:
        if random.random() < 0.5:
            specs.extend(
                [
                    ("H", -1, random.randint(*SEGMENT_LENGTH_RANGE)),
                    ("V", vertical_sign, random.randint(*SEGMENT_LENGTH_RANGE)),
                    ("H", 1, random.randint(*SEGMENT_LENGTH_RANGE)),
                ]
            )
        else:
            specs.extend(
                [
                    ("H", 1, random.randint(*SEGMENT_LENGTH_RANGE)),
                    ("V", vertical_sign, random.randint(*SEGMENT_LENGTH_RANGE)),
                    ("H", -1, random.randint(*SEGMENT_LENGTH_RANGE)),
                ]
            )

    vertical_count = 1
    while len(specs) < n_instructions:
        can_add_vertical = vertical_count < MAX_VERTICAL_SEGMENTS
        add_vertical = can_add_vertical and random.random() < 0.35
        if add_vertical:
            specs.append(("V", vertical_sign, random.randint(*SEGMENT_LENGTH_RANGE)))
            vertical_count += 1
        else:
            specs.append(("H", random.choice([-1, 1]), random.randint(*SEGMENT_LENGTH_RANGE)))

    return specs


def _choose_palette_columns(specs: list[tuple[str, int, int]], width: int) -> list[int] | None:
    intervals: list[tuple[int, int]] = []
    for axis, sign, length in specs:
        if axis == "H" and sign < 0:
            lo, hi = length + 1, width - 2
        elif axis == "H" and sign > 0:
            lo, hi = 1, width - length - 2
        else:
            lo, hi = 1, width - 2
        if lo > hi:
            return None
        intervals.append((int(lo), int(hi)))

    chosen: list[int] = []

    def dfs(index: int, prev_col: int) -> bool:
        if index == len(intervals):
            return True

        lo, hi = intervals[index]
        lo = max(lo, prev_col + PALETTE_MIN_SEP)
        candidates = list(range(lo, hi + 1))
        random.shuffle(candidates)
        for col in candidates:
            chosen.append(int(col))
            if dfs(index + 1, int(col)):
                return True
            chosen.pop()
        return False

    if not dfs(0, -PALETTE_MIN_SEP):
        return None
    return chosen


def _expand_with_margin(cells: list[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _segment_candidates(
    height: int,
    width: int,
    axis: str,
    sign: int,
    length: int,
    palette_c: int,
) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []

    if axis == "H":
        rows = list(range(1, height - 1))
        random.shuffle(rows)
        if sign < 0:
            ends = list(range(length, palette_c))
            random.shuffle(ends)
            for row in rows:
                for end in ends:
                    start = end - length + 1
                    cells = [(int(row), int(c)) for c in range(start, end + 1)]
                    candidates.append(cells)
        else:
            max_start = width - length - 1
            starts = list(range(palette_c + 1, max_start + 1))
            random.shuffle(starts)
            for row in rows:
                for start in starts:
                    cells = [(int(row), int(c)) for c in range(start, start + length)]
                    candidates.append(cells)
        return candidates

    cols = list(range(1, width - 1))
    random.shuffle(cols)
    if sign > 0:
        starts = list(range(2, height - length))
    else:
        starts = list(range(1, height - length - 1))
    random.shuffle(starts)
    for col in cols:
        for start in starts:
            cells = [(int(start + k), int(col)) for k in range(length)]
            candidates.append(cells)
    return candidates


def _relative_bounds(specs: list[tuple[str, int, int]]) -> tuple[int, int, int, int]:
    visited = _trace_offsets(specs)
    rows = [r for r, _c in visited]
    cols = [c for _r, c in visited]
    return min(rows), max(rows), min(cols), max(cols)


def _choose_start(
    height: int,
    width: int,
    bounds: tuple[int, int, int, int],
    blocked: set[tuple[int, int]],
) -> tuple[int, int] | None:
    min_r, max_r, min_c, max_c = bounds
    row_lo = 1 - min_r
    row_hi = height - 2 - max_r
    col_lo = 1 - min_c
    col_hi = width - 2 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return None

    candidates = [
        (int(r), int(c))
        for r in range(row_lo, row_hi + 1)
        for c in range(col_lo, col_hi + 1)
        if (int(r), int(c)) not in blocked
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _try_generate() -> np.ndarray | None:
    top_palette = random.random() < 0.5
    specs = _sample_specs(top_palette)
    bounds = _relative_bounds(specs)

    min_r, max_r, min_c, max_c = bounds
    need_h = max_r - min_r + 5
    need_w = max_c - min_c + 7
    height_lo = max(GRID_H_RANGE[0], need_h)
    width_lo = max(GRID_W_RANGE[0], need_w, len(specs) * PALETTE_MIN_SEP + 4)
    if height_lo > GRID_H_RANGE[1] or width_lo > GRID_W_RANGE[1]:
        return None

    height = random.randint(height_lo, GRID_H_RANGE[1])
    width = random.randint(width_lo, GRID_W_RANGE[1])

    palette_cols = _choose_palette_columns(specs, width)
    if palette_cols is None:
        return None

    all_colors = list(range(10))
    bg = random.choice(all_colors)
    other_colors = random.sample([color for color in all_colors if color != bg], len(specs) + 1)
    instruction_colors = [int(color) for color in other_colors[:-1]]
    start_color = int(other_colors[-1])
    instructions = [
        (int(color), axis, int(sign), int(length))
        for color, (axis, sign, length) in zip(instruction_colors, specs)
    ]

    palette_edge = 0 if top_palette else height - 1
    palette_cells = [(palette_edge, int(c)) for c in palette_cols]
    blocked = _expand_with_margin(palette_cells, height, width)
    start_pos = _choose_start(height, width, bounds, blocked)
    if start_pos is None:
        return None
    blocked |= _expand_with_margin([start_pos], height, width)

    segment_cells: dict[int, list[tuple[int, int]]] = {}
    placement_order = sorted(
        range(len(instructions)),
        key=lambda idx: instructions[idx][3],
        reverse=True,
    )
    for idx in placement_order:
        color, axis, sign, length = instructions[idx]
        palette_c = int(palette_cols[idx])
        candidates = _segment_candidates(height, width, axis, sign, length, palette_c)
        random.shuffle(candidates)
        placed = None
        for cells in candidates:
            if any(cell in blocked for cell in cells):
                continue
            placed = cells
            break
        if placed is None:
            return None
        segment_cells[int(color)] = placed
        blocked |= _expand_with_margin(placed, height, width)

    grid = np.full((height, width), bg, dtype=int)
    grid[start_pos] = start_color
    for color, col in zip(instruction_colors, palette_cols):
        grid[palette_edge, int(col)] = int(color)
    for color, cells in segment_cells.items():
        for r, c in cells:
            grid[r, c] = int(color)

    solver_instructions = [(color, _step(axis, sign)[0], _step(axis, sign)[1], length) for color, axis, sign, length in instructions]
    expected = _render_path((height, width), bg, start_color, start_pos, solver_instructions)

    counts = _color_counts(grid)
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    scene = _extract_scene(grid)
    if scene["palette_row"] != palette_edge:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
