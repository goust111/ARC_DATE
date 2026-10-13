from __future__ import annotations

import random

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _row_segments(row: np.ndarray, wall_color: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    c = 0
    width = row.shape[0]
    while c < width:
        if int(row[c]) != wall_color:
            c += 1
            continue
        c0 = c
        while c + 1 < width and int(row[c + 1]) == wall_color:
            c += 1
        segments.append((c0, c))
        c += 1
    return segments


def _segment_containing(segments: list[tuple[int, int]], col: int) -> tuple[int, int] | None:
    for c0, c1 in segments:
        if c0 <= col <= c1:
            return (c0, c1)
    return None


def _paint_if_bg(out: np.ndarray, bg: int, r: int, c: int, color: int) -> None:
    if color != bg and int(out[r, c]) == bg:
        out[r, c] = color


def _parse_puzzle(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert 5 <= h <= 30 and 2 <= w <= 30, "grid must fit ARC bounds"

    bg = _mode_color(g)

    code_non_bg = np.flatnonzero(g[0] != bg)
    assert code_non_bg.size > 0, "palette row needs an effective code segment"
    palette_end = int(code_non_bg[-1])
    palette = [int(v) for v in g[0, : palette_end + 1]]

    separator_colors = {int(v) for v in g[1]}
    assert len(separator_colors) == 1, "separator row must be uniform"
    separator_color = next(iter(separator_colors))
    assert separator_color != bg, "separator row cannot be background"

    wall_rows: list[int] = []
    wall_color: int | None = None
    segments_by_row: dict[int, list[tuple[int, int]]] = {}
    for r in range(2, h - 1):
        row = g[r]
        non_bg = np.flatnonzero(row != bg)
        if non_bg.size == 0:
            continue
        colors = {int(v) for v in row[non_bg]}
        assert len(colors) == 1, "each wall row must use one wall color"
        color = next(iter(colors))
        if wall_color is None:
            wall_color = color
        assert color == wall_color, "all wall rows must share one wall color"
        assert 0 < non_bg.size < w, "each wall row must be broken rather than solid"
        segments = _row_segments(row, wall_color)
        assert segments, "wall row must contain at least one wall segment"
        segments_by_row[r] = segments
        wall_rows.append(r)

    assert wall_color is not None and wall_rows, "expected at least one broken wall row"
    assert len(palette) == len(wall_rows) + 1, "palette depth count must match wall count + 1"

    bottom_non_bg = np.argwhere(g[h - 1] != bg)
    assert bottom_non_bg.shape[0] == 1, "expected exactly one bottom-edge marker"
    marker_col = int(bottom_non_bg[0, 0])
    marker = (h - 1, marker_col)
    assert int(g[marker]) == palette[0] != bg, "marker color must equal the first code color"

    for r in range(2, h - 1):
        allowed = {bg}
        if r in segments_by_row:
            allowed.add(wall_color)
        assert {int(v) for v in g[r]} <= allowed, "field rows may only contain background or wall cells"

    return {
        "grid": g,
        "bg": bg,
        "palette": palette,
        "separator_color": separator_color,
        "wall_color": wall_color,
        "wall_rows_desc": sorted(wall_rows, reverse=True),
        "segments_by_row": segments_by_row,
        "marker": marker,
    }


def _solve_with_palette(parsed: dict, palette: list[int]) -> np.ndarray:
    g = parsed["grid"]
    bg = parsed["bg"]
    wall_color = parsed["wall_color"]
    segments_by_row = parsed["segments_by_row"]
    out = g.copy()
    visited: set[tuple[int, int, int]] = set()

    def walk(r: int, c: int, depth: int) -> None:
        state = (r, c, depth)
        if state in visited:
            return
        visited.add(state)

        color = int(palette[depth])
        cur_r = int(r)
        while True:
            if out[cur_r, c] == bg:
                _paint_if_bg(out, bg, cur_r, c, color)
            next_r = cur_r - 1
            if next_r < 2:
                return
            if int(g[next_r, c]) == wall_color:
                segment = _segment_containing(segments_by_row[next_r], c)
                assert segment is not None, "a branch must hit one wall segment"
                c0, c1 = segment
                for cc in range(c0, c1 + 1):
                    _paint_if_bg(out, bg, cur_r, cc, color)
                next_depth = depth + 1
                assert next_depth < len(palette), "crossing count cannot exceed palette depth"
                for gap_c in (c0 - 1, c1 + 1):
                    if 0 <= gap_c < g.shape[1] and int(g[next_r, gap_c]) == bg:
                        walk(next_r, gap_c, next_depth)
                return
            assert int(g[next_r, c]) == bg, "vertical continuation may only pass through background"
            cur_r = next_r

    walk(parsed["marker"][0], parsed["marker"][1], 0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Percolate the bottom marker upward, switching palette depth each time the path passes a broken wall.

    Input structure
    - The top row is a palette row; the second row is a full-width separator; the field is below them.
    - The field has a dominant background, several horizontal broken wall rows in one shared wall color, and a single bottom-edge marker.
    - The effective palette segment runs from the left edge to the last non-background cell of the top row, may contain background blanks inside, and has exactly one slot more than the number of wall rows.

    Rule
    - Keep the palette row, separator row, wall cells, and marker unchanged.
    - Starting from the marker, move straight inward along its column and paint only background trajectory cells with the current depth color.
    - When the next cell would be a wall, spread along the near side of that continuous wall segment, then enter each adjacent opening cell at the segment ends and continue inward from there at the next depth.
    - If a palette slot is the background color, that depth still advances and branches, but its trajectory cells remain background.

    Key insight
    - The wall topology determines the reachable trajectory shape, while the palette row assigns colors by the number of walls already crossed rather than by geometry or object identity.
    """
    parsed = _parse_puzzle(grid)
    return _solve_with_palette(parsed, parsed["palette"])


ALL_COLORS = list(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _choose_wall_count() -> int:
    return random.choice([1, 2, 2, 3, 3])


def _choose_steps(wall_count: int) -> list[int]:
    if wall_count == 1:
        return [random.randint(2, 4)]
    if wall_count == 2:
        top_step = random.randint(1, 2)
        bottom_step = random.randint(top_step + 1, top_step + 4)
        return [bottom_step, top_step]
    last_step = random.randint(1, 2)
    middle_step = random.randint(last_step + 1, last_step + 3)
    first_min = middle_step + last_step + 1
    first_step = random.randint(first_min, first_min + 3)
    return [first_step, middle_step, last_step]


def _choose_palette(wall_count: int, bg: int, wall_color: int, separator_color: int) -> list[int]:
    usable = [c for c in ALL_COLORS if c not in {bg, wall_color, separator_color}]
    random.shuffle(usable)

    palette = [usable.pop()]
    blank_positions = list(range(1, wall_count))
    force_blank = bool(blank_positions) and random.random() < 0.65
    forced_blank_index = random.choice(blank_positions) if force_blank else None

    for depth in range(1, wall_count):
        if depth == forced_blank_index:
            palette.append(bg)
        elif random.random() < 0.35:
            palette.append(bg)
        else:
            if not usable:
                usable = [c for c in ALL_COLORS if c not in {bg, wall_color, separator_color, palette[0]}]
                random.shuffle(usable)
            palette.append(usable.pop())

    remaining = [c for c in ALL_COLORS if c not in {bg, wall_color, separator_color, palette[0]}]
    random.shuffle(remaining)
    palette.append(remaining[0])
    return palette


def _choose_dimensions(steps: list[int]) -> tuple[int, int, list[int]] | None:
    wall_count = len(steps)
    top_gap = random.randint(1, 3)
    between_gaps = [random.randint(1, 3) for _ in range(wall_count - 1)]
    bottom_gap = random.randint(2, 4)

    wall_rows_asc: list[int] = []
    r = 2 + top_gap
    wall_rows_asc.append(r)
    for gap in between_gaps:
        r += gap + 1
        wall_rows_asc.append(r)
    height = wall_rows_asc[-1] + bottom_gap + 2
    assert height <= 30

    spread = sum(steps)
    min_width = max(10, 2 * spread + 1)
    if min_width > 30:
        return None
    width = random.randint(min_width, min(30, min_width + 8))
    return height, width, wall_rows_asc


def _build_core_plan(height: int, width: int, wall_rows_asc: list[int], steps: list[int]) -> dict | None:
    spread = sum(steps)
    marker_col = random.randint(spread, width - 1 - spread)
    wall_rows_desc = list(reversed(wall_rows_asc))

    transitions: list[dict[int, tuple[tuple[int, int], tuple[int, ...]]]] = []
    states_by_depth: list[tuple[int, ...]] = [(marker_col,)]
    active_cols = [marker_col]

    for depth, wall_r in enumerate(wall_rows_desc):
        step = steps[depth]
        row_transitions: dict[int, tuple[tuple[int, int], tuple[int, ...]]] = {}
        next_cols: list[int] = []

        for parent_col in active_cols:
            c0 = parent_col - step + 1
            c1 = parent_col + step - 1
            if not (0 <= c0 <= c1 < width):
                return None
            children = (parent_col - step, parent_col + step)
            if not (0 <= children[0] < children[1] < width):
                return None
            row_transitions[parent_col] = ((c0, c1), children)
            next_cols.extend(children)

        segments = sorted(spec[0] for spec in row_transitions.values())
        for (a0, a1), (b0, b1) in zip(segments, segments[1:]):
            if b0 - a1 < 2:
                return None

        active_cols = sorted(next_cols)
        if len(active_cols) != len(set(active_cols)):
            return None
        transitions.append(row_transitions)
        states_by_depth.append(tuple(active_cols))

    return {
        "height": height,
        "width": width,
        "wall_rows_asc": wall_rows_asc,
        "wall_rows_desc": wall_rows_desc,
        "marker_col": marker_col,
        "states_by_depth": states_by_depth,
        "transitions": transitions,
    }


def _random_extra_segments(width: int, core_segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    blocked = np.zeros(width, dtype=bool)
    for c0, c1 in core_segments:
        blocked[max(0, c0 - 2) : min(width, c1 + 3)] = True

    extras: list[tuple[int, int]] = []
    for _ in range(random.randint(0, 2)):
        for _ in range(30):
            seg_len = random.randint(1, min(4, width))
            c0 = random.randint(0, width - seg_len)
            c1 = c0 + seg_len - 1
            if np.any(blocked[max(0, c0 - 1) : min(width, c1 + 2)]):
                continue
            extras.append((c0, c1))
            blocked[max(0, c0 - 2) : min(width, c1 + 3)] = True
            break
    return sorted(extras)


def _render_input(plan: dict, bg: int, separator_color: int, wall_color: int, palette: list[int]) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    grid = np.full((plan["height"], plan["width"]), bg, dtype=int)
    grid[0, : len(palette)] = palette
    grid[1, :] = separator_color

    all_segments_by_row: dict[int, list[tuple[int, int]]] = {}
    for depth, wall_r in enumerate(plan["wall_rows_desc"]):
        core_segments = sorted(spec[0] for spec in plan["transitions"][depth].values())
        extras = _random_extra_segments(plan["width"], core_segments)
        all_segments = sorted(core_segments + extras)
        all_segments_by_row[wall_r] = all_segments
        for c0, c1 in all_segments:
            grid[wall_r, c0 : c1 + 1] = wall_color

    grid[-1, plan["marker_col"]] = palette[0]
    return grid, all_segments_by_row


def _render_from_plan(grid: np.ndarray, plan: dict, palette: list[int] | None = None) -> np.ndarray:
    palette = plan["palette"] if palette is None else palette
    out = np.array(grid, dtype=int, copy=True)
    bg = plan["bg"]
    visited: set[tuple[int, int]] = set()

    def walk(start_r: int, col: int, depth: int) -> None:
        state = (depth, col)
        if state in visited:
            return
        visited.add(state)

        color = int(palette[depth])
        if depth == len(plan["wall_rows_desc"]):
            for r in range(start_r, 1, -1):
                _paint_if_bg(out, bg, r, col, color)
            return

        wall_r = plan["wall_rows_desc"][depth]
        assert start_r > wall_r
        for r in range(start_r, wall_r, -1):
            _paint_if_bg(out, bg, r, col, color)

        segment, children = plan["transitions"][depth][col]
        c0, c1 = segment
        near_r = wall_r + 1
        for cc in range(c0, c1 + 1):
            _paint_if_bg(out, bg, near_r, cc, color)
        for child_col in children:
            walk(wall_r, child_col, depth + 1)

    walk(grid.shape[0] - 1, plan["marker_col"], 0)
    return out


def _wrong_straight_only(grid: np.ndarray, plan: dict) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    bg = plan["bg"]
    color = int(plan["palette"][0])
    col = plan["marker_col"]
    for r in range(grid.shape[0] - 1, 1, -1):
        _paint_if_bg(out, bg, r, col, color)
    return out


def _validate_instance(grid: np.ndarray, plan: dict, expected: np.ndarray) -> bool:
    try:
        parsed = _parse_puzzle(grid)
    except AssertionError:
        return False

    if parsed["bg"] != plan["bg"]:
        return False
    if parsed["palette"] != plan["palette"]:
        return False
    if parsed["wall_rows_desc"] != plan["wall_rows_desc"]:
        return False
    if parsed["marker"][1] != plan["marker_col"]:
        return False

    for depth, wall_r in enumerate(plan["wall_rows_desc"]):
        openings_from_row: list[int] = []
        for parent_col in plan["states_by_depth"][depth]:
            segment = _segment_containing(parsed["segments_by_row"][wall_r], parent_col)
            if segment != plan["transitions"][depth][parent_col][0]:
                return False
            c0, c1 = segment
            if not (0 <= c0 <= parent_col <= c1 < grid.shape[1]):
                return False
            if c0 == 0 or c1 == grid.shape[1] - 1:
                return False
            if int(grid[wall_r, c0 - 1]) != plan["bg"] or int(grid[wall_r, c1 + 1]) != plan["bg"]:
                return False
            openings_from_row.extend([c0 - 1, c1 + 1])
        if tuple(sorted(openings_from_row)) != plan["states_by_depth"][depth + 1]:
            return False

    if np.array_equal(grid, expected):
        return False
    if np.any(expected[0] != grid[0]) or np.any(expected[1] != grid[1]):
        return False
    for wall_r, segments in parsed["segments_by_row"].items():
        for c0, c1 in segments:
            if not np.array_equal(expected[wall_r, c0 : c1 + 1], grid[wall_r, c0 : c1 + 1]):
                return False

    flat_palette = [plan["palette"][0]] * len(plan["palette"])
    if np.array_equal(_render_from_plan(grid, plan, flat_palette), expected):
        return False
    if np.array_equal(_wrong_straight_only(grid, plan), expected):
        return False

    if any(color == plan["bg"] for color in plan["palette"][1:-1]):
        filled_palette = [plan["palette"][0]]
        for color in plan["palette"][1:]:
            filled_palette.append(plan["palette"][0] if color == plan["bg"] else color)
        if np.array_equal(_render_from_plan(grid, plan, filled_palette), expected):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    wall_count = _choose_wall_count()
    steps = _choose_steps(wall_count)
    dims = _choose_dimensions(steps)
    if dims is None:
        return None
    height, width, wall_rows_asc = dims
    core_plan = _build_core_plan(height, width, wall_rows_asc, steps)
    if core_plan is None:
        return None

    bg = random.choice(ALL_COLORS)
    separator_color, wall_color = random.sample([c for c in ALL_COLORS if c != bg], 2)
    palette = _choose_palette(wall_count, bg, wall_color, separator_color)

    plan = dict(core_plan)
    plan.update(
        {
            "bg": bg,
            "separator_color": separator_color,
            "wall_color": wall_color,
            "palette": palette,
        }
    )

    grid, all_segments_by_row = _render_input(plan, bg, separator_color, wall_color, palette)
    plan["all_segments_by_row"] = all_segments_by_row

    expected = _render_from_plan(grid, plan)
    if not _validate_instance(grid, plan, expected):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
