from __future__ import annotations

import random

import numpy as np


BG = 8
DIVIDER = 5
MIX = 9

LEFT = 1
DOWN = 2
RIGHT = 3
UP = 4

DIR_TO_DELTA = {
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
    UP: (-1, 0),
}

BRUSH_COLORS = [0, 6, 7]
GEN_RETRIES = 300


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER)]
    assert len(rows) == 1, "expected one full-width grey divider row"
    return int(rows[0])


def _programs_from_instruction_zone(zone: np.ndarray) -> list[tuple[int, list[int]]]:
    row_has_signal = np.any(zone != BG, axis=1)
    programs: list[tuple[int, list[int]]] = []

    r = 0
    while r < zone.shape[0]:
        if not bool(row_has_signal[r]):
            r += 1
            continue

        r0 = r
        while r + 1 < zone.shape[0] and bool(row_has_signal[r + 1]):
            r += 1
        block = zone[r0 : r + 1]

        cols = np.flatnonzero(np.any(block != BG, axis=0))
        assert cols.size > 0, "each instruction band must contain visible pixels"
        c0, c1 = int(cols[0]), int(cols[-1])

        seq: list[int] = []
        for c in range(c0, c1 + 1):
            vals = [int(block[rr, c]) for rr in range(block.shape[0]) if int(block[rr, c]) != BG]
            assert vals, "instruction columns must be vertically filled inside a band"
            seq.append(vals[0])

        assert len(seq) >= 2, "each band needs a brush color and at least one move"
        brush = int(seq[0])
        steps = [int(v) for v in seq[1:]]
        for step in steps:
            assert step in DIR_TO_DELTA, "unknown direction code"
        programs.append((brush, steps))
        r += 1

    assert programs, "expected at least one instruction band"
    return programs


def _find_starts(canvas: np.ndarray, brush_colors: list[int]) -> dict[int, tuple[int, int]]:
    starts: dict[int, tuple[int, int]] = {}
    for color in sorted(set(int(c) for c in brush_colors)):
        pts = np.argwhere(canvas == int(color))
        assert pts.shape[0] == 1, "each brush color must have exactly one start pixel"
        starts[int(color)] = tuple(map(int, pts[0]))
    return starts


def _trace_paths(
    shape: tuple[int, int],
    programs: list[tuple[int, list[int]]],
    starts: dict[int, tuple[int, int]],
) -> tuple[list[list[set[int]]], int]:
    h, w = shape
    owners = [[set() for _ in range(w)] for _ in range(h)]
    invalid_steps = 0

    for brush, steps in programs:
        r, c = starts[int(brush)]
        owners[r][c].add(int(brush))
        for step in steps:
            dr, dc = DIR_TO_DELTA[int(step)]
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                r, c = nr, nc
                owners[r][c].add(int(brush))
            else:
                invalid_steps += 1

    return owners, invalid_steps


def _owners_to_output(owners: list[list[set[int]]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    for r in range(h):
        for c in range(w):
            if not owners[r][c]:
                continue
            out[r, c] = MIX if len(owners[r][c]) > 1 else next(iter(owners[r][c]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode movement instructions from the upper color bands and draw the resulting
    brush paths on the lower canvas. Cells covered by two or more different brush
    colors are recolored maroon.
    """
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    programs = _programs_from_instruction_zone(g[:sep, :])
    canvas = g[sep + 1 :, :]
    starts = _find_starts(canvas, [brush for brush, _steps in programs])
    owners, _ = _trace_paths(canvas.shape, programs, starts)
    return _owners_to_output(owners, canvas.shape)


def _row_bands(height: int, count: int) -> list[tuple[int, int]] | None:
    band_heights = [random.randint(2, 3) for _ in range(count)]
    required = sum(band_heights) + (count - 1)
    if required > height:
        return None

    extra = height - required
    slots = [0] * (count + 1)
    for _ in range(extra):
        slots[random.randint(0, count)] += 1

    bands: list[tuple[int, int]] = []
    row = slots[0]
    for idx, band_h in enumerate(band_heights):
        bands.append((row, row + band_h - 1))
        row += band_h
        if idx < count - 1:
            row += 1 + slots[idx + 1]
    return bands


def _build_band_program(
    band: tuple[int, int],
    width: int,
    col_lo: int,
    col_hi: int,
) -> tuple[tuple[int, int], list[int]] | None:
    max_steps = width - 1
    r0, r1 = band
    if r0 >= r1 or col_lo >= col_hi:
        return None

    for _ in range(100):
        sr = random.randint(r0, r1)
        sc = random.randint(col_lo, col_hi)

        horiz1 = []
        if sc > col_lo:
            horiz1.append(LEFT)
        if sc < col_hi:
            horiz1.append(RIGHT)
        if not horiz1:
            continue
        d1 = random.choice(horiz1)
        room1 = sc - col_lo if d1 == LEFT else col_hi - sc
        n1 = random.randint(1, min(3, room1))
        c1 = sc + DIR_TO_DELTA[d1][1] * n1

        vert = []
        if sr > r0:
            vert.append(UP)
        if sr < r1:
            vert.append(DOWN)
        if not vert:
            continue
        d2 = random.choice(vert)
        room2 = sr - r0 if d2 == UP else r1 - sr
        n2 = random.randint(1, room2)
        r2 = sr + DIR_TO_DELTA[d2][0] * n2

        horiz2 = []
        if c1 > col_lo:
            horiz2.append(LEFT)
        if c1 < col_hi:
            horiz2.append(RIGHT)
        if not horiz2:
            continue
        d3 = random.choice(horiz2)
        room3 = c1 - col_lo if d3 == LEFT else col_hi - c1
        n3 = random.randint(1, min(3, room3))

        steps = [d1] * n1 + [d2] * n2 + [d3] * n3
        if 1 <= len(steps) <= max_steps:
            return (sr, sc), steps

    return None


def _build_oob_row_program(band: tuple[int, int], width: int) -> tuple[tuple[int, int], list[int]]:
    r0, r1 = band
    row = random.randint(r0, r1)
    start_c = random.randint(1, width - 2)
    direction = random.choice([LEFT, RIGHT])
    if direction == LEFT:
        steps = [LEFT] * (start_c + 1)
    else:
        steps = [RIGHT] * (width - start_c)
    return (row, start_c), steps


def _band_away_from_row(height: int, blocked_row: int) -> tuple[int, int] | None:
    choices = []
    for size in (2, 3):
        for r0 in range(0, height - size + 1):
            r1 = r0 + size - 1
            if r0 <= blocked_row <= r1:
                continue
            choices.append((r0, r1))
    if not choices:
        return None
    return random.choice(choices)


def _build_cross_scene(
    height: int,
    width: int,
    colors: list[int],
    need_oob: bool,
) -> tuple[list[tuple[int, list[int]]], dict[int, tuple[int, int]]] | None:
    max_steps = width - 1
    for _ in range(100):
        cross_r = random.randint(1, height - 2)
        cross_c = random.randint(2, width - 3)

        horizontal_from_left = random.choice([True, False])
        if horizontal_from_left:
            lo = 1 if need_oob else 0
            hi = cross_c - 1
            if lo > hi:
                continue
            start_a = (cross_r, random.randint(lo, hi))
            to_cross_a = RIGHT
            away_a = RIGHT
            dist_a = cross_c - start_a[1]
            boundary_a = width - 1 - cross_c
        else:
            lo = cross_c + 1
            hi = width - 2 if need_oob else width - 1
            if lo > hi:
                continue
            start_a = (cross_r, random.randint(lo, hi))
            to_cross_a = LEFT
            away_a = LEFT
            dist_a = start_a[1] - cross_c
            boundary_a = cross_c

        if need_oob:
            if boundary_a == 0 or dist_a + boundary_a + 1 > max_steps:
                continue
            tail_a = boundary_a + 1
        else:
            tail_a = random.randint(0, min(2, boundary_a, max_steps - dist_a))
        steps_a = [to_cross_a] * dist_a + [away_a] * tail_a

        vertical_from_top = random.choice([True, False])
        if vertical_from_top:
            start_b = (random.randint(0, cross_r - 1), cross_c)
            to_cross_b = DOWN
            away_b = DOWN
            dist_b = cross_r - start_b[0]
            boundary_b = height - 1 - cross_r
        else:
            start_b = (random.randint(cross_r + 1, height - 1), cross_c)
            to_cross_b = UP
            away_b = UP
            dist_b = start_b[0] - cross_r
            boundary_b = cross_r
        if dist_b > max_steps:
            continue
        tail_b = random.randint(0, min(2, boundary_b, max_steps - dist_b))
        steps_b = [to_cross_b] * dist_b + [away_b] * tail_b

        programs = [(colors[0], steps_a), (colors[1], steps_b)]
        starts = {
            int(colors[0]): tuple(map(int, start_a)),
            int(colors[1]): tuple(map(int, start_b)),
        }

        if len(colors) == 3:
            band = _band_away_from_row(height, cross_r)
            if band is None:
                continue
            col_ranges = []
            if cross_c - 1 >= 0:
                col_ranges.append((0, cross_c - 1))
            if cross_c + 1 <= width - 1:
                col_ranges.append((cross_c + 1, width - 1))
            random.shuffle(col_ranges)
            extra = None
            for col_lo, col_hi in col_ranges:
                extra = _build_band_program(band, width, col_lo, col_hi)
                if extra is not None:
                    break
            if extra is None:
                continue
            start_c, steps_c = extra
            starts[int(colors[2])] = tuple(map(int, start_c))
            programs.append((colors[2], steps_c))

        return programs, starts

    return None


def _build_no_cross_scene(
    height: int,
    width: int,
    colors: list[int],
    need_oob: bool,
) -> tuple[list[tuple[int, list[int]]], dict[int, tuple[int, int]]] | None:
    bands = _row_bands(height, len(colors))
    if bands is None:
        return None

    order = colors[:]
    random.shuffle(order)

    programs: list[tuple[int, list[int]]] = []
    starts: dict[int, tuple[int, int]] = {}
    for idx, color in enumerate(order):
        band = bands[idx]
        if need_oob and idx == 0:
            start, steps = _build_oob_row_program(band, width)
        else:
            built = _build_band_program(band, width, 0, width - 1)
            if built is None:
                return None
            start, steps = built
        starts[int(color)] = tuple(map(int, start))
        programs.append((int(color), steps))

    return programs, starts


def _render_input(
    canvas_shape: tuple[int, int],
    programs: list[tuple[int, list[int]]],
    starts: dict[int, tuple[int, int]],
) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    band_heights = [random.randint(1, 2) for _ in programs]
    top_pad = random.randint(0, 1)
    bottom_pad = random.randint(0, 1)
    instruction_h = top_pad + bottom_pad + sum(band_heights) + max(0, len(programs) - 1)
    total_h = instruction_h + 1 + canvas_h

    grid = np.full((total_h, canvas_w), BG, dtype=int)
    grid[instruction_h, :] = DIVIDER

    order = list(range(len(programs)))
    random.shuffle(order)
    row = top_pad
    for pos, prog_idx in enumerate(order):
        brush, steps = programs[prog_idx]
        seq = [int(brush)] + [int(v) for v in steps]
        c0 = random.randint(0, canvas_w - len(seq))
        for rr in range(row, row + band_heights[prog_idx]):
            grid[rr, c0 : c0 + len(seq)] = seq
        row += band_heights[prog_idx]
        if pos != len(order) - 1:
            row += 1

    canvas_r0 = instruction_h + 1
    for color, (r, c) in starts.items():
        grid[canvas_r0 + r, c] = int(color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        brush_count = random.randint(2, 3)
        canvas_h = random.randint(8 if brush_count == 3 else 7, 12)
        canvas_w = random.randint(max(10, canvas_h), 16)
        colors = random.sample(BRUSH_COLORS, brush_count)
        need_cross = bool(random.getrandbits(1))
        need_oob = bool(random.getrandbits(1))

        if need_cross:
            built = _build_cross_scene(canvas_h, canvas_w, colors, need_oob)
        else:
            built = _build_no_cross_scene(canvas_h, canvas_w, colors, need_oob)
        if built is None:
            continue

        programs, starts = built
        owners, invalid_steps = _trace_paths((canvas_h, canvas_w), programs, starts)
        expected = _owners_to_output(owners, (canvas_h, canvas_w))

        intersections = [
            (r, c)
            for r in range(canvas_h)
            for c in range(canvas_w)
            if len(owners[r][c]) > 1
        ]
        if len(intersections) != (1 if need_cross else 0):
            continue
        if invalid_steps != (1 if need_oob else 0):
            continue
        if any(owners[r][c] != {color} for color, (r, c) in starts.items()):
            continue

        painted = int(np.count_nonzero(expected != BG))
        if painted < brush_count * 3 or painted > (canvas_h * canvas_w) // 2:
            continue

        all_dirs = {step for _brush, steps in programs for step in steps}
        if len(all_dirs) < 2:
            continue

        grid = _render_input((canvas_h, canvas_w), programs, starts)
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
