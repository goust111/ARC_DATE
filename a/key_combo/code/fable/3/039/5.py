from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
DIR_BY_COLOR = {
    1: (-1, 0),  # blue -> up
    2: (0, 1),   # red -> right
    3: (1, 0),   # green -> down
    4: (0, -1),  # yellow -> left
}
DIRECTION_COLORS = tuple(DIR_BY_COLOR)
DIVIDER_COLOR = 5
PATH_COLORS = (6, 7, 8, 9)

MIN_SEGMENTS = 4
MAX_SEGMENTS = 6
MIN_STEP = 1
MAX_STEP = 3
MIN_CANVAS_H = 7
MIN_CANVAS_W = 10
MAX_ATTEMPTS = 400


def _majority_color(values: np.ndarray) -> tuple[int, int]:
    counts = Counter(int(v) for v in values)
    color, count = counts.most_common(1)[0]
    if len(counts) > 1 and counts.most_common(2)[1][1] == count:
        raise AssertionError("row majority must be unique")
    return int(color), int(count)


def _analyze_canvas(canvas: np.ndarray) -> tuple[int, tuple[int, int], int] | None:
    counts = Counter(int(v) for v in canvas.ravel())
    bg, bg_count = counts.most_common(1)[0]
    if len(counts) > 1 and counts.most_common(2)[1][1] == bg_count:
        return None
    start_coords = np.argwhere(canvas != bg)
    if start_coords.shape[0] != 1:
        return None
    start = tuple(map(int, start_coords[0]))
    start_color = int(canvas[start])
    if start_color == bg:
        return None
    return int(bg), start, start_color


def _extract_bands(program: np.ndarray, divider_color: int) -> list[tuple[int, int]] | None:
    if program.ndim != 2 or program.shape[0] == 0:
        return None

    row_colors: list[int] = []
    width = int(program.shape[1])
    for row in program:
        try:
            color, count = _majority_color(row)
        except AssertionError:
            return None
        if color == divider_color or color not in DIR_BY_COLOR:
            return None
        if count * 2 <= width:
            return None
        row_colors.append(color)

    bands: list[tuple[int, int]] = []
    current = row_colors[0]
    thickness = 1
    for color in row_colors[1:]:
        if color == current:
            thickness += 1
        else:
            bands.append((current, thickness))
            current = color
            thickness = 1
    bands.append((current, thickness))
    return bands


def _find_layout(grid: np.ndarray) -> tuple[int, bool, list[tuple[int, int]], int, tuple[int, int], int]:
    h, _w = grid.shape
    candidates = []

    for divider_r in range(1, h - 1):
        if not np.all(grid[divider_r] == grid[divider_r, 0]):
            continue

        divider_color = int(grid[divider_r, 0])
        for program_on_top in (True, False):
            if program_on_top:
                program = grid[:divider_r, :]
                canvas = grid[divider_r + 1 :, :]
            else:
                program = grid[divider_r + 1 :, :]
                canvas = grid[:divider_r, :]

            canvas_info = _analyze_canvas(canvas)
            bands = _extract_bands(program, divider_color)
            if canvas_info is None or bands is None:
                continue

            bg, start_local, start_color = canvas_info
            score = (len(bands), program.shape[0], canvas.shape[0])
            candidates.append((score, divider_r, program_on_top, bands, bg, start_local, start_color))

    assert candidates, "failed to locate divider/program/canvas layout"
    candidates.sort(reverse=True)
    _score, divider_r, program_on_top, bands, bg, start_local, start_color = candidates[0]

    if program_on_top:
        start = (divider_r + 1 + start_local[0], start_local[1])
    else:
        start = start_local
    return int(divider_r), bool(program_on_top), bands, int(bg), start, int(start_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress noisy horizontal color bands into movement instructions and draw the resulting path on the other side of the divider.

    Input structure:
    - A single full-width divider row splits the grid into a program region and a canvas region.
    - The program region is tiled by horizontal color bands; each band spans several rows and each row is mostly one direction color with sparse off-color noise.
    - The canvas region is otherwise empty background except for one colored start pixel.

    Rule:
    - Read the program region from top to bottom by taking each row's majority color and compressing consecutive equal majorities into one band.
    - Each band's majority color gives a fixed direction, and that band's thickness gives the number of steps to move.
    - Starting from the lone start pixel, walk the program step by step and paint every visited canvas cell with the start pixel's color, while leaving the divider and program region unchanged.

    Key insight:
    - The visible bands are not the answer image; they are a noisy encoding of a motion program where majority color means direction and band thickness means distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_r, program_on_top, bands, bg, start, start_color = _find_layout(g)
    h, w = g.shape
    out = g.copy()

    if program_on_top:
        canvas_r0, canvas_r1 = divider_r + 1, h
    else:
        canvas_r0, canvas_r1 = 0, divider_r

    r, c = map(int, start)
    assert canvas_r0 <= r < canvas_r1 and 0 <= c < w
    assert int(g[r, c]) == start_color

    for color, steps in bands:
        dr, dc = DIR_BY_COLOR[color]
        for _ in range(int(steps)):
            r += dr
            c += dc
            assert canvas_r0 <= r < canvas_r1, "path left the canvas region"
            assert 0 <= c < w, "path left the grid horizontally"
            assert int(g[r, c]) in (bg, start_color), "canvas should contain only background and the start pixel"
            out[r, c] = start_color

    return out


def _sample_sequence() -> tuple[list[int], list[int]]:
    seq = list(DIRECTION_COLORS)
    random.shuffle(seq)
    target_len = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    while len(seq) < target_len:
        candidate = random.choice(DIRECTION_COLORS)
        if candidate != seq[-1]:
            seq.append(candidate)
    lengths = [random.randint(MIN_STEP, MAX_STEP) for _ in seq]
    return seq, lengths


def _simulate_relative_path(seq: list[int], lengths: list[int]) -> list[tuple[int, int]] | None:
    path = [(0, 0)]
    seen = {(0, 0)}
    r = c = 0

    for color, steps in zip(seq, lengths):
        dr, dc = DIR_BY_COLOR[color]
        for _ in range(int(steps)):
            r += dr
            c += dc
            if (r, c) in seen:
                return None
            seen.add((r, c))
            path.append((r, c))
    return path


def _paint_noisy_band_row(width: int, color: int, force_noise: bool) -> np.ndarray:
    row = np.full(width, int(color), dtype=int)
    max_noise = min(3, max(1, width // 5))
    noise_count = random.randint(1 if force_noise else 0, max_noise)
    if noise_count:
        cols = random.sample(range(width), noise_count)
        noise_palette = [other for other in DIRECTION_COLORS if other != color]
        for col in cols:
            row[col] = random.choice(noise_palette)
    return row


def _gen_attempt() -> np.ndarray | None:
    seq, lengths = _sample_sequence()
    path = _simulate_relative_path(seq, lengths)
    if path is None:
        return None

    rows = [p[0] for p in path]
    cols = [p[1] for p in path]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    pad_top = random.randint(1, 3)
    pad_bottom = random.randint(1, 4)
    pad_left = random.randint(1, 3)
    pad_right = random.randint(1, 4)

    canvas_h = (max_r - min_r + 1) + pad_top + pad_bottom
    canvas_w = (max_c - min_c + 1) + pad_left + pad_right
    if canvas_h < MIN_CANVAS_H:
        canvas_h += MIN_CANVAS_H - canvas_h
    if canvas_w < MIN_CANVAS_W:
        canvas_w += MIN_CANVAS_W - canvas_w

    program_h = sum(lengths)
    total_h = program_h + 1 + canvas_h
    if total_h > 30 or canvas_w > 30:
        return None

    start_r = -min_r + pad_top
    start_c = -min_c + pad_left
    if not (0 <= start_r < canvas_h and 0 <= start_c < canvas_w):
        return None

    translated = [(r + start_r, c + start_c) for r, c in path]
    if min(r for r, _ in translated) < 0 or max(r for r, _ in translated) >= canvas_h:
        return None
    if min(c for _, c in translated) < 0 or max(c for _, c in translated) >= canvas_w:
        return None

    grid = np.full((total_h, canvas_w), BG, dtype=int)

    forced_noise_row = random.randrange(program_h)
    row_index = 0
    for color, steps in zip(seq, lengths):
        for _ in range(int(steps)):
            grid[row_index, :] = _paint_noisy_band_row(canvas_w, color, row_index == forced_noise_row)
            row_index += 1

    grid[program_h, :] = DIVIDER_COLOR

    start_color = random.choice(PATH_COLORS)
    grid[program_h + 1 + start_r, start_c] = start_color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
