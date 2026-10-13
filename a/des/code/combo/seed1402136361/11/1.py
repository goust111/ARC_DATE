import random
from typing import NamedTuple

import numpy as np


class FrameSpec(NamedTuple):
    r0: int
    c0: int
    r1: int
    c1: int
    frame_color: int
    n_rows: int
    n_cols: int
    band_h: int
    band_w: int


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=10)
    return int(np.argmax(counts))


def _constant_step(positions: list[int]) -> int | None:
    if len(positions) <= 1:
        return None
    diffs = [b - a for a, b in zip(positions, positions[1:])]
    if len(set(diffs)) != 1:
        return None
    return diffs[0]


def _find_frame(grid: np.ndarray, bg: int) -> FrameSpec:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[FrameSpec] = []

    for r0 in range(h - 3):
        for r1 in range(r0 + 3, h):
            inner_h = r1 - r0 - 1
            for c0 in range(w - 3):
                for c1 in range(c0 + 3, w):
                    inner_w = c1 - c0 - 1
                    frame_color = int(g[r1, c0])
                    if frame_color == bg:
                        continue
                    if int(g[r0, c0]) != frame_color or int(g[r0, c1]) != frame_color:
                        continue
                    if not np.all(g[r1, c0 : c1 + 1] == frame_color):
                        continue
                    if not np.all(g[r0 : r1 + 1, c1] == frame_color):
                        continue

                    top_edge = g[r0, c0 : c1 + 1]
                    left_edge = g[r0 : r1 + 1, c0]
                    if np.any(top_edge == bg) or np.any(left_edge == bg):
                        continue

                    top_marks = [c for c in range(c0 + 1, c1) if int(g[r0, c]) != frame_color]
                    left_marks = [r for r in range(r0 + 1, r1) if int(g[r, c0]) != frame_color]
                    if not top_marks or not left_marks:
                        continue
                    if len(top_marks) >= inner_w or len(left_marks) >= inner_h:
                        continue

                    n_cols = len(top_marks)
                    n_rows = len(left_marks)
                    if inner_w % n_cols != 0 or inner_h % n_rows != 0:
                        continue

                    band_w = inner_w // n_cols
                    band_h = inner_h // n_rows
                    if band_w <= 0 or band_h <= 0:
                        continue

                    col_step = _constant_step(top_marks)
                    row_step = _constant_step(left_marks)
                    if n_cols > 1 and col_step != band_w:
                        continue
                    if n_rows > 1 and row_step != band_h:
                        continue

                    col_bands = {(c - (c0 + 1)) // band_w for c in top_marks}
                    row_bands = {(r - (r0 + 1)) // band_h for r in left_marks}
                    if col_bands != set(range(n_cols)) or row_bands != set(range(n_rows)):
                        continue

                    candidates.append(
                        FrameSpec(
                            r0=r0,
                            c0=c0,
                            r1=r1,
                            c1=c1,
                            frame_color=frame_color,
                            n_rows=n_rows,
                            n_cols=n_cols,
                            band_h=band_h,
                            band_w=band_w,
                        )
                    )

    if not candidates:
        raise ValueError("failed to find indexed frame")

    return max(
        candidates,
        key=lambda spec: (
            (spec.r1 - spec.r0 + 1) * (spec.c1 - spec.c0 + 1),
            spec.n_rows * spec.n_cols,
        ),
    )


def _majority_color(block: np.ndarray, bg: int) -> int:
    values, counts = np.unique(np.asarray(block, dtype=int), return_counts=True)
    if values.size == 1 and int(values[0]) == bg:
        return bg
    max_count = int(counts.max())
    winners = values[counts == max_count]
    if winners.size == 1:
        return int(winners[0])
    if bg in set(map(int, winners)):
        return bg
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read row/column markers from a rectangular frame and compress the interior into a table.

    The top and left frame edges contain equally spaced marker pixels. Their counts
    determine how many column and row bands the interior is split into. Each output
    cell is the majority color of the corresponding interior band intersection.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame = _find_frame(g, bg)

    interior = g[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1]
    out = np.full((frame.n_rows, frame.n_cols), bg, dtype=int)

    for out_r in range(frame.n_rows):
        r_start = out_r * frame.band_h
        r_stop = r_start + frame.band_h
        for out_c in range(frame.n_cols):
            c_start = out_c * frame.band_w
            c_stop = c_start + frame.band_w
            block = interior[r_start:r_stop, c_start:c_stop]
            out[out_r, out_c] = _majority_color(block, bg)

    return out


# --------------- generator ---------------

_GEN_RETRIES = 400
_ROWS_RANGE = (2, 5)
_COLS_RANGE = (2, 5)
_BAND_H_RANGE = (2, 6)
_BAND_W_RANGE = (2, 6)
_MARGIN_RANGE = (1, 4)
_CONTENT_COLOR_COUNT_RANGE = (2, 6)


def _sample_color(rng: random.Random, excluded: set[int]) -> int:
    choices = [color for color in range(10) if color not in excluded]
    return rng.choice(choices)


def _random_marker_sequence(rng: random.Random, length: int, bg: int, frame_color: int) -> list[int]:
    pool = [color for color in range(10) if color not in {bg, frame_color}]
    palette_size = rng.randint(2, min(4, len(pool)))
    palette = rng.sample(pool, palette_size)
    return [rng.choice(palette) for _ in range(length)]


def _noise_budget(area: int) -> int:
    if area <= 4:
        return 1
    return max(1, (area - 1) // 3)


def _fill_cell(
    block: np.ndarray,
    output_color: int,
    bg: int,
    content_palette: list[int],
    rng: random.Random,
    style: str,
) -> None:
    if output_color == bg:
        block[:] = bg
        return

    if style == "solid":
        block[:] = output_color
        return

    block[:] = output_color
    h, w = block.shape
    area = h * w
    coords = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(coords)
    noise_count = rng.randint(1, _noise_budget(area))

    for r, c in coords[:noise_count]:
        alternatives = [bg] + [color for color in content_palette if color != output_color]
        block[r, c] = rng.choice(alternatives)


def _choose_output_table(
    rng: random.Random,
    n_rows: int,
    n_cols: int,
    bg: int,
    content_palette: list[int],
) -> tuple[np.ndarray, list[list[str]]]:
    table = np.full((n_rows, n_cols), bg, dtype=int)
    styles = [["empty" for _ in range(n_cols)] for _ in range(n_rows)]

    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    rng.shuffle(cells)

    noisy_target = rng.randint(1, max(1, len(cells) // 2))
    noisy_cells = set(cells[:noisy_target])

    for r in range(n_rows):
        for c in range(n_cols):
            if rng.random() < 0.35:
                continue
            table[r, c] = rng.choice(content_palette)
            styles[r][c] = "noisy" if (r, c) in noisy_cells else "solid"

    if np.all(table == bg):
        r, c = rng.choice(cells)
        table[r, c] = rng.choice(content_palette)
        styles[r][c] = "solid"
    if not np.any(table == bg):
        r, c = rng.choice(cells)
        table[r, c] = bg
        styles[r][c] = "empty"
    if all(style != "noisy" for row in styles for style in row):
        non_bg_cells = [(r, c) for r in range(n_rows) for c in range(n_cols) if int(table[r, c]) != bg]
        if non_bg_cells:
            r, c = rng.choice(non_bg_cells)
            styles[r][c] = "noisy"

    return table, styles


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    frame_color = _sample_color(rng, {bg})

    n_rows = rng.randint(*_ROWS_RANGE)
    n_cols = rng.randint(*_COLS_RANGE)
    band_h = rng.randint(*_BAND_H_RANGE)
    band_w = rng.randint(*_BAND_W_RANGE)

    inner_h = n_rows * band_h
    inner_w = n_cols * band_w
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    margin_top = rng.randint(*_MARGIN_RANGE)
    margin_bottom = rng.randint(*_MARGIN_RANGE)
    margin_left = rng.randint(*_MARGIN_RANGE)
    margin_right = rng.randint(*_MARGIN_RANGE)

    grid_h = frame_h + margin_top + margin_bottom
    grid_w = frame_w + margin_left + margin_right
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    r0 = margin_top
    c0 = margin_left
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    top_offset = rng.randrange(band_w)
    left_offset = rng.randrange(band_h)
    top_markers = _random_marker_sequence(rng, n_cols, bg, frame_color)
    left_markers = _random_marker_sequence(rng, n_rows, bg, frame_color)
    for idx, color in enumerate(top_markers):
        grid[r0, c0 + 1 + top_offset + idx * band_w] = color
    for idx, color in enumerate(left_markers):
        grid[r0 + 1 + left_offset + idx * band_h, c0] = color

    content_palette_size = rng.randint(*_CONTENT_COLOR_COUNT_RANGE)
    content_palette = rng.sample([color for color in range(10) if color not in {bg, frame_color}], content_palette_size)
    output_table, styles = _choose_output_table(rng, n_rows, n_cols, bg, content_palette)

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    for rr in range(n_rows):
        rs = rr * band_h
        re = rs + band_h
        for cc in range(n_cols):
            cs = cc * band_w
            ce = cs + band_w
            _fill_cell(
                interior[rs:re, cs:ce],
                int(output_table[rr, cc]),
                bg,
                content_palette,
                rng,
                styles[rr][cc],
            )

    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != bg or list(counts).count(counts[bg]) != 1:
        return None

    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, output_table):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid indexed-frame sample")
