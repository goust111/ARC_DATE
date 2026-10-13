import random
from typing import Sequence

import numpy as np


SEPARATOR_COLOR = 5
MAX_GENERATION_ATTEMPTS = 400
PANEL_HEIGHT_RANGE = (9, 14)
PANEL_WIDTH_RANGE = (9, 20)
TOTAL_LINE_RANGE = (4, 6)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = np.argwhere(mask)
    assert len(points) > 0, "expected at least one foreground pixel"
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, _w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != bg:
            if r > 0 and h - r - 1 == r:
                candidates.append((r, int(row[0])))
    assert len(candidates) == 1, "expected one centered full-width separator row"
    return candidates[0]


def _dominant_non_bg(values: Sequence[int], bg: int) -> int:
    arr = np.asarray(values, dtype=int)
    arr = arr[arr != bg]
    uniq, counts = np.unique(arr, return_counts=True)
    assert len(uniq) > 0, "line sample must contain non-background colors"
    best = int(counts.max())
    assert int(np.count_nonzero(counts == best)) == 1, "line color must be uniquely recoverable"
    return int(uniq[counts.argmax()])


def _recover_weave(top: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(top != bg)
    sub = top[r0 : r1 + 1, c0 : c1 + 1]

    horizontal_rows = [r0 + i for i in range(sub.shape[0]) if np.all(sub[i] != bg)]
    vertical_cols = [c0 + j for j in range(sub.shape[1]) if np.all(sub[:, j] != bg)]
    assert horizontal_rows and vertical_cols, "expected both horizontal and vertical lines"

    horizontals = [(r, _dominant_non_bg(top[r, c0 : c1 + 1], bg)) for r in horizontal_rows]
    verticals = [(c, _dominant_non_bg(top[r0 : r1 + 1, c], bg)) for c in vertical_cols]
    return (r0, c0, r1, c1), horizontals, verticals


def _weave_canvas(
    top: np.ndarray,
    bg: int,
    box: tuple[int, int, int, int],
    horizontals: Sequence[tuple[int, int]],
    verticals: Sequence[tuple[int, int]],
) -> np.ndarray:
    canvas = np.full_like(top, bg)
    r0, c0, r1, c1 = box

    for row, color in horizontals:
        canvas[row, c0 : c1 + 1] = int(color)
    for col, color in verticals:
        canvas[r0 : r1 + 1, col] = int(color)

    for row, h_color in horizontals:
        for col, v_color in verticals:
            shown = int(top[row, col])
            if shown == int(h_color):
                canvas[row, col] = int(v_color)
            else:
                assert shown == int(v_color), "intersection must show one of its two crossing line colors"
                canvas[row, col] = int(h_color)
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reweave the hidden crossing colors from the blueprint into the blank lower canvas.

    Input structure:
    - bg is the most common color.
    - A single full-width separator row splits the grid into an upper blueprint and an equally sized lower canvas.
    - The blueprint contains one woven set of one-pixel horizontal and vertical colored line segments; the lower canvas starts blank.

    Rule:
    - Detect the separator and keep the blueprint and separator unchanged.
    - Recover each horizontal line row and each vertical line column from the blueprint, then project those same line segments into the lower canvas.
    - At every crossing, the blueprint shows only the top line's color, so the lower canvas uses the other crossing line's color instead; non-crossing line pixels keep their own line color.

    Key insight:
    - The visible intersection color in the blueprint is not the desired answer there; it reveals which line is on top, so the output must place the hidden line color at that same crossing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    separator_row, _separator_color = _find_separator_row(g, bg)
    top = g[:separator_row]
    bottom = g[separator_row + 1 :]
    assert top.shape == bottom.shape, "blueprint and canvas must have the same shape"

    box, horizontals, verticals = _recover_weave(top, bg)
    canvas = _weave_canvas(top, bg, box, horizontals, verticals)

    out = g.copy()
    out[separator_row + 1 :] = canvas
    return out


def _sample_spaced_positions(length: int, count: int) -> list[int] | None:
    margin = 1
    min_gap = 3 if count == 2 else 2
    usable = length - 2 * margin
    required = count + (count - 1) * (min_gap - 1)
    if usable < required:
        return None

    compressed = usable - (count - 1) * (min_gap - 1)
    picks = sorted(random.sample(range(compressed), count))
    return [margin + pick + i * (min_gap - 1) for i, pick in enumerate(picks)]


def _choose_line_counts() -> tuple[int, int]:
    total = random.randint(*TOTAL_LINE_RANGE)
    horizontal_count = random.randint(2, total - 2)
    vertical_count = total - horizontal_count
    return horizontal_count, vertical_count


def _build_expected(
    grid: np.ndarray,
    bg: int,
    panel_h: int,
    horizontals: Sequence[tuple[int, int]],
    verticals: Sequence[tuple[int, int]],
) -> np.ndarray:
    top = grid[:panel_h]
    rows = [row for row, _color in horizontals]
    cols = [col for col, _color in verticals]
    box = (min(rows), min(cols), max(rows), max(cols))

    out = grid.copy()
    out[panel_h + 1 :] = _weave_canvas(top, bg, box, horizontals, verticals)
    return out


def _gen_attempt() -> np.ndarray | None:
    horizontal_count, vertical_count = _choose_line_counts()

    min_h = max(PANEL_HEIGHT_RANGE[0], 2 * horizontal_count + 1)
    min_w = max(PANEL_WIDTH_RANGE[0], 2 * vertical_count + 1)
    if min_h > PANEL_HEIGHT_RANGE[1] or min_w > PANEL_WIDTH_RANGE[1]:
        return None

    panel_h = random.randint(min_h, PANEL_HEIGHT_RANGE[1])
    panel_w = random.randint(min_w, PANEL_WIDTH_RANGE[1])
    horizontal_rows = _sample_spaced_positions(panel_h, horizontal_count)
    vertical_cols = _sample_spaced_positions(panel_w, vertical_count)
    if horizontal_rows is None or vertical_cols is None:
        return None

    bg_choices = [color for color in range(10) if color != SEPARATOR_COLOR]
    bg = random.choice(bg_choices)
    available = [color for color in range(10) if color not in {bg, SEPARATOR_COLOR}]
    total_lines = horizontal_count + vertical_count
    if len(available) < total_lines:
        return None

    line_colors = random.sample(available, total_lines)
    horizontal_colors = line_colors[:horizontal_count]
    vertical_colors = line_colors[horizontal_count:]

    lines: list[tuple[str, int, int]] = []
    lines.extend(("h", row, color) for row, color in zip(horizontal_rows, horizontal_colors))
    lines.extend(("v", col, color) for col, color in zip(vertical_cols, vertical_colors))
    random.shuffle(lines)

    h_ranks = [idx for idx, (kind, _pos, _color) in enumerate(lines) if kind == "h"]
    v_ranks = [idx for idx, (kind, _pos, _color) in enumerate(lines) if kind == "v"]
    if max(h_ranks) < min(v_ranks) or max(v_ranks) < min(h_ranks):
        return None

    height = 2 * panel_h + 1
    grid = np.full((height, panel_w), bg, dtype=int)
    grid[panel_h, :] = SEPARATOR_COLOR

    r0, r1 = min(horizontal_rows), max(horizontal_rows)
    c0, c1 = min(vertical_cols), max(vertical_cols)
    top = grid[:panel_h]
    for kind, pos, color in lines:
        if kind == "h":
            top[pos, c0 : c1 + 1] = int(color)
        else:
            top[r0 : r1 + 1, pos] = int(color)

    horizontals = list(zip(horizontal_rows, horizontal_colors))
    verticals = list(zip(vertical_cols, vertical_colors))
    expected = _build_expected(grid, bg, panel_h, horizontals, verticals)

    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-line puzzle")
