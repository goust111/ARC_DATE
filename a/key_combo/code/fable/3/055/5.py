from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
PALETTE = list(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _extract_runs(line: np.ndarray, bg: int, *, offset: int) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    start = None
    color = None
    for idx, value in enumerate(map(int, line)):
        if value == int(bg):
            if color is not None:
                runs.append({"color": int(color), "start": int(start + offset), "end": int(idx - 1 + offset)})
                start = None
                color = None
            continue
        if color is None:
            start = int(idx)
            color = int(value)
            continue
        if int(value) != int(color):
            runs.append({"color": int(color), "start": int(start + offset), "end": int(idx - 1 + offset)})
            start = int(idx)
            color = int(value)
    if color is not None:
        runs.append({"color": int(color), "start": int(start + offset), "end": int(len(line) - 1 + offset)})
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct hidden filled rectangles from matching top-row and left-column color runs.

    Input structure:
    - The interior is entirely background; only the top row and the left column contain colored runs.
    - A paired color appears once on the top row and once on the left column, and those two runs define one hidden rectangle.
    - Some colors may appear on only one side as decoys and therefore do not define any rectangle.

    Rule:
    - Read the top-row runs from left to right.
    - For each color that has exactly one matching left-column run, fill the rectangle whose row span comes from the left run and whose column span comes from the top run.
    - Later rectangles overwrite earlier ones, then remove the top row and left column from the result.

    Key insight:
    - The rectangles themselves are absent; each color is visible only as a horizontal and vertical projection, and the top-row reading order determines the painting depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    bg = _get_bg(g)
    top_runs = _extract_runs(g[0, 1:], bg, offset=1)
    left_runs = _extract_runs(g[1:, 0], bg, offset=1)

    top_by_color: dict[int, list[dict[str, int]]] = {}
    left_by_color: dict[int, list[dict[str, int]]] = {}
    for run in top_runs:
        top_by_color.setdefault(int(run["color"]), []).append(run)
    for run in left_runs:
        left_by_color.setdefault(int(run["color"]), []).append(run)

    canvas = np.full(g.shape, int(bg), dtype=int)
    for run in top_runs:
        color = int(run["color"])
        if len(top_by_color.get(color, [])) != 1 or len(left_by_color.get(color, [])) != 1:
            continue
        left = left_by_color[color][0]
        canvas[left["start"] : left["end"] + 1, run["start"] : run["end"] + 1] = color

    return canvas[1:, 1:]


def _distribute(total: int, slots: int) -> list[int]:
    parts = [0] * int(slots)
    for _ in range(int(total)):
        parts[random.randrange(slots)] += 1
    return parts


def _sample_axis_runs(axis_len: int, colors: list[int]) -> list[dict[str, int]] | None:
    count = len(colors)
    if count == 0:
        return []

    min_len = 2
    max_len = min(6, axis_len - (count - 1))
    if max_len < min_len:
        return None

    for _ in range(100):
        lengths: list[int] = []
        remaining = axis_len - (count - 1)
        ok = True
        for idx in range(count):
            remaining_after = count - idx - 1
            low = min_len
            high = min(max_len, remaining - remaining_after * min_len)
            if high < low:
                ok = False
                break
            length = random.randint(low, high)
            lengths.append(length)
            remaining -= length
        if not ok:
            continue

        extra = axis_len - sum(lengths) - (count - 1)
        if extra < 0:
            continue

        extras = _distribute(extra, count + 1)
        leading = extras[0]
        middles = [1 + extra_gap for extra_gap in extras[1:count]]
        trailing = extras[count]

        runs: list[dict[str, int]] = []
        pos = leading
        for color, length, gap_after in zip(colors, lengths, middles + [trailing]):
            start = pos
            end = start + length - 1
            runs.append({"color": int(color), "start": int(start), "end": int(end)})
            pos = end + 1 + gap_after
        return runs
    return None


def _render_input(
    out_h: int,
    out_w: int,
    bg: int,
    top_runs: list[dict[str, int]],
    left_runs: list[dict[str, int]],
) -> np.ndarray:
    grid = np.full((out_h + 1, out_w + 1), int(bg), dtype=int)
    for run in top_runs:
        grid[0, run["start"] + 1 : run["end"] + 2] = int(run["color"])
    for run in left_runs:
        grid[run["start"] + 1 : run["end"] + 2, 0] = int(run["color"])
    return grid


def _render_expected(
    out_h: int,
    out_w: int,
    bg: int,
    top_runs: list[dict[str, int]],
    left_runs: list[dict[str, int]],
) -> np.ndarray:
    expected = np.full((out_h, out_w), int(bg), dtype=int)
    left_by_color = {int(run["color"]): run for run in left_runs}
    for run in top_runs:
        color = int(run["color"])
        if color not in left_by_color:
            continue
        left = left_by_color[color]
        expected[left["start"] : left["end"] + 1, run["start"] : run["end"] + 1] = color
    return expected


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choices([2, 3, 4], weights=[2, 4, 3])[0]
    top_decoys = random.choices([0, 1, 2], weights=[2, 3, 2])[0]
    left_decoys = random.choices([0, 1, 2], weights=[2, 3, 2])[0]
    if top_decoys + left_decoys == 0 and random.random() < 0.7:
        if random.random() < 0.5:
            top_decoys = 1
        else:
            left_decoys = 1

    total_colors = pair_count + top_decoys + left_decoys
    out_h = random.randint(max(8, 3 * (pair_count + left_decoys) - 1), 24)
    out_w = random.randint(max(8, 3 * (pair_count + top_decoys) - 1), 24)
    if max(out_h + 1, out_w + 1) > 30:
        return None

    bg = random.choice(PALETTE)
    usable = [color for color in PALETTE if color != int(bg)]
    if len(usable) < total_colors:
        return None

    random.shuffle(usable)
    pair_colors = usable[:pair_count]
    top_only_colors = usable[pair_count : pair_count + top_decoys]
    left_only_colors = usable[pair_count + top_decoys : total_colors]

    top_colors = pair_colors + top_only_colors
    left_colors = pair_colors + left_only_colors
    random.shuffle(top_colors)
    random.shuffle(left_colors)

    top_runs = _sample_axis_runs(out_w, top_colors)
    left_runs = _sample_axis_runs(out_h, left_colors)
    if top_runs is None or left_runs is None:
        return None

    expected = _render_expected(out_h, out_w, bg, top_runs, left_runs)
    if not np.any(expected != int(bg)):
        return None

    grid = _render_input(out_h, out_w, bg, top_runs, left_runs)
    if not np.all(grid[1:, 1:] == int(bg)):
        return None

    paired = {int(run["color"]) for run in top_runs} & {int(run["color"]) for run in left_runs}
    if len(paired) != pair_count:
        return None
    if len({int(run["color"]) for run in top_runs}) != len(top_runs):
        return None
    if len({int(run["color"]) for run in left_runs}) != len(left_runs):
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
    raise RuntimeError("failed to generate valid input")
