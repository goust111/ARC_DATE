from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 5
FRAME = 1
RED = 2
YELLOW = 4
FREE_COLORS = (0, 3, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300
MAX_PATH_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for color in np.unique(grid):
        if int(color) == int(bg):
            continue
        for comp in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue
            border_size = 2 * (height + width) - 4
            if len(comp) != border_size:
                continue
            good = True
            for c in range(c0, c1 + 1):
                if int(grid[r0, c]) != int(color) or int(grid[r1, c]) != int(color):
                    good = False
                    break
            if not good:
                continue
            for r in range(r0, r1 + 1):
                if int(grid[r, c0]) != int(color) or int(grid[r, c1]) != int(color):
                    good = False
                    break
            if good:
                frames.append((r0, c0, r1, c1))
    frames.sort(key=lambda box: (int(box[1]), int(box[0])))
    assert frames
    return frames


def _find_legend(grid: np.ndarray, bg: int, frames: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    outside = np.ones_like(grid, dtype=bool)
    for r0, c0, r1, c1 in frames:
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    candidates: list[tuple[int, int]] = []
    for comp in _components((grid != int(bg)) & outside):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 != r1:
            continue
        if len(comp) != c1 - c0 + 1 or len(comp) < 4:
            continue
        vals = [int(grid[r0, c]) for c in range(c0, c1 + 1)]
        colors = list(dict.fromkeys(vals))
        if len(colors) != 2:
            continue
        if all(vals[i] == vals[i % 2] for i in range(len(vals))) and vals[0] != vals[1]:
            candidates.append((int(vals[0]), int(vals[1])))

    assert len(candidates) == 1
    return candidates[0]


def _trace_path(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    assert coords
    cells = set(coords)
    degrees = {}
    for cell in coords:
        degrees[cell] = sum((nbr in cells) for nbr in _neighbors4(*cell))
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2
    assert all(degree in (1, 2) for degree in degrees.values())

    ordered = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while True:
        nxts = [nbr for nbr in _neighbors4(*cur) if nbr in cells and nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt

    assert len(ordered) == len(coords)
    return ordered


def _line_from_path(path: list[tuple[int, int]], legend: tuple[int, int], bg: int) -> np.ndarray:
    length = len(path)
    segments = [0] * length
    turns = 0
    for i in range(2, length):
        prev_dir = (path[i - 1][0] - path[i - 2][0], path[i - 1][1] - path[i - 2][1])
        next_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        segments[i] = segments[i - 1]
        if next_dir != prev_dir:
            turns += 1
            segments[i] += 1
    if length >= 2:
        segments[1] = 0

    line = np.full(length, int(bg), dtype=int)
    for i in range(length):
        line[i] = int(legend[segments[i] % 2])
    line[0] = YELLOW if turns % 2 == 0 else RED
    return line


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold each framed path into a horizontal line, alternating legend colors at turns.

    The endpoint appearing first in reading order is treated as the entrance so the
    path direction is deterministic.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(Counter(map(int, g.reshape(-1))).most_common(1)[0][0])
    frames = _find_frames(g, bg)
    legend = _find_legend(g, bg, frames)

    lines: list[np.ndarray] = []
    for r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        path_mask = inner != int(bg)
        assert np.any(path_mask)
        path = _trace_path(path_mask)
        lines.append(_line_from_path(path, legend, bg))

    out_h = 2 * len(lines) - 1
    out_w = max(len(line) for line in lines)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for i, line in enumerate(lines):
        out[2 * i, : len(line)] = line
    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    assert parts >= 0
    if parts == 0:
        assert total == 0
        return []
    assert total >= parts
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _stitch_path(
    start_row: int,
    horizontal_parts: list[int],
    vertical_parts: list[int],
    vertical_step: int,
) -> list[tuple[int, int]]:
    r, c = int(start_row), 0
    path = [(r, c)]
    for i, hlen in enumerate(horizontal_parts):
        for _ in range(hlen):
            c += 1
            path.append((r, c))
        if i < len(vertical_parts):
            for _ in range(vertical_parts[i]):
                r += vertical_step
                path.append((r, c))
    return path


def _build_base_path(inner_h: int, inner_w: int, turns: int) -> list[tuple[int, int]] | None:
    if turns == 0:
        if inner_h < 3:
            return None
        row = random.randint(1, inner_h - 2)
        return [(row, c) for c in range(inner_w)]

    if turns % 2 == 0:
        hseg = turns // 2 + 1
        vseg = turns // 2
        if inner_w - 1 < hseg or inner_h < vseg + 3:
            return None

        direction = random.choice((-1, 1))
        options: list[tuple[int, int]] = []
        if direction == 1:
            for start_row in range(1, inner_h - 1 - vseg):
                for end_row in range(start_row + vseg, inner_h - 1):
                    options.append((start_row, end_row))
        else:
            for start_row in range(vseg + 1, inner_h - 1):
                for end_row in range(1, start_row - vseg + 1):
                    options.append((start_row, end_row))
        if not options:
            return None

        start_row, end_row = random.choice(options)
        dy = abs(end_row - start_row)
        hparts = _positive_composition(inner_w - 1, hseg)
        vparts = _positive_composition(dy, vseg)
        return _stitch_path(start_row, hparts, vparts, 1 if end_row > start_row else -1)

    hseg = (turns + 1) // 2
    vseg = hseg
    if inner_w < hseg + 2 or inner_h < vseg + 2:
        return None

    direction = random.choice((-1, 1))
    start_options = range(1, inner_h - vseg) if direction == 1 else range(vseg, inner_h - 1)
    start_rows = list(start_options)
    if not start_rows:
        return None

    start_row = random.choice(start_rows)
    if direction == 1:
        total_vertical = inner_h - 1 - start_row
    else:
        total_vertical = start_row
    if total_vertical < vseg:
        return None

    x_end = random.randint(hseg, inner_w - 2)
    hparts = _positive_composition(x_end, hseg)
    vparts = _positive_composition(total_vertical, vseg)
    return _stitch_path(start_row, hparts, vparts, direction)


def _rotate_path(
    path: list[tuple[int, int]],
    inner_h: int,
    inner_w: int,
    quarter_turns: int,
) -> tuple[list[tuple[int, int]], int, int]:
    coords = list(path)
    h, w = inner_h, inner_w
    for _ in range(quarter_turns % 4):
        coords = [(c, h - 1 - r) for r, c in coords]
        h, w = w, h
    return coords, h, w


def _sample_path(parity: int) -> tuple[int, int, list[tuple[int, int]], int]:
    turn_choices = [t for t in range(6) if t % 2 == parity]
    for _ in range(MAX_PATH_ATTEMPTS):
        turns = random.choice(turn_choices)
        inner_h = random.randint(5, 9)
        inner_w = random.randint(6, 12)
        path = _build_base_path(inner_h, inner_w, turns)
        if path is None:
            continue
        path, inner_h, inner_w = _rotate_path(path, inner_h, inner_w, random.randint(0, 3))
        return inner_h, inner_w, path, turns
    raise RuntimeError("failed to sample a valid path")


def _render_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    path: list[tuple[int, int]],
    path_color: int,
) -> None:
    grid[top, left : left + inner_w + 2] = FRAME
    grid[top + inner_h + 1, left : left + inner_w + 2] = FRAME
    grid[top : top + inner_h + 2, left] = FRAME
    grid[top : top + inner_h + 2, left + inner_w + 1] = FRAME
    for r, c in path:
        grid[top + 1 + r, left + 1 + c] = int(path_color)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(2, 4)
    parities = [0, 1]
    while len(parities) < frame_count:
        parities.append(random.randint(0, 1))
    random.shuffle(parities)

    specs = [_sample_path(parity) for parity in parities]
    inner_sizes = [(inner_h, inner_w) for inner_h, inner_w, _path, _turns in specs]
    frame_sizes = [(inner_h + 2, inner_w + 2) for inner_h, inner_w in inner_sizes]

    palette = list(random.sample(FREE_COLORS, frame_count + 2))
    legend_colors = (palette[0], palette[1])
    path_colors = palette[2:]

    gaps = [random.randint(2, 4) for _ in range(frame_count - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    legend_gap = random.randint(1, 3)
    legend_top = random.choice((True, False))
    legend_len = random.randint(4, 7)

    frame_band_h = max(height for height, _width in frame_sizes)
    total_frame_w = sum(width for _height, width in frame_sizes) + sum(gaps)
    grid_w = left_margin + total_frame_w + right_margin
    grid_h = top_margin + frame_band_h + bottom_margin + legend_gap + 1
    if grid_w * grid_h < 80:
        grid_w += 4
        grid_h += 2

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    frame_top_base = top_margin if legend_top else top_margin
    if legend_top:
        frame_top_base = top_margin + 1 + legend_gap
        legend_row = top_margin
    else:
        frame_top_base = top_margin
        legend_row = top_margin + frame_band_h + legend_gap

    x = left_margin
    boxes: list[tuple[int, int, int, int]] = []
    for i, (spec, path_color) in enumerate(zip(specs, path_colors)):
        inner_h, inner_w, path, _turns = spec
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        top = frame_top_base + random.randint(0, frame_band_h - frame_h)
        _render_frame(grid, top, x, inner_h, inner_w, path, path_color)
        boxes.append((top, x, top + frame_h - 1, x + frame_w - 1))
        x += frame_w
        if i + 1 < frame_count:
            x += gaps[i]

    legend_start_max = grid_w - right_margin - legend_len
    if legend_start_max < left_margin:
        return None
    legend_start = random.randint(left_margin, legend_start_max)
    for i in range(legend_len):
        grid[legend_row, legend_start + i] = int(legend_colors[i % 2])

    counts = Counter(map(int, grid.reshape(-1)))
    if counts[BG] <= max(count for color, count in counts.items() if color != BG):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape[0] != 2 * frame_count - 1:
        return None
    if not np.any(out == RED) or not np.any(out == YELLOW):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
