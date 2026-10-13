from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _nonbg_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _path_cells(spec: dict) -> list[tuple[int, int]]:
    length = int(spec["length"])
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    if spec["orientation"] == "h":
        return [(r0, c0 + k) for k in range(length)]
    return [(r0 + k, c0) for k in range(length)]


def _protected_path_cells(spec: dict) -> list[tuple[int, int]]:
    cells = _path_cells(spec)
    if spec["op"] != "extend":
        return cells

    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    length = int(spec["length"])
    if spec["orientation"] == "h":
        return cells + [(r0, c0 - 1), (r0, c0 + length)]
    return cells + [(r0 - 1, c0), (r0 + length, c0)]


def _signal_cells(spec: dict) -> list[tuple[int, int]]:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    if spec["orientation"] == "h":
        return [(r0, c0 + k) for k in range(3)]
    return [(r0 + k, c0) for k in range(3)]


def _transformed_path_cells(spec: dict) -> list[tuple[int, int]]:
    op = spec["op"]
    orientation = spec["orientation"]
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    length = int(spec["length"])

    if orientation == "h":
        if op == "trim_start":
            c0 += 1
            length -= 1
        elif op == "trim_end":
            length -= 1
        else:
            c0 -= 1
            length += 2
        return [(r0, c0 + k) for k in range(length)]

    if op == "trim_start":
        r0 += 1
        length -= 1
    elif op == "trim_end":
        length -= 1
    else:
        r0 -= 1
        length += 2
    return [(r0 + k, c0) for k in range(length)]


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[int, dict], list[dict]]:
    bg = get_bg(grid)
    paths: dict[int, dict] = {}
    signals: list[dict] = []

    for coords in _nonbg_components(grid, bg):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        colors = {int(grid[r, c]) for r, c in coords}
        size = len(coords)

        is_path = len(colors) == 1 and (
            (height == 1 and width == size and size >= 3) or (width == 1 and height == size and size >= 3)
        )
        if is_path:
            orientation = "h" if height == 1 else "v"
            color = next(iter(colors))
            assert color not in paths, "path colors must be unique"
            paths[color] = {
                "color": int(color),
                "orientation": orientation,
                "r0": int(r0),
                "c0": int(c0),
                "length": int(size),
                "cells": coords,
            }
            continue

        is_signal = size == 3 and len(colors) == 3 and ((height, width) == (1, 3) or (height, width) == (3, 1))
        if is_signal:
            orientation = "h" if height == 1 else "v"
            if orientation == "h":
                color_order = [int(grid[r0, c0 + k]) for k in range(3)]
            else:
                color_order = [int(grid[r0 + k, c0]) for k in range(3)]
            signals.append(
                {
                    "orientation": orientation,
                    "r0": int(r0),
                    "c0": int(c0),
                    "colors": color_order,
                    "cells": coords,
                }
            )
            continue

        raise AssertionError("unexpected component in puzzle input")

    assert paths, "expected at least one path"
    assert signals, "expected at least one signal bar"
    return bg, paths, signals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode 1x3 or 3x1 signal bars and use them to truncate or extend matching straight paths.

    Input structure:
    - Background = most common color.
    - Several monochrome straight path segments, each horizontal or vertical.
    - Several 3-cell signal bars, each horizontal or vertical, with three distinct colors.
    - Each signal bar shares exactly one color with exactly one path.

    Rule:
    - Pair each signal bar to the path whose color appears inside the bar.
    - Signal orientation must match the operated path orientation.
    - Matching color at the first slot trims one cell from the path's start end.
    - Matching color at the middle slot extends the path by one cell at both ends.
    - Matching color at the last slot trims one cell from the path's end.
    - Remove all signal bars in the output.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, paths, signals = _parse_scene(g)

    out = g.copy()
    for signal in signals:
        for r, c in signal["cells"]:
            out[r, c] = bg
    for path in paths.values():
        for r, c in path["cells"]:
            out[r, c] = bg

    paired: dict[int, dict] = {}
    path_colors = set(paths)
    for signal in signals:
        matches = [color for color in signal["colors"] if color in path_colors]
        assert len(matches) == 1, "each signal must match exactly one path color"
        color = matches[0]
        assert color not in paired, "each path must have exactly one signal"
        assert paths[color]["orientation"] == signal["orientation"], "signal axis must match path axis"
        paired[color] = signal

    assert len(paired) == len(paths), "every path must be paired"

    for color, path in paths.items():
        signal = paired[color]
        match_idx = signal["colors"].index(color)
        op = ("trim_start", "extend", "trim_end")[match_idx]
        spec = {
            "color": int(color),
            "orientation": path["orientation"],
            "r0": int(path["r0"]),
            "c0": int(path["c0"]),
            "length": int(path["length"]),
            "op": op,
        }
        for r, c in _transformed_path_cells(spec):
            assert 0 <= r < h and 0 <= c < w, "transformed path must stay in bounds"
            out[r, c] = int(color)

    return out


OPS = ("trim_start", "extend", "trim_end")
GRID_H_RANGE = (15, 23)
GRID_W_RANGE = (15, 23)
PATH_COUNT_OPTIONS = (3, 4)
PATH_MARGIN = 3
MAX_PATH_LEN = 8
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _cells_available(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _sample_orientations(count: int) -> list[str]:
    orientations = ["h"] * (count // 2) + ["v"] * (count // 2)
    if count % 2:
        orientations.append(random.choice(["h", "v"]))
    random.shuffle(orientations)
    assert "h" in orientations and "v" in orientations
    return orientations


def _sample_operations(count: int) -> list[str]:
    ops = list(OPS) * (count // 3)
    if count % 3:
        ops.extend(random.sample(list(OPS), count % 3))
    random.shuffle(ops)
    assert all(op in ops for op in OPS)
    return ops


def _place_path(grid: np.ndarray, blocked: np.ndarray, spec: dict) -> bool:
    h, w = grid.shape
    span = w if spec["orientation"] == "h" else h
    max_len = min(MAX_PATH_LEN, span - 2 * PATH_MARGIN)
    if max_len < 3:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        length = random.randint(3, max_len)
        if spec["orientation"] == "h":
            r0 = random.randint(PATH_MARGIN, h - PATH_MARGIN - 1)
            c0 = random.randint(PATH_MARGIN, w - PATH_MARGIN - length)
        else:
            r0 = random.randint(PATH_MARGIN, h - PATH_MARGIN - length)
            c0 = random.randint(PATH_MARGIN, w - PATH_MARGIN - 1)

        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
        spec["length"] = int(length)
        protected = _protected_path_cells(spec)
        if not _cells_available(blocked, protected):
            continue

        for r, c in _path_cells(spec):
            grid[r, c] = int(spec["color"])
        _mark_blocked(blocked, protected)
        return True

    return False


def _signal_colors(path_color: int, op: str, filler_pool: list[int]) -> list[int]:
    a, b = random.sample(filler_pool, 2)
    if op == "trim_start":
        return [int(path_color), int(a), int(b)]
    if op == "trim_end":
        return [int(a), int(b), int(path_color)]
    return [int(a), int(path_color), int(b)]


def _place_signal(grid: np.ndarray, blocked: np.ndarray, path: dict, filler_pool: list[int]) -> dict | None:
    h, w = grid.shape
    colors = _signal_colors(int(path["color"]), path["op"], filler_pool)

    if path["orientation"] == "h":
        candidates = [(r, c0) for r in (0, 1, h - 2, h - 1) for c0 in range(w - 2)]
    else:
        candidates = [(r0, c) for c in (0, 1, w - 2, w - 1) for r0 in range(h - 2)]
    random.shuffle(candidates)

    for r0, c0 in candidates:
        spec = {"orientation": path["orientation"], "r0": int(r0), "c0": int(c0), "colors": colors}
        cells = _signal_cells(spec)
        if not _cells_available(blocked, cells):
            continue

        for idx, (r, c) in enumerate(cells):
            grid[r, c] = int(colors[idx])
        _mark_blocked(blocked, cells)
        return spec

    return None


def _expected_output(inp: np.ndarray, bg: int, paths: list[dict], signals: list[dict]) -> np.ndarray:
    out = inp.copy()
    for signal in signals:
        for r, c in _signal_cells(signal):
            out[r, c] = bg
    for path in paths:
        for r, c in _path_cells(path):
            out[r, c] = bg
    for path in paths:
        for r, c in _transformed_path_cells(path):
            out[r, c] = int(path["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    num_paths = random.choice(PATH_COUNT_OPTIONS)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    path_colors = [int(c) for c in palette[1 : 1 + num_paths]]
    filler_pool = [int(c) for c in palette[1 + num_paths :]]
    if len(filler_pool) < 2:
        return None

    orientations = _sample_orientations(num_paths)
    operations = _sample_operations(num_paths)
    paths = [
        {"color": color, "orientation": orient, "op": op}
        for color, orient, op in zip(path_colors, orientations, operations)
    ]
    random.shuffle(paths)

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for path in paths:
        if not _place_path(grid, blocked, path):
            return None

    signals: list[dict] = []
    for path in paths:
        signal = _place_signal(grid, blocked, path, filler_pool)
        if signal is None:
            return None
        signals.append(signal)

    expected = _expected_output(grid, bg, paths, signals)

    if get_bg(grid) != bg:
        return None
    if np.array_equal(expected, grid):
        return None
    if not {"h", "v"} <= {path["orientation"] for path in paths}:
        return None
    if not all(op in {path["op"] for path in paths} for op in OPS):
        return None

    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for task 3")
