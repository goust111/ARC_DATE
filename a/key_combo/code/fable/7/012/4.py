"""ARC task: decode a marker-directed run-length strip into concentric bands."""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400
_generation_index = 0


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (n, 2) arrays of (row, column)."""
    seen = np.zeros(mask.shape, dtype=bool)
    found: list[np.ndarray] = []
    height, width = mask.shape
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        queue = deque([(row, col)])
        seen[row, col] = True
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        found.append(np.asarray(cells, dtype=int))
    return found


def _frame_candidates(grid: np.ndarray, background: int):
    candidates = []
    for color in set(map(int, grid.ravel())) - {background}:
        for coords in _components(grid == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
                continue
            expected = {
                *((r0, c) for c in range(c0, c1 + 1)),
                *((r1, c) for c in range(c0, c1 + 1)),
                *((r, c0) for r in range(r0, r1 + 1)),
                *((r, c1) for r in range(r0, r1 + 1)),
            }
            actual = set(map(tuple, coords.tolist()))
            if actual == expected:
                candidates.append((color, (r0, c0, r1, c1), actual))
    return candidates


def _decode_strip(grid: np.ndarray, remaining: set[tuple[int, int]], layers: int):
    """Find the perpendicular singleton marker and return strip colors from its end."""
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    if len(remaining) != layers + 1:
        raise AssertionError("expected one strip pixel per layer plus one endpoint marker")

    for marker in remaining:
        strip = remaining - {marker}
        rows = {r for r, _ in strip}
        cols = {c for _, c in strip}
        ordered: list[tuple[int, int]] | None = None
        endpoints: tuple[tuple[int, int], tuple[int, int]] | None = None

        if len(rows) == 1:
            row = next(iter(rows))
            values = sorted(c for _, c in strip)
            if values == list(range(values[0], values[-1] + 1)):
                endpoints = ((row, values[0]), (row, values[-1]))
                # A perpendicular marker keeps its role structurally visible.
                near = [p for p in endpoints if marker[1] == p[1] and abs(marker[0] - p[0]) == 1]
                if len(near) == 1:
                    ordered = [(row, c) for c in values]
                    if near[0] == endpoints[1]:
                        ordered.reverse()
        elif len(cols) == 1:
            col = next(iter(cols))
            values = sorted(r for r, _ in strip)
            if values == list(range(values[0], values[-1] + 1)):
                endpoints = ((values[0], col), (values[-1], col))
                near = [p for p in endpoints if marker[0] == p[0] and abs(marker[1] - p[1]) == 1]
                if len(near) == 1:
                    ordered = [(r, col) for r in values]
                    if near[0] == endpoints[1]:
                        ordered.reverse()

        if ordered is not None:
            marker_color = int(grid[marker])
            if all(int(grid[p]) != marker_color for p in ordered):
                candidates.append((marker, ordered))

    assert len(candidates) == 1, "endpoint geometry must determine one marker and one reading direction"
    marker, ordered = candidates[0]
    return marker, [int(grid[p]) for p in ordered]


def _runs(colors: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for color in colors:
        if runs and runs[-1][0] == color:
            old_color, length = runs[-1]
            runs[-1] = (old_color, length + 1)
        else:
            runs.append((color, 1))
    return runs


def _paint_runs(out: np.ndarray, frame: tuple[int, int, int, int], runs: list[tuple[int, int]]) -> None:
    r0, c0, r1, c1 = frame
    inner_h, inner_w = r1 - r0 - 1, c1 - c0 - 1
    rr, cc = np.indices((inner_h, inner_w))
    depth = np.minimum.reduce((rr, cc, inner_h - 1 - rr, inner_w - 1 - cc))
    start = 0
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    for color, thickness in runs:
        interior[(depth >= start) & (depth < start + thickness)] = color
        start += thickness
    assert start == (min(inner_h, inner_w) + 1) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """Decode a marker-directed color strip into run-thick concentric rings inside a rectangular frame.

    Input structure:
    - A dominant background contains one separated, one-pixel-thick hollow rectangular frame with an empty interior.
    - A separated horizontal or vertical multicolor strip has total length equal to the frame interior's number of concentric layers.
    - One differently colored singleton marker touches one strip endpoint perpendicularly, making both the marker and marked end visible without color-id semantics.

    Rule:
    - Read the strip starting at the endpoint beside the marker and run-length decode its consecutive equal colors.
    - From the frame interior's own outer boundary, paint each run color inward for exactly that run's pixel length.
    - Preserve the frame, strip, marker, and all exterior background cells.

    Key insight:
    - The perpendicular endpoint marker fixes the reading direction, while each visible same-color run fixes a concentric band's thickness rather than merely naming one band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    background = _background(g)

    frames = _frame_candidates(g, background)
    assert len(frames) == 1, "expected exactly one hollow rectangular frame"
    _frame_color, frame, frame_cells = frames[0]
    r0, c0, r1, c1 = frame
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == background), "frame interior must start empty"

    inner_h, inner_w = r1 - r0 - 1, c1 - c0 - 1
    layers = (min(inner_h, inner_w) + 1) // 2
    foreground = set(map(tuple, np.argwhere(g != background).tolist()))
    remaining = foreground - frame_cells
    _marker, strip_colors = _decode_strip(g, remaining, layers)
    runs = _runs(strip_colors)
    assert sum(length for _color, length in runs) == layers

    out = g.copy()
    _paint_runs(out, frame, runs)
    return out


def _composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    points = [0, *cuts, total]
    return [b - a for a, b in zip(points, points[1:])]


def _frame_cells(frame: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    return {
        *((r0, c) for c in range(c0, c1 + 1)),
        *((r1, c) for c in range(c0, c1 + 1)),
        *((r, c0) for r in range(r0, r1 + 1)),
        *((r, c1) for r in range(r0, r1 + 1)),
    }


def _strip_layouts(shape: tuple[int, int], length: int, blocked: set[tuple[int, int]]):
    height, width = shape
    layouts = []
    for axis in (0, 1):
        for fixed in range(1, (height if axis == 0 else width) - 1):
            limit = width if axis == 0 else height
            for start in range(1, limit - length):
                strip = ([(fixed, start + i) for i in range(length)] if axis == 0
                         else [(start + i, fixed) for i in range(length)])
                for endpoint_index in (0, -1):
                    er, ec = strip[endpoint_index]
                    for side in (-1, 1):
                        marker = (er + side, ec) if axis == 0 else (er, ec + side)
                        occupied = set(strip) | {marker}
                        if not (0 <= marker[0] < height and 0 <= marker[1] < width):
                            continue
                        # Keep a full empty-cell moat between the two semantic objects.
                        if any(max(abs(r - br), abs(c - bc)) <= 1 for r, c in occupied for br, bc in blocked):
                            continue
                        layouts.append((strip, marker, endpoint_index))
    return layouts


def _oracle(grid: np.ndarray, frame: tuple[int, int, int, int], layer_colors: list[int]) -> np.ndarray:
    out = grid.copy()
    _paint_runs(out, frame, _runs(layer_colors))
    return out


def _gen_attempt(mode: int) -> np.ndarray | None:
    layers = random.randint(4, 6)
    short_inner = random.choice((2 * layers - 1, 2 * layers))
    long_inner = short_inner + random.randint(0, 5)
    if random.choice((False, True)):
        inner_h, inner_w = short_inner, long_inner
    else:
        inner_h, inner_w = long_inner, short_inner

    min_height, min_width = inner_h + 6, inner_w + 6
    if min_height > 30 or min_width > 30:
        return None
    height = random.randint(max(18, min_height), 30)
    width = random.randint(max(18, min_width), 30)
    fr0 = random.randint(1, height - inner_h - 3)
    fc0 = random.randint(1, width - inner_w - 3)
    frame = (fr0, fc0, fr0 + inner_h + 1, fc0 + inner_w + 1)
    frame_cells = _frame_cells(frame)

    layouts = _strip_layouts((height, width), layers, frame_cells)
    if not layouts:
        return None
    strip, marker, endpoint_index = random.choice(layouts)

    if mode == 0:
        lengths = [1] * layers  # deliberate pixel-palette lure
    else:
        part_count = random.randint(2, min(4, layers - 1))
        lengths = _composition(layers, part_count)
        if max(lengths) == 1:
            return None

    role_count = len(lengths) + 3
    if role_count > len(PALETTE):
        return None
    chosen = random.sample(PALETTE, role_count)
    background, frame_color, marker_color = chosen[:3]
    run_colors = chosen[3:]
    read_colors = [color for color, length in zip(run_colors, lengths) for _ in range(length)]

    grid = np.full((height, width), background, dtype=int)
    for cell in frame_cells:
        grid[cell] = frame_color
    grid[marker] = marker_color
    draw_colors = read_colors if endpoint_index == 0 else list(reversed(read_colors))
    for cell, color in zip(strip, draw_colors):
        grid[cell] = color

    expected = _oracle(grid, frame, read_colors)
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None

    # Semantic checks: only the cavity changes, direction matters, and repeated
    # run lengths cannot be replaced by one ring per distinct run.
    exterior = np.ones(grid.shape, dtype=bool)
    r0, c0, r1, c1 = frame
    exterior[r0 + 1 : r1, c0 + 1 : c1] = False
    if not np.array_equal(actual[exterior], grid[exterior]):
        return None
    reverse = _oracle(grid, frame, list(reversed(read_colors)))
    if np.array_equal(reverse, expected):
        return None
    if mode != 0:
        shortcut = _oracle(grid, frame, run_colors[:-1] + [run_colors[-1]] * (layers - len(run_colors) + 1))
        if np.array_equal(shortcut, expected):
            return None
    return grid


def generate_input() -> np.ndarray:
    """Generate diverse valid examples, cycling lure/repeated-run coverage."""
    global _generation_index
    mode = 0 if _generation_index % 3 == 0 else 1
    _generation_index += 1
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid run-length ring puzzle")
