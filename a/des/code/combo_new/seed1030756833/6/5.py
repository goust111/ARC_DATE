from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for sr, sc in np.argwhere(grid == color):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if _in_bounds((h, w), rr, cc) and not seen[rr, cc] and int(grid[rr, cc]) == color:
                    seen[rr, cc] = True
                    q.append((rr, cc))

        arr = np.array(coords, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        components.append(
            {
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": len(coords),
            }
        )

    return components


def _terminal_component(grid: np.ndarray, color: int) -> dict:
    dominoes = []
    for comp in _components_of_color(grid, color):
        r0, c0, r1, c1 = comp["bbox"]
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if comp["area"] == 2 and ((bh, bw) == (1, 2) or (bh, bw) == (2, 1)):
            dominoes.append(comp)
    assert len(dominoes) == 1, "each color must have exactly one 2-cell terminal"
    return dominoes[0]


def _scan_track(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> tuple[str, list[tuple[int, int]]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1

    if (bh, bw) == (1, 2):
        r = r0
        options: list[tuple[int, int, int, int]] = []
        if c0 - 1 >= 0 and int(grid[r, c0 - 1]) in (GREY, color):
            options.append((0, -1, r, c0 - 1))
        if c1 + 1 < w and int(grid[r, c1 + 1]) in (GREY, color):
            options.append((0, 1, r, c1 + 1))
        orientation = "horizontal"
    else:
        c = c0
        options = []
        if r0 - 1 >= 0 and int(grid[r0 - 1, c]) in (GREY, color):
            options.append((-1, 0, r0 - 1, c))
        if r1 + 1 < h and int(grid[r1 + 1, c]) in (GREY, color):
            options.append((1, 0, r1 + 1, c))
        orientation = "vertical"

    assert len(options) == 1, "terminal must touch exactly one straight track side"
    dr, dc, r, c = options[0]
    path: list[tuple[int, int]] = []
    while _in_bounds((h, w), r, c) and int(grid[r, c]) in (GREY, color):
        path.append((r, c))
        r += dr
        c += dc

    assert path, "track must contain at least one non-terminal cell"
    return orientation, path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored tokens along their grey conveyor tracks into terminal-grown bar charts.

    Input structure:
    - The dominant background contains several non-intersecting one-cell-wide Grey straight tracks.
    - Each track is horizontal or vertical and has a unique non-background, non-Grey 2x1 terminal block at one end.
    - Along each track, excluding the terminal, isolated single pixels match that track's terminal color.

    Rule:
    - For each terminal color, count the matching colored pixels lying on its own Grey track.
    - Replace those scattered colored pixels with Grey, preserving the track line and the terminal block.
    - If the track is horizontal, grow a solid two-column bar upward from the terminal with height equal to the count.
    - If the track is vertical, grow a solid two-row bar leftward from the terminal with length equal to the count.

    Key insight:
    - The colored pixels do not move visibly along the Grey conveyor; their count becomes a perpendicular bar grown from the same-colored terminal.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    assert bg != GREY, "background and Grey track color must differ"

    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, GREY)]
    assert colors, "expected at least one terminal color"

    specs: list[tuple[str, tuple[int, int, int, int], int, int]] = []
    scatter_cells: list[tuple[int, int]] = []

    for color in colors:
        terminal = _terminal_component(g, color)
        bbox = terminal["bbox"]
        orientation, track_path = _scan_track(g, color, bbox)

        terminal_cells = set(terminal["coords"])
        track_scatter = [(r, c) for r, c in track_path if int(g[r, c]) == color]
        expected_color_cells = terminal_cells | set(track_scatter)
        actual_color_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == color)}
        assert actual_color_cells == expected_color_cells, "all colored pixels must belong to their terminal track"

        specs.append((orientation, bbox, color, len(track_scatter)))
        scatter_cells.extend(track_scatter)

    out = g.copy()
    for r, c in scatter_cells:
        out[r, c] = GREY

    for orientation, bbox, color, count in specs:
        if count == 0:
            continue
        r0, c0, r1, c1 = bbox
        if orientation == "horizontal":
            assert r0 - count >= 0, "upward bar must remain in bounds"
            assert np.all(g[r0 - count : r0, c0 : c1 + 1] == bg), "bar growth area must be background"
            out[r0 - count : r0, c0 : c1 + 1] = color
        else:
            assert c0 - count >= 0, "leftward bar must remain in bounds"
            assert np.all(g[r0 : r1 + 1, c0 - count : c0] == bg), "bar growth area must be background"
            out[r0 : r1 + 1, c0 - count : c0] = color

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 200
MIN_SIZE = 16
MAX_SIZE = 30
MIN_TRACKS = 2
MAX_TRACKS = 4
MAX_COUNT = 6


def _expanded_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _sample_scatter_distances(track_len: int, count: int) -> set[int]:
    choices = list(range(2, track_len + 1))
    for _ in range(80):
        picked = sorted(random.sample(choices, count))
        if all(b - a > 1 for a, b in zip(picked, picked[1:])):
            return set(picked)

    fallback = list(range(2, track_len + 1, 2))
    assert len(fallback) >= count
    return set(random.sample(fallback, count))


def _sample_track_spec(h: int, w: int, orientation: str) -> dict | None:
    count = random.randint(1, MAX_COUNT)
    min_len = max(4, 2 * count)
    max_len = min(14, (w if orientation == "horizontal" else h) - 3)
    if min_len > max_len:
        return None
    track_len = random.randint(min_len, max_len)
    direction = random.choice((-1, 1))

    if orientation == "horizontal":
        if h <= count:
            return None
        r = random.randint(count, h - 1)
        if direction < 0:
            lo, hi = track_len, w - 2
        else:
            lo, hi = 0, w - track_len - 2
        if lo > hi:
            return None
        c0 = random.randint(lo, hi)
        terminal = [(r, c0), (r, c0 + 1)]
        if direction < 0:
            path = [(r, c0 - d) for d in range(1, track_len + 1)]
        else:
            path = [(r, c0 + 1 + d) for d in range(1, track_len + 1)]
        bar = [(rr, cc) for rr in range(r - count, r) for cc in (c0, c0 + 1)]
    else:
        if w <= count:
            return None
        c = random.randint(count, w - 1)
        if direction < 0:
            lo, hi = track_len, h - 2
        else:
            lo, hi = 0, h - track_len - 2
        if lo > hi:
            return None
        r0 = random.randint(lo, hi)
        terminal = [(r0, c), (r0 + 1, c)]
        if direction < 0:
            path = [(r0 - d, c) for d in range(1, track_len + 1)]
        else:
            path = [(r0 + 1 + d, c) for d in range(1, track_len + 1)]
        bar = [(rr, cc) for rr in (r0, r0 + 1) for cc in range(c - count, c)]

    distances = _sample_scatter_distances(track_len, count)
    scatter = {path[d - 1] for d in distances}
    input_cells = set(path) | set(terminal)
    semantic_cells = input_cells | set(bar)

    return {
        "path": path,
        "terminal": terminal,
        "scatter": scatter,
        "input_cells": input_cells,
        "semantic_cells": semantic_cells,
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    palette = list(range(10))
    bg_choices = [c for c in palette if c != GREY]
    bg = random.choice(bg_choices)
    colors = random.sample([c for c in palette if c not in (bg, GREY)], random.randint(MIN_TRACKS, MAX_TRACKS))

    orientations = ["horizontal", "vertical"]
    while len(orientations) < len(colors):
        orientations.append(random.choice(("horizontal", "vertical")))
    random.shuffle(orientations)

    grid = np.full((h, w), bg, dtype=int)
    reserved_with_gap: set[tuple[int, int]] = set()

    for color, orientation in zip(colors, orientations):
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            spec = _sample_track_spec(h, w, orientation)
            if spec is None:
                continue
            if spec["semantic_cells"] & reserved_with_gap:
                continue

            for r, c in spec["path"]:
                grid[r, c] = GREY
            for r, c in spec["terminal"]:
                grid[r, c] = color
            for r, c in spec["scatter"]:
                grid[r, c] = color

            reserved_with_gap |= _expanded_cells(spec["semantic_cells"], h, w)
            placed = True
            break

        if not placed:
            return None

    if _background_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid conveyor terminal sample")
