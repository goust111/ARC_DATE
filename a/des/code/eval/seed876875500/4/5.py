from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

DIR_OFFSETS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
PERPENDICULARS = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}
SIDE_TO_DIR = {
    "top": "D",
    "bottom": "U",
    "left": "R",
    "right": "L",
}
REFLECTION_TO_COLOR = {
    1: GREEN,
    2: YELLOW,
    3: ORANGE,
}

EMITTER_PALETTE = [BLUE, RED, MAGENTA, BROWN]
SIDES = ["top", "bottom", "left", "right"]
MIRROR_LENGTHS = [2, 3]

GRID_H_RANGE = (17, 24)
GRID_W_RANGE = (17, 24)
PATH_COUNT_RANGE = (2, 4)
MIN_SEGMENT_LENGTH = 2
MAX_SEGMENT_LENGTH = 8
GENERATION_RETRIES = 300


def _advance(cell: tuple[int, int], direction: str, steps: int = 1) -> tuple[int, int]:
    dr, dc = DIR_OFFSETS[direction]
    r, c = cell
    return int(r + dr * steps), int(c + dc * steps)


def _segment_cells(start: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]]:
    return [_advance(start, direction, step) for step in range(1, length + 1)]


def _is_in_bounds(h: int, w: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _is_interior(h: int, w: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 < r < h - 1 and 0 < c < w - 1


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _receiver_color(emitter_color: int, reflections: int) -> int:
    if reflections <= 0:
        return int(emitter_color)
    return int(REFLECTION_TO_COLOR.get(reflections, ORANGE))


def _mirror_cells(
    hit: tuple[int, int],
    incoming_dir: str,
    outgoing_dir: str,
    length: int,
) -> list[tuple[int, int]]:
    r, c = hit
    if incoming_dir in ("U", "D"):
        assert outgoing_dir in ("L", "R")
        if outgoing_dir == "L":
            return [(r, c + i) for i in range(length)]
        return [(r, c - i) for i in range(length)]

    assert incoming_dir in ("L", "R")
    assert outgoing_dir in ("U", "D")
    if outgoing_dir == "U":
        return [(r + i, c) for i in range(length)]
    return [(r - i, c) for i in range(length)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4((int(rr), int(cc))):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _parse_mirrors(grid: np.ndarray) -> dict[tuple[int, int], dict[str, object]]:
    mirrors: dict[tuple[int, int], dict[str, object]] = {}
    for cells in _components(grid == GREY):
        rows = sorted({r for r, _ in cells})
        cols = sorted({c for _, c in cells})
        if len(rows) == 1:
            ordered = sorted(cells, key=lambda rc: rc[1])
            xs = [c for _, c in ordered]
            assert xs == list(range(xs[0], xs[-1] + 1))
            orientation = "H"
        else:
            assert len(cols) == 1
            ordered = sorted(cells, key=lambda rc: rc[0])
            ys = [r for r, _ in ordered]
            assert ys == list(range(ys[0], ys[-1] + 1))
            orientation = "V"
        assert 2 <= len(ordered) <= 3
        info = {
            "orientation": orientation,
            "cells": ordered,
        }
        for cell in ordered:
            assert cell not in mirrors
            mirrors[cell] = info
    return mirrors


def _emitters(grid: np.ndarray) -> list[tuple[int, int, int, str]]:
    h, w = grid.shape
    emitters: list[tuple[int, int, int, str]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(grid[r, c])
            if color in (BLACK, GREY, AZURE):
                continue
            if r == 0:
                direction = "D"
            elif r == h - 1:
                direction = "U"
            elif c == 0:
                direction = "R"
            else:
                direction = "L"
            emitters.append((int(color), int(r), int(c), direction))
    return emitters


def _trace_receiver(
    grid: np.ndarray,
    mirrors: dict[tuple[int, int], dict[str, object]],
    start: tuple[int, int],
    direction: str,
) -> tuple[int, int, int]:
    h, w = grid.shape
    r, c = start
    d = direction
    reflections = 0
    seen: set[tuple[int, int, str]] = set()

    while True:
        state = (int(r), int(c), d)
        assert state not in seen, "ray path must not cycle"
        seen.add(state)

        r, c = _advance((int(r), int(c)), d, 1)
        assert 0 <= r < h and 0 <= c < w, "ray left the grid before reaching a receiver"

        color = int(grid[r, c])
        if color == BLACK:
            continue
        if color == AZURE:
            return int(r), int(c), int(reflections)
        if color != GREY:
            raise AssertionError("ray hit an unexpected non-receiver object")

        info = mirrors[(int(r), int(c))]
        cells = info["cells"]
        if info["orientation"] == "H":
            assert d in ("U", "D")
            cols = [cc for _, cc in cells]
            center = (cols[0] + cols[-1]) / 2.0
            d = "L" if c < center else "R"
        else:
            assert info["orientation"] == "V"
            assert d in ("L", "R")
            rows = [rr for rr, _ in cells]
            center = (rows[0] + rows[-1]) / 2.0
            d = "U" if r < center else "D"
        reflections += 1


def main(grid: np.ndarray) -> np.ndarray:
    """Trace each emitter-to-receiver light path and recolor receivers by reflection count."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mirrors = _parse_mirrors(g)
    out = g.copy()

    for emitter_color, r, c, direction in _emitters(g):
        rr, cc, reflections = _trace_receiver(g, mirrors, (r, c), direction)
        out[rr, cc] = _receiver_color(emitter_color, reflections)

    return out


def _edge_positions(side: str, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    assert side == "right"
    return [(r, w - 1) for r in range(1, h - 1)]


def _max_steps_to_interior(h: int, w: int, start: tuple[int, int], direction: str) -> int:
    r, c = start
    if direction == "U":
        return int(r - 1)
    if direction == "D":
        return int(h - 2 - r)
    if direction == "L":
        return int(c - 1)
    return int(w - 2 - c)


def _valid_mirror_placement(
    h: int,
    w: int,
    mirror: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    grey_cells: set[tuple[int, int]],
) -> bool:
    mirror_set = set(mirror)
    if any(not _is_interior(h, w, cell) for cell in mirror):
        return False
    if any(cell in occupied for cell in mirror):
        return False
    for cell in mirror:
        for nbr in _neighbors4(cell):
            if nbr in grey_cells and nbr not in mirror_set:
                return False
    return True


def _extend_path(
    h: int,
    w: int,
    current: tuple[int, int],
    direction: str,
    bends_left: int,
    occupied: set[tuple[int, int]],
    grey_cells: set[tuple[int, int]],
    receiver_cells: set[tuple[int, int]],
    rng: random.Random,
) -> dict[str, object] | None:
    max_steps = min(MAX_SEGMENT_LENGTH, _max_steps_to_interior(h, w, current, direction))
    if max_steps < MIN_SEGMENT_LENGTH:
        return None

    if bends_left == 0:
        lengths = list(range(MIN_SEGMENT_LENGTH, max_steps + 1))
        rng.shuffle(lengths)
        for length in lengths:
            receiver = _advance(current, direction, length)
            if not _is_interior(h, w, receiver):
                continue
            segment = _segment_cells(current, direction, length)
            if any(cell in occupied for cell in segment):
                continue
            if receiver in receiver_cells:
                continue
            if any(nbr in receiver_cells for nbr in _neighbors4(receiver)):
                continue
            return {
                "receiver": receiver,
                "mirrors": [],
                "used": set(segment),
                "grey": set(),
                "receivers": {receiver},
            }
        return None

    turns = list(PERPENDICULARS[direction])
    rng.shuffle(turns)
    lengths = list(range(MIN_SEGMENT_LENGTH, max_steps + 1))
    rng.shuffle(lengths)

    for new_direction in turns:
        mirror_lengths = MIRROR_LENGTHS[:]
        rng.shuffle(mirror_lengths)
        for length in lengths:
            hit = _advance(current, direction, length)
            segment = _segment_cells(current, direction, length)
            if any(cell in occupied for cell in segment):
                continue
            for mirror_len in mirror_lengths:
                mirror = _mirror_cells(hit, direction, new_direction, mirror_len)
                mirror_set = set(mirror)
                extra_mirror_cells = mirror_set - {hit}
                if not _valid_mirror_placement(h, w, mirror, occupied | (set(segment) - {hit}), grey_cells):
                    continue
                child = _extend_path(
                    h,
                    w,
                    hit,
                    new_direction,
                    bends_left - 1,
                    occupied | set(segment) | extra_mirror_cells,
                    grey_cells | mirror_set,
                    receiver_cells,
                    rng,
                )
                if child is None:
                    continue
                return {
                    "receiver": child["receiver"],
                    "mirrors": [{"cells": mirror}] + list(child["mirrors"]),
                    "used": set(segment) | extra_mirror_cells | set(child["used"]),
                    "grey": mirror_set | set(child["grey"]),
                    "receivers": set(child["receivers"]),
                }

    return None


def _place_path(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    grey_cells: set[tuple[int, int]],
    receiver_cells: set[tuple[int, int]],
    side: str,
    emitter_color: int,
    reflections: int,
    rng: random.Random,
) -> tuple[dict[str, object], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    emitters = _edge_positions(side, h, w)
    rng.shuffle(emitters)

    for emitter in emitters:
        if emitter in occupied:
            continue
        result = _extend_path(
            h,
            w,
            emitter,
            SIDE_TO_DIR[side],
            reflections,
            occupied | {emitter},
            grey_cells,
            receiver_cells,
            rng,
        )
        if result is None:
            continue

        grid[emitter] = int(emitter_color)
        for mirror in result["mirrors"]:
            for cell in mirror["cells"]:
                grid[cell] = GREY
        receiver = result["receiver"]
        grid[receiver] = AZURE

        path_info = {
            "emitter": emitter,
            "color": int(emitter_color),
            "receiver": receiver,
            "reflections": int(reflections),
        }
        return (
            path_info,
            occupied | {emitter} | set(result["used"]),
            grey_cells | set(result["grey"]),
            receiver_cells | {receiver},
        )

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    path_count = rng.randint(*PATH_COUNT_RANGE)

    sides = rng.sample(SIDES, k=path_count)
    colors = rng.sample(EMITTER_PALETTE, k=path_count)
    reflections = rng.sample([0, 1, 2, 3], k=path_count)

    order = sorted(range(path_count), key=lambda idx: reflections[idx], reverse=True)

    grid = np.full((h, w), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()
    grey_cells: set[tuple[int, int]] = set()
    receiver_cells: set[tuple[int, int]] = set()
    paths: list[dict[str, object]] = []

    for idx in order:
        placed = _place_path(
            grid,
            occupied,
            grey_cells,
            receiver_cells,
            sides[idx],
            colors[idx],
            reflections[idx],
            rng,
        )
        if placed is None:
            return None
        path_info, occupied, grey_cells, receiver_cells = placed
        paths.append(path_info)

    expected = grid.copy()
    for path in paths:
        rr, cc = path["receiver"]
        expected[rr, cc] = _receiver_color(int(path["color"]), int(path["reflections"]))

    if len(_emitters(grid)) != path_count:
        return None
    if int(np.count_nonzero(grid == AZURE)) != path_count:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid light-mirror puzzle")
