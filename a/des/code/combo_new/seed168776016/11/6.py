import random
from typing import Iterable

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BACKGROUND = 0
GEN_ATTEMPTS = 800
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 80


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _neighbors8(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _find_source_square(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    candidates: list[np.ndarray] = []

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if not bool(mask[r0 : r0 + 3, c0 : c0 + 3].all()):
                continue

            source = np.zeros_like(mask, dtype=bool)
            source[r0 : r0 + 3, c0 : c0 + 3] = True
            attachments: set[tuple[int, int]] = set()
            for r in range(r0, r0 + 3):
                for c in range(c0, c0 + 3):
                    for rr, cc in _neighbors4(r, c, h, w):
                        if mask[rr, cc] and not source[rr, cc]:
                            attachments.add((rr, cc))
            if len(attachments) == 1:
                candidates.append(source)

    assert len(candidates) == 1, "expected exactly one 3x3 source square"
    return candidates[0]


def _analyze_color(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    h, w = mask.shape
    source = _find_source_square(mask)
    outside = mask & ~source
    assert bool(outside.any()), "each color needs a non-source path/sink"

    degrees = np.zeros_like(mask, dtype=int)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        degrees[r, c] = sum(1 for rr, cc in _neighbors4(r, c, h, w) if mask[rr, cc])

    sinks = [(int(r), int(c)) for r, c in np.argwhere(outside & (degrees == 1))]
    assert len(sinks) == 1, "expected exactly one 1x1 sink endpoint"
    sink = sinks[0]

    prev: tuple[int, int] | None = None
    cur = sink
    last_dir: tuple[int, int] | None = None
    turns = 0
    safety = int(mask.sum()) + 2

    for _ in range(safety):
        source_neighbors = [
            (rr, cc) for rr, cc in _neighbors4(cur[0], cur[1], h, w) if source[rr, cc]
        ]
        if source_neighbors:
            assert len(source_neighbors) == 1, "path must attach to source at one cell"
            step_dir = (
                int(source_neighbors[0][0] - cur[0]),
                int(source_neighbors[0][1] - cur[1]),
            )
            if last_dir is not None and step_dir != last_dir:
                turns += 1
            assert 0 <= turns <= 3, "turn count must stay in range 0..3"
            return source, sink, turns

        next_pixels = [
            (rr, cc)
            for rr, cc in _neighbors4(cur[0], cur[1], h, w)
            if mask[rr, cc] and not source[rr, cc] and (prev is None or (rr, cc) != prev)
        ]
        assert len(next_pixels) == 1, "outside path must be a simple chain"
        nxt = next_pixels[0]
        step_dir = (int(nxt[0] - cur[0]), int(nxt[1] - cur[1]))
        if last_dir is not None and step_dir != last_dir:
            turns += 1
        prev, cur = cur, nxt
        last_dir = step_dir

    raise AssertionError("path tracing did not reach the source")


def _paint_centered_square(grid: np.ndarray, center: tuple[int, int], radius: int, color: int) -> None:
    h, w = grid.shape
    r, c = center
    r0 = max(0, int(r - radius))
    r1 = min(h - 1, int(r + radius))
    c0 = max(0, int(c - radius))
    c1 = min(w - 1, int(c + radius))
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each 1x1 sink into a turn-sized square based on the bends in its same-color path.

    Input structure:
    - Background is black and the grid is roughly 20x20 to 25x25.
    - Each non-background color forms one connected object made of a solid 3x3 source block, a 1-pixel-wide orthogonal path, and a 1x1 sink pixel.
    - The source block and sink share the path color, and every color appears in exactly one such pair.

    Rule:
    - For each color, identify the 3x3 source and the 1x1 sink, then trace the same-color path between them.
    - Count the number of direction changes T along that path.
    - Replace the sink by a solid square of side length 2T+1 centered on the original sink pixel, clipping at grid borders if needed.
    - Leave the original source block and the connecting path unchanged.

    Key insight:
    - The decisive hidden parameter is not path length but how many times the path turns before reaching the sink.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _bg_color(g)

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        mask = g == int(color)
        if not bool(mask.any()):
            continue
        _source, sink, turns = _analyze_color(mask)
        _paint_centered_square(out, sink, turns, int(color))

    return out


def _turn_dir(direction: int, delta: int) -> int:
    return int((direction + delta) % 4)


def _rasterize(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(vertices) - 1):
        r0, c0 = vertices[i]
        r1, c1 = vertices[i + 1]
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            seg = [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
        else:
            step = 1 if r1 >= r0 else -1
            seg = [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]
        if cells:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _source_cells_from_start(start: tuple[int, int], direction: int) -> set[tuple[int, int]]:
    sr, sc = start
    if direction == 0:
        r0, c0 = sr + 1, sc - 1
    elif direction == 1:
        r0, c0 = sr - 1, sc - 3
    elif direction == 2:
        r0, c0 = sr - 3, sc - 1
    else:
        r0, c0 = sr - 1, sc + 1
    return {(int(r), int(c)) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}


def _attachment_cells(source: set[tuple[int, int]], path: set[tuple[int, int]]) -> set[tuple[int, int]]:
    attachments: set[tuple[int, int]] = set()
    for r, c in source:
        for dr, dc in DIRS:
            rr = r + dr
            cc = c + dc
            if (rr, cc) in path:
                attachments.add((rr, cc))
    return attachments


def _make_shape(turns: int, rng: random.Random) -> dict[str, object] | None:
    for _ in range(SHAPE_ATTEMPTS):
        directions = [rng.randrange(4)]
        for _ in range(turns):
            directions.append(_turn_dir(directions[-1], rng.choice((-1, 1))))

        lengths: list[int] = []
        for i in range(turns + 1):
            if i == 0:
                lo, hi = (1, 5) if turns == 0 else (2, 5)
            elif i == turns:
                lo, hi = turns + 1, min(turns + 4, 7)
            else:
                lo, hi = 1, 4
            lengths.append(rng.randint(lo, hi))

        vertices = [(0, 0)]
        cur = (0, 0)
        for direction, length in zip(directions, lengths):
            dr, dc = DIRS[direction]
            cur = (int(cur[0] + dr * length), int(cur[1] + dc * length))
            vertices.append(cur)

        path_list = _rasterize(vertices)
        if len(path_list) != len(set(path_list)):
            continue

        source = _source_cells_from_start((0, 0), directions[0])
        path = set(path_list)
        if source & path:
            continue
        if _attachment_cells(source, path) != {(0, 0)}:
            continue

        sink = path_list[-1]
        reserved = {
            (int(r), int(c))
            for r in range(int(sink[0] - turns), int(sink[0] + turns + 1))
            for c in range(int(sink[1] - turns), int(sink[1] + turns + 1))
        }
        if source & reserved:
            continue

        return {
            "turns": int(turns),
            "source": source,
            "path_list": path_list,
            "path": path,
            "sink": sink,
            "reserved": reserved,
        }

    return None


def _translated(cells: Iterable[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in cells}


def _clipped(center: tuple[int, int], radius: int, h: int, w: int) -> bool:
    if radius == 0:
        return False
    r, c = center
    return (
        int(r - radius) < 0
        or int(r + radius) >= h
        or int(c - radius) < 0
        or int(c + radius) >= w
    )


def _buffer_cells(cells: Iterable[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    buffered: set[tuple[int, int]] = set()
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            buffered.add((int(r), int(c)))
        for rr, cc in _neighbors8(int(r), int(c), h, w):
            buffered.add((rr, cc))
    return buffered


def _place_shape(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    shape: dict[str, object],
    color: int,
    want_clip: bool,
    rng: random.Random,
) -> dict[str, object] | None:
    h, w = grid.shape
    source = set(shape["source"])
    path = set(shape["path"])
    input_cells = source | path
    reserved = set(shape["reserved"])
    sink = tuple(shape["sink"])
    turns = int(shape["turns"])

    rows = [r for r, _ in input_cells]
    cols = [c for _, c in input_cells]
    dr_min = -min(rows)
    dr_max = h - 1 - max(rows)
    dc_min = -min(cols)
    dc_max = w - 1 - max(cols)
    if dr_min > dr_max or dc_min > dc_max:
        return None

    offsets = [
        (dr, dc)
        for dr in range(int(dr_min), int(dr_max) + 1)
        for dc in range(int(dc_min), int(dc_max) + 1)
    ]
    rng.shuffle(offsets)

    preferred: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for dr, dc in offsets:
        source_t = _translated(source, dr, dc)
        path_t = _translated(path, dr, dc)
        input_t = source_t | path_t
        reserved_t = {
            (int(r + dr), int(c + dc))
            for r, c in reserved
            if 0 <= int(r + dr) < h and 0 <= int(c + dc) < w
        }
        footprint = input_t | reserved_t
        if footprint & blocked:
            continue

        clipped = _clipped((int(sink[0] + dr), int(sink[1] + dc)), turns, h, w)
        if clipped == want_clip:
            preferred.append((dr, dc))
        else:
            fallback.append((dr, dc))

    choices = preferred if preferred else fallback
    if not choices:
        return None

    dr, dc = rng.choice(choices)
    source_t = _translated(source, dr, dc)
    path_t_list = [(int(r + dr), int(c + dc)) for r, c in shape["path_list"]]
    path_t = set(path_t_list)
    input_t = source_t | path_t
    reserved_t = {
        (int(r + dr), int(c + dc))
        for r, c in reserved
        if 0 <= int(r + dr) < h and 0 <= int(c + dc) < w
    }

    for r, c in input_t:
        grid[r, c] = int(color)

    return {
        "color": int(color),
        "turns": turns,
        "sink": (int(sink[0] + dr), int(sink[1] + dc)),
        "input_cells": input_t,
        "reserved_cells": reserved_t,
    }


def _expected_output(grid: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for placement in placements:
        _paint_centered_square(
            out,
            tuple(placement["sink"]),
            int(placement["turns"]),
            int(placement["color"]),
        )
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(2, 4)
    if pair_count == 4:
        h = rng.randint(23, 25)
        w = rng.randint(23, 25)
    else:
        h = rng.randint(20, 25)
        w = rng.randint(20, 25)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    colors = rng.sample(COLOR_POOL, pair_count)
    turns_list = rng.sample([0, 1, 2, 3], pair_count)

    order = sorted(range(pair_count), key=lambda i: (-turns_list[i], colors[i]))
    blocked: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []

    want_clip_index = None
    clip_candidates = [i for i, turns in enumerate(turns_list) if turns > 0]
    if clip_candidates and rng.random() < 0.45:
        want_clip_index = rng.choice(clip_candidates)

    for idx in order:
        turns = int(turns_list[idx])
        color = int(colors[idx])
        want_clip = idx == want_clip_index

        placed = None
        for _ in range(PLACEMENT_ATTEMPTS):
            shape = _make_shape(turns, rng)
            if shape is None:
                continue
            placed = _place_shape(grid, blocked, shape, color, want_clip, rng)
            if placed is None:
                continue
            blocked |= _buffer_cells(
                set(placed["input_cells"]) | set(placed["reserved_cells"]),
                h,
                w,
            )
            placements.append(placed)
            break

        if placed is None:
            return None

    expected = _expected_output(grid, placements)

    seen_colors = sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND)
    if len(seen_colors) != pair_count:
        return None

    expected_turns = {int(p["color"]): int(p["turns"]) for p in placements}
    for color in seen_colors:
        try:
            _source, sink, turns = _analyze_color(grid == int(color))
        except AssertionError:
            return None
        if turns != expected_turns[int(color)]:
            return None
        if tuple(sink) != tuple(next(p["sink"] for p in placements if int(p["color"]) == int(color))):
            return None

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
