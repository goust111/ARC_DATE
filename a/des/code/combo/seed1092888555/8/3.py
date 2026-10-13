from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BG = 0
MAX_GENERATION_ATTEMPTS = 400


def _segment_info(g: np.ndarray, color: int) -> dict:
    coords = np.argwhere(g == color)
    assert len(coords) >= 2
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    if r0 == r1:
        length = c1 - c0 + 1
        assert len(coords) == length
        assert np.all(g[r0, c0 : c1 + 1] == color)
        endpoints = [(r0, c0), (r0, c1)]
    else:
        assert c0 == c1
        length = r1 - r0 + 1
        assert len(coords) == length
        assert np.all(g[r0 : r1 + 1, c0] == color)
        endpoints = [(r0, c0), (r1, c0)]

    return {"color": color, "length": int(length), "endpoints": endpoints}


def _manhattan(p: tuple[int, int], q: tuple[int, int]) -> int:
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a chain of colored line segments into one horizontal line.

    Each non-black color appears in exactly one straight 1-pixel-thick segment.
    Segment endpoints 4-touch segment endpoints of neighboring segments, forming
    a simple chain. Starting from the top-left outer endpoint, traverse the
    chain and serialize segment colors left-to-right, repeating each color by
    the segment's original length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert len(colors) >= 2
    segments = [_segment_info(g, color) for color in colors]

    neighbors: dict[int, list[tuple[int, int, tuple[int, int], tuple[int, int]]]] = {
        i: [] for i in range(len(segments))
    }
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            found = []
            for ei in segments[i]["endpoints"]:
                for ej in segments[j]["endpoints"]:
                    if _manhattan(ei, ej) == 1:
                        found.append((ei, ej))
            assert len(found) <= 1
            if found:
                ei, ej = found[0]
                neighbors[i].append((j, 0, ei, ej))
                neighbors[j].append((i, 0, ej, ei))

    degrees = [len(neighbors[i]) for i in range(len(segments))]
    assert degrees.count(1) == 2
    assert all(d in (1, 2) for d in degrees)

    endpoint_choices = []
    for idx, degree in enumerate(degrees):
        if degree != 1:
            continue
        touching = neighbors[idx][0][2]
        outer = [p for p in segments[idx]["endpoints"] if p != touching]
        assert len(outer) == 1
        endpoint_choices.append((outer[0], idx))

    start_outer, start_idx = min(endpoint_choices)
    order: list[dict] = []
    prev_idx = None
    cur_idx = start_idx
    enter_endpoint = start_outer

    while True:
        seg = segments[cur_idx]
        order.append(seg)
        exit_endpoint = seg["endpoints"][0] if seg["endpoints"][1] == enter_endpoint else seg["endpoints"][1]

        next_options = []
        for nxt_idx, _, my_ep, nxt_ep in neighbors[cur_idx]:
            if nxt_idx == prev_idx:
                continue
            if my_ep == exit_endpoint:
                next_options.append((nxt_idx, nxt_ep))
        assert len(next_options) <= 1
        if not next_options:
            break
        prev_idx, cur_idx = cur_idx, next_options[0][0]
        enter_endpoint = next_options[0][1]

    assert len(order) == len(segments)

    out_colors = []
    for seg in order:
        out_colors.extend([seg["color"]] * seg["length"])
    return np.asarray(out_colors, dtype=int).reshape(1, -1)


def _dir_kind(d: tuple[int, int]) -> int:
    return 0 if d[0] == 0 else 1


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    return [
        (start[0] + k * direction[0], start[1] + k * direction[1])
        for k in range(length)
    ]


def _try_build_chain(rng: random.Random) -> tuple[list[dict], list[int]] | None:
    n_segments = rng.randint(3, 7)
    lengths = [rng.randint(2, 6) for _ in range(n_segments)]

    first_dir = rng.choice(DIRS4)
    segments = [{"cells": _segment_cells((0, 0), first_dir, lengths[0]), "dir": first_dir}]
    occupied = set(segments[0]["cells"])
    prev_end = segments[0]["cells"][-1]

    for i in range(1, n_segments):
        prev_dir = segments[-1]["dir"]
        wanted_same = rng.choice([False, True])
        candidates = []
        for link_dir in DIRS4:
            start = (prev_end[0] + link_dir[0], prev_end[1] + link_dir[1])
            if start in occupied:
                continue
            back_dir = (-link_dir[0], -link_dir[1])
            for new_dir in DIRS4:
                if new_dir == back_dir:
                    continue
                if (_dir_kind(new_dir) == _dir_kind(prev_dir)) != wanted_same:
                    continue
                cells = _segment_cells(start, new_dir, lengths[i])
                ok = True
                for idx, cell in enumerate(cells):
                    if cell in occupied:
                        ok = False
                        break
                    neighbors = [
                        (cell[0] + dr, cell[1] + dc)
                        for dr, dc in DIRS4
                        if (cell[0] + dr, cell[1] + dc) in occupied
                    ]
                    if idx == 0:
                        if sorted(neighbors) != [prev_end]:
                            ok = False
                            break
                    elif neighbors:
                        ok = False
                        break
                if ok:
                    candidates.append({"cells": cells, "dir": new_dir})

        if not candidates:
            return None
        seg = rng.choice(candidates)
        segments.append(seg)
        occupied.update(seg["cells"])
        prev_end = seg["cells"][-1]

    colors = rng.sample(COLOR_POOL, n_segments)
    return segments, colors


def _render_chain(segments: list[dict], colors: list[int], rng: random.Random) -> np.ndarray:
    all_cells = [cell for seg in segments for cell in seg["cells"]]
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    top_pad = rng.randint(1, 4)
    left_pad = rng.randint(1, 4)
    bottom_pad = rng.randint(1, 4)
    right_pad = rng.randint(1, 4)

    h = (max_r - min_r + 1) + top_pad + bottom_pad
    w = (max_c - min_c + 1) + left_pad + right_pad
    grid = np.full((h, w), BG, dtype=int)

    shift_r = top_pad - min_r
    shift_c = left_pad - min_c
    for seg, color in zip(segments, colors):
        for r, c in seg["cells"]:
            grid[r + shift_r, c + shift_c] = color
    return grid


def _expected_output(lengths: list[int], colors: list[int]) -> np.ndarray:
    seq = []
    for length, color in zip(lengths, colors):
        seq.extend([color] * length)
    return np.asarray(seq, dtype=int).reshape(1, -1)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _try_build_chain(rng)
        if built is None:
            continue
        segments, colors = built
        lengths = [len(seg["cells"]) for seg in segments]
        grid = _render_chain(segments, colors, rng)
        if np.array_equal(main(grid), _expected_output(lengths, colors)):
            return grid
    raise RuntimeError("failed to generate valid input")
