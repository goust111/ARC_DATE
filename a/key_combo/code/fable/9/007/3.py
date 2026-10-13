from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIRS_CLOCKWISE = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_INDEX = {d: i for i, d in enumerate(DIRS_CLOCKWISE)}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("left", "top", "right", "bottom")
COLORS = tuple(range(1, 10))
BG = 0
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _turn(direction: tuple[int, int], symbol: str) -> tuple[int, int]:
    idx = DIR_INDEX[direction]
    if symbol == "L":
        return DIRS_CLOCKWISE[(idx - 1) % 4]
    if symbol == "R":
        return DIRS_CLOCKWISE[(idx + 1) % 4]
    raise ValueError(f"bad turn symbol: {symbol}")


def _turn_symbol(prev_dir: tuple[int, int], next_dir: tuple[int, int]) -> str:
    delta = (DIR_INDEX[next_dir] - DIR_INDEX[prev_dir]) % 4
    if delta == 1:
        return "R"
    if delta == 3:
        return "L"
    raise AssertionError(f"non-turn or U-turn: {prev_dir} -> {next_dir}")


def _build_path(
    start: tuple[int, int],
    init_dir: tuple[int, int],
    turns: list[str],
    lengths: list[int],
) -> list[tuple[int, int]]:
    pts = [tuple(map(int, start))]
    direction = init_dir
    for i, length in enumerate(lengths):
        assert length >= 1
        for _ in range(int(length)):
            r, c = pts[-1]
            pts.append((r + direction[0], c + direction[1]))
        if i < len(turns):
            direction = _turn(direction, turns[i])
    return pts


def _dilate8(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _point_set_is_open_polyline(points: list[tuple[int, int]]) -> bool:
    pts = set(points)
    if len(pts) != len(points):
        return False
    endpoints = 0
    for r, c in pts:
        deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        if deg == 1:
            endpoints += 1
        elif deg != 2:
            return False
    return endpoints == 2


def _trace_polyline(line_cells: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    endpoints = []
    for r, c in line_cells:
        deg = sum((nr, nc) in line_cells for nr, nc in _neighbors4(r, c))
        assert deg in (1, 2), (r, c, deg)
        if deg == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, endpoints

    boundary_endpoints = [
        pt for pt in endpoints if pt[0] in (0, h - 1) or pt[1] in (0, w - 1)
    ]
    assert len(boundary_endpoints) == 1, boundary_endpoints
    start = boundary_endpoints[0]

    path = [start]
    prev = None
    cur = start
    while True:
        nxt = [
            (nr, nc)
            for nr, nc in _neighbors4(*cur)
            if (nr, nc) in line_cells and (nr, nc) != prev
        ]
        if not nxt:
            break
        assert len(nxt) == 1, (cur, nxt)
        prev, cur = cur, nxt[0]
        path.append(cur)
        assert len(path) <= len(line_cells)

    assert len(path) == len(line_cells)
    return path


def _extract_program(path: list[tuple[int, int]]) -> tuple[tuple[int, int], list[str], list[int]]:
    assert len(path) >= 2
    init_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    assert init_dir in DIR_INDEX

    turns: list[str] = []
    lengths: list[int] = []
    current_dir = init_dir
    run_len = 0
    for a, b in zip(path, path[1:]):
        step = (b[0] - a[0], b[1] - a[1])
        assert step in DIR_INDEX
        if step == current_dir:
            run_len += 1
            continue
        lengths.append(run_len)
        turns.append(_turn_symbol(current_dir, step))
        current_dir = step
        run_len = 1
    lengths.append(run_len)

    assert len(lengths) == len(turns) + 1
    return init_dir, turns, lengths


def _transform_side(side: str, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for r, c in points:
        if side == "left":
            out.append((r, c))
        elif side == "top":
            out.append((c, -r))
        elif side == "right":
            out.append((-r, -c))
        elif side == "bottom":
            out.append((-c, r))
        else:
            raise ValueError(f"bad side: {side}")
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize_set(points: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    r0 = min(rs)
    c0 = min(cs)
    return frozenset((r - r0, c - c0) for r, c in points)


def _congruent_under_d4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    def transforms(r: int, c: int) -> list[tuple[int, int]]:
        return [
            (r, c),
            (r, -c),
            (-r, c),
            (-r, -c),
            (c, r),
            (c, -r),
            (-c, r),
            (-c, -r),
        ]

    target = _normalize_set(b)
    transformed = [[] for _ in range(8)]
    for r, c in a:
        vals = transforms(r, c)
        for i, val in enumerate(vals):
            transformed[i].append(val)
    return any(_normalize_set(set(t)) == target for t in transformed)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert len(colors) == 2, colors

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    seed_candidates = [color for color, count in counts.items() if count == 1]
    assert len(seed_candidates) == 1, counts
    seed_color = seed_candidates[0]
    line_color = next(color for color in colors if color != seed_color)

    seed = tuple(map(int, np.argwhere(g == seed_color)[0]))
    sr, sc = seed
    for nr, nc in _neighbors8(sr, sc):
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
            assert int(g[nr, nc]) == bg, (seed, nr, nc)

    line_mask = g == line_color
    comps = _component_cells(line_mask)
    assert len(comps) == 1, len(comps)
    line_cells = set(comps[0])
    path = _trace_polyline(line_cells, g.shape)
    init_dir, turns, lengths = _extract_program(path)

    replay_path = _build_path(seed, init_dir, turns, list(reversed(lengths)))
    replay_cells = set(replay_path)
    assert len(replay_cells) == len(replay_path)
    assert _point_set_is_open_polyline(replay_path)
    for r, c in replay_cells:
        assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
        assert (r, c) not in line_cells

    return {
        "bg": bg,
        "seed": seed,
        "seed_color": seed_color,
        "line_color": line_color,
        "line_cells": line_cells,
        "path": path,
        "init_dir": init_dir,
        "turns": turns,
        "lengths": lengths,
        "replay_path": replay_path,
        "replay_cells": replay_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replay a second polyline at the seed by keeping turn order but reversing segment lengths.

    Input structure:
    - The background is uniform, with one one-pixel-wide monochrome open orthogonal polyline and one isolated seed pixel in a different color.
    - Exactly one endpoint of the original polyline lies on the grid boundary; that boundary endpoint is the reading start.
    - The seed does not touch the original polyline.

    Rule:
    - Trace the original polyline from its unique boundary endpoint to the other endpoint.
    - Record its initial direction, its left/right turn sequence in traversal order, and its straight segment lengths.
    - Starting from the seed, draw a new polyline in the seed color with the same initial direction and the same turn sequence, but use the segment lengths in reverse order.
    - Keep the original polyline unchanged.

    Key insight:
    - The visible line is a program: turns and lengths are separate pieces of information, and only the length sequence is reversed during replay.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    out = np.array(grid, dtype=int, copy=True)
    for r, c in info["replay_path"]:
        out[r, c] = info["seed_color"]
    return out


def _random_turns(n_segments: int) -> list[str]:
    mode = random.choice(("mixed", "same", "alternating"))
    if mode == "same":
        turn = random.choice(("L", "R"))
        return [turn] * (n_segments - 1)
    if mode == "alternating":
        first = random.choice(("L", "R"))
        return [
            first if i % 2 == 0 else ("L" if first == "R" else "R")
            for i in range(n_segments - 1)
        ]
    while True:
        turns = [random.choice(("L", "R")) for _ in range(n_segments - 1)]
        if "L" in turns and "R" in turns:
            return turns


def _local_program() -> tuple[list[str], list[int]]:
    n_segments = random.randint(4, 6)
    turns = _random_turns(n_segments)
    lengths = [random.randint(2, 6) for _ in range(n_segments)]
    mismatch_pairs = sum(
        lengths[i] != lengths[-1 - i] for i in range(n_segments // 2)
    )
    if mismatch_pairs < 2:
        raise ValueError("degenerate palindromic lengths")
    return turns, lengths


def _valid_local_path(points: list[tuple[int, int]]) -> bool:
    if not _point_set_is_open_polyline(points):
        return False
    if any(c < 0 for _, c in points):
        return False
    return sum(c == 0 for _, c in points) == 1


def _place_original(
    side: str,
    points: list[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = _bbox(points)
    if side in ("left", "right"):
        lo = 1 - r0
        hi = height - 2 - r1
        if lo > hi:
            return None
        shift_r = random.randint(lo, hi)
        shift_c = 0 if side == "left" else width - 1
    else:
        lo = 1 - c0
        hi = width - 2 - c1
        if lo > hi:
            return None
        shift_c = random.randint(lo, hi)
        shift_r = 0 if side == "top" else height - 1
    return [(r + shift_r, c + shift_c) for r, c in points]


def _candidate_seed_starts(
    replay_points: list[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(replay_points)
    row_lo = 1 - r0
    row_hi = height - 2 - r1
    col_lo = 1 - c0
    col_hi = width - 2 - c1
    if row_lo > row_hi or col_lo > col_hi:
        return []
    return [
        (sr, sc)
        for sr in range(row_lo, row_hi + 1)
        for sc in range(col_lo, col_hi + 1)
    ]


def _render_input(
    height: int,
    width: int,
    original: list[tuple[int, int]],
    seed: tuple[int, int],
    line_color: int,
    seed_color: int,
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for r, c in original:
        grid[r, c] = line_color
    grid[seed] = seed_color
    return grid


def _attempt_generate() -> np.ndarray | None:
    try:
        turns, lengths = _local_program()
    except ValueError:
        return None

    local_original = _build_path((0, 0), (0, 1), turns, lengths)
    local_replay = _build_path((0, 0), (0, 1), turns, list(reversed(lengths)))
    if not (_valid_local_path(local_original) and _valid_local_path(local_replay)):
        return None
    if _congruent_under_d4(set(local_original), set(local_replay)):
        return None

    side = random.choice(SIDES)
    original_side = _transform_side(side, local_original)
    replay_side = _transform_side(side, local_replay)

    ob = _bbox(original_side)
    rb = _bbox(replay_side)
    min_h = max(ob[2] - ob[0] + 3, rb[2] - rb[0] + 4)
    min_w = max(ob[3] - ob[1] + 3, rb[3] - rb[1] + 4)
    if min_h > 30 or min_w > 30:
        return None
    height = random.randint(min_h, min(30, min_h + 6))
    width = random.randint(min_w, min(30, min_w + 6))

    original = _place_original(side, original_side, height, width)
    if original is None:
        return None
    original_cells = set(original)
    forbidden = _dilate8(original_cells)

    starts = _candidate_seed_starts(replay_side, height, width)
    random.shuffle(starts)
    replay = None
    seed = None
    for sr, sc in starts:
        candidate = [(r + sr, c + sc) for r, c in replay_side]
        candidate_cells = set(candidate)
        if candidate_cells & original_cells:
            continue
        if candidate_cells & forbidden:
            continue
        replay = candidate
        seed = (sr, sc)
        break
    if replay is None or seed is None:
        return None

    line_color, seed_color = random.sample(COLORS, 2)
    grid = _render_input(height, width, original, seed, line_color, seed_color)
    expected = grid.copy()
    for r, c in replay:
        expected[r, c] = seed_color

    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return None
    if info["lengths"] == info["lengths"][::-1]:
        return None
    if len(info["turns"]) < 3:
        return None
    if _congruent_under_d4(info["line_cells"], info["replay_cells"]):
        return None
    if np.array_equal(grid, expected):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
