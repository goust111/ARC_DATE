from __future__ import annotations

import numpy as np


BACKGROUND = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

COARSE_DIM_MIN = 4
COARSE_DIM_MAX = 8
COARSE_NODE_MIN = 5
COARSE_NODE_MAX = 11
MARGIN_MAX = 3
SEGMENT_COUNT_MIN = 3
SEGMENT_COUNT_MAX = 7
MIN_SEGMENT_LEN = 2
MAX_GENERATION_ATTEMPTS = 500

_RNG = np.random.default_rng()


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _ordered_path(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert coords, "expected a non-empty path"

    occupied = set(coords)
    endpoints: list[tuple[int, int]] = []
    for r, c in coords:
        degree = sum((nr, nc) in occupied for nr, nc in _neighbors(r, c, h, w))
        assert degree <= 2, "path must be one-pixel-wide and branch-free"
        if degree == 1:
            endpoints.append((r, c))
        elif degree == 0:
            assert len(coords) == 1, "isolated pixels only allowed for singleton paths"

    if len(coords) == 1:
        return [coords[0]]

    # The puzzle says "from one end to the other"; use the upper-left endpoint
    # as the visible canonical start so the ordering is unique.
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    start = min(endpoints)

    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        next_cells = [nb for nb in _neighbors(cur[0], cur[1], h, w) if nb in occupied and nb != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        nxt = next_cells[0]
        ordered.append(nxt)
        prev, cur = cur, nxt

    assert len(ordered) == len(coords), "path must be a single connected chain"
    return ordered


def _turn_count(path_cells: list[tuple[int, int]]) -> int:
    if len(path_cells) <= 2:
        return 0

    directions = []
    for (r0, c0), (r1, c1) in zip(path_cells, path_cells[1:]):
        directions.append((r1 - r0, c1 - c0))
    return sum(d0 != d1 for d0, d1 in zip(directions, directions[1:]))


def _segment_runs(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    path = _ordered_path(grid != BACKGROUND)
    colors = [int(grid[r, c]) for r, c in path]

    segments: list[tuple[int, list[tuple[int, int]]]] = []
    start = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or colors[i] != colors[start]:
            segments.append((colors[start], path[start:i]))
            start = i

    assert segments, "expected at least one monochrome segment"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """Convert a multicolor path into bottom-aligned bars encoding per-segment turn counts."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g != BACKGROUND), "expected foreground path"

    segments = _segment_runs(g)
    heights = [_turn_count(cells) + 1 for _, cells in segments]
    out = np.full((max(heights), len(segments)), BACKGROUND, dtype=int)

    for col, ((color, _cells), height) in enumerate(zip(segments, heights)):
        out[-height:, col] = int(color)
    return out


def _candidate_coarse_steps(
    current: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr = current[0] + dr
        nc = current[1] + dc
        nxt = (nr, nc)
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if nxt in occupied:
            continue
        occupied_neighbors = [
            (ar, ac)
            for ar, ac in _neighbors(nr, nc, h, w)
            if (ar, ac) in occupied
        ]
        if set(occupied_neighbors) == {current}:
            candidates.append(nxt)
    return candidates


def _future_option_count(
    nxt: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> int:
    return len(_candidate_coarse_steps(nxt, occupied, h, w))


def _extend_coarse_path(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    target_len: int,
    rng: np.random.Generator,
) -> bool:
    if len(path) == target_len:
        return True

    current = path[-1]
    prev_dir = None
    if len(path) >= 2:
        prev = path[-2]
        prev_dir = (current[0] - prev[0], current[1] - prev[1])

    raw_candidates = _candidate_coarse_steps(current, occupied, h, w)
    if not raw_candidates:
        return False

    candidate_data: list[tuple[tuple[int, int], int, int]] = []
    for nxt in raw_candidates:
        occupied.add(nxt)
        future = _future_option_count(nxt, occupied, h, w)
        occupied.remove(nxt)
        step_dir = (nxt[0] - current[0], nxt[1] - current[1])
        turn_bonus = int(prev_dir is not None and step_dir != prev_dir)
        candidate_data.append((nxt, future, turn_bonus))

    perm = rng.permutation(len(candidate_data))
    candidate_data = [candidate_data[int(i)] for i in perm]
    candidate_data.sort(key=lambda item: (item[1], item[2]), reverse=True)

    for nxt, _future, _turn_bonus in candidate_data:
        path.append(nxt)
        occupied.add(nxt)
        if _extend_coarse_path(path, occupied, h, w, target_len, rng):
            return True
        occupied.remove(nxt)
        path.pop()

    return False


def _random_coarse_path(h: int, w: int, target_len: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(h) for c in range(w)]
    for idx in rng.permutation(len(starts)):
        start = starts[int(idx)]
        path = [start]
        occupied = {start}
        if _extend_coarse_path(path, occupied, h, w, target_len, rng):
            return path
    return None


def _expand_coarse_path(coarse_path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    expanded = [(2 * coarse_path[0][0], 2 * coarse_path[0][1])]
    for (r0, c0), (r1, c1) in zip(coarse_path, coarse_path[1:]):
        rr0 = 2 * r0
        cc0 = 2 * c0
        rr1 = 2 * r1
        cc1 = 2 * c1
        expanded.append(((rr0 + rr1) // 2, (cc0 + cc1) // 2))
        expanded.append((rr1, cc1))

    min_r = min(r for r, _ in expanded)
    min_c = min(c for _, c in expanded)
    return [(r - min_r, c - min_c) for r, c in expanded]


def _sample_partition(total: int, parts: int, min_part: int, rng: np.random.Generator) -> list[int]:
    assert total >= parts * min_part
    lengths = np.full(parts, min_part, dtype=int)
    for _ in range(total - parts * min_part):
        lengths[int(rng.integers(parts))] += 1
    return [int(x) for x in lengths]


def _sample_colors(n_segments: int, rng: np.random.Generator) -> list[int]:
    colors: list[int] = []
    used: list[int] = []
    for _ in range(n_segments):
        prev = colors[-1] if colors else None
        reuse = [c for c in used if c != prev]
        fresh = [c for c in COLOR_POOL if c not in used and c != prev]
        all_choices = [c for c in COLOR_POOL if c != prev]

        if reuse and (not fresh or rng.random() < 0.35):
            pool = reuse
        else:
            pool = fresh or all_choices

        color = int(pool[int(rng.integers(len(pool)))])
        colors.append(color)
        if color not in used:
            used.append(color)

    assert len(set(colors)) >= 2
    return colors


def _segment_turns_from_lengths(path: list[tuple[int, int]], lengths: list[int]) -> list[int]:
    turns: list[int] = []
    start = 0
    for length in lengths:
        turns.append(_turn_count(path[start : start + length]))
        start += length
    assert start == len(path)
    return turns


def _gen_attempt(rng: np.random.Generator, complex_mode: bool) -> np.ndarray | None:
    coarse_h = int(rng.integers(COARSE_DIM_MIN, COARSE_DIM_MAX))
    coarse_w = int(rng.integers(COARSE_DIM_MIN, COARSE_DIM_MAX))
    max_nodes = min(COARSE_NODE_MAX, max(COARSE_NODE_MIN, (2 * coarse_h * coarse_w) // 3))
    target_nodes = int(rng.integers(COARSE_NODE_MIN, max_nodes + 1))

    coarse_path = _random_coarse_path(coarse_h, coarse_w, target_nodes, rng)
    if coarse_path is None:
        return None

    path = _expand_coarse_path(coarse_path)
    total_turns = _turn_count(path)
    if total_turns < 2:
        return None

    n_segments_max = min(SEGMENT_COUNT_MAX, len(path) // MIN_SEGMENT_LEN)
    if n_segments_max < SEGMENT_COUNT_MIN:
        return None

    pivot = max(SEGMENT_COUNT_MIN, len(path) // 3)
    if complex_mode:
        n_segments_high = min(n_segments_max, pivot)
        n_segments = int(rng.integers(SEGMENT_COUNT_MIN, n_segments_high + 1))
    else:
        n_segments_low = min(n_segments_max, pivot)
        n_segments = int(rng.integers(n_segments_low, n_segments_max + 1))

    lengths = _sample_partition(len(path), n_segments, MIN_SEGMENT_LEN, rng)
    segment_turns = _segment_turns_from_lengths(path, lengths)
    if not any(turn == 0 for turn in segment_turns):
        return None
    if not any(turn > 0 for turn in segment_turns):
        return None
    if complex_mode:
        if max(segment_turns) < 2:
            return None
    else:
        if max(segment_turns) != 1:
            return None

    colors = _sample_colors(n_segments, rng)

    base_h = max(r for r, _ in path) + 1
    base_w = max(c for _, c in path) + 1
    top = int(rng.integers(MARGIN_MAX + 1))
    bottom = int(rng.integers(MARGIN_MAX + 1))
    left = int(rng.integers(MARGIN_MAX + 1))
    right = int(rng.integers(MARGIN_MAX + 1))

    grid = np.full((base_h + top + bottom, base_w + left + right), BACKGROUND, dtype=int)

    start = 0
    for color, length in zip(colors, lengths):
        for r, c in path[start : start + length]:
            grid[top + r, left + c] = int(color)
        start += length

    try:
        ordered = _ordered_path(grid != BACKGROUND)
        assert len(ordered) == len(path)
        segments = _segment_runs(grid)
        assert len(segments) == n_segments
        out = main(grid)
        assert out.shape[1] == n_segments
        assert out.shape[0] == max(segment_turns) + 1
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    complex_mode = bool(_RNG.integers(2))
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG, complex_mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
