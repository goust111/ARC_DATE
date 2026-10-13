from __future__ import annotations

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = np.array(list(range(10)), dtype=int)
RNG = np.random.default_rng()
GENERATION_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors(cell: tuple[int, int], height: int, width: int):
    r, c = cell
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield (nr, nc)


def _ordered_path(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    foreground = {tuple(map(int, rc)) for rc in np.argwhere(grid != bg)}
    assert foreground, "expected a non-empty foreground path"

    start = next(iter(foreground))
    seen = {start}
    stack = [start]
    while stack:
        cell = stack.pop()
        for nxt in _neighbors(cell, h, w):
            if nxt in foreground and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    assert seen == foreground, "foreground must be one connected path"

    endpoints: list[tuple[int, int]] = []
    for cell in foreground:
        degree = sum(nxt in foreground for nxt in _neighbors(cell, h, w))
        assert degree in (1, 2), "path must be single-pixel wide and unbranched"
        if degree == 1:
            endpoints.append(cell)
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    current = min(endpoints)
    order = [current]
    previous: tuple[int, int] | None = None
    while True:
        next_cells = [
            nxt
            for nxt in _neighbors(current, h, w)
            if nxt in foreground and nxt != previous
        ]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unambiguous"
        previous, current = current, next_cells[0]
        order.append(current)

    assert len(order) == len(foreground), "path traversal must visit every cell once"
    return order


def _segments_from_path(
    grid: np.ndarray, path: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    color = int(grid[path[0]])
    length = 0
    for cell in path:
        cell_color = int(grid[cell])
        if cell_color == color:
            length += 1
            continue
        segments.append((color, length))
        color = cell_color
        length = 1
    segments.append((color, length))
    assert len(segments) >= 2, "expected multiple monochrome path segments"
    return segments


def _paint_segments(
    shape: tuple[int, int],
    bg: int,
    path: list[tuple[int, int]],
    segments: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    index = 0
    for color, length in segments:
        for _ in range(length):
            r, c = path[index]
            out[r, c] = color
            index += 1
    assert index == len(path)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Sort same-color path segments by length and repaint the unchanged path."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    path = _ordered_path(g, bg)
    segments = _segments_from_path(g, path)
    sorted_segments = sorted(segments, key=lambda segment: segment[1])
    return _paint_segments(g.shape, bg, path, sorted_segments)


def _continuation_count(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    height: int,
    width: int,
) -> int:
    extended = occupied | {cell}
    count = 0
    for nxt in _neighbors(cell, height, width):
        if nxt in extended:
            continue
        occupied_neighbors = sum(
            other in extended for other in _neighbors(nxt, height, width)
        )
        if occupied_neighbors == 1:
            count += 1
    return count


def _has_turn(path: list[tuple[int, int]]) -> bool:
    if len(path) < 3:
        return False
    directions = {
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    }
    return len(directions) > 1


def _generate_local_path(length: int) -> list[tuple[int, int]] | None:
    for _ in range(120):
        local_h = int(RNG.integers(6, 14))
        local_w = int(RNG.integers(6, 14))
        if local_h * local_w < length * 2:
            continue

        start = (int(RNG.integers(local_h)), int(RNG.integers(local_w)))
        path = [start]
        occupied = {start}

        while len(path) < length:
            current = path[-1]
            previous = path[-2] if len(path) >= 2 else None
            candidates: list[tuple[tuple[int, int], int, bool]] = []

            for nxt in _neighbors(current, local_h, local_w):
                if nxt in occupied:
                    continue
                occupied_neighbors = sum(
                    other in occupied for other in _neighbors(nxt, local_h, local_w)
                )
                if occupied_neighbors != 1:
                    continue
                is_straight = False
                if previous is not None:
                    dr1 = current[0] - previous[0]
                    dc1 = current[1] - previous[1]
                    dr2 = nxt[0] - current[0]
                    dc2 = nxt[1] - current[1]
                    is_straight = (dr1, dc1) == (dr2, dc2)
                score = _continuation_count(nxt, occupied, local_h, local_w)
                candidates.append((nxt, score, is_straight))

            if not candidates:
                break

            best_score = max(score for _, score, _ in candidates)
            filtered = [item for item in candidates if item[1] == best_score]
            turning = [item for item in filtered if not item[2]]
            if turning and RNG.random() < 0.65:
                filtered = turning
            choice = filtered[int(RNG.integers(len(filtered)))]
            path.append(choice[0])
            occupied.add(choice[0])

        if len(path) != length:
            continue
        if not _has_turn(path) and length >= 8 and RNG.random() < 0.8:
            continue
        return path
    return None


def _canonical_shifted_path(
    path: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    min_r = min(r for r, _ in path)
    min_c = min(c for _, c in path)
    normalized = [(r - min_r, c - min_c) for r, c in path]
    if normalized[0] > normalized[-1]:
        normalized = list(reversed(normalized))

    bbox_h = max(r for r, _ in normalized) + 1
    bbox_w = max(c for _, c in normalized) + 1
    top = int(RNG.integers(1, 5))
    bottom = int(RNG.integers(1, 5))
    left = int(RNG.integers(1, 5))
    right = int(RNG.integers(1, 5))
    shifted = [(r + top, c + left) for r, c in normalized]
    return shifted, (bbox_h + top + bottom, bbox_w + left + right)


def _sample_lengths() -> list[int] | None:
    for _ in range(80):
        count = int(RNG.integers(4, 8))
        want_ties = bool(RNG.integers(0, 2))
        if want_ties:
            lengths = [int(v) for v in RNG.integers(1, 7, size=count)]
            if len(set(lengths)) == count or len(set(lengths)) == 1:
                continue
        else:
            lengths = [
                int(v)
                for v in RNG.choice(np.arange(1, 10), size=count, replace=False)
            ]
        if not (12 <= sum(lengths) <= 34):
            continue
        if lengths == sorted(lengths):
            perm = RNG.permutation(count)
            lengths = [lengths[int(i)] for i in perm]
        if lengths == sorted(lengths):
            continue
        return lengths
    return None


def _sample_segment_colors(count: int, bg: int) -> list[int]:
    palette = [color for color in COLOR_POOL.tolist() if color != bg]
    return [int(v) for v in RNG.choice(palette, size=count, replace=False)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(RNG.choice(COLOR_POOL))
    lengths = _sample_lengths()
    if lengths is None:
        return None

    path = _generate_local_path(sum(lengths))
    if path is None:
        return None
    path, shape = _canonical_shifted_path(path)

    colors = _sample_segment_colors(len(lengths), bg)
    segments = list(zip(colors, lengths))
    inp = _paint_segments(shape, bg, path, segments)
    expected = _paint_segments(shape, bg, path, sorted(segments, key=lambda item: item[1]))

    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input")
