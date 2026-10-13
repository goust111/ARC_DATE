from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_SIZE = 7
MAX_SIZE = 12
MIN_SEGMENTS = 2
MAX_SEGMENTS = 4
MIN_SEG_LEN = 2
MAX_SEG_LEN = 5
MIN_SELECTED = 9
MAX_SELECTED = 18
MAX_ATTEMPTS = 500


def _ring_coords(size: int, layer: int) -> list[tuple[int, int]]:
    r0 = c0 = layer
    r1 = c1 = size - 1 - layer
    if r0 > r1:
        return []
    if r0 == r1:
        return [(r0, c0)]

    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _spiral_coords(size: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for layer in range((size + 1) // 2):
        coords.extend(_ring_coords(size, layer))
    return coords


def _counterclockwise_ring_coords(size: int, layer: int) -> list[tuple[int, int]]:
    r0 = c0 = layer
    r1 = c1 = size - 1 - layer
    if r0 > r1:
        return []
    if r0 == r1:
        return [(r0, c0)]

    coords: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        coords.append((r, c0))
    for c in range(c0 + 1, c1 + 1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0 - 1, -1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0, -1):
        coords.append((r0, c))
    return coords


def _counterclockwise_spiral_coords(size: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for layer in range((size + 1) // 2):
        coords.extend(_counterclockwise_ring_coords(size, layer))
    return coords


def _row_major_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [(r, c) for r in range(h) for c in range(w)]


def _col_major_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [(r, c) for c in range(w) for r in range(h)]


def _collect_non_black(grid: np.ndarray, order: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[int]]:
    coords = [(r, c) for r, c in order if int(grid[r, c]) != BLACK]
    values = [int(grid[r, c]) for r, c in coords]
    return coords, values


def _reverse_segments(values: list[int]) -> list[int]:
    out: list[int] = []
    segment: list[int] = []
    for value in values:
        if value == GREY:
            out.extend(reversed(segment))
            out.append(GREY)
            segment.clear()
        else:
            segment.append(value)
    out.extend(reversed(segment))
    return out


def _apply_with_order(grid: np.ndarray, order: list[tuple[int, int]]) -> np.ndarray:
    coords, values = _collect_non_black(grid, order)
    transformed = _reverse_segments(values)
    out = np.asarray(grid, dtype=int).copy()
    for (r, c), value in zip(coords, transformed):
        out[r, c] = value
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse each grey-delimited color segment inside the hidden clockwise spiral reading order.

    Input structure:
    - The grid is a square black canvas with scattered non-black pixels.
    - Reading every cell from the outer ring inward in clockwise spiral order yields a hidden 1D sequence when black cells are skipped.
    - Grey pixels inside that sequence are separators between contiguous non-grey segments.

    Rule:
    - Traverse the whole grid in clockwise spiral order and record only the non-black colors.
    - Split that recorded sequence at grey separators and reverse each non-grey segment while keeping every grey separator at the same sequence index.
    - Write the transformed non-black sequence back onto the same non-black cells in the same clockwise spiral order.
    - Preserve the grid size and preserve which cells are non-black.

    Key insight:
    - The true object is the 1D sequence induced by the spiral scan, not any 2D row or column pattern; grey acts as a sequence delimiter rather than a geometric marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w, "expected a square grid"
    return _apply_with_order(g, _spiral_coords(h))


def _segment_lengths() -> list[int]:
    for _ in range(MAX_ATTEMPTS):
        count = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
        lengths = [random.randint(MIN_SEG_LEN, MAX_SEG_LEN) for _ in range(count)]
        total = sum(lengths) + count - 1
        if MIN_SELECTED <= total <= MAX_SELECTED and max(lengths) >= 3:
            return lengths
    raise RuntimeError("failed to sample segment lengths")


def _partition_total(total: int, caps: list[int], minimum: int = 2) -> list[int] | None:
    if total < minimum * len(caps) or sum(caps) < total:
        return None
    parts = [minimum] * len(caps)
    remaining = total - sum(parts)
    while remaining:
        choices = [i for i, cap in enumerate(caps) if parts[i] < cap]
        if not choices:
            return None
        idx = random.choice(choices)
        parts[idx] += 1
        remaining -= 1
    random.shuffle(parts)
    return parts


def _side_label(size: int, layer: int, coord: tuple[int, int]) -> str:
    r, c = coord
    r0 = c0 = layer
    r1 = c1 = size - 1 - layer
    if r == r0 and c < c1:
        return "top"
    if c == c1 and r < r1:
        return "right"
    if r == r1 and c > c0:
        return "bottom"
    return "left"


def _crosses_corner(size: int, layer: int, coords: list[tuple[int, int]]) -> bool:
    return len({_side_label(size, layer, coord) for coord in coords}) >= 2


def _choose_window(size: int, layer: int, length: int, force_corner: bool) -> list[tuple[int, int]] | None:
    path = _ring_coords(size, layer)
    if len(path) < length:
        return None

    candidates = [path[start : start + length] for start in range(len(path) - length + 1)]
    corner_candidates = [window for window in candidates if _crosses_corner(size, layer, window)]

    if force_corner and corner_candidates:
        return random.choice(corner_candidates)
    if corner_candidates and random.random() < 0.7:
        return random.choice(corner_candidates)
    return random.choice(candidates)


def _build_token_sequence(segment_lengths: list[int]) -> list[int]:
    tokens: list[int] = []
    non_grey_used: set[int] = set()
    for index, length in enumerate(segment_lengths):
        for _ in range(MAX_ATTEMPTS):
            segment = [random.choice(PALETTE) for _ in range(length)]
            if segment == list(reversed(segment)):
                continue
            if len(set(segment)) == 1:
                continue
            break
        else:
            raise RuntimeError("failed to sample a non-degenerate segment")
        tokens.extend(segment)
        non_grey_used.update(segment)
        if index != len(segment_lengths) - 1:
            tokens.append(GREY)

    if len(non_grey_used) < 2:
        return _build_token_sequence(segment_lengths)
    if tokens == _reverse_segments(tokens):
        return _build_token_sequence(segment_lengths)
    return tokens


def _ring_index(size: int, coord: tuple[int, int]) -> int:
    r, c = coord
    return min(r, c, size - 1 - r, size - 1 - c)


def _expected_from_tokens(size: int, coords: list[tuple[int, int]], tokens: list[int]) -> np.ndarray:
    grid = np.full((size, size), BLACK, dtype=int)
    for (r, c), value in zip(coords, tokens):
        grid[r, c] = value
    return grid


def _validate_instance(grid: np.ndarray, windows: list[list[tuple[int, int]]], tokens: list[int]) -> bool:
    size = int(grid.shape[0])
    spiral = _spiral_coords(size)
    coords, values = _collect_non_black(grid, spiral)
    if coords != [coord for window in windows for coord in window]:
        return False
    if values != tokens:
        return False

    separators = [i for i, value in enumerate(values) if value == GREY]
    if not separators or separators[0] == 0 or separators[-1] == len(values) - 1:
        return False
    if any(b - a == 1 for a, b in zip(separators, separators[1:])):
        return False

    segments = [segment.tolist() for segment in np.split(np.array(values, dtype=int), separators)]
    normalized: list[list[int]] = []
    for idx, segment in enumerate(segments):
        if idx:
            segment = segment[1:]
        if len(segment) < 2 or segment == list(reversed(segment)):
            return False
        normalized.append(segment)
    if len(normalized) < 2:
        return False

    if len({_ring_index(size, coord) for coord in coords}) < 2:
        return False
    if not any(_crosses_corner(size, _ring_index(size, window[0]), window) for window in windows):
        return False

    out = main(grid)
    expected_values = _reverse_segments(values)
    expected = _expected_from_tokens(size, coords, expected_values)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, grid):
        return False
    if not np.array_equal((out != BLACK), (grid != BLACK)):
        return False

    grey_mask = grid == GREY
    if not np.array_equal(grey_mask, out == GREY):
        return False

    wrong_orders = [
        _row_major_coords(grid.shape),
        _col_major_coords(grid.shape),
        _counterclockwise_spiral_coords(size),
    ]
    for order in wrong_orders:
        if np.array_equal(_apply_with_order(grid, order), out):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(MIN_SIZE, MAX_SIZE)
    layers = [_ring_coords(size, layer) for layer in range((size + 1) // 2)]
    eligible_layers = [layer for layer, path in enumerate(layers) if len(path) >= 3]
    if len(eligible_layers) < 2:
        return None

    segment_lengths = _segment_lengths()
    total_selected = sum(segment_lengths) + len(segment_lengths) - 1

    max_ring_count = min(4, len(eligible_layers), total_selected // 2)
    if max_ring_count < 2:
        return None
    ring_count = random.randint(2, max_ring_count)

    deeper_layers = eligible_layers[1:]
    if len(deeper_layers) < ring_count - 1:
        return None
    chosen_layers = [0] + sorted(random.sample(deeper_layers, ring_count - 1))
    caps = [min(7, len(layers[layer])) for layer in chosen_layers]
    window_lengths = _partition_total(total_selected, caps, minimum=2)
    if window_lengths is None:
        return None

    windows: list[list[tuple[int, int]]] = []
    has_corner_window = False
    for layer, length in zip(chosen_layers, window_lengths):
        window = _choose_window(size, layer, length, force_corner=not has_corner_window)
        if window is None:
            return None
        has_corner_window = has_corner_window or _crosses_corner(size, layer, window)
        windows.append(window)

    coords = [coord for window in windows for coord in window]
    tokens = _build_token_sequence(segment_lengths)
    grid = _expected_from_tokens(size, coords, tokens)
    if not _validate_instance(grid, windows, tokens):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-segment input")
