from __future__ import annotations

import random

import numpy as np


# --------------- shared helpers ---------------


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    row = grid[0]
    mapping: dict[int, tuple[int, int]] = {}
    c = 0
    while c < row.shape[0]:
        if int(row[c]) == bg:
            c += 1
            continue
        start = c
        while c < row.shape[0] and int(row[c]) != bg:
            c += 1
        run = row[start:c]
        assert len(run) == 3, "legend blocks must be 1x3"
        key = int(run[0])
        assert key not in mapping, "legend keys must be unique"
        mapping[key] = (int(run[1]), int(run[2]))
    assert mapping, "expected at least one legend block"
    return mapping


def _frame_bbox_if_any(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    coord_set = set(coords)
    if not border.issubset(coord_set):
        return None

    extras = coord_set - border
    if len(extras) > 1:
        return None
    if any(not (r0 < r < r1 and c0 < c < c1) for r, c in extras):
        return None

    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            bbox = _frame_bbox_if_any(coords)
            if bbox is not None and bbox[0] > 0:
                frames.append(bbox)
    frames.sort()
    assert frames, "expected at least one frame"
    return frames


def _pattern_row(pattern: tuple[int, int], width: int) -> np.ndarray:
    row = np.empty(width, dtype=int)
    row[0::2] = pattern[0]
    row[1::2] = pattern[1]
    return row


# --------------- solver ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top legend as key -> 1x2 pattern, then fill each framed interior
    by horizontally tiling the matched pattern on every row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 7

    bg = _background_color(g)
    mapping = _legend_mapping(g, bg)
    frames = _find_frames(g, bg)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        markers = np.argwhere(interior != bg)
        assert len(markers) == 1, "each frame must contain exactly one marker"

        mr, mc = map(int, markers[0])
        key = int(interior[mr, mc])
        assert key in mapping, "marker color must exist in the legend"

        fill_row = _pattern_row(mapping[key], c1 - c0 - 1)
        tiled = np.tile(fill_row, (r1 - r0 - 1, 1))
        out[r0 + 1 : r1, c0 + 1 : c1] = tiled

    return out


# --------------- generator ---------------

_GEN_RETRIES = 300
_COLOR_IDS = tuple(range(10))
_LEGEND_COUNTS = (2, 3, 4)
_FRAME_COUNT_RANGE = (2, 6)
_INTERIOR_H_RANGE = (2, 6)
_INTERIOR_W_RANGE = (2, 7)


def _ordered_pattern(rng: random.Random, bg: int, border: int, key: int) -> tuple[int, int]:
    colors = [c for c in _COLOR_IDS if c not in {bg, border, key}]
    a, b = rng.sample(colors, 2)
    return int(a), int(b)


def _frame_keys(rng: random.Random, keys: list[int], frame_count: int) -> list[int]:
    if frame_count <= len(keys):
        chosen = rng.sample(keys, frame_count)
    else:
        chosen = keys.copy()
        chosen.extend(rng.choice(keys) for _ in range(frame_count - len(keys)))
        rng.shuffle(chosen)
    assert len(set(chosen)) >= min(2, len(keys))
    return chosen


def _place_frames(
    rng: random.Random, height: int, width: int, frame_keys: list[int]
) -> list[tuple[int, int, int, int, int, int, int]] | None:
    top_reserved = 2
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int, int, int, int, int, int]] = []

    for key in frame_keys:
        placed = False
        for _ in range(200):
            inner_h = rng.randint(*_INTERIOR_H_RANGE)
            inner_w = rng.randint(*_INTERIOR_W_RANGE)
            frame_h = inner_h + 2
            frame_w = inner_w + 2
            max_r0 = height - frame_h - 1
            max_c0 = width - frame_w - 1
            if max_r0 < top_reserved or max_c0 < 1:
                return None

            r0 = rng.randint(top_reserved, max_r0)
            c0 = rng.randint(1, max_c0)
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1

            if occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue

            marker_r = rng.randint(r0 + 1, r1 - 1)
            marker_c = rng.randint(c0 + 1, c1 - 1)
            placements.append((r0, c0, r1, c1, key, marker_r, marker_c))
            occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            placed = True
            break

        if not placed:
            return None

    return placements


def _validate_generated(
    grid: np.ndarray,
    bg: int,
    border: int,
    mapping: dict[int, tuple[int, int]],
    placements: list[tuple[int, int, int, int, int, int, int]],
) -> None:
    assert _background_color(grid) == bg
    assert _legend_mapping(grid, bg) == mapping
    assert len(_find_frames(grid, bg)) == len(placements)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid)

    for r0, c0, r1, c1, key, marker_r, marker_c in placements:
        assert int(grid[marker_r, marker_c]) == key
        expected_row = _pattern_row(mapping[key], c1 - c0 - 1)
        expected_fill = np.tile(expected_row, (r1 - r0 - 1, 1))
        assert np.array_equal(out[r0 + 1 : r1, c0 + 1 : c1], expected_fill)
        assert np.all(out[r0, c0 : c1 + 1] == border)
        assert np.all(out[r1, c0 : c1 + 1] == border)
        assert np.all(out[r0 : r1 + 1, c0] == border)
        assert np.all(out[r0 : r1 + 1, c1] == border)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_count = rng.choice(_LEGEND_COUNTS)
    frame_count = rng.randint(*_FRAME_COUNT_RANGE)

    bg = rng.choice(_COLOR_IDS)
    border_choices = [c for c in _COLOR_IDS if c != bg]
    border = rng.choice(border_choices)

    key_choices = [c for c in _COLOR_IDS if c not in {bg, border}]
    if len(key_choices) < legend_count:
        return None
    keys = rng.sample(key_choices, legend_count)

    mapping: dict[int, tuple[int, int]] = {}
    used_patterns: set[tuple[int, int]] = set()
    for key in keys:
        for _ in range(50):
            pattern = _ordered_pattern(rng, bg, border, key)
            if pattern not in used_patterns:
                mapping[key] = pattern
                used_patterns.add(pattern)
                break
        else:
            return None

    frame_marker_keys = _frame_keys(rng, keys, frame_count)

    legend_width = 4 * legend_count - 1
    min_height = max(12, 8 + 2 * frame_count)
    max_height = max(min_height, min_height + 6)
    min_width = max(legend_width + 1, 14 + frame_count)
    max_width = max(min_width, min_width + 8)
    height = rng.randint(min_height, max_height)
    width = rng.randint(min_width, max_width)

    placements = _place_frames(rng, height, width, frame_marker_keys)
    if placements is None:
        return None

    grid = np.full((height, width), bg, dtype=int)

    legend_order = keys[:]
    rng.shuffle(legend_order)
    start_col = rng.randint(0, width - legend_width)
    col = start_col
    for idx, key in enumerate(legend_order):
        p0, p1 = mapping[key]
        grid[0, col] = key
        grid[0, col + 1] = p0
        grid[0, col + 2] = p1
        col += 3
        if idx != len(legend_order) - 1:
            col += 1

    for r0, c0, r1, c1, key, marker_r, marker_c in placements:
        grid[r0, c0 : c1 + 1] = border
        grid[r1, c0 : c1 + 1] = border
        grid[r0 : r1 + 1, c0] = border
        grid[r0 : r1 + 1, c1] = border
        grid[marker_r, marker_c] = key

    try:
        _validate_generated(grid, bg, border, mapping, placements)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
