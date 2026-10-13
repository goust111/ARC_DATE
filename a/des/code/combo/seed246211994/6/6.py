from __future__ import annotations

import random
from collections import Counter

import numpy as np


_BG = 0
_LATTICE = 8
_TILE = 5
_INTERIOR_OFFSETS = (
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
)
_INTERIOR = set(_INTERIOR_OFFSETS)

_TILE_ROWS_RANGE = (2, 6)
_TILE_COLS_RANGE = (2, 6)
_PADDING_RANGE = (0, 4)
_MAX_DISTINCT_MARKER_COLORS = 4
_LATTICE_MARKER_PROB = 0.4
_NON_LATTICE_MARKERS = (1, 2, 3, 4, 5, 6, 7, 9)
_GEN_RETRIES = 300


def _window_lattice_color(grid: np.ndarray, r: int, c: int, bg: int) -> int | None:
    lattice = None
    for dr in range(_TILE):
        for dc in range(_TILE):
            if (dr, dc) == (0, 0):
                continue
            value = int(grid[r + dr, c + dc])
            if (dr, dc) in _INTERIOR:
                if value != bg:
                    return None
                continue
            if value == bg:
                return None
            if lattice is None:
                lattice = value
            elif value != lattice:
                return None
    return lattice


def _infer_lattice_color(grid: np.ndarray, bg: int) -> int:
    counts: Counter[int] = Counter()
    h, w = grid.shape
    for r in range(h - _TILE + 1):
        for c in range(w - _TILE + 1):
            lattice = _window_lattice_color(grid, r, c, bg)
            if lattice is not None:
                counts[int(lattice)] += 1

    assert counts, "expected at least one complete lattice tile"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "lattice color must be structurally unique"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clear each corner marker and fill the adjacent down-right cavity with that marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _BG
    lattice = _infer_lattice_color(g, bg)
    h, w = g.shape

    anchors: list[tuple[int, int, int]] = []
    for r in range(h - _TILE + 1):
        for c in range(w - _TILE + 1):
            if _window_lattice_color(g, r, c, bg) == lattice and int(g[r, c]) != bg:
                anchors.append((r, c, int(g[r, c])))

    assert anchors, "expected at least one marked lattice corner"

    out = g.copy()
    for r, c, color in anchors:
        out[r, c] = bg
        for dr, dc in _INTERIOR_OFFSETS:
            out[r + dr, c + dc] = color
    return out


def _base_patch(tile_rows: int, tile_cols: int) -> np.ndarray:
    h = _TILE * tile_rows + 1
    w = _TILE * tile_cols + 1
    patch = np.full((h, w), _LATTICE, dtype=int)
    rr, cc = np.mgrid[:h, :w]
    dr = rr % _TILE
    dc = cc % _TILE

    empty_mask = (dr == 0) & (dc == 0)
    for off_r, off_c in _INTERIOR_OFFSETS:
        empty_mask |= (dr == off_r) & (dc == off_c)
    patch[empty_mask] = _BG
    return patch


def _choose_marker_colors(marker_count: int) -> list[int]:
    include_lattice = random.random() < _LATTICE_MARKER_PROB
    max_distinct = min(_MAX_DISTINCT_MARKER_COLORS, marker_count)

    if include_lattice:
        non_lattice_count = random.randint(0, min(len(_NON_LATTICE_MARKERS), max_distinct - 1))
    else:
        non_lattice_count = random.randint(1, min(len(_NON_LATTICE_MARKERS), max_distinct))

    palette = random.sample(_NON_LATTICE_MARKERS, non_lattice_count)
    if include_lattice:
        palette.append(_LATTICE)

    assignments = palette[:]
    while len(assignments) < marker_count:
        assignments.append(random.choice(palette))
    random.shuffle(assignments)
    return assignments


def _embed_patch(patch: np.ndarray, top: int, left: int, canvas_shape: tuple[int, int]) -> np.ndarray:
    canvas = np.full(canvas_shape, _BG, dtype=int)
    h, w = patch.shape
    canvas[top : top + h, left : left + w] = patch
    return canvas


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    tile_rows = random.randint(*_TILE_ROWS_RANGE)
    tile_cols = random.randint(*_TILE_COLS_RANGE)
    total_tiles = tile_rows * tile_cols

    patch = _base_patch(tile_rows, tile_cols)
    pad_top = random.randint(*_PADDING_RANGE)
    pad_left = random.randint(*_PADDING_RANGE)
    pad_bottom = random.randint(*_PADDING_RANGE)
    pad_right = random.randint(*_PADDING_RANGE)
    canvas_shape = (
        patch.shape[0] + pad_top + pad_bottom,
        patch.shape[1] + pad_left + pad_right,
    )

    max_markers = max(1, min(total_tiles, (2 * total_tiles + 2) // 3))
    marker_count = random.randint(1, max_markers)
    tiles = [(tr, tc) for tr in range(tile_rows) for tc in range(tile_cols)]
    random.shuffle(tiles)
    marked_tiles = tiles[:marker_count]
    marker_colors = _choose_marker_colors(marker_count)

    solved_patch = patch.copy()
    for (tr, tc), color in zip(marked_tiles, marker_colors):
        r0 = tr * _TILE
        c0 = tc * _TILE
        for dr, dc in _INTERIOR_OFFSETS:
            solved_patch[r0 + dr, c0 + dc] = color

    input_patch = solved_patch.copy()
    for (tr, tc), color in zip(marked_tiles, marker_colors):
        r0 = tr * _TILE
        c0 = tc * _TILE
        for dr, dc in _INTERIOR_OFFSETS:
            input_patch[r0 + dr, c0 + dc] = _BG
        input_patch[r0, c0] = color

    inp = _embed_patch(input_patch, pad_top, pad_left, canvas_shape)
    out = _embed_patch(solved_patch, pad_top, pad_left, canvas_shape)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            inp, _out = attempt
            return inp
    raise AssertionError("failed to generate a valid input")
