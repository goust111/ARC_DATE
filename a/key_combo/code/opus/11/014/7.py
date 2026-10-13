import random
from itertools import combinations

import numpy as np


BACKGROUND = 0
COLORS = (1, 2, 3)
DIRECTIONS = {
    "h": (0, 1),
    "v": (1, 0),
    "d1": (1, 1),
    "d2": (1, -1),
}


def _line_step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert not (dr == 0 and dc == 0), "marker pair must contain two distinct cells"
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "marker pair must be horizontal, vertical, or diagonal"
    sr = 0 if dr == 0 else dr // abs(dr)
    sc = 0 if dc == 0 else dc // abs(dc)
    steps = max(abs(dr), abs(dc))
    return int(sr), int(sc), int(steps)


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc, steps = _line_step(a, b)
    return [(a[0] + i * sr, a[1] + i * sc) for i in range(steps + 1)]


def _max_steps(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    rows, cols = shape
    r, c = start
    dr, dc = direction
    limits = []
    if dr > 0:
        limits.append(rows - 1 - r)
    elif dr < 0:
        limits.append(r)
    if dc > 0:
        limits.append(cols - 1 - c)
    elif dc < 0:
        limits.append(c)
    return int(min(limits)) if limits else max(rows, cols)


def _max_symmetric_half(shape: tuple[int, int], center: tuple[int, int], direction: tuple[int, int]) -> int:
    dr, dc = direction
    return min(
        _max_steps(shape, center, (dr, dc)),
        _max_steps(shape, center, (-dr, -dc)),
    )


def _segment_from_center(
    center: tuple[int, int], direction: tuple[int, int], half: int, color: int
) -> dict[str, object]:
    dr, dc = direction
    start = (center[0] - half * dr, center[1] - half * dc)
    end = (center[0] + half * dr, center[1] + half * dc)
    cells = _line_cells(start, end)
    return {
        "color": int(color),
        "start": start,
        "end": end,
        "cells": cells,
        "length": len(cells),
    }


def _band_mask(shape: tuple[int, int], cells: list[tuple[int, int]]) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        r0 = max(0, r - 1)
        r1 = min(rows, r + 2)
        c0 = max(0, c - 1)
        c1 = min(cols, c + 2)
        mask[r0:r1, c0:c1] = True
    return mask


def _render_output(shape: tuple[int, int], segments: list[dict[str, object]], grid: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    best_length = np.full(shape, 10**9, dtype=int)
    for seg in segments:
        length = int(seg["length"])
        color = int(seg["color"])
        mask = _band_mask(shape, seg["cells"])
        replace = mask & (length < best_length)
        out[replace] = color
        best_length[replace] = length
    out[grid != BACKGROUND] = grid[grid != BACKGROUND]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair, dilate the line into a band, and let shorter bands win overlaps.

    Input structure:
    - The grid is a black background with exactly three colors, each appearing in exactly two marker cells.
    - Each same-color marker pair is aligned horizontally, vertically, or on a 45-degree diagonal.
    - The six markers are the only non-background input pixels.

    Rule:
    - For each color pair, draw the inclusive straight line segment between its two markers.
    - Dilate each segment by painting the clipped 3x3 neighborhood of every line cell with that segment's color.
    - If multiple dilated bands cover one cell, keep the color of the segment with the shortest line length.
    - Restore the original marker cells so they always keep their own colors.

    Key insight:
    - The overlap priority is encoded by geometry rather than color or draw order: shorter connecting segments overpower longer ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    assert len(colors) == 3, "expected exactly three colors"

    segments: list[dict[str, object]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each color must appear exactly twice"
        start = tuple(map(int, coords[0]))
        end = tuple(map(int, coords[1]))
        cells = _line_cells(start, end)
        segments.append(
            {
                "color": color,
                "start": start,
                "end": end,
                "cells": cells,
                "length": len(cells),
            }
        )

    return _render_output(g.shape, segments, g)


GEN_SIZE_RANGE = (13, 18)
GEN_LENGTH_TRIPLES = ((5, 7, 9), (5, 7, 11), (5, 9, 11), (7, 9, 11))
GEN_BACKBONE_RANKS = ("medium", "long")
GEN_MAX_ATTEMPTS = 600


def _marker_mask(shape: tuple[int, int], segments: list[dict[str, object]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for seg in segments:
        for cell in (seg["start"], seg["end"]):
            mask[cell] = True
    return mask


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _valid_markers(segments: list[dict[str, object]]) -> bool:
    markers = [tuple(seg["start"]) for seg in segments] + [tuple(seg["end"]) for seg in segments]
    if len(set(markers)) != 6:
        return False
    if min(_chebyshev(a, b) for a, b in combinations(markers, 2)) < 2:
        return False
    for i, seg_i in enumerate(segments):
        endpoints = {tuple(seg_i["start"]), tuple(seg_i["end"])}
        for j, seg_j in enumerate(segments):
            if i == j:
                continue
            cells_j = set(seg_j["cells"])
            if endpoints & cells_j:
                return False
    return True


def _pair_overlap_masks(shape: tuple[int, int], segments: list[dict[str, object]]) -> dict[tuple[int, int], np.ndarray]:
    marker_mask = _marker_mask(shape, segments)
    bands = [_band_mask(shape, seg["cells"]) for seg in segments]
    pair_masks: dict[tuple[int, int], np.ndarray] = {}
    for i, j in combinations(range(len(segments)), 2):
        overlap = bands[i] & bands[j] & ~marker_mask
        if np.any(overlap):
            pair_masks[(i, j)] = overlap
    return pair_masks


def _build_input(shape: tuple[int, int], segments: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for seg in segments:
        grid[seg["start"]] = int(seg["color"])
        grid[seg["end"]] = int(seg["color"])
    return grid


def _rank_to_color() -> dict[str, int]:
    assigned = random.sample(list(COLORS), len(COLORS))
    return dict(zip(("short", "medium", "long"), assigned))


def _choose_lengths(size: int) -> tuple[int, int, int]:
    candidates = [triple for triple in GEN_LENGTH_TRIPLES if triple[-1] <= size]
    return random.choice(candidates)


def _oracle_accepts(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    side = random.randint(*GEN_SIZE_RANGE)
    shape = (side, side)

    short_len, medium_len, long_len = _choose_lengths(side)
    half_by_rank = {
        "short": (short_len - 1) // 2,
        "medium": (medium_len - 1) // 2,
        "long": (long_len - 1) // 2,
    }

    backbone_rank = random.choice(GEN_BACKBONE_RANKS)
    aux_ranks = [rank for rank in ("short", "medium", "long") if rank != backbone_rank]
    color_by_rank = _rank_to_color()

    direction_names = random.sample(list(DIRECTIONS), 3)
    backbone_dir = DIRECTIONS[direction_names[0]]
    aux_dirs = [DIRECTIONS[direction_names[1]], DIRECTIONS[direction_names[2]]]
    backbone_half = half_by_rank[backbone_rank]

    centers = [(r, c) for r in range(side) for c in range(side)]
    random.shuffle(centers)

    for center in centers:
        if _max_symmetric_half(shape, center, backbone_dir) < backbone_half:
            continue

        backbone = _segment_from_center(center, backbone_dir, backbone_half, color_by_rank[backbone_rank])
        backbone_cells = backbone["cells"]
        valid_indices = list(range(1, len(backbone_cells) - 1))
        index_pairs = [(i, j) for i in valid_indices for j in valid_indices if j - i >= 4]
        random.shuffle(index_pairs)
        if not index_pairs:
            continue

        for i, j in index_pairs:
            pivots = [backbone_cells[i], backbone_cells[j]]
            segments = [backbone]
            feasible = True
            for pivot, rank, direction in zip(pivots, aux_ranks, aux_dirs):
                half = half_by_rank[rank]
                if _max_symmetric_half(shape, pivot, direction) < half:
                    feasible = False
                    break
                segments.append(_segment_from_center(pivot, direction, half, color_by_rank[rank]))
            if not feasible:
                continue

            if not _valid_markers(segments):
                continue

            pair_masks = _pair_overlap_masks(shape, segments)
            if set(pair_masks) != {(0, 1), (0, 2)}:
                continue

            overlap_sizes = [int(mask.sum()) for mask in pair_masks.values()]
            if min(overlap_sizes) < 3 or max(overlap_sizes) > 15:
                continue

            triple_overlap = (
                _band_mask(shape, segments[0]["cells"])
                & _band_mask(shape, segments[1]["cells"])
                & _band_mask(shape, segments[2]["cells"])
                & ~_marker_mask(shape, segments)
            )
            if np.any(triple_overlap):
                continue

            grid = _build_input(shape, segments)
            expected = _render_output(shape, segments, grid)

            if np.count_nonzero(expected) > int(0.45 * expected.size):
                continue
            if any(np.count_nonzero(expected == color) < 6 for color in COLORS):
                continue

            winner_01 = segments[0] if segments[0]["length"] < segments[1]["length"] else segments[1]
            winner_02 = segments[0] if segments[0]["length"] < segments[2]["length"] else segments[2]
            if not np.all(expected[pair_masks[(0, 1)]] == int(winner_01["color"])):
                continue
            if not np.all(expected[pair_masks[(0, 2)]] == int(winner_02["color"])):
                continue

            if not _oracle_accepts(grid, expected):
                continue
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
