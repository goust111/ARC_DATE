from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


BACKGROUND = 0
MARKER = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
SAMPLE_SIZE = 3
OUTPUT_SIZE = SAMPLE_SIZE * SAMPLE_SIZE
GEN_RETRIES = 500

# Grey 3-cell L markers touching one corner of the 3x3 sample.
MARKER_OFFSETS = {
    "se": ((2, 3), (3, 2), (3, 3)),
    "sw": ((2, -1), (3, -1), (3, 0)),
    "ne": ((-1, 2), (-1, 3), (0, 3)),
    "nw": ((-1, -1), (-1, 0), (0, -1)),
}

# Normalize the marker bbox to a 2x2 mask and recover the sample's top-left.
MARKER_SHAPES = {
    ((0, 1), (1, 0), (1, 1)): (-2, -2),  # marker sits south-east of the sample
    ((0, 0), (1, 0), (1, 1)): (-2, 1),   # marker sits south-west of the sample
    ((0, 0), (0, 1), (1, 1)): (1, -2),   # marker sits north-east of the sample
    ((0, 0), (0, 1), (1, 0)): (1, 1),    # marker sits north-west of the sample
}

_SEEN_SIGNATURES: set[str] = set()


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sample_colors(sample: np.ndarray) -> list[int]:
    return sorted({int(v) for v in np.unique(sample) if int(v) not in (BACKGROUND, MARKER)})


def _find_sample_from_marker(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    gray = np.argwhere(grid == MARKER)
    assert len(gray) == 3, "expected exactly one 3-cell grey marker"
    r0, c0, r1, c1 = _bbox(gray)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "marker must be a 2x2 L triomino"

    normalized = tuple(sorted((int(r - r0), int(c - c0)) for r, c in gray))
    assert normalized in MARKER_SHAPES, "unrecognized grey L orientation"
    dr, dc = MARKER_SHAPES[normalized]
    sr = r0 + dr
    sc = c0 + dc
    assert 0 <= sr <= grid.shape[0] - SAMPLE_SIZE
    assert 0 <= sc <= grid.shape[1] - SAMPLE_SIZE

    sample = grid[sr : sr + SAMPLE_SIZE, sc : sc + SAMPLE_SIZE]
    colors = _sample_colors(sample)
    assert len(colors) == 1, "sample must use exactly one nonzero foreground color"
    fg_count = int(np.count_nonzero(sample == colors[0]))
    assert 3 <= fg_count <= 6, "sample foreground count must be 3..6"
    assert np.count_nonzero(sample == MARKER) == 0
    return (sr, sc), sample.copy()


def _find_duplicate(grid: np.ndarray, sample: np.ndarray, source: tuple[int, int]) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    for r in range(grid.shape[0] - SAMPLE_SIZE + 1):
        for c in range(grid.shape[1] - SAMPLE_SIZE + 1):
            if np.array_equal(grid[r : r + SAMPLE_SIZE, c : c + SAMPLE_SIZE], sample):
                matches.append((r, c))
    assert source in matches, "marked sample must be searchable as a 3x3 window"
    assert len(matches) == 2, "expected exactly one duplicate besides the sample"
    duplicate = [pos for pos in matches if pos != source]
    assert len(duplicate) == 1
    dr, dc = duplicate[0]
    assert 1 <= dr <= grid.shape[0] - 4 and 1 <= dc <= grid.shape[1] - 4
    ring = grid[dr - 1 : dr + 4, dc - 1 : dc + 4].copy()
    ring[1:4, 1:4] = BACKGROUND
    assert np.all(ring == BACKGROUND), "duplicate must have a 1-cell empty margin"
    return dr, dc


def _kronecker_invert(mask: np.ndarray, color: int) -> np.ndarray:
    tile = np.where(mask, int(color), BACKGROUND).astype(int)
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for r in range(SAMPLE_SIZE):
        for c in range(SAMPLE_SIZE):
            if not mask[r, c]:
                out[r * SAMPLE_SIZE : (r + 1) * SAMPLE_SIZE, c * SAMPLE_SIZE : (c + 1) * SAMPLE_SIZE] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the marked 3x3 sample, locate its matching copy, and expand that copy's binary mask by an inverted self-Kronecker rule.

    Input structure:
    - The grid has black background, exactly one grey 3-cell L marker, one marked 3x3 sample, and one other 3x3 duplicate.
    - The sample contains black plus exactly one non-grey foreground color, with 3..6 foreground cells.
    - The duplicate matches the sample cell-for-cell and has a one-cell black margin on every side.

    Rule:
    - Use the grey L orientation to recover which 3x3 region is the sample.
    - Search the whole grid for the unique other 3x3 region that matches this sample exactly.
    - Binarize that duplicate: foreground cells become 1 and black cells become 0.
    - Build a 9x9 output of 3x3 blocks: each duplicate background cell emits one copy of the binary mask in the duplicate's foreground color, while each duplicate foreground cell emits an all-black 3x3 block.

    Key insight:
    - The grey L identifies the reference sample, but the 9x9 result is produced from the far duplicate's binary mask after foreground/background inversion at the block level.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    source, sample = _find_sample_from_marker(g)
    dr, dc = _find_duplicate(g, sample, source)
    duplicate = g[dr : dr + SAMPLE_SIZE, dc : dc + SAMPLE_SIZE]
    colors = _sample_colors(duplicate)
    assert len(colors) == 1
    fg = colors[0]
    return _kronecker_invert(duplicate != BACKGROUND, fg)


def _occupied_cells(sample_pos: tuple[int, int], marker_offsets: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    sr, sc = sample_pos
    cells = {(sr + r, sc + c) for r in range(SAMPLE_SIZE) for c in range(SAMPLE_SIZE)}
    cells |= {(sr + dr, sc + dc) for dr, dc in marker_offsets}
    return cells


def _sample_position_candidates(shape: tuple[int, int], orientation: str) -> list[tuple[int, int]]:
    H, W = shape
    offsets = MARKER_OFFSETS[orientation]
    cells = [(r, c) for r in range(SAMPLE_SIZE) for c in range(SAMPLE_SIZE)] + list(offsets)
    candidates: list[tuple[int, int]] = []
    for sr in range(H):
        for sc in range(W):
            if all(0 <= sr + dr < H and 0 <= sc + dc < W for dr, dc in cells):
                candidates.append((sr, sc))
    return candidates


def _duplicate_position_candidates(shape: tuple[int, int], forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    H, W = shape
    candidates: list[tuple[int, int]] = []
    for dr in range(1, H - 3):
        for dc in range(1, W - 3):
            ring = {
                (r, c)
                for r in range(dr - 1, dr + 4)
                for c in range(dc - 1, dc + 4)
            }
            if ring & forbidden:
                continue
            candidates.append((dr, dc))
    return candidates


def _random_sample(rng: np.random.Generator) -> np.ndarray:
    color = int(rng.choice(PALETTE))
    fg_count = int(rng.integers(3, 7))
    sample = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE), dtype=int)
    positions = rng.choice(SAMPLE_SIZE * SAMPLE_SIZE, size=fg_count, replace=False)
    sample.flat[positions] = color
    return sample


def _signature(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _build_instance(
    shape: tuple[int, int],
    orientation: str,
    sample_pos: tuple[int, int],
    duplicate_pos: tuple[int, int],
    sample: np.ndarray,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    sr, sc = sample_pos
    dr, dc = duplicate_pos
    grid[sr : sr + SAMPLE_SIZE, sc : sc + SAMPLE_SIZE] = sample
    for mr, mc in MARKER_OFFSETS[orientation]:
        grid[sr + mr, sc + mc] = MARKER
    grid[dr : dr + SAMPLE_SIZE, dc : dc + SAMPLE_SIZE] = sample
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        source, sample = _find_sample_from_marker(grid)
        dr, dc = _find_duplicate(grid, sample, source)
        colors = _sample_colors(sample)
        if len(colors) != 1:
            return False
        fg = colors[0]
        expected = _kronecker_invert(sample != BACKGROUND, fg)
        if not np.array_equal(main(grid), expected):
            return False

        occupied = set()
        sr, sc = source
        occupied |= {(sr + r, sc + c) for r in range(SAMPLE_SIZE) for c in range(SAMPLE_SIZE)}
        occupied |= {(dr + r, dc + c) for r in range(SAMPLE_SIZE) for c in range(SAMPLE_SIZE)}
        gray = np.argwhere(grid == MARKER)
        occupied |= {tuple(map(int, rc)) for rc in gray}
        for r, c in np.argwhere(grid != BACKGROUND):
            if (int(r), int(c)) not in occupied:
                return False

        # Reject trivial output collisions within the same process.
        sig = _signature(expected)
        if sig in _SEEN_SIGNATURES:
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        shape = (int(rng.integers(15, 21)), int(rng.integers(15, 21)))
        orientation = str(rng.choice(tuple(MARKER_OFFSETS)))
        sample = _random_sample(rng)

        sample_positions = _sample_position_candidates(shape, orientation)
        if not sample_positions:
            continue
        sample_pos = sample_positions[int(rng.integers(len(sample_positions)))]
        forbidden = _occupied_cells(sample_pos, MARKER_OFFSETS[orientation])
        duplicate_positions = _duplicate_position_candidates(shape, forbidden)
        if not duplicate_positions:
            continue
        duplicate_pos = duplicate_positions[int(rng.integers(len(duplicate_positions)))]

        grid = _build_instance(shape, orientation, sample_pos, duplicate_pos, sample)
        if not _validate_instance(grid):
            continue

        _SEEN_SIGNATURES.add(_signature(main(grid)))
        return grid
    raise RuntimeError("failed to generate a valid new ARC input")
