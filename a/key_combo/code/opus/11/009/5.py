import random
from collections import Counter

import numpy as np


BG = 0
PROTO_A_COLOR = 4
PROTO_B_COLOR = 7
MARKER_COLORS = (1, 2, 3, 5, 6, 8, 9)
PROTOTYPE_BOX_H = 2
PROTOTYPE_BOX_W = 3
GEN_RETRIES = 400

MASK_POOL = (
    np.array([[1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 0, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 0, 0], [0, 1, 1]], dtype=int),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _band_bounds(rows: list[int], height: int, mode: str = "floor") -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for idx, row in enumerate(rows):
        if idx == len(rows) - 1:
            end = height - 1
        else:
            if mode == "floor":
                end = (row + rows[idx + 1]) // 2
            elif mode == "ceil":
                end = (row + rows[idx + 1] + 1) // 2
            else:
                raise ValueError(f"unknown midpoint mode: {mode}")
        bounds.append((start, end))
        start = end + 1
    return bounds


def _extract_prototypes(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    assert h >= 4 and w >= 4, "grid is too small for the prototype showcase"
    a = grid[0:2, w - 3 : w].copy()
    b = grid[2:4, w - 3 : w].copy()

    assert set(map(int, np.unique(a))) <= {BG, PROTO_A_COLOR}
    assert set(map(int, np.unique(b))) <= {BG, PROTO_B_COLOR}
    assert 2 <= int((a == PROTO_A_COLOR).sum()) <= 4
    assert 2 <= int((b == PROTO_B_COLOR).sum()) <= 4
    assert int(np.any(a[0] == PROTO_A_COLOR)) and int(np.any(a[1] == PROTO_A_COLOR))
    assert int(np.any(b[0] == PROTO_B_COLOR)) and int(np.any(b[1] == PROTO_B_COLOR))
    assert int(np.any(a[:, 0] == PROTO_A_COLOR)) and int(np.any(a[:, 2] == PROTO_A_COLOR))
    assert int(np.any(b[:, 0] == PROTO_B_COLOR)) and int(np.any(b[:, 2] == PROTO_B_COLOR))
    assert not np.array_equal(a != BG, b != BG), "prototype masks must differ"
    return a, b


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    assert _get_bg(grid) == BG, "background must be black"

    showcase_mask = np.zeros_like(grid, dtype=bool)
    showcase_mask[0:4, w - 3 : w] = True
    outside = grid.copy()
    outside[showcase_mask] = BG
    nonzero = np.argwhere(outside != BG)
    assert nonzero.size > 0, "expected left-column markers"
    assert np.all(nonzero[:, 1] == 0), "all non-showcase foreground pixels must be left-column markers"

    markers = [(int(r), int(grid[r, 0])) for r in nonzero[:, 0]]
    markers.sort()
    rows = [row for row, _color in markers]
    colors = [color for _row, color in markers]
    assert 3 <= len(markers) <= 5
    assert len(rows) == len(set(rows)), "marker rows must be unique"
    assert len(colors) == len(set(colors)), "marker colors must be unique"
    assert all(1 <= row <= h - 2 for row in rows), "markers must stay off top/bottom edge"
    assert all(rows[i + 1] - rows[i] >= 4 for i in range(len(rows) - 1)), "adjacent markers need row gap >= 4"
    return markers


def _render_output(
    shape: tuple[int, int],
    markers: list[tuple[int, int]],
    proto_a: np.ndarray,
    proto_b: np.ndarray,
    *,
    midpoint_mode: str = "floor",
    parity_flip: bool = False,
    global_phase: bool = False,
) -> np.ndarray:
    h, w = shape
    rows = [row for row, _color in markers]
    bounds = _band_bounds(rows, h, mode=midpoint_mode)
    out = np.zeros(shape, dtype=int)
    col_phase = np.arange(w) % PROTOTYPE_BOX_W

    for band_idx, (start, end) in enumerate(bounds):
        proto = proto_b if (band_idx + int(parity_flip)) % 2 else proto_a
        for r in range(start, end + 1):
            phase_r = r % PROTOTYPE_BOX_H if global_phase else (r - start) % PROTOTYPE_BOX_H
            row_pattern = proto[phase_r, col_phase]
            mask = row_pattern != BG
            out[r, mask] = row_pattern[mask]

    out[0:4, w - 3 : w] = BG
    for row, color in markers:
        out[row, 0] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Split the grid into marker-defined horizontal bands and tile alternating prototypes into those bands.

    Input structure:
    - A black background grid contains 3-5 distinct singleton markers in the leftmost column.
    - Marker rows are unique and separated vertically; only their row positions matter for the partition.
    - The top-right 3x2 showcase contains prototype A in yellow, and the 3x2 block directly below it contains prototype B in orange.

    Rule:
    - Sort the left-column markers by row and split the grid at the floor midpoint between consecutive marker rows.
    - Number the resulting horizontal bands from top to bottom; even bands use prototype A and odd bands use prototype B.
    - Tile the chosen 3x2 prototype from that band's own top-left corner across the full band, truncate at the band edges, keep the markers on top, and clear the two showcase boxes to black.

    Key insight:
    - The markers do not encode colors for the fill; they only define dynamic band boundaries, and each band's parity selects which prototype geometry gets tiled there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    proto_a, proto_b = _extract_prototypes(g)
    markers = _extract_markers(g)
    return _render_output(g.shape, markers, proto_a, proto_b)


def _random_prototype(color: int, rng: random.Random, used_masks: set[bytes]) -> np.ndarray:
    choices = list(MASK_POOL)
    rng.shuffle(choices)
    for mask in choices:
        key = mask.tobytes()
        if key in used_masks:
            continue
        used_masks.add(key)
        proto = np.where(mask == 1, color, BG)
        return proto
    raise RuntimeError("failed to sample a distinct prototype mask")


def _max_marker_count(height: int) -> int:
    return max(3, min(5, 1 + (height - 3) // 4))


def _sample_marker_rows(height: int, count: int, rng: random.Random) -> list[int] | None:
    candidates = list(range(1, height - 1))
    for _ in range(250):
        rows = sorted(rng.sample(candidates, count))
        if any(rows[i + 1] - rows[i] < 4 for i in range(count - 1)):
            continue
        if not any((rows[i + 1] - rows[i]) % 2 == 1 for i in range(count - 1)):
            continue
        bounds = _band_bounds(rows, height, mode="floor")
        heights = [end - start + 1 for start, end in bounds]
        if min(heights) < 2:
            continue
        if not any(start % 2 == 1 for start, _end in bounds[1:]):
            continue
        if not any(band_h % 2 == 1 for band_h in heights):
            continue
        return rows
    return None


def _validate_structure(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    markers = _extract_markers(grid)
    proto_a, proto_b = _extract_prototypes(grid)
    return markers, proto_a, proto_b


def _validate_semantics(grid: np.ndarray) -> None:
    markers, proto_a, proto_b = _validate_structure(grid)
    correct = _render_output(grid.shape, markers, proto_a, proto_b)
    swapped = _render_output(grid.shape, markers, proto_a, proto_b, parity_flip=True)
    global_phase = _render_output(grid.shape, markers, proto_a, proto_b, global_phase=True)
    ceil_midpoint = _render_output(grid.shape, markers, proto_a, proto_b, midpoint_mode="ceil")

    assert not np.array_equal(correct, swapped), "parity selection must affect the result"
    assert not np.array_equal(correct, global_phase), "band-local tiling origin must matter"
    assert not np.array_equal(correct, ceil_midpoint), "floor midpoint choice must matter"
    assert np.all(correct[0:4, grid.shape[1] - 3 : grid.shape[1]] == BG), "showcase must be cleared"
    for row, color in markers:
        assert int(correct[row, 0]) == color, "markers must be preserved over the tiling"


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        height = rng.randint(15, 22)
        width = rng.randint(12, 20)
        max_count = _max_marker_count(height)
        count = rng.randint(3, max_count)

        rows = _sample_marker_rows(height, count, rng)
        if rows is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        colors = rng.sample(MARKER_COLORS, count)
        for row, color in zip(rows, colors, strict=True):
            grid[row, 0] = color

        used_masks: set[bytes] = set()
        proto_a = _random_prototype(PROTO_A_COLOR, rng, used_masks)
        proto_b = _random_prototype(PROTO_B_COLOR, rng, used_masks)
        grid[0:2, width - 3 : width] = proto_a
        grid[2:4, width - 3 : width] = proto_b

        try:
            _validate_semantics(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle-5 input")
