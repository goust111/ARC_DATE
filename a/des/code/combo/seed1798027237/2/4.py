from __future__ import annotations

from collections import deque
import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
LAYER_RANGE = (2, 5)
INNER_SIZE_RANGE = (3, 8)
EXTRA_MARGIN_RANGE = (4, 6)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    return set(coords) == _frame_cells(bbox)


def _compress_runs(values: list[int]) -> tuple[list[int], list[int]]:
    assert values
    colors = [int(values[0])]
    lengths = [1]
    for value in values[1:]:
        value = int(value)
        if value == colors[-1]:
            lengths[-1] += 1
        else:
            colors.append(value)
            lengths.append(1)
    return colors, lengths


def _read_edge_band(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    edge_arrays = {
        "top": g[0, :],
        "bottom": g[h - 1, :],
        "left": g[:, 0],
        "right": g[:, w - 1],
    }
    counts = {side: int(np.count_nonzero(arr != BG)) for side, arr in edge_arrays.items()}
    side = max(counts, key=counts.get)
    ordered_counts = sorted(counts.values(), reverse=True)
    assert ordered_counts[0] > 0
    if len(ordered_counts) > 1:
        assert ordered_counts[0] > ordered_counts[1]

    arr = edge_arrays[side]
    nz = np.flatnonzero(arr != BG)
    assert nz.size > 0
    lo = int(nz[0])
    hi = int(nz[-1])
    assert np.all(arr[lo : hi + 1] != BG)
    values = [int(v) for v in arr[lo : hi + 1]]

    edge_len = w if side in ("top", "bottom") else h
    d_lo = min(lo, edge_len - 1 - lo)
    d_hi = min(hi, edge_len - 1 - hi)
    assert d_lo != d_hi
    if d_lo > d_hi:
        values.reverse()

    colors, lengths = _compress_runs(values)
    assert all(length >= 2 for length in lengths)
    return colors


def _find_frames(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    mask = g != BG
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False

    frames: list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = []
    for coords in _find_components(mask):
        if not _is_rectangular_frame(coords):
            continue
        frames.append((_bbox(coords), coords))

    assert frames
    frames.sort(key=lambda item: (item[0][0], item[0][1], -item[0][2], -item[0][3]))
    for i in range(1, len(frames)):
        pr0, pc0, pr1, pc1 = frames[i - 1][0]
        cr0, cc0, cr1, cc1 = frames[i][0]
        assert (cr0, cc0, cr1, cc1) == (pr0 + 2, pc0 + 2, pr1 - 2, pc1 - 2)
    return frames


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    band_colors = _read_edge_band(g)
    frames = _find_frames(g)
    assert len(band_colors) == len(frames)

    out = g.copy()
    for color, (_bbox_, coords) in zip(band_colors, frames):
        for r, c in coords:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor concentric rectangular frames using the color order encoded by a border strip.

    Input structure:
    - Black background.
    - Several 1-pixel concentric rectangular frames separated by 1-pixel black gaps.
    - One outer grid edge contains a contiguous multicolor strip made of solid segments.

    Rule:
    - Read the strip from the endpoint that is closer to a corner toward the opposite endpoint.
    - Order the internal frames from outermost to innermost.
    - Recolor those frames with the strip colors in that order.
    - Leave the strip and black gaps unchanged.
    """
    return _solve(np.asarray(grid, dtype=int))


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _split_with_minimum(rng: np.random.Generator, total: int, low: int) -> tuple[int, int]:
    left = int(rng.integers(low, total - low + 1))
    right = total - left
    assert left >= low and right >= low
    return left, right


def _random_lengths(rng: np.random.Generator, count: int, total: int) -> list[int]:
    lengths = [2] * count
    for _ in range(total - 2 * count):
        lengths[int(rng.integers(0, count))] += 1
    return lengths


def _place_band(
    grid: np.ndarray,
    side: str,
    colors: list[int],
    lengths: list[int],
    near_low_end: bool,
    near_margin: int,
) -> None:
    assert len(colors) == len(lengths)
    total = sum(lengths)
    band_colors = colors if near_low_end else list(reversed(colors))

    if side == "top":
        start = near_margin if near_low_end else grid.shape[1] - near_margin - total
        c = start
        for color, length in zip(band_colors, lengths):
            grid[0, c : c + length] = color
            c += length
        return
    if side == "bottom":
        start = near_margin if near_low_end else grid.shape[1] - near_margin - total
        c = start
        for color, length in zip(band_colors, lengths):
            grid[-1, c : c + length] = color
            c += length
        return
    if side == "left":
        start = near_margin if near_low_end else grid.shape[0] - near_margin - total
        r = start
        for color, length in zip(band_colors, lengths):
            grid[r : r + length, 0] = color
            r += length
        return

    start = near_margin if near_low_end else grid.shape[0] - near_margin - total
    r = start
    for color, length in zip(band_colors, lengths):
        grid[r : r + length, -1] = color
        r += length


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    layers = int(rng.integers(LAYER_RANGE[0], LAYER_RANGE[1] + 1))
    inner_h = int(rng.integers(INNER_SIZE_RANGE[0], INNER_SIZE_RANGE[1] + 1))
    inner_w = int(rng.integers(INNER_SIZE_RANGE[0], INNER_SIZE_RANGE[1] + 1))

    outer_h = inner_h + 4 * (layers - 1)
    outer_w = inner_w + 4 * (layers - 1)
    extra_h = int(rng.integers(EXTRA_MARGIN_RANGE[0], EXTRA_MARGIN_RANGE[1] + 1))
    extra_w = int(rng.integers(EXTRA_MARGIN_RANGE[0], EXTRA_MARGIN_RANGE[1] + 1))
    h = outer_h + extra_h
    w = outer_w + extra_w

    top_margin, bottom_margin = _split_with_minimum(rng, extra_h, 2)
    left_margin, right_margin = _split_with_minimum(rng, extra_w, 2)
    assert top_margin + bottom_margin == extra_h
    assert left_margin + right_margin == extra_w

    grid = np.zeros((h, w), dtype=int)

    frame_colors = list(map(int, rng.choice(PALETTE, size=layers, replace=False)))
    band_colors = frame_colors.copy()
    while band_colors == frame_colors:
        rng.shuffle(band_colors)

    layer_bboxes: list[tuple[int, int, int, int]] = []
    for layer in range(layers):
        bbox = (
            top_margin + 2 * layer,
            left_margin + 2 * layer,
            top_margin + outer_h - 1 - 2 * layer,
            left_margin + outer_w - 1 - 2 * layer,
        )
        layer_bboxes.append(bbox)
        _draw_frame(grid, bbox, frame_colors[layer])

    side = ("top", "bottom", "left", "right")[int(rng.integers(0, 4))]
    edge_len = w if side in ("top", "bottom") else h
    min_total = 2 * layers
    max_near_margin = min(3, (edge_len - min_total - 1) // 2)
    assert max_near_margin >= 0
    near_margin = int(rng.integers(0, max_near_margin + 1))
    max_total = edge_len - 2 * near_margin - 1
    extra_band = int(rng.integers(0, min(9, max_total - min_total) + 1))
    total = min_total + extra_band
    lengths = _random_lengths(rng, layers, total)
    near_low_end = bool(rng.integers(0, 2))
    _place_band(grid, side, band_colors, lengths, near_low_end, near_margin)

    expected = grid.copy()
    for color, bbox in zip(band_colors, layer_bboxes):
        _draw_frame(expected, bbox, color)

    solved = _solve(grid)
    assert np.array_equal(solved, expected)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        try:
            grid, _expected = _generate_case(rng)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
