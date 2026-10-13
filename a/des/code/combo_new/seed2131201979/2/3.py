from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 2
SEPARATOR = 5
GEN_ATTEMPTS = 400
MIN_REGION_SIZE = 5
MAX_REGION_SIZE = 12
PALETTE_POOL = (1, 3, 4, 6, 7, 8, 9)


def _dominant_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    best = int(np.argmax(counts))
    max_count = int(counts[best])
    assert int(np.sum(counts == max_count)) == 1, "expected a unique dominant color"
    return int(colors[best])


def _separator_structure(body: np.ndarray) -> tuple[int, int, int]:
    candidates = []
    for color0 in np.unique(body):
        color = int(color0)
        full_rows = np.flatnonzero(np.all(body == color, axis=1))
        full_cols = np.flatnonzero(np.all(body == color, axis=0))
        if len(full_rows) == 1 and len(full_cols) == 1:
            candidates.append((color, int(full_rows[0]), int(full_cols[0])))

    assert len(candidates) == 1, f"expected one separator cross, got {candidates}"
    return candidates[0]


def _palette_from_row(row: np.ndarray) -> list[int]:
    runs: list[int] = []
    prev = int(row[0])
    runs.append(prev)
    for value0 in row[1:]:
        value = int(value0)
        if value != prev:
            runs.append(value)
            prev = value

    assert len(runs) == 4, f"expected four palette blocks, got {runs}"
    assert len(set(runs)) == 4, "palette colors must be distinct"
    return runs


def _quadrant_slices(cross_r: int, cross_c: int, body_shape: tuple[int, int]) -> list[tuple[slice, slice]]:
    height, width = body_shape
    assert 0 < cross_r < height - 1, "separator row must leave non-empty top and bottom regions"
    assert 0 < cross_c < width - 1, "separator column must leave non-empty left and right regions"
    return [
        (slice(0, cross_r), slice(0, cross_c)),
        (slice(0, cross_r), slice(cross_c + 1, width)),
        (slice(cross_r + 1, height), slice(0, cross_c)),
        (slice(cross_r + 1, height), slice(cross_c + 1, width)),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each quadrant's marker pattern by indexing into the bottom palette with that quadrant's marker count.

    Input structure:
    - A one-cell-thick separator color forms one full horizontal line and one full vertical line, splitting the grid body into four rectangular quadrants.
    - Inside each quadrant, the background is black and the only foreground cells are red 1x1 marker pixels.
    - The final input row is a palette strip made of four left-to-right monochrome color blocks.

    Rule:
    - Ignore the bottom palette row when locating the separator cross and the four quadrants.
    - In each quadrant, count the red markers; if the count is N, choose the N-th palette color from the bottom strip.
    - Replace that quadrant's red marker pixels with the chosen palette color and leave every other quadrant cell black.
    - Keep the separator cross unchanged and remove the palette row from the output.

    Key insight:
    - The bottom strip is an ordered lookup table rather than content to copy directly: each quadrant uses its local marker count to choose one color, but the marker positions themselves are preserved.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3 and g.shape[1] >= 5, "grid is too small for the described structure"

    body = g[:-1]
    sep, cross_r, cross_c = _separator_structure(body)
    palette = _palette_from_row(g[-1])

    non_separator = body[body != sep]
    bg = _dominant_color(non_separator)
    marker_colors = [int(color) for color in np.unique(non_separator) if int(color) != int(bg)]
    assert len(marker_colors) == 1, f"expected one marker color, got {marker_colors}"
    marker = marker_colors[0]

    out = body.copy()
    out[out == marker] = bg

    for rs, cs in _quadrant_slices(cross_r, cross_c, body.shape):
        region = body[rs, cs]
        marker_mask = region == marker
        count = int(np.count_nonzero(marker_mask))
        assert 1 <= count <= 4, f"marker count must be between 1 and 4, got {count}"
        out_region = out[rs, cs]
        out_region[marker_mask] = palette[count - 1]

    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    cuts = sorted(random.sample(range(1, total), parts - 1))
    pieces = []
    prev = 0
    for cut in cuts + [total]:
        pieces.append(cut - prev)
        prev = cut
    return pieces


def _sample_marker_points(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    for _ in range(120):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for cell in candidates:
            if all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _render_expected(
    body_shape: tuple[int, int],
    cross_r: int,
    cross_c: int,
    palette: list[int],
    points_by_quadrant: list[list[tuple[int, int]]],
) -> np.ndarray:
    out = np.full(body_shape, BG, dtype=int)
    out[cross_r, :] = SEPARATOR
    out[:, cross_c] = SEPARATOR
    for region_index, (rs, cs) in enumerate(_quadrant_slices(cross_r, cross_c, body_shape)):
        color = palette[len(points_by_quadrant[region_index]) - 1]
        for r, c in points_by_quadrant[region_index]:
            out[rs.start + r, cs.start + c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE)
    bottom_h = random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE)
    left_w = random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE)
    right_w = random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE)

    body_h = top_h + 1 + bottom_h
    body_w = left_w + 1 + right_w
    if body_h + 1 > 30 or body_w > 30:
        return None

    palette = random.sample(PALETTE_POOL, 4)
    palette_widths = _positive_composition(body_w, 4)
    counts = [1, 2, 3, 4]
    random.shuffle(counts)

    quadrants = [
        (top_h, left_w),
        (top_h, right_w),
        (bottom_h, left_w),
        (bottom_h, right_w),
    ]
    points_by_quadrant: list[list[tuple[int, int]]] = []
    for (qh, qw), count in zip(quadrants, counts):
        pts = _sample_marker_points(qh, qw, count)
        if pts is None:
            return None
        points_by_quadrant.append(pts)

    body = np.full((body_h, body_w), BG, dtype=int)
    cross_r = top_h
    cross_c = left_w
    body[cross_r, :] = SEPARATOR
    body[:, cross_c] = SEPARATOR

    for (rs, cs), pts in zip(_quadrant_slices(cross_r, cross_c, body.shape), points_by_quadrant):
        for r, c in pts:
            body[rs.start + r, cs.start + c] = MARKER

    palette_row = np.empty(body_w, dtype=int)
    start = 0
    for color, width in zip(palette, palette_widths):
        palette_row[start : start + width] = int(color)
        start += width

    inp = np.vstack([body, palette_row[None, :]])
    expected = _render_expected(body.shape, cross_r, cross_c, palette, points_by_quadrant)

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-cross puzzle")
