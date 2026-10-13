from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
GREEN = 3

PATTERN_COLORS = [2, 4, 5, 6, 7, 8, 9]
QUADRANTS = ("ul", "ur", "ll", "lr")
OPPOSITE = {"ul": "lr", "ur": "ll", "ll": "ur", "lr": "ul"}
MISSING_FROM_OFFSET = {
    (0, 0): "ul",
    (0, 1): "ur",
    (1, 0): "ll",
    (1, 1): "lr",
}

# --------------- solver ---------------


def _non_separator_spans(flags: list[bool]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_sep in enumerate(flags):
        if not is_sep and start is None:
            start = idx
        elif is_sep and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(flags)))
    assert spans, "expected at least one non-separator span"
    return spans


def _quadrant_slices(height: int, width: int) -> dict[str, tuple[slice, slice]]:
    top = slice(0, height // 2)
    bottom = slice((height + 1) // 2, height)
    left = slice(0, width // 2)
    right = slice((width + 1) // 2, width)
    quads = {
        "ul": (top, left),
        "ur": (top, right),
        "ll": (bottom, left),
        "lr": (bottom, right),
    }
    shapes = {(q[0].stop - q[0].start, q[1].stop - q[1].start) for q in quads.values()}
    assert len(shapes) == 1, "quadrants must have equal shapes"
    return quads


def _missing_quadrant_from_marker(region: np.ndarray) -> str:
    green = np.argwhere(region == GREEN)
    assert green.shape[0] == 3, "region must contain exactly one 3-cell green L marker"
    r0, c0 = green.min(axis=0)
    r1, c1 = green.max(axis=0)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "green marker must occupy a 2x2 box"
    box = region[r0 : r1 + 1, c0 : c1 + 1]
    missing = np.argwhere(box != GREEN)
    assert missing.shape[0] == 1, "green marker box must have one missing corner"
    return MISSING_FROM_OFFSET[(int(missing[0, 0]), int(missing[0, 1]))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    In subregions separated by blue full rows/columns, a green L marker encodes which
    quadrant is missing from a 4-fold rotationally symmetric pattern. Restore the
    missing quadrant from the opposite quadrant rotated by 180 degrees, then erase
    the marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_flags = [bool(np.all(g[r, :] == BLUE)) for r in range(g.shape[0])]
    col_flags = [bool(np.all(g[:, c] == BLUE)) for c in range(g.shape[1])]
    row_spans = _non_separator_spans(row_flags)
    col_spans = _non_separator_spans(col_flags)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = out[r0:r1, c0:c1].copy()
            missing = _missing_quadrant_from_marker(region)
            region[region == GREEN] = BG

            quads = _quadrant_slices(*region.shape)
            src_rows, src_cols = quads[OPPOSITE[missing]]
            dst_rows, dst_cols = quads[missing]
            region[dst_rows, dst_cols] = np.rot90(region[src_rows, src_cols], 2)
            out[r0:r1, c0:c1] = region

    return out


# --------------- generator ---------------

ROW_COUNT_RANGE = (2, 3)
COL_COUNT_RANGE = (2, 3)
CELL_SIZE_CHOICES = (9, 11, 13)
OUTER_BORDER_PROBABILITY = 0.5
CENTER_STYLES = ("none", "dot", "cross")
MARKER_CORNERS = ("tl", "tr", "bl", "br")
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _connected_seed(size: int, color: int) -> np.ndarray:
    area = size * size
    target = random.randint(2, area - 1)
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], size, size))

    while int(mask.sum()) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nb in _neighbors4(cell[0], cell[1], size, size):
            if not mask[nb]:
                frontier.add(nb)

    assert int(mask.sum()) == target
    return np.where(mask, color, BG).astype(int)


def _full_pattern(quadrant_size: int, color: int) -> np.ndarray:
    seed = _connected_seed(quadrant_size, color)
    full_size = quadrant_size * 2 + 1
    patch = np.full((full_size, full_size), BG, dtype=int)
    patch[:quadrant_size, :quadrant_size] = seed
    patch[:quadrant_size, quadrant_size + 1 :] = np.rot90(seed, -1)
    patch[quadrant_size + 1 :, :quadrant_size] = np.rot90(seed, 1)
    patch[quadrant_size + 1 :, quadrant_size + 1 :] = np.rot90(seed, 2)

    style = random.choice(CENTER_STYLES)
    if style == "dot":
        patch[quadrant_size, quadrant_size] = color
    elif style == "cross":
        arm = random.randint(1, quadrant_size)
        patch[quadrant_size, quadrant_size - arm : quadrant_size + arm + 1] = color
        patch[quadrant_size - arm : quadrant_size + arm + 1, quadrant_size] = color

    return patch


def _layout_spans(lengths: list[int], outer_border: bool) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = [0] if outer_border else []
    pos = 1 if outer_border else 0

    for idx, length in enumerate(lengths):
        spans.append((pos, pos + length))
        pos += length
        if idx < len(lengths) - 1:
            separators.append(pos)
            pos += 1

    if outer_border:
        separators.append(pos)
        pos += 1
    return spans, separators, pos


def _erase_quadrant(patch: np.ndarray, which: str) -> np.ndarray:
    out = patch.copy()
    rows, cols = _quadrant_slices(*patch.shape)[which]
    out[rows, cols] = BG
    return out


def _corner_origin(height: int, width: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - 2
    if corner == "bl":
        return height - 2, 0
    if corner == "br":
        return height - 2, width - 2
    raise AssertionError("unknown marker corner")


def _place_marker(region: np.ndarray, missing_quadrant: str, corner: str) -> None:
    r0, c0 = _corner_origin(*region.shape, corner)
    region[r0 : r0 + 2, c0 : c0 + 2] = GREEN
    dr, dc = next(offset for offset, quad in MISSING_FROM_OFFSET.items() if quad == missing_quadrant)
    region[r0 + dr, c0 + dc] = BG


def _make_region(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    max_quadrant = (min(height, width) - 5) // 2
    assert max_quadrant >= 2
    quadrant_size = random.randint(2, max_quadrant)
    color = random.choice(PATTERN_COLORS)
    full = _full_pattern(quadrant_size, color)
    patch_size = full.shape[0]

    top = (height - patch_size) // 2
    left = (width - patch_size) // 2
    bottom = top + patch_size
    right = left + patch_size
    assert min(top, left, height - bottom, width - right) >= 2

    output_region = np.full((height, width), BG, dtype=int)
    output_region[top:bottom, left:right] = full

    missing = random.choice(QUADRANTS)
    input_region = output_region.copy()
    input_region[top:bottom, left:right] = _erase_quadrant(full, missing)
    _place_marker(input_region, missing, random.choice(MARKER_CORNERS))
    return input_region, output_region


def _build_sample() -> tuple[np.ndarray, np.ndarray]:
    row_count = random.randint(*ROW_COUNT_RANGE)
    col_count = random.randint(*COL_COUNT_RANGE)
    row_lengths = [random.choice(CELL_SIZE_CHOICES) for _ in range(row_count)]
    col_lengths = [random.choice(CELL_SIZE_CHOICES) for _ in range(col_count)]
    outer_border = random.random() < OUTER_BORDER_PROBABILITY

    row_spans, row_separators, height = _layout_spans(row_lengths, outer_border)
    col_spans, col_separators, width = _layout_spans(col_lengths, outer_border)

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    inp[row_separators, :] = BLUE
    inp[:, col_separators] = BLUE
    out[row_separators, :] = BLUE
    out[:, col_separators] = BLUE

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region_in, region_out = _make_region(r1 - r0, c1 - c0)
            inp[r0:r1, c0:c1] = region_in
            out[r0:r1, c0:c1] = region_out

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    inp, expected = _build_sample()
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate valid input")
