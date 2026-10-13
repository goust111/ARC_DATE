from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _orbit_local(source_r: int, source_c: int, size: int) -> list[tuple[int, int]]:
    return [
        (source_r, source_c),
        (source_c, size - 1 - source_r),
        (size - 1 - source_r, size - 1 - source_c),
        (size - 1 - source_c, source_r),
    ]


def _inverse_to_source(quadrant: int, local_r: int, local_c: int, size: int) -> tuple[int, int]:
    if quadrant == 0:
        return local_r, local_c
    if quadrant == 1:
        return size - 1 - local_c, local_r
    if quadrant == 2:
        return size - 1 - local_r, size - 1 - local_c
    if quadrant == 3:
        return local_c, size - 1 - local_r
    raise ValueError(f"invalid quadrant index: {quadrant}")


def _majority_value(values: list[int]) -> int:
    return Counter(map(int, values)).most_common(1)[0][0]


def _split_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 0, "expected an even-sized square collage"
    half = h // 2
    return [
        g[:half, :half],
        g[:half, half:],
        g[half:, half:],
        g[half:, :half],
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """Harvest the error pixels from a four-quadrant rotational collage.

    Input structure:
    - The input is an even-sized square grid split into four equal quadrants.
    - The top-left quadrant is a source tile; the other three quadrants are its
      90-degree, 180-degree, and 270-degree clockwise rotations, except for a
      small number of deviant pixels.

    Rule:
    - For each source-tile coordinate, collect the four pixels on its rotation
      orbit across the quadrants and take their majority value as the consensus.
    - Any orbit member that differs from the consensus is an error pixel.
    - Send each error pixel back through the inverse rotation of its quadrant and
      write the error pixel's own color at that source coordinate.

    Key insight:
    - The four quadrants are redundant rotated copies, so the repeated orbit is a
      built-in voting system; the noise is the payload, not something to discard.
    """
    quadrants = _split_quadrants(grid)
    size = quadrants[0].shape[0]
    out = np.full((size, size), _get_bg(grid), dtype=int)

    for source_r in range(size):
        for source_c in range(size):
            orbit = _orbit_local(source_r, source_c, size)
            values = [int(quadrants[q][r, c]) for q, (r, c) in enumerate(orbit)]
            consensus = _majority_value(values)
            for quadrant, ((local_r, local_c), value) in enumerate(zip(orbit, values)):
                if value == consensus:
                    continue
                dst_r, dst_c = _inverse_to_source(quadrant, local_r, local_c, size)
                out[dst_r, dst_c] = value
    return out


BACKGROUND = 0
PALETTE = list(range(1, 10))
SOURCE_SIDE_RANGE = range(5, 15)
SOURCE_COLOR_COUNT_RANGE = (2, 5)
SOURCE_OBJECT_COUNT_RANGE = (2, 5)
PAYLOAD_COLOR_COUNT_RANGE = (1, 4)
ERROR_COUNT_RANGE = (3, 9)
MAX_GENERATION_ATTEMPTS = 300


def _paint_rect(tile: np.ndarray, color: int) -> None:
    size = tile.shape[0]
    h = random.randint(1, max(1, size // 2))
    w = random.randint(1, max(1, size // 2))
    r0 = random.randint(0, size - h)
    c0 = random.randint(0, size - w)
    tile[r0 : r0 + h, c0 : c0 + w] = color


def _paint_line(tile: np.ndarray, color: int) -> None:
    size = tile.shape[0]
    if random.random() < 0.5:
        r = random.randint(0, size - 1)
        c0 = random.randint(0, size - 2)
        c1 = random.randint(c0 + 1, size - 1)
        tile[r, c0 : c1 + 1] = color
    else:
        c = random.randint(0, size - 1)
        r0 = random.randint(0, size - 2)
        r1 = random.randint(r0 + 1, size - 1)
        tile[r0 : r1 + 1, c] = color


def _paint_blob(tile: np.ndarray, color: int) -> None:
    size = tile.shape[0]
    target = random.randint(3, min(size * size // 3, 10))
    start = (random.randint(0, size - 1), random.randint(0, size - 1))
    cells = {start}
    while len(cells) < target:
        r, c = random.choice(tuple(cells))
        neighbors = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                neighbors.append((nr, nc))
        cells.add(random.choice(neighbors))
    for r, c in cells:
        tile[r, c] = color


def _make_source_tile(size: int) -> np.ndarray:
    tile = np.full((size, size), BACKGROUND, dtype=int)
    colors = random.sample(PALETTE, random.randint(*SOURCE_COLOR_COUNT_RANGE))
    object_count = random.randint(*SOURCE_OBJECT_COUNT_RANGE)
    painters = [_paint_rect, _paint_line, _paint_blob]
    for _ in range(object_count):
        color = random.choice(colors)
        random.choice(painters)(tile, color)

    # Keep the background dominant so the output background remains unambiguous.
    if np.count_nonzero(tile != BACKGROUND) > (size * size * 3) // 5:
        keep = random.sample(
            [(int(r), int(c)) for r, c in np.argwhere(tile != BACKGROUND)],
            k=(size * size) // 2,
        )
        reduced = np.full_like(tile, BACKGROUND)
        for r, c in keep:
            reduced[r, c] = tile[r, c]
        tile = reduced

    if np.all(tile == BACKGROUND):
        _paint_blob(tile, random.choice(colors))
    return tile


def _payload_shape(size: int) -> list[tuple[int, int]]:
    modes = [
        "hline",
        "vline",
        "diag",
        "anti",
        "plus",
        "box",
        "lshape",
        "scatter",
    ]
    mode = random.choice(modes)
    coords: list[tuple[int, int]] = []

    if mode == "hline":
        row = random.randint(0, size - 1)
        c0 = random.randint(0, size - 3)
        c1 = random.randint(c0 + 2, size - 1)
        coords = [(row, c) for c in range(c0, c1 + 1)]
    elif mode == "vline":
        col = random.randint(0, size - 1)
        r0 = random.randint(0, size - 3)
        r1 = random.randint(r0 + 2, size - 1)
        coords = [(r, col) for r in range(r0, r1 + 1)]
    elif mode == "diag":
        length = random.randint(3, size)
        r0 = random.randint(0, size - length)
        c0 = random.randint(0, size - length)
        coords = [(r0 + k, c0 + k) for k in range(length)]
    elif mode == "anti":
        length = random.randint(3, size)
        r0 = random.randint(0, size - length)
        c1 = random.randint(length - 1, size - 1)
        coords = [(r0 + k, c1 - k) for k in range(length)]
    elif mode == "plus":
        center_r = random.randint(1, size - 2)
        center_c = random.randint(1, size - 2)
        arm = random.randint(1, min(center_r, center_c, size - 1 - center_r, size - 1 - center_c))
        coords = [(center_r, center_c)]
        coords += [(center_r + d, center_c) for d in range(-arm, arm + 1) if d != 0]
        coords += [(center_r, center_c + d) for d in range(-arm, arm + 1) if d != 0]
    elif mode == "box":
        h = random.randint(3, size)
        w = random.randint(3, size)
        r0 = random.randint(0, size - h)
        c0 = random.randint(0, size - w)
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        coords = [(r0, c) for c in range(c0, c1 + 1)]
        coords += [(r1, c) for c in range(c0, c1 + 1)]
        coords += [(r, c0) for r in range(r0 + 1, r1)]
        coords += [(r, c1) for r in range(r0 + 1, r1)]
    elif mode == "lshape":
        h = random.randint(3, size)
        w = random.randint(3, size)
        r0 = random.randint(0, size - h)
        c0 = random.randint(0, size - w)
        coords = [(r0 + k, c0) for k in range(h)]
        coords += [(r0 + h - 1, c0 + k) for k in range(1, w)]
    else:
        count = random.randint(*ERROR_COUNT_RANGE)
        coords = random.sample([(r, c) for r in range(size) for c in range(size)], count)

    coords = list(dict.fromkeys(coords))
    max_count = min(len(coords), random.randint(*ERROR_COUNT_RANGE))
    if len(coords) > max_count:
        coords = random.sample(coords, max_count)
    return coords


def _make_payload(size: int) -> np.ndarray:
    payload = np.full((size, size), BACKGROUND, dtype=int)
    coords = _payload_shape(size)
    color_count = min(len(PALETTE), random.randint(*PAYLOAD_COLOR_COUNT_RANGE))
    colors = random.sample(PALETTE, color_count)
    for r, c in coords:
        payload[r, c] = random.choice(colors)
    if np.all(payload == BACKGROUND):
        r, c = random.randint(0, size - 1), random.randint(0, size - 1)
        payload[r, c] = random.choice(colors)
    return payload


def _build_collage(source: np.ndarray) -> np.ndarray:
    size = source.shape[0]
    collage = np.full((2 * size, 2 * size), BACKGROUND, dtype=int)
    collage[:size, :size] = source
    collage[:size, size:] = np.rot90(source, -1)
    collage[size:, size:] = np.rot90(source, 2)
    collage[size:, :size] = np.rot90(source, 1)
    return collage


def _quadrant_slices(size: int) -> list[tuple[slice, slice]]:
    return [
        (slice(0, size), slice(0, size)),
        (slice(0, size), slice(size, 2 * size)),
        (slice(size, 2 * size), slice(size, 2 * size)),
        (slice(size, 2 * size), slice(0, size)),
    ]


def _inject_errors(source: np.ndarray, payload: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = source.shape[0]
    collage = _build_collage(source)
    expected = np.full((size, size), BACKGROUND, dtype=int)
    slices = _quadrant_slices(size)
    active = [(int(r), int(c)) for r, c in np.argwhere(payload != BACKGROUND)]
    quadrant_choices = list(range(4))
    random.shuffle(quadrant_choices)

    for idx, (source_r, source_c) in enumerate(active):
        error_color = int(payload[source_r, source_c])
        source_color = int(source[source_r, source_c])
        if error_color == source_color:
            choices = [color for color in PALETTE if color != source_color]
            error_color = random.choice(choices)

        if idx < 4:
            quadrant = quadrant_choices[idx]
        else:
            quadrant = random.randint(0, 3)

        local_r, local_c = _orbit_local(source_r, source_c, size)[quadrant]
        rs, cs = slices[quadrant]
        collage[rs.start + local_r, cs.start + local_c] = error_color
        expected[source_r, source_c] = error_color

    return collage, expected


def _generate_instance() -> tuple[np.ndarray, np.ndarray]:
    size = random.choice(list(SOURCE_SIDE_RANGE))
    source = _make_source_tile(size)
    payload = _make_payload(size)
    return _inject_errors(source, payload)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, expected = _generate_instance()
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate valid input")
