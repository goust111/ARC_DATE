import itertools
import random

import numpy as np


BACKGROUND = 0
BLOCK_COLOR = 8
RED = 2
GREEN = 3
BLUE = 1
MAGENTA = 6
MARKER_COLORS = (RED, GREEN)
OUTLINE_COLOR = {RED: BLUE, GREEN: MAGENTA}

# Actual ring cells around a 2x2 block, expressed relative to the block's top-left.
# Each entry is (ring_row_offset, ring_col_offset, outward_dr, outward_dc).
RING_SPECS = [
    (-1, -1, -1, -1),
    (-1, 0, -1, 0),
    (-1, 1, -1, 0),
    (-1, 2, -1, 1),
    (0, -1, 0, -1),
    (1, -1, 0, -1),
    (0, 2, 0, 1),
    (1, 2, 0, 1),
    (2, -1, 1, -1),
    (2, 0, 1, 0),
    (2, 1, 1, 0),
    (2, 2, 1, 1),
]

_OUTPUT_SIGS: set[tuple] = set()


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _outline_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [
        (rr, cc)
        for rr in range(r - 1, r + 2)
        for cc in range(c - 1, c + 2)
        if not (rr == r and cc == c)
    ]


def _find_block_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == BLOCK_COLOR)
    assert coords.shape[0] == 4, "expected exactly four block-colored cells"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "block must be 2x2"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BLOCK_COLOR), "block must be filled"
    return r0, c0, r1, c1


def _project_to_ring(r: int, c: int, block_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = block_bbox
    assert not (r0 <= r <= r1 and c0 <= c <= c1), "marker cannot start inside the block"

    if r < r0:
        nr = r0 - 1
    elif r > r1:
        nr = r1 + 1
    else:
        nr = r

    if c < c0:
        nc = c0 - 1
    elif c > c1:
        nc = c1 + 1
    else:
        nc = c

    dr = abs(r - nr)
    dc = abs(c - nc)
    valid = (dr == 0 and dc > 0) or (dc == 0 and dr > 0) or (dr == dc and dr > 0)
    assert valid, "marker must align with the block by row, column, or 45-degree diagonal"
    return nr, nc


def _build_output(
    shape: tuple[int, int],
    block_bbox: tuple[int, int, int, int],
    destinations: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    r0, c0, r1, c1 = block_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BLOCK_COLOR

    for center, color in zip(destinations, colors):
        out[center] = int(color)

    for center, color in zip(destinations, colors):
        outline_color = OUTLINE_COLOR[int(color)]
        for rr, cc in _outline_cells(center):
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            assert out[rr, cc] == BACKGROUND, "outlined 3x3 neighborhoods must not overlap"
            out[rr, cc] = outline_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each aligned marker into the 2x2 block's neighbor ring, then outline it with a color chosen by the marker color.

    Input structure:
    - The background is black.
    - A cyan 2x2 filled block sits near the middle of the grid.
    - Three or more isolated marker pixels are red and/or green, and each one lies on a row, column, or 45-degree diagonal ray pointing toward the 2x2 block.

    Rule:
    - Keep the cyan 2x2 block fixed.
    - Remove each marker from its original location, then slide it straight toward the block until it reaches the matching one-cell ring position around the block.
    - Keep the slid marker's own color at that ring position.
    - Around every slid green marker, draw a centered 3x3 hollow square in magenta; around every slid red marker, draw the same 3x3 hollow square in blue.
    - If an outline cell would overlap the cyan 2x2 block, the cyan block stays unchanged.

    Key insight:
    - The marker's final position is determined only by ray-alignment with the central 2x2 block, while the outline color is determined independently by the marker's original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    block_bbox = _find_block_bbox(g)
    h, w = map(int, g.shape)

    marker_positions = []
    marker_colors = []
    for r_raw, c_raw in np.argwhere(np.isin(g, MARKER_COLORS)):
        r, c = int(r_raw), int(c_raw)
        color = int(g[r, c])
        marker_positions.append((r, c))
        marker_colors.append(color)

    assert len(marker_positions) >= 1, "expected at least one marker"

    out = g.copy()
    for r, c in marker_positions:
        out[r, c] = BACKGROUND

    destinations = []
    used_destinations = set()
    for (r, c), color in zip(marker_positions, marker_colors):
        nr, nc = _project_to_ring(r, c, block_bbox)
        assert 0 <= nr < h and 0 <= nc < w, "destination must stay in-bounds"
        assert g[nr, nc] != BLOCK_COLOR, "marker destination cannot replace the block"
        assert (nr, nc) not in used_destinations, "markers must slide to distinct ring cells"
        used_destinations.add((nr, nc))
        destinations.append((nr, nc))
        out[nr, nc] = color

    for i, a in enumerate(destinations):
        for b in destinations[i + 1 :]:
            assert _chebyshev(a, b) >= 3, "3x3 neighborhoods must be disjoint"

    r0, c0, r1, c1 = block_bbox
    for center, color in zip(destinations, marker_colors):
        outline_color = OUTLINE_COLOR[color]
        for rr, cc in _outline_cells(center):
            if not (0 <= rr < h and 0 <= cc < w):
                raise AssertionError("3x3 outline must fit inside the grid")
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            assert out[rr, cc] == BACKGROUND, "outlines must not overlap each other"
            out[rr, cc] = outline_color

    return out


def _valid_destination_index_sets() -> dict[int, list[tuple[int, ...]]]:
    index_sets: dict[int, list[tuple[int, ...]]] = {3: [], 4: []}
    rel_cells = [(spec[0], spec[1]) for spec in RING_SPECS]
    for size in (3, 4):
        for combo in itertools.combinations(range(len(RING_SPECS)), size):
            cells = [rel_cells[idx] for idx in combo]
            if all(_chebyshev(a, b) >= 3 for i, a in enumerate(cells) for b in cells[i + 1 :]):
                index_sets[size].append(combo)
    return index_sets


VALID_DESTINATION_INDEX_SETS = _valid_destination_index_sets()
CORNER_INDICES = {0, 3, 8, 11}


def _choose_shape_and_block() -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    h = random.randint(11, 16)
    w = random.randint(11, 16)

    row_choices = [r for r in range(2, h - 3) if abs((r + 0.5) - (h / 2.0)) <= 2.5]
    col_choices = [c for c in range(2, w - 3) if abs((c + 0.5) - (w / 2.0)) <= 2.5]
    r0 = random.choice(row_choices)
    c0 = random.choice(col_choices)
    return (h, w), (r0, c0, r0 + 1, c0 + 1)


def _choose_destination_indices() -> tuple[int, ...]:
    if random.random() < 0.35:
        return random.choice(VALID_DESTINATION_INDEX_SETS[4])

    candidates = list(VALID_DESTINATION_INDEX_SETS[3])
    mixed = [combo for combo in candidates if 0 < sum(idx not in CORNER_INDICES for idx in combo) < 3]
    side_heavy = [combo for combo in candidates if sum(idx not in CORNER_INDICES for idx in combo) >= 2]

    roll = random.random()
    if roll < 0.45 and side_heavy:
        return random.choice(side_heavy)
    if roll < 0.80 and mixed:
        return random.choice(mixed)
    return random.choice(candidates)


def _choose_marker_colors(count: int) -> list[int]:
    if count == 1:
        return [random.choice(MARKER_COLORS)]

    if random.random() < 0.8:
        colors = [random.choice(MARKER_COLORS) for _ in range(count)]
        if RED not in colors:
            colors[random.randrange(count)] = RED
        if GREEN not in colors:
            colors[random.randrange(count)] = GREEN
        return colors

    chosen = random.choice(MARKER_COLORS)
    return [chosen] * count


def _max_source_distance(shape: tuple[int, int], dest: tuple[int, int], dr: int, dc: int) -> int:
    h, w = shape
    row_room = h if dr == 0 else (dest[0] if dr < 0 else (h - 1 - dest[0]))
    col_room = w if dc == 0 else (dest[1] if dc < 0 else (w - 1 - dest[1]))
    return int(min(row_room, col_room, 8))


def _choose_source_positions(
    shape: tuple[int, int],
    block_bbox: tuple[int, int, int, int],
    dest_indices: tuple[int, ...],
) -> list[tuple[int, int]] | None:
    h, w = shape
    r0, c0, _r1, _c1 = block_bbox
    specs = []
    for idx in dest_indices:
        ro, co, dr, dc = RING_SPECS[idx]
        dest = (r0 + ro, c0 + co)
        max_dist = _max_source_distance(shape, dest, dr, dc)
        if max_dist < 1:
            return None
        specs.append((dest, dr, dc, max_dist))

    occupied: set[tuple[int, int]] = set()
    sources: list[tuple[int, int]] = []
    order = list(range(len(specs)))
    random.shuffle(order)
    assigned: dict[int, tuple[int, int]] = {}
    for pos in order:
        dest, dr, dc, max_dist = specs[pos]
        distances = list(range(1, max_dist + 1))
        random.shuffle(distances)
        distances.sort(key=lambda d: (d <= 2, random.random()))

        placed = False
        for dist in distances:
            src = (dest[0] + dr * dist, dest[1] + dc * dist)
            sr, sc = src
            if not (0 <= sr < h and 0 <= sc < w):
                continue
            if (sr, sc) in occupied:
                continue
            if r0 <= sr <= r0 + 1 and c0 <= sc <= c0 + 1:
                continue
            assigned[pos] = src
            occupied.add(src)
            placed = True
            break
        if not placed:
            return None

    for i in range(len(specs)):
        sources.append(assigned[i])
    return sources


def _semantic_check(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    block_bbox = _find_block_bbox(inp)
    markers = [(int(r), int(c), int(inp[r, c])) for r, c in np.argwhere(np.isin(inp, MARKER_COLORS))]
    if not (3 <= len(markers) <= 4):
        return False

    destinations = []
    for r, c, color in markers:
        if color not in OUTLINE_COLOR:
            return False
        nr, nc = _project_to_ring(r, c, block_bbox)
        destinations.append((nr, nc))

    if len(set(destinations)) != len(destinations):
        return False
    if any(_chebyshev(a, b) < 3 for i, a in enumerate(destinations) for b in destinations[i + 1 :]):
        return False

    h, w = inp.shape
    for center in destinations:
        r, c = center
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(inp, expected):
        return False

    # Every marker must contribute at least one non-block outline pixel.
    r0, c0, r1, c1 = block_bbox
    for center, (_, _, color) in zip(destinations, markers):
        outline_color = OUTLINE_COLOR[color]
        non_block_cells = [
            (rr, cc)
            for rr, cc in _outline_cells(center)
            if not (r0 <= rr <= r1 and c0 <= cc <= c1)
        ]
        if not non_block_cells:
            return False
        if not all(expected[rr, cc] == outline_color for rr, cc in non_block_cells):
            return False

    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    shape, block_bbox = _choose_shape_and_block()
    dest_indices = _choose_destination_indices()
    sources = _choose_source_positions(shape, block_bbox, dest_indices)
    if sources is None:
        return None

    r0, c0, _r1, _c1 = block_bbox
    destinations = [(r0 + RING_SPECS[idx][0], c0 + RING_SPECS[idx][1]) for idx in dest_indices]
    colors = _choose_marker_colors(len(destinations))

    out = _build_output(shape, block_bbox, destinations, colors)
    inp = np.full(shape, BACKGROUND, dtype=int)
    inp[block_bbox[0] : block_bbox[2] + 1, block_bbox[1] : block_bbox[3] + 1] = BLOCK_COLOR
    for (sr, sc), color in zip(sources, colors):
        if inp[sr, sc] != BACKGROUND:
            return None
        inp[sr, sc] = int(color)

    if not _semantic_check(inp, out):
        return None

    out_sig = (tuple(out.shape), tuple(map(int, out.ravel())))
    if out_sig in _OUTPUT_SIGS:
        return None
    _OUTPUT_SIGS.add(out_sig)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError("failed to generate a valid input")
