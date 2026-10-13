import numpy as np


BACKGROUND = 0
BLUE = 1
FOREGROUND_COLORS = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Two hand-picked seed layouts that reliably create surviving blue markers on
# two dense diagonals while leaving other selected diagonals sparse.
PATTERN_A = np.array(
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (2, 2),
        (3, 1),
        (3, 2),
    ],
    dtype=int,
)
PATTERN_B = np.array(
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 3),
        (2, 2),
        (2, 3),
    ],
    dtype=int,
)
PATTERNS = (PATTERN_A, PATTERN_B)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _build_grid(shape: tuple[int, int], cells: list[tuple[int, int]], colors: np.ndarray) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for (r, c), color in zip(cells, colors):
        grid[r, c] = int(color)
    return grid


def _prefilter_output(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((2 * h, 2 * w), dtype=int)
    offsets = {int(r - c) for r, c in np.argwhere(grid != BACKGROUND)}

    for r in range(h):
        for c in range(w):
            rr = 2 * r
            cc = 2 * c
            value = int(grid[r, c])
            if value != BACKGROUND:
                out[rr : rr + 2, cc : cc + 2] = value
            elif (r - c) in offsets:
                out[rr, cc] = BLUE
                out[rr + 1, cc + 1] = BLUE
    return out


def _diagonal_stats(occupancy: np.ndarray) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    counts: dict[int, int] = {}
    marker_cells: dict[int, int] = {}
    supported_cells: dict[int, int] = {}
    h, w = occupancy.shape

    for r, c in np.argwhere(occupancy):
        offset = int(r - c)
        counts[offset] = counts.get(offset, 0) + 1

    for offset in counts:
        marker_cells[offset] = 0
        supported_cells[offset] = 0

    for r in range(h):
        for c in range(w):
            if occupancy[r, c]:
                continue
            offset = int(r - c)
            if offset not in counts:
                continue

            marker_cells[offset] += 1
            supported_tl = (
                r > 0
                and c > 0
                and occupancy[r - 1, c]
                and occupancy[r, c - 1]
            )
            supported_br = (
                r + 1 < h
                and c + 1 < w
                and occupancy[r + 1, c]
                and occupancy[r, c + 1]
            )
            if supported_tl or supported_br:
                supported_cells[offset] += 1

    return counts, marker_cells, supported_cells


def _valid_occupancy(occupancy: np.ndarray) -> bool:
    foreground_count = int(np.count_nonzero(occupancy))
    if not 6 <= foreground_count <= 8:
        return False

    counts, marker_cells, supported_cells = _diagonal_stats(occupancy)
    dense_supported = [
        offset
        for offset, count in counts.items()
        if count >= 2 and marker_cells[offset] > 0 and supported_cells[offset] > 0
    ]
    sparse_clean = [
        offset
        for offset, count in counts.items()
        if count == 1 and marker_cells[offset] > 0 and supported_cells[offset] == 0
    ]
    sparse_supported = [
        offset
        for offset, count in counts.items()
        if count == 1 and supported_cells[offset] > 0
    ]

    return len(dense_supported) >= 2 and len(sparse_clean) >= 1 and not sparse_supported


def _cells_far_enough(cells: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    r, c = candidate
    for rr, cc in cells:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _sample_colors(rng: np.random.Generator, count: int) -> np.ndarray:
    distinct = int(rng.integers(2, min(5, count) + 1))
    palette = rng.choice(FOREGROUND_COLORS, size=distinct, replace=False)
    colors = np.empty(count, dtype=int)
    colors[:distinct] = palette
    if count > distinct:
        colors[distinct:] = rng.choice(palette, size=count - distinct, replace=True)
    rng.shuffle(colors)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Double the grid, mark selected NW-SE diagonals in blue, then erase unsupported blue pixels.

    Input structure:
    - The input is a black background grid with several non-blue foreground pixels.
    - Foreground pixels may use multiple colors, and blue appears only as an output marker.
    - Some NW-SE diagonals contain multiple foreground pixels, while others contain only one.

    Rule:
    - Create a black output canvas with double the input height and width.
    - Replace each foreground input pixel with a solid 2x2 block of the same color.
    - For each background input cell whose row-minus-column offset matches any foreground pixel, draw a blue main diagonal in its 2x2 block.
    - Then inspect every blue output pixel once and erase it if it has fewer than two non-black 4-neighbors.

    Key insight:
    - The first stage marks every selected NW-SE diagonal, and the one-pass 4-neighbor filter keeps only the structurally supported blue pixels, making the denser diagonal pattern stand out.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pre = _prefilter_output(g)
    out = pre.copy()

    for r, c in np.argwhere(pre == BLUE):
        neighbors = 0
        for dr, dc in DIR4:
            rr = int(r + dr)
            cc = int(c + dc)
            if _in_bounds(pre.shape, rr, cc) and pre[rr, cc] != BACKGROUND:
                neighbors += 1
        if neighbors < 2:
            out[r, c] = BACKGROUND

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(500):
        h = int(rng.integers(6, 13))
        w = int(rng.integers(6, 13))
        pattern = PATTERNS[int(rng.integers(0, len(PATTERNS)))]

        ph = int(pattern[:, 0].max()) + 1
        pw = int(pattern[:, 1].max()) + 1
        if ph > h or pw > w:
            continue

        r0 = int(rng.integers(0, h - ph + 1))
        c0 = int(rng.integers(0, w - pw + 1))
        cells = [(r0 + int(r), c0 + int(c)) for r, c in pattern]

        target_count = int(rng.integers(6, 9))
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in cells
        ]
        rng.shuffle(candidates)

        for candidate in candidates:
            if len(cells) >= target_count:
                break
            if not _cells_far_enough(cells, candidate):
                continue

            trial_cells = cells + [candidate]
            occ = _build_grid((h, w), trial_cells, np.ones(len(trial_cells), dtype=int))
            if _valid_occupancy(occ):
                cells = trial_cells

        cells = sorted(cells)
        occupancy = _build_grid((h, w), cells, np.ones(len(cells), dtype=int))
        if not _valid_occupancy(occupancy):
            continue

        colors = _sample_colors(rng, len(cells))
        grid = _build_grid((h, w), cells, colors)
        out = main(grid)
        if np.count_nonzero(out == BLUE) == 0:
            continue
        if np.count_nonzero(out == BLUE) >= np.count_nonzero(_prefilter_output(grid) == BLUE):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 7")
