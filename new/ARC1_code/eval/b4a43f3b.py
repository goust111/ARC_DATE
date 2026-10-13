from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the compact upper template into every marked lower mask cell as an expanded block image.

    Input structure
    - The grid has one solid grey separator row.
    - Above it is a template region made of monochrome 2x2 blocks.
    - Below it is a same-size mask containing only background and red marker pixels.

    Rule
    - Compress each 2x2 block in the upper region to one color, giving a small tile.
    - Build a blank output whose coarse cell grid is the lower mask and whose cell size is the tile size.
    - For every red marker pixel in the lower mask, paste the tile into the corresponding output block.

    Key insight
    The top half encodes the motif at half resolution, while the lower red mask encodes which coarse cells receive that motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    separator_rows = np.flatnonzero(np.all(g == Color.GREY, axis=1))
    assert len(separator_rows) == 1, "expected exactly one solid grey separator row"
    sep = int(separator_rows[0])

    top = g[:sep]
    mask = g[sep + 1 :]
    assert top.shape == mask.shape, "template and marker mask must have the same shape"
    assert top.shape[0] > 0 and top.shape[1] > 0, "template region must be non-empty"
    assert top.shape[0] % 2 == 0 and top.shape[1] % 2 == 0, "template must split into 2x2 blocks"

    mask_colors = set(map(int, np.unique(mask)))
    assert mask_colors <= {int(bg), Color.RED}, "marker mask must contain only background and red"
    assert Color.RED in mask_colors, "expected at least one red marker"

    tile_h, tile_w = top.shape[0] // 2, top.shape[1] // 2
    tile = np.empty((tile_h, tile_w), dtype=int)
    for tr in range(tile_h):
        for tc in range(tile_w):
            block = top[2 * tr : 2 * tr + 2, 2 * tc : 2 * tc + 2]
            color = int(block[0, 0])
            assert np.all(block == color), "each template block must be monochrome"
            tile[tr, tc] = color

    out = np.full((mask.shape[0] * tile_h, mask.shape[1] * tile_w), int(bg), dtype=int)
    for mr, mc in np.argwhere(mask == Color.RED):
        r0 = int(mr) * tile_h
        c0 = int(mc) * tile_w
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile

    return out

# --------------- generator (o5x) ---------------

# Grid dimensions and block expansion.
TILE_SHAPE = (3, 3)
MASK_SHAPE = (6, 6)
TEMPLATE_SCALE = 2
SEPARATOR_ROWS = 1

# Colors used by the generated input.
MARKER_COLOR = Color.RED
SEPARATOR_COLOR = Color.GREY
TILE_PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

# Tile and mask sampling controls.
TILE_FILLED_RANGE = (4, 7)
TILE_DISTINCT_RANGE = (2, 4)
MAX_CELLS_PER_TILE_COLOR = 4
MASK_RED_RANGE = (5, 12)
MASK_COMPONENT_RANGE = (1, 3)
EMPTY_LABEL = -1
NEIGHBOR4_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Retry budget for rejected masks or oracle mismatches.
GEN_RETRY_LIMIT = 300


def _gen_neighbors4(r: int, c: int):
    for dr, dc in NEIGHBOR4_DELTAS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < MASK_SHAPE[0] and 0 <= nc < MASK_SHAPE[1]:
            yield nr, nc


def _gen_split_positive(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    vals = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(cut - prev)
        prev = cut
    return vals


def _gen_color_counts(total: int, distinct: int) -> list[int]:
    counts = [1] * distinct
    for _ in range(total - distinct):
        choices = [i for i, count in enumerate(counts) if count < MAX_CELLS_PER_TILE_COLOR]
        counts[random.choice(choices)] += 1
    random.shuffle(counts)
    return counts


def _gen_tile() -> np.ndarray:
    tile = np.zeros(TILE_SHAPE, dtype=int)
    filled = random.randint(*TILE_FILLED_RANGE)
    distinct = random.randint(TILE_DISTINCT_RANGE[0], min(TILE_DISTINCT_RANGE[1], filled))
    colors = random.sample(TILE_PALETTE, distinct)
    positions = random.sample([(r, c) for r in range(TILE_SHAPE[0]) for c in range(TILE_SHAPE[1])], filled)
    assignments = []
    for color, count in zip(colors, _gen_color_counts(filled, distinct)):
        assignments.extend([color] * count)
    random.shuffle(assignments)
    for (r, c), color in zip(positions, assignments):
        tile[r, c] = color
    return tile


def _gen_touching_other(labels: np.ndarray, r: int, c: int, current: int) -> bool:
    return any(0 <= labels[nr, nc] != current for nr, nc in _gen_neighbors4(r, c))


def _gen_seed_cells(labels: np.ndarray, comp_id: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(MASK_SHAPE[0])
        for c in range(MASK_SHAPE[1])
        if labels[r, c] < 0 and not _gen_touching_other(labels, r, c, comp_id)
    ]


def _gen_frontier_cells(labels: np.ndarray, comp_id: int) -> list[tuple[int, int]]:
    frontier = []
    for r in range(MASK_SHAPE[0]):
        for c in range(MASK_SHAPE[1]):
            if labels[r, c] >= 0:
                continue
            if any(labels[nr, nc] == comp_id for nr, nc in _gen_neighbors4(r, c)) and not _gen_touching_other(labels, r, c, comp_id):
                frontier.append((r, c))
    return frontier


def _gen_mask() -> np.ndarray | None:
    total = random.randint(*MASK_RED_RANGE)
    comp_count = random.randint(MASK_COMPONENT_RANGE[0], min(MASK_COMPONENT_RANGE[1], total))
    sizes = sorted(_gen_split_positive(total, comp_count), reverse=True)
    labels = np.full(MASK_SHAPE, EMPTY_LABEL, dtype=int)

    for comp_id, target in enumerate(sizes):
        seeds = _gen_seed_cells(labels, comp_id)
        if not seeds:
            return None
        seed_r, seed_c = random.choice(seeds)
        labels[seed_r, seed_c] = comp_id

        while int(np.sum(labels == comp_id)) < target:
            frontier = _gen_frontier_cells(labels, comp_id)
            if not frontier:
                return None
            nr, nc = random.choice(frontier)
            labels[nr, nc] = comp_id

    mask = np.zeros(MASK_SHAPE, dtype=int)
    mask[labels >= 0] = MARKER_COLOR
    return mask


def _gen_passes_oracle(inp: np.ndarray, tile: np.ndarray, mask: np.ndarray) -> bool:
    expected = np.kron(mask == MARKER_COLOR, tile).astype(int)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    tile = _gen_tile()
    mask = _gen_mask()
    if mask is None:
        return None

    template = np.repeat(np.repeat(tile, TEMPLATE_SCALE, axis=0), TEMPLATE_SCALE, axis=1)
    separator = np.full((SEPARATOR_ROWS, MASK_SHAPE[1]), SEPARATOR_COLOR, dtype=int)
    inp = np.vstack([template, separator, mask])
    if not _gen_passes_oracle(inp, tile, mask):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for b4a43f3b")
