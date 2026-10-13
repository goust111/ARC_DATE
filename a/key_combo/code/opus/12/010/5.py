from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MAROON = 9

GLYPH_TO_COLOR = {
    ((1, 1, 1), (0, 1, 0), (0, 1, 0)): BLUE,
    ((1, 0, 0), (0, 1, 1), (0, 1, 1)): RED,
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)): GREEN,
}

MAX_ATTEMPTS = 400


def _decode_glyph(glyph: np.ndarray) -> int:
    key = tuple(tuple(int(v == GREY) for v in row) for row in glyph)
    assert key in GLYPH_TO_COLOR, "unrecognized 3x3 glyph"
    return GLYPH_TO_COLOR[key]


def _infer_min_tile(pattern: np.ndarray, unknown: int | None = None) -> np.ndarray:
    h, w = pattern.shape
    if unknown is None:
        observed = np.argwhere(np.ones_like(pattern, dtype=bool))
    else:
        observed = np.argwhere(pattern != unknown)
    assert len(observed) > 0, "pattern area must contain visible cells"

    best_key = None
    best_tile = None
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            tile = np.full((pr, pc), -1, dtype=int)
            ok = True
            for r, c in observed:
                rr, cc = int(r % pr), int(c % pc)
                value = int(pattern[r, c])
                if tile[rr, cc] == -1:
                    tile[rr, cc] = value
                elif tile[rr, cc] != value:
                    ok = False
                    break
            if ok and np.all(tile != -1):
                key = (pr * pc, pr, pc)
                if best_key is None or key < best_key:
                    best_key = key
                    best_tile = tile.copy()

    assert best_tile is not None, "no consistent complete repeating tile found"
    return best_tile


def _tile_area(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pr, pc = tile.shape
    h, w = shape
    rows = np.arange(h)[:, None] % pr
    cols = np.arange(w)[None, :] % pc
    return tile[rows, cols]


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the left glyph into a color, infer the occluded repeating tile on the right, and recolor the tile foreground.

    Input structure:
    - The grid is black-background outside a pattern area that starts at column 4.
    - The upper-left 3x3 block contains a five-pixel grey glyph encoding one of three colors.
    - Columns 4..end form a binary periodic pattern using black and grey, with some cells hidden by maroon rectangles.

    Rule:
    - Decode the 3x3 glyph: top-T -> blue, down-right elbow -> red, cross -> green.
    - In the pattern area, treat maroon as unknown and infer the smallest complete rectangular tile consistent with all visible cells, using pattern-area-local row/column residues.
    - Tile that rectangle over the full pattern area, then recolor every grey tile cell to the decoded color while keeping black cells black.
    - Leave the glyph area and the separator strip unchanged.

    Key insight:
    - The glyph supplies only the replacement color, while the occluded periodic region supplies only the placement geometry; neither source is sufficient by itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3 and g.shape[1] >= 5, "grid is too small for the task layout"

    color = _decode_glyph(g[:3, :3])
    pattern = g[:, 4:]
    allowed = {BLACK, GREY, MAROON}
    assert set(map(int, np.unique(pattern))) <= allowed, "pattern area must use only black, grey, and maroon"

    tile = _infer_min_tile(pattern, unknown=MAROON)
    repaired = _tile_area(tile, pattern.shape)
    recolored = repaired.copy()
    recolored[recolored == GREY] = color

    out = g.copy()
    out[:, 4:] = recolored
    return out


def _sample_grid_shape() -> tuple[int, int]:
    height = random.randint(9, 16)
    pattern_width = random.randint(6, 13)
    return height, pattern_width + 4


def _sample_periods(height: int, pattern_width: int) -> tuple[int, int]:
    pr = random.randint(2, min(5, height - 2))
    pc = random.randint(2, min(6, pattern_width - 2))
    return pr, pc


def _sample_tile(pr: int, pc: int, pattern_shape: tuple[int, int]) -> np.ndarray | None:
    area = pr * pc
    min_grey = max(2, area // 4)
    max_grey = min(area - 2, max(min_grey, (area * 3) // 5))
    if min_grey > max_grey:
        return None

    coords = [(r, c) for r in range(pr) for c in range(pc)]
    for _ in range(200):
        grey_count = random.randint(min_grey, max_grey)
        grey_cells = set(random.sample(coords, grey_count))
        tile = np.zeros((pr, pc), dtype=int)
        for r, c in grey_cells:
            tile[r, c] = GREY

        if len({r for r, _ in grey_cells}) < min(2, pr):
            continue
        if len({c for _, c in grey_cells}) < min(2, pc):
            continue

        full = _tile_area(tile, pattern_shape)
        inferred = _infer_min_tile(full, unknown=None)
        if inferred.shape == tile.shape and np.array_equal(inferred, tile):
            return tile
    return None


def _residue_counts(height: int, width: int, pr: int, pc: int, r0: int = 0, c0: int = 0) -> np.ndarray:
    counts = np.zeros((pr, pc), dtype=int)
    rr = np.arange(r0, r0 + height) % pr
    cc = np.arange(c0, c0 + width) % pc
    np.add.at(counts, (rr[:, None], cc[None, :]), 1)
    return counts


def _place_occlusions(pattern: np.ndarray, pr: int, pc: int) -> np.ndarray | None:
    height, width = pattern.shape
    total_counts = _residue_counts(height, width, pr, pc)
    hidden = np.zeros((height, width), dtype=bool)
    hidden_counts = np.zeros((pr, pc), dtype=int)
    rect_count = random.randint(2, 4)

    for _ in range(rect_count):
        rect_h = random.randint(1, min(4, height))
        rect_w = random.randint(1, min(5, width))
        choices = []
        for r0 in range(height - rect_h + 1):
            for c0 in range(width - rect_w + 1):
                if hidden[r0 : r0 + rect_h, c0 : c0 + rect_w].any():
                    continue
                add_counts = _residue_counts(rect_h, rect_w, pr, pc, r0, c0)
                if np.any(hidden_counts + add_counts >= total_counts):
                    continue
                choices.append((r0, c0, add_counts))
        if not choices:
            return None
        r0, c0, add_counts = random.choice(choices)
        hidden[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        hidden_counts += add_counts

    area = height * width
    hidden_area = int(hidden.sum())
    if hidden_area < max(3, area // 12) or hidden_area > max(8, area // 3):
        return None
    if not np.any(hidden & (pattern == GREY)):
        return None
    if not np.any(~hidden & (pattern == GREY)):
        return None
    if np.any(hidden_counts >= total_counts):
        return None
    return hidden


def _expected_output(inp: np.ndarray, tile: np.ndarray, color: int) -> np.ndarray:
    out = inp.copy()
    repaired = _tile_area(tile, inp[:, 4:].shape)
    recolored = repaired.copy()
    recolored[recolored == GREY] = color
    out[:, 4:] = recolored
    return out


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    height, width = _sample_grid_shape()
    pattern_width = width - 4
    pr, pc = _sample_periods(height, pattern_width)
    tile = _sample_tile(pr, pc, (height, pattern_width))
    if tile is None:
        return None

    pattern = _tile_area(tile, (height, pattern_width))
    hidden = _place_occlusions(pattern, pr, pc)
    if hidden is None:
        return None

    glyph_key, color = random.choice(list(GLYPH_TO_COLOR.items()))
    inp = np.zeros((height, width), dtype=int)
    inp[:3, :3] = np.array(glyph_key, dtype=int) * GREY
    pattern_view = inp[:, 4:]
    pattern_view[:] = pattern
    pattern_view[hidden] = MAROON

    expected = _expected_output(inp, tile, color)

    inferred_tile = _infer_min_tile(inp[:, 4:], unknown=MAROON)
    if inferred_tile.shape != tile.shape or not np.array_equal(inferred_tile, tile):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    if np.array_equal(inp, expected):
        return None
    if int(np.count_nonzero(expected[:, 4:] == color)) == 0:
        return None
    if set(map(int, np.unique(inp[:, 4:]))) != {BLACK, GREY, MAROON}:
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input for puzzle 5")
