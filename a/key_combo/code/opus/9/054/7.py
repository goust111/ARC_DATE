import hashlib
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

TEMPLATE_COLORS = (BLUE, RED, GREEN, PINK, TEAL, MAROON)
GEN_MAX_ATTEMPTS = 800
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _full_grey_rows(grid: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.all(grid == GREY, axis=1))


def _compress_template(top: np.ndarray) -> np.ndarray:
    h, w = top.shape
    assert h > 0 and w > 0 and h % 2 == 0 and w % 2 == 0
    tile = np.empty((h // 2, w // 2), dtype=int)
    for r in range(0, h, 2):
        for c in range(0, w, 2):
            block = top[r : r + 2, c : c + 2]
            color = int(block[0, 0])
            assert np.all(block == color), "template must be built from monochrome 2x2 blocks"
            tile[r // 2, c // 2] = color
    return tile


def _expand_template(tile: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)


def _corner_pair(lower: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = lower.shape
    yellow = [tuple(map(int, p)) for p in np.argwhere(lower == YELLOW)]
    assert len(yellow) == 2, "expected exactly two yellow corner markers"
    diag_a = {(0, 0), (h - 1, w - 1)}
    diag_b = {(0, w - 1), (h - 1, 0)}
    yellow_set = set(yellow)
    assert yellow_set in (diag_a, diag_b), "yellow markers must occupy one diagonal pair of corners"
    return tuple(sorted(yellow))


def _nearest_chebyshev_visibility(shape: tuple[int, int], corners: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    h, w = shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    (r1, c1), (r2, c2) = corners
    d1 = np.maximum(np.abs(rows - r1), np.abs(cols - c1))
    d2 = np.maximum(np.abs(rows - r2), np.abs(cols - c2))
    return np.minimum(d1, d2) % 2 == 0


def _single_corner_visibility(shape: tuple[int, int], corner: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    r0, c0 = corner
    return np.maximum(np.abs(rows - r0), np.abs(cols - c0)) % 2 == 0


def _nearest_manhattan_visibility(shape: tuple[int, int], corners: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    h, w = shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    (r1, c1), (r2, c2) = corners
    d1 = np.abs(rows - r1) + np.abs(cols - c1)
    d2 = np.abs(rows - r2) + np.abs(cols - c2)
    return np.minimum(d1, d2) % 2 == 0


def _selected_blocks(lower: np.ndarray, tile_h: int, tile_w: int) -> list[tuple[int, int]]:
    assert lower.shape == (2 * tile_h, 2 * tile_w)
    blocks: list[tuple[int, int]] = []
    for br in range(2):
        for bc in range(2):
            sub = lower[br * tile_h : (br + 1) * tile_h, bc * tile_w : (bc + 1) * tile_w]
            if np.any(sub == ORANGE):
                blocks.append((br, bc))
    return blocks


def _stamp_output(
    tile: np.ndarray,
    lower_shape: tuple[int, int],
    blocks: list[tuple[int, int]],
    visibility: np.ndarray,
) -> np.ndarray:
    tile_h, tile_w = tile.shape
    out = np.zeros(lower_shape, dtype=int)
    for br, bc in blocks:
        r0 = br * tile_h
        c0 = bc * tile_w
        mask = visibility[r0 : r0 + tile_h, c0 : c0 + tile_w]
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = np.where(mask, tile, BLACK)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the compressed upper template into marked lower blocks, but reveal only pixels on even nearest-corner Chebyshev rings.

    Input structure
    - A single solid grey separator row splits the grid into an upper template region and a lower marker region of the same size.
    - The upper region is made of monochrome 2x2 blocks using black plus non-grey template colors.
    - The lower region uses black background, exactly two yellow corner markers on one diagonal, and orange mask pixels inside coarse blocks.

    Rule
    - Compress each monochrome 2x2 block in the upper region into one tile pixel.
    - The lower region is a 2x2 coarse grid whose block size equals the compressed tile size.
    - For every pixel in the lower region, compute the Chebyshev distance to the nearer yellow corner; only even distances are visible.
    - For each coarse block containing at least one orange marker, stamp the compressed tile into that block, but keep only the tile pixels landing on visible positions; all other output pixels stay black.

    Key insight
    - The result is the pixelwise intersection of two independent structures: a compressed stamp from the top and an even-ring visibility mask induced by the lower yellow corners.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_rows = _full_grey_rows(g)
    assert len(sep_rows) == 1, "expected exactly one full grey separator row"
    sep = int(sep_rows[0])

    top = g[:sep]
    lower = g[sep + 1 :]
    assert top.shape == lower.shape, "top and bottom regions must have the same shape"

    tile = _compress_template(top)
    tile_h, tile_w = tile.shape
    assert lower.shape == (2 * tile_h, 2 * tile_w), "lower region must align to a 2x2 coarse grid of tile-sized blocks"

    lower_colors = set(map(int, np.unique(lower)))
    assert lower_colors <= {BLACK, YELLOW, ORANGE}, "lower region may contain only black, yellow, and orange"
    blocks = _selected_blocks(lower, tile_h, tile_w)
    assert blocks, "expected at least one orange-marked coarse block"

    corners = _corner_pair(lower)
    visibility = _nearest_chebyshev_visibility(lower.shape, corners)
    return _stamp_output(tile, lower.shape, blocks, visibility)


def _sample_tile_shape() -> tuple[int, int]:
    mode = random.choice(("odd_h", "odd_w", "both_odd"))
    odd_h = (3, 5, 7)
    even_h = (4, 6)
    odd_w = (3, 5, 7, 9)
    even_w = (4, 6, 8)
    if mode == "odd_h":
        return random.choice(odd_h), random.choice(even_w)
    if mode == "odd_w":
        return random.choice(even_h), random.choice(odd_w)
    return random.choice(odd_h), random.choice(odd_w)


def _sample_tile(tile_h: int, tile_w: int) -> np.ndarray:
    area = tile_h * tile_w
    min_filled = max(6, area // 3)
    max_filled = min(area - 1, (2 * area) // 3 + 2)
    filled = random.randint(min_filled, max_filled)
    color_count = random.randint(2, min(4, filled))
    colors = random.sample(TEMPLATE_COLORS, color_count)
    positions = random.sample([(r, c) for r in range(tile_h) for c in range(tile_w)], filled)

    tile = np.zeros((tile_h, tile_w), dtype=int)
    counts = [1] * color_count
    for _ in range(filled - color_count):
        counts[random.randrange(color_count)] += 1
    random.shuffle(counts)

    color_stream: list[int] = []
    for color, count in zip(colors, counts):
        color_stream.extend([color] * count)
    random.shuffle(color_stream)

    for (r, c), color in zip(positions, color_stream):
        tile[r, c] = color
    return tile


def _corner_blocks(diagonal: str) -> tuple[set[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]:
    if diagonal == "tl_br":
        return {(0, 0), (1, 1)}, ((0, 0), (1, 1))
    return {(0, 1), (1, 0)}, ((0, 1), (1, 0))


def _place_lower_markers(
    tile_h: int,
    tile_w: int,
    diagonal: str,
    blocks: list[tuple[int, int]],
) -> np.ndarray:
    lower = np.zeros((2 * tile_h, 2 * tile_w), dtype=int)
    if diagonal == "tl_br":
        corners = ((0, 0), (2 * tile_h - 1, 2 * tile_w - 1))
    else:
        corners = ((0, 2 * tile_w - 1), (2 * tile_h - 1, 0))
    for r, c in corners:
        lower[r, c] = YELLOW

    marker_r = tile_h // 2
    marker_c = tile_w // 2
    for br, bc in blocks:
        lower[br * tile_h + marker_r, bc * tile_w + marker_c] = ORANGE
    return lower


def _sample_blocks(diagonal: str) -> list[tuple[int, int]]:
    all_blocks = [(0, 0), (0, 1), (1, 0), (1, 1)]
    yellow_blocks, _ = _corner_blocks(diagonal)
    dark_blocks = [b for b in all_blocks if b not in yellow_blocks]
    count = random.choice((2, 3, 4))
    while True:
        blocks = random.sample(all_blocks, count)
        block_set = set(blocks)
        if block_set & yellow_blocks and block_set & set(dark_blocks):
            return sorted(blocks)


def _block_variants(
    tile: np.ndarray,
    visibility: np.ndarray,
    blocks: list[tuple[int, int]],
) -> list[np.ndarray]:
    tile_h, tile_w = tile.shape
    variants = []
    for br, bc in blocks:
        r0 = br * tile_h
        c0 = bc * tile_w
        variants.append(np.where(visibility[r0 : r0 + tile_h, c0 : c0 + tile_w], tile, BLACK))
    return variants


def _validate_semantics(inp: np.ndarray) -> tuple[bool, np.ndarray | None]:
    sep = int(_full_grey_rows(inp)[0])
    top = inp[:sep]
    lower = inp[sep + 1 :]
    tile = _compress_template(top)
    tile_h, tile_w = tile.shape
    corners = _corner_pair(lower)
    blocks = _selected_blocks(lower, tile_h, tile_w)
    if len(blocks) < 2:
        return False, None

    visibility = _nearest_chebyshev_visibility(lower.shape, corners)
    out = _stamp_output(tile, lower.shape, blocks, visibility)
    variants = _block_variants(tile, visibility, blocks)
    tile_fg = tile != BLACK

    if np.count_nonzero(tile_fg) < 6 or len(set(map(int, np.unique(tile[tile_fg])))) < 2:
        return False, None

    for br, bc in blocks:
        r0 = br * tile_h
        c0 = bc * tile_w
        mask = visibility[r0 : r0 + tile_h, c0 : c0 + tile_w]
        visible_fg = int(np.count_nonzero(tile_fg & mask))
        hidden_fg = int(np.count_nonzero(tile_fg & (~mask)))
        if visible_fg == 0 or hidden_fg == 0:
            return False, None

    if len({variant.tobytes() for variant in variants}) < 2:
        return False, None

    non_black_colors = sorted(set(map(int, np.unique(out))) - {BLACK})
    if len(non_black_colors) < 2:
        return False, None

    full_tile_out = _stamp_output(tile, lower.shape, blocks, np.ones_like(lower, dtype=bool))
    if np.array_equal(out, full_tile_out):
        return False, None

    if all(
        np.array_equal(out, _stamp_output(tile, lower.shape, blocks, alt_visibility))
        for alt_visibility in (
            _single_corner_visibility(lower.shape, corners[0]),
            _single_corner_visibility(lower.shape, corners[1]),
            _nearest_manhattan_visibility(lower.shape, corners),
        )
    ):
        return False, None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return False, None
    return True, out


def _attempt() -> np.ndarray | None:
    tile_h, tile_w = _sample_tile_shape()
    diagonal = random.choice(("tl_br", "tr_bl"))
    tile = _sample_tile(tile_h, tile_w)
    blocks = _sample_blocks(diagonal)
    lower = _place_lower_markers(tile_h, tile_w, diagonal, blocks)
    template = _expand_template(tile)
    separator = np.full((1, lower.shape[1]), GREY, dtype=int)
    inp = np.vstack([template, separator, lower])

    ok, out = _validate_semantics(inp)
    if not ok or out is None:
        return None

    signature = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    if signature in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(signature)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid non-degenerate example")
