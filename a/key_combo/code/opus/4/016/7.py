from __future__ import annotations

import numpy as np


BLACK = 0
SPRITE_SIZE = 3
SPRITE_COUNT = 4
STRIP_WIDTH = SPRITE_SIZE * SPRITE_COUNT
MAX_ATTEMPTS = 500
HEIGHT_PATTERNS = [
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 4),
    (2, 3, 4),
]

_RNG = np.random.default_rng()
_SEEN_INPUTS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()
_HEIGHT_CURSOR = 0


def _array_sig(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, grid.shape)), grid.tobytes()


def _separator_row(grid: np.ndarray) -> int:
    candidates = []
    for r in range(grid.shape[0]):
        if np.any(grid[r] != BLACK):
            continue
        if r == 0 or r == grid.shape[0] - 1:
            continue
        if np.any(grid[:r] != BLACK) and np.any(grid[r + 1 :] != BLACK):
            candidates.append(r)
    assert len(candidates) == 1, "expected exactly one all-black separator row"
    return int(candidates[0])


def _extract_sprites(top: np.ndarray) -> list[np.ndarray]:
    assert top.shape == (SPRITE_SIZE, STRIP_WIDTH), "top zone must be a 3x12 sprite strip"
    sprites = []
    for idx in range(SPRITE_COUNT):
        tile = top[:, idx * SPRITE_SIZE : (idx + 1) * SPRITE_SIZE].copy()
        colors = [int(c) for c in np.unique(tile) if int(c) != BLACK]
        assert len(colors) == 1, "each sprite must be monochrome"
        assert np.count_nonzero(tile) > 0, "each sprite needs foreground pixels"
        sprites.append(tile)
    return sprites


def _extract_legend_entries(bottom: np.ndarray) -> list[tuple[int, int]]:
    occupied = [int(c) for c in np.flatnonzero(np.any(bottom != BLACK, axis=0))]
    assert 2 <= len(occupied) <= 3, "legend must contain 2 or 3 entries"
    if len(occupied) > 1:
        gaps = [b - a for a, b in zip(occupied, occupied[1:])]
        assert all(gap == 2 for gap in gaps), "legend entries must be one column wide with one black gap"

    entries = []
    for c in occupied:
        col = bottom[:, c]
        rows = np.flatnonzero(col != BLACK)
        assert rows.size > 0, "legend column must contain foreground"
        r0 = int(rows[0])
        color = int(col[r0])
        assert np.all(col[r0:] == color), "legend must be a solid bottom-aligned bar"
        assert np.all(col[:r0] == BLACK), "legend top padding must stay black"
        entries.append((color, int(rows.size)))

    heights = [height for _color, height in entries]
    assert len(set(heights)) == len(heights), "legend heights must be distinct"
    assert all(1 <= height <= SPRITE_COUNT for height in heights), "legend heights must index sprite ranks"
    return entries


def _rank_sprites(sprites: list[np.ndarray]) -> tuple[list[int], list[int]]:
    counts = [int(np.count_nonzero(sprite)) for sprite in sprites]
    assert len(set(counts)) == len(counts), "sprite foreground areas must be unique"
    ranked = sorted(range(len(sprites)), key=lambda idx: (-counts[idx], idx))
    return counts, ranked


def _tile_square(sprite: np.ndarray, k: int) -> np.ndarray:
    return np.tile(sprite, (k, k))


def _compose_output(sprites: list[np.ndarray], legend: list[tuple[int, int]]) -> np.ndarray:
    counts, ranked = _rank_sprites(sprites)
    del counts
    panels = [_tile_square(sprites[ranked[k - 1]], k) for _color, k in legend]
    out_h = max(panel.shape[0] for panel in panels)
    out_w = sum(panel.shape[1] for panel in panels)
    out = np.zeros((out_h, out_w), dtype=int)
    c0 = 0
    for panel in panels:
        h, w = panel.shape
        out[:h, c0 : c0 + w] = panel
        c0 += w
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select sprites by area rank using legend height, then tile each selected sprite into a same-height square panel.

    Input structure:
    - The grid is split by one all-black separator row into a top sprite strip and a lower legend zone.
    - The top zone is a 3x12 strip of four adjacent 3x3 monochrome sprites on black background.
    - The bottom zone contains 2 or 3 one-column legend bars, bottom-aligned, separated by one black column.

    Rule:
    - Count non-black pixels in each of the four sprites and rank the sprites from largest area to smallest.
    - For each legend bar from left to right, let its height be k; choose the rank-k sprite from that ranking.
    - Build a 3k x 3k square by tiling the chosen 3x3 sprite into a k x k slot grid, then horizontally concatenate all squares.
    - Top-align the squares in the output and leave the uncovered cells below shorter squares black.

    Key insight:
    - Each legend height k has two inseparable meanings: it picks the rank-k sprite and simultaneously sets the square tiling size k x k.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row(g)
    sprites = _extract_sprites(g[:sep])
    legend = _extract_legend_entries(g[sep + 1 :])
    return _compose_output(sprites, legend)


def _random_mask(count: int) -> np.ndarray:
    mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    cells = _RNG.choice(SPRITE_SIZE * SPRITE_SIZE, size=count, replace=False)
    mask.flat[cells] = True
    return mask


def _make_sprite(color: int, count: int) -> np.ndarray:
    sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    sprite[_random_mask(count)] = int(color)
    return sprite


def _next_height_pattern() -> tuple[int, ...]:
    global _HEIGHT_CURSOR
    pattern = HEIGHT_PATTERNS[_HEIGHT_CURSOR % len(HEIGHT_PATTERNS)]
    _HEIGHT_CURSOR += 1
    perm = _RNG.permutation(len(pattern))
    return tuple(int(pattern[i]) for i in perm)


def _build_input(sprites: list[np.ndarray], legend: list[tuple[int, int]], start_col: int) -> np.ndarray:
    bottom_h = max(height for _color, height in legend)
    grid = np.zeros((SPRITE_SIZE + 1 + bottom_h, STRIP_WIDTH), dtype=int)
    for idx, sprite in enumerate(sprites):
        c0 = idx * SPRITE_SIZE
        grid[:SPRITE_SIZE, c0 : c0 + SPRITE_SIZE] = sprite

    for idx, (color, height) in enumerate(legend):
        c = start_col + 2 * idx
        grid[SPRITE_SIZE + 1 + bottom_h - height :, c] = int(color)
    return grid


def _expected_from_metadata(sprites: list[np.ndarray], legend: list[tuple[int, int]]) -> np.ndarray:
    return _compose_output(sprites, legend)


def _semantic_checks(grid: np.ndarray, sprites: list[np.ndarray], legend: list[tuple[int, int]], expected: np.ndarray) -> bool:
    sep = _separator_row(grid)
    parsed_sprites = _extract_sprites(grid[:sep])
    parsed_legend = _extract_legend_entries(grid[sep + 1 :])

    for built, parsed in zip(sprites, parsed_sprites):
        if not np.array_equal(built, parsed):
            return False
    if parsed_legend != legend:
        return False

    counts, ranked = _rank_sprites(sprites)
    if counts == sorted(counts, reverse=True) or counts == sorted(counts):
        return False
    if any(np.all(grid[r] == BLACK) for r in range(SPRITE_SIZE)):
        return False

    selected_positions = [ranked[height - 1] for _color, height in legend]
    if not any(pos != height - 1 for pos, (_color, height) in zip(selected_positions, legend)):
        return False

    if expected.shape != (3 * max(height for _color, height in legend), sum(3 * height for _color, height in legend)):
        return False

    c0 = 0
    max_h = expected.shape[0]
    for _color, height in legend:
        panel_h = 3 * height
        panel_w = 3 * height
        if panel_h < max_h and np.any(expected[panel_h:, c0 : c0 + panel_w] != BLACK):
            return False
        c0 += panel_w

    return True


def _gen_attempt() -> np.ndarray | None:
    heights = _next_height_pattern()
    legend_count = len(heights)
    palette = _RNG.choice(np.arange(1, 10), size=SPRITE_COUNT + legend_count, replace=False)
    sprite_colors = [int(v) for v in palette[:SPRITE_COUNT]]
    legend_colors = [int(v) for v in palette[SPRITE_COUNT:]]

    counts = [int(v) for v in _RNG.choice(np.arange(2, 9), size=SPRITE_COUNT, replace=False)]
    sprites = [_make_sprite(color, count) for color, count in zip(sprite_colors, counts)]
    legend = list(zip(legend_colors, heights))

    if len(set(counts)) != SPRITE_COUNT:
        return None

    strip = np.concatenate(sprites, axis=1)
    if any(np.all(strip[row] == BLACK) for row in range(SPRITE_SIZE)):
        return None

    span = 2 * legend_count - 1
    start_col = int(_RNG.integers(0, STRIP_WIDTH - span + 1))
    grid = _build_input(sprites, legend, start_col)
    expected = _expected_from_metadata(sprites, legend)

    if not _semantic_checks(grid, sprites, legend, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    in_sig = _array_sig(grid)
    out_sig = _array_sig(expected)
    if in_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
        return None

    _SEEN_INPUTS.add(in_sig)
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
