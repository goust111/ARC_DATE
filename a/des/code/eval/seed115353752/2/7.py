from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
AZURE = 8
RED = 2

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 300
GRID_COUNT_RANGE = (2, 5)
TILE_SIDE_RANGE = (4, 6)


def _find_separator_color(grid: np.ndarray) -> int:
    candidates: list[int] = []
    for color in np.unique(grid):
        value = int(color)
        if bool(np.any(np.all(grid == value, axis=1))) and bool(np.any(np.all(grid == value, axis=0))):
            candidates.append(value)
    assert len(candidates) == 1, "expected exactly one separator color forming full rows and columns"
    return candidates[0]


def _divider_spans(lines: list[int], length: int) -> list[tuple[int, int]]:
    dividers = sorted(set(int(x) for x in lines))
    if not dividers or dividers[0] != 0:
        dividers = [-1] + dividers
    dividers = dividers + [int(length)]

    spans: list[tuple[int, int]] = []
    for a, b in zip(dividers, dividers[1:]):
        if int(b - a) <= 1:
            continue
        spans.append((int(a + 1), int(b - 1)))

    assert spans, "expected at least one tile span"
    size = spans[0][1] - spans[0][0] + 1
    assert size >= 1
    assert all((r1 - r0 + 1) == size for r0, r1 in spans), "tile spans must be uniform"
    return spans


def _extract_tiles(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int, np.ndarray]]]:
    g = np.asarray(grid, dtype=int)
    separator = _find_separator_color(g)
    row_lines = [int(r) for r in range(g.shape[0]) if bool(np.all(g[int(r), :] == separator))]
    col_lines = [int(c) for c in range(g.shape[1]) if bool(np.all(g[:, int(c)] == separator))]
    assert row_lines and col_lines, "expected separator rows and columns"

    row_spans = _divider_spans(row_lines, g.shape[0])
    col_spans = _divider_spans(col_lines, g.shape[1])
    tile_h = row_spans[0][1] - row_spans[0][0] + 1
    tile_w = col_spans[0][1] - col_spans[0][0] + 1

    tiles: list[tuple[int, int, np.ndarray]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert tile.shape == (tile_h, tile_w)
            tiles.append((int(r0), int(c0), tile))
    assert len(tiles) >= 2, "expected multiple tile copies"
    return separator, BLACK, tiles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the consensus red/black tile by majority vote, then mark per-tile disagreements in yellow.

    Input structure:
    - Black background.
    - Azure full rows/cols divide the grid into equal rectangular tiles.
    - Each tile contains a noisy copy of the same red pattern.

    Rule:
    - For every tile-local position, take the majority value across all tiles.
    - Replace every tile with that consensus pattern.
    - In each tile, positions that originally disagreed with the consensus become yellow.
    - Separator lines stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    separator, bg, tiles = _extract_tiles(g)

    palette = set(map(int, np.unique(g)))
    palette.discard(int(bg))
    palette.discard(int(separator))
    assert len(palette) == 1, "expected exactly one non-background pattern color"
    pattern = next(iter(palette))
    assert pattern != YELLOW, "input should not already use the highlight color"

    tile_masks: list[np.ndarray] = []
    for _r0, _c0, tile in tiles:
        assert set(map(int, np.unique(tile))).issubset({int(bg), int(pattern)})
        tile_masks.append(tile == pattern)

    stack = np.stack(tile_masks, axis=0)
    tile_count = int(stack.shape[0])
    red_count = stack.sum(axis=0)
    assert np.all(red_count * 2 != tile_count), "majority vote must be unambiguous at every position"
    consensus_red = red_count * 2 > tile_count
    consensus_tile = np.where(consensus_red, pattern, bg).astype(int)

    out = g.copy()
    tile_h, tile_w = consensus_tile.shape
    for r0, c0, tile in tiles:
        diff = (tile == pattern) != consensus_red
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = consensus_tile
        region = out[r0 : r0 + tile_h, c0 : c0 + tile_w]
        region[diff] = YELLOW

    return out


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr, dc in DIRS4
        if 0 <= r + dr < h and 0 <= c + dc < w
    ]


def _pattern_blob(h: int, w: int) -> np.ndarray:
    target = random.randint(max(3, (h * w) // 4), min(h * w - 3, max(4, (3 * h * w) // 5)))
    pat = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    pat[start] = True

    while int(pat.sum()) < target:
        filled = [tuple(map(int, rc)) for rc in np.argwhere(pat)]
        r, c = random.choice(filled)
        frontier = [(nr, nc) for nr, nc in _neighbors(r, c, h, w) if not pat[nr, nc]]
        if frontier:
            nr, nc = random.choice(frontier)
            pat[nr, nc] = True
            continue

        empties = [tuple(map(int, rc)) for rc in np.argwhere(~pat)]
        nr, nc = random.choice(empties)
        pat[nr, nc] = True
    return pat


def _pattern_bars(h: int, w: int) -> np.ndarray:
    pat = np.zeros((h, w), dtype=bool)
    seg_count = random.randint(2, 4)
    for _ in range(seg_count):
        if bool(random.random() < 0.5):
            r = random.randrange(h)
            c0 = random.randrange(w)
            c1 = random.randrange(c0 + 1, w + 1)
            if c1 - c0 == 1 and w > 1 and bool(random.random() < 0.7):
                c1 = min(w, c0 + 2)
            pat[r, c0:c1] = True
        else:
            c = random.randrange(w)
            r0 = random.randrange(h)
            r1 = random.randrange(r0 + 1, h + 1)
            if r1 - r0 == 1 and h > 1 and bool(random.random() < 0.7):
                r1 = min(h, r0 + 2)
            pat[r0:r1, c] = True
    return pat


def _pattern_box(h: int, w: int) -> np.ndarray:
    pat = np.zeros((h, w), dtype=bool)
    r0 = random.randrange(h - 1)
    c0 = random.randrange(w - 1)
    r1 = random.randrange(r0 + 1, h)
    c1 = random.randrange(c0 + 1, w)
    if bool(random.random() < 0.5):
        pat[r0 : r1 + 1, c0 : c1 + 1] = True
    else:
        pat[r0, c0 : c1 + 1] = True
        pat[r1, c0 : c1 + 1] = True
        pat[r0 : r1 + 1, c0] = True
        pat[r0 : r1 + 1, c1] = True
    if bool(random.random() < 0.6):
        arm_r, arm_c = random.choice([tuple(x) for x in np.argwhere(pat)])
        nr, nc = random.choice(_neighbors(arm_r, arm_c, h, w))
        pat[nr, nc] = True
    return pat


def _random_pattern(h: int, w: int) -> np.ndarray:
    builders = (_pattern_blob, _pattern_bars, _pattern_box)
    for _ in range(100):
        pat = random.choice(builders)(h, w)
        if bool(random.random() < 0.35):
            extra = (random.randrange(h), random.randrange(w))
            pat[extra] = True
        red_cells = int(pat.sum())
        if 3 <= red_cells <= (h * w - 3):
            return pat
    raise RuntimeError("failed to generate a usable consensus pattern")


def _tile_origins(count: int, size: int, bordered: bool) -> list[int]:
    offset = 1 if bordered else 0
    return [int(offset + i * (size + 1)) for i in range(count)]


def _blank_grid(rows: int, cols: int, tile_h: int, tile_w: int, border_rows: bool, border_cols: bool) -> tuple[np.ndarray, list[int], list[int]]:
    height = int(rows * tile_h + (rows - 1) + (2 if border_rows else 0))
    width = int(cols * tile_w + (cols - 1) + (2 if border_cols else 0))
    grid = np.full((height, width), BLACK, dtype=int)

    row_starts = _tile_origins(rows, tile_h, border_rows)
    col_starts = _tile_origins(cols, tile_w, border_cols)

    if border_rows:
        grid[0, :] = AZURE
        grid[-1, :] = AZURE
    if border_cols:
        grid[:, 0] = AZURE
        grid[:, -1] = AZURE
    for start in row_starts[1:]:
        grid[int(start - 1), :] = AZURE
    for start in col_starts[1:]:
        grid[:, int(start - 1)] = AZURE

    return grid, row_starts, col_starts


def _assign_corruptions(tile_count: int, tile_h: int, tile_w: int) -> list[set[tuple[int, int]]]:
    max_per_position = (tile_count - 1) // 2
    assert max_per_position >= 1

    changes = [set() for _ in range(tile_count)]
    pos_counts = np.zeros((tile_h, tile_w), dtype=int)
    clean_tile = random.randrange(tile_count)
    mutable_tiles = [idx for idx in range(tile_count) if idx != clean_tile]

    if max_per_position >= 2 and len(mutable_tiles) >= 2 and bool(random.random() < 0.75):
        repeat_pos = (random.randrange(tile_h), random.randrange(tile_w))
        repeat_k = random.randint(2, min(max_per_position, len(mutable_tiles)))
        for idx in random.sample(mutable_tiles, repeat_k):
            changes[idx].add(repeat_pos)
            pos_counts[repeat_pos] += 1

    desired = [0] * tile_count
    for idx in mutable_tiles:
        desired[idx] = random.choices((0, 1, 2), weights=(2, 4, 3), k=1)[0]

    if sum(desired) == 0:
        forced = random.choice(mutable_tiles)
        desired[forced] = 1

    for idx in range(tile_count):
        while len(changes[idx]) < desired[idx]:
            available = [
                (r, c)
                for r in range(tile_h)
                for c in range(tile_w)
                if pos_counts[r, c] < max_per_position and (r, c) not in changes[idx]
            ]
            if not available:
                break
            repeated = [(r, c) for r, c in available if 0 < pos_counts[r, c] < max_per_position]
            if repeated and bool(random.random() < 0.6):
                pos = random.choice(repeated)
            else:
                pos = random.choice(available)
            changes[idx].add(pos)
            pos_counts[pos] += 1

    if not any(changes):
        pos = (random.randrange(tile_h), random.randrange(tile_w))
        idx = random.choice(mutable_tiles)
        changes[idx].add(pos)

    assert all(len(tile_changes) <= 2 for tile_changes in changes)
    assert int(pos_counts.max()) <= max_per_position
    return changes


def _build_pair(
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
    border_rows: bool,
    border_cols: bool,
    consensus_red: np.ndarray,
    corruptions: list[set[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    inp, row_starts, col_starts = _blank_grid(rows, cols, tile_h, tile_w, border_rows, border_cols)
    out, _, _ = _blank_grid(rows, cols, tile_h, tile_w, border_rows, border_cols)

    clean_tile = np.where(consensus_red, RED, BLACK).astype(int)
    tile_idx = 0
    for r0 in row_starts:
        for c0 in col_starts:
            tile_mask = consensus_red.copy()
            for rr, cc in corruptions[tile_idx]:
                tile_mask[int(rr), int(cc)] = ~tile_mask[int(rr), int(cc)]

            tile_in = np.where(tile_mask, RED, BLACK).astype(int)
            tile_out = clean_tile.copy()
            for rr, cc in corruptions[tile_idx]:
                tile_out[int(rr), int(cc)] = YELLOW

            inp[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile_in
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile_out
            tile_idx += 1

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_COUNT_RANGE)
    cols = random.randint(*GRID_COUNT_RANGE)
    tile_h = random.randint(*TILE_SIDE_RANGE)
    tile_w = random.randint(*TILE_SIDE_RANGE)
    border_rows = bool(random.getrandbits(1))
    border_cols = bool(random.getrandbits(1))

    consensus = _random_pattern(tile_h, tile_w)
    corruptions = _assign_corruptions(rows * cols, tile_h, tile_w)
    inp, expected = _build_pair(rows, cols, tile_h, tile_w, border_rows, border_cols, consensus, corruptions)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
