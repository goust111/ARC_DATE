import hashlib
import itertools

import numpy as np


TILE = 3
CLOCK_COLOR = 6
SPRITE_COLORS = [1, 2, 3, 4, 7]
COUNT_CHOICES = [2, 3, 4, 5, 6, 7]
MAX_ATTEMPTS = 500

_ANGLE_TO_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
_DELTA_TO_ANGLE = {v: k for k, v in _ANGLE_TO_DELTA.items()}
_SEEN_OUTPUT_SIGNATURES: set[str] = set()
_CALL_INDEX = 0


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    pending = [tuple(map(int, coords[0]))]
    seen = {pending[0]}
    while pending:
        r, c = pending.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < TILE and 0 <= nc < TILE and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                pending.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_mask_library() -> dict[int, list[np.ndarray]]:
    library: dict[int, list[np.ndarray]] = {count: [] for count in COUNT_CHOICES}
    cells = list(range(TILE * TILE))
    for count in COUNT_CHOICES:
        for combo in itertools.combinations(cells, count):
            mask = np.zeros((TILE, TILE), dtype=bool)
            mask.flat[list(combo)] = True
            if not _mask_connected(mask):
                continue
            if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
                continue
            library[count].append(mask)
        assert library[count], f"missing candidate masks for count {count}"
    return library


MASK_LIBRARY = _build_mask_library()


def _split_input(grid: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == 2 * TILE, "expected exactly two rows of 3x3 tiles"
    assert w % TILE == 0, "grid width must be a multiple of 3"
    tile_count = w // TILE
    assert tile_count >= 2, "expected at least two tile pairs"

    clocks = [g[:TILE, c : c + TILE].copy() for c in range(0, w, TILE)]
    sprites = [g[TILE:, c : c + TILE].copy() for c in range(0, w, TILE)]
    return clocks, sprites


def _parse_clock(tile: np.ndarray) -> int:
    assert tile.shape == (TILE, TILE), "clock tile must be 3x3"
    fg = np.argwhere(tile == CLOCK_COLOR)
    assert len(fg) == 2, "clock tile must contain exactly two magenta pixels"
    assert int(tile[1, 1]) == CLOCK_COLOR, "clock center must be magenta"
    assert np.all((tile == 0) | (tile == CLOCK_COLOR)), "clock tile may only contain black and magenta"

    other = [tuple(map(int, rc)) for rc in fg if tuple(map(int, rc)) != (1, 1)]
    assert len(other) == 1, "clock tile must have exactly one pointer pixel"
    dr = other[0][0] - 1
    dc = other[0][1] - 1
    assert (dr, dc) in _DELTA_TO_ANGLE, "pointer must be a cardinal neighbor of the center"
    return _DELTA_TO_ANGLE[(dr, dc)]


def _parse_sprite(tile: np.ndarray) -> tuple[int, int]:
    assert tile.shape == (TILE, TILE), "sprite tile must be 3x3"
    colors = sorted(int(v) for v in np.unique(tile) if int(v) != 0)
    assert len(colors) == 1, "sprite tile must contain exactly one foreground color"
    color = colors[0]
    assert color in SPRITE_COLORS, "unexpected sprite color"
    count = int(np.count_nonzero(tile))
    assert 2 <= count <= 7, "sprite foreground count must stay within the puzzle range"
    return color, count


def _rotate_clockwise(tile: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(tile, -int(turns))


def _expected_from_pairs(clocks: list[np.ndarray], sprites: list[np.ndarray]) -> np.ndarray:
    assert len(clocks) == len(sprites)
    parsed = []
    for idx, (clock, sprite) in enumerate(zip(clocks, sprites)):
        _color, count = _parse_sprite(sprite)
        turns = _parse_clock(clock)
        parsed.append((idx, count, turns, _rotate_clockwise(sprite, turns)))

    parsed.sort(key=lambda item: (-item[1], item[0]))
    return np.concatenate([item[3] for item in parsed], axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort a strip of sprite tiles by foreground density and rotate each tile by its paired clock marker.

    Input structure:
    - The grid has two horizontal rows of 3x3 tiles with black background.
    - The top row contains clock tiles: each has a magenta center pixel and exactly one magenta pointer
      in the up/right/down/left neighbor, encoding 0/90/180/270 degrees clockwise.
    - The bottom row contains one monochrome non-black sprite per tile; each tile's foreground-pixel
      count is unique within the strip.

    Rule:
    - Pair each top clock tile with the sprite tile directly below it.
    - Count non-black pixels in every sprite tile and sort the pairs from highest count to lowest.
    - Rotate each sprite tile clockwise by the angle encoded in its paired clock tile.
    - Output a single horizontal strip of the rotated sprite tiles in the sorted order.

    Key insight:
    - The tile order comes from one information source (sprite pixel counts), while the tile orientation
      comes from a different source (clock markers), so both must be read and combined correctly.
    """
    clocks, sprites = _split_input(grid)
    return _expected_from_pairs(clocks, sprites)


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    tile = np.zeros((TILE, TILE), dtype=int)
    tile[mask] = int(color)
    return tile


def _make_clock(turns: int) -> np.ndarray:
    tile = np.zeros((TILE, TILE), dtype=int)
    tile[1, 1] = CLOCK_COLOR
    dr, dc = _ANGLE_TO_DELTA[int(turns)]
    tile[1 + dr, 1 + dc] = CLOCK_COLOR
    return tile


def _output_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(map(int, grid.shape))).encode()).hexdigest()


def _required_angles(tile_count: int, call_index: int, rng: np.random.Generator) -> list[int]:
    coverage_modes = [
        [1, 2],
        [3, 0],
        [0, 1, 2, 3],
    ]
    required = coverage_modes[call_index % len(coverage_modes)][:]
    if tile_count < len(required):
        required = required[:tile_count]
    while len(required) < tile_count:
        required.append(int(rng.integers(0, 4)))
    rng.shuffle(required)
    if all(angle == 0 for angle in required):
        required[int(rng.integers(0, tile_count))] = int(rng.integers(1, 4))
    return required


def _sample_sprite_mask(count: int, rng: np.random.Generator) -> np.ndarray:
    base = MASK_LIBRARY[int(count)][int(rng.integers(0, len(MASK_LIBRARY[int(count)])))]
    return np.rot90(base, int(rng.integers(0, 4))).copy()


def _build_case(rng: np.random.Generator, call_index: int) -> tuple[np.ndarray, np.ndarray]:
    tile_count = int(rng.integers(4, 6))
    colors = rng.choice(np.array(SPRITE_COLORS), size=tile_count, replace=False).tolist()
    counts = rng.choice(np.array(COUNT_CHOICES), size=tile_count, replace=False).tolist()
    rng.shuffle(counts)
    if counts == sorted(counts, reverse=True):
        counts[0], counts[-1] = counts[-1], counts[0]

    turns = _required_angles(tile_count, call_index, rng)
    clocks = [_make_clock(turn) for turn in turns]
    sprites = []
    for color, count in zip(colors, counts):
        sprite = _colorize(_sample_sprite_mask(count, rng), int(color))
        sprites.append(sprite)

    grid = np.vstack(
        [
            np.concatenate(clocks, axis=1),
            np.concatenate(sprites, axis=1),
        ]
    )
    out = _expected_from_pairs(clocks, sprites)
    return grid, out


def _validate_case(grid: np.ndarray, expected: np.ndarray) -> bool:
    clocks, sprites = _split_input(grid)
    counts = [int(np.count_nonzero(tile)) for tile in sprites]
    turns = [_parse_clock(tile) for tile in clocks]
    sorted_order = sorted(range(len(sprites)), key=lambda idx: (-counts[idx], idx))

    if len(set(counts)) != len(counts):
        return False
    if counts == sorted(counts, reverse=True):
        return False
    if all(turn == 0 for turn in turns):
        return False
    if all(idx == pos for pos, idx in enumerate(sorted_order)):
        return False

    changed_rotations = [
        not np.array_equal(sprite, _rotate_clockwise(sprite, turn))
        for sprite, turn in zip(sprites, turns)
        if turn != 0
    ]
    if not changed_rotations or not all(changed_rotations):
        return False

    rotated_only = np.concatenate(
        [_rotate_clockwise(sprite, turn) for sprite, turn in zip(sprites, turns)],
        axis=1,
    )
    sorted_only = np.concatenate([sprites[idx] for idx in sorted_order], axis=1)
    reverse_sorted = np.concatenate(
        [_rotate_clockwise(sprites[idx], turns[idx]) for idx in sorted(range(len(sprites)), key=lambda idx: (counts[idx], idx))],
        axis=1,
    )

    if np.array_equal(expected, rotated_only):
        return False
    if np.array_equal(expected, sorted_only):
        return False
    if np.array_equal(expected, reverse_sorted):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    global _CALL_INDEX

    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        grid, expected = _build_case(rng, _CALL_INDEX)
        if not _validate_case(grid, expected):
            continue

        sig = _output_signature(expected)
        if sig in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(sig)
        _CALL_INDEX += 1
        return grid

    raise RuntimeError("failed to generate a valid non-degenerate input")
