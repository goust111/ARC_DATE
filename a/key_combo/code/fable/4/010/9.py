from __future__ import annotations

import math
import random

import numpy as np


_BACKGROUND = 0
_PALETTE = tuple(range(1, 10))
_MAX_ATTEMPTS = 100

# Cycling these cases makes every three consecutive generated examples include
# square/wide/tall layouts, corner/edge/interior corruption, and both sparse and
# strongly repeated recovery masks.  Order within each cycle is still random.
_shape_cycle: list[str] = []
_location_cycle: list[str] = []
_frequency_cycle: list[str] = []


def _next_case(cycle: list[str], choices: tuple[str, ...]) -> str:
    if not cycle:
        cycle.extend(choices)
        random.shuffle(cycle)
    return cycle.pop()


def _block_shape(grid_shape: tuple[int, int]) -> tuple[int, int]:
    """Infer (h, w): the grid is h by w blocks, each block is h by w."""
    height, width = map(int, grid_shape)
    h, w = math.isqrt(height), math.isqrt(width)
    assert h * h == height and w * w == width, (
        "input height and width must be perfect squares"
    )
    assert h >= 2 and w >= 2, "expected a nontrivial array of blocks"
    return h, w


def _blocks(g: np.ndarray, h: int, w: int):
    """Yield outer block coordinate and its same-frame h by w mask."""
    for i in range(h):
        for j in range(w):
            yield (i, j), g[i * h : (i + 1) * h, j * w : (j + 1) * w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a small color image from redundant global masks after one block is corrupted.

    Input structure:
    - The large black-background grid is arranged as h by w blocks, each of size h by w.
    - Every normal block at outer coordinate (i,j) contains black and one color; that color is
      the small image value at (i,j), and its within-block positions form that color's global mask.
    - Exactly one block is noise containing at least two non-black colors instead of one valid mask.

    Rule:
    - Write each normal block's sole non-black color at its matching outer coordinate.
    - At the noisy block coordinate, inspect that same coordinate inside every normal block; all
      normal masks covering it have one common owner color, which is written into the missing cell.
    - Discard every pixel of the noisy block and return the resulting h by w image.

    Key insight:
    - A block is simultaneously one output pixel and a full-image color mask, so a value erased in
      its own block remains recoverable at the same local coordinate in its same-colored siblings.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.issubdtype(g.dtype, np.integer), "ARC colors must be integers"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, (
        "ARC grids must be at most 30x30"
    )
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette colors 0..9"

    h, w = _block_shape(g.shape)
    out = np.full((h, w), _BACKGROUND, dtype=int)
    normal: list[tuple[tuple[int, int], np.ndarray, int]] = []
    noisy: list[tuple[tuple[int, int], np.ndarray]] = []

    for position, block in _blocks(g, h, w):
        colors = set(map(int, np.unique(block))) - {_BACKGROUND}
        if len(colors) == 1:
            owner = colors.pop()
            normal.append((position, block, owner))
            out[position] = owner
        elif len(colors) >= 2:
            noisy.append((position, block))
        else:
            raise AssertionError("a block may not be entirely black")

    assert len(noisy) == 1, "expected exactly one multicolor noise block"
    assert len(normal) == h * w - 1, "all remaining blocks must be normal"
    bad_position, _noise = noisy[0]

    # Outer block coordinates and inner mask coordinates deliberately share the
    # small-image (row, column) reference frame.
    witnesses = {
        owner
        for _position, block, owner in normal
        if int(block[bad_position]) != _BACKGROUND
    }
    assert len(witnesses) == 1, "normal masks must uniquely recover the noisy cell"
    out[bad_position] = witnesses.pop()
    assert np.all(out != _BACKGROUND), "every small-image cell must have a foreground color"

    # This is a semantic consistency check, not merely a format check: every
    # surviving block must encode the complete mask implied by the decoded image.
    for _position, block, owner in normal:
        assert np.array_equal(block == owner, out == owner), (
            "normal block does not equal its owner's decoded global mask"
        )
        assert np.all((block == _BACKGROUND) | (block == owner)), (
            "normal block contains an unrelated color"
        )

    return out


def _choose_shape(mode: str) -> tuple[int, int]:
    if mode == "square":
        side = random.randint(3, 5)
        return side, side
    if mode == "wide":
        h = random.randint(3, 4)
        return h, random.randint(h + 1, 5)
    if mode == "tall":
        w = random.randint(3, 4)
        return random.randint(w + 1, 5), w
    raise ValueError(f"unknown shape mode: {mode}")


def _choose_bad_position(h: int, w: int, mode: str) -> tuple[int, int]:
    if mode == "corner":
        return random.choice(((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    if mode == "edge":
        positions = (
            [(0, c) for c in range(1, w - 1)]
            + [(h - 1, c) for c in range(1, w - 1)]
            + [(r, 0) for r in range(1, h - 1)]
            + [(r, w - 1) for r in range(1, h - 1)]
        )
        return random.choice(positions)
    if mode == "interior":
        return random.randrange(1, h - 1), random.randrange(1, w - 1)
    raise ValueError(f"unknown location mode: {mode}")


def _target_count(cell_count: int, color_count: int, mode: str) -> int:
    maximum = cell_count - 2 * (color_count - 1)
    if mode == "sparse":
        return 2
    if mode == "medium":
        return min(3, maximum)
    if mode == "dense":
        return min(maximum, max(4, math.ceil(cell_count / color_count) + 1))
    raise ValueError(f"unknown frequency mode: {mode}")


def _small_image(
    h: int,
    w: int,
    bad_position: tuple[int, int],
    frequency_mode: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    cell_count = h * w
    color_count = random.randint(3, min(5, cell_count // 2))
    palette = tuple(random.sample(_PALETTE, color_count))
    target = random.choice(palette)
    count = _target_count(cell_count, color_count, frequency_mode)

    positions = [(r, c) for r in range(h) for c in range(w) if (r, c) != bad_position]
    random.shuffle(positions)
    small = np.full((h, w), -1, dtype=int)
    small[bad_position] = target

    cursor = 0
    for position in positions[cursor : cursor + count - 1]:
        small[position] = target
    cursor += count - 1

    other_colors = [color for color in palette if color != target]
    random.shuffle(other_colors)
    for color in other_colors:
        for position in positions[cursor : cursor + 2]:
            small[position] = color
        cursor += 2

    # Remaining cells never use target, preserving the intended sparse/medium/
    # dense recovery-witness branch exactly.
    for position in positions[cursor:]:
        small[position] = random.choice(other_colors)

    assert np.all(small >= 0)
    assert int(np.count_nonzero(small == target)) == count
    assert all(np.count_nonzero(small == color) >= 2 for color in palette)
    return small, palette


def _encode(
    small: np.ndarray,
    palette: tuple[int, ...],
    bad_position: tuple[int, int],
) -> np.ndarray:
    h, w = map(int, small.shape)
    encoded = np.zeros((h * h, w * w), dtype=int)

    for i in range(h):
        for j in range(w):
            if (i, j) == bad_position:
                continue
            owner = int(small[i, j])
            block = np.where(small == owner, owner, _BACKGROUND)
            encoded[i * h : (i + 1) * h, j * w : (j + 1) * w] = block

    true_color = int(small[bad_position])
    distractors = [color for color in palette if color != true_color]
    noise_color_count = random.randint(2, min(3, len(distractors)))
    noise_colors = random.sample(distractors, noise_color_count)

    noise = np.zeros((h, w), dtype=int)
    all_positions = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(all_positions)
    noise_area = random.randint(max(3, h * w // 3), min(h * w - 2, math.ceil(0.65 * h * w)))
    chosen = set(all_positions[:noise_area])
    chosen.add(bad_position)  # make the same-frame lookup visibly misleading

    chosen_list = list(chosen)
    random.shuffle(chosen_list)
    for index, position in enumerate(chosen_list):
        noise[position] = noise_colors[index % noise_color_count]
    # If adding bad_position enlarged the set, cycling still guarantees every
    # selected noise color occurs because noise_area is at least three.
    assert len(set(map(int, np.unique(noise))) - {_BACKGROUND}) >= 2
    assert true_color not in set(map(int, np.unique(noise)))
    assert int(noise[bad_position]) != true_color

    i, j = bad_position
    encoded[i * h : (i + 1) * h, j * w : (j + 1) * w] = noise
    return encoded


def _semantic_generation_check(
    encoded: np.ndarray,
    expected: np.ndarray,
    bad_position: tuple[int, int],
) -> bool:
    """Reject degenerate or merely self-consistent samples before returning them."""
    try:
        decoded = main(encoded)
    except (AssertionError, ValueError):
        return False
    if not np.array_equal(decoded, expected):
        return False

    h, w = expected.shape
    invalid = []
    witnesses = []
    for position, block in _blocks(encoded, h, w):
        colors = set(map(int, np.unique(block))) - {_BACKGROUND}
        if len(colors) >= 2:
            invalid.append(position)
        elif position != bad_position and int(block[bad_position]) != _BACKGROUND:
            witnesses.append(next(iter(colors)))

    true_color = int(expected[bad_position])
    i, j = bad_position
    noise = encoded[i * h : (i + 1) * h, j * w : (j + 1) * w]
    return (
        invalid == [bad_position]
        and len(witnesses) >= 1
        and set(witnesses) == {true_color}
        and len(witnesses) == int(np.count_nonzero(expected == true_color)) - 1
        and true_color not in set(map(int, np.unique(noise)))
        and int(noise[bad_position]) != true_color
        and len(np.unique(expected)) >= 3
        and not np.all(expected == expected.flat[0])
    )


def generate_input() -> np.ndarray:
    """Generate a varied valid redundant-mask codeword with one adversarial noise block."""
    shape_mode = _next_case(_shape_cycle, ("square", "wide", "tall"))
    location_mode = _next_case(_location_cycle, ("corner", "edge", "interior"))
    frequency_mode = _next_case(_frequency_cycle, ("sparse", "medium", "dense"))

    for _ in range(_MAX_ATTEMPTS):
        h, w = _choose_shape(shape_mode)
        bad_position = _choose_bad_position(h, w, location_mode)
        expected, palette = _small_image(h, w, bad_position, frequency_mode)
        encoded = _encode(expected, palette, bad_position)
        if _semantic_generation_check(encoded, expected, bad_position):
            assert max(encoded.shape) <= 30 and max(expected.shape) <= 30
            return encoded
    raise RuntimeError("failed to generate a valid redundant-mask decoding input")
