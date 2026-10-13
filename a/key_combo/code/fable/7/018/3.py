"""ARC task 3: let the number of foreground colors choose the block scale."""

from __future__ import annotations

from collections import Counter
import random

import numpy as np


ARC_COLORS = tuple(range(10))
SCALE_CHOICES = (2, 3, 4)
AMBIGUITY_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 400

# Avoid returning two different inputs with the same answer during one loaded
# generator session.  This makes the requested non-collision property explicit.
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _unique_background(grid: np.ndarray) -> int:
    """Return the uniquely most frequent color."""
    counts = Counter(map(int, grid.flat))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], (
        "background must be the unique most frequent color"
    )
    return int(ranked[0][0])


def _compress_uniform_blocks(grid: np.ndarray, side: int) -> np.ndarray:
    """Collapse aligned square uniform blocks of the given side length."""
    height, width = map(int, grid.shape)
    assert side >= 1 and height % side == 0 and width % side == 0, (
        "block side must divide both grid dimensions"
    )
    blocks = grid.reshape(height // side, side, width // side, side).transpose(0, 2, 1, 3)
    representatives = blocks[:, :, 0, 0]
    assert np.all(blocks == representatives[:, :, None, None]), (
        "every selected block must be monochromatic"
    )
    return representatives.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a uniformly enlarged pattern using its foreground-color count as the block side.

    Input structure:
    - The input is a nonempty grid with one uniquely dominant background color.
    - It is an aligned array of monochromatic square blocks and contains several
      distinct non-background colors; its pattern also permits other pure-block scales.

    Rule:
    - Count the distinct colors other than the background, calling the count n.
    - Partition the entire grid, in its own row-column coordinate system starting at
      the upper-left corner, into aligned n by n blocks and replace each block by its color.

    Key insight:
    - Several block sizes look structurally valid, so divisibility or largest-block
      compression cannot set the scale; the visible number of foreground colors does.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "expected a nonempty grid"

    background = _unique_background(g)
    foreground_colors = set(map(int, np.unique(g))) - {background}
    n = len(foreground_colors)
    assert n >= 2, "expected at least two foreground colors to encode the scale"

    return _compress_uniform_blocks(g, n)


def _valid_uniform_scales(grid: np.ndarray) -> list[int]:
    """Enumerate aligned square block sides that compress without mixing colors."""
    height, width = map(int, grid.shape)
    valid: list[int] = []
    for side in range(1, min(height, width) + 1):
        if height % side or width % side:
            continue
        try:
            _compress_uniform_blocks(grid, side)
        except AssertionError:
            continue
        valid.append(side)
    return valid


def _make_core(height: int, width: int, background: int, colors: list[int]) -> np.ndarray:
    """Create a varied motif with a uniquely dominant background and every color visible."""
    area = height * width
    max_foreground = (area - 1) // 2
    min_foreground = len(colors)
    # Vary density while retaining a strict, human-visible dominant background.
    foreground_count = random.randint(min_foreground, max_foreground)

    cells = list(range(area))
    random.shuffle(cells)
    chosen = cells[:foreground_count]
    values = colors.copy()  # guarantees that every foreground color occurs
    values.extend(random.choice(colors) for _ in range(foreground_count - len(colors)))
    random.shuffle(values)

    core = np.full((height, width), background, dtype=int)
    for index, color in zip(chosen, values):
        core[index // width, index % width] = color
    return core


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    core: np.ndarray,
    encoded_scale: int,
    ambiguity_factor: int,
) -> bool:
    """Check the described semantics independently of the solver implementation."""
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    counts = Counter(map(int, inp.flat)).most_common()
    if len(counts) < 3 or counts[0][1] == counts[1][1]:
        return False
    background = counts[0][0]
    n = len(set(map(int, inp.flat)) - {background})
    if n != encoded_scale:
        return False

    # Direct structural oracle: the intended result is the ambiguity-expanded core.
    oracle = np.repeat(np.repeat(core, ambiguity_factor, axis=0), ambiguity_factor, axis=1)
    if not np.array_equal(expected, oracle):
        return False

    valid_scales = _valid_uniform_scales(inp)
    large_scale = encoded_scale * ambiguity_factor
    if encoded_scale not in valid_scales or large_scale not in valid_scales:
        return False
    if max(valid_scales) != large_scale:
        return False

    # The alternative largest-block shortcut is valid-looking but gives a different answer.
    largest_answer = _compress_uniform_blocks(inp, max(valid_scales))
    if not np.array_equal(largest_answer, core) or np.array_equal(largest_answer, expected):
        return False

    # Confirm the requested scale itself is uniform without calling main().
    intended_answer = _compress_uniform_blocks(inp, encoded_scale)
    if not np.array_equal(intended_answer, expected):
        return False

    # Prevent degenerate constant/symmetric motifs that obscure the retained layout.
    if len(np.unique(expected)) != encoded_scale + 1:
        return False
    if np.array_equal(core, np.flipud(core)) and np.array_equal(core, np.fliplr(core)):
        return False
    return True


def generate_input() -> np.ndarray:
    """Generate diverse, multiply-compressible examples satisfying the color-count rule."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        encoded_scale = random.choice(SCALE_CHOICES)
        ambiguity_factor = random.choice(AMBIGUITY_CHOICES)
        max_core_side = 30 // (encoded_scale * ambiguity_factor)
        if max_core_side < 3:
            continue

        core_height = random.randint(3, min(5, max_core_side))
        core_width = random.randint(3, min(5, max_core_side))
        background = random.choice(ARC_COLORS)
        palette = random.sample([c for c in ARC_COLORS if c != background], encoded_scale)
        core = _make_core(core_height, core_width, background, palette)

        expected = np.repeat(
            np.repeat(core, ambiguity_factor, axis=0), ambiguity_factor, axis=1
        )
        inp = np.repeat(
            np.repeat(expected, encoded_scale, axis=0), encoded_scale, axis=1
        )

        if not _validate_generated(
            inp, expected, core, encoded_scale, ambiguity_factor
        ):
            continue
        signature = (tuple(map(int, expected.shape)), expected.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        return inp

    raise RuntimeError("generate_input: failed to construct a valid nondegenerate example")
