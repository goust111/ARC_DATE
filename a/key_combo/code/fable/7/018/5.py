from __future__ import annotations

import random

import numpy as np


BLACK = 0
PLACEHOLDER = -1
MIN_BLOCK_SIDE = 4
MAX_BLOCK_SIDE = 8
MAX_GENERATION_ATTEMPTS = 500


def _layout(grid: np.ndarray) -> tuple[int, list[tuple[slice, slice]], list[int]]:
    """Validate the visible frame and return block slices and corner keys."""
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width and height >= 11 and height % 2 == 1, (
        "expected an odd square grid with four blocks"
    )

    middle = height // 2
    block_side = middle - 1
    separator = int(g[middle, middle])
    assert separator != BLACK, "the central cross must be non-black"
    assert np.all(g[middle, :] == separator) and np.all(g[:, middle] == separator), (
        "expected one uniform central cross"
    )

    # The one-cell outer margin makes the four extreme corner pixels unambiguous
    # legends rather than overwritten cells of the motifs.
    edge_positions = np.arange(1, height - 1) != middle
    assert np.all(g[0, 1:-1][edge_positions] == BLACK)
    assert np.all(g[-1, 1:-1][edge_positions] == BLACK)
    assert np.all(g[1:-1, 0][edge_positions] == BLACK)
    assert np.all(g[1:-1, -1][edge_positions] == BLACK), (
        "expected a black outer margin apart from corner keys and cross endpoints"
    )

    keys = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    assert len(set(keys)) == 4, "the four corner key colors must be distinct"
    assert all(color not in {BLACK, separator} for color in keys), (
        "corner keys must differ from black and the separator"
    )

    blocks = [
        (slice(1, middle), slice(1, middle)),
        (slice(1, middle), slice(middle + 1, height - 1)),
        (slice(middle + 1, height - 1), slice(1, middle)),
        (slice(middle + 1, height - 1), slice(middle + 1, height - 1)),
    ]
    assert all(g[rs, cs].shape == (block_side, block_side) for rs, cs in blocks)
    return block_side, blocks, keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the blacked-out quadrant with the unused normalized rotation and its corner-key color.

    Input structure:
    - An odd square grid has a one-cell black outer margin, a uniform central cross,
      and four equal square blocks between them.  The four extreme corner pixels are
      distinct color keys.  Exactly one block is all black; the other three contain
      public colors plus occurrences of their corresponding corner-key color.

    Rule:
    - In each complete block, replace its own corner-key color by one common placeholder.
      The normalized blocks are three distinct quarter-turn rotations of one asymmetric
      base motif.  Select the only unused rotation, replace its placeholder cells by the
      missing block's corner-key color, and return that repaired block alone.

    Key insight:
    - The differently colored quadrants become congruent only after key-color
      normalization; their missing fourth rotation determines shape and the retained
      outer-corner key independently determines its recoloring.
    """
    g = np.asarray(grid, dtype=int)
    block_side, block_slices, keys = _layout(g)
    blocks = [g[rs, cs].copy() for rs, cs in block_slices]

    missing = [index for index, block in enumerate(blocks) if np.all(block == BLACK)]
    assert len(missing) == 1, "expected exactly one entirely black block"
    missing_index = missing[0]

    normalized: dict[int, np.ndarray] = {}
    for index, (block, key) in enumerate(zip(blocks, keys)):
        if index == missing_index:
            continue
        assert np.any(block == key), "each complete block must use its corner-key color"
        norm = block.copy()
        norm[norm == key] = PLACEHOLDER
        normalized[index] = norm

    anchor = next(iter(normalized.values()))
    rotations = [np.rot90(anchor, turns).copy() for turns in range(4)]
    assert all(rotation.shape == (block_side, block_side) for rotation in rotations)
    assert len({rotation.tobytes() for rotation in rotations}) == 4, (
        "the normalized base motif must have no quarter-turn symmetry"
    )

    used_turns: set[int] = set()
    for norm in normalized.values():
        matches = [turns for turns, candidate in enumerate(rotations) if np.array_equal(norm, candidate)]
        assert len(matches) == 1, "complete blocks must be rotations of one normalized motif"
        used_turns.add(matches[0])
    assert len(used_turns) == 3, "the complete blocks must show three distinct rotations"

    remaining = ({0, 1, 2, 3} - used_turns).pop()
    repaired = rotations[remaining].copy()
    assert np.any(repaired == PLACEHOLDER), "the base motif must contain a recolorable placeholder"
    repaired[repaired == PLACEHOLDER] = keys[missing_index]
    assert np.all((0 <= repaired) & (repaired <= 9)), "output must use ARC palette colors"
    return repaired.astype(int, copy=False)


def _random_base(side: int, public_colors: list[int]) -> np.ndarray | None:
    """Create a sparse, multi-colored, rotationally asymmetric normalized motif."""
    area = side * side
    occupied_count = random.randint(max(7, int(np.ceil(0.35 * area))), int(np.floor(0.58 * area)))
    placeholder_count = random.randint(3, occupied_count - 4)

    positions = random.sample(range(area), occupied_count)
    random.shuffle(positions)
    base = np.full((side, side), BLACK, dtype=int)
    flat = base.ravel()
    flat[positions[:placeholder_count]] = PLACEHOLDER

    public_positions = positions[placeholder_count:]
    split = random.randint(2, len(public_positions) - 2)
    flat[public_positions[:split]] = public_colors[0]
    flat[public_positions[split:]] = public_colors[1]

    rotations = [np.rot90(base, turns) for turns in range(4)]
    if len({rotation.tobytes() for rotation in rotations}) != 4:
        return None
    return base


def _extract_blocks(grid: np.ndarray) -> list[np.ndarray]:
    _, slices, _ = _layout(grid)
    return [grid[rs, cs].copy() for rs, cs in slices]


def _semantically_valid(
    grid: np.ndarray,
    expected: np.ndarray,
    base: np.ndarray,
    turns_by_block: list[int],
    missing_index: int,
    keys: list[int],
) -> bool:
    """Check the stated semantics, including dependencies that closure alone misses."""
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    blocks = _extract_blocks(grid)
    if sum(np.all(block == BLACK) for block in blocks) != 1:
        return False
    if turns_by_block[missing_index] in {
        turns_by_block[i] for i in range(4) if i != missing_index
    }:
        return False

    # Normalization is essential: differently keyed raw blocks must not already
    # be rotations of one another, while normalized blocks must be exact rotations.
    complete = [i for i in range(4) if i != missing_index]
    for pos, i in enumerate(complete):
        norm = blocks[i].copy()
        norm[norm == keys[i]] = PLACEHOLDER
        if not np.array_equal(norm, np.rot90(base, turns_by_block[i])):
            return False
        for j in complete[pos + 1 :]:
            if any(np.array_equal(np.rot90(blocks[i], k), blocks[j]) for k in range(4)):
                return False

    # Both the unused rotation and the missing corner key visibly affect the result.
    if int(np.sum(expected == keys[missing_index])) != int(np.sum(base == PLACEHOLDER)):
        return False
    if any(np.array_equal(expected, blocks[i]) for i in complete):
        return False
    wrong_key = keys[(missing_index + 1) % 4]
    recolored_wrong = expected.copy()
    recolored_wrong[recolored_wrong == keys[missing_index]] = wrong_key
    return not np.array_equal(expected, recolored_wrong)


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    side = random.randint(MIN_BLOCK_SIDE, MAX_BLOCK_SIDE)

    # Color identities are sampled only as visible roles; their numeric ids never
    # determine ordering, geometry, rotation, or matching.
    separator, *rest = random.sample(range(1, 10), 7)
    keys = rest[:4]
    public_colors = rest[4:]
    base = _random_base(side, public_colors)
    if base is None:
        raise ValueError("symmetric base")

    turns_by_block = random.sample(range(4), 4)
    missing_index = random.randrange(4)
    size = 2 * side + 3
    middle = size // 2
    grid = np.full((size, size), BLACK, dtype=int)
    grid[middle, :] = separator
    grid[:, middle] = separator
    grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1] = keys

    block_slices = [
        (slice(1, middle), slice(1, middle)),
        (slice(1, middle), slice(middle + 1, size - 1)),
        (slice(middle + 1, size - 1), slice(1, middle)),
        (slice(middle + 1, size - 1), slice(middle + 1, size - 1)),
    ]
    for index, (rs, cs) in enumerate(block_slices):
        if index == missing_index:
            continue
        block = np.rot90(base, turns_by_block[index]).copy()
        block[block == PLACEHOLDER] = keys[index]
        grid[rs, cs] = block

    expected = np.rot90(base, turns_by_block[missing_index]).copy()
    expected[expected == PLACEHOLDER] = keys[missing_index]
    if not _semantically_valid(grid, expected, base, turns_by_block, missing_index, keys):
        raise ValueError("semantic validation failed")
    return grid, expected


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance satisfying every solver invariant."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, _ = _generate_case()
        except (AssertionError, ValueError):
            continue
        assert max(grid.shape) <= 30
        return grid
    raise RuntimeError("failed to generate a valid 转色归位 input")
