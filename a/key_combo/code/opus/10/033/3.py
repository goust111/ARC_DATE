from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
ALLOWED_COLORS = (2, 4, 5, 6, 7, 8, 9)
GRID_SIZE = 4
BLOCK_SIZE = 4
OUTPUT_SIZE = 8
ATTEMPT_LIMIT = 500

STATE_EMPTY = 0
STATE_A_ONLY = 1
STATE_B_ONLY = 2
STATE_BOTH = 3

STATUS_TEMPLATES = (
    (STATE_BOTH, STATE_A_ONLY, STATE_B_ONLY, STATE_EMPTY),
    (STATE_BOTH, STATE_BOTH, STATE_A_ONLY, STATE_EMPTY),
    (STATE_BOTH, STATE_A_ONLY, STATE_A_ONLY, STATE_B_ONLY),
    (STATE_BOTH, STATE_BOTH, STATE_A_ONLY, STATE_B_ONLY),
    (STATE_BOTH, STATE_BOTH, STATE_EMPTY, STATE_B_ONLY),
)

BASE_PATTERNS = {
    STATE_EMPTY: (
        ("..", ".."),
    ),
    STATE_A_ONLY: (
        ("A.", ".."),
        ("AA", ".."),
        ("A.", "A."),
        ("AA", "A."),
        ("AA", "AA"),
    ),
    STATE_B_ONLY: (
        ("B.", ".."),
        ("BB", ".."),
        ("B.", "B."),
        ("BB", "B."),
    ),
    STATE_BOTH: (
        ("AB", ".."),
        ("A.", "B."),
        ("AA", "B."),
        ("AB", "A."),
        ("AB", "BA"),
    ),
}


def _counts(grid: np.ndarray) -> Counter[int]:
    return Counter(int(v) for v in grid.flat if int(v) != BLACK)


def _classify_colors(grid: np.ndarray) -> tuple[int, int]:
    counts = _counts(grid)
    assert len(counts) == 2, "expected exactly two non-black colors"
    (color1, count1), (color2, count2) = sorted(counts.items())
    assert count1 != count2, "non-black colors must have different frequencies"
    if count1 > count2:
        return color1, color2
    return color2, color1


def _xor_compose(normal: np.ndarray, mirrored: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=int)
    mirrored = np.asarray(mirrored, dtype=int)
    out = np.zeros_like(normal)

    normal_fg = normal != BLACK
    mirrored_fg = mirrored != BLACK

    one_fg = normal_fg ^ mirrored_fg
    out[one_fg] = np.where(normal_fg[one_fg], normal[one_fg], mirrored[one_fg])

    different_fg = normal_fg & mirrored_fg & (normal != mirrored)
    out[different_fg] = BLUE
    return out


def _block_origin(block_r: int, block_c: int) -> tuple[int, int]:
    return block_r * BLOCK_SIZE, block_c * BLOCK_SIZE


def _quadrant_state(quadrant: np.ndarray, major: int, minor: int) -> int:
    has_major = bool(np.any(quadrant == major))
    has_minor = bool(np.any(quadrant == minor))
    if has_major and has_minor:
        return STATE_BOTH
    if has_major:
        return STATE_A_ONLY
    if has_minor:
        return STATE_B_ONLY
    return STATE_EMPTY


def _spec_output(grid: np.ndarray) -> np.ndarray:
    major, minor = _classify_colors(grid)
    normal = np.asarray(grid, dtype=int)
    mirrored = np.fliplr(normal)
    xor_patch = _xor_compose(normal, mirrored)

    major_blocks: list[list[np.ndarray | None]] = [[None, None], [None, None]]
    minor_blocks: list[list[np.ndarray | None]] = [[None, None], [None, None]]

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            block_r = r // 2
            block_c = c // 2
            color = int(normal[r, c])
            if color == major:
                major_blocks[block_r][block_c] = normal.copy()
            elif color == minor:
                minor_blocks[block_r][block_c] = mirrored.copy()

    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for block_r in range(2):
        for block_c in range(2):
            r0, c0 = _block_origin(block_r, block_c)
            major_stamp = major_blocks[block_r][block_c]
            minor_stamp = minor_blocks[block_r][block_c]
            if major_stamp is not None and minor_stamp is not None:
                patch = xor_patch
            elif major_stamp is not None:
                patch = normal
            elif minor_stamp is not None:
                patch = mirrored
            else:
                patch = None
            if patch is not None:
                out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the two input colors by frequency, stamp normal or mirrored copies by quadrant, and resolve collisions with XOR.

    Input structure:
    - The input is a 4x4 grid on a black background.
    - Exactly two non-black colors appear, both chosen from {2,4,5,6,7,8,9}.
    - One non-black color occurs strictly more often than the other.

    Rule:
    - Count the two non-black colors; the more frequent color is the normal-stamp trigger and the less frequent color is the mirrored-stamp trigger.
    - Build an 8x8 black output split into four 4x4 blocks, with each input cell mapped to a block by `(r//2, c//2)`.
    - If a block receives the high-frequency color, write the full 4x4 input there; if it receives the low-frequency color, write the left-right mirror there; if it receives both, replace that block with the pixelwise XOR composition of the normal and mirrored stamps.

    Key insight:
    - Individual cell positions do not stamp at distinct offsets: they only decide which of the four quadrant blocks receives a normal copy, a mirrored copy, or an XOR conflict between the two.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 4x4 input grid"
    counts = _counts(g)
    assert len(counts) == 2, "expected exactly two non-black colors"
    assert set(counts).issubset(ALLOWED_COLORS), "unexpected input colors"
    high_color, low_color = _classify_colors(g)
    assert counts[high_color] >= 3, "high-frequency color must appear at least 3 times"
    assert counts[low_color] >= 2, "low-frequency color must appear at least 2 times"

    normal = g.copy()
    mirrored = np.fliplr(normal)
    xor_patch = _xor_compose(normal, mirrored)

    high_blocks = np.zeros((2, 2), dtype=bool)
    low_blocks = np.zeros((2, 2), dtype=bool)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            block_r = r // 2
            block_c = c // 2
            if int(normal[r, c]) == high_color:
                high_blocks[block_r, block_c] = True
            elif int(normal[r, c]) == low_color:
                low_blocks[block_r, block_c] = True

    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for block_r in range(2):
        for block_c in range(2):
            r0, c0 = _block_origin(block_r, block_c)
            if high_blocks[block_r, block_c] and low_blocks[block_r, block_c]:
                patch = xor_patch
            elif high_blocks[block_r, block_c]:
                patch = normal
            elif low_blocks[block_r, block_c]:
                patch = mirrored
            else:
                patch = None
            if patch is not None:
                out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = patch
    return out


def _pattern_to_array(pattern: tuple[str, str], major: int, minor: int) -> np.ndarray:
    color_map = {".": BLACK, "A": major, "B": minor}
    return np.array([[color_map[ch] for ch in row] for row in pattern], dtype=int)


def _randomize_local_pattern(pattern: np.ndarray) -> np.ndarray:
    out = pattern.copy()
    if random.choice((False, True)):
        out = np.fliplr(out)
    return np.rot90(out, random.randrange(4))


def _sample_quadrant_patch(state: int, major: int, minor: int) -> np.ndarray:
    pattern = random.choice(BASE_PATTERNS[state])
    return _randomize_local_pattern(_pattern_to_array(pattern, major, minor))


def _assemble_grid(states: tuple[int, int, int, int], major: int, minor: int) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for idx, state in enumerate(states):
        block_r = idx // 2
        block_c = idx % 2
        r0 = block_r * 2
        c0 = block_c * 2
        grid[r0 : r0 + 2, c0 : c0 + 2] = _sample_quadrant_patch(state, major, minor)
    return grid


def _has_all_xor_witnesses(grid: np.ndarray) -> bool:
    mirrored = np.fliplr(grid)
    same_fg = (grid != BLACK) & (mirrored != BLACK) & (grid == mirrored)
    one_fg = (grid != BLACK) ^ (mirrored != BLACK)
    different_fg = (grid != BLACK) & (mirrored != BLACK) & (grid != mirrored)
    return bool(np.any(same_fg) and np.any(one_fg) and np.any(different_fg))


def _validate_grid(grid: np.ndarray, intended_states: tuple[int, int, int, int]) -> bool:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        return False

    counts = _counts(grid)
    if set(counts) - set(ALLOWED_COLORS):
        return False
    if len(counts) != 2:
        return False
    if len(set(counts.values())) != 2:
        return False

    major, minor = _classify_colors(grid)
    if counts[major] < 3 or counts[minor] < 2:
        return False
    if np.array_equal(grid, np.fliplr(grid)):
        return False

    actual_states = []
    for block_r in range(2):
        for block_c in range(2):
            quadrant = grid[block_r * 2 : block_r * 2 + 2, block_c * 2 : block_c * 2 + 2]
            actual_states.append(_quadrant_state(quadrant, major, minor))
    if tuple(actual_states) != intended_states:
        return False

    if STATE_BOTH not in actual_states:
        return False
    if not _has_all_xor_witnesses(grid):
        return False

    expected = _spec_output(grid)
    if not np.array_equal(main(grid), expected):
        return False

    xor_patch = _xor_compose(grid, np.fliplr(grid))
    if not np.any(xor_patch == BLUE):
        return False
    nonzero_exactly_one = xor_patch[(xor_patch != BLACK) & (xor_patch != BLUE)]
    if nonzero_exactly_one.size == 0:
        return False
    return True


def _sample_states() -> tuple[int, int, int, int]:
    states = list(random.choice(STATUS_TEMPLATES))
    random.shuffle(states)
    return tuple(states)


def _attempt_generate() -> np.ndarray | None:
    major, minor = random.sample(ALLOWED_COLORS, 2)
    intended_states = _sample_states()
    grid = _assemble_grid(intended_states, major, minor)
    if _validate_grid(grid, intended_states):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(ATTEMPT_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frequency-split XOR stamp input")
