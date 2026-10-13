import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SEPARATOR = 5
PATTERN_COLORS = np.array([6, 7, 8, 9], dtype=int)

INSTRUCTION_TO_OP = {
    RED: "rot_cw",
    BLUE: "flip_h",
    GREEN: "flip_v",
    YELLOW: "transpose",
}

_RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 300


def _find_separator_row(grid: np.ndarray) -> int:
    candidates = [
        r
        for r in range(grid.shape[0])
        if grid[r, 0] != BG and np.all(grid[r] == grid[r, 0])
    ]
    assert len(candidates) == 1, "expected exactly one full-width non-background separator row"
    return int(candidates[0])


def _extract_instruction_sequence(top: np.ndarray) -> list[int]:
    nonzero_rows = np.flatnonzero(np.any(top != BG, axis=1))
    assert len(nonzero_rows) == 1, "expected exactly one instruction row above the separator"
    row = top[int(nonzero_rows[0])]
    seq = [int(v) for v in row[row != BG]]
    assert seq, "instruction sequence must be non-empty"
    assert all(color in INSTRUCTION_TO_OP for color in seq), "unexpected instruction color"
    return seq


def _extract_pattern(bottom: np.ndarray) -> np.ndarray:
    coords = np.argwhere(bottom != BG)
    assert len(coords) > 0, "expected a target pattern below the separator"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    assert (r0, c0) == (0, 0), "pattern must be aligned to the lower region's top-left corner"
    assert r1 == c1, "pattern must occupy a square bounding box"

    size = int(r1 + 1)
    pattern = bottom[:size, :size]
    assert np.all(pattern != BG), "pattern square must be fully filled with non-background colors"
    assert not np.any(bottom[size:, :] != BG), "pixels below the pattern must be background"
    assert not np.any(bottom[:, size:] != BG), "pixels right of the pattern must be background"
    return pattern.copy()


def _apply_single(pattern: np.ndarray, op_name: str) -> np.ndarray:
    if op_name == "rot_cw":
        return np.rot90(pattern, -1)
    if op_name == "flip_h":
        return np.fliplr(pattern)
    if op_name == "flip_v":
        return np.flipud(pattern)
    if op_name == "transpose":
        return pattern.T.copy()
    raise AssertionError(f"unknown operation {op_name}")


def _apply_sequence(pattern: np.ndarray, sequence: list[int]) -> np.ndarray:
    result = pattern.copy()
    for color in sequence:
        result = _apply_single(result, INSTRUCTION_TO_OP[int(color)])
    return result


def _symmetry_variants(pattern: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(pattern, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _is_fully_asymmetric(pattern: np.ndarray) -> bool:
    keys = {variant.tobytes() for variant in _symmetry_variants(pattern)}
    return len(keys) == 8


def _random_pattern(size: int) -> np.ndarray:
    color_count = int(_RNG.integers(2, len(PATTERN_COLORS) + 1))
    palette = _RNG.choice(PATTERN_COLORS, size=color_count, replace=False)
    return _RNG.choice(palette, size=(size, size), replace=True).astype(int)


def _gen_attempt() -> np.ndarray | None:
    pattern_size = int(_RNG.integers(2, 6))
    extra_steps = int(_RNG.integers(0, 4))
    sequence = [BLUE, RED, GREEN, YELLOW]
    if extra_steps:
        sequence.extend(int(v) for v in _RNG.choice(sequence, size=extra_steps, replace=True))
    _RNG.shuffle(sequence)

    min_width = max(pattern_size + 2, len(sequence))
    width = int(_RNG.integers(min_width, min(18, min_width + 6) + 1))
    top_height = int(_RNG.integers(1, 5))
    lower_height = int(pattern_size + _RNG.integers(1, 6))
    height = top_height + 1 + lower_height

    instruction_row = int(_RNG.integers(0, top_height))
    seq_start = int(_RNG.integers(0, width - len(sequence) + 1))

    pattern = _random_pattern(pattern_size)
    if not _is_fully_asymmetric(pattern):
        return None

    transformed = _apply_sequence(pattern, sequence)
    if np.array_equal(transformed, pattern):
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[instruction_row, seq_start : seq_start + len(sequence)] = sequence
    grid[top_height, :] = SEPARATOR
    grid[top_height + 1 : top_height + 1 + pattern_size, :pattern_size] = pattern

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[top_height + 1 : top_height + 1 + pattern_size, :pattern_size], transformed):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """Apply the top instruction-color sequence to the lower square pattern and clear the legend area.

    Input structure:
    - The background is black.
    - A single full-width gray separator row splits the grid into an upper instruction region and a lower work region.
    - Above the separator, exactly one row contains a left-to-right sequence of colored instruction pixels.
    - Below the separator, a small fully filled square pattern sits at the lower region's top-left corner, with all other cells black.

    Rule:
    - Read the colored instruction pixels from left to right.
    - Interpret red as a 90-degree clockwise rotation, blue as a horizontal flip, green as a vertical flip, and yellow as a transpose across the main diagonal.
    - Extract the lower square pattern and apply those spatial transforms in sequence, each step acting on the previous result.
    - Return a grid of the same size with the separator preserved, the transformed square pasted back into the lower region's top-left corner, the rest of the lower region cleared to black, and the upper instruction region fully cleared.

    Key insight:
    - The top row is a visual program: each color encodes one geometric transform, and the output depends on chaining them in order rather than treating them independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_row = _find_separator_row(g)
    top = g[:separator_row]
    bottom = g[separator_row + 1 :]
    sequence = _extract_instruction_sequence(top)
    pattern = _extract_pattern(bottom)
    transformed = _apply_sequence(pattern, sequence)

    out = np.full_like(g, BG)
    out[separator_row, :] = g[separator_row, 0]
    size = int(transformed.shape[0])
    out[separator_row + 1 : separator_row + 1 + size, :size] = transformed
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instruction-sequence sample")
