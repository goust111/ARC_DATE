import random
from collections import Counter

import numpy as np


BACKGROUND = 0
SEPARATOR = 2
PALETTE = (1, 3, 4, 7, 8, 9)
NOISE_PROB = 0.55
QUADRANT_SIZE = 8
COMPRESSED_SIZE = 4
BLOCK_SIZE = 2
RETRY_LIMIT = 400


def _most_common_color(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values.ravel())
    color, freq = counts.most_common(1)[0]
    assert list(counts.values()).count(freq) == 1, "expected a unique majority color"
    return int(color)


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]
    assert len(rows) == 1 and len(cols) == 1, "expected exactly one red separator cross"
    return int(rows[0]), int(cols[0])


def _extract_quadrants(grid: np.ndarray, cross_r: int, cross_c: int) -> list[np.ndarray]:
    q = cross_r
    assert q == cross_c == grid.shape[0] - cross_r - 1 == grid.shape[1] - cross_c - 1
    return [
        grid[:cross_r, :cross_c].copy(),
        grid[:cross_r, cross_c + 1 :].copy(),
        grid[cross_r + 1 :, :cross_c].copy(),
        grid[cross_r + 1 :, cross_c + 1 :].copy(),
    ]


def _block_slices(quadrant: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    grey = 5
    full_grey_rows = np.all(quadrant == grey, axis=1)
    full_grey_cols = np.all(quadrant == grey, axis=0)

    def runs(indices: list[int]) -> list[tuple[int, int]]:
        if not indices:
            return []
        out = []
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            out.append((start, prev + 1))
            start = prev = idx
        out.append((start, prev + 1))
        return out

    row_slices = runs([r for r, is_sep in enumerate(full_grey_rows) if not is_sep])
    col_slices = runs([c for c, is_sep in enumerate(full_grey_cols) if not is_sep])
    if (
        len(row_slices) == COMPRESSED_SIZE
        and len(col_slices) == COMPRESSED_SIZE
        and {r1 - r0 for r0, r1 in row_slices} == {BLOCK_SIZE}
        and {c1 - c0 for c0, c1 in col_slices} == {BLOCK_SIZE}
    ):
        return row_slices, col_slices

    h, w = quadrant.shape
    assert h == w == QUADRANT_SIZE, "expected 8x8 quadrants"
    return (
        [(r * BLOCK_SIZE, (r + 1) * BLOCK_SIZE) for r in range(COMPRESSED_SIZE)],
        [(c * BLOCK_SIZE, (c + 1) * BLOCK_SIZE) for c in range(COMPRESSED_SIZE)],
    )


def _compress_quadrant(quadrant: np.ndarray) -> np.ndarray:
    row_slices, col_slices = _block_slices(quadrant)
    out = np.zeros((len(row_slices), len(col_slices)), dtype=int)
    for rr, (r0, r1) in enumerate(row_slices):
        for cc, (c0, c1) in enumerate(col_slices):
            block = quadrant[r0:r1, c0:c1]
            assert block.size == 4, "expected 2x2 source blocks"
            assert not np.any(block == BACKGROUND), "content blocks must not contain background"
            out[rr, cc] = _most_common_color(block)
    return out


def _framed_sprite(sprite: np.ndarray) -> np.ndarray:
    framed = np.zeros((sprite.shape[0] + 2, sprite.shape[1] + 2), dtype=int)
    framed[1:-1, 1:-1] = sprite
    return framed


def _centered_paste(quadrant: np.ndarray, sprite: np.ndarray) -> np.ndarray:
    out = quadrant.copy()
    top = (quadrant.shape[0] - sprite.shape[0]) // 2
    left = (quadrant.shape[1] - sprite.shape[1]) // 2
    assert top >= 0 and left >= 0
    assert top * 2 + sprite.shape[0] == quadrant.shape[0]
    assert left * 2 + sprite.shape[1] == quadrant.shape[1]
    out[top : top + sprite.shape[0], left : left + sprite.shape[1]] = sprite
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the filled diagonal quadrants, compress their noisy 2x2 color blocks, and place the framed summary into the opposite empty quadrants.

    Input structure:
    - A red one-cell separator cross splits the grid into four equal quadrants.
    - Exactly one diagonal pair of quadrants contains a colored 2x2-block lattice; the opposite diagonal pair is entirely black.
    - In each content quadrant, every 2x2 block has one visible majority color, possibly with one noisy minority pixel.

    Rule:
    - Preserve the red separator cross and the two content quadrants.
    - Identify either the NW+SE or the NE+SW diagonal as the content pair.
    - Collapse one content quadrant by taking the majority color of every 2x2 block, producing a 4x4 summary.
    - Add a one-cell black border around that summary and center the framed 6x6 sprite inside both empty quadrants.

    Key insight:
    - The empty quadrants are not filled with a copy of the large source pattern; they receive its majority-vote compression, so diagonal completion and block summarization are both required.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cross_r, cross_c = _find_separator_cross(g)
    quadrants = _extract_quadrants(g, cross_r, cross_c)
    nonempty = [bool(np.any(q != BACKGROUND)) for q in quadrants]
    filled = {idx for idx, flag in enumerate(nonempty) if flag}
    assert filled in ({0, 3}, {1, 2}), "expected one filled diagonal pair"

    sample_idx = 0 if 0 in filled else 1
    compressed = _compress_quadrant(quadrants[sample_idx])
    framed = _framed_sprite(compressed)

    out_quadrants = [q.copy() for q in quadrants]
    for idx in range(4):
        if idx not in filled:
            assert np.all(out_quadrants[idx] == BACKGROUND), "empty quadrants must be pure black"
            out_quadrants[idx] = _centered_paste(out_quadrants[idx], framed)

    out = g.copy()
    out[:cross_r, :cross_c] = out_quadrants[0]
    out[:cross_r, cross_c + 1 :] = out_quadrants[1]
    out[cross_r + 1 :, :cross_c] = out_quadrants[2]
    out[cross_r + 1 :, cross_c + 1 :] = out_quadrants[3]
    out[cross_r, :] = SEPARATOR
    out[:, cross_c] = SEPARATOR
    return out


def _sample_majority_pattern() -> np.ndarray:
    distinct = random.randint(3, len(PALETTE))
    pool = random.sample(PALETTE, distinct)
    cells = pool[:]
    while len(cells) < COMPRESSED_SIZE * COMPRESSED_SIZE:
        cells.append(random.choice(pool))
    random.shuffle(cells)
    pattern = np.array(cells, dtype=int).reshape(COMPRESSED_SIZE, COMPRESSED_SIZE)
    assert len(set(int(v) for v in pattern.ravel())) >= 3
    return pattern


def _noise_mask() -> np.ndarray:
    mask = np.zeros((COMPRESSED_SIZE, COMPRESSED_SIZE), dtype=bool)
    noisy_count = random.randint(4, 10)
    for idx in random.sample(range(COMPRESSED_SIZE * COMPRESSED_SIZE), noisy_count):
        mask[idx // COMPRESSED_SIZE, idx % COMPRESSED_SIZE] = True
    return mask


def _render_content_quadrant(pattern: np.ndarray, noise_mask: np.ndarray) -> np.ndarray:
    quadrant = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=int)
    for br in range(COMPRESSED_SIZE):
        for bc in range(COMPRESSED_SIZE):
            majority = int(pattern[br, bc])
            block = np.full((BLOCK_SIZE, BLOCK_SIZE), majority, dtype=int)
            if noise_mask[br, bc]:
                choices = [color for color in PALETTE if color != majority]
                block[random.randrange(BLOCK_SIZE), random.randrange(BLOCK_SIZE)] = random.choice(choices)
            r0 = br * BLOCK_SIZE
            c0 = bc * BLOCK_SIZE
            quadrant[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = block
    return quadrant


def _quadrant_output_sprite(pattern: np.ndarray) -> np.ndarray:
    return _framed_sprite(pattern)


def _diag_indices(diag_name: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if diag_name == "nw_se":
        return (0, 3), (1, 2)
    return (1, 2), (0, 3)


def _build_scene(diag_name: str, pattern: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    grid = np.zeros((2 * QUADRANT_SIZE + 1, 2 * QUADRANT_SIZE + 1), dtype=int)
    grid[QUADRANT_SIZE, :] = SEPARATOR
    grid[:, QUADRANT_SIZE] = SEPARATOR

    content_a = _render_content_quadrant(pattern, mask_a)
    content_b = _render_content_quadrant(pattern, mask_b)

    filled, _ = _diag_indices(diag_name)
    quadrant_map = {
        0: (slice(0, QUADRANT_SIZE), slice(0, QUADRANT_SIZE)),
        1: (slice(0, QUADRANT_SIZE), slice(QUADRANT_SIZE + 1, 2 * QUADRANT_SIZE + 1)),
        2: (slice(QUADRANT_SIZE + 1, 2 * QUADRANT_SIZE + 1), slice(0, QUADRANT_SIZE)),
        3: (slice(QUADRANT_SIZE + 1, 2 * QUADRANT_SIZE + 1), slice(QUADRANT_SIZE + 1, 2 * QUADRANT_SIZE + 1)),
    }

    for idx, content in zip(filled, (content_a, content_b)):
        rs, cs = quadrant_map[idx]
        grid[rs, cs] = content
    return grid


def _validate_input(grid: np.ndarray) -> tuple[np.ndarray, set[int]]:
    assert grid.shape == (17, 17)
    cross_r, cross_c = _find_separator_cross(grid)
    assert (cross_r, cross_c) == (QUADRANT_SIZE, QUADRANT_SIZE)
    quadrants = _extract_quadrants(grid, cross_r, cross_c)
    nonempty = [bool(np.any(q != BACKGROUND)) for q in quadrants]
    filled = {idx for idx, flag in enumerate(nonempty) if flag}
    assert filled in ({0, 3}, {1, 2})

    compressed = _compress_quadrant(quadrants[min(filled)])
    for idx, quadrant in enumerate(quadrants):
        if idx in filled:
            assert np.array_equal(_compress_quadrant(quadrant), compressed)
            assert len(set(int(v) for v in quadrant.ravel())) >= 3
        else:
            assert np.all(quadrant == BACKGROUND)

    assert len(set(int(v) for v in compressed.ravel())) >= 3
    assert any(np.any(quadrants[idx] != np.kron(compressed, np.ones((2, 2), dtype=int))) for idx in filled)
    return compressed, filled


def _validate_output(inp: np.ndarray, out: np.ndarray) -> None:
    compressed, filled = _validate_input(inp)
    framed = _quadrant_output_sprite(compressed)
    cross_r, cross_c = QUADRANT_SIZE, QUADRANT_SIZE
    quadrants_in = _extract_quadrants(inp, cross_r, cross_c)
    quadrants_out = _extract_quadrants(out, cross_r, cross_c)
    for idx in filled:
        assert np.array_equal(quadrants_out[idx], quadrants_in[idx])
    for idx in ({0, 1, 2, 3} - filled):
        expected = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=int)
        expected = _centered_paste(expected, framed)
        assert np.array_equal(quadrants_out[idx], expected)
    assert np.all(out[cross_r, :] == SEPARATOR)
    assert np.all(out[:, cross_c] == SEPARATOR)


def _attempt() -> np.ndarray | None:
    diag_name = random.choice(("nw_se", "ne_sw"))
    pattern = _sample_majority_pattern()
    mask_a = _noise_mask()
    mask_b = _noise_mask()
    if not mask_a.any() and not mask_b.any():
        return None
    if random.random() < 0.8 and np.array_equal(mask_a, mask_b):
        return None

    inp = _build_scene(diag_name, pattern, mask_a, mask_b)
    try:
        out = main(inp)
        _validate_output(inp, out)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-complement compression puzzle")
