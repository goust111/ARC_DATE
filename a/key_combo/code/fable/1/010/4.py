import math
import random

import numpy as np


BG = 0
FG_COLORS = tuple(range(1, 10))
N_CHOICES = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 500


def _infer_order(shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    assert h == w, "grid must be square"
    n = math.isqrt(h)
    assert n * n == h, "side length must be a perfect square"
    assert n >= 2
    return n


def _foreground_color(grid: np.ndarray) -> int:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != BG)
    assert len(colors) == 1, "expected exactly one foreground color"
    return colors[0]


def _block_slice(n: int, br: int, bc: int) -> tuple[slice, slice]:
    return slice(br * n, (br + 1) * n), slice(bc * n, (bc + 1) * n)


def _block_counts(grid: np.ndarray, n: int) -> np.ndarray:
    counts = np.zeros((n, n), dtype=int)
    for br in range(n):
        for bc in range(n):
            rs, cs = _block_slice(n, br, bc)
            counts[br, bc] = int(np.count_nonzero(grid[rs, cs]))
    return counts


def _full_output(motif: np.ndarray, color: int) -> np.ndarray:
    stamp = motif.astype(int)
    return (np.kron(stamp, stamp) * int(color)).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reforge a damaged self-stamp by using the occupied block layout as the stamp pattern itself.

    Input structure:
    - The grid is square, its side length is n^2, and it can be partitioned into an n by n array of n by n blocks.
    - The background is black and there is exactly one foreground color.
    - Blocks with at least two foreground pixels contain damaged copies of one shared n by n motif, while some empty blocks may contain a single noise pixel.

    Rule:
    - Count foreground pixels inside each n by n block.
    - Mark a block as occupied iff it contains at least two foreground pixels; blocks with exactly one foreground pixel are noise and count as empty.
    - Read the occupied-block layout as an n by n binary motif, then redraw that full motif inside every occupied block and clear every other block to black.

    Key insight:
    - The complete motif never appears inside any one block; it is recovered only from the higher-level pattern of which blocks are occupied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    n = _infer_order(g.shape)
    color = _foreground_color(g)
    assert np.all((g == BG) | (g == color))

    counts = _block_counts(g, n)
    motif = counts >= 2
    motif_area = int(motif.sum())
    assert motif_area >= 3

    for br in range(n):
        for bc in range(n):
            rs, cs = _block_slice(n, br, bc)
            block_mask = g[rs, cs] != BG
            visible = int(block_mask.sum())
            if visible >= 2:
                assert visible < motif_area, "copies must be incomplete"
                assert np.all(block_mask <= motif), "occupied blocks must contain motif subsets"
            else:
                assert visible in (0, 1), "empty blocks may only be blank or singleton noise"

    return _full_output(motif, color)


def _random_motif(n: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(n) for c in range(n)]
    area_min = max(4, int(math.ceil(0.35 * n * n)))
    area_max = min(n * n - 2, int(math.floor(0.65 * n * n)))
    if area_min > area_max:
        return None

    for _ in range(200):
        area = random.randint(area_min, area_max)
        chosen = random.sample(cells, area)
        motif = np.zeros((n, n), dtype=bool)
        for r, c in chosen:
            motif[r, c] = True
        if int(motif.any(axis=0).sum()) < 2:
            continue
        if int(motif.any(axis=1).sum()) < 2:
            continue
        return motif
    return None


def _choose_keep_counts(block_count: int, max_keep: int) -> list[int]:
    keep_counts = [2, max_keep]
    while len(keep_counts) < block_count:
        roll = random.random()
        if roll < 0.35:
            keep_counts.append(2)
        elif roll < 0.70:
            keep_counts.append(max_keep)
        else:
            keep_counts.append(random.randint(2, max_keep))
    random.shuffle(keep_counts)
    return keep_counts[:block_count]


def _build_input(motif: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    n = motif.shape[0]
    grid = np.zeros((n * n, n * n), dtype=int)
    motif_cells = [tuple(map(int, rc)) for rc in np.argwhere(motif)]
    common_hole = random.choice(motif_cells)
    remaining = [cell for cell in motif_cells if cell != common_hole]
    keep_counts = _choose_keep_counts(len(motif_cells), len(remaining))

    for (br, bc), keep_count in zip(motif_cells, keep_counts):
        visible = random.sample(remaining, keep_count)
        rs, cs = _block_slice(n, br, bc)
        patch = np.zeros((n, n), dtype=int)
        for r, c in visible:
            patch[r, c] = color
        grid[rs, cs] = patch

    empty_blocks = [tuple(map(int, rc)) for rc in np.argwhere(~motif)]
    max_noise = min(4, len(empty_blocks) - 1)
    noise_count = random.randint(1, max_noise)
    noise_blocks = random.sample(empty_blocks, noise_count)
    for br, bc in noise_blocks:
        rs, cs = _block_slice(n, br, bc)
        nr = random.randrange(n)
        nc = random.randrange(n)
        grid[rs.start + nr, cs.start + nc] = color

    return grid, common_hole, noise_blocks


def _validate_instance(
    inp: np.ndarray,
    motif: np.ndarray,
    color: int,
    common_hole: tuple[int, int],
    noise_blocks: list[tuple[int, int]],
) -> bool:
    try:
        n = _infer_order(inp.shape)
        if n != int(motif.shape[0]):
            return False
        if _foreground_color(inp) != int(color):
            return False
    except AssertionError:
        return False

    counts = _block_counts(inp, n)
    occupied = counts >= 2
    noise = counts == 1
    if not np.array_equal(occupied, motif):
        return False
    if int(noise.sum()) < 1:
        return False
    if int(noise.sum()) != len(noise_blocks):
        return False
    if int((counts == 0).sum()) < 1:
        return False
    if not set(noise_blocks) == {
        (int(r), int(c)) for r, c in np.argwhere(noise)
    }:
        return False

    motif_area = int(motif.sum())
    if motif_area < 4:
        return False

    union_visible = np.zeros_like(motif, dtype=bool)
    saw_extra_damage = False
    for br, bc in np.argwhere(motif):
        rs, cs = _block_slice(n, int(br), int(bc))
        block_mask = inp[rs, cs] != BG
        visible = int(block_mask.sum())
        if not 2 <= visible < motif_area:
            return False
        if not np.all(block_mask <= motif):
            return False
        union_visible |= block_mask
        if visible < motif_area - 1:
            saw_extra_damage = True
    if union_visible[common_hole]:
        return False
    if np.array_equal(union_visible, motif):
        return False
    if not saw_extra_damage:
        return False

    expected = _full_output(motif, color)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(N_CHOICES)
    motif = _random_motif(n)
    if motif is None:
        return None

    color = random.choice(FG_COLORS)
    inp, common_hole, noise_blocks = _build_input(motif, color)
    if _validate_instance(inp, motif, color, common_hole, noise_blocks):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-stamp restoration input")
