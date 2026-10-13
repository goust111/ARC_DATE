from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_ATTEMPTS = 500


def _mixed_parity_mask(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = np.indices(shape)
    return (rows % 2) != (cols % 2)


def _infer_lattice_background(g: np.ndarray) -> int:
    vals = np.unique(g[_mixed_parity_mask(g.shape)])
    assert vals.size == 1, "all mixed odd/even lattice cells must share the background color"
    return int(vals[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two interleaved half-resolution layers and keep only their matching foreground pixels.

    Input structure:
    - The input is a 2H by 2W grid.
    - The even-row/even-column lattice stores image A.
    - The odd-row/odd-column lattice stores image B after a left-right mirror.
    - The mixed-parity cells are all the background color.

    Rule:
    - Extract A from the even/even lattice and mirrored B from the odd/odd lattice.
    - Flip B left-right to align it with A.
    - Return an H by W grid whose cells keep the shared non-background color exactly where A and B agree; all disagreements and background agreements become background.

    Key insight:
    - The apparent separator cells are not noise: the odd/odd lattice is a second mirrored image, so ordinary downsampling loses the verification layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h2, w2 = g.shape
    assert h2 % 2 == 0 and w2 % 2 == 0, "input shape must be 2H by 2W"

    bg = _infer_lattice_background(g)
    a = g[0::2, 0::2]
    b = np.fliplr(g[1::2, 1::2])

    same_foreground = (a == b) & (a != int(bg))
    out = np.full(a.shape, int(bg), dtype=int)
    out[same_foreground] = a[same_foreground]
    return out


# --------------- generator ---------------

_BASE_SHAPES = (
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
)


def _oriented_shape() -> np.ndarray:
    shape = random.choice(_BASE_SHAPES).copy()
    if random.randrange(2):
        shape = np.fliplr(shape)
    shape = np.rot90(shape, random.randrange(4))
    return shape.astype(bool)


def _expanded_box(r0: int, c0: int, h: int, w: int, H: int, W: int) -> tuple[int, int, int, int]:
    return max(0, r0 - 1), max(0, c0 - 1), min(H, r0 + h + 1), min(W, c0 + w + 1)


def _place_agreement_pattern(H: int, W: int, palette: list[int]) -> np.ndarray | None:
    out = np.full((H, W), BG, dtype=int)
    max_objects = min(4, max(2, (H * W) // 24))
    object_count = random.randint(2, max_objects)

    used_colors = random.sample(palette, k=min(len(palette), object_count))
    for idx in range(object_count):
        placed = False
        for _ in range(80):
            shape = _oriented_shape()
            sh, sw = shape.shape
            if sh > H or sw > W:
                continue
            r0 = random.randrange(0, H - sh + 1)
            c0 = random.randrange(0, W - sw + 1)
            er0, ec0, er1, ec1 = _expanded_box(r0, c0, sh, sw, H, W)
            if np.any(out[er0:er1, ec0:ec1] != BG):
                continue
            color = used_colors[idx % len(used_colors)]
            region = out[r0 : r0 + sh, c0 : c0 + sw]
            region[shape] = int(color)
            placed = True
            break
        if not placed:
            return None

    if np.count_nonzero(out != BG) < 6:
        return None
    if len(set(map(int, out[out != BG]))) < 2:
        return None
    if np.array_equal(out, np.fliplr(out)):
        return None
    return out


def _sample_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    count = min(int(count), len(cells))
    if count <= 0:
        return []
    return random.sample(cells, count)


def _different_color(color: int, palette: list[int]) -> int:
    choices = [int(c) for c in palette if int(c) != int(color)]
    return int(random.choice(choices))


def _paint_mismatches(a: np.ndarray, b: np.ndarray, palette: list[int]) -> bool:
    H, W = a.shape
    empty = [(int(r), int(c)) for r, c in np.argwhere((a == BG) & (b == BG))]
    random.shuffle(empty)

    need = max(2, (H * W) // 18)
    required = 3 * need + 2
    if len(empty) < required:
        return False

    a_only = _sample_cells(empty, need)
    empty = [cell for cell in empty if cell not in set(a_only)]
    b_only = _sample_cells(empty, need)
    empty = [cell for cell in empty if cell not in set(b_only)]
    both_diff = _sample_cells(empty, need)
    empty = [cell for cell in empty if cell not in set(both_diff)]

    for r, c in a_only:
        a[r, c] = int(random.choice(palette))
    for r, c in b_only:
        b[r, c] = int(random.choice(palette))
    for r, c in both_diff:
        color = int(random.choice(palette))
        a[r, c] = color
        b[r, c] = _different_color(color, palette)

    # Add a few harmless extra mismatches so the false-positive layer is not too regular.
    extra_max = min(len(empty), max(2, (H * W) // 12))
    extra_count = random.randint(1, extra_max) if extra_max else 0
    for r, c in _sample_cells(empty, extra_count):
        mode = random.randrange(3)
        if mode == 0:
            a[r, c] = int(random.choice(palette))
        elif mode == 1:
            b[r, c] = int(random.choice(palette))
        else:
            color = int(random.choice(palette))
            a[r, c] = color
            b[r, c] = _different_color(color, palette)
    return True


def _interleave(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    H, W = a.shape
    grid = np.full((2 * H, 2 * W), BG, dtype=int)
    grid[0::2, 0::2] = a
    grid[1::2, 1::2] = np.fliplr(b)
    return grid


def _oracle(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    bg = _infer_lattice_background(grid)
    a = grid[0::2, 0::2]
    b = np.fliplr(grid[1::2, 1::2])
    expected = np.full(a.shape, int(bg), dtype=int)
    keep = (a == b) & (a != int(bg))
    expected[keep] = a[keep]
    return expected, a.copy(), b.copy(), int(bg)


def _semantics_ok(grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if grid.shape != (2 * expected.shape[0], 2 * expected.shape[1]):
        return False

    try:
        oracle, a, b, bg = _oracle(grid)
    except AssertionError:
        return False
    if bg != BG:
        return False
    if not np.array_equal(oracle, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    foreground = expected != BG
    if np.count_nonzero(foreground) < 6:
        return False
    if len(set(map(int, expected[foreground]))) < 2:
        return False

    cleared = expected == BG
    has_a_only = np.any(cleared & (a != BG) & (b == BG))
    has_b_only = np.any(cleared & (a == BG) & (b != BG))
    has_diff = np.any(cleared & (a != BG) & (b != BG) & (a != b))
    has_empty = np.any(cleared & (a == BG) & (b == BG))
    if not (has_a_only and has_b_only and has_diff and has_empty):
        return False

    stored_b = grid[1::2, 1::2]
    no_flip = np.full(expected.shape, BG, dtype=int)
    no_flip_keep = (a == stored_b) & (a != BG)
    no_flip[no_flip_keep] = a[no_flip_keep]

    if np.array_equal(a, expected):
        return False
    if np.array_equal(b, expected):
        return False
    if np.array_equal(no_flip, expected):
        return False
    if np.array_equal(stored_b, b):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(6, 15)
    W = random.randint(6, 15)
    palette_size = random.randint(3, 6)
    palette = random.sample(list(COLORS), palette_size)

    expected = _place_agreement_pattern(H, W, palette)
    if expected is None:
        return None

    a = expected.copy()
    b = expected.copy()
    if not _paint_mismatches(a, b, palette):
        return None

    grid = _interleave(a, b)
    if not _semantics_ok(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-lattice comparison input")
