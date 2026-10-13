import numpy as np


_RNG = np.random.default_rng()
_MAX_ATTEMPTS = 400

_ROW_RANGE = (10, 17)          # 10..16
_HALF_WIDTH_RANGE = (5, 9)     # 5..8 => total width 11..17
_RECT_COUNT_RANGE = (2, 5)
_CENTER_SEGMENT_COUNT = (0, 3)
_DEVIATION_TOTAL_RANGE = (3, 6)  # 3..5


def _classify_halves(left_proj: np.ndarray, right_half: np.ndarray) -> np.ndarray:
    out = np.zeros_like(right_half, dtype=int)
    both = (left_proj == 1) & (right_half == 1)
    right_only = (left_proj == 0) & (right_half == 1)
    left_only = (left_proj == 1) & (right_half == 0)
    out[both] = 1
    out[right_only] = 2
    out[left_only] = 3
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the mirrored left half against the actual right half and output a three-color diagnosis.

    Input structure:
    - The input is a 2D black background grid with odd width, so the vertical center column is explicit.
    - Blue foreground pixels form an almost left-right symmetric pattern that may also occupy the center column.
    - A few asymmetric pixels exist: some appear only on the right half, and some appear only on the left half.

    Rule:
    - Let center = (width - 1) // 2 and inspect only cells strictly to the right of that center column.
    - For each right-half cell, compare it with its mirror-reflected partner on the left half.
    - Output only the right half: keep blue (1) where both sides are blue, paint red (2) where only the right side is blue, paint green (3) where only the mirrored left side is blue, and keep black (0) where both sides are black.

    Key insight:
    - The green cells are not copied from visible right-half pixels; they are projected from blue pixels that exist only on the left and become visible only after mirror comparison.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert w % 2 == 1, "expected an odd number of columns"
    assert set(np.unique(g)).issubset({0, 1}), "input must use only black and blue"

    center = (w - 1) // 2
    left_proj = np.fliplr(g[:, :center])
    right_half = g[:, center + 1 :]
    assert left_proj.shape == right_half.shape
    return _classify_halves(left_proj, right_half)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _make_base_mask(h: int, half_w: int) -> np.ndarray:
    base = np.zeros((h, half_w), dtype=int)

    rect_count = int(_RNG.integers(*_RECT_COUNT_RANGE))
    for _ in range(rect_count):
        rh = int(_RNG.integers(2, min(h, 6) + 1))
        rw = int(_RNG.integers(1, min(half_w, 4) + 1))
        r0 = int(_RNG.integers(0, h - rh + 1))
        c0 = int(_RNG.integers(0, half_w - rw + 1))
        base[r0 : r0 + rh, c0 : c0 + rw] = 1

    if half_w >= 3 and _RNG.random() < 0.65:
        c = int(_RNG.integers(0, half_w))
        r0 = int(_RNG.integers(0, h - 1))
        r1 = int(_RNG.integers(r0 + 1, h + 1))
        base[r0:r1, c] = 1

    if h >= 3 and _RNG.random() < 0.65:
        r = int(_RNG.integers(0, h))
        c0 = int(_RNG.integers(0, half_w - 1))
        c1 = int(_RNG.integers(c0 + 1, half_w + 1))
        base[r, c0:c1] = 1

    if _RNG.random() < 0.35 and h >= 5 and half_w >= 4:
        hole_h = int(_RNG.integers(1, min(3, h - 2) + 1))
        hole_w = int(_RNG.integers(1, min(2, half_w - 1) + 1))
        r0 = int(_RNG.integers(1, h - hole_h))
        c0 = int(_RNG.integers(1, half_w - hole_w + 1))
        base[r0 : r0 + hole_h, c0 : c0 + hole_w] = 0

    return base


def _boundary_ones(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r, c in np.argwhere(mask == 1):
        if any(mask[nr, nc] == 0 for nr, nc in _neighbors4(int(r), int(c), h, w)):
            cells.append((int(r), int(c)))
    return cells


def _fringe_zeros(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r, c in np.argwhere(mask == 0):
        if any(mask[nr, nc] == 1 for nr, nc in _neighbors8(int(r), int(c), h, w)):
            cells.append((int(r), int(c)))
    return cells


def _sample_deviation_counts(base: np.ndarray) -> tuple[int, int] | None:
    present = int(base.sum())
    absent = int(base.size - present)
    if present < 5 or absent < 2:
        return None

    valid = []
    for total in range(*_DEVIATION_TOTAL_RANGE):
        for left_only in range(1, total):
            right_only = total - left_only
            if present - left_only < 4:
                continue
            if absent < right_only:
                continue
            valid.append((left_only, right_only))
    if not valid:
        return None
    return valid[int(_RNG.integers(0, len(valid)))]


def _center_column(h: int) -> np.ndarray:
    col = np.zeros(h, dtype=int)
    seg_count = int(_RNG.integers(*_CENTER_SEGMENT_COUNT))
    for _ in range(seg_count):
        r0 = int(_RNG.integers(0, h))
        r1 = int(_RNG.integers(r0 + 1, min(h, r0 + 5) + 1))
        col[r0:r1] = 1
    if _RNG.random() < 0.35:
        col[int(_RNG.integers(0, h))] = 1
    return col


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    left_proj: np.ndarray,
    right_half: np.ndarray,
    left_only_count: int,
    right_only_count: int,
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if actual.shape != right_half.shape:
        return False

    colors = set(np.unique(actual))
    if not {1, 2, 3}.issubset(colors):
        return False
    if 0 not in colors:
        return False
    if int(np.sum(actual == 2)) != right_only_count:
        return False
    if int(np.sum(actual == 3)) != left_only_count:
        return False
    if int(np.sum(actual == 1)) < 4:
        return False
    if np.array_equal(actual, right_half):
        return False
    if np.array_equal(actual, left_proj):
        return False
    if np.array_equal(actual, ((left_proj == 1) & (right_half == 1)).astype(int)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(*_ROW_RANGE))
    half_w = int(_RNG.integers(*_HALF_WIDTH_RANGE))
    w = 2 * half_w + 1

    base = _make_base_mask(h, half_w)
    density = float(base.mean())
    if not (0.18 <= density <= 0.58):
        return None

    counts = _sample_deviation_counts(base)
    if counts is None:
        return None
    left_only_count, right_only_count = counts

    left_candidates = _boundary_ones(base)
    if len(left_candidates) < left_only_count:
        left_candidates = [tuple(map(int, rc)) for rc in np.argwhere(base == 1)]

    right_candidates = _fringe_zeros(base)
    if len(right_candidates) < right_only_count:
        right_candidates = [tuple(map(int, rc)) for rc in np.argwhere(base == 0)]

    if len(left_candidates) < left_only_count or len(right_candidates) < right_only_count:
        return None

    left_proj = base.copy()
    right_half = base.copy()

    left_only_pick = _RNG.choice(len(left_candidates), size=left_only_count, replace=False)
    for idx in np.atleast_1d(left_only_pick):
        r, c = left_candidates[int(idx)]
        right_half[r, c] = 0

    right_only_pick = _RNG.choice(len(right_candidates), size=right_only_count, replace=False)
    for idx in np.atleast_1d(right_only_pick):
        r, c = right_candidates[int(idx)]
        left_proj[r, c] = 0
        right_half[r, c] = 1

    expected = _classify_halves(left_proj, right_half)
    if int(np.sum(expected == 1)) < 4 or int(np.sum(expected == 0)) < 4:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[:, :half_w] = np.fliplr(left_proj)
    grid[:, half_w] = _center_column(h)
    grid[:, half_w + 1 :] = right_half

    if not _semantic_ok(grid, expected, left_proj, right_half, left_only_count, right_only_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-bias diagnosis input")
