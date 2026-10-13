import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_RNG = np.random.default_rng()
_MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _checkerboard(shape: tuple[int, int], colors: tuple[int, int], phase: int) -> np.ndarray:
    a, b = map(int, colors)
    rr, cc = np.indices(shape)
    return np.where((rr + cc + phase) % 2 == 0, a, b).astype(int)


def _expanded_bboxes(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    seen: set[tuple[int, int, int, int]] = set()
    for top in (0, 1):
        for left in (0, 1):
            for bottom in (0, 1):
                for right in (0, 1):
                    nr0, nc0 = r0 - top, c0 - left
                    nr1, nc1 = r1 + bottom, c1 + right
                    if nr0 < 0 or nc0 < 0 or nr1 >= h or nc1 >= w:
                        continue
                    seen.add((nr0, nc0, nr1, nc1))
    return list(seen)


def _gap_directions(wrong: np.ndarray, gap: np.ndarray) -> list[tuple[int, int]]:
    wrong_bbox = _bbox(wrong)
    if wrong_bbox is None:
        return []

    r0, c0, r1, c1 = wrong_bbox
    if not np.all(wrong[r0 : r1 + 1, c0 : c1 + 1]):
        return []

    h, w = wrong.shape
    directions: list[tuple[int, int]] = []

    if c0 - 1 >= 0:
        expected_gap = np.zeros_like(wrong, dtype=bool)
        expected_gap[r0 : r1 + 1, c0 - 1] = True
        if np.array_equal(gap, expected_gap):
            directions.append((0, 1))

    if c1 + 1 < w:
        expected_gap = np.zeros_like(wrong, dtype=bool)
        expected_gap[r0 : r1 + 1, c1 + 1] = True
        if np.array_equal(gap, expected_gap):
            directions.append((0, -1))

    if r0 - 1 >= 0:
        expected_gap = np.zeros_like(wrong, dtype=bool)
        expected_gap[r0 - 1, c0 : c1 + 1] = True
        if np.array_equal(gap, expected_gap):
            directions.append((1, 0))

    if r1 + 1 < h:
        expected_gap = np.zeros_like(wrong, dtype=bool)
        expected_gap[r1 + 1, c0 : c1 + 1] = True
        if np.array_equal(gap, expected_gap):
            directions.append((-1, 0))

    return directions


def _find_solution(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    uniques = [int(v) for v in np.unique(g)]
    solutions: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    for bg in uniques:
        mask = g != bg
        if not np.any(mask):
            continue

        checker_colors = [color for color in uniques if color != bg]
        if len(checker_colors) != 2:
            continue

        fg_bbox = _bbox(mask)
        assert fg_bbox is not None

        for r0, c0, r1, c1 in _expanded_bboxes(fg_bbox, g.shape):
            outside = np.ones(g.shape, dtype=bool)
            outside[r0 : r1 + 1, c0 : c1 + 1] = False
            if np.any(g[outside] != bg):
                continue

            inside = g[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(np.isin(inside, [bg, *checker_colors])):
                continue

            for phase in (0, 1):
                expected = _checkerboard(inside.shape, tuple(checker_colors), phase)
                wrong = (inside != bg) & (inside != expected)
                gap = inside == bg

                directions = _gap_directions(wrong, gap)
                if not directions:
                    continue

                wrong_bbox = _bbox(wrong)
                assert wrong_bbox is not None
                mr0, mc0, mr1, mc1 = wrong_bbox

                for dr, dc in directions:
                    sr0, sc0 = mr0 - dr, mc0 - dc
                    sr1, sc1 = mr1 - dr, mc1 - dc
                    if sr0 < 0 or sc0 < 0 or sr1 >= inside.shape[0] or sc1 >= inside.shape[1]:
                        continue

                    moved_block = inside[mr0 : mr1 + 1, mc0 : mc1 + 1]
                    source_block = expected[sr0 : sr1 + 1, sc0 : sc1 + 1]
                    if not np.array_equal(moved_block, source_block):
                        continue

                    key = (expected.shape, expected.tobytes())
                    if key not in seen:
                        seen.add(key)
                        solutions.append(expected)

    assert len(solutions) == 1, "expected a unique checkerboard restoration"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the shifted checkerboard sub-rectangle and crop to the checkerboard region.

    Input structure:
    - A background grid contains one large axis-aligned rectangle filled by two non-background colors in an alternating checkerboard pattern.
    - One rectangular sub-block of that checkerboard has been translated by exactly one cell up, down, left, or right.
    - The exposed cells on the block's original side become background, while the translated block now disagrees with the surrounding checkerboard parity.

    Rule:
    - Identify the checkerboard rectangle, the two checker colors, and the parity phase that would make the region perfect.
    - Inside that rectangle, detect the unique rectangular zone whose cells violate checkerboard parity and the adjacent one-cell background strip that marks the vacated side.
    - Use the strip to infer the translation direction, move the shifted block back by one cell, and restore the perfect checkerboard.
    - Return the repaired checkerboard cropped to its bounding rectangle.

    Key insight:
    - The moved block is not labeled directly; it is revealed by a solid rectangle of parity errors plus a one-cell background gap on the opposite side, which together determine both the block's extent and its offset direction.
    """
    return _find_solution(np.asarray(grid, dtype=int))


def _pick_start(lo: int, hi: int, edge: int | None = None) -> int:
    assert lo <= hi
    if edge is not None and lo <= edge <= hi and _RNG.random() < 0.35:
        return int(edge)
    return int(_RNG.integers(lo, hi + 1))


def _gen_attempt() -> np.ndarray | None:
    out_h = int(_RNG.integers(5, 23))
    out_w = int(_RNG.integers(5, 23))

    colors = [int(v) for v in _RNG.choice(np.arange(10), size=3, replace=False)]
    bg, c0, c1 = colors
    phase = int(_RNG.integers(0, 2))
    checker = _checkerboard((out_h, out_w), (c0, c1), phase)

    block_h = int(_RNG.integers(2, out_h))
    block_w = int(_RNG.integers(2, out_w))
    dr, dc = DIRS4[int(_RNG.integers(0, len(DIRS4)))]

    if dr == 1:
        sr0 = _pick_start(0, out_h - block_h - 1, edge=0)
        sc0 = _pick_start(0, out_w - block_w)
    elif dr == -1:
        sr0 = _pick_start(1, out_h - block_h, edge=out_h - block_h)
        sc0 = _pick_start(0, out_w - block_w)
    elif dc == 1:
        sr0 = _pick_start(0, out_h - block_h)
        sc0 = _pick_start(0, out_w - block_w - 1, edge=0)
    else:
        sr0 = _pick_start(0, out_h - block_h)
        sc0 = _pick_start(1, out_w - block_w, edge=out_w - block_w)

    sr1 = sr0 + block_h
    sc1 = sc0 + block_w

    corrupted = checker.copy()
    corrupted[sr0:sr1, sc0:sc1] = bg
    corrupted[sr0 + dr : sr1 + dr, sc0 + dc : sc1 + dc] = checker[sr0:sr1, sc0:sc1]

    extra_h = int(_RNG.integers(0, min(8, 30 - out_h) + 1))
    extra_w = int(_RNG.integers(0, min(8, 30 - out_w) + 1))
    top = int(_RNG.integers(0, extra_h + 1))
    left = int(_RNG.integers(0, extra_w + 1))
    bottom = extra_h - top
    right = extra_w - left

    grid = np.full((out_h + top + bottom, out_w + left + right), bg, dtype=int)
    grid[top : top + out_h, left : left + out_w] = corrupted

    if np.array_equal(corrupted, checker):
        return None

    try:
        solved = _find_solution(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, checker):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-shift instance")
