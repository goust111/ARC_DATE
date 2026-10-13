import random

import numpy as np


BG = 0
VALID_GAPS = (1, 3, 7)
GEN_ATTEMPTS = 400
PAIR_COUNT_OPTIONS = (3, 4)
PLACEMENT_ATTEMPTS = 200
PALETTE = tuple(range(1, 10))


def _axis_delta(axis: str, sign: int) -> tuple[int, int]:
    if axis == "h":
        return 0, sign
    return sign, 0


def _seed_pair_from_center(center: tuple[int, int], orientation: str, gap: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = center
    offset = (gap + 1) // 2
    if orientation == "h":
        return (r, c - offset), (r, c + offset)
    return (r - offset, c), (r + offset, c)


def _extend_branch(
    cells: set[tuple[int, int]],
    start: tuple[int, int],
    axis: str,
    sign: int,
    length: int,
) -> None:
    if length <= 0:
        return

    dr, dc = _axis_delta(axis, sign)
    ray = []
    sr, sc = start
    for step in range(1, length + 1):
        rr = sr + dr * step
        cc = sc + dc * step
        ray.append((rr, cc))
        cells.add((rr, cc))

    child_len = length // 2
    if child_len == 0:
        return

    mid = ray[(length - 1) // 2]
    child_axis = "v" if axis == "h" else "h"
    _extend_branch(cells, mid, child_axis, -1, child_len)
    _extend_branch(cells, mid, child_axis, 1, child_len)


def _tree_points(center: tuple[int, int], orientation: str, gap: int) -> set[tuple[int, int]]:
    assert gap in VALID_GAPS, f"unsupported seed gap: {gap}"

    r, c = center
    offset = (gap + 1) // 2
    cells: set[tuple[int, int]] = set()

    if orientation == "h":
        for cc in range(c - offset, c + offset + 1):
            cells.add((r, cc))
        branch_axis = "v"
    else:
        for rr in range(r - offset, r + offset + 1):
            cells.add((rr, c))
        branch_axis = "h"

    branch_len = gap // 2
    if branch_len > 0:
        _extend_branch(cells, center, branch_axis, -1, branch_len)
        _extend_branch(cells, center, branch_axis, 1, branch_len)

    return cells


def _expanded_points(points: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _parse_pair(coords: np.ndarray) -> tuple[tuple[int, int], str, int]:
    assert coords.shape == (2, 2), f"expected exactly two seeds, got {coords.shape}"
    (r0, c0), (r1, c1) = [tuple(map(int, coord)) for coord in coords]

    if r0 == r1:
        gap = abs(c1 - c0) - 1
        center = (r0, (c0 + c1) // 2)
        return center, "h", gap
    if c0 == c1:
        gap = abs(r1 - r0) - 1
        center = ((r0 + r1) // 2, c0)
        return center, "v", gap
    raise AssertionError("seed pair must share a row or a column")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a recursive tree from each same-colored seed pair.

    Input structure:
    - Black (0) is the background.
    - Every non-background color appears exactly twice as a seed pair.
    - The two seeds of a pair lie on the same row or the same column.
    - The number of background cells between paired seeds is 1, 3, or 7.

    Rule:
    - Connect each seed pair with a straight trunk in its own color.
    - Let the trunk midpoint be the branch root; from it, extend one perpendicular branch in each opposite direction.
    - The first branch length is half of the seed gap, rounded down, and every recursive child branch uses half of its parent branch length.
    - Each child pair sprouts from the midpoint of its parent branch, rotated by 90 degrees, and recursion stops when the next branch length would be 0.

    Key insight:
    - The puzzle is fully determined by pairing equal-color seeds, reading their orientation from row/column alignment, and decoding recursion depth from the gap size 1/3/7.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.zeros_like(g)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    for color in colors:
        coords = np.argwhere(g == color)
        center, orientation, gap = _parse_pair(coords)
        assert gap in VALID_GAPS, f"unexpected seed gap {gap} for color {color}"

        for r, c in _tree_points(center, orientation, gap):
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "tree exits the canvas"
            assert out[r, c] in (BG, color), "different trees must not overlap"
            out[r, c] = color

    return out


def _choose_gaps(n_pairs: int) -> list[int]:
    gaps = [1, 3, 7]
    while len(gaps) < n_pairs:
        gaps.append(random.choice(VALID_GAPS))
    random.shuffle(gaps)
    return gaps


def _grid_shape(n_pairs: int, n_large: int) -> tuple[int, int]:
    min_side = 18 + 2 * (n_pairs - 3) + 2 * max(0, n_large - 1)
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)
    return h, w


def _place_tree(
    blocked: np.ndarray,
    orientation: str,
    gap: int,
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    h, w = blocked.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        center = (random.randint(1, h - 2), random.randint(1, w - 2))
        points = _tree_points(center, orientation, gap)
        if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in points):
            continue
        if any(blocked[r, c] for r, c in points):
            continue
        return center, points
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_pairs = random.choice(PAIR_COUNT_OPTIONS)
        gaps = _choose_gaps(n_pairs)
        h, w = _grid_shape(n_pairs, sum(gap == 7 for gap in gaps))
        blocked = np.zeros((h, w), dtype=bool)
        grid = np.zeros((h, w), dtype=int)

        specs = [
            {
                "gap": gap,
                "orientation": random.choice(("h", "v")),
                "color": color,
            }
            for gap, color in zip(sorted(gaps, reverse=True), random.sample(PALETTE, n_pairs))
        ]

        placed = []
        failed = False
        for spec in specs:
            result = _place_tree(blocked, spec["orientation"], spec["gap"])
            if result is None:
                failed = True
                break

            center, points = result
            seed_a, seed_b = _seed_pair_from_center(center, spec["orientation"], spec["gap"])
            grid[seed_a] = spec["color"]
            grid[seed_b] = spec["color"]

            for r, c in _expanded_points(points, (h, w)):
                blocked[r, c] = True
            placed.append((center, spec["orientation"], spec["gap"], spec["color"]))

        if failed:
            continue

        if len(placed) != n_pairs:
            continue

        try:
            _ = main(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid recursive-tree input")
