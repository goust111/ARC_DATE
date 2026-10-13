"""ARC task: extract the cyclic-shift outlier among separated pattern blocks."""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _background(grid: np.ndarray) -> int:
    """Return the unique most frequent color."""
    counts = Counter(int(x) for x in grid.ravel())
    ordered = counts.most_common()
    assert len(ordered) == 1 or ordered[0][1] > ordered[1][1]
    return int(ordered[0][0])


def _foreground_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    """Return 4-connected non-background components as (row, column) arrays."""
    mask = grid != bg
    seen = np.zeros(grid.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = grid.shape
    for start_r, start_c in np.argwhere(mask):
        start_r, start_c = int(start_r), int(start_c)
        if seen[start_r, start_c]:
            continue
        seen[start_r, start_c] = True
        queue = deque([(start_r, start_c)])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        components.append(np.asarray(cells, dtype=int))
    return components


def _crop_component(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _shift_equivalent(a: np.ndarray, b: np.ndarray) -> bool:
    """Compare two blocks under every row/column cyclic shift in block coordinates."""
    if a.shape != b.shape:
        return False
    height, width = a.shape
    return any(
        np.array_equal(np.roll(a, shift=(dr, dc), axis=(0, 1)), b)
        for dr in range(height)
        for dc in range(width)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Find and crop the sole pattern block outside the common cyclic-shift equivalence class.

    Input structure:
    - A uniquely dominant background separates at least four same-sized, 4-connected multicolor pattern blocks.
    - Every block has the same color counts; all but one are distinct row/column cyclic rolls of one base block.

    Rule:
    - Crop every block in the input-grid coordinate system, then compare whole crops under all combinations of
      cyclic row and cyclic column shifts in each block's own coordinate system.
    - Exactly one block matches no other block, while all remaining blocks are mutually shift-equivalent.
      Return the unmatched block cropped unchanged to its bounding box.

    Key insight:
    - Raw appearance and color histograms cannot reveal the outlier; membership in the common two-dimensional
      cyclic-shift equivalence class can.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)
    components = _foreground_components(g, bg)
    assert len(components) >= 4
    blocks = [_crop_component(g, component) for component in components]

    shape = blocks[0].shape
    assert all(block.shape == shape for block in blocks)
    histograms = [Counter(int(x) for x in block.ravel()) for block in blocks]
    assert all(histogram == histograms[0] for histogram in histograms)

    matches = [[False] * len(blocks) for _ in blocks]
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            matches[i][j] = matches[j][i] = _shift_equivalent(blocks[i], blocks[j])

    outliers = [i for i, row in enumerate(matches) if not any(row)]
    assert len(outliers) == 1
    common = [i for i in range(len(blocks)) if i != outliers[0]]
    assert all(matches[i][j] for pos, i in enumerate(common) for j in common[pos + 1 :])
    return blocks[outliers[0]].copy()


def _all_rolls(block: np.ndarray) -> list[np.ndarray]:
    return [
        np.roll(block, shift=(dr, dc), axis=(0, 1))
        for dr in range(block.shape[0])
        for dc in range(block.shape[1])
    ]


def _make_base(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    """Make an aperiodic dense pattern in which every selected color is visible."""
    area = height * width
    values = colors * (area // len(colors)) + random.sample(colors, area % len(colors))
    random.shuffle(values)
    base = np.asarray(values, dtype=int).reshape(height, width)
    rolls = _all_rolls(base)
    if len({roll.tobytes() for roll in rolls}) != area:
        return None
    return base


def _make_outlier(base: np.ndarray) -> np.ndarray | None:
    """Preserve the exact histogram while leaving the base's complete roll orbit."""
    rolls = {roll.tobytes() for roll in _all_rolls(base)}
    height, width = base.shape
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(100):
        (r1, c1), (r2, c2) = random.sample(cells, 2)
        if base[r1, c1] == base[r2, c2]:
            continue
        candidate = base.copy()
        candidate[r1, c1], candidate[r2, c2] = candidate[r2, c2], candidate[r1, c1]
        if candidate.tobytes() not in rolls:
            return candidate
    return None


def _random_locations(canvas_shape: tuple[int, int], block_shape: tuple[int, int], count: int):
    """Place rectangles with at least one full background row/column between them."""
    height, width = canvas_shape
    block_h, block_w = block_shape
    locations: list[tuple[int, int]] = []
    for _ in range(1000):
        if len(locations) == count:
            return locations
        r = random.randint(1, height - block_h - 1)
        c = random.randint(1, width - block_w - 1)
        if all(
            r + block_h < rr or rr + block_h < r or c + block_w < cc or cc + block_w < c
            for rr, cc in locations
        ):
            locations.append((r, c))
    return None


def _semantic_valid(grid: np.ndarray, expected: np.ndarray, expected_index: int) -> bool:
    """Independently check uniqueness, anti-shortcuts, and the requested equivalence relation."""
    bg = _background(grid)
    components = _foreground_components(grid, bg)
    blocks = [_crop_component(grid, component) for component in components]
    if len(blocks) < 4 or not (0 <= expected_index < len(blocks)):
        return False
    if not np.array_equal(blocks[expected_index], expected):
        return False
    if len({block.tobytes() for block in blocks}) != len(blocks):
        return False  # exact matching cannot replace cyclic comparison
    if any(Counter(map(int, block.ravel())) != Counter(map(int, blocks[0].ravel())) for block in blocks):
        return False  # color statistics cannot identify the answer

    relation = np.eye(len(blocks), dtype=bool)
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            relation[i, j] = relation[j, i] = _shift_equivalent(blocks[i], blocks[j])
    class_sizes = sorted(int(np.count_nonzero(row)) for row in relation)
    if class_sizes != [1] + [len(blocks) - 1] * (len(blocks) - 1):
        return False
    if int(np.count_nonzero(relation[expected_index])) != 1:
        return False
    try:
        return np.array_equal(main(grid), expected)
    except (AssertionError, ValueError):
        return False


def generate_input() -> np.ndarray:
    """Generate diverse, non-degenerate examples satisfying the puzzle semantics."""
    for _ in range(400):
        block_h = random.randint(3, 5)
        block_w = random.randint(3, 6)
        count = random.randint(4, min(6, block_h * block_w))
        palette = random.sample(range(1, 10), random.randint(2, min(4, block_h * block_w)))
        base = _make_base(block_h, block_w, palette)
        if base is None:
            continue
        outlier = _make_outlier(base)
        if outlier is None:
            continue

        shifts = random.sample([(r, c) for r in range(block_h) for c in range(block_w)], count - 1)
        common = [np.roll(base, shift=shift, axis=(0, 1)) for shift in shifts]
        blocks = common + [outlier]
        random.shuffle(blocks)
        expected_index = next(i for i, block in enumerate(blocks) if block is outlier)

        min_h = max(14, 2 * block_h + 5)
        min_w = max(16, 3 * block_w + 6)
        canvas_h = random.randint(min_h, 30)
        canvas_w = random.randint(min_w, 30)
        locations = _random_locations((canvas_h, canvas_w), (block_h, block_w), count)
        if locations is None:
            continue

        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        for block, (r, c) in zip(blocks, locations):
            grid[r : r + block_h, c : c + block_w] = block

        # Component traversal is row-major, so map the shuffled outlier through spatial order.
        spatial_order = sorted(range(count), key=lambda i: locations[i])
        component_index = spatial_order.index(expected_index)
        if _semantic_valid(grid, outlier, component_index):
            return grid
    raise RuntimeError("failed to generate a valid cyclic-shift outlier input")
