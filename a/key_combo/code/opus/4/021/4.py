import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

N_CHOICES = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 400
PATTERN_ATTEMPTS = 300


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _recolor_anchor_block(pattern: np.ndarray) -> np.ndarray:
    recolored = pattern.copy()
    recolored[recolored == BLUE] = GREEN
    return recolored


def _expand_pattern(pattern: np.ndarray) -> np.ndarray:
    n = int(pattern.shape[0])
    base_block = np.asarray(pattern, dtype=int)
    anchor_block = _recolor_anchor_block(base_block)
    out = np.zeros((n * n, n * n), dtype=int)

    for r in range(n):
        for c in range(n):
            cell = int(base_block[r, c])
            if cell == BLACK:
                continue
            block = anchor_block if cell == RED else base_block
            out[r * n : (r + 1) * n, c * n : (c + 1) * n] = block

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a two-color seed pattern into a self-similar stamp field with anchor-controlled recoloring.

    Input structure:
    - The grid is mostly black background and contains one compact square seed pattern.
    - The seed uses blue pixels as base foreground and red pixels as anchor foreground.
    - The seed pattern's non-black bounding box is an N x N square with no empty border row or column.

    Rule:
    - Crop the seed pattern to its tight N x N bounding box.
    - Treat that crop as an N x N macro-grid: each non-black seed pixel activates one N x N output block at the same local (r, c) position.
    - A blue seed pixel copies the pattern unchanged into its block, while a red seed pixel copies a recolored version where blue becomes green and red stays red.

    Key insight:
    - Red anchor pixels have a dual role: they both select which macro-blocks are populated and decide which of those blocks switch the seed's blue pixels to green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    box = bbox(g != BLACK)
    assert box is not None, "expected a non-empty seed pattern"
    pattern = crop_bbox(g, box)
    assert pattern.shape[0] == pattern.shape[1], "seed bbox must be square"
    assert set(map(int, np.unique(pattern))) <= {BLACK, BLUE, RED}

    return _expand_pattern(pattern)


def _sample_spine_columns(n: int) -> list[int]:
    for _ in range(200):
        if random.random() < 0.5:
            start, end = 0, n - 1
        else:
            start, end = n - 1, 0

        cols = [start]
        for row in range(1, n - 1):
            remaining = (n - 1) - row
            prev = cols[-1]
            options = []
            for step in (-1, 0, 1):
                col = prev + step
                if not (0 <= col < n):
                    continue
                if col - remaining <= end <= col + remaining:
                    options.append(col)
            if not options:
                break
            cols.append(random.choice(options))
        else:
            cols.append(end)
            if min(cols) == 0 and max(cols) == n - 1:
                return cols

    return list(range(n)) if random.random() < 0.5 else list(range(n - 1, -1, -1))


def _frontier_cells(fg: set[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    frontier = set()
    for r, c in fg:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and (rr, cc) not in fg:
                    frontier.add((rr, cc))
    return list(frontier)


def _choose_anchor_cells(fg_cells: list[tuple[int, int]], n: int) -> set[tuple[int, int]] | None:
    interior = [cell for cell in fg_cells if 0 < cell[0] < n - 1 and 0 < cell[1] < n - 1]
    border = [cell for cell in fg_cells if cell not in interior]
    max_anchors = min(3, len(fg_cells) - 1)
    anchor_count = random.randint(1, max_anchors)

    branch = random.choice(("mixed", "interior", "border"))
    if branch == "interior" and interior:
        anchor_count = min(anchor_count, len(interior))
        if anchor_count >= 1:
            return set(random.sample(interior, anchor_count))
    if branch == "border" and border:
        anchor_count = min(anchor_count, len(border))
        if anchor_count >= 1:
            return set(random.sample(border, anchor_count))

    if branch == "mixed" and interior and border and anchor_count >= 2:
        anchors = {random.choice(interior), random.choice(border)}
        remaining = [cell for cell in fg_cells if cell not in anchors]
        while len(anchors) < anchor_count and remaining:
            pick = random.choice(remaining)
            anchors.add(pick)
            remaining = [cell for cell in remaining if cell not in anchors]
        return anchors

    if anchor_count >= 1:
        return set(random.sample(fg_cells, anchor_count))
    return None


def _generate_pattern(n: int) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        cols = _sample_spine_columns(n)
        fg = {(r, cols[r]) for r in range(n)}
        target_area = random.randint(max(3, n), 7)
        while len(fg) < target_area:
            frontier = _frontier_cells(fg, n)
            if not frontier:
                break
            fg.add(random.choice(frontier))

        fg_cells = sorted(fg)
        anchors = _choose_anchor_cells(fg_cells, n)
        if not anchors:
            continue

        pattern = np.zeros((n, n), dtype=int)
        for cell in fg_cells:
            pattern[cell] = RED if cell in anchors else BLUE

        if _pattern_is_valid(pattern):
            return pattern

    raise RuntimeError(f"failed to generate a valid {n}x{n} seed pattern")


def _pattern_is_valid(pattern: np.ndarray) -> bool:
    n = int(pattern.shape[0])
    if pattern.shape != (n, n):
        return False
    if set(map(int, np.unique(pattern))) - {BLACK, BLUE, RED}:
        return False

    fg_mask = pattern != BLACK
    box = bbox(fg_mask)
    if box != (0, 0, n - 1, n - 1):
        return False

    fg_count = int(np.count_nonzero(fg_mask))
    blue_count = int(np.count_nonzero(pattern == BLUE))
    red_count = int(np.count_nonzero(pattern == RED))
    if not (3 <= fg_count <= 7):
        return False
    if blue_count < 1 or not (1 <= red_count <= 3):
        return False

    # Reject uniform-looking border usage so examples cover corner, edge, and inner roles.
    rows = {r for r, _c in np.argwhere(fg_mask)}
    cols = {c for _r, c in np.argwhere(fg_mask)}
    if rows != set(range(n)) or 0 not in cols or (n - 1) not in cols:
        return False

    return True


def _validate_case(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    box = bbox(grid != BLACK)
    if box is None:
        return False
    pattern = crop_bbox(grid, box)
    n = int(pattern.shape[0])
    if pattern.shape != (n, n):
        return False
    if grid.shape != (n * n, n * n):
        return False
    if not _pattern_is_valid(pattern):
        return False

    out = main(grid)
    expected = _expand_pattern(pattern)
    if max(out.shape) > 30 or not np.array_equal(out, expected):
        return False

    fg_count = int(np.count_nonzero(pattern != BLACK))
    blue_count = int(np.count_nonzero(pattern == BLUE))
    red_count = int(np.count_nonzero(pattern == RED))
    if np.count_nonzero(out) != fg_count * fg_count:
        return False
    if np.count_nonzero(out == BLUE) != blue_count * blue_count:
        return False
    if np.count_nonzero(out == GREEN) != red_count * blue_count:
        return False
    if np.count_nonzero(out == RED) != fg_count * red_count:
        return False

    base_block = pattern
    anchor_block = _recolor_anchor_block(pattern)
    if np.array_equal(base_block, anchor_block):
        return False

    for r in range(n):
        for c in range(n):
            block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
            cell = int(pattern[r, c])
            if cell == BLACK and np.any(block):
                return False
            if cell == BLUE and not np.array_equal(block, base_block):
                return False
            if cell == RED and not np.array_equal(block, anchor_block):
                return False

    return np.count_nonzero(out == GREEN) > 0 and np.count_nonzero(out == BLUE) > 0


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n = random.choice(N_CHOICES)
        pattern = _generate_pattern(n)
        grid = np.zeros((n * n, n * n), dtype=int)
        grid[:n, :n] = pattern
        if _validate_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
