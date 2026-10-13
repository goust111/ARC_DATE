import random
from collections import deque

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

MAX_GENERATION_ATTEMPTS = 400
_BRANCH_FACTORS = {
    1: ((1, 1),),
    2: ((1, 2), (2, 1)),
    3: ((1, 3), (3, 1)),
    4: ((2, 2),),
}
_GEN_CALL_COUNT = 0


def _detect_bars(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    row_ids = np.flatnonzero(np.all(np.isin(g, (BLUE, GREEN)), axis=1))
    col_ids = np.flatnonzero(np.all(np.isin(g, (RED, GREEN)), axis=0))

    assert 1 <= len(row_ids) <= 3
    assert 1 <= len(col_ids) <= 3

    for r in row_ids:
        for c in col_ids:
            assert g[r, c] == GREEN

    non_col_ids = np.setdiff1d(np.arange(g.shape[1]), col_ids, assume_unique=True)
    non_row_ids = np.setdiff1d(np.arange(g.shape[0]), row_ids, assume_unique=True)
    if non_col_ids.size:
        assert np.all(g[np.ix_(row_ids, non_col_ids)] == BLUE)
    if non_row_ids.size:
        assert np.all(g[np.ix_(non_row_ids, col_ids)] == RED)

    crossings = len(row_ids) * len(col_ids)
    assert 1 <= crossings <= 4
    return row_ids, col_ids, crossings


def _assemble_output(grid: np.ndarray, crossings: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((2 * h, 2 * w), GREY, dtype=int)
    out[h:, w:] = g
    if crossings >= 2:
        out[h:, :w] = np.fliplr(g)
    if crossings >= 3:
        out[:h, w:] = np.flipud(g)
    if crossings == 4:
        out[:h, :w] = np.rot90(g, 2)
    return out


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _orange_component_sizes(grid: np.ndarray) -> list[int]:
    mask = np.asarray(grid == ORANGE, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            size = 0
            while q:
                cr, cc = q.popleft()
                size += 1
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sizes.append(size)
    return sizes


def _random_orange_shape() -> np.ndarray:
    for _ in range(200):
        area = random.randint(3, 5)
        dims = [(h, w) for h in range(2, 4) for w in range(2, 4) if h * w >= area]
        h, w = random.choice(dims)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set()
        for cell in _neighbors4(*start):
            frontier.add(cell)

        while int(mask.sum()) < area and frontier:
            valid_frontier = [
                (r, c)
                for r, c in frontier
                if 0 <= r < h and 0 <= c < w and not mask[r, c]
            ]
            if not valid_frontier:
                break
            cell = random.choice(valid_frontier)
            mask[cell] = True
            frontier.discard(cell)
            for nbr in _neighbors4(*cell):
                frontier.add(nbr)

        if int(mask.sum()) != area:
            continue

        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if min(cropped.shape) < 2:
            continue
        if np.array_equal(cropped, np.fliplr(cropped)):
            continue
        if np.array_equal(cropped, np.flipud(cropped)):
            continue
        if np.array_equal(cropped, np.rot90(cropped, 2)):
            continue
        return cropped
    raise RuntimeError("failed to generate a non-symmetric orange shape")


def _candidate_shape_positions(
    h: int,
    w: int,
    row_ids: list[int],
    col_ids: list[int],
    shape: np.ndarray,
) -> list[tuple[int, int]]:
    row_bounds = [-1, *row_ids, h]
    col_bounds = [-1, *col_ids, w]
    shape_h, shape_w = shape.shape
    placements: list[tuple[int, int]] = []

    for i in range(len(row_bounds) - 1):
        row_lo = row_bounds[i] + 1
        row_hi = row_bounds[i + 1] - 1
        if row_lo > row_hi:
            continue
        top_margin = 1 if i > 0 else 0
        bottom_margin = 1 if i < len(row_bounds) - 2 else 0
        top_start = row_lo + top_margin
        top_end = row_hi - bottom_margin - shape_h + 1
        if top_start > top_end:
            continue

        for j in range(len(col_bounds) - 1):
            col_lo = col_bounds[j] + 1
            col_hi = col_bounds[j + 1] - 1
            if col_lo > col_hi:
                continue
            left_margin = 1 if j > 0 else 0
            right_margin = 1 if j < len(col_bounds) - 2 else 0
            left_start = col_lo + left_margin
            left_end = col_hi - right_margin - shape_w + 1
            if left_start > left_end:
                continue

            for top in range(top_start, top_end + 1):
                for left in range(left_start, left_end + 1):
                    placements.append((top, left))
    return placements


def _needed_transforms(crossings: int) -> tuple[str, ...]:
    if crossings == 1:
        return ()
    if crossings == 2:
        return ("lr",)
    if crossings == 3:
        return ("lr", "ud")
    return ("lr", "ud", "rot")


def _has_required_asymmetry(grid: np.ndarray, crossings: int) -> bool:
    if "lr" in _needed_transforms(crossings) and np.array_equal(grid, np.fliplr(grid)):
        return False
    if "ud" in _needed_transforms(crossings) and np.array_equal(grid, np.flipud(grid)):
        return False
    if "rot" in _needed_transforms(crossings) and np.array_equal(grid, np.rot90(grid, 2)):
        return False
    return True


def _validate_generated_input(grid: np.ndarray, expected_crossings: int) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 15

    row_ids, col_ids, crossings = _detect_bars(g)
    assert crossings == expected_crossings

    orange_sizes = _orange_component_sizes(g)
    assert orange_sizes and orange_sizes == [sum(orange_sizes)]
    assert orange_sizes[0] in (3, 4, 5)

    for r, c in np.argwhere(g == ORANGE):
        assert r not in set(row_ids)
        assert c not in set(col_ids)
        assert abs(r - row_ids).min() >= 2
        assert abs(c - col_ids).min() >= 2

    assert _has_required_asymmetry(g, crossings)

    branch_outputs = {n: _assemble_output(g, n) for n in (1, 2, 3, 4)}
    target = branch_outputs[crossings]
    for n, out in branch_outputs.items():
        if n != crossings:
            assert not np.array_equal(out, target)

    assert np.array_equal(main(g), target)


def _next_crossing_target() -> int:
    global _GEN_CALL_COUNT
    crossings = (1, 2, 3, 4)[_GEN_CALL_COUNT % 4]
    _GEN_CALL_COUNT += 1
    return crossings


def _attempt_generate(crossings: int) -> np.ndarray | None:
    row_count, col_count = random.choice(_BRANCH_FACTORS[crossings])
    h = random.randint(max(7, row_count + 4), 15)
    w = random.randint(max(7, col_count + 4), 15)
    shape = _random_orange_shape()

    for _ in range(80):
        row_ids = sorted(random.sample(range(1, h - 1), row_count))
        col_ids = sorted(random.sample(range(1, w - 1), col_count))
        placements = _candidate_shape_positions(h, w, row_ids, col_ids, shape)
        if not placements:
            continue

        grid = np.full((h, w), GREY, dtype=int)
        for r in row_ids:
            grid[r, :] = BLUE
        for c in col_ids:
            grid[:, c] = RED
        for r in row_ids:
            for c in col_ids:
                grid[r, c] = GREEN

        top, left = random.choice(placements)
        shape_h, shape_w = shape.shape
        patch = grid[top : top + shape_h, left : left + shape_w]
        patch[shape] = ORANGE

        try:
            _validate_generated_input(grid, crossings)
        except AssertionError:
            continue
        return grid
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count row-column crossings and encode that count by how many mirrored quadrants are populated.

    Input structure:
    - The background is grey.
    - One to three full blue row bars and one to three full red column bars partition the grid.
    - Every row-column intersection is green, and a small 4-connected orange shape sits entirely away from the bars inside one partition region.

    Rule:
    - Detect the complete blue row bars and red column bars, then count their green intersection points.
    - Build a grid with doubled height and doubled width, treating the input as the lower-right quadrant.
    - If the crossing count is 1, keep only the lower-right original quadrant; if it is 2, also fill the lower-left with a left-right mirror; if it is 3, also fill the upper-right with an up-down mirror; if it is 4, also fill the upper-left with the 180-degree rotation.

    Key insight:
    - The row-column crossings are not only objects to find; their count is the code that selects how many mirrored quadrants appear in the enlarged output.
    """
    g = np.asarray(grid, dtype=int)
    _row_ids, _col_ids, crossings = _detect_bars(g)
    return _assemble_output(g, crossings)


def generate_input() -> np.ndarray:
    target_crossings = _next_crossing_target()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(target_crossings)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input for crossing count {target_crossings}")
