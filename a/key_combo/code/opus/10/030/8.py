from __future__ import annotations

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _center_only(grid: np.ndarray, bg: int, mid: int) -> np.ndarray:
    out = np.full_like(grid, bg)
    out[:, mid] = grid[:, mid]
    return out


def _full_edge_rays(grid: np.ndarray, bg: int, mid: int) -> np.ndarray:
    out = _center_only(grid, bg, mid)
    for r in range(grid.shape[0]):
        color = int(grid[r, mid])
        if color == bg:
            continue
        out[r, :mid] = color
        out[r, mid + 1 :] = color
        out[r, mid] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve the middle-column source pixels, erase all other colored pixels, then extend each source horizontally until an erased position blocks the ray.

    Input structure:
    - The grid has a black-dominant background and an odd number of columns, so there is one vertical center column.
    - Several singleton colored pixels lie exactly on that center column; these are the only ray sources.
    - All other colored singleton pixels lie off the center column and act as hidden blockers after they are erased.

    Rule:
    - Record every non-background pixel outside the center column as a ghost blocker position.
    - Erase all off-center colored pixels and keep only the original center-column pixels.
    - From each surviving center-column pixel, emit one horizontal ray leftward and one rightward in the same row using that source color.
    - Paint background cells along each ray until the next ghost blocker position or the grid edge; the blocker cell itself stays black.
    - If multiple rays would claim the same cell, the source from the smaller row index keeps that cell.

    Key insight:
    - The off-center pixels matter twice: they visibly disappear, but their invisible remembered positions still determine where the surviving center-column rays stop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h > 0 and w > 0 and w % 2 == 1, "expected a non-empty grid with odd width"

    bg = _get_bg(g)
    mid = w // 2

    ghost_blockers = {
        (int(r), int(c))
        for r, c in np.argwhere((g != bg) & (np.arange(w)[None, :] != mid))
    }

    out = _center_only(g, bg, mid)
    claimed = out != bg

    sources = [
        (int(r), int(g[r, mid]))
        for r in range(h)
        if int(g[r, mid]) != bg
    ]

    for r, color in sources:
        for step in (-1, 1):
            c = mid + step
            while 0 <= c < w:
                if (r, c) in ghost_blockers:
                    break
                if not claimed[r, c]:
                    out[r, c] = color
                    claimed[r, c] = True
                c += step

    return out


_PALETTE = tuple(range(1, 10))
_HEIGHTS = tuple(range(9, 18, 2))
_WIDTHS = tuple(range(11, 20, 2))
_MAX_ATTEMPTS = 400


def _sample_spaced_rows(rng: np.random.Generator, height: int, count: int) -> list[int] | None:
    rows = np.arange(1, height - 1)
    for _ in range(80):
        picked = sorted(int(x) for x in rng.choice(rows, size=count, replace=False))
        if all(b - a >= 2 for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _candidate_extra_cells(height: int, width: int, source_rows: set[int], mid: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(height):
        if r in source_rows:
            continue
        for c in range(width):
            if c == mid:
                continue
            cells.append((r, c))
    return cells


def _has_adjacent_same_side(off_center: set[tuple[int, int]], row: int, col: int, mid: int) -> bool:
    side = -1 if col < mid else 1
    for dc in (-1, 1):
        cc = col + dc
        if (row, cc) in off_center and ((cc < mid) == (side < 0)):
            return True
    return False


def _row_side_count(off_center: set[tuple[int, int]], row: int, side: int, mid: int) -> int:
    return sum(1 for rr, cc in off_center if rr == row and ((cc < mid) if side < 0 else (cc > mid)))


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    h, w = g.shape
    assert w % 2 == 1
    bg = _get_bg(g)
    mid = w // 2

    center_rows = [int(r) for r in range(h) if int(g[r, mid]) != bg]
    assert 3 <= len(center_rows) <= 4

    coords = [tuple(map(int, rc)) for rc in np.argwhere(g != bg)]
    assert 8 <= len(coords) <= 12
    assert len(coords) == len(set(coords))

    off_center = {(r, c) for r, c in coords if c != mid}
    assert len(off_center) == len(coords) - len(center_rows)

    for r in center_rows:
        left = [c for rr, c in off_center if rr == r and c < mid]
        right = [c for rr, c in off_center if rr == r and c > mid]
        assert len(left) == 1 and len(right) == 1
        assert left[0] <= mid - 2 and right[0] >= mid + 2

    for r, c in off_center:
        assert not _has_adjacent_same_side(off_center, r, c, mid)
    for row in range(h):
        assert _row_side_count(off_center, row, -1, mid) <= 1
        assert _row_side_count(off_center, row, 1, mid) <= 1


def _validate_semantics(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    mid = g.shape[1] // 2
    out = main(g)
    center_only = _center_only(g, bg, mid)
    edge_rays = _full_edge_rays(g, bg, mid)

    assert not np.array_equal(out, center_only), "ghost logic must change the result beyond center extraction"
    assert not np.array_equal(out, edge_rays), "ghost blockers must shorten at least one ray"
    assert np.all(out[:, mid] == g[:, mid]), "center column must be preserved exactly"

    for r in range(g.shape[0]):
        color = int(g[r, mid])
        if color == bg:
            continue
        left_blocker = max(c for c in range(mid) if int(g[r, c]) != bg)
        right_blocker = min(c for c in range(mid + 1, g.shape[1]) if int(g[r, c]) != bg)
        assert np.all(out[r, left_blocker] == bg)
        assert np.all(out[r, right_blocker] == bg)
        assert np.all(out[r, left_blocker + 1 : mid] == color)
        assert np.all(out[r, mid + 1 : right_blocker] == color)


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.choice(_HEIGHTS))
    width = int(rng.choice(_WIDTHS))
    mid = width // 2

    center_count = 3 if rng.random() < 0.65 else 4
    source_rows = _sample_spaced_rows(rng, height, center_count)
    if source_rows is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    center_colors = rng.choice(_PALETTE, size=center_count, replace=False)
    off_center: set[tuple[int, int]] = set()

    for row, color in zip(source_rows, center_colors):
        grid[row, mid] = int(color)

        left_col = int(rng.integers(0, mid - 1))
        right_col = int(rng.integers(mid + 2, width))
        off_center.add((row, left_col))
        off_center.add((row, right_col))
        grid[row, left_col] = int(rng.choice(_PALETTE))
        grid[row, right_col] = int(rng.choice(_PALETTE))

    if center_count == 3:
        extra_budget = int(rng.integers(0, 4))
        extra_candidates = _candidate_extra_cells(height, width, set(source_rows), mid)
        rng.shuffle(extra_candidates)
        for r, c in extra_candidates:
            if extra_budget == 0:
                break
            side = -1 if c < mid else 1
            if (
                (r, c) in off_center
                or _has_adjacent_same_side(off_center, r, c, mid)
                or _row_side_count(off_center, r, side, mid) >= 1
            ):
                continue
            off_center.add((r, c))
            grid[r, c] = int(rng.choice(_PALETTE))
            extra_budget -= 1

    try:
        _validate_input(grid)
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
