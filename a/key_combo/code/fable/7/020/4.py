"""ARC task 4: smoke flows toward the farther end of horizontal shelves."""

from __future__ import annotations

from collections import deque

import numpy as np


BACKGROUND = 0
SHELF = 5
SMOKE = 7
_RNG = np.random.default_rng()


def _shelf_span(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Return the inclusive span of the horizontal shelf containing (r, c)."""
    left = c
    while left > 0 and int(grid[r, left - 1]) == SHELF:
        left -= 1
    right = c
    while right + 1 < grid.shape[1] and int(grid[r, right + 1]) == SHELF:
        right += 1
    return left, right


def _validate_structure(grid: np.ndarray) -> list[tuple[int, int]]:
    assert grid.ndim == 2 and grid.size, "expected a non-empty 2D grid"
    h, w = map(int, grid.shape)
    assert h <= 30 and w <= 30
    assert set(map(int, np.unique(grid))) <= {BACKGROUND, SHELF, SMOKE}
    assert np.count_nonzero(grid == BACKGROUND) > grid.size // 2

    starts = [tuple(map(int, rc)) for rc in np.argwhere(grid == SMOKE)]
    assert len(starts) >= 2, "expected several orange bottom-row heat sources"
    assert all(r == h - 1 for r, _c in starts)

    # Every gray component is a single, uninterrupted horizontal segment.
    seen: set[tuple[int, int]] = set()
    shelf_count = 0
    for r, c in map(tuple, np.argwhere(grid == SHELF)):
        r, c = int(r), int(c)
        if (r, c) in seen:
            continue
        left, right = _shelf_span(grid, r, c)
        assert right - left + 1 >= 2
        cells = {(r, x) for x in range(left, right + 1)}
        seen.update(cells)
        shelf_count += 1
        # No other gray pixel may touch this shelf, even diagonally.
        for rr, cc in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == SHELF:
                        assert nr == r and left <= nc <= right
    assert shelf_count >= 2, "expected several separated shelves"
    return starts


def _grow(grid: np.ndarray, *, mode: str = "far") -> tuple[np.ndarray, list[dict]]:
    """Simulate paths and return both the painted grid and collision trace."""
    starts = _validate_structure(grid)
    h, w = map(int, grid.shape)
    out = grid.copy()
    queue = deque((r, c, 0) for r, c in starts)
    visited: set[tuple[int, int, int]] = set()
    events: list[dict] = []

    while queue:
        r, c, depth = queue.popleft()
        if (r, c, depth) in visited:
            continue
        visited.add((r, c, depth))

        while r > 0:
            above = int(grid[r - 1, c])
            if above == BACKGROUND:
                r -= 1
                out[r, c] = SMOKE
                continue
            assert above == SHELF, "a path may meet only background or a shelf"
            left, right = _shelf_span(grid, r - 1, c)
            dl, dr = c - left, right - c
            if mode == "vertical":
                directions: list[int] = []
            elif mode == "both":
                directions = [-1, 1]
            elif mode == "near":
                directions = [-1] if dl < dr else [1] if dr < dl else [-1, 1]
            else:
                directions = [-1] if dl > dr else [1] if dr > dl else [-1, 1]

            event = {
                "row": r - 1,
                "hit": c,
                "left": left,
                "right": right,
                "dl": dl,
                "dr": dr,
                "directions": tuple(directions),
                "depth": depth,
                "trapped": [],
            }
            events.append(event)

            for direction in directions:
                end = left if direction < 0 else right
                for x in range(c, end + direction, direction):
                    assert int(grid[r, x]) in (BACKGROUND, SMOKE)
                    out[r, x] = SMOKE
                exit_c = end + direction
                if not (0 <= exit_c < w):
                    event["trapped"].append(direction)
                    continue
                assert int(grid[r, exit_c]) in (BACKGROUND, SMOKE)
                out[r, exit_c] = SMOKE
                queue.append((r, exit_c, depth + 1))
            break

    return out, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow orange smoke upward, routing it beneath each shelf toward its farther end.

    Input structure:
    - A black grid contains several orange heat-source pixels only on the bottom row.
    - Several mutually nonadjacent gray objects are one-cell-thick horizontal shelves
      of length at least two.

    Rule:
    - Each source paints vertically upward until the cell above belongs to a shelf.
    - Compare the hit column with that shelf's two endpoints in grid coordinates and
      paint beneath the shelf only toward the farther endpoint, including the first
      cell beyond it; resume rising there.  Equal distances create both paths.
    - A chosen endpoint on the canvas wall has no beyond-end cell, so that path stops
      after retaining its painted under-shelf segment. Shared path cells are painted once.

    Key insight:
    - Every collision recomputes direction from the local shelf endpoints; it is not
      a fixed sidestep, and successive shelves can repeatedly redirect one lineage.
    """
    g = np.asarray(grid, dtype=int)
    return _grow(g, mode="far")[0]


def _gray_segments(grid: np.ndarray) -> list[tuple[int, int, int]]:
    segments = []
    for r in range(grid.shape[0]):
        c = 0
        while c < grid.shape[1]:
            if grid[r, c] != SHELF:
                c += 1
                continue
            left = c
            while c + 1 < grid.shape[1] and grid[r, c + 1] == SHELF:
                c += 1
            segments.append((r, left, c))
            c += 1
    return segments


def _semantic_check(grid: np.ndarray) -> bool:
    """Reject samples that do not visibly require every stated rule."""
    try:
        out, events = _grow(grid, mode="far")
        near, _ = _grow(grid, mode="near")
        both, _ = _grow(grid, mode="both")
        vertical, _ = _grow(grid, mode="vertical")
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False
    # Counterexamples exclude fixed vertical growth, nearer-end flow, and always-split flow.
    if any(np.array_equal(out, wrong) for wrong in (near, both, vertical)):
        return False
    unequal = [e for e in events if e["dl"] != e["dr"]]
    ties = [e for e in events if e["dl"] == e["dr"]]
    trapped = [e for e in events if e["trapped"]]
    if not unequal or not ties or not trapped:
        return False
    if not any(e["depth"] >= 1 for e in events):
        return False
    # Require nonfixed behavior: at least one farther-left and one farther-right decision.
    chosen = {e["directions"] for e in unequal}
    if (-1,) not in chosen or (1,) not in chosen:
        return False
    if not np.array_equal(out[grid != BACKGROUND], grid[grid != BACKGROUND]):
        return False
    return np.count_nonzero((out == SMOKE) & (grid == BACKGROUND)) >= 12


def _can_place_shelf(grid: np.ndarray, r: int, left: int, right: int) -> bool:
    h, w = grid.shape
    if not (0 <= r < h - 1 and 0 <= left < right < w):
        return False
    r0, r1 = max(0, r - 1), min(h, r + 2)
    c0, c1 = max(0, left - 1), min(w, right + 2)
    return not np.any(grid[r0:r1, c0:c1] == SHELF)


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(19, 28))
    w = int(_RNG.integers(24, 31))
    grid = np.zeros((h, w), dtype=int)

    # Left band: a noncentral hit goes to the right, then a higher shelf sends it left.
    left1 = int(_RNG.integers(1, 3))
    len1 = int(_RNG.integers(5, 8))
    right1 = left1 + len1 - 1
    source_chain = left1 + 1
    exit1 = right1 + 1

    right2 = exit1 + int(_RNG.integers(2, 4))
    left2 = exit1 - int(_RNG.integers(4, 6))
    if left2 < 1 or right2 >= w // 2 - 1:
        return None

    # Middle band: an odd shelf hit exactly at its center must bifurcate.
    center = int(_RNG.integers(max(right2 + 4, w // 2 - 1), min(w - 8, w // 2 + 2) + 1))
    radius = int(_RNG.integers(2, 4))
    tie_left, tie_right = center - radius, center + radius

    # Right wall band: hit near the inner end, making the wall endpoint farther.
    trap_len = int(_RNG.integers(5, 8))
    trap_left, trap_right = w - trap_len, w - 1
    source_trap = trap_left + 1
    if tie_right + 2 >= trap_left:
        return None

    trap_row = h - int(_RNG.integers(3, 5))
    lower_row = trap_row - int(_RNG.integers(2, 4))
    tie_row = lower_row - int(_RNG.integers(2, 4))
    upper_row = tie_row - int(_RNG.integers(3, 5))
    if upper_row < 2:
        return None

    shelves = [
        (lower_row, left1, right1),
        (upper_row, left2, right2),
        (tie_row, tie_left, tie_right),
        (trap_row, trap_left, trap_right),
    ]
    for r, left, right in shelves:
        if not _can_place_shelf(grid, r, left, right):
            return None
        grid[r, left : right + 1] = SHELF

    sources = {source_chain, center, source_trap}
    # Vary the control count; extra sources remain structurally ordinary.
    candidates = [c for c in range(1, w - 1) if all(abs(c - x) >= 2 for x in sources)]
    _RNG.shuffle(candidates)
    for c in candidates[: int(_RNG.integers(0, 3))]:
        if all(abs(c - x) >= 2 for x in sources):
            sources.add(int(c))
    for c in sources:
        grid[h - 1, c] = SMOKE

    # Add zero to two separated shelves; they may become meaningful higher collisions.
    for _ in range(int(_RNG.integers(0, 3))):
        for _attempt in range(40):
            length = int(_RNG.integers(2, 6))
            r = int(_RNG.integers(1, h - 2))
            left = int(_RNG.integers(0, w - length + 1))
            right = left + length - 1
            if _can_place_shelf(grid, r, left, right):
                grid[r, left : right + 1] = SHELF
                break

    if bool(_RNG.integers(0, 2)):
        grid = np.fliplr(grid).copy()

    if not (4 <= len(_gray_segments(grid)) <= 6):
        return None
    return grid if _semantic_check(grid) else None


def generate_input() -> np.ndarray:
    """Generate a diverse, branch-complete valid input grid."""
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a semantically complete smoke-flow input")
