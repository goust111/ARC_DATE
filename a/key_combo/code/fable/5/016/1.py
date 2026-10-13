from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_GRAY = 5
_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    """Return the modal color, following the ARC background convention."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components as (row, column) coordinate lists."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    height, width = mask.shape
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        seen[row, col] = True
        queue = deque([(row, col)])
        component: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for dr, dc in _DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(component)
    return result


def _solid_square(grid: np.ndarray, color: int) -> tuple[int, int, int] | None:
    """Recognize one color as one filled square and return (row, col, side)."""
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    side = r1 - r0 + 1
    if side != c1 - c0 + 1 or int(coords.shape[0]) != side * side:
        return None
    if not bool((grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)).all()):
        return None
    return r0, c0, side


def _parse(grid: np.ndarray) -> tuple[int, int, list[dict[str, int]]]:
    """Identify the background, river, and differently sized mover squares."""
    bg = _background(grid)
    counts = Counter(map(int, grid.ravel()))
    river_candidates = [(count, color) for color, count in counts.items() if color not in {bg, _GRAY}]
    assert river_candidates, "a non-background river color is required"
    river = int(max(river_candidates)[1])

    movers: list[dict[str, int]] = []
    for color in counts:
        if color in {bg, river, _GRAY}:
            continue
        square = _solid_square(grid, int(color))
        assert square is not None, "every non-river object must be one solid square"
        row, col, side = square
        movers.append({"color": int(color), "row": row, "col": col, "side": side})

    assert len(movers) >= 3, "the branching width sieve needs at least three movers"
    assert len({m["side"] for m in movers}) == len(movers), "mover sizes must be distinct"
    movers.sort(key=lambda m: m["side"], reverse=True)
    return bg, river, movers


def _reachable_placements(
    grid: np.ndarray,
    river: int,
    mover: dict[str, int],
) -> set[tuple[int, int]]:
    """BFS over top-left placements whose whole square lies on free river/markers."""
    height, width = grid.shape
    side = int(mover["side"])
    sr, sc = int(mover["row"]), int(mover["col"])

    free = (grid == int(river)) | (grid == _GRAY)
    free[sr : sr + side, sc : sc + side] = True
    ph, pw = height - side + 1, width - side + 1
    assert ph > 0 and pw > 0

    # Integral-image window sums make the placement test independent of color ids.
    integral = np.pad(free.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    window = (
        integral[side:, side:]
        - integral[:-side, side:]
        - integral[side:, :-side]
        + integral[:-side, :-side]
    )
    feasible = window == side * side
    assert 0 <= sr < ph and 0 <= sc < pw and bool(feasible[sr, sc])

    reached = {(sr, sc)}
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ph and 0 <= cc < pw and feasible[rr, cc] and (rr, cc) not in reached:
                reached.add((rr, cc))
                queue.append((rr, cc))
    return reached


def _destination_candidates(
    grid: np.ndarray,
    river: int,
    mover: dict[str, int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """List (gray marker, top-left placement) pairs reachable by this mover."""
    side = int(mover["side"])
    markers = [tuple(map(int, p)) for p in np.argwhere(grid == _GRAY)]
    reached = _reachable_placements(grid, river, mover)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for marker in markers:
        mr, mc = marker
        for row, col in reached:
            if row <= mr < row + side and col <= mc < col + side:
                candidates.append((marker, (row, col)))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route differently sized squares through a branching river to their uniquely reachable gray goals.

    Input structure:
    - The modal color is background; one large connected color is the branching river.
    - At least three distinctly colored, distinctly sized solid squares rest on the river.
    - Single gray cells mark goals in terminal pockets beyond passages of different widths.

    Rule:
    - Process squares from largest to smallest and search in top-left-placement space using 4-neighbor slides.
    - A placement is legal only when the complete square covers river or a gray marker and no other square.
    - Exactly one reachable placement covers a remaining marker; refill the source with river and place the square there.
    - A landed square remains an obstacle while all smaller squares are routed.

    Key insight:
    - Passage widths form a size sieve, while an earlier large square closes an otherwise usable route, so both size order and collision state are necessary.
    """
    work = np.asarray(grid, dtype=int).copy()
    _bg, river, movers = _parse(work)

    for mover in movers:
        candidates = _destination_candidates(work, river, mover)
        assert len(candidates) == 1, "each mover must have exactly one reachable marker placement"
        _marker, (dst_r, dst_c) = candidates[0]
        src_r, src_c, side = int(mover["row"]), int(mover["col"]), int(mover["side"])
        work[src_r : src_r + side, src_c : src_c + side] = int(river)
        work[dst_r : dst_r + side, dst_c : dst_c + side] = int(mover["color"])

    assert not bool((work == _GRAY).any()), "every generated goal must be consumed"
    return work


def _base_geometry(include_four: bool) -> tuple[np.ndarray, dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    """Build a variable-length canonical river plus known sources and markers."""
    gate_col = random.randint(13, 15)
    end_col = random.randint(gate_col + 8, min(27, gate_col + 11))
    branch_top = random.randint(2, 4)
    root_bottom = random.randint(17, 18)
    d4_end = root_bottom + random.randint(3, 4)
    height = max(23, d4_end + 2)
    width = end_col + 2

    river = np.zeros((height, width), dtype=bool)
    river[3:root_bottom, 2:9] = True                 # common launch room
    river[8:11, 8:11] = True                        # width-3 throat rejects size 4
    river[8:12, 10:gate_col] = True                 # four-wide fork approach
    river[8:12, gate_col : gate_col + 3] = True     # size-3 gate + one-cell bypass
    river[10:12, gate_col + 3 : end_col + 1] = True # downstream size-2 terminal
    river[branch_top:9, 10:12] = True               # size-2 upward dead end
    if include_four:
        river[root_bottom - 1 : d4_end + 1, 3:7] = True

    markers = {
        3: (8, gate_col + 2),
        2: (branch_top, 10),
        1: (10, end_col),
    }
    if include_four:
        markers[4] = (d4_end, 6)

    # These separated placements leave every larger mover a route out of the room.
    sources = {
        3: (9, 3),
        2: (5, 5),
        1: (15, 7),
    }
    if include_four:
        # The largest mover starts below the others, beside its downward branch;
        # this keeps all four source squares visibly separate and lets it leave
        # without using a corridor occupied by a smaller, not-yet-moved square.
        sources[4] = (12, 3)
        sources[1] = (16, 7)
    else:
        # Randomize the empty upper area when there is no fourth square.
        sources[3] = random.choice([(8, 3), (9, 3), (10, 3)])
        sources[2] = random.choice([(4, 5), (5, 5), (6, 5)])
    return river, sources, markers


def _paint_instance(
    river_mask: np.ndarray,
    sources: dict[int, tuple[int, int]],
    markers: dict[int, tuple[int, int]],
) -> np.ndarray:
    """Choose a palette, render the puzzle, then apply a random dihedral view."""
    palette = list(range(10))
    palette.remove(_GRAY)
    bg, river = random.sample(palette, 2)
    mover_colors = random.sample([c for c in palette if c not in {bg, river}], len(sources))

    # Random padding varies canvas size and absolute placement without changing topology.
    top = random.randint(0, 2)
    bottom = random.randint(0, 2)
    left = random.randint(0, 2)
    right = random.randint(0, 2)
    while river_mask.shape[0] + top + bottom > 30:
        bottom = max(0, bottom - 1)
    while river_mask.shape[1] + left + right > 30:
        right = max(0, right - 1)

    shifted = np.pad(river_mask, ((top, bottom), (left, right)), constant_values=False)
    grid = np.full(shifted.shape, int(bg), dtype=int)
    grid[shifted] = int(river)

    for row, col in markers.values():
        grid[row + top, col + left] = _GRAY
    for color, side in zip(mover_colors, sorted(sources, reverse=True)):
        row, col = sources[side]
        row, col = row + top, col + left
        grid[row : row + side, col : col + side] = int(color)

    grid = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        grid = np.fliplr(grid)
    return np.ascontiguousarray(grid)


def _semantic_checks(grid: np.ndarray) -> bool:
    """Check the stated rule, uniqueness, non-degeneracy, and obstacle dependence."""
    try:
        bg, river, movers = _parse(grid)
        if bg == river or len(movers) not in {3, 4}:
            return False
        if max(grid.shape) > 30 or min(grid.shape) < 12:
            return False

        # The visible river/markers/square footprints must be one connected substrate.
        substrate = (grid == river) | (grid == _GRAY)
        for m in movers:
            r, c, n = m["row"], m["col"], m["side"]
            substrate[r : r + n, c : c + n] = True
        if len(_components(substrate)) != 1:
            return False

        work = grid.copy()
        landed: dict[int, tuple[int, int, int, int]] = {}
        for mover in movers:
            candidates = _destination_candidates(work, river, mover)
            if len(candidates) != 1:
                return False
            _marker, (dr, dc) = candidates[0]
            sr, sc, side, color = mover["row"], mover["col"], mover["side"], mover["color"]
            work[sr : sr + side, sc : sc + side] = river
            work[dr : dr + side, dc : dc + side] = color
            landed[side] = (dr, dc, side, color)

        if bool((work == _GRAY).any()) or np.array_equal(work, grid):
            return False
        if not np.array_equal(main(grid), work):
            return False

        # Counterfactual: the landed size-3 square is what removes the small goal
        # from the size-2 square's reachable target set.  Keep its goal consumed,
        # but turn the square itself back into river.
        by_side = {m["side"]: m for m in movers}
        if 3 not in landed or 2 not in by_side:
            return False
        counter = grid.copy()
        # Apply any optional size-4 move normally first.
        if 4 in by_side:
            m4 = by_side[4]
            cand4 = _destination_candidates(counter, river, m4)
            if len(cand4) != 1:
                return False
            marker4, (r4, c4) = cand4[0]
            counter[m4["row"] : m4["row"] + 4, m4["col"] : m4["col"] + 4] = river
            counter[r4 : r4 + 4, c4 : c4 + 4] = m4["color"]
            if counter[marker4] == _GRAY:
                return False

        m3 = by_side[3]
        cand3 = _destination_candidates(counter, river, m3)
        if len(cand3) != 1:
            return False
        marker3, (r3, c3) = cand3[0]
        counter[m3["row"] : m3["row"] + 3, m3["col"] : m3["col"] + 3] = river
        counter[marker3] = river  # consume the goal without leaving the blocking square

        medium = by_side[2].copy()
        medium_candidates = _destination_candidates(counter, river, medium)
        if len({marker for marker, _place in medium_candidates}) < 2:
            return False

        # Wrong order is visibly ambiguous, proving that descending size is not cosmetic.
        initial_medium = _destination_candidates(grid, river, by_side[2])
        if len({marker for marker, _place in initial_medium}) < 2:
            return False
        return True
    except (AssertionError, ValueError, IndexError):
        return False


def generate_input() -> np.ndarray:
    """Generate varied, oracle-checked examples of the width-sieve river puzzle."""
    for _ in range(200):
        include_four = random.random() < 0.5
        river_mask, sources, markers = _base_geometry(include_four)
        grid = _paint_instance(river_mask, sources, markers)
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a semantically valid width-sieve river")
