"""ARC task: transfer bead colors encountered on a blocker-filtered track path."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8
MAROON = 9
BEAD_COLORS = (1, 2, 3, 6)

N4 = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Use the common ARC (row, column) coordinate convention."""
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _forbidden(shape: tuple[int, int], blockers: list[tuple[int, int]]) -> np.ndarray:
    """Return the blocker cells and their Chebyshev-radius-one neighborhoods."""
    mask = np.zeros(shape, dtype=bool)
    for r, c in blockers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    mask[rr, cc] = True
    return mask


def _track_mask(grid: np.ndarray) -> np.ndarray:
    """All non-background colored cells except blockers and grey placeholders are track."""
    return ~np.isin(grid, (BLACK, GREY, ORANGE))


def _unique_ordered_path(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    """Find the ordered path and prove that it is the only simple start-end path.

    In an undirected graph, a chosen start-end path is the unique simple path iff
    every edge on it is a bridge.  Tarjan's bridge test also permits arbitrary
    dead branches and cycles hanging from one path vertex.
    """
    assert passable[start] and passable[end]
    shape = passable.shape
    discovery = np.full(shape, -1, dtype=int)
    low = np.full(shape, -1, dtype=int)
    bridges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    clock = 0

    def edge(a: tuple[int, int], b: tuple[int, int]):
        return (a, b) if a < b else (b, a)

    def visit(node: tuple[int, int], parent: tuple[int, int] | None) -> None:
        nonlocal clock
        r, c = node
        discovery[r, c] = low[r, c] = clock
        clock += 1
        for dr, dc in N4:
            nxt = (r + dr, c + dc)
            nr, nc = nxt
            if not _in_bounds(shape, nr, nc) or not passable[nr, nc]:
                continue
            if discovery[nr, nc] < 0:
                visit(nxt, node)
                low[r, c] = min(low[r, c], low[nr, nc])
                if low[nr, nc] > discovery[r, c]:
                    bridges.add(edge(node, nxt))
            elif nxt != parent:
                low[r, c] = min(low[r, c], discovery[nr, nc])

    visit(start, None)
    assert discovery[end] >= 0, "the two endpoints must be connected after exclusion"

    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue and end not in parent:
        r, c = queue.popleft()
        for dr, dc in N4:
            nxt = (r + dr, c + dc)
            nr, nc = nxt
            if (
                _in_bounds(shape, nr, nc)
                and passable[nr, nc]
                and nxt not in parent
            ):
                parent[nxt] = (r, c)
                queue.append(nxt)

    path = [end]
    while path[-1] != start:
        previous = parent.get(path[-1])
        assert previous is not None
        path.append(previous)
    path.reverse()
    assert all(edge(a, b) in bridges for a, b in zip(path, path[1:])), (
        "the safe start-end simple path must be unique"
    )
    return path


def _grey_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """Extract grey placeholder blocks using four-connectivity."""
    unseen = {tuple(map(int, p)) for p in np.argwhere(grid == GREY)}
    components: list[list[tuple[int, int]]] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        queue = deque([seed])
        component = [seed]
        while queue:
            r, c = queue.popleft()
            for dr, dc in N4:
                nxt = (r + dr, c + dc)
                if nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
                    component.append(nxt)
        components.append(component)
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Transfer, in walking order, safe-path bead colors to the grey placeholder blocks.

    Input structure:
    - A black background contains a branched four-connected teal track, one
      yellow start, one maroon end, colored one-cell beads, orange blockers,
      and a separate vertical column of grey filled-square placeholders.

    Rule:
    - Remove from the track every cell in an orange blocker's eight-neighborhood.
      Follow the unique remaining simple path from yellow to maroon, record the
      non-teal bead colors in encounter order, and recolor the separated grey
      blocks from top to bottom with those colors.  Preserve all other cells.

    Key insight:
    - Orange diagonal exclusion selects the safe route; only bead order on that
      route matters, so geometrically tempting beads on severed dead branches
      must be ignored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= set(range(10))

    starts = [tuple(map(int, p)) for p in np.argwhere(g == YELLOW)]
    ends = [tuple(map(int, p)) for p in np.argwhere(g == MAROON)]
    blockers = [tuple(map(int, p)) for p in np.argwhere(g == ORANGE)]
    assert len(starts) == len(ends) == 1
    assert blockers, "at least one orange blocker is required"

    forbidden = _forbidden(g.shape, blockers)
    assert not forbidden[starts[0]] and not forbidden[ends[0]]
    safe_track = _track_mask(g) & ~forbidden
    path = _unique_ordered_path(safe_track, starts[0], ends[0])

    structural_colors = {TEAL, YELLOW, MAROON}
    bead_colors = [int(g[r, c]) for r, c in path if int(g[r, c]) not in structural_colors]
    assert bead_colors, "the safe path must contain colored beads"
    assert all(color not in (BLACK, GREY, ORANGE) for color in bead_colors)

    placeholders = _grey_components(g)
    assert len(placeholders) == len(bead_colors), (
        "placeholder count must equal the safe-path bead count"
    )

    boxes = []
    for component in placeholders:
        rows = [p[0] for p in component]
        cols = [p[1] for p in component]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        assert r1 - r0 == c1 - c0, "each placeholder must be square"
        assert len(component) == (r1 - r0 + 1) * (c1 - c0 + 1), (
            "each placeholder must be filled"
        )
        boxes.append((r0, r1, c0, c1, component))
    boxes.sort(key=lambda item: item[0])
    assert all(a[1] < b[0] for a, b in zip(boxes, boxes[1:])), (
        "placeholder blocks must have an unambiguous top-to-bottom order"
    )

    out = g.copy()
    for box, color in zip(boxes, bead_colors):
        for r, c in box[4]:
            out[r, c] = color
    return out


def _snake_path(r0: int, left: int, right: int) -> tuple[list[tuple[int, int]], list[int]]:
    """Create a three-run snake with six-row gaps for visible detours."""
    levels = [r0, r0 + 6, r0 + 12]
    path = [(levels[0], left)]

    def extend(target: tuple[int, int]) -> None:
        r, c = path[-1]
        tr, tc = target
        while (r, c) != (tr, tc):
            if r != tr:
                r += 1 if tr > r else -1
            else:
                c += 1 if tc > c else -1
            path.append((r, c))

    extend((levels[0], right))
    extend((levels[1], right))
    extend((levels[1], left))
    extend((levels[2], left))
    extend((levels[2], right))
    assert len(path) == len(set(path))
    return path, levels


def _has_unique_path(
    mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> bool:
    try:
        _unique_ordered_path(mask, start, end)
        return True
    except AssertionError:
        return False


def _generate_attempt() -> np.ndarray | None:
    """Build one candidate and validate the intended semantics independently."""
    try:
        height = random.randint(20, 28)
        width = random.randint(24, 30)
        r0 = random.randint(1, height - 14)
        placeholders_on_right = random.choice((False, True))
        if placeholders_on_right:
            left = random.randint(2, 4)
            right = width - random.randint(7, 9)
            placeholder_col = width - 3
        else:
            left = random.randint(6, 8)
            right = width - random.randint(2, 4)
            placeholder_col = 1
        assert right - left >= 10

        base_path, levels = _snake_path(r0, left, right)
        forward = random.choice((False, True))
        ordered_main = base_path if forward else list(reversed(base_path))
        start, end = ordered_main[0], ordered_main[-1]

        # Each selected gap adds a full alternate endpoint route before its
        # orange blocker diagonally cuts one outside corner.
        gap_indices = random.sample((0, 1), k=random.randint(1, 2))
        track = set(base_path)
        blockers: list[tuple[int, int]] = []
        lure_positions: list[tuple[int, int]] = []
        for gap_index in gap_indices:
            safe_row = levels[gap_index]
            bypass_row = safe_row + 3
            track.update((r, left) for r in range(safe_row, bypass_row + 1))
            track.update((r, right) for r in range(safe_row, bypass_row + 1))
            track.update((bypass_row, c) for c in range(left, right + 1))
            if gap_index == 0:
                blockers.append((bypass_row + 1, left - 1))
            else:
                blockers.append((bypass_row + 1, right + 1))
            lure_positions.extend(
                random.sample(
                    [(bypass_row, c) for c in range(left + 2, right - 1)],
                    k=2,
                )
            )

        # Put four or five true beads across all three path runs.  Forward
        # examples include two on the oppositely traversed middle run, ensuring
        # walking order differs from row-major geometric order.
        bead_count = random.randint(4, 5)
        run_candidates = [
            [(r, c) for r, c in ordered_main if r == level and left + 2 <= c <= right - 2]
            for level in levels
        ]
        chosen = {random.choice(run) for run in run_candidates}
        if forward:
            remaining_middle = [p for p in run_candidates[1] if p not in chosen]
            chosen.add(random.choice(remaining_middle))
        all_candidates = set().union(*map(set, run_candidates)) - chosen
        chosen.update(random.sample(sorted(all_candidates), bead_count - len(chosen)))
        ordered_beads = [p for p in ordered_main if p in chosen]
        assert len(ordered_beads) == bead_count
        assert ordered_beads != sorted(ordered_beads)

        colors = list(BEAD_COLORS)
        if bead_count == 5:
            colors.append(random.choice(BEAD_COLORS))
        for _ in range(20):
            random.shuffle(colors)
            color_at = dict(zip(ordered_beads, colors))
            row_major_colors = [color_at[p] for p in sorted(ordered_beads)]
            if row_major_colors != colors:
                break
        else:
            return None

        grid = np.full((height, width), BLACK, dtype=int)
        for r, c in track:
            grid[r, c] = TEAL
        for position, color in color_at.items():
            grid[position] = color
        for position in lure_positions:
            grid[position] = random.choice(BEAD_COLORS)
        grid[start] = YELLOW
        grid[end] = MAROON
        for r, c in blockers:
            assert grid[r, c] == BLACK
            grid[r, c] = ORANGE

        block_size = 2
        gap = random.randint(1, 2)
        total_height = bead_count * block_size + (bead_count - 1) * gap
        top = random.randint(1, height - total_height - 1)
        placeholder_cells: list[list[tuple[int, int]]] = []
        for i in range(bead_count):
            rr = top + i * (block_size + gap)
            cells = [
                (r, c)
                for r in range(rr, rr + block_size)
                for c in range(placeholder_col, placeholder_col + block_size)
            ]
            placeholder_cells.append(cells)
            for r, c in cells:
                assert grid[r, c] == BLACK
                grid[r, c] = GREY

        # Semantic checks: the safe path is exactly the designed walk, the
        # unfiltered graph is ambiguous, every blocker is necessary, and a
        # four-neighbor-only interpretation fails to cut any alternate route.
        initial_track = np.zeros(grid.shape, dtype=bool)
        for p in track:
            initial_track[p] = True
        full_forbidden = _forbidden(grid.shape, blockers)
        safe_track = initial_track & ~full_forbidden
        assert _unique_ordered_path(safe_track, start, end) == ordered_main
        assert not _has_unique_path(initial_track, start, end)
        for ignored in range(len(blockers)):
            remaining = [p for i, p in enumerate(blockers) if i != ignored]
            assert not _has_unique_path(initial_track & ~_forbidden(grid.shape, remaining), start, end)

        forbidden4 = np.zeros(grid.shape, dtype=bool)
        for r, c in blockers:
            forbidden4[r, c] = True
            for dr, dc in N4:
                rr, cc = r + dr, c + dc
                if _in_bounds(grid.shape, rr, cc):
                    forbidden4[rr, cc] = True
        assert not _has_unique_path(initial_track & ~forbidden4, start, end)
        assert all(sum(initial_track[r + dr, c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)) == 1 for r, c in blockers)
        assert all(p not in set(ordered_main) and safe_track[p] for p in lure_positions)

        expected = grid.copy()
        for cells, color in zip(placeholder_cells, colors):
            for r, c in cells:
                expected[r, c] = color
        actual = main(grid)
        assert np.array_equal(actual, expected)
        changed = actual != grid
        assert int(changed.sum()) == bead_count * block_size * block_size
        return grid
    except (AssertionError, ValueError, IndexError):
        return None


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance satisfying all rule assumptions."""
    for _ in range(200):
        result = _generate_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a semantically valid safe-path bead task")
