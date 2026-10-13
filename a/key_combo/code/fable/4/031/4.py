"""ARC task 4: recolor only the blockers bypassed by invisible rising paths."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


BLACK = 0
GRAY = 5
START_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor only gray blockers successfully bypassed by invisible upward paths.

    Input structure:
    - Black is the background, gray pixels are blockers, and the bottom row
      contains at least two one-pixel starts of pairwise distinct non-gray colors.
    - Every path-relevant rightward sidestep stays inside the grid, except for a
      possible final blocker at the right edge; no blocker is bypassed by two paths.

    Rule:
    - From each bottom-row start, repeatedly move up when the cell above is not
      gray.  When it is gray, move one cell right without changing row; if that
      right step would leave the grid, stop.  Continue until the top or a stop.
    - Erase every start and draw none of the traversed path cells.  Recolor each
      gray blocker only when the corresponding right sidestep actually succeeds,
      using that path's start color; leave every other gray blocker unchanged.

    Key insight:
    - A successful detour shifts the path's column, so blocker ownership requires
      tracing the whole invisible route rather than matching blockers to the
      original start columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    height, width = g.shape
    assert height >= 2 and width >= 2, "grid is too small for an upward path"

    start_columns = np.flatnonzero((g[-1] != BLACK) & (g[-1] != GRAY))
    assert len(start_columns) >= 1, "expected colored starts on the bottom row"
    start_colors = [int(g[-1, c]) for c in start_columns]
    assert len(set(start_colors)) == len(start_colors), "start colors must be distinct"
    assert np.all((g[:-1] == BLACK) | (g[:-1] == GRAY)), (
        "only black background and gray blockers may occur above the starts"
    )

    out = g.copy()
    out[-1, start_columns] = BLACK
    owners: dict[tuple[int, int], int] = {}

    for start_c, color in zip(start_columns, start_colors):
        r, c = height - 1, int(start_c)
        while r > 0:
            blocker = (r - 1, c)
            if int(g[blocker]) == GRAY:
                if c + 1 >= width:
                    break
                previous = owners.get(blocker)
                assert previous is None or previous == color, (
                    "a blocker may be bypassed by at most one path"
                )
                owners[blocker] = color
                out[blocker] = color
                c += 1
            else:
                r -= 1

    return out


def _oracle_trace(grid: np.ndarray) -> dict:
    """Independently record path events for generator-side semantic checks."""
    height, width = grid.shape
    starts = [int(c) for c in np.flatnonzero((grid[-1] != BLACK) & (grid[-1] != GRAY))]
    events: list[tuple[tuple[int, int], int, int]] = []
    failed: list[tuple[tuple[int, int], int, int]] = []
    paths: dict[int, set[tuple[int, int]]] = {}
    checked: set[tuple[int, int]] = set()
    endings: dict[int, str] = {}

    for start_c in starts:
        color = int(grid[-1, start_c])
        position = (height - 1, start_c)
        visited = {position}
        while position[0] != 0:
            r, c = position
            above = (r - 1, c)
            checked.add(above)
            if int(grid[above]) == GRAY:
                if c == width - 1:
                    failed.append((above, color, start_c))
                    endings[start_c] = "right_edge"
                    break
                events.append((above, color, start_c))
                position = (r, c + 1)
            else:
                position = above
            visited.add(position)
        else:
            endings[start_c] = "top"
        paths[start_c] = visited

    return {
        "events": events,
        "failed": failed,
        "paths": paths,
        "checked": checked,
        "endings": endings,
    }


def _expected_from_events(
    grid: np.ndarray, intended: dict[tuple[int, int], int]
) -> np.ndarray:
    expected = grid.copy()
    start_mask = (expected[-1] != BLACK) & (expected[-1] != GRAY)
    expected[-1, start_mask] = BLACK
    for (r, c), color in intended.items():
        expected[r, c] = color
    return expected


def _same_column_shortcut(grid: np.ndarray) -> np.ndarray:
    """A deliberately wrong solution used as an anti-degeneration check."""
    out = grid.copy()
    starts = np.flatnonzero((grid[-1] != BLACK) & (grid[-1] != GRAY))
    for c in starts:
        color = int(grid[-1, c])
        out[:-1, c][grid[:-1, c] == GRAY] = color
        out[-1, c] = BLACK
    return out


def _validate_generated(
    grid: np.ndarray,
    intended: dict[tuple[int, int], int],
    normal_starts: tuple[int, ...],
    edge_start: int,
    terminal_blocker: tuple[int, int],
) -> bool:
    """Check puzzle semantics, unique attribution, and shortcut counterexamples."""
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    starts = [int(c) for c in np.flatnonzero((grid[-1] != BLACK) & (grid[-1] != GRAY))]
    colors = [int(grid[-1, c]) for c in starts]
    if len(starts) < 2 or len(colors) != len(set(colors)):
        return False
    if not set(np.unique(grid)).issubset({BLACK, GRAY, *colors}):
        return False

    trace = _oracle_trace(grid)
    attributed: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    per_start = defaultdict(list)
    for coord, color, start_c in trace["events"]:
        attributed[coord].append((color, start_c))
        per_start[start_c].append(coord)

    # Every bypassed blocker has one owner, and the independently traced owners
    # exactly match the construction metadata.
    if any(len(owners) != 1 for owners in attributed.values()):
        return False
    traced = {coord: owners[0][0] for coord, owners in attributed.items()}
    if traced != intended:
        return False
    if any(not per_start[c] for c in starts):
        return False

    # Both stopping branches occur in every generated example.
    if any(trace["endings"].get(c) != "top" for c in normal_starts):
        return False
    if trace["endings"].get(edge_start) != "right_edge":
        return False
    failed_coords = {coord for coord, _color, _start in trace["failed"]}
    if failed_coords != {terminal_blocker} or terminal_blocker in traced:
        return False

    # Full tracing is indispensable: normal paths have multiple successful
    # shifts, including a blocker outside their original columns.
    if not all(len(per_start[c]) >= 2 for c in normal_starts):
        return False
    if not any(coord[1] != c for c in normal_starts for coord in per_start[c]):
        return False

    expected = _expected_from_events(grid, intended)
    try:
        actual = main(grid)
    except (AssertionError, IndexError, ValueError):
        return False
    if not np.array_equal(actual, expected):
        return False

    # Paths and starts must truly disappear, while several decoy/terminal gray
    # obstacles remain.  These also refute two tempting shortcut rules.
    for start_c, cells in trace["paths"].items():
        for r, c in cells:
            if (r, c) == (grid.shape[0] - 1, start_c):
                if int(expected[r, c]) != BLACK:
                    return False
            elif int(grid[r, c]) == BLACK and int(expected[r, c]) != BLACK:
                return False
    if int(np.count_nonzero(expected == GRAY)) < len(starts):
        return False
    if np.array_equal(_same_column_shortcut(grid), expected):
        return False
    recolor_all = grid.copy()
    recolor_all[-1, starts] = BLACK
    recolor_all[recolor_all == GRAY] = colors[0]
    if np.array_equal(recolor_all, expected):
        return False
    return not np.array_equal(grid, expected)


def _build_candidate(rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    """Construct separated path lanes plus decoys and one right-edge stop."""
    height = int(rng.integers(12, 20))
    count = int(rng.integers(2, 5))
    base = int(rng.integers(1, 3))
    last_normal = base + (count - 2) * 5
    min_width = max(12, last_normal + 8)
    width = int(rng.integers(min_width, min(30, min_width + 6)))

    normal_starts = tuple(base + i * 5 for i in range(count - 1))
    edge_start = width - 2
    starts = (*normal_starts, edge_start)
    colors = [int(v) for v in rng.choice(START_COLORS, size=count, replace=False)]

    grid = np.zeros((height, width), dtype=int)
    for c, color in zip(starts, colors):
        grid[-1, c] = color

    intended: dict[tuple[int, int], int] = {}
    obstacle_cells: set[tuple[int, int]] = set()

    # Each ordinary lane has two or three successful detours.  A gray decoy is
    # then placed higher in its original column, where column matching would
    # incorrectly recolor it after the real path has shifted away.
    available_rows = np.arange(3, height - 2)
    for start_c, color in zip(normal_starts, colors[:-1]):
        detour_count = int(rng.integers(2, 4))
        rows = sorted(
            (int(v) for v in rng.choice(available_rows, size=detour_count, replace=False)),
            reverse=True,
        )
        for shift, row in enumerate(rows):
            coord = (row, start_c + shift)
            grid[coord] = GRAY
            obstacle_cells.add(coord)
            intended[coord] = color

        decoy_choices = [r for r in range(1, rows[0]) if r not in rows]
        if not decoy_choices:
            return None
        decoy = (int(rng.choice(decoy_choices)), start_c)
        grid[decoy] = GRAY
        obstacle_cells.add(decoy)

    # The final lane demonstrates both outcomes: one successful detour reaches
    # the last column, then a second gray blocker cannot be bypassed and remains.
    success_row = int(rng.integers(6, height - 2))
    terminal_row = int(rng.integers(1, success_row - 1))
    edge_success = (success_row, edge_start)
    terminal_blocker = (terminal_row, width - 1)
    intended[edge_success] = colors[-1]
    grid[edge_success] = GRAY
    grid[terminal_blocker] = GRAY
    obstacle_cells.update((edge_success, terminal_blocker))

    edge_decoy_rows = [
        r for r in range(1, success_row)
        if r != terminal_row and (r, edge_start) not in obstacle_cells
    ]
    if not edge_decoy_rows:
        return None
    edge_decoy = (int(rng.choice(edge_decoy_rows)), edge_start)
    grid[edge_decoy] = GRAY
    obstacle_cells.add(edge_decoy)

    # Add scattered, provably irrelevant obstacles outside every traversed or
    # examined cell.  They cannot alter the paths and make "recolor all" fail.
    preliminary = _oracle_trace(grid)
    path_cells = set().union(*preliminary["paths"].values())
    forbidden = path_cells | preliminary["checked"] | obstacle_cells
    candidates = []
    for r in range(1, height - 1):
        for c in range(width):
            if (r, c) in forbidden:
                continue
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in obstacle_cells):
                continue
            candidates.append((r, c))
    rng.shuffle(candidates)
    for coord in candidates[: int(rng.integers(1, min(5, len(candidates) + 1)))]:
        grid[coord] = GRAY
        obstacle_cells.add(coord)

    meta = {
        "intended": intended,
        "normal_starts": normal_starts,
        "edge_start": edge_start,
        "terminal_blocker": terminal_blocker,
    }
    return grid, meta


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked instance with explicit anti-shortcuts."""
    rng = np.random.default_rng()
    for _ in range(400):
        built = _build_candidate(rng)
        if built is None:
            continue
        grid, meta = built
        if not _validate_generated(grid, **meta):
            continue
        out = main(grid)
        signature = (tuple(map(int, out.shape)), out.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        return grid
    raise RuntimeError("generate_input: failed to construct a valid diverse example")
