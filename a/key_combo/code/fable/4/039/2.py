from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
Room = tuple[int, int, int, int]  # half-open (r0, c0, r1, c1)

_BLACK = 0
_PALETTE = tuple(range(1, 10))
_GEN_RETRIES = 400


def _infer_dividers(grid: np.ndarray) -> tuple[list[int], list[int]]:
    """Return paired horizontal rows and vertical columns in (r, c) coordinates."""
    height, width = grid.shape

    vertical = [
        c
        for c in range(1, width - 1)
        if int(grid[0, c]) != _BLACK
        and int(grid[0, c]) == int(grid[height - 1, c])
    ]
    horizontal = [
        r
        for r in range(1, height - 1)
        if int(grid[r, 0]) != _BLACK
        and int(grid[r, 0]) == int(grid[r, width - 1])
    ]
    return horizontal, vertical


def _rooms(
    shape: tuple[int, int], horizontal: list[int], vertical: list[int]
) -> list[Room]:
    """Enumerate strict-interior rooms bounded by the canvas edge and dividers."""
    height, width = shape
    row_bounds = [0, *sorted(horizontal), height - 1]
    col_bounds = [0, *sorted(vertical), width - 1]
    return [
        (row_bounds[i] + 1, col_bounds[j] + 1, row_bounds[i + 1], col_bounds[j + 1])
        for i in range(len(row_bounds) - 1)
        for j in range(len(col_bounds) - 1)
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """Infer invisible dividers from valid edge-marker pairs and fill exactly singly occupied rooms.

    Input structure:
    - A black canvas has colored single-pixel markers on its outer boundary and colored scattered pixels strictly inside it.
    - Equal-colored, exactly aligned markers on top/bottom or left/right opposite edges are valid pairs; all other boundary markers are noise.
    - Each valid top/bottom pair denotes a full vertical divider, and each valid left/right pair denotes a full horizontal divider.

    Rule:
    - Use only valid opposite-edge pairs to partition the strict interior of the boundary into rectangular rooms.
    - Keep every inferred divider and the complete outer clue boundary black in the output.
    - Fill a room with its scattered pixel's color exactly when that room contains one pixel; leave rooms with zero or multiple pixels black and discard all input markers and points.

    Key insight:
    - The walls are absent from the input: exact color, opposition, and alignment jointly identify their endpoints, after which occupancy must be counted independently inside each inferred room.
    """
    g = np.asarray(grid)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = g.shape
    assert height >= 5 and width >= 5, "expected an outer boundary and interior rooms"
    assert np.all(g[[0, 0, height - 1, height - 1], [0, width - 1, 0, width - 1]] == _BLACK), (
        "corner cells must be black so every edge marker has one unambiguous edge role"
    )

    horizontal, vertical = _infer_dividers(g)
    assert horizontal and vertical, "expected at least one divider in each direction"

    # A valid puzzle has no scattered point on a divider: the only colored cells
    # on a divider are its two boundary generators.
    for r in horizontal:
        assert np.all(g[r, 1 : width - 1] == _BLACK), "a scattered point lies on a divider"
    for c in vertical:
        assert np.all(g[1 : height - 1, c] == _BLACK), "a scattered point lies on a divider"

    out = np.zeros_like(g)
    for r0, c0, r1, c1 in _rooms(g.shape, horizontal, vertical):
        patch = g[r0:r1, c0:c1]
        points = np.argwhere(patch != _BLACK)
        if len(points) == 1:
            rr, cc = map(int, points[0])
            out[r0:r1, c0:c1] = patch[rr, cc]
    return out


# ---------------------------------------------------------------------------
# Generator and an independent semantic oracle.
# ---------------------------------------------------------------------------


def _segments_and_lines(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    segments: list[tuple[int, int]] = []
    lines: list[int] = []
    cursor = 1
    for i, size in enumerate(sizes):
        segments.append((cursor, cursor + size))
        cursor += size
        if i + 1 < len(sizes):
            lines.append(cursor)
            cursor += 1
    return segments, lines, cursor + 1


def _render_expected(
    shape: tuple[int, int],
    horizontal: list[int],
    vertical: list[int],
    points: list[tuple[int, int, int]],
) -> np.ndarray:
    """Render the stated rule without reading or calling main()."""
    out = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in _rooms(shape, horizontal, vertical):
        occupants = [
            (r, c, color)
            for r, c, color in points
            if r0 <= r < r1 and c0 <= c < c1
        ]
        if len(occupants) == 1:
            out[r0:r1, c0:c1] = occupants[0][2]
    return out


def _singleton_cover(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    """Choose singleton rooms covering every row band and every column band."""
    if n_rows >= n_cols:
        cols = list(range(n_cols))
        random.shuffle(cols)
        return [(r, cols[r % n_cols]) for r in range(n_rows)]
    rows = list(range(n_rows))
    random.shuffle(rows)
    return [(rows[c % n_rows], c) for c in range(n_cols)]


def _choose_spaced_cells(
    row_segment: tuple[int, int], col_segment: tuple[int, int], count: int
) -> list[Coord] | None:
    """Put points away from walls and keep their 8-neighborhoods disjoint."""
    if count == 0:
        return []
    r0, r1 = row_segment
    c0, c1 = col_segment
    candidates = [
        (r, c)
        for r in range(r0 + 1, r1 - 1)
        for c in range(c0 + 1, c1 - 1)
    ]
    random.shuffle(candidates)

    chosen: list[Coord] = []
    for cell in candidates:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 2 for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _noise_positions(active: list[int]) -> tuple[int, int, int, int] | None:
    """Pick positions a,b,c,d; a/b/d share one edge and remain visually separate."""
    values = list(dict.fromkeys(active))
    random.shuffle(values)
    triples = [
        (a, b, d)
        for a in values
        for b in values
        for d in values
        if len({a, b, d}) == 3
        and abs(a - b) >= 2
        and abs(a - d) >= 2
        and abs(b - d) >= 2
    ]
    random.shuffle(triples)
    for a, b, d in triples:
        options = [c for c in values if c not in (a, b) and abs(c - a) >= 2]
        if options:
            return a, b, random.choice(options), d
    return None


def _place_boundary_markers(
    grid: np.ndarray,
    horizontal: list[int],
    vertical: list[int],
    expected: np.ndarray,
) -> dict | None:
    """Add legal generators plus explicit counterexamples to every pairing shortcut."""
    height, width = grid.shape
    legal_colors = random.sample(_PALETTE, 2)

    for c in vertical:
        color = random.choice(legal_colors)
        grid[0, c] = grid[height - 1, c] = color
    for r in horizontal:
        color = random.choice(legal_colors)
        grid[r, 0] = grid[r, width - 1] = color

    # Noise coordinates cross colored output rooms. Thus mistakenly treating any
    # noise relation as a divider necessarily changes the answer.
    active_cols = [
        c for c in range(1, width - 1) if c not in vertical and bool(np.any(expected[1:-1, c] != _BLACK))
    ]
    active_rows = [
        r for r in range(1, height - 1) if r not in horizontal and bool(np.any(expected[r, 1:-1] != _BLACK))
    ]
    col_pos = _noise_positions(active_cols)
    row_pos = _noise_positions(active_rows)
    if col_pos is None or row_pos is None:
        return None

    ca, cb, cc, cd = col_pos
    ra, rb, rc, rd = row_pos
    noise = random.sample([color for color in _PALETTE if color not in legal_colors], 7)

    # Aligned but differently colored opposite-edge markers.
    grid[0, ca], grid[height - 1, ca] = noise[0], noise[1]
    grid[ra, 0], grid[ra, width - 1] = noise[3], noise[4]
    # Equal-colored opposite-edge markers that are not aligned.
    grid[0, cb] = grid[height - 1, cc] = noise[2]
    grid[rb, 0] = grid[rc, width - 1] = noise[5]
    # Equal-colored markers on adjacent rather than opposite edges.
    grid[0, cd] = grid[rd, 0] = noise[6]

    return {
        "aligned_extra": ([ra], [ca]),
        "misaligned_extra": ([rb, rc], [cb, cc]),
        "adjacent_extra": ([rd], [cd]),
    }


def _valid_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    horizontal: list[int],
    vertical: list[int],
    points: list[tuple[int, int, int]],
    kinds: list[int],
    noise_meta: dict,
) -> bool:
    """Check the described semantics, uniqueness, dependencies, and anti-shortcuts."""
    if max(grid.shape) > 30 or grid.ndim != 2:
        return False
    if _infer_dividers(grid) != (horizontal, vertical):
        return False
    if not ({0, 1, 2} <= set(kinds)) or kinds.count(1) < 2:
        return False
    if len({color for color in np.unique(expected) if int(color) != _BLACK}) < 2:
        return False
    if np.any(expected[0, :]) or np.any(expected[-1, :]) or np.any(expected[:, 0]) or np.any(expected[:, -1]):
        return False
    if any(np.any(expected[r, :]) for r in horizontal):
        return False
    if any(np.any(expected[:, c]) for c in vertical):
        return False

    interior_seen = sorted(
        (int(r), int(c), int(grid[r, c]))
        for r in range(1, grid.shape[0] - 1)
        for c in range(1, grid.shape[1] - 1)
        if int(grid[r, c]) != _BLACK
    )
    if interior_seen != sorted(points):
        return False

    try:
        if not np.array_equal(main(grid), expected):
            return False
    except (AssertionError, ValueError, IndexError):
        return False

    # Every inferred line must matter. Removing any one changes room membership
    # or exposes a formerly black wall cell in the independently rendered result.
    for line in horizontal:
        if np.array_equal(_render_expected(grid.shape, [r for r in horizontal if r != line], vertical, points), expected):
            return False
    for line in vertical:
        if np.array_equal(_render_expected(grid.shape, horizontal, [c for c in vertical if c != line], points), expected):
            return False

    # The three noise constructions each defeat a distinct relaxed pairing rule.
    for extra_rows, extra_cols in noise_meta.values():
        for row in extra_rows:
            if np.array_equal(
                _render_expected(grid.shape, sorted(horizontal + [row]), vertical, points),
                expected,
            ):
                return False
        for col in extra_cols:
            if np.array_equal(
                _render_expected(grid.shape, horizontal, sorted(vertical + [col]), points),
                expected,
            ):
                return False
        wrong_rows = sorted(set(horizontal) | set(extra_rows))
        wrong_cols = sorted(set(vertical) | set(extra_cols))
        if np.array_equal(_render_expected(grid.shape, wrong_rows, wrong_cols, points), expected):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_row_bands = random.randint(2, 4)
    n_col_bands = random.randint(2, 4)
    row_sizes = [random.randint(5, 6) for _ in range(n_row_bands)]
    col_sizes = [random.randint(5, 6) for _ in range(n_col_bands)]
    row_segments, horizontal, height = _segments_and_lines(row_sizes)
    col_segments, vertical, width = _segments_and_lines(col_sizes)
    if height > 30 or width > 30:
        return None

    singleton_rooms = _singleton_cover(n_row_bands, n_col_bands)
    remaining = [
        (ri, ci)
        for ri in range(n_row_bands)
        for ci in range(n_col_bands)
        if (ri, ci) not in singleton_rooms
    ]
    if len(remaining) < 2:
        return None
    random.shuffle(remaining)
    empty_room, multi_room = remaining[:2]

    room_kinds: dict[tuple[int, int], int] = {
        room: 1 for room in singleton_rooms
    }
    room_kinds[empty_room] = 0
    room_kinds[multi_room] = 2
    for room in remaining[2:]:
        room_kinds[room] = random.choices((0, 1, 2), weights=(3, 4, 2), k=1)[0]

    # Give two mandatory singleton rooms visibly different colors.
    first_color, second_color = random.sample(_PALETTE, 2)
    singleton_color = {
        singleton_rooms[0]: first_color,
        singleton_rooms[1]: second_color,
    }

    points: list[tuple[int, int, int]] = []
    kinds: list[int] = []
    for ri in range(n_row_bands):
        for ci in range(n_col_bands):
            room = (ri, ci)
            kind = room_kinds[room]
            kinds.append(kind)
            cells = _choose_spaced_cells(row_segments[ri], col_segments[ci], kind)
            if cells is None:
                return None
            if kind == 1:
                colors = [singleton_color.get(room, random.choice(_PALETTE))]
            elif kind == 2:
                colors = random.sample(_PALETTE, 2)
            else:
                colors = []
            points.extend((r, c, color) for (r, c), color in zip(cells, colors))

    expected = _render_expected((height, width), horizontal, vertical, points)
    grid = np.zeros((height, width), dtype=int)
    for r, c, color in points:
        grid[r, c] = color

    noise_meta = _place_boundary_markers(grid, horizontal, vertical, expected)
    if noise_meta is None:
        return None
    if not _valid_semantics(grid, expected, horizontal, vertical, points, kinds, noise_meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, validated instance with all occupancy and noise branches."""
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-divider instance")
