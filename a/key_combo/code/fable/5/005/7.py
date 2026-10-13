from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
TRACK_COLOR = 5
BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 400


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(shape: tuple[int, int], cell: Coord):
    h, w = shape
    r, c = cell
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _trace_simple_path(mask: np.ndarray) -> list[Coord]:
    """Return the ordered coordinates of one induced, unbranched 4-path."""
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    assert len(cells) >= 7, "the visible track must contain a nontrivial path"

    degrees = {
        cell: sum(neighbor in cells for neighbor in _neighbors4(mask.shape, cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "the track must have exactly two endpoints"
    assert all(degrees[cell] in (1, 2) for cell in cells), "the track must not branch"

    ordered: list[Coord] = []
    previous: Coord | None = None
    current = endpoints[0]
    while True:
        ordered.append(current)
        choices = [
            neighbor
            for neighbor in _neighbors4(mask.shape, current)
            if neighbor in cells and neighbor != previous
        ]
        if not choices:
            break
        assert len(choices) == 1, "the track must have an unambiguous successor"
        previous, current = current, choices[0]

    assert len(ordered) == len(cells), "all track cells must belong to one path"
    return ordered


def _anchor_color_and_span(g: np.ndarray, bg: int, path: list[Coord]) -> tuple[int, int, int]:
    """Find the uniquely short two-cell colored block and its path interval."""
    candidates: list[tuple[int, list[Coord]]] = []
    for color in map(int, np.unique(g)):
        if color in (bg, TRACK_COLOR):
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        if len(coords) == 2 and coords[1] in set(_neighbors4(g.shape, coords[0])):
            candidates.append((color, coords))
    assert len(candidates) == 1, "exactly one colored two-cell anchor block is required"

    anchor_color, anchor_cells = candidates[0]
    position = {cell: i for i, cell in enumerate(path)}
    indices = sorted(position[cell] for cell in anchor_cells)
    assert indices[1] == indices[0] + 1, "anchor cells must be consecutive along the track"
    return anchor_color, indices[0], indices[1]


def _runs_in_path_order(
    g: np.ndarray,
    path: list[Coord],
    start: int,
    stop: int,
    anchor_color: int,
) -> list[tuple[int, int]]:
    """Read maximal train runs on path[start:stop] in increasing path order."""
    values = [int(g[path[i]]) for i in range(start, stop)]
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(values):
        color = values[i]
        if color == TRACK_COLOR:
            i += 1
            continue
        assert color != anchor_color, "anchor color may occur only in the anchor block"
        j = i + 1
        while j < len(values) and values[j] == color:
            j += 1
        assert j == len(values) or values[j] == TRACK_COLOR, (
            "different input trains must have at least one grey separator cell"
        )
        assert j - i >= 3, "every train must be longer than the two-cell anchor"
        runs.append((color, j - i))
        i = j
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress colored train segments toward a two-cell anchor along a winding grey track.

    Input structure:
    - The dominant background surrounds one non-closed, single-cell-wide 4-connected path.
    - Grey (5) path cells, colored train cells, and one uniquely short two-cell colored anchor
      together form that unbranched path; every train is a longer monochrome run.
    - Distinct train runs are separated by at least one visible grey path cell.

    Rule:
    - On each side of the anchor independently, read trains by along-track distance from
      near to far and slide them toward the anchor without crossing it.
    - Pack the runs end-to-end in that order while preserving every run's color and length.
    - Restore every vacated path cell to grey and leave the anchor and background fixed.

    Key insight:
    - Compression uses the one-dimensional coordinate obtained by tracing the curved path,
      not rows, columns, or planar distance; a train can therefore change shape at a bend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "ARC grids are at most 30x30"

    bg = _background(g)
    path = _trace_simple_path(g != bg)
    assert any(int(g[cell]) == TRACK_COLOR for cell in path), "grey track cells are required"
    anchor_color, anchor_lo, anchor_hi = _anchor_color_and_span(g, bg, path)

    left_runs = _runs_in_path_order(g, path, 0, anchor_lo, anchor_color)
    left_runs.reverse()  # near-to-far when walking away from the anchor
    right_runs = _runs_in_path_order(g, path, anchor_hi + 1, len(path), anchor_color)

    assert sum(length for _, length in left_runs) <= anchor_lo
    assert sum(length for _, length in right_runs) <= len(path) - anchor_hi - 1

    out = g.copy()
    for cell in path:
        out[cell] = TRACK_COLOR
    out[path[anchor_lo]] = anchor_color
    out[path[anchor_hi]] = anchor_color

    cursor = anchor_lo - 1
    for color, length in left_runs:
        for i in range(length):
            out[path[cursor - i]] = color
        cursor -= length

    cursor = anchor_hi + 1
    for color, length in right_runs:
        for i in range(length):
            out[path[cursor + i]] = color
        cursor += length

    return out


# ----------------------------- generator -----------------------------


def _is_induced_path(path: list[Coord], shape: tuple[int, int]) -> bool:
    if len(path) != len(set(path)):
        return False
    cells = set(path)
    if any(not (0 <= r < shape[0] and 0 <= c < shape[1]) for r, c in cells):
        return False
    for i, cell in enumerate(path):
        expected = 1 if i in (0, len(path) - 1) else 2
        actual = sum(neighbor in cells for neighbor in _neighbors4(shape, cell))
        if actual != expected:
            return False
    return True


def _serpentine_path() -> tuple[list[Coord], tuple[int, int]]:
    """Make a varied folded path whose parallel lanes remain visually separated."""
    lane_count = random.randint(3, 5)
    lane_gap = random.choice((3, 4))
    span = (lane_count - 1) * lane_gap
    h = random.randint(span + 3, min(25, span + 8))
    w = random.randint(14, 25)
    first_row = random.randint(1, h - span - 2)

    rows = [first_row + i * lane_gap for i in range(lane_count)]
    path: list[Coord] = []
    start_on_left = random.random() < 0.5
    current_col = random.randint(1, 3) if start_on_left else random.randint(w - 4, w - 2)

    for lane, row in enumerate(rows):
        if lane % 2 == 0:
            end_on_left = not start_on_left
        else:
            end_on_left = start_on_left
        end_col = random.randint(1, 3) if end_on_left else random.randint(w - 4, w - 2)
        if abs(end_col - current_col) < 8:
            raise ValueError("lane is too short")

        step = 1 if end_col > current_col else -1
        lane_cells = [(row, c) for c in range(current_col, end_col + step, step)]
        path.extend(lane_cells if not path else lane_cells[1:])
        current_col = end_col

        if lane + 1 < lane_count:
            next_row = rows[lane + 1]
            path.extend((r, current_col) for r in range(row + 1, next_row + 1))

    if not _is_induced_path(path, (h, w)):
        raise ValueError("failed to make a simple induced path")
    return path, (h, w)


def _step(a: Coord, b: Coord) -> Coord:
    return b[0] - a[0], b[1] - a[1]


def _turn_indices(path: list[Coord]) -> list[int]:
    return [i for i in range(1, len(path) - 1) if _step(path[i - 1], path[i]) != _step(path[i], path[i + 1])]


def _straight_anchor_candidates(path: list[Coord]) -> list[int]:
    turns = _turn_indices(path)
    candidates = []
    for a in range(24, len(path) - 26):
        direction = _step(path[a - 1], path[a])
        if not (
            _step(path[a], path[a + 1]) == direction
            and _step(path[a + 1], path[a + 2]) == direction
        ):
            continue
        if any(4 <= abs(a - turn) <= 9 for turn in turns):
            candidates.append(a)
    return candidates


def _place_side_runs(
    anchor_edge: int,
    direction: int,
    side_limit: int,
    lengths: list[int],
) -> list[tuple[int, int]] | None:
    """Return inclusive source intervals ordered from anchor-near to anchor-far."""
    gap = random.randint(4, 7)
    cursor = anchor_edge + direction * (gap + 1)
    intervals: list[tuple[int, int]] = []
    for run_index, length in enumerate(lengths):
        other = cursor + direction * (length - 1)
        lo, hi = sorted((cursor, other))
        if lo < 2 or hi > side_limit - 2:
            return None
        intervals.append((lo, hi))
        if run_index + 1 < len(lengths):
            separator = random.randint(2, 6)
            cursor = other + direction * (separator + 1)
    return intervals


def _normalized_shape(coords: list[Coord]) -> frozenset[Coord]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _distance_to_anchor(path: list[Coord], interval: tuple[int, int], anchor: tuple[int, int]) -> int:
    anchor_cells = [path[anchor[0]], path[anchor[1]]]
    return min(
        abs(path[i][0] - ar) + abs(path[i][1] - ac)
        for i in range(interval[0], interval[1] + 1)
        for ar, ac in anchor_cells
    )


def _render_case(path: list[Coord], shape: tuple[int, int]):
    anchor_candidates = _straight_anchor_candidates(path)
    if not anchor_candidates:
        return None

    for _ in range(120):
        anchor_lo = random.choice(anchor_candidates)
        anchor_hi = anchor_lo + 1
        counts = random.choice(((2, 1), (1, 2), (2, 2)))
        lengths_left = [random.randint(3, 5) for _ in range(counts[0])]
        lengths_right = [random.randint(3, 5) for _ in range(counts[1])]

        left = _place_side_runs(anchor_lo, -1, anchor_lo, lengths_left)
        right = _place_side_runs(anchor_hi, 1, len(path) - 1, lengths_right)
        if left is None or right is None:
            continue

        # A deliberately misleading planar cue: on at least one populated side,
        # the farther train along the rail is physically closer to the anchor.
        inversion = False
        for intervals in (left, right):
            if len(intervals) >= 2:
                near_d = _distance_to_anchor(path, intervals[0], (anchor_lo, anchor_hi))
                far_d = _distance_to_anchor(path, intervals[1], (anchor_lo, anchor_hi))
                inversion |= far_d < near_d
        if not inversion:
            continue

        colors = random.sample(PALETTE, 1 + len(left) + len(right))
        anchor_color = colors[0]
        left_colors = colors[1 : 1 + len(left)]
        right_colors = colors[1 + len(left) :]

        inp = np.full(shape, BACKGROUND, dtype=int)
        for cell in path:
            inp[cell] = TRACK_COLOR
        inp[path[anchor_lo]] = anchor_color
        inp[path[anchor_hi]] = anchor_color
        for color, (lo, hi) in zip(left_colors, left):
            for i in range(lo, hi + 1):
                inp[path[i]] = color
        for color, (lo, hi) in zip(right_colors, right):
            for i in range(lo, hi + 1):
                inp[path[i]] = color

        expected = np.full(shape, BACKGROUND, dtype=int)
        for cell in path:
            expected[cell] = TRACK_COLOR
        expected[path[anchor_lo]] = anchor_color
        expected[path[anchor_hi]] = anchor_color

        destinations: dict[int, tuple[int, int]] = {}
        cursor = anchor_lo - 1
        for color, length in zip(left_colors, lengths_left):
            destinations[color] = (cursor - length + 1, cursor)
            for i in range(cursor - length + 1, cursor + 1):
                expected[path[i]] = color
            cursor -= length
        cursor = anchor_hi + 1
        for color, length in zip(right_colors, lengths_right):
            destinations[color] = (cursor, cursor + length - 1)
            for i in range(cursor, cursor + length):
                expected[path[i]] = color
            cursor += length

        sources = {
            color: interval
            for color, interval in list(zip(left_colors, left)) + list(zip(right_colors, right))
        }
        metadata = {
            "anchor": (anchor_lo, anchor_hi),
            "anchor_color": anchor_color,
            "left_colors": left_colors,
            "right_colors": right_colors,
            "sources": sources,
            "destinations": destinations,
        }
        if _semantic_case_valid(inp, expected, path, metadata):
            return inp, expected
    return None


def _compressed_colors_from_anchor(
    grid: np.ndarray, path: list[Coord], indices: range
) -> list[int]:
    sequence: list[int] = []
    previous = TRACK_COLOR
    for i in indices:
        color = int(grid[path[i]])
        if color != TRACK_COLOR and color != previous:
            sequence.append(color)
        previous = color
    return sequence


def _semantic_case_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    path: list[Coord],
    metadata: dict,
) -> bool:
    """Check the stated semantics independently of calling the solver."""
    if inp.shape != expected.shape or max(inp.shape) > 30 or np.array_equal(inp, expected):
        return False
    if _background(inp) != BACKGROUND or _background(expected) != BACKGROUND:
        return False
    if not _is_induced_path(path, inp.shape):
        return False

    a, b = metadata["anchor"]
    anchor_color = metadata["anchor_color"]
    anchor_cells = (path[a], path[b])
    if any(int(inp[cell]) != anchor_color or int(expected[cell]) != anchor_color for cell in anchor_cells):
        return False

    train_colors = metadata["left_colors"] + metadata["right_colors"]
    if len(train_colors) < 3 or len(set(train_colors)) != len(train_colors):
        return False
    if any(int(np.sum(inp == color)) != int(np.sum(expected == color)) for color in train_colors):
        return False

    # The near-to-far order on each anchor side must survive compression.
    left_seen = _compressed_colors_from_anchor(expected, path, range(a - 1, -1, -1))
    right_seen = _compressed_colors_from_anchor(expected, path, range(b + 1, len(path)))
    if left_seen != metadata["left_colors"] or right_seen != metadata["right_colors"]:
        return False

    # All unoccupied rail cells are restored to grey, and nothing off the rail changes.
    path_set = set(path)
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            if (r, c) not in path_set and int(inp[r, c]) != int(expected[r, c]):
                return False
    occupied = {anchor_color, *train_colors}
    if any(int(expected[cell]) not in occupied | {TRACK_COLOR} for cell in path):
        return False

    # At least one train's planar shape changes because its packed interval crosses
    # a different bend. This rules out a mere rigid translation shortcut.
    shape_changed = False
    moved_on_both_sides = [False, False]
    for side_index, colors in enumerate((metadata["left_colors"], metadata["right_colors"])):
        for color in colors:
            src = metadata["sources"][color]
            dst = metadata["destinations"][color]
            src_coords = [path[i] for i in range(src[0], src[1] + 1)]
            dst_coords = [path[i] for i in range(dst[0], dst[1] + 1)]
            shape_changed |= _normalized_shape(src_coords) != _normalized_shape(dst_coords)
            moved_on_both_sides[side_index] |= src != dst
    return shape_changed and all(moved_on_both_sides)


def _transform(grid: np.ndarray, rotations: int, mirror: bool) -> np.ndarray:
    out = np.rot90(grid, rotations)
    if mirror:
        out = np.fliplr(out)
    return out.copy()


def _generation_attempt() -> np.ndarray | None:
    try:
        path, shape = _serpentine_path()
        rendered = _render_case(path, shape)
    except (AssertionError, ValueError):
        return None
    if rendered is None:
        return None
    inp, expected = rendered

    rotations = random.randrange(4)
    mirror = random.random() < 0.5
    inp = _transform(inp, rotations, mirror)
    expected = _transform(expected, rotations, mirror)

    try:
        actual = main(inp)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid winding-track compression input")
