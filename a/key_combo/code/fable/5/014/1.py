from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
Direction = tuple[int, int]

_DIRS: tuple[Direction, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SHAPES: tuple[tuple[Coord, ...], ...] = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, -1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, -1)),
)


def _background(grid: np.ndarray) -> int:
    """Return the uniquely most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    ordered = counts.most_common()
    assert len(ordered) >= 2 and ordered[0][1] > ordered[1][1], (
        "background must be uniquely most frequent"
    )
    return int(ordered[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    height, width = mask.shape
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    result: list[np.ndarray] = []
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        cells = [start]
        while queue:
            r, c = queue.popleft()
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    queue.append((nr, nc))
                    cells.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _three_cell_bar_center(coords: np.ndarray) -> tuple[Coord, Direction] | None:
    """Recognize a length-three horizontal/vertical bar and return center + axis."""
    if coords.shape != (3, 2):
        return None
    rows = sorted(map(int, coords[:, 0]))
    cols = sorted(map(int, coords[:, 1]))
    if len(set(rows)) == 1 and cols == list(range(cols[0], cols[0] + 3)):
        return (rows[0], cols[1]), (0, 1)
    if len(set(cols)) == 1 and rows == list(range(rows[0], rows[0] + 3)):
        return (rows[1], cols[0]), (1, 0)
    return None


def _find_launcher(grid: np.ndarray, bg: int) -> tuple[int, int, Coord, Direction]:
    """Identify the marker/control pair solely from its visible geometry."""
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    candidates: list[tuple[int, int, Coord, Direction]] = []

    for marker_color in colors:
        if counts[marker_color] != 3:
            continue
        recognized = _three_cell_bar_center(np.argwhere(grid == marker_color))
        if recognized is None:
            continue
        center, bar_axis = recognized
        cr, cc = center
        for control_color in colors:
            if control_color == marker_color or counts[control_color] != 1:
                continue
            rr, rc = map(int, np.argwhere(grid == control_color)[0])
            dr, dc = cr - rr, cc - rc
            if abs(dr) + abs(dc) != 1:
                continue
            if dr * bar_axis[0] + dc * bar_axis[1] != 0:
                continue
            candidates.append((marker_color, control_color, center, (dr, dc)))

    assert len(candidates) == 1, "expected one unambiguous marker/control launcher"
    return candidates[0]


def _ray_coordinates(shape: tuple[int, int], center: Coord, direction: Direction) -> list[Coord]:
    height, width = shape
    cr, cc = center
    dr, dc = direction
    cells: list[Coord] = []
    r, c = cr + dr, cc + dc
    while 0 <= r < height and 0 <= c < width:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _is_barrier_color(
    grid: np.ndarray,
    color: int,
    center: Coord,
    direction: Direction,
) -> bool:
    """Check that every component is a perpendicular segment crossing the forward ray."""
    components = _components(grid == color)
    if len(components) < 2:
        return False
    cr, cc = center
    dr, dc = direction
    pr, pc = -dc, dr
    for coords in components:
        if coords.shape[0] < 3:
            return False
        ts = [int((r - cr) * dr + (c - cc) * dc) for r, c in coords]
        qs = sorted(int((r - cr) * pr + (c - cc) * pc) for r, c in coords)
        if len(set(ts)) != 1 or ts[0] <= 0:
            return False
        if qs != list(range(qs[0], qs[-1] + 1)) or 0 not in qs:
            return False
    return True


def _analyze(grid: np.ndarray):
    """Resolve every role from frequency plus visible launcher/barrier structure."""
    bg = _background(grid)
    marker, control, center, direction = _find_launcher(grid, bg)
    remaining = [
        int(v)
        for v in np.unique(grid)
        if int(v) not in {bg, marker, control}
    ]
    barrier_candidates = [
        color
        for color in remaining
        if _is_barrier_color(grid, color, center, direction)
    ]
    assert len(barrier_candidates) == 1, "expected one structurally unique barrier color"
    barrier = barrier_candidates[0]
    target_candidates = [color for color in remaining if color != barrier]
    assert len(target_candidates) == 1, "expected one target-object color"
    return bg, marker, control, barrier, target_candidates[0], center, direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase whole target objects hit while a launcher ray is enabled, toggling at every crossed barrier segment.

    Input structure:
    - The uniquely most frequent color is background. A unique three-cell straight marker bar has a differently colored singleton control pixel adjacent to its center.
    - Several components of one color are straight segments perpendicular to, and crossing, the forward ray; another color forms separated 4-connected target objects on and off that ray.

    Rule:
    - Fire from the marker center in the cardinal direction away from the control pixel, initially enabled, and toggle enabled/disabled once upon entering each continuous barrier segment.
    - If the ray touches a target component while enabled, recolor every pixel of that component as background. Preserve targets touched only while disabled, off-ray targets, the launcher, and all barriers; the ray leaves no trace.

    Key insight:
    - Barriers are parity switches rather than stopping walls: a hit is effective exactly when an even number of barrier segments precedes it along the launcher-defined ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"

    bg, _marker, _control, barrier, target, center, direction = _analyze(g)
    active_cells: set[Coord] = set()
    enabled = True
    inside_barrier = False
    for r, c in _ray_coordinates(g.shape, center, direction):
        is_barrier = int(g[r, c]) == barrier
        if is_barrier and not inside_barrier:
            enabled = not enabled
        inside_barrier = is_barrier
        if not is_barrier and enabled and int(g[r, c]) == target:
            active_cells.add((r, c))

    out = g.copy()
    for coords in _components(g == target):
        if any((int(r), int(c)) in active_cells for r, c in coords):
            out[coords[:, 0], coords[:, 1]] = bg
    return out


# Generator configuration. Colors are deliberately sampled without assigning any
# numerical meaning; all roles remain visibly encoded by geometry.
_PALETTE = tuple(range(10))
_MIN_SIZE, _MAX_SIZE = 25, 30
_MAX_ATTEMPTS = 200


def _cell(center: Coord, direction: Direction, t: int, q: int) -> Coord:
    """Map launcher-relative (forward t, perpendicular q) to absolute (r, c)."""
    cr, cc = center
    dr, dc = direction
    pr, pc = -dc, dr
    return cr + t * dr + q * pr, cc + t * dc + q * pc


def _shape_cells(
    center: Coord,
    direction: Direction,
    anchor_t: int,
    anchor_q: int,
    shape: tuple[Coord, ...],
    flip_q: int,
) -> set[Coord]:
    return {
        _cell(center, direction, anchor_t + dt, anchor_q + flip_q * dq)
        for dt, dq in shape
    }


def _clear_with_padding(grid: np.ndarray, cells: set[Coord], bg: int) -> bool:
    """Require in-bounds empty cells and a one-cell visual gap around the object."""
    height, width = grid.shape
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width) or int(grid[r, c]) != bg:
            return False
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells or not (0 <= nr < height and 0 <= nc < width):
                    continue
                if int(grid[nr, nc]) != bg:
                    return False
    return True


def _paint(grid: np.ndarray, cells: set[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _place_target(
    grid: np.ndarray,
    bg: int,
    target: int,
    center: Coord,
    direction: Direction,
    anchor_t: int,
    anchor_q: int = 0,
    attempts: int = 30,
) -> set[Coord] | None:
    for _ in range(attempts):
        shape = random.choice(_SHAPES)
        flip_q = random.choice((-1, 1))
        cells = _shape_cells(center, direction, anchor_t, anchor_q, shape, flip_q)
        if _clear_with_padding(grid, cells, bg):
            _paint(grid, cells, target)
            return cells
    return None


def _launcher_center(
    shape: tuple[int, int], direction: Direction
) -> tuple[Coord, int]:
    """Choose a center with room behind it and at least 19 ray cells ahead."""
    height, width = shape
    dr, dc = direction
    axis_size = height if dr else width
    max_margin = min(7, axis_size - 20)
    margin = random.randint(5, max_margin)
    if dr > 0:
        r = margin
    elif dr < 0:
        r = height - 1 - margin
    else:
        r = random.randint(5, height - 6)
    if dc > 0:
        c = margin
    elif dc < 0:
        c = width - 1 - margin
    else:
        c = random.randint(5, width - 6)
    center = (r, c)
    forward = (height - 1 - r if dr > 0 else r) if dr else (
        width - 1 - c if dc > 0 else c
    )
    assert forward >= 19
    return center, forward


def _barrier_positions(ray_length: int) -> list[int] | None:
    """Sample 3-4 separated switches, leaving room for clear target objects."""
    by_count: dict[int, list[tuple[int, ...]]] = {}
    for count in (3, 4):
        choices = [
            combo
            for combo in itertools.combinations(range(5, ray_length - 2), count)
            if all(b - a >= 5 for a, b in zip(combo, combo[1:]))
        ]
        if choices:
            by_count[count] = choices
    if not by_count:
        return None
    count = random.choice(list(by_count))
    return list(random.choice(by_count[count]))


def _expected_from_records(
    grid: np.ndarray,
    bg: int,
    records: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for record in records:
        if bool(record["on_forward_ray"]) and bool(record["enabled"]):
            for r, c in record["cells"]:
                out[r, c] = bg
    return out


def _validate_candidate(
    grid: np.ndarray,
    expected: np.ndarray,
    bg: int,
    marker: int,
    control: int,
    barrier: int,
    target: int,
    barrier_ts: list[int],
    records: list[dict],
) -> bool:
    """Check scene semantics and explicit counterexamples to common shortcuts."""
    try:
        analysis = _analyze(grid)
        if analysis[:5] != (bg, marker, control, barrier, target):
            return False
        if not np.array_equal(main(grid), expected) or np.array_equal(grid, expected):
            return False
    except (AssertionError, ValueError, IndexError):
        return False

    ray_records = sorted(
        (record for record in records if bool(record["on_forward_ray"])),
        key=lambda record: int(record["t"]),
    )
    if not ray_records or not any(not bool(record["enabled"]) for record in ray_records):
        return False
    if not any(
        bool(record["enabled"]) and sum(t < int(record["t"]) for t in barrier_ts) >= 2
        for record in ray_records
    ):
        return False

    # Wrong shortcut 1: erase every target touched by the ray, ignoring switches.
    erase_all = grid.copy()
    for record in ray_records:
        for r, c in record["cells"]:
            erase_all[r, c] = bg
    if np.array_equal(erase_all, expected):
        return False

    # Wrong shortcut 2: alternate target objects rather than counting barriers.
    alternate = grid.copy()
    for index, record in enumerate(ray_records):
        if index % 2 == 0:
            for r, c in record["cells"]:
                alternate[r, c] = bg
    if np.array_equal(alternate, expected):
        return False

    # Wrong shortcut 3: treat the first barrier as a stopping wall.
    stop_at_first = grid.copy()
    first_barrier = barrier_ts[0]
    for record in ray_records:
        if int(record["t"]) < first_barrier:
            for r, c in record["cells"]:
                stop_at_first[r, c] = bg
    if np.array_equal(stop_at_first, expected):
        return False

    # Every generated barrier is consequential: removing any one toggle changes
    # the prescribed status of at least one later ray target.
    for removed in barrier_ts:
        consequential = False
        for record in ray_records:
            t = int(record["t"])
            original = sum(b < t for b in barrier_ts) % 2 == 0
            without = sum(b < t for b in barrier_ts if b != removed) % 2 == 0
            consequential |= original != without
        if not consequential:
            return False

    # The reverse-line and off-axis distractors must remain intact.
    outside = [record for record in records if not bool(record["on_forward_ray"])]
    if len(outside) < 2:
        return False
    for record in outside:
        if any(int(expected[r, c]) != target for r, c in record["cells"]):
            return False

    # No target objects may merge, and targets cannot masquerade as barriers.
    if len(_components(grid == target)) != len(records):
        return False
    if _is_barrier_color(grid, target, analysis[5], analysis[6]):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(_MIN_SIZE, _MAX_SIZE)
    width = random.randint(_MIN_SIZE, _MAX_SIZE)
    direction = random.choice(_DIRS)
    center, ray_length = _launcher_center((height, width), direction)
    barrier_ts = _barrier_positions(ray_length)
    if barrier_ts is None:
        return None

    bg, marker, control, barrier, target = random.sample(_PALETTE, 5)
    grid = np.full((height, width), bg, dtype=int)
    dr, dc = direction
    pr, pc = -dc, dr
    cr, cc = center

    marker_cells = {(cr, cc), (cr - pr, cc - pc), (cr + pr, cc + pc)}
    _paint(grid, marker_cells, marker)
    grid[cr - dr, cc - dc] = control

    for t in barrier_ts:
        low = random.randint(1, 3)
        high = random.randint(1, 3)
        cells = {_cell(center, direction, t, q) for q in range(-low, high + 1)}
        if not _clear_with_padding(grid, cells, bg):
            return None
        _paint(grid, cells, barrier)

    # One odd interval is deliberately empty. This makes two consecutive ray
    # targets share a result and defeats object-by-object alternation shortcuts.
    if len(barrier_ts) == 3:
        empty_interval = 1
    else:
        empty_interval = random.choice((1, 3))

    records: list[dict] = []
    boundaries = [0] + barrier_ts + [ray_length + 1]
    for interval in range(len(boundaries) - 1):
        if interval == empty_interval:
            continue
        previous = boundaries[interval]
        following = boundaries[interval + 1]
        anchor_min = 2 if interval == 0 else previous + 2
        anchor_max = ray_length - 1 if interval == len(barrier_ts) else following - 3
        if anchor_min > anchor_max:
            return None
        anchor_t = random.randint(anchor_min, anchor_max)
        cells = _place_target(grid, bg, target, center, direction, anchor_t)
        if cells is None:
            return None
        records.append(
            {
                "cells": cells,
                "t": anchor_t,
                "on_forward_ray": True,
                "enabled": interval % 2 == 0,
            }
        )

    # A target on the same infinite line but behind the control is a direction
    # counterexample: it is not on the launcher-defined forward ray.
    reverse_t = -4
    reverse_cells = _place_target(grid, bg, target, center, direction, reverse_t)
    if reverse_cells is None:
        return None
    records.append(
        {"cells": reverse_cells, "t": reverse_t, "on_forward_ray": False, "enabled": False}
    )

    # Add one or two off-axis distractors with varied positions and shapes.
    for _ in range(random.randint(1, 2)):
        placed = None
        placed_t = 0
        for _try in range(100):
            placed_t = random.randint(-4, ray_length - 1)
            anchor_q = random.choice(tuple(range(-9, -3)) + tuple(range(4, 10)))
            placed = _place_target(
                grid, bg, target, center, direction, placed_t, anchor_q, attempts=1
            )
            if placed is not None:
                break
        if placed is None:
            return None
        records.append(
            {"cells": placed, "t": placed_t, "on_forward_ray": False, "enabled": False}
        )

    expected = _expected_from_records(grid, bg, records)
    if not _validate_candidate(
        grid, expected, bg, marker, control, barrier, target, barrier_ts, records
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked obstacle-switch-ray instance."""
    for _ in range(_MAX_ATTEMPTS):
        result = _generate_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {_MAX_ATTEMPTS} attempts")
