from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
RED = 2
GREEN = 3
CYAN = 8

DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ORIENTATIONS = ("horizontal", "vertical")
MAX_GENERATION_ATTEMPTS = 800

Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    """ARC coordinates are always (row, column), as in arc_utils.in_bounds."""
    r, c = cell
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _add(a: Coord, b: Coord) -> Coord:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Coord, b: Coord) -> Coord:
    return a[0] - b[0], a[1] - b[1]


def _is_boundary(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    h, w = shape
    return r in (0, h - 1) or c in (0, w - 1)


def _diagonal_graph(cells: set[Coord]) -> dict[Coord, list[Coord]]:
    graph: dict[Coord, list[Coord]] = {}
    for r, c in cells:
        graph[(r, c)] = [
            (r + dr, c + dc)
            for dr, dc in DIAGONALS
            if (r + dr, c + dc) in cells
        ]
    return graph


def _ordered_path(cells: set[Coord], start: Coord) -> list[Coord]:
    """Follow the unique diagonal chain from one specified boundary endpoint."""
    graph = _diagonal_graph(cells)
    assert graph and all(1 <= len(v) <= 2 for v in graph.values()), (
        "cyan polyline must be one unbranched diagonal chain"
    )
    endpoints = [cell for cell, neighbors in graph.items() if len(neighbors) == 1]
    assert len(endpoints) == 2 and start in endpoints

    ordered = [start]
    previous: Coord | None = None
    current = start
    while True:
        onward = [cell for cell in graph[current] if cell != previous]
        if not onward:
            break
        assert len(onward) == 1
        previous, current = current, onward[0]
        ordered.append(current)

    assert len(ordered) == len(cells), "cyan polyline must be connected"
    return ordered


def _mirror_at(bend: Coord, incoming: Coord, outgoing: Coord) -> tuple[str, tuple[Coord, ...]]:
    """Infer one erased mirror in the path's own incoming-direction frame."""
    dr0, dc0 = incoming
    dr1, dc1 = outgoing
    center = bend[0] + dr0, bend[1] + dc0

    if dr1 == -dr0 and dc1 == dc0:
        orientation = "horizontal"
        cells = ((center[0], center[1] - 1), center, (center[0], center[1] + 1))
    elif dr1 == dr0 and dc1 == -dc0:
        orientation = "vertical"
        cells = ((center[0] - 1, center[1]), center, (center[0] + 1, center[1]))
    else:
        raise AssertionError("each bend must flip exactly one direction component")
    return orientation, cells


def _reconstruction_from(
    grid: np.ndarray, cyan_cells: set[Coord], start: Coord
) -> tuple[list[Coord], list[tuple[str, tuple[Coord, ...]]]] | None:
    ordered = _ordered_path(cyan_cells, start)
    mirrors: list[tuple[str, tuple[Coord, ...]]] = []

    for i in range(1, len(ordered) - 1):
        incoming = _sub(ordered[i], ordered[i - 1])
        outgoing = _sub(ordered[i + 1], ordered[i])
        if incoming == outgoing:
            continue
        try:
            mirror = _mirror_at(ordered[i], incoming, outgoing)
        except AssertionError:
            return None
        _orientation, segment = mirror
        # A completely erased mirror must fit and occupy formerly black cells.
        if any(not _in_bounds(grid.shape, cell) for cell in segment):
            return None
        if any(int(grid[cell]) != BACKGROUND for cell in segment):
            return None
        mirrors.append(mirror)

    if not mirrors:
        return None
    return ordered, mirrors


def _valid_reconstructions(
    grid: np.ndarray,
) -> list[tuple[list[Coord], list[tuple[str, tuple[Coord, ...]]]]]:
    cyan_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == CYAN)}
    assert len(cyan_cells) >= 5, "expected a visible cyan diagonal polyline"
    graph = _diagonal_graph(cyan_cells)
    endpoints = [cell for cell, neighbors in graph.items() if len(neighbors) == 1]
    assert len(endpoints) == 2
    assert all(_is_boundary(grid.shape, cell) for cell in endpoints), (
        "both cyan endpoints must lie on the boundary"
    )

    answers = []
    for start in endpoints:
        candidate = _reconstruction_from(grid, cyan_cells, start)
        if candidate is not None:
            answers.append(candidate)
    return answers


def main(grid: np.ndarray) -> np.ndarray:
    """Recover erased red mirror segments from the turns of a cyan diagonal polyline.

    Input structure:
    - A black grid contains one unbranched cyan polyline whose 45-degree diagonal
      segments enter and leave at the boundary, plus one or two horizontal or
      vertical green transparent bars crossed only on straight path portions.
    - No red pixel is present; the visible boundary geometry makes exactly one
      traversal direction yield complete in-bounds erased mirrors.

    Rule:
    - Traverse the cyan chain from its uniquely valid entry endpoint.  At every
      bend, put the mirror center one diagonal cell ahead in the incoming
      direction.  A flipped row component gives a horizontal length-3 red
      segment; a flipped column component gives a vertical length-3 red segment.
    - Preserve every cyan and green input pixel and add all inferred red cells.

    Key insight:
    - A bend is an inverse reflection trace: its missing forward cell locates the
      mirror, while the one direction component that flips determines the
      mirror's orientation.  Straight crossings of green bars are decoys.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, GREEN, CYAN})
    assert np.any(g == GREEN), "expected at least one transparent green bar"

    candidates = _valid_reconstructions(g)
    assert len(candidates) == 1, "visible clues must select one traversal direction"
    _ordered, mirrors = candidates[0]

    out = g.copy()
    for _orientation, segment in mirrors:
        for cell in segment:
            out[cell] = RED
    return out


@dataclass(frozen=True)
class _Scene:
    grid: np.ndarray
    path: tuple[Coord, ...]
    bends: tuple[Coord, ...]
    mirrors: tuple[tuple[str, tuple[Coord, ...]], ...]
    bars: tuple[tuple[Coord, ...], ...]


def _reflected(direction: Coord, orientation: str) -> Coord:
    dr, dc = direction
    return (-dr, dc) if orientation == "horizontal" else (dr, -dc)


def _segment(center: Coord, orientation: str) -> tuple[Coord, ...]:
    r, c = center
    if orientation == "horizontal":
        return ((r, c - 1), (r, c), (r, c + 1))
    return ((r - 1, c), (r, c), (r + 1, c))


def _path_is_clean(path: list[Coord]) -> bool:
    """Reject self-crossings and accidental diagonal shortcuts between arms."""
    if len(set(path)) != len(path):
        return False
    graph = _diagonal_graph(set(path))
    for i, cell in enumerate(path):
        expected = set()
        if i:
            expected.add(path[i - 1])
        if i + 1 < len(path):
            expected.add(path[i + 1])
        if set(graph[cell]) != expected:
            return False
    return True


def _construct_ray() -> tuple[
    tuple[int, int], list[Coord], list[Coord], list[tuple[str, tuple[Coord, ...]]]
] | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    shape = (h, w)

    # The first reflection is one step in from a random entry edge.  Its true
    # center is interior, whereas the reverse-traversal center lies on that edge
    # and cannot center a length-3 segment perpendicular to the edge.
    entry_edge = random.choice(("top", "bottom", "left", "right"))
    if entry_edge == "top":
        start = (0, random.randint(3, w - 4))
        dr, dc = 1, random.choice((-1, 1))
        first_orientation = "vertical"
    elif entry_edge == "bottom":
        start = (h - 1, random.randint(3, w - 4))
        dr, dc = -1, random.choice((-1, 1))
        first_orientation = "vertical"
    elif entry_edge == "left":
        start = (random.randint(3, h - 4), 0)
        dr, dc = random.choice((-1, 1)), 1
        first_orientation = "horizontal"
    else:
        start = (random.randint(3, h - 4), w - 1)
        dr, dc = random.choice((-1, 1)), -1
        first_orientation = "horizontal"
    direction = (dr, dc)

    bend_count = random.randint(2, 4)
    second_orientation = (
        "vertical" if first_orientation == "horizontal" else "horizontal"
    )
    orientations = [first_orientation, second_orientation]
    orientations.extend(random.choice(ORIENTATIONS) for _ in range(bend_count - 2))

    path = [start]
    bends: list[Coord] = []
    mirrors: list[tuple[str, tuple[Coord, ...]]] = []
    current = start

    for bend_index, orientation in enumerate(orientations):
        steps = 1 if bend_index == 0 else random.randint(2, 7)
        for _ in range(steps):
            current = _add(current, direction)
            if not _in_bounds(shape, current):
                return None
            path.append(current)

        center = _add(current, direction)
        segment = _segment(center, orientation)
        if any(not _in_bounds(shape, cell) for cell in segment):
            return None
        if set(segment) & set(path):
            return None
        if any(set(segment) & set(old_segment) for _old_o, old_segment in mirrors):
            return None

        new_direction = _reflected(direction, orientation)
        if not _in_bounds(shape, _add(current, new_direction)):
            return None
        bends.append(current)
        mirrors.append((orientation, segment))
        direction = new_direction

    # After the final bounce, the polyline must visibly continue before exiting.
    tail_count = 0
    while _in_bounds(shape, _add(current, direction)):
        current = _add(current, direction)
        path.append(current)
        tail_count += 1
        if tail_count > h + w:
            return None
    if tail_count < 2 or not _is_boundary(shape, path[-1]):
        return None
    if len(path) < 11 or not _path_is_clean(path):
        return None

    path_set = set(path)
    all_mirror_cells = {cell for _orientation, cells in mirrors for cell in cells}
    if path_set & all_mirror_cells:
        return None
    return shape, path, bends, mirrors


def _try_bar(
    shape: tuple[int, int],
    crossing: Coord,
    orientation: str,
    path_set: set[Coord],
    mirror_cells: set[Coord],
    occupied: set[Coord],
) -> tuple[Coord, ...] | None:
    length = random.choice((5, 7, 9))
    r, c = crossing
    offsets = list(range(1, length - 1))
    random.shuffle(offsets)
    for offset in offsets:
        if orientation == "horizontal":
            start = c - offset
            cells = tuple((r, start + j) for j in range(length))
        else:
            start = r - offset
            cells = tuple((start + j, c) for j in range(length))
        cell_set = set(cells)
        if any(not _in_bounds(shape, cell) for cell in cells):
            continue
        if cell_set & path_set != {crossing}:
            continue
        if cell_set & mirror_cells or cell_set & occupied:
            continue
        # Keep separate decoys visually distinct.
        if any(
            max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1
            for a in cell_set
            for b in occupied
        ):
            continue
        return cells
    return None


def _place_bars(
    shape: tuple[int, int],
    path: list[Coord],
    bends: list[Coord],
    mirrors: list[tuple[str, tuple[Coord, ...]]],
) -> list[tuple[Coord, ...]] | None:
    bend_indices = {path.index(cell) for cell in bends}
    eligible = [
        i
        for i in range(2, len(path) - 2)
        if _sub(path[i], path[i - 1]) == _sub(path[i + 1], path[i])
        and all(abs(i - bend_i) >= 2 for bend_i in bend_indices)
    ]
    if not eligible:
        return None
    random.shuffle(eligible)

    count = random.choice((1, 2))
    wanted_orientations = (
        [random.choice(ORIENTATIONS)]
        if count == 1
        else random.sample(list(ORIENTATIONS), 2)
    )
    path_set = set(path)
    mirror_cells = {cell for _orientation, segment in mirrors for cell in segment}
    occupied: set[Coord] = set()
    bars: list[tuple[Coord, ...]] = []

    for orientation in wanted_orientations:
        placed = None
        random.shuffle(eligible)
        for i in eligible:
            candidate = _try_bar(
                shape, path[i], orientation, path_set, mirror_cells, occupied
            )
            if candidate is not None:
                placed = candidate
                eligible.remove(i)
                break
        if placed is None:
            return None
        bars.append(placed)
        occupied.update(placed)
    return bars


def _render_scene(
    shape: tuple[int, int],
    path: list[Coord],
    bends: list[Coord],
    mirrors: list[tuple[str, tuple[Coord, ...]]],
    bars: list[tuple[Coord, ...]],
) -> _Scene:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for bar in bars:
        for cell in bar:
            grid[cell] = GREEN
    # The cyan ray is above a transparent bar at the crossing pixel, matching the
    # forward reference task's overwrite convention.
    for cell in path:
        grid[cell] = CYAN
    return _Scene(
        grid=grid,
        path=tuple(path),
        bends=tuple(bends),
        mirrors=tuple(mirrors),
        bars=tuple(bars),
    )


def _semantic_check(scene: _Scene) -> bool:
    """Check the stated inverse rule, uniqueness, branch coverage, and decoys."""
    grid = scene.grid
    try:
        candidates = _valid_reconstructions(grid)
        actual = main(grid)
    except (AssertionError, IndexError):
        return False
    if len(candidates) != 1:
        return False

    ordered, inferred = candidates[0]
    if tuple(ordered) != scene.path:
        return False
    if tuple(inferred) != scene.mirrors:
        return False
    if {orientation for orientation, _cells in inferred} != set(ORIENTATIONS):
        return False
    if len(inferred) != len(scene.bends):
        return False

    expected = grid.copy()
    for _orientation, segment in scene.mirrors:
        for cell in segment:
            if int(grid[cell]) != BACKGROUND:
                return False
            expected[cell] = RED
    if not np.array_equal(actual, expected):
        return False

    # Every green bar has exactly one cyan interruption, and that point is on a
    # straight (not turning) portion, excluding the "every bar reflects" shortcut.
    path_index = {cell: i for i, cell in enumerate(scene.path)}
    for bar in scene.bars:
        crossings = set(bar) & set(scene.path)
        if len(crossings) != 1:
            return False
        crossing = next(iter(crossings))
        i = path_index[crossing]
        if not (0 < i < len(scene.path) - 1):
            return False
        if _sub(scene.path[i], scene.path[i - 1]) != _sub(scene.path[i + 1], scene.path[i]):
            return False
        if sum(int(grid[cell]) == GREEN for cell in bar) != len(bar) - 1:
            return False

    # Since inferred red cells were black and every input foreground pixel is
    # preserved, input can be recovered from output by changing red back to black;
    # therefore two distinct generated inputs cannot collapse to one output.
    recovered = actual.copy()
    recovered[recovered == RED] = BACKGROUND
    return np.array_equal(recovered, grid)


def _gen_attempt() -> np.ndarray | None:
    built = _construct_ray()
    if built is None:
        return None
    shape, path, bends, mirrors = built
    bars = _place_bars(shape, path, bends, mirrors)
    if bars is None:
        return None
    scene = _render_scene(shape, path, bends, mirrors, bars)
    return scene.grid if _semantic_check(scene) else None


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked inverse-reflection ARC input."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid erased-mirror input")
