from __future__ import annotations

import random

import numpy as np


DOOR_COUNT_RANGE = (2, 5)
HEIGHT_RANGE = (10, 26)
WIDTH_RANGE = (8, 24)
ROW_GAP_CHOICES = (2, 3)
COLOR_POOL = tuple(range(1, 10))

GEN_RETRIES = 500
COLUMN_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _inclusive_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    assert r0 == r1 or c0 == c1, "path segments must be orthogonal"
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    doors: list[tuple[int, int, int, int]] = []
    markers: list[tuple[int, int, int]] = []

    for r in range(g.shape[0]):
        cols = np.flatnonzero(g[r] != bg)
        if cols.size == 0:
            continue
        if cols.size == 1:
            c = int(cols[0])
            markers.append((r, c, int(g[r, c])))
            continue
        if cols.size == 3 and int(cols[2] - cols[0]) == 2:
            c0 = int(cols[0])
            left = int(g[r, c0])
            center = int(g[r, c0 + 1])
            right = int(g[r, c0 + 2])
            if left == right and left != bg and center != bg and center != left:
                doors.append((r, c0 + 1, left, center))
                continue
        raise AssertionError("expected only isolated markers and horizontal wall-door-wall stubs")

    assert doors, "expected at least one door stub"
    assert len(markers) == 2, "expected exactly two markers"

    wall_colors = {wall for _r, _c, wall, _door in doors}
    door_colors = {door for _r, _c, _wall, door in doors}
    marker_colors = {color for _r, _c, color in markers}
    assert len(wall_colors) == 1 and len(door_colors) == 1 and len(marker_colors) == 1

    wall_color = next(iter(wall_colors))
    door_color = next(iter(door_colors))
    path_color = next(iter(marker_colors))
    assert len({bg, wall_color, door_color, path_color}) == 4, "roles must use distinct colors"

    ordered_doors = sorted((int(r), int(c)) for r, c, _wall, _door in doors)
    door_rows = [r for r, _c in ordered_doors]
    assert len(set(door_rows)) == len(door_rows), "door rows must be distinct"
    assert all(b - a >= 2 for a, b in zip(door_rows, door_rows[1:])), "walls need a corridor between them"

    ordered_markers = sorted(((int(r), int(c)) for r, c, _color in markers), key=lambda rc: rc[0])
    top_marker, bottom_marker = ordered_markers
    assert top_marker[0] < door_rows[0], "top marker must be above the first hidden wall"
    assert bottom_marker[0] > door_rows[-1], "bottom marker must be below the last hidden wall"

    return {
        "bg": bg,
        "wall_color": wall_color,
        "door_color": door_color,
        "path_color": path_color,
        "doors": ordered_doors,
        "start": top_marker,
        "end": bottom_marker,
    }


def _plan_path_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    doors: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    cur_r, cur_c = map(int, start)
    cells.add((cur_r, cur_c))

    for door_r, door_c in doors:
        door_r = int(door_r)
        door_c = int(door_c)
        approach = (door_r - 1, cur_c)
        for cell in _inclusive_line((cur_r, cur_c), approach):
            cells.add(cell)
        for cell in _inclusive_line(approach, (door_r - 1, door_c)):
            cells.add(cell)
        cur_r, cur_c = door_r + 1, door_c

    for cell in _inclusive_line((cur_r, cur_c), (int(end[0]), cur_c)):
        cells.add(cell)
    for cell in _inclusive_line((int(end[0]), cur_c), end):
        cells.add(cell)
    return cells


def _render_from_spec(
    grid: np.ndarray,
    wall_color: int,
    door_color: int,
    path_color: int,
    doors: list[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for door_r, door_c in doors:
        out[int(door_r), :] = int(wall_color)
        out[int(door_r), int(door_c)] = int(door_color)

    for r, c in path_cells:
        assert int(out[r, c]) not in {int(wall_color), int(door_color)}, "path may not overwrite walls or doors"
        out[r, c] = int(path_color)
    return out


def _expected_output_from_layout(
    grid: np.ndarray,
    wall_color: int,
    door_color: int,
    path_color: int,
    door_rows: list[int],
    door_cols: list[int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> np.ndarray:
    doors = list(zip(door_rows, door_cols))
    return _render_from_spec(grid, wall_color, door_color, path_color, doors, _plan_path_cells(start, end, doors))


def _wrong_output_with_overwritten_centers(
    grid: np.ndarray,
    wall_color: int,
    door_color: int,
    path_color: int,
    door_rows: list[int],
    door_cols: list[int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for door_r, door_c in zip(door_rows, door_cols):
        out[int(door_r), :] = int(wall_color)
        out[int(door_r), int(door_c)] = int(door_color)

    cur_r, cur_c = map(int, start)
    for door_r, door_c in zip(door_rows, door_cols):
        for cell in _inclusive_line((cur_r, cur_c), (door_r - 1, cur_c)):
            out[cell] = int(path_color)
        for cell in _inclusive_line((door_r - 1, cur_c), (door_r - 1, door_c)):
            out[cell] = int(path_color)
        for cell in _inclusive_line((door_r - 1, door_c), (door_r + 1, door_c)):
            out[cell] = int(path_color)
        cur_r, cur_c = door_r + 1, door_c
    for cell in _inclusive_line((cur_r, cur_c), (int(end[0]), cur_c)):
        out[cell] = int(path_color)
    for cell in _inclusive_line((int(end[0]), cur_c), end):
        out[cell] = int(path_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden full-width walls from door stubs, then draw the marker route through the doorway sequence.

    Input structure:
    - A black background contains several horizontal three-pixel stubs, each on its own row.
    - In every stub the two end pixels share the wall color and the middle pixel is the door color.
    - Two same-colored marker pixels sit above the top stub and below the bottom stub, with no other foreground cells present.

    Rule:
    - Treat each three-pixel stub as the only visible fragment of a full horizontal wall on that row.
    - Fill every other cell on each stub row with the wall color while keeping the middle door cell unchanged.
    - Starting from the upper marker, draw the marker-colored route through the corridor system: go vertically to the row above the next wall, move horizontally on that row to the door center column, pass through that doorway into the next corridor, and after the last doorway continue down to the lower marker row and then horizontally to the lower marker.

    Key insight:
    - The visible three-pixel stubs are not independent objects: each one uniquely reveals an unseen full wall together with its single traversable opening, so the route only becomes well-defined after reconstructing those hidden walls.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    path_cells = _plan_path_cells(scene["start"], scene["end"], scene["doors"])
    return _render_from_spec(
        np.asarray(grid, dtype=int),
        scene["wall_color"],
        scene["door_color"],
        scene["path_color"],
        scene["doors"],
        path_cells,
    )


def _sample_spaced_positions(
    rng: random.Random,
    n: int,
    lo: int,
    hi: int,
    min_gap: int,
) -> list[int] | None:
    span = int(hi - lo + 1)
    required = int(n + (n - 1) * (min_gap - 1))
    if span < required:
        return None
    compressed = span - (n - 1) * (min_gap - 1)
    picks = sorted(rng.sample(range(compressed), n))
    return [int(lo + picks[i] + i * (min_gap - 1)) for i in range(n)]


def _sample_route_columns(
    rng: random.Random,
    width: int,
    n_doors: int,
    zigzag: bool,
) -> tuple[list[int], int, int] | None:
    for _ in range(COLUMN_RETRIES):
        door_cols = [rng.randint(1, width - 2) for _ in range(n_doors)]
        diffs = [door_cols[i + 1] - door_cols[i] for i in range(n_doors - 1)]
        if len(set(door_cols)) < 2:
            continue
        if not any(diff != 0 for diff in diffs):
            continue
        if not any(abs(diff) >= 2 for diff in diffs):
            continue

        start_col = rng.randrange(width)
        end_col = rng.randrange(width)
        hops = [abs(start_col - door_cols[0]), *[abs(diff) for diff in diffs], abs(end_col - door_cols[-1])]
        if sum(hops) < 5 or max(hops) < 2:
            continue

        if zigzag:
            nonzero = [diff for diff in diffs if diff != 0]
            if len(nonzero) < 2 or len({diff > 0 for diff in nonzero}) < 2:
                continue

        return door_cols, int(start_col), int(end_col)
    return None


def _build_input(
    height: int,
    width: int,
    wall_color: int,
    door_color: int,
    path_color: int,
    door_rows: list[int],
    door_cols: list[int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros((int(height), int(width)), dtype=int)
    for door_r, door_c in zip(door_rows, door_cols):
        grid[int(door_r), int(door_c - 1)] = int(wall_color)
        grid[int(door_r), int(door_c)] = int(door_color)
        grid[int(door_r), int(door_c + 1)] = int(wall_color)
    grid[start] = int(path_color)
    grid[end] = int(path_color)
    return grid


def _validate_case(
    grid: np.ndarray,
    out: np.ndarray,
    wall_color: int,
    door_color: int,
    path_color: int,
    door_rows: list[int],
    door_cols: list[int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    expected = _expected_output_from_layout(
        grid,
        wall_color,
        door_color,
        path_color,
        door_rows,
        door_cols,
        start,
        end,
    )
    if not np.array_equal(out, expected):
        return False

    path_cells = _plan_path_cells(start, end, list(zip(door_rows, door_cols)))
    if any((door_r, door_c) in path_cells for door_r, door_c in zip(door_rows, door_cols)):
        return False

    for door_r, door_c in zip(door_rows, door_cols):
        if not np.all(out[int(door_r), : int(door_c)] == int(wall_color)):
            return False
        if not np.all(out[int(door_r), int(door_c) + 1 :] == int(wall_color)):
            return False
        if int(out[int(door_r), int(door_c)]) != int(door_color):
            return False

    wrong_hidden = grid.copy()
    for r, c in path_cells:
        wrong_hidden[r, c] = int(path_color)
    if np.array_equal(out, wrong_hidden):
        return False

    wrong_centers = _wrong_output_with_overwritten_centers(
        grid,
        wall_color,
        door_color,
        path_color,
        door_rows,
        door_cols,
        start,
        end,
    )
    if np.array_equal(out, wrong_centers):
        return False

    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_doors = rng.randint(*DOOR_COUNT_RANGE)
    min_gap = rng.choice(ROW_GAP_CHOICES)
    min_height = max(HEIGHT_RANGE[0], n_doors + 2 + (n_doors - 1) * (min_gap - 1))
    if min_height > HEIGHT_RANGE[1]:
        return None

    height = rng.randint(min_height, HEIGHT_RANGE[1])
    width = rng.randint(WIDTH_RANGE[0], WIDTH_RANGE[1])
    door_rows = _sample_spaced_positions(rng, n_doors, 1, height - 2, min_gap)
    if door_rows is None:
        return None

    zigzag = n_doors >= 3 and rng.random() < 0.6
    sampled = _sample_route_columns(rng, width, n_doors, zigzag)
    if sampled is None:
        return None
    door_cols, start_col, end_col = sampled

    start_row = rng.randint(0, door_rows[0] - 1)
    end_row = rng.randint(door_rows[-1] + 1, height - 1)
    start = (int(start_row), int(start_col))
    end = (int(end_row), int(end_col))

    wall_color, door_color, path_color = rng.sample(COLOR_POOL, 3)
    grid = _build_input(height, width, wall_color, door_color, path_color, door_rows, door_cols, start, end)

    try:
        out = main(grid)
    except Exception:
        return None

    if not _validate_case(grid, out, wall_color, door_color, path_color, door_rows, door_cols, start, end):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        candidate = _generate_attempt(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
