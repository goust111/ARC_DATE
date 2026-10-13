from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ANCHOR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_PAIRS = 1
MAX_PAIRS = 4
MIN_LEG = 2
MAX_LEG = 8
OUTER_ATTEMPTS = 400
PAIR_ATTEMPTS = 400

Coord = tuple[int, int]


def _inclusive_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _neighbors4(cell: Coord, height: int, width: int) -> list[Coord]:
    r, c = cell
    out: list[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            out.append((nr, nc))
    return out


def _path_cells(start: Coord, target: Coord, first_axis: str) -> list[Coord]:
    sr, sc = start
    tr, tc = target
    assert sr != tr and sc != tc

    if first_axis == "h":
        cells = [(sr, c) for c in _inclusive_range(sc, tc)]
        cells.extend((r, tc) for r in list(_inclusive_range(sr, tr))[1:])
        return cells
    if first_axis == "v":
        cells = [(r, sc) for r in _inclusive_range(sr, tr)]
        cells.extend((tr, c) for c in list(_inclusive_range(sc, tc))[1:])
        return cells
    raise ValueError(f"unknown first axis: {first_axis}")


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape

    grey_positions = {tuple(map(int, cell)) for cell in np.argwhere(g == GREY)}
    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, GREY)]

    out = g.copy()
    out[g == GREY] = BG

    used_markers: set[Coord] = set()
    for color in colors:
        anchors = [tuple(map(int, cell)) for cell in np.argwhere(g == color)]
        assert len(anchors) == 2, "each anchor color must appear exactly twice"
        assert anchors[0][0] != anchors[1][0] and anchors[0][1] != anchors[1][1]

        marker_lists: list[list[Coord]] = []
        for anchor in anchors:
            markers = [cell for cell in _neighbors4(anchor, height, width) if cell in grey_positions]
            assert len(markers) <= 1, "an anchor may touch at most one grey marker"
            marker_lists.append(markers)

        start_indices = [idx for idx, markers in enumerate(marker_lists) if len(markers) == 1]
        assert len(start_indices) == 1, "exactly one anchor in the pair must carry the grey marker"
        start_idx = start_indices[0]
        target_idx = 1 - start_idx
        assert len(marker_lists[target_idx]) == 0

        start = anchors[start_idx]
        target = anchors[target_idx]
        marker = marker_lists[start_idx][0]
        assert marker not in used_markers, "each grey marker must belong to a single pair"
        used_markers.add(marker)

        first_axis = "h" if marker[0] == start[0] else "v"
        for r, c in _path_cells(start, target, first_axis):
            value = int(g[r, c])
            assert value in (BG, GREY, color), "paths may only pass through black, grey, or their own anchors"
            if (r, c) not in grey_positions:
                out[r, c] = color

    assert used_markers == grey_positions, "every grey marker must be paired with exactly one anchor pair"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each same-colored anchor pair with an L path chosen by the adjacent grey orientation marker.

    Input structure:
    - The background is black.
    - Every non-grey foreground color appears exactly twice as single-cell anchors.
    - For each color pair, exactly one of the two anchors has one orthogonally adjacent grey marker, and the two anchors are not in the same row or column.

    Rule:
    - Pair the anchors by color.
    - The anchor touching grey is the start anchor. If its grey marker sits left or right of it, draw the pair's L path by moving horizontally first to the other anchor's column, then vertically to the other anchor. If the grey marker sits above or below, move vertically first, then horizontally.
    - Paint the whole L path in the anchor color, but keep every original grey marker cell black in the final output.

    Key insight:
    - Matching colors identifies the endpoints, while the grey marker only disambiguates which of the two possible L turns to use.
    """
    return _solve(grid)


def _balanced_axes(pair_count: int, rng: random.Random) -> list[str]:
    if pair_count == 1:
        return [rng.choice(("h", "v"))]
    extra_horizontal = rng.choice((0, 1)) if pair_count % 2 else 0
    horizontal_count = pair_count // 2 + extra_horizontal
    vertical_count = pair_count - horizontal_count
    axes = ["h"] * horizontal_count + ["v"] * vertical_count
    rng.shuffle(axes)
    return axes


def _sample_pair(height: int, width: int, first_axis: str, rng: random.Random) -> dict[str, object] | None:
    start = (rng.randrange(1, height - 1), rng.randrange(1, width - 1))
    sr, sc = start
    vdir = rng.choice((-1, 1))
    hdir = rng.choice((-1, 1))

    max_vertical = height - 1 - sr if vdir == 1 else sr
    max_horizontal = width - 1 - sc if hdir == 1 else sc
    if max_vertical < MIN_LEG or max_horizontal < MIN_LEG:
        return None

    dr = rng.randint(MIN_LEG, min(MAX_LEG, max_vertical))
    dc = rng.randint(MIN_LEG, min(MAX_LEG, max_horizontal))
    target = (sr + vdir * dr, sc + hdir * dc)

    marker = (sr, sc - hdir) if first_axis == "h" else (sr - vdir, sc)
    path = _path_cells(start, target, first_axis)
    assert marker not in path

    return {
        "start": start,
        "target": target,
        "marker": marker,
        "path": path,
        "first_axis": first_axis,
    }


def _is_clear(candidate: dict[str, object], reserved_paths: set[Coord], features: set[Coord]) -> bool:
    start = candidate["start"]
    target = candidate["target"]
    marker = candidate["marker"]
    path = set(candidate["path"])

    assert isinstance(start, tuple) and isinstance(target, tuple) and isinstance(marker, tuple)
    candidate_features = {start, target, marker}

    if path & reserved_paths:
        return False
    if candidate_features & reserved_paths:
        return False
    if path & features:
        return False

    for cell in candidate_features:
        for other in features:
            if abs(cell[0] - other[0]) + abs(cell[1] - other[1]) <= 1:
                return False
    return True


def _build_grid(height: int, width: int, pair_specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in pair_specs:
        color = int(spec["color"])
        start = spec["start"]
        target = spec["target"]
        marker = spec["marker"]
        assert isinstance(start, tuple) and isinstance(target, tuple) and isinstance(marker, tuple)
        grid[start] = color
        grid[target] = color
        grid[marker] = GREY
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(OUTER_ATTEMPTS):
        pair_count = rng.randint(MIN_PAIRS, MAX_PAIRS)
        base_side = 9 + 2 * pair_count
        height = rng.randint(base_side, min(24, base_side + 6))
        width = rng.randint(base_side, min(24, base_side + 6))

        colors = rng.sample(ANCHOR_COLORS, pair_count)
        axes = _balanced_axes(pair_count, rng)

        pair_specs: list[dict[str, object]] = []
        reserved_paths: set[Coord] = set()
        features: set[Coord] = set()

        failed = False
        for color, first_axis in zip(colors, axes):
            placed = False
            for _ in range(PAIR_ATTEMPTS):
                candidate = _sample_pair(height, width, first_axis, rng)
                if candidate is None or not _is_clear(candidate, reserved_paths, features):
                    continue

                candidate["color"] = color
                pair_specs.append(candidate)
                reserved_paths.update(candidate["path"])
                features.update({candidate["start"], candidate["target"], candidate["marker"]})
                placed = True
                break

            if not placed:
                failed = True
                break

        if failed:
            continue

        grid = _build_grid(height, width, pair_specs)
        solved = _solve(grid)
        if np.array_equal(grid, solved):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
