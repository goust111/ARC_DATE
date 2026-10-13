from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GRAY = 5
CYAN = 8

GRID_SIZE_RANGE = (14, 20)
CROP_SIZES = (9, 11, 13)
DISTRACTOR_COUNT_RANGE = (5, 10)
GEN_ATTEMPTS = 400

Coord = tuple[int, int]
Track = tuple[int, int, int, int]


def _rotate_once(r: int, c: int, bbox: tuple[int, int, int, int], clockwise: bool) -> Coord:
    r0, c0, r1, c1 = map(int, bbox)
    cr2 = r0 + r1
    cc2 = c0 + c1
    if clockwise:
        nr2 = cr2 + 2 * c - cc2
        nc2 = cc2 - 2 * r + cr2
    else:
        nr2 = cr2 - 2 * c + cc2
        nc2 = cc2 + 2 * r - cr2

    assert nr2 % 2 == 0 and nc2 % 2 == 0, "rotation must land on integer grid cells"
    return nr2 // 2, nc2 // 2


def _blue_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == BLUE)
    assert coords.size > 0, "expected blue shape inside the crop"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return r0, c0, r1, c1


def _gray_tracks(grid: np.ndarray) -> list[Track]:
    gray = np.argwhere(grid == GRAY)
    assert gray.size > 0, "expected gray rails"

    cols = sorted({int(c) for _r, c in gray})
    assert len(cols) == 2, "expected exactly two gray rail columns"

    tracks: list[Track] = []
    for col in cols:
        rows = sorted(int(r) for r, c in gray if int(c) == col)
        assert rows == list(range(rows[0], rows[-1] + 1)), "gray rail must be one vertical segment"
        tracks.append((col, rows[0], rows[-1], len(rows)))
    return tracks


def _crop_bounds(tracks: list[Track], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    assert len(tracks) == 2
    height, _width = map(int, shape)
    tracks = sorted(tracks)
    max_len = max(length for _col, _top, _bottom, length in tracks)
    longest = [track for track in tracks if track[3] == max_len]

    top = min(track[1] for track in longest)
    bottom = max(track[2] for track in longest)
    r0 = max(0, top - 1)
    r1 = min(height - 1, bottom + 1)
    c0 = int(tracks[0][0])
    c1 = int(tracks[1][0])
    return r0, r1, c0, c1


def _decode_corner_direction(crop: np.ndarray) -> bool:
    tl = int(crop[0, 0]) == CYAN
    tr = int(crop[0, -1]) == CYAN
    bl = int(crop[-1, 0]) == CYAN
    br = int(crop[-1, -1]) == CYAN

    if tl and br and not tr and not bl:
        return True
    if tr and bl and not tl and not br:
        return False
    raise AssertionError("corner markers must occupy exactly one diagonal")


def _complete_crop(crop: np.ndarray, clockwise: bool) -> np.ndarray:
    out = crop.copy()
    bbox = _blue_bbox(crop)
    blue_coords = [tuple(map(int, rc)) for rc in np.argwhere(crop == BLUE)]
    height, width = map(int, crop.shape)

    for r, c in blue_coords:
        nr, nc = _rotate_once(r, c, bbox, clockwise)
        if 0 <= nr < height and 0 <= nc < width and int(out[nr, nc]) == BG:
            out[nr, nc] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the gray-rail window, decode the cyan corner diagonal as a 90-degree rotation direction, and mark the blue shape's missing rotated cells in red.

    Input structure:
    - A black background contains exactly two vertical gray rails that define the crop columns.
    - Cyan corner markers appear on one diagonal of the crop rectangle, while extra cyan pixels outside the crop are distractors.
    - Inside the rail-bounded window, a blue shape is almost 4-fold rotationally symmetric and is missing one directional sector.

    Rule:
    - Sort the gray rails by column, crop from the left rail column to the right rail column, and from one row above the longest rail's top to one row below its bottom.
    - Read the crop-corner cyan diagonal: top-left plus bottom-right means rotate clockwise, while top-right plus bottom-left means rotate counterclockwise.
    - In the crop only, rotate every blue pixel by one 90-degree step around the blue shape bbox center and color each still-black rotated target red.

    Key insight:
    - The cyan corner markers have two jobs at once: they belong to the cropped frame and also encode which one-step rotational completion must be applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    tracks = _gray_tracks(g)
    r0, r1, c0, c1 = _crop_bounds(tracks, g.shape)
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    clockwise = _decode_corner_direction(crop)
    return _complete_crop(crop, clockwise)


def _rotate_rel_cw(rel: Coord, steps: int) -> Coord:
    dr, dc = map(int, rel)
    steps %= 4
    for _ in range(steps):
        dr, dc = dc, -dr
    return dr, dc


def _grow_quadrant_motif(radius: int, target_area: int) -> set[Coord]:
    cells = {(radius, radius)}
    while len(cells) < target_area:
        frontier = set()
        for a, b in cells:
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                aa, bb = a + da, b + db
                if 1 <= aa <= radius and 1 <= bb <= radius and (aa, bb) not in cells:
                    frontier.add((aa, bb))
        assert frontier, "quadrant motif frontier should stay non-empty"
        cells.add(random.choice(sorted(frontier)))

    if not any(a == radius for a, _b in cells):
        cells.add((radius, random.randint(1, radius)))
    if not any(b == radius for _a, b in cells):
        cells.add((random.randint(1, radius), radius))
    return {(-a, b) for a, b in cells}


def _axis_members(distances: list[int]) -> dict[int, set[Coord]]:
    members: dict[int, set[Coord]] = {}
    for direction in range(4):
        members[direction] = set()
    for dist in distances:
        north = (-dist, 0)
        for direction in range(4):
            members[direction].add(_rotate_rel_cw(north, direction))
    return members


def _build_local_blue_shape(size: int) -> tuple[set[Coord], int]:
    center = size // 2
    max_radius = min(4, center - 2)
    radius = random.randint(2, max_radius)
    target_area = random.randint(max(3, radius), min(radius * radius, radius + 2))
    motif = _grow_quadrant_motif(radius, target_area)

    full_rel: set[Coord] = set()
    rotated_quadrants: dict[int, set[Coord]] = {}
    for direction in range(4):
        rotated = {_rotate_rel_cw(coord, direction) for coord in motif}
        rotated_quadrants[direction] = rotated
        full_rel |= rotated

    axis_choices = list(range(1, radius))
    axis_count = 1 if len(axis_choices) == 1 else random.randint(1, min(2, len(axis_choices)))
    distances = random.sample(axis_choices, axis_count)
    axis_by_dir = _axis_members(distances)
    for coords in axis_by_dir.values():
        full_rel |= coords

    missing_quadrant = random.randrange(4)
    missing_axes = {missing_quadrant, (missing_quadrant + 1) % 4}
    observed_rel = set(full_rel)
    observed_rel -= rotated_quadrants[missing_quadrant]
    for axis_dir in missing_axes:
        observed_rel -= axis_by_dir[axis_dir]

    if random.random() < 0.5:
        observed_rel.add((0, 0))

    local = {(center + dr, center + dc) for dr, dc in observed_rel}
    assert all(0 <= r < size and 0 <= c < size for r, c in local)
    return local, missing_quadrant


def _base_crop(size: int, blue_cells: set[Coord], clockwise: bool) -> np.ndarray:
    crop = np.full((size, size), BG, dtype=int)
    crop[1:-1, 0] = GRAY
    crop[1:-1, -1] = GRAY
    if clockwise:
        crop[0, 0] = CYAN
        crop[-1, -1] = CYAN
    else:
        crop[0, -1] = CYAN
        crop[-1, 0] = CYAN
    for r, c in blue_cells:
        crop[r, c] = BLUE
    return crop


def _place_crop_in_grid(crop: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    size = int(crop.shape[0])
    height = random.randint(max(GRID_SIZE_RANGE[0], size + 3), GRID_SIZE_RANGE[1])
    width = random.randint(max(GRID_SIZE_RANGE[0], size + 3), GRID_SIZE_RANGE[1])
    top = random.randint(1, height - size - 1)
    left = random.randint(1, width - size - 1)
    grid = np.full((height, width), BG, dtype=int)
    grid[top : top + size, left : left + size] = crop
    return grid, (top, left)


def _add_cyan_distractors(grid: np.ndarray, crop_top: int, crop_left: int, crop_size: int) -> None:
    height, width = map(int, grid.shape)
    outside: list[Coord] = []
    special: list[Coord] = []
    top = crop_top
    bottom = crop_top + crop_size - 1
    left = crop_left
    right = crop_left + crop_size - 1

    for r in range(height):
        for c in range(width):
            inside_crop = top <= r <= bottom and left <= c <= right
            if inside_crop or int(grid[r, c]) != BG:
                continue
            outside.append((r, c))
            if r in {top, bottom} or c in {left, right}:
                special.append((r, c))

    count = random.randint(*DISTRACTOR_COUNT_RANGE)
    chosen: set[Coord] = set()
    if special:
        special_n = min(len(special), random.randint(1, min(3, count)))
        chosen.update(random.sample(special, special_n))
    remaining = [coord for coord in outside if coord not in chosen]
    chosen.update(random.sample(remaining, count - len(chosen)))

    for r, c in chosen:
        grid[r, c] = CYAN


def _oracle_case() -> tuple[np.ndarray, np.ndarray]:
    crop_size = random.choice(CROP_SIZES)
    blue_cells, missing_quadrant = _build_local_blue_shape(crop_size)
    clockwise = random.choice((True, False))
    crop = _base_crop(crop_size, blue_cells, clockwise)
    expected = _complete_crop(crop, clockwise)
    wrong = _complete_crop(crop, not clockwise)

    bbox = _blue_bbox(crop)
    center = crop_size // 2
    assert bbox[0] + bbox[2] == 2 * center and bbox[1] + bbox[3] == 2 * center
    assert not np.array_equal(expected, wrong), "corner direction must affect the output"
    assert int(np.sum(expected == RED)) >= 4, "completion must be visually clear"

    quadrants = {
        0: [(r, c) for r, c in blue_cells if r < center and c > center],
        1: [(r, c) for r, c in blue_cells if r > center and c > center],
        2: [(r, c) for r, c in blue_cells if r > center and c < center],
        3: [(r, c) for r, c in blue_cells if r < center and c < center],
    }
    for idx, coords in quadrants.items():
        if idx == missing_quadrant:
            assert not coords, "missing quadrant should be empty"
        else:
            assert coords, "the three visible quadrants must all exist"

    grid, (top, left) = _place_crop_in_grid(crop)
    _add_cyan_distractors(grid, top, left, crop_size)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid, expected = _oracle_case()
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid gray-rail rotation-completion puzzle")
