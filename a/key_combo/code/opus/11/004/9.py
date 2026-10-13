import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GENERATION_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _normalize(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _shift(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _paint(grid: np.ndarray, coords: set[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _components_from_cells(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS4:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def _grey_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)}
    return _components_from_cells(cells)


def _size_to_color(size: int) -> int:
    if size <= 2:
        return GREEN
    if size <= 4:
        return BLUE
    return YELLOW


def _detect_trail_direction(grid: np.ndarray, red: tuple[int, int]) -> tuple[int, int]:
    h, w = grid.shape
    rr, cc = red
    matches = []
    for dr, dc in DIRS8:
        ok = True
        for step in range(1, 4):
            r = rr + dr * step
            c = cc + dc * step
            if not _in_bounds((h, w), r, c) or int(grid[r, c]) != PINK:
                ok = False
                break
        if not ok:
            continue
        r4 = rr + dr * 4
        c4 = cc + dc * 4
        if _in_bounds((h, w), r4, c4) and int(grid[r4, c4]) == PINK:
            continue
        matches.append((dr, dc))
    assert len(matches) == 1, f"expected one pink trail direction, got {matches}"
    return matches[0]


def _ray_target(grid: np.ndarray, red: tuple[int, int], trail_dir: tuple[int, int]) -> tuple[int, int]:
    h, w = grid.shape
    dr, dc = -trail_dir[0], -trail_dir[1]
    r = red[0] + dr
    c = red[1] + dc
    while _in_bounds((h, w), r, c):
        value = int(grid[r, c])
        if value != BLACK:
            assert value == GREY, f"ray should hit grey, got {value}"
            return (r, c)
        r += dr
        c += dc
    raise AssertionError("ray left the grid without hitting grey")


def _marker_targets_from_grid(grid: np.ndarray) -> list[tuple[int, int]]:
    reds = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    assert 2 <= len(reds) <= 3
    targets = []
    for red in reds:
        trail_dir = _detect_trail_direction(grid, red)
        targets.append(_ray_target(grid, red, trail_dir))
    return targets


def _render_from_targets(grid: np.ndarray, targets: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in targets:
        out[r, c] = BLACK
    for comp in _grey_components(out):
        color = _size_to_color(len(comp))
        for r, c in comp:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Erase ray-selected grey pixels, then recolor the split grey components by their post-erasure sizes.

    Input structure:
    - The grid has a black background with 2-4 separate grey 4-connected objects.
    - There are 2-3 red single-pixel markers, each followed by exactly three pink pixels in a straight
      horizontal, vertical, or 45-degree diagonal trail.
    - For each red marker, looking away from its pink trail reaches background cells first and then a grey pixel.

    Rule:
    - For each red marker, infer the trail direction from the adjacent pink line and reverse it.
    - Cast a ray from the red pixel along that reversed direction until the first non-black cell; erase that grey
      pixel to black.
    - After all such erasures, find every remaining grey 4-connected component and recolor it by size:
      size 1-2 -> green, size 3-4 -> blue, size >=5 -> yellow.

    Key insight:
    - The pink trail does not point to what changes; it points away from it, so the ray selects a single grey
      articulation or boundary pixel whose removal can change object topology before the size-based recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    targets = _marker_targets_from_grid(g)
    return _render_from_targets(g, targets)


def _split_shape(rng: np.random.Generator) -> set[tuple[int, int]]:
    small = int(rng.integers(1, 3))
    medium = int(rng.integers(3, 5))
    dir1 = DIRS4[int(rng.integers(len(DIRS4)))]
    if int(rng.integers(2)) == 0:
        dir2 = (-dir1[0], -dir1[1])
    else:
        orth = [d for d in DIRS4 if d != dir1 and d != (-dir1[0], -dir1[1])]
        dir2 = orth[int(rng.integers(len(orth)))]
    coords = {(0, 0)}
    for step in range(1, small + 1):
        coords.add((dir1[0] * step, dir1[1] * step))
    for step in range(1, medium + 1):
        coords.add((dir2[0] * step, dir2[1] * step))
    return _normalize(coords)


def _line_shape(length: int, horizontal: bool) -> set[tuple[int, int]]:
    if horizontal:
        return {(0, c) for c in range(length)}
    return {(r, 0) for r in range(length)}


def _rect_shape(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w)}


def _grow_polyomino(size: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        pick = frontier[int(rng.integers(len(frontier)))]
        cells.add(pick)
    return _normalize(cells)


def _large_shape(rng: np.random.Generator) -> set[tuple[int, int]]:
    if int(rng.integers(2)) == 0:
        length = int(rng.integers(6, 9))
        return _line_shape(length, horizontal=bool(int(rng.integers(2))))
    dims = [(2, 3), (3, 2), (2, 4), (4, 2)]
    h, w = dims[int(rng.integers(len(dims)))]
    return _rect_shape(h, w)


def _extra_target_shape(rng: np.random.Generator) -> set[tuple[int, int]]:
    choice = int(rng.integers(3))
    if choice == 0:
        dims = [(2, 2), (2, 3), (3, 2)]
        h, w = dims[int(rng.integers(len(dims)))]
        return _rect_shape(h, w)
    if choice == 1:
        length = int(rng.integers(4, 7))
        return _line_shape(length, horizontal=bool(int(rng.integers(2))))
    return _grow_polyomino(int(rng.integers(4, 7)), rng)


def _filler_shape(rng: np.random.Generator) -> set[tuple[int, int]]:
    choice = int(rng.integers(3))
    if choice == 0:
        return _grow_polyomino(int(rng.integers(4, 9)), rng)
    if choice == 1:
        length = int(rng.integers(4, 8))
        return _line_shape(length, horizontal=bool(int(rng.integers(2))))
    dims = [(2, 2), (2, 3), (3, 2)]
    h, w = dims[int(rng.integers(len(dims)))]
    return _rect_shape(h, w)


def _can_place_shape(
    shape: set[tuple[int, int]],
    top: int,
    left: int,
    grid_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    shifted = _shift(shape, top, left)
    for r, c in shifted:
        if not _in_bounds(grid_shape, r, c):
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied:
                    return False
    return True


def _place_shape(
    shape: set[tuple[int, int]],
    grid_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    positions = [
        (r, c)
        for r in range(1, grid_shape[0] - max_r - 1)
        for c in range(1, grid_shape[1] - max_c - 1)
    ]
    if not positions:
        return None
    order = rng.permutation(len(positions))
    for idx in order:
        top, left = positions[int(idx)]
        if _can_place_shape(shape, top, left, grid_shape, occupied):
            shifted = _shift(shape, top, left)
            occupied.update(shifted)
            return shifted
    return None


def _removal_components(
    cells: set[tuple[int, int]], target: tuple[int, int]
) -> list[set[tuple[int, int]]]:
    remainder = set(cells)
    remainder.remove(target)
    return _components_from_cells(remainder)


def _find_marker_slot(
    grid: np.ndarray, target: tuple[int, int], rng: np.random.Generator
) -> tuple[tuple[int, int], int] | None:
    gaps = [1, 2, 3]
    dir_order = rng.permutation(len(DIRS8))
    gap_order = rng.permutation(len(gaps))
    for dir_idx in dir_order:
        dr, dc = DIRS8[int(dir_idx)]
        for gap_idx in gap_order:
            gap = gaps[int(gap_idx)]
            cells = []
            ok = True
            for step in range(1, gap + 5):
                r = target[0] + dr * step
                c = target[1] + dc * step
                if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BLACK:
                    ok = False
                    break
                cells.append((r, c))
            if ok:
                return (dr, dc), gap
    return None


def _place_marker(
    grid: np.ndarray, target: tuple[int, int], slot: tuple[tuple[int, int], int]
) -> None:
    (dr, dc), gap = slot
    red_r = target[0] + dr * (gap + 1)
    red_c = target[1] + dc * (gap + 1)
    grid[red_r, red_c] = RED
    for step in range(gap + 2, gap + 5):
        grid[target[0] + dr * step, target[1] + dc * step] = PINK


def _pick_split_target(
    grid: np.ndarray, cells: set[tuple[int, int]], rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[tuple[int, int], int]] | None:
    candidates = []
    for cell in cells:
        comps = _removal_components(cells, cell)
        colors = {_size_to_color(len(comp)) for comp in comps}
        if len(comps) == 2 and colors == {GREEN, BLUE}:
            candidates.append(cell)
    if not candidates:
        return None
    order = rng.permutation(len(candidates))
    for idx in order:
        target = candidates[int(idx)]
        slot = _find_marker_slot(grid, target, rng)
        if slot is not None:
            return target, slot
    return None


def _pick_non_split_target(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    rng: np.random.Generator,
    *,
    require_yellow: bool,
) -> tuple[tuple[int, int], tuple[tuple[int, int], int]] | None:
    candidates = []
    for cell in cells:
        comps = _removal_components(cells, cell)
        if len(comps) != 1:
            continue
        color = _size_to_color(len(comps[0]))
        if require_yellow and color != YELLOW:
            continue
        candidates.append(cell)
    if not candidates:
        return None
    order = rng.permutation(len(candidates))
    for idx in order:
        target = candidates[int(idx)]
        slot = _find_marker_slot(grid, target, rng)
        if slot is not None:
            return target, slot
    return None


def _validate_instance(
    grid: np.ndarray,
    explicit_targets: list[tuple[int, int]],
    split_target: tuple[int, int],
) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    grey_components = _grey_components(grid)
    if not (2 <= len(grey_components) <= 4):
        return False
    if not all(4 <= len(comp) <= 10 for comp in grey_components):
        return False

    reds = np.argwhere(grid == RED)
    pinks = np.argwhere(grid == PINK)
    if not (2 <= len(reds) <= 3) or len(pinks) != 3 * len(reds):
        return False

    try:
        detected_targets = _marker_targets_from_grid(grid)
    except Exception:
        return False
    if set(detected_targets) != set(explicit_targets):
        return False
    if len(set(detected_targets)) != len(detected_targets):
        return False

    source_cells = None
    for comp in grey_components:
        if split_target in comp:
            source_cells = comp
            break
    if source_cells is None:
        return False
    split_comps = _removal_components(source_cells, split_target)
    if len(split_comps) != 2:
        return False

    expected = _render_from_targets(grid, explicit_targets)
    if np.any(expected == GREY):
        return False
    recolor_palette = set(map(int, expected.ravel()))
    if not {GREEN, BLUE, YELLOW}.issubset(recolor_palette):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(15, 19))
    w = int(rng.integers(15, 19))
    marker_count = int(rng.integers(2, 4))
    object_count = int(rng.integers(max(2, marker_count), 5))

    labeled_shapes: list[tuple[str, set[tuple[int, int]]]] = [
        ("split", _split_shape(rng)),
        ("yellow", _large_shape(rng)),
    ]
    if marker_count == 3:
        labeled_shapes.append(("extra", _extra_target_shape(rng)))
    while len(labeled_shapes) < object_count:
        labeled_shapes.append(("filler", _filler_shape(rng)))

    occupied: set[tuple[int, int]] = set()
    placed: list[tuple[str, set[tuple[int, int]]]] = []
    for label, shape in labeled_shapes:
        coords = _place_shape(shape, (h, w), occupied, rng)
        if coords is None:
            return None
        placed.append((label, coords))

    grid = np.zeros((h, w), dtype=int)
    for _, coords in placed:
        _paint(grid, coords, GREY)

    split_cells = next(coords for label, coords in placed if label == "split")
    yellow_cells = next(coords for label, coords in placed if label == "yellow")
    extra_cells = next((coords for label, coords in placed if label == "extra"), None)

    split_pick = _pick_split_target(grid, split_cells, rng)
    if split_pick is None:
        return None
    split_target, split_slot = split_pick
    _place_marker(grid, split_target, split_slot)

    yellow_pick = _pick_non_split_target(grid, yellow_cells, rng, require_yellow=True)
    if yellow_pick is None:
        return None
    yellow_target, yellow_slot = yellow_pick
    _place_marker(grid, yellow_target, yellow_slot)

    explicit_targets = [split_target, yellow_target]

    if marker_count == 3:
        assert extra_cells is not None
        extra_pick = _pick_non_split_target(grid, extra_cells, rng, require_yellow=False)
        if extra_pick is None:
            return None
        extra_target, extra_slot = extra_pick
        _place_marker(grid, extra_target, extra_slot)
        explicit_targets.append(extra_target)

    if not _validate_instance(grid, explicit_targets, split_target):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
