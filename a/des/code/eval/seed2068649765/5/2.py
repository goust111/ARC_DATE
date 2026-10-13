from __future__ import annotations

import random

import numpy as np


BG = 0
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
HORIZONTAL = {LEFT, RIGHT}
VERTICAL = {UP, DOWN}
DIR_TO_ROT = {UP: 0, LEFT: 1, DOWN: 2, RIGHT: 3}


GRID_MIN_BY_PAIR_COUNT = {2: 14, 3: 17, 4: 20}
GRID_MAX = 24
PAIR_COUNT_RANGE = (2, 4)
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MIN_LEG = 2
PAIR_PLACE_RETRIES = 250
GENERATE_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in (UP, DOWN, LEFT, RIGHT):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)

    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_local(mask: np.ndarray) -> tuple[int, int]:
    holes = np.argwhere(~mask)
    assert holes.shape == (1, 2)
    return int(holes[0, 0]), int(holes[0, 1])


def _orth_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
    assert c0 == c1
    step = 1 if r1 > r0 else -1
    return [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]


def _is_forward(hole: tuple[int, int], bend: tuple[int, int], direction: tuple[int, int]) -> bool:
    if direction == UP:
        return bend[1] == hole[1] and bend[0] < hole[0]
    if direction == DOWN:
        return bend[1] == hole[1] and bend[0] > hole[0]
    if direction == LEFT:
        return bend[0] == hole[0] and bend[1] < hole[1]
    assert direction == RIGHT
    return bend[0] == hole[0] and bend[1] > hole[1]


def _parse_sprite(
    g: np.ndarray, color: int, coords: list[tuple[int, int]]
) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape

    color_mask = patch == int(color)
    assert int(color_mask.sum()) == len(coords)
    assert not np.any((patch != BG) & (patch != int(color)))
    assert len(coords) == h * w - 1

    holes = np.argwhere(patch == BG)
    assert holes.shape == (1, 2)
    hole_r, hole_c = map(int, holes[0])

    dirs: list[tuple[int, int]] = []
    if hole_r == 0:
        dirs.append(UP)
    if hole_r == h - 1:
        dirs.append(DOWN)
    if hole_c == 0:
        dirs.append(LEFT)
    if hole_c == w - 1:
        dirs.append(RIGHT)
    assert dirs

    return {
        "hole": (int(r0 + hole_r), int(c0 + hole_c)),
        "dirs": dirs,
    }


def _candidate_connection(
    g: np.ndarray,
    color: int,
    sprite_a: dict[str, object],
    sprite_b: dict[str, object],
) -> list[tuple[int, int]]:
    hole_a = tuple(sprite_a["hole"])
    hole_b = tuple(sprite_b["hole"])
    dirs_a = list(sprite_a["dirs"])
    dirs_b = list(sprite_b["dirs"])

    valid_paths: list[list[tuple[int, int]]] = []
    for dir_a in dirs_a:
        for dir_b in dirs_b:
            if (dir_a in HORIZONTAL) == (dir_b in HORIZONTAL):
                continue

            if dir_a in HORIZONTAL:
                bend = (hole_a[0], hole_b[1])
            else:
                bend = (hole_b[0], hole_a[1])

            if bend == hole_a or bend == hole_b:
                continue
            if not _is_forward(hole_a, bend, dir_a):
                continue
            if not _is_forward(hole_b, bend, dir_b):
                continue

            path = _orth_line(hole_a, bend) + _orth_line(bend, hole_b)[1:]
            ok = True
            for r, c in path:
                if int(g[r, c]) != BG:
                    ok = False
                    break
            if ok:
                valid_paths.append(path)

    assert len(valid_paths) == 1
    return valid_paths[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color sprites, infer their exit directions from the missing cell,
    and draw the missing L-shaped bridge between each unresolved pair.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    colors = [int(v) for v in np.unique(g) if int(v) != BG]
    for color in colors:
        comps = _components(g == int(color))
        assert 1 <= len(comps) <= 2
        if len(comps) == 1:
            continue

        sprite_a = _parse_sprite(g, color, comps[0])
        sprite_b = _parse_sprite(g, color, comps[1])
        for r, c in _candidate_connection(g, color, sprite_a, sprite_b):
            out[r, c] = int(color)

    return out


def _base_mask() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(3, 4)
    hole_c = random.randint(1, w - 2)
    mask = np.ones((h, w), dtype=bool)
    mask[0, hole_c] = False
    return mask


def _rotated_mask(base_mask: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    return np.rot90(base_mask, DIR_TO_ROT[direction])


def _placement_from_hole(mask: np.ndarray, hole: tuple[int, int]) -> tuple[int, int]:
    hole_r, hole_c = _hole_local(mask)
    return int(hole[0] - hole_r), int(hole[1] - hole_c)


def _sprite_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells = np.argwhere(mask)
    return {(int(top + r), int(left + c)) for r, c in cells}


def _candidate_holes_for_bend(
    shape: tuple[int, int],
    bend: tuple[int, int],
    mask: np.ndarray,
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = mask.shape
    hole_r, hole_c = _hole_local(mask)

    if direction in HORIZONTAL:
        row = bend[0]
        top = row - hole_r
        if top < 0 or top + ph > h:
            return []

        min_col = hole_c
        max_col = w - pw + hole_c
        if direction == RIGHT:
            stop = min(max_col, bend[1] - MIN_LEG)
            return [(int(row), int(col)) for col in range(int(min_col), int(stop) + 1)]

        start = max(min_col, bend[1] + MIN_LEG)
        return [(int(row), int(col)) for col in range(int(start), int(max_col) + 1)]

    col = bend[1]
    left = col - hole_c
    if left < 0 or left + pw > w:
        return []

    min_row = hole_r
    max_row = h - ph + hole_r
    if direction == DOWN:
        stop = min(max_row, bend[0] - MIN_LEG)
        return [(int(row), int(col)) for row in range(int(min_row), int(stop) + 1)]

    start = max(min_row, bend[0] + MIN_LEG)
    return [(int(row), int(col)) for row in range(int(start), int(max_row) + 1)]


def _try_place_pair(
    shape: tuple[int, int],
    color: int,
    demo: bool,
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(PAIR_PLACE_RETRIES):
        base = _base_mask()
        horiz_dir = random.choice([LEFT, RIGHT])
        vert_dir = random.choice([UP, DOWN])
        horiz_mask = _rotated_mask(base, horiz_dir)
        vert_mask = _rotated_mask(base, vert_dir)

        bend = (random.randrange(h), random.randrange(w))
        if bend in blocked:
            continue

        horiz_holes = _candidate_holes_for_bend(shape, bend, horiz_mask, horiz_dir)
        vert_holes = _candidate_holes_for_bend(shape, bend, vert_mask, vert_dir)
        if not horiz_holes or not vert_holes:
            continue

        random.shuffle(horiz_holes)
        random.shuffle(vert_holes)
        horiz_holes = horiz_holes[:12]
        vert_holes = vert_holes[:12]

        for horiz_hole in horiz_holes:
            horiz_top, horiz_left = _placement_from_hole(horiz_mask, horiz_hole)
            horiz_cells = _sprite_cells(horiz_mask, horiz_top, horiz_left)
            if horiz_cells & blocked:
                continue

            for vert_hole in vert_holes:
                vert_top, vert_left = _placement_from_hole(vert_mask, vert_hole)
                vert_cells = _sprite_cells(vert_mask, vert_top, vert_left)
                if vert_cells & blocked or horiz_cells & vert_cells:
                    continue

                path = _orth_line(horiz_hole, bend) + _orth_line(bend, vert_hole)[1:]
                path_set = set(path)
                if len(path_set) != len(path):
                    continue
                if path_set & blocked:
                    continue
                if (path_set - {horiz_hole, vert_hole}) & (horiz_cells | vert_cells):
                    continue

                return {
                    "color": int(color),
                    "demo": bool(demo),
                    "path": path,
                    "sprites": [
                        {
                            "mask": horiz_mask,
                            "top": int(horiz_top),
                            "left": int(horiz_left),
                        },
                        {
                            "mask": vert_mask,
                            "top": int(vert_top),
                            "left": int(vert_left),
                        },
                    ],
                }

    return None


def _mark_blocked(spec: dict[str, object], blocked: set[tuple[int, int]]) -> None:
    for sprite in spec["sprites"]:
        blocked.update(_sprite_cells(sprite["mask"], int(sprite["top"]), int(sprite["left"])))
    blocked.update(tuple(cell) for cell in spec["path"])


def _render_case(shape: tuple[int, int], specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)

    for spec in specs:
        color = int(spec["color"])
        for sprite in spec["sprites"]:
            mask = np.asarray(sprite["mask"], dtype=bool)
            top = int(sprite["top"])
            left = int(sprite["left"])
            view = inp[top : top + mask.shape[0], left : left + mask.shape[1]]
            assert np.all(view[mask] == BG)
            view[mask] = color
        if spec["demo"]:
            for r, c in spec["path"]:
                inp[r, c] = color

    expected = inp.copy()
    for spec in specs:
        if spec["demo"]:
            continue
        for r, c in spec["path"]:
            expected[r, c] = int(spec["color"])

    return inp, expected


def _valid_generated_case(inp: np.ndarray, expected: np.ndarray, specs: list[dict[str, object]]) -> bool:
    colors = [int(v) for v in np.unique(inp) if int(v) != BG]
    if len(colors) != len(specs):
        return False

    demo_colors = 0
    for spec in specs:
        color = int(spec["color"])
        comps = _components(inp == color)
        if spec["demo"]:
            demo_colors += 1
            if len(comps) != 1:
                return False
            continue
        if len(comps) != 2:
            return False
        try:
            _parse_sprite(inp, color, comps[0])
            _parse_sprite(inp, color, comps[1])
        except AssertionError:
            return False
    if demo_colors != 1:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _generate_case() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    side_min = GRID_MIN_BY_PAIR_COUNT[pair_count]
    shape = (random.randint(side_min, GRID_MAX), random.randint(side_min, GRID_MAX))
    colors = random.sample(COLOR_POOL, pair_count)
    demo_index = random.randrange(pair_count)

    blocked: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []
    for idx, color in enumerate(colors):
        spec = _try_place_pair(shape, int(color), idx == demo_index, blocked)
        if spec is None:
            return None
        _mark_blocked(spec, blocked)
        specs.append(spec)

    inp, expected = _render_case(shape, specs)
    if not _valid_generated_case(inp, expected, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-path ARC input")
