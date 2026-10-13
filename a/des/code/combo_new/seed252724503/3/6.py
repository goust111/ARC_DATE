import numpy as np

BG = 0
YELLOW = 4
GRAY = 5
TOWER_COLORS = (1, 2, 3, 6, 7, 8, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

MIRROR_PARTNERS = {
    "slash": ((-1, 1), (1, -1)),
    "backslash": ((-1, -1), (1, 1)),
}

GENERATE_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _reflect(direction: tuple[int, int], mirror_kind: str) -> tuple[int, int]:
    dr, dc = direction
    if mirror_kind == "backslash":
        return (dc, dr)
    assert mirror_kind == "slash"
    return (-dc, -dr)


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = cr + dr
                        nc = cc + dc
                        if not _in_bounds(mask.shape, nr, nc):
                            continue
                        if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                            continue
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_towers(grid: np.ndarray) -> list[dict[str, object]]:
    towers: list[dict[str, object]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in (BG, YELLOW, GRAY):
                continue
            dirs = []
            for dr, dc in DIRS:
                nr = r + dr
                nc = c + dc
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GRAY:
                    dirs.append((dr, dc))
            assert len(dirs) == 1, "each tower must have exactly one gray direction marker"
            towers.append(
                {
                    "pos": (r, c),
                    "color": color,
                    "direction": dirs[0],
                }
            )
    assert towers, "expected at least one tower"
    return towers


def _detect_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mirror_map: dict[tuple[int, int], str] = {}
    for comp in _components8(grid == YELLOW):
        assert len(comp) == 2, "each mirror must be an isolated diagonal segment of length 2"
        (r0, c0), (r1, c1) = comp
        dr = r1 - r0
        dc = c1 - c0
        assert abs(dr) == 1 and abs(dc) == 1, "mirror pixels must be diagonally adjacent"
        kind = "backslash" if dr == dc else "slash"
        for cell in comp:
            mirror_map[cell] = kind
    return mirror_map


def _trace_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    tower_positions: set[tuple[int, int]],
    mirror_map: dict[tuple[int, int], str],
) -> tuple[tuple[int, int], str, tuple[str, ...]]:
    r, c = start
    dr, dc = direction
    last_bg: tuple[int, int] | None = None
    reflections: list[str] = []
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "ray entered a reflection loop"
        seen_states.add(state)

        nr = r + dr
        nc = c + dc
        if not _in_bounds(grid.shape, nr, nc):
            assert last_bg is not None, "ray must have at least one background cell before the boundary"
            return (last_bg, "boundary", tuple(reflections))

        if (nr, nc) in tower_positions:
            assert (nr, nc) != start, "ray may not loop back into its own tower"
            assert last_bg is not None, "ray must have a background cell before the blocking tower"
            return (last_bg, "tower", tuple(reflections))

        cell = int(grid[nr, nc])
        if cell == BG:
            last_bg = (nr, nc)
        elif cell == GRAY:
            pass
        elif cell == YELLOW:
            kind = mirror_map[(nr, nc)]
            reflections.append(kind)
            dr, dc = _reflect((dr, dc), kind)
        else:
            raise AssertionError("unexpected colored object on ray path")

        r, c = nr, nc


def _scene_info(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], str], list[dict[str, object]]]:
    towers = _detect_towers(grid)
    tower_positions = {tower["pos"] for tower in towers}
    mirror_map = _detect_mirrors(grid)
    rays: list[dict[str, object]] = []
    for tower in towers:
        endpoint, stop_reason, reflections = _trace_ray(
            grid,
            tower["pos"],
            tower["direction"],
            tower_positions,
            mirror_map,
        )
        rays.append(
            {
                "start": tower["pos"],
                "color": tower["color"],
                "direction": tower["direction"],
                "endpoint": endpoint,
                "stop_reason": stop_reason,
                "reflections": reflections,
            }
        )
    return towers, mirror_map, rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each colored signal tower along its gray-coded direction, reflecting on yellow diagonal mirrors and marking its landing cell.

    Input structure:
    - The background is black.
    - Each signal tower is a single non-black, non-yellow, non-gray pixel.
    - Every tower has exactly one 4-neighbor gray pixel whose relative position encodes the emission direction.
    - Yellow pixels appear only as isolated diagonal 2-pixel mirror segments, each oriented as "\" or "/".

    Rule:
    - Keep the original towers and mirrors in place, but clear every gray direction marker to black.
    - From each tower, cast a ray in the direction indicated by its adjacent gray pixel; the ray leaves no trail through background cells.
    - When the ray enters a yellow mirror pixel, reflect it according to that mirror's diagonal orientation.
    - Stop when the next step would leave the grid or hit another tower, and color the last background cell before that stop with the tower's color.

    Key insight:
    - The gray pixels encode direction only; the actual transformation is a ray-tracing problem with orientation-dependent reflections on short diagonal mirror markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    out[out == GRAY] = BG

    _towers, _mirror_map, rays = _scene_info(g)
    for ray in rays:
        er, ec = ray["endpoint"]
        current = int(out[er, ec])
        assert current in (BG, int(ray["color"])), "ray endpoints must not collide with other objects"
        out[er, ec] = int(ray["color"])

    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    assert lo <= hi
    return int(rng.integers(lo, hi + 1))


def _distance_to_boundary(box: tuple[int, int, int, int], cell: tuple[int, int], direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = box
    r, c = cell
    if direction == UP:
        return r - r0
    if direction == DOWN:
        return r1 - r
    if direction == LEFT:
        return c - c0
    assert direction == RIGHT
    return c1 - c


def _outer_dirs(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = box
    h, w = shape
    vdir = UP if r0 == 0 else DOWN
    hdir = LEFT if c0 == 0 else RIGHT
    assert (r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)
    return (vdir, hdir)


def _paint_tower(grid: np.ndarray, tower: tuple[int, int], direction: tuple[int, int], color: int) -> None:
    tr, tc = tower
    gr = tr + direction[0]
    gc = tc + direction[1]
    assert int(grid[tr, tc]) == BG
    assert int(grid[gr, gc]) == BG
    grid[tr, tc] = int(color)
    grid[gr, gc] = GRAY


def _enumerate_turn_placements(
    box: tuple[int, int, int, int],
    mirror_kind: str,
    shape: tuple[int, int],
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = box
    outer = set(_outer_dirs(box, shape))
    placements: list[dict[str, object]] = []

    for start_dir in DIRS:
        final_dir = _reflect(start_dir, mirror_kind)
        if final_dir not in outer:
            continue
        for mr in range(r0, r1 + 1):
            for mc in range(c0, c1 + 1):
                mirror = (mr, mc)
                if _distance_to_boundary(box, mirror, final_dir) < 2:
                    continue
                max_steps = _distance_to_boundary(box, mirror, (-start_dir[0], -start_dir[1]))
                if max_steps < 3:
                    continue
                for steps in range(3, max_steps + 1):
                    tower = (mr - start_dir[0] * steps, mc - start_dir[1] * steps)
                    gray = (tower[0] + start_dir[0], tower[1] + start_dir[1])
                    if not (_in_bounds(shape, tower[0], tower[1]) and _in_bounds(shape, gray[0], gray[1])):
                        continue
                    for pr_off, pc_off in MIRROR_PARTNERS[mirror_kind]:
                        partner = (mr + pr_off, mc + pc_off)
                        if not _in_bounds(shape, partner[0], partner[1]):
                            continue
                        if not (r0 <= partner[0] <= r1 and c0 <= partner[1] <= c1):
                            continue
                        cells = {tower, gray, mirror, partner}
                        if len(cells) != 4:
                            continue
                        placements.append(
                            {
                                "tower": tower,
                                "direction": start_dir,
                                "mirror": (mirror, partner),
                            }
                        )
    return placements


def _place_turn_single(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    color: int,
    mirror_kind: str,
    rng: np.random.Generator,
) -> None:
    placements = _enumerate_turn_placements(box, mirror_kind, grid.shape)
    assert placements, "expected at least one valid turn placement"
    placement = placements[_randint(rng, 0, len(placements) - 1)]
    tower = placement["tower"]
    direction = placement["direction"]
    mirror_a, mirror_b = placement["mirror"]

    _paint_tower(grid, tower, direction, color)
    assert int(grid[mirror_a]) == BG and int(grid[mirror_b]) == BG
    grid[mirror_a] = YELLOW
    grid[mirror_b] = YELLOW


def _place_straight_single(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    color: int,
    rng: np.random.Generator,
) -> None:
    r0, c0, r1, c1 = box
    outer_dirs = _outer_dirs(box, grid.shape)
    direction = outer_dirs[_randint(rng, 0, 1)]

    if direction == UP:
        tower = (_randint(rng, r0 + 3, r1), _randint(rng, c0 + 1, c1 - 1))
    elif direction == DOWN:
        tower = (_randint(rng, r0, r1 - 3), _randint(rng, c0 + 1, c1 - 1))
    elif direction == LEFT:
        tower = (_randint(rng, r0 + 1, r1 - 1), _randint(rng, c0 + 3, c1))
    else:
        assert direction == RIGHT
        tower = (_randint(rng, r0 + 1, r1 - 1), _randint(rng, c0, c1 - 3))

    _paint_tower(grid, tower, direction, color)


def _enumerate_blocker_pair_placements(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = box
    vdir, _hdir = _outer_dirs(box, shape)
    placements: list[dict[str, object]] = []

    for br in range(r0 + 1, r1):
        if _distance_to_boundary(box, (br, c0), vdir) < 3:
            continue
        for bc in range(c0 + 1, c1):
            blocker = (br, bc)
            blocker_gray = (br + vdir[0], bc + vdir[1])
            if not _in_bounds(shape, blocker_gray[0], blocker_gray[1]):
                continue

            left_space = bc - c0
            right_space = c1 - bc
            if left_space >= 4:
                for dist in range(4, left_space + 1):
                    emitter = (br, bc - dist)
                    placements.append(
                        {
                            "emitter": emitter,
                            "emitter_dir": RIGHT,
                            "blocker": blocker,
                            "blocker_dir": vdir,
                        }
                    )
            if right_space >= 4:
                for dist in range(4, right_space + 1):
                    emitter = (br, bc + dist)
                    placements.append(
                        {
                            "emitter": emitter,
                            "emitter_dir": LEFT,
                            "blocker": blocker,
                            "blocker_dir": vdir,
                        }
                    )
    return placements


def _place_blocker_pair(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    emitter_color: int,
    blocker_color: int,
    rng: np.random.Generator,
) -> None:
    placements = _enumerate_blocker_pair_placements(box, grid.shape)
    assert placements, "expected at least one blocker-pair placement"
    placement = placements[_randint(rng, 0, len(placements) - 1)]

    _paint_tower(grid, placement["emitter"], placement["emitter_dir"], emitter_color)
    _paint_tower(grid, placement["blocker"], placement["blocker_dir"], blocker_color)


def _split_boxes(h: int, w: int, rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    split_r = _randint(rng, 8, h - 9)
    split_c = _randint(rng, 8, w - 9)
    return [
        (0, 0, split_r - 2, split_c - 2),
        (0, split_c + 2, split_r - 2, w - 1),
        (split_r + 2, 0, h - 1, split_c - 2),
        (split_r + 2, split_c + 2, h - 1, w - 1),
    ]


def _build_scene(rng: np.random.Generator) -> np.ndarray:
    h = _randint(rng, 18, 26)
    w = _randint(rng, 18, 26)
    grid = np.full((h, w), BG, dtype=int)
    boxes = _split_boxes(h, w, rng)

    roles = ["slash_single", "backslash_single", "blocker_pair"]
    roles.append("blocker_pair" if bool(rng.integers(0, 2)) else "straight_single")
    order = rng.permutation(len(boxes))
    roles = [roles[i] for i in rng.permutation(len(roles))]

    tower_count = sum(2 if role == "blocker_pair" else 1 for role in roles)
    colors = list(rng.choice(np.array(TOWER_COLORS, dtype=int), size=tower_count, replace=False))
    color_index = 0

    for box_index, role in zip(order, roles):
        box = boxes[int(box_index)]
        if role == "slash_single":
            _place_turn_single(grid, box, int(colors[color_index]), "slash", rng)
            color_index += 1
        elif role == "backslash_single":
            _place_turn_single(grid, box, int(colors[color_index]), "backslash", rng)
            color_index += 1
        elif role == "straight_single":
            _place_straight_single(grid, box, int(colors[color_index]), rng)
            color_index += 1
        else:
            assert role == "blocker_pair"
            _place_blocker_pair(grid, box, int(colors[color_index]), int(colors[color_index + 1]), rng)
            color_index += 2

    return grid


def _valid_scene(grid: np.ndarray) -> bool:
    try:
        towers, mirror_map, rays = _scene_info(grid)
    except AssertionError:
        return False

    if len(towers) < 4:
        return False
    if len({tuple(ray["endpoint"]) for ray in rays}) != len(rays):
        return False
    if not any(ray["stop_reason"] == "boundary" for ray in rays):
        return False
    if not any(ray["stop_reason"] == "tower" for ray in rays):
        return False
    if not any("slash" in ray["reflections"] for ray in rays):
        return False
    if not any("backslash" in ray["reflections"] for ray in rays):
        return False
    if any(int(grid[tuple(ray["endpoint"])]) != BG for ray in rays):
        return False
    if any(int(grid[pos]) != YELLOW for pos in mirror_map):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    return out.shape == grid.shape


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_ATTEMPTS):
        grid = _build_scene(rng)
        if _valid_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid signal-tower mirror scene")
