import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

TOWER_COLORS = [BLUE, RED, GREEN, ORANGE]
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MIX_MAP = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, GREEN)): YELLOW,
    frozenset((BLUE, GREEN)): AZURE,
}
GEN_RETRIES = 400
EXTRA_TOWER_TRIES = 80


def _compute_coverages(grid: np.ndarray) -> tuple[np.ndarray, list[list[set[int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    ray_counts = np.zeros((h, w), dtype=int)
    ray_colors = [[set() for _ in range(w)] for _ in range(h)]

    towers = np.argwhere((g != BLACK) & (g != GREY))
    for r, c in towers:
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        launch_dirs = []
        for dr, dc in DIRECTIONS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == GREY:
                launch_dirs.append((dr, dc))

        for dr, dc in launch_dirs:
            rr = r + 2 * dr
            cc = c + 2 * dc
            while 0 <= rr < h and 0 <= cc < w:
                value = int(g[rr, cc])
                if value != BLACK and value != GREY:
                    break
                if value == BLACK:
                    ray_counts[rr, cc] += 1
                    ray_colors[rr][cc].add(color)
                rr += dr
                cc += dc

    return ray_counts, ray_colors


def _resolve_cell_color(ray_count: int, colors: set[int]) -> int | None:
    if ray_count <= 0 or not colors:
        return None
    if len(colors) == 1:
        return next(iter(colors))
    if ray_count == 2 and len(colors) == 2:
        return MIX_MAP.get(frozenset(colors), MAROON)
    return MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire colored rays from single-pixel signal towers according to adjacent grey direction markers and mix intersections by fixed color rules.

    Input structure:
    - The background is black.
    - Signal towers are isolated single pixels in non-black colors other than grey and maroon.
    - Grey pixels may appear immediately above, below, left, or right of a tower; each such grey pixel is a direction marker telling that tower to emit in that cardinal direction.

    Rule:
    - For every tower, read its launch directions from the grey markers in its four-neighborhood.
    - Each marked direction emits a straight ray starting one cell beyond the grey marker; the ray continues until the grid edge or until the next tower, and it only covers background cells while leaving towers and grey markers unchanged.
    - After all rays are collected, color singly covered cells with that ray color.
    - If exactly two different colors meet, mix red+blue to magenta, red+green to yellow, and blue+green to azure; all other two-color meetings and every three-or-more-ray crossing become maroon.

    Key insight:
    - Grey neighbors are not decorative: they encode which directions each tower launches into, and the visible output depends on combining all launched rays first and only then resolving overlaps with a fixed color-mixing table.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    ray_counts, ray_colors = _compute_coverages(g)

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) != BLACK:
                continue
            color = _resolve_cell_color(int(ray_counts[r, c]), ray_colors[r][c])
            if color is not None:
                out[r, c] = color

    return out


def _path_to_cross(pos: tuple[int, int], direction: tuple[int, int], cross: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = pos
    dr, dc = direction
    rr = r + 2 * dr
    cc = c + 2 * dc
    cells: set[tuple[int, int]] = set()
    while True:
        cells.add((rr, cc))
        if (rr, cc) == cross:
            return cells
        rr += dr
        cc += dc


def _tower_exclusion(pos: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = pos
    cells = {(r, c)}
    for dr, dc in DIRECTIONS:
        cells.add((r + dr, c + dc))
        cells.add((r + 2 * dr, c + 2 * dc))
    return cells


def _can_place_tower(
    h: int,
    w: int,
    pos: tuple[int, int],
    dirs: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    tower_blocked: set[tuple[int, int]],
) -> bool:
    r, c = pos
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if pos in occupied or pos in reserved or pos in tower_blocked:
        return False
    for dr, dc in dirs:
        marker = (r + dr, c + dc)
        if marker in occupied or marker in reserved:
            return False
        if not (0 <= marker[0] < h and 0 <= marker[1] < w):
            return False
    return True


def _commit_tower(
    towers: list[dict[str, object]],
    pos: tuple[int, int],
    color: int,
    dirs: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    tower_blocked: set[tuple[int, int]],
) -> None:
    towers.append({"pos": pos, "color": color, "dirs": set(dirs)})
    occupied.add(pos)
    for dr, dc in dirs:
        occupied.add((pos[0] + dr, pos[1] + dc))
    tower_blocked.update(_tower_exclusion(pos))


def _build_pair_core(
    rng: random.Random,
    h: int,
    w: int,
    colors: tuple[int, int],
) -> tuple[list[dict[str, object]], set[tuple[int, int]], tuple[int, int]] | None:
    cross = (rng.randint(5, h - 6), rng.randint(5, w - 6))
    h_side = rng.choice(["left", "right"])
    v_side = rng.choice(["up", "down"])
    hdist = rng.randint(2, 4)
    vdist = rng.randint(2, 4)

    if h_side == "left":
        h_pos = (cross[0], cross[1] - hdist)
        h_dir = (0, 1)
    else:
        h_pos = (cross[0], cross[1] + hdist)
        h_dir = (0, -1)

    if v_side == "up":
        v_pos = (cross[0] - vdist, cross[1])
        v_dir = (1, 0)
    else:
        v_pos = (cross[0] + vdist, cross[1])
        v_dir = (-1, 0)

    if h_pos == v_pos:
        return None

    oriented = list(colors)
    rng.shuffle(oriented)
    reserved = _path_to_cross(h_pos, h_dir, cross) | _path_to_cross(v_pos, v_dir, cross)
    towers: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()
    tower_blocked: set[tuple[int, int]] = set()

    if not _can_place_tower(h, w, h_pos, {h_dir}, occupied, reserved, tower_blocked):
        return None
    _commit_tower(towers, h_pos, oriented[0], {h_dir}, occupied, tower_blocked)

    if not _can_place_tower(h, w, v_pos, {v_dir}, occupied, reserved, tower_blocked):
        return None
    _commit_tower(towers, v_pos, oriented[1], {v_dir}, occupied, tower_blocked)
    return towers, reserved, cross


def _build_triple_core(
    rng: random.Random,
    h: int,
    w: int,
) -> tuple[list[dict[str, object]], set[tuple[int, int]], tuple[int, int]] | None:
    cross = (rng.randint(5, h - 6), rng.randint(5, w - 6))
    layout = rng.choice(["horizontal_pair", "vertical_pair"])
    use_fourth = rng.random() < 0.35

    towers: list[tuple[tuple[int, int], tuple[int, int]]] = []
    reserved: set[tuple[int, int]] = set()

    if layout == "horizontal_pair":
        left = (cross[0], cross[1] - rng.randint(2, 4))
        right = (cross[0], cross[1] + rng.randint(2, 4))
        towers.append((left, (0, 1)))
        towers.append((right, (0, -1)))
        if rng.choice([True, False]):
            vertical = (cross[0] - rng.randint(2, 4), cross[1])
            towers.append((vertical, (1, 0)))
            if use_fourth:
                towers.append(((cross[0] + rng.randint(2, 4), cross[1]), (-1, 0)))
        else:
            vertical = (cross[0] + rng.randint(2, 4), cross[1])
            towers.append((vertical, (-1, 0)))
            if use_fourth:
                towers.append(((cross[0] - rng.randint(2, 4), cross[1]), (1, 0)))
    else:
        up = (cross[0] - rng.randint(2, 4), cross[1])
        down = (cross[0] + rng.randint(2, 4), cross[1])
        towers.append((up, (1, 0)))
        towers.append((down, (-1, 0)))
        if rng.choice([True, False]):
            horizontal = (cross[0], cross[1] - rng.randint(2, 4))
            towers.append((horizontal, (0, 1)))
            if use_fourth:
                towers.append(((cross[0], cross[1] + rng.randint(2, 4)), (0, -1)))
        else:
            horizontal = (cross[0], cross[1] + rng.randint(2, 4))
            towers.append((horizontal, (0, -1)))
            if use_fourth:
                towers.append(((cross[0], cross[1] - rng.randint(2, 4)), (0, 1)))

    if len({pos for pos, _ in towers}) != len(towers):
        return None

    for pos, direction in towers:
        reserved |= _path_to_cross(pos, direction, cross)

    colors = rng.sample(TOWER_COLORS, len(towers))
    placed: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()
    tower_blocked: set[tuple[int, int]] = set()
    for (pos, direction), color in zip(towers, colors):
        if not _can_place_tower(h, w, pos, {direction}, occupied, reserved, tower_blocked):
            return None
        _commit_tower(placed, pos, color, {direction}, occupied, tower_blocked)
    return placed, reserved, cross


def _augment_core_towers(
    rng: random.Random,
    h: int,
    w: int,
    towers: list[dict[str, object]],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> None:
    for tower in towers:
        if rng.random() < 0.25:
            continue
        remaining = [d for d in DIRECTIONS if d not in tower["dirs"]]
        rng.shuffle(remaining)
        extra_budget = rng.randint(0, min(2, len(remaining)))
        for dr, dc in remaining:
            if extra_budget <= 0:
                break
            marker = (tower["pos"][0] + dr, tower["pos"][1] + dc)
            if not (0 <= marker[0] < h and 0 <= marker[1] < w):
                continue
            if marker in occupied or marker in reserved:
                continue
            tower["dirs"].add((dr, dc))
            occupied.add(marker)
            extra_budget -= 1


def _add_extra_towers(
    rng: random.Random,
    h: int,
    w: int,
    towers: list[dict[str, object]],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    tower_blocked: set[tuple[int, int]],
    cross: tuple[int, int],
    palette: list[int],
) -> None:
    extra_count = rng.randint(0, 3)
    for _ in range(extra_count):
        for _attempt in range(EXTRA_TOWER_TRIES):
            pos = (rng.randint(1, h - 2), rng.randint(1, w - 2))
            if pos[0] == cross[0] or pos[1] == cross[1]:
                continue
            dir_count = rng.randint(1, 4)
            dirs = set(rng.sample(DIRECTIONS, dir_count))
            if not _can_place_tower(h, w, pos, dirs, occupied, reserved, tower_blocked):
                continue
            color = rng.choice(palette)
            _commit_tower(towers, pos, color, dirs, occupied, tower_blocked)
            break


def _render_input(h: int, w: int, towers: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for tower in towers:
        r, c = tower["pos"]
        grid[r, c] = int(tower["color"])
        for dr, dc in tower["dirs"]:
            grid[r + dr, c + dc] = GREY
    return grid


def _scenario_targets() -> list[tuple[str, object]]:
    return [
        ("pair", (RED, BLUE)),
        ("pair", (RED, GREEN)),
        ("pair", (BLUE, GREEN)),
        ("other_pair", None),
        ("triple", None),
    ]


def _choose_other_pair(rng: random.Random) -> tuple[int, int]:
    return rng.choice([(ORANGE, RED), (ORANGE, BLUE), (ORANGE, GREEN)])


def _has_target_branch(grid: np.ndarray, target_kind: str, target_data: object) -> bool:
    ray_counts, ray_colors = _compute_coverages(grid)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != BLACK or int(ray_counts[r, c]) == 0:
                continue
            colors = ray_colors[r][c]
            count = int(ray_counts[r, c])
            if target_kind == "pair" and count == 2 and colors == set(target_data):
                return True
            if target_kind == "other_pair":
                if count == 2 and len(colors) == 2 and frozenset(colors) not in MIX_MAP:
                    return True
            if target_kind == "triple" and count >= 3:
                return True
    return False


def _nondegenerate(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape:
        return False
    if np.array_equal(inp, out):
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if int(np.count_nonzero(inp == GREY)) < 2:
        return False
    if int(np.count_nonzero((inp != BLACK) & (inp != GREY))) < 2:
        return False
    if int(np.count_nonzero(out != inp)) < 8:
        return False
    if int(np.sum(inp == BLACK)) * 2 <= int(inp.size):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 24)
    w = rng.randint(14, 24)
    target_kind, target_data = rng.choice(_scenario_targets())
    if target_kind == "pair":
        built = _build_pair_core(rng, h, w, target_data)
    elif target_kind == "other_pair":
        built = _build_pair_core(rng, h, w, _choose_other_pair(rng))
    else:
        built = _build_triple_core(rng, h, w)

    if built is None:
        return None
    towers, reserved, cross = built

    occupied = set()
    tower_blocked = set()
    for tower in towers:
        occupied.add(tower["pos"])
        for dr, dc in tower["dirs"]:
            occupied.add((tower["pos"][0] + dr, tower["pos"][1] + dc))
        tower_blocked.update(_tower_exclusion(tower["pos"]))

    _augment_core_towers(rng, h, w, towers, occupied, reserved)
    extra_palette = TOWER_COLORS if target_kind == "other_pair" else [BLUE, RED, GREEN]
    _add_extra_towers(rng, h, w, towers, occupied, reserved, tower_blocked, cross, extra_palette)

    inp = _render_input(h, w, towers)
    out = main(inp)
    if not _has_target_branch(inp, target_kind, target_data):
        return None
    if not _nondegenerate(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower sample")
