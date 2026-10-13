import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500
GRID_MIN = 15
GRID_MAX = 27

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
REFLECT = {
    "/": {
        DIRS["right"]: DIRS["up"],
        DIRS["left"]: DIRS["down"],
        DIRS["up"]: DIRS["right"],
        DIRS["down"]: DIRS["left"],
    },
    "\\": {
        DIRS["right"]: DIRS["down"],
        DIRS["left"]: DIRS["up"],
        DIRS["up"]: DIRS["left"],
        DIRS["down"]: DIRS["right"],
    },
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _tower_geometry(top: int, left: int, direction: str, offset: int) -> dict:
    core = {
        (top, left),
        (top, left + 1),
        (top + 1, left),
        (top + 1, left + 1),
    }
    if direction == "up":
        nub = (top - 1, left + offset)
    elif direction == "down":
        nub = (top + 2, left + offset)
    elif direction == "left":
        nub = (top + offset, left - 1)
    else:
        assert direction == "right"
        nub = (top + offset, left + 2)

    cells = tuple(sorted(core | {nub}))
    return {
        "core": tuple(sorted(core)),
        "nub": nub,
        "direction": DIRS[direction],
        "cells": cells,
    }


def _parse_tower(coords: list[tuple[int, int]]) -> dict | None:
    if len(coords) != 5:
        return None
    pts = set(coords)
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    for top in range(min(rows) - 1, max(rows) + 1):
        for left in range(min(cols) - 1, max(cols) + 1):
            core = {
                (top, left),
                (top, left + 1),
                (top + 1, left),
                (top + 1, left + 1),
            }
            if not core.issubset(pts):
                continue
            extra = pts - core
            if len(extra) != 1:
                continue
            nub = next(iter(extra))
            nr, nc = nub
            if nr == top - 1 and nc in (left, left + 1):
                direction = DIRS["up"]
            elif nr == top + 2 and nc in (left, left + 1):
                direction = DIRS["down"]
            elif nc == left - 1 and nr in (top, top + 1):
                direction = DIRS["left"]
            elif nc == left + 2 and nr in (top, top + 1):
                direction = DIRS["right"]
            else:
                continue
            return {
                "core": tuple(sorted(core)),
                "nub": nub,
                "direction": direction,
                "cells": tuple(sorted(pts)),
            }
    return None


def _find_towers(grid: np.ndarray, mix_color: int) -> list[dict]:
    colors = sorted(set(map(int, grid.ravel())) - {BG, GREY, mix_color})
    towers: list[dict] = []
    for color in colors:
        coords = [tuple(map(int, cell)) for cell in np.argwhere(grid == color)]
        tower = _parse_tower(coords)
        assert tower is not None, f"color {color} is not a valid tower"
        tower["color"] = int(color)
        towers.append(tower)
    assert towers, "expected at least one tower"
    return towers


def _mirror_orientations(grid: np.ndarray) -> dict[tuple[int, int], str]:
    grey_cells = {tuple(map(int, cell)) for cell in np.argwhere(grid == GREY)}
    orientations: dict[tuple[int, int], str] = {}
    for r, c in grey_cells:
        slash = (r - 1, c + 1) in grey_cells or (r + 1, c - 1) in grey_cells
        backslash = (r - 1, c - 1) in grey_cells or (r + 1, c + 1) in grey_cells
        assert slash != backslash, "grey pixels must form clean diagonal mirrors"
        orientations[(r, c)] = "/" if slash else "\\"
    return orientations


def _trace_rays(
    grid: np.ndarray,
    towers: list[dict],
    mirror_map: dict[tuple[int, int], str],
) -> tuple[dict[int, set[tuple[int, int]]], int]:
    core_cells = {cell for tower in towers for cell in tower["core"]}
    paths: dict[int, set[tuple[int, int]]] = {}
    reflection_count = 0

    for tower in towers:
        color = int(tower["color"])
        r, c = tower["nub"]
        dr, dc = tower["direction"]
        seen: set[tuple[int, int, int, int]] = set()
        visited: set[tuple[int, int]] = set()

        while True:
            state = (r, c, dr, dc)
            assert state not in seen, "ray path must not cycle"
            seen.add(state)

            nr, nc = int(r + dr), int(c + dc)
            if not _in_bounds(grid.shape, nr, nc):
                break
            if (nr, nc) in core_cells:
                break
            if (nr, nc) in mirror_map:
                dr, dc = REFLECT[mirror_map[(nr, nc)]][(dr, dc)]
                r, c = nr, nc
                reflection_count += 1
                continue

            if int(grid[nr, nc]) == BG:
                visited.add((nr, nc))
            r, c = nr, nc

        paths[color] = visited

    return paths, reflection_count


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    mix_color = int(g[-1, -1])
    assert mix_color not in (BG, GREY), "bottom-right pixel must be the mix indicator"
    assert int((g == mix_color).sum()) == 1, "mix color must be unique"

    towers = _find_towers(g, mix_color)
    mirror_map = _mirror_orientations(g)
    paths, reflections = _trace_rays(g, towers, mirror_map)

    passes: dict[tuple[int, int], set[int]] = {}
    for color, cells in paths.items():
        for cell in cells:
            passes.setdefault(cell, set()).add(int(color))

    crossings = sum(1 for colors in passes.values() if len(colors) >= 2)
    return {
        "mix_color": mix_color,
        "towers": towers,
        "mirror_map": mirror_map,
        "paths": paths,
        "passes": passes,
        "reflections": int(reflections),
        "crossings": int(crossings),
        "path_lengths": {color: len(cells) for color, cells in paths.items()},
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire colored rays from tower arrows, reflect them on diagonal mirrors, and mix crossings with the bottom-right marker color.

    Input structure:
    - bg = black. The grid contains several colored signal towers, grey diagonal mirror segments, and a 1x1 mix-color indicator in the bottom-right corner.
    - Each signal tower is a same-color 2x2 solid block with one extra same-color nub on one side; the nub side encodes the emission direction.
    - Each mirror is a grey 45-degree segment, either "/" oriented or "\" oriented, with no other grey structures present.

    Rule:
    - For every tower, emit one ray starting beyond its nub and march cell by cell in that direction.
    - When the ray enters a grey mirror cell, update its direction by the mirror's reflection rule and keep tracing; mirror cells themselves stay grey.
    - A ray stops when its next step would leave the grid or enter any tower's 2x2 core; only background cells on the traveled path are recolored.
    - If background cell p is visited by exactly one tower color, paint p with that color; if p is visited by two or more different tower colors, paint p with the bottom-right mix-indicator color. Towers, mirrors, and the indicator remain unchanged.

    Key insight:
    - The nub position identifies each tower's launch direction, grey diagonal adjacency identifies mirror orientation, and the final output comes from simulating all ray paths before resolving multi-color overlaps into the dedicated mix color.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_grid(g)
    out = g.copy()
    mix_color = int(analysis["mix_color"])

    for (r, c), colors in analysis["passes"].items():
        if len(colors) == 1:
            out[r, c] = int(next(iter(colors)))
        else:
            out[r, c] = mix_color

    return out


def _transform_point(r: int, c: int, height: int, width: int, sym: int) -> tuple[int, int]:
    if sym == 0:
        return r, c
    if sym == 1:
        return r, width - 1 - c
    if sym == 2:
        return height - 1 - r, c
    assert sym == 3
    return height - 1 - r, width - 1 - c


def _transform_scene(scene: dict, height: int, width: int, sym: int) -> dict:
    towers = []
    for tower in scene["towers"]:
        core = tuple(sorted(_transform_point(r, c, height, width, sym) for r, c in tower["core"]))
        nub = _transform_point(tower["nub"][0], tower["nub"][1], height, width, sym)
        towers.append(
            {
                "core": core,
                "nub": nub,
                "cells": tuple(sorted(set(core) | {nub})),
            }
        )

    mirrors = []
    for seg in scene["mirrors"]:
        mirrors.append(tuple(sorted(_transform_point(r, c, height, width, sym) for r, c in seg)))

    return {"towers": towers, "mirrors": mirrors}


def _scene_occupancy(scene: dict) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for tower in scene["towers"]:
        occupied.update(tower["cells"])
    for seg in scene["mirrors"]:
        occupied.update(seg)
    return occupied


def _render_scene(height: int, width: int, scene: dict, tower_colors: list[int], mix_color: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    occupied: dict[tuple[int, int], int] = {}

    for seg in scene["mirrors"]:
        for cell in seg:
            occupied[cell] = occupied.get(cell, 0) + 1
            grid[cell] = GREY

    for tower, color in zip(scene["towers"], tower_colors):
        for cell in tower["cells"]:
            occupied[cell] = occupied.get(cell, 0) + 1
            grid[cell] = int(color)

    assert all(v == 1 for v in occupied.values()), "scene objects must not overlap"
    assert grid[height - 1, width - 1] == BG, "indicator cell must be free"
    grid[height - 1, width - 1] = int(mix_color)
    return grid


def _build_straight_cross(height: int, width: int, rng: np.random.Generator) -> dict:
    cross_r = int(rng.integers(6, height - 4))
    cross_c = int(rng.integers(6, width - 4))

    a_offset = int(rng.integers(0, 2))
    a_left = int(rng.integers(2, cross_c - 2))
    tower_a = _tower_geometry(cross_r - a_offset, a_left, "right", a_offset)

    b_offset = int(rng.integers(0, 2))
    b_top = int(rng.integers(2, cross_r - 2))
    tower_b = _tower_geometry(b_top, cross_c - b_offset, "down", b_offset)

    return {"towers": [tower_a, tower_b], "mirrors": []}


def _build_straight_disjoint(height: int, width: int, rng: np.random.Generator) -> dict | None:
    rows = list(range(5, height - 4))
    for _ in range(40):
        r1, r2 = sorted(int(x) for x in rng.choice(rows, size=2, replace=False))
        if r2 - r1 >= 3:
            break
    else:
        return None

    a_offset = int(rng.integers(0, 2))
    a_left = int(rng.integers(2, width - 5))
    tower_a = _tower_geometry(r1 - a_offset, a_left, "right", a_offset)

    b_offset = int(rng.integers(0, 2))
    b_left = int(rng.integers(2, width - 5))
    tower_b = _tower_geometry(r2 - b_offset, b_left, "right", b_offset)

    return {"towers": [tower_a, tower_b], "mirrors": []}


def _build_reflect_cross(height: int, width: int, rng: np.random.Generator) -> dict:
    length = int(rng.integers(2, 5))
    mirror_r = int(rng.integers(length + 3, height - 5))
    mirror_c = int(rng.integers(6, width - 4))
    mirror = tuple((mirror_r - i, mirror_c - i) for i in range(length))

    a_offset = int(rng.integers(0, 2))
    a_left = int(rng.integers(2, mirror_c - 2))
    tower_a = _tower_geometry(mirror_r - a_offset, a_left, "right", a_offset)

    cross_r = int(rng.integers(mirror_r + 3, height - 2))
    b_offset = int(rng.integers(0, 2))
    b_left = int(rng.integers(2, mirror_c - 2))
    tower_b = _tower_geometry(cross_r - b_offset, b_left, "right", b_offset)

    return {"towers": [tower_a, tower_b], "mirrors": [mirror]}


def _build_reflect_disjoint(height: int, width: int, rng: np.random.Generator) -> dict:
    length = int(rng.integers(2, 5))
    mirror_r = int(rng.integers(length + 3, height - 5))
    mirror_c = int(rng.integers(6, width - 7))
    mirror = tuple((mirror_r - i, mirror_c - i) for i in range(length))

    a_offset = int(rng.integers(0, 2))
    a_left = int(rng.integers(2, mirror_c - 2))
    tower_a = _tower_geometry(mirror_r - a_offset, a_left, "right", a_offset)

    b_offset = int(rng.integers(0, 2))
    down_col = int(rng.integers(mirror_c + 4, width - 3))
    b_top = int(rng.integers(2, height - 6))
    tower_b = _tower_geometry(b_top, down_col - b_offset, "down", b_offset)

    return {"towers": [tower_a, tower_b], "mirrors": [mirror]}


def _build_base_scene(height: int, width: int, scenario: str, rng: np.random.Generator) -> dict | None:
    if scenario == "straight_cross":
        return _build_straight_cross(height, width, rng)
    if scenario == "straight_disjoint":
        return _build_straight_disjoint(height, width, rng)
    if scenario == "reflect_cross":
        return _build_reflect_cross(height, width, rng)
    assert scenario == "reflect_disjoint"
    return _build_reflect_disjoint(height, width, rng)


def _clear_for_tower(grid: np.ndarray, cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    height, width = grid.shape
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c):
            return False
        if (r, c) == (height - 1, width - 1) or int(grid[r, c]) != BG:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not _in_bounds(grid.shape, rr, cc):
                    continue
                if (rr, cc) in cell_set:
                    continue
                if int(grid[rr, cc]) != BG:
                    return False
    return True


def _clear_for_mirror(grid: np.ndarray, cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    height, width = grid.shape
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c):
            return False
        if (r, c) == (height - 1, width - 1) or int(grid[r, c]) != BG:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not _in_bounds(grid.shape, rr, cc):
                    continue
                if (rr, cc) in cell_set:
                    continue
                if int(grid[rr, cc]) != BG:
                    return False
    return True


def _flags_match(metrics: dict, want_reflection: bool, want_cross: bool) -> bool:
    return (metrics["reflections"] > 0) == want_reflection and (metrics["crossings"] > 0) == want_cross


def _quality_ok(metrics: dict) -> bool:
    return all(length >= 2 for length in metrics["path_lengths"].values()) and sum(metrics["path_lengths"].values()) >= 8


def _try_add_extra_tower(
    grid: np.ndarray,
    rng: np.random.Generator,
    want_reflection: bool,
    want_cross: bool,
) -> np.ndarray:
    if float(rng.random()) >= 0.45:
        return grid

    used = set(map(int, np.unique(grid))) - {BG, GREY}
    available = [color for color in COLOR_POOL if color not in used]
    if not available:
        return grid

    height, width = grid.shape
    for _ in range(50):
        direction = str(rng.choice(["up", "down", "left", "right"]))
        offset = int(rng.integers(0, 2))
        top = int(rng.integers(1, height - 2))
        left = int(rng.integers(1, width - 2))
        tower = _tower_geometry(top, left, direction, offset)
        if not _clear_for_tower(grid, tower["cells"]):
            continue

        color = int(rng.choice(available))
        candidate = grid.copy()
        for r, c in tower["cells"]:
            candidate[r, c] = color
        try:
            metrics = _analyze_grid(candidate)
        except AssertionError:
            continue
        if _flags_match(metrics, want_reflection, want_cross) and _quality_ok(metrics):
            return candidate

    return grid


def _try_add_extra_mirrors(
    grid: np.ndarray,
    rng: np.random.Generator,
    want_reflection: bool,
    want_cross: bool,
) -> np.ndarray:
    target = int(rng.integers(0, 3))
    height, width = grid.shape
    out = grid.copy()

    for _ in range(target):
        placed = False
        for _ in range(60):
            orient = str(rng.choice(["/", "\\"]))
            length = int(rng.integers(2, 5))
            if orient == "\\":
                start_r = int(rng.integers(2, height - length - 1))
                start_c = int(rng.integers(2, width - length - 1))
                cells = tuple((start_r + i, start_c + i) for i in range(length))
            else:
                start_r = int(rng.integers(2, height - length - 1))
                start_c = int(rng.integers(2, width - length - 1))
                cells = tuple((start_r + i, start_c + length - 1 - i) for i in range(length))

            if not _clear_for_mirror(out, cells):
                continue

            candidate = out.copy()
            for r, c in cells:
                candidate[r, c] = GREY
            try:
                metrics = _analyze_grid(candidate)
            except AssertionError:
                continue
            if _flags_match(metrics, want_reflection, want_cross) and _quality_ok(metrics):
                out = candidate
                placed = True
                break
        if not placed:
            continue

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    scenario = str(
        rng.choice(
            [
                "straight_cross",
                "straight_disjoint",
                "reflect_cross",
                "reflect_disjoint",
            ]
        )
    )
    want_reflection = scenario.startswith("reflect")
    want_cross = scenario.endswith("cross")

    scene = _build_base_scene(height, width, scenario, rng)
    if scene is None:
        return None
    if (height - 1, width - 1) in _scene_occupancy(scene):
        return None

    sym = int(rng.integers(0, 4))
    scene = _transform_scene(scene, height, width, sym)
    if (height - 1, width - 1) in _scene_occupancy(scene):
        return None

    tower_count = len(scene["towers"])
    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    tower_colors = [int(c) for c in colors[:tower_count]]
    mix_color = int(colors[tower_count])

    grid = _render_scene(height, width, scene, tower_colors, mix_color)
    try:
        metrics = _analyze_grid(grid)
    except AssertionError:
        return None
    if not (_flags_match(metrics, want_reflection, want_cross) and _quality_ok(metrics)):
        return None

    grid = _try_add_extra_tower(grid, rng, want_reflection, want_cross)
    grid = _try_add_extra_mirrors(grid, rng, want_reflection, want_cross)

    try:
        final_metrics = _analyze_grid(grid)
    except AssertionError:
        return None
    if not (_flags_match(final_metrics, want_reflection, want_cross) and _quality_ok(final_metrics)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower mirror puzzle")
