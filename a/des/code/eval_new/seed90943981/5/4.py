import numpy as np


BLACK = 0
GRAY = 5
RELAY_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
RNG = np.random.default_rng()

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

BRANCHES = ("paint_receiver", "receiver_no_relay", "relay_no_receiver", "empty")

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
MIN_SIZE = 18
MAX_SIZE = 28
MIN_OPEN_PATH = 6
PAIR_SEPARATION = MIN_OPEN_PATH + 5

UP_PATTERN = np.array(
    [
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dtype=bool,
)
DOWN_PATTERN = UP_PATTERN[::-1, :]
LEFT_PATTERN = np.array(
    [
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ],
    dtype=bool,
)
RIGHT_PATTERN = LEFT_PATTERN[:, ::-1]


def _randint(lo: int, hi: int) -> int:
    return int(RNG.integers(lo, hi + 1))


def _choice(seq):
    return seq[int(RNG.integers(0, len(seq)))]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
        (r, c - 1),                 (r, c + 1),
        (r + 1, c - 1), (r + 1, c), (r + 1, c + 1),
    ]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _tower_shape(square_r: int, square_c: int, step: tuple[int, int]) -> dict[str, object]:
    square = {
        (square_r + dr, square_c + dc)
        for dr in range(3)
        for dc in range(3)
    }
    if step == UP:
        antenna = [(square_r - 1, square_c + 1), (square_r - 2, square_c + 1)]
        ray_start = (square_r - 3, square_c + 1)
    elif step == DOWN:
        antenna = [(square_r + 3, square_c + 1), (square_r + 4, square_c + 1)]
        ray_start = (square_r + 5, square_c + 1)
    elif step == LEFT:
        antenna = [(square_r + 1, square_c - 1), (square_r + 1, square_c - 2)]
        ray_start = (square_r + 1, square_c - 3)
    else:
        antenna = [(square_r + 1, square_c + 3), (square_r + 1, square_c + 4)]
        ray_start = (square_r + 1, square_c + 5)
    cells = square | set(antenna)
    return {
        "square_top": int(square_r),
        "square_left": int(square_c),
        "step": step,
        "square_cells": tuple(sorted(square)),
        "antenna_cells": tuple(antenna),
        "all_cells": tuple(sorted(cells)),
        "ray_start": ray_start,
    }


def _ray_start_in_bounds(tower: dict[str, object], shape: tuple[int, int]) -> bool:
    h, w = shape
    r, c = tower["ray_start"]
    return 0 <= r < h and 0 <= c < w


def _can_place_gray(cells: tuple[tuple[int, int], ...], shape: tuple[int, int], gray_cells: set[tuple[int, int]]) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in gray_cells:
            return False
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in gray_cells:
                return False
    return True


def _tower_from_component(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(coords) != 11:
        return None
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True

    if local.shape == UP_PATTERN.shape and np.array_equal(local, UP_PATTERN):
        return _tower_shape(r0 + 2, c0, UP)
    if local.shape == DOWN_PATTERN.shape and np.array_equal(local, DOWN_PATTERN):
        return _tower_shape(r0, c0, DOWN)
    if local.shape == LEFT_PATTERN.shape and np.array_equal(local, LEFT_PATTERN):
        return _tower_shape(r0, c0 + 2, LEFT)
    if local.shape == RIGHT_PATTERN.shape and np.array_equal(local, RIGHT_PATTERN):
        return _tower_shape(r0, c0, RIGHT)
    return None


def _receiver_from_component(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(coords) != 4:
        return None
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return None
    expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if set(coords) != expected:
        return None
    return {"cells": tuple(sorted(expected))}


def _parse_scene(grid: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    gray_comps = _components(grid == GRAY, connectivity=4)
    towers: list[dict[str, object]] = []
    receivers: list[dict[str, object]] = []
    for comp in gray_comps:
        tower = _tower_from_component(comp)
        if tower is not None:
            towers.append(tower)
            continue
        receiver = _receiver_from_component(comp)
        if receiver is not None:
            receivers.append(receiver)
            continue
        raise AssertionError("unexpected gray component shape")
    assert towers, "expected at least one signal tower"
    assert receivers, "expected at least one receiver"
    for idx, tower in enumerate(towers):
        tower["id"] = int(idx)
    return towers, receivers


def _trace_path(
    tower: dict[str, object],
    square_owner: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], str]:
    h, w = shape
    r, c = tower["ray_start"]
    dr, dc = tower["step"]
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        owner = square_owner.get((r, c))
        if owner is not None and owner != tower["id"]:
            return path, "tower"
        path.append((r, c))
        r += dr
        c += dc
    return path, "boundary"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each receiver hit by a tower ray using the last relay color seen on that ray.

    Input structure:
    - Background is black.
    - Gray objects are either 3x3 signal towers with a two-cell antenna pointing up, down, left, or right, or 2x2 receivers with no antenna.
    - Colored non-black, non-gray single pixels are relays scattered on the empty background.

    Rule:
    - Each tower emits a one-cell-wide ray starting just beyond the antenna tip and moving in the antenna direction.
    - The ray is invisible and continues until it leaves the grid or would enter another tower's 3x3 body.
    - While scanning the ray, remember the color of the farthest relay encountered so far.
    - Whenever the ray passes through a receiver after at least one relay has been seen, recolor that entire 2x2 receiver to the remembered relay color.
    - Towers, untouched receivers, and all relays remain otherwise unchanged.

    Key insight:
    - The task is a chained inference: decode tower orientation, follow the exact ray, keep only the last relay on that path, and apply that color to any receiver the ray intersects.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    towers, receivers = _parse_scene(g)

    square_owner: dict[tuple[int, int], int] = {}
    for tower in towers:
        for cell in tower["square_cells"]:
            square_owner[cell] = int(tower["id"])

    receiver_lookup: dict[tuple[int, int], int] = {}
    for rid, receiver in enumerate(receivers):
        for cell in receiver["cells"]:
            receiver_lookup[cell] = int(rid)

    for tower in towers:
        ray_color = None
        r, c = tower["ray_start"]
        dr, dc = tower["step"]
        while 0 <= r < h and 0 <= c < w:
            owner = square_owner.get((r, c))
            if owner is not None and owner != tower["id"]:
                break
            value = int(g[r, c])
            if value not in (BLACK, GRAY):
                ray_color = int(value)
            rec_id = receiver_lookup.get((r, c))
            if rec_id is not None and ray_color is not None:
                for rr, cc in receivers[rec_id]["cells"]:
                    out[rr, cc] = int(ray_color)
            r += dr
            c += dc

    return out


def _sample_independent_tower(shape: tuple[int, int], gray_cells: set[tuple[int, int]]) -> dict[str, object] | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        step = _choice(DIRS)
        if step == UP:
            if h < 11 or w < 5:
                return None
            lo_r, hi_r = 8, h - 3
            lo_c, hi_c = 1, w - 4
        elif step == DOWN:
            if h < 11 or w < 5:
                return None
            lo_r, hi_r = 0, h - 11
            lo_c, hi_c = 1, w - 4
        elif step == LEFT:
            if h < 5 or w < 11:
                return None
            lo_r, hi_r = 1, h - 4
            lo_c, hi_c = 8, w - 3
        else:
            if h < 5 or w < 11:
                return None
            lo_r, hi_r = 1, h - 4
            lo_c, hi_c = 0, w - 11
        if lo_r > hi_r or lo_c > hi_c:
            continue
        tower = _tower_shape(_randint(lo_r, hi_r), _randint(lo_c, hi_c), step)
        if not _ray_start_in_bounds(tower, shape):
            continue
        if _can_place_gray(tower["all_cells"], shape, gray_cells):
            return tower
    return None


def _sample_blocker_pair(shape: tuple[int, int], gray_cells: set[tuple[int, int]]) -> tuple[dict[str, object], dict[str, object]] | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        source_step = _choice(DIRS)
        if source_step == UP:
            lo_sr, hi_sr = 12, h - 3
            lo_sc, hi_sc = 1, w - 4
            if lo_sr > hi_sr or lo_sc > hi_sc:
                continue
            sr = _randint(lo_sr, hi_sr)
            sc = _randint(lo_sc, hi_sc)
            br = _randint(1, sr - PAIR_SEPARATION)
            bc = sc
        elif source_step == DOWN:
            lo_sr, hi_sr = 0, h - 14
            lo_sc, hi_sc = 1, w - 4
            if lo_sr > hi_sr or lo_sc > hi_sc:
                continue
            sr = _randint(lo_sr, hi_sr)
            sc = _randint(lo_sc, hi_sc)
            br = _randint(sr + PAIR_SEPARATION, h - 3)
            bc = sc
        elif source_step == LEFT:
            lo_sr, hi_sr = 1, h - 4
            lo_sc, hi_sc = 12, w - 3
            if lo_sr > hi_sr or lo_sc > hi_sc:
                continue
            sr = _randint(lo_sr, hi_sr)
            sc = _randint(lo_sc, hi_sc)
            br = sr
            bc = _randint(1, sc - PAIR_SEPARATION)
        else:
            lo_sr, hi_sr = 1, h - 4
            lo_sc, hi_sc = 0, w - 14
            if lo_sr > hi_sr or lo_sc > hi_sc:
                continue
            sr = _randint(lo_sr, hi_sr)
            sc = _randint(lo_sc, hi_sc)
            br = sr
            bc = _randint(sc + PAIR_SEPARATION, w - 3)

        source = _tower_shape(sr, sc, source_step)
        if not _ray_start_in_bounds(source, shape):
            continue
        if not _can_place_gray(source["all_cells"], shape, gray_cells):
            continue

        blocker_options = []
        for blocker_step in DIRS:
            if blocker_step == OPPOSITE[source_step]:
                continue
            blocker = _tower_shape(br, bc, blocker_step)
            if not _ray_start_in_bounds(blocker, shape):
                continue
            if _can_place_gray(blocker["all_cells"], shape, gray_cells | set(source["all_cells"])):
                blocker_options.append(blocker)
        if blocker_options:
            blocker = _choice(blocker_options)
            return source, blocker
    return None


def _cell_to_rays(paths: list[list[tuple[int, int]]]) -> dict[tuple[int, int], set[int]]:
    mapping: dict[tuple[int, int], set[int]] = {}
    for idx, path in enumerate(paths):
        for cell in path:
            mapping.setdefault(cell, set()).add(int(idx))
    return mapping


def _relay_candidates(
    path: list[tuple[int, int]],
    tower_id: int,
    cell_rays: dict[tuple[int, int], set[int]],
    occupied_any: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int]]]:
    out: list[tuple[int, tuple[int, int]]] = []
    for idx, cell in enumerate(path):
        if cell in occupied_any:
            continue
        if cell_rays.get(cell) == {tower_id}:
            out.append((idx, cell))
    return out


def _gray_placeable(
    cells: tuple[tuple[int, int], ...],
    shape: tuple[int, int],
    occupied_any: set[tuple[int, int]],
    occupied_gray: set[tuple[int, int]],
) -> bool:
    h, w = shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied_any:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied_gray and (nr, nc) not in cell_set:
                return False
    return True


def _receiver_options(
    shape: tuple[int, int],
    tower: dict[str, object],
    path: list[tuple[int, int]],
    cell_rays: dict[tuple[int, int], set[int]],
    occupied_any: set[tuple[int, int]],
    occupied_gray: set[tuple[int, int]],
) -> list[dict[str, object]]:
    tower_id = int(tower["id"])
    dr, dc = tower["step"]
    options: list[dict[str, object]] = []
    for start_idx in range(len(path) - 1):
        (r0, c0), (r1, c1) = path[start_idx], path[start_idx + 1]
        if (r1 - r0, c1 - c0) != (dr, dc):
            continue
        if dr == 0:
            for side in (-1, 1):
                cells = ((r0, c0), (r1, c1), (r0 + side, c0), (r1 + side, c1))
                if any(cell_rays.get(cell, {tower_id}) - {tower_id} for cell in cells):
                    continue
                if _gray_placeable(cells, shape, occupied_any, occupied_gray):
                    options.append({"start_index": int(start_idx), "cells": cells})
        else:
            for side in (-1, 1):
                cells = ((r0, c0), (r1, c1), (r0, c0 + side), (r1, c1 + side))
                if any(cell_rays.get(cell, {tower_id}) - {tower_id} for cell in cells):
                    continue
                if _gray_placeable(cells, shape, occupied_any, occupied_gray):
                    options.append({"start_index": int(start_idx), "cells": cells})
    return options


def _paint_cells(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _shuffle(seq):
    seq = list(seq)
    RNG.shuffle(seq)
    return seq


def _place_branch(
    grid: np.ndarray,
    tower: dict[str, object],
    path: list[tuple[int, int]],
    branch: str,
    cell_rays: dict[tuple[int, int], set[int]],
    occupied_any: set[tuple[int, int]],
    occupied_gray: set[tuple[int, int]],
    available_colors: list[int],
) -> bool:
    shape = grid.shape
    tower_id = int(tower["id"])

    if branch == "empty":
        return True

    if branch == "receiver_no_relay":
        options = _shuffle(_receiver_options(shape, tower, path, cell_rays, occupied_any, occupied_gray))
        if not options:
            return False
        cells = options[0]["cells"]
        _paint_cells(grid, cells, GRAY)
        occupied_any.update(cells)
        occupied_gray.update(cells)
        return True

    if branch == "relay_no_receiver":
        if not available_colors:
            return False
        candidates = _relay_candidates(path, tower_id, cell_rays, occupied_any)
        if not candidates:
            return False
        max_relays = min(2, len(candidates), len(available_colors))
        relay_count = 1 if max_relays == 1 else _randint(1, max_relays)
        picked = sorted(_shuffle(candidates)[:relay_count])
        for _idx, cell in picked:
            color = available_colors.pop()
            _paint_cells(grid, (cell,), color)
            occupied_any.add(cell)
        return True

    assert branch == "paint_receiver"
    if not available_colors:
        return False
    options = _shuffle(_receiver_options(shape, tower, path, cell_rays, occupied_any, occupied_gray))
    for option in options:
        before = [
            pair
            for pair in _relay_candidates(path, tower_id, cell_rays, occupied_any)
            if pair[0] < option["start_index"]
        ]
        if not before:
            continue
        max_relays = min(2, len(before), len(available_colors))
        relay_count = 1 if max_relays == 1 else _randint(1, max_relays)
        chosen = sorted(_shuffle(before)[:relay_count])
        cells = option["cells"]
        _paint_cells(grid, cells, GRAY)
        occupied_any.update(cells)
        occupied_gray.update(cells)
        for _idx, cell in chosen:
            color = available_colors.pop()
            _paint_cells(grid, (cell,), color)
            occupied_any.add(cell)
        return True
    return False


def _initial_feasible_branches(
    shape: tuple[int, int],
    towers: list[dict[str, object]],
    paths: list[list[tuple[int, int]]],
    cell_rays: dict[tuple[int, int], set[int]],
    occupied_any: set[tuple[int, int]],
    occupied_gray: set[tuple[int, int]],
) -> list[list[str]]:
    branch_sets: list[list[str]] = []
    for tower, path in zip(towers, paths):
        tower_id = int(tower["id"])
        receiver_opts = _receiver_options(shape, tower, path, cell_rays, occupied_any, occupied_gray)
        relay_cells = _relay_candidates(path, tower_id, cell_rays, occupied_any)
        feasible = ["empty"]
        if relay_cells:
            feasible.append("relay_no_receiver")
        if receiver_opts:
            feasible.append("receiver_no_relay")
        if any(any(idx < opt["start_index"] for idx, _cell in relay_cells) for opt in receiver_opts):
            feasible.append("paint_receiver")
        branch_sets.append(feasible)
    return branch_sets


def _gen_attempt() -> np.ndarray | None:
    h = _randint(MIN_SIZE, MAX_SIZE)
    w = _randint(MIN_SIZE, MAX_SIZE)
    shape = (h, w)

    tower_count = _choice([2, 3, 4])
    want_blocker = tower_count >= 2 and RNG.random() < 0.5

    towers: list[dict[str, object]] = []
    gray_cells: set[tuple[int, int]] = set()

    if want_blocker:
        pair = _sample_blocker_pair(shape, gray_cells)
        if pair is None:
            return None
        for tower in pair:
            towers.append(tower)
            gray_cells.update(tower["all_cells"])

    while len(towers) < tower_count:
        tower = _sample_independent_tower(shape, gray_cells)
        if tower is None:
            return None
        towers.append(tower)
        gray_cells.update(tower["all_cells"])

    for idx, tower in enumerate(towers):
        tower["id"] = int(idx)

    square_owner: dict[tuple[int, int], int] = {}
    all_tower_cells: set[tuple[int, int]] = set()
    for tower in towers:
        for cell in tower["square_cells"]:
            square_owner[cell] = int(tower["id"])
        all_tower_cells.update(tower["all_cells"])

    paths: list[list[tuple[int, int]]] = []
    stop_reasons: list[str] = []
    for tower in towers:
        path, stop_reason = _trace_path(tower, square_owner, shape)
        if any(cell in all_tower_cells for cell in path):
            return None
        if len(path) < 2:
            return None
        paths.append(path)
        stop_reasons.append(stop_reason)

    if want_blocker and "tower" not in stop_reasons:
        return None

    cell_rays = _cell_to_rays(paths)
    occupied_any = set(all_tower_cells)
    occupied_gray = set(all_tower_cells)

    feasible = _initial_feasible_branches(shape, towers, paths, cell_rays, occupied_any, occupied_gray)
    paint_candidates = [idx for idx, branch_set in enumerate(feasible) if "paint_receiver" in branch_set]
    if not paint_candidates:
        return None

    guaranteed_idx = int(_choice(paint_candidates))
    branches: list[str] = []
    for idx, branch_set in enumerate(feasible):
        if idx == guaranteed_idx:
            branches.append("paint_receiver")
        else:
            choices = [branch for branch in BRANCHES if branch in branch_set]
            branches.append(str(_choice(choices)))

    grid = np.full(shape, BLACK, dtype=int)
    for tower in towers:
        _paint_cells(grid, tower["all_cells"], GRAY)

    available_colors = [int(color) for color in RNG.permutation(RELAY_PALETTE)]
    order = sorted(
        range(len(towers)),
        key=lambda idx: (BRANCHES.index(branches[idx]), len(paths[idx])),
    )
    for idx in order:
        ok = _place_branch(
            grid,
            towers[idx],
            paths[idx],
            branches[idx],
            cell_rays,
            occupied_any,
            occupied_gray,
            available_colors,
        )
        if not ok:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tower-relay-receiver input")
