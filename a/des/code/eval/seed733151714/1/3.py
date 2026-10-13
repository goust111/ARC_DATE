import numpy as np


BG = 0
YELLOW = 4
GREY = 5
COLOR_POOL = [1, 2, 3, 6, 7, 8, 9]

CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SUCCESS_START_DIRS = [(-1, 0), (0, -1), (0, 1)]
TOWER_DIRS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 1),
]
SIDE_TO_FINAL_DIR = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
MIRROR_SECOND_OFFSETS = {
    "\\": [(-1, -1), (1, 1)],
    "/": [(-1, 1), (1, -1)],
}

GRID_H_MIN = 18
GRID_H_MAX = 27
GRID_W_MIN = 18
GRID_W_MAX = 27
TOTAL_TOWERS_MIN = 2
TOTAL_TOWERS_MAX = 5
SUCCESS_PLACEMENT_TRIES = 220
FAIL_PLACEMENT_TRIES = 220
MAX_GENERATE_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _reflect(direction: tuple[int, int], orient: str) -> tuple[int, int]:
    dr, dc = direction
    if orient == "\\":
        return (dc, dr)
    return (-dc, -dr)


def _parse_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mirror_by_cell: dict[tuple[int, int], str] = {}
    for comp in _components(grid == YELLOW, connectivity=8):
        assert len(comp) == 2, "every mirror must be a 2-pixel yellow segment"
        (r0, c0), (r1, c1) = sorted(comp)
        dr = r1 - r0
        dc = c1 - c0
        assert abs(dr) == 1 and abs(dc) == 1, "mirror pixels must be diagonally adjacent"
        orient = "\\" if dr == dc else "/"
        for cell in comp:
            mirror_by_cell[cell] = orient
    return mirror_by_cell


def _gap_side(r0: int, c0: int, size: int, gap: tuple[int, int]) -> str | None:
    r, c = gap
    if r == r0 and c0 < c < c0 + size - 1:
        return "top"
    if r == r0 + size - 1 and c0 < c < c0 + size - 1:
        return "bottom"
    if c == c0 and r0 < r < r0 + size - 1:
        return "left"
    if c == c0 + size - 1 and r0 < r < r0 + size - 1:
        return "right"
    return None


def _parse_receivers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    receivers: list[dict] = []
    used: set[tuple[int, int]] = set()
    for size in (4, 3):
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                bbox = {(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)}
                if bbox & used:
                    continue
                border: list[tuple[int, int]] = []
                for c in range(c0, c0 + size):
                    border.append((r0, c))
                    border.append((r0 + size - 1, c))
                for r in range(r0 + 1, r0 + size - 1):
                    border.append((r, c0))
                    border.append((r, c0 + size - 1))
                border = list(dict.fromkeys(border))
                interior = [
                    (r, c)
                    for r in range(r0 + 1, r0 + size - 1)
                    for c in range(c0 + 1, c0 + size - 1)
                ]
                if any(int(grid[r, c]) != BG for r, c in interior):
                    continue
                gap_cells = [cell for cell in border if int(grid[cell]) == BG]
                if len(gap_cells) != 1:
                    continue
                gap = gap_cells[0]
                side = _gap_side(r0, c0, size, gap)
                if side is None:
                    continue
                if any(int(grid[r, c]) != GREY for r, c in border if (r, c) != gap):
                    continue
                used |= bbox
                receivers.append(
                    {
                        "size": size,
                        "top_left": (r0, c0),
                        "gap": gap,
                        "side": side,
                        "interior": tuple(interior),
                    }
                )
    return sorted(receivers, key=lambda rec: (rec["top_left"][0], rec["top_left"][1]))


def _is_isolated_tower_body(grid: np.ndarray, light: tuple[int, int]) -> bool:
    r, c = light
    if r + 3 >= grid.shape[0]:
        return False
    body = {(r + k, c) for k in (1, 2, 3)}
    if any(int(grid[br, bc]) != GREY for br, bc in body):
        return False
    for br, bc in body:
        for nr, nc in ((br - 1, bc), (br + 1, bc), (br, bc - 1), (br, bc + 1)):
            if not _in_bounds(grid.shape, nr, nc) or (nr, nc) in body:
                continue
            if int(grid[nr, nc]) == GREY:
                return False
    return True


def _parse_towers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    colors = sorted(set(int(v) for v in grid.ravel()) - {BG, YELLOW, GREY})
    towers: list[dict] = []
    seen_colors: set[int] = set()
    for color in colors:
        comps = _components(grid == color, connectivity=8)
        assert len(comps) == 1, "each tower color must form a single 2-pixel marker"
        comp = comps[0]
        assert len(comp) == 2, "tower marker must contain light and pointer only"
        light = None
        for r, c in comp:
            if _is_isolated_tower_body(grid, (r, c)):
                assert light is None, "tower light must be unique"
                light = (r, c)
        assert light is not None, "tower light must sit above a 3-pixel grey body"
        pointer = comp[0] if comp[1] == light else comp[1]
        direction = (pointer[0] - light[0], pointer[1] - light[1])
        assert direction != (0, 0) and max(abs(direction[0]), abs(direction[1])) == 1
        towers.append(
            {
                "color": color,
                "light": light,
                "pointer": pointer,
                "direction": direction,
                "body": tuple((light[0] + k, light[1]) for k in (1, 2, 3)),
            }
        )
        assert color not in seen_colors, "tower colors must be unique"
        seen_colors.add(color)
    return sorted(towers, key=lambda tower: (tower["light"][0], tower["light"][1]))


def _simulate_scene(
    grid: np.ndarray,
    towers: list[dict] | None = None,
    receivers: list[dict] | None = None,
    mirror_by_cell: dict[tuple[int, int], str] | None = None,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    towers = _parse_towers(g) if towers is None else towers
    receivers = _parse_receivers(g) if receivers is None else receivers
    mirror_by_cell = _parse_mirrors(g) if mirror_by_cell is None else mirror_by_cell

    receiver_by_gap = {tuple(rec["gap"]): rec for rec in receivers}
    out = g.copy()
    hits: list[tuple[int, tuple[int, int]]] = []

    for tower in towers:
        color = int(tower["color"])
        dr, dc = tower["direction"]
        r = tower["pointer"][0] + dr
        c = tower["pointer"][1] + dc
        seen: set[tuple[int, int, int, int]] = set()

        while _in_bounds(g.shape, r, c):
            state = (r, c, dr, dc)
            if state in seen:
                break
            seen.add(state)

            if (r, c) in receiver_by_gap:
                out[r, c] = color
                rec = receiver_by_gap[(r, c)]
                for rr, cc in rec["interior"]:
                    if int(g[rr, cc]) == BG:
                        out[rr, cc] = color
                hits.append((color, (r, c)))
                break

            cell = int(g[r, c])
            if cell == BG:
                out[r, c] = color
                r += dr
                c += dc
                continue
            if cell == YELLOW:
                orient = mirror_by_cell[(r, c)]
                dr, dc = _reflect((dr, dc), orient)
                r += dr
                c += dc
                continue
            break

    return out, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored rays from tower pointers, reflect them on yellow diagonal mirrors,
    and fill a receiver when a ray reaches its grey-frame entrance gap.
    """
    out, _hits = _simulate_scene(np.asarray(grid, dtype=int))
    return out


def _is_orthogonal(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] * b[0] + a[1] * b[1] == 0


def _success_sequences(final_dir: tuple[int, int]) -> list[tuple[tuple[int, int], ...]]:
    out: list[tuple[tuple[int, int], ...]] = []
    for length in (1, 2, 3):
        def dfs(path: list[tuple[int, int]]) -> None:
            if len(path) == length:
                if path[-1] == final_dir:
                    out.append(tuple(path))
                return
            choices = SUCCESS_START_DIRS if not path else CARDINAL_DIRS
            for direction in choices:
                if path and not _is_orthogonal(path[-1], direction):
                    continue
                path.append(direction)
                dfs(path)
                path.pop()

        dfs([])
    return out


SUCCESS_SEQUENCE_CACHE = {
    final_dir: _success_sequences(final_dir) for final_dir in SIDE_TO_FINAL_DIR.values()
}


def _turn_orientation(d_in: tuple[int, int], d_out: tuple[int, int]) -> str:
    if _reflect(d_in, "\\") == d_out:
        return "\\"
    if _reflect(d_in, "/") == d_out:
        return "/"
    raise ValueError("invalid turn for diagonal mirror reflection")


def _receiver_cells(
    top_left: tuple[int, int], size: int, side: str, gap_offset: int
) -> dict:
    r0, c0 = top_left
    if side == "top":
        gap = (r0, c0 + gap_offset)
    elif side == "bottom":
        gap = (r0 + size - 1, c0 + gap_offset)
    elif side == "left":
        gap = (r0 + gap_offset, c0)
    else:
        gap = (r0 + gap_offset, c0 + size - 1)

    border = set()
    for c in range(c0, c0 + size):
        border.add((r0, c))
        border.add((r0 + size - 1, c))
    for r in range(r0, r0 + size):
        border.add((r, c0))
        border.add((r, c0 + size - 1))
    border.remove(gap)

    interior = {
        (r, c)
        for r in range(r0 + 1, r0 + size - 1)
        for c in range(c0 + 1, c0 + size - 1)
    }
    return {
        "size": size,
        "top_left": top_left,
        "side": side,
        "gap": gap,
        "border": border,
        "interior": interior,
    }


def _path_to_boundary(
    start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = start
    dr, dc = direction
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _tower_geometry(
    light: tuple[int, int], direction: tuple[int, int]
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...], tuple[int, int]]:
    pointer = (light[0] + direction[0], light[1] + direction[1])
    body = tuple((light[0] + k, light[1]) for k in (1, 2, 3))
    start = (light[0] + 2 * direction[0], light[1] + 2 * direction[1])
    return pointer, body, start


def _neighbors8(cell: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if _in_bounds(shape, nr, nc):
                out.append((nr, nc))
    return out


def _draw_receiver(grid: np.ndarray, receiver: dict) -> None:
    for r, c in receiver["border"]:
        grid[r, c] = GREY


def _draw_tower(grid: np.ndarray, color: int, light: tuple[int, int], direction: tuple[int, int]) -> None:
    pointer, body, _start = _tower_geometry(light, direction)
    grid[light] = color
    grid[pointer] = color
    for r, c in body:
        grid[r, c] = GREY


def _draw_mirror(grid: np.ndarray, cells: tuple[tuple[int, int], tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = YELLOW


def _build_success_layout(
    shape: tuple[int, int],
    receiver: dict,
    sequence: tuple[tuple[int, int], ...],
    lengths: list[int],
) -> dict | None:
    gap = receiver["gap"]
    target = gap
    path_cells: list[tuple[int, int]] = []
    mirror_primary: list[tuple[int, int]] = [(-1, -1)] * (len(sequence) - 1)

    for idx in range(len(sequence) - 1, -1, -1):
        dr, dc = sequence[idx]
        segment = [
            (target[0] - dr * step, target[1] - dc * step)
            for step in range(lengths[idx], 0, -1)
        ]
        path_cells.extend(segment)
        source = (target[0] - dr * (lengths[idx] + 1), target[1] - dc * (lengths[idx] + 1))
        if idx > 0:
            mirror_primary[idx - 1] = source
        else:
            start = source
        target = source

    light = (start[0] - 2 * sequence[0][0], start[1] - 2 * sequence[0][1])
    pointer, body, start_check = _tower_geometry(light, sequence[0])
    assert start_check == start

    if not all(_in_bounds(shape, r, c) for r, c in path_cells):
        return None
    if not _in_bounds(shape, *light) or not _in_bounds(shape, *pointer):
        return None
    if not all(_in_bounds(shape, r, c) for r, c in body):
        return None

    mirror_cells: list[tuple[tuple[int, int], tuple[int, int]]] = []
    occupied_now = set(path_cells) | set(receiver["border"]) | set(receiver["interior"]) | {receiver["gap"], light, pointer} | set(body)
    for idx, primary in enumerate(mirror_primary):
        orient = _turn_orientation(sequence[idx], sequence[idx + 1])
        candidates = [
            (primary[0] + dr, primary[1] + dc)
            for dr, dc in MIRROR_SECOND_OFFSETS[orient]
            if _in_bounds(shape, primary[0] + dr, primary[1] + dc)
        ]
        candidates = [cell for cell in candidates if cell not in occupied_now]
        if not candidates:
            return None
        second = candidates[0]
        mirror_cells.append((primary, second))
        occupied_now.add(primary)
        occupied_now.add(second)

    tower_cells = {light, pointer, *body}
    if tower_cells & (set(receiver["border"]) | set(receiver["interior"]) | {receiver["gap"]}):
        return None
    if tower_cells & set(path_cells):
        return None

    return {
        "receiver": receiver,
        "path_cells": set(path_cells),
        "light": light,
        "direction": sequence[0],
        "tower_cells": tower_cells,
        "mirrors": tuple(mirror_cells),
    }


def _try_place_success(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    color: int,
    rng: np.random.Generator,
    force_turn: bool,
) -> dict | None:
    h, w = grid.shape
    blocked = occupied | reserved
    gap_choices = {3: [1], 4: [1, 2]}
    max_seg = max(2, min(h, w) // 4)

    for _ in range(SUCCESS_PLACEMENT_TRIES):
        size = int(rng.choice([3, 4]))
        r_min = 2
        c_min = 2
        r_max = h - size - 2
        c_max = w - size - 2
        if r_max < r_min or c_max < c_min:
            continue
        receiver = _receiver_cells(
            (int(rng.integers(r_min, r_max + 1)), int(rng.integers(c_min, c_max + 1))),
            size,
            str(rng.choice(["top", "bottom", "left", "right"])),
            int(rng.choice(gap_choices[size])),
        )
        receiver_all = receiver["border"] | receiver["interior"] | {receiver["gap"]}
        if receiver_all & blocked:
            continue

        final_dir = SIDE_TO_FINAL_DIR[receiver["side"]]
        sequences = SUCCESS_SEQUENCE_CACHE[final_dir]
        if force_turn:
            sequences = [seq for seq in sequences if len(seq) >= 2]
        sequence = sequences[int(rng.integers(0, len(sequences)))]
        lengths = [int(rng.integers(1, max_seg + 1)) for _ in sequence]
        layout = _build_success_layout(grid.shape, receiver, sequence, lengths)
        if layout is None:
            continue

        object_cells = set(layout["tower_cells"]) | receiver["border"]
        for mirror in layout["mirrors"]:
            object_cells.update(mirror)
        black_cells = layout["path_cells"] | receiver["interior"] | {receiver["gap"]}
        if object_cells & blocked or black_cells & blocked or object_cells & black_cells:
            continue

        existing_yellow = {tuple(cell) for cell in np.argwhere(grid == YELLOW)}
        planned_yellow: set[tuple[int, int]] = set()
        mirrors_ok = True
        for mirror in layout["mirrors"]:
            pair = set(mirror)
            for cell in pair:
                for nbr in _neighbors8(cell, grid.shape):
                    if nbr in existing_yellow or (nbr in planned_yellow and nbr not in pair):
                        mirrors_ok = False
                        break
                if not mirrors_ok:
                    break
            if not mirrors_ok:
                break
            planned_yellow.update(pair)
        if not mirrors_ok:
            continue

        _draw_receiver(grid, receiver)
        _draw_tower(grid, color, layout["light"], layout["direction"])
        for mirror in layout["mirrors"]:
            _draw_mirror(grid, mirror)

        occupied.update(object_cells)
        reserved.update(black_cells)
        return {
            "receiver": receiver,
            "mirrors": layout["mirrors"],
            "tower_color": color,
        }

    return None


def _try_place_fail(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    color: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    blocked = occupied | reserved

    for _ in range(FAIL_PLACEMENT_TRIES):
        direction = TOWER_DIRS[int(rng.integers(0, len(TOWER_DIRS)))]
        if direction == (1, 0):
            continue
        light = (
            int(rng.integers(1, h - 3)),
            int(rng.integers(1, w - 1)),
        )
        pointer, body, start = _tower_geometry(light, direction)
        tower_cells = {light, pointer, *body}
        if not _in_bounds(grid.shape, *pointer) or not _in_bounds(grid.shape, *start):
            continue
        if any(not _in_bounds(grid.shape, r, c) for r, c in body):
            continue
        if tower_cells & blocked:
            continue

        path_cells = _path_to_boundary(start, direction, grid.shape)
        if len(path_cells) < 2:
            continue
        path_set = set(path_cells)
        if path_set & blocked or path_set & tower_cells:
            continue

        _draw_tower(grid, color, light, direction)
        occupied.update(tower_cells)
        reserved.update(path_set)
        return True

    return False


def _build_case(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_MIN, GRID_H_MAX))
    w = int(rng.integers(GRID_W_MIN, GRID_W_MAX))
    total_towers = int(rng.integers(TOTAL_TOWERS_MIN, TOTAL_TOWERS_MAX + 1))
    success_count = int(rng.integers(1, total_towers))
    colors = list(rng.choice(COLOR_POOL, size=total_towers, replace=False))

    grid = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    placed_successes: list[dict] = []

    forced_turn_index = int(rng.integers(0, success_count))
    for idx in range(success_count):
        result = _try_place_success(
            grid,
            occupied,
            reserved,
            int(colors[idx]),
            rng,
            force_turn=(idx == forced_turn_index),
        )
        if result is None:
            return None
        placed_successes.append(result)

    for idx in range(success_count, total_towers):
        if not _try_place_fail(grid, occupied, reserved, int(colors[idx]), rng):
            return None

    try:
        towers = _parse_towers(grid)
        receivers = _parse_receivers(grid)
        mirror_by_cell = _parse_mirrors(grid)
        out, hits = _simulate_scene(grid, towers=towers, receivers=receivers, mirror_by_cell=mirror_by_cell)
    except AssertionError:
        return None
    if len(towers) != total_towers or len(receivers) != success_count:
        return None
    if not mirror_by_cell:
        return None
    if len(hits) != success_count or len({gap for _color, gap in hits}) != success_count:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower ARC input")
