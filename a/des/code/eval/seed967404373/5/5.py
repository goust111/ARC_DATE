import random

import numpy as np


BG_COLOR = 0
VALVE_COLOR = 4
PIPE_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)

GRID_MIN = 12
GRID_MAX = 20
PAIR_COUNT_MIN = 1
PAIR_COUNT_MAX = 3
VISIBLE_LEN_MIN = 4
VISIBLE_LEN_MAX = 8
BLOCKER_LEN_OPTIONS = (2, 3)
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 200


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _candidate_path(orientation: str, r: int, c: int, side: int) -> list[tuple[int, int]]:
    if orientation == "h":
        far_r = r + 2 * side
        near_r = r + side
        return [
            (near_r, c - 1),
            (far_r, c - 1),
            (far_r, c),
            (far_r, c + 1),
            (near_r, c + 1),
        ]
    far_c = c + 2 * side
    near_c = c + side
    return [
        (r - 1, near_c),
        (r - 1, far_c),
        (r, far_c),
        (r + 1, far_c),
        (r + 1, near_c),
    ]


def _pipe_info(g: np.ndarray) -> dict[int, dict[str, int | str]]:
    info: dict[int, dict[str, int | str]] = {}
    for color in sorted({int(x) for x in np.unique(g)} - {BG_COLOR, VALVE_COLOR}):
        coords = np.argwhere(g == color)
        assert coords.size > 0, "each pipe color must appear at least once"

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1:
            orientation = "h"
        elif len(cols) == 1:
            orientation = "v"
        else:
            raise AssertionError("each color must form a straight pipe")

        info[color] = {
            "orientation": orientation,
            "length": int(coords.shape[0]),
        }
    return info


def _find_valves(g: np.ndarray) -> list[dict[str, int]]:
    h, w = g.shape
    valves: list[dict[str, int]] = []
    for rr, cc in np.argwhere(g == VALVE_COLOR):
        r = int(rr)
        c = int(cc)
        assert 0 < r < h - 1 and 0 < c < w - 1, "valves must have four neighbors"

        up = int(g[r - 1, c])
        down = int(g[r + 1, c])
        left = int(g[r, c - 1])
        right = int(g[r, c + 1])

        assert up == down != BG_COLOR and up != VALVE_COLOR
        assert left == right != BG_COLOR and left != VALVE_COLOR
        assert up != left, "crossing pipes must have different colors"

        valves.append({"r": r, "c": c, "v_color": up, "h_color": left})
    return valves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve pipe crossings marked by yellow valves.

    The longer pipe keeps the straight-through cell at each valve. When the two
    visible pipe lengths are equal, the horizontal pipe wins the valve. The
    losing pipe draws the shortest free U-shaped detour around the straight pipe;
    if both sides are free, it uses the side with the smaller row/column index.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pipes = _pipe_info(g)
    valves = _find_valves(g)
    out = g.copy()
    detours: list[tuple[int, int, int, str]] = []

    for valve in valves:
        r = valve["r"]
        c = valve["c"]
        h_color = valve["h_color"]
        v_color = valve["v_color"]

        assert pipes[h_color]["orientation"] == "h"
        assert pipes[v_color]["orientation"] == "v"

        h_len = int(pipes[h_color]["length"])
        v_len = int(pipes[v_color]["length"])

        if h_len >= v_len:
            priority = h_color
            detours.append((r, c, v_color, "v"))
        else:
            priority = v_color
            detours.append((r, c, h_color, "h"))

        out[r, c] = priority

    for r, c, color, orientation in detours:
        chosen = None
        for side in (-1, 1):
            cells = _candidate_path(orientation, r, c, side)
            if all(_in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == BG_COLOR for rr, cc in cells):
                chosen = cells
                break

        assert chosen is not None, "losing pipe must have a valid detour side"
        for rr, cc in chosen:
            current = int(out[rr, cc])
            assert current in {BG_COLOR, color}, "detours must not overwrite another pipe"
            out[rr, cc] = color

    return out


def _split_visible_length(length: int) -> tuple[int, int]:
    left = random.randint(1, length - 1)
    return left, length - left


def _sample_lengths(priority_case: str) -> tuple[int, int]:
    if priority_case == "equal":
        length = random.randint(VISIBLE_LEN_MIN, VISIBLE_LEN_MAX)
        return length, length

    shorter = random.randint(VISIBLE_LEN_MIN, VISIBLE_LEN_MAX - 1)
    longer = random.randint(shorter + 1, min(VISIBLE_LEN_MAX, shorter + 3))
    if priority_case == "horizontal":
        return longer, shorter
    return shorter, longer


def _relative_blocker_cells(orientation: str, blocked_side: int, length: int) -> list[tuple[int, int]]:
    if orientation == "h":
        if blocked_side < 0:
            return [(-k, -1) for k in range(1, length + 1)]
        return [(k, 1) for k in range(1, length + 1)]
    if blocked_side < 0:
        return [(-1, -k) for k in range(1, length + 1)]
    return [(1, k) for k in range(1, length + 1)]


def _pair_layout(
    h_color: int,
    v_color: int,
    priority_case: str,
    detour_mode: str,
    blocker_color: int | None,
) -> dict[str, object]:
    h_vis, v_vis = _sample_lengths(priority_case)
    left_len, right_len = _split_visible_length(h_vis)
    up_len, down_len = _split_visible_length(v_vis)

    paint: dict[tuple[int, int], int] = {}
    for dc in range(-left_len, 0):
        paint[(0, dc)] = h_color
    for dc in range(1, right_len + 1):
        paint[(0, dc)] = h_color
    for dr in range(-up_len, 0):
        paint[(dr, 0)] = v_color
    for dr in range(1, down_len + 1):
        paint[(dr, 0)] = v_color
    paint[(0, 0)] = VALVE_COLOR

    nonpriority = "v" if priority_case in {"horizontal", "equal"} else "h"
    neg_path = _candidate_path(nonpriority, 0, 0, -1)
    pos_path = _candidate_path(nonpriority, 0, 0, 1)

    blank_cells = set(neg_path) | set(pos_path)
    blocker_cells: list[tuple[int, int]] = []
    blocker_fill: dict[tuple[int, int], int] = {}
    if detour_mode != "both":
        assert blocker_color is not None, "forced detours require a blocker pipe"
        blocked_side = 1 if detour_mode == "neg_only" else -1
        blocker_cells = _relative_blocker_cells(
            nonpriority, blocked_side, random.choice(BLOCKER_LEN_OPTIONS)
        )
        blocker_fill = {cell: blocker_color for cell in blocker_cells}

    footprint = set(paint) | blank_cells | set(blocker_cells)
    min_r = min(r for r, _ in footprint)
    max_r = max(r for r, _ in footprint)
    min_c = min(c for _, c in footprint)
    max_c = max(c for _, c in footprint)

    return {
        "paint": paint,
        "blocker_fill": blocker_fill,
        "footprint": footprint,
        "bbox": (min_r, max_r, min_c, max_c),
    }


def _dilate_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _try_place_layout(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    layout: dict[str, object],
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    min_r, max_r, min_c, max_c = layout["bbox"]
    row_min = -min_r
    row_max = h - 1 - max_r
    col_min = -min_c
    col_max = w - 1 - max_c
    if row_min > row_max or col_min > col_max:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        base_r = random.randint(row_min, row_max)
        base_c = random.randint(col_min, col_max)

        translated = {(base_r + r, base_c + c) for r, c in layout["footprint"]}
        if any(cell in reserved for cell in translated):
            continue

        for (r, c), color in layout["paint"].items():
            rr = base_r + r
            cc = base_c + c
            if int(grid[rr, cc]) != BG_COLOR:
                break
            grid[rr, cc] = color
        else:
            for (r, c), color in layout["blocker_fill"].items():
                rr = base_r + r
                cc = base_c + c
                if int(grid[rr, cc]) != BG_COLOR:
                    break
                grid[rr, cc] = color
            else:
                return translated

            for (r, c), _color in layout["paint"].items():
                grid[base_r + r, base_c + c] = BG_COLOR

    return None


def _place_standalone_pipe(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    color: int,
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        orientation = random.choice(["h", "v"])
        length = random.randint(3, 6)

        if orientation == "h":
            if length > w - 2:
                continue
            r = random.randint(1, h - 2)
            c0 = random.randint(1, w - length - 1)
            cells = {(r, c0 + k) for k in range(length)}
        else:
            if length > h - 2:
                continue
            r0 = random.randint(1, h - length - 1)
            c = random.randint(1, w - 2)
            cells = {(r0 + k, c) for k in range(length)}

        if any(cell in reserved for cell in cells):
            continue

        for rr, cc in cells:
            if int(grid[rr, cc]) != BG_COLOR:
                break
        else:
            for rr, cc in cells:
                grid[rr, cc] = color
            return cells

    return None


def _sample_scenarios() -> list[dict[str, str]]:
    n_pairs = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    while True:
        scenarios = []
        forced = 0
        for _ in range(n_pairs):
            priority_case = random.choice(["horizontal", "vertical", "equal"])
            detour_mode = random.choice(["both", "neg_only", "pos_only"])
            if detour_mode != "both":
                forced += 1
            scenarios.append({"priority_case": priority_case, "detour_mode": detour_mode})
        if 2 * n_pairs + forced <= len(PIPE_COLORS):
            return scenarios


def _gen_attempt() -> np.ndarray | None:
    scenarios = _sample_scenarios()
    n_pairs = len(scenarios)
    min_size = max(GRID_MIN, 12 + 2 * (n_pairs - 1))
    h = random.randint(min_size, GRID_MAX)
    w = random.randint(min_size, GRID_MAX)

    available_colors = list(PIPE_COLORS)
    random.shuffle(available_colors)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    reserved: set[tuple[int, int]] = set()

    for scenario in scenarios:
        h_color = available_colors.pop()
        v_color = available_colors.pop()
        blocker_color = available_colors.pop() if scenario["detour_mode"] != "both" else None

        layout = _pair_layout(
            h_color=h_color,
            v_color=v_color,
            priority_case=scenario["priority_case"],
            detour_mode=scenario["detour_mode"],
            blocker_color=blocker_color,
        )
        translated = _try_place_layout(grid, reserved, layout)
        if translated is None:
            return None
        reserved |= _dilate_cells(translated, grid.shape)

    if available_colors and random.random() < 0.5:
        extra_color = available_colors.pop()
        cells = _place_standalone_pipe(grid, reserved, extra_color)
        if cells is None:
            return None
        reserved |= _dilate_cells(cells, grid.shape)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.any(out == VALVE_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-valve puzzle input")
