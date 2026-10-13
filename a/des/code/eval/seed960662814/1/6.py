from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
YELLOW = 4
MAGENTA = 6
ORANGE = 7
TOWER_COLORS = (1, 2, 3, 5, 8, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
PERPENDICULAR = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in (UP, DOWN, LEFT, RIGHT):
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_towers(grid: np.ndarray) -> list[dict[str, int]]:
    mask = (grid != BLACK) & (grid != YELLOW)
    towers: list[dict[str, int]] = []

    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "tower component must be monochromatic"
        color = next(iter(colors))
        assert comp.shape[0] == 4, "tower must have area 4"
        assert r1 - r0 == 1 and c1 - c0 == 1, "tower must be a 2x2 block"
        assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)), "tower must be solid"
        towers.append({"r": r0, "c": c0, "color": int(color)})

    towers.sort(key=lambda tower: (tower["r"], tower["c"], tower["color"]))
    assert towers, "expected at least one signal tower"
    return towers


def _tower_sources(tower: dict[str, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r = int(tower["r"])
    c = int(tower["c"])
    return [
        ((r, c), UP),
        ((r, c + 1), UP),
        ((r + 1, c), DOWN),
        ((r + 1, c + 1), DOWN),
        ((r, c), LEFT),
        ((r + 1, c), LEFT),
        ((r, c + 1), RIGHT),
        ((r + 1, c + 1), RIGHT),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Towers emit width-2 primary rays; relays recolor and spawn perpendicular orange rays.

    Every source ray is traced independently against the original input obstacles.
    Claims on black cells are resolved afterward so multi-source intersections become magenta.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    towers = _find_towers(g)
    claims: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    relay_hits: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    next_source_id = 0

    def add_claim(r: int, c: int, source_id: int, color: int) -> None:
        claims[(r, c)].append((source_id, color))

    def trace_primary(
        start: tuple[int, int],
        direction: tuple[int, int],
        tower_color: int,
        source_id: int,
    ) -> None:
        r, c = start
        dr, dc = direction
        while True:
            r += dr
            c += dc
            if not (0 <= r < height and 0 <= c < width):
                return
            value = int(g[r, c])
            if value == BLACK:
                add_claim(r, c, source_id, tower_color)
                continue
            if value == YELLOW:
                key = (r, c)
                hit = (tower_color, direction)
                assert key not in relay_hits, "relay hit by multiple primary rays"
                relay_hits[key] = hit
            return

    def trace_orange(origin: tuple[int, int], direction: tuple[int, int], source_id: int) -> None:
        r, c = origin
        dr, dc = direction
        while True:
            r += dr
            c += dc
            if not (0 <= r < height and 0 <= c < width):
                return
            if int(g[r, c]) != BLACK:
                return
            add_claim(r, c, source_id, ORANGE)

    for tower in towers:
        tower_color = int(tower["color"])
        for start, direction in _tower_sources(tower):
            trace_primary(start, direction, tower_color, next_source_id)
            next_source_id += 1

    for (rr, cc), (_tower_color, incoming) in relay_hits.items():
        for direction in PERPENDICULAR[incoming]:
            trace_orange((rr, cc), direction, next_source_id)
            next_source_id += 1

    out = g.copy()
    for (rr, cc), (tower_color, _incoming) in relay_hits.items():
        out[rr, cc] = int(tower_color)

    for (r, c), hits in claims.items():
        if len({source_id for source_id, _color in hits}) >= 2:
            out[r, c] = MAGENTA
        else:
            out[r, c] = int(hits[0][1])

    return out


GRID_HEIGHT_RANGE = (10, 22)
GRID_WIDTH_RANGE = (10, 22)
RELAY_SCENE_MIN_SIZE = 10
CROSS_SCENE_MIN_SIZE = 12
CROSS_RELAY_MIN_SIZE = 14
GENERATION_RETRIES = 400
SCENARIOS = ("cross_only", "single_relay", "cross_with_relays")


def _render_scene(
    height: int,
    width: int,
    towers: list[dict[str, int]],
    relays: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for tower in towers:
        r = int(tower["r"])
        c = int(tower["c"])
        color = int(tower["color"])
        assert bool(np.all(grid[r : r + 2, c : c + 2] == BLACK))
        grid[r : r + 2, c : c + 2] = color
    for rr, cc in relays:
        assert int(grid[rr, cc]) == BLACK
        grid[rr, cc] = YELLOW
    return grid


def _sample_cross_layout(
    height: int,
    width: int,
    need_vertical_relay_gap: bool,
    need_horizontal_relay_gap: bool,
) -> tuple[dict[str, int], tuple[int, int], dict[str, int], tuple[int, int]] | None:
    if height < CROSS_SCENE_MIN_SIZE or width < CROSS_SCENE_MIN_SIZE:
        return None

    ir_min = 4
    ir_max = height - 5
    ic_min = 4
    ic_max = width - 5
    if ir_min > ir_max or ic_min > ic_max:
        return None

    v_dir = random.choice((UP, DOWN))
    h_dir = random.choice((LEFT, RIGHT))
    ir = random.randint(ir_min, ir_max)
    ic = random.randint(ic_min, ic_max)

    if v_dir == DOWN:
        max_row = ir - (4 if need_vertical_relay_gap else 2)
        if max_row < 1:
            return None
        vr = random.randint(1, max_row)
    else:
        min_row = ir + (4 if need_vertical_relay_gap else 2)
        if min_row > height - 3:
            return None
        vr = random.randint(min_row, height - 3)

    if h_dir == RIGHT:
        max_col = ic - (4 if need_horizontal_relay_gap else 2)
        if max_col < 1:
            return None
        hc = random.randint(1, max_col)
    else:
        min_col = ic + (4 if need_horizontal_relay_gap else 2)
        if min_col > width - 3:
            return None
        hc = random.randint(min_col, width - 3)

    vertical = {"r": int(vr), "c": int(ic)}
    horizontal = {"r": int(ir), "c": int(hc)}
    return vertical, v_dir, horizontal, h_dir


def _single_tower_relay_candidate(
    height: int,
    width: int,
) -> tuple[dict[str, int], tuple[int, int]] | None:
    if height < RELAY_SCENE_MIN_SIZE or width < RELAY_SCENE_MIN_SIZE:
        return None

    row_choices = list(range(2, height - 3))
    col_choices = list(range(2, width - 3))
    if not row_choices or not col_choices:
        return None

    tower = {
        "r": random.choice(row_choices),
        "c": random.choice(col_choices),
        "color": random.choice(TOWER_COLORS),
    }

    candidates: list[tuple[int, int]] = []
    for start, direction in _tower_sources(tower):
        sr, sc = start
        dr, dc = direction
        for step in range(2, 7):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (0 <= rr < height and 0 <= cc < width):
                break
            if direction in (UP, DOWN):
                if not (1 <= rr <= height - 2 and 2 <= cc <= width - 3):
                    continue
            else:
                if not (2 <= rr <= height - 3 and 1 <= cc <= width - 2):
                    continue
            candidates.append((int(rr), int(cc)))

    if not candidates:
        return None

    relay = random.choice(candidates)
    return tower, relay


def _gen_cross_only() -> np.ndarray | None:
    height = random.randint(max(CROSS_SCENE_MIN_SIZE, GRID_HEIGHT_RANGE[0]), GRID_HEIGHT_RANGE[1])
    width = random.randint(max(CROSS_SCENE_MIN_SIZE, GRID_WIDTH_RANGE[0]), GRID_WIDTH_RANGE[1])
    layout = _sample_cross_layout(height, width, False, False)
    if layout is None:
        return None

    vertical, _v_dir, horizontal, _h_dir = layout
    colors = random.sample(TOWER_COLORS, 2)
    towers = [
        {"r": vertical["r"], "c": vertical["c"], "color": colors[0]},
        {"r": horizontal["r"], "c": horizontal["c"], "color": colors[1]},
    ]
    return _render_scene(height, width, towers, [])


def _gen_single_relay() -> np.ndarray | None:
    height = random.randint(max(RELAY_SCENE_MIN_SIZE, GRID_HEIGHT_RANGE[0]), GRID_HEIGHT_RANGE[1])
    width = random.randint(max(RELAY_SCENE_MIN_SIZE, GRID_WIDTH_RANGE[0]), GRID_WIDTH_RANGE[1])
    candidate = _single_tower_relay_candidate(height, width)
    if candidate is None:
        return None

    tower, relay = candidate
    return _render_scene(height, width, [tower], [relay])


def _sample_vertical_relay(
    tower: dict[str, int],
    toward_direction: tuple[int, int],
    cross_row: int,
) -> tuple[int, int] | None:
    offset = random.randint(0, 1)
    col = int(tower["c"]) + offset
    if toward_direction == DOWN:
        lo = int(tower["r"]) + 3
        hi = int(cross_row) - 1
    else:
        lo = int(cross_row) + 2
        hi = int(tower["r"]) - 2
    if lo > hi:
        return None
    return random.randint(lo, hi), col


def _sample_horizontal_relay(
    tower: dict[str, int],
    toward_direction: tuple[int, int],
    cross_col: int,
) -> tuple[int, int] | None:
    offset = random.randint(0, 1)
    row = int(tower["r"]) + offset
    if toward_direction == RIGHT:
        lo = int(tower["c"]) + 3
        hi = int(cross_col) - 1
    else:
        lo = int(cross_col) + 2
        hi = int(tower["c"]) - 2
    if lo > hi:
        return None
    return row, random.randint(lo, hi)


def _gen_cross_with_relays() -> np.ndarray | None:
    height = random.randint(max(CROSS_RELAY_MIN_SIZE, GRID_HEIGHT_RANGE[0]), GRID_HEIGHT_RANGE[1])
    width = random.randint(max(CROSS_RELAY_MIN_SIZE, GRID_WIDTH_RANGE[0]), GRID_WIDTH_RANGE[1])
    relay_mode = random.choice(("vertical", "horizontal", "both"))
    need_vertical = relay_mode in {"vertical", "both"}
    need_horizontal = relay_mode in {"horizontal", "both"}
    layout = _sample_cross_layout(height, width, need_vertical, need_horizontal)
    if layout is None:
        return None

    vertical, v_dir, horizontal, h_dir = layout
    colors = random.sample(TOWER_COLORS, 2)
    towers = [
        {"r": vertical["r"], "c": vertical["c"], "color": colors[0]},
        {"r": horizontal["r"], "c": horizontal["c"], "color": colors[1]},
    ]

    relays: list[tuple[int, int]] = []
    if need_vertical:
        relay = _sample_vertical_relay(vertical, v_dir, horizontal["r"])
        if relay is None:
            return None
        relays.append(relay)
    if need_horizontal:
        relay = _sample_horizontal_relay(horizontal, h_dir, vertical["c"])
        if relay is None:
            return None
        relays.append(relay)

    return _render_scene(height, width, towers, relays)


def _validate_scene(grid: np.ndarray, scenario: str) -> bool:
    out = main(grid)
    relay_count = int(np.count_nonzero(grid == YELLOW))
    orange_count = int(np.count_nonzero(out == ORANGE))
    magenta_count = int(np.count_nonzero(out == MAGENTA))

    if scenario == "cross_only":
        return relay_count == 0 and orange_count == 0 and magenta_count >= 4
    if scenario == "single_relay":
        return relay_count == 1 and orange_count >= 2 and magenta_count >= 1 and int(np.count_nonzero(out == YELLOW)) == 0
    if scenario == "cross_with_relays":
        return relay_count in {1, 2} and orange_count >= 2 and magenta_count >= 3 and int(np.count_nonzero(out == YELLOW)) == 0
    return False


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choice(SCENARIOS)
    if scenario == "cross_only":
        grid = _gen_cross_only()
    elif scenario == "single_relay":
        grid = _gen_single_relay()
    else:
        grid = _gen_cross_with_relays()

    if grid is None:
        return None
    if not _validate_scene(grid, scenario):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        try:
            grid = _gen_attempt()
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-ray scene")
