from __future__ import annotations

import numpy as np


BG = 0
SIDES = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# T-shaped 5-cell arrows, expressed relative to the tip pixel.
ARROW_OFFSETS = {
    "up": ((0, 0), (1, 0), (2, 0), (2, -1), (2, 1)),
    "down": ((0, 0), (-1, 0), (-2, 0), (-2, -1), (-2, 1)),
    "left": ((0, 0), (0, 1), (0, 2), (-1, 2), (1, 2)),
    "right": ((0, 0), (0, -1), (0, -2), (-1, -2), (1, -2)),
}


def _normalized_shape(direction: str) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    offsets = ARROW_OFFSETS[direction]
    min_r = min(dr for dr, _ in offsets)
    min_c = min(dc for _, dc in offsets)
    norm = frozenset((dr - min_r, dc - min_c) for dr, dc in offsets)
    tip = (-min_r, -min_c)
    return norm, tip


SHAPE_TO_DIRECTION: dict[frozenset[tuple[int, int]], str] = {}
TIP_IN_BBOX: dict[str, tuple[int, int]] = {}
for _direction in SIDES:
    _shape, _tip = _normalized_shape(_direction)
    SHAPE_TO_DIRECTION[_shape] = _direction
    TIP_IN_BBOX[_direction] = _tip


def _majority_color(values: np.ndarray) -> int:
    counts: dict[int, int] = {}
    for value in values.tolist():
        value = int(value)
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _edge_side_by_color(grid: np.ndarray) -> dict[int, str]:
    h, w = grid.shape
    if h < 3 or w < 3:
        return {}
    top = _majority_color(grid[0, 1:-1])
    bottom = _majority_color(grid[h - 1, 1:-1])
    left = _majority_color(grid[1:-1, 0])
    right = _majority_color(grid[1:-1, w - 1])
    return {
        int(top): "up",
        int(bottom): "down",
        int(left): "left",
        int(right): "right",
    }


def _cells_from_tip(tip: tuple[int, int], direction: str) -> np.ndarray:
    r0, c0 = tip
    cells = [(int(r0 + dr), int(c0 + dc)) for dr, dc in ARROW_OFFSETS[direction]]
    return np.array(cells, dtype=int)


def _tip_bounds(direction: str, h: int, w: int, require_move: bool = False) -> tuple[int, int, int, int] | None:
    if direction == "up":
        r0, r1 = 1 + int(require_move), h - 4
        c0, c1 = 2, w - 3
    elif direction == "down":
        r0, r1 = 3, h - 2 - int(require_move)
        c0, c1 = 2, w - 3
    elif direction == "left":
        r0, r1 = 2, h - 3
        c0, c1 = 1 + int(require_move), w - 4
    else:
        r0, r1 = 2, h - 3
        c0, c1 = 3, w - 2 - int(require_move)
    if r0 > r1 or c0 > c1:
        return None
    return (int(r0), int(r1), int(c0), int(c1))


def _target_tip(direction: str, lane: int, h: int, w: int) -> tuple[int, int]:
    if direction == "up":
        return (1, int(lane))
    if direction == "down":
        return (h - 2, int(lane))
    if direction == "left":
        return (int(lane), 1)
    return (int(lane), w - 2)


def _steps_to_target(tip: tuple[int, int], direction: str, h: int, w: int) -> int:
    r, c = tip
    if direction == "up":
        return int(r - 1)
    if direction == "down":
        return int((h - 2) - r)
    if direction == "left":
        return int(c - 1)
    return int((w - 2) - c)


def _lane_of(direction: str, tip: tuple[int, int]) -> int:
    return int(tip[1] if direction in ("up", "down") else tip[0])


def _interior_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if visited[r, c] or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR_VECTORS.values():
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            components.append({"color": color, "coords": np.array(coords, dtype=int)})
    return components


def _parse_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _interior_components(grid):
        coords = comp["coords"]
        if int(coords.shape[0]) != 5:
            continue
        min_r = int(coords[:, 0].min())
        min_c = int(coords[:, 1].min())
        norm = frozenset((int(r - min_r), int(c - min_c)) for r, c in coords)
        direction = SHAPE_TO_DIRECTION.get(norm)
        if direction is None:
            continue
        tip_r, tip_c = TIP_IN_BBOX[direction]
        arrows.append(
            {
                "color": int(comp["color"]),
                "direction": direction,
                "tip": (int(min_r + tip_r), int(min_c + tip_c)),
                "coords": coords.astype(int, copy=False),
            }
        )
    return arrows


def _simulate_moves(
    grid: np.ndarray,
    arrows: list[dict[str, object]],
    side_by_color: dict[int, str],
    return_state: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]]] | np.ndarray:
    h, w = grid.shape
    out = np.array(grid, dtype=int, copy=True)
    movers: list[dict[str, object]] = []
    state: list[dict[str, object]] = []

    for arrow in arrows:
        target_side = side_by_color.get(int(arrow["color"]))
        is_mover = target_side is not None and str(arrow["direction"]) == target_side
        max_steps = _steps_to_target(tuple(arrow["tip"]), str(arrow["direction"]), h, w) if is_mover else 0
        record = {
            "color": int(arrow["color"]),
            "direction": str(arrow["direction"]),
            "tip": tuple(arrow["tip"]),
            "is_mover": bool(is_mover),
            "max_steps": int(max_steps),
            "actual_steps": 0,
            "final_tip": tuple(arrow["tip"]),
        }
        state.append(record)
        if is_mover:
            movers.append({"arrow": arrow, "state": record})
            coords = arrow["coords"]
            out[coords[:, 0], coords[:, 1]] = BG

    occupied = {tuple(map(int, pos)) for pos in np.argwhere(out != BG)}
    movers.sort(
        key=lambda item: (
            int(item["state"]["max_steps"]),
            int(item["state"]["tip"][0]),
            int(item["state"]["tip"][1]),
        )
    )

    for item in movers:
        arrow = item["arrow"]
        record = item["state"]
        direction = str(arrow["direction"])
        dr, dc = DIR_VECTORS[direction]
        coords = arrow["coords"].astype(int, copy=False)
        best = 0
        for step in range(1, int(record["max_steps"]) + 1):
            shifted = coords + np.array([[dr * step, dc * step]], dtype=int)
            cells = [tuple(map(int, cell)) for cell in shifted.tolist()]
            if any(cell in occupied for cell in cells):
                break
            best = step
        shifted = coords + np.array([[dr * best, dc * best]], dtype=int)
        for rr, cc in shifted.tolist():
            out[int(rr), int(cc)] = int(arrow["color"])
            occupied.add((int(rr), int(cc)))
        tip_r, tip_c = tuple(arrow["tip"])
        record["actual_steps"] = int(best)
        record["final_tip"] = (int(tip_r + dr * best), int(tip_c + dc * best))

    if return_state:
        return out, state
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each color-matched arrow straight toward its matching border, but only when its pointing direction agrees.

    Input structure:
    - Black background.
    - The four grid edges carry one-pixel border lines in four distinct colors.
    - The interior contains monochrome 5-cell T-shaped arrows whose color matches exactly one border side.

    Rule:
    - Match each arrow color to the border side with the same color.
    - Only arrows already pointing toward that same side are allowed to move.
    - Move each eligible arrow straight ahead until its tip is one cell from the matching border, or until the next step would overlap an occupied foreground pixel; nearer eligible arrows claim space first and farther ones stop just before conflict.

    Key insight:
    - Color determines the destination side, but orientation independently decides whether the arrow moves at all.
    """
    g = np.asarray(grid, dtype=int)
    side_by_color = _edge_side_by_color(g)
    arrows = _parse_arrows(g)
    return _simulate_moves(g, arrows, side_by_color, return_state=False)


GEN_RETRIES = 500
GRID_MIN = 12
GRID_MAX = 24
ARROW_COUNT_MIN = 4
ARROW_COUNT_MAX = 8
CONFLICT_PROBABILITY = 0.5
PLACEMENT_TRIES = 120


def _inflated_cells(cells: np.ndarray, h: int, w: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells.tolist():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 1 <= rr < h - 1 and 1 <= cc < w - 1:
                    blocked.add((rr, cc))
    return blocked


def _render_grid(h: int, w: int, side_colors: dict[str, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = int(side_colors["up"])
    grid[h - 1, :] = int(side_colors["down"])
    grid[1 : h - 1, 0] = int(side_colors["left"])
    grid[1 : h - 1, w - 1] = int(side_colors["right"])
    for spec in specs:
        cells = _cells_from_tip(tuple(spec["tip"]), str(spec["direction"]))
        grid[cells[:, 0], cells[:, 1]] = int(spec["color"])
    return grid


def _register_spec(
    specs: list[dict[str, object]],
    blocked: set[tuple[int, int]],
    spec: dict[str, object],
    h: int,
    w: int,
) -> None:
    specs.append(spec)
    cells = _cells_from_tip(tuple(spec["tip"]), str(spec["direction"]))
    blocked.update(_inflated_cells(cells, h, w))


def _tip_from_distance(direction: str, lane: int, dist: int, h: int, w: int) -> tuple[int, int]:
    tr, tc = _target_tip(direction, lane, h, w)
    dr, dc = DIR_VECTORS[direction]
    return (int(tr - dr * dist), int(tc - dc * dist))


def _place_conflict_pair(
    rng: np.random.Generator,
    h: int,
    w: int,
    side_colors: dict[str, int],
    specs: list[dict[str, object]],
    blocked: set[tuple[int, int]],
) -> tuple[str, int] | None:
    directions = list(SIDES)
    rng.shuffle(directions)
    for direction in directions:
        bounds = _tip_bounds(direction, h, w, require_move=True)
        if bounds is None:
            continue
        r0, r1, c0, c1 = bounds
        max_dist = (h - 5) if direction in ("up", "down") else (w - 5)
        if max_dist < 5:
            continue
        lane_low, lane_high = (c0, c1) if direction in ("up", "down") else (r0, r1)
        for _ in range(PLACEMENT_TRIES):
            lane = int(rng.integers(lane_low, lane_high + 1))
            near_dist = int(rng.integers(1, max_dist - 2))
            far_low = near_dist + 3
            if far_low > max_dist:
                continue
            far_dist = int(rng.integers(far_low, max_dist + 1))
            pair: list[dict[str, object]] = []
            pair_blocked = set(blocked)
            ok = True
            for dist in (near_dist, far_dist):
                tip = _tip_from_distance(direction, lane, dist, h, w)
                cells = _cells_from_tip(tip, direction)
                if any((int(r), int(c)) in pair_blocked for r, c in cells.tolist()):
                    ok = False
                    break
                pair.append({"color": int(side_colors[direction]), "direction": direction, "tip": tip})
                pair_blocked.update(_inflated_cells(cells, h, w))
            if not ok:
                continue
            for spec in pair:
                _register_spec(specs, blocked, spec, h, w)
            return (direction, lane)
    return None


def _random_spec(
    rng: np.random.Generator,
    h: int,
    w: int,
    side_colors: dict[str, int],
    move_flag: bool,
    blocked: set[tuple[int, int]],
    mover_lanes: set[tuple[str, int]],
) -> dict[str, object] | None:
    sides = list(SIDES)
    rng.shuffle(sides)
    for target_side in sides:
        color = int(side_colors[target_side])
        if move_flag:
            direction_options = [target_side]
        else:
            direction_options = [side for side in SIDES if side != target_side]
            rng.shuffle(direction_options)
        for direction in direction_options:
            bounds = _tip_bounds(direction, h, w, require_move=move_flag)
            if bounds is None:
                continue
            r0, r1, c0, c1 = bounds
            for _ in range(PLACEMENT_TRIES):
                tip = (
                    int(rng.integers(r0, r1 + 1)),
                    int(rng.integers(c0, c1 + 1)),
                )
                lane = _lane_of(direction, tip)
                if move_flag and (direction, lane) in mover_lanes:
                    continue
                cells = _cells_from_tip(tip, direction)
                if any((int(r), int(c)) in blocked for r, c in cells.tolist()):
                    continue
                return {"color": color, "direction": direction, "tip": tip}
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    palette = rng.choice(np.arange(1, 10, dtype=int), size=4, replace=False).tolist()
    side_colors = {side: int(color) for side, color in zip(SIDES, palette)}

    total_arrows = int(rng.integers(ARROW_COUNT_MIN, ARROW_COUNT_MAX + 1))
    conflict_requested = bool(rng.random() < CONFLICT_PROBABILITY)

    n_movers = total_arrows // 2
    if total_arrows % 2 == 1 and rng.random() < 0.5:
        n_movers += 1
    n_movers = max(1, min(total_arrows - 1, n_movers))
    if conflict_requested:
        n_movers = max(n_movers, 2)

    specs: list[dict[str, object]] = []
    blocked: set[tuple[int, int]] = set()
    mover_lanes: set[tuple[str, int]] = set()

    if conflict_requested:
        key = _place_conflict_pair(rng, h, w, side_colors, specs, blocked)
        if key is None:
            return None
        mover_lanes.add(key)

    remaining_movers = n_movers - len(specs)
    move_flags = [True] * remaining_movers + [False] * (total_arrows - len(specs) - remaining_movers)
    rng.shuffle(move_flags)

    for move_flag in move_flags:
        spec = _random_spec(rng, h, w, side_colors, move_flag, blocked, mover_lanes)
        if spec is None:
            return None
        _register_spec(specs, blocked, spec, h, w)
        if move_flag:
            mover_lanes.add((str(spec["direction"]), _lane_of(str(spec["direction"]), tuple(spec["tip"]))))

    grid = _render_grid(h, w, side_colors, specs)
    parsed = _parse_arrows(grid)
    if len(parsed) != len(specs):
        return None

    side_by_color = _edge_side_by_color(grid)
    out, state = _simulate_moves(grid, parsed, side_by_color, return_state=True)
    assert isinstance(out, np.ndarray)

    movers = [record for record in state if bool(record["is_mover"])]
    stalled = [record for record in movers if int(record["actual_steps"]) < int(record["max_steps"])]
    if len(movers) != n_movers:
        return None
    if not any(int(record["actual_steps"]) > 0 for record in movers):
        return None
    if not any(not bool(record["is_mover"]) for record in state):
        return None
    if conflict_requested:
        if not stalled:
            return None
    elif stalled:
        return None

    if len(_parse_arrows(out)) != len(specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-border puzzle")
