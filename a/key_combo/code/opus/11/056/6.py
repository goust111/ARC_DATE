from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
WALL = 5
PAIR_COLORS = (1, 2, 3)
CELL_SIZE = 12
STAGE_SIZE = 25
CELL_ORIGINS = ((0, 0), (0, 13), (13, 0), (13, 13))
MAX_STEPS = 50
RETRY_LIMIT = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_dicts(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"coords": coords, "bbox": _bbox(coords), "area": len(coords)})
    return comps


def _rect_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _is_domino_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == 2 and (r1 - r0 + 1) * (c1 - c0 + 1) == 2


def _pair_start(upper: tuple[int, int, int, int], lower: tuple[int, int, int, int]) -> tuple[tuple[int, int], int]:
    ur0, uc0, ur1, uc1 = upper
    lr0, lc0, _lr1, lc1 = lower
    assert ur0 < lr0, "the upper rectangle must be strictly above the lower rectangle"
    if uc1 < lc0:
        return (ur1, uc1), 1
    if lc1 < uc0:
        return (ur1, uc0), -1
    raise AssertionError("same-color rectangles must be diagonally separated without column overlap")


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [color for color in sorted(set(map(int, g.ravel()))) if color not in {BG, WALL}]
    assert 2 <= len(colors) <= 3, "expected 2 or 3 foreground pair colors"
    pairs: list[dict] = []
    for color in colors:
        comps = _component_dicts(g == int(color))
        assert len(comps) == 2, "each pair color must form exactly two components"
        assert all(_is_domino_rect(comp) for comp in comps), "each component must be a 1x2 or 2x1 rectangle"
        comps.sort(key=lambda comp: comp["bbox"][0])
        upper, lower = comps
        assert upper["bbox"][0] < lower["bbox"][0], "same-color rectangles must have a unique upper object"
        start, dc = _pair_start(upper["bbox"], lower["bbox"])
        pairs.append(
            {
                "color": int(color),
                "upper": tuple(map(int, upper["bbox"])),
                "lower": tuple(map(int, lower["bbox"])),
                "start": (int(start[0]), int(start[1])),
                "dc": int(dc),
            }
        )
    pairs.sort(key=lambda pair: (pair["upper"][0], pair["upper"][1], pair["color"]))
    return pairs


def _trace_pair(
    grid: np.ndarray,
    pair: dict,
    blocked_paths: set[tuple[int, int]] | None = None,
) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocked = set() if blocked_paths is None else set(blocked_paths)
    start = pair["start"]
    dr, dc = 1, int(pair["dc"])
    upper_cells = _rect_cells(pair["upper"])
    target_cells = _rect_cells(pair["lower"])
    rect_obstacles = {(r, c) for r, c in np.argwhere((g != BG) & (g != WALL))}
    rect_obstacles -= target_cells

    path: list[tuple[int, int]] = []
    events: list[str] = []
    seen: set[tuple[tuple[int, int], int, int]] = set()
    cur = (int(start[0]), int(start[1]))

    def is_wall(cell: tuple[int, int]) -> bool:
        return _in_bounds(g.shape, cell[0], cell[1]) and int(g[cell]) == WALL

    def is_blocked(cell: tuple[int, int], *, allow_target: bool = False) -> bool:
        if not _in_bounds(g.shape, cell[0], cell[1]):
            return True
        if cell in blocked:
            return True
        if is_wall(cell):
            return True
        if cell in rect_obstacles:
            return True
        if not allow_target and cell in target_cells:
            return True
        return False

    for _step in range(MAX_STEPS):
        state = (cur, int(dr), int(dc))
        if state in seen:
            return {
                "success": False,
                "reason": "loop",
                "path": path,
                "events": events,
            }
        seen.add(state)

        nxt = (int(cur[0] + dr), int(cur[1] + dc))
        if not _in_bounds(g.shape, nxt[0], nxt[1]):
            return {"success": False, "reason": "out", "path": path, "events": events}
        if nxt in blocked:
            return {"success": False, "reason": "blocker", "path": path, "events": events}
        if nxt in target_cells:
            return {"success": True, "reason": "target", "path": path, "events": events}
        if nxt in upper_cells:
            return {"success": False, "reason": "source", "path": path, "events": events}
        if nxt in rect_obstacles:
            return {"success": False, "reason": "rect", "path": path, "events": events}

        if is_wall(nxt):
            cand_h = (int(cur[0] + dr), int(cur[1] - dc))
            if not is_blocked(cand_h, allow_target=True):
                dc = -dc
                events.append("head_h")
                continue
            cand_v = (int(cur[0] - dr), int(cur[1] + dc))
            if not is_blocked(cand_v, allow_target=True):
                dr = -dr
                events.append("head_v")
                continue
            return {"success": False, "reason": "stuck", "path": path, "events": events}

        cur = nxt
        path.append(cur)

        side_v = (int(cur[0] + dr), int(cur[1]))
        side_h = (int(cur[0]), int(cur[1] + dc))
        if is_wall(side_v):
            dr = -dr
            events.append("side_v")
        if is_wall(side_h):
            dc = -dc
            events.append("side_h")

    return {"success": False, "reason": "limit", "path": path, "events": events}


def _direct_diag_cells(pair: dict) -> list[tuple[int, int]]:
    start = pair["start"]
    dc = int(pair["dc"])
    target_cells = _rect_cells(pair["lower"])
    cells: list[tuple[int, int]] = []
    cur = (int(start[0]), int(start[1]))
    for _ in range(MAX_STEPS):
        cur = (int(cur[0] + 1), int(cur[1] + dc))
        cells.append(cur)
        if cur in target_cells:
            return cells
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color diagonal rectangle pair with a bouncing diagonal ray when the reflected path reaches the lower rectangle.

    Input structure:
    - The grid has black background, 2-3 foreground colors, and scattered grey single-cell wall pixels.
    - Each foreground color forms exactly two small solid rectangles, each of size 1x2 or 2x1.
    - For every color, the upper rectangle and lower rectangle are diagonally offset, so the upper rectangle has a unique lower facing corner toward the lower one.

    Rule:
    - Keep every original rectangle and every grey wall unchanged.
    - For each color, process pairs in reading order of the upper rectangle; emit a 45-degree ray from the upper facing corner toward the lower rectangle.
    - The ray advances one diagonal step at a time. If the next diagonal cell is a grey wall, try a head-on bounce by flipping the horizontal direction first; if that reflected next step is blocked or out of bounds, flip the vertical direction instead; if both are impossible, stop. After each normal advance, a grey wall touching the ray vertically flips the vertical direction, and a grey wall touching it horizontally flips the horizontal direction.
    - If the ray reaches any pixel of the lower rectangle within 50 steps, keep its full traversed background path in that pair color. If it leaves the grid, gets stuck, exceeds 50 steps, or runs into an earlier successful path, discard that pair's path entirely.

    Key insight:
    - The rectangles still define the color pairing and launch direction, but the actual connector is determined by simulating diagonal reflections off grey walls, with earlier successful connectors becoming obstacles for later ones.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    blocked_paths: set[tuple[int, int]] = set()

    for pair in _extract_pairs(g):
        trace = _trace_pair(g, pair, blocked_paths)
        if trace["success"]:
            for r, c in trace["path"]:
                if int(out[r, c]) == BG:
                    out[r, c] = int(pair["color"])
            blocked_paths.update(trace["path"])

    return out


TEMPLATES = {
    "head_h_success": {
        "upper": (0, 4, 1, 4),
        "lower": (5, 0, 6, 0),
        "walls": {(6, 6), (2, 3), (7, 1), (3, 6)},
    },
    "side_h_success": {
        "upper": (0, 4, 0, 5),
        "lower": (4, 0, 5, 0),
        "walls": {(3, 1), (6, 1), (1, 1), (3, 4)},
    },
    "head_v_fail": {
        "upper": (0, 3, 1, 3),
        "lower": (4, 6, 5, 6),
        "walls": {(3, 3), (3, 5), (4, 8)},
    },
    "head_v_success": {
        "upper": (1, 3, 2, 3),
        "lower": (7, 8, 8, 8),
        "walls": {(10, 7), (1, 8), (8, 2), (4, 5), (10, 5), (5, 1)},
    },
}


def _mirror_box_h(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (r0, width - 1 - c1, r1, width - 1 - c0)


def _mirror_cells_h(cells: set[tuple[int, int]], width: int) -> set[tuple[int, int]]:
    return {(r, width - 1 - c) for r, c in cells}


def _apply_template(name: str, mirror: bool) -> dict:
    spec = TEMPLATES[name]
    upper = tuple(spec["upper"])
    lower = tuple(spec["lower"])
    walls = set(spec["walls"])
    if mirror:
        upper = _mirror_box_h(upper, CELL_SIZE)
        lower = _mirror_box_h(lower, CELL_SIZE)
        walls = _mirror_cells_h(walls, CELL_SIZE)
    cells = set(walls) | _rect_cells(upper) | _rect_cells(lower)
    max_r = max(r for r, _c in cells)
    max_c = max(c for _r, c in cells)
    return {
        "upper": upper,
        "lower": lower,
        "walls": walls,
        "max_r": max_r,
        "max_c": max_c,
    }


def _shift_box(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _place_block(stage: np.ndarray, origin: tuple[int, int], color: int, name: str, mirror: bool) -> None:
    local = _apply_template(name, mirror)
    row_slack = CELL_SIZE - (local["max_r"] + 1)
    col_slack = CELL_SIZE - (local["max_c"] + 1)
    dr = origin[0] + random.randint(0, max(0, row_slack))
    dc = origin[1] + random.randint(0, max(0, col_slack))
    upper = _shift_box(local["upper"], dr, dc)
    lower = _shift_box(local["lower"], dr, dc)
    walls = _shift_cells(local["walls"], dr, dc)
    for r, c in walls:
        stage[r, c] = WALL
    for r, c in _rect_cells(upper) | _rect_cells(lower):
        stage[r, c] = int(color)


def _crop_stage(stage: np.ndarray) -> np.ndarray:
    coords = np.argwhere(stage != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(0, min(int(r0), 2))
    left = random.randint(0, min(int(c0), 2))
    bottom = random.randint(0, min(int(stage.shape[0] - 1 - r1), 2))
    right = random.randint(0, min(int(stage.shape[1] - 1 - c1), 2))
    return stage[int(r0) - top : int(r1) + bottom + 1, int(c0) - left : int(c1) + right + 1].copy()


def _direct_path_has_wall(grid: np.ndarray, pair: dict) -> bool:
    diag = _direct_diag_cells(pair)
    assert diag and diag[-1] in _rect_cells(pair["lower"]), "ordinary diagonal must reach the lower rectangle"
    return any(int(grid[r, c]) == WALL for r, c in diag[:-1])


def _validate_generated(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert _get_bg(g) == BG
    assert set(map(int, g.ravel())) <= {BG, WALL, *PAIR_COLORS}

    pairs = _extract_pairs(g)
    assert all(_direct_path_has_wall(g, pair) for pair in pairs), "every ordinary connector must be blocked by a wall"

    blocked_paths: set[tuple[int, int]] = set()
    successes = 0
    visible_head_v = False
    visible_head_h = False
    visible_side_h = False
    visible_side_v = False
    for pair in pairs:
        trace = _trace_pair(g, pair, blocked_paths)
        assert trace["reason"] in {"target", "out", "blocker", "stuck", "limit"}, "unexpected failure mode"
        visible_head_v |= "head_v" in trace["events"] and trace["success"]
        visible_head_h |= "head_h" in trace["events"] and trace["success"]
        visible_side_h |= "side_h" in trace["events"] and trace["success"]
        visible_side_v |= "side_v" in trace["events"] and trace["success"]
        if trace["success"]:
            assert trace["path"], "successful connectors must draw at least one path cell"
            successes += 1
            blocked_paths.update(trace["path"])
    assert successes >= 1, "at least one pair must connect successfully"
    assert visible_head_v, "generator must expose a successful vertical head-on branch"
    assert visible_head_h, "generator must expose a successful horizontal head-on branch"
    assert visible_side_h, "generator must expose a successful horizontal side-touch branch"
    assert visible_side_v, "generator must expose a successful vertical side-touch branch"
    assert not np.array_equal(main(g), g), "output must change"


def _gen_attempt() -> np.ndarray | None:
    stage = np.zeros((STAGE_SIZE, STAGE_SIZE), dtype=int)
    n_pairs = random.choice((2, 3))
    colors = random.sample(list(PAIR_COLORS), n_pairs)

    block_specs = [
        ("head_v_success", random.choice((False, True))),
        (random.choice(("head_h_success", "side_h_success")), random.choice((False, True))),
    ]
    if n_pairs == 3:
        third_name = "head_v_fail" if random.random() < 0.7 else random.choice(("head_h_success", "side_h_success"))
        block_specs.append((third_name, random.choice((False, True))))
    random.shuffle(block_specs)

    origins = random.sample(list(CELL_ORIGINS), n_pairs)
    for color, origin, (name, mirror) in zip(colors, origins, block_specs):
        _place_block(stage, origin, color, name, mirror)

    # Add 0-2 harmless extra walls in empty corners to widen the visual distribution
    for _ in range(random.randint(0, 2)):
        r = random.choice((0, stage.shape[0] - 1, random.randint(0, stage.shape[0] - 1)))
        c = random.choice((0, stage.shape[1] - 1, random.randint(0, stage.shape[1] - 1)))
        if int(stage[r, c]) == BG:
            stage[r, c] = WALL

    grid = _crop_stage(stage)
    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bouncing-diagonal-connector input")
