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

INPUT_COLORS = (BLUE, RED, YELLOW)
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_ORDER = ("up", "right", "down", "left")
EDGE_TO_DIR = {
    (0, 1): "up",
    (1, 2): "right",
    (2, 1): "down",
    (1, 0): "left",
}
DIR_TO_EDGE = {v: k for k, v in EDGE_TO_DIR.items()}
MIX_MAP = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}

GRID_H_RANGE = (15, 22)
GRID_W_RANGE = (15, 22)
TOWER_COUNT_RANGE = (3, 6)
GEN_ATTEMPTS = 400
SCENARIO_ATTEMPTS = 200
PLACEMENT_PADDING = 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _tower_bbox(center_r: int, center_c: int) -> tuple[int, int, int, int]:
    return center_r - 1, center_c - 1, center_r + 1, center_c + 1


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = PLACEMENT_PADDING
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _mix(existing: int, incoming: int) -> int:
    if existing == BLACK:
        return incoming
    mixed = MIX_MAP.get(frozenset((existing, incoming)))
    if mixed is not None:
        return mixed
    return existing


def _detect_towers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    covered = np.zeros_like(g, dtype=bool)
    towers: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            block = g[r0 : r0 + 3, c0 : c0 + 3]
            antennas = []
            ok = True
            for rr in range(3):
                for cc in range(3):
                    val = int(block[rr, cc])
                    if (rr, cc) in EDGE_TO_DIR:
                        if val == GREY:
                            continue
                        if val == BLACK:
                            ok = False
                            break
                        antennas.append(
                            {
                                "dir": EDGE_TO_DIR[(rr, cc)],
                                "color": val,
                                "r": r0 + rr,
                                "c": c0 + cc,
                            }
                        )
                    elif val != GREY:
                        ok = False
                        break
                if not ok:
                    break
            if not ok or not (1 <= len(antennas) <= 2):
                continue

            assert not np.any(covered[r0 : r0 + 3, c0 : c0 + 3]), "towers must not overlap"
            covered[r0 : r0 + 3, c0 : c0 + 3] = True
            antennas.sort(key=lambda a: DIR_ORDER.index(a["dir"]))
            towers.append(
                {
                    "top": r0,
                    "left": c0,
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "center": (r0 + 1, c0 + 1),
                    "antennas": antennas,
                }
            )

    assert towers, "expected at least one signal tower"
    assert np.all((g == BLACK) | covered), "input may only contain black background and signal towers"
    towers.sort(key=lambda t: (t["top"], t["left"]))
    return towers


def main(grid: np.ndarray) -> np.ndarray:
    """Broadcast each tower's antenna color in its outward direction and mix beam crossings."""
    g = np.asarray(grid, dtype=int)
    towers = _detect_towers(g)

    tower_mask = np.zeros_like(g, dtype=bool)
    for tower in towers:
        r0, c0, r1, c1 = tower["bbox"]
        tower_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    out = g.copy()
    for tower in towers:
        for antenna in tower["antennas"]:
            dr, dc = DIRS[antenna["dir"]]
            r = antenna["r"] + dr
            c = antenna["c"] + dc
            while _in_bounds(g.shape, r, c):
                if not tower_mask[r, c]:
                    out[r, c] = _mix(int(out[r, c]), int(antenna["color"]))
                r += dr
                c += dc

    return out


def _paint_tower(grid: np.ndarray, center_r: int, center_c: int, antennas: dict[str, int]) -> dict:
    r0, c0, r1, c1 = _tower_bbox(center_r, center_c)
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for direction, color in antennas.items():
        rr, cc = DIR_TO_EDGE[direction]
        grid[r0 + rr, c0 + cc] = int(color)
    return {
        "center": (center_r, center_c),
        "bbox": (r0, c0, r1, c1),
        "antennas": dict(antennas),
    }


def _can_place_tower(
    grid_shape: tuple[int, int], placed: list[dict], center_r: int, center_c: int, pad: int = PLACEMENT_PADDING
) -> bool:
    h, w = grid_shape
    r0, c0, r1, c1 = _tower_bbox(center_r, center_c)
    if not (0 <= r0 and 0 <= c0 and r1 < h and c1 < w):
        return False
    bbox = (r0, c0, r1, c1)
    return all(not _boxes_too_close(bbox, tower["bbox"], pad=pad) for tower in placed)


def _try_add_tower(grid: np.ndarray, placed: list[dict], center_r: int, center_c: int, antennas: dict[str, int]) -> bool:
    if not _can_place_tower(grid.shape, placed, center_r, center_c):
        return False
    placed.append(_paint_tower(grid, center_r, center_c, antennas))
    return True


def _random_color() -> int:
    return int(random.choice(INPUT_COLORS))


def _augment_antennas(required: dict[str, int], extra_dirs: tuple[str, ...] | None = None) -> dict[str, int]:
    antennas = dict(required)
    if len(antennas) == 2 or random.random() >= 0.55:
        return antennas
    remaining = [d for d in DIR_ORDER if d not in antennas]
    if extra_dirs is not None:
        filtered = [d for d in remaining if d in extra_dirs]
        if filtered:
            remaining = filtered
    antennas[random.choice(remaining)] = _random_color()
    return antennas


def _rand_between(lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _place_pair_or_same(grid: np.ndarray, placed: list[dict], same_color: bool) -> dict | None:
    h, w = grid.shape
    for _ in range(SCENARIO_ATTEMPTS):
        cross_r = random.randint(4, h - 5)
        cross_c = random.randint(4, w - 5)

        left_or_right = random.choice(("left", "right"))
        up_or_down = random.choice(("up", "down"))
        hc = _rand_between(1, cross_c - 3) if left_or_right == "left" else _rand_between(cross_c + 3, w - 2)
        vr = _rand_between(1, cross_r - 3) if up_or_down == "up" else _rand_between(cross_r + 3, h - 2)
        if hc is None or vr is None:
            continue

        if same_color:
            color_h = color_v = _random_color()
            expected = color_h
        else:
            pair = random.choice(tuple(MIX_MAP.keys()))
            color_h, color_v = tuple(pair)
            if random.random() < 0.5:
                color_h, color_v = color_v, color_h
            expected = MIX_MAP[frozenset((color_h, color_v))]

        horizontal = _augment_antennas({"right" if left_or_right == "left" else "left": color_h})
        vertical = _augment_antennas({"down" if up_or_down == "up" else "up": color_v})

        towers = [
            ((cross_r, hc), horizontal),
            ((vr, cross_c), vertical),
        ]
        if any(not _can_place_tower(grid.shape, placed, r, c) for (r, c), _ in towers):
            continue

        local = []
        ok = True
        for (r, c), antennas in towers:
            if not _can_place_tower(grid.shape, placed + local, r, c):
                ok = False
                break
            local.append({"bbox": _tower_bbox(r, c)})
        if not ok:
            continue

        for (r, c), antennas in towers:
            _try_add_tower(grid, placed, r, c, antennas)
        return {"cross": (cross_r, cross_c), "expected": expected}
    return None


def _place_triple(grid: np.ndarray, placed: list[dict]) -> dict | None:
    h, w = grid.shape
    for _ in range(SCENARIO_ATTEMPTS):
        cross_r = random.randint(4, h - 5)
        cross_c = random.randint(4, w - 5)
        left_c = _rand_between(1, cross_c - 3)
        right_c = _rand_between(cross_c + 3, w - 2)
        vertical_side = random.choice(("up", "down"))
        vertical_r = _rand_between(1, cross_r - 3) if vertical_side == "up" else _rand_between(cross_r + 3, h - 2)
        if left_c is None or right_c is None or vertical_r is None:
            continue

        colors = list(INPUT_COLORS)
        random.shuffle(colors)
        towers = [
            ((cross_r, left_c), _augment_antennas({"right": colors[0]})),
            ((cross_r, right_c), _augment_antennas({"left": colors[1]})),
            ((vertical_r, cross_c), _augment_antennas({"down" if vertical_side == "up" else "up": colors[2]})),
        ]

        local = []
        ok = True
        for (r, c), _ in towers:
            if not _can_place_tower(grid.shape, placed + local, r, c):
                ok = False
                break
            local.append({"bbox": _tower_bbox(r, c)})
        if not ok:
            continue

        for (r, c), antennas in towers:
            _try_add_tower(grid, placed, r, c, antennas)
        return {"cross": (cross_r, cross_c)}
    return None


def _place_blocker_line(grid: np.ndarray, placed: list[dict]) -> bool:
    h, w = grid.shape
    for _ in range(SCENARIO_ATTEMPTS):
        row = random.randint(1, h - 2)
        left_to_right = random.choice((True, False))
        if left_to_right:
            src_c = _rand_between(1, w - 10)
            block_c = _rand_between(src_c + 5 if src_c is not None else 1, w - 2) if src_c is not None else None
            if src_c is None or block_c is None:
                continue
            source = ((row, src_c), _augment_antennas({"right": _random_color()}))
            blocker = ((row, block_c), _augment_antennas({random.choice(("up", "down")): _random_color()}, extra_dirs=("up", "down")))
        else:
            src_c = _rand_between(9, w - 2)
            block_c = _rand_between(1, src_c - 5) if src_c is not None else None
            if src_c is None or block_c is None:
                continue
            source = ((row, src_c), _augment_antennas({"left": _random_color()}))
            blocker = ((row, block_c), _augment_antennas({random.choice(("up", "down")): _random_color()}, extra_dirs=("up", "down")))

        towers = [source, blocker]
        local = []
        ok = True
        for (r, c), _ in towers:
            if not _can_place_tower(grid.shape, placed + local, r, c):
                ok = False
                break
            local.append({"bbox": _tower_bbox(r, c)})
        if not ok:
            continue

        for (r, c), antennas in towers:
            _try_add_tower(grid, placed, r, c, antennas)
        return True
    return False


def _place_random_tower(grid: np.ndarray, placed: list[dict]) -> bool:
    h, w = grid.shape
    for _ in range(SCENARIO_ATTEMPTS):
        center_r = random.randint(1, h - 2)
        center_c = random.randint(1, w - 2)
        antenna_count = random.choice((1, 2))
        dirs = random.sample(DIR_ORDER, k=antenna_count)
        antennas = {direction: _random_color() for direction in dirs}
        if _try_add_tower(grid, placed, center_r, center_c, antennas):
            return True
    return False


def _generated_grid_ok(grid: np.ndarray, scenario: dict) -> bool:
    towers = _detect_towers(grid)
    if not (TOWER_COUNT_RANGE[0] <= len(towers) <= TOWER_COUNT_RANGE[1]):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    cross_r, cross_c = scenario["cross"]
    if scenario.get("expected") is not None and int(out[cross_r, cross_c]) != int(scenario["expected"]):
        return False

    if scenario.get("triple"):
        if int(out[cross_r, cross_c]) == BLACK:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    placed: list[dict] = []

    scenario_kind = random.choice(("pair_mix", "same_color", "triple"))
    if scenario_kind == "pair_mix":
        scenario = _place_pair_or_same(grid, placed, same_color=False)
    elif scenario_kind == "same_color":
        scenario = _place_pair_or_same(grid, placed, same_color=True)
    else:
        scenario = _place_triple(grid, placed)
        if scenario is not None:
            scenario["triple"] = True
    if scenario is None:
        return None

    if random.random() < 0.5:
        _place_blocker_line(grid, placed)

    target_count = random.randint(max(TOWER_COUNT_RANGE[0], len(placed)), TOWER_COUNT_RANGE[1])
    while len(placed) < target_count:
        if not _place_random_tower(grid, placed):
            return None

    if not _generated_grid_ok(grid, scenario):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower input")
