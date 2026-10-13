from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds, neighbors4


RED = Color.RED
BLUE = Color.BLUE
YELLOW = Color.YELLOW
GREY = Color.GREY
ROLE_COLORS = {RED, BLUE, YELLOW, GREY}

HORIZONTAL = "H"
VERTICAL = "V"


def _unit(x: int) -> int:
    return int((x > 0) - (x < 0))


def _add(p: tuple[int, int], step: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return int(p[0] + step[0] * k), int(p[1] + step[1] * k)


def _ray(origin: tuple[int, int], step: tuple[int, int], length: int) -> list[tuple[int, int]]:
    return [_add(origin, step, k) for k in range(1, int(length) + 1)]


def _fold_steps(orient: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if orient == HORIZONTAL:
        return (-1, 0), (1, 0)
    return (0, -1), (0, 1)


def _component_coords(comp: dict) -> list[tuple[int, int]]:
    return [tuple(map(int, p)) for p in comp["coords"]]


def _parse_rod(g: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    coord_set = set(coords)
    marker_grey = [p for p in coords if int(g[p]) == GREY]
    marker_blue = [p for p in coords if int(g[p]) == BLUE]
    assert len(marker_grey) == 1 and len(marker_blue) == 1, "each rod needs one grey end and one blue hinge"

    endpoint = marker_grey[0]
    hinge = marker_blue[0]

    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    assert len(rows) == 1 or len(cols) == 1, "rod must be horizontal or vertical"

    if len(rows) == 1:
        orient = HORIZONTAL
        r = next(iter(rows))
        c0, c1 = min(cols), max(cols)
        expected = {(r, c) for c in range(c0, c1 + 1)}
        assert expected == coord_set, "rod line must have no gaps"
        assert endpoint[1] in (c0, c1) and hinge[1] not in (c0, c1), "markers must be end and interior hinge"
        fixed_step = (0, _unit(endpoint[1] - hinge[1]))
    else:
        orient = VERTICAL
        c = next(iter(cols))
        r0, r1 = min(rows), max(rows)
        expected = {(r, c) for r in range(r0, r1 + 1)}
        assert expected == coord_set, "rod line must have no gaps"
        assert endpoint[0] in (r0, r1) and hinge[0] not in (r0, r1), "markers must be end and interior hinge"
        fixed_step = (_unit(endpoint[0] - hinge[0]), 0)

    assert fixed_step != (0, 0), "grey end and hinge must be distinct"
    degrees = {
        p: sum((nr, nc) in coord_set for nr, nc in neighbors4(*p))
        for p in (endpoint, hinge)
    }
    assert degrees[endpoint] == 1 and degrees[hinge] == 2, "endpoint and hinge degrees must identify the rod"

    fixed_red: list[tuple[int, int]] = []
    k = 1
    while True:
        p = _add(hinge, fixed_step, k)
        assert p in coord_set, "fixed side must be continuous"
        if p == endpoint:
            break
        assert int(g[p]) == RED, "fixed side between markers must be red"
        fixed_red.append(p)
        k += 1

    tail_step = (-fixed_step[0], -fixed_step[1])
    tail: list[tuple[int, int]] = []
    k = 1
    while True:
        p = _add(hinge, tail_step, k)
        if p not in coord_set:
            break
        assert int(g[p]) == RED, "moving tail must contain only red cells"
        tail.append(p)
        k += 1
    assert tail, "each rod must have a non-empty tail"

    covered = {endpoint, hinge, *fixed_red, *tail}
    assert covered == coord_set, "rod component contains extra cells"
    return {
        "endpoint": endpoint,
        "hinge": hinge,
        "orient": orient,
        "tail": tail,
        "tail_len": len(tail),
    }


def _extract_rods(g: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        colors=[RED, BLUE, GREY],
    )
    rods = [_parse_rod(g, _component_coords(comp)) for comp in comps]
    assert rods, "expected at least one hinged rod"
    tail_lengths = [int(r["tail_len"]) for r in rods]
    assert len(set(tail_lengths)) == len(tail_lengths), "tail lengths must be unique"
    return rods


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each hinged red rod into the yellow socket whose distance matches its tail.

    Input structure:
    - The background is the dominant color.
    - Each object is a one-cell-wide horizontal or vertical red rod with a grey
      endpoint marker at one end and a blue hinge marker inside the rod.
    - Yellow single-pixel sockets are separate from the rods, with exactly one
      socket per rod and mutually distinct red tail lengths beyond the hinges.

    Rule:
    - Keep the grey-to-blue fixed segment of every rod in place.
    - Clear the red tail on the far side of the blue hinge.
    - Redraw that tail as a perpendicular ray next to the hinge; of the two
      perpendicular directions, choose the only one whose last cell is a yellow
      socket exactly tail_len cells from the hinge. The socket cell becomes red.

    Key insight:
    The fold direction is the unique key-lock distance match between a rod tail
    length and a socket, not the direction with more open space.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))

    rods = _extract_rods(g, bg)
    sockets = {tuple(map(int, p)) for p in np.argwhere(g == YELLOW)}
    assert len(sockets) == len(rods), "socket count must match rod count"

    clear: list[tuple[int, int]] = []
    paint: list[tuple[int, int]] = []
    socket_hits: list[tuple[int, int]] = []

    for rod in rods:
        hinge = rod["hinge"]
        tail_len = int(rod["tail_len"])
        candidates = []
        for fold_step in _fold_steps(rod["orient"]):
            folded = _ray(hinge, fold_step, tail_len)
            if not all(in_bounds(g.shape, *p) for p in folded):
                continue
            if folded[-1] in sockets:
                candidates.append((fold_step, folded))

        assert len(candidates) == 1, "exactly one perpendicular socket must match the tail length"
        _fold_step, folded = candidates[0]
        for p in folded[:-1]:
            assert p not in sockets, "only the folded endpoint may be a socket"
        clear.extend(rod["tail"])
        paint.extend(folded)
        socket_hits.append(folded[-1])

    assert set(socket_hits) == sockets, "every socket must be claimed exactly once"
    assert len(set(clear)) == len(clear), "input tails must not overlap"
    assert len(set(paint)) == len(paint), "folded tails must not collide"

    clear_set = set(clear)
    for p in paint:
        value = int(g[p])
        assert p in clear_set or p in sockets or value == bg, "folded tail may only enter background or its socket"

    out = g.copy()
    for p in clear:
        out[p] = bg
    for p in paint:
        out[p] = RED
    return out


# --------------- generator ---------------

BG_COLORS = tuple(c for c in range(10) if c not in ROLE_COLORS)
GRID_H_RANGE = (15, 30)
GRID_W_RANGE = (15, 30)
ROD_COUNT_RANGE = (3, 5)
FIXED_DIST_RANGE = (2, 6)
TAIL_LEN_RANGE = (2, 8)
MAX_GENERATE_ATTEMPTS = 500
PLACE_ATTEMPTS = 160
SAMPLE_RECORD_ATTEMPTS = 1200

_RECORD_CACHE: dict[tuple[int, int], list[dict]] = {}


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    H, W = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    out.add((nr, nc))
    return out


def _space_relation(shape: tuple[int, int], orient: str, hinge: tuple[int, int], fold_step: tuple[int, int]) -> str:
    H, W = shape
    r, c = hinge
    if orient == HORIZONTAL:
        chosen = r if fold_step == (-1, 0) else H - 1 - r
        other = H - 1 - r if fold_step == (-1, 0) else r
    else:
        chosen = c if fold_step == (0, -1) else W - 1 - c
        other = W - 1 - c if fold_step == (0, -1) else c
    if chosen < other:
        return "smaller"
    if chosen > other:
        return "larger"
    return "equal"


def _make_record(
    shape: tuple[int, int],
    orient: str,
    hinge: tuple[int, int],
    fixed_step: tuple[int, int],
    fixed_dist: int,
    tail_len: int,
    fold_step: tuple[int, int],
) -> dict | None:
    endpoint = _add(hinge, fixed_step, fixed_dist)
    fixed_red = _ray(hinge, fixed_step, fixed_dist - 1)
    tail_step = (-fixed_step[0], -fixed_step[1])
    tail = _ray(hinge, tail_step, tail_len)
    folded = _ray(hinge, fold_step, tail_len)

    cells = [endpoint, hinge, *fixed_red, *tail, *folded]
    if any(not in_bounds(shape, *p) for p in cells):
        return None

    input_cells = {endpoint, hinge, *fixed_red, *tail}
    folded_set = set(folded)
    if len(input_cells) != 2 + len(fixed_red) + len(tail):
        return None
    if folded_set & input_cells:
        return None

    footprint = input_cells | folded_set
    return {
        "orient": orient,
        "hinge": hinge,
        "endpoint": endpoint,
        "fixed_red": fixed_red,
        "tail": tail,
        "tail_len": int(tail_len),
        "fold_step": fold_step,
        "folded": folded,
        "socket": folded[-1],
        "footprint": footprint,
        "space_relation": _space_relation(shape, orient, hinge, fold_step),
        "fixed_dist": int(fixed_dist),
    }


def _all_records(shape: tuple[int, int]) -> list[dict]:
    if shape in _RECORD_CACHE:
        return _RECORD_CACHE[shape]

    H, W = shape
    records: list[dict] = []
    for orient in (HORIZONTAL, VERTICAL):
        fixed_steps = ((0, -1), (0, 1)) if orient == HORIZONTAL else ((-1, 0), (1, 0))
        for r in range(H):
            for c in range(W):
                hinge = (r, c)
                for fixed_step in fixed_steps:
                    for fixed_dist in range(FIXED_DIST_RANGE[0], FIXED_DIST_RANGE[1] + 1):
                        for tail_len in range(TAIL_LEN_RANGE[0], TAIL_LEN_RANGE[1] + 1):
                            for fold_step in _fold_steps(orient):
                                rec = _make_record(shape, orient, hinge, fixed_step, fixed_dist, tail_len, fold_step)
                                if rec is not None:
                                    records.append(rec)

    _RECORD_CACHE[shape] = records
    return records


def _compatible(
    rec: dict,
    blocked: set[tuple[int, int]],
    used_tail_lengths: set[int],
) -> bool:
    if int(rec["tail_len"]) in used_tail_lengths:
        return False
    return not bool(rec["footprint"] & blocked)


def _pick_record(
    records: list[dict],
    blocked: set[tuple[int, int]],
    used_tail_lengths: set[int],
    predicate,
) -> dict | None:
    candidates = [r for r in records if predicate(r) and _compatible(r, blocked, used_tail_lengths)]
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_record(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    used_tail_lengths: set[int],
    predicate,
) -> dict | None:
    H, W = shape
    available_lengths = [n for n in range(TAIL_LEN_RANGE[0], TAIL_LEN_RANGE[1] + 1) if n not in used_tail_lengths]
    if not available_lengths:
        return None

    for _ in range(SAMPLE_RECORD_ATTEMPTS):
        orient = random.choice((HORIZONTAL, VERTICAL))
        fixed_steps = ((0, -1), (0, 1)) if orient == HORIZONTAL else ((-1, 0), (1, 0))
        fixed_step = random.choice(fixed_steps)
        fixed_dist = random.randint(FIXED_DIST_RANGE[0], FIXED_DIST_RANGE[1])
        tail_len = random.choice(available_lengths)
        fold_step = random.choice(_fold_steps(orient))
        hinge = (random.randrange(H), random.randrange(W))

        rec = _make_record(shape, orient, hinge, fixed_step, fixed_dist, tail_len, fold_step)
        if rec is None:
            continue
        if not predicate(rec):
            continue
        if not _compatible(rec, blocked, used_tail_lengths):
            continue
        return rec
    return None


def _place_records(shape: tuple[int, int], count: int) -> list[dict] | None:
    blocked: set[tuple[int, int]] = set()
    used_tail_lengths: set[int] = set()
    chosen: list[dict] = []

    requirements = [
        lambda r: r["space_relation"] == "smaller",
        lambda r: r["orient"] == HORIZONTAL,
        lambda r: r["orient"] == VERTICAL,
    ]
    random.shuffle(requirements)

    for i in range(count):
        if i < len(requirements):
            predicate = requirements[i]
        else:
            predicate = lambda _r: True

        rec = _sample_record(shape, blocked, used_tail_lengths, predicate)
        if rec is None and i >= len(requirements):
            rec = _sample_record(shape, blocked, used_tail_lengths, lambda _r: True)
        if rec is None:
            return None

        chosen.append(rec)
        used_tail_lengths.add(int(rec["tail_len"]))
        blocked |= _expanded(rec["footprint"], shape, radius=1)

    return chosen


def _paint_input(records: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for rec in records:
        grid[rec["endpoint"]] = GREY
        grid[rec["hinge"]] = BLUE
        for p in rec["fixed_red"]:
            grid[p] = RED
        for p in rec["tail"]:
            grid[p] = RED
        grid[rec["socket"]] = YELLOW
    return grid


def _expected_output(records: list[dict], grid: np.ndarray, bg: int) -> np.ndarray:
    out = grid.copy()
    for rec in records:
        for p in rec["tail"]:
            out[p] = int(bg)
        for p in rec["folded"]:
            out[p] = RED
    return out


def _candidate_socket_count(grid: np.ndarray, rod: dict, sockets: set[tuple[int, int]]) -> int:
    count = 0
    for step in _fold_steps(rod["orient"]):
        folded = _ray(rod["hinge"], step, int(rod["tail_len"]))
        if all(in_bounds(grid.shape, *p) for p in folded) and folded[-1] in sockets:
            count += 1
    return count


def _semantic_checks(grid: np.ndarray, records: list[dict], expected: np.ndarray) -> bool:
    if len({r["orient"] for r in records}) < 2:
        return False
    if not any(r["space_relation"] == "smaller" for r in records):
        return False
    if len({r["fold_step"] for r in records}) < 2:
        return False
    if len({int(r["fixed_dist"]) for r in records}) < 2:
        return False
    if np.array_equal(grid, expected):
        return False

    bg = int(get_bg(grid))
    sockets = {tuple(map(int, p)) for p in np.argwhere(grid == YELLOW)}
    try:
        rods = _extract_rods(grid, bg)
    except Exception:
        return False
    if len(rods) != len(records) or len(sockets) != len(records):
        return False
    if any(_candidate_socket_count(grid, rod, sockets) != 1 for rod in rods):
        return False

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    colors = set(map(int, np.unique(grid)))
    if not {RED, BLUE, YELLOW, GREY}.issubset(colors):
        return False
    if int(np.count_nonzero(grid == bg)) <= int(grid.size // 2):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    shape = (H, W)
    count = random.randint(*ROD_COUNT_RANGE)
    bg = random.choice(BG_COLORS)

    for _ in range(PLACE_ATTEMPTS):
        records = _place_records(shape, count)
        if records is None:
            continue
        grid = _paint_input(records, shape, bg)
        expected = _expected_output(records, grid, bg)
        if _semantic_checks(grid, records, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid hinged socket-fold input")
