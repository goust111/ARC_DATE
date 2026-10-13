from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
SPARK = 3
HINGE = 4
ENDPOINT = 5

ROD_COLORS = {RED, HINGE, ENDPOINT}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 16
GRID_MAX = 28
CHAIN_CHOICES = [2, 3, 3, 4]
DISTRACTOR_CHOICES = [1, 1, 2, 2, 3]
FIXED_LEN_RANGE = (2, 6)
TAIL_LEN_RANGE = (2, 5)
MAX_GENERATE_ATTEMPTS = 400
MAX_PLACE_TRIES = 600


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color, matching arc_utils conventions."""
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r - 1), int(c)
    yield int(r + 1), int(c)
    yield int(r), int(c - 1)
    yield int(r), int(c + 1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean mask as (N,2) coord arrays."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(m[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(m.shape, nr, nc):
                        continue
                    if not bool(m[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _ray_cells(origin: tuple[int, int], step: tuple[int, int], count: int) -> list[tuple[int, int]]:
    r, c = map(int, origin)
    dr, dc = map(int, step)
    return [(int(r + dr * k), int(c + dc * k)) for k in range(1, int(count) + 1)]


def _touch_pairs(
    source_cells: set[tuple[int, int]] | list[tuple[int, int]],
    target_cells: set[tuple[int, int]] | list[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    source = {tuple(map(int, cell)) for cell in source_cells}
    target = {tuple(map(int, cell)) for cell in target_cells}
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for cell in source:
        for nbr in neighbors4(*cell):
            if nbr in target:
                pairs.append((cell, nbr))
    return pairs


def _touches(
    source_cells: set[tuple[int, int]] | list[tuple[int, int]],
    target_cells: set[tuple[int, int]] | list[tuple[int, int]],
) -> bool:
    return bool(_touch_pairs(source_cells, target_cells))


def _expanded4(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in neighbors4(r, c):
            if in_bounds(shape, nr, nc):
                out.add((int(nr), int(nc)))
    return out


def _fold_step(shape: tuple[int, int], orient: str, hinge: tuple[int, int]) -> tuple[int, int] | None:
    h, w = map(int, shape)
    r, c = map(int, hinge)
    if orient == "h":
        up = int(r)
        down = int(h - 1 - r)
        if up == down:
            return None
        return (-1, 0) if up > down else (1, 0)
    left = int(c)
    right = int(w - 1 - c)
    if left == right:
        return None
    return (0, -1) if left > right else (0, 1)


def _make_record(
    shape: tuple[int, int],
    orient: str,
    hinge: tuple[int, int],
    fixed_sign: int,
    fixed_len: int,
    tail_len: int,
) -> dict[str, object] | None:
    assert orient in {"h", "v"}
    fixed_len = int(fixed_len)
    tail_len = int(tail_len)
    if fixed_len < 2 or tail_len < 1:
        return None

    fixed_step = (0, int(fixed_sign)) if orient == "h" else (int(fixed_sign), 0)
    tail_step = (-fixed_step[0], -fixed_step[1])
    fold_step = _fold_step(shape, orient, hinge)
    if fold_step is None:
        return None

    fixed = _ray_cells(hinge, fixed_step, fixed_len - 1)
    tail = _ray_cells(hinge, tail_step, tail_len)
    folded = _ray_cells(hinge, fold_step, tail_len)
    endpoint = fixed[-1]
    fixed_red = fixed[:-1]

    all_cells = [tuple(map(int, hinge)), *fixed, *tail, *folded]
    if any(not in_bounds(shape, r, c) for r, c in all_cells):
        return None

    input_cells = {tuple(map(int, hinge)), endpoint, *fixed_red, *tail}
    folded_set = set(folded)
    envelope = input_cells | folded_set
    if len(input_cells) != 1 + len(fixed_red) + 1 + len(tail):
        return None
    if len(envelope) != len(input_cells) + len(folded_set):
        return None

    return {
        "orient": orient,
        "hinge": tuple(map(int, hinge)),
        "endpoint": tuple(map(int, endpoint)),
        "fixed_red": list(fixed_red),
        "tail": list(tail),
        "folded": list(folded),
        "tail_set": set(tail),
        "input_set": input_cells,
        "non_tail_set": input_cells - set(tail),
        "envelope": envelope,
        "fixed_len": fixed_len,
        "tail_len": tail_len,
    }


def _parse_rod(grid: np.ndarray, coords: np.ndarray) -> dict[str, object]:
    coord_list = [tuple(map(int, cell)) for cell in np.asarray(coords, dtype=int)]
    coord_set = set(coord_list)
    assert coord_set, "empty rod component"

    hinges = [cell for cell in coord_list if int(grid[cell]) == HINGE]
    endpoints = [cell for cell in coord_list if int(grid[cell]) == ENDPOINT]
    reds = [cell for cell in coord_list if int(grid[cell]) == RED]
    assert len(hinges) == 1 and len(endpoints) == 1, "each rod needs one hinge and one endpoint"
    assert len(reds) == len(coord_list) - 2, "all non-marker cells must be red"

    hinge = hinges[0]
    endpoint = endpoints[0]
    rows = {r for r, _c in coord_list}
    cols = {c for _r, c in coord_list}
    assert len(rows) == 1 or len(cols) == 1, "rod must be straight"
    orient = "h" if len(rows) == 1 else "v"

    if orient == "h":
        assert hinge[0] == endpoint[0]
        fixed_sign = int(np.sign(endpoint[1] - hinge[1]))
    else:
        assert hinge[1] == endpoint[1]
        fixed_sign = int(np.sign(endpoint[0] - hinge[0]))
    assert fixed_sign != 0, "hinge and endpoint must differ"

    fixed_step = (0, fixed_sign) if orient == "h" else (fixed_sign, 0)
    tail_step = (-fixed_step[0], -fixed_step[1])
    fold_step = _fold_step(grid.shape, orient, hinge)
    assert fold_step is not None, "fold direction must be unique"

    fixed_cells: list[tuple[int, int]] = []
    step = 1
    while True:
        cell = (int(hinge[0] + fixed_step[0] * step), int(hinge[1] + fixed_step[1] * step))
        if cell not in coord_set:
            break
        fixed_cells.append(cell)
        step += 1
    assert fixed_cells and fixed_cells[-1] == endpoint, "endpoint must be at the fixed-side end"
    assert all(int(grid[cell]) == RED for cell in fixed_cells[:-1]), "fixed arm must be red before the endpoint"

    tail: list[tuple[int, int]] = []
    step = 1
    while True:
        cell = (int(hinge[0] + tail_step[0] * step), int(hinge[1] + tail_step[1] * step))
        if cell not in coord_set:
            break
        tail.append(cell)
        step += 1
    assert tail, "each rod must have a movable tail"
    assert all(int(grid[cell]) == RED for cell in tail), "tail must be red"

    expected = {hinge, *fixed_cells, *tail}
    assert expected == coord_set, "rod must be gap-free and collinear"

    folded = _ray_cells(hinge, fold_step, len(tail))
    assert all(in_bounds(grid.shape, r, c) for r, c in folded)

    return {
        "orient": orient,
        "hinge": hinge,
        "endpoint": endpoint,
        "fixed_red": fixed_cells[:-1],
        "tail": tail,
        "folded": folded,
        "tail_set": set(tail),
        "input_set": set(coord_set),
        "non_tail_set": coord_set - set(tail),
        "envelope": coord_set | set(folded),
    }


def _solve(grid: np.ndarray, *, strict: bool) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == BG, "background must be black"

    spark_cells = [tuple(map(int, cell)) for cell in np.argwhere(g == SPARK)]
    assert len(spark_cells) == 1, "expected exactly one spark"
    spark = spark_cells[0]

    rods = [_parse_rod(g, comp) for comp in components_mask(np.isin(g, list(ROD_COLORS)))]
    assert rods, "expected at least one rod"

    initial: list[int] = []
    for idx, rod in enumerate(rods):
        if _touches({spark}, rod["non_tail_set"]):
            raise AssertionError("spark may only touch a free tail")
        if _touches({spark}, rod["tail_set"]):
            initial.append(int(idx))
    if strict:
        assert len(initial) == 1, "spark must touch exactly one free tail"
    else:
        assert initial, "spark must ignite at least one rod"

    out = g.copy()
    fired: list[int] = []
    fired_set: set[int] = set()
    queue = list(initial)

    while queue:
        idx = int(queue.pop(0))
        if idx in fired_set:
            continue
        rod = rods[idx]
        for r, c in rod["tail"]:
            assert int(out[r, c]) == RED, "tail to clear must still be present"
            out[r, c] = bg
        for r, c in rod["folded"]:
            if strict:
                assert int(out[r, c]) == bg, "folded tail must land on empty background"
            out[r, c] = RED
        fired_set.add(idx)
        fired.append(idx)

        source = set(rod["folded"])
        for j, other in enumerate(rods):
            if j in fired_set:
                continue
            if _touches(source, other["tail_set"]):
                queue.append(int(j))

    return out, {"rods": rods, "spark": spark, "fired": fired}


def _fold_all(grid: np.ndarray, rods: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for rod in rods:
        for r, c in rod["tail"]:
            out[r, c] = BG
        for r, c in rod["folded"]:
            out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold only the rods reached by the spark-triggered cascade, keeping untouched rods unchanged.

    Input structure:
    - The background is black.
    - Several disconnected horizontal or vertical rods are drawn with red body pixels,
      one grey fixed endpoint, and one internal yellow hinge on each rod.
    - A single green spark pixel is 4-neighbor adjacent to exactly one rod's free-tail
      side, meaning the side beyond that rod's hinge away from its grey endpoint.

    Rule:
    - A lit rod keeps its grey-endpoint-to-hinge segment fixed, removes the red tail on
      the far side of the hinge, and redraws that tail with the same length on the
      perpendicular ray that points toward the grid edge with more space.
    - Only the newly landed red tail pixels act as the next fire source.
    - Any still-unfolded rod whose free tail is 4-neighbor adjacent to those newly
      landed pixels also folds; repeat until no new rod is reached.
    - The spark, hinges, endpoints, and every unreached rod remain unchanged.

    Key insight:
    - This is not "fold every rod"; the green spark starts a contact cascade, and only
      new folded-tail pixels can transmit the fire to another rod's free-tail segment.
    """
    out, _info = _solve(np.asarray(grid, dtype=int), strict=True)
    return out


def _input_conflicts(
    shape: tuple[int, int],
    new_cells: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
) -> bool:
    if not occupied_input:
        return False
    return bool(new_cells & _expanded4(occupied_input, shape))


def _random_free_record(
    shape: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    occupied_all: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = map(int, shape)
    for _ in range(MAX_PLACE_TRIES):
        orient = random.choice(["h", "v"])
        fixed_sign = random.choice([-1, 1])
        fixed_len = random.randint(*FIXED_LEN_RANGE)
        tail_len = random.randint(*TAIL_LEN_RANGE)
        hinge = (random.randrange(h), random.randrange(w))
        rec = _make_record(shape, orient, hinge, fixed_sign, fixed_len, tail_len)
        if rec is None:
            continue
        if rec["envelope"] & occupied_all:
            continue
        if _input_conflicts(shape, rec["input_set"], occupied_input):
            continue
        if _touches(rec["folded"], occupied_all):
            continue
        return rec
    return None


def _choose_spark(
    shape: tuple[int, int],
    rec: dict[str, object],
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], bool] | None:
    tail = list(rec["tail"])
    tail_set = set(tail)
    candidates: list[tuple[tuple[int, int], tuple[int, int], bool]] = []

    for idx, cell in enumerate(tail):
        for dr, dc in DIR4:
            spark = (int(cell[0] + dr), int(cell[1] + dc))
            if not in_bounds(shape, *spark):
                continue
            if spark in occupied or spark in rec["envelope"]:
                continue
            touched = {nbr for nbr in neighbors4(*spark) if nbr in rec["input_set"]}
            if not touched or not touched <= tail_set:
                continue
            if len(_touch_pairs({spark}, tail_set)) != 1:
                continue
            interior = idx < len(tail) - 1
            candidates.append((spark, cell, interior))

    if not candidates:
        return None

    interior_choices = [cand for cand in candidates if cand[2]]
    if interior_choices and random.random() < 0.7:
        spark, _cell, interior = random.choice(interior_choices)
        return spark, interior
    spark, _cell, interior = random.choice(candidates)
    return spark, interior


def _contact_record(
    shape: tuple[int, int],
    source: set[tuple[int, int]],
    forbidden_sources: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    occupied_all: set[tuple[int, int]],
    *,
    require_interior: bool,
) -> tuple[dict[str, object], bool] | None:
    h, w = map(int, shape)
    _ = (h, w)
    source_list = list(source)
    if not source_list:
        return None

    for _ in range(MAX_PLACE_TRIES):
        orient = random.choice(["h", "v"])
        fixed_sign = random.choice([-1, 1])
        fixed_len = random.randint(*FIXED_LEN_RANGE)
        tail_len = random.randint(*TAIL_LEN_RANGE)
        d_choices = list(range(1, tail_len + 1))
        if require_interior and tail_len >= 2:
            interior_choices = [d for d in d_choices if d < tail_len]
            if not interior_choices:
                continue
            d = random.choice(interior_choices)
        else:
            d = random.choice(d_choices)

        fixed_step = (0, fixed_sign) if orient == "h" else (fixed_sign, 0)
        tail_step = (-fixed_step[0], -fixed_step[1])
        source_cell = random.choice(source_list)
        adj = random.choice(DIR4)
        tail_cell = (int(source_cell[0] + adj[0]), int(source_cell[1] + adj[1]))
        hinge = (int(tail_cell[0] - tail_step[0] * d), int(tail_cell[1] - tail_step[1] * d))

        rec = _make_record(shape, orient, hinge, fixed_sign, fixed_len, tail_len)
        if rec is None:
            continue
        if rec["envelope"] & occupied_all:
            continue
        if _input_conflicts(shape, rec["input_set"], occupied_input):
            continue
        if _touches(rec["folded"], occupied_all):
            continue

        current_pairs = _touch_pairs(source, rec["tail_set"])
        if len(current_pairs) != 1:
            continue
        if _touches(source, rec["non_tail_set"]):
            continue
        if _touches(forbidden_sources, rec["tail_set"]):
            continue
        if _touches(forbidden_sources, rec["non_tail_set"]):
            continue

        touched_tail = current_pairs[0][1]
        interior = tuple(map(int, touched_tail)) != tuple(map(int, rec["tail"][-1]))
        if require_interior and not interior:
            continue
        return rec, interior
    return None


def _build_pair(
    shape: tuple[int, int],
    chain: list[dict[str, object]],
    distractors: list[dict[str, object]],
    spark: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    inp[spark] = SPARK
    out[spark] = SPARK

    for rec in chain:
        inp[rec["hinge"]] = HINGE
        out[rec["hinge"]] = HINGE
        inp[rec["endpoint"]] = ENDPOINT
        out[rec["endpoint"]] = ENDPOINT
        for cell in rec["fixed_red"]:
            inp[cell] = RED
            out[cell] = RED
        for cell in rec["tail"]:
            inp[cell] = RED
        for cell in rec["folded"]:
            out[cell] = RED

    for rec in distractors:
        inp[rec["hinge"]] = HINGE
        out[rec["hinge"]] = HINGE
        inp[rec["endpoint"]] = ENDPOINT
        out[rec["endpoint"]] = ENDPOINT
        for cell in rec["fixed_red"]:
            inp[cell] = RED
            out[cell] = RED
        for cell in rec["tail"]:
            inp[cell] = RED
            out[cell] = RED

    return inp, out


def _validate_generated(
    shape: tuple[int, int],
    chain: list[dict[str, object]],
    distractors: list[dict[str, object]],
    spark: tuple[int, int],
    used_interior_contact: bool,
) -> bool:
    if len(chain) < 2 or len(distractors) < 1 or not used_interior_contact:
        return False

    if len(_touch_pairs({spark}, chain[0]["tail_set"])) != 1:
        return False
    if _touches({spark}, chain[0]["non_tail_set"]):
        return False

    previous_sources: set[tuple[int, int]] = {spark}
    for i in range(1, len(chain)):
        if len(_touch_pairs(set(chain[i - 1]["folded"]), chain[i]["tail_set"])) != 1:
            return False
        if _touches(previous_sources, chain[i]["tail_set"]):
            return False
        if _touches(previous_sources, chain[i]["non_tail_set"]):
            return False
        previous_sources |= set(chain[i - 1]["folded"])

    all_sources = {spark}
    for rec in chain:
        all_sources |= set(rec["folded"])
    for rec in distractors:
        if _touches(all_sources, rec["tail_set"]):
            return False

    inp, expected = _build_pair(shape, chain, distractors, spark)
    if np.count_nonzero(inp != BG) * 2 >= inp.size:
        return False

    try:
        actual, info = _solve(inp, strict=True)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(info["fired"]) != len(chain):
        return False
    if len(components_mask(np.isin(inp, list(ROD_COLORS)))) != len(chain) + len(distractors):
        return False
    if np.array_equal(actual, inp):
        return False
    if np.array_equal(actual, _fold_all(inp, info["rods"])):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    chain_len = random.choice(CHAIN_CHOICES)
    distractor_count = random.choice(DISTRACTOR_CHOICES)
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)

    occupied_input: set[tuple[int, int]] = set()
    occupied_all: set[tuple[int, int]] = set()
    chain: list[dict[str, object]] = []
    distractors: list[dict[str, object]] = []

    first = _random_free_record(shape, occupied_input, occupied_all)
    if first is None:
        return None
    spark_info = _choose_spark(shape, first, first["input_set"])
    if spark_info is None:
        return None
    spark, used_interior_contact = spark_info
    chain.append(first)
    occupied_input |= set(first["input_set"])
    occupied_all |= set(first["envelope"])
    occupied_input.add(spark)
    occupied_all.add(spark)

    current_source = set(first["folded"])
    previous_sources: set[tuple[int, int]] = {spark}

    for i in range(1, chain_len):
        need_interior = (i == chain_len - 1) and (not used_interior_contact)
        placed = _contact_record(
            shape,
            current_source,
            previous_sources,
            occupied_input,
            occupied_all,
            require_interior=need_interior,
        )
        if placed is None:
            return None
        rec, interior = placed
        used_interior_contact = used_interior_contact or interior
        chain.append(rec)
        occupied_input |= set(rec["input_set"])
        occupied_all |= set(rec["envelope"])
        previous_sources |= current_source
        current_source = set(rec["folded"])

    all_sources = {spark}
    for rec in chain:
        all_sources |= set(rec["folded"])

    for _ in range(distractor_count):
        rec = _random_free_record(shape, occupied_input, occupied_all)
        if rec is None:
            return None
        if _touches(all_sources, rec["tail_set"]):
            return None
        distractors.append(rec)
        occupied_input |= set(rec["input_set"])
        occupied_all |= set(rec["envelope"])

    if not _validate_generated(shape, chain, distractors, spark, used_interior_contact):
        return None

    inp, _expected = _build_pair(shape, chain, distractors, spark)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-domino ARC input")
