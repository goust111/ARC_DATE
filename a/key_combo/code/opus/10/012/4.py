import random

import numpy as np


BLACK = 0
GREY = 5
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 500
GRID_MIN = 12
GRID_MAX = 16
PAIR_COUNT_OPTIONS = (3, 3, 4)
L_DR_RANGE = (2, 4)
L_DC_RANGE = (2, 5)
DIAG_LEN_RANGE = (2, 4)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = 0 if r1 == r0 else (1 if r1 > r0 else -1)
    dc = 0 if c1 == c0 else (1 if c1 > c0 else -1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _path_vh(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    corner = (b[0], a[1])
    return _segment_cells(a, corner) + _segment_cells(corner, b)[1:]


def _path_hv(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    corner = (a[0], b[1])
    return _segment_cells(a, corner) + _segment_cells(corner, b)[1:]


def _path_diag(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    if abs(a[0] - b[0]) != abs(a[1] - b[1]):
        return None
    return _segment_cells(a, b)


def _ordered_pairs(grid: np.ndarray) -> list[dict]:
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BLACK, GREY)]
    pairs = []
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(coords) == 2, "each endpoint color must appear exactly twice"
        a, b = sorted(coords)
        pairs.append({"color": color, "a": a, "b": b})
    pairs.sort(key=lambda item: (item["a"], item["b"]))
    return pairs


def _path_is_clear(
    grid: np.ndarray,
    path: list[tuple[int, int]] | None,
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    if path is None:
        return False
    endpoints = {a, b}
    for r, c in path:
        if (r, c) in endpoints:
            continue
        if int(grid[r, c]) != BLACK:
            return False
    return True


def _choose_strategy(
    grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]
) -> tuple[str, list[tuple[int, int]]]:
    candidates = [
        ("vh", _path_vh(a, b)),
        ("diag", _path_diag(a, b)),
        ("hv", _path_hv(a, b)),
    ]
    for name, path in candidates:
        if _path_is_clear(grid, path, a, b):
            return name, path
    raise AssertionError("every endpoint pair must have at least one valid path")


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    assert _get_bg(g) == BLACK, "expected black background"

    out = g.copy()
    used: dict[int, str] = {}
    for pair in _ordered_pairs(g):
        color = int(pair["color"])
        a = pair["a"]
        b = pair["b"]
        strategy, path = _choose_strategy(out, a, b)
        used[color] = strategy
        for r, c in path:
            if (r, c) not in {a, b}:
                assert int(out[r, c]) == BLACK
                out[r, c] = color
    return out, used


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored endpoint pair with the first blocker-free fallback path.

    Input structure:
    - A black background contains 3-4 non-grey foreground colors, each appearing exactly twice as endpoints.
    - Grey pixels act as blockers, and untouched black cells are the only cells available for new line segments.
    - Endpoint pairs may be arranged so that different colors need different connection geometries.

    Rule:
    - For each color, take its two endpoints and order them lexicographically as A then B; process the pairs in that same geometric order.
    - Try three connection strategies in priority order: vertical-then-horizontal L, strict 45-degree diagonal, then horizontal-then-vertical L.
    - A strategy is valid only if every non-endpoint cell on that candidate path is still black, so grey blockers and already drawn paths force fallback.
    - Paint the first valid path in the endpoint color while leaving the original endpoints and grey blockers unchanged.

    Key insight:
    - The task is not “always draw one fixed connector”; the blockers select among three path topologies, and the visible top-left pair order makes the fallback process global.
    """
    return _solve(grid)[0]


def _make_pair_spec(strategy: str) -> dict:
    if strategy == "diag":
        dr = random.randint(*DIAG_LEN_RANGE)
        dc_mag = dr
    else:
        dr = random.randint(*L_DR_RANGE)
        dc_mag = random.randint(*L_DC_RANGE)
        while dc_mag == dr:
            dc_mag = random.randint(*L_DC_RANGE)

    if random.random() < 0.5:
        a = (0, 0)
        b = (dr, dc_mag)
    else:
        a = (0, dc_mag)
        b = (dr, 0)

    vh = _path_vh(a, b)
    diag = _path_diag(a, b)
    hv = _path_hv(a, b)
    blockers: set[tuple[int, int]] = set()

    if strategy in {"diag", "hv"}:
        blockers.add(random.choice(vh[1:-1]))
    elif random.random() < 0.6:
        extras = [cell for cell in hv[1:-1] if cell not in set(vh)]
        if extras:
            blockers.add(random.choice(extras))

    if strategy == "diag" and random.random() < 0.5:
        extras = [cell for cell in hv[1:-1] if cell not in set(diag or []) and cell not in blockers]
        if extras:
            blockers.add(random.choice(extras))

    occupied = {a, b} | blockers
    height = max(r for r, _ in occupied) + 1
    width = max(c for _, c in occupied) + 1
    return {
        "strategy": strategy,
        "a": a,
        "b": b,
        "vh": vh,
        "diag": diag,
        "hv": hv,
        "blockers": sorted(blockers),
        "height": height,
        "width": width,
    }


def _place_specs(specs: list[dict], height: int, width: int) -> list[dict] | None:
    reserved = np.zeros((height, width), dtype=bool)
    placed = []
    for spec in sorted(specs, key=lambda item: item["height"] * item["width"], reverse=True):
        candidates = []
        for r0 in range(height - spec["height"] + 1):
            for c0 in range(width - spec["width"] + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(height - 1, r0 + spec["height"])
                cc0 = max(0, c0 - 1)
                cc1 = min(width - 1, c0 + spec["width"])
                if not reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        rr0 = max(0, r0 - 1)
        rr1 = min(height - 1, r0 + spec["height"])
        cc0 = max(0, c0 - 1)
        cc1 = min(width - 1, c0 + spec["width"])
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

        placed.append(
            {
                **spec,
                "offset": (r0, c0),
                "a_global": (spec["a"][0] + r0, spec["a"][1] + c0),
                "b_global": (spec["b"][0] + r0, spec["b"][1] + c0),
                "blockers_global": [(r + r0, c + c0) for r, c in spec["blockers"]],
                "vh_global": [(r + r0, c + c0) for r, c in spec["vh"]],
                "diag_global": None if spec["diag"] is None else [(r + r0, c + c0) for r, c in spec["diag"]],
                "hv_global": [(r + r0, c + c0) for r, c in spec["hv"]],
            }
        )
    return placed


def _candidate_cells(spec: dict) -> set[tuple[int, int]]:
    cells = set(spec["vh_global"]) | set(spec["hv_global"])
    if spec["diag_global"] is not None:
        cells |= set(spec["diag_global"])
    return cells


def _build_input_from_specs(height: int, width: int, placed: list[dict], colors: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec, color in zip(placed, colors):
        spec["color"] = int(color)
        ar, ac = spec["a_global"]
        br, bc = spec["b_global"]
        grid[ar, ac] = color
        grid[br, bc] = color
        for r, c in spec["blockers_global"]:
            grid[r, c] = GREY

    forbidden = set()
    for spec in placed:
        forbidden |= _candidate_cells(spec)
        forbidden.add(spec["a_global"])
        forbidden.add(spec["b_global"])
        forbidden |= set(spec["blockers_global"])

    extra_candidates = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in forbidden and grid[r, c] == BLACK
    ]
    extra_count = random.randint(0, min(3, len(extra_candidates)))
    for r, c in random.sample(extra_candidates, extra_count):
        grid[r, c] = GREY
    return grid


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    pair_count = random.choice(PAIR_COUNT_OPTIONS)

    strategies = ["vh", "diag", "hv"]
    if pair_count == 4:
        strategies.append(random.choice(strategies))
    random.shuffle(strategies)

    specs = [_make_pair_spec(strategy) for strategy in strategies]
    placed = _place_specs(specs, height, width)
    if placed is None:
        return None

    colors = random.sample(PAIR_COLORS, pair_count)
    grid = _build_input_from_specs(height, width, placed, colors)

    try:
        out, used = _solve(grid)
    except AssertionError:
        return None

    intended = {int(spec["color"]): spec["strategy"] for spec in placed}
    if used != intended:
        return None

    if set(used.values()) != {"vh", "diag", "hv"}:
        return None

    path_cells = np.argwhere(out != grid)
    if len(path_cells) == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
