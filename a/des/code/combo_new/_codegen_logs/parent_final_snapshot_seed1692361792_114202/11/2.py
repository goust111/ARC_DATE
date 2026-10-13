from __future__ import annotations

import random

import numpy as np


BG = 0
TRAIL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

DIRECTION_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Arrow geometry expressed relative to the tip cell.
ARROW_OFFSETS = {
    "up": ((0, 0), (1, 0), (2, 0), (2, -1), (2, 1)),
    "down": ((0, 0), (-1, 0), (-2, 0), (-2, -1), (-2, 1)),
    "left": ((0, 0), (0, 1), (0, 2), (-1, 2), (1, 2)),
    "right": ((0, 0), (0, -1), (0, -2), (-1, -2), (1, -2)),
}

ARROW_SIGNATURES: dict[str, tuple[frozenset[tuple[int, int]], tuple[int, int]]] = {}
for _name, _offsets in ARROW_OFFSETS.items():
    _min_r = min(dr for dr, _dc in _offsets)
    _min_c = min(dc for _dr, dc in _offsets)
    ARROW_SIGNATURES[_name] = (
        frozenset((dr - _min_r, dc - _min_c) for dr, dc in _offsets),
        (-_min_r, -_min_c),
    )


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    starts = {tuple(map(int, coord)) for coord in np.argwhere(grid == color)}
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for start in starts:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in starts and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(sorted(comp))
    return comps


def _match_arrow(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
    if len(coords) != 5:
        return None
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    norm = frozenset((r - min_r, c - min_c) for r, c in coords)
    for direction, (pattern, tip_norm) in ARROW_SIGNATURES.items():
        if norm == pattern:
            return direction, (min_r + tip_norm[0], min_c + tip_norm[1])
    return None


def _is_straight_line(coords: list[tuple[int, int]]) -> bool:
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    return len(rows) == 1 or len(cols) == 1


def _shift_cells(
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    step: tuple[int, int],
    amount: int,
) -> set[tuple[int, int]]:
    dr, dc = step
    return {(r + dr * amount, c + dc * amount) for r, c in cells}


def _arrow_cells_from_tip(tip: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    tr, tc = tip
    return {(tr + dr, tc + dc) for dr, dc in ARROW_OFFSETS[direction]}


def _parse_scene(grid: np.ndarray) -> list[dict[str, object]]:
    colors = [int(v) for v in np.unique(grid) if int(v) != BG]
    records: list[dict[str, object]] = []
    for color in colors:
        comps = _components_of_color(grid, color)
        assert len(comps) == 2, "each moving color must form one arrow and one barrier"
        arrow_info = None
        barrier = None
        arrow_cells = None
        for comp in comps:
            matched = _match_arrow(comp)
            if matched is not None:
                assert arrow_info is None, "expected exactly one arrow per color"
                arrow_info = matched
                arrow_cells = set(comp)
                continue
            if _is_straight_line(comp):
                assert barrier is None, "expected exactly one barrier per color"
                barrier = set(comp)
                continue
            raise AssertionError("unexpected component shape")

        assert arrow_info is not None and arrow_cells is not None and barrier is not None
        direction, tip = arrow_info
        step = DIRECTION_STEPS[direction]

        distance = 0
        hit = None
        while True:
            distance += 1
            rr = tip[0] + step[0] * distance
            cc = tip[1] + step[1] * distance
            assert _in_bounds(grid.shape, rr, cc), "arrow ray must hit its barrier in bounds"
            if (rr, cc) in barrier:
                hit = (rr, cc)
                break
            assert int(grid[rr, cc]) == BG, "ray to matching barrier must pass through black space"

        shift = distance - 1
        assert shift >= 1, "arrow must move at least one step"

        final_arrow = _shift_cells(arrow_cells, step, shift)
        assert final_arrow.isdisjoint(barrier), "final arrow must stop next to its barrier"
        assert (tip[0] + step[0] * (shift + 1), tip[1] + step[1] * (shift + 1)) == hit

        records.append(
            {
                "color": color,
                "direction": direction,
                "step": step,
                "shift": shift,
                "arrow": arrow_cells,
                "barrier": barrier,
                "final_arrow": final_arrow,
            }
        )

    assert records, "expected at least one colored arrow-barrier pair"
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored arrow toward its matching barrier and paint the swept trail gray.

    Input structure:
    - The background is black.
    - Each moving color appears in exactly two disconnected components: a 5-pixel arrow
      and a horizontal or vertical single-pixel-wide barrier segment of the same color.
    - Every arrow is a 3x3 directional shape: a 3-pixel shaft ending at a 3-pixel tail bar,
      so its tip can point up, down, left, or right.

    Rule:
    - For each color, determine the arrow's pointing direction from its 5-pixel shape.
    - Translate the whole arrow step by step along that direction until the next cell in
      front of the tip is a pixel of the same-color barrier.
    - Paint gray on the original arrow location and on every intermediate translated
      location, but keep the final resting arrow in its original color and leave the
      barrier unchanged.

    Key insight:
    - The task separates into two independent matches: infer direction from the arrow
      geometry, then use color identity to find the correct stopping barrier along the tip ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    records = _parse_scene(g)
    out = g.copy()

    for record in records:
        color = int(record["color"])
        step = record["step"]
        shift = int(record["shift"])
        arrow = set(record["arrow"])

        for amount in range(shift):
            for r, c in _shift_cells(arrow, step, amount):
                if int(g[r, c]) not in (BG, color):
                    raise AssertionError("swept arrow path must not cross other colors")
                out[r, c] = TRAIL

    for record in records:
        color = int(record["color"])
        for r, c in record["final_arrow"]:
            out[r, c] = color

    return out


GRID_MIN = 10
GRID_MAX = 26
OBJECT_COUNT_MIN = 1
OBJECT_COUNT_MAX = 4
GAP_MIN = 1
GAP_MAX = 7
PARALLEL_LINE_MIN = 2
PARALLEL_LINE_MAX = 7
PERP_LINE_MIN = 2
PERP_LINE_MAX = 7
RESERVED_PADDING = 1
OBJECT_ATTEMPTS = 240
GENERATE_ATTEMPTS = 400


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = r + dr
                nc = c + dc
                if _in_bounds(shape, nr, nc):
                    expanded.add((nr, nc))
    return expanded


def _build_parallel_barrier(
    hit: tuple[int, int],
    step: tuple[int, int],
    shape: tuple[int, int],
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    max_len = 0
    r, c = hit
    while _in_bounds(shape, r, c) and max_len < PARALLEL_LINE_MAX:
        max_len += 1
        r += step[0]
        c += step[1]
    if max_len < PARALLEL_LINE_MIN:
        return None
    length = rng.randint(PARALLEL_LINE_MIN, max_len)
    return {(
        hit[0] + step[0] * amount,
        hit[1] + step[1] * amount,
    ) for amount in range(length)}


def _build_perpendicular_barrier(
    hit: tuple[int, int],
    direction: str,
    shape: tuple[int, int],
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    if direction in ("up", "down"):
        neg_step = (0, -1)
        pos_step = (0, 1)
    else:
        neg_step = (-1, 0)
        pos_step = (1, 0)

    max_neg = 0
    r, c = hit
    while max_neg < PERP_LINE_MAX - 1 and _in_bounds(shape, r + neg_step[0], c + neg_step[1]):
        max_neg += 1
        r += neg_step[0]
        c += neg_step[1]

    max_pos = 0
    r, c = hit
    while max_pos < PERP_LINE_MAX - 1 and _in_bounds(shape, r + pos_step[0], c + pos_step[1]):
        max_pos += 1
        r += pos_step[0]
        c += pos_step[1]

    candidates: list[tuple[int, int]] = []
    for neg in range(max_neg + 1):
        for pos in range(max_pos + 1):
            length = neg + pos + 1
            if PERP_LINE_MIN <= length <= PERP_LINE_MAX and length >= 2:
                candidates.append((neg, pos))
    if not candidates:
        return None

    neg, pos = rng.choice(candidates)
    cells = {hit}
    for amount in range(1, neg + 1):
        cells.add((hit[0] + neg_step[0] * amount, hit[1] + neg_step[1] * amount))
    for amount in range(1, pos + 1):
        cells.add((hit[0] + pos_step[0] * amount, hit[1] + pos_step[1] * amount))
    return cells


def _place_object(
    shape: tuple[int, int],
    direction: str,
    color: int,
    reserved: set[tuple[int, int]],
    rng: random.Random,
) -> dict[str, object] | None:
    step = DIRECTION_STEPS[direction]
    for _ in range(OBJECT_ATTEMPTS):
        gap = rng.randint(GAP_MIN, GAP_MAX)
        final_tip = (rng.randrange(shape[0]), rng.randrange(shape[1]))
        final_arrow = _arrow_cells_from_tip(final_tip, direction)
        if not all(_in_bounds(shape, r, c) for r, c in final_arrow):
            continue

        initial_tip = (final_tip[0] - step[0] * gap, final_tip[1] - step[1] * gap)
        initial_arrow = _arrow_cells_from_tip(initial_tip, direction)
        if not all(_in_bounds(shape, r, c) for r, c in initial_arrow):
            continue

        sweep = set()
        for amount in range(gap + 1):
            sweep |= _shift_cells(initial_arrow, step, amount)

        hit = (final_tip[0] + step[0], final_tip[1] + step[1])
        if not _in_bounds(shape, hit[0], hit[1]):
            continue

        if rng.random() < 0.5:
            barrier = _build_parallel_barrier(hit, step, shape, rng)
        else:
            barrier = _build_perpendicular_barrier(hit, direction, shape, rng)
        if barrier is None or not all(_in_bounds(shape, r, c) for r, c in barrier):
            continue
        if hit not in barrier or not sweep.isdisjoint(barrier):
            continue

        footprint = sweep | barrier
        padded = _expand(footprint, shape, RESERVED_PADDING)
        if padded & reserved:
            continue

        return {
            "color": color,
            "direction": direction,
            "step": step,
            "gap": gap,
            "initial_arrow": initial_arrow,
            "barrier": barrier,
            "reserved": padded,
        }
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    min_side = max(GRID_MIN, 8 + object_count * 3)
    height = rng.randint(min_side, GRID_MAX)
    width = rng.randint(min_side, GRID_MAX)
    grid = np.full((height, width), BG, dtype=int)

    directions = rng.sample(list(DIRECTION_STEPS.keys()), k=object_count)
    colors = rng.sample(PALETTE, k=object_count)
    reserved: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []

    for direction, color in zip(directions, colors):
        obj = _place_object(grid.shape, direction, color, reserved, rng)
        if obj is None:
            return None
        placed.append(obj)
        reserved |= set(obj["reserved"])

    for obj in placed:
        color = int(obj["color"])
        for r, c in obj["initial_arrow"]:
            grid[r, c] = color
        for r, c in obj["barrier"]:
            grid[r, c] = color

    records = _parse_scene(grid)
    if len(records) != object_count:
        return None
    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if np.any(grid == TRAIL):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid arrow-to-barrier input")
