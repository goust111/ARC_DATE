from __future__ import annotations

from collections import Counter
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    GRAY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


_GEN_EMITTER_COLORS = (
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
_GEN_TEST_COLOR = Color.BLUE
_GEN_MIN_HEIGHT = 7
_GEN_MAX_HEIGHT = 18
_GEN_MIN_WIDTH = 8
_GEN_MAX_WIDTH = 20
_GEN_MAX_BUDGET = 9
_GEN_MIN_PATH_PIXELS = 7
_GEN_MAX_ATTEMPTS = 600


def _top_grey_run(grid: np.ndarray) -> int:
    width = int(grid.shape[1])
    budget = 0
    while budget < width and int(grid[0, budget]) == Color.GREY:
        budget += 1
    assert 1 <= budget < width, "expected a non-empty grey prefix on the top row"
    assert not np.any(grid[0, budget:] == Color.GREY), "grey pixels must form one left-anchored run"
    assert int(np.count_nonzero(grid == Color.GREY)) == budget, "grey pixels may only appear in the counter run"
    return budget


def _extract_emitter(grid: np.ndarray) -> tuple[int, int]:
    colored = np.argwhere((grid != Color.BLACK) & (grid != Color.GREY))
    assert colored.shape == (1, 2), "expected exactly one non-black, non-grey emitter pixel"
    r, c = map(int, colored[0])
    height, width = map(int, grid.shape)
    assert r == height - 1, "emitter must be on the bottom edge"
    assert 0 < c < width - 1, "emitter may not be placed in a corner"
    return c, int(grid[r, c])


def _boundary_side(r: int, c: int, dr: int, dc: int, height: int, width: int) -> str:
    hit_vertical = (r == 0 and dr == -1) or (r == height - 1 and dr == 1)
    hit_horizontal = (c == 0 and dc == -1) or (c == width - 1 and dc == 1)
    assert hit_vertical or hit_horizontal, "not a boundary contact"
    assert not (hit_vertical and hit_horizontal), "the puzzle forbids corner hits"
    if r == 0 and dr == -1:
        return "top"
    if r == height - 1 and dr == 1:
        return "bottom"
    if c == 0 and dc == -1:
        return "left"
    return "right"


def _trace_case(height: int, width: int, start_c: int, budget: int, color: int) -> dict:
    assert 4 <= height <= 30 and 4 <= width <= 30
    assert 1 <= start_c <= width - 2
    assert budget >= 0

    out = np.full((height, width), Color.BLACK, dtype=int)
    r, c = height - 1, start_c
    dr, dc = -1, 1
    out[r, c] = int(color)

    bounce_sides: list[str] = []
    boundary_hits: list[tuple[int, int, str]] = []
    top_contacts: list[int] = []

    max_steps = (budget + 3) * (height + width) * 4
    steps = 0

    while True:
        hit_vertical = (r == 0 and dr == -1) or (r == height - 1 and dr == 1)
        hit_horizontal = (c == 0 and dc == -1) or (c == width - 1 and dc == 1)

        if hit_vertical or hit_horizontal:
            side = _boundary_side(r, c, dr, dc, height, width)
            boundary_hits.append((r, c, side))
            if r == 0:
                top_contacts.append(c)

            if len(bounce_sides) == budget:
                out[r, c] = Color.RED
                return {
                    "budget": int(budget),
                    "output": out,
                    "start_c": int(start_c),
                    "path_pixels": int(np.count_nonzero(out != Color.BLACK)),
                    "boundary_hits": tuple(boundary_hits),
                    "bounce_sides": tuple(bounce_sides),
                    "stop_side": side,
                    "stop": (int(r), int(c)),
                    "top_contacts": tuple(int(x) for x in top_contacts),
                }

            if hit_vertical:
                dr = -dr
            if hit_horizontal:
                dc = -dc
            bounce_sides.append(side)

        r += dr
        c += dc
        steps += 1
        assert 0 <= r < height and 0 <= c < width, "reflection step left the grid"
        out[r, c] = int(color)
        assert steps <= max_steps, "ray trace exceeded the expected bound"


def _shape_signature(case: dict) -> tuple[tuple[int, int], bytes]:
    out = case["output"]
    shape = tuple(map(int, out.shape))
    coded = np.zeros_like(out, dtype=np.uint8)
    coded[out == Color.RED] = 2
    coded[(out != Color.BLACK) & (out != Color.RED)] = 1
    return shape, coded.tobytes()


def _budget_respects_counter(case: dict, budget: int) -> bool:
    return all(col >= budget for col in case["top_contacts"])


def _enumerate_cases(height: int, width: int, start_c: int) -> list[dict]:
    raw_cases: list[dict] = []
    max_budget = min(width - 2, _GEN_MAX_BUDGET)
    for budget in range(1, max_budget + 1):
        try:
            case = _trace_case(height, width, start_c, budget, _GEN_TEST_COLOR)
        except AssertionError:
            continue
        if not _budget_respects_counter(case, budget):
            continue
        if case["path_pixels"] < _GEN_MIN_PATH_PIXELS:
            continue
        touched_sides = set(case["bounce_sides"]) | {case["stop_side"]}
        if len(touched_sides) < 2:
            continue
        raw_cases.append(case)

    if len(raw_cases) < 2:
        return []

    counts = Counter(_shape_signature(case) for case in raw_cases)
    unique_cases = [case for case in raw_cases if counts[_shape_signature(case)] == 1]
    if len(unique_cases) < 2:
        return []

    budgets = {case["budget"] for case in unique_cases}
    if len(budgets) < 2:
        return []

    return unique_cases


def _choose_case(cases: list[dict]) -> dict:
    sides = sorted({case["stop_side"] for case in cases})
    chosen_side = random.choice(sides)
    side_cases = [case for case in cases if case["stop_side"] == chosen_side]
    side_cases.sort(key=lambda case: (case["budget"], case["path_pixels"]))

    split = (len(side_cases) + 1) // 2
    pools = [side_cases[:split]]
    if split < len(side_cases):
        pools.append(side_cases[split:])
    pool = random.choice(pools)
    return random.choice(pool)


def _build_input(height: int, width: int, start_c: int, budget: int, color: int) -> np.ndarray:
    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[0, :budget] = Color.GREY
    grid[height - 1, start_c] = int(color)
    return grid


def _validate_generated_input(inp: np.ndarray, chosen: dict, sibling_cases: list[dict]) -> None:
    budget = _top_grey_run(inp)
    start_c, color = _extract_emitter(inp)
    assert budget == chosen["budget"]
    assert start_c == chosen["start_c"]
    assert color in _GEN_EMITTER_COLORS
    assert any(
        case["budget"] != chosen["budget"] and _shape_signature(case) != _shape_signature(chosen)
        for case in sibling_cases
    ), "the grey counter must matter for the same geometry"

    traced = _trace_case(inp.shape[0], inp.shape[1], start_c, budget, color)
    assert _budget_respects_counter(traced, budget), "the path may not cross the grey counter cells"
    assert np.array_equal(traced["output"], chosen["output"])

    out = main(inp)
    assert np.array_equal(out, chosen["output"])
    assert int(np.count_nonzero(out == Color.RED)) == 1, "output must contain one red stop marker"
    stop_r, stop_c = map(int, np.argwhere(out == Color.RED)[0])
    assert stop_r in (0, out.shape[0] - 1) or stop_c in (0, out.shape[1] - 1), "red stop must be on a boundary"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a bottom-edge diagonal ray with a grey bounce budget and mark the stopping wall hit in red.

    Input structure:
    - The grid has a black background, one colored emitter pixel on the bottom edge, and a contiguous grey prefix on the top row.
    - The grey prefix length is the bounce budget, and grey does not appear anywhere else.
    - The emitter is not in a corner, and the valid ray path never touches a corner or any grey counter cell.

    Rule:
    - Remove the grey counter from the output and start a ray from the emitter moving up-right one cell per step.
    - Whenever the ray reaches a boundary cell and would leave the grid on one axis, flip that direction component and count one bounce; paint those traveled cells with the emitter color.
    - After exactly as many bounces as there are grey cells, keep moving until the next boundary cell, paint that cell red, and stop there.

    Key insight:
    - The top-row grey cells are not picture content; they encode how many reflections to allow before the next wall contact becomes the red terminal marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert 4 <= height <= 30 and 4 <= width <= 30

    budget = _top_grey_run(g)
    start_c, color = _extract_emitter(g)
    assert int(np.count_nonzero(g != Color.BLACK)) == budget + 1, "expected only the counter and the emitter"

    traced = _trace_case(height, width, start_c, budget, color)
    assert _budget_respects_counter(traced, budget), "counter cells cannot lie on the ray path"
    return traced["output"]


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        height = random.randint(_GEN_MIN_HEIGHT, _GEN_MAX_HEIGHT)
        width = random.randint(_GEN_MIN_WIDTH, _GEN_MAX_WIDTH)
        start_c = random.randint(1, width - 2)

        cases = _enumerate_cases(height, width, start_c)
        if not cases:
            continue

        chosen = _choose_case(cases)
        color = random.choice(_GEN_EMITTER_COLORS)
        inp = _build_input(height, width, start_c, chosen["budget"], color)

        traced = _trace_case(height, width, start_c, chosen["budget"], color)
        chosen = dict(chosen)
        chosen["output"] = traced["output"]
        _validate_generated_input(inp, chosen, cases)
        return inp

    raise RuntimeError("failed to generate a valid reflection-budget input")
