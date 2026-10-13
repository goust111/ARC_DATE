from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, bbox_coords, colors_present, get_bg
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        colors = set(map(int, grid.ravel()))
        if exclude is not None:
            colors -= set(map(int, exclude))
        return sorted(colors)

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


_GEN_HEIGHT_VALUES = tuple(range(2, 10))
_GEN_GAP_VALUES = tuple(range(2, 7))
_GEN_MIN_RODS = 3
_GEN_MAX_RODS = 5
_GEN_MAX_ATTEMPTS = 400
_GEN_MODE_CYCLE = ("stop", "full", "partial", "partial")
_GEN_CALL_COUNT = 0


def _rod_columns_for_color(grid: np.ndarray, color: int) -> list[dict] | None:
    """Return all ground-anchored vertical rods of one color, or None if the color is not rod-only."""
    h, w = grid.shape
    rods: list[dict] = []
    counted = 0

    for c in range(w):
        rows = np.flatnonzero(grid[:-1, c] == int(color))
        if rows.size == 0:
            continue
        counted += int(rows.size)
        top = int(rows[0])
        expected = np.arange(top, h - 1)
        if rows[-1] != h - 2 or not np.array_equal(rows, expected):
            return None
        rods.append({"col": int(c), "top": top, "height": int(rows.size)})

    if not rods:
        return None
    if counted != int(np.sum(grid[:-1] == int(color))):
        return None
    return rods


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    assert h >= 4 and w >= 4, "grid is too small"

    ground_color = int(g[-1, 0])
    assert np.all(g[-1, :] == ground_color), "expected a solid ground row at the bottom"

    upper = g[:-1, :]
    bg = int(get_bg(upper))

    rod_candidates: list[tuple[int, list[dict]]] = []
    for color in colors_present(upper, exclude={bg}):
        rods = _rod_columns_for_color(g, int(color))
        if rods is not None:
            rod_candidates.append((int(color), rods))
    assert len(rod_candidates) == 1, "expected exactly one rod color"

    rod_color, rods = rod_candidates[0]
    rods = sorted(rods, key=lambda rod: rod["col"])
    assert len(rods) >= 2, "expected multiple rods"

    trigger_coords = np.argwhere((upper != bg) & (upper != rod_color))
    assert trigger_coords.shape[0] == 1, "expected exactly one trigger pixel"
    r0, c0, r1, c1 = bbox_coords(trigger_coords)
    assert (r0, c0) == (r1, c1), "trigger must be a single pixel"
    trigger = (int(r0), int(c0))

    left = rods[0]
    right = rods[-1]
    if trigger == (left["top"], left["col"] - 1):
        direction = 1
        ordered = rods
    elif trigger == (right["top"], right["col"] + 1):
        direction = -1
        ordered = list(reversed(rods))
    else:
        raise AssertionError("trigger must sit outside the top of the outermost rod")

    return {
        "grid": g,
        "bg": bg,
        "ground_color": ground_color,
        "rod_color": rod_color,
        "trigger": trigger,
        "direction": direction,
        "rods_lr": rods,
        "ordered_rods": ordered,
        "fall_row": h - 2,
    }


def _reach_end(rod: dict, direction: int) -> int:
    return int(rod["col"] + direction * (rod["height"] - 1))


def _reaches_next(rod: dict, nxt: dict, direction: int) -> bool:
    end = _reach_end(rod, direction)
    if direction == 1:
        return end >= int(nxt["col"])
    return end <= int(nxt["col"])


def _fall_count(scene: dict) -> int:
    ordered = scene["ordered_rods"]
    direction = int(scene["direction"])
    fallen = 1
    while fallen < len(ordered) and _reaches_next(ordered[fallen - 1], ordered[fallen], direction):
        fallen += 1
    return int(fallen)


def _render_chain(scene: dict, fall_count: int) -> np.ndarray:
    g = scene["grid"]
    out = g.copy()
    bg = int(scene["bg"])
    rod_color = int(scene["rod_color"])
    direction = int(scene["direction"])
    fall_row = int(scene["fall_row"])
    trig_r, trig_c = scene["trigger"]

    out[trig_r, trig_c] = bg

    for rod in scene["ordered_rods"][:fall_count]:
        c = int(rod["col"])
        top = int(rod["top"])
        height = int(rod["height"])
        out[top : g.shape[0] - 1, c] = bg
        if direction == 1:
            out[fall_row, c : c + height] = rod_color
        else:
            out[fall_row, c - height + 1 : c + 1] = rod_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Knock the outer rod inward and continue only while each fallen rod reaches the next one.

    Input structure:
    - The bottom row is a solid ground strip of one color.
    - Above the ground stand several same-color vertical rods, each occupying one column and touching the row just above the ground.
    - A single trigger pixel sits directly outside the top of the leftmost or rightmost rod, marking the inward push direction.

    Rule:
    - Remove the trigger pixel in the output.
    - The pushed rod rotates around its base into the row just above the ground, becoming a same-color horizontal segment whose length equals that rod's height.
    - Process rods one by one in the push direction: if a fallen rod's horizontal segment reaches the next still-upright rod's column, that next rod also falls the same way; otherwise the chain stops and all remaining rods stay vertical.

    Key insight:
    - The result is a directional prefix domino chain: every rod is decided only by whether the previous fallen rod covers the next gap, so the first failed reach fixes the exact stopping point.
    """
    scene = _parse_scene(grid)
    return _render_chain(scene, _fall_count(scene))


def _branch_mode(call_index: int) -> str:
    return _GEN_MODE_CYCLE[int(call_index) % len(_GEN_MODE_CYCLE)]


def _choose_fall_count(n_rods: int, call_index: int) -> tuple[int, str]:
    mode = _branch_mode(call_index)
    if mode == "stop":
        return 1, mode
    if mode == "full":
        return n_rods, mode
    return random.randint(2, n_rods - 1), mode


def _height_domains(gaps: list[int], fall_count: int, n_rods: int) -> list[list[int]]:
    domains: list[list[int]] = []
    max_height = max(_GEN_HEIGHT_VALUES)
    min_height = min(_GEN_HEIGHT_VALUES)

    for i in range(n_rods):
        lo = min_height
        hi = max_height
        if i < fall_count - 1:
            lo = gaps[i] + 1
            if i + 1 < len(gaps):
                hi = min(hi, gaps[i] + gaps[i + 1])
        elif i == fall_count - 1 and fall_count < n_rods:
            hi = min(hi, gaps[i])
        if lo > hi:
            return []
        domains.append(list(range(lo, hi + 1)))
    return domains


def _choose_heights(gaps: list[int], fall_count: int, n_rods: int) -> list[int] | None:
    domains = _height_domains(gaps, fall_count, n_rods)
    if not domains:
        return None

    order = sorted(range(n_rods), key=lambda i: (len(domains[i]), i))
    heights: list[int | None] = [None] * n_rods
    used: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        values = domains[idx][:]
        random.shuffle(values)
        for value in values:
            if value in used:
                continue
            heights[idx] = int(value)
            used.add(int(value))
            if dfs(pos + 1):
                return True
            used.remove(int(value))
            heights[idx] = None
        return False

    if not dfs(0):
        return None
    return [int(v) for v in heights]


def _render_rightward_input(columns: list[int], heights: list[int], rod_color: int, trigger_color: int, ground_color: int) -> np.ndarray:
    top_padding = random.randint(0, 3)
    h = max(heights) + 1 + top_padding
    fall_count = len(columns)
    max_end = max(columns[i] + heights[i] - 1 for i in range(fall_count))
    w = max(max_end, columns[-1]) + 1 + random.randint(0, 2)
    grid = np.full((h, w), Color.BLACK, dtype=int)
    grid[-1, :] = int(ground_color)

    for c, height in zip(columns, heights):
        top = h - 1 - height
        grid[top : h - 1, c] = int(rod_color)

    trigger = (h - 1 - heights[0], columns[0] - 1)
    grid[trigger] = int(trigger_color)
    return grid


def _validate_generated_grid(grid: np.ndarray, expected_fall_count: int, mode: str) -> None:
    scene = _parse_scene(grid)
    rods_lr = scene["rods_lr"]
    ordered = scene["ordered_rods"]
    direction = int(scene["direction"])
    actual_fall_count = _fall_count(scene)

    heights = [int(rod["height"]) for rod in rods_lr]
    gaps = [int(rods_lr[i + 1]["col"] - rods_lr[i]["col"]) for i in range(len(rods_lr) - 1)]
    assert len(set(heights)) == len(heights), "rod heights must be pairwise distinct"
    assert len(set(gaps)) == len(gaps), "adjacent gaps must be pairwise distinct"

    assert actual_fall_count == expected_fall_count, "constructed chain length changed"
    for i in range(actual_fall_count - 1):
        assert _reaches_next(ordered[i], ordered[i + 1], direction), "earlier chain step must reach"
    if actual_fall_count < len(ordered):
        assert not _reaches_next(ordered[actual_fall_count - 1], ordered[actual_fall_count], direction), "chain should stop at first miss"

    for i in range(len(ordered) - 2):
        end = _reach_end(ordered[i], direction)
        third_col = int(ordered[i + 2]["col"])
        if direction == 1:
            assert end < third_col, "a rod should not skip over the immediate next rod"
        else:
            assert end > third_col, "a rod should not skip over the immediate next rod"

    expected = _render_chain(scene, actual_fall_count)
    only_first = _render_chain(scene, 1)
    all_fall = _render_chain(scene, len(ordered))

    assert np.array_equal(main(grid), expected), "main disagrees with the parsed scene"
    assert not np.array_equal(expected, grid), "output should differ from input"
    if mode == "stop":
        assert actual_fall_count == 1
        assert not np.array_equal(expected, all_fall), "stop sample must reject the all-fall shortcut"
    elif mode == "full":
        assert actual_fall_count == len(ordered)
        assert not np.array_equal(expected, only_first), "full sample must reject the only-first shortcut"
    else:
        assert 1 < actual_fall_count < len(ordered)
        assert not np.array_equal(expected, only_first), "partial sample must reject the only-first shortcut"
        assert not np.array_equal(expected, all_fall), "partial sample must reject the all-fall shortcut"


def _gen_attempt(call_index: int) -> np.ndarray | None:
    n_rods = random.randint(_GEN_MIN_RODS, _GEN_MAX_RODS)
    expected_fall_count, mode = _choose_fall_count(n_rods, call_index)

    gaps = random.sample(_GEN_GAP_VALUES, n_rods - 1)
    random.shuffle(gaps)
    heights = _choose_heights(gaps, expected_fall_count, n_rods)
    if heights is None:
        return None

    columns = [1]
    for gap in gaps:
        columns.append(columns[-1] + int(gap))

    max_end = max(columns[i] + heights[i] - 1 for i in range(expected_fall_count))
    width_upper_bound = max(max_end, columns[-1]) + 3
    if width_upper_bound > 30:
        return None

    rod_color, trigger_color, ground_color = random.sample(range(1, 10), 3)
    grid = _render_rightward_input(columns, heights, rod_color, trigger_color, ground_color)

    if max(grid.shape) > 30:
        return None

    if call_index % 2 == 1:
        grid = np.fliplr(grid)

    try:
        _validate_generated_grid(grid, expected_fall_count, mode)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    call_index = _GEN_CALL_COUNT
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(call_index)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid domino-chain input")
