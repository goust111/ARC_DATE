from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG = Color.YELLOW
IN_COLOR = Color.BLUE
LEFT_COLOR = Color.RED
RIGHT_COLOR = Color.GREEN

NOTCH_SIDES = ("up", "right", "down", "left")
MOVE_LEFT_SIDES = {"up", "left"}
STEP1_SIDES = {"up", "right"}
CYCLE = [9, 7, 5, 3, 1]
SIDE_TO_INDEX = {side: idx for idx, side in enumerate(NOTCH_SIDES)}

GRID_H_RANGE = (18, 25)
GRID_W_RANGE = (20, 27)
COUNT_CHOICES = [3, 4, 5]
COUNT_WEIGHTS = [0.20, 0.45, 0.35]
INPUT_SIDES = [3, 5, 7]
ROW_GAP = 1
MAX_GENERATION_ATTEMPTS = 500


def _advance_side(side: int, steps: int) -> int:
    idx = CYCLE.index(int(side))
    return int(CYCLE[(idx + int(steps)) % len(CYCLE)])


def _draw_solid_square(grid: np.ndarray, center_r: int, center_c: int, side: int, color: int) -> np.ndarray:
    out = grid.copy()
    half = int(side) // 2
    r0 = int(center_r) - half
    c0 = int(center_c) - half
    for r in range(r0, r0 + int(side)):
        if not (0 <= r < out.shape[0]):
            continue
        for c in range(c0, c0 + int(side)):
            if 0 <= c < out.shape[1]:
                out[r, c] = int(color)
    return out


def _in_bounds_square_coords(shape: tuple[int, int], center_r: int, center_c: int, side: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    half = int(side) // 2
    for r in range(int(center_r) - half, int(center_r) + half + 1):
        if not (0 <= r < shape[0]):
            continue
        for c in range(int(center_c) - half, int(center_c) + half + 1):
            if 0 <= c < shape[1]:
                coords.append((r, c))
    return coords


def _square_with_notch(side: int, notch_side: str) -> np.ndarray:
    patch = np.full((int(side), int(side)), IN_COLOR, dtype=int)
    mid = int(side) // 2
    if notch_side == "up":
        patch[0, mid] = BG
    elif notch_side == "down":
        patch[-1, mid] = BG
    elif notch_side == "left":
        patch[mid, 0] = BG
    elif notch_side == "right":
        patch[mid, -1] = BG
    else:
        raise ValueError(f"unknown notch side: {notch_side}")
    return patch


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[IN_COLOR],
        return_format="dict",
    )
    objects: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        side = int(r1 - r0 + 1)
        assert side == int(c1 - c0 + 1), "each foreground object must have a square bbox"
        assert side in INPUT_SIDES, "input square sides must be 3, 5, or 7"
        assert side % 2 == 1, "square side must be odd"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        holes = np.argwhere(patch == bg)
        assert holes.shape == (1, 2), "each square must have exactly one border-center notch"
        hr, hc = map(int, holes[0])
        mid = side // 2
        side_lookup = {(0, mid): "up", (mid, side - 1): "right", (side - 1, mid): "down", (mid, 0): "left"}
        assert (hr, hc) in side_lookup, "notch must be centered on one side"
        notch_side = side_lookup[(hr, hc)]

        filled = patch.copy()
        filled[hr, hc] = IN_COLOR
        assert np.all(filled == IN_COLOR), "object must be a solid square except for the single notch"

        center_r = r0 + mid
        center_c = c0 + mid
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "side": side,
                "center_r": center_r,
                "center_c": center_c,
                "notch_side": notch_side,
            }
        )

    objects.sort(key=lambda obj: (obj["center_r"], obj["center_c"]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each notched odd square to the side encoded by its border gap, then resize it by the step count encoded by that same gap.

    Input structure:
    - Yellow is the background and blue is the only foreground color.
    - The foreground consists of 3-5 separated solid odd-sided squares with side in {3, 5, 7}.
    - Each square has exactly one notch: the center cell of one bbox edge is yellow instead of blue.

    Rule:
    - Clear the canvas to yellow.
    - Process squares from top to bottom by center row.
    - A notch on the left or top means slide the square left until its original left edge would touch column 0, then recolor red.
    - A notch on the right or bottom means slide the square right until its original right edge would touch the last column, then recolor green.
    - The same notch also sets the resize step in the cycle 9 -> 7 -> 5 -> 3 -> 1 -> 9: top/right advance 1 step, bottom/left advance 2 steps.
    - Draw the resized solid square around the slid center; any part outside the grid is clipped away.

    Key insight:
    - The notch side is a two-bit code: one partition controls horizontal destination and color, while the other partition independently controls how far the side-length cycle advances.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = np.full((H, W), bg, dtype=int)

    for obj in _extract_objects(g):
        side = int(obj["side"])
        notch_side = obj["notch_side"]
        steps = 1 if notch_side in STEP1_SIDES else 2
        new_side = _advance_side(side, steps)

        if notch_side in MOVE_LEFT_SIDES:
            center_c = side // 2
            color = LEFT_COLOR
        else:
            center_c = W - 1 - side // 2
            color = RIGHT_COLOR

        coords = _in_bounds_square_coords(out.shape, obj["center_r"], center_c, new_side)
        assert all(out[r, c] == bg for r, c in coords), "transformed squares must not overlap"
        out = _draw_solid_square(out, obj["center_r"], center_c, new_side, color)

    return out


def _choose_count() -> int:
    return int(random.choices(COUNT_CHOICES, weights=COUNT_WEIGHTS, k=1)[0])


def _choose_notch_sides(count: int) -> list[str]:
    if count == 3:
        return random.sample(list(NOTCH_SIDES), 3)
    sides = list(NOTCH_SIDES)
    while len(sides) < count:
        sides.append(random.choice(NOTCH_SIDES))
    random.shuffle(sides)
    return sides


def _choose_input_side(notch_side: str) -> int:
    if notch_side in {"down", "left"}:
        return random.choice([5, 7])
    return random.choice(INPUT_SIDES)


def _sample_specs() -> list[dict]:
    count = _choose_count()
    notch_sides = _choose_notch_sides(count)
    specs = []
    for notch_side in notch_sides:
        side = _choose_input_side(notch_side)
        steps = 1 if notch_side in STEP1_SIDES else 2
        out_side = _advance_side(side, steps)
        specs.append(
            {
                "notch_side": notch_side,
                "input_side": side,
                "output_side": out_side,
                "required_side": max(side, out_side),
                "move_left": notch_side in MOVE_LEFT_SIDES,
                "step": steps,
            }
        )
    random.shuffle(specs)
    return specs


def _sample_height(specs: list[dict]) -> int | None:
    min_h = sum(int(spec["required_side"]) for spec in specs) + ROW_GAP * (len(specs) - 1)
    lo = max(GRID_H_RANGE[0], min_h)
    hi = GRID_H_RANGE[1]
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _place_rows(specs: list[dict], height: int) -> list[dict]:
    min_h = sum(int(spec["required_side"]) for spec in specs) + ROW_GAP * (len(specs) - 1)
    extra = int(height) - min_h
    slack = [0] * (len(specs) + 1)
    for _ in range(extra):
        slack[random.randrange(len(slack))] += 1

    placed: list[dict] = []
    cursor = slack[0]
    for idx, spec in enumerate(specs):
        req = int(spec["required_side"])
        center_r = cursor + req // 2
        placed.append({**spec, "center_r": center_r, "row_index": idx})
        cursor += req
        if idx + 1 < len(specs):
            cursor += ROW_GAP + slack[idx + 1]
    return placed


def _input_col(width: int, side: int, center_r: int, row_index: int, notch_side: str) -> int:
    span = int(width) - int(side) - 2
    assert span >= 0, "input square must fit with interior margin"
    offset = (
        3 * int(center_r)
        + 5 * int(side)
        + 7 * int(row_index)
        + 11 * SIDE_TO_INDEX[notch_side]
        + int(width)
    ) % (span + 1)
    return 1 + offset


def _render_input(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((int(height), int(width)), BG, dtype=int)
    for spec in specs:
        side = int(spec["input_side"])
        patch = _square_with_notch(side, spec["notch_side"])
        center_r = int(spec["center_r"])
        center_c = _input_col(width, side, center_r, spec["row_index"], spec["notch_side"]) + side // 2
        r0 = center_r - side // 2
        c0 = center_c - side // 2
        assert r0 >= 0 and c0 >= 0 and r0 + side <= height and c0 + side <= width
        assert np.all(grid[r0 : r0 + side, c0 : c0 + side] == BG), "input squares must not overlap"
        grid[r0 : r0 + side, c0 : c0 + side] = patch
    return grid


def _row_interval(center_r: int, side: int) -> tuple[int, int]:
    half = int(side) // 2
    return int(center_r) - half, int(center_r) + half


def _semantic_checks(inp: np.ndarray) -> bool:
    objs = _extract_objects(inp)
    if not (3 <= len(objs) <= 5):
        return False

    intervals = []
    move_and_step = []
    for obj in objs:
        side = int(obj["side"])
        r0, r1 = _row_interval(obj["center_r"], side)
        intervals.append((r0, r1))
        notch_side = obj["notch_side"]
        move_key = "L" if notch_side in MOVE_LEFT_SIDES else "R"
        step_key = 1 if notch_side in STEP1_SIDES else 2
        move_and_step.append((move_key, step_key))

    for i in range(len(intervals) - 1):
        if intervals[i][1] >= intervals[i + 1][0]:
            return False

    has_same_dir_diff_step = any(
        move_and_step[i][0] == move_and_step[j][0] and move_and_step[i][1] != move_and_step[j][1]
        for i in range(len(move_and_step))
        for j in range(i + 1, len(move_and_step))
    )
    has_same_step_diff_dir = any(
        move_and_step[i][1] == move_and_step[j][1] and move_and_step[i][0] != move_and_step[j][0]
        for i in range(len(move_and_step))
        for j in range(i + 1, len(move_and_step))
    )
    if not (has_same_dir_diff_step and has_same_step_diff_dir):
        return False

    out = main(inp)
    if out.shape != inp.shape or np.array_equal(out, inp):
        return False

    out_components = find_components(
        out,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[LEFT_COLOR, RIGHT_COLOR],
        return_format="dict",
    )
    if len(out_components) != len(objs):
        return False

    expected = []
    for obj in objs:
        side = int(obj["side"])
        notch_side = obj["notch_side"]
        steps = 1 if notch_side in STEP1_SIDES else 2
        out_side = _advance_side(side, steps)
        color = LEFT_COLOR if notch_side in MOVE_LEFT_SIDES else RIGHT_COLOR
        center_c = side // 2 if notch_side in MOVE_LEFT_SIDES else inp.shape[1] - 1 - side // 2
        expected.append((obj["center_r"], center_c, out_side, color))

    actual = []
    for comp in out_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        color = int(comp["color"])
        actual.append((r0 + (r1 - r0 + 1) // 2, c0 + (c1 - c0 + 1) // 2, r1 - r0 + 1, color))

    if sorted(expected) != sorted(actual):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        specs = _sample_specs()
        height = _sample_height(specs)
        if height is None:
            continue
        width = random.randint(*GRID_W_RANGE)
        placed = _place_rows(specs, height)
        inp = _render_input(height, width, placed)
        if _semantic_checks(inp):
            return inp
    raise RuntimeError("failed to generate a valid notched-square slide-resize task")
