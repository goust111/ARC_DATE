from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, paste


BG = 0
BLUE = 1
RED = 2
INPUT_COLORS = (3, 4, 5, 6, 7)
OBJ_SIZE = 4
GEN_ATTEMPTS = 400


def _make_frame(color: int) -> np.ndarray:
    patch = np.full((OBJ_SIZE, OBJ_SIZE), int(color), dtype=int)
    patch[1:3, 1:3] = BG
    return patch


def _make_solid(color: int) -> np.ndarray:
    return np.full((OBJ_SIZE, OBJ_SIZE), int(color), dtype=int)


def _full_bar_rows(grid: np.ndarray, color: int) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == int(color))]


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (int(a1) < int(b0) or int(b1) < int(a0))


def _parse_grid(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blue_rows = _full_bar_rows(g, BLUE)
    red_rows = _full_bar_rows(g, RED)
    assert len(blue_rows) == 1, "expected exactly one blue full-row bar"
    assert len(red_rows) == 1, "expected exactly one red full-row bar"
    blue_row = int(blue_rows[0])
    red_row = int(red_rows[0])
    assert blue_row < red_row, "blue bar must be above red bar"

    work = g.copy()
    work[blue_row, :] = BG
    work[red_row, :] = BG
    comps = find_components(work, bg=BG, connectivity=4, monochromatic=False, return_format="dict")

    objects: list[dict] = []
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        assert bbox_h(bbox) == OBJ_SIZE and bbox_w(bbox) == OBJ_SIZE, "every foreground object must be 4x4"
        r0, c0, r1, c1 = bbox
        patch = crop_bbox(work, bbox)
        colors = sorted(int(v) for v in np.unique(patch) if int(v) != BG)
        assert len(colors) == 1, "each object must be monochromatic"
        color = int(colors[0])
        is_hollow = bool(np.all(patch[1:3, 1:3] == BG))
        if is_hollow:
            assert np.all(patch[[0, 3], :] == color), "frame border must be monochromatic"
            assert np.all(patch[:, [0, 3]] == color), "frame border must be monochromatic"
        else:
            assert np.all(patch == color), "solid block must be fully filled"
        objects.append(
            {
                "bbox": bbox,
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "color": color,
                "kind": "hollow" if is_hollow else "solid",
            }
        )

    assert objects, "expected at least one 4x4 object"
    return blue_row, red_row, objects


def _distance_to_target(obj: dict, blue_row: int, red_row: int) -> int:
    if obj["kind"] == "hollow":
        return int(obj["r0"] - blue_row - 1)
    return int(red_row - obj["r1"] - 1)


def _landing_top(obj: dict, placed: list[dict], blue_row: int, red_row: int, *, use_collision: bool) -> int:
    if obj["kind"] == "hollow":
        top = int(blue_row + 1)
        if use_collision:
            for other in placed:
                if other["r1"] < obj["r0"] and _intervals_overlap(obj["c0"], obj["c1"], other["c0"], other["c1"]):
                    top = max(top, int(other["r1"] + 1))
        assert top <= obj["r0"], "upward-moving object cannot move downward"
        return int(top)

    top = int(red_row - OBJ_SIZE)
    if use_collision:
        for other in placed:
            if other["r0"] > obj["r1"] and _intervals_overlap(obj["c0"], obj["c1"], other["c0"], other["c1"]):
                top = min(top, int(other["r0"] - OBJ_SIZE))
    assert top >= obj["r0"], "downward-moving object cannot move upward"
    return int(top)


def _simulate_objects(
    grid: np.ndarray,
    *,
    use_collision: bool = True,
    forced_kind: str | None = None,
) -> tuple[int, int, list[dict]]:
    blue_row, red_row, objects = _parse_grid(grid)
    sim_objects: list[dict] = []
    for obj in objects:
        kind = forced_kind if forced_kind is not None else obj["kind"]
        sim_obj = dict(obj)
        sim_obj["kind"] = kind
        sim_obj["distance"] = _distance_to_target(sim_obj, blue_row, red_row)
        sim_objects.append(sim_obj)

    sim_objects.sort(key=lambda obj: (int(obj["distance"]), int(obj["c0"])))

    placed: list[dict] = []
    for obj in sim_objects:
        top = _landing_top(obj, placed, blue_row, red_row, use_collision=use_collision)
        landed = dict(obj)
        landed["r0"] = int(top)
        landed["r1"] = int(top + OBJ_SIZE - 1)
        landed["bbox"] = (landed["r0"], landed["c0"], landed["r1"], landed["c1"])
        placed.append(landed)
    return blue_row, red_row, placed


def _render_solution(shape: tuple[int, int], blue_row: int, red_row: int, placed: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[int(blue_row), :] = BLUE
    out[int(red_row), :] = RED
    for obj in placed:
        if obj["kind"] == "hollow":
            sprite = _make_frame(BLUE)
            paste(out, sprite, int(obj["r0"]), int(obj["c0"]), transparent=BG)
        else:
            sprite = _make_solid(RED)
            paste(out, sprite, int(obj["r0"]), int(obj["c0"]))
    return out


def _solve(grid: np.ndarray, *, use_collision: bool = True, forced_kind: str | None = None) -> np.ndarray:
    blue_row, red_row, placed = _simulate_objects(grid, use_collision=use_collision, forced_kind=forced_kind)
    return _render_solution(np.asarray(grid).shape, blue_row, red_row, placed)


def main(input_grid: np.ndarray) -> np.ndarray:
    """Classify each 4x4 object by its 2x2 interior, slide it to the type-matched row bar, and repaint by type.

    Input structure:
    - The grid has black background, one blue full horizontal bar near the top, and one red full horizontal bar near the bottom.
    - Between the bars are several monochrome 4x4 foreground objects in colors 3/4/5/6/7.
    - Each object is either a hollow 4x4 frame with a black 2x2 interior or a solid 4x4 block.

    Rule:
    - Detect the blue and red full-row bars and all 4x4 objects.
    - Classify an object as hollow iff its internal 2x2 region is black; hollow objects slide vertically upward toward the blue bar, while solid objects slide vertically downward toward the red bar.
    - Process objects in increasing distance to their target bar, breaking ties by left column; each object stops one row before it would touch its target bar or any previously landed object's bounding box.
    - Keep both bars unchanged, repaint all hollow frames blue with black interiors, and repaint all solid blocks red.

    Key insight:
    - Original object color is a distractor: the hidden 2x2 interior determines both the destination bar and the final recolor, while early landings create real physical blockers for later objects.
    """
    return _solve(np.asarray(input_grid, dtype=int), use_collision=True)


def _sample_spaced_positions(low: int, high: int, count: int, gap: int = 5) -> list[int] | None:
    candidates = list(range(int(low), int(high) + 1))
    for _ in range(120):
        random.shuffle(candidates)
        chosen: list[int] = []
        for value in candidates:
            if all(abs(int(value) - prev) >= int(gap) for prev in chosen):
                chosen.append(int(value))
                if len(chosen) == int(count):
                    return sorted(chosen)
    return None


def _shared_color_assignment(n_hollow: int, n_solid: int) -> tuple[list[int], list[int]]:
    shared = random.choice(INPUT_COLORS)
    hollow_colors = [shared] + [random.choice(INPUT_COLORS) for _ in range(n_hollow - 1)]
    solid_colors = [shared] + [random.choice(INPUT_COLORS) for _ in range(n_solid - 1)]
    random.shuffle(hollow_colors)
    random.shuffle(solid_colors)
    return hollow_colors, solid_colors


def _lane_pair(width: int) -> tuple[int, int]:
    pairs = [
        (left, right)
        for left in range(width - OBJ_SIZE + 1)
        for right in range(width - OBJ_SIZE + 1)
        if left < right and right - left >= OBJ_SIZE + 1
    ]
    return random.choice(pairs)


def _count_blocked(placed: list[dict], blue_row: int, red_row: int) -> int:
    blocked = 0
    for obj in placed:
        direct_top = int(blue_row + 1) if obj["kind"] == "hollow" else int(red_row - OBJ_SIZE)
        if int(obj["r0"]) != direct_top:
            blocked += 1
    return blocked


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        blue_row, red_row, objects = _parse_grid(grid)
        if len(objects) < 4:
            return False
        hollow_colors = {obj["color"] for obj in objects if obj["kind"] == "hollow"}
        solid_colors = {obj["color"] for obj in objects if obj["kind"] == "solid"}
        if not hollow_colors or not solid_colors:
            return False
        if not (hollow_colors & solid_colors):
            return False

        _, _, placed = _simulate_objects(grid, use_collision=True)
        if _count_blocked(placed, blue_row, red_row) == 0:
            return False

        correct = _render_solution(grid.shape, blue_row, red_row, placed)
        no_collision = _solve(grid, use_collision=False)
        all_hollow = _solve(grid, use_collision=True, forced_kind="hollow")
        all_solid = _solve(grid, use_collision=True, forced_kind="solid")

        if np.array_equal(correct, no_collision):
            return False
        if np.array_equal(correct, all_hollow):
            return False
        if np.array_equal(correct, all_solid):
            return False
        return True
    except AssertionError:
        return False


def _generate_one() -> np.ndarray | None:
    height = random.randint(20, 26)
    width = random.randint(9, 12)
    blue_row = random.randint(0, 1)
    red_row = height - 1

    total = random.randint(4, 6)
    n_hollow = random.randint(2, min(3, total - 2))
    n_solid = total - n_hollow
    if not (2 <= n_solid <= 3):
        return None

    low = blue_row + 2
    high = red_row - OBJ_SIZE - 1
    hollow_rows = _sample_spaced_positions(low, high, n_hollow, gap=5)
    solid_rows = _sample_spaced_positions(low, high, n_solid, gap=5)
    if hollow_rows is None or solid_rows is None:
        return None

    lane_a, lane_b = _lane_pair(width)
    if random.random() < 0.5:
        hollow_col, solid_col = lane_a, lane_b
    else:
        hollow_col, solid_col = lane_b, lane_a

    hollow_colors, solid_colors = _shared_color_assignment(n_hollow, n_solid)

    grid = np.zeros((height, width), dtype=int)
    grid[blue_row, :] = BLUE
    grid[red_row, :] = RED

    hollow_rows = random.sample(hollow_rows, len(hollow_rows))
    solid_rows = random.sample(solid_rows, len(solid_rows))
    for r0, color in zip(hollow_rows, hollow_colors):
        paste(grid, _make_frame(color), int(r0), int(hollow_col), transparent=BG)
    for r0, color in zip(solid_rows, solid_colors):
        paste(grid, _make_solid(color), int(r0), int(solid_col))

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
