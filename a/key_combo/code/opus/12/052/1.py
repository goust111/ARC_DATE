from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
OBJECT_COLORS = (1, 2, 7, 9)
LABEL_COLORS = (3, 4, 6, 8)
DIR_BY_LABEL = {
    3: "left",
    4: "down",
    6: "up",
    8: "right",
}

GEN_ATTEMPTS = 400
GEN_SIZE_RANGE = (11, 15)
GEN_SCENARIOS = (
    "horizontal_collision",
    "vertical_collision",
    "mixed",
    "all_directions",
    "single_active",
)
GEN_SCENARIO_WEIGHTS = (3, 3, 3, 2, 2)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _make_sprite(color: int) -> np.ndarray:
    sprite = np.full((3, 3), int(color), dtype=int)
    sprite[1, 1] = BACKGROUND
    return sprite


def _bbox_of(obj: dict) -> tuple[int, int, int, int]:
    return obj["r0"], obj["c0"], obj["r0"] + 2, obj["c0"] + 2


def _naive_target(obj: dict, height: int, width: int) -> tuple[int, int]:
    max_r = height - 4
    max_c = width - 4
    direction = obj["direction"]
    if direction == "left":
        return obj["r0"], 0
    if direction == "right":
        return obj["r0"], max_c
    if direction == "up":
        return 0, obj["c0"]
    if direction == "down":
        return max_r, obj["c0"]
    return obj["r0"], obj["c0"]


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 5, "expected a non-empty interior plus label borders"

    height, width = g.shape
    interior = g[: height - 1, : width - 1]
    assert set(map(int, np.unique(interior))) <= {BACKGROUND, *OBJECT_COLORS}, "unexpected interior colors"
    assert set(map(int, np.unique(g[:, width - 1]))) <= {BACKGROUND, *LABEL_COLORS}, "unexpected row-label colors"
    assert set(map(int, np.unique(g[height - 1, :]))) <= {BACKGROUND, *LABEL_COLORS}, "unexpected column-label colors"

    objects: list[dict] = []
    for color in OBJECT_COLORS:
        coords = np.argwhere(interior == color)
        if coords.size == 0:
            continue
        assert coords.shape[0] == 8, f"expected one hollow 3x3 square of color {color}"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        assert (r1 - r0, c1 - c0) == (2, 2), f"expected a 3x3 bbox for color {color}"
        sprite = interior[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(sprite, _make_sprite(color)), f"malformed hollow square for color {color}"
        objects.append(
            {
                "color": int(color),
                "r0": r0,
                "c0": c0,
                "center_r": r0 + 1,
                "center_c": c0 + 1,
            }
        )

    objects.sort(key=lambda obj: (obj["r0"], obj["c0"], obj["color"]))
    assert 2 <= len(objects) <= 4, "expected 2-4 hollow squares"
    return objects


def _decorate_objects(objects: list[dict], grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    decorated: list[dict] = []
    for obj in objects:
        row_label = int(grid[obj["center_r"], width - 1])
        col_label = int(grid[height - 1, obj["center_c"]])
        direction = DIR_BY_LABEL[row_label] if row_label == col_label and row_label in DIR_BY_LABEL else None
        enriched = dict(obj)
        enriched["row_label"] = row_label
        enriched["col_label"] = col_label
        enriched["direction"] = direction
        decorated.append(enriched)
    return decorated


def _pack_group(group: list[dict], direction: str, height: int, width: int) -> dict[int, tuple[int, int]]:
    max_r = height - 4
    max_c = width - 4
    placed: list[tuple[int, int]] = []
    final_by_color: dict[int, tuple[int, int]] = {}

    if direction == "left":
        order = sorted(group, key=lambda obj: (obj["c0"], obj["r0"], obj["c0"]))
        for obj in order:
            target_c = 0
            for other_r0, other_c0 in placed:
                if _intervals_overlap(obj["r0"], obj["r0"] + 2, other_r0, other_r0 + 2):
                    target_c = max(target_c, other_c0 + 3)
            assert target_c <= obj["c0"], "left-moving square would need to move right"
            final_by_color[obj["color"]] = (obj["r0"], target_c)
            placed.append((obj["r0"], target_c))
    elif direction == "right":
        order = sorted(group, key=lambda obj: (max_c - obj["c0"], obj["r0"], obj["c0"]))
        for obj in order:
            target_c = max_c
            for other_r0, other_c0 in placed:
                if _intervals_overlap(obj["r0"], obj["r0"] + 2, other_r0, other_r0 + 2):
                    target_c = min(target_c, other_c0 - 3)
            assert target_c >= obj["c0"], "right-moving square would need to move left"
            final_by_color[obj["color"]] = (obj["r0"], target_c)
            placed.append((obj["r0"], target_c))
    elif direction == "up":
        order = sorted(group, key=lambda obj: (obj["r0"], obj["r0"], obj["c0"]))
        for obj in order:
            target_r = 0
            for other_r0, other_c0 in placed:
                if _intervals_overlap(obj["c0"], obj["c0"] + 2, other_c0, other_c0 + 2):
                    target_r = max(target_r, other_r0 + 3)
            assert target_r <= obj["r0"], "up-moving square would need to move down"
            final_by_color[obj["color"]] = (target_r, obj["c0"])
            placed.append((target_r, obj["c0"]))
    else:
        assert direction == "down"
        order = sorted(group, key=lambda obj: (max_r - obj["r0"], obj["r0"], obj["c0"]))
        for obj in order:
            target_r = max_r
            for other_r0, other_c0 in placed:
                if _intervals_overlap(obj["c0"], obj["c0"] + 2, other_c0, other_c0 + 2):
                    target_r = min(target_r, other_r0 - 3)
            assert target_r >= obj["r0"], "down-moving square would need to move up"
            final_by_color[obj["color"]] = (target_r, obj["c0"])
            placed.append((target_r, obj["c0"]))

    return final_by_color


def _final_positions(objects: list[dict], height: int, width: int) -> dict[int, tuple[int, int]]:
    final_by_color = {
        obj["color"]: (obj["r0"], obj["c0"])
        for obj in objects
    }
    for direction in ("left", "right", "up", "down"):
        group = [obj for obj in objects if obj["direction"] == direction]
        if group:
            final_by_color.update(_pack_group(group, direction, height, width))
    return final_by_color


def _assert_non_overlapping(final_by_color: dict[int, tuple[int, int]], objects: list[dict]) -> None:
    boxes: list[tuple[int, int, int, int, int]] = []
    for obj in objects:
        r0, c0 = final_by_color[obj["color"]]
        boxes.append((r0, c0, r0 + 2, c0 + 2, obj["color"]))
    for i in range(len(boxes)):
        a = boxes[i]
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if _intervals_overlap(a[0], a[2], b[0], b[2]) and _intervals_overlap(a[1], a[3], b[1], b[3]):
                raise AssertionError(f"final squares overlap: colors {a[4]} and {b[4]}")


def _render_output_like_input(grid: np.ndarray, objects: list[dict], final_by_color: dict[int, tuple[int, int]]) -> np.ndarray:
    out = np.zeros_like(grid)
    out[:, -1] = grid[:, -1]
    out[-1, :] = grid[-1, :]
    for obj in objects:
        r0, c0 = final_by_color[obj["color"]]
        out[r0 : r0 + 3, c0 : c0 + 3] = _make_sprite(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate hollow squares only where the row and column labels match, then slide them in the direction encoded by that shared label.

    Input structure:
    - The grid is black except for a right-side row-label column, a bottom column-label row, and 2-4 distinct 3x3 hollow squares in the interior.
    - Interior square colors come from {blue, red, orange, maroon}; label colors come from {green, yellow, pink, teal} or black.
    - Each square is fully inside rows 0..H-2 and cols 0..W-2, so the last row and last column are reserved for labels.

    Rule:
    - For each hollow square, read the right-edge label on its center row and the bottom-edge label on its center column.
    - If those two labels match and are non-black, the shared color activates the square and encodes its direction: green=left, yellow=down, pink=up, teal=right.
    - Erase every activated square from its source location and slide it to the corresponding interior boundary; squares moving toward the same boundary pack in priority order, with the nearer one reaching the boundary first and ties broken by smaller row then smaller column.
    - Squares whose two labels differ, or use black, stay in place; the label row and label column are unchanged.

    Key insight:
    - The row/column label match is a gate, not a paint instruction: the matching color simultaneously decides whether a square moves and which direction that movement takes.
    """
    g = np.asarray(grid, dtype=int)
    objects = _decorate_objects(_extract_objects(g), g)
    final_by_color = _final_positions(objects, *g.shape)
    _assert_non_overlapping(final_by_color, objects)
    return _render_output_like_input(g, objects, final_by_color)


def _can_place(candidate: dict, placed: list[dict], height: int, width: int) -> bool:
    r0, c0 = candidate["r0"], candidate["c0"]
    if not (0 <= r0 <= height - 4 and 0 <= c0 <= width - 4):
        return False
    center_r, center_c = r0 + 1, c0 + 1
    for obj in placed:
        if center_r == obj["r0"] + 1 or center_c == obj["c0"] + 1:
            return False
        or0, oc0, or1, oc1 = _bbox_of(obj)
        if _intervals_overlap(r0, r0 + 2, or0, or1) and _intervals_overlap(c0, c0 + 2, oc0, oc1):
            return False
    if candidate["direction"] == "left" and c0 == 0:
        return False
    if candidate["direction"] == "right" and c0 == width - 4:
        return False
    if candidate["direction"] == "up" and r0 == 0:
        return False
    if candidate["direction"] == "down" and r0 == height - 4:
        return False
    return True


def _sample_random_object(
    placed: list[dict],
    height: int,
    width: int,
    direction: str | None,
    max_tries: int = 80,
) -> dict | None:
    max_r = height - 4
    max_c = width - 4
    for _ in range(max_tries):
        candidate = {
            "r0": random.randint(0, max_r),
            "c0": random.randint(0, max_c),
            "direction": direction,
        }
        if _can_place(candidate, placed, height, width):
            return candidate
    return None


def _place_collision_group(
    scenario: str,
    placed: list[dict],
    height: int,
    width: int,
) -> list[dict] | None:
    max_r = height - 4
    max_c = width - 4
    vertical = scenario == "vertical_collision"
    if vertical:
        direction = random.choice(("up", "down"))
        count = 3 if max_r >= 7 else 2
        if max_c < 2:
            return None
        base_c = random.randint(0, max_c - 2)
        c_offsets = random.sample((0, 1, 2), count)
        cols = [base_c + off for off in c_offsets]
        cols.sort()
        if direction == "up":
            base_r = random.randint(1, max_r - 3 * (count - 1))
            rows = [base_r + 3 * idx for idx in range(count)]
        else:
            base_r = random.randint(1, max_r - 3 * (count - 1))
            rows = [max_r - base_r - 3 * idx for idx in range(count)]
        group = [
            {"r0": rows[idx], "c0": cols[idx], "direction": direction}
            for idx in range(count)
        ]
    else:
        direction = random.choice(("left", "right"))
        count = 3 if max_c >= 7 else 2
        if max_r < 2:
            return None
        base_r = random.randint(0, max_r - 2)
        r_offsets = random.sample((0, 1, 2), count)
        rows = [base_r + off for off in r_offsets]
        rows.sort()
        if direction == "left":
            base_c = random.randint(1, max_c - 3 * (count - 1))
            cols = [base_c + 3 * idx for idx in range(count)]
        else:
            base_c = random.randint(1, max_c - 3 * (count - 1))
            cols = [max_c - base_c - 3 * idx for idx in range(count)]
        group = [
            {"r0": rows[idx], "c0": cols[idx], "direction": direction}
            for idx in range(count)
        ]

    test_placed = list(placed)
    for candidate in group:
        if not _can_place(candidate, test_placed, height, width):
            return None
        test_placed.append(candidate)
    return group


def _build_plan(height: int, width: int, scenario: str) -> dict | None:
    count = 4 if scenario == "all_directions" else random.randint(2, 4)
    placed: list[dict] = []
    objects: list[dict] = []

    if scenario in {"horizontal_collision", "vertical_collision"}:
        group = _place_collision_group(scenario, placed, height, width)
        if group is None:
            return None
        placed.extend(group)
        objects.extend(group)
        while len(objects) < count:
            candidate = _sample_random_object(placed, height, width, None)
            if candidate is None:
                return None
            placed.append(candidate)
            objects.append(candidate)
    elif scenario == "all_directions":
        for direction in random.sample(("left", "right", "up", "down"), 4):
            candidate = _sample_random_object(placed, height, width, direction)
            if candidate is None:
                return None
            placed.append(candidate)
            objects.append(candidate)
    elif scenario == "single_active":
        active_index = random.randrange(count)
        directions = [None] * count
        directions[active_index] = random.choice(("left", "right", "up", "down"))
        for direction in directions:
            candidate = _sample_random_object(placed, height, width, direction)
            if candidate is None:
                return None
            placed.append(candidate)
            objects.append(candidate)
    else:
        active_count = random.randint(2, min(3, count))
        active_slots = set(random.sample(range(count), active_count))
        active_directions = [
            random.choice(("left", "right", "up", "down")) if idx in active_slots else None
            for idx in range(count)
        ]
        for direction in active_directions:
            candidate = _sample_random_object(placed, height, width, direction)
            if candidate is None:
                return None
            placed.append(candidate)
            objects.append(candidate)

    colors = random.sample(OBJECT_COLORS, len(objects))
    for idx, obj in enumerate(objects):
        obj["color"] = colors[idx]
    return {"height": height, "width": width, "scenario": scenario, "objects": objects}


def _random_extra_label() -> int:
    return random.choice((0, 0, 0, 3, 4, 6, 8))


def _labels_for_plan(plan: dict) -> tuple[np.ndarray, np.ndarray]:
    height = plan["height"]
    width = plan["width"]
    row_labels = np.array([_random_extra_label() for _ in range(height)], dtype=int)
    col_labels = np.array([_random_extra_label() for _ in range(width)], dtype=int)

    for obj in plan["objects"]:
        center_r = obj["r0"] + 1
        center_c = obj["c0"] + 1
        if obj["direction"] is None:
            row_label, col_label = random.sample(LABEL_COLORS, 2)
        else:
            row_label = next(label for label, direction in DIR_BY_LABEL.items() if direction == obj["direction"])
            col_label = row_label
        row_labels[center_r] = row_label
        col_labels[center_c] = col_label
        obj["center_r"] = center_r
        obj["center_c"] = center_c
        obj["row_label"] = int(row_label)
        obj["col_label"] = int(col_label)

    corner = _random_extra_label()
    row_labels[height - 1] = corner
    col_labels[width - 1] = corner
    return row_labels, col_labels


def _render_plan_input(plan: dict, row_labels: np.ndarray, col_labels: np.ndarray) -> np.ndarray:
    grid = np.zeros((plan["height"], plan["width"]), dtype=int)
    for obj in plan["objects"]:
        r0, c0 = obj["r0"], obj["c0"]
        grid[r0 : r0 + 3, c0 : c0 + 3] = _make_sprite(obj["color"])
    grid[:, -1] = row_labels
    grid[-1, :] = col_labels
    return grid


def _active_collision_happens(objects: list[dict], final_by_color: dict[int, tuple[int, int]], height: int, width: int) -> bool:
    for direction in ("left", "right", "up", "down"):
        group = [obj for obj in objects if obj["direction"] == direction]
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]
                b = group[j]
                if direction in {"left", "right"}:
                    if not _intervals_overlap(a["r0"], a["r0"] + 2, b["r0"], b["r0"] + 2):
                        continue
                else:
                    if not _intervals_overlap(a["c0"], a["c0"] + 2, b["c0"], b["c0"] + 2):
                        continue
                if final_by_color[a["color"]] != _naive_target(a, height, width):
                    return True
                if final_by_color[b["color"]] != _naive_target(b, height, width):
                    return True
    return False


def _validate_plan(plan: dict, grid: np.ndarray, out: np.ndarray) -> bool:
    try:
        parsed = _decorate_objects(_extract_objects(grid), grid)
    except AssertionError:
        return False

    if len(parsed) != len(plan["objects"]):
        return False

    expected_by_color = {
        obj["color"]: (obj["r0"], obj["c0"], obj["direction"], obj["row_label"], obj["col_label"])
        for obj in plan["objects"]
    }
    for obj in parsed:
        if obj["color"] not in expected_by_color:
            return False
        if (
            obj["r0"],
            obj["c0"],
            obj["direction"],
            obj["row_label"],
            obj["col_label"],
        ) != expected_by_color[obj["color"]]:
            return False

    active_count = sum(obj["direction"] is not None for obj in plan["objects"])
    if active_count == 0:
        return False
    if any(obj["row_label"] == 0 or obj["col_label"] == 0 for obj in plan["objects"]):
        return False

    final_by_color = _final_positions(plan["objects"], plan["height"], plan["width"])
    try:
        _assert_non_overlapping(final_by_color, plan["objects"])
    except AssertionError:
        return False

    if plan["scenario"] in {"horizontal_collision", "vertical_collision"}:
        if not _active_collision_happens(plan["objects"], final_by_color, plan["height"], plan["width"]):
            return False
    if plan["scenario"] != "all_directions":
        inactive_count = sum(obj["direction"] is None for obj in plan["objects"])
        if inactive_count == 0:
            return False

    expected_out = _render_output_like_input(grid, plan["objects"], final_by_color)
    return np.array_equal(out, expected_out)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GEN_SIZE_RANGE)
    width = random.randint(*GEN_SIZE_RANGE)
    scenario = random.choices(GEN_SCENARIOS, weights=GEN_SCENARIO_WEIGHTS, k=1)[0]
    plan = _build_plan(height, width, scenario)
    if plan is None:
        return None

    row_labels, col_labels = _labels_for_plan(plan)
    grid = _render_plan_input(plan, row_labels, col_labels)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not _validate_plan(plan, grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gated-sliding ARC input")
