from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG_COLOR = 0
RULER_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _detect_ruler_color(grid: np.ndarray, bg: int) -> int:
    top = Counter(int(v) for v in grid[0, :] if int(v) != bg)
    left = Counter(int(v) for v in grid[:, 0] if int(v) != bg)
    shared = [color for color in top if color in left]
    if shared:
        return int(max(shared, key=lambda color: top[color] + left[color]))

    border = Counter(int(v) for v in np.concatenate((grid[0, :], grid[:, 0])) if int(v) != bg)
    assert border, "expected a visible ruler color on the top row and left column"
    return int(border.most_common(1)[0][0])


def _tick_map_from_border(line: np.ndarray, ruler_color: int) -> dict[int, int]:
    positions = [idx for idx in range(1, len(line)) if int(line[idx]) == int(ruler_color)]
    assert positions, "expected at least one interior ruler tick"
    return {int(pos): int(i + 1) for i, pos in enumerate(positions)}


def _extract_objects(
    grid: np.ndarray,
    bg: int,
    ruler_color: int,
    row_ticks: dict[int, int],
    col_ticks: dict[int, int],
) -> list[dict[str, int]]:
    objects: list[dict[str, int]] = []
    for color in map(int, np.unique(grid)):
        if color in (int(bg), int(ruler_color)):
            continue

        coords = np.argwhere(grid == int(color))
        if int(coords.shape[0]) != 4:
            continue

        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if (int(r1 - r0), int(c1 - c0)) != (1, 1):
            continue
        if not bool((grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)).all()):
            continue
        if r0 not in row_ticks or c0 not in col_ticks:
            continue

        objects.append(
            {
                "color": int(color),
                "row": int(r0),
                "col": int(c0),
                "key": int(min(row_ticks[r0], col_ticks[c0])),
            }
        )

    assert objects, "expected at least one colored 2x2 object"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each 2x2 object's ruler-defined number and stack the objects in ascending-number order.

    Input structure:
    - The background is a dominant color.
    - The top row and left column contain a shared ruler color, with isolated tick pixels at every other position.
    - The interior contains several distinct-colored 2x2 solid squares whose top-left corners align with one row tick and one column tick.

    Rule:
    - Number the left-column ticks from top to bottom and the top-row ticks from left to right, ignoring the corner cell.
    - For each 2x2 square, read its row tick value R and column tick value C from the aligned top-left corner, then assign the square the value min(R, C).
    - Build an output grid with the same width as the input and height equal to twice the number of objects, then place the objects from smallest number to largest, breaking ties by top-to-bottom then left-to-right input order, with each 2x2 square left-aligned in the first two columns.

    Key insight:
    - The sparse border pixels act as coordinate rulers; the hidden sort key is the smaller of the two ruler indices aligned with each object's top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    bg = _most_common_color(g)
    ruler_color = _detect_ruler_color(g, bg)
    row_ticks = _tick_map_from_border(g[:, 0], ruler_color)
    col_ticks = _tick_map_from_border(g[0, :], ruler_color)
    objects = _extract_objects(g, bg, ruler_color, row_ticks, col_ticks)

    ordered = sorted(objects, key=lambda obj: (int(obj["key"]), int(obj["row"]), int(obj["col"])))

    out = np.full((2 * len(ordered), w), int(bg), dtype=int)
    for i, obj in enumerate(ordered):
        out[2 * i : 2 * i + 2, 0:2] = int(obj["color"])
    return out


def _build_ticks(start: int, limit: int) -> list[int]:
    return [pos for pos in range(start, limit, 2)]


def _placement_slots(
    height: int,
    width: int,
    row_start: int,
    col_start: int,
    row_phase: int,
    col_phase: int,
) -> tuple[dict[int, int], dict[int, int], list[dict[str, int]]]:
    tick_rows = _build_ticks(row_start, height)
    tick_cols = _build_ticks(col_start, width)
    row_ticks = {int(r): int(i + 1) for i, r in enumerate(tick_rows)}
    col_ticks = {int(c): int(i + 1) for i, c in enumerate(tick_cols)}

    place_rows = [r for r in tick_rows if r + 1 < height][row_phase::2]
    place_cols = [c for c in tick_cols if c + 1 < width][col_phase::2]

    slots: list[dict[str, int]] = []
    for r in place_rows:
        for c in place_cols:
            row_value = row_ticks[int(r)]
            col_value = col_ticks[int(c)]
            if row_value < col_value:
                category = "row"
            elif row_value > col_value:
                category = "col"
            else:
                category = "eq"
            slots.append(
                {
                    "row": int(r),
                    "col": int(c),
                    "row_value": int(row_value),
                    "col_value": int(col_value),
                    "key": int(min(row_value, col_value)),
                    "category": category,
                }
            )

    return row_ticks, col_ticks, slots


def _sample_slots(slots: list[dict[str, int]]) -> list[dict[str, int]] | None:
    if len(slots) < 3:
        return None

    categories = {slot["category"] for slot in slots}
    if "row" not in categories or "col" not in categories:
        return None

    eq_slots = [slot for slot in slots if slot["category"] == "eq"]
    distinct_key_count = len({slot["key"] for slot in slots})
    if distinct_key_count < 3:
        return None

    want_duplicate = bool(random.getrandbits(1))
    require_eq = bool(eq_slots) and bool(random.getrandbits(1))

    max_objects = min(len(slots), len(OBJECT_COLORS))
    if want_duplicate:
        object_count = random.randint(3, max_objects)
    else:
        max_distinct = min(max_objects, distinct_key_count)
        if max_distinct < 3:
            return None
        object_count = random.randint(3, max_distinct)

    for _ in range(300):
        chosen = random.sample(slots, object_count)
        chosen_categories = {slot["category"] for slot in chosen}
        if "row" not in chosen_categories or "col" not in chosen_categories:
            continue
        if require_eq and "eq" not in chosen_categories:
            continue

        keys = [slot["key"] for slot in chosen]
        unique_key_count = len(set(keys))
        if want_duplicate and unique_key_count == len(keys):
            continue
        if not want_duplicate and unique_key_count != len(keys):
            continue

        return chosen

    return None


def _generated_grid_is_valid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    bg = _most_common_color(grid)
    if int(bg) != BG_COLOR:
        return False

    ruler_color = _detect_ruler_color(grid, bg)
    if int(ruler_color) != RULER_COLOR:
        return False

    row_ticks = _tick_map_from_border(grid[:, 0], ruler_color)
    col_ticks = _tick_map_from_border(grid[0, :], ruler_color)
    objects = _extract_objects(grid, bg, ruler_color, row_ticks, col_ticks)

    if len(objects) < 3:
        return False

    occupied = set()
    for obj in objects:
        cells = {(obj["row"] + dr, obj["col"] + dc) for dr in range(2) for dc in range(2)}
        if occupied & cells:
            return False
        occupied |= cells

    try:
        out = main(grid)
    except Exception:
        return False

    return out.ndim == 2 and out.shape == (2 * len(objects), grid.shape[1]) and max(out.shape) <= 30


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 30)
    width = random.randint(12, 30)
    row_start = random.choice((1, 2))
    col_start = random.choice((1, 2))
    row_phase = random.choice((0, 1))
    col_phase = random.choice((0, 1))

    row_ticks, col_ticks, slots = _placement_slots(height, width, row_start, col_start, row_phase, col_phase)
    if len(row_ticks) < 4 or len(col_ticks) < 4:
        return None

    chosen_slots = _sample_slots(slots)
    if chosen_slots is None:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    for row in row_ticks:
        grid[int(row), 0] = RULER_COLOR
    for col in col_ticks:
        grid[0, int(col)] = RULER_COLOR
    if bool(random.getrandbits(1)):
        grid[0, 0] = RULER_COLOR

    colors = random.sample(list(OBJECT_COLORS), len(chosen_slots))
    random.shuffle(chosen_slots)
    for slot, color in zip(chosen_slots, colors):
        r = int(slot["row"])
        c = int(slot["col"])
        grid[r : r + 2, c : c + 2] = int(color)

    if not _generated_grid_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
