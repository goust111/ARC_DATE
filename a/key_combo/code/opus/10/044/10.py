from __future__ import annotations

import random
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

WIDTH_CHOICES = (3, 4, 5)
DEPTH_CHOICES = (2, 3, 4)
MAX_MARKERS = 6
MAX_ATTEMPTS = 500


def _copy_int_grid(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid, dtype=int).copy()


def _adjacent(cup_a: dict[str, int], cup_b: dict[str, int]) -> bool:
    return int(cup_b["left"]) - int(cup_a["right"]) <= 2


def _detect_cups(grid: np.ndarray) -> list[dict[str, Any]]:
    g = _copy_int_grid(grid)
    h, w = g.shape
    found: dict[tuple[int, int, int], dict[str, Any]] = {}

    for bottom in range(h):
        for left in range(w):
            if g[bottom, left] != BLUE:
                continue
            for width in WIDTH_CHOICES:
                right = left + width - 1
                if right >= w:
                    continue
                if not np.all(g[bottom, left : right + 1] == BLUE):
                    continue

                top_left = bottom
                while top_left > 0 and g[top_left - 1, left] == BLUE:
                    top_left -= 1
                top_right = bottom
                while top_right > 0 and g[top_right - 1, right] == BLUE:
                    top_right -= 1

                top = max(top_left, top_right)
                depth = bottom - top
                if depth not in DEPTH_CHOICES:
                    continue
                if width < 3:
                    continue
                if not np.all(g[top : bottom + 1, left] == BLUE):
                    continue
                if not np.all(g[top : bottom + 1, right] == BLUE):
                    continue
                if np.any(g[top:bottom, left + 1 : right] == BLUE):
                    continue

                interior_rows = list(range(top, bottom))
                interior_cols = list(range(left + 1, right))
                if not interior_rows or not interior_cols:
                    continue

                key = (left, right, bottom)
                found[key] = {
                    "left": left,
                    "right": right,
                    "top": top,
                    "bottom": bottom,
                    "width": width,
                    "depth": depth,
                    "capacity": depth,
                    "area": depth * (width - 2),
                    "markers": int(np.sum(g[top:bottom, left + 1 : right] == RED)),
                }

    cups = sorted(found.values(), key=lambda cup: (int(cup["left"]), int(cup["right"])))
    if not cups:
        raise ValueError("no cups detected")
    return cups


def _simulate_amounts(
    cups: list[dict[str, Any]], start_amounts: list[int]
) -> tuple[list[int], list[tuple[int, int, int]]]:
    amounts = [int(v) for v in start_amounts]
    events: list[tuple[int, int, int]] = []

    for _ in range(16):
        overfull = [i for i, cup in enumerate(cups) if amounts[i] > int(cup["capacity"])]
        if not overfull:
            return amounts, events

        moved = False
        for i in overfull:
            capacity = int(cups[i]["capacity"])
            if amounts[i] <= capacity:
                continue

            overflow = amounts[i] - capacity
            choices: list[tuple[int, int, int]] = []

            if i > 0 and _adjacent(cups[i - 1], cups[i]):
                spare = int(cups[i - 1]["capacity"]) - amounts[i - 1]
                if spare > 0:
                    choices.append((-spare, 0, i - 1))
            if i + 1 < len(cups) and _adjacent(cups[i], cups[i + 1]):
                spare = int(cups[i + 1]["capacity"]) - amounts[i + 1]
                if spare > 0:
                    choices.append((-spare, 1, i + 1))

            if not choices:
                raise ValueError("overflow has no eligible neighbor")

            choices.sort()
            target = int(choices[0][2])
            amounts[i] -= overflow
            amounts[target] += overflow
            events.append((i, target, overflow))
            moved = True

        if not moved:
            break

    raise ValueError("overflow did not stabilize")


def _render_output(
    grid: np.ndarray, cups: list[dict[str, Any]], amounts: list[int]
) -> np.ndarray:
    out = _copy_int_grid(grid)

    for cup, amount in zip(cups, amounts):
        left = int(cup["left"])
        right = int(cup["right"])
        top = int(cup["top"])
        bottom = int(cup["bottom"])
        depth = int(cup["depth"])
        fill_layers = min(int(amount), depth)

        out[top:bottom, left + 1 : right] = BLACK
        for offset in range(fill_layers):
            row = bottom - 1 - offset
            out[row, left + 1 : right] = RED

        if fill_layers == depth and top > 0:
            out[top - 1, left : right + 1] = RED

    return out


def _solve(grid: np.ndarray) -> tuple[list[dict[str, Any]], list[int], list[tuple[int, int, int]]]:
    cups = _detect_cups(grid)
    start_amounts = [int(cup["markers"]) for cup in cups]
    final_amounts, events = _simulate_amounts(cups, start_amounts)
    return cups, final_amounts, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each cup by liquid layers, redirecting overflow toward the best adjacent receiver.

    Input structure:
    - Black background with 3-4 blue U-shaped cups arranged horizontally; cups may share a wall or be separated by one blank column.
    - Each cup opens upward, and red marker pixels are scattered inside its lower interior.
    - The cup outline fixes its left/right walls, its base, and therefore its interior depth in rows.

    Rule:
    - Detect every blue cup and count its red markers.
    - Treat each cup's capacity as its interior depth: one marker contributes one horizontal liquid layer.
    - Fill each cup from the bottom upward; if a cup has more layers than its depth, send the overflow to the adjacent underfilled cup with the largest remaining capacity, breaking ties to the left, and continue until all cups fit.
    - Any cup that ends exactly full receives a red cap line one row above its opening.

    Key insight:
    - The red pixels encode volume, not final paint locations; you must combine cup geometry with neighbor-aware overflow simulation before drawing the final liquid levels.
    """
    g = _copy_int_grid(grid)
    cups, final_amounts, _events = _solve(g)
    return _render_output(g, cups, final_amounts)


def _draw_cups(
    widths: list[int], depths: list[int], gap_kinds: list[int]
) -> tuple[np.ndarray, list[dict[str, int]]]:
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)

    positions: list[dict[str, int]] = []
    left = left_margin
    max_depth = max(depths)
    baseline = top_margin + max_depth

    for i, (width, depth) in enumerate(zip(widths, depths)):
        right = left + width - 1
        top = baseline - depth
        bottom = baseline
        positions.append(
            {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "width": width,
                "depth": depth,
                "capacity": depth,
                "area": depth * (width - 2),
            }
        )
        if i + 1 < len(widths):
            left = right if gap_kinds[i] == 0 else right + 2

    total_width = positions[-1]["right"] + 1 + right_margin
    total_height = baseline + 1 + bottom_margin
    if total_width > 30 or total_height > 30:
        raise ValueError("grid would exceed ARC size limit")

    grid = np.full((total_height, total_width), BLACK, dtype=int)
    for cup in positions:
        left = int(cup["left"])
        right = int(cup["right"])
        top = int(cup["top"])
        bottom = int(cup["bottom"])
        grid[top : bottom + 1, left] = BLUE
        grid[top : bottom + 1, right] = BLUE
        grid[bottom, left : right + 1] = BLUE

    return grid, positions


def _pick_nonempty_amount(max_value: int) -> int:
    if max_value < 1:
        raise ValueError("amount bound must be positive")
    return random.randint(1, max_value)


def _fill_other_amounts(
    amounts: list[int], cups: list[dict[str, int]], locked: set[int]
) -> None:
    for i, cup in enumerate(cups):
        if i in locked:
            continue
        limit = min(MAX_MARKERS, int(cup["area"]))
        max_safe = min(int(cup["capacity"]), limit)
        amounts[i] = _pick_nonempty_amount(max_safe)


def _ensure_underfilled(amounts: list[int], cups: list[dict[str, int]], protected: set[int]) -> bool:
    for i, cup in enumerate(cups):
        if i in protected:
            continue
        if int(cup["capacity"]) >= 2:
            amounts[i] = min(amounts[i], int(cup["capacity"]) - 1)
            if amounts[i] >= 1:
                return True
    return any(amounts[i] < int(cups[i]["capacity"]) for i in range(len(cups)))


def _assign_mode(
    cups: list[dict[str, int]], mode: str
) -> tuple[list[int], dict[str, Any]]:
    n = len(cups)
    limits = [min(MAX_MARKERS, int(cup["area"])) for cup in cups]
    amounts = [0] * n
    meta: dict[str, Any] = {"mode": mode}

    if mode == "direct_right":
        source = random.randint(0, n - 2)
        target = source + 1
        max_over = limits[source] - int(cups[source]["capacity"])
        if max_over < 1 or int(cups[target]["capacity"]) < 2:
            raise ValueError("direct_right infeasible")
        overflow = random.randint(1, min(2, max_over, int(cups[target]["capacity"]) - 1))
        target_spare = random.randint(overflow, int(cups[target]["capacity"]) - 1)
        amounts[source] = int(cups[source]["capacity"]) + overflow
        amounts[target] = int(cups[target]["capacity"]) - target_spare
        locked = {source, target}
        _fill_other_amounts(amounts, cups, locked)
        if source > 0:
            other = source - 1
            max_spare = max(0, target_spare - 1)
            spare = random.randint(0, min(max_spare, int(cups[other]["capacity"]) - 1))
            amounts[other] = int(cups[other]["capacity"]) - spare
            locked.add(other)
        meta.update({"source": source, "target": target, "overflow": overflow})
        protected = {source}

    elif mode == "direct_left":
        source = random.randint(1, n - 1)
        target = source - 1
        max_over = limits[source] - int(cups[source]["capacity"])
        if max_over < 1 or int(cups[target]["capacity"]) < 2:
            raise ValueError("direct_left infeasible")
        overflow = random.randint(1, min(2, max_over, int(cups[target]["capacity"]) - 1))
        target_spare = random.randint(overflow, int(cups[target]["capacity"]) - 1)
        amounts[source] = int(cups[source]["capacity"]) + overflow
        amounts[target] = int(cups[target]["capacity"]) - target_spare
        locked = {source, target}
        _fill_other_amounts(amounts, cups, locked)
        if source + 1 < n:
            other = source + 1
            max_spare = max(0, target_spare - 1)
            spare = random.randint(0, min(max_spare, int(cups[other]["capacity"]) - 1))
            amounts[other] = int(cups[other]["capacity"]) - spare
            locked.add(other)
        meta.update({"source": source, "target": target, "overflow": overflow})
        protected = {source}

    elif mode == "tie_left":
        if n < 3:
            raise ValueError("tie_left requires at least 3 cups")
        source = random.randint(1, n - 2)
        left = source - 1
        right = source + 1
        max_over = limits[source] - int(cups[source]["capacity"])
        max_shared = min(int(cups[left]["capacity"]) - 1, int(cups[right]["capacity"]) - 1)
        if max_over < 1 or max_shared < 1:
            raise ValueError("tie_left infeasible")
        overflow = random.randint(1, min(2, max_over, max_shared))
        shared_spare = random.randint(overflow, max_shared)
        amounts[source] = int(cups[source]["capacity"]) + overflow
        amounts[left] = int(cups[left]["capacity"]) - shared_spare
        amounts[right] = int(cups[right]["capacity"]) - shared_spare
        locked = {source, left, right}
        _fill_other_amounts(amounts, cups, locked)
        meta.update(
            {
                "source": source,
                "target": left,
                "overflow": overflow,
                "shared_spare": shared_spare,
            }
        )
        protected = {source}

    elif mode == "cascade_right":
        if n < 3:
            raise ValueError("cascade_right requires at least 3 cups")
        source = random.randint(0, n - 3)
        middle = source + 1
        target = source + 2
        max_over = limits[source] - int(cups[source]["capacity"])
        max_middle_spare = int(cups[middle]["capacity"]) - 1
        max_target_spare = int(cups[target]["capacity"]) - 1
        if max_over < 2 or max_middle_spare < 1 or max_target_spare < 1:
            raise ValueError("cascade_right infeasible")
        overflow = random.randint(2, min(3, max_over))
        middle_spare = random.randint(1, min(overflow - 1, max_middle_spare))
        second_overflow = overflow - middle_spare
        if max_target_spare < second_overflow:
            raise ValueError("cascade_right target too small")
        target_spare = random.randint(second_overflow, max_target_spare)
        amounts[source] = int(cups[source]["capacity"]) + overflow
        amounts[middle] = int(cups[middle]["capacity"]) - middle_spare
        amounts[target] = int(cups[target]["capacity"]) - target_spare
        locked = {source, middle, target}
        _fill_other_amounts(amounts, cups, locked)
        if source > 0:
            other = source - 1
            spare = random.randint(0, min(max(0, middle_spare - 1), int(cups[other]["capacity"]) - 1))
            amounts[other] = int(cups[other]["capacity"]) - spare
            locked.add(other)
        meta.update(
            {
                "source": source,
                "middle": middle,
                "target": target,
                "overflow": overflow,
                "middle_spare": middle_spare,
                "second_overflow": second_overflow,
            }
        )
        protected = {source, middle}

    elif mode == "cascade_left":
        if n < 3:
            raise ValueError("cascade_left requires at least 3 cups")
        source = random.randint(2, n - 1)
        middle = source - 1
        target = source - 2
        max_over = limits[source] - int(cups[source]["capacity"])
        max_middle_spare = int(cups[middle]["capacity"]) - 1
        max_target_spare = int(cups[target]["capacity"]) - 1
        if max_over < 2 or max_middle_spare < 1 or max_target_spare < 1:
            raise ValueError("cascade_left infeasible")
        overflow = random.randint(2, min(3, max_over))
        middle_spare = random.randint(1, min(overflow - 1, max_middle_spare))
        second_overflow = overflow - middle_spare
        if max_target_spare < second_overflow:
            raise ValueError("cascade_left target too small")
        target_spare = random.randint(second_overflow, max_target_spare)
        amounts[source] = int(cups[source]["capacity"]) + overflow
        amounts[middle] = int(cups[middle]["capacity"]) - middle_spare
        amounts[target] = int(cups[target]["capacity"]) - target_spare
        locked = {source, middle, target}
        _fill_other_amounts(amounts, cups, locked)
        if source + 1 < n:
            other = source + 1
            spare = random.randint(0, min(max(0, middle_spare - 1), int(cups[other]["capacity"]) - 1))
            amounts[other] = int(cups[other]["capacity"]) - spare
            locked.add(other)
        meta.update(
            {
                "source": source,
                "middle": middle,
                "target": target,
                "overflow": overflow,
                "middle_spare": middle_spare,
                "second_overflow": second_overflow,
            }
        )
        protected = {source, middle}

    else:
        raise ValueError(f"unknown mode: {mode}")

    for i, cup in enumerate(cups):
        limit = limits[i]
        if not (1 <= amounts[i] <= limit):
            raise ValueError("marker count out of bounds")

    if not _ensure_underfilled(amounts, cups, protected):
        raise ValueError("need at least one underfilled cup")

    return amounts, meta


def _place_markers(grid: np.ndarray, cups: list[dict[str, int]], amounts: list[int]) -> np.ndarray:
    out = _copy_int_grid(grid)
    for cup, amount in zip(cups, amounts):
        cells: list[tuple[int, int]] = []
        for row in range(int(cup["bottom"]) - 1, int(cup["top"]) - 1, -1):
            cols = list(range(int(cup["left"]) + 1, int(cup["right"])))
            random.shuffle(cols)
            for col in cols:
                cells.append((row, col))
        if amount > len(cells):
            raise ValueError("not enough room for markers")
        for row, col in cells[:amount]:
            out[row, col] = RED
    return out


def _naive_output(grid: np.ndarray, cups: list[dict[str, Any]]) -> np.ndarray:
    clipped = [min(int(cup["markers"]), int(cup["capacity"])) for cup in cups]
    return _render_output(grid, cups, clipped)


def _validate_generated(
    inp: np.ndarray,
    cups: list[dict[str, int]],
    amounts: list[int],
    meta: dict[str, Any],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    try:
        detected = _detect_cups(inp)
    except Exception:
        return False
    if len(detected) != len(cups):
        return False

    for want, got, amount in zip(cups, detected, amounts):
        for key in ("left", "right", "top", "bottom", "depth", "capacity", "area"):
            if int(want[key]) != int(got[key]):
                return False
        if int(got["markers"]) != int(amount):
            return False

    if not any(amounts[i] > int(cups[i]["capacity"]) for i in range(len(cups))):
        return False
    if sum(amounts) > sum(int(cup["capacity"]) for cup in cups):
        return False

    try:
        final_amounts, events = _simulate_amounts(cups, amounts)
    except Exception:
        return False

    if meta["mode"].startswith("cascade"):
        if len(events) != 2:
            return False
    else:
        if len(events) != 1:
            return False

    if meta["mode"] == "tie_left":
        source = int(meta["source"])
        left = source - 1
        right = source + 1
        left_spare = int(cups[left]["capacity"]) - amounts[left]
        right_spare = int(cups[right]["capacity"]) - amounts[right]
        if left_spare != right_spare:
            return False
        if events[0][1] != left:
            return False

    if meta["mode"] == "direct_right" and events[0][1] != int(meta["target"]):
        return False
    if meta["mode"] == "direct_left" and events[0][1] != int(meta["target"]):
        return False
    if meta["mode"] == "cascade_right":
        if [evt[1] for evt in events] != [int(meta["middle"]), int(meta["target"])]:
            return False
    if meta["mode"] == "cascade_left":
        if [evt[1] for evt in events] != [int(meta["middle"]), int(meta["target"])]:
            return False

    out = main(inp)
    if not isinstance(out, np.ndarray) or out.shape != inp.shape or max(out.shape) > 30:
        return False
    if not np.array_equal(out, _render_output(inp, cups, final_amounts)):
        return False
    if np.array_equal(out, _naive_output(inp, detected)):
        return False

    capacities = [int(cup["capacity"]) for cup in cups]
    full_count = sum(int(amount) == capacity for amount, capacity in zip(final_amounts, capacities))
    if full_count < 1:
        return False
    if not any(int(amount) < capacity for amount, capacity in zip(final_amounts, capacities)):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    mode = random.choice(
        ["direct_right", "direct_left", "tie_left", "cascade_right", "cascade_left"]
    )
    n_cups = random.randint(3, 4)
    widths = [random.choice(WIDTH_CHOICES) for _ in range(n_cups)]
    depths = [random.choice(DEPTH_CHOICES) for _ in range(n_cups)]
    gaps = [random.randint(0, 1) for _ in range(n_cups - 1)]

    try:
        base_grid, cups = _draw_cups(widths, depths, gaps)
        amounts, meta = _assign_mode(cups, mode)
        inp = _place_markers(base_grid, cups, amounts)
    except Exception:
        return None

    return inp if _validate_generated(inp, cups, amounts, meta) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-overflow example")
