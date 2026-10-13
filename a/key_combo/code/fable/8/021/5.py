from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
LANE_WIDTH = 6
LANE_GAP = 1


def _get_bg(grid: np.ndarray) -> int:
    """Local clone of arc_utils.get_bg with the same (r, c) grid conventions."""
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _extract_shelves(grid: np.ndarray, bg: int) -> dict[int, list[tuple[int, int, int]]]:
    """Return row -> [(c0, c1, color)] for every horizontal monochrome shelf segment."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    shelves: dict[int, list[tuple[int, int, int]]] = {}
    for r in range(1, h):
        c = 0
        row_segments: list[tuple[int, int, int]] = []
        while c < w:
            if int(g[r, c]) == int(bg):
                c += 1
                continue
            col = int(g[r, c])
            c0 = c
            while c + 1 < w and int(g[r, c + 1]) == int(col):
                c += 1
            c1 = c
            if c1 - c0 + 1 >= 2:
                row_segments.append((int(c0), int(c1), int(col)))
            c += 1
        if row_segments:
            shelves[int(r)] = row_segments
    return shelves


def _next_shelf(
    shelves: dict[int, list[tuple[int, int, int]]],
    start_row: int,
    col: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Find the first shelf below (start_row, col), returning (row, c0, c1, color)."""
    for r in range(int(start_row) + 1, int(height)):
        for c0, c1, color in shelves.get(int(r), []):
            if int(c0) <= int(col) <= int(c1):
                return int(r), int(c0), int(c1), int(color)
    return None


def _simulate_stone(
    shelves: dict[int, list[tuple[int, int, int]]],
    shape: tuple[int, int],
    start_col: int,
    initial_color: int,
    *,
    match_mode: str = "current",
    roll_mode: str = "near",
) -> tuple[int, int, int]:
    """Simulate one stone and return its final (row, col, color)."""
    h, _w = map(int, shape)
    row = 0
    col = int(start_col)
    initial = int(initial_color)
    current = int(initial_color)

    while True:
        hit = _next_shelf(shelves, row, col, h)
        if hit is None:
            return int(h - 1), int(col), int(current)

        shelf_row, c0, c1, shelf_color = hit
        compare_color = current if match_mode == "current" else initial
        if int(compare_color) == int(shelf_color):
            return int(shelf_row - 1), int(col), int(current)

        current = int(shelf_color)
        left_dist = int(col - c0)
        right_dist = int(c1 - col)
        assert left_dist != right_dist, "generator should forbid equidistant roll choices"

        if roll_mode == "near":
            go_left = left_dist < right_dist
        elif roll_mode == "far":
            go_left = left_dist > right_dist
        else:
            raise ValueError("roll_mode must be 'near' or 'far'")

        col = int(c0 - 1) if go_left else int(c1 + 1)
        row = int(shelf_row - 1)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Simulate recoloring rolling stones until each one sticks to a matching shelf or reaches the bottom.

    Input structure:
    - Black background with several single-pixel stones scattered on the top row.
    - Lower rows contain solid horizontal shelf segments; each shelf has a single visible color.
    - Each stone's path is independent, and every shelf hit is strictly nearer to one end than the other.

    Rule:
    - Each stone falls vertically down its current column.
    - When it reaches a shelf, compare the shelf color to the stone's current color.
    - If the colors match, the stone sticks in the cell directly above the shelf.
    - Otherwise the stone is recolored to the shelf color, rolls along the shelf top toward the nearer end,
      moves one cell past that end, and continues falling from there.
    - Output only the final resting stones on a black background; shelves and travel paths disappear.

    Key insight:
    - Matching uses the stone's evolving current color, not its original color, so the whole cascade history
      determines both the final resting place and the final color.
    """
    g = np.asarray(input_grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    shelves = _extract_shelves(g, bg)

    out = np.full((h, w), int(bg), dtype=int)
    for c in range(w):
        if int(g[0, c]) == int(bg):
            continue
        final_r, final_c, final_color = _simulate_stone(
            shelves,
            g.shape,
            c,
            int(g[0, c]),
            match_mode="current",
            roll_mode="near",
        )
        out[int(final_r), int(final_c)] = int(final_color)
    return out


def _stick_choices(local_col: int, lane_width: int) -> list[tuple[int, int]]:
    """All shelf segments containing local_col with unequal end distances."""
    choices: list[tuple[int, int]] = []
    for c0 in range(0, int(local_col) + 1):
        for c1 in range(int(local_col), int(lane_width)):
            length = int(c1 - c0 + 1)
            if length < 3 or length > 5:
                continue
            if int(local_col - c0) == int(c1 - local_col):
                continue
            choices.append((int(c0), int(c1)))
    return choices


def _roll_choices(local_col: int, lane_width: int, direction: int) -> list[tuple[int, int, int]]:
    """All shelf segments causing a unique left/right roll within one lane."""
    choices: list[tuple[int, int, int]] = []
    c = int(local_col)
    w = int(lane_width)
    if int(direction) < 0:
        for near in range(1, c + 1):
            overflow = c - near - 1
            if overflow < 0:
                continue
            for far in range(near + 1, w - c):
                c0 = c - near
                c1 = c + far
                length = c1 - c0 + 1
                if 3 <= length <= 5:
                    choices.append((int(c0), int(c1), int(overflow)))
    else:
        for near in range(1, w - c):
            overflow = c + near + 1
            if overflow >= w:
                continue
            for far in range(near + 1, c + 1):
                c0 = c - far
                c1 = c + near
                length = c1 - c0 + 1
                if 3 <= length <= 5:
                    choices.append((int(c0), int(c1), int(overflow)))
    return choices


def _event_rows(height: int, count: int) -> list[int] | None:
    """Sample strictly increasing shelf rows with clear vertical gaps."""
    rows: list[int] = []
    row = random.randint(2, 5)
    for idx in range(int(count)):
        rows.append(int(row))
        if idx != int(count) - 1:
            row += random.randint(2, 4)
    if rows[-1] > int(height) - 2:
        return None
    return rows


def _pick_color(exclude: set[int]) -> int:
    choices = [c for c in PALETTE if int(c) not in {int(x) for x in exclude}]
    if not choices:
        choices = list(PALETTE)
    return int(random.choice(choices))


def _build_lane(branch: str, height: int, lane_width: int) -> dict | None:
    """Build one independent lane and record its exact cascade metadata."""
    local_col = random.choice([2, 3])
    initial_color = random.choice(PALETTE)
    current_color = int(initial_color)
    current_col = int(local_col)

    if branch == "immediate":
        n_recolors = 0
    elif branch == "cascade":
        n_recolors = random.randint(1, 3)
    elif branch == "bottom":
        n_recolors = random.randint(1, 3)
    else:
        raise ValueError(f"unknown branch: {branch}")

    total_events = int(n_recolors + (0 if branch == "bottom" else 1))
    rows = _event_rows(height, total_events)
    if rows is None:
        return None

    events: list[dict] = []
    for idx in range(n_recolors):
        roll_options: list[tuple[int, int, int, int]] = []
        for direction in (-1, 1):
            for c0, c1, overflow in _roll_choices(current_col, lane_width, direction):
                roll_options.append((int(c0), int(c1), int(overflow), int(direction)))
        if not roll_options:
            return None

        c0, c1, overflow, direction = random.choice(roll_options)
        exclude = {current_color}
        if branch == "cascade" and idx == n_recolors - 1:
            exclude.add(initial_color)
        shelf_color = _pick_color(exclude)
        events.append(
            {
                "row": int(rows[idx]),
                "c0": int(c0),
                "c1": int(c1),
                "color": int(shelf_color),
                "entry_col": int(current_col),
                "overflow_col": int(overflow),
                "direction": int(direction),
                "kind": "recolor",
            }
        )
        current_color = int(shelf_color)
        current_col = int(overflow)

    if branch != "bottom":
        stick_row = int(rows[-1])
        stick_options = _stick_choices(current_col, lane_width)
        if not stick_options:
            return None
        c0, c1 = random.choice(stick_options)
        events.append(
            {
                "row": int(stick_row),
                "c0": int(c0),
                "c1": int(c1),
                "color": int(current_color),
                "entry_col": int(current_col),
                "overflow_col": None,
                "direction": 0,
                "kind": "stick",
            }
        )
        final = (int(stick_row - 1), int(current_col), int(current_color))
    else:
        final = (int(height - 1), int(current_col), int(current_color))

    return {
        "branch": str(branch),
        "initial_col": int(local_col),
        "initial_color": int(initial_color),
        "events": events,
        "final": final,
    }


def _render_case(height: int, lanes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    width = len(lanes) * LANE_WIDTH + max(0, len(lanes) - 1) * LANE_GAP
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)

    for lane_idx, lane in enumerate(lanes):
        offset = lane_idx * (LANE_WIDTH + LANE_GAP)
        stone_c = offset + int(lane["initial_col"])
        inp[0, stone_c] = int(lane["initial_color"])

        for event in lane["events"]:
            row = int(event["row"])
            c0 = offset + int(event["c0"])
            c1 = offset + int(event["c1"])
            inp[row, c0 : c1 + 1] = int(event["color"])

        final_r, final_local_c, final_color = lane["final"]
        out[int(final_r), offset + int(final_local_c)] = int(final_color)

    return inp, out


def _generate_case(return_meta: bool = False):
    """Construct a valid non-interfering puzzle instance plus its oracle output."""
    for _ in range(300):
        lane_count = random.randint(3, 4)
        height = random.randint(12, 24)

        branches = ["immediate", "cascade", "bottom"]
        if lane_count == 4:
            branches.append(random.choice(["cascade", "bottom", "immediate"]))
        random.shuffle(branches)

        lanes: list[dict] = []
        ok = True
        for branch in branches:
            lane = _build_lane(branch, height, LANE_WIDTH)
            if lane is None:
                ok = False
                break
            lanes.append(lane)
        if not ok:
            continue

        # At least one non-immediate lane must prove current-color semantics.
        if not any(
            lane["branch"] == "cascade" and int(lane["initial_color"]) != int(lane["final"][2])
            for lane in lanes
        ):
            continue

        inp, expected = _render_case(height, lanes)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue

        meta = {"height": int(height), "lanes": lanes, "expected": expected}
        if return_meta:
            return inp, meta
        return inp

    raise RuntimeError("failed to generate a valid rolling-stones instance")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)

