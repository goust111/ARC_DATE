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
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9

    NOT_BLACK = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


def get_bg(grid: np.ndarray) -> int:
    """Mirror arc_utils.get_bg: background is the most frequent color."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    """Minimal 4-connected monochrome component finder in arc_utils style."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _parse_scene(grid: np.ndarray) -> tuple[int, list[int], dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape
    assert h >= 4, "need a top schedule row plus room for floating bars"

    bg = get_bg(g)
    schedule_cols = [int(c) for c in np.flatnonzero(g[0] != bg)]
    assert schedule_cols, "expected a non-empty top-row schedule"
    schedule = [int(g[0, c]) for c in schedule_cols]
    assert len(schedule) == len(set(schedule)), "schedule colors must be unique"

    bars: dict[int, dict] = {}
    for comp in _find_components(g[1:], bg):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        row = r0 + 1
        assert r0 == r1, "every floating object must be one cell tall"
        assert int(comp["area"]) == c1 - c0 + 1, "every floating object must be a solid horizontal bar"
        color = int(comp["color"])
        assert color not in bars, "bar colors must be unique"
        bars[color] = {
            "color": color,
            "row": row,
            "c0": c0,
            "c1": c1,
            "len": c1 - c0 + 1,
        }
    assert bars, "expected at least one floating bar below the schedule"
    return bg, schedule, bars


def _settle_bars(
    height: int,
    schedule: list[int],
    bars: dict[int, dict],
    *,
    order: list[int] | None = None,
    block_pending: bool = False,
    include_unmatched_as_obstacles: bool = False,
) -> list[dict]:
    current_order = list(schedule if order is None else order)
    matched = [color for color in current_order if color in bars]
    unmatched = [color for color in bars if color not in schedule]

    settled: list[dict] = []
    for idx, color in enumerate(matched):
        bar = bars[color]
        obstacles = list(settled)
        if include_unmatched_as_obstacles:
            obstacles.extend(bars[c] for c in unmatched)
        if block_pending:
            obstacles.extend(bars[c] for c in matched[idx + 1 :])

        stop_rows = [
            int(other["row"])
            for other in obstacles
            if int(other["row"]) > int(bar["row"])
            and _intervals_overlap(int(bar["c0"]), int(bar["c1"]), int(other["c0"]), int(other["c1"]))
        ]
        dest_row = min(stop_rows) - 1 if stop_rows else height - 1
        settled.append(
            {
                "color": color,
                "row": int(dest_row),
                "c0": int(bar["c0"]),
                "c1": int(bar["c1"]),
                "len": int(bar["len"]),
            }
        )
    return settled


def _render_solution(
    grid: np.ndarray,
    bg: int,
    schedule: list[int],
    bars: dict[int, dict],
    *,
    order: list[int] | None = None,
    block_pending: bool = False,
    include_unmatched_as_obstacles: bool = False,
    copy_unmatched: bool = False,
) -> np.ndarray:
    out = np.full_like(grid, bg)
    out[0, :] = grid[0, :]
    if copy_unmatched:
        for color, bar in bars.items():
            if color in schedule:
                continue
            out[int(bar["row"]), int(bar["c0"]) : int(bar["c1"]) + 1] = int(color)
    for bar in _settle_bars(
        grid.shape[0],
        schedule,
        bars,
        order=order,
        block_pending=block_pending,
        include_unmatched_as_obstacles=include_unmatched_as_obstacles,
    ):
        out[int(bar["row"]), int(bar["c0"]) : int(bar["c1"]) + 1] = int(bar["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the scheduled horizontal bars in top-row order, deleting unscheduled bars and letting early bars pass through later ones.

    Input structure:
    - The dominant background fills the grid.
    - The top row contains colored single pixels; their left-to-right order is the schedule.
    - Below the top row are several non-touching monochrome one-cell-thick horizontal bars with distinct colors.
    - Some floating bars use colors that never appear in the schedule row.

    Rule:
    - Delete every floating bar whose color is absent from the top-row schedule.
    - Process the remaining bars in schedule order from left to right in the top row.
    - When a bar falls, ignore all bars that have not been processed yet; only already-settled bars below its start row and the bottom edge can stop it.
    - If a falling bar overlaps a settled bar in columns, it stops one row above the nearest such settled bar; otherwise it lands on the bottom row. Keep the schedule row unchanged.

    Key insight:
    - The top row is a timetable rather than decoration: it overrides geometric cues such as height, so an early bar can pass through later floating bars and only those later bars stack on top of it.
    """
    g = np.asarray(grid, dtype=int)
    bg, schedule, bars = _parse_scene(g)
    return _render_solution(g, bg, schedule, bars)


_GEN_ATTEMPTS = 600


def _make_group_intervals(z0: int, z1: int, colors: list[int]) -> tuple[dict[int, tuple[int, int]], tuple[int, int]] | None:
    width = z1 - z0 + 1
    core_len = 2 if width >= 6 and random.random() < 0.45 else 1
    if width - core_len + 1 <= 0:
        return None

    core_left_lo = z0 + 1 if width >= 4 else z0
    core_left_hi = z1 - core_len if width >= 4 else z1 - core_len + 1
    if core_left_lo > core_left_hi:
        core_left_lo = z0
        core_left_hi = z1 - core_len + 1
    if core_left_lo > core_left_hi:
        return None

    core_start = random.randint(core_left_lo, core_left_hi)
    core_end = core_start + core_len - 1

    possible_lengths = []
    for length in range(2, min(6, width) + 1):
        lo = max(z0, core_end - length + 1)
        hi = min(core_start, z1 - length + 1)
        if lo <= hi:
            possible_lengths.append(length)
    if len(possible_lengths) < len(colors):
        return None

    chosen_lengths = random.sample(possible_lengths, len(colors))
    random.shuffle(chosen_lengths)
    intervals: dict[int, tuple[int, int]] = {}
    for color, length in zip(colors, chosen_lengths):
        lo = max(z0, core_end - length + 1)
        hi = min(core_start, z1 - length + 1)
        c0 = random.randint(lo, hi)
        intervals[int(color)] = (int(c0), int(c0 + length - 1))
    return intervals, (core_start, core_end)


def _zone_layout(independent_count: int) -> tuple[int, dict[str, tuple[int, int]]] | None:
    group_width = random.randint(5, 8)
    labels = ["group"] + [f"ind{i}" for i in range(independent_count)]
    widths = {"group": group_width}
    for i in range(independent_count):
        widths[f"ind{i}"] = random.randint(3, 5)
    random.shuffle(labels)

    gaps = [random.randint(1, 2) for _ in range(len(labels) + 1)]
    width = sum(widths[label] for label in labels) + sum(gaps)
    if width > 24:
        return None

    extra = random.randint(0, min(4, 24 - width))
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    ranges: dict[str, tuple[int, int]] = {}
    col = gaps[0]
    for idx, label in enumerate(labels):
        z0 = col
        z1 = col + widths[label] - 1
        ranges[label] = (z0, z1)
        col = z1 + 1 + gaps[idx + 1]
    return col, ranges


def _order_signature(order: list[int]) -> tuple[int, ...]:
    return tuple(map(int, order))


def _validate_generated_case(grid: np.ndarray) -> bool:
    try:
        bg, schedule, bars = _parse_scene(grid)
    except AssertionError:
        return False

    matched = [color for color in schedule if color in bars]
    unmatched = [color for color in bars if color not in schedule]
    if len(matched) < 3 or not unmatched:
        return False

    actual = _render_solution(grid, bg, schedule, bars)
    if not np.array_equal(actual, main(grid)):
        return False
    if not np.array_equal(actual[0], grid[0]):
        return False
    if any(np.any(actual[1:] == color) for color in unmatched):
        return False

    settled = _settle_bars(grid.shape[0], schedule, bars)
    if not settled or not any(int(bar["row"]) < grid.shape[0] - 1 for bar in settled):
        return False

    inversion = False
    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            a = bars[matched[i]]
            b = bars[matched[j]]
            if _intervals_overlap(int(a["c0"]), int(a["c1"]), int(b["c0"]), int(b["c1"])) and int(a["row"]) < int(b["row"]):
                inversion = True
                break
        if inversion:
            break
    if not inversion:
        return False

    if np.array_equal(actual, _render_solution(grid, bg, schedule, bars, block_pending=True)):
        return False
    if np.array_equal(actual, _render_solution(grid, bg, schedule, bars, include_unmatched_as_obstacles=True)):
        return False

    heuristic_orders = [
        sorted(matched, key=lambda color: (bars[color]["row"], bars[color]["c0"])),
        sorted(matched, key=lambda color: (-bars[color]["row"], bars[color]["c0"])),
        sorted(matched, key=lambda color: (bars[color]["len"], bars[color]["row"], bars[color]["c0"])),
        sorted(matched, key=lambda color: (-bars[color]["len"], bars[color]["row"], bars[color]["c0"])),
        sorted(matched, key=lambda color: (bars[color]["c0"] + bars[color]["c1"], bars[color]["row"])),
    ]
    seen_orders = {_order_signature(schedule)}
    for order in heuristic_orders:
        sig = _order_signature(order)
        if sig in seen_orders:
            continue
        seen_orders.add(sig)
        if np.array_equal(actual, _render_solution(grid, bg, schedule, bars, order=order)):
            return False

    colors_below = set(map(int, np.unique(grid[1:]))) - {bg}
    if len(colors_below) != len(bars):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    scheduled_count = random.randint(3, 5)
    stack_size = 2 if scheduled_count == 3 else random.randint(2, min(3, scheduled_count - 1))
    independent_count = scheduled_count - stack_size
    extra_count = random.randint(1, 2)

    schedule_colors = random.sample(Color.NOT_BLACK, scheduled_count)
    extra_colors = random.sample([c for c in Color.NOT_BLACK if c not in schedule_colors], extra_count)
    schedule = schedule_colors[:]
    random.shuffle(schedule)

    group_colors = random.sample(schedule_colors, stack_size)
    group_order = [color for color in schedule if color in group_colors]
    anchor_color = group_order[0]

    layout = _zone_layout(independent_count)
    if layout is None:
        return None
    width, zones = layout

    group_result = _make_group_intervals(*zones["group"], group_colors)
    if group_result is None:
        return None
    intervals, _core = group_result

    independent_zone_labels = [label for label in zones if label.startswith("ind")]
    random.shuffle(independent_zone_labels)
    independent_colors = [color for color in schedule_colors if color not in group_colors]
    random.shuffle(independent_colors)
    for color, label in zip(independent_colors, independent_zone_labels):
        z0, z1 = zones[label]
        zone_width = z1 - z0 + 1
        length = random.randint(2, min(5, zone_width))
        c0 = random.randint(z0, z1 - length + 1)
        intervals[int(color)] = (int(c0), int(c0 + length - 1))

    group_z0, group_z1 = zones["group"]
    for idx, color in enumerate(extra_colors):
        zone_width = group_z1 - group_z0 + 1
        length = random.randint(2, min(6, zone_width))
        c0 = random.randint(group_z0, group_z1 - length + 1)
        c1 = c0 + length - 1
        anchor = intervals[anchor_color]
        if idx == 0 and not _intervals_overlap(c0, c1, anchor[0], anchor[1]):
            c0 = max(group_z0, min(anchor[1], group_z1 - length + 1))
            c1 = c0 + length - 1
        if not _intervals_overlap(c0, c1, anchor[0], anchor[1]):
            return None
        intervals[int(color)] = (int(c0), int(c1))

    total_bars = scheduled_count + extra_count
    min_height = max(12, 2 * total_bars + 4)
    if min_height > 22:
        return None
    height = random.randint(min_height, 22)
    available_rows = list(range(2, height - 1, 2))
    if len(available_rows) < total_bars:
        return None

    group_rows = sorted(random.sample(available_rows, stack_size))
    row_map: dict[int, int] = {}
    for _ in range(30):
        perm = group_rows[:]
        random.shuffle(perm)
        candidate = {color: row for color, row in zip(group_order, perm)}
        if any(candidate[anchor_color] < candidate[color] for color in group_order[1:]):
            row_map.update(candidate)
            break
    else:
        return None

    remaining_rows = [row for row in available_rows if row not in group_rows]
    lower_than_anchor = [row for row in remaining_rows if row > row_map[anchor_color]]
    if not lower_than_anchor:
        return None
    extra_anchor_row = random.choice(lower_than_anchor)
    row_map[extra_colors[0]] = int(extra_anchor_row)
    remaining_rows.remove(extra_anchor_row)

    leftovers = [color for color in schedule_colors + extra_colors[1:] if color not in row_map]
    random.shuffle(remaining_rows)
    for color in leftovers:
        if not remaining_rows:
            return None
        row_map[int(color)] = int(remaining_rows.pop())

    grid = np.zeros((height, width), dtype=int)
    schedule_cols = sorted(random.sample(range(width), scheduled_count))
    for col, color in zip(schedule_cols, schedule):
        grid[0, int(col)] = int(color)

    for color in schedule_colors + extra_colors:
        r = row_map[int(color)]
        c0, c1 = intervals[int(color)]
        grid[r, c0 : c1 + 1] = int(color)

    return grid if _validate_generated_case(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid schedule-fall ARC input")
