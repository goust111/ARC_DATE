from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 7
RECT_IN = 1
RECT_OUT = 2
BAR = 5
MARKER = 9

SIDES = ("top", "bottom", "left", "right")
GRID_MIN = 14
GRID_MAX = 20
RECT_H_CHOICES = (3, 4)
RECT_W_CHOICES = (5, 6)
MARKER_COUNT_CHOICES = (4, 5)
GEN_RETRIES = 600


def _most_common_color(grid: np.ndarray) -> int:
    return Counter(map(int, grid.flat)).most_common(1)[0][0]


def _find_bar_row(grid: np.ndarray, bg: int) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == BAR)]
    assert len(rows) == 1, "expected exactly one full-width trigger bar"
    bar_r = rows[0]
    assert bar_r > 0
    return bar_r


def _find_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == RECT_IN)
    assert coords.size > 0, "missing blue rectangle"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == RECT_IN), "rectangle must be solid"
    assert int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)
    return r0, c0, r1, c1


def _marker_records(
    grid: np.ndarray, rect: tuple[int, int, int, int]
) -> list[dict[str, int | str | tuple[int, int]]]:
    r0, c0, r1, c1 = rect
    records: list[dict[str, int | str | tuple[int, int]]] = []
    for r_raw, c_raw in np.argwhere(grid == MARKER):
        r = int(r_raw)
        c = int(c_raw)
        side: str | None = None
        dist: int | None = None
        lane: int | None = None
        if c0 <= c <= c1 and r < r0:
            side = "top"
            dist = r0 - r
            lane = c
        elif c0 <= c <= c1 and r > r1:
            side = "bottom"
            dist = r - r1
            lane = c
        elif r0 <= r <= r1 and c < c0:
            side = "left"
            dist = c0 - c
            lane = r
        elif r0 <= r <= r1 and c > c1:
            side = "right"
            dist = c - c1
            lane = r
        assert side is not None and dist is not None and lane is not None, "marker must align to one rectangle side"
        records.append({"src": (r, c), "side": side, "dist": dist, "lane": lane})
    assert len(records) >= 2
    return records


def _destination_for(
    rect: tuple[int, int, int, int], side: str, lane: int, enters: bool
) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    if side == "top":
        return (r0 + 1 if enters else r0 - 1, lane)
    if side == "bottom":
        return (r1 - 1 if enters else r1 + 1, lane)
    if side == "left":
        return (lane, c0 + 1 if enters else c0 - 1)
    assert side == "right"
    return (lane, c1 - 1 if enters else c1 + 1)


def _paint_splash(out: np.ndarray, start: tuple[int, int], bg: int) -> None:
    r, c = start
    for dc in (-1, 1):
        rr = r - 1
        cc = c + dc
        while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            if int(out[rr, cc]) == bg:
                out[rr, cc] = MARKER
            rr -= 1
            cc += dc


def _render_output_from_parsed(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    records: list[dict[str, int | str | tuple[int, int]]],
    bar_r: int,
) -> np.ndarray:
    bg = _most_common_color(grid)
    out = np.full_like(grid, bg)
    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = RECT_OUT
    out[bar_r, :] = BAR

    dists = [int(rec["dist"]) for rec in records]
    nearest_dist = min(dists)
    assert dists.count(nearest_dist) == 1, "nearest marker must be unique"
    nearest_src = next(rec["src"] for rec in records if int(rec["dist"]) == nearest_dist)

    destinations: list[tuple[int, int]] = []
    for rec in records:
        src = rec["src"]
        side = str(rec["side"])
        lane = int(rec["lane"])
        enters = src == nearest_src
        dst = _destination_for(rect, side, lane, enters)
        destinations.append(dst)

    assert len(destinations) == len(set(destinations)), "marker destinations must be unique"
    for dst in destinations:
        out[dst] = MARKER
    for dst in destinations:
        if dst[0] == bar_r - 1:
            _paint_splash(out, dst, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide aligned maroon markers toward the blue rectangle, then splash upward diagonals only from markers that stop directly above the trigger bar.

    Input structure
    - An orange background contains one solid blue rectangle, several isolated maroon marker pixels, and one full-width gray horizontal trigger bar below the rectangle.
    - Every maroon marker lies outside the rectangle but is aligned to it by sharing one rectangle column span or one rectangle row span.
    - Exactly one marker is strictly nearest to the rectangle by Manhattan distance to its aligned side.

    Rule
    - Recolor the blue rectangle red and erase the original maroon marker locations.
    - Move the unique globally nearest marker straight inward across the rectangle boundary so it lands one cell inside the rectangle along its aligned direction.
    - Move every other marker straight toward the rectangle so it lands in the exterior cell immediately adjacent to its aligned side.
    - For each moved marker whose final row is exactly one row above the gray trigger bar, emit maroon rays along both upward diagonals until they leave the grid, painting only orange background cells and never overwriting existing red, gray, or maroon cells.

    Key insight
    - The diagonal splash is not attached to a specific source marker type; it is triggered only by the stopping position created by the earlier slide step, so the nearest-marker rule and the adjacency-to-bar test must be composed in order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    rect = _find_rectangle(g)
    bar_r = _find_bar_row(g, bg)
    records = _marker_records(g, rect)
    return _render_output_from_parsed(g, rect, records, bar_r)


def _choose_side_counts(rect_h: int, rect_w: int, marker_count: int) -> dict[str, int] | None:
    bottom_count = random.choice((1, 1, 2))
    if bottom_count >= marker_count:
        return None

    other_side_count = random.randint(1, min(3, marker_count - bottom_count))
    other_sides = random.sample(["top", "left", "right"], k=other_side_count)
    counts = {side: 0 for side in SIDES}
    counts["bottom"] = bottom_count
    for side in other_sides:
        counts[side] = 1

    remaining = marker_count - sum(counts.values())
    capacities = {"top": rect_w, "bottom": rect_w, "left": rect_h, "right": rect_h}
    growable = [side for side in SIDES if counts[side] > 0 and counts[side] < capacities[side]]
    for _ in range(remaining):
        valid = [side for side in growable if counts[side] < capacities[side]]
        if not valid:
            return None
        side = random.choice(valid)
        counts[side] += 1
    if counts["bottom"] == 0 or sum(counts[side] for side in ("top", "left", "right")) == 0:
        return None
    return counts


def _sample_lanes(values: list[int], count: int, require_gap: bool) -> list[int] | None:
    if count == 0:
        return []
    if not require_gap:
        return random.sample(values, count)

    choices: list[tuple[int, ...]] = []

    def rec(start: int, chosen: list[int]) -> None:
        if len(chosen) == count:
            choices.append(tuple(chosen))
            return
        for idx in range(start, len(values)):
            v = values[idx]
            if chosen and abs(v - chosen[-1]) == 1:
                continue
            rec(idx + 1, chosen + [v])

    rec(0, [])
    if not choices:
        return None
    return list(random.choice(choices))


def _assign_distances(
    slots: list[dict[str, int | str]],
    nearest_index: int,
    margins: dict[str, int],
) -> bool:
    near_slot = slots[nearest_index]
    near_side = str(near_slot["side"])
    near_options = [d for d in range(2, margins[near_side] + 1)]
    if not near_options:
        return False
    random.shuffle(near_options)

    others = [idx for idx in range(len(slots)) if idx != nearest_index]
    others.sort(key=lambda idx: margins[str(slots[idx]["side"])])

    for nearest_dist in near_options:
        used = {nearest_dist}
        slots[nearest_index]["dist"] = nearest_dist

        def rec(pos: int) -> bool:
            if pos == len(others):
                return True
            idx = others[pos]
            side = str(slots[idx]["side"])
            min_dist = 3 if side == "bottom" else 2
            options = [
                d
                for d in range(min_dist, margins[side] + 1)
                if d > nearest_dist and d not in used
            ]
            random.shuffle(options)
            for dist in options:
                slots[idx]["dist"] = dist
                used.add(dist)
                if rec(pos + 1):
                    return True
                used.remove(dist)
            slots[idx]["dist"] = 0
            return False

        if rec(0):
            return True
        slots[nearest_index]["dist"] = 0
    return False


def _slot_source(rect: tuple[int, int, int, int], side: str, lane: int, dist: int) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    if side == "top":
        return (r0 - dist, lane)
    if side == "bottom":
        return (r1 + dist, lane)
    if side == "left":
        return (lane, c0 - dist)
    assert side == "right"
    return (lane, c1 + dist)


def _build_slots(
    rect: tuple[int, int, int, int], counts: dict[str, int]
) -> tuple[list[dict[str, int | str]], int] | None:
    r0, c0, r1, c1 = rect
    slots: list[dict[str, int | str]] = []
    lane_values = {
        "top": list(range(c0, c1 + 1)),
        "bottom": list(range(c0, c1 + 1)),
        "left": list(range(r0, r1 + 1)),
        "right": list(range(r0, r1 + 1)),
    }
    for side in SIDES:
        require_gap = side == "bottom" and counts[side] > 1
        lanes = _sample_lanes(lane_values[side], counts[side], require_gap=require_gap)
        if lanes is None:
            return None
        for lane in lanes:
            slots.append({"side": side, "lane": lane, "dist": 0})

    candidate_indices = [i for i, slot in enumerate(slots) if str(slot["side"]) in {"top", "left", "right"}]
    if not candidate_indices:
        return None
    nearest_index = random.choice(candidate_indices)
    return slots, nearest_index


def _validate_structure(
    grid: np.ndarray,
    latent_rect: tuple[int, int, int, int],
    latent_bar_r: int,
    latent_slots: list[dict[str, int | str]],
    nearest_index: int,
) -> None:
    bg = _most_common_color(grid)
    assert bg == BG
    rect = _find_rectangle(grid)
    assert rect == latent_rect
    bar_r = _find_bar_row(grid, bg)
    assert bar_r == latent_bar_r
    records = _marker_records(grid, rect)
    assert len(records) == len(latent_slots)
    parsed = {
        (str(rec["side"]), int(rec["lane"])): int(rec["dist"])
        for rec in records
    }
    expected = {
        (str(slot["side"]), int(slot["lane"])): int(slot["dist"])
        for slot in latent_slots
    }
    assert parsed == expected
    dists = list(parsed.values())
    assert len(dists) == len(set(dists))
    nearest_dist = int(latent_slots[nearest_index]["dist"])
    assert nearest_dist == min(dists) and dists.count(nearest_dist) == 1

    non_nearest = [slot for i, slot in enumerate(latent_slots) if i != nearest_index]
    trigger_count = 0
    non_trigger_count = 0
    for slot in non_nearest:
        dst = _destination_for(rect, str(slot["side"]), int(slot["lane"]), enters=False)
        if dst[0] == bar_r - 1:
            trigger_count += 1
        else:
            non_trigger_count += 1
    assert trigger_count >= 1
    assert non_trigger_count >= 1

    srcs = [tuple(_slot_source(rect, str(slot["side"]), int(slot["lane"]), int(slot["dist"]))) for slot in latent_slots]
    assert len(srcs) == len(set(srcs))
    for i, (r1_src, c1_src) in enumerate(srcs):
        assert 0 <= r1_src < grid.shape[0] and 0 <= c1_src < grid.shape[1]
        assert grid[r1_src, c1_src] == MARKER
        for j, (r2_src, c2_src) in enumerate(srcs):
            if i >= j:
                continue
            assert abs(r1_src - r2_src) + abs(c1_src - c2_src) > 1

    out = main(grid)
    expected_out = _render_output_from_parsed(grid, rect, records, bar_r)
    assert np.array_equal(out, expected_out)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    rect_h = random.choice(RECT_H_CHOICES)
    rect_w = random.choice(RECT_W_CHOICES)

    min_top = 2
    min_bottom = 5
    min_side = 2
    if height - rect_h - min_top - min_bottom < 0 or width - rect_w - 2 * min_side < 0:
        return None

    r0 = random.randint(min_top, height - rect_h - min_bottom)
    c0 = random.randint(min_side, width - rect_w - min_side)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    rect = (r0, c0, r1, c1)
    bar_r = r1 + 2
    if bar_r >= height:
        return None

    marker_count = random.choice(MARKER_COUNT_CHOICES)
    counts = _choose_side_counts(rect_h, rect_w, marker_count)
    if counts is None:
        return None

    built = _build_slots(rect, counts)
    if built is None:
        return None
    slots, nearest_index = built

    margins = {
        "top": r0,
        "bottom": height - 1 - r1,
        "left": c0,
        "right": width - 1 - c1,
    }
    if not _assign_distances(slots, nearest_index, margins):
        return None

    nearest_side = str(slots[nearest_index]["side"])
    if nearest_side == "bottom":
        return None

    srcs = []
    destinations = []
    for idx, slot in enumerate(slots):
        side = str(slot["side"])
        lane = int(slot["lane"])
        dist = int(slot["dist"])
        src = _slot_source(rect, side, lane, dist)
        if not (0 <= src[0] < height and 0 <= src[1] < width):
            return None
        if src[0] == bar_r:
            return None
        dst = _destination_for(rect, side, lane, enters=(idx == nearest_index))
        srcs.append(src)
        destinations.append(dst)

    if len(srcs) != len(set(srcs)) or len(destinations) != len(set(destinations)):
        return None
    if any(abs(r1a - r1b) + abs(c1a - c1b) <= 1 for i, (r1a, c1a) in enumerate(srcs) for r1b, c1b in srcs[i + 1 :]):
        return None

    non_nearest = [slot for i, slot in enumerate(slots) if i != nearest_index]
    if not any(
        _destination_for(rect, str(slot["side"]), int(slot["lane"]), enters=False)[0] == bar_r - 1
        for slot in non_nearest
    ):
        return None
    if not any(
        _destination_for(rect, str(slot["side"]), int(slot["lane"]), enters=False)[0] != bar_r - 1
        for slot in non_nearest
    ):
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = RECT_IN
    grid[bar_r, :] = BAR
    for src in srcs:
        grid[src] = MARKER

    try:
        _validate_structure(grid, rect, bar_r, slots, nearest_index)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
