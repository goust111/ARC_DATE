from __future__ import annotations

from collections import Counter

import numpy as np


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

ARM_COLORS = [BLUE, GREEN, GREY, PINK, ORANGE, TEAL, MAROON]
ORIENTATIONS = ("down", "up", "right", "left")
MAX_GEN_ATTEMPTS = 400

_RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _border_direction(shape: tuple[int, int], start: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    r, c = map(int, start)
    if r == 0 and 0 < c < w - 1:
        return 1, 0
    if r == h - 1 and 0 < c < w - 1:
        return -1, 0
    if c == 0 and 0 < r < h - 1:
        return 0, 1
    if c == w - 1 and 0 < r < h - 1:
        return 0, -1
    raise AssertionError("the red start must be a non-corner border cell")


def _cross_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, center)
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)
    assert bg == BLACK

    starts = np.argwhere(g == RED)
    assert starts.shape == (1, 2)
    start = tuple(map(int, starts[0]))
    direction = _border_direction(g.shape, start)

    portals: list[dict] = []
    cell_to_portal: dict[tuple[int, int], int] = {}
    centers = np.argwhere(g == YELLOW)
    assert len(centers) >= 2

    for rr, cc in centers:
        r = int(rr)
        c = int(cc)
        assert 0 < r < h - 1 and 0 < c < w - 1
        arm = int(g[r - 1, c])
        assert arm == int(g[r + 1, c]) == int(g[r, c - 1]) == int(g[r, c + 1])
        assert arm not in (BLACK, RED, YELLOW)
        cells = _cross_cells((r, c))
        idx = len(portals)
        portals.append({"center": (r, c), "color": arm, "cells": cells})
        for cell in cells:
            assert cell not in cell_to_portal
            cell_to_portal[cell] = idx

    for rr, cc in np.argwhere(g != BLACK):
        cell = (int(rr), int(cc))
        val = int(g[cell])
        if cell == start:
            assert val == RED
            assert cell not in cell_to_portal
            continue
        idx = cell_to_portal.get(cell)
        assert idx is not None
        portal = portals[idx]
        expected = YELLOW if cell == portal["center"] else int(portal["color"])
        assert val == expected

    color_to_portals: dict[int, list[int]] = {}
    for idx, portal in enumerate(portals):
        color_to_portals.setdefault(int(portal["color"]), []).append(idx)
    assert any(len(ids) == 1 for ids in color_to_portals.values())
    assert all(len(ids) in (1, 2) for ids in color_to_portals.values())

    return {
        "grid": g,
        "start": start,
        "direction": direction,
        "portals": portals,
        "cell_to_portal": cell_to_portal,
        "color_to_portals": color_to_portals,
    }


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    scene = _parse_scene(grid)
    g = scene["grid"]
    dr, dc = scene["direction"]
    cur_r, cur_c = scene["start"]

    out = g.copy()
    seen: set[tuple[int, int]] = set()
    hit_portals: list[int] = []
    exit_portals: list[int] = []
    hit_colors: list[int] = []
    painted: list[tuple[int, int]] = []
    lanes = [cur_c if dr else cur_r]
    terminal_color = None
    end_mode = None

    max_steps = g.shape[0] * g.shape[1] * 4
    for _ in range(max_steps):
        state = (int(cur_r), int(cur_c))
        assert state not in seen, "the ray entered a loop"
        seen.add(state)

        nr = int(cur_r + dr)
        nc = int(cur_c + dc)
        if not in_bounds(g.shape, nr, nc):
            end_mode = "boundary"
            break

        if int(g[nr, nc]) == BLACK:
            out[nr, nc] = RED
            painted.append((nr, nc))
            cur_r, cur_c = nr, nc
            continue

        portal_idx = scene["cell_to_portal"].get((nr, nc))
        assert portal_idx is not None, "the ray touched an invalid foreground pattern"
        portal = scene["portals"][portal_idx]
        color = int(portal["color"])
        hits = scene["color_to_portals"][color]
        hit_portals.append(portal_idx)
        hit_colors.append(color)

        if len(hits) == 1:
            terminal_color = color
            end_mode = "unique"
            break

        assert len(hits) == 2
        other_idx = hits[1] if hits[0] == portal_idx else hits[0]
        exit_portals.append(other_idx)
        exit_center = scene["portals"][other_idx]["center"]
        cur_r = int(exit_center[0] + dr)
        cur_c = int(exit_center[1] + dc)
        assert in_bounds(g.shape, cur_r, cur_c)
        assert scene["cell_to_portal"].get((cur_r, cur_c)) == other_idx
        lanes.append(cur_c if dr else cur_r)
    else:
        raise AssertionError("the ray did not terminate")

    assert end_mode in {"boundary", "unique"}
    return out, {
        "teleports": len(exit_portals),
        "end_mode": end_mode,
        "terminal_color": terminal_color,
        "paint_count": len(painted),
        "lanes": lanes,
        "hit_colors": hit_colors,
        "hit_portals": hit_portals,
        "exit_portals": exit_portals,
        "used_portal_ids": set(hit_portals) | set(exit_portals),
        "portals": scene["portals"],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the red ray inward, teleporting between pluses that share an arm color until it exits or reaches a unique-color plus.

    Input structure:
    - Black background with exactly one red start pixel on a non-corner border cell.
    - Several plus markers whose centers are yellow and whose four arms all share one non-background color.
    - Most arm colors appear on exactly two pluses, while at least one arm color appears on exactly one plus.

    Rule:
    - Launch a ray from the red start in the inward direction perpendicular to that border and paint every visited background cell red.
    - When the next step would enter any cell of a plus, inspect that plus's arm color instead of painting the plus.
    - If that arm color belongs to two pluses, teleport to the other plus and continue in the same direction from the exit arm cell just beyond its center.
    - If that arm color is unique, stop immediately; if no plus is met before the edge, stop when the ray leaves the grid.
    - Keep the red start pixel and every plus marker unchanged.

    Key insight:
    - The pluses are portals keyed by their arm color, so the visible red trace is a chain of straight segments connected by teleport jumps; only a color with no partner is a real terminal.
    """
    return _solve(np.asarray(grid, dtype=int))[0]


def _straight_stop_output(grid: np.ndarray) -> np.ndarray:
    scene = _parse_scene(grid)
    g = scene["grid"]
    dr, dc = scene["direction"]
    cur_r, cur_c = scene["start"]
    out = g.copy()

    while True:
        nr = int(cur_r + dr)
        nc = int(cur_c + dc)
        if not in_bounds(g.shape, nr, nc):
            break
        if int(g[nr, nc]) != BLACK:
            break
        out[nr, nc] = RED
        cur_r, cur_c = nr, nc

    return out


def _coarse_lanes(span: int) -> list[int]:
    return list(range(2, span - 1, 3))


def _local_to_actual(a: int, b: int, orient: str, h: int, w: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    if orient == "down":
        return a, b
    if orient == "up":
        return h - 1 - a, b
    if orient == "right":
        return b, a
    if orient == "left":
        return b, w - 1 - a
    raise ValueError(f"unknown orientation: {orient}")


def _can_place_center(center: tuple[int, int], existing: list[tuple[int, int]], forward: int, lanes: int) -> bool:
    a, b = map(int, center)
    if not (1 <= a < forward - 1 and 1 <= b < lanes - 1):
        return False

    new_cells = set(_cross_cells((a, b)))
    for ea, eb in existing:
        old_cells = set(_cross_cells((int(ea), int(eb))))
        if new_cells & old_cells:
            return False
        if b == int(eb) and abs(a - int(ea)) < 4:
            return False
        if a == int(ea) and abs(b - int(eb)) < 4:
            return False
    return True


def _pick_extra_center(
    rng: np.random.Generator,
    forward: int,
    lane_count: int,
    lane: int,
    existing: list[tuple[int, int]],
) -> tuple[int, int] | None:
    rows = list(range(2, forward - 2))
    rng.shuffle(rows)
    for a in rows:
        center = (int(a), int(lane))
        if _can_place_center(center, existing, forward, lane_count):
            existing.append(center)
            return center
    return None


def _render_grid(
    orient: str,
    forward: int,
    lane_count: int,
    start_lane: int,
    portal_specs: list[dict],
) -> np.ndarray:
    if orient in {"down", "up"}:
        h, w = forward, lane_count
    else:
        h, w = lane_count, forward

    grid = np.full((h, w), BLACK, dtype=int)
    start = _local_to_actual(0, start_lane, orient, h, w)
    grid[start] = RED

    for spec in portal_specs:
        center = spec["center"]
        color = int(spec["color"])
        for cell in _cross_cells(center):
            rr, cc = _local_to_actual(cell[0], cell[1], orient, h, w)
            grid[rr, cc] = YELLOW if cell == center else color

    return grid


def _validate_generated(grid: np.ndarray, expected: dict) -> bool:
    out, info = _solve(grid)
    assert info["teleports"] == int(expected["pair_count"])
    assert info["end_mode"] == expected["end_mode"]
    assert info["hit_colors"][: expected["pair_count"]] == expected["pair_colors"]
    if expected["end_mode"] == "unique":
        assert info["terminal_color"] == int(expected["terminal_color"])
        assert info["hit_colors"][-1] == int(expected["terminal_color"])
    else:
        assert info["terminal_color"] is None

    expected_used = 2 * int(expected["pair_count"]) + (1 if expected["end_mode"] == "unique" else 0)
    assert len(info["used_portal_ids"]) == expected_used
    assert len(info["portals"]) > expected_used
    assert len(info["lanes"]) == int(expected["pair_count"]) + 1
    assert len(set(info["lanes"])) == len(info["lanes"])
    assert info["paint_count"] >= 4
    assert not np.array_equal(out, grid)
    assert not np.array_equal(out, _straight_stop_output(grid))
    return True


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    end_mode = "unique" if _RNG.random() < 0.65 else "boundary"
    if end_mode == "unique":
        pair_count = int(_RNG.integers(1, 3))
    else:
        pair_count = int(_RNG.integers(1, 4))

    candidate_lanes = [m for m in range(14, 21) if len(_coarse_lanes(m)) >= pair_count + 2]
    if not candidate_lanes:
        return None
    lane_count = int(_RNG.choice(candidate_lanes))
    lane_slots = _coarse_lanes(lane_count)
    path_lanes = list(map(int, _RNG.permutation(lane_slots)[: pair_count + 1]))

    cursor = 0
    entrances: list[int] = []
    exits: list[int] = []
    for _ in range(pair_count):
        entrance = int(cursor + _RNG.integers(3, 5))
        exit_a = int(entrance + _RNG.integers(3, 5))
        entrances.append(entrance)
        exits.append(exit_a)
        cursor = exit_a + 1

    terminal_a = None
    if end_mode == "unique":
        terminal_a = int(cursor + _RNG.integers(3, 5))
        min_forward = terminal_a + 2
    else:
        min_forward = cursor + 3
    if min_forward > 30:
        return None

    max_forward = min(30, min_forward + 4)
    forward = int(_RNG.integers(min_forward, max_forward + 1))
    orient = str(_RNG.choice(ORIENTATIONS))

    color_order = list(map(int, _RNG.permutation(ARM_COLORS)))
    color_pos = 0
    pair_colors = color_order[color_pos : color_pos + pair_count]
    color_pos += pair_count

    centers: list[tuple[int, int]] = []
    portal_specs: list[dict] = []
    for i, color in enumerate(pair_colors):
        entrance_center = (int(entrances[i]), int(path_lanes[i]))
        exit_center = (int(exits[i]), int(path_lanes[i + 1]))
        if not _can_place_center(entrance_center, centers, forward, lane_count):
            return None
        centers.append(entrance_center)
        portal_specs.append({"center": entrance_center, "color": int(color)})

        if not _can_place_center(exit_center, centers, forward, lane_count):
            return None
        centers.append(exit_center)
        portal_specs.append({"center": exit_center, "color": int(color)})

    terminal_color = None
    if end_mode == "unique":
        terminal_color = int(color_order[color_pos])
        color_pos += 1
        terminal_center = (int(terminal_a), int(path_lanes[-1]))
        if not _can_place_center(terminal_center, centers, forward, lane_count):
            return None
        centers.append(terminal_center)
        portal_specs.append({"center": terminal_center, "color": terminal_color})

    unused_slots = [lane for lane in lane_slots if lane not in path_lanes]
    if not unused_slots:
        return None

    if color_pos >= len(color_order):
        return None
    unique_distractor_color = int(color_order[color_pos])
    color_pos += 1
    unique_lane = int(_RNG.choice(unused_slots))
    unique_center = _pick_extra_center(_RNG, forward, lane_count, unique_lane, centers)
    if unique_center is None:
        return None
    portal_specs.append({"center": unique_center, "color": unique_distractor_color})

    if color_pos < len(color_order) and _RNG.random() < 0.55:
        pair_color = int(color_order[color_pos])
        lane_a = int(_RNG.choice(unused_slots))
        lane_b = int(_RNG.choice(unused_slots))
        extra_a = _pick_extra_center(_RNG, forward, lane_count, lane_a, centers)
        extra_b = _pick_extra_center(_RNG, forward, lane_count, lane_b, centers)
        if extra_a is not None and extra_b is not None:
            portal_specs.append({"center": extra_a, "color": pair_color})
            portal_specs.append({"center": extra_b, "color": pair_color})

    grid = _render_grid(orient, forward, lane_count, path_lanes[0], portal_specs)
    expected = {
        "pair_count": pair_count,
        "end_mode": end_mode,
        "pair_colors": list(map(int, pair_colors)),
        "terminal_color": terminal_color,
    }
    try:
        if _validate_generated(grid, expected):
            return grid, expected
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid plus-portal puzzle")
