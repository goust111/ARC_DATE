from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SWITCH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_SIZE_RANGE = (12, 18)
SWITCH_COUNT_RANGE = (2, 4)
LIGHT_COUNT_RANGE = (5, 12)
MAX_GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray, target: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != target:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == target:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _extract_switches(grid: np.ndarray) -> list[dict[str, int | tuple[int, int]]]:
    switches: list[dict[str, int | tuple[int, int]]] = []
    for color in sorted({int(x) for x in np.unique(grid) if int(x) not in (BLACK, GREY)}):
        for coords in _find_components(grid, color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            r0 = int(rows.min())
            r1 = int(rows.max())
            c0 = int(cols.min())
            c1 = int(cols.max())
            if coords.shape[0] != 4 or (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
                continue
            expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
            if set(map(tuple, coords)) != expected:
                continue
            switches.append({"color": color, "top": r0, "left": c0, "rows": (r0, r0 + 1), "cols": (c0, c0 + 1)})
    switches.sort(key=lambda sw: (int(sw["top"]), int(sw["left"]), int(sw["color"])))
    assert switches, "expected at least one 2x2 switch"
    return switches


def _extract_lights(grid: np.ndarray) -> list[tuple[int, int]]:
    lights: list[tuple[int, int]] = []
    for coords in _find_components(grid, GREY):
        assert coords.shape[0] == 1, "grey lights must be singleton pixels"
        lights.append(tuple(int(x) for x in coords[0]))
    return sorted(lights)


def _pick_light_color(r: int, c: int, switches: list[dict[str, int | tuple[int, int]]]) -> int:
    affecting = []
    for sw in switches:
        rows = sw["rows"]
        cols = sw["cols"]
        if r in rows or c in cols:
            affecting.append(sw)
    if not affecting:
        return GREY

    affecting.sort(key=lambda sw: (int(sw["top"]), int(sw["left"])))
    best = affecting[0]
    return int(best["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """Broadcast each 2x2 switch color to grey lights in the same switch rows or columns."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    switches = _extract_switches(g)
    lights = _extract_lights(g)

    out = g.copy()
    for r, c in lights:
        out[r, c] = _pick_light_color(r, c, switches)
    return out


def _sample_spaced_starts(limit: int, count: int, min_gap: int) -> list[int] | None:
    starts = list(range(limit))
    for _ in range(120):
        random.shuffle(starts)
        picked: list[int] = []
        for value in starts:
            if all(abs(value - prev) >= min_gap for prev in picked):
                picked.append(value)
                if len(picked) == count:
                    return sorted(picked)
    return None


def _candidate_switch_count(h: int, w: int) -> int:
    max_rows = max(1, (h - 1) // 3)
    max_cols = max(1, (w - 1) // 3)
    return min(SWITCH_COUNT_RANGE[1], max_rows, max_cols)


def _place_switches(h: int, w: int) -> list[dict[str, int | tuple[int, int]]] | None:
    max_count = _candidate_switch_count(h, w)
    if max_count < SWITCH_COUNT_RANGE[0]:
        return None
    count = random.randint(SWITCH_COUNT_RANGE[0], max_count)
    rows = _sample_spaced_starts(h - 1, count, 3)
    cols = _sample_spaced_starts(w - 1, count, 3)
    if rows is None or cols is None:
        return None

    colors = random.sample(SWITCH_COLORS, count)
    random.shuffle(cols)
    switches = []
    for top, left, color in zip(rows, cols, colors):
        switches.append({"color": color, "top": top, "left": left, "rows": (top, top + 1), "cols": (left, left + 1)})
    switches.sort(key=lambda sw: (int(sw["top"]), int(sw["left"])))
    return switches


def _occupied_mask(h: int, w: int, switches: list[dict[str, int | tuple[int, int]]]) -> np.ndarray:
    occupied = np.zeros((h, w), dtype=bool)
    for sw in switches:
        top = int(sw["top"])
        left = int(sw["left"])
        occupied[top : top + 2, left : left + 2] = True
    return occupied


def _add_light(
    pos: tuple[int, int],
    occupied: np.ndarray,
    lights: list[tuple[int, int]],
) -> bool:
    r, c = pos
    h, w = occupied.shape
    if not (0 <= r < h and 0 <= c < w) or occupied[r, c]:
        return False
    for rr, cc in lights:
        if abs(rr - r) + abs(cc - c) == 1:
            return False
    occupied[r, c] = True
    lights.append(pos)
    return True


def _rows_and_cols(switches: list[dict[str, int | tuple[int, int]]]) -> tuple[set[int], set[int]]:
    rows = set()
    cols = set()
    for sw in switches:
        rows.update(int(x) for x in sw["rows"])
        cols.update(int(x) for x in sw["cols"])
    return rows, cols


def _categorize_positions(
    h: int,
    w: int,
    switches: list[dict[str, int | tuple[int, int]]],
    occupied: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    neutral_rows, neutral_cols = _rows_and_cols(switches)
    unaffected = []
    single = []
    conflict = []
    any_free = []
    for r in range(h):
        for c in range(w):
            if occupied[r, c]:
                continue
            any_free.append((r, c))
            affecting = []
            for sw in switches:
                if r in sw["rows"] or c in sw["cols"]:
                    affecting.append(sw)
            if len(affecting) == 0:
                unaffected.append((r, c))
            elif len(affecting) == 1:
                single.append((r, c))
            else:
                tops = [int(sw["top"]) for sw in affecting]
                if tops.count(min(tops)) == 1:
                    conflict.append((r, c))
    return unaffected, single, conflict, any_free


def _render_input(
    h: int,
    w: int,
    switches: list[dict[str, int | tuple[int, int]]],
    lights: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for sw in switches:
        top = int(sw["top"])
        left = int(sw["left"])
        grid[top : top + 2, left : left + 2] = int(sw["color"])
    for r, c in lights:
        grid[r, c] = GREY
    return grid


def _oracle_output(
    h: int,
    w: int,
    switches: list[dict[str, int | tuple[int, int]]],
    lights: list[tuple[int, int]],
) -> np.ndarray:
    out = _render_input(h, w, switches, lights)
    for r, c in lights:
        out[r, c] = _pick_light_color(r, c, switches)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    switches = _place_switches(h, w)
    if switches is None:
        return None

    occupied = _occupied_mask(h, w, switches)
    lights: list[tuple[int, int]] = []
    unaffected, single, conflict, any_free = _categorize_positions(h, w, switches, occupied)
    if not unaffected or not single or not conflict:
        return None

    must_have = [
        random.choice(unaffected),
        random.choice(single),
        random.choice(conflict),
    ]
    if len(single) >= 2:
        must_have.append(random.choice(single))

    for pos in must_have:
        if not _add_light(pos, occupied, lights):
            return None

    target_light_count = random.randint(max(len(lights), LIGHT_COUNT_RANGE[0]), LIGHT_COUNT_RANGE[1])
    while len(lights) < target_light_count:
        unaffected, single, conflict, any_free = _categorize_positions(h, w, switches, occupied)
        buckets = []
        if unaffected:
            buckets.extend([unaffected] * 2)
        if single:
            buckets.extend([single] * 2)
        if conflict:
            buckets.extend([conflict] * 2)
        if any_free:
            buckets.append(any_free)
        if not buckets:
            break
        pos = random.choice(random.choice(buckets))
        if not _add_light(pos, occupied, lights):
            continue

    if len(lights) < LIGHT_COUNT_RANGE[0]:
        return None

    inp = _render_input(h, w, switches, lights)
    expected = _oracle_output(h, w, switches, lights)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if not np.any(expected == GREY):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
