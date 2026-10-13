from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
YELLOW = 4
MAGENTA = 6


def _is_diagonal(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a != b and abs(a[0] - b[0]) == abs(a[1] - b[1])


def _diag_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    assert _is_diagonal(a, b), "blue endpoints must be 45-degree diagonal-aligned"
    steps = abs(r1 - r0)
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _ring_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    out = []
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if rr == r and cc == c:
                continue
            out.append((rr, cc))
    return out


def _find_unique_blue_pairing(blue_points: Iterable[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pts = tuple(sorted(tuple(map(int, p)) for p in blue_points))
    assert pts and len(pts) % 2 == 0, "expected an even non-zero number of blue endpoints"

    solutions: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []

    def rec(remaining: tuple[tuple[int, int], ...], cur: list[tuple[tuple[int, int], tuple[int, int]]]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(cur.copy())
            return
        first = remaining[0]
        for i in range(1, len(remaining)):
            second = remaining[i]
            if not _is_diagonal(first, second):
                continue
            pair = (first, second) if first <= second else (second, first)
            nxt = remaining[1:i] + remaining[i + 1 :]
            rec(nxt, cur + [pair])

    rec(pts, [])
    assert len(solutions) == 1, "blue endpoints must admit exactly one diagonal pairing"
    return sorted(solutions[0])


def _line_cells_from_pairs(
    pairs: Iterable[tuple[tuple[int, int], tuple[int, int]]]
) -> set[tuple[int, int]]:
    line_cells: set[tuple[int, int]] = set()
    for a, b in pairs:
        line_cells.update(_diag_cells(a, b))
    return line_cells


def _apply_rule(
    grid: np.ndarray,
    pairs: Iterable[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    line_cells = _line_cells_from_pairs(pairs)
    for r, c in line_cells:
        out[r, c] = BLUE

    magentas = [tuple(map(int, rc)) for rc in np.argwhere(g == MAGENTA)]
    activated = [p for p in magentas if p in line_cells]

    for r, c in magentas:
        out[r, c] = MAGENTA

    for r, c in activated:
        assert 1 <= r < h - 1 and 1 <= c < w - 1, "activated trigger must fit a full 3x3 frame"
        for rr, cc in _ring_cells((r, c)):
            out[rr, cc] = YELLOW
        out[r, c] = MAGENTA

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each uniquely matched blue diagonal pair, then expand only the magenta triggers that lie on those drawn diagonals.

    Input structure:
    - A black background contains 2-3 blue endpoint pairs and several single magenta pixels.
    - Each blue endpoint belongs to exactly one 45-degree diagonal pair.
    - Some magenta pixels lie exactly on the blue-to-blue diagonal segments, while others are off those segments as distractors.

    Rule:
    - Find the unique pairing of blue pixels into diagonal-aligned pairs and draw each full diagonal segment in blue, including endpoints.
    - A magenta pixel is activated iff its coordinate lies on one of those drawn segment cells.
    - For every activated magenta pixel, draw a centered 3x3 hollow square in yellow; keep the trigger center itself magenta.
    - Yellow frame pixels overwrite any blue diagonal cells underneath them, while non-activated magenta distractors stay unchanged.

    Key insight:
    - The magenta markers cannot be expanded directly from the input: they must first be filtered by whether the blue connection step passes through them, and only those line-dependent triggers produce yellow frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    blue_points = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    pairs = _find_unique_blue_pairing(blue_points)
    return _apply_rule(g, pairs)


# --------------- generator ---------------

GRID_MIN = 13
GRID_MAX = 16
SEGMENT_COUNT_CHOICES = (2, 3)
SEGMENT_LEN_MIN = 4
SEGMENT_LEN_MAX = 7
MAX_GENERATION_ATTEMPTS = 400
SEGMENT_PLACE_ATTEMPTS = 120
TRIGGER_PLACE_ATTEMPTS = 120
RANDOM_DISTRactor_ATTEMPTS = 200


def _chebyshev_box(center: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    r, c = center
    return {
        (rr, cc)
        for rr in range(r - radius, r + radius + 1)
        for cc in range(c - radius, c + radius + 1)
    }


def _path_step(path: list[tuple[int, int]]) -> tuple[int, int]:
    return path[1][0] - path[0][0], path[1][1] - path[0][1]


def _sample_segment_path(h: int, w: int, dist: int, dc: int) -> list[tuple[int, int]]:
    r0 = random.randint(0, h - dist - 1)
    if dc == 1:
        c0 = random.randint(0, w - dist - 1)
    else:
        c0 = random.randint(dist, w - 1)
    return [(r0 + k, c0 + dc * k) for k in range(dist + 1)]


def _segment_extension_cells(path: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = _path_step(path)
    a_prev = (path[0][0] - dr, path[0][1] - dc)
    b_next = (path[-1][0] + dr, path[-1][1] + dc)
    cells = []
    for p in (a_prev, b_next):
        if 0 <= p[0] < h and 0 <= p[1] < w:
            cells.append(p)
    return cells


def _is_same_supporting_diagonal(cell: tuple[int, int], path: list[tuple[int, int]]) -> bool:
    return _is_diagonal(cell, path[0]) and (
        (path[0][0] - path[0][1] == cell[0] - cell[1])
        or (path[0][0] + path[0][1] == cell[0] + cell[1])
    )


def _place_segments(
    h: int,
    w: int,
    n_segments: int,
    active_segment_ids: set[int],
) -> list[dict] | None:
    segments: list[dict] = []
    blue_points: list[tuple[int, int]] = []
    occupied_paths: set[tuple[int, int]] = set()
    reserved_boxes: set[tuple[int, int]] = set()

    for seg_id in range(n_segments):
        want_trigger = seg_id in active_segment_ids
        placed = False
        for _ in range(SEGMENT_PLACE_ATTEMPTS):
            dist = random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            dc = random.choice((-1, 1))
            path = _sample_segment_path(h, w, dist, dc)
            a, b = path[0], path[-1]

            if any(_is_diagonal(p, q) for p in (a, b) for q in blue_points):
                continue
            if set(path) & occupied_paths:
                continue
            if set(path) & reserved_boxes:
                continue

            trigger = None
            if want_trigger:
                candidates = []
                for idx in range(2, dist - 1):
                    center = path[idx]
                    box = _chebyshev_box(center)
                    if not (1 <= center[0] < h - 1 and 1 <= center[1] < w - 1):
                        continue
                    if (box - set(path)) & occupied_paths:
                        continue
                    if box & reserved_boxes:
                        continue
                    if a in box or b in box:
                        continue
                    candidates.append(center)
                if not candidates:
                    continue
                trigger = random.choice(candidates)

            segments.append({"path": path, "trigger": trigger})
            blue_points.extend([a, b])
            occupied_paths.update(path)
            if trigger is not None:
                reserved_boxes.update(_chebyshev_box(trigger))
            placed = True
            break

        if not placed:
            return None

    pairs = _find_unique_blue_pairing(blue_points)
    if len(pairs) != n_segments:
        return None
    return segments


def _place_inactive_triggers(
    h: int,
    w: int,
    segments: list[dict],
) -> list[tuple[int, int]] | None:
    occupied_paths = {cell for seg in segments for cell in seg["path"]}
    active_centers = {seg["trigger"] for seg in segments if seg["trigger"] is not None}
    reserved_boxes = set().union(*(_chebyshev_box(p) for p in active_centers)) if active_centers else set()

    candidates_on_extensions = []
    for seg in segments:
        for cell in _segment_extension_cells(seg["path"], (h, w)):
            if cell in occupied_paths or cell in reserved_boxes or cell in active_centers:
                continue
            candidates_on_extensions.append(cell)
    if not candidates_on_extensions:
        return None

    inactive = [random.choice(candidates_on_extensions)]
    used = set(inactive)

    extra_count = random.choice((0, 1))
    for _ in range(extra_count):
        for _ in range(RANDOM_DISTRactor_ATTEMPTS):
            cell = (random.randint(0, h - 1), random.randint(0, w - 1))
            if cell in used or cell in occupied_paths or cell in reserved_boxes or cell in active_centers:
                continue
            if any(_is_same_supporting_diagonal(cell, seg["path"]) and cell not in _segment_extension_cells(seg["path"], (h, w)) for seg in segments):
                continue
            used.add(cell)
            inactive.append(cell)
            break

    return inactive


def _build_input_grid(
    h: int,
    w: int,
    segments: list[dict],
    inactive: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for seg in segments:
        grid[seg["path"][0]] = BLUE
        grid[seg["path"][-1]] = BLUE
        if seg["trigger"] is not None:
            grid[seg["trigger"]] = MAGENTA
    for cell in inactive:
        grid[cell] = MAGENTA
    return grid


def _line_only_output(grid: np.ndarray, pairs: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in _line_cells_from_pairs(pairs):
        out[r, c] = BLUE
    for r, c in np.argwhere(grid == MAGENTA):
        out[int(r), int(c)] = MAGENTA
    return out


def _outline_all_magentas_output(
    grid: np.ndarray, pairs: list[tuple[tuple[int, int], tuple[int, int]]]
) -> np.ndarray:
    out = _line_only_output(grid, pairs)
    h, w = out.shape
    for r0, c0 in np.argwhere(grid == MAGENTA):
        r, c = int(r0), int(c0)
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            continue
        for rr, cc in _ring_cells((r, c)):
            out[rr, cc] = YELLOW
        out[r, c] = MAGENTA
    return out


def _validate_example(inp: np.ndarray, segments: list[dict], inactive: list[tuple[int, int]]) -> bool:
    blue_points = [tuple(map(int, rc)) for rc in np.argwhere(inp == BLUE)]
    pairs = _find_unique_blue_pairing(blue_points)
    expected = _apply_rule(inp, pairs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    line_cells = _line_cells_from_pairs(pairs)
    active = [seg["trigger"] for seg in segments if seg["trigger"] is not None]
    if not active or not inactive:
        return False
    if any(p not in line_cells for p in active):
        return False
    if any(p in line_cells for p in inactive):
        return False

    if np.array_equal(expected, _line_only_output(inp, pairs)):
        return False
    if np.array_equal(expected, _outline_all_magentas_output(inp, pairs)):
        return False

    for center in active:
        assert center is not None
        ring = _ring_cells(center)
        if not any(inp[r, c] == BG and expected[r, c] == YELLOW for r, c in ring):
            return False
        if not any((r, c) in line_cells and expected[r, c] == YELLOW for r, c in ring):
            return False

    if not any(
        any(_is_same_supporting_diagonal(cell, seg["path"]) and cell not in seg["path"] for seg in segments)
        for cell in inactive
    ):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n_segments = random.choice(SEGMENT_COUNT_CHOICES)

    if n_segments == 2:
        n_active = random.choices([1, 2], weights=[3, 2])[0]
    else:
        n_active = random.choices([1, 2, 3], weights=[2, 4, 2])[0]
    active_segment_ids = set(random.sample(range(n_segments), n_active))

    segments = _place_segments(h, w, n_segments, active_segment_ids)
    if segments is None:
        return None

    inactive = _place_inactive_triggers(h, w, segments)
    if inactive is None:
        return None

    inp = _build_input_grid(h, w, segments, inactive)
    if not _validate_example(inp, segments, inactive):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
