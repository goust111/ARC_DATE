from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
MAROON = 9
MARKER_COLORS = [1, 2, 3, 4, 6, 8]
TRACK_BODY = {GREY, ORANGE, MAROON}
SPECIAL_COLORS = {BLACK, GREY, ORANGE, MAROON}


def _is_endpoint_color(color: int) -> bool:
    return int(color) not in SPECIAL_COLORS


def _nonblack_runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(line.size)
    while i < n:
        if int(line[i]) == BLACK:
            i += 1
            continue
        start = i
        while i < n and int(line[i]) != BLACK:
            i += 1
        runs.append((start, i - 1))
    return runs


def _parse_track_run(line: np.ndarray, start: int, end: int) -> dict | None:
    if end - start + 1 < 7:
        return None

    left_color = int(line[start])
    right_color = int(line[end])
    if not (_is_endpoint_color(left_color) and _is_endpoint_color(right_color)):
        return None
    if left_color == right_color:
        return None

    body = [int(v) for v in line[start + 1 : end]]
    if len(body) < 5 or any(v not in TRACK_BODY for v in body):
        return None
    if body.count(ORANGE) != 1 or body.count(MAROON) != 1:
        return None

    slider = start + 1 + body.index(ORANGE)
    barrier = start + 1 + body.index(MAROON)
    if abs(slider - barrier) < 2:
        return None

    return {
        "start": int(start),
        "end": int(end),
        "start_color": left_color,
        "end_color": right_color,
        "slider": int(slider),
        "barrier": int(barrier),
    }


def _find_tracks(g: np.ndarray) -> list[dict]:
    h, w = map(int, g.shape)
    tracks: list[dict] = []

    for r in range(h):
        for c0, c1 in _nonblack_runs(g[r, :]):
            parsed = _parse_track_run(g[r, :], c0, c1)
            if parsed is None:
                continue
            parsed.update({"orientation": "h", "fixed": int(r)})
            tracks.append(parsed)

    for c in range(w):
        for r0, r1 in _nonblack_runs(g[:, c]):
            parsed = _parse_track_run(g[:, c], r0, r1)
            if parsed is None:
                continue
            parsed.update({"orientation": "v", "fixed": int(c)})
            tracks.append(parsed)

    slider_cells: set[tuple[int, int]] = set()
    barrier_cells: set[tuple[int, int]] = set()
    for tr in tracks:
        if tr["orientation"] == "h":
            slider = (int(tr["fixed"]), int(tr["slider"]))
            barrier = (int(tr["fixed"]), int(tr["barrier"]))
        else:
            slider = (int(tr["slider"]), int(tr["fixed"]))
            barrier = (int(tr["barrier"]), int(tr["fixed"]))
        assert slider not in slider_cells, "each orange slider must belong to one track"
        assert barrier not in barrier_cells, "each maroon barrier must belong to one track"
        slider_cells.add(slider)
        barrier_cells.add(barrier)

    expected_sliders = {tuple(map(int, p)) for p in np.argwhere(g == ORANGE)}
    expected_barriers = {tuple(map(int, p)) for p in np.argwhere(g == MAROON)}
    assert tracks and slider_cells == expected_sliders and barrier_cells == expected_barriers
    return tracks


def main(grid: np.ndarray) -> np.ndarray:
    """Send each slider away from its blocker and recolor it as the endpoint it reaches.

    Input structure:
    - The background is black.
    - Each track is one horizontal or vertical contiguous line: a colored endpoint,
      at least five grey body cells, and a different colored endpoint.
    - Inside each track body exactly one orange slider and one maroon blocker
      replace grey cells, with at least one grey cell between them.
    - Tracks do not intersect, so every slider and blocker belongs to one line.

    Rule:
    - On each track, compare the blocker position with the orange slider position.
    - The slider moves in the straight direction away from the blocker until it
      reaches the colored endpoint on that side of the track.
    - The slider's original body cell becomes grey.
    - The reached endpoint remains occupied in its own marker color; the blocker
      stays maroon and all other grey body cells and unreached endpoint markers
      stay fixed.

    Key insight:
    - The maroon blocker is the direction code, while the two endpoint marker
      colors are the possible final colors for the escaping slider.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for tr in _find_tracks(g):
        start = int(tr["start"])
        end = int(tr["end"])
        slider = int(tr["slider"])
        barrier = int(tr["barrier"])
        fixed = int(tr["fixed"])
        escape_positive = barrier < slider
        target = end if escape_positive else start
        target_color = int(tr["end_color"] if escape_positive else tr["start_color"])

        if tr["orientation"] == "h":
            out[fixed, slider] = GREY
            out[fixed, target] = target_color
        else:
            out[slider, fixed] = GREY
            out[target, fixed] = target_color

    return out


# --------------- generator ---------------

GEN_SIZE_RANGE = (18, 30)
GEN_TRACK_COUNT_RANGE = (4, 6)
GEN_INTERIOR_LEN_RANGE = (5, 12)
GEN_PLACEMENT_ATTEMPTS = 120
GEN_ATTEMPTS = 500


def _track_cells(orientation: str, fixed: int, start: int, interior_len: int) -> list[tuple[int, int]]:
    total_len = int(interior_len + 2)
    if orientation == "h":
        return [(int(fixed), int(start + k)) for k in range(total_len)]
    return [(int(start + k), int(fixed)) for k in range(total_len)]


def _padded_cells(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    padded.add((rr, cc))
    return padded


def _slider_barrier_offsets(interior_len: int, escape_positive: bool) -> tuple[int, int]:
    positions = list(range(1, interior_len + 1))
    if escape_positive:
        pairs = [(s, b) for s in positions for b in positions if b <= s - 2]
    else:
        pairs = [(s, b) for s in positions for b in positions if b >= s + 2]
    return random.choice(pairs)


def _render_track(
    grid: np.ndarray,
    orientation: str,
    fixed: int,
    start: int,
    interior_len: int,
    escape_positive: bool,
) -> dict:
    color_a, color_b = random.sample(MARKER_COLORS, 2)
    slider_offset, barrier_offset = _slider_barrier_offsets(interior_len, escape_positive)
    end = start + interior_len + 1

    if orientation == "h":
        grid[fixed, start] = color_a
        grid[fixed, end] = color_b
        grid[fixed, start + 1 : end] = GREY
        grid[fixed, start + slider_offset] = ORANGE
        grid[fixed, start + barrier_offset] = MAROON
        slider_cell = (fixed, start + slider_offset)
        target_cell = (fixed, end if escape_positive else start)
        target_color = color_b if escape_positive else color_a
    else:
        grid[start, fixed] = color_a
        grid[end, fixed] = color_b
        grid[start + 1 : end, fixed] = GREY
        grid[start + slider_offset, fixed] = ORANGE
        grid[start + barrier_offset, fixed] = MAROON
        slider_cell = (start + slider_offset, fixed)
        target_cell = (end if escape_positive else start, fixed)
        target_color = color_b if escape_positive else color_a

    return {
        "slider": slider_cell,
        "target": target_cell,
        "target_color": int(target_color),
    }


def _place_one_track(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    orientation: str,
    escape_positive: bool,
) -> dict | None:
    h, w = map(int, grid.shape)
    axis_len = w if orientation == "h" else h
    max_interior = min(GEN_INTERIOR_LEN_RANGE[1], axis_len - 2)
    if max_interior < GEN_INTERIOR_LEN_RANGE[0]:
        return None

    for _ in range(GEN_PLACEMENT_ATTEMPTS):
        interior_len = random.randint(GEN_INTERIOR_LEN_RANGE[0], max_interior)
        total_len = interior_len + 2
        start = random.randint(0, axis_len - total_len)
        fixed_limit = h if orientation == "h" else w
        fixed = random.randrange(fixed_limit)
        cells = _track_cells(orientation, fixed, start, interior_len)
        if any(cell in reserved for cell in cells):
            continue
        padded = _padded_cells(cells, h, w)
        if any(cell in reserved for cell in padded):
            continue
        reserved.update(padded)
        return _render_track(grid, orientation, fixed, start, interior_len, escape_positive)

    return None


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        sr, sc = spec["slider"]
        tr, tc = spec["target"]
        out[sr, sc] = GREY
        out[tr, tc] = int(spec["target_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_SIZE_RANGE)
    w = random.randint(*GEN_SIZE_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    reserved: set[tuple[int, int]] = set()

    roles = [("h", True), ("h", False), ("v", True), ("v", False)]
    for _ in range(random.randint(GEN_TRACK_COUNT_RANGE[0], GEN_TRACK_COUNT_RANGE[1]) - len(roles)):
        roles.append((random.choice(("h", "v")), bool(random.getrandbits(1))))
    random.shuffle(roles)

    specs: list[dict] = []
    for orientation, escape_positive in roles:
        spec = _place_one_track(grid, reserved, orientation, escape_positive)
        if spec is None:
            return None
        specs.append(spec)

    expected = _expected_from_specs(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
