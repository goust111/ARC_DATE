from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random


Side = str


def _foreground_roles(g: np.ndarray, bg: int) -> tuple[int, int]:
    foreground = colors_present(g, exclude={bg})
    assert len(foreground) == 2, "expected one trail color and one marker color"

    counts = {color: int(np.sum(g == color)) for color in foreground}
    min_count = min(counts.values())
    max_count = max(counts.values())
    marker_candidates = [color for color, count in counts.items() if count == min_count]
    trail_candidates = [color for color, count in counts.items() if count == max_count]
    assert len(marker_candidates) == 1 and len(trail_candidates) == 1, "foreground color roles must be unique by count"
    assert min_count < max_count, "marker color must be rarer than trail color"
    return marker_candidates[0], trail_candidates[0]


def _edge_distance(side: Side, shape: tuple[int, int], r: int, c: int) -> int:
    h, w = shape
    if side == "top":
        return r
    if side == "bottom":
        return h - 1 - r
    if side == "left":
        return c
    if side == "right":
        return w - 1 - c
    raise AssertionError(f"unknown side {side!r}")


def _source_side(g: np.ndarray, marker: int) -> tuple[Side, int]:
    marker_coords = np.argwhere(g == marker)
    assert marker_coords.size > 0, "expected marker pixels"

    max_distances: dict[Side, int] = {}
    for side in ("top", "bottom", "left", "right"):
        max_distances[side] = max(
            _edge_distance(side, g.shape, int(r), int(c)) for r, c in marker_coords
        )

    nearest_distance = min(max_distances.values())
    candidates = [side for side, distance in max_distances.items() if distance == nearest_distance]
    assert len(candidates) == 1, "marker pixels must identify exactly one source edge"
    return candidates[0], nearest_distance + 1


def _max_trail_run_from_edge(g: np.ndarray, side: Side, trail: int) -> int:
    h, w = g.shape
    best = 0

    if side == "left":
        for r in range(h):
            run = 0
            for c in range(w):
                if int(g[r, c]) != trail:
                    break
                run += 1
            best = max(best, run)
    elif side == "right":
        for r in range(h):
            run = 0
            for c in range(w - 1, -1, -1):
                if int(g[r, c]) != trail:
                    break
                run += 1
            best = max(best, run)
    elif side == "top":
        for c in range(w):
            run = 0
            for r in range(h):
                if int(g[r, c]) != trail:
                    break
                run += 1
            best = max(best, run)
    elif side == "bottom":
        for c in range(w):
            run = 0
            for r in range(h - 1, -1, -1):
                if int(g[r, c]) != trail:
                    break
                run += 1
            best = max(best, run)
    else:
        raise AssertionError(f"unknown side {side!r}")

    assert best > 0, "source edge must contain at least one trail run"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the patterned edge band to the opposite edge along every row or column whose band slice contains the trail color.

    Input structure
    - The dominant color is the background.
    - The two foreground colors are a frequent trail color and a rarer marker color.
    - All marker pixels lie nearest to one grid edge, and that edge contains trail-color runs that define the source-band thickness.

    Rule
    - Choose the source edge identified by the marker pixels.
    - A row or column perpendicular to that edge moves if its source-band slice contains the trail color.
    - Copy that slice unchanged to the opposite edge, erase the remaining trail cells to background, and keep marker-only lines fixed.

    Key insight
    - The marker color identifies the active edge, while the trail color identifies which parallel rows or columns carry an edge-band slice across the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = int(get_bg(g))
    marker, trail = _foreground_roles(g, bg)
    side, marker_depth = _source_side(g, marker)
    band = _max_trail_run_from_edge(g, side, trail)
    assert marker_depth <= band, "all markers must fit inside the source band"

    if side in ("left", "right"):
        assert 2 * band <= w, "source and destination bands must not overlap"
    else:
        assert 2 * band <= h, "source and destination bands must not overlap"

    out = np.full_like(g, bg)

    if side == "left":
        moving_rows: list[bool] = []
        for r in range(h):
            src = g[r, :band]
            moves = bool(np.any(src == trail))
            moving_rows.append(moves)
            if moves:
                out[r, w - band : w] = src
            else:
                out[r, g[r] == marker] = marker
        assert all(moving_rows[int(r)] for r, _c in np.argwhere(g == trail)), "every trail cell must lie on a moving row"
    elif side == "right":
        moving_rows = []
        for r in range(h):
            src = g[r, w - band : w]
            moves = bool(np.any(src == trail))
            moving_rows.append(moves)
            if moves:
                out[r, :band] = src
            else:
                out[r, g[r] == marker] = marker
        assert all(moving_rows[int(r)] for r, _c in np.argwhere(g == trail)), "every trail cell must lie on a moving row"
    elif side == "top":
        moving_cols: list[bool] = []
        for c in range(w):
            src = g[:band, c]
            moves = bool(np.any(src == trail))
            moving_cols.append(moves)
            if moves:
                out[h - band : h, c] = src
            else:
                out[g[:, c] == marker, c] = marker
        assert all(moving_cols[int(c)] for _r, c in np.argwhere(g == trail)), "every trail cell must lie on a moving column"
    else:
        assert side == "bottom"
        moving_cols = []
        for c in range(w):
            src = g[h - band : h, c]
            moves = bool(np.any(src == trail))
            moving_cols.append(moves)
            if moves:
                out[:band, c] = src
            else:
                out[g[:, c] == marker, c] = marker
        assert all(moving_cols[int(c)] for _r, c in np.argwhere(g == trail)), "every trail cell must lie on a moving column"

    return out

# --------------- generator (att2) ---------------

GEN_COLORS = range(10)
GEN_SIDES = ("top", "bottom", "left", "right")
GEN_GRID_MIN, GEN_GRID_MAX = 8, 20
GEN_BAND, GEN_CROSSBAR_OFFSET, GEN_SAFE_MARKER_MARGIN = 3, 1, 2
GEN_GROUP_MIN, GEN_GROUP_MAX, GEN_GROUP_SPAN = 2, 4, 3
GEN_GAP_MIN, GEN_GAP_MAX, GEN_MIN_FREE_AXIS = 1, 3, 4
GEN_CROSSBAR_MARKER_MIN, GEN_CROSSBAR_MARKER_MAX = 0, 2
GEN_NONMOVING_MIN, GEN_NONMOVING_MAX = 1, 4
GEN_NONMOVING_DEPTH_MIN, GEN_NONMOVING_DEPTH_MAX = 1, 2
GEN_MAX_ATTEMPTS = 300


def _gen_src_slice(side: str, bg: int, trail: int, marker: int, marker_count: int) -> np.ndarray:
    by_depth = [bg, bg, trail]
    for depth in range(marker_count):
        by_depth[depth] = marker
    if side in ("right", "bottom"):
        by_depth = list(reversed(by_depth))
    return np.array(by_depth, dtype=int)


def _gen_set_band_slice(grid: np.ndarray, side: str, pos: int, values: np.ndarray, dest: bool = False) -> None:
    h, w = grid.shape
    if (side == "left" and not dest) or (side == "right" and dest):
        grid[pos, :GEN_BAND] = values
    elif (side == "right" and not dest) or (side == "left" and dest):
        grid[pos, w - GEN_BAND : w] = values
    elif (side == "top" and not dest) or (side == "bottom" and dest):
        grid[:GEN_BAND, pos] = values
    else:
        grid[h - GEN_BAND : h, pos] = values


def _gen_fill_crossbar_trail(grid: np.ndarray, side: str, pos: int, trail: int) -> None:
    h, w = grid.shape
    if side == "left":
        grid[pos, GEN_BAND:w] = trail
    elif side == "right":
        grid[pos, : w - GEN_BAND] = trail
    elif side == "top":
        grid[GEN_BAND:h, pos] = trail
    else:
        grid[: h - GEN_BAND, pos] = trail


def _gen_set_source_markers(grid: np.ndarray, side: str, pos: int, depth: int, marker: int) -> None:
    h, w = grid.shape
    if side == "left":
        grid[pos, :depth] = marker
    elif side == "right":
        grid[pos, w - depth : w] = marker
    elif side == "top":
        grid[:depth, pos] = marker
    else:
        grid[h - depth : h, pos] = marker


def _gen_axis_layout(axis_len: int, num_groups: int) -> tuple[list[int], list[int], list[int]]:
    free_total = axis_len - GEN_GROUP_SPAN * num_groups
    gaps_between: list[int] = []
    remaining = free_total
    for i in range(num_groups - 1):
        future_gaps = num_groups - 2 - i
        high = min(GEN_GAP_MAX, remaining - future_gaps)
        gap = random.randint(GEN_GAP_MIN, high)
        gaps_between.append(gap)
        remaining -= gap

    before = random.randint(0, remaining)
    pos = before
    starts: list[int] = []
    occupied: set[int] = set()
    for i in range(num_groups):
        starts.append(pos)
        occupied.update(range(pos, pos + GEN_GROUP_SPAN))
        pos += GEN_GROUP_SPAN
        if i < num_groups - 1:
            pos += gaps_between[i]

    available = [i for i in range(axis_len) if i not in occupied]
    return starts, gaps_between, available


def _gen_choose_nonmoving(marker_positions: list[int], safe_marker_positions: list[int]) -> list[int]:
    nonmoving_count = random.randint(GEN_NONMOVING_MIN, min(GEN_NONMOVING_MAX, len(marker_positions)))
    chosen_nonmoving = [random.choice(safe_marker_positions)]
    remaining_positions = [p for p in marker_positions if p != chosen_nonmoving[0]]
    chosen_nonmoving.extend(random.sample(remaining_positions, nonmoving_count - 1))
    return chosen_nonmoving


def _gen_render_group_slices(
    out: np.ndarray, side: str, group_starts: list[int], bg: int, trail: int, marker: int
) -> list[tuple[int, bool, np.ndarray]]:
    moving_specs: list[tuple[int, bool, np.ndarray]] = []
    for start in group_starts:
        crossbar = start + GEN_CROSSBAR_OFFSET
        crossbar_markers = random.randint(GEN_CROSSBAR_MARKER_MIN, GEN_CROSSBAR_MARKER_MAX)
        for pos in (start, crossbar, start + GEN_GROUP_SPAN - 1):
            is_crossbar = pos == crossbar
            if is_crossbar:
                src = _gen_src_slice(side, bg, trail, marker, crossbar_markers)
            else:
                src = np.array([trail, trail, trail], dtype=int)
            moving_specs.append((pos, is_crossbar, src))
            _gen_set_band_slice(out, side, pos, src, dest=True)
    return moving_specs


def _gen_render_nonmoving_markers(
    out: np.ndarray, inp: np.ndarray, side: str, positions: list[int], marker: int
) -> None:
    for pos in positions:
        depth = random.randint(GEN_NONMOVING_DEPTH_MIN, GEN_NONMOVING_DEPTH_MAX)
        _gen_set_source_markers(out, side, pos, depth, marker)
        _gen_set_source_markers(inp, side, pos, depth, marker)


def _gen_render_input_movers(
    inp: np.ndarray, side: str, moving_specs: list[tuple[int, bool, np.ndarray]], trail: int
) -> None:
    for pos, is_crossbar, src in moving_specs:
        _gen_set_band_slice(inp, side, pos, src)
        if is_crossbar:
            _gen_fill_crossbar_trail(inp, side, pos, trail)


def _gen_attempt() -> np.ndarray | None:
    bg, trail, marker = random.sample(GEN_COLORS, 3)
    side = random.choice(GEN_SIDES)
    num_groups = random.randint(GEN_GROUP_MIN, GEN_GROUP_MAX)

    min_axis_len = GEN_GROUP_SPAN * num_groups + max(GEN_MIN_FREE_AXIS, num_groups - 1)
    axis_len = random.randint(max(GEN_GRID_MIN, min_axis_len), GEN_GRID_MAX)
    perp_len = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    if side in ("left", "right"):
        h, w = axis_len, perp_len
    else:
        h, w = perp_len, axis_len

    group_starts, _gaps_between, marker_positions = _gen_axis_layout(axis_len, num_groups)
    safe_marker_positions = [
        p for p in marker_positions if GEN_SAFE_MARKER_MARGIN <= p <= axis_len - GEN_SAFE_MARKER_MARGIN - 1
    ]
    if not safe_marker_positions:
        return None

    chosen_nonmoving = _gen_choose_nonmoving(marker_positions, safe_marker_positions)

    out = np.full((h, w), bg, dtype=int)
    inp = np.full((h, w), bg, dtype=int)

    moving_specs = _gen_render_group_slices(out, side, group_starts, bg, trail, marker)
    _gen_render_nonmoving_markers(out, inp, side, chosen_nonmoving, marker)
    _gen_render_input_movers(inp, side, moving_specs, trail)

    if int(np.sum(inp == marker)) >= int(np.sum(inp == trail)):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d255d7a7 input")
