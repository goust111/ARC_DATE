from __future__ import annotations

import random

import numpy as np


CW_COLORS = (1, 2)
CCW_COLORS = (3, 4)
MARKER_COLORS = CW_COLORS + CCW_COLORS
CHIRALITY = {color: True for color in CW_COLORS}
CHIRALITY.update({color: False for color in CCW_COLORS})

OBSTACLE_COLORS = (5, 6, 7, 8, 9)
GEN_RETRIES = 500
GRID_ROWS_RANGE = (18, 26)
GRID_COLS_RANGE = (18, 26)
OBSTACLE_ROWS_RANGE = (8, 14)
OBSTACLE_COLS_RANGE = (8, 14)
MIN_OBSTACLE_WIDTH = 5
EDGE_MARGIN = 4
MIN_SIGHT_GAP = 3
MIN_CYCLE_SPACING = 4
MIN_WALK = 4
LONG_MARGIN = 6

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _outside_info(solid_cell: tuple[int, int], outside_cell: tuple[int, int]) -> str:
    sr, sc = solid_cell
    orow, ocol = outside_cell
    diff = (orow - sr, ocol - sc)
    mapping = {
        (-1, 0): "top",
        (1, 0): "bottom",
        (0, -1): "left",
        (0, 1): "right",
    }
    assert diff in mapping, "outside cell must be 4-neighbor adjacent to solid"
    return mapping[diff]


def _boundary_cycle(solid: np.ndarray):
    """Trace the clockwise boundary-edge cycle of a connected solid polyomino."""
    h, w = solid.shape
    edges = []
    next_by_vertex = {}
    edge_by_key = {}

    def add(v0, v1, outside, cell):
        idx = len(edges)
        edges.append(
            {
                "v0": v0,
                "v1": v1,
                "outside": outside,
                "solid": cell,
                "side": _outside_info(cell, outside),
            }
        )
        assert v0 not in next_by_vertex, "boundary should have unique outgoing edge per vertex"
        next_by_vertex[v0] = idx
        edge_by_key[(outside, cell)] = idx

    for r in range(h):
        for c in range(w):
            if not solid[r, c]:
                continue
            cell = (r, c)
            if r == 0 or not solid[r - 1, c]:
                add((r, c), (r, c + 1), (r - 1, c), cell)
            if c == w - 1 or not solid[r, c + 1]:
                add((r, c + 1), (r + 1, c + 1), (r, c + 1), cell)
            if r == h - 1 or not solid[r + 1, c]:
                add((r + 1, c + 1), (r + 1, c), (r + 1, c), cell)
            if c == 0 or not solid[r, c - 1]:
                add((r + 1, c), (r, c), (r, c - 1), cell)

    assert edges, "solid must have a boundary"
    cycle = []
    used = set()
    idx = 0
    while idx not in used:
        used.add(idx)
        edge = edges[idx]
        cycle.append(edge)
        idx = next_by_vertex[edge["v1"]]
    assert idx == 0 and len(used) == len(edges), "expected a single simple exterior cycle"
    return cycle, edge_by_key


def _split_special_cells(grid: np.ndarray, bg: int, obstacle: int):
    flags = {}
    markers = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, obstacle):
                continue
            pos = (r, c)
            if any(_in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == obstacle for nr, nc in _neighbors4(r, c)):
                flags.setdefault(color, []).append(pos)
            else:
                markers.append((color, pos))
    return flags, markers


def _find_sightline(
    grid: np.ndarray,
    marker_pos: tuple[int, int],
    solid: np.ndarray,
    bg: int,
):
    hits = []
    h, w = grid.shape
    r0, c0 = marker_pos
    for dr, dc in DIRS:
        r, c = r0 + dr, c0 + dc
        last_bg = None
        while 0 <= r < h and 0 <= c < w:
            if solid[r, c]:
                assert last_bg is not None, "marker should not touch the obstacle directly"
                hits.append(((dr, dc), last_bg, (r, c)))
                break
            if int(grid[r, c]) != bg:
                break
            last_bg = (r, c)
            r += dr
            c += dc
    assert len(hits) == 1, "each marker must have exactly one unobstructed sightline to the solid"
    return hits[0]


def _trace_to_flag(
    cycle,
    edge_by_key,
    start_outside: tuple[int, int],
    touched_solid: tuple[int, int],
    flags_of_color: set[tuple[int, int]],
    clockwise: bool,
):
    start_idx = edge_by_key[(start_outside, touched_solid)]
    n = len(cycle)
    for step in range(n):
        idx = (start_idx + step) % n if clockwise else (start_idx - step) % n
        cell = cycle[idx]["outside"]
        if any((nr, nc) in flags_of_color for nr, nc in _neighbors4(*cell)):
            return cell, step
    raise AssertionError("no same-color flag reachable along the contour")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marker from its clear sightline onto the obstacle wall, then follow the exterior contour with the color-implied handedness until it reaches the first cell beside its matching flag.

    Input structure:
    - A dominant background surrounds one large connected obstacle of a single color.
    - Several boundary cells of that obstacle are recolored into flags, each sharing a color with one or more exterior single-pixel markers.
    - Every marker sits away from the obstacle and has exactly one clear cardinal sightline to the obstacle-or-flag solid.

    Rule:
    - For each marker, walk straight along its unique sightline and stop at the last background cell before the solid boundary.
    - Treat obstacle cells together with flags as one solid silhouette, and enumerate its exterior contour.
    - If the marker color belongs to the clockwise color class, follow that contour clockwise; otherwise follow it counterclockwise.
    - Move the marker to the first contour cell that is 4-neighbor adjacent to a same-color flag. Erase the old marker positions; keep obstacle cells and flags unchanged.

    Key insight:
    - The matching flag color tells you which flag matters, but the marker color class tells you which way to wrap around the same silhouette; the correct destination is the first same-color flag encounter along that handed contour, not the nearest flag by distance.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    non_bg = [int(c) for c in np.unique(g) if int(c) != bg]
    assert non_bg, "expected at least one foreground color"
    obstacle = max(non_bg, key=lambda c: int(np.count_nonzero(g == c)))

    flags, markers = _split_special_cells(g, bg, obstacle)
    assert markers, "expected at least one marker"
    solid = (g == obstacle).copy()
    for cells in flags.values():
        for r, c in cells:
            solid[r, c] = True
    assert _connected(solid), "obstacle plus flags must form one connected solid silhouette"

    cycle, edge_by_key = _boundary_cycle(solid)
    flag_sets = {color: set(cells) for color, cells in flags.items()}

    out = g.copy()
    claimed = set()
    for color, marker_pos in sorted(markers):
        assert color in CHIRALITY, "unknown marker color"
        assert color in flag_sets, "each marker color must have at least one matching flag"
        _, contour_start, touched_solid = _find_sightline(g, marker_pos, solid, bg)
        dest, _ = _trace_to_flag(
            cycle,
            edge_by_key,
            contour_start,
            touched_solid,
            flag_sets[color],
            clockwise=CHIRALITY[color],
        )
        assert int(g[dest]) == bg, "marker destination must be a background cell"
        assert dest not in claimed, "marker destinations must be distinct"
        claimed.add(dest)
        out[marker_pos] = bg
        out[dest] = color
    return out


def _make_obstacle_mask(h: int, w: int) -> np.ndarray | None:
    obs_h = random.randint(*OBSTACLE_ROWS_RANGE)
    obs_w = random.randint(*OBSTACLE_COLS_RANGE)
    if obs_h + 2 * EDGE_MARGIN > h or obs_w + 2 * EDGE_MARGIN > w:
        return None

    left = [random.randint(0, max(0, obs_w // 4))]
    right = [random.randint(left[0] + MIN_OBSTACLE_WIDTH - 1, obs_w - 1)]
    for _ in range(1, obs_h):
        new_left = left[-1] + random.choice((-1, 0, 1))
        new_left = min(max(new_left, 0), obs_w - MIN_OBSTACLE_WIDTH)
        min_right = new_left + MIN_OBSTACLE_WIDTH - 1
        max_right = obs_w - 1
        new_right = right[-1] + random.choice((-1, 0, 1))
        new_right = min(max(new_right, min_right), max_right)
        left.append(new_left)
        right.append(new_right)

    if min(r - l + 1 for l, r in zip(left, right)) < MIN_OBSTACLE_WIDTH:
        return None

    r0 = random.randint(EDGE_MARGIN, h - EDGE_MARGIN - obs_h)
    c0 = random.randint(EDGE_MARGIN, w - EDGE_MARGIN - obs_w)
    mask = np.zeros((h, w), dtype=bool)
    for i, (lft, rgt) in enumerate(zip(left, right)):
        mask[r0 + i, c0 + lft : c0 + rgt + 1] = True

    if not _connected(mask):
        return None
    return mask


def _marker_candidate(edge, shape: tuple[int, int]):
    h, w = shape
    outside = edge["outside"]
    solid = edge["solid"]
    side = edge["side"]
    if side == "top":
        marker = (0, outside[1])
        gap = solid[0] - marker[0] - 1
    elif side == "bottom":
        marker = (h - 1, outside[1])
        gap = marker[0] - solid[0] - 1
    elif side == "left":
        marker = (outside[0], 0)
        gap = solid[1] - marker[1] - 1
    else:
        marker = (outside[0], w - 1)
        gap = marker[1] - solid[1] - 1
    if gap < MIN_SIGHT_GAP:
        return None
    if not _in_bounds(shape, *outside):
        return None
    return {
        "edge_idx": None,
        "start": outside,
        "touch": solid,
        "side": side,
        "marker": marker,
    }


def _straight_flag_candidates(solid: np.ndarray, edge_by_key) -> list[dict]:
    out = []
    h, w = solid.shape
    for r in range(h):
        for c in range(w):
            if not solid[r, c]:
                continue
            outs = [(nr, nc) for nr, nc in _neighbors4(r, c) if _in_bounds(solid.shape, nr, nc) and not solid[nr, nc]]
            if len(outs) != 1:
                continue
            outside = outs[0]
            key = (outside, (r, c))
            if key not in edge_by_key:
                continue
            out.append({"flag": (r, c), "stop_idx": edge_by_key[key], "stop": outside})
    return out


def _cyclic_gap(a: int, b: int, n: int) -> int:
    d = abs(a - b)
    return min(d, n - d)


def _choose_starts(cycle, solid: np.ndarray) -> list[dict] | None:
    side_to_candidates = {side: [] for side in ("top", "right", "bottom", "left")}
    for idx, edge in enumerate(cycle):
        candidate = _marker_candidate(edge, solid.shape)
        if candidate is None:
            continue
        candidate["edge_idx"] = idx
        side_to_candidates[candidate["side"]].append(candidate)

    chosen = []
    used_markers = set()
    for side in ("top", "right", "bottom", "left"):
        candidates = side_to_candidates[side][:]
        random.shuffle(candidates)
        pick = None
        for cand in candidates:
            if cand["marker"] in used_markers:
                continue
            if any(_cyclic_gap(cand["edge_idx"], other["edge_idx"], len(cycle)) < MIN_CYCLE_SPACING for other in chosen):
                continue
            pick = cand
            break
        if pick is None:
            return None
        chosen.append(pick)
        used_markers.add(pick["marker"])
    random.shuffle(chosen)
    return chosen


def _assign_flags(starts: list[dict], flag_candidates: list[dict], cycle_len: int) -> list[dict] | None:
    colors = list(MARKER_COLORS)
    random.shuffle(colors)
    for start, color in zip(starts, colors):
        start["color"] = color
        start["clockwise"] = CHIRALITY[color]

    required_long = {random.choice(CW_COLORS), random.choice(CCW_COLORS)}
    for start in starts:
        start["want_long"] = start["color"] in required_long or (random.random() < 0.35)

    ordered_flags = flag_candidates[:]
    random.shuffle(ordered_flags)
    solution = []

    def rec(i: int) -> bool:
        if i == len(starts):
            return True
        start = starts[i]
        for cand in ordered_flags:
            if any(cand["flag"] == placed["flag"] or cand["stop"] == placed["stop"] for placed in solution):
                continue
            if any(_cyclic_gap(cand["stop_idx"], placed["stop_idx"], cycle_len) < MIN_CYCLE_SPACING for placed in solution):
                continue
            if any(_cyclic_gap(cand["stop_idx"], placed["edge_idx"], cycle_len) < 2 for placed in starts):
                continue

            if start["clockwise"]:
                dist = (cand["stop_idx"] - start["edge_idx"]) % cycle_len
            else:
                dist = (start["edge_idx"] - cand["stop_idx"]) % cycle_len
            alt = cycle_len - dist
            if dist < MIN_WALK:
                continue
            if start["want_long"]:
                if dist < alt + LONG_MARGIN:
                    continue
            elif dist <= 1:
                continue

            solution.append({**start, **cand, "distance": dist, "alt_distance": alt})
            if rec(i + 1):
                return True
            solution.pop()
        return False

    if not rec(0):
        return None

    cw_long = [item for item in solution if item["clockwise"] and item["distance"] > item["alt_distance"]]
    ccw_long = [item for item in solution if (not item["clockwise"]) and item["distance"] > item["alt_distance"]]
    if not cw_long or not ccw_long:
        return None
    return solution


def _render_input(shape, obstacle_color: int, solid: np.ndarray, assignments: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[solid] = obstacle_color
    for item in assignments:
        fr, fc = item["flag"]
        mr, mc = item["marker"]
        grid[fr, fc] = item["color"]
        grid[mr, mc] = item["color"]
    return grid


def _semantic_check(inp: np.ndarray) -> None:
    out = main(inp)
    bg = _get_bg(inp)
    non_bg = [int(c) for c in np.unique(inp) if int(c) != bg]
    obstacle = max(non_bg, key=lambda c: int(np.count_nonzero(inp == c)))
    flags, markers = _split_special_cells(inp, bg, obstacle)
    assert {color for color, _ in markers} == set(MARKER_COLORS)
    assert {color for color in flags} == set(MARKER_COLORS)

    solid = inp == obstacle
    for cells in flags.values():
        for r, c in cells:
            solid[r, c] = True
    assert _connected(solid)

    cycle, edge_by_key = _boundary_cycle(solid)
    flag_sets = {color: set(cells) for color, cells in flags.items()}
    used_dests = set()
    cw_long = 0
    ccw_long = 0
    for color, marker in markers:
        (_, _), start, touched = _find_sightline(inp, marker, solid, bg)
        dest, dist = _trace_to_flag(cycle, edge_by_key, start, touched, flag_sets[color], CHIRALITY[color])
        assert dest not in used_dests
        used_dests.add(dest)
        alt = len(cycle) - dist
        if CHIRALITY[color] and dist > alt:
            cw_long += 1
        if (not CHIRALITY[color]) and dist > alt:
            ccw_long += 1
        moved = np.argwhere(out == color)
        assert moved.shape == (2, 2), "one marker and one flag should remain for each color"
        assert any(tuple(map(int, cell)) == dest for cell in moved), "output must place marker at traced destination"
    assert cw_long >= 1 and ccw_long >= 1


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS_RANGE)
    w = random.randint(*GRID_COLS_RANGE)
    solid = _make_obstacle_mask(h, w)
    if solid is None:
        return None

    cycle, edge_by_key = _boundary_cycle(solid)
    if len(cycle) < 30:
        return None
    flag_candidates = _straight_flag_candidates(solid, edge_by_key)
    if len(flag_candidates) < len(MARKER_COLORS):
        return None

    starts = _choose_starts(cycle, solid)
    if starts is None:
        return None
    assignments = _assign_flags(starts, flag_candidates, len(cycle))
    if assignments is None:
        return None

    obstacle_color = random.choice(OBSTACLE_COLORS)
    inp = _render_input((h, w), obstacle_color, solid, assignments)

    obstacle_only = inp == obstacle_color
    if not _connected(obstacle_only):
        return None

    try:
        _semantic_check(inp)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
