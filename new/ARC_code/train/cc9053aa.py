from collections import deque

import numpy as np

from arc_utils import Color, in_bounds, neighbors4, neighbors8
import random


def _distances(passable: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    q = deque([start])
    dist = {start: 0}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(passable.shape, nr, nc) and passable[nr, nc]:
                nxt = (nr, nc)
                if nxt not in dist:
                    dist[nxt] = dist[(r, c)] + 1
                    q.append(nxt)
    return dist


def _unique_shortest_path(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> set[tuple[int, int]]:
    dist_start = _distances(passable, start)
    assert end in dist_start
    dist_end = _distances(passable, end)
    length = dist_start[end]
    path = {
        p
        for p in dist_start
        if p in dist_end and dist_start[p] + dist_end[p] == length
    }
    assert len(path) == length + 1
    return path


def _has_unique_simple_path(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> bool:
    stack = [(start, {start})]
    count = 0
    while stack:
        (r, c), seen = stack.pop()
        if (r, c) == end:
            count += 1
            if count > 1:
                return False
            continue
        for nr, nc in neighbors4(r, c):
            nxt = (nr, nc)
            if in_bounds(passable.shape, nr, nc) and passable[nr, nc] and nxt not in seen:
                stack.append((nxt, seen | {nxt}))
    return count == 1


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the unique safe route between the two maroon endpoints from teal to maroon.

    Input structure
    - A black background contains a teal 4-connected track, exactly two maroon endpoint pixels, and one or more orange blocker pixels embedded near the track.

    Rule
    - Treat each orange blocker and its 8-neighborhood as forbidden.
    - In the remaining teal/maroon track graph, there is exactly one simple
      4-connected path between the maroon endpoints.
    - Recolor teal pixels on that path to maroon; leave endpoints, blockers, and all other cells unchanged.

    Key insight
    - The orange pixels block nearby teal cells diagonally as well as orthogonally, which removes shorter decoy routes and makes the intended endpoint-to-endpoint path unique.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {
        Color.BLACK,
        Color.ORANGE,
        Color.TEAL,
        Color.MAROON,
    }

    endpoints = [tuple(map(int, p)) for p in np.argwhere(g == Color.MAROON)]
    blockers = [tuple(map(int, p)) for p in np.argwhere(g == Color.ORANGE)]
    assert len(endpoints) == 2
    assert blockers

    forbidden = np.zeros(g.shape, dtype=bool)
    for r, c in blockers:
        forbidden[r, c] = True
        for nr, nc in neighbors8(r, c):
            if in_bounds(g.shape, nr, nc):
                forbidden[nr, nc] = True
    assert all(not forbidden[r, c] for r, c in endpoints)

    passable = ((g == Color.TEAL) | (g == Color.MAROON)) & ~forbidden
    assert _has_unique_simple_path(passable, endpoints[0], endpoints[1])
    path = _unique_shortest_path(passable, endpoints[0], endpoints[1])

    out = g.copy()
    for r, c in path:
        if g[r, c] == Color.TEAL:
            out[r, c] = Color.MAROON
    return out

# --------------- generator (att1) ---------------

# Canvas and lattice bounds.
_GEN_HEIGHT_CHOICES = [7, 7, 9, 9, 9]
_GEN_WIDTH = 9
_GEN_BAR_ROW_START = 1
_GEN_BAR_ROW_STEP = 2
_GEN_BAR_COUNT_BY_HEIGHT = {7: [3], 9: [3, 4]}
_GEN_BAR_START_COL_RANGE = (1, 2)
_GEN_BAR_END_OFFSET_RANGE = (3, 2)
_GEN_MAX_BAR_GAPS = 2

# Connector and endpoint search limits.
_GEN_MIN_CONNECTORS = 2
_GEN_MAX_CONNECTORS = 5
_GEN_ENDPOINT_PRIMARY_LIMIT = 8
_GEN_ENDPOINT_SECONDARY_LIMIT = 10
_GEN_TARGET_PATH_RANGE = (12, 20)
_GEN_FALLBACK_MIN_DISTANCE = 10

# Track, blocker, and validation constraints.
_GEN_TRACK_SIZE_RANGE = (26, 42)
_GEN_BLOCKER_CHOICES = [1, 2]
_GEN_MIN_BLOCKER_ADJACENCY = 2
_GEN_BLOCKER_TRIES_FACTOR = 3
_GEN_MAX_BLOCKER_TRIES = 50
_GEN_BLOCKER_SEPARATION = 2
_GEN_COVERAGE_RANGE = (0.45, 0.72)
_GEN_TURN_RANGE = (2, 8)
_GEN_MAX_ATTEMPTS = 300

_GEN_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _gen_bfs_dist(track_set, start):
    """BFS distances from start over track_set."""
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in _GEN_DIRS4:
            nr, nc = r+dr, c+dc
            if (nr, nc) in track_set and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def _gen_path_cells(track_set, start, end):
    """Return (set of cells on ALL shortest paths, path_length) for uniqueness check."""
    dist_s = _gen_bfs_dist(track_set, start)
    if end not in dist_s:
        return None, None
    dist_e = _gen_bfs_dist(track_set, end)
    length = dist_s[end]
    cells = {p for p in dist_s if p in dist_e and dist_s[p] + dist_e[p] == length}
    return cells, length


def _gen_has_unique_simple_path(track_set, start, end):
    """Return True iff exactly one simple path connects start to end."""
    stack = [(start, {start})]
    count = 0
    while stack:
        cell, seen = stack.pop()
        if cell == end:
            count += 1
            if count > 1:
                return False
            continue
        r, c = cell
        for dr, dc in _GEN_DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in track_set and nxt not in seen:
                stack.append((nxt, seen | {nxt}))
    return count == 1


def _gen_choose_layout():
    """Choose grid size and horizontal bar columns."""
    H = random.choice(_GEN_HEIGHT_CHOICES)
    W = _GEN_WIDTH
    if H == 7:
        n_bars = _GEN_BAR_COUNT_BY_HEIGHT[H][0]
    else:
        n_bars = random.choice(_GEN_BAR_COUNT_BY_HEIGHT[H])
    bar_rows = [
        _GEN_BAR_ROW_START + _GEN_BAR_ROW_STEP * i for i in range(n_bars)
    ]
    if bar_rows[-1] >= H:
        return None

    bar_cols = []
    for _ in range(n_bars):
        c_start = random.randint(*_GEN_BAR_START_COL_RANGE)
        c_end = random.randint(
            W - _GEN_BAR_END_OFFSET_RANGE[0],
            W - _GEN_BAR_END_OFFSET_RANGE[1],
        )
        cols = list(range(c_start, c_end + 1))
        n_gaps = random.randint(0, _GEN_MAX_BAR_GAPS)
        interior = [c for c in cols if c != c_start and c != c_end]
        if n_gaps > 0 and len(interior) >= n_gaps * 2:
            gap_candidates = interior[:]
            random.shuffle(gap_candidates)
            gaps_chosen = []
            for gc in gap_candidates:
                if len(gaps_chosen) >= n_gaps:
                    break
                if all(abs(gc - g) > 1 for g in gaps_chosen):
                    gaps_chosen.append(gc)
            for gc in gaps_chosen:
                cols.remove(gc)
        bar_cols.append(set(cols))

    return H, W, bar_rows, bar_cols


def _gen_build_lattice(bar_rows, bar_cols):
    """Build connected horizontal bars plus vertical connectors."""
    track = set()
    for i, row in enumerate(bar_rows):
        for c in bar_cols[i]:
            track.add((row, c))

    for i in range(len(bar_rows) - 1):
        top_row = bar_rows[i]
        bot_row = bar_rows[i + 1]
        shared_cols = sorted(bar_cols[i] & bar_cols[i + 1])
        if len(shared_cols) < _GEN_MIN_CONNECTORS:
            return None
        n_conn = random.randint(
            _GEN_MIN_CONNECTORS, min(_GEN_MAX_CONNECTORS, len(shared_cols))
        )
        if len(shared_cols) >= 3 and n_conn >= _GEN_MIN_CONNECTORS:
            chosen = [shared_cols[0], shared_cols[-1]]
            mid = shared_cols[1:-1]
            random.shuffle(mid)
            chosen += mid[:n_conn - 2]
        else:
            chosen = random.sample(shared_cols, n_conn)

        for c in chosen:
            for r in range(top_row, bot_row + 1):
                track.add((r, c))

    if not track:
        return None
    start_cell = next(iter(track))
    visited = _gen_bfs_dist(track, start_cell)
    if len(visited) != len(track):
        return None
    return track


def _gen_border_candidates(track, H, W, bar_rows, bar_cols):
    """Find possible endpoint stems from lattice cells to the border."""
    border_candidates = []
    for (r, c) in track:
        if r == bar_rows[0]:
            if all((rr, c) not in track for rr in range(0, r)):
                border_candidates.append(((r, c), (0, c), 'top', r))
        if r == bar_rows[-1]:
            if all((rr, c) not in track for rr in range(r + 1, H)):
                border_candidates.append(((r, c), (H - 1, c), 'bot', H - 1 - r))
        bar_idx = bar_rows.index(r) if r in bar_rows else -1
        if bar_idx >= 0:
            if c == min(bar_cols[bar_idx]):
                if all((r, cc) not in track for cc in range(0, c)):
                    border_candidates.append(((r, c), (r, 0), 'left', c))
            if c == max(bar_cols[bar_idx]):
                if all((r, cc) not in track for cc in range(c + 1, W)):
                    border_candidates.append(((r, c), (r, W - 1), 'right', W - 1 - c))
    return border_candidates


def _gen_pair_distance(track, c1, c2):
    att1, _, _, stem_len1 = c1
    att2, _, _, stem_len2 = c2
    d = _gen_bfs_dist(track, att1)
    if att2 not in d:
        return None
    return d[att2] + stem_len1 + stem_len2


def _gen_choose_endpoints(track, border_candidates):
    """Pick two border endpoints whose stem-adjusted distance is task-like."""
    if len(border_candidates) < 2:
        return None
    random.shuffle(border_candidates)

    best_pair = None
    best_dist = 0
    for i in range(min(_GEN_ENDPOINT_PRIMARY_LIMIT, len(border_candidates))):
        for j in range(
            i + 1, min(_GEN_ENDPOINT_SECONDARY_LIMIT, len(border_candidates))
        ):
            c1 = border_candidates[i]
            c2 = border_candidates[j]
            total_d = _gen_pair_distance(track, c1, c2)
            if total_d is None:
                continue
            if (
                _GEN_TARGET_PATH_RANGE[0] <= total_d <= _GEN_TARGET_PATH_RANGE[1]
                and total_d > best_dist
            ):
                best_dist = total_d
                best_pair = (c1, c2)

    if best_pair is None:
        for i in range(len(border_candidates)):
            for j in range(i + 1, len(border_candidates)):
                c1 = border_candidates[i]
                c2 = border_candidates[j]
                total_d = _gen_pair_distance(track, c1, c2)
                if total_d is None:
                    continue
                if total_d >= _GEN_FALLBACK_MIN_DISTANCE:
                    best_pair = (c1, c2)
                    best_dist = total_d
                    break
            if best_pair:
                break
    return best_pair


def _gen_add_stem(track, attachment, side, H, W):
    r, c = attachment
    if side == 'top':
        for sr in range(0, r):
            track.add((sr, c))
    elif side == 'bot':
        for sr in range(r + 1, H):
            track.add((sr, c))
    elif side == 'left':
        for sc in range(0, c):
            track.add((r, sc))
    elif side == 'right':
        for sc in range(c + 1, W):
            track.add((r, sc))


def _gen_add_and_validate_endpoints(track, best_pair, H, W):
    """Add stems and return endpoint cells if they satisfy degree/border checks."""
    ep1_info, ep2_info = best_pair
    att1, pos1, side1, _ = ep1_info
    att2, pos2, side2, _ = ep2_info
    _gen_add_stem(track, att1, side1, H, W)
    _gen_add_stem(track, att2, side2, H, W)
    endpoint_cells = [pos1, pos2]

    for ep in endpoint_cells:
        deg = sum(1 for dr, dc in _GEN_DIRS4
                  if (ep[0]+dr, ep[1]+dc) in track)
        if deg != 1:
            return None

    for ep in endpoint_cells:
        if not (ep[0] == 0 or ep[0] == H - 1 or ep[1] == 0 or ep[1] == W - 1):
            return None
    return endpoint_cells


def _gen_blocker_candidates(track, endpoint_cells, H, W):
    """Find interior track cells that can act as orange blockers."""
    blocker_candidates = []
    for (r, c) in track:
        if (r, c) in endpoint_cells:
            continue
        if r <= 0 or r >= H - 1 or c <= 0 or c >= W - 1:
            continue
        nbr8 = [(r+dr, c+dc) for dr in [-1,0,1] for dc in [-1,0,1] if (dr,dc) != (0,0)]
        if any(n in endpoint_cells for n in nbr8):
            continue
        adj_track = sum(1 for dr, dc in _GEN_DIRS4 if (r+dr, c+dc) in track)
        if adj_track < _GEN_MIN_BLOCKER_ADJACENCY:
            continue
        blocker_candidates.append((r, c))
    return blocker_candidates


def _gen_forbidden(blockers):
    forbidden = set()
    for br, bc in blockers:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                forbidden.add((br + dr, bc + dc))
    return forbidden


def _gen_choose_blockers(n_blockers, blocker_candidates, attempt):
    if n_blockers == 1:
        idx = attempt % len(blocker_candidates)
        return [blocker_candidates[idx]]

    idx1 = random.randint(0, len(blocker_candidates) - 1)
    idx2 = random.randint(0, len(blocker_candidates) - 1)
    if idx1 == idx2:
        return None
    blockers = [blocker_candidates[idx1], blocker_candidates[idx2]]
    r1, c1 = blockers[0]
    r2, c2 = blockers[1]
    if (
        abs(r1-r2) <= _GEN_BLOCKER_SEPARATION
        and abs(c1-c2) <= _GEN_BLOCKER_SEPARATION
    ):
        return None
    return blockers


def _gen_ordered_path(passable, start, end, path_cells):
    """Order a unique shortest path from end back to start using BFS ranks."""
    dist_s = _gen_bfs_dist(passable, start)
    path_ordered = [end]
    current = end
    ok = True
    while current != start:
        found = False
        for dr, dc in _GEN_DIRS4:
            nxt = (current[0]+dr, current[1]+dc)
            if nxt in path_cells and dist_s.get(nxt, -1) == dist_s[current] - 1:
                path_ordered.append(nxt)
                current = nxt
                found = True
                break
        if not found:
            ok = False
            break
    return path_ordered if ok else None


def _gen_turn_count(path_ordered):
    turns = 0
    for k in range(1, len(path_ordered) - 1):
        d1 = (path_ordered[k][0] - path_ordered[k-1][0], path_ordered[k][1] - path_ordered[k-1][1])
        d2 = (path_ordered[k+1][0] - path_ordered[k][0], path_ordered[k+1][1] - path_ordered[k][1])
        if d1 != d2:
            turns += 1
    return turns


def _gen_validate_blocked_path(track, endpoint_cells, forbidden):
    passable = track - forbidden

    if endpoint_cells[0] not in passable or endpoint_cells[1] not in passable:
        return False

    if not _gen_has_unique_simple_path(passable, endpoint_cells[0], endpoint_cells[1]):
        return False

    path_cells, path_length = _gen_path_cells(passable, endpoint_cells[0], endpoint_cells[1])
    if path_cells is None:
        return False

    if len(path_cells) != path_length + 1:
        return False

    if path_length < _GEN_TARGET_PATH_RANGE[0] or path_length > _GEN_TARGET_PATH_RANGE[1]:
        return False

    coverage = len(path_cells) / len(track)
    if coverage < _GEN_COVERAGE_RANGE[0] or coverage > _GEN_COVERAGE_RANGE[1]:
        return False

    path_ordered = _gen_ordered_path(
        passable, endpoint_cells[0], endpoint_cells[1], path_cells
    )
    if path_ordered is None or len(path_ordered) != path_length + 1:
        return False

    turns = _gen_turn_count(path_ordered)
    return _GEN_TURN_RANGE[0] <= turns <= _GEN_TURN_RANGE[1]


def _gen_render_grid(track, endpoint_cells, blockers, H, W):
    grid = np.zeros((H, W), dtype=int)
    blocker_set = set(blockers)
    for (r, c) in track:
        if (r, c) in endpoint_cells:
            grid[r, c] = Color.MAROON
        elif (r, c) in blocker_set:
            grid[r, c] = Color.ORANGE
        else:
            grid[r, c] = Color.TEAL
    return grid


def _gen_try_blockers(track, endpoint_cells, H, W):
    """Place blockers, validate the unique route, and return a rendered grid."""
    n_blockers = random.choice(_GEN_BLOCKER_CHOICES)
    blocker_candidates = _gen_blocker_candidates(track, endpoint_cells, H, W)
    if len(blocker_candidates) < n_blockers:
        return None

    random.shuffle(blocker_candidates)

    for attempt in range(
        min(
            _GEN_MAX_BLOCKER_TRIES,
            len(blocker_candidates) * _GEN_BLOCKER_TRIES_FACTOR,
        )
    ):
        blockers = _gen_choose_blockers(n_blockers, blocker_candidates, attempt)
        if blockers is None:
            continue

        forbidden = _gen_forbidden(blockers)
        if any(ep in forbidden for ep in endpoint_cells):
            continue

        if not _gen_validate_blocked_path(track, endpoint_cells, forbidden):
            continue

        grid = _gen_render_grid(track, endpoint_cells, blockers, H, W)
        try:
            out = main(grid)
            if out is None:
                continue
        except (AssertionError, Exception):
            continue
        return grid

    return None


def _gen_attempt():
    """Single attempt to generate a valid input grid."""
    layout = _gen_choose_layout()
    if layout is None:
        return None
    H, W, bar_rows, bar_cols = layout

    track = _gen_build_lattice(bar_rows, bar_cols)
    if track is None:
        return None

    border_candidates = _gen_border_candidates(track, H, W, bar_rows, bar_cols)
    best_pair = _gen_choose_endpoints(track, border_candidates)
    if best_pair is None:
        return None

    endpoint_cells = _gen_add_and_validate_endpoints(track, best_pair, H, W)
    if endpoint_cells is None:
        return None

    if len(track) < _GEN_TRACK_SIZE_RANGE[0] or len(track) > _GEN_TRACK_SIZE_RANGE[1]:
        return None

    return _gen_try_blockers(track, endpoint_cells, H, W)


def generate_input() -> np.ndarray:
    """Generate a valid input grid for the maze shortest-path task."""
    for _ in range(_GEN_MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_MAX_ATTEMPTS} attempts")
