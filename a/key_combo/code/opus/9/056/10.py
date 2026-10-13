import heapq
import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
PINK = 6
CYAN = 8
COLORS = (RED, GREEN)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SENTINEL_BEND = (-1, -1)
GEN_ATTEMPTS = 400


def _neighbors4(r, c):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _component_coords(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_key(coords):
    return tuple(map(int, coords.min(axis=0)))


def _extract_control_info(grid):
    pts = np.argwhere(grid == PINK)
    assert pts.shape[0] == 2, "expected exactly two control markers"
    (r0, c0), (r1, c1) = [tuple(map(int, p)) for p in pts]
    h, w = grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    if c0 == c1:
        phase = min(r0, r1)
        period = abs(r0 - r1)
        assert period >= 1
        for r in range(h):
            if (r - phase) % period == 0:
                forbidden[r, :] = True
        kind = "rows"
    else:
        assert r0 == r1, "control markers must share a row or a column"
        phase = min(c0, c1)
        period = abs(c0 - c1)
        assert period >= 1
        for c in range(w):
            if (c - phase) % period == 0:
                forbidden[:, c] = True
        kind = "cols"
    return {"kind": kind, "period": period, "forbidden": forbidden}


def _step_state(state, ndir):
    r, c, prev_dir, turns, bend_r, bend_c = state
    dr, dc = DIRS[ndir]
    nr, nc = r + dr, c + dc
    if prev_dir == -1:
        nturns = 0
        nbend_r, nbend_c = bend_r, bend_c
    elif ndir == prev_dir:
        nturns = turns
        nbend_r, nbend_c = bend_r, bend_c
    else:
        nturns = turns + 1
        if nturns > 3:
            return None
        if (bend_r, bend_c) == SENTINEL_BEND:
            nbend_r, nbend_c = r, c
        else:
            nbend_r, nbend_c = bend_r, bend_c
    return nr, nc, ndir, nturns, nbend_r, nbend_c


def _best_path(grid, color, start_coords, goal_coords, forbidden):
    h, w = grid.shape
    start_set = {tuple(map(int, p)) for p in start_coords}
    goal_set = {tuple(map(int, p)) for p in goal_coords}

    blocked = forbidden.copy()
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val not in (BG, color):
                blocked[r, c] = True
    for r, c in start_set | goal_set:
        blocked[r, c] = False

    best_primary = {}
    best_full = {}
    prev = {}
    heap = []
    start_states = []
    for r, c in sorted(start_set):
        st = (r, c, -1, 0, SENTINEL_BEND[0], SENTINEL_BEND[1])
        start_states.append(st)
        best_primary[st] = (0, SENTINEL_BEND[0], SENTINEL_BEND[1])
        best_full[st] = (0, SENTINEL_BEND[0], SENTINEL_BEND[1], 0)
        heapq.heappush(heap, (best_full[st], st))

    goal_primary = None
    goal_states = []
    while heap:
        full_cost, state = heapq.heappop(heap)
        if best_full.get(state) != full_cost:
            continue
        primary = best_primary[state]
        if goal_primary is not None and primary > goal_primary and primary[0] > goal_primary[0]:
            break

        r, c, _, _, _, _ = state
        if (r, c) in goal_set:
            if goal_primary is None or primary < goal_primary:
                goal_primary = primary
                goal_states = [state]
            elif primary == goal_primary:
                goal_states.append(state)
            continue

        for ndir in range(4):
            nstate = _step_state(state, ndir)
            if nstate is None:
                continue
            nr, nc, _, _, bend_r, bend_c = nstate
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc]:
                continue
            nprimary = (primary[0] + 1, bend_r, bend_c)
            nfull = (primary[0] + 1, bend_r, bend_c, nstate[3])
            old_primary = best_primary.get(nstate)
            if old_primary is None or nprimary < old_primary:
                best_primary[nstate] = nprimary
                best_full[nstate] = nfull
                prev[nstate] = state
                heapq.heappush(heap, (nfull, nstate))
            elif nprimary == old_primary and nfull < best_full[nstate]:
                best_full[nstate] = nfull
                prev[nstate] = state
                heapq.heappush(heap, (nfull, nstate))

    assert goal_primary is not None, "expected a legal path"

    counts = {st: 1 for st in start_states}
    sorted_states = sorted(best_primary, key=lambda st: (best_primary[st], best_full[st], st))
    for state in sorted_states:
        cur_count = counts.get(state, 0)
        if cur_count == 0:
            continue
        r, c, _, _, _, _ = state
        if (r, c) in goal_set:
            continue
        primary = best_primary[state]
        for ndir in range(4):
            nstate = _step_state(state, ndir)
            if nstate is None or nstate not in best_primary:
                continue
            nr, nc = nstate[0], nstate[1]
            if not (0 <= nr < h and 0 <= nc < w) or blocked[nr, nc]:
                continue
            if best_primary[nstate] == (primary[0] + 1, nstate[4], nstate[5]):
                counts[nstate] = min(2, counts.get(nstate, 0) + cur_count)

    goal_states = [st for st in goal_states if best_primary[st] == goal_primary]
    goal_state = min(goal_states, key=lambda st: (best_full[st], st))
    path_count = 0
    for st in goal_states:
        path_count = min(2, path_count + counts.get(st, 0))

    path = []
    cur = goal_state
    while True:
        path.append((cur[0], cur[1]))
        if cur not in prev:
            break
        cur = prev[cur]
    path.reverse()

    return {
        "path": path,
        "steps": goal_primary[0],
        "first_bend": (goal_primary[1], goal_primary[2]),
        "turns": goal_state[3],
        "path_count": path_count,
    }


def _ordered_components(grid, color):
    comps = _component_coords(grid, color)
    comps.sort(key=_bbox_key)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each color's ordered components while avoiding periodic forbidden bands inferred from magenta controls.

    Input structure:
    - The grid uses black as background, with red and green foreground components that each appear in 2-3 separate 4-connected pieces.
    - Exactly two magenta control pixels lie on the left border or top border; their spacing defines a periodic family of forbidden full rows or full columns.
    - Cyan pixels are fixed obstacle walls, and all other non-background pixels are blockers for paths of the opposite color.

    Rule:
    - Read the two magenta controls: same-column controls create forbidden horizontal rows, same-row controls create forbidden vertical columns, repeating with the controls' spacing as the period.
    - For red and green separately, sort that color's 4-connected components by the row-major order of each component's top-left bbox corner.
    - Connect each consecutive pair by the shortest legal orthogonal path that uses at most 3 turns, may pass only through background or the same color, and never enters forbidden rows/columns, cyan walls, or the other foreground color.
    - When multiple legal paths have the same length, choose the one whose first bend point has the smaller row; if still tied, choose the smaller column, then paint the chosen path with the component color.

    Key insight:
    - The crucial obstacles are not drawn explicitly: the magenta controls encode an invisible periodic wall pattern, so the solver must reconstruct those forbidden rows or columns before routing the same-color chains.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    control = _extract_control_info(g)
    forbidden = control["forbidden"]

    base_components = {color: _ordered_components(g, color) for color in COLORS}
    for color in COLORS:
        comps = base_components[color]
        for comp_a, comp_b in zip(comps, comps[1:]):
            result = _best_path(out, color, comp_a, comp_b, forbidden)
            for r, c in result["path"]:
                if int(out[r, c]) == BG:
                    out[r, c] = color
    return out


def _band_rows(h, period, phase):
    forbidden_rows = [r for r in range(h) if (r - phase) % period == 0]
    bands = []
    for a, b in zip(forbidden_rows, forbidden_rows[1:]):
        if b - a >= 2:
            bands.append((a + 1, b - 1))
    return forbidden_rows, bands


def _rows_for_sequence(seq):
    if seq == ["clear"]:
        return [1, 0]
    if seq == ["detour2"]:
        return [0, 0]
    if seq == ["detour3"]:
        return [0, 0]
    if seq == ["clear", "detour2"] or seq == ["clear", "detour3"]:
        return [1, 0, 0]
    if seq == ["detour2", "clear"] or seq == ["detour3", "clear"]:
        return [0, 0, 1]
    if seq == ["clear", "clear"]:
        return [1, 0, 1]
    if seq == ["detour2", "detour2"]:
        return [0, 0, 0]
    raise AssertionError(f"unsupported template sequence: {seq}")


def _global_row(band, local_row, mirror):
    top, bottom = band
    if not mirror:
        return top + local_row
    return bottom - local_row


def _paint_hseg(grid, row, col, length, color):
    grid[row, col : col + length] = color
    return np.array([(row, c) for c in range(col, col + length)], dtype=int)


def _place_obstacle_segment(grid, row, c0, c1):
    if c0 <= c1:
        grid[row, c0 : c1 + 1] = CYAN


def _build_chain(grid, band, color, seq, mirror):
    h, w = grid.shape
    band_h = band[1] - band[0] + 1
    assert band_h >= 2
    comp_rows = _rows_for_sequence(seq)
    lengths = [random.randint(2, 3) for _ in comp_rows]
    cols = []
    cur_col = random.randint(1, 3)
    for i, templ in enumerate(seq):
        cols.append(cur_col)
        min_gap = 3 if templ == "clear" else 4
        max_gap = 6 if templ == "clear" else 8
        cur_col += lengths[i] + random.randint(min_gap, max_gap)
    cols.append(cur_col)
    if cols[-1] + lengths[-1] > w - 2:
        return None

    comp_coords = []
    for row_local, col, length in zip(comp_rows, cols, lengths):
        row = _global_row(band, row_local, mirror)
        if np.any(grid[row, col : col + length] != BG):
            return None
        comp_coords.append(_paint_hseg(grid, row, col, length, color))

    for idx, templ in enumerate(seq):
        left_row = _global_row(band, comp_rows[idx], mirror)
        right_row = _global_row(band, comp_rows[idx + 1], mirror)
        left_end = cols[idx] + lengths[idx] - 1
        right_start = cols[idx + 1]
        if templ == "clear":
            continue
        if left_row != right_row:
            return None
        aux_local = comp_rows[idx] + 1
        if aux_local >= band_h:
            return None
        aux_row = _global_row(band, aux_local, mirror)
        if templ == "detour2":
            if right_start - left_end < 3:
                return None
            if np.any(grid[left_row, left_end + 1 : right_start] != BG):
                return None
            _place_obstacle_segment(grid, left_row, left_end + 1, right_start - 1)
        elif templ == "detour3":
            if right_start - left_end < 5:
                return None
            cut = random.randint(left_end + 2, right_start - 2)
            if np.any(grid[left_row, left_end + 1 : cut + 1] != BG):
                return None
            if np.any(grid[aux_row, cut + 2 : right_start] != BG):
                return None
            _place_obstacle_segment(grid, left_row, left_end + 1, cut)
            _place_obstacle_segment(grid, aux_row, cut + 2, right_start - 1)
        else:
            return None

    return comp_coords


def _analysis(grid):
    g = grid.copy()
    forbidden = _extract_control_info(g)["forbidden"]
    results = []
    added_by_color = {}
    for color in COLORS:
        comps = _ordered_components(g, color)
        assert 2 <= len(comps) <= 3
        added = set()
        for comp_a, comp_b in zip(comps, comps[1:]):
            actual = _best_path(g, color, comp_a, comp_b, forbidden)
            no_forbidden = _best_path(g, color, comp_a, comp_b, np.zeros_like(forbidden, dtype=bool))
            path_set = set(actual["path"])
            add_set = {cell for cell in path_set if int(g[cell]) == BG}
            results.append(
                {
                    "color": color,
                    "actual": actual,
                    "no_forbidden": no_forbidden,
                    "added": add_set,
                }
            )
            for r, c in actual["path"]:
                if int(g[r, c]) == BG:
                    g[r, c] = color
                    added.add((r, c))
        added_by_color[color] = added
    return results, added_by_color


def _validate_candidate(grid):
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.count_nonzero(grid == PINK) != 2:
        return False
    if np.any((grid == RED) & (grid == PINK)):
        return False

    info = _extract_control_info(grid)
    forbidden = info["forbidden"]
    for color in COLORS:
        comps = _ordered_components(grid, color)
        if not (2 <= len(comps) <= 3):
            return False
        for comp in comps:
            if comp.shape[0] < 2 or comp.shape[0] > 5:
                return False
            for r, c in comp:
                if forbidden[r, c]:
                    return False

    try:
        analysis, added_by_color = _analysis(grid)
    except AssertionError:
        return False

    if not analysis:
        return False
    if any(item["actual"]["path_count"] != 1 for item in analysis):
        return False
    if not any(item["actual"]["turns"] <= 1 for item in analysis):
        return False
    if not any(item["actual"]["turns"] >= 2 for item in analysis):
        return False

    forbidden_matters = False
    for item in analysis:
        a = item["actual"]
        b = item["no_forbidden"]
        if a["steps"] != b["steps"] or a["first_bend"] != b["first_bend"] or a["path"] != b["path"] or b["path_count"] != 1:
            forbidden_matters = True
            break
    if not forbidden_matters:
        return False

    if added_by_color[RED] & added_by_color[GREEN]:
        return False
    total_added = len(added_by_color[RED] | added_by_color[GREEN])
    if total_added < 6:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if max(out.shape) > 30:
        return False
    return True


def _base_layout():
    p = random.choices([3, 4, 5], weights=[2, 4, 4])[0]
    h = random.randint(max(16, 2 + 2 * p), min(22, 30))
    w = random.randint(16, 22)
    phase = 1

    grid = np.full((h, w), BG, dtype=int)
    forbidden_rows, bands = _band_rows(h, p, phase)
    if len(bands) < 2:
        return None

    grid[phase, 0] = PINK
    grid[phase + p, 0] = PINK

    band_choices = random.sample(bands, 2)
    random.shuffle(band_choices)
    mirrors = [bool(random.getrandbits(1)), bool(random.getrandbits(1))]

    layouts = [
        {RED: ["detour3"], GREEN: ["clear", "detour2"]},
        {RED: ["detour2", "clear"], GREEN: ["clear"]},
        {RED: ["detour2"], GREEN: ["clear", "clear"]},
        {RED: ["clear", "detour3"], GREEN: ["detour2"]},
    ]
    seqs = random.choice(layouts)

    for color, band, mirror in [(RED, band_choices[0], mirrors[0]), (GREEN, band_choices[1], mirrors[1])]:
        if _build_chain(grid, band, color, seqs[color], mirror) is None:
            return None

    noise_rows = [r for r in range(h) if r not in forbidden_rows]
    for _ in range(random.randint(0, 3)):
        rr = random.choice(noise_rows)
        cc = random.randint(w // 2, w - 2)
        if int(grid[rr, cc]) == BG:
            grid[rr, cc] = CYAN

    return grid


def _maybe_transform(grid):
    out = grid
    if random.random() < 0.5:
        out = out.T.copy()
        if random.random() < 0.5:
            out = np.fliplr(out)
    else:
        if random.random() < 0.5:
            out = np.flipud(out)
    return out.copy()


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _base_layout()
        if grid is None:
            continue
        grid = _maybe_transform(grid)
        if _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
