from __future__ import annotations

from collections import Counter, deque

import numpy as np


TARGET_COLOR = 1
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _corner_bg(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return int(Counter(corners).most_common(1)[0][0])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    cur = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != 0:
                continue
            cur += 1
            q = deque([(r, c)])
            labels[r, c] = cur
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == 0:
                        labels[nr, nc] = cur
                        q.append((nr, nc))
    return labels, cur


def _extract_gears(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _corner_bg(g)

    counts = Counter(map(int, g.flatten()))
    non_bg = [c for c in counts if int(c) != int(bg)]
    assert non_bg, "expected non-background content"
    gear_color = int(max(non_bg, key=lambda c: (int(counts[int(c)]), -int(c))))

    gear_mask = g == int(gear_color)
    labels, nlab = _label_components(gear_mask)
    assert nlab > 0, "expected at least one gear"

    markers_by_label: dict[int, list[tuple[int, int, int]]] = {k: [] for k in range(1, nlab + 1)}
    for r, c in np.argwhere((g != int(bg)) & (g != int(gear_color))):
        rr, cc = int(r), int(c)
        neigh = {
            int(labels[rr + dr, cc + dc])
            for dr, dc in DIR4
            if 0 <= rr + dr < h and 0 <= cc + dc < w and int(labels[rr + dr, cc + dc]) != 0
        }
        assert len(neigh) == 1, "each marker must attach to exactly one gear"
        lab = next(iter(neigh))
        markers_by_label[lab].append((rr, cc, int(g[rr, cc])))

    gears: list[dict] = []
    for lab in range(1, nlab + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(labels == int(lab))]
        markers = markers_by_label[lab]
        assert len(markers) in (1, 2), "each gear must have one marker, except the driver with two"

        all_cells = set(coords)
        all_cells.update((r, c) for r, c, _ in markers)
        rows = [r for r, _ in all_cells]
        cols = [c for _, c in all_cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert (r1 - r0) == (c1 - c0), "gear bbox must be square"

        border = []
        for c in range(c0, c1 + 1):
            if (r0, c) in all_cells:
                border.append((r0, c))
        for r in range(r0 + 1, r1 + 1):
            if (r, c1) in all_cells:
                border.append((r, c1))
        for c in range(c1 - 1, c0 - 1, -1):
            if (r1, c) in all_cells:
                border.append((r1, c))
        for r in range(r1 - 1, r0, -1):
            if (r, c0) in all_cells:
                border.append((r, c0))
        assert border, "expected visible teeth on the outer border"
        pos_to_idx = {pos: i for i, pos in enumerate(border)}
        assert len(pos_to_idx) == len(border), "tooth positions must be unique"

        gears.append(
            {
                "label": lab,
                "bbox": (r0, c0, r1, c1),
                "cycle": border,
                "pos_to_idx": pos_to_idx,
                "markers": markers,
            }
        )

    return bg, gear_color, gears


def _build_graph(gears: list[dict]) -> list[list[int]]:
    graph = [[] for _ in gears]
    for i in range(len(gears)):
        r0a, c0a, r1a, c1a = gears[i]["bbox"]
        for j in range(i + 1, len(gears)):
            r0b, c0b, r1b, c1b = gears[j]["bbox"]
            horiz = (c1a + 1 == c0b or c1b + 1 == c0a) and max(r0a, r0b) <= min(r1a, r1b)
            vert = (r1a + 1 == r0b or r1b + 1 == r0a) and max(c0a, c0b) <= min(c1a, c1b)
            if horiz or vert:
                graph[i].append(j)
                graph[j].append(i)
    return graph


def _rotation_plan(grid: np.ndarray) -> tuple[int, int, list[dict], int, int, list[int]]:
    bg, gear_color, gears = _extract_gears(grid)
    driver_candidates = [i for i, gear in enumerate(gears) if len(gear["markers"]) == 2]
    assert len(driver_candidates) == 1, "expected exactly one driver gear"
    driver = int(driver_candidates[0])

    driver_markers = gears[driver]["markers"]
    target_marks = [m for m in driver_markers if int(m[2]) == int(TARGET_COLOR)]
    current_marks = [m for m in driver_markers if int(m[2]) != int(TARGET_COLOR)]
    assert len(target_marks) == 1 and len(current_marks) == 1, "driver must expose one target hint and one current marker"

    cycle = gears[driver]["cycle"]
    pos_to_idx = gears[driver]["pos_to_idx"]
    cur_pos = (int(current_marks[0][0]), int(current_marks[0][1]))
    tgt_pos = (int(target_marks[0][0]), int(target_marks[0][1]))
    assert cur_pos in pos_to_idx and tgt_pos in pos_to_idx, "driver markers must lie on tooth tips"
    steps = (int(pos_to_idx[tgt_pos]) - int(pos_to_idx[cur_pos])) % len(cycle)
    assert steps > 0, "target hint must differ from the current position"

    graph = _build_graph(gears)
    dist = [-1] * len(gears)
    q = deque([driver])
    dist[driver] = 0
    while q:
        i = q.popleft()
        for j in graph[i]:
            if int(dist[j]) == -1:
                dist[j] = int(dist[i]) + 1
                q.append(j)

    return bg, gear_color, gears, driver, steps, dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate every marker around meshed square gears according to the driver gear's hinted clockwise turn.

    Input structure:
    - The background surrounds several same-colored square gears: hollow square rings with regularly spaced outer teeth.
    - Each gear carries a single colored marker on one tooth tip, except one driver gear that also has a second target-hint marker.
    - Meshed gears sit flush side-to-side or top-to-bottom, with alternating tooth phases showing which gears engage.

    Rule:
    - Identify the unique driver gear: it is the only gear with two edge markers, and the fixed hint color marks the target tooth.
    - Count how many tooth steps the driver's current marker must move clockwise to reach the target tooth.
    - Propagate that same tooth count through the meshing graph: directly meshed gears turn in the opposite direction, gears at even graph distance keep the driver's direction, and disconnected gears do not move.
    - Move each current marker along its own gear's tooth-tip cycle by the required signed number of steps, and remove the driver's target-hint marker.

    Key insight:
    - The gears themselves stay in place; only the marker positions change, and the correct direction for each gear is determined by parity along the meshing chain from the driver.
    """
    g = np.asarray(grid, dtype=int)
    _bg, gear_color, gears, _driver, steps, dist = _rotation_plan(g)

    out = g.copy()
    for gear in gears:
        for r, c, _col in gear["markers"]:
            out[int(r), int(c)] = int(gear_color)

    for i, gear in enumerate(gears):
        if len(gear["markers"]) == 2:
            current = next(m for m in gear["markers"] if int(m[2]) != int(TARGET_COLOR))
        else:
            current = gear["markers"][0]

        cur_pos = (int(current[0]), int(current[1]))
        cur_idx = int(gear["pos_to_idx"][cur_pos])
        delta = 0
        if int(dist[i]) >= 0:
            delta = steps if int(dist[i]) % 2 == 0 else -steps
        new_pos = gear["cycle"][(cur_idx + delta) % len(gear["cycle"])]
        out[int(new_pos[0]), int(new_pos[1])] = int(current[2])

    return out


def _tooth_cycle_local(size_param: int, phase: int) -> list[tuple[int, int]]:
    span = 2 * int(size_param) + 4
    side_positions = list(range(2 + int(phase), span - 2, 2))
    cycle: list[tuple[int, int]] = []
    for c in side_positions:
        cycle.append((0, int(c)))
    for r in side_positions:
        cycle.append((int(r), span - 1))
    for c in reversed(side_positions):
        cycle.append((span - 1, int(c)))
    for r in reversed(side_positions):
        cycle.append((int(r), 0))
    return cycle


def _draw_gear_patch(size_param: int, phase: int, gear_color: int, bg: int, markers: dict[int, int]) -> np.ndarray:
    span = 2 * int(size_param) + 4
    patch = np.full((span, span), int(bg), dtype=int)

    patch[1, 1 : span - 1] = int(gear_color)
    patch[span - 2, 1 : span - 1] = int(gear_color)
    patch[1 : span - 1, 1] = int(gear_color)
    patch[1 : span - 1, span - 2] = int(gear_color)

    cycle = _tooth_cycle_local(size_param, phase)
    for r, c in cycle:
        patch[int(r), int(c)] = int(gear_color)
    for idx, color in markers.items():
        r, c = cycle[int(idx)]
        patch[int(r), int(c)] = int(color)
    return patch


def _grow_connected_cells(rng: np.random.Generator, rows: int, cols: int, count: int) -> set[tuple[int, int]]:
    start = (int(rng.integers(rows)), int(rng.integers(cols)))
    cells = {start}
    while len(cells) < count:
        frontier = []
        for r, c in sorted(cells):
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        assert frontier, "connected growth frontier disappeared"
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _isolated_candidates(rows: int, cols: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in occupied:
                continue
            ok = True
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in occupied:
                    ok = False
                    break
            if ok:
                out.append((r, c))
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    size_param = int(rng.integers(2, 5))
    gear_span = 2 * size_param + 4
    max_slots = 30 // gear_span
    if max_slots < 2:
        return None

    layouts = [(r, c) for r in range(2, max_slots + 1) for c in range(2, max_slots + 1)]
    rows_slots, cols_slots = layouts[int(rng.integers(len(layouts)))]
    total_slots = rows_slots * cols_slots

    max_main = min(5, total_slots)
    if max_main < 2:
        return None
    main_count = int(rng.integers(2, max_main + 1))
    main_cells = _grow_connected_cells(rng, rows_slots, cols_slots, main_count)

    occupied = set(main_cells)
    iso_cells = _isolated_candidates(rows_slots, cols_slots, occupied)
    if iso_cells and bool(rng.integers(2)):
        occupied.add(iso_cells[int(rng.integers(len(iso_cells)))])

    colors = list(range(10))
    pool = [c for c in colors if int(c) != int(TARGET_COLOR)]
    bg, gear_color = rng.choice(pool, size=2, replace=False).tolist()

    marker_pool = [c for c in colors if int(c) not in {int(bg), int(gear_color), int(TARGET_COLOR)}]
    if len(marker_pool) < len(occupied):
        return None
    marker_colors = rng.choice(marker_pool, size=len(occupied), replace=False).tolist()

    main_list = sorted(main_cells)
    driver_cell = main_list[int(rng.integers(len(main_list)))]

    rows_used = [r for r, _ in occupied]
    cols_used = [c for _, c in occupied]
    span_h = (max(rows_used) - min(rows_used) + 1) * gear_span
    span_w = (max(cols_used) - min(cols_used) + 1) * gear_span
    if span_h + 2 > 30 or span_w + 2 > 30:
        return None

    grid_h = int(rng.integers(span_h + 2, min(31, span_h + 7)))
    grid_w = int(rng.integers(span_w + 2, min(31, span_w + 7)))
    free_h = grid_h - span_h
    free_w = grid_w - span_w
    if free_h < 2 or free_w < 2:
        return None
    top = int(rng.integers(1, free_h))
    left = int(rng.integers(1, free_w))

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    cycle_len = 4 * size_param
    step_count = int(rng.integers(1, cycle_len))

    marker_iter = iter(marker_colors)
    main_set = set(main_cells)
    cell_to_marker_color = {cell: int(next(marker_iter)) for cell in sorted(occupied)}

    for cell in sorted(occupied):
        rr, cc = cell
        phase = (rr + cc) % 2
        cur_idx = int(rng.integers(cycle_len))
        markers = {}
        if cell == driver_cell:
            markers[cur_idx] = cell_to_marker_color[cell]
            markers[(cur_idx + step_count) % cycle_len] = int(TARGET_COLOR)
        else:
            markers[cur_idx] = cell_to_marker_color[cell]

        patch = _draw_gear_patch(size_param, phase, int(gear_color), int(bg), markers)
        gr = top + (rr - min(rows_used)) * gear_span
        gc = left + (cc - min(cols_used)) * gear_span
        grid[gr : gr + gear_span, gc : gc + gear_span] = patch

    try:
        _bg2, gear2, gears2, driver2, steps2, dist2 = _rotation_plan(grid)
    except AssertionError:
        return None

    if int(_bg2) != int(bg) or int(gear2) != int(gear_color):
        return None
    if int(steps2) != int(step_count):
        return None
    if int(driver2) < 0 or len(gears2) != len(occupied):
        return None
    if not any(int(d) > 0 for d in dist2 if int(d) >= 0):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gear puzzle input")
