import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
DOT_COLORS = (4, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 400
TAIL_PACK_RESERVE = 4


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _start_direction(r: int, c: int, h: int, w: int) -> tuple[int, int]:
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    if c == w - 1:
        return 0, -1
    raise AssertionError("red start must lie on the border")


def _iter_line(a: tuple[int, int], b: tuple[int, int]):
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        for cc in range(ac, bc + step, step):
            yield ar, cc
    elif ac == bc:
        step = 1 if br >= ar else -1
        for rr in range(ar, br + step, step):
            yield rr, ac
    else:
        raise AssertionError("expected an axis-aligned segment")


def _bar_components(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    components: list[dict] = []
    cell_to_component: dict[tuple[int, int], dict] = {}

    for color in (BLUE, GREEN):
        seen = np.zeros((h, w), dtype=bool)
        for r, c in np.argwhere(g == color):
            rr = int(r)
            cc = int(c)
            if seen[rr, cc]:
                continue

            stack = [(rr, cc)]
            seen[rr, cc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cd = stack.pop()
                coords.append((cr, cd))
                for nr, nc in ((cr - 1, cd), (cr + 1, cd), (cr, cd - 1), (cr, cd + 1)):
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comp = {
                "color": color,
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            }
            components.append(comp)
            for pos in coords:
                cell_to_component[pos] = comp

    return components, cell_to_component


def _mark_segment(
    trail: set[tuple[int, int]],
    source: np.ndarray,
    dot_palette: set[int],
    a: tuple[int, int],
    b: tuple[int, int],
) -> None:
    for r, c in _iter_line(a, b):
        value = int(source[r, c])
        if value in (BLUE, GREEN) or value in dot_palette:
            continue
        trail.add((r, c))


def _trace_info(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    dot_palette = set(map(int, np.unique(g))) - {bg, RED, BLUE, GREEN}

    starts = np.argwhere(g == RED)
    assert starts.shape == (1, 2), "expected exactly one red start pixel"
    sr, sc = map(int, starts[0])
    dr, dc = _start_direction(sr, sc, h, w)

    components, component_at = _bar_components(g)
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        if dr != 0:
            assert r0 == r1, "bars must be horizontal for vertical travel"
        else:
            assert c0 == c1, "bars must be vertical for horizontal travel"

    trail: set[tuple[int, int]] = {(sr, sc)}
    collected_positions: list[tuple[int, int]] = []
    collected_colors: list[int] = []

    r, c = sr, sc
    while True:
        rr, cc = r + dr, c + dc
        hit = None
        while _in_bounds(g.shape, rr, cc):
            value = int(g[rr, cc])
            if value in (BLUE, GREEN):
                hit = (rr, cc)
                break
            if value in dot_palette:
                collected_positions.append((rr, cc))
                collected_colors.append(value)
            else:
                trail.add((rr, cc))
            r, c = rr, cc
            rr += dr
            cc += dc

        if hit is None:
            boundary_cell = (r, c)
            break

        hr, hc = hit
        comp = component_at[hit]
        normal = (-dc, dr) if int(comp["color"]) == BLUE else (dc, -dr)
        r0, c0, r1, c1 = comp["bbox"]
        if normal[0] < 0:
            side = (r0 - 1, hc)
        elif normal[0] > 0:
            side = (r1 + 1, hc)
        elif normal[1] < 0:
            side = (hr, c0 - 1)
        else:
            side = (hr, c1 + 1)

        assert _in_bounds(g.shape, side[0], side[1]), "detour must stay inside the grid"
        assert int(g[side]) not in (BLUE, GREEN), "detour landing cell must be free"

        turn_end = (side[0] - dr, side[1] - dc)
        _mark_segment(trail, g, dot_palette, (r, c), turn_end)
        _mark_segment(trail, g, dot_palette, side, side)
        r, c = side

    return {
        "trail": trail,
        "collected_positions": collected_positions,
        "collected_colors": collected_colors,
        "boundary_cell": boundary_cell,
        "direction": (dr, dc),
        "background": bg,
        "dot_palette": dot_palette,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Let the red ray weave around blue/green shifters, collect colored dots, and pack them at the far edge.

    Input structure:
    - The grid has a black background, one red start pixel on the border, several one-cell-thick blue/green bars perpendicular to the ray, and several uniquely colored single-cell dots.
    - Blue and green bars act as lane shifters rather than solid blockers.

    Rule:
    - Follow the red ray inward from the border; blue bars force a detour on the left side relative to travel and green bars force a detour on the right side, after which the ray resumes the same forward direction on a shifted lane.
    - When the forward ray hits a colored dot, remove that dot from its original cell and record its color in hit order.
    - Keep the bars in place, paint the traversed non-dot background cells red, and leave untouched dots unchanged.
    - On the final lane, place the collected dot colors tightly from the opposite boundary back inward, with the earliest hit closest to the far edge.

    Key insight:
    - Each bar permanently shifts the later lane, so the dots that get collected and the packed color order are determined by the whole detour history.
    """
    g = np.asarray(grid, dtype=int)
    info = _trace_info(g)
    out = g.copy()

    bg = int(info["background"])
    for r, c in info["collected_positions"]:
        out[r, c] = bg
    for r, c in info["trail"]:
        out[r, c] = RED

    dr, dc = info["direction"]
    br, bc = info["boundary_cell"]
    for index, color in enumerate(info["collected_colors"]):
        rr = br - index * dr
        cc = bc - index * dc
        assert _in_bounds(out.shape, rr, cc), "packed dots must fit on the final lane"
        out[rr, cc] = int(color)

    return out


def _sample_hit_rows(n_hits: int) -> tuple[int, list[int]] | None:
    row = random.randint(2, 4)
    rows = [row]
    for _ in range(n_hits - 1):
        row += random.randint(3, 5)
        rows.append(row)
    height = rows[-1] + random.randint(4, 7) + 1
    if height > 30:
        return None
    return height, rows


def _sample_lanes(width: int, n_hits: int) -> list[int] | None:
    for _ in range(120):
        lane = random.randint(3, width - 4)
        lanes = [lane]
        shifts: list[int] = []
        ok = True
        for idx in range(n_hits):
            options = [lane + delta for delta in (-3, -2, -1, 1, 2, 3) if 1 <= lane + delta <= width - 2]
            if idx == n_hits - 1 and shifts:
                if all(delta > 0 for delta in shifts):
                    negative = [value for value in options if value < lane]
                    if negative:
                        options = negative
                elif all(delta < 0 for delta in shifts):
                    positive = [value for value in options if value > lane]
                    if positive:
                        options = positive
            if not options:
                ok = False
                break
            new_lane = random.choice(options)
            shifts.append(new_lane - lane)
            lanes.append(new_lane)
            lane = new_lane
        if ok and any(delta > 0 for delta in shifts) and any(delta < 0 for delta in shifts):
            return lanes
    return None


def _base_path(height: int, lanes: list[int], hit_rows: list[int]) -> tuple[set[tuple[int, int]], list[list[tuple[int, int]]]]:
    path = {(0, lanes[0])}
    forward_runs: list[list[tuple[int, int]]] = []
    current_row = 0
    current_lane = lanes[0]

    for idx, hit_row in enumerate(hit_rows):
        run = [(r, current_lane) for r in range(current_row + 1, hit_row)]
        forward_runs.append(run)
        path.update(run)

        new_lane = lanes[idx + 1]
        step = 1 if new_lane >= current_lane else -1
        for col in range(current_lane, new_lane + step, step):
            path.add((hit_row - 1, col))
        path.add((hit_row, new_lane))

        current_row = hit_row
        current_lane = new_lane

    final_run = [(r, current_lane) for r in range(current_row + 1, height)]
    forward_runs.append(final_run)
    path.update(final_run)
    return path, forward_runs


def _trim_run(run: list[tuple[int, int]], reserve_tail: int = 0) -> list[tuple[int, int]]:
    cells = list(run)
    if reserve_tail and len(cells) > reserve_tail:
        cells = cells[:-reserve_tail]
    if len(cells) >= 4:
        return cells[1:-1]
    return cells


def _pick_spaced_positions(
    candidates: list[tuple[int, int]],
    count: int,
    taken: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    chosen: list[tuple[int, int]] = []
    for pos in candidates:
        if pos in taken:
            continue
        if any(abs(pos[0] - rr) <= 1 and abs(pos[1] - cc) <= 1 for rr, cc in chosen):
            continue
        if any(abs(pos[0] - rr) <= 1 and abs(pos[1] - cc) <= 1 for rr, cc in taken):
            continue
        chosen.append(pos)
        if len(chosen) == count:
            break
    return chosen


def _gen_attempt() -> np.ndarray | None:
    n_hits = random.randint(2, 5)
    sampled = _sample_hit_rows(n_hits)
    if sampled is None:
        return None
    height, hit_rows = sampled

    width = random.randint(14, 24)
    lanes = _sample_lanes(width, n_hits)
    if lanes is None:
        return None

    base = np.full((height, width), BG, dtype=int)
    base[0, lanes[0]] = RED

    sign_colors: list[int] = []
    for idx, hit_row in enumerate(hit_rows):
        current_lane = lanes[idx]
        new_lane = lanes[idx + 1]
        if new_lane > current_lane:
            color = BLUE
            end = new_lane - 1
            start_hi = current_lane if new_lane - current_lane >= 2 else current_lane - 1
            start = random.randint(max(0, current_lane - 2), start_hi)
        else:
            color = GREEN
            start = new_lane + 1
            end_lo = current_lane if current_lane - new_lane >= 2 else current_lane + 1
            end = random.randint(end_lo, min(width - 1, current_lane + 2))
        sign_colors.append(color)
        base[hit_row, start : end + 1] = color

    path_cells, forward_runs = _base_path(height, lanes, hit_rows)
    usable_segments: list[tuple[int, list[tuple[int, int]]]] = []
    for idx, run in enumerate(forward_runs):
        reserve = TAIL_PACK_RESERVE if idx == len(forward_runs) - 1 else 0
        trimmed = _trim_run(run, reserve_tail=reserve)
        if trimmed:
            usable_segments.append((idx, trimmed))
    if len(usable_segments) < 2:
        return None

    hit_count = random.randint(2, min(4, len(usable_segments)))
    chosen_segments = sorted(random.sample(usable_segments, hit_count), key=lambda item: item[0])
    hit_positions = [random.choice(cells) for _, cells in chosen_segments]

    miss_count = random.randint(1, min(2, len(DOT_COLORS) - hit_count))
    used = set(hit_positions)

    lane_candidates = sorted({(r, c) for c in set(lanes) for r in range(1, height - 1) if base[r, c] == BG and (r, c) not in path_cells})
    random.shuffle(lane_candidates)
    miss_positions = _pick_spaced_positions(lane_candidates, miss_count, used)
    if len(miss_positions) < miss_count:
        free_candidates = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1) if base[r, c] == BG and (r, c) not in path_cells]
        random.shuffle(free_candidates)
        taken = used | set(miss_positions)
        miss_positions.extend(_pick_spaced_positions(free_candidates, miss_count - len(miss_positions), taken))
    if len(miss_positions) < miss_count:
        return None

    palette = random.sample(DOT_COLORS, hit_count + miss_count)
    for color, pos in zip(palette[:hit_count], hit_positions):
        base[pos] = color
    for color, pos in zip(palette[hit_count:], miss_positions):
        base[pos] = color

    rotation = random.randrange(4)
    grid = np.rot90(base, rotation).copy()

    try:
        info = _trace_info(grid)
        out = main(grid)
    except Exception:
        return None

    if len(info["collected_colors"]) != hit_count:
        return None
    if len(info["dot_palette"]) != hit_count + miss_count:
        return None
    if sign_colors.count(BLUE) == 0 or sign_colors.count(GREEN) == 0:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
