from __future__ import annotations

from collections import deque
import random

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

TURN_RIGHT = PINK
TURN_LEFT = BLUE
TURN_COLORS = {TURN_RIGHT, TURN_LEFT}
REGION_COLORS = [BLUE, RED, YELLOW, PINK, ORANGE, TEAL, MAROON]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
RECT_SIZES = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (2, 5), (5, 2)]
MAX_ATTEMPTS = 800


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _turn(direction: tuple[int, int], control_color: int) -> tuple[int, int]:
    dr, dc = direction
    if control_color == TURN_RIGHT:
        return dc, -dr
    assert control_color == TURN_LEFT
    return -dc, dr


def _same_color_components(grid: np.ndarray, colors: set[int] | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c]:
                continue
            if colors is not None and color not in colors:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(cells, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _extract_scene(grid: np.ndarray) -> tuple[tuple[int, int], dict[tuple[int, int], int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    starts = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    assert len(starts) == 1, "expected exactly one green start pixel"
    start = starts[0]
    assert start[0] in {0, g.shape[0] - 1} or start[1] in {0, g.shape[1] - 1}, "start must be on the boundary"

    controls: dict[tuple[int, int], int] = {}
    regions: list[dict] = []
    for comp in _same_color_components(g):
        color = int(comp["color"])
        area = int(comp["area"])
        if color == BLACK or color == GREY:
            continue
        if color == GREEN:
            assert area == 1 and tuple(map(int, comp["coords"][0])) == start, "green may only be the start in input"
            continue
        if color in TURN_COLORS and area == 1:
            cell = tuple(map(int, comp["coords"][0]))
            controls[cell] = color
            continue
        regions.append(comp)

    assert len(regions) >= 3, "expected at least three colored regions"
    return start, controls, regions


def _initial_direction(grid: np.ndarray, start: tuple[int, int], controls: dict[tuple[int, int], int]) -> tuple[int, int]:
    h, w = grid.shape
    sr, sc = start
    candidates = []
    for dr, dc in DIRS:
        r = sr + dr
        c = sc + dc
        while 0 <= r < h and 0 <= c < w:
            value = int(grid[r, c])
            if value != BLACK:
                if (r, c) in controls:
                    candidates.append((dr, dc))
                break
            r += dr
            c += dc
    assert len(candidates) == 1, "expected exactly one visible control from the start"
    return candidates[0]


def _trace_path(
    grid: np.ndarray,
    start: tuple[int, int],
    controls: dict[tuple[int, int], int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = np.asarray(grid, dtype=int).copy()
    path_cells: list[tuple[int, int]] = []
    r, c = start
    dr, dc = _initial_direction(out, start, controls)
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "path must terminate"
        seen_states.add(state)

        nr = r + dr
        nc = c + dc
        if not in_bounds(out.shape, nr, nc):
            return out, path_cells

        value = int(grid[nr, nc])
        if value == BLACK:
            out[nr, nc] = GREEN
            path_cells.append((nr, nc))
            r, c = nr, nc
            continue

        if (nr, nc) in controls:
            dr, dc = _turn((dr, dc), controls[(nr, nc)])
            continue

        return out, path_cells


def _region_lookup(shape: tuple[int, int], regions: list[dict]) -> np.ndarray:
    lookup = np.full(shape, -1, dtype=int)
    for idx, comp in enumerate(regions):
        coords = comp["coords"]
        lookup[coords[:, 0], coords[:, 1]] = idx
    return lookup


def _touched_region_order(
    shape: tuple[int, int],
    path_cells: list[tuple[int, int]],
    regions: list[dict],
) -> list[int]:
    lookup = _region_lookup(shape, regions)
    seen: set[int] = set()
    order: list[int] = []
    for r, c in path_cells:
        for nr, nc in neighbors4(r, c):
            if not in_bounds(shape, nr, nc):
                continue
            idx = int(lookup[nr, nc])
            if idx >= 0 and idx not in seen:
                seen.add(idx)
                order.append(idx)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the corridor path and rotate colors only among the regions first touched along that path.

    Input structure:
    - A black background contains one green start pixel on the boundary, several one-pixel blue/pink turn controls in the corridors, a connected grey divider structure, and 4-connected monochrome colored regions.
    - The colored regions are separated from one another by grey walls and black corridors; region colors are distinct and never use green.
    - Any one-pixel blue or pink component is treated as a turn control, while larger colored components are treated as regions.

    Rule:
    - From the green boundary start, look along the four cardinal directions and choose the unique direction whose first non-black sighted cell is a turn control.
    - Move step by step through black cells, painting them green; when the next cell is pink turn right, when it is blue turn left, and when the next cell is any other non-black cell or the move would leave the grid, stop without overwriting that cell.
    - Read the newly painted green path cells in creation order; whenever a cell first has a 4-neighbor belonging to an unseen colored region, append that region to the touched sequence.
    - Recolor only the touched regions by a one-step cyclic backward shift along that sequence, leaving untouched regions, controls, grey walls, and the green path unchanged.

    Key insight:
    - The turn-control path does not merely draw a trail: its geometry determines both which regions participate in the recoloring and the exact cyclic order of their color transfer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    start, controls, regions = _extract_scene(g)
    traced, path_cells = _trace_path(g, start, controls)
    order = _touched_region_order(g.shape, path_cells, regions)
    assert len(order) >= 1, "path must touch at least one region"

    out = traced.copy()
    for i, dst_idx in enumerate(order):
        src_idx = order[(i - 1) % len(order)]
        dst = regions[dst_idx]["coords"]
        out[dst[:, 0], dst[:, 1]] = int(regions[src_idx]["color"])
    return out


def _boundary_start_candidates(h: int, w: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    starts = []
    for c in range(1, w - 1):
        starts.append(((0, c), (1, 0)))
        starts.append(((h - 1, c), (-1, 0)))
    for r in range(1, h - 1):
        starts.append(((r, 0), (0, 1)))
        starts.append(((r, w - 1), (0, -1)))
    return starts


def _ray_cells(pos: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = direction
    return [(r + dr * k, c + dc * k) for k in range(1, length + 1)]


def _edge_distance(h: int, w: int, pos: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = pos
    dr, dc = direction
    if dr < 0:
        return r
    if dr > 0:
        return h - 1 - r
    if dc < 0:
        return c
    return w - 1 - c


def _build_path_geometry(
    h: int,
    w: int,
    start: tuple[int, int],
    initial_direction: tuple[int, int],
    turn_colors: list[int],
    end_mode: str,
) -> dict | None:
    seg_choices = [2, 3, 4]
    final_choices = [2, 3, 4, 5]

    def search(
        turn_idx: int,
        pos: tuple[int, int],
        direction: tuple[int, int],
        path_cells: list[tuple[int, int]],
        controls: list[tuple[tuple[int, int], int]],
        occupied: set[tuple[int, int]],
    ) -> dict | None:
        if turn_idx == len(turn_colors):
            random.shuffle(final_choices)
            if end_mode == "edge":
                exact = _edge_distance(h, w, pos, direction)
                if exact not in final_choices:
                    return None
                cells = _ray_cells(pos, direction, exact)
                if not all(in_bounds((h, w), r, c) for r, c in cells):
                    return None
                if any(cell in occupied for cell in cells):
                    return None
                return {
                    "start": start,
                    "initial_direction": initial_direction,
                    "path_cells": path_cells + cells,
                    "controls": controls,
                    "end_mode": end_mode,
                    "stop_cell": None,
                }

            for seg_len in final_choices:
                cells = _ray_cells(pos, direction, seg_len)
                stop_cell = _ray_cells(pos, direction, seg_len + 1)[-1]
                if not all(in_bounds((h, w), r, c) for r, c in cells):
                    continue
                if not in_bounds((h, w), stop_cell[0], stop_cell[1]):
                    continue
                if any(cell in occupied for cell in cells):
                    continue
                if stop_cell in occupied:
                    continue
                return {
                    "start": start,
                    "initial_direction": initial_direction,
                    "path_cells": path_cells + cells,
                    "controls": controls,
                    "end_mode": end_mode,
                    "stop_cell": stop_cell,
                }
            return None

        lengths = seg_choices[:]
        random.shuffle(lengths)
        control_color = turn_colors[turn_idx]
        for seg_len in lengths:
            cells = _ray_cells(pos, direction, seg_len)
            control = _ray_cells(pos, direction, seg_len + 1)[-1]
            if not all(in_bounds((h, w), r, c) for r, c in cells):
                continue
            if not in_bounds((h, w), control[0], control[1]):
                continue
            if any(cell in occupied for cell in cells):
                continue
            if control in occupied:
                continue

            next_dir = _turn(direction, control_color)
            result = search(
                turn_idx + 1,
                cells[-1],
                next_dir,
                path_cells + cells,
                controls + [(control, control_color)],
                occupied | set(cells) | {control},
            )
            if result is not None:
                return result
        return None

    return search(0, start, initial_direction, [], [], {start})


def _path_segments(start: tuple[int, int], path_cells: list[tuple[int, int]]) -> list[dict]:
    if not path_cells:
        return []
    anchors = [start] + path_cells[:-1]
    dirs = [
        (path_cells[i][0] - anchors[i][0], path_cells[i][1] - anchors[i][1])
        for i in range(len(path_cells))
    ]
    segments = []
    seg_start = 0
    for i in range(1, len(path_cells)):
        if dirs[i] != dirs[i - 1]:
            segments.append(
                {
                    "direction": dirs[i - 1],
                    "start_idx": seg_start,
                    "end_idx": i - 1,
                    "cells": path_cells[seg_start:i],
                }
            )
            seg_start = i
    segments.append(
        {
            "direction": dirs[-1],
            "start_idx": seg_start,
            "end_idx": len(path_cells) - 1,
            "cells": path_cells[seg_start:],
        }
    )
    return segments


def _adjacent_path_indices(
    region_cells: set[tuple[int, int]],
    path_cells: list[tuple[int, int]],
) -> set[int]:
    cell_to_idx = {cell: i for i, cell in enumerate(path_cells)}
    hits = set()
    for r, c in region_cells:
        for nr, nc in neighbors4(r, c):
            idx = cell_to_idx.get((nr, nc))
            if idx is not None:
                hits.add(idx)
    return hits


def _touch_candidates(
    shape: tuple[int, int],
    start: tuple[int, int],
    controls: list[tuple[tuple[int, int], int]],
    path_cells: list[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    start_and_controls = {start} | {cell for cell, _ in controls}
    candidates = []
    for seg in _path_segments(start, path_cells):
        seg_cells = seg["cells"]
        dr, dc = seg["direction"]
        side_dirs = [(-1, 0), (1, 0)] if dr == 0 else [(0, -1), (0, 1)]
        for side_dr, side_dc in side_dirs:
            for span in range(2, min(4, len(seg_cells)) + 1):
                for offset in range(0, len(seg_cells) - span + 1):
                    chosen = seg_cells[offset : offset + span]
                    chosen_indices = set(range(seg["start_idx"] + offset, seg["start_idx"] + offset + span))
                    rows = [r for r, _ in chosen]
                    cols = [c for _, c in chosen]
                    for depth in (2, 3):
                        if span * depth < 5 or span * depth > 15:
                            continue
                        cells = set()
                        if dr == 0:
                            row = rows[0]
                            c0 = min(cols)
                            c1 = max(cols)
                            r0 = row + side_dr
                            r1 = row + side_dr * depth
                            rr0 = min(r0, r1)
                            rr1 = max(r0, r1)
                            for rr in range(rr0, rr1 + 1):
                                for cc in range(c0, c1 + 1):
                                    cells.add((rr, cc))
                        else:
                            col = cols[0]
                            r0 = min(rows)
                            r1 = max(rows)
                            c0 = col + side_dc
                            c1 = col + side_dc * depth
                            cc0 = min(c0, c1)
                            cc1 = max(c0, c1)
                            for rr in range(r0, r1 + 1):
                                for cc in range(cc0, cc1 + 1):
                                    cells.add((rr, cc))

                        if not cells:
                            continue
                        if any(not in_bounds(shape, r, c) for r, c in cells):
                            continue
                        if cells & start_and_controls:
                            continue
                        if any(cell in path_cells for cell in cells):
                            continue
                        if any(
                            any((nr, nc) in start_and_controls for nr, nc in neighbors4(r, c))
                            for r, c in cells
                        ):
                            continue
                        hits = _adjacent_path_indices(cells, path_cells)
                        if hits != chosen_indices:
                            continue
                        candidates.append(
                            {
                                "anchor": min(chosen_indices),
                                "cells": cells,
                            }
                        )
    uniq = {}
    for cand in candidates:
        key = frozenset(cand["cells"])
        if key not in uniq or cand["anchor"] < uniq[key]["anchor"]:
            uniq[key] = cand
    return list(uniq.values())


def _rect_free_candidates(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    path_cells: list[tuple[int, int]],
    start: tuple[int, int],
    controls: list[tuple[tuple[int, int], int]],
) -> list[set[tuple[int, int]]]:
    h, w = shape
    path_set = set(path_cells)
    control_cells = {cell for cell, _ in controls}
    out = []
    for rh, rw in RECT_SIZES:
        if rh * rw < 5 or rh * rw > 15:
            continue
        for r0 in range(h - rh + 1):
            for c0 in range(w - rw + 1):
                cells = {(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)}
                if cells & blocked:
                    continue
                ok = True
                for r, c in cells:
                    for nr, nc in neighbors4(r, c):
                        if (nr, nc) == start or (nr, nc) in control_cells:
                            ok = False
                        if (nr, nc) in path_set:
                            ok = False
                        if (nr, nc) in blocked and (nr, nc) not in cells:
                            ok = False
                    if not ok:
                        break
                if ok:
                    out.append(cells)
    uniq = []
    seen = set()
    for cells in out:
        key = frozenset(cells)
        if key not in seen:
            seen.add(key)
            uniq.append(cells)
    return uniq


def _choose_touched_regions(
    shape: tuple[int, int],
    start: tuple[int, int],
    controls: list[tuple[tuple[int, int], int]],
    path_cells: list[tuple[int, int]],
    touch_count: int,
) -> list[set[tuple[int, int]]] | None:
    candidates = _touch_candidates(shape, start, controls, path_cells)
    if len(candidates) < touch_count:
        return None

    for _ in range(120):
        selected = []
        blocked = {start} | set(path_cells) | {cell for cell, _ in controls}
        last_anchor = -1
        pool = candidates[:]
        random.shuffle(pool)
        while len(selected) < touch_count:
            options = []
            for cand in pool:
                if cand["anchor"] <= last_anchor:
                    continue
                cells = cand["cells"]
                if cells & blocked:
                    continue
                if any(
                    any((nr, nc) in blocked and (nr, nc) not in set(path_cells) for nr, nc in neighbors4(r, c))
                    for r, c in cells
                ):
                    continue
                options.append(cand)
            if not options:
                break
            cand = random.choice(options)
            selected.append(cand["cells"])
            last_anchor = cand["anchor"]
            blocked |= cand["cells"]
            pool.remove(cand)
        if len(selected) == touch_count:
            return selected
    return None


def _grow_grey_component(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    target_size: int,
    seed: tuple[int, int] | None,
) -> set[tuple[int, int]] | None:
    h, w = shape
    if seed is not None and (not in_bounds(shape, seed[0], seed[1]) or seed in blocked):
        return None

    free_cells = {(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked}
    if seed is None:
        choices = list(free_cells)
        random.shuffle(choices)
        if not choices:
            return None
        seed = choices[0]

    grey = {seed}
    frontier = set()
    for nr, nc in neighbors4(seed[0], seed[1]):
        if (nr, nc) in free_cells:
            frontier.add((nr, nc))

    while len(grey) < target_size:
        if not frontier:
            return None
        choice = random.choice(list(frontier))
        frontier.remove(choice)
        grey.add(choice)
        for nr, nc in neighbors4(choice[0], choice[1]):
            if (nr, nc) in free_cells and (nr, nc) not in grey:
                frontier.add((nr, nc))
    return grey


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    start, controls, regions = _extract_scene(grid)
    traced, path_cells = _trace_path(grid, start, controls)
    order = _touched_region_order(grid.shape, path_cells, regions)
    out = traced.copy()
    for i, idx in enumerate(order):
        src = regions[order[(i - 1) % len(order)]]
        dst_coords = regions[idx]["coords"]
        out[dst_coords[:, 0], dst_coords[:, 1]] = int(src["color"])
    return out


def _validate_scene(inp: np.ndarray, expected_out: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray) and inp.ndim == 2
    assert max(inp.shape) <= 30

    start, controls, regions = _extract_scene(inp)
    assert 4 <= len(regions) <= 6, "region count must be 4-6"
    region_colors = [int(comp["color"]) for comp in regions]
    assert len(region_colors) == len(set(region_colors)), "regions must have distinct colors"
    assert all(color not in {BLACK, GREEN, GREY} for color in region_colors)
    assert all(5 <= int(comp["area"]) <= 15 for comp in regions), "region sizes must stay in range"

    grey_comps = [comp for comp in _same_color_components(inp, {GREY})]
    assert len(grey_comps) == 1, "grey divider must be one connected component"
    assert 20 <= int(grey_comps[0]["area"]) <= 30, "grey divider size must stay in range"

    traced, path_cells = _trace_path(inp, start, controls)
    assert len(path_cells) >= 6, "path should be visually substantial"
    order = _touched_region_order(inp.shape, path_cells, regions)
    assert len(order) >= 3, "path must touch at least three regions"
    assert len(order) < len(regions), "at least one region should remain untouched"

    expected = _independent_expected(inp)
    assert np.array_equal(expected, expected_out), "independent rule reconstruction mismatch"
    actual = main(inp)
    assert np.array_equal(actual, expected_out), "solver output mismatch"

    untouched = set(range(len(regions))) - set(order)
    for idx in untouched:
        coords = regions[idx]["coords"]
        assert np.all(actual[coords[:, 0], coords[:, 1]] == int(regions[idx]["color"]))

    if path_cells:
        for r, c in path_cells:
            assert traced[r, c] == GREEN


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(15, 19)
    w = random.randint(15, 19)

    start_candidates = _boundary_start_candidates(h, w)
    random.shuffle(start_candidates)
    geometry = None
    for start, initial_direction in start_candidates:
        turn_count = random.randint(2, 4)
        turn_colors = [random.choice([TURN_LEFT, TURN_RIGHT]) for _ in range(turn_count)]
        end_mode = random.choice(["edge", "obstacle"])
        geometry = _build_path_geometry(h, w, start, initial_direction, turn_colors, end_mode)
        if geometry is not None:
            break
    if geometry is None:
        return None

    start = geometry["start"]
    path_cells: list[tuple[int, int]] = geometry["path_cells"]
    controls: list[tuple[tuple[int, int], int]] = geometry["controls"]
    stop_cell = geometry["stop_cell"]

    if len(path_cells) < 8:
        return None

    touch_count = random.randint(3, 5)
    touched = _choose_touched_regions((h, w), start, controls, path_cells, touch_count)
    if touched is None:
        return None

    blocked = {start} | set(path_cells) | {cell for cell, _ in controls}
    for cells in touched:
        blocked |= cells

    extra_regions = random.randint(1, min(2, 6 - touch_count))
    untouched_count = extra_regions
    untouched = []
    for _ in range(80):
        candidates = _rect_free_candidates((h, w), blocked, path_cells, start, controls)
        if len(candidates) < untouched_count:
            return None
        untouched = []
        local_blocked = set(blocked)
        for _j in range(untouched_count):
            options = [cells for cells in candidates if not (cells & local_blocked)]
            if not options:
                untouched = []
                break
            cells = random.choice(options)
            untouched.append(cells)
            local_blocked |= cells
        if len(untouched) == untouched_count:
            blocked = local_blocked
            break
    if len(untouched) != untouched_count:
        return None

    grey_size = random.randint(20, 30)
    grey = _grow_grey_component((h, w), blocked, grey_size, stop_cell)
    if grey is None:
        return None
    blocked |= grey

    all_regions = touched + untouched
    if len(all_regions) > len(REGION_COLORS):
        return None

    output_colors = random.sample(REGION_COLORS, len(all_regions))
    output_grid = np.zeros((h, w), dtype=int)
    output_grid[start] = GREEN
    for cell, color in controls:
        output_grid[cell] = color
    for r, c in grey:
        output_grid[r, c] = GREY
    for cells, color in zip(all_regions, output_colors, strict=False):
        for r, c in cells:
            output_grid[r, c] = color

    region_comps = []
    for cells, color in zip(all_regions, output_colors, strict=False):
        coords = np.array(sorted(cells), dtype=int)
        region_comps.append({"coords": coords, "color": color})
    order = _touched_region_order((h, w), path_cells, region_comps)
    if len(order) != touch_count:
        return None

    input_grid = output_grid.copy()
    for i, region_idx in enumerate(order):
        src_color = output_colors[order[(i + 1) % len(order)]]
        for r, c in all_regions[region_idx]:
            input_grid[r, c] = src_color

    expected_out = output_grid.copy()
    expected_out = _independent_expected(input_grid)
    _validate_scene(input_grid, expected_out)
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid = _attempt_generate()
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
