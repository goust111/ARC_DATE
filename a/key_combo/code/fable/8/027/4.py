from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np


ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

try:
    from arc_utils import find_components, get_bg, in_bounds, neighbors4
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = map(int, shape)
        return 0 <= int(r) < h and 0 <= int(c) < w

    def neighbors4(r: int, c: int):
        yield int(r) - 1, int(c)
        yield int(r) + 1, int(c)
        yield int(r), int(c) - 1
        yield int(r), int(c) + 1

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        return_format: str = "dict",
    ):
        assert connectivity == 4 and monochromatic and return_format == "dict"
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        allowed = {int(c) for c in colors} if colors is not None else set(map(int, np.unique(g))) - {int(bg)}
        seen: set[tuple[int, int]] = set()
        out: list[dict] = []
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if (r, c) in seen or int(g[r, c]) == int(bg) or int(g[r, c]) not in allowed:
                    continue
                color = int(g[r, c])
                stack = [(r, c)]
                coords: list[tuple[int, int]] = []
                seen.add((r, c))
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in neighbors4(rr, cc):
                        if not in_bounds(g.shape, nr, nc):
                            continue
                        if (nr, nc) in seen or int(g[nr, nc]) != color:
                            continue
                        seen.add((nr, nc))
                        stack.append((nr, nc))
                arr = np.asarray(coords, dtype=int)
                r0 = int(arr[:, 0].min())
                c0 = int(arr[:, 1].min())
                r1 = int(arr[:, 0].max())
                c1 = int(arr[:, 1].max())
                out.append({"color": color, "coords": arr, "bbox": (r0, c0, r1, c1), "area": int(arr.shape[0])})
        return out


DIRS4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG8: tuple[tuple[int, int], ...] = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
)

GEN_RETRIES = 500
LOCAL_OBSTACLE_RETRIES = 200
PLACEMENT_RETRIES = 250
STAGING_SHAPE = (26, 26)
OBSTACLE_COUNT_RANGE = (2, 4)
GAP_RANGE = (2, 4)


def _coords_of_value(grid: np.ndarray, value: int) -> np.ndarray:
    return np.argwhere(np.asarray(grid, dtype=int) == int(value))


def _neighbors8(r: int, c: int):
    for dr, dc in DIAG8:
        yield int(r) + dr, int(c) + dc


def _adjacent_to_mask(mask: np.ndarray, pos: tuple[int, int]) -> bool:
    return any(in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) for nr, nc in neighbors4(*pos))


def _line_cells(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    sr, sc = start
    dr, dc = direction
    return [(int(sr + dr * i), int(sc + dc * i)) for i in range(int(steps) + 1)]


def _turn_options(entry_dir: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = entry_dir
    if dr:
        return (dr, -1), (dr, 1)
    return (-1, dc), (1, dc)


def _walk_single_turn(
    occ_grid: np.ndarray,
    comp_mask: np.ndarray,
    start: tuple[int, int],
    turn: tuple[int, int],
    bg: int,
) -> list[tuple[int, int]]:
    path = [tuple(map(int, start))]
    r, c = map(int, start)
    tr, tc = turn
    while True:
        nr, nc = r + tr, c + tc
        if not in_bounds(occ_grid.shape, nr, nc):
            break
        if int(occ_grid[nr, nc]) != int(bg):
            break
        if not _adjacent_to_mask(comp_mask, (nr, nc)):
            break
        path.append((int(nr), int(nc)))
        r, c = nr, nc
    return path


def _best_contour_path(
    occ_grid: np.ndarray,
    comp_mask: np.ndarray,
    start: tuple[int, int],
    entry_dir: tuple[int, int],
    bg: int,
) -> dict:
    turns = _turn_options(entry_dir)
    candidates = []
    for turn in turns:
        path = _walk_single_turn(occ_grid, comp_mask, start, turn, bg)
        candidates.append({"turn": turn, "path": path, "length": len(path) - 1})
    lengths = [item["length"] for item in candidates]
    best = max(lengths)
    winners = [item for item in candidates if item["length"] == best]
    assert len(winners) == 1, "contour choice must be uniquely determined"
    assert best > 0, "contour slide must advance at least one cell"
    loser_length = min(lengths)
    winner = winners[0]
    return {
        "path": winner["path"],
        "turn": winner["turn"],
        "winning_length": int(winner["length"]),
        "losing_length": int(loser_length),
        "all_lengths": tuple(int(v) for v in lengths),
    }


def _roles(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    vals, counts = np.unique(g, return_counts=True)
    singleton_colors = [int(v) for v, n in zip(vals, counts) if int(v) != bg and int(n) == 1]
    assert len(singleton_colors) == 1, "expected one single-pixel marker color"
    marker = singleton_colors[0]
    other_colors = [int(v) for v in vals if int(v) not in (bg, marker)]
    assert len(other_colors) == 1, "expected one obstacle color besides bg and marker"
    obstacle = other_colors[0]

    marker_coords = _coords_of_value(g, marker)
    assert marker_coords.shape == (1, 2), "marker must be a singleton pixel"
    marker_pos = tuple(map(int, marker_coords[0]))

    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[obstacle],
        return_format="dict",
    )
    assert len(comps) >= 2, "expected multiple disconnected obstacle components"
    comps.sort(key=lambda obj: tuple(map(int, obj["bbox"])))

    comp_index = np.full(g.shape, -1, dtype=int)
    comp_masks: list[np.ndarray] = []
    for idx, comp in enumerate(comps):
        coords = np.asarray(comp["coords"], dtype=int)
        mask = np.zeros(g.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        comp_masks.append(mask)
        assert np.all(comp_index[coords[:, 0], coords[:, 1]] == -1)
        comp_index[coords[:, 0], coords[:, 1]] = idx

    return g, bg, marker, obstacle, marker_pos, comps, comp_masks, comp_index


def _sightline_candidates(
    g: np.ndarray,
    start: tuple[int, int],
    bg: int,
    comp_index: np.ndarray,
    remaining: set[int],
) -> list[dict]:
    sr, sc = start
    candidates: list[dict] = []
    for direction in DIRS4:
        dr, dc = direction
        r, c = sr + dr, sc + dc
        prev_bg: tuple[int, int] | None = None
        while in_bounds(g.shape, r, c):
            comp_id = int(comp_index[r, c])
            if comp_id in remaining:
                assert prev_bg is not None, "marker cannot start adjacent to an unvisited obstacle"
                candidates.append(
                    {"comp_id": comp_id, "entry_dir": direction, "contour_start": prev_bg}
                )
                break
            if comp_id != -1 or int(g[r, c]) != int(bg):
                break
            prev_bg = (int(r), int(c))
            r += dr
            c += dc
    return candidates


def _trace_route(grid: np.ndarray) -> dict:
    g, bg, marker, obstacle, marker_pos, comps, comp_masks, comp_index = _roles(grid)

    route = [marker_pos]
    current = marker_pos
    remaining = set(range(len(comps)))
    steps = []

    while remaining:
        candidates = _sightline_candidates(g, current, bg, comp_index, remaining)
        assert len(candidates) == 1, "each relay endpoint must have exactly one sightline to an unvisited obstacle"
        chosen = candidates[0]
        comp_id = int(chosen["comp_id"])
        entry_dir = tuple(map(int, chosen["entry_dir"]))
        contour_start = tuple(map(int, chosen["contour_start"]))
        assert _adjacent_to_mask(comp_masks[comp_id], contour_start), "contour start must touch the chosen obstacle"

        dr, dc = entry_dir
        gap = abs(contour_start[0] - current[0]) + abs(contour_start[1] - current[1])
        assert gap >= 1 and gap == abs(dr) * gap + abs(dc) * gap, "sightline must be cardinal"
        approach = _line_cells(current, entry_dir, gap)

        contour = _best_contour_path(g, comp_masks[comp_id], contour_start, entry_dir, bg)
        route.extend(approach[1:])
        route.extend(contour["path"][1:])
        current = contour["path"][-1]
        remaining.remove(comp_id)
        steps.append(
            {
                "comp_id": comp_id,
                "entry_dir": entry_dir,
                "gap": int(gap),
                "contour_start": contour_start,
                "slide_dir": tuple(map(int, contour["turn"])),
                "winning_length": int(contour["winning_length"]),
                "losing_length": int(contour["losing_length"]),
                "path": tuple(route[-(gap + contour["winning_length"]):]),
                "end": current,
            }
        )

    assert not _sightline_candidates(g, current, bg, comp_index, remaining), "no further relay step should remain"
    return {
        "bg": bg,
        "marker": marker,
        "obstacle": obstacle,
        "start": marker_pos,
        "end": current,
        "route": tuple(route),
        "steps": steps,
        "obstacle_count": len(comps),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay the marker from obstacle to obstacle by alternating a straight sightline and the longer adjacent contour slide.

    Input structure:
    - A dominant background contains one singleton marker pixel and several disconnected obstacle blocks of one other color.
    - From the marker, exactly one horizontal or vertical sightline reaches an obstacle through only background.
    - After each contour slide ends, the new endpoint again has exactly one sightline to a still-unvisited obstacle, until all obstacles are used.

    Rule:
    - Follow the unique sightline to the last background cell before the next obstacle.
    - From that cell, test the two diagonal contour-following turns and keep the one that stays 4-adjacent to that obstacle for longer.
    - Repeat this straight-run then contour-slide relay once per obstacle, and paint every visited path cell in the marker color while leaving obstacles unchanged.

    Key insight:
    The answer is the whole relay chain: each local left-or-right contour decision determines the next visible obstacle, so the full trajectory is recovered only by iterating the one-step slide rule across all obstacles.
    """
    g = np.asarray(grid, dtype=int)
    trace = _trace_route(g)
    out = g.copy()
    for r, c in trace["route"]:
        out[int(r), int(c)] = int(trace["marker"])
    return out


def _gen_boundary(
    rng: random.Random,
    height: int,
    start_row: int,
    walk_len: int,
    start_left: int,
) -> list[int]:
    lefts = [0] * height
    lefts[start_row] = int(start_left)

    for r in range(start_row - 1, -1, -1):
        if lefts[r + 1] == 0:
            lefts[r] = 0
        else:
            lefts[r] = lefts[r + 1] - rng.randint(0, 1)

    for k in range(1, walk_len + 1):
        lefts[start_row + k] = start_left + k

    stop_row = start_row + walk_len + 1
    if stop_row < height:
        lefts[stop_row] = lefts[stop_row - 1]
        for r in range(stop_row + 1, height):
            if rng.random() < 0.55:
                lefts[r] = lefts[r - 1] + rng.randint(0, 1)
            else:
                lefts[r] = lefts[r - 1]
    return lefts


def _build_canonical_obstacle(rng: random.Random) -> np.ndarray:
    for _ in range(LOCAL_OBSTACLE_RETRIES):
        height = rng.randint(5, 8)
        max_walk = min(4, height - 3)
        if max_walk < 2:
            continue
        walk_len = rng.randint(2, max_walk)
        start_row = rng.randint(1, height - walk_len - 2)
        start_left = rng.randint(1, 3)
        body_width = rng.randint(3, 6)

        lefts = _gen_boundary(rng, height, start_row, walk_len, start_left)
        min_left = min(lefts)
        max_left = max(lefts)
        shift = min_left - 2
        width = body_width + max_left - min_left + 2

        canvas = np.zeros((height, width), dtype=int)
        for rr, left in enumerate(lefts):
            c0 = left - shift
            canvas[rr, c0 : c0 + body_width] = 1

        contour_start = (start_row, start_left - shift - 1)
        prev_cell = (contour_start[0], contour_start[1] - 1)
        if canvas[contour_start] != 0 or canvas[prev_cell] != 0:
            continue

        mask = canvas == 1
        contour = _best_contour_path(canvas, mask, contour_start, (0, 1), 0)
        end = contour["path"][-1]
        if end[0] <= contour_start[0]:
            continue
        if contour["winning_length"] <= contour["losing_length"]:
            continue

        canvas[contour_start] = 2
        canvas[prev_cell] = 3
        return canvas
    raise RuntimeError("failed to synthesize a canonical obstacle")


def _transform_canvas(canvas: np.ndarray, rotations: int, mirror: bool) -> np.ndarray:
    out = np.rot90(canvas, int(rotations))
    if mirror:
        out = np.fliplr(out)
    return out.copy()


def _random_obstacle_spec(rng: random.Random) -> dict:
    base = _build_canonical_obstacle(rng)
    transformed = _transform_canvas(base, rng.randrange(4), rng.random() < 0.5)

    obstacle_coords = np.argwhere(transformed == 1)
    contour_start_coords = np.argwhere(transformed == 2)
    prev_coords = np.argwhere(transformed == 3)
    assert contour_start_coords.shape == (1, 2)
    assert prev_coords.shape == (1, 2)

    contour_start = tuple(map(int, contour_start_coords[0]))
    prev_cell = tuple(map(int, prev_coords[0]))
    entry_dir = (contour_start[0] - prev_cell[0], contour_start[1] - prev_cell[1])

    occ = np.where(transformed == 1, 1, 0)
    contour = _best_contour_path(occ, occ == 1, contour_start, entry_dir, 0)
    return {
        "start": contour_start,
        "entry_dir": entry_dir,
        "obstacle": [tuple(map(int, cell)) for cell in obstacle_coords],
        "path": [tuple(map(int, cell)) for cell in contour["path"]],
        "shape": occ.shape,
    }


def _translate_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in cells]


def _all_in_bounds(shape: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
    return all(in_bounds(shape, r, c) for r, c in cells)


def _bbox_of_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_and_shift(
    rng: random.Random,
    obstacle_cells: list[tuple[int, int]],
    route_cells: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    all_cells = obstacle_cells + route_cells
    r0, c0, r1, c1 = _bbox_of_cells(all_cells)
    base_h = r1 - r0 + 1
    base_w = c1 - c0 + 1

    top = rng.randint(0, min(3, 30 - base_h))
    bottom = rng.randint(0, min(3, 30 - base_h - top))
    left = rng.randint(0, min(3, 30 - base_w))
    right = rng.randint(0, min(3, 30 - base_w - left))

    dr = top - r0
    dc = left - c0
    shifted_obs = _translate_cells(obstacle_cells, dr, dc)
    shifted_route = _translate_cells(route_cells, dr, dc)
    return shifted_obs, shifted_route, (base_h + top + bottom, base_w + left + right)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    stage_h, stage_w = STAGING_SHAPE
    obstacle_count = rng.randint(OBSTACLE_COUNT_RANGE[0], OBSTACLE_COUNT_RANGE[1])
    current = (rng.randint(6, stage_h - 7), rng.randint(6, stage_w - 7))

    obstacle_cells: set[tuple[int, int]] = set()
    obstacle_halo: set[tuple[int, int]] = set()
    route_cells = [current]
    route_set = {current}

    for _ in range(obstacle_count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            spec = _random_obstacle_spec(rng)
            gap = rng.randint(GAP_RANGE[0], GAP_RANGE[1])
            dr, dc = spec["entry_dir"]
            target_start = (current[0] + dr * gap, current[1] + dc * gap)
            shift_r = target_start[0] - spec["start"][0]
            shift_c = target_start[1] - spec["start"][1]

            obs = _translate_cells(spec["obstacle"], shift_r, shift_c)
            contour = _translate_cells(spec["path"], shift_r, shift_c)
            approach = _line_cells(current, spec["entry_dir"], gap)
            added_route = approach[1:] + contour[1:]

            if not _all_in_bounds(STAGING_SHAPE, obs + added_route):
                continue
            if any(cell in obstacle_cells for cell in added_route):
                continue
            if any(cell in route_set for cell in added_route):
                continue
            if any(cell in obstacle_cells or cell in route_set for cell in obs):
                continue
            if any(cell in obstacle_halo for cell in obs):
                continue

            obstacle_cells.update(obs)
            for r, c in obs:
                obstacle_halo.add((r, c))
                for nr, nc in _neighbors8(r, c):
                    if in_bounds(STAGING_SHAPE, nr, nc):
                        obstacle_halo.add((nr, nc))

            route_cells.extend(added_route)
            route_set.update(added_route)
            current = contour[-1]
            placed = True
            break
        if not placed:
            return None

    shifted_obs, shifted_route, final_shape = _crop_and_shift(rng, list(obstacle_cells), route_cells)
    bg, obstacle, marker = rng.sample(range(10), 3)
    grid = np.full(final_shape, bg, dtype=int)
    for r, c in shifted_obs:
        grid[r, c] = obstacle
    start = shifted_route[0]
    grid[start] = marker

    try:
        trace = _trace_route(grid)
    except AssertionError:
        return None

    if len(trace["steps"]) != obstacle_count:
        return None
    if len(set(trace["route"])) != len(trace["route"]):
        return None
    if any(step["winning_length"] <= step["losing_length"] for step in trace["steps"]):
        return None
    if tuple(trace["route"]) != tuple(shifted_route):
        return None
    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-slide ARC input")
