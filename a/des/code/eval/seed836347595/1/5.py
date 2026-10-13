from __future__ import annotations

import random
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
CYAN = 8
MAROON = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SPECIAL_COLORS = {BLACK, BLUE, RED, GREEN, GREY, MAROON}
CAR_COLORS = [YELLOW, PINK, ORANGE, CYAN]

GRID_SIZE_RANGE = (18, 27)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120
RECT_ATTEMPTS = 240

Coord = tuple[int, int]
Rect = tuple[int, int, int, int]


def _add(a: Coord, b: Coord) -> Coord:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Coord, b: Coord) -> Coord:
    return a[0] - b[0], a[1] - b[1]


def _neg(a: Coord) -> Coord:
    return -a[0], -a[1]


def _neighbors(mask: np.ndarray, cell: Coord) -> list[Coord]:
    h, w = mask.shape
    r, c = cell
    out: list[Coord] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            out.append((rr, cc))
    return out


def _step_same_track(mask: np.ndarray, prev: Coord, curr: Coord) -> Coord:
    neigh = _neighbors(mask, curr)
    assert len(neigh) in (2, 4), "track cells must have degree 2 or 4"
    assert prev in neigh, "previous step must stay on the track"
    if len(neigh) == 2:
        return neigh[0] if neigh[1] == prev else neigh[1]

    direction = _sub(curr, prev)
    nxt = _add(curr, direction)
    assert nxt in neigh, "degree-4 cells must support straight-through continuation"
    return nxt


def _find_cars(grid: np.ndarray) -> list[dict[str, Any]]:
    car_colors = sorted(set(map(int, grid.ravel())) - SPECIAL_COLORS)
    assert car_colors, "expected at least one car"

    arrow_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == MAROON)}
    cars: list[dict[str, Any]] = []
    used_arrows: set[Coord] = set()

    for color in car_colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (1, 2), "each car color must appear exactly once"
        car = tuple(map(int, coords[0]))
        arrows = []
        for dr, dc in DIR4:
            arrow = (car[0] + dr, car[1] + dc)
            if arrow in arrow_cells:
                arrows.append(arrow)
        assert len(arrows) == 1, "each car must have exactly one adjacent maroon arrow"
        used_arrows.add(arrows[0])
        cars.append({"color": int(color), "car": car, "arrow": arrows[0]})

    assert used_arrows == arrow_cells, "every maroon pixel must be an arrow"
    return cars


def _trace_cycle(mask: np.ndarray, car: Coord, arrow: Coord) -> list[Coord]:
    cycle = [car, arrow]
    prev, curr = car, arrow
    limit = int(mask.sum()) + 5

    for _ in range(limit):
        nxt = _step_same_track(mask, prev, curr)
        if nxt == car:
            break
        cycle.append(nxt)
        prev, curr = curr, nxt
    else:
        raise AssertionError("failed to close track cycle")

    assert len(cycle) >= 4, "track loop is too short"
    assert len(set(cycle)) == len(cycle), "track loop must be simple"
    assert _step_same_track(mask, cycle[-1], car) == arrow, "car/arrow must define the loop orientation"
    return cycle


def _analyze_scene(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mask = g != BLACK
    track_cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    assert track_cells, "expected non-black track cells"

    degrees: dict[Coord, int] = {}
    for cell in track_cells:
        degree = len(_neighbors(mask, cell))
        degrees[cell] = degree
        assert degree in (2, 4), "all track cells must have degree 2 or 4"
        if degree == 4:
            assert int(g[cell]) in (RED, BLUE), "degree-4 cells must be colored intersections"

    cars = _find_cars(g)
    tracks: list[dict[str, Any]] = []
    cell_to_tracks: dict[Coord, list[int]] = {}
    intersection_dirs: dict[Coord, list[tuple[int, Coord]]] = {}

    for track_id, car_info in enumerate(cars):
        cycle = _trace_cycle(mask, car_info["car"], car_info["arrow"])
        tracks.append({**car_info, "track_id": track_id, "cycle": cycle})
        for idx, curr in enumerate(cycle):
            cell_to_tracks.setdefault(curr, []).append(track_id)
            prev = cycle[idx - 1]
            nxt = cycle[(idx + 1) % len(cycle)]
            if degrees[curr] == 4:
                direction = _sub(curr, prev)
                assert nxt == _add(curr, direction), "track orientation must stay straight through intersections"
                intersection_dirs.setdefault(curr, []).append((track_id, direction))

    covered = set(cell_to_tracks)
    assert covered == track_cells, "every track cell must belong to a traced loop"

    for cell, owners in cell_to_tracks.items():
        if degrees[cell] == 4:
            assert len(owners) == 2, "each intersection must belong to exactly two tracks"
            dirs = intersection_dirs[cell]
            assert len(dirs) == 2
            vecs = [vec for _track_id, vec in dirs]
            assert vecs[0] != vecs[1]
            assert vecs[0] != _neg(vecs[1]), "intersecting track directions must be perpendicular"
        else:
            assert len(owners) == 1, "non-intersection cells must belong to exactly one track"

    return {
        "grid": g,
        "mask": mask,
        "degrees": degrees,
        "tracks": tracks,
        "cell_to_tracks": cell_to_tracks,
        "intersection_dirs": intersection_dirs,
    }


def _route_cars(analysis: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    g = analysis["grid"]
    mask = analysis["mask"]
    degrees = analysis["degrees"]
    intersection_dirs = analysis["intersection_dirs"]
    cell_to_tracks = analysis["cell_to_tracks"]

    branch_counts = {"red": 0, "blue": 0}
    routed: list[dict[str, Any]] = []

    for track in analysis["tracks"]:
        prev = track["car"]
        curr = track["arrow"]
        seen_states: set[tuple[Coord, Coord]] = set()
        limit = int(mask.sum()) * 6

        for _ in range(limit):
            state = (prev, curr)
            assert state not in seen_states, "car looped without reaching any station"
            seen_states.add(state)

            cell_color = int(g[curr])
            if cell_color == GREEN:
                dest_track = analysis["cell_to_tracks"][curr]
                assert len(dest_track) == 1
                routed.append(
                    {
                        "color": int(track["color"]),
                        "start_track": int(track["track_id"]),
                        "start": track["car"],
                        "arrow": track["arrow"],
                        "dest": curr,
                        "dest_track": int(dest_track[0]),
                    }
                )
                break

            if degrees[curr] == 4:
                incoming = _sub(curr, prev)
                if cell_color == RED:
                    branch_counts["red"] += 1
                    nxt = _add(curr, incoming)
                    assert nxt in _neighbors(mask, curr)
                else:
                    assert cell_color == BLUE, "degree-4 cells must be red or blue"
                    branch_counts["blue"] += 1
                    candidates = [
                        vec
                        for _track_id, vec in intersection_dirs[curr]
                        if vec != incoming and vec != _neg(incoming)
                    ]
                    assert len(candidates) == 1, "blue intersection must offer exactly one track switch"
                    nxt = _add(curr, candidates[0])
                    assert nxt in _neighbors(mask, curr)
            else:
                nxt = _step_same_track(mask, prev, curr)

            prev, curr = curr, nxt
        else:
            raise AssertionError("car failed to reach a station")

    assert len(routed) == len(analysis["tracks"])
    return routed, branch_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored cars along rectangular loop tracks.

    Input structure:
    - Black background.
    - Grey rectangular loops form 1-pixel-wide tracks.
    - Red/blue pixels are shared intersections of two perpendicular tracks.
    - Each track has one colored car, one adjacent maroon arrow on the track,
      and one green station.

    Rule:
    - The arrow's relative position gives the car's initial direction.
    - Cars move one pixel at a time.
    - At red intersections they go straight through.
    - At blue intersections they switch to the other track's oriented direction.
    - They stop on the first green station they reach.

    Output:
    - Cars recolor their destination station.
    - Original car cells and all maroon arrow cells become grey.
    - All other pixels stay unchanged.
    """
    analysis = _analyze_scene(grid)
    routed, _branch_counts = _route_cars(analysis)

    out = analysis["grid"].copy()
    for track in analysis["tracks"]:
        out[track["car"]] = GREY
        out[track["arrow"]] = GREY

    destinations = [item["dest"] for item in routed]
    assert len(set(destinations)) == len(destinations), "cars must stop on distinct stations"
    for item in routed:
        out[item["dest"]] = int(item["color"])

    return out


def _rect_cycle(rect: Rect) -> list[Coord]:
    r0, c0, r1, c1 = rect
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        cells.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        cells.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cells.append((r, c0))
    return cells


def _rect_role(rect: Rect, cell: Coord) -> str:
    r0, c0, r1, c1 = rect
    r, c = cell
    if (r in (r0, r1)) and (c in (c0, c1)):
        return "corner"
    if r in (r0, r1):
        return "h"
    if c in (c0, c1):
        return "v"
    raise AssertionError("cell is not on the rectangle perimeter")


def _bbox_intersects(a: Rect, b: Rect, pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _layout_info(rects: list[Rect], height: int, width: int) -> dict[str, Any] | None:
    entries: dict[Coord, list[tuple[int, str]]] = {}
    base_cycles: list[list[Coord]] = []

    for track_id, rect in enumerate(rects):
        cycle = _rect_cycle(rect)
        base_cycles.append(cycle)
        for cell in cycle:
            entries.setdefault(cell, []).append((track_id, _rect_role(rect, cell)))

    intersections: set[Coord] = set()
    for cell, owners in entries.items():
        if len(owners) > 2:
            return None
        if len(owners) == 2:
            if len({track_id for track_id, _role in owners}) != 2:
                return None
            if {role for _track_id, role in owners} != {"h", "v"}:
                return None
            intersections.add(cell)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in entries:
        mask[r, c] = True

    degrees: dict[Coord, int] = {}
    for cell in entries:
        degree = len(_neighbors(mask, cell))
        degrees[cell] = degree
        if degree not in (2, 4):
            return None
        if (cell in intersections) != (degree == 4):
            return None

    for cell in intersections:
        for nb in _neighbors(mask, cell):
            if nb in intersections:
                return None

    usable_by_track: list[set[Coord]] = []
    pair_indices_by_track: list[list[int]] = []
    for track_id, rect in enumerate(rects):
        cycle = base_cycles[track_id]
        usable = {
            cell
            for cell in cycle
            if _rect_role(rect, cell) != "corner" and cell not in intersections
        }
        if len(usable) < 3:
            return None
        usable_by_track.append(usable)
        pair_indices = [
            idx
            for idx in range(len(cycle))
            if cycle[idx] in usable and cycle[(idx + 1) % len(cycle)] in usable
        ]
        if not pair_indices:
            return None
        pair_indices_by_track.append(pair_indices)

    return {
        "entries": entries,
        "intersections": intersections,
        "mask": mask,
        "degrees": degrees,
        "base_cycles": base_cycles,
        "usable_by_track": usable_by_track,
        "pair_indices_by_track": pair_indices_by_track,
    }


def _random_rect(height: int, width: int, family: str) -> Rect | None:
    if family == "v":
        h_lo = max(7, height // 2)
        h_hi = height - 2
        w_lo = 5
        w_hi = min(8, width - 4)
    else:
        h_lo = 5
        h_hi = min(8, height - 4)
        w_lo = max(7, width // 2)
        w_hi = width - 2

    if h_lo > h_hi or w_lo > w_hi:
        return None

    h = random.randint(h_lo, h_hi)
    w = random.randint(w_lo, w_hi)
    top_hi = height - h - 1
    left_hi = width - w - 1
    if top_hi < 1 or left_hi < 1:
        return None
    r0 = random.randint(1, top_hi)
    c0 = random.randint(1, left_hi)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _sample_rectangles(height: int, width: int) -> tuple[list[Rect], dict[str, Any]] | None:
    families = ["v", "h"]
    if random.random() < 0.6:
        families.append(random.choice(["v", "h"]))
    random.shuffle(families)

    rects: list[Rect] = []
    rect_families: list[str] = []

    for family in families:
        placed = False
        for _ in range(RECT_ATTEMPTS):
            rect = _random_rect(height, width, family)
            if rect is None:
                continue
            if any(
                family == other_family and _bbox_intersects(rect, other_rect, pad=1)
                for other_rect, other_family in zip(rects, rect_families)
            ):
                continue

            trial = rects + [rect]
            info = _layout_info(trial, height, width)
            if info is None:
                continue

            if rects:
                new_track_id = len(trial) - 1
                touches_existing = any(
                    any(track_id == new_track_id for track_id, _role in owners)
                    for cell, owners in info["entries"].items()
                    if len(owners) == 2
                )
                if not touches_existing:
                    continue

            rects.append(rect)
            rect_families.append(family)
            placed = True
            break

        if not placed:
            return None

    info = _layout_info(rects, height, width)
    if info is None or len(info["intersections"]) < 2:
        return None
    return rects, info


def _choose_track_specs(rects: list[Rect], info: dict[str, Any]) -> list[dict[str, Any]] | None:
    specs: list[dict[str, Any]] = []
    colors = random.sample(CAR_COLORS, len(rects))
    orientations = [random.choice([True, False]) for _ in rects]

    for track_id, (rect, color, clockwise) in enumerate(zip(rects, colors, orientations)):
        cycle = list(info["base_cycles"][track_id])
        if not clockwise:
            cycle.reverse()

        usable = info["usable_by_track"][track_id]
        pair_indices = [
            idx
            for idx in range(len(cycle))
            if cycle[idx] in usable and cycle[(idx + 1) % len(cycle)] in usable
        ]
        if not pair_indices:
            return None

        car_idx = random.choice(pair_indices)
        car = cycle[car_idx]
        arrow = cycle[(car_idx + 1) % len(cycle)]
        stations = [cell for cell in cycle if cell in usable and cell not in {car, arrow}]
        if not stations:
            return None

        station = random.choice(stations)
        specs.append(
            {
                "track_id": track_id,
                "rect": rect,
                "color": int(color),
                "cycle": cycle,
                "car": car,
                "arrow": arrow,
                "station": station,
            }
        )

    return specs


def _render_scene(
    height: int,
    width: int,
    info: dict[str, Any],
    specs: list[dict[str, Any]],
    intersection_colors: dict[Coord, int],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for r, c in info["entries"]:
        grid[r, c] = GREY
    for cell, color in intersection_colors.items():
        grid[cell] = int(color)
    for spec in specs:
        grid[spec["station"]] = GREEN
    for spec in specs:
        grid[spec["arrow"]] = MAROON
    for spec in specs:
        grid[spec["car"]] = int(spec["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)

    sampled = _sample_rectangles(height, width)
    if sampled is None:
        return None
    rects, info = sampled

    require_foreign_stop = random.choice([False, True])

    for _ in range(PLACEMENT_ATTEMPTS):
        specs = _choose_track_specs(rects, info)
        if specs is None:
            continue

        intersections = list(info["intersections"])
        random.shuffle(intersections)
        split = random.randint(1, len(intersections) - 1)
        blue_cells = set(intersections[:split])
        intersection_colors = {
            cell: (BLUE if cell in blue_cells else RED) for cell in intersections
        }

        grid = _render_scene(height, width, info, specs, intersection_colors)

        try:
            analysis = _analyze_scene(grid)
            routed, branch_counts = _route_cars(analysis)
            out = main(grid)
        except AssertionError:
            continue

        if branch_counts["red"] == 0 or branch_counts["blue"] == 0:
            continue
        if len(set(item["dest"] for item in routed)) != len(routed):
            continue
        if require_foreign_stop and not any(item["dest_track"] != item["start_track"] for item in routed):
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid track-routing instance")
