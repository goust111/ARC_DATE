from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


BG = 0
RED = 2
SIZES = (5, 7)
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
MAX_GRID_SIZE = 30
_RNG = np.random.default_rng()
_OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}


@dataclass(frozen=True)
class Gear:
    color: int
    r0: int
    c0: int
    size: int
    teeth: tuple[bool, ...]
    red_count: int


def _in_bounds(r: int, c: int, height: int, width: int) -> bool:
    return 0 <= int(r) < int(height) and 0 <= int(c) < int(width)


def _frame_border_coords(r0: int, c0: int, size: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(int(c0), int(c0 + size)):
        coords.append((int(r0), int(c)))
        coords.append((int(r0 + size - 1), int(c)))
    for r in range(int(r0 + 1), int(r0 + size - 1)):
        coords.append((int(r), int(c0)))
        coords.append((int(r), int(c0 + size - 1)))
    return coords


def _clockwise_slots(r0: int, c0: int, size: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for dc in range(1, int(size - 1)):
        coords.append((int(r0 - 1), int(c0 + dc)))
    for dr in range(1, int(size - 1)):
        coords.append((int(r0 + dr), int(c0 + size)))
    for dc in range(int(size - 2), 0, -1):
        coords.append((int(r0 + size), int(c0 + dc)))
    for dr in range(int(size - 2), 0, -1):
        coords.append((int(r0 + dr), int(c0 - 1)))
    return coords


def _edge_slots(r0: int, c0: int, size: int, side: str) -> list[tuple[int, int]]:
    if side == "U":
        return [(int(r0 - 1), int(c0 + dc)) for dc in range(1, int(size - 1))]
    if side == "D":
        return [(int(r0 + size), int(c0 + dc)) for dc in range(1, int(size - 1))]
    if side == "L":
        return [(int(r0 + dr), int(c0 - 1)) for dr in range(1, int(size - 1))]
    if side == "R":
        return [(int(r0 + dr), int(c0 + size)) for dr in range(1, int(size - 1))]
    raise ValueError(f"unknown side: {side}")


def _parse_gears(grid: np.ndarray) -> tuple[list[Gear], int]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected 2D grid")
    height, width = map(int, g.shape)
    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, RED)]
    gears: list[Gear] = []

    for color in colors:
        color_cells = {(int(r), int(c)) for r, c in np.argwhere(g == int(color))}
        matches: list[Gear] = []
        for size in SIZES:
            for r0 in range(0, int(height - size + 1)):
                for c0 in range(0, int(width - size + 1)):
                    border = _frame_border_coords(int(r0), int(c0), int(size))
                    if any(int(g[r, c]) != int(color) for r, c in border):
                        continue
                    interior = g[int(r0 + 1) : int(r0 + size - 1), int(c0 + 1) : int(c0 + size - 1)]
                    if bool((interior == int(color)).any()):
                        continue

                    allowed = set(border)
                    for rr, cc in _clockwise_slots(int(r0), int(c0), int(size)):
                        if _in_bounds(int(rr), int(cc), int(height), int(width)):
                            allowed.add((int(rr), int(cc)))
                    if not color_cells.issubset(allowed):
                        continue

                    slots = _clockwise_slots(int(r0), int(c0), int(size))
                    teeth = tuple((int(rr), int(cc)) in color_cells for rr, cc in slots)
                    red_count = int((interior == RED).sum())
                    matches.append(
                        Gear(
                            color=int(color),
                            r0=int(r0),
                            c0=int(c0),
                            size=int(size),
                            teeth=teeth,
                            red_count=red_count,
                        )
                    )

        if len(matches) != 1:
            raise ValueError(f"color {color} does not define exactly one gear")
        gears.append(matches[0])

    gears.sort(key=lambda gear: (int(gear.r0), int(gear.c0), int(gear.color)))

    driver_candidates = [idx for idx, gear in enumerate(gears) if int(gear.red_count) > 0]
    if len(driver_candidates) != 1:
        raise ValueError("expected exactly one driver gear")

    for r, c in np.argwhere(g == RED):
        inside = False
        for gear in gears:
            if int(gear.r0) < int(r) < int(gear.r0 + gear.size - 1) and int(gear.c0) < int(c) < int(gear.c0 + gear.size - 1):
                inside = True
                break
        if not inside:
            raise ValueError("red marker must be inside a gear")

    return gears, int(driver_candidates[0])


def _tooth_cells(gear: Gear) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for (r, c), present in zip(_clockwise_slots(int(gear.r0), int(gear.c0), int(gear.size)), gear.teeth)
        if bool(present)
    }


def _meshing_graph(gears: list[Gear]) -> list[set[int]]:
    graph = [set() for _ in gears]
    tooth_sets = [_tooth_cells(gear) for gear in gears]

    for i, gear_a in enumerate(gears):
        for j in range(int(i + 1), int(len(gears))):
            gear_b = gears[j]
            teeth_a = tooth_sets[i]
            teeth_b = tooth_sets[j]
            linked = False

            if int(gear_a.c0 + gear_a.size + 3) == int(gear_b.c0):
                a_rows = {int(r) for r, c in teeth_a if int(c) == int(gear_a.c0 + gear_a.size)}
                b_rows = {int(r) for r, c in teeth_b if int(c) == int(gear_b.c0 - 1)}
                linked = bool(a_rows & b_rows)
            elif int(gear_b.c0 + gear_b.size + 3) == int(gear_a.c0):
                a_rows = {int(r) for r, c in teeth_a if int(c) == int(gear_a.c0 - 1)}
                b_rows = {int(r) for r, c in teeth_b if int(c) == int(gear_b.c0 + gear_b.size)}
                linked = bool(a_rows & b_rows)
            elif int(gear_a.r0 + gear_a.size + 3) == int(gear_b.r0):
                a_cols = {int(c) for r, c in teeth_a if int(r) == int(gear_a.r0 + gear_a.size)}
                b_cols = {int(c) for r, c in teeth_b if int(r) == int(gear_b.r0 - 1)}
                linked = bool(a_cols & b_cols)
            elif int(gear_b.r0 + gear_b.size + 3) == int(gear_a.r0):
                a_cols = {int(c) for r, c in teeth_a if int(r) == int(gear_a.r0 - 1)}
                b_cols = {int(c) for r, c in teeth_b if int(r) == int(gear_b.r0 + gear_b.size)}
                linked = bool(a_cols & b_cols)

            if linked:
                graph[i].add(int(j))
                graph[j].add(int(i))

    return graph


def _turn_directions(graph: list[set[int]], driver: int) -> dict[int, int]:
    turns = {int(driver): 1}
    queue: deque[int] = deque([int(driver)])
    while queue:
        cur = int(queue.popleft())
        for nxt in graph[cur]:
            expected = -int(turns[cur])
            if int(nxt) in turns:
                if int(turns[int(nxt)]) != int(expected):
                    raise ValueError("meshing graph is not bipartite")
                continue
            turns[int(nxt)] = int(expected)
            queue.append(int(nxt))
    return turns


def _rotate_teeth(teeth: tuple[bool, ...], shift: int) -> tuple[bool, ...]:
    count = int(len(teeth))
    if count == 0:
        return teeth
    shift %= int(count)
    if shift == 0:
        return teeth
    result = [False] * int(count)
    for idx, present in enumerate(teeth):
        if bool(present):
            result[(int(idx) + int(shift)) % int(count)] = True
    return tuple(result)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate square-frame gears along their meshing chain.

    Exactly one gear contains red marker pixels inside its frame. Their count is
    the step count N. The driver rotates clockwise; meshed neighbors alternate
    direction by graph distance. Teeth slide along the non-corner perimeter-slot
    sequence while the square frame itself stays fixed. Red marker pixels are
    cleared in the output.
    """
    g = np.asarray(grid, dtype=int)
    gears, driver = _parse_gears(g)
    graph = _meshing_graph(gears)
    turns = _turn_directions(graph, int(driver))
    steps = int(gears[int(driver)].red_count)

    out = g.copy()
    out[out == RED] = BG

    height, width = map(int, out.shape)
    for gear in gears:
        for r, c in _clockwise_slots(int(gear.r0), int(gear.c0), int(gear.size)):
            if _in_bounds(int(r), int(c), int(height), int(width)):
                out[int(r), int(c)] = BG

    for idx, gear in enumerate(gears):
        shift = int(steps * turns.get(int(idx), 0))
        rotated = _rotate_teeth(gear.teeth, int(shift))
        for (r, c), present in zip(_clockwise_slots(int(gear.r0), int(gear.c0), int(gear.size)), rotated):
            if bool(present) and _in_bounds(int(r), int(c), int(height), int(width)):
                out[int(r), int(c)] = int(gear.color)

    return out


def _outer_bbox(spec: dict) -> tuple[int, int, int, int]:
    return (
        int(spec["r0"] - 1),
        int(spec["c0"] - 1),
        int(spec["r0"] + spec["size"]),
        int(spec["c0"] + spec["size"]),
    )


def _boxes_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    return not (
        int(box_a[2]) < int(box_b[0])
        or int(box_b[2]) < int(box_a[0])
        or int(box_a[3]) < int(box_b[1])
        or int(box_b[3]) < int(box_a[1])
    )


def _potential_mesh(spec_a: dict, spec_b: dict) -> bool:
    if int(spec_a["c0"] + spec_a["size"] + 3) == int(spec_b["c0"]) or int(spec_b["c0"] + spec_b["size"] + 3) == int(spec_a["c0"]):
        top = max(int(spec_a["r0"] + 1), int(spec_b["r0"] + 1))
        bottom = min(int(spec_a["r0"] + spec_a["size"] - 2), int(spec_b["r0"] + spec_b["size"] - 2))
        return int(top) <= int(bottom)
    if int(spec_a["r0"] + spec_a["size"] + 3) == int(spec_b["r0"]) or int(spec_b["r0"] + spec_b["size"] + 3) == int(spec_a["r0"]):
        left = max(int(spec_a["c0"] + 1), int(spec_b["c0"] + 1))
        right = min(int(spec_a["c0"] + spec_a["size"] - 2), int(spec_b["c0"] + spec_b["size"] - 2))
        return int(left) <= int(right)
    return False


def _placement_ok(candidate: dict, specs: list[dict], allowed_neighbor: int | None) -> bool:
    cand_box = _outer_bbox(candidate)
    for idx, other in enumerate(specs):
        if _boxes_overlap(cand_box, _outer_bbox(other)):
            return False
        if int(idx) != int(allowed_neighbor) if allowed_neighbor is not None else True:
            if _potential_mesh(candidate, other):
                return False
    return True


def _attach_spec(parent: dict, size: int, color: int, side: str) -> dict | None:
    for _ in range(24):
        if side in ("L", "R"):
            parent_pos = int(_RNG.integers(1, int(parent["size"] - 1)))
            child_pos = int(_RNG.integers(1, int(size - 1)))
            r0 = int(parent["r0"] + parent_pos - child_pos)
            c0 = int(parent["c0"] + parent["size"] + 3) if side == "R" else int(parent["c0"] - size - 3)
            child = {
                "size": int(size),
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "required": set(),
                "used_sides": {str(_OPPOSITE[side])},
            }
            row = int(parent["r0"] + parent_pos)
            if side == "R":
                parent["required"].add((int(row), int(parent["c0"] + parent["size"])))
                child["required"].add((int(row), int(child["c0"] - 1)))
            else:
                parent["required"].add((int(row), int(parent["c0"] - 1)))
                child["required"].add((int(row), int(child["c0"] + child["size"])))
            return child

        parent_pos = int(_RNG.integers(1, int(parent["size"] - 1)))
        child_pos = int(_RNG.integers(1, int(size - 1)))
        r0 = int(parent["r0"] + parent["size"] + 3) if side == "D" else int(parent["r0"] - size - 3)
        c0 = int(parent["c0"] + parent_pos - child_pos)
        child = {
            "size": int(size),
            "color": int(color),
            "r0": int(r0),
            "c0": int(c0),
            "required": set(),
            "used_sides": {str(_OPPOSITE[side])},
        }
        col = int(parent["c0"] + parent_pos)
        if side == "D":
            parent["required"].add((int(parent["r0"] + parent["size"]), int(col)))
            child["required"].add((int(child["r0"] - 1), int(col)))
        else:
            parent["required"].add((int(parent["r0"] - 1), int(col)))
            child["required"].add((int(child["r0"] + child["size"]), int(col)))
        return child
    return None


def _build_layout(connected_count: int, isolated_count: int) -> tuple[list[dict], int] | None:
    total = int(connected_count + isolated_count)
    sizes = [int(_RNG.choice(SIZES)) for _ in range(int(total))]
    if total >= 2 and _RNG.random() < 0.5:
        sizes[0] = 5
        sizes[1] = 7
        _RNG.shuffle(sizes)

    colors = list(_RNG.choice(PALETTE, size=int(total), replace=False))
    specs: list[dict] = [
        {
            "size": int(sizes[0]),
            "color": int(colors[0]),
            "r0": 0,
            "c0": 0,
            "required": set(),
            "used_sides": set(),
        }
    ]

    for idx in range(1, int(connected_count)):
        placed = False
        parents = list(range(len(specs)))
        _RNG.shuffle(parents)
        for parent_idx in parents:
            parent = specs[int(parent_idx)]
            free_sides = [side for side in ("U", "R", "D", "L") if side not in parent["used_sides"]]
            _RNG.shuffle(free_sides)
            for side in free_sides:
                snapshot_required = set(parent["required"])
                child = _attach_spec(parent, int(sizes[idx]), int(colors[idx]), str(side))
                if child is None:
                    parent["required"] = snapshot_required
                    continue
                if not _placement_ok(child, specs, int(parent_idx)):
                    parent["required"] = snapshot_required
                    continue
                parent["used_sides"].add(str(side))
                specs.append(child)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    for idx in range(int(connected_count), int(total)):
        placed = False
        boxes = [_outer_bbox(spec) for spec in specs]
        min_r = min(int(box[0]) for box in boxes)
        max_r = max(int(box[2]) for box in boxes)
        min_c = min(int(box[1]) for box in boxes)
        max_c = max(int(box[3]) for box in boxes)
        span_r = int(max_r - min_r + 1)
        span_c = int(max_c - min_c + 1)
        for _ in range(240):
            r0 = int(_RNG.integers(int(min_r - span_r - 4), int(max_r + 5)))
            c0 = int(_RNG.integers(int(min_c - span_c - 4), int(max_c + 5)))
            spec = {
                "size": int(sizes[idx]),
                "color": int(colors[idx]),
                "r0": int(r0),
                "c0": int(c0),
                "required": set(),
                "used_sides": set(),
            }
            if _placement_ok(spec, specs, None):
                specs.append(spec)
                placed = True
                break
        if not placed:
            return None

    for spec in specs:
        teeth = set(spec["required"])
        density = float(_RNG.uniform(0.25, 0.60))
        for side in ("U", "R", "D", "L"):
            coords = _edge_slots(int(spec["r0"]), int(spec["c0"]), int(spec["size"]), str(side))
            if side in spec["used_sides"]:
                continue
            for coord in coords:
                if _RNG.random() < density:
                    teeth.add((int(coord[0]), int(coord[1])))
        if not teeth:
            slots = _clockwise_slots(int(spec["r0"]), int(spec["c0"]), int(spec["size"]))
            coord = slots[int(_RNG.integers(0, int(len(slots))))]
            teeth.add((int(coord[0]), int(coord[1])))
        spec["teeth"] = teeth

    driver = int(_RNG.integers(0, int(connected_count)))
    return specs, int(driver)


def _driver_red_cells(spec: dict) -> tuple[set[tuple[int, int]], int]:
    interior = [
        (int(r), int(c))
        for r in range(int(spec["r0"] + 1), int(spec["r0"] + spec["size"] - 1))
        for c in range(int(spec["c0"] + 1), int(spec["c0"] + spec["size"] - 1))
    ]
    limit = int(len(interior))
    while True:
        count = int(_RNG.integers(1, int(limit + 1)))
        if not (int(spec["size"]) == 7 and int(count) % 20 == 0):
            break
    chosen = _RNG.choice(len(interior), size=int(count), replace=False)
    return {interior[int(i)] for i in chosen}, int(count)


def _render_grid(specs: list[dict], driver: int, red_cells: set[tuple[int, int]]) -> np.ndarray:
    all_points: list[tuple[int, int]] = []
    for spec in specs:
        all_points.extend(_frame_border_coords(int(spec["r0"]), int(spec["c0"]), int(spec["size"])))
        all_points.extend((int(r), int(c)) for r, c in spec["teeth"])
    all_points.extend((int(r), int(c)) for r, c in red_cells)

    min_r = min(int(r) for r, _c in all_points)
    max_r = max(int(r) for r, _c in all_points)
    min_c = min(int(c) for _r, c in all_points)
    max_c = max(int(c) for _r, c in all_points)

    pad_top = int(_RNG.integers(1, 4))
    pad_bottom = int(_RNG.integers(1, 4))
    pad_left = int(_RNG.integers(1, 4))
    pad_right = int(_RNG.integers(1, 4))
    shift_r = int(pad_top - min_r)
    shift_c = int(pad_left - min_c)
    height = int(max_r - min_r + 1 + pad_top + pad_bottom)
    width = int(max_c - min_c + 1 + pad_left + pad_right)
    grid = np.full((int(height), int(width)), BG, dtype=int)

    for spec in specs:
        color = int(spec["color"])
        for r, c in _frame_border_coords(int(spec["r0"]), int(spec["c0"]), int(spec["size"])):
            grid[int(r + shift_r), int(c + shift_c)] = int(color)
        for r, c in spec["teeth"]:
            grid[int(r + shift_r), int(c + shift_c)] = int(color)

    for r, c in red_cells:
        grid[int(r + shift_r), int(c + shift_c)] = RED

    return grid


def _gen_attempt() -> np.ndarray | None:
    connected_count = int(_RNG.integers(1, 5))
    isolated_count = int(_RNG.integers(0, 2))
    total = int(connected_count + isolated_count)
    if total > len(PALETTE):
        return None

    layout = _build_layout(int(connected_count), int(isolated_count))
    if layout is None:
        return None
    specs, driver = layout
    red_cells, step_count = _driver_red_cells(specs[int(driver)])
    grid = _render_grid(specs, int(driver), red_cells)
    if int(grid.shape[0]) > MAX_GRID_SIZE or int(grid.shape[1]) > MAX_GRID_SIZE:
        return None

    try:
        gears, parsed_driver = _parse_gears(grid)
        graph = _meshing_graph(gears)
        turns = _turn_directions(graph, int(parsed_driver))
        out = main(grid)
    except ValueError:
        return None

    if len(gears) != int(total):
        return None
    if int(len(turns)) != int(connected_count):
        return None
    if int(gears[int(parsed_driver)].red_count) != int(step_count):
        return None

    stripped = grid.copy()
    stripped[stripped == RED] = BG
    if np.array_equal(out, stripped):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after retries")
