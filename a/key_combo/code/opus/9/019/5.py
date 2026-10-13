from __future__ import annotations

from importlib.machinery import SourceFileLoader
from itertools import combinations
from pathlib import Path
import hashlib

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_019_5",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
neighbors4 = _ARC_UTILS.neighbors4


BG = 0
PATH = 3
MARKER_POOL = [1, 2, 4, 5, 6, 7, 8, 9]
DISTANCE_VALUES = tuple(range(2, 9))
DISTANCE_SETS = [tuple(combo) for combo in combinations(DISTANCE_VALUES, 4)]
DISTANCE_SUM_BUCKETS: dict[int, list[tuple[int, int, int, int]]] = {}
for combo in DISTANCE_SETS:
    DISTANCE_SUM_BUCKETS.setdefault(sum(combo), []).append(combo)
TIE_DISTANCE_PAIRS = [
    (a, b)
    for combos in DISTANCE_SUM_BUCKETS.values()
    for a, b in combinations(combos, 2)
]

_RNG = np.random.default_rng()
_SEEN_INPUTS: set[str] = set()


def _sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(map(int, grid.shape))).encode()).hexdigest()


def _boundary(coord: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = coord
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _parse_components(grid: np.ndarray) -> list[dict[str, object]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    records: list[dict[str, object]] = []
    for comp in comps:
        coords = [tuple(map(int, coord)) for coord in comp["coords"]]
        marker_cells = [coord for coord in coords if int(grid[coord]) != PATH]
        path_cells = [coord for coord in coords if int(grid[coord]) == PATH]
        assert len(marker_cells) == 1
        assert len(path_cells) >= 2

        marker = marker_cells[0]
        marker_color = int(grid[marker])
        path_set = set(path_cells)
        degree = {
            cell: sum((nbr in path_set) for nbr in neighbors4(*cell))
            for cell in path_cells
        }
        assert all(1 <= d <= 2 for d in degree.values())
        endpoints = [cell for cell, d in degree.items() if d == 1]
        assert len(endpoints) == 2

        far_candidates = [cell for cell in endpoints if abs(cell[0] - marker[0]) + abs(cell[1] - marker[1]) == 1]
        assert len(far_candidates) == 1
        far_end = far_candidates[0]
        gate_candidates = [cell for cell in endpoints if cell != far_end]
        assert len(gate_candidates) == 1
        gate = gate_candidates[0]
        assert _boundary(gate, grid.shape)

        parking = [nbr for nbr in neighbors4(*gate) if nbr in path_set]
        assert len(parking) == 1
        stop = tuple(map(int, parking[0]))
        distance = len(path_cells) - 1
        assert 2 <= distance <= 8

        records.append(
            {
                "marker": marker,
                "marker_color": marker_color,
                "gate": gate,
                "stop": stop,
                "distance": distance,
                "path_cells": path_set,
            }
        )
    return records


def _frame_pixels(stops: list[tuple[int, int]]) -> set[tuple[int, int]]:
    rows = [r for r, _c in stops]
    cols = [c for _r, c in stops]
    r0 = min(rows) + 1
    r1 = max(rows) - 1
    c0 = min(cols) + 1
    c1 = max(cols) - 1
    assert r1 - r0 + 1 >= 3
    assert c1 - c0 + 1 >= 3
    pixels: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pixels.add((r0, c))
        pixels.add((r1, c))
    for r in range(r0, r1 + 1):
        pixels.add((r, c0))
        pixels.add((r, c1))
    return pixels


def _group_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    per_color: dict[int, list[dict[str, object]]] = {}
    for record in records:
        color = int(record["marker_color"])
        per_color.setdefault(color, []).append(record)

    groups: list[dict[str, object]] = []
    for color, items in per_color.items():
        assert len(items) == 4
        stops = [tuple(map(int, item["stop"])) for item in items]
        avg_distance = sum(int(item["distance"]) for item in items) / 4.0
        first_marker = min(tuple(map(int, item["marker"])) for item in items)
        groups.append(
            {
                "color": color,
                "items": items,
                "stops": stops,
                "avg_distance": avg_distance,
                "first_marker": first_marker,
                "frame_pixels": _frame_pixels(stops),
            }
        )
    return groups


def _solve_from_records(grid: np.ndarray, records: list[dict[str, object]]) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    for record in records:
        marker = tuple(map(int, record["marker"]))
        stop = tuple(map(int, record["stop"]))
        gate = tuple(map(int, record["gate"]))
        marker_color = int(record["marker_color"])
        out[marker] = BG
        for cell in record["path_cells"]:
            out[cell] = BG
        out[gate] = PATH
        out[stop] = marker_color

    groups = _group_records(records)
    groups.sort(
        key=lambda group: (
            float(group["avg_distance"]),
            tuple(map(int, group["first_marker"])),
        ),
        reverse=True,
    )
    for group in groups:
        color = int(group["color"])
        for r, c in group["frame_pixels"]:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored marker to its gate stop, then use the stop extrema to draw overlapping inset frames.

    Input structure:
    - A black background contains several disconnected foreground components made of green single-cell paths and one non-black, non-green marker pixel each.
    - Every path ends at a boundary gate center: a green endpoint whose neighboring black cells form the visible black-green-black gate motif.
    - Marker colors appear in 2 or 3 groups, with exactly 4 markers of each color.

    Rule:
    - For each component, move the marker along its green path until it reaches the unique path cell adjacent to the gate center, and erase the original marker plus all traversed green cells while keeping the gate center green.
    - Record the slide distance as the number of green path cells traversed, which equals the component's green-cell count minus the preserved gate center.
    - Group the moved markers by color; each color's 4 stop positions define row/column extrema, and a one-cell-wide rectangular frame is drawn on the inset bbox from (min_row+1, min_col+1) to (max_row-1, max_col-1).
    - When frame pixels from different colors overlap, the group with shorter average slide distance overwrites the longer one; if averages tie, the color whose marker appears first in reading order in the input overwrites the later one.

    Key insight:
    - The slide operation does double duty: it determines the final marker anchors used for the frame extrema and also produces the path-length statistic that resolves frame occlusion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert int(get_bg(g)) == BG
    records = _parse_components(g)
    groups = _group_records(records)
    assert 2 <= len(groups) <= 3
    return _solve_from_records(g, records)


def _take_pair(available: set[int]) -> list[int]:
    candidates = [
        tuple(pair)
        for pair in combinations(sorted(available), 2)
        if abs(pair[0] - pair[1]) >= 2
    ]
    if not candidates:
        raise ValueError("failed to sample a spaced pair")
    pair = list(map(int, candidates[int(_RNG.integers(len(candidates)))]))
    blocked = {value + delta for value in pair for delta in (-1, 0, 1)}
    for value in blocked:
        available.discard(value)
    return pair


def _assign_distances(num_groups: int, tie_case: bool) -> list[tuple[int, int, int, int]]:
    if tie_case:
        first, second = TIE_DISTANCE_PAIRS[int(_RNG.integers(len(TIE_DISTANCE_PAIRS)))]
        choices = [tuple(map(int, first)), tuple(map(int, second))]
        while len(choices) < num_groups:
            candidate = DISTANCE_SETS[int(_RNG.integers(len(DISTANCE_SETS)))]
            if candidate not in choices:
                choices.append(tuple(map(int, candidate)))
        _RNG.shuffle(choices)
        return choices

    picked: list[tuple[int, int, int, int]] = []
    used_avgs: set[float] = set()
    for candidate in _RNG.permutation(len(DISTANCE_SETS)):
        combo = tuple(map(int, DISTANCE_SETS[int(candidate)]))
        avg = sum(combo) / 4.0
        if avg in used_avgs:
            continue
        picked.append(combo)
        used_avgs.add(avg)
        if len(picked) == num_groups:
            return picked
    raise ValueError("failed to assign distance sets")


def _build_group_specs(h: int, w: int) -> list[dict[str, object]]:
    num_groups = 2 if int(_RNG.integers(0, 2)) == 0 else 3
    colors = list(map(int, _RNG.choice(MARKER_POOL, size=num_groups, replace=False)))
    types = ["LR", "TB"]
    if num_groups == 3:
        types.append("LR" if int(_RNG.integers(0, 2)) == 0 else "TB")
    _RNG.shuffle(types)

    tie_case = bool(num_groups >= 2 and int(_RNG.integers(0, 4)) == 0)
    distance_sets = _assign_distances(num_groups, tie_case)

    left_available = set(range(10, h - 10))
    right_available = set(range(10, h - 10))
    top_available = set(range(10, w - 10))
    bottom_available = set(range(10, w - 10))
    specs: list[dict[str, object]] = []
    for color, group_type, distance_set in zip(colors, types, distance_sets):
        shuffled_distances = list(map(int, distance_set))
        _RNG.shuffle(shuffled_distances)
        if group_type == "LR":
            lrows = _take_pair(left_available)
            rrows = _take_pair(right_available)
            stops = [
                (int(lrows[0]), 1, "L"),
                (int(lrows[1]), 1, "L"),
                (int(rrows[0]), w - 2, "R"),
                (int(rrows[1]), w - 2, "R"),
            ]
        else:
            tcols = _take_pair(top_available)
            bcols = _take_pair(bottom_available)
            stops = [
                (1, int(tcols[0]), "T"),
                (1, int(tcols[1]), "T"),
                (h - 2, int(bcols[0]), "B"),
                (h - 2, int(bcols[1]), "B"),
            ]

        rows = [r for r, _c, _side in stops]
        cols = [c for _r, c, _side in stops]
        if max(rows) - min(rows) < 4 or max(cols) - min(cols) < 4:
            raise ValueError("degenerate frame extrema")

        specs.append(
            {
                "color": int(color),
                "type": group_type,
                "stops": stops,
                "distances": shuffled_distances,
            }
        )
    return specs


def _place_component(
    grid: np.ndarray,
    color: int,
    stop: tuple[int, int, str],
    distance: int,
    records: list[dict[str, object]],
) -> None:
    h, w = grid.shape
    r, c, side = stop
    path_cells: list[tuple[int, int]] = []
    if side == "L":
        gate = (r, 0)
        path_cells = [(r, cc) for cc in range(1, distance + 1)]
        marker = (r, distance + 1)
    elif side == "R":
        gate = (r, w - 1)
        path_cells = [(r, cc) for cc in range(w - distance - 1, w - 1)]
        marker = (r, w - distance - 2)
    elif side == "T":
        gate = (0, c)
        path_cells = [(rr, c) for rr in range(1, distance + 1)]
        marker = (distance + 1, c)
    else:
        gate = (h - 1, c)
        path_cells = [(rr, c) for rr in range(h - distance - 1, h - 1)]
        marker = (h - distance - 2, c)

    coords = [gate, marker, *path_cells]
    for rr, cc in coords:
        assert 0 <= rr < h and 0 <= cc < w
        assert int(grid[rr, cc]) == BG

    grid[gate] = PATH
    for rr, cc in path_cells:
        grid[rr, cc] = PATH
    grid[marker] = int(color)
    records.append(
        {
            "marker": marker,
            "marker_color": int(color),
            "gate": gate,
            "stop": (r, c),
            "distance": int(distance),
            "path_cells": set(path_cells) | {gate},
        }
    )


def _check_overlap(groups: list[dict[str, object]]) -> bool:
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i]["frame_pixels"] & groups[j]["frame_pixels"]:
                return True
    return False


def _priority_tuple(group: dict[str, object]) -> tuple[float, tuple[int, int]]:
    return (float(group["avg_distance"]), tuple(map(int, group["first_marker"])))


def _validate_case(inp: np.ndarray, spec_records: list[dict[str, object]]) -> bool:
    try:
        parsed = _parse_components(inp)
    except AssertionError:
        return False

    parsed_key = sorted(
        (
            int(record["marker_color"]),
            tuple(map(int, record["marker"])),
            tuple(map(int, record["gate"])),
            tuple(map(int, record["stop"])),
            int(record["distance"]),
        )
        for record in parsed
    )
    spec_key = sorted(
        (
            int(record["marker_color"]),
            tuple(map(int, record["marker"])),
            tuple(map(int, record["gate"])),
            tuple(map(int, record["stop"])),
            int(record["distance"]),
        )
        for record in spec_records
    )
    if parsed_key != spec_key:
        return False

    groups = _group_records(parsed)
    if not (2 <= len(groups) <= 3):
        return False
    if not _check_overlap(groups):
        return False

    expected = _solve_from_records(inp, spec_records)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    pixel_owners: dict[tuple[int, int], tuple[tuple[float, tuple[int, int]], int]] = {}
    for group in groups:
        priority = _priority_tuple(group)
        color = int(group["color"])
        for cell in group["frame_pixels"]:
            previous = pixel_owners.get(cell)
            if previous is None or priority < previous[0]:
                pixel_owners[cell] = (priority, color)
    for cell, (_priority, color) in pixel_owners.items():
        if int(actual[cell]) != color:
            return False
    return True


def _generate_case() -> np.ndarray:
    h = int(_RNG.integers(28, 31))
    w = int(_RNG.integers(28, 31))
    grid = np.zeros((h, w), dtype=int)
    specs = _build_group_specs(h, w)
    records: list[dict[str, object]] = []
    for spec in specs:
        for stop, distance in zip(spec["stops"], spec["distances"]):
            _place_component(grid, int(spec["color"]), stop, int(distance), records)

    if not _validate_case(grid, records):
        raise ValueError("semantic validation failed")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid = _generate_case()
        except (AssertionError, ValueError):
            continue
        signature = _sig(grid)
        if signature in _SEEN_INPUTS:
            continue
        _SEEN_INPUTS.add(signature)
        return grid
    raise RuntimeError("failed to generate a valid input")


if __name__ == "__main__":
    target = Path(__file__)
    print(target)
