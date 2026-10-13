import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
RAYS = ("left", "right", "up", "down")

GRID_MIN = 11
GRID_MAX = 18
BAR_COUNT_CHOICES = (3, 4)
MAX_BAR_LENGTH = 4
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _rotate_cw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _rotate_ccw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _rotate_steps_cw(src: tuple[int, int], dst: tuple[int, int]) -> int:
    cur = src
    for k in range(4):
        if cur == dst:
            return k
        cur = _rotate_cw(cur)
    raise AssertionError("direction pair is not cardinal")


def _line_cells(pivot: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    pr, pc = pivot
    dr, dc = direction
    return [(pr + k * dr, pc + k * dc) for k in range(1, length + 1)]


def _space_to_edge(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = start
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _desired_direction(pivot: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    pr, pc = pivot
    tr, tc = target
    if pr == tr and pc != tc:
        return 0, 1 if tc > pc else -1
    if pc == tc and pr != tr:
        return 1 if tr > pr else -1, 0
    raise AssertionError("pivot must share a row or column with the target")


def _extract_markers(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    yellow = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]
    assert len(yellow) == 2, "expected exactly two yellow edge markers"

    top_bottom = [(r, c) for r, c in yellow if r in (0, h - 1) and 0 < c < w - 1]
    left_right = [(r, c) for r, c in yellow if c in (0, w - 1) and 0 < r < h - 1]
    assert len(top_bottom) == 1 and len(left_right) == 1, "expected one top/bottom and one left/right marker"

    target = (left_right[0][0], top_bottom[0][1])
    assert int(g[target]) == BLACK, "the virtual target intersection must be empty"
    return target, top_bottom[0], left_right[0]


def _red_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    red = {tuple(map(int, rc)) for rc in np.argwhere(np.asarray(grid, dtype=int) == RED)}
    comps: list[set[tuple[int, int]]] = []
    while red:
        start = red.pop()
        comp = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in red:
                    red.remove((nr, nc))
                    comp.add((nr, nc))
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _parse_pointers(grid: np.ndarray, target: tuple[int, int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    pivots = [tuple(map(int, rc)) for rc in np.argwhere(g == GREY)]
    assert pivots, "expected at least one grey pivot"

    components = _red_components(g)
    assert components, "expected at least one red pointer bar"
    comp_of: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(components):
        for cell in comp:
            comp_of[cell] = idx

    seen_components: set[int] = set()
    pointers = []
    for pivot in pivots:
        pr, pc = pivot
        adjacent = []
        for nr, nc in _neighbors4(pr, pc):
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == RED:
                adjacent.append((nr, nc))
        assert len(adjacent) == 1, "each pivot must touch exactly one end of one red bar"

        first = adjacent[0]
        comp_idx = comp_of[first]
        assert comp_idx not in seen_components, "each red component must belong to exactly one pivot"
        dr = first[0] - pr
        dc = first[1] - pc

        cells = []
        k = 1
        while True:
            rr = pr + k * dr
            cc = pc + k * dc
            if not _in_bounds(g.shape, rr, cc) or int(g[rr, cc]) != RED:
                break
            cells.append((rr, cc))
            k += 1

        assert cells, "bar must have positive length"
        assert set(cells) == components[comp_idx], "each red component must be a straight contiguous bar"

        want = _desired_direction(pivot, target)
        pointers.append(
            {
                "pivot": pivot,
                "input_dir": (dr, dc),
                "output_dir": want,
                "length": len(cells),
                "input_cells": cells,
                "rotation_cw": _rotate_steps_cw((dr, dc), want),
            }
        )
        seen_components.add(comp_idx)

    assert len(seen_components) == len(components), "every red bar must attach to one pivot"
    return pointers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each red pointer around its grey pivot so it aims at the virtual target defined by the two edge markers.

    Input structure:
    - Black background with exactly two yellow edge markers: one on the top or
      bottom edge fixing a column, and one on the left or right edge fixing a row.
    - Their row/column projection intersection is empty and acts as a virtual target.
    - Several straight red bars each touch one grey pivot at exactly one end, and
      every pivot lies on the target row or the target column.

    Rule:
    - Infer the target from the row of the side marker and the column of the top/bottom marker.
    - For each pivoted red bar, keep the pivot fixed and rotate the bar by 0, 1, 2,
      or 3 quarter-turns until its direction from the pivot points toward that target.
    - Keep the bar length unchanged, clear the old red cells, and leave the yellow
      markers and grey pivots unchanged.

    Key insight:
    - The target is never drawn directly; it must be reconstructed from the two edge
      markers, and each bar's final direction is determined independently by the
      pivot-target alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    target, _tb_marker, _lr_marker = _extract_markers(g)
    pointers = _parse_pointers(g, target)

    out = g.copy()
    out[out == RED] = BLACK
    for pointer in pointers:
        for rr, cc in _line_cells(pointer["pivot"], pointer["output_dir"], pointer["length"]):
            assert _in_bounds(out.shape, rr, cc), "rotated bar must stay on the canvas"
            assert int(out[rr, cc]) not in (YELLOW, GREY), "rotated bar cannot overwrite markers or pivots"
            out[rr, cc] = RED
    return out


def _ray_pivot(target: tuple[int, int], ray: str, distance: int) -> tuple[int, int]:
    tr, tc = target
    if ray == "left":
        return tr, tc - distance
    if ray == "right":
        return tr, tc + distance
    if ray == "up":
        return tr - distance, tc
    if ray == "down":
        return tr + distance, tc
    raise AssertionError("unknown ray")


def _ray_goal_direction(ray: str) -> tuple[int, int]:
    if ray == "left":
        return 0, 1
    if ray == "right":
        return 0, -1
    if ray == "up":
        return 1, 0
    if ray == "down":
        return -1, 0
    raise AssertionError("unknown ray")


def _ray_max_distance(shape: tuple[int, int], target: tuple[int, int], ray: str) -> int:
    h, w = shape
    tr, tc = target
    if ray == "left":
        return tc - 1
    if ray == "right":
        return w - 2 - tc
    if ray == "up":
        return tr - 1
    if ray == "down":
        return h - 2 - tr
    raise AssertionError("unknown ray")


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _build_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == RED] = BLACK
    for spec in specs:
        _paint(out, _line_cells(spec["pivot"], spec["output_dir"], spec["length"]), RED)
    return out


def _validate_generated(inp: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    try:
        target, _tb, _lr = _extract_markers(inp)
        parsed = _parse_pointers(inp, target)
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if len(parsed) != len(specs):
        return False
    if len(parsed) not in BAR_COUNT_CHOICES:
        return False
    if int(inp[target]) != BLACK or int(expected[target]) != BLACK:
        return False

    spec_by_pivot = {spec["pivot"]: spec for spec in specs}
    parsed_pivots = {item["pivot"] for item in parsed}
    if parsed_pivots != set(spec_by_pivot):
        return False

    rotations = []
    output_dirs = set()
    for item in parsed:
        spec = spec_by_pivot[item["pivot"]]
        if item["length"] != spec["length"]:
            return False
        if item["input_dir"] != spec["input_dir"]:
            return False
        if item["output_dir"] != spec["output_dir"]:
            return False
        rotations.append(item["rotation_cw"])
        output_dirs.add(item["output_dir"])

    if len(set(rotations)) != len(rotations):
        return False
    if len(output_dirs) < 3:
        return False
    if all(rot == 0 for rot in rotations):
        return False

    out_red = {tuple(map(int, rc)) for rc in np.argwhere(expected == RED)}
    if target in out_red:
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    target = (rng.randint(3, h - 4), rng.randint(3, w - 4))

    n_bars = 4 if rng.random() < 0.6 else 3
    rays = rng.sample(RAYS, n_bars)
    rotations = rng.sample([0, 1, 2, 3], n_bars)
    rng.shuffle(rays)

    grid = np.full((h, w), BLACK, dtype=int)
    tb_marker = (0 if rng.random() < 0.5 else h - 1, target[1])
    lr_marker = (target[0], 0 if rng.random() < 0.5 else w - 1)
    grid[tb_marker] = YELLOW
    grid[lr_marker] = YELLOW

    specs = []
    occupied_in = {tb_marker, lr_marker}
    occupied_out = {tb_marker, lr_marker}

    for ray, rot in zip(rays, rotations):
        desired = _ray_goal_direction(ray)
        input_dir = desired
        for _ in range(rot):
            input_dir = _rotate_ccw(input_dir)

        max_dist = _ray_max_distance((h, w), target, ray)
        distances = list(range(2, max_dist + 1))
        rng.shuffle(distances)
        placed = False
        for distance in distances:
            pivot = _ray_pivot(target, ray, distance)
            if pivot in occupied_in:
                continue

            max_len = min(
                MAX_BAR_LENGTH,
                distance - 1,
                _space_to_edge((h, w), pivot, input_dir),
            )
            if max_len < 1:
                continue

            lengths = list(range(1, max_len + 1))
            rng.shuffle(lengths)
            if max_len > 1 and rng.random() < 0.5:
                lengths = sorted(lengths, reverse=True)

            for length in lengths:
                input_cells = _line_cells(pivot, input_dir, length)
                output_cells = _line_cells(pivot, desired, length)
                if target in input_cells or target in output_cells:
                    continue
                if any(cell in occupied_in for cell in input_cells):
                    continue
                if any(cell in occupied_out for cell in output_cells):
                    continue
                if any(not _in_bounds((h, w), r, c) for r, c in input_cells + output_cells):
                    continue

                grid[pivot] = GREY
                _paint(grid, input_cells, RED)
                occupied_in.add(pivot)
                occupied_in.update(input_cells)
                occupied_out.add(pivot)
                occupied_out.update(output_cells)
                specs.append(
                    {
                        "pivot": pivot,
                        "input_dir": input_dir,
                        "output_dir": desired,
                        "length": length,
                        "ray": ray,
                        "rotation_cw": rot,
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    expected = _build_expected(grid, specs)
    if not _validate_generated(grid, expected, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt(random.Random())
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid input")
