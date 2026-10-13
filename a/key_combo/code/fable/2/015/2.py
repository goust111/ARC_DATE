from __future__ import annotations

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_arc_utils_fable_2_015_2", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"unable to load arc_utils from {_ARC_UTILS_PATH}")
_arc_utils = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_arc_utils)

Color = _arc_utils.Color
find_components = _arc_utils.find_components
get_bg = _arc_utils.get_bg


BG = Color.BLACK
PIVOT = Color.GREY
SOURCE_ARM = Color.RED
TRAIL = Color.GREEN
COUNTER = Color.YELLOW

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
NEIGHBORS4 = (UP, RIGHT, DOWN, LEFT)

GRID_SIZE_RANGE = (18, 30)
DEVICE_COUNT_RANGE = (2, 4)
ARM_LENGTH_RANGE = (2, 6)
COUNTER_HEIGHT_OPTIONS = {
    0: (4, 8),
    1: (5, 9),
    2: (6, 10),
    3: (7, 11),
}
SOURCE_DIRECTIONS_BY_TURNS = {
    0: (UP, LEFT, RIGHT),
    1: (UP, LEFT),
    2: (LEFT, RIGHT),
    3: (UP, RIGHT),
}
PLACEMENT_ATTEMPTS = 400
GENERATION_RETRIES = 400


@dataclass(frozen=True)
class Device:
    pivot: tuple[int, int]
    direction: tuple[int, int]
    length: int
    counter_height: int

    @property
    def turns(self) -> int:
        return self.counter_height % 4


def _rotate_cw(direction: tuple[int, int], turns: int) -> tuple[int, int]:
    dr, dc = direction
    for _ in range(turns % 4):
        dr, dc = dc, -dr
    return int(dr), int(dc)


def _arm_coords(
    pivot: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    pr, pc = pivot
    dr, dc = direction
    return [(pr + step * dr, pc + step * dc) for step in range(1, length + 1)]


def _counter_coords(pivot: tuple[int, int], height: int) -> list[tuple[int, int]]:
    pr, pc = pivot
    return [(pr + offset, pc) for offset in range(2, height + 2)]


def _line_coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, coord)) for coord in coords}


def _validate_straight_bar(comp: dict, *, vertical_only: bool = False) -> set[tuple[int, int]]:
    coords = _line_coord_set(comp["coords"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if vertical_only:
        assert c0 == c1, "counter bar must be vertical"
        expected = {(r, c0) for r in range(r0, r1 + 1)}
    elif r0 == r1:
        expected = {(r0, c) for c in range(c0, c1 + 1)}
    else:
        assert c0 == c1, "arm must be a straight horizontal or vertical bar"
        expected = {(r, c0) for r in range(r0, r1 + 1)}
    assert coords == expected, "bar cells must be contiguous"
    return coords


def _extract_devices(grid: np.ndarray) -> list[Device]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    pivot_comps = find_components(
        g, bg=bg, connectivity=4, monochromatic=True, colors=[PIVOT]
    )
    red_comps = find_components(
        g, bg=bg, connectivity=4, monochromatic=True, colors=[SOURCE_ARM]
    )
    yellow_comps = find_components(
        g, bg=bg, connectivity=4, monochromatic=True, colors=[COUNTER]
    )

    assert pivot_comps, "expected at least one grey pivot"
    assert len(red_comps) == len(pivot_comps), "each pivot needs one red arm"
    assert len(yellow_comps) == len(pivot_comps), "each pivot needs one counter bar"

    red_sets: list[set[tuple[int, int]]] = []
    red_lookup: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(red_comps):
        coords = _validate_straight_bar(comp)
        for coord in coords:
            red_lookup[coord] = idx
        red_sets.append(coords)

    yellow_sets: list[set[tuple[int, int]]] = []
    yellow_lookup: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(yellow_comps):
        coords = _validate_straight_bar(comp, vertical_only=True)
        assert comp["area"] >= 4, "counter bar height must be at least four"
        for coord in coords:
            yellow_lookup[coord] = idx
        yellow_sets.append(coords)

    used_red: set[int] = set()
    used_yellow: set[int] = set()
    devices: list[Device] = []
    for comp in pivot_comps:
        assert comp["area"] == 1, "pivot marker must be one pixel"
        pr, pc = map(int, comp["coords"][0])

        assert 0 <= pr < H and 0 <= pc < W
        assert pr + 2 < H, "counter bar must fit below the pivot"
        assert int(g[pr + 1, pc]) == bg, "pivot and counter bar must be separated by one black cell"

        yellow_idx = yellow_lookup.get((pr + 2, pc))
        assert yellow_idx is not None, "expected a counter bar directly below the pivot"
        assert yellow_idx not in used_yellow, "counter bar may belong to only one pivot"
        yellow_comp = yellow_comps[yellow_idx]
        yr0, yc0, _yr1, yc1 = map(int, yellow_comp["bbox"])
        assert yc0 == yc1 == pc and yr0 == pr + 2
        counter_height = int(yellow_comp["area"])

        red_neighbors = []
        for dr, dc in NEIGHBORS4:
            nr = pr + dr
            nc = pc + dc
            if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == SOURCE_ARM:
                red_neighbors.append((nr, nc))
        assert len(red_neighbors) == 1, "pivot must touch exactly one red arm cell"

        near = red_neighbors[0]
        direction = (near[0] - pr, near[1] - pc)
        red_idx = red_lookup.get(near)
        assert red_idx is not None, "adjacent red cell must belong to a red component"
        assert red_idx not in used_red, "red arm may belong to only one pivot"

        red_coords = red_sets[red_idx]
        length = len(red_coords)
        assert red_coords == set(_arm_coords((pr, pc), direction, length)), "red arm must extend straight from the pivot"

        used_red.add(red_idx)
        used_yellow.add(yellow_idx)
        devices.append(Device((pr, pc), direction, length, counter_height))

    assert len(used_red) == len(red_comps), "every red arm must be paired with a pivot"
    assert len(used_yellow) == len(yellow_comps), "every counter bar must be paired with a pivot"
    return sorted(devices, key=lambda device: device.pivot)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each red arm clockwise around its grey pivot by the yellow counter height modulo four, leaving a green trail.

    Input structure:
    - A black background contains several independent devices.
    - Each device has one grey pivot pixel, one straight red arm attached to the pivot, and one vertical yellow counter bar.
    - The yellow bar starts two rows below the pivot, so one black cell separates pivot and counter.

    Rule:
    - Read the yellow counter height h for each device and compute h mod 4.
    - Recolor the original red arm cells green.
    - Draw an equal-length red arm from the same pivot after rotating the original arm clockwise by that many quarter-turns.
    - Keep the grey pivot and yellow counter bar unchanged; if the new red arm overlaps the green trail, red stays visible.

    Key insight:
    - The yellow column is not scenery: its height encodes the number of clockwise quarter-turns, with heights like 4 and 8 meaning a full wraparound.
    """
    g = np.asarray(grid, dtype=int)
    devices = _extract_devices(g)

    out = g.copy()
    for device in devices:
        for r, c in _arm_coords(device.pivot, device.direction, device.length):
            out[r, c] = TRAIL

    for device in devices:
        rotated_direction = _rotate_cw(device.direction, device.turns)
        for r, c in _arm_coords(device.pivot, rotated_direction, device.length):
            out[r, c] = SOURCE_ARM

    return out


def _device_signature(device: Device) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    return (device.pivot, device.direction, device.length, device.counter_height)


def _relative_core_coords(
    direction: tuple[int, int], turns: int, length: int, height: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    src = {(step * direction[0], step * direction[1]) for step in range(1, length + 1)}
    dst_dir = _rotate_cw(direction, turns)
    dst = {(step * dst_dir[0], step * dst_dir[1]) for step in range(1, length + 1)}
    counter = {(offset, 0) for offset in range(2, height + 2)}
    actual = {(0, 0)} | src | counter
    reserve = actual | dst | {(1, 0)}
    return actual, reserve


def _expand_padding(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                padded.add((r + dr, c + dc))
    return padded


def _paint_device_input(grid: np.ndarray, device: Device) -> None:
    pr, pc = device.pivot
    grid[pr, pc] = PIVOT
    for r, c in _arm_coords(device.pivot, device.direction, device.length):
        grid[r, c] = SOURCE_ARM
    for r, c in _counter_coords(device.pivot, device.counter_height):
        grid[r, c] = COUNTER


def _render_expected(grid: np.ndarray, devices: list[Device]) -> np.ndarray:
    out = grid.copy()
    for device in devices:
        for r, c in _arm_coords(device.pivot, device.direction, device.length):
            out[r, c] = TRAIL
    for device in devices:
        rotated_direction = _rotate_cw(device.direction, device.turns)
        for r, c in _arm_coords(device.pivot, rotated_direction, device.length):
            out[r, c] = SOURCE_ARM
    return out


def _sample_turns(count: int) -> list[int]:
    if count == 2:
        turns = [0, random.choice((1, 2, 3))]
    elif count == 3:
        turns = [0, *random.sample((1, 2, 3), 2)]
    else:
        turns = [0, 1, 2, 3]
    random.shuffle(turns)
    return turns


def _place_devices(H: int, W: int, turns_list: list[int]) -> list[Device] | None:
    occupied: set[tuple[int, int]] = set()
    devices: list[Device] = []

    for turns in turns_list:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            direction = random.choice(SOURCE_DIRECTIONS_BY_TURNS[turns])
            length = random.randint(*ARM_LENGTH_RANGE)
            height = random.choice(COUNTER_HEIGHT_OPTIONS[turns])

            _actual_rel, reserve_rel = _relative_core_coords(direction, turns, length, height)
            reserve_rel = _expand_padding(reserve_rel)
            min_dr = min(dr for dr, _dc in reserve_rel)
            max_dr = max(dr for dr, _dc in reserve_rel)
            min_dc = min(dc for _dr, dc in reserve_rel)
            max_dc = max(dc for _dr, dc in reserve_rel)

            min_pr = -min_dr
            max_pr = H - 1 - max_dr
            min_pc = -min_dc
            max_pc = W - 1 - max_dc
            if min_pr > max_pr or min_pc > max_pc:
                continue

            pr = random.randint(min_pr, max_pr)
            pc = random.randint(min_pc, max_pc)
            reserve_abs = {(pr + dr, pc + dc) for dr, dc in reserve_rel}
            if reserve_abs & occupied:
                continue

            device = Device((pr, pc), direction, length, height)
            devices.append(device)
            occupied |= reserve_abs
            placed = True
            break

        if not placed:
            return None

    if len(devices) > 1 and len({device.direction for device in devices}) == 1:
        return None
    return devices


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    device_count = random.randint(*DEVICE_COUNT_RANGE)
    turns_list = _sample_turns(device_count)
    devices = _place_devices(H, W, turns_list)
    if devices is None:
        return None

    grid = np.full((H, W), BG, dtype=int)
    for device in devices:
        _paint_device_input(grid, device)

    parsed_devices = _extract_devices(grid)
    expected_signatures = {_device_signature(device) for device in devices}
    parsed_signatures = {_device_signature(device) for device in parsed_devices}
    if parsed_signatures != expected_signatures:
        return None

    if not any(device.turns == 0 for device in devices):
        return None
    if not any(device.turns != 0 for device in devices):
        return None

    expected = _render_expected(grid, devices)
    if not np.any(expected == TRAIL):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid counting-pillar rotation input")
