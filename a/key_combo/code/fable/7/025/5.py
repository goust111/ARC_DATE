from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG = Color.BLACK
WATER = Color.BLUE
TERRAIN = Color.GREY

GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class Basin:
    left: int
    right: int
    floor: int
    top_left: int
    top_right: int

    @property
    def width(self) -> int:
        return self.right - self.left - 1

    @property
    def spill_row(self) -> int:
        return max(self.top_left, self.top_right)

    @property
    def spill_side(self) -> str:
        return "left" if self.top_left > self.top_right else "right"

    @property
    def capacity(self) -> int:
        return self.width * (self.floor - self.spill_row)

    @property
    def interior_span(self) -> range:
        return range(self.left + 1, self.right)

    def contains_col(self, c: int) -> bool:
        return self.left < c < self.right

    def overflow_col(self) -> int:
        return self.left - 1 if self.spill_side == "left" else self.right + 1


def _parse_reservoir_bar(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    blue_components = find_components(
        grid, bg=BG, connectivity=4, monochromatic=True, colors=[WATER], return_format="dict"
    )
    bars = []
    for comp in blue_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == r1 and int(comp["area"]) == c1 - c0 + 1:
            bars.append(comp)
    assert len(bars) == 1, "expected exactly one horizontal blue reservoir bar"
    bar = bars[0]
    r0, c0, r1, c1 = map(int, bar["bbox"])
    length = c1 - c0 + 1
    assert length % 2 == 1, "reservoir bar must have odd length"
    return (r0, c0, r1, c1), c0 + length // 2


def _parse_basins(grid: np.ndarray) -> list[Basin]:
    grey_components = find_components(
        grid, bg=BG, connectivity=4, monochromatic=True, colors=[TERRAIN], return_format="dict"
    )
    basins: list[Basin] = []
    for comp in grey_components:
        coords = comp["coords"]
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if c1 - c0 + 1 < 3:
            continue
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if not mask[-1, :].all():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        top_left = r0 + int(np.flatnonzero(mask[:, 0])[0])
        top_right = r0 + int(np.flatnonzero(mask[:, -1])[0])
        if not mask[top_left - r0 :, 0].all():
            continue
        if not mask[top_right - r0 :, -1].all():
            continue
        if mask[:-1, 1:-1].any():
            continue
        basin = Basin(left=c0, right=c1, floor=r1, top_left=top_left, top_right=top_right)
        if basin.width < 2 or basin.capacity <= 0:
            continue
        basins.append(basin)
    basins.sort(key=lambda b: (b.floor, b.left))
    assert basins, "expected at least one basin"
    return basins


def _find_source_basin(basins: list[Basin], inject_col: int) -> Basin:
    candidates = [basin for basin in basins if basin.contains_col(inject_col)]
    assert candidates, "no basin under the injection column"
    return min(candidates, key=lambda basin: basin.floor)


def _find_next_basin(basins: list[Basin], current: Basin) -> Basin | None:
    col = current.overflow_col()
    candidates = [
        basin
        for basin in basins
        if basin.floor > current.floor and basin.contains_col(col)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda basin: basin.floor)


def _paint_layers(out: np.ndarray, basin: Basin, water_units: int) -> None:
    if water_units <= 0:
        return
    width = basin.width
    assert water_units % width == 0, "generator promises whole-layer fills"
    layers = water_units // width
    row_start = basin.floor - layers
    out[row_start : basin.floor, basin.left + 1 : basin.right] = WATER


def _solve_chain(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bar_bbox, inject_col = _parse_reservoir_bar(g)
    basins = _parse_basins(g)
    current = _find_source_basin(basins, inject_col)

    out = g.copy()
    r0, c0, _r1, c1 = bar_bbox
    out[r0, c0 : c1 + 1] = BG

    remaining = c1 - c0 + 1
    while current is not None and remaining > 0:
        fill = min(remaining, current.capacity)
        _paint_layers(out, current, fill)
        remaining -= fill
        if remaining == 0 or fill < current.capacity:
            break
        current = _find_next_basin(basins, current)

    assert remaining == 0, "input promised enough downstream basin capacity"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour the counted water budget from the top reservoir bar into chained basins and draw the final resting water.

    Input structure:
    - Black is background, gray pixels form several separate U-shaped basins with flat floors and unequal rim heights.
    - A single odd-length horizontal blue bar near the top is the reservoir bar; its midpoint column is the injection column.
    - Lower rims have open air beside them, and a deeper basin below that overflow column receives the spill next.

    Rule:
    - Remove the reservoir bar and treat its length as the total water budget.
    - Water falls into the first basin under the injection column, fills that basin by whole horizontal layers, and can retain only up to its lower rim.
    - Once a basin is full to its lower rim, any remaining water spills over that lower side, drops into the next basin below the overflow column, and the same process repeats.

    Key insight:
    - Each basin keeps only its pre-spill capacity; the reservoir bar length is a single global budget that propagates along the overflow chain rather than being decided independently per basin.
    """
    return _solve_chain(grid)


def _draw_basin(grid: np.ndarray, basin: Basin) -> None:
    grid[basin.floor, basin.left : basin.right + 1] = TERRAIN
    grid[basin.top_left : basin.floor + 1, basin.left] = TERRAIN
    grid[basin.top_right : basin.floor + 1, basin.right] = TERRAIN


def _build_chain() -> tuple[list[Basin], int]:
    basin_count = random.randint(2, 4)
    first_floor_max = 10 if basin_count <= 2 else (9 if basin_count == 3 else 8)
    first_floor = random.randint(7, first_floor_max)

    basins_rel: list[Basin] = []
    floors = [first_floor]
    for _ in range(1, basin_count):
        floors.append(floors[-1] + random.randint(4, 6))

    width0 = random.randint(2, 5)
    left0 = 0
    right0 = width0 + 1
    basins_rel.append(Basin(left=left0, right=right0, floor=floors[0], top_left=0, top_right=0))

    chain_sides: list[str] = []
    for i in range(basin_count - 1):
        chain_sides.append(random.choice(["left", "right"]))

    for i in range(1, basin_count):
        prev = basins_rel[-1]
        drop_col = prev.left - 1 if chain_sides[i - 1] == "left" else prev.right + 1
        width = random.randint(2, 5)
        offset_inside = random.randint(1, width)
        left = drop_col - offset_inside
        right = left + width + 1
        basins_rel.append(Basin(left=left, right=right, floor=floors[i], top_left=0, top_right=0))

    min_x = min(b.left for b in basins_rel)
    max_x = max(b.right for b in basins_rel)
    total_width = max_x - min_x + 1
    if total_width > 24:
        raise ValueError("layout too wide")
    shift = random.randint(2 - min_x, 27 - max_x)

    basins: list[Basin] = []
    for i, basin_rel in enumerate(basins_rel):
        left = basin_rel.left + shift
        right = basin_rel.right + shift
        retain_layers = random.randint(2, 5)
        if i < basin_count - 1:
            spill_side = chain_sides[i]
            high_extra = random.randint(1, 3)
            spill_row = basin_rel.floor - retain_layers
            if spill_side == "left":
                top_left = spill_row
                top_right = spill_row - high_extra
            else:
                top_left = spill_row - high_extra
                top_right = spill_row
        else:
            spill_row = basin_rel.floor - retain_layers
            if random.random() < 0.5:
                top_left = spill_row
                top_right = spill_row - random.randint(1, 3)
            else:
                top_left = spill_row - random.randint(1, 3)
                top_right = spill_row
        if min(top_left, top_right) < 2:
            raise ValueError("basin too close to top")
        basins.append(
            Basin(
                left=left,
                right=right,
                floor=basin_rel.floor,
                top_left=top_left,
                top_right=top_right,
            )
        )

    if max(b.floor for b in basins) >= 29:
        raise ValueError("layout too tall")

    source = basins[0]
    inject_col = random.choice(list(source.interior_span))
    return basins, inject_col


def _expected_from_meta(grid: np.ndarray, basins: list[Basin], inject_col: int, budget: int) -> np.ndarray:
    out = grid.copy()
    out[out == WATER] = BG

    current = min((b for b in basins if b.contains_col(inject_col)), key=lambda b: b.floor)
    remaining = budget
    while current is not None and remaining > 0:
        fill = min(current.capacity, remaining)
        _paint_layers(out, current, fill)
        remaining -= fill
        if remaining == 0 or fill < current.capacity:
            break
        current = _find_next_basin(basins, current)
    if remaining != 0:
        raise ValueError("generator exceeded downstream capacity")
    return out


def _generate_case(return_meta: bool = False):
    basins, inject_col = _build_chain()
    height = max(b.floor for b in basins) + 1
    width = max(b.right for b in basins) + 2
    if height > 30 or width > 30:
        raise ValueError("grid too large")

    grid = np.full((height, width), BG, dtype=int)
    for basin in basins:
        _draw_basin(grid, basin)

    source = basins[0]
    exact_fit = random.random() < 0.35
    stop_index = random.randint(0, len(basins) - 1)
    prefix_capacity = sum(b.capacity for b in basins[:stop_index])
    stop = basins[stop_index]
    max_layers = stop.floor - stop.spill_row
    if exact_fit or max_layers == 1:
        layers = max_layers
    else:
        layers = random.randint(1, max_layers - 1)
    budget = prefix_capacity + layers * stop.width

    bar_left = inject_col - budget // 2
    bar_right = inject_col + budget // 2
    if bar_left < 0 or bar_right >= width:
        raise ValueError("reservoir bar out of bounds")
    if source.top_left <= 1 and source.top_right <= 1:
        raise ValueError("source basin too shallow")
    if not source.contains_col(inject_col):
        raise ValueError("invalid injection column")

    grid[0, bar_left : bar_right + 1] = WATER

    expected = _expected_from_meta(grid, basins, inject_col, budget)
    solved = main(grid)
    if not np.array_equal(expected, solved):
        raise ValueError("solver mismatch on generated case")

    if return_meta:
        return grid, {"basins": basins, "inject_col": inject_col, "budget": budget, "expected": expected}
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            return _generate_case(return_meta=False)
        except Exception:
            continue
    raise RuntimeError("failed to generate valid overflow-chain input")
