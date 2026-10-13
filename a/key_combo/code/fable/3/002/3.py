from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import (  # noqa: E402
    Color,
    bbox_coords,
    colors_present,
    components_mask,
    get_bg,
)


BG = int(Color.BLACK)
ANCHOR = int(Color.GREY)
MOVING_COLORS = [
    int(Color.BLUE),
    int(Color.RED),
    int(Color.GREEN),
    int(Color.YELLOW),
    int(Color.PINK),
    int(Color.ORANGE),
    int(Color.TEAL),
    int(Color.MAROON),
]


def _find_anchor(g: np.ndarray) -> tuple[int, int, int, int]:
    comps = components_mask(g == ANCHOR, connectivity=4)
    candidates: list[tuple[int, int, int, int]] = []
    for coords in comps:
        if int(coords.shape[0]) != 4:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 - r0 == 1 and c1 - c0 == 1 and bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == ANCHOR)):
            candidates.append((int(r0), int(c0), int(r1), int(c1)))

    assert len(candidates) == 1, "expected exactly one solid gray 2x2 anchor square"
    assert int(np.count_nonzero(g == ANCHOR)) == 4, "gray may only appear in the anchor square"
    return candidates[0]


def _slider_info(
    coords: np.ndarray, anchor: tuple[int, int, int, int]
) -> dict[str, int | np.ndarray | tuple[int, int, int, int]] | None:
    ar0, ac0, ar1, ac1 = anchor
    r0, c0, r1, c1 = bbox_coords(coords)
    rows = {int(r) for r, _c in coords}
    cols = {int(c) for _r, c in coords}
    anchor_rows = set(range(ar0, ar1 + 1))
    anchor_cols = set(range(ac0, ac1 + 1))

    matches: list[dict[str, int | np.ndarray | tuple[int, int, int, int]]] = []
    if rows <= anchor_rows:
        if c1 < ac0:
            matches.append({"axis": 0, "dr": 0, "dc": 1, "gap": ac0 - c1 - 1, "bbox": (r0, c0, r1, c1)})
        elif c0 > ac1:
            matches.append({"axis": 0, "dr": 0, "dc": -1, "gap": c0 - ac1 - 1, "bbox": (r0, c0, r1, c1)})

    if cols <= anchor_cols:
        if r1 < ar0:
            matches.append({"axis": 1, "dr": 1, "dc": 0, "gap": ar0 - r1 - 1, "bbox": (r0, c0, r1, c1)})
        elif r0 > ar1:
            matches.append({"axis": 1, "dr": -1, "dc": 0, "gap": r0 - ar1 - 1, "bbox": (r0, c0, r1, c1)})

    if not matches:
        return None
    assert len(matches) == 1, "a colored component must not be aligned as both row and column slider"
    matches[0]["coords"] = coords.copy()
    return matches[0]


def _off_anchor_road(coords: np.ndarray, anchor: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = anchor
    anchor_rows = set(range(ar0, ar1 + 1))
    anchor_cols = set(range(ac0, ac1 + 1))
    return all(int(r) not in anchor_rows and int(c) not in anchor_cols for r, c in coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each anchor-lane object toward the gray anchor by the size of its same-colored counter object.

    Input structure:
    - The black background contains one fixed gray 2x2 anchor square.
    - Every other color appears as exactly two 4-connected objects: one slider
      lies wholly in an anchor row band or anchor column band, and its same-colored
      partner counter lies away from all anchor rows and columns.
    - The clear gap between each slider and the anchor is at least the pixel
      count of that slider's same-colored counter object.

    Rule:
    - For each color, count the pixels in the off-road counter object.
    - Translate the same-colored slider that many cells straight toward the
      anchor, preserving its shape and color.
    - Leave the gray anchor and every counter object exactly where they are.

    Key insight:
    - The slider does not necessarily move until contact, and its own size is
      not the step count; the displacement is transmitted through the color
      pairing from the separate same-colored counter object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == BG, "expected black to be the background"

    anchor = _find_anchor(g)
    colors = colors_present(g, exclude={BG, ANCHOR})
    assert colors, "expected at least one slider/counter color"

    moves: list[tuple[int, np.ndarray, int, int]] = []
    for color in colors:
        comps = components_mask(g == int(color), connectivity=4)
        assert len(comps) == 2, "each foreground color must form exactly two objects"

        classified = [(coords, _slider_info(coords, anchor)) for coords in comps]
        sliders = [(coords, info) for coords, info in classified if info is not None]
        counters = [coords for coords, info in classified if info is None]
        assert len(sliders) == 1 and len(counters) == 1, "each color needs one lane slider and one off-road counter"

        counter = counters[0]
        assert _off_anchor_road(counter, anchor), "counter objects must stay away from the anchor row/column roads"
        count = int(counter.shape[0])
        slider_info = sliders[0][1]
        assert slider_info is not None
        gap = int(slider_info["gap"])
        assert gap >= count, "slider-counter count must fit in the clear gap before the anchor"

        coords = np.asarray(slider_info["coords"], dtype=int)
        moves.append((int(color), coords, int(slider_info["dr"]) * count, int(slider_info["dc"]) * count))

    base = g.copy()
    for _color, coords, _dr, _dc in moves:
        base[coords[:, 0], coords[:, 1]] = BG

    out = base.copy()
    occupied = np.zeros(g.shape, dtype=bool)
    height, width = map(int, g.shape)
    for color, coords, dr, dc in moves:
        moved = coords.copy()
        moved[:, 0] += int(dr)
        moved[:, 1] += int(dc)

        assert bool(
            np.all((0 <= moved[:, 0]) & (moved[:, 0] < height) & (0 <= moved[:, 1]) & (moved[:, 1] < width))
        ), "moved slider must remain in bounds"
        assert bool(np.all(base[moved[:, 0], moved[:, 1]] == BG)), "slider destination must be clear"
        assert not bool(np.any(occupied[moved[:, 0], moved[:, 1]])), "moved sliders must not overlap"

        occupied[moved[:, 0], moved[:, 1]] = True
        out[moved[:, 0], moved[:, 1]] = int(color)

    return out


# --------------- generator ---------------

GEN_RETRY_LIMIT = 500
GEN_GRID_RANGE = (22, 30)
GEN_PAIR_RANGE = (3, 4)
GEN_COUNTER_AREA_RANGE = (1, 4)
GEN_EXTRA_GAP_RANGE = (0, 2)
GEN_PLACE_RETRY = 250


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return np.array(sorted((r - r0, c - c0) for r, c in cells), dtype=int)


def _random_polyomino(area: int, max_h: int, max_w: int) -> np.ndarray:
    """Grow a small 4-connected local polyomino inside the requested bounds."""
    assert 1 <= int(area) <= int(max_h) * int(max_w)
    for _ in range(200):
        start = (random.randrange(max_h), random.randrange(max_w))
        cells = {start}
        while len(cells) < int(area):
            frontier = []
            for cell in cells:
                for nb in _neighbors4(cell):
                    if 0 <= nb[0] < max_h and 0 <= nb[1] < max_w and nb not in cells:
                        frontier.append(nb)
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == int(area):
            coords = _normalize_cells(cells)
            if coords[:, 0].max() < max_h and coords[:, 1].max() < max_w:
                return coords
    raise RuntimeError("failed to grow a bounded polyomino")


def _offset_cells(local: np.ndarray, r0: int, c0: int) -> np.ndarray:
    cells = local.copy()
    cells[:, 0] += int(r0)
    cells[:, 1] += int(c0)
    return cells


def _can_place(grid: np.ndarray, cells: np.ndarray, *, anchor: tuple[int, int, int, int] | None = None) -> bool:
    height, width = map(int, grid.shape)
    cell_set = {(int(r), int(c)) for r, c in cells}
    for r, c in cell_set:
        if not (0 <= r < height and 0 <= c < width) or int(grid[r, c]) != BG:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in cell_set:
                    continue
                if 0 <= rr < height and 0 <= cc < width and int(grid[rr, cc]) != BG:
                    return False

    if anchor is not None and not _off_anchor_road(cells, anchor):
        return False
    return True


def _paint_cells(grid: np.ndarray, cells: np.ndarray, color: int) -> None:
    grid[cells[:, 0], cells[:, 1]] = int(color)


def _direction_set(count: int) -> list[str]:
    directions = ["left", "right", "up", "down"]
    if count == 4:
        random.shuffle(directions)
        return directions

    horizontal = random.choice(["left", "right"])
    vertical = random.choice(["up", "down"])
    remaining = [d for d in directions if d not in {horizontal, vertical}]
    return random.sample([horizontal, vertical] + remaining, count)


def _slider_local(direction: str) -> np.ndarray:
    if direction in {"left", "right"}:
        max_h, max_w = 2, 3
    else:
        max_h, max_w = 3, 2
    max_area = min(5, max_h * max_w)
    area = random.randint(1, max_area)
    return _random_polyomino(area, max_h, max_w)


def _side_capacity(shape: tuple[int, int], anchor: tuple[int, int, int, int], direction: str) -> int:
    height, width = shape
    ar0, ac0, ar1, ac1 = anchor
    if direction == "left":
        return ac0
    if direction == "right":
        return width - ac1 - 1
    if direction == "up":
        return ar0
    assert direction == "down"
    return height - ar1 - 1


def _place_slider(
    grid: np.ndarray,
    anchor: tuple[int, int, int, int],
    direction: str,
    local: np.ndarray,
    gap: int,
    color: int,
) -> np.ndarray | None:
    ar0, ac0, ar1, ac1 = anchor
    h = int(local[:, 0].max()) + 1
    w = int(local[:, 1].max()) + 1

    if direction == "left":
        top = ar0 if h == 2 else random.choice([ar0, ar1])
        left = ac0 - int(gap) - w
    elif direction == "right":
        top = ar0 if h == 2 else random.choice([ar0, ar1])
        left = ac1 + int(gap) + 1
    elif direction == "up":
        top = ar0 - int(gap) - h
        left = ac0 if w == 2 else random.choice([ac0, ac1])
    else:
        assert direction == "down"
        top = ar1 + int(gap) + 1
        left = ac0 if w == 2 else random.choice([ac0, ac1])

    cells = _offset_cells(local, top, left)
    if not _can_place(grid, cells):
        return None
    _paint_cells(grid, cells, color)
    return cells


def _place_counter(
    grid: np.ndarray,
    anchor: tuple[int, int, int, int],
    area: int,
    color: int,
) -> np.ndarray | None:
    height, width = map(int, grid.shape)
    for _ in range(GEN_PLACE_RETRY):
        if random.random() < 0.5:
            local = _random_polyomino(area, 3, 4)
        else:
            local = _random_polyomino(area, 4, 3)
        h = int(local[:, 0].max()) + 1
        w = int(local[:, 1].max()) + 1
        r0 = random.randint(1, height - h - 1)
        c0 = random.randint(1, width - w - 1)
        cells = _offset_cells(local, r0, c0)
        if _can_place(grid, cells, anchor=anchor):
            _paint_cells(grid, cells, color)
            return cells
    return None


def _expected_from_records(grid: np.ndarray, records: list[dict[str, int | np.ndarray]]) -> np.ndarray:
    out = grid.copy()
    for rec in records:
        cells = np.asarray(rec["slider"], dtype=int)
        out[cells[:, 0], cells[:, 1]] = BG
    occupied = np.zeros(grid.shape, dtype=bool)
    for rec in records:
        cells = np.asarray(rec["slider"], dtype=int)
        moved = cells.copy()
        moved[:, 0] += int(rec["dr"]) * int(rec["count"])
        moved[:, 1] += int(rec["dc"]) * int(rec["count"])
        if bool(np.any(occupied[moved[:, 0], moved[:, 1]])):
            raise ValueError("generated moved sliders overlap")
        if not bool(np.all(out[moved[:, 0], moved[:, 1]] == BG)):
            raise ValueError("generated slider destination is not clear")
        occupied[moved[:, 0], moved[:, 1]] = True
        out[moved[:, 0], moved[:, 1]] = int(rec["color"])
    return out


def _semantic_checks(records: list[dict[str, int | np.ndarray]]) -> bool:
    counts = [int(rec["count"]) for rec in records]
    gaps = [int(rec["gap"]) for rec in records]
    axes = {int(rec["axis"]) for rec in records}
    slider_areas = [int(np.asarray(rec["slider"]).shape[0]) for rec in records]

    return (
        len(records) >= 3
        and axes == {0, 1}
        and len(set(counts)) >= 2
        and any(gap == count for gap, count in zip(gaps, counts))
        and any(gap > count for gap, count in zip(gaps, counts))
        and any(area != count for area, count in zip(slider_areas, counts))
    )


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_GRID_RANGE)
    width = random.randint(*GEN_GRID_RANGE)
    grid = np.full((height, width), BG, dtype=int)

    ar = random.randint(9, height - 11)
    ac = random.randint(9, width - 11)
    anchor = (ar, ac, ar + 1, ac + 1)
    grid[ar : ar + 2, ac : ac + 2] = ANCHOR

    pair_count = random.randint(*GEN_PAIR_RANGE)
    directions = _direction_set(pair_count)
    colors = random.sample(MOVING_COLORS, pair_count)
    records: list[dict[str, int | np.ndarray]] = []

    for i, (direction, color) in enumerate(zip(directions, colors)):
        local = _slider_local(direction)
        h = int(local[:, 0].max()) + 1
        w = int(local[:, 1].max()) + 1
        extent = w if direction in {"left", "right"} else h
        max_gap = _side_capacity(grid.shape, anchor, direction) - extent
        if max_gap < 2:
            return None

        if i == 0:
            count_hi = min(GEN_COUNTER_AREA_RANGE[1], max_gap)
            count = random.randint(GEN_COUNTER_AREA_RANGE[0], count_hi)
            extra = 0
        elif i == 1:
            count_hi = min(GEN_COUNTER_AREA_RANGE[1], max_gap - 1)
            if count_hi < GEN_COUNTER_AREA_RANGE[0]:
                return None
            count = random.randint(GEN_COUNTER_AREA_RANGE[0], count_hi)
            extra = random.randint(1, min(GEN_EXTRA_GAP_RANGE[1], max_gap - count))
        else:
            count_hi = min(GEN_COUNTER_AREA_RANGE[1], max_gap)
            count = random.randint(GEN_COUNTER_AREA_RANGE[0], count_hi)
            extra_hi = min(GEN_EXTRA_GAP_RANGE[1], max_gap - count)
            extra = random.randint(0, extra_hi)
        gap = count + extra

        slider = _place_slider(grid, anchor, direction, local, gap, color)
        if slider is None:
            return None
        counter = _place_counter(grid, anchor, count, color)
        if counter is None:
            return None

        if direction == "left":
            dr, dc, axis = 0, 1, 0
        elif direction == "right":
            dr, dc, axis = 0, -1, 0
        elif direction == "up":
            dr, dc, axis = 1, 0, 1
        else:
            dr, dc, axis = -1, 0, 1
        records.append(
            {
                "color": int(color),
                "slider": slider,
                "counter": counter,
                "count": int(count),
                "gap": int(gap),
                "dr": int(dr),
                "dc": int(dc),
                "axis": int(axis),
            }
        )

    if not _semantic_checks(records):
        return None

    try:
        expected = _expected_from_records(grid, records)
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid same-color counter slide input")
