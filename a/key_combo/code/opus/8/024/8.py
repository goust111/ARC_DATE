from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg


PAIR_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
WALL_COLOR = Color.GREY
FILL_SMALL = Color.PINK
FILL_MID = Color.ORANGE
FILL_LARGE = Color.TEAL

GRID_MIN = 18
GRID_MAX = 24
CHAMBER_COUNT_OPTIONS = (3, 4)
CHAMBER_SIZE_OPTIONS = (3, 4, 5, 6, 7)
GEN_MAX_ATTEMPTS = 500

SLOT_CORNERS = ("NW", "NE", "SW", "SE")


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()) == 0
        or int(cols.min()) == 0
        or int(rows.max()) == shape[0] - 1
        or int(cols.max()) == shape[1] - 1
    )


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _color_pairs(grid: np.ndarray) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    counts = Counter(map(int, grid.ravel()))
    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color in PAIR_COLORS:
        if counts[color] != 2:
            continue
        pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(pts) == 2
        a, b = pts
        assert abs(a[0] - b[0]) == abs(a[1] - b[1]) and a != b
        pairs[int(color)] = (a, b)
    return pairs


def _diagonal_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    assert abs(dr) == abs(dc) and dr != 0
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    length = abs(dr)
    return [(int(a[0] + k * step_r), int(a[1] + k * step_c)) for k in range(length + 1)]


def _draw_all_pair_lines(grid: np.ndarray, *, skip_colors: set[int] | None = None) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    skip = set() if skip_colors is None else {int(color) for color in skip_colors}
    for color, (a, b) in _color_pairs(out).items():
        if color in skip:
            continue
        for r, c in _diagonal_path(a, b):
            if out[r, c] == Color.BLACK or out[r, c] == color:
                out[r, c] = color
    return out


def _closed_regions(grid: np.ndarray) -> list[dict]:
    regions = []
    for coords in components_mask(grid == Color.BLACK, connectivity=4):
        if _touches_border(coords, grid.shape):
            continue
        regions.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "bbox": _bbox(coords),
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw same-color diagonals, then fill enclosed black regions by relative area.

    Input structure:
    - The grid is mostly black and contains gray horizontal/vertical wall segments.
    - Three or four colors from {blue, red, green, yellow} each appear exactly twice as single pixels.
    - Each same-color pair is diagonally aligned, so its two pixels are the endpoints of a 45-degree segment.

    Rule:
    - For every same-color pair, draw the full 45-degree diagonal between the two endpoints in that pair's color.
    - Treat the new diagonals together with the gray walls as barriers, then find every 4-connected black region.
    - Keep open black regions unchanged; among closed black regions, fill the smallest with pink, the largest with teal, and every other closed region with orange.

    Key insight:
    - The colored pairs matter only because their diagonals change the topology of the black background: only after those walls are added do the enclosed regions and their area ranking become well-defined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = _draw_all_pair_lines(g)
    regions = _closed_regions(out)
    if not regions:
        return out

    ordered = sorted(regions, key=lambda region: region["area"])
    if len(ordered) == 1:
        color_map = {id(ordered[0]): FILL_SMALL}
    elif len(ordered) == 2:
        color_map = {
            id(ordered[0]): FILL_SMALL,
            id(ordered[1]): FILL_LARGE,
        }
    else:
        color_map = {id(region): FILL_MID for region in ordered}
        color_map[id(ordered[0])] = FILL_SMALL
        color_map[id(ordered[-1])] = FILL_LARGE

    for region in ordered:
        fill = int(color_map[id(region)])
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = fill
    return out


def _slot_bounds(shape: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    height, width = shape
    mid_r = height // 2
    mid_c = width // 2
    return {
        "NW": (1, 1, mid_r - 2, mid_c - 2),
        "NE": (1, mid_c + 1, mid_r - 2, width - 2),
        "SW": (mid_r + 1, 1, height - 2, mid_c - 2),
        "SE": (mid_r + 1, mid_c + 1, height - 2, width - 2),
    }


def _fit_sizes(shape: tuple[int, int], chamber_count: int) -> list[int] | None:
    bounds = _slot_bounds(shape)
    max_d = min(
        min(r1 - r0 + 1, c1 - c0 + 1) - 1
        for r0, c0, r1, c1 in bounds.values()
    )
    valid = [size for size in CHAMBER_SIZE_OPTIONS if size <= max_d]
    if len(valid) < chamber_count:
        return None
    return valid


def _diag_cells(corner: str, r0: int, c0: int, size: int) -> list[tuple[int, int]]:
    if corner in {"NW", "SE"}:
        return [(r0 + k, c0 + size - k) for k in range(size + 1)]
    return [(r0 + k, c0 + k) for k in range(size + 1)]


def _wall_cells(corner: str, r0: int, c0: int, size: int) -> list[tuple[int, int]]:
    r1 = r0 + size
    c1 = c0 + size
    if corner == "NW":
        return [(r0, c) for c in range(c0, c1)] + [(r, c0) for r in range(r0, r1)]
    if corner == "NE":
        return [(r0, c) for c in range(c0 + 1, c1 + 1)] + [(r, c1) for r in range(r0, r1)]
    if corner == "SW":
        return [(r1, c) for c in range(c0, c1)] + [(r, c0) for r in range(r0 + 1, r1 + 1)]
    return [(r1, c) for c in range(c0 + 1, c1 + 1)] + [(r, c1) for r in range(r0 + 1, r1 + 1)]


def _notch_candidates(corner: str, r0: int, c0: int, size: int) -> list[list[tuple[int, int]]]:
    r1 = r0 + size
    c1 = c0 + size
    candidates: list[list[tuple[int, int]]] = []
    for offset in range(1, size - 1):
        max_len = size - offset - 1
        for length in range(3, max_len + 1):
            if corner == "NW":
                candidates.append([(r0 + offset, c0 + step) for step in range(length)])
                candidates.append([(r0 + step, c0 + offset) for step in range(length)])
            elif corner == "NE":
                candidates.append([(r0 + offset, c1 - step) for step in range(length)])
                candidates.append([(r0 + step, c1 - offset) for step in range(length)])
            elif corner == "SW":
                candidates.append([(r1 - offset, c0 + step) for step in range(length)])
                candidates.append([(r1 - step, c0 + offset) for step in range(length)])
            else:
                candidates.append([(r1 - offset, c1 - step) for step in range(length)])
                candidates.append([(r1 - step, c1 - offset) for step in range(length)])
    return candidates


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _place_chamber(
    grid: np.ndarray,
    *,
    corner: str,
    size: int,
    color: int,
    bounds: tuple[int, int, int, int],
) -> dict:
    row_lo, col_lo, row_hi, col_hi = bounds
    top_max = row_hi - size
    left_max = col_hi - size
    assert row_lo <= top_max and col_lo <= left_max
    r0 = random.randint(row_lo, top_max)
    c0 = random.randint(col_lo, left_max)
    diag = _diag_cells(corner, r0, c0, size)
    walls = _wall_cells(corner, r0, c0, size)
    _paint(grid, walls, WALL_COLOR)
    _paint(grid, [diag[0], diag[-1]], color)
    return {
        "corner": corner,
        "size": int(size),
        "color": int(color),
        "bbox": (r0, c0, r0 + size, c0 + size),
        "diag": diag,
        "notch": None,
    }


def _choose_notches(chambers: list[dict], require_twist: bool) -> None:
    candidates = [(index, _notch_candidates(ch["corner"], ch["bbox"][0], ch["bbox"][1], ch["size"])) for index, ch in enumerate(chambers)]
    candidates = [(index, options) for index, options in candidates if options]
    if not candidates:
        return

    target_count = random.choices([0, 1, 2], weights=[2, 4, 3], k=1)[0]
    if require_twist:
        target_count = max(1, target_count)
    random.shuffle(candidates)
    chosen = 0
    for index, options in candidates:
        if chosen >= target_count:
            break
        chambers[index]["notch"] = random.choice(options)
        chosen += 1


def _add_center_distractors(grid: np.ndarray) -> None:
    height, width = grid.shape
    row_choices = [height // 2 - 1, height // 2]
    col_choices = [width // 2 - 1, width // 2]
    segment_count = random.randint(0, 2)
    for _ in range(segment_count):
        if random.random() < 0.5:
            row = random.choice(row_choices)
            length = random.randint(3, min(6, width - 4))
            start = random.randint(2, width - length - 2)
            cells = [(row, start + k) for k in range(length)]
        else:
            col = random.choice(col_choices)
            length = random.randint(3, min(6, height - 4))
            start = random.randint(2, height - length - 2)
            cells = [(start + k, col) for k in range(length)]
        if all(int(grid[r, c]) == Color.BLACK for r, c in cells):
            _paint(grid, cells, WALL_COLOR)


def _map_regions_to_chambers(regions: list[dict], chambers: list[dict]) -> dict[int, dict] | None:
    mapping: dict[int, dict] = {}
    used_regions: set[int] = set()
    for index, chamber in enumerate(chambers):
        r0, c0, r1, c1 = chamber["bbox"]
        matches = []
        for region_index, region in enumerate(regions):
            rr0, cc0, rr1, cc1 = region["bbox"]
            if r0 <= rr0 <= rr1 <= r1 and c0 <= cc0 <= cc1 <= c1:
                matches.append((region_index, region))
        if len(matches) != 1:
            return None
        region_index, region = matches[0]
        if region_index in used_regions:
            return None
        used_regions.add(region_index)
        mapping[index] = region
    if len(used_regions) != len(regions):
        return None
    return mapping


def _filled_region_color(output: np.ndarray, region: dict) -> int:
    coords = region["coords"]
    vals = set(map(int, output[coords[:, 0], coords[:, 1]]))
    assert len(vals) == 1
    return int(next(iter(vals)))


def _validate_generated(grid: np.ndarray, chambers: list[dict], require_twist: bool) -> bool:
    pair_map = _color_pairs(grid)
    if len(pair_map) != len(chambers):
        return False

    path_cells: set[tuple[int, int]] = set()
    for chamber in chambers:
        color = int(chamber["color"])
        if color not in pair_map:
            return False
        diag = chamber["diag"]
        if pair_map[color] != (diag[0], diag[-1]) and pair_map[color] != (diag[-1], diag[0]):
            return False
        for cell in diag[1:-1]:
            if cell in path_cells:
                return False
            path_cells.add(cell)
            if int(grid[cell]) != Color.BLACK:
                return False

    if _closed_regions(grid):
        return False

    lined = _draw_all_pair_lines(grid)
    regions = _closed_regions(lined)
    if len(regions) != len(chambers):
        return False

    areas = [region["area"] for region in regions]
    if len(set(areas)) != len(areas):
        return False

    mapping = _map_regions_to_chambers(regions, chambers)
    if mapping is None:
        return False

    for chamber in chambers:
        color = int(chamber["color"])
        reduced = _draw_all_pair_lines(grid, skip_colors={color})
        reduced_regions = _closed_regions(reduced)
        if len(reduced_regions) != len(chambers) - 1:
            return False

    output = main(grid)
    ordered = sorted(mapping.items(), key=lambda item: item[1]["area"])
    expected_colors = {}
    if len(ordered) == 1:
        expected_colors[ordered[0][0]] = FILL_SMALL
    elif len(ordered) == 2:
        expected_colors[ordered[0][0]] = FILL_SMALL
        expected_colors[ordered[1][0]] = FILL_LARGE
    else:
        for chamber_index, _region in ordered[1:-1]:
            expected_colors[chamber_index] = FILL_MID
        expected_colors[ordered[0][0]] = FILL_SMALL
        expected_colors[ordered[-1][0]] = FILL_LARGE

    for chamber_index, region in mapping.items():
        if _filled_region_color(output, region) != int(expected_colors[chamber_index]):
            return False

    if require_twist:
        by_size = [index for index, _chamber in sorted(enumerate(chambers), key=lambda item: item[1]["size"])]
        by_area = [index for index, _region in sorted(mapping.items(), key=lambda item: item[1]["area"])]
        if by_size == by_area:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    shape = (height, width)
    chamber_count = random.choice(CHAMBER_COUNT_OPTIONS)
    valid_sizes = _fit_sizes(shape, chamber_count)
    if valid_sizes is None:
        return None

    chosen_corners = random.sample(list(SLOT_CORNERS), chamber_count)
    bounds = _slot_bounds(shape)
    sizes = random.sample(valid_sizes, chamber_count)
    colors = random.sample(list(PAIR_COLORS), chamber_count)
    random.shuffle(sizes)
    random.shuffle(colors)

    grid = np.full(shape, Color.BLACK, dtype=int)
    chambers = [
        _place_chamber(
            grid,
            corner=corner,
            size=size,
            color=color,
            bounds=bounds[corner],
        )
        for corner, size, color in zip(chosen_corners, sizes, colors)
    ]

    require_twist = random.random() < 0.65
    _choose_notches(chambers, require_twist)
    for chamber in chambers:
        if chamber["notch"] is not None:
            _paint(grid, chamber["notch"], WALL_COLOR)

    _add_center_distractors(grid)

    if not _validate_generated(grid, chambers, require_twist):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
