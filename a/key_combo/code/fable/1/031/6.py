from __future__ import annotations

from collections import Counter
from pathlib import Path
import importlib.util
import random
import sys

import numpy as np


def _load_arc_utils():
    arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    try:
        parent = str(arc_utils_path.parent)
        if parent not in sys.path:
            sys.path.append(parent)
        from arc_utils import Color, find_components, get_bg

        return Color, find_components, get_bg
    except Exception:
        if arc_utils_path.exists():
            spec = importlib.util.spec_from_file_location("_arc_utils_local", arc_utils_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Color, module.find_components, module.get_bg

        class _FallbackColor:
            BLACK = 0
            BLUE = 1
            RED = 2
            GREEN = 3
            YELLOW = 4
            GREY = 5
            GRAY = 5
            PINK = 6
            ORANGE = 7
            TEAL = 8
            MAROON = 9

        def _fallback_get_bg(grid: np.ndarray) -> int:
            return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])

        def _fallback_find_components(
            grid: np.ndarray,
            *,
            bg: int | None = None,
            connectivity: int = 4,
            monochromatic: bool = True,
            colors: list[int] | None = None,
            min_area: int = 1,
            return_format: str = "dict",
        ) -> list[dict]:
            del connectivity, monochromatic, return_format
            if bg is None:
                bg = _fallback_get_bg(grid)
            if colors is None:
                palette = sorted(set(map(int, np.asarray(grid).ravel())) - {int(bg)})
            else:
                palette = [int(color) for color in colors if int(color) != int(bg)]
            out: list[dict] = []
            for color in palette:
                coords = np.argwhere(np.asarray(grid) == color)
                if coords.shape[0] < int(min_area):
                    continue
                r0, c0 = coords.min(axis=0)
                r1, c1 = coords.max(axis=0)
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(coords.shape[0]),
                    }
                )
            return out

        return _FallbackColor, _fallback_find_components, _fallback_get_bg


Color, find_components, get_bg = _load_arc_utils()


BLACK = int(Color.BLACK)
GREY = int(Color.GREY)
MARKER_PALETTE = [
    int(Color.BLUE),
    int(Color.RED),
    int(Color.GREEN),
    int(Color.YELLOW),
    int(Color.PINK),
    int(Color.ORANGE),
    int(Color.TEAL),
    int(Color.MAROON),
]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MIN_MARKERS = 3
MAX_MARKERS = 6
MIN_MAIN_H = 8
MAX_MAIN_H = 18
MIN_MAIN_W = 9
MAX_MAIN_W = 18
GENERATION_ATTEMPTS = 300
MIN_ADDED_CELLS = 8
MAX_FILL_RATIO = 0.52


def _separator_index(grid: np.ndarray) -> int:
    candidates = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(candidates) == 1, "expected one full-height grey separator column"
    sep = candidates[0]
    assert 0 < sep < grid.shape[1] - 1, "separator must leave a main area and a control column"
    return sep


def _control_order(control_area: np.ndarray) -> list[int]:
    assert control_area.shape[1] == 1, "expected one control column"
    order = [int(color) for color in control_area[:, 0] if int(color) not in (BLACK, GREY)]
    assert len(order) >= 2, "expected at least two listed colors"
    assert len(order) == len(set(order)), "control colors must be unique"
    return order


def _marker_positions(main_grid: np.ndarray, order: list[int]) -> dict[int, tuple[int, int]]:
    components = find_components(main_grid, bg=BLACK, connectivity=4, monochromatic=True)
    markers: dict[int, tuple[int, int]] = {}
    for component in components:
        color = int(component["color"])
        assert color != GREY, "separator color must not appear in the main area"
        assert int(component["area"]) == 1, "expected singleton markers only"
        coord = tuple(map(int, component["coords"][0]))
        markers[color] = coord

    assert len(markers) == len(components), "marker colors must be distinct"
    assert set(order) == set(markers), "control column must list every marker color exactly once"
    return markers


def _fire_in_order(
    main_grid: np.ndarray,
    order: list[int],
    markers: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, int, bool]:
    out = main_grid.copy()
    h, w = out.shape
    marker_coords = set(markers.values())
    added = 0
    blocked_by_prior_ray = False

    for color in order:
        r, c = markers[color]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w and out[nr, nc] == BLACK:
                out[nr, nc] = color
                added += 1
                nr += dr
                nc += dc
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] != BLACK and (nr, nc) not in marker_coords:
                blocked_by_prior_ray = True

    return out, added, blocked_by_prior_ray


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignite singleton markers in control-column order and keep only the cropped ray-filled main area.

    Input structure
    - The grid has a black main area on the left, one full-height gray separator column, and one control column on the far right.
    - The main area contains several distinct non-black singleton markers.
    - The control column lists the same marker colors from top to bottom, with black gaps allowed between entries.

    Rule
    - Read the control colors from top to bottom.
    - For each listed color, locate the matching singleton marker in the main area and extend same-color rays up, down, left, and right.
    - Each ray advances only through black cells and stops immediately before the first non-black cell or the main-area boundary, so untouched markers and earlier rays both block later growth.
    - After all colors fire, return only the left main area, discarding the gray separator and control column.

    Key insight
    - The control column is the true firing order; marker position does not decide precedence, and every earlier ray becomes a new obstacle for later colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert int(get_bg(g)) == BLACK, "expected black to be the background color"

    sep = _separator_index(g)
    main_grid = g[:, :sep].copy()
    control_area = g[:, sep + 1 :]

    order = _control_order(control_area)
    markers = _marker_positions(main_grid, order)
    out, _added, _blocked = _fire_in_order(main_grid, order, markers)
    return out


def _control_rows(height: int, count: int) -> list[int]:
    bases = sorted(random.sample(range(height - (count - 1)), count))
    return [base + i for i, base in enumerate(bases)]


def _chebyshev_clear(pos: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return all(max(abs(pos[0] - r), abs(pos[1] - c)) >= 2 for r, c in occupied)


def _core_positions(height: int, width: int) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    if random.random() < 0.5:
        r = random.randint(2, height - 3)
        c1 = random.randint(1, width - 6)
        c2 = random.randint(c1 + 4, width - 2)
        cm = random.randint(c1 + 1, c2 - 1)
        r3 = random.choice([rr for rr in range(1, height - 1) if abs(rr - r) >= 2])
        positions = [(r, c1), (r, c2), (r3, cm)]
        reserved = {(r, c) for c in range(c1, c2 + 1)}
        reserved.update((rr, cm) for rr in range(min(r, r3), max(r, r3) + 1))
        return positions, reserved

    c = random.randint(2, width - 3)
    r1 = random.randint(1, height - 6)
    r2 = random.randint(r1 + 4, height - 2)
    rm = random.randint(r1 + 1, r2 - 1)
    c3 = random.choice([cc for cc in range(1, width - 1) if abs(cc - c) >= 2])
    positions = [(r1, c), (r2, c), (rm, c3)]
    reserved = {(r, c) for r in range(r1, r2 + 1)}
    reserved.update((rm, cc) for cc in range(min(c, c3), max(c, c3) + 1))
    return positions, reserved


def _extra_positions(
    height: int,
    width: int,
    positions: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
    extra_count: int,
) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if (r, c) not in reserved and (r, c) not in positions and _chebyshev_clear((r, c), positions)
    ]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    occupied = positions[:]
    for candidate in candidates:
        if _chebyshev_clear(candidate, occupied):
            chosen.append(candidate)
            occupied.append(candidate)
            if len(chosen) == extra_count:
                return chosen
    return None


def _row_major_order(markers: dict[int, tuple[int, int]]) -> list[int]:
    return [color for color, _ in sorted(markers.items(), key=lambda item: item[1])]


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(MIN_MARKERS, MAX_MARKERS)
    height = random.randint(max(MIN_MAIN_H, 2 * count - 1), MAX_MAIN_H)
    width = random.randint(MIN_MAIN_W, MAX_MAIN_W)

    core, reserved = _core_positions(height, width)
    extras = _extra_positions(height, width, core, reserved, count - len(core))
    if extras is None:
        return None
    positions = core + extras

    colors = random.sample(MARKER_PALETTE, count)
    markers = {color: pos for color, pos in zip(colors, positions)}
    pair_colors = colors[:2]
    cross_color = colors[2]

    order = colors[:]
    random.shuffle(order)
    if order.index(cross_color) < min(order.index(pair_colors[0]), order.index(pair_colors[1])):
        earlier_pair = random.choice(pair_colors)
        i = order.index(earlier_pair)
        j = order.index(cross_color)
        order[i], order[j] = order[j], order[i]

    row_major = _row_major_order(markers)
    if order == row_major:
        return None

    main_grid = np.full((height, width), BLACK, dtype=int)
    for color, (r, c) in markers.items():
        main_grid[r, c] = color

    out, added, blocked_by_prior_ray = _fire_in_order(main_grid, order, markers)
    alt_out, _alt_added, _alt_blocked = _fire_in_order(main_grid, row_major, markers)

    if not blocked_by_prior_ray:
        return None
    if added < MIN_ADDED_CELLS:
        return None
    if np.array_equal(out, alt_out):
        return None
    if np.count_nonzero(out != BLACK) / out.size > MAX_FILL_RATIO:
        return None

    grid = np.full((height, width + 2), BLACK, dtype=int)
    grid[:, width] = GREY
    grid[:, :width] = main_grid
    control_rows = _control_rows(height, count)
    for row, color in zip(control_rows, order):
        grid[row, width + 1] = color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ordered-cross-rays input")
