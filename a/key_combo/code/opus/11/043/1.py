from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

PANEL_CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
PANEL_EDGES = ((0, 1), (1, 0), (1, 2), (2, 1))

TEMPLATES = (
    {"central": "T_from_left", "sides": ("L_from_top", "R_from_top")},
    {"central": "T_from_right", "sides": ("L_from_top", "R_from_top")},
    {"central": "B_from_left", "sides": ("L_from_bottom", "R_from_bottom")},
    {"central": "B_from_right", "sides": ("L_from_bottom", "R_from_bottom")},
    {"central": "L_from_top", "sides": ("T_from_left", "B_from_left")},
    {"central": "L_from_bottom", "sides": ("T_from_left", "B_from_left")},
    {"central": "R_from_top", "sides": ("T_from_right", "B_from_right")},
    {"central": "R_from_bottom", "sides": ("T_from_right", "B_from_right")},
)


def _only(values: list[int]) -> int:
    uniq = sorted(set(int(v) for v in values))
    assert len(uniq) == 1
    return uniq[0]


def _find_anchor(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.shape[0] == 4, "expected a single 2x2 grey anchor block"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1
    assert np.all(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == GREY)
    return int(r0), int(c0)


def _decode_legend(grid: np.ndarray) -> list[int]:
    panel = np.asarray(grid[:3, -3:], dtype=int)
    bottom = _only([int(panel[r, c]) for r, c in PANEL_CORNERS])
    middle = _only([int(panel[r, c]) for r, c in PANEL_EDGES])
    top = int(panel[1, 1])
    assert len({top, middle, bottom}) == 3
    assert BLACK not in {top, middle, bottom}
    assert GREY not in {top, middle, bottom}
    return [top, middle, bottom]


def _marker_ray_kind(
    marker: tuple[int, int], shape: tuple[int, int], anchor_rc: tuple[int, int]
) -> str:
    h, w = map(int, shape)
    r, c = map(int, marker)
    ar, ac = map(int, anchor_rc)
    top_row = ar - 1
    bot_row = ar + 2
    left_col = ac - 1
    right_col = ac + 2

    if c == 0 and r == top_row:
        return "T_from_left"
    if c == w - 1 and r == top_row:
        return "T_from_right"
    if c == 0 and r == bot_row:
        return "B_from_left"
    if c == w - 1 and r == bot_row:
        return "B_from_right"
    if r == 0 and c == left_col:
        return "L_from_top"
    if r == h - 1 and c == left_col:
        return "L_from_bottom"
    if r == 0 and c == right_col:
        return "R_from_top"
    if r == h - 1 and c == right_col:
        return "R_from_bottom"
    raise AssertionError("marker is not on a valid anchor approach line")


def _ray_path(
    marker: tuple[int, int], shape: tuple[int, int], anchor_rc: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r, c = map(int, marker)
    kind = _marker_ray_kind(marker, shape, anchor_rc)

    if kind == "T_from_left":
        return [(r, cc) for cc in range(1, w - 1)]
    if kind == "T_from_right":
        return [(r, cc) for cc in range(w - 2, 0, -1)]
    if kind == "B_from_left":
        return [(r, cc) for cc in range(1, w - 1)]
    if kind == "B_from_right":
        return [(r, cc) for cc in range(w - 2, 0, -1)]
    if kind == "L_from_top":
        return [(rr, c) for rr in range(1, h - 1)]
    if kind == "L_from_bottom":
        return [(rr, c) for rr in range(h - 2, 0, -1)]
    if kind == "R_from_top":
        return [(rr, c) for rr in range(1, h - 1)]
    if kind == "R_from_bottom":
        return [(rr, c) for rr in range(h - 2, 0, -1)]
    raise AssertionError("unreachable")


def _extract_marker_paths(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    order = _decode_legend(grid)
    colors = set(order)
    ar, ac = _find_anchor(grid)
    h, w = grid.shape

    markers: dict[int, tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            if r < 3 and c >= w - 3:
                continue
            if (r in (0, h - 1) or c in (0, w - 1)) and int(grid[r, c]) in colors:
                color = int(grid[r, c])
                assert color not in markers, "each legend color must have exactly one border marker"
                markers[color] = (r, c)

    assert set(markers) == colors, "legend colors and marker colors must match one-to-one"
    return {color: _ray_path(markers[color], grid.shape, (ar, ac)) for color in colors}


def _draw_rays(
    grid: np.ndarray, order: list[int], paths: dict[int, list[tuple[int, int]]]
) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    out = np.array(grid, dtype=int, copy=True)
    occupied: set[tuple[int, int]] = set()
    drawn: dict[int, list[tuple[int, int]]] = {}

    for color in order:
        segment: list[tuple[int, int]] = []
        for cell in paths[color]:
            if cell in occupied:
                break
            out[cell] = int(color)
            segment.append(cell)
        occupied.update(segment)
        drawn[int(color)] = segment
    return out, drawn


def _full_draw(grid: np.ndarray, order: list[int], paths: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for color in order:
        for r, c in paths[color]:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend border-to-anchor rays and truncate lower layers at crossings using a legend-encoded stack order.

    Input structure:
    - Black background with a fixed 3x3 legend panel in the upper-right corner, a 2x2 grey anchor block near the center, and three single-pixel border markers.
    - The three marker colors match the three colors visible in the legend panel one-to-one.
    - Each marker sits on one of the row/column approach lines leading straight toward the anchor block.

    Rule:
    - Read the legend panel as an overlap stack: the center color is top, the edge-center color is middle, and the corner color is bottom.
    - From each border marker, extend a straight ray of the same color along its anchor-selected approach line across the board, without repainting the border marker itself.
    - Draw the rays from top layer to bottom layer; whenever a lower ray reaches a cell already occupied by a higher ray, stop that lower ray immediately before the crossing cell.
    - Keep the legend panel, the anchor block, and the marker pixels unchanged.

    Key insight:
    - The 3x3 legend is not decorative: its visible nested overlap pattern encodes the ray priority, and that priority determines which crossings remain passable and which rays get cut short.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    order = _decode_legend(g)
    paths = _extract_marker_paths(g)
    out, _ = _draw_rays(g, order, paths)
    return out


def _marker_position(kind: str, shape: tuple[int, int], anchor_rc: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    ar, ac = map(int, anchor_rc)
    top_row = ar - 1
    bot_row = ar + 2
    left_col = ac - 1
    right_col = ac + 2

    mapping = {
        "T_from_left": (top_row, 0),
        "T_from_right": (top_row, w - 1),
        "B_from_left": (bot_row, 0),
        "B_from_right": (bot_row, w - 1),
        "L_from_top": (0, left_col),
        "L_from_bottom": (h - 1, left_col),
        "R_from_top": (0, right_col),
        "R_from_bottom": (h - 1, right_col),
    }
    return mapping[kind]


def _place_panel(grid: np.ndarray, top: int, middle: int, bottom: int) -> None:
    panel = np.full((3, 3), int(bottom), dtype=int)
    for r, c in PANEL_EDGES:
        panel[r, c] = int(middle)
    panel[1, 1] = int(top)
    grid[:3, -3:] = panel


def _path_intersections(paths: dict[int, list[tuple[int, int]]]) -> set[tuple[int, int]]:
    colors = sorted(paths)
    hits: set[tuple[int, int]] = set()
    for i, a in enumerate(colors):
        aset = set(paths[a])
        for b in colors[i + 1 :]:
            hits |= aset & set(paths[b])
    return hits


def _validate_input(grid: np.ndarray) -> bool:
    try:
        order = _decode_legend(grid)
        paths = _extract_marker_paths(grid)
        actual, drawn = _draw_rays(grid, order, paths)
    except AssertionError:
        return False

    if any(len(path) < 3 for path in paths.values()):
        return False

    intersections = _path_intersections(paths)
    if len(intersections) < 2:
        return False

    if np.array_equal(actual, grid):
        return False

    if any(len(drawn[color]) == 0 for color in order):
        return False

    if not any(len(drawn[color]) < len(paths[color]) for color in order):
        return False

    full = _full_draw(grid, order, paths)
    if np.array_equal(full, actual):
        return False

    actual_sig = (actual.shape, actual.tobytes())
    for perm in itertools.permutations(order):
        if list(perm) == order:
            continue
        alt, _ = _draw_rays(grid, list(perm), paths)
        if (alt.shape, alt.tobytes()) == actual_sig:
            return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, actual)


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(14, 18)
        w = random.randint(14, 18)
        ar = random.randint(4, h - 7)
        ac = random.randint(4, w - 7)

        grid = np.full((h, w), BLACK, dtype=int)
        grid[ar : ar + 2, ac : ac + 2] = GREY

        top, middle, bottom = random.sample(PALETTE, 3)
        _place_panel(grid, top, middle, bottom)

        template = random.choice(TEMPLATES)
        high_side, low_side = random.sample(list(template["sides"]), 2)
        color_by_ray = {
            template["central"]: middle,
            high_side: top,
            low_side: bottom,
        }

        for ray_kind, color in color_by_ray.items():
            mr, mc = _marker_position(ray_kind, (h, w), (ar, ac))
            if mr < 3 and mc >= w - 3:
                break
            grid[mr, mc] = int(color)
        else:
            if _validate_input(grid):
                return grid

    raise RuntimeError("failed to generate a valid ray-truncation ARC input")
