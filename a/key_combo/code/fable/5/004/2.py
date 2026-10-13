from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import bbox_coords, find_components, get_bg


Orientation = str
Coord = tuple[int, int]


def _coord_set(coords: np.ndarray) -> set[Coord]:
    return {(int(r), int(c)) for r, c in coords}


def _reflect_cells(cells: set[Coord], orientation: Orientation, axis2: int) -> set[Coord]:
    if orientation == "V":
        return {(r, int(axis2) - c) for r, c in cells}
    return {(int(axis2) - r, c) for r, c in cells}


def _reflected_bbox_axis2(a: dict, b: dict, orientation: Orientation) -> int | None:
    ar0, ac0, ar1, ac1 = map(int, a["bbox"])
    br0, bc0, br1, bc1 = map(int, b["bbox"])
    if int(a["area"]) != int(b["area"]):
        return None

    if orientation == "V":
        if ar0 != br0 or ar1 != br1 or (ac1 - ac0) != (bc1 - bc0):
            return None
        axis2 = ac0 + bc1
        if axis2 != ac1 + bc0:
            return None
    else:
        if ac0 != bc0 or ac1 != bc1 or (ar1 - ar0) != (br1 - br0):
            return None
        axis2 = ar0 + br1
        if axis2 != ar1 + br0:
            return None

    return int(axis2)


def _component_side(comp: dict, orientation: Orientation, axis2: int) -> str | None:
    coords = comp["coords"]
    if orientation == "V":
        vals = [2 * int(c) - int(axis2) for _r, c in coords]
        if all(v < 0 for v in vals):
            return "left"
        if all(v > 0 for v in vals):
            return "right"
        return None

    vals = [2 * int(r) - int(axis2) for r, _c in coords]
    if all(v < 0 for v in vals):
        return "top"
    if all(v > 0 for v in vals):
        return "bottom"
    return None


def _opposite(side: str) -> str:
    return {
        "left": "right",
        "right": "left",
        "top": "bottom",
        "bottom": "top",
    }[side]


def _add_mapping(mapping: dict[tuple[str, int], int], side: str, color: int, other: int) -> bool:
    key = (side, int(color))
    other = int(other)
    if key in mapping and mapping[key] != other:
        return False
    mapping[key] = other
    return True


def _pair_edges(components: list[dict]) -> dict[tuple[Orientation, int], list[tuple[int, int]]]:
    by_axis: dict[tuple[Orientation, int], list[tuple[int, int]]] = {}
    cell_sets = [_coord_set(comp["coords"]) for comp in components]
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            for orientation in ("V", "H"):
                axis2 = _reflected_bbox_axis2(components[i], components[j], orientation)
                if axis2 is None:
                    continue
                if _reflect_cells(cell_sets[i], orientation, axis2) == cell_sets[j]:
                    by_axis.setdefault((orientation, axis2), []).append((i, j))
    return by_axis


def _axis_info(
    key: tuple[Orientation, int],
    edges: list[tuple[int, int]],
    components: list[dict],
    shape: tuple[int, int],
) -> dict | None:
    orientation, axis2 = key
    h, w = shape
    if orientation == "V" and axis2 == w - 1:
        return None
    if orientation == "H" and axis2 == h - 1:
        return None

    used: set[int] = set()
    mapping: dict[tuple[str, int], int] = {}
    side_colors: dict[str, set[int]] = {"left": set(), "right": set(), "top": set(), "bottom": set()}

    for i, j in edges:
        if i in used or j in used:
            return None

        side_i = _component_side(components[i], orientation, axis2)
        side_j = _component_side(components[j], orientation, axis2)
        if side_i is None or side_j is None or side_j != _opposite(side_i):
            return None

        color_i = int(components[i]["color"])
        color_j = int(components[j]["color"])
        if not _add_mapping(mapping, side_i, color_i, color_j):
            return None
        if not _add_mapping(mapping, side_j, color_j, color_i):
            return None

        side_colors[side_i].add(color_i)
        side_colors[side_j].add(color_j)
        used.update((i, j))

    if orientation == "V":
        if side_colors["left"] & side_colors["right"]:
            return None
        color_set = side_colors["left"] | side_colors["right"]
    else:
        if side_colors["top"] & side_colors["bottom"]:
            return None
        color_set = side_colors["top"] | side_colors["bottom"]

    if not color_set:
        return None

    return {
        "orientation": orientation,
        "axis2": int(axis2),
        "components": used,
        "mapping": mapping,
        "colors": color_set,
        "edges": edges,
    }


def _solve(grid: np.ndarray) -> tuple[dict, int, set[Coord], int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(components) >= 5 and len(components) % 2 == 1

    groups = _pair_edges(components)
    infos = []
    for key, edges in groups.items():
        info = _axis_info(key, edges, components, g.shape)
        if info is not None:
            infos.append(info)

    vertical = [info for info in infos if info["orientation"] == "V"]
    horizontal = [info for info in infos if info["orientation"] == "H"]

    solutions: list[tuple[dict, int, set[Coord], int]] = []
    all_ids = set(range(len(components)))
    for v_info in vertical:
        for h_info in horizontal:
            if v_info["components"] & h_info["components"]:
                continue
            if v_info["colors"] & h_info["colors"]:
                continue

            covered = v_info["components"] | h_info["components"]
            orphan_ids = sorted(all_ids - covered)
            if len(orphan_ids) != 1:
                continue

            orphan_id = orphan_ids[0]
            orphan = components[orphan_id]
            orphan_color = int(orphan["color"])
            in_v = orphan_color in v_info["colors"]
            in_h = orphan_color in h_info["colors"]
            if in_v == in_h:
                continue

            axis_info = v_info if in_v else h_info
            side = _component_side(orphan, axis_info["orientation"], axis_info["axis2"])
            if side is None:
                continue

            target_color = axis_info["mapping"].get((side, orphan_color))
            if target_color is None:
                continue

            cells = _coord_set(orphan["coords"])
            target = _reflect_cells(cells, axis_info["orientation"], axis_info["axis2"])
            if target == cells:
                continue

            h, w = g.shape
            if any(not (0 <= r < h and 0 <= c < w) for r, c in target):
                continue
            if any(int(g[r, c]) != bg for r, c in target):
                continue

            solutions.append((axis_info, orphan_id, target, int(target_color)))

    assert len(solutions) == 1, f"expected a unique dual-axis orphan interpretation, got {len(solutions)}"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the orphan shape by using its color to choose one of two hidden mirror axes.

    Input structure:
    - The background is the dominant color, and the foreground consists of separated small monochrome shapes.
    - All shapes except one form reflection pairs: some pairs share one hidden vertical axis, and the rest share one hidden horizontal axis.
    - Each axis has its own left-right or top-bottom color mapping table, and the two tables use disjoint color sets.
    - Neither hidden axis is the center line of the grid.

    Rule:
    - Infer the vertical axis from vertical mirror-pair midpoints and the horizontal axis from horizontal mirror-pair midpoints.
    - Read each axis color table from its paired shapes: a color on one side always maps to the paired color on the opposite side.
    - The orphan's color belongs to exactly one table; reflect the whole orphan shape across that table's axis and paint it with the mapped opposite-side color.
    - Preserve every input pixel.

    Key insight:
    - Geometry gives two separate hidden axes, while the disjoint color tables decide which axis owns the orphan.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _axis_info, _orphan_id, target, target_color = _solve(g)
    for r, c in target:
        out[r, c] = target_color
    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_SIZE_RANGE = (20, 29)
PAIR_COUNT_RANGE = (2, 4)
SHAPE_AREA_RANGE = (2, 6)
SHAPE_BBOX_MAX = 4
MAX_ATTEMPTS = 500
MAX_SHAPE_ATTEMPTS = 80


def _normalize_shape(cells: set[Coord]) -> frozenset[Coord]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _random_shape() -> frozenset[Coord]:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        target = random.randint(*SHAPE_AREA_RANGE)
        cells: set[Coord] = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        shape = _normalize_shape(cells)
        rows = [r for r, _c in shape]
        cols = [c for _r, c in shape]
        if max(rows) + 1 <= SHAPE_BBOX_MAX and max(cols) + 1 <= SHAPE_BBOX_MAX:
            return shape
    return frozenset({(0, 0), (0, 1), (1, 0)})


def _shape_cells(shape: frozenset[Coord], r0: int, c0: int) -> set[Coord]:
    return {(r0 + r, c0 + c) for r, c in shape}


def _strict_side(cells: set[Coord], orientation: Orientation, axis2: int, side: str) -> bool:
    if orientation == "V":
        vals = [2 * c - axis2 for _r, c in cells]
        return all(v < 0 for v in vals) if side == "left" else all(v > 0 for v in vals)
    vals = [2 * r - axis2 for r, _c in cells]
    return all(v < 0 for v in vals) if side == "top" else all(v > 0 for v in vals)


def _in_bounds(cells: set[Coord], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _free(blocked: np.ndarray, cells: set[Coord]) -> bool:
    return all(not bool(blocked[r, c]) for r, c in cells)


def _mark_blocked(blocked: np.ndarray, cells: set[Coord], pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        blocked[r0:r1, c0:c1] = True


def _paint(grid: np.ndarray, cells: set[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _candidate_anchors(h: int, w: int, shape: frozenset[Coord]) -> list[tuple[int, int]]:
    sh = max(r for r, _c in shape) + 1
    sw = max(c for _r, c in shape) + 1
    anchors = [(r0, c0) for r0 in range(h - sh + 1) for c0 in range(w - sw + 1)]
    random.shuffle(anchors)
    return anchors


def _place_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    orientation: Orientation,
    axis2: int,
    mapping_pair: tuple[int, int],
) -> bool:
    h, w = grid.shape
    side = "left" if orientation == "V" else "top"
    shape = _random_shape()
    for r0, c0 in _candidate_anchors(h, w, shape):
        cells = _shape_cells(shape, r0, c0)
        mirror = _reflect_cells(cells, orientation, axis2)
        if not _in_bounds(mirror, h, w):
            continue
        if not _strict_side(cells, orientation, axis2, side):
            continue
        if not _strict_side(mirror, orientation, axis2, _opposite(side)):
            continue
        if not _free(blocked, cells | mirror):
            continue

        _paint(grid, cells, mapping_pair[0])
        _paint(grid, mirror, mapping_pair[1])
        _mark_blocked(blocked, cells | mirror)
        return True
    return False


def _place_orphan(
    grid: np.ndarray,
    blocked: np.ndarray,
    orientation: Orientation,
    axis2: int,
    other_orientation: Orientation,
    other_axis2: int,
    side: str,
    color: int,
    target_color: int,
) -> tuple[set[Coord], set[Coord], int] | None:
    h, w = grid.shape
    shape = _random_shape()
    for r0, c0 in _candidate_anchors(h, w, shape):
        cells = _shape_cells(shape, r0, c0)
        target = _reflect_cells(cells, orientation, axis2)
        wrong_target = _reflect_cells(cells, other_orientation, other_axis2)
        reserved = cells | target | wrong_target
        if target == cells or wrong_target == cells or wrong_target == target:
            continue
        if not _in_bounds(target, h, w) or not _in_bounds(wrong_target, h, w):
            continue
        if not _strict_side(cells, orientation, axis2, side):
            continue
        if not _strict_side(target, orientation, axis2, _opposite(side)):
            continue
        if not _free(blocked, reserved):
            continue

        _paint(grid, cells, color)
        _mark_blocked(blocked, reserved)
        return cells, target, int(target_color)
    return None


def _choose_axis2(length: int) -> int:
    candidates = [
        axis2
        for axis2 in range(7, 2 * length - 8)
        if axis2 != length - 1
    ]
    return random.choice(candidates)


def _expected_output(inp: np.ndarray, target: set[Coord], color: int) -> np.ndarray:
    out = inp.copy()
    for r, c in target:
        out[r, c] = int(color)
    return out


def _semantic_accepts(
    inp: np.ndarray,
    expected: np.ndarray,
    v_axis2: int,
    h_axis2: int,
    v_colors: set[int],
    h_colors: set[int],
    orphan_color: int,
    target: set[Coord],
) -> bool:
    if v_axis2 == inp.shape[1] - 1 or h_axis2 == inp.shape[0] - 1:
        return False
    if v_colors & h_colors:
        return False
    if (orphan_color in v_colors) == (orphan_color in h_colors):
        return False
    if len(target) < SHAPE_AREA_RANGE[0]:
        return False
    if np.array_equal(inp, expected):
        return False
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    v_axis2 = _choose_axis2(w)
    h_axis2 = _choose_axis2(h)

    palette = COLORS[:]
    random.shuffle(palette)
    v_pairs = [(palette[0], palette[1]), (palette[2], palette[3])]
    h_pairs = [(palette[4], palette[5]), (palette[6], palette[7])]
    v_colors = set(v_pairs[0] + v_pairs[1])
    h_colors = set(h_pairs[0] + h_pairs[1])

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    v_count = random.randint(*PAIR_COUNT_RANGE)
    h_count = random.randint(*PAIR_COUNT_RANGE)
    v_plan = [0, 1] + [random.randrange(2) for _ in range(v_count - 2)]
    h_plan = [0, 1] + [random.randrange(2) for _ in range(h_count - 2)]
    random.shuffle(v_plan)
    random.shuffle(h_plan)

    tasks: list[tuple[Orientation, int]] = [("V", idx) for idx in v_plan] + [("H", idx) for idx in h_plan]
    random.shuffle(tasks)
    for orientation, idx in tasks:
        pairs = v_pairs if orientation == "V" else h_pairs
        axis2 = v_axis2 if orientation == "V" else h_axis2
        if not _place_pair(grid, blocked, orientation, axis2, pairs[idx]):
            return None

    orphan_orientation = random.choice(["V", "H"])
    if orphan_orientation == "V":
        pair = random.choice(v_pairs)
        side_index = random.randrange(2)
        side = "left" if side_index == 0 else "right"
        color = pair[side_index]
        target_color = pair[1 - side_index]
        other_orientation, other_axis2 = "H", h_axis2
        axis2 = v_axis2
    else:
        pair = random.choice(h_pairs)
        side_index = random.randrange(2)
        side = "top" if side_index == 0 else "bottom"
        color = pair[side_index]
        target_color = pair[1 - side_index]
        other_orientation, other_axis2 = "V", v_axis2
        axis2 = h_axis2

    placed_orphan = _place_orphan(
        grid,
        blocked,
        orphan_orientation,
        axis2,
        other_orientation,
        other_axis2,
        side,
        color,
        target_color,
    )
    if placed_orphan is None:
        return None

    _cells, target, paint_color = placed_orphan
    expected = _expected_output(grid, target, paint_color)
    if not _semantic_accepts(grid, expected, v_axis2, h_axis2, v_colors, h_colors, color, target):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid dual-axis orphan input after {MAX_ATTEMPTS} attempts")
