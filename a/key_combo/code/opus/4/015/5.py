from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox


Coord = tuple[int, int]
Shape = tuple[tuple[int, int], ...]

DIRS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ROLE_TO_COLOR = {"shared": Color.RED, "p3": Color.GREEN, "p4": Color.YELLOW}

SHAPE_CATALOG: tuple[Shape, ...] = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)),
)

GRID_SIDE_RANGE = (16, 23)
MAX_GENERATION_ATTEMPTS = 400
MIN_MARKER_PATH = 1


def _shape_to_mask(shape: Shape) -> tuple[tuple[int, ...], ...]:
    rows = max(r for r, _c in shape) + 1
    cols = max(c for _r, c in shape) + 1
    mask = np.zeros((rows, cols), dtype=int)
    for r, c in shape:
        mask[r, c] = 1
    return tuple(tuple(int(v) for v in row) for row in mask)


def _normalize_shape(coords: np.ndarray) -> tuple[tuple[int, ...], ...]:
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    rel = coords - np.array([[r0, c0]])
    patch[rel[:, 0], rel[:, 1]] = 1
    return tuple(tuple(int(v) for v in row) for row in patch)


def _apply_transform(shape: Shape, rotation: int, flip_h: bool, flip_v: bool) -> Shape:
    pts = np.asarray(shape, dtype=int)
    mask = np.array(_shape_to_mask(shape), dtype=int)
    for _ in range(rotation % 4):
        mask = np.rot90(mask)
    if flip_h:
        mask = np.fliplr(mask)
    if flip_v:
        mask = np.flipud(mask)
    return tuple(map(tuple, np.argwhere(mask > 0)))


def _random_shape_variant(shape: Shape) -> Shape:
    return _apply_transform(
        shape,
        rotation=random.randrange(4),
        flip_h=bool(random.randrange(2)),
        flip_v=bool(random.randrange(2)),
    )


def _shape_dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _shape_cells(shape: Shape, top: Coord) -> tuple[Coord, ...]:
    r0, c0 = top
    return tuple((r0 + r, c0 + c) for r, c in shape)


def _paint_cells(grid: np.ndarray, cells: tuple[Coord, ...] | list[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _expanded_cells(cells: tuple[Coord, ...] | list[Coord], shape: tuple[int, int], pad: int = 1) -> set[Coord]:
    height, width = shape
    out: set[Coord] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _scan_to_first_blue(
    grid: np.ndarray,
    blue_by_cell: dict[Coord, int],
    start: Coord,
    direction: Coord,
) -> tuple[int, tuple[Coord, ...], Coord] | None:
    height, width = grid.shape
    r, c = start
    dr, dc = direction
    path: list[Coord] = []
    r += dr
    c += dc
    while 0 <= r < height and 0 <= c < width:
        color = int(grid[r, c])
        if color == Color.BLACK:
            path.append((r, c))
            r += dr
            c += dc
            continue
        if color == Color.BLUE:
            return blue_by_cell[(r, c)], tuple(path), (r, c)
        return None
    return None


def _marker_hits(
    grid: np.ndarray,
    blue_by_cell: dict[Coord, int],
    marker: Coord,
) -> list[tuple[int, tuple[Coord, ...], Coord, Coord]]:
    hits: list[tuple[int, tuple[Coord, ...], Coord, Coord]] = []
    for direction in DIRS:
        hit = _scan_to_first_blue(grid, blue_by_cell, marker, direction)
        if hit is not None:
            comp_idx, path, first_blue = hit
            hits.append((comp_idx, path, direction, first_blue))
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate only the blue placeholders seen by gray markers, recolor them from same-shape templates, erase the unpaired duplicate, and draw gray sightlines.

    Input structure:
    - A black grid contains four small 4-connected blue placeholder sprites, with exactly one normalized shape duplicated and two other shapes unique.
    - Three separate colored template sprites use distinct non-blue/non-gray colors, and each template shape exactly matches one placeholder shape class.
    - Three isolated gray marker pixels each have exactly one unobstructed cardinal sightline to a blue placeholder; templates and other markers act as blockers.

    Rule:
    - Split the blue pixels into 4-connected placeholder components and the colored pixels into 4-connected template components.
    - For each gray marker, scan in the four cardinal directions; the only direction whose first non-black pixel is blue activates that blue component.
    - Recolor every activated placeholder with the color of the template whose bbox-normalized mask matches it exactly.
    - Erase every non-activated blue placeholder to black, keep the templates and gray markers, and fill the black cells on each valid sightline with gray.

    Key insight:
    - Shape matching alone is ambiguous because one placeholder shape appears twice; gray sightlines decide which duplicate is active, and only then does template shape determine the repaint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    blue_components = components_mask(g == Color.BLUE, connectivity=4)
    gray_markers = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.GREY)]
    template_components: list[dict[str, object]] = []
    for color in sorted(set(map(int, g.ravel())) - {Color.BLACK, Color.BLUE, Color.GREY}):
        for coords in components_mask(g == int(color), connectivity=4):
            template_components.append({"color": int(color), "coords": coords, "shape": _normalize_shape(coords)})

    assert len(blue_components) == 4, "expected four blue placeholders"
    assert len(gray_markers) == 3, "expected three gray markers"
    assert len(template_components) == 3, "expected three template components"

    shape_counts = Counter(_normalize_shape(coords) for coords in blue_components)
    assert sorted(shape_counts.values()) == [1, 1, 2], "expected exactly one duplicated blue shape"

    templates_by_shape: dict[tuple[tuple[int, ...], ...], int] = {}
    for template in template_components:
        shape = template["shape"]
        assert shape not in templates_by_shape, "template shapes must be unique"
        templates_by_shape[shape] = int(template["color"])

    blue_info: list[dict[str, object]] = []
    blue_by_cell: dict[Coord, int] = {}
    for idx, coords in enumerate(blue_components):
        shape = _normalize_shape(coords)
        assert shape in templates_by_shape, "every placeholder shape needs a matching template"
        cells = tuple(map(tuple, coords.tolist()))
        blue_info.append({"shape": shape, "cells": cells})
        for cell in cells:
            blue_by_cell[(int(cell[0]), int(cell[1]))] = idx

    activated: dict[int, tuple[Coord, ...]] = {}
    for marker in gray_markers:
        hits = _marker_hits(g, blue_by_cell, marker)
        assert len(hits) == 1, "each gray marker must have exactly one valid sightline"
        comp_idx, path, _direction, _first_blue = hits[0]
        assert comp_idx not in activated, "each activated placeholder must be seen by only one marker"
        activated[comp_idx] = path

    assert len(activated) == 3, "expected exactly three activated placeholders"

    out = g.copy()
    for idx, info in enumerate(blue_info):
        cells = info["cells"]
        if idx in activated:
            color = templates_by_shape[info["shape"]]
        else:
            color = Color.BLACK
        for r, c in cells:
            out[r, c] = color

    for path in activated.values():
        for r, c in path:
            out[r, c] = Color.GREY
    return out


def _sample_shapes() -> dict[str, Shape]:
    base_shapes = random.sample(list(SHAPE_CATALOG), 3)
    return {
        "shared": _random_shape_variant(base_shapes[0]),
        "p3": _random_shape_variant(base_shapes[1]),
        "p4": _random_shape_variant(base_shapes[2]),
    }


def _all_positions(shape: Shape, grid_shape: tuple[int, int]) -> list[Coord]:
    height, width = grid_shape
    sh, sw = _shape_dims(shape)
    return [(r, c) for r in range(height - sh + 1) for c in range(width - sw + 1)]


def _can_place(cells: tuple[Coord, ...], blocked: set[Coord]) -> bool:
    return all(cell not in blocked for cell in cells)


def _place_objects(
    height: int,
    width: int,
    shapes: dict[str, Shape],
) -> tuple[np.ndarray, dict[str, dict[str, object]], set[Coord]] | None:
    grid = np.zeros((height, width), dtype=int)
    blocked: set[Coord] = set()
    objects = {
        "p1": {"shape": shapes["shared"], "color": Color.BLUE, "active": True, "role": "shared"},
        "p2": {"shape": shapes["shared"], "color": Color.BLUE, "active": False, "role": "shared"},
        "p3": {"shape": shapes["p3"], "color": Color.BLUE, "active": True, "role": "p3"},
        "p4": {"shape": shapes["p4"], "color": Color.BLUE, "active": True, "role": "p4"},
        "t_shared": {"shape": shapes["shared"], "color": ROLE_TO_COLOR["shared"], "role": "shared"},
        "t_p3": {"shape": shapes["p3"], "color": ROLE_TO_COLOR["p3"], "role": "p3"},
        "t_p4": {"shape": shapes["p4"], "color": ROLE_TO_COLOR["p4"], "role": "p4"},
    }

    order = list(objects)
    random.shuffle(order)
    placed: dict[str, dict[str, object]] = {}

    for name in order:
        shape = objects[name]["shape"]
        candidates = _all_positions(shape, (height, width))
        random.shuffle(candidates)
        chosen = None
        for top in candidates:
            cells = _shape_cells(shape, top)
            if _can_place(cells, blocked):
                chosen = cells
                break
        if chosen is None:
            return None
        objects[name]["cells"] = chosen
        _paint_cells(grid, chosen, int(objects[name]["color"]))
        blocked |= _expanded_cells(chosen, (height, width), pad=1)
        placed[name] = objects[name]
    return grid, placed, blocked


def _candidate_markers(
    grid: np.ndarray,
    blocked: set[Coord],
    blue_by_cell: dict[Coord, str],
    target_name: str,
) -> list[dict[str, object]]:
    height, width = grid.shape
    candidates: list[dict[str, object]] = []
    for r in range(height):
        for c in range(width):
            if int(grid[r, c]) != Color.BLACK or (r, c) in blocked:
                continue
            hits = []
            for direction in DIRS:
                hit = _scan_to_first_blue(grid, blue_by_cell, (r, c), direction)
                if hit is not None:
                    comp_name, path, first_blue = hit
                    hits.append((comp_name, path, direction, first_blue))
            if len(hits) != 1:
                continue
            comp_name, path, direction, first_blue = hits[0]
            if comp_name != target_name or len(path) < MIN_MARKER_PATH:
                continue
            candidates.append(
                {
                    "marker": (r, c),
                    "path": path,
                    "direction": direction,
                    "first_blue": first_blue,
                }
            )
    return candidates


def _place_markers(
    grid: np.ndarray,
    blocked: set[Coord],
    placed: dict[str, dict[str, object]],
) -> list[dict[str, object]] | None:
    blue_by_cell: dict[Coord, str] = {}
    for name in ("p1", "p2", "p3", "p4"):
        for cell in placed[name]["cells"]:
            blue_by_cell[cell] = name

    markers: list[dict[str, object]] = []
    blocked_now = set(blocked)
    order = ["p1", "p3", "p4"]
    random.shuffle(order)

    for target in order:
        candidates = _candidate_markers(grid, blocked_now, blue_by_cell, target)
        if not candidates:
            return None
        random.shuffle(candidates)
        choice = candidates[0]
        marker = choice["marker"]
        grid[marker] = Color.GREY
        blocked_now |= _expanded_cells([marker], grid.shape, pad=1)
        choice["target"] = target
        markers.append(choice)

    if len({tuple(marker["direction"]) for marker in markers}) < 2:
        return None
    return markers


def _build_expected(
    input_grid: np.ndarray,
    placed: dict[str, dict[str, object]],
    markers: list[dict[str, object]],
) -> np.ndarray:
    out = input_grid.copy()
    for name in ("p1", "p3", "p4"):
        color = ROLE_TO_COLOR[str(placed[name]["role"])]
        _paint_cells(out, placed[name]["cells"], color)
    _paint_cells(out, placed["p2"]["cells"], Color.BLACK)
    for marker in markers:
        _paint_cells(out, marker["path"], Color.GREY)
    return out


def _validate_metadata(
    input_grid: np.ndarray,
    expected: np.ndarray,
    placed: dict[str, dict[str, object]],
    markers: list[dict[str, object]],
) -> bool:
    blue_shapes = [_shape_to_mask(tuple(placed[name]["shape"])) for name in ("p1", "p2", "p3", "p4")]
    if sorted(Counter(blue_shapes).values()) != [1, 1, 2]:
        return False
    if int(np.count_nonzero(input_grid == Color.GREY)) != 3:
        return False
    if int(np.count_nonzero(expected == Color.BLUE)) != 0:
        return False
    for marker in markers:
        hits = _marker_hits(
            input_grid,
            {cell: idx for idx, name in enumerate(("p1", "p2", "p3", "p4")) for cell in placed[name]["cells"]},
            marker["marker"],
        )
        if len(hits) != 1:
            return False
    if not np.all(expected[np.where(input_grid == Color.GREY)] == Color.GREY):
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randrange(*GRID_SIDE_RANGE)
    width = random.randrange(*GRID_SIDE_RANGE)
    shapes = _sample_shapes()

    placed_result = _place_objects(height, width, shapes)
    if placed_result is None:
        return None
    grid, placed, blocked = placed_result

    markers = _place_markers(grid, blocked, placed)
    if markers is None:
        return None

    expected = _build_expected(grid, placed, markers)
    if not _validate_metadata(grid, expected, placed, markers):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid sightline-shape puzzle")
