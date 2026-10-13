from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
OBJECT_COLORS = (1, 2, 3, 4)
MARKER = 5
SHADOW = 7
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SELECTED_BASE_SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
]

UNSELECTED_BASE_SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)),
]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expanded_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _edge_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                cells.append((r, c))
    return cells


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in (BG, MARKER, SHADOW):
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            coords: set[tuple[int, int]] = set()
            while queue:
                rr, cc = queue.popleft()
                coords.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _parse_scene(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    components = sorted(_components(g), key=lambda comp: comp["color"])
    cell_to_object = {
        cell: index for index, comp in enumerate(components) for cell in comp["coords"]
    }

    markers = [tuple(map(int, pos)) for pos in np.argwhere(g == MARKER)]
    shadows = {tuple(map(int, pos)) for pos in np.argwhere(g == SHADOW)}
    selected_by_object: dict[int, tuple[int, int]] = {}

    for marker in markers:
        neighbors = {
            cell_to_object[(nr, nc)]
            for nr, nc in _neighbors4(*marker)
            if (nr, nc) in cell_to_object
        }
        assert len(neighbors) == 1
        object_index = next(iter(neighbors))
        assert object_index not in selected_by_object
        selected_by_object[object_index] = marker

    used_shadows: set[tuple[int, int]] = set()
    parsed = []
    for index, comp in enumerate(components):
        info = dict(comp)
        info["selected"] = index in selected_by_object
        info["marker"] = selected_by_object.get(index)
        info["shadow"] = None
        info["delta"] = None

        adjacent_shadows = {
            (nr, nc)
            for r, c in comp["coords"]
            for nr, nc in _neighbors4(r, c)
            if (nr, nc) in shadows
        }

        if info["selected"]:
            assert len(adjacent_shadows) == 1
            shadow = next(iter(adjacent_shadows))
            assert shadow not in used_shadows
            used_shadows.add(shadow)
            assert info["marker"] is not None
            assert abs(shadow[0] - info["marker"][0]) + abs(shadow[1] - info["marker"][1]) != 1

            anchors = [
                (r, c)
                for r, c in comp["coords"]
                if abs(r - shadow[0]) + abs(c - shadow[1]) == 1
            ]
            assert len(anchors) == 1
            anchor = anchors[0]
            info["shadow"] = shadow
            info["delta"] = (shadow[0] - anchor[0], shadow[1] - anchor[1])
        else:
            assert not adjacent_shadows

        parsed.append(info)

    assert used_shadows == shadows
    return parsed


def _render_from_scene(
    grid: np.ndarray,
    parsed: list[dict],
    *,
    delta_overrides: dict[int, tuple[int, int]] | None = None,
    pad_overrides: dict[int, int] | None = None,
    erase_special: bool = True,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    delta_overrides = delta_overrides or {}
    pad_overrides = pad_overrides or {}

    for index, obj in enumerate(parsed):
        if not obj["selected"]:
            continue

        r0, c0, r1, c1 = obj["bbox"]
        pad = pad_overrides.get(index, 1)
        er0 = max(0, r0 - pad)
        ec0 = max(0, c0 - pad)
        er1 = min(out.shape[0] - 1, r1 + pad)
        ec1 = min(out.shape[1] - 1, c1 + pad)

        for r in range(er0, er1 + 1):
            for c in range(ec0, ec1 + 1):
                if int(out[r, c]) == BG:
                    out[r, c] = obj["color"]

        dr, dc = delta_overrides.get(index, obj["delta"])
        assert dr is not None and dc is not None
        for r, c in _edge_cells((er0, ec0, er1, ec1)):
            rr = r + dr
            cc = c + dc
            if _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == BG:
                out[rr, cc] = SHADOW

    if erase_special:
        out[np.asarray(grid) == MARKER] = BG
        out[np.asarray(grid) == SHADOW] = BG

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select gray-marked neighbor objects, expand them, and copy their shadow offset onto the expanded border.

    Input structure:
    - A black background contains several small 4-connected monochrome objects in distinct non-gray, non-orange colors.
    - Some objects have one gray marker pixel touching them by 4-neighborhood and one orange shadow pixel touching them by 4-neighborhood.
    - Unselected objects have neither a gray marker nor an orange shadow.

    Rule:
    - For each gray marker, find the unique adjacent colored object and treat it as selected.
    - For each selected object, read the orange shadow's one-step offset from its unique adjacent object pixel.
    - Expand that object's bounding box by one cell in every direction and fill only black cells in the expanded rectangle with the object's color.
    - For every edge cell of the expanded rectangle, copy the shadow offset once: paint orange at the shifted position if it is in bounds and currently black.
    - Finally erase all original gray markers and original orange shadow seed pixels to black; leave every unselected object unchanged.

    Key insight:
    - Gray picks which object is operated on, the selected object's bounding box defines the filled expansion zone, and the orange seed contributes only its relative direction, which becomes a repeated border decoration.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse_scene(g)
    return _render_from_scene(g, parsed)


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in shape))


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    coords = [tuple(cell) for cell in shape]
    for _ in range(random.randrange(4)):
        coords = [(-c, r) for r, c in coords]
    if random.random() < 0.5:
        coords = [(r, -c) for r, c in coords]
    return _normalize_shape(tuple(coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _translated_cells(
    shape: tuple[tuple[int, int], ...], top: int, left: int
) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape}


def _attachment_candidates(
    object_cells: set[tuple[int, int]]
) -> list[dict]:
    bbox = _bbox(object_cells)
    r0, c0, r1, c1 = bbox
    candidates: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for r, c in object_cells:
        for dr, dc in DIRS4:
            pos = (r + dr, c + dc)
            if pos in object_cells:
                continue
            pr, pc = pos
            if r0 <= pr <= r1 and c0 <= pc <= c1:
                continue
            candidates.setdefault(pos, []).append((r, c, dr, dc))

    out = []
    for pos, anchors in candidates.items():
        if len(anchors) != 1:
            continue
        ar, ac, dr, dc = anchors[0]
        out.append({"pos": pos, "anchor": (ar, ac), "delta": (dr, dc)})
    return out


def _reserve_cells_for_selected(
    object_cells: set[tuple[int, int]],
    delta: tuple[int, int],
    shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    bbox = _bbox(object_cells)
    expanded = _expanded_bbox(bbox, shape)
    dr, dc = delta
    shifted_edges = set()
    for r, c in _edge_cells(expanded):
        rr = r + dr
        cc = c + dc
        if not _in_bounds(shape, rr, cc):
            return None
        shifted_edges.add((rr, cc))

    r0, c0, r1, c1 = expanded
    filled = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    return filled | shifted_edges


def _moat(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _pick_selected_shapes(count: int) -> list[tuple[tuple[int, int], ...]]:
    pool = list(SELECTED_BASE_SHAPES)
    random.shuffle(pool)
    chosen = []
    used_dims = set()
    while pool and len(chosen) < count:
        shape = _transform_shape(pool.pop())
        dims = _shape_dims(shape)
        if dims in used_dims:
            continue
        used_dims.add(dims)
        chosen.append(shape)
    assert len(chosen) == count
    return chosen


def _pick_unselected_shape() -> tuple[tuple[int, int], ...]:
    return _transform_shape(random.choice(UNSELECTED_BASE_SHAPES))


def _candidate_placements(
    shape: tuple[tuple[int, int], ...],
    grid_shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    *,
    selected: bool,
    desired_delta: tuple[int, int] | None = None,
) -> list[dict]:
    height, width = _shape_dims(shape)
    h, w = grid_shape
    candidates = []

    for top in range(h - height + 1):
        for left in range(w - width + 1):
            object_cells = _translated_cells(shape, top, left)
            if object_cells & forbidden:
                continue

            if not selected:
                reserve = set(object_cells)
                if reserve & forbidden:
                    continue
                candidates.append(
                    {
                        "object_cells": object_cells,
                        "reserve": reserve,
                        "marker": None,
                        "shadow": None,
                        "delta": None,
                    }
                )
                continue

            assert desired_delta is not None
            reserve = _reserve_cells_for_selected(object_cells, desired_delta, grid_shape)
            if reserve is None or reserve & forbidden:
                continue
            attachments = [
                item
                for item in _attachment_candidates(object_cells)
                if _in_bounds(grid_shape, item["pos"][0], item["pos"][1])
            ]
            shadow_options = [item for item in attachments if item["delta"] == desired_delta]
            if not shadow_options:
                continue
            marker_options = attachments[:]
            random.shuffle(shadow_options)
            random.shuffle(marker_options)
            for shadow in shadow_options:
                for marker in marker_options:
                    if marker["pos"] == shadow["pos"]:
                        continue
                    if (
                        abs(marker["pos"][0] - shadow["pos"][0])
                        + abs(marker["pos"][1] - shadow["pos"][1])
                    ) == 1:
                        continue
                    candidates.append(
                        {
                            "object_cells": object_cells,
                            "reserve": reserve,
                            "marker": marker["pos"],
                            "shadow": shadow["pos"],
                            "delta": desired_delta,
                        }
                    )
    random.shuffle(candidates)
    return candidates


def _paint_object(
    grid: np.ndarray,
    color: int,
    placement: dict,
) -> None:
    for r, c in placement["object_cells"]:
        grid[r, c] = color
    if placement["marker"] is not None:
        mr, mc = placement["marker"]
        sr, sc = placement["shadow"]
        grid[mr, mc] = MARKER
        grid[sr, sc] = SHADOW


def _selected_count_for(object_count: int) -> int:
    if object_count == 3:
        return 2
    return random.choice((2, 3))


def _generate_layout() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    grid = np.zeros((h, w), dtype=int)

    object_count = random.choice((3, 4))
    selected_count = _selected_count_for(object_count)
    colors = random.sample(list(OBJECT_COLORS), object_count)
    selected_colors = set(random.sample(colors, selected_count))
    shadow_dirs = random.sample(DIRS4, selected_count)
    selected_shapes = _pick_selected_shapes(selected_count)

    specs = []
    selected_index = 0
    for color in colors:
        if color in selected_colors:
            specs.append(
                {
                    "color": color,
                    "selected": True,
                    "shape": selected_shapes[selected_index],
                    "delta": shadow_dirs[selected_index],
                }
            )
            selected_index += 1
        else:
            specs.append(
                {
                    "color": color,
                    "selected": False,
                    "shape": _pick_unselected_shape(),
                    "delta": None,
                }
            )

    specs.sort(
        key=lambda item: (
            0 if item["selected"] else 1,
            -len(item["shape"]),
            item["color"],
        )
    )

    forbidden: set[tuple[int, int]] = set()
    used_selected_dims = set()
    for spec in specs:
        if spec["selected"]:
            dims = _shape_dims(spec["shape"])
            if dims in used_selected_dims:
                return None
            used_selected_dims.add(dims)

        candidates = _candidate_placements(
            spec["shape"],
            grid.shape,
            forbidden,
            selected=spec["selected"],
            desired_delta=spec["delta"],
        )
        if not candidates:
            return None

        placement = random.choice(candidates[: min(20, len(candidates))])
        _paint_object(grid, spec["color"], placement)
        forbidden |= _moat(placement["reserve"], grid.shape)

    return grid


def _check_visible_hole(
    output: np.ndarray,
    obj: dict,
    pos: tuple[int, int],
) -> bool:
    box = _expanded_bbox(obj["bbox"], output.shape)
    if not (box[0] <= pos[0] <= box[2] and box[1] <= pos[1] <= box[3]):
        return False
    if int(output[pos]) != BG:
        return False
    for nr, nc in _neighbors4(*pos):
        if _in_bounds(output.shape, nr, nc) and int(output[nr, nc]) == obj["color"]:
            return True
    return False


def _validate_scene(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    try:
        parsed = _parse_scene(grid)
    except AssertionError:
        return False

    if len(parsed) not in (3, 4):
        return False
    colors = [obj["color"] for obj in parsed]
    if len(set(colors)) != len(colors):
        return False
    if not set(colors).issubset(set(OBJECT_COLORS)):
        return False

    selected = [obj for obj in parsed if obj["selected"]]
    unselected = [obj for obj in parsed if not obj["selected"]]
    if len(selected) not in (2, 3) or not unselected:
        return False

    deltas = [obj["delta"] for obj in selected]
    if len(deltas) != len(set(deltas)):
        return False

    for obj in parsed:
        if not (3 <= len(obj["coords"]) <= 6):
            return False
        if obj["selected"]:
            assert obj["marker"] is not None and obj["shadow"] is not None and obj["delta"] is not None
            if not _reserve_cells_for_selected(obj["coords"], obj["delta"], grid.shape):
                return False

    expected = _render_from_scene(grid, parsed)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    selected_indices = [index for index, obj in enumerate(parsed) if obj["selected"]]
    first = selected_indices[0]
    wrong_dirs = [direction for direction in DIRS4 if direction != parsed[first]["delta"]]
    wrong_shadow = _render_from_scene(
        grid,
        parsed,
        delta_overrides={first: random.choice(wrong_dirs)},
    )
    if np.array_equal(wrong_shadow, actual):
        return False

    no_expand = _render_from_scene(
        grid,
        parsed,
        pad_overrides={first: 0},
    )
    if np.array_equal(no_expand, actual):
        return False

    if not _check_visible_hole(actual, parsed[first], parsed[first]["marker"]):
        return False
    if not _check_visible_hole(actual, parsed[first], parsed[first]["shadow"]):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_layout()
        if grid is not None and _validate_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
