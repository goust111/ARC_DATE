from __future__ import annotations

import random

import numpy as np


BG = 0
TAG_COLOR = 9
OBJECT_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
DIRECTIONS = ("up", "down", "left", "right")


def _components(mask: np.ndarray, *, connectivity: int) -> list[np.ndarray]:
    steps4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    steps8 = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )
    steps = steps4 if connectivity == 4 else steps8
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _normalize_coords(coords: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = sorted((r - min_r, c - min_c) for r, c in coords)
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    return norm, int(height), int(width)


def _transform_coords(coords: list[tuple[int, int]], height: int, width: int) -> tuple[list[tuple[int, int]], int, int]:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    new_coords = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    return _normalize_coords(new_coords)


def _sample_polyomino() -> tuple[list[tuple[int, int]], int, int]:
    area = random.randint(3, 8)
    coords = {(0, 0)}

    while len(coords) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in coords:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt in coords:
                    continue
                trial = coords | {nxt}
                rs = [rr for rr, _ in trial]
                cs = [cc for _, cc in trial]
                if max(rs) - min(rs) + 1 <= 5 and max(cs) - min(cs) + 1 <= 5:
                    frontier.append(nxt)
        coords.add(random.choice(frontier))

    norm, h, w = _normalize_coords(coords)
    return _transform_coords(norm, h, w)


def _sample_shape() -> tuple[list[tuple[int, int]], int, int]:
    mode = random.choices(("rect", "notched", "poly"), weights=(3, 2, 5), k=1)[0]

    if mode == "rect":
        while True:
            h = random.randint(1, 4)
            w = random.randint(1, 4)
            if h * w >= 3:
                coords = [(r, c) for r in range(h) for c in range(w)]
                return _transform_coords(coords, h, w)

    if mode == "notched":
        while True:
            h = random.randint(2, 4)
            w = random.randint(2, 4)
            coords = {(r, c) for r in range(h) for c in range(w)}
            corner = random.choice(((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))
            coords.remove(corner)
            if len(coords) >= 3:
                norm, nh, nw = _normalize_coords(coords)
                return _transform_coords(norm, nh, nw)

    return _sample_polyomino()


def _side_tag_offsets(coords: list[tuple[int, int]], height: int, width: int, direction: str) -> list[tuple[int, int]]:
    coord_set = set(coords)
    if direction == "right":
        return [(r, width) for r in range(height) if (r, width - 1) in coord_set]
    if direction == "left":
        return [(r, -1) for r in range(height) if (r, 0) in coord_set]
    if direction == "down":
        return [(height, c) for c in range(width) if (height - 1, c) in coord_set]
    assert direction == "up"
    return [(-1, c) for c in range(width) if (0, c) in coord_set]


def _copy_delta(direction: str, height: int, width: int) -> tuple[int, int]:
    if direction == "right":
        return (0, width + 1)
    if direction == "left":
        return (0, -(width + 1))
    if direction == "down":
        return (height + 1, 0)
    assert direction == "up"
    return (-(height + 1), 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _visible_output_cells(
    coords: list[tuple[int, int]],
    anchor: tuple[int, int],
    delta: tuple[int, int],
    copies: int,
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], bool, bool]:
    visible: set[tuple[int, int]] = set()
    copy_visible = False
    clipped = False
    h, w = shape
    ar, ac = anchor
    dr, dc = delta

    for k in range(copies + 1):
        for r, c in coords:
            rr = ar + r + k * dr
            cc = ac + c + k * dc
            if _in_bounds((h, w), rr, cc):
                visible.add((rr, cc))
                if k >= 1:
                    copy_visible = True
            elif k >= 1:
                clipped = True

    return visible, copy_visible, clipped


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each colored object in the direction indicated by its maroon side-tags.

    The number of maroon tags gives the number of copies. Copies are translated
    by one full object span plus one black separator cell, and maroon tags are
    erased in the output. Pixels that would fall outside the grid are clipped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    object_mask = (g != BG) & (g != TAG_COLOR)
    components = _components(object_mask, connectivity=4)
    out = g.copy()
    out[g == TAG_COLOR] = BG

    tag_positions = {tuple(map(int, rc)) for rc in np.argwhere(g == TAG_COLOR)}
    tag_usage = {pos: 0 for pos in tag_positions}

    for comp in components:
        coords = [(int(r), int(c)) for r, c in comp]
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochrome"
        assert len(coords) >= 3, "each object must contain at least three pixels"
        color = next(iter(colors))

        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1

        tags: set[tuple[int, int]] = set()
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == TAG_COLOR:
                        tags.add((rr, cc))

        assert tags, "each object must have at least one maroon tag"
        for pos in tags:
            tag_usage[pos] += 1

        if all(c > c1 for _, c in tags):
            delta = (0, width + 1)
        elif all(c < c0 for _, c in tags):
            delta = (0, -(width + 1))
        elif all(r > r1 for r, _ in tags):
            delta = (height + 1, 0)
        elif all(r < r0 for r, _ in tags):
            delta = (-(height + 1), 0)
        else:
            raise AssertionError("all tags for an object must lie on one side of its bounding box")

        for k in range(1, len(tags) + 1):
            dr = k * delta[0]
            dc = k * delta[1]
            for r, c in coords:
                rr = r + dr
                cc = c + dc
                if _in_bounds(g.shape, rr, cc):
                    prev = int(out[rr, cc])
                    assert prev in (BG, color), "copied objects should not overlap unrelated objects"
                    out[rr, cc] = color

    assert all(count == 1 for count in tag_usage.values()), "each tag must belong to exactly one object"
    return out


# --------------- generator ---------------

GEN_GRID_H_RANGE = (12, 19)
GEN_GRID_W_RANGE = (12, 19)
GEN_OBJECT_COUNT_RANGE = (1, 3)
GEN_MAX_TAGS = 3
GEN_SCENE_ATTEMPTS = 400
GEN_OBJECT_ATTEMPTS = 120

_GEN_CALL_COUNT = 0


def _make_object_spec(color: int, direction: str) -> dict | None:
    for _ in range(60):
        coords, height, width = _sample_shape()
        tag_offsets = _side_tag_offsets(coords, height, width, direction)
        if not tag_offsets:
            continue
        copies = random.randint(1, min(GEN_MAX_TAGS, len(tag_offsets)))
        return {
            "color": int(color),
            "coords": coords,
            "height": int(height),
            "width": int(width),
            "direction": direction,
            "tag_offsets": random.sample(tag_offsets, copies),
            "copies": int(copies),
            "delta": _copy_delta(direction, height, width),
        }
    return None


def _anchor_ranges(spec: dict, shape: tuple[int, int]) -> tuple[range, range] | None:
    h, w = shape
    obj_h = spec["height"]
    obj_w = spec["width"]
    r_min = 0
    r_max = h - obj_h
    c_min = 0
    c_max = w - obj_w

    if spec["direction"] == "up":
        r_min = 1
    elif spec["direction"] == "down":
        r_max = h - obj_h - 1
    elif spec["direction"] == "left":
        c_min = 1
    else:
        c_max = w - obj_w - 1

    if r_min > r_max or c_min > c_max:
        return None
    return range(r_min, r_max + 1), range(c_min, c_max + 1)


def _find_placement(
    spec: dict,
    grid_shape: tuple[int, int],
    input_blocked: set[tuple[int, int]],
    output_blocked: set[tuple[int, int]],
    *,
    require_truncation: bool | None,
) -> dict | None:
    ranges = _anchor_ranges(spec, grid_shape)
    if ranges is None:
        return None

    row_range, col_range = ranges
    anchors = [(r, c) for r in row_range for c in col_range]
    random.shuffle(anchors)

    for anchor in anchors:
        ar, ac = anchor
        object_cells = {(ar + r, ac + c) for r, c in spec["coords"]}
        tag_cells = {(ar + r, ac + c) for r, c in spec["tag_offsets"]}
        if any(cell in input_blocked for cell in object_cells | tag_cells):
            continue

        visible, copy_visible, clipped = _visible_output_cells(
            spec["coords"], anchor, spec["delta"], spec["copies"], grid_shape
        )
        if not copy_visible:
            continue
        if require_truncation is not None and clipped != require_truncation:
            continue
        if any(cell in output_blocked for cell in visible):
            continue

        return {
            "anchor": anchor,
            "object_cells": object_cells,
            "tag_cells": tag_cells,
            "visible_output": visible,
        }

    return None


def _build_scene(required_direction: str, require_truncation: bool) -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GEN_GRID_H_RANGE)
    width = random.randint(*GEN_GRID_W_RANGE)
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)

    object_count = random.randint(*GEN_OBJECT_COUNT_RANGE)
    colors = random.sample(OBJECT_COLORS, object_count)
    input_blocked: set[tuple[int, int]] = set()
    output_blocked: set[tuple[int, int]] = set()

    for index, color in enumerate(colors):
        placed = None
        for _ in range(GEN_OBJECT_ATTEMPTS):
            direction = required_direction if index == 0 else random.choice(DIRECTIONS)
            spec = _make_object_spec(color, direction)
            if spec is None:
                continue
            placed = _find_placement(
                spec,
                (height, width),
                input_blocked,
                output_blocked,
                require_truncation=require_truncation if index == 0 else None,
            )
            if placed is not None:
                _paint_cells(grid, placed["object_cells"], color)
                _paint_cells(grid, placed["tag_cells"], TAG_COLOR)
                _paint_cells(expected, placed["visible_output"], color)
                input_blocked |= _halo(placed["object_cells"] | placed["tag_cells"], (height, width))
                output_blocked |= _halo(placed["visible_output"], (height, width))
                break
        if placed is None:
            return None

    return grid, expected


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    required_direction = DIRECTIONS[(_GEN_CALL_COUNT - 1) % len(DIRECTIONS)]
    require_truncation = ((_GEN_CALL_COUNT - 1) // len(DIRECTIONS)) % 2 == 1

    for _ in range(GEN_SCENE_ATTEMPTS):
        scene = _build_scene(required_direction, require_truncation)
        if scene is None:
            continue
        grid, expected = scene
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
