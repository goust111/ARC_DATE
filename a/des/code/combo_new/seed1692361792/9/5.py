from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

DIRECTIONS = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ANCHOR_SIDES = ("top", "bottom", "left", "right")

GRID_H_RANGE = (14, 27)
GRID_W_RANGE = (14, 27)
OBJECT_COUNT_OPTIONS = np.array([1, 2, 3, 4], dtype=int)
OBJECT_COUNT_WEIGHTS = np.array([1, 2, 2, 1], dtype=float)
SHAPE_H_RANGE = (4, 8)
SHAPE_W_RANGE = (4, 8)
MAX_DISTANCE = 5
PLACEMENT_ATTEMPTS = 80
OBJECT_ATTEMPTS = 60
GEN_ATTEMPTS = 400


def _open_frame_mask(height: int, width: int, opening: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if opening != "up":
        mask[0, :] = True
    if opening != "down":
        mask[-1, :] = True
    if opening != "left":
        mask[:, 0] = True
    if opening != "right":
        mask[:, -1] = True
    return mask


def _is_connected(coords: np.ndarray) -> bool:
    cells = {tuple(map(int, cell)) for cell in coords.tolist()}
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _opening_direction(coords: np.ndarray) -> str:
    arr = np.asarray(coords, dtype=int)
    assert arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0
    r0, c0 = map(int, arr.min(axis=0))
    r1, c1 = map(int, arr.max(axis=0))
    counts = {
        "up": int(np.sum(arr[:, 0] == r0)),
        "down": int(np.sum(arr[:, 0] == r1)),
        "left": int(np.sum(arr[:, 1] == c0)),
        "right": int(np.sum(arr[:, 1] == c1)),
    }
    best = min(counts.values())
    dirs = [name for name, count in counts.items() if count == best]
    assert len(dirs) == 1, "opening side must be uniquely determined by edge coverage"
    return dirs[0]


def _anchor_distance(anchor: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = map(int, anchor)
    height, width = map(int, shape)
    if r == 0 or r == height - 1:
        return int(min(c, width - 1 - c))
    if c == 0 or c == width - 1:
        return int(min(r, height - 1 - r))
    raise AssertionError("anchor must lie on the outer boundary")


def _extract_specs(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, GREY)]
    specs: list[dict[str, object]] = []

    for color in colors:
        coords = np.argwhere(g == int(color))
        boundary_mask = (
            (coords[:, 0] == 0)
            | (coords[:, 0] == height - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == width - 1)
        )
        anchor_cells = coords[boundary_mask]
        assert anchor_cells.shape[0] == 1, "each color must have exactly one boundary anchor"
        object_cells = coords[~boundary_mask]
        assert object_cells.shape[0] > 0, "each color must have a non-anchor object"
        assert _is_connected(object_cells), "object cells must form one 4-connected component"
        anchor = tuple(map(int, anchor_cells[0]))
        specs.append(
            {
                "color": int(color),
                "anchor": anchor,
                "distance": _anchor_distance(anchor, g.shape),
                "direction": _opening_direction(object_cells),
                "cells": object_cells.astype(int, copy=False),
            }
        )

    assert specs, "expected at least one moving object"
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each open rectangular object along its opening direction by the corner-distance of its matching boundary anchor, leaving a grey afterimage.

    Input structure:
    - The background is black, and grey does not appear as an input object color.
    - Each non-black color appears as one interior 4-connected open rectangular frame plus one same-colored single-pixel anchor on the outer boundary.
    - Every frame is a U/C-like border whose bounding box has exactly one least-covered side, revealing the opening direction.

    Rule:
    - For each color, separate the boundary singleton anchor from the interior object.
    - Determine the object's opening direction by finding which bounding-box edge has the fewest object pixels.
    - Compute D as the anchor's distance along its boundary edge to the nearest grid corner.
    - Translate the whole object by D cells in its opening direction, paint vacated original-only cells grey, and erase every boundary anchor to black.

    Key insight:
    - The movement direction comes from shape geometry, while the movement distance comes from a remote same-colored boundary anchor's corner distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = np.full_like(g, BG)
    pending: list[tuple[int, np.ndarray]] = []

    for spec in _extract_specs(g):
        color = int(spec["color"])
        distance = int(spec["distance"])
        cells = np.asarray(spec["cells"], dtype=int)
        dr, dc = DIR_VECTORS[str(spec["direction"])]
        shifted = cells + np.array([dr * distance, dc * distance], dtype=int)
        assert bool(
            np.all(
                (0 <= shifted[:, 0])
                & (shifted[:, 0] < g.shape[0])
                & (0 <= shifted[:, 1])
                & (shifted[:, 1] < g.shape[1])
            )
        ), "shifted object must stay in bounds"

        shifted_set = {tuple(map(int, cell)) for cell in shifted.tolist()}
        for r, c in map(tuple, cells.tolist()):
            rr = int(r)
            cc = int(c)
            if (rr, cc) not in shifted_set:
                assert out[rr, cc] == BG, "grey afterimages must not collide"
                out[rr, cc] = GREY
        pending.append((color, shifted))

    for color, shifted in pending:
        assert np.all(out[shifted[:, 0], shifted[:, 1]] == BG), "shifted objects must not overlap"
        out[shifted[:, 0], shifted[:, 1]] = int(color)

    return out


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = map(int, shape)
    r0, c0, r1, c1 = map(int, box)
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(height - 1, r1 + pad),
        min(width - 1, c1 + pad),
    )


def _point_in_box(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = map(int, point)
    r0, c0, r1, c1 = map(int, box)
    return r0 <= r <= r1 and c0 <= c <= c1


def _anchor_candidates(height: int, width: int, side: str, distance: int) -> list[tuple[int, int]]:
    if side == "top":
        cols = sorted({int(distance), int(width - 1 - distance)})
        return [(0, c) for c in cols]
    if side == "bottom":
        cols = sorted({int(distance), int(width - 1 - distance)})
        return [(height - 1, c) for c in cols]
    if side == "left":
        rows = sorted({int(distance), int(height - 1 - distance)})
        return [(r, 0) for r in rows]
    rows = sorted({int(distance), int(height - 1 - distance)})
    return [(r, width - 1) for r in rows]


def _render_input(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray | None:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        cells = np.asarray(spec["cells"], dtype=int)
        anchor = tuple(map(int, spec["anchor"]))
        if np.any(grid[cells[:, 0], cells[:, 1]] != BG):
            return None
        grid[cells[:, 0], cells[:, 1]] = color
        if grid[anchor] != BG:
            return None
        grid[anchor] = color
    return grid


def _render_output(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray | None:
    out = np.full(shape, BG, dtype=int)
    shifted_specs: list[tuple[int, np.ndarray]] = []
    for spec in specs:
        color = int(spec["color"])
        distance = int(spec["distance"])
        cells = np.asarray(spec["cells"], dtype=int)
        dr, dc = DIR_VECTORS[str(spec["direction"])]
        shifted = cells + np.array([dr * distance, dc * distance], dtype=int)
        if not np.all(
            (0 <= shifted[:, 0])
            & (shifted[:, 0] < shape[0])
            & (0 <= shifted[:, 1])
            & (shifted[:, 1] < shape[1])
        ):
            return None
        shifted_set = {tuple(map(int, cell)) for cell in shifted.tolist()}
        for r, c in map(tuple, cells.tolist()):
            rr = int(r)
            cc = int(c)
            if (rr, cc) not in shifted_set:
                if out[rr, cc] != BG:
                    return None
                out[rr, cc] = GREY
        shifted_specs.append((color, shifted))
    for color, shifted in shifted_specs:
        if np.any(out[shifted[:, 0], shifted[:, 1]] != BG):
            return None
        out[shifted[:, 0], shifted[:, 1]] = int(color)
    return out


def _sample_one_object(
    rng: np.random.Generator,
    shape: tuple[int, int],
    color: int,
    state: dict[str, list[tuple[int, int, int, int]]],
) -> dict[str, object] | None:
    height, width = map(int, shape)

    for _ in range(OBJECT_ATTEMPTS):
        direction = str(rng.choice(DIRECTIONS))
        anchor_side = str(rng.choice(ANCHOR_SIDES))
        obj_h = int(rng.integers(*SHAPE_H_RANGE))
        obj_w = int(rng.integers(*SHAPE_W_RANGE))

        move_limit = int(height - obj_h - 2) if direction in ("up", "down") else int(width - obj_w - 2)
        anchor_limit = int((width - 1) // 2) if anchor_side in ("top", "bottom") else int((height - 1) // 2)
        d_max = min(MAX_DISTANCE, move_limit, anchor_limit)
        if d_max < 1:
            continue
        distance = int(rng.integers(1, d_max + 1))

        r_min = 1
        r_max = height - obj_h - 1
        c_min = 1
        c_max = width - obj_w - 1
        if direction == "up":
            r_min = max(r_min, distance)
        elif direction == "down":
            r_max = min(r_max, height - obj_h - 1 - distance)
        elif direction == "left":
            c_min = max(c_min, distance)
        else:
            c_max = min(c_max, width - obj_w - 1 - distance)
        if r_min > r_max or c_min > c_max:
            continue

        mask = _open_frame_mask(obj_h, obj_w, direction)
        pos_rows = np.arange(r_min, r_max + 1)
        pos_cols = np.arange(c_min, c_max + 1)
        positions = [(int(r), int(c)) for r in pos_rows for c in pos_cols]
        if not positions:
            continue
        positions = [positions[i] for i in rng.permutation(len(positions))]
        anchors = _anchor_candidates(height, width, anchor_side, distance)
        anchors = [anchors[i] for i in rng.permutation(len(anchors))]

        for anchor in anchors:
            anchor_box = _expand_box((anchor[0], anchor[1], anchor[0], anchor[1]), 1, shape)
            for r0, c0 in positions[:PLACEMENT_ATTEMPTS]:
                r1 = r0 + obj_h - 1
                c1 = c0 + obj_w - 1
                orig_box = (r0, c0, r1, c1)
                dr, dc = DIR_VECTORS[direction]
                shifted_box = (
                    r0 + dr * distance,
                    c0 + dc * distance,
                    r1 + dr * distance,
                    c1 + dc * distance,
                )
                sweep_box = (
                    min(orig_box[0], shifted_box[0]),
                    min(orig_box[1], shifted_box[1]),
                    max(orig_box[2], shifted_box[2]),
                    max(orig_box[3], shifted_box[3]),
                )

                if _point_in_box(anchor, sweep_box):
                    continue

                input_zone = _expand_box(orig_box, 1, shape)
                motion_zone = _expand_box(sweep_box, 1, shape)
                output_zone = _expand_box(shifted_box, 1, shape)
                if any(_boxes_intersect(input_zone, box) for box in state["input_zones"]):
                    continue
                if any(_boxes_intersect(motion_zone, box) for box in state["motion_zones"]):
                    continue
                if any(_boxes_intersect(output_zone, box) for box in state["output_zones"]):
                    continue
                if any(_boxes_intersect(anchor_box, box) for box in state["input_zones"]):
                    continue
                if any(_boxes_intersect(anchor_box, box) for box in state["motion_zones"]):
                    continue
                if any(_boxes_intersect(anchor_box, box) for box in state["output_zones"]):
                    continue
                if any(_boxes_intersect(anchor_box, box) for box in state["anchor_zones"]):
                    continue

                cells_local = np.argwhere(mask)
                cells = cells_local + np.array([r0, c0], dtype=int)
                spec = {
                    "color": int(color),
                    "anchor": anchor,
                    "distance": distance,
                    "direction": direction,
                    "cells": cells.astype(int, copy=False),
                    "input_zone": input_zone,
                    "motion_zone": motion_zone,
                    "output_zone": output_zone,
                    "anchor_zone": anchor_box,
                }
                return spec

    return None


def _valid_specs(shape: tuple[int, int], specs: list[dict[str, object]]) -> bool:
    grid = _render_input(shape, specs)
    expected = _render_output(shape, specs)
    if grid is None or expected is None:
        return False
    if np.any(grid == GREY):
        return False
    for spec in specs:
        cells = np.asarray(spec["cells"], dtype=int)
        if not np.all((cells[:, 0] > 0) & (cells[:, 0] < shape[0] - 1) & (cells[:, 1] > 0) & (cells[:, 1] < shape[1] - 1)):
            return False
        if _opening_direction(cells) != str(spec["direction"]):
            return False
        if _anchor_distance(tuple(map(int, spec["anchor"])), shape) != int(spec["distance"]):
            return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*GRID_H_RANGE))
    width = int(rng.integers(*GRID_W_RANGE))
    shape = (height, width)
    count = int(rng.choice(OBJECT_COUNT_OPTIONS, p=OBJECT_COUNT_WEIGHTS / OBJECT_COUNT_WEIGHTS.sum()))
    colors = [int(color) for color in rng.choice(PALETTE, size=count, replace=False)]

    state: dict[str, list[tuple[int, int, int, int]]] = {
        "input_zones": [],
        "motion_zones": [],
        "output_zones": [],
        "anchor_zones": [],
    }
    specs: list[dict[str, object]] = []

    for color in colors:
        spec = _sample_one_object(rng, shape, color, state)
        if spec is None:
            return None
        specs.append(spec)
        state["input_zones"].append(tuple(map(int, spec["input_zone"])))
        state["motion_zones"].append(tuple(map(int, spec["motion_zone"])))
        state["output_zones"].append(tuple(map(int, spec["output_zone"])))
        state["anchor_zones"].append(tuple(map(int, spec["anchor_zone"])))

    if not _valid_specs(shape, specs):
        return None
    return _render_input(shape, specs)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
