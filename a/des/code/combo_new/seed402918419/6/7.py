from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRECTIONS = ("up", "down", "left", "right")

GRID_MIN = 12
GRID_MAX = 30
OBJECT_COUNT_RANGE = (1, 4)
SHAPE_SIZE_RANGE = (3, 6)
SHAPE_SPAN_RANGE = (2, 4)
SEED_GAP_RANGE = (2, 4)

MAX_SHAPE_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _normalize_coords(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r, max_r, min_c, max_c = _bbox(coords)
    del max_r, max_c
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _flip_coords(
    coords: tuple[tuple[int, int], ...], axis: str
) -> tuple[tuple[int, int], ...]:
    _min_r, max_r, _min_c, max_c = _bbox(list(coords))
    if axis == "horizontal":
        return tuple(sorted((max_r - r, c) for r, c in coords))
    if axis == "vertical":
        return tuple(sorted((r, max_c - c) for r, c in coords))
    raise AssertionError(f"unknown flip axis: {axis}")


def _reflect_point(
    point: tuple[int, int],
    direction: str,
    min_r: int,
    max_r: int,
    min_c: int,
    max_c: int,
) -> tuple[int, int]:
    r, c = point
    if direction in ("up", "down"):
        return min_r + max_r - r, c
    return r, min_c + max_c - c


def _nearest_unique(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    seed: tuple[int, int],
) -> tuple[int, int]:
    sr, sc = seed
    ranked = sorted(
        ((r - sr) * (r - sr) + (c - sc) * (c - sc), int(r), int(c))
        for r, c in coords
    )
    assert ranked, "expected a non-empty cluster"
    if len(ranked) > 1:
        assert ranked[0][0] < ranked[1][0], "nearest anchor pixel must be unique"
    return ranked[0][1], ranked[0][2]


def main(grid: np.ndarray) -> np.ndarray:
    """Grow a mirrored copy of each colored cluster from its matching seed point.

    Input structure:
    - The background is black.
    - Each non-black color appears as one small 4-connected cluster plus one isolated single-pixel seed of the same color.
    - Every seed lies clearly above, below, left, or right of its paired cluster, with at least two background cells in between.

    Rule:
    - For each color, pair the isolated seed with the non-singleton cluster of that color.
    - If the seed is above or below the cluster, mirror the cluster across a horizontal axis; if the seed is left or right, mirror it across a vertical axis.
    - Place the mirrored copy so that the mirrored image of the cluster pixel nearest to the seed lands exactly on the seed cell, while keeping the original cluster and seed unchanged.

    Key insight:
    - The seed provides both the mirror direction and the placement anchor: identify the color pair, infer the side of the cluster, then translate the flipped shape so its nearest mirrored pixel is pinned to the seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 0 < h <= 30 and 0 < w <= 30

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors, "expected at least one colored object"

    for color in colors:
        comps = _components_of_color(g, color)
        singletons = [comp[0] for comp in comps if len(comp) == 1]
        clusters = [comp for comp in comps if len(comp) > 1]
        assert len(comps) == 2, f"color {color} must have exactly one cluster and one seed"
        assert len(singletons) == 1 and len(clusters) == 1

        seed = singletons[0]
        cluster = clusters[0]
        assert 3 <= len(cluster) <= 6
        cluster_cells = set(cluster)
        min_r, max_r, min_c, max_c = _bbox(cluster)
        sr, sc = seed

        if sr < min_r and min_c <= sc <= max_c:
            direction = "up"
        elif sr > max_r and min_c <= sc <= max_c:
            direction = "down"
        elif sc < min_c and min_r <= sr <= max_r:
            direction = "left"
        elif sc > max_c and min_r <= sr <= max_r:
            direction = "right"
        else:
            raise AssertionError(f"seed {seed} is not axis-aligned with cluster color {color}")

        anchor = _nearest_unique(cluster, seed)
        mirrored_anchor = _reflect_point(anchor, direction, min_r, max_r, min_c, max_c)
        shift_r = sr - mirrored_anchor[0]
        shift_c = sc - mirrored_anchor[1]

        for cell in cluster:
            mr, mc = _reflect_point(cell, direction, min_r, max_r, min_c, max_c)
            rr, cc = mr + shift_r, mc + shift_c
            assert 0 <= rr < h and 0 <= cc < w, "mirrored copy exceeds the grid"
            if (rr, cc) in cluster_cells:
                raise AssertionError("mirrored copy overlaps the source cluster")
            if (rr, cc) != seed:
                assert out[rr, cc] == BG, "mirrored copy must land on empty background"
            out[rr, cc] = color

    return out


def _random_connected_shape(
    rng: np.random.Generator, direction: str
) -> tuple[tuple[int, int], ...] | None:
    axis = "horizontal" if direction in ("up", "down") else "vertical"
    for _ in range(MAX_SHAPE_ATTEMPTS):
        target = int(rng.integers(SHAPE_SIZE_RANGE[0], SHAPE_SIZE_RANGE[1] + 1))
        coords = {(0, 0)}
        while len(coords) < target:
            frontier: list[tuple[int, int]] = []
            seen = set()
            for r, c in coords:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in coords or (nr, nc) in seen:
                        continue
                    if not (-3 <= nr <= 3 and -3 <= nc <= 3):
                        continue
                    frontier.append((nr, nc))
                    seen.add((nr, nc))
            if not frontier:
                break
            coords.add(frontier[int(rng.integers(len(frontier)))])

        shape = _normalize_coords(coords)
        min_r, max_r, min_c, max_c = _bbox(list(shape))
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        if not (SHAPE_SPAN_RANGE[0] <= height <= SHAPE_SPAN_RANGE[1]):
            continue
        if not (SHAPE_SPAN_RANGE[0] <= width <= SHAPE_SPAN_RANGE[1]):
            continue
        if direction in ("up", "down") and width % 2 == 0:
            continue
        if direction in ("left", "right") and height % 2 == 0:
            continue
        if shape == _flip_coords(shape, "horizontal"):
            continue
        if shape == _flip_coords(shape, "vertical"):
            continue
        if shape == _flip_coords(shape, axis):
            continue
        return shape
    return None


def _make_object(
    rng: np.random.Generator, color: int, direction: str
) -> dict[str, object] | None:
    shape = _random_connected_shape(rng, direction)
    if shape is None:
        return None

    min_r, max_r, min_c, max_c = _bbox(list(shape))
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    gap = int(rng.integers(SEED_GAP_RANGE[0], SEED_GAP_RANGE[1] + 1))

    if direction == "up":
        seed = (-(gap + 1), width // 2)
    elif direction == "down":
        seed = (height + gap, width // 2)
    elif direction == "left":
        seed = (height // 2, -(gap + 1))
    else:
        seed = (height // 2, width + gap)

    try:
        anchor = _nearest_unique(shape, seed)
    except AssertionError:
        return None
    mirrored_anchor = _reflect_point(anchor, direction, min_r, max_r, min_c, max_c)
    shift_r = seed[0] - mirrored_anchor[0]
    shift_c = seed[1] - mirrored_anchor[1]
    copy = {
        tuple(
            int(v)
            for v in (
                _reflect_point(cell, direction, min_r, max_r, min_c, max_c)[0] + shift_r,
                _reflect_point(cell, direction, min_r, max_r, min_c, max_c)[1] + shift_c,
            )
        )
        for cell in shape
    }
    if copy.intersection(shape):
        return None

    input_cells = set(shape)
    input_cells.add(seed)
    output_cells = set(shape)
    output_cells.add(seed)
    output_cells.update(copy)

    out_min_r, out_max_r, out_min_c, out_max_c = _bbox(output_cells)
    norm_input = tuple(sorted((r - out_min_r, c - out_min_c) for r, c in input_cells))
    norm_output = tuple(sorted((r - out_min_r, c - out_min_c) for r, c in output_cells))

    return {
        "color": int(color),
        "direction": direction,
        "input": norm_input,
        "output": norm_output,
        "bbox_h": int(out_max_r - out_min_r + 1),
        "bbox_w": int(out_max_c - out_min_c + 1),
    }


def _choose_grid_shape(
    rng: np.random.Generator, objects: list[dict[str, object]]
) -> tuple[int, int] | None:
    min_h = max(GRID_MIN, max(int(obj["bbox_h"]) for obj in objects))
    min_w = max(GRID_MIN, max(int(obj["bbox_w"]) for obj in objects))
    reserve_area = sum((int(obj["bbox_h"]) + 2) * (int(obj["bbox_w"]) + 2) for obj in objects)
    choices = [
        (h, w)
        for h in range(min_h, GRID_MAX + 1)
        for w in range(min_w, GRID_MAX + 1)
        if h * w >= reserve_area
    ]
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _place_objects(
    rng: np.random.Generator, objects: list[dict[str, object]], height: int, width: int
) -> list[tuple[dict[str, object], int, int]] | None:
    reserved = np.zeros((height, width), dtype=bool)
    placed: list[tuple[dict[str, object], int, int]] = []

    ordered = sorted(
        objects,
        key=lambda obj: (len(obj["output"]), int(obj["bbox_h"]) * int(obj["bbox_w"])),
        reverse=True,
    )
    for obj in ordered:
        box_h = int(obj["bbox_h"])
        box_w = int(obj["bbox_w"])
        positions = [(r, c) for r in range(height - box_h + 1) for c in range(width - box_w + 1)]
        order = rng.permutation(len(positions))
        found = None
        for idx in order:
            top, left = positions[int(idx)]
            cells = [(top + r, left + c) for r, c in obj["output"]]
            if any(reserved[r, c] for r, c in cells):
                continue
            found = (top, left, cells)
            break
        if found is None:
            return None

        top, left, cells = found
        for r, c in cells:
            for nr in range(max(0, r - 1), min(height, r + 2)):
                for nc in range(max(0, c - 1), min(width, c + 2)):
                    reserved[nr, nc] = True
        placed.append((obj, top, left))

    return placed


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_objects = int(rng.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))
    direction_order = list(rng.permutation(np.array(DIRECTIONS)))
    directions = direction_order[:n_objects]
    colors = [int(color) for color in rng.choice(np.arange(1, 10), size=n_objects, replace=False)]

    objects: list[dict[str, object]] = []
    for color, direction in zip(colors, directions):
        obj = _make_object(rng, color, direction)
        if obj is None:
            return None
        objects.append(obj)

    for _ in range(8):
        shape = _choose_grid_shape(rng, objects)
        if shape is None:
            return None
        height, width = shape
        placed = _place_objects(rng, objects, height, width)
        if placed is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        for obj, top, left in placed:
            color = int(obj["color"])
            for r, c in obj["input"]:
                grid[top + r, left + c] = color

        try:
            out = main(grid)
        except AssertionError:
            return None
        if out.shape != grid.shape:
            return None
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-seed ARC input")
