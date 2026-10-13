from __future__ import annotations

import random

import numpy as np


DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(vals[int(counts.argmax())])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _find_wall(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    H, W = grid.shape
    walls: list[tuple[str, int, int]] = []

    for r in range(H):
        row = grid[r, :]
        if int(row[0]) != int(bg) and np.all(row == row[0]):
            walls.append(("h", int(r), int(row[0])))
    for c in range(W):
        col = grid[:, c]
        if int(col[0]) != int(bg) and np.all(col == col[0]):
            walls.append(("v", int(c), int(col[0])))

    assert len(walls) == 1, "expected exactly one full-line wall"
    return walls[0]


def _find_components(grid: np.ndarray, skip: set[int]) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(H):
        for c in range(W):
            col = int(g[r, c])
            if seen[r, c] or col in skip:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == col:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": int(col),
                    "coords": coords,
                    "bbox": (int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))),
                }
            )

    return comps


def _ray_len(cells: set[tuple[int, int]], start: tuple[int, int], direction: str) -> int:
    dr, dc = DIRS[direction]
    r, c = start
    count = 0
    while True:
        r += dr
        c += dc
        if (int(r), int(c)) not in cells:
            return int(count)
        count += 1


def _detect_direction(coords: list[tuple[int, int]]) -> str | None:
    cells = {(int(r), int(c)) for r, c in coords}
    candidates: list[str] = []

    for r, c in cells:
        dirs_here = [name for name, (dr, dc) in DIRS.items() if (int(r + dr), int(c + dc)) in cells]
        if len(dirs_here) != 2:
            continue

        has_vertical = any(name in {"up", "down"} for name in dirs_here)
        has_horizontal = any(name in {"left", "right"} for name in dirs_here)
        if not (has_vertical and has_horizontal):
            continue

        lengths = {name: _ray_len(cells, (int(r), int(c)), name) for name in dirs_here}
        short = [name for name, length in lengths.items() if int(length) == 1]
        long = [name for name, length in lengths.items() if int(length) >= 2]
        if len(short) == 1 and len(long) == 1:
            candidates.append(short[0])

    uniq = sorted(set(candidates))
    if len(uniq) == 1:
        return uniq[0]
    return None


def _direction_to_wall(coords: list[tuple[int, int]], wall_axis: str, wall_idx: int) -> str:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]

    if wall_axis == "v":
        if max(cols) < int(wall_idx):
            return "right"
        if min(cols) > int(wall_idx):
            return "left"
    else:
        if max(rows) < int(wall_idx):
            return "down"
        if min(rows) > int(wall_idx):
            return "up"

    raise AssertionError("object must lie fully on one side of the wall")


def _project_hits(coords: list[tuple[int, int]], direction: str, wall_idx: int) -> dict[tuple[int, int], int]:
    hits: dict[tuple[int, int], int] = {}

    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if direction == "right":
            hit = (int(rr), int(wall_idx))
            dist = int(wall_idx - cc)
        elif direction == "left":
            hit = (int(rr), int(wall_idx))
            dist = int(cc - wall_idx)
        elif direction == "down":
            hit = (int(wall_idx), int(cc))
            dist = int(wall_idx - rr)
        else:
            assert direction == "up"
            hit = (int(wall_idx), int(cc))
            dist = int(rr - wall_idx)

        assert dist > 0
        prev = hits.get(hit)
        if prev is None or int(dist) < int(prev):
            hits[hit] = int(dist)

    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored object onto the grey wall using the direction encoded by its internal L-arrow.

    Input structure:
    - The background is the dominant color.
    - A one-pixel-thick grey wall is a single full-width row or full-height column.
    - On one side of the wall sit several small monochrome 4-connected objects, each containing an L-shaped arrow cue.

    Rule:
    - Find each colored object and infer its arrow direction from the unique corner where one branch has length 1 and the perpendicular branch is longer.
    - Project every object pixel independently along that direction until it reaches the wall, marking the wall cell it hits.
    - Recolor every hit wall cell with the color of the nearest hitting object pixel; untouched wall cells stay grey, and all source objects disappear to background.

    Key insight:
    - The short branch of the embedded L gives the projection direction, while the wall receives a one-dimensional accumulation of all pixels with nearest-object priority on overlaps.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = _bg_color(g)
    wall_axis, wall_idx, wall_color = _find_wall(g, bg)

    out = np.full((H, W), int(bg), dtype=int)
    if wall_axis == "h":
        out[int(wall_idx), :] = int(wall_color)
    else:
        out[:, int(wall_idx)] = int(wall_color)

    best_hits: dict[tuple[int, int], tuple[int, int]] = {}
    for comp in _find_components(g, {int(bg), int(wall_color)}):
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        expected = _direction_to_wall(coords, wall_axis, wall_idx)
        direction = _detect_direction(coords)
        if direction is None:
            direction = expected
        assert direction == expected, "arrow direction must point toward the wall"

        for hit, dist in _project_hits(coords, direction, wall_idx).items():
            prev = best_hits.get(hit)
            if prev is None or int(dist) < int(prev[0]):
                best_hits[hit] = (int(dist), int(comp["color"]))

    for (r, c), (_dist, color) in best_hits.items():
        out[int(r), int(c)] = int(color)

    return out


WALL_COLOR = 5
BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]
OBJECT_COUNT_RANGE = (2, 5)
GENERATE_RETRIES = 400
PLACE_RETRIES = 120


def _shape_cells(direction: str, size: int, tip_at_first_end: bool) -> list[tuple[int, int]]:
    stem = int(size - 1)
    assert 4 <= size <= 7

    if direction == "right":
        tip_r = 0 if tip_at_first_end else stem - 1
        cells = [(int(r), 0) for r in range(stem)] + [(int(tip_r), 1)]
    elif direction == "left":
        tip_r = 0 if tip_at_first_end else stem - 1
        cells = [(int(r), 1) for r in range(stem)] + [(int(tip_r), 0)]
    elif direction == "down":
        tip_c = 0 if tip_at_first_end else stem - 1
        cells = [(0, int(c)) for c in range(stem)] + [(1, int(tip_c))]
    else:
        assert direction == "up"
        tip_c = 0 if tip_at_first_end else stem - 1
        cells = [(1, int(c)) for c in range(stem)] + [(0, int(tip_c))]

    return [(int(r), int(c)) for r, c in cells]


def _shape_dims(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [int(r) for r, _ in cells]
    cols = [int(c) for _, c in cells]
    return int(max(rows) + 1), int(max(cols) + 1)


def _place_cells(anchor: tuple[int, int], shape: list[tuple[int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(int(ar + r), int(ac + c)) for r, c in shape}


def _expanded(cells: set[tuple[int, int]], H: int, W: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((int(rr), int(cc)))
    return out


def _anchor_ranges(direction: str, H: int, W: int, wall_idx: int, shape: list[tuple[int, int]]) -> tuple[range, range] | None:
    h, w = _shape_dims(shape)

    if direction == "right":
        max_c0 = int(wall_idx - w - 1)
        if max_c0 < 0:
            return None
        return range(0, H - h + 1), range(0, max_c0 + 1)
    if direction == "left":
        min_c0 = int(wall_idx + 2)
        if min_c0 > W - w:
            return None
        return range(0, H - h + 1), range(min_c0, W - w + 1)
    if direction == "down":
        max_r0 = int(wall_idx - h - 1)
        if max_r0 < 0:
            return None
        return range(0, max_r0 + 1), range(0, W - w + 1)

    assert direction == "up"
    min_r0 = int(wall_idx + 2)
    if min_r0 > H - h:
        return None
    return range(min_r0, H - h + 1), range(0, W - w + 1)


def _scene_hits(objs: list[dict[str, object]], wall_idx: int) -> tuple[bool, bool]:
    hit_sources: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for idx, obj in enumerate(objs):
        hit_map = _project_hits(obj["coords"], obj["direction"], wall_idx)
        for hit, dist in hit_map.items():
            hit_sources.setdefault(hit, []).append((int(idx), int(dist)))

    overlap = any(len(v) >= 2 for v in hit_sources.values())
    tie = any(len({dist for _idx, dist in items}) != len(items) for items in hit_sources.values())
    return bool(overlap), bool(tie)


def _render_input(H: int, W: int, bg: int, wall_axis: str, wall_idx: int, objs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((H, W), int(bg), dtype=int)
    if wall_axis == "h":
        grid[int(wall_idx), :] = WALL_COLOR
    else:
        grid[:, int(wall_idx)] = WALL_COLOR

    for obj in objs:
        for r, c in obj["coords"]:
            grid[int(r), int(c)] = int(obj["color"])
    return grid


def _expected_output(H: int, W: int, bg: int, wall_axis: str, wall_idx: int, objs: list[dict[str, object]]) -> np.ndarray:
    out = np.full((H, W), int(bg), dtype=int)
    if wall_axis == "h":
        out[int(wall_idx), :] = WALL_COLOR
    else:
        out[:, int(wall_idx)] = WALL_COLOR

    best_hits: dict[tuple[int, int], tuple[int, int]] = {}
    for obj in objs:
        hit_map = _project_hits(obj["coords"], obj["direction"], wall_idx)
        for hit, dist in hit_map.items():
            prev = best_hits.get(hit)
            if prev is None or int(dist) < int(prev[0]):
                best_hits[hit] = (int(dist), int(obj["color"]))

    for (r, c), (_dist, color) in best_hits.items():
        out[int(r), int(c)] = int(color)
    return out


def _place_scene(
    H: int,
    W: int,
    wall_axis: str,
    wall_idx: int,
    direction: str,
    shapes: list[list[tuple[int, int]]],
    colors: list[int],
    need_overlap: bool,
) -> list[dict[str, object]] | None:
    placed: list[dict[str, object]] = []
    blocked: set[tuple[int, int]] = set()

    first_overlap_idx = random.randrange(len(shapes)) if need_overlap else -1
    second_overlap_idx = random.randrange(len(shapes)) if need_overlap else -1
    if need_overlap and second_overlap_idx == first_overlap_idx:
        second_overlap_idx = (int(second_overlap_idx) + 1) % len(shapes)

    for i, (shape, color) in enumerate(zip(shapes, colors)):
        ranges = _anchor_ranges(direction, H, W, wall_idx, shape)
        if ranges is None:
            return None
        row_range, col_range = ranges

        anchors = [(int(r0), int(c0)) for r0 in row_range for c0 in col_range]
        random.shuffle(anchors)
        chosen: dict[str, object] | None = None

        for anchor in anchors[:PLACE_RETRIES * 6]:
            cells = _place_cells(anchor, shape)
            if cells & blocked:
                continue

            coords = sorted(cells)
            obj = {
                "color": int(color),
                "coords": coords,
                "direction": direction,
            }

            if need_overlap and i == second_overlap_idx and placed:
                new_hits = set(_project_hits(coords, direction, wall_idx))
                if not any(new_hits & set(_project_hits(prev["coords"], direction, wall_idx)) for prev in placed):
                    continue

            chosen = obj
            break

        if chosen is None:
            return None

        placed.append(chosen)
        blocked |= _expanded(set(chosen["coords"]), H, W)

    overlap, tie = _scene_hits(placed, wall_idx)
    if tie:
        return None
    if need_overlap and not overlap:
        return None
    if not need_overlap and overlap:
        return None
    return placed


def _gen_attempt() -> np.ndarray | None:
    direction = random.choice(["up", "down", "left", "right"])
    wall_axis = "v" if direction in {"left", "right"} else "h"
    need_overlap = bool(random.getrandbits(1))

    H = random.randint(12, 24)
    W = random.randint(12, 24)

    if wall_axis == "v":
        if direction == "right":
            wall_idx = random.randint(5, W - 4)
        else:
            wall_idx = random.randint(3, W - 6)
    else:
        if direction == "down":
            wall_idx = random.randint(5, H - 4)
        else:
            wall_idx = random.randint(3, H - 6)

    obj_count = random.randint(*OBJECT_COUNT_RANGE)
    bg = random.choice(BG_CHOICES)
    palette = [c for c in range(10) if c not in {bg, WALL_COLOR}]
    if obj_count > len(palette):
        return None
    colors = random.sample(palette, obj_count)

    sizes = [random.randint(4, 7) for _ in range(obj_count)]
    shapes = [_shape_cells(direction, size, bool(random.getrandbits(1))) for size in sizes]

    placed = _place_scene(H, W, wall_axis, wall_idx, direction, shapes, colors, need_overlap)
    if placed is None:
        return None

    inp = _render_input(H, W, bg, wall_axis, wall_idx, placed)
    expected = _expected_output(H, W, bg, wall_axis, wall_idx, placed)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated placement attempts")
