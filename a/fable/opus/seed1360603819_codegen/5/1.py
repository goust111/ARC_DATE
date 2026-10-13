from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

GEN_RETRIES = 400
GRID_H_RANGE = (22, 32)
GRID_W_RANGE = (24, 34)
LEGEND_COUNT_RANGE = (2, 4)
LEGEND_LENGTH_VALUES = tuple(range(3, 10))
OBJECT_SIZE_RANGE = (6, 11)
OBJECT_COUNT_BY_LEGEND = {
    2: (2, 3),
    3: (3, 4),
    4: (3, 5),
}
HOLE_SIZE_OPTIONS = ((1, 1), (1, 2), (2, 1), (2, 2))
CORE_RETRIES = 120
PLACE_RETRIES = 160


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _color_components(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, np.array(coords, dtype=int)))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _prune_arrow(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = {tuple(map(int, coord)) for coord in coords}
    queue = deque()
    for cell in cells:
        degree = sum((nb in cells) for nb in _neighbors4(*cell))
        if degree <= 1:
            queue.append(cell)

    removed: list[tuple[int, int]] = []
    while queue:
        cell = queue.popleft()
        if cell not in cells:
            continue
        degree = sum((nb in cells) for nb in _neighbors4(*cell))
        if degree > 1:
            continue
        cells.remove(cell)
        removed.append(cell)
        for nb in _neighbors4(*cell):
            if nb in cells:
                queue.append(nb)

    core = np.array(sorted(cells), dtype=int)
    arrow = np.array(sorted(removed), dtype=int)
    assert core.size > 0, "object core must remain after pruning"
    assert len(removed) == 3, "expected a 3-pixel arrow"
    return core, arrow


def _count_holes(grid: np.ndarray, color: int, core: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(core)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros(patch.shape, dtype=bool)
    mask[core[:, 0] - r0, core[:, 1] - c0] = True
    assert np.all(patch[mask] == color)

    holes = 0
    for comp in _components(~mask):
        rr = comp[:, 0]
        cc = comp[:, 1]
        if np.any(rr == 0) or np.any(rr == patch.shape[0] - 1) or np.any(cc == 0) or np.any(cc == patch.shape[1] - 1):
            continue
        assert np.all(patch[rr, cc] == BLACK), "core interior holes must be black"
        holes += 1
    return holes


def _arrow_root_and_dir(core: np.ndarray, arrow: np.ndarray) -> tuple[tuple[int, int], str]:
    core_set = {tuple(map(int, coord)) for coord in core}
    roots = []
    for cell in map(tuple, arrow):
        if any(nb in core_set for nb in _neighbors4(*cell)):
            roots.append(cell)
    assert len(roots) == 1, "arrow must have a unique root touching the object"
    root = roots[0]
    r0, c0, r1, c1 = _bbox(core)
    rr, cc = root
    if rr < r0:
        return root, "up"
    if rr > r1:
        return root, "down"
    if cc < c0:
        return root, "left"
    if cc > c1:
        return root, "right"
    raise AssertionError("arrow root must lie outside the object bbox")


def _legend_lengths(grid: np.ndarray) -> dict[int, int]:
    bars: list[tuple[int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if c0 != 0 or r0 != r1:
            continue
        assert len(comp) == c1 - c0 + 1, "legend bar must be a solid 1-row segment"
        bars.append((r0, c1 - c0 + 1))
    bars.sort()
    assert bars, "expected left-edge grey legend bars"
    return {i + 1: length for i, (_row, length) in enumerate(bars)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Map each object's black-hole count to the length of the h-th grey legend bar,
    then extend a same-color ray from the attached arrow in the arrow direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    hole_to_length = _legend_lengths(g)
    out = g.copy()

    for color, coords in _color_components(g):
        core, arrow = _prune_arrow(coords)
        hole_count = _count_holes(g, color, core)
        assert hole_count in hole_to_length, "hole count must be present in the legend"
        ray_len = hole_to_length[hole_count]

        (rr, cc), direction = _arrow_root_and_dir(core, arrow)
        dr, dc = DIRS[direction]
        for step in range(ray_len):
            nr = rr + dr * step
            nc = cc + dc * step
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = color

    return out


def _has_leaf(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    cells = {tuple(map(int, coord)) for coord in coords}
    for cell in cells:
        degree = sum((nb in cells) for nb in _neighbors4(*cell))
        if degree <= 1:
            return True
    return False


def _make_core_mask(hole_count: int) -> np.ndarray | None:
    min_size = max(OBJECT_SIZE_RANGE[0], 4 + hole_count)
    max_size = OBJECT_SIZE_RANGE[1]
    for _ in range(CORE_RETRIES):
        oh = random.randint(min_size, max_size)
        ow = random.randint(min_size, max_size)
        mask = np.ones((oh, ow), dtype=bool)
        blocked = np.zeros((oh, ow), dtype=bool)
        placed = 0
        for _ in range(hole_count * 30):
            if placed == hole_count:
                break
            hh, ww = random.choice(HOLE_SIZE_OPTIONS)
            if oh - hh - 1 <= 1 or ow - ww - 1 <= 1:
                continue
            hr = random.randint(1, oh - hh - 1)
            hc = random.randint(1, ow - ww - 1)
            rr0 = max(0, hr - 1)
            rr1 = min(oh, hr + hh + 1)
            cc0 = max(0, hc - 1)
            cc1 = min(ow, hc + ww + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            mask[hr : hr + hh, hc : hc + ww] = False
            blocked[rr0:rr1, cc0:cc1] = True
            placed += 1
        if placed != hole_count:
            continue
        if _has_leaf(mask):
            continue
        return mask
    return None


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _line_cells(root: tuple[int, int], direction: str, length: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    cells = []
    for step in range(length):
        rr = root[0] + dr * step
        cc = root[1] + dc * step
        if _in_bounds(shape, rr, cc):
            cells.append((rr, cc))
    return cells


def _component_cells(
    core_mask: np.ndarray,
    r0: int,
    c0: int,
    direction: str,
    anchor: int,
    head_offset: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    oh, ow = core_mask.shape
    cells = [(r0 + rr, c0 + cc) for rr, cc in np.argwhere(core_mask)]
    if direction == "right":
        root = (r0 + anchor, c0 + ow)
        arrow = [root, (root[0], root[1] + 1), (root[0] + head_offset, root[1] + 1)]
    elif direction == "left":
        root = (r0 + anchor, c0 - 1)
        arrow = [root, (root[0], root[1] - 1), (root[0] + head_offset, root[1] - 1)]
    elif direction == "down":
        root = (r0 + oh, c0 + anchor)
        arrow = [root, (root[0] + 1, root[1]), (root[0] + 1, root[1] + head_offset)]
    else:
        assert direction == "up"
        root = (r0 - 1, c0 + anchor)
        arrow = [root, (root[0] - 1, root[1]), (root[0] - 1, root[1] + head_offset)]
    return cells + arrow, root


def _mark_component_buffer(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _place_single_object(
    shape: tuple[int, int],
    blocked: np.ndarray,
    hole_count: int,
    direction: str,
    ray_len: int,
) -> dict | None:
    core_mask = _make_core_mask(hole_count)
    if core_mask is None:
        return None
    oh, ow = core_mask.shape

    if direction in ("left", "right"):
        anchors = list(range(1, oh - 1))
    else:
        anchors = list(range(1, ow - 1))
    if not anchors:
        return None

    for _ in range(PLACE_RETRIES):
        anchor = random.choice(anchors)
        head_offset = random.choice((-1, 1))
        if direction in ("left", "right") and not (0 <= anchor + head_offset < oh):
            continue
        if direction in ("up", "down") and not (0 <= anchor + head_offset < ow):
            continue

        r_lo = 0
        r_hi = shape[0] - oh
        c_lo = 0
        c_hi = shape[1] - ow
        if direction == "left":
            c_lo = max(c_lo, 2)
        if direction == "right":
            c_hi = min(c_hi, shape[1] - ow - 2)
        if direction == "up":
            r_lo = max(r_lo, 2)
        if direction == "down":
            r_hi = min(r_hi, shape[0] - oh - 2)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        component_cells, root = _component_cells(core_mask, r0, c0, direction, anchor, head_offset)
        ray_cells = _line_cells(root, direction, ray_len, shape)

        if any(not _in_bounds(shape, r, c) for r, c in component_cells):
            continue
        if any(blocked[r, c] for r, c in component_cells):
            continue
        if any(blocked[r, c] for r, c in ray_cells):
            continue

        return {
            "core_mask": core_mask,
            "r0": r0,
            "c0": c0,
            "direction": direction,
            "anchor": anchor,
            "head_offset": head_offset,
            "root": root,
            "component_cells": component_cells,
            "ray_cells": ray_cells,
        }
    return None


def _legend_lengths_sample(count: int) -> list[int]:
    for _ in range(60):
        lengths = random.sample(LEGEND_LENGTH_VALUES, count)
        if lengths != sorted(lengths) and lengths != sorted(lengths, reverse=True):
            return lengths
    return random.sample(LEGEND_LENGTH_VALUES, count)


def _object_specs(legend_count: int) -> list[tuple[int, str]]:
    lo, hi = OBJECT_COUNT_BY_LEGEND[legend_count]
    object_count = random.randint(lo, hi)
    counts = list(range(1, legend_count + 1))
    random.shuffle(counts)
    hole_counts = counts[: min(object_count, legend_count)]
    while len(hole_counts) < object_count:
        hole_counts.append(random.randint(1, legend_count))
    random.shuffle(hole_counts)

    dirs = list(DIRS)
    if object_count <= 4:
        directions = random.sample(dirs, object_count)
    else:
        directions = dirs + [random.choice(dirs) for _ in range(object_count - 4)]
        random.shuffle(directions)
    return list(zip(hole_counts, directions))


def _render_expected(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        for r, c in obj["ray_cells"]:
            out[r, c] = obj["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_H_RANGE)
    cols = random.randint(*GRID_W_RANGE)
    legend_count = random.randint(*LEGEND_COUNT_RANGE)
    legend_lengths = _legend_lengths_sample(legend_count)
    specs = _object_specs(legend_count)
    max_objects = len(specs)
    if max_objects > len(PALETTE):
        return None

    grid = np.full((rows, cols), BLACK, dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)

    for idx, length in enumerate(legend_lengths):
        rr = 2 * idx
        if rr >= rows:
            return None
        if length > cols:
            return None
        grid[rr, 0:length] = GREY
        _mark_component_buffer(blocked, [(rr, cc) for cc in range(length)])

    colors = random.sample(PALETTE, max_objects)
    planned = []
    for (hole_count, direction), color in zip(specs, colors):
        planned.append(
            {
                "hole_count": hole_count,
                "direction": direction,
                "ray_len": legend_lengths[hole_count - 1],
                "color": color,
            }
        )
    planned.sort(key=lambda item: item["ray_len"], reverse=True)

    placed: list[dict] = []
    for spec in planned:
        placed_obj = _place_single_object(
            grid.shape,
            blocked,
            spec["hole_count"],
            spec["direction"],
            spec["ray_len"],
        )
        if placed_obj is None:
            return None
        placed_obj.update(spec)
        placed.append(placed_obj)
        _mark_component_buffer(blocked, placed_obj["component_cells"])
        for r, c in placed_obj["ray_cells"]:
            blocked[r, c] = True

        for r, c in placed_obj["component_cells"]:
            grid[r, c] = spec["color"]

    expected = _render_expected(grid, placed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
