from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
SIDES = ("up", "down", "left", "right")
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 250
SHAPE_RETRIES = 200


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for comp in _components(grid != bg):
        if len(comp) != 2:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r0 != r1 or c1 - c0 != 1:
            continue
        left = int(grid[r0, c0])
        right = int(grid[r0, c1])
        if left == right:
            continue
        if left in mapping:
            assert mapping[left] == right
        mapping[left] = right
    assert mapping, "expected at least one horizontal 2x1 legend entry"
    return mapping


def _find_fences(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    fences: list[dict[str, object]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            if len(comp) < 8:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue
            border = _perimeter_cells(r0, c0, r1, c1)
            if set(comp) != border:
                continue
            fences.append({"color": color, "bbox": (r0, c0, r1, c1)})
    assert fences, "expected at least one hollow rectangular fence"
    return fences


def _indicator_for_fence(grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> tuple[str, int]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[str, tuple[int, int], int]] = []
    strips = {
        "up": [(r0 - 1, c) for c in range(c0 + 1, c1)],
        "down": [(r1 + 1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0 - 1) for r in range(r0 + 1, r1)],
        "right": [(r, c1 + 1) for r in range(r0 + 1, r1)],
    }
    for side, cells in strips.items():
        present = []
        for r, c in cells:
            if 0 <= r < h and 0 <= c < w and int(grid[r, c]) != bg:
                present.append((r, c))
        if len(present) == 1:
            rr, cc = present[0]
            candidates.append((side, (rr, cc), int(grid[rr, cc])))
    assert len(candidates) == 1, "each fence must have exactly one adjacent indicator pixel"
    side, _cell, color = candidates[0]
    return side, color


def _island_mask(
    grid: np.ndarray, bbox: tuple[int, int, int, int], island_color: int
) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = bbox
    inside = grid[r0 + 1 : r1, c0 + 1 : c1] == island_color
    assert inside.any(), "indicator color must also appear as an interior island"
    pts = np.argwhere(inside)
    sr0, sc0 = pts.min(axis=0)
    sr1, sc1 = pts.max(axis=0)
    return inside[sr0 : sr1 + 1, sc0 : sc1 + 1].copy(), int(r0 + 1 + sr0), int(c0 + 1 + sc0)


def _projection_origin(
    fence_bbox: tuple[int, int, int, int],
    island_top: int,
    island_left: int,
    mask_shape: tuple[int, int],
    side: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = fence_bbox
    height, width = mask_shape
    if side == "right":
        return island_top, c1 + 1
    if side == "left":
        return island_top, c0 - width
    if side == "down":
        return r1 + 1, island_left
    assert side == "up"
    return r0 - height, island_left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each fence's interior island pattern outward on the side marked by the indicator pixel.

    - Hollow rectangular fences are keyed by color.
    - An isolated 2x1 horizontal legend maps fence color -> projection color.
    - The single exterior indicator pixel for a fence gives the projection side and the island color.
    - The tight island bbox is mirrored across the requested axis, then placed flush outside that side
      while keeping its parallel coordinate aligned.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    bg = BACKGROUND if np.any(g == BACKGROUND) else int(np.bincount(g.reshape(-1)).argmax())

    legend = _legend_mapping(g, bg)
    fences = _find_fences(g, bg)

    for fence in fences:
        fence_color = int(fence["color"])
        bbox = fence["bbox"]
        side, island_color = _indicator_for_fence(g, bg, bbox)
        assert fence_color in legend, "each fence color must appear in the legend"

        mask, island_top, island_left = _island_mask(g, bbox, island_color)
        mirrored = np.fliplr(mask) if side in ("left", "right") else np.flipud(mask)
        pr0, pc0 = _projection_origin(bbox, island_top, island_left, mirrored.shape, side)
        proj_color = legend[fence_color]

        for rr, cc in np.argwhere(mirrored):
            r = int(pr0 + rr)
            c = int(pc0 + cc)
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            out[r, c] = proj_color

    return out


def _random_shape(side: str, rng: random.Random) -> np.ndarray | None:
    min_h = 2 if side in ("up", "down") else 1
    min_w = 2 if side in ("left", "right") else 1
    for _ in range(SHAPE_RETRIES):
        height = rng.randint(min_h, 5)
        width = rng.randint(min_w, 5)
        density = rng.uniform(0.35, 0.65)
        mask = np.zeros((height, width), dtype=bool)
        for r in range(height):
            for c in range(width):
                mask[r, c] = rng.random() < density
        area = int(mask.sum())
        if area < 2 or area >= height * width:
            continue
        comps = _components(mask)
        if not 1 <= len(comps) <= 3:
            continue
        mask = _crop_mask(mask)
        if side in ("left", "right"):
            if mask.shape[1] < 2 or np.array_equal(mask, np.fliplr(mask)):
                continue
        else:
            if mask.shape[0] < 2 or np.array_equal(mask, np.flipud(mask)):
                continue
        return mask
    return None


def _make_task(fence_color: int, island_color: int, output_color: int, rng: random.Random) -> dict[str, object] | None:
    side = rng.choice(SIDES)
    shape = _random_shape(side, rng)
    if shape is None:
        return None
    shape_h, shape_w = map(int, shape.shape)
    extra_top = rng.randint(0, 2)
    extra_bottom = rng.randint(0, 2)
    extra_left = rng.randint(0, 2)
    extra_right = rng.randint(0, 2)
    return {
        "side": side,
        "fence_color": int(fence_color),
        "island_color": int(island_color),
        "output_color": int(output_color),
        "shape": shape,
        "inner_h": int(shape_h + extra_top + extra_bottom),
        "inner_w": int(shape_w + extra_left + extra_right),
        "offset_r": int(extra_top),
        "offset_c": int(extra_left),
    }


def _task_shape_origin(task: dict[str, object]) -> tuple[int, int]:
    return 1 + int(task["offset_r"]), 1 + int(task["offset_c"])


def _task_fence_size(task: dict[str, object]) -> tuple[int, int]:
    return int(task["inner_h"]) + 2, int(task["inner_w"]) + 2


def _task_input_cells(task: dict[str, object]) -> list[tuple[int, int, int]]:
    fence_h, fence_w = _task_fence_size(task)
    fence_color = int(task["fence_color"])
    island_color = int(task["island_color"])
    shape = np.asarray(task["shape"], dtype=bool)
    shape_top, shape_left = _task_shape_origin(task)

    cells: list[tuple[int, int, int]] = []
    for r, c in _perimeter_cells(0, 0, fence_h - 1, fence_w - 1):
        cells.append((r, c, fence_color))
    for rr, cc in np.argwhere(shape):
        cells.append((shape_top + int(rr), shape_left + int(cc), island_color))

    side = str(task["side"])
    if side == "right":
        cells.append((shape_top, fence_w, island_color))
    elif side == "left":
        cells.append((shape_top, -1, island_color))
    elif side == "down":
        cells.append((fence_h, shape_left, island_color))
    else:
        cells.append((-1, shape_left, island_color))
    return cells


def _task_projection_cells(task: dict[str, object]) -> list[tuple[int, int]]:
    shape = np.asarray(task["shape"], dtype=bool)
    mirrored = np.fliplr(shape) if task["side"] in ("left", "right") else np.flipud(shape)
    fence_h, fence_w = _task_fence_size(task)
    shape_top, shape_left = _task_shape_origin(task)

    if task["side"] == "right":
        base_r, base_c = shape_top, fence_w
    elif task["side"] == "left":
        base_r, base_c = shape_top, -mirrored.shape[1]
    elif task["side"] == "down":
        base_r, base_c = fence_h, shape_left
    else:
        base_r, base_c = -mirrored.shape[0], shape_left

    return [(base_r + int(rr), base_c + int(cc)) for rr, cc in np.argwhere(mirrored)]


def _task_reserve_bbox(task: dict[str, object], pad: int = 1) -> tuple[int, int, int, int]:
    coords = [(r, c) for r, c, _ in _task_input_cells(task)] + _task_projection_cells(task)
    r0, c0, r1, c1 = _bbox(coords)
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _place_tasks(
    height: int, width: int, tasks: list[dict[str, object]], rng: random.Random
) -> tuple[list[dict[str, object]], np.ndarray] | None:
    reserved = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, object]] = []
    tasks_sorted = sorted(
        tasks,
        key=lambda task: (int(task["inner_h"]) + 2) * (int(task["inner_w"]) + 2),
        reverse=True,
    )

    for task in tasks_sorted:
        rr0, cc0, rr1, cc1 = _task_reserve_bbox(task)
        min_r = -rr0
        max_r = height - 1 - rr1
        min_c = -cc0
        max_c = width - 1 - cc1
        if min_r > max_r or min_c > max_c:
            return None

        for _ in range(PLACEMENT_RETRIES):
            origin_r = rng.randint(min_r, max_r)
            origin_c = rng.randint(min_c, max_c)
            if reserved[origin_r + rr0 : origin_r + rr1 + 1, origin_c + cc0 : origin_c + cc1 + 1].any():
                continue
            reserved[origin_r + rr0 : origin_r + rr1 + 1, origin_c + cc0 : origin_c + cc1 + 1] = True
            placed_task = dict(task)
            placed_task["origin"] = (origin_r, origin_c)
            placed.append(placed_task)
            break
        else:
            return None

    return placed, reserved


def _place_legend(
    grid: np.ndarray, reserved: np.ndarray, tasks: list[dict[str, object]], rng: random.Random
) -> bool:
    count = len(tasks)
    group_h = 2 * count - 1
    group_w = 2
    height, width = grid.shape
    order = tasks[:]
    rng.shuffle(order)

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(0, height - group_h)
        c0 = rng.randint(0, width - group_w)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(height - 1, r0 + group_h)
        cc1 = min(width - 1, c0 + group_w)
        if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        for idx, task in enumerate(order):
            row = r0 + 2 * idx
            grid[row, c0] = int(task["fence_color"])
            grid[row, c0 + 1] = int(task["output_color"])
        return True
    return False


def _render_pair(
    height: int, width: int, tasks: list[dict[str, object]], rng: random.Random
) -> tuple[np.ndarray, np.ndarray] | None:
    placed = _place_tasks(height, width, tasks, rng)
    if placed is None:
        return None
    placed_tasks, reserved = placed
    inp = np.zeros((height, width), dtype=int)

    for task in placed_tasks:
        origin_r, origin_c = task["origin"]
        for rr, cc, color in _task_input_cells(task):
            inp[origin_r + rr, origin_c + cc] = int(color)

    if not _place_legend(inp, reserved, placed_tasks, rng):
        return None

    out = inp.copy()
    for task in placed_tasks:
        origin_r, origin_c = task["origin"]
        proj_color = int(task["output_color"])
        shape = np.asarray(task["shape"], dtype=bool)
        mirrored = np.fliplr(shape) if task["side"] in ("left", "right") else np.flipud(shape)
        for rr, cc in _task_projection_cells(task):
            out[origin_r + rr, origin_c + cc] = proj_color

    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    fence_count = rng.randint(1, 3)
    colors = rng.sample(range(1, 10), 3 * fence_count)
    tasks: list[dict[str, object]] = []
    for idx in range(fence_count):
        task = _make_task(colors[3 * idx], colors[3 * idx + 1], colors[3 * idx + 2], rng)
        if task is None:
            return None
        tasks.append(task)

    height = rng.randint(20, 30)
    width = rng.randint(20, 30)
    rendered = _render_pair(height, width, tasks, rng)
    if rendered is None:
        return None
    inp, expected = rendered

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
