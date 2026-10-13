from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
POINTER = 4
FRAME = 5
PALETTE = (1, 2, 3, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")

MAX_GEN_ATTEMPTS = 400
OBJECT_COUNT_VALUES = (1, 2, 3, 4)
OBJECT_COUNT_WEIGHTS = (1, 2, 3, 3)
INTERIOR_HEIGHT_RANGE = (3, 6)
INTERIOR_WIDTH_RANGE = (3, 6)
MAX_THICKNESS = 6
PLACEMENT_TRIES = 120
SHAPE_TRIES = 40
CLIP_PROB = 0.35


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    ring = set(cells)
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    if ring != expected:
        return None
    return r0, c0, r1, c1


def _pointer_direction(pointer: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = pointer
    r0, c0, r1, c1 = bbox
    sides = []
    if r == r0 + 1:
        sides.append("up")
    if r == r1 - 1:
        sides.append("down")
    if c == c0 + 1:
        sides.append("left")
    if c == c1 - 1:
        sides.append("right")
    assert len(sides) == 1, "pointer must sit on exactly one inner wall"
    return sides[0]


def _project_cells(
    color_points: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    direction: str,
    thickness: int,
    grid_shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = grid_shape
    r0, c0, r1, c1 = bbox
    shadow: set[tuple[int, int]] = set()
    if direction == "up":
        active = sorted({c for _, c in color_points})
        for col in active:
            for step in range(1, thickness + 1):
                rr = r0 - step
                if 0 <= rr < h:
                    shadow.add((rr, col))
    elif direction == "down":
        active = sorted({c for _, c in color_points})
        for col in active:
            for step in range(1, thickness + 1):
                rr = r1 + step
                if 0 <= rr < h:
                    shadow.add((rr, col))
    elif direction == "left":
        active = sorted({r for r, _ in color_points})
        for row in active:
            for step in range(1, thickness + 1):
                cc = c0 - step
                if 0 <= cc < w:
                    shadow.add((row, cc))
    else:
        active = sorted({r for r, _ in color_points})
        for row in active:
            for step in range(1, thickness + 1):
                cc = c1 + step
                if 0 <= cc < w:
                    shadow.add((row, cc))
    return shadow


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each framed interior pattern outward from the wall indicated by the yellow pointer.

    Input structure:
    - The background is black.
    - Several disjoint grey one-cell-thick rectangular frame rings appear on the grid.
    - Inside each frame there is exactly one yellow pointer pixel on one inner wall and
      a set of same-colored non-pointer pixels forming an interior pattern.

    Rule:
    - For each grey frame, read which inner wall the yellow pointer touches; that wall's
      outward normal is the projection direction.
    - Ignore the grey frame and yellow pointer, count the remaining colored pixels as N,
      and compress them into a one-dimensional silhouette on the axis perpendicular to
      the projection direction.
    - Starting just outside the corresponding outer wall, paint that silhouette outward
      for exactly N cells of thickness, clipping at the grid boundary.

    Key insight:
    The yellow pixel selects the outward direction, while the number of non-pointer
    colored pixels controls the projection thickness and their occupied rows or columns
    determine the silhouette to extrude.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(g == FRAME):
        bbox = _ring_bbox(comp)
        if bbox is not None:
            frames.append(bbox)
    frames.sort()

    for bbox in frames:
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        ptrs = np.argwhere(interior == POINTER)
        assert ptrs.shape[0] == 1, "each frame must contain exactly one pointer"
        pr, pc = map(int, ptrs[0])
        pointer = (r0 + 1 + pr, c0 + 1 + pc)
        direction = _pointer_direction(pointer, bbox)

        color_points = {
            (r0 + 1 + int(rr), c0 + 1 + int(cc))
            for rr, cc in np.argwhere((interior != BG) & (interior != POINTER) & (interior != FRAME))
        }
        assert color_points, "each frame must contain at least one colored pixel"
        colors = {int(g[r, c]) for r, c in color_points}
        assert len(colors) == 1, "interior pattern must use a single color"
        color = next(iter(colors))
        thickness = len(color_points)

        for rr, cc in _project_cells(color_points, bbox, direction, thickness, (h, w)):
            out[rr, cc] = color

    return out


def _choose_pointer(rng: random.Random, ih: int, iw: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return 0, rng.randrange(1, iw - 1)
    if direction == "down":
        return ih - 1, rng.randrange(1, iw - 1)
    if direction == "left":
        return rng.randrange(1, ih - 1), 0
    return rng.randrange(1, ih - 1), iw - 1


def _grow_blob(
    rng: random.Random, cells: list[tuple[int, int]], allowed: set[tuple[int, int]], target: int
) -> set[tuple[int, int]] | None:
    blob = {rng.choice(cells)}
    while len(blob) < target:
        frontier = []
        for r, c in blob:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in allowed and (nr, nc) not in blob:
                    frontier.append((nr, nc))
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            return None
        blob.add(rng.choice(frontier))
    return blob


def _generate_shape(
    rng: random.Random, ih: int, iw: int, pointer: tuple[int, int], count: int
) -> set[tuple[int, int]] | None:
    free = [(r, c) for r in range(ih) for c in range(iw) if (r, c) != pointer]
    allowed = set(free)
    modes = ("blob", "scatter", "hybrid")
    for _ in range(SHAPE_TRIES):
        mode = rng.choice(modes)
        if mode == "scatter":
            shape = set(rng.sample(free, count))
        elif mode == "blob":
            shape = _grow_blob(rng, free, allowed, count)
            if shape is None:
                continue
        else:
            blob_size = rng.randint(1, count)
            blob = _grow_blob(rng, free, allowed, blob_size)
            if blob is None:
                continue
            remaining = [cell for cell in free if cell not in blob]
            extras = set(rng.sample(remaining, count - blob_size)) if count > blob_size else set()
            shape = set(blob) | extras

        if shape:
            return shape
    return None


def _bbox_cells(fr: int, fc: int, fh: int, fw: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(fr, fr + fh) for c in range(fc, fc + fw)}


def _shadow_from_local(
    fr: int,
    fc: int,
    ih: int,
    iw: int,
    direction: str,
    shape: set[tuple[int, int]],
    thickness: int,
    grid_shape: tuple[int, int],
) -> set[tuple[int, int]]:
    bbox = (fr, fc, fr + ih + 1, fc + iw + 1)
    color_points = {(fr + 1 + r, fc + 1 + c) for r, c in shape}
    return _project_cells(color_points, bbox, direction, thickness, grid_shape)


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _placement_bounds(
    grid_h: int, grid_w: int, fh: int, fw: int, thickness: int, direction: str, clip: bool
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if direction == "up":
        row_bounds = (1, min(thickness - 1, grid_h - fh)) if clip else (thickness, grid_h - fh)
        col_bounds = (0, grid_w - fw)
    elif direction == "down":
        row_bounds = (max(0, grid_h - fh - thickness + 1), grid_h - fh - 1) if clip else (0, grid_h - fh - thickness)
        col_bounds = (0, grid_w - fw)
    elif direction == "left":
        row_bounds = (0, grid_h - fh)
        col_bounds = (1, min(thickness - 1, grid_w - fw)) if clip else (thickness, grid_w - fw)
    else:
        row_bounds = (0, grid_h - fh)
        col_bounds = (max(0, grid_w - fw - thickness + 1), grid_w - fw - 1) if clip else (0, grid_w - fw - thickness)

    if row_bounds[0] > row_bounds[1] or col_bounds[0] > col_bounds[1]:
        return None
    return row_bounds, col_bounds


def _render_input(grid: np.ndarray, obj: dict) -> None:
    fr, fc = obj["frame_top"], obj["frame_left"]
    ih, iw = obj["inner_h"], obj["inner_w"]
    color = obj["color"]
    pr, pc = obj["pointer"]

    grid[fr, fc : fc + iw + 2] = FRAME
    grid[fr + ih + 1, fc : fc + iw + 2] = FRAME
    grid[fr : fr + ih + 2, fc] = FRAME
    grid[fr : fr + ih + 2, fc + iw + 1] = FRAME
    grid[fr + 1 + pr, fc + 1 + pc] = POINTER
    for r, c in obj["shape"]:
        grid[fr + 1 + r, fc + 1 + c] = color


def _render_expected(inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        shadow = _shadow_from_local(
            obj["frame_top"],
            obj["frame_left"],
            obj["inner_h"],
            obj["inner_w"],
            obj["direction"],
            obj["shape"],
            obj["count"],
            inp.shape,
        )
        for r, c in shadow:
            out[r, c] = obj["color"]
    return out


def _place_object(
    rng: random.Random,
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    color: int,
    direction: str,
) -> dict | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_TRIES):
        ih = rng.randint(INTERIOR_HEIGHT_RANGE[0], INTERIOR_HEIGHT_RANGE[1])
        iw = rng.randint(INTERIOR_WIDTH_RANGE[0], INTERIOR_WIDTH_RANGE[1])
        fh, fw = ih + 2, iw + 2
        max_count = min(MAX_THICKNESS, ih * iw - 1)
        if max_count < 1:
            continue
        count = rng.randint(1, max_count)
        pointer = _choose_pointer(rng, ih, iw, direction)
        shape = _generate_shape(rng, ih, iw, pointer, count)
        if shape is None:
            continue

        clip = rng.random() < CLIP_PROB
        bounds = _placement_bounds(h, w, fh, fw, count, direction, clip)
        if bounds is None:
            bounds = _placement_bounds(h, w, fh, fw, count, direction, False)
        if bounds is None:
            continue
        (r_lo, r_hi), (c_lo, c_hi) = bounds
        fr = rng.randint(r_lo, r_hi)
        fc = rng.randint(c_lo, c_hi)

        frame_cells = _bbox_cells(fr, fc, fh, fw)
        shadow = _shadow_from_local(fr, fc, ih, iw, direction, shape, count, grid.shape)
        if not shadow:
            continue
        footprint = frame_cells | shadow
        halo = _halo(footprint, h, w)
        if not reserved.isdisjoint(halo):
            continue

        obj = {
            "frame_top": fr,
            "frame_left": fc,
            "inner_h": ih,
            "inner_w": iw,
            "color": color,
            "direction": direction,
            "pointer": pointer,
            "shape": shape,
            "count": count,
        }
        _render_input(grid, obj)
        reserved.update(halo)
        return obj
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.choices(OBJECT_COUNT_VALUES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    min_side = max(14, 10 + 3 * object_count)
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)

    grid = np.full((h, w), BG, dtype=int)
    reserved: set[tuple[int, int]] = set()
    directions = rng.sample(DIRECTIONS, object_count)
    colors = rng.sample(PALETTE, object_count)
    objects: list[dict] = []

    for color, direction in zip(colors, directions):
        obj = _place_object(rng, grid, reserved, color, direction)
        if obj is None:
            return None
        objects.append(obj)

    expected = _render_expected(grid, objects)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
