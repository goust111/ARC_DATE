from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
WALL_PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MIN = 18
GRID_MAX = 28
RING_COUNT_RANGE = (2, 4)
RING_H_RANGE = (5, 8)
RING_W_RANGE = (5, 8)
RUNWAY_EXTRA_RANGE = (1, 4)
OBSTACLE_LEN_RANGE = (2, 5)
MAX_GEN_ATTEMPTS = 400
MAX_RING_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _ring_info(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    colors = {int(grid[r, c]) for r, c in comp}
    if len(colors) != 1:
        return None

    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    comp_set = set(comp)
    border = _perimeter_cells(r0, c0, r1, c1)
    if not comp_set <= border:
        return None

    missing = sorted(border - comp_set)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if grid[r, c] != BG and grid[r, c] != YELLOW:
                return None

    interior_yellows = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) == YELLOW
    ]
    if not interior_yellows:
        return None

    if gap[0] == r0:
        side = "top"
    elif gap[0] == r1:
        side = "bottom"
    elif gap[1] == c0:
        side = "left"
    else:
        side = "right"

    return {
        "bbox": (r0, c0, r1, c1),
        "color": next(iter(colors)),
        "gap": gap,
        "side": side,
        "dir": DIRS[side],
        "yellows": interior_yellows,
    }


def _find_rings(grid: np.ndarray) -> list[dict]:
    wall_mask = (grid != BG) & (grid != YELLOW)
    rings: list[dict] = []
    for comp in _components(wall_mask):
        info = _ring_info(grid, comp)
        if info is not None:
            rings.append(info)
    rings.sort(key=lambda info: (info["bbox"][0], info["bbox"][1], info["gap"]))
    return rings


def _track_cells(wall_mask: np.ndarray, gap: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = wall_mask.shape
    dr, dc = direction
    r, c = gap[0] + dr, gap[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and not wall_mask[r, c]:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject the yellow pixels from each broken wall loop and stack them at the far obstacle.

    Input structure:
    - The background is black.
    - Several non-touching wall loops appear as hollow rectangular frames with exactly one missing wall cell.
    - Each loop contains one or more yellow single pixels, and extra wall pieces outside the loops may act as obstacles.

    Rule:
    - For each wall loop, locate its unique gap and use the side of that gap to determine the outward direction away from the loop interior.
    - Remove every yellow pixel from inside the loop, then send that many yellow pixels straight outward along the gap's row or column.
    - The pixels do not stop at the gap; they travel until the first obstacle or grid edge, then stack backward into a contiguous straight segment whose length equals the original yellow count.
    - Keep all wall cells unchanged.

    Key insight:
    - The gap is both the exit and the direction code: its topology picks the outward ray, and the final yellow segment is anchored by the far obstacle rather than by the gap itself.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    wall_mask = (g != BG) & (g != YELLOW)
    rings = _find_rings(g)

    for info in rings:
        for r, c in info["yellows"]:
            out[r, c] = BG

    for info in rings:
        track = _track_cells(wall_mask, info["gap"], info["dir"])
        count = len(info["yellows"])
        if len(track) < count:
            raise AssertionError("gap runway shorter than yellow count")
        for r, c in track[-count:]:
            out[r, c] = YELLOW

    return out


def _frame_cells(r0: int, c0: int, h: int, w: int, side: str, offset: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    cells = _perimeter_cells(r0, c0, r1, c1)
    if side == "top":
        gap = (r0, c0 + offset)
    elif side == "bottom":
        gap = (r1, c0 + offset)
    elif side == "left":
        gap = (r0 + offset, c0)
    else:
        gap = (r0 + offset, c1)
    cells.remove(gap)
    return cells, gap


def _track_from_layout(
    h: int,
    w: int,
    r0: int,
    c0: int,
    gap: tuple[int, int],
    side: str,
    runway: int,
) -> list[tuple[int, int]]:
    dr, dc = DIRS[side]
    return [(gap[0] + dr * step, gap[1] + dc * step) for step in range(1, runway + 1)]


def _wall_obstacle_cells(
    grid_shape: tuple[int, int],
    gap: tuple[int, int],
    side: str,
    runway: int,
    length: int,
) -> set[tuple[int, int]] | None:
    h, w = grid_shape
    dr, dc = DIRS[side]
    anchor_r = gap[0] + dr * (runway + 1)
    anchor_c = gap[1] + dc * (runway + 1)
    if not (0 <= anchor_r < h and 0 <= anchor_c < w):
        return None

    cells: set[tuple[int, int]] = set()
    if side in ("left", "right"):
        span0 = length // 2
        span1 = length - span0 - 1
        r0 = anchor_r - span0
        r1 = anchor_r + span1
        if r0 < 0 or r1 >= h:
            return None
        for r in range(r0, r1 + 1):
            cells.add((r, anchor_c))
    else:
        span0 = length // 2
        span1 = length - span0 - 1
        c0 = anchor_c - span0
        c1 = anchor_c + span1
        if c0 < 0 or c1 >= w:
            return None
        for c in range(c0, c1 + 1):
            cells.add((anchor_r, c))
    return cells


def _padded(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _choose_sides(count: int) -> list[str]:
    if count == 1:
        return [random.choice(SIDES)]
    sides = [random.choice(("left", "right")), random.choice(("top", "bottom"))]
    remaining = [side for side in SIDES if side not in sides]
    sides.extend(random.sample(remaining, count - 2))
    random.shuffle(sides)
    return sides


def _choose_kinds(count: int) -> list[str]:
    if count == 1:
        return [random.choice(("edge", "wall"))]
    kinds = ["edge", "wall"]
    kinds.extend(random.choice(("edge", "wall")) for _ in range(count - 2))
    random.shuffle(kinds)
    return kinds


def _sample_ring(
    shape: tuple[int, int],
    side: str,
    kind: str,
    reserved: set[tuple[int, int]],
    force_multi: bool,
) -> dict | None:
    h_grid, w_grid = shape
    for _ in range(MAX_RING_ATTEMPTS):
        ring_h = random.randint(*RING_H_RANGE)
        ring_w = random.randint(*RING_W_RANGE)
        offset = random.randint(1, (ring_w - 2) if side in ("top", "bottom") else (ring_h - 2))
        max_inside = min(5, (ring_h - 2) * (ring_w - 2))
        min_inside = 2 if force_multi and max_inside >= 2 else 1
        yellow_count = random.randint(min_inside, max_inside)
        runway = yellow_count + random.randint(*RUNWAY_EXTRA_RANGE)

        if side == "right":
            if kind == "edge":
                c0 = w_grid - ring_w - runway
                c_choices = [c0] if c0 >= 1 else []
            else:
                max_c0 = w_grid - ring_w - runway - 2
                c_choices = [] if max_c0 < 1 else [random.randint(1, max_c0)]
            r_choices = [] if h_grid - ring_h - 1 < 1 else [random.randint(1, h_grid - ring_h - 1)]
        elif side == "left":
            if kind == "edge":
                c0 = runway
                c_choices = [c0] if c0 + ring_w <= w_grid - 2 else []
            else:
                min_c0 = runway + 2
                c_choices = [] if min_c0 > w_grid - ring_w - 1 else [random.randint(min_c0, w_grid - ring_w - 1)]
            r_choices = [] if h_grid - ring_h - 1 < 1 else [random.randint(1, h_grid - ring_h - 1)]
        elif side == "top":
            if kind == "edge":
                r0 = runway
                r_choices = [r0] if r0 + ring_h <= h_grid - 2 else []
            else:
                min_r0 = runway + 2
                r_choices = [] if min_r0 > h_grid - ring_h - 1 else [random.randint(min_r0, h_grid - ring_h - 1)]
            c_choices = [] if w_grid - ring_w - 1 < 1 else [random.randint(1, w_grid - ring_w - 1)]
        else:
            if kind == "edge":
                r0 = h_grid - ring_h - runway
                r_choices = [r0] if r0 >= 1 else []
            else:
                max_r0 = h_grid - ring_h - runway - 2
                r_choices = [] if max_r0 < 1 else [random.randint(1, max_r0)]
            c_choices = [] if w_grid - ring_w - 1 < 1 else [random.randint(1, w_grid - ring_w - 1)]

        if not r_choices or not c_choices:
            continue
        r0 = r_choices[0]
        c0 = c_choices[0]

        wall_cells, gap = _frame_cells(r0, c0, ring_h, ring_w, side, offset)
        track = _track_from_layout(h_grid, w_grid, r0, c0, gap, side, runway)
        if any(not (0 <= r < h_grid and 0 <= c < w_grid) for r, c in track):
            continue

        if kind == "wall":
            obstacle = _wall_obstacle_cells(
                shape,
                gap,
                side,
                runway,
                random.randint(*OBSTACLE_LEN_RANGE),
            )
            if obstacle is None:
                continue
        else:
            obstacle = set()

        full_box = {
            (r, c)
            for r in range(r0, r0 + ring_h)
            for c in range(c0, c0 + ring_w)
        }
        footprint = full_box | set(track) | obstacle
        if footprint & reserved:
            continue
        if _padded(footprint, shape) & reserved:
            continue

        interior = [
            (r, c)
            for r in range(r0 + 1, r0 + ring_h - 1)
            for c in range(c0 + 1, c0 + ring_w - 1)
        ]
        yellows = random.sample(interior, yellow_count)
        stack = track[-yellow_count:]
        if set(stack) & reserved:
            continue

        return {
            "bbox": (r0, c0, r0 + ring_h - 1, c0 + ring_w - 1),
            "side": side,
            "kind": kind,
            "gap": gap,
            "dir": DIRS[side],
            "wall_cells": wall_cells,
            "obstacle_cells": obstacle,
            "track": track,
            "stack": stack,
            "yellows": yellows,
            "color": -1,
            "runway": runway,
            "footprint": footprint,
        }
    return None


def _expected_from_meta(grid: np.ndarray, metas: list[dict]) -> np.ndarray:
    out = grid.copy()
    for meta in metas:
        for r, c in meta["yellows"]:
            out[r, c] = BG
    for meta in metas:
        for r, c in meta["stack"]:
            out[r, c] = YELLOW
    return out


def _validate_generated(grid: np.ndarray, metas: list[dict], expected: np.ndarray) -> None:
    parsed = _find_rings(grid)
    if len(parsed) != len(metas):
        raise AssertionError("ring count mismatch")

    parsed_by_gap = {info["gap"]: info for info in parsed}
    meta_by_gap = {meta["gap"]: meta for meta in metas}
    if set(parsed_by_gap) != set(meta_by_gap):
        raise AssertionError("generated gaps are not uniquely recoverable")

    has_edge = any(meta["kind"] == "edge" for meta in metas)
    has_wall = any(meta["kind"] == "wall" for meta in metas)
    if len(metas) >= 2 and not (has_edge and has_wall):
        raise AssertionError("missing edge/wall branch coverage")
    if not any(len(meta["yellows"]) > 1 for meta in metas):
        raise AssertionError("all rings have only one yellow")

    for gap, meta in meta_by_gap.items():
        info = parsed_by_gap[gap]
        if info["side"] != meta["side"]:
            raise AssertionError("gap side misread")
        if len(info["yellows"]) != len(meta["yellows"]):
            raise AssertionError("yellow count mismatch")
        wall_mask = (grid != BG) & (grid != YELLOW)
        track = _track_cells(wall_mask, gap, info["dir"])
        if track != meta["track"]:
            raise AssertionError("track mismatch")
        if track[-len(meta["yellows"]) :] != meta["stack"]:
            raise AssertionError("far stack mismatch")

    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise AssertionError("solver does not match generated semantics")


def _generate_once() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    ring_count = random.randint(*RING_COUNT_RANGE)
    sides = _choose_sides(ring_count)
    kinds = _choose_kinds(ring_count)

    metas: list[dict] = []
    reserved: set[tuple[int, int]] = set()
    need_multi = True

    for side, kind in zip(sides, kinds):
        meta = _sample_ring((h, w), side, kind, reserved, need_multi)
        if meta is None:
            return None
        metas.append(meta)
        reserved |= _padded(meta["footprint"], (h, w))
        if len(meta["yellows"]) > 1:
            need_multi = False

    colors = random.sample(WALL_PALETTE, len(metas) + sum(meta["kind"] == "wall" for meta in metas))
    color_idx = 0
    for meta in metas:
        meta["color"] = colors[color_idx]
        color_idx += 1
        if meta["kind"] == "wall":
            meta["obstacle_color"] = colors[color_idx]
            color_idx += 1
        else:
            meta["obstacle_color"] = None

    grid = np.full((h, w), BG, dtype=int)
    for meta in metas:
        for r, c in meta["wall_cells"]:
            grid[r, c] = meta["color"]
        for r, c in meta["yellows"]:
            grid[r, c] = YELLOW
        for r, c in meta["obstacle_cells"]:
            grid[r, c] = meta["obstacle_color"]

    expected = _expected_from_meta(grid, metas)
    _validate_generated(grid, metas, expected)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid broken-loop sample")
