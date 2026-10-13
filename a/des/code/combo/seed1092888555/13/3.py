from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 8
NON_SQUARE_AREAS = (2, 3, 5, 6, 7, 8)
SQUARE_AREAS = (1, 4, 9)
PALETTE = tuple(c for c in range(10) if c != BG)
GEN_ATTEMPTS = 400


def _components_by_color(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, np.asarray(pts, dtype=int)))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redraw each 4-connected object according to its area:
    perfect-square areas become centered solid squares, otherwise horizontal lines.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full(g.shape, BG, dtype=int)

    for color, pts in _components_by_color(g, BG):
        area = int(pts.shape[0])
        rows = pts[:, 0]
        cols = pts[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1

        side = int(math.isqrt(area))
        if side * side == area:
            start_r = r0 + (box_h - side) // 2
            start_c = c0 + (box_w - side) // 2
            out[start_r : start_r + side, start_c : start_c + side] = color
        else:
            out[r0, c0 : c0 + area] = color

    return out


def _grow_connected_shape(area: int, box_h: int, box_w: int) -> np.ndarray | None:
    for _ in range(250):
        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}

        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if min(rs) != 0 or max(rs) != box_h - 1 or min(cs) != 0 or max(cs) != box_w - 1:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if area < box_h * box_w and random.random() < 0.75:
            if mask.all():
                continue

        return mask
    return None


def _sample_bbox(area: int, square_branch: bool) -> tuple[int, int] | None:
    side = int(math.isqrt(area))
    candidates: list[tuple[int, int]] = []
    for box_h in range(1, 6):
        for box_w in range(1, 8):
            capacity = box_h * box_w
            if capacity < area or capacity > area + 4:
                continue
            if square_branch and (box_h < side or box_w < side):
                continue
            candidates.append((box_h, box_w))

    random.shuffle(candidates)
    for box_h, box_w in candidates:
        if _grow_connected_shape(area, box_h, box_w) is not None:
            return box_h, box_w
    return None


def _make_object_spec(area: int, color: int) -> dict | None:
    square_branch = area in SQUARE_AREAS
    dims = _sample_bbox(area, square_branch)
    if dims is None:
        return None
    box_h, box_w = dims
    mask = _grow_connected_shape(area, box_h, box_w)
    if mask is None:
        return None

    side = int(math.isqrt(area))
    reserve_h = box_h
    reserve_w = box_w if square_branch else area
    return {
        "area": area,
        "color": color,
        "box_h": box_h,
        "box_w": box_w,
        "mask": mask,
        "square_branch": square_branch,
        "side": side,
        "reserve_h": reserve_h,
        "reserve_w": reserve_w,
    }


def _place_specs(grid_h: int, grid_w: int, specs: list[dict]) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    order = sorted(
        range(len(specs)),
        key=lambda i: (-specs[i]["reserve_w"] * specs[i]["reserve_h"], -specs[i]["area"]),
    )

    for idx in order:
        spec = specs[idx]
        candidates: list[tuple[int, int]] = []
        need_h = spec["reserve_h"]
        need_w = spec["reserve_w"]
        for r0 in range(grid_h - need_h + 1):
            for c0 in range(grid_w - need_w + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(grid_h, r0 + need_h + 1)
                cc1 = min(grid_w, c0 + need_w + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        chosen = random.choice(candidates)
        placements[idx] = chosen
        r0, c0 = chosen
        occupied[r0 : r0 + need_h, c0 : c0 + need_w] = True

    if any(p is None for p in placements):
        return None
    return [(int(p[0]), int(p[1])) for p in placements]


def _build_expected(grid_h: int, grid_w: int, specs: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    out = np.full((grid_h, grid_w), BG, dtype=int)
    for spec, (r0, c0) in zip(specs, placements):
        color = spec["color"]
        area = spec["area"]
        if spec["square_branch"]:
            start_r = r0 + (spec["box_h"] - spec["side"]) // 2
            start_c = c0 + (spec["box_w"] - spec["side"]) // 2
            out[start_r : start_r + spec["side"], start_c : start_c + spec["side"]] = color
        else:
            out[r0, c0 : c0 + area] = color
    return out


def _build_input(grid_h: int, grid_w: int, specs: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec, (r0, c0) in zip(specs, placements):
        grid[r0 : r0 + spec["box_h"], c0 : c0 + spec["box_w"]][spec["mask"]] = spec["color"]
    return grid


def _sample_areas(n_objects: int) -> list[int]:
    n_square = random.randint(1, n_objects - 1)
    n_non_square = n_objects - n_square
    square_areas = [random.choice(SQUARE_AREAS) for _ in range(n_square)]
    nonsquare_areas = [random.choice(NON_SQUARE_AREAS) for _ in range(n_non_square)]
    areas = square_areas + nonsquare_areas
    random.shuffle(areas)
    return areas


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(12, 20)
    grid_w = random.randint(14, 24)
    n_objects = random.randint(2, 5)
    colors = random.sample(PALETTE, n_objects)
    areas = _sample_areas(n_objects)

    specs: list[dict] = []
    for area, color in zip(areas, colors):
        spec = _make_object_spec(area, color)
        if spec is None:
            return None
        specs.append(spec)

    placements = _place_specs(grid_h, grid_w, specs)
    if placements is None:
        return None

    inp = _build_input(grid_h, grid_w, specs, placements)
    expected = _build_expected(grid_h, grid_w, specs, placements)

    comps = _components_by_color(inp, BG)
    if len(comps) != n_objects:
        return None
    observed = sorted((int(color), int(len(pts))) for color, pts in comps)
    target = sorted((int(spec["color"]), int(spec["area"])) for spec in specs)
    if observed != target:
        return None
    if not any(spec["square_branch"] for spec in specs):
        return None
    if not any(not spec["square_branch"] for spec in specs):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")
