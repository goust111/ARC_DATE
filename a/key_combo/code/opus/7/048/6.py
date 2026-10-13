from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5


ARROW_SHAPES = {
    "tl": {
        "cells": ((0, 1), (1, 0), (1, 1)),
        "direction": (-1, -1),
        "start_offset": (-1, -1),
        "anchor_offset": (1, 1),
    },
    "tr": {
        "cells": ((0, 0), (1, 0), (1, 1)),
        "direction": (-1, 1),
        "start_offset": (-1, 2),
        "anchor_offset": (1, 0),
    },
    "bl": {
        "cells": ((0, 0), (0, 1), (1, 1)),
        "direction": (1, -1),
        "start_offset": (2, -1),
        "anchor_offset": (0, 1),
    },
    "br": {
        "cells": ((0, 0), (0, 1), (1, 0)),
        "direction": (1, 1),
        "start_offset": (2, 2),
        "anchor_offset": (0, 0),
    },
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _infer_arrow(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    assert len(coords) == 3, "each arrow must contain exactly three blue pixels"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 == r0 + 1 and c1 == c0 + 1, "each arrow must be a 2x2 square missing one corner"
    local = {(r - r0, c - c0) for r, c in coords}
    for missing, spec in ARROW_SHAPES.items():
        if set(spec["cells"]) == local:
            dr, dc = spec["direction"]
            sr, sc = spec["start_offset"]
            ar, ac = spec["anchor_offset"]
            return {
                "bbox": (r0, c0, r1, c1),
                "missing": missing,
                "direction": (dr, dc),
                "start": (r0 + sr, c0 + sc),
                "anchor": (r0 + ar, c0 + ac),
                "coords": tuple(sorted(coords)),
            }
    raise AssertionError("invalid arrow shape")


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows = [_infer_arrow(grid, comp) for comp in _components(grid == BLUE)]
    arrows.sort(key=lambda arrow: (arrow["bbox"][0], arrow["bbox"][1]))
    return arrows


def _trace_rays(grid: np.ndarray, arrows: list[dict]) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = grid.shape
    out = grid.copy()
    paths: list[list[tuple[int, int]]] = []
    for arrow in arrows:
        dr, dc = arrow["direction"]
        r, c = arrow["start"]
        path: list[tuple[int, int]] = []
        while 0 <= r < h and 0 <= c < w:
            cell = int(out[r, c])
            if cell == GREEN:
                path.append((r, c))
                break
            if cell not in (BLACK, GREY):
                break
            out[r, c] = GREEN
            path.append((r, c))
            r += dr
            c += dc
        paths.append(path)
    return out, paths


def _aligned_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc == 0:
        return [a]
    if dr == 0 or dc == 0 or abs(dr) == abs(dc):
        sr = 0 if dr == 0 else dr // abs(dr)
        sc = 0 if dc == 0 else dc // abs(dc)
        steps = max(abs(dr), abs(dc))
        return [(r0 + i * sr, c0 + i * sc) for i in range(steps + 1)]
    return None


def _bresenham_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    points: list[tuple[int, int]] = []
    x0, y0 = c0, r0
    x1, y1 = c1, r1
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    segment = _aligned_segment(a, b)
    if segment is not None:
        return segment
    return _bresenham_segment(a, b)


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    arrows = _find_arrows(grid)
    ray_grid, ray_paths = _trace_rays(grid, arrows)
    ray_sets = [set(path) for path in ray_paths]

    out = ray_grid.copy()
    connected_pairs: list[tuple[int, int]] = []
    line_paths: list[list[tuple[int, int]]] = []
    for i in range(len(arrows)):
        for j in range(i + 1, len(arrows)):
            if ray_sets[i] & ray_sets[j]:
                connected_pairs.append((i, j))
                cells = _segment_cells(arrows[i]["anchor"], arrows[j]["anchor"])
                line_paths.append(cells)
                for r, c in cells:
                    base = int(ray_grid[r, c])
                    if base == BLUE:
                        continue
                    if base == GREEN:
                        out[r, c] = YELLOW
                    elif base in (BLACK, GREY):
                        out[r, c] = RED

    meta = {
        "arrows": arrows,
        "ray_grid": ray_grid,
        "ray_paths": ray_paths,
        "connected_pairs": connected_pairs,
        "line_paths": line_paths,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend arrow rays, detect ray intersections, and connect intersecting arrows with recolored straight segments.

    Input structure:
    - The grid is square with black background, several blue 3-pixel L-shaped arrows, and some isolated grey noise pixels.
    - Each arrow occupies a 2x2 box with one missing corner; that missing corner indicates one diagonal direction.
    - The arrows are separated from each other so their identity is unambiguous before any drawing happens.

    Rule:
    - Sort the arrows by the top-left corner of their 2x2 bounding boxes.
    - For each arrow in that order, start one step outside the missing corner and extend a green diagonal ray in the missing-corner direction.
    - Rays paint black and grey cells green, stop before blue cells, and also stop when they first reach an existing green ray cell.
    - After all rays are drawn, every pair of arrows whose ray paths share at least one pixel is connected by a straight segment between the arrows' anchor pixels.
    - Along each segment, blue arrow pixels stay blue, former green ray pixels become yellow, and black or grey cells become red; later segments overwrite earlier ones using the ray-only colors as the reference.

    Key insight:
    - The missing corner determines both the ray direction and where the ray begins, while ray intersections are the hidden trigger that decides which arrow pairs should be connected and which segment pixels become red versus yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square grid"
    out, _meta = _solve_with_meta(g)
    return out


BASE_SIZE = 15
BASE_ARROWS = (
    (1, 1, "br"),
    (6, 6, "tl"),
    (1, 10, "bl"),
)
OPTIONAL_ARROW = (10, 2, "bl")
REQUIRED_GREYS = ((1, 6),)
FINAL_SIZE_RANGE = (15, 22)
EXTRA_GREY_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 400


def _paint_arrow(grid: np.ndarray, top: int, left: int, missing: str) -> None:
    for dr, dc in ARROW_SHAPES[missing]["cells"]:
        grid[top + dr, left + dc] = BLUE


def _base_scene(include_extra: bool) -> np.ndarray:
    grid = np.zeros((BASE_SIZE, BASE_SIZE), dtype=int)
    for top, left, missing in BASE_ARROWS:
        _paint_arrow(grid, top, left, missing)
    if include_extra:
        _paint_arrow(grid, *OPTIONAL_ARROW)
    for r, c in REQUIRED_GREYS:
        grid[r, c] = GREY
    return grid


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _embed_square(grid: np.ndarray, size: int) -> np.ndarray:
    base = np.zeros((size, size), dtype=int)
    margin = size - grid.shape[0]
    top = random.randint(0, margin)
    left = random.randint(0, margin)
    base[top : top + grid.shape[0], left : left + grid.shape[1]] = grid
    return base


def _is_isolated_noise(grid: np.ndarray, r: int, c: int) -> bool:
    if grid[r, c] != BLACK:
        return False
    h, w = grid.shape
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if (nr, nc) == (r, c):
                continue
            if grid[nr, nc] in (BLUE, GREY):
                return False
    return True


def _add_extra_noise(grid: np.ndarray) -> None:
    candidates = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if _is_isolated_noise(grid, r, c)]
    random.shuffle(candidates)
    limit = random.randint(*EXTRA_GREY_RANGE)
    added = 0
    for r, c in candidates:
        if not _is_isolated_noise(grid, r, c):
            continue
        grid[r, c] = GREY
        added += 1
        if added >= limit:
            break


def _has_nonparticipant(meta: dict) -> bool:
    used = set()
    for i, j in meta["connected_pairs"]:
        used.add(i)
        used.add(j)
    return len(used) < len(meta["arrows"])


def _valid_candidate(grid: np.ndarray, include_extra: bool) -> bool:
    try:
        out, meta = _solve_with_meta(grid)
    except AssertionError:
        return False

    if len(meta["arrows"]) != (4 if include_extra else 3):
        return False
    if len(meta["connected_pairs"]) < 2:
        return False
    if include_extra and not _has_nonparticipant(meta):
        return False
    if not np.any(out == RED):
        return False
    if not np.any(out == YELLOW):
        return False
    if not any(
        int(grid[r, c]) == GREY
        for path in meta["line_paths"]
        for r, c in path
    ):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        include_extra = random.random() < 0.5
        base = _base_scene(include_extra)
        transformed = _random_symmetry(base)
        size = random.randint(*FINAL_SIZE_RANGE)
        if size < transformed.shape[0]:
            continue
        grid = _embed_square(transformed, size)
        _add_extra_noise(grid)
        if _valid_candidate(grid, include_extra):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
