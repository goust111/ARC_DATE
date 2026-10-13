import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

RECTANGLE_COLORS = (BLUE, RED, GREEN, ORANGE)
CORNERS = ("tl", "tr", "bl", "br")
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (16, 22)
GRID_W_RANGE = (18, 24)
RECT_COUNT_OPTIONS = (3, 4)
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 6)
DOT_COUNT_RANGE = (2, 6)

RECT_TRIES = 200
GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _corner_from_local(local_r: int, local_c: int, h: int, w: int) -> tuple[str, tuple[int, int]]:
    if (local_r, local_c) == (0, 0):
        return "tl", (-1, -1)
    if (local_r, local_c) == (0, w - 1):
        return "tr", (-1, 1)
    if (local_r, local_c) == (h - 1, 0):
        return "bl", (1, -1)
    if (local_r, local_c) == (h - 1, w - 1):
        return "br", (1, 1)
    raise AssertionError("grey control pixel must be at a rectangle corner")


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]], Counter]:
    g = np.asarray(grid, dtype=int)
    rectangles = []
    dot_cells: list[tuple[int, int]] = []
    dot_counts: Counter = Counter()

    for coords in _components(g != BLACK):
        if coords.shape[0] == 1:
            r, c = map(int, coords[0])
            color = int(g[r, c])
            assert color != GREY, "grey must only appear inside rectangles"
            dot_cells.append((r, c))
            dot_counts[color] += 1
            continue

        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        assert coords.shape[0] == h * w, "rectangle component must fill its bounding box"
        assert not np.any(patch == BLACK), "rectangle bbox cannot contain background holes"

        values = {int(v) for v in np.unique(patch)}
        assert GREY in values and len(values) == 2, "rectangle must contain one grey corner and one fill color"
        color = next(v for v in values if v != GREY)

        grey_locs = np.argwhere(patch == GREY)
        assert grey_locs.shape[0] == 1, "rectangle must have exactly one grey corner"
        local_r, local_c = map(int, grey_locs[0])
        corner, direction = _corner_from_local(local_r, local_c, h, w)

        rectangles.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "grey": (r0 + local_r, c0 + local_c),
                "corner": corner,
                "dir": direction,
            }
        )

    rect_colors = [rect["color"] for rect in rectangles]
    assert len(rect_colors) == len(set(rect_colors)), "rectangle colors must be unique"
    assert rectangles, "expected at least one rectangle"
    for color in dot_counts:
        assert color in rect_colors, "dot colors must correspond to rectangle colors"
    return rectangles, dot_cells, dot_counts


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rectangles, dot_cells, dot_counts = _parse_scene(g)

    out = g.copy()
    if dot_cells:
        rows = [r for r, _c in dot_cells]
        cols = [c for _r, c in dot_cells]
        out[rows, cols] = BLACK

    for rect in rectangles:
        color = int(rect["color"])
        grey_r, grey_c = rect["grey"]
        dr, dc = rect["dir"]
        out[grey_r, grey_c] = color

        for step in range(1, int(dot_counts[color]) + 1):
            rr = grey_r + dr * step
            cc = grey_c + dc * step
            assert 0 <= rr < h and 0 <= cc < w, "ray must stay inside the grid"
            assert int(g[rr, cc]) == BLACK, "ray path must be empty in the input"
            out[rr, cc] = GREY

    return out


def _make_rect_cells(r0: int, c0: int, hh: int, ww: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)}


def _expand(points: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in points:
        for nr in range(max(0, r - radius), min(h, r + radius + 1)):
            for nc in range(max(0, c - radius), min(w, c + radius + 1)):
                expanded.add((nr, nc))
    return expanded


def _placement_bounds(h: int, w: int, hh: int, ww: int, corner: str, ray_len: int) -> tuple[int, int, int, int] | None:
    if corner == "tl":
        bounds = (ray_len, h - hh, ray_len, w - ww)
    elif corner == "tr":
        bounds = (ray_len, h - hh, 0, w - ww - ray_len)
    elif corner == "bl":
        bounds = (0, h - hh - ray_len, ray_len, w - ww)
    elif corner == "br":
        bounds = (0, h - hh - ray_len, 0, w - ww - ray_len)
    else:
        raise AssertionError("unknown corner")

    r_lo, r_hi, c_lo, c_hi = bounds
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return bounds


def _grey_and_dir(r0: int, c0: int, hh: int, ww: int, corner: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner == "tl":
        return (r0, c0), (-1, -1)
    if corner == "tr":
        return (r0, c0 + ww - 1), (-1, 1)
    if corner == "bl":
        return (r0 + hh - 1, c0), (1, -1)
    if corner == "br":
        return (r0 + hh - 1, c0 + ww - 1), (1, 1)
    raise AssertionError("unknown corner")


def _render_expected(h: int, w: int, rectangles: list[dict]) -> np.ndarray:
    out = np.full((h, w), BLACK, dtype=int)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        color = int(rect["color"])
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        for rr, cc in rect["ray_cells"]:
            out[rr, cc] = GREY
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_rects = random.choice(RECT_COUNT_OPTIONS)

    colors = random.sample(RECTANGLE_COLORS, n_rects)
    corners = random.sample(CORNERS, n_rects)
    dot_counts = {color: random.randint(*DOT_COUNT_RANGE) for color in colors}

    jobs = list(zip(colors, corners))
    random.shuffle(jobs)
    jobs.sort(key=lambda item: dot_counts[item[0]], reverse=True)

    rectangles: list[dict] = []
    rect_cells_all: set[tuple[int, int]] = set()
    rect_halo_all: set[tuple[int, int]] = set()
    ray_cells_all: set[tuple[int, int]] = set()
    ray_halo_all: set[tuple[int, int]] = set()

    for color, corner in jobs:
        placed = False
        for _ in range(RECT_TRIES):
            hh = random.randint(*RECT_H_RANGE)
            ww = random.randint(*RECT_W_RANGE)
            bounds = _placement_bounds(h, w, hh, ww, corner, dot_counts[color])
            if bounds is None:
                continue
            r_lo, r_hi, c_lo, c_hi = bounds
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            grey, direction = _grey_and_dir(r0, c0, hh, ww, corner)
            ray_cells = {
                (grey[0] + direction[0] * step, grey[1] + direction[1] * step)
                for step in range(1, dot_counts[color] + 1)
            }
            rect_cells = _make_rect_cells(r0, c0, hh, ww)
            rect_halo = _expand(rect_cells, h, w, radius=1)
            cand_ray_halo = _expand(ray_cells, h, w, radius=1)

            if rect_halo & rect_cells_all:
                continue
            if rect_halo & ray_cells_all:
                continue
            if ray_cells & rect_halo_all:
                continue
            if ray_cells & ray_halo_all:
                continue

            rectangles.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r0 + hh - 1, c0 + ww - 1),
                    "grey": grey,
                    "corner": corner,
                    "dir": direction,
                    "ray_cells": tuple(sorted(ray_cells)),
                }
            )
            rect_cells_all |= rect_cells
            rect_halo_all |= rect_halo
            ray_cells_all |= ray_cells
            ray_halo_all |= cand_ray_halo
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), BLACK, dtype=int)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        color = int(rect["color"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        gr, gc = rect["grey"]
        grid[gr, gc] = GREY

    dot_halo_all: set[tuple[int, int]] = set()
    dot_jobs = list(colors)
    random.shuffle(dot_jobs)
    dot_jobs.sort(key=lambda color: dot_counts[color], reverse=True)
    for color in dot_jobs:
        placed = 0
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if grid[r, c] == BLACK and (r, c) not in ray_cells_all and (r, c) not in rect_halo_all
        ]
        random.shuffle(candidates)
        for r, c in candidates:
            if (r, c) in dot_halo_all:
                continue
            grid[r, c] = int(color)
            dot_halo_all |= _expand({(r, c)}, h, w, radius=1)
            placed += 1
            if placed == dot_counts[color]:
                break
        if placed != dot_counts[color]:
            return None

    expected = _render_expected(h, w, rectangles)
    actual = main(grid)
    assert np.array_equal(actual, expected), "generator must agree with main()"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
