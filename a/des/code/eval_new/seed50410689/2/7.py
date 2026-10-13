from __future__ import annotations

import random
from typing import Any

import numpy as np


BLACK = 0
LIGHT = 9
SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_PRIORITY = {side: i for i, side in enumerate(SIDE_ORDER)}
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _uniform_nonblack(arr: np.ndarray) -> int | None:
    vals = np.unique(np.asarray(arr, dtype=int))
    if vals.size != 1:
        return None
    val = int(vals[0])
    if val == BLACK:
        return None
    return val


def _detect_windows(grid: np.ndarray) -> list[dict[str, Any]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    windows: list[dict[str, Any]] = []
    for comp in _components(g == BLACK):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        ir0, ir1 = min(rs), max(rs)
        ic0, ic1 = min(cs), max(cs)
        ih = ir1 - ir0 + 1
        iw = ic1 - ic0 + 1
        if len(comp) != ih * iw:
            continue
        if ir0 == 0 or ic0 == 0 or ir1 == h - 1 or ic1 == w - 1:
            continue

        top = _uniform_nonblack(g[ir0 - 1, ic0 : ic1 + 1])
        bottom = _uniform_nonblack(g[ir1 + 1, ic0 : ic1 + 1])
        left = _uniform_nonblack(g[ir0 : ir1 + 1, ic0 - 1])
        right = _uniform_nonblack(g[ir0 : ir1 + 1, ic1 + 1])
        if None in (top, bottom, left, right):
            continue
        sides = {"top": top, "bottom": bottom, "left": left, "right": right}
        if len(set(sides.values())) != 4:
            continue

        corners = {
            (ir0 - 1, ic0 - 1): {top, left},
            (ir0 - 1, ic1 + 1): {top, right},
            (ir1 + 1, ic0 - 1): {bottom, left},
            (ir1 + 1, ic1 + 1): {bottom, right},
        }
        if any(int(g[r, c]) not in allowed for (r, c), allowed in corners.items()):
            continue

        windows.append(
            {
                "outer_bbox": (ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1),
                "inner_bbox": (ir0, ic0, ir1, ic1),
                "inner_shape": (ih, iw),
                "sides": sides,
            }
        )

    windows.sort(key=lambda window: window["outer_bbox"])
    return windows


def _path_clear(cells: np.ndarray) -> bool:
    if cells.size == 0:
        return True
    return bool(np.all((cells == BLACK) | (cells == LIGHT)))


def _map_sources(
    grid: np.ndarray, windows: list[dict[str, Any]]
) -> list[tuple[tuple[int, int], int, str, bool]]:
    g = np.asarray(grid, dtype=int)
    mappings: list[tuple[tuple[int, int], int, str, bool]] = []
    for r, c in map(tuple, np.argwhere(g == LIGHT)):
        candidates: list[tuple[int, int, str]] = []
        for win_idx, window in enumerate(windows):
            r0, c0, r1, c1 = window["outer_bbox"]
            ir0, ic0, ir1, ic1 = window["inner_bbox"]

            if ic0 <= c <= ic1 and r < r0 and _path_clear(g[r + 1 : r0, c]):
                candidates.append((r0 - r, win_idx, "top"))
            if ic0 <= c <= ic1 and r > r1 and _path_clear(g[r1 + 1 : r, c]):
                candidates.append((r - r1, win_idx, "bottom"))
            if ir0 <= r <= ir1 and c < c0 and _path_clear(g[r, c + 1 : c0]):
                candidates.append((c0 - c, win_idx, "left"))
            if ir0 <= r <= ir1 and c > c1 and _path_clear(g[r, c1 + 1 : c]):
                candidates.append((c - c1, win_idx, "right"))

        if not candidates:
            continue

        candidates.sort(key=lambda item: (item[0], item[1], SIDE_PRIORITY[item[2]]))
        best_dist = candidates[0][0]
        best = [cand for cand in candidates if cand[0] == best_dist]
        mappings.append(((int(r), int(c)), best[0][1], best[0][2], len(best) == 1))
    return mappings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor window interiors with rays fired by aligned outside light sources.

    Input structure:
    - Background is black.
    - The grid contains one or more rectangular "windows": black rectangular interiors surrounded by a 1-cell border whose top, right, bottom, and left sides have four different colors.
    - Single maroon light-source pixels sit outside the windows, aligned with a window interior row or interior column.

    Rule:
    - For each light source, find the nearest aligned window reached along a cardinal direction through empty background.
    - The ray enters through one side of that window; every interior cell on the crossed row or crossed column is recolored to that entry side's border color.
    - If several rays hit the same interior cell, keep the color from the side with clockwise priority top > right > bottom > left.
    - Light sources stay in place, window borders stay unchanged, and untouched interior cells remain black.

    Key insight:
    - A source does not carry its own color into the window; it only selects a direction, and the window's entry-side border color determines the repainting, with a fixed side-priority rule resolving row/column crossings.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    windows = _detect_windows(g)

    per_window: list[tuple[np.ndarray, np.ndarray]] = []
    for window in windows:
        ih, iw = window["inner_shape"]
        per_window.append((np.full((ih, iw), 99, dtype=int), np.full((ih, iw), BLACK, dtype=int)))

    for (r, c), win_idx, side, _unique in _map_sources(g, windows):
        window = windows[win_idx]
        ir0, ic0, ir1, ic1 = window["inner_bbox"]
        priority_grid, color_grid = per_window[win_idx]
        color = int(window["sides"][side])
        priority = SIDE_PRIORITY[side]

        if side in ("top", "bottom"):
            col = c - ic0
            for rr in range(ir1 - ir0 + 1):
                if priority < priority_grid[rr, col]:
                    priority_grid[rr, col] = priority
                    color_grid[rr, col] = color
        else:
            row = r - ir0
            for cc in range(ic1 - ic0 + 1):
                if priority < priority_grid[row, cc]:
                    priority_grid[row, cc] = priority
                    color_grid[row, cc] = color

    for window, (_priority_grid, color_grid) in zip(windows, per_window):
        ir0, ic0, ir1, ic1 = window["inner_bbox"]
        for rr in range(ir1 - ir0 + 1):
            for cc in range(ic1 - ic0 + 1):
                color = int(color_grid[rr, cc])
                if color != BLACK:
                    out[ir0 + rr, ic0 + cc] = color

    return out


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int, gap: int = 0) -> bool:
    return not (a1 + gap < b0 or b1 + gap < a0)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _paint_window(
    grid: np.ndarray,
    rng: random.Random,
    r0: int,
    c0: int,
    ih: int,
    iw: int,
    side_colors: dict[str, int],
) -> None:
    grid[r0, c0 + 1 : c0 + iw + 1] = side_colors["top"]
    grid[r0 + ih + 1, c0 + 1 : c0 + iw + 1] = side_colors["bottom"]
    grid[r0 + 1 : r0 + ih + 1, c0] = side_colors["left"]
    grid[r0 + 1 : r0 + ih + 1, c0 + iw + 1] = side_colors["right"]

    corner_options = [
        ("top", "left"),
        ("top", "right"),
        ("bottom", "left"),
        ("bottom", "right"),
    ]
    corner_cells = [
        (r0, c0),
        (r0, c0 + iw + 1),
        (r0 + ih + 1, c0),
        (r0 + ih + 1, c0 + iw + 1),
    ]
    for (a, b), (rr, cc) in zip(corner_options, corner_cells):
        grid[rr, cc] = rng.choice([side_colors[a], side_colors[b]])


def _choose_side_set(rng: random.Random, require_cross: bool) -> set[str]:
    if require_cross:
        pattern = rng.choice(["orthogonal", "triple", "quad"])
    else:
        pattern = rng.choices(
            ["single", "orthogonal", "opposite", "triple", "quad"],
            weights=[1, 3, 1, 2, 1],
            k=1,
        )[0]

    if pattern == "single":
        return {rng.choice(list(SIDE_ORDER))}
    if pattern == "orthogonal":
        return {rng.choice(["top", "bottom"]), rng.choice(["left", "right"])}
    if pattern == "opposite":
        return set(rng.choice([("top", "bottom"), ("left", "right")]))
    if pattern == "triple":
        missing = rng.choice(list(SIDE_ORDER))
        return set(side for side in SIDE_ORDER if side != missing)
    return set(SIDE_ORDER)


def _try_place_sources(
    grid: np.ndarray,
    rng: random.Random,
    window: dict[str, Any],
    side_set: set[str],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = window["outer_bbox"]
    ir0, ic0, ir1, ic1 = window["inner_bbox"]
    positions: list[tuple[int, int]] = []

    vertical_cols: dict[str, int] = {}
    horizontal_rows: dict[str, int] = {}

    if "top" in side_set and "bottom" in side_set and rng.random() < 0.5:
        shared = rng.randint(ic0, ic1)
        vertical_cols["top"] = shared
        vertical_cols["bottom"] = shared
    for side in ("top", "bottom"):
        if side in side_set and side not in vertical_cols:
            vertical_cols[side] = rng.randint(ic0, ic1)

    if "left" in side_set and "right" in side_set and rng.random() < 0.5:
        shared = rng.randint(ir0, ir1)
        horizontal_rows["left"] = shared
        horizontal_rows["right"] = shared
    for side in ("left", "right"):
        if side in side_set and side not in horizontal_rows:
            horizontal_rows[side] = rng.randint(ir0, ir1)

    max_dists = {
        "top": min(4, r0),
        "bottom": min(4, h - 1 - r1),
        "left": min(4, c0),
        "right": min(4, w - 1 - c1),
    }
    for side in side_set:
        if max_dists[side] < 1:
            return []

    side_order = list(side_set)
    rng.shuffle(side_order)
    for side in side_order:
        for _ in range(10):
            dist = rng.randint(1, max_dists[side])
            if side == "top":
                rr, cc = r0 - dist, vertical_cols["top"]
                path = grid[rr + 1 : r0, cc]
            elif side == "bottom":
                rr, cc = r1 + dist, vertical_cols["bottom"]
                path = grid[r1 + 1 : rr, cc]
            elif side == "left":
                rr, cc = horizontal_rows["left"], c0 - dist
                path = grid[rr, cc + 1 : c0]
            else:
                rr, cc = horizontal_rows["right"], c1 + dist
                path = grid[rr, c1 + 1 : cc]

            if grid[rr, cc] != BLACK:
                continue
            if not np.all(path == BLACK):
                continue
            grid[rr, cc] = LIGHT
            positions.append((rr, cc))
            break
        else:
            for rr, cc in positions:
                grid[rr, cc] = BLACK
            return []

    return positions


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_windows = rng.choices([1, 2, 3], weights=[2, 3, 2], k=1)[0]
    min_dim = 10 + 5 * n_windows
    h = rng.randint(min_dim, 30)
    w = rng.randint(min_dim, 30)
    grid = np.zeros((h, w), dtype=int)

    windows: list[dict[str, Any]] = []
    for _ in range(n_windows):
        placed = False
        for _ in range(200):
            ih = rng.randint(2, 5)
            iw = rng.randint(2, 5)
            outer_h = ih + 2
            outer_w = iw + 2
            r0 = rng.randint(2, h - outer_h - 2)
            c0 = rng.randint(2, w - outer_w - 2)

            outer_bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            inner_bbox = (r0 + 1, c0 + 1, r0 + ih, c0 + iw)
            if any(
                _rects_overlap(outer_bbox, other["outer_bbox"], gap=2)
                or _intervals_overlap(inner_bbox[0], inner_bbox[2], other["inner_bbox"][0], other["inner_bbox"][2])
                or _intervals_overlap(inner_bbox[1], inner_bbox[3], other["inner_bbox"][1], other["inner_bbox"][3])
                for other in windows
            ):
                continue

            side_values = rng.sample(COLOR_POOL, 4)
            side_colors = {
                "top": side_values[0],
                "right": side_values[1],
                "bottom": side_values[2],
                "left": side_values[3],
            }
            window = {
                "outer_bbox": outer_bbox,
                "inner_bbox": inner_bbox,
                "inner_shape": (ih, iw),
                "sides": side_colors,
            }
            _paint_window(grid, rng, r0, c0, ih, iw, side_colors)
            windows.append(window)
            placed = True
            break
        if not placed:
            return None

    cross_windows = [i for i in range(n_windows) if rng.random() < 0.7]
    if not cross_windows:
        cross_windows = [rng.randrange(n_windows)]

    all_sources: list[tuple[int, int]] = []
    for idx, window in enumerate(windows):
        side_set = _choose_side_set(rng, idx in cross_windows)
        positions = _try_place_sources(grid, rng, window, side_set)
        if not positions:
            return None
        all_sources.extend(positions)

    detected = _detect_windows(grid)
    if len(detected) != len(windows):
        return None

    mappings = _map_sources(grid, detected)
    if len(mappings) != len(all_sources) or not all(unique for _src, _w, _side, unique in mappings):
        return None

    used_sides = {side for _src, _w, side, _unique in mappings}
    if len(used_sides) < 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
