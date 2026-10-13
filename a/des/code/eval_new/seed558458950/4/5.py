import numpy as np
from collections import deque


def _mode_color(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _pair_key(a: int, b: int) -> tuple[int, int]:
    """Canonical unordered pair key for two colors."""
    if a <= b:
        return (a, b)
    return (b, a)


def _run_at_row(grid: np.ndarray, bg: int, r: int, c: int):
    """Return legend row (A,B,C) if this column has pattern A bg B bg C."""
    h, w = grid.shape
    if r < 0 or r >= h or c < 0 or c + 4 >= w:
        return None
    a, b1, b, b2, c3 = grid[r, c : c + 5]
    if b1 == bg and b2 == bg and a != bg and b != bg and c3 != bg:
        return int(a), int(b), int(c3)
    return None


def _find_legend(grid: np.ndarray, bg: int):
    """Detect a corner legend block formed by rows A bg B bg C.

    Returns a mapping ``(A,B)->C`` and a set of legend cells to keep unchanged.
    """
    h, w = grid.shape
    best_start = None
    best_rows = []

    for r in range(h):
        for c in range(w - 4):
            run = _run_at_row(grid, bg, r, c)
            if run is None:
                continue

            rows = []
            rr = r
            while True:
                cur = _run_at_row(grid, bg, rr, c)
                if cur is None:
                    break
                rows.append((rr, cur))
                rr += 1

            if len(rows) < 2:
                continue

            if best_start is None or len(rows) > len(best_rows):
                best_start = (r, c)
                best_rows = rows

    if not best_rows:
        return {}, set()

    legend: dict[tuple[int, int], int] = {}
    legend_cells = set()
    for row, (a, b, c3) in best_rows:
        r0 = best_start[1]
        legend[_pair_key(a, b)] = c3
        legend[ _pair_key(b, a)] = c3
        for dc in range(5):
            legend_cells.add((row, r0 + dc))

    return legend, legend_cells


def _component_center(cells: list[tuple[int, int]]) -> tuple[int, int]:
    """Return the pixel closest to the centroid of an object."""
    arr = np.array(cells, dtype=float)
    centroid_r = arr[:, 0].mean()
    centroid_c = arr[:, 1].mean()
    d2 = (arr[:, 0] - centroid_r) ** 2 + (arr[:, 1] - centroid_c) ** 2
    idx = int(np.argmin(d2))
    return int(arr[idx, 0]), int(arr[idx, 1])


def _shape_type(cells: list[tuple[int, int]]) -> str | None:
    """Classify a small tower as horizontal, vertical, or L-shape."""
    if not cells:
        return None

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    uniq_r = sorted(set(rows))
    uniq_c = sorted(set(cols))

    if len(uniq_r) == 1:
        if len(cells) == len(uniq_c) and len(cells) >= 3 and max(uniq_c) - min(uniq_c) + 1 == len(cells):
            return "horizontal"

    if len(uniq_c) == 1:
        if len(cells) == len(uniq_r) and len(cells) >= 3 and max(uniq_r) - min(uniq_r) + 1 == len(cells):
            return "vertical"

    if len(uniq_r) == 2 and len(uniq_c) == 2 and len(cells) == 3:
        return "l"

    return None


def _collect_towers(grid: np.ndarray, bg: int, forbidden: set[tuple[int, int]]):
    """Collect all small non-bg same-color towers, excluding forbidden legend cells."""
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    groups: dict[str, list[dict]] = {"horizontal": [], "vertical": [], "l": []}
    tower_cells: set[tuple[int, int]] = set()
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden:
                visited[r, c] = True
                continue
            if visited[r, c] or grid[r, c] == bg:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if (nr, nc) in forbidden or visited[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            if not (3 <= len(cells) <= 5):
                continue

            kind = _shape_type(cells)
            if kind is None:
                continue

            tower_cells.update(cells)
            groups[kind].append(
                {
                    "shape": kind,
                    "color": color,
                    "cells": cells,
                    "center": _component_center(cells),
                }
            )

    return groups, tower_cells


def _add_ray_color(cell_map: dict[tuple[int, int], list[int]], cell: tuple[int, int], color: int,
                  forbidden: set[tuple[int, int]]):
    """Append ray color to a cell tracker unless the pixel must stay fixed."""
    if cell in forbidden:
        return
    prev = cell_map.get(cell)
    if prev is None:
        cell_map[cell] = [color]
    elif prev[-1] != color:
        prev.append(color)


def _mix_color(colors: list[int], legend_map: dict[tuple[int, int], int]) -> int:
    """Apply pairwise mixing in order when a pixel is crossed by multiple rays."""
    if not colors:
        raise ValueError("colors must not be empty")
    if len(colors) == 1:
        return colors[0]

    cur = colors[0]
    for nxt in colors[1:]:
        merged = legend_map.get(_pair_key(cur, nxt))
        if merged is not None:
            cur = merged
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """Connect grouped signal towers with two-segment rays and blend intersections via legend rows.

    Input structure:
      - Background color is the dominant color.
      - The grid contains small single-color signal towers shaped as horizontal bars, vertical bars, or 3-cell L-shapes.
      - A corner legend stores rows in the form A, bg, B, bg, C, representing mixing from colors A and B to color C.
      - All other pixels keep their original color unless overwritten by signal rays.

    Rule:
      - Classify non-background pixels (excluding the legend) into connected tower components and group them by shape: horizontal, vertical, and L.
      - For each group, sort tower centers by column then row, and connect each consecutive pair with an L-Path: horizontal run in start color from start center to target column, then vertical run in end color to target center.
      - Keep tower pixels and legend pixels unchanged. When a pixel receives multiple ray colors, resolve the overlap through the legend mapping; if no mapping exists for a pair, retain the current progressive color.
      - The output size equals the input size.

    Key insight:
      - Shape determines the tower groups, not color, so each group follows a separate ordering and connection process, and intersection behavior is controlled by an explicit mixing legend.
    """
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError("input must be a 2D grid")

    out = g.copy()
    bg = _mode_color(g)

    legend_map, legend_cells = _find_legend(g, bg)
    object_forbidden = set(legend_cells)

    groups, tower_cells = _collect_towers(g, bg, object_forbidden)
    object_forbidden.update(tower_cells)

    ray_colors: dict[tuple[int, int], list[int]] = {}

    for kind in ("horizontal", "vertical", "l"):
        objs = sorted(groups[kind], key=lambda item: (item["center"][1], item["center"][0]))
        for cur, nxt in zip(objs, objs[1:]):
            sr, sc = cur["center"]
            er, ec = nxt["center"]

            if sc <= ec:
                cols = range(sc, ec + 1)
            else:
                cols = range(sc, ec - 1, -1)
            for cc in cols:
                _add_ray_color(ray_colors, (sr, cc), cur["color"], object_forbidden)

            if sr <= er:
                rows = range(sr, er + 1)
            else:
                rows = range(sr, er - 1, -1)
            for rr in rows:
                _add_ray_color(ray_colors, (rr, ec), nxt["color"], object_forbidden)

    for (r, c), colors in ray_colors.items():
        out[r, c] = _mix_color(colors, legend_map)

    return out.astype(np.int64)


def generate_input() -> np.ndarray:
    """Generate a random ARC-like input for this puzzle family."""
    rng = np.random.default_rng()
    colors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def can_place(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
        h, w = occupied.shape
        for r, c in cells:
            if r < 0 or r >= h or c < 0 or c >= w:
                return False
            if occupied[r, c]:
                return False
        for r, c in cells:
            for dr, dc in dirs + [(0, 0)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if occupied[rr, cc]:
                        return False
        return True

    def place_shape(shape: str, h: int, w: int, occupied: np.ndarray):
        if shape == "horizontal":
            length = int(rng.choice([3, 5]))
            r_min = 5
            r_max = h - 2
            if r_min > r_max:
                return None
            r = int(rng.integers(r_min, r_max + 1))
            c = int(rng.integers(6, w - length))
            return [(r, c + i) for i in range(length)]

        if shape == "vertical":
            length = int(rng.choice([3, 5]))
            c_min = 6
            c_max = w - 2
            if c_min > c_max:
                return None
            c = int(rng.integers(c_min, c_max + 1))
            r = int(rng.integers(5, h - length + 1))
            return [(r + i, c) for i in range(length)]

        # L-shape (2x2 with one corner missing, size 3)
        offsets = [
            [(0, 0), (1, 0), (1, 1)],
            [(0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 0)],
            [(0, 0), (0, 1), (1, 1)],
        ]
        off = rng.choice(offsets)
        rs = [o[0] for o in off]
        cs = [o[1] for o in off]
        r = int(rng.integers(5, h - max(rs) - 1))
        c = int(rng.integers(6, w - max(cs)))
        return [(r + dr, c + dc) for dr, dc in off]

    for _ in range(400):
        h = int(rng.integers(16, 30))
        w = int(rng.integers(16, 30))
        grid = np.zeros((h, w), dtype=np.int64)
        bg = 0
        grid[:, :] = bg

        legend_count = int(rng.integers(2, min(6, h - 2)))
        if legend_count < 2:
            continue

        obj_color_pool = [int(c) for c in rng.choice(colors, size=rng.integers(3, 6), replace=False)]
        all_pairs = [(a, b) for i, a in enumerate(obj_color_pool) for b in obj_color_pool[i + 1 :]]
        rng.shuffle(all_pairs)

        if not all_pairs:
            continue

        legend_count = min(legend_count, len(all_pairs))
        legend_rows = all_pairs[:legend_count]

        legend_cells = []
        used_mix = set()
        for rr, (a, b) in enumerate(legend_rows):
            choices = [c for c in colors if c not in (a, b)]
            c3 = int(rng.choice(choices))
            while (a, c3) in used_mix and (b, c3) in used_mix and c3 in (a, b):
                c3 = int(rng.choice(choices))
            grid[rr, 0] = a
            grid[rr, 1] = bg
            grid[rr, 2] = b
            grid[rr, 3] = bg
            grid[rr, 4] = c3
            used_mix.add((a, c3))
            used_mix.add((b, c3))
            for c in range(5):
                legend_cells.append((rr, c))

        occupied = np.zeros((h, w), dtype=bool)
        for p in legend_cells:
            occupied[p] = True

        counts = {
            "horizontal": int(rng.choice([0, 0, 1, 2, 3])),
            "vertical": int(rng.choice([0, 0, 1, 2, 3])),
            "l": int(rng.choice([0, 0, 1, 2, 3])),
        }

        if max(counts.values()) < 2:
            shape = rng.choice(["horizontal", "vertical", "l"])
            counts[shape] = 2

        towers_ok = True
        for shape, num in counts.items():
            for _ in range(num):
                placed = False
                color = int(rng.choice(obj_color_pool))
                for _ in range(120):
                    cells = place_shape(shape, h, w, occupied)
                    if cells is None:
                        break
                    if can_place(cells, occupied):
                        for rr, cc in cells:
                            grid[rr, cc] = color
                            occupied[rr, cc] = True
                        placed = True
                        break
                if not placed:
                    towers_ok = False
                    break
            if not towers_ok:
                break

        if not towers_ok:
            continue

        return grid

    raise RuntimeError("unable to generate valid input after retries")
