"""ARC task: compress a notched rectangle array after chasing its pointers."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np


_DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_GEN_TRIES = 500


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    """Return 4-connected components as (n, 2) arrays of (row, column)."""
    mask = grid == int(color)
    seen = np.zeros(grid.shape, dtype=bool)
    result: list[np.ndarray] = []
    h, w = grid.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        todo = [(sr, sc)]
        cells = []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in _DIRS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    todo.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _rectangle_objects(grid: np.ndarray, bg: int) -> list[dict]:
    """Select solid/notched rectangles; tiny foreground components are noise."""
    objects = []
    for color in set(map(int, grid.ravel())) - {int(bg)}:
        for cells in _components_of_color(grid, color):
            r0, c0 = cells.min(axis=0)
            r1, c1 = cells.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            rh, rw = r1 - r0 + 1, c1 - c0 + 1
            if rh < 3 or rw < 3:
                continue
            area = int(cells.shape[0])
            if area not in (rh * rw, rh * rw - 1):
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            wrong = np.argwhere(patch != color)
            direction = None
            if area == rh * rw:
                if wrong.size:
                    continue
            else:
                if wrong.shape[0] != 1:
                    continue
                rr, cc = map(int, wrong[0])
                if int(patch[rr, cc]) != bg:
                    continue
                candidates = []
                if rr == 0 and cc == (rw - 1) // 2 and rw % 2 == 1:
                    candidates.append("U")
                if rr == rh - 1 and cc == (rw - 1) // 2 and rw % 2 == 1:
                    candidates.append("D")
                if cc == 0 and rr == (rh - 1) // 2 and rh % 2 == 1:
                    candidates.append("L")
                if cc == rw - 1 and rr == (rh - 1) // 2 and rh % 2 == 1:
                    candidates.append("R")
                if len(candidates) != 1:
                    continue
                direction = candidates[0]
            objects.append({"bbox": (r0, c0, r1, c1), "color": color, "direction": direction})
    return objects


def _parse_array(grid: np.ndarray) -> tuple[list[list[dict]], int]:
    bg = _background(grid)
    objects = _rectangle_objects(grid, bg)
    assert len(objects) >= 4, "expected a complete array of large rectangles"

    row_bands = sorted({(o["bbox"][0], o["bbox"][2]) for o in objects})
    col_bands = sorted({(o["bbox"][1], o["bbox"][3]) for o in objects})
    assert len(row_bands) >= 2 and len(col_bands) >= 2
    assert len(objects) == len(row_bands) * len(col_bands), "rectangle bands must form a complete array"

    by_bands = {}
    for obj in objects:
        rb = (obj["bbox"][0], obj["bbox"][2])
        cb = (obj["bbox"][1], obj["bbox"][3])
        assert (rb, cb) not in by_bands
        by_bands[rb, cb] = obj
    array = [[by_bands[rb, cb] for cb in col_bands] for rb in row_bands]
    return array, bg


def main(grid: np.ndarray) -> np.ndarray:
    """Compress a notched rectangle array, coloring each cell by the solid rectangle reached by its pointer chain.

    Input structure:
    - The dominant background contains a complete row-band/column-band array of
      monochrome filled rectangles, each at least 3 by 3, plus tiny noise components.
    - A rectangle is either solid or has exactly one background-colored gap at
      the midpoint of one edge; that edge faces an adjacent array rectangle.

    Rule:
    - Ignore the tiny distractors and compress the rectangle array to one output
      cell per row-band and column-band.
    - A solid rectangle contributes its own color.  From a notched rectangle,
      step to the orthogonally adjacent rectangle indicated by its notch and
      keep following new notches until a solid rectangle is reached; use that
      final rectangle's color.

    Key insight:
    - A gap is a directional pointer, not damage to repair: intermediate colors
      and geometric nearness are irrelevant because the pointer chain may turn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    array, _bg = _parse_array(g)
    nr, nc = len(array), len(array[0])
    out = np.empty((nr, nc), dtype=int)

    for sr in range(nr):
        for sc in range(nc):
            r, c = sr, sc
            visited = set()
            while array[r][c]["direction"] is not None:
                assert (r, c) not in visited, "pointer chains must be acyclic"
                visited.add((r, c))
                dr, dc = _DIRS[array[r][c]["direction"]]
                r, c = r + dr, c + dc
                assert 0 <= r < nr and 0 <= c < nc, "notch must point to an adjacent rectangle"
            out[sr, sc] = int(array[r][c]["color"])
    return out


def _neighbors(cell: tuple[int, int], nr: int, nc: int):
    r, c = cell
    for name, (dr, dc) in _DIRS.items():
        rr, cc = r + dr, c + dc
        if 0 <= rr < nr and 0 <= cc < nc:
            yield name, (rr, cc)


def _random_forest(nr: int, nc: int):
    """Grow two rooted forests outward, recording child-to-parent pointers."""
    cells = [(r, c) for r in range(nr) for c in range(nc)]
    roots = random.sample(cells, 2)
    parent = {root: None for root in roots}
    assigned = set(roots)
    while len(assigned) < len(cells):
        frontier = []
        for par in assigned:
            for _name, child in _neighbors(par, nr, nc):
                if child not in assigned:
                    frontier.append((child, par))
        if not frontier:
            return None
        child, par = random.choice(frontier)
        parent[child] = par
        assigned.add(child)
    return parent, roots


def _path(cell, parent):
    result = [cell]
    while parent[result[-1]] is not None:
        result.append(parent[result[-1]])
    return result


def _forest_is_challenging(parent, roots, nr: int, nc: int) -> bool:
    paths = [_path(cell, parent) for cell in parent]
    directions = set()
    turning_paths = []
    for path in paths:
        steps = []
        for a, b in zip(path, path[1:]):
            step = (b[0] - a[0], b[1] - a[1])
            directions.add(step)
            steps.append(step)
        if len(path) >= 4 and any(a != b for a, b in zip(steps, steps[1:])):
            turning_paths.append(path)
    if not turning_paths or len(directions) < 3:
        return False

    # Exclude the tempting shortcut "choose the geometrically nearest solid root".
    nearest_counterexample = False
    for cell in parent:
        reached = _path(cell, parent)[-1]
        distances = {root: abs(cell[0] - root[0]) + abs(cell[1] - root[1]) for root in roots}
        if distances[reached] > min(distances.values()):
            nearest_counterexample = True
            break
    return nearest_counterexample


def _direction_to(a: tuple[int, int], b: tuple[int, int]) -> str:
    delta = (b[0] - a[0], b[1] - a[1])
    return next(name for name, step in _DIRS.items() if step == delta)


def _sample_geometry(nr: int, nc: int):
    for _ in range(100):
        heights = [random.choice((3, 3, 5)) for _ in range(nr)]
        widths = [random.choice((3, 3, 5)) for _ in range(nc)]
        rgaps = [random.choice((2, 3)) for _ in range(nr - 1)]
        cgaps = [random.choice((2, 3)) for _ in range(nc - 1)]
        core_h = sum(heights) + sum(rgaps)
        core_w = sum(widths) + sum(cgaps)
        if core_h > 27 or core_w > 27:
            continue
        h = random.randint(core_h + 2, min(30, core_h + 6))
        w = random.randint(core_w + 2, min(30, core_w + 6))
        top = random.randint(1, h - core_h - 1)
        left = random.randint(1, w - core_w - 1)
        rows, cols = [], []
        r = top
        for hh, gap in zip(heights, rgaps + [0]):
            rows.append((r, r + hh - 1))
            r += hh + gap
        c = left
        for ww, gap in zip(widths, cgaps + [0]):
            cols.append((c, c + ww - 1))
            c += ww + gap
        return (h, w), rows, cols
    return None


def _render_notch(grid: np.ndarray, box, direction: str, bg: int):
    r0, c0, r1, c1 = box
    if direction == "U":
        grid[r0, (c0 + c1) // 2] = bg
    elif direction == "D":
        grid[r1, (c0 + c1) // 2] = bg
    elif direction == "L":
        grid[(r0 + r1) // 2, c0] = bg
    else:
        grid[(r0 + r1) // 2, c1] = bg


def _add_noise(grid: np.ndarray, boxes, bg: int):
    h, w = grid.shape
    forbidden = set()
    for r0, c0, r1, c1 in boxes:
        for r in range(max(0, r0 - 1), min(h, r1 + 2)):
            for c in range(max(0, c0 - 1), min(w, c1 + 2)):
                forbidden.add((r, c))
    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden]
    random.shuffle(candidates)
    chosen = []
    target = random.randint(5, 10)
    for pos in candidates:
        if all(abs(pos[0] - q[0]) + abs(pos[1] - q[1]) > 1 for q in chosen):
            chosen.append(pos)
            if len(chosen) == target:
                break
    if len(chosen) < 5:
        return None
    noise_palette = random.sample([c for c in range(10) if c != bg], 2)
    for i, (r, c) in enumerate(chosen):
        grid[r, c] = noise_palette[i % 2] if i < 2 else random.choice(noise_palette)
    return chosen


def _generate_attempt() -> np.ndarray | None:
    nr = random.choice((3, 3, 4))
    nc = random.choice((3, 3, 4))
    geometry = _sample_geometry(nr, nc)
    if geometry is None:
        return None
    shape, rows, cols = geometry

    forest_data = None
    for _ in range(120):
        candidate = _random_forest(nr, nc)
        if candidate is not None and _forest_is_challenging(*candidate, nr, nc):
            forest_data = candidate
            break
    if forest_data is None:
        return None
    parent, roots = forest_data

    bg = random.randrange(10)
    palette = [c for c in range(10) if c != bg]
    root_colors = random.sample(palette, 2)
    colors = {roots[i]: root_colors[i] for i in range(2)}
    for cell in parent:
        if cell not in colors:
            colors[cell] = random.choice(palette)

    # A deep turning chain must visibly defeat both self-color and one-hop-color rules.
    witnesses = []
    for cell in parent:
        path = _path(cell, parent)
        steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        if len(path) >= 4 and any(a != b for a, b in zip(steps, steps[1:])):
            root_color = colors[path[-1]]
            if colors[path[0]] != root_color and colors[path[1]] != root_color:
                witnesses.append(cell)
    if not witnesses:
        return None

    grid = np.full(shape, bg, dtype=int)
    boxes = []
    expected = np.empty((nr, nc), dtype=int)
    for r in range(nr):
        for c in range(nc):
            cell = (r, c)
            box = (rows[r][0], cols[c][0], rows[r][1], cols[c][1])
            boxes.append(box)
            r0, c0, r1, c1 = box
            grid[r0 : r1 + 1, c0 : c1 + 1] = colors[cell]
            if parent[cell] is not None:
                _render_notch(grid, box, _direction_to(cell, parent[cell]), bg)
            expected[cell] = colors[_path(cell, parent)[-1]]

    clean = grid.copy()
    noise = _add_noise(grid, boxes, bg)
    if noise is None:
        return None
    try:
        actual = main(grid)
        clean_actual = main(clean)
    except (AssertionError, IndexError, KeyError, StopIteration):
        return None
    if not np.array_equal(actual, expected) or not np.array_equal(clean_actual, expected):
        return None
    if len(set(map(int, expected.ravel()))) < 2 or expected.shape == grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked rectangle pointer array with distractors."""
    for _ in range(_GEN_TRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate notch-relay input")
