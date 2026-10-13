from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _labeled_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    labels = -np.ones(mask.shape, dtype=int)
    comps = _components(mask)
    for i, pts in enumerate(comps):
        for r, c in pts:
            labels[r, c] = i
    return labels, comps


def _touches_edge(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _trace_cycle_dirs(points: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    pts = set(points)
    if len(pts) < 4:
        return None

    neigh: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in pts:
        ns = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in pts]
        if len(ns) != 2:
            return None
        neigh[(r, c)] = sorted(ns)

    start = min(pts)
    prev = start
    cur = neigh[start][0]
    dirs = [(cur[0] - prev[0], cur[1] - prev[1])]
    visited = {start}

    while cur != start:
        if cur in visited:
            return None
        visited.add(cur)
        choices = [p for p in neigh[cur] if p != prev]
        if len(choices) != 1:
            return None
        nxt = choices[0]
        dirs.append((nxt[0] - cur[0], nxt[1] - cur[1]))
        prev, cur = cur, nxt
        if len(dirs) > len(pts) + 1:
            return None

    if len(visited) != len(pts):
        return None
    return dirs


def _dir_axis(direction: tuple[int, int]) -> int:
    return 0 if direction[0] else 1


def _run_lengths(dirs: list[tuple[int, int]]) -> list[tuple[tuple[int, int], int]]:
    runs: list[list] = []
    for d in dirs:
        if runs and runs[-1][0] == d:
            runs[-1][1] += 1
        else:
            runs.append([d, 1])
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0][1] += runs[-1][1]
        runs.pop()
    return [(tuple(d), int(n)) for d, n in runs]


def _is_stair_run(runs: list[tuple[tuple[int, int], int]]) -> bool:
    if len(runs) < 4:
        return False
    if any(n != 1 for _d, n in runs):
        return False
    vdirs = {d for d, _n in runs if d[0] != 0}
    hdirs = {d for d, _n in runs if d[1] != 0}
    return len(vdirs) == 1 and len(hdirs) == 1


def _logical_side_count(points: list[tuple[int, int]]) -> int:
    dirs = _trace_cycle_dirs(points)
    if dirs is None:
        pts = set(points)
        corners = 0
        for r, c in pts:
            has_v = (r - 1, c) in pts or (r + 1, c) in pts
            has_h = (r, c - 1) in pts or (r, c + 1) in pts
            if has_v and has_h:
                corners += 1
        return corners

    runs = _run_lengths(dirs)
    if any(n > 1 for _d, n in runs):
        pivot = next(i for i, (_d, n) in enumerate(runs) if n > 1)
        runs = runs[pivot:] + runs[:pivot]

    count = 0
    i = 0
    while i < len(runs):
        if runs[i][1] != 1:
            count += 1
            i += 1
            continue

        j = i + 1
        while j < len(runs) and runs[j][1] == 1 and _dir_axis(runs[j][0]) != _dir_axis(runs[j - 1][0]):
            j += 1
        stretch = runs[i:j]
        count += 1 if _is_stair_run(stretch) else len(stretch)
        i = j
    return count


def _find_polygons(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    bg_mask = g == BG
    border_labels, border_comps = _labeled_components(~bg_mask)
    polygons: list[dict] = []
    used_pairs: set[tuple[int, tuple[tuple[int, int], ...]]] = set()

    for interior in _components(bg_mask):
        if _touches_edge(interior, h, w):
            continue
        adjacent_labels: set[int] = set()
        for r, c in interior:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(border_labels[nr, nc]) >= 0:
                    adjacent_labels.add(int(border_labels[nr, nc]))
        assert len(adjacent_labels) == 1, "each interior must touch exactly one polygon border"
        label = next(iter(adjacent_labels))
        key = (label, tuple(sorted(interior)))
        if key in used_pairs:
            continue
        used_pairs.add(key)
        border = border_comps[label]
        polygons.append(
            {
                "interior": interior,
                "border": border,
                "sides": _logical_side_count(border),
                "colors": {int(g[r, c]) for r, c in border},
            }
        )

    assert polygons, "expected at least one closed polygon"
    return polygons


def _majority_color(g: np.ndarray, border: list[tuple[int, int]]) -> int:
    counts = Counter(int(g[r, c]) for r, c in border)
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "odd polygon must have a unique most common border color"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed polygon according to the parity of its border side count.

    Input structure:
    - The background is Azure.
    - Several separated one-pixel-thick closed polygon borders are drawn with
      colored horizontal or vertical line segments.
    - Each polygon encloses an Azure interior region; polygons do not overlap,
      touch, nest, or share vertices.

    Rule:
    - Keep every border pixel unchanged.
    - For each enclosed Azure interior, recover its surrounding border component
      and count the polygon sides by the direction changes around that border.
    - If the side count is odd, fill the interior with the border color that
      appears most often on that polygon.
    - If the side count is even, fill the interior with the one border color
      used by that polygon and by no other polygon border.

    Key insight:
    - The border first encodes which color rule to use through side-count
      parity; odd polygons use local frequency, while even polygons use global
      color uniqueness across all polygon borders.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()
    polygons = _find_polygons(g)
    color_sets = [set(poly["colors"]) for poly in polygons]

    for i, poly in enumerate(polygons):
        if int(poly["sides"]) % 2 == 1:
            fill = _majority_color(g, poly["border"])
        else:
            others: set[int] = set()
            for j, colors in enumerate(color_sets):
                if i != j:
                    others.update(colors)
            unique = sorted(color_sets[i] - others)
            assert len(unique) == 1, "even polygon must have exactly one border color unique to it"
            fill = int(unique[0])
        for r, c in poly["interior"]:
            out[r, c] = fill
    return out


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        cells = [(r0, c) for c in range(lo, hi + 1)]
        return cells if c0 <= c1 else list(reversed(cells))
    assert c0 == c1
    lo, hi = sorted((r0, r1))
    cells = [(r, c0) for r in range(lo, hi + 1)]
    return cells if r0 <= r1 else list(reversed(cells))


def _stair(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    br, bc = b
    out = [(r, c)]
    dr = 1 if br > r else -1
    dc = 1 if bc > c else -1
    while (r, c) != (br, bc):
        if r != br:
            r += dr
            out.append((r, c))
        if c != bc:
            c += dc
            out.append((r, c))
    return out


def _shape_rectangle(rng: random.Random) -> dict:
    h = rng.randint(4, 8)
    w = rng.randint(4, 9)
    sides = [
        _line((0, 0), (0, w)),
        _line((0, w), (h, w)),
        _line((h, w), (h, 0)),
        _line((h, 0), (0, 0)),
    ]
    return {"sides": sides, "height": h + 1, "width": w + 1, "logical": 4}


def _shape_cut_corner(rng: random.Random) -> dict:
    h = rng.randint(6, 9)
    w = rng.randint(6, 10)
    cut_h = rng.randint(2, h - 3)
    cut_w = rng.randint(2, w - 3)
    verts = [(0, cut_w), (0, w), (h, w), (h, 0), (cut_h, 0), (cut_h, cut_w)]
    sides = [_line(verts[i], verts[(i + 1) % len(verts)]) for i in range(len(verts))]
    return {"sides": sides, "height": h + 1, "width": w + 1, "logical": 6}


def _shape_stair_triangle(rng: random.Random) -> dict:
    n = rng.randint(4, 7)
    sides = [
        _line((0, 0), (0, n)),
        _stair((0, n), (n, 0)),
        _line((n, 0), (0, 0)),
    ]
    return {"sides": sides, "height": n + 1, "width": n + 1, "logical": 3}


def _shape_stair_pentagon(rng: random.Random) -> dict:
    h = rng.randint(7, 10)
    w = rng.randint(7, 10)
    cut = rng.randint(3, min(h, w) - 3)
    sides = [
        _line((0, cut), (0, w)),
        _line((0, w), (h, w)),
        _line((h, w), (h, 0)),
        _line((h, 0), (cut, 0)),
        _stair((cut, 0), (0, cut)),
    ]
    return {"sides": sides, "height": h + 1, "width": w + 1, "logical": 5}


def _make_shape(kind: str, rng: random.Random) -> dict:
    if kind == "rect":
        return _shape_rectangle(rng)
    if kind == "cut6":
        return _shape_cut_corner(rng)
    if kind == "tri3":
        return _shape_stair_triangle(rng)
    return _shape_stair_pentagon(rng)


def _side_lengths(sides: list[list[tuple[int, int]]]) -> list[int]:
    return [len(set(side)) for side in sides]


def _assign_side_colors(shape: dict, rng: random.Random, common: tuple[int, int], unique: int | None) -> list[int]:
    sides = shape["sides"]
    if unique is not None:
        colors = [common[i % len(common)] for i in range(len(sides))]
        colors[rng.randrange(len(sides))] = int(unique)
        return colors

    lengths = _side_lengths(sides)
    colors = [common[0]] * len(sides)
    colors[int(np.argmin(lengths))] = common[1]
    return colors


def _draw_shape(g: np.ndarray, r0: int, c0: int, shape: dict, side_colors: list[int]) -> None:
    for side, color in zip(shape["sides"], side_colors):
        for r, c in side:
            g[r0 + r, c0 + c] = int(color)


def _fits(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + height > h - 1 or c0 + width > w - 1:
        return False
    rlo, rhi = max(0, r0 - 1), min(h, r0 + height + 1)
    clo, chi = max(0, c0 - 1), min(w, c0 + width + 1)
    return not bool(occupied[rlo:rhi, clo:chi].any())


def _place_all(g: np.ndarray, shapes: list[dict], rng: random.Random) -> bool:
    occupied = np.zeros(g.shape, dtype=bool)
    order = sorted(range(len(shapes)), key=lambda i: shapes[i]["height"] * shapes[i]["width"], reverse=True)
    for idx in order:
        shape = shapes[idx]
        positions = []
        for r0 in range(1, g.shape[0] - shape["height"]):
            for c0 in range(1, g.shape[1] - shape["width"]):
                if _fits(occupied, r0, c0, int(shape["height"]), int(shape["width"])):
                    positions.append((r0, c0))
        if not positions:
            return False
        r0, c0 = rng.choice(positions)
        shape["top"] = r0
        shape["left"] = c0
        rlo, rhi = max(0, r0 - 1), min(g.shape[0], r0 + int(shape["height"]) + 1)
        clo, chi = max(0, c0 - 1), min(g.shape[1], c0 + int(shape["width"]) + 1)
        occupied[rlo:rhi, clo:chi] = True
    return True


def _valid_generated(g: np.ndarray, expected_count: int) -> bool:
    try:
        polygons = _find_polygons(g)
        out = main(g)
    except AssertionError:
        return False
    if len(polygons) != expected_count:
        return False
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    if np.array_equal(out, g):
        return False
    if not np.array_equal(out[g != BG], g[g != BG]):
        return False
    sides = [int(poly["sides"]) for poly in polygons]
    if not any(side % 2 == 1 for side in sides):
        return False
    if not any(side % 2 == 0 for side in sides):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(2, 4)
    kinds = [rng.choice(("tri3", "pent5")), rng.choice(("rect", "cut6"))]
    while len(kinds) < count:
        kinds.append(rng.choice(("tri3", "pent5", "rect", "cut6")))
    rng.shuffle(kinds)

    height = rng.randint(18, 30)
    width = rng.randint(18, 30)
    shapes = [_make_shape(kind, rng) for kind in kinds]
    even_indices = [i for i, shape in enumerate(shapes) if int(shape["logical"]) % 2 == 0]

    palette = list(PALETTE)
    rng.shuffle(palette)
    common = (palette.pop(), palette.pop())
    if len(palette) < len(even_indices):
        return None
    unique_by_index = {idx: palette.pop() for idx in even_indices}

    grid = np.full((height, width), BG, dtype=int)
    if not _place_all(grid, shapes, rng):
        return None

    for i, shape in enumerate(shapes):
        colors = _assign_side_colors(shape, rng, common, unique_by_index.get(i))
        _draw_shape(grid, int(shape["top"]), int(shape["left"]), shape, colors)

    if not _valid_generated(grid, count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed1030756833/13/6 input")
