import random
from collections import Counter, deque

import numpy as np


BG = 5
MAROON = 9
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8]

GRID_H_RANGE = (25, 30)
GRID_W_RANGE = (25, 30)
HOLE_COUNT_RANGE = (2, 4)
HOLE_BBOX_MIN = 2
HOLE_BBOX_MAX = 4
GEN_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 60
FRAGMENT_ATTEMPTS = 200


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _normalize(cells) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _rotate(shape: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    pts = list(shape)
    for _ in range(turns % 4):
        h = max(r for r, _ in pts) + 1
        pts = [(c, h - 1 - r) for r, c in pts]
    return _normalize(pts)


def _canonical_shape(cells) -> tuple[tuple[int, int], ...]:
    base = _normalize(cells)
    return min(_rotate(base, k) for k in range(4))


def _shift(shape: tuple[tuple[int, int], ...], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(dr + r, dc + c) for r, c in shape]


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _shape_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _shape_boundary_count(shape: tuple[tuple[int, int], ...]) -> int:
    s = set(shape)
    return sum(any((nr, nc) not in s for nr, nc in neighbors4(r, c)) for r, c in s)


def _shape_interior_count(shape: tuple[tuple[int, int], ...]) -> int:
    s = set(shape)
    return sum(all((nr, nc) in s for nr, nc in neighbors4(r, c)) for r, c in s)


def _dense_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    h = rng.randint(max(3, HOLE_BBOX_MIN), HOLE_BBOX_MAX)
    w = rng.randint(max(3, HOLE_BBOX_MIN), HOLE_BBOX_MAX)
    cells = {(r, c) for r in range(h) for c in range(w)}
    remove_budget = rng.randint(0, max(1, (h * w) // 3))
    for _ in range(remove_budget):
        candidates = [cell for cell in cells if any((nr, nc) not in cells for nr, nc in neighbors4(*cell))]
        if not candidates or len(cells) <= 4:
            break
        removed = False
        for cell in rng.sample(candidates, len(candidates)):
            trial = set(cells)
            trial.remove(cell)
            if len({r for r, _ in trial}) < 2 or len({c for _, c in trial}) < 2:
                continue
            if _shape_connected(trial):
                cells = trial
                removed = True
                break
        if not removed:
            break
    return _normalize(cells)


def _growth_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    h = rng.randint(HOLE_BBOX_MIN, HOLE_BBOX_MAX)
    w = rng.randint(HOLE_BBOX_MIN, HOLE_BBOX_MAX)
    area = h * w
    lo = max(4, int(round(area * 0.5)))
    hi = max(lo, int(round(area * 0.8)))
    hi = min(hi, area)
    target = rng.randint(lo, hi)
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    return _normalize(cells)


def _random_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    for _ in range(100):
        shape = _dense_shape(rng) if rng.random() < 0.55 else _growth_shape(rng)
        if len(shape) < 4:
            continue
        if len({r for r, _ in shape}) < 2 or len({c for _, c in shape}) < 2:
            continue
        if _shape_boundary_count(shape) == len(shape) and rng.random() < 0.35:
            continue
        return shape
    raise RuntimeError("failed to generate shape")


def _hole_boundary(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    boundary = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                boundary.add((nr, nc))
    return boundary


def _extract_scene(g: np.ndarray):
    comps = _components(g != BG)
    assert comps, "input must contain non-grey objects"
    template = max(comps, key=len)
    template_set = set(template)
    fragments = [comp for comp in comps if comp is not template]

    h, w = g.shape
    holes = []
    for comp in _components(g == BG):
        if _touches_border(comp, h, w):
            continue
        hole_set = set(comp)
        boundary = set()
        valid = True
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in hole_set:
                    continue
                if int(g[nr, nc]) == BG:
                    continue
                if (nr, nc) not in template_set:
                    valid = False
                    break
                boundary.add((nr, nc))
            if not valid:
                break
        if valid and boundary:
            holes.append(
                {
                    "cells": comp,
                    "cell_set": hole_set,
                    "boundary": boundary,
                    "sig": _canonical_shape(comp),
                }
            )

    holes.sort(key=lambda item: min(item["cells"]))
    return template, fragments, holes


def _hole_colors(g: np.ndarray, boundary: set[tuple[int, int]]) -> tuple[int, int]:
    counts = Counter(int(g[r, c]) for r, c in boundary if int(g[r, c]) not in (BG, MAROON))
    assert len(counts) == 2, "each hole must expose exactly two border colors"
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[0][0], ordered[1][0]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _, fragments, holes = _extract_scene(g)
    fragment_counts = Counter(_canonical_shape(comp) for comp in fragments)
    hole_counts = Counter(hole["sig"] for hole in holes)
    assert holes, "template must contain at least one hole"
    assert fragment_counts == hole_counts, "fragments must match holes up to rotation"

    out = g.copy()
    for comp in fragments:
        for r, c in comp:
            out[r, c] = BG

    remaining = Counter(fragment_counts)
    for hole in holes:
        sig = hole["sig"]
        assert remaining[sig] > 0
        remaining[sig] -= 1
        fill_color, edge_color = _hole_colors(g, hole["boundary"])
        for r, c in hole["cells"]:
            on_edge = any((nr, nc) not in hole["cell_set"] for nr, nc in neighbors4(r, c))
            out[r, c] = edge_color if on_edge else fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _layout_holes(
    rng: random.Random,
    template_h: int,
    template_w: int,
    shapes: list[tuple[tuple[int, int], ...]],
):
    ranked = sorted(shapes, key=lambda s: (_shape_dims(s)[0] * _shape_dims(s)[1], len(s)), reverse=True)
    for _ in range(LAYOUT_ATTEMPTS):
        used_boxes: list[tuple[int, int, int, int]] = []
        infos = []
        ok = True
        for base in ranked:
            actual = _rotate(base, rng.randrange(4))
            sh, sw = _shape_dims(actual)
            candidates = []
            for r0 in range(2, template_h - sh - 1):
                for c0 in range(2, template_w - sw - 1):
                    box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                    if all(_boxes_separated(box, other, gap=2) for other in used_boxes):
                        candidates.append((r0, c0, box))
            if not candidates:
                ok = False
                break
            r0, c0, box = rng.choice(candidates)
            used_boxes.append(box)
            infos.append({"shape": actual, "box": box, "offset": (r0, c0)})
        if ok:
            return infos
    return None


def _choose_border_counts(rng: random.Random, boundary_size: int, tie: bool) -> tuple[int, int] | None:
    if tie:
        max_each = boundary_size // 2
        if max_each < 1:
            return None
        lo = 1 if max_each == 1 else 2
        k = rng.randint(lo, max_each)
        return k, k

    total_max = min(boundary_size, 7)
    if total_max < 3:
        return None
    total_min = 3 if total_max == 3 else 4
    total = rng.randint(total_min, total_max)
    major = rng.randint(total // 2 + 1, total - 1)
    minor = total - major
    return major, minor


def _can_place_fragment(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if occupied[r, c]:
            return False
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    hole_count = rng.randint(*HOLE_COUNT_RANGE)

    shapes = []
    seen = set()
    for _ in range(hole_count):
        for _ in range(120):
            shape = _random_shape(rng)
            sig = _canonical_shape(shape)
            if sig not in seen:
                shapes.append(shape)
                seen.add(sig)
                break
        else:
            return None

    if all(_shape_interior_count(shape) == 0 for shape in shapes):
        return None

    max_shape_h = max(_shape_dims(shape)[0] for shape in shapes)
    max_shape_w = max(_shape_dims(shape)[1] for shape in shapes)
    cluster_floor = 12 if hole_count <= 2 else 13 if hole_count == 3 else 14
    template_h_min = max(cluster_floor, max_shape_h + 5)
    template_w_min = max(cluster_floor, max_shape_w + 5)
    template_h_max = min(h - 8, 15)
    template_w_max = min(w - 8, 15)
    if template_h_min > template_h_max or template_w_min > template_w_max:
        return None

    template_h = rng.randint(template_h_min, template_h_max)
    template_w = rng.randint(template_w_min, template_w_max)
    if h - template_h - 4 <= 0 or w - template_w - 4 <= 0:
        return None
    top = rng.randint(2, h - template_h - 2)
    left = rng.randint(2, w - template_w - 2)

    hole_layout = _layout_holes(rng, template_h, template_w, shapes)
    if hole_layout is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[top : top + template_h, left : left + template_w] = MAROON

    hole_infos = []
    for info in hole_layout:
        abs_cells = _shift(info["shape"], top + info["offset"][0], left + info["offset"][1])
        for r, c in abs_cells:
            grid[r, c] = BG
        hole_infos.append(
            {
                "shape": info["shape"],
                "cells": abs_cells,
                "cell_set": set(abs_cells),
            }
        )

    tie_target = hole_count // 2 if hole_count % 2 == 0 else rng.choice([hole_count // 2, hole_count // 2 + 1])
    order = list(range(hole_count))
    rng.shuffle(order)
    tie_indices = set(order[:tie_target])

    for idx, info in enumerate(hole_infos):
        boundary = sorted(_hole_boundary(info["cell_set"], h, w))
        counts = _choose_border_counts(rng, len(boundary), idx in tie_indices)
        if counts is None:
            return None
        color_a, color_b = rng.sample(COLOR_POOL, 2)
        if idx in tie_indices:
            assign = [(color_a, counts[0]), (color_b, counts[1])]
        elif rng.random() < 0.5:
            assign = [(color_a, counts[0]), (color_b, counts[1])]
        else:
            assign = [(color_b, counts[0]), (color_a, counts[1])]

        rng.shuffle(boundary)
        cursor = 0
        for color, amount in assign:
            for r, c in boundary[cursor : cursor + amount]:
                grid[r, c] = color
            cursor += amount
        info["border_colors"] = {assign[0][0], assign[1][0]}

    occupied = grid != BG
    for info in sorted(hole_infos, key=lambda item: len(item["shape"]), reverse=True):
        frag_shape = _rotate(info["shape"], rng.randrange(4))
        fh, fw = _shape_dims(frag_shape)
        frag_colors = [color for color in COLOR_POOL if color not in info["border_colors"]]
        if not frag_colors:
            frag_colors = COLOR_POOL
        frag_color = rng.choice(frag_colors)

        placed = False
        for _ in range(FRAGMENT_ATTEMPTS):
            r0 = rng.randint(1, h - fh - 1)
            c0 = rng.randint(1, w - fw - 1)
            abs_cells = _shift(frag_shape, r0, c0)
            if _can_place_fragment(abs_cells, occupied):
                for r, c in abs_cells:
                    grid[r, c] = frag_color
                    occupied[r, c] = True
                placed = True
                break
        if not placed:
            return None

    try:
        _solve(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
