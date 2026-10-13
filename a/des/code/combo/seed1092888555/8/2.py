import random
from collections import Counter, defaultdict

import numpy as np


BG = 8  # azure
FG_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_ATTEMPTS = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _connected_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            local = {(rr - r0, cc - c0) for rr, cc in cells}
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                    "local": local,
                }
            )
    return comps


def _best_alignment(ref: set[tuple[int, int]], other: set[tuple[int, int]], max_shift: int = 2):
    best = None
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            shifted = {(r + dr, c + dc) for r, c in other}
            diff = len(ref ^ shifted)
            key = (diff, abs(dr) + abs(dc), abs(dr), abs(dc), dr, dc)
            if best is None or key < best[0]:
                best = (key, shifted, (dr, dc))
    return best[1], best[2], best[0][0]


def _group_output(group: list[dict], color: int, bg: int) -> np.ndarray:
    n = len(group)
    assert 3 <= n <= 4, "each color group must contain 3 to 4 objects"

    shapes = [obj["local"] for obj in group]

    exemplar_idx = None
    exemplar_cost = None
    for i, ref in enumerate(shapes):
        cost = 0
        for j, other in enumerate(shapes):
            if i == j:
                continue
            _, _, diff = _best_alignment(ref, other)
            cost += diff
        key = (cost, i)
        if exemplar_cost is None or key < exemplar_cost:
            exemplar_cost = key
            exemplar_idx = i

    ref = shapes[exemplar_idx]
    aligned = []
    for shape in shapes:
        shifted, _offset, _diff = _best_alignment(ref, shape)
        aligned.append(shifted)

    counts = Counter()
    for shape in aligned:
        for cell in shape:
            counts[cell] += 1
    standard = {cell for cell, cnt in counts.items() if cnt > n // 2}
    assert standard, "majority-vote standard shape must be non-empty"

    diffs = [len(shape ^ standard) for shape in aligned]
    bad = [i for i, d in enumerate(diffs) if d > 0]
    assert len(bad) == 1, "expected one defective object per color group"
    defect = aligned[bad[0]]
    delta = standard ^ defect
    assert 1 <= len(delta) <= 2, "defect must differ by 1 to 2 pixels"

    coords = standard | defect
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    for r, c in delta:
        out[r - min_r, c - min_c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the defective object within each same-color group, then output only
    the pixels where that object differs from the group's majority-vote shape.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    comps = _connected_components(g, bg)

    groups = defaultdict(list)
    first_seen = {}
    for comp in comps:
        color = int(comp["color"])
        groups[color].append(comp)
        r0, c0, _r1, _c1 = comp["bbox"]
        pos = (r0, c0)
        if color not in first_seen or pos < first_seen[color]:
            first_seen[color] = pos

    ordered_colors = [color for color, _pos in sorted(first_seen.items(), key=lambda kv: kv[1])]
    rows = [_group_output(groups[color], color, bg) for color in ordered_colors]
    total_h = sum(row.shape[0] for row in rows)
    max_w = max(row.shape[1] for row in rows)
    out = np.full((total_h, max_w), bg, dtype=int)
    cursor = 0
    for row in rows:
        h, w = row.shape
        out[cursor : cursor + h, :w] = row
        cursor += h
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _random_shape(rng: random.Random) -> set[tuple[int, int]]:
    h = rng.randint(2, 5)
    w = rng.randint(2, 5)
    hi = min(h * w, 8)
    lo = min(hi, max(4, (h * w) // 2))
    target = rng.randint(lo, hi)
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < target:
        r, c = rng.choice(tuple(cells))
        nbrs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        rng.shuffle(nbrs)
        for nr, nc in nbrs:
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                cells.add((nr, nc))
                break
    norm = _normalize(cells)
    box_h = max(r for r, _ in norm) + 1
    box_w = max(c for _, c in norm) + 1
    if box_h == 1 or box_w == 1:
        return _random_shape(rng)
    return norm


def _add_defect(shape: set[tuple[int, int]], rng: random.Random, count: int) -> set[tuple[int, int]] | None:
    current = set(shape)
    for _ in range(count):
        candidates = set()
        for r, c in current:
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if (nr, nc) not in current:
                    candidates.add((nr, nc))
        if not candidates:
            return None
        chosen = rng.choice(tuple(candidates))
        current.add(chosen)
    return _normalize(current)


def _remove_defect(shape: set[tuple[int, int]], rng: random.Random, count: int) -> set[tuple[int, int]] | None:
    if len(shape) - count < 3:
        return None
    current = set(shape)
    for _ in range(count):
        removable = []
        for cell in current:
            trial = current - {cell}
            if _is_connected(trial):
                removable.append(cell)
        if not removable:
            return None
        current.remove(rng.choice(removable))
    return _normalize(current)


def _make_group_shapes(rng: random.Random):
    base = _random_shape(rng)
    branch = rng.choice(("add", "remove"))
    changes = rng.choice((1, 2))
    if branch == "add":
        defect = _add_defect(base, rng, changes)
    else:
        defect = _remove_defect(base, rng, changes)
    if defect is None or defect == base:
        return None

    count = rng.choice((3, 4))
    shapes = [set(base) for _ in range(count - 1)] + [set(defect)]
    rng.shuffle(shapes)
    return shapes


def _bbox(shape: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _fits(mask: np.ndarray, top: int, left: int, shape: set[tuple[int, int]]) -> bool:
    h, w = mask.shape
    sh, sw = _bbox(shape)
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + sh + 1)
    c1 = min(w, left + sw + 1)
    return not mask[r0:r1, c0:c1].any()


def _paint(grid: np.ndarray, mask: np.ndarray, top: int, left: int, shape: set[tuple[int, int]], color: int) -> None:
    sh, sw = _bbox(shape)
    mask[top : top + sh, left : left + sw] = True
    for r, c in shape:
        grid[top + r, left + c] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 28)
    w = rng.randint(16, 28)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    num_groups = rng.randint(2, 4)
    colors = rng.sample(FG_COLORS, num_groups)

    objects = []
    for color in colors:
        shapes = _make_group_shapes(rng)
        if shapes is None:
            return None
        for shape in shapes:
            objects.append((color, shape))
    rng.shuffle(objects)

    for color, shape in objects:
        sh, sw = _bbox(shape)
        placed = False
        for _ in range(200):
            top = rng.randint(0, h - sh)
            left = rng.randint(0, w - sw)
            if _fits(occupied, top, left, shape):
                _paint(grid, occupied, top, left, shape, color)
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.size == 0 or np.all(out == BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
