from __future__ import annotations

import random

import numpy as np


AZURE = 8
RED = 2
GREEN = 3
YELLOW = 4
MAROON = 9

SIDES = ("top", "bottom", "left", "right")
GENERATE_RETRIES = 400
OBJECT_TRIES = 120


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _rect_area_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expanded_bbox_cells(
    r0: int, c0: int, r1: int, c1: int, h: int, w: int, margin: int = 1
) -> set[tuple[int, int]]:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    return {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}


def _find_absorbers(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    absorbers: list[dict[str, object]] = []
    border_lookup: dict[tuple[int, int], int] = {}
    for comp in _components4(g == MAROON):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _rect_border_cells(r0, c0, r1, c1)
        if set(comp) != border:
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not np.all(inner == AZURE):
            continue
        idx = len(absorbers)
        absorbers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_h": r1 - r0 - 1,
                "inner_w": c1 - c0 - 1,
            }
        )
        for cell in border:
            border_lookup[cell] = idx
    return absorbers, border_lookup


def _find_emitters(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    emitters: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in np.argwhere(g == RED):
        r = int(r)
        c = int(c)
        if r == 0 and 0 < c < w - 1:
            emitters.append(((r, c), (1, 0)))
        elif r == h - 1 and 0 < c < w - 1:
            emitters.append(((r, c), (-1, 0)))
        elif c == 0 and 0 < r < h - 1:
            emitters.append(((r, c), (0, 1)))
        elif c == w - 1 and 0 < r < h - 1:
            emitters.append(((r, c), (0, -1)))
    emitters.sort()
    return emitters


def _cast_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    border_lookup: dict[tuple[int, int], int],
) -> dict[str, object] | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    dr, dc = direction
    r, c = start
    path: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        value = int(g[r, c])
        if value == AZURE:
            path.append((r, c))
            continue
        if value == MAROON and (r, c) in border_lookup:
            return {
                "path": path,
                "absorber_idx": border_lookup[(r, c)],
                "impact": (r, c),
                "direction": direction,
            }
        return None


def _fill_absorber(out: np.ndarray, absorber: dict[str, object], direction: tuple[int, int], amount: int) -> None:
    if amount <= 0:
        return
    r0, c0, r1, c1 = absorber["bbox"]
    inner_h = int(absorber["inner_h"])
    inner_w = int(absorber["inner_w"])
    dr, dc = direction
    if dr == 1:
        rows = min(amount, inner_h)
        out[r0 + 1 : r0 + 1 + rows, c0 + 1 : c1] = YELLOW
    elif dr == -1:
        rows = min(amount, inner_h)
        out[r1 - rows : r1, c0 + 1 : c1] = YELLOW
    elif dc == 1:
        cols = min(amount, inner_w)
        out[r0 + 1 : r1, c0 + 1 : c0 + 1 + cols] = YELLOW
    else:
        cols = min(amount, inner_w)
        out[r0 + 1 : r1, c1 - cols : c1] = YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace edge emitters into hollow absorbers, marking the ray path and encoding its length as interior fill.

    Input structure:
    - The background is Azure.
    - Red single-pixel emitters sit on the grid edge and point orthogonally inward.
    - Maroon one-pixel-thick hollow rectangles act as absorbers with Azure interiors.

    Rule:
    - For each edge emitter, cast a straight ray inward according to its edge: top goes down, bottom goes up, left goes right, and right goes left.
    - If the ray crosses only Azure cells and first hits an absorber border, color every crossed Azure cell Green.
    - Let the ray length be the number of crossed Azure cells; fill the struck absorber interior from the impact side with Yellow rows or columns equal to that length.
    - If the ray reaches the opposite edge without striking an absorber border, leave that emitter unchanged.

    Key insight:
    - The same straight-line distance controls two coupled outputs: the external Green trace and the amount of Yellow striping inside the absorber.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    absorbers, border_lookup = _find_absorbers(g)
    for start, direction in _find_emitters(g):
        ray = _cast_ray(g, start, direction, border_lookup)
        if ray is None:
            continue
        path = ray["path"]
        for r, c in path:
            out[r, c] = GREEN
        absorber = absorbers[int(ray["absorber_idx"])]
        _fill_absorber(out, absorber, direction, len(path))
    return out


def _can_place_absorber(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    h: int,
    w: int,
    absorber_areas: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    emitters: set[tuple[int, int]],
) -> bool:
    if not (1 <= r0 < r1 < h - 1 and 1 <= c0 < c1 < w - 1):
        return False
    area = _rect_area_cells(r0, c0, r1, c1)
    if area & reserved_paths:
        return False
    if area & emitters:
        return False
    if _expanded_bbox_cells(r0, c0, r1, c1, h, w, margin=1) & absorber_areas:
        return False
    return True


def _apply_absorber(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    absorber_areas: set[tuple[int, int]],
) -> None:
    for r, c in _rect_border_cells(r0, c0, r1, c1):
        grid[r, c] = MAROON
    absorber_areas.update(_rect_area_cells(r0, c0, r1, c1))


def _place_hit(
    grid: np.ndarray,
    side: str,
    absorber_areas: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    emitters: set[tuple[int, int]],
    planned_hits: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for _ in range(OBJECT_TRIES):
        if side in ("top", "bottom"):
            max_len = min(6, (h - 6) // 2)
            if max_len < 1:
                return False
            length = random.randint(1, max_len)
            max_inner_h = min(8, h - length - 4)
            max_inner_w = min(8, w - 4)
            if max_inner_h < length or max_inner_w < 2:
                continue
            inner_h = random.randint(length, max_inner_h)
            inner_w = random.randint(2, max_inner_w)
            c0 = random.randint(1, w - inner_w - 3)
            c1 = c0 + inner_w + 1
            x = random.randint(c0 + 1, c1 - 1)
            if side == "top":
                emit = (0, x)
                r0 = length + 1
                r1 = r0 + inner_h + 1
                path = [(r, x) for r in range(1, r0)]
            else:
                emit = (h - 1, x)
                r1 = h - 2 - length
                r0 = r1 - inner_h - 1
                path = [(r, x) for r in range(h - 2, r1, -1)]
        else:
            max_len = min(6, (w - 6) // 2)
            if max_len < 1:
                return False
            length = random.randint(1, max_len)
            max_inner_w = min(8, w - length - 4)
            max_inner_h = min(8, h - 4)
            if max_inner_w < length or max_inner_h < 2:
                continue
            inner_w = random.randint(length, max_inner_w)
            inner_h = random.randint(2, max_inner_h)
            r0 = random.randint(1, h - inner_h - 3)
            r1 = r0 + inner_h + 1
            y = random.randint(r0 + 1, r1 - 1)
            if side == "left":
                emit = (y, 0)
                c0 = length + 1
                c1 = c0 + inner_w + 1
                path = [(y, c) for c in range(1, c0)]
            else:
                emit = (y, w - 1)
                c1 = w - 2 - length
                c0 = c1 - inner_w - 1
                path = [(y, c) for c in range(w - 2, c1, -1)]

        if emit in emitters:
            continue
        if not _can_place_absorber(r0, c0, r1, c1, h, w, absorber_areas, reserved_paths, emitters):
            continue
        if any(cell in absorber_areas or cell in reserved_paths for cell in path):
            continue
        grid[emit] = RED
        emitters.add(emit)
        planned_hits.add(emit)
        reserved_paths.update(path)
        _apply_absorber(grid, r0, c0, r1, c1, absorber_areas)
        return True
    return False


def _place_miss(
    grid: np.ndarray,
    side: str,
    absorber_areas: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    emitters: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for _ in range(OBJECT_TRIES):
        if side in ("top", "bottom"):
            x = random.randint(1, w - 2)
            emit = (0, x) if side == "top" else (h - 1, x)
            opposite = (h - 1, x) if side == "top" else (0, x)
            path = [(r, x) for r in range(1, h - 1)]
        else:
            y = random.randint(1, h - 2)
            emit = (y, 0) if side == "left" else (y, w - 1)
            opposite = (y, w - 1) if side == "left" else (y, 0)
            path = [(y, c) for c in range(1, w - 1)]

        if emit in emitters or opposite in emitters:
            continue
        if any(cell in absorber_areas or cell in reserved_paths for cell in path):
            continue
        grid[emit] = RED
        emitters.add(emit)
        reserved_paths.update(path)
        return True
    return False


def _place_extra_absorber(
    grid: np.ndarray,
    absorber_areas: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    emitters: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for _ in range(OBJECT_TRIES):
        inner_h = random.randint(2, min(8, h - 4))
        inner_w = random.randint(2, min(8, w - 4))
        r0 = random.randint(1, h - inner_h - 3)
        c0 = random.randint(1, w - inner_w - 3)
        r1 = r0 + inner_h + 1
        c1 = c0 + inner_w + 1
        if not _can_place_absorber(r0, c0, r1, c1, h, w, absorber_areas, reserved_paths, emitters):
            continue
        _apply_absorber(grid, r0, c0, r1, c1, absorber_areas)
        return True
    return False


def _validate_grid(grid: np.ndarray, expected_hits: int, expected_misses: int) -> bool:
    absorbers, border_lookup = _find_absorbers(grid)
    emitters = _find_emitters(grid)
    if len(emitters) != expected_hits + expected_misses:
        return False
    if len(absorbers) < expected_hits:
        return False
    hit_count = 0
    miss_count = 0
    lengths: list[int] = []
    for start, direction in emitters:
        ray = _cast_ray(grid, start, direction, border_lookup)
        if ray is None:
            miss_count += 1
        else:
            hit_count += 1
            lengths.append(len(ray["path"]))
    if hit_count != expected_hits or miss_count != expected_misses:
        return False
    if expected_hits > 0 and (not lengths or min(lengths) < 1):
        return False
    out = main(grid)
    if out.shape != grid.shape:
        return False
    if expected_hits > 0:
        if not np.any(out == GREEN):
            return False
        if not np.any(out == YELLOW):
            return False
        if np.array_equal(out, grid):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        h = random.randint(14, 30)
        w = random.randint(14, 30)
        grid = np.full((h, w), AZURE, dtype=int)

        emitter_count = random.randint(2, 6)
        hit_count = emitter_count // 2 if random.random() < 0.5 else (emitter_count + 1) // 2
        hit_count = max(1, min(hit_count, emitter_count - 1))
        miss_count = emitter_count - hit_count
        extra_absorbers = random.randint(0, 2)

        absorber_areas: set[tuple[int, int]] = set()
        reserved_paths: set[tuple[int, int]] = set()
        emitters: set[tuple[int, int]] = set()
        planned_hits: set[tuple[int, int]] = set()

        hit_sides = [random.choice(SIDES) for _ in range(hit_count)]
        random.shuffle(hit_sides)
        ok = True
        for side in hit_sides:
            if not _place_hit(grid, side, absorber_areas, reserved_paths, emitters, planned_hits):
                ok = False
                break
        if not ok:
            continue

        miss_sides = [random.choice(SIDES) for _ in range(miss_count)]
        random.shuffle(miss_sides)
        for side in miss_sides:
            if not _place_miss(grid, side, absorber_areas, reserved_paths, emitters):
                ok = False
                break
        if not ok:
            continue

        for _extra in range(extra_absorbers):
            if not _place_extra_absorber(grid, absorber_areas, reserved_paths, emitters):
                ok = False
                break
        if not ok:
            continue

        if _validate_grid(grid, hit_count, miss_count):
            return grid

    raise RuntimeError("failed to generate a valid input grid")
