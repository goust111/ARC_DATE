from __future__ import annotations

import random

import numpy as np


BG = 7  # ARC orange
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _center_distance_sq(r: int, c: int, h: int, w: int) -> float:
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    return (r - cr) ** 2 + (c - cc) ** 2


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each color's non-singleton objects by the vector encoded by its singleton pair."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _dominant_color(g)
    out = np.full_like(g, bg)

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue

        comps = _find_components(g == color)
        singletons = [comp[0] for comp in comps if len(comp) == 1]
        movers = [comp for comp in comps if len(comp) > 1]
        assert len(singletons) == 2, "each foreground color must have exactly one instruction pair"
        assert movers, "each foreground color must have at least one movable object"

        (r1, c1), (r2, c2) = singletons
        assert (r1 == r2) ^ (c1 == c2), "instruction pixels must align in one row or one column"
        dist1 = _center_distance_sq(r1, c1, h, w)
        dist2 = _center_distance_sq(r2, c2, h, w)
        assert dist1 != dist2, "instruction direction must be unambiguous"
        tail = (r1, c1) if dist1 < dist2 else (r2, c2)
        head = (r2, c2) if tail == (r1, c1) else (r1, c1)

        dr = head[0] - tail[0]
        dc = head[1] - tail[1]
        step = abs(dr) + abs(dc)
        assert step >= 2, "instruction pixels must be non-adjacent"
        dr = 0 if dr == 0 else dr // abs(dr)
        dc = 0 if dc == 0 else dc // abs(dc)

        for comp in movers:
            for r, c in comp:
                nr = r + dr * step
                nc = c + dc * step
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = color

    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_polyomino(size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    frontier = {(0, 0)}
    while len(cells) < size:
        r, c = random.choice(tuple(frontier))
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            if (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.add((nr, nc))
                break
    return _normalize(cells)


def _coords_with_anchor(shape: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _touches_occupied(coords: list[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in coords:
        if (r, c) in occupied:
            return True
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in occupied:
                return True
    return False


def _translated(coords: list[tuple[int, int]], dr: int, dc: int, step: int) -> list[tuple[int, int]]:
    return [(r + dr * step, c + dc * step) for r, c in coords]


def _in_bounds_cells(coords: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r, c in coords if 0 <= r < h and 0 <= c < w]


def _pick_marker_pair(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    dr: int,
    dc: int,
    step: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            hr = r + dr * step
            hc = c + dc * step
            if not (0 <= hr < h and 0 <= hc < w):
                continue
            tail = (r, c)
            head = (hr, hc)
            if tail in occupied or head in occupied:
                continue
            if _center_distance_sq(*tail, h, w) >= _center_distance_sq(*head, h, w):
                continue
            if _touches_occupied([tail, head], occupied, h, w):
                continue
            candidates.append((tail, head))
    return random.choice(candidates) if candidates else None


def _place_color(
    h: int,
    w: int,
    occupied_in: set[tuple[int, int]],
    occupied_out: set[tuple[int, int]],
    color: int,
) -> tuple[list[tuple[int, int, int]], tuple[tuple[int, int], tuple[int, int]]] | None:
    direction = random.choice(DIRS4)
    dr, dc = direction
    max_step = max(2, min(h, w) // 2)
    steps = list(range(2, max_step + 1))
    random.shuffle(steps)

    clip_flag = random.choice([False, True])
    obj_count = random.randint(1, 2)

    for step in steps:
        objects: list[list[tuple[int, int]]] = []
        local_in = set(occupied_in)
        local_out = set(occupied_out)
        ok = True

        for _ in range(obj_count):
            placed = False
            for _try in range(80):
                shape = _random_polyomino(random.randint(2, 8))
                sh = max(r for r, _ in shape) + 1
                sw = max(c for _, c in shape) + 1
                top = random.randint(0, h - sh)
                left = random.randint(0, w - sw)
                coords = _coords_with_anchor(shape, top, left)
                if _touches_occupied(coords, local_in, h, w):
                    continue

                shifted = _translated(coords, dr, dc, step)
                visible = _in_bounds_cells(shifted, h, w)
                if clip_flag:
                    if len(visible) == len(coords) or not visible:
                        continue
                else:
                    if len(visible) != len(coords):
                        continue

                if len(set(visible)) != len(visible):
                    continue
                if _touches_occupied(visible, local_out, h, w):
                    continue

                objects.append(coords)
                local_in.update(coords)
                local_out.update(visible)
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        marker_pair = _pick_marker_pair(h, w, local_in, dr, dc, step)
        if marker_pair is None:
            continue

        tail, head = marker_pair
        local_in.update([tail, head])
        encoded: list[tuple[int, int, int]] = []
        for obj in objects:
            for r, c in obj:
                encoded.append((r, c, color))
        return encoded, marker_pair

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(12, 20)
        w = random.randint(12, 20)
        grid = np.full((h, w), BG, dtype=int)

        n_colors = random.randint(2, min(5, len(PALETTE)))
        colors = random.sample(PALETTE, n_colors)
        occupied_in: set[tuple[int, int]] = set()
        occupied_out: set[tuple[int, int]] = set()
        placements: list[tuple[int, list[tuple[int, int, int]], tuple[tuple[int, int], tuple[int, int]]]] = []

        success = True
        for color in colors:
            placed = _place_color(h, w, occupied_in, occupied_out, color)
            if placed is None:
                success = False
                break

            obj_pixels, (tail, head) = placed
            for r, c, _color in obj_pixels:
                occupied_in.add((r, c))
                grid[r, c] = _color
            for pos in (tail, head):
                occupied_in.add(pos)
                grid[pos] = color

            shifted = _in_bounds_cells(
                _translated([(r, c) for r, c, _ in obj_pixels], head[0] - tail[0] and (1 if head[0] > tail[0] else -1), head[1] - tail[1] and (1 if head[1] > tail[1] else -1), abs(head[0] - tail[0]) + abs(head[1] - tail[1])),
                h,
                w,
            )
            occupied_out.update(shifted)
            placements.append((color, obj_pixels, (tail, head)))

        if not success:
            continue

        if len(np.unique(grid)) < 3:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input")
