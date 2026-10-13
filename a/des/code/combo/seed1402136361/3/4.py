from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).flat))
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    rows, cols = mask.shape

    for r in range(rows):
        for c in range(cols):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < rows and 0 <= nc < cols and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _mask_is_connected(mask: np.ndarray) -> bool:
    comps = _components_from_mask(mask)
    return len(comps) == 1


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if not bool(mask[r, c]):
                continue
            if all(bool(mask[nr, nc]) for nr, nc in _neighbors4(r, c)):
                cells.append((r, c))
    return cells


def _bfs_distances(mask: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    rows, cols = mask.shape
    q = deque([start])
    dist = {start: 0}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < rows and 0 <= nc < cols and bool(mask[nr, nc]) and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def _choose_seed(mask: np.ndarray) -> tuple[tuple[int, int], int] | None:
    candidates = _interior_cells(mask)
    if not candidates:
        return None

    scored: list[tuple[tuple[int, int], int]] = []
    area = int(np.count_nonzero(mask))
    for cell in candidates:
        dists = _bfs_distances(mask, cell)
        if len(dists) != area:
            continue
        scored.append((cell, max(dists.values())))
    if not scored:
        return None

    best_depth = max(depth for _cell, depth in scored)
    min_depth = 3 if best_depth >= 3 and random.random() < 0.6 else 2
    valid = [item for item in scored if item[1] >= min_depth]
    if not valid:
        valid = [item for item in scored if item[1] == best_depth]
    return random.choice(valid)


def _gen_notched_rectangle() -> np.ndarray:
    for _ in range(80):
        h = random.randint(4, 8)
        w = random.randint(4, 8)
        mask = np.ones((h, w), dtype=bool)
        for _ in range(random.randint(0, 3)):
            side = random.choice(("top", "bottom", "left", "right"))
            trial = mask.copy()
            if side in {"top", "bottom"}:
                depth = random.randint(1, h - 2)
                span = random.randint(1, w - 1)
                start = random.randint(0, w - span)
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                trial[rows, start : start + span] = False
            else:
                depth = random.randint(1, w - 2)
                span = random.randint(1, h - 1)
                start = random.randint(0, h - span)
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                trial[start : start + span, cols] = False
            if np.count_nonzero(trial) < 8 or not _mask_is_connected(trial):
                continue
            if not _interior_cells(trial):
                continue
            mask = trial
        mask = _trim_mask(mask)
        if np.count_nonzero(mask) >= 8 and _interior_cells(mask):
            return mask
    raise RuntimeError("failed to generate notched rectangle")


def _gen_blob() -> np.ndarray:
    for _ in range(80):
        h = random.randint(5, 8)
        w = random.randint(5, 8)
        mask = np.zeros((h, w), dtype=bool)
        r0 = random.randint(1, h - 4)
        c0 = random.randint(1, w - 4)
        mask[r0 : r0 + 3, c0 : c0 + 3] = True

        target = random.randint(10, min(22, h * w - 1))
        while np.count_nonzero(mask) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
            if not frontier:
                break
            for nr, nc in random.sample(list(frontier), k=min(len(frontier), random.randint(1, 2))):
                mask[nr, nc] = True
                if np.count_nonzero(mask) >= target:
                    break

        mask = _trim_mask(mask)
        if np.count_nonzero(mask) >= 10 and _interior_cells(mask):
            return mask
    raise RuntimeError("failed to generate blob")


def _gen_diamond() -> np.ndarray:
    radius = random.randint(2, 4)
    size = 2 * radius + 1
    mask = np.zeros((size, size), dtype=bool)
    for dr in range(-radius, radius + 1):
        width = radius - abs(dr)
        row = radius + dr
        mask[row, radius - width : radius + width + 1] = True
    return mask


def _generate_shape() -> tuple[np.ndarray, tuple[int, int], int]:
    for _ in range(120):
        family = random.choice(("rect", "blob", "diamond"))
        if family == "rect":
            mask = _gen_notched_rectangle()
        elif family == "blob":
            mask = _gen_blob()
        else:
            mask = _gen_diamond()

        chosen = _choose_seed(mask)
        if chosen is None:
            continue
        seed, depth = chosen
        if depth >= 2:
            return mask, seed, depth
    raise RuntimeError("failed to generate a valid object")


def _fits(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    coords = np.argwhere(mask)
    rows = coords[:, 0] + top
    cols = coords[:, 1] + left
    return not bool(np.any(blocked[rows, cols]))


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    rows, cols = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    blocked[nr, nc] = True


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    seed: tuple[int, int],
    main_color: int,
    seed_color: int,
) -> int | None:
    rows, cols = grid.shape
    h, w = mask.shape
    if h > rows - 2 or w > cols - 2:
        return None

    options: list[tuple[int, int]] = []
    for top in range(1, rows - h):
        for left in range(1, cols - w):
            if _fits(blocked, mask, top, left):
                options.append((top, left))
    if not options:
        return None

    top, left = random.choice(options)
    placed: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r) + top
        cc = int(c) + left
        grid[rr, cc] = main_color
        placed.append((rr, cc))
    sr, sc = seed
    grid[top + sr, left + sc] = seed_color
    _mark_blocked(blocked, placed)
    return _bfs_distances(mask, seed)[max(_bfs_distances(mask, seed), key=_bfs_distances(mask, seed).get)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each object's interior seed outward in alternating 4-neighbor layers.

    Input structure:
    - The dominant color is the background.
    - Each foreground object is 4-connected and mostly monochrome.
    - Every object contains exactly one interior seed pixel of a different color.

    Rule:
    - For each object, treat the seed as layer 0 and keep its color.
    - Recolor all object cells by 4-neighbor distance from the seed.
    - Even-distance layers use the seed color; odd-distance layers use the object's main color.
    - Background stays unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    out = g.copy()
    components = _components_from_mask(g != bg)
    assert components, "expected at least one foreground object"

    for comp in components:
        rows = comp[:, 0]
        cols = comp[:, 1]
        values = g[rows, cols]
        colors, counts = np.unique(values, return_counts=True)
        assert len(colors) == 2, "each object must contain one main color and one seed color"

        seed_candidates = colors[counts == 1]
        assert len(seed_candidates) == 1, "expected exactly one singleton seed color per object"
        seed_color = int(seed_candidates[0])
        main_color = int(colors[colors != seed_color][0])

        seed_pos = np.argwhere((g == seed_color) & (g != bg))
        seed_coords = [(int(r), int(c)) for r, c in seed_pos if (int(r), int(c)) in {(int(rr), int(cc)) for rr, cc in comp}]
        assert len(seed_coords) == 1
        seed = seed_coords[0]

        comp_set = {(int(r), int(c)) for r, c in comp}
        assert all(nei in comp_set for nei in _neighbors4(*seed)), "seed must be inside its object"

        q = deque([seed])
        dist = {seed: 0}
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in comp_set and (nr, nc) not in dist:
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    q.append((nr, nc))
        assert len(dist) == len(comp_set)

        for r, c in comp_set:
            out[r, c] = seed_color if dist[(r, c)] % 2 == 0 else main_color

    return out


_GRID_MIN = 12
_GRID_MAX = 20
_OBJECT_COUNT_MIN = 1
_OBJECT_COUNT_MAX = 4
_MAX_FOREGROUND_FRACTION = 0.38
_GEN_ATTEMPTS = 400


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(_GRID_MIN, _GRID_MAX)
    cols = random.randint(_GRID_MIN, _GRID_MAX)
    bg = random.randrange(10)
    grid = np.full((rows, cols), bg, dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)

    object_count = random.randint(_OBJECT_COUNT_MIN, _OBJECT_COUNT_MAX)
    palette = [color for color in range(10) if color != bg]
    role_colors = random.sample(palette, 2 * object_count)

    max_depths: list[int] = []
    for idx in range(object_count):
        placed = False
        for _ in range(120):
            mask, seed, depth = _generate_shape()
            main_color = int(role_colors[2 * idx])
            seed_color = int(role_colors[2 * idx + 1])
            h, w = mask.shape
            if h > rows - 2 or w > cols - 2:
                continue
            options: list[tuple[int, int]] = []
            for top in range(1, rows - h):
                for left in range(1, cols - w):
                    if _fits(blocked, mask, top, left):
                        options.append((top, left))
            if not options:
                continue
            top, left = random.choice(options)
            placed_cells: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                rr = int(r) + top
                cc = int(c) + left
                grid[rr, cc] = main_color
                placed_cells.append((rr, cc))
            sr, sc = seed
            grid[top + sr, left + sc] = seed_color
            _mark_blocked(blocked, placed_cells)
            max_depths.append(depth)
            placed = True
            break
        if not placed:
            return None

    if np.mean(grid != bg) > _MAX_FOREGROUND_FRACTION:
        return None
    if max(max_depths) < 3:
        return None

    color_counts = Counter(map(int, grid.flat))
    if color_counts[bg] <= max(count for color, count in color_counts.items() if color != bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
