import random
from collections import deque

import numpy as np


BG = 0
LOCK = 5
SHADOW = 9
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8)

LOCK_COUNT_RANGE = (2, 4)
SHAPE_AREA_RANGE = (4, 6)
SHAPE_MAX_SPAN = 4
GRID_BASE_RANGE = (18, 24)
LOCK_EXTRA_SPACE = 3
LOCK_PADDING = 2
KEY_PADDING = 1
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_shape(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _rotate_shape(
    shape: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    turns: int,
) -> tuple[tuple[int, int], ...]:
    pts = list(shape)
    for _ in range(turns % 4):
        pts = [(c, -r) for r, c in pts]
        pts = list(_normalize_shape(pts))
    return _normalize_shape(pts)


def _shape_dims(shape: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _canonical_shape(shape: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_rotate_shape(shape, k) for k in range(4))


def _match_rotation(
    shape: tuple[tuple[int, int], ...],
    target: tuple[tuple[int, int], ...],
) -> int | None:
    for turns in range(4):
        if _rotate_shape(shape, turns) == target:
            return turns
    return None


def _render_shape(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), BG, dtype=int)
    for r, c in shape:
        patch[r, c] = color
    return patch


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    coords = np.argwhere(patch != BG)
    for dr, dc in coords:
        grid[top + int(dr), left + int(dc)] = int(patch[int(dr), int(dc)])


def _can_place(grid: np.ndarray, patch: np.ndarray, top: int, left: int, padding: int) -> bool:
    ph, pw = patch.shape
    h, w = grid.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    coords = np.argwhere(patch != BG)
    for dr, dc in coords:
        rr = top + int(dr)
        cc = left + int(dc)
        r0 = max(0, rr - padding)
        r1 = min(h, rr + padding + 1)
        c0 = max(0, cc - padding)
        c1 = min(w, cc + padding + 1)
        if np.any(grid[r0:r1, c0:c1] != BG):
            return False
    return True


def _random_polyomino(rng: random.Random, area: int) -> tuple[tuple[int, int], ...] | None:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(tuple(frontier)))

    shape = _normalize_shape(tuple(cells))
    h, w = _shape_dims(shape)
    if h < 2 or w < 2 or h > SHAPE_MAX_SPAN or w > SHAPE_MAX_SPAN:
        return None
    if len({_rotate_shape(shape, k) for k in range(4)}) != 4:
        return None
    return shape


def _sample_shapes(rng: random.Random, count: int) -> list[tuple[tuple[int, int], ...]] | None:
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for _ in range(200):
        if len(shapes) == count:
            return shapes
        area = rng.randint(SHAPE_AREA_RANGE[0], SHAPE_AREA_RANGE[1])
        shape = _random_polyomino(rng, area)
        if shape is None:
            continue
        sig = _canonical_shape(shape)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(shape)
    return None


def _lock_patch(
    rng: random.Random,
    base_shape: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[tuple[int, int], ...]]:
    hole_shape = _rotate_shape(base_shape, rng.randrange(4))
    hole_h, hole_w = _shape_dims(hole_shape)
    inner_h = rng.randint(hole_h + 2, hole_h + 5)
    inner_w = rng.randint(hole_w + 2, hole_w + 5)
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = LOCK
    patch[-1, :] = LOCK
    patch[:, 0] = LOCK
    patch[:, -1] = LOCK

    top = 1 + rng.randint(1, inner_h - hole_h - 1)
    left = 1 + rng.randint(1, inner_w - hole_w - 1)
    hole_cells: list[tuple[int, int]] = []
    for r, c in hole_shape:
        patch[top + r, left + c] = SHADOW
        hole_cells.append((top + r, left + c))
    return patch, hole_cells, hole_shape


def _place_keys(
    rng: random.Random,
    grid: np.ndarray,
    keys: list[tuple[np.ndarray, int]],
) -> bool:
    h, w = grid.shape
    sides = ("top", "bottom", "left", "right")
    for patch, _color in keys:
        ph, pw = patch.shape
        placed = False
        for _ in range(120):
            side = rng.choice(sides)
            if side == "top":
                top = rng.choice((0, 1))
                left = rng.randint(0, w - pw)
            elif side == "bottom":
                top = h - ph - rng.choice((0, 1))
                left = rng.randint(0, w - pw)
            elif side == "left":
                top = rng.randint(0, h - ph)
                left = rng.choice((0, 1))
            else:
                top = rng.randint(0, h - ph)
                left = w - pw - rng.choice((0, 1))
            if _can_place(grid, patch, top, left, KEY_PADDING):
                _paste(grid, patch, top, left)
                placed = True
                break
        if not placed:
            return False
    return True


def _extract_locks(grid: np.ndarray) -> list[dict[str, object]]:
    grey_comps = _components(grid == LOCK)
    maroon_comps = [_normalize_shape(comp) for comp in _components(grid == SHADOW)]
    maroon_raw = _components(grid == SHADOW)
    locks = []
    used_maroon = set()
    for comp in grey_comps:
        r0, c0, r1, c1 = _bbox(comp)
        matches = []
        for idx, hole in enumerate(maroon_raw):
            if idx in used_maroon:
                continue
            if all(r0 < r < r1 and c0 < c < c1 for r, c in hole):
                matches.append(idx)
        assert len(matches) == 1
        idx = matches[0]
        used_maroon.add(idx)
        locks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "hole_cells": maroon_raw[idx],
                "hole_shape": maroon_comps[idx],
            }
        )
    assert len(used_maroon) == len(maroon_raw)
    return locks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the lock shadow with the same shape up to rotation,
    erase the key, and repaint that lock interior region using the key color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    locks = _extract_locks(g)
    key_mask = ~np.isin(g, np.array([BG, LOCK, SHADOW], dtype=int))
    key_comps = _components(key_mask)

    used_locks: set[int] = set()
    for comp in key_comps:
        color_values = {int(g[r, c]) for r, c in comp}
        assert len(color_values) == 1
        color = color_values.pop()
        key_shape = _normalize_shape(comp)

        matches = []
        for idx, lock in enumerate(locks):
            if idx in used_locks:
                continue
            rot = _match_rotation(key_shape, lock["hole_shape"])
            if rot is not None:
                matches.append((idx, rot))
        assert len(matches) == 1

        lock_idx, _rot = matches[0]
        used_locks.add(lock_idx)

        for r, c in comp:
            out[r, c] = BG
        for r, c in locks[lock_idx]["hole_cells"]:
            out[r, c] = color

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    lock_count = rng.randint(LOCK_COUNT_RANGE[0], LOCK_COUNT_RANGE[1])
    shapes = _sample_shapes(rng, lock_count)
    if shapes is None:
        return None

    height = rng.randint(
        GRID_BASE_RANGE[0] + lock_count * LOCK_EXTRA_SPACE,
        GRID_BASE_RANGE[1] + lock_count * LOCK_EXTRA_SPACE,
    )
    width = rng.randint(
        GRID_BASE_RANGE[0] + lock_count * LOCK_EXTRA_SPACE,
        GRID_BASE_RANGE[1] + lock_count * LOCK_EXTRA_SPACE + 2,
    )
    inp = np.full((height, width), BG, dtype=int)
    out = inp.copy()
    key_colors = rng.sample(KEY_COLORS, lock_count)

    key_specs: list[tuple[np.ndarray, int]] = []
    for shape, color in zip(shapes, key_colors):
        lock_patch, local_hole_cells, _hole_shape = _lock_patch(rng, shape)
        ph, pw = lock_patch.shape
        max_top = height - ph - 3
        max_left = width - pw - 3
        if max_top < 3 or max_left < 3:
            return None

        placed = False
        for _ in range(120):
            top = rng.randint(3, max_top)
            left = rng.randint(3, max_left)
            if not _can_place(inp, lock_patch, top, left, LOCK_PADDING):
                continue
            _paste(inp, lock_patch, top, left)
            _paste(out, lock_patch, top, left)
            for dr, dc in local_hole_cells:
                out[top + dr, left + dc] = color
            placed = True
            break
        if not placed:
            return None

        key_shape = _rotate_shape(shape, rng.randrange(4))
        key_specs.append((_render_shape(key_shape, color), color))

    if not _place_keys(rng, inp, key_specs):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-key input")
