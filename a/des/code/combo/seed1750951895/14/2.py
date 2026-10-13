from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FG = 5
BLUE = 1
GREEN = 3
MAGENTA = 6
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _symmetry_kind(mask: np.ndarray) -> str:
    left_right = bool(np.array_equal(mask, mask[:, ::-1]))
    up_down = bool(np.array_equal(mask, mask[::-1, :]))
    diagonal = bool(mask.shape[0] == mask.shape[1] and np.array_equal(mask, mask.T))
    assert sum((left_right, up_down, diagonal)) == 1, "object must have exactly one symmetry type"
    if left_right:
        return "LR"
    if up_down:
        return "UD"
    return "D"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove singleton grey noise and recolor each larger object by its reflection symmetry.

    Left-right mirror symmetry -> blue.
    Up-down mirror symmetry -> green.
    Main-diagonal mirror symmetry -> magenta.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    out = np.full_like(g, BG)
    for coords in _components(g != BG):
        if len(coords) == 1:
            continue
        kind = _symmetry_kind(_mask_from_coords(coords))
        color = {"LR": BLUE, "UD": GREEN, "D": MAGENTA}[kind]
        out[coords[:, 0], coords[:, 1]] = color
    return out


GRID_ROWS = (17, 28)
GRID_COLS = (17, 28)
OBJECT_COUNT_RANGE = (3, 7)
SINGLETON_COUNT_RANGE = (3, 9)
MIN_OBJECT_L1 = 4
MIN_SINGLE_L1_TO_OBJECT = 3
MIN_SINGLE_L1_TO_SINGLE = 2
GEN_RETRIES = 400
MASK_RETRIES = 40
LRUD_HEIGHT_RANGE = range(3, 11)
LRUD_WIDTH_RANGE = range(3, 11)
D_SIZE_RANGE = range(4, 9)
AREA_MIN = 6
AREA_MAX = 28


def _orbit(kind: str, shape: tuple[int, int], r: int, c: int) -> tuple[tuple[int, int], ...]:
    h, w = shape
    if kind == "LR":
        cells = {(r, c), (r, w - 1 - c)}
    elif kind == "UD":
        cells = {(r, c), (h - 1 - r, c)}
    else:
        cells = {(r, c), (c, r)}
    return tuple(sorted(cells))


def _connected(mask: np.ndarray) -> bool:
    parts = _components(mask)
    return len(parts) == 1 and len(parts[0]) == int(mask.sum())


def _initial_mask(kind: str, h: int, w: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    mask = np.zeros((h, w), dtype=bool)
    forbidden: set[tuple[int, int]] = set()
    if kind == "LR":
        axis_cols = [w // 2] if w % 2 else [w // 2 - 1, w // 2]
        mask[:, axis_cols] = True
        row = random.choice([r for r in range(h) if r != h - 1 - r])
        mask[row, :] = True
        mirror_row = h - 1 - row
        forbidden.update(_orbit(kind, mask.shape, mirror_row, 0))
    elif kind == "UD":
        axis_rows = [h // 2] if h % 2 else [h // 2 - 1, h // 2]
        mask[axis_rows, :] = True
        col = random.choice([c for c in range(w) if c != w - 1 - c])
        mask[:, col] = True
        mirror_col = w - 1 - col
        forbidden.update(_orbit(kind, mask.shape, 0, mirror_col))
    else:
        for i in range(h):
            mask[i, i] = True
        for i in range(h - 1):
            mask[i, i + 1] = True
            mask[i + 1, i] = True
    return mask, forbidden


def _dim_options(kind: str) -> list[tuple[int, int]]:
    if kind == "D":
        return [(n, n) for n in D_SIZE_RANGE if 3 * n - 2 <= AREA_MAX]
    options = []
    for h in LRUD_HEIGHT_RANGE:
        for w in LRUD_WIDTH_RANGE:
            if min(h, w) < 3:
                continue
            options.append((h, w))
    return options


def _frontier_orbits(mask: np.ndarray, kind: str, forbidden: set[tuple[int, int]], limit: int) -> list[tuple[tuple[int, int], ...]]:
    h, w = mask.shape
    seen = set()
    choices = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            orbit = _orbit(kind, mask.shape, r, c)
            if orbit in seen or any(cell in forbidden for cell in orbit):
                continue
            seen.add(orbit)
            new_cells = [cell for cell in orbit if not mask[cell]]
            if not new_cells or len(new_cells) > limit:
                continue
            touches = False
            for rr, cc in new_cells:
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        touches = True
                        break
                if touches:
                    break
            if touches:
                choices.append(orbit)
    random.shuffle(choices)
    return choices


def _grow_mask(kind: str) -> np.ndarray | None:
    for _ in range(MASK_RETRIES):
        h, w = random.choice(_dim_options(kind))
        mask, forbidden = _initial_mask(kind, h, w)
        if int(mask.sum()) > AREA_MAX:
            continue
        target = random.randint(max(AREA_MIN, int(mask.sum())), AREA_MAX)
        while int(mask.sum()) < target:
            choices = _frontier_orbits(mask, kind, forbidden, target - int(mask.sum()))
            if not choices:
                break
            for cell in choices[0]:
                mask[cell] = True
        area = int(mask.sum())
        if not (AREA_MIN <= area <= AREA_MAX):
            continue
        if not _connected(mask):
            continue
        try:
            if _symmetry_kind(mask) == kind:
                return mask
        except AssertionError:
            continue
    return None


def _counts() -> tuple[int, int, int]:
    total = random.randint(*OBJECT_COUNT_RANGE)
    base = total // 3
    rem = total % 3
    counts = [base, base, base]
    order = [0, 1, 2]
    random.shuffle(order)
    for i in range(rem):
        counts[order[i]] += 1
    return counts[0], counts[1], counts[2]


def _min_l1(a: np.ndarray, b: np.ndarray) -> int:
    if len(a) == 0 or len(b) == 0:
        return 999
    return int(np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2).min())


def _valid_positions(shape: tuple[int, int], mask: np.ndarray, placed: np.ndarray | None) -> list[tuple[int, int]]:
    rows, cols = shape
    h, w = mask.shape
    local = np.argwhere(mask)
    positions = []
    for r0 in range(rows - h + 1):
        for c0 in range(cols - w + 1):
            coords = local + np.array([r0, c0])
            if placed is not None and _min_l1(coords, placed) < MIN_OBJECT_L1:
                continue
            positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _paint(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask) + np.array([r0, c0])
    grid[coords[:, 0], coords[:, 1]] = color
    return coords


def _scatter_singletons(grid: np.ndarray, multi_coords: np.ndarray, count: int) -> bool:
    singles: list[tuple[int, int]] = []
    h, w = grid.shape
    for _ in range(count):
        single_arr = np.asarray(singles, dtype=int) if singles else np.empty((0, 2), dtype=int)
        choices = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != BG:
                    continue
                cell = np.array([[r, c]])
                if _min_l1(cell, multi_coords) < MIN_SINGLE_L1_TO_OBJECT:
                    continue
                if _min_l1(cell, single_arr) < MIN_SINGLE_L1_TO_SINGLE:
                    continue
                choices.append((r, c))
        if not choices:
            return False
        r, c = random.choice(choices)
        grid[r, c] = FG
        singles.append((r, c))
    return True


def _attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    counts = dict(zip(("LR", "UD", "D"), _counts()))

    objects: list[tuple[str, np.ndarray]] = []
    for kind in ("LR", "UD", "D"):
        for _ in range(counts[kind]):
            mask = _grow_mask(kind)
            if mask is None:
                return None
            objects.append((kind, mask))

    random.shuffle(objects)
    objects.sort(key=lambda item: int(item[1].sum()), reverse=True)
    inp = np.full((rows, cols), BG, dtype=int)
    out = np.full((rows, cols), BG, dtype=int)
    placed_parts = []

    for kind, mask in objects:
        placed = np.concatenate(placed_parts, axis=0) if placed_parts else None
        positions = _valid_positions(inp.shape, mask, placed)
        if not positions:
            return None
        r0, c0 = positions[0]
        coords = _paint(inp, mask, r0, c0, FG)
        _paint(out, mask, r0, c0, {"LR": BLUE, "UD": GREEN, "D": MAGENTA}[kind])
        placed_parts.append(coords)

    multi_coords = np.concatenate(placed_parts, axis=0)
    singleton_count = random.randint(*SINGLETON_COUNT_RANGE)
    if int(multi_coords.shape[0]) + singleton_count >= (rows * cols + 1) // 2:
        return None
    if not _scatter_singletons(inp, multi_coords, singleton_count):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
