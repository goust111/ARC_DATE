from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

INPUT_COLORS = (4, 5, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MASK_ATTEMPTS = 120
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        queue = deque([(rr, cc)])
        coords: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            coords.append((cr, cc))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        parts.append(coords)
    return parts


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _normalized_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start_r, start_c = map(int, coords[0])
    queue = deque([(start_r, start_c)])
    seen[start_r, start_c] = True
    count = 0
    while queue:
        r, c = queue.popleft()
        count += 1
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return count == int(mask.sum())


def _classify_mask(mask: np.ndarray) -> str:
    lr = np.array_equal(mask, np.fliplr(mask))
    ud = np.array_equal(mask, np.flipud(mask))
    rot = np.array_equal(mask, np.rot90(mask, 2))
    if lr and not ud:
        return "lr"
    if ud and not lr:
        return "ud"
    if rot and not lr and not ud:
        return "rot"
    raise AssertionError("object must have exactly one allowed symmetry")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each monochrome object by its unique symmetry type.

    Left-right reflection symmetry becomes blue, up-down reflection symmetry
    becomes red, and 180-degree rotational symmetry without reflection becomes
    green. Object positions stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    parts = _components(g != bg)
    assert parts, "expected at least one foreground object"

    out = g.copy()
    color_map = {"lr": BLUE, "ud": RED, "rot": GREEN}
    for coords in parts:
        kind = _classify_mask(_normalized_mask(coords))
        out[tuple(np.array(coords).T)] = color_map[kind]
    return out


def _grow_connected(
    shape: tuple[int, int],
    seeds: set[tuple[int, int]],
    target_size: int,
    *,
    allowed: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    h, w = shape
    cells = set(seeds)
    if allowed is not None and not cells <= allowed:
        return None

    while len(cells) < target_size:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                nxt = (nr, nc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if nxt in cells:
                    continue
                if allowed is not None and nxt not in allowed:
                    continue
                frontier.append(nxt)
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _mask_from_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_lr_mask() -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        h = random.randint(3, 7)
        w = random.randint(3, 8)
        rep_w = (w + 1) // 2
        bridge_col = rep_w - 1
        seed = {(random.randrange(h), bridge_col)}
        rep_area = h * rep_w
        target = random.randint(max(3, rep_w), min(rep_area, rep_w + h + 3))
        rep = _grow_connected((h, rep_w), seed, target)
        if rep is None:
            continue
        if all(c == bridge_col for _, c in rep):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in rep:
            mask[r, c] = True
            mask[r, w - 1 - c] = True
        mask = _trim(mask)
        if mask.sum() < 4 or not _is_connected(mask):
            continue
        try:
            if _classify_mask(mask) == "lr":
                return mask
        except AssertionError:
            pass
    raise RuntimeError("failed to build left-right symmetric mask")


def _random_ud_mask() -> np.ndarray:
    return _random_lr_mask().T


def _random_rot_mask() -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        h = random.choice((3, 5, 7))
        w = random.choice((3, 5, 7))
        cr = h // 2
        cc = w // 2
        allowed = {
            (r, c)
            for r in range(h)
            for c in range(w)
            if r < cr or (r == cr and c <= cc)
        }
        max_target = min(len(allowed), h + w - 1)
        target = random.randint(3, max_target)
        rep = _grow_connected((h, w), {(cr, cc)}, target, allowed=allowed)
        if rep is None:
            continue
        if all(r == cr for r, _ in rep) or all(c == cc for _, c in rep):
            continue

        cells = set(rep)
        for r, c in tuple(rep):
            cells.add((h - 1 - r, w - 1 - c))
        mask = _trim(_mask_from_cells((h, w), cells))
        if mask.sum() < 5 or not _is_connected(mask):
            continue
        try:
            if _classify_mask(mask) == "rot":
                return mask
        except AssertionError:
            pass
    raise RuntimeError("failed to build rotational mask")


def _random_mask(kind: str) -> np.ndarray:
    if kind == "lr":
        return _random_lr_mask()
    if kind == "ud":
        return _random_ud_mask()
    if kind == "rot":
        return _random_rot_mask()
    raise ValueError(kind)


def _translated_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    rows, cols = np.nonzero(mask)
    return [(top + int(r), left + int(c)) for r, c in zip(rows, cols)]


def _mark_forbidden(forbidden: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = forbidden.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True


def _place_masks(
    shape: tuple[int, int],
    masks: list[np.ndarray],
) -> list[list[tuple[int, int]]] | None:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    placed: list[list[tuple[int, int]]] = []
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)
    slots: dict[int, list[tuple[int, int]]] = {}

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        candidates = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
        random.shuffle(candidates)
        here = None
        for top, left in candidates:
            cells = _translated_cells(mask, top, left)
            if all(not forbidden[r, c] for r, c in cells):
                here = cells
                _mark_forbidden(forbidden, cells)
                break
        if here is None:
            return None
        slots[idx] = here

    for idx in range(len(masks)):
        placed.append(slots[idx])
    return placed


def _sample_kinds() -> list[str]:
    count = random.randint(3, 6)
    kinds = ["lr", "ud", "rot"]
    for _ in range(count - 3):
        kinds.append(random.choice(("lr", "ud", "rot")))
    random.shuffle(kinds)
    return kinds


def _valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    bg = _background_color(grid)
    colors = set(map(int, grid[grid != bg].flat))
    if len(colors) != 1:
        return False

    seen = Counter()
    for coords in _components(grid != bg):
        try:
            seen[_classify_mask(_normalized_mask(coords))] += 1
        except AssertionError:
            return False
    if not all(seen[k] >= 1 for k in ("lr", "ud", "rot")):
        return False

    out_colors = set(map(int, out[grid != bg].flat))
    return {BLUE, RED, GREEN} <= out_colors


def _gen_attempt() -> np.ndarray | None:
    kinds = _sample_kinds()
    try:
        masks = [_random_mask(kind) for kind in kinds]
    except RuntimeError:
        return None

    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    grid_h = random.randint(max(14, max_h + 5), 24)
    grid_w = random.randint(max(14, max_w + 5), 24)

    placements = _place_masks((grid_h, grid_w), masks)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    color = random.choice(INPUT_COLORS)
    for cells in placements:
        for r, c in cells:
            grid[r, c] = color

    if not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
