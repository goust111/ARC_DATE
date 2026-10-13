from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
HORIZONTAL = "horizontal"
VERTICAL = "vertical"
ROTATION = "rotation"
SYMMETRIES = (HORIZONTAL, VERTICAL, ROTATION)

PALETTE_COLORS = (2, 3, 4)  # red, green, yellow in ARC color ids
PATTERN_COLORS = (1, 5, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

SIZE_VALUES = tuple(range(15, 30, 2))
MIN_TARGET_AREA = 18
MAX_GENERATION_ATTEMPTS = 500


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _transform_cell(cell: tuple[int, int], shape: tuple[int, int], symmetry: str) -> tuple[int, int]:
    r, c = cell
    h, w = shape
    if symmetry == HORIZONTAL:
        return h - 1 - r, c
    if symmetry == VERTICAL:
        return r, w - 1 - c
    if symmetry == ROTATION:
        return h - 1 - r, w - 1 - c
    raise AssertionError(symmetry)


def _orbit(cell: tuple[int, int], shape: tuple[int, int], symmetry: str) -> tuple[tuple[int, int], ...]:
    mate = _transform_cell(cell, shape, symmetry)
    if mate == cell:
        return (cell,)
    return tuple(sorted((cell, mate)))


def _transform_mask(mask: np.ndarray, symmetry: str) -> np.ndarray:
    if symmetry == HORIZONTAL:
        return np.flipud(mask)
    if symmetry == VERTICAL:
        return np.fliplr(mask)
    if symmetry == ROTATION:
        return np.rot90(mask, 2)
    raise AssertionError(symmetry)


def _symmetry_cost(mask: np.ndarray, symmetry: str) -> int:
    mirrored = _transform_mask(mask, symmetry)
    return int(np.count_nonzero(mask != mirrored) // 2)


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _neighbor_count(mask: np.ndarray, cell: tuple[int, int]) -> int:
    r, c = cell
    total = 0
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if _inside(mask.shape, nr, nc) and bool(mask[nr, nc]):
            total += 1
    return total


def _find_palette(grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    assert h >= 3, "expected room for a 3x1 palette"

    specs = [
        ("top_left", [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], (3, 0)),
        ("top_right", [(0, w - 1), (1, w - 1), (2, w - 1)], [(0, w - 2), (1, w - 2), (2, w - 2)], (3, w - 1)),
        ("bottom_left", [(h - 3, 0), (h - 2, 0), (h - 1, 0)], [(h - 3, 1), (h - 2, 1), (h - 1, 1)], (h - 4, 0)),
        (
            "bottom_right",
            [(h - 3, w - 1), (h - 2, w - 1), (h - 1, w - 1)],
            [(h - 3, w - 2), (h - 2, w - 2), (h - 1, w - 2)],
            (h - 4, w - 1),
        ),
    ]

    strict = []
    fallback = []
    for _name, cells, adj, tail in specs:
        vals = [int(grid[r, c]) for r, c in cells]
        if any(v == BG for v in vals) or len(set(vals)) != 3:
            continue
        fallback.append((vals, set(cells)))

        ok = True
        for r, c in adj:
            if _inside(grid.shape, r, c) and int(grid[r, c]) != BG:
                ok = False
                break
        tr, tc = tail
        if ok and _inside(grid.shape, tr, tc) and int(grid[tr, tc]) != BG:
            ok = False
        if ok:
            strict.append((vals, set(cells)))

    candidates = strict or fallback
    assert len(candidates) == 1, "expected exactly one corner palette"
    colors, cells = candidates[0]
    return list(colors), cells


def _pattern_mask_and_color(grid: np.ndarray, palette_cells: set[tuple[int, int]]) -> tuple[np.ndarray, int]:
    keep = np.ones(grid.shape, dtype=bool)
    for r, c in palette_cells:
        keep[r, c] = False

    colors = sorted({int(grid[r, c]) for r, c in zip(*np.where(keep & (grid != BG)), strict=False)})
    assert len(colors) == 1, "expected exactly one non-palette pattern color"
    color = int(colors[0])
    return (grid == color) & keep, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the nearly centered symmetric object and encode the chosen symmetry type with the repair color.

    Input structure:
    - The background is black.
    - One large connected monochrome object is almost symmetric about the grid center under either a horizontal mirror, a vertical mirror, or a 180-degree rotation.
    - A 3x1 corner palette marks the three symmetry types from top to bottom.

    Rule:
    - Remove the corner palette from consideration, then compare the object against all three center symmetries.
    - Pick the symmetry whose pairwise disagreements are fewest; ties are broken by the top-to-bottom palette order.
    - Keep every already-symmetric object pixel in its original color, but for each disagreeing symmetric pair, fill both positions and recolor that repaired pair with the palette color of the chosen symmetry.
    - Erase the palette itself back to black.

    Key insight:
    - The task is not to follow a fixed axis marker; it is to score three candidate center symmetries, choose the closest one, and use the palette only as a visible legend for the repair color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_colors, palette_cells = _find_palette(g)
    pattern_mask, pattern_color = _pattern_mask_and_color(g, palette_cells)

    scored = []
    for idx, symmetry in enumerate(SYMMETRIES):
        mirrored = _transform_mask(pattern_mask, symmetry)
        mismatch = pattern_mask != mirrored
        cost = int(np.count_nonzero(mismatch) // 2)
        scored.append((cost, idx, mirrored, mismatch))

    _cost, palette_idx, mirrored, mismatch = min(scored, key=lambda item: (item[0], item[1]))
    repair_color = int(palette_colors[palette_idx])
    repaired = pattern_mask | mirrored

    out = np.full_like(g, BG)
    out[repaired & ~mismatch] = int(pattern_color)
    out[mismatch] = repair_color
    return out


def _reserve_mask(shape: tuple[int, int], corner: str) -> np.ndarray:
    h, w = shape
    reserve = np.zeros((h, w), dtype=bool)
    if corner == "top_left":
        reserve[:4, :3] = True
    elif corner == "top_right":
        reserve[:4, w - 3 :] = True
    elif corner == "bottom_left":
        reserve[h - 4 :, :3] = True
    elif corner == "bottom_right":
        reserve[h - 4 :, w - 3 :] = True
    else:
        raise AssertionError(corner)
    return reserve


def _place_palette(grid: np.ndarray, corner: str) -> None:
    h, w = grid.shape
    if corner == "top_left":
        rows = [0, 1, 2]
        col = 0
    elif corner == "top_right":
        rows = [0, 1, 2]
        col = w - 1
    elif corner == "bottom_left":
        rows = [h - 3, h - 2, h - 1]
        col = 0
    elif corner == "bottom_right":
        rows = [h - 3, h - 2, h - 1]
        col = w - 1
    else:
        raise AssertionError(corner)

    for row, color in zip(rows, PALETTE_COLORS, strict=True):
        grid[row, col] = int(color)


def _initial_mask(shape: tuple[int, int], symmetry: str, forbidden: np.ndarray) -> np.ndarray | None:
    h, w = shape
    cr, cc = h // 2, w // 2
    mask = np.zeros(shape, dtype=bool)

    if symmetry == HORIZONTAL:
        span = random.randint(1, min(4, w - 4))
        start = random.randint(2, w - span - 2)
        if bool(forbidden[cr, start : start + span].any()):
            return None
        mask[cr, start : start + span] = True
        return mask

    if symmetry == VERTICAL:
        span = random.randint(1, min(4, h - 4))
        start = random.randint(2, h - span - 2)
        if bool(forbidden[start : start + span, cc].any()):
            return None
        mask[start : start + span, cc] = True
        return mask

    if symmetry == ROTATION:
        if bool(forbidden[cr, cc]):
            return None
        mask[cr, cc] = True
        return mask

    raise AssertionError(symmetry)


def _frontier_orbits(mask: np.ndarray, symmetry: str, forbidden: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    frontier: list[tuple[tuple[int, int], ...]] = []
    for r, c in map(tuple, np.argwhere(mask)):
        r = int(r)
        c = int(c)
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not _inside(mask.shape, nr, nc) or bool(mask[nr, nc]) or bool(forbidden[nr, nc]):
                continue
            orb = _orbit((nr, nc), mask.shape, symmetry)
            if orb in seen:
                continue
            if any(bool(forbidden[rr, cc]) for rr, cc in orb):
                continue
            if any(bool(mask[rr, cc]) for rr, cc in orb):
                continue
            seen.add(orb)
            frontier.append(orb)
    return frontier


def _grow_base_mask(shape: tuple[int, int], symmetry: str, forbidden: np.ndarray) -> np.ndarray | None:
    mask = _initial_mask(shape, symmetry, forbidden)
    if mask is None:
        return None

    area_cap = min(90, max(MIN_TARGET_AREA, (shape[0] * shape[1]) // 4))
    target = random.randint(MIN_TARGET_AREA, area_cap)

    for _ in range(800):
        if int(mask.sum()) >= target:
            break
        frontier = _frontier_orbits(mask, symmetry, forbidden)
        if not frontier:
            break
        orb = random.choice(frontier)
        for r, c in orb:
            mask[r, c] = True

    if int(mask.sum()) < MIN_TARGET_AREA:
        return None
    if not _connected(mask):
        return None
    return mask


def _disturbance_candidates(
    mask: np.ndarray, symmetry: str, forbidden: np.ndarray
) -> tuple[list[tuple[tuple[int, int], list[tuple[int, int]]]], list[tuple[tuple[int, int], list[tuple[int, int]]]]]:
    removable: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    addable: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    h, w = mask.shape

    for r in range(h):
        for c in range(w):
            orb = _orbit((r, c), mask.shape, symmetry)
            if len(orb) != 2:
                continue

            if all(bool(mask[rr, cc]) for rr, cc in orb):
                if _neighbor_count(mask, (r, c)) >= 2:
                    removable.setdefault(orb, []).append((r, c))
            elif all(not bool(mask[rr, cc]) for rr, cc in orb) and not any(bool(forbidden[rr, cc]) for rr, cc in orb):
                if _neighbor_count(mask, (r, c)) >= 1:
                    addable.setdefault(orb, []).append((r, c))

    rem = [(orb, cells) for orb, cells in removable.items() if cells]
    add = [(orb, cells) for orb, cells in addable.items() if cells]
    return rem, add


def _sample_disturbance(mask: np.ndarray, symmetry: str, forbidden: np.ndarray) -> np.ndarray | None:
    removable, addable = _disturbance_candidates(mask, symmetry, forbidden)
    if not removable:
        return None

    mode = random.choice(("missing", "extra", "mixed"))
    if mode == "missing":
        miss_count = random.randint(1, min(4, len(removable)))
        add_count = 0
    elif mode == "extra":
        if not addable:
            return None
        miss_count = 0
        add_count = random.randint(1, min(3, len(addable)))
    else:
        if not addable:
            return None
        miss_count = random.randint(1, min(3, len(removable)))
        add_count = random.randint(1, min(2, len(addable)))

    disturbed = mask.copy()

    for orb, cells in random.sample(removable, miss_count):
        cell = random.choice(cells)
        disturbed[cell] = False

    for orb, cells in random.sample(addable, add_count):
        cell = max(cells, key=lambda rc: _neighbor_count(mask, rc))
        disturbed[cell] = True

    if not _connected(disturbed):
        return None
    return disturbed


def _render_input(mask: np.ndarray, color: int, corner: str) -> np.ndarray:
    grid = np.full(mask.shape, BG, dtype=int)
    grid[mask] = int(color)
    _place_palette(grid, corner)
    return grid


def _valid_instance(mask: np.ndarray, symmetry: str) -> bool:
    costs = [_symmetry_cost(mask, sym) for sym in SYMMETRIES]
    idx = SYMMETRIES.index(symmetry)
    best = min(costs)
    if costs[idx] != best or costs.count(best) != 1:
        return False
    return costs[idx] >= 1


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(SIZE_VALUES)
    w = random.choice(SIZE_VALUES)
    symmetry = random.choice(SYMMETRIES)
    corner = random.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
    color = random.choice(PATTERN_COLORS)

    forbidden = _reserve_mask((h, w), corner)
    base = _grow_base_mask((h, w), symmetry, forbidden)
    if base is None:
        return None

    # Reject accidental alternative symmetries before injecting noise.
    if any(_symmetry_cost(base, sym) == 0 for sym in SYMMETRIES if sym != symmetry):
        return None

    disturbed = _sample_disturbance(base, symmetry, forbidden)
    if disturbed is None or not _valid_instance(disturbed, symmetry):
        return None

    grid = _render_input(disturbed, color, corner)
    out = main(grid)
    repair_color = PALETTE_COLORS[SYMMETRIES.index(symmetry)]

    if np.array_equal(grid, out):
        return None
    if int(np.count_nonzero(out == repair_color)) == 0:
        return None
    if int(np.count_nonzero(out == color)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-repair input")
