from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

_RNG = np.random.default_rng()
_SOURCE_COLORS = np.array([BLUE, GREEN, YELLOW, GREY], dtype=int)
_MAX_ATTEMPTS = 400
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _choose_interval(prev_top: int, prev_bottom: int, height: int) -> tuple[int, int]:
    top_low = max(0, prev_top - 2)
    top_high = min(height - 2, prev_bottom)
    top = int(_RNG.integers(top_low, top_high + 1))
    bottom_low = max(top + 1, prev_top)
    bottom_high = min(height - 1, prev_bottom + 2)
    bottom = int(_RNG.integers(bottom_low, bottom_high + 1))
    return top, bottom


def _build_full_pattern(height: int, width: int) -> np.ndarray | None:
    axis = width // 2
    span = int(_RNG.integers(2, min(5, axis + 1)))
    patch_h = int(_RNG.integers(5, min(9, height - 2)))
    top_max = height - patch_h - 1
    if top_max < 4:
        return None
    top0 = int(_RNG.integers(4, top_max + 1))

    local_w = span + 1
    local = np.zeros((patch_h, local_w), dtype=bool)
    tops = np.zeros(local_w, dtype=int)
    bottoms = np.zeros(local_w, dtype=int)

    tops[-1] = int(_RNG.integers(0, min(3, patch_h - 1)))
    bottoms[-1] = int(_RNG.integers(max(tops[-1] + 2, patch_h - 3), patch_h))
    for idx in range(local_w - 2, -1, -1):
        tops[idx], bottoms[idx] = _choose_interval(int(tops[idx + 1]), int(bottoms[idx + 1]), patch_h)

    for idx in range(local_w):
        local[tops[idx] : bottoms[idx] + 1, idx] = True

    grid = np.zeros((height, width), dtype=bool)
    left_c0 = axis - span
    for lr, lc in np.argwhere(local):
        c = left_c0 + int(lc)
        r = top0 + int(lr)
        grid[r, c] = True
        grid[r, width - 1 - c] = True

    if grid.sum() < 20:
        return None
    if len(np.unique(np.argwhere(grid)[:, 1])) < 5:
        return None
    return grid


def _top_rows(mask: np.ndarray) -> dict[int, int]:
    cols = np.where(mask.any(axis=0))[0]
    return {int(c): int(np.flatnonzero(mask[:, c])[0]) for c in cols}


def _grow_hole_component(
    full_mask: np.ndarray,
    start: tuple[int, int],
    blocked: set[tuple[int, int]],
    right_min_col: int,
) -> set[tuple[int, int]] | None:
    target_size = int(_RNG.integers(1, 4))
    component = {start}
    for _ in range(target_size - 1):
        frontier = []
        for r, c in sorted(component):
            for dr, dc in _DIRS4:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < full_mask.shape[0]
                    and 0 <= nc < full_mask.shape[1]
                    and full_mask[nr, nc]
                    and nc >= right_min_col
                    and (nr, nc) not in component
                    and (nr, nc) not in blocked
                ):
                    frontier.append((nr, nc))
        if not frontier:
            break
        component.add(frontier[int(_RNG.integers(0, len(frontier)))])
    return component if 1 <= len(component) <= 3 else None


def _make_holes(full_mask: np.ndarray) -> np.ndarray | None:
    height, width = full_mask.shape
    axis = width // 2
    right_min_col = axis + 1
    right_mask = full_mask.copy()
    right_mask[:, :right_min_col] = False
    if not right_mask.any():
        return None

    top_by_col = _top_rows(right_mask)
    if not top_by_col:
        return None

    hole_mask = np.zeros_like(full_mask, dtype=bool)
    blocked: set[tuple[int, int]] = set()
    count = int(_RNG.integers(1, 4))

    mandatory_col = int(_RNG.choice(np.array(sorted(top_by_col), dtype=int)))
    first_seed = (top_by_col[mandatory_col], mandatory_col)
    first = _grow_hole_component(full_mask, first_seed, blocked, right_min_col)
    if first is None:
        return None
    for cell in first:
        hole_mask[cell] = True
    for r, c in first:
        blocked.add((r, c))
        for dr, dc in _DIRS4:
            blocked.add((r + dr, c + dc))

    for _ in range(count - 1):
        candidates = [
            (int(r), int(c))
            for r, c in np.argwhere(right_mask & ~hole_mask)
            if (int(r), int(c)) not in blocked
        ]
        if not candidates:
            return None
        seed = candidates[int(_RNG.integers(0, len(candidates)))]
        comp = _grow_hole_component(full_mask, seed, blocked, right_min_col)
        if comp is None:
            return None
        for cell in comp:
            hole_mask[cell] = True
        for r, c in comp:
            blocked.add((r, c))
            for dr, dc in _DIRS4:
                blocked.add((r + dr, c + dc))

    comps = _components(hole_mask)
    if not (1 <= len(comps) <= 3):
        return None
    if any(not (1 <= len(comp) <= 3) for comp in comps):
        return None
    return hole_mask


def _pick_spread_columns(candidates: list[int], need: int, chosen: list[int]) -> list[int] | None:
    remaining = [c for c in candidates if c not in chosen]
    _RNG.shuffle(remaining)
    for min_gap in (2, 1, 0):
        result = chosen[:]
        for c in remaining:
            if all(abs(c - old) > min_gap for old in result):
                result.append(int(c))
            if len(result) == need:
                return result
    return None


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(15, 20))
    width = int(_RNG.choice(np.array([13, 15, 17, 19], dtype=int)))

    full_mask = _build_full_pattern(height, width)
    if full_mask is None:
        return None
    hole_mask = _make_holes(full_mask)
    if hole_mask is None:
        return None

    visible_mask = full_mask & ~hole_mask
    full_tops = _top_rows(full_mask)
    visible_tops = _top_rows(visible_mask)

    hidden_cols = [
        c for c, top in full_tops.items()
        if c in visible_tops and visible_tops[c] > top
    ]
    hidden_cols += [c for c in full_tops if c not in visible_tops]
    hidden_cols = sorted(set(hidden_cols))
    visible_cols = sorted([
        c for c, top in full_tops.items()
        if c in visible_tops and visible_tops[c] == top
    ])
    if not hidden_cols or not visible_cols:
        return None

    n_sources = int(_RNG.integers(2, 5))
    picked = [int(_RNG.choice(np.array(hidden_cols, dtype=int))), int(_RNG.choice(np.array(visible_cols, dtype=int)))]
    all_cols = sorted(full_tops)
    picked = _pick_spread_columns(all_cols, n_sources, picked)
    if picked is None:
        return None

    source_cols = picked[:n_sources]
    _RNG.shuffle(source_cols)
    colors = _RNG.choice(_SOURCE_COLORS, size=n_sources, replace=False)

    inp = np.zeros((height, width), dtype=int)
    inp[visible_mask] = RED
    for col, color in zip(source_cols, colors):
        inp[0, int(col)] = int(color)

    out = main(inp)

    used_hidden = any(full_tops[c] < visible_tops.get(c, height + 1) for c in source_cols)
    used_visible = any(c in visible_tops and full_tops[c] == visible_tops[c] for c in source_cols)
    if not (used_hidden and used_visible):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.all(inp[hole_mask] == BLACK):
        return None
    if not np.all(out[hole_mask] == BLACK):
        return None
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer erased right-half mirror pixels as invisible walls, then stop top-row rays at those walls without drawing the recovered red cells.

    Input structure:
    - The background is black.
    - A red pattern sits in the middle-lower area and is left-right symmetric across the grid's vertical axis, except that some right-half red pixels have been erased to black.
    - The top row contains several isolated non-red source pixels in distinct columns.

    Rule:
    - Reconstruct the full red wall mask by reflecting every visible red pixel across the vertical axis and taking the union with the visible red pixels.
    - For each top-row source pixel, extend a vertical ray downward through black cells only.
    - Stop the ray immediately before the first blocking cell in that column, where a blocker is either a visible non-background pixel or one of the inferred mirrored red wall pixels.
    - Keep the original grid content unchanged; do not draw the inferred red pixels themselves.

    Key insight:
    - The missing right-half reds are not output objects; they are hidden obstacles inferred from mirror symmetry, and only those invisible obstacles determine where some rays must stop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    visible_red = g == RED
    inferred_red = visible_red | np.fliplr(visible_red)
    blockers = inferred_red | (g != BLACK)
    blockers[0, :] = False

    out = g.copy()
    for c in np.flatnonzero(g[0] != BLACK):
        color = int(g[0, c])
        for r in range(1, g.shape[0]):
            if blockers[r, c]:
                break
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
