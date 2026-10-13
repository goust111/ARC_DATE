from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BG = 0
BLUE = 1
RED = 2
MASK = 3
YELLOW = 4
ORANGE = 7

UNKNOWN = -1

TILES = (
    (((0, 0), (0, 1), (1, 0), (1, 1)), BLUE),
    (((0, 1), (1, 0), (1, 1)), RED),
    (((0, 0), (0, 1), (1, 1)), YELLOW),
    (((0, 0), (1, 0), (1, 1)), ORANGE),
)

BASE_MOTIF = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=bool,
)

MASK_SIZES = ((3, 4), (3, 5), (4, 4), (4, 5))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _c4_orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    pts: list[tuple[int, int]] = []
    rr, cc = int(r), int(c)
    for _ in range(4):
        pts.append((rr, cc))
        rr, cc = cc, n - 1 - rr
    return tuple(dict.fromkeys(pts))


def _mask_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == MASK
    b = _bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1])
    assert int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1)
    return r0, c0, r1, c1


def _render_cover(shape: tuple[int, int], placements: list[tuple[tuple[tuple[int, int], ...], int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for cells, color in placements:
        for r, c in cells:
            out[r, c] = color
    return out


def _unique_exact_cover(
    fg_mask: np.ndarray,
    fixed_colors: np.ndarray | None = None,
    limit: int = 2,
) -> list[tuple[tuple[tuple[int, int], ...], int]] | None:
    h, w = fg_mask.shape
    fg_cells = frozenset((int(r), int(c)) for r, c in np.argwhere(fg_mask))
    if not fg_cells:
        return []

    if fixed_colors is None:
        fixed_colors = np.full((h, w), UNKNOWN, dtype=int)

    candidates: list[tuple[frozenset[tuple[int, int]], int]] = []
    by_cell: dict[tuple[int, int], list[int]] = {cell: [] for cell in fg_cells}

    for r0 in range(h - 1):
        for c0 in range(w - 1):
            for rels, color in TILES:
                cells = frozenset((r0 + dr, c0 + dc) for dr, dc in rels)
                if not cells <= fg_cells:
                    continue
                if any(int(fixed_colors[r, c]) not in (UNKNOWN, color) for r, c in cells):
                    continue
                idx = len(candidates)
                candidates.append((cells, color))
                for cell in cells:
                    by_cell[cell].append(idx)

    if any(not by_cell[cell] for cell in fg_cells):
        return None

    solutions: list[list[int]] = []

    def search(remaining: frozenset[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) >= limit:
            return
        if not remaining:
            solutions.append(chosen.copy())
            return

        cell = min(
            remaining,
            key=lambda p: sum(1 for idx in by_cell[p] if candidates[idx][0] <= remaining),
        )
        options = [idx for idx in by_cell[cell] if candidates[idx][0] <= remaining]
        options.sort(key=lambda idx: (len(candidates[idx][0]), candidates[idx][0]))
        for idx in options:
            search(remaining - candidates[idx][0], chosen + [idx])

    search(fg_cells, [])
    if len(solutions) != 1:
        return None

    answer: list[tuple[tuple[tuple[int, int], ...], int]] = []
    for idx in solutions[0]:
        cells, color = candidates[idx]
        answer.append((tuple(sorted(cells)), int(color)))
    return answer


def _symmetry_fg_and_hints(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = grid.shape[0]
    mask = grid == MASK
    fg_mask = (grid != BG) & ~mask
    color_hints = np.full(grid.shape, UNKNOWN, dtype=int)
    color_hints[fg_mask] = grid[fg_mask]

    for r, c in np.argwhere(mask):
        orbit = _c4_orbit(int(r), int(c), n)
        visible_colors = [
            int(grid[rr, cc])
            for rr, cc in orbit
            if not mask[rr, cc] and int(grid[rr, cc]) not in (BG, MASK)
        ]
        if visible_colors:
            fg_mask[r, c] = True
            unique = set(visible_colors)
            if len(unique) == 1:
                color_hints[r, c] = unique.pop()
        else:
            fg_mask[r, c] = False

    return fg_mask, color_hints


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the green rectangle by combining C4 foreground symmetry with exact-cover tiling colors.

    Input structure:
    - The grid is a square odd-sized canvas with black background and one solid green rectangle.
    - Outside the green rectangle, the non-black cells belong to a foreground mask that is C4-symmetric about the grid center.
    - The foreground admits an exact cover by 2x2 squares and three allowed 2x2 L-tromino orientations, each shape carrying its own color.

    Rule:
    - Return only the green rectangle's bounding box.
    - First use C4 symmetry of foreground occupancy to decide which masked cells are foreground and which are background; if an occupied orbit shows a single visible color, keep that as a color hint.
    - Then solve the unique exact cover of the full foreground mask, constrained by every visible colored cell and the symmetry-derived hints, and emit the recovered colors inside the rectangle.

    Key insight:
    - Symmetry identifies the hidden foreground support, but not every hidden color because different tile shapes can occupy the same C4 orbit; the unique local tiling resolves those remaining ambiguities.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 1

    r0, c0, r1, c1 = _mask_bbox(g)
    fg_mask, color_hints = _symmetry_fg_and_hints(g)
    placements = _unique_exact_cover(fg_mask, color_hints)
    assert placements is not None

    full = _render_cover(g.shape, placements)
    visible = g != MASK
    assert np.array_equal(full[visible], np.where(g[visible] == BG, BG, g[visible]))
    return full[r0 : r1 + 1, c0 : c1 + 1]


def _base_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    offset = (n - BASE_MOTIF.shape[0]) // 2
    mask[offset : offset + 7, offset : offset + 7] = BASE_MOTIF
    return mask


def _square_group_cells(anchor: tuple[int, int], n: int) -> frozenset[tuple[int, int]] | None:
    cells: set[tuple[int, int]] = set()
    cur = anchor
    anchors: list[tuple[int, int]] = []
    for _ in range(4):
        anchors.append(cur)
        cur = (cur[1], n - 2 - cur[0])
    if len(set(anchors)) != 4:
        return None
    for r0, c0 in anchors:
        if r0 < 0 or c0 < 0 or r0 + 1 >= n or c0 + 1 >= n:
            return None
        cells.update({(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)})
    if len(cells) != 16:
        return None
    return frozenset(cells)


@lru_cache(maxsize=None)
def _square_group_options(n: int) -> tuple[frozenset[tuple[int, int]], ...]:
    base = _base_mask(n)
    seen: set[tuple[tuple[int, int], ...]] = set()
    groups: list[frozenset[tuple[int, int]]] = []
    for r0 in range(n - 1):
        for c0 in range(n - 1):
            cells = _square_group_cells((r0, c0), n)
            if cells is None:
                continue
            if any(base[r, c] for r, c in cells):
                continue
            key = tuple(sorted(cells))
            if key in seen:
                continue
            seen.add(key)
            groups.append(cells)
    return tuple(groups)


def _build_pattern(n: int) -> np.ndarray | None:
    base = _base_mask(n)
    groups = list(_square_group_options(n))
    if len(groups) < 2:
        return None

    for _ in range(120):
        g1, g2 = random.sample(groups, 2)
        if g1 & g2:
            continue
        fg_mask = base.copy()
        for r, c in g1 | g2:
            fg_mask[r, c] = True
        placements = _unique_exact_cover(fg_mask)
        if placements is None:
            continue
        return _render_cover((n, n), placements)
    return None


def _direct_color_set(full: np.ndarray, rect_mask: np.ndarray, r: int, c: int) -> set[int]:
    orbit = _c4_orbit(r, c, full.shape[0])
    return {
        int(full[rr, cc])
        for rr, cc in orbit
        if not rect_mask[rr, cc] and int(full[rr, cc]) != BG
    }


def _candidate_masks(full: np.ndarray) -> list[tuple[int, int, int, int]]:
    n = full.shape[0]
    center = n // 2
    candidates: list[tuple[int, int, int, int]] = []
    fg = full != BG

    for h, w in MASK_SIZES:
        for r0 in range(n - h + 1):
            for c0 in range(n - w + 1):
                if not (r0 + h - 1 < center or r0 > center):
                    continue
                if not (c0 + w - 1 < center or c0 > center):
                    continue

                rect_mask = np.zeros_like(fg, dtype=bool)
                rect_mask[r0 : r0 + h, c0 : c0 + w] = True
                crop_fg = fg[r0 : r0 + h, c0 : c0 + w]
                if not crop_fg.any() or bool(np.all(crop_fg)):
                    continue

                direct = 0
                ambiguous = 0
                for r in range(r0, r0 + h):
                    for c in range(c0, c0 + w):
                        if not fg[r, c]:
                            continue
                        colors = _direct_color_set(full, rect_mask, r, c)
                        if not colors:
                            direct = -999
                            break
                        if len(colors) == 1:
                            direct += 1
                        else:
                            ambiguous += 1
                    if direct < 0:
                        break
                if direct <= 0 or ambiguous <= 0:
                    continue
                candidates.append((r0, c0, h, w))

    return candidates


def _valid_input_from_full(full: np.ndarray) -> np.ndarray | None:
    candidates = _candidate_masks(full)
    random.shuffle(candidates)
    for r0, c0, h, w in candidates:
        inp = full.copy()
        inp[r0 : r0 + h, c0 : c0 + w] = MASK
        expected = full[r0 : r0 + h, c0 : c0 + w]
        try:
            got = main(inp)
        except AssertionError:
            continue
        if np.array_equal(got, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(200):
        n = random.choice((11, 13, 15))
        full = _build_pattern(n)
        if full is None:
            continue
        inp = _valid_input_from_full(full)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid symmetry-and-tiling example")
