from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PALETTE = (BLUE, RED, GREEN, YELLOW)
MAX_GENERATION_ATTEMPTS = 800


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _outside_mask(grid: np.ndarray) -> np.ndarray:
    """Flood all non-grey cells reachable from the border."""
    h, w = grid.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def seed(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and not outside[r, c] and grid[r, c] != GREY:
            outside[r, c] = True
            q.append((r, c))

    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    for c in range(w):
        seed(0, c)
        seed(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not outside[nr, nc] and grid[nr, nc] != GREY:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _hole_order_is_spatial(holes: list[np.ndarray]) -> bool:
    """Reject layouts whose area rank can be read off by a simple axis order."""
    n = len(holes)
    centers = [coords.mean(axis=0) for coords in holes]
    area_order = sorted(range(n), key=lambda i: (-int(holes[i].shape[0]), float(centers[i][0]), float(centers[i][1])))
    simple_orders = [
        sorted(range(n), key=lambda i: (float(centers[i][0]), float(centers[i][1]))),
        sorted(range(n), key=lambda i: (float(centers[i][1]), float(centers[i][0]))),
        sorted(range(n), key=lambda i: (float(centers[i][0] + centers[i][1]), float(centers[i][0]))),
        sorted(range(n), key=lambda i: (float(centers[i][0] - centers[i][1]), float(centers[i][0]))),
    ]
    return any(area_order == order or area_order == order[::-1] for order in simple_orders)


def _fill_component(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed black holes by pairing hole-area rank with outside color-count rank.

    Input structure:
    - A black background contains one connected grey boundary structure in the middle.
    - The grey structure separates the canvas into an outside region and several enclosed black 4-connected holes.
    - Outside the grey boundary, 3-4 foreground colors appear only as scattered isolated pixels, with distinct counts.

    Rule:
    - Flood from the border through every non-grey cell to identify the outside region.
    - Count the outside pixels of each non-black foreground color and sort the colors by count from largest to smallest.
    - Find every enclosed black 4-connected hole inside the grey boundary and sort the holes by area from largest to smallest.
    - Fill the largest hole with the most frequent outside color, the next hole with the next most frequent color, and so on.
    - Keep the grey boundary unchanged and erase the outside scattered pixels back to black.

    Key insight:
    - The color-to-hole correspondence is purely ordinal: one ranking comes from outside color counts, the other from enclosed hole areas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    outside = _outside_mask(g)
    hole_mask = (g == BLACK) & (~outside)
    holes = _components4(hole_mask)
    assert holes, "expected at least one enclosed hole"

    colors = [int(c) for c in np.unique(g[outside]) if int(c) not in (BLACK, GREY)]
    assert len(colors) == len(holes), "outside color count must match hole count"

    color_counts = [(int(np.count_nonzero((g == color) & outside)), color) for color in colors]
    assert len({count for count, _color in color_counts}) == len(color_counts), "outside counts must be distinct"
    assert len({int(hole.shape[0]) for hole in holes}) == len(holes), "hole areas must be distinct"

    ranked_colors = [color for _count, color in sorted(color_counts, key=lambda item: item[0], reverse=True)]
    ranked_holes = sorted(holes, key=lambda coords: int(coords.shape[0]), reverse=True)

    out = g.copy()
    out[outside & (g != BLACK) & (g != GREY)] = BLACK
    for color, hole in zip(ranked_colors, ranked_holes):
        _fill_component(out, hole, color)
    return out


def _make_cross4() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    for _ in range(200):
        wl, wr = random.randint(1, 4), random.randint(1, 4)
        ht, hb = random.randint(1, 3), random.randint(1, 3)
        areas = [wl * ht, wr * ht, wl * hb, wr * hb]
        if min(areas) < 2 or max(areas) > 8 or len(set(areas)) != 4:
            continue

        h = ht + hb + 3
        w = wl + wr + 3
        grey = np.zeros((h, w), dtype=int)
        grey[0, :] = GREY
        grey[-1, :] = GREY
        grey[:, 0] = GREY
        grey[:, -1] = GREY
        grey[:, wl + 1] = GREY
        grey[ht + 1, :] = GREY
        holes = [
            (1, 1, ht, wl),
            (1, wl + 2, ht, wr),
            (ht + 2, 1, hb, wl),
            (ht + 2, wl + 2, hb, wr),
        ]
        return grey, holes
    return None


def _make_vsplit3(split_left: bool) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    for _ in range(200):
        wl, wr = random.randint(1, 3), random.randint(1, 3)
        ht, hb = random.randint(1, 2), random.randint(1, 2)
        if split_left:
            areas = [wl * ht, wl * hb, wr * (ht + hb + 1)]
        else:
            areas = [wr * ht, wr * hb, wl * (ht + hb + 1)]
        if min(areas) < 2 or max(areas) > 8 or len(set(areas)) != 3:
            continue

        h = ht + hb + 3
        w = wl + wr + 3
        grey = np.zeros((h, w), dtype=int)
        grey[0, :] = GREY
        grey[-1, :] = GREY
        grey[:, 0] = GREY
        grey[:, -1] = GREY
        split_c = wl + 1
        split_r = ht + 1
        grey[:, split_c] = GREY
        if split_left:
            grey[split_r, : split_c + 1] = GREY
            holes = [
                (1, 1, ht, wl),
                (split_r + 1, 1, hb, wl),
                (1, split_c + 1, ht + hb + 1, wr),
            ]
        else:
            grey[split_r, split_c:] = GREY
            holes = [
                (1, 1, ht + hb + 1, wl),
                (1, split_c + 1, ht, wr),
                (split_r + 1, split_c + 1, hb, wr),
            ]
        return grey, holes
    return None


def _make_hsplit3(split_top: bool) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    for _ in range(200):
        wl, wr = random.randint(1, 2), random.randint(1, 2)
        ht, hb = random.randint(1, 3), random.randint(1, 3)
        if split_top:
            areas = [wl * ht, wr * ht, (wl + wr + 1) * hb]
        else:
            areas = [wl * hb, wr * hb, (wl + wr + 1) * ht]
        if min(areas) < 2 or max(areas) > 8 or len(set(areas)) != 3:
            continue

        h = ht + hb + 3
        w = wl + wr + 3
        grey = np.zeros((h, w), dtype=int)
        grey[0, :] = GREY
        grey[-1, :] = GREY
        grey[:, 0] = GREY
        grey[:, -1] = GREY
        split_c = wl + 1
        split_r = ht + 1
        grey[split_r, :] = GREY
        if split_top:
            grey[: split_r + 1, split_c] = GREY
            holes = [
                (1, 1, ht, wl),
                (1, split_c + 1, ht, wr),
                (split_r + 1, 1, hb, wl + wr + 1),
            ]
        else:
            grey[split_r:, split_c] = GREY
            holes = [
                (1, 1, ht, wl + wr + 1),
                (split_r + 1, 1, hb, wl),
                (split_r + 1, split_c + 1, hb, wr),
            ]
        return grey, holes
    return None


def _sample_structure() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    builders = [
        _make_cross4,
        lambda: _make_vsplit3(True),
        lambda: _make_vsplit3(False),
        lambda: _make_hsplit3(True),
        lambda: _make_hsplit3(False),
    ]
    random.shuffle(builders)
    for builder in builders:
        made = builder()
        if made is not None:
            return made
    return None


def _render_frame(
    local_frame: np.ndarray,
    holes: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    local_h, local_w = local_frame.shape
    top = random.randint(2, 5)
    bottom = random.randint(2, 5)
    left = random.randint(2, 5)
    right = random.randint(2, 5)
    h = local_h + top + bottom
    w = local_w + left + right
    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[top : top + local_h, left : left + local_w] = local_frame

    expected_holes: list[np.ndarray] = []
    for r0, c0, hh, ww in holes:
        coords = np.array(
            [(top + r, left + c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)],
            dtype=int,
        )
        expected_holes.append(coords)

    actual_holes = _components4((grid == BLACK) & (~_outside_mask(grid)))
    if len(actual_holes) != len(expected_holes):
        return None
    if _hole_order_is_spatial(actual_holes):
        return None
    return grid, actual_holes


def _available_scatter_cells(grid: np.ndarray, outside: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not outside[r, c] or grid[r, c] != BLACK:
                continue
            if any(0 <= nr < h and 0 <= nc < w and grid[nr, nc] == GREY for nr, nc in neighbors8(r, c)):
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _select_isolated_positions(
    grid: np.ndarray,
    outside: np.ndarray,
    total: int,
) -> list[tuple[int, int]] | None:
    candidates = _available_scatter_cells(grid, outside)
    for _ in range(80):
        chosen: list[tuple[int, int]] = []
        blocked: set[tuple[int, int]] = set()
        pool = candidates[:]
        random.shuffle(pool)
        for r, c in pool:
            if (r, c) in blocked:
                continue
            chosen.append((r, c))
            for nr, nc in neighbors8(r, c):
                blocked.add((nr, nc))
            blocked.add((r, c))
            if len(chosen) == total:
                return chosen
    return None


def _build_expected(
    grid: np.ndarray,
    holes: list[np.ndarray],
    color_counts: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    outside = _outside_mask(grid)
    out[outside & (grid != BLACK) & (grid != GREY)] = BLACK
    ranked_colors = [color for color, _count in sorted(color_counts.items(), key=lambda item: item[1], reverse=True)]
    ranked_holes = sorted(holes, key=lambda coords: int(coords.shape[0]), reverse=True)
    for color, hole in zip(ranked_colors, ranked_holes):
        _fill_component(out, hole, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    built = _sample_structure()
    if built is None:
        return None
    local_frame, holes_spec = built

    rendered = _render_frame(local_frame, holes_spec)
    if rendered is None:
        return None
    grid, holes = rendered

    outside = _outside_mask(grid)
    n_holes = len(holes)
    counts = random.sample(range(2, 8), n_holes)
    colors = random.sample(list(PALETTE), n_holes)
    color_counts = {color: count for color, count in zip(colors, counts)}

    positions = _select_isolated_positions(grid, outside, sum(counts))
    if positions is None:
        return None

    cursor = 0
    for color, count in color_counts.items():
        pts = positions[cursor : cursor + count]
        cursor += count
        rows = {r for r, _c in pts}
        cols = {c for _r, c in pts}
        if len(rows) == 1 or len(cols) == 1:
            return None
        for r, c in pts:
            grid[r, c] = color

    expected = _build_expected(grid, holes, color_counts)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
