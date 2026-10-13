from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
WALL = 5
REGION_COLORS = (1, 2, 3, 4, 7, 6, 8, 9)
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))

_GEN_RNG = np.random.default_rng()
_GEN_MAX_ATTEMPTS = 1000


def _corridor_mask(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero_rows = np.flatnonzero(np.all(grid == BG, axis=1))
    zero_cols = np.flatnonzero(np.all(grid == BG, axis=0))
    mask = np.zeros(grid.shape, dtype=bool)
    if zero_rows.size:
        mask[zero_rows, :] = True
    if zero_cols.size:
        mask[:, zero_cols] = True
    return mask, zero_rows, zero_cols


def _colored_components(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if seen[sr, sc] or color in (BG, WALL):
                continue

            seen[sr, sc] = True
            pixels: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            while q:
                r, c = q.popleft()
                pixels.append((r, c))
                for dr, dc in NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "pixels": tuple(pixels),
                    "area": len(pixels),
                }
            )
    return components


def _touches_corridor(
    pixels: tuple[tuple[int, int], ...], corridor: np.ndarray
) -> bool:
    h, w = map(int, corridor.shape)
    for r, c in pixels:
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and corridor[nr, nc]:
                return True
    return False


def _edge_pixels_for_shrink(
    grid: np.ndarray, pixels: tuple[tuple[int, int], ...]
) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    edges = []
    for r, c in pixels:
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in (BG, WALL):
                edges.append((r, c))
                break
    return edges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored regions touching invisible all-black corridors, and shrink the regions that do not touch any corridor.

    Input structure:
    - A black-background grid contains grey wall pixels and several monochromatic colored regions.
    - Invisible corridors are full rows or full columns made only of black background pixels.

    Rule:
    - A colored region is corridor-adjacent if any of its 4-neighbors lies on a full black row or full black column.
    - Corridor-adjacent regions grow one layer into original black, non-corridor cells without crossing walls or other colors.
    - Corridor-isolated regions shrink one layer: colored pixels touching original black or grey become black.
    - Grey walls and corridor pixels stay unchanged; contested growth cells go to the larger original region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    out = np.array(g, copy=True)
    corridor, _, _ = _corridor_mask(g)
    growth: dict[tuple[int, int], tuple[int, int]] = {}

    for comp in _colored_components(g):
        color = int(comp["color"])
        area = int(comp["area"])
        pixels = comp["pixels"]

        if _touches_corridor(pixels, corridor):
            h, w = map(int, g.shape)
            for r, c in pixels:
                for dr, dc in NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if int(g[nr, nc]) != BG or corridor[nr, nc]:
                        continue
                    current = growth.get((nr, nc))
                    if current is None or area > current[0] or (
                        area == current[0] and color < current[1]
                    ):
                        growth[(nr, nc)] = (area, color)
        else:
            for r, c in _edge_pixels_for_shrink(g, pixels):
                out[r, c] = BG

    for (r, c), (_, color) in growth.items():
        out[r, c] = color

    out[g == WALL] = WALL
    out[corridor] = BG
    return out


def _side_cells(
    top: int, left: int, height: int, width: int, side: str
) -> list[tuple[int, int]]:
    bottom = top + height - 1
    right = left + width - 1
    if side == "top":
        return [(top - 1, c) for c in range(left, right + 1)]
    if side == "bottom":
        return [(bottom + 1, c) for c in range(left, right + 1)]
    if side == "left":
        return [(r, left - 1) for r in range(top, bottom + 1)]
    if side == "right":
        return [(r, right + 1) for r in range(top, bottom + 1)]
    raise ValueError(side)


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _can_place_rect(
    grid: np.ndarray,
    reserved: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    openings: set[str],
    corridor_row: int,
    corridor_col: int,
) -> bool:
    h, w = map(int, grid.shape)
    bottom = top + height - 1
    right = left + width - 1
    if top < 1 or left < 1 or bottom >= h - 1 or right >= w - 1:
        return False
    if top <= corridor_row <= bottom or left <= corridor_col <= right:
        return False

    rect = grid[top : bottom + 1, left : right + 1]
    rect_reserved = reserved[top : bottom + 1, left : right + 1]
    if np.any(rect != BG) or np.any(rect_reserved):
        return False

    for side in ("top", "bottom", "left", "right"):
        cells = _side_cells(top, left, height, width, side)
        if side in openings:
            for r, c in cells:
                if not _inside((h, w), r, c):
                    return False
                if r == corridor_row or c == corridor_col:
                    if int(grid[r, c]) != BG:
                        return False
                elif int(grid[r, c]) != BG or reserved[r, c]:
                    return False
            continue

        for r, c in cells:
            if not _inside((h, w), r, c):
                return False
            if r == corridor_row or c == corridor_col:
                continue
            if int(grid[r, c]) != BG or reserved[r, c]:
                return False

    return True


def _place_rect(
    grid: np.ndarray,
    reserved: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    color: int,
    openings: set[str],
    corridor_row: int,
    corridor_col: int,
) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top : bottom + 1, left : right + 1] = color

    for side in ("top", "bottom", "left", "right"):
        for r, c in _side_cells(top, left, height, width, side):
            if not _inside(grid.shape, r, c) or r == corridor_row or c == corridor_col:
                continue
            if side in openings:
                reserved[r, c] = True
            elif int(grid[r, c]) == BG:
                grid[r, c] = WALL


def _sample_adjacent_rect(
    rng: np.random.Generator, h: int, w: int, corridor_row: int, corridor_col: int
) -> tuple[int, int, int, int, set[str]] | None:
    height = int(rng.integers(3, 5))
    width = int(rng.integers(3, 5))
    sides = list(rng.permutation(("above", "below", "left", "right")))

    for side in sides:
        if side in ("above", "below"):
            candidates = [
                c
                for c in range(1, w - width)
                if not (c <= corridor_col <= c + width - 1)
            ]
            if not candidates:
                continue
            left = int(candidates[int(rng.integers(0, len(candidates)))])
            if side == "above":
                top = corridor_row - height
                openings = {"top", "bottom"}
            else:
                top = corridor_row + 1
                openings = {"top", "bottom"}
        else:
            candidates = [
                r
                for r in range(1, h - height)
                if not (r <= corridor_row <= r + height - 1)
            ]
            if not candidates:
                continue
            top = int(candidates[int(rng.integers(0, len(candidates)))])
            if side == "left":
                left = corridor_col - width
                openings = {"left", "right"}
            else:
                left = corridor_col + 1
                openings = {"left", "right"}
        return top, left, height, width, openings
    return None


def _sample_isolated_rect(
    rng: np.random.Generator, h: int, w: int, corridor_row: int, corridor_col: int
) -> tuple[int, int, int, int, set[str]] | None:
    height = int(rng.integers(3, 5))
    width = int(rng.integers(3, 5))
    candidates: list[tuple[int, int]] = []
    for top in range(1, h - height):
        bottom = top + height - 1
        if not (bottom < corridor_row - 1 or top > corridor_row + 1):
            continue
        for left in range(1, w - width):
            right = left + width - 1
            if right < corridor_col - 1 or left > corridor_col + 1:
                candidates.append((top, left))
    if not candidates:
        return None
    top, left = candidates[int(rng.integers(0, len(candidates)))]
    return top, left, height, width, set()


def _seal_empty_lines(
    grid: np.ndarray, reserved: np.ndarray, corridor_row: int, corridor_col: int
) -> bool:
    h, w = map(int, grid.shape)
    for r in range(h):
        if r == corridor_row or not np.all(grid[r, :] == BG):
            continue
        candidates = [
            c
            for c in range(w)
            if c != corridor_col and not reserved[r, c] and int(grid[r, c]) == BG
        ]
        if not candidates:
            return False
        grid[r, int(candidates[0])] = WALL

    for c in range(w):
        if c == corridor_col or not np.all(grid[:, c] == BG):
            continue
        candidates = [
            r
            for r in range(h)
            if r != corridor_row and not reserved[r, c] and int(grid[r, c]) == BG
        ]
        if not candidates:
            return False
        grid[int(candidates[-1]), c] = WALL
    return True


def _generated_case_is_valid(grid: np.ndarray) -> bool:
    corridor, zero_rows, zero_cols = _corridor_mask(grid)
    if zero_rows.size != 1 or zero_cols.size != 1:
        return False

    components = _colored_components(grid)
    if not (5 <= len(components) <= 7):
        return False
    if min(int(comp["area"]) for comp in components) < 9:
        return False

    adjacency = [_touches_corridor(comp["pixels"], corridor) for comp in components]
    if sum(adjacency) < 2 or len(adjacency) - sum(adjacency) < 2:
        return False

    out = main(grid)
    if out.shape != grid.shape or not np.all(out[corridor] == BG):
        return False
    if not np.all(out[grid == WALL] == WALL):
        return False

    grown = np.count_nonzero((grid == BG) & (~corridor) & (out != BG))
    shrunk = np.count_nonzero((grid != BG) & (grid != WALL) & (out == BG))
    return bool(grown > 0 and shrunk > 0)


def _build_candidate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 20))
    w = int(rng.integers(16, 20))
    corridor_row = int(rng.integers(6, h - 6))
    corridor_col = int(rng.integers(6, w - 6))

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    reserved[corridor_row, :] = True
    reserved[:, corridor_col] = True

    total_regions = int(rng.integers(5, 8))
    adjacent_regions = int(rng.integers(2, min(4, total_regions - 1)))
    roles = ["adjacent"] * adjacent_regions + ["isolated"] * (total_regions - adjacent_regions)
    roles = [str(v) for v in rng.permutation(roles)]
    colors = [int(v) for v in rng.choice(REGION_COLORS, size=total_regions, replace=False)]

    for role, color in zip(roles, colors):
        placed = False
        for _ in range(120):
            if role == "adjacent":
                sample = _sample_adjacent_rect(rng, h, w, corridor_row, corridor_col)
            else:
                sample = _sample_isolated_rect(rng, h, w, corridor_row, corridor_col)
            if sample is None:
                continue
            top, left, height, width, openings = sample
            if not _can_place_rect(
                grid,
                reserved,
                top,
                left,
                height,
                width,
                openings,
                corridor_row,
                corridor_col,
            ):
                continue
            _place_rect(
                grid,
                reserved,
                top,
                left,
                height,
                width,
                color,
                openings,
                corridor_row,
                corridor_col,
            )
            placed = True
            break
        if not placed:
            return None

    if not _seal_empty_lines(grid, reserved, corridor_row, corridor_col):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        candidate = _build_candidate(_GEN_RNG)
        if candidate is not None and _generated_case_is_valid(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid invisible-corridor region case")
