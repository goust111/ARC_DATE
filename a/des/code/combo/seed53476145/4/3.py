from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
GRID_H_RANGE = (11, 19)
GRID_W_RANGE = (11, 19)
MAX_MARKERS = 6
EXTRA_PLACEMENT_TRIES = 80


def _plus_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {
        (r, c),
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    }


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _can_place(center: tuple[int, int], shape: tuple[int, int], forbidden: set[tuple[int, int]]) -> bool:
    h, w = shape
    cells = _plus_cells(center)
    return all(0 <= r < h and 0 <= c < w and (r, c) not in forbidden for r, c in cells)


def _commit_center(
    center: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> None:
    cells = _plus_cells(center)
    occupied.update(cells)
    forbidden.update(_expanded(cells, shape))


def _find_markers(grid: np.ndarray) -> list[tuple[int, tuple[int, int]]]:
    markers: list[tuple[int, tuple[int, int]]] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == BG:
            continue
        coords = np.argwhere(grid == color)
        if coords.shape[0] != 5:
            continue
        cells = {tuple(map(int, coord)) for coord in coords}
        sum_r = sum(r for r, _c in cells)
        sum_c = sum(c for _r, c in cells)
        if sum_r % 5 != 0 or sum_c % 5 != 0:
            continue
        center = (sum_r // 5, sum_c // 5)
        if cells == _plus_cells(center):
            markers.append((color, center))
    markers.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
    return markers


def _trace_rays(
    grid: np.ndarray,
    markers: list[tuple[int, tuple[int, int]]],
) -> tuple[dict[tuple[int, int], set[int]], int, int]:
    h, w = grid.shape
    blockers = grid != BG
    visits: dict[tuple[int, int], set[int]] = {}
    edge_stops = 0
    marker_stops = 0

    for color, (r0, c0) in markers:
        for dr, dc in DIRECTIONS:
            r = r0 + 2 * dr
            c = c0 + 2 * dc
            while 0 <= r < h and 0 <= c < w and not blockers[r, c]:
                visits.setdefault((r, c), set()).add(color)
                r += dr
                c += dc
            if 0 <= r < h and 0 <= c < w:
                marker_stops += 1
            else:
                edge_stops += 1

    return visits, edge_stops, marker_stops


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    markers = _find_markers(g)
    if not markers:
        return out

    visits, _edge_stops, _marker_stops = _trace_rays(g, markers)
    for (r, c), colors in visits.items():
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        else:
            out[r, c] = GREY
    return out


def _sample_base_layout(rng: np.random.Generator, h: int, w: int) -> list[tuple[int, int]] | None:
    orientation = "h" if int(rng.integers(2)) == 0 else "v"

    if orientation == "h":
        valid_rows = [r for r in range(1, h - 1) if max(r - 1, h - 2 - r) >= 4]
        valid_pairs = [(c1, c2) for c1 in range(2, w - 3) for c2 in range(c1 + 6, w - 1)]
        if not valid_rows or not valid_pairs:
            return None
        row = int(valid_rows[int(rng.integers(len(valid_rows)))])
        c1, c2 = valid_pairs[int(rng.integers(len(valid_pairs)))]
        up_room = row - 1
        down_room = h - 2 - row
        side_choices = []
        if up_room >= 4:
            side_choices.append(-1)
        if down_room >= 4:
            side_choices.append(1)
        sign = int(side_choices[int(rng.integers(len(side_choices)))])
        offset_max = up_room if sign < 0 else down_room
        offset = int(rng.integers(4, offset_max + 1))
        cross_row = row + sign * offset
        cross_col = int(rng.integers(c1 + 2, c2 - 1))
        return [(row, c1), (row, c2), (cross_row, cross_col)]

    valid_cols = [c for c in range(1, w - 1) if max(c - 1, w - 2 - c) >= 4]
    valid_pairs = [(r1, r2) for r1 in range(2, h - 3) for r2 in range(r1 + 6, h - 1)]
    if not valid_cols or not valid_pairs:
        return None
    col = int(valid_cols[int(rng.integers(len(valid_cols)))])
    r1, r2 = valid_pairs[int(rng.integers(len(valid_pairs)))]
    left_room = col - 1
    right_room = w - 2 - col
    side_choices = []
    if left_room >= 4:
        side_choices.append(-1)
    if right_room >= 4:
        side_choices.append(1)
    sign = int(side_choices[int(rng.integers(len(side_choices)))])
    offset_max = left_room if sign < 0 else right_room
    offset = int(rng.integers(4, offset_max + 1))
    cross_row = int(rng.integers(r1 + 2, r2 - 1))
    cross_col = col + sign * offset
    return [(r1, col), (r2, col), (cross_row, cross_col)]


def _sample_extra_center(
    rng: np.random.Generator,
    h: int,
    w: int,
    existing: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    for _ in range(EXTRA_PLACEMENT_TRIES):
        mode = int(rng.integers(3))
        if mode == 0:
            row = existing[int(rng.integers(len(existing)))][0]
            center = (row, int(rng.integers(1, w - 1)))
        elif mode == 1:
            col = existing[int(rng.integers(len(existing)))][1]
            center = (int(rng.integers(1, h - 1)), col)
        else:
            center = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
        if _can_place(center, (h, w), forbidden):
            return center
    return None


def _scene_features(grid: np.ndarray) -> dict[str, int] | None:
    markers = _find_markers(grid)
    if len(markers) != len([c for c in np.unique(grid) if int(c) != BG]):
        return None
    visits, edge_stops, marker_stops = _trace_rays(grid, markers)
    grey_count = sum(len(colors) > 1 for colors in visits.values())
    solid_count = sum(len(colors) == 1 for colors in visits.values())
    return {
        "markers": len(markers),
        "ray_pixels": len(visits),
        "grey_count": grey_count,
        "solid_count": solid_count,
        "edge_stops": edge_stops,
        "marker_stops": marker_stops,
    }


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
    centers = _sample_base_layout(rng, h, w)
    if centers is None:
        return None

    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    for center in centers:
        if not _can_place(center, (h, w), forbidden):
            return None
        _commit_center(center, (h, w), occupied, forbidden)

    target_markers = int(rng.integers(3, min(MAX_MARKERS, len(INPUT_COLORS)) + 1))
    while len(centers) < target_markers:
        candidate = _sample_extra_center(rng, h, w, centers, forbidden)
        if candidate is None:
            break
        centers.append(candidate)
        _commit_center(candidate, (h, w), occupied, forbidden)

    colors = rng.choice(np.array(INPUT_COLORS, dtype=int), size=len(centers), replace=False)
    grid = np.full((h, w), BG, dtype=int)
    for color, center in zip(colors.tolist(), centers):
        for r, c in _plus_cells(center):
            grid[r, c] = int(color)

    features = _scene_features(grid)
    if features is None:
        return None
    if features["markers"] < 3:
        return None
    if features["ray_pixels"] < 8:
        return None
    if features["grey_count"] < 1:
        return None
    if features["solid_count"] < 4:
        return None
    if features["marker_stops"] < 2:
        return None
    if features["edge_stops"] < 4:
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
