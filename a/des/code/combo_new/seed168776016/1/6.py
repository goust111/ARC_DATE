from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PLUS_CORE = {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}
CORNERS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MARKER_OFFSETS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_ATTEMPTS = 400
GRID_H_RANGE = (17, 27)
GRID_W_RANGE = (17, 27)
CLUSTER_MARGIN = 6
MIN_SHIFT = 4
MAX_SHIFT = 6


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(np.array(cells, dtype=int))
    return components


def _bbox_center(coords: np.ndarray) -> tuple[int, int]:
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return (int((r0 + r1 + 1) // 2), int((c0 + c1 + 1) // 2))


def _pair_components(
    components: list[np.ndarray], markers: list[tuple[int, int]]
) -> list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]]:
    remaining = list(markers)
    ordered = sorted(
        components,
        key=lambda coords: (_bbox_center(coords), int(coords[:, 0].min()), int(coords[:, 1].min())),
    )

    pairs: list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]] = []
    for coords in ordered:
        center = _bbox_center(coords)
        best_idx = min(
            range(len(remaining)),
            key=lambda idx: (
                (remaining[idx][0] - center[0]) ** 2 + (remaining[idx][1] - center[1]) ** 2,
                remaining[idx][0],
                remaining[idx][1],
            ),
        )
        marker = remaining.pop(best_idx)
        pairs.append((coords, marker, center))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather the blue puzzle fragments onto their nearest red anchors and shade the merged silhouette by overlap count.

    Input structure:
    - The background is black.
    - Several blue 4-connected objects appear at separate locations.
    - The background also contains the same number of red single-pixel anchor markers.

    Rule:
    - Treat each blue object as one fragment and compute its rounded bounding-box center.
    - Pair each fragment with the nearest still-unmatched red marker, then translate the fragment so that its center lands on that marker.
    - Build the output on a black canvas of the same size: cells covered once become blue, cells covered twice become green, and cells covered three or more times become yellow.
    - Red markers are not kept in the output.

    Key insight:
    - The red pixels are hidden placement anchors for whole fragments, and the final colors encode only how many translated fragments overlap at each cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g == BLUE)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    assert components, "expected at least one blue fragment"
    assert len(components) == len(markers), "fragment/marker counts must match"

    cover = np.zeros(g.shape, dtype=int)
    for coords, marker, center in _pair_components(components, markers):
        dr = int(marker[0] - center[0])
        dc = int(marker[1] - center[1])
        for r, c in coords:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                cover[rr, cc] += 1

    out = np.full(g.shape, BG, dtype=int)
    out[cover == 1] = BLUE
    out[cover == 2] = GREEN
    out[cover >= 3] = YELLOW
    return out


def _random_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    cells = set(PLUS_CORE)
    for corner in CORNERS:
        if rng.random() < 0.5:
            cells.add(corner)
    return tuple(sorted(cells))


def _marker_position(center: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = MARKER_OFFSETS[direction]
    return (int(center[0] + dr), int(center[1] + dc))


def _origin_center(
    marker: tuple[int, int], direction: str, shift: int
) -> tuple[int, int]:
    if direction == "up":
        return (int(marker[0] - shift), int(marker[1]))
    if direction == "down":
        return (int(marker[0] + shift), int(marker[1]))
    if direction == "left":
        return (int(marker[0]), int(marker[1] - shift))
    return (int(marker[0]), int(marker[1] + shift))


def _shape_pixels(center: tuple[int, int], shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    return [(int(center[0] + dr), int(center[1] + dc)) for dr, dc in shape]


def _dist2(a: tuple[int, int], b: tuple[int, int]) -> int:
    return int((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _expected_output(
    grid_shape: tuple[int, int],
    shape_by_dir: dict[str, tuple[tuple[int, int], ...]],
    marker_by_dir: dict[str, tuple[int, int]],
) -> np.ndarray:
    cover = np.zeros(grid_shape, dtype=int)
    h, w = grid_shape
    for direction, marker in marker_by_dir.items():
        for rr, cc in _shape_pixels(marker, shape_by_dir[direction]):
            if 0 <= rr < h and 0 <= cc < w:
                cover[rr, cc] += 1

    out = np.full(grid_shape, BG, dtype=int)
    out[cover == 1] = BLUE
    out[cover == 2] = GREEN
    out[cover >= 3] = YELLOW
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    row_lo = CLUSTER_MARGIN
    row_hi = h - CLUSTER_MARGIN - 1
    col_lo = CLUSTER_MARGIN
    col_hi = w - CLUSTER_MARGIN - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    cluster_center = (rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi))

    directions = list(MARKER_OFFSETS)
    if rng.random() < 0.5:
        directions = rng.sample(directions, 3)

    marker_by_dir = {
        direction: _marker_position(cluster_center, direction) for direction in directions
    }

    shape_by_dir = {direction: _random_shape(rng) for direction in directions}
    if len(set(shape_by_dir.values())) == 1:
        tweak = rng.choice(directions)
        old_shape = set(shape_by_dir[tweak])
        if set(CORNERS).issubset(old_shape):
            old_shape.remove(rng.choice(CORNERS))
        else:
            old_shape.add(rng.choice([corner for corner in CORNERS if corner not in old_shape]))
        shape_by_dir[tweak] = tuple(sorted(old_shape))

    origin_by_dir: dict[str, tuple[int, int]] = {}
    for direction, marker in marker_by_dir.items():
        if direction == "up":
            max_shift = int(marker[0] - 1)
        elif direction == "down":
            max_shift = int(h - marker[0] - 2)
        elif direction == "left":
            max_shift = int(marker[1] - 1)
        else:
            max_shift = int(w - marker[1] - 2)
        if max_shift < MIN_SHIFT:
            return None
        shift = rng.randint(MIN_SHIFT, min(MAX_SHIFT, max_shift))
        origin_by_dir[direction] = _origin_center(marker, direction, shift)

    for direction, origin in origin_by_dir.items():
        own_dist = _dist2(origin, marker_by_dir[direction])
        for other_direction, other_marker in marker_by_dir.items():
            if other_direction == direction:
                continue
            if own_dist >= _dist2(origin, other_marker):
                return None

    grid = np.full((h, w), BG, dtype=int)
    occupied = set(marker_by_dir.values())
    for marker in marker_by_dir.values():
        grid[marker] = RED

    for direction in directions:
        pixels = _shape_pixels(origin_by_dir[direction], shape_by_dir[direction])
        if any(
            rr < 0 or rr >= h or cc < 0 or cc >= w or (int(rr), int(cc)) in occupied
            for rr, cc in pixels
        ):
            return None
        for pixel in pixels:
            occupied.add((int(pixel[0]), int(pixel[1])))
            grid[pixel] = BLUE

    if len(_find_components(grid == BLUE)) != len(directions):
        return None

    expected = _expected_output(grid.shape, shape_by_dir, marker_by_dir)
    colors = set(map(int, expected.reshape(-1)))
    if not {BLUE, GREEN, YELLOW}.issubset(colors):
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
