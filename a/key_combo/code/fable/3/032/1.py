from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
RED = 2
ARROW_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
GENERATION_ATTEMPTS = 4000
PLACEMENT_ATTEMPTS = 160
MAX_VISIBLE_CROSSINGS = 10
_VALID_START_CACHE: dict[
    tuple[int, int, int], list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]]
] = {}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_arrow_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    components: list[dict[str, object]] = []

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        mask = grid == color
        visited = np.zeros_like(mask, dtype=bool)
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or visited[r, c]:
                    continue
                stack = [(r, c)]
                visited[r, c] = True
                coords: list[tuple[int, int]] = []
                while stack:
                    cr, cc = stack.pop()
                    coords.append((cr, cc))
                    for nr, nc in _neighbors4(cr, cc):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                arr = np.array(coords, dtype=int)
                bbox = _bbox(arr)
                components.append({"color": color, "coords": arr, "bbox": bbox})

    return components


def _infer_direction(sprite_mask: np.ndarray) -> tuple[int, int]:
    h, w = sprite_mask.shape
    corner_map = {
        (-1, -1): bool(sprite_mask[0, 0]),
        (-1, 1): bool(sprite_mask[0, w - 1]),
        (1, -1): bool(sprite_mask[h - 1, 0]),
        (1, 1): bool(sprite_mask[h - 1, w - 1]),
    }
    missing_corners = [direction for direction, filled in corner_map.items() if not filled]
    if len(missing_corners) == 1:
        return missing_corners[0]

    filled = np.argwhere(sprite_mask)
    holes = np.argwhere(~sprite_mask)
    if holes.size:
        delta = holes.mean(axis=0) - filled.mean(axis=0)
        dr = -1 if delta[0] < 0 else 1 if delta[0] > 0 else 0
        dc = -1 if delta[1] < 0 else 1 if delta[1] > 0 else 0
        if dr and dc:
            return int(dr), int(dc)

    row_mid = h / 2.0
    col_mid = w / 2.0
    quadrant_counts: dict[tuple[int, int], int] = {}
    for direction in DIAGONALS:
        dr, dc = direction
        row_mask = np.arange(h) < row_mid if dr < 0 else np.arange(h) >= row_mid
        col_mask = np.arange(w) < col_mid if dc < 0 else np.arange(w) >= col_mid
        quadrant_counts[direction] = int(sprite_mask[np.ix_(row_mask, col_mask)].sum())

    return min(DIAGONALS, key=lambda direction: quadrant_counts[direction])


def _start_from_bbox(bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    dr, dc = direction
    return (r0 - 1 if dr < 0 else r1 + 1, c0 - 1 if dc < 0 else c1 + 1)


def _trace_path(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]
) -> tuple[set[tuple[int, int]], bool]:
    h, w = shape
    r, c = start
    dr, dc = direction
    visited: set[tuple[int, int]] = set()
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        if not (0 <= r < h and 0 <= c < w):
            return visited, False

        state = (r, c, dr, dc)
        if state in seen:
            return visited, False
        seen.add(state)
        visited.add((r, c))

        if r in (0, h - 1) and c in (0, w - 1):
            return visited, True

        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= h:
            dr = -dr
            nr = r + dr
        if nc < 0 or nc >= w:
            dc = -dc
            nc = c + dc
        r, c = nr, nc


def _trajectory_counts(
    shape: tuple[int, int], paths: list[set[tuple[int, int]]], arrow_mask: np.ndarray
) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for path in paths:
        for r, c in path:
            if not arrow_mask[r, c]:
                counts[r, c] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each arrowhead's diagonal firing direction, reflect the invisible trajectories, and paint only shared path cells red.

    Input structure:
    - The grid is a black background with two or three differently colored arrowhead sprites.
    - All arrowheads share the same size and shape; one bbox corner is missing from each sprite.
    - No other foreground objects are present.

    Rule:
    - For each arrowhead, use the missing bbox corner to infer one diagonal direction.
    - Start an invisible trajectory from the matching outer bbox corner, move diagonally one cell at a time, and reflect off grid borders.
    - Stop a trajectory when it reaches a grid corner cell, then paint only the background cells visited by two or more trajectories red while leaving the arrowheads unchanged.

    Key insight:
    - The output depends entirely on reconstructing hidden motion: the only visible clues are the arrowheads' missing corners, which determine both the launch point and the full reflected path.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    components = _find_arrow_components(g)

    counts = np.zeros(g.shape, dtype=int)
    for component in components:
        bbox = component["bbox"]
        color = int(component["color"])
        r0, c0, r1, c1 = bbox
        sprite_mask = g[r0 : r1 + 1, c0 : c1 + 1] == color
        direction = _infer_direction(sprite_mask)
        start = _start_from_bbox(bbox, direction)
        path, _terminated = _trace_path(g.shape, start, direction)
        for r, c in path:
            if g[r, c] == BLACK:
                counts[r, c] += 1

    out[(counts >= 2) & (g == BLACK)] = RED
    return out


def _make_arrow(color: int, direction: tuple[int, int], scale: int) -> np.ndarray:
    size = 2 * scale
    sprite = np.full((size, size), color, dtype=int)
    miss_r = 0 if direction[0] < 0 else 1
    miss_c = 0 if direction[1] < 0 else 1
    sprite[miss_r * scale : (miss_r + 1) * scale, miss_c * scale : (miss_c + 1) * scale] = BLACK
    return sprite


def _bbox_from_start(
    start: tuple[int, int], direction: tuple[int, int], size: int
) -> tuple[int, int, int, int]:
    sr, sc = start
    dr, dc = direction
    r0 = sr + 1 if dr < 0 else sr - size
    c0 = sc + 1 if dc < 0 else sc - size
    return r0, c0, r0 + size - 1, c0 + size - 1


def _expanded_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a[0] - pad, a[1] - pad, a[2] + pad, a[3] + pad
    br0, bc0, br1, bc1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _slope(direction: tuple[int, int]) -> int:
    return int(direction[0] * direction[1])


def _valid_start_states(
    h: int, w: int, size: int
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]]:
    key = (h, w, size)
    if key in _VALID_START_CACHE:
        return _VALID_START_CACHE[key]

    gcd_hw = math.gcd(h - 1, w - 1)
    states: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            for direction in DIAGONALS:
                dr, dc = direction
                row_to_wall = r if dr < 0 else (h - 1 - r)
                col_to_wall = c if dc < 0 else (w - 1 - c)
                if (row_to_wall - col_to_wall) % gcd_hw != 0:
                    continue
                bbox = _bbox_from_start((r, c), direction, size)
                if bbox[0] < 1 or bbox[1] < 1 or bbox[2] > h - 2 or bbox[3] > w - 2:
                    continue
                states.append(((r, c), direction, bbox))

    _VALID_START_CACHE[key] = states
    return states


def _choose_states(
    candidates: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]], n_arrows: int
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]] | None:
    positive = [state for state in candidates if _slope(state[1]) == 1]
    negative = [state for state in candidates if _slope(state[1]) == -1]
    if not positive or not negative:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        chosen: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]] = []

        for pool in (positive, negative):
            for _inner in range(40):
                candidate = random.choice(pool)
                if all(not _expanded_overlap(candidate[2], prev[2], pad=1) for prev in chosen):
                    chosen.append(candidate)
                    break
            else:
                break
        else:
            while len(chosen) < n_arrows:
                for _inner in range(80):
                    candidate = random.choice(candidates)
                    if all(not _expanded_overlap(candidate[2], prev[2], pad=1) for prev in chosen):
                        chosen.append(candidate)
                        break
                else:
                    break

            if len(chosen) == n_arrows:
                return chosen

    return None


def _gen_attempt() -> np.ndarray | None:
    n_arrows = 2 if random.random() < 0.55 else 3
    scale = 1 if random.random() < 0.6 else 2
    min_side = 12 + (2 if scale == 2 else 0) + (2 if n_arrows == 3 else 0)
    h = random.randint(min_side, 26)
    w = random.randint(min_side, 26)
    size = 2 * scale

    candidates = _valid_start_states(h, w, size)
    if len(candidates) < 12:
        return None

    states = _choose_states(candidates, n_arrows)
    if states is None:
        return None

    colors = random.sample(ARROW_COLORS, n_arrows)
    grid = np.zeros((h, w), dtype=int)
    arrow_mask = np.zeros((h, w), dtype=bool)
    paths: list[set[tuple[int, int]]] = []

    for color, (start, direction, bbox) in zip(colors, states):
        sprite = _make_arrow(color, direction, scale)
        r0, c0, r1, c1 = bbox
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        sprite_mask = sprite != BLACK
        region[sprite_mask] = sprite[sprite_mask]
        arrow_mask[r0 : r1 + 1, c0 : c1 + 1] |= sprite_mask

        path, terminated = _trace_path((h, w), start, direction)
        if not terminated or len(path) < 4:
            return None
        paths.append(path)

    for path in paths:
        if any(arrow_mask[r, c] for r, c in path):
            return None

    counts = _trajectory_counts((h, w), paths, arrow_mask)
    crossing_mask = (counts >= 2) & ~arrow_mask
    if not crossing_mask.any():
        return None

    corner_mask = np.zeros((h, w), dtype=bool)
    corner_mask[0, 0] = True
    corner_mask[0, w - 1] = True
    corner_mask[h - 1, 0] = True
    corner_mask[h - 1, w - 1] = True
    interior_crossings = crossing_mask & ~corner_mask
    if not interior_crossings.any():
        return None

    if int(crossing_mask.sum()) > MAX_VISIBLE_CROSSINGS or int(interior_crossings.sum()) > 6:
        return None

    output = main(grid)
    if np.array_equal(output, grid):
        return None
    if not np.array_equal(output == RED, crossing_mask | (grid == RED)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-trajectory input")
