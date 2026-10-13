from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
ARC_MAX = 30
GEN_ATTEMPTS = 400
GRID_SIZE_MIN = 17
GRID_SIZE_MAX = 24
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 4
PALETTE = tuple(color for color in range(10) if color != BG)


def _shape_variants(shape: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for k in range(4):
        rotated = np.rot90(shape, k)
        for candidate in (rotated, np.fliplr(rotated)):
            key = tuple(tuple(int(x) for x in row) for row in candidate.astype(int))
            if key not in seen:
                seen.add(key)
                variants.append(candidate.copy())
    return variants


_BASE_SHAPES = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1, 1]], dtype=bool),
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
]

SHAPES: list[np.ndarray] = []
_seen_shapes: set[tuple[tuple[int, ...], ...]] = set()
for _base in _BASE_SHAPES:
    for _variant in _shape_variants(_base):
        _key = tuple(tuple(int(x) for x in row) for row in _variant.astype(int))
        if _key not in _seen_shapes:
            _seen_shapes.add(_key)
            SHAPES.append(_variant.copy())


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _distance_to_edge(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return min(r0, c0, h - 1 - r1, w - 1 - c1)


def _bbox_center_cell(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _scaled_cells(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    center_r, center_c = _bbox_center_cell(bbox)
    # Align the midpoint cell of the enlarged bbox with the source bbox midpoint cell.
    top = center_r - (2 * box_h - 1) // 2
    left = center_c - (2 * box_w - 1) // 2

    out: set[tuple[int, int]] = set()
    for r, c in coords:
        local_r = int(r) - r0
        local_c = int(c) - c0
        base_r = top + 2 * local_r
        base_c = left + 2 * local_c
        for dr in (0, 1):
            for dc in (0, 1):
                out.add((base_r + dr, base_c + dc))
    return out


def _connected_components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )

    return comps


def _transformed_cells(
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    if _distance_to_edge(bbox, shape) < 3:
        return {_bbox_center_cell(bbox)}
    return _scaled_cells(coords, bbox)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink edge-near objects to their bbox center and enlarge center-region objects around the same bbox center.

    Input structure:
    - The background is Azure.
    - The foreground contains several 4-connected monochrome objects in distinct colors.
    - Each object is a simple small shape such as a thin rectangle or an L shape.
    - Objects can lie near the border or farther inside the grid, giving different bbox-to-edge distances.

    Rule:
    - Split the foreground into 4-connected monochrome objects and compute each object's bounding box.
    - Let D be the minimum gap from that bounding box to any grid edge.
    - If D < 3, erase the object and keep only the bbox center pixel; if D >= 3, replace every source pixel by a 2x2 block and place the enlarged result so its bbox midpoint stays aligned with the original bbox midpoint.

    Key insight:
    - The hidden classifier is not shape or color but distance from the object's bbox to the nearest outer border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full_like(g, BG)
    for comp in _connected_components(g, bg=BG):
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        bbox = comp["bbox"]
        cells = _transformed_cells(coords, bbox, g.shape)
        for r, c in cells:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            assert int(out[r, c]) == int(BG), "transformed objects must stay disjoint"
            out[r, c] = color
    return out


def _inflate_cells(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    radius: int = 1,
) -> set[tuple[int, int]]:
    h, w = shape
    inflated: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    inflated.add((nr, nc))
    return inflated


def _shape_cells(shape: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(shape):
        cells.add((top + int(r), left + int(c)))
    return cells


def _candidate_positions(
    canvas_shape: tuple[int, int],
    sprite_shape: np.ndarray,
    want_far: bool,
) -> list[tuple[int, int]]:
    h, w = canvas_shape
    sh, sw = sprite_shape.shape
    positions: list[tuple[int, int]] = []
    for top in range(h - sh + 1):
        for left in range(w - sw + 1):
            bbox = (top, left, top + sh - 1, left + sw - 1)
            dist = _distance_to_edge(bbox, canvas_shape)
            if want_far and dist >= 3:
                positions.append((top, left))
            if (not want_far) and dist < 3:
                positions.append((top, left))
    random.shuffle(positions)
    return positions


def _branch_labels(n_objects: int) -> list[str]:
    n_far = n_objects // 2
    if n_objects % 2 == 1 and random.random() < 0.5:
        n_far += 1
    n_far = max(1, min(n_objects - 1, n_far))
    labels = ["far"] * n_far + ["near"] * (n_objects - n_far)
    random.shuffle(labels)
    return labels


def _try_place_object(
    grid: np.ndarray,
    sprite: np.ndarray,
    color: int,
    branch: str,
    blocked_in: set[tuple[int, int]],
    blocked_out: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    positions = _candidate_positions(grid.shape, sprite, want_far=(branch == "far"))
    for top, left in positions:
        input_cells = _shape_cells(sprite, top, left)
        if input_cells & blocked_in:
            continue

        coords = np.array(sorted(input_cells), dtype=int)
        bbox = _bbox(coords)
        output_cells = _transformed_cells(coords, bbox, grid.shape)
        if output_cells & blocked_out:
            continue

        for r, c in input_cells:
            grid[r, c] = color
        return input_cells, output_cells
    return None


def _valid_instance(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > ARC_MAX:
        return False
    if not np.any(grid != BG):
        return False

    comps = _connected_components(grid, bg=BG)
    if len(comps) < 2:
        return False
    if len({int(comp["color"]) for comp in comps}) != len(comps):
        return False

    distances = [_distance_to_edge(comp["bbox"], grid.shape) for comp in comps]
    if not any(dist < 3 for dist in distances):
        return False
    if not any(dist >= 3 for dist in distances):
        return False

    if int(np.count_nonzero(grid == BG)) <= int(np.count_nonzero(grid != BG)):
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    w = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    n_objects = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    colors = random.sample(PALETTE, n_objects)
    branches = sorted(_branch_labels(n_objects), key=lambda label: label != "far")
    sprites = [random.choice(SHAPES) for _ in range(n_objects)]

    grid = np.full((h, w), BG, dtype=int)
    blocked_in: set[tuple[int, int]] = set()
    blocked_out: set[tuple[int, int]] = set()

    for color, branch, sprite in zip(colors, branches, sprites):
        placed = _try_place_object(grid, sprite, color, branch, blocked_in, blocked_out)
        if placed is None:
            return None
        input_cells, output_cells = placed
        blocked_in |= _inflate_cells(input_cells, grid.shape, radius=1)
        blocked_out |= _inflate_cells(output_cells, grid.shape, radius=1)

    if _valid_instance(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
