from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
LEGEND_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
CANONICAL_WRONG_ROLES = (1, 4, 2)

GRID_ROWS_RANGE = (16, 24)
GRID_COLS_RANGE = (16, 24)
OBJECT_COUNT_EXTRAS = (0, 2)
PLACEMENT_GAP = 1
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_non_bg(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _legend_roles(sub: np.ndarray) -> tuple[int, int, int] | None:
    if sub.shape != (3, 3):
        return None
    if np.any(sub == BG):
        return None
    corner_vals = {int(sub[r, c]) for r, c in ((0, 0), (0, 2), (2, 0), (2, 2))}
    edge_vals = {int(sub[r, c]) for r, c in ((0, 1), (1, 0), (1, 2), (2, 1))}
    if len(corner_vals) != 1 or len(edge_vals) != 1:
        return None
    corner_color = next(iter(corner_vals))
    edge_color = next(iter(edge_vals))
    center_color = int(sub[1, 1])
    if len({corner_color, edge_color, center_color}) != 3:
        return None
    return corner_color, edge_color, center_color


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int]]:
    candidates: list[tuple[tuple[int, int, int, int], tuple[int, int, int]]] = []
    for coords in _components_non_bg(grid):
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if len(coords) != 9:
            continue
        roles = _legend_roles(sub)
        if roles is not None:
            candidates.append(((r0, c0, r1, c1), roles))
    assert len(candidates) == 1, "expected exactly one 3x3 legend rectangle"
    return candidates[0]


def _paint_rect(out: np.ndarray, bbox: tuple[int, int, int, int], roles: tuple[int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    corner_color, edge_color, center_color = roles
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if h == 1 and w == 1:
                out[r, c] = corner_color
            elif h == 1:
                out[r, c] = corner_color if c in (c0, c1) else edge_color
            elif w == 1:
                out[r, c] = corner_color if r in (r0, r1) else edge_color
            elif r in (r0, r1) and c in (c0, c1):
                out[r, c] = corner_color
            elif r in (r0, r1) or c in (c0, c1):
                out[r, c] = edge_color
            else:
                out[r, c] = center_color


def _is_solid_monochrome_rect(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    vals = np.unique(sub)
    return vals.size == 1 and int(vals[0]) != BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique three-color legend rectangle to recolor every other solid rectangle by geometric role.

    Input structure:
    - The canvas has a black background and exactly one 3x3 multicolor legend rectangle.
    - In that legend, the four corners share one color, the four edge-midpoints share a second color,
      and the center uses a third distinct color.
    - Every other foreground component is a solid monochrome rectangle or a one-cell-wide line segment.

    Rule:
    - Keep the 3x3 legend unchanged and read it as a mapping from geometric roles to colors.
    - For each other rectangle, color its four corners with the legend's corner color, its remaining boundary
      cells with the legend's edge color, and its strict interior with the legend's center color.
    - Degenerate cases follow the same roles: 2xN rectangles have no interior, 2x2 rectangles are all corners,
      and 1xN lines use corner colors at the two ends and edge color in the middle.

    Key insight:
    - The small tricolor rectangle is a legend, not a template to copy literally: its three colors label corner,
      edge, and interior roles that must be replayed on each rectangle at that rectangle's own local scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    legend_bbox, roles = _find_legend(g)
    out = g.copy()

    transformed = 0
    for coords in _components_non_bg(g):
        bbox = _bbox(coords)
        if bbox == legend_bbox and len(coords) == 9 and _legend_roles(g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]) is not None:
            continue
        assert _is_solid_monochrome_rect(g, bbox), "every non-legend foreground object must be a solid rectangle"
        assert len(coords) == (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1), "rectangle component must fill its bbox"
        _paint_rect(out, bbox, roles)
        transformed += 1

    assert transformed >= 1, "expected at least one rectangle besides the legend"
    return out


def _sample_role_colors() -> tuple[int, int, int]:
    while True:
        roles = tuple(random.sample(LEGEND_COLOR_POOL, 3))
        if roles != CANONICAL_WRONG_ROLES:
            return roles


def _build_legend(roles: tuple[int, int, int]) -> np.ndarray:
    corner_color, edge_color, center_color = roles
    return np.array(
        [
            [corner_color, edge_color, corner_color],
            [edge_color, center_color, edge_color],
            [corner_color, edge_color, corner_color],
        ],
        dtype=int,
    )


def _sample_rect(kind: str) -> tuple[int, int]:
    if kind == "legend":
        return 3, 3
    if kind == "full":
        while True:
            h = random.randint(3, 7)
            w = random.randint(3, 8)
            if h >= 4 or w >= 4:
                return h, w
    if kind == "thin":
        length = random.randint(3, 8)
        return (2, length) if random.random() < 0.5 else (length, 2)
    if kind == "line":
        length = random.randint(4, 8)
        return (1, length) if random.random() < 0.5 else (length, 1)
    if kind == "square2":
        return 2, 2
    raise ValueError(f"unknown kind: {kind}")


def _bboxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_items(rows: int, cols: int, dims: list[tuple[str, int, int]]) -> dict[int, tuple[int, int, int, int]] | None:
    placements: dict[int, tuple[int, int, int, int]] = {}
    placed_bboxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(dims)), key=lambda i: dims[i][1] * dims[i][2], reverse=True)
    for idx in order:
        _kind, h, w = dims[idx]
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, rows - h)
            c0 = random.randint(0, cols - w)
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_bboxes_separated(bbox, other) for other in placed_bboxes):
                placements[idx] = bbox
                placed_bboxes.append(bbox)
                placed = True
                break
        if not placed:
            return None
    return placements


def _paint_input(grid: np.ndarray, legend_bbox: tuple[int, int, int, int], legend: np.ndarray, object_bboxes: list[tuple[int, int, int, int]]) -> None:
    lr0, lc0, lr1, lc1 = legend_bbox
    grid[lr0 : lr1 + 1, lc0 : lc1 + 1] = legend
    for r0, c0, r1, c1 in object_bboxes:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY


def _expected_output(grid: np.ndarray, object_bboxes: list[tuple[int, int, int, int]], roles: tuple[int, int, int]) -> np.ndarray:
    out = grid.copy()
    for bbox in object_bboxes:
        _paint_rect(out, bbox, roles)
    return out


def _wrong_tiled_output(grid: np.ndarray, object_bboxes: list[tuple[int, int, int, int]], legend: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1 in object_bboxes:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r, c] = int(legend[(r - r0) % 3, (c - c0) % 3])
    return out


def _wrong_no_center_output(grid: np.ndarray, object_bboxes: list[tuple[int, int, int, int]], roles: tuple[int, int, int]) -> np.ndarray:
    out = grid.copy()
    corner_color, edge_color, _center_color = roles
    for r0, c0, r1, c1 in object_bboxes:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r in (r0, r1) and c in (c0, c1)) or (r0 == r1 and c in (c0, c1)) or (c0 == c1 and r in (r0, r1)):
                    out[r, c] = corner_color
                else:
                    out[r, c] = edge_color
    return out


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    object_bboxes: list[tuple[int, int, int, int]],
    roles: tuple[int, int, int],
    legend: np.ndarray,
) -> bool:
    if np.array_equal(grid, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(expected, _expected_output(grid, object_bboxes, CANONICAL_WRONG_ROLES)):
        return False
    if np.array_equal(expected, _wrong_tiled_output(grid, object_bboxes, legend)):
        return False
    if np.array_equal(expected, _wrong_no_center_output(grid, object_bboxes, roles)):
        return False

    found_bbox, found_roles = _find_legend(grid)
    if found_bbox != legend_bbox or found_roles != roles:
        return False

    has_line = False
    has_thin = False
    has_interior = False
    center_color = roles[2]
    for r0, c0, r1, c1 in object_bboxes:
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == 1 or w == 1:
            has_line = True
        if min(h, w) == 2:
            has_thin = True
        if h >= 3 and w >= 3:
            has_interior = True
    if not (has_line and has_thin and has_interior):
        return False
    if not np.any(expected == center_color):
        return False

    legend_sub = expected[legend_bbox[0] : legend_bbox[2] + 1, legend_bbox[1] : legend_bbox[3] + 1]
    if not np.array_equal(legend_sub, legend):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    roles = _sample_role_colors()
    legend = _build_legend(roles)

    kinds = ["legend", "full", "thin", "line", "square2"]
    extra_count = random.randint(*OBJECT_COUNT_EXTRAS)
    for _ in range(extra_count):
        kinds.append(random.choice(("full", "thin", "line", "square2")))

    dims = [(kind, *_sample_rect(kind)) for kind in kinds]
    placements = _place_items(rows, cols, dims)
    if placements is None:
        return None

    legend_idx = kinds.index("legend")
    legend_bbox = placements[legend_idx]
    object_bboxes = [placements[i] for i in range(len(kinds)) if i != legend_idx]

    grid = np.full((rows, cols), BG, dtype=int)
    _paint_input(grid, legend_bbox, legend, object_bboxes)
    expected = _expected_output(grid, object_bboxes, roles)

    if _semantic_checks(grid, expected, legend_bbox, object_bboxes, roles, legend):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-rectangle ARC sample")
