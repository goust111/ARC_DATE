from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
RING_PALETTE = (2, 3, 4, 6, 7, 8, 9)
FRAME_DIMENSIONS = ((3, 4), (4, 3), (4, 4), (3, 5), (5, 3))
MAX_ATTEMPTS = 500


def _chebyshev_distance(r: int, c: int, center_r: int, center_c: int) -> int:
    return max(abs(int(r) - int(center_r)), abs(int(c) - int(center_c)))


def _ring_mask(size: int, dist: int) -> np.ndarray:
    center = size // 2
    rr = np.arange(size)[:, None]
    cc = np.arange(size)[None, :]
    return np.maximum(np.abs(rr - center), np.abs(cc - center)) == dist


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_ring_map(grid: np.ndarray) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w and h % 2 == 1, "expected an odd square grid"
    center = h // 2
    assert int(g[center, center]) == BLUE, "expected a blue center marker"

    ring_map: dict[int, int] = {}
    for dist in range(1, center + 1):
        mask = _ring_mask(h, dist)
        vals = np.unique(g[mask])
        if len(vals) == 1 and int(vals[0]) not in (BLACK, GREY):
            ring_map[dist] = int(vals[0])
            continue
        if ring_map:
            break
        raise AssertionError("expected the first concentric ring at distance 1")
    assert 3 <= len(ring_map) <= 4, "expected 3-4 concentric rings"
    return ring_map


def _extract_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    center = h // 2
    frames: list[dict[str, object]] = []
    for comp in _components(g == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "grey object must be a frame"

        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        assert set(comp) == border, "grey component must be a rectangular border"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size >= 2, "frame interior must have area >= 2"
        assert np.all(interior == BLACK), "frame interior must be black in the input"

        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        dist = _chebyshev_distance(center_r, center_c, center, center)
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (height, width),
                "center": (center_r, center_c),
                "dist": dist,
            }
        )
    assert 3 <= len(frames) <= 4, "expected 3-4 frames"
    return sorted(frames, key=lambda item: int(item["dist"]))


def _resolve_offset(ring_map: dict[int, int], frame_dists: list[int]) -> int:
    ring_radii = sorted(ring_map)
    candidates = sorted({fd - radius for fd in frame_dists for radius in ring_radii if fd >= radius})
    valid: list[int] = []
    for offset in candidates:
        keys = [fd - offset for fd in frame_dists]
        if len(set(keys)) != len(keys):
            continue
        if all(key in ring_map for key in keys):
            valid.append(offset)
    assert len(valid) == 1, f"ambiguous distance alignment: {valid}"
    return int(valid[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame interior with the color indexed by its center-distance from the concentric ring palette.

    Input structure:
    - The grid is an odd square on a black background.
    - A blue center pixel is surrounded by 3-4 centered concentric square rings, each ring using one distinct non-grey color.
    - Elsewhere there are 3-4 separate one-pixel-thick grey hollow rectangular frames with black interiors.

    Rule:
    - Read the concentric rings as a distance-to-color lookup table, where ring radius identifies the palette entry.
    - For each grey frame, compute the floored center cell of its bounding box, measure that center's Chebyshev distance to the grid center, align those frame distances to the ring radii, and use the matched ring color.
    - Fill the frame's black interior with that color while leaving the grey border, the blue center, and the concentric rings unchanged.

    Key insight:
    - The rings are not the objects being transformed; they act as a read-only indexed palette, and each distant frame uses its geometric center-distance as the lookup key that determines the fill color.
    """
    g = np.asarray(grid, dtype=int)
    ring_map = _find_ring_map(g)
    frames = _extract_frames(g)
    frame_dists = [int(frame["dist"]) for frame in frames]
    offset = _resolve_offset(ring_map, frame_dists)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        key = int(frame["dist"]) - offset
        fill_color = ring_map[key]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    return out


def _draw_rings(size: int, ring_map: dict[int, int]) -> np.ndarray:
    grid = np.full((size, size), BLACK, dtype=int)
    center = size // 2
    grid[center, center] = BLUE
    for dist, color in ring_map.items():
        grid[_ring_mask(size, dist)] = int(color)
    return grid


def _frame_bbox(center_r: int, center_c: int, height: int, width: int) -> tuple[int, int, int, int]:
    r0 = center_r - (height - 1) // 2
    c0 = center_c - (width - 1) // 2
    return r0, c0, r0 + height - 1, c0 + width - 1


def _frame_cells(bbox: tuple[int, int, int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    border: list[tuple[int, int]] = []
    interior: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                border.append((r, c))
            else:
                interior.append((r, c))
    return border, interior


def _candidate_centers(size: int, dist: int) -> list[tuple[int, int]]:
    center = size // 2
    coords = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if _chebyshev_distance(r, c, center, center) == dist
    ]
    random.shuffle(coords)
    return coords


def _can_place_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    ring_count: int,
) -> bool:
    size = grid.shape[0]
    center = size // 2
    r0, c0, r1, c1 = bbox
    if r0 < 1 or c0 < 1 or r1 >= size - 1 or c1 >= size - 1:
        return False

    border, interior = _frame_cells(bbox)
    if len(interior) < 2:
        return False
    for r, c in border + interior:
        if _chebyshev_distance(r, c, center, center) <= ring_count + 2:
            return False

    pad_r0 = max(0, r0 - 1)
    pad_c0 = max(0, c0 - 1)
    pad_r1 = min(size - 1, r1 + 1)
    pad_c1 = min(size - 1, c1 + 1)
    if np.any(grid[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] != BLACK):
        return False
    return True


def _place_frame(
    grid: np.ndarray,
    center_r: int,
    center_c: int,
    height: int,
    width: int,
) -> dict[str, object]:
    bbox = _frame_bbox(center_r, center_c, height, width)
    border, _ = _frame_cells(bbox)
    for r, c in border:
        grid[r, c] = GREY
    r0, c0, r1, c1 = bbox
    return {
        "bbox": bbox,
        "shape": (height, width),
        "center": (center_r, center_c),
        "dist": _chebyshev_distance(center_r, center_c, grid.shape[0] // 2, grid.shape[0] // 2),
        "interior_slice": (slice(r0 + 1, r1), slice(c0 + 1, c1)),
    }


def _render_expected(grid: np.ndarray, ring_map: dict[int, int], frames: list[dict[str, object]], offset: int) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        key = int(frame["dist"]) - offset
        fill_color = ring_map[key]
        rs, cs = frame["interior_slice"]  # type: ignore[misc]
        out[rs, cs] = fill_color
    return out


def _semantic_check(
    grid: np.ndarray,
    expected: np.ndarray,
    ring_map: dict[int, int],
    frames: list[dict[str, object]],
    offset: int,
) -> bool:
    try:
        parsed_rings = _find_ring_map(grid)
        parsed_frames = _extract_frames(grid)
        if parsed_rings != ring_map:
            return False
        if sorted(int(frame["dist"]) for frame in parsed_frames) != sorted(int(frame["dist"]) for frame in frames):
            return False
        if sorted(tuple(frame["shape"]) for frame in parsed_frames) != sorted(tuple(frame["shape"]) for frame in frames):
            return False
        if sorted(tuple(frame["center"]) for frame in parsed_frames) != sorted(tuple(frame["center"]) for frame in frames):
            return False
        if _resolve_offset(parsed_rings, [int(frame["dist"]) for frame in parsed_frames]) != offset:
            return False
        solved = main(grid)
        if not np.array_equal(solved, expected):
            return False

        same_shape_diff_color = False
        fill_by_shape: dict[tuple[int, int], set[int]] = {}
        for frame in parsed_frames:
            r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
            fill_color = int(solved[r0 + 1, c0 + 1])
            fill_by_shape.setdefault(tuple(frame["shape"]), set()).add(fill_color)
        same_shape_diff_color = any(len(colors) >= 2 for colors in fill_by_shape.values())
        if not same_shape_diff_color:
            return False

        has_diagonal_center = any(
            frame["center"][0] != grid.shape[0] // 2 and frame["center"][1] != grid.shape[1] // 2  # type: ignore[index]
            for frame in parsed_frames
        )
        if not has_diagonal_center:
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.choice((3, 4))
    offset = random.choice((7, 8))
    max_frame_dist = ring_count + offset
    min_side = max(21, 2 * (max_frame_dist + 1) + 1)
    side_choices = [side for side in (21, 23, 25, 27, 29) if side >= min_side]
    if not side_choices:
        return None
    size = random.choice(side_choices)

    palette = list(RING_PALETTE)
    random.shuffle(palette)
    ring_map = {dist: palette[dist - 1] for dist in range(1, ring_count + 1)}
    grid = _draw_rings(size, ring_map)

    base_shape = random.choice(FRAME_DIMENSIONS)
    remaining_shapes = [random.choice(FRAME_DIMENSIONS) for _ in range(ring_count - 2)]
    frame_shapes = [base_shape, base_shape] + remaining_shapes
    random.shuffle(frame_shapes)

    distances = [offset + dist for dist in range(1, ring_count + 1)]
    random.shuffle(distances)

    planned_frames: list[dict[str, object]] = []
    for dist, (height, width) in zip(distances, frame_shapes):
        centers = _candidate_centers(size, dist)
        placed = False
        for center_r, center_c in centers:
            bbox = _frame_bbox(center_r, center_c, height, width)
            if not _can_place_frame(grid, bbox, ring_count):
                continue
            planned_frames.append(_place_frame(grid, center_r, center_c, height, width))
            placed = True
            break
        if not placed:
            return None

    expected = _render_expected(grid, ring_map, planned_frames, offset)
    if not _semantic_check(grid, expected, ring_map, planned_frames, offset):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
