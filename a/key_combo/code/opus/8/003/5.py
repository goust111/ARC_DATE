from __future__ import annotations

import random

import numpy as np


BG = 0
SOURCE_COLOR = 1
FRAME_COLOR = 5
ODD_FILL = 6
EVEN_FILL = 7

CORNERS = ("TL", "TR", "BR", "BL")
TARGET_COLORS = (2, 3, 4, 8, 9)

PATTERN_TEMPLATES = (
    ((1, 1), (1, 2), (2, 1), (2, 2)),
    ((1, 1), (2, 2), (3, 3)),
    ((1, 1), (2, 2), (3, 3), (1, 2), (2, 1)),
    ((1, 1), (2, 2), (3, 3), (2, 3), (3, 2)),
    ((1, 1), (2, 2), (3, 3), (4, 4)),
    ((1, 1), (2, 2), (4, 4), (1, 2), (2, 1)),
    ((1, 1), (3, 3), (4, 4), (3, 4), (4, 3)),
)


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == FRAME_COLOR)
    if pts.size == 0:
        raise ValueError("missing grey frame")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corner_positions(frame_bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    return {
        "TL": (r0, c0),
        "TR": (r0, c1),
        "BR": (r1, c1),
        "BL": (r1, c0),
    }


def _rotate_offset(kind: str, dr: int, dc: int) -> tuple[int, int]:
    if kind == "TL":
        return dr, dc
    if kind == "TR":
        return dc, -dr
    if kind == "BR":
        return -dr, -dc
    if kind == "BL":
        return -dc, dr
    raise ValueError(kind)


def _inverse_rotate(kind: str, ar: int, ac: int) -> tuple[int, int]:
    if kind == "TL":
        return ar, ac
    if kind == "TR":
        return -ac, ar
    if kind == "BR":
        return -ar, -ac
    if kind == "BL":
        return ac, -ar
    raise ValueError(kind)


def _actual_cell(corner: tuple[int, int], kind: str, dr: int, dc: int) -> tuple[int, int]:
    rr_off, cc_off = _rotate_offset(kind, dr, dc)
    return corner[0] + rr_off, corner[1] + cc_off


def _collect_corner_pixels(
    grid: np.ndarray,
    corner: tuple[int, int],
    kind: str,
    max_offset: int = 4,
) -> list[tuple[tuple[int, int], int]]:
    items: list[tuple[tuple[int, int], int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in (BG, FRAME_COLOR):
                continue
            dr, dc = _inverse_rotate(kind, r - corner[0], c - corner[1])
            if 1 <= dr <= max_offset and 1 <= dc <= max_offset:
                items.append(((dr, dc), color))
    return items


def _chebyshev_fill(out: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    anchors = np.argwhere((out != BG) & (out != FRAME_COLOR))
    if anchors.size == 0:
        raise ValueError("missing anchor pixels")

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(out[r, c]) != BG:
                continue
            dist = min(max(abs(r - ar), abs(c - ac)) for ar, ac in anchors)
            out[r, c] = ODD_FILL if dist % 2 == 1 else EVEN_FILL
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Copy the source corner pattern to marker corners and fill the frame interior by nearest-anchor Chebyshev parity.

    Input structure:
    - The grid has a black background, one grey hollow rectangular frame, and a black interior.
    - Exactly one frame corner already contains a multi-pixel blue pattern drawn along that corner's inward diagonal direction.
    - Any subset of the other corners may contain a single non-grey marker pixel at the source pattern's deepest diagonal offset; each marker color is the color for that corner's copied pattern.

    Rule:
    - Detect the grey frame and its four corners, then identify the source corner as the only corner with more than one nearby colored interior pixel.
    - Extract the source pattern as canonical top-left-oriented offsets, and copy that pattern to every non-source marker corner using the marker's own color and the appropriate 90-degree corner rotation.
    - Treat every pattern pixel from the source and copied corners as an anchor. For each remaining black pixel strictly inside the frame, compute the minimum Chebyshev distance to the anchor set and color odd distances magenta and even distances orange.

    Key insight:
    - The corner replication is not the final answer; it first builds a distributed anchor set whose nearest-anchor Chebyshev distance field determines the interior parity coloring.
    """
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("expected a 2D grid")

    frame_bbox = _frame_bbox(grid)
    r0, c0, r1, c1 = frame_bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise ValueError("frame must have an interior")
    if not np.all(grid[r0, c0 : c1 + 1] == FRAME_COLOR):
        raise ValueError("invalid top frame edge")
    if not np.all(grid[r1, c0 : c1 + 1] == FRAME_COLOR):
        raise ValueError("invalid bottom frame edge")
    if not np.all(grid[r0 : r1 + 1, c0] == FRAME_COLOR):
        raise ValueError("invalid left frame edge")
    if not np.all(grid[r0 : r1 + 1, c1] == FRAME_COLOR):
        raise ValueError("invalid right frame edge")

    corners = _corner_positions(frame_bbox)
    per_corner = {
        kind: _collect_corner_pixels(grid, corner, kind)
        for kind, corner in corners.items()
    }
    source_candidates = [kind for kind, items in per_corner.items() if len(items) > 1]
    if len(source_candidates) != 1:
        raise ValueError("expected exactly one source corner")
    source_kind = source_candidates[0]

    source_items = per_corner[source_kind]
    source_colors = {color for _, color in source_items}
    if len(source_colors) != 1:
        raise ValueError("source pattern must be monochrome")
    pattern_offsets = {offset for offset, _ in source_items}
    diag_depths = [dr for dr, dc in pattern_offsets if dr == dc]
    if not diag_depths:
        raise ValueError("pattern must include a diagonal anchor")
    k = max(diag_depths)

    out = grid.copy()
    for kind, corner in corners.items():
        if kind == source_kind:
            continue
        marker_r, marker_c = _actual_cell(corner, kind, k, k)
        marker_color = int(grid[marker_r, marker_c])
        if marker_color in (BG, FRAME_COLOR):
            continue
        for dr, dc in pattern_offsets:
            rr, cc = _actual_cell(corner, kind, dr, dc)
            out[rr, cc] = marker_color

    return _chebyshev_fill(out, frame_bbox)


def _sample_pattern() -> tuple[set[tuple[int, int]], int]:
    pattern = set(random.choice(PATTERN_TEMPLATES))
    k = max(max(dr, dc) for dr, dc in pattern)
    return pattern, k


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = FRAME_COLOR
    grid[bottom, left : right + 1] = FRAME_COLOR
    grid[top : bottom + 1, left] = FRAME_COLOR
    grid[top : bottom + 1, right] = FRAME_COLOR


def _place_pattern(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    kind: str,
    pattern: set[tuple[int, int]],
    color: int,
    *,
    marker_only: bool,
    k: int,
) -> None:
    corner = _corner_positions(frame_bbox)[kind]
    offsets = {(k, k)} if marker_only else pattern
    for dr, dc in offsets:
        rr, cc = _actual_cell(corner, kind, dr, dc)
        grid[rr, cc] = color


def generate_input() -> np.ndarray:
    pattern, k = _sample_pattern()

    min_inner = 2 * k + 4
    inner_h = random.randint(min_inner, 22)
    inner_w = random.randint(min_inner, 22)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    grid_h = random.randint(frame_h + 2, min(frame_h + 8, 30))
    grid_w = random.randint(frame_w + 2, min(frame_w + 8, 30))
    top = random.randint(1, grid_h - frame_h - 1)
    left = random.randint(1, grid_w - frame_w - 1)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _draw_frame(grid, top, left, frame_h, frame_w)
    frame_bbox = (top, left, top + frame_h - 1, left + frame_w - 1)

    source_kind = random.choice(CORNERS)
    target_count = random.randint(1, 3)
    other_corners = [kind for kind in CORNERS if kind != source_kind]
    target_kinds = random.sample(other_corners, target_count)
    target_colors = random.sample(TARGET_COLORS, target_count)

    _place_pattern(
        grid,
        frame_bbox,
        source_kind,
        pattern,
        SOURCE_COLOR,
        marker_only=False,
        k=k,
    )
    for kind, color in zip(target_kinds, target_colors):
        _place_pattern(
            grid,
            frame_bbox,
            kind,
            pattern,
            color,
            marker_only=True,
            k=k,
        )

    return grid
