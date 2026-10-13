from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
TOP = 1
BOTTOM = 2
LEFT = 3
RIGHT = 4
STAMP = 6
RECT = 7

STAMP_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
MOVE_BY_COLOR = {
    TOP: (-1, 0),
    BOTTOM: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
FRAME_COLORS = (TOP, BOTTOM, LEFT, RIGHT)
GEN_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _solid_color(line: np.ndarray) -> int:
    vals = np.unique(line)
    assert len(vals) == 1
    return int(vals[0])


def _stamp_cells(center_r: int, center_c: int) -> list[tuple[int, int]]:
    return [(center_r + dr, center_c + dc) for dr, dc in STAMP_OFFSETS]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each marker one cell toward its matching frame side, then stamp the outside crosshair inside its rectangle.

    Input structure:
    - The grid has a one-cell-thick four-color frame with black corners.
    - Inside the frame are 2 or 3 solid rectangles plus one small magenta crosshair sprite on the background.
    - Each rectangle is mostly orange and contains 1 or 2 side-colored marker pixels.

    Rule:
    - Read the frame-side colors as directions: top/up, bottom/down, left/left, right/right.
    - For each marker, restore its old cell to rectangle color, move it one cell toward the matching frame side,
      and use that moved position as the center of a 3x3 crosshair stamp.
    - Apply the stamp only inside the marker's own rectangle bbox, so any arm that would leave the rectangle is clipped.

    Key insight:
    - The border colors are directional labels, and the moved marker location, not the original one, is the center
      of the rectangle-confined crosshair imprint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 7 and w >= 7

    bg = int(g[0, 0])
    top_color = _solid_color(g[0, 1:-1])
    bottom_color = _solid_color(g[-1, 1:-1])
    left_color = _solid_color(g[1:-1, 0])
    right_color = _solid_color(g[1:-1, -1])
    directions = {
        top_color: (-1, 0),
        bottom_color: (1, 0),
        left_color: (0, -1),
        right_color: (0, 1),
    }

    interior_mask = g[1:-1, 1:-1] != bg
    components = [coords + np.array([1, 1]) for coords in _components4(interior_mask)]
    assert len(components) >= 3

    components.sort(key=len)
    stamp_component = components[0]
    stamp_colors = {int(g[r, c]) for r, c in stamp_component if int(g[r, c]) != bg}
    assert len(stamp_colors) == 1
    stamp_color = stamp_colors.pop()

    markers: list[
        tuple[int, int, int, int, int, int, tuple[int, int, int, int]]
    ] = []
    for rect_coords in components[1:]:
        rect_bbox = _bbox(rect_coords)
        rect_colors = [int(g[r, c]) for r, c in rect_coords]
        rect_color = Counter(rect_colors).most_common(1)[0][0]
        for r, c in rect_coords:
            color = int(g[r, c])
            if color in directions:
                markers.append((int(c), int(r), int(r), int(c), color, rect_color, rect_bbox))

    out = g.copy()
    for _sort_c, _sort_r, r, c, color, rect_color, rect_bbox in sorted(markers):
        r0, c0, r1, c1 = rect_bbox
        dr, dc = directions[color]
        nr, nc = r + dr, c + dc
        assert r0 <= nr <= r1 and c0 <= nc <= c1

        out[r, c] = rect_color
        out[nr, nc] = color
        for sr, sc in _stamp_cells(nr, nc):
            if r0 <= sr <= r1 and c0 <= sc <= c1:
                out[sr, sc] = stamp_color
    return out


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_rectangles(
    height: int,
    width: int,
    count: int,
) -> list[tuple[int, int, int, int]] | None:
    sizes = [(random.randint(5, 8), random.randint(5, 8)) for _ in range(count)]
    sizes.sort(key=lambda hw: hw[0] * hw[1], reverse=True)
    placed: list[tuple[int, int, int, int]] = []

    for rect_h, rect_w in sizes:
        options = []
        for r0 in range(1, height - rect_h):
            for c0 in range(1, width - rect_w):
                box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
                if all(_boxes_separated(box, other, gap=1) for other in placed):
                    options.append(box)
        if not options:
            return None
        placed.append(random.choice(options))
    return placed


def _visible_stamp_local(
    rect_h: int,
    rect_w: int,
    center_r: int,
    center_c: int,
) -> set[tuple[int, int]]:
    cells = set()
    for dr, dc in STAMP_OFFSETS:
        nr, nc = center_r + dr, center_c + dc
        if 0 <= nr < rect_h and 0 <= nc < rect_w:
            cells.add((nr, nc))
    return cells


def _marker_candidates(
    rect_h: int,
    rect_w: int,
    color: int,
    clipped: bool,
) -> list[tuple[int, int, int, int, set[tuple[int, int]]]]:
    dr, dc = MOVE_BY_COLOR[color]
    candidates = []
    for src_r in range(1, rect_h - 1):
        for src_c in range(1, rect_w - 1):
            center_r = src_r + dr
            center_c = src_c + dc
            if not (0 <= center_r < rect_h and 0 <= center_c < rect_w):
                continue
            stamp_cells = _visible_stamp_local(rect_h, rect_w, center_r, center_c)
            is_clipped = len(stamp_cells) < len(STAMP_OFFSETS)
            if is_clipped == clipped:
                candidates.append((src_r, src_c, center_r, center_c, stamp_cells))
    random.shuffle(candidates)
    return candidates


def _place_markers_for_rectangle(
    rect_h: int,
    rect_w: int,
    marker_specs: list[tuple[int, bool]],
) -> list[tuple[int, int, int]] | None:
    used_sources: set[tuple[int, int]] = set()
    used_centers: set[tuple[int, int]] = set()
    used_stamp_cells: set[tuple[int, int]] = set()
    placed: list[tuple[int, int, int]] = []

    for color, clipped in marker_specs:
        candidates = _marker_candidates(rect_h, rect_w, color, clipped)
        chosen = None
        for src_r, src_c, center_r, center_c, stamp_cells in candidates:
            src = (src_r, src_c)
            center = (center_r, center_c)
            if src in used_sources or src in used_stamp_cells:
                continue
            if center in used_centers:
                continue
            if stamp_cells & used_sources:
                continue
            if stamp_cells & used_stamp_cells:
                continue
            chosen = (src_r, src_c, color, center, stamp_cells)
            break
        if chosen is None:
            return None
        src_r, src_c, color, center, stamp_cells = chosen
        used_sources.add((src_r, src_c))
        used_centers.add(center)
        used_stamp_cells |= stamp_cells
        placed.append((src_r, src_c, color))
    return placed


def _paint_frame(grid: np.ndarray) -> None:
    grid[0, 1:-1] = TOP
    grid[-1, 1:-1] = BOTTOM
    grid[1:-1, 0] = LEFT
    grid[1:-1, -1] = RIGHT


def _cross_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    height, width = grid.shape
    positions = []
    for r0 in range(1, height - 2):
        for c0 in range(1, width - 2):
            if np.all(grid[r0 : r0 + 3, c0 : c0 + 3] == BG):
                positions.append((r0, c0))
    return positions


def _draw_cross_sprite(grid: np.ndarray, top_r: int, left_c: int) -> None:
    grid[top_r + 1, left_c] = STAMP
    grid[top_r + 1, left_c + 1] = STAMP
    grid[top_r + 1, left_c + 2] = STAMP
    grid[top_r, left_c + 1] = STAMP
    grid[top_r + 2, left_c + 1] = STAMP


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(15, 22)
    width = random.randint(15, 22)
    rect_count = random.choice((2, 3))
    rect_boxes = _place_rectangles(height, width, rect_count)
    if rect_boxes is None:
        return None

    marker_counts = [random.choice((1, 2)) for _ in range(rect_count)]
    while sum(marker_counts) < 3:
        idxs = [idx for idx, count in enumerate(marker_counts) if count == 1]
        if not idxs:
            break
        marker_counts[random.choice(idxs)] = 2
    total_markers = sum(marker_counts)

    if total_markers >= 4:
        marker_colors = list(FRAME_COLORS) + [
            random.choice(FRAME_COLORS) for _ in range(total_markers - 4)
        ]
    else:
        marker_colors = random.sample(FRAME_COLORS, total_markers)
    random.shuffle(marker_colors)

    marker_modes = [True, False] + [random.random() < 0.6 for _ in range(total_markers - 2)]
    random.shuffle(marker_modes)

    rect_specs: list[list[tuple[int, bool]]] = []
    cursor = 0
    for count in marker_counts:
        specs = list(zip(marker_colors[cursor : cursor + count], marker_modes[cursor : cursor + count]))
        rect_specs.append(specs)
        cursor += count

    grid = np.zeros((height, width), dtype=int)
    _paint_frame(grid)

    for box, specs in zip(rect_boxes, rect_specs):
        r0, c0, r1, c1 = box
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        grid[r0 : r1 + 1, c0 : c1 + 1] = RECT
        markers = _place_markers_for_rectangle(rect_h, rect_w, specs)
        if markers is None:
            return None
        for local_r, local_c, color in markers:
            grid[r0 + local_r, c0 + local_c] = color

    positions = _cross_positions(grid)
    if not positions:
        return None
    _draw_cross_sprite(grid, *random.choice(positions))

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
