from __future__ import annotations

import random
from collections import Counter

import numpy as np


_LOCAL_TO_CORNER = {
    frozenset({(0, 0), (0, 1), (1, 0)}): "top_left",
    frozenset({(0, 0), (0, 1), (1, 1)}): "top_right",
    frozenset({(0, 0), (1, 0), (1, 1)}): "bottom_left",
    frozenset({(0, 1), (1, 0), (1, 1)}): "bottom_right",
}

_CORNER_TO_CELLS = {
    "top_left": ((0, 0), (0, 1), (1, 0)),
    "top_right": ((0, 0), (0, 1), (1, 1)),
    "bottom_left": ((0, 0), (1, 0), (1, 1)),
    "bottom_right": ((0, 1), (1, 0), (1, 1)),
}

_CORNER_POINTS = {
    "top_left": lambda h, w: (0, 0),
    "top_right": lambda h, w: (0, w - 1),
    "bottom_left": lambda h, w: (h - 1, 0),
    "bottom_right": lambda h, w: (h - 1, w - 1),
}

_GRID_SIZES = (9, 11, 13)
_MARKER_COUNTS = (3, 4, 5)
_MARKER_COLORS = (1, 2, 3, 4, 7)
_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
_MAX_ATTEMPTS = 600


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_markers(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    markers: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
            local = frozenset((rr - r0, cc - c0) for rr, cc in coords)
            assert len(coords) == 3, "each marker must contain exactly three pixels"
            assert (r1 - r0, c1 - c0) == (1, 1), "each marker must fit inside a 2x2 box"
            assert local in _LOCAL_TO_CORNER, "expected a 3-pixel L marker"
            markers.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "corner": _LOCAL_TO_CORNER[local],
                }
            )
    assert markers, "expected at least one marker"
    return markers


def _distance_to_corner(anchor: tuple[int, int], corner: str, shape: tuple[int, int]) -> int:
    r, c = anchor
    cr, cc = _CORNER_POINTS[corner](*shape)
    return abs(r - cr) + abs(c - cc)


def _arm_length(distance: int) -> int:
    return (int(distance) + 1) // 2


def _draw_corner_l(out: np.ndarray, corner: str, arm: int, color: int) -> None:
    h, w = out.shape
    arm = int(arm)
    color = int(color)
    if corner == "top_left":
        out[0, :arm] = color
        out[:arm, 0] = color
    elif corner == "top_right":
        out[0, w - arm :] = color
        out[:arm, w - 1] = color
    elif corner == "bottom_left":
        out[h - 1, :arm] = color
        out[h - arm :, 0] = color
    else:
        assert corner == "bottom_right"
        out[h - 1, w - arm :] = color
        out[h - arm :, w - 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each small L marker into a distance-sized corner L with near markers painted on top.

    Input structure:
    - The grid is an odd square on a black background.
    - Foreground objects are monochromatic 4-connected 3-pixel L markers, each
      occupying three cells of a 2x2 box.
    - The occupied cells of each small L determine which outer corner it faces.

    Rule:
    - Clear the grid to black.
    - For each marker, map its 2x2 L orientation to the corresponding outer
      corner, then measure the Manhattan distance D from the marker bbox's
      top-left cell to that corner.
    - Draw an L at that outer corner in the marker color, with both arms having
      length ceil(D / 2) including the corner pixel.
    - Paint markers in descending D order so farther markers are drawn first and
      nearer markers overwrite them where their corner Ls overlap.

    Key insight:
    - The same distance value controls two things at once: how large the corner
      expansion is and which competing marker remains visible after overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 1, "expected an odd square grid"

    bg = _most_common_color(g)
    markers = _find_markers(g, bg)

    draw_plan: list[tuple[int, int, int, int, str]] = []
    for marker in markers:
        r0, c0, _r1, _c1 = marker["bbox"]
        corner = str(marker["corner"])
        distance = _distance_to_corner((int(r0), int(c0)), corner, g.shape)
        arm = _arm_length(distance)
        draw_plan.append((distance, int(r0), int(c0), int(marker["color"]), corner))
        assert arm >= 1

    out = np.zeros_like(g)
    for distance, _r0, _c0, color, corner in sorted(draw_plan, key=lambda item: (-item[0], item[1], item[2], item[3])):
        _draw_corner_l(out, corner, _arm_length(distance), color)
    return out


def _choose_corner_multiset(marker_count: int) -> list[str]:
    if marker_count == 3:
        pattern = [2, 1]
    elif marker_count == 4:
        pattern = random.choice(([2, 1, 1], [3, 1]))
    else:
        assert marker_count == 5
        pattern = random.choice(([2, 1, 1, 1], [3, 1, 1]))

    chosen_corners = random.sample(_CORNERS, k=len(pattern))
    markers: list[str] = []
    for corner, count in zip(chosen_corners, pattern):
        markers.extend([corner] * count)
    random.shuffle(markers)
    return markers


def _box_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r0 + dr, c0 + dc) for dr in (0, 1) for dc in (0, 1)}


def _boxes_conflict(a: tuple[int, int], b: tuple[int, int]) -> bool:
    box_a = _box_cells(a)
    box_b = _box_cells(b)
    if box_a & box_b:
        return True
    for r, c in box_a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in box_b:
                return True
    return False


def _place_markers(size: int, corners: list[str]) -> list[dict[str, object]] | None:
    candidates = [(r, c) for r in range(1, size - 2) for c in range(1, size - 2)]
    random.shuffle(candidates)
    colors = random.sample(_MARKER_COLORS, k=len(corners))

    order = sorted(
        range(len(corners)),
        key=lambda idx: (-corners.count(corners[idx]), corners[idx], idx),
    )

    placed: list[dict[str, object] | None] = [None] * len(corners)
    used_positions: list[tuple[int, int]] = []
    used_distances: set[int] = set()
    used_arms_by_corner: dict[str, set[int]] = {corner: set() for corner in _CORNERS}

    def backtrack(step: int) -> bool:
        if step == len(order):
            return True

        marker_idx = order[step]
        corner = corners[marker_idx]
        local_candidates = candidates[:]
        random.shuffle(local_candidates)

        for top_left in local_candidates:
            if any(_boxes_conflict(top_left, other) for other in used_positions):
                continue

            distance = _distance_to_corner(top_left, corner, (size, size))
            if distance in used_distances:
                continue
            if not (3 <= distance <= size - 1):
                continue

            arm = _arm_length(distance)
            if arm in used_arms_by_corner[corner]:
                continue

            record = {
                "color": colors[marker_idx],
                "corner": corner,
                "top_left": top_left,
                "distance": distance,
                "arm": arm,
            }
            placed[marker_idx] = record
            used_positions.append(top_left)
            used_distances.add(distance)
            used_arms_by_corner[corner].add(arm)
            if backtrack(step + 1):
                return True
            used_arms_by_corner[corner].remove(arm)
            used_distances.remove(distance)
            used_positions.pop()
            placed[marker_idx] = None

        return False

    if not backtrack(0):
        return None
    return [item for item in placed if item is not None]


def _render_input(size: int, markers: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for marker in markers:
        r0, c0 = marker["top_left"]
        color = int(marker["color"])
        for dr, dc in _CORNER_TO_CELLS[str(marker["corner"])]:
            grid[int(r0 + dr), int(c0 + dc)] = color
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] > 30:
        return False

    markers = _find_markers(grid, 0)
    if not (3 <= len(markers) <= 5):
        return False

    colors = [int(marker["color"]) for marker in markers]
    if len(set(colors)) != len(colors):
        return False

    distances: list[int] = []
    corner_counts = Counter()
    arm_sets: dict[str, set[int]] = {corner: set() for corner in _CORNERS}
    positions = [marker["bbox"][:2] for marker in markers]

    for i, pos_a in enumerate(positions):
        for pos_b in positions[i + 1 :]:
            if _boxes_conflict((int(pos_a[0]), int(pos_a[1])), (int(pos_b[0]), int(pos_b[1]))):
                return False

    for marker in markers:
        r0, c0, _r1, _c1 = marker["bbox"]
        corner = str(marker["corner"])
        corner_counts[corner] += 1
        distance = _distance_to_corner((int(r0), int(c0)), corner, grid.shape)
        arm = _arm_length(distance)
        if distance < 3 or distance > grid.shape[0] - 1:
            return False
        if arm in arm_sets[corner]:
            return False
        arm_sets[corner].add(arm)
        distances.append(distance)

    if len(set(distances)) != len(distances):
        return False
    if max(corner_counts.values()) < 2:
        return False
    if np.array_equal(main(grid), np.zeros_like(grid)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(_GRID_SIZES)
    marker_count = random.choice(_MARKER_COUNTS)
    corners = _choose_corner_multiset(marker_count)
    markers = _place_markers(size, corners)
    if markers is None:
        return None

    grid = _render_input(size, markers)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
