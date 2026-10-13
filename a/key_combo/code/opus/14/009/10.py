from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PINK = 6
ORANGE = 7

CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
CORNER_COUNT_TO_COLOR = {
    0: PINK,
    1: BLUE,
    2: GREEN,
    4: RED,
}


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        deltas = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        )
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _classify_mask(mask: np.ndarray) -> int:
    assert mask.shape == (3, 3)
    corner_count = sum(int(mask[r, c]) for r, c in CORNERS)
    assert corner_count in CORNER_COUNT_TO_COLOR
    return CORNER_COUNT_TO_COLOR[corner_count]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each cropped 3x3 orange object by corner occupancy and encode that class into the right marker column.

    Input structure:
    - The grid has a black background, a solid grey marker column at the far right, and 3-4 orange foreground objects elsewhere.
    - Every foreground object is a 5-pixel orange shape whose bounding box is exactly 3x3.
    - Objects are separated from one another and from the marker column by background cells.

    Rule:
    - For each orange object, crop its 3x3 bounding box and count how many of its four corners are occupied.
    - Map the corner count to a class color with the fixed code 0->6, 1->1, 2->3, 4->2.
    - Recolor the entire source object to that class color.
    - Also paint the cell in the grey marker column whose row matches the object's bounding-box center row with the same class color.

    Key insight:
    - The 3x3 crop is only an intermediate measurement device: its corner occupancy determines one class color that is then fanned out to both the object itself and a separate marker-column location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    marker_col = w - 1
    assert np.all(g[:, marker_col] == GREY)

    out = g.copy()
    orange_components = _components(g[:, :marker_col] == ORANGE, connectivity=8)
    assert 3 <= len(orange_components) <= 4

    marker_rows: set[int] = set()
    for coords in orange_components:
        assert len(coords) == 5
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3)
        mask = np.zeros((3, 3), dtype=bool)
        for r, c in coords:
            mask[r - r0, c - c0] = True
        color = _classify_mask(mask)
        out[coords[:, 0], coords[:, 1]] = color
        marker_r = r0 + 1
        assert marker_r not in marker_rows
        marker_rows.add(marker_r)
        out[marker_r, marker_col] = color

    return out


def _is_connected_3x3(mask: np.ndarray) -> bool:
    return len(_components(mask, connectivity=8)) == 1


def _enumerate_patterns() -> dict[int, list[np.ndarray]]:
    patterns = {key: [] for key in CORNER_COUNT_TO_COLOR}
    for cells in combinations(range(9), 5):
        mask = np.zeros((3, 3), dtype=bool)
        for idx in cells:
            mask[idx // 3, idx % 3] = True
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if not (len(rows) == 3 and len(cols) == 3):
            continue
        if not _is_connected_3x3(mask):
            continue
        corner_count = sum(int(mask[r, c]) for r, c in CORNERS)
        if corner_count in patterns:
            patterns[corner_count].append(mask.copy())
    return patterns


PATTERNS_BY_CORNER_COUNT = _enumerate_patterns()
COUNTS = tuple(CORNER_COUNT_TO_COLOR)


def _sample_corner_counts() -> list[int]:
    object_count = 4 if random.random() < 0.7 else 3
    return random.sample(COUNTS, k=object_count)


def _sample_canvas(object_count: int) -> tuple[int, int]:
    min_height = 15 if object_count == 4 else 13
    height = random.randint(min_height, 19)
    width = random.randint(12, 18)
    return height, width


def _sample_center_rows(height: int, object_count: int) -> list[int]:
    lanes = list(range(1, height - 1, 4))
    assert len(lanes) >= object_count
    return sorted(random.sample(lanes, k=object_count))


def _sample_left_col(width: int) -> int:
    return random.randint(0, width - 5)


def _render_object(grid: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    for r, c in np.argwhere(mask):
        grid[top + int(r), left + int(c)] = ORANGE


def _generate_candidate() -> np.ndarray:
    corner_counts = _sample_corner_counts()
    height, width = _sample_canvas(len(corner_counts))
    grid = np.zeros((height, width), dtype=int)
    grid[:, -1] = GREY

    center_rows = _sample_center_rows(height, len(corner_counts))
    for corner_count, center_row in zip(corner_counts, center_rows):
        mask = random.choice(PATTERNS_BY_CORNER_COUNT[corner_count])
        top = center_row - 1
        left = _sample_left_col(width)
        _render_object(grid, mask, top, left)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_candidate()
        try:
            out = main(grid)
        except AssertionError:
            continue
        orange_components = _components(grid[:, :-1] == ORANGE, connectivity=8)
        marker_rows = [r for r in range(out.shape[0]) if int(out[r, -1]) != GREY]
        if len(marker_rows) != len(orange_components):
            continue
        if len({int(out[r, -1]) for r in marker_rows}) != len(orange_components):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input grid")
