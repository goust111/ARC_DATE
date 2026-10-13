from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
ORANGE = 7
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

FRAME_SIZE_RANGE = (11, 13)
OUTER_MARGIN_RANGE = (1, 3)
OBSTACLE_COUNT_CHOICES = (2, 3, 4)
OBSTACLE_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2))
GEN_RETRY_LIMIT = 300
MIN_VISIBLE_DIFF = 4


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _interior_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame_pixels = grid == GREY
    bbox = _bbox(frame_pixels)
    assert bbox is not None, "expected one grey hollow frame"
    r0, c0, r1, c1 = bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"
    expected = _frame_mask(grid.shape, bbox)
    assert np.array_equal(frame_pixels, expected), "grey pixels must form exactly one 1-cell-thick hollow rectangle"
    return bbox


def _distance_from_frame(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    blocked: np.ndarray,
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    dist = np.full(shape, -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for c in range(c0, c1 + 1):
        dist[r0, c] = 0
        dist[r1, c] = 0
        q.append((r0, c))
        if r1 != r0:
            q.append((r1, c))
    for r in range(r0 + 1, r1):
        dist[r, c0] = 0
        dist[r, c1] = 0
        q.append((r, c0))
        if c1 != c0:
            q.append((r, c1))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (r0 + 1 <= nr <= r1 - 1 and c0 + 1 <= nc <= c1 - 1):
                continue
            if blocked[nr, nc] or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))

    return dist


def _wave_color(distance: int) -> int:
    residue = distance % 4
    if residue == 2:
        return RED
    if residue in (1, 3):
        return GREY
    return BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the inside of a hollow frame with obstacle-avoiding BFS wave bands and erase the obstacles.

    Input structure:
    - A black background contains exactly one grey hollow rectangular frame.
    - The frame interior is otherwise black except for several orange obstacle blocks.
    - Each obstacle is fully inside the frame and does not touch the grey border.

    Rule:
    - Keep the grey frame border unchanged.
    - For every non-obstacle cell inside the frame, compute its 4-connected shortest-path distance to the nearest frame-border pixel.
    - Paths may not pass through orange obstacle pixels.
    - Recolor interior cells by distance modulo 4: 1 -> grey, 2 -> red, 3 -> grey, 0 -> black.
    - Erase the orange obstacles to black and leave all pixels outside the frame black.

    Key insight:
    - The color bands are not geometric rings around the frame; they are BFS wavefront layers that bend around the blocking obstacle pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, GREY, ORANGE}, "unexpected colors in input"

    frame_bbox = _find_frame_bbox(g)
    frame = _frame_mask(g.shape, frame_bbox)
    interior = _interior_mask(g.shape, frame_bbox)
    blocked = g == ORANGE

    assert not np.any(blocked & frame), "obstacles must stay off the frame border"
    assert np.all((g == ORANGE) <= interior), "obstacles must lie inside the frame"

    dist = _distance_from_frame(g.shape, frame_bbox, blocked)
    reachable = interior & ~blocked
    assert np.all(dist[reachable] > 0), "every non-obstacle interior pixel must reach the frame"

    out = np.full_like(g, BLACK)
    out[frame] = GREY
    for r, c in np.argwhere(reachable):
        out[r, c] = _wave_color(int(dist[r, c]))
    return out


def _paired_margin() -> tuple[int, int]:
    a = random.randint(*OUTER_MARGIN_RANGE)
    b = min(OUTER_MARGIN_RANGE[1], max(OUTER_MARGIN_RANGE[0], a + random.choice((-1, 0, 1))))
    return a, b


def _sample_frame_bbox() -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    frame_h = random.randint(*FRAME_SIZE_RANGE)
    frame_w = random.randint(*FRAME_SIZE_RANGE)
    top, bottom = _paired_margin()
    left, right = _paired_margin()
    height = top + frame_h + bottom
    width = left + frame_w + right
    bbox = (top, left, top + frame_h - 1, left + frame_w - 1)
    return (height, width), bbox


def _expanded_bbox(
    bbox: tuple[int, int, int, int],
    pad: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_obstacles(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = frame_bbox
    usable_r0, usable_c0 = r0 + 2, c0 + 2
    usable_r1, usable_c1 = r1 - 2, c1 - 2
    dims = [random.choice(OBSTACLE_SHAPES) for _ in range(count)]
    random.shuffle(dims)
    dims.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

    placed: list[tuple[int, int, int, int]] = []
    forbidden: list[tuple[int, int, int, int]] = []
    for h, w in dims:
        candidates = []
        for rr in range(usable_r0, usable_r1 - h + 2):
            for cc in range(usable_c0, usable_c1 - w + 2):
                box = (rr, cc, rr + h - 1, cc + w - 1)
                if any(_boxes_overlap(box, banned) for banned in forbidden):
                    continue
                candidates.append(box)
        if not candidates:
            return None
        box = random.choice(candidates)
        placed.append(box)
        forbidden.append(_expanded_bbox(box, 1, shape))
    return placed


def _obstacle_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in boxes:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _visible_wave_difference(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    blocked: np.ndarray,
) -> int:
    interior = _interior_mask(shape, frame_bbox)
    clear = _distance_from_frame(shape, frame_bbox, np.zeros(shape, dtype=bool))
    bent = _distance_from_frame(shape, frame_bbox, blocked)
    reachable = interior & ~blocked
    if not np.all(bent[reachable] > 0):
        return -1

    diff = 0
    for r, c in np.argwhere(reachable):
        if _wave_color(int(clear[r, c])) != _wave_color(int(bent[r, c])):
            diff += 1
    return diff


def _gen_attempt() -> np.ndarray | None:
    shape, frame_bbox = _sample_frame_bbox()
    obstacle_count = random.choice(OBSTACLE_COUNT_CHOICES)
    boxes = _place_obstacles(shape, frame_bbox, obstacle_count)
    if boxes is None:
        return None

    blocked = _obstacle_mask(shape, boxes)
    visible_diff = _visible_wave_difference(shape, frame_bbox, blocked)
    if visible_diff < MIN_VISIBLE_DIFF:
        return None

    grid = np.zeros(shape, dtype=int)
    grid[_frame_mask(shape, frame_bbox)] = GREY
    grid[blocked] = ORANGE
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRY_LIMIT} attempts")
