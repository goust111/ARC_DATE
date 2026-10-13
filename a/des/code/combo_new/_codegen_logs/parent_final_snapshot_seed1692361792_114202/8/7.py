from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

MARKER_COLORS = (RED, BLUE, GREEN)
BORDER_COLORS = (GREY, RED, BLUE, GREEN)
PATTERN_COLORS = (4, 6, 7, 8, 9)

BASE_CANVAS = 30
MAX_GENERATION_ATTEMPTS = 300
FRAME_COUNT_CHOICES = (1, 2, 3, 4)
FRAME_COUNT_WEIGHTS = (2, 3, 2, 1)
PLACEMENT_TRIES = 160
PATTERN_TRIES = 120
MARKER_TRIES = 80


def _border_walk(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int, int]]:
    walk: list[tuple[int, int, int]] = []
    for c in range(c0, c1 + 1):
        walk.append((r0, c, 0))
    for r in range(r0 + 1, r1 + 1):
        walk.append((r, c1, 1))
    for c in range(c1 - 1, c0 - 1, -1):
        walk.append((r1, c, 0))
    for r in range(r1 - 1, r0, -1):
        walk.append((r, c0, 1))
    return walk


def _paint_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _apply_marker_ops(inner: np.ndarray, marker_ops: list[int]) -> np.ndarray:
    out = inner.copy()
    for op in marker_ops:
        out = np.fliplr(out) if op == 0 else np.flipud(out)
    return out


def _component(mask: np.ndarray, start_r: int, start_c: int, seen: np.ndarray) -> list[tuple[int, int]]:
    q: deque[tuple[int, int]] = deque([(start_r, start_c)])
    seen[start_r, start_c] = True
    pts: list[tuple[int, int]] = []
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        pts.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return pts


def _is_frame_component(grid: np.ndarray, pts: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int] | None]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False, None

    perimeter = {(r, c) for r, c, _ in _border_walk(r0, c0, r1, c1)}
    if perimeter != set(pts):
        return False, None
    for r, c in perimeter:
        if int(grid[r, c]) not in BORDER_COLORS:
            return False, None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0 or not np.any(inner != BLACK):
        return False, None
    return True, (r0, c0, r1, c1)


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = np.isin(grid, BORDER_COLORS)
    seen = np.zeros_like(mask, dtype=bool)
    frames: list[tuple[int, int, int, int]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            pts = _component(mask, r, c, seen)
            ok, bbox = _is_frame_component(grid, pts)
            if ok and bbox is not None:
                frames.append(bbox)
    frames.sort()
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each marked grey frame and mirror its interior pattern according to border-marker positions.

    Input structure:
    - The background is black.
    - One or more disjoint one-pixel-thick rectangular frames are drawn in grey, with some border cells replaced by red, blue, or green marker pixels.
    - Each frame encloses a small connected non-black pattern inside its interior area.

    Rule:
    - Traverse each frame's border clockwise from the top-left corner: top edge left-to-right, right edge top-to-bottom, bottom edge right-to-left, then left edge bottom-to-top.
    - Every marker found on the top or bottom edge applies a left-right flip to the whole interior; every marker on the left or right edge applies an up-down flip.
    - Apply the flips cumulatively in that traversal order.
    - In the output, all marker pixels are restored to grey and each frame interior is replaced by its final transformed state.

    Key insight:
    - Marker color is a distractor; only whether the marker lies on a horizontal edge or a vertical edge determines the flip.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _find_frames(g):
        inner = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        marker_ops = [
            op
            for r, c, op in _border_walk(r0, c0, r1, c1)
            if int(g[r, c]) in MARKER_COLORS
        ]
        transformed = _apply_marker_ops(inner, marker_ops)
        _paint_border(out, r0, c0, r1, c1, GREY)
        out[r0 + 1 : r1, c0 + 1 : c1] = transformed

    return out


def _random_connected_mask(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    area = h * w
    if h < 2 or w < 2 or area < 4:
        return None

    for _ in range(PATTERN_TRIES):
        target = rng.randint(3, area - 1)
        mask = np.zeros((h, w), dtype=bool)
        seed = (rng.randrange(h), rng.randrange(w))
        mask[seed] = True
        cells = [seed]
        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(tuple(frontier))
            mask[nr, nc] = True
            cells.append((nr, nc))
        if len(cells) != target:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _make_inner_pattern(rng: random.Random, inner_h: int, inner_w: int) -> np.ndarray | None:
    for _ in range(PATTERN_TRIES):
        shape_h = rng.randint(2, inner_h)
        shape_w = rng.randint(2, inner_w)
        mask = _random_connected_mask(rng, shape_h, shape_w)
        if mask is None:
            continue

        inner = np.full((inner_h, inner_w), BLACK, dtype=int)
        off_r = rng.randint(0, inner_h - shape_h)
        off_c = rng.randint(0, inner_w - shape_w)
        color = rng.choice(PATTERN_COLORS)
        inner_slice = inner[off_r : off_r + shape_h, off_c : off_c + shape_w]
        inner_slice[mask] = color

        if np.array_equal(inner, np.fliplr(inner)):
            continue
        if np.array_equal(inner, np.flipud(inner)):
            continue
        if np.array_equal(inner, np.flipud(np.fliplr(inner))):
            continue
        return inner
    return None


def _sample_markers(rng: random.Random, outer_h: int, outer_w: int) -> list[tuple[int, int, int]] | None:
    side_positions = {
        0: [(0, c) for c in range(1, outer_w - 1)],
        1: [(r, outer_w - 1) for r in range(1, outer_h - 1)],
        2: [(outer_h - 1, c) for c in range(outer_w - 2, 0, -1)],
        3: [(r, 0) for r in range(outer_h - 2, 0, -1)],
    }

    for _ in range(MARKER_TRIES):
        marker_count = rng.randint(1, 3)
        sides = sorted(rng.randrange(4) for _ in range(marker_count))
        horizontal_count = sum(side in (0, 2) for side in sides)
        vertical_count = marker_count - horizontal_count
        if horizontal_count % 2 == 0 and vertical_count % 2 == 0:
            continue

        counts = Counter(sides)
        if any(counts[side] > len(side_positions[side]) for side in counts):
            continue

        coords: list[tuple[int, int, int]] = []
        for side in range(4):
            count = counts.get(side, 0)
            if count == 0:
                continue
            picks = sorted(rng.sample(range(len(side_positions[side])), count))
            for idx in picks:
                r, c = side_positions[side][idx]
                coords.append((r, c, rng.choice(MARKER_COLORS)))
        return coords
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    pr0 = max(0, r0 - pad)
    pc0 = max(0, c0 - pad)
    pr1 = min(occupied.shape[0] - 1, r1 + pad)
    pc1 = min(occupied.shape[1] - 1, c1 + pad)
    return not occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    pr0 = max(0, r0 - pad)
    pc0 = max(0, c0 - pad)
    pr1 = min(occupied.shape[0] - 1, r1 + pad)
    pc1 = min(occupied.shape[1] - 1, c1 + pad)
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _frame_bounds_from_input(grid: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(grid != BLACK)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.full((BASE_CANVAS, BASE_CANVAS), BLACK, dtype=int)
    expected = np.full_like(grid, BLACK)
    occupied = np.zeros_like(grid, dtype=bool)

    frame_count = rng.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    max_outer = 9 if frame_count >= 3 else 11

    for _frame_idx in range(frame_count):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            outer_h = rng.randint(6, max_outer)
            outer_w = rng.randint(6, max_outer)
            r0 = rng.randint(1, BASE_CANVAS - outer_h - 1)
            c0 = rng.randint(1, BASE_CANVAS - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if not _can_place(occupied, r0, c0, r1, c1, pad=1):
                continue

            inner = _make_inner_pattern(rng, outer_h - 2, outer_w - 2)
            if inner is None:
                continue
            markers = _sample_markers(rng, outer_h, outer_w)
            if markers is None:
                continue

            _paint_border(grid, r0, c0, r1, c1, GREY)
            _paint_border(expected, r0, c0, r1, c1, GREY)
            grid[r0 + 1 : r1, c0 + 1 : c1] = inner

            for mr, mc, color in markers:
                grid[r0 + mr, c0 + mc] = color

            ops = [
                op
                for rr, cc, op in _border_walk(r0, c0, r1, c1)
                if int(grid[rr, cc]) in MARKER_COLORS
            ]
            expected[r0 + 1 : r1, c0 + 1 : c1] = _apply_marker_ops(inner, ops)

            _mark_occupied(occupied, r0, c0, r1, c1, pad=1)
            placed = True
            break
        if not placed:
            return None

    min_r, min_c, max_r, max_c = _frame_bounds_from_input(grid)
    crop_r0 = rng.randint(0, min_r)
    crop_c0 = rng.randint(0, min_c)
    crop_r1 = rng.randint(max_r, BASE_CANVAS - 1)
    crop_c1 = rng.randint(max_c, BASE_CANVAS - 1)

    cropped_inp = grid[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]
    cropped_out = expected[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]
    if not np.array_equal(main(cropped_inp), cropped_out):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-marker input")
