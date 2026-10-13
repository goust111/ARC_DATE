from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


FRAME_COLOR = 5
MARKER_COLORS = (0, 1, 2, 3, 4, 6, 7, 9)

MIN_GRID_SIZE = 18
MAX_GRID_SIZE = 30
MIN_FRAME_COUNT = 1
MAX_FRAME_COUNT = 4
FRAME_SIDES = (6, 8, 10, 12, 14)
OUTER_MARGIN = 1
FRAME_GAP = 2
PLACE_TRIES_PER_FRAME = 100
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(int(r), int(c))])
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))
            comps.append(np.array(cells, dtype=int))
    return comps


def _quadrant_slices(top: int, left: int, inner_side: int) -> tuple[tuple[slice, slice], ...]:
    half = inner_side // 2
    return (
        (slice(top, top + half), slice(left, left + half)),
        (slice(top, top + half), slice(left + half, left + inner_side)),
        (slice(top + half, top + inner_side), slice(left, left + half)),
        (slice(top + half, top + inner_side), slice(left + half, left + inner_side)),
    )


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components_of_color(grid, FRAME_COLOR):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        side = int(r1 - r0 + 1)
        if side < 4 or side != int(c1 - c0 + 1) or side % 2 != 0:
            continue
        if int(coords.shape[0]) != 4 * side - 4:
            continue

        patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        if not bool((patch[0, :] == FRAME_COLOR).all()):
            continue
        if not bool((patch[-1, :] == FRAME_COLOR).all()):
            continue
        if not bool((patch[:, 0] == FRAME_COLOR).all()):
            continue
        if not bool((patch[:, -1] == FRAME_COLOR).all()):
            continue
        if bool((patch[1:-1, 1:-1] == FRAME_COLOR).any()):
            continue

        frames.append((int(r0), int(c0), int(r1), int(c1)))

    frames.sort()
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the densest quadrant inside each square frame.

    Each grey frame encloses an even-by-even interior. The interior is split into
    four equal quadrants. The quadrant containing the most same-colored markers is
    expanded so that all background cells in that quadrant become the marker color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    frames = _find_frames(g)
    assert frames, "expected at least one grey square frame"

    for r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_side = int(inner.shape[0])
        assert inner.shape == (inner_side, inner_side)
        assert inner_side % 2 == 0 and inner_side >= 2

        marker_colors = set(map(int, np.unique(inner))) - {bg}
        assert marker_colors, "frame interior must contain markers"
        assert len(marker_colors) == 1, "all markers inside a frame must share one color"
        marker_color = marker_colors.pop()

        quadrants = _quadrant_slices(r0 + 1, c0 + 1, inner_side)
        counts: list[int] = []
        for rows, cols in quadrants:
            quad = g[rows, cols]
            other = set(map(int, np.unique(quad))) - {bg, marker_color}
            assert not other, "quadrants may only contain background and marker pixels"
            counts.append(int(np.count_nonzero(quad == marker_color)))

        best = max(counts)
        assert counts.count(best) == 1, "densest quadrant must be unique"
        winner = int(np.argmax(counts))
        rows, cols = quadrants[winner]
        patch = out[rows, cols].copy()
        patch[patch == bg] = marker_color
        out[rows, cols] = patch

    return out


def _paint_frame(grid: np.ndarray, top: int, left: int, side: int) -> None:
    grid[top : top + side, left] = FRAME_COLOR
    grid[top : top + side, left + side - 1] = FRAME_COLOR
    grid[top, left : left + side] = FRAME_COLOR
    grid[top + side - 1, left : left + side] = FRAME_COLOR


def _sample_counts(rng: random.Random, cells_per_quadrant: int) -> tuple[int, list[int]]:
    winner = rng.randrange(4)
    low = max(2, cells_per_quadrant // 2)
    high = cells_per_quadrant - 1
    if low > high:
        low = high
    winner_count = rng.randint(low, high)

    counts = [0, 0, 0, 0]
    counts[winner] = winner_count
    for idx in range(4):
        if idx == winner:
            continue
        counts[idx] = rng.randint(0, min(cells_per_quadrant, winner_count - 1))
    return winner, counts


def _place_markers(
    grid: np.ndarray,
    rng: random.Random,
    top: int,
    left: int,
    side: int,
    marker_color: int,
) -> tuple[int, list[int]]:
    inner_side = side - 2
    quadrants = _quadrant_slices(top + 1, left + 1, inner_side)
    winner, counts = _sample_counts(rng, (inner_side // 2) ** 2)

    for idx, (rows, cols) in enumerate(quadrants):
        cells = [(r, c) for r in range(rows.start, rows.stop) for c in range(cols.start, cols.stop)]
        for r, c in rng.sample(cells, counts[idx]):
            grid[r, c] = int(marker_color)

    return winner, counts


def _fits(occupied: np.ndarray, top: int, left: int, side: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, int(top - FRAME_GAP))
    c0 = max(0, int(left - FRAME_GAP))
    r1 = min(h, int(top + side + FRAME_GAP))
    c1 = min(w, int(left + side + FRAME_GAP))
    return not bool(occupied[r0:r1, c0:c1].any())


def _reserve(occupied: np.ndarray, top: int, left: int, side: int) -> None:
    h, w = occupied.shape
    r0 = max(0, int(top - FRAME_GAP))
    c0 = max(0, int(left - FRAME_GAP))
    r1 = min(h, int(top + side + FRAME_GAP))
    c1 = min(w, int(left + side + FRAME_GAP))
    occupied[r0:r1, c0:c1] = True


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, list[dict]] | None:
    bg = 8
    height = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    width = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    max_frames = MAX_FRAME_COUNT if height * width >= 450 else MAX_FRAME_COUNT - 1
    frame_count = rng.randint(MIN_FRAME_COUNT, max_frames)
    sides = [side for side in FRAME_SIDES if side <= min(height, width) - 2 * OUTER_MARGIN]
    if not sides:
        return None

    frames: list[dict] = []
    for _ in range(frame_count):
        placed = False
        for _ in range(PLACE_TRIES_PER_FRAME):
            side = int(rng.choice(sides))
            if side > height - 2 * OUTER_MARGIN or side > width - 2 * OUTER_MARGIN:
                continue
            top = rng.randint(OUTER_MARGIN, height - side - OUTER_MARGIN)
            left = rng.randint(OUTER_MARGIN, width - side - OUTER_MARGIN)
            if not _fits(occupied, int(top), int(left), int(side)):
                continue

            _paint_frame(grid, int(top), int(left), int(side))
            marker_color = int(rng.choice(MARKER_COLORS))
            winner, counts = _place_markers(grid, rng, int(top), int(left), int(side), marker_color)
            _reserve(occupied, int(top), int(left), int(side))
            frames.append(
                {
                    "top": int(top),
                    "left": int(left),
                    "side": int(side),
                    "marker_color": int(marker_color),
                    "winner": int(winner),
                    "counts": tuple(int(x) for x in counts),
                }
            )
            placed = True
            break
        if not placed:
            return None

    return grid, frames


def _expected_output(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    bg = _background_color(grid)
    out = np.array(grid, copy=True)
    for frame in frames:
        quadrants = _quadrant_slices(
            int(frame["top"]) + 1,
            int(frame["left"]) + 1,
            int(frame["side"]) - 2,
        )
        rows, cols = quadrants[int(frame["winner"])]
        patch = out[rows, cols].copy()
        patch[patch == bg] = int(frame["marker_color"])
        out[rows, cols] = patch
    return out


def _generate_case(rng: random.Random, return_meta: bool = False):
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is None:
            continue
        if return_meta:
            return result
        return result[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")


def generate_input() -> np.ndarray:
    return _generate_case(random.Random(), return_meta=False)
