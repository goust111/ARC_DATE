from __future__ import annotations

import random
from collections import deque

import numpy as np


_BG = 0
_PALETTE = tuple(range(1, 10))
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_GEN_MAX_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components as lists of (row, column) coordinates."""
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    components: list[list[tuple[int, int]]] = []
    while unseen:
        seed = unseen.pop()
        component = [seed]
        queue = deque([seed])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _DIRECTIONS:
                neighbor = (r + dr, c + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _spiral_offsets(length: int, initial_direction: tuple[int, int]) -> list[tuple[int, int]]:
    """Offsets of a clockwise 1,1,2,2,... square spiral, excluding its origin."""
    assert length >= 0
    direction_index = _DIRECTIONS.index(initial_direction)
    r = c = 0
    offsets: list[tuple[int, int]] = []
    arm_length = 1
    while len(offsets) < length:
        for _ in range(2):
            dr, dc = _DIRECTIONS[direction_index]
            for _ in range(arm_length):
                if len(offsets) == length:
                    return offsets
                r += dr
                c += dc
                offsets.append((r, c))
            direction_index = (direction_index + 1) % 4
        arm_length += 1
    return offsets


def _parse(grid: np.ndarray) -> tuple[tuple[int, int], int, list[tuple[int, int]], int, tuple[int, int]]:
    """Identify marker, blob, their colors, and the bbox-relative heading."""
    g = np.asarray(grid)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"

    colors = [int(color) for color in np.unique(g) if int(color) != _BG]
    assert len(colors) == 2, "expected exactly two non-black foreground colors"
    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    marker_candidates = [color for color in colors if counts[color] == 1]
    assert len(marker_candidates) == 1, "exactly one foreground color must be a singleton marker"

    marker_color = marker_candidates[0]
    blob_color = next(color for color in colors if color != marker_color)
    marker_rc = tuple(map(int, np.argwhere(g == marker_color)[0]))
    blob_components = _components4(g == blob_color)
    assert len(blob_components) == 1, "the counter blob must be 4-connected"
    blob = blob_components[0]
    assert len(blob) >= 2, "the counter blob must contain multiple pixels"

    rows = [r for r, _c in blob]
    cols = [c for _r, c in blob]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mr, mc = marker_rc
    headings: list[tuple[int, int]] = []
    if c0 <= mc <= c1 and mr < r0:
        headings.append((1, 0))
    if c0 <= mc <= c1 and mr > r1:
        headings.append((-1, 0))
    if r0 <= mr <= r1 and mc < c0:
        headings.append((0, 1))
    if r0 <= mr <= r1 and mc > c1:
        headings.append((0, -1))
    assert len(headings) == 1, "marker and blob must define exactly one cardinal heading"
    return marker_rc, marker_color, blob, blob_color, headings[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Burn a counter blob into a clockwise square spiral whose heading and length it encodes.

    Input structure:
    - A black background contains exactly two foreground objects of different colors: one
      isolated single-pixel marker and one multi-pixel 4-connected solid counter blob.
    - The marker lies strictly above, below, left, or right of the blob while its column
      or row, respectively, intersects the blob's bounding box.
    - The complete encoded spiral fits inside the unchanged grid dimensions.

    Rule:
    - Erase every pixel of the blob and keep the marker.
    - Starting next to the marker in the blob's bbox-relative cardinal direction, paint
      exactly N cells in marker color, where N is the blob's actual pixel count.
    - Draw clockwise arms of lengths 1,1,2,2,3,3,... and stop as soon as N cells are painted.

    Key insight:
    - The blob is fuel rather than an obstacle or destination: its relative side fixes the
      first step, while its pixel area independently fixes the spiral's stopping point.
    """
    g = np.asarray(grid, dtype=int)
    marker, marker_color, blob, blob_color, heading = _parse(g)
    offsets = _spiral_offsets(len(blob), heading)

    out = g.copy()
    out[out == blob_color] = _BG
    mr, mc = marker
    for dr, dc in offsets:
        r, c = mr + dr, mc + dc
        assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "encoded spiral must fit in the grid"
        out[r, c] = marker_color
    return out


def _hole_free(mask: np.ndarray) -> bool:
    """Whether every background cell in the mask bbox can reach its bbox boundary."""
    empty = ~mask
    if not empty.any():
        return True
    padded = np.pad(empty, 1, constant_values=True)
    reached = np.zeros_like(padded, dtype=bool)
    reached[0, 0] = True
    queue = deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1] and padded[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                queue.append((nr, nc))
    return not bool(np.any(padded & ~reached))


def _random_blob(area: int) -> np.ndarray:
    """Grow a compact, connected, hole-free polyomino with exactly the requested area."""
    side = max(5, int(np.ceil(np.sqrt(area))) + 4)
    mask = np.zeros((side, side), dtype=bool)
    center = side // 2
    mask[center, center] = True
    while int(mask.sum()) < area:
        candidates: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in _DIRECTIONS:
                nr, nc = int(r + dr), int(c + dc)
                if 0 < nr < side - 1 and 0 < nc < side - 1 and not mask[nr, nc]:
                    candidates.add((nr, nc))
        if not candidates:
            raise RuntimeError("failed to grow counter blob")
        # Favor cells near the seed, but retain enough randomness for irregular bboxes.
        ordered = sorted(candidates, key=lambda rc: (abs(rc[0] - center) + abs(rc[1] - center), random.random()))
        pool = ordered[: max(2, len(ordered) // 2)]
        r, c = random.choice(pool)
        trial = mask.copy()
        trial[r, c] = True
        if _hole_free(trial):
            mask = trial

    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _render_attempt() -> np.ndarray | None:
    heading = random.choice(_DIRECTIONS)
    # Include arm-boundary and mid-arm stopping cases with equal probability.
    if random.random() < 0.5:
        area = random.choice((9, 12, 16, 20, 25))
    else:
        area = random.choice((7, 8, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 24))
    blob_mask = _random_blob(area)
    bh, bw = blob_mask.shape
    marker_color, blob_color = random.sample(_PALETTE, 2)

    offsets = _spiral_offsets(area, heading)
    min_dr = min([0] + [dr for dr, _ in offsets])
    max_dr = max([0] + [dr for dr, _ in offsets])
    min_dc = min([0] + [dc for _, dc in offsets])
    max_dc = max([0] + [dc for _, dc in offsets])

    margin = random.randint(1, 3)
    h = random.randint(max(15, max_dr - min_dr + 2 * margin + 1), 26)
    w = random.randint(max(15, max_dc - min_dc + 2 * margin + 1), 26)
    marker_rows = range(margin - min_dr, h - margin - max_dr)
    marker_cols = range(margin - min_dc, w - margin - max_dc)
    marker_positions = [(r, c) for r in marker_rows for c in marker_cols]
    random.shuffle(marker_positions)

    spiral_cells = {(dr, dc) for dr, dc in offsets} | {(0, 0)}
    gap = random.randint(2, 4)
    for mr, mc in marker_positions:
        placements: list[tuple[int, int]] = []
        if heading == (-1, 0):
            for local_c in range(bw):
                c0 = mc - local_c
                for r0 in range(0, mr - gap - bh + 1):
                    placements.append((r0, c0))
        elif heading == (1, 0):
            for local_c in range(bw):
                c0 = mc - local_c
                for r0 in range(mr + gap + 1, h - bh + 1):
                    placements.append((r0, c0))
        elif heading == (0, -1):
            for local_r in range(bh):
                r0 = mr - local_r
                for c0 in range(0, mc - gap - bw + 1):
                    placements.append((r0, c0))
        else:
            for local_r in range(bh):
                r0 = mr - local_r
                for c0 in range(mc + gap + 1, w - bw + 1):
                    placements.append((r0, c0))
        random.shuffle(placements)
        for r0, c0 in placements:
            if r0 < 0 or c0 < 0 or r0 + bh > h or c0 + bw > w:
                continue
            blob_cells = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(blob_mask)}
            absolute_spiral = {(mr + dr, mc + dc) for dr, dc in spiral_cells}
            if blob_cells & absolute_spiral:
                continue
            grid = np.zeros((h, w), dtype=int)
            grid[mr, mc] = marker_color
            patch = grid[r0 : r0 + bh, c0 : c0 + bw]
            patch[blob_mask] = blob_color
            return grid
    return None


def generate_input() -> np.ndarray:
    """Generate a varied, semantically validated fuel-spiral input grid."""
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _render_attempt()
        if grid is None:
            continue
        try:
            marker, marker_color, blob, blob_color, heading = _parse(grid)
            solved = main(grid)
        except AssertionError:
            continue
        expected = grid.copy()
        expected[expected == blob_color] = _BG
        mr, mc = marker
        path = _spiral_offsets(len(blob), heading)
        for dr, dc in path:
            expected[mr + dr, mc + dc] = marker_color
        if not np.array_equal(solved, expected):
            continue
        if np.any(solved == blob_color) or int(np.count_nonzero(solved == marker_color)) != len(blob) + 1:
            continue
        return grid
    raise RuntimeError("failed to generate a valid fuel-spiral input")
