from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
MAROON = 9
PATTERN_COLORS = (1, 2, 3, 5, 6, 7, 8)

GRID_MIN = 16
GRID_MAX = 30
FRAME_COUNTS = (1, 2, 3)
INTERIOR_HEIGHTS = tuple(range(4, 9))
INTERIOR_WIDTHS = tuple(range(4, 11))
PATTERN_MODES = ("blocks", "bands", "mixed")

GEN_RETRIES = 400
PLACEMENT_TRIES = 200
PATTERN_RETRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _frame_boxes(grid: np.ndarray, frame_color: int = MAROON) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == frame_color):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0 + 1, r1):
            border.add((r, c0))
            border.add((r, c1))

        if set(pts) != border:
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == frame_color):
            continue
        boxes.append((r0, c0, r1, c1))
    boxes.sort()
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each symmetric pattern inside a maroon frame, then add an external yellow count marker.

    Input structure:
    - Black background surrounds several one-cell-thick Maroon rectangular frames.
    - Each frame encloses a multicolor interior pattern that should be mirror-symmetric across a vertical axis.
    - One to three interior cells have been replaced by background, creating asymmetric black gaps.

    Rule:
    - For every Maroon frame, use the frame's vertical midpoint as the symmetry axis for its interior.
    - If an interior black cell has a non-black mirror counterpart across that axis, restore the missing cell with the counterpart's color.
    - Count how many cells were restored in that frame.
    - Draw that many Yellow pixels in a vertical run immediately outside the frame's right edge.

    Key insight:
    - The framed pattern and the external tally are driven by the same hidden value: the number of asymmetric mirror-gaps.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for r0, c0, r1, c1 in _frame_boxes(g):
        restored: list[tuple[int, int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if g[r, c] != BLACK:
                    continue
                mc = c0 + c1 - c
                if mc == c or mc <= c0 or mc >= c1:
                    continue
                mirror_color = int(g[r, mc])
                if mirror_color in (BLACK, YELLOW, MAROON):
                    continue
                restored.append((r, c, mirror_color))

        for r, c, color in restored:
            out[r, c] = color

        marker_col = c1 + 1
        if marker_col < w:
            for offset in range(len(restored)):
                marker_row = r0 + offset
                if marker_row < h:
                    out[marker_row, marker_col] = YELLOW

    return out


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = MAROON
    grid[r1, c0 : c1 + 1] = MAROON
    grid[r0 : r1 + 1, c0] = MAROON
    grid[r0 : r1 + 1, c1] = MAROON


def _reserve_ok(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + width + 2)
    return not occupied[r0:r1, c0:c1].any()


def _reserve(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + width + 2)
    occupied[r0:r1, c0:c1] = True


def _neighbor_support(pattern: np.ndarray, r: int, c: int) -> int:
    h, w = pattern.shape
    support = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and pattern[nr, nc] != BLACK:
            support += 1
    return support


def _paint_mirrored_block(pattern: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    pattern[top : top + height, left : left + width] = color
    mirror_left = pattern.shape[1] - left - width
    pattern[top : top + height, mirror_left : mirror_left + width] = color


def _candidate_pairs(pattern: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = pattern.shape
    out: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            color = int(pattern[r, c])
            if color == BLACK or int(pattern[r, mc]) != color:
                continue
            if _neighbor_support(pattern, r, c) == 0 or _neighbor_support(pattern, r, mc) == 0:
                continue
            out.append((r, c, mc))
    return out


def _make_symmetric_pattern(rng: random.Random, height: int, width: int, n_missing: int) -> np.ndarray | None:
    half = width // 2
    if half == 0:
        return None

    for _ in range(PATTERN_RETRIES):
        pattern = np.zeros((height, width), dtype=int)
        palette_size = rng.randint(2, 4)
        palette = rng.sample(PATTERN_COLORS, palette_size)
        mode = rng.choice(PATTERN_MODES)
        shape_count = rng.randint(4, 8)

        for idx in range(shape_count):
            color = palette[idx] if idx < len(palette) else rng.choice(palette)
            if mode == "blocks":
                block_h = rng.randint(1, min(3, height))
                block_w = rng.randint(1, min(3, half))
            elif mode == "bands":
                block_h = rng.randint(1, max(1, min(height, (height + 1) // 2)))
                block_w = rng.randint(1, half)
            else:
                if rng.random() < 0.5:
                    block_h = rng.randint(1, min(2, height))
                    block_w = rng.randint(1, min(2, half))
                else:
                    block_h = rng.randint(1, min(4, height))
                    block_w = rng.randint(1, half)

            top = rng.randint(0, height - block_h)
            left = rng.randint(0, half - block_w)
            _paint_mirrored_block(pattern, top, left, block_h, block_w, color)

        if width % 2 == 1:
            for _ in range(rng.randint(0, 2)):
                if rng.random() < 0.75:
                    top = rng.randint(0, height - 1)
                    block_h = rng.randint(1, height - top)
                    pattern[top : top + block_h, half] = rng.choice(palette)

        for _ in range(rng.randint(0, 2)):
            block_h = rng.randint(1, min(2, height))
            block_w = rng.randint(1, min(2, half))
            top = rng.randint(0, height - block_h)
            left = rng.randint(0, half - block_w)
            _paint_mirrored_block(pattern, top, left, block_h, block_w, BLACK)

        used_colors = {int(v) for v in pattern.flatten() if int(v) != BLACK}
        if len(used_colors) < 2:
            continue

        density = float(np.count_nonzero(pattern)) / float(pattern.size)
        if not (0.2 <= density <= 0.85):
            continue

        if len(_candidate_pairs(pattern)) < n_missing:
            continue
        return pattern

    return None


def _damage_pattern(rng: random.Random, full_pattern: np.ndarray, n_missing: int) -> np.ndarray | None:
    damaged = full_pattern.copy()
    pairs = _candidate_pairs(full_pattern)
    if len(pairs) < n_missing:
        return None
    for r, c, mc in rng.sample(pairs, n_missing):
        if rng.random() < 0.5:
            damaged[r, c] = BLACK
        else:
            damaged[r, mc] = BLACK
    return damaged


def _place_frames(rng: random.Random, height: int, width: int, n_frames: int) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[tuple[int, int, int, int]] = []

    for _ in range(n_frames):
        success = False
        for _ in range(PLACEMENT_TRIES):
            interior_h = rng.choice(INTERIOR_HEIGHTS)
            parity = rng.choice((0, 1))
            width_choices = [v for v in INTERIOR_WIDTHS if v % 2 == parity]
            interior_w = rng.choice(width_choices)
            frame_h = interior_h + 2
            frame_w = interior_w + 2

            max_top = height - frame_h - 1
            max_left = width - frame_w - 3
            if max_top < 1 or max_left < 1:
                continue

            top = rng.randint(1, max_top)
            left = rng.randint(1, max_left)
            if not _reserve_ok(occupied, top, left, frame_h, frame_w):
                continue

            _reserve(occupied, top, left, frame_h, frame_w)
            placed.append((top, left, interior_h, interior_w))
            success = True
            break

        if not success:
            return None

    return placed


def _expected_output(grid: np.ndarray, items: list[tuple[int, int, np.ndarray, int]]) -> np.ndarray:
    out = grid.copy()
    for top, left, full_pattern, n_missing in items:
        interior_h, interior_w = full_pattern.shape
        r0, c0 = top, left
        r1 = top + interior_h + 1
        c1 = left + interior_w + 1
        out[r0 + 1 : r1, c0 + 1 : c1] = full_pattern
        out[r0 : r0 + n_missing, c1 + 1] = YELLOW
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    n_frames = rng.choice(FRAME_COUNTS)

    placements = _place_frames(rng, height, width, n_frames)
    if placements is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    expected_items: list[tuple[int, int, np.ndarray, int]] = []

    for top, left, interior_h, interior_w in placements:
        n_missing = rng.randint(1, 3)
        full_pattern = _make_symmetric_pattern(rng, interior_h, interior_w, n_missing)
        if full_pattern is None:
            return None
        damaged_pattern = _damage_pattern(rng, full_pattern, n_missing)
        if damaged_pattern is None:
            return None

        r0, c0 = top, left
        r1 = top + interior_h + 1
        c1 = left + interior_w + 1
        _paint_frame(grid, r0, c0, r1, c1)
        grid[r0 + 1 : r1, c0 + 1 : c1] = damaged_pattern
        expected_items.append((top, left, full_pattern, n_missing))

    expected = _expected_output(grid, expected_items)
    solved = main(grid)
    if np.array_equal(grid, solved):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
