from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
PATTERN_COLORS = np.arange(2, 10, dtype=int)
MASTER_SIZE = 30
MAX_ATTEMPTS = 300
PLACEMENT_TRIES = 500
PATTERN_TRIES = 300
PAD_LIMIT = 4
DIR4 = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=int)

_RNG = np.random.default_rng()


def _rotate_clockwise(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -int(turns % 4))


def _pattern_bbox(patch: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere((patch != BLACK) & (patch != BLUE))
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _check_side(side: np.ndarray, mid: int) -> int | None:
    if np.any(side[np.arange(side.size) != mid] != BLUE):
        return None
    mid_val = int(side[mid])
    if mid_val not in (BLACK, BLUE):
        return None
    return int(mid_val == BLACK)


def _frame_gap_count(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> int | None:
    outer_h = int(r1 - r0 + 1)
    outer_w = int(c1 - c0 + 1)
    if outer_h < 5 or outer_w < 5 or outer_h % 2 == 0 or outer_w % 2 == 0:
        return None
    if (
        int(grid[r0, c0]) != BLUE
        or int(grid[r0, c1]) != BLUE
        or int(grid[r1, c0]) != BLUE
        or int(grid[r1, c1]) != BLUE
    ):
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return None
    if np.any(inner == BLUE):
        return None
    if not np.any((inner != BLACK) & (inner != BLUE)):
        return None

    mid_r = outer_h // 2
    mid_c = outer_w // 2

    gaps = 0
    for side, mid in (
        (grid[r0, c0 : c1 + 1], mid_c),
        (grid[r1, c0 : c1 + 1], mid_c),
        (grid[r0 : r1 + 1, c0], mid_r),
        (grid[r0 : r1 + 1, c1], mid_r),
    ):
        gap = _check_side(np.asarray(side), mid)
        if gap is None:
            return None
        gaps += int(gap)
    return gaps


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = map(int, grid.shape)
    frames: list[tuple[int, int, int, int, int]] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            if int(grid[r0, c0]) != BLUE:
                continue
            for r1 in range(r0 + 4, h, 2):
                if int(grid[r1, c0]) != BLUE:
                    continue
                for c1 in range(c0 + 4, w, 2):
                    if int(grid[r0, c1]) != BLUE or int(grid[r1, c1]) != BLUE:
                        continue
                    gaps = _frame_gap_count(grid, r0, c0, r1, c1)
                    if gaps is not None:
                        frames.append((int(r0), int(c0), int(r1), int(c1), int(gaps)))
    frames.sort()
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each framed inner pattern by the number of border gaps.

    Input structure:
    - The background is black.
    - Several blue one-pixel rectangular frames appear on the grid.
    - Each frame contains a non-black, non-blue asymmetric pattern, and each
      side of a frame may miss its midpoint pixel as a gap.

    Rule:
    - For every frame, count how many of its four side midpoints are missing.
    - Rotate the interior pattern clockwise by that many quarter-turns.
    - Put the rotated pattern back into the same frame while leaving the blue
      border, including its gaps, unchanged.

    Key insight:
    - The border damage is not noise: the total number of midpoint gaps encodes
      the rotation amount for the pattern inside that frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for r0, c0, r1, c1, gaps in _find_frames(g):
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        bbox = _pattern_bbox(inner)
        if bbox is None:
            continue
        pr0, pc0, pr1, pc1 = bbox
        patch = inner[pr0 : pr1 + 1, pc0 : pc1 + 1]
        rotated = _rotate_clockwise(patch, gaps)

        new_inner = np.full_like(inner, BLACK)
        top = (int(inner.shape[0]) - int(rotated.shape[0])) // 2
        left = (int(inner.shape[1]) - int(rotated.shape[1])) // 2
        new_inner[top : top + rotated.shape[0], left : left + rotated.shape[1]] = rotated
        out[r0 + 1 : r1, c0 + 1 : c1] = new_inner

    return out


def _draw_frame(canvas: np.ndarray, r0: int, c0: int, outer: int, gaps: set[int]) -> None:
    r1 = r0 + outer - 1
    c1 = c0 + outer - 1
    canvas[r0, c0 : c1 + 1] = BLUE
    canvas[r1, c0 : c1 + 1] = BLUE
    canvas[r0 : r1 + 1, c0] = BLUE
    canvas[r0 : r1 + 1, c1] = BLUE

    mid = outer // 2
    if 0 in gaps:
        canvas[r0, c0 + mid] = BLACK
    if 1 in gaps:
        canvas[r0 + mid, c1] = BLACK
    if 2 in gaps:
        canvas[r1, c0 + mid] = BLACK
    if 3 in gaps:
        canvas[r0 + mid, c0] = BLACK


def _mask_canonical_key(mask: np.ndarray) -> bytes:
    variants = [_rotate_clockwise(mask.astype(np.uint8), k).tobytes() for k in range(4)]
    return min(variants)


def _make_pattern_mask(size: int) -> np.ndarray | None:
    area_lo = max(4, size + 1)
    area_hi = size * size - 1
    for _ in range(PATTERN_TRIES):
        target = int(_RNG.integers(area_lo, area_hi + 1))
        mask = np.zeros((size, size), dtype=bool)
        start = (int(_RNG.integers(size)), int(_RNG.integers(size)))
        mask[start] = True
        frontier = [start]
        stall = 0
        while int(mask.sum()) < target and stall < target * 40:
            r, c = frontier[int(_RNG.integers(len(frontier)))]
            dr, dc = DIR4[int(_RNG.integers(len(DIR4)))]
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < size and 0 <= cc < size and not mask[rr, cc]:
                mask[rr, cc] = True
                frontier.append((rr, cc))
            stall += 1
        if int(mask.sum()) != target:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask
    return None


def _place_boxes(outer: int, n_frames: int) -> list[tuple[int, int]] | None:
    blocked = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=bool)
    positions: list[tuple[int, int]] = []
    for _ in range(n_frames):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = int(_RNG.integers(0, MASTER_SIZE - outer + 1))
            c0 = int(_RNG.integers(0, MASTER_SIZE - outer + 1))
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(MASTER_SIZE, r0 + outer + 1)
            cc1 = min(MASTER_SIZE, c0 + outer + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            positions.append((r0, c0))
            placed = True
            break
        if not placed:
            return None
    return positions


def _crop_with_padding(
    canvas: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(canvas != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_pad = int(_RNG.integers(0, min(PAD_LIMIT, int(r0)) + 1))
    left_pad = int(_RNG.integers(0, min(PAD_LIMIT, int(c0)) + 1))
    bottom_space = int(MASTER_SIZE - 1 - r1)
    right_space = int(MASTER_SIZE - 1 - c1)
    bottom_pad = int(_RNG.integers(0, min(PAD_LIMIT, bottom_space) + 1))
    right_pad = int(_RNG.integers(0, min(PAD_LIMIT, right_space) + 1))
    sr0 = int(r0) - top_pad
    sc0 = int(c0) - left_pad
    sr1 = int(r1) + bottom_pad + 1
    sc1 = int(c1) + right_pad + 1
    return canvas[sr0:sr1, sc0:sc1], (sr0, sc0, sr1, sc1)


def _gen_attempt() -> np.ndarray | None:
    inner = int(_RNG.choice([5, 7]))
    outer = inner + 2
    max_frames = 5 if inner == 5 else 4
    n_frames = int(_RNG.integers(2, max_frames + 1))
    pattern_size = int(_RNG.integers(3, inner))

    positions = _place_boxes(outer, n_frames)
    if positions is None:
        return None

    colors = [int(x) for x in _RNG.choice(PATTERN_COLORS, size=n_frames, replace=False)]
    gap_counts = [int(x) for x in _RNG.permutation(5)[:n_frames]]
    if all(count % 4 == 0 for count in gap_counts):
        idx = int(_RNG.integers(n_frames))
        gap_counts[idx] = int(_RNG.choice([1, 2, 3]))

    seen_shapes: set[bytes] = set()
    mask_specs: list[np.ndarray] = []
    for _ in range(n_frames):
        for _ in range(PATTERN_TRIES):
            mask = _make_pattern_mask(pattern_size)
            if mask is None:
                continue
            key = _mask_canonical_key(mask)
            if key in seen_shapes:
                continue
            seen_shapes.add(key)
            orientation = int(_RNG.integers(4))
            mask_specs.append(_rotate_clockwise(mask.astype(bool), orientation))
            break
        else:
            return None

    canvas = np.zeros((MASTER_SIZE, MASTER_SIZE), dtype=int)
    expected = np.zeros_like(canvas)

    offset = (inner - pattern_size) // 2
    for idx, (r0, c0) in enumerate(positions):
        count = gap_counts[idx]
        gap_sides = set(int(x) for x in _RNG.choice(4, size=count, replace=False))
        color = colors[idx]
        mask = mask_specs[idx]

        patch = np.where(mask, color, BLACK)
        top = r0 + 1 + offset
        left = c0 + 1 + offset

        _draw_frame(canvas, r0, c0, outer, gap_sides)
        _draw_frame(expected, r0, c0, outer, gap_sides)
        canvas[top : top + pattern_size, left : left + pattern_size] = patch
        expected[top : top + pattern_size, left : left + pattern_size] = _rotate_clockwise(
            patch, count
        )

    cropped, (sr0, sc0, sr1, sc1) = _crop_with_padding(canvas)
    expected_cropped = expected[sr0:sr1, sc0:sc1]

    frames = _find_frames(cropped)
    if len(frames) != n_frames:
        return None
    if sorted(g for *_bbox, g in frames) != sorted(gap_counts):
        return None
    if np.array_equal(cropped, expected_cropped):
        if not any(count % 4 for count in gap_counts):
            return None
    if not np.array_equal(main(cropped), expected_cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
