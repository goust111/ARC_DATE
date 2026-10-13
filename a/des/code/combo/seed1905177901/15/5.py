from __future__ import annotations

from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8_CLOCKWISE = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS8_CLOCKWISE)}

BG_COLOR = 0
INTERIOR_COLOR = 5
BORDER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400
_RNG = np.random.default_rng()


def _edge_background_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    edge_vals = np.concatenate([grid[0], grid[h - 1], grid[1 : h - 1, 0], grid[1 : h - 1, w - 1]])
    return int(Counter(map(int, edge_vals)).most_common(1)[0][0])


def _largest_component(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIRS8_CLOCKWISE if connectivity == 8 else DIRS4
    best: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if len(coords) > len(best):
                best = coords

    return np.array(best, dtype=int)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _contour_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    contour = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                    contour[r, c] = True
                    break
    return contour


def _trace_contour(mask: np.ndarray) -> list[tuple[int, int]]:
    contour = _contour_mask(mask)
    points = np.argwhere(contour)
    assert points.size > 0, "expected a non-empty contour"

    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    start = tuple(map(int, points[0]))
    backtrack = (start[0], start[1] - 1)
    backtrack0 = backtrack
    current = start
    ordered = [start]

    for _ in range(int(contour.sum()) + 16):
        step = (backtrack[0] - current[0], backtrack[1] - current[1])
        idx = DIR_TO_IDX[step]
        nxt = None
        prev_scan = None
        for k in range(8):
            j = (idx + 1 + k) % 8
            nr = current[0] + DIRS8_CLOCKWISE[j][0]
            nc = current[1] + DIRS8_CLOCKWISE[j][1]
            if 0 <= nr < contour.shape[0] and 0 <= nc < contour.shape[1] and contour[nr, nc]:
                nxt = (nr, nc)
                prev_scan = (
                    current[0] + DIRS8_CLOCKWISE[(j - 1) % 8][0],
                    current[1] + DIRS8_CLOCKWISE[(j - 1) % 8][1],
                )
                break
        assert nxt is not None and prev_scan is not None, "failed to trace contour"
        backtrack = prev_scan
        current = nxt
        if current == start and backtrack == backtrack0:
            break
        ordered.append(current)
    else:
        raise AssertionError("contour trace did not close")

    if len(ordered) != int(contour.sum()) or len(set(ordered)) != len(ordered):
        raise AssertionError("contour is not a simple single cycle")

    area2 = 0
    for (r1, c1), (r2, c2) in zip(ordered, ordered[1:] + ordered[:1]):
        area2 += c1 * r2 - c2 * r1
    if area2 < 0:
        ordered = [ordered[0]] + ordered[:0:-1]

    return ordered


def _ordered_boundary_colors(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _edge_background_color(g)
    fg = g != bg
    coords = _largest_component(fg, connectivity=8)
    assert coords.size > 0, "expected a foreground component"

    r0, c0, r1, c1 = _bbox(coords)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    comp = np.zeros_like(patch, dtype=bool)
    for r, c in coords:
        comp[int(r - r0), int(c - c0)] = True

    ordered = _trace_contour(comp)
    return np.array([int(patch[r, c]) for r, c in ordered], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    colors = _ordered_boundary_colors(grid)
    return colors.reshape(1, -1)


def _walk_monotone(start: int, end: int, length: int) -> list[int]:
    if length <= 1:
        return [int(start)]
    delta = int(end) - int(start)
    if delta == 0:
        return [int(start)] * length
    sign = 1 if delta > 0 else -1
    steps = abs(delta)
    return [int(start + sign * np.floor(i * steps / (length - 1))) for i in range(length)]


def _generate_shape_mask(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(8, 17))
    w = int(rng.integers(10, 21))
    mid = int(rng.integers(h // 3, h - h // 3))

    left_top = int(rng.integers(1, max(2, w // 4 + 1)))
    left_mid = int(rng.integers(0, left_top + 1))
    left_bottom = int(rng.integers(left_mid, max(left_mid + 1, w // 4 + 1)))

    right_top_min = max(left_top + 5, (3 * w) // 4)
    right_mid_min = max(left_mid + 5, right_top_min)
    right_bottom_min = max(left_bottom + 5, (3 * w) // 4)
    if right_top_min >= w or right_mid_min > w or right_bottom_min > w:
        return None

    right_top = int(rng.integers(right_top_min, w))
    right_mid = int(rng.integers(max(right_top, right_mid_min), w))
    right_bottom = int(rng.integers(right_bottom_min, right_mid + 1))

    left_profile = _walk_monotone(left_top, left_mid, mid + 1) + _walk_monotone(left_mid, left_bottom, h - mid)[1:]
    right_profile = _walk_monotone(right_top, right_mid, mid + 1) + _walk_monotone(right_mid, right_bottom, h - mid)[1:]

    mask = np.zeros((h, w), dtype=bool)
    for r, (left, right) in enumerate(zip(left_profile, right_profile)):
        if right - left + 1 < 5:
            return None
        mask[r, left : right + 1] = True

    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < 0.5:
        mask = np.fliplr(mask)

    contour = _contour_mask(mask)
    interior = mask & ~contour
    if int(interior.sum()) < 6:
        return None

    try:
        ordered = _trace_contour(mask)
    except AssertionError:
        return None

    if len(ordered) < 18:
        return None
    return mask


def _sample_segment_lengths(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    base = np.full(parts, 2, dtype=int)
    remaining = total - int(base.sum())
    if remaining < 0:
        raise ValueError("segment base exceeds total")
    extras = rng.multinomial(remaining, np.full(parts, 1.0 / parts))
    return base + extras


def _paint_boundary(mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    boundary_order = _trace_contour(mask)
    contour_len = len(boundary_order)

    max_segments = min(9, contour_len // 2)
    if max_segments < 3:
        raise AssertionError("contour too short")
    num_segments = int(rng.integers(3, max_segments + 1))

    lengths = _sample_segment_lengths(contour_len, num_segments, rng)
    if len(set(map(int, lengths))) == 1 and contour_len >= num_segments * 2 + 2:
        idx = int(rng.integers(num_segments))
        jdx = (idx + 1) % num_segments
        if lengths[jdx] > 2:
            lengths[idx] += 1
            lengths[jdx] -= 1

    palette_size = int(rng.integers(2, min(5, num_segments) + 1))
    palette = list(map(int, rng.choice(BORDER_COLORS, size=palette_size, replace=False)))
    shift = int(rng.integers(palette_size))
    palette = palette[shift:] + palette[:shift]
    seg_colors = [palette[i % palette_size] for i in range(num_segments)]

    patch = np.full(mask.shape, BG_COLOR, dtype=int)
    contour = _contour_mask(mask)
    patch[mask & ~contour] = INTERIOR_COLOR

    sequence: list[int] = []
    for seg_len, color in zip(lengths, seg_colors):
        sequence.extend([int(color)] * int(seg_len))
    assert len(sequence) == contour_len

    for (r, c), color in zip(boundary_order, sequence):
        patch[r, c] = int(color)

    return patch, np.array(sequence, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    mask = _generate_shape_mask(_RNG)
    if mask is None:
        return None

    obj, expected = _paint_boundary(mask, _RNG)
    obj_h, obj_w = obj.shape

    canvas_h = int(max(obj_h + 2, _RNG.integers(14, 29)))
    canvas_w = int(max(obj_w + 2, _RNG.integers(14, 29)))
    if canvas_h <= obj_h or canvas_w <= obj_w:
        return None

    top = int(_RNG.integers(1, canvas_h - obj_h))
    left = int(_RNG.integers(1, canvas_w - obj_w))

    grid = np.full((canvas_h, canvas_w), BG_COLOR, dtype=int)
    grid[top : top + obj_h, left : left + obj_w] = obj

    if len(np.unique(expected)) < 2:
        return None
    if not np.array_equal(main(grid), expected.reshape(1, -1)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
