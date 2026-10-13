from __future__ import annotations

from collections import deque
import random

import numpy as np


ORANGE = 7
GENERATION_RETRIES = 400
MIN_BACKGROUND_FRACTION = 0.58
PADDING = 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _row_hull_gap_count(mask: np.ndarray) -> int:
    total = 0
    for row in mask:
        cols = np.flatnonzero(row)
        if cols.size >= 2:
            total += int((~row[cols[0] : cols[-1] + 1]).sum())
    return total


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _finalize_shape(mask: np.ndarray, need_gap: bool) -> np.ndarray | None:
    if mask is None or not np.any(mask):
        return None
    mask = _trim(mask.astype(bool, copy=False))
    if len(_components(mask)) != 1:
        return None
    gap_count = _row_hull_gap_count(mask)
    if need_gap and gap_count < 2:
        return None
    if not need_gap and gap_count != 0:
        return None
    return mask


def _make_frame(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 4 or max_w < 4:
        return None
    h = rng.randint(4, max_h)
    w = rng.randint(4, max_w)
    thickness = 1
    if min(h, w) >= 7 and rng.random() < 0.35:
        thickness = 2
    if 2 * thickness >= min(h, w):
        thickness = 1
    mask = np.ones((h, w), dtype=bool)
    mask[thickness : h - thickness, thickness : w - thickness] = False
    return _finalize_shape(mask, need_gap=True)


def _make_notched_rectangle(
    rng: random.Random, max_h: int, max_w: int, from_top: bool
) -> np.ndarray | None:
    if max_h < 4 or max_w < 4:
        return None
    h = rng.randint(4, max_h)
    w = rng.randint(4, max_w)
    gap_w = rng.randint(1, w - 2)
    gap_c = rng.randint(1, w - gap_w - 1)
    depth = rng.randint(1, h - 2)
    mask = np.ones((h, w), dtype=bool)
    if from_top:
        mask[:depth, gap_c : gap_c + gap_w] = False
    else:
        mask[h - depth :, gap_c : gap_c + gap_w] = False
    return _finalize_shape(mask, need_gap=True)


def _make_double_notch(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 6 or max_w < 6:
        return None
    h = rng.randint(6, max_h)
    w = rng.randint(6, max_w)
    top_depth = rng.randint(1, h - 3)
    bottom_depth = rng.randint(1, h - top_depth - 2)
    top_gap_w = rng.randint(1, w - 2)
    bottom_gap_w = rng.randint(1, w - 2)
    top_gap_c = rng.randint(1, w - top_gap_w - 1)
    bottom_gap_c = rng.randint(1, w - bottom_gap_w - 1)
    mask = np.ones((h, w), dtype=bool)
    mask[:top_depth, top_gap_c : top_gap_c + top_gap_w] = False
    mask[h - bottom_depth :, bottom_gap_c : bottom_gap_c + bottom_gap_w] = False
    return _finalize_shape(mask, need_gap=True)


def _make_h_shape(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 5 or max_w < 5:
        return None
    h = rng.randint(5, max_h)
    w = rng.randint(5, max_w)
    left_w = rng.randint(1, min(2, w - 2))
    right_w = rng.randint(1, min(2, w - left_w - 1))
    if left_w + right_w >= w:
        return None
    bridge_h = 1 if h < 7 or rng.random() < 0.7 else 2
    bridge_r = rng.randint(1, h - bridge_h - 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :left_w] = True
    mask[:, w - right_w :] = True
    mask[bridge_r : bridge_r + bridge_h, :] = True
    return _finalize_shape(mask, need_gap=True)


def _make_rectangle(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None
    h = rng.randint(2, max_h)
    w = rng.randint(2, max_w)
    mask = np.ones((h, w), dtype=bool)
    return _finalize_shape(mask, need_gap=False)


def _make_l_shape(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 3 or max_w < 3:
        return None
    h = rng.randint(3, max_h)
    w = rng.randint(3, max_w)
    arm_w = rng.randint(1, min(3, w - 1))
    arm_h = rng.randint(1, min(3, h - 1))
    orientation = rng.randrange(4)
    mask = np.zeros((h, w), dtype=bool)
    if orientation == 0:
        mask[:, :arm_w] = True
        mask[h - arm_h :, :] = True
    elif orientation == 1:
        mask[:, w - arm_w :] = True
        mask[h - arm_h :, :] = True
    elif orientation == 2:
        mask[:, :arm_w] = True
        mask[:arm_h, :] = True
    else:
        mask[:, w - arm_w :] = True
        mask[:arm_h, :] = True
    return _finalize_shape(mask, need_gap=False)


def _sample_shape(
    rng: random.Random, need_gap: bool, max_h: int, max_w: int
) -> np.ndarray | None:
    if need_gap:
        builders = (
            _make_frame,
            lambda rr, mh, mw: _make_notched_rectangle(rr, mh, mw, True),
            lambda rr, mh, mw: _make_notched_rectangle(rr, mh, mw, False),
            _make_double_notch,
            _make_h_shape,
        )
    else:
        builders = (_make_rectangle, _make_l_shape)
    for _ in range(40):
        mask = rng.choice(builders)(rng, max_h, max_w)
        if mask is not None:
            return mask
    return None


def _valid_positions(occupied: np.ndarray, shape: np.ndarray) -> list[tuple[int, int]]:
    h, w = occupied.shape
    sh, sw = shape.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            r1 = min(h, r0 + sh + PADDING)
            c1 = min(w, c0 + sw + PADDING)
            if occupied[max(0, r0 - PADDING) : r1, max(0, c0 - PADDING) : c1].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_shape(grid: np.ndarray, shape: np.ndarray, r0: int, c0: int, color: int) -> None:
    grid[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]][shape] = color


def _place_objects(
    rng: random.Random, h: int, w: int, shapes: list[np.ndarray], color: int, bg: int
) -> np.ndarray | None:
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    for shape in sorted(shapes, key=lambda mask: mask.shape[0] * mask.shape[1], reverse=True):
        positions = _valid_positions(occupied, shape)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_shape(grid, shape, r0, c0, color)
        sh, sw = shape.shape
        occupied[max(0, r0 - PADDING) : min(h, r0 + sh + PADDING), max(0, c0 - PADDING) : min(w, c0 + sw + PADDING)] = True
    return grid


def _sample_canvas_shape(rng: random.Random, object_count: int) -> tuple[int, int]:
    min_side = {2: 10, 3: 12, 4: 14, 5: 16}[object_count]
    max_side = {2: 15, 3: 17, 4: 18, 5: 20}[object_count]
    h = rng.randint(min_side, max_side)
    w = rng.randint(min_side, max_side)
    return h, w


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(2, 5)
    h, w = _sample_canvas_shape(rng, object_count)
    available_colors = [color for color in range(10) if color != ORANGE]
    bg, fg = rng.sample(available_colors, 2)

    num_concave = rng.randint(1, object_count)
    max_h = min(h - 1, 8 if object_count <= 3 else 6)
    max_w = min(w - 1, 8 if object_count <= 3 else 6)

    shapes: list[np.ndarray] = []
    for index in range(object_count):
        need_gap = index < num_concave
        shape = _sample_shape(rng, need_gap=need_gap, max_h=max_h, max_w=max_w)
        if shape is None:
            return None
        shapes.append(shape)
    rng.shuffle(shapes)

    grid = _place_objects(rng, h, w, shapes, fg, bg)
    if grid is None:
        return None
    if float(np.mean(grid == bg)) < MIN_BACKGROUND_FRACTION:
        return None
    if np.any(grid == ORANGE):
        return None

    components = _components(grid == fg)
    if len(components) != object_count:
        return None

    out = main(grid)
    if not np.any(out == ORANGE):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    for coords in _components(g != bg):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[rows - r0, cols - c0] = True

        for dr in range(local.shape[0]):
            filled_cols = np.flatnonzero(local[dr])
            if filled_cols.size < 2:
                continue
            lo = int(filled_cols[0])
            hi = int(filled_cols[-1])
            row = r0 + dr
            segment = g[row, c0 + lo : c0 + hi + 1]
            bg_mask = segment == bg
            if np.any(bg_mask):
                out[row, c0 + lo : c0 + hi + 1][bg_mask] = ORANGE

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
