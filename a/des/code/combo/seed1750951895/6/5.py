from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
MAROON = 9
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    offsets = NEIGHBORS_4 if connectivity == 4 else NEIGHBORS_8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _detect_frames(grid: np.ndarray) -> tuple[dict[int, tuple[int, int, int, int]], np.ndarray]:
    h, w = grid.shape
    frame_by_color: dict[int, tuple[int, int, int, int]] = {}
    frame_mask = np.zeros((h, w), dtype=bool)

    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(grid == color, connectivity=4):
            r0, c0, r1, c1 = _bbox(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 3 or ww < 3:
                continue
            if len(comp) != 2 * hh + 2 * ww - 4:
                continue

            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((hh, ww), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(patch == color, border):
                continue
            if not np.all(patch[1:-1, 1:-1] == BG):
                continue
            assert color not in frame_by_color, "each frame color must be unique"
            frame_by_color[color] = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
            frame_mask[r0 : r1 + 1, c0 : c1 + 1] |= border

    return frame_by_color, frame_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract minority-colored "keys" from multicolor objects and stamp their relative
    positions as maroon lockholes inside frames of the same color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames, frame_mask = _detect_frames(g)
    assert frames, "expected at least one frame"

    out = g.copy()
    object_mask = (g != BG) & ~frame_mask
    objects = _components(object_mask, connectivity=8)
    matched_pixels = 0

    for comp in objects:
        colors = [int(g[r, c]) for r, c in comp]
        counts = Counter(colors)
        if len(counts) < 2:
            continue
        body_color, body_count = counts.most_common(1)[0]
        assert list(counts.values()).count(body_count) == 1, "object body color must be unique"
        r0, c0, _, _ = _bbox(comp)

        for r, c in comp:
            color = int(g[r, c])
            if color == body_color:
                continue
            assert color in frames, "every minority color must match one frame"
            ir0, ic0, ir1, ic1 = frames[color]
            tr = int(ir0 + (r - r0))
            tc = int(ic0 + (c - c0))
            assert ir0 <= tr <= ir1 and ic0 <= tc <= ic1, "key offset must fit frame interior"
            out[r, c] = body_color
            out[tr, tc] = MAROON
            matched_pixels += 1

    assert matched_pixels > 0, "expected at least one minority-colored key pixel"
    return out


def _shape_full(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _shape_notch(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    r, c = random.choice(corners)
    mask[r, c] = False
    return mask


def _shape_crosscut(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    if h >= 4 and w >= 4:
        if random.random() < 0.5:
            cut_r = random.randint(1, h - 2)
            cut_c = random.randint(1, w - 2)
            mask[cut_r, cut_c] = False
            if random.random() < 0.5 and cut_c + 1 < w - 1:
                mask[cut_r, cut_c + 1] = False
        else:
            cut_r = random.randint(1, h - 2)
            cut_c = random.randint(1, w - 2)
            mask[cut_r, cut_c] = False
            if random.random() < 0.5 and cut_r + 1 < h - 1:
                mask[cut_r + 1, cut_c] = False
    return mask


def _sample_shape(h: int, w: int) -> np.ndarray:
    return random.choice([_shape_full, _shape_notch, _shape_crosscut])(h, w)


def _can_place(rect: tuple[int, int, int, int], occupied: np.ndarray, pad: int = 1) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark(rect: tuple[int, int, int, int], occupied: np.ndarray) -> None:
    r0, c0, r1, c1 = rect
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_rect(
    occupied: np.ndarray,
    height: int,
    width: int,
    tries: int = 200,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    if height > h or width > w:
        return None
    for _ in range(tries):
        r0 = random.randint(0, h - height)
        c0 = random.randint(0, w - width)
        rect = (r0, c0, r0 + height - 1, c0 + width - 1)
        if _can_place(rect, occupied, pad=1):
            _mark(rect, occupied)
            return r0, c0
    return None


def _partition_colors(frame_colors: list[int], object_count: int) -> list[list[int]]:
    groups = [[] for _ in range(object_count)]
    order = list(range(object_count))
    random.shuffle(order)
    for idx, color in enumerate(frame_colors):
        groups[order[idx % object_count]].append(color)
    for group in groups:
        assert 1 <= len(group) <= 2
    return groups


def _minority_positions(mask: np.ndarray, color_count: int) -> list[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    edges = [(r, c) for r, c in coords if r in (0, mask.shape[0] - 1) or c in (0, mask.shape[1] - 1)]
    pool = edges if len(edges) >= color_count else coords
    random.shuffle(pool)
    return pool[:color_count]


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    gh = random.randint(18, 28)
    gw = random.randint(18, 30)
    grid = np.full((gh, gw), BG, dtype=int)
    expected = grid.copy()
    occupied = np.zeros((gh, gw), dtype=bool)

    frame_count = random.randint(2, 5)
    min_objects = max(1, (frame_count + 1) // 2)
    object_count = random.randint(min_objects, min(3, frame_count))

    palette = COLOR_POOL[:]
    random.shuffle(palette)
    frame_colors = palette[:frame_count]
    body_colors = palette[frame_count : frame_count + object_count]
    if len(body_colors) < object_count:
        return _build_case()

    color_groups = _partition_colors(frame_colors, object_count)

    frame_specs: dict[int, tuple[int, int, int, int]] = {}
    for obj_idx, frame_colors_for_object in enumerate(color_groups):
        body_color = body_colors[obj_idx]
        obj_h = random.randint(3, 6)
        obj_w = random.randint(3, 6)
        shape = _sample_shape(obj_h, obj_w)
        area = int(shape.sum())
        if area <= len(frame_colors_for_object) + 1:
            return _build_case()

        pos = _place_rect(occupied, obj_h, obj_w)
        if pos is None:
            return _build_case()
        or0, oc0 = pos

        for r in range(obj_h):
            for c in range(obj_w):
                if shape[r, c]:
                    grid[or0 + r, oc0 + c] = body_color
                    expected[or0 + r, oc0 + c] = body_color

        minorities = _minority_positions(shape, len(frame_colors_for_object))
        if len(minorities) != len(frame_colors_for_object):
            return _build_case()
        for color, (rr, cc) in zip(frame_colors_for_object, minorities):
            grid[or0 + rr, oc0 + cc] = color
            expected[or0 + rr, oc0 + cc] = body_color

            inner_h = obj_h + random.randint(0, 3)
            inner_w = obj_w + random.randint(0, 3)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            frame_pos = _place_rect(occupied, outer_h, outer_w)
            if frame_pos is None:
                return _build_case()
            fr0, fc0 = frame_pos
            grid[fr0, fc0 : fc0 + outer_w] = color
            grid[fr0 + outer_h - 1, fc0 : fc0 + outer_w] = color
            grid[fr0 : fr0 + outer_h, fc0] = color
            grid[fr0 : fr0 + outer_h, fc0 + outer_w - 1] = color

            expected[fr0, fc0 : fc0 + outer_w] = color
            expected[fr0 + outer_h - 1, fc0 : fc0 + outer_w] = color
            expected[fr0 : fr0 + outer_h, fc0] = color
            expected[fr0 : fr0 + outer_h, fc0 + outer_w - 1] = color
            expected[fr0 + 1 + rr, fc0 + 1 + cc] = MAROON
            frame_specs[color] = (fr0 + 1, fc0 + 1, fr0 + inner_h, fc0 + inner_w)

    assert frame_specs
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            grid, expected = _build_case()
            produced = main(grid)
            if np.array_equal(produced, expected):
                return grid
        except AssertionError:
            pass
    raise AssertionError("failed to generate a valid puzzle instance")
