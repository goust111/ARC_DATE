from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIR_UP = (-1, 0)
DIR_RIGHT = (0, 1)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIRS = (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0, "expected a nonempty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _component_to_mask(cells: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return _crop_mask(mask)


def _rotated_stamp(mask: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    rot_k = {
        DIR_UP: 0,
        DIR_RIGHT: 3,
        DIR_DOWN: 2,
        DIR_LEFT: 1,
    }[direction]
    return _crop_mask(np.rot90(mask, k=rot_k))


def _stamp_anchor(
    bbox: tuple[int, int, int, int], stamp_shape: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    ir0, ic0 = r0 + 1, c0 + 1
    ir1, ic1 = r1 - 1, c1 - 1
    ih = ir1 - ir0 + 1
    iw = ic1 - ic0 + 1
    sh, sw = map(int, stamp_shape)
    assert 1 <= sh <= ih and 1 <= sw <= iw, "stamp must fit inside the fence interior"

    if direction == DIR_RIGHT:
        return ir0 + (ih - sh) // 2, ic0
    if direction == DIR_LEFT:
        return ir0 + (ih - sh) // 2, ic1 - sw + 1
    if direction == DIR_DOWN:
        return ir0, ic0 + (iw - sw) // 2
    if direction == DIR_UP:
        return ir1 - sh + 1, ic0 + (iw - sw) // 2
    raise AssertionError(direction)


def _find_fences(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    fences: list[dict] = []
    for color in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue

            expected = set()
            for rr in range(r0, r1 + 1):
                expected.add((rr, c0))
                expected.add((rr, c1))
            for cc in range(c0, c1 + 1):
                expected.add((r0, cc))
                expected.add((r1, cc))
            comp_set = set(comp)
            if not expected.issubset(comp_set):
                continue
            extras = comp_set - expected
            if len(extras) > 1:
                continue
            if any(not (r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1) for rr, cc in extras):
                continue

            inner_pts = [
                (rr, cc)
                for rr in range(r0 + 1, r1)
                for cc in range(c0 + 1, c1)
                if int(g[rr, cc]) != int(bg)
            ]
            if len(inner_pts) != 2:
                continue

            (r_a, c_a), (r_b, c_b) = inner_pts
            if abs(r_a - r_b) + abs(c_a - c_b) != 1:
                continue

            v_a = int(g[r_a, c_a])
            v_b = int(g[r_b, c_b])
            if v_a == v_b or int(color) not in (v_a, v_b):
                continue

            if v_a == int(color):
                fence_pos = (r_a, c_a)
                marker_pos = (r_b, c_b)
                marker_color = v_b
            else:
                fence_pos = (r_b, c_b)
                marker_pos = (r_a, c_a)
                marker_color = v_a

            direction = (fence_pos[0] - marker_pos[0], fence_pos[1] - marker_pos[1])
            if direction not in DIRS:
                continue

            fences.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "fence_color": int(color),
                    "marker_color": int(marker_color),
                    "marker_cells": (
                        (int(r_a), int(c_a), int(v_a)),
                        (int(r_b), int(c_b), int(v_b)),
                    ),
                    "direction": direction,
                }
            )

    fences.sort(key=lambda item: item["bbox"])
    assert fences, "expected at least one fenced region"
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each fenced region to the outside stamp with the same marker color, rotate that stamp
    to the marker direction, and stamp it against the opposite interior wall while keeping the
    two-pixel direction marker unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    fences = _find_fences(g, bg)

    stamp_by_color: dict[int, np.ndarray] = {}
    for fence in fences:
        marker_color = int(fence["marker_color"])
        if marker_color in stamp_by_color:
            continue
        comps = _components(g == marker_color)
        assert comps, "marker color must appear in the grid"
        outside = [comp for comp in comps if len(comp) > 1]
        if not outside:
            outside = [max(comps, key=len)]
        assert len(outside) == 1, "expected a unique outside stamp for each marker color"
        stamp_by_color[marker_color] = _component_to_mask(outside[0])

    out = g.copy()
    for fence in fences:
        marker_color = int(fence["marker_color"])
        stamp = _rotated_stamp(stamp_by_color[marker_color], fence["direction"])
        top, left = _stamp_anchor(fence["bbox"], stamp.shape, fence["direction"])
        for rr, cc in np.argwhere(stamp):
            out[top + int(rr), left + int(cc)] = marker_color
        for mr, mc, value in fence["marker_cells"]:
            out[int(mr), int(mc)] = int(value)
    return out


# --------------- generator ---------------

_SHAPES = [
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
]


def _free_positions(
    occ: np.ndarray, h: int, w: int, *, padding: int = 1, border: int = 1
) -> list[tuple[int, int]]:
    rows, cols = occ.shape
    out: list[tuple[int, int]] = []
    r_end = rows - border - h
    c_end = cols - border - w
    if r_end < border or c_end < border:
        return out
    for r0 in range(border, r_end + 1):
        for c0 in range(border, c_end + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(rows, r0 + h + padding)
            cc1 = min(cols, c0 + w + padding)
            if not occ[rr0:rr1, cc0:cc1].any():
                out.append((int(r0), int(c0)))
    return out


def _mark_box(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _local_stamp_box(
    interior_shape: tuple[int, int], stamp_shape: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, int, int, int]:
    ih, iw = map(int, interior_shape)
    sh, sw = map(int, stamp_shape)
    if direction == DIR_RIGHT:
        top, left = (ih - sh) // 2, 0
    elif direction == DIR_LEFT:
        top, left = (ih - sh) // 2, iw - sw
    elif direction == DIR_DOWN:
        top, left = 0, (iw - sw) // 2
    elif direction == DIR_UP:
        top, left = ih - sh, (iw - sw) // 2
    else:
        raise AssertionError(direction)
    return int(top), int(left), int(top + sh - 1), int(left + sw - 1)


def _choose_marker_cells(
    interior_shape: tuple[int, int],
    direction: tuple[int, int],
    stamp_box: tuple[int, int, int, int],
    fence_color: int,
    marker_color: int,
) -> list[tuple[int, int, int]] | None:
    ih, iw = map(int, interior_shape)
    sr0, sc0, sr1, sc1 = stamp_box

    if direction == DIR_RIGHT:
        min_c = sc1 + 2
        max_c = iw - 2
        if min_c > max_c:
            return None
        row = random.randint(0, ih - 1)
        col = random.randint(min_c, max_c)
        return [(row, col, marker_color), (row, col + 1, fence_color)]

    if direction == DIR_LEFT:
        max_c = sc0 - 3
        if max_c < 0:
            return None
        row = random.randint(0, ih - 1)
        col = random.randint(0, max_c)
        return [(row, col, fence_color), (row, col + 1, marker_color)]

    if direction == DIR_DOWN:
        min_r = sr1 + 2
        max_r = ih - 2
        if min_r > max_r:
            return None
        row = random.randint(min_r, max_r)
        col = random.randint(0, iw - 1)
        return [(row, col, marker_color), (row + 1, col, fence_color)]

    if direction == DIR_UP:
        max_r = sr0 - 3
        if max_r < 0:
            return None
        row = random.randint(0, max_r)
        col = random.randint(0, iw - 1)
        return [(row, col, fence_color), (row + 1, col, marker_color)]

    raise AssertionError(direction)


def _draw_fence(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for rr, cc in np.argwhere(mask):
        grid[top + int(rr), left + int(cc)] = int(color)


def _render_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        stamp = _rotated_stamp(spec["shape"], spec["direction"])
        top, left = _stamp_anchor(spec["bbox"], stamp.shape, spec["direction"])
        _paint_mask(out, stamp, top, left, spec["marker_color"])
        for rr, cc, value in spec["marker_cells"]:
            out[rr, cc] = value
    return out


def _attempt_case() -> np.ndarray | None:
    n_fences = random.randint(2, 4)
    grid_min = 22 + 2 * n_fences
    rows = random.randint(grid_min, grid_min + 6)
    cols = random.randint(grid_min, grid_min + 6)

    palette = random.sample(list(range(1, 10)), 2 * n_fences)
    fence_colors = palette[:n_fences]
    marker_colors = palette[n_fences:]
    random.shuffle(fence_colors)
    random.shuffle(marker_colors)

    raw_specs: list[dict] = []
    for idx in range(n_fences):
        shape = _crop_mask(random.choice(_SHAPES))
        direction = random.choice(DIRS)
        span = max(int(shape.shape[0]), int(shape.shape[1]))
        interior_h = random.randint(span + 3, span + 6)
        interior_w = random.randint(span + 3, span + 6)
        rotated = _rotated_stamp(shape, direction)
        stamp_box = _local_stamp_box((interior_h, interior_w), rotated.shape, direction)
        marker_cells_local = _choose_marker_cells(
            (interior_h, interior_w),
            direction,
            stamp_box,
            fence_colors[idx],
            marker_colors[idx],
        )
        if marker_cells_local is None:
            return None

        raw_specs.append(
            {
                "shape": shape,
                "direction": direction,
                "interior_h": int(interior_h),
                "interior_w": int(interior_w),
                "fence_h": int(interior_h + 2),
                "fence_w": int(interior_w + 2),
                "fence_color": int(fence_colors[idx]),
                "marker_color": int(marker_colors[idx]),
                "marker_cells_local": marker_cells_local,
            }
        )

    occ = np.zeros((rows, cols), dtype=bool)
    place_order = sorted(range(n_fences), key=lambda i: raw_specs[i]["fence_h"] * raw_specs[i]["fence_w"], reverse=True)
    for idx in place_order:
        spec = raw_specs[idx]
        candidates = _free_positions(occ, spec["fence_h"], spec["fence_w"], padding=1, border=1)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        spec["bbox"] = (r0, c0, r0 + spec["fence_h"] - 1, c0 + spec["fence_w"] - 1)
        _mark_box(occ, r0, c0, spec["fence_h"], spec["fence_w"])

    stamp_order = list(range(n_fences))
    random.shuffle(stamp_order)
    for idx in stamp_order:
        spec = raw_specs[idx]
        sh, sw = map(int, spec["shape"].shape)
        candidates = _free_positions(occ, sh, sw, padding=1, border=1)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        spec["stamp_pos"] = (int(r0), int(c0))
        _mark_box(occ, r0, c0, sh, sw)

    inp = np.full((rows, cols), BG, dtype=int)
    final_specs: list[dict] = []
    for spec in raw_specs:
        r0, c0, r1, c1 = spec["bbox"]
        _draw_fence(inp, spec["bbox"], spec["fence_color"])
        marker_cells_abs: list[tuple[int, int, int]] = []
        for lr, lc, value in spec["marker_cells_local"]:
            ar = r0 + 1 + int(lr)
            ac = c0 + 1 + int(lc)
            inp[ar, ac] = int(value)
            marker_cells_abs.append((int(ar), int(ac), int(value)))
        sr, sc = spec["stamp_pos"]
        _paint_mask(inp, spec["shape"], sr, sc, spec["marker_color"])
        final_specs.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "shape": spec["shape"],
                "direction": spec["direction"],
                "fence_color": spec["fence_color"],
                "marker_color": spec["marker_color"],
                "marker_cells": marker_cells_abs,
            }
        )

    _render_expected(inp, final_specs)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        sample = _attempt_case()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid puzzle instance")
