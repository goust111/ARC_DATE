from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GRID = 30
GEN_RETRIES = 200


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_points(points: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox(points)
    coords = tuple(sorted((r - r0, c - c0) for r, c in points))
    return r1 - r0 + 1, c1 - c0 + 1, coords


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    assert pts, "shape mask must be non-empty"
    return _normalize_points(pts)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=bool),
]


def _build_shape_pool() -> list[np.ndarray]:
    seen: dict[tuple[int, int, tuple[tuple[int, int], ...]], np.ndarray] = {}
    for base in BASE_SHAPES:
        for k in range(4):
            rotated = np.rot90(base, k)
            for flipped in (False, True):
                variant = np.fliplr(rotated) if flipped else rotated
                trimmed = _trim_mask(variant)
                seen.setdefault(_mask_signature(trimmed), trimmed.astype(bool))
    return list(seen.values())


SHAPE_POOL = _build_shape_pool()


def _find_separator(grid: np.ndarray) -> int:
    color_rows = np.where(np.any((grid != BG) & (grid != GREY), axis=1))[0]
    assert len(color_rows) > 0, "legend rows with colored borders are required"
    start = int(color_rows.max()) + 1
    for r in range(start, grid.shape[0]):
        if np.all(grid[r] == BG):
            return r
    raise AssertionError("missing separator row after legend")


def _legend_mapping(legend: np.ndarray) -> dict[tuple[int, int, tuple[tuple[int, int], ...]], int]:
    mapping: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for color in np.unique(legend):
        color = int(color)
        if color in (BG, GREY):
            continue
        for pts in _components(legend == color):
            r0, c0, r1, c1 = _bbox(pts)
            sub = legend[r0 : r1 + 1, c0 : c1 + 1]
            shape_mask = sub == GREY
            assert shape_mask.any(), "each legend frame must contain one grey template"
            sig = _mask_signature(shape_mask)
            assert sig not in mapping, "legend template shapes must be unique"
            mapping[sig] = color
    assert mapping, "legend mapping must be non-empty"
    return mapping


def _hole_signature(points: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    holes = []
    for bg_pts in _components(~mask):
        if all(0 < r < mask.shape[0] - 1 and 0 < c < mask.shape[1] - 1 for r, c in bg_pts):
            holes.append(bg_pts)
    assert len(holes) == 1, "each grey object must contain exactly one enclosed hole"
    return _normalize_points(holes[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each large grey object's hole shape to the grey template inside a colored legend frame,
    recolor the object with that frame color, and remove the legend zone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep = _find_separator(g)
    legend = g[:sep]
    main_zone = g[sep + 1 :].copy()

    mapping = _legend_mapping(legend)
    for pts in _components(main_zone == GREY):
        sig = _hole_signature(pts)
        assert sig in mapping, "every object hole must match a legend template"
        color = mapping[sig]
        for r, c in pts:
            main_zone[r, c] = color

    return main_zone


def _sample_templates(rng: random.Random) -> list[np.ndarray]:
    count = rng.choices([2, 3, 4], weights=[4, 4, 1])[0]
    return [shape.copy() for shape in rng.sample(SHAPE_POOL, count)]


def _build_frame(shape: np.ndarray, color: int, frame_h: int, rng: random.Random) -> np.ndarray:
    shape_h, shape_w = shape.shape
    frame_w = shape_w + 4 + rng.randint(0, 1)
    frame = np.zeros((frame_h, frame_w), dtype=int)
    frame[0, :] = color
    frame[-1, :] = color
    frame[:, 0] = color
    frame[:, -1] = color

    top = 2
    left = 2
    max_top = frame_h - shape_h - 2
    max_left = frame_w - shape_w - 2
    if max_top > top:
        top = rng.randint(top, max_top)
    if max_left > left:
        left = rng.randint(left, max_left)
    frame[top : top + shape_h, left : left + shape_w][shape] = GREY
    return frame


def _build_legend(shapes: list[np.ndarray], colors: list[int], rng: random.Random) -> np.ndarray:
    pairs = list(zip(shapes, colors))
    rng.shuffle(pairs)
    shapes = [shape for shape, _ in pairs]
    colors = [color for _, color in pairs]

    frame_h = max(shape.shape[0] for shape in shapes) + 4 + rng.randint(0, 1)
    frames = [_build_frame(shape, color, frame_h, rng) for shape, color in zip(shapes, colors)]
    gaps = [rng.randint(1, 2) for _ in range(max(0, len(frames) - 1))]
    width = sum(frame.shape[1] for frame in frames) + sum(gaps)
    legend = np.zeros((frame_h, width), dtype=int)
    x = 0
    for i, frame in enumerate(frames):
        fw = frame.shape[1]
        legend[:, x : x + fw] = frame
        x += fw
        if i < len(gaps):
            x += gaps[i]
    return legend


def _build_object(shape: np.ndarray, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    shape_h, shape_w = shape.shape
    top = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    left = rng.randint(1, 3)
    right = rng.randint(1, 3)
    obj_h = shape_h + top + bottom
    obj_w = shape_w + left + right

    inp = np.full((obj_h, obj_w), GREY, dtype=int)
    out = np.full((obj_h, obj_w), color, dtype=int)
    hole = inp[top : top + shape_h, left : left + shape_w]
    hole_out = out[top : top + shape_h, left : left + shape_w]
    hole[shape] = BG
    hole_out[shape] = BG
    return inp, out


def _build_main(
    objects_in: list[np.ndarray], objects_out: list[np.ndarray], rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    n = len(objects_in)
    row_count = 1 if n <= 2 or rng.random() < 0.45 else 2
    order = list(range(n))
    rng.shuffle(order)
    rows = [[] for _ in range(row_count)]
    for i, idx in enumerate(order):
        rows[i % row_count].append(idx)

    placements: dict[int, tuple[int, int]] = {}
    row_widths: list[int] = []
    y = 0
    for ridx, row in enumerate(rows):
        row_h = max(objects_in[idx].shape[0] for idx in row)
        x = 0
        for j, idx in enumerate(row):
            if j > 0:
                x += rng.randint(1, 3)
            obj_h, obj_w = objects_in[idx].shape
            yoff = rng.randint(0, row_h - obj_h)
            placements[idx] = (y + yoff, x)
            x += obj_w
        row_widths.append(x)
        y += row_h
        if ridx + 1 < row_count:
            y += rng.randint(1, 3)

    core_h = y
    core_w = max(row_widths)
    top_pad = rng.randint(0, 1)
    bottom_pad = rng.randint(0, 1)
    left_pad = rng.randint(0, 2)
    right_pad = rng.randint(0, 2)
    main_in = np.zeros((core_h + top_pad + bottom_pad, core_w + left_pad + right_pad), dtype=int)
    main_out = np.zeros_like(main_in)
    for idx, (r, c) in placements.items():
        obj_in = objects_in[idx]
        obj_out = objects_out[idx]
        h, w = obj_in.shape
        rr = top_pad + r
        cc = left_pad + c
        main_in[rr : rr + h, cc : cc + w] = obj_in
        main_out[rr : rr + h, cc : cc + w] = obj_out
    return main_in, main_out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_RETRIES):
        shapes = _sample_templates(rng)
        colors = rng.sample(PALETTE, len(shapes))

        legend = _build_legend(shapes, colors, rng)

        obj_count = rng.randint(len(shapes), min(len(shapes) + 3, 6))
        assignments = list(range(len(shapes)))
        while len(assignments) < obj_count:
            assignments.append(rng.randrange(len(shapes)))
        rng.shuffle(assignments)

        objects_in: list[np.ndarray] = []
        objects_out: list[np.ndarray] = []
        for idx in assignments:
            obj_in, obj_out = _build_object(shapes[idx], colors[idx], rng)
            objects_in.append(obj_in)
            objects_out.append(obj_out)

        main_in, main_out = _build_main(objects_in, objects_out, rng)
        width = max(legend.shape[1], main_in.shape[1]) + rng.randint(0, 2)
        height = legend.shape[0] + 1 + main_in.shape[0]
        if width > MAX_GRID or height > MAX_GRID:
            continue

        legend_left = rng.randint(0, width - legend.shape[1])
        main_left = rng.randint(0, width - main_in.shape[1])
        grid = np.zeros((height, width), dtype=int)
        grid[: legend.shape[0], legend_left : legend_left + legend.shape[1]] = legend
        grid[legend.shape[0] + 1 :, main_left : main_left + main_in.shape[1]] = main_in

        expected = np.zeros((main_in.shape[0], width), dtype=int)
        expected[:, main_left : main_left + main_out.shape[1]] = main_out
        if np.array_equal(main(grid), expected):
            return grid, expected
    raise RuntimeError("failed to generate a valid case")


def generate_input() -> np.ndarray:
    rng = random.Random()
    grid, _ = _generate_case(rng)
    return grid
