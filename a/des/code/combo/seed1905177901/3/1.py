from __future__ import annotations

from collections import deque
import random

import numpy as np


AZURE = 8
GREY = 5
PALETTE_POOL = [0, 1, 2, 3, 4, 6, 7, 9]

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
LEGEND_LEN_RANGE = (1, 4)
OBJECT_PADDING = 1
OBJECT_DIM_MAX = 12
GENERATION_ATTEMPTS = 400
OBJECT_SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 250


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _patch_from_component(points: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(points)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        patch[r - r0, c - c0] = True
    return patch, r0, c0


def _hole_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    holes: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for region in _components(~mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in region):
            continue
        holes.append(region)
    return holes


def _dominant_color(values: np.ndarray) -> int:
    counts = np.bincount(values.ravel(), minlength=10)
    return int(np.argmax(counts))


def _object_color(grid_wo_legend: np.ndarray, bg: int) -> int:
    vals, cnts = np.unique(grid_wo_legend[grid_wo_legend != bg], return_counts=True)
    assert len(vals) > 0
    return int(vals[int(np.argmax(cnts))])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey object's enclosed windows using the bottom legend row.

    The kth non-background color on the last row is the fill color for objects
    that contain exactly k enclosed background holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape
    assert h >= 2

    bg = _dominant_color(g)
    legend = [int(v) for v in g[-1].tolist() if int(v) != bg]
    assert legend

    work = g[:-1]
    obj_color = _object_color(work, bg)
    out = g.copy()

    for comp in _components(work == obj_color):
        patch, r0, c0 = _patch_from_component(comp)
        holes = _hole_regions(patch)
        if not holes:
            continue
        count = len(holes)
        assert count <= len(legend)
        fill_color = int(legend[count - 1])
        for hole in holes:
            for rr, cc in hole:
                out[r0 + rr, c0 + cc] = fill_color

    out[-1, :] = bg
    return out


def _split_positive(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    out: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        out.append(int(cut - prev))
        prev = cut
    return out


def _base_object_mask(hole_count: int) -> np.ndarray:
    if hole_count == 0:
        return np.ones(
            (
                random.randint(3, 6),
                random.randint(3, 6),
            ),
            dtype=bool,
        )

    row_count = random.randint(1, min(3, hole_count))
    hole_rows = _split_positive(hole_count, row_count)
    border = random.randint(1, 2)
    col_gap = random.randint(1, 2)
    row_gap = random.randint(1, 2)

    row_specs: list[list[tuple[int, int]]] = []
    row_heights: list[int] = []
    row_widths: list[int] = []
    for holes_in_row in hole_rows:
        sizes = []
        for _ in range(holes_in_row):
            hh = random.randint(1, 2)
            ww = random.randint(1, 3)
            sizes.append((hh, ww))
        row_specs.append(sizes)
        row_heights.append(max(hh for hh, _ in sizes))
        row_widths.append(sum(ww for _, ww in sizes) + col_gap * (holes_in_row - 1))

    inner_h = sum(row_heights) + row_gap * (row_count - 1)
    inner_w = max(row_widths)
    patch = np.ones((inner_h + 2 * border, inner_w + 2 * border), dtype=bool)

    cur_r = border
    for row_h, row_w, sizes in zip(row_heights, row_widths, row_specs, strict=False):
        cur_c = border + random.randint(0, inner_w - row_w)
        for hh, ww in sizes:
            top = cur_r + random.randint(0, row_h - hh)
            patch[top : top + hh, cur_c : cur_c + ww] = False
            cur_c += ww + col_gap
        cur_r += row_h + row_gap
    return patch


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _add_protrusions(mask: np.ndarray) -> np.ndarray:
    pad = 6
    h, w = mask.shape
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    canvas[pad : pad + h, pad : pad + w] = mask

    for _ in range(random.randint(0, 3)):
        coords = np.argwhere(canvas)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            width = int(c1 - c0 + 1)
            span = random.randint(1, min(4, width))
            depth = random.randint(1, 2)
            start = random.randint(int(c0), int(c1) - span + 1)
            if side == "top":
                if int(r0) - depth < 0:
                    continue
                canvas[int(r0) - depth : int(r0), start : start + span] = True
            else:
                if int(r1) + depth >= canvas.shape[0]:
                    continue
                canvas[int(r1) + 1 : int(r1) + depth + 1, start : start + span] = True
        else:
            height = int(r1 - r0 + 1)
            span = random.randint(1, min(4, height))
            depth = random.randint(1, 2)
            start = random.randint(int(r0), int(r1) - span + 1)
            if side == "left":
                if int(c0) - depth < 0:
                    continue
                canvas[start : start + span, int(c0) - depth : int(c0)] = True
            else:
                if int(c1) + depth >= canvas.shape[1]:
                    continue
                canvas[start : start + span, int(c1) + 1 : int(c1) + depth + 1] = True

    return _crop_mask(canvas)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.array(out, dtype=bool)


def _make_object(hole_count: int) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    for _ in range(OBJECT_SHAPE_ATTEMPTS):
        mask = _base_object_mask(hole_count)
        mask = _add_protrusions(mask)
        mask = _random_transform(mask)
        if mask.shape[0] > OBJECT_DIM_MAX or mask.shape[1] > OBJECT_DIM_MAX:
            continue
        if len(_components(mask)) != 1:
            continue
        holes = _hole_regions(mask)
        if len(holes) != hole_count:
            continue
        return mask, holes
    raise RuntimeError("failed to build object with requested hole count")


def _sample_hole_counts(legend_len: int) -> list[int]:
    positive = list(range(1, legend_len + 1))
    if random.random() < 0.5:
        chosen = positive
    else:
        chosen = sorted(random.sample(positive, random.randint(1, legend_len)))

    counts = [0] + chosen
    target_n = random.randint(len(counts), min(6, len(counts) + 2))
    while len(counts) < target_n:
        if random.random() < 0.5:
            counts.append(0)
        else:
            counts.append(random.choice(chosen))
    random.shuffle(counts)
    return counts


def _legend_layout(width: int, legend_len: int) -> tuple[list[int], int]:
    step = random.choice([1, 2])
    span = 1 + step * (legend_len - 1)
    start = random.randint(1, width - span - 1)
    cols = [start + i * step for i in range(legend_len)]
    return cols, span


def _paste_object(
    grid: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
) -> None:
    h, w = mask.shape
    region = grid[top : top + h, left : left + w]
    region[mask] = int(color)


def _try_place_objects(
    height: int,
    width: int,
    objects: list[tuple[np.ndarray, list[list[tuple[int, int]]], int]],
) -> list[tuple[int, int]] | None:
    usable_h = height - 1
    blocked = np.zeros((usable_h, width), dtype=bool)
    placements: list[tuple[int, int]] = []

    order = list(range(len(objects)))
    order.sort(
        key=lambda i: (
            -(objects[i][0].shape[0] * objects[i][0].shape[1]),
            -objects[i][2],
        )
    )

    pos_by_index: dict[int, tuple[int, int]] = {}
    for idx in order:
        mask, _holes, _count = objects[idx]
        h, w = mask.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, usable_h - h - 1)
            left = random.randint(0, width - w)
            r0 = max(0, top - OBJECT_PADDING)
            c0 = max(0, left - OBJECT_PADDING)
            r1 = min(usable_h, top + h + OBJECT_PADDING)
            c1 = min(width, left + w + OBJECT_PADDING)
            if bool(blocked[r0:r1, c0:c1].any()):
                continue
            blocked[r0:r1, c0:c1] = True
            pos_by_index[idx] = (int(top), int(left))
            placed = True
            break
        if not placed:
            return None

    for idx in range(len(objects)):
        placements.append(pos_by_index[idx])
    return placements


def _expected_output(
    inp: np.ndarray,
    objects: list[tuple[np.ndarray, list[list[tuple[int, int]]], int]],
    placements: list[tuple[int, int]],
    legend_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    out[-1, :] = AZURE
    for (mask, holes, count), (top, left) in zip(objects, placements, strict=False):
        if count == 0:
            continue
        fill = int(legend_colors[count - 1])
        for hole in holes:
            for r, c in hole:
                out[top + r, left + c] = fill
    return out


def _build_sample() -> tuple[np.ndarray, np.ndarray]:
    legend_len = random.randint(*LEGEND_LEN_RANGE)
    legend_colors = random.sample(PALETTE_POOL, legend_len)
    hole_counts = _sample_hole_counts(legend_len)

    objects = []
    for count in hole_counts:
        mask, holes = _make_object(count)
        objects.append((mask, holes, count))

    max_h = max(mask.shape[0] for mask, _, _ in objects)
    max_w = max(mask.shape[1] for mask, _, _ in objects)
    padded_area = sum(
        (mask.shape[0] + 2 * OBJECT_PADDING) * (mask.shape[1] + 2 * OBJECT_PADDING)
        for mask, _, _ in objects
    )

    legend_cols_hint, legend_span = _legend_layout(max(GRID_W_RANGE[0], 10), legend_len)
    _ = legend_cols_hint
    min_h = max(GRID_H_RANGE[0], max_h + 3, int(np.ceil(np.sqrt(padded_area))) - 2)
    min_w = max(GRID_W_RANGE[0], max(max_w + 2, legend_span + 3), int(np.ceil(np.sqrt(padded_area))) + 1)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        raise RuntimeError("object pack too large")

    height = random.randint(min_h, GRID_H_RANGE[1])
    width = random.randint(min_w, GRID_W_RANGE[1])

    placements = _try_place_objects(height, width, objects)
    if placements is None:
        raise RuntimeError("placement failed")

    inp = np.full((height, width), AZURE, dtype=int)
    for (mask, _holes, _count), (top, left) in zip(objects, placements, strict=False):
        _paste_object(inp, mask, top, left, GREY)

    legend_cols, _legend_span = _legend_layout(width, legend_len)
    for col, color in zip(legend_cols, legend_colors, strict=False):
        inp[-1, col] = int(color)

    out = _expected_output(inp, objects, placements, legend_colors)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            inp, expected = _build_sample()
        except RuntimeError:
            continue
        if _dominant_color(inp) != AZURE:
            continue
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
